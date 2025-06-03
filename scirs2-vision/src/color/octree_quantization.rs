//! Octree-based color quantization
//!
//! This module implements octree color quantization, an efficient algorithm
//! for reducing the number of colors in an image while maintaining quality.

use crate::error::Result;
use image::{DynamicImage, Rgb, RgbImage};
use std::collections::VecDeque;

/// Octree node for color quantization
#[derive(Debug)]
struct OctreeNode {
    /// Sum of red values
    red_sum: u64,
    /// Sum of green values
    green_sum: u64,
    /// Sum of blue values
    blue_sum: u64,
    /// Number of pixels represented
    pixel_count: u64,
    /// Child nodes (8 for octree)
    children: [Option<Box<OctreeNode>>; 8],
    /// Whether this is a leaf node
    is_leaf: bool,
    /// Node level in the tree (0-7)
    level: u8,
}

impl OctreeNode {
    /// Create a new octree node
    fn new(level: u8) -> Self {
        Self {
            red_sum: 0,
            green_sum: 0,
            blue_sum: 0,
            pixel_count: 0,
            children: Default::default(),
            is_leaf: level == 7,
            level,
        }
    }

    /// Add a color to this node
    fn add_color(&mut self, r: u8, g: u8, b: u8, level: u8) {
        if self.is_leaf {
            self.red_sum += r as u64;
            self.green_sum += g as u64;
            self.blue_sum += b as u64;
            self.pixel_count += 1;
        } else {
            // Determine which child to add to
            let index = Self::get_child_index(r, g, b, level);

            if self.children[index].is_none() {
                self.children[index] = Some(Box::new(OctreeNode::new(level + 1)));
            }

            self.children[index]
                .as_mut()
                .unwrap()
                .add_color(r, g, b, level + 1);

            // Update this node's sums
            self.red_sum += r as u64;
            self.green_sum += g as u64;
            self.blue_sum += b as u64;
            self.pixel_count += 1;
        }
    }

    /// Get the child index for a color at a given level
    fn get_child_index(r: u8, g: u8, b: u8, level: u8) -> usize {
        let shift = 7 - level;
        let r_bit = ((r >> shift) & 1) as usize;
        let g_bit = ((g >> shift) & 1) as usize;
        let b_bit = ((b >> shift) & 1) as usize;

        (r_bit << 2) | (g_bit << 1) | b_bit
    }

    /// Count leaf nodes
    fn count_leaves(&self) -> usize {
        if self.is_leaf {
            1
        } else {
            self.children
                .iter()
                .filter_map(|child| child.as_ref())
                .map(|child| child.count_leaves())
                .sum()
        }
    }

    /// Get the average color for this node
    fn get_color(&self) -> [u8; 3] {
        if self.pixel_count > 0 {
            [
                (self.red_sum / self.pixel_count) as u8,
                (self.green_sum / self.pixel_count) as u8,
                (self.blue_sum / self.pixel_count) as u8,
            ]
        } else {
            [0, 0, 0]
        }
    }

    /// Find the nearest color in the tree
    fn find_nearest_color(&self, r: u8, g: u8, b: u8, level: u8) -> [u8; 3] {
        if self.is_leaf || level == 7 {
            self.get_color()
        } else {
            let index = Self::get_child_index(r, g, b, level);

            if let Some(child) = &self.children[index] {
                child.find_nearest_color(r, g, b, level + 1)
            } else {
                // No exact child, find the nearest one
                let mut best_color = self.get_color();
                let mut best_distance = u32::MAX;

                for child in self.children.iter().flatten() {
                    let color = child.get_color();
                    let distance = color_distance_squared(r, g, b, color[0], color[1], color[2]);

                    if distance < best_distance {
                        best_distance = distance;
                        best_color = color;
                    }
                }

                best_color
            }
        }
    }

    /// Merge this node into a leaf (reduce colors)
    fn merge_to_leaf(&mut self) {
        if !self.is_leaf {
            self.is_leaf = true;
            self.children = Default::default();
        }
    }

    /// Get all nodes at a specific level
    fn get_nodes_at_level(&mut self, target_level: u8) -> Vec<*mut OctreeNode> {
        let mut nodes = Vec::new();

        if self.level == target_level && !self.is_leaf {
            nodes.push(self as *mut OctreeNode);
        }

        for child in self.children.iter_mut().flatten() {
            nodes.extend(child.get_nodes_at_level(target_level));
        }

        nodes
    }
}

/// Octree color quantizer
pub struct OctreeQuantizer {
    root: OctreeNode,
    max_colors: usize,
}

impl OctreeQuantizer {
    /// Create a new octree quantizer
    ///
    /// # Arguments
    ///
    /// * `max_colors` - Maximum number of colors in the palette
    pub fn new(max_colors: usize) -> Self {
        Self {
            root: OctreeNode::new(0),
            max_colors,
        }
    }

    /// Add a color to the octree
    pub fn add_color(&mut self, r: u8, g: u8, b: u8) {
        self.root.add_color(r, g, b, 0);
    }

    /// Reduce the number of colors by merging nodes
    pub fn reduce_colors(&mut self) {
        while self.root.count_leaves() > self.max_colors {
            // Find the deepest level with nodes
            for level in (0..7).rev() {
                let nodes = unsafe {
                    self.root
                        .get_nodes_at_level(level)
                        .into_iter()
                        .map(|ptr| &mut *ptr)
                        .collect::<Vec<_>>()
                };

                if !nodes.is_empty() {
                    // Merge the node with the least pixels
                    let min_node = nodes
                        .into_iter()
                        .min_by_key(|node| node.pixel_count)
                        .unwrap();

                    min_node.merge_to_leaf();
                    break;
                }
            }
        }
    }

    /// Find the nearest color in the palette
    pub fn find_nearest_color(&self, r: u8, g: u8, b: u8) -> [u8; 3] {
        self.root.find_nearest_color(r, g, b, 0)
    }

    /// Get all colors in the palette
    pub fn get_palette(&self) -> Vec<[u8; 3]> {
        let mut palette = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(&self.root);

        while let Some(node) = queue.pop_front() {
            if node.is_leaf && node.pixel_count > 0 {
                palette.push(node.get_color());
            } else {
                for child in node.children.iter().flatten() {
                    queue.push_back(child);
                }
            }
        }

        palette
    }
}

/// Calculate squared distance between two colors
fn color_distance_squared(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> u32 {
    let dr = r1 as i32 - r2 as i32;
    let dg = g1 as i32 - g2 as i32;
    let db = b1 as i32 - b2 as i32;

    (dr * dr + dg * dg + db * db) as u32
}

/// Perform octree color quantization on an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `max_colors` - Maximum number of colors in the output
///
/// # Returns
///
/// * Result containing the quantized image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::color::octree_quantize;
/// use image::{DynamicImage, RgbImage, Rgb};
///
/// // Create a simple test image
/// let mut img = RgbImage::new(4, 4);
/// for x in 0..4 {
///     for y in 0..4 {
///         let r = (x * 64) as u8;
///         let g = (y * 64) as u8;
///         let b = ((x + y) * 32) as u8;
///         img.put_pixel(x, y, Rgb([r, g, b]));
///     }
/// }
///
/// let dynamic_img = DynamicImage::ImageRgb8(img);
/// let quantized = octree_quantize(&dynamic_img, 8).unwrap();
/// assert_eq!(quantized.width(), 4);
/// assert_eq!(quantized.height(), 4);
/// ```
pub fn octree_quantize(img: &DynamicImage, max_colors: usize) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Build the octree
    let mut quantizer = OctreeQuantizer::new(max_colors);

    for pixel in rgb.pixels() {
        quantizer.add_color(pixel[0], pixel[1], pixel[2]);
    }

    // Reduce colors
    quantizer.reduce_colors();

    // Create quantized image
    let mut result = RgbImage::new(width, height);

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let quantized_color = quantizer.find_nearest_color(pixel[0], pixel[1], pixel[2]);
        result.put_pixel(x, y, Rgb(quantized_color));
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Adaptive octree quantization with importance weighting
///
/// # Arguments
///
/// * `img` - Input image
/// * `max_colors` - Maximum number of colors
/// * `importance_map` - Optional importance map (same size as image)
///
/// # Returns
///
/// * Result containing the quantized image
pub fn adaptive_octree_quantize(
    img: &DynamicImage,
    max_colors: usize,
    importance_map: Option<&DynamicImage>,
) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Get importance weights
    let weights = if let Some(map) = importance_map {
        let gray_map = map.to_luma8();
        gray_map
            .pixels()
            .map(|p| p[0] as f32 / 255.0)
            .collect::<Vec<_>>()
    } else {
        vec![1.0; (width * height) as usize]
    };

    // Build weighted octree
    let mut quantizer = OctreeQuantizer::new(max_colors);

    for (idx, pixel) in rgb.pixels().enumerate() {
        let weight = (weights[idx] * 10.0).max(1.0) as usize;

        // Add color multiple times based on importance
        for _ in 0..weight {
            quantizer.add_color(pixel[0], pixel[1], pixel[2]);
        }
    }

    // Reduce colors
    quantizer.reduce_colors();

    // Create quantized image
    let mut result = RgbImage::new(width, height);

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let quantized_color = quantizer.find_nearest_color(pixel[0], pixel[1], pixel[2]);
        result.put_pixel(x, y, Rgb(quantized_color));
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Generate a color palette from an image using octree
///
/// # Arguments
///
/// * `img` - Input image
/// * `palette_size` - Number of colors in the palette
///
/// # Returns
///
/// * Vector of RGB colors
pub fn extract_palette(img: &DynamicImage, palette_size: usize) -> Vec<[u8; 3]> {
    let rgb = img.to_rgb8();

    let mut quantizer = OctreeQuantizer::new(palette_size);

    for pixel in rgb.pixels() {
        quantizer.add_color(pixel[0], pixel[1], pixel[2]);
    }

    quantizer.reduce_colors();
    quantizer.get_palette()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_node_basic() {
        let mut node = OctreeNode::new(0);
        node.add_color(255, 0, 0, 0);
        node.add_color(0, 255, 0, 0);

        assert!(node.count_leaves() > 0);
    }

    #[test]
    fn test_octree_quantize() {
        let img = DynamicImage::new_rgb8(10, 10);
        let result = octree_quantize(&img, 16);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.width(), 10);
        assert_eq!(quantized.height(), 10);
    }

    #[test]
    fn test_extract_palette() {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0]));
        img.put_pixel(1, 0, Rgb([0, 255, 0]));
        img.put_pixel(0, 1, Rgb([0, 0, 255]));
        img.put_pixel(1, 1, Rgb([255, 255, 255]));

        let palette = extract_palette(&DynamicImage::ImageRgb8(img), 4);
        assert!(palette.len() <= 4);
        assert!(!palette.is_empty());
    }

    #[test]
    fn test_child_index() {
        assert_eq!(OctreeNode::get_child_index(255, 255, 255, 0), 7);
        assert_eq!(OctreeNode::get_child_index(0, 0, 0, 0), 0);
        assert_eq!(OctreeNode::get_child_index(128, 128, 128, 0), 7);
    }
}
