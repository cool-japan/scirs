//! Watershed segmentation algorithm
//!
//! The watershed algorithm treats an image as a topographic surface where
//! bright pixels are high and dark pixels are low. It finds the lines that
//! run along the tops of ridges, effectively segmenting the image into regions.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage, RgbImage};
use ndarray::Array2;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Pixel with priority for watershed algorithm
#[derive(Clone, Copy, Debug)]
struct PixelPriority {
    y: usize,
    x: usize,
    priority: u8,
}

impl Eq for PixelPriority {}

impl PartialEq for PixelPriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Ord for PixelPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for PixelPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Apply watershed segmentation to an image
///
/// # Arguments
///
/// * `img` - Input image (will be converted to grayscale)
/// * `markers` - Optional marker image indicating seed regions (0 = no marker)
/// * `connectivity` - Connectivity type (4 or 8)
///
/// # Returns
///
/// * Result containing labeled regions image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::segmentation::watershed;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let segmented = watershed(&img, None, 8)?;
/// # Ok(())
/// # }
/// ```
pub fn watershed(
    img: &DynamicImage,
    markers: Option<&Array2<u32>>,
    connectivity: u8,
) -> Result<Array2<u32>> {
    if connectivity != 4 && connectivity != 8 {
        return Err(VisionError::InvalidParameter(
            "Connectivity must be 4 or 8".to_string(),
        ));
    }

    // Convert to grayscale
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Initialize labels
    let mut labels = match markers {
        Some(m) => {
            if m.dim() != (height as usize, width as usize) {
                return Err(VisionError::InvalidParameter(
                    "Marker dimensions must match image dimensions".to_string(),
                ));
            }
            m.clone()
        }
        None => Array2::zeros((height as usize, width as usize)),
    };

    // Find marker positions
    let mut queue = BinaryHeap::new();
    let mut in_queue = Array2::from_elem((height as usize, width as usize), false);

    // Initialize queue with marker boundaries
    for y in 0..height as usize {
        for x in 0..width as usize {
            if labels[[y, x]] > 0 {
                // Check neighbors
                for (dy, dx) in get_neighbors(connectivity) {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;

                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        let ny = ny as usize;
                        let nx = nx as usize;

                        if labels[[ny, nx]] == 0 && !in_queue[[ny, nx]] {
                            queue.push(PixelPriority {
                                y: ny,
                                x: nx,
                                priority: gray.get_pixel(nx as u32, ny as u32)[0],
                            });
                            in_queue[[ny, nx]] = true;
                        }
                    }
                }
            }
        }
    }

    // If no markers provided, use local minima
    if queue.is_empty() {
        let local_minima = find_local_minima(&gray, connectivity);
        let mut next_label = 1;

        for (y, x) in local_minima {
            labels[[y, x]] = next_label;
            next_label += 1;

            // Add neighbors to queue
            for (dy, dx) in get_neighbors(connectivity) {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;

                if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                    let ny = ny as usize;
                    let nx = nx as usize;

                    if labels[[ny, nx]] == 0 && !in_queue[[ny, nx]] {
                        queue.push(PixelPriority {
                            y: ny,
                            x: nx,
                            priority: gray.get_pixel(nx as u32, ny as u32)[0],
                        });
                        in_queue[[ny, nx]] = true;
                    }
                }
            }
        }
    }

    // Watershed flooding
    const WATERSHED_LINE: u32 = 0;

    while let Some(pixel) = queue.pop() {
        let mut neighbor_labels = Vec::new();

        // Check labeled neighbors
        for (dy, dx) in get_neighbors(connectivity) {
            let ny = pixel.y as i32 + dy;
            let nx = pixel.x as i32 + dx;

            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                let ny = ny as usize;
                let nx = nx as usize;

                let label = labels[[ny, nx]];
                if label > 0 && !neighbor_labels.contains(&label) {
                    neighbor_labels.push(label);
                }
            }
        }

        // Assign label
        use std::cmp::Ordering;
        match neighbor_labels.len().cmp(&1) {
            Ordering::Equal => {
                // Single neighbor label - propagate it
                labels[[pixel.y, pixel.x]] = neighbor_labels[0];

                // Add unlabeled neighbors to queue
                for (dy, dx) in get_neighbors(connectivity) {
                    let ny = pixel.y as i32 + dy;
                    let nx = pixel.x as i32 + dx;

                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        let ny = ny as usize;
                        let nx = nx as usize;

                        if labels[[ny, nx]] == 0 && !in_queue[[ny, nx]] {
                            queue.push(PixelPriority {
                                y: ny,
                                x: nx,
                                priority: gray.get_pixel(nx as u32, ny as u32)[0],
                            });
                            in_queue[[ny, nx]] = true;
                        }
                    }
                }
            }
            Ordering::Greater => {
                // Multiple labels - this is a watershed line
                labels[[pixel.y, pixel.x]] = WATERSHED_LINE;
            }
            Ordering::Less => {
                // No neighbors - should not happen in normal flow
            }
        }
    }

    Ok(labels)
}

/// Apply marker-controlled watershed segmentation
///
/// This version uses distance transform and markers for better control.
///
/// # Arguments
///
/// * `img` - Input image
/// * `markers` - Marker image with seed regions
/// * `mask` - Optional mask to limit segmentation area
///
/// # Returns
///
/// * Result containing labeled regions
pub fn watershed_markers(
    img: &DynamicImage,
    markers: &Array2<u32>,
    mask: Option<&Array2<bool>>,
) -> Result<Array2<u32>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    if markers.dim() != (height as usize, width as usize) {
        return Err(VisionError::InvalidParameter(
            "Marker dimensions must match image dimensions".to_string(),
        ));
    }

    if let Some(m) = mask {
        if m.dim() != (height as usize, width as usize) {
            return Err(VisionError::InvalidParameter(
                "Mask dimensions must match image dimensions".to_string(),
            ));
        }
    }

    // Apply watershed with markers
    let mut result = watershed(img, Some(markers), 8)?;

    // Apply mask if provided
    if let Some(m) = mask {
        for y in 0..height as usize {
            for x in 0..width as usize {
                if !m[[y, x]] {
                    result[[y, x]] = 0;
                }
            }
        }
    }

    Ok(result)
}

/// Find local minima in an image
fn find_local_minima(img: &GrayImage, connectivity: u8) -> Vec<(usize, usize)> {
    let (width, height) = img.dimensions();
    let mut minima = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let center_val = img.get_pixel(x, y)[0];
            let mut is_minimum = true;

            for (dy, dx) in get_neighbors(connectivity) {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;

                if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                    let neighbor_val = img.get_pixel(nx as u32, ny as u32)[0];
                    if neighbor_val < center_val {
                        is_minimum = false;
                        break;
                    }
                }
            }

            if is_minimum {
                minima.push((y as usize, x as usize));
            }
        }
    }

    minima
}

/// Get neighbor offsets based on connectivity
fn get_neighbors(connectivity: u8) -> &'static [(i32, i32)] {
    match connectivity {
        4 => &[(-1, 0), (0, -1), (0, 1), (1, 0)],
        8 => &[
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
        _ => &[],
    }
}

/// Convert labeled regions to a color image for visualization
///
/// # Arguments
///
/// * `labels` - Labeled regions from watershed
/// * `colormap` - Optional custom colormap (label -> RGB color)
///
/// # Returns
///
/// * Color image with each region in a different color
pub fn labels_to_color_image(
    labels: &Array2<u32>,
    colormap: Option<&HashMap<u32, [u8; 3]>>,
) -> RgbImage {
    let (height, width) = labels.dim();
    let mut img = RgbImage::new(width as u32, height as u32);

    // Generate default colormap if not provided
    let default_colormap: HashMap<u32, [u8; 3]>;
    let cmap = match colormap {
        Some(c) => c,
        None => {
            default_colormap = generate_colormap(labels);
            &default_colormap
        }
    };

    // Apply colors
    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]];
            let color = cmap.get(&label).unwrap_or(&[0, 0, 0]);
            img.put_pixel(x as u32, y as u32, image::Rgb(*color));
        }
    }

    img
}

/// Generate a colormap for labels
fn generate_colormap(labels: &Array2<u32>) -> HashMap<u32, [u8; 3]> {
    use std::collections::HashSet;

    let unique_labels: HashSet<u32> = labels.iter().cloned().collect();
    let mut colormap = HashMap::new();

    // Watershed lines are black
    colormap.insert(0, [0, 0, 0]);

    // Generate colors using golden ratio for better distribution
    let golden_ratio = 0.618_034;
    let mut hue = 0.0;

    for &label in unique_labels.iter() {
        if label > 0 {
            hue += golden_ratio;
            hue %= 1.0;

            // Convert HSV to RGB
            let (r, g, b) = hsv_to_rgb(hue * 360.0, 0.8, 0.9);
            colormap.insert(label, [r, g, b]);
        }
    }

    colormap
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Compute gradient magnitude for watershed
///
/// Higher gradients indicate stronger boundaries between regions.
pub fn compute_gradient_magnitude(img: &DynamicImage) -> Result<Array2<f32>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let mut gradient = Array2::zeros((height as usize, width as usize));

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let _center = gray.get_pixel(x, y)[0] as f32;

            // Sobel gradients
            let gx = gray.get_pixel(x + 1, y)[0] as f32 - gray.get_pixel(x - 1, y)[0] as f32;
            let gy = gray.get_pixel(x, y + 1)[0] as f32 - gray.get_pixel(x, y - 1)[0] as f32;

            gradient[[y as usize, x as usize]] = (gx * gx + gy * gy).sqrt();
        }
    }

    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use ndarray::{s, Array2};

    #[test]
    fn test_watershed_basic() {
        let img = DynamicImage::new_luma8(10, 10);
        let result = watershed(&img, None, 8);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.dim(), (10, 10));
    }

    #[test]
    fn test_watershed_with_markers() {
        let img = DynamicImage::new_luma8(10, 10);
        let mut markers = Array2::zeros((10, 10));

        // Set some markers
        markers[[2, 2]] = 1;
        markers[[7, 7]] = 2;

        let result = watershed(&img, Some(&markers), 8);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels[[2, 2]], 1);
        assert_eq!(labels[[7, 7]], 2);
    }

    #[test]
    fn test_invalid_connectivity() {
        let img = DynamicImage::new_luma8(10, 10);
        let result = watershed(&img, None, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_labels_to_color() {
        let mut labels = Array2::zeros((10, 10));
        labels.slice_mut(s![0..5, ..]).fill(1);
        labels.slice_mut(s![5.., ..]).fill(2);

        let color_img = labels_to_color_image(&labels, None);
        assert_eq!(color_img.dimensions(), (10, 10));
    }

    #[test]
    fn test_gradient_magnitude() {
        // Create image with edge
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                img.put_pixel(x, y, image::Luma([if x < 5 { 0 } else { 255 }]));
            }
        }
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let gradient = compute_gradient_magnitude(&dynamic_img).unwrap();

        // Should have high gradient at x=5
        assert!(gradient[[5, 4]] > 0.0 || gradient[[5, 5]] > 0.0);
    }

    #[test]
    fn test_watershed_markers_with_mask() {
        let img = DynamicImage::new_luma8(10, 10);
        let mut markers = Array2::zeros((10, 10));
        markers[[5, 5]] = 1;

        let mut mask = Array2::from_elem((10, 10), true);
        mask.slice_mut(s![0..3, ..]).fill(false);

        let result = watershed_markers(&img, &markers, Some(&mask)).unwrap();

        // Masked area should be 0
        for y in 0..3 {
            for x in 0..10 {
                assert_eq!(result[[y, x]], 0);
            }
        }
    }
}
