//! Color quantization algorithms
//!
//! This module provides methods for reducing the number of distinct colors
//! in an image while preserving visual quality.

use crate::error::Result;
use image::{DynamicImage, Rgb, RgbImage};
use rand::prelude::*;
use rand::{rng, Rng};
use rayon::prelude::*;
use std::collections::HashMap;

/// K-means color quantization parameters
#[derive(Debug, Clone)]
pub struct KMeansParams {
    /// Number of colors in the palette
    pub k: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub epsilon: f32,
    /// Initialization method
    pub init_method: InitMethod,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            k: 16,
            max_iterations: 100,
            epsilon: 1.0,
            init_method: InitMethod::KMeansPlusPlus,
        }
    }
}

/// Initialization method for k-means
#[derive(Debug, Clone, Copy)]
pub enum InitMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization
    KMeansPlusPlus,
    /// Use most frequent colors
    Frequency,
}

/// Perform k-means color quantization
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - K-means parameters
///
/// # Returns
///
/// * Result containing quantized image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::color::{kmeans_quantize, KMeansParams};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let quantized = kmeans_quantize(&img, &KMeansParams::default())?;
/// # Ok(())
/// # }
/// ```
pub fn kmeans_quantize(img: &DynamicImage, params: &KMeansParams) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Extract color samples
    let mut colors = Vec::new();
    for pixel in rgb.pixels() {
        colors.push([pixel[0] as f32, pixel[1] as f32, pixel[2] as f32]);
    }

    // Initialize cluster centers
    let mut centers = initialize_centers(&colors, params);

    // K-means iterations
    for _iter in 0..params.max_iterations {
        // Assign pixels to clusters
        let assignments: Vec<usize> = colors
            .par_iter()
            .map(|color| {
                let mut min_dist = f32::INFINITY;
                let mut best_cluster = 0;

                for (i, center) in centers.iter().enumerate() {
                    let dist = color_distance(color, center);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = i;
                    }
                }

                best_cluster
            })
            .collect();

        // Update centers
        let new_centers = update_centers(&colors, &assignments, params.k);

        // Check convergence
        let mut max_change = 0.0f32;
        for (old, new) in centers.iter().zip(new_centers.iter()) {
            let change = color_distance(old, new);
            if change > max_change {
                max_change = change;
            }
        }

        centers = new_centers;

        if max_change < params.epsilon {
            break;
        }
    }

    // Create quantized image
    let mut result = RgbImage::new(width, height);
    let mut idx = 0;

    for y in 0..height {
        for x in 0..width {
            let color = &colors[idx];
            let mut min_dist = f32::INFINITY;
            let mut best_center = &centers[0];

            for center in &centers {
                let dist = color_distance(color, center);
                if dist < min_dist {
                    min_dist = dist;
                    best_center = center;
                }
            }

            result.put_pixel(
                x,
                y,
                Rgb([
                    best_center[0] as u8,
                    best_center[1] as u8,
                    best_center[2] as u8,
                ]),
            );

            idx += 1;
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Initialize cluster centers
fn initialize_centers(colors: &[[f32; 3]], params: &KMeansParams) -> Vec<[f32; 3]> {
    match params.init_method {
        InitMethod::Random => initialize_random(colors, params.k),
        InitMethod::KMeansPlusPlus => initialize_kmeans_plus_plus(colors, params.k),
        InitMethod::Frequency => initialize_frequency(colors, params.k),
    }
}

/// Random initialization
fn initialize_random(colors: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    let mut rng = rng();
    let mut centers = Vec::new();
    let mut indices: Vec<_> = (0..colors.len()).collect();
    indices.shuffle(&mut rng);

    for i in 0..k.min(colors.len()) {
        centers.push(colors[indices[i]]);
    }

    centers
}

/// K-means++ initialization
fn initialize_kmeans_plus_plus(colors: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    let mut rng = rng();
    let mut centers = Vec::new();

    // Choose first center randomly
    centers.push(colors[rng.random_range(0..colors.len())]);

    // Choose remaining centers
    for _ in 1..k {
        let mut distances = vec![0.0f32; colors.len()];
        let mut sum = 0.0f32;

        // Compute distances to nearest center
        for (i, color) in colors.iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            for center in &centers {
                let dist = color_distance(color, center);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            distances[i] = min_dist * min_dist; // Square for probability
            sum += distances[i];
        }

        // Choose next center with probability proportional to squared distance
        let mut threshold = rng.random::<f32>() * sum;
        let mut chosen = 0;

        for (i, &dist) in distances.iter().enumerate() {
            threshold -= dist;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }

        centers.push(colors[chosen]);
    }

    centers
}

/// Frequency-based initialization
fn initialize_frequency(colors: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    // Count color frequencies
    let mut color_counts = HashMap::new();

    for color in colors {
        let key = (color[0] as u8, color[1] as u8, color[2] as u8);
        *color_counts.entry(key).or_insert(0) += 1;
    }

    // Sort by frequency
    let mut sorted: Vec<_> = color_counts.into_iter().collect();
    sorted.sort_by_key(|(_, count)| -count);

    // Take top k colors
    let mut centers = Vec::new();
    for ((r, g, b), _) in sorted.iter().take(k.min(sorted.len())) {
        centers.push([*r as f32, *g as f32, *b as f32]);
    }

    // Fill remaining with random if needed
    let mut rng = rng();
    while centers.len() < k {
        centers.push(colors[rng.random_range(0..colors.len())]);
    }

    centers
}

/// Update cluster centers
fn update_centers(colors: &[[f32; 3]], assignments: &[usize], k: usize) -> Vec<[f32; 3]> {
    let mut new_centers = vec![[0.0, 0.0, 0.0]; k];
    let mut counts = vec![0; k];

    // Accumulate colors
    for (color, &cluster) in colors.iter().zip(assignments.iter()) {
        new_centers[cluster][0] += color[0];
        new_centers[cluster][1] += color[1];
        new_centers[cluster][2] += color[2];
        counts[cluster] += 1;
    }

    // Compute means
    for i in 0..k {
        if counts[i] > 0 {
            new_centers[i][0] /= counts[i] as f32;
            new_centers[i][1] /= counts[i] as f32;
            new_centers[i][2] /= counts[i] as f32;
        }
    }

    new_centers
}

/// Compute squared Euclidean distance between colors
fn color_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    dr * dr + dg * dg + db * db
}

/// Median cut color quantization
///
/// # Arguments
///
/// * `img` - Input image
/// * `n_colors` - Number of colors in palette
///
/// # Returns
///
/// * Result containing quantized image
pub fn median_cut_quantize(img: &DynamicImage, n_colors: usize) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Extract colors
    let mut colors = Vec::new();
    for pixel in rgb.pixels() {
        colors.push([pixel[0], pixel[1], pixel[2]]);
    }

    // Build initial box
    let mut boxes = vec![ColorBox::new(&colors)];

    // Recursively split boxes
    while boxes.len() < n_colors && boxes.iter().any(|b| b.can_split()) {
        // Find box with largest volume
        let mut max_volume = 0;
        let mut split_idx = 0;

        for (i, box_) in boxes.iter().enumerate() {
            if box_.can_split() && box_.volume() > max_volume {
                max_volume = box_.volume();
                split_idx = i;
            }
        }

        // Split the box
        let box_to_split = boxes.remove(split_idx);
        let (box1, box2) = box_to_split.split();
        boxes.push(box1);
        boxes.push(box2);
    }

    // Get palette colors (average of each box)
    let palette: Vec<[u8; 3]> = boxes.iter().map(|b| b.average()).collect();

    // Create quantized image
    let mut result = RgbImage::new(width, height);

    for (pixel, result_pixel) in rgb.pixels().zip(result.pixels_mut()) {
        let color = [pixel[0], pixel[1], pixel[2]];
        let mut min_dist = u32::MAX;
        let mut best_color = palette[0];

        for &palette_color in &palette {
            let dist = color_distance_u8(&color, &palette_color);
            if dist < min_dist {
                min_dist = dist;
                best_color = palette_color;
            }
        }

        *result_pixel = Rgb(best_color);
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Color box for median cut algorithm
struct ColorBox {
    colors: Vec<[u8; 3]>,
    min: [u8; 3],
    max: [u8; 3],
}

impl ColorBox {
    fn new(colors: &[[u8; 3]]) -> Self {
        let mut min = [255u8; 3];
        let mut max = [0u8; 3];

        for color in colors {
            for i in 0..3 {
                min[i] = min[i].min(color[i]);
                max[i] = max[i].max(color[i]);
            }
        }

        Self {
            colors: colors.to_vec(),
            min,
            max,
        }
    }

    fn can_split(&self) -> bool {
        self.colors.len() > 1
    }

    fn volume(&self) -> u32 {
        let r = (self.max[0] - self.min[0]) as u32;
        let g = (self.max[1] - self.min[1]) as u32;
        let b = (self.max[2] - self.min[2]) as u32;
        r * g * b
    }

    fn split(mut self) -> (ColorBox, ColorBox) {
        // Find longest axis
        let r_range = self.max[0] - self.min[0];
        let g_range = self.max[1] - self.min[1];
        let b_range = self.max[2] - self.min[2];

        let axis = if r_range >= g_range && r_range >= b_range {
            0
        } else if g_range >= b_range {
            1
        } else {
            2
        };

        // Sort by the longest axis
        self.colors.sort_by_key(|c| c[axis]);

        // Split at median
        let mid = self.colors.len() / 2;
        let colors2 = self.colors.split_off(mid);

        (ColorBox::new(&self.colors), ColorBox::new(&colors2))
    }

    fn average(&self) -> [u8; 3] {
        let mut sum = [0u32; 3];

        for color in &self.colors {
            sum[0] += color[0] as u32;
            sum[1] += color[1] as u32;
            sum[2] += color[2] as u32;
        }

        let n = self.colors.len() as u32;
        [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8]
    }
}

/// Compute squared distance between u8 colors
fn color_distance_u8(a: &[u8; 3], b: &[u8; 3]) -> u32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    (dr * dr + dg * dg + db * db) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_quantize() {
        let img = DynamicImage::new_rgb8(20, 20);
        let params = KMeansParams {
            k: 4,
            max_iterations: 10,
            ..Default::default()
        };

        let result = kmeans_quantize(&img, &params);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.width(), 20);
        assert_eq!(quantized.height(), 20);
    }

    #[test]
    fn test_median_cut() {
        let img = DynamicImage::new_rgb8(20, 20);
        let result = median_cut_quantize(&img, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_color_box() {
        let colors = vec![[0, 0, 0], [255, 255, 255], [128, 128, 128]];

        let box_ = ColorBox::new(&colors);
        assert_eq!(box_.min, [0, 0, 0]);
        assert_eq!(box_.max, [255, 255, 255]);
        assert!(box_.can_split());
    }
}
