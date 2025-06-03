//! Mean shift segmentation algorithm
//!
//! Mean shift is a non-parametric clustering technique that doesn't require
//! specifying the number of clusters beforehand.

use crate::error::Result;
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

/// Parameters for mean shift segmentation
#[derive(Debug, Clone)]
pub struct MeanShiftParams {
    /// Spatial bandwidth (radius in pixels)
    pub spatial_bandwidth: f32,
    /// Color bandwidth (radius in color space)
    pub color_bandwidth: f32,
    /// Minimum region size (pixels)
    pub min_region_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl Default for MeanShiftParams {
    fn default() -> Self {
        Self {
            spatial_bandwidth: 7.0,
            color_bandwidth: 10.0,
            min_region_size: 20,
            max_iterations: 20,
            convergence_threshold: 0.1,
        }
    }
}

/// Perform mean shift segmentation
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - Mean shift parameters
///
/// # Returns
///
/// * Result containing segmented label map
///
/// # Example
///
/// ```rust
/// use scirs2_vision::segmentation::{mean_shift, MeanShiftParams};
/// use image::{DynamicImage, RgbImage, Rgb};
///
/// // Create a small test image with distinct regions
/// let mut img = RgbImage::new(10, 10);
/// // Top half - red region
/// for y in 0..5 {
///     for x in 0..10 {
///         img.put_pixel(x, y, Rgb([255, 0, 0]));
///     }
/// }
/// // Bottom half - blue region
/// for y in 5..10 {
///     for x in 0..10 {
///         img.put_pixel(x, y, Rgb([0, 0, 255]));
///     }
/// }
///
/// let params = MeanShiftParams {
///     spatial_bandwidth: 5.0,
///     color_bandwidth: 10.0,
///     min_region_size: 10,
///     max_iterations: 5,
///     convergence_threshold: 0.5,
/// };
///
/// let labels = mean_shift(&DynamicImage::ImageRgb8(img), &params).unwrap();
/// assert_eq!(labels.dim(), (10, 10));
/// ```
pub fn mean_shift(img: &DynamicImage, params: &MeanShiftParams) -> Result<Array2<u32>> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Convert to feature space (x, y, L, a, b)
    let features = image_to_feature_space(&rgb);
    let n_pixels = (width * height) as usize;

    // Initialize modes for each pixel
    let mut modes = features.clone();

    // Perform mean shift iterations
    for _iter in 0..params.max_iterations {
        let old_modes = modes.clone();

        // Update each pixel's mode in parallel
        let updates: Vec<_> = (0..n_pixels)
            .into_par_iter()
            .map(|idx| {
                let current_mode = modes.slice(ndarray::s![idx, ..]).to_owned();
                compute_mean_shift(
                    &features,
                    &current_mode,
                    params.spatial_bandwidth,
                    params.color_bandwidth,
                )
            })
            .collect();

        // Update modes
        for (idx, new_mode) in updates.into_iter().enumerate() {
            modes.slice_mut(ndarray::s![idx, ..]).assign(&new_mode);
        }

        // Check convergence
        let mut max_shift = 0.0f32;
        for i in 0..n_pixels {
            let shift = euclidean_distance(
                &modes.slice(ndarray::s![i, ..]).to_owned(),
                &old_modes.slice(ndarray::s![i, ..]).to_owned(),
            );
            if shift > max_shift {
                max_shift = shift;
            }
        }

        if max_shift < params.convergence_threshold {
            break;
        }
    }

    // Cluster modes and assign labels
    let labels = cluster_modes(&modes, params.spatial_bandwidth, params.color_bandwidth);

    // Reshape to image dimensions
    let label_map = Array2::from_shape_vec((height as usize, width as usize), labels)?;

    // Merge small regions
    let final_labels = merge_small_regions(&label_map, params.min_region_size);

    Ok(final_labels)
}

/// Convert image to feature space (x, y, L, a, b)
fn image_to_feature_space(img: &RgbImage) -> Array2<f32> {
    let (width, height) = img.dimensions();
    let n_pixels = (width * height) as usize;
    let mut features = Array2::zeros((n_pixels, 5));

    let mut idx = 0;
    for y in 0..height {
        for x in 0..width {
            let rgb = img.get_pixel(x, y);
            let (l, a, b) = rgb_to_lab(rgb[0], rgb[1], rgb[2]);

            features[[idx, 0]] = x as f32;
            features[[idx, 1]] = y as f32;
            features[[idx, 2]] = l;
            features[[idx, 3]] = a;
            features[[idx, 4]] = b;

            idx += 1;
        }
    }

    features
}

/// Compute mean shift update for a point
fn compute_mean_shift(
    features: &Array2<f32>,
    point: &Array1<f32>,
    spatial_bandwidth: f32,
    color_bandwidth: f32,
) -> Array1<f32> {
    let n_points = features.nrows();
    let mut weighted_sum = Array1::zeros(5);
    let mut weight_sum = 0.0f32;

    for i in 0..n_points {
        let feature = features.slice(ndarray::s![i, ..]);

        // Compute spatial distance
        let spatial_dist =
            ((feature[0] - point[0]).powi(2) + (feature[1] - point[1]).powi(2)).sqrt();

        // Compute color distance
        let color_dist = ((feature[2] - point[2]).powi(2)
            + (feature[3] - point[3]).powi(2)
            + (feature[4] - point[4]).powi(2))
        .sqrt();

        // Check if within bandwidth
        if spatial_dist <= spatial_bandwidth && color_dist <= color_bandwidth {
            // Gaussian kernel weights
            let spatial_weight = (-0.5 * (spatial_dist / spatial_bandwidth).powi(2)).exp();
            let color_weight = (-0.5 * (color_dist / color_bandwidth).powi(2)).exp();
            let weight = spatial_weight * color_weight;

            weighted_sum += &(feature.to_owned() * weight);
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        point.clone()
    }
}

/// Cluster modes and assign labels
fn cluster_modes(modes: &Array2<f32>, spatial_bandwidth: f32, color_bandwidth: f32) -> Vec<u32> {
    let n_points = modes.nrows();
    let mut labels = vec![u32::MAX; n_points];
    let mut current_label = 0u32;

    for i in 0..n_points {
        if labels[i] != u32::MAX {
            continue;
        }

        // Start new cluster
        labels[i] = current_label;

        // Find all points belonging to this cluster
        for (j, label) in labels.iter_mut().enumerate().take(n_points).skip(i + 1) {
            if *label != u32::MAX {
                continue;
            }

            let mode_i = modes.slice(ndarray::s![i, ..]);
            let mode_j = modes.slice(ndarray::s![j, ..]);

            // Check if modes are close enough
            let spatial_dist =
                ((mode_i[0] - mode_j[0]).powi(2) + (mode_i[1] - mode_j[1]).powi(2)).sqrt();
            let color_dist = ((mode_i[2] - mode_j[2]).powi(2)
                + (mode_i[3] - mode_j[3]).powi(2)
                + (mode_i[4] - mode_j[4]).powi(2))
            .sqrt();

            if spatial_dist <= spatial_bandwidth && color_dist <= color_bandwidth {
                *label = current_label;
            }
        }

        current_label += 1;
    }

    labels
}

/// Euclidean distance between two points
fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let diff = a - b;
    diff.mapv(|x| x * x).sum().sqrt()
}

/// Merge small regions with neighbors
fn merge_small_regions(labels: &Array2<u32>, min_size: usize) -> Array2<u32> {
    let (height, width) = labels.dim();
    let mut result = labels.clone();

    // Count region sizes
    let mut region_sizes = HashMap::new();
    for &label in labels.iter() {
        *region_sizes.entry(label).or_insert(0) += 1;
    }

    // Find small regions
    let small_regions: Vec<_> = region_sizes
        .iter()
        .filter(|(_, &size)| size < min_size)
        .map(|(&label, _)| label)
        .collect();

    // Merge each small region with its largest neighbor
    for small_label in small_regions {
        let mut neighbor_counts = HashMap::new();

        // Count neighboring labels
        for y in 0..height {
            for x in 0..width {
                if result[[y, x]] == small_label {
                    // Check 4-neighbors
                    for &(dy, dx) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;

                        if ny < height && nx < width {
                            let neighbor_label = result[[ny, nx]];
                            if neighbor_label != small_label {
                                *neighbor_counts.entry(neighbor_label).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Find most common neighbor
        if let Some((&new_label, _)) = neighbor_counts.iter().max_by_key(|(_, &count)| count) {
            // Replace small region with neighbor label
            for y in 0..height {
                for x in 0..width {
                    if result[[y, x]] == small_label {
                        result[[y, x]] = new_label;
                    }
                }
            }
        }
    }

    result
}

/// Convert RGB to Lab color space (simplified)
fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    // Normalize RGB
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;

    // Convert to XYZ (sRGB -> XYZ)
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.119_192 * g + 0.9503041 * b;

    // Normalize by D65 illuminant
    let x = x / 0.95047;
    let y = y / 1.00000;
    let z = z / 1.08883;

    // Convert to Lab
    let fx = if x > 0.008856 {
        x.powf(1.0 / 3.0)
    } else {
        7.787 * x + 16.0 / 116.0
    };
    let fy = if y > 0.008856 {
        y.powf(1.0 / 3.0)
    } else {
        7.787 * y + 16.0 / 116.0
    };
    let fz = if z > 0.008856 {
        z.powf(1.0 / 3.0)
    } else {
        7.787 * z + 16.0 / 116.0
    };

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    (l, a, b)
}

/// Convert mean shift segmentation labels to color image
pub fn mean_shift_labels_to_color(labels: &Array2<u32>) -> RgbImage {
    let (height, width) = labels.dim();
    let mut result = RgbImage::new(width as u32, height as u32);

    // Find unique labels
    let mut unique_labels = std::collections::HashSet::new();
    for &label in labels.iter() {
        unique_labels.insert(label);
    }

    // Generate colors for each label
    let mut label_colors = HashMap::new();
    let golden_ratio = 0.618_034;
    let mut hue = 0.0;

    for &label in &unique_labels {
        hue += golden_ratio;
        hue %= 1.0;

        // Convert HSV to RGB
        let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
        label_colors.insert(
            label,
            [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8],
        );
    }

    // Apply colors
    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]];
            let color = label_colors.get(&label).copied().unwrap_or([0, 0, 0]);
            result.put_pixel(x as u32, y as u32, Rgb(color));
        }
    }

    result
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r + m, g + m, b + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_shift_basic() {
        let img = DynamicImage::new_rgb8(20, 20);
        let params = MeanShiftParams {
            spatial_bandwidth: 5.0,
            color_bandwidth: 10.0,
            min_region_size: 5,
            max_iterations: 5,
            convergence_threshold: 0.5,
        };

        let result = mean_shift(&img, &params);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.dim(), (20, 20));
    }

    #[test]
    fn test_rgb_to_lab() {
        // Test white
        let (l, a, b) = rgb_to_lab(255, 255, 255);
        assert!(l > 95.0 && l < 105.0);
        assert!(a.abs() < 5.0);
        assert!(b.abs() < 5.0);

        // Test black
        let (l, _a, _b) = rgb_to_lab(0, 0, 0);
        assert!(l < 5.0);
    }

    #[test]
    fn test_feature_space_conversion() {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0]));
        img.put_pixel(1, 0, Rgb([0, 255, 0]));
        img.put_pixel(0, 1, Rgb([0, 0, 255]));
        img.put_pixel(1, 1, Rgb([255, 255, 255]));

        let features = image_to_feature_space(&img);
        assert_eq!(features.dim(), (4, 5));

        // Check spatial coordinates
        assert_eq!(features[[0, 0]], 0.0); // x
        assert_eq!(features[[0, 1]], 0.0); // y
        assert_eq!(features[[3, 0]], 1.0); // x
        assert_eq!(features[[3, 1]], 1.0); // y
    }
}
