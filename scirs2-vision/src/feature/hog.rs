//! HOG (Histogram of Oriented Gradients) descriptor
//!
//! This module implements the HOG descriptor for object detection and recognition.

use crate::error::Result;
use crate::feature::image_to_array;
use image::DynamicImage;
use ndarray::{s, Array1, Array2, Array3};
use std::f32::consts::PI;

/// Configuration for HOG descriptor
#[derive(Debug, Clone)]
pub struct HogConfig {
    /// Cell size in pixels
    pub cell_size: usize,
    /// Block size in cells
    pub block_size: usize,
    /// Block stride in cells
    pub block_stride: usize,
    /// Number of orientation bins
    pub num_bins: usize,
    /// Use unsigned gradients (0-180°) or signed gradients (0-360°)
    pub unsigned_gradients: bool,
    /// Apply Gaussian weighting to cells within blocks
    pub gaussian_weighting: bool,
    /// Normalization method
    pub normalization: HogNormalization,
}

/// HOG normalization methods
#[derive(Debug, Clone, Copy)]
pub enum HogNormalization {
    /// L2 normalization
    L2,
    /// L2-Hys normalization (L2 norm followed by clipping and renormalizing)
    L2Hys,
    /// L1 normalization
    L1,
    /// L1-sqrt normalization
    L1Sqrt,
}

impl Default for HogConfig {
    fn default() -> Self {
        Self {
            cell_size: 8,
            block_size: 2,
            block_stride: 1,
            num_bins: 9,
            unsigned_gradients: true,
            gaussian_weighting: true,
            normalization: HogNormalization::L2Hys,
        }
    }
}

/// HOG descriptor
#[derive(Debug, Clone)]
pub struct HogDescriptor {
    /// Feature vector
    pub features: Vec<f32>,
    /// Number of cells in x direction
    pub cells_x: usize,
    /// Number of cells in y direction
    pub cells_y: usize,
    /// Number of blocks in x direction
    pub blocks_x: usize,
    /// Number of blocks in y direction
    pub blocks_y: usize,
}

/// Compute HOG features for an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `config` - HOG configuration
///
/// # Returns
///
/// * Result containing HOG descriptor
pub fn compute_hog(img: &DynamicImage, config: &HogConfig) -> Result<HogDescriptor> {
    // Convert to grayscale
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Compute gradients
    let (magnitudes, orientations) = compute_gradients(&array)?;

    // Compute cell histograms
    let cells_x = width / config.cell_size;
    let cells_y = height / config.cell_size;

    let mut cell_histograms = Array3::zeros((cells_y, cells_x, config.num_bins));

    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let hist = compute_cell_histogram(
                &magnitudes,
                &orientations,
                cy,
                cx,
                config.cell_size,
                config.num_bins,
                config.unsigned_gradients,
            )?;

            for (bin, &value) in hist.iter().enumerate() {
                cell_histograms[[cy, cx, bin]] = value;
            }
        }
    }

    // Compute block features
    let blocks_x = (cells_x - config.block_size) / config.block_stride + 1;
    let blocks_y = (cells_y - config.block_size) / config.block_stride + 1;

    let mut features = Vec::new();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_features = compute_block_features(
                &cell_histograms,
                by * config.block_stride,
                bx * config.block_stride,
                config.block_size,
                config.gaussian_weighting,
                config.normalization,
            )?;

            features.extend_from_slice(&block_features);
        }
    }

    Ok(HogDescriptor {
        features,
        cells_x,
        cells_y,
        blocks_x,
        blocks_y,
    })
}

/// Compute image gradients
fn compute_gradients(image: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>)> {
    let (height, width) = image.dim();
    let mut magnitudes = Array2::zeros((height, width));
    let mut orientations = Array2::zeros((height, width));

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Compute gradients using centered differences
            let dx = image[[y, x + 1]] - image[[y, x - 1]];
            let dy = image[[y + 1, x]] - image[[y - 1, x]];

            magnitudes[[y, x]] = (dx * dx + dy * dy).sqrt();
            orientations[[y, x]] = dy.atan2(dx);
        }
    }

    Ok((magnitudes, orientations))
}

/// Compute histogram for a single cell
fn compute_cell_histogram(
    magnitudes: &Array2<f32>,
    orientations: &Array2<f32>,
    cell_y: usize,
    cell_x: usize,
    cell_size: usize,
    num_bins: usize,
    unsigned_gradients: bool,
) -> Result<Array1<f32>> {
    let mut histogram = Array1::zeros(num_bins);

    let y_start = cell_y * cell_size;
    let x_start = cell_x * cell_size;
    let y_end = y_start + cell_size;
    let x_end = x_start + cell_size;

    let angle_range = if unsigned_gradients { PI } else { 2.0 * PI };
    let bin_width = angle_range / num_bins as f32;

    for y in y_start..y_end {
        for x in x_start..x_end {
            let magnitude = magnitudes[[y, x]];
            let mut orientation = orientations[[y, x]];

            // Convert to unsigned gradient if needed
            if unsigned_gradients && orientation < 0.0 {
                orientation += PI;
            }

            // Ensure orientation is in valid range
            while orientation < 0.0 {
                orientation += angle_range;
            }
            while orientation >= angle_range {
                orientation -= angle_range;
            }

            // Compute bin index
            let bin_idx = ((orientation / bin_width).floor() as usize) % num_bins;
            let bin_center = (bin_idx as f32 + 0.5) * bin_width;

            // Bilinear interpolation between adjacent bins
            let next_bin = (bin_idx + 1) % num_bins;
            let angle_diff = (orientation - bin_center).abs();

            let weight_current = 1.0 - (angle_diff / bin_width);
            let weight_next = angle_diff / bin_width;

            histogram[bin_idx] += magnitude * weight_current;
            histogram[next_bin] += magnitude * weight_next;
        }
    }

    Ok(histogram)
}

/// Compute features for a single block
fn compute_block_features(
    cell_histograms: &Array3<f32>,
    block_y: usize,
    block_x: usize,
    block_size: usize,
    gaussian_weighting: bool,
    normalization: HogNormalization,
) -> Result<Vec<f32>> {
    let mut block_features = Vec::new();

    // Gaussian weights for cells within block
    let gaussian_weights = if gaussian_weighting {
        compute_gaussian_weights(block_size)
    } else {
        vec![1.0; block_size * block_size]
    };

    // Concatenate cell histograms within block
    for dy in 0..block_size {
        for dx in 0..block_size {
            let cell_y = block_y + dy;
            let cell_x = block_x + dx;

            let weight_idx = dy * block_size + dx;
            let weight = gaussian_weights[weight_idx];

            let cell_hist = cell_histograms.slice(s![cell_y, cell_x, ..]);

            for &bin_value in cell_hist.iter() {
                block_features.push(bin_value * weight);
            }
        }
    }

    // Normalize block features
    normalize_block_features(&mut block_features, normalization);

    Ok(block_features)
}

/// Compute Gaussian weights for block cells
fn compute_gaussian_weights(block_size: usize) -> Vec<f32> {
    let mut weights = vec![0.0; block_size * block_size];
    let sigma = block_size as f32 * 0.5;
    let center = (block_size - 1) as f32 * 0.5;

    for y in 0..block_size {
        for x in 0..block_size {
            let dy = y as f32 - center;
            let dx = x as f32 - center;
            let dist_sq = dx * dx + dy * dy;
            weights[y * block_size + x] = (-dist_sq / (2.0 * sigma * sigma)).exp();
        }
    }

    // Normalize weights
    let sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }

    weights
}

/// Normalize block features
fn normalize_block_features(features: &mut [f32], method: HogNormalization) {
    match method {
        HogNormalization::L2 => {
            let norm = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for feature in features.iter_mut() {
                    *feature /= norm;
                }
            }
        }
        HogNormalization::L2Hys => {
            // L2 normalization
            let mut norm = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for feature in features.iter_mut() {
                    *feature /= norm;
                }
            }

            // Clip values
            let clip_threshold = 0.2;
            for feature in features.iter_mut() {
                *feature = feature.min(clip_threshold);
            }

            // Renormalize
            norm = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for feature in features.iter_mut() {
                    *feature /= norm;
                }
            }
        }
        HogNormalization::L1 => {
            let norm = features.iter().map(|&x| x.abs()).sum::<f32>();
            if norm > 1e-6 {
                for feature in features.iter_mut() {
                    *feature /= norm;
                }
            }
        }
        HogNormalization::L1Sqrt => {
            // L1 normalization
            let norm = features.iter().map(|&x| x.abs()).sum::<f32>();
            if norm > 1e-6 {
                for feature in features.iter_mut() {
                    *feature /= norm;
                }
            }

            // Square root
            for feature in features.iter_mut() {
                *feature = feature.signum() * feature.abs().sqrt();
            }
        }
    }
}

/// Visualize HOG features
pub fn visualize_hog(
    descriptor: &HogDescriptor,
    cell_size: usize,
    bin_count: usize,
) -> Array2<f32> {
    let height = descriptor.cells_y * cell_size;
    let width = descriptor.cells_x * cell_size;
    let mut visualization = Array2::zeros((height, width));

    // Draw oriented lines for each cell
    for cy in 0..descriptor.cells_y {
        for cx in 0..descriptor.cells_x {
            // Get histogram for this cell
            let hist_start = (cy * descriptor.cells_x + cx) * bin_count;
            let histogram = &descriptor.features[hist_start..hist_start + bin_count];

            // Find dominant orientation
            let max_idx = histogram
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let angle = (max_idx as f32 + 0.5) * PI / bin_count as f32;
            let magnitude = histogram[max_idx];

            // Draw line in cell
            let center_y = cy * cell_size + cell_size / 2;
            let center_x = cx * cell_size + cell_size / 2;

            let dx = (angle.cos() * cell_size as f32 * 0.4 * magnitude) as isize;
            let dy = (angle.sin() * cell_size as f32 * 0.4 * magnitude) as isize;

            // Simple line drawing (Bresenham's algorithm would be better)
            for t in 0..10 {
                let t = t as f32 / 9.0;
                let x = (center_x as isize + (dx as f32 * (t - 0.5)) as isize) as usize;
                let y = (center_y as isize + (dy as f32 * (t - 0.5)) as isize) as usize;

                if x < width && y < height {
                    visualization[[y, x]] = 1.0;
                }
            }
        }
    }

    visualization
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hog_config() {
        let config = HogConfig::default();
        assert_eq!(config.cell_size, 8);
        assert_eq!(config.num_bins, 9);
        assert!(config.unsigned_gradients);
    }

    #[test]
    fn test_gaussian_weights() {
        let weights = compute_gaussian_weights(2);
        assert_eq!(weights.len(), 4);

        // Sum should be approximately 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Center values should be larger
        assert!(weights[0] > 0.0);
        assert!(weights[0] == weights[3]); // Symmetry
    }
}
