//! Image processing transformers for feature extraction
//!
//! This module provides utilities for extracting features from images,
//! including patch extraction, HOG features, and image normalization.

use ndarray::{par_azip, s, Array1, Array2, Array3, Array4};
use std::f64::consts::PI;

use crate::error::{Result, TransformError};

/// Extract patches from 2D images
pub struct PatchExtractor {
    /// Patch size (height, width)
    patch_size: (usize, usize),
    /// Maximum number of patches to extract
    max_patches: Option<usize>,
    /// Random seed for reproducible patch selection
    random_state: Option<u64>,
}

impl PatchExtractor {
    /// Create a new patch extractor
    pub fn new(_patchsize: (usize, usize)) -> Self {
        PatchExtractor {
            patch_size: _patchsize,
            max_patches: None,
            random_state: None,
        }
    }

    /// Set maximum number of patches to extract
    pub fn with_max_patches(mut self, maxpatches: usize) -> Self {
        self.max_patches = Some(maxpatches);
        self
    }

    /// Set random seed for reproducible patch selection
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Extract patches from a 2D grayscale image
    pub fn extract_patches_2d(&self, image: &Array2<f64>) -> Result<Array3<f64>> {
        let (img_height, img_width) = (image.shape()[0], image.shape()[1]);
        let (patch_height, patch_width) = self.patch_size;

        if patch_height > img_height || patch_width > img_width {
            return Err(TransformError::InvalidInput(format!(
                "Patch size ({patch_height}, {patch_width}) exceeds image size ({img_height}, {img_width})"
            )));
        }

        let n_patches_h = img_height - patch_height + 1;
        let n_patches_w = img_width - patch_width + 1;
        let total_patches = n_patches_h * n_patches_w;

        let n_patches = if let Some(max_p) = self.max_patches {
            max_p.min(total_patches)
        } else {
            total_patches
        };

        let mut patches = Array3::zeros((n_patches, patch_height, patch_width));

        if n_patches == total_patches {
            // Extract all patches
            let mut patch_idx = 0;
            for i in 0..n_patches_h {
                for j in 0..n_patches_w {
                    let patch = image.slice(s![i..i + patch_height, j..j + patch_width]);
                    patches.slice_mut(s![patch_idx, .., ..]).assign(&patch);
                    patch_idx += 1;
                }
            }
        } else {
            // Random patch selection
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};

            let mut rng = if let Some(seed) = self.random_state {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::seed_from_u64(rand::random::<u64>())
            };

            for patch_idx in 0..n_patches {
                let i = rng.gen_range(0..n_patches_h);
                let j = rng.gen_range(0..n_patches_w);
                let patch = image.slice(s![i..i + patch_height, j..j + patch_width]);
                patches.slice_mut(s![patch_idx, .., ..]).assign(&patch);
            }
        }

        Ok(patches)
    }

    /// Extract patches from a batch of 2D images
    pub fn extract_patches_batch(&self, images: &Array3<f64>) -> Result<Array4<f64>> {
        let n_images = images.shape()[0];
        let (img_height, img_width) = (images.shape()[1], images.shape()[2]);
        let (patch_height, patch_width) = self.patch_size;

        if patch_height > img_height || patch_width > img_width {
            return Err(TransformError::InvalidInput(format!(
                "Patch size ({patch_height}, {patch_width}) exceeds image size ({img_height}, {img_width})"
            )));
        }

        let n_patches_per_image = if let Some(max_p) = self.max_patches {
            let total = (img_height - patch_height + 1) * (img_width - patch_width + 1);
            max_p.min(total)
        } else {
            (img_height - patch_height + 1) * (img_width - patch_width + 1)
        };

        let mut all_patches =
            Array4::zeros((n_images * n_patches_per_image, patch_height, patch_width, 1));

        for (img_idx, image) in images.outer_iter().enumerate() {
            let patches = self.extract_patches_2d(&image.to_owned())?;
            let start_idx = img_idx * n_patches_per_image;

            for (patch_idx, patch) in patches.outer_iter().enumerate() {
                all_patches
                    .slice_mut(s![start_idx + patch_idx, .., .., 0])
                    .assign(&patch);
            }
        }

        Ok(all_patches)
    }
}

/// Histogram of Oriented Gradients (HOG) feature extractor
pub struct HOGDescriptor {
    /// Size of each cell in pixels (height, width)
    cell_size: (usize, usize),
    /// Size of each block in cells (height, width)
    block_size: (usize, usize),
    /// Number of orientation bins
    n_bins: usize,
    /// Block normalization method
    block_norm: BlockNorm,
}

/// Block normalization methods for HOG
#[derive(Clone, Copy, Debug)]
pub enum BlockNorm {
    /// L1 normalization
    L1,
    /// L2 normalization
    L2,
    /// L1-sqrt normalization
    L1Sqrt,
    /// L2-Hys normalization (L2 with clipping)
    L2Hys,
}

impl HOGDescriptor {
    /// Create a new HOG descriptor
    pub fn new(_cellsize: (usize, usize), block_size: (usize, usize), n_bins: usize) -> Self {
        HOGDescriptor {
            cell_size: _cellsize,
            block_size,
            n_bins,
            block_norm: BlockNorm::L2Hys,
        }
    }

    /// Set block normalization method
    pub fn with_block_norm(mut self, blocknorm: BlockNorm) -> Self {
        self.block_norm = blocknorm;
        self
    }

    /// Compute gradients for an image
    fn compute_gradients(&self, image: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (height, width) = (image.shape()[0], image.shape()[1]);
        let mut grad_x = Array2::zeros((height, width));
        let mut grad_y = Array2::zeros((height, width));

        // Compute x-gradients
        for i in 0..height {
            for j in 1..width - 1 {
                grad_x[[i, j]] = image[[i, j + 1]] - image[[i, j - 1]];
            }
            // Handle boundaries
            grad_x[[i, 0]] = image[[i, 1]] - image[[i, 0]];
            grad_x[[i, width - 1]] = image[[i, width - 1]] - image[[i, width - 2]];
        }

        // Compute y-gradients
        for j in 0..width {
            for i in 1..height - 1 {
                grad_y[[i, j]] = image[[i + 1, j]] - image[[i - 1, j]];
            }
            // Handle boundaries
            grad_y[[0, j]] = image[[1, j]] - image[[0, j]];
            grad_y[[height - 1, j]] = image[[height - 1, j]] - image[[height - 2, j]];
        }

        (grad_x, grad_y)
    }

    /// Compute HOG features for a single image
    pub fn compute(&self, image: &Array2<f64>) -> Result<Array1<f64>> {
        let (height, width) = (image.shape()[0], image.shape()[1]);
        let (cell_h, cell_w) = self.cell_size;
        let (block_h, block_w) = self.block_size;

        // Compute gradients
        let (grad_x, grad_y) = self.compute_gradients(image);

        // Compute magnitude and orientation
        let magnitude = (&grad_x * &grad_x + &grad_y * &grad_y).mapv(f64::sqrt);
        let mut orientation = grad_y.mapv(|y| y.atan2(0.0));
        orientation.zip_mut_with(&grad_x, |o, &x| *o = (*o).atan2(x));

        // Number of cells
        let n_cells_h = height / cell_h;
        let n_cells_w = width / cell_w;

        // Build orientation histograms for each cell
        let mut cell_histograms = Array3::zeros((n_cells_h, n_cells_w, self.n_bins));
        let bin_size = PI / self.n_bins as f64;

        for cell_i in 0..n_cells_h {
            for cell_j in 0..n_cells_w {
                let start_i = cell_i * cell_h;
                let start_j = cell_j * cell_w;

                for i in start_i..start_i.min(start_i + cell_h).min(height) {
                    for j in start_j..start_j.min(start_j + cell_w).min(width) {
                        let mag = magnitude[[i, j]];
                        let mut angle = orientation[[i, j]];

                        // Convert to 0-pi range
                        if angle < 0.0 {
                            angle += PI;
                        }

                        // Compute bin indices
                        let bin_idx = (angle / bin_size) as usize;
                        let bin_idx = bin_idx.min(self.n_bins - 1);

                        cell_histograms[[cell_i, cell_j, bin_idx]] += mag;
                    }
                }
            }
        }

        // Number of blocks
        let n_blocks_h = n_cells_h - block_h + 1;
        let n_blocks_w = n_cells_w - block_w + 1;
        let block_features = block_h * block_w * self.n_bins;

        let mut features = Vec::with_capacity(n_blocks_h * n_blocks_w * block_features);

        // Extract and normalize blocks
        for block_i in 0..n_blocks_h {
            for block_j in 0..n_blocks_w {
                let mut block_hist = Vec::with_capacity(block_features);

                // Collect histograms from cells in this block
                for i in 0..block_h {
                    for j in 0..block_w {
                        let cell_hist = cell_histograms.slice(s![block_i + i, block_j + j, ..]);
                        block_hist.extend(cell_hist.iter());
                    }
                }

                // Normalize block
                let block_hist = self.normalize_block(&block_hist);
                features.extend(block_hist);
            }
        }

        Ok(Array1::from_vec(features))
    }

    /// Normalize a block histogram
    fn normalize_block(&self, hist: &[f64]) -> Vec<f64> {
        let epsilon = 1e-8;

        match self.block_norm {
            BlockNorm::L1 => {
                let norm: f64 = hist.iter().sum::<f64>() + epsilon;
                hist.iter().map(|&v| v / norm).collect()
            }
            BlockNorm::L2 => {
                let norm = hist.iter().map(|&v| v * v).sum::<f64>().sqrt() + epsilon;
                hist.iter().map(|&v| v / norm).collect()
            }
            BlockNorm::L1Sqrt => {
                let norm: f64 = hist.iter().sum::<f64>() + epsilon;
                hist.iter().map(|&v| (v / norm).sqrt()).collect()
            }
            BlockNorm::L2Hys => {
                // L2 normalization with clipping
                let mut norm = hist.iter().map(|&v| v * v).sum::<f64>().sqrt() + epsilon;
                let mut normalized: Vec<f64> = hist.iter().map(|&v| v / norm).collect();

                // Clip values to 0.2
                let clip_val = 0.2;
                for v in &mut normalized {
                    if *v > clip_val {
                        *v = clip_val;
                    }
                }

                // Re-normalize
                norm = normalized.iter().map(|&v| v * v).sum::<f64>().sqrt() + epsilon;
                normalized.iter_mut().for_each(|v| *v /= norm);

                normalized
            }
        }
    }
}

/// Image normalization transformer
pub struct ImageNormalizer {
    /// Normalization method
    method: ImageNormMethod,
    /// Channel-wise statistics (mean, std) for standardization
    channel_stats: Option<(Array1<f64>, Array1<f64>)>,
}

/// Image normalization methods
#[derive(Clone, Copy, Debug)]
pub enum ImageNormMethod {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Standardization (zero mean, unit variance)
    Standard,
    /// Normalization to [-1, 1]
    Symmetric,
    /// Custom range normalization
    Range(f64, f64),
}

impl ImageNormalizer {
    /// Create a new image normalizer
    pub fn new(method: ImageNormMethod) -> Self {
        ImageNormalizer {
            method,
            channel_stats: None,
        }
    }

    /// Fit the normalizer on a batch of images
    pub fn fit(&mut self, images: &Array4<f64>) -> Result<()> {
        if let ImageNormMethod::Standard = self.method {
            let n_channels = images.shape()[3];
            let mut means = Array1::zeros(n_channels);
            let mut stds = Array1::zeros(n_channels);

            // Compute channel-wise statistics
            for c in 0..n_channels {
                let channel_data = images.slice(s![.., .., .., c]);
                let flat_data: Vec<f64> = channel_data.iter().cloned().collect();

                let mean = flat_data.iter().sum::<f64>() / flat_data.len() as f64;
                let variance = flat_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / flat_data.len() as f64;

                means[c] = mean;
                stds[c] = variance.sqrt();
            }

            self.channel_stats = Some((means, stds));
        }

        Ok(())
    }

    /// Transform images
    pub fn transform(&self, images: &Array4<f64>) -> Result<Array4<f64>> {
        let mut result = images.clone();

        match self.method {
            ImageNormMethod::MinMax => {
                // Normalize each image independently to [0, 1]
                for mut image in result.outer_iter_mut() {
                    let min = image.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;

                    if range > 0.0 {
                        image.mapv_inplace(|v| (v - min) / range);
                    }
                }
            }
            ImageNormMethod::Standard => {
                if let Some((ref means, ref stds)) = self.channel_stats {
                    let n_channels = images.shape()[3];

                    for c in 0..n_channels {
                        let mean = means[c];
                        let std = stds[c].max(1e-8); // Avoid division by zero

                        result
                            .slice_mut(s![.., .., .., c])
                            .mapv_inplace(|v| (v - mean) / std);
                    }
                } else {
                    return Err(TransformError::NotFitted(
                        "ImageNormalizer must be fitted before transform".into(),
                    ));
                }
            }
            ImageNormMethod::Symmetric => {
                // Normalize to [-1, 1]
                for mut image in result.outer_iter_mut() {
                    let min = image.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;

                    if range > 0.0 {
                        image.mapv_inplace(|v| 2.0 * (v - min) / range - 1.0);
                    }
                }
            }
            ImageNormMethod::Range(new_min, new_max) => {
                let new_range = new_max - new_min;

                for mut image in result.outer_iter_mut() {
                    let min = image.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;

                    if range > 0.0 {
                        image.mapv_inplace(|v| new_min + new_range * (v - min) / range);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, images: &Array4<f64>) -> Result<Array4<f64>> {
        self.fit(images)?;
        self.transform(images)
    }
}

/// Convert RGB images to grayscale
#[allow(dead_code)]
pub fn rgb_to_grayscale(images: &Array4<f64>) -> Result<Array3<f64>> {
    let shape = images.shape();
    if shape[3] != 3 {
        return Err(TransformError::InvalidInput(format!(
            "Expected 3 channels for RGB, got {}",
            shape[3]
        )));
    }

    let (n_samples, height, width) = (shape[0], shape[1], shape[2]);
    let mut grayscale = Array3::zeros((n_samples, height, width));

    // Use standard RGB to grayscale conversion weights
    let weights = [0.2989, 0.5870, 0.1140];

    par_azip!((mut gray in grayscale.outer_iter_mut(),
               rgb in images.outer_iter()) {
        for i in 0..height {
            for j in 0..width {
                gray[[i, j]] = weights[0] * rgb[[i, j, 0]]
                             + weights[1] * rgb[[i, j, 1]]
                             + weights[2] * rgb[[i, j, 2]];
            }
        }
    });

    Ok(grayscale)
}

/// Resize images using bilinear interpolation
#[allow(dead_code)]
pub fn resize_images(images: &Array4<f64>, newsize: (usize, usize)) -> Result<Array4<f64>> {
    let (n_samples, old_h, old_w, n_channels) = {
        let shape = images.shape();
        (shape[0], shape[1], shape[2], shape[3])
    };
    let (new_h, new_w) = newsize;

    let mut resized = Array4::zeros((n_samples, new_h, new_w, n_channels));

    let scale_h = old_h as f64 / new_h as f64;
    let scale_w = old_w as f64 / new_w as f64;

    par_azip!((mut resized_img in resized.outer_iter_mut(),
               original_img in images.outer_iter()) {
        for i in 0..new_h {
            for j in 0..new_w {
                // Map to original coordinates
                let orig_i = i as f64 * scale_h;
                let orig_j = j as f64 * scale_w;

                // Get integer parts and fractions
                let i0 = orig_i.floor() as usize;
                let j0 = orig_j.floor() as usize;
                let i1 = (i0 + 1).min(old_h - 1);
                let j1 = (j0 + 1).min(old_w - 1);

                let di = orig_i - i0 as f64;
                let dj = orig_j - j0 as f64;

                // Bilinear interpolation for each channel
                for c in 0..n_channels {
                    let v00 = original_img[[i0, j0, c]];
                    let v01 = original_img[[i0, j1, c]];
                    let v10 = original_img[[i1, j0, c]];
                    let v11 = original_img[[i1, j1, c]];

                    let v0 = v00 * (1.0 - dj) + v01 * dj;
                    let v1 = v10 * (1.0 - dj) + v11 * dj;
                    let v = v0 * (1.0 - di) + v1 * di;

                    resized_img[[i, j, c]] = v;
                }
            }
        }
    });

    Ok(resized)
}
