//! Non-Local Means denoising module
//!
//! This module implements Non-Local Means (NLM) denoising techniques for signal and image
//! processing. NLM leverages the redundancy in signals by comparing patches rather than individual
//! points, allowing for better detail preservation than traditional filters.
//!
//! The implementation includes:
//! - 1D Non-Local Means for signals
//! - 2D Non-Local Means for images
//! - Fast approximations for efficient computation
//! - Block-matching variants for improved performance
//!
//! # Example
//! ```ignore
//! # FIXME: Integer overflow in nlm_denoise functions
//! use ndarray::{Array1, Array2};
//! use scirs2_signal::nlm::{nlm_denoise_1d, nlm_denoise_2d, NlmConfig};
//! use rand::Rng;
//!
//! // Create a test signal
//! let n = 500;
//! let mut clean_signal = Array1::zeros(n);
//! for i in 100..400 {
//!     clean_signal[i] = 1.0;
//! }
//!
//! // Add noise
//! let mut rng = rand::thread_rng();
//! let mut noisy_signal = clean_signal.clone();
//! for i in 0..n {
//!     noisy_signal[i] += 0.2 * rng.random_range(-1.0..1.0);
//! }
//!
//! // Apply Non-Local Means denoising
//! let config = NlmConfig::default();
//! let denoised = nlm_denoise_1d(&noisy_signal, &config).unwrap();
//! ```

use ndarray::{s, Array1, Array2, Array3, Axis};
use std::cmp;
use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

/// Configuration for Non-Local Means denoising
#[derive(Debug, Clone)]
pub struct NlmConfig {
    /// Patch size for similarity computation
    pub patch_size: usize,

    /// Search window size around each point
    pub search_window: usize,

    /// Filtering parameter controlling decay of weights
    pub h: f64,

    /// Whether to use fast approximation
    pub fast_mode: bool,

    /// Step size for fast mode (sampling every n-th patch)
    pub step_size: usize,

    /// Whether to apply boundary reflection
    pub boundary: bool,

    /// Whether to use distance weightage in the search window
    pub distance_weighting: bool,
}

impl Default for NlmConfig {
    fn default() -> Self {
        Self {
            patch_size: 7,
            search_window: 21,
            h: 0.4,
            fast_mode: false,
            step_size: 3,
            boundary: true,
            distance_weighting: true,
        }
    }
}

/// Applies Non-Local Means denoising to a 1D signal.
///
/// Non-Local Means compares patches of the signal to find similar regions,
/// then computes a weighted average based on patch similarity.
///
/// # Arguments
/// * `signal` - Noisy input signal
/// * `config` - NLM configuration parameters
///
/// # Returns
/// * The denoised signal
///
/// # Example
/// ```ignore
/// # FIXME: Integer overflow in nlm_denoise functions
/// use ndarray::Array1;
/// use scirs2_signal::nlm::{nlm_denoise_1d, NlmConfig};
///
/// let signal = Array1::from_vec(vec![1.2, 2.3, 3.1, 2.2, 1.3, 0.2, -0.3, -1.1]);
/// let config = NlmConfig::default();
/// let denoised = nlm_denoise_1d(&signal, &config).unwrap();
/// ```
pub fn nlm_denoise_1d(signal: &Array1<f64>, config: &NlmConfig) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Ensure patch size is valid
    if config.patch_size < 3 || config.patch_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Patch size must be odd and at least 3".to_string(),
        ));
    }

    // Ensure search window is valid
    if config.search_window < config.patch_size {
        return Err(SignalError::ValueError(
            "Search window must be at least as large as patch size".to_string(),
        ));
    }

    // Compute half sizes
    let half_patch = config.patch_size / 2;
    let half_search = config.search_window / 2;

    // Pad signal for boundary handling
    let padded_signal = if config.boundary {
        pad_signal_1d(signal, cmp::max(half_patch, half_search))
    } else {
        signal.clone()
    };

    // Initialize denoised signal
    let mut denoised = Array1::zeros(n);

    // Determine step size for fast mode
    let step = if config.fast_mode {
        config.step_size
    } else {
        1
    };

    // Precompute Gaussian kernel for distance weighting if enabled
    let distance_kernel = if config.distance_weighting {
        let sigma_d = half_search as f64 / 3.0;
        let mut kernel = Array1::zeros(config.search_window);
        for i in 0..config.search_window {
            let d = (i as f64 - half_search as f64).powi(2);
            kernel[i] = (-d / (2.0 * sigma_d.powi(2))).exp();
        }
        Some(kernel)
    } else {
        None
    };

    // Compute estimated noise variance
    let estimated_sigma = estimate_noise_sigma(&padded_signal);

    // Adjust filtering parameter based on noise level
    let h_adjusted = config.h.powi(2) * estimated_sigma.powi(2) * config.patch_size as f64;

    // Process each point in the signal
    for i in 0..n {
        // Get patch centered at current point
        let center_idx = i + (if config.boundary { half_search } else { 0 });
        let center_patch = extract_patch_1d(&padded_signal, center_idx, config.patch_size);

        // Determine search region boundaries
        let search_start = if config.boundary {
            center_idx - half_search
        } else {
            cmp::max(half_patch, center_idx.saturating_sub(half_search))
        };

        let search_end = if config.boundary {
            center_idx + half_search + 1
        } else {
            cmp::min(
                padded_signal.len() - half_patch,
                center_idx + half_search + 1,
            )
        };

        // Initialize weight sum and weighted sum
        let mut weight_sum = 0.0;
        let mut weighted_sum = 0.0;

        // Search for similar patches
        for j in (search_start..search_end).step_by(step) {
            // Skip if the same position in fast mode (use center pixel weight 1.0)
            if config.fast_mode && j == center_idx {
                continue;
            }

            // Extract patch at search position
            let search_patch = extract_patch_1d(&padded_signal, j, config.patch_size);

            // Compute patch distance
            let dist = compute_patch_distance(&center_patch, &search_patch);

            // Apply spatial distance weighting if enabled
            let dist_weight =
                if let (true, Some(kernel)) = (config.distance_weighting, &distance_kernel) {
                    let idx = ((j as isize - center_idx as isize) + half_search as isize) as usize;
                    if idx < kernel.len() {
                        kernel[idx]
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

            // Compute weight using Gaussian kernel
            let weight = (-dist / h_adjusted).exp() * dist_weight;

            // Accumulate weighted contribution
            let idx = j - (if config.boundary { half_search } else { 0 });
            if idx < padded_signal.len() {
                weighted_sum += weight * padded_signal[idx];
                weight_sum += weight;
            }
        }

        // Add self-weight if using fast mode
        if config.fast_mode {
            weighted_sum += padded_signal[center_idx];
            weight_sum += 1.0;
        }

        // Normalize and store result
        if weight_sum > 1e-10 {
            denoised[i] = weighted_sum / weight_sum;
        } else {
            denoised[i] = padded_signal[center_idx];
        }
    }

    Ok(denoised)
}

/// Applies Non-Local Means denoising to a 2D image.
///
/// This is an extension of the 1D version that operates on 2D patches
/// and is particularly effective for image denoising.
///
/// # Arguments
/// * `image` - Noisy input image
/// * `config` - NLM configuration parameters
///
/// # Returns
/// * The denoised image
///
/// # Example
/// ```ignore
/// # FIXME: Integer overflow in nlm_denoise functions
/// use ndarray::Array2;
/// use scirs2_signal::nlm::{nlm_denoise_2d, NlmConfig};
///
/// let image = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64 / 20.0);
/// let config = NlmConfig::default();
/// let denoised = nlm_denoise_2d(&image, &config).unwrap();
/// ```
pub fn nlm_denoise_2d(image: &Array2<f64>, config: &NlmConfig) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Ensure patch size is valid
    if config.patch_size < 3 || config.patch_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Patch size must be odd and at least 3".to_string(),
        ));
    }

    // Ensure search window is valid
    if config.search_window < config.patch_size {
        return Err(SignalError::ValueError(
            "Search window must be at least as large as patch size".to_string(),
        ));
    }

    // Compute half sizes
    let half_patch = config.patch_size / 2;
    let half_search = config.search_window / 2;

    // Pad image for boundary handling
    let padded_image = if config.boundary {
        pad_image_2d(image, cmp::max(half_patch, half_search))
    } else {
        image.clone()
    };

    // Initialize denoised image
    let mut denoised = Array2::zeros((height, width));

    // Determine step size for fast mode
    let step = if config.fast_mode {
        config.step_size
    } else {
        1
    };

    // Compute estimated noise variance
    let estimated_sigma = estimate_noise_sigma_2d(&padded_image);

    // Adjust filtering parameter based on noise level
    let h_adjusted = config.h.powi(2) * estimated_sigma.powi(2) * config.patch_size.pow(2) as f64;

    // Precompute Gaussian kernel for patch comparisons (patch weights)
    let mut patch_weights = Array2::zeros((config.patch_size, config.patch_size));
    let sigma_g = config.patch_size as f64 / 6.0;
    let gauss_norm = 2.0 * PI * sigma_g.powi(2);

    for i in 0..config.patch_size {
        for j in 0..config.patch_size {
            let di = (i as f64 - half_patch as f64).powi(2);
            let dj = (j as f64 - half_patch as f64).powi(2);
            patch_weights[[i, j]] = (-0.5 * (di + dj) / sigma_g.powi(2)).exp() / gauss_norm;
        }
    }

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Get patch centered at current pixel
            let center_i = i + (if config.boundary { half_search } else { 0 });
            let center_j = j + (if config.boundary { half_search } else { 0 });
            let center_patch =
                extract_patch_2d(&padded_image, center_i, center_j, config.patch_size);

            // Determine search region boundaries
            let search_i_start = if config.boundary {
                center_i - half_search
            } else {
                cmp::max(half_patch, center_i.saturating_sub(half_search))
            };

            let search_i_end = if config.boundary {
                center_i + half_search + 1
            } else {
                cmp::min(
                    padded_image.dim().0 - half_patch,
                    center_i + half_search + 1,
                )
            };

            let search_j_start = if config.boundary {
                center_j - half_search
            } else {
                cmp::max(half_patch, center_j.saturating_sub(half_search))
            };

            let search_j_end = if config.boundary {
                center_j + half_search + 1
            } else {
                cmp::min(
                    padded_image.dim().1 - half_patch,
                    center_j + half_search + 1,
                )
            };

            // Initialize weight sum and weighted sum
            let mut weight_sum = 0.0;
            let mut weighted_sum = 0.0;

            // Search for similar patches
            for si in (search_i_start..search_i_end).step_by(step) {
                for sj in (search_j_start..search_j_end).step_by(step) {
                    // Skip if the same position in fast mode
                    if config.fast_mode && si == center_i && sj == center_j {
                        continue;
                    }

                    // Extract patch at search position
                    let search_patch = extract_patch_2d(&padded_image, si, sj, config.patch_size);

                    // Compute weighted patch distance
                    let dist = compute_patch_distance_2d(
                        &center_patch,
                        &search_patch,
                        Some(&patch_weights),
                    );

                    // Apply spatial distance weighting if enabled
                    let dist_weight = if config.distance_weighting {
                        let di = (si as f64 - center_i as f64).powi(2);
                        let dj = (sj as f64 - center_j as f64).powi(2);
                        let spatial_dist = (di + dj).sqrt();
                        let spatial_sigma = half_search as f64 / 3.0;
                        (-spatial_dist / (2.0 * spatial_sigma.powi(2))).exp()
                    } else {
                        1.0
                    };

                    // Compute weight using Gaussian kernel
                    let weight = (-dist / h_adjusted).exp() * dist_weight;

                    // Accumulate weighted contribution
                    let idx_i = si - (if config.boundary { half_search } else { 0 });
                    let idx_j = sj - (if config.boundary { half_search } else { 0 });

                    if idx_i < padded_image.dim().0 && idx_j < padded_image.dim().1 {
                        weighted_sum += weight * padded_image[[idx_i, idx_j]];
                        weight_sum += weight;
                    }
                }
            }

            // Add self-weight if using fast mode
            if config.fast_mode {
                weighted_sum += padded_image[[center_i, center_j]];
                weight_sum += 1.0;
            }

            // Normalize and store result
            if weight_sum > 1e-10 {
                denoised[[i, j]] = weighted_sum / weight_sum;
            } else {
                denoised[[i, j]] = padded_image[[center_i, center_j]];
            }
        }
    }

    Ok(denoised)
}

/// Fast implementation of Non-Local Means for 2D images using block-matching.
///
/// This implementation accelerates NLM by first grouping similar patches into blocks,
/// then processing each block efficiently.
///
/// # Arguments
/// * `image` - Noisy input image
/// * `config` - NLM configuration parameters
/// * `max_blocks` - Maximum number of similar blocks to use for each reference patch
///
/// # Returns
/// * The denoised image
pub fn nlm_block_matching_2d(
    image: &Array2<f64>,
    config: &NlmConfig,
    max_blocks: usize,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Ensure patch size is valid
    if config.patch_size < 3 || config.patch_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Patch size must be odd and at least 3".to_string(),
        ));
    }

    // Ensure search window is valid
    if config.search_window < config.patch_size {
        return Err(SignalError::ValueError(
            "Search window must be at least as large as patch size".to_string(),
        ));
    }

    // Compute half sizes
    let half_patch = config.patch_size / 2;
    let half_search = config.search_window / 2;

    // Pad image for boundary handling
    let padded_image = if config.boundary {
        pad_image_2d(image, cmp::max(half_patch, half_search))
    } else {
        image.clone()
    };

    // Compute estimated noise variance
    let estimated_sigma = estimate_noise_sigma_2d(&padded_image);

    // Adjust filtering parameter based on noise level
    let h_adjusted = config.h.powi(2) * estimated_sigma.powi(2) * config.patch_size.pow(2) as f64;

    // Initialize denoised image and weight accumulator
    let mut denoised = Array2::zeros((height, width));
    let mut weight_sum = Array2::<f64>::zeros((height, width));

    // Determine step size for processing (subsampling the image)
    let step = config.step_size;

    // Process the image in blocks
    for i in (half_patch..height + half_search - half_patch).step_by(step) {
        for j in (half_patch..width + half_search - half_patch).step_by(step) {
            // Get reference patch
            let ref_patch = extract_patch_2d(&padded_image, i, j, config.patch_size);

            // Find similar patches within search window
            let mut similar_blocks = Vec::new();
            let mut block_weights = Vec::new();

            // Search for similar patches
            for si in i.saturating_sub(half_search)
                ..cmp::min(i + half_search + 1, height + 2 * half_search)
            {
                for sj in j.saturating_sub(half_search)
                    ..cmp::min(j + half_search + 1, width + 2 * half_search)
                {
                    // Skip if the same position
                    if si == i && sj == j {
                        continue;
                    }

                    // Extract patch at search position
                    let search_patch = extract_patch_2d(&padded_image, si, sj, config.patch_size);

                    // Compute patch distance
                    let dist = compute_patch_distance_2d(&ref_patch, &search_patch, None);

                    // Apply distance weighting if enabled
                    let dist_weight = if config.distance_weighting {
                        let di = (si as f64 - i as f64).powi(2);
                        let dj = (sj as f64 - j as f64).powi(2);
                        let spatial_dist = (di + dj).sqrt();
                        let spatial_sigma = half_search as f64 / 3.0;
                        (-spatial_dist / (2.0 * spatial_sigma.powi(2))).exp()
                    } else {
                        1.0
                    };

                    // Compute weight
                    let weight = (-dist / h_adjusted).exp() * dist_weight;

                    // Store block and weight if significant
                    if weight > 0.01 {
                        similar_blocks.push((si, sj));
                        block_weights.push(weight);

                        // Limit the number of blocks
                        if similar_blocks.len() >= max_blocks {
                            break;
                        }
                    }
                }

                if similar_blocks.len() >= max_blocks {
                    break;
                }
            }

            // Always include reference patch
            similar_blocks.push((i, j));
            block_weights.push(1.0);

            // Normalize weights
            let total_weight: f64 = block_weights.iter().sum();
            let normalized_weights: Vec<f64> =
                block_weights.iter().map(|&w| w / total_weight).collect();

            // Apply weighted average to each pixel in the patch
            for pi in 0..config.patch_size {
                for pj in 0..config.patch_size {
                    let global_i =
                        i + pi - half_patch - (if config.boundary { half_search } else { 0 });
                    let global_j =
                        j + pj - half_patch - (if config.boundary { half_search } else { 0 });

                    // Skip if outside image bounds
                    if global_i >= height || global_j >= width {
                        continue;
                    }

                    // Compute weighted average for this pixel across all similar patches
                    let mut pixel_sum = 0.0;
                    for (block_idx, &(si, sj)) in similar_blocks.iter().enumerate() {
                        let si_adj = si + pi - half_patch;
                        let sj_adj = sj + pj - half_patch;
                        pixel_sum += normalized_weights[block_idx] * padded_image[[si_adj, sj_adj]];
                    }

                    // Accumulate weighted result
                    denoised[[global_i, global_j]] += pixel_sum;
                    weight_sum[[global_i, global_j]] += 1.0;
                }
            }
        }
    }

    // Normalize accumulated weights
    for i in 0..height {
        for j in 0..width {
            if weight_sum[[i, j]] > 0.0 {
                denoised[[i, j]] /= weight_sum[[i, j]];
            } else {
                // If no contribution, use original pixel
                denoised[[i, j]] = padded_image[[
                    i + (if config.boundary { half_search } else { 0 }),
                    j + (if config.boundary { half_search } else { 0 }),
                ]];
            }
        }
    }

    Ok(denoised)
}

/// Implements a multi-scale NLM approach for improved denoising.
///
/// This method processes the image at multiple scales to capture both
/// small and large-scale structure, then combines the results.
///
/// # Arguments
/// * `image` - Noisy input image
/// * `config` - NLM configuration parameters
/// * `num_scales` - Number of scales to use (typically 2-3)
///
/// # Returns
/// * The denoised image
pub fn nlm_multiscale_2d(
    image: &Array2<f64>,
    config: &NlmConfig,
    num_scales: usize,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    // Initialize result image
    let mut result = Array2::zeros((height, width));

    // Process each scale
    for scale in 0..num_scales {
        // Create scale-specific configuration
        let mut scale_config = config.clone();

        // Adjust patch and search window sizes for current scale
        let scale_factor = 2u32.pow(scale as u32) as usize;
        scale_config.patch_size = cmp::max(3, config.patch_size / scale_factor);
        scale_config.search_window = cmp::max(
            scale_config.patch_size + 2,
            config.search_window / scale_factor,
        );

        // Adjust filtering parameter
        scale_config.h = config.h * scale_factor as f64;

        // Resize image for current scale if not the first scale
        let scale_image = if scale == 0 {
            image.clone()
        } else {
            // Downsample image for this scale
            let scaled_height = height / scale_factor;
            let scaled_width = width / scale_factor;
            downsample_image(image, scaled_height, scaled_width)?
        };

        // Apply NLM at current scale
        let denoised_scale = nlm_denoise_2d(&scale_image, &scale_config)?;

        // Upsample denoised result to original size if needed
        let upsampled = if scale == 0 {
            denoised_scale
        } else {
            upsample_image(&denoised_scale, height, width)?
        };

        // Compute weight for this scale (higher weights for finer scales)
        let scale_weight = 1.0 / (2.0_f64.powi(scale as i32));

        // Accumulate weighted result
        for i in 0..height {
            for j in 0..width {
                result[[i, j]] += scale_weight * upsampled[[i, j]];
            }
        }
    }

    // Normalize by total weight
    let total_weight: f64 = (0..num_scales)
        .map(|s| 1.0 / (2.0_f64.powi(s as i32)))
        .sum();
    result.mapv_inplace(|x| x / total_weight);

    Ok(result)
}

/// Color image NLM denoising that preserves color relationships.
///
/// This function applies NLM to color images, taking into account color
/// channel correlations for better denoising.
///
/// # Arguments
/// * `image` - Noisy color image (3D array with last axis being RGB channels)
/// * `config` - NLM configuration parameters
///
/// # Returns
/// * The denoised color image
pub fn nlm_color_image(image: &Array3<f64>, config: &NlmConfig) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return Err(SignalError::ValueError(
            "Input must be a 3-channel color image".to_string(),
        ));
    }

    // Compute half sizes
    let half_patch = config.patch_size / 2;
    let half_search = config.search_window / 2;

    // Pad image for boundary handling
    let padded_image = if config.boundary {
        pad_color_image(image, cmp::max(half_patch, half_search))
    } else {
        image.clone()
    };

    // Initialize denoised image
    let mut denoised = Array3::zeros((height, width, channels));

    // Compute estimated noise variance per channel
    let mut channel_sigma = Vec::with_capacity(channels);
    for c in 0..channels {
        let channel = padded_image.index_axis(Axis(2), c).to_owned();
        channel_sigma.push(estimate_noise_sigma_2d(&channel));
    }

    // Adjust filtering parameter based on noise level (average across channels)
    let avg_sigma = channel_sigma.iter().sum::<f64>() / channels as f64;
    let h_adjusted =
        config.h.powi(2) * avg_sigma.powi(2) * config.patch_size.pow(2) as f64 * channels as f64;

    // Determine step size for fast mode
    let step = if config.fast_mode {
        config.step_size
    } else {
        1
    };

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Current coordinates in padded image
            let center_i = i + (if config.boundary { half_search } else { 0 });
            let center_j = j + (if config.boundary { half_search } else { 0 });

            // Determine search region boundaries
            let search_i_start = if config.boundary {
                center_i - half_search
            } else {
                cmp::max(half_patch, center_i.saturating_sub(half_search))
            };

            let search_i_end = if config.boundary {
                center_i + half_search + 1
            } else {
                cmp::min(
                    padded_image.dim().0 - half_patch,
                    center_i + half_search + 1,
                )
            };

            let search_j_start = if config.boundary {
                center_j - half_search
            } else {
                cmp::max(half_patch, center_j.saturating_sub(half_search))
            };

            let search_j_end = if config.boundary {
                center_j + half_search + 1
            } else {
                cmp::min(
                    padded_image.dim().1 - half_patch,
                    center_j + half_search + 1,
                )
            };

            // Initialize weight sum and weighted sum for each channel
            let mut weight_sum = 0.0;
            let mut weighted_sums = vec![0.0; channels];

            // Search for similar patches
            for si in (search_i_start..search_i_end).step_by(step) {
                for sj in (search_j_start..search_j_end).step_by(step) {
                    // Skip if the same position in fast mode
                    if config.fast_mode && si == center_i && sj == center_j {
                        continue;
                    }

                    // Compute patch distance (summed across all channels)
                    let mut total_dist = 0.0;

                    for c in 0..channels {
                        // Extract patches from each channel
                        let channel_view = padded_image.index_axis(Axis(2), c).to_owned();
                        let center_patch =
                            extract_patch_2d(&channel_view, center_i, center_j, config.patch_size);

                        let search_patch =
                            extract_patch_2d(&channel_view, si, sj, config.patch_size);

                        // Add weighted channel distance
                        total_dist += compute_patch_distance_2d(&center_patch, &search_patch, None);
                    }

                    // Apply spatial distance weighting if enabled
                    let dist_weight = if config.distance_weighting {
                        let di = (si as f64 - center_i as f64).powi(2);
                        let dj = (sj as f64 - center_j as f64).powi(2);
                        let spatial_dist = (di + dj).sqrt();
                        let spatial_sigma = half_search as f64 / 3.0;
                        (-spatial_dist / (2.0 * spatial_sigma.powi(2))).exp()
                    } else {
                        1.0
                    };

                    // Compute weight using Gaussian kernel
                    let weight = (-total_dist / h_adjusted).exp() * dist_weight;

                    // Accumulate weighted contribution for each channel
                    let idx_i = si - (if config.boundary { half_search } else { 0 });
                    let idx_j = sj - (if config.boundary { half_search } else { 0 });

                    if idx_i < padded_image.dim().0 && idx_j < padded_image.dim().1 {
                        for c in 0..channels {
                            weighted_sums[c] += weight * padded_image[[idx_i, idx_j, c]];
                        }
                        weight_sum += weight;
                    }
                }
            }

            // Add self-weight if using fast mode
            if config.fast_mode {
                for c in 0..channels {
                    weighted_sums[c] += padded_image[[center_i, center_j, c]];
                }
                weight_sum += 1.0;
            }

            // Normalize and store result for each channel
            if weight_sum > 1e-10 {
                for c in 0..channels {
                    denoised[[i, j, c]] = weighted_sums[c] / weight_sum;
                }
            } else {
                for c in 0..channels {
                    denoised[[i, j, c]] = padded_image[[center_i, center_j, c]];
                }
            }
        }
    }

    Ok(denoised)
}

/// Helper function to pad a 1D signal with reflection at boundaries
fn pad_signal_1d(signal: &Array1<f64>, pad_size: usize) -> Array1<f64> {
    let n = signal.len();
    let mut padded = Array1::zeros(n + 2 * pad_size);

    // Copy original signal
    for i in 0..n {
        padded[i + pad_size] = signal[i];
    }

    // Reflect boundaries
    for i in 0..pad_size {
        // Left boundary
        padded[pad_size - 1 - i] = signal[i.min(n - 1)];

        // Right boundary
        padded[n + pad_size + i] = signal[n - 1 - i.min(n - 1)];
    }

    padded
}

/// Helper function to pad a 2D image with reflection at boundaries
fn pad_image_2d(image: &Array2<f64>, pad_size: usize) -> Array2<f64> {
    let (height, width) = image.dim();
    let mut padded = Array2::zeros((height + 2 * pad_size, width + 2 * pad_size));

    // Copy original image
    for i in 0..height {
        for j in 0..width {
            padded[[i + pad_size, j + pad_size]] = image[[i, j]];
        }
    }

    // Pad top and bottom edges
    for i in 0..pad_size {
        for j in 0..width {
            // Top edge
            padded[[pad_size - 1 - i, j + pad_size]] = image[[i.min(height - 1), j]];

            // Bottom edge
            padded[[height + pad_size + i, j + pad_size]] =
                image[[height - 1 - i.min(height - 1), j]];
        }
    }

    // Pad left and right edges
    for i in 0..height + 2 * pad_size {
        for j in 0..pad_size {
            // Handle corner cases
            let _src_i = if i < pad_size {
                i.min(pad_size - 1)
            } else if i >= height + pad_size {
                height - 1 - (i - (height + pad_size)).min(height - 1)
            } else {
                i - pad_size
            };

            // Left edge
            padded[[i, pad_size - 1 - j]] = padded[[i, pad_size + j.min(width - 1)]];

            // Right edge
            padded[[i, width + pad_size + j]] =
                padded[[i, width + pad_size - 1 - j.min(width - 1)]];
        }
    }

    padded
}

/// Helper function to pad a 3D color image with reflection at boundaries
fn pad_color_image(image: &Array3<f64>, pad_size: usize) -> Array3<f64> {
    let (height, width, channels) = image.dim();
    let mut padded = Array3::zeros((height + 2 * pad_size, width + 2 * pad_size, channels));

    // Pad each channel separately
    for c in 0..channels {
        let channel = image.index_axis(Axis(2), c).to_owned();
        let padded_channel = pad_image_2d(&channel, pad_size);

        for i in 0..padded_channel.dim().0 {
            for j in 0..padded_channel.dim().1 {
                padded[[i, j, c]] = padded_channel[[i, j]];
            }
        }
    }

    padded
}

/// Helper function to extract a patch from a 1D signal
fn extract_patch_1d(signal: &Array1<f64>, center: usize, size: usize) -> Array1<f64> {
    let half_size = size / 2;

    // Check bounds to avoid panic
    if center < half_size || center + half_size >= signal.len() {
        // Handle boundary case
        let mut patch = Array1::zeros(size);
        for i in 0..size {
            let idx = center as isize - half_size as isize + i as isize;
            if idx >= 0 && idx < signal.len() as isize {
                patch[i] = signal[idx as usize];
            }
        }
        patch
    } else {
        let start = center - half_size;
        let end = start + size;
        signal.slice(s![start..end]).to_owned()
    }
}

/// Helper function to extract a patch from a 2D image
fn extract_patch_2d(
    image: &Array2<f64>,
    center_i: usize,
    center_j: usize,
    size: usize,
) -> Array2<f64> {
    let half_size = size / 2;

    // Check bounds to avoid panic
    if center_i < half_size
        || center_j < half_size
        || center_i + half_size >= image.dim().0
        || center_j + half_size >= image.dim().1
    {
        // Handle boundary case
        let mut patch = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                let idx_i = center_i as isize - half_size as isize + i as isize;
                let idx_j = center_j as isize - half_size as isize + j as isize;

                if idx_i >= 0
                    && idx_j >= 0
                    && idx_i < image.dim().0 as isize
                    && idx_j < image.dim().1 as isize
                {
                    patch[[i, j]] = image[[idx_i as usize, idx_j as usize]];
                }
            }
        }
        patch
    } else {
        let start_i = center_i - half_size;
        let end_i = start_i + size;
        let start_j = center_j - half_size;
        let end_j = start_j + size;

        image.slice(s![start_i..end_i, start_j..end_j]).to_owned()
    }
}

/// Helper function to compute distance between two 1D patches
fn compute_patch_distance(patch1: &Array1<f64>, patch2: &Array1<f64>) -> f64 {
    // Ensure patches have the same size
    if patch1.len() != patch2.len() {
        return f64::INFINITY;
    }

    // Compute mean square difference
    let n = patch1.len();
    let mut sum_diff_sq = 0.0;

    for i in 0..n {
        let diff = patch1[i] - patch2[i];
        sum_diff_sq += diff * diff;
    }

    sum_diff_sq / n as f64
}

/// Helper function to compute weighted distance between two 2D patches
fn compute_patch_distance_2d(
    patch1: &Array2<f64>,
    patch2: &Array2<f64>,
    weights: Option<&Array2<f64>>,
) -> f64 {
    // Ensure patches have the same size
    if patch1.dim() != patch2.dim() {
        return f64::INFINITY;
    }

    let (height, width) = patch1.dim();
    let n = height * width;

    // Compute weighted mean square difference
    let mut sum_diff_sq = 0.0;
    let mut sum_weights = 0.0;

    match weights {
        Some(w) => {
            // Weighted distance
            for i in 0..height {
                for j in 0..width {
                    let diff = patch1[[i, j]] - patch2[[i, j]];
                    let weight = w[[i, j]];
                    sum_diff_sq += weight * diff * diff;
                    sum_weights += weight;
                }
            }

            if sum_weights > 0.0 {
                sum_diff_sq / sum_weights
            } else {
                0.0
            }
        }
        None => {
            // Unweighted distance
            for i in 0..height {
                for j in 0..width {
                    let diff = patch1[[i, j]] - patch2[[i, j]];
                    sum_diff_sq += diff * diff;
                }
            }

            sum_diff_sq / n as f64
        }
    }
}

/// Helper function to estimate noise standard deviation in a 1D signal
fn estimate_noise_sigma(signal: &Array1<f64>) -> f64 {
    // Use median absolute deviation method for robust estimation

    // Compute first-order differences
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }

    let mut diffs = Vec::with_capacity(n - 1);
    for i in 1..n {
        diffs.push((signal[i] - signal[i - 1]).abs());
    }

    // Sort differences and find median
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if diffs.len() % 2 == 0 {
        (diffs[diffs.len() / 2 - 1] + diffs[diffs.len() / 2]) / 2.0
    } else {
        diffs[diffs.len() / 2]
    };

    // Scale factor for Gaussian noise estimation from absolute differences
    // (dividing by sqrt(2) because we used first-order differences)
    median / 0.6745 / std::f64::consts::SQRT_2
}

/// Helper function to estimate noise standard deviation in a 2D image
fn estimate_noise_sigma_2d(image: &Array2<f64>) -> f64 {
    // Use median absolute deviation of Laplacian for robust estimation
    let (height, width) = image.dim();
    if height < 3 || width < 3 {
        return 0.0;
    }

    // Apply discrete Laplacian operator
    let mut laplacian = Vec::new();

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            // 4-connected Laplacian
            let lap_val = 4.0 * image[[i, j]]
                - image[[i + 1, j]]
                - image[[i - 1, j]]
                - image[[i, j + 1]]
                - image[[i, j - 1]];
            laplacian.push(lap_val.abs());
        }
    }

    // Sort values and find median
    laplacian.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if laplacian.len() % 2 == 0 {
        (laplacian[laplacian.len() / 2 - 1] + laplacian[laplacian.len() / 2]) / 2.0
    } else {
        laplacian[laplacian.len() / 2]
    };

    // Scale factor for Gaussian noise estimation from Laplacian
    median / 0.6745 / std::f64::consts::SQRT_2
}

/// Helper function to downsample a 2D image
fn downsample_image(
    image: &Array2<f64>,
    new_height: usize,
    new_width: usize,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    if new_height > height || new_width > width {
        return Err(SignalError::ValueError(
            "Downsampled dimensions must be smaller than original".to_string(),
        ));
    }

    // Initialize downsampled image
    let mut downsampled = Array2::zeros((new_height, new_width));

    // Compute scale factors
    let h_scale = height as f64 / new_height as f64;
    let w_scale = width as f64 / new_width as f64;

    // Perform area-based downsampling
    for i in 0..new_height {
        for j in 0..new_width {
            // Determine source area
            let src_i_start = (i as f64 * h_scale).floor() as usize;
            let src_i_end = ((i + 1) as f64 * h_scale).ceil() as usize;
            let src_j_start = (j as f64 * w_scale).floor() as usize;
            let src_j_end = ((j + 1) as f64 * w_scale).ceil() as usize;

            // Compute average over area
            let mut sum = 0.0;
            let mut count = 0;

            for si in src_i_start..cmp::min(src_i_end, height) {
                for sj in src_j_start..cmp::min(src_j_end, width) {
                    sum += image[[si, sj]];
                    count += 1;
                }
            }

            if count > 0 {
                downsampled[[i, j]] = sum / count as f64;
            }
        }
    }

    Ok(downsampled)
}

/// Helper function to upsample a 2D image
fn upsample_image(
    image: &Array2<f64>,
    new_height: usize,
    new_width: usize,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    if new_height < height || new_width < width {
        return Err(SignalError::ValueError(
            "Upsampled dimensions must be larger than original".to_string(),
        ));
    }

    // Initialize upsampled image
    let mut upsampled = Array2::zeros((new_height, new_width));

    // Compute scale factors
    let h_scale = (height - 1) as f64 / (new_height - 1) as f64;
    let w_scale = (width - 1) as f64 / (new_width - 1) as f64;

    // Perform bilinear interpolation
    for i in 0..new_height {
        for j in 0..new_width {
            // Compute source coordinates
            let src_i = i as f64 * h_scale;
            let src_j = j as f64 * w_scale;

            // Get integer and fractional parts
            let i0 = src_i.floor() as usize;
            let j0 = src_j.floor() as usize;
            let di = src_i - i0 as f64;
            let dj = src_j - j0 as f64;

            // Handle boundary cases
            let i1 = cmp::min(i0 + 1, height - 1);
            let j1 = cmp::min(j0 + 1, width - 1);

            // Bilinear interpolation
            upsampled[[i, j]] = (1.0 - di) * (1.0 - dj) * image[[i0, j0]]
                + di * (1.0 - dj) * image[[i1, j0]]
                + (1.0 - di) * dj * image[[i0, j1]]
                + di * dj * image[[i1, j1]];
        }
    }

    Ok(upsampled)
}
