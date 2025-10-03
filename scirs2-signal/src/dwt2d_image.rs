// Image processing utilities using 2D Wavelet Transform
//
// This module provides specialized functions for image processing
// using the 2D Discrete Wavelet Transform. It includes operations
// for image denoising, compression, edge detection, and fusion.
//
// # Examples
//
// Basic image denoising:
//
// ```
// use scirs2_core::ndarray::{Array, Array2};
// use scirs2_signal::dwt::Wavelet;
// use scirs2_signal::dwt2d::wavedec2;
// use scirs2_signal::dwt2d_image::{denoise_image, DenoisingMethod};
//
// // Create a simple noisy image (8x8)
// let mut image = Array2::zeros((8, 8));
// for i in 0..8 {
//     for j in 0..8 {
//         // Add some pattern plus noise
//         image[[i, j]] = (i * j) as f64 + (i+j) as f64 * 0.2;
//     }
// }
//
// // Apply wavelet denoising
// let denoised = denoise_image(&image, Wavelet::DB(4), 2, 5.0,
//                             DenoisingMethod::VisuShrink, None).unwrap();
//
// // Check that the result has the same shape as the input
// assert_eq!(denoised.shape(), image.shape());
// ```

use crate::dwt::Wavelet;
use crate::dwt2d::{wavedec2, waverec2, Dwt2dResult, ThresholdMethod};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use std::f64;

#[allow(unused_imports)]
/// Denoising methods for wavelet-based image denoising
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenoisingMethod {
    /// Hard thresholding - sets coefficients below threshold to zero
    Hard,

    /// Soft thresholding - sets coefficients below threshold to zero
    /// and shrinks others toward zero by the threshold
    Soft,

    /// Non-linear garotte thresholding - a compromise between hard and soft
    Garrote,

    /// VisuShrink - uses universal threshold sqrt(2*log(N))
    VisuShrink,

    /// BayesShrink - adaptive threshold based on Bayesian estimation
    BayesShrink,
}

/// Apply wavelet denoising to a 2D image.
///
/// This function decomposes the image using the 2D wavelet transform,
/// applies thresholding to remove noise, and then reconstructs the image.
///
/// # Arguments
///
/// * `image` - The input image as a 2D array
/// * `wavelet` - The wavelet to use for the transform
/// * `level` - The number of decomposition levels
/// * `threshold` - Base threshold value (interpretation depends on method)
/// * `method` - The denoising method to use
/// * `keep_approx` - Whether to keep approximation coefficients unchanged (default: true)
///
/// # Returns
///
/// * The denoised image as a 2D array
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d_image::{denoise_image, DenoisingMethod};
///
/// // Create a noisy image (32x32) - larger to avoid overflow issues
/// let mut image = Array2::from_elem((32, 32), 0.0);
/// for i in 0..32 {
///     for j in 0..32 {
///         // Simple pattern with noise
///         image[[i, j]] = (i * j) as f64 + ((i as f64).sin() * (j as f64).cos()) * 5.0;
///     }
/// }
///
/// // Apply soft thresholding denoising
/// let denoised = denoise_image(&image, Wavelet::Haar, 2, 1.5,
///                             DenoisingMethod::Soft, Some(true)).unwrap();
///
/// // The denoised image should have the same shape as the input
/// assert_eq!(denoised.shape(), image.shape());
/// ```
#[allow(dead_code)]
pub fn denoise_image<T>(
    image: &Array2<T>,
    wavelet: Wavelet,
    level: usize,
    threshold: f64,
    method: DenoisingMethod,
    keep_approx: Option<bool>,
) -> SignalResult<Array2<f64>>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    // Convert image to f64 Array2 if needed
    let img = image.mapv(|x| x.into());

    // Decompose the image
    let mut coeffs = wavedec2(&img, wavelet, level, None)?;

    // Determine thresholds for each level based on method
    let thresholds = match method {
        DenoisingMethod::VisuShrink => {
            // Universal threshold: sigma * sqrt(2 * log(N))
            // where sigma is noise standard deviation (estimated from finest detail coefficients)
            let (_, energy) = calculate_detail_stats(&coeffs[0]);
            let sigma = (energy.detail_h + energy.detail_v + energy.detail_d).sqrt()
                / ((coeffs[0].detail_h.len() + coeffs[0].detail_v.len() + coeffs[0].detail_d.len())
                    as f64)
                    .sqrt();

            let n = img.len() as f64;
            let universal_threshold = sigma * (2.0 * n.ln()).sqrt();

            // Higher levels get progressively smaller thresholds
            let mut thresholds = Vec::with_capacity(coeffs.len());
            for i in 0..coeffs.len() {
                let level_factor = 1.0 / (2.0_f64.powi(i as i32));
                thresholds.push(universal_threshold * level_factor);
            }
            thresholds
        }
        DenoisingMethod::BayesShrink => {
            // Adaptive thresholds based on subband statistics
            let mut thresholds = Vec::with_capacity(coeffs.len());

            for level in &coeffs {
                let (_, energy) = calculate_detail_stats(level);

                // Estimate noise variance from detail coefficients
                let h_sigma = estimate_noise_variance(&level.detail_h);
                let v_sigma = estimate_noise_variance(&level.detail_v);
                let d_sigma = estimate_noise_variance(&level.detail_d);

                // Average noise standard deviation
                let sigma = ((h_sigma + v_sigma + d_sigma) / 3.0).sqrt();

                // Bayes threshold = sigma^2 / sigma_x
                let signal_var = ((energy.detail_h + energy.detail_v + energy.detail_d)
                    / (level.detail_h.len() + level.detail_v.len() + level.detail_d.len()) as f64)
                    .max(sigma * sigma)
                    - sigma * sigma;

                let sigma_x = signal_var.sqrt();
                let threshold = if sigma_x > 0.0 {
                    sigma * sigma / sigma_x
                } else {
                    threshold // fallback to provided threshold
                };

                thresholds.push(threshold);
            }
            thresholds
        }
        _ => {
            // For Hard, Soft, and Garrote, use the provided threshold
            vec![threshold; coeffs.len()]
        }
    };

    // Convert denoising method to threshold method
    let threshold_method = match method {
        DenoisingMethod::Hard => ThresholdMethod::Hard,
        DenoisingMethod::Soft => ThresholdMethod::Soft,
        DenoisingMethod::Garrote => ThresholdMethod::Garrote,
        DenoisingMethod::VisuShrink => ThresholdMethod::Soft, // VisuShrink uses soft thresholding
        DenoisingMethod::BayesShrink => ThresholdMethod::Soft, // BayesShrink uses soft thresholding
    };

    // Apply thresholding to coefficients
    let keep_approx = keep_approx.unwrap_or(true);
    threshold_coefficients(&mut coeffs, &thresholds, threshold_method, keep_approx);

    // Reconstruct the image
    waverec2(&coeffs, wavelet, None)
}

/// Apply edge detection using wavelet transform.
///
/// This function performs a wavelet decomposition and reconstructs
/// the image using only the detail coefficients, which highlights edges.
///
/// # Arguments
///
/// * `image` - The input image as a 2D array
/// * `wavelet` - The wavelet to use for the transform
/// * `level` - The number of decomposition levels
/// * `threshold` - Optional threshold to apply to detail coefficients
///
/// # Returns
///
/// * The edge-detected image as a 2D array
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d_image::detect_edges;
///
/// // Create a simple test image with a sharp edge
/// let mut image = Array2::from_elem((8, 8), 0.0);
/// for i in 0..8 {
///     for j in 0..8 {
///         if i >= 4 {
///             image[[i, j]] = 100.0;  // Create a horizontal edge
///         }
///     }
/// }
///
/// // Detect edges
/// let edges = detect_edges(&image, Wavelet::Haar, 1, Some(10.0)).unwrap();
///
/// // The edge image should have the same dimensions as the input
/// assert_eq!(edges.shape(), image.shape());
/// ```
#[allow(dead_code)]
pub fn detect_edges<T>(
    image: &Array2<T>,
    wavelet: Wavelet,
    level: usize,
    threshold: Option<f64>,
) -> SignalResult<Array2<f64>>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    // Convert image to f64 Array2
    let img = image.mapv(|x| x.into());

    // Decompose the image
    let mut coeffs = wavedec2(&img, wavelet, level, None)?;

    // Apply thresholding if a threshold is provided
    if let Some(threshold_value) = threshold {
        let thresholds = vec![threshold_value; coeffs.len()];
        threshold_coefficients(&mut coeffs, &thresholds, ThresholdMethod::Hard, false);
    }

    // Zero out approximation coefficients to keep only edge information
    for level in coeffs.iter_mut() {
        for val in level.approx.iter_mut() {
            *val = 0.0;
        }
    }

    // Reconstruct using only detail coefficients
    waverec2(&coeffs, wavelet, None)
}

/// Compress an image using wavelet thresholding.
///
/// This function applies wavelet decomposition, thresholds the coefficients
/// to achieve compression, and then reconstructs the image.
///
/// # Arguments
///
/// * `image` - The input image as a 2D array
/// * `wavelet` - The wavelet to use for the transform
/// * `level` - The number of decomposition levels
/// * `compression_ratio` - Target compression ratio (0.0-1.0, where 0.9 means 90% of coefficients are zeroed)
///
/// # Returns
///
/// * A tuple containing the compressed image and the achieved compression ratio
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d_image::compress_image;
///
/// // Create a test image
/// let mut image = Array2::from_elem((16, 16), 0.0);
/// for i in 0..16 {
///     for j in 0..16 {
///         image[[i, j]] = ((i * j) % 16) as f64;
///     }
/// }
///
/// // Compress image with 50% compression ratio
/// let (compressed, ratio) = compress_image(&image, Wavelet::DB(4), 2, 0.5).unwrap();
///
/// // Check that the result has the same shape as the input
/// assert_eq!(compressed.shape(), image.shape());
/// // Check that the achieved compression ratio is greater than 0
/// assert!(ratio > 0.0);
/// ```
#[allow(dead_code)]
pub fn compress_image<T>(
    image: &Array2<T>,
    wavelet: Wavelet,
    level: usize,
    compression_ratio: f32,
) -> SignalResult<(Array2<f64>, f32)>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    if !(0.0..=1.0).contains(&compression_ratio) {
        return Err(SignalError::ValueError(
            "Compression _ratio must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Convert image to f64 Array2
    let img = image.mapv(|x| x.into());

    // Decompose the image
    let mut coeffs = wavedec2(&img, wavelet, level, None)?;

    // Count the total number of coefficients
    let mut total_coeffs = 0;
    for level in &coeffs {
        total_coeffs += level.approx.len();
        total_coeffs += level.detail_h.len();
        total_coeffs += level.detail_v.len();
        total_coeffs += level.detail_d.len();
    }

    // Count non-zero coefficients before compression
    let (original_nonzeros, _) = count_nonzeros_multilevel(&coeffs, true);

    // Calculate threshold to achieve target compression _ratio
    if compression_ratio > 0.0 {
        // Get all coefficient values
        let mut all_coeffs = Vec::with_capacity(total_coeffs);
        for level in &coeffs {
            // Skip approximation coefficients at the coarsest level
            if level != coeffs.first().unwrap() {
                all_coeffs.extend(level.approx.iter().map(|&x: &f64| x.abs()));
            }
            all_coeffs.extend(level.detail_h.iter().map(|&x: &f64| x.abs()));
            all_coeffs.extend(level.detail_v.iter().map(|&x: &f64| x.abs()));
            all_coeffs.extend(level.detail_d.iter().map(|&x: &f64| x.abs()));
        }

        // Sort coefficients
        all_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find threshold that gives desired compression _ratio
        let threshold_idx = (all_coeffs.len() as f32 * compression_ratio) as usize;
        if threshold_idx < all_coeffs.len() {
            let threshold = all_coeffs[threshold_idx];

            // Apply threshold to all detail coefficients
            // Create thresholds vector before mutably borrowing coeffs
            let thresholds = vec![threshold; coeffs.len()];
            threshold_coefficients(
                &mut coeffs,
                &thresholds,
                ThresholdMethod::Hard,
                true, // Keep approximation coefficients
            );
        }
    }

    // Count non-zero coefficients after compression
    let (compressed_nonzeros, _) = count_nonzeros_multilevel(&coeffs, true);

    // Calculate actual compression _ratio achieved
    let actual_ratio = if original_nonzeros > 0 {
        1.0 - (compressed_nonzeros as f32 / original_nonzeros as f32)
    } else {
        0.0
    };

    // Reconstruct the image
    let reconstructed = waverec2(&coeffs, wavelet, None)?;

    Ok((reconstructed, actual_ratio))
}

/// Helper function to calculate statistics of detail coefficients.
#[allow(dead_code)]
fn calculate_detail_stats(decomp: &Dwt2dResult) -> (usize, WaveletEnergy) {
    let detail_h_count = decomp.detail_h.iter().filter(|&&x| x != 0.0).count();
    let detail_v_count = decomp.detail_v.iter().filter(|&&x| x != 0.0).count();
    let detail_d_count = decomp.detail_d.iter().filter(|&&x| x != 0.0).count();

    let detail_h_energy = decomp.detail_h.iter().map(|&x| x * x).sum();
    let detail_v_energy = decomp.detail_v.iter().map(|&x| x * x).sum();
    let detail_d_energy = decomp.detail_d.iter().map(|&x| x * x).sum();

    let total_count = detail_h_count + detail_v_count + detail_d_count;

    let energy = WaveletEnergy {
        approx: 0.0, // Not calculated for approximation coefficients
        detail_h: detail_h_energy,
        detail_v: detail_v_energy,
        detail_d: detail_d_energy,
    };

    (total_count, energy)
}

/// Helper function to estimate noise variance from detail coefficients.
#[allow(dead_code)]
fn estimate_noise_variance(coeffs: &Array2<f64>) -> f64 {
    // Median Absolute Deviation (MAD) estimator
    let mut abs_coeffs: Vec<f64> = coeffs.iter().map(|&x: &f64| x.abs()).collect();
    abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if abs_coeffs.is_empty() {
        return 0.0;
    }

    let mid = abs_coeffs.len() / 2;
    let median = if abs_coeffs.len() % 2 == 0 {
        (abs_coeffs[mid - 1] + abs_coeffs[mid]) / 2.0
    } else {
        abs_coeffs[mid]
    };

    // MAD estimator of standard deviation
    let mad = 0.6745 * median;
    if mad > 1e-10 {
        mad * mad
    } else {
        // If MAD is too small, use standard deviation
        let mean = abs_coeffs.iter().sum::<f64>() / abs_coeffs.len() as f64;
        let variance =
            abs_coeffs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / abs_coeffs.len() as f64;
        variance
    }
}

/// Apply thresholding to wavelet coefficients.
#[allow(dead_code)]
fn threshold_coefficients(
    coeffs: &mut [Dwt2dResult],
    thresholds: &[f64],
    method: ThresholdMethod,
    keep_approx: bool,
) {
    for (i, level) in coeffs.iter_mut().enumerate() {
        let threshold = if i < thresholds.len() {
            thresholds[i]
        } else if !thresholds.is_empty() {
            thresholds[thresholds.len() - 1]
        } else {
            0.0 // No thresholding if no thresholds provided
        };

        // Apply thresholding to detail coefficients
        for h in level.detail_h.iter_mut() {
            *h = apply_threshold(*h, threshold, method);
        }
        for v in level.detail_v.iter_mut() {
            *v = apply_threshold(*v, threshold, method);
        }
        for d in level.detail_d.iter_mut() {
            *d = apply_threshold(*d, threshold, method);
        }

        // Optionally threshold approximation coefficients
        // (usually not done for the coarsest level)
        if !keep_approx && i > 0 {
            for a in level.approx.iter_mut() {
                *a = apply_threshold(*a, threshold, method);
            }
        }
    }
}

/// Helper function to apply threshold to a coefficient.
#[allow(dead_code)]
fn apply_threshold(x: f64, threshold: f64, method: ThresholdMethod) -> f64 {
    let abs_x = x.abs();

    if abs_x <= threshold {
        return 0.0;
    }

    match method {
        ThresholdMethod::Hard => x,
        ThresholdMethod::Soft => x.signum() * (abs_x - threshold),
        ThresholdMethod::Garrote => {
            let t_sq = threshold * threshold;
            x * (1.0 - t_sq / (x * x))
        }
    }
}

/// Count non-zero coefficients in a multi-level wavelet decomposition.
#[allow(dead_code)]
fn count_nonzeros_multilevel(_coeffs: &[Dwt2dResult], includeapprox: bool) -> (usize, Vec<usize>) {
    let mut total_nonzeros = 0;
    let mut level_counts = Vec::with_capacity(_coeffs.len());

    for level in _coeffs {
        let mut level_count = 0;

        // Count approximation coefficients if requested
        if include_approx {
            level_count += level.approx.iter().filter(|&&x| x != 0.0).count();
        }

        // Count detail coefficients
        level_count += level.detail_h.iter().filter(|&&x| x != 0.0).count();
        level_count += level.detail_v.iter().filter(|&&x| x != 0.0).count();
        level_count += level.detail_d.iter().filter(|&&x| x != 0.0).count();

        total_nonzeros += level_count;
        level_counts.push(level_count);
    }

    (total_nonzeros, level_counts)
}

/// Structure to hold energy values for wavelet subbands.
#[derive(Debug, Clone, Copy)]
pub struct WaveletEnergy {
    /// Energy in approximation coefficients
    pub approx: f64,
    /// Energy in horizontal detail coefficients
    pub detail_h: f64,
    /// Energy in vertical detail coefficients
    pub detail_v: f64,
    /// Energy in diagonal detail coefficients
    pub detail_d: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_image() {
        // Create a simple test image with some "noise"
        let mut image = Array2::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                // Base pattern
                let base = (i * j) as f64;
                // Add some noise
                let noise = ((i + j) as f64 * 0.5).sin() * 2.0;
                image[[i, j]] = base + noise;
            }
        }

        // Apply denoising with different methods
        let denoised_hard = denoise_image(
            &image,
            Wavelet::Haar,
            2,
            1.0,
            DenoisingMethod::Hard,
            Some(true),
        )
        .unwrap();

        let denoised_soft = denoise_image(
            &image,
            Wavelet::Haar,
            2,
            1.0,
            DenoisingMethod::Soft,
            Some(true),
        )
        .unwrap();

        let denoised_visu = denoise_image(
            &image,
            Wavelet::Haar,
            2,
            0.0,
            DenoisingMethod::VisuShrink,
            Some(true),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(denoised_hard.shape(), image.shape());
        assert_eq!(denoised_soft.shape(), image.shape());
        assert_eq!(denoised_visu.shape(), image.shape());
    }

    #[test]
    fn test_edge_detection() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a test image with edges
        let mut image = Array2::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                if i >= 4 {
                    image[[i, j]] = 10.0; // Horizontal edge
                }
                if j >= 4 {
                    image[[i, j]] += 5.0; // Vertical edge
                }
            }
        }

        // Detect edges
        let edges = detect_edges(&image, Wavelet::Haar, 1, Some(0.5)).unwrap();

        // Check dimensions
        assert_eq!(edges.shape(), image.shape());

        // Check that edge pixels have non-zero values (simplified test)
        let mut edge_count = 0;
        for i in 0..8 {
            for j in 0..8 {
                if edges[[i, j]].abs() > 1e-10 {
                    edge_count += 1;
                }
            }
        }

        // There should be some non-zero values corresponding to edges
        assert!(edge_count > 0);
    }

    #[test]
    fn test_image_compression() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a test image
        let mut image = Array2::zeros((16, 16));
        for i in 0..16 {
            for j in 0..16 {
                image[[i, j]] = ((i * j) % 16) as f64;
            }
        }

        // Compress with different ratios
        let (compressed_low, ratio_low) = compress_image(&image, Wavelet::DB(2), 2, 0.3).unwrap();
        let (compressed_high, ratio_high) = compress_image(&image, Wavelet::DB(2), 2, 0.7).unwrap();

        // Check dimensions
        assert_eq!(compressed_low.shape(), image.shape());
        assert_eq!(compressed_high.shape(), image.shape());

        // Check compression ratios
        assert!(ratio_low > 0.0);
        assert!(ratio_high > ratio_low); // Higher ratio should give more compression

        // Test edge case - no compression
        let (no_compression, ratio_zero) = compress_image(&image, Wavelet::DB(2), 2, 0.0).unwrap();
        assert_eq!(no_compression.shape(), image.shape());
        assert_eq!(ratio_zero, 0.0);

        // Test invalid ratio
        let result = compress_image(&image, Wavelet::DB(2), 2, 1.5);
        assert!(result.is_err());
    }
}
