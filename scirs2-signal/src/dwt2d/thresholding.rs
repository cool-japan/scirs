//! 2D DWT Thresholding Functions
//!
//! This module provides comprehensive thresholding capabilities for wavelet coefficients,
//! including various thresholding methods and both single-level and multi-level operations.
//! These functions are essential for denoising, compression, and feature extraction applications.

use super::types::{Dwt2dResult, ThresholdMethod};
use super::simd::simd_threshold_coefficients;
use scirs2_core::ndarray::Array2;

/// Apply thresholding to wavelet coefficients for denoising or compression.
///
/// This function applies a threshold to the detail coefficients of a wavelet decomposition.
/// It is commonly used for denoising (removing low-amplitude noise) and compression
/// (removing less significant coefficients). Only detail coefficients are thresholded;
/// approximation coefficients are left unchanged.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to threshold (will be modified in-place)
/// * `threshold` - The threshold value (coefficients with absolute value below this will be modified)
/// * `method` - The thresholding method to apply (Hard, Soft, or Garrote)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, threshold_dwt2d, ThresholdMethod};
///
/// // Create a sample "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose with wavelet transform
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::DB(4), None).unwrap();
///
/// // Apply hard thresholding to detail coefficients
/// threshold_dwt2d(&mut decomposition, 1.0, ThresholdMethod::Hard);
///
/// // Reconstruct the denoised image
/// let denoised = dwt2d_reconstruct(&decomposition, Wavelet::DB(4), None).unwrap();
/// ```
pub fn threshold_dwt2d(decomposition: &mut Dwt2dResult, threshold: f64, method: ThresholdMethod) {
    // Apply SIMD-optimized thresholding to detail coefficients
    // Note: ndarray's as_slice_mut() gives us direct access to the underlying data
    if let Some(h_slice) = decomposition.detail_h.as_slice_mut() {
        simd_threshold_coefficients(h_slice, threshold, method);
    } else {
        // Fallback for non-contiguous arrays
        for h in decomposition.detail_h.iter_mut() {
            *h = apply_threshold(*h, threshold, method);
        }
    }

    if let Some(v_slice) = decomposition.detail_v.as_slice_mut() {
        simd_threshold_coefficients(v_slice, threshold, method);
    } else {
        // Fallback for non-contiguous arrays
        for v in decomposition.detail_v.iter_mut() {
            *v = apply_threshold(*v, threshold, method);
        }
    }

    if let Some(d_slice) = decomposition.detail_d.as_slice_mut() {
        simd_threshold_coefficients(d_slice, threshold, method);
    } else {
        // Fallback for non-contiguous arrays
        for d in decomposition.detail_d.iter_mut() {
            *d = apply_threshold(*d, threshold, method);
        }
    }
}

/// Apply thresholding to multi-level wavelet coefficients.
///
/// Similar to `threshold_dwt2d`, but operates on a multi-level decomposition from `wavedec2`.
/// This allows for level-dependent thresholding, which can be more effective for certain
/// applications.
///
/// # Arguments
///
/// * `coeffs` - The multi-level wavelet decomposition to threshold (modified in-place)
/// * `threshold` - The threshold value, or vector of threshold values (one per level)
/// * `method` - The thresholding method to apply
///
/// # Examples
///
/// Using a different threshold for each level:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2, threshold_wavedec2, ThresholdMethod};
///
/// // Create a sample image
/// let mut data = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Multi-level decomposition (3 levels)
/// let mut coeffs = wavedec2(&data, Wavelet::Haar, 3, None).unwrap();
///
/// // Apply different thresholds for each level (higher thresholds for finer details)
/// let thresholds = vec![5.0, 10.0, 15.0];
/// threshold_wavedec2(&mut coeffs, &thresholds, ThresholdMethod::Soft);
///
/// // Reconstruct from thresholded coefficients
/// let result = waverec2(&coeffs, Wavelet::Haar, None).unwrap();
/// ```
pub fn threshold_wavedec2(coeffs: &mut [Dwt2dResult], threshold: &[f64], method: ThresholdMethod) {
    for (i, level) in coeffs.iter_mut().enumerate() {
        // Get the appropriate threshold for this level
        let level_threshold = if i < threshold.len() {
            threshold[i]
        } else {
            // If not enough thresholds provided, use the last one
            *threshold.last().unwrap_or(&0.0)
        };

        // Apply thresholding to this level
        threshold_dwt2d(level, level_threshold, method);
    }
}

/// Helper function to apply a threshold to a single coefficient.
pub fn apply_threshold(x: f64, threshold: f64, method: ThresholdMethod) -> f64 {
    let abs_x = x.abs();

    // If coefficient is below threshold, always zero it out
    if abs_x <= threshold {
        return 0.0;
    }

    // Apply the appropriate thresholding method
    match method {
        ThresholdMethod::Hard => x, // Hard thresholding keeps the value unchanged
        ThresholdMethod::Soft => {
            // Soft thresholding shrinks the value toward zero by the threshold amount
            x.signum() * (abs_x - threshold)
        }
        ThresholdMethod::Garrote => {
            // Non-linear garrote thresholding
            x * (1.0 - (threshold * threshold) / (x * x))
        }
    }
}

/// Apply adaptive thresholding to a 2D array of coefficients.
///
/// This function applies coefficient-specific thresholding where the threshold
/// can vary spatially across the image. This is useful for advanced denoising
/// applications where noise characteristics vary locally.
///
/// # Arguments
///
/// * `data` - The 2D array to threshold (modified in-place)
/// * `threshold` - The threshold value to apply uniformly
/// * `method` - The thresholding method to use
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::{apply_adaptive_thresholding, ThresholdMethod};
///
/// // Create a sample coefficient array
/// let mut coeffs = Array2::from_shape_fn((4, 4), |(i, j)| {
///     (i as f64 + j as f64) * 0.5 - 1.0
/// });
///
/// // Apply adaptive thresholding
/// apply_adaptive_thresholding(&mut coeffs, 0.8, ThresholdMethod::Soft);
/// ```
pub fn apply_adaptive_thresholding(data: &mut Array2<f64>, threshold: f64, method: ThresholdMethod) {
    // For now, apply uniform thresholding
    // In future versions, this could implement spatially-varying thresholds
    for coeff in data.iter_mut() {
        *coeff = apply_threshold(*coeff, threshold, method);
    }
}

/// Estimate noise variance from the finest detail coefficients.
///
/// This function estimates the noise standard deviation from the diagonal detail coefficients
/// using the robust median absolute deviation (MAD) estimator. This is commonly used
/// in wavelet denoising applications.
///
/// # Arguments
///
/// * `decomp` - The wavelet decomposition containing detail coefficients
///
/// # Returns
///
/// * The estimated noise standard deviation
///
/// # Algorithm
///
/// The noise variance is estimated using:
/// σ = median(|HH|) / 0.6745
///
/// where HH represents the diagonal detail coefficients and 0.6745 is the 75th percentile
/// of the standard normal distribution, making this a robust estimator.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, estimate_noise_variance};
///
/// // Create a noisy image
/// let mut noisy_image = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         noisy_image[[i, j]] = (i as f64).sin() + 0.1 * ((i * j) as f64).cos();
///     }
/// }
///
/// let decomp = dwt2d_decompose(&noisy_image, Wavelet::DB(4), None).unwrap();
/// let noise_std = estimate_noise_variance(&decomp);
/// ```
pub fn estimate_noise_variance(decomp: &Dwt2dResult) -> f64 {
    // Use the diagonal detail coefficients (HH band) for noise estimation
    // as they typically contain mostly noise in natural images
    let mut hh_coeffs: Vec<f64> = decomp.detail_d.iter().map(|&x| x.abs()).collect();

    if hh_coeffs.is_empty() {
        return 0.0;
    }

    // Sort coefficients for median calculation
    hh_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate median
    let median = if hh_coeffs.len() % 2 == 0 {
        let mid = hh_coeffs.len() / 2;
        (hh_coeffs[mid - 1] + hh_coeffs[mid]) / 2.0
    } else {
        hh_coeffs[hh_coeffs.len() / 2]
    };

    // Convert median absolute deviation to standard deviation estimate
    // 0.6745 is the 75th percentile of the standard normal distribution
    median / 0.6745
}

/// Calculate the compression ratio between original and compressed coefficients.
///
/// This function computes the compression ratio by comparing the number of
/// non-zero coefficients before and after thresholding.
///
/// # Arguments
///
/// * `original` - The original decomposition before thresholding
/// * `compressed` - The decomposition after thresholding
///
/// # Returns
///
/// * The compression ratio (original_nonzeros / compressed_nonzeros)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, threshold_dwt2d, calculate_compression_ratio, ThresholdMethod};
///
/// // Create test data
/// let data = Array2::from_shape_fn((8, 8), |(i, j)| (i * j) as f64);
/// let original = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
/// let mut compressed = original.clone();
///
/// // Apply thresholding
/// threshold_dwt2d(&mut compressed, 2.0, ThresholdMethod::Hard);
///
/// // Calculate compression ratio
/// let ratio = calculate_compression_ratio(&original, &compressed);
/// ```
pub fn calculate_compression_ratio(original: &Dwt2dResult, compressed: &Dwt2dResult) -> f64 {
    let original_nonzeros = count_nonzeros_in_decomposition(original);
    let compressed_nonzeros = count_nonzeros_in_decomposition(compressed);

    if compressed_nonzeros == 0 {
        f64::INFINITY
    } else {
        original_nonzeros as f64 / compressed_nonzeros as f64
    }
}

/// Count non-zero coefficients in a wavelet decomposition.
///
/// This helper function counts the number of non-zero coefficients in all subbands
/// of a wavelet decomposition.
///
/// # Arguments
///
/// * `decomp` - The wavelet decomposition to analyze
///
/// # Returns
///
/// * The total number of non-zero coefficients
fn count_nonzeros_in_decomposition(decomp: &Dwt2dResult) -> usize {
    let approx_count = decomp.approx.iter().filter(|&&x| x != 0.0).count();
    let detail_h_count = decomp.detail_h.iter().filter(|&&x| x != 0.0).count();
    let detail_v_count = decomp.detail_v.iter().filter(|&&x| x != 0.0).count();
    let detail_d_count = decomp.detail_d.iter().filter(|&&x| x != 0.0).count();

    approx_count + detail_h_count + detail_v_count + detail_d_count
}