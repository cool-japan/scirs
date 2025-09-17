//! 2D DWT Utility Functions
//!
//! This module provides utility functions for 2D wavelet transform operations,
//! including image quality metrics, noise analysis, and denoising capabilities.

use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use super::types::{Dwt2dResult, Dwt2dConfig, ThresholdMethod};
use super::simd::PlatformCapabilities;
use super::thresholding::{apply_adaptive_thresholding, estimate_noise_variance};
use ndarray::Array2;
use scirs2_core::validation::{check_positive, check_finite};

/// Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
///
/// PSNR is a widely used metric for assessing image quality, particularly
/// useful for evaluating reconstruction accuracy and compression quality.
///
/// # Arguments
///
/// * `original` - The original (reference) image
/// * `reconstructed` - The processed/reconstructed image to evaluate
///
/// # Returns
///
/// * PSNR value in decibels (dB). Higher values indicate better quality.
///   - Values > 40 dB typically indicate very good quality
///   - Values 30-40 dB indicate good quality
///   - Values < 30 dB may indicate noticeable degradation
///
/// # Errors
///
/// Returns an error if:
/// * Arrays have different shapes
/// * Arrays are empty
/// * All pixel values are identical (infinite PSNR case)
///
/// # Formula
///
/// PSNR = 20 * log10(MAX_VALUE / RMSE)
///
/// where MAX_VALUE is the maximum possible pixel value (computed from data range)
/// and RMSE is the root mean square error between the images.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::calculate_psnr;
///
/// // Create original and slightly noisy version
/// let original = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
/// let mut noisy = original.clone();
/// noisy[[0, 0]] += 0.1; // Add small noise
///
/// let psnr = calculate_psnr(&original, &noisy).unwrap();
/// println!("PSNR: {:.2} dB", psnr);
/// ```
pub fn calculate_psnr(original: &Array2<f64>, reconstructed: &Array2<f64>) -> SignalResult<f64> {
    if original.shape() != reconstructed.shape() {
        return Err(SignalError::ValueError(
            "Input arrays must have the same shape".to_string(),
        ));
    }

    if original.is_empty() {
        return Err(SignalError::ValueError(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Calculate MSE (Mean Squared Error)
    let mut mse = 0.0;
    let total_pixels = original.len();

    for (orig_pixel, recon_pixel) in original.iter().zip(reconstructed.iter()) {
        let diff = orig_pixel - recon_pixel;
        mse += diff * diff;
    }

    mse /= total_pixels as f64;

    // Handle perfect reconstruction (MSE = 0)
    if mse < f64::EPSILON {
        return Ok(f64::INFINITY);
    }

    // Calculate dynamic range (MAX_VALUE)
    let max_val = original.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = original.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let dynamic_range = max_val - min_val;

    if dynamic_range < f64::EPSILON {
        // All pixels have the same value - perfect match gives infinite PSNR
        if mse < f64::EPSILON {
            return Ok(f64::INFINITY);
        } else {
            return Ok(0.0); // Uniform image with error
        }
    }

    // Calculate PSNR
    let psnr = 20.0 * (dynamic_range / mse.sqrt()).log10();

    Ok(psnr)
}

/// Calculate Structural Similarity Index Measure (SSIM) between two images.
///
/// SSIM is a perceptually-motivated image quality metric that considers
/// luminance, contrast, and structural similarities. It ranges from -1 to 1,
/// with 1 indicating perfect structural similarity.
///
/// # Arguments
///
/// * `original` - The original (reference) image
/// * `reconstructed` - The processed image to evaluate
/// * `window_size` - Size of the sliding window for local SSIM calculation
/// * `k1` - First stability constant (default: 0.01)
/// * `k2` - Second stability constant (default: 0.03)
///
/// # Returns
///
/// * SSIM value between -1 and 1. Values closer to 1 indicate better quality.
///   - SSIM > 0.9: Excellent quality
///   - SSIM 0.7-0.9: Good quality
///   - SSIM < 0.7: Poor quality
///
/// # Errors
///
/// Returns an error if:
/// * Arrays have different shapes
/// * Window size is too large for the image
/// * Arrays are empty
///
/// # Algorithm
///
/// SSIM is calculated as:
/// SSIM = (2μ₁μ₂ + c₁)(2σ₁₂ + c₂) / ((μ₁² + μ₂²) + c₁)((σ₁² + σ₂²) + c₂)
///
/// where μ₁, μ₂ are local means, σ₁², σ₂² are local variances,
/// σ₁₂ is local covariance, and c₁, c₂ are stability constants.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt2d::calculate_ssim;
///
/// // Create test images
/// let original = Array2::from_shape_fn((16, 16), |(i, j)| {
///     ((i as f64 * 0.5).sin() + (j as f64 * 0.3).cos()) * 100.0
/// });
///
/// let mut processed = original.clone();
/// // Add some processing effects
/// for ((i, j), val) in processed.indexed_iter_mut() {
///     *val += ((i + j) as f64 * 0.1).sin();
/// }
///
/// let ssim = calculate_ssim(&original, &processed, 8, 0.01, 0.03).unwrap();
/// println!("SSIM: {:.4}", ssim);
/// ```
pub fn calculate_ssim(
    original: &Array2<f64>,
    reconstructed: &Array2<f64>,
    window_size: usize,
    k1: f64,
    k2: f64,
) -> SignalResult<f64> {
    if original.shape() != reconstructed.shape() {
        return Err(SignalError::ValueError(
            "Input arrays must have the same shape".to_string(),
        ));
    }

    let (height, width) = original.dim();
    if height < window_size || width < window_size {
        return Err(SignalError::ValueError(
            "Window size is too large for the image dimensions".to_string(),
        ));
    }

    // Calculate dynamic range
    let max_val = original.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = original.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let dynamic_range = max_val - min_val;

    // Stability constants
    let c1 = (k1 * dynamic_range).powi(2);
    let c2 = (k2 * dynamic_range).powi(2);

    let mut ssim_values = Vec::new();

    // Slide window across the image
    let step = window_size / 2; // Overlap windows by 50%
    let step = step.max(1);

    for i in (0..=(height.saturating_sub(window_size))).step_by(step) {
        for j in (0..=(width.saturating_sub(window_size))).step_by(step) {
            // Extract windows
            let window1 = original.slice(ndarray::s![i..i + window_size, j..j + window_size]);
            let window2 = reconstructed.slice(ndarray::s![i..i + window_size, j..j + window_size]);

            // Calculate local statistics
            let n = (window_size * window_size) as f64;
            let mu1 = window1.iter().sum::<f64>() / n;
            let mu2 = window2.iter().sum::<f64>() / n;

            let sigma1_sq = window1.iter().map(|&x| (x - mu1).powi(2)).sum::<f64>() / (n - 1.0);
            let sigma2_sq = window2.iter().map(|&x| (x - mu2).powi(2)).sum::<f64>() / (n - 1.0);

            let sigma12 = window1
                .iter()
                .zip(window2.iter())
                .map(|(&x1, &x2)| (x1 - mu1) * (x2 - mu2))
                .sum::<f64>()
                / (n - 1.0);

            // Calculate SSIM for this window
            let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
            let denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2);

            if denominator.abs() > f64::EPSILON {
                ssim_values.push(numerator / denominator);
            }
        }
    }

    if ssim_values.is_empty() {
        return Ok(1.0); // Perfect similarity for identical single-pixel images
    }

    // Return mean SSIM across all windows
    let mean_ssim = ssim_values.iter().sum::<f64>() / ssim_values.len() as f64;
    Ok(mean_ssim.clamp(-1.0, 1.0))
}

/// Enhanced 2D DWT with adaptive optimization
///
/// This function automatically selects the best implementation strategy based on
/// input characteristics, hardware capabilities, and performance requirements.
pub fn dwt2d_decompose_adaptive<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<Dwt2dResult>
where
    T: num_traits::Float + num_traits::NumCast + std::fmt::Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    let (rows, cols) = data.dim();

    // Validate input dimensions
    check_positive(rows, "rows")?;
    check_positive(cols, "cols")?;

    // Adaptive strategy selection based on image characteristics
    let total_elements = rows * cols;
    let caps = PlatformCapabilities::detect();

    // Choose configuration based on data size and hardware capabilities
    let config = if total_elements < 1024 {
        // Small images: use simple implementation
        Dwt2dConfig {
            preallocate_memory: false,
            use_inplace: false,
            memory_alignment: 16,
            chunk_size: None,
        }
    } else if total_elements < 100000 {
        // Medium images: use standard optimizations
        Dwt2dConfig {
            preallocate_memory: true,
            use_inplace: false,
            memory_alignment: if caps.simd_available { 32 } else { 16 },
            chunk_size: Some(8192),
        }
    } else {
        // Large images: use full optimizations
        Dwt2dConfig {
            preallocate_memory: true,
            use_inplace: false,
            memory_alignment: if caps.avx512_available {
                64
            } else if caps.simd_available {
                32
            } else {
                16
            },
            chunk_size: Some(16384),
        }
    };

    // Use optimized implementation with adaptive configuration
    crate::dwt2d::dwt2d_decompose_optimized(data, wavelet, mode, &config)
}

/// Enhanced multi-level 2D DWT with progressive validation
///
/// This function provides enhanced error checking and validation at each decomposition level,
/// making it more robust for production use.
pub fn wavedec2_enhanced<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    levels: usize,
    mode: Option<&str>,
) -> SignalResult<Vec<Dwt2dResult>>
where
    T: num_traits::Float + num_traits::NumCast + std::fmt::Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    if levels == 0 {
        return Err(SignalError::ValueError(
            "Levels must be greater than 0".to_string(),
        ));
    }

    let (rows, cols) = data.dim();
    check_positive(rows, "rows")?;
    check_positive(cols, "cols")?;

    // Enhanced size validation with better error messages
    let min_size = 2usize.pow(levels as u32);
    if rows < min_size {
        return Err(SignalError::ValueError(format!(
            "Number of rows ({}) is too small for {} levels of decomposition (minimum: {})",
            rows, levels, min_size
        )));
    }
    if cols < min_size {
        return Err(SignalError::ValueError(format!(
            "Number of columns ({}) is too small for {} levels of decomposition (minimum: {})",
            cols, levels, min_size
        )));
    }

    // Validate input data for numerical stability
    if let Some(data_slice) = data.as_slice() {
        for &val in data_slice {
            if let Some(f64_val) = num_traits::NumCast::from(val) {
                check_finite(f64_val, "input data")?;
            }
        }
    }

    let mut result = Vec::with_capacity(levels);

    // Perform first level with adaptive optimization
    let mut decomposition = dwt2d_decompose_adaptive(data, wavelet, mode)?;

    // Validate first level results
    crate::dwt2d::validation::validate_decomposition_level(&decomposition, 1, rows, cols)?;
    result.push(decomposition.clone());

    // Perform remaining levels with progressive validation
    for level in 1..levels {
        let prev_shape = decomposition.approx.shape().to_vec();
        decomposition = dwt2d_decompose_adaptive(&decomposition.approx, wavelet, mode)?;

        // Validate this level
        crate::dwt2d::validation::validate_decomposition_level(&decomposition, level + 1, prev_shape[0], prev_shape[1])?;
        result.push(decomposition.clone());
    }

    // Reverse so index 0 is the deepest level
    result.reverse();

    Ok(result)
}

/// Advanced denoising using 2D wavelet thresholding with adaptive threshold selection
///
/// This function automatically estimates optimal thresholds for each subband based on
/// noise characteristics and applies adaptive thresholding for improved denoising.
pub fn denoise_dwt2d_adaptive(
    noisy_image: &Array2<f64>,
    wavelet: Wavelet,
    noise_variance: Option<f64>,
    method: ThresholdMethod,
) -> SignalResult<Array2<f64>> {
    if noisy_image.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Validate input
    if let Some(data_slice) = noisy_image.as_slice() {
        for &val in data_slice {
            check_finite(val, "noisy image data")?;
        }
    }

    // Decompose the noisy image
    let mut decomposition = dwt2d_decompose_adaptive(noisy_image, wavelet, None)?;

    // Estimate noise variance if not provided
    let sigma = if let Some(var) = noise_variance {
        var.sqrt()
    } else {
        estimate_noise_variance(&decomposition)
    };

    // Apply adaptive thresholding to each detail subband
    // Use different thresholds for different orientations
    let threshold_h = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt() * 0.8; // Horizontal details
    let threshold_v = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt() * 0.8; // Vertical details
    let threshold_d = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt() * 1.2; // Diagonal details (usually more noisy)

    // Apply thresholding to detail coefficients only
    apply_adaptive_thresholding(&mut decomposition.detail_h, threshold_h, method);
    apply_adaptive_thresholding(&mut decomposition.detail_v, threshold_v, method);
    apply_adaptive_thresholding(&mut decomposition.detail_d, threshold_d, method);

    // Reconstruct the denoised image
    crate::dwt2d::dwt2d_reconstruct(&decomposition, wavelet, None)
}

/// Multi-level adaptive denoising
pub fn denoise_wavedec2_adaptive(
    noisy_image: &Array2<f64>,
    wavelet: Wavelet,
    levels: usize,
    noise_variance: Option<f64>,
    method: ThresholdMethod,
) -> SignalResult<Array2<f64>> {
    if noisy_image.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Multi-level decomposition
    let mut coeffs = wavedec2_enhanced(noisy_image, wavelet, levels, None)?;

    // Estimate noise variance from finest detail level if not provided
    let sigma = if let Some(var) = noise_variance {
        var.sqrt()
    } else {
        estimate_noise_variance(&coeffs[coeffs.len() - 1])
    };

    // Apply level-dependent thresholding
    let mut thresholds = Vec::with_capacity(levels);
    for level in 0..levels {
        // Higher thresholds for finer levels (more noise)
        let level_factor = 2.0_f64.powi((levels - level - 1) as i32).sqrt();
        let base_threshold = sigma * (2.0 * (noisy_image.len() as f64).ln()).sqrt();
        thresholds.push(base_threshold * level_factor);
    }

    // Apply thresholding to each level
    crate::dwt2d::threshold_wavedec2(&mut coeffs, &thresholds, method);

    // Reconstruct the denoised image
    crate::dwt2d::waverec2(&coeffs, wavelet, None)
}

/// Helper function for integer ceiling division
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::dwt::Wavelet;

    #[test]
    fn test_calculate_psnr_perfect() {
        let original = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
        let identical = original.clone();

        let psnr = calculate_psnr(&original, &identical).unwrap();
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_calculate_psnr_different_shapes() {
        let img1 = Array2::zeros((4, 4));
        let img2 = Array2::zeros((4, 3));

        let result = calculate_psnr(&img1, &img2);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_ssim_perfect() {
        let original = Array2::from_shape_fn((8, 8), |(i, j)| (i * j) as f64);
        let identical = original.clone();

        let ssim = calculate_ssim(&original, &identical, 4, 0.01, 0.03).unwrap();
        assert!((ssim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_ssim_different_shapes() {
        let img1 = Array2::zeros((8, 8));
        let img2 = Array2::zeros((8, 7));

        let result = calculate_ssim(&img1, &img2, 4, 0.01, 0.03);
        assert!(result.is_err());
    }

    #[test]
    fn test_denoise_dwt2d_adaptive() {
        // Create a simple test image
        let clean_image = Array2::from_shape_fn((16, 16), |(i, j)| {
            ((i as f64 * 0.3).sin() * (j as f64 * 0.2).cos()) * 10.0
        });

        // Add noise
        let mut noisy_image = clean_image.clone();
        for val in noisy_image.iter_mut() {
            *val += 0.5; // Add constant noise for simplicity
        }

        // Test denoising
        let denoised = denoise_dwt2d_adaptive(&noisy_image, Wavelet::DB(4), Some(0.25), ThresholdMethod::Soft);
        assert!(denoised.is_ok());

        let denoised = denoised.unwrap();
        assert_eq!(denoised.shape(), noisy_image.shape());
    }

    #[test]
    fn test_wavedec2_enhanced() {
        let data = Array2::from_shape_fn((16, 16), |(i, j)| (i + j) as f64);
        let result = wavedec2_enhanced(&data, Wavelet::Haar, 2, None);

        assert!(result.is_ok());
        let coeffs = result.unwrap();
        assert_eq!(coeffs.len(), 2);
    }
}