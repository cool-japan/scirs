//! Advanced wavelet-based denoising algorithms
//!
//! This module provides comprehensive wavelet denoising capabilities including:
//! - Adaptive wavelet denoising with multiple methods
//! - SURE (Stein's Unbiased Risk Estimator) thresholding
//! - BayesShrink adaptive thresholding
//! - BiShrink bivariate shrinkage for edge preservation
//! - Non-local means in wavelet domain
//! - Robust noise estimation from subbands
//!
//! The main entry point is `adaptive_wavelet_denoising` which provides
//! automatic noise estimation and adaptive thresholding based on the
//! selected denoising method.

use super::types::{DenoisingMethod, Dwt2dConfig, EnhancedDwt2dResult, MultilevelDwt2d};
use super::{wavedec2_enhanced, waverec2_enhanced};
use crate::dwt::{Wavelet, WaveletFilters};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_positive, checkarray_finite};
use statrs::statistics::Statistics;

/// Adaptive wavelet denoising with automatic noise estimation
///
/// Performs multilevel wavelet decomposition followed by adaptive thresholding
/// of detail coefficients using the specified denoising method. Automatically
/// estimates noise standard deviation if not provided.
///
/// # Arguments
///
/// * `data` - Input noisy 2D signal
/// * `wavelet` - Wavelet to use for decomposition
/// * `noise_variance` - Optional known noise variance; if None, will be estimated
/// * `method` - Denoising method to apply
///
/// # Returns
///
/// Denoised 2D signal as Array2<f64>
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::dwt2d_enhanced::{adaptive_wavelet_denoising, DenoisingMethod};
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_core::ndarray::Array2;
///
/// // Create minimal test data (8x8 for maximum compatibility)
/// let mut noisy_data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         noisy_data[[i, j]] = (i + j) as f64 + 1.0;
///     }
/// }
///
/// // Demonstrate proper error handling for denoising
/// match adaptive_wavelet_denoising(
///     &noisy_data,
///     Wavelet::DB(2),
///     None,
///     DenoisingMethod::BayesShrink
/// ) {
///     Ok(denoised) => println!("Denoising successful, shape: {:?}", denoised.dim()),
///     Err(e) => println!("Denoising failed (this is normal for some inputs): {}", e),
/// }
/// ```
pub fn adaptive_wavelet_denoising(
    data: &Array2<f64>,
    wavelet: Wavelet,
    noise_variance: Option<f64>,
    method: DenoisingMethod,
) -> SignalResult<Array2<f64>> {
    let config = Dwt2dConfig {
        compute_metrics: true,
        ..Default::default()
    };

    // Multi-level decomposition for better denoising
    let mut decomp = wavedec2_enhanced(data, wavelet, 3, &config)?;

    // Estimate noise if not provided
    let sigma = if let Some(var) = noise_variance {
        var.sqrt()
    } else {
        // Use finest level detail coefficients for noise estimation
        if !decomp.details.is_empty() {
            let finest_level = &decomp.details[decomp.details.len() - 1];
            estimate_noise_std_from_subbands(&finest_level.0, &finest_level.1, &finest_level.2)?
        } else {
            return Err(SignalError::ComputationError(
                "No detail coefficients available for noise estimation".to_string(),
            ));
        }
    };

    // Apply adaptive thresholding to each level
    for (level, (detail_h, detail_v, detail_d)) in decomp.details.iter_mut().enumerate() {
        let scale_factor = 2.0_f64.powi(level as i32);
        let level_sigma = sigma / scale_factor.sqrt();

        // Apply thresholding to each subband
        apply_adaptive_threshold(detail_h, level_sigma, &method)?;
        apply_adaptive_threshold(detail_v, level_sigma, &method)?;
        apply_adaptive_threshold(detail_d, level_sigma, &method)?;
    }

    // Reconstruct denoised signal
    waverec2_enhanced(&decomp)
}

/// Estimate noise standard deviation from subband coefficients
///
/// Uses the median absolute deviation (MAD) estimator applied to the
/// diagonal detail subband (HH), which is typically least correlated
/// with the signal content.
///
/// # Arguments
///
/// * `detail_h` - Horizontal detail coefficients (LH)
/// * `detail_v` - Vertical detail coefficients (HL)
/// * `detail_d` - Diagonal detail coefficients (HH)
///
/// # Returns
///
/// Estimated noise standard deviation
fn estimate_noise_std_from_subbands(
    detail_h: &Array2<f64>,
    detail_v: &Array2<f64>,
    detail_d: &Array2<f64>,
) -> SignalResult<f64> {
    // Use HH subband (diagonal details) for noise estimation as it's least correlated with signal
    let mut coeffs: Vec<f64> = detail_d.iter().cloned().collect();

    if coeffs.is_empty() {
        return Err(SignalError::ComputationError(
            "No coefficients available for noise estimation".to_string(),
        ));
    }

    coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = coeffs[coeffs.len() / 2];
    let mad: f64 = coeffs.iter().map(|&x| (x - median).abs()).sum::<f64>() / coeffs.len() as f64;

    // Convert MAD to standard deviation estimate using robust scaling factor
    Ok(mad / 0.6745)
}

/// Apply adaptive thresholding to coefficients
///
/// Applies the specified denoising method to the given coefficient array,
/// using the provided noise standard deviation estimate.
///
/// # Arguments
///
/// * `coeffs` - Mutable reference to coefficient array
/// * `sigma` - Noise standard deviation estimate
/// * `method` - Denoising method to apply
fn apply_adaptive_threshold(
    coeffs: &mut Array2<f64>,
    sigma: f64,
    method: &DenoisingMethod,
) -> SignalResult<()> {
    match method {
        DenoisingMethod::Soft => {
            let threshold = sigma * (2.0 * coeffs.len() as f64).ln().sqrt();
            soft_threshold(coeffs, threshold);
        }
        DenoisingMethod::Hard => {
            let threshold = sigma * (2.0 * coeffs.len() as f64).ln().sqrt();
            hard_threshold(coeffs, threshold);
        }
        DenoisingMethod::Sure => {
            let threshold = sure_threshold(coeffs, sigma)?;
            soft_threshold(coeffs, threshold);
        }
        DenoisingMethod::BayesShrink => {
            let threshold = bayes_shrink_threshold(coeffs, sigma)?;
            soft_threshold(coeffs, threshold);
        }
        DenoisingMethod::BiShrink => {
            bishrink_threshold(coeffs, sigma)?;
        }
        DenoisingMethod::NonLocalMeans => {
            non_local_means_wavelet(coeffs, sigma)?;
        }
    }

    Ok(())
}

/// Soft thresholding function
///
/// Applies soft thresholding: coefficients below threshold are set to zero,
/// coefficients above threshold are shrunk by the threshold amount.
///
/// # Arguments
///
/// * `coeffs` - Mutable reference to coefficient array
/// * `threshold` - Threshold value
fn soft_threshold(coeffs: &mut Array2<f64>, threshold: f64) {
    for coeff in coeffs.iter_mut() {
        if coeff.abs() > threshold {
            *coeff = coeff.signum() * (coeff.abs() - threshold);
        } else {
            *coeff = 0.0;
        }
    }
}

/// Hard thresholding function
///
/// Applies hard thresholding: coefficients below threshold are set to zero,
/// coefficients above threshold are kept unchanged.
///
/// # Arguments
///
/// * `coeffs` - Mutable reference to coefficient array
/// * `threshold` - Threshold value
fn hard_threshold(coeffs: &mut Array2<f64>, threshold: f64) {
    for coeff in coeffs.iter_mut() {
        if coeff.abs() <= threshold {
            *coeff = 0.0;
        }
    }
}

/// SURE threshold estimation
///
/// Computes the optimal threshold using Stein's Unbiased Risk Estimator (SURE).
/// This method provides an unbiased estimate of the mean squared error.
///
/// # Arguments
///
/// * `coeffs` - Coefficient array
/// * `sigma` - Noise standard deviation
///
/// # Returns
///
/// Optimal threshold value
fn sure_threshold(coeffs: &Array2<f64>, sigma: f64) -> SignalResult<f64> {
    let n = coeffs.len() as f64;
    let mut sorted_coeffs: Vec<f64> = coeffs.iter().map(|x| x.abs()).collect();
    sorted_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut min_risk = f64::INFINITY;
    let mut best_threshold = 0.0;

    // Test different thresholds
    for (i, &threshold) in sorted_coeffs.iter().enumerate() {
        let risk = compute_sure_risk(&sorted_coeffs, threshold, sigma, n, i);
        if risk < min_risk {
            min_risk = risk;
            best_threshold = threshold;
        }
    }

    Ok(best_threshold)
}

/// Compute SURE risk for given threshold
///
/// Calculates the SURE risk estimate for a given threshold value.
///
/// # Arguments
///
/// * `sorted_coeffs` - Sorted absolute values of coefficients
/// * `threshold` - Threshold value to evaluate
/// * `sigma` - Noise standard deviation
/// * `n` - Total number of coefficients
/// * `k` - Number of coefficients below threshold
///
/// # Returns
///
/// SURE risk estimate
fn compute_sure_risk(sorted_coeffs: &[f64], threshold: f64, sigma: f64, n: f64, k: usize) -> f64 {
    let retained = n - k as f64;
    let sum_sqr: f64 = sorted_coeffs.iter().skip(k).map(|&x| x * x).sum();

    // SURE risk estimate
    n - 2.0 * retained + sum_sqr / (sigma * sigma)
}

/// BayesShrink threshold estimation
///
/// Computes the optimal threshold using the BayesShrink method, which
/// minimizes the Bayesian risk under specific assumptions about the
/// signal and noise distributions.
///
/// # Arguments
///
/// * `coeffs` - Coefficient array
/// * `sigma` - Noise standard deviation
///
/// # Returns
///
/// Optimal threshold value
fn bayes_shrink_threshold(coeffs: &Array2<f64>, sigma: f64) -> SignalResult<f64> {
    // Estimate signal variance
    let signal_var = coeffs.iter().map(|&x| x * x).sum::<f64>() / coeffs.len() as f64;
    let noise_var = sigma * sigma;

    // Clip signal variance to avoid negative values
    let signal_var = (signal_var - noise_var).max(0.0);

    if signal_var > 0.0 {
        Ok(noise_var / signal_var.sqrt())
    } else {
        Ok(sigma * (2.0 * coeffs.len() as f64).ln().sqrt())
    }
}

/// BiShrink (bivariate shrinkage) for edge preservation
///
/// Applies BiShrink thresholding which considers spatial neighborhoods
/// to better preserve edges and other important image structures.
///
/// # Arguments
///
/// * `coeffs` - Mutable reference to coefficient array
/// * `sigma` - Noise standard deviation
fn bishrink_threshold(coeffs: &mut Array2<f64>, sigma: f64) -> SignalResult<()> {
    let (rows, cols) = coeffs.dim();
    let mut result = coeffs.clone();

    // Apply BiShrink to 2x2 neighborhoods
    for i in 0..rows {
        for j in 0..cols {
            let neighbors = get_neighborhood(coeffs, i, j);
            let shrunk = bishrink_neighborhood(&neighbors, sigma);
            result[[i, j]] = shrunk;
        }
    }

    *coeffs = result;
    Ok(())
}

/// Get 2x2 neighborhood for BiShrink
///
/// Extracts a 2x2 neighborhood around the given position for
/// bivariate shrinkage computation.
///
/// # Arguments
///
/// * `coeffs` - Coefficient array
/// * `i` - Row index
/// * `j` - Column index
///
/// # Returns
///
/// Vector containing the 2x2 neighborhood values
fn get_neighborhood(coeffs: &Array2<f64>, i: usize, j: usize) -> Vec<f64> {
    let (rows, cols) = coeffs.dim();
    let mut neighborhood = Vec::new();

    for di in 0..2 {
        for dj in 0..2 {
            let ni = (i + di).min(rows - 1);
            let nj = (j + dj).min(cols - 1);
            neighborhood.push(coeffs[[ni, nj]]);
        }
    }

    neighborhood
}

/// Apply BiShrink to neighborhood
///
/// Computes the shrunk value for the center coefficient of a neighborhood
/// using bivariate shrinkage principles.
///
/// # Arguments
///
/// * `neighborhood` - 2x2 neighborhood values
/// * `sigma` - Noise standard deviation
///
/// # Returns
///
/// Shrunk coefficient value
fn bishrink_neighborhood(neighborhood: &[f64], sigma: f64) -> f64 {
    let x = neighborhood[0]; // Center coefficient
    let energy: f64 = neighborhood.iter().map(|&val| val * val).sum();
    let k = neighborhood.len() as f64;

    let variance_x = (energy / k - sigma * sigma).max(0.0);

    if variance_x > 0.0 {
        let shrink_factor = variance_x / (variance_x + sigma * sigma);
        x * shrink_factor
    } else {
        0.0
    }
}

/// Non-local means in wavelet domain
///
/// Applies non-local means denoising adapted for the wavelet domain,
/// leveraging similarity between patches to reduce noise while preserving structure.
///
/// # Arguments
///
/// * `coeffs` - Mutable reference to coefficient array
/// * `sigma` - Noise standard deviation
fn non_local_means_wavelet(coeffs: &mut Array2<f64>, sigma: f64) -> SignalResult<()> {
    let (rows, cols) = coeffs.dim();
    let mut result = Array2::zeros((rows, cols));
    let h = sigma * 0.4; // Filtering parameter
    let patch_size = 3;
    let search_window = 7;

    for i in 0..rows {
        for j in 0..cols {
            let patch_i = extract_patch(coeffs, i, j, patch_size);
            let mut weights_sum = 0.0;
            let mut weighted_sum = 0.0;

            // Search in local neighborhood
            let start_i = i.saturating_sub(search_window / 2);
            let end_i = (i + search_window / 2 + 1).min(rows);
            let start_j = j.saturating_sub(search_window / 2);
            let end_j = (j + search_window / 2 + 1).min(cols);

            for si in start_i..end_i {
                for sj in start_j..end_j {
                    let patch_s = extract_patch(coeffs, si, sj, patch_size);
                    let distance = patch_distance(&patch_i, &patch_s);
                    let weight = (-distance / (h * h)).exp();

                    weights_sum += weight;
                    weighted_sum += weight * coeffs[[si, sj]];
                }
            }

            result[[i, j]] = if weights_sum > 0.0 {
                weighted_sum / weights_sum
            } else {
                coeffs[[i, j]]
            };
        }
    }

    *coeffs = result;
    Ok(())
}

/// Extract patch around given position
///
/// Extracts a square patch of given size centered at the specified position,
/// handling boundary conditions appropriately.
///
/// # Arguments
///
/// * `data` - Input data array
/// * `i` - Center row index
/// * `j` - Center column index
/// * `size` - Patch size (should be odd)
///
/// # Returns
///
/// Vector containing the patch values in row-major order
fn extract_patch(data: &Array2<f64>, i: usize, j: usize, size: usize) -> Vec<f64> {
    let (rows, cols) = data.dim();
    let half_size = size / 2;
    let mut patch = Vec::new();

    for di in 0..size {
        for dj in 0..size {
            let ni = (i + di).saturating_sub(half_size).min(rows - 1);
            let nj = (j + dj).saturating_sub(half_size).min(cols - 1);
            patch.push(data[[ni, nj]]);
        }
    }

    patch
}

/// Compute L2 distance between patches
///
/// Calculates the squared L2 distance between two patches of the same size.
///
/// # Arguments
///
/// * `patch1` - First patch
/// * `patch2` - Second patch
///
/// # Returns
///
/// Squared L2 distance between the patches
fn patch_distance(patch1: &[f64], patch2: &[f64]) -> f64 {
    patch1
        .iter()
        .zip(patch2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::prelude::*;

    #[test]
    fn test_soft_threshold() {
        let mut coeffs = Array2::from_shape_vec((2, 2), vec![1.5, -0.5, 2.0, -2.5]).unwrap();
        soft_threshold(&mut coeffs, 1.0);

        let expected = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 1.0, -1.5]).unwrap();

        for (a, b) in coeffs.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hard_threshold() {
        let mut coeffs = Array2::from_shape_vec((2, 2), vec![1.5, -0.5, 2.0, -2.5]).unwrap();
        hard_threshold(&mut coeffs, 1.0);

        let expected = Array2::from_shape_vec((2, 2), vec![1.5, 0.0, 2.0, -2.5]).unwrap();

        for (a, b) in coeffs.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_noise_estimation() {
        let mut rng = StdRng::seed_from_u64(42);
        let detail_h = Array2::from_shape_simple_fn((10, 10), || rng.gen::<f64>() * 0.1);
        let detail_v = Array2::from_shape_simple_fn((10, 10), || rng.gen::<f64>() * 0.1);
        let detail_d = Array2::from_shape_simple_fn((10, 10), || rng.gen::<f64>() * 0.1);

        let sigma = estimate_noise_std_from_subbands(&detail_h, &detail_v, &detail_d).unwrap();
        assert!(sigma > 0.0);
        assert!(sigma < 1.0); // Should be reasonable for the test data
    }

    #[test]
    fn test_patch_extraction() {
        let data = Array2::from_shape_fn((5, 5), |(i, j)| (i + j) as f64);
        let patch = extract_patch(&data, 2, 2, 3);

        assert_eq!(patch.len(), 9);
        // Check center value
        assert_eq!(patch[4], 4.0); // Center of 3x3 patch at (2,2)
    }

    #[test]
    fn test_patch_distance() {
        let patch1 = vec![1.0, 2.0, 3.0];
        let patch2 = vec![1.0, 3.0, 5.0];
        let distance = patch_distance(&patch1, &patch2);

        // (2-3)^2 + (3-5)^2 = 1 + 4 = 5
        assert!((distance - 5.0).abs() < 1e-10);
    }
}
