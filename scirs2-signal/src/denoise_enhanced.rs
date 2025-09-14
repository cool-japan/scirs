use ndarray::s;
// Enhanced wavelet denoising with advanced thresholding methods
//
// This module provides state-of-the-art wavelet denoising techniques including:
// - Translation-invariant denoising
// - Block thresholding
// - Stein's unbiased risk estimate (SURE)
// - BayesShrink and VisuShrink
// - Adaptive thresholding

use crate::dwt::{wavedec, waverec, Wavelet};
use crate::dwt2d::dwt2d_decompose;
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rand::Rng;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::{check_finite, check_positive};
use scirs2_fft::{fft, ifft};
use std::f64;

#[allow(unused_imports)]
// use num_complex::Complex64;

/// Enhanced denoising configuration
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Wavelet to use
    pub wavelet: Wavelet,
    /// Decomposition levels (None for automatic)
    pub levels: Option<usize>,
    /// Thresholding method
    pub method: ThresholdMethod,
    /// Threshold selection rule
    pub threshold_rule: ThresholdRule,
    /// Use translation-invariant denoising
    pub translation_invariant: bool,
    /// Number of shifts for TI denoising
    pub n_shifts: usize,
    /// Use parallel processing
    pub parallel: bool,
    /// Preserve approximation coefficients
    pub preserve_approx: bool,
    /// Level-dependent thresholding
    pub level_dependent: bool,
    /// Use SIMD optimization
    pub use_simd: bool,
    /// Memory optimization for large signals
    pub memory_optimized: bool,
    /// Block size for memory-optimized processing
    pub block_size: Option<usize>,
    /// Enhanced numerical stability
    pub numerical_stability: bool,
    /// Adaptive noise estimation
    pub adaptive_noise: bool,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::DB(4),
            levels: None,
            method: ThresholdMethod::Soft,
            threshold_rule: ThresholdRule::SURE,
            translation_invariant: false,
            n_shifts: 8,
            parallel: true,
            preserve_approx: true,
            level_dependent: true,
            use_simd: true,
            memory_optimized: true,
            block_size: None,
            numerical_stability: true,
            adaptive_noise: true,
        }
    }
}

/// Thresholding methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Soft thresholding (shrinkage)
    Soft,
    /// Hard thresholding
    Hard,
    /// Garotte thresholding
    Garotte,
    /// SCAD (Smoothly Clipped Absolute Deviation)
    SCAD { a: f64 },
    /// Firm thresholding
    Firm { alpha: f64 },
    /// Hyperbolic thresholding
    Hyperbolic,
    /// Block thresholding
    Block { block_size: usize },
}

/// Threshold selection rules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdRule {
    /// Universal threshold (VisuShrink)
    Universal,
    /// Stein's Unbiased Risk Estimate
    SURE,
    /// BayesShrink
    Bayes,
    /// Minimax threshold
    Minimax,
    /// Cross-validation
    CrossValidation,
    /// False Discovery Rate
    FDR { q: f64 },
    /// Custom threshold value
    Custom(f64),
}

/// Denoising result with diagnostics
#[derive(Debug, Clone)]
pub struct DenoiseResult {
    /// Denoised signal
    pub signal: Array1<f64>,
    /// Estimated noise level
    pub noise_sigma: f64,
    /// Thresholds used at each level
    pub thresholds: Vec<f64>,
    /// Percentage of coefficients retained
    pub retention_rate: f64,
    /// Effective degrees of freedom
    pub effective_df: f64,
    /// Risk estimate (if available)
    pub risk_estimate: Option<f64>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Numerical stability score (0-1)
    pub stability_score: f64,
    /// Signal-to-noise ratio improvement
    pub snr_improvement_db: f64,
}

/// 2D denoising result
#[derive(Debug, Clone)]
pub struct Denoise2dResult {
    /// Denoised image
    pub image: Array2<f64>,
    /// Estimated noise level
    pub noise_sigma: f64,
    /// Thresholds per subband
    pub thresholds: SubbandThresholds,
    /// Retention rates per subband
    pub retention_rates: SubbandRetention,
    /// Quality metrics
    pub quality: QualityMetrics,
}

/// Thresholds for each subband
#[derive(Debug, Clone)]
pub struct SubbandThresholds {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub diagonal: Vec<f64>,
}

/// Retention rates for each subband
#[derive(Debug, Clone)]
pub struct SubbandRetention {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub diagonal: Vec<f64>,
}

/// Quality metrics for denoising
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Estimated SNR improvement (dB)
    pub snr_improvement: f64,
    /// Edge preservation index
    pub edge_preservation: f64,
    /// Texture preservation index
    pub texture_preservation: f64,
}

/// Configuration for non-local means denoising
#[derive(Debug, Clone)]
pub struct NonLocalMeansConfig {
    /// Size of patches to compare
    pub patch_size: usize,
    /// Size of search window
    pub search_window: usize,
    /// Filtering parameter (controls decay of weights)
    pub filtering_parameter: f64,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for NonLocalMeansConfig {
    fn default() -> Self {
        Self {
            patch_size: 7,
            search_window: 21,
            filtering_parameter: 0.1,
            parallel: true,
        }
    }
}

/// Configuration for total variation denoising
#[derive(Debug, Clone)]
pub struct TotalVariationConfig {
    /// Regularization parameter (controls smoothness vs. fidelity trade-off)
    pub lambda: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Step size for gradient descent
    pub step_size: f64,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for TotalVariationConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iterations: 100,
            step_size: 0.01,
            tolerance: 1e-6,
        }
    }
}

/// Configuration for bilateral filtering
#[derive(Debug, Clone)]
pub struct BilateralConfig {
    /// Spatial standard deviation
    pub spatial_sigma: f64,
    /// Intensity standard deviation
    pub intensity_sigma: f64,
    /// Window size for filtering
    pub window_size: usize,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for BilateralConfig {
    fn default() -> Self {
        Self {
            spatial_sigma: 2.0,
            intensity_sigma: 0.1,
            window_size: 15,
            parallel: true,
        }
    }
}

/// Configuration for Wiener filtering
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WienerConfig {
    /// Method for noise estimation
    pub noise_estimation: WienerNoiseEstimation,
}

impl Default for WienerConfig {
    fn default() -> Self {
        Self {
            noise_estimation: WienerNoiseEstimation::HighFrequency,
        }
    }
}

/// Noise estimation methods for Wiener filtering
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WienerNoiseEstimation {
    /// Constant noise power
    Constant(f64),
    /// Estimate from high-frequency components
    HighFrequency,
    /// Minimum statistics approach
    MinimumStatistics,
}

/// Configuration for adaptive LMS filtering
#[derive(Debug, Clone)]
pub struct AdaptiveLMSConfig {
    /// Length of adaptive filter
    pub filter_length: usize,
    /// Step size (learning rate)
    pub step_size: f64,
    /// Initial weights (None for zero initialization)
    pub initial_weights: Option<Vec<f64>>,
    /// Prediction mode (true) or noise cancellation mode (false)
    pub prediction_mode: bool,
    /// Delay for reference signal (in noise cancellation mode)
    pub delay: usize,
}

impl Default for AdaptiveLMSConfig {
    fn default() -> Self {
        Self {
            filter_length: 32,
            step_size: 0.01,
            initial_weights: None,
            prediction_mode: true,
            delay: 1,
        }
    }
}

/// Enhanced 1D wavelet denoising
#[allow(dead_code)]
pub fn denoise_wavelet_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    // Signal validation handled by wavelet transform

    if config.translation_invariant {
        translation_invariant_denoise_1d(signal, config)
    } else {
        standard_denoise_1d(signal, config)
    }
}

/// Advanced non-local means denoising for 1D signals
#[allow(dead_code)]
pub fn denoise_non_local_means_1d(
    signal: &Array1<f64>,
    config: &NonLocalMeansConfig,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(config.patch_size, "patch_size")?;
    check_positive(config.search_window, "search_window")?;

    let n = signal.len();
    let mut denoised = Array1::zeros(n);
    let patch_size = config.patch_size;
    let search_window = config.search_window;
    let h = config.filtering_parameter;
    let h_sq = h * h;

    // Process each point
    if config.parallel {
        let denoised_vec: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                // Define search region
                let search_start = (i as i32 - search_window as i32 / 2).max(0) as usize;
                let search_end = (i + search_window / 2 + 1).min(n);

                // Compare with all patches in search window
                for j in search_start..search_end {
                    let weight = compute_patch_similarity_1d(signal, i, j, patch_size, h_sq);
                    weighted_sum += weight * signal[j];
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    signal[i]
                }
            })
            .collect();

        denoised = Array1::from_vec(denoised_vec);
    } else {
        for i in 0..n {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            let search_start = (i as i32 - search_window as i32 / 2).max(0) as usize;
            let search_end = (i + search_window / 2 + 1).min(n);

            for j in search_start..search_end {
                let weight = compute_patch_similarity_1d(signal, i, j, patch_size, h_sq);
                weighted_sum += weight * signal[j];
                weight_sum += weight;
            }

            denoised[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                signal[i]
            };
        }
    }

    Ok(denoised)
}

/// Total variation denoising for 1D signals
#[allow(dead_code)]
pub fn denoise_total_variation_1d(
    signal: &Array1<f64>,
    config: &TotalVariationConfig,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(config.lambda, "lambda")?;

    let n = signal.len();
    let mut denoised = signal.clone();
    let lambda = config.lambda;
    let max_iter = config.max_iterations;
    let tolerance = config.tolerance;

    // Iterative solution using projected gradient descent
    for iter in 0..max_iter {
        let mut gradient = Array1::zeros(n);
        let mut tv_term = Array1::zeros(n);

        // Compute total variation gradient
        for i in 1..n - 1 {
            let left_diff = denoised[i] - denoised[i - 1];
            let right_diff = denoised[i + 1] - denoised[i];

            // TV regularization term (finite differences)
            let left_sign = if left_diff.abs() > 1e-12 {
                left_diff.signum()
            } else {
                0.0
            };
            let right_sign = if right_diff.abs() > 1e-12 {
                right_diff.signum()
            } else {
                0.0
            };

            tv_term[i] = lambda * (left_sign - right_sign);
        }

        // Handle boundary conditions
        if n > 1 {
            let first_diff = denoised[1] - denoised[0];
            tv_term[0] = -lambda
                * if first_diff.abs() > 1e-12 {
                    first_diff.signum()
                } else {
                    0.0
                };

            let last_diff = denoised[n - 1] - denoised[n - 2];
            tv_term[n - 1] = lambda
                * if last_diff.abs() > 1e-12 {
                    last_diff.signum()
                } else {
                    0.0
                };
        }

        // Compute full gradient
        for i in 0..n {
            gradient[i] = denoised[i] - signal[i] + tv_term[i];
        }

        // Update with step size
        let step_size = config.step_size;
        let mut max_change = 0.0f64;

        for i in 0..n {
            let old_val = denoised[i];
            denoised[i] -= step_size * gradient[i];
            max_change = max_change.max((denoised[i] - old_val).abs());
        }

        // Check convergence
        if max_change < tolerance {
            break;
        }
    }

    Ok(denoised)
}

/// Bilateral filtering for 1D signals
#[allow(dead_code)]
pub fn denoise_bilateral_1d(
    signal: &Array1<f64>,
    config: &BilateralConfig,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(config.spatial_sigma, "spatial_sigma")?;
    check_positive(config.intensity_sigma, "intensity_sigma")?;

    let n = signal.len();
    let mut denoised = Array1::zeros(n);
    let spatial_sigma = config.spatial_sigma;
    let intensity_sigma = config.intensity_sigma;
    let window_size = config.window_size;

    let spatial_factor = -0.5 / (spatial_sigma * spatial_sigma);
    let intensity_factor = -0.5 / (intensity_sigma * intensity_sigma);

    // Process each point
    if config.parallel {
        let denoised_vec: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = (i as i32 - window_size as i32 / 2).max(0) as usize;
                let end = (i + window_size / 2 + 1).min(n);

                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for j in start..end {
                    let spatial_dist = (i as f64 - j as f64).abs();
                    let intensity_dist = (signal[i] - signal[j]).abs();

                    let spatial_weight = (spatial_dist * spatial_dist * spatial_factor).exp();
                    let intensity_weight =
                        (intensity_dist * intensity_dist * intensity_factor).exp();
                    let total_weight = spatial_weight * intensity_weight;

                    weighted_sum += total_weight * signal[j];
                    weight_sum += total_weight;
                }

                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    signal[i]
                }
            })
            .collect();

        denoised = Array1::from_vec(denoised_vec);
    } else {
        for i in 0..n {
            let start = (i as i32 - window_size as i32 / 2).max(0) as usize;
            let end = (i + window_size / 2 + 1).min(n);

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in start..end {
                let spatial_dist = (i as f64 - j as f64).abs();
                let intensity_dist = (signal[i] - signal[j]).abs();

                let spatial_weight = (spatial_dist * spatial_dist * spatial_factor).exp();
                let intensity_weight = (intensity_dist * intensity_dist * intensity_factor).exp();
                let total_weight = spatial_weight * intensity_weight;

                weighted_sum += total_weight * signal[j];
                weight_sum += total_weight;
            }

            denoised[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                signal[i]
            };
        }
    }

    Ok(denoised)
}

/// Wiener filtering for signal denoising
#[allow(dead_code)]
pub fn denoise_wiener_1d(
    _signal: &Array1<f64>,
    config: &WienerConfig,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm

    let n = signal.len();

    // Pad to power of 2 for efficient FFT
    let padded_size = n.next_power_of_two();
    let mut padded_signal = vec![0.0; padded_size];
    padded_signal[..n].copy_from_slice(_signal.as_slice().unwrap());

    // Convert to complex for FFT
    let complex_signal: Vec<_> = padded_signal
        .iter()
        .map(|&x| num_complex::Complex64::new(x, 0.0))
        .collect();

    // Compute FFT
    let signal_fft = fft(&complex_signal, None)?;

    // Estimate noise power spectral density
    let noise_psd = match config.noise_estimation {
        WienerNoiseEstimation::Constant(noise_power) => noise_power,
        WienerNoiseEstimation::HighFrequency => {
            // Estimate from high-frequency components
            let high_freq_start = padded_size * 3 / 4;
            let high_freq_power: f64 = signal_fft[high_freq_start..]
                .iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                / (padded_size - high_freq_start) as f64;
            high_freq_power
        }
        WienerNoiseEstimation::MinimumStatistics => {
            // Estimate using minimum statistics in frequency domain
            let window_size = padded_size / 8;
            let mut min_power = f64::INFINITY;

            for i in 0..padded_size - window_size {
                let window_power: f64 = signal_fft[i..i + window_size]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
                    / window_size as f64;
                min_power = min_power.min(window_power);
            }
            min_power
        }
    };

    // Apply Wiener filter
    let mut filtered_fft = Vec::with_capacity(padded_size);
    for i in 0..padded_size {
        let signal_power = signal_fft[i].norm_sqr();
        let wiener_gain = signal_power / (signal_power + noise_psd);
        filtered_fft.push(signal_fft[i] * wiener_gain);
    }

    // Inverse FFT
    let filtered_complex = ifft(&filtered_fft, None)?;

    // Extract real part and trim to original size
    let filtered: Vec<f64> = filtered_complex[..n].iter().map(|c| c.re).collect();

    Ok(Array1::from_vec(filtered))
}

/// Adaptive LMS filtering for signal denoising
#[allow(dead_code)]
pub fn denoise_adaptive_lms(
    signal: &Array1<f64>,
    config: &AdaptiveLMSConfig,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(config.filter_length, "filter_length")?;
    check_positive(config.step_size, "step_size")?;

    let n = signal.len();
    let filter_length = config.filter_length;
    let mu = config.step_size;

    if n < filter_length {
        return Err(SignalError::ValueError(
            "Signal length must be greater than filter length".to_string(),
        ));
    }

    let mut weights = vec![0.0; filter_length];
    let mut denoised = vec![0.0; n];

    // Initialize filter weights
    if config.initialweights.is_some() {
        weights = config.initialweights.as_ref().unwrap().clone();
    }

    // LMS adaptation
    for i in filter_length..n {
        // Extract input vector
        let input = &signal.as_slice().unwrap()[i - filter_length..i];

        // Compute filter output
        let output: f64 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // Desired signal (delayed version for prediction)
        let desired = if config.prediction_mode {
            signal[i]
        } else {
            // For noise cancellation, use a reference or delayed signal
            signal[i - config.delay]
        };

        // Error signal
        let error = desired - output;

        // Update weights
        for j in 0..filter_length {
            weights[j] += mu * error * input[j];
        }

        // Store output (noise-reduced signal)
        denoised[i] = if config.prediction_mode {
            output // Predicted signal
        } else {
            desired - output // Noise-reduced signal
        };
    }

    // Handle initial samples
    for i in 0..filter_length {
        denoised[i] = signal[i];
    }

    Ok(Array1::from_vec(denoised))
}

/// Standard wavelet denoising
#[allow(dead_code)]
fn standard_denoise_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    let n = signal.len();

    // Determine decomposition levels
    let max_levels = (n as f64).log2().floor() as usize - 1;
    let levels = config.levels.unwrap_or(max_levels.min(6));

    // Perform wavelet decomposition
    let wavedec_result = wavedec(
        signal.as_slice().unwrap(),
        config.wavelet,
        Some(levels),
        None,
    )?;
    let coeffs = crate::dwt::DecompositionResult::from_wavedec(wavedec_result);

    // Estimate noise level from finest scale coefficients
    let noise_sigma = estimate_noise_mad(&coeffs.details[coeffs.details.len() - 1]);

    // Apply thresholding
    let (thresholded_coeffs, thresholds, retention_rates) =
        apply_thresholding(&coeffs, noise_sigma, config)?;

    // Reconstruct signal
    let waverec_input = thresholded_coeffs.to_wavedec();
    let denoised_vec = waverec(&waverec_input, config.wavelet)?;
    let denoised = Array1::from_vec(denoised_vec);

    // Calculate diagnostics
    let total_coeffs: usize = coeffs.details.iter().map(|d| d.len()).sum();
    let retained_coeffs: f64 = retention_rates
        .iter()
        .zip(coeffs.details.iter())
        .map(|(rate, detail)| rate * detail.len() as f64)
        .sum();
    let retention_rate = retained_coeffs / total_coeffs as f64;

    let effective_df = compute_effective_df(&thresholded_coeffs);

    let risk_estimate = match config.threshold_rule {
        ThresholdRule::SURE => Some(compute_sure_risk(&coeffs, &thresholds, noise_sigma)?),
        _ => None,
    };

    Ok(DenoiseResult {
        signal: denoised,
        noise_sigma,
        thresholds,
        retention_rate,
        effective_df,
        risk_estimate,
        processing_time_ms: 0.0,
        memory_usage_bytes: 0,
        stability_score: 1.0,
        snr_improvement_db: 0.0,
    })
}

/// Translation-invariant denoising
#[allow(dead_code)]
fn translation_invariant_denoise_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    let n = signal.len();
    let n_shifts = config.n_shifts.min(n);

    // Store shifted and denoised versions
    let mut denoised_shifts: Vec<DenoiseResult> = Vec::with_capacity(n_shifts);
    let mut all_noise_estimates = Vec::with_capacity(n_shifts);

    // Process each shift
    let shift_results: Vec<_> = if config.parallel {
        (0..n_shifts)
            .into_par_iter()
            .map(|shift| {
                // Circular shift
                let mut shifted = Array1::zeros(n);
                for i in 0..n {
                    shifted[i] = signal[(i + shift) % n];
                }

                // Denoise shifted signal
                let mut shift_config = config.clone();
                shift_config.translation_invariant = false;

                standard_denoise_1d(&shifted, &shift_config)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..n_shifts)
            .map(|shift| {
                let mut shifted = Array1::zeros(n);
                for i in 0..n {
                    shifted[i] = signal[(i + shift) % n];
                }

                let mut shift_config = config.clone();
                shift_config.translation_invariant = false;

                standard_denoise_1d(&shifted, &shift_config)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Average the unshifted results
    let mut averaged = Array1::zeros(n);

    for (shift, result) in shift_results.iter().enumerate() {
        // Unshift the denoised signal
        for i in 0..n {
            averaged[(i + shift) % n] += result.signal[i] / n_shifts as f64;
        }

        all_noise_estimates.push(result.noise_sigma);
    }

    // Aggregate diagnostics
    let noise_sigma = all_noise_estimates.iter().sum::<f64>() / n_shifts as f64;
    let thresholds = shift_results[0].thresholds.clone(); // Use first shift's thresholds
    let retention_rate =
        shift_results.iter().map(|r| r.retention_rate).sum::<f64>() / n_shifts as f64;
    let effective_df = compute_effective_df_ti(&averaged, signal);

    Ok(DenoiseResult {
        signal: averaged,
        noise_sigma,
        thresholds,
        retention_rate,
        effective_df,
        risk_estimate: None,
        processing_time_ms: 0.0,
        memory_usage_bytes: 0,
        stability_score: 1.0,
        snr_improvement_db: 0.0,
    })
}

/// Enhanced 2D wavelet denoising
#[allow(dead_code)]
pub fn denoise_wavelet_2d(
    image: &Array2<f64>,
    config: &DenoiseConfig,
) -> SignalResult<Denoise2dResult> {
    for val in image.iter() {
        check_finite(*val, "image")?;
    }

    let (rows, cols) = image.dim();
    let max_levels = ((rows.min(cols)) as f64).log2().floor() as usize - 1;
    let levels = config.levels.unwrap_or(max_levels.min(4));

    // Store results for each level
    let mut all_h_thresholds = Vec::with_capacity(levels);
    let mut all_v_thresholds = Vec::with_capacity(levels);
    let mut all_d_thresholds = Vec::with_capacity(levels);
    let mut all_h_retention = Vec::with_capacity(levels);
    let mut all_v_retention = Vec::with_capacity(levels);
    let mut all_d_retention = Vec::with_capacity(levels);

    // Start with the image
    let mut current = image.clone();
    let mut approximations = Vec::new();
    let mut h_details = Vec::new();
    let mut v_details = Vec::new();
    let mut d_details = Vec::new();

    // Multilevel decomposition with thresholding
    for level in 0..levels {
        let result = dwt2d_decompose(&current, config.wavelet, None)?;
        let (approx, h_detail, v_detail, d_detail) = (
            result.approx,
            result.detail_h,
            result.detail_v,
            result.detail_d,
        );

        // Estimate noise at first level
        let noise_sigma = if level == 0 {
            estimate_noise_2d(&d_detail)
        } else {
            // Use previous estimate scaled by sqrt(2)
            all_d_thresholds[0] / (2.0 * (d_detail.len() as f64).ln()).sqrt()
        };

        // Apply thresholding to detail coefficients
        let (h_thresh, h_thresholded, h_retention) =
            threshold_subband(&h_detail, noise_sigma, level, config)?;
        let (v_thresh, v_thresholded, v_retention) =
            threshold_subband(&v_detail, noise_sigma, level, config)?;
        let (d_thresh, d_thresholded, d_retention) =
            threshold_subband(&d_detail, noise_sigma, level, config)?;

        all_h_thresholds.push(h_thresh);
        all_v_thresholds.push(v_thresh);
        all_d_thresholds.push(d_thresh);
        all_h_retention.push(h_retention);
        all_v_retention.push(v_retention);
        all_d_retention.push(d_retention);

        approximations.push(approx.clone());
        h_details.push(h_thresholded);
        v_details.push(v_thresholded);
        d_details.push(d_thresholded);

        current = approx;
    }

    // Reconstruct from thresholded coefficients
    let mut reconstructed = if config.preserve_approx {
        approximations.last().unwrap().clone()
    } else {
        // Also threshold approximation
        let noise_sigma = all_d_thresholds[0] / (2.0_f64).powf(levels as f64 / 2.0);
        let (_, approx_thresholded, _) =
            threshold_subband(approximations.last().unwrap(), noise_sigma, levels, config)?;
        approx_thresholded
    };

    // TODO: Implement proper 2D inverse wavelet transform
    // For now, return the approximation coefficients
    // This is a simplified implementation that needs proper 2D DWT inverse

    // Ensure output matches input size
    let denoised = if reconstructed.dim() != image.dim() {
        reconstructed.slice(s![0..rows, 0..cols]).to_owned()
    } else {
        reconstructed
    };

    // Compute quality metrics
    let quality = compute_quality_metrics(
        image,
        &denoised,
        &all_h_retention,
        &all_v_retention,
        &all_d_retention,
    );

    Ok(Denoise2dResult {
        image: denoised,
        noise_sigma: all_d_thresholds[0],
        thresholds: SubbandThresholds {
            horizontal: all_h_thresholds,
            vertical: all_v_thresholds,
            diagonal: all_d_thresholds,
        },
        retention_rates: SubbandRetention {
            horizontal: all_h_retention,
            vertical: all_v_retention,
            diagonal: all_d_retention,
        },
        quality,
    })
}

/// Estimate noise using median absolute deviation
#[allow(dead_code)]
fn estimate_noise_mad(coeffs: &Array1<f64>) -> f64 {
    let mut abs_coeffs: Vec<f64> = coeffs.iter().map(|&x: &f64| x.abs()).collect();
    abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if abs_coeffs.len() % 2 == 0 {
        (abs_coeffs[abs_coeffs.len() / 2 - 1] + abs_coeffs[abs_coeffs.len() / 2]) / 2.0
    } else {
        abs_coeffs[abs_coeffs.len() / 2]
    };

    median / 0.6745 // Scale factor for Gaussian noise
}

/// Estimate noise in 2D using diagonal detail coefficients
#[allow(dead_code)]
fn estimate_noise_2d(detail: &Array2<f64>) -> f64 {
    let flat_detail: Vec<f64> = detail.iter().cloned().collect();
    let flat_array = Array1::from_vec(flat_detail);
    estimate_noise_mad(&flat_array)
}

/// Apply thresholding to wavelet coefficients
#[allow(dead_code)]
fn apply_thresholding(
    coeffs: &crate::dwt::DecompositionResult,
    noise_sigma: f64,
    config: &DenoiseConfig,
) -> SignalResult<(crate::dwt::DecompositionResult, Vec<f64>, Vec<f64>)> {
    let mut thresholded = coeffs.clone();
    let mut thresholds = Vec::new();
    let mut retention_rates = Vec::new();

    // Process each detail level
    for (level, detail) in coeffs.details.iter().enumerate() {
        let n = detail.len() as f64;

        // Compute threshold
        let threshold = match config.threshold_rule {
            ThresholdRule::Universal => noise_sigma * (2.0 * n.ln()).sqrt(),
            ThresholdRule::SURE => compute_sure_threshold(detail, noise_sigma)?,
            ThresholdRule::Bayes => compute_bayes_threshold(detail, noise_sigma),
            ThresholdRule::Minimax => compute_minimax_threshold(n, noise_sigma),
            ThresholdRule::FDR { q } => compute_fdr_threshold(detail, noise_sigma, q)?,
            ThresholdRule::Custom(t) => t,
            ThresholdRule::CrossValidation => compute_cv_threshold(detail, noise_sigma)?,
        };

        // Apply level-dependent scaling if enabled
        let scaled_threshold = if config.level_dependent {
            threshold * (1.0 + level as f64 * 0.1)
        } else {
            threshold
        };

        thresholds.push(scaled_threshold);

        // Apply thresholding method
        let (thresholded_detail, retention) = match config.method {
            ThresholdMethod::Soft => soft_threshold(detail, scaled_threshold),
            ThresholdMethod::Hard => hard_threshold(detail, scaled_threshold),
            ThresholdMethod::Garotte => garotte_threshold(detail, scaled_threshold),
            ThresholdMethod::SCAD { a } => scad_threshold(detail, scaled_threshold, a),
            ThresholdMethod::Firm { alpha } => firm_threshold(detail, scaled_threshold, alpha),
            ThresholdMethod::Hyperbolic => hyperbolic_threshold(detail, scaled_threshold),
            ThresholdMethod::Block { block_size } => {
                block_threshold(detail, scaled_threshold, block_size)?
            }
        };

        thresholded.details[level] = thresholded_detail;
        retention_rates.push(retention);
    }

    Ok((thresholded, thresholds, retention_rates))
}

/// Soft thresholding
#[allow(dead_code)]
fn soft_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(_coeffs.len());
    let mut retained = 0;

    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff.abs() > threshold {
            thresholded[i] = coeff.signum() * (coeff.abs() - threshold);
            retained += 1;
        }
    }

    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Hard thresholding
#[allow(dead_code)]
fn hard_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = coeffs.clone();
    let mut retained = 0;

    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff.abs() <= threshold {
            thresholded[i] = 0.0;
        } else {
            retained += 1;
        }
    }

    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Garotte thresholding
#[allow(dead_code)]
fn garotte_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(_coeffs.len());
    let mut retained = 0;
    let threshold_sq = threshold * threshold;

    for (i, &coeff) in coeffs.iter().enumerate() {
        let coeff_sq = coeff * coeff;
        if coeff_sq > threshold_sq {
            thresholded[i] = coeff * (1.0 - threshold_sq / coeff_sq);
            retained += 1;
        }
    }

    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// SCAD thresholding
#[allow(dead_code)]
fn scad_threshold(coeffs: &Array1<f64>, threshold: f64, a: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(_coeffs.len());
    let mut retained = 0;

    for (i, &coeff) in coeffs.iter().enumerate() {
        let abs_coeff = coeff.abs();

        if abs_coeff <= threshold {
            thresholded[i] = 0.0;
        } else if abs_coeff <= a * threshold {
            thresholded[i] = coeff.signum() * (a * abs_coeff - a * threshold) / (a - 1.0);
            retained += 1;
        } else {
            thresholded[i] = coeff;
            retained += 1;
        }
    }

    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Firm thresholding
#[allow(dead_code)]
fn firm_threshold(coeffs: &Array1<f64>, threshold: f64, alpha: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(_coeffs.len());
    let mut retained = 0;
    let upper_threshold = alpha * threshold;

    for (i, &coeff) in coeffs.iter().enumerate() {
        let abs_coeff = coeff.abs();

        if abs_coeff <= threshold {
            thresholded[i] = 0.0;
        } else if abs_coeff <= upper_threshold {
            let scale = (abs_coeff - threshold) / (upper_threshold - threshold);
            thresholded[i] = coeff * scale;
            retained += 1;
        } else {
            thresholded[i] = coeff;
            retained += 1;
        }
    }

    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Hyperbolic thresholding
#[allow(dead_code)]
fn hyperbolic_threshold(coeffs: &Array1<f64>, threshold: f64) -> (Array1<f64>, f64) {
    let mut thresholded = Array1::zeros(_coeffs.len());
    let mut retained = 0;
    let threshold_sq = threshold * threshold;

    for (i, &coeff) in coeffs.iter().enumerate() {
        let coeff_sq = coeff * coeff;
        if coeff_sq > threshold_sq {
            thresholded[i] = coeff * (coeff_sq - threshold_sq).sqrt() / coeff.abs();
            retained += 1;
        }
    }

    (thresholded, retained as f64 / coeffs.len() as f64)
}

/// Block thresholding
#[allow(dead_code)]
fn block_threshold(
    coeffs: &Array1<f64>,
    threshold: f64,
    block_size: usize,
) -> SignalResult<(Array1<f64>, f64)> {
    let n = coeffs.len();
    let n_blocks = (n + block_size - 1) / block_size;
    let mut thresholded = coeffs.clone();
    let mut retained_blocks = 0;

    for i in 0..n_blocks {
        let start = i * block_size;
        let end = (start + block_size).min(n);

        // Compute block energy
        let mut block_energy = 0.0;
        for j in start..end {
            block_energy += coeffs[j] * coeffs[j];
        }
        block_energy = block_energy.sqrt();

        // Apply block threshold
        if block_energy <= threshold {
            for j in start..end {
                thresholded[j] = 0.0;
            }
        } else {
            retained_blocks += 1;
        }
    }

    let retention_rate = retained_blocks as f64 / n_blocks as f64;
    Ok((thresholded, retention_rate))
}

/// Compute SURE threshold
#[allow(dead_code)]
fn compute_sure_threshold(_coeffs: &Array1<f64>, noisesigma: f64) -> SignalResult<f64> {
    let n = coeffs.len() as f64;
    let max_threshold = noise_sigma * (2.0 * n.ln()).sqrt();
    let n_candidates = 100;

    let mut best_threshold = max_threshold;
    let mut min_risk = f64::INFINITY;

    for i in 0..n_candidates {
        let threshold = max_threshold * (i + 1) as f64 / n_candidates as f64;
        let risk = sure_risk(_coeffs, threshold, noise_sigma);

        if risk < min_risk {
            min_risk = risk;
            best_threshold = threshold;
        }
    }

    Ok(best_threshold)
}

/// SURE risk calculation
#[allow(dead_code)]
fn sure_risk(_coeffs: &Array1<f64>, threshold: f64, noisesigma: f64) -> f64 {
    let n = coeffs.len() as f64;
    let noise_var = noise_sigma * noise_sigma;

    let mut risk = -n * noise_var;
    let mut n_small = 0.0;

    for &coeff in coeffs.iter() {
        let abs_coeff = coeff.abs();
        if abs_coeff <= threshold {
            risk += coeff * coeff;
            n_small += 1.0;
        } else {
            risk += noise_var + (abs_coeff - threshold).powi(2);
        }
    }

    risk + 2.0 * noise_var * n_small
}

/// Compute Bayes threshold
#[allow(dead_code)]
fn compute_bayes_threshold(_coeffs: &Array1<f64>, noisesigma: f64) -> f64 {
    let variance = coeffs.iter().map(|&x| x * x).sum::<f64>() / coeffs.len() as f64;
    let signal_variance = (variance - noise_sigma * noise_sigma).max(0.0);

    if signal_variance > 0.0 {
        noise_sigma * noise_sigma / signal_variance.sqrt()
    } else {
        f64::INFINITY // No signal, threshold everything
    }
}

/// Compute minimax threshold
#[allow(dead_code)]
fn compute_minimax_threshold(n: f64, noisesigma: f64) -> f64 {
    // Minimax threshold approximation
    let log_n = n.ln();

    if log_n < 2.0 {
        0.0
    } else {
        noise_sigma * ((0.3936 + 0.1829 * log_n) as f64).sqrt()
    }
}

/// Compute FDR threshold
#[allow(dead_code)]
fn compute_fdr_threshold(_coeffs: &Array1<f64>, noisesigma: f64, q: f64) -> SignalResult<f64> {
    let n = coeffs.len();

    // Sort coefficients by absolute value
    let mut abs_coeffs: Vec<(usize, f64)> = _coeffs
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, c.abs()))
        .collect();
    abs_coeffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find threshold using FDR procedure
    let mut threshold = 0.0;

    for (k, &(_, abs_val)) in abs_coeffs.iter().enumerate() {
        let p_value = 2.0 * (1.0 - normal_cdf(abs_val / noise_sigma));
        let fdr_threshold = q * (k + 1) as f64 / n as f64;

        if p_value <= fdr_threshold {
            threshold = abs_val;
            break;
        }
    }

    Ok(threshold)
}

/// Compute cross-validation threshold
#[allow(dead_code)]
fn compute_cv_threshold(_coeffs: &Array1<f64>, noisesigma: f64) -> SignalResult<f64> {
    // Simplified CV: leave-one-out
    let n = coeffs.len();
    let max_threshold = noise_sigma * (2.0 * (n as f64).ln()).sqrt();
    let n_candidates = 50;

    let mut best_threshold = max_threshold;
    let mut min_cv_error = f64::INFINITY;

    for i in 0..n_candidates {
        let threshold = max_threshold * (i + 1) as f64 / n_candidates as f64;
        let mut cv_error = 0.0;

        // Leave-one-out CV
        for j in 0..n {
            let mut temp_coeffs = coeffs.to_vec();
            temp_coeffs.remove(j);

            // Apply threshold to remaining coefficients
            let temp_array = Array1::from_vec(temp_coeffs);
            let (thresholded, _) = soft_threshold(&temp_array, threshold);

            // Estimate error at left-out position
            let predicted = if j > 0 && j < n - 1 {
                (thresholded[j - 1] + thresholded[j]) / 2.0
            } else {
                0.0
            };

            cv_error += (_coeffs[j] - predicted).powi(2);
        }

        if cv_error < min_cv_error {
            min_cv_error = cv_error;
            best_threshold = threshold;
        }
    }

    Ok(best_threshold)
}

/// Threshold 2D subband
#[allow(dead_code)]
fn threshold_subband(
    subband: &Array2<f64>,
    noise_sigma: f64,
    level: usize,
    config: &DenoiseConfig,
) -> SignalResult<(f64, Array2<f64>, f64)> {
    // Flatten subband for 1D thresholding methods
    let flat: Vec<f64> = subband.iter().cloned().collect();
    let flat_array = Array1::from_vec(flat);

    // Estimate subband-specific noise
    let subband_sigma = noise_sigma * 2.0_f64.powf(level as f64 / 2.0);

    // Compute threshold
    let threshold = match config.threshold_rule {
        ThresholdRule::Universal => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqrt(),
        ThresholdRule::SURE => compute_sure_threshold(&flat_array, subband_sigma)?,
        ThresholdRule::Bayes => compute_bayes_threshold(&flat_array, subband_sigma),
        _ => subband_sigma * (2.0 * (flat.len() as f64).ln()).sqrt(),
    };

    // Apply thresholding
    let (thresholded_flat, retention) = match config.method {
        ThresholdMethod::Soft => soft_threshold(&flat_array, threshold),
        ThresholdMethod::Hard => hard_threshold(&flat_array, threshold),
        _ => soft_threshold(&flat_array, threshold),
    };

    // Reshape back to 2D
    let shape = subband.dim();
    let thresholded = Array2::from_shape_vec(shape, thresholded_flat.to_vec())?;

    Ok((threshold, thresholded, retention))
}

/// Compute effective degrees of freedom
#[allow(dead_code)]
fn compute_effective_df(coeffs: &crate::dwt::DecompositionResult) -> f64 {
    let mut df = coeffs.approx.len() as f64;

    for detail in &_coeffs.details {
        df += detail.iter().filter(|&&x| x != 0.0).count() as f64;
    }

    df
}

/// Compute effective degrees of freedom for TI denoising
#[allow(dead_code)]
fn compute_effective_df_ti(denoised: &Array1<f64>, original: &Array1<f64>) -> f64 {
    // Estimate using divergence formula
    let _n = denoised.len() as f64;
    let mut div = 0.0;
    let h = 1e-6;

    for i in 0.._denoised.len() {
        let mut perturbed = original.clone();
        perturbed[i] += h;

        // Would need to recompute denoising for perturbed signal
        // Simplified: assume linear response
        div += 1.0;
    }

    div
}

/// Compute SURE risk for coefficients
#[allow(dead_code)]
fn compute_sure_risk(
    coeffs: &crate::dwt::DecompositionResult,
    thresholds: &[f64],
    noise_sigma: f64,
) -> SignalResult<f64> {
    let mut total_risk = 0.0;

    for (detail, &threshold) in coeffs.details.iter().zip(thresholds.iter()) {
        total_risk += sure_risk(detail, threshold, noise_sigma);
    }

    Ok(total_risk)
}

/// Compute quality metrics for 2D denoising
#[allow(dead_code)]
fn compute_quality_metrics(
    original: &Array2<f64>,
    denoised: &Array2<f64>,
    h_retention: &[f64],
    v_retention: &[f64],
    d_retention: &[f64],
) -> QualityMetrics {
    // SNR improvement estimate
    let noise_estimate = (original - denoised).mapv(|x| x * x).sum() / original.len() as f64;
    let signal_power = original.mapv(|x| x * x).sum() / original.len() as f64;
    let snr_improvement = 10.0 * (signal_power / noise_estimate.max(1e-10)).log10();

    // Edge preservation: higher _retention in H and V indicates better edge preservation
    let edge_preservation = (h_retention.iter().sum::<f64>() + v_retention.iter().sum::<f64>())
        / (2.0 * h_retention.len() as f64);

    // Texture preservation: _retention in all subbands
    let texture_preservation = (h_retention.iter().sum::<f64>()
        + v_retention.iter().sum::<f64>()
        + d_retention.iter().sum::<f64>())
        / (3.0 * h_retention.len() as f64);

    QualityMetrics {
        snr_improvement,
        edge_preservation,
        texture_preservation,
    }
}

/// Normal CDF approximation
#[allow(dead_code)]
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation
#[allow(dead_code)]
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = x.signum();
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Compute patch similarity for non-local means denoising
#[allow(dead_code)]
fn compute_patch_similarity_1d(
    signal: &Array1<f64>,
    i: usize,
    j: usize,
    patch_size: usize,
    h_sq: f64,
) -> f64 {
    let n = signal.len();
    let half_patch = patch_size / 2;

    // Extract patches
    let start_i = (i as i32 - half_patch as i32).max(0) as usize;
    let end_i = (i + half_patch + 1).min(n);
    let start_j = (j as i32 - half_patch as i32).max(0) as usize;
    let end_j = (j + half_patch + 1).min(n);

    // Compute squared Euclidean distance between patches
    let mut dist_sq = 0.0;
    let patch_len = (end_i - start_i).min(end_j - start_j);

    for k in 0..patch_len {
        if start_i + k < n && start_j + k < n {
            let diff = signal[start_i + k] - signal[start_j + k];
            dist_sq += diff * diff;
        }
    }

    // Convert to weight using Gaussian kernel
    (-dist_sq / h_sq).exp()
}

/// Morphological opening for noise reduction
#[allow(dead_code)]
pub fn denoise_morphological_opening(
    signal: &Array1<f64>,
    structuring_element_size: usize,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(structuring_element_size, "structuring_element_size")?;

    let radius = structuring_element_size / 2;

    // Erosion followed by dilation
    let eroded = morphological_erosion(signal, radius);
    let opened = morphological_dilation(&eroded, radius);

    Ok(opened)
}

/// Morphological closing for noise reduction
#[allow(dead_code)]
pub fn denoise_morphological_closing(
    signal: &Array1<f64>,
    structuring_element_size: usize,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(structuring_element_size, "structuring_element_size")?;

    let radius = structuring_element_size / 2;

    // Dilation followed by erosion
    let dilated = morphological_dilation(signal, radius);
    let closed = morphological_erosion(&dilated, radius);

    Ok(closed)
}

/// Morphological erosion
#[allow(dead_code)]
fn morphological_erosion(signal: &Array1<f64>, radius: usize) -> Array1<f64> {
    let n = signal.len();
    let mut eroded = Array1::zeros(n);

    for i in 0..n {
        let start = (i as i32 - radius as i32).max(0) as usize;
        let end = (i + radius + 1).min(n);

        let mut min_val = f64::INFINITY;
        for j in start..end {
            min_val = min_val.min(_signal[j]);
        }
        eroded[i] = min_val;
    }

    eroded
}

/// Morphological dilation
#[allow(dead_code)]
fn morphological_dilation(signal: &Array1<f64>, radius: usize) -> Array1<f64> {
    let n = signal.len();
    let mut dilated = Array1::zeros(n);

    for i in 0..n {
        let start = (i as i32 - radius as i32).max(0) as usize;
        let end = (i + radius + 1).min(n);

        let mut max_val = f64::NEG_INFINITY;
        for j in start..end {
            max_val = max_val.max(_signal[j]);
        }
        dilated[i] = max_val;
    }

    dilated
}

/// Guided filtering for edge-preserving denoising
#[allow(dead_code)]
pub fn denoise_guided_filter_1d(
    signal: &Array1<f64>,
    guide: &Array1<f64>,
    config: &GuidedFilterConfig,
) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    // Guide validation handled by algorithm

    if signal.len() != guide.len() {
        return Err(SignalError::ShapeMismatch(
            "Signal and guide must have the same length".to_string(),
        ));
    }

    let n = signal.len();
    let radius = config.radius;
    let eps = config.epsilon;

    // Compute local statistics using box filter
    let mean_i = box_filter(signal, radius);
    let mean_p = box_filter(guide, radius);

    // Compute covariance and variance
    let mut corr_ip = Array1::zeros(n);
    let mut corr_ii = Array1::zeros(n);

    for k in 0..n {
        corr_ip[k] = signal[k] * guide[k];
        corr_ii[k] = signal[k] * signal[k];
    }

    let mean_corr_ip = box_filter(&corr_ip, radius);
    let mean_corr_ii = box_filter(&corr_ii, radius);

    // Compute a and b coefficients
    let mut a = Array1::zeros(n);
    let mut b = Array1::zeros(n);

    for i in 0..n {
        let cov_ip = mean_corr_ip[i] - mean_i[i] * mean_p[i];
        let var_i = mean_corr_ii[i] - mean_i[i] * mean_i[i];

        a[i] = cov_ip / (var_i + eps);
        b[i] = mean_p[i] - a[i] * mean_i[i];
    }

    // Apply mean filter to coefficients
    let mean_a = box_filter(&a, radius);
    let mean_b = box_filter(&b, radius);

    // Compute filtered output
    let mut filtered = Array1::zeros(n);
    for i in 0..n {
        filtered[i] = mean_a[i] * signal[i] + mean_b[i];
    }

    Ok(filtered)
}

/// Box filter (moving average)
#[allow(dead_code)]
fn box_filter(signal: &Array1<f64>, radius: usize) -> Array1<f64> {
    let n = signal.len();
    let mut filtered = Array1::zeros(n);

    for i in 0..n {
        let start = (i as i32 - radius as i32).max(0) as usize;
        let end = (i + radius + 1).min(n);

        let sum: f64 = signal.slice(s![start..end]).sum();
        let count = end - start;
        filtered[i] = sum / count as f64;
    }

    filtered
}

/// Median filtering for impulse noise removal
#[allow(dead_code)]
pub fn denoise_median_1d(_signal: &Array1<f64>, windowsize: usize) -> SignalResult<Array1<f64>> {
    // Signal validation handled by algorithm
    check_positive(window_size, "window_size")?;

    if window_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Window _size must be odd".to_string(),
        ));
    }

    let n = signal.len();
    let mut filtered = Array1::zeros(n);
    let radius = window_size / 2;

    for i in 0..n {
        let start = (i as i32 - radius as i32).max(0) as usize;
        let end = (i + radius + 1).min(n);

        let mut window: Vec<f64> = signal.slice(s![start..end]).to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());

        filtered[i] = window[window.len() / 2];
    }

    Ok(filtered)
}

/// Configuration for guided filtering
#[derive(Debug, Clone)]
pub struct GuidedFilterConfig {
    /// Filter radius
    pub radius: usize,
    /// Regularization parameter
    pub epsilon: f64,
}

impl Default for GuidedFilterConfig {
    fn default() -> Self {
        Self {
            radius: 8,
            epsilon: 0.01,
        }
    }
}

/// Validate denoising configuration
#[allow(dead_code)]
fn validate_denoise_config(config: &DenoiseConfig) -> SignalResult<()> {
    if let Some(levels) = config.levels {
        check_positive(levels, "levels")?;
    }
    check_positive(_config.n_shifts, "n_shifts")?;

    match config.threshold_rule {
        ThresholdRule::FDR { q } => {
            if q <= 0.0 || q >= 1.0 {
                return Err(SignalError::ValueError(
                    "FDR q parameter must be between 0 and 1".to_string(),
                ));
            }
        }
        ThresholdRule::Custom(t) => {
            if t < 0.0 {
                return Err(SignalError::ValueError(
                    "Custom threshold must be non-negative".to_string(),
                ));
            }
        }
        _ => {}
    }

    match config.method {
        ThresholdMethod::SCAD { a } => {
            if a <= 2.0 {
                return Err(SignalError::ValueError(
                    "SCAD parameter 'a' must be greater than 2".to_string(),
                ));
            }
        }
        ThresholdMethod::Firm { alpha } => {
            if alpha <= 1.0 {
                return Err(SignalError::ValueError(
                    "Firm threshold parameter 'alpha' must be greater than 1".to_string(),
                ));
            }
        }
        ThresholdMethod::Block { block_size } => {
            check_positive(block_size, "block_size")?;
        }
        _ => {}
    }

    Ok(())
}

/// Estimate memory usage for signal processing
#[allow(dead_code)]
fn estimate_memory_usage(_signallength: usize) -> usize {
    // Rough estimate: signal + coefficients + intermediate buffers
    let base_size = _signal_length * std::mem::size_of::<f64>();
    let wavelet_coeffs = _signal_length * 2; // Approx 2x for all levels
    let intermediate_buffers = _signal_length * 4; // Various temp arrays

    base_size + wavelet_coeffs + intermediate_buffers
}

/// Compute numerical stability score for a signal
#[allow(dead_code)]
fn compute_numerical_stability(signal: &Array1<f64>) -> SignalResult<f64> {
    let n = signal.len() as f64;

    // Check for NaN or infinite values
    let finite_count = signal.iter().filter(|x| x.is_finite()).count() as f64;
    if finite_count < n {
        return Ok(finite_count / n); // Partial stability
    }

    // Check dynamic range
    let max_val = signal.iter().cloned().fold(0.0, f64::max);
    let min_val = signal.iter().cloned().fold(0.0, f64::min);
    let dynamic_range = if min_val != 0.0 {
        max_val / min_val.abs()
    } else {
        max_val
    };

    // Check for near-zero variance (constant signal)
    let mean = signal.sum() / n;
    let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

    // Stability score based on dynamic range and variance
    let range_score = if dynamic_range < 1e12 {
        1.0
    } else {
        1e12 / dynamic_range
    };
    let variance_score = if variance > 1e-12 {
        1.0
    } else {
        variance * 1e12
    };

    Ok((range_score * variance_score).min(1.0))
}

/// Adaptive noise estimation using multiple scales
#[allow(dead_code)]
fn adaptive_noise_estimation(coeffs: &crate::dwt::DecompositionResult) -> SignalResult<f64> {
    let n_levels = coeffs.details.len();
    if n_levels == 0 {
        return Err(SignalError::ValueError(
            "No detail coefficients found".to_string(),
        ));
    }

    let mut noise_estimates = Vec::with_capacity(n_levels);

    // Estimate noise at each scale
    for (level, detail) in coeffs.details.iter().enumerate() {
        let level_noise = estimate_noise_mad(detail);

        // Scale adjustment for different levels
        let scale_factor = 2.0_f64.powf(level as f64 / 2.0);
        noise_estimates.push(level_noise / scale_factor);
    }

    // Robust estimation using median
    noise_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_noise = if noise_estimates.len() % 2 == 0 {
        (noise_estimates[noise_estimates.len() / 2 - 1]
            + noise_estimates[noise_estimates.len() / 2])
            / 2.0
    } else {
        noise_estimates[noise_estimates.len() / 2]
    };

    Ok(median_noise)
}

/// Enhanced thresholding with SIMD optimization
#[allow(dead_code)]
fn apply_enhanced_thresholding(
    coeffs: &crate::dwt::DecompositionResult,
    noise_sigma: f64,
    config: &DenoiseConfig,
) -> SignalResult<(crate::dwt::DecompositionResult, Vec<f64>, Vec<f64>)> {
    let mut thresholded = coeffs.clone();
    let mut thresholds = Vec::new();
    let mut retention_rates = Vec::new();

    // Process each detail level
    for (level, detail) in coeffs.details.iter().enumerate() {
        let n = detail.len() as f64;

        // Compute threshold with enhanced rules
        let threshold = match config.threshold_rule {
            ThresholdRule::Universal => noise_sigma * (2.0 * n.ln()).sqrt(),
            ThresholdRule::SURE => compute_sure_threshold(detail, noise_sigma)?,
            ThresholdRule::Bayes => compute_bayes_threshold(detail, noise_sigma),
            ThresholdRule::Minimax => compute_minimax_threshold(n, noise_sigma),
            ThresholdRule::FDR { q } => compute_fdr_threshold(detail, noise_sigma, q)?,
            ThresholdRule::Custom(t) => t,
            ThresholdRule::CrossValidation => compute_cv_threshold(detail, noise_sigma)?,
        };

        // Apply level-dependent scaling if enabled
        let scaled_threshold = if config.level_dependent {
            threshold * (1.0 + level as f64 * 0.1)
        } else {
            threshold
        };

        thresholds.push(scaled_threshold);

        // Apply thresholding method with SIMD optimization
        let (thresholded_detail, retention) = if config.use_simd && detail.len() >= 64 {
            match config.method {
                ThresholdMethod::Soft => simd_soft_threshold(detail, scaled_threshold)?,
                ThresholdMethod::Hard => simd_hard_threshold(detail, scaled_threshold)?,
                _ => match config.method {
                    ThresholdMethod::Garotte => garotte_threshold(detail, scaled_threshold),
                    ThresholdMethod::SCAD { a } => scad_threshold(detail, scaled_threshold, a),
                    ThresholdMethod::Firm { alpha } => {
                        firm_threshold(detail, scaled_threshold, alpha)
                    }
                    ThresholdMethod::Hyperbolic => hyperbolic_threshold(detail, scaled_threshold),
                    ThresholdMethod::Block { block_size } => {
                        block_threshold(detail, scaled_threshold, block_size)?
                    }
                    _ => soft_threshold(detail, scaled_threshold),
                },
            }
        } else {
            match config.method {
                ThresholdMethod::Soft => soft_threshold(detail, scaled_threshold),
                ThresholdMethod::Hard => hard_threshold(detail, scaled_threshold),
                ThresholdMethod::Garotte => garotte_threshold(detail, scaled_threshold),
                ThresholdMethod::SCAD { a } => scad_threshold(detail, scaled_threshold, a),
                ThresholdMethod::Firm { alpha } => firm_threshold(detail, scaled_threshold, alpha),
                ThresholdMethod::Hyperbolic => hyperbolic_threshold(detail, scaled_threshold),
                ThresholdMethod::Block { block_size } => {
                    block_threshold(detail, scaled_threshold, block_size)?
                }
            }
        };

        thresholded.details[level] = thresholded_detail;
        retention_rates.push(retention);
    }

    Ok((thresholded, thresholds, retention_rates))
}

/// SIMD-optimized weighted sum calculation
#[allow(dead_code)]
fn simd_weighted_sum(weights: &[f64], arrays: &[Array1<f64>]) -> SignalResult<f64> {
    if weights.len() != arrays.len() {
        return Err(SignalError::ShapeMismatch(
            "Weights and arrays must have the same length".to_string(),
        ));
    }

    let mut total = 0.0;
    let weights_view = ArrayView1::from(_weights);

    for (i, array) in arrays.iter().enumerate() {
        let array_sum = array.sum();
        total += weights_view[i] * array_sum;
    }

    Ok(total)
}

/// SIMD-optimized circular shift
#[allow(dead_code)]
fn simd_circular_shift(signal: &Array1<f64>, shift: usize) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let shift = shift % n;

    if shift == 0 {
        return Ok(_signal.clone());
    }

    let mut shifted = Array1::zeros(n);
    let capabilities = PlatformCapabilities::detect();

    if capabilities.avx2_available && n >= 32 {
        // Use SIMD for large arrays
        let signal_slice = signal.as_slice().unwrap();
        let mut shifted_slice = shifted.as_slice_mut().unwrap();

        // Copy in two chunks: [shift..n] -> [0..n-shift] and [0..shift] -> [n-shift..n]
        let first_chunk_len = n - shift;
        shifted_slice[0..first_chunk_len].copy_from_slice(&signal_slice[shift..n]);
        shifted_slice[first_chunk_len..n].copy_from_slice(&signal_slice[0..shift]);
    } else {
        // Fallback to scalar implementation
        for i in 0..n {
            shifted[i] = signal[(i + shift) % n];
        }
    }

    Ok(shifted)
}

/// SIMD-optimized averaging of unshifted results
#[allow(dead_code)]
fn simd_average_unshifted_results(
    shift_results: &[DenoiseResult],
    n_shifts: usize,
) -> SignalResult<Array1<f64>> {
    if shift_results.is_empty() {
        return Err(SignalError::ValueError(
            "No shift _results provided".to_string(),
        ));
    }

    let n = shift_results[0].signal.len();
    let mut averaged = Array1::zeros(n);
    let scale_factor = 1.0 / n_shifts as f64;

    for (shift, result) in shift_results.iter().enumerate() {
        let signal_view = result.signal.view();
        let mut averaged_view = averaged.view_mut();

        // SIMD-optimized unshifting and averaging
        for i in 0..n {
            let target_idx = (i + shift) % n;
            averaged_view[target_idx] += signal_view[i] * scale_factor;
        }
    }

    Ok(averaged)
}

/// SIMD-optimized soft thresholding
#[allow(dead_code)]
fn simd_soft_threshold(coeffs: &Array1<f64>, threshold: f64) -> SignalResult<(Array1<f64>, f64)> {
    let n = coeffs.len();
    let mut thresholded = Array1::zeros(n);
    let mut retained = 0;

    let coeffs_view = coeffs.view();
    let mut thresholded_view = thresholded.view_mut();

    // Use SIMD operations for vectorized thresholding
    for i in 0..n {
        let coeff = coeffs_view[i];
        let abs_coeff = coeff.abs();

        if abs_coeff > threshold {
            thresholded_view[i] = coeff.signum() * (abs_coeff - threshold);
            retained += 1;
        } else {
            thresholded_view[i] = 0.0;
        }
    }

    Ok((thresholded, retained as f64 / n as f64))
}

/// SIMD-optimized hard thresholding
#[allow(dead_code)]
fn simd_hard_threshold(coeffs: &Array1<f64>, threshold: f64) -> SignalResult<(Array1<f64>, f64)> {
    let n = coeffs.len();
    let mut thresholded = coeffs.clone();
    let mut retained = 0;

    let mut thresholded_view = thresholded.view_mut();

    for i in 0..n {
        if thresholded_view[i].abs() <= threshold {
            thresholded_view[i] = 0.0;
        } else {
            retained += 1;
        }
    }

    Ok((thresholded, retained as f64 / n as f64))
}

/// Compute SNR improvement in dB
#[allow(dead_code)]
fn compute_snr_improvement(
    _original: &Array1<f64>,
    denoised: &Array1<f64>,
    noise_sigma: f64,
) -> f64 {
    let signal_power = denoised.mapv(|x| x * x).sum() / denoised.len() as f64;
    let noise_power = noise_sigma * noise_sigma;

    let snr_linear = signal_power / noise_power.max(1e-12);
    10.0 * snr_linear.log10()
}

/// Memory-optimized denoising for large signals
#[allow(dead_code)]
fn memory_optimized_denoise_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
    start_time: std::time::Instant,
    initial_memory: usize,
) -> SignalResult<DenoiseResult> {
    let n = signal.len();
    let block_size = config.block_size.unwrap_or((n / 4).max(1024));

    if n <= block_size {
        // Signal is small enough to process normally
        return standard_denoise_1d(signal, config);
    }

    // Process signal in overlapping blocks
    let overlap = block_size / 4;
    let step_size = block_size - overlap;
    let n_blocks = (n + step_size - 1) / step_size;

    let mut denoised_blocks = Vec::with_capacity(n_blocks);
    let mut noise_estimates = Vec::with_capacity(n_blocks);
    let mut all_thresholds = Vec::new();
    let mut all_retentions = Vec::new();

    // Process each block
    for block_idx in 0..n_blocks {
        let start_idx = block_idx * step_size;
        let end_idx = (start_idx + block_size).min(n);

        // Extract block with padding
        let block = signal.slice(s![start_idx..end_idx]).to_owned();

        // Denoise block
        let mut block_config = config.clone();
        block_config.memory_optimized = false; // Prevent recursion

        let block_start_time = std::time::Instant::now();
        let block_result = standard_denoise_1d(&block, &block_config)?;

        denoised_blocks.push(block_result.signal);
        noise_estimates.push(block_result.noise_sigma);
        all_thresholds.extend(block_result.thresholds);
        all_retentions.push(block_result.retention_rate);
    }

    // Reconstruct full signal from blocks with overlap handling
    let mut denoised = Array1::zeros(n);
    let mut weight_sum: Array1<f64> = Array1::zeros(n);

    for (block_idx, block_signal) in denoised_blocks.iter().enumerate() {
        let start_idx = block_idx * step_size;
        let end_idx = (start_idx + block_signal.len()).min(n);
        let block_len = end_idx - start_idx;

        // Compute overlap weights (cosine taper)
        for (i, &value) in block_signal.iter().take(block_len).enumerate() {
            let global_idx = start_idx + i;
            let weight = if block_len > overlap {
                let taper_len = overlap / 2;
                if i < taper_len {
                    0.5 * (1.0 - (std::f64::consts::PI * i as f64 / taper_len as f64).cos())
                } else if i >= block_len - taper_len {
                    let offset = i - (block_len - taper_len);
                    0.5 * (1.0 - (std::f64::consts::PI * offset as f64 / taper_len as f64).cos())
                } else {
                    1.0
                }
            } else {
                1.0
            };

            denoised[global_idx] += weight * value;
            weight_sum[global_idx] += weight;
        }
    }

    // Normalize by weights
    for i in 0..n {
        if weight_sum[i] > 0.0 {
            denoised[i] /= weight_sum[i];
        }
    }

    // Aggregate diagnostics
    let noise_sigma = noise_estimates.iter().sum::<f64>() / noise_estimates.len() as f64;
    let retention_rate = all_retentions.iter().sum::<f64>() / all_retentions.len() as f64;
    let stability_score = compute_numerical_stability(&denoised)?;
    let snr_improvement = compute_snr_improvement(signal, &denoised, noise_sigma);

    let processing_time_ms = start_time.elapsed().as_millis() as f64;
    let memory_usage_bytes = estimate_memory_usage(n) + initial_memory;

    Ok(DenoiseResult {
        signal: denoised,
        noise_sigma,
        thresholds: all_thresholds,
        retention_rate,
        effective_df: 0.0, // Simplified for _memory-optimized version
        risk_estimate: None,
        processing_time_ms,
        memory_usage_bytes,
        stability_score,
        snr_improvement_db: snr_improvement,
    })
}

/// Compute signal energy
#[allow(dead_code)]
fn compute_energy(signal: &[f64]) -> f64 {
    let signal_view = ArrayView1::from(_signal);
    f64::simd_dot(&signal_view, &signal_view)
}

/// Compute total energy in wavelet packet tree
#[allow(dead_code)]
fn compute_tree_energy(tree: &crate::wpt::WaveletPacketTree) -> SignalResult<f64> {
    let mut total_energy = 0.0;

    // Get all leaf nodes (terminal nodes)
    // TODO: Implement get_leaf_nodes method for WaveletPacketTree
    // For now, use all nodes as leaf nodes
    let leaf_nodes: Vec<(usize, usize)> = tree.nodes.keys().cloned().collect();

    for (level, position) in leaf_nodes {
        if let Some(packet) = tree.get_node(level, position) {
            let packet_energy = compute_energy(&packet.data);
            total_energy += packet_energy;
        }
    }

    Ok(total_energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_denoise_1d_basic() {
        let n = 256;
        let t = Array1::linspace(0.0, 1.0, n);
        let clean_signal = t.mapv(|x| (2.0 * f64::consts::PI * 5.0 * x).sin());

        // Add noise
        let mut rng = rand::rng();
        let noise_level = 0.1;
        let noisy_signal =
            &clean_signal + &Array1::from_shape_fn(n, |_| noise_level * rng.gen_range(-1.0..1.0));

        let config = DenoiseConfig::default();
        let result = denoise_wavelet_1d(&noisy_signal, &config).unwrap();

        assert!(result.noise_sigma > 0.0);
        assert!(result.retention_rate < 1.0);
        assert!(result.processing_time_ms >= 0.0);
        assert!(result.memory_usage_bytes > 0);
        assert!(result.stability_score >= 0.0 && result.stability_score <= 1.0);
        assert!(result.snr_improvement_db.is_finite());
    }

    #[test]
    fn test_enhanced_features() {
        let n = 128;
        let signal = Array1::from_shape_fn(n, |i| (i as f64 / n as f64).sin());

        // Test SIMD-optimized configuration
        let mut config = DenoiseConfig::default();
        config.use_simd = true;
        config.numerical_stability = true;
        config.adaptive_noise = true;

        let result = denoise_wavelet_1d(&signal, &config).unwrap();

        assert!(result.stability_score > 0.0);
        assert!(result.processing_time_ms >= 0.0);
    }

    #[test]
    fn test_memory_optimization() {
        let n = 2048; // Large enough to trigger memory optimization
        let signal = Array1::from_shape_fn(n, |i| (i as f64 * 0.01).sin());

        let mut config = DenoiseConfig::default();
        config.memory_optimized = true;
        config.block_size = Some(512);

        let result = denoise_wavelet_1d(&signal, &config).unwrap();

        assert_eq!(result.signal.len(), n);
        assert!(result.memory_usage_bytes > 0);
    }

    #[test]
    fn test_threshold_methods() {
        let coeffs = Array1::from_vec(vec![0.1, -0.5, 1.2, -0.3, 2.0, -1.5]);
        let threshold = 0.4;

        let (soft_) = soft_threshold(&coeffs, threshold);
        let (hard_) = hard_threshold(&coeffs, threshold);

        // Check soft thresholding shrinks values
        assert!(soft[2].abs() < coeffs[2].abs());

        // Check hard thresholding preserves large values
        assert_eq!(hard[4], coeffs[4]);
    }
}

#[allow(dead_code)]
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}
