//! Alternative denoising algorithms
//!
//! This module provides implementations of various denoising algorithms beyond
//! basic wavelet denoising, including non-local means, total variation, bilateral
//! filtering, Wiener filtering, and adaptive LMS filtering.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_positive;
use scirs2_fft::{fft, ifft};

/// Advanced non-local means denoising for 1D signals
pub fn denoise_non_local_means_1d(
    signal: &Array1<f64>,
    config: &NonLocalMeansConfig,
) -> SignalResult<Array1<f64>> {
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
pub fn denoise_total_variation_1d(
    signal: &Array1<f64>,
    config: &TotalVariationConfig,
) -> SignalResult<Array1<f64>> {
    check_positive(config.lambda, "lambda")?;

    let n = signal.len();
    let mut denoised = signal.clone();
    let lambda = config.lambda;
    let max_iter = config.max_iterations;
    let tolerance = config.tolerance;

    // Iterative solution using projected gradient descent
    for _iter in 0..max_iter {
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
pub fn denoise_bilateral_1d(
    signal: &Array1<f64>,
    config: &BilateralConfig,
) -> SignalResult<Array1<f64>> {
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
pub fn denoise_wiener_1d(signal: &Array1<f64>, config: &WienerConfig) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Pad to power of 2 for efficient FFT
    let padded_size = n.next_power_of_two();
    let mut padded_signal = vec![0.0; padded_size];
    padded_signal[..n].copy_from_slice(signal.as_slice().unwrap());

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
pub fn denoise_adaptive_lms(
    signal: &Array1<f64>,
    config: &AdaptiveLMSConfig,
) -> SignalResult<Array1<f64>> {
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
    if let Some(ref initial_weights) = config.initial_weights {
        weights = initial_weights.clone();
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

/// Guided filtering for 1D signals
pub fn denoise_guided_filter_1d(
    signal: &Array1<f64>,
    guide: &Array1<f64>,
    config: &GuidedFilterConfig,
) -> SignalResult<Array1<f64>> {
    if signal.len() != guide.len() {
        return Err(SignalError::ValueError(
            "Signal and guide must have the same length".to_string(),
        ));
    }

    let n = signal.len();
    let radius = config.radius;
    let epsilon = config.epsilon;

    // Compute means using box filter
    let mean_guide = box_filter(guide, radius);
    let mean_signal = box_filter(signal, radius);

    // Compute correlation and variance
    let mut corr_guide_signal = Array1::zeros(n);
    let mut var_guide = Array1::zeros(n);

    for i in 0..n {
        corr_guide_signal[i] = guide[i] * signal[i];
        var_guide[i] = guide[i] * guide[i];
    }

    let mean_corr = box_filter(&corr_guide_signal, radius);
    let mean_var_guide = box_filter(&var_guide, radius);

    // Compute coefficients a and b
    let mut a = Array1::zeros(n);
    let mut b = Array1::zeros(n);

    for i in 0..n {
        let cov_guide_signal = mean_corr[i] - mean_guide[i] * mean_signal[i];
        let variance_guide = mean_var_guide[i] - mean_guide[i] * mean_guide[i];

        a[i] = cov_guide_signal / (variance_guide + epsilon);
        b[i] = mean_signal[i] - a[i] * mean_guide[i];
    }

    // Average coefficients over neighborhood
    let mean_a = box_filter(&a, radius);
    let mean_b = box_filter(&b, radius);

    // Compute output
    let mut output = Array1::zeros(n);
    for i in 0..n {
        output[i] = mean_a[i] * guide[i] + mean_b[i];
    }

    Ok(output)
}

/// Median filtering for 1D signals
pub fn denoise_median_1d(signal: &Array1<f64>, window_size: usize) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Window size must be odd".to_string(),
        ));
    }

    let n = signal.len();
    let mut denoised = Array1::zeros(n);
    let half_window = window_size / 2;

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);

        let mut window: Vec<f64> = signal.slice(s![start..end]).to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());

        denoised[i] = window[window.len() / 2];
    }

    Ok(denoised)
}

/// Compute patch similarity for non-local means
fn compute_patch_similarity_1d(
    signal: &Array1<f64>,
    i: usize,
    j: usize,
    patch_size: usize,
    h_sq: f64,
) -> f64 {
    let n = signal.len();
    let half_patch = patch_size / 2;

    let start_i = i.saturating_sub(half_patch);
    let end_i = (i + half_patch + 1).min(n);
    let start_j = j.saturating_sub(half_patch);
    let end_j = (j + half_patch + 1).min(n);

    // Compute patch difference
    let mut diff_sum = 0.0;
    let mut count = 0;

    for (ki, kj) in (start_i..end_i).zip(start_j..end_j) {
        let diff = signal[ki] - signal[kj];
        diff_sum += diff * diff;
        count += 1;
    }

    if count > 0 {
        let normalized_diff = diff_sum / count as f64;
        (-normalized_diff / h_sq).exp()
    } else {
        1.0
    }
}

/// Box filter (moving average) implementation
fn box_filter(signal: &Array1<f64>, radius: usize) -> Array1<f64> {
    let n = signal.len();
    let mut filtered = Array1::zeros(n);
    let window_size = 2 * radius + 1;

    for i in 0..n {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(n);

        let sum: f64 = signal.slice(s![start..end]).iter().sum();
        let count = end - start;
        filtered[i] = sum / count as f64;
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_local_means() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0]);
        let config = NonLocalMeansConfig::default();
        let result = denoise_non_local_means_1d(&signal, &config);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_total_variation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let config = TotalVariationConfig::default();
        let result = denoise_total_variation_1d(&signal, &config);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_bilateral_filter() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let config = BilateralConfig::default();
        let result = denoise_bilateral_1d(&signal, &config);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_wiener_filter() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]);
        let config = WienerConfig::default();
        let result = denoise_wiener_1d(&signal, &config);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_adaptive_lms() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
        let config = AdaptiveLMSConfig {
            filter_length: 4, // Use a smaller filter length that works with our signal size
            ..Default::default()
        };
        let result = denoise_adaptive_lms(&signal, &config);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_guided_filter() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let guide = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let config = GuidedFilterConfig::default();
        let result = denoise_guided_filter_1d(&signal, &guide, &config);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_median_filter() {
        let signal = Array1::from_vec(vec![1.0, 10.0, 3.0, 2.0, 1.0]);
        let result = denoise_median_1d(&signal, 3);
        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.len(), signal.len());
        // Median filter should reduce the spike
        assert!(denoised[1] < signal[1]);
    }
}
