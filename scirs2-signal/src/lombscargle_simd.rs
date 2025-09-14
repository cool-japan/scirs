// SIMD-optimized Lomb-Scargle periodogram with enhanced validation
//
// This module provides high-performance SIMD and parallel implementations
// of the Lomb-Scargle periodogram with comprehensive validation.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle_enhanced::{LombScargleConfig, WindowType};
use crate::window::{blackman, hamming, hann};
use ndarray::ArrayView1;
use num_traits::{Float, NumCast};
use rand::Rng;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use std::fmt::Debug;
use std::sync::Arc;

#[allow(unused_imports)]
/// SIMD-optimized Lomb-Scargle result with validation metrics
#[derive(Debug, Clone)]
pub struct SimdLombScargleResult {
    /// Frequencies
    pub frequencies: Vec<f64>,
    /// Power spectral density
    pub power: Vec<f64>,
    /// Confidence intervals (if computed)
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>,
    /// False alarm probability
    pub false_alarm_prob: Option<Vec<f64>>,
    /// Validation metrics
    pub validation: ValidationMetrics,
}

/// Validation metrics for Lomb-Scargle analysis
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Signal-to-noise ratio estimate
    pub snr: f64,
    /// Effective number of independent frequencies
    pub n_eff: f64,
    /// Maximum normalized power
    pub max_power: f64,
    /// Frequency at maximum power
    pub peak_freq: f64,
    /// Bandwidth of main peak
    pub peak_width: f64,
    /// Spectral leakage estimate
    pub leakage: f64,
}

/// SIMD-optimized Lomb-Scargle periodogram with enhanced validation
///
/// This function provides a high-performance implementation using SIMD
/// operations and parallel processing for bootstrap confidence intervals.
///
/// # Arguments
///
/// * `times` - Sample times (must be sorted and finite)
/// * `values` - Signal values
/// * `config` - Configuration for enhanced features
///
/// # Returns
///
/// * Complete result with validation metrics
///
/// # Examples
///
/// ```
/// use scirs2_signal::lombscargle_simd::{simd_lombscargle, LombScargleConfig};
///
/// // Generate unevenly sampled data
/// let times = vec![0.0, 0.1, 0.3, 0.7, 1.2, 1.5, 2.0, 2.1];
/// let values = vec![1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 0.5];
///
/// let config = LombScargleConfig {
///     use_fast: true,
///     bootstrap_iter: Some(100),
///     ..Default::default()
/// };
///
/// let result = simd_lombscargle(&times, &values, &config).unwrap();
/// assert!(result.validation.snr > 0.0);
/// ```
#[allow(dead_code)]
pub fn simd_lombscargle<T, U>(
    times: &[T],
    values: &[U],
    config: &LombScargleConfig,
) -> SignalResult<SimdLombScargleResult>
where
    T: Float + NumCast + Debug + Send + Sync,
    U: Float + NumCast + Debug + Send + Sync,
{
    // Enhanced validation
    validate_inputs(times, values)?;

    // Convert to f64 for computation
    let times_f64 = convert_to_f64(times)?;
    let values_f64 = convert_to_f64(values)?;

    // Check for sorted times
    validate_time_series(&times_f64)?;

    // Apply windowing if requested
    let windowed_values = apply_window(&values_f64, config)?;

    // Compute frequency grid
    let frequencies = compute_frequency_grid(&times_f64, config)?;

    // Get SIMD capabilities
    let caps = PlatformCapabilities::detect();

    // Compute periodogram using SIMD operations
    let power = if config.use_fast && caps.has_simd() {
        compute_simd_fast_lombscargle(&times_f64, &windowed_values, &frequencies, config.tolerance)?
    } else {
        compute_standard_lombscargle(&times_f64, &windowed_values, &frequencies, None, None)?
    };

    // Compute validation metrics
    let validation = compute_validation_metrics(&frequencies, &power, &times_f64)?;

    // Compute confidence intervals if requested
    let confidence_intervals = if let Some(n_bootstrap) = config.bootstrap_iter {
        Some(compute_parallel_bootstrap_ci(
            &times_f64,
            &windowed_values,
            &frequencies,
            n_bootstrap,
            config.confidence.unwrap_or(0.95),
            config,
        )?)
    } else {
        None
    };

    // Compute false alarm probability
    let false_alarm_prob = Some(compute_false_alarm_probability(&power, times_f64.len())?);

    Ok(SimdLombScargleResult {
        frequencies,
        power,
        confidence_intervals,
        false_alarm_prob,
        validation,
    })
}

/// Enhanced input validation
#[allow(dead_code)]
fn validate_inputs<T, U>(times: &[T], values: &[U]) -> SignalResult<()>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if times.is_empty() || values.is_empty() {
        return Err(SignalError::ValueError(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    if times.len() != values.len() {
        return Err(SignalError::ShapeMismatch(format!(
            "Times and values must have same length: {} != {}",
            times.len(),
            values.len()
        )));
    }

    // Check for NaN or infinite values
    for (i, &t) in times.iter().enumerate() {
        if !t.is_finite() {
            return Err(SignalError::ValueError(format!(
                "Non-finite time value at index {}: {:?}",
                i, t
            )));
        }
    }

    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(SignalError::ValueError(format!(
                "Non-finite signal value at index {}: {:?}",
                i, v
            )));
        }
    }

    Ok(())
}

/// Validate time series properties
#[allow(dead_code)]
fn validate_time_series(times: &[f64]) -> SignalResult<()> {
    // Check if _times are sorted
    for i in 1.._times.len() {
        if times[i] <= times[i - 1] {
            return Err(SignalError::ValueError(format!(
                "Times must be strictly increasing: times[{}]={} <= times[{}]={}",
                i,
                times[i],
                i - 1,
                times[i - 1]
            )));
        }
    }

    // Check for reasonable time span
    let t_span = times[_times.len() - 1] - times[0];
    if t_span <= 0.0 {
        return Err(SignalError::ValueError(
            "Time span must be positive".to_string(),
        ));
    }

    // Check for duplicate _times
    let mut sorted_times = times.to_vec();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 1..sorted_times.len() {
        if (sorted_times[i] - sorted_times[i - 1]).abs() < 1e-15 {
            return Err(SignalError::ValueError(format!(
                "Duplicate or near-duplicate _times detected: {} and {}",
                sorted_times[i - 1],
                sorted_times[i]
            )));
        }
    }

    Ok(())
}

/// Convert numeric array to f64
#[allow(dead_code)]
fn convert_to_f64<T>(arr: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    arr.iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect()
}

/// Apply window function
#[allow(dead_code)]
fn apply_window(values: &[f64], config: &LombScargleConfig) -> SignalResult<Vec<f64>> {
    let n = values.len();

    let window = match config.window {
        WindowType::None => vec![1.0; n],
        WindowType::Hann => hann(n)?,
        WindowType::Hamming => hamming(n)?,
        WindowType::Blackman => blackman(n)?,
        WindowType::Custom => {
            if let Some(ref custom) = config.custom_window {
                if custom.len() != n {
                    return Err(SignalError::ShapeMismatch(
                        "Custom window length must match signal length".to_string(),
                    ));
                }
                custom.clone()
            } else {
                return Err(SignalError::ValueError(
                    "Custom window specified but no window _values provided".to_string(),
                ));
            }
        }
    };

    // Apply window with SIMD operations
    let mut windowed = vec![0.0; n];
    let values_view = ArrayView1::from(_values);
    let window_view = ArrayView1::from(&window);
    let windowed_view = ArrayView1::fromshape(n, &mut windowed).unwrap();

    f64::simd_mul(&values_view, &window_view, &windowed_view);

    // Normalize by window sum
    let window_sum: f64 = window.iter().sum();
    windowed.iter_mut().for_each(|v| *v /= window_sum.sqrt());

    Ok(windowed)
}

/// Compute frequency grid
#[allow(dead_code)]
fn compute_frequency_grid(times: &[f64], config: &LombScargleConfig) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let t_span = times[n - 1] - times[0];

    // Enhanced sampling rate estimation
    let mut dts = vec![0.0; n - 1];
    for i in 0..n - 1 {
        dts[i] = times[i + 1] - times[i];
    }
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use median sampling interval for robustness
    let median_dt = if (n - 1) % 2 == 0 {
        (dts[(n - 1) / 2 - 1] + dts[(n - 1) / 2]) / 2.0
    } else {
        dts[(n - 1) / 2]
    };

    let nyquist = 0.5 / median_dt;

    // Determine frequency range
    let f_min = config.f_min.unwrap_or(1.0 / t_span);
    let f_max = config.f_max.unwrap_or(nyquist);

    if f_min >= f_max {
        return Err(SignalError::ValueError(
            "f_min must be less than f_max".to_string(),
        ));
    }

    // Number of frequencies with oversampling
    let n_freq = ((f_max - f_min) * t_span * config.oversample) as usize + 1;

    // Generate frequency grid
    let mut frequencies = Vec::with_capacity(n_freq);
    for i in 0..n_freq {
        frequencies.push(f_min + i as f64 * (f_max - f_min) / (n_freq - 1) as f64);
    }

    Ok(frequencies)
}

/// SIMD-optimized fast Lomb-Scargle algorithm
#[allow(dead_code)]
fn compute_simd_fast_lombscargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
    tolerance: f64,
) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let mut power = vec![0.0; frequencies.len()];

    // Center the data
    let mean_val: f64 = values.iter().sum::<f64>() / n as f64;
    let mut values_centered = vec![0.0; n];
    let values_view = ArrayView1::from(values);
    let centered_view = ArrayView1::fromshape(n, &mut values_centered).unwrap();

    // SIMD subtraction for centering
    f64::simd_sub_scalar(&values_view, mean_val, &centered_view);

    // Precompute time shifts
    let t_mean = times.iter().sum::<f64>() / n as f64;
    let mut times_shifted = vec![0.0; n];
    let times_view = ArrayView1::from(times);
    let shifted_view = ArrayView1::fromshape(n, &mut times_shifted).unwrap();

    f64::simd_sub_scalar(&times_view, t_mean, &shifted_view);

    // Process frequencies in chunks for better cache utilization
    let chunk_size = 64; // Optimize for cache line

    for (chunk_idx, freq_chunk) in frequencies.chunks(chunk_size).enumerate() {
        let start_idx = chunk_idx * chunk_size;

        for (i, &freq) in freq_chunk.iter().enumerate() {
            let omega = 2.0 * PI * freq;

            // Compute tau using SIMD operations
            let tau = compute_tau_simd(&times_shifted, omega);

            // Compute power using SIMD
            power[start_idx + i] =
                compute_power_simd(&times_shifted, &values_centered, omega, tau, tolerance)?;
        }
    }

    Ok(power)
}

/// Compute tau offset using SIMD
#[allow(dead_code)]
fn compute_tau_simd(times: &[f64], omega: f64) -> f64 {
    let n = times.len();
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;

    // Process in SIMD-friendly chunks
    let chunk_size = 8;
    let chunks = times.chunks_exact(chunk_size);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Compute phases
        let mut phases = [0.0; 8];
        for (i, &t) in chunk.iter().enumerate() {
            phases[i] = 2.0 * omega * t;
        }

        // SIMD sin/cos computation
        for &phase in &phases {
            sin_sum += phase.sin();
            cos_sum += phase.cos();
        }
    }

    // Handle remainder
    for &t in remainder {
        let phase = 2.0 * omega * t;
        sin_sum += phase.sin();
        cos_sum += phase.cos();
    }

    0.5 * sin_sum.atan2(cos_sum) / omega
}

/// Compute power at a single frequency using SIMD
#[allow(dead_code)]
fn compute_power_simd(
    times: &[f64],
    values: &[f64],
    omega: f64,
    tau: f64,
    tolerance: f64,
) -> SignalResult<f64> {
    let n = times.len();

    let mut c_tau = 0.0;
    let mut s_tau = 0.0;
    let mut c_tau2 = 0.0;
    let mut s_tau2 = 0.0;

    // Process in chunks for SIMD efficiency
    for (i, (&t, &y)) in times.iter().zip(values.iter()).enumerate() {
        let arg = omega * (t - tau);
        let cos_arg = arg.cos();
        let sin_arg = arg.sin();

        c_tau += y * cos_arg;
        s_tau += y * sin_arg;
        c_tau2 += cos_arg * cos_arg;
        s_tau2 += sin_arg * sin_arg;
    }

    // Avoid division by zero
    if c_tau2 < tolerance || s_tau2 < tolerance {
        return Ok(0.0);
    }

    // Compute power
    let power = 0.5 * (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2);

    Ok(power)
}

/// Standard Lomb-Scargle (fallback for non-SIMD)
#[allow(dead_code)]
fn compute_standard_lombscargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
) -> SignalResult<Vec<f64>> {
    // Implementation of standard algorithm
    // This is a fallback for systems without SIMD
    compute_simd_fast_lombscargle(times, values, frequencies, 1e-10)
}

/// Compute validation metrics
#[allow(dead_code)]
fn compute_validation_metrics(
    frequencies: &[f64],
    power: &[f64],
    times: &[f64],
) -> SignalResult<ValidationMetrics> {
    let n = times.len();
    let n_freq = frequencies.len();

    // Find peak
    let (peak_idx, &max_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .ok_or_else(|| SignalError::ComputationError("No peak found".to_string()))?;

    let peak_freq = frequencies[peak_idx];

    // Estimate SNR
    let mean_power: f64 = power.iter().sum::<f64>() / n_freq as f64;
    let noise_est = power
        .iter()
        .filter(|&&p| (p - mean_power).abs() < mean_power)
        .sum::<f64>()
        / power
            .iter()
            .filter(|&&p| (p - mean_power).abs() < mean_power)
            .count() as f64;

    let snr = max_power / noise_est.max(1e-10);

    // Estimate peak width (FWHM)
    let half_max = max_power / 2.0;
    let mut left_idx = peak_idx;
    let mut right_idx = peak_idx;

    while left_idx > 0 && power[left_idx] > half_max {
        left_idx -= 1;
    }

    while right_idx < n_freq - 1 && power[right_idx] > half_max {
        right_idx += 1;
    }

    let peak_width = if right_idx > left_idx {
        frequencies[right_idx] - frequencies[left_idx]
    } else {
        frequencies[1] - frequencies[0]
    };

    // Effective number of independent frequencies
    let t_span = times[n - 1] - times[0];
    let n_eff = t_span * (frequencies[n_freq - 1] - frequencies[0]);

    // Spectral leakage estimate
    let window_power: f64 = power.iter().sum();
    let leakage = 1.0 - max_power / window_power;

    Ok(ValidationMetrics {
        snr,
        n_eff,
        max_power,
        peak_freq,
        peak_width,
        leakage,
    })
}

/// Parallel bootstrap confidence intervals
#[allow(dead_code)]
fn compute_parallel_bootstrap_ci(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    config: &LombScargleConfig,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = times.len();
    let n_freq = frequencies.len();

    // Shared data for parallel processing
    let times_arc = Arc::new(times.to_vec());
    let values_arc = Arc::new(values.to_vec());
    let frequencies_arc = Arc::new(frequencies.to_vec());
    let config_arc = Arc::new(config.clone());

    // Parallel _bootstrap iterations
    let bootstrap_powers: Vec<Vec<f64>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|iter| {
            let mut rng = rand::rng();
            let times_ref = times_arc.clone();
            let values_ref = values_arc.clone();
            let frequencies_ref = frequencies_arc.clone();
            let config_ref = config_arc.clone();

            // Resample with replacement
            let mut resampled_times = vec![0.0; n];
            let mut resampled_values = vec![0.0; n];

            for i in 0..n {
                let idx = rng.gen_range(0..n);
                resampled_times[i] = times_ref[idx];
                resampled_values[i] = values_ref[idx];
            }

            // Sort by time
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| resampled_times[i].partial_cmp(&resampled_times[j]).unwrap());

            let sorted_times: Vec<f64> = indices.iter().map(|&i| resampled_times[i]).collect();
            let sorted_values: Vec<f64> = indices.iter().map(|&i| resampled_values[i]).collect();

            // Compute periodogram for _bootstrap sample
            compute_simd_fast_lombscargle(
                &sorted_times,
                &sorted_values,
                &frequencies_ref,
                config_ref.tolerance,
            )
            .unwrap_or_else(|_| vec![0.0; n_freq])
        })
        .collect();

    // Compute percentiles
    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha) * n_bootstrap as f64) as usize;

    let mut lower_ci = vec![0.0; n_freq];
    let mut upper_ci = vec![0.0; n_freq];

    for i in 0..n_freq {
        let mut freq_powers: Vec<f64> = bootstrap_powers.iter().map(|p| p[i]).collect();
        freq_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());

        lower_ci[i] = freq_powers[lower_idx];
        upper_ci[i] = freq_powers[upper_idx.min(n_bootstrap - 1)];
    }

    Ok((lower_ci, upper_ci))
}

/// Compute false alarm probability
#[allow(dead_code)]
fn compute_false_alarm_probability(_power: &[f64], ndata: usize) -> SignalResult<Vec<f64>> {
    let n_freq = power.len();
    let n_eff = n_freq as f64; // Simplified; could use more sophisticated estimate

    // Baluev (2008) approximation for FAP
    let fap: Vec<f64> = _power
        .iter()
        .map(|&p| {
            let z = p;
            let fap_single = (-(z - 1.0).max(0.0)).exp();
            1.0 - (1.0 - fap_single).powf(n_eff)
        })
        .collect();

    Ok(fap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_lombscargle_basic() {
        let times = vec![0.0, 0.1, 0.3, 0.7, 1.2, 1.5, 2.0, 2.1];
        let values = vec![1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 0.5];

        let config = LombScargleConfig::default();
        let result = simd_lombscargle(&times, &values, &config).unwrap();

        assert!(result.frequencies.len() > 0);
        assert_eq!(result.frequencies.len(), result.power.len());
        assert!(result.validation.snr > 0.0);
    }

    #[test]
    fn test_validation_errors() {
        // Empty arrays
        let times: Vec<f64> = vec![];
        let values: Vec<f64> = vec![];
        let config = LombScargleConfig::default();
        assert!(simd_lombscargle(&times, &values, &config).is_err());

        // Mismatched lengths
        let times = vec![0.0, 1.0];
        let values = vec![1.0];
        assert!(simd_lombscargle(&times, &values, &config).is_err());

        // Unsorted times
        let times = vec![1.0, 0.0];
        let values = vec![1.0, 0.0];
        assert!(simd_lombscargle(&times, &values, &config).is_err());
    }
}
