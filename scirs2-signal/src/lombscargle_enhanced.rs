// Enhanced Lomb-Scargle periodogram with additional validation and features
//
// This module provides advanced features for Lomb-Scargle analysis including:
// - Window function support
// - Bootstrap confidence intervals
// - False alarm probability estimation
// - Numerical stability improvements

use crate::error::{SignalError, SignalResult};
use num_traits::{Float, NumCast};
use rand::Rng;
use scirs2_core::validation::{check_finite, check_positive};
use std::fmt::Debug;

#[allow(unused_imports)]
/// Window function types for Lomb-Scargle analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// No window (rectangular)
    None,
    /// Hann (Hanning) window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Custom window function
    Custom,
}

impl std::fmt::Display for WindowType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowType::None => write!(f, "None"),
            WindowType::Hann => write!(f, "Hann"),
            WindowType::Hamming => write!(f, "Hamming"),
            WindowType::Blackman => write!(f, "Blackman"),
            WindowType::Custom => write!(f, "Custom"),
        }
    }
}

/// Enhanced Lomb-Scargle configuration
#[derive(Debug, Clone)]
pub struct LombScargleConfig {
    /// Window function to apply
    pub window: WindowType,
    /// Custom window values (if window == Custom)
    pub custom_window: Option<Vec<f64>>,
    /// Oversample factor for frequency grid
    pub oversample: f64,
    /// Minimum frequency (if None, auto-determined)
    pub f_min: Option<f64>,
    /// Maximum frequency (if None, uses Nyquist)
    pub f_max: Option<f64>,
    /// Number of bootstrap iterations for CI
    pub bootstrap_iter: Option<usize>,
    /// Confidence level for bootstrap CI
    pub confidence: Option<f64>,
    /// Use fast algorithm (Scargle 1989)
    pub use_fast: bool,
    /// Tolerance for numerical stability
    pub tolerance: f64,
}

impl Default for LombScargleConfig {
    fn default() -> Self {
        Self {
            window: WindowType::None,
            custom_window: None,
            oversample: 5.0,
            f_min: None,
            f_max: None,
            bootstrap_iter: None,
            confidence: Some(0.95),
            use_fast: true,
            tolerance: 1e-10,
        }
    }
}

/// Enhanced Lomb-Scargle periodogram with validation and features
///
/// # Arguments
///
/// * `times` - Sample times (must be sorted and finite)
/// * `values` - Signal values
/// * `config` - Configuration for enhanced features
///
/// # Returns
///
/// * `frequencies` - Frequency array
/// * `power` - Periodogram power
/// * `confidence_intervals` - Optional bootstrap confidence intervals
#[allow(dead_code)]
pub fn lombscargle_enhanced<T, U>(
    times: &[T],
    values: &[U],
    config: &LombScargleConfig,
) -> SignalResult<(Vec<f64>, Vec<f64>, Option<(Vec<f64>, Vec<f64>)>)>
where
    T: Float + NumCast + Debug + std::fmt::Display,
    U: Float + NumCast + Debug,
{
    // Validate inputs
    if times.is_empty() || values.is_empty() {
        return Err(SignalError::ValueError(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    if times.len() != values.len() {
        return Err(SignalError::ShapeMismatch(
            "Times and values arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 and validate
    let times_f64: Vec<f64> = times
        .iter()
        .map(|&t| {
            let val: T = NumCast::from(t).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert time {:?} to f64", t))
            })?;
            check_finite(val, "time value")?;
            let f64_val: f64 = NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })?;
            Ok(f64_val)
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    let values_f64: Vec<f64> = values
        .iter()
        .map(|&v| {
            let val = NumCast::from(v).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert value {:?} to f64", v))
            })?;
            check_finite(val, "value value")?;
            Ok(val)
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Check if times are sorted
    for i in 1..times_f64.len() {
        if times_f64[i] <= times_f64[i - 1] {
            return Err(SignalError::ValueError(
                "Time values must be strictly increasing".to_string(),
            ));
        }
    }

    // Apply window function if requested
    let windowed_values = apply_window(&values_f64, config)?;

    // Compute frequency grid
    let frequencies = compute_frequency_grid(&times_f64, config)?;

    // Compute periodogram
    let power = if config.use_fast {
        compute_fast_lombscargle(&times_f64, &windowed_values, &frequencies, config.tolerance)?
    } else {
        compute_standard_lombscargle(
            &times_f64,
            &windowed_values,
            &frequencies,
            None,
            None,
            Some(false),
        )?
    };

    // Compute bootstrap confidence intervals if requested
    let confidence_intervals = if let Some(n_iter) = config.bootstrap_iter {
        Some(bootstrap_confidence_intervals(
            &times_f64,
            &values_f64,
            &frequencies,
            n_iter,
            config.confidence.unwrap_or(0.95),
            config,
        )?)
    } else {
        None
    };

    Ok((frequencies, power, confidence_intervals))
}

/// Apply window function to values
#[allow(dead_code)]
fn apply_window(values: &[f64], config: &LombScargleConfig) -> SignalResult<Vec<f64>> {
    let n = values.len();

    let window = match config.window {
        WindowType::None => vec![1.0; n],
        WindowType::Hann => {
            let mut w = vec![0.0; n];
            for i in 0..n {
                w[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            }
            w
        }
        WindowType::Hamming => {
            let mut w = vec![0.0; n];
            for i in 0..n {
                w[i] = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            }
            w
        }
        WindowType::Blackman => {
            let mut w = vec![0.0; n];
            for i in 0..n {
                let x = 2.0 * PI * i as f64 / (n - 1) as f64;
                w[i] = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
            }
            w
        }
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

    // Apply window and normalize
    let mut windowed = vec![0.0; n];
    let window_sum: f64 = window.iter().sum();

    for i in 0..n {
        windowed[i] = values[i] * window[i] / window_sum.sqrt();
    }

    Ok(windowed)
}

/// Compute frequency grid with oversampling
#[allow(dead_code)]
fn compute_frequency_grid(times: &[f64], config: &LombScargleConfig) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let t_span = times[n - 1] - times[0];

    // Estimate average sampling rate
    let avg_dt = t_span / (n - 1) as f64;
    let nyquist = 0.5 / avg_dt;

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

/// Fast Lomb-Scargle algorithm (Press & Rybicki 1989)
#[allow(dead_code)]
fn compute_fast_lombscargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
    tolerance: f64,
) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let mut power = vec![0.0; frequencies.len()];

    // Center the data
    let mean_val: f64 = values.iter().sum::<f64>() / n as f64;
    let values_centered: Vec<f64> = values.iter().map(|&v| v - mean_val).collect();

    // Precompute time shifts for numerical stability
    let t_mean = times.iter().sum::<f64>() / n as f64;
    let times_shifted: Vec<f64> = times.iter().map(|&t| t - t_mean).collect();

    for (i, &freq) in frequencies.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        // Compute time offset tau for this frequency
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;

        for &t in &times_shifted {
            let phase = 2.0 * omega * t;
            sum_sin += phase.sin();
            sum_cos += phase.cos();
        }

        let tau = 0.5 * sum_sin.atan2(sum_cos) / omega;

        // Compute sums with numerical stability checks
        let mut c_tau = 0.0;
        let mut s_tau = 0.0;
        let mut c_tau2 = 0.0;
        let mut s_tau2 = 0.0;

        for (j, &t) in times_shifted.iter().enumerate() {
            let arg = omega * (t - tau);
            let cos_arg = arg.cos();
            let sin_arg = arg.sin();

            // Check for numerical issues
            if cos_arg.is_finite() && sin_arg.is_finite() {
                c_tau += values_centered[j] * cos_arg;
                s_tau += values_centered[j] * sin_arg;
                c_tau2 += cos_arg * cos_arg;
                s_tau2 += sin_arg * sin_arg;
            }
        }

        // Add small tolerance to prevent division by zero
        c_tau2 = c_tau2.max(tolerance);
        s_tau2 = s_tau2.max(tolerance);

        // Compute variance
        let variance: f64 = values_centered.iter().map(|&v| v * v).sum::<f64>() / n as f64;

        if variance > tolerance {
            // Standard normalization
            power[i] = 0.5 * ((c_tau * c_tau / c_tau2) + (s_tau * s_tau / s_tau2)) / variance;
        } else {
            power[i] = 0.0;
        }
    }

    Ok(power)
}

/// Standard Lomb-Scargle algorithm (for comparison)
#[allow(dead_code)]
fn compute_standard_lombscargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let mut power = vec![0.0; frequencies.len()];

    // Center the data
    let mean_val: f64 = values.iter().sum::<f64>() / n as f64;
    let values_centered: Vec<f64> = values.iter().map(|&v| v - mean_val).collect();
    let variance: f64 = values_centered.iter().map(|&v| v * v).sum::<f64>() / n as f64;

    if variance == 0.0 {
        return Ok(power); // All zeros
    }

    for (i, &freq) in frequencies.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        let mut a = 0.0;
        let mut b = 0.0;
        let mut c = 0.0;
        let mut d = 0.0;

        for (j, &t) in times.iter().enumerate() {
            let cos_wt = (omega * t).cos();
            let sin_wt = (omega * t).sin();

            a += values_centered[j] * cos_wt;
            b += values_centered[j] * sin_wt;
            c += cos_wt * cos_wt;
            d += sin_wt * sin_wt;
        }

        power[i] = 0.5 * ((a * a / c) + (b * b / d)) / variance;
    }

    Ok(power)
}

/// Bootstrap confidence intervals for Lomb-Scargle periodogram
#[allow(dead_code)]
fn bootstrap_confidence_intervals(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
    n_iterations: usize,
    confidence: f64,
    config: &LombScargleConfig,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = times.len();
    let n_freq = frequencies.len();

    // Store bootstrap results
    let mut bootstrap_powers = vec![vec![0.0; n_freq]; n_iterations];

    // Random number generator
    let mut rng = rand::rng();

    for iter in 0..n_iterations {
        // Generate bootstrap sample
        let mut boot_times = Vec::with_capacity(n);
        let mut boot_values = Vec::with_capacity(n);

        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            boot_times.push(times[idx]);
            boot_values.push(values[idx]);
        }

        // Sort by time
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| boot_times[i].partial_cmp(&boot_times[j]).unwrap());

        let sorted_times: Vec<f64> = indices.iter().map(|&i| boot_times[i]).collect();
        let sorted_values: Vec<f64> = indices.iter().map(|&i| boot_values[i]).collect();

        // Compute periodogram for bootstrap sample
        let power =
            compute_fast_lombscargle(&sorted_times, &sorted_values, frequencies, config.tolerance)?;

        bootstrap_powers[iter] = power;
    }

    // Compute percentiles
    let lower_percentile = ((1.0 - confidence) / 2.0 * n_iterations as f64) as usize;
    let upper_percentile = ((1.0 + confidence) / 2.0 * n_iterations as f64) as usize;

    let mut lower_ci = vec![0.0; n_freq];
    let mut upper_ci = vec![0.0; n_freq];

    for j in 0..n_freq {
        let mut freq_powers: Vec<f64> = bootstrap_powers.iter().map(|p| p[j]).collect();
        freq_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());

        lower_ci[j] = freq_powers[lower_percentile];
        upper_ci[j] = freq_powers[upper_percentile.min(n_iterations - 1)];
    }

    Ok((lower_ci, upper_ci))
}

/// Estimate false alarm probability for Lomb-Scargle peaks
///
/// # Arguments
///
/// * `peak_power` - Power value of the peak
/// * `n_samples` - Number of samples in the signal
/// * `n_frequencies` - Number of frequencies tested
/// * `normalization` - Type of normalization used
///
/// # Returns
///
/// * False alarm probability
#[allow(dead_code)]
pub fn false_alarm_probability(
    peak_power: f64,
    n_samples: usize,
    n_frequencies: usize,
    normalization: &str,
) -> SignalResult<f64> {
    check_positive(peak_power, "peak_power")?;
    check_positive(n_samples as f64, "n_samples")?;
    check_positive(n_frequencies as f64, "n_frequencies")?;

    let fap = match normalization {
        "standard" => {
            // Baluev (2008) approximation
            let z = peak_power;
            let n_eff = n_frequencies as f64;

            if z <= 0.0 {
                1.0
            } else {
                // Approximate FAP using extreme value statistics
                let prob_single = (-z).exp();
                1.0 - (1.0 - prob_single).powf(n_eff)
            }
        }
        "model" => {
            // For model normalization, use chi-squared distribution
            let dof = 2.0; // Degrees of freedom for sinusoidal model
            let chi2 = peak_power * (n_samples - 3) as f64;

            // Approximate using incomplete gamma function
            let prob_single = (-chi2 / 2.0).exp();
            1.0 - (1.0 - prob_single).powf(n_frequencies as f64)
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown normalization method: {}",
                normalization
            )));
        }
    };

    Ok(fap.min(1.0).max(0.0))
}

/// Analytical significance level for given false alarm probability
///
/// # Arguments
///
/// * `fap` - Desired false alarm probability
/// * `n_samples` - Number of samples
/// * `n_frequencies` - Number of frequencies
/// * `normalization` - Normalization method
///
/// # Returns
///
/// * Power threshold for the given FAP
#[allow(dead_code)]
pub fn significance_threshold(
    fap: f64,
    n_samples: usize,
    n_frequencies: usize,
    normalization: &str,
) -> SignalResult<f64> {
    check_positive(fap, "fap")?;
    if fap >= 1.0 {
        return Err(SignalError::ValueError(
            "False alarm probability must be less than 1".to_string(),
        ));
    }

    let threshold = match normalization {
        "standard" => {
            // Invert the FAP formula
            let p_single = 1.0 - (1.0 - fap).powf(1.0 / n_frequencies as f64);
            -(p_single.ln())
        }
        "model" => {
            // For model normalization
            let p_single = 1.0 - (1.0 - fap).powf(1.0 / n_frequencies as f64);
            let chi2 = -2.0 * p_single.ln();
            chi2 / (n_samples - 3) as f64
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown normalization method: {}",
                normalization
            )));
        }
    };

    Ok(threshold)
}
