//! Lomb-Scargle periodogram implementation for unevenly sampled data.
//!
//! This module provides functions for spectral analysis of unevenly sampled signals
//! using the Lomb-Scargle periodogram technique.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array, Array1, ArrayView1};
use num_traits::{Float, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Compute normalized Lomb-Scargle periodogram for unevenly sampled data.
///
/// The Lomb-Scargle periodogram is a method for detecting periodic signals in
/// unevenly-sampled time series data. This implementation is based on the algorithm
/// described by Press & Rybicki (1989) and adapted from SciPy's implementation.
///
/// # Arguments
///
/// * `x` - Sample times (must be sorted increasingly)
/// * `y` - Signal values corresponding to sample times
/// * `freqs` - Array of frequencies to evaluate (if None, automatic frequency selection is used)
/// * `normalization` - Normalization method ('standard', 'model', 'log', or 'psd')
/// * `center_data` - If true, subtract the mean from the signal
/// * `fit_mean` - If true, include a constant offset in the model
/// * `nyquist_factor` - Factor determining the maximum frequency (default: 1)
///
/// # Returns
///
/// * Tuple of (frequencies, periodogram values)
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lombscargle::{lombscargle, AutoFreqMethod};
/// use ndarray::Array1;
/// use std::f64::consts::PI;
///
/// // Generate unevenly sampled data with a 1 Hz sinusoid
/// let n = 100;
/// let rng = rand::thread_rng();
/// let mut t = Array1::linspace(0.0, 10.0, n);
/// // Add some random noise to make sampling uneven
/// for i in 0..n {
///     t[i] += 0.1 * rand::random::<f64>();
/// }
/// let y: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 1.0 * ti).sin()).collect();
///
/// // Compute Lomb-Scargle periodogram
/// let (freqs, power) = lombscargle(
///     &t,
///     &y,
///     None,
///     Some("standard"),
///     Some(true),
///     Some(true),
///     None,
///     None,
/// ).unwrap();
///
/// // Find the frequency with maximum power
/// let mut max_idx = 0;
/// let mut max_power = 0.0;
/// for (i, &p) in power.iter().enumerate() {
///     if p > max_power {
///         max_power = p;
///         max_idx = i;
///     }
/// }
///
/// // The frequency with maximum power should be close to 1 Hz
/// assert!((freqs[max_idx] - 1.0).abs() < 0.1);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn lombscargle<T, U>(
    x: &[T],
    y: &[U],
    freqs: Option<&[f64]>,
    normalization: Option<&str>,
    center_data: Option<bool>,
    fit_mean: Option<bool>,
    nyquist_factor: Option<f64>,
    freq_method: Option<AutoFreqMethod>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    // Validate inputs
    if x.is_empty() {
        return Err(SignalError::ValueError(
            "Sample times array is empty".to_string(),
        ));
    }

    if y.is_empty() {
        return Err(SignalError::ValueError(
            "Signal values array is empty".to_string(),
        ));
    }

    if x.len() != y.len() {
        return Err(SignalError::ValueError(format!(
            "Sample times and signal values must have the same length, got {} and {}",
            x.len(),
            y.len()
        )));
    }

    // Convert inputs to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let y_f64: Vec<f64> = y
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Check that sample times are sorted
    for i in 1..x_f64.len() {
        if x_f64[i] < x_f64[i - 1] {
            return Err(SignalError::ValueError(
                "Sample times must be sorted in increasing order".to_string(),
            ));
        }
    }

    // Default parameter values
    let center = center_data.unwrap_or(true);
    let fit_offset = fit_mean.unwrap_or(true);
    let norm_method = match normalization.unwrap_or("standard") {
        "standard" => NormalizationMethod::Standard,
        "model" => NormalizationMethod::Model,
        "log" => NormalizationMethod::Log,
        "psd" => NormalizationMethod::Psd,
        _ => {
            return Err(SignalError::ValueError(format!(
                "Invalid normalization method. Valid options are 'standard', 'model', 'log', or 'psd', got {}",
                normalization.unwrap_or("standard")
            )));
        }
    };

    // Determine frequencies to evaluate if not provided
    let frequencies = if let Some(f) = freqs {
        f.to_vec()
    } else {
        // Auto-determine frequencies
        autofrequency(&x_f64, nyquist_factor.unwrap_or(1.0), freq_method)?
    };

    // Compute periodogram
    let pgram = _lombscargle_impl(
        &Array1::from(x_f64),
        &Array1::from(y_f64),
        &Array1::from(frequencies.clone()),
        center,
        fit_offset,
        norm_method,
    )?;

    Ok((frequencies, pgram.to_vec()))
}

/// Methods for automatically determining frequency grids
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoFreqMethod {
    /// Frequency grid based on Fourier transform (default)
    Fft,
    /// Frequency grid linear in frequency
    Linear,
    /// Frequency grid logarithmic in frequency
    Log,
}

impl Default for AutoFreqMethod {
    fn default() -> Self {
        AutoFreqMethod::Fft
    }
}

impl std::str::FromStr for AutoFreqMethod {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fft" => Ok(AutoFreqMethod::Fft),
            "linear" => Ok(AutoFreqMethod::Linear),
            "log" => Ok(AutoFreqMethod::Log),
            _ => Err(SignalError::ValueError(format!(
                "Invalid frequency method: '{}'. Valid options are: 'fft', 'linear', 'log'",
                s
            ))),
        }
    }
}

/// Methods for normalizing the periodogram
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizationMethod {
    /// Normalize by the residual variance of the data
    Standard,
    /// Normalize by the sum of squares of the data
    Model,
    /// Return the logarithm of the standard normalization
    Log,
    /// Normalize by the noise power (suitable for power spectral density)
    Psd,
}

/// Automatically determine frequency grid for Lomb-Scargle periodogram
///
/// # Arguments
///
/// * `times` - Sample times
/// * `nyquist_factor` - Factor determining the maximum frequency
/// * `method` - Method for grid spacing
///
/// # Returns
///
/// * Array of frequencies
fn autofrequency(
    times: &[f64],
    nyquist_factor: f64,
    method: Option<AutoFreqMethod>,
) -> SignalResult<Vec<f64>> {
    if times.is_empty() {
        return Err(SignalError::ValueError(
            "Sample times array is empty".to_string(),
        ));
    }

    if nyquist_factor <= 0.0 {
        return Err(SignalError::ValueError(
            "nyquist_factor must be positive".to_string(),
        ));
    }

    // Determine minimum and maximum times
    let t_min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let t_max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if t_min == t_max {
        return Err(SignalError::ValueError(
            "All sample times are identical".to_string(),
        ));
    }

    // Determine baseline timestep from median of differences
    let n = times.len();
    let mut dts = Vec::with_capacity(n - 1);

    for i in 1..n {
        dts.push(times[i] - times[i - 1]);
    }

    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dt = if dts.len() % 2 == 0 {
        (dts[dts.len() / 2 - 1] + dts[dts.len() / 2]) / 2.0
    } else {
        dts[dts.len() / 2]
    };

    // Maximum frequency based on Nyquist limit
    let f_max = 0.5 * nyquist_factor / dt;

    // Minimum frequency
    let t_range = t_max - t_min;
    let f_min = 1.0 / t_range;

    // Use specified method to determine frequency grid
    let method_val = method.unwrap_or_default();

    match method_val {
        AutoFreqMethod::Fft => {
            // Determine number of frequencies using average Nyquist frequency
            let n_samples = (t_range / dt).floor() as usize;

            // Round up to the next power of 2 for FFT efficiency
            let n_freq = n_samples.next_power_of_two();

            // Generate linearly spaced frequencies from f_min to f_max
            let mut freqs = Vec::with_capacity(n_freq / 2 + 1);
            for i in 0..=n_freq / 2 {
                let f = f_min + (f_max - f_min) * (i as f64 / (n_freq / 2) as f64);
                freqs.push(f);
            }

            Ok(freqs)
        }
        AutoFreqMethod::Linear => {
            // Number of frequencies in the linear grid
            let n_freq = (20.0 * t_range / dt).floor() as usize;

            // Generate linearly spaced frequencies from f_min to f_max
            let mut freqs = Vec::with_capacity(n_freq);
            for i in 0..n_freq {
                let f = f_min + (f_max - f_min) * (i as f64 / (n_freq - 1) as f64);
                freqs.push(f);
            }

            Ok(freqs)
        }
        AutoFreqMethod::Log => {
            // Number of frequencies in the log grid
            let n_freq = (100.0 * (f_max / f_min).ln()).floor() as usize;

            // Generate logarithmically spaced frequencies from f_min to f_max
            let mut freqs = Vec::with_capacity(n_freq);
            for i in 0..n_freq {
                let log_f =
                    (f_min.ln()) + ((f_max / f_min).ln()) * (i as f64 / (n_freq - 1) as f64);
                freqs.push(log_f.exp());
            }

            Ok(freqs)
        }
    }
}

/// Compute the Lomb-Scargle periodogram (implementation)
fn _lombscargle_impl(
    t: &Array1<f64>,
    y: &Array1<f64>,
    frequency: &Array1<f64>,
    center_data: bool,
    fit_mean: bool,
    normalization: NormalizationMethod,
) -> SignalResult<Array1<f64>> {
    let n_samples = t.len();
    let n_freqs = frequency.len();

    // Center the data if requested
    let (y_centered, y_mean) = if center_data {
        let mean = y.sum() / n_samples as f64;
        (y - mean, mean)
    } else {
        (y.clone(), 0.0)
    };

    // Initialize the periodogram array
    let mut power = Array1::zeros(n_freqs);

    // Calculate the weighted sum of y^2
    let y2_sum = y_centered.iter().map(|&y| y * y).sum::<f64>();

    if y2_sum == 0.0 {
        return Err(SignalError::ValueError(
            "All data values are identical".to_string(),
        ));
    }

    // Compute the periodogram for each frequency
    for (i, &freq) in frequency.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        // Calculate tau for this frequency
        let (s2omega, c2omega) = t
            .iter()
            .map(|&ti| {
                let s = (2.0 * omega * ti).sin();
                let c = (2.0 * omega * ti).cos();
                (s, c)
            })
            .fold((0.0, 0.0), |acc, (s, c)| (acc.0 + s, acc.1 + c));

        let tau = 0.5 * (s2omega / c2omega).atan2(1.0);

        // Calculate the trigonometric terms
        let (mut c_tau, mut s_tau) = (0.0, 0.0);
        let (mut c_tau2, mut s_tau2) = (0.0, 0.0);
        let (mut cs_tau) = 0.0;

        for (ti, &yi) in t.iter().zip(y_centered.iter()) {
            let c = (omega * ti - tau).cos();
            let s = (omega * ti - tau).sin();

            c_tau += yi * c;
            s_tau += yi * s;
            c_tau2 += c * c;
            s_tau2 += s * s;
            cs_tau += c * s;
        }

        // Compute the periodogram value based on the normalization method
        let p = match normalization {
            NormalizationMethod::Standard => {
                // Compute the generalized Lomb-Scargle periodogram
                if fit_mean {
                    // Include a constant offset (mean) in the model
                    let y_dot_h = y_centered.iter().sum::<f64>();
                    let h_dot_h = n_samples as f64;

                    // Calculate YY = sum(y^2)
                    let yy = y_centered.iter().map(|&y| y * y).sum::<f64>();

                    // Calculate the numerator
                    let n1 = c_tau * c_tau / c_tau2;
                    let n2 = s_tau * s_tau / s_tau2;

                    // Compute the final power value
                    (n1 + n2) / (yy - y_dot_h * y_dot_h / h_dot_h)
                } else {
                    // Standard Lomb-Scargle periodogram (no constant offset)
                    (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2) / y2_sum
                }
            }
            NormalizationMethod::Model => {
                // Normalize by the model chi-squared (Baluev 2008)
                if fit_mean {
                    let y_dot_h = y_centered.iter().sum::<f64>();
                    let h_dot_h = n_samples as f64;

                    // Calculate YY = sum(y^2)
                    let yy = y_centered.iter().map(|&y| y * y).sum::<f64>();

                    // Compute the model chi-squared
                    let denominator = yy - y_dot_h * y_dot_h / h_dot_h;

                    // Calculate the numerator
                    let n1 = c_tau * c_tau / c_tau2;
                    let n2 = s_tau * s_tau / s_tau2;

                    // Compute the final power value
                    1.0 - (denominator - (n1 + n2)) / denominator
                } else {
                    1.0 - (y2_sum - (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2)) / y2_sum
                }
            }
            NormalizationMethod::Log => {
                // Return the logarithm of the standard normalization
                let standard = if fit_mean {
                    let y_dot_h = y_centered.iter().sum::<f64>();
                    let h_dot_h = n_samples as f64;

                    // Calculate YY = sum(y^2)
                    let yy = y_centered.iter().map(|&y| y * y).sum::<f64>();

                    // Calculate the numerator
                    let n1 = c_tau * c_tau / c_tau2;
                    let n2 = s_tau * s_tau / s_tau2;

                    // Compute the final power value
                    (n1 + n2) / (yy - y_dot_h * y_dot_h / h_dot_h)
                } else {
                    (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2) / y2_sum
                };

                standard.ln()
            }
            NormalizationMethod::Psd => {
                // Normalize for a power spectral density (Horne & Baliunas 1986)
                if fit_mean {
                    let y_dot_h = y_centered.iter().sum::<f64>();
                    let h_dot_h = n_samples as f64;

                    // Calculate YY = sum(y^2)
                    let yy = y_centered.iter().map(|&y| y * y).sum::<f64>();

                    // Calculate the numerator
                    let n1 = c_tau * c_tau / c_tau2;
                    let n2 = s_tau * s_tau / s_tau2;

                    // Compute the final power value
                    0.5 * n_samples as f64 * (n1 + n2) / (yy - y_dot_h * y_dot_h / h_dot_h)
                } else {
                    0.5 * n_samples as f64 * (c_tau * c_tau / c_tau2 + s_tau * s_tau / s_tau2)
                        / y2_sum
                }
            }
        };

        power[i] = p;
    }

    Ok(power)
}

/// Determine significance levels for Lomb-Scargle periodogram.
///
/// # Arguments
///
/// * `power` - Periodogram values
/// * `fap_levels` - False alarm probability levels (e.g., [0.01, 0.05, 0.1])
/// * `normalization` - Normalization method used for the periodogram
/// * `n_samples` - Number of samples in the original signal
///
/// # Returns
///
/// * Vector of power thresholds corresponding to the requested FAP levels
pub fn significance_levels(
    power: &[f64],
    fap_levels: &[f64],
    normalization: &str,
    n_samples: usize,
) -> SignalResult<Vec<f64>> {
    if power.is_empty() {
        return Err(SignalError::ValueError("Periodogram is empty".to_string()));
    }

    if fap_levels.is_empty() {
        return Err(SignalError::ValueError(
            "No FAP levels provided".to_string(),
        ));
    }

    // Validate FAP levels
    for &fap in fap_levels {
        if fap <= 0.0 || fap >= 1.0 {
            return Err(SignalError::ValueError(
                "FAP levels must be between 0 and 1 (exclusive)".to_string(),
            ));
        }
    }

    // Determine the equivalent z-statistic based on normalization
    let z_statistics = match normalization {
        "standard" => power.to_vec(),
        "model" => power
            .iter()
            .map(|&p| -n_samples as f64 * (1.0 - p).ln())
            .collect(),
        "log" => power.iter().map(|&p| p.exp()).collect(),
        "psd" => power.iter().map(|&p| p * 2.0 / n_samples as f64).collect(),
        _ => {
            return Err(SignalError::ValueError(format!(
                "Invalid normalization method: {}. Valid options are 'standard', 'model', 'log', or 'psd'",
                normalization
            )));
        }
    };

    // Compute significance levels based on Chi-squared distribution
    // For large N, -2*ln(FAP) follows a chi-squared distribution with 2 degrees of freedom
    let mut sig_levels = Vec::with_capacity(fap_levels.len());

    for &fap in fap_levels {
        // Compute threshold based on exponential distribution
        let threshold = -fap.ln();

        // Convert back to the original normalization
        let power_threshold = match normalization {
            "standard" => threshold,
            "model" => 1.0 - (-threshold / n_samples as f64).exp(),
            "log" => threshold.ln(),
            "psd" => threshold * n_samples as f64 / 2.0,
            _ => unreachable!(), // Already validated above
        };

        sig_levels.push(power_threshold);
    }

    Ok(sig_levels)
}

/// Find peaks in the Lomb-Scargle periodogram.
///
/// # Arguments
///
/// * `frequency` - Frequency array
/// * `power` - Periodogram values
/// * `power_threshold` - Power threshold for peak detection
/// * `freq_window` - Frequency window for peak grouping (if None, no grouping is performed)
///
/// # Returns
///
/// * Tuple of (peak frequencies, peak powers)
pub fn find_peaks(
    frequency: &[f64],
    power: &[f64],
    power_threshold: f64,
    freq_window: Option<f64>,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if frequency.is_empty() || power.is_empty() {
        return Err(SignalError::ValueError(
            "Frequency or power array is empty".to_string(),
        ));
    }

    if frequency.len() != power.len() {
        return Err(SignalError::ValueError(format!(
            "Frequency and power arrays must have the same length, got {} and {}",
            frequency.len(),
            power.len()
        )));
    }

    // Find local maxima
    let mut peak_indices = Vec::new();

    for i in 1..power.len() - 1 {
        if power[i] > power[i - 1] && power[i] > power[i + 1] && power[i] >= power_threshold {
            peak_indices.push(i);
        }
    }

    // Check endpoints
    if power.len() >= 2 {
        if power[0] > power[1] && power[0] >= power_threshold {
            peak_indices.push(0);
        }

        if power[power.len() - 1] > power[power.len() - 2]
            && power[power.len() - 1] >= power_threshold
        {
            peak_indices.push(power.len() - 1);
        }
    }

    // Sort peaks by power (descending)
    peak_indices.sort_by(|&a, &b| power[b].partial_cmp(&power[a]).unwrap());

    // Group peaks if a frequency window is specified
    let mut result_freqs = Vec::new();
    let mut result_powers = Vec::new();

    if let Some(window) = freq_window {
        if window <= 0.0 {
            return Err(SignalError::ValueError(
                "Frequency window must be positive".to_string(),
            ));
        }

        let mut used_indices = vec![false; power.len()];

        for &idx in &peak_indices {
            if used_indices[idx] {
                continue;
            }

            // This is a new peak group
            result_freqs.push(frequency[idx]);
            result_powers.push(power[idx]);

            // Mark all peaks within the frequency window as used
            for &other_idx in &peak_indices {
                if !used_indices[other_idx]
                    && (frequency[other_idx] - frequency[idx]).abs() <= window
                {
                    used_indices[other_idx] = true;
                }
            }
        }
    } else {
        // No grouping, just return all peaks
        for &idx in &peak_indices {
            result_freqs.push(frequency[idx]);
            result_powers.push(power[idx]);
        }
    }

    Ok((result_freqs, result_powers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array;
    use std::f64::consts::PI;

    #[test]
    fn test_lombscargle_sine_wave() {
        // Create a sine wave with known frequency
        let frequency = 0.2; // 0.2 Hz
        let n = 100;

        // Unevenly sampled time points
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            // Add some randomness to make sampling uneven
            let ti = i as f64 * 0.5 + 0.1 * (i as f64).sin();
            t.push(ti);
        }

        // Generate sine wave
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * frequency * ti).sin())
            .collect();

        // Define frequency grid
        let freq_min = 0.01;
        let freq_max = 0.5;
        let n_freqs = 500;
        let freqs: Vec<f64> = (0..n_freqs)
            .map(|i| freq_min + (freq_max - freq_min) * (i as f64) / (n_freqs - 1) as f64)
            .collect();

        // Compute Lomb-Scargle periodogram
        let (f, power) = lombscargle(
            &t,
            &y,
            Some(&freqs),
            Some("standard"),
            Some(true),
            Some(true),
            None,
            None,
        )
        .unwrap();

        // Find the frequency with maximum power
        let max_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // The frequency with maximum power should be close to the true frequency
        assert_relative_eq!(f[max_idx], frequency, epsilon = 0.01);

        // Power at the true frequency should be close to 1.0 (for standard normalization)
        let power_at_true_freq = power[max_idx];
        assert!(power_at_true_freq > 0.9);
    }

    #[test]
    fn test_autofrequency() {
        // Test with evenly spaced times
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

        // FFT method
        let freqs_fft = autofrequency(&t, 1.0, Some(AutoFreqMethod::Fft)).unwrap();
        assert!(!freqs_fft.is_empty());

        // Linear method
        let freqs_linear = autofrequency(&t, 1.0, Some(AutoFreqMethod::Linear)).unwrap();
        assert!(!freqs_linear.is_empty());

        // Log method
        let freqs_log = autofrequency(&t, 1.0, Some(AutoFreqMethod::Log)).unwrap();
        assert!(!freqs_log.is_empty());

        // Check minimum and maximum frequencies
        let t_min = *t.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let t_max = *t.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let t_range = t_max - t_min;
        let dt = 0.1; // Known timestep

        let f_min = 1.0 / t_range;
        let f_max = 0.5 / dt;

        // Frequencies should be within the expected range
        for &f in &freqs_fft {
            assert!(f >= f_min);
            assert!(f <= f_max);
        }

        for &f in &freqs_linear {
            assert!(f >= f_min);
            assert!(f <= f_max);
        }

        for &f in &freqs_log {
            assert!(f >= f_min);
            assert!(f <= f_max);
        }
    }

    #[test]
    fn test_significance_levels() {
        // Create a test periodogram
        let power = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let fap_levels = vec![0.01, 0.05, 0.1];
        let n_samples = 100;

        // Calculate significance levels for different normalizations
        let sig_standard = significance_levels(&power, &fap_levels, "standard", n_samples).unwrap();
        let sig_model = significance_levels(&power, &fap_levels, "model", n_samples).unwrap();
        let sig_log = significance_levels(&power, &fap_levels, "log", n_samples).unwrap();
        let sig_psd = significance_levels(&power, &fap_levels, "psd", n_samples).unwrap();

        // Each normalization should return the correct number of levels
        assert_eq!(sig_standard.len(), fap_levels.len());
        assert_eq!(sig_model.len(), fap_levels.len());
        assert_eq!(sig_log.len(), fap_levels.len());
        assert_eq!(sig_psd.len(), fap_levels.len());

        // Thresholds should be ordered from highest to lowest (since FAP levels are from lowest to highest)
        for i in 1..fap_levels.len() {
            assert!(sig_standard[i - 1] > sig_standard[i]);
            assert!(sig_model[i - 1] > sig_model[i]);
            assert!(sig_log[i - 1] > sig_log[i]);
            assert!(sig_psd[i - 1] > sig_psd[i]);
        }
    }

    #[test]
    fn test_find_peaks() {
        // Create a test periodogram with known peaks
        let freq = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let power = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7, 0.6, 0.3];

        // Find peaks with threshold 0.5 and no grouping
        let (peak_freqs, peak_powers) = find_peaks(&freq, &power, 0.5, None).unwrap();

        // Should find peaks at indices 1, 3, 5, 7, 8
        assert_eq!(peak_freqs.len(), 5);
        assert!(peak_freqs.contains(&0.2));
        assert!(peak_freqs.contains(&0.4));
        assert!(peak_freqs.contains(&0.6));
        assert!(peak_freqs.contains(&0.8));
        assert!(peak_freqs.contains(&0.9));

        // Test with grouping
        let (grouped_freqs, grouped_powers) = find_peaks(&freq, &power, 0.5, Some(0.15)).unwrap();

        // With grouping, we should have fewer peaks (peaks within 0.15 of each other are grouped)
        assert!(grouped_freqs.len() < peak_freqs.len());
    }

    #[test]
    fn test_lombscargle_multi_frequency() {
        // Create a signal with two frequencies
        let freq1 = 0.1; // 0.1 Hz
        let freq2 = 0.3; // 0.3 Hz
        let n = 200;

        // Unevenly sampled time points
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            // Add some randomness to make sampling uneven
            let ti = i as f64 * 0.5 + 0.05 * (i as f64).cos();
            t.push(ti);
        }

        // Generate signal with two frequencies
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 0.8 * (2.0 * PI * freq1 * ti).sin() + 0.6 * (2.0 * PI * freq2 * ti).sin())
            .collect();

        // Define frequency grid
        let freq_min = 0.01;
        let freq_max = 0.5;
        let n_freqs = 1000;
        let freqs: Vec<f64> = (0..n_freqs)
            .map(|i| freq_min + (freq_max - freq_min) * (i as f64) / (n_freqs - 1) as f64)
            .collect();

        // Compute Lomb-Scargle periodogram
        let (f, power) = lombscargle(
            &t,
            &y,
            Some(&freqs),
            Some("standard"),
            Some(true),
            Some(true),
            None,
            None,
        )
        .unwrap();

        // Find peaks
        let (peak_freqs, peak_powers) = find_peaks(&f, &power, 0.5, Some(0.02)).unwrap();

        // Should find peaks near the two input frequencies
        assert!(peak_freqs.len() >= 2);

        // Check if the peaks are close to the true frequencies
        let mut found_freq1 = false;
        let mut found_freq2 = false;

        for &freq in &peak_freqs {
            if (freq - freq1).abs() < 0.02 {
                found_freq1 = true;
            }
            if (freq - freq2).abs() < 0.02 {
                found_freq2 = true;
            }
        }

        assert!(found_freq1);
        assert!(found_freq2);
    }
}
