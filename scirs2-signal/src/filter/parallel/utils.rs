//! Utility functions for parallel filtering operations
//!
//! This module provides helper functions and utilities used across
//! the parallel filtering modules.

use crate::error::{SignalError, SignalResult};
use rand::Rng;

/// Generate a test signal for filter testing
///
/// Creates a composite test signal with multiple frequency components
/// and optional noise for testing filter implementations.
///
/// # Arguments
///
/// * `length` - Length of the signal to generate
/// * `frequencies` - Frequencies to include in the signal
/// * `amplitudes` - Corresponding amplitudes for each frequency
/// * `noise_level` - Standard deviation of additive white noise
/// * `sample_rate` - Sample rate for the signal
///
/// # Returns
///
/// * Generated test signal
pub fn generate_test_signal(
    length: usize,
    frequencies: &[f64],
    amplitudes: &[f64],
    noise_level: f64,
    sample_rate: f64,
) -> SignalResult<Vec<f64>> {
    if frequencies.len() != amplitudes.len() {
        return Err(SignalError::ValueError(
            "Frequencies and amplitudes must have the same length".to_string(),
        ));
    }

    if length == 0 {
        return Ok(Vec::new());
    }

    let mut signal = vec![0.0; length];
    let dt = 1.0 / sample_rate;

    // Add frequency components
    for (i, &freq) in frequencies.iter().enumerate() {
        let amplitude = amplitudes[i];
        for (n, sample) in signal.iter_mut().enumerate() {
            let t = n as f64 * dt;
            *sample += amplitude * (2.0 * std::f64::consts::PI * freq * t).sin();
        }
    }

    // Add noise if requested
    if noise_level > 0.0 {
        for sample in signal.iter_mut() {
            *sample += generate_gaussian_noise(noise_level);
        }
    }

    Ok(signal)
}

/// Generate white Gaussian noise sample
fn generate_gaussian_noise(noise_level: f64) -> f64 {
    // Box-Muller transform for Gaussian noise
    use std::f64::consts::PI;

    static mut HAVE_SPARE: bool = false;
    static mut SPARE: f64 = 0.0;

    unsafe {
        if HAVE_SPARE {
            HAVE_SPARE = false;
            return SPARE;
        }

        HAVE_SPARE = true;

        let mut rng = rand::rng();
        let u = rng.gen::<f64>();
        let v = rng.gen::<f64>();

        let magnitude = noise_level * (-2.0_f64 * u.ln()).sqrt();
        let angle = 2.0 * PI * v;

        SPARE = magnitude * angle.cos();
        magnitude * angle.sin()
    }
}

/// Compute signal-to-noise ratio (SNR)
///
/// Calculates the SNR between a clean signal and a noisy signal.
///
/// # Arguments
///
/// * `clean_signal` - Original clean signal
/// * `noisy_signal` - Signal with noise added
///
/// # Returns
///
/// * SNR in decibels
pub fn compute_snr_db(clean_signal: &[f64], noisy_signal: &[f64]) -> SignalResult<f64> {
    if clean_signal.len() != noisy_signal.len() {
        return Err(SignalError::ValueError(
            "Signals must have the same length".to_string(),
        ));
    }

    if clean_signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signals cannot be empty".to_string(),
        ));
    }

    // Calculate signal power
    let signal_power: f64 = clean_signal.iter().map(|&x| x * x).sum();

    // Calculate noise power
    let noise_power: f64 = clean_signal
        .iter()
        .zip(noisy_signal.iter())
        .map(|(&clean, &noisy)| (clean - noisy).powi(2))
        .sum();

    if noise_power <= 0.0 {
        return Ok(f64::INFINITY); // Perfect signal (no noise)
    }

    let snr_linear = signal_power / noise_power;
    let snr_db = 10.0 * snr_linear.log10();

    Ok(snr_db)
}

/// Compute mean squared error (MSE) between two signals
///
/// # Arguments
///
/// * `reference` - Reference signal
/// * `test` - Test signal to compare
///
/// # Returns
///
/// * Mean squared error
pub fn compute_mse(reference: &[f64], test: &[f64]) -> SignalResult<f64> {
    if reference.len() != test.len() {
        return Err(SignalError::ValueError(
            "Signals must have the same length".to_string(),
        ));
    }

    if reference.is_empty() {
        return Ok(0.0);
    }

    let mse: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(&r, &t)| (r - t).powi(2))
        .sum::<f64>()
        / reference.len() as f64;

    Ok(mse)
}

/// Normalize signal to unit energy
///
/// # Arguments
///
/// * `signal` - Input signal to normalize
///
/// # Returns
///
/// * Normalized signal
pub fn normalize_signal_energy(signal: &[f64]) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Ok(Vec::new());
    }

    let energy: f64 = signal.iter().map(|&x| x * x).sum();

    if energy <= 0.0 {
        return Err(SignalError::ValueError(
            "Signal has zero or negative energy".to_string(),
        ));
    }

    let norm_factor = energy.sqrt();
    let normalized: Vec<f64> = signal.iter().map(|&x| x / norm_factor).collect();

    Ok(normalized)
}

/// Normalize signal to unit amplitude (peak normalization)
///
/// # Arguments
///
/// * `signal` - Input signal to normalize
///
/// # Returns
///
/// * Normalized signal
pub fn normalize_signal_amplitude(signal: &[f64]) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Ok(Vec::new());
    }

    let max_abs = signal.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);

    if max_abs <= 0.0 {
        return Err(SignalError::ValueError(
            "Signal has zero amplitude".to_string(),
        ));
    }

    let normalized: Vec<f64> = signal.iter().map(|&x| x / max_abs).collect();

    Ok(normalized)
}

/// Zero-pad signal to specified length
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `target_length` - Target length after padding
/// * `mode` - Padding mode ("center", "end")
///
/// # Returns
///
/// * Zero-padded signal
pub fn zero_pad_signal(signal: &[f64], target_length: usize, mode: &str) -> SignalResult<Vec<f64>> {
    if target_length < signal.len() {
        return Err(SignalError::ValueError(
            "Target length must be at least as long as input signal".to_string(),
        ));
    }

    if target_length == signal.len() {
        return Ok(signal.to_vec());
    }

    let pad_total = target_length - signal.len();

    match mode {
        "center" => {
            let pad_left = pad_total / 2;
            let pad_right = pad_total - pad_left;

            let mut padded = Vec::with_capacity(target_length);
            padded.extend(vec![0.0; pad_left]);
            padded.extend_from_slice(signal);
            padded.extend(vec![0.0; pad_right]);

            Ok(padded)
        }
        "end" => {
            let mut padded = Vec::with_capacity(target_length);
            padded.extend_from_slice(signal);
            padded.extend(vec![0.0; pad_total]);

            Ok(padded)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown padding mode: {}. Use 'center' or 'end'",
            mode
        ))),
    }
}

/// Compute cross-correlation between two signals
///
/// # Arguments
///
/// * `x` - First signal
/// * `y` - Second signal
/// * `mode` - Correlation mode ("full", "valid", "same")
///
/// # Returns
///
/// * Cross-correlation result
pub fn cross_correlation(x: &[f64], y: &[f64], mode: &str) -> SignalResult<Vec<f64>> {
    if x.is_empty() || y.is_empty() {
        return Err(SignalError::ValueError(
            "Input signals cannot be empty".to_string(),
        ));
    }

    let m = x.len();
    let n = y.len();

    match mode {
        "full" => {
            let output_len = m + n - 1;
            let mut result = vec![0.0; output_len];

            for i in 0..output_len {
                let mut sum = 0.0;
                let start_j = if i >= n - 1 { i - (n - 1) } else { 0 };
                let end_j = (i + 1).min(m);

                for j in start_j..end_j {
                    let y_idx = i - j;
                    sum += x[j] * y[y_idx];
                }
                result[i] = sum;
            }
            Ok(result)
        }
        "valid" => {
            if n > m {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, first signal must not be shorter than second".to_string(),
                ));
            }

            let output_len = m - n + 1;
            let mut result = vec![0.0; output_len];

            for i in 0..output_len {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += x[i + j] * y[j];
                }
                result[i] = sum;
            }
            Ok(result)
        }
        "same" => {
            let output_len = m;
            let mut result = vec![0.0; output_len];
            let offset = (n - 1) / 2;

            for i in 0..output_len {
                let mut sum = 0.0;
                for j in 0..n {
                    let x_idx = i + j;
                    if x_idx >= offset && x_idx - offset < m {
                        sum += x[x_idx - offset] * y[j];
                    }
                }
                result[i] = sum;
            }
            Ok(result)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown correlation mode: {}",
            mode
        ))),
    }
}

/// Find local maxima in a signal
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `min_distance` - Minimum distance between peaks
/// * `threshold` - Minimum height threshold
///
/// # Returns
///
/// * Indices of detected peaks
pub fn find_peaks(signal: &[f64], min_distance: usize, threshold: f64) -> Vec<usize> {
    if signal.len() < 3 {
        return Vec::new();
    }

    let mut peaks = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > threshold {
            // Check minimum distance constraint
            let too_close = peaks
                .iter()
                .any(|&peak_idx| (i as isize - peak_idx as isize).abs() < min_distance as isize);

            if !too_close {
                peaks.push(i);
            }
        }
    }

    peaks
}

/// Compute power spectral density using Welch's method
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of analysis windows
/// * `overlap_factor` - Overlap factor (0.0 to 1.0)
///
/// # Returns
///
/// * Power spectral density estimate
pub fn welch_psd_estimate(
    signal: &[f64],
    window_size: usize,
    overlap_factor: f64,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() || window_size == 0 {
        return Err(SignalError::ValueError(
            "Signal and window size must be non-zero".to_string(),
        ));
    }

    if overlap_factor < 0.0 || overlap_factor >= 1.0 {
        return Err(SignalError::ValueError(
            "Overlap factor must be between 0.0 and 1.0".to_string(),
        ));
    }

    let step_size = ((1.0 - overlap_factor) * window_size as f64) as usize;
    if step_size == 0 {
        return Err(SignalError::ValueError(
            "Step size cannot be zero".to_string(),
        ));
    }

    let n_windows = (signal.len() - window_size) / step_size + 1;
    if n_windows == 0 {
        return Err(SignalError::ValueError(
            "Signal too short for analysis".to_string(),
        ));
    }

    // Initialize PSD accumulator
    let mut psd = vec![0.0; window_size / 2 + 1];

    // Process each window
    for window_idx in 0..n_windows {
        let start = window_idx * step_size;
        let end = (start + window_size).min(signal.len());

        if end - start < window_size {
            break;
        }

        let window_data = &signal[start..end];

        // Apply Hanning window
        let windowed: Vec<f64> = window_data
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = 0.5
                    * (1.0
                        - (2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64).cos());
                x * w
            })
            .collect();

        // Compute FFT (simplified - in practice would use efficient FFT)
        for k in 0..psd.len() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (n, &x) in windowed.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / window_size as f64;
                real += x * angle.cos();
                imag += x * angle.sin();
            }

            let magnitude_squared = real * real + imag * imag;
            psd[k] += magnitude_squared;
        }
    }

    // Normalize by number of windows
    for p in psd.iter_mut() {
        *p /= n_windows as f64;
    }

    Ok(psd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_generate_test_signal() {
        let frequencies = vec![10.0, 20.0];
        let amplitudes = vec![1.0, 0.5];

        let signal = generate_test_signal(100, &frequencies, &amplitudes, 0.0, 1000.0).unwrap();
        assert_eq!(signal.len(), 100);
    }

    #[test]
    fn test_compute_snr_db() {
        let clean = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let noisy = vec![1.1, 2.1, 3.1, 2.1, 1.1]; // Small amount of noise

        let snr = compute_snr_db(&clean, &noisy).unwrap();
        assert!(snr > 0.0); // Should have positive SNR
    }

    #[test]
    fn test_compute_mse() {
        let signal1 = vec![1.0, 2.0, 3.0];
        let signal2 = vec![1.0, 2.0, 3.0];

        let mse = compute_mse(&signal1, &signal2).unwrap();
        assert!((mse - 0.0).abs() < 1e-10); // Should be zero for identical signals

        let signal3 = vec![1.0, 2.0, 4.0];
        let mse = compute_mse(&signal1, &signal3).unwrap();
        assert!(mse > 0.0); // Should be positive for different signals
    }

    #[test]
    fn test_normalize_signal_energy() {
        let signal = vec![3.0, 4.0]; // Energy = 9 + 16 = 25, norm = 5
        let normalized = normalize_signal_energy(&signal).unwrap();

        let energy: f64 = normalized.iter().map(|&x| x * x).sum();
        assert!((energy - 1.0).abs() < 1e-10); // Should have unit energy
    }

    #[test]
    fn test_normalize_signal_amplitude() {
        let signal = vec![-2.0, 1.0, 4.0, -3.0]; // Max abs = 4
        let normalized = normalize_signal_amplitude(&signal).unwrap();

        let max_abs = normalized.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        assert!((max_abs - 1.0).abs() < 1e-10); // Should have unit amplitude
    }

    #[test]
    fn test_zero_pad_signal() {
        let signal = vec![1.0, 2.0, 3.0];

        // Test center padding
        let padded = zero_pad_signal(&signal, 7, "center").unwrap();
        assert_eq!(padded.len(), 7);
        assert_eq!(padded[2], 1.0); // Original signal should be centered

        // Test end padding
        let padded = zero_pad_signal(&signal, 6, "end").unwrap();
        assert_eq!(padded.len(), 6);
        assert_eq!(padded[0], 1.0); // Original signal should be at start
    }

    #[test]
    fn test_cross_correlation() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.5, 1.0];

        let corr = cross_correlation(&x, &y, "full").unwrap();
        assert_eq!(corr.len(), x.len() + y.len() - 1);

        let corr = cross_correlation(&x, &y, "valid").unwrap();
        assert_eq!(corr.len(), x.len() - y.len() + 1);

        let corr = cross_correlation(&x, &y, "same").unwrap();
        assert_eq!(corr.len(), x.len());
    }

    #[test]
    fn test_find_peaks() {
        let signal = vec![0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
        let peaks = find_peaks(&signal, 2, 0.5);

        assert!(peaks.contains(&3)); // Peak at index 3 (value 3.0)
        assert!(peaks.contains(&5)); // Peak at index 5 (value 2.0)
    }

    #[test]
    fn test_welch_psd_estimate() {
        // Generate a simple sinusoid
        let signal: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 256.0).sin())
            .collect();

        let psd = welch_psd_estimate(&signal, 64, 0.5).unwrap();
        assert!(!psd.is_empty());
        assert_eq!(psd.len(), 33); // 64/2 + 1
    }

    #[test]
    fn test_error_conditions() {
        // Test mismatched signal lengths
        let signal1 = vec![1.0, 2.0, 3.0];
        let signal2 = vec![1.0, 2.0];
        let result = compute_snr_db(&signal1, &signal2);
        assert!(result.is_err());

        // Test invalid padding mode
        let result = zero_pad_signal(&signal1, 5, "invalid");
        assert!(result.is_err());

        // Test invalid correlation mode
        let result = cross_correlation(&signal1, &signal2, "invalid");
        assert!(result.is_err());
    }
}
