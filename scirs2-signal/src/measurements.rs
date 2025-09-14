// Signal measurements and analysis functions
//
// This module provides functions for measuring various properties of signals,
// such as RMS level, peak-to-peak amplitude, signal-to-noise ratio (SNR),
// and total harmonic distortion (THD).

use crate::error::{SignalError, SignalResult};
use num_traits::{Float, NumCast};
use rand::Rng;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Calculate the root mean square (RMS) level of a signal.
///
/// # Arguments
///
/// * `x` - Input signal
///
/// # Returns
///
/// * RMS value of the signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::measurements::rms;
///
/// // Calculate RMS of a sine wave
/// let signal = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
/// let rms_value = rms(&signal).unwrap();
///
/// // RMS of a sine wave with amplitude 1 is 1/√2 ≈ 0.707
/// assert!(((rms_value - 0.707) as f64).abs() < 0.01);
/// ```
#[allow(dead_code)]
pub fn rms<T>(x: &[T]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate RMS: sqrt(mean(x²))
    let sum_of_squares: f64 = x_f64.iter().map(|&x| x * x).sum();
    let mean_square = sum_of_squares / x_f64.len() as f64;
    let rms = mean_square.sqrt();

    Ok(rms)
}

/// Calculate the peak-to-peak amplitude of a signal.
///
/// # Arguments
///
/// * `x` - Input signal
///
/// # Returns
///
/// * Peak-to-peak amplitude of the signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::measurements::peak_to_peak;
///
/// // Calculate peak-to-peak amplitude of a signal
/// let signal = vec![-1.5, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5];
/// let pp_value = peak_to_peak(&signal).unwrap();
///
/// assert_eq!(pp_value, 2.5); // Max 1.0 - Min -1.5 = 2.5
/// ```
#[allow(dead_code)]
pub fn peak_to_peak<T>(x: &[T]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Find minimum and maximum
    let min_val = x_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = x_f64.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Calculate peak-to-peak amplitude
    let peak_to_peak = max_val - min_val;

    Ok(peak_to_peak)
}

/// Calculate the crest factor of a signal (peak amplitude to RMS ratio).
///
/// # Arguments
///
/// * `x` - Input signal
///
/// # Returns
///
/// * Peak-to-RMS ratio of the signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::measurements::peak_to_rms;
///
/// // Calculate crest factor of a sine wave
/// let signal = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
/// let crest_factor = peak_to_rms(&signal).unwrap();
///
/// // Crest factor of a sine wave is √2 ≈ 1.414
/// assert!(((crest_factor - 1.414) as f64).abs() < 0.01);
/// ```
#[allow(dead_code)]
pub fn peak_to_rms<T>(x: &[T]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate RMS
    let rms_val = rms(&x_f64)?;

    // Find peak amplitude (maximum absolute value)
    let peak = x_f64.iter().fold(0.0, |a, &b| a.max(b.abs()));

    // Calculate peak-to-RMS ratio
    let peak_to_rms = peak / rms_val;

    Ok(peak_to_rms)
}

/// Calculate the signal-to-noise ratio (SNR) in decibels.
///
/// # Arguments
///
/// * `signal` - Clean signal reference
/// * `signal_plusnoise` - Signal with noise
///
/// # Returns
///
/// * SNR in decibels
///
/// # Examples
///
/// ```
/// use scirs2_signal::measurements::snr;
/// use rand::Rng;  // Import the Rng trait to access gen_range
///
/// // Create a clean signal
/// let clean = (0..100).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
///
/// // Add noise to create signal_plusnoise
/// let mut rng = rand::rng();
/// let noisy: Vec<f64> = clean.iter()
///     .map(|&x| x + rng.gen_range(-0.1f64..0.1f64))
///     .collect();
///
/// // Calculate SNR
/// let snr_db = snr(&clean, &noisy).unwrap();
/// ```
#[allow(dead_code)]
pub fn snr<T, U>(signal: &[T], signal_plusnoise: &[U]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if signal.is_empty() || signal_plusnoise.is_empty() {
        return Err(SignalError::ValueError(
            "Input signals are empty".to_string(),
        ));
    }

    if signal.len() != signal_plusnoise.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Signal lengths do not match: {} vs {}",
            signal.len(),
            signal_plusnoise.len()
        )));
    }

    // Convert to f64 for internal processing
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let signal_plusnoise_f64: Vec<f64> = signal_plusnoise
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate _noise by subtracting _signal from signal_plusnoise
    let noise: Vec<f64> = signal_f64
        .iter()
        .zip(signal_plusnoise_f64.iter())
        .map(|(&s, &spn)| spn - s)
        .collect();

    // Calculate power of _signal and _noise
    let signal_power: f64 =
        signal_f64.iter().map(|&x| x * x).sum::<f64>() / signal_f64.len() as f64;
    let noise_power: f64 = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() as f64;

    // Avoid division by zero
    if noise_power.abs() < f64::EPSILON {
        return Err(SignalError::ComputationError(
            "Noise power is too small, SNR calculation would result in division by zero"
                .to_string(),
        ));
    }

    // Calculate SNR in decibels: 10 * log10(signal_power / noise_power)
    let snr_db = 10.0 * (signal_power / noise_power).log10();

    Ok(snr_db)
}

/// Calculate the total harmonic distortion (THD) of a signal.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `fs` - Sampling frequency
/// * `f0` - Fundamental frequency
/// * `n_harmonics` - Number of harmonics to include (default 5)
///
/// # Returns
///
/// * THD as a decimal fraction
///
/// # Examples
///
/// ```
/// use scirs2_signal::measurements::thd;
///
/// // Create a signal with some harmonic distortion
/// let fs = 1000.0; // 1 kHz sampling rate
/// let f0 = 100.0;  // 100 Hz fundamental
/// let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();
///
/// // Signal with fundamental and some harmonics
/// let signal: Vec<f64> = t.iter()
///     .map(|&t| (2.0 * std::f64::consts::PI * f0 * t).sin() +
///               0.1 * (2.0 * std::f64::consts::PI * 2.0 * f0 * t).sin() +
///               0.05 * (2.0 * std::f64::consts::PI * 3.0 * f0 * t).sin())
///     .collect();
///
/// // Calculate THD
/// let thd_val = thd(&signal, fs, f0, Some(5)).unwrap();
///
/// // THD should be approximately sqrt(0.1² + 0.05²)/1.0 ≈ 0.112
/// assert!(((thd_val - 0.112) as f64).abs() < 0.01);
/// ```
#[allow(dead_code)]
pub fn thd<T>(_signal: &[T], fs: f64, f0: f64, nharmonics: Option<usize>) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
{
    if _signal.is_empty() {
        return Err(SignalError::ValueError(
            "Input _signal is empty".to_string(),
        ));
    }

    if fs <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Sampling frequency must be positive, got {}",
            fs
        )));
    }

    if f0 <= 0.0 || f0 >= fs / 2.0 {
        return Err(SignalError::ValueError(format!(
            "Fundamental frequency must be positive and less than Nyquist frequency (fs/2), got {}",
            f0
        )));
    }

    // Get number of _harmonics or use default
    let n_harmonics = nharmonics.unwrap_or(5);

    // Convert to f64 for internal processing
    let signal_f64: Vec<f64> = _signal
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Simple implementation using FFT
    // This would be better implemented using rustfft when available.
    // For now, we'll implement a basic DFT algorithm directly

    let n = signal_f64.len();
    let fft_bin_size = fs / n as f64;

    // Find the FFT bin corresponding to the fundamental frequency
    let f0_bin = (f0 / fft_bin_size).round() as usize;

    // Calculate fundamental amplitude
    let mut fundamental_power = 0.0;
    for k in 0..n {
        let phi = 2.0 * std::f64::consts::PI * (f0_bin as f64 * k as f64) / n as f64;
        let real = signal_f64
            .iter()
            .enumerate()
            .map(|(i, &x)| x * (phi * i as f64).cos())
            .sum::<f64>();
        let imag = signal_f64
            .iter()
            .enumerate()
            .map(|(i, &x)| x * (phi * i as f64).sin())
            .sum::<f64>();

        fundamental_power = (real * real + imag * imag) / (n * n) as f64;
    }

    // Calculate harmonic amplitudes
    let mut harmonic_power_sum = 0.0;
    for h in 2..(n_harmonics + 2) {
        let h_bin = (h as f64 * f0 / fft_bin_size).round() as usize;
        if h_bin >= n / 2 {
            // Skip _harmonics above Nyquist frequency
            break;
        }

        // Calculate harmonic amplitude
        let mut h_power = 0.0;
        for k in 0..n {
            let phi = 2.0 * std::f64::consts::PI * (h_bin as f64 * k as f64) / n as f64;
            let real = signal_f64
                .iter()
                .enumerate()
                .map(|(i, &x)| x * (phi * i as f64).cos())
                .sum::<f64>();
            let imag = signal_f64
                .iter()
                .enumerate()
                .map(|(i, &x)| x * (phi * i as f64).sin())
                .sum::<f64>();

            h_power = (real * real + imag * imag) / (n * n) as f64;
        }

        harmonic_power_sum += h_power;
    }

    // Avoid division by zero
    if fundamental_power.abs() < f64::EPSILON {
        return Err(SignalError::ComputationError(
            "Fundamental power is too small, THD calculation would result in division by zero"
                .to_string(),
        ));
    }

    // Calculate THD as sqrt(sum of harmonic powers / fundamental power)
    let thd = (harmonic_power_sum / fundamental_power).sqrt();

    Ok(thd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;
    use std::f64::consts::PI;
    #[test]
    fn test_rms() {
        // DC signal
        let dc_signal = vec![2.0; 100];
        let rms_val = rms(&dc_signal).unwrap();
        assert_relative_eq!(rms_val, 2.0, epsilon = 1e-10);

        // Sine wave with amplitude 1
        let sine_wave: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
            .collect();
        let rms_val = rms(&sine_wave).unwrap();
        assert_relative_eq!(rms_val, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-2);
    }

    #[test]
    fn test_peak_to_peak() {
        // Signal with known min and max
        let signal = vec![-1.5, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5];
        let pp_val = peak_to_peak(&signal).unwrap();
        assert_relative_eq!(pp_val, 2.5, epsilon = 1e-10);

        // Sine wave with amplitude 1
        let sine_wave: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
            .collect();
        let pp_val = peak_to_peak(&sine_wave).unwrap();
        assert_relative_eq!(pp_val, 2.0, epsilon = 1e-2);
    }

    #[test]
    fn test_peak_to_rms() {
        // DC signal (crest factor = 1)
        let dc_signal = vec![2.0; 100];
        let cf_val = peak_to_rms(&dc_signal).unwrap();
        assert_relative_eq!(cf_val, 1.0, epsilon = 1e-10);

        // Sine wave (crest factor = sqrt(2))
        let sine_wave: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
            .collect();
        let cf_val = peak_to_rms(&sine_wave).unwrap();
        assert_relative_eq!(cf_val, 2.0_f64.sqrt(), epsilon = 1e-2);
    }

    #[test]
    fn test_snr() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a clean signal
        let clean: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
            .collect();

        // Add noise with known power
        let noise_amplitude = 0.1; // -20 dB relative to signal
                                   // Using rand API that's compatible with 0.9.0
        let mut rng = rand::rng();
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&x| x + noise_amplitude * (2.0 * PI * rng.gen_range(0.0..1.0)).sin())
            .collect();

        // Calculate SNR
        let snr_db = snr(&clean, &noisy).unwrap();

        // Expected SNR ≈ 20 dB (signal amplitude = 1, noise amplitude = 0.1)
        // In power: (1^2)/(0.1^2) = 100 => 10*log10(100) = 20 dB
        assert!(((snr_db - 20.0) as f64).abs() < 1.0);
    }
}
