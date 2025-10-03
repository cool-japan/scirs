// Utility functions for Lomb-Scargle validation
//
// This module provides common utility functions used across different validation modules.

use crate::error::{SignalError, SignalResult};

/// Find peaks in a power spectrum using a simple threshold
#[allow(dead_code)]
pub fn find_peaks_simple(power: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();

    for i in 1..power.len().saturating_sub(1) {
        if power[i] > threshold && power[i] > power[i-1] && power[i] > power[i+1] {
            peaks.push(i);
        }
    }

    peaks
}

/// Find peaks in a power spectrum (more sophisticated version)
#[allow(dead_code)]
pub fn find_peaks(power: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = power.len();

    if n < 3 {
        return peaks;
    }

    for i in 1..n-1 {
        let current = power[i];
        let prev = power[i-1];
        let next = power[i+1];

        // Check if it's a local maximum above threshold
        if current > threshold && current > prev && current > next {
            // Additional check: ensure it's significantly above neighboring values
            let min_neighbor = prev.min(next);
            if current > min_neighbor * 1.1 { // 10% higher than neighbors
                peaks.push(i);
            }
        }
    }

    peaks
}

/// Calculate relative error between two values
#[allow(dead_code)]
pub fn relative_error(actual: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-15 {
        if actual.abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

/// Calculate Root Mean Square Error between two vectors
#[allow(dead_code)]
pub fn rmse(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() {
        return f64::NAN;
    }

    let sum_sq_diff: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).powi(2))
        .sum();

    (sum_sq_diff / x.len() as f64).sqrt()
}

/// Calculate Mean Absolute Error between two vectors
#[allow(dead_code)]
pub fn mae(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() {
        return f64::NAN;
    }

    let sum_abs_diff: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .sum();

    sum_abs_diff / x.len() as f64
}

/// Calculate the maximum absolute error between two vectors
#[allow(dead_code)]
pub fn max_abs_error(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() {
        return f64::NAN;
    }

    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .fold(0.0, f64::max)
}

/// Check if two floating point numbers are approximately equal
#[allow(dead_code)]
pub fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance || ((a - b).abs() / (a.abs() + b.abs() + 1e-15)) <= tolerance
}

/// Check if two vectors are approximately equal element-wise
#[allow(dead_code)]
pub fn vectors_approx_equal(x: &[f64], y: &[f64], tolerance: f64) -> bool {
    if x.len() != y.len() {
        return false;
    }

    x.iter()
        .zip(y.iter())
        .all(|(&xi, &yi)| approx_equal(xi, yi, tolerance))
}

/// Generate linearly spaced values between start and end
#[allow(dead_code)]
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }

    let step = (end - start) / (n - 1) as f64;
    (0..n)
        .map(|i| start + i as f64 * step)
        .collect()
}

/// Calculate the power spectral density normalization factor
#[allow(dead_code)]
pub fn psd_normalization_factor(n: usize, sampling_rate: f64) -> f64 {
    2.0 / (n as f64 * sampling_rate)
}

/// Convert power to decibels
#[allow(dead_code)]
pub fn power_to_db(power: f64, reference: f64) -> f64 {
    10.0 * (power / reference).log10()
}

/// Convert decibels to power
#[allow(dead_code)]
pub fn db_to_power(db: f64, reference: f64) -> f64 {
    reference * 10.0_f64.powf(db / 10.0)
}

/// Calculate signal-to-noise ratio in dB
#[allow(dead_code)]
pub fn snr_db(signal_power: f64, noise_power: f64) -> f64 {
    10.0 * (signal_power / noise_power).log10()
}

/// Generate white noise with specified standard deviation
#[allow(dead_code)]
pub fn generate_white_noise(n: usize, std_dev: f64) -> Vec<f64> {
    use scirs2_core::random::prelude::*;
    let mut rng = scirs2_core::random::rng();

    (0..n)
        .map(|_| std_dev * rng.gen_range(-1.0..1.0))
        .collect()
}

/// Calculate the frequency resolution for a given sampling rate and number of samples
#[allow(dead_code)]
pub fn frequency_resolution(sampling_rate: f64, n_samples: usize) -> f64 {
    sampling_rate / n_samples as f64
}

/// Calculate the Nyquist frequency for a given sampling rate
#[allow(dead_code)]
pub fn nyquist_frequency(sampling_rate: f64) -> f64 {
    sampling_rate / 2.0
}

/// Validate that a frequency vector is monotonically increasing
#[allow(dead_code)]
pub fn validate_frequency_vector(freqs: &[f64]) -> SignalResult<()> {
    if freqs.is_empty() {
        return Err(SignalError::InvalidInput("Frequency vector cannot be empty".to_string()));
    }

    for i in 1..freqs.len() {
        if freqs[i] <= freqs[i-1] {
            return Err(SignalError::InvalidInput(
                "Frequency vector must be monotonically increasing".to_string()
            ));
        }
    }

    Ok(())
}

/// Validate that time and signal vectors have the same length
#[allow(dead_code)]
pub fn validate_time_signal_lengths(t: &[f64], y: &[f64]) -> SignalResult<()> {
    if t.len() != y.len() {
        return Err(SignalError::InvalidInput(
            format!("Time and signal vectors must have the same length: {} vs {}",
                   t.len(), y.len())
        ));
    }

    if t.is_empty() {
        return Err(SignalError::InvalidInput("Time and signal vectors cannot be empty".to_string()));
    }

    Ok(())
}