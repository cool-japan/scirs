//! Utility functions for WPT validation
//!
//! This module contains helper functions for signal generation,
//! error calculation, and other common operations used throughout
//! the validation framework.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::Rng;
use std::f64::consts::PI;

/// Generate test signal based on configuration
pub fn generate_test_signal(config: &TestSignalConfig) -> SignalResult<Array1<f64>> {
    let length = config.length;
    let mut signal = Array1::zeros(length);
    let t: Vec<f64> = (0..length).map(|i| i as f64).collect();

    match config.signal_type {
        TestSignalType::Sinusoid => {
            let freq = config.parameters.get("frequency").unwrap_or(&0.1);
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            for (i, &ti) in t.iter().enumerate() {
                signal[i] = amplitude * (2.0 * PI * freq * ti / length as f64).sin();
            }
        }
        TestSignalType::Chirp => {
            let f0 = config.parameters.get("f0").unwrap_or(&0.05);
            let f1 = config.parameters.get("f1").unwrap_or(&0.4);
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            for (i, &ti) in t.iter().enumerate() {
                let freq = f0 + (f1 - f0) * ti / length as f64;
                signal[i] = amplitude * (2.0 * PI * freq * ti).sin();
            }
        }
        TestSignalType::WhiteNoise => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            let mut rng = scirs2_core::random::rng();
            for i in 0..length {
                signal[i] = amplitude * rng.gen_range(-1.0..1.0);
            }
        }
        TestSignalType::PinkNoise => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            let mut rng = scirs2_core::random::rng();
            // Simplified pink noise generation
            for i in 0..length {
                signal[i] = amplitude * rng.gen_range(-1.0..1.0) * (1.0 / (i + 1) as f64).sqrt();
            }
        }
        TestSignalType::Impulse => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            let position = config.parameters.get("position").unwrap_or(&0.5);
            let pos_idx = ((position * length as f64) as usize).min(length - 1);
            signal[pos_idx] = *amplitude;
        }
        TestSignalType::Step => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            let position = config.parameters.get("position").unwrap_or(&0.5);
            let pos_idx = ((position * length as f64) as usize).min(length - 1);
            for i in pos_idx..length {
                signal[i] = *amplitude;
            }
        }
        TestSignalType::Polynomial => {
            let degree = *config.parameters.get("degree").unwrap_or(&2.0) as usize;
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            for (i, &ti) in t.iter().enumerate() {
                let x = 2.0 * ti / length as f64 - 1.0; // Normalize to [-1, 1]
                signal[i] = amplitude * x.powi(degree as i32);
            }
        }
        TestSignalType::Piecewise => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            let segments = 4;
            let segment_length = length / segments;
            for i in 0..length {
                let segment = i / segment_length;
                signal[i] = amplitude * (segment % 2) as f64 * 2.0 - amplitude;
            }
        }
        TestSignalType::Fractal => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            let hurst = config.parameters.get("hurst").unwrap_or(&0.5);
            // Simplified fractal noise
            let mut rng = scirs2_core::random::rng();
            for i in 0..length {
                signal[i] = amplitude * rng.gen_range(-1.0..1.0) * ((i + 1) as f64).powf(-hurst);
            }
        }
        TestSignalType::Composite => {
            let amplitude = config.parameters.get("amplitude").unwrap_or(&1.0);
            // Composite of sinusoid and noise
            let mut rng = scirs2_core::random::rng();
            for (i, &ti) in t.iter().enumerate() {
                let sinusoid = (2.0 * PI * 0.1 * ti / length as f64).sin();
                let noise = 0.1 * rng.gen_range(-1.0..1.0);
                signal[i] = amplitude * (sinusoid + noise);
            }
        }
    }

    Ok(signal)
}

/// Calculate reconstruction error between original and reconstructed signals
pub fn calculate_reconstruction_error(
    original: &Array1<f64>,
    reconstructed: &Array1<f64>,
) -> SignalResult<ReconstructionError> {
    if original.len() != reconstructed.len() {
        return Err(SignalError::ValueError(
            "Signal lengths must match".to_string(),
        ));
    }

    let diff = original - reconstructed;
    let max_error = diff.mapv(|x| x.abs()).fold(0.0, |acc, &x| acc.max(x));
    let rms_error = (diff.mapv(|x| x * x).sum() / original.len() as f64).sqrt();

    Ok(ReconstructionError {
        max_error,
        rms_error,
    })
}

/// Calculate the L2 (Euclidean) norm of a vector
pub fn calculate_vector_norm(vec: &Array1<f64>) -> f64 {
    (vec.mapv(|x| x * x).sum()).sqrt()
}

/// Calculate the mean of an array
pub fn calculate_mean(data: &Array1<f64>) -> f64 {
    data.sum() / data.len() as f64
}

/// Calculate the standard deviation of an array
pub fn calculate_std_dev(data: &Array1<f64>) -> f64 {
    let mean = calculate_mean(data);
    let variance = data.mapv(|x| (x - mean).powi(2)).sum() / data.len() as f64;
    variance.sqrt()
}

/// Calculate the signal-to-noise ratio in dB
pub fn calculate_snr_db(signal: &Array1<f64>, noise: &Array1<f64>) -> SignalResult<f64> {
    if signal.len() != noise.len() {
        return Err(SignalError::ValueError(
            "Signal and noise must have the same length".to_string(),
        ));
    }

    let signal_power = signal.mapv(|x| x * x).sum() / signal.len() as f64;
    let noise_power = noise.mapv(|x| x * x).sum() / noise.len() as f64;

    if noise_power <= 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Normalize a signal to unit energy
pub fn normalize_signal(signal: &Array1<f64>) -> Array1<f64> {
    let norm = calculate_vector_norm(signal);
    if norm > 0.0 {
        signal / norm
    } else {
        signal.clone()
    }
}

/// Calculate correlation coefficient between two signals
pub fn calculate_correlation(signal1: &Array1<f64>, signal2: &Array1<f64>) -> SignalResult<f64> {
    if signal1.len() != signal2.len() {
        return Err(SignalError::ValueError(
            "Signals must have the same length".to_string(),
        ));
    }

    let mean1 = calculate_mean(signal1);
    let mean2 = calculate_mean(signal2);

    let numerator: f64 = signal1.iter().zip(signal2.iter())
        .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
        .sum();

    let sum_sq1: f64 = signal1.iter().map(|&x| (x - mean1).powi(2)).sum();
    let sum_sq2: f64 = signal2.iter().map(|&x| (x - mean2).powi(2)).sum();

    let denominator = (sum_sq1 * sum_sq2).sqrt();

    if denominator > 0.0 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

/// Generate a window function (Hanning window)
pub fn generate_hanning_window(length: usize) -> Array1<f64> {
    let mut window = Array1::zeros(length);
    for i in 0..length {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / (length - 1) as f64).cos());
    }
    window
}

/// Apply a window function to a signal
pub fn apply_window(signal: &Array1<f64>, window: &Array1<f64>) -> SignalResult<Array1<f64>> {
    if signal.len() != window.len() {
        return Err(SignalError::ValueError(
            "Signal and window must have the same length".to_string(),
        ));
    }

    Ok(signal * window)
}

/// Zero-pad a signal to a specified length
pub fn zero_pad(signal: &Array1<f64>, target_length: usize) -> Array1<f64> {
    if signal.len() >= target_length {
        return signal.clone();
    }

    let mut padded = Array1::zeros(target_length);
    for (i, &val) in signal.iter().enumerate() {
        padded[i] = val;
    }
    padded
}

/// Trim a signal to a specified length
pub fn trim_signal(signal: &Array1<f64>, target_length: usize) -> Array1<f64> {
    if signal.len() <= target_length {
        return signal.clone();
    }

    signal.slice(scirs2_core::ndarray::s![..target_length]).to_owned()
}

/// Calculate the energy of a signal
pub fn calculate_energy(signal: &Array1<f64>) -> f64 {
    signal.mapv(|x| x * x).sum()
}

/// Calculate the power of a signal
pub fn calculate_power(signal: &Array1<f64>) -> f64 {
    calculate_energy(signal) / signal.len() as f64
}

/// Find the maximum absolute value in a signal
pub fn find_max_abs(signal: &Array1<f64>) -> f64 {
    signal.mapv(|x| x.abs()).fold(0.0, |acc, &x| acc.max(x))
}

/// Check if a signal contains any infinite or NaN values
pub fn is_signal_valid(signal: &Array1<f64>) -> bool {
    signal.iter().all(|&x| x.is_finite())
}

/// Generate a random signal with specified properties
pub fn generate_random_signal(length: usize, amplitude: f64, seed: Option<u64>) -> Array1<f64> {
    let mut rng = if let Some(s) = seed {
        scirs2_core::random::rng()
    } else {
        scirs2_core::random::rng()
    };

    let mut signal = Array1::zeros(length);
    for i in 0..length {
        signal[i] = amplitude * rng.gen_range(-1.0..1.0);
    }
    signal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoid_generation() {
        let config = TestSignalConfig {
            signal_type: TestSignalType::Sinusoid,
            length: 100,
            parameters: [("frequency".to_string(), 1.0), ("amplitude".to_string(), 2.0)]
                .iter().cloned().collect(),
        };

        let signal = generate_test_signal(&config).unwrap();
        assert_eq!(signal.len(), 100);
        assert!(find_max_abs(&signal) <= 2.1); // Allow small numerical error
    }

    #[test]
    fn test_reconstruction_error() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let reconstructed = Array1::from_vec(vec![1.01, 1.99, 3.02, 3.98]);

        let error = calculate_reconstruction_error(&original, &reconstructed).unwrap();
        assert!(error.max_error < 0.1);
        assert!(error.rms_error < 0.1);
    }

    #[test]
    fn test_signal_statistics() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(calculate_mean(&signal), 3.0);
        assert!((calculate_std_dev(&signal) - (2.0_f64).sqrt()).abs() < 1e-10);
        assert!((calculate_vector_norm(&signal) - (55.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let signal1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let signal2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let correlation = calculate_correlation(&signal1, &signal2).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_calculation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let noise = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1]);

        let snr = calculate_snr_db(&signal, &noise).unwrap();
        assert!(snr > 10.0); // Should be significantly positive
    }

    #[test]
    fn test_window_functions() {
        let window = generate_hanning_window(10);
        assert_eq!(window.len(), 10);
        assert!((window[0] - 0.0).abs() < 1e-10);
        assert!((window[window.len()-1] - 0.0).abs() < 1e-10);

        let signal = Array1::ones(10);
        let windowed = apply_window(&signal, &window).unwrap();
        assert_eq!(windowed.len(), 10);
    }

    #[test]
    fn test_signal_validation() {
        let valid_signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(is_signal_valid(&valid_signal));

        let invalid_signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(!is_signal_valid(&invalid_signal));
    }

    #[test]
    fn test_signal_padding_and_trimming() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let padded = zero_pad(&signal, 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[4], 0.0);

        let trimmed = trim_signal(&padded, 2);
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[0], 1.0);
        assert_eq!(trimmed[1], 2.0);
    }
}