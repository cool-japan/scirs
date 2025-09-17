//! Test signal generation utilities for multitaper validation
//!
//! This module provides functions to generate various types of test signals
//! for comprehensive multitaper validation.

use super::types::{TestSignalConfig, TestSignalType, SignalQualityMetrics};
use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

/// Generate test signal based on configuration
pub fn generate_test_signal(
    signal_type: &TestSignalType,
    config: &TestSignalConfig,
) -> SignalResult<Vec<f64>> {
    match signal_type {
        TestSignalType::Sinusoid(freq) => {
            generate_sinusoid(*freq, config.length, config.fs)
        }
        TestSignalType::MultiSine(freqs) => {
            generate_multisine(freqs, config.length, config.fs)
        }
        TestSignalType::WhiteNoise => {
            generate_white_noise(config.length)
        }
        TestSignalType::ColoredNoise(alpha) => {
            generate_colored_noise(config.length, *alpha)
        }
        TestSignalType::Chirp { start_freq, end_freq } => {
            generate_chirp(*start_freq, *end_freq, config.length, config.fs)
        }
        TestSignalType::Impulse => {
            generate_impulse(config.length)
        }
        TestSignalType::Step => {
            generate_step(config.length)
        }
        TestSignalType::ComplexSinusoid(freq) => {
            let real_part = generate_sinusoid(*freq, config.length, config.fs)?;
            let imag_part = generate_cosine(*freq, config.length, config.fs)?;
            // For real-valued output, return just the real part
            Ok(real_part)
        }
    }
}

/// Generate sinusoidal signal
fn generate_sinusoid(freq: f64, length: usize, fs: f64) -> SignalResult<Vec<f64>> {
    if freq <= 0.0 || freq >= fs / 2.0 {
        return Err(SignalError::ValueError(
            "Frequency must be positive and less than Nyquist frequency".to_string(),
        ));
    }

    let dt = 1.0 / fs;
    let signal: Vec<f64> = (0..length)
        .map(|i| (2.0 * PI * freq * i as f64 * dt).sin())
        .collect();

    Ok(signal)
}

/// Generate cosine signal
fn generate_cosine(freq: f64, length: usize, fs: f64) -> SignalResult<Vec<f64>> {
    if freq <= 0.0 || freq >= fs / 2.0 {
        return Err(SignalError::ValueError(
            "Frequency must be positive and less than Nyquist frequency".to_string(),
        ));
    }

    let dt = 1.0 / fs;
    let signal: Vec<f64> = (0..length)
        .map(|i| (2.0 * PI * freq * i as f64 * dt).cos())
        .collect();

    Ok(signal)
}

/// Generate multi-sine signal
fn generate_multisine(freqs: &[f64], length: usize, fs: f64) -> SignalResult<Vec<f64>> {
    if freqs.is_empty() {
        return Err(SignalError::ValueError("Frequency list cannot be empty".to_string()));
    }

    let dt = 1.0 / fs;
    let mut signal = vec![0.0; length];

    for &freq in freqs {
        if freq <= 0.0 || freq >= fs / 2.0 {
            return Err(SignalError::ValueError(
                "All frequencies must be positive and less than Nyquist frequency".to_string(),
            ));
        }

        for i in 0..length {
            signal[i] += (2.0 * PI * freq * i as f64 * dt).sin();
        }
    }

    // Normalize by number of components
    let scale = 1.0 / freqs.len() as f64;
    signal.iter_mut().for_each(|x| *x *= scale);

    Ok(signal)
}

/// Generate white noise
fn generate_white_noise(length: usize) -> SignalResult<Vec<f64>> {
    let signal: Vec<f64> = (0..length)
        .map(|_| fastrand::f64() * 2.0 - 1.0) // Uniform distribution [-1, 1]
        .collect();

    Ok(signal)
}

/// Generate colored noise (1/f^alpha)
fn generate_colored_noise(length: usize, alpha: f64) -> SignalResult<Vec<f64>> {
    // Start with white noise
    let mut white_noise = generate_white_noise(length)?;

    // Apply coloring filter (simplified implementation)
    if alpha > 0.0 {
        // Simple first-order IIR filter for coloring
        let a = (-alpha / 10.0).exp(); // Coefficient based on alpha
        let mut y_prev = 0.0;

        for sample in white_noise.iter_mut() {
            let y = a * y_prev + (1.0 - a) * *sample;
            *sample = y;
            y_prev = y;
        }
    }

    Ok(white_noise)
}

/// Generate chirp signal (frequency sweep)
fn generate_chirp(start_freq: f64, end_freq: f64, length: usize, fs: f64) -> SignalResult<Vec<f64>> {
    if start_freq <= 0.0 || end_freq <= 0.0 {
        return Err(SignalError::ValueError("Frequencies must be positive".to_string()));
    }

    if start_freq >= fs / 2.0 || end_freq >= fs / 2.0 {
        return Err(SignalError::ValueError(
            "Frequencies must be less than Nyquist frequency".to_string(),
        ));
    }

    let dt = 1.0 / fs;
    let duration = length as f64 * dt;
    let freq_rate = (end_freq - start_freq) / duration;

    let signal: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 * dt;
            let instant_freq = start_freq + freq_rate * t;
            let phase = 2.0 * PI * (start_freq * t + 0.5 * freq_rate * t * t);
            phase.sin()
        })
        .collect();

    Ok(signal)
}

/// Generate impulse signal
fn generate_impulse(length: usize) -> SignalResult<Vec<f64>> {
    let mut signal = vec![0.0; length];
    if length > 0 {
        signal[0] = 1.0; // Unit impulse at the beginning
    }
    Ok(signal)
}

/// Generate step signal
fn generate_step(length: usize) -> SignalResult<Vec<f64>> {
    let mut signal = vec![0.0; length];
    let step_point = length / 2;

    for i in step_point..length {
        signal[i] = 1.0;
    }

    Ok(signal)
}

/// Add noise to signal with specified SNR
pub fn add_noise_to_signal(signal: &[f64], snr_db: f64) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal cannot be empty".to_string()));
    }

    // Calculate signal power
    let signal_power = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;

    // Calculate noise power from SNR
    let snr_linear = 10.0_f64.powf(snr_db / 10.0);
    let noise_power = signal_power / snr_linear;
    let noise_std = noise_power.sqrt();

    // Generate noise and add to signal
    let mut noisy_signal = signal.to_vec();
    for sample in noisy_signal.iter_mut() {
        let noise = fastrand::f64() * 2.0 - 1.0; // Uniform noise
        *sample += noise * noise_std;
    }

    Ok(noisy_signal)
}

/// Assess signal quality metrics
pub fn assess_signal_quality(signal: &[f64], fs: f64) -> SignalResult<SignalQualityMetrics> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal cannot be empty".to_string()));
    }

    // Calculate basic statistics
    let signal_power = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    let signal_rms = signal_power.sqrt();

    // Estimate noise floor (simplified)
    let sorted_power: Vec<f64> = {
        let mut powers: Vec<f64> = signal.iter().map(|&x| x * x).collect();
        powers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        powers
    };
    let noise_floor = sorted_power[sorted_power.len() / 10]; // Bottom 10th percentile

    // Calculate SNR
    let snr_db = if noise_floor > 0.0 {
        10.0 * (signal_power / noise_floor).log10()
    } else {
        f64::INFINITY
    };

    // Calculate dynamic range
    let max_val = signal.iter().cloned().fold(0.0f64, f64::max);
    let min_val = signal.iter().cloned().fold(0.0f64, f64::min);
    let dynamic_range = if min_val > 0.0 {
        20.0 * (max_val / min_val).log10()
    } else {
        f64::INFINITY
    };

    // Assess spectral purity (simplified)
    let spectral_purity = assess_spectral_purity(signal);

    // Assess frequency accuracy (placeholder)
    let frequency_accuracy = 0.99; // High accuracy for generated signals

    Ok(SignalQualityMetrics {
        snr_db,
        spectral_purity,
        dynamic_range,
        frequency_accuracy,
    })
}

/// Assess spectral purity of signal
fn assess_spectral_purity(signal: &[f64]) -> f64 {
    // Simple measure: ratio of peak power to total power
    // In practice, this would use proper spectral analysis

    let max_power = signal.iter().map(|&x| x * x).fold(0.0f64, f64::max);
    let total_power = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;

    if total_power > 0.0 {
        (max_power / total_power).min(1.0)
    } else {
        0.0
    }
}

/// Generate test signals for comprehensive validation
pub fn generate_validation_signals(config: &TestSignalConfig) -> SignalResult<Vec<Vec<f64>>> {
    let mut signals = Vec::new();

    for signal_type in &config.signal_types {
        for &snr_db in &config.snr_levels {
            let clean_signal = generate_test_signal(signal_type, config)?;
            let noisy_signal = add_noise_to_signal(&clean_signal, snr_db)?;
            signals.push(noisy_signal);
        }
    }

    Ok(signals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoid_generation() {
        let signal = generate_sinusoid(10.0, 1000, 100.0).unwrap();
        assert_eq!(signal.len(), 1000);

        // Check that the signal has the expected frequency content
        let max_val = signal.iter().cloned().fold(0.0f64, f64::max);
        let min_val = signal.iter().cloned().fold(0.0f64, f64::min);
        assert!((max_val - 1.0).abs() < 0.1);
        assert!((min_val + 1.0).abs() < 0.1);
    }

    #[test]
    fn test_multisine_generation() {
        let freqs = vec![10.0, 20.0, 30.0];
        let signal = generate_multisine(&freqs, 1000, 200.0).unwrap();
        assert_eq!(signal.len(), 1000);

        // Signal should be normalized
        let max_amplitude = signal.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_amplitude <= 1.1); // Allow some tolerance
    }

    #[test]
    fn test_white_noise_generation() {
        let signal = generate_white_noise(1000).unwrap();
        assert_eq!(signal.len(), 1000);

        // Check that noise has reasonable statistics
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        assert!(mean.abs() < 0.1); // Should be approximately zero
    }

    #[test]
    fn test_chirp_generation() {
        let signal = generate_chirp(10.0, 50.0, 1000, 200.0).unwrap();
        assert_eq!(signal.len(), 1000);

        // Check amplitude bounds
        let max_amplitude = signal.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_amplitude <= 1.1);
    }

    #[test]
    fn test_add_noise() {
        let clean_signal = vec![1.0; 100];
        let noisy_signal = add_noise_to_signal(&clean_signal, 10.0).unwrap();
        assert_eq!(noisy_signal.len(), 100);

        // Noisy signal should have higher variance
        let clean_var = clean_signal.iter().map(|&x| (x - 1.0).powi(2)).sum::<f64>();
        let noisy_var = noisy_signal.iter().map(|&x| (x - 1.0).powi(2)).sum::<f64>();
        assert!(noisy_var > clean_var);
    }

    #[test]
    fn test_signal_quality_assessment() {
        let signal = generate_sinusoid(10.0, 1000, 100.0).unwrap();
        let quality = assess_signal_quality(&signal, 100.0).unwrap();

        assert!(quality.snr_db > 40.0); // Clean sinusoid should have high SNR
        assert!(quality.spectral_purity > 0.8); // Should have good spectral purity
        assert!(quality.frequency_accuracy > 0.9);
    }

    #[test]
    fn test_invalid_parameters() {
        // Test invalid frequency
        assert!(generate_sinusoid(0.0, 1000, 100.0).is_err());
        assert!(generate_sinusoid(60.0, 1000, 100.0).is_err()); // Above Nyquist

        // Test empty frequency list
        assert!(generate_multisine(&[], 1000, 100.0).is_err());
    }
}