//! Edge case validation for Lomb-Scargle implementation
//!
//! This module tests various edge cases including sparse sampling,
//! extreme dynamic ranges, short time series, and high frequency resolution.

use super::types::*;
use super::utils::find_peaks;
use crate::error::SignalResult;
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

/// Validate edge cases
#[allow(dead_code)]
pub fn validate_edge_cases(
    config: &ScipyValidationConfig,
) -> SignalResult<EdgeCaseValidationResult> {
    let sparse_sampling = test_sparse_sampling(config)?;
    let extreme_dynamic_range = test_extreme_dynamic_range(config)?;
    let short_time_series = test_short_time_series(config)?;
    let high_freq_resolution = test_high_frequency_resolution(config)?;

    let stability_score = calculate_edge_case_stability_score(
        sparse_sampling,
        extreme_dynamic_range,
        short_time_series,
        high_freq_resolution,
    );

    Ok(EdgeCaseValidationResult {
        sparse_sampling,
        extreme_dynamic_range,
        short_time_series,
        high_freq_resolution,
        stability_score,
    })
}

/// Test sparse sampling edge case
#[allow(dead_code)]
pub fn test_sparse_sampling(config: &ScipyValidationConfig) -> SignalResult<bool> {
    // Test with very sparse sampling (10 points over long duration)
    let t: Vec<f64> = vec![0.0, 1.0, 2.5, 4.0, 6.2, 8.1, 10.0, 12.3, 15.0, 20.0];
    let signal: Vec<f64> = t
        .iter()
        .map(|&time| (2.0 * PI * 0.1 * time).sin())
        .collect();
    let freqs: Vec<f64> = Array1::linspace(0.01, 1.0, 50).to_vec();

    match lombscargle(
        &t,
        &signal,
        Some(&freqs),
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    ) {
        Ok((_, power)) => {
            // Check if results are reasonable
            let max_power = power.iter().cloned().fold(0.0, f64::max);
            Ok(max_power > 0.0 && max_power.is_finite())
        }
        Err(_) => Ok(false),
    }
}

/// Test extreme dynamic range
#[allow(dead_code)]
pub fn test_extreme_dynamic_range(config: &ScipyValidationConfig) -> SignalResult<bool> {
    let n = 100;
    let fs = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Signal with extreme dynamic range
    let mut signal: Vec<f64> = t
        .iter()
        .map(|&time| (2.0 * PI * 1.0 * time).sin())
        .collect();
    signal[50] += 1e6; // Add huge spike
    signal[51] += -1e6;

    let freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 50).to_vec();

    match lombscargle(
        &t,
        &signal,
        Some(&freqs),
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    ) {
        Ok((_, power)) => {
            // Check if algorithm remains stable
            Ok(power.iter().all(|&p: &f64| p.is_finite() && p >= 0.0))
        }
        Err(_) => Ok(false),
    }
}

/// Test short time series
#[allow(dead_code)]
pub fn test_short_time_series(config: &ScipyValidationConfig) -> SignalResult<bool> {
    // Test with minimum viable data (5 points)
    let t: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let signal: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0]; // Alternating signal
    let freqs: Vec<f64> = vec![1.0, 2.0, 5.0];

    match lombscargle(
        &t,
        &signal,
        Some(&freqs),
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    ) {
        Ok((_, power)) => Ok(power.iter().all(|&p: &f64| p.is_finite() && p >= 0.0)),
        Err(_) => Ok(false),
    }
}

/// Test high frequency resolution
#[allow(dead_code)]
pub fn test_high_frequency_resolution(config: &ScipyValidationConfig) -> SignalResult<bool> {
    let n = 1000;
    let fs = 100.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Two very close frequencies
    let f1 = 10.0;
    let f2 = 10.05; // Very close frequency
    let signal: Vec<f64> = t
        .iter()
        .map(|&time| (2.0 * PI * f1 * time).sin() + (2.0 * PI * f2 * time).sin())
        .collect();

    // High resolution frequency grid
    let freqs: Vec<f64> = Array1::linspace(9.0, 11.0, 1000).to_vec();

    match lombscargle(
        &t,
        &signal,
        Some(&freqs),
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    ) {
        Ok((_, power)) => {
            // Should be able to resolve close frequencies
            let peak_indices = find_peaks(&power, 0.5);
            Ok(peak_indices.len() >= 2) // Should find at least 2 peaks
        }
        Err(_) => Ok(false),
    }
}

/// Calculate edge case stability score
#[allow(dead_code)]
pub fn calculate_edge_case_stability_score(
    sparse: bool,
    dynamic_range: bool,
    short_series: bool,
    high_freq: bool,
) -> f64 {
    let scores = [sparse, dynamic_range, short_series, high_freq];
    let passed = scores.iter().filter(|&&s| s).count();
    passed as f64 / scores.len() as f64
}
