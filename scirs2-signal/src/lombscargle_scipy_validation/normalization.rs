//! Normalization method validation
//!
//! This module validates different normalization methods used in Lomb-Scargle
//! periodogram computation by comparing results across methods.

use super::types::*;
use super::utils::*;
use crate::error::SignalResult;
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Validate different normalization methods
#[allow(dead_code)]
pub fn validate_normalization_methods(
    config: &ScipyValidationConfig,
) -> SignalResult<NormalizationValidationResult> {
    let normalizations = vec!["standard", "model", "log", "psd"];
    let mut method_results = HashMap::new();
    let mut best_score = 0.0;
    let mut best_method = "standard".to_string();

    for norm_method in &normalizations {
        let mut accuracy_result = AccuracyValidationResult {
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            rmse: 0.0,
            correlation: 0.0,
            passed_cases: 0,
            total_cases: 0,
        };

        // Test this normalization method
        let mut total_corr = 0.0;
        let mut valid_tests = 0;

        for &fs in &config.sampling_frequencies[..config.sampling_frequencies.len().min(2)] {
            // Limit for normalization testing
            for &n in &config.test_lengths[..config.test_lengths.len().min(3)] {
                for &test_freq in &config.test_frequencies[..config.test_frequencies.len().min(2)] {
                    if let Ok((_, _, _, corr)) =
                        validate_single_normalization_case(n, fs, test_freq, norm_method, config)
                    {
                        total_corr += corr;
                        valid_tests += 1;
                        accuracy_result.total_cases += 1;
                        if corr > 0.99 {
                            accuracy_result.passed_cases += 1;
                        }
                    }
                }
            }
        }

        if valid_tests > 0 {
            accuracy_result.correlation = total_corr / valid_tests as f64;
            let score = accuracy_result.correlation
                * (accuracy_result.passed_cases as f64 / accuracy_result.total_cases.max(1) as f64);

            if score > best_score {
                best_score = score;
                best_method = norm_method.to_string();
            }
        }

        method_results.insert(norm_method.to_string(), accuracy_result);
    }

    let consistency_score = calculate_normalization_consistency(&method_results);

    Ok(NormalizationValidationResult {
        method_results,
        best_method,
        consistency_score,
    })
}

/// Validate a single normalization method case
#[allow(dead_code)]
pub fn validate_single_normalization_case(
    n: usize,
    fs: f64,
    test_freq: f64,
    normalization: &str,
    _config: &ScipyValidationConfig,
) -> SignalResult<(f64, f64, f64, f64)> {
    // Generate test signal
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&time| (2.0 * PI * test_freq * time).sin())
        .collect();

    let freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 50).to_vec();

    // Test our implementation with specific normalization
    let (_, our_power) = lombscargle(
        &t,
        &signal,
        Some(&freqs),
        Some(normalization),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    )?;

    // For normalization validation, we compare with our own reference implementation
    let reference_power = super::accuracy::compute_reference_lombscargle(&t, &signal, &freqs)?;

    let (abs_err, rel_err, rmse) = calculate_error_metrics(&our_power, &reference_power)?;
    let correlation = calculate_correlation(&our_power, &reference_power)?;

    Ok((abs_err, rel_err, rmse, correlation))
}
