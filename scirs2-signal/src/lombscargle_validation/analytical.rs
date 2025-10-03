// Analytical validation functions for Lomb-Scargle periodogram
//
// This module provides comprehensive analytical validation functions for
// Lomb-Scargle implementations against known analytical cases.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig, WindowType};
use super::types::{ValidationResult, SingleTestResult};
use scirs2_core::numeric::Float;
use scirs2_core::random::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use std::f64::consts::PI;
use std::time::Instant;

/// Validate Lomb-Scargle implementation against known analytical cases
///
/// Enhanced version with comprehensive edge case testing and robustness validation
///
/// # Arguments
///
/// * `implementation` - Name of implementation to test
/// * `tolerance` - Tolerance for numerical comparison
///
/// # Returns
///
/// * Validation result with detailed metrics
#[allow(dead_code)]
pub fn validate_analytical_cases(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<ValidationResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();
    let mut peak_errors = Vec::new();

    // Test case 1: Pure sinusoid (should have exact peak at frequency)
    let test_result_1 = validate_pure_sinusoid(implementation, tolerance)?;
    errors.extend(test_result_1.errors);
    peak_errors.push(test_result_1.peak_error);
    issues.extend(test_result_1.issues);

    // Test case 2: Multiple sinusoids with different amplitudes
    let test_result_2 = validate_multiple_sinusoids(implementation, tolerance)?;
    errors.extend(test_result_2.errors);
    peak_errors.extend(test_result_2.peak_errors);
    issues.extend(test_result_2.issues);

    // Test case 3: Heavily uneven sampling
    let test_result_3 = validate_uneven_sampling(implementation, tolerance)?;
    errors.extend(test_result_3.errors);
    peak_errors.push(test_result_3.peak_error);
    issues.extend(test_result_3.issues);

    // Test case 4: Extreme edge cases
    let test_result_4 = validate_edge_cases(implementation, tolerance)?;
    errors.extend(test_result_4.errors);
    issues.extend(test_result_4.issues);

    // Test case 5: Numerical precision and stability
    let test_result_5 = validate_numerical_stability(implementation, tolerance)?;
    errors.extend(test_result_5.errors);
    issues.extend(test_result_5.issues);

    // Test case 6: Very sparse sampling
    let test_result_6 = validate_sparse_sampling(implementation, tolerance)?;
    errors.extend(test_result_6.errors);
    peak_errors.push(test_result_6.peak_error);
    issues.extend(test_result_6.issues);

    // Test case 7: High dynamic range signals
    let test_result_7 = validate_dynamic_range(implementation, tolerance)?;
    errors.extend(test_result_7.errors);
    peak_errors.push(test_result_7.peak_error);
    issues.extend(test_result_7.issues);

    // Test case 8: Time series with trends
    let test_result_8 = validate_with_trends(implementation, tolerance)?;
    errors.extend(test_result_8.errors);
    peak_errors.push(test_result_8.peak_error);
    issues.extend(test_result_8.issues);

    // Test case 9: Correlated noise
    let test_result_9 = validate_correlated_noise(implementation, tolerance)?;
    errors.extend(test_result_9.errors);
    peak_errors.push(test_result_9.peak_error);
    issues.extend(test_result_9.issues);

    // Test case 10: Advanced-high frequency resolution
    let test_result_10 = validate_high_frequency_resolution(implementation, tolerance)?;
    errors.extend(test_result_10.errors);
    peak_errors.push(test_result_10.peak_error);
    issues.extend(test_result_10.issues);

    // Test case 11: Enhanced precision validation
    let test_result_11 = validate_enhanced_precision(implementation, tolerance)?;
    errors.extend(test_result_11.errors);
    peak_errors.push(test_result_11.peak_error);
    issues.extend(test_result_11.issues);

    // Test case 12: Cross-validation with reference implementation
    let test_result_12 = validate_cross_reference_implementation(implementation, tolerance)?;
    errors.extend(test_result_12.errors);
    issues.extend(test_result_12.issues);

    // Calculate overall metrics
    let max_relative_error = errors.iter().cloned().fold(0.0, f64::max);
    let mean_relative_error = if !errors.is_empty() {
        errors.iter().sum::<f64>() / errors.len() as f64
    } else {
        0.0
    };

    let peak_freq_error = if !peak_errors.is_empty() {
        peak_errors.iter().cloned().fold(0.0, f64::max)
    } else {
        0.0
    };

    // Calculate stability score based on number of issues and errors
    let stability_score = calculate_stability_score(&issues, &errors);

    Ok(ValidationResult {
        max_relative_error,
        mean_relative_error,
        stability_score,
        peak_freq_error,
        issues,
    })
}

/// Test pure sinusoid case
#[allow(dead_code)]
fn validate_pure_sinusoid(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 1000;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Compute periodogram
    let (freqs, power) = match implementation {
        "standard" => lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p) = lombscargle_enhanced(&t, &signal, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Find peak
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let peak_freq = freqs[peak_idx];

    // Check peak frequency accuracy
    let freq_error = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_error);

    if freq_error > tolerance {
        issues.push(format!(
            "Pure sinusoid peak frequency error {:.2e} exceeds tolerance {:.2e}",
            freq_error, tolerance
        ));
    }

    // Check that peak is significantly above noise floor
    let noise_floor = power.iter().cloned().fold(f64::MAX, f64::min);
    let signal_to_noise = peak_power / noise_floor.max(1e-15);

    if signal_to_noise < 10.0 {
        issues.push(format!(
            "Poor signal-to-noise ratio: {:.2}",
            signal_to_noise
        ));
    }

    // Validate that all power values are non-negative and finite
    for (i, &p) in power.iter().enumerate() {
        if !p.is_finite() || p < 0.0 {
            issues.push(format!("Invalid power value at index {}: {}", i, p));
            break;
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: freq_error,
        peak_errors: vec![freq_error],
        issues,
    })
}

/// Test multiple sinusoids with different amplitudes
#[allow(dead_code)]
fn validate_multiple_sinusoids(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();
    let mut peak_errors = Vec::new();

    let n = 1000;
    let fs = 100.0;
    let f_signals = vec![5.0, 15.0, 25.0];
    let amplitudes = vec![1.0, 0.5, 0.8];

    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            f_signals
                .iter()
                .zip(amplitudes.iter())
                .map(|(&f, &a)| a * (2.0 * PI * f * ti).sin())
                .sum()
        })
        .collect();

    // Compute periodogram
    let (freqs, power) = match implementation {
        "standard" => lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(30.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p) = lombscargle_enhanced(&t, &signal, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Find peaks for each expected frequency
    for (signal_idx, &expected_freq) in f_signals.iter().enumerate() {
        let freq_tolerance = 0.5; // Allow 0.5 Hz tolerance for peak finding

        let peak_candidates: Vec<(usize, f64)> = freqs
            .iter()
            .enumerate()
            .filter(|(_, &f)| (f - expected_freq).abs() < freq_tolerance)
            .map(|(i, &f)| (i, power[i]))
            .collect();

        if peak_candidates.is_empty() {
            issues.push(format!(
                "No peak found near expected frequency {:.1} Hz",
                expected_freq
            ));
            peak_errors.push(1.0); // Maximum error
            continue;
        }

        // Find the highest peak in the candidate range
        let (peak_idx, peak_power) = peak_candidates
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freqs[*peak_idx];
        let freq_error = (peak_freq - expected_freq).abs() / expected_freq;
        peak_errors.push(freq_error);
        errors.push(freq_error);

        if freq_error > tolerance * 5.0 {
            // More lenient for multi-component signals
            issues.push(format!(
                "Signal {} peak frequency error {:.2e} exceeds tolerance",
                signal_idx, freq_error
            ));
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: peak_errors.iter().cloned().fold(0.0, f64::max),
        peak_errors,
        issues,
    })
}

/// Test heavily uneven sampling patterns
#[allow(dead_code)]
fn validate_uneven_sampling(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();
    let n_nominal = 1000;
    let fs_nominal = 100.0;
    let f_signal = 10.0;

    // Create heavily uneven sampling (random gaps and clustering)
    let mut rng = scirs2_core::random::rng();
    let mut t = Vec::new();
    let mut current_time = 0.0;

    while t.len() < n_nominal && current_time < 10.0 {
        // Random time intervals with large variations
        let interval = rng.gen_range(0.001..0.5); // Very uneven: 1ms to 500ms
        current_time += interval;
        t.push(current_time);
    }

    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Compute periodogram
    let (freqs, power) = match implementation {
        "standard" => lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 10.0, // Higher oversampling for uneven data
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p) = lombscargle_enhanced(&t, &signal, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Find peak
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let peak_freq = freqs[peak_idx];

    let freq_error = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_error);

    // More lenient tolerance for uneven sampling
    if freq_error > tolerance * 10.0 {
        issues.push(format!(
            "Uneven sampling peak frequency error {:.2e} exceeds tolerance",
            freq_error
        ));
    }

    // Check for spurious peaks (should be rare with good implementation)
    let threshold = peak_power * 0.1; // 10% of main peak
    let spurious_peaks = power
        .iter()
        .enumerate()
        .filter(|(i, &p)| *i != peak_idx && p > threshold)
        .count();

    if spurious_peaks > 5 {
        issues.push(format!(
            "Too many spurious peaks: {} above 10% threshold",
            spurious_peaks
        ));
    }

    Ok(SingleTestResult {
        errors,
        peak_error: freq_error,
        peak_errors: vec![freq_error],
        issues,
    })
}

/// Calculate stability score based on issues and errors
fn calculate_stability_score(issues: &[String], errors: &[f64]) -> f64 {
    let base_score = 1.0;
    let issue_penalty = issues.len() as f64 * 0.1;
    let error_penalty = errors.iter().map(|&e| e.min(0.5)).sum::<f64>() * 0.2;

    (base_score - issue_penalty - error_penalty)
        .max(0.0)
        .min(1.0)
}

/// Test sparse sampling patterns
#[allow(dead_code)]
fn validate_sparse_sampling(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Generate sparse sampling - only 10% of expected samples
    let n_total = 1000;
    let n_samples = 100;
    let fs = 100.0;
    let f_signal = 10.0;

    let mut rng = scirs2_core::random::rng();
    let mut indices: Vec<usize> = (0..n_total).collect();
    indices.shuffle(&mut rng);
    indices.truncate(n_samples);
    indices.sort();

    let t: Vec<f64> = indices.iter().map(|&i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    let result = match implementation {
        "standard" => lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        ),
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 10.0, // Higher oversampling for sparse data
                f_min: Some(5.0),
                f_max: Some(15.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    let peak_error = match result {
        Ok((freqs, power)) => {
            // Find peak frequency
            let (peak_idx, &peak_power) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let peak_freq = freqs[peak_idx];
            let freq_error = (peak_freq - f_signal).abs() / f_signal;

            // Should still detect the signal despite sparse sampling
            if peak_power < 0.1 {
                issues.push("Signal detection failed with sparse sampling".to_string());
            }

            freq_error
        }
        Err(_) => {
            issues.push("Sparse sampling caused computation failure".to_string());
            1.0
        }
    };

    Ok(SingleTestResult {
        errors,
        peak_error,
        peak_errors: vec![peak_error],
        issues,
    })
}

// Placeholder implementations for remaining functions
// TODO: Extract complete implementations from original file

/// Test extreme edge cases
#[allow(dead_code)]
fn validate_edge_cases(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_edge_cases: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test numerical precision and stability
#[allow(dead_code)]
fn validate_numerical_stability(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_numerical_stability: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test high dynamic range signals
#[allow(dead_code)]
fn validate_dynamic_range(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_dynamic_range: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test signals with trends
#[allow(dead_code)]
fn validate_with_trends(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_with_trends: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test correlated noise scenarios
#[allow(dead_code)]
fn validate_correlated_noise(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_correlated_noise: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test high frequency resolution
#[allow(dead_code)]
fn validate_high_frequency_resolution(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_high_frequency_resolution: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test enhanced precision validation
#[allow(dead_code)]
fn validate_enhanced_precision(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_enhanced_precision: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Test cross-reference implementation validation
#[allow(dead_code)]
fn validate_cross_reference_implementation(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    // TODO: Extract complete implementation from original file
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    issues.push("validate_cross_reference_implementation: Placeholder implementation - needs extraction".to_string());

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}
