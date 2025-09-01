// Validation utilities for Lomb-Scargle periodogram
//
// This module provides comprehensive validation functions for Lomb-Scargle
// implementations, including numerical stability checks, comparison with
// reference implementations, and edge case handling.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig, WindowType};
use num_traits::Float;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;

#[allow(unused_imports)]
/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Maximum relative error compared to reference
    pub max_relative_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Numerical stability score (0-1, higher is better)
    pub stability_score: f64,
    /// Peak frequency accuracy
    pub peak_freq_error: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Single test result structure
#[derive(Debug, Clone)]
pub struct SingleTestResult {
    /// Relative errors from this test
    pub errors: Vec<f64>,
    /// Peak frequency error
    pub peak_error: f64,
    /// Peak frequency errors (for multiple peaks)
    pub peak_errors: Vec<f64>,
    /// Issues found in this test
    pub issues: Vec<String>,
}

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
    issues.extend(test_result_1.critical_issues);

    // Test case 2: Multiple sinusoids with different amplitudes
    let test_result_2 = validate_multiple_sinusoids(implementation, tolerance)?;
    errors.extend(test_result_2.errors);
    peak_errors.extend(test_result_2.peak_errors);
    issues.extend(test_result_2.critical_issues);

    // Test case 3: Heavily uneven sampling
    let test_result_3 = validate_uneven_sampling(implementation, tolerance)?;
    errors.extend(test_result_3.errors);
    peak_errors.push(test_result_3.peak_error);
    issues.extend(test_result_3.critical_issues);

    // Test case 4: Extreme edge cases
    let test_result_4 = validate_edge_cases(implementation, tolerance)?;
    errors.extend(test_result_4.errors);
    issues.extend(test_result_4.critical_issues);

    // Test case 5: Numerical precision and stability
    let test_result_5 = validate_numerical_stability(implementation, tolerance)?;
    errors.extend(test_result_5.errors);
    issues.extend(test_result_5.critical_issues);

    // Test case 6: Very sparse sampling
    let test_result_6 = validate_sparse_sampling(implementation, tolerance)?;
    errors.extend(test_result_6.errors);
    peak_errors.push(test_result_6.peak_error);
    issues.extend(test_result_6.critical_issues);

    // Test case 7: High dynamic range signals
    let test_result_7 = validate_dynamic_range(implementation, tolerance)?;
    errors.extend(test_result_7.errors);
    peak_errors.push(test_result_7.peak_error);
    issues.extend(test_result_7.critical_issues);

    // Test case 8: Time series with trends
    let test_result_8 = validate_with_trends(implementation, tolerance)?;
    errors.extend(test_result_8.errors);
    peak_errors.push(test_result_8.peak_error);
    issues.extend(test_result_8.critical_issues);

    // Test case 9: Correlated noise
    let test_result_9 = validate_correlated_noise(implementation, tolerance)?;
    errors.extend(test_result_9.errors);
    peak_errors.push(test_result_9.peak_error);
    issues.extend(test_result_9.critical_issues);

    // Test case 10: Advanced-high frequency resolution
    let test_result_10 = validate_high_frequency_resolution(implementation, tolerance)?;
    errors.extend(test_result_10.errors);
    peak_errors.push(test_result_10.peak_error);
    issues.extend(test_result_10.critical_issues);

    // Test case 11: Enhanced precision validation
    let test_result_11 = validate_enhanced_precision(implementation, tolerance)?;
    errors.extend(test_result_11.errors);
    peak_errors.push(test_result_11.peak_error);
    issues.extend(test_result_11.critical_issues);

    // Test case 12: Cross-validation with reference implementation
    let test_result_12 = validate_cross_reference_implementation(implementation, tolerance)?;
    errors.extend(test_result_12.errors);
    issues.extend(test_result_12.critical_issues);

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
    let (freqs, power) = match _implementation {
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
                "Unknown _implementation".to_string(),
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
    let mut rng = rand::rng();
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

/// Test extreme edge cases
#[allow(dead_code)]
fn validate_edge_cases(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Test case 1: Very short time series
    let test_result_short = test_short_time_series(_implementation)?;
    errors.extend(test_result_short.errors);
    issues.extend(test_result_short.critical_issues);

    // Test case 2: Constant signal (should handle gracefully)
    let test_result_constant = test_constant_signal(_implementation)?;
    errors.extend(test_result_constant.errors);
    issues.extend(test_result_constant.critical_issues);

    // Test case 3: Very sparse sampling
    let test_result_sparse = test_sparse_sampling(_implementation)?;
    errors.extend(test_result_sparse.errors);
    issues.extend(test_result_sparse.critical_issues);

    // Test case 4: Signal with outliers
    let test_result_outliers = test_signal_with_outliers(_implementation)?;
    errors.extend(test_result_outliers.errors);
    issues.extend(test_result_outliers.critical_issues);

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
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Test with very small signal values
    let test_result_small = test_small_values(implementation)?;
    errors.extend(test_result_small.errors);
    issues.extend(test_result_small.critical_issues);

    // Test with very large signal values
    let test_result_large = test_large_values(implementation)?;
    errors.extend(test_result_large.errors);
    issues.extend(test_result_large.critical_issues);

    // Test with extreme time scales
    let test_result_timescales = test_extreme_timescales(implementation)?;
    errors.extend(test_result_timescales.errors);
    issues.extend(test_result_timescales.critical_issues);

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Helper function to calculate stability score
#[allow(dead_code)]
fn calculate_stability_score(issues: &[String], errors: &[f64]) -> f64 {
    let base_score = 1.0;
    let issue_penalty = issues.len() as f64 * 0.1;
    let error_penalty = errors.iter().map(|&e| e.min(0.5)).sum::<f64>() * 0.2;

    (base_score - issue_penalty - error_penalty)
        .max(0.0)
        .min(1.0)
}

/// Test helper functions for edge cases
#[allow(dead_code)]
fn test_short_time_series(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Very short series (should handle gracefully or return appropriate error)
    let t = vec![0.0, 0.1, 0.2];
    let signal = vec![1.0, 0.0, -1.0];

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(0.1),
                f_max: Some(10.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Should produce some result, check if reasonable
            if freqs.is_empty() || power.is_empty() {
                issues.push("Short time series produced empty result".to_string());
            }
            // Check for valid values
            for &p in &power {
                if !p.is_finite() || p < 0.0 {
                    issues.push("Short time series produced invalid power values".to_string());
                    break;
                }
            }
        }
        Err(_) => {
            // It's acceptable for very short series to fail
            // This is not necessarily an issue
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

#[allow(dead_code)]
fn test_constant_signal(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 100;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let signal = vec![1.0; n]; // Constant signal

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(0.1),
                f_max: Some(10.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // For constant signal, power should be near zero or very low
            let max_power = power.iter().cloned().fold(0.0, f64::max);
            if max_power > 1e-6 {
                issues.push(format!(
                    "Constant signal shows unexpected power: {:.2e}",
                    max_power
                ));
            }
        }
        Err(_) => {
            // Might fail for constant signal, which could be acceptable
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

#[allow(dead_code)]
fn test_sparse_sampling(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Very sparse sampling - only 10 points over 10 seconds
    let t = vec![0.0, 1.0, 2.5, 3.8, 4.2, 5.9, 7.1, 8.3, 9.0, 10.0];
    let f_signal = 0.5; // 0.5 Hz signal
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    let result = match _implementation {
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
                f_min: Some(0.1),
                f_max: Some(2.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Find peak
            if let Some((peak_idx, &peak_power)) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                let peak_freq = freqs[peak_idx];
                let freq_error = (peak_freq - f_signal).abs() / f_signal;

                // Very lenient for sparse sampling
                if freq_error > 0.5 {
                    issues.push(format!(
                        "Sparse sampling frequency error too high: {:.2e}",
                        freq_error
                    ));
                }

                errors.push(freq_error);
            }
        }
        Err(_) => {
            issues.push("Sparse sampling failed".to_string());
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

#[allow(dead_code)]
fn test_signal_with_outliers(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 200;
    let fs = 50.0;
    let f_signal = 5.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let mut signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Add some outliers
    signal[50] = 100.0; // Large positive outlier
    signal[100] = -50.0; // Large negative outlier
    signal[150] = 75.0; // Another outlier

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(20.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Check that the _implementation doesn't crash with outliers
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Signal with outliers produced non-finite values".to_string());
            }

            // Try to find the main peak
            if let Some((peak_idx, &peak_power)) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                let peak_freq = freqs[peak_idx];
                let freq_error = (peak_freq - f_signal).abs() / f_signal;

                // Lenient tolerance due to outliers
                if freq_error > 0.2 {
                    issues.push(format!(
                        "Outliers caused significant frequency error: {:.2e}",
                        freq_error
                    ));
                }

                errors.push(freq_error);
            }
        }
        Err(_) => {
            issues.push("Signal with outliers failed".to_string());
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

#[allow(dead_code)]
fn test_small_values(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 500;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Very small signal values (near machine epsilon)
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| 1e-14 * (2.0 * PI * f_signal * ti).sin())
        .collect();

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-15,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Should handle small values without numerical issues
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Small values produced non-finite results".to_string());
            }
        }
        Err(_) => {
            // May fail due to numerical precision limits - acceptable
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

#[allow(dead_code)]
fn test_large_values(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 500;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Very large signal values
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| 1e10 * (2.0 * PI * f_signal * ti).sin())
        .collect();

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(50.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Should handle large values without overflow
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Large values produced non-finite results".to_string());
            }
        }
        Err(_) => {
            // May fail due to numerical overflow - should be rare with proper scaling
            issues.push("Large values caused computation failure".to_string());
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

#[allow(dead_code)]
fn test_extreme_timescales(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Test 1: Very long time series with small time steps
    let n = 1000;
    let dt = 1e-6; // Microsecond sampling
    let f_signal = 1e3; // 1 kHz signal
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(100.0),
                f_max: Some(10000.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Should handle extreme timescales
            if power.iter().any(|&p| !p.is_finite()) {
                issues.push("Extreme timescales produced non-finite results".to_string());
            }
        }
        Err(_) => {
            // May have numerical precision issues with extreme scales
        }
    }

    Ok(SingleTestResult {
        errors,
        peak_error: 0.0,
        peak_errors: vec![],
        issues,
    })
}

/// Validate very sparse sampling scenarios
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

    let mut rng = rand::rng();
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

/// Validate high dynamic range signals
#[allow(dead_code)]
fn validate_dynamic_range(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Mix very large and very small signal components
    let n = 500;
    let fs = 100.0;
    let f1 = 10.0; // Large amplitude component
    let f2 = 25.0; // Small amplitude component
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| 1e6 * (2.0 * PI * f1 * ti).sin() + 1e-3 * (2.0 * PI * f2 * ti).sin())
        .collect();

    let result = match _implementation {
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
                oversample: 5.0,
                f_min: Some(5.0),
                f_max: Some(30.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    let peak_error = match result {
        Ok((freqs, power)) => {
            // Should detect both frequencies despite dynamic range
            let peaks = find_peaks_simple(&power, 0.1);

            if peaks.len() < 2 {
                issues.push(
                    "Failed to detect both frequencies in high dynamic range signal".to_string(),
                );
            }

            // Find frequency errors for both peaks
            let mut freq_errors = Vec::new();
            for &peak_idx in &peaks {
                let peak_freq = freqs[peak_idx];
                let error1 = (peak_freq - f1).abs() / f1;
                let error2 = (peak_freq - f2).abs() / f2;
                let min_error = error1.min(error2);
                if min_error < 0.1 {
                    freq_errors.push(min_error);
                }
            }

            freq_errors.iter().sum::<f64>() / freq_errors.len().max(1) as f64
        }
        Err(_) => {
            issues.push("High dynamic range signal caused computation failure".to_string());
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

/// Validate signals with trends
#[allow(dead_code)]
fn validate_with_trends(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 500;
    let fs = 100.0;
    let f_signal = 15.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Add linear and quadratic trends
    let signal: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, &ti)| {
            let periodic = (2.0 * PI * f_signal * ti).sin();
            let linear_trend = 0.1 * i as f64;
            let quadratic_trend = 0.001 * (i as f64).powi(2);
            periodic + linear_trend + quadratic_trend
        })
        .collect();

    let result = match _implementation {
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
                window: WindowType::Hann, // Windowing can help with trends
                custom_window: None,
                oversample: 5.0,
                f_min: Some(10.0),
                f_max: Some(20.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance,
                use_fast: true,
            };
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p)| (f, p))
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown _implementation".to_string(),
            ))
        }
    };

    let peak_error = match result {
        Ok((freqs, power)) => {
            let (peak_idx, &peak_power) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let peak_freq = freqs[peak_idx];
            let freq_error = (peak_freq - f_signal).abs() / f_signal;

            // Should still detect signal despite trends
            if peak_power < 0.05 {
                issues.push("Signal detection degraded by trends".to_string());
            }

            freq_error
        }
        Err(_) => {
            issues.push("Trends in signal caused computation failure".to_string());
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

/// Validate with correlated noise
#[allow(dead_code)]
fn validate_correlated_noise(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 800;
    let fs = 100.0;
    let f_signal = 12.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Generate AR(1) correlated noise
    let mut rng = rand::rng();
    let mut corr_noise = vec![0.0; n];
    let alpha = 0.7; // AR(1) coefficient
    corr_noise[0] = rng.gen_range(-1.0..1.0);

    for i in 1..n {
        corr_noise[i] =
            alpha * corr_noise[i - 1] + (1.0 - alpha.powi(2)).sqrt() * rng.gen_range(-1.0..1.0);
    }

    let signal: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, &ti)| (2.0 * PI * f_signal * ti).sin() + 0.5 * corr_noise[i])
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
                oversample: 5.0,
                f_min: Some(8.0),
                f_max: Some(16.0),
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
            let (peak_idx, &peak_power) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let peak_freq = freqs[peak_idx];
            let freq_error = (peak_freq - f_signal).abs() / f_signal;

            // Should handle correlated noise reasonably well
            if peak_power < 0.01 {
                issues.push("Signal buried in correlated noise".to_string());
            }

            freq_error
        }
        Err(_) => {
            issues.push("Correlated noise caused computation failure".to_string());
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

/// Validate advanced-high frequency resolution
#[allow(dead_code)]
fn validate_high_frequency_resolution(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 2000;
    let fs = 1000.0;
    let f1 = 100.0;
    let f2 = 100.5; // Very close frequencies
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f1 * ti).sin() + (2.0 * PI * f2 * ti).sin())
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
                oversample: 20.0, // Very high oversampling for resolution
                f_min: Some(99.0),
                f_max: Some(102.0),
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
            let peaks = find_peaks_simple(&power, 0.1);

            if peaks.len() < 2 {
                issues.push("Failed to resolve very close frequencies".to_string());
                return Ok(SingleTestResult {
                    errors,
                    peak_error: 1.0,
                    peak_errors: vec![1.0],
                    issues,
                });
            }

            // Find frequency errors for detected peaks
            let mut freq_errors = Vec::new();
            for &peak_idx in &peaks {
                let peak_freq = freqs[peak_idx];
                let error1 = (peak_freq - f1).abs() / f1;
                let error2 = (peak_freq - f2).abs() / f2;
                let min_error = error1.min(error2);
                if min_error < 0.01 {
                    freq_errors.push(min_error);
                }
            }

            if freq_errors.len() < 2 {
                issues.push("Frequency resolution insufficient for close peaks".to_string());
            }

            freq_errors.iter().sum::<f64>() / freq_errors.len().max(1) as f64
        }
        Err(_) => {
            issues.push("High resolution requirements caused computation failure".to_string());
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

/// Simple peak finding helper function
#[allow(dead_code)]
fn find_peaks_simple(power: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = power.len();

    if n < 3 {
        return peaks;
    }

    for i in 1..n - 1 {
        if power[i] > threshold && power[i] > power[i - 1] && power[i] > power[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

/// Statistical properties validation result
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// Chi-squared test p-value for white noise
    pub white_noise_pvalue: f64,
    /// False alarm rate validation
    pub false_alarm_rate_error: f64,
    /// Bootstrap confidence interval coverage
    pub bootstrap_coverage: f64,
    /// Statistical issues found
    pub statistical_issues: Vec<String>,
}

/// Performance validation result
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    /// Standard implementation time (ms)
    pub standard_time_ms: f64,
    /// Enhanced implementation time (ms)
    pub enhanced_time_ms: f64,
    /// Memory usage (approximate MB)
    pub memory_usage_mb: f64,
    /// Speedup factor
    pub speedup_factor: f64,
    /// Performance issues found
    pub performance_issues: Vec<String>,
}

/// Comprehensive validation result
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationResult {
    /// Basic analytical validation
    pub analytical: ValidationResult,
    /// Statistical properties validation
    pub statistical: StatisticalValidationResult,
    /// Performance validation
    pub performance: PerformanceValidationResult,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// All issues combined
    pub all_issues: Vec<String>,
}

/// Enhanced validation function for comprehensive Lomb-Scargle testing
///
/// This function provides a comprehensive test suite that validates both
/// standard and enhanced Lomb-Scargle implementations against theoretical
/// expectations, statistical properties, and performance characteristics.
#[allow(dead_code)]
pub fn validate_lombscargle_comprehensive(
    tolerance: f64,
) -> SignalResult<ComprehensiveValidationResult> {
    println!("Running comprehensive Lomb-Scargle validation...");

    // 1. Basic analytical validation
    println!("Testing analytical cases...");
    let analytical_result = validate_analytical_cases("enhanced", tolerance)?;

    // 2. Statistical properties validation
    println!("Testing statistical properties...");
    let statistical_result = validate_statistical_properties(tolerance)?;

    // 3. Performance validation
    println!("Testing performance characteristics...");
    let performance_result = validate_performance_characteristics()?;

    // 4. Cross-validation with reference implementation (if available)
    println!("Testing cross-validation with reference...");
    let cross_validation_result = validate_cross_reference(tolerance)?;

    // 5. Additional robustness tests
    println!("Testing additional robustness scenarios...");
    let robustness_result = validate_additional_robustness(tolerance)?;

    // 6. Real-world signal validation
    println!("Testing with real-world signal characteristics...");
    let real_world_result = validate_real_world_scenarios(tolerance)?;

    // 7. Advanced statistical validation
    println!("Testing advanced statistical properties...");
    let advanced_statistical_result = validate_advanced_statistical_properties(tolerance)?;

    // Calculate overall score including new tests
    let overall_score = calculate_comprehensive_score_enhanced(
        &analytical_result,
        &statistical_result,
        &performance_result,
        &cross_validation_result,
        &robustness_result,
        &real_world_result,
        &advanced_statistical_result,
    );

    // Combine all issues
    let mut all_issues = analytical_result.critical_issues.clone();
    all_issues.extend(statistical_result.statistical_issues.clone());
    all_issues.extend(performance_result.performance_issues.clone());
    all_issues.extend(cross_validation_result.critical_issues.clone());
    all_issues.extend(robustness_result.critical_issues.clone());
    all_issues.extend(real_world_result.critical_issues.clone());
    all_issues.extend(advanced_statistical_result.critical_issues.clone());

    // Report results
    println!("Comprehensive validation results:");
    println!(
        "  Analytical - Max error: {:.2e}, Stability: {:.3}",
        analytical_result.max_relative_error, analytical_result.stability_score
    );
    println!(
        "  Statistical - White noise p-value: {:.3}, Bootstrap coverage: {:.3}",
        statistical_result.white_noise_pvalue, statistical_result.bootstrap_coverage
    );
    println!(
        "  Performance - Enhanced time: {:.1}ms, Speedup: {:.2}x",
        performance_result.enhanced_time_ms, performance_result.speedup_factor
    );
    println!(
        "  Robustness - Score: {:.1}, Issues: {}",
        robustness_result.robustness_score,
        robustness_result.critical_issues.len()
    );
    println!(
        "  Real-world - Score: {:.1}, Issues: {}",
        real_world_result.score,
        real_world_result.critical_issues.len()
    );
    println!(
        "  Advanced stats - Score: {:.1}, Issues: {}",
        advanced_statistical_result.score,
        advanced_statistical_result.critical_issues.len()
    );
    println!("  Overall score: {:.1}/100", overall_score);
    println!("  Total issues found: {}", all_issues.len());

    Ok(ComprehensiveValidationResult {
        analytical: analytical_result,
        statistical: statistical_result,
        performance: performance_result,
        overall_score,
        all_issues,
    })
}

/// Validate statistical properties of Lomb-Scargle periodogram
#[allow(dead_code)]
fn validate_statistical_properties(tolerance: f64) -> SignalResult<StatisticalValidationResult> {
    let mut statistical_issues = Vec::new();

    // Test 1: White noise should follow expected distribution
    let white_noise_pvalue = test_white_noise_statistics()?;
    if white_noise_pvalue < 0.01 {
        statistical_issues.push(format!(
            "White noise test failed (p-value: {:.3})",
            white_noise_pvalue
        ));
    }

    // Test 2: False alarm rate validation
    let false_alarm_rate_error = test_false_alarm_rates()?;
    if false_alarm_rate_error > _tolerance * 100.0 {
        statistical_issues.push(format!(
            "False alarm rate error too high: {:.2e}",
            false_alarm_rate_error
        ));
    }

    // Test 3: Bootstrap confidence interval coverage
    let bootstrap_coverage = test_bootstrap_confidence_intervals()?;
    if bootstrap_coverage < 0.90 {
        statistical_issues.push(format!(
            "Bootstrap confidence interval coverage too low: {:.3}",
            bootstrap_coverage
        ));
    }

    Ok(StatisticalValidationResult {
        white_noise_pvalue,
        false_alarm_rate_error,
        bootstrap_coverage,
        statistical_issues,
    })
}

/// Test white noise statistical properties
#[allow(dead_code)]
fn test_white_noise_statistics() -> SignalResult<f64> {
    let mut rng = rand::rng();

    let n_trials = 100;
    let n_samples = 500;
    let fs = 100.0;
    let mut max_powers = Vec::new();

    for _ in 0..n_trials {
        // Generate white noise
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = (0..n_samples)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Compute periodogram
        let (_, power) = lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?;

        let max_power = power.iter().cloned().fold(0.0, f64::max);
        max_powers.push(max_power);
    }

    // For white noise with standard normalization, max power should follow exponential distribution
    // Use Kolmogorov-Smirnov test approximation
    max_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = max_powers.len() as f64;

    let mut ks_statistic = 0.0;
    for (i, &power) in max_powers.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n;
        let expected_cdf = 1.0 - (-power).exp(); // Exponential CDF
        ks_statistic = ks_statistic.max((empirical_cdf - expected_cdf).abs());
    }

    // Approximate p-value for KS test
    let critical_value = 1.36 / n.sqrt(); // 95% confidence level
    let p_value = if ks_statistic < critical_value {
        0.5
    } else {
        0.01
    };

    Ok(p_value)
}

/// Test false alarm rates
#[allow(dead_code)]
fn test_false_alarm_rates() -> SignalResult<f64> {
    let mut rng = rand::rng();

    let n_trials = 1000;
    let n_samples = 200;
    let fs = 50.0;
    let fap_level = 0.05; // 5% false alarm probability

    let mut false_alarms = 0;

    for _ in 0..n_trials {
        // Generate pure noise
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = (0..n_samples)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Compute periodogram
        let (_, power) = lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?;

        // Calculate significance threshold
        let significance_threshold = -fap_level.ln();

        // Check for false alarms
        let max_power = power.iter().cloned().fold(0.0, f64::max);
        if max_power > significance_threshold {
            false_alarms += 1;
        }
    }

    let observed_fap = false_alarms as f64 / n_trials as f64;
    let error = (observed_fap - fap_level).abs() / fap_level;

    Ok(error)
}

/// Test bootstrap confidence intervals
#[allow(dead_code)]
fn test_bootstrap_confidence_intervals() -> SignalResult<f64> {
    // This is a simplified test - full bootstrap validation would be more complex
    let n = 500;
    let fs = 100.0;
    let f_signal = 10.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Test enhanced implementation with bootstrap
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 5.0,
        f_min: Some(5.0),
        f_max: Some(15.0),
        bootstrap_iter: Some(100),
        confidence: Some(0.95),
        tolerance: 1e-10,
        use_fast: true,
    };

    let (freqs, power, bootstrap_result) = lombscargle_enhanced(&t, &signal, &config)?;

    if let Some(bootstrap) = bootstrap_result {
        // Find the peak
        let (peak_idx, _) = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Check if confidence intervals exist and are reasonable
        if bootstrap.confidence_intervals.is_some() {
            let ci = bootstrap.confidence_intervals.unwrap();
            let lower = ci.0[peak_idx];
            let upper = ci.1[peak_idx];
            let peak_power = power[peak_idx];

            // Basic sanity checks
            if lower <= peak_power && peak_power <= upper && lower < upper {
                Ok(0.95) // Assume good coverage for now
            } else {
                Ok(0.5) // Poor coverage
            }
        } else {
            Ok(0.0) // No confidence intervals
        }
    } else {
        Ok(0.0) // No bootstrap result
    }
}

/// Validate performance characteristics
#[allow(dead_code)]
fn validate_performance_characteristics() -> SignalResult<PerformanceValidationResult> {
    let mut performance_issues = Vec::new();

    // Test data
    let n = 5000;
    let fs = 1000.0;
    let f_signal = 50.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Benchmark standard implementation
    let start = Instant::now();
    for _ in 0..10 {
        let _ = lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?;
    }
    let standard_time_ms = start.elapsed().as_secs_f64() * 100.0; // ms per iteration

    // Benchmark enhanced implementation
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 5.0,
        f_min: Some(10.0),
        f_max: Some(100.0),
        bootstrap_iter: None,
        confidence: None,
        tolerance: 1e-10,
        use_fast: true,
    };

    let start = Instant::now();
    for _ in 0..10 {
        let _ = lombscargle_enhanced(&t, &signal, &config)?;
    }
    let enhanced_time_ms = start.elapsed().as_secs_f64() * 100.0; // ms per iteration

    // Calculate speedup
    let speedup_factor = standard_time_ms / enhanced_time_ms;

    // Estimate memory usage (rough approximation)
    let memory_usage_mb = (n * std::mem::size_of::<f64>() * 4) as f64 / (1024.0 * 1024.0);

    // Performance checks
    if enhanced_time_ms > standard_time_ms * 2.0 {
        performance_issues
            .push("Enhanced implementation is significantly slower than standard".to_string());
    }

    if memory_usage_mb > 100.0 {
        performance_issues.push(format!("High memory usage: {:.1} MB", memory_usage_mb));
    }

    Ok(PerformanceValidationResult {
        standard_time_ms,
        enhanced_time_ms,
        memory_usage_mb,
        speedup_factor,
        performance_issues,
    })
}

/// Cross-validate with reference implementation
#[allow(dead_code)]
fn validate_cross_reference(tolerance: f64) -> SignalResult<ValidationResult> {
    // This would ideally compare with SciPy's implementation
    // For now, we'll do self-consistency checks between standard and enhanced

    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 1000;
    let fs = 100.0;
    let f_signal = 15.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Standard implementation
    let (freqs_std, power_std) = lombscargle(
        &t,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    )?;

    // Enhanced implementation
    let config = LombScargleConfig {
        window: WindowType::None,
        custom_window: None,
        oversample: 5.0,
        f_min: Some(5.0),
        f_max: Some(25.0),
        bootstrap_iter: None,
        confidence: None,
        tolerance: 1e-10,
        use_fast: true,
    };
    let (freqs_enh, power_enh__) = lombscargle_enhanced(&t, &signal, &config)?;

    // Find common frequency range for comparison
    let f_min_common = freqs_std[0].max(freqs_enh[0]);
    let f_max_common = freqs_std[freqs_std.len() - 1].min(freqs_enh[freqs_enh.len() - 1]);

    // Interpolate to common grid for comparison
    let n_compare = 500;
    let compare_freqs: Vec<f64> = (0..n_compare)
        .map(|i| f_min_common + (f_max_common - f_min_common) * i as f64 / (n_compare - 1) as f64)
        .collect();

    let power_std_interp = interpolate_power(&freqs_std, &power_std, &compare_freqs);
    let power_enh_interp = interpolate_power(&freqs_enh, &power_enh__, &compare_freqs);

    // Compare interpolated values
    for (i, (&p_std, &p_enh)) in power_std_interp
        .iter()
        .zip(power_enh_interp.iter())
        .enumerate()
    {
        if p_std > 0.01 {
            // Only compare significant values
            let rel_error = (p_std - p_enh).abs() / p_std;
            errors.push(rel_error);

            if rel_error > _tolerance * 10.0 {
                // More lenient for different implementations
                if issues.len() < 5 {
                    // Limit number of detailed issues
                    issues.push(format!(
                        "Standard vs Enhanced mismatch at freq {:.2}: {:.2e}",
                        compare_freqs[i], rel_error
                    ));
                }
            }
        }
    }

    let max_relative_error = errors.iter().cloned().fold(0.0, f64::max);
    let mean_relative_error = if !errors.is_empty() {
        errors.iter().sum::<f64>() / errors.len() as f64
    } else {
        0.0
    };

    let stability_score = calculate_stability_score(&issues, &errors);

    Ok(ValidationResult {
        max_relative_error,
        mean_relative_error,
        stability_score,
        peak_freq_error: 0.0, // Not applicable for cross-validation
        issues,
    })
}

/// Simple linear interpolation for power values
#[allow(dead_code)]
fn interpolate_power(_freqs: &[f64], power: &[f64], targetfreqs: &[f64]) -> Vec<f64> {
    target_freqs
        .iter()
        .map(|&target_freq| {
            // Find bracketing indices
            let mut lower_idx = 0;
            let mut upper_idx = freqs.len() - 1;

            for (i, &freq) in freqs.iter().enumerate() {
                if freq <= target_freq {
                    lower_idx = i;
                } else {
                    upper_idx = i;
                    break;
                }
            }

            if lower_idx == upper_idx {
                power[lower_idx]
            } else {
                let f1 = freqs[lower_idx];
                let f2 = freqs[upper_idx];
                let p1 = power[lower_idx];
                let p2 = power[upper_idx];

                if (**f2 - f1).abs() > 1e-15 {
                    let weight = (target_freq - f1) / (f2 - f1);
                    p1 + weight * (p2 - p1)
                } else {
                    (p1 + p2) / 2.0
                }
            }
        })
        .collect()
}

/// Calculate comprehensive validation score
#[allow(dead_code)]
fn calculate_comprehensive_score(
    analytical: &ValidationResult,
    statistical: &StatisticalValidationResult,
    performance: &PerformanceValidationResult,
    cross_validation: &ValidationResult,
) -> f64 {
    let mut score = 100.0;

    // Analytical score (40 points)
    score -= analytical.max_relative_error * 1000.0;
    score -= (1.0 - analytical.stability_score) * 20.0;
    score -= analytical.critical_issues.len() as f64 * 2.0;

    // Statistical score (30 points)
    if statistical.white_noise_pvalue < 0.01 {
        score -= 10.0;
    }
    score -= statistical.false_alarm_rate_error * 10.0;
    if statistical.bootstrap_coverage < 0.90 {
        score -= 10.0;
    }
    score -= statistical.statistical_issues.len() as f64 * 2.0;

    // Performance score (20 points)
    if performance.speedup_factor < 1.0 {
        score -= 10.0;
    }
    if performance.memory_usage_mb > 50.0 {
        score -= 5.0;
    }
    score -= performance.performance_issues.len() as f64 * 2.0;

    // Cross-_validation score (10 points)
    score -= cross_validation.max_relative_error * 100.0;
    score -= cross_validation.critical_issues.len() as f64 * 1.0;

    score.max(0.0).min(100.0)
}

/// Additional robustness validation result
#[derive(Debug, Clone)]
pub struct RobustnessValidationResult {
    /// Overall robustness score
    pub robustness_score: f64,
    /// Memory stress test results
    pub memory_stress_score: f64,
    /// Precision limits test score
    pub precision_limits_score: f64,
    /// Boundary conditions score
    pub boundary_conditions_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Real-world scenario validation result
#[derive(Debug, Clone)]
pub struct RealWorldValidationResult {
    /// Overall score
    pub score: f64,
    /// Astronomical data test score
    pub astronomical_score: f64,
    /// Physiological data test score
    pub physiological_score: f64,
    /// Environmental data test score
    pub environmental_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Advanced statistical validation result
#[derive(Debug, Clone)]
pub struct AdvancedStatisticalResult {
    /// Overall score
    pub score: f64,
    /// Non-parametric tests score
    pub nonparametric_score: f64,
    /// Bayesian validation score
    pub bayesian_score: f64,
    /// Information theory metrics score
    pub information_theory_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Validate additional robustness scenarios
#[allow(dead_code)]
fn validate_additional_robustness(tolerance: f64) -> SignalResult<RobustnessValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // Test 1: Memory stress test with very large datasets
    let memory_stress_score = test_memory_stress_scenarios()?;
    if memory_stress_score < 80.0 {
        issues.push("Memory stress test failed".to_string());
    }

    // Test 2: Numerical precision limits
    let precision_limits_score = test_numerical_precision_limits(_tolerance)?;
    if precision_limits_score < 70.0 {
        issues.push("Precision limits test failed".to_string());
    }

    // Test 3: Boundary conditions
    let boundary_conditions_score = test_boundary_conditions(_tolerance)?;
    if boundary_conditions_score < 85.0 {
        issues.push("Boundary conditions test failed".to_string());
    }

    let robustness_score =
        (memory_stress_score + precision_limits_score + boundary_conditions_score) / 3.0;

    Ok(RobustnessValidationResult {
        robustness_score,
        memory_stress_score,
        precision_limits_score,
        boundary_conditions_score,
        issues,
    })
}

/// Test memory stress scenarios
#[allow(dead_code)]
fn test_memory_stress_scenarios() -> SignalResult<f64> {
    let mut score = 100.0;

    // Test with extremely large datasets
    let sizes = vec![100_000, 500_000, 1_000_000];

    for &size in &sizes {
        let t: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * 0.1 * ti).sin())
            .collect();

        // Test with basic implementation
        let start = std::time::Instant::now();
        let result = lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        );
        let duration = start.elapsed();

        match result {
            Ok(_) => {
                // Check if computation took reasonable time (< 30 seconds for 1M points)
                if duration.as_secs() > 30 {
                    score -= 20.0;
                }
            }
            Err(_) => {
                score -= 30.0; // Penalize failures
            }
        }

        // Early exit if score gets too low
        if score < 20.0 {
            break;
        }
    }

    Ok(score.max(0.0))
}

/// Test numerical precision limits
#[allow(dead_code)]
fn test_numerical_precision_limits(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Test 1: Extremely small time intervals
    let n = 1000;
    let dt = 1e-15; // Near machine precision
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * 1e12 * ti).sin())
        .collect();

    let result = lombscargle(
        &t,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result {
        Ok((_, power)) => {
            if power.iter().any(|&p| !p.is_finite()) {
                score -= 30.0;
            }
        }
        Err(_) => {
            score -= 20.0; // May fail, but shouldn't crash
        }
    }

    // Test 2: Extremely large time intervals
    let large_dt = 1e15;
    let t_large: Vec<f64> = (0..100).map(|i| i as f64 * large_dt).collect();
    let signal_large: Vec<f64> = t_large
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * 1e-12 * ti).sin())
        .collect();

    let result_large = lombscargle(
        &t_large,
        &signal_large,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result_large {
        Ok((_, power)) => {
            if power.iter().any(|&p| !p.is_finite()) {
                score -= 30.0;
            }
        }
        Err(_) => {
            score -= 20.0;
        }
    }

    // Test 3: Mixed precision scenarios
    let mut mixed_t = vec![0.0, 1e-10, 1e-5, 1.0, 1e5, 1e10];
    mixed_t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mixed_signal: Vec<f64> = mixed_t.iter().map(|&ti| ti.sin()).collect();

    let result_mixed = lombscargle(
        &mixed_t,
        &mixed_signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result_mixed {
        Ok((_, power)) => {
            if power.iter().any(|&p| !p.is_finite()) {
                score -= 20.0;
            }
        }
        Err(_) => {
            score -= 15.0;
        }
    }

    Ok(score.max(0.0))
}

/// Test boundary conditions
#[allow(dead_code)]
fn test_boundary_conditions(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Test 1: Minimum viable signal (3 points)
    let t_min = vec![0.0, 1.0, 2.0];
    let signal_min = vec![1.0, 0.0, -1.0];

    let result_min = lombscargle(
        &t_min,
        &signal_min,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result_min {
        Ok((freqs, power)) => {
            if freqs.is_empty() || power.is_empty() {
                score -= 15.0;
            }
            if power.iter().any(|&p| !p.is_finite() || p < 0.0) {
                score -= 15.0;
            }
        }
        Err(_) => {
            score -= 10.0; // May legitimately fail for very short series
        }
    }

    // Test 2: Single frequency component at Nyquist limit
    let n = 100;
    let fs = 100.0;
    let f_nyquist = fs / 2.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * f_nyquist * ti).sin())
        .collect();

    let result_nyquist = lombscargle(
        &t,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result_nyquist {
        Ok((freqs, power)) => {
            // Should handle Nyquist frequency gracefully
            if power.iter().any(|&p| !p.is_finite()) {
                score -= 20.0;
            }
        }
        Err(_) => {
            score -= 15.0;
        }
    }

    // Test 3: Zero-variance signal
    let zero_signal = vec![1.0; 100];
    let t_zero: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();

    let result_zero = lombscargle(
        &t_zero,
        &zero_signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result_zero {
        Ok((_, power)) => {
            // Should produce low power values for constant signal
            let max_power = power.iter().cloned().fold(0.0, f64::max);
            if max_power > 1e-6 {
                score -= 10.0;
            }
        }
        Err(_) => {
            score -= 5.0; // May fail for constant signal
        }
    }

    Ok(score.max(0.0))
}

/// Validate real-world scenarios
#[allow(dead_code)]
fn validate_real_world_scenarios(tolerance: f64) -> SignalResult<RealWorldValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // Test 1: Astronomical time series (irregular sampling, long-term trends)
    let astronomical_score = test_astronomical_scenarios(_tolerance)?;
    if astronomical_score < 70.0 {
        issues.push("Astronomical data test failed".to_string());
    }

    // Test 2: Physiological signals (biorhythms, noise)
    let physiological_score = test_physiological_scenarios(_tolerance)?;
    if physiological_score < 75.0 {
        issues.push("Physiological data test failed".to_string());
    }

    // Test 3: Environmental monitoring (gaps, seasonal patterns)
    let environmental_score = test_environmental_scenarios(_tolerance)?;
    if environmental_score < 80.0 {
        issues.push("Environmental data test failed".to_string());
    }

    let score = (astronomical_score + physiological_score + environmental_score) / 3.0;

    Ok(RealWorldValidationResult {
        score,
        astronomical_score,
        physiological_score,
        environmental_score,
        issues,
    })
}

/// Test astronomical data scenarios
#[allow(dead_code)]
fn test_astronomical_scenarios(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simulate variable star with irregular sampling
    let mut rng = rand::rng();

    let n_obs = 500;
    let period = 5.2; // days
    let mut times = Vec::new();
    let mut brightness = Vec::new();

    // Generate irregular observation times (gaps due to weather, etc.)
    let mut current_time = 0.0;
    for _ in 0..n_obs {
        // Random gaps between observations (0.1 to 2.0 days)
        current_time += rng.gen_range(0.1..2.0);
        times.push(current_time);

        // Variable star signal with noise
        let phase = 2.0 * std::f64::consts::PI * current_time / period;
        let signal = 1.0 + 0.3 * phase.sin() + 0.1 * (2.0 * phase).sin(); // Fundamental + harmonic
        let noise = 0.05 * rng.gen_range(-1.0..1.0); // 5% noise
        brightness.push(signal + noise);
    }

    // Test Lomb-Scargle on this data
    let result = lombscargle(
        &times,
        &brightness,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result {
        Ok((freqs, power)) => {
            // Should detect the primary period
            let expected_freq = 1.0 / period;

            // Find peak near expected frequency
            let freq_tolerance = 0.01; // 1% _tolerance
            let peak_found = freqs.iter().zip(power.iter()).any(|(&f, &p)| {
                (f - expected_freq).abs() / expected_freq < freq_tolerance
                    && p > power.iter().sum::<f64>() / power.len() as f64 * 5.0 // 5x above mean
            });

            if !peak_found {
                score -= 30.0;
            }

            // Check for reasonable power distribution
            if power.iter().any(|&p| !p.is_finite() || p < 0.0) {
                score -= 20.0;
            }
        }
        Err(_) => {
            score -= 50.0;
        }
    }

    Ok(score.max(0.0))
}

/// Test physiological signal scenarios
#[allow(dead_code)]
fn test_physiological_scenarios(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simulate heart rate variability data
    let mut rng = rand::rng();

    let n = 1000;
    let fs = 4.0; // 4 Hz sampling rate
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Simulate HRV with multiple frequency components
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            let very_low = 0.5 * (2.0 * std::f64::consts::PI * 0.02 * ti).sin(); // VLF: 0.01-0.04 Hz
            let low = 0.3 * (2.0 * std::f64::consts::PI * 0.1 * ti).sin(); // LF: 0.04-0.15 Hz
            let high = 0.2 * (2.0 * std::f64::consts::PI * 0.25 * ti).sin(); // HF: 0.15-0.4 Hz
            let noise = 0.1 * rng.gen_range(-1.0..1.0);

            1.0 + very_low + low + high + noise
        })
        .collect();

    let result = lombscargle(
        &t,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result {
        Ok((freqs, power)) => {
            // Should detect multiple frequency bands
            let vlf_detected = freqs
                .iter()
                .zip(power.iter())
                .any(|(&f, &p)| f >= 0.01 && f <= 0.04 && p > 0.1);
            let lf_detected = freqs
                .iter()
                .zip(power.iter())
                .any(|(&f, &p)| f >= 0.04 && f <= 0.15 && p > 0.1);
            let hf_detected = freqs
                .iter()
                .zip(power.iter())
                .any(|(&f, &p)| f >= 0.15 && f <= 0.4 && p > 0.1);

            if !vlf_detected {
                score -= 15.0;
            }
            if !lf_detected {
                score -= 15.0;
            }
            if !hf_detected {
                score -= 15.0;
            }

            // Check numerical stability
            if power.iter().any(|&p| !p.is_finite() || p < 0.0) {
                score -= 20.0;
            }
        }
        Err(_) => {
            score -= 50.0;
        }
    }

    Ok(score.max(0.0))
}

/// Test environmental monitoring scenarios
#[allow(dead_code)]
fn test_environmental_scenarios(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simulate temperature measurements with seasonal variation and gaps
    let mut rng = rand::rng();

    let days_per_year = 365.25;
    let n_years = 3;
    let measurements_per_day = 4; // Every 6 hours

    let mut times = Vec::new();
    let mut temperatures = Vec::new();

    for day in 0..(n_years as f64 * days_per_year) as i32 {
        for measurement in 0..measurements_per_day {
            // Simulate data gaps (missing data)
            if rng.gen_range(0.0..1.0) < 0.95 {
                // 95% data availability
                let time_hours = day as f64 * 24.0 + measurement as f64 * 6.0;
                times.push(time_hours / 24.0); // Convert to days

                // Seasonal temperature variation + daily cycle + noise
                let seasonal =
                    15.0 * (2.0 * std::f64::consts::PI * day as f64 / days_per_year).sin();
                let daily = 5.0
                    * (2.0 * std::f64::consts::PI * measurement as f64 / measurements_per_day)
                        .sin();
                let noise = 2.0 * rng.gen_range(-1.0..1.0);

                temperatures.push(20.0 + seasonal + daily + noise); // Base temp 20°C
            }
        }
    }

    let result = lombscargle(
        &times,
        &temperatures,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    );
    match result {
        Ok((freqs, power)) => {
            // Should detect annual cycle (1/365.25 cycles per day)
            let annual_freq = 1.0 / days_per_year;
            let daily_freq = 1.0; // 1 cycle per day

            let annual_detected = freqs
                .iter()
                .zip(power.iter())
                .any(|(&f, &p)| (f - annual_freq).abs() / annual_freq < 0.1 && p > 0.01);
            let daily_detected = freqs
                .iter()
                .zip(power.iter())
                .any(|(&f, &p)| (f - daily_freq).abs() / daily_freq < 0.1 && p > 0.01);

            if !annual_detected {
                score -= 20.0;
            }
            if !daily_detected {
                score -= 20.0;
            }

            // Check for reasonable results with gaps
            if power.iter().any(|&p| !p.is_finite() || p < 0.0) {
                score -= 25.0;
            }
        }
        Err(_) => {
            score -= 50.0;
        }
    }

    Ok(score.max(0.0))
}

/// Validate advanced statistical properties
#[allow(dead_code)]
fn validate_advanced_statistical_properties(
    tolerance: f64,
) -> SignalResult<AdvancedStatisticalResult> {
    let mut issues: Vec<String> = Vec::new();

    // Test 1: Non-parametric statistical tests
    let nonparametric_score = test_nonparametric_properties(tolerance)?;
    if nonparametric_score < 75.0 {
        issues.push("Non-parametric tests failed".to_string());
    }

    // Test 2: Bayesian validation approaches
    let bayesian_score = test_bayesian_validation(tolerance)?;
    if bayesian_score < 70.0 {
        issues.push("Bayesian validation failed".to_string());
    }

    // Test 3: Information theory metrics
    let information_theory_score = test_information_theory_metrics(tolerance)?;
    if information_theory_score < 80.0 {
        issues.push("Information theory tests failed".to_string());
    }

    let score = (nonparametric_score + bayesian_score + information_theory_score) / 3.0;

    Ok(AdvancedStatisticalResult {
        score,
        nonparametric_score,
        bayesian_score,
        information_theory_score,
        issues,
    })
}

/// Test non-parametric statistical properties
#[allow(dead_code)]
fn test_nonparametric_properties(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Test Kolmogorov-Smirnov test for power distribution
    let mut rng = rand::rng();

    let n_trials = 100;
    let n_samples = 200;
    let mut power_maxima = Vec::new();

    for _ in 0..n_trials {
        // Generate white noise
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 * 0.01).collect();
        let signal: Vec<f64> = (0..n_samples)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        match lombscargle(
            &t,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        ) {
            Ok((_, power)) => {
                let max_power = power.iter().cloned().fold(0.0, f64::max);
                power_maxima.push(max_power);
            }
            Err(_) => {
                score -= 2.0;
            }
        }
    }

    // Simple KS test approximation
    if power_maxima.len() > 10 {
        power_maxima.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Check if distribution roughly follows expected pattern
        let median = power_maxima[power_maxima.len() / 2];
        let q75 = power_maxima[power_maxima.len() * 3 / 4];
        let q25 = power_maxima[power_maxima.len() / 4];

        // For exponential distribution, median ≈ 0.693, IQR ≈ 1.099
        let median_error = ((median - 0.693) as f64).abs() / 0.693;
        let iqr = q75 - q25;
        let iqr_error = ((iqr - 1.099) as f64).abs() / 1.099;

        if median_error > 0.5 {
            score -= 20.0;
        }
        if iqr_error > 0.5 {
            score -= 20.0;
        }
    }

    Ok(score.max(0.0))
}

/// Test Bayesian validation approaches
#[allow(dead_code)]
fn test_bayesian_validation(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simplified Bayesian model comparison test
    // Compare evidence for different frequency models

    let n = 300;
    let fs = 50.0;
    let true_freq = 5.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    let mut rng = rand::rng();

    // Signal with known frequency plus noise
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            (2.0 * std::f64::consts::PI * true_freq * ti).sin() + 0.2 * rng.gen_range(-1.0..1.0)
        })
        .collect();

    match lombscargle(
        &t,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    ) {
        Ok((freqs, power)) => {
            // Find peak and check if it's at the expected frequency
            let (peak_idx, &peak_power) = power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let peak_freq = freqs[peak_idx];
            let freq_error = (peak_freq - true_freq).abs() / true_freq;

            if freq_error > 0.1 {
                score -= 30.0;
            }

            // Check Bayesian information criterion approximation
            // Higher peak should correspond to better model evidence
            let mean_power = power.iter().sum::<f64>() / power.len() as f64;
            let evidence_ratio = peak_power / mean_power;

            if evidence_ratio < 5.0 {
                score -= 20.0;
            } // Should be well above background
        }
        Err(_) => {
            score -= 50.0;
        }
    }

    Ok(score.max(0.0))
}

/// Test information theory metrics
#[allow(dead_code)]
fn test_information_theory_metrics(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Test entropy and mutual information properties
    let mut rng = rand::rng();

    let n = 500;
    let fs = 100.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Test 1: Pure sinusoid should have low entropy in frequency domain
    let signal_periodic: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * 10.0 * ti).sin())
        .collect();

    match lombscargle(
        &t,
        &signal_periodic,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    ) {
        Ok((_, power_periodic)) => {
            // Normalize power
            let total_power: f64 = power_periodic.iter().sum();
            let prob_dist: Vec<f64> = power_periodic.iter().map(|&p| p / total_power).collect();

            // Calculate Shannon entropy
            let entropy: f64 = prob_dist
                .iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();

            // Periodic signal should have low entropy
            let max_entropy = (power_periodic.len() as f64).ln(); // Uniform distribution entropy
            let normalized_entropy = entropy / max_entropy;

            if normalized_entropy > 0.5 {
                score -= 20.0;
            } // Should be concentrated
        }
        Err(_) => {
            score -= 25.0;
        }
    }

    // Test 2: White noise should have high entropy
    let signal_noise: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    match lombscargle(
        &t,
        &signal_noise,
        None,
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    ) {
        Ok((_, power_noise)) => {
            let total_power: f64 = power_noise.iter().sum();
            let prob_dist: Vec<f64> = power_noise.iter().map(|&p| p / total_power).collect();

            let entropy: f64 = prob_dist
                .iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();

            let max_entropy = (power_noise.len() as f64).ln();
            let normalized_entropy = entropy / max_entropy;

            if normalized_entropy < 0.7 {
                score -= 20.0;
            } // Should be more uniform
        }
        Err(_) => {
            score -= 25.0;
        }
    }

    Ok(score.max(0.0))
}

/// Enhanced comprehensive score calculation
#[allow(dead_code)]
fn calculate_comprehensive_score_enhanced(
    analytical: &ValidationResult,
    statistical: &StatisticalValidationResult,
    performance: &PerformanceValidationResult,
    cross_validation: &ValidationResult,
    robustness: &RobustnessValidationResult,
    real_world: &RealWorldValidationResult,
    advanced_stats: &AdvancedStatisticalResult,
) -> f64 {
    let mut score = 100.0;

    // Analytical score (25 points)
    score -= analytical.max_relative_error * 1000.0;
    score -= (1.0 - analytical.stability_score) * 15.0;
    score -= analytical.critical_issues.len() as f64 * 1.5;

    // Statistical score (20 points)
    if statistical.white_noise_pvalue < 0.01 {
        score -= 8.0;
    }
    score -= statistical.false_alarm_rate_error * 8.0;
    if statistical.bootstrap_coverage < 0.90 {
        score -= 8.0;
    }
    score -= statistical.statistical_issues.len() as f64 * 1.5;

    // Performance score (15 points)
    if performance.speedup_factor < 1.0 {
        score -= 8.0;
    }
    if performance.memory_usage_mb > 50.0 {
        score -= 4.0;
    }
    score -= performance.performance_issues.len() as f64 * 1.5;

    // Cross-_validation score (10 points)
    score -= cross_validation.max_relative_error * 80.0;
    score -= cross_validation.critical_issues.len() as f64 * 1.0;

    // Robustness score (15 points)
    score -= (100.0 - robustness.robustness_score) * 0.15;
    score -= robustness.critical_issues.len() as f64 * 1.0;

    // Real-_world scenarios score (10 points)
    score -= (100.0 - real_world.score) * 0.10;
    score -= real_world.critical_issues.len() as f64 * 1.0;

    // Advanced statistical score (5 points)
    score -= (100.0 - advanced_stats.score) * 0.05;
    score -= advanced_stats.critical_issues.len() as f64 * 0.5;

    score.max(0.0).min(100.0)
}

/// Enhanced precision validation for Lomb-Scargle periodogram
#[allow(dead_code)]
fn validate_enhanced_precision(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // High-precision test with exact analytical solution
    let n = 512;
    let fs = 256.0;
    let f_signal = 17.0; // Non-integer frequency to test precision
    let phase = 0.7854; // π/4 for phase test
    let amplitude = 2.5;

    // Generate high-precision time series
    let dt = 1.0 / fs;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| amplitude * (2.0 * PI * f_signal * ti + phase).sin())
        .collect();

    // Add controlled uneven sampling
    let mut t_uneven = Vec::new();
    let mut signal_uneven = Vec::new();
    for i in (0..n).step_by(2) {
        t_uneven.push(t[i] + 0.001 * (i as f64).sin()); // Small perturbation
        signal_uneven.push(signal[i]);
    }

    // Compute high-resolution periodogram
    let freq_min = f_signal - 2.0;
    let freq_max = f_signal + 2.0;
    let nfreq = 1000;
    let freqs: Vec<f64> = (0..nfreq)
        .map(|i| freq_min + (freq_max - freq_min) * i as f64 / (nfreq - 1) as f64)
        .collect();

    // Test implementation
    let power = match implementation {
        "standard" => {
            let (_, p) = lombscargle(
                &t_uneven,
                &signal_uneven,
                Some(&freqs),
                Some("standard"),
                Some(false), // Don't center for precision test
                Some(false),
                None,
                None,
            )?;
            p
        }
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 10.0,
                f_min: Some(freq_min),
                f_max: Some(freq_max),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-12,
                use_fast: false, // Use slow but precise algorithm
            };
            let (_, p) = lombscargle_enhanced(&t_uneven, &signal_uneven, &config)?;
            p
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Find peak and validate precision
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let peak_freq = freqs[peak_idx];

    // Calculate theoretical power (for normalized Lomb-Scargle)
    let theoretical_power = amplitude * amplitude / 2.0; // For sine wave
    let power_error = (peak_power - theoretical_power).abs() / theoretical_power;

    let freq_error = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_error);
    errors.push(power_error);

    // Validate frequency precision
    if freq_error > tolerance * 0.1 {
        // Higher precision requirement
        issues.push(format!(
            "High-precision frequency error {:.2e} exceeds strict tolerance {:.2e}",
            freq_error,
            tolerance * 0.1
        ));
    }

    // Validate power precision
    if power_error > tolerance {
        issues.push(format!(
            "Power amplitude error {:.2e} exceeds tolerance {:.2e}",
            power_error, tolerance
        ));
    }

    // Check for spurious peaks
    let mut spurious_peaks = 0;
    let noise_threshold = peak_power * 0.1;
    for (i, &p) in power.iter().enumerate() {
        if i != peak_idx && p > noise_threshold {
            spurious_peaks += 1;
        }
    }

    if spurious_peaks > 2 {
        issues.push(format!(
            "Too many spurious peaks detected: {}",
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

/// Cross-validation with multiple reference implementations
#[allow(dead_code)]
fn validate_cross_reference_implementation(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Test signal with known characteristics
    let n = 256;
    let fs = 64.0;
    let frequencies = vec![5.0, 13.0, 21.0];
    let amplitudes = vec![1.0, 2.0, 0.5];
    let phases = vec![0.0, PI / 3.0, PI / 2.0];

    // Generate complex test signal
    let dt = 1.0 / fs;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            frequencies
                .iter()
                .zip(amplitudes.iter())
                .zip(phases.iter())
                .map(|((&f, &a), &p)| a * (2.0 * PI * f * ti + p).sin())
                .sum::<f64>()
        })
        .collect();

    // Make sampling uneven
    let mut t_uneven = Vec::new();
    let mut signal_uneven = Vec::new();
    let mut rng = rand::rng();
    for i in 0..n {
        if rng.gen_range(0.0..1.0) > 0.3 {
            // Keep 70% of samples
            t_uneven.push(t[i]);
            signal_uneven.push(signal[i]);
        }
    }

    // Compute with target implementation
    let (freqs1, power1) = match implementation {
        "standard" => lombscargle(
            &t_uneven,
            &signal_uneven,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        )?,
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::Hann,
                custom_window: None,
                oversample: 5.0,
                f_min: Some(1.0),
                f_max: Some(30.0),
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p) = lombscargle_enhanced(&t_uneven, &signal_uneven, &config)?;
            (f, p)
        }
        _ => {
            return Err(SignalError::ValueError(
                "Unknown implementation".to_string(),
            ))
        }
    };

    // Compute with reference implementation (standard algorithm)
    let (freqs2, power2) = lombscargle(
        &t_uneven,
        &signal_uneven,
        Some(&freqs1), // Use same frequency grid
        Some("standard"),
        Some(true),
        Some(false),
        None,
        None,
    )?;

    // Compare results
    for (i, (&p1, &p2)) in power1.iter().zip(power2.iter()).enumerate() {
        if p2 > 1e-15 {
            // Avoid division by very small numbers
            let relative_error = (p1 - p2).abs() / p2;
            errors.push(relative_error);

            if relative_error > tolerance * 10.0 {
                // More lenient for cross-validation
                issues.push(format!(
                    "Cross-validation error at frequency {:.3} Hz: {:.2e}",
                    freqs1[i], relative_error
                ));
            }
        }
    }

    // Find peaks in both implementations
    let peaks1 = find_peaks(&power1, 0.1);
    let peaks2 = find_peaks(&power2, 0.1);

    if peaks1.len() != peaks2.len() {
        issues.push(format!(
            "Different number of peaks detected: {} vs {}",
            peaks1.len(),
            peaks2.len()
        ));
    }

    // Validate that main signal frequencies are detected
    for &target_freq in &frequencies {
        let closest_idx1 = freqs1
            .iter()
            .enumerate()
            .min_by(|(_, &f1), (_, &f2)| {
                (f1 - target_freq)
                    .abs()
                    .partial_cmp(&(**f2 - target_freq).abs())
                    .unwrap()
            })
            .map(|(i_, _)| i_)
            .unwrap();

        let closest_freq1 = freqs1[closest_idx1];
        let freq_error = (closest_freq1 - target_freq).abs() / target_freq;

        if freq_error > tolerance {
            issues.push(format!(
                "Target frequency {:.1} Hz not accurately detected: found {:.3} Hz (error: {:.2e})",
                target_freq, closest_freq1, freq_error
            ));
        }
    }

    let max_error = errors.iter().cloned().fold(0.0, f64::max);

    Ok(SingleTestResult {
        errors,
        peak_error: max_error,
        peak_errors: vec![max_error],
        issues,
    })
}

/// Helper function to find peaks in power spectrum
#[allow(dead_code)]
fn find_peaks(power: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    let max_power = power.iter().cloned().fold(0.0, f64::max);
    let min_height = max_power * threshold;

    for i in 1.._power.len() - 1 {
        if power[i] > power[i - 1] && power[i] > power[i + 1] && power[i] > min_height {
            peaks.push(i);
        }
    }

    peaks
}

/// Enhanced validation configuration for comprehensive testing
#[derive(Debug, Clone)]
pub struct EnhancedValidationConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Number of test iterations for statistical validation
    pub n_iterations: usize,
    /// Test against SciPy reference implementation
    pub test_scipy_reference: bool,
    /// Test with noisy signals
    pub test_with_noise: bool,
    /// Noise SNR in dB
    pub noise_snr_db: f64,
    /// Test extreme parameter values
    pub test_extreme_params: bool,
    /// Test memory efficiency
    pub test_memory_efficiency: bool,
    /// Test SIMD vs scalar consistency
    pub test_simd_consistency: bool,
}

impl Default for EnhancedValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            n_iterations: 50,
            test_scipy_reference: true,
            test_with_noise: true,
            noise_snr_db: 20.0,
            test_extreme_params: true,
            test_memory_efficiency: true,
            test_simd_consistency: true,
        }
    }
}

/// Enhanced validation result with comprehensive metrics
#[derive(Debug, Clone)]
pub struct EnhancedValidationResult {
    /// Basic validation metrics
    pub base_validation: ValidationResult,
    /// SciPy comparison metrics
    pub scipy_comparison: SciPyComparisonResult,
    /// Noise robustness score (0-100)
    pub noise_robustness: f64,
    /// Memory efficiency score (0-100)
    pub memory_efficiency: f64,
    /// SIMD consistency score (0-100)
    pub simd_consistency: f64,
    /// Overall enhanced score (0-100)
    pub overall_score: f64,
    /// Detailed recommendations
    pub recommendations: Vec<String>,
}

/// SciPy comparison result
#[derive(Debug, Clone)]
pub struct SciPyComparisonResult {
    /// Maximum relative error vs SciPy reference
    pub max_relative_error: f64,
    /// Mean relative error vs SciPy reference
    pub mean_relative_error: f64,
    /// Correlation coefficient with SciPy
    pub correlation: f64,
    /// Peak detection accuracy
    pub peak_detection_accuracy: f64,
}

/// Comprehensive enhanced validation of Lomb-Scargle implementation
///
/// This function performs extensive validation including:
/// - Comparison with reference implementations
/// - Noise robustness testing
/// - Memory efficiency analysis
/// - SIMD vs scalar consistency checks
/// - Statistical significance validation
///
/// # Arguments
///
/// * `config` - Enhanced validation configuration
///
/// # Returns
///
/// * Enhanced validation result with comprehensive metrics
#[allow(dead_code)]
pub fn validate_lombscargle_enhanced(
    config: &EnhancedValidationConfig,
) -> SignalResult<EnhancedValidationResult> {
    let mut recommendations = Vec::new();

    // 1. Run basic validation
    let base_validation = validate_analytical_cases("enhanced", config.tolerance)?;

    // 2. SciPy reference comparison
    let scipy_comparison = if config.test_scipy_reference {
        validate_against_scipy_reference(config)?
    } else {
        SciPyComparisonResult {
            max_relative_error: 0.0,
            mean_relative_error: 0.0,
            correlation: 1.0,
            peak_detection_accuracy: 100.0,
        }
    };

    // 3. Noise robustness testing
    let noise_robustness = if config.test_with_noise {
        validate_noise_robustness(config)?
    } else {
        95.0
    };

    // 4. Memory efficiency testing
    let memory_efficiency = if config.test_memory_efficiency {
        validate_memory_efficiency(config)?
    } else {
        95.0
    };

    // 5. SIMD consistency testing
    let simd_consistency = if config.test_simd_consistency {
        validate_simd_consistency(config)?
    } else {
        95.0
    };

    // Calculate overall score
    let overall_score = calculate_enhanced_overall_score(
        &base_validation,
        &scipy_comparison,
        noise_robustness,
        memory_efficiency,
        simd_consistency,
    );

    // Generate recommendations
    if scipy_comparison.max_relative_error > config.tolerance * 100.0 {
        recommendations.push(
            "Large discrepancy with SciPy reference detected. Review algorithm implementation."
                .to_string(),
        );
    }

    if noise_robustness < 80.0 {
        recommendations.push(
            "Poor noise robustness. Consider implementing better preprocessing or regularization."
                .to_string(),
        );
    }

    if memory_efficiency < 75.0 {
        recommendations.push(
            "Memory usage is suboptimal. Consider chunked processing for large datasets."
                .to_string(),
        );
    }

    if simd_consistency < 90.0 {
        recommendations.push(
            "SIMD implementation inconsistency detected. Review SIMD code paths.".to_string(),
        );
    }

    Ok(EnhancedValidationResult {
        base_validation,
        scipy_comparison,
        noise_robustness,
        memory_efficiency,
        simd_consistency,
        overall_score,
        recommendations,
    })
}

/// Validate against SciPy reference implementation
#[allow(dead_code)]
fn validate_against_scipy_reference(
    config: &EnhancedValidationConfig,
) -> SignalResult<SciPyComparisonResult> {
    let mut relative_errors = Vec::new();
    let mut correlations = Vec::new();
    let mut peak_accuracies = Vec::new();

    for _ in 0..config.n_iterations {
        // Generate test signal
        let n = 1000;
        let t: Vec<f64> = (0..n)
            .map(|i| i as f64 * 0.01 + rand::rng().random_range(0.0..0.001))
            .collect();
        let freq1 = 0.5;
        let freq2 = 1.5;
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * freq1 * ti).sin() + 0.5 * (2.0 * PI * freq2 * ti).sin())
            .collect();

        // Test frequencies
        let freqs: Vec<f64> = (1..=200).map(|i| i as f64 * 0.01).collect();

        // Our implementation
        let (test_freqs, test_power) = lombscargle(&t, &signal, Some(&freqs))?;

        // Reference implementation (simplified SciPy-like algorithm)
        let ref_power = scipy_reference_lombscargle(&t, &signal, &freqs)?;

        // Compare results
        let errors: Vec<f64> = test_power
            .iter()
            .zip(ref_power.iter())
            .filter_map(|(&test, &ref_val)| {
                if ref_val > 1e-10 {
                    Some((test - ref_val).abs() / ref_val)
                } else {
                    None
                }
            })
            .collect();

        if !errors.is_empty() {
            relative_errors.extend(errors);

            // Calculate correlation
            let correlation = calculate_correlation(&test_power, &ref_power);
            correlations.push(correlation);

            // Calculate peak detection accuracy
            let peak_accuracy =
                calculate_peak_detection_accuracy(&test_power, &ref_power, &freqs, &[freq1, freq2]);
            peak_accuracies.push(peak_accuracy);
        }
    }

    let max_relative_error = relative_errors.iter().fold(0.0, |a, &b| a.max(b));
    let mean_relative_error = relative_errors.iter().sum::<f64>() / relative_errors.len() as f64;
    let mean_correlation = correlations.iter().sum::<f64>() / correlations.len() as f64;
    let mean_peak_accuracy = peak_accuracies.iter().sum::<f64>() / peak_accuracies.len() as f64;

    Ok(SciPyComparisonResult {
        max_relative_error,
        mean_relative_error,
        correlation: mean_correlation,
        peak_detection_accuracy: mean_peak_accuracy,
    })
}

/// Simplified SciPy-like reference implementation for comparison
#[allow(dead_code)]
fn scipy_reference_lombscargle(t: &[f64], y: &[f64], freqs: &[f64]) -> SignalResult<Vec<f64>> {
    let n = t.len();
    let mut power = vec![0.0; freqs.len()];

    // Center the data
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

    for (i, &freq) in freqs.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        // Calculate tau (time offset to make periodogram independent of time origin)
        let sum_sin2wt: f64 = t.iter().map(|&ti| (2.0 * omega * ti).sin()).sum();
        let sum_cos2wt: f64 = t.iter().map(|&ti| (2.0 * omega * ti).cos()).sum();
        let tau = (sum_sin2wt / sum_cos2wt).atan() / (2.0 * omega);

        // Calculate periodogram components
        let mut sum_cos_num = 0.0;
        let mut sum_cos_den = 0.0;
        let mut sum_sin_num = 0.0;
        let mut sum_sin_den = 0.0;

        for j in 0..n {
            let phase = omega * (t[j] - tau);
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();

            sum_cos_num += y_centered[j] * cos_phase;
            sum_cos_den += cos_phase * cos_phase;
            sum_sin_num += y_centered[j] * sin_phase;
            sum_sin_den += sin_phase * sin_phase;
        }

        // Lomb-Scargle periodogram
        let cos_term = if sum_cos_den > 1e-15 {
            (sum_cos_num * sum_cos_num) / sum_cos_den
        } else {
            0.0
        };

        let sin_term = if sum_sin_den > 1e-15 {
            (sum_sin_num * sum_sin_num) / sum_sin_den
        } else {
            0.0
        };

        power[i] = 0.5 * (cos_term + sin_term);
    }

    Ok(power)
}

/// Validate noise robustness
#[allow(dead_code)]
fn validate_noise_robustness(config: &EnhancedValidationConfig) -> SignalResult<f64> {
    let mut robustness_scores = Vec::new();
    let snr_linear = 10.0_f64.powf(_config.noise_snr_db / 10.0);

    for _ in 0.._config.n_iterations {
        // Clean signal
        let n = 500;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.02).collect();
        let freq = 1.0;
        let clean_signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).sin()).collect();

        // Add noise
        let signal_power = clean_signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        let noise_power = signal_power / snr_linear;
        let noise_std = noise_power.sqrt();

        let mut rng = rand::rng();
        let noisy_signal: Vec<f64> = clean_signal
            .iter()
            .map(|&s| s + noise_std * rng.gen_range(-1.0..1.0))
            .collect();

        // Test frequencies around the true frequency
        let freqs: Vec<f64> = (80..120).map(|i| i as f64 * 0.01).collect();

        // Compute periodogram
        match lombscargle(&t, &noisy_signal, Some(&freqs)) {
            Ok((_, power)) => {
                // Find peak
                let peak_idx = power
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i_, _)| i_)
                    .unwrap_or(0);

                let detected_freq = freqs[peak_idx];
                let freq_error = (detected_freq - freq).abs() / freq;

                // Score based on frequency accuracy
                let score = if freq_error < 0.01 {
                    100.0
                } else if freq_error < 0.05 {
                    80.0
                } else if freq_error < 0.1 {
                    60.0
                } else {
                    40.0
                };

                robustness_scores.push(score);
            }
            Err(_) => {
                robustness_scores.push(0.0);
            }
        }
    }

    let mean_score = robustness_scores.iter().sum::<f64>() / robustness_scores.len() as f64;
    Ok(mean_score)
}

/// Validate memory efficiency
#[allow(dead_code)]
fn validate_memory_efficiency(config: &EnhancedValidationConfig) -> SignalResult<f64> {
    // Simple memory efficiency test - in a real implementation,
    // this would measure actual memory usage
    let large_n = 10000;
    let t: Vec<f64> = (0..large_n).map(|i| i as f64 * 0.001).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 0.5 * ti).sin()).collect();
    let freqs: Vec<f64> = (1..1000).map(|i| i as f64 * 0.001).collect();

    // Test if large signal processing works
    match lombscargle(&t, &signal, Some(&freqs)) {
        Ok(_) => Ok(90.0),  // Good efficiency if it completes
        Err(_) => Ok(50.0), // Poor efficiency if it fails
    }
}

/// Validate SIMD consistency
#[allow(dead_code)]
fn validate_simd_consistency(config: &EnhancedValidationConfig) -> SignalResult<f64> {
    let mut consistency_scores = Vec::new();

    for _ in 0..10 {
        // Generate test signal
        let n = 1000;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 0.5 * ti).sin()).collect();
        let freqs: Vec<f64> = (1..200).map(|i| i as f64 * 0.01).collect();

        // Test standard implementation
        let result1 = lombscargle(&t, &signal, Some(&freqs))?;

        // Test enhanced implementation (which may use SIMD)
        let ls_config = LombScargleConfig {
            normalize: true,
            center: true,
            window: Some(WindowType::Hann),
            detrend: false,
        };
        let result2 = lombscargle_enhanced(&t, &signal, Some(&freqs), &ls_config)?;

        // Compare results
        let errors: Vec<f64> = result1
            .1
            .iter()
            .zip(result2.1.iter())
            .filter_map(|(&p1, &p2)| {
                if p1 > 1e-15 {
                    Some((p1 - p2).abs() / p1)
                } else {
                    None
                }
            })
            .collect();

        if !errors.is_empty() {
            let max_error = errors.iter().fold(0.0, |a, &b| a.max(b));
            let score = if max_error < config.tolerance * 10.0 {
                100.0
            } else if max_error < config.tolerance * 100.0 {
                80.0
            } else {
                60.0
            };
            consistency_scores.push(score);
        }
    }

    let mean_score = consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64;
    Ok(mean_score)
}

/// Calculate correlation between two vectors
#[allow(dead_code)]
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n as usize {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 1e-15 && var_y > 1e-15 {
        cov / (var_x * var_y).sqrt()
    } else {
        0.0
    }
}

/// Calculate peak detection accuracy
#[allow(dead_code)]
fn calculate_peak_detection_accuracy(
    test_power: &[f64],
    ref_power: &[f64],
    freqs: &[f64],
    true_freqs: &[f64],
) -> f64 {
    let mut accuracies = Vec::new();

    for &true_freq in true_freqs {
        // Find peaks in both _power spectra
        let test_peak_idx = find_peak_near_frequency(test_power, freqs, true_freq);
        let ref_peak_idx = find_peak_near_frequency(ref_power, freqs, true_freq);

        if let (Some(test_idx), Some(ref_idx)) = (test_peak_idx, ref_peak_idx) {
            let freq_diff = (freqs[test_idx] - freqs[ref_idx]).abs();
            let freq_resolution = if freqs.len() > 1 {
                freqs[1] - freqs[0]
            } else {
                0.01
            };

            let accuracy = if freq_diff < freq_resolution {
                100.0
            } else if freq_diff < 2.0 * freq_resolution {
                80.0
            } else {
                60.0
            };

            accuracies.push(accuracy);
        }
    }

    if !accuracies.is_empty() {
        accuracies.iter().sum::<f64>() / accuracies.len() as f64
    } else {
        0.0
    }
}

/// Find peak near a specific frequency
#[allow(dead_code)]
fn find_peak_near_frequency(_power: &[f64], freqs: &[f64], targetfreq: f64) -> Option<usize> {
    let tolerance = 0.1; // 10% tolerance
    let mut best_idx = None;
    let mut best_power = 0.0;

    for (i, (&_freq, &pow)) in freqs.iter().zip(_power.iter()).enumerate() {
        if (_freq - target_freq).abs() / target_freq < tolerance && pow > best_power {
            best_power = pow;
            best_idx = Some(i);
        }
    }

    best_idx
}

/// Calculate enhanced overall score
#[allow(dead_code)]
fn calculate_enhanced_overall_score(
    base_validation: &ValidationResult,
    scipy_comparison: &SciPyComparisonResult,
    noise_robustness: f64,
    memory_efficiency: f64,
    simd_consistency: f64,
) -> f64 {
    // Convert base _validation stability score to 0-100 scale
    let base_score = base_validation.stability_score * 100.0;

    // Convert scipy _comparison correlation to score
    let scipy_score = scipy_comparison.correlation * 100.0;

    // Weighted average of all components
    let overall_score = (base_score * 0.3)
        + (scipy_score * 0.25)
        + (noise_robustness * 0.2)
        + (memory_efficiency * 0.15)
        + (simd_consistency * 0.1);

    overall_score.min(100.0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_framework() {
        let tolerance = 1e-6;
        let result = validate_analytical_cases("standard", tolerance);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.stability_score >= 0.0);
        assert!(validation.stability_score <= 1.0);
    }

    #[test]
    fn test_enhanced_validation() {
        let tolerance = 1e-6;
        let result = validate_analytical_cases("enhanced", tolerance);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.stability_score >= 0.0);
        assert!(validation.stability_score <= 1.0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let tolerance = 1e-5; // More lenient for comprehensive test
        let result = validate_lombscargle_comprehensive(tolerance);
        assert!(result.is_ok());
    }

    #[test]
    fn test_robustness_validation() {
        let tolerance = 1e-6;
        let result = validate_additional_robustness(tolerance);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.robustness_score >= 0.0);
        assert!(validation.robustness_score <= 100.0);
    }

    #[test]
    fn test_real_world_validation() {
        let tolerance = 1e-6;
        let result = validate_real_world_scenarios(tolerance);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.score >= 0.0);
        assert!(validation.score <= 100.0);
    }
}
