// Robustness validation utilities for Lomb-Scargle periodogram
//
// This module provides robustness validation functions for Lomb-Scargle
// implementations, including memory stress tests, numerical precision limits,
// boundary conditions, and edge case handling.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig, WindowType};
use super::types::{RobustnessValidationResult, SingleTestResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use std::time::Instant;

// Define PI constant for compatibility
const PI: f64 = std::f64::consts::PI;

/// Validate additional robustness scenarios
#[allow(dead_code)]
pub fn validate_additional_robustness(tolerance: f64) -> SignalResult<RobustnessValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // Test 1: Memory stress test with very large datasets
    let memory_stress_score = test_memory_stress_scenarios()?;
    if memory_stress_score < 80.0 {
        issues.push("Memory stress test failed".to_string());
    }

    // Test 2: Numerical precision limits
    let precision_limits_score = test_numerical_precision_limits(tolerance)?;
    if precision_limits_score < 70.0 {
        issues.push("Precision limits test failed".to_string());
    }

    // Test 3: Boundary conditions
    let boundary_conditions_score = test_boundary_conditions(tolerance)?;
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
pub fn test_memory_stress_scenarios() -> SignalResult<f64> {
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
pub fn test_numerical_precision_limits(tolerance: f64) -> SignalResult<f64> {
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
pub fn test_boundary_conditions(tolerance: f64) -> SignalResult<f64> {
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

/// Test helper functions for edge cases
#[allow(dead_code)]
pub fn test_short_time_series(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Very short series (should handle gracefully or return appropriate error)
    let t = vec![0.0, 0.1, 0.2];
    let signal = vec![1.0, 0.0, -1.0];

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
                "Unknown implementation".to_string(),
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
pub fn test_constant_signal(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    let n = 100;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let signal = vec![1.0; n]; // Constant signal

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
                "Unknown implementation".to_string(),
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
pub fn test_sparse_sampling(implementation: &str) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors = Vec::new();

    // Very sparse sampling - only 10 points over 10 seconds
    let t = vec![0.0, 1.0, 2.5, 3.8, 4.2, 5.9, 7.1, 8.3, 9.0, 10.0];
    let f_signal = 0.5; // 0.5 Hz signal
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
                "Unknown implementation".to_string(),
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
pub fn test_signal_with_outliers(implementation: &str) -> SignalResult<SingleTestResult> {
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
                "Unknown implementation".to_string(),
            ))
        }
    };

    match result {
        Ok((freqs, power)) => {
            // Check that the implementation doesn't crash with outliers
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
pub fn test_small_values(implementation: &str) -> SignalResult<SingleTestResult> {
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
                "Unknown implementation".to_string(),
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
pub fn test_large_values(implementation: &str) -> SignalResult<SingleTestResult> {
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
                "Unknown implementation".to_string(),
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
pub fn test_extreme_timescales(implementation: &str) -> SignalResult<SingleTestResult> {
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
                "Unknown implementation".to_string(),
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