//! Edge case testing for Lomb-Scargle validation
//!
//! This module provides comprehensive edge case testing to ensure robust handling
//! of unusual or boundary conditions in Lomb-Scargle implementations.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use std::f64::consts::PI;

use super::config::EdgeCaseRobustnessResults;

/// Results for comprehensive edge case validation
#[derive(Debug, Clone)]
pub struct EdgeCaseValidationResult {
    pub empty_signal_handled: bool,
    pub single_point_handled: bool,
    pub duplicate_times_handled: bool,
    pub large_values_stable: bool,
    pub small_values_stable: bool,
    pub nan_input_handled: bool,
    pub constant_signal_correct: bool,
    pub irregular_sampling_stable: bool,
    pub tests_passed: usize,
    pub total_tests: usize,
    pub success_rate: f64,
}

impl Default for EdgeCaseValidationResult {
    fn default() -> Self {
        Self {
            empty_signal_handled: false,
            single_point_handled: false,
            duplicate_times_handled: false,
            large_values_stable: false,
            small_values_stable: false,
            nan_input_handled: false,
            constant_signal_correct: false,
            irregular_sampling_stable: false,
            tests_passed: 0,
            total_tests: 0,
            success_rate: 0.0,
        }
    }
}

/// Numerical robustness test results
#[derive(Debug, Clone, Default)]
pub struct NumericalRobustnessResult {
    pub close_frequency_resolved: bool,
    pub high_dynamic_range_stable: bool,
    pub noisy_signal_stable: bool,
    pub extreme_frequencies_stable: bool,
    pub overall_robustness_score: f64,
}

/// Helper function to run Lomb-Scargle implementations
fn run_lombscargle(
    implementation: &str,
    times: &[f64],
    signal: &[f64],
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    match implementation {
        "standard" => lombscargle(
            times,
            signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        ),
        "enhanced" => {
            let (f, p) = lombscargle(
                times,
                signal,
                None,
                Some("standard"),
                Some(true),
                Some(false),
                Some(1.0),
                None,
            )?;
            Ok((f, p))
        }
        _ => Err(SignalError::ValueError(
            "Unknown implementation".to_string(),
        )),
    }
}

/// Test edge case robustness comprehensively
pub fn test_edge_case_robustness(implementation: &str) -> SignalResult<EdgeCaseRobustnessResults> {
    let mut results = vec![false; 6];

    // Test 1: Empty signal
    results[0] = test_empty_signal(implementation);

    // Test 2: Single point
    results[1] = test_single_point(implementation);

    // Test 3: Constant signal
    results[2] = test_constant_signal(implementation);

    // Test 4: Invalid values (NaN/Inf)
    results[3] = test_invalid_values(implementation);

    // Test 5: Duplicate time points
    results[4] = test_duplicate_times(implementation);

    // Test 6: Non-monotonic time series
    results[5] = test_non_monotonic_times(implementation);

    let overall_robustness = results
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .sum::<f64>()
        / results.len() as f64;

    Ok(EdgeCaseRobustnessResults {
        empty_signal_handling: results[0],
        single_point_handling: results[1],
        constant_signal_handling: results[2],
        invalid_value_handling: results[3],
        duplicate_time_handling: results[4],
        non_monotonic_handling: results[5],
        overall_robustness,
        extreme_frequency_handling: 0.0,
        numerical_edge_cases: 0.0,
    })
}

/// Test empty signal handling
pub fn test_empty_signal(implementation: &str) -> bool {
    let t: Vec<f64> = vec![];
    let signal: Vec<f64> = vec![];

    match run_lombscargle(implementation, &t, &signal) {
        Ok(_) => false, // Should not succeed
        Err(_) => true, // Should gracefully fail
    }
}

/// Test single point handling
pub fn test_single_point(implementation: &str) -> bool {
    let t = vec![1.0];
    let signal = vec![0.5];

    match run_lombscargle(implementation, &t, &signal) {
        Ok(_) => false, // Should not succeed with single point
        Err(_) => true, // Should gracefully fail
    }
}

/// Test constant signal handling
pub fn test_constant_signal(implementation: &str) -> bool {
    let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let signal = vec![1.0; 50];

    match run_lombscargle(implementation, &t, &signal) {
        Ok((_, power)) => {
            // Should handle constant signal gracefully (low power at all frequencies)
            let max_power = power.iter().cloned().fold(0.0, f64::max);
            max_power < 1.0 // Reasonable for constant signal
        }
        Err(_) => false, // Should not fail completely
    }
}

/// Test invalid value handling
pub fn test_invalid_values(implementation: &str) -> bool {
    let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let signal = vec![
        1.0,
        f64::NAN,
        2.0,
        f64::INFINITY,
        0.5,
        -1.0,
        3.0,
        f64::NEG_INFINITY,
        1.5,
        0.0,
    ];

    match run_lombscargle(implementation, &t, &signal) {
        Ok(_) => false, // Should detect and reject invalid values
        Err(_) => true, // Should gracefully fail
    }
}

/// Test duplicate time point handling
pub fn test_duplicate_times(implementation: &str) -> bool {
    let t = vec![0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.4]; // Duplicates
    let signal = vec![1.0, 0.5, -0.5, 2.0, 1.5, -1.0, 0.0];

    match run_lombscargle(implementation, &t, &signal) {
        Ok(_) => false, // Should detect duplicate times
        Err(_) => true, // Should gracefully fail
    }
}

/// Test non-monotonic time handling
pub fn test_non_monotonic_times(implementation: &str) -> bool {
    let t = vec![0.0, 0.2, 0.1, 0.4, 0.3, 0.5]; // Non-monotonic
    let signal = vec![1.0, 0.5, -0.5, 2.0, 1.5, -1.0];

    match run_lombscargle(implementation, &t, &signal) {
        Ok(_) => false, // Should detect non-monotonic times
        Err(_) => true, // Should gracefully fail
    }
}

/// Comprehensive edge case validation
pub fn validate_edge_cases_comprehensive() -> SignalResult<EdgeCaseValidationResult> {
    let mut result = EdgeCaseValidationResult::default();
    let mut tests_passed = 0;
    let mut total_tests = 0;

    // Test 1: Empty signal
    total_tests += 1;
    if lombscargle(&[], &[], None, None, None, None, None, None).is_err() {
        tests_passed += 1;
        result.empty_signal_handled = true;
    }

    // Test 2: Single data point
    total_tests += 1;
    let single_time = vec![1.0];
    let single_data = vec![1.0];
    if lombscargle(
        &single_time,
        &single_data,
        None,
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        None,
    )
    .is_err()
    {
        tests_passed += 1;
        result.single_point_handled = true;
    }

    // Test 3: Two identical time points (should fail)
    total_tests += 1;
    let duplicate_time = vec![1.0, 1.0, 2.0];
    let duplicate_data = vec![1.0, 2.0, 3.0];
    if lombscargle(
        &duplicate_time,
        &duplicate_data,
        None,
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        None,
    )
    .is_err()
    {
        tests_passed += 1;
        result.duplicate_times_handled = true;
    }

    // Test 4: Extremely large values
    total_tests += 1;
    let large_time: Vec<f64> = (0..100).map(|i| i as f64 * 1e12).collect();
    let large_data: Vec<f64> = (0..100).map(|i| (i as f64 * 1e9).sin()).collect();
    match lombscargle(&large_time, &large_data, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            if freqs.iter().all(|&f: &f64| f.is_finite())
                && power.iter().all(|&p: &f64| p.is_finite())
            {
                tests_passed += 1;
                result.large_values_stable = true;
            }
        }
        Err(_) => {} // Acceptable to fail with extreme values
    }

    // Test 5: Extremely small values
    total_tests += 1;
    let small_time: Vec<f64> = (0..100).map(|i| i as f64 * 1e-12).collect();
    let small_data: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 1e-15).collect();
    match lombscargle(&small_time, &small_data, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            if freqs.iter().all(|&f: &f64| f.is_finite())
                && power.iter().all(|&p: &f64| p.is_finite())
            {
                tests_passed += 1;
                result.small_values_stable = true;
            }
        }
        Err(_) => {} // Acceptable to fail with extreme values
    }

    // Test 6: NaN/Inf in input (should be caught)
    total_tests += 1;
    let nan_time = vec![1.0, 2.0, f64::NAN, 4.0];
    let nan_data = vec![1.0, 2.0, 3.0, 4.0];
    if lombscargle(&nan_time, &nan_data, None, None, None, None, None, None).is_err() {
        tests_passed += 1;
        result.nan_input_handled = true;
    }

    // Test 7: Constant signal
    total_tests += 1;
    let const_time: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let const_data = vec![5.0; 100];
    match lombscargle(&const_time, &const_data, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            // For constant signal, power should be near zero at all non-zero frequencies
            let non_zero_power_count = power.iter().skip(1).filter(|&&p| p > 1e-10).count();
            if non_zero_power_count < power.len() / 10 {
                tests_passed += 1;
                result.constant_signal_correct = true;
            }
        }
        Err(_) => {}
    }

    // Test 8: Very irregular sampling
    total_tests += 1;
    let irregular_time = vec![0.0, 0.1, 1.0, 1.01, 10.0, 15.0, 15.001, 20.0];
    let irregular_data: Vec<f64> = irregular_time
        .iter()
        .map(|&t| (2.0 * PI * t).sin())
        .collect();
    match lombscargle(
        &irregular_time,
        &irregular_data,
        None,
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        None,
    ) {
        Ok((freqs, power)) => {
            if freqs.iter().all(|&f: &f64| f.is_finite())
                && power.iter().all(|&p: &f64| p.is_finite())
            {
                tests_passed += 1;
                result.irregular_sampling_stable = true;
            }
        }
        Err(_) => {}
    }

    result.tests_passed = tests_passed;
    result.total_tests = total_tests;
    result.success_rate = tests_passed as f64 / total_tests as f64;

    Ok(result)
}

/// Validate numerical robustness with challenging conditions
pub fn validate_numerical_robustness_extreme() -> SignalResult<NumericalRobustnessResult> {
    let mut result = NumericalRobustnessResult::default();
    let mut robustness_scores = Vec::new();

    // Test 1: Close frequency resolution
    let fs = 100.0;
    let n = 1000;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Two very close frequencies
    let f1 = 10.0;
    let f2 = 10.05; // Very close
    let signal: Vec<f64> = t.iter()
        .map(|&ti| (2.0 * PI * f1 * ti).sin() + 0.8 * (2.0 * PI * f2 * ti).sin())
        .collect();

    match lombscargle(&t, &signal, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            // Check if both peaks are detected
            let mut peaks_found = 0;
            for (i, &p) in power.iter().enumerate() {
                if i > 0 && i < power.len() - 1 {
                    if p > power[i-1] && p > power[i+1] && p > 0.1 {
                        let freq = freqs[i];
                        if (freq - f1).abs() < 0.5 || (freq - f2).abs() < 0.5 {
                            peaks_found += 1;
                        }
                    }
                }
            }
            result.close_frequency_resolved = peaks_found >= 2;
            robustness_scores.push(if result.close_frequency_resolved { 1.0 } else { 0.0 });
        }
        Err(_) => {
            robustness_scores.push(0.0);
        }
    }

    // Test 2: High dynamic range signal
    let high_signal: Vec<f64> = t.iter()
        .map(|&ti| 1000.0 * (2.0 * PI * 5.0 * ti).sin() + 0.001 * (2.0 * PI * 25.0 * ti).sin())
        .collect();

    match lombscargle(&t, &high_signal, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            let max_power = power.iter().cloned().fold(0.0, f64::max);
            let min_power = power.iter().cloned().fold(f64::INFINITY, f64::min);
            let dynamic_range = if min_power > 0.0 { max_power / min_power } else { f64::INFINITY };
            result.high_dynamic_range_stable = dynamic_range.is_finite() && dynamic_range > 1000.0;
            robustness_scores.push(if result.high_dynamic_range_stable { 1.0 } else { 0.0 });
        }
        Err(_) => {
            robustness_scores.push(0.0);
        }
    }

    // Test 3: Noisy signal stability
    let mut rng = scirs2_core::random::thread_rng();
    let noise_level = 0.5;
    let noisy_signal: Vec<f64> = t.iter()
        .map(|&ti| {
            use scirs2_core::random::Rng;
            (2.0 * PI * 10.0 * ti).sin() + noise_level * (rng.gen::<f64>() - 0.5)
        })
        .collect();

    match lombscargle(&t, &noisy_signal, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            // Check if main frequency is still detectable
            let peak_freq_idx = power.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let detected_freq = freqs[peak_freq_idx];
            result.noisy_signal_stable = (detected_freq - 10.0).abs() < 2.0;
            robustness_scores.push(if result.noisy_signal_stable { 1.0 } else { 0.0 });
        }
        Err(_) => {
            robustness_scores.push(0.0);
        }
    }

    // Test 4: Extreme frequencies
    let extreme_signal: Vec<f64> = t.iter()
        .map(|&ti| (2.0 * PI * 0.01 * ti).sin() + (2.0 * PI * 49.9 * ti).sin())
        .collect();

    match lombscargle(&t, &extreme_signal, None, None, None, None, None, None) {
        Ok((freqs, power)) => {
            result.extreme_frequencies_stable = power.iter().all(|&p| p.is_finite());
            robustness_scores.push(if result.extreme_frequencies_stable { 1.0 } else { 0.0 });
        }
        Err(_) => {
            robustness_scores.push(0.0);
        }
    }

    // Calculate overall robustness score
    result.overall_robustness_score = if robustness_scores.is_empty() {
        0.0
    } else {
        robustness_scores.iter().sum::<f64>() / robustness_scores.len() as f64
    };

    Ok(result)
}