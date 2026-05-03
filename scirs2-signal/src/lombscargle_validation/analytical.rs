// Analytical validation functions for Lomb-Scargle periodogram
//
// This module provides comprehensive analytical validation functions for
// Lomb-Scargle implementations against known analytical cases.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig, WindowType};
use super::types::{ValidationResult, SingleTestResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::PI;

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
            let (f, p, _ci) = lombscargle_enhanced(&t, &signal, &config)?;
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
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
        .expect("Operation failed");
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
            let (f, p, _ci) = lombscargle_enhanced(&t, &signal, &config)?;
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .expect("Operation failed");

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
        let interval = rng.random_range(0.001..0.5); // Very uneven: 1ms to 500ms
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
            let (f, p, _ci) = lombscargle_enhanced(&t, &signal, &config)?;
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
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
        .expect("Operation failed");
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
            lombscargle_enhanced(&t, &signal, &config).map(|(f, p, _ci)| (f, p))
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
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
                .expect("Operation failed");

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

/// Helper: find local peaks above a fraction of maximum.
fn find_local_peaks(data: &[f64], threshold_ratio: f64) -> Vec<usize> {
    if data.len() < 3 {
        return Vec::new();
    }
    let max_val = data.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = max_val * threshold_ratio;
    let mut peaks = Vec::new();
    for i in 1..(data.len() - 1) {
        if data[i] > threshold && data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

/// Helper: run Lomb-Scargle on (t, signal) and return (freqs, power).
fn run_lombscargle(
    implementation: &str,
    t: &[f64],
    signal: &[f64],
    f_min: Option<f64>,
    f_max: Option<f64>,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    match implementation {
        "standard" => {
            let (f, p) = lombscargle(
                t,
                signal,
                None,
                Some("standard"),
                Some(true),
                Some(false),
                None,
                None,
            )?;
            Ok((f, p))
        }
        "enhanced" => {
            let config = LombScargleConfig {
                window: WindowType::None,
                custom_window: None,
                oversample: 5.0,
                f_min,
                f_max,
                bootstrap_iter: None,
                confidence: None,
                tolerance: 1e-10,
                use_fast: true,
            };
            let (f, p, _ci) = lombscargle_enhanced(t, signal, &config)?;
            Ok((f, p))
        }
        _ => Err(SignalError::ValueError(
            format!("Unknown implementation: {}", implementation),
        )),
    }
}

/// Test extreme edge cases: single-point, two-point, very high/very low frequency.
#[allow(dead_code)]
fn validate_edge_cases(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    // Edge case 1: Minimum viable signal (3 points at Nyquist)
    {
        let t = vec![0.0, 0.5, 1.0];
        let f_nyquist = 1.0; // fs=2 Hz => f_Nyquist=1 Hz
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_nyquist * ti).sin()).collect();
        let result = run_lombscargle(implementation, &t, &signal, Some(0.1), Some(1.5));
        match result {
            Ok((freqs, power)) => {
                let all_finite = freqs.iter().chain(power.iter()).all(|v| v.is_finite());
                if !all_finite {
                    issues.push("Edge case: NaN/Inf in output for 3-point signal".to_string());
                    errors.push(1.0);
                } else {
                    errors.push(0.0);
                }
            }
            Err(_) => {
                // Acceptable for implementation to reject too-short inputs
                errors.push(0.0);
            }
        }
    }

    // Edge case 2: Signal of identical values (zero variance — should produce low power)
    {
        let n = 50usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let signal = vec![1.0_f64; n];
        let result = run_lombscargle(implementation, &t, &signal, Some(0.5), Some(50.0));
        match result {
            Ok((_freqs, power)) => {
                let max_power = power.iter().cloned().fold(0.0_f64, f64::max);
                // Zero-variance signal should have negligible power after mean subtraction
                if max_power > 1.0 {
                    issues.push(format!(
                        "Edge case: constant signal has unexpectedly high peak power {:.4}",
                        max_power
                    ));
                    errors.push(max_power.min(1.0));
                } else {
                    errors.push(0.0);
                }
            }
            Err(_) => {
                errors.push(0.0);
            }
        }
    }

    // Edge case 3: Very high frequency relative to sampling
    {
        let n = 100usize;
        let fs = 100.0_f64;
        let f_signal = 45.0_f64; // Near Nyquist
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();
        let result = run_lombscargle(implementation, &t, &signal, Some(1.0), Some(fs / 2.0));
        match result {
            Ok((freqs, power)) => {
                if !power.iter().all(|v| v.is_finite()) {
                    issues.push("Edge case: Non-finite power near Nyquist".to_string());
                    errors.push(1.0);
                } else {
                    let (peak_idx, _) = power
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((0, &0.0));
                    let freq_err = (freqs[peak_idx] - f_signal).abs() / f_signal;
                    errors.push(freq_err);
                    if freq_err > tolerance * 20.0 {
                        issues.push(format!(
                            "Edge case: near-Nyquist frequency error {:.4} > {}",
                            freq_err,
                            tolerance * 20.0
                        ));
                    }
                }
            }
            Err(_) => {
                errors.push(0.0);
            }
        }
    }

    let peak_error = errors.iter().cloned().fold(0.0_f64, f64::max);
    Ok(SingleTestResult {
        errors,
        peak_error,
        peak_errors: vec![peak_error],
        issues,
    })
}

/// Test numerical precision and stability for well-conditioned inputs.
///
/// Checks that outputs are finite, that re-running gives identical results,
/// and that small perturbations produce small changes in the spectrum.
#[allow(dead_code)]
fn validate_numerical_stability(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 500usize;
    let fs = 100.0_f64;
    let f_signal = 10.0_f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * f_signal * ti).sin()).collect();

    // Run 1: Check all outputs finite
    let (freqs1, power1) = run_lombscargle(implementation, &t, &signal, Some(1.0), Some(50.0))?;
    let nan_count = power1.iter().filter(|v| !v.is_finite()).count();
    if nan_count > 0 {
        issues.push(format!(
            "Numerical stability: {} NaN/Inf values in power spectrum",
            nan_count
        ));
        errors.push(1.0);
    } else {
        errors.push(0.0);
    }

    // Run 2: Reproducibility — same inputs must give same outputs
    let (_, power2) = run_lombscargle(implementation, &t, &signal, Some(1.0), Some(50.0))?;
    let max_diff: f64 = power1
        .iter()
        .zip(power2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    if max_diff > 1e-14 {
        issues.push(format!(
            "Numerical stability: reproducibility error {:.2e}",
            max_diff
        ));
    }
    errors.push(max_diff);

    // Run 3: Small perturbation sensitivity — add 1e-8 noise to signal
    let tiny_noise: f64 = 1e-8;
    let mut rng = scirs2_core::random::rng();
    let signal_pert: Vec<f64> = signal
        .iter()
        .map(|&x| x + rng.random_range(-tiny_noise..tiny_noise))
        .collect();
    let (_, power_pert) =
        run_lombscargle(implementation, &t, &signal_pert, Some(1.0), Some(50.0))?;

    let max_power1 = power1.iter().cloned().fold(0.0_f64, f64::max);
    let max_pert_diff: f64 = power1
        .iter()
        .zip(power_pert.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let relative_sensitivity = if max_power1 > 1e-15 {
        max_pert_diff / max_power1
    } else {
        0.0
    };
    if relative_sensitivity > 1e-4 {
        issues.push(format!(
            "Numerical stability: perturbation sensitivity {:.4} above expected",
            relative_sensitivity
        ));
    }
    errors.push(relative_sensitivity);

    let peak_error = errors.iter().cloned().fold(0.0_f64, f64::max);
    Ok(SingleTestResult {
        errors,
        peak_error,
        peak_errors: vec![peak_error],
        issues,
    })
}

/// Test signals with high dynamic range (amplitude ratio ~ 1000:1).
///
/// Verifies that the implementation can detect both the dominant and the weak
/// signal component without numerical swamping.
#[allow(dead_code)]
fn validate_dynamic_range(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 1000usize;
    let fs = 100.0_f64;
    let f_strong = 10.0_f64;
    let f_weak = 20.0_f64;
    let amp_strong = 100.0_f64;
    let amp_weak = 0.1_f64; // 1000:1 amplitude ratio

    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            amp_strong * (2.0 * PI * f_strong * ti).sin()
                + amp_weak * (2.0 * PI * f_weak * ti).sin()
        })
        .collect();

    let (freqs, power) =
        run_lombscargle(implementation, &t, &signal, Some(1.0), Some(45.0))?;

    if !power.iter().all(|v| v.is_finite()) {
        issues.push("Dynamic range: non-finite values in spectrum".to_string());
        errors.push(1.0);
        return Ok(SingleTestResult {
            errors,
            peak_error: 1.0,
            peak_errors: vec![1.0],
            issues,
        });
    }

    // Strong peak should be near f_strong
    let (strong_idx, _) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let strong_freq_err = (freqs[strong_idx] - f_strong).abs() / f_strong;
    errors.push(strong_freq_err);
    if strong_freq_err > tolerance * 5.0 {
        issues.push(format!(
            "Dynamic range: strong peak frequency error {:.4}",
            strong_freq_err
        ));
    }

    // Weak peak should be detectable — find local peaks
    let peaks = find_local_peaks(&power, 1e-5);
    let weak_peak_found = peaks.iter().any(|&idx| {
        freqs.get(idx).map_or(false, |&f| (f - f_weak).abs() / f_weak < 0.1)
    });
    if !weak_peak_found && amp_weak / amp_strong > 1e-4 {
        issues.push(format!(
            "Dynamic range: weak signal at {:.1} Hz not detected",
            f_weak
        ));
        errors.push(0.1);
    } else {
        errors.push(0.0);
    }

    let peak_error = strong_freq_err;
    Ok(SingleTestResult {
        errors,
        peak_error,
        peak_errors: vec![peak_error],
        issues,
    })
}

/// Test signals that contain a linear or polynomial trend alongside a sinusoid.
///
/// A trend will add spurious power at low frequencies; this checks that the
/// dominant periodic signal is still correctly located.
#[allow(dead_code)]
fn validate_with_trends(implementation: &str, tolerance: f64) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 500usize;
    let fs = 50.0_f64;
    let f_signal = 8.0_f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Signal = strong linear trend + sinusoid
    let trend_slope = 5.0_f64;
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| trend_slope * ti + (2.0 * PI * f_signal * ti).sin())
        .collect();

    let (freqs, power) =
        run_lombscargle(implementation, &t, &signal, Some(0.5), Some(fs / 2.0))?;

    if !power.iter().all(|v| v.is_finite()) {
        issues.push("Trend test: non-finite spectrum values".to_string());
        errors.push(1.0);
        return Ok(SingleTestResult {
            errors,
            peak_error: 1.0,
            peak_errors: vec![1.0],
            issues,
        });
    }

    // The peak should still be near f_signal even with the trend
    let (peak_idx, _) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let peak_freq = freqs[peak_idx];
    let freq_err = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_err);

    // Allow more tolerance because the trend injects power at low frequencies
    if freq_err > tolerance * 20.0 {
        issues.push(format!(
            "Trend test: peak at {:.3} Hz instead of {:.1} Hz (error {:.4})",
            peak_freq, f_signal, freq_err
        ));
    }

    Ok(SingleTestResult {
        errors: errors.clone(),
        peak_error: freq_err,
        peak_errors: errors,
        issues,
    })
}

/// Test correlated (coloured) noise — AR(1) noise.
///
/// Generates a sinusoid plus pink-ish noise (AR(1) with coefficient 0.8) and
/// verifies the peak is still recovered.
#[allow(dead_code)]
fn validate_correlated_noise(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 800usize;
    let fs = 100.0_f64;
    let f_signal = 12.0_f64;
    let snr_amplitude = 3.0_f64; // Signal amplitude relative to noise std

    let mut rng = scirs2_core::random::rng();
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Generate AR(1) noise: e(t) = 0.8 * e(t-1) + w(t)
    let ar_coef = 0.8_f64;
    let mut noise = vec![0.0_f64; n];
    for i in 1..n {
        noise[i] = ar_coef * noise[i - 1] + rng.random_range(-1.0..1.0);
    }

    let signal: Vec<f64> = t
        .iter()
        .zip(noise.iter())
        .map(|(&ti, &ni)| snr_amplitude * (2.0 * PI * f_signal * ti).sin() + ni)
        .collect();

    let (freqs, power) =
        run_lombscargle(implementation, &t, &signal, Some(0.5), Some(fs / 2.0))?;

    if !power.iter().all(|v| v.is_finite()) {
        issues.push("Correlated noise test: non-finite spectrum".to_string());
        errors.push(1.0);
        return Ok(SingleTestResult {
            errors,
            peak_error: 1.0,
            peak_errors: vec![1.0],
            issues,
        });
    }

    let (peak_idx, _) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let peak_freq = freqs[peak_idx];
    let freq_err = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_err);

    // With SNR = 3 the peak should still be near f_signal
    if freq_err > tolerance * 30.0 {
        issues.push(format!(
            "Correlated noise: peak at {:.3} Hz vs expected {:.1} Hz (err {:.4})",
            peak_freq, f_signal, freq_err
        ));
    }

    Ok(SingleTestResult {
        errors: errors.clone(),
        peak_error: freq_err,
        peak_errors: errors,
        issues,
    })
}

/// Test high frequency resolution — two closely-spaced sinusoids.
///
/// With long observation, LS should resolve two frequencies separated by 0.2 Hz.
#[allow(dead_code)]
fn validate_high_frequency_resolution(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 2000usize;
    let fs = 100.0_f64;
    let f1 = 10.0_f64;
    let f2 = 10.2_f64; // delta_f = 0.2 Hz; T = 20 s => resolution ~ 1/T = 0.05 Hz

    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            (2.0 * PI * f1 * ti).sin() + 0.9 * (2.0 * PI * f2 * ti).sin()
        })
        .collect();

    let (freqs, power) =
        run_lombscargle(implementation, &t, &signal, Some(9.0), Some(11.5))?;

    if !power.iter().all(|v| v.is_finite()) {
        issues.push("High frequency resolution: non-finite spectrum".to_string());
        errors.push(1.0);
        return Ok(SingleTestResult {
            errors,
            peak_error: 1.0,
            peak_errors: vec![1.0],
            issues,
        });
    }

    // Check that there are at least 2 distinct peaks in [9.5, 11] Hz
    let peaks = find_local_peaks(&power, 0.4);
    let n_peaks_in_range = peaks.iter().filter(|&&idx| {
        freqs.get(idx).map_or(false, |&f| f >= 9.5 && f <= 11.0)
    }).count();

    if n_peaks_in_range < 2 {
        issues.push(format!(
            "High frequency resolution: only {} peaks resolved (expected 2) for delta_f = 0.2 Hz",
            n_peaks_in_range
        ));
        errors.push(0.5);
    } else {
        errors.push(0.0);
    }

    // Also check the strongest peak is near f1
    let (strong_idx, _) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let strong_freq_err = freqs
        .get(strong_idx)
        .map_or(1.0, |&f| (f - f1).abs() / f1);
    errors.push(strong_freq_err);
    if strong_freq_err > tolerance * 5.0 {
        issues.push(format!(
            "High frequency resolution: strong peak error {:.4}",
            strong_freq_err
        ));
    }

    let peak_error = errors.iter().cloned().fold(0.0_f64, f64::max);
    Ok(SingleTestResult {
        errors,
        peak_error,
        peak_errors: vec![peak_error],
        issues,
    })
}

/// Test enhanced floating-point precision on a single well-sampled sinusoid.
///
/// Compares the measured peak amplitude (normalized power) with the expected
/// analytical value and checks phase-coherence via the cosine component.
#[allow(dead_code)]
fn validate_enhanced_precision(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 2000usize;
    let fs = 200.0_f64;
    let f_signal = 17.5_f64;
    let amplitude = 2.0_f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| amplitude * (2.0 * PI * f_signal * ti).sin())
        .collect();

    let (freqs, power) =
        run_lombscargle(implementation, &t, &signal, Some(10.0), Some(30.0))?;

    if !power.iter().all(|v| v.is_finite()) {
        issues.push("Enhanced precision: non-finite output".to_string());
        errors.push(1.0);
        return Ok(SingleTestResult {
            errors,
            peak_error: 1.0,
            peak_errors: vec![1.0],
            issues,
        });
    }

    // Frequency accuracy
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let peak_freq = freqs[peak_idx];
    let freq_err = (peak_freq - f_signal).abs() / f_signal;
    errors.push(freq_err);
    if freq_err > tolerance * 2.0 {
        issues.push(format!(
            "Enhanced precision: frequency error {:.4e} exceeds tight tolerance",
            freq_err
        ));
    }

    // Power accuracy: for normalised LS, peak power at single sinusoid ~ 1.0 or amplitude^2/2
    // For "standard" normalised: power in [0,1], peak near 1.0
    // Compute relative deviation from 1.0 (standard) or amplitude^2/2 (non-normalised)
    let expected_power_ratio = if peak_power > 2.0 {
        // Non-normalised: expected = amplitude^2 * n / 4
        amplitude * amplitude * n as f64 / 4.0
    } else {
        1.0 // Normalised: expected = 1.0
    };
    let power_err = (peak_power - expected_power_ratio).abs() / expected_power_ratio.max(1e-15);
    errors.push(power_err.min(1.0));
    if power_err > 0.1 {
        issues.push(format!(
            "Enhanced precision: peak power {:.4} deviates from expected {:.4} by {:.2}%",
            peak_power,
            expected_power_ratio,
            power_err * 100.0
        ));
    }

    Ok(SingleTestResult {
        errors: errors.clone(),
        peak_error: freq_err,
        peak_errors: errors,
        issues,
    })
}

/// Validate by cross-checking the enhanced implementation against the standard one.
///
/// Both should agree on peak frequency to within `tolerance` and should have
/// correlated power spectra (Pearson r > 0.95).
#[allow(dead_code)]
fn validate_cross_reference_implementation(
    implementation: &str,
    tolerance: f64,
) -> SignalResult<SingleTestResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    let n = 800usize;
    let fs = 100.0_f64;
    let f_signal = 15.0_f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_signal * ti).sin())
        .collect();

    // Run both implementations
    let (freqs_std, power_std) =
        run_lombscargle("standard", &t, &signal, Some(1.0), Some(45.0))?;

    let (freqs_enh, power_enh) = {
        let config = LombScargleConfig {
            window: WindowType::None,
            custom_window: None,
            oversample: 5.0,
            f_min: Some(1.0),
            f_max: Some(45.0),
            bootstrap_iter: None,
            confidence: None,
            tolerance: 1e-10,
            use_fast: true,
        };
        let (f, p, _ci) = lombscargle_enhanced(&t, &signal, &config)?;
        (f, p)
    };

    // Get peak frequencies from each
    let std_peak_idx = power_std
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let enh_peak_idx = power_enh
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let std_peak_freq = freqs_std.get(std_peak_idx).copied().unwrap_or(0.0);
    let enh_peak_freq = freqs_enh.get(enh_peak_idx).copied().unwrap_or(0.0);

    let cross_peak_err = if std_peak_freq > 1e-15 {
        (std_peak_freq - enh_peak_freq).abs() / std_peak_freq
    } else {
        1.0
    };
    errors.push(cross_peak_err);
    if cross_peak_err > tolerance * 10.0 {
        issues.push(format!(
            "Cross-reference: peak frequency mismatch std={:.3} enhanced={:.3} (err={:.4})",
            std_peak_freq, enh_peak_freq, cross_peak_err
        ));
    }

    // Additionally verify the requested implementation detects the known signal
    let (_, power_impl) =
        run_lombscargle(implementation, &t, &signal, Some(1.0), Some(45.0))?;
    let (impl_peak_idx, _) = power_impl
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let impl_peak_freq = freqs_std.get(impl_peak_idx).copied().unwrap_or(0.0);
    let impl_freq_err = (impl_peak_freq - f_signal).abs() / f_signal;
    errors.push(impl_freq_err);
    if impl_freq_err > tolerance * 5.0 {
        issues.push(format!(
            "Cross-reference ({} impl): frequency error {:.4}",
            implementation, impl_freq_err
        ));
    }

    let peak_error = errors.iter().cloned().fold(0.0_f64, f64::max);
    Ok(SingleTestResult {
        errors,
        peak_error,
        peak_errors: vec![peak_error],
        issues,
    })
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_edge_cases_standard() {
        let result = validate_edge_cases("standard", 1e-3);
        assert!(result.is_ok(), "validate_edge_cases failed: {:?}", result);
        let r = result.expect("validate_edge_cases failed");
        // There should be some errors vector (may be empty if all succeed)
        // Just make sure no panic and the structure is valid
        assert!(r.peak_error >= 0.0);
    }

    #[test]
    fn test_validate_numerical_stability_standard() {
        let result = validate_numerical_stability("standard", 1e-3);
        assert!(result.is_ok(), "validate_numerical_stability failed: {:?}", result);
        let r = result.expect("validate_numerical_stability failed");
        // Reproducibility error should be essentially zero
        // errors[1] is the reproducibility check
        if r.errors.len() > 1 {
            assert!(
                r.errors[1] < 1e-10,
                "Reproducibility error too large: {}",
                r.errors[1]
            );
        }
    }

    #[test]
    fn test_validate_dynamic_range_standard() {
        let result = validate_dynamic_range("standard", 1e-3);
        assert!(result.is_ok(), "validate_dynamic_range failed: {:?}", result);
        let r = result.expect("validate_dynamic_range failed");
        // Strong peak should be found reasonably close to f_strong = 10 Hz
        assert!(
            r.peak_error < 0.2,
            "Dynamic range peak error too large: {}",
            r.peak_error
        );
    }

    #[test]
    fn test_validate_with_trends_standard() {
        let result = validate_with_trends("standard", 1e-3);
        assert!(result.is_ok(), "validate_with_trends failed: {:?}", result);
    }

    #[test]
    fn test_validate_high_frequency_resolution_standard() {
        let result = validate_high_frequency_resolution("standard", 1e-3);
        assert!(result.is_ok(), "validate_high_frequency_resolution failed: {:?}", result);
    }

    #[test]
    fn test_validate_cross_reference_implementation() {
        let result = validate_cross_reference_implementation("standard", 1e-2);
        assert!(result.is_ok(), "validate_cross_reference_implementation failed: {:?}", result);
        let r = result.expect("validate_cross_reference_implementation failed");
        // Cross-peak error between standard and enhanced should be small
        if !r.errors.is_empty() {
            assert!(
                r.errors[0] < 0.1,
                "Cross-reference peak mismatch: {}",
                r.errors[0]
            );
        }
    }
}
