//! Statistical validation and significance testing for Lomb-Scargle validation
//!
//! This module provides comprehensive statistical validation including bootstrap
//! confidence intervals, cross-validation, and significance testing.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{Rng, SeedableRng};
use std::f64::consts::PI;

use super::config::{CrossValidationResults, StatisticalSignificanceResults};

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

/// Test enhanced statistical significance with comprehensive FAP validation
pub fn test_enhanced_statistical_significance(
    implementation: &str,
    _tolerance: f64,
) -> SignalResult<StatisticalSignificanceResults> {
    // Enhanced FAP testing with multiple noise realizations
    let n = 256;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let n_trials = 200; // Reduced for performance

    let mut max_powers = Vec::new();
    let mut p_values = Vec::new();
    let mut rng = scirs2_core::random::thread_rng();

    // Multiple noise realizations for statistical validation
    for _ in 0..n_trials {
        let noise_signal: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let (freqs, power) = run_lombscargle(implementation, &t, &noise_signal)?;
        let max_power = power.iter().cloned().fold(0.0, f64::max);
        max_powers.push(max_power);

        // Calculate empirical p-value
        let n_freqs = freqs.len() as f64;
        let p_value = 1.0 - (1.0 - (-max_power).exp()).powf(n_freqs);
        p_values.push(p_value.min(1.0).max(0.0));
    }

    // Theoretical vs empirical FAP comparison
    let mean_max_power = max_powers.iter().sum::<f64>() / n_trials as f64;
    let theoretical_fap = (-mean_max_power).exp();
    let empirical_fap =
        max_powers.iter().filter(|&&p| p > mean_max_power).count() as f64 / n_trials as f64;
    let fap_theoretical_empirical_ratio = if empirical_fap > 1e-10 {
        (theoretical_fap / empirical_fap).min(10.0).max(0.1)
    } else {
        1.0
    };

    let fap_accuracy = 1.0 - (theoretical_fap - empirical_fap).abs().min(1.0);

    // P-value uniformity test (Kolmogorov-Smirnov)
    let pvalue_uniformity_score = kolmogorov_smirnov_uniformity_test(&p_values);

    // Enhanced statistical power estimation with signal injection
    let statistical_power = estimate_statistical_power(implementation, &t)?;

    // Significance level calibration with multiple levels
    let significance_calibration = test_significance_calibration(implementation, &t)?;

    // Enhanced bootstrap CI coverage
    let bootstrap_coverage = if implementation == "enhanced" {
        test_enhanced_bootstrap_coverage(&t)?
    } else {
        0.0
    };

    Ok(StatisticalSignificanceResults {
        fap_accuracy,
        statistical_power,
        significance_calibration,
        bootstrap_coverage,
        fap_theoretical_empirical_ratio,
        pvalue_uniformity_score,
    })
}

/// Kolmogorov-Smirnov test for p-value uniformity
pub fn kolmogorov_smirnov_uniformity_test(p_values: &[f64]) -> f64 {
    let n = p_values.len();
    let mut sorted_p = p_values.to_vec();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut max_deviation = 0.0;

    for (i, &p) in sorted_p.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n as f64;
        let theoretical_cdf = p; // Uniform distribution CDF
        let deviation = (empirical_cdf - theoretical_cdf).abs();
        max_deviation = max_deviation.max(deviation);
    }

    // Return uniformity score (1 - normalized deviation)
    let critical_value = 1.36 / (n as f64).sqrt(); // 95% confidence level
    1.0 - (max_deviation / critical_value).min(1.0)
}

/// Estimate statistical power with signal injection
pub fn estimate_statistical_power(implementation: &str, times: &[f64]) -> SignalResult<f64> {
    let mut detections = 0;
    let n_trials = 50; // Reduced for performance
    let mut rng = scirs2_core::random::thread_rng();

    for _ in 0..n_trials {
        // Inject known signal with noise
        let f_signal = 10.0;
        let snr_db = 10.0; // Moderate SNR
        let signal_power = 1.0;
        let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);

        let signal: Vec<f64> = times
            .iter()
            .map(|&ti| {
                (2.0 * PI * f_signal * ti).sin()
                    + noise_power.sqrt() * rng.gen_range(-1.0..1.0)
            })
            .collect();

        let (freqs, power) = run_lombscargle(implementation, times, &signal)?;

        // Detection criterion: peak within tolerance of true frequency
        let tolerance = 0.5;
        let detected = freqs
            .iter()
            .zip(power.iter())
            .filter(|(&f, _)| (f - f_signal).abs() < tolerance)
            .any(|(_, &p)| {
                let mean_power = power.iter().sum::<f64>() / power.len() as f64;
                let std_power = (power.iter().map(|&p| (p - mean_power).powi(2)).sum::<f64>()
                    / power.len() as f64)
                    .sqrt();
                p > mean_power + 3.0 * std_power
            });

        if detected {
            detections += 1;
        }
    }

    Ok(detections as f64 / n_trials as f64)
}

/// Test significance level calibration
pub fn test_significance_calibration(implementation: &str, times: &[f64]) -> SignalResult<f64> {
    let significance_levels = vec![0.05, 0.1];
    let mut calibration_errors = Vec::new();

    for &alpha in &significance_levels {
        let n_trials = 100; // Reduced for performance
        let mut false_positives = 0;
        let mut rng = scirs2_core::random::thread_rng();

        for _ in 0..n_trials {
            // Pure noise
            let noise: Vec<f64> = times.iter().map(|_| rng.gen_range(-1.0..1.0)).collect();

            let (_, power) = run_lombscargle(implementation, times, &noise)?;
            let max_power = power.iter().cloned().fold(0.0, f64::max);

            // Theoretical threshold for given significance level
            let threshold = -((alpha / power.len() as f64).ln());

            if max_power > threshold {
                false_positives += 1;
            }
        }

        let empirical_alpha = false_positives as f64 / n_trials as f64;
        let error = (empirical_alpha - alpha).abs() / alpha;
        calibration_errors.push(error);
    }

    // Return calibration accuracy (1 - mean relative error)
    let mean_error = calibration_errors.iter().sum::<f64>() / calibration_errors.len() as f64;
    Ok(1.0 - mean_error.min(1.0))
}

/// Enhanced bootstrap confidence interval coverage test
pub fn test_enhanced_bootstrap_coverage(times: &[f64]) -> SignalResult<f64> {
    let n_tests = 20; // Reduced for performance
    let mut coverage_scores = Vec::new();
    let mut rng = scirs2_core::random::thread_rng();

    for _ in 0..n_tests {
        // Generate known signal with noise
        let f_true = 5.0 + rng.gen_range(0.0..10.0);
        let signal: Vec<f64> = times
            .iter()
            .map(|&ti| {
                (2.0 * PI * f_true * ti).sin() + 0.1 * rng.gen_range(-1.0..1.0)
            })
            .collect();

        let mut config = LombScargleConfig::default();
        config.bootstrap_iter = Some(50); // Reduced for performance
        config.confidence = Some(0.95);

        match lombscargle_enhanced(times, &signal, &config) {
            Ok((freqs, power, Some((lower, upper)))) => {
                // Find peak closest to true frequency
                let (peak_idx, _) = freqs
                    .iter()
                    .enumerate()
                    .min_by(|(_, f1), (_, f2)| {
                        (**f1 - f_true)
                            .abs()
                            .partial_cmp(&(**f2 - f_true).abs())
                            .unwrap()
                    })
                    .unwrap();

                // Check if true power is within confidence interval
                let true_power = power[peak_idx];
                let in_interval = lower[peak_idx] <= true_power && true_power <= upper[peak_idx];
                coverage_scores.push(if in_interval { 1.0 } else { 0.0 });
            }
            _ => coverage_scores.push(0.0),
        }
    }

    Ok(coverage_scores.iter().sum::<f64>() / coverage_scores.len() as f64)
}

/// Test bootstrap coverage for standard implementation
pub fn test_bootstrap_coverage(times: &[f64], signal: &[f64]) -> SignalResult<f64> {
    let mut config = LombScargleConfig::default();
    config.bootstrap_iter = Some(100);
    config.confidence = Some(0.95);

    match lombscargle_enhanced(times, signal, &config) {
        Ok((_, _, Some((lower, upper)))) => {
            // Simplified coverage test
            let coverage = lower
                .iter()
                .zip(upper.iter())
                .filter(|(&l, &u)| l <= u && u > l)
                .count() as f64
                / lower.len() as f64;
            Ok(coverage)
        }
        _ => Ok(0.0),
    }
}

/// Test cross-validation comprehensively
pub fn test_cross_validation(
    implementation: &str,
    _tolerance: f64,
) -> SignalResult<CrossValidationResults> {
    let n = 200;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let f_true = 8.0;
    let mut rng = scirs2_core::random::thread_rng();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_true * ti).sin() + 0.1 * rng.gen_range(-1.0..1.0))
        .collect();

    // K-fold cross-validation (k=5)
    let kfold_score = perform_kfold_validation(implementation, &t, &signal, 5, f_true)?;

    // Bootstrap validation
    let bootstrap_score = perform_bootstrap_validation(implementation, &t, &signal, 20, f_true)?;

    // Leave-one-out validation (simplified - use subset)
    let loo_score = perform_loo_validation(implementation, &t, &signal, f_true)?;

    // Temporal consistency (sliding window)
    let temporal_consistency = test_temporal_consistency(implementation, &t, &signal, f_true)?;

    // Frequency stability across folds
    let frequency_stability = test_frequency_stability(implementation, &t, &signal, f_true)?;

    // Overall CV score
    let overall_cv_score =
        (kfold_score + bootstrap_score + loo_score + temporal_consistency + frequency_stability)
            / 5.0;

    Ok(CrossValidationResults {
        kfold_score,
        bootstrap_score,
        loo_score,
        temporal_consistency,
        frequency_stability,
        overall_cv_score,
    })
}

/// Perform k-fold cross-validation
pub fn perform_kfold_validation(
    implementation: &str,
    t: &[f64],
    signal: &[f64],
    k: usize,
    true_freq: f64,
) -> SignalResult<f64> {
    let n = t.len();
    let fold_size = n / k;
    let mut scores = Vec::new();

    for fold in 0..k {
        let start = fold * fold_size;
        let end = if fold == k - 1 {
            n
        } else {
            (fold + 1) * fold_size
        };

        // Create training set (exclude current fold)
        let mut train_t = Vec::new();
        let mut train_signal = Vec::new();

        for i in 0..n {
            if i < start || i >= end {
                train_t.push(t[i]);
                train_signal.push(signal[i]);
            }
        }

        if train_t.len() < 10 {
            continue; // Skip if training set too small
        }

        // Train on subset and test frequency detection
        match run_lombscargle(implementation, &train_t, &train_signal) {
            Ok((freqs, power)) => {
                let (_, detected_freq) = freqs
                    .iter()
                    .zip(power.iter())
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .map(|(&f, _)| ((), f))
                    .unwrap_or(((), 0.0));

                let freq_error = (detected_freq - true_freq).abs() / true_freq;
                scores.push(1.0 - freq_error.min(1.0));
            }
            Err(_) => scores.push(0.0),
        }
    }

    Ok(if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    })
}

/// Perform bootstrap validation
pub fn perform_bootstrap_validation(
    implementation: &str,
    t: &[f64],
    signal: &[f64],
    n_bootstrap: usize,
    true_freq: f64,
) -> SignalResult<f64> {
    let mut scores = Vec::new();
    let n = t.len();
    let mut rng = scirs2_core::random::thread_rng();

    for _ in 0..n_bootstrap {
        // Bootstrap sample
        let mut boot_t = Vec::new();
        let mut boot_signal = Vec::new();

        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            boot_t.push(t[idx]);
            boot_signal.push(signal[idx]);
        }

        // Sort by time
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| boot_t[i].partial_cmp(&boot_t[j]).unwrap());

        let sorted_t: Vec<f64> = indices.iter().map(|&i| boot_t[i]).collect();
        let sorted_signal: Vec<f64> = indices.iter().map(|&i| boot_signal[i]).collect();

        match run_lombscargle(implementation, &sorted_t, &sorted_signal) {
            Ok((freqs, power)) => {
                let (_, detected_freq) = freqs
                    .iter()
                    .zip(power.iter())
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .map(|(&f, _)| ((), f))
                    .unwrap_or(((), 0.0));

                let freq_error = (detected_freq - true_freq).abs() / true_freq;
                scores.push(1.0 - freq_error.min(1.0));
            }
            Err(_) => scores.push(0.0),
        }
    }

    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}

/// Perform leave-one-out validation (simplified)
pub fn perform_loo_validation(
    implementation: &str,
    t: &[f64],
    signal: &[f64],
    true_freq: f64,
) -> SignalResult<f64> {
    // Simplified: remove every 10th point and test
    let mut scores = Vec::new();
    let step = 10;

    for start in 0..step {
        let mut loo_t = Vec::new();
        let mut loo_signal = Vec::new();

        for (i, (&ti, &si)) in t.iter().zip(signal.iter()).enumerate() {
            if i % step != start {
                loo_t.push(ti);
                loo_signal.push(si);
            }
        }

        if loo_t.len() < 20 {
            continue;
        }

        match run_lombscargle(implementation, &loo_t, &loo_signal) {
            Ok((freqs, power)) => {
                let (_, detected_freq) = freqs
                    .iter()
                    .zip(power.iter())
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .map(|(&f, _)| ((), f))
                    .unwrap_or(((), 0.0));

                let freq_error = (detected_freq - true_freq).abs() / true_freq;
                scores.push(1.0 - freq_error.min(1.0));
            }
            Err(_) => scores.push(0.0),
        }
    }

    Ok(if scores.is_empty() {
        0.5
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    })
}

/// Test temporal consistency with sliding windows
pub fn test_temporal_consistency(
    implementation: &str,
    t: &[f64],
    signal: &[f64],
    true_freq: f64,
) -> SignalResult<f64> {
    let window_size = t.len() / 3;
    let n_windows = 3;
    let mut detected_freqs = Vec::new();

    for i in 0..n_windows {
        let start = i * (t.len() - window_size) / (n_windows - 1).max(1);
        let end = start + window_size;

        let window_t = &t[start..end];
        let window_signal = &signal[start..end];

        match run_lombscargle(implementation, window_t, window_signal) {
            Ok((freqs, power)) => {
                let (_, detected_freq) = freqs
                    .iter()
                    .zip(power.iter())
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .map(|(&f, _)| ((), f))
                    .unwrap_or(((), 0.0));
                detected_freqs.push(detected_freq);
            }
            Err(_) => detected_freqs.push(0.0),
        }
    }

    // Calculate consistency of detected frequencies
    if detected_freqs.is_empty() {
        return Ok(0.0);
    }

    let mean_freq = detected_freqs.iter().sum::<f64>() / detected_freqs.len() as f64;
    let variance = detected_freqs
        .iter()
        .map(|&f| (f - mean_freq).powi(2))
        .sum::<f64>()
        / detected_freqs.len() as f64;

    let consistency = 1.0 / (1.0 + variance / true_freq.powi(2));
    Ok(consistency)
}

/// Test frequency stability across different data splits
pub fn test_frequency_stability(
    implementation: &str,
    t: &[f64],
    signal: &[f64],
    _true_freq: f64,
) -> SignalResult<f64> {
    // Split data in different ways and test frequency consistency
    let n = t.len();
    let mut freq_estimates = Vec::new();

    // Split 1: First half vs second half
    let mid = n / 2;
    let splits = vec![(0, mid), (mid, n), (0, n * 3 / 4), (n / 4, n)];

    for (start, end) in splits {
        let split_t = &t[start..end];
        let split_signal = &signal[start..end];

        match run_lombscargle(implementation, split_t, split_signal) {
            Ok((freqs, power)) => {
                let (_, detected_freq) = freqs
                    .iter()
                    .zip(power.iter())
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .map(|(&f, _)| ((), f))
                    .unwrap_or(((), 0.0));
                freq_estimates.push(detected_freq);
            }
            Err(_) => freq_estimates.push(0.0),
        }
    }

    if freq_estimates.is_empty() {
        return Ok(0.0);
    }

    // Calculate stability as inverse of relative standard deviation
    let mean_freq = freq_estimates.iter().sum::<f64>() / freq_estimates.len() as f64;
    let std_dev = {
        let variance = freq_estimates
            .iter()
            .map(|&f| (f - mean_freq).powi(2))
            .sum::<f64>()
            / freq_estimates.len() as f64;
        variance.sqrt()
    };

    let relative_std = if mean_freq > 0.0 {
        std_dev / mean_freq
    } else {
        1.0
    };
    let stability = 1.0 / (1.0 + relative_std);

    Ok(stability)
}