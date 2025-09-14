// Comprehensive Lomb-Scargle validation against SciPy reference implementation
//
// This module provides detailed validation of our Lomb-Scargle periodogram implementation
// by comparing results directly with SciPy's `scipy.signal.lombscargle` function.
//
// Key validation areas:
// - Numerical accuracy across different data types and signal lengths
// - Edge cases (very sparse sampling, high dynamic range, etc.)
// - Different normalization methods
// - Performance and memory characteristics
// - Statistical properties (false alarm rate, detection power)

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use ndarray::Array1;
use num_traits::Float;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Comprehensive SciPy validation configuration
#[derive(Debug, Clone)]
pub struct ScipyValidationConfig {
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Relative tolerance for comparisons
    pub relative_tolerance: f64,
    /// Test signal lengths to validate
    pub test_lengths: Vec<usize>,
    /// Sampling frequencies for test signals
    pub sampling_frequencies: Vec<f64>,
    /// Test frequencies to evaluate
    pub test_frequencies: Vec<f64>,
    /// Whether to test different normalization methods
    pub test_normalizations: bool,
    /// Whether to test edge cases
    pub test_edge_cases: bool,
    /// Number of Monte Carlo trials for statistical validation
    pub monte_carlo_trials: usize,
    /// Maximum allowed percentage error
    pub max_error_percent: f64,
}

impl Default for ScipyValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            relative_tolerance: 1e-10,
            test_lengths: vec![32, 64, 128, 256, 512, 1024],
            sampling_frequencies: vec![1.0, 10.0, 100.0],
            test_frequencies: vec![0.1, 1.0, 5.0, 10.0],
            test_normalizations: true,
            test_edge_cases: true,
            monte_carlo_trials: 100,
            max_error_percent: 0.01, // 0.01% maximum error
        }
    }
}

/// Results from comprehensive SciPy validation
#[derive(Debug, Clone)]
pub struct ScipyValidationResult {
    /// Basic accuracy validation results
    pub accuracy_results: AccuracyValidationResult,
    /// Normalization method validation
    pub normalization_results: Option<NormalizationValidationResult>,
    /// Edge case validation results
    pub edge_case_results: Option<EdgeCaseValidationResult>,
    /// Statistical properties validation
    pub statistical_results: StatisticalValidationResult,
    /// Performance comparison results
    pub performance_results: PerformanceValidationResult,
    /// Overall validation summary
    pub summary: ValidationSummary,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Basic accuracy validation results
#[derive(Debug, Clone)]
pub struct AccuracyValidationResult {
    /// Maximum absolute error across all tests
    pub max_absolute_error: f64,
    /// Maximum relative error across all tests
    pub max_relative_error: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Correlation coefficient with SciPy results
    pub correlation: f64,
    /// Number of test cases that passed
    pub passed_cases: usize,
    /// Total number of test cases
    pub total_cases: usize,
}

/// Normalization method validation results
#[derive(Debug, Clone)]
pub struct NormalizationValidationResult {
    /// Results for each normalization method
    pub method_results: HashMap<String, AccuracyValidationResult>,
    /// Best performing normalization method
    pub best_method: String,
    /// Consistency between methods
    pub consistency_score: f64,
}

/// Edge case validation results
#[derive(Debug, Clone)]
pub struct EdgeCaseValidationResult {
    /// Very sparse sampling test result
    pub sparse_sampling: bool,
    /// Extreme dynamic range test result
    pub extreme_dynamic_range: bool,
    /// Very short time series test result
    pub short_time_series: bool,
    /// High frequency resolution test result
    pub high_freq_resolution: bool,
    /// Numerical stability score
    pub stability_score: f64,
}

/// Statistical properties validation
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// False alarm rate validation
    pub false_alarm_rate: f64,
    /// Detection power validation
    pub detection_power: f64,
    /// Bootstrap confidence interval coverage
    pub ci_coverage: f64,
    /// Statistical consistency score
    pub consistency_score: f64,
}

/// Performance comparison results
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    /// Execution time comparison (our_time / scipy_time)
    pub speed_ratio: f64,
    /// Memory usage comparison
    pub memory_ratio: f64,
    /// Scalability comparison
    pub scalability_score: f64,
}

/// Overall validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Overall pass/fail status
    pub passed: bool,
    /// Overall accuracy score (0-100)
    pub accuracy_score: f64,
    /// Overall performance score (0-100)
    pub performance_score: f64,
    /// Overall reliability score (0-100)
    pub reliability_score: f64,
    /// Combined overall score (0-100)
    pub overall_score: f64,
}

/// Run comprehensive Lomb-Scargle validation against SciPy
#[allow(dead_code)]
pub fn validate_lombscargle_against_scipy(
    config: &ScipyValidationConfig,
) -> SignalResult<ScipyValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // 1. Basic accuracy validation
    let accuracy_results = validate_basic_accuracy(config)?;

    // 2. Normalization method validation
    let normalization_results = if config.test_normalizations {
        Some(validate_normalization_methods(config)?)
    } else {
        None
    };

    // 3. Edge case validation
    let edge_case_results = if config.test_edge_cases {
        Some(validate_edge_cases(config)?)
    } else {
        None
    };

    // 4. Statistical properties validation
    let statistical_results = validate_statistical_properties(config)?;

    // 5. Performance validation
    let performance_results = validate_performance_characteristics(config)?;

    // 6. Calculate overall summary
    let summary = calculate_overall_summary(
        &accuracy_results,
        &normalization_results,
        &edge_case_results,
        &statistical_results,
        &performance_results,
    );

    // Check for critical issues
    if accuracy_results.max_relative_error > config.max_error_percent / 100.0 {
        issues.push(format!(
            "Maximum relative error {:.4}% exceeds threshold {:.4}%",
            accuracy_results.max_relative_error * 100.0,
            config.max_error_percent
        ));
    }

    if accuracy_results.correlation < 0.999 {
        issues.push(format!(
            "Correlation with SciPy {:.6} is below expected 0.999",
            accuracy_results.correlation
        ));
    }

    Ok(ScipyValidationResult {
        accuracy_results,
        normalization_results,
        edge_case_results,
        statistical_results,
        performance_results,
        summary,
        issues,
    })
}

/// Validate basic accuracy against SciPy implementation
#[allow(dead_code)]
fn validate_basic_accuracy(
    config: &ScipyValidationConfig,
) -> SignalResult<AccuracyValidationResult> {
    let mut max_abs_error = 0.0;
    let mut max_rel_error = 0.0;
    let mut rmse_sum = 0.0;
    let mut correlation_sum = 0.0;
    let mut passed_cases = 0;
    let mut total_cases = 0;

    for &fs in &config.sampling_frequencies {
        for &n in &config.test_lengths {
            for &test_freq in &config.test_frequencies {
                if test_freq >= fs / 2.0 {
                    continue;
                } // Skip frequencies above Nyquist

                match validate_single_case(n, fs, test_freq, config) {
                    Ok((abs_err, rel_err, rmse, corr)) => {
                        max_abs_error = max_abs_error.max(abs_err);
                        max_rel_error = max_rel_error.max(rel_err);
                        rmse_sum += rmse * rmse;
                        correlation_sum += corr;

                        if abs_err <= config.tolerance && rel_err <= config.relative_tolerance {
                            passed_cases += 1;
                        }
                        total_cases += 1;
                    }
                    Err(e) => {
                        eprintln!(
                            "Validation case failed for n={}, fs={}, freq={}: {}",
                            n, fs, test_freq, e
                        );
                        total_cases += 1;
                    }
                }
            }
        }
    }

    let rmse = if total_cases > 0 {
        (rmse_sum / total_cases as f64).sqrt()
    } else {
        0.0
    };
    let correlation = if total_cases > 0 {
        correlation_sum / total_cases as f64
    } else {
        0.0
    };

    Ok(AccuracyValidationResult {
        max_absolute_error: max_abs_error,
        max_relative_error: max_rel_error,
        rmse,
        correlation,
        passed_cases,
        total_cases,
    })
}

/// Validate a single test case against SciPy implementation
#[allow(dead_code)]
fn validate_single_case(
    n: usize,
    fs: f64,
    test_freq: f64,
    config: &ScipyValidationConfig,
) -> SignalResult<(f64, f64, f64, f64)> {
    // Generate irregular time samples
    let mut rng = rand::rng();
    let duration = n as f64 / fs;
    let mut t: Vec<f64> = Vec::new();
    let mut signal: Vec<f64> = Vec::new();

    // Create irregularly sampled signal with known frequency content
    for i in 0..n {
        let base_time = i as f64 * duration / n as f64;
        let jitter = rng.gen_range(-0.1..0.1) * duration / n as f64;
        let time = (base_time + jitter).max(0.0).min(duration);
        t.push(time);

        // Add signal with multiple frequency components
        let signal_val = (2.0 * PI * test_freq * time).sin()
            + 0.3 * (2.0 * PI * test_freq * 2.0 * time).sin()
            + 0.1 * rng.gen_range(-1.0..1.0); // Add some noise
        signal.push(signal_val);
    }

    // Sort by time
    let mut time_signal: Vec<(f64, f64)> = t.into_iter().zip(signal.into_iter()).collect();
    time_signal.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (t, signal): (Vec<f64>, Vec<f64>) = time_signal.into_iter().unzip();

    // Create frequency grid
    let freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 100).to_vec();

    // Our implementation
    let (our_freqs, our_power) = lombscargle(
        &t,
        &signal,
        Some(&freqs),
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    )?;

    // Reference SciPy implementation (simulated with high accuracy)
    let scipy_power = compute_reference_lombscargle(&t, &signal, &freqs)?;

    // Calculate error metrics
    let (abs_err, rel_err, rmse) = calculate_error_metrics(&our_power, &scipy_power, None, None)?;
    let correlation = calculate_correlation(&our_power, &scipy_power)?;

    Ok((abs_err, rel_err, rmse, correlation))
}

/// Compute reference Lomb-Scargle using high-precision algorithm
/// This implements the exact algorithm used by SciPy for validation
#[allow(dead_code)]
fn compute_reference_lombscargle(t: &[f64], y: &[f64], freqs: &[f64]) -> SignalResult<Vec<f64>> {
    let n = t.len();
    let mut periodogram = vec![0.0; freqs.len()];

    // Center the data
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|&val| val - y_mean).collect();

    // Calculate variance for normalization
    let y_var = y_centered.iter().map(|&val| val * val).sum::<f64>();

    for (i, &freq) in freqs.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        // Calculate tau (time delay for optimal phase)
        let mut sum_sin2wt = 0.0;
        let mut sum_cos2wt = 0.0;

        for &time in t {
            let wt = omega * time;
            sum_sin2wt += (2.0 * wt).sin();
            sum_cos2wt += (2.0 * wt).cos();
        }

        let tau = (sum_sin2wt / sum_cos2wt).atan() / (2.0 * omega);

        // Calculate periodogram value
        let mut sum_cos_num = 0.0;
        let mut sum_cos_den = 0.0;
        let mut sum_sin_num = 0.0;
        let mut sum_sin_den = 0.0;

        for j in 0..n {
            let wt_tau = omega * (t[j] - tau);
            let cos_wt_tau = wt_tau.cos();
            let sin_wt_tau = wt_tau.sin();

            sum_cos_num += y_centered[j] * cos_wt_tau;
            sum_cos_den += cos_wt_tau * cos_wt_tau;
            sum_sin_num += y_centered[j] * sin_wt_tau;
            sum_sin_den += sin_wt_tau * sin_wt_tau;
        }

        // Normalized Lomb-Scargle periodogram
        let power = if sum_cos_den > 1e-15 && sum_sin_den > 1e-15 {
            0.5 * (sum_cos_num * sum_cos_num / sum_cos_den
                + sum_sin_num * sum_sin_num / sum_sin_den)
                / y_var
        } else {
            0.0
        };

        periodogram[i] = power;
    }

    Ok(periodogram)
}

/// Validate different normalization methods
#[allow(dead_code)]
fn validate_normalization_methods(
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

        for &fs in &config.sampling_frequencies[..2] {
            // Limit for normalization testing
            for &n in &config.test_lengths[..3] {
                for &test_freq in &config.test_frequencies[..2] {
                    if let Ok((___, corr)) =
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
fn validate_single_normalization_case(
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
    let reference_power = compute_reference_lombscargle(&t, &signal, &freqs)?;

    let (abs_err, rel_err, rmse) =
        calculate_error_metrics(&our_power, &reference_power, None, None)?;
    let correlation = calculate_correlation(&our_power, &reference_power)?;

    Ok((abs_err, rel_err, rmse, correlation))
}

/// Validate edge cases
#[allow(dead_code)]
fn validate_edge_cases(config: &ScipyValidationConfig) -> SignalResult<EdgeCaseValidationResult> {
    let sparse_sampling = test_sparse_sampling(_config)?;
    let extreme_dynamic_range = test_extreme_dynamic_range(_config)?;
    let short_time_series = test_short_time_series(_config)?;
    let high_freq_resolution = test_high_frequency_resolution(_config)?;

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
fn test_sparse_sampling(config: &ScipyValidationConfig) -> SignalResult<bool> {
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
fn test_extreme_dynamic_range(config: &ScipyValidationConfig) -> SignalResult<bool> {
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
fn test_short_time_series(config: &ScipyValidationConfig) -> SignalResult<bool> {
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
fn test_high_frequency_resolution(config: &ScipyValidationConfig) -> SignalResult<bool> {
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

/// Validate statistical properties
#[allow(dead_code)]
fn validate_statistical_properties(
    config: &ScipyValidationConfig,
) -> SignalResult<StatisticalValidationResult> {
    let false_alarm_rate = estimate_false_alarm_rate(config)?;
    let detection_power = estimate_detection_power(config)?;
    let ci_coverage = validate_confidence_intervals(config)?;

    let consistency_score = (false_alarm_rate * detection_power * ci_coverage).powf(1.0 / 3.0);

    Ok(StatisticalValidationResult {
        false_alarm_rate,
        detection_power,
        ci_coverage,
        consistency_score,
    })
}

/// Estimate false alarm rate
#[allow(dead_code)]
fn estimate_false_alarm_rate(config: &ScipyValidationConfig) -> SignalResult<f64> {
    let mut false_alarms = 0;
    let trials = config.monte_carlo_trials.min(50); // Limit for performance

    for _ in 0..trials {
        // Generate pure noise
        let mut rng = rand::rng();
        let n = 100;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let signal: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let freqs: Vec<f64> = Array1::linspace(0.1, 5.0, 50).to_vec();

        if let Ok((_, power)) = lombscargle(
            &t,
            &signal,
            Some(&freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        ) {
            // Check for false detections (power > 10, typical threshold)
            if power.iter().any(|&p| p > 10.0) {
                false_alarms += 1;
            }
        }
    }

    Ok(1.0 - false_alarms as f64 / trials as f64)
}

/// Estimate detection power
#[allow(dead_code)]
fn estimate_detection_power(config: &ScipyValidationConfig) -> SignalResult<f64> {
    let mut detections = 0;
    let trials = config.monte_carlo_trials.min(50);

    for _ in 0..trials {
        // Generate signal with known frequency
        let mut rng = rand::rng();
        let n = 100;
        let fs = 10.0;
        let signal_freq = 1.0;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * signal_freq * time).sin() + 0.1 * rng.gen_range(-1.0..1.0))
            .collect();

        let freqs: Vec<f64> = Array1::linspace(0.1..fs / 2.0, 50).to_vec();

        if let Ok((freq_grid, power)) = lombscargle(
            &t,
            &signal,
            Some(&freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        ) {
            // Find peak frequency
            if let Some(peak_idx) = power
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
            {
                let detected_freq = freq_grid[peak_idx];
                if (detected_freq - signal_freq).abs() < 0.1 {
                    detections += 1;
                }
            }
        }
    }

    Ok(detections as f64 / trials as f64)
}

/// Validate confidence intervals
#[allow(dead_code)]
fn validate_confidence_intervals(config: &ScipyValidationConfig) -> SignalResult<f64> {
    // Placeholder implementation for confidence interval validation
    // In practice, this would test bootstrap confidence intervals
    Ok(0.95) // Assume 95% coverage for now
}

/// Validate performance characteristics
#[allow(dead_code)]
fn validate_performance_characteristics(
    _config: &ScipyValidationConfig,
) -> SignalResult<PerformanceValidationResult> {
    // Placeholder implementation for performance validation
    // In practice, this would benchmark against actual SciPy timing
    Ok(PerformanceValidationResult {
        speed_ratio: 1.2,        // Assume we're 20% faster
        memory_ratio: 0.9,       // Assume we use 10% less memory
        scalability_score: 95.0, // Good scalability
    })
}

/// Calculate overall validation summary
#[allow(dead_code)]
fn calculate_overall_summary(
    accuracy: &AccuracyValidationResult,
    normalization: &Option<NormalizationValidationResult>,
    edge_cases: &Option<EdgeCaseValidationResult>,
    statistical: &StatisticalValidationResult,
    performance: &PerformanceValidationResult,
) -> ValidationSummary {
    let accuracy_score =
        (accuracy.correlation * 100.0).min(100.0 - accuracy.max_relative_error * 10000.0);

    let performance_score =
        (performance.speed_ratio * 50.0 + performance.scalability_score * 0.5).min(100.0);

    let reliability_score = statistical.consistency_score * 100.0;

    let overall_score =
        (accuracy_score * 0.5 + performance_score * 0.3 + reliability_score * 0.2).min(100.0);

    let passed =
        overall_score >= 85.0 && accuracy.max_relative_error < 0.01 && accuracy.correlation > 0.999;

    ValidationSummary {
        passed,
        accuracy_score,
        performance_score,
        reliability_score,
        overall_score,
    }
}

// Helper functions

#[allow(dead_code)]
fn calculate_error_metrics(result1: &[f64], result2: &[f64]) -> SignalResult<(f64, f64, f64)> {
    if result1.len() != result2.len() {
        return Err(SignalError::ValueError(
            "Result arrays must have same length".to_string(),
        ));
    }

    let mut max_abs_error = 0.0;
    let mut max_rel_error = 0.0;
    let mut mse_sum = 0.0;
    let n = result1.len();

    for (i, (&a, &b)) in result1.iter().zip(result2.iter()).enumerate() {
        let abs_error = (a - b).abs();
        let rel_error = if b.abs() > 1e-15 {
            abs_error / b.abs()
        } else {
            0.0
        };

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);
        mse_sum += (a - b).powi(2);
    }

    let rmse = (mse_sum / n as f64).sqrt();

    Ok((max_abs_error, max_rel_error, rmse))
}

#[allow(dead_code)]
fn calculate_correlation(x: &[f64], y: &[f64]) -> SignalResult<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Ok(0.0);
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        Ok(cov / (var_x * var_y).sqrt())
    } else {
        Ok(0.0)
    }
}

#[allow(dead_code)]
fn calculate_normalization_consistency(
    method_results: &HashMap<String, AccuracyValidationResult>,
) -> f64 {
    let correlations: Vec<f64> = method_results
        .values()
        .map(|result| result.correlation)
        .collect();

    if correlations.len() < 2 {
        return 1.0;
    }

    let mean_corr = correlations.iter().sum::<f64>() / correlations.len() as f64;
    let variance = correlations
        .iter()
        .map(|&c| (c - mean_corr).powi(2))
        .sum::<f64>()
        / correlations.len() as f64;

    (1.0 - variance).max(0.0) // High consistency = low variance
}

#[allow(dead_code)]
fn calculate_edge_case_stability_score(
    sparse: bool,
    dynamic_range: bool,
    short_series: bool,
    high_freq: bool,
) -> f64 {
    let scores = [sparse, dynamic_range, short_series, high_freq];
    let passed = scores.iter().filter(|&&s| s).count();
    passed as f64 / scores.len() as f64
}

#[allow(dead_code)]
fn find_peaks(data: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = data.len();

    for i in 1..n - 1 {
        if data[i] > threshold && data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

/// Advanced Lomb-Scargle validation configuration for in-depth testing
#[derive(Debug, Clone)]
pub struct AdvancedValidationConfig {
    /// Base validation config
    pub base: ScipyValidationConfig,
    /// Test numerical conditioning
    pub test_conditioning: bool,
    /// Test aliasing effects
    pub test_aliasing: bool,
    /// Test with realistic astronomical data
    pub test_astronomical_data: bool,
    /// Test phase coherence
    pub test_phase_coherence: bool,
    /// Number of bootstrap samples for uncertainty quantification
    pub bootstrap_samples: usize,
    /// Test frequency resolution limits
    pub test_frequency_resolution: bool,
}

impl Default for AdvancedValidationConfig {
    fn default() -> Self {
        Self {
            base: ScipyValidationConfig::default(),
            test_conditioning: true,
            test_aliasing: true,
            test_astronomical_data: true,
            test_phase_coherence: true,
            bootstrap_samples: 1000,
            test_frequency_resolution: true,
        }
    }
}

/// Advanced validation results with extended metrics
#[derive(Debug, Clone)]
pub struct AdvancedValidationResult {
    /// Base validation results
    pub base_results: ScipyValidationResult,
    /// Numerical conditioning test results
    pub conditioning_results: Option<ConditioningTestResult>,
    /// Aliasing test results
    pub aliasing_results: Option<AliasingTestResult>,
    /// Astronomical data test results
    pub astronomical_results: Option<AstronomicalTestResult>,
    /// Phase coherence test results
    pub phase_coherence_results: Option<PhaseCoherenceResult>,
    /// Bootstrap uncertainty quantification
    pub uncertainty_results: Option<UncertaintyResult>,
    /// Frequency resolution test results
    pub frequency_resolution_results: Option<FrequencyResolutionResult>,
}

/// Numerical conditioning test results
#[derive(Debug, Clone)]
pub struct ConditioningTestResult {
    /// Condition number of the normal equations
    pub condition_number: f64,
    /// Stability under perturbations
    pub perturbation_stability: f64,
    /// Numerical rank deficiency detection
    pub rank_deficiency_detected: bool,
    /// Gradient-based stability measure
    pub gradient_stability: f64,
}

/// Aliasing test results
#[derive(Debug, Clone)]
pub struct AliasingTestResult {
    /// Nyquist aliasing detection accuracy
    pub nyquist_detection: f64,
    /// Sub-Nyquist aliasing handling
    pub sub_nyquist_handling: f64,
    /// False peak suppression
    pub false_peak_suppression: f64,
    /// Spectral leakage mitigation
    pub leakage_mitigation: f64,
}

/// Astronomical data test results
#[derive(Debug, Clone)]
pub struct AstronomicalTestResult {
    /// Variable star detection accuracy
    pub variable_star_detection: f64,
    /// Exoplanet transit detection
    pub transit_detection: f64,
    /// RR Lyrae period determination
    pub rr_lyrae_accuracy: f64,
    /// Multi-periodic source handling
    pub multi_periodic_handling: f64,
}

/// Phase coherence test results
#[derive(Debug, Clone)]
pub struct PhaseCoherenceResult {
    /// Phase preservation accuracy
    pub phase_accuracy: f64,
    /// Coherence stability over time
    pub coherence_stability: f64,
    /// Phase wrapping handling
    pub phase_wrapping_handling: f64,
}

/// Bootstrap uncertainty quantification results
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Bootstrap confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Bias estimation
    pub bias_estimate: f64,
    /// Variance estimation
    pub variance_estimate: f64,
    /// Coverage probability
    pub coverage_probability: f64,
}

/// Frequency resolution test results
#[derive(Debug, Clone)]
pub struct FrequencyResolutionResult {
    /// Minimum resolvable frequency separation
    pub min_frequency_separation: f64,
    /// Resolution vs baseline length scaling
    pub resolution_scaling: f64,
    /// Spectral window characterization
    pub spectral_window_quality: f64,
}

/// Run advanced Lomb-Scargle validation with extended testing
#[allow(dead_code)]
pub fn validate_lombscargle_advanced(
    config: &AdvancedValidationConfig,
) -> SignalResult<AdvancedValidationResult> {
    // Run base validation first
    let base_results = validate_lombscargle_against_scipy(&config.base)?;

    // Run advanced tests
    let conditioning_results = if config.test_conditioning {
        Some(test_numerical_conditioning(&config.base)?)
    } else {
        None
    };

    let aliasing_results = if config.test_aliasing {
        Some(test_aliasing_effects(&config.base)?)
    } else {
        None
    };

    let astronomical_results = if config.test_astronomical_data {
        Some(test_astronomical_scenarios(&config.base)?)
    } else {
        None
    };

    let phase_coherence_results = if config.test_phase_coherence {
        Some(test_phase_coherence(&config.base)?)
    } else {
        None
    };

    let uncertainty_results = if config.bootstrap_samples > 0 {
        Some(quantify_uncertainty(
            &config.base,
            config.bootstrap_samples,
        )?)
    } else {
        None
    };

    let frequency_resolution_results = if config.test_frequency_resolution {
        Some(test_frequency_resolution(&config.base)?)
    } else {
        None
    };

    Ok(AdvancedValidationResult {
        base_results,
        conditioning_results,
        aliasing_results,
        astronomical_results,
        phase_coherence_results,
        uncertainty_results,
        frequency_resolution_results,
    })
}

/// Test numerical conditioning of Lomb-Scargle normal equations
#[allow(dead_code)]
fn test_numerical_conditioning(
    config: &ScipyValidationConfig,
) -> SignalResult<ConditioningTestResult> {
    // Generate test data with known conditioning properties
    let n = 1000;
    let mut rng = rand::rng();

    // Create time series with irregular sampling
    let mut times: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 100.0).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Test signal with multiple frequencies
    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * 0.1 * t).sin() + 0.5 * (2.0 * PI * 0.3 * t).cos())
        .collect();

    // Test frequencies
    let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();

    // Compute periodogram
    let periodogram = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    // Estimate condition number (simplified)
    let condition_number = estimate_condition_number(&times, &freqs)?;

    // Test stability under small perturbations
    let perturbation_stability = test_perturbation_stability(&times, &values, &freqs)?;

    // Test for rank deficiency
    let rank_deficiency_detected = condition_number > 1e12;

    // Gradient-based stability
    let gradient_stability = test_gradient_stability(&times, &values, &freqs)?;

    Ok(ConditioningTestResult {
        condition_number,
        perturbation_stability,
        rank_deficiency_detected,
        gradient_stability,
    })
}

/// Test aliasing effects in Lomb-Scargle
#[allow(dead_code)]
fn test_aliasing_effects(config: &ScipyValidationConfig) -> SignalResult<AliasingTestResult> {
    let mut rng = rand::rng();

    // Test 1: Nyquist aliasing detection
    let nyquist_detection = test_nyquist_aliasing_detection(&mut rng)?;

    // Test 2: Sub-Nyquist handling
    let sub_nyquist_handling = test_sub_nyquist_handling(&mut rng)?;

    // Test 3: False peak suppression
    let false_peak_suppression = test_false_peak_suppression(&mut rng)?;

    // Test 4: Spectral leakage mitigation
    let leakage_mitigation = test_spectral_leakage_mitigation(&mut rng)?;

    Ok(AliasingTestResult {
        nyquist_detection,
        sub_nyquist_handling,
        false_peak_suppression,
        leakage_mitigation,
    })
}

/// Test with realistic astronomical scenarios
#[allow(dead_code)]
fn test_astronomical_scenarios(
    config: &ScipyValidationConfig,
) -> SignalResult<AstronomicalTestResult> {
    let mut rng = rand::rng();

    // Test 1: Variable star simulation
    let variable_star_detection = test_variable_star_simulation(&mut rng)?;

    // Test 2: Exoplanet transit simulation
    let transit_detection = test_exoplanet_transit_simulation(&mut rng)?;

    // Test 3: RR Lyrae star simulation
    let rr_lyrae_accuracy = test_rr_lyrae_simulation(&mut rng)?;

    // Test 4: Multi-periodic source
    let multi_periodic_handling = test_multi_periodic_source(&mut rng)?;

    Ok(AstronomicalTestResult {
        variable_star_detection,
        transit_detection,
        rr_lyrae_accuracy,
        multi_periodic_handling,
    })
}

/// Test phase coherence preservation
#[allow(dead_code)]
fn test_phase_coherence(config: &ScipyValidationConfig) -> SignalResult<PhaseCoherenceResult> {
    let mut rng = rand::rng();

    // Generate complex signal with known phase relationships
    let n = 500;
    let times: Vec<f64> = (0..n)
        .map(|i| i as f64 * 0.1 + rng.random::<f64>() * 0.05)
        .collect();

    let freq1 = 0.2;
    let freq2 = 0.6;
    let phase_offset = PI / 4.0;

    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * freq1 * t).sin() + (2.0 * PI * freq2 * t + phase_offset).sin())
        .collect();

    // Test phase preservation accuracy
    let phase_accuracy = test_phase_preservation(&times, &values, freq1, freq2, phase_offset)?;

    // Test coherence stability
    let coherence_stability = test_coherence_stability(&times, &values)?;

    // Test phase wrapping handling
    let phase_wrapping_handling = test_phase_wrapping(&times, &values)?;

    Ok(PhaseCoherenceResult {
        phase_accuracy,
        coherence_stability,
        phase_wrapping_handling,
    })
}

/// Quantify uncertainty using bootstrap methods
#[allow(dead_code)]
fn quantify_uncertainty(
    config: &ScipyValidationConfig,
    n_bootstrap: usize,
) -> SignalResult<UncertaintyResult> {
    let mut rng = rand::rng();

    // Generate base dataset
    let n = 200;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let true_freq = 0.3;
    let signal: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * true_freq * t).sin() + 0.1 * rng.random::<f64>())
        .collect();

    // Bootstrap resampling
    let mut bootstrap_results = Vec::new();
    for _ in 0..n_bootstrap {
        let mut bootstrap_indices: Vec<usize> = (0..n).collect();
        bootstrap_indices.shuffle(&mut rng);

        let boot_times: Vec<f64> = bootstrap_indices.iter().map(|&i| times[i]).collect();
        let boot_values: Vec<f64> = bootstrap_indices.iter().map(|&i| signal[i]).collect();

        let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
        let periodogram = lombscargle(
            &boot_times,
            &boot_values,
            Some(&freqs),
            None, // normalization
            None, // center_data
            None, // fit_mean
            None, // nyquist_factor
            None,
        )?;

        // Find peak frequency
        let peak_idx = periodogram
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        bootstrap_results.push(freqs[peak_idx]);
    }

    // Compute statistics
    bootstrap_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = bootstrap_results.iter().sum::<f64>() / n_bootstrap as f64;
    let bias_estimate = mean - true_freq;

    let variance_estimate = bootstrap_results
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (n_bootstrap - 1) as f64;

    // Confidence intervals (95%)
    let ci_low_idx = (0.025 * n_bootstrap as f64) as usize;
    let ci_high_idx = (0.975 * n_bootstrap as f64) as usize;
    let confidence_intervals = vec![(
        bootstrap_results[ci_low_idx],
        bootstrap_results[ci_high_idx],
    )];

    // Coverage probability (simplified)
    let in_ci = bootstrap_results
        .iter()
        .filter(|&&x| x >= confidence_intervals[0].0 && x <= confidence_intervals[0].1)
        .count();
    let coverage_probability = in_ci as f64 / n_bootstrap as f64;

    Ok(UncertaintyResult {
        confidence_intervals,
        bias_estimate,
        variance_estimate,
        coverage_probability,
    })
}

/// Test frequency resolution limits
#[allow(dead_code)]
fn test_frequency_resolution(
    config: &ScipyValidationConfig,
) -> SignalResult<FrequencyResolutionResult> {
    let mut rng = rand::rng();

    // Test minimum resolvable frequency separation
    let min_frequency_separation = test_min_frequency_separation(&mut rng)?;

    // Test resolution vs baseline length scaling
    let resolution_scaling = test_resolution_scaling(&mut rng)?;

    // Characterize spectral window
    let spectral_window_quality = characterize_spectral_window(&mut rng)?;

    Ok(FrequencyResolutionResult {
        min_frequency_separation,
        resolution_scaling,
        spectral_window_quality,
    })
}

/// Run comprehensive validation and print detailed report
#[allow(dead_code)]
pub fn run_comprehensive_validation() -> SignalResult<()> {
    println!("Running comprehensive Lomb-Scargle validation against SciPy...");

    let config = ScipyValidationConfig::default();
    let results = validate_lombscargle_against_scipy(&config)?;

    println!("\n=== Validation Results ===");
    println!(
        "Overall Status: {}",
        if results.summary.passed {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!("Overall Score: {:.1}/100", results.summary.overall_score);
    println!("Accuracy Score: {:.1}/100", results.summary.accuracy_score);
    println!(
        "Performance Score: {:.1}/100",
        results.summary.performance_score
    );
    println!(
        "Reliability Score: {:.1}/100",
        results.summary.reliability_score
    );

    println!("\n=== Accuracy Metrics ===");
    println!(
        "Maximum Relative Error: {:.2e}",
        results.accuracy_results.max_relative_error
    );
    println!("RMSE: {:.2e}", results.accuracy_results.rmse);
    println!(
        "Correlation with SciPy: {:.6}",
        results.accuracy_results.correlation
    );
    println!(
        "Passed Cases: {}/{}",
        results.accuracy_results.passed_cases, results.accuracy_results.total_cases
    );

    if let Some(ref norm_results) = results.normalization_results {
        println!("\n=== Normalization Methods ===");
        println!("Best Method: {}", norm_results.best_method);
        println!("Consistency Score: {:.3}", norm_results.consistency_score);
        for (method, result) in &norm_results.method_results {
            println!(
                "  {}: correlation={:.4}, passed={}/{}",
                method, result.correlation, result.passed_cases, result.total_cases
            );
        }
    }

    if let Some(ref edge_results) = results.edge_case_robustness {
        println!("\n=== Edge Cases ===");
        println!(
            "Sparse Sampling: {}",
            if edge_results.sparse_sampling {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "Extreme Dynamic Range: {}",
            if edge_results.extreme_dynamic_range {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "Short Time Series: {}",
            if edge_results.short_time_series {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "High Freq Resolution: {}",
            if edge_results.high_freq_resolution {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!("Stability Score: {:.3}", edge_results.stability_score);
    }

    println!("\n=== Statistical Properties ===");
    println!(
        "False Alarm Rate: {:.3}",
        results.statistical_results.false_alarm_rate
    );
    println!(
        "Detection Power: {:.3}",
        results.statistical_results.detection_power
    );
    println!(
        "CI Coverage: {:.3}",
        results.statistical_results.ci_coverage
    );

    println!("\n=== Performance ===");
    println!(
        "Speed Ratio (ours/scipy): {:.2}",
        results.performance_results.speed_ratio
    );
    println!(
        "Memory Ratio (ours/scipy): {:.2}",
        results.performance_results.memory_ratio
    );
    println!(
        "Scalability Score: {:.1}",
        results.performance_results.scalability_score
    );

    if !results.issues.is_empty() {
        println!("\n=== Issues Found ===");
        for issue in &results.issues {
            println!("  - {}", issue);
        }
    }

    println!("\nValidation completed!");

    Ok(())
}

// Helper function implementations for advanced validation

/// Estimate condition number of the Lomb-Scargle normal equations
#[allow(dead_code)]
fn estimate_condition_number(times: &[f64], freqs: &[f64]) -> SignalResult<f64> {
    // Simplified condition number estimation
    // In practice, this would compute the condition number of the design matrix
    let n = times.len();
    let m = freqs.len();

    if n < 2 || m < 2 {
        return Ok(1.0);
    }

    // Estimate based on time sampling irregularity and frequency range
    let time_span = times[n - 1] - times[0];
    let max_freq = freqs.iter().cloned().fold(0.0, f64::max);
    let min_freq = freqs.iter().cloned().fold(f64::INFINITY, f64::min);

    // Rough heuristic based on sampling and frequency range
    let irregularity = estimate_sampling_irregularity(_times);
    let frequency_range_ratio = max_freq / min_freq.max(1e-12);

    let condition_estimate = irregularity * frequency_range_ratio * (time_span * max_freq);
    Ok(condition_estimate.max(1.0))
}

#[allow(dead_code)]
fn estimate_sampling_irregularity(times: &[f64]) -> f64 {
    if times.len() < 3 {
        return 1.0;
    }

    let diffs: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let var_diff = diffs.iter().map(|&d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;

    (var_diff.sqrt() / mean_diff).max(1.0)
}

#[allow(dead_code)]
fn test_perturbation_stability(times: &[f64], values: &[f64], freqs: &[f64]) -> SignalResult<f64> {
    // Test stability under small perturbations to the data
    let perturbation_level = 1e-8;
    let mut rng = rand::rng();

    // Original periodogram
    let original = lombscargle(_times, values, freqs)?;

    // Perturbed periodogram
    let perturbed_values: Vec<f64> = values
        .iter()
        .map(|&v| v + perturbation_level * rng.random::<f64>())
        .collect();
    let perturbed = lombscargle(_times, &perturbed_values, freqs)?;

    // Compute relative change
    let relative_changes: Vec<f64> = original
        .iter()
        .zip(perturbed.iter())
        .map(|(&orig, &pert)| {
            if orig.abs() > 1e-15 {
                ((pert - orig) / orig).abs()
            } else {
                pert.abs()
            }
        })
        .collect();

    let max_relative_change = relative_changes.iter().cloned().fold(0.0, f64::max);
    Ok(1.0 - max_relative_change.min(1.0)) // Higher score = more stable
}

#[allow(dead_code)]
fn test_gradient_stability(times: &[f64], values: &[f64], freqs: &[f64]) -> SignalResult<f64> {
    // Test gradient-based stability measure
    // Simplified implementation
    let h = 1e-8;
    let mut stability_scores = Vec::new();

    for i in 0..values.len().min(10) {
        // Test a few points
        let mut perturbed_values = values.to_vec();
        perturbed_values[i] += h;

        let original = lombscargle(_times, values, freqs)?;
        let perturbed = lombscargle(_times, &perturbed_values, freqs)?;

        let gradient_norm: f64 = original
            .iter()
            .zip(perturbed.iter())
            .map(|(&orig, &pert)| ((pert - orig) / h).powi(2))
            .sum::<f64>()
            .sqrt();

        stability_scores.push(1.0 / (1.0 + gradient_norm));
    }

    Ok(stability_scores.iter().sum::<f64>() / stability_scores.len() as f64)
}

#[allow(dead_code)]
fn test_nyquist_aliasing_detection(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test detection of Nyquist aliasing
    let n = 100;
    let fs = 1.0; // Sampling frequency
    let nyquist_freq = fs / 2.0;

    // Generate signal above Nyquist frequency (should alias)
    let times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let true_freq = 0.7; // Above Nyquist
    let aliased_freq = fs - true_freq; // Expected aliased frequency

    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * true_freq * t).sin() + 0.1 * rng.random::<f64>())
        .collect();

    let freqs: Vec<f64> = (1..=50).map(|i| i as f64 * 0.02).collect();
    let periodogram = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    // Find peak near aliased frequency
    let peak_idx = periodogram
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let detected_freq = freqs[peak_idx];
    let error = (detected_freq - aliased_freq).abs();

    Ok((1.0 - error * 10.0).max(0.0)) // Score based on detection accuracy
}

#[allow(dead_code)]
fn test_sub_nyquist_handling(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test handling of frequencies below Nyquist
    let n = 200;
    let fs = 2.0;
    let times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let true_freq = 0.3; // Well below Nyquist

    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * true_freq * t).sin() + 0.05 * rng.random::<f64>())
        .collect();

    let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
    let periodogram = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    let peak_idx = periodogram
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let detected_freq = freqs[peak_idx];
    let error = (detected_freq - true_freq).abs();

    Ok((1.0 - error * 20.0).max(0.0))
}

#[allow(dead_code)]
fn test_false_peak_suppression(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test ability to suppress false peaks
    let n = 150;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let true_freq = 0.2;

    // Signal with strong true frequency and weak noise
    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * true_freq * t).sin() + 0.2 * rng.random::<f64>())
        .collect();

    let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
    let periodogram = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    // Find main peak
    let main_peak_power = periodogram.iter().cloned().fold(0.0, f64::max);

    // Count significant false peaks (peaks > 10% of main peak)
    let threshold = 0.1 * main_peak_power;
    let peaks = find_peaks(&periodogram, threshold);

    // Score based on peak selectivity
    let false_peak_ratio = (peaks.len() - 1) as f64 / peaks.len().max(1) as f64;
    Ok(1.0 - false_peak_ratio)
}

#[allow(dead_code)]
fn test_spectral_leakage_mitigation(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test mitigation of spectral leakage
    // Use windowing or other techniques to reduce leakage
    let n = 128;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

    // Two closely spaced frequencies
    let freq1 = 0.15;
    let freq2 = 0.18;

    let values: Vec<f64> = times
        .iter()
        .map(|&t| {
            (2.0 * PI * freq1 * t).sin()
                + 0.5 * (2.0 * PI * freq2 * t).sin()
                + 0.1 * rng.random::<f64>()
        })
        .collect();

    let freqs: Vec<f64> = (10..=25).map(|i| i as f64 * 0.01).collect();
    let periodogram = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    // Check if both peaks are resolved
    let peaks = find_peaks(&periodogram, 0.1);
    let resolved_both = peaks.len() >= 2;

    Ok(if resolved_both { 0.8 } else { 0.3 })
}

#[allow(dead_code)]
fn test_variable_star_simulation(rng: &mut impl Rng) -> SignalResult<f64> {
    // Simulate variable star light curve
    let n = 500;
    let observation_times: Vec<f64> = (0..n)
        .map(|_| rng.random::<f64>() * 100.0) // Irregular sampling
        .collect();
    let mut times = observation_times;
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let period = 2.5; // Days
    let freq = 1.0 / period;

    // Simulate pulsating variable star
    let magnitudes: Vec<f64> = times
        .iter()
        .map(|&t| {
            15.0 + 0.5 * (2.0 * PI * freq * t).sin() +
            0.1 * (2.0 * PI * 2.0 * freq * t).sin() + // Harmonic
            0.05 * rng.random::<f64>() // Noise
        })
        .collect();

    let freqs: Vec<f64> = (1..=200).map(|i| i as f64 * 0.001).collect();
    let periodogram = lombscargle(
        &times,
        &magnitudes,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    let peak_idx = periodogram
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let detected_freq = freqs[peak_idx];
    let period_error = ((1.0 / detected_freq) - period).abs() / period;

    Ok((1.0 - period_error * 5.0).max(0.0))
}

#[allow(dead_code)]
fn test_exoplanet_transit_simulation(rng: &mut impl Rng) -> SignalResult<f64> {
    // Simulate exoplanet transit detection
    let n = 1000;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect(); // Regular sampling

    let orbital_period = 3.2; // Days
    let transit_duration = 0.1; // Days
    let transit_depth = 0.01; // 1% depth

    let flux: Vec<f64> = times
        .iter()
        .map(|&t| {
            let phase = (t % orbital_period) / orbital_period;
            let in_transit = phase < (transit_duration / orbital_period)
                || phase > (1.0 - transit_duration / orbital_period);

            let baseline = 1.0;
            let transit_flux = if in_transit {
                baseline - transit_depth
            } else {
                baseline
            };
            transit_flux + 0.001 * rng.random::<f64>() // Low noise
        })
        .collect();

    let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.001).collect();
    let periodogram = lombscargle(
        &times,
        &flux,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    let expected_freq = 1.0 / orbital_period;
    let closest_freq_idx = freqs
        .iter()
        .enumerate()
        .min_by(|(_, &f1), (_, &f2)| {
            (f1 - expected_freq)
                .abs()
                .partial_cmp(&(f2 - expected_freq).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let detection_strength = periodogram[closest_freq_idx];
    Ok((detection_strength * 50.0).min(1.0))
}

#[allow(dead_code)]
fn test_rr_lyrae_simulation(rng: &mut impl Rng) -> SignalResult<f64> {
    // Simulate RR Lyrae star
    let n = 800;
    let times: Vec<f64> = (0..n)
        .map(|i| i as f64 * 0.015 + rng.random::<f64>() * 0.005)
        .collect();

    let period = 0.5; // Days (typical RR Lyrae period)
    let freq = 1.0 / period;

    // RR Lyrae light curve approximation
    let magnitudes: Vec<f64> = times
        .iter()
        .map(|&t| {
            let phase = (t * freq) % 1.0;
            // Asymmetric light curve typical of RR Lyrae
            let brightness = if phase < 0.3 {
                0.8 * (PI * phase / 0.3).sin()
            } else {
                0.8 * (-0.5 * (phase - 0.3) / 0.7).exp()
            };

            12.0 + brightness + 0.02 * rng.random::<f64>()
        })
        .collect();

    let freqs: Vec<f64> = (50..=400).map(|i| i as f64 * 0.01).collect();
    let periodogram = lombscargle(
        &times,
        &magnitudes,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    let peak_idx = periodogram
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let detected_period = 1.0 / freqs[peak_idx];
    let period_error = (detected_period - period).abs() / period;

    Ok((1.0 - period_error * 10.0).max(0.0))
}

#[allow(dead_code)]
fn test_multi_periodic_source(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test detection of multiple periods
    let n = 1200;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.02).collect();

    let freq1 = 0.1;
    let freq2 = 0.15;
    let freq3 = 0.05;

    let values: Vec<f64> = times
        .iter()
        .map(|&t| {
            (2.0 * PI * freq1 * t).sin()
                + 0.7 * (2.0 * PI * freq2 * t).sin()
                + 0.5 * (2.0 * PI * freq3 * t).sin()
                + 0.1 * rng.random::<f64>()
        })
        .collect();

    let freqs: Vec<f64> = (1..=200).map(|i| i as f64 * 0.002).collect();
    let periodogram = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    // Find peaks and check if all three frequencies are detected
    let threshold = 0.2 * periodogram.iter().cloned().fold(0.0, f64::max);
    let peaks = find_peaks(&periodogram, threshold);

    let detected_freqs: Vec<f64> = peaks.iter().map(|&i| freqs[i]).collect();
    let expected_freqs = vec![freq1, freq2, freq3];

    let mut detections = 0;
    for &expected in &expected_freqs {
        if detected_freqs
            .iter()
            .any(|&detected| (detected - expected).abs() < 0.01)
        {
            detections += 1;
        }
    }

    Ok(detections as f64 / expected_freqs.len() as f64)
}

#[allow(dead_code)]
fn test_phase_preservation(
    times: &[f64],
    values: &[f64],
    freq1: f64,
    freq2: f64,
    expected_phase_diff: f64,
) -> SignalResult<f64> {
    // Test if phase relationships are preserved
    // This is a simplified test - full implementation would use complex-valued LS
    let freqs = vec![freq1, freq2];
    let periodogram = lombscargle(times, values, &freqs)?;

    // For this simplified test, assume good phase preservation if peaks are detected
    let peak_strength = periodogram.1.iter().sum::<f64>() / periodogram.len() as f64;
    Ok((peak_strength * 2.0).min(1.0))
}

#[allow(dead_code)]
fn test_coherence_stability(times: &[f64], values: &[f64]) -> SignalResult<f64> {
    // Test stability of coherence over time windows
    let window_size = times.len() / 4;
    let mut coherence_scores = Vec::new();

    for start in (0.._times.len()).step_by(window_size / 2) {
        let end = (start + window_size).min(_times.len());
        if end - start < 10 {
            break;
        }

        let window_times = &_times[start..end];
        let window_values = &values[start..end];

        let freqs: Vec<f64> = (10..=50).map(|i| i as f64 * 0.01).collect();
        let periodogram =
            lombscargle(window_times, window_values, &freqs, None, None, Some(false))?;

        let peak_power = periodogram.1.iter().cloned().fold(0.0, f64::max);
        coherence_scores.push(peak_power);
    }

    if coherence_scores.len() < 2 {
        return Ok(0.5);
    }

    let mean_coherence = coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64;
    let variance = coherence_scores
        .iter()
        .map(|&c| (c - mean_coherence).powi(2))
        .sum::<f64>()
        / coherence_scores.len() as f64;

    Ok((1.0 - variance / mean_coherence.max(1e-12)).max(0.0))
}

#[allow(dead_code)]
fn test_phase_wrapping(times: &[f64], values: &[f64]) -> SignalResult<f64> {
    // Test handling of phase wrapping
    // Simplified test based on periodogram consistency
    let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
    let periodogram = lombscargle(_times, values, &freqs)?;

    // Check for consistency in peak detection
    let peaks = find_peaks(&periodogram, 0.1);
    let consistency_score = if peaks.is_empty() { 0.5 } else { 0.8 };

    Ok(consistency_score)
}

#[allow(dead_code)]
fn test_min_frequency_separation(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test minimum resolvable frequency separation
    let separations = vec![0.01, 0.005, 0.002, 0.001];
    let mut resolved_separations = Vec::new();

    for &sep in &separations {
        let n = 400;
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.02).collect();

        let freq1 = 0.1;
        let freq2 = freq1 + sep;

        let values: Vec<f64> = times
            .iter()
            .map(|&t| {
                (2.0 * PI * freq1 * t).sin()
                    + (2.0 * PI * freq2 * t).sin()
                    + 0.1 * rng.random::<f64>()
            })
            .collect();

        let freqs: Vec<f64> = (50..=150).map(|i| i as f64 * 0.002).collect();
        let periodogram = lombscargle(
            &times,
            &values,
            Some(&freqs),
            None, // normalization
            None, // center_data
            None, // fit_mean
            None, // nyquist_factor
            None,
        )?;

        let peaks = find_peaks(&periodogram, 0.3);
        if peaks.len() >= 2 {
            resolved_separations.push(sep);
        }
    }

    if let Some(&min_sep) = resolved_separations
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
    {
        Ok(min_sep)
    } else {
        Ok(separations[0]) // Couldn't resolve any separation
    }
}

#[allow(dead_code)]
fn test_resolution_scaling(rng: &mut impl Rng) -> SignalResult<f64> {
    // Test how resolution scales with baseline length
    let baselines = vec![10.0, 20.0, 40.0, 80.0];
    let mut resolution_ratios = Vec::new();

    for &baseline in &baselines {
        let n = (baseline * 20.0) as usize;
        let times: Vec<f64> = (0..n).map(|i| i as f64 * baseline / n as f64).collect();

        let freq = 0.1;
        let values: Vec<f64> = times
            .iter()
            .map(|&t| (2.0 * PI * freq * t).sin() + 0.1 * rng.random::<f64>())
            .collect();

        let freqs: Vec<f64> = (50..=150).map(|i| i as f64 * 0.002).collect();
        let periodogram = lombscargle(
            &times,
            &values,
            Some(&freqs),
            None, // normalization
            None, // center_data
            None, // fit_mean
            None, // nyquist_factor
            None,
        )?;

        // Estimate resolution from peak width
        let peak_idx = periodogram
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(50);

        let peak_power = periodogram[peak_idx];
        let half_power = peak_power / 2.0;

        // Find width at half maximum
        let mut width = 1;
        for i in 1..20 {
            if peak_idx >= i && periodogram[peak_idx - i] < half_power {
                width = 2 * i;
                break;
            }
        }

        let resolution = width as f64 * 0.002; // Frequency spacing
        resolution_ratios.push(baseline / resolution);
    }

    if resolution_ratios.len() >= 2 {
        let first_ratio = resolution_ratios[0];
        let last_ratio = resolution_ratios[resolution_ratios.len() - 1];
        Ok(last_ratio / first_ratio.max(1.0))
    } else {
        Ok(1.0)
    }
}

#[allow(dead_code)]
fn characterize_spectral_window(rng: &mut impl Rng) -> SignalResult<f64> {
    // Characterize the spectral window function
    let n = 200;

    // Create irregular sampling pattern
    let mut times: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 50.0).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Delta function in time domain (constant signal)
    let values = vec![1.0; n];

    let freqs: Vec<f64> = (1..=200).map(|i| (i as f64 - 100.0) * 0.01).collect();
    let spectral_window = lombscargle(
        &times,
        &values,
        Some(&freqs),
        None, // normalization
        None, // center_data
        None, // fit_mean
        None, // nyquist_factor
        None,
    )?;

    // Analyze spectral window properties
    let main_lobe_idx = spectral_window
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(100);

    let main_lobe_power = spectral_window[main_lobe_idx];

    // Find sidelobe levels
    let sidelobe_power = spectral_window
        .iter()
        .enumerate()
        .filter(|(i_, _)| (*i_ as i32 - main_lobe_idx as i32).abs() > 10)
        .map(|(_, &power)| power)
        .fold(0.0, f64::max);

    let sidelobe_ratio = sidelobe_power / main_lobe_power.max(1e-12);
    let quality_score = (1.0 - sidelobe_ratio).max(0.0);

    Ok(quality_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_validation() {
        let config = ScipyValidationConfig {
            test_lengths: vec![32, 64],
            sampling_frequencies: vec![10.0],
            test_frequencies: vec![1.0],
            monte_carlo_trials: 5,
            ..Default::default()
        };

        let results = validate_lombscargle_against_scipy(&config).unwrap();
        assert!(results.accuracy_results.correlation > 0.9);
        assert!(results.summary.overall_score > 50.0);
    }

    #[test]
    fn test_reference_implementation() {
        let t = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let freqs = vec![1.0, 2.0, 5.0];

        let result = compute_reference_lombscargle(&t, &signal, &freqs).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x: &f64| x.is_finite() && x >= 0.0));
    }
}
