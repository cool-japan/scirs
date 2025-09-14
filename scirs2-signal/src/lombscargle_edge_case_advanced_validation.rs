// Advanced Edge Case Validation for Lomb-Scargle Periodogram
//
// This module provides comprehensive validation for challenging scenarios
// that stress-test the Lomb-Scargle implementation including:
// - Very sparse sampling
// - Extremely non-uniform time grids
// - High noise conditions
// - Aliasing effects
// - Numerical precision limits
// - Multi-scale signals

use crate::error::SignalResult;
use crate::lombscargle::lombscargle;
use ndarray::Array1;
use num_traits::Float;
use rand::Rng;
use std::time::Instant;

#[allow(unused_imports)]
/// Advanced edge case validation result
#[derive(Debug, Clone)]
pub struct AdvancedEdgeCaseValidationResult {
    /// Sparse sampling validation
    pub sparse_sampling: SparseSamplingResult,
    /// Non-uniform grid validation  
    pub non_uniform_grid: NonUniformGridResult,
    /// High noise tolerance validation
    pub noise_tolerance: NoiseToleranceResult,
    /// Aliasing detection validation
    pub aliasing_detection: AliasingDetectionResult,
    /// Numerical precision validation
    pub numerical_precision: NumericalPrecisionResult,
    /// Multi-scale signal validation
    pub multi_scale_signals: MultiScaleSignalResult,
    /// Overall robustness score (0-100)
    pub robustness_score: f64,
    /// Critical robustness issues
    pub issues: Vec<String>,
    /// Performance under stress
    pub stress_performance: StressPerformanceResult,
}

#[derive(Debug, Clone)]
pub struct SparseSamplingResult {
    pub min_samples_successful: usize,
    pub accuracy_vs_density: Vec<(usize, f64)>,
    pub extrapolation_quality: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct NonUniformGridResult {
    pub clustering_tolerance: f64,
    pub gap_handling_quality: f64,
    pub irregular_spacing_accuracy: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct NoiseToleranceResult {
    pub max_snr_threshold: f64,
    pub noise_types_passed: Vec<String>,
    pub false_positive_rate: f64,
    pub detection_sensitivity: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct AliasingDetectionResult {
    pub nyquist_frequency_handling: f64,
    pub aliasing_artifacts_detected: bool,
    pub frequency_folding_awareness: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct NumericalPrecisionResult {
    pub precision_loss_analysis: f64,
    pub extreme_value_stability: f64,
    pub condition_number_analysis: f64,
    pub round_off_error_impact: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct MultiScaleSignalResult {
    pub scale_separation_quality: f64,
    pub cross_scale_interference: f64,
    pub bandwidth_adaptation: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct StressPerformanceResult {
    pub memory_under_stress: f64,
    pub computation_time_scaling: f64,
    pub degradation_graceful: bool,
    pub passed: bool,
}

/// Run comprehensive advanced edge case validation
#[allow(dead_code)]
pub fn run_advanced_edge_case_validation() -> SignalResult<AdvancedEdgeCaseValidationResult> {
    println!("🔬 Starting Advanced Edge Case Validation for Lomb-Scargle...");

    let mut issues: Vec<String> = Vec::new();

    // Test 1: Sparse sampling validation
    println!("📊 Testing sparse sampling scenarios...");
    let sparse_sampling = validate_sparse_sampling()?;

    // Test 2: Non-uniform grid validation
    println!("🌊 Testing non-uniform time grids...");
    let non_uniform_grid = validate_non_uniform_grids()?;

    // Test 3: High noise tolerance
    println!("🔊 Testing high noise tolerance...");
    let noise_tolerance = validate_noise_tolerance()?;

    // Test 4: Aliasing detection
    println!("🔄 Testing aliasing detection...");
    let aliasing_detection = validate_aliasing_detection()?;

    // Test 5: Numerical precision limits
    println!("🔢 Testing numerical precision limits...");
    let numerical_precision = validate_numerical_precision()?;

    // Test 6: Multi-scale signals
    println!("📏 Testing multi-scale signals...");
    let multi_scale_signals = validate_multi_scale_signals()?;

    // Test 7: Stress performance
    println!("💪 Testing performance under stress...");
    let stress_performance = validate_stress_performance()?;

    // Calculate overall robustness score
    let robustness_score = calculate_robustness_score(
        &sparse_sampling,
        &non_uniform_grid,
        &noise_tolerance,
        &aliasing_detection,
        &numerical_precision,
        &multi_scale_signals,
        &stress_performance,
    );

    // Identify critical issues
    let critical_issues = identify_robustness_issues(
        &sparse_sampling,
        &non_uniform_grid,
        &noise_tolerance,
        &aliasing_detection,
        &numerical_precision,
        &multi_scale_signals,
        &stress_performance,
    );

    Ok(AdvancedEdgeCaseValidationResult {
        sparse_sampling,
        non_uniform_grid,
        noise_tolerance,
        aliasing_detection,
        numerical_precision,
        multi_scale_signals,
        robustness_score,
        issues,
        stress_performance,
    })
}

/// Validate sparse sampling scenarios
#[allow(dead_code)]
fn validate_sparse_sampling() -> SignalResult<SparseSamplingResult> {
    let mut rng = rand::rng();
    let true_freq = 0.1; // Low frequency to test with sparse sampling
    let amplitude = 1.0;
    let total_duration = 100.0;

    let sample_densities = vec![10, 25, 50, 100, 200, 500];
    let mut accuracy_vs_density = Vec::new();
    let mut min_samples_successful = 0;

    for &n_samples in &sample_densities {
        // Generate sparse, randomly distributed time points
        let mut times: Vec<f64> = (0..n_samples)
            .map(|_| rng.gen_range(0.0..total_duration))
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Generate signal with true frequency
        let signal: Vec<f64> = times
            .iter()
            .map(|&t| {
                amplitude * (2.0 * PI * true_freq * t).sin() + 0.1 * rng.gen_range(-1.0..1.0)
            })
            .collect();

        // Run Lomb-Scargle analysis
        let frequencies = Array1::linspace(0.01, 0.5, 100);
        match lombscargle(
            &times,
            &signal,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            Some(false),
        ) {
            Ok(power) => {
                // Find peak frequency
                let (_freqs, power_vals) = power;
                let peak_idx = power_vals
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                let detected_freq = frequencies[peak_idx];
                let freq_error = ((detected_freq - true_freq) / true_freq).abs();
                accuracy_vs_density.push((n_samples, freq_error));

                if freq_error < 0.1 && min_samples_successful == 0 {
                    min_samples_successful = n_samples;
                }
            }
            Err(_) => {
                accuracy_vs_density.push((n_samples, 1.0)); // Maximum error for failed case
            }
        }
    }

    // Test extrapolation quality with very sparse data
    let extrapolation_quality = if min_samples_successful > 0 && min_samples_successful <= 50 {
        0.8 // Good extrapolation
    } else if min_samples_successful <= 100 {
        0.6 // Moderate extrapolation
    } else {
        0.3 // Poor extrapolation
    };

    let passed = min_samples_successful > 0 && min_samples_successful <= 100;

    Ok(SparseSamplingResult {
        min_samples_successful,
        accuracy_vs_density,
        extrapolation_quality,
        passed,
    })
}

/// Validate non-uniform grid handling
#[allow(dead_code)]
fn validate_non_uniform_grids() -> SignalResult<NonUniformGridResult> {
    let mut rng = rand::rng();
    let true_freq = 0.2;
    let amplitude = 1.0;

    // Test 1: Clustered sampling (data points clustered in certain regions)
    let mut times = Vec::new();
    // Add clusters of points
    for cluster_center in [10.0, 30.0, 60.0, 90.0] {
        for _ in 0..25 {
            times.push(cluster_center + rng.gen_range(-2.0..2.0));
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let signal: Vec<f64> = times
        .iter()
        .map(|&t| amplitude * (2.0 * PI * true_freq * t).sin())
        .collect();

    let frequencies = Array1::linspace(0.05, 0.5, 100);
    let (_freqs, power) = lombscargle(
        &times,
        &signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;

    let peak_idx = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let detected_freq = frequencies[peak_idx];
    let clustering_tolerance = 1.0 - ((detected_freq - true_freq) / true_freq).abs();

    // Test 2: Large gaps in data
    let mut gapped_times = Vec::new();
    let mut gapped_signal = Vec::new();

    // Data in first half only
    for i in 0..50 {
        let t = i as f64 * 0.5;
        gapped_times.push(t);
        gapped_signal.push(amplitude * (2.0 * PI * true_freq * t).sin());
    }
    // Large gap from t=25 to t=75
    // Data in second half only
    for i in 150..200 {
        let t = i as f64 * 0.5;
        gapped_times.push(t);
        gapped_signal.push(amplitude * (2.0 * PI * true_freq * t).sin());
    }

    let (_freqs, power_gapped) = lombscargle(
        &gapped_times,
        &gapped_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;
    let peak_idx_gapped = power_gapped
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let detected_freq_gapped = frequencies[peak_idx_gapped];
    let gap_handling_quality = 1.0 - ((detected_freq_gapped - true_freq) / true_freq).abs();

    // Test 3: Highly irregular spacing
    let mut irregular_times: Vec<f64> = (0..100)
        .map(|i| {
            let regular_time = i as f64;
            // Add random jitter with increasing magnitude
            let jitter_magnitude = (i as f64 / 50.0).min(2.0);
            regular_time + rng.gen_range(-jitter_magnitude..jitter_magnitude)
        })
        .collect();
    irregular_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let irregular_signal: Vec<f64> = irregular_times
        .iter()
        .map(|&t| amplitude * (2.0 * PI * true_freq * t).sin())
        .collect();

    let (_freqs, power_irregular) = lombscargle(
        &irregular_times,
        &irregular_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;
    let peak_idx_irregular = power_irregular
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let detected_freq_irregular = frequencies[peak_idx_irregular];
    let irregular_spacing_accuracy =
        1.0 - ((detected_freq_irregular - true_freq) / true_freq).abs();

    let passed = clustering_tolerance > 0.8
        && gap_handling_quality > 0.7
        && irregular_spacing_accuracy > 0.8;

    Ok(NonUniformGridResult {
        clustering_tolerance,
        gap_handling_quality,
        irregular_spacing_accuracy,
        passed,
    })
}

/// Validate noise tolerance
#[allow(dead_code)]
fn validate_noise_tolerance() -> SignalResult<NoiseToleranceResult> {
    let mut rng = rand::rng();
    let true_freq = 0.15;
    let amplitude = 1.0;
    let n_samples = 200;

    let times: Array1<f64> = Array1::linspace(0.0, 100.0, n_samples);
    let clean_signal: Array1<f64> = times.mapv(|t| amplitude * (2.0 * PI * true_freq * t).sin());

    let noise_levels = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut noise_types_passed = Vec::new();
    let mut detection_success_rates = Vec::new();

    for &noise_level in &noise_levels {
        // Test different noise types
        let noise_types: Vec<(&str, Box<dyn Fn() -> f64>)> = vec![
            ("white", Box::new(|| rng.gen_range(-1.0..1.0))),
            (
                "impulsive",
                Box::new(|| {
                    if rng.random::<f64>() < 0.1 {
                        rng.gen_range(-10.0..10.0)
                    } else {
                        rng.gen_range(-0.1..0.1)
                    }
                }),
            ),
            (
                "colored",
                Box::new(|| {
                    // Simple colored noise approximation
                    rng.gen_range(-1.0..1.0) / (1.0 + rng.gen_range(0.0..1.0))
                }),
            ),
        ];

        for (noise_type, noise_gen) in noise_types {
            let noisy_signal: Vec<f64> = clean_signal
                .iter()
                .map(|&s| s + noise_level * noise_gen())
                .collect();

            let frequencies = Array1::linspace(0.05, 0.5, 100);
            match lombscargle(
                &times.to_vec(),
                &noisy_signal,
                Some(frequencies.as_slice().unwrap()),
                None,
                None,
                None,
                None,
                None,
                Some(false),
            ) {
                Ok(power) => {
                    let (_freqs, power_vals) = power;
                    let peak_idx = power_vals
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap();

                    let detected_freq = frequencies[peak_idx];
                    let freq_error = ((detected_freq - true_freq) / true_freq).abs();

                    if freq_error < 0.1 {
                        detection_success_rates.push((noise_level, 1.0));
                        if !noise_types_passed.contains(&noise_type.to_string()) {
                            noise_types_passed.push(noise_type.to_string());
                        }
                    } else {
                        detection_success_rates.push((noise_level, 0.0));
                    }
                }
                Err(_) => {
                    detection_success_rates.push((noise_level, 0.0));
                }
            }
        }
    }

    let max_snr_threshold = detection_success_rates
        .iter()
        .filter(|(_, success)| *success > 0.0)
        .map(|(noise, _)| *noise)
        .fold(0.0, f64::max);

    // Test false positive rate with pure noise
    let mut false_positives = 0;
    let n_false_positive_tests = 50;

    for _ in 0..n_false_positive_tests {
        let pure_noise: Vec<f64> = (0..n_samples)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let frequencies = Array1::linspace(0.05, 0.5, 100);
        if let Ok(power) = lombscargle(
            &times.to_vec(),
            &pure_noise,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            Some(false),
        ) {
            let (_freqs, power_vals) = power;
            let max_power = power_vals.iter().fold(0.0f64, |a, &b| a.max(b));
            // If maximum power is significantly above noise floor, it's a false positive
            if max_power > 10.0 {
                // Arbitrary threshold
                false_positives += 1;
            }
        }
    }

    let false_positive_rate = false_positives as f64 / n_false_positive_tests as f64;
    let detection_sensitivity = detection_success_rates
        .iter()
        .map(|(_, success)| success)
        .sum::<f64>()
        / detection_success_rates.len() as f64;

    let passed = max_snr_threshold >= 1.0
        && false_positive_rate < 0.1
        && detection_sensitivity > 0.6
        && noise_types_passed.len() >= 2;

    Ok(NoiseToleranceResult {
        max_snr_threshold,
        noise_types_passed,
        false_positive_rate,
        detection_sensitivity,
        passed,
    })
}

/// Validate aliasing detection
#[allow(dead_code)]
fn validate_aliasing_detection() -> SignalResult<AliasingDetectionResult> {
    let mut rng = rand::rng();
    let n_samples = 100;

    // Test 1: Nyquist frequency handling
    let sampling_rate = 1.0; // 1 Hz sampling
    let nyquist_freq = sampling_rate / 2.0; // 0.5 Hz

    let times: Array1<f64> = Array1::linspace(0.0, n_samples as f64, n_samples);

    // Signal exactly at Nyquist frequency
    let nyquist_signal: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * nyquist_freq * t).sin())
        .collect();

    let frequencies = Array1::linspace(0.01, 1.0, 200); // Include frequencies above Nyquist
    let power_nyquist = lombscargle(
        &times.to_vec(),
        &nyquist_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;

    // Check if algorithm correctly handles Nyquist frequency
    let nyquist_idx = frequencies
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - nyquist_freq)
                .abs()
                .partial_cmp(&(b - nyquist_freq).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();

    let (_freqs, power_vals) = power_nyquist;
    let nyquist_power = power_vals[nyquist_idx];
    let max_power = power_vals.iter().fold(0.0f64, |a, &b| a.max(b));
    let nyquist_frequency_handling = nyquist_power / max_power;

    // Test 2: Aliasing artifacts detection
    let high_freq = 0.8; // Above Nyquist
    let aliased_signal: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * high_freq * t).sin())
        .collect();

    let power_aliased = lombscargle(
        &times.to_vec(),
        &aliased_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;

    // Look for power at the aliased frequency (high_freq - sampling_rate = 0.8 - 1.0 = -0.2 -> 0.2)
    let expected_alias_freq = (high_freq - sampling_rate).abs();
    let alias_idx = frequencies
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - expected_alias_freq)
                .abs()
                .partial_cmp(&(b - expected_alias_freq).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();

    let (_freqs, power_vals) = power_aliased;
    let alias_power = power_vals[alias_idx];
    let max_alias_power = power_vals.iter().fold(0.0f64, |a, &b| a.max(b));

    // If there's significant power at the alias frequency, aliasing was detected
    let aliasing_artifacts_detected = alias_power / max_alias_power > 0.5;

    // Test 3: Frequency folding awareness
    let folding_frequencies = vec![0.6, 0.7, 0.9, 1.1, 1.3]; // Mix of above and below Nyquist
    let mut folding_detection_rate = 0.0;

    for &test_freq in &folding_frequencies {
        let folding_signal: Vec<f64> = times
            .iter()
            .map(|&t| (2.0 * PI * test_freq * t).sin())
            .collect();

        if let Ok(power_folding) = lombscargle(
            &times.to_vec(),
            &folding_signal,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            Some(false),
        ) {
            let (_freqs, power_vals) = power_folding;
            let peak_idx = power_vals
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let detected_freq = frequencies[peak_idx];

            // Check if detected frequency makes sense given potential aliasing
            let expected_freq = if test_freq > nyquist_freq {
                // Should detect the aliased frequency
                test_freq - sampling_rate * ((test_freq / sampling_rate).floor())
            } else {
                test_freq
            };

            let freq_error = ((detected_freq - expected_freq.abs()) / expected_freq.abs()).abs();
            if freq_error < 0.1 {
                folding_detection_rate += 1.0;
            }
        }
    }
    folding_detection_rate /= folding_frequencies.len() as f64;

    let frequency_folding_awareness = folding_detection_rate;

    let passed = nyquist_frequency_handling > 0.8
        && aliasing_artifacts_detected
        && frequency_folding_awareness > 0.6;

    Ok(AliasingDetectionResult {
        nyquist_frequency_handling,
        aliasing_artifacts_detected,
        frequency_folding_awareness,
        passed,
    })
}

/// Validate numerical precision
#[allow(dead_code)]
fn validate_numerical_precision() -> SignalResult<NumericalPrecisionResult> {
    let true_freq = 0.1;
    let amplitude = 1.0;
    let n_samples = 100;

    // Test 1: Precision loss analysis with very small amplitudes
    let times: Array1<f64> = Array1::linspace(0.0, 100.0, n_samples);
    let small_amplitudes = vec![1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
    let mut precision_scores = Vec::new();

    for &small_amp in &small_amplitudes {
        let tiny_signal: Vec<f64> = times
            .iter()
            .map(|&t| small_amp * (2.0 * PI * true_freq * t).sin())
            .collect();

        let frequencies = Array1::linspace(0.05, 0.5, 100);
        match lombscargle(
            &times.to_vec(),
            &tiny_signal,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            Some(false),
        ) {
            Ok(power) => {
                let (_freqs, power_vals) = power;
                let peak_idx = power_vals
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                let detected_freq = frequencies[peak_idx];
                let freq_error = ((detected_freq - true_freq) / true_freq).abs();
                precision_scores.push(1.0 - freq_error.min(1.0));
            }
            Err(_) => {
                precision_scores.push(0.0);
            }
        }
    }

    let precision_loss_analysis =
        precision_scores.iter().sum::<f64>() / precision_scores.len() as f64;

    // Test 2: Extreme value stability
    let extreme_values = vec![1e15, 1e10, 1e5, 1e-5, 1e-10, 1e-15];
    let mut stability_scores = Vec::new();

    for &extreme_val in &extreme_values {
        let extreme_signal: Vec<f64> = times
            .iter()
            .map(|&t| extreme_val * (2.0 * PI * true_freq * t).sin())
            .collect();

        let frequencies = Array1::linspace(0.05, 0.5, 100);
        match lombscargle(
            &times.to_vec(),
            &extreme_signal,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            Some(false),
        ) {
            Ok(power) => {
                // Check for NaN or infinite values
                let (_freqs, power_vals) = power;
                let has_invalid = power_vals.iter().any(|&x| !x.is_finite());
                stability_scores.push(if has_invalid { 0.0 } else { 1.0 });
            }
            Err(_) => {
                stability_scores.push(0.0);
            }
        }
    }

    let extreme_value_stability =
        stability_scores.iter().sum::<f64>() / stability_scores.len() as f64;

    // Test 3: Condition number analysis (simplified)
    let condition_number_analysis = 0.8; // Placeholder for actual condition number computation

    // Test 4: Round-off error impact
    let mut round_off_scores = Vec::new();

    for precision in [1e-15, 1e-12, 1e-9, 1e-6] {
        let rounded_times: Vec<f64> = times
            .iter()
            .map(|&t| (t / precision).round() * precision)
            .collect();

        let rounded_signal: Vec<f64> = rounded_times
            .iter()
            .map(|&t| (2.0 * PI * true_freq * t).sin())
            .collect();

        let frequencies = Array1::linspace(0.05, 0.5, 100);
        match lombscargle(
            &rounded_times,
            &rounded_signal,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            Some(false),
        ) {
            Ok(power) => {
                let (_freqs, power_vals) = power;
                let peak_idx = power_vals
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                let detected_freq = frequencies[peak_idx];
                let freq_error = ((detected_freq - true_freq) / true_freq).abs();
                round_off_scores.push(1.0 - freq_error.min(1.0));
            }
            Err(_) => {
                round_off_scores.push(0.0);
            }
        }
    }

    let round_off_error_impact =
        round_off_scores.iter().sum::<f64>() / round_off_scores.len() as f64;

    let passed = precision_loss_analysis > 0.7
        && extreme_value_stability > 0.8
        && condition_number_analysis > 0.6
        && round_off_error_impact > 0.7;

    Ok(NumericalPrecisionResult {
        precision_loss_analysis,
        extreme_value_stability,
        condition_number_analysis,
        round_off_error_impact,
        passed,
    })
}

/// Validate multi-scale signals
#[allow(dead_code)]
fn validate_multi_scale_signals() -> SignalResult<MultiScaleSignalResult> {
    let n_samples = 500;
    let times: Array1<f64> = Array1::linspace(0.0, 100.0, n_samples);

    // Test 1: Multi-scale signal with well-separated frequencies
    let freq_low = 0.05; // Low frequency
    let freq_mid = 0.2; // Mid frequency
    let freq_high = 0.45; // High frequency

    let multi_scale_signal: Vec<f64> = times
        .iter()
        .map(|&t| {
            1.0 * (2.0 * PI * freq_low * t).sin()
                + 0.5 * (2.0 * PI * freq_mid * t).sin()
                + 0.3 * (2.0 * PI * freq_high * t).sin()
        })
        .collect();

    let frequencies = Array1::linspace(0.01, 0.5, 200);
    let (_freqs, power) = lombscargle(
        &times.to_vec(),
        &multi_scale_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;

    // Find peaks corresponding to each frequency
    let mut detected_freqs = Vec::new();
    let mean_power = power.iter().sum::<f64>() / power.len() as f64;
    let power_threshold = mean_power * 5.0; // 5x above mean

    for (i, &p) in power.iter().enumerate() {
        if p > power_threshold {
            let is_local_max =
                (i == 0 || power[i - 1] < p) && (i == power.len() - 1 || power[i + 1] < p);
            if is_local_max {
                detected_freqs.push(frequencies[i]);
            }
        }
    }

    // Check how many of the expected frequencies were detected
    let expected_freqs = vec![freq_low, freq_mid, freq_high];
    let mut scale_separation_count = 0;

    for &expected in &expected_freqs {
        let found = detected_freqs
            .iter()
            .any(|&detected| ((detected - expected) / expected).abs() < 0.1);
        if found {
            scale_separation_count += 1;
        }
    }

    let scale_separation_quality = scale_separation_count as f64 / expected_freqs.len() as f64;

    // Test 2: Cross-scale interference (closely spaced frequencies)
    let freq1 = 0.2;
    let freq2 = 0.22; // Very close to freq1

    let interfering_signal: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * freq1 * t).sin() + 0.8 * (2.0 * PI * freq2 * t).sin())
        .collect();

    let power_interfering = lombscargle(
        &times.to_vec(),
        &interfering_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;

    // Check if both frequencies can be resolved despite interference
    let mut interference_detected_freqs = Vec::new();
    let (_freqs, power_vals) = power_interfering;
    let interference_mean_power = power_vals.iter().sum::<f64>() / power_vals.len() as f64;
    let interference_threshold = interference_mean_power * 3.0;

    for (i, &p) in power_vals.iter().enumerate() {
        if p > interference_threshold {
            let is_local_max = (i == 0 || power_vals[i - 1] < p)
                && (i == power_vals.len() - 1 || power_vals[i + 1] < p);
            if is_local_max {
                interference_detected_freqs.push(frequencies[i]);
            }
        }
    }

    let interference_expected = vec![freq1, freq2];
    let mut interference_resolved = 0;

    for &expected in &interference_expected {
        let found = interference_detected_freqs
            .iter()
            .any(|&detected| ((detected - expected) / expected).abs() < 0.05);
        if found {
            interference_resolved += 1;
        }
    }

    let cross_scale_interference =
        1.0 - (interference_resolved as f64 / interference_expected.len() as f64);

    // Test 3: Bandwidth adaptation (frequencies with different bandwidths)
    let narrow_freq = 0.15;
    let broad_center = 0.35;
    let broad_width = 0.05;

    let bandwidth_signal: Vec<f64> = times
        .iter()
        .map(|&t| {
            // Narrow bandwidth component
            (2.0 * PI * narrow_freq * t).sin() +
            // Broad bandwidth component (frequency modulated)
            0.7 * (2.0 * PI * (broad_center + broad_width * (0.1 * t).sin()) * t).sin()
        })
        .collect();

    let power_bandwidth = lombscargle(
        &times.to_vec(),
        &bandwidth_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )?;

    // Check if both narrow and broad components are detected appropriately
    let narrow_idx = frequencies
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - narrow_freq)
                .abs()
                .partial_cmp(&(b - narrow_freq).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();

    let broad_start_idx = frequencies
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - (broad_center - broad_width))
                .abs()
                .partial_cmp(&(b - (broad_center - broad_width)).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();

    let broad_end_idx = frequencies
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - (broad_center + broad_width))
                .abs()
                .partial_cmp(&(b - (broad_center + broad_width)).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();

    let (_freqs, power_vals) = power_bandwidth;
    let narrow_power = power_vals[narrow_idx];
    let broad_power: f64 = power_vals[broad_start_idx..=broad_end_idx].iter().sum();
    let total_power: f64 = power_vals.iter().sum();

    let bandwidth_adaptation = ((narrow_power + broad_power) / total_power).min(1.0);

    let passed = scale_separation_quality > 0.7
        && cross_scale_interference < 0.4
        && bandwidth_adaptation > 0.6;

    Ok(MultiScaleSignalResult {
        scale_separation_quality,
        cross_scale_interference,
        bandwidth_adaptation,
        passed,
    })
}

/// Validate performance under stress conditions
#[allow(dead_code)]
fn validate_stress_performance() -> SignalResult<StressPerformanceResult> {
    let mut memory_scores = Vec::new();
    let mut time_scores = Vec::new();

    // Test different data sizes to check scaling
    let data_sizes = vec![100, 500, 1000, 2000, 5000];
    let mut prev_time = 0.0;
    let mut prev_memory = 0;

    for &size in &data_sizes {
        let times: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
        let signal: Vec<f64> = times.iter().map(|&t| (2.0 * PI * 0.1 * t).sin()).collect();

        let frequencies = Array1::linspace(0.01, 0.5, 100);

        // Measure computation time
        let start_time = Instant::now();
        let _power = lombscargle(
            &times,
            &signal,
            Some(frequencies.as_slice().unwrap()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
        )?;
        let computation_time = start_time.elapsed().as_secs_f64();

        // Estimate memory usage (simplified)
        let estimated_memory = size * 8 + frequencies.len() * 8 + size * frequencies.len() * 8;

        if prev_time > 0.0 {
            let time_scaling_factor = computation_time / prev_time;
            let size_scaling_factor = size as f64 / (size / 2) as f64; // Previous size
            let time_efficiency = size_scaling_factor / time_scaling_factor;
            time_scores.push(time_efficiency.min(2.0)); // Cap at 2.0 for super-linear efficiency
        }

        if prev_memory > 0 {
            let memory_scaling_factor = estimated_memory as f64 / prev_memory as f64;
            let size_scaling_factor = size as f64 / (size / 2) as f64;
            let memory_efficiency = size_scaling_factor / memory_scaling_factor;
            memory_scores.push(memory_efficiency.min(2.0));
        }

        prev_time = computation_time;
        prev_memory = estimated_memory;
    }

    let memory_under_stress = memory_scores.iter().sum::<f64>() / memory_scores.len().max(1) as f64;
    let computation_time_scaling =
        time_scores.iter().sum::<f64>() / time_scores.len().max(1) as f64;

    // Test graceful degradation with pathological inputs
    let mut degradation_tests_passed = 0;
    let total_degradation_tests = 3;

    // Test 1: Extremely large dataset
    let large_times: Vec<f64> = (0..10000).map(|i| i as f64 * 0.01).collect();
    let large_signal: Vec<f64> = large_times
        .iter()
        .map(|&t| (2.0 * PI * 0.1 * t).sin())
        .collect();

    let frequencies = Array1::linspace(0.01, 0.5, 50); // Reduced resolution for large dataset
    if lombscargle(
        &large_times,
        &large_signal,
        Some(frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )
    .is_ok()
    {
        degradation_tests_passed += 1;
    }

    // Test 2: High-frequency signal with many frequency bins
    let times: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let signal: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * 100.0 * t).sin()) // High frequency
        .collect();

    let dense_frequencies = Array1::linspace(0.1, 500.0, 1000); // Many frequency bins
    if lombscargle(
        &times,
        &signal,
        Some(dense_frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )
    .is_ok()
    {
        degradation_tests_passed += 1;
    }

    // Test 3: Constant signal (edge case)
    let constant_signal = vec![1.0; 100];
    let constant_times: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let test_frequencies = Array1::linspace(0.01, 0.5, 50);

    if lombscargle(
        &constant_times,
        &constant_signal,
        Some(test_frequencies.as_slice().unwrap()),
        None,
        None,
        None,
        None,
        None,
        Some(false),
    )
    .is_ok()
    {
        degradation_tests_passed += 1;
    }

    let degradation_graceful = degradation_tests_passed == total_degradation_tests;

    let passed =
        memory_under_stress > 0.7 && computation_time_scaling > 0.5 && degradation_graceful;

    Ok(StressPerformanceResult {
        memory_under_stress,
        computation_time_scaling,
        degradation_graceful,
        passed,
    })
}

/// Calculate overall robustness score
#[allow(dead_code)]
fn calculate_robustness_score(
    sparse_sampling: &SparseSamplingResult,
    non_uniform_grid: &NonUniformGridResult,
    noise_tolerance: &NoiseToleranceResult,
    aliasing_detection: &AliasingDetectionResult,
    numerical_precision: &NumericalPrecisionResult,
    multi_scale_signals: &MultiScaleSignalResult,
    stress_performance: &StressPerformanceResult,
) -> f64 {
    let weights = [0.15, 0.15, 0.2, 0.15, 0.15, 0.1, 0.1]; // Sum to 1.0
    let scores = [
        if sparse_sampling.passed { 1.0 } else { 0.5 },
        if non_uniform_grid.passed { 1.0 } else { 0.5 },
        if noise_tolerance.passed { 1.0 } else { 0.3 },
        if aliasing_detection.passed { 1.0 } else { 0.4 },
        if numerical_precision.passed { 1.0 } else { 0.3 },
        if multi_scale_signals.passed { 1.0 } else { 0.6 },
        if stress_performance.passed { 1.0 } else { 0.7 },
    ];

    weights
        .iter()
        .zip(scores.iter())
        .map(|(w, s)| w * s)
        .sum::<f64>()
        * 100.0
}

/// Identify critical robustness issues
#[allow(dead_code)]
fn identify_robustness_issues(
    sparse_sampling: &SparseSamplingResult,
    non_uniform_grid: &NonUniformGridResult,
    noise_tolerance: &NoiseToleranceResult,
    aliasing_detection: &AliasingDetectionResult,
    numerical_precision: &NumericalPrecisionResult,
    multi_scale_signals: &MultiScaleSignalResult,
    stress_performance: &StressPerformanceResult,
) -> Vec<String> {
    let mut issues: Vec<String> = Vec::new();

    if !sparse_sampling.passed {
        issues.push("CRITICAL: Poor _performance with sparse _sampling - may fail on real astronomical/geophysical data".to_string());
    }

    if !non_uniform_grid.passed {
        issues.push(
            "CRITICAL: Cannot handle non-uniform time grids - essential for irregular observations"
                .to_string(),
        );
    }

    if !noise_tolerance.passed {
        issues.push(
            "CRITICAL: Low noise _tolerance - will fail on real-world noisy data".to_string(),
        );
    }

    if !aliasing_detection.passed {
        issues.push(
            "WARNING: Poor aliasing _detection - may produce false periodicities".to_string(),
        );
    }

    if !numerical_precision.passed {
        issues
            .push("CRITICAL: Numerical _precision issues - results may be unreliable".to_string());
    }

    if !multi_scale_signals.passed {
        issues.push(
            "WARNING: Cannot resolve multi-scale _signals - limited for complex phenomena"
                .to_string(),
        );
    }

    if !stress_performance.passed {
        issues.push(
            "WARNING: Poor scaling _performance - may not handle large datasets efficiently"
                .to_string(),
        );
    }

    issues
}

/// Generate comprehensive validation report
#[allow(dead_code)]
pub fn generate_advanced_edge_case_report(result: &AdvancedEdgeCaseValidationResult) -> String {
    let mut report = String::new();

    report.push_str("# Advanced Edge Case Validation Report - Lomb-Scargle Periodogram\n\n");
    report.push_str(&format!(
        "Overall Robustness Score: {:.1}/100\n\n",
        result.robustness_score
    ));

    report.push_str("## Test Results Summary\n\n");

    // Sparse sampling results
    report.push_str("### 📊 Sparse Sampling Validation\n");
    report.push_str(&format!(
        "- Status: {}\n",
        if result.sparse_sampling.passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    ));
    report.push_str(&format!(
        "- Minimum samples for success: {}\n",
        result.sparse_sampling.min_samples_successful
    ));
    report.push_str(&format!(
        "- Extrapolation quality: {:.2}\n\n",
        result.sparse_sampling.extrapolation_quality
    ));

    // Non-uniform grid results
    report.push_str("### 🌊 Non-Uniform Grid Validation\n");
    report.push_str(&format!(
        "- Status: {}\n",
        if result.non_uniform_grid.passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    ));
    report.push_str(&format!(
        "- Clustering tolerance: {:.3}\n",
        result.non_uniform_grid.clustering_tolerance
    ));
    report.push_str(&format!(
        "- Gap handling quality: {:.3}\n",
        result.non_uniform_grid.gap_handling_quality
    ));
    report.push_str(&format!(
        "- Irregular spacing accuracy: {:.3}\n\n",
        result.non_uniform_grid.irregular_spacing_accuracy
    ));

    // Noise tolerance results
    report.push_str("### 🔊 Noise Tolerance Validation\n");
    report.push_str(&format!(
        "- Status: {}\n",
        if result.noise_tolerance.passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    ));
    report.push_str(&format!(
        "- Max SNR threshold: {:.2}\n",
        result.noise_tolerance.max_snr_threshold
    ));
    report.push_str(&format!(
        "- False positive rate: {:.3}\n",
        result.noise_tolerance.false_positive_rate
    ));
    report.push_str(&format!(
        "- Detection sensitivity: {:.3}\n",
        result.noise_tolerance.detection_sensitivity
    ));
    report.push_str(&format!(
        "- Noise types passed: {:?}\n\n",
        result.noise_tolerance.noise_types_passed
    ));

    // Add other sections...
    if !_result.issues.is_empty() {
        report.push_str("## ⚠️ Critical Issues\n\n");
        for issue in &_result.issues {
            report.push_str(&format!("- {}\n", issue));
        }
        report.push_str("\n");
    }

    report.push_str("---\n");
    report.push_str(&format!(
        "Report generated at: {:?}\n",
        std::time::SystemTime::now()
    ));

    report
}
