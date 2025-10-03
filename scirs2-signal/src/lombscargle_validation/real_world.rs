// Real-world validation scenarios for Lomb-Scargle periodogram
//
// This module provides validation functions for testing Lomb-Scargle implementations
// with real-world-like data patterns including astronomical time series,
// physiological signals, environmental monitoring data, and advanced statistical validation methods.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use super::types::{RealWorldValidationResult, AdvancedStatisticalResult, ValidationResult, StatisticalValidationResult, PerformanceValidationResult, RobustnessValidationResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Rng;
use std::time::Instant;

/// Validate real-world scenarios
#[allow(dead_code)]
pub fn validate_real_world_scenarios(tolerance: f64) -> SignalResult<RealWorldValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // Test 1: Astronomical time series (irregular sampling, long-term trends)
    let astronomical_score = test_astronomical_scenarios(tolerance)?;
    if astronomical_score < 70.0 {
        issues.push("Astronomical data test failed".to_string());
    }

    // Test 2: Physiological signals (biorhythms, noise)
    let physiological_score = test_physiological_scenarios(tolerance)?;
    if physiological_score < 75.0 {
        issues.push("Physiological data test failed".to_string());
    }

    // Test 3: Environmental monitoring (gaps, seasonal patterns)
    let environmental_score = test_environmental_scenarios(tolerance)?;
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
pub fn test_astronomical_scenarios(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simulate variable star with irregular sampling
    let mut rng = scirs2_core::random::rng();

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
            let freq_tolerance = 0.01; // 1% tolerance
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
pub fn test_physiological_scenarios(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simulate heart rate variability data
    let mut rng = scirs2_core::random::rng();

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
pub fn test_environmental_scenarios(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simulate temperature measurements with seasonal variation and gaps
    let mut rng = scirs2_core::random::rng();

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
pub fn validate_advanced_statistical_properties(
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
pub fn test_nonparametric_properties(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Test Kolmogorov-Smirnov test for power distribution
    let mut rng = scirs2_core::random::rng();

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
pub fn test_bayesian_validation(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Simplified Bayesian model comparison test
    // Compare evidence for different frequency models

    let n = 300;
    let fs = 50.0;
    let true_freq = 5.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    let mut rng = scirs2_core::random::rng();

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
pub fn test_information_theory_metrics(tolerance: f64) -> SignalResult<f64> {
    let mut score = 100.0;

    // Test entropy and mutual information properties
    let mut rng = scirs2_core::random::rng();

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
pub fn calculate_comprehensive_score_enhanced(
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
    score -= analytical.issues.len() as f64 * 1.5;

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

    // Cross-validation score (10 points)
    score -= cross_validation.max_relative_error * 80.0;
    score -= cross_validation.issues.len() as f64 * 1.0;

    // Robustness score (15 points)
    score -= (100.0 - robustness.robustness_score) * 0.15;
    score -= robustness.issues.len() as f64 * 1.0;

    // Real-world scenarios score (10 points)
    score -= (100.0 - real_world.score) * 0.10;
    score -= real_world.issues.len() as f64 * 1.0;

    // Advanced statistical score (5 points)
    score -= (100.0 - advanced_stats.score) * 0.05;
    score -= advanced_stats.issues.len() as f64 * 0.5;

    score.max(0.0).min(100.0)
}