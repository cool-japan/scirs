//! Statistical properties validation
//!
//! This module validates statistical properties of the Lomb-Scargle implementation
//! including false alarm rates, detection power, and confidence intervals.

use super::types::*;
use crate::error::SignalResult;
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::PI;

/// Validate statistical properties
#[allow(dead_code)]
pub fn validate_statistical_properties(
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
pub fn estimate_false_alarm_rate(config: &ScipyValidationConfig) -> SignalResult<f64> {
    let mut false_alarms = 0;
    let trials = config.monte_carlo_trials.min(50); // Limit for performance

    for _ in 0..trials {
        // Generate pure noise
        let mut rng = scirs2_core::random::rng();
        let n = 100;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let signal: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

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
pub fn estimate_detection_power(config: &ScipyValidationConfig) -> SignalResult<f64> {
    let mut detections = 0;
    let trials = config.monte_carlo_trials.min(50);

    for _ in 0..trials {
        // Generate signal with known frequency
        let mut rng = scirs2_core::random::rng();
        let n = 100;
        let fs = 10.0;
        let signal_freq = 1.0;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * signal_freq * time).sin() + 0.1 * rng.random_range(-1.0..1.0))
            .collect();

        let freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 50).to_vec();

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
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("Operation failed"))
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

/// Validate confidence intervals via genuine bootstrap resampling.
///
/// For each of several Monte Carlo trials, generates a signal with known
/// clean (noiseless) periodogram power at a known frequency, resamples the
/// noisy observations (with replacement) to build a bootstrap distribution
/// of the periodogram power at that frequency, forms a 95% percentile
/// confidence interval from it, and checks whether the true (noiseless)
/// value actually falls inside. The empirical coverage rate (fraction of
/// trials where it does) replaces a previous stand-in that always returned
/// a hardcoded `0.95` regardless of input.
///
/// NOTE: this resamples the signal values with replacement while holding
/// the (required-to-be-sorted) sample-time grid fixed, a simplified
/// approximation of a full paired/residual bootstrap.
#[allow(dead_code)]
pub fn validate_confidence_intervals(config: &ScipyValidationConfig) -> SignalResult<f64> {
    let outer_trials = config.monte_carlo_trials.min(20).max(1);
    let bootstrap_resamples = 30;
    let confidence_level = 0.95;

    let n = 80;
    let fs = 10.0;
    let true_freq = 1.0;
    let freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 40).to_vec();
    let true_freq_idx = freqs
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - true_freq)
                .abs()
                .partial_cmp(&(b - true_freq).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut covered = 0usize;
    let mut total = 0usize;

    for _ in 0..outer_trials {
        let mut rng = scirs2_core::random::rng();
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let clean: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * true_freq * time).sin())
            .collect();
        let signal: Vec<f64> = clean
            .iter()
            .map(|&c| c + 0.3 * rng.random_range(-1.0..1.0))
            .collect();

        // "True" population value: the periodogram power of the noiseless
        // signal at the known embedded frequency.
        let true_power = match lombscargle(
            &t,
            &clean,
            Some(&freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        ) {
            Ok((_, power)) => power[true_freq_idx],
            Err(_) => continue,
        };

        let mut bootstrap_powers = Vec::with_capacity(bootstrap_resamples);
        for _ in 0..bootstrap_resamples {
            let boot_signal: Vec<f64> = (0..n).map(|_| signal[rng.random_range(0..n)]).collect();

            if let Ok((_, power)) = lombscargle(
                &t,
                &boot_signal,
                Some(&freqs),
                Some("standard"),
                Some(true),
                Some(true),
                Some(1.0),
                Some(AutoFreqMethod::Fft),
            ) {
                bootstrap_powers.push(power[true_freq_idx]);
            }
        }

        if bootstrap_powers.len() < 2 {
            continue;
        }
        bootstrap_powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence_level;
        let last = bootstrap_powers.len() - 1;
        let lower_idx =
            (((alpha / 2.0) * bootstrap_powers.len() as f64).floor() as usize).min(last);
        let upper_idx =
            (((1.0 - alpha / 2.0) * bootstrap_powers.len() as f64).ceil() as usize).min(last);

        let ci_lower = bootstrap_powers[lower_idx];
        let ci_upper = bootstrap_powers[upper_idx.max(lower_idx)];

        total += 1;
        if true_power >= ci_lower && true_power <= ci_upper {
            covered += 1;
        }
    }

    if total == 0 {
        return Ok(0.0);
    }

    Ok(covered as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_confidence_intervals_is_computed_not_hardcoded() {
        let config = ScipyValidationConfig {
            monte_carlo_trials: 15,
            ..ScipyValidationConfig::default()
        };
        let coverage =
            validate_confidence_intervals(&config).expect("CI validation should succeed");

        assert!((0.0..=1.0).contains(&coverage));

        // A genuine empirical coverage rate must be expressible as
        // k / n_trials for some integer k; the old stub returned a fixed
        // 0.95 regardless of `n_trials`, which this would fail for most
        // trial counts (0.95 * 15 = 14.25, not an integer).
        let n_trials = config.monte_carlo_trials.min(20).max(1) as f64;
        let scaled = coverage * n_trials;
        assert!(
            (scaled - scaled.round()).abs() < 1e-6,
            "coverage {coverage} is not an empirical k/{n_trials} proportion"
        );
    }
}
