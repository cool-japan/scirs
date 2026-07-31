//! Advanced validation for Lomb-Scargle implementation
//!
//! This module provides advanced validation tests including numerical conditioning,
//! aliasing effects, astronomical scenarios, phase coherence, and uncertainty quantification.

use super::core::validate_lombscargle_against_scipy;
use super::types::*;
use super::utils::find_peaks;
use crate::error::SignalResult;
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::PI;

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
pub fn test_numerical_conditioning(
    config: &ScipyValidationConfig,
) -> SignalResult<ConditioningTestResult> {
    // Generate test data with known conditioning properties
    let n = 1000;
    let mut rng = scirs2_core::random::rng();

    // Create time series with irregular sampling
    let mut times: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 100.0).collect();
    times.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

    // Test signal with multiple frequencies
    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * 0.1 * t).sin() + 0.5 * (2.0 * PI * 0.3 * t).cos())
        .collect();

    // Test frequencies
    let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();

    // Compute periodogram
    let _periodogram = lombscargle(
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
pub fn test_aliasing_effects(config: &ScipyValidationConfig) -> SignalResult<AliasingTestResult> {
    let mut rng = scirs2_core::random::rng();

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
pub fn test_astronomical_scenarios(
    config: &ScipyValidationConfig,
) -> SignalResult<AstronomicalTestResult> {
    let mut rng = scirs2_core::random::rng();

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
pub fn test_phase_coherence(config: &ScipyValidationConfig) -> SignalResult<PhaseCoherenceResult> {
    let mut rng = scirs2_core::random::rng();

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
pub fn quantify_uncertainty(
    config: &ScipyValidationConfig,
    n_bootstrap: usize,
) -> SignalResult<UncertaintyResult> {
    let mut rng = scirs2_core::random::rng();

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
            .1
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Operation failed"))
            .map(|(i, _)| i)
            .unwrap_or(0);

        bootstrap_results.push(freqs[peak_idx]);
    }

    // Compute statistics
    bootstrap_results.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
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
pub fn test_frequency_resolution(
    config: &ScipyValidationConfig,
) -> SignalResult<FrequencyResolutionResult> {
    let mut rng = scirs2_core::random::rng();

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
    let irregularity = estimate_sampling_irregularity(times);
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
    let mut rng = scirs2_core::random::rng();

    // Original periodogram
    let original = lombscargle(times, values, Some(freqs), None, None, None, None, None)?;

    // Perturbed periodogram
    let perturbed_values: Vec<f64> = values
        .iter()
        .map(|&v| v + perturbation_level * rng.random::<f64>())
        .collect();
    let perturbed = lombscargle(
        times,
        &perturbed_values,
        Some(freqs),
        None,
        None,
        None,
        None,
        None,
    )?;

    // Compute relative change
    let relative_changes: Vec<f64> = original
        .1
        .iter()
        .zip(perturbed.1.iter())
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

        let original = lombscargle(times, values, Some(freqs), None, None, None, None, None)?;
        let perturbed = lombscargle(
            times,
            &perturbed_values,
            Some(freqs),
            None,
            None,
            None,
            None,
            None,
        )?;

        let gradient_norm: f64 = original
            .1
            .iter()
            .zip(perturbed.1.iter())
            .map(|(&orig, &pert)| ((pert - orig) / h).powi(2))
            .sum::<f64>()
            .sqrt();

        stability_scores.push(1.0 / (1.0 + gradient_norm));
    }

    Ok(stability_scores.iter().sum::<f64>() / stability_scores.len() as f64)
}

/// Find the periodogram peak frequency for a signal sampled at `fs` with
/// `n` points, at true frequency `freq`, evaluated over `eval_freqs`.
fn peak_frequency(t: &[f64], signal: &[f64], eval_freqs: &[f64]) -> SignalResult<Option<f64>> {
    let (freq_grid, power) = lombscargle(
        t,
        signal,
        Some(eval_freqs),
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    )?;
    Ok(power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| freq_grid[i]))
}

/// Genuine Monte Carlo test: a signal sampled at `fs` with true frequency
/// above Nyquist (`fs/2`) aliases to `|f_true - round(f_true/fs)*fs|`;
/// verify the periodogram's peak lands at the *aliased* frequency (the
/// physically correct behavior for evenly-influenced sampling), rather than
/// asserting a fixed pass rate regardless of the actual data.
#[allow(dead_code)]
fn test_nyquist_aliasing_detection(rng: &mut impl Rng) -> SignalResult<f64> {
    let trials = 20;
    let mut correct = 0;
    let fs = 10.0;
    let n = 60;

    for _ in 0..trials {
        let f_true: f64 = fs / 2.0 + rng.random_range(0.5..2.0); // above Nyquist
        let k = (f_true / fs).round();
        let f_alias = (f_true - k * fs).abs();

        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * f_true * time).sin())
            .collect();
        let eval_freqs: Vec<f64> = Array1::linspace(0.05, fs / 2.0, 80).to_vec();

        if let Some(peak) = peak_frequency(&t, &signal, &eval_freqs)? {
            if (peak - f_alias).abs() < 0.2 {
                correct += 1;
            }
        }
    }
    Ok(correct as f64 / trials as f64)
}

/// Genuine Monte Carlo test: a signal with true frequency safely below
/// Nyquist should have its periodogram peak land at that true frequency
/// (no aliasing artifacts).
#[allow(dead_code)]
fn test_sub_nyquist_handling(rng: &mut impl Rng) -> SignalResult<f64> {
    let trials = 20;
    let mut correct = 0;
    let fs = 10.0;
    let n = 80;

    for _ in 0..trials {
        let f_true = rng.random_range(0.3..(fs / 4.0));
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * f_true * time).sin())
            .collect();
        let eval_freqs: Vec<f64> = Array1::linspace(0.05, fs / 2.0, 100).to_vec();

        if let Some(peak) = peak_frequency(&t, &signal, &eval_freqs)? {
            if (peak - f_true).abs() < 0.15 {
                correct += 1;
            }
        }
    }
    Ok(correct as f64 / trials as f64)
}

/// Genuine Monte Carlo test: pure noise (no embedded signal) should
/// (usually) not produce a periodogram peak exceeding a standard
/// significance threshold; the fraction of trials *without* such a false
/// peak is the suppression score.
#[allow(dead_code)]
fn test_false_peak_suppression(rng: &mut impl Rng) -> SignalResult<f64> {
    let trials = 30;
    let mut suppressed = 0;
    let n = 100;
    let fs = 10.0;

    for _ in 0..trials {
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let noise: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let eval_freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 60).to_vec();

        let (_, power) = lombscargle(
            &t,
            &noise,
            Some(&eval_freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        )?;
        // Standard false-alarm threshold for the normalized LS periodogram
        // (Scargle 1982): a level of ~8-10 corresponds to a very low
        // false-alarm probability for this many independent frequencies.
        if !power.iter().any(|&p| p > 9.0) {
            suppressed += 1;
        }
    }
    Ok(suppressed as f64 / trials as f64)
}

/// Genuine test: for a signal at a frequency that does *not* land exactly
/// on an evaluation-grid bin, spectral leakage spreads power into
/// neighboring bins; score how concentrated the response stays around the
/// true peak (higher = better leakage mitigation) using the real
/// periodogram.
#[allow(dead_code)]
fn test_spectral_leakage_mitigation(rng: &mut impl Rng) -> SignalResult<f64> {
    let trials = 15;
    let mut scores = Vec::with_capacity(trials);
    let n = 200;
    let fs = 20.0;

    for _ in 0..trials {
        // A frequency deliberately off-grid relative to the evaluation bins.
        let f_true = 1.0 + rng.random_range(0.0..1.0) * 0.37;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * f_true * time).sin())
            .collect();
        let eval_freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 128).to_vec();

        let (_, power) = lombscargle(
            &t,
            &signal,
            Some(&eval_freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        )?;

        let total_power: f64 = power.iter().sum();
        if total_power < 1e-12 {
            continue;
        }
        let peak_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Power within +-2 bins of the peak, as a fraction of total power.
        let lo = peak_idx.saturating_sub(2);
        let hi = (peak_idx + 2).min(power.len() - 1);
        let local_power: f64 = power[lo..=hi].iter().sum();
        scores.push((local_power / total_power).clamp(0.0, 1.0));
    }

    if scores.is_empty() {
        return Ok(0.0);
    }
    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}

/// Genuine detection-accuracy test simulating a periodic variable star
/// (single dominant sinusoidal mode).
#[allow(dead_code)]
fn test_variable_star_simulation(rng: &mut impl Rng) -> SignalResult<f64> {
    monte_carlo_period_detection(rng, 20, |rng| {
        let period = rng.random_range(0.5..5.0);
        (period, |t: f64, period: f64| (2.0 * PI * t / period).sin())
    })
}

/// Genuine detection-accuracy test simulating a periodic exoplanet transit
/// signal (a narrow periodic dip rather than a pure sinusoid).
#[allow(dead_code)]
fn test_exoplanet_transit_simulation(rng: &mut impl Rng) -> SignalResult<f64> {
    monte_carlo_period_detection(rng, 20, |rng| {
        let period = rng.random_range(1.0..8.0);
        (period, |t: f64, period: f64| {
            let phase = (t / period).fract();
            if phase < 0.05 {
                -1.0
            } else {
                0.0
            }
        })
    })
}

/// Genuine detection-accuracy test simulating an RR Lyrae star's
/// asymmetric, harmonic-rich light curve (fundamental + two harmonics with
/// distinct amplitudes/phases, unlike a pure sinusoid).
#[allow(dead_code)]
fn test_rr_lyrae_simulation(rng: &mut impl Rng) -> SignalResult<f64> {
    monte_carlo_period_detection(rng, 20, |rng| {
        let period = rng.random_range(0.3..1.0);
        (period, |t: f64, period: f64| {
            let w = 2.0 * PI * t / period;
            w.sin() + 0.4 * (2.0 * w).sin() + 0.15 * (3.0 * w - 0.5).sin()
        })
    })
}

/// Genuine test of detecting *two* independent periodic sources embedded in
/// the same time series: checks that the two strongest periodogram peaks
/// match both true frequencies.
#[allow(dead_code)]
fn test_multi_periodic_source(rng: &mut impl Rng) -> SignalResult<f64> {
    let trials = 15;
    let mut both_found = 0;
    let n = 300;
    let fs = 20.0;

    for _ in 0..trials {
        let f1 = rng.random_range(0.5..2.0);
        let f2 = f1 + rng.random_range(1.0..3.0); // keep them well separated
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * f1 * time).sin() + 0.7 * (2.0 * PI * f2 * time + 0.3).sin())
            .collect();
        let eval_freqs: Vec<f64> = Array1::linspace(0.1, fs / 2.0, 200).to_vec();

        let (freq_grid, power) = lombscargle(
            &t,
            &signal,
            Some(&eval_freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        )?;

        let peaks = find_peaks(&power, 0.1);
        let detected_freqs: Vec<f64> = peaks.iter().map(|&i| freq_grid[i]).collect();

        let found_f1 = detected_freqs.iter().any(|&f| (f - f1).abs() < 0.2);
        let found_f2 = detected_freqs.iter().any(|&f| (f - f2).abs() < 0.2);
        if found_f1 && found_f2 {
            both_found += 1;
        }
    }
    Ok(both_found as f64 / trials as f64)
}

/// Shared Monte Carlo driver for the astronomical-scenario tests: generates
/// a periodic signal from `make_signal(rng) -> (period, waveform_fn)`,
/// evaluates its periodogram, and scores the fraction of trials where the
/// recovered period matches the true one.
fn monte_carlo_period_detection(
    rng: &mut impl Rng,
    trials: usize,
    mut make_signal: impl FnMut(&mut dyn Rng) -> (f64, fn(f64, f64) -> f64),
) -> SignalResult<f64> {
    let mut correct = 0;
    let n = 150;
    let duration_periods = 12.0; // observe ~12 true periods

    for _ in 0..trials {
        let (period, waveform) = make_signal(rng);
        let duration = duration_periods * period;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * duration / n as f64).collect();
        let signal: Vec<f64> = t.iter().map(|&time| waveform(time, period)).collect();

        let true_freq = 1.0 / period;
        let eval_freqs: Vec<f64> =
            Array1::linspace((true_freq * 0.2).max(0.01), true_freq * 3.0, 120).to_vec();

        if let Some(peak) = peak_frequency(&t, &signal, &eval_freqs)? {
            if (peak - true_freq).abs() / true_freq < 0.1 {
                correct += 1;
            }
        }
    }
    Ok(correct as f64 / trials as f64)
}

/// Estimate the phase of a signal's component at `freq` via linear least
/// squares, fitting `y ~= c_coef*cos(2*pi*f*t) + s_coef*sin(2*pi*f*t)` and
/// returning the phase `phi` such that the fitted component is (up to
/// amplitude) `sin(2*pi*f*t + phi)`.
///
/// Expanding `sin(wt+phi) = sin(wt)*cos(phi) + cos(wt)*sin(phi)` shows
/// `c_coef = sin(phi)` and `s_coef = cos(phi)`, so
/// `phi = atan2(c_coef, s_coef) = atan2(sin(phi), cos(phi))` recovers it
/// directly (note the argument order: `c_coef` -- the *cosine* term's
/// coefficient -- goes first, since it equals `sin(phi)`).
fn estimate_phase_at(times: &[f64], values: &[f64], freq: f64) -> f64 {
    let mut sum_cc = 0.0;
    let mut sum_ss = 0.0;
    let mut sum_cs = 0.0;
    let mut sum_yc = 0.0;
    let mut sum_ys = 0.0;

    for (&t, &y) in times.iter().zip(values.iter()) {
        let w = 2.0 * PI * freq * t;
        let (s, c) = (w.sin(), w.cos());
        sum_cc += c * c;
        sum_ss += s * s;
        sum_cs += c * s;
        sum_yc += y * c;
        sum_ys += y * s;
    }

    // Solve the 2x2 normal equations [[cc, cs], [cs, ss]] * [c_coef,
    // s_coef] = [yc, ys] (fitting y ~= c_coef*cos(wt) + s_coef*sin(wt)).
    let det = sum_cc * sum_ss - sum_cs * sum_cs;
    if det.abs() < 1e-12 {
        return 0.0;
    }
    let c_coef = (sum_yc * sum_ss - sum_ys * sum_cs) / det;
    let s_coef = (sum_ys * sum_cc - sum_yc * sum_cs) / det;
    c_coef.atan2(s_coef)
}

/// Angular difference between two phases, wrapped into `[-pi, pi]`.
fn wrapped_phase_diff(a: f64, b: f64) -> f64 {
    let mut diff = a - b;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }
    diff
}

/// Genuine test: estimate the phase of each of the two known frequency
/// components and check that their *difference* matches the phase offset
/// actually embedded in the signal by the caller.
#[allow(dead_code)]
fn test_phase_preservation(
    times: &[f64],
    values: &[f64],
    freq1: f64,
    freq2: f64,
    expected_phase_diff: f64,
) -> SignalResult<f64> {
    let phase1 = estimate_phase_at(times, values, freq1);
    let phase2 = estimate_phase_at(times, values, freq2);
    let observed_diff = wrapped_phase_diff(phase2, phase1);
    let error = wrapped_phase_diff(observed_diff, expected_phase_diff).abs();
    // Score decays linearly from 1.0 (exact match) to 0.0 (worst-case pi error).
    Ok((1.0 - error / PI).clamp(0.0, 1.0))
}

/// Genuine test: split the series into two halves and check that the
/// dominant frequency's phase estimate is consistent between them (a
/// coherent periodic signal should yield closely-agreeing phase estimates
/// in both halves; noise would not).
#[allow(dead_code)]
fn test_coherence_stability(times: &[f64], values: &[f64]) -> SignalResult<f64> {
    let n = times.len();
    if n < 4 {
        return Ok(0.0);
    }
    let mid = n / 2;

    // Identify the dominant frequency from the full series first, via a
    // coarse grid search followed by a finer local refinement (since even
    // small frequency-grid quantization error would otherwise show up as
    // large apparent phase drift between two time-separated windows,
    // masking genuine coherence with a frequency-estimation artifact).
    let fs_estimate = if n > 1 {
        (n as f64 - 1.0) / (times[n - 1] - times[0]).max(1e-9)
    } else {
        1.0
    };
    let coarse_freqs: Vec<f64> = Array1::linspace(0.05, (fs_estimate / 2.0).max(0.1), 200).to_vec();
    let coarse_freq = match peak_frequency(times, values, &coarse_freqs)? {
        Some(f) if f > 1e-9 => f,
        _ => return Ok(0.0),
    };
    let step = coarse_freqs
        .get(1)
        .map(|&f1| f1 - coarse_freqs[0])
        .unwrap_or(0.01);
    let fine_freqs: Vec<f64> =
        Array1::linspace((coarse_freq - step).max(1e-6), coarse_freq + step, 400).to_vec();
    let dominant_freq = peak_frequency(times, values, &fine_freqs)?.unwrap_or(coarse_freq);

    let phase_first = estimate_phase_at(&times[..mid], &values[..mid], dominant_freq);
    let phase_second = estimate_phase_at(&times[mid..], &values[mid..], dominant_freq);
    let diff = wrapped_phase_diff(phase_first, phase_second).abs();
    Ok((1.0 - diff / PI).clamp(0.0, 1.0))
}

/// Genuine (self-verifying) test of wrapped-phase-difference handling: a
/// phase pair separated by many full turns should still report a small
/// wrapped difference once correctly unwrapped, and the sign/near-boundary
/// behavior of the wrap must be internally consistent.
#[allow(dead_code)]
fn test_phase_wrapping(times: &[f64], values: &[f64]) -> SignalResult<f64> {
    let mut checks_passed = 0;
    let mut total_checks = 0;

    // A phase advanced by many full turns plus a small real offset should
    // wrap back to (approximately) that small offset.
    for &turns in &[1, 3, 7, 20] {
        for &offset in &[0.1_f64, 0.5, -0.3, 1.2, -2.5] {
            let wrapped = wrapped_phase_diff(offset + 2.0 * PI * turns as f64, 0.0);
            total_checks += 1;
            if (wrapped - offset).abs() < 1e-6 {
                checks_passed += 1;
            }
        }
    }

    // Also exercise it on the actual (times, values) series: the dominant
    // frequency's self-difference must wrap to exactly zero.
    let fs_estimate = if times.len() > 1 {
        (times.len() as f64 - 1.0) / (times[times.len() - 1] - times[0]).max(1e-9)
    } else {
        1.0
    };
    let eval_freqs: Vec<f64> = Array1::linspace(0.05, (fs_estimate / 2.0).max(0.1), 60).to_vec();
    if let Some(f) = peak_frequency(times, values, &eval_freqs)? {
        let phase = estimate_phase_at(times, values, f);
        total_checks += 1;
        if wrapped_phase_diff(phase, phase).abs() < 1e-9 {
            checks_passed += 1;
        }
    }

    Ok(checks_passed as f64 / total_checks.max(1) as f64)
}

/// Genuine test of the minimum resolvable frequency separation: generates
/// two equal-amplitude sinusoids with decreasing frequency separation and
/// finds the smallest separation at which the periodogram still shows two
/// distinct peaks (rather than merging into one), approaching the
/// classical Rayleigh resolution limit `~1/T`.
#[allow(dead_code)]
fn test_min_frequency_separation(rng: &mut impl Rng) -> SignalResult<f64> {
    let fs = 20.0;
    let n = 400;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let duration = t[n - 1] - t[0];
    let f_center = 2.0 + rng.random_range(-0.2..0.2);

    let rayleigh = 1.0 / duration;
    let candidate_separations: Vec<f64> = (1..=20)
        .map(|k| rayleigh * (0.3 + 0.2 * k as f64))
        .collect();

    let mut min_resolved = candidate_separations
        .last()
        .copied()
        .unwrap_or(rayleigh * 5.0);

    for &sep in &candidate_separations {
        let f1 = f_center - sep / 2.0;
        let f2 = f_center + sep / 2.0;
        let signal: Vec<f64> = t
            .iter()
            .map(|&time| (2.0 * PI * f1 * time).sin() + (2.0 * PI * f2 * time).sin())
            .collect();
        let eval_freqs: Vec<f64> =
            Array1::linspace((f_center - 2.0 * sep).max(1e-3), f_center + 2.0 * sep, 200).to_vec();

        let (_, power) = lombscargle(
            &t,
            &signal,
            Some(&eval_freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        )?;
        let peaks = find_peaks(&power, 0.3);
        if peaks.len() >= 2 && sep < min_resolved {
            min_resolved = sep;
        }
    }

    Ok(min_resolved)
}

/// Genuine test of how the minimum resolvable frequency separation scales
/// with the observation baseline length `T`: the classical Rayleigh
/// criterion predicts `separation ~ 1/T` (scaling exponent -1); this
/// empirically measures the exponent by comparing the minimum resolvable
/// separation at two different baseline lengths.
#[allow(dead_code)]
fn test_resolution_scaling(rng: &mut impl Rng) -> SignalResult<f64> {
    let fs = 20.0;
    let f_center = 2.0 + rng.random_range(-0.2..0.2);

    let min_sep_for_n = |n: usize| -> SignalResult<f64> {
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let duration = t[n - 1] - t[0];
        let rayleigh = 1.0 / duration;
        let candidate_separations: Vec<f64> = (1..=15)
            .map(|k| rayleigh * (0.3 + 0.3 * k as f64))
            .collect();

        let mut min_resolved = candidate_separations
            .last()
            .copied()
            .unwrap_or(rayleigh * 5.0);
        for &sep in &candidate_separations {
            let f1 = f_center - sep / 2.0;
            let f2 = f_center + sep / 2.0;
            let signal: Vec<f64> = t
                .iter()
                .map(|&time| (2.0 * PI * f1 * time).sin() + (2.0 * PI * f2 * time).sin())
                .collect();
            let eval_freqs: Vec<f64> =
                Array1::linspace((f_center - 2.0 * sep).max(1e-3), f_center + 2.0 * sep, 150)
                    .to_vec();
            let (_, power) = lombscargle(
                &t,
                &signal,
                Some(&eval_freqs),
                Some("standard"),
                Some(true),
                Some(true),
                Some(1.0),
                Some(AutoFreqMethod::Fft),
            )?;
            if find_peaks(&power, 0.3).len() >= 2 && sep < min_resolved {
                min_resolved = sep;
            }
        }
        Ok(min_resolved)
    };

    let n_short = 100;
    let n_long = 400; // 4x the baseline (same fs)
    let sep_short = min_sep_for_n(n_short)?;
    let sep_long = min_sep_for_n(n_long)?;

    if sep_short <= 1e-9 || sep_long <= 1e-9 {
        return Ok(1.0);
    }
    // Empirical scaling exponent: sep ~ T^exponent; Rayleigh predicts -1.
    let size_ratio = n_long as f64 / n_short as f64;
    let sep_ratio = sep_long / sep_short;
    Ok(sep_ratio.ln() / size_ratio.ln())
}

/// Genuine spectral-window characterization: computes the periodogram of a
/// *unit-amplitude constant* signal (isolating the effect of the sampling
/// pattern alone, with no astrophysical signal), and scores how
/// concentrated its response is near zero frequency (a good, low-leakage
/// sampling window has most of its "spectral window" power near f=0).
#[allow(dead_code)]
fn characterize_spectral_window(rng: &mut impl Rng) -> SignalResult<f64> {
    let n = 150;
    let fs = 10.0;
    // Irregular sampling pattern (deterministic jitter, exercised via rng
    // so different calls probe different sampling realizations).
    let t: Vec<f64> = (0..n)
        .map(|i| i as f64 / fs + rng.random_range(-0.02..0.02))
        .collect();
    let mut t = t;
    t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let window_signal = vec![1.0; n];

    let eval_freqs: Vec<f64> = Array1::linspace(0.05, fs / 2.0, 100).to_vec();
    let (_, power) = lombscargle(
        &t,
        &window_signal,
        Some(&eval_freqs),
        Some("standard"),
        Some(false), // do not center a constant signal (would zero it out)
        Some(false),
        Some(1.0),
        Some(AutoFreqMethod::Fft),
    )?;

    let total_power: f64 = power.iter().sum();
    if total_power < 1e-300 {
        return Ok(0.0);
    }
    // Power in the lowest-frequency quarter of the evaluated band, as a
    // fraction of the total: higher is a "cleaner" (more concentrated,
    // lower-leakage) spectral window.
    let quarter = (power.len() / 4).max(1);
    let low_freq_power: f64 = power[..quarter].iter().sum();
    Ok((low_freq_power / total_power).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aliasing_effects_returns_real_scores() {
        let config = ScipyValidationConfig::default();
        let result = test_aliasing_effects(&config).expect("should succeed");

        // The fabricated implementation always returned the exact
        // constants 0.8/0.9/0.85/0.75 regardless of the (unused) rng.
        for score in [
            result.nyquist_detection,
            result.sub_nyquist_handling,
            result.false_peak_suppression,
            result.leakage_mitigation,
        ] {
            assert!((0.0..=1.0).contains(&score));
        }
        // Sub-Nyquist detection (an easy case) should be reliable.
        assert!(
            result.sub_nyquist_handling > 0.7,
            "sub_nyquist_handling={}",
            result.sub_nyquist_handling
        );
    }

    #[test]
    fn test_astronomical_scenarios_detects_known_periods() {
        let config = ScipyValidationConfig::default();
        let result = test_astronomical_scenarios(&config).expect("should succeed");

        for score in [
            result.variable_star_detection,
            result.transit_detection,
            result.rr_lyrae_accuracy,
            result.multi_periodic_handling,
        ] {
            assert!((0.0..=1.0).contains(&score));
        }
        // A clean, densely-sampled sinusoid (variable star case) should be
        // detected reliably -- the old stub always returned 0.9 regardless
        // of whether detection could genuinely succeed.
        assert!(
            result.variable_star_detection > 0.6,
            "variable_star_detection={}",
            result.variable_star_detection
        );
    }

    #[test]
    fn test_phase_preservation_detects_known_phase_offset() {
        let n = 500;
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let freq1 = 0.2;
        let freq2 = 0.6;
        let phase_offset = PI / 4.0;
        let values: Vec<f64> = times
            .iter()
            .map(|&t| (2.0 * PI * freq1 * t).sin() + (2.0 * PI * freq2 * t + phase_offset).sin())
            .collect();

        // Correct phase offset: should score very close to 1.0.
        let good_score =
            test_phase_preservation(&times, &values, freq1, freq2, phase_offset).expect("ok");
        assert!(good_score > 0.9, "good_score={good_score}");

        // A deliberately wrong expected offset (pi away) should score
        // near 0.0; the old stub always returned 0.9 regardless of
        // whether the claimed phase relationship was actually correct.
        let bad_score =
            test_phase_preservation(&times, &values, freq1, freq2, phase_offset + PI).expect("ok");
        assert!(bad_score < 0.2, "bad_score={bad_score}");
    }

    #[test]
    fn test_frequency_resolution_gives_sensible_values() {
        let config = ScipyValidationConfig::default();
        let result = test_frequency_resolution(&config).expect("should succeed");

        assert!(result.min_frequency_separation > 0.0);
        assert!(result.min_frequency_separation.is_finite());
        assert!(result.resolution_scaling.is_finite());
        assert!((0.0..=1.0).contains(&result.spectral_window_quality));
    }

    #[test]
    fn test_coherence_stability_high_for_coherent_signal() {
        let n = 300;
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let freq = 0.5;
        let values: Vec<f64> = times.iter().map(|&t| (2.0 * PI * freq * t).sin()).collect();

        let score = test_coherence_stability(&times, &values).expect("should succeed");
        // A single pure, noiseless tone spanning the whole series should
        // show highly consistent phase between the first and second half.
        assert!(score > 0.8, "score={score}");
    }
}
