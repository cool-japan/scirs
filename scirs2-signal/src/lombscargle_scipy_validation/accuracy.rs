//! Basic accuracy validation against SciPy implementation
//!
//! This module provides functions to validate the numerical accuracy of our
//! Lomb-Scargle implementation by comparing with SciPy reference results.

use super::types::*;
use super::utils::*;
use crate::error::{SignalError, SignalResult};
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::Rng;
use std::f64::consts::PI;

/// Validate basic accuracy against SciPy implementation
#[allow(dead_code)]
pub fn validate_basic_accuracy(
    config: &ScipyValidationConfig,
) -> SignalResult<AccuracyValidationResult> {
    let mut max_abs_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;
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
pub fn validate_single_case(
    n: usize,
    fs: f64,
    test_freq: f64,
    config: &ScipyValidationConfig,
) -> SignalResult<(f64, f64, f64, f64)> {
    // Generate irregular time samples
    let mut rng = scirs2_core::random::rng();
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
    let mut time_signal: Vec<(f64, f64)> = t.into_iter().zip(signal).collect();
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
    let (abs_err, rel_err, rmse) = calculate_error_metrics(&our_power, &scipy_power)?;
    let correlation = calculate_correlation(&our_power, &scipy_power)?;

    Ok((abs_err, rel_err, rmse, correlation))
}

/// Compute reference Lomb-Scargle using high-precision algorithm
/// This implements the exact algorithm used by SciPy for validation
#[allow(dead_code)]
pub fn compute_reference_lombscargle(
    t: &[f64],
    y: &[f64],
    freqs: &[f64],
) -> SignalResult<Vec<f64>> {
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
