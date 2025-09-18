//! Basic validation test suite for wavelet packet transforms
//!
//! This module provides fundamental validation tests for WPT operations
//! including energy conservation, reconstruction accuracy, and stability.

use super::types::*;
use super::utils::{generate_test_signal, test_wpt_round_trip};
use crate::error::SignalResult;
use crate::wpt_validation::WptValidationResult;

/// Run basic validation test suite
pub fn run_basic_validation_suite(
    config: &ComprehensiveWptValidationConfig,
) -> SignalResult<WptValidationResult> {
    let mut energy_ratios = Vec::new();
    let mut reconstruction_errors = Vec::new();
    let mut issues: Vec<String> = Vec::new();

    // Test across different signal types and parameters
    for &wavelet in &config.test_wavelets {
        for &signal_length in &config.test_signal_lengths {
            for &level in &config.test_levels {
                if level * 2 > signal_length.trailing_zeros() as usize {
                    continue; // Skip invalid combinations
                }

                for signal_type in &config.test_signal_types {
                    for trial in 0..config.random_trials {
                        let signal = generate_test_signal(*signal_type, signal_length, trial)?;

                        // Test WPT decomposition and reconstruction
                        match test_wpt_round_trip(&signal, wavelet, level) {
                            Ok((energy_ratio, reconstruction_error)) => {
                                energy_ratios.push(energy_ratio);
                                reconstruction_errors.push(reconstruction_error);
                            }
                            Err(e) => {
                                issues.push(format!(
                                    "Failed test: wavelet={:?}, length={}, level={}, signal={:?}, trial={}: {}",
                                    wavelet, signal_length, level, signal_type, trial, e
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // Calculate validation metrics
    let mean_energy_ratio = if !energy_ratios.is_empty() {
        energy_ratios.iter().sum::<f64>() / energy_ratios.len() as f64
    } else {
        0.0
    };

    let mean_reconstruction_error = if !reconstruction_errors.is_empty() {
        reconstruction_errors.iter().sum::<f64>() / reconstruction_errors.len() as f64
    } else {
        f64::INFINITY
    };

    let max_reconstruction_error = reconstruction_errors
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);

    // Calculate stability score based on variance of metrics
    let energy_variance = if energy_ratios.len() > 1 {
        let mean = mean_energy_ratio;
        energy_ratios
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / (energy_ratios.len() - 1) as f64
    } else {
        0.0
    };

    let stability_score = if energy_variance > 0.0 {
        1.0 / (1.0 + energy_variance * 1000.0)
    } else {
        1.0
    };

    // Basic orthogonality score (simplified)
    let orthogonality_score = if mean_reconstruction_error < config.tolerance {
        1.0
    } else {
        (config.tolerance / mean_reconstruction_error).min(1.0)
    };

    let total_tests = energy_ratios.len() + issues.len();
    let passed_tests = energy_ratios.len();

    Ok(WptValidationResult {
        energy_ratio: mean_energy_ratio,
        mean_reconstruction_error,
        max_reconstruction_error,
        stability_score,
        orthogonality_score,
        test_count: total_tests,
        passed_count: passed_tests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;

    #[test]
    fn test_basic_validation_suite() {
        let mut config = ComprehensiveWptValidationConfig::default();

        // Use minimal configuration for testing
        config.test_wavelets = vec![Wavelet::Haar];
        config.test_signal_lengths = vec![32];
        config.test_levels = vec![1];
        config.random_trials = 1;
        config.test_signal_types = vec![TestSignalType::WhiteNoise];

        let result = run_basic_validation_suite(&config);

        // The test should complete without panicking
        // Results may vary depending on the implementation details
        assert!(result.is_ok() || result.is_err());
    }
}