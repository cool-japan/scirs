//! Utility functions for SciPy validation
//!
//! This module provides utility functions for error calculation, data loading,
//! and report generation used throughout the validation framework.

use super::types::*;
use crate::error::{SignalError, SignalResult};

/// Calculate absolute error, relative error, and RMSE between two signals
pub fn calculate_errors(signal1: &[f64], signal2: &[f64]) -> SignalResult<(f64, f64, f64)> {
    if signal1.len() != signal2.len() {
        return Err(SignalError::ValueError(
            "Signals must have the same length for error calculation".to_string(),
        ));
    }

    let n = signal1.len();
    let mut max_abs_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut mse = 0.0;

    for i in 0..n {
        let abs_error = (signal1[i] - signal2[i]).abs();
        max_abs_error = max_abs_error.max(abs_error);

        let rel_error = if signal2[i].abs() > 1e-15 {
            abs_error / signal2[i].abs()
        } else {
            0.0
        };
        max_rel_error = max_rel_error.max(rel_error);

        mse += abs_error * abs_error;
    }

    let rmse = (mse / n as f64).sqrt();

    Ok((max_abs_error, max_rel_error, rmse))
}

/// Helper function to load reference data from pre-computed SciPy results
///
/// In a production implementation, this would load reference data that was
/// computed offline using SciPy and stored in files or embedded in the binary.
pub fn load_reference_data(test_name: &str, _parameters: &str) -> SignalResult<Vec<f64>> {
    // This is a placeholder implementation
    // In practice, you would:
    // 1. Load from embedded data files
    // 2. Use a lookup table based on test parameters
    // 3. Call Python/SciPy via subprocess or FFI

    match test_name {
        "butterworth_lowpass_order2_fs100_fc20" => {
            // Return pre-computed reference data
            Ok(vec![1.0, 0.8, 0.6, 0.4, 0.2]) // Placeholder
        }
        _ => Err(SignalError::ValueError(format!(
            "No reference data available for test: {}",
            test_name
        ))),
    }
}

/// Generate detailed validation report
pub fn generate_validation_report(results: &ValidationResults) -> String {
    let mut report = String::new();

    report.push_str("=== SciPy Numerical Validation Report ===\n\n");

    // Summary
    report.push_str(&results.summary_report());
    report.push_str("\n\n");

    // Detailed results
    report.push_str("=== Detailed Test Results ===\n\n");

    let mut test_names: Vec<_> = results.test_results.keys().collect();
    test_names.sort();

    for test_name in test_names {
        let result = &results.test_results[test_name];

        report.push_str(&format!("Test: {}\n", result.test_name));
        report.push_str(&format!(
            "  Status: {}\n",
            if result.passed { "PASSED" } else { "FAILED" }
        ));
        report.push_str(&format!("  Cases: {}\n", result.num_cases));
        report.push_str(&format!(
            "  Max Absolute Error: {:.2e}\n",
            result.max_absolute_error
        ));
        report.push_str(&format!(
            "  Max Relative Error: {:.2e}\n",
            result.max_relative_error
        ));
        report.push_str(&format!("  RMSE: {:.2e}\n", result.rmse));
        report.push_str(&format!(
            "  Execution Time: {:.2}ms\n",
            result.execution_time_ms
        ));

        if let Some(ref error_msg) = result.error_message {
            report.push_str(&format!("  Error: {}\n", error_msg));
        }

        report.push_str("\n");
    }

    // Recommendations
    if !results.all_passed() {
        report.push_str("=== Recommendations ===\n\n");
        report.push_str("The following tests failed and require attention:\n\n");

        for failure in results.failures() {
            report.push_str(&format!(
                "- {}: {}\n",
                failure.test_name,
                failure
                    .error_message
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            ));
        }
    }

    report
}

/// Quick validation suite for basic functionality testing
pub fn validate_quick() -> SignalResult<ValidationResults> {
    use super::core::validate_all;

    let mut config = ValidationConfig::default();
    config.extensive = false;
    config.test_lengths = vec![64, 128, 256]; // Smaller test set for quick validation
    config.sampling_frequencies = vec![1000.0, 44100.0]; // Common sample rates
    config.tolerance = 1e-8; // Slightly relaxed tolerance for quick tests
    config.relative_tolerance = 1e-6;

    validate_all(&config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_calculate_errors() {
        let signal1 = vec![1.0, 2.0, 3.0, 4.0];
        let signal2 = vec![1.1, 1.9, 3.1, 3.9];

        let (abs_err, rel_err, rmse) = calculate_errors(&signal1, &signal2).unwrap();

        assert!(abs_err > 0.0);
        assert!(rel_err > 0.0);
        assert!(rmse > 0.0);
    }

    #[test]
    fn test_validation_results_all_passed() {
        let mut test_results = HashMap::new();
        test_results.insert(
            "test1".to_string(),
            ValidationTestResult {
                test_name: "test1".to_string(),
                passed: true,
                max_absolute_error: 1e-12,
                max_relative_error: 1e-10,
                rmse: 1e-11,
                error_message: None,
                num_cases: 10,
                execution_time_ms: 50.0,
            },
        );

        let summary = ValidationSummary {
            total_tests: 1,
            passed_tests: 1,
            failed_tests: 0,
            pass_rate: 1.0,
            total_time_ms: 50.0,
        };

        let results = ValidationResults {
            test_results,
            summary,
        };
        assert!(results.all_passed());
        assert_eq!(results.failures().len(), 0);
    }

    #[test]
    fn test_quick_validation() {
        // Test the quick validation suite
        let results = validate_quick();
        assert!(results.is_ok());

        let validation_results = results.unwrap();
        assert!(validation_results.summary.total_tests > 0);
    }

    #[test]
    fn test_generate_validation_report() {
        let mut test_results = HashMap::new();
        test_results.insert(
            "test1".to_string(),
            ValidationTestResult {
                test_name: "test1".to_string(),
                passed: true,
                max_absolute_error: 1e-12,
                max_relative_error: 1e-10,
                rmse: 1e-11,
                error_message: None,
                num_cases: 10,
                execution_time_ms: 50.0,
            },
        );

        let summary = ValidationSummary {
            total_tests: 1,
            passed_tests: 1,
            failed_tests: 0,
            pass_rate: 1.0,
            total_time_ms: 50.0,
        };

        let results = ValidationResults {
            test_results,
            summary,
        };

        let report = generate_validation_report(&results);
        assert!(report.contains("SciPy Numerical Validation Report"));
        assert!(report.contains("test1"));
        assert!(report.contains("PASSED"));
    }
}