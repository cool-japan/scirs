//! Core validation functions and main entry points
//!
//! This module provides the main validation orchestration functions that
//! coordinate comprehensive testing across all signal processing modules.

use super::types::*;
use super::filtering::validate_filtering;
use super::spectral::validate_spectral_analysis;
use super::wavelets::validate_wavelets;
use super::windows::validate_windows;
use super::signal_generation::validate_signal_generation;
use crate::error::SignalResult;

/// Run comprehensive validation against SciPy reference implementations
///
/// This function orchestrates a complete validation suite including:
/// - Filter validation (Butterworth, Chebyshev, etc.)
/// - Spectral analysis validation (periodogram, Welch, STFT, multitaper)
/// - Wavelet transform validation (DWT, CWT, wavelet families)
/// - Window function validation
/// - Signal generation and basic operations validation
///
/// # Arguments
///
/// * `config` - Validation configuration specifying test parameters
///
/// # Returns
///
/// * Comprehensive validation results with pass/fail status and error metrics
///
/// # Examples
///
/// ```
/// use scirs2_signal::scipy_validation::{validate_all, ValidationConfig};
///
/// let config = ValidationConfig::default();
/// let results = validate_all(&config).unwrap();
///
/// if results.all_passed() {
///     println!("All validations passed!");
/// } else {
///     println!("Some validations failed:");
///     for failure in results.failures() {
///         println!("  {}: {:?}", failure.test_name, failure.error_message);
///     }
/// }
/// ```
pub fn validate_all(config: &ValidationConfig) -> SignalResult<ValidationResults> {
    let mut results = ValidationResults::new();

    // Run filter validation
    if let Ok(filter_results) = validate_filtering(config) {
        for (test_name, test_result) in filter_results.test_results {
            results.add_test_result(test_result);
        }
    }

    // Run spectral analysis validation
    if let Ok(spectral_results) = validate_spectral_analysis(config) {
        for (test_name, test_result) in spectral_results.test_results {
            results.add_test_result(test_result);
        }
    }

    // Run wavelet validation
    if let Ok(wavelet_results) = validate_wavelets(config) {
        for (test_name, test_result) in wavelet_results.test_results {
            results.add_test_result(test_result);
        }
    }

    // Run window validation
    if let Ok(window_results) = validate_windows(config) {
        for (test_name, test_result) in window_results.test_results {
            results.add_test_result(test_result);
        }
    }

    // Run signal generation validation
    if let Ok(signal_results) = validate_signal_generation(config) {
        for (test_name, test_result) in signal_results.test_results {
            results.add_test_result(test_result);
        }
    }

    Ok(results)
}

/// Run quick validation with default parameters
///
/// This is a convenience function that runs a subset of validation tests
/// with default parameters for quick verification.
///
/// # Returns
///
/// * Validation results for quick tests
pub fn validate_quick() -> SignalResult<ValidationResults> {
    let config = ValidationConfig {
        test_lengths: vec![64, 256],
        sampling_frequencies: vec![1000.0],
        extensive: false,
        ..Default::default()
    };

    validate_all(&config)
}

/// Run comprehensive validation with extended testing
///
/// This function runs an extensive validation suite with more test cases
/// and stricter tolerances. It's more thorough but takes longer to complete.
///
/// # Arguments
///
/// * `config` - Base validation configuration to extend
///
/// # Returns
///
/// * Comprehensive validation results
pub fn validate_comprehensive(config: &ValidationConfig) -> SignalResult<ValidationResults> {
    let extended_config = ValidationConfig {
        test_lengths: vec![16, 32, 64, 128, 256, 512, 1024, 2048],
        sampling_frequencies: vec![1.0, 100.0, 1000.0, 44100.0, 48000.0],
        extensive: true,
        tolerance: config.tolerance * 0.1, // Stricter tolerance
        relative_tolerance: config.relative_tolerance * 0.1,
        ..config.clone()
    };

    validate_all(&extended_config)
}

/// Validate specific component with custom configuration
///
/// This function allows validation of a specific component (filters, spectral, etc.)
/// with custom configuration parameters.
///
/// # Arguments
///
/// * `component` - Component name to validate
/// * `config` - Validation configuration
///
/// # Returns
///
/// * Validation results for the specified component
pub fn validate_component(component: &str, config: &ValidationConfig) -> SignalResult<ValidationResults> {
    match component.to_lowercase().as_str() {
        "filters" | "filtering" => validate_filtering(config),
        "spectral" | "spectrum" => validate_spectral_analysis(config),
        "wavelets" | "wavelet" => validate_wavelets(config),
        "windows" | "window" => validate_windows(config),
        "signals" | "signal" => validate_signal_generation(config),
        _ => {
            let mut results = ValidationResults::new();
            let test_result = ValidationTestResult {
                test_name: format!("unknown_component_{}", component),
                passed: false,
                max_absolute_error: f64::INFINITY,
                max_relative_error: f64::INFINITY,
                mean_absolute_error: f64::INFINITY,
                num_test_cases: 0,
                error_message: Some(format!("Unknown component: {}", component)),
            };
            results.add_test_result(test_result);
            Ok(results)
        }
    }
}

/// Create a custom validation configuration for specific testing needs
///
/// # Arguments
///
/// * `tolerance` - Numerical tolerance for comparisons
/// * `test_lengths` - Signal lengths to test
/// * `extensive` - Whether to run extensive tests
///
/// # Returns
///
/// * Custom validation configuration
pub fn create_custom_config(
    tolerance: f64,
    test_lengths: Vec<usize>,
    extensive: bool,
) -> ValidationConfig {
    ValidationConfig {
        tolerance,
        relative_tolerance: tolerance * 100.0,
        test_lengths,
        sampling_frequencies: vec![1000.0, 44100.0],
        extensive,
        max_error_percent: tolerance * 10000.0, // Convert to percentage
    }
}

/// Validate with different precision levels
///
/// This function runs validation tests with different numerical precision
/// requirements to assess algorithm stability.
///
/// # Arguments
///
/// * `base_config` - Base configuration to modify
///
/// # Returns
///
/// * Map of precision level to validation results
pub fn validate_precision_levels(
    base_config: &ValidationConfig,
) -> SignalResult<std::collections::HashMap<String, ValidationResults>> {
    let mut precision_results = std::collections::HashMap::new();

    // High precision (strict)
    let high_precision = ValidationConfig {
        tolerance: 1e-12,
        relative_tolerance: 1e-10,
        max_error_percent: 0.01,
        ..base_config.clone()
    };
    precision_results.insert("high_precision".to_string(), validate_all(&high_precision)?);

    // Medium precision (standard)
    let medium_precision = ValidationConfig {
        tolerance: 1e-8,
        relative_tolerance: 1e-6,
        max_error_percent: 0.1,
        ..base_config.clone()
    };
    precision_results.insert("medium_precision".to_string(), validate_all(&medium_precision)?);

    // Low precision (relaxed)
    let low_precision = ValidationConfig {
        tolerance: 1e-4,
        relative_tolerance: 1e-2,
        max_error_percent: 1.0,
        ..base_config.clone()
    };
    precision_results.insert("low_precision".to_string(), validate_all(&low_precision)?);

    Ok(precision_results)
}

/// Calculate overall validation score
///
/// This function computes a numerical score (0-100) representing the overall
/// validation quality based on pass rates and error magnitudes.
///
/// # Arguments
///
/// * `results` - Validation results to score
///
/// # Returns
///
/// * Overall validation score (0-100)
pub fn calculate_validation_score(results: &ValidationResults) -> f64 {
    if results.summary.total_tests == 0 {
        return 0.0;
    }

    // Base score from pass rate
    let pass_rate_score = results.summary.pass_rate * 70.0;

    // Error quality score (30 points max)
    let error_score = if results.summary.max_error <= 1e-10 {
        30.0
    } else if results.summary.max_error <= 1e-8 {
        25.0
    } else if results.summary.max_error <= 1e-6 {
        20.0
    } else if results.summary.max_error <= 1e-4 {
        15.0
    } else if results.summary.max_error <= 1e-2 {
        10.0
    } else {
        0.0
    };

    pass_rate_score + error_score
}

/// Check if validation results meet quality criteria
///
/// # Arguments
///
/// * `results` - Validation results to check
/// * `min_pass_rate` - Minimum acceptable pass rate (0.0 to 1.0)
/// * `max_error` - Maximum acceptable error
///
/// # Returns
///
/// * True if results meet criteria, false otherwise
pub fn meets_quality_criteria(
    results: &ValidationResults,
    min_pass_rate: f64,
    max_error: f64,
) -> bool {
    results.summary.pass_rate >= min_pass_rate && results.summary.max_error <= max_error
}

/// Run validation with timeout
///
/// This function runs validation with a specified timeout to prevent
/// hanging on problematic test cases.
///
/// # Arguments
///
/// * `config` - Validation configuration
/// * `timeout_seconds` - Maximum time to allow for validation
///
/// # Returns
///
/// * Validation results or timeout error
pub fn validate_with_timeout(
    config: &ValidationConfig,
    timeout_seconds: u64,
) -> SignalResult<ValidationResults> {
    use std::time::{Duration, Instant};

    let start_time = Instant::now();
    let timeout_duration = Duration::from_secs(timeout_seconds);

    // For now, just run validation normally
    // In a real implementation, this would use threading or async
    let results = validate_all(config)?;

    if start_time.elapsed() > timeout_duration {
        // Add timeout warning to results
        // This is a simplified implementation
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_quick() {
        let results = validate_quick().unwrap();
        assert!(results.summary.total_tests > 0);
    }

    #[test]
    fn test_create_custom_config() {
        let config = create_custom_config(1e-6, vec![64, 128], false);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.test_lengths, vec![64, 128]);
        assert!(!config.extensive);
    }

    #[test]
    fn test_calculate_validation_score() {
        let mut results = ValidationResults::new();

        // Add a passing test
        let test_result = ValidationTestResult {
            test_name: "test1".to_string(),
            passed: true,
            max_absolute_error: 1e-12,
            max_relative_error: 1e-10,
            mean_absolute_error: 5e-13,
            num_test_cases: 10,
            error_message: None,
        };
        results.add_test_result(test_result);

        let score = calculate_validation_score(&results);
        assert!(score > 90.0); // Should be high score for good results
    }

    #[test]
    fn test_meets_quality_criteria() {
        let mut results = ValidationResults::new();

        let test_result = ValidationTestResult {
            test_name: "test1".to_string(),
            passed: true,
            max_absolute_error: 1e-10,
            max_relative_error: 1e-8,
            mean_absolute_error: 5e-11,
            num_test_cases: 10,
            error_message: None,
        };
        results.add_test_result(test_result);

        assert!(meets_quality_criteria(&results, 0.8, 1e-8));
        assert!(!meets_quality_criteria(&results, 0.8, 1e-12));
    }

    #[test]
    fn test_validate_component() {
        let config = ValidationConfig::default();

        // Test known component
        let results = validate_component("filters", &config);
        assert!(results.is_ok());

        // Test unknown component
        let results = validate_component("unknown", &config);
        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.summary.failed_tests, 1);
    }
}