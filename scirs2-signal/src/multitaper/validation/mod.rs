//! Comprehensive validation suite for multitaper spectral estimation
//!
//! This module provides extensive validation functionality for multitaper methods including:
//!
//! - DPSS validation (orthogonality, concentration, eigenvalue ordering)
//! - Spectral estimation accuracy assessment
//! - Numerical stability testing
//! - Performance benchmarking and optimization
//! - Cross-validation with reference implementations
//! - SIMD operation validation
//! - Robustness testing under extreme conditions
//!
//! # Examples
//!
//! ## Basic Validation
//!
//! ```
//! use scirs2_signal::multitaper::validation::{
//!     validate_multitaper_comprehensive, TestSignalConfig, ValidationConfig
//! };
//!
//! let signal_config = TestSignalConfig::default();
//! let validation_config = ValidationConfig::default();
//!
//! let result = validate_multitaper_comprehensive(&signal_config, &validation_config).unwrap();
//! println!("Overall validation score: {}", result.overall_score);
//! ```
//!
//! ## Enhanced Validation with Extended Metrics
//!
//! ```
//! use scirs2_signal::multitaper::validation::{
//!     run_comprehensive_enhanced_validation, ComprehensiveTestConfig
//! };
//!
//! let config = ComprehensiveTestConfig::default();
//! let result = run_comprehensive_enhanced_validation(&config).unwrap();
//!
//! println!("Enhanced score: {}", result.enhanced_score);
//! for recommendation in &result.recommendations {
//!     println!("Recommendation: {}", recommendation);
//! }
//! ```
//!
//! ## Signal Generation for Testing
//!
//! ```
//! use scirs2_signal::multitaper::validation::{
//!     generate_test_signal, TestSignalConfig, TestSignalType
//! };
//!
//! let config = TestSignalConfig::default();
//! let signal = generate_test_signal(&TestSignalType::Sinusoid(100.0), &config).unwrap();
//! println!("Generated signal with {} samples", signal.len());
//! ```

// Module organization
pub mod types;
pub mod core;
pub mod dpss;
pub mod spectral;
pub mod stability;
pub mod performance;
pub mod cross_validation;
pub mod simd;
pub mod signal_generation;

// Re-export all types for easy access
pub use types::*;

// Re-export main validation functions
pub use core::{
    validate_multitaper_comprehensive,
    run_comprehensive_enhanced_validation,
};

// Re-export specialized validation functions
pub use dpss::validate_dpss_comprehensive;
pub use spectral::validate_spectral_accuracy;
pub use stability::test_numerical_stability_enhanced;
pub use performance::{
    benchmark_performance,
    benchmark_memory_access,
    profile_computation_phases,
};
pub use cross_validation::{
    cross_validate_with_reference,
    cross_validate_with_multiple_references,
    calculate_correlation,
    compute_relative_errors,
};
pub use simd::{
    validate_simd_operations,
    validate_multitaper_with_simd,
};
pub use signal_generation::{
    generate_test_signal,
    add_noise_to_signal,
    assess_signal_quality,
    generate_validation_signals,
};

/// Run quick validation with default parameters
///
/// This is a convenience function for basic validation testing.
///
/// # Returns
///
/// * Overall validation score (0-100)
pub fn quick_validation() -> crate::error::SignalResult<f64> {
    let signal_config = TestSignalConfig {
        length: 512,
        num_tests: 5,
        ..Default::default()
    };
    let validation_config = ValidationConfig {
        monte_carlo_iterations: 10,
        ..Default::default()
    };

    let result = validate_multitaper_comprehensive(&signal_config, &validation_config)?;
    Ok(result.overall_score)
}

/// Validate specific signal type with custom parameters
///
/// # Arguments
///
/// * `signal_type` - Type of signal to test
/// * `length` - Signal length
/// * `fs` - Sampling frequency
/// * `nw` - Time-bandwidth product
/// * `k` - Number of tapers
///
/// # Returns
///
/// * Validation score for the specific configuration
pub fn validate_signal_type(
    signal_type: TestSignalType,
    length: usize,
    fs: f64,
    nw: f64,
    k: usize,
) -> crate::error::SignalResult<f64> {
    let signal_config = TestSignalConfig {
        length,
        fs,
        nw,
        k,
        signal_types: vec![signal_type],
        num_tests: 3,
        ..Default::default()
    };

    let validation_config = ValidationConfig::default();
    let result = validate_multitaper_comprehensive(&signal_config, &validation_config)?;
    Ok(result.overall_score)
}

/// Validate SIMD performance for multitaper operations
///
/// # Arguments
///
/// * `length` - Signal length for testing
///
/// # Returns
///
/// * SIMD validation metrics
pub fn validate_simd_performance(length: usize) -> crate::error::SignalResult<SimdValidationMetrics> {
    let signal_config = TestSignalConfig {
        length,
        test_simd: true,
        ..Default::default()
    };

    validate_multitaper_with_simd(&signal_config)
}

/// Benchmark multitaper performance for different signal lengths
///
/// # Arguments
///
/// * `lengths` - Vector of signal lengths to test
///
/// # Returns
///
/// * Vector of performance metrics for each length
pub fn benchmark_scaling_performance(
    lengths: &[usize],
) -> crate::error::SignalResult<Vec<PerformanceMetrics>> {
    let mut results = Vec::new();

    for &length in lengths {
        let config = TestSignalConfig {
            length,
            num_tests: 5,
            ..Default::default()
        };

        let metrics = benchmark_performance(&config)?;
        results.push(metrics);
    }

    Ok(results)
}

/// Generate comprehensive validation report
///
/// This function runs all validation tests and generates a detailed report.
///
/// # Arguments
///
/// * `config` - Comprehensive test configuration
///
/// # Returns
///
/// * Formatted validation report as string
pub fn generate_validation_report(
    config: &ComprehensiveTestConfig,
) -> crate::error::SignalResult<String> {
    let results = run_comprehensive_enhanced_validation(config)?;

    let mut report = String::new();
    report.push_str("=== Multitaper Validation Report ===\n\n");

    // Overall scores
    report.push_str(&format!("Overall Score: {:.1}%\n", results.standard_metrics.overall_score));
    report.push_str(&format!("Enhanced Score: {:.1}%\n\n", results.enhanced_score));

    // Standard metrics
    report.push_str("=== Standard Validation Metrics ===\n");
    report.push_str(&format!("DPSS Orthogonality Error: {:.2e}\n",
        results.standard_metrics.dpss_validation.orthogonality_error));
    report.push_str(&format!("Spectral Estimation MSE: {:.2e}\n",
        results.standard_metrics.spectral_accuracy.mse));
    report.push_str(&format!("Numerical Condition Number: {:.1}\n",
        results.standard_metrics.numerical_stability.condition_number));
    report.push_str(&format!("SIMD Speedup: {:.1}x\n",
        results.standard_metrics.performance.simd_speedup));

    if results.standard_metrics.cross_validation.reference_available {
        report.push_str(&format!("Reference Correlation: {:.3}\n",
            results.standard_metrics.cross_validation.reference_correlation));
    }

    // Robustness metrics
    report.push_str("\n=== Robustness Metrics ===\n");
    report.push_str(&format!("Extreme Case Stability: {:.1}%\n",
        results.robustness.extreme_case_stability * 100.0));
    report.push_str(&format!("Numerical Consistency: {:.1}%\n",
        results.robustness.numerical_consistency * 100.0));
    report.push_str(&format!("Memory Scaling: {:.1}%\n",
        results.robustness.memory_scaling * 100.0));

    // SIMD metrics
    if results.simd_metrics.platform_compatible {
        report.push_str("\n=== SIMD Validation ===\n");
        report.push_str(&format!("SIMD Correctness: {:.1}%\n",
            results.simd_metrics.correctness_score * 100.0));
        report.push_str(&format!("Performance Improvement: {:.1}x\n",
            results.simd_metrics.performance_improvement));
    }

    // Issues and recommendations
    if !results.standard_metrics.issues.is_empty() {
        report.push_str("\n=== Issues Found ===\n");
        for issue in &results.standard_metrics.issues {
            report.push_str(&format!("- {}\n", issue));
        }
    }

    report.push_str("\n=== Recommendations ===\n");
    for recommendation in &results.recommendations {
        report.push_str(&format!("- {}\n", recommendation));
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_validation() {
        let score = quick_validation().unwrap();
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_validate_signal_type() {
        let score = validate_signal_type(
            TestSignalType::Sinusoid(100.0),
            512,
            1000.0,
            4.0,
            7,
        ).unwrap();
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_simd_validation() {
        let metrics = validate_simd_performance(256).unwrap();
        assert!(metrics.correctness_score >= 0.0 && metrics.correctness_score <= 1.0);
        assert!(metrics.performance_improvement >= 1.0);
    }

    #[test]
    fn test_scaling_benchmark() {
        let lengths = vec![128, 256, 512];
        let results = benchmark_scaling_performance(&lengths).unwrap();
        assert_eq!(results.len(), 3);

        // Performance should generally decrease with larger signals
        for metrics in &results {
            assert!(metrics.standard_time_ms > 0.0);
            assert!(metrics.enhanced_time_ms > 0.0);
        }
    }

    #[test]
    fn test_validation_report() {
        let config = ComprehensiveTestConfig {
            signal_config: TestSignalConfig {
                length: 256,
                num_tests: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let report = generate_validation_report(&config).unwrap();
        assert!(report.contains("Multitaper Validation Report"));
        assert!(report.contains("Overall Score"));
        assert!(report.contains("Recommendations"));
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let signal_config = TestSignalConfig {
            length: 128,
            k: 5,
            num_tests: 1,
            test_simd: false, // Disable for faster testing
            test_robustness: false,
            ..Default::default()
        };

        // Test signal generation
        let signal = generate_test_signal(&TestSignalType::WhiteNoise, &signal_config).unwrap();
        assert_eq!(signal.len(), 128);

        // Test quality assessment
        let quality = assess_signal_quality(&signal, signal_config.fs).unwrap();
        assert!(quality.snr_db.is_finite());

        // Test validation
        let validation_config = ValidationConfig {
            tolerance: 1e-6,
            monte_carlo_iterations: 5,
            ..Default::default()
        };

        let result = validate_multitaper_comprehensive(&signal_config, &validation_config).unwrap();
        assert!(result.overall_score >= 0.0);
    }
}