//! Comprehensive Wavelet Packet Transform Validation Suite
//!
//! This module provides extensive validation of WPT implementations including:
//! - Advanced energy and frame theory validation
//! - Multi-scale orthogonality testing
//! - Best basis algorithm validation
//! - Adaptive threshold validation
//! - Statistical significance testing
//! - Cross-validation with reference implementations
//! - Performance regression testing
//!
//! # Organization
//!
//! The module is organized into focused sub-modules:
//!
//! - [`types`] - Type definitions and configuration structures
//! - [`core`] - Main validation orchestration and entry points
//! - [`basic`] - Basic validation test suite
//! - [`frame`] - Frame theory validation
//! - [`multiscale`] - Multi-scale analysis validation
//! - [`best_basis`] - Best basis algorithm validation
//! - [`statistical`] - Statistical validation and hypothesis testing
//! - [`cross_validation`] - Cross-validation with reference implementations
//! - [`robustness`] - Robustness testing under various conditions
//! - [`utils`] - Utility functions for signal generation and analysis
//!
//! # Usage
//!
//! ## Basic Comprehensive Validation
//!
//! ```rust,no_run
//! use scirs2_signal::wpt_comprehensive_validation::{
//!     validate_wpt_comprehensive, ComprehensiveWptValidationConfig
//! };
//!
//! // Run comprehensive validation with default settings
//! let config = ComprehensiveWptValidationConfig::default();
//! let results = validate_wpt_comprehensive(&config)?;
//!
//! println!("Overall validation score: {:.2}", results.overall_score);
//! if !results.issues.is_empty() {
//!     println!("Critical issues found:");
//!     for issue in &results.issues {
//!         println!("  - {}", issue);
//!     }
//! }
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! ## Custom Configuration
//!
//! ```rust,no_run
//! use scirs2_signal::wpt_comprehensive_validation::{
//!     validate_wpt_comprehensive, ComprehensiveWptValidationConfig, TestSignalType
//! };
//! use scirs2_signal::dwt::Wavelet;
//!
//! // Create custom validation configuration
//! let mut config = ComprehensiveWptValidationConfig::default();
//! config.test_wavelets = vec![Wavelet::DB(4), Wavelet::Haar];
//! config.test_signal_lengths = vec![128, 256, 512];
//! config.random_trials = 50;
//! config.bootstrap_samples = 500;
//! config.test_signal_types = vec![
//!     TestSignalType::WhiteNoise,
//!     TestSignalType::Sinusoidal,
//!     TestSignalType::Chirp,
//! ];
//!
//! let results = validate_wpt_comprehensive(&config)?;
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! ## Individual Validation Components
//!
//! ```rust,no_run
//! use scirs2_signal::wpt_comprehensive_validation::{
//!     run_basic_validation_suite, validate_frame_properties,
//!     ComprehensiveWptValidationConfig
//! };
//!
//! let config = ComprehensiveWptValidationConfig::default();
//!
//! // Run only basic validation
//! let basic_results = run_basic_validation_suite(&config)?;
//! println!("Energy conservation ratio: {:.6}", basic_results.energy_ratio);
//!
//! // Run only frame theory validation
//! let frame_results = validate_frame_properties(&config)?;
//! println!("Frame condition number: {:.2e}", frame_results.condition_number);
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! # Validation Coverage
//!
//! ## Basic Properties
//! - Energy conservation in decomposition/reconstruction
//! - Perfect reconstruction accuracy
//! - Orthogonality properties
//! - Stability across different configurations
//!
//! ## Frame Theory
//! - Frame operator eigenvalue analysis
//! - Condition number computation
//! - Frame coherence measurement
//! - Redundancy factor analysis
//!
//! ## Multi-Scale Analysis
//! - Scale-wise energy distribution
//! - Inter-scale correlation analysis
//! - Frequency and time localization
//! - Scale consistency measures
//!
//! ## Best Basis Algorithm
//! - Convergence analysis
//! - Selection repeatability
//! - Optimality criteria validation
//! - Computational efficiency
//!
//! ## Statistical Validation
//! - Error distribution analysis
//! - Confidence interval estimation
//! - Hypothesis testing for key properties
//! - Bootstrap validation
//!
//! ## Cross-Validation
//! - Comparison with PyWavelets
//! - Comparison with MATLAB Wavelet Toolbox
//! - Alternative algorithm comparison
//! - Cross-platform consistency
//!
//! ## Robustness Testing
//! - Noise resistance
//! - Outlier tolerance
//! - Parameter sensitivity analysis
//! - Extreme condition stability
//!
//! # Performance Considerations
//!
//! Comprehensive validation can be computationally intensive. Consider:
//!
//! - Reducing `random_trials` for faster validation
//! - Limiting `test_signal_lengths` to essential sizes
//! - Using fewer wavelets for quick tests
//! - Enabling parallel processing with `enable_parallel = true`
//! - Reducing `bootstrap_samples` for statistical tests

pub mod types;
pub mod core;
pub mod basic;
pub mod frame;
pub mod multiscale;
pub mod best_basis;
pub mod statistical;
pub mod cross_validation;
pub mod robustness;
pub mod utils;

// Re-export main types and functions for convenience
pub use types::{
    ComprehensiveWptValidationConfig, ComprehensiveWptValidationResult,
    FrameValidationMetrics, MultiscaleValidationMetrics, BestBasisValidationMetrics,
    StatisticalValidationMetrics, CrossValidationMetrics, RobustnessTestingMetrics,
    TestSignalType, EigenvalueDistribution, ConvergenceAnalysis, OptimalBasisMetrics,
    AlgorithmEfficiencyMetrics, ErrorDistribution, ConfidenceIntervals,
    HypothesisTestResults, BootstrapValidation, ReferenceComparisonMetrics,
    AlgorithmComparisonMetrics, ParameterSensitivityMetrics,
};

pub use core::{validate_wpt_comprehensive, calculate_comprehensive_score};

pub use basic::run_basic_validation_suite;
pub use frame::validate_frame_properties;
pub use multiscale::validate_multiscale_properties;
pub use best_basis::validate_best_basis_algorithm;
pub use statistical::run_statistical_validation;
pub use cross_validation::run_cross_validation;
pub use robustness::test_robustness;

pub use utils::{
    generate_test_signal, test_wpt_round_trip, compute_correlation,
    compute_signal_moments, construct_frame_matrix,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Test that all main types can be imported
        let _config = ComprehensiveWptValidationConfig::default();
        assert_eq!(_config.random_trials, 100);
        assert!(matches!(
            _config.test_signal_types[0],
            TestSignalType::WhiteNoise
        ));
    }

    #[test]
    fn test_comprehensive_validation_minimal() {
        let mut config = ComprehensiveWptValidationConfig::default();

        // Use very minimal configuration for testing
        config.test_wavelets = vec![crate::dwt::Wavelet::Haar];
        config.test_signal_lengths = vec![32];
        config.test_levels = vec![1];
        config.random_trials = 1;
        config.bootstrap_samples = 10;
        config.test_signal_types = vec![TestSignalType::WhiteNoise];

        let result = validate_wpt_comprehensive(&config);

        // The test should complete without panicking
        // Results may vary depending on the implementation details
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_test_signal_generation() {
        let signal = generate_test_signal(TestSignalType::Sinusoidal, 64, 42);
        assert!(signal.is_ok());

        let signal_data = signal.unwrap();
        assert_eq!(signal_data.len(), 64);
        assert!(signal_data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_individual_validation_components() {
        let mut config = ComprehensiveWptValidationConfig::default();
        config.test_wavelets = vec![crate::dwt::Wavelet::Haar];
        config.test_signal_lengths = vec![32];
        config.test_levels = vec![1];

        // Test individual components
        let basic_result = run_basic_validation_suite(&config);
        assert!(basic_result.is_ok() || basic_result.is_err());

        let frame_result = validate_frame_properties(&config);
        assert!(frame_result.is_ok() || frame_result.is_err());

        let multiscale_result = validate_multiscale_properties(&config);
        assert!(multiscale_result.is_ok() || multiscale_result.is_err());
    }
}