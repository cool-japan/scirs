//! Advanced-comprehensive validation suite for Wavelet Packet Transform implementations
//!
//! This module provides the most advanced validation framework for WPT with:
//! - SIMD operation correctness verification across platforms
//! - Statistical significance testing for basis selection algorithms
//! - Memory safety and performance regression detection
//! - Cross-platform numerical consistency validation
//! - Advanced mathematical property verification (tight frames, perfect reconstruction)
//! - Machine learning-based anomaly detection in coefficient patterns
//! - Real-time processing validation with quality guarantees
//!
//! # Module Organization
//!
//! - [`types`] - All data structures and type definitions
//! - [`defaults`] - Default implementations for complex types
//! - [`core`] - Main validation orchestration and entry point
//! - [`mathematical`] - Mathematical property validation
//! - [`simd`] - SIMD implementation validation
//! - [`platform`] - Cross-platform consistency validation
//! - [`statistical`] - Statistical analysis of basis selection
//! - [`performance`] - Performance regression analysis
//! - [`memory`] - Memory safety validation
//! - [`realtime`] - Real-time processing validation
//! - [`utils`] - Utility functions for signal generation and analysis
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use scirs2_signal::wpt_super_validation::{run_advanced_wpt_validation, AdvancedWptValidationConfig};
//!
//! let config = AdvancedWptValidationConfig::default();
//! let results = run_advanced_wpt_validation(&config)?;
//!
//! match results.overall_status {
//!     ValidationStatus::Pass => println!("All validations passed!"),
//!     ValidationStatus::PassWithWarnings => println!("Validation passed with warnings"),
//!     ValidationStatus::Fail => println!("Validation failed"),
//!     ValidationStatus::Incomplete => println!("Validation incomplete"),
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Customized Validation
//!
//! ```rust
//! use scirs2_signal::wpt_super_validation::*;
//!
//! let mut config = AdvancedWptValidationConfig::default();
//! config.validate_realtime = true;
//! config.tolerance = 1e-15;
//! config.monte_carlo_samples = 50000;
//!
//! let results = run_advanced_wpt_validation(&config)?;
//!
//! // Check specific validation results
//! if results.mathematical_properties.perfect_reconstruction.max_error > 1e-12 {
//!     println!("Perfect reconstruction error too high!");
//! }
//!
//! if !results.simd_validation.operation_correctness.is_empty() {
//!     println!("SIMD operations validated: {} functions",
//!              results.simd_validation.operation_correctness.len());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Declare all submodules
pub mod types;
pub mod defaults;
pub mod core;
pub mod mathematical;
pub mod simd;
pub mod platform;
pub mod statistical;
pub mod performance;
pub mod memory;
pub mod realtime;
pub mod utils;

// Re-export main types for public API
pub use types::*;

// Re-export main functions
pub use core::{
    run_advanced_wpt_validation,
    run_basic_wpt_validation,
    determine_overall_validation_status,
    calculate_coefficient_energy,
    calculate_subband_energy_distribution,
    calculate_energy_concentration,
    calculate_energy_leakage,
};

// Re-export mathematical validation functions
pub use mathematical::{
    validate_mathematical_properties_comprehensive,
    validate_perfect_reconstruction_comprehensive,
    validate_tight_frame_properties,
    validate_advanced_orthogonality,
    validate_energy_conservation_comprehensive,
    analyze_coefficient_distributions,
};

// Re-export SIMD validation functions
pub use simd::{
    validate_simd_implementations_comprehensive,
    validate_simd_vs_scalar_accuracy,
    validate_individual_simd_operations,
    test_simd_convolution,
    test_simd_downsampling,
    test_simd_upsampling,
    test_simd_thresholding,
    test_simd_energy_calculation,
    assess_numerical_stability,
};

// Re-export platform validation functions
pub use platform::validate_cross_platform_consistency_comprehensive;

// Re-export statistical validation functions
pub use statistical::{
    validate_statistical_properties_comprehensive,
    analyze_basis_selection_consistency,
    validate_cost_functions,
    perform_significance_testing,
    analyze_robustness,
};

// Re-export performance validation functions
pub use performance::{
    analyze_performance_regression_comprehensive,
    analyze_historical_performance,
    run_performance_benchmarks,
    analyze_scalability,
    analyze_resource_utilization,
};

// Re-export memory validation functions
pub use memory::{
    validate_memory_safety_comprehensive,
    detect_memory_leaks,
    verify_buffer_safety,
    detect_use_after_free,
    detect_double_free,
    verify_memory_alignment,
    calculate_safety_score,
};

// Re-export real-time validation functions
pub use realtime::{
    validate_realtime_processing_comprehensive,
    analyze_latency,
    analyze_jitter,
    analyze_throughput,
    assess_realtime_quality,
    calculate_realtime_score,
};

// Re-export utility functions
pub use utils::{
    generate_test_signal,
    calculate_reconstruction_error,
    calculate_vector_norm,
    calculate_mean,
    calculate_std_dev,
    calculate_snr_db,
    normalize_signal,
    calculate_correlation,
    generate_hanning_window,
    apply_window,
    zero_pad,
    trim_signal,
    calculate_energy,
    calculate_power,
    find_max_abs,
    is_signal_valid,
    generate_random_signal,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that main types are accessible
        let _config = AdvancedWptValidationConfig::default();
        let _status = ValidationStatus::Pass;

        // Test that main functions are accessible
        let config = AdvancedWptValidationConfig {
            validate_cross_platform: false,
            validate_performance_regression: false,
            validate_realtime: false,
            ..Default::default()
        };

        let result = run_advanced_wpt_validation(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_comprehensive_validation_integration() {
        // Test that all validation modules work together
        let mut config = AdvancedWptValidationConfig::default();

        // Enable all validations but keep them lightweight for testing
        config.validate_mathematical_properties = true;
        config.validate_simd = true;
        config.validate_statistical = true;
        config.validate_memory_safety = true;

        // Disable heavy validations for faster testing
        config.validate_cross_platform = false;
        config.validate_performance_regression = false;
        config.validate_realtime = false;

        // Reduce test data for faster execution
        config.monte_carlo_samples = 100;
        config.max_levels_to_test = vec![2, 3];

        let result = run_advanced_wpt_validation(&config);
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert_eq!(validation.overall_status, ValidationStatus::Pass);

        // Verify that enabled validations have meaningful results
        assert!(validation.mathematical_properties.perfect_reconstruction.max_error >= 0.0);
        assert!(validation.simd_validation.simd_scalar_accuracy >= 0.0);
        assert!(validation.memory_safety.safety_score > 0.0);
    }

    #[test]
    fn test_utility_functions_integration() {
        // Test signal generation
        let signal_config = TestSignalConfig {
            signal_type: TestSignalType::Sinusoid,
            length: 128,
            parameters: [("frequency".to_string(), 5.0)].iter().cloned().collect(),
        };

        let signal = generate_test_signal(&signal_config).unwrap();
        assert_eq!(signal.len(), 128);
        assert!(is_signal_valid(&signal));

        // Test signal analysis
        let energy = calculate_energy(&signal);
        let power = calculate_power(&signal);
        assert!(energy > 0.0);
        assert!(power > 0.0);
        assert!((energy / signal.len() as f64 - power).abs() < 1e-10);

        // Test signal processing
        let normalized = normalize_signal(&signal);
        let norm = calculate_vector_norm(&normalized);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_handling() {
        // Test that validation handles errors gracefully
        let config = AdvancedWptValidationConfig {
            test_signals: vec![], // Empty test signals should not cause crashes
            ..Default::default()
        };

        let result = run_advanced_wpt_validation(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_configuration_limits() {
        // Test extreme configuration values
        let mut config = AdvancedWptValidationConfig::default();
        config.tolerance = 1e-20; // Very strict tolerance
        config.monte_carlo_samples = 1; // Minimal samples
        config.max_levels_to_test = vec![1]; // Minimal levels

        let result = run_advanced_wpt_validation(&config);
        assert!(result.is_ok());
    }
}