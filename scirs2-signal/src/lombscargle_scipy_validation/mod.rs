//! Comprehensive Lomb-Scargle validation against SciPy reference implementation
//!
//! This module provides detailed validation of our Lomb-Scargle periodogram implementation
//! by comparing results directly with SciPy's `scipy.signal.lombscargle` function.
//!
//! Key validation areas:
//! - Numerical accuracy across different data types and signal lengths
//! - Edge cases (very sparse sampling, high dynamic range, etc.)
//! - Different normalization methods
//! - Performance and memory characteristics
//! - Statistical properties (false alarm rate, detection power)

// Core module organization
pub mod types;
pub mod core;
pub mod accuracy;
pub mod normalization;
pub mod edge_cases;
pub mod statistical;
pub mod performance;
pub mod advanced;
pub mod utils;
pub mod reporting;

// Re-export all types for backward compatibility
pub use types::*;

// Re-export main validation functions
pub use core::{validate_lombscargle_against_scipy, calculate_overall_summary};

// Re-export specific validation functions
pub use accuracy::{validate_basic_accuracy, validate_single_case, compute_reference_lombscargle};
pub use normalization::{validate_normalization_methods, validate_single_normalization_case};
pub use edge_cases::{
    validate_edge_cases, test_sparse_sampling, test_extreme_dynamic_range,
    test_short_time_series, test_high_frequency_resolution, calculate_edge_case_stability_score
};
pub use statistical::{
    validate_statistical_properties, estimate_false_alarm_rate,
    estimate_detection_power, validate_confidence_intervals
};
pub use performance::validate_performance_characteristics;

// Re-export advanced validation functions
pub use advanced::{
    validate_lombscargle_advanced, test_numerical_conditioning, test_aliasing_effects,
    test_astronomical_scenarios, test_phase_coherence, quantify_uncertainty,
    test_frequency_resolution
};

// Re-export utility functions
pub use utils::{
    calculate_error_metrics, calculate_correlation, calculate_normalization_consistency,
    find_peaks
};

// Re-export reporting functions
pub use reporting::run_comprehensive_validation;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_validation() {
        let config = ScipyValidationConfig {
            test_lengths: vec![32, 64],
            sampling_frequencies: vec![10.0],
            test_frequencies: vec![1.0],
            monte_carlo_trials: 5,
            ..Default::default()
        };

        let results = validate_lombscargle_against_scipy(&config).unwrap();
        assert!(results.accuracy_results.correlation > 0.9);
        assert!(results.summary.overall_score > 50.0);
    }

    #[test]
    fn test_reference_implementation() {
        let t = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let freqs = vec![1.0, 2.0, 5.0];

        let result = compute_reference_lombscargle(&t, &signal, &freqs).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x: &f64| x.is_finite() && x >= 0.0));
    }
}