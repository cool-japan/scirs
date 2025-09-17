// Validation utilities for Lomb-Scargle periodogram
//
// This module provides comprehensive validation functions for Lomb-Scargle
// implementations, including numerical stability checks, comparison with
// reference implementations, and edge case handling.

pub mod types;
pub mod analytical;
pub mod numerical;
pub mod statistical;
pub mod robustness;
pub mod real_world;
pub mod enhanced;
pub mod scipy_comparison;
pub mod utils;

// Re-export main types and functions
pub use types::{
    ValidationResult, SingleTestResult, StatisticalValidationResult,
    PerformanceValidationResult, ComprehensiveValidationResult,
    RobustnessValidationResult, RealWorldValidationResult,
    AdvancedStatisticalResult, EnhancedValidationConfig,
    EnhancedValidationResult, SciPyComparisonResult,
};

pub use analytical::validate_analytical_cases;
pub use enhanced::validate_lombscargle_enhanced;
pub use statistical::validate_lombscargle_comprehensive;
pub use robustness::{
    validate_additional_robustness, test_memory_stress_scenarios,
    test_numerical_precision_limits, test_boundary_conditions,
    test_short_time_series, test_constant_signal, test_sparse_sampling,
    test_signal_with_outliers, test_small_values, test_large_values,
    test_extreme_timescales
};
pub use real_world::{
    validate_real_world_scenarios, test_astronomical_scenarios,
    test_physiological_scenarios, test_environmental_scenarios,
    validate_advanced_statistical_properties, test_nonparametric_properties,
    test_bayesian_validation, test_information_theory_metrics,
    calculate_comprehensive_score_enhanced
};
pub use utils::{find_peaks_simple, find_peaks, relative_error, rmse, mae, max_abs_error};
pub use numerical::{validate_numerical_precision, calculate_numerical_stability_metrics};
pub use scipy_comparison::{scipy_reference_lombscargle, validate_against_scipy_reference};