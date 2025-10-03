//! Extrapolation Modules
//!
//! This module contains the refactored components of the extrapolation system,
//! broken down into focused, maintainable modules.

// Core modules
pub mod advanced;
pub mod config;
pub mod core;
pub mod types;

// Specialized functionality
pub mod factory;
pub mod physics;

// Re-export main types from types module
pub use types::{
    ARFittingMethod, AdaptiveSelectionCriterion, BoundaryType, DataCharacteristics,
    EnsembleCombinationStrategy, ExtrapolationDirection, ExtrapolationMethod, PhysicsLaw,
};

// Re-export configuration types from config module
pub use config::{
    AdaptiveExtrapolationConfig, AutoregressiveExtrapolationConfig, ConfidenceExtrapolationConfig,
    ConfidenceExtrapolationResult, EnsembleExtrapolationConfig, ExtrapolationConfig,
    ExtrapolationConfigBuilder, ExtrapolationParameters,
};

// Re-export core functionality
pub use advanced::AdvancedExtrapolator;
pub use core::Extrapolator;

// Re-export factory functions
pub use factory::{
    make_adaptive_extrapolator, make_autoregressive_extrapolator, make_confidence_extrapolator,
    make_conservative_extrapolator, make_cubic_extrapolator, make_ensemble_extrapolator,
    make_exponential_extrapolator, make_high_confidence_extrapolator, make_linear_extrapolator,
    make_monotonic_extrapolator, make_nearest_extrapolator, make_oscillatory_extrapolator,
    make_periodic_extrapolator, make_power_law_extrapolator, make_reflection_extrapolator,
    make_robust_ensemble_extrapolator, make_smooth_function_extrapolator, make_zeros_extrapolator,
};

// Re-export physics-informed functions
pub use physics::{
    analyze_physics_characteristics, make_boundary_layer_extrapolator,
    make_boundary_preserving_extrapolator, make_conservation_law_extrapolator,
    make_constrained_extrapolator, make_diffusion_equation_extrapolator,
    make_physics_informed_extrapolator, make_smart_adaptive_extrapolator,
    make_wave_equation_extrapolator,
};

// Convenience re-exports for backward compatibility
pub use advanced::AdvancedExtrapolator as AdvancedExtrapolationEngine;
pub use core::Extrapolator as ExtrapolationEngine;

/// Creates a default linear extrapolator for quick usage
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
/// * `lower_value` - Function value at lower boundary
/// * `upper_value` - Function value at upper boundary
///
/// # Returns
///
/// A basic extrapolator configured for linear extrapolation
pub fn create_basic_extrapolator<T: scirs2_core::numeric::Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
) -> Extrapolator<T> {
    use crate::extrapolation_modules::factory::make_linear_extrapolator;

    // Estimate derivatives from boundary values
    let domain_width = upper_bound - lower_bound;
    let slope = (upper_value - lower_value) / domain_width;

    make_linear_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        slope,
        slope,
    )
}

/// Creates an advanced extrapolator with ensemble methods for robust extrapolation
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
/// * `lower_value` - Function value at lower boundary
/// * `upper_value` - Function value at upper boundary
/// * `lower_derivative` - Derivative at lower boundary
/// * `upper_derivative` - Derivative at upper boundary
///
/// # Returns
///
/// An advanced extrapolator with multiple methods for robust extrapolation
pub fn create_robust_extrapolator<
    T: scirs2_core::numeric::Float + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
) -> AdvancedExtrapolator<T> {
    use crate::extrapolation_modules::factory::make_robust_ensemble_extrapolator;

    make_robust_ensemble_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
    )
}

/// Creates an extrapolator with automatic method selection based on data characteristics
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
/// * `lower_value` - Function value at lower boundary
/// * `upper_value` - Function value at upper boundary
/// * `lower_derivative` - Derivative at lower boundary
/// * `upper_derivative` - Derivative at upper boundary
/// * `data_characteristics` - Analyzed characteristics of the data
///
/// # Returns
///
/// An advanced extrapolator with method selection based on data analysis
pub fn create_smart_extrapolator<
    T: scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Display
        + std::default::Default
        + std::ops::AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    data_characteristics: &DataCharacteristics<T>,
) -> AdvancedExtrapolator<T> {
    use crate::extrapolation_modules::physics::make_smart_adaptive_extrapolator;

    make_smart_adaptive_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
        data_characteristics,
    )
}

/// Analyzes data to determine appropriate extrapolation method
///
/// # Arguments
///
/// * `values` - Array of function values
/// * `gradients` - Optional array of gradients
///
/// # Returns
///
/// Data characteristics for informed extrapolation method selection
pub fn analyze_data_for_extrapolation<
    T: scirs2_core::numeric::Float + scirs2_core::numeric::FromPrimitive,
>(
    values: &[T],
    gradients: Option<&[T]>,
) -> DataCharacteristics<T> {
    use crate::extrapolation_modules::physics::analyze_physics_characteristics;

    analyze_physics_characteristics(values, gradients, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_extrapolator_creation() {
        let extrapolator = create_basic_extrapolator(0.0, 10.0, 0.0, 10.0);

        // Test that it was created successfully
        assert_eq!(extrapolator.lower_bound(), 0.0);
        assert_eq!(extrapolator.upper_bound(), 10.0);
    }

    #[test]
    fn test_linear_extrapolation() {
        let extrapolator = make_linear_extrapolator(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, -5.0, epsilon = 1e-10);

        // Test upper extrapolation
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_extrapolation() {
        let extrapolator = Extrapolator::new(
            0.0,
            10.0,
            5.0,
            15.0,
            ExtrapolationMethod::Constant,
            ExtrapolationMethod::Constant,
        );

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);

        // Test upper extrapolation
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zeros_extrapolation() {
        let extrapolator = make_zeros_extrapolator(0.0, 10.0);

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);

        // Test upper extrapolation
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_periodic_extrapolation() {
        let extrapolator = make_periodic_extrapolator(0.0, 1.0, Some(1.0));

        // Test that periodic extrapolation doesn't fail
        // (actual behavior depends on implementation details)
        let result = extrapolator.extrapolate(-0.3);
        assert!(result.is_ok());

        let result = extrapolator.extrapolate(1.3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradient_vec = vec![1.0, 1.0, 1.0, 1.0];
        let gradients = Some(gradient_vec.as_slice());

        let characteristics = analyze_data_for_extrapolation(&values, gradients);

        // Monotonic increasing data
        assert!(characteristics.is_monotonic);
        assert!(!characteristics.is_periodic);
        assert!(!characteristics.is_oscillatory);
    }

    #[test]
    fn test_robust_extrapolator_creation() {
        let extrapolator = create_robust_extrapolator(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);

        // Test that advanced extrapolator was created
        assert!(extrapolator.has_ensemble());

        // Test extrapolation works
        let result = extrapolator.extrapolate_advanced(-5.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_domain_checking() {
        let extrapolator = create_basic_extrapolator(0.0, 10.0, 0.0, 10.0);

        // Test domain checking
        assert!(extrapolator.is_in_domain(5.0));
        assert!(!extrapolator.is_in_domain(-1.0));
        assert!(!extrapolator.is_in_domain(11.0));

        // Test domain width
        assert_abs_diff_eq!(extrapolator.domain_width(), 10.0, epsilon = 1e-10);
    }
}
