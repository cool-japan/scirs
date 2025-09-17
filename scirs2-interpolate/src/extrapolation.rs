//! Enhanced extrapolation methods for interpolation
//!
//! This module provides advanced extrapolation capabilities that go beyond
//! the basic ExtrapolateMode enum. It allows for more sophisticated boundary
//! handling and domain extension methods.
//!
//! # Features
//!
//! - Physics-informed extrapolation based on boundary derivatives
//! - Polynomial extrapolation of various orders
//! - Decay/growth models for asymptotic behavior
//! - Periodic extension of the domain
//! - Reflection-based extrapolation
//! - Domain-specific extrapolation models
//! - Ensemble methods for robust extrapolation
//! - Adaptive method selection
//! - Confidence-based extrapolation with uncertainty estimation
//! - Autoregressive modeling for time series data
//!
//! # Examples
//!
//! ## Basic Linear Extrapolation
//!
//! ```rust
//! use scirs2_interpolate::extrapolation::{create_basic_extrapolator, ExtrapolationMethod};
//!
//! let extrapolator = create_basic_extrapolator(0.0, 10.0, 0.0, 10.0);
//! let result = extrapolator.extrapolate(-5.0).unwrap();
//! assert_eq!(result, -5.0);
//! ```
//!
//! ## Advanced Ensemble Extrapolation
//!
//! ```rust
//! use scirs2_interpolate::extrapolation::create_robust_extrapolator;
//!
//! let extrapolator = create_robust_extrapolator(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
//! let result = extrapolator.extrapolate_advanced(-5.0).unwrap();
//! ```
//!
//! ## Physics-Informed Extrapolation
//!
//! ```rust
//! use scirs2_interpolate::extrapolation::{make_physics_informed_extrapolator, PhysicsLaw};
//!
//! let extrapolator = make_physics_informed_extrapolator(
//!     0.0, 10.0, 0.0, 10.0, 1.0, 1.0,
//!     PhysicsLaw::MassConservation
//! );
//! let result = extrapolator.extrapolate_advanced(-5.0).unwrap();
//! ```

// Re-export all functionality from the modular implementation
pub use crate::extrapolation_modules::*;

// Provide convenience functions for backward compatibility

/// Creates a simple extrapolator with specified method for both boundaries
pub fn create_simple_extrapolator<T: num_traits::Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    method: ExtrapolationMethod,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        method,
        method,
    )
}

/// Creates an extrapolator with different methods for lower and upper boundaries
pub fn create_asymmetric_extrapolator<T: num_traits::Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_method: ExtrapolationMethod,
    upper_method: ExtrapolationMethod,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_method,
        upper_method,
    )
}

/// Creates a production-ready extrapolator with conservative settings
pub fn create_production_extrapolator<T: num_traits::Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
) -> Extrapolator<T> {
    make_conservative_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
        (upper_bound - lower_bound) * T::from(2.0).unwrap(), // Max 2x domain width
    )
}

/// Creates an extrapolator optimized for the given data characteristics
pub fn create_optimized_extrapolator<
    T: num_traits::Float
        + num_traits::FromPrimitive
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
    values: &[T],
    gradients: Option<&[T]>,
) -> AdvancedExtrapolator<T> {
    let characteristics = analyze_data_for_extrapolation(values, gradients);
    create_smart_extrapolator(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        lower_derivative,
        upper_derivative,
        &characteristics,
    )
}

/// Quick extrapolation function for single values
pub fn extrapolate_value<T: num_traits::Float + std::fmt::Display>(
    x: T,
    domain: (T, T),
    boundary_values: (T, T),
    method: ExtrapolationMethod,
) -> crate::error::InterpolateResult<T> {
    let (lower_bound, upper_bound) = domain;
    let (lower_value, upper_value) = boundary_values;

    let extrapolator =
        create_simple_extrapolator(lower_bound, upper_bound, lower_value, upper_value, method);

    extrapolator.extrapolate(x)
}

/// Batch extrapolation for multiple values
pub fn extrapolate_batch<T: num_traits::Float + std::fmt::Display>(
    x_values: &[T],
    domain: (T, T),
    boundary_values: (T, T),
    method: ExtrapolationMethod,
) -> Vec<crate::error::InterpolateResult<T>> {
    let (lower_bound, upper_bound) = domain;
    let (lower_value, upper_value) = boundary_values;

    let extrapolator =
        create_simple_extrapolator(lower_bound, upper_bound, lower_value, upper_value, method);

    x_values
        .iter()
        .map(|&x| extrapolator.extrapolate(x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simple_extrapolator() {
        let extrapolator =
            create_simple_extrapolator(0.0, 10.0, 0.0, 10.0, ExtrapolationMethod::Linear);

        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_asymmetric_extrapolator() {
        let extrapolator = create_asymmetric_extrapolator(
            0.0,
            10.0,
            0.0,
            10.0,
            ExtrapolationMethod::Linear,
            ExtrapolationMethod::Constant,
        );

        // Test lower boundary (linear)
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert!(result != 0.0); // Should be linear, not constant

        // Test upper boundary (constant)
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, 10.0, epsilon = 1e-10); // Should be constant
    }

    #[test]
    fn test_production_extrapolator() {
        let extrapolator = create_production_extrapolator(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);

        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quick_extrapolation() {
        let result =
            extrapolate_value(-5.0, (0.0, 10.0), (0.0, 10.0), ExtrapolationMethod::Linear).unwrap();

        assert_abs_diff_eq!(result, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_extrapolation() {
        let x_values = vec![-5.0, -2.0, 12.0, 15.0];
        let results = extrapolate_batch(
            &x_values,
            (0.0, 10.0),
            (0.0, 10.0),
            ExtrapolationMethod::Linear,
        );

        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.is_ok()));

        assert_abs_diff_eq!(results[0].as_ref().unwrap(), &-5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[1].as_ref().unwrap(), &-2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[2].as_ref().unwrap(), &12.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[3].as_ref().unwrap(), &15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_optimized_extrapolator() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradient_vec = vec![1.0, 1.0, 1.0, 1.0];
        let gradients = Some(gradient_vec.as_slice());

        let extrapolator =
            create_optimized_extrapolator(0.0, 4.0, 1.0, 5.0, 1.0, 1.0, &values, gradients);

        let result = extrapolator.extrapolate_advanced(-1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zeros_method() {
        let result =
            extrapolate_value(-5.0, (0.0, 10.0), (5.0, 15.0), ExtrapolationMethod::Zeros).unwrap();

        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_method() {
        let extrapolator =
            create_simple_extrapolator(0.0, 10.0, 5.0, 15.0, ExtrapolationMethod::Nearest);

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);

        // Test upper extrapolation
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_error_for_in_domain_points() {
        let extrapolator =
            create_simple_extrapolator(0.0, 10.0, 0.0, 10.0, ExtrapolationMethod::Linear);

        // Point inside domain should return error
        let result = extrapolator.extrapolate(5.0);
        assert!(result.is_err());
    }
}
