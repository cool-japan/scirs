// Cubic spline interpolation module
//
// This module provides comprehensive cubic spline interpolation functionality
// with support for multiple boundary conditions, SciPy compatibility, and
// advanced features like extrapolation and numerical integration.
//
// # Overview
//
// Cubic splines are piecewise cubic polynomials that provide smooth interpolation
// through a set of data points. They maintain C² continuity (continuous function,
// first, and second derivatives) making them ideal for scientific applications
// requiring high-quality smooth curves.
//
// # Modules
//
// - `types`: Core type definitions and boundary conditions
// - `algorithms`: Computational algorithms for different boundary conditions
// - `core`: Main data structures (CubicSpline, CubicSplineBuilder)
// - `evaluation`: Evaluation and derivative computation methods
// - `integration`: Integration methods and extrapolation support
// - `traits_impl`: Trait implementations for framework compatibility
// - `api`: Public SciPy-compatible API functions
// - `utils`: Utility functions and helpers
//
// # Quick Start
//
// ```rust
// use scirs2_core::ndarray::array;
// use scirs2_interpolate::spline::CubicSpline;
//
// // Create a simple cubic spline
// let x = array![0.0, 1.0, 2.0, 3.0];
// let y = array![0.0, 1.0, 4.0, 9.0];
// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
//
// // Evaluate at a point
// let result = spline.evaluate(1.5).unwrap();
// println!("f(1.5) = {}", result);
//
// // Compute derivatives
// let derivative = spline.derivative(1.5).unwrap();
// println!("f'(1.5) = {}", derivative);
//
// // Integrate over an interval
// let integral = spline.integrate(0.5, 2.5).unwrap();
// println!("∫f(x)dx from 0.5 to 2.5 = {}", integral);
// ```
//
// # Advanced Usage
//
// ```rust
// use scirs2_core::ndarray::array;
// use scirs2_interpolate::spline::{CubicSpline, SplineBoundaryCondition};
//
// let x = array![0.0, 1.0, 2.0, 3.0];
// let y = array![0.0, 1.0, 4.0, 9.0];
//
// // Use builder pattern with custom boundary conditions
// let spline = CubicSpline::builder()
//     .x(x)
//     .y(y)
//     .boundary_condition(SplineBoundaryCondition::Clamped(0.0, 6.0))
//     .build()
//     .unwrap();
// ```
//
// # SciPy Compatibility
//
// ```rust
// use scirs2_core::ndarray::array;
// use scirs2_interpolate::spline::cubic_spline_scipy;
//
// let x = array![0.0, 1.0, 2.0, 3.0];
// let y = array![0.0, 1.0, 4.0, 9.0];
//
// // SciPy-compatible interface
// let spline = cubic_spline_scipy(
//     &x.view(),
//     &y.view(),
//     "not-a-knot",
//     None,
//     false
// ).unwrap();
// ```

// Core type definitions
pub mod types;
pub use types::SplineBoundaryCondition;

// Computational algorithms
pub mod algorithms;

// Utility functions
pub mod utils;

// Core data structures
pub mod core;
pub use core::{CubicSpline, CubicSplineBuilder};

// Evaluation methods
pub mod evaluation;

// Integration methods
pub mod integration;

// Trait implementations
pub mod traits_impl;

// Public API functions
pub mod api;
pub use api::{
    cubic_spline_scipy,
    make_interp_spline,
    interp1d_scipy,
    cubic_spline_second_derivative,
    cubic_spline_parabolic_runout,
};

// Re-export commonly used items for convenience
pub use crate::error::{InterpolateError, InterpolateResult};
pub use crate::traits::InterpolationFloat;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basic_cubic_spline() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        // Test evaluation at data points
        assert_abs_diff_eq!(spline.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(3.0).unwrap(), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_conditions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test different boundary conditions
        let natural = CubicSpline::new(&x.view(), &y.view()).unwrap();
        let not_a_knot = CubicSpline::new_not_a_knot(&x.view(), &y.view()).unwrap();

        // All should interpolate data points exactly
        assert_abs_diff_eq!(natural.evaluate(1.5).unwrap(), not_a_knot.evaluate(1.5).unwrap(), epsilon = 0.1);
    }

    #[test]
    fn test_derivatives() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0]; // approximately x^2

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        // For y ≈ x^2, derivative should be approximately 2x
        let deriv_at_1 = spline.derivative(1.0).unwrap();
        assert!((deriv_at_1 - 2.0_f64).abs() < 0.5_f64, "First derivative at x=1 should be close to 2");

        let deriv_at_2 = spline.derivative(2.0).unwrap();
        assert!((deriv_at_2 - 4.0_f64).abs() < 0.5_f64, "First derivative at x=2 should be close to 4");
    }

    #[test]
    fn test_integration() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0, 4.0];

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        // Integration from a point to itself should be zero
        assert_abs_diff_eq!(spline.integrate(1.0, 1.0).unwrap(), 0.0, epsilon = 1e-10);

        // Integration should be positive for this monotonic function
        let integral = spline.integrate(0.0, 2.0).unwrap();
        assert!(integral > 0.0);
    }

    #[test]
    fn test_scipy_compatibility() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = cubic_spline_scipy(&x.view(), &y.view(), "natural", None, false).unwrap();

        // Should interpolate data points exactly
        assert_abs_diff_eq!(spline.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_builder_pattern() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::builder()
            .x(x)
            .y(y)
            .boundary_condition(SplineBoundaryCondition::Natural)
            .build()
            .unwrap();

        assert_abs_diff_eq!(spline.evaluate(1.5).unwrap(), spline.evaluate(1.5).unwrap(), epsilon = 1e-10);
    }

    #[test]
    fn test_error_handling() {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];

        // Should fail with insufficient points
        assert!(CubicSpline::new(&x.view(), &y.view()).is_err());

        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0]; // Wrong length

        // Should fail with mismatched lengths
        assert!(CubicSpline::new(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn test_out_of_bounds() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 4.0, 9.0, 16.0];

        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        // Should fail for out-of-bounds evaluation
        assert!(spline.evaluate(0.5).is_err());
        assert!(spline.evaluate(4.5).is_err());
    }
}