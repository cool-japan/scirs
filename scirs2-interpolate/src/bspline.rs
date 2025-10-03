//! B-spline basis functions and B-spline curves
//!
//! This module provides functionality for B-spline basis functions and
//! univariate spline interpolation using B-splines.
//!
//! The main class is `BSpline`, which represents a univariate spline as a
//! linear combination of B-spline basis functions:
//!
//! S(x) = Σ(j=0..n-1) c_j * B_{j,k;t}(x)
//!
//! where B_{j,k;t} are B-spline basis functions of degree k with knots t,
//! and c_j are spline coefficients.
//!
//! # Modular Organization
//!
//! This module has been refactored into focused submodules:
//! - `types`: Core type definitions and workspace management
//! - `core`: Main BSpline struct and basic operations
//! - `evaluation`: Advanced evaluation methods and optimizations
//! - `factory`: Factory functions for creating B-splines
//! - `solvers`: Linear algebra routines for B-spline computations

// Import the modular implementation
use crate::bspline_modules;

// Re-export the public API
pub use crate::bspline_modules::{
    condition_number, generate_knots, lu_decomposition, make_auto_bspline, make_interp_bspline,
    make_lsq_bspline, make_periodic_bspline, make_smoothing_bspline, matrix_multiply,
    matrix_vector_multiply, solve_least_squares, solve_linear_system, solve_multiple_rhs,
    solve_with_lu, transpose_matrix, BSpline, BSplineWorkspace, BSplineWorkspaceBuilder,
    EvaluationStats, ExtrapolateMode, WorkspaceConfig, WorkspaceMemoryStats, WorkspaceProvider,
};

// Convenience re-exports for common patterns
pub use crate::bspline_modules::prelude::*;

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

// Type aliases for convenience
pub type BSplineF32 = BSpline<f32>;
pub type BSplineF64 = BSpline<f64>;

/// Create a simple linear B-spline for quick interpolation
///
/// This is a convenience function for creating degree-1 B-splines.
pub fn linear_bspline<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + scirs2_core::numeric::Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    make_interp_bspline(x, y, 1, ExtrapolateMode::Extrapolate)
}

/// Create a cubic B-spline for smooth interpolation
///
/// This is a convenience function for creating degree-3 B-splines.
pub fn cubic_bspline<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + scirs2_core::numeric::Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    make_interp_bspline(x, y, 3, ExtrapolateMode::Extrapolate)
}

/// Module information and version
pub mod info {
    pub use crate::bspline_modules::info::*;
}

// Trait implementations using the modular structure

// Implementation of SplineInterpolator trait for BSpline

impl<T> crate::traits::SplineInterpolator<T> for BSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Send
        + Sync
        + 'static
        + crate::traits::InterpolationFloat,
{
    fn derivative(
        &self,
        query_points: &ArrayView2<T>,
        order: usize,
    ) -> crate::InterpolateResult<Vec<T>> {
        if query_points.ncols() != 1 {
            return Err(crate::InterpolateError::invalid_input(
                "BSpline only supports 1D interpolation",
            ));
        }

        let mut results = Vec::with_capacity(query_points.nrows());
        for row in query_points.outer_iter() {
            let x = row[0];
            let deriv = self.derivative(x, order)?;
            results.push(deriv);
        }
        Ok(results)
    }

    fn integrate(&self, bounds: &[(T, T)]) -> crate::InterpolateResult<Vec<T>> {
        let mut results = Vec::with_capacity(bounds.len());
        for &(a, b) in bounds {
            let integral = self.integrate(a, b)?;
            results.push(integral);
        }
        Ok(results)
    }

    fn antiderivative(
        &self,
    ) -> crate::InterpolateResult<Box<dyn crate::traits::SplineInterpolator<T>>> {
        let antideriv = self.antiderivative(1)?;
        Ok(Box::new(antideriv))
    }

    fn find_roots(&self, bounds: &[(T, T)], tolerance: T) -> crate::InterpolateResult<Vec<T>> {
        use crate::utils::find_multiple_roots;

        let mut all_roots = Vec::new();

        for &(a, b) in bounds {
            if a >= b {
                continue;
            }

            let eval_fn = |x: T| -> crate::InterpolateResult<T> { self.evaluate(x) };

            match find_multiple_roots(a, b, tolerance, 10, eval_fn) {
                Ok(mut roots) => all_roots.append(&mut roots),
                Err(_) => continue,
            }
        }

        all_roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_roots.dedup_by(|a, b| (*a - *b).abs() < tolerance);

        Ok(all_roots)
    }

    fn find_extrema(
        &self,
        bounds: &[(T, T)],
        tolerance: T,
    ) -> crate::InterpolateResult<Vec<(T, T, crate::traits::ExtremaType)>> {
        use crate::utils::find_multiple_roots;

        let mut extrema = Vec::new();

        for &(a, b) in bounds {
            if a >= b {
                continue;
            }

            let deriv_fn = |x: T| -> crate::InterpolateResult<T> { self.derivative(x, 1) };

            let critical_points = match find_multiple_roots(a, b, tolerance, 20, deriv_fn) {
                Ok(points) => points,
                Err(_) => continue,
            };

            for cp in critical_points {
                if cp < a || cp > b {
                    continue;
                }

                let second_deriv = match self.derivative(cp, 2) {
                    Ok(d2) => d2,
                    Err(_) => continue,
                };

                let f_value = match self.evaluate(cp) {
                    Ok(val) => val,
                    Err(_) => continue,
                };

                let extrema_type = if second_deriv > T::zero() {
                    crate::traits::ExtremaType::Minimum
                } else if second_deriv < T::zero() {
                    crate::traits::ExtremaType::Maximum
                } else {
                    crate::traits::ExtremaType::InflectionPoint
                };

                extrema.push((cp, f_value, extrema_type));
            }
        }

        extrema.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(extrema)
    }
}

// Implementation of Interpolator trait for BSpline
impl<T> crate::traits::Interpolator<T> for BSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Send
        + Sync
        + 'static
        + crate::traits::InterpolationFloat,
{
    fn evaluate(&self, query_points: &ArrayView2<T>) -> crate::InterpolateResult<Vec<T>> {
        if query_points.ncols() != 1 {
            return Err(crate::InterpolateError::invalid_input(
                "BSpline only supports 1D interpolation",
            ));
        }

        let mut results = Vec::with_capacity(query_points.nrows());
        for row in query_points.outer_iter() {
            let x = row[0];
            let value = self.evaluate(x)?;
            results.push(value);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        1
    }

    fn len(&self) -> usize {
        self.coefficients().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Interpolator;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_bspline() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2

        let spline = linear_bspline(&x.view(), &y.view()).unwrap();

        // Test evaluation
        let value = spline.evaluate(2.5);
        assert!(value.is_ok());

        // Test that it's degree 1
        assert_eq!(spline.degree(), 1);
    }

    #[test]
    fn test_cubic_bspline() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 8.0, 27.0, 64.0]; // y = x^3

        let spline = cubic_bspline(&x.view(), &y.view()).unwrap();

        // Test evaluation
        let value = spline.evaluate(2.5);
        assert!(value.is_ok());

        // Test that it's degree 3
        assert_eq!(spline.degree(), 3);
    }

    #[test]
    fn test_modular_api_integration() {
        // Test the complete workflow using the modular API
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2

        // Create interpolating B-spline
        let spline = make_interp_bspline(
            &x.view(),
            &y.view(),
            3, // cubic
            ExtrapolateMode::Extrapolate,
        );
        assert!(spline.is_ok());

        let spline = spline.unwrap();

        // Test evaluation
        let test_point = 2.5;
        let value = spline.evaluate(test_point);
        assert!(value.is_ok());

        // Test derivative
        let deriv = spline.derivative(test_point, 1);
        assert!(deriv.is_ok());

        // Test integration
        let integral = spline.integrate(0.0, 4.0);
        assert!(integral.is_ok());
    }

    #[test]
    fn test_workspace_optimization() {
        // Test workspace-based evaluation
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 2.0, 1.0];

        let spline =
            make_interp_bspline(&x.view(), &y.view(), 2, ExtrapolateMode::Extrapolate).unwrap();

        let workspace = BSplineWorkspace::new();

        // Test workspace evaluation
        let value1 = spline.evaluate_with_workspace(1.5, &workspace);
        assert!(value1.is_ok());

        let value2 = spline.evaluate_with_workspace(2.5, &workspace);
        assert!(value2.is_ok());

        // Check that workspace recorded evaluations
        let stats = workspace.get_statistics();
        assert!(stats.evaluation_count >= 2);
    }

    #[test]
    fn test_advanced_evaluation() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 8.0, 27.0, 64.0, 125.0]; // y = x^3

        let spline =
            make_interp_bspline(&x.view(), &y.view(), 3, ExtrapolateMode::Extrapolate).unwrap();

        // Test fast recursive evaluation
        let fast_value = spline.evaluate_fast_recursive(2.5);
        assert!(fast_value.is_ok());

        // Test batch evaluation
        let test_points = array![1.5, 2.5, 3.5];
        let batch_values = spline.evaluate_array(&test_points.view());
        assert!(batch_values.is_ok());
        assert_eq!(batch_values.unwrap().len(), 3);
    }

    #[test]
    fn test_factory_functions() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        // Test knot generation
        let knots = generate_knots(&x.view(), 3, "clamped");
        assert!(knots.is_ok());

        // Test least-squares fitting
        let knots = knots.unwrap();
        let lsq_spline = make_lsq_bspline(
            &x.view(),
            &y.view(),
            &knots.view(),
            3,
            None,
            ExtrapolateMode::Extrapolate,
        );
        assert!(lsq_spline.is_ok());

        // Test automatic spline creation
        let auto_spline =
            make_auto_bspline(&x.view(), &y.view(), 3, 0.1, ExtrapolateMode::Extrapolate);
        assert!(auto_spline.is_ok());
    }

    #[test]
    fn test_extrapolation_modes() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 4.0, 9.0, 16.0];

        // Test extrapolate mode
        let spline_extrapolate =
            make_interp_bspline(&x.view(), &y.view(), 2, ExtrapolateMode::Extrapolate).unwrap();

        let val_out = spline_extrapolate.evaluate(5.0); // Outside domain
        assert!(val_out.is_ok());

        // Test NaN mode
        let spline_nan =
            make_interp_bspline(&x.view(), &y.view(), 2, ExtrapolateMode::Nan).unwrap();

        let val_nan = spline_nan.evaluate(5.0);
        assert!(val_nan.is_ok());
        assert!(val_nan.unwrap().is_nan());

        // Test error mode
        let spline_error =
            make_interp_bspline(&x.view(), &y.view(), 2, ExtrapolateMode::Error).unwrap();

        let val_error = spline_error.evaluate(5.0);
        assert!(val_error.is_err());
    }

    #[test]
    fn test_trait_implementations() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline =
            make_interp_bspline(&x.view(), &y.view(), 2, ExtrapolateMode::Extrapolate).unwrap();

        // Test Interpolator trait
        let query_points = array![1.5, 2.5];
        let values = spline.evaluate_array(&query_points.view()).unwrap();
        assert_eq!(values.len(), 2);

        assert_eq!(spline.dimension(), 1);
        assert_eq!(spline.len(), 4);
    }
}
