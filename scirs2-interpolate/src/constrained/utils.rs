//! Utility functions for constrained splines
//!
//! This module contains utility functions used internally by the constrained
//! spline implementation, including methods for checking constraints and
//! accessing properties of constrained splines.

use crate::bspline::BSpline;
use crate::error::InterpolateResult;
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use super::builder::FittingMethod;
use super::types::{ConstrainedSpline, Constraint, ConstraintType};

impl<T> ConstrainedSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Evaluate the spline at a given x value
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The value of the spline at x
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use ndarray::array;
    /// use scirs2_interpolate::constrained::{ConstrainedSpline, Constraint};
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = array![0.0, 0.8, 1.5, 1.9, 2.3, 2.5];
    ///
    /// let constraint = Constraint::<f64>::monotone_increasing(None, None);
    /// let spline = ConstrainedSpline::<f64>::interpolate(
    ///     &x.view(),
    ///     &y.view(),
    ///     vec![constraint],
    ///     3,
    ///     ExtrapolateMode::Extrapolate,
    /// ).unwrap();
    ///
    /// let value = spline.evaluate(2.5).unwrap();
    /// # }
    /// ```
    pub fn evaluate(&self, x: T) -> InterpolateResult<T> {
        self.bspline.evaluate(x)
    }

    /// Evaluate the spline at multiple x values
    ///
    /// # Arguments
    ///
    /// * `x` - Array of x coordinates at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// Array of values of the spline at the given x coordinates
    pub fn evaluate_array(
        &self,
        x: &ndarray::ArrayView1<T>,
    ) -> InterpolateResult<ndarray::Array1<T>> {
        self.bspline.evaluate_array(x)
    }

    /// Evaluate the derivative of the spline at a given x value
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate at which to evaluate the derivative
    /// * `order` - The order of the derivative (1 = first derivative, 2 = second derivative, etc.)
    ///
    /// # Returns
    ///
    /// The value of the specified derivative at x
    pub fn derivative(&self, x: T, order: usize) -> InterpolateResult<T> {
        self.bspline.derivative(x, order)
    }

    /// Check if the spline satisfies a specific constraint at a point
    ///
    /// # Arguments
    ///
    /// * `constraint_type` - The type of constraint to check
    /// * `x` - The x coordinate at which to check the constraint
    /// * `parameter` - Additional parameter for certain constraint types
    ///
    /// # Returns
    ///
    /// true if the constraint is satisfied, false otherwise
    pub fn check_constraint(
        &self,
        constraint_type: ConstraintType,
        x: T,
        parameter: Option<T>,
    ) -> InterpolateResult<bool> {
        match constraint_type {
            ConstraintType::MonotoneIncreasing => {
                let derivative = self.derivative(x, 1)?;
                Ok(derivative >= -T::epsilon())
            }
            ConstraintType::MonotoneDecreasing => {
                let derivative = self.derivative(x, 1)?;
                Ok(derivative <= T::epsilon())
            }
            ConstraintType::Convex => {
                let second_derivative = self.derivative(x, 2)?;
                Ok(second_derivative >= -T::epsilon())
            }
            ConstraintType::Concave => {
                let second_derivative = self.derivative(x, 2)?;
                Ok(second_derivative <= T::epsilon())
            }
            ConstraintType::Positive => {
                let value = self.evaluate(x)?;
                Ok(value >= -T::epsilon())
            }
            ConstraintType::UpperBound => {
                let value = self.evaluate(x)?;
                let bound = parameter.unwrap_or(T::one());
                Ok(value <= bound + T::epsilon())
            }
            ConstraintType::LowerBound => {
                let value = self.evaluate(x)?;
                let bound = parameter.unwrap_or(T::zero());
                Ok(value >= bound - T::epsilon())
            }
        }
    }

    /// Get the list of constraints applied to the spline
    pub fn constraints(&self) -> &[Constraint<T>] {
        &self.constraints
    }

    /// Get the underlying B-spline
    pub fn bspline(&self) -> &BSpline<T> {
        &self.bspline
    }

    /// Get the fitting method used
    pub fn method(&self) -> FittingMethod {
        self.method
    }
}
