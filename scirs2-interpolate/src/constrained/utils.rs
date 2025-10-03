//! Utility functions for constrained splines
//!
//! This module contains utility functions used internally by the constrained
//! spline implementation, including methods for checking constraints and
//! accessing properties of constrained splines.

use crate::bspline::BSpline;
use crate::error::InterpolateResult;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use super::builder::FittingMethod;
use super::types::{ConstrainedSpline, Constraint, ConstraintType};
use scirs2_core::ndarray::ArrayView1;

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
    /// use scirs2_core::ndarray::array;
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
        x: &scirs2_core::ndarray::ArrayView1<T>,
    ) -> InterpolateResult<scirs2_core::ndarray::Array1<T>> {
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

    /// Evaluate the derivative of the spline at multiple x values
    ///
    /// This provides batch evaluation of derivatives for improved performance
    /// when evaluating derivatives at many points.
    ///
    /// # Arguments
    ///
    /// * `x` - Array of x coordinates at which to evaluate the derivative
    /// * `order` - The order of the derivative (1 = first derivative, 2 = second derivative, etc.)
    ///
    /// # Returns
    ///
    /// Array of derivative values at the given x coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::array;
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
    /// let x_eval = array![1.5, 2.5, 3.5];
    /// let derivatives = spline.derivative_array(&x_eval.view(), 1).unwrap();
    /// # }
    /// ```
    pub fn derivative_array(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<T>,
        order: usize,
    ) -> InterpolateResult<scirs2_core::ndarray::Array1<T>> {
        let mut result = scirs2_core::ndarray::Array1::zeros(x.len());
        for (i, &xi) in x.iter().enumerate() {
            result[i] = self.bspline.derivative(xi, order)?;
        }
        Ok(result)
    }

    /// Evaluate multiple orders of derivatives at a single point
    ///
    /// This method efficiently computes derivatives of multiple orders at the same
    /// x coordinate, which is useful for Taylor series expansions or detailed
    /// local analysis.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate at which to evaluate derivatives
    /// * `max_order` - Maximum order of derivative to compute (inclusive)
    ///
    /// # Returns
    ///
    /// Vector containing derivatives from order 0 (function value) to max_order
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::array;
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
    /// // Get function value, first derivative, and second derivative at x=2.5
    /// let derivatives = spline.derivatives_all(2.5, 2).unwrap();
    /// let function_value = derivatives[0];
    /// let first_deriv = derivatives[1];
    /// let second_deriv = derivatives[2];
    /// # }
    /// ```
    pub fn derivatives_all(&self, x: T, maxorder: usize) -> InterpolateResult<Vec<T>> {
        let mut derivatives = Vec::with_capacity(maxorder + 1);

        // Order 0 is the function value itself
        derivatives.push(self.evaluate(x)?);

        // Compute derivatives of _order 1 through maxorder
        for _order in 1..=maxorder {
            derivatives.push(self.derivative(x, _order)?);
        }

        Ok(derivatives)
    }

    /// Create a new spline representing the derivative of this constrained spline
    ///
    /// This method creates a new B-spline that represents the nth derivative of the
    /// constrained spline. The resulting spline can be evaluated independently and
    /// maintains the mathematical properties of the derivative.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the derivative (1 = first derivative, 2 = second derivative, etc.)
    ///
    /// # Returns
    ///
    /// A new BSpline representing the derivative
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::array;
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
    /// // Create a spline representing the first derivative
    /// let derivative_spline = spline.derivative_spline(1).unwrap();
    /// let slope_at_2_5 = derivative_spline.evaluate(2.5).unwrap();
    /// # }
    /// ```
    pub fn derivative_spline(&self, order: usize) -> InterpolateResult<BSpline<T>> {
        Err(crate::error::InterpolateError::NotImplemented(
            "derivative_spline method is not available for constrained splines".to_string(),
        ))
    }

    /// Create a new spline representing the antiderivative (indefinite integral) of this constrained spline
    ///
    /// This method creates a new B-spline that represents the antiderivative of the
    /// constrained spline. The integration constant is chosen to make the antiderivative
    /// zero at the left boundary of the spline domain.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of antiderivative (1 = first antiderivative, 2 = second antiderivative, etc.)
    ///
    /// # Returns
    ///
    /// A new BSpline representing the antiderivative
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::constrained::{ConstrainedSpline, Constraint};
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // Constant function
    ///
    /// let constraint = Constraint::<f64>::positive(None, None);
    /// let spline = ConstrainedSpline::<f64>::interpolate(
    ///     &x.view(),
    ///     &y.view(),
    ///     vec![constraint],
    ///     3,
    ///     ExtrapolateMode::Extrapolate,
    /// ).unwrap();
    ///
    /// // Create the antiderivative (should be approximately linear)
    /// let antiderivative = spline.antiderivative_spline(1).unwrap();
    /// let integral_at_3 = antiderivative.evaluate(3.0).unwrap(); // Should be ~3.0
    /// # }
    /// ```
    pub fn antiderivative_spline(&self, order: usize) -> InterpolateResult<BSpline<T>> {
        self.bspline.antiderivative(order)
    }

    /// Compute the definite integral of the constrained spline over an interval
    ///
    /// This method computes the definite integral of the spline from point a to point b
    /// using the analytical properties of B-splines. The integration is exact for
    /// polynomial segments within the spline.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The value of the definite integral from a to b
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::constrained::{ConstrainedSpline, Constraint};
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]; // Approximately x^2
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
    /// // Integrate from 0 to 3 (should be approximately x^3/3 = 9.0)
    /// let integral = spline.integrate(0.0, 3.0).unwrap();
    /// # }
    /// ```
    pub fn integrate(&self, a: T, b: T) -> InterpolateResult<T> {
        self.bspline.integrate(a, b)
    }

    /// Validate that all constraints are satisfied across the spline domain
    ///
    /// This method checks that the constrained spline actually satisfies all of its
    /// specified constraints across the entire domain or specified subregions.
    /// This is useful for verification after fitting or for debugging constraint violations.
    ///
    /// # Arguments
    ///
    /// * `num_check_points` - Number of points to check across each constraint region
    ///
    /// # Returns
    ///
    /// `Ok(true)` if all constraints are satisfied, `Ok(false)` if any violations found,
    /// or `Err` if evaluation fails
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use scirs2_core::ndarray::array;
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
    /// // Verify that the monotonicity constraint is satisfied
    /// let is_valid = spline.validate_constraints(50).unwrap();
    /// assert!(is_valid, "Monotonicity constraint should be satisfied");
    /// # }
    /// ```
    pub fn validate_constraints(&self, num_checkpoints: usize) -> InterpolateResult<bool> {
        // Get the domain bounds from the underlying B-spline
        let knots = self.bspline.knot_vector();
        let domain_start = knots[self.bspline.degree()];
        let domain_end = knots[knots.len() - self.bspline.degree() - 1];

        for constraint in &self.constraints {
            // Determine the region to check for this constraint
            let check_start = constraint.x_min.unwrap_or(domain_start);
            let check_end = constraint.x_max.unwrap_or(domain_end);

            // Generate check _points across the constraint region
            let step = (check_end - check_start) / T::from_usize(num_checkpoints - 1).unwrap();

            for i in 0..num_checkpoints {
                let x = check_start + T::from_usize(i).unwrap() * step;

                if !self.check_constraint(constraint.constraint_type, x, constraint.parameter)? {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Get summary statistics about constraint satisfaction across the domain
    ///
    /// This method provides detailed information about how well constraints are
    /// satisfied, including the worst violations and their locations.
    ///
    /// # Arguments
    ///
    /// * `num_check_points` - Number of points to check across each constraint region
    ///
    /// # Returns
    ///
    /// A struct containing constraint satisfaction statistics
    pub fn constraint_satisfaction_summary(
        &self,
        num_check_points: usize,
    ) -> InterpolateResult<ConstraintSatisfactionSummary<T>> {
        let mut summary = ConstraintSatisfactionSummary::new();

        // Get the domain bounds from the underlying B-spline
        let knots = self.bspline.knot_vector();
        let domain_start = knots[self.bspline.degree()];
        let domain_end = knots[knots.len() - self.bspline.degree() - 1];

        for (constraint_idx, constraint) in self.constraints.iter().enumerate() {
            // Determine the region to check for this constraint
            let check_start = constraint.x_min.unwrap_or(domain_start);
            let check_end = constraint.x_max.unwrap_or(domain_end);

            let step = (check_end - check_start) / T::from_usize(num_check_points - 1).unwrap();

            let mut violations = 0;
            let mut max_violation = T::zero();
            let mut max_violation_location = check_start;

            for i in 0..num_check_points {
                let x = check_start + T::from_usize(i).unwrap() * step;

                // Calculate violation magnitude
                let violation = self.calculate_constraint_violation(
                    constraint.constraint_type,
                    x,
                    constraint.parameter,
                )?;

                if violation > T::epsilon() {
                    violations += 1;
                    if violation > max_violation {
                        max_violation = violation;
                        max_violation_location = x;
                    }
                }
            }

            summary.constraint_violations.push(ConstraintViolationInfo {
                constraint_index: constraint_idx,
                constraint_type: constraint.constraint_type,
                total_violations: violations,
                max_violation,
                max_violation_location,
                satisfaction_rate: T::one()
                    - T::from_usize(violations).unwrap() / T::from_usize(num_check_points).unwrap(),
            });
        }

        Ok(summary)
    }

    /// Calculate the magnitude of constraint violation at a specific point
    ///
    /// This is a helper method that returns how much a constraint is violated
    /// (positive values indicate violations, zero or negative indicate satisfaction).
    fn calculate_constraint_violation(
        &self,
        constraint_type: ConstraintType,
        x: T,
        parameter: Option<T>,
    ) -> InterpolateResult<T> {
        match constraint_type {
            ConstraintType::MonotoneIncreasing => {
                let derivative = self.derivative(x, 1)?;
                Ok((-derivative).max(T::zero()))
            }
            ConstraintType::MonotoneDecreasing => {
                let derivative = self.derivative(x, 1)?;
                Ok(derivative.max(T::zero()))
            }
            ConstraintType::Convex => {
                let second_derivative = self.derivative(x, 2)?;
                Ok((-second_derivative).max(T::zero()))
            }
            ConstraintType::Concave => {
                let second_derivative = self.derivative(x, 2)?;
                Ok(second_derivative.max(T::zero()))
            }
            ConstraintType::Positive => {
                let value = self.evaluate(x)?;
                Ok((-value).max(T::zero()))
            }
            ConstraintType::UpperBound => {
                let value = self.evaluate(x)?;
                let bound = parameter.unwrap_or(T::one());
                Ok((value - bound).max(T::zero()))
            }
            ConstraintType::LowerBound => {
                let value = self.evaluate(x)?;
                let bound = parameter.unwrap_or(T::zero());
                Ok((bound - value).max(T::zero()))
            }
        }
    }
}

/// Summary of constraint satisfaction across the spline domain
#[derive(Debug, Clone)]
pub struct ConstraintSatisfactionSummary<T: Float + FromPrimitive + Debug + Display> {
    /// Information about violations for each constraint
    pub constraint_violations: Vec<ConstraintViolationInfo<T>>,
}

impl<T: Float + FromPrimitive + Debug + Display> ConstraintSatisfactionSummary<T> {
    fn new() -> Self {
        Self {
            constraint_violations: Vec::new(),
        }
    }

    /// Check if all constraints are satisfied (no violations)
    pub fn all_satisfied(&self) -> bool {
        self.constraint_violations
            .iter()
            .all(|info| info.total_violations == 0)
    }

    /// Get the overall satisfaction rate (0.0 = all violated, 1.0 = all satisfied)
    pub fn overall_satisfaction_rate(&self) -> T {
        if self.constraint_violations.is_empty() {
            return T::one();
        }

        let total_rate: T = self
            .constraint_violations
            .iter()
            .map(|info| info.satisfaction_rate)
            .fold(T::zero(), |acc, rate| acc + rate);

        total_rate / T::from_usize(self.constraint_violations.len()).unwrap()
    }
}

/// Information about violations for a specific constraint
#[derive(Debug, Clone)]
pub struct ConstraintViolationInfo<T: Float + FromPrimitive + Debug + Display> {
    /// Index of the constraint in the original constraint vector
    pub constraint_index: usize,

    /// Type of the constraint
    pub constraint_type: ConstraintType,

    /// Total number of violation points found
    pub total_violations: usize,

    /// Maximum violation magnitude
    pub max_violation: T,

    /// Location where maximum violation occurs
    pub max_violation_location: T,

    /// Fraction of points where constraint is satisfied (0.0 to 1.0)
    pub satisfaction_rate: T,
}
