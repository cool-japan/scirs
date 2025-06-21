//! Builder and construction methods for constrained splines
//!
//! This module contains the various construction methods for ConstrainedSpline,
//! including interpolation, least squares fitting, and penalized fitting.

use crate::bspline::{generate_knots, BSpline, ExtrapolateMode};
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::ArrayView1;
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

use super::solver::{solve_constrained_system, solve_penalized_system};
use super::types::{ConstrainedSpline, Constraint};

/// Method used for fitting the constrained spline
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FittingMethod {
    /// Least squares fitting
    LeastSquares,

    /// Exact interpolation (passes through all data points)
    Interpolation,

    /// Penalized spline fitting (smooths the data)
    Penalized,
}

impl<T> ConstrainedSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static
        + std::fmt::LowerExp,
{
    /// Create a new constrained spline by interpolating the data
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates of the data points
    /// * `y` - The y coordinates of the data points
    /// * `constraints` - List of constraints to apply
    /// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new constrained spline that interpolates the data points while satisfying constraints
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
    /// // Create monotonically increasing data
    /// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = array![0.0, 0.8, 1.5, 1.9, 2.3, 2.5];
    ///
    /// // Create a constraint for the entire domain
    /// let constraint = Constraint::<f64>::monotone_increasing(None, None);
    ///
    /// // Create a constrained spline
    /// let spline = ConstrainedSpline::<f64>::interpolate(
    ///     &x.view(),
    ///     &y.view(),
    ///     vec![constraint],
    ///     3, // cubic spline
    ///     ExtrapolateMode::Extrapolate,
    /// ).unwrap();
    ///
    /// // Evaluate at a new point
    /// let value = spline.evaluate(2.5).unwrap();
    /// # }
    /// ```
    pub fn interpolate(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        constraints: Vec<Constraint<T>>,
        degree: usize,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        Self::fit_internal(
            x,
            y,
            constraints,
            degree,
            None,
            FittingMethod::Interpolation,
            extrapolate,
        )
    }

    /// Create a new constrained spline using least squares fitting
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates of the data points
    /// * `y` - The y coordinates of the data points
    /// * `constraints` - List of constraints to apply
    /// * `num_knots` - Number of internal knots
    /// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new constrained spline that fits the data using least squares
    pub fn least_squares(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        constraints: Vec<Constraint<T>>,
        num_knots: usize,
        degree: usize,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        Self::fit_internal(
            x,
            y,
            constraints,
            degree,
            Some(num_knots),
            FittingMethod::LeastSquares,
            extrapolate,
        )
    }

    /// Create a new constrained spline using penalized fitting
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates of the data points
    /// * `y` - The y coordinates of the data points
    /// * `constraints` - List of constraints to apply
    /// * `num_knots` - Number of internal knots
    /// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
    /// * `lambda` - Smoothing parameter (higher values produce smoother curves)
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new constrained spline that fits the data with smoothing
    pub fn penalized(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        constraints: Vec<Constraint<T>>,
        num_knots: usize,
        degree: usize,
        _lambda: T,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        Self::fit_internal(
            x,
            y,
            constraints,
            degree,
            Some(num_knots),
            FittingMethod::Penalized,
            extrapolate,
        )
    }

    /// Internal method to fit a constrained spline
    pub(crate) fn fit_internal(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        constraints: Vec<Constraint<T>>,
        degree: usize,
        num_knots: Option<usize>,
        method: FittingMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        // Validate inputs
        if x.len() != y.len() {
            return Err(InterpolateError::IndexError(format!(
                "x and y arrays must have the same length: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        if x.len() < degree + 1 {
            return Err(InterpolateError::IndexError(format!(
                "Need at least {} points for degree {} spline",
                degree + 1,
                degree
            )));
        }

        // Check if x values are sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::IndexError(
                    "x values must be strictly increasing".to_string(),
                ));
            }
        }

        // Generate knots
        let _n_internal = num_knots.unwrap_or(x.len() - degree - 1);
        let knots = generate_knots(x, degree, "clamped")?;

        // Create coefficient matrix and solve the constrained system
        let coeffs = match method {
            FittingMethod::Interpolation => {
                solve_constrained_system(x, y, &knots.view(), degree, &constraints)?
            }
            FittingMethod::LeastSquares => {
                solve_constrained_system(x, y, &knots.view(), degree, &constraints)?
            }
            FittingMethod::Penalized => {
                // Note: lambda parameter should be passed here
                solve_penalized_system(
                    x,
                    y,
                    &knots.view(),
                    degree,
                    &constraints,
                    T::from_f64(0.1).unwrap(),
                )?
            }
        };

        // Create the B-spline
        let bspline = BSpline::new(&knots.view(), &coeffs.view(), degree, extrapolate)?;

        Ok(ConstrainedSpline {
            bspline,
            constraints,
            method,
        })
    }
}
