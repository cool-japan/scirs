//! Core cubic spline data structures and implementations
//!
//! This module contains the main `CubicSpline` struct and `CubicSplineBuilder` struct
//! along with their core implementations. These provide the fundamental data structures
//! for cubic spline interpolation.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use ndarray::{Array1, Array2, ArrayView1};
use super::types::SplineBoundaryCondition;
use super::algorithms::*;

/// A cubic spline interpolator
///
/// This struct represents a constructed cubic spline that can be used for interpolation
/// and derivative computation. Each spline consists of piecewise cubic polynomials that
/// maintain C² continuity (continuous function, first, and second derivatives).
///
/// # Type Parameters
///
/// * `F` - The floating point type (f32 or f64)
///
/// # Structure
///
/// The spline stores:
/// - `x`: The sorted x coordinates of the data points
/// - `y`: The y coordinates of the data points
/// - `coeffs`: Polynomial coefficients for each segment
///
/// Each row of `coeffs` contains [a, b, c, d] representing the cubic polynomial:
/// ```text
/// y(x) = a + b*(x-x_i) + c*(x-x_i)² + d*(x-x_i)³
/// ```
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use scirs2_interpolate::spline::CubicSpline;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
/// let value = spline.evaluate(1.5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CubicSpline<F: InterpolationFloat> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Coefficients for cubic polynomials (n-1 segments, 4 coefficients each)
    /// Each row represents [a, b, c, d] for a segment
    /// y(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    coeffs: Array2<F>,
}

/// Builder for cubic splines with custom boundary conditions
///
/// This builder allows for flexible construction of cubic splines with different
/// boundary conditions. It uses the builder pattern to provide a fluent API
/// for spline configuration.
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use scirs2_interpolate::spline::{CubicSpline, SplineBoundaryCondition};
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// let spline = CubicSpline::builder()
///     .x(x)
///     .y(y)
///     .boundary_condition(SplineBoundaryCondition::NotAKnot)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CubicSplineBuilder<F: InterpolationFloat> {
    x: Option<Array1<F>>,
    y: Option<Array1<F>>,
    boundary_condition: SplineBoundaryCondition<F>,
}

impl<F: InterpolationFloat> CubicSplineBuilder<F> {
    /// Create a new builder with default settings
    ///
    /// Default boundary condition is Natural (zero second derivative at endpoints).
    pub fn new() -> Self {
        Self {
            x: None,
            y: None,
            boundary_condition: SplineBoundaryCondition::Natural,
        }
    }

    /// Set the x coordinates
    ///
    /// # Arguments
    ///
    /// * `x` - Array of x coordinates (must be sorted in ascending order)
    ///
    /// # Returns
    ///
    /// The builder with x coordinates set
    pub fn x(mut self, x: Array1<F>) -> Self {
        self.x = Some(x);
        self
    }

    /// Set the y coordinates
    ///
    /// # Arguments
    ///
    /// * `y` - Array of y coordinates (must have same length as x)
    ///
    /// # Returns
    ///
    /// The builder with y coordinates set
    pub fn y(mut self, y: Array1<F>) -> Self {
        self.y = Some(y);
        self
    }

    /// Set the boundary condition
    ///
    /// # Arguments
    ///
    /// * `bc` - The boundary condition to use
    ///
    /// # Returns
    ///
    /// The builder with boundary condition set
    pub fn boundary_condition(mut self, bc: SplineBoundaryCondition<F>) -> Self {
        self.boundary_condition = bc;
        self
    }

    /// Build the spline
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object or an error if construction fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - x or y coordinates are not set
    /// - Arrays have different lengths
    /// - Insufficient points for the boundary condition
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    pub fn build(self) -> InterpolateResult<CubicSpline<F>> {
        let x = self
            .x
            .ok_or_else(|| InterpolateError::invalid_input("x coordinates not set".to_string()))?;
        let y = self
            .y
            .ok_or_else(|| InterpolateError::invalid_input("y coordinates not set".to_string()))?;

        CubicSpline::with_boundary_condition(&x.view(), &y.view(), self.boundary_condition)
    }
}

impl<F: InterpolationFloat> Default for CubicSplineBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: InterpolationFloat + ToString> CubicSpline<F> {
    /// Create a new builder for cubic splines
    ///
    /// # Returns
    ///
    /// A new `CubicSplineBuilder` instance
    pub fn builder() -> CubicSplineBuilder<F> {
        CubicSplineBuilder::new()
    }

    /// Create a new cubic spline with natural boundary conditions
    ///
    /// Natural boundary conditions set the second derivative to zero at both endpoints,
    /// which minimizes the total curvature of the spline.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different lengths
    /// - Less than 3 points provided
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    ///
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// // Interpolate at x = 1.5
    /// let y_interp = spline.evaluate(1.5).unwrap();
    /// println!("Interpolated value at x=1.5: {}", y_interp);
    /// ```
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::insufficient_points(
                3,
                x.len(),
                "cubic spline",
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients for natural cubic spline
        let coeffs = compute_natural_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Get the x coordinates
    ///
    /// # Returns
    ///
    /// Reference to the array of x coordinates
    pub fn x(&self) -> &Array1<F> {
        &self.x
    }

    /// Get the y coordinates
    ///
    /// # Returns
    ///
    /// Reference to the array of y coordinates
    pub fn y(&self) -> &Array1<F> {
        &self.y
    }

    /// Get the polynomial coefficients
    ///
    /// # Returns
    ///
    /// Reference to the 2D array of polynomial coefficients
    /// Shape: (n-1, 4) where n is the number of data points
    /// Each row contains [a, b, c, d] for the polynomial in that segment
    pub fn coeffs(&self) -> &Array2<F> {
        &self.coeffs
    }

    /// Create a new cubic spline with not-a-knot boundary conditions
    ///
    /// Not-a-knot boundary conditions force the third derivative to be continuous
    /// at the second and second-to-last data points, providing maximum smoothness.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different lengths
    /// - Less than 4 points provided (required for not-a-knot)
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    pub fn new_not_a_knot(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 4 {
            return Err(InterpolateError::insufficient_points(
                4,
                x.len(),
                "not-a-knot cubic spline",
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients for not-a-knot cubic spline
        let coeffs = compute_not_a_knot_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a cubic spline with custom boundary conditions
    ///
    /// This is the most general constructor that supports all boundary condition types.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `bc` - The boundary condition to use
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different lengths
    /// - Insufficient points for the chosen boundary condition
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    /// - Invalid boundary condition parameters
    pub fn with_boundary_condition(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        bc: SplineBoundaryCondition<F>,
    ) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        // Check minimum points based on boundary condition
        let min_points = match bc {
            SplineBoundaryCondition::NotAKnot => 4,
            _ => 3,
        };

        if x.len() < min_points {
            return Err(InterpolateError::insufficient_points(
                min_points,
                x.len(),
                &format!("cubic spline with {:?} boundary condition", bc),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Validate periodic boundary condition
        if let SplineBoundaryCondition::Periodic = bc {
            let tolerance = F::from_f64(1e-10).unwrap_or_else(|| F::epsilon());
            if (y[0] - y[y.len() - 1]).abs() > tolerance {
                return Err(InterpolateError::invalid_input(
                    "For periodic boundary conditions, first and last y values must be equal".to_string(),
                ));
            }
        }

        // Get coefficients based on boundary condition
        let coeffs = match bc {
            SplineBoundaryCondition::Natural => compute_natural_cubic_spline(x, y)?,
            SplineBoundaryCondition::NotAKnot => compute_not_a_knot_cubic_spline(x, y)?,
            SplineBoundaryCondition::Clamped(dy0, dyn_) => {
                compute_clamped_cubic_spline(x, y, dy0, dyn_)?
            }
            SplineBoundaryCondition::Periodic => compute_periodic_cubic_spline(x, y)?,
            SplineBoundaryCondition::SecondDerivative(d2y0, d2yn) => {
                compute_second_derivative_cubic_spline(x, y, d2y0, d2yn)?
            }
            SplineBoundaryCondition::ParabolicRunout => {
                compute_parabolic_runout_cubic_spline(x, y)?
            }
        };

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }
}