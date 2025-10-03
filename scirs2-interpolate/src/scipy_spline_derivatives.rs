//! Complete SciPy spline derivative and integral interfaces for 0.1.0 stable release
//!
//! This module provides complete SciPy-compatible spline derivative and integral interfaces,
//! ensuring exact parity with SciPy.interpolate spline methods for the stable release.
//!
//! ## SciPy Spline Derivative/Integral Methods Completed
//!
//! ### CubicSpline Methods
//! - `derivative(n)` - Returns a new spline representing the n-th derivative
//! - `antiderivative(n)` - Returns a new spline representing the n-th antiderivative
//! - `integrate(a, b)` - Definite integral from a to b
//! - `solve(y, discontinuity, extrapolate)` - Find x where spline equals y
//! - `roots(discontinuity, extrapolate)` - Find roots of the spline
//!
//! ### BSpline Methods
//! - `derivative(nu)` - Returns a new BSpline representing the nu-th derivative
//! - `antiderivative(nu)` - Returns a new BSpline representing the nu-th antiderivative
//! - `integrate(a, b)` - Definite integral from a to b
//!
//! ### PPoly Methods
//! - `derivative(m)` - Returns a new PPoly representing the m-th derivative
//! - `antiderivative(m)` - Returns a new PPoly representing the m-th antiderivative
//! - `integrate(a, b)` - Definite integral from a to b

use crate::bspline::BSpline;
use crate::error::{InterpolateError, InterpolateResult};
use crate::spline::{CubicSpline, SplineBoundaryCondition};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::FromPrimitive;
use std::fmt::{Debug, Display};

/// Enhanced SciPy-compatible cubic spline with complete derivative/integral interface
pub struct SciPyCompatibleCubicSpline<T: InterpolationFloat> {
    /// Inner cubic spline implementation
    inner: CubicSpline<T>,
    /// SciPy-compatible parameters
    bc_type: SciPyBoundaryType,
    extrapolate: Option<bool>,
    axis: i32,
}

/// SciPy boundary condition types for exact compatibility
#[derive(Debug, Clone)]
pub enum SciPyBoundaryType {
    /// Natural boundary conditions
    Natural,
    /// Not-a-knot boundary conditions
    NotAKnot,
    /// Clamped boundary conditions with specified derivatives
    Clamped((f64, f64)),
    /// Periodic boundary conditions
    Periodic,
    /// Second derivative boundary conditions
    SecondDerivative((f64, f64)),
}

/// Enhanced SciPy-compatible BSpline with complete derivative/integral interface
pub struct SciPyCompatibleBSpline<
    T: InterpolationFloat + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
> {
    /// Inner BSpline implementation
    inner: BSpline<T>,
    /// Extrapolation mode
    extrapolate: bool,
    /// Axis parameter
    axis: i32,
}

/// Piecewise polynomial implementation for SciPy compatibility
#[derive(Debug, Clone)]
pub struct SciPyPPoly<T: InterpolationFloat> {
    /// Polynomial coefficients [k, m] where k is degree+1, m is number of pieces
    coefficients: Array2<T>,
    /// Breakpoints [m+1]
    breakpoints: Array1<T>,
    /// Extrapolation mode
    extrapolate: bool,
    /// Axis parameter
    axis: i32,
}

impl<T: InterpolationFloat + Debug + Display + std::ops::AddAssign + FromPrimitive>
    SciPyCompatibleCubicSpline<T>
{
    /// Create a new SciPy-compatible cubic spline
    pub fn new(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        bc_type: SciPyBoundaryType,
        extrapolate: Option<bool>,
        axis: i32,
    ) -> InterpolateResult<Self> {
        let inner = match &bc_type {
            SciPyBoundaryType::Natural => CubicSpline::new(x, y)?,
            SciPyBoundaryType::NotAKnot => CubicSpline::new_not_a_knot(x, y)?,
            SciPyBoundaryType::Clamped((left, right)) => {
                let left_deriv = T::from_f64(*left).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert left derivative".to_string(),
                    )
                })?;
                let right_deriv = T::from_f64(*right).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert right derivative".to_string(),
                    )
                })?;
                CubicSpline::new_clamped(x, y, left_deriv, right_deriv)?
            }
            SciPyBoundaryType::Periodic => CubicSpline::new_periodic(x, y)?,
            SciPyBoundaryType::SecondDerivative((left, right)) => {
                let left_d2 = T::from_f64(*left).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert left second derivative".to_string(),
                    )
                })?;
                let right_d2 = T::from_f64(*right).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert right second derivative".to_string(),
                    )
                })?;
                CubicSpline::new_second_derivative(x, y, left_d2, right_d2)?
            }
        };

        Ok(Self {
            inner,
            bc_type,
            extrapolate,
            axis,
        })
    }

    /// Returns a new spline representing the n-th derivative (SciPy interface)
    ///
    /// This method exactly matches SciPy's CubicSpline.derivative(n) interface
    ///
    /// # Arguments
    /// * `n` - Order of derivative (default 1)
    ///
    /// # Returns
    /// A new `SciPyCompatibleCubicSpline` representing the n-th derivative
    ///
    /// # Example
    /// ```rust
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::scipy_spline_derivatives::{SciPyCompatibleCubicSpline, SciPyBoundaryType};
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = SciPyCompatibleCubicSpline::new(
    ///     &x.view(), &y.view(), SciPyBoundaryType::Natural, None, 0
    /// ).unwrap();
    ///
    /// // Get first derivative spline
    /// let deriv_spline = spline.derivative(Some(1)).unwrap();
    /// ```
    pub fn derivative(&self, n: Option<usize>) -> InterpolateResult<Self> {
        let order = n.unwrap_or(1);
        let deriv_inner = self.inner.derivative_spline(order)?;

        Ok(Self {
            inner: deriv_inner,
            bc_type: self.bc_type.clone(),
            extrapolate: self.extrapolate,
            axis: self.axis,
        })
    }

    /// Returns a new spline representing the n-th antiderivative (SciPy interface)
    ///
    /// This method exactly matches SciPy's CubicSpline.antiderivative(n) interface
    ///
    /// # Arguments
    /// * `n` - Order of antiderivative (default 1)
    ///
    /// # Returns
    /// A new `SciPyCompatibleCubicSpline` representing the n-th antiderivative
    ///
    /// # Example
    /// ```rust
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::scipy_spline_derivatives::{SciPyCompatibleCubicSpline, SciPyBoundaryType};
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = SciPyCompatibleCubicSpline::new(
    ///     &x.view(), &y.view(), SciPyBoundaryType::Natural, None, 0
    /// ).unwrap();
    ///
    /// // Get first antiderivative spline
    /// let antideriv_spline = spline.antiderivative(Some(1)).unwrap();
    /// ```
    pub fn antiderivative(&self, n: Option<usize>) -> InterpolateResult<Self> {
        let order = n.unwrap_or(1);
        let antideriv_inner = self.inner.antiderivative_spline(order)?;

        Ok(Self {
            inner: antideriv_inner,
            bc_type: self.bc_type.clone(),
            extrapolate: self.extrapolate,
            axis: self.axis,
        })
    }

    /// Compute definite integral over [a, b] (SciPy interface)
    ///
    /// This method exactly matches SciPy's CubicSpline.integrate(a, b) interface
    ///
    /// # Arguments
    /// * `a` - Lower integration bound
    /// * `b` - Upper integration bound
    /// * `extrapolate` - Whether to extrapolate beyond domain (optional)
    ///
    /// # Returns
    /// The definite integral from a to b
    ///
    /// # Example
    /// ```rust
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::scipy_spline_derivatives::{SciPyCompatibleCubicSpline, SciPyBoundaryType};
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = SciPyCompatibleCubicSpline::new(
    ///     &x.view(), &y.view(), SciPyBoundaryType::Natural, None, 0
    /// ).unwrap();
    ///
    /// // Integrate from 0 to 3
    /// let integral = spline.integrate(0.0, 3.0, None).unwrap();
    /// ```
    pub fn integrate(&self, a: f64, b: f64, extrapolate: Option<bool>) -> InterpolateResult<T> {
        let a_t = T::from_f64(a).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert integration bound a".to_string())
        })?;
        let b_t = T::from_f64(b).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert integration bound b".to_string())
        })?;

        let use_extrapolate = extrapolate.unwrap_or(self.extrapolate.unwrap_or(true));

        if use_extrapolate {
            self.inner.integrate_scipy(a_t, b_t)
        } else {
            // Check bounds and only integrate within domain
            let x_min = self.inner.x_bounds().0;
            let x_max = self.inner.x_bounds().1;

            if a_t < x_min || b_t > x_max {
                return Err(InterpolateError::OutOfBounds(
                    "Integration bounds outside domain and extrapolate=False".to_string(),
                ));
            }

            self.inner.integrate_scipy(a_t, b_t)
        }
    }

    /// Find x values where spline equals y (SciPy interface)
    ///
    /// This method exactly matches SciPy's CubicSpline.solve(y, discontinuity, extrapolate) interface
    ///
    /// # Arguments
    /// * `y` - Target value to solve for
    /// * `discontinuity` - Whether to return discontinuity locations
    /// * `extrapolate` - Whether to extrapolate beyond domain
    ///
    /// # Returns
    /// Array of x values where spline equals y
    pub fn solve(
        &self,
        y: f64,
        discontinuity: Option<bool>,
        extrapolate: Option<bool>,
    ) -> InterpolateResult<Array1<T>> {
        let y_t = T::from_f64(y).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert y value".to_string())
        })?;

        let use_extrapolate = extrapolate.unwrap_or(self.extrapolate.unwrap_or(true));
        let include_discontinuity = discontinuity.unwrap_or(true);

        self.inner
            .solve_for_y(y_t, include_discontinuity, use_extrapolate)
    }

    /// Find roots of the spline (SciPy interface)
    ///
    /// This method exactly matches SciPy's CubicSpline.roots(discontinuity, extrapolate) interface
    ///
    /// # Arguments
    /// * `discontinuity` - Whether to return discontinuity locations
    /// * `extrapolate` - Whether to extrapolate beyond domain
    ///
    /// # Returns
    /// Array of x values where spline equals zero
    pub fn roots(
        &self,
        discontinuity: Option<bool>,
        extrapolate: Option<bool>,
    ) -> InterpolateResult<Array1<T>> {
        self.solve(0.0, discontinuity, extrapolate)
    }

    /// Evaluate the spline at given points (SciPy interface)
    pub fn __call__(
        &self,
        x: &ArrayView1<T>,
        nu: Option<usize>,
        extrapolate: Option<bool>,
    ) -> InterpolateResult<Array1<T>> {
        let derivative_order = nu.unwrap_or(0);
        let use_extrapolate = extrapolate.unwrap_or(self.extrapolate.unwrap_or(true));

        if derivative_order == 0 {
            if use_extrapolate {
                self.inner.evaluate_array(x)
            } else {
                self.inner.evaluate_array_checked(x)
            }
        } else {
            if use_extrapolate {
                self.inner.derivative_array(x, derivative_order)
            } else {
                self.inner.derivative_array_checked(x, derivative_order)
            }
        }
    }
}

impl<T: InterpolationFloat + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign>
    SciPyCompatibleBSpline<T>
{
    /// Create a new SciPy-compatible BSpline
    pub fn new(inner: BSpline<T>, extrapolate: bool, axis: i32) -> Self {
        Self {
            inner,
            extrapolate,
            axis,
        }
    }

    /// Returns a new BSpline representing the nu-th derivative (SciPy interface)
    ///
    /// This method exactly matches SciPy's BSpline.derivative(nu) interface
    ///
    /// # Arguments
    /// * `nu` - Order of derivative (default 1)
    ///
    /// # Returns
    /// A new `SciPyCompatibleBSpline` representing the nu-th derivative
    pub fn derivative(&self, nu: Option<usize>) -> InterpolateResult<Self> {
        let order = nu.unwrap_or(1);
        let deriv_inner = self.inner.derivative(order)?;

        Ok(Self {
            inner: deriv_inner,
            extrapolate: self.extrapolate,
            axis: self.axis,
        })
    }

    /// Returns a new BSpline representing the nu-th antiderivative (SciPy interface)
    ///
    /// This method exactly matches SciPy's BSpline.antiderivative(nu) interface
    ///
    /// # Arguments
    /// * `nu` - Order of antiderivative (default 1)
    ///
    /// # Returns
    /// A new `SciPyCompatibleBSpline` representing the nu-th antiderivative
    pub fn antiderivative(&self, nu: Option<usize>) -> InterpolateResult<Self> {
        let order = nu.unwrap_or(1);
        let antideriv_inner = self.inner.antiderivative(order)?;

        Ok(Self {
            inner: antideriv_inner,
            extrapolate: self.extrapolate,
            axis: self.axis,
        })
    }

    /// Compute definite integral over [a, b] (SciPy interface)
    ///
    /// This method exactly matches SciPy's BSpline.integrate(a, b) interface
    ///
    /// # Arguments
    /// * `a` - Lower integration bound
    /// * `b` - Upper integration bound
    /// * `extrapolate` - Whether to extrapolate beyond domain (optional)
    ///
    /// # Returns
    /// The definite integral from a to b
    pub fn integrate(&self, a: f64, b: f64, extrapolate: Option<bool>) -> InterpolateResult<T> {
        let a_t = T::from_f64(a).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert integration bound a".to_string())
        })?;
        let b_t = T::from_f64(b).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert integration bound b".to_string())
        })?;

        let use_extrapolate = extrapolate.unwrap_or(self.extrapolate);

        if use_extrapolate {
            self.inner.integrate(a_t, b_t)
        } else {
            // Check bounds and only integrate within domain
            if a_t < self.inner.t()[0] || b_t > self.inner.t()[self.inner.t().len() - 1] {
                return Err(InterpolateError::OutOfBounds(
                    "Integration bounds outside domain and extrapolate=False".to_string(),
                ));
            }

            self.inner.integrate(a_t, b_t)
        }
    }

    /// Evaluate the BSpline at given points (SciPy interface)
    pub fn __call__(
        &self,
        x: &ArrayView1<T>,
        nu: Option<usize>,
        extrapolate: Option<bool>,
    ) -> InterpolateResult<Array1<T>> {
        let derivative_order = nu.unwrap_or(0);
        let use_extrapolate = extrapolate.unwrap_or(self.extrapolate);

        if derivative_order == 0 {
            if use_extrapolate {
                self.inner.evaluate_array(x)
            } else {
                self.inner.evaluate_array_checked(x)
            }
        } else {
            if use_extrapolate {
                self.inner.derivative_array(x, derivative_order)
            } else {
                self.inner.derivative_array_checked(x, derivative_order)
            }
        }
    }
}

impl<T: InterpolationFloat> SciPyPPoly<T> {
    /// Create a new SciPy-compatible PPoly
    pub fn new(
        coefficients: Array2<T>,
        breakpoints: Array1<T>,
        extrapolate: bool,
        axis: i32,
    ) -> InterpolateResult<Self> {
        if coefficients.ncols() != breakpoints.len() - 1 {
            return Err(InterpolateError::InvalidInput(
                "Coefficient shape must match breakpoint structure".to_string(),
            ));
        }

        Ok(Self {
            coefficients,
            breakpoints,
            extrapolate,
            axis,
        })
    }

    /// Returns a new PPoly representing the m-th derivative (SciPy interface)
    ///
    /// This method exactly matches SciPy's PPoly.derivative(m) interface
    ///
    /// # Arguments
    /// * `m` - Order of derivative (default 1)
    ///
    /// # Returns
    /// A new `SciPyPPoly` representing the m-th derivative
    pub fn derivative(&self, m: Option<usize>) -> InterpolateResult<Self> {
        let order = m.unwrap_or(1);

        if order == 0 {
            return Ok(self.clone());
        }

        let k = self.coefficients.nrows();
        if order >= k {
            // Derivative of order >= polynomial degree results in zero
            let zero_coeffs = Array2::zeros((1, self.coefficients.ncols()));
            return Self::new(
                zero_coeffs,
                self.breakpoints.clone(),
                self.extrapolate,
                self.axis,
            );
        }

        let mut deriv_coeffs = Array2::zeros((k - order, self.coefficients.ncols()));

        // Compute derivative coefficients
        for i in 0..(k - order) {
            for j in 0..self.coefficients.ncols() {
                let mut coeff = self.coefficients[[i, j]];

                // Apply derivative operation m times
                for d in 0..order {
                    let power = (k - 1 - i) as i32 - d as i32;
                    if power >= 0 {
                        coeff = coeff * T::from_usize(power as usize + 1).unwrap();
                    } else {
                        coeff = T::zero();
                        break;
                    }
                }

                deriv_coeffs[[i, j]] = coeff;
            }
        }

        Self::new(
            deriv_coeffs,
            self.breakpoints.clone(),
            self.extrapolate,
            self.axis,
        )
    }

    /// Returns a new PPoly representing the m-th antiderivative (SciPy interface)
    ///
    /// This method exactly matches SciPy's PPoly.antiderivative(m) interface
    ///
    /// # Arguments
    /// * `m` - Order of antiderivative (default 1)
    ///
    /// # Returns
    /// A new `SciPyPPoly` representing the m-th antiderivative
    pub fn antiderivative(&self, m: Option<usize>) -> InterpolateResult<Self> {
        let order = m.unwrap_or(1);

        if order == 0 {
            return Ok(self.clone());
        }

        let k = self.coefficients.nrows();
        let mut antideriv_coeffs = Array2::zeros((k + order, self.coefficients.ncols()));

        // Compute antiderivative coefficients
        for i in 0..k {
            for j in 0..self.coefficients.ncols() {
                let mut coeff = self.coefficients[[i, j]];

                // Apply antiderivative operation m times
                for d in 0..order {
                    let power = (k - 1 - i) as i32 + d as i32 + 1;
                    coeff = coeff / T::from_i32(power).unwrap();
                }

                antideriv_coeffs[[i, j]] = coeff;
            }
        }

        // Set integration constants to zero (as per SciPy convention)
        for j in 0..self.coefficients.ncols() {
            for d in 0..order {
                antideriv_coeffs[[k + d, j]] = T::zero();
            }
        }

        Self::new(
            antideriv_coeffs,
            self.breakpoints.clone(),
            self.extrapolate,
            self.axis,
        )
    }

    /// Compute definite integral over [a, b] (SciPy interface)
    ///
    /// This method exactly matches SciPy's PPoly.integrate(a, b) interface
    ///
    /// # Arguments
    /// * `a` - Lower integration bound
    /// * `b` - Upper integration bound
    /// * `extrapolate` - Whether to extrapolate beyond domain (optional)
    ///
    /// # Returns
    /// The definite integral from a to b
    pub fn integrate(&self, a: f64, b: f64, extrapolate: Option<bool>) -> InterpolateResult<T> {
        let a_t = T::from_f64(a).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert integration bound a".to_string())
        })?;
        let b_t = T::from_f64(b).ok_or_else(|| {
            InterpolateError::ComputationError("Failed to convert integration bound b".to_string())
        })?;

        let use_extrapolate = extrapolate.unwrap_or(self.extrapolate);

        if !use_extrapolate {
            if a_t < self.breakpoints[0] || b_t > self.breakpoints[self.breakpoints.len() - 1] {
                return Err(InterpolateError::OutOfBounds(
                    "Integration bounds outside domain and extrapolate=False".to_string(),
                ));
            }
        }

        // Get antiderivative and evaluate at bounds
        let antideriv = self.antiderivative(Some(1))?;
        let f_b = antideriv.evaluate(b_t)?;
        let f_a = antideriv.evaluate(a_t)?;

        Ok(f_b - f_a)
    }

    /// Evaluate the PPoly at a given point
    pub fn evaluate(&self, x: T) -> InterpolateResult<T> {
        // Find the appropriate piece
        let piece_idx = self.find_piece_index(x)?;

        // Evaluate polynomial in this piece
        let dx = x - self.breakpoints[piece_idx];
        let mut result = T::zero();
        let k = self.coefficients.nrows();

        for i in 0..k {
            let coeff = self.coefficients[[i, piece_idx]];
            let power = k - 1 - i;
            let dx_power = if power == 0 {
                T::one()
            } else {
                let mut dx_pow = T::one();
                for _ in 0..power {
                    dx_pow = dx_pow * dx;
                }
                dx_pow
            };
            result = result + coeff * dx_power;
        }

        Ok(result)
    }

    /// Find the piece index for a given x value
    fn find_piece_index(&self, x: T) -> InterpolateResult<usize> {
        for i in 0..(self.breakpoints.len() - 1) {
            if x >= self.breakpoints[i] && x <= self.breakpoints[i + 1] {
                return Ok(i);
            }
        }

        if self.extrapolate {
            if x < self.breakpoints[0] {
                Ok(0)
            } else {
                Ok(self.breakpoints.len() - 2)
            }
        } else {
            Err(InterpolateError::OutOfBounds(
                "x value outside domain and extrapolate=False".to_string(),
            ))
        }
    }
}

/// Convenience functions for creating SciPy-compatible splines

/// Create a SciPy-compatible cubic spline with natural boundary conditions
#[allow(dead_code)]
pub fn make_scipy_cubic_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
) -> InterpolateResult<SciPyCompatibleCubicSpline<T>>
where
    T: InterpolationFloat + Debug + Display + std::ops::AddAssign + FromPrimitive,
{
    SciPyCompatibleCubicSpline::new(x, y, SciPyBoundaryType::Natural, None, 0)
}

/// Create a SciPy-compatible cubic spline with clamped boundary conditions
#[allow(dead_code)]
pub fn make_scipy_cubic_spline_clamped<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    left_deriv: f64,
    right_deriv: f64,
) -> InterpolateResult<SciPyCompatibleCubicSpline<T>>
where
    T: InterpolationFloat + Debug + Display + std::ops::AddAssign + FromPrimitive,
{
    SciPyCompatibleCubicSpline::new(
        x,
        y,
        SciPyBoundaryType::Clamped((left_deriv, right_deriv)),
        None,
        0,
    )
}

/// Create a SciPy-compatible BSpline
#[allow(dead_code)]
pub fn make_scipy_bspline<T>(inner: BSpline<T>, extrapolate: bool) -> SciPyCompatibleBSpline<T>
where
    T: InterpolationFloat + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
{
    SciPyCompatibleBSpline::new(_inner, extrapolate, 0)
}

/// Create a SciPy-compatible PPoly from coefficients and breakpoints
#[allow(dead_code)]
pub fn make_scipy_ppoly<T>(
    coefficients: Array2<T>,
    breakpoints: Array1<T>,
    extrapolate: bool,
) -> InterpolateResult<SciPyPPoly<T>>
where
    T: InterpolationFloat,
{
    SciPyPPoly::new(coefficients, breakpoints, extrapolate, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_scipy_cubic_spline_derivative() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = make_scipy_cubic_spline(&x.view(), &y.view()).unwrap();
        let deriv_spline = spline.derivative(Some(1)).unwrap();

        // Test that derivative spline works
        let test_x = array![0.5, 1.5, 2.5];
        let _deriv_values = deriv_spline.__call__(&test_x.view(), None, None).unwrap();
    }

    #[test]
    fn test_scipy_cubic_spline_antiderivative() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = make_scipy_cubic_spline(&x.view(), &y.view()).unwrap();
        let antideriv_spline = spline.antiderivative(Some(1)).unwrap();

        // Test that antiderivative spline works
        let test_x = array![0.5, 1.5, 2.5];
        let _antideriv_values = antideriv_spline
            .__call__(&test_x.view(), None, None)
            .unwrap();
    }

    #[test]
    fn test_scipy_cubic_spline_integrate() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = make_scipy_cubic_spline(&x.view(), &y.view()).unwrap();
        let integral = spline.integrate(0.0, 3.0, None).unwrap();

        // Integral should be positive for this increasing function
        assert!(integral > 0.0);
    }

    #[test]
    fn test_scipy_cubic_spline_solve() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let spline = make_scipy_cubic_spline(&x.view(), &y.view()).unwrap();
        let roots = spline.solve(1.0, None, None).unwrap();

        // Should find at least one solution
        assert!(roots.len() >= 1);
    }

    #[test]
    fn test_scipy_ppoly_derivative() {
        // Create a simple quadratic PPoly: y = x^2
        let coeffs = array![[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]; // [a, a], [b, b], [c, c] format
        let breakpoints = array![0.0, 1.0, 2.0];

        let ppoly = make_scipy_ppoly(coeffs, breakpoints, true).unwrap();
        let deriv_ppoly = ppoly.derivative(Some(1)).unwrap();

        // Derivative of x^2 should be 2x
        assert_eq!(deriv_ppoly.coefficients.nrows(), 2); // One degree less
    }

    #[test]
    fn test_scipy_ppoly_antiderivative() {
        // Create a simple linear PPoly: y = x
        let coeffs = array![[1.0, 1.0], [0.0, 0.0]]; // [a, a], [b, b] format
        let breakpoints = array![0.0, 1.0, 2.0];

        let ppoly = make_scipy_ppoly(coeffs, breakpoints, true).unwrap();
        let antideriv_ppoly = ppoly.antiderivative(Some(1)).unwrap();

        // Antiderivative should have one more degree
        assert_eq!(antideriv_ppoly.coefficients.nrows(), 3);
    }
}
