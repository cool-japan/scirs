//! Evaluation and derivative computation methods for cubic splines
//!
//! This module contains all methods related to evaluating splines and computing
//! their derivatives at given points. It includes both single-point and batch
//! evaluation functions, as well as specialized methods for different use cases.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use ndarray::{Array1, ArrayView1};
use super::core::CubicSpline;

impl<F: InterpolationFloat + ToString> CubicSpline<F> {
    /// Evaluate the spline at a single point
    ///
    /// Computes the interpolated value at the given x coordinate using the
    /// cubic polynomial representation of the spline.
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinate at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The interpolated y value at `xnew`
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::OutOfBounds` if `xnew` is outside the
    /// interpolation range [x₀, xₙ].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// let result = spline.evaluate(1.5).unwrap();
    /// println!("f(1.5) = {}", result);
    /// ```
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        // Check if xnew is within the range
        if xnew < self.x()[0] || xnew > self.x()[self.x().len() - 1] {
            return Err(InterpolateError::OutOfBounds(
                "xnew is outside the interpolation range".to_string(),
            ));
        }

        // Find the index of the segment containing xnew
        let mut idx = 0;
        for i in 0..self.x().len() - 1 {
            if xnew >= self.x()[i] && xnew <= self.x()[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: xnew is exactly the last point
        if xnew == self.x()[self.x().len() - 1] {
            return Ok(self.y()[self.x().len() - 1]);
        }

        // Evaluate the cubic polynomial
        let dx = xnew - self.x()[idx];
        let a = self.coeffs()[[idx, 0]];
        let b = self.coeffs()[[idx, 1]];
        let c = self.coeffs()[[idx, 2]];
        let d = self.coeffs()[[idx, 3]];

        let result = a + b * dx + c * dx * dx + d * dx * dx * dx;
        Ok(result)
    }

    /// Evaluate the spline at multiple points
    ///
    /// Efficiently computes interpolated values at multiple x coordinates.
    /// This is more efficient than calling `evaluate` multiple times.
    ///
    /// # Arguments
    ///
    /// * `xnew` - Array of x coordinates at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// Array of interpolated y values at each point in `xnew`
    ///
    /// # Errors
    ///
    /// Returns an error if any point in `xnew` is outside the interpolation range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// let xnew = array![0.5, 1.5, 2.5];
    /// let results = spline.evaluate_array(&xnew.view()).unwrap();
    /// ```
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());
        for (i, &x) in xnew.iter().enumerate() {
            result[i] = self.evaluate(x)?;
        }
        Ok(result)
    }

    /// Get the first derivative of the spline at the given point
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinate at which to evaluate the derivative
    ///
    /// # Returns
    ///
    /// The first derivative at `xnew`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// let slope = spline.derivative(1.5).unwrap();
    /// println!("f'(1.5) = {}", slope);
    /// ```
    pub fn derivative(&self, xnew: F) -> InterpolateResult<F> {
        self.derivative_n(xnew, 1)
    }

    /// Get the nth derivative of the spline at the given point
    ///
    /// Computes derivatives of any order from 0 to 3. For cubic splines,
    /// derivatives of order greater than 3 are always zero.
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinate at which to evaluate the derivative
    /// * `order` - Order of derivative (0 = function value, 1 = first derivative, etc.)
    ///
    /// # Returns
    ///
    /// The nth derivative at `xnew`
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::OutOfBounds` if `xnew` is outside the
    /// interpolation range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// let first_deriv = spline.derivative_n(1.5, 1).unwrap();
    /// let second_deriv = spline.derivative_n(1.5, 2).unwrap();
    /// let third_deriv = spline.derivative_n(1.5, 3).unwrap();
    /// ```
    pub fn derivative_n(&self, xnew: F, order: usize) -> InterpolateResult<F> {
        // Check order validity
        if order == 0 {
            return self.evaluate(xnew);
        }

        if order > 3 {
            // Cubic spline has zero derivatives of order > 3
            return Ok(F::zero());
        }

        // Check if xnew is within the range
        if xnew < self.x()[0] || xnew > self.x()[self.x().len() - 1] {
            return Err(InterpolateError::OutOfBounds(
                "xnew is outside the interpolation range".to_string(),
            ));
        }

        // Find the index of the segment containing xnew
        let mut idx = 0;
        for i in 0..self.x().len() - 1 {
            if xnew >= self.x()[i] && xnew <= self.x()[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: xnew is exactly the last point
        if xnew == self.x()[self.x().len() - 1] {
            idx = self.x().len() - 2;
        }

        let dx = xnew - self.x()[idx];
        let b = self.coeffs()[[idx, 1]];
        let c = self.coeffs()[[idx, 2]];
        let d = self.coeffs()[[idx, 3]];

        match order {
            1 => {
                // First derivative: b + 2*c*dx + 3*d*dx^2
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string(),
                    )
                })?;
                let three = F::from_f64(3.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 3.0 to float type".to_string(),
                    )
                })?;
                Ok(b + two * c * dx + three * d * dx * dx)
            }
            2 => {
                // Second derivative: 2*c + 6*d*dx
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string(),
                    )
                })?;
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string(),
                    )
                })?;
                Ok(two * c + six * d * dx)
            }
            3 => {
                // Third derivative: 6*d
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string(),
                    )
                })?;
                Ok(six * d)
            }
            _ => Ok(F::zero()),
        }
    }

    /// Compute derivatives at multiple points
    ///
    /// Efficiently computes derivatives of a specified order at multiple x coordinates.
    ///
    /// # Arguments
    ///
    /// * `xnew` - Array of points to evaluate derivatives at
    /// * `order` - Order of derivative (1, 2, or 3)
    ///
    /// # Returns
    ///
    /// Array of derivative values at each point
    ///
    /// # Errors
    ///
    /// Returns an error if any point in `xnew` is outside the interpolation range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// let xnew = array![0.5, 1.5, 2.5];
    /// let derivatives = spline.derivative_array(&xnew.view(), 1).unwrap();
    /// ```
    pub fn derivative_array(
        &self,
        xnew: &ArrayView1<F>,
        order: usize,
    ) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());

        for (i, &x) in xnew.iter().enumerate() {
            result[i] = self.derivative_n(x, order)?;
        }

        Ok(result)
    }

    /// Compute all derivatives up to the specified order at a point
    ///
    /// Returns an array containing the function value and all derivatives
    /// from order 1 up to `maxorder` at the given point.
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinate at which to evaluate derivatives
    /// * `maxorder` - Maximum order of derivative to compute (≤ 3)
    ///
    /// # Returns
    ///
    /// Array of shape (maxorder + 1) containing [f(x), f'(x), f''(x), f'''(x)]
    /// up to the specified maximum order.
    ///
    /// # Errors
    ///
    /// Returns an error if `xnew` is outside the interpolation range or if
    /// `maxorder > 3`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// let all_derivs = spline.derivatives_all(1.5, 3).unwrap();
    /// // all_derivs[0] = f(1.5)
    /// // all_derivs[1] = f'(1.5)
    /// // all_derivs[2] = f''(1.5)
    /// // all_derivs[3] = f'''(1.5)
    /// ```
    pub fn derivatives_all(&self, xnew: F, maxorder: usize) -> InterpolateResult<Array1<F>> {
        if maxorder > 3 {
            return Err(InterpolateError::invalid_input(
                "Maximum derivative order for cubic splines is 3".to_string(),
            ));
        }

        let mut result = Array1::zeros(maxorder + 1);

        // Function value
        result[0] = self.evaluate(xnew)?;

        // Derivatives
        for order in 1..=maxorder {
            result[order] = self.derivative_n(xnew, order)?;
        }

        Ok(result)
    }

    /// Evaluate the spline array with bounds checking
    ///
    /// Similar to `evaluate_array` but provides more detailed error reporting
    /// when points are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `xnew` - Array of x coordinates to evaluate
    ///
    /// # Returns
    ///
    /// Array of interpolated values
    ///
    /// # Errors
    ///
    /// Returns detailed error information if any evaluation point is out of bounds.
    pub fn evaluate_array_checked(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());

        for (i, &x) in xnew.iter().enumerate() {
            match self.evaluate(x) {
                Ok(val) => result[i] = val,
                Err(e) => {
                    return Err(InterpolateError::invalid_input(format!(
                        "Evaluation failed at index {}: {}",
                        i, e
                    )));
                }
            }
        }

        Ok(result)
    }

    /// Compute derivative array with bounds checking
    ///
    /// Similar to `derivative_array` but provides more detailed error reporting
    /// when points are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `xnew` - Array of x coordinates to evaluate derivatives at
    /// * `order` - Order of derivative
    ///
    /// # Returns
    ///
    /// Array of derivative values
    ///
    /// # Errors
    ///
    /// Returns detailed error information if any evaluation point is out of bounds.
    pub fn derivative_array_checked(
        &self,
        xnew: &ArrayView1<F>,
        order: usize,
    ) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());

        for (i, &x) in xnew.iter().enumerate() {
            match self.derivative_n(x, order) {
                Ok(val) => result[i] = val,
                Err(e) => {
                    return Err(InterpolateError::invalid_input(format!(
                        "Derivative evaluation failed at index {}: {}",
                        i, e
                    )));
                }
            }
        }

        Ok(result)
    }
}

/// Helper functions for efficient evaluation
impl<F: InterpolationFloat + ToString> CubicSpline<F> {
    /// Find the segment index for a given x value using binary search
    ///
    /// This is an internal helper function that efficiently finds which
    /// polynomial segment contains the given x value.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate to search for
    ///
    /// # Returns
    ///
    /// The index of the segment containing x, or the last segment if x
    /// is exactly at the right boundary.
    #[inline]
    pub(crate) fn find_segment(&self, x: F) -> usize {
        // Use binary search for better performance with large datasets
        let mut left = 0;
        let mut right = self.x().len() - 1;

        // Handle exact match at endpoints
        if x <= self.x()[0] {
            return 0;
        }
        if x >= self.x()[right] {
            return right - 1;
        }

        while left < right - 1 {
            let mid = (left + right) / 2;
            if x < self.x()[mid] {
                right = mid;
            } else {
                left = mid;
            }
        }

        left
    }

    /// Evaluate the polynomial in a segment without bounds checking
    ///
    /// This is an internal helper function for fast evaluation when the
    /// segment index is already known and bounds have been checked.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate to evaluate
    /// * `segment_idx` - The index of the polynomial segment
    ///
    /// # Returns
    ///
    /// The evaluated polynomial value
    #[inline]
    pub(crate) fn evaluate_segment_unchecked(&self, x: F, segment_idx: usize) -> F {
        let dx = x - self.x()[segment_idx];
        let a = self.coeffs()[[segment_idx, 0]];
        let b = self.coeffs()[[segment_idx, 1]];
        let c = self.coeffs()[[segment_idx, 2]];
        let d = self.coeffs()[[segment_idx, 3]];

        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    /// Evaluate the derivative of a polynomial in a segment without bounds checking
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate to evaluate
    /// * `segment_idx` - The index of the polynomial segment
    /// * `order` - The derivative order (1, 2, or 3)
    ///
    /// # Returns
    ///
    /// The evaluated derivative value, or an error if the conversion fails
    #[inline]
    pub(crate) fn evaluate_derivative_segment_unchecked(
        &self,
        x: F,
        segment_idx: usize,
        order: usize,
    ) -> InterpolateResult<F> {
        let dx = x - self.x()[segment_idx];
        let b = self.coeffs()[[segment_idx, 1]];
        let c = self.coeffs()[[segment_idx, 2]];
        let d = self.coeffs()[[segment_idx, 3]];

        match order {
            1 => {
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string(),
                    )
                })?;
                let three = F::from_f64(3.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 3.0 to float type".to_string(),
                    )
                })?;
                Ok(b + two * c * dx + three * d * dx * dx)
            }
            2 => {
                let two = F::from_f64(2.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 2.0 to float type".to_string(),
                    )
                })?;
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string(),
                    )
                })?;
                Ok(two * c + six * d * dx)
            }
            3 => {
                let six = F::from_f64(6.0).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert constant 6.0 to float type".to_string(),
                    )
                })?;
                Ok(six * d)
            }
            _ => Ok(F::zero()),
        }
    }
}