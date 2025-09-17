//! Integration methods for cubic splines
//!
//! This module provides integration functionality for cubic splines, including
//! basic integration within the spline domain and extrapolation-aware integration
//! for compatibility with SciPy's CubicSpline.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use super::core::CubicSpline;
use super::types::IntegrationRegion;

impl<F: InterpolationFloat + ToString> CubicSpline<F> {
    /// Integrate the spline from a to b
    ///
    /// Computes the definite integral of the cubic spline over the interval [a, b].
    /// The integration is performed analytically using the polynomial coefficients.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The definite integral of the spline from a to b
    ///
    /// # Errors
    ///
    /// Returns `InterpolateError::OutOfBounds` if either `a` or `b` is outside
    /// the interpolation domain.
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
    /// let integral = spline.integrate(0.5, 2.5).unwrap();
    /// println!("Integral from 0.5 to 2.5: {}", integral);
    /// ```
    pub fn integrate(&self, a: F, b: F) -> InterpolateResult<F> {
        // Handle reversed bounds
        if a > b {
            return Ok(-self.integrate(b, a)?);
        }

        if a == b {
            return Ok(F::zero());
        }

        // Check bounds
        let x_min = self.x()[0];
        let x_max = self.x()[self.x().len() - 1];

        if a < x_min || b > x_max {
            return Err(InterpolateError::OutOfBounds(
                "Integration bounds outside interpolation range".to_string(),
            ));
        }

        // Find the segments containing a and b
        let mut idx_a = 0;
        let mut idx_b = 0;

        for i in 0..self.x().len() - 1 {
            if a >= self.x()[i] && a <= self.x()[i + 1] {
                idx_a = i;
            }
            if b >= self.x()[i] && b <= self.x()[i + 1] {
                idx_b = i;
            }
        }

        let mut integral = F::zero();

        // If both points are in the same segment
        if idx_a == idx_b {
            integral = self.integrate_segment(idx_a, a, b)?;
        } else {
            // Integrate from a to the end of its segment
            integral = integral + self.integrate_segment(idx_a, a, self.x()[idx_a + 1])?;

            // Integrate all complete segments in between
            for i in (idx_a + 1)..idx_b {
                integral = integral + self.integrate_segment(i, self.x()[i], self.x()[i + 1])?;
            }

            // Integrate from the start of b's segment to b
            integral = integral + self.integrate_segment(idx_b, self.x()[idx_b], b)?;
        }

        Ok(integral)
    }

    /// SciPy-compatible integration method
    ///
    /// This is an alias for the `integrate` method to maintain compatibility
    /// with SciPy's CubicSpline interface.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The definite integral from a to b
    pub fn integrate_scipy(&self, a: F, b: F) -> InterpolateResult<F> {
        self.integrate(a, b)
    }

    /// Integrate the spline from a to b with extrapolation support
    ///
    /// This enhanced integration method supports extrapolation when integration bounds
    /// extend beyond the spline domain, providing full SciPy compatibility.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    /// * `extrapolate` - Extrapolation mode for out-of-bounds integration
    ///   - `None`: Use default behavior (error for out-of-bounds)
    ///   - `Some(ExtrapolateMode::Error)`: Raise error for out-of-bounds
    ///   - `Some(ExtrapolateMode::Extrapolate)`: Linear extrapolation using endpoint derivatives
    ///   - `Some(ExtrapolateMode::Nearest)`: Constant extrapolation using boundary values
    ///
    /// # Returns
    ///
    /// The definite integral of the spline from a to b
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    /// use scirs2_interpolate::interp1d::ExtrapolateMode;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    /// let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// // Integrate within domain
    /// let integral1 = spline.integrate_with_extrapolation(0.5, 2.5, None).unwrap();
    ///
    /// // Integrate with extrapolation beyond domain
    /// let integral2 = spline.integrate_with_extrapolation(-1.0, 4.0,
    ///     Some(ExtrapolateMode::Extrapolate)).unwrap();
    /// ```
    pub fn integrate_with_extrapolation(
        &self,
        a: F,
        b: F,
        extrapolate: Option<crate::interp1d::ExtrapolateMode>,
    ) -> InterpolateResult<F> {
        if a == b {
            return Ok(F::zero());
        }

        // Handle reversed bounds
        if a > b {
            return Ok(-self.integrate_with_extrapolation(b, a, extrapolate)?);
        }

        let x_min = self.x()[0];
        let x_max = self.x()[self.x().len() - 1];

        // If both bounds are within domain, use standard integration
        if a >= x_min && b <= x_max {
            return self.integrate(a, b);
        }

        // Handle extrapolation cases
        let extrapolate_mode = extrapolate.unwrap_or(crate::interp1d::ExtrapolateMode::Error);

        match extrapolate_mode {
            crate::interp1d::ExtrapolateMode::Error => {
                return Err(InterpolateError::OutOfBounds(
                    "Integration bounds outside interpolation range and extrapolate=false".to_string(),
                ));
            }
            crate::interp1d::ExtrapolateMode::Extrapolate => {
                self.integrate_with_linear_extrapolation(a, b)
            }
            crate::interp1d::ExtrapolateMode::Nearest => {
                self.integrate_with_constant_extrapolation(a, b)
            }
        }
    }

    /// Integrate a single polynomial segment from x1 to x2
    ///
    /// This method computes the definite integral of a cubic polynomial
    /// over the specified interval.
    ///
    /// # Arguments
    ///
    /// * `segment` - Index of the polynomial segment
    /// * `x1` - Lower bound of integration
    /// * `x2` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The integral of the polynomial segment from x1 to x2
    ///
    /// # Errors
    ///
    /// Returns an error if float conversion fails.
    pub(crate) fn integrate_segment(&self, segment: usize, x1: F, x2: F) -> InterpolateResult<F> {
        if x1 == x2 {
            return Ok(F::zero());
        }

        let x_base = self.x()[segment];
        let dx1 = x1 - x_base;
        let dx2 = x2 - x_base;

        let a = self.coeffs()[[segment, 0]];
        let b = self.coeffs()[[segment, 1]];
        let c = self.coeffs()[[segment, 2]];
        let d = self.coeffs()[[segment, 3]];

        // Convert constants
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
        let four = F::from_f64(4.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 4.0 to float type".to_string(),
            )
        })?;

        // Antiderivative of a + b*dx + c*dx^2 + d*dx^3 is:
        // a*dx + b*dx^2/2 + c*dx^3/3 + d*dx^4/4
        let antiderivative = |dx: F| -> InterpolateResult<F> {
            Ok(a * dx
                + b * dx * dx / two
                + c * dx * dx * dx / three
                + d * dx * dx * dx * dx / four)
        };

        let upper = antiderivative(dx2)?;
        let lower = antiderivative(dx1)?;

        Ok(upper - lower)
    }

    /// Integrate with linear extrapolation beyond domain boundaries
    ///
    /// This method extends the spline linearly beyond its domain using the
    /// endpoint derivatives, then integrates over the extended range.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The integral including extrapolated regions
    fn integrate_with_linear_extrapolation(&self, a: F, b: F) -> InterpolateResult<F> {
        let x_min = self.x()[0];
        let x_max = self.x()[self.x().len() - 1];
        let mut integral = F::zero();

        // Determine integration regions
        let a_clamped = a.max(x_min);
        let b_clamped = b.min(x_max);

        // Left extrapolation region
        if a < x_min {
            let region_end = b.min(x_min);
            integral = integral + self.integrate_left_extrapolation(a, region_end)?;
        }

        // Interior region
        if a_clamped < b_clamped {
            integral = integral + self.integrate(a_clamped, b_clamped)?;
        }

        // Right extrapolation region
        if b > x_max {
            let region_start = a.max(x_max);
            integral = integral + self.integrate_right_extrapolation(region_start, b)?;
        }

        Ok(integral)
    }

    /// Integrate with constant extrapolation beyond domain boundaries
    ///
    /// This method extends the spline with constant values beyond its domain,
    /// then integrates over the extended range.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The integral including extrapolated regions
    fn integrate_with_constant_extrapolation(&self, a: F, b: F) -> InterpolateResult<F> {
        let x_min = self.x()[0];
        let x_max = self.x()[self.x().len() - 1];
        let mut integral = F::zero();

        // Determine integration regions
        let a_clamped = a.max(x_min);
        let b_clamped = b.min(x_max);

        // Left extrapolation region (constant y[0])
        if a < x_min {
            let region_end = b.min(x_min);
            let width = region_end - a;
            integral = integral + self.y()[0] * width;
        }

        // Interior region
        if a_clamped < b_clamped {
            integral = integral + self.integrate(a_clamped, b_clamped)?;
        }

        // Right extrapolation region (constant y[n-1])
        if b > x_max {
            let region_start = a.max(x_max);
            let width = b - region_start;
            let n = self.y().len() - 1;
            integral = integral + self.y()[n] * width;
        }

        Ok(integral)
    }

    /// Integrate the left linear extrapolation from a to b
    ///
    /// Uses linear extrapolation based on the left endpoint derivative.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound (must be < x_min)
    /// * `b` - Upper bound (must be <= x_min)
    ///
    /// # Returns
    ///
    /// The integral of the left extrapolation
    fn integrate_left_extrapolation(&self, a: F, b: F) -> InterpolateResult<F> {
        let x0 = self.x()[0];
        let y0 = self.y()[0];
        let dy0 = self.derivative_n(x0, 1)?;

        // Linear function: y(x) = y0 + dy0 * (x - x0)
        // Integral from a to b: (b-a) * y0 + dy0 * (b^2 - a^2)/2 - dy0 * x0 * (b-a)
        let width = b - a;
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string(),
            )
        })?;

        let linear_term = width * y0;
        let quadratic_term = dy0 * (b * b - a * a) / two;
        let offset_term = dy0 * x0 * width;

        Ok(linear_term + quadratic_term - offset_term)
    }

    /// Integrate the right linear extrapolation from a to b
    ///
    /// Uses linear extrapolation based on the right endpoint derivative.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound (must be >= x_max)
    /// * `b` - Upper bound (must be > x_max)
    ///
    /// # Returns
    ///
    /// The integral of the right extrapolation
    fn integrate_right_extrapolation(&self, a: F, b: F) -> InterpolateResult<F> {
        let n = self.x().len() - 1;
        let xn = self.x()[n];
        let yn = self.y()[n];
        let dyn_val = self.derivative_n(xn, 1)?;

        // Linear function: y(x) = yn + dyn * (x - xn)
        let width = b - a;
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string(),
            )
        })?;

        let linear_term = width * yn;
        let quadratic_term = dyn_val * (b * b - a * a) / two;
        let offset_term = dyn_val * xn * width;

        Ok(linear_term + quadratic_term - offset_term)
    }

    /// Classify the integration region relative to the spline domain
    ///
    /// This helper function determines whether the integration bounds fall
    /// within the spline domain or require extrapolation.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// Classification of the integration region
    pub(crate) fn classify_integration_region(&self, a: F, b: F) -> IntegrationRegion {
        let x_min = self.x()[0];
        let x_max = self.x()[self.x().len() - 1];

        if b <= x_min {
            IntegrationRegion::LeftExtrapolation
        } else if a >= x_max {
            IntegrationRegion::RightExtrapolation
        } else {
            IntegrationRegion::Interior
        }
    }
}