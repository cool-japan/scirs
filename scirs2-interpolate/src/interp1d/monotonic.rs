//! Monotonic interpolation methods beyond PCHIP
//!
//! This module provides additional monotonic interpolation methods that preserve
//! the monotonicity of the input data, avoiding unwanted oscillations that can occur
//! with standard cubic spline interpolation.
//!
//! The methods include:
//! - Hyman filtered cubic spline interpolation
//! - Steffen's method for guaranteed monotonicity
//! - Modified Akima interpolation with monotonicity preservation
//!
//! These methods have different characteristics and may be suitable for different
//! types of data, but all preserve monotonicity where the data is monotonic.

use crate::error::{InterpolateError, InterpolateResult};
use crate::spline::CubicSpline;
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Enum for the different monotonic interpolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonotonicMethod {
    /// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    /// The default method, already implemented in the pchip module
    Pchip,

    /// Hyman filtering applied to a cubic spline
    /// Reduces the derivatives at each point if necessary to ensure monotonicity
    Hyman,

    /// Steffen's method for monotonic interpolation
    /// Guarantees monotonicity and no overshooting by construction
    Steffen,

    /// Modified Akima interpolation with monotonicity preservation
    /// Combines Akima's robustness to outliers with monotonicity preservation
    ModifiedAkima,
}

/// Monotonic interpolator that preserves monotonicity in the data
///
/// All these methods guarantee that the resulting interpolation function will
/// preserve monotonicity in the data, meaning that if the data is increasing/decreasing
/// over an interval, the interpolant will be as well.
#[derive(Debug, Clone)]
pub struct MonotonicInterpolator<F: Float> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Derivatives at points
    derivatives: Array1<F>,
    /// Interpolation method used
    method: MonotonicMethod,
    /// Extrapolation mode
    extrapolate: bool,
}

impl<F: Float> MonotonicInterpolator<F> {
    /// Get the interpolation method
    pub fn method(&self) -> MonotonicMethod {
        self.method
    }
}

impl<F: Float + FromPrimitive + Debug> MonotonicInterpolator<F> {
    /// Create a new monotonic interpolator
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `method` - The monotonic interpolation method to use
    /// * `extrapolate` - Whether to extrapolate beyond the data range
    ///
    /// # Returns
    ///
    /// A new `MonotonicInterpolator` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `x` and `y` have different lengths
    /// - `x` is not sorted in ascending order
    /// - There are fewer than 2 points
    /// - For some methods, if there are fewer than the required number of points
    pub fn new(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        method: MonotonicMethod,
        extrapolate: bool,
    ) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::ValueError(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::ValueError(
                "at least 2 points are required for interpolation".to_string(),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::ValueError(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Clone input arrays
        let x_arr = x.to_owned();
        let y_arr = y.to_owned();

        // Compute derivatives based on the selected method
        let derivatives = match method {
            MonotonicMethod::Pchip => Self::find_pchip_derivatives(&x_arr, &y_arr)?,
            MonotonicMethod::Hyman => Self::find_hyman_derivatives(&x_arr, &y_arr)?,
            MonotonicMethod::Steffen => Self::find_steffen_derivatives(&x_arr, &y_arr)?,
            MonotonicMethod::ModifiedAkima => {
                Self::find_modified_akima_derivatives(&x_arr, &y_arr)?
            }
        };

        Ok(MonotonicInterpolator {
            x: x_arr,
            y: y_arr,
            derivatives,
            method,
            extrapolate,
        })
    }

    /// Evaluate the interpolation at the given point
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinate at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y value at `x_new`
    ///
    /// # Errors
    ///
    /// Returns an error if `x_new` is outside the interpolation range and
    /// extrapolation is disabled.
    pub fn evaluate(&self, x_new: F) -> InterpolateResult<F> {
        // Check if we're extrapolating
        let is_extrapolating = x_new < self.x[0] || x_new > self.x[self.x.len() - 1];
        if is_extrapolating && !self.extrapolate {
            return Err(InterpolateError::DomainError(
                "x_new is outside the interpolation range".to_string(),
            ));
        }

        // Find index of segment containing x_new
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if x_new >= self.x[i] && x_new <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Handle extrapolation case
        if is_extrapolating {
            if x_new < self.x[0] {
                // Extrapolate below the data range using the first segment
                idx = 0;
            } else {
                // Extrapolate above the data range using the last segment
                idx = self.x.len() - 2;
            }
        }

        // Special case: x_new is exactly at a knot point
        for i in 0..self.x.len() {
            if x_new == self.x[i] {
                return Ok(self.y[i]);
            }
        }

        // Get coordinates and derivatives for the segment
        let x1 = self.x[idx];
        let x2 = self.x[idx + 1];
        let y1 = self.y[idx];
        let y2 = self.y[idx + 1];
        let d1 = self.derivatives[idx];
        let d2 = self.derivatives[idx + 1];

        // Normalized position within the interval [x1, x2]
        let h = x2 - x1;
        let t = (x_new - x1) / h;

        // Compute Hermite basis functions
        let h00 = Self::h00(t);
        let h10 = Self::h10(t);
        let h01 = Self::h01(t);
        let h11 = Self::h11(t);

        // Evaluate cubic Hermite polynomial
        let result = h00 * y1 + h10 * h * d1 + h01 * y2 + h11 * h * d2;

        Ok(result)
    }

    /// Evaluate the interpolation at multiple points
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinates at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y values at `x_new`
    ///
    /// # Errors
    ///
    /// Returns an error if any point in `x_new` is outside the interpolation range
    /// and extrapolation is disabled.
    pub fn evaluate_array(&self, x_new: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(x_new.len());
        for (i, &x) in x_new.iter().enumerate() {
            result[i] = self.evaluate(x)?;
        }
        Ok(result)
    }

    /// Hermite basis function h₀₀(t)
    fn h00(t: F) -> F {
        let two = F::from_f64(2.0).unwrap();
        let three = F::from_f64(3.0).unwrap();
        (two * t * t * t) - (three * t * t) + F::one()
    }

    /// Hermite basis function h₁₀(t)
    fn h10(t: F) -> F {
        let two = F::from_f64(2.0).unwrap();
        (t * t * t) - (two * t * t) + t
    }

    /// Hermite basis function h₀₁(t)
    fn h01(t: F) -> F {
        let two = F::from_f64(2.0).unwrap();
        let three = F::from_f64(3.0).unwrap();
        -(two * t * t * t) + (three * t * t)
    }

    /// Hermite basis function h₁₁(t)
    fn h11(t: F) -> F {
        (t * t * t) - (t * t)
    }

    /// Compute PCHIP derivatives - same as the regular PCHIP implementation
    fn find_pchip_derivatives(x: &Array1<F>, y: &Array1<F>) -> InterpolateResult<Array1<F>> {
        let n = x.len();
        let mut derivatives = Array1::zeros(n);

        // Handle special case: only two points, use linear interpolation
        if n == 2 {
            let slope = (y[1] - y[0]) / (x[1] - x[0]);
            derivatives[0] = slope;
            derivatives[1] = slope;
            return Ok(derivatives);
        }

        // Calculate slopes between segments: m_k = (y_{k+1} - y_k) / (x_{k+1} - x_k)
        let mut slopes = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            slopes[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }

        // Calculate spacings: h_k = x_{k+1} - x_k
        let mut h = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            h[i] = x[i + 1] - x[i];
        }

        // For interior points, use PCHIP formula
        let two = F::from_f64(2.0).unwrap();
        let three = F::from_f64(3.0).unwrap();

        for i in 1..n - 1 {
            // Determine if slopes have different signs or if either is zero
            let prev_slope = slopes[i - 1];
            let curr_slope = slopes[i];

            let sign_prev = if prev_slope > F::zero() {
                F::one()
            } else if prev_slope < F::zero() {
                -F::one()
            } else {
                F::zero()
            };

            let sign_curr = if curr_slope > F::zero() {
                F::one()
            } else if curr_slope < F::zero() {
                -F::one()
            } else {
                F::zero()
            };

            // If signs are different or either slope is zero, set derivative to zero
            if sign_prev * sign_curr <= F::zero() {
                derivatives[i] = F::zero();
            } else {
                // Use weighted harmonic mean
                let w1 = two * h[i] + h[i - 1];
                let w2 = h[i] + two * h[i - 1];

                // Compute harmonic mean
                if prev_slope.abs() < F::epsilon() || curr_slope.abs() < F::epsilon() {
                    derivatives[i] = F::zero();
                } else {
                    let whmean_inv = (w1 / prev_slope + w2 / curr_slope) / (w1 + w2);
                    derivatives[i] = F::one() / whmean_inv;
                }
            }
        }

        // Special case: endpoints
        // For the first point
        let h0 = h[0];
        let h1 = if n > 2 { h[1] } else { h[0] };
        let m0 = slopes[0];
        let m1 = if n > 2 { slopes[1] } else { slopes[0] };

        // One-sided three-point estimate for the derivative
        let mut d = ((two * h0 + h1) * m0 - h0 * m1) / (h0 + h1);

        // Try to preserve shape
        let sign_d = if d >= F::zero() { F::one() } else { -F::one() };
        let sign_m0 = if m0 >= F::zero() { F::one() } else { -F::one() };
        let sign_m1 = if m1 >= F::zero() { F::one() } else { -F::one() };

        // If the signs are different or abs(d) > 3*abs(m0), adjust d
        if sign_d != sign_m0 {
            d = F::zero();
        } else if (sign_m0 != sign_m1) && (d.abs() > three * m0.abs()) {
            d = three * m0;
        }
        derivatives[0] = d;

        // For the last point
        let h0 = h[n - 2];
        let h1 = if n > 2 { h[n - 3] } else { h[n - 2] };
        let m0 = slopes[n - 2];
        let m1 = if n > 2 { slopes[n - 3] } else { slopes[n - 2] };

        // One-sided three-point estimate for the derivative
        let mut d = ((two * h0 + h1) * m0 - h0 * m1) / (h0 + h1);

        // Try to preserve shape
        let sign_d = if d >= F::zero() { F::one() } else { -F::one() };
        let sign_m0 = if m0 >= F::zero() { F::one() } else { -F::one() };
        let sign_m1 = if m1 >= F::zero() { F::one() } else { -F::one() };

        // If the signs are different or abs(d) > 3*abs(m0), adjust d
        if sign_d != sign_m0 {
            d = F::zero();
        } else if (sign_m0 != sign_m1) && (d.abs() > three * m0.abs()) {
            d = three * m0;
        }
        derivatives[n - 1] = d;

        Ok(derivatives)
    }

    /// Compute Hyman-filtered cubic spline derivatives
    ///
    /// This method starts with a standard cubic spline and then applies Hyman filtering
    /// to ensure monotonicity is preserved.
    ///
    /// Reference: Hyman, J. M. (1983). "Accurate monotonicity preserving cubic interpolation".
    fn find_hyman_derivatives(x: &Array1<F>, y: &Array1<F>) -> InterpolateResult<Array1<F>> {
        let n = x.len();
        let mut derivatives = Array1::zeros(n);

        // Start with a natural cubic spline (zero second derivatives at endpoints)
        if let Ok(spline) = CubicSpline::new(&x.view(), &y.view()) {
            // Get the derivatives from the cubic spline
            for i in 0..n {
                derivatives[i] = spline.derivative(x[i]).unwrap_or(F::zero());
            }
        } else {
            // If the cubic spline fails, fall back to PCHIP
            return Self::find_pchip_derivatives(x, y);
        }

        // Calculate segment slopes
        let mut slopes = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            slopes[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }

        // Apply Hyman filtering to ensure monotonicity
        let three = F::from_f64(3.0).unwrap();

        for i in 0..n {
            // For interior points, check adjacent slopes
            if i > 0 && i < n - 1 {
                let left_slope = slopes[i - 1];
                let right_slope = slopes[i];

                // Determine monotonicity
                if left_slope * right_slope <= F::zero() {
                    // Not monotonic, set derivative to zero
                    derivatives[i] = F::zero();
                } else {
                    // Monotonic, check for overshoot
                    let max_slope = three * F::min(left_slope.abs(), right_slope.abs());
                    if derivatives[i].abs() > max_slope {
                        // Reduce derivative to ensure monotonicity
                        derivatives[i] = max_slope * derivatives[i].signum();
                    }
                }
            } else if i == 0 {
                // First point: ensure derivative matches sign of first slope
                let slope = slopes[0];
                if slope * derivatives[0] < F::zero() {
                    // Different signs, set to zero
                    derivatives[0] = F::zero();
                } else if derivatives[0].abs() > three * slope.abs() {
                    // Reduce derivative to at most 3 times the slope
                    derivatives[0] = three * slope;
                }
            } else {
                // Last point: ensure derivative matches sign of last slope
                let slope = slopes[n - 2];
                if slope * derivatives[n - 1] < F::zero() {
                    // Different signs, set to zero
                    derivatives[n - 1] = F::zero();
                } else if derivatives[n - 1].abs() > three * slope.abs() {
                    // Reduce derivative to at most 3 times the slope
                    derivatives[n - 1] = three * slope;
                }
            }
        }

        Ok(derivatives)
    }

    /// Compute Steffen's monotonic derivatives
    ///
    /// Steffen's method guarantees monotonicity and no overshooting by construction.
    /// It's more restrictive than other methods but ensures a well-behaved interpolant.
    ///
    /// Reference: Steffen, M. (1990). "A simple method for monotonic interpolation in one dimension".
    fn find_steffen_derivatives(x: &Array1<F>, y: &Array1<F>) -> InterpolateResult<Array1<F>> {
        let n = x.len();
        let mut derivatives = Array1::zeros(n);

        // Handle special case: only two points, use linear interpolation
        if n == 2 {
            let slope = (y[1] - y[0]) / (x[1] - x[0]);
            derivatives[0] = slope;
            derivatives[1] = slope;
            return Ok(derivatives);
        }

        // Calculate segment slopes
        let mut slopes = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            slopes[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }

        // Calculate derivatives for interior points using Steffen's method
        for i in 1..n - 1 {
            let p1 = slopes[i - 1]; // Left slope
            let p2 = slopes[i]; // Right slope

            // Calculate auxiliary values for monotonicity conditions
            let h1 = x[i] - x[i - 1];
            let h2 = x[i + 1] - x[i];

            // Weighted average of secant slopes
            let a = (h2 * p1 + h1 * p2) / (h1 + h2);

            // Bound derivatives to prevent overshooting
            let min_slope = F::min(p1.abs(), p2.abs());
            let min_slope_signed = if p1 * p2 > F::zero() {
                min_slope * p1.signum()
            } else {
                F::zero()
            };

            // Set derivative to maintain monotonicity
            derivatives[i] = min_slope_signed;

            // Only if the slopes have the same sign, use a weighted average approach
            if p1 * p2 > F::zero() {
                // Bound the derivative to ensure monotonicity
                derivatives[i] =
                    F::min(a.abs(), min_slope * F::from_f64(2.0).unwrap()) * a.signum();
            }
        }

        // Steffen's method for endpoints - use one-sided formulas
        // For the first point
        let p1 = slopes[0];
        derivatives[0] = if p1.abs() <= F::epsilon() {
            F::zero()
        } else {
            p1
        };

        // For the last point
        let p2 = slopes[n - 2];
        derivatives[n - 1] = if p2.abs() <= F::epsilon() {
            F::zero()
        } else {
            p2
        };

        Ok(derivatives)
    }

    /// Compute modified Akima derivatives with monotonicity preservation
    ///
    /// This method combines Akima's approach for robustness to outliers with
    /// modifications to ensure monotonicity is preserved.
    ///
    /// Reference: Akima, H. (1970). "A new method of interpolation and smooth curve fitting based on local procedures".
    fn find_modified_akima_derivatives(
        x: &Array1<F>,
        y: &Array1<F>,
    ) -> InterpolateResult<Array1<F>> {
        let n = x.len();
        let mut derivatives = Array1::zeros(n);

        // Handle special cases with few points
        if n == 2 {
            // Linear case
            let slope = (y[1] - y[0]) / (x[1] - x[0]);
            derivatives[0] = slope;
            derivatives[1] = slope;
            return Ok(derivatives);
        } else if n == 3 {
            // Simple case with three points
            let slope1 = (y[1] - y[0]) / (x[1] - x[0]);
            let slope2 = (y[2] - y[1]) / (x[2] - x[1]);

            // Set derivatives using weighted arithmetic mean at the middle point
            derivatives[0] = slope1;
            derivatives[1] = (slope1 + slope2) / F::from_f64(2.0).unwrap();
            derivatives[2] = slope2;

            // Apply monotonicity filter
            if slope1 * slope2 <= F::zero() {
                derivatives[1] = F::zero();
            }

            return Ok(derivatives);
        }

        // Calculate segment slopes
        let mut slopes = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            slopes[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }

        // Calculate Akima weights
        // We'll use a small epsilon to avoid division by zero
        let epsilon = F::from_f64(1e-10).unwrap();

        // For interior points
        for i in 1..n - 1 {
            let s1 = if i > 1 {
                slopes[i - 2]
            } else {
                F::from_f64(2.0).unwrap() * slopes[0] - slopes[1]
            };
            let s2 = slopes[i - 1];
            let s3 = slopes[i];
            let s4 = if i < n - 2 {
                slopes[i + 1]
            } else {
                F::from_f64(2.0).unwrap() * slopes[n - 2] - slopes[n - 3]
            };

            // Calculate weights using Akima's formula
            let w1 = (s3 - s2).abs();
            let w2 = (s1 - s4).abs();

            // Apply modified Akima formula with monotonicity preservation
            if w1.abs() < epsilon && w2.abs() < epsilon {
                // Special case: equal slopes or very close to it
                // Use arithmetic mean
                derivatives[i] = (s2 + s3) / F::from_f64(2.0).unwrap();
            } else {
                // Regular case: use weighted mean
                derivatives[i] = (w1 * s3 + w2 * s2) / (w1 + w2);
            }

            // Check for monotonicity and apply filter if needed
            if s2 * s3 <= F::zero() {
                // Non-monotonic segment - set derivative to zero
                derivatives[i] = F::zero();
            } else {
                // Monotonic segment - bound derivative to prevent overshooting
                let three = F::from_f64(3.0).unwrap();
                let max_slope = three * F::min(s2.abs(), s3.abs());
                if derivatives[i].abs() > max_slope {
                    derivatives[i] = max_slope * derivatives[i].signum();
                }
            }
        }

        // Endpoints - use modified Akima formulation
        // For the first point
        let s1 = slopes[0];
        let s2 = if n > 2 { slopes[1] } else { s1 };
        if s1 * s2 <= F::zero() {
            derivatives[0] = F::zero(); // Non-monotonic
        } else {
            derivatives[0] = (F::from_f64(2.0).unwrap() * s1 * s2) / (s1 + s2);
            // Apply bounds
            let three = F::from_f64(3.0).unwrap();
            let max_slope = three * s1.abs();
            if derivatives[0].abs() > max_slope {
                derivatives[0] = max_slope * derivatives[0].signum();
            }
        }

        // For the last point
        let s1 = slopes[n - 2];
        let s2 = if n > 2 { slopes[n - 3] } else { s1 };
        if s1 * s2 <= F::zero() {
            derivatives[n - 1] = F::zero(); // Non-monotonic
        } else {
            derivatives[n - 1] = (F::from_f64(2.0).unwrap() * s1 * s2) / (s1 + s2);
            // Apply bounds
            let three = F::from_f64(3.0).unwrap();
            let max_slope = three * s1.abs();
            if derivatives[n - 1].abs() > max_slope {
                derivatives[n - 1] = max_slope * derivatives[n - 1].signum();
            }
        }

        Ok(derivatives)
    }
}

/// Convenience function for monotonic interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
/// * `method` - The monotonic interpolation method to use
/// * `extrapolate` - Whether to extrapolate beyond the data range
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::interp1d::monotonic::{monotonic_interpolate, MonotonicMethod};
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = monotonic_interpolate(
///     &x.view(),
///     &y.view(),
///     &x_new.view(),
///     MonotonicMethod::Steffen,
///     true
/// ).unwrap();
/// ```
pub fn monotonic_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
    method: MonotonicMethod,
    extrapolate: bool,
) -> InterpolateResult<Array1<F>> {
    let interp = MonotonicInterpolator::new(x, y, method, extrapolate)?;
    interp.evaluate_array(x_new)
}

/// Convenience function for Hyman filtered cubic spline interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
/// * `extrapolate` - Whether to extrapolate beyond the data range
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::interp1d::monotonic::hyman_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = hyman_interpolate(&x.view(), &y.view(), &x_new.view(), true).unwrap();
/// ```
pub fn hyman_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
    extrapolate: bool,
) -> InterpolateResult<Array1<F>> {
    monotonic_interpolate(x, y, x_new, MonotonicMethod::Hyman, extrapolate)
}

/// Convenience function for Steffen's monotonic interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
/// * `extrapolate` - Whether to extrapolate beyond the data range
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::interp1d::monotonic::steffen_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = steffen_interpolate(&x.view(), &y.view(), &x_new.view(), true).unwrap();
/// ```
pub fn steffen_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
    extrapolate: bool,
) -> InterpolateResult<Array1<F>> {
    monotonic_interpolate(x, y, x_new, MonotonicMethod::Steffen, extrapolate)
}

/// Convenience function for modified Akima monotonic interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
/// * `extrapolate` - Whether to extrapolate beyond the data range
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::interp1d::monotonic::modified_akima_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = modified_akima_interpolate(&x.view(), &y.view(), &x_new.view(), true).unwrap();
/// ```
pub fn modified_akima_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
    extrapolate: bool,
) -> InterpolateResult<Array1<F>> {
    monotonic_interpolate(x, y, x_new, MonotonicMethod::ModifiedAkima, extrapolate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_monotonic_methods_at_data_points() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test each method at the data points
        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            let interp = MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).unwrap();

            // All methods should exactly reproduce the data points
            assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
            assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
            assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
            assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);
        }
    }

    #[test]
    fn test_monotonicity_preservation() {
        // Test with data that has monotonic segments
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 0.5, 0.0, 0.5, 2.0];

        // Test each method
        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            let interp = MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).unwrap();

            // Check monotonicity preservation in the first segment (increasing)
            let y_0_25 = interp.evaluate(0.25).unwrap();
            let y_0_50 = interp.evaluate(0.50).unwrap();
            let y_0_75 = interp.evaluate(0.75).unwrap();
            assert!(y_0_25 <= y_0_50 && y_0_50 <= y_0_75);

            // Check monotonicity preservation in the second segment (decreasing)
            let y_1_25 = interp.evaluate(1.25).unwrap();
            let y_1_50 = interp.evaluate(1.50).unwrap();
            let y_1_75 = interp.evaluate(1.75).unwrap();
            assert!(y_1_25 >= y_1_50 && y_1_50 >= y_1_75);

            // Check third segment (decreasing)
            let y_2_25 = interp.evaluate(2.25).unwrap();
            let y_2_50 = interp.evaluate(2.50).unwrap();
            let y_2_75 = interp.evaluate(2.75).unwrap();
            assert!(y_2_25 >= y_2_50 && y_2_50 >= y_2_75);

            // Check fourth segment (increasing)
            let y_3_25 = interp.evaluate(3.25).unwrap();
            let y_3_50 = interp.evaluate(3.50).unwrap();
            let y_3_75 = interp.evaluate(3.75).unwrap();
            assert!(y_3_25 <= y_3_50 && y_3_50 <= y_3_75);

            // Check fifth segment (increasing)
            let y_4_25 = interp.evaluate(4.25).unwrap();
            let y_4_50 = interp.evaluate(4.50).unwrap();
            let y_4_75 = interp.evaluate(4.75).unwrap();
            assert!(y_4_25 <= y_4_50 && y_4_50 <= y_4_75);
        }
    }

    #[test]
    fn test_avoid_overshooting() {
        // Test with data that has a sharp turn
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0];

        // Test each method
        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            let interp = MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).unwrap();

            // Check values in each segment
            // First segment (flat)
            let y_0_5 = interp.evaluate(0.5).unwrap();
            assert_relative_eq!(y_0_5, 0.0, max_relative = 1e-8);

            // Second segment (flat)
            let y_1_5 = interp.evaluate(1.5).unwrap();
            assert_relative_eq!(y_1_5, 0.0, max_relative = 1e-8);

            // Third segment (increasing)
            let y_2_25 = interp.evaluate(2.25).unwrap();
            let y_2_50 = interp.evaluate(2.50).unwrap();
            let y_2_75 = interp.evaluate(2.75).unwrap();

            // Values should be monotonic
            assert!(y_2_25 <= y_2_50 && y_2_50 <= y_2_75);

            // Values should not become negative (undershoot)
            assert!(y_2_25 >= 0.0);
            assert!(y_2_50 >= 0.0);
            assert!(y_2_75 >= 0.0);

            // Values should not exceed 1.0 (overshoot)
            assert!(y_2_25 <= 1.0);
            assert!(y_2_50 <= 1.0);
            assert!(y_2_75 <= 1.0);

            // Fourth segment (flat)
            let y_3_5 = interp.evaluate(3.5).unwrap();
            assert_relative_eq!(y_3_5, 1.0, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_special_cases() {
        // Test with just two points
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];

        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            let interp = MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).unwrap();

            // Should exactly match linear interpolation
            assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
            assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.5);
            assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        }

        // Test with three points
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0, 0.0];

        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            let interp = MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).unwrap();

            // Should preserve monotonicity in each segment
            let y_0_25 = interp.evaluate(0.25).unwrap();
            let y_0_50 = interp.evaluate(0.50).unwrap();
            let y_0_75 = interp.evaluate(0.75).unwrap();
            assert!(y_0_25 <= y_0_50 && y_0_50 <= y_0_75);

            let y_1_25 = interp.evaluate(1.25).unwrap();
            let y_1_50 = interp.evaluate(1.50).unwrap();
            let y_1_75 = interp.evaluate(1.75).unwrap();
            assert!(y_1_25 >= y_1_50 && y_1_50 >= y_1_75);
        }
    }

    #[test]
    fn test_extrapolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test with extrapolation enabled
        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            let interp_extrap =
                MonotonicInterpolator::new(&x.view(), &y.view(), *method, true).unwrap();
            let _y_minus_1 = interp_extrap.evaluate(-1.0).unwrap();
            let _y_plus_4 = interp_extrap.evaluate(4.0).unwrap();

            // Test with extrapolation disabled
            let interp_no_extrap =
                MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).unwrap();
            assert!(interp_no_extrap.evaluate(-1.0).is_err());
            assert!(interp_no_extrap.evaluate(4.0).is_err());
        }
    }

    #[test]
    fn test_convenience_functions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let x_new = array![0.5, 1.5, 2.5];

        // Test each convenience function
        let y_hyman = hyman_interpolate(&x.view(), &y.view(), &x_new.view(), false).unwrap();
        let y_steffen = steffen_interpolate(&x.view(), &y.view(), &x_new.view(), false).unwrap();
        let y_akima =
            modified_akima_interpolate(&x.view(), &y.view(), &x_new.view(), false).unwrap();
        let y_generic = monotonic_interpolate(
            &x.view(),
            &y.view(),
            &x_new.view(),
            MonotonicMethod::Pchip,
            false,
        )
        .unwrap();

        // Each method should return a result with the correct length
        assert_eq!(y_hyman.len(), 3);
        assert_eq!(y_steffen.len(), 3);
        assert_eq!(y_akima.len(), 3);
        assert_eq!(y_generic.len(), 3);

        // For monotonic data, all interpolated values should be monotonic
        for result in [&y_hyman, &y_steffen, &y_akima, &y_generic] {
            assert!(result[0] > 0.0 && result[0] < 1.0);
            assert!(result[1] > 1.0 && result[1] < 4.0);
            assert!(result[2] > 4.0 && result[2] < 9.0);
        }
    }

    #[test]
    fn test_error_conditions() {
        // Test with different length arrays
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0];

        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            assert!(MonotonicInterpolator::new(&x.view(), &y.view(), *method, false).is_err());
        }

        // Test with unsorted x
        let x_unsorted = array![0.0, 2.0, 1.0, 3.0];
        let y_valid = array![0.0, 1.0, 4.0, 9.0];

        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            assert!(MonotonicInterpolator::new(
                &x_unsorted.view(),
                &y_valid.view(),
                *method,
                false
            )
            .is_err());
        }

        // Test with too few points
        let x_short = array![0.0];
        let y_short = array![0.0];

        for method in &[
            MonotonicMethod::Pchip,
            MonotonicMethod::Hyman,
            MonotonicMethod::Steffen,
            MonotonicMethod::ModifiedAkima,
        ] {
            assert!(
                MonotonicInterpolator::new(&x_short.view(), &y_short.view(), *method, false)
                    .is_err()
            );
        }
    }
}
