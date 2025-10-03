//! PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation
//!
//! This module provides an implementation of the PCHIP algorithm, which produces
//! shape-preserving interpolants that maintain monotonicity in the data.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator
///
/// This interpolator preserves monotonicity in the interpolation data and does
/// not overshoot if the data is not smooth. The first derivatives are guaranteed
/// to be continuous, but the second derivatives may jump at the knot points.
///
/// The algorithm determines the derivatives at points x_k by using the PCHIP algorithm
/// from Fritsch and Butland (1984).
///
/// # References
///
/// 1. F. N. Fritsch and J. Butland, "A method for constructing local monotone
///    piecewise cubic interpolants", SIAM J. Sci. Comput., 5(2), 300-304 (1984).
/// 2. C. Moler, "Numerical Computing with Matlab", 2004.
#[derive(Debug, Clone)]
pub struct PchipInterpolator<F: Float> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Derivatives at points
    derivatives: Array1<F>,
    /// Extrapolation mode
    extrapolate: bool,
}

impl<F: Float + FromPrimitive + Debug> PchipInterpolator<F> {
    /// Create a new PCHIP interpolator
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `extrapolate` - Whether to extrapolate beyond the data range
    ///
    /// # Returns
    ///
    /// A new `PchipInterpolator` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `x` and `y` have different lengths
    /// - `x` is not sorted in ascending order
    /// - There are fewer than 2 points
    /// - `y` contains complex values (not supported)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::interp1d::pchip::PchipInterpolator;
    ///
    /// let x = array![0.0f64, 1.0, 2.0, 3.0];
    /// let y = array![0.0f64, 1.0, 4.0, 9.0];
    ///
    /// // Create a PCHIP interpolator
    /// let interp = PchipInterpolator::new(&x.view(), &y.view(), true).unwrap();
    ///
    /// // Interpolate at x = 1.5
    /// let y_interp = interp.evaluate(1.5);
    /// ```
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>, extrapolate: bool) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                x.len(),
                "PCHIP interpolation",
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

        // Clone input arrays
        let x_arr = x.to_owned();
        let y_arr = y.to_owned();

        // Compute derivatives at points
        let derivatives = Self::find_derivatives(&x_arr, &y_arr)?;

        Ok(PchipInterpolator {
            x: x_arr,
            y: y_arr,
            derivatives,
            extrapolate,
        })
    }

    /// Evaluate the interpolation at the given point
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinate at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y value at `xnew`
    ///
    /// # Errors
    ///
    /// Returns an error if `xnew` is outside the interpolation range and
    /// extrapolation is disabled.
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        // Check if we're extrapolating
        let is_extrapolating = xnew < self.x[0] || xnew > self.x[self.x.len() - 1];
        if is_extrapolating && !self.extrapolate {
            return Err(InterpolateError::OutOfBounds(
                "xnew is outside the interpolation range".to_string(),
            ));
        }

        // Find index of segment containing xnew
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if xnew >= self.x[i] && xnew <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Handle extrapolation case
        if is_extrapolating {
            if xnew < self.x[0] {
                // Extrapolate below the data range using the first segment
                idx = 0;
            } else {
                // Extrapolate above the data range using the last segment
                idx = self.x.len() - 2;
            }
        }

        // Special case: xnew is exactly at a knot point
        for i in 0..self.x.len() {
            if xnew == self.x[i] {
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
        let t = (xnew - x1) / h;

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
    /// * `xnew` - The x coordinates at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y values at `xnew`
    ///
    /// # Errors
    ///
    /// Returns an error if any point in `xnew` is outside the interpolation range
    /// and extrapolation is disabled.
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());
        for (i, &x) in xnew.iter().enumerate() {
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
        // three is not used in this function
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
        // No constants needed for this function
        (t * t * t) - (t * t)
    }

    /// Helper for edge case derivative estimation
    ///
    /// # Arguments
    ///
    /// * `h0` - Spacing: x[1] - x[0] for the first point, or x[n-1] - x[n-2] for the last point
    /// * `h1` - Spacing: x[2] - x[1] for the first point, or x[n] - x[n-1] for the last point
    /// * `m0` - Slope of the first segment
    /// * `m1` - Slope of the second segment
    ///
    /// # Returns
    ///
    /// The estimated derivative at the endpoint
    fn edge_case(h0: F, h1: F, m0: F, m1: F) -> F {
        // One-sided three-point estimate for the derivative
        let two = F::from_f64(2.0).unwrap();
        let three = F::from_f64(3.0).unwrap();

        let d = ((two * h0 + h1) * m0 - h0 * m1) / (h0 + h1);

        // Try to preserve shape
        let sign_d = if d >= F::zero() { F::one() } else { -F::one() };
        let sign_m0 = if m0 >= F::zero() { F::one() } else { -F::one() };
        let sign_m1 = if m1 >= F::zero() { F::one() } else { -F::one() };

        // If the signs are different or abs(d) > 3*abs(m0), adjust d
        if sign_d != sign_m0 {
            F::zero()
        } else if (sign_m0 != sign_m1) && (d.abs() > three * m0.abs()) {
            three * m0
        } else {
            d
        }
    }

    /// Compute derivatives at points for PCHIP interpolation
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinates
    /// * `y` - Y coordinates
    ///
    /// # Returns
    ///
    /// Array of derivatives, one for each point in x
    fn find_derivatives(x: &Array1<F>, y: &Array1<F>) -> InterpolateResult<Array1<F>> {
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
        derivatives[0] = Self::edge_case(h[0], h[1], slopes[0], slopes[1]);

        // For the last point
        derivatives[n - 1] = Self::edge_case(h[n - 2], h[n - 3], slopes[n - 2], slopes[n - 3]);

        Ok(derivatives)
    }
}

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation convenience function
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `xnew` - The points at which to interpolate
/// * `extrapolate` - Whether to extrapolate beyond the data range
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::pchip_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let xnew = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = pchip_interpolate(&x.view(), &y.view(), &xnew.view(), true).unwrap();
/// ```
#[allow(dead_code)]
pub fn pchip_interpolate<F: crate::traits::InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    xnew: &ArrayView1<F>,
    extrapolate: bool,
) -> InterpolateResult<Array1<F>> {
    let interp = PchipInterpolator::new(x, y, extrapolate)?;
    interp.evaluate_array(xnew)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pchip_interpolation_basic() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = PchipInterpolator::new(&x.view(), &y.view(), false).unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // Test points between data points - PCHIP should preserve shape
        let y_interp_0_5 = interp.evaluate(0.5).unwrap();
        let y_interp_1_5 = interp.evaluate(1.5).unwrap();
        let y_interp_2_5 = interp.evaluate(2.5).unwrap();

        // For this monotonically increasing data, PCHIP should preserve monotonicity
        assert!(y_interp_0_5 > 0.0 && y_interp_0_5 < 1.0);
        assert!(y_interp_1_5 > 1.0 && y_interp_1_5 < 4.0);
        assert!(y_interp_2_5 > 4.0 && y_interp_2_5 < 9.0);
    }

    #[test]
    fn test_pchip_monotonicity_preservation() {
        // Test with data that has monotonic segments
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 0.5, 0.0, 0.5, 2.0];

        let interp = PchipInterpolator::new(&x.view(), &y.view(), false).unwrap();

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

        // Check monotonicity preservation in the last segment (increasing)
        let y_4_25 = interp.evaluate(4.25).unwrap();
        let y_4_50 = interp.evaluate(4.50).unwrap();
        let y_4_75 = interp.evaluate(4.75).unwrap();
        assert!(y_4_25 <= y_4_50 && y_4_50 <= y_4_75);
    }

    #[test]
    fn test_pchip_extrapolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test with extrapolation enabled
        let interp_extrap = PchipInterpolator::new(&x.view(), &y.view(), true).unwrap();
        let _y_minus_1 = interp_extrap.evaluate(-1.0).unwrap();
        let _y_plus_4 = interp_extrap.evaluate(4.0).unwrap();

        // Test with extrapolation disabled
        let interp_no_extrap = PchipInterpolator::new(&x.view(), &y.view(), false).unwrap();
        assert!(interp_no_extrap.evaluate(-1.0).is_err());
        assert!(interp_no_extrap.evaluate(4.0).is_err());
    }

    #[test]
    fn test_pchip_interpolate_function() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let xnew = array![0.5, 1.5, 2.5];

        let y_interp = pchip_interpolate(&x.view(), &y.view(), &xnew.view(), false).unwrap();

        // Test that the function returns the expected number of points
        assert_eq!(y_interp.len(), 3);

        // For this monotonically increasing data, all interpolated values should be monotonically increasing
        assert!(y_interp[0] > 0.0 && y_interp[0] < 1.0);
        assert!(y_interp[1] > 1.0 && y_interp[1] < 4.0);
        assert!(y_interp[2] > 4.0 && y_interp[2] < 9.0);
    }

    #[test]
    fn test_pchip_error_conditions() {
        // Test with different length arrays
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0];
        assert!(PchipInterpolator::new(&x.view(), &y.view(), false).is_err());

        // Test with unsorted x
        let x_unsorted = array![0.0, 2.0, 1.0, 3.0];
        let y_valid = array![0.0, 1.0, 4.0, 9.0];
        assert!(PchipInterpolator::new(&x_unsorted.view(), &y_valid.view(), false).is_err());

        // Test with too few points
        let x_short = array![0.0];
        let y_short = array![0.0];
        assert!(PchipInterpolator::new(&x_short.view(), &y_short.view(), false).is_err());
    }
}
