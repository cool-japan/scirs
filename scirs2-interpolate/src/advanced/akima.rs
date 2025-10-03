//! Akima spline interpolation
//!
//! This module provides Akima spline interpolation, which is designed
//! to be more robust to outliers than cubic splines.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Akima spline interpolation object
///
/// Represents a piecewise cubic polynomial that passes through all given points
/// with continuous first derivatives, but adapts better to local changes.
#[derive(Debug, Clone)]
pub struct AkimaSpline<F: Float + FromPrimitive> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Coefficients for cubic polynomials (n-1 segments, 4 coefficients each)
    /// Each row represents [a, b, c, d] for a segment
    /// y(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    coeffs: Array2<F>,
}

impl<F: Float + FromPrimitive + Debug> AkimaSpline<F> {
    /// Create a new Akima spline
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `AkimaSpline` object
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::advanced::akima::AkimaSpline;
    ///
    /// let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    /// let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];
    ///
    /// let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();
    ///
    /// // Interpolate at x = 2.5
    /// let y_interp = spline.evaluate(2.5).unwrap();
    /// println!("Interpolated value at x=2.5: {}", y_interp);
    /// ```
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 5 {
            return Err(InterpolateError::invalid_input(
                "at least 5 points are required for Akima spline".to_string(),
            ));
        }

        // Check that x is strictly increasing
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x array must be strictly increasing".to_string(),
                ));
            }
        }

        // Create Akima spline
        let n = x.len();
        let mut slopes = Array1::zeros(n + 3);

        // Calculate the slopes
        for i in 0..n - 1 {
            let m_i = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
            slopes[i + 2] = m_i;
        }

        // Set up the artificial end slopes
        // Akima's original method uses a special formula for the endpoints
        slopes[0] = F::from_f64(3.0).unwrap() * slopes[2] - F::from_f64(2.0).unwrap() * slopes[3];
        slopes[1] = F::from_f64(2.0).unwrap() * slopes[2] - slopes[3];
        slopes[n + 1] = F::from_f64(2.0).unwrap() * slopes[n] - slopes[n - 1];
        slopes[n + 2] =
            F::from_f64(3.0).unwrap() * slopes[n] - F::from_f64(2.0).unwrap() * slopes[n - 1];

        // Calculate the derivatives at each point using Akima's formula
        let mut derivatives = Array1::zeros(n);
        for i in 0..n {
            let w1 = (slopes[i + 3] - slopes[i + 2]).abs();
            let w2 = (slopes[i + 1] - slopes[i]).abs();

            if w1 + w2 == F::zero() {
                // If both weights are zero, use the average of the slopes
                derivatives[i] = (slopes[i + 1] + slopes[i + 2]) / F::from_f64(2.0).unwrap();
            } else {
                // Otherwise, use weighted average
                derivatives[i] = (w1 * slopes[i + 1] + w2 * slopes[i + 2]) / (w1 + w2);
            }
        }

        // Calculate the polynomial coefficients for each segment
        let mut coeffs = Array2::zeros((n - 1, 4));
        for i in 0..n - 1 {
            let dx = x[i + 1] - x[i];
            let dy = y[i + 1] - y[i];

            let a = y[i];
            let b = derivatives[i];
            let c = (F::from_f64(3.0).unwrap() * dy / dx
                - F::from_f64(2.0).unwrap() * derivatives[i]
                - derivatives[i + 1])
                / dx;
            let d = (derivatives[i] + derivatives[i + 1] - F::from_f64(2.0).unwrap() * dy / dx)
                / (dx * dx);

            coeffs[[i, 0]] = a;
            coeffs[[i, 1]] = b;
            coeffs[[i, 2]] = c;
            coeffs[[i, 3]] = d;
        }

        Ok(Self {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Evaluate the spline at a given point
    ///
    /// # Arguments
    ///
    /// * `xnew` - The point at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The interpolated value at `xnew`
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        // Check if xnew is within the interpolation range
        if xnew < self.x[0] || xnew > self.x[self.x.len() - 1] {
            return Err(InterpolateError::OutOfBounds(
                "xnew is outside the interpolation range".to_string(),
            ));
        }

        // Find the index of the segment containing xnew
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if xnew >= self.x[i] && xnew <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: xnew is exactly the last point
        if xnew == self.x[self.x.len() - 1] {
            return Ok(self.y[self.y.len() - 1]);
        }

        // Evaluate the polynomial at xnew
        let dx = xnew - self.x[idx];
        let a = self.coeffs[[idx, 0]];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];

        let result = a + b * dx + c * dx * dx + d * dx * dx * dx;
        Ok(result)
    }

    /// Evaluate the spline at multiple points
    ///
    /// # Arguments
    ///
    /// * `xnew` - The points at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// The interpolated values at `xnew`
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(xnew.len());
        for (i, &x) in xnew.iter().enumerate() {
            result[i] = self.evaluate(x)?;
        }
        Ok(result)
    }

    /// Compute the derivative of the spline at a given point
    ///
    /// # Arguments
    ///
    /// * `xnew` - The point at which to evaluate the derivative
    ///
    /// # Returns
    ///
    /// The derivative of the spline at `xnew`
    pub fn derivative(&self, xnew: F) -> InterpolateResult<F> {
        // Check if xnew is within the interpolation range
        if xnew < self.x[0] || xnew > self.x[self.x.len() - 1] {
            return Err(InterpolateError::OutOfBounds(
                "xnew is outside the interpolation range".to_string(),
            ));
        }

        // Find the index of the segment containing xnew
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if xnew >= self.x[i] && xnew <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: xnew is exactly the last point
        if xnew == self.x[self.x.len() - 1] {
            // Use the derivative at the last internal point
            idx = self.x.len() - 2;
            let dx = self.x[idx + 1] - self.x[idx];
            let b = self.coeffs[[idx, 1]];
            let c = self.coeffs[[idx, 2]];
            let d = self.coeffs[[idx, 3]];
            return Ok(b
                + F::from_f64(2.0).unwrap() * c * dx
                + F::from_f64(3.0).unwrap() * d * dx * dx);
        }

        // Evaluate the derivative at xnew
        let dx = xnew - self.x[idx];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];

        let result =
            b + F::from_f64(2.0).unwrap() * c * dx + F::from_f64(3.0).unwrap() * d * dx * dx;
        Ok(result)
    }
}

/// Create an Akima spline interpolator
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
///
/// # Returns
///
/// A new `AkimaSpline` object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::advanced::akima::make_akima_spline;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];
///
/// let spline = make_akima_spline(&x.view(), &y.view()).unwrap();
///
/// // Interpolate at x = 2.5
/// let y_interp = spline.evaluate(2.5).unwrap();
/// println!("Interpolated value at x=2.5: {}", y_interp);
/// ```
#[allow(dead_code)]
pub fn make_akima_spline<F: crate::traits::InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<AkimaSpline<F>> {
    AkimaSpline::new(x, y)
}

/// Convenience function for Akima interpolation
#[allow(dead_code)]
pub fn akima_interpolate<F: crate::traits::InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    xnew: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>> {
    let spline = AkimaSpline::new(x, y)?;
    spline.evaluate_array(xnew)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_akima_spline() {
        // Data with an outlier
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.0, 1.0, 4.0, 20.0, 16.0, 25.0]; // Point at x=3 is an outlier

        let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();

        // Test at the knot points
        assert_abs_diff_eq!(spline.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(3.0).unwrap(), 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(4.0).unwrap(), 16.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spline.evaluate(5.0).unwrap(), 25.0, epsilon = 1e-10);

        // Test at some intermediate points
        // Akima should handle the outlier more gracefully than a cubic spline
        let y_2_5 = spline.evaluate(2.5).unwrap();
        let y_3_5 = spline.evaluate(3.5).unwrap();

        // Ensure we're interpolating and not just jumping
        assert!(y_2_5 > 4.0);
        assert!(y_2_5 < 20.0);
        assert!(y_3_5 < 20.0);
        assert!(y_3_5 > 16.0);

        // Test error for point outside range
        assert!(spline.evaluate(-1.0).is_err());
        assert!(spline.evaluate(6.0).is_err());
    }

    #[test]
    fn test_akima_spline_derivative() {
        // Simple quadratic data: y = x^2
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();

        // Test derivatives at some points
        // For x^2, the derivative is 2x
        let d_1 = spline.derivative(1.0).unwrap();
        let d_2 = spline.derivative(2.0).unwrap();
        let d_3 = spline.derivative(3.0).unwrap();

        // Allow some error but should be close to the exact values
        assert!((d_1 - 2.0).abs() < 0.3);
        assert!((d_2 - 4.0).abs() < 0.3);
        assert!((d_3 - 6.0).abs() < 0.3);
    }

    #[test]
    fn test_make_akima_spline() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let spline = make_akima_spline(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(spline.evaluate(2.5).unwrap(), 6.25, epsilon = 0.5);
    }
}
