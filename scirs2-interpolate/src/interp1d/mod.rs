//! One-dimensional interpolation methods
//!
//! This module provides functionality for interpolating one-dimensional data.

mod basic_interp;
pub mod monotonic;
pub mod pchip;

// Re-export interpolation functions
pub use basic_interp::{cubic_interpolate, linear_interpolate, nearest_interpolate};
pub use monotonic::{
    hyman_interpolate, modified_akima_interpolate, monotonic_interpolate, steffen_interpolate,
    MonotonicInterpolator, MonotonicMethod,
};
pub use pchip::{pchip_interpolate, PchipInterpolator};

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Available interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    #[default]
    Linear,
    /// Cubic interpolation
    Cubic,
    /// PCHIP interpolation (monotonic)
    Pchip,
}

/// Options for extrapolation behavior
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExtrapolateMode {
    /// Return error when extrapolating
    #[default]
    Error,
    /// Extrapolate using the interpolation method
    Extrapolate,
    /// Use nearest valid value
    Nearest,
}

/// One-dimensional interpolation object
///
/// Provides a way to interpolate values at arbitrary points within a range
/// based on a set of known x and y values.
#[derive(Debug, Clone)]
pub struct Interp1d<F: Float> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Interpolation method
    method: InterpolationMethod,
    /// Extrapolation mode
    extrapolate: ExtrapolateMode,
}

impl<F: Float + FromPrimitive + Debug> Interp1d<F> {
    /// Create a new interpolation object
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `method` - The interpolation method to use
    /// * `extrapolate` - The extrapolation behavior
    ///
    /// # Returns
    ///
    /// A new `Interp1d` object
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use scirs2_interpolate::interp1d::{Interp1d, InterpolationMethod, ExtrapolateMode};
    ///
    /// let x = array![0.0f64, 1.0, 2.0, 3.0];
    /// let y = array![0.0f64, 1.0, 4.0, 9.0];
    ///
    /// // Create a linear interpolator
    /// let interp = Interp1d::new(
    ///     &x.view(), &y.view(),
    ///     InterpolationMethod::Linear,
    ///     ExtrapolateMode::Error
    /// ).unwrap();
    ///
    /// // Interpolate at x = 1.5
    /// let y_interp = interp.evaluate(1.5);
    /// assert!(y_interp.is_ok());
    /// assert!((y_interp.unwrap() - 2.5).abs() < 1e-10);
    /// ```
    pub fn new(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        method: InterpolationMethod,
        extrapolate: ExtrapolateMode,
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

        // For cubic interpolation, need at least 4 points
        if method == InterpolationMethod::Cubic && x.len() < 4 {
            return Err(InterpolateError::ValueError(
                "at least 4 points are required for cubic interpolation".to_string(),
            ));
        }

        Ok(Interp1d {
            x: x.to_owned(),
            y: y.to_owned(),
            method,
            extrapolate,
        })
    }

    /// Evaluate the interpolation at the given points
    ///
    /// # Arguments
    ///
    /// * `x_new` - The x coordinate at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y value at `x_new`
    pub fn evaluate(&self, x_new: F) -> InterpolateResult<F> {
        // Check if we're extrapolating
        let is_extrapolating = x_new < self.x[0] || x_new > self.x[self.x.len() - 1];

        if is_extrapolating {
            match self.extrapolate {
                ExtrapolateMode::Error => {
                    return Err(InterpolateError::DomainError(
                        "x_new is outside the interpolation range".to_string(),
                    ));
                }
                ExtrapolateMode::Nearest => {
                    if x_new < self.x[0] {
                        return Ok(self.y[0]);
                    } else {
                        return Ok(self.y[self.y.len() - 1]);
                    }
                }
                ExtrapolateMode::Extrapolate => {
                    // For extrapolation, we'll use linear extrapolation based on the edge segments
                    if x_new < self.x[0] {
                        // Use the first segment for extrapolation below the range
                        let x0 = self.x[0];
                        let x1 = self.x[1];
                        let y0 = self.y[0];
                        let y1 = self.y[1];

                        // Linear extrapolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                        let slope = (y1 - y0) / (x1 - x0);
                        return Ok(y0 + (x_new - x0) * slope);
                    } else {
                        // Use the last segment for extrapolation above the range
                        let n = self.x.len();
                        let x0 = self.x[n - 2];
                        let x1 = self.x[n - 1];
                        let y0 = self.y[n - 2];
                        let y1 = self.y[n - 1];

                        // Linear extrapolation formula: y = y1 + (x - x1) * (y1 - y0) / (x1 - x0)
                        let slope = (y1 - y0) / (x1 - x0);
                        return Ok(y1 + (x_new - x1) * slope);
                    }
                }
            }
        }

        // Find the index of the segment containing x_new
        let mut idx = 0;
        for i in 0..self.x.len() - 1 {
            if x_new >= self.x[i] && x_new <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        // Special case: x_new is exactly the last point
        if x_new == self.x[self.x.len() - 1] {
            return Ok(self.y[self.x.len() - 1]);
        }

        // Apply the selected interpolation method
        match self.method {
            InterpolationMethod::Nearest => {
                nearest_interp(&self.x.view(), &self.y.view(), idx, x_new)
            }
            InterpolationMethod::Linear => {
                linear_interp(&self.x.view(), &self.y.view(), idx, x_new)
            }
            InterpolationMethod::Cubic => cubic_interp(&self.x.view(), &self.y.view(), idx, x_new),
            InterpolationMethod::Pchip => {
                // For PCHIP, we'll create a PCHIP interpolator and use it
                // This is not the most efficient approach, but it keeps the interface consistent
                let extrapolate = self.extrapolate == ExtrapolateMode::Extrapolate
                    || self.extrapolate == ExtrapolateMode::Nearest;
                let pchip = PchipInterpolator::new(&self.x.view(), &self.y.view(), extrapolate)?;
                pchip.evaluate(x_new)
            }
        }
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
    pub fn evaluate_array(&self, x_new: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(x_new.len());
        for (i, &x) in x_new.iter().enumerate() {
            result[i] = self.evaluate(x)?;
        }
        Ok(result)
    }
}

/// Perform nearest neighbor interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `x_new` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
fn nearest_interp<F: Float>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    x_new: F,
) -> InterpolateResult<F> {
    // Find which of the two points is closer
    let dist_left = (x_new - x[idx]).abs();
    let dist_right = (x_new - x[idx + 1]).abs();

    if dist_left <= dist_right {
        Ok(y[idx])
    } else {
        Ok(y[idx + 1])
    }
}

/// Perform linear interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `x_new` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
fn linear_interp<F: Float>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    x_new: F,
) -> InterpolateResult<F> {
    let x0 = x[idx];
    let x1 = x[idx + 1];
    let y0 = y[idx];
    let y1 = y[idx + 1];

    // Avoid division by zero
    if x0 == x1 {
        return Ok(y0); // or y1, they should be the same
    }

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    Ok(y0 + (x_new - x0) * (y1 - y0) / (x1 - x0))
}

/// Perform cubic interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `x_new` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
fn cubic_interp<F: Float + FromPrimitive>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    x_new: F,
) -> InterpolateResult<F> {
    // We need 4 points for cubic interpolation
    // If we're near the edges, we need to adjust the indices
    let (i0, i1, i2, i3) = if idx == 0 {
        (0, 0, 1, 2)
    } else if idx == x.len() - 2 {
        (idx - 1, idx, idx + 1, idx + 1)
    } else {
        // Handles both idx == x.len() - 3 and idx > x.len() - 3 cases since they're identical
        (idx - 1, idx, idx + 1, idx + 2)
    };

    let _x0 = x[i0];
    let x1 = x[i1];
    let x2 = x[i2];
    let _x3 = x[i3];

    let y0 = y[i0];
    let y1 = y[i1];
    let y2 = y[i2];
    let y3 = y[i3];

    // Normalized position within the interval [x1, x2]
    let t = if x2 != x1 {
        (x_new - x1) / (x2 - x1)
    } else {
        F::zero()
    };

    // Calculate cubic interpolation using Catmull-Rom spline
    // p(t) = 0.5 * ((2*p1) +
    //               (-p0 + p2) * t +
    //               (2*p0 - 5*p1 + 4*p2 - p3) * t^2 +
    //               (-p0 + 3*p1 - 3*p2 + p3) * t^3)

    let two = F::from_f64(2.0).unwrap();
    let three = F::from_f64(3.0).unwrap();
    let four = F::from_f64(4.0).unwrap();
    let five = F::from_f64(5.0).unwrap();
    let half = F::from_f64(0.5).unwrap();

    let t2 = t * t;
    let t3 = t2 * t;

    let c0 = two * y1;
    let c1 = -y0 + y2;
    let c2 = two * y0 - five * y1 + four * y2 - y3;
    let c3 = -y0 + three * y1 - three * y2 + y3;

    let result = half * (c0 + c1 * t + c2 * t2 + c3 * t3);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_nearest_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Nearest,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // Test points between data points
        assert_relative_eq!(interp.evaluate(0.4).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(0.6).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(1.4).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(1.6).unwrap(), 4.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // Test points between data points
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.5);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 2.5);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 6.5);
    }

    #[test]
    fn test_cubic_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // For this particular dataset (a quadratic y = x²),
        // cubic interpolation might not reproduce it exactly due to the specific spline algorithm
        // so we use wider tolerances
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.25, epsilon = 0.1);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 2.25, epsilon = 0.1);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 6.25, epsilon = 1.0);
    }

    #[test]
    fn test_pchip_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Pchip,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // For this monotonically increasing dataset,
        // PCHIP should preserve monotonicity
        let y_05 = interp.evaluate(0.5).unwrap();
        let y_15 = interp.evaluate(1.5).unwrap();
        let y_25 = interp.evaluate(2.5).unwrap();

        assert!(y_05 > 0.0 && y_05 < 1.0);
        assert!(y_15 > 1.0 && y_15 < 4.0);
        assert!(y_25 > 4.0 && y_25 < 9.0);
    }

    #[test]
    fn test_extrapolation_modes() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test error mode
        let interp_error = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        )
        .unwrap();

        assert!(interp_error.evaluate(-1.0).is_err());
        assert!(interp_error.evaluate(4.0).is_err());

        // Test nearest mode
        let interp_nearest = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Nearest,
        )
        .unwrap();

        assert_relative_eq!(interp_nearest.evaluate(-1.0).unwrap(), 0.0);
        assert_relative_eq!(interp_nearest.evaluate(4.0).unwrap(), 9.0);

        // Test extrapolate mode
        let interp_extrapolate = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // For this data, the linear extrapolation is based on the slope of the segments
        // For x=-1.0, we use the first segment (0,0) - (1,1) which has slope 1
        assert_relative_eq!(interp_extrapolate.evaluate(-1.0).unwrap(), -1.0);

        // For x=4.0, we use the last segment (2,4) - (3,9) which has slope 5
        // So the result is 9 + (4-3)*5 = 9 + 5 = 14
        assert_relative_eq!(interp_extrapolate.evaluate(4.0).unwrap(), 14.0);
    }

    #[test]
    fn test_convenience_functions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let x_new = array![0.5, 1.5, 2.5];

        // Test nearest interpolation
        let y_nearest = nearest_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
        // Point 0.5 is exactly halfway between x[0]=0.0 and x[1]=1.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[0], 0.0);
        // Point 1.5 is exactly halfway between x[1]=1.0 and x[2]=2.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[1], 1.0);
        // Point 2.5 is exactly halfway between x[2]=2.0 and x[3]=3.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[2], 4.0);

        // Test linear interpolation
        let y_linear = linear_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
        assert_relative_eq!(y_linear[0], 0.5);
        assert_relative_eq!(y_linear[1], 2.5);
        assert_relative_eq!(y_linear[2], 6.5);

        // Test cubic interpolation
        let y_cubic = cubic_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
        // Allow a wider tolerance for cubic interpolation since it depends on the specific spline implementation
        assert!((y_cubic[0] - 0.25).abs() < 0.15);
        assert!((y_cubic[1] - 2.25).abs() < 0.15);
        // For point 2.5, allow an even wider tolerance
        assert!((y_cubic[2] - 6.25).abs() < 1.0);

        // Test PCHIP interpolation
        let y_pchip = pchip_interpolate(&x.view(), &y.view(), &x_new.view(), false).unwrap();
        // For monotonically increasing data, PCHIP should preserve monotonicity
        assert!(y_pchip[0] > 0.0 && y_pchip[0] < 1.0);
        assert!(y_pchip[1] > 1.0 && y_pchip[1] < 4.0);
        assert!(y_pchip[2] > 4.0 && y_pchip[2] < 9.0);
    }

    #[test]
    fn test_error_conditions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0];

        // Test different lengths
        let result = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        );
        assert!(result.is_err());

        // Test unsorted x
        let x_unsorted = array![0.0, 2.0, 1.0, 3.0];
        let y_valid = array![0.0, 1.0, 4.0, 9.0];

        let result = Interp1d::new(
            &x_unsorted.view(),
            &y_valid.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        );
        assert!(result.is_err());

        // Test too few points for cubic
        let x_short = array![0.0, 1.0];
        let y_short = array![0.0, 1.0];

        let result = Interp1d::new(
            &x_short.view(),
            &y_short.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        );
        assert!(result.is_err());
    }
}
