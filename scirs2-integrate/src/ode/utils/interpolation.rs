//! Interpolation for dense output.
//!
//! This module provides interpolation functions for generating
//! dense output from ODE solvers, allowing evaluation at arbitrary points
//! within the integration range without recomputing the entire solution.

use crate::IntegrateFloat;
use ndarray::Array1;
use num_traits::Float;
use std::fmt::Debug;

/// Continuous output method for interpolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContinuousOutputMethod {
    /// Simple linear interpolation between points
    Linear,
    /// Cubic Hermite interpolation using function values and derivatives
    #[default]
    CubicHermite,
    /// Specialized high-order interpolation for specific ODE methods
    MethodSpecific,
}

/// Find the index in a sorted array where the given value would be inserted
///
/// # Arguments
///
/// * `sorted_array` - Sorted array
/// * `value` - Value to look for
///
/// # Returns
///
/// The index where value would be inserted to keep the array sorted
pub fn find_index<F: Float>(sorted_array: &[F], value: F) -> usize {
    // Binary search
    let mut left = 0;
    let mut right = sorted_array.len();

    while left < right {
        let mid = (left + right) / 2;
        if sorted_array[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    left
}

/// Linear interpolation
///
/// # Arguments
///
/// * `x` - x-coordinates
/// * `y` - y-values at each x-coordinate
/// * `x_new` - Point where to interpolate
///
/// # Returns
///
/// Linearly interpolated value at x_new
pub fn linear_interpolation<F: IntegrateFloat>(x: &[F], y: &[Array1<F>], x_new: F) -> Array1<F> {
    let i = find_index(x, x_new);

    if i == 0 {
        return y[0].clone();
    }

    if i >= x.len() {
        return y[x.len() - 1].clone();
    }

    let x0 = x[i - 1];
    let x1 = x[i];
    let y0 = &y[i - 1];
    let y1 = &y[i];

    // Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    let t = (x_new - x0) / (x1 - x0);

    let mut result = y0.clone();
    for (r, (a, b)) in result.iter_mut().zip(y0.iter().zip(y1.iter())) {
        *r = *a + t * (*b - *a);
    }

    result
}

/// Cubic Hermite interpolation
///
/// # Arguments
///
/// * `x` - x-coordinates
/// * `y` - y-values at each x-coordinate
/// * `dy` - Derivatives at each x-coordinate
/// * `x_new` - Point where to interpolate
///
/// # Returns
///
/// Cubic interpolated value at x_new
pub fn cubic_hermite_interpolation<F: IntegrateFloat>(
    x: &[F],
    y: &[Array1<F>],
    dy: &[Array1<F>],
    x_new: F,
) -> Array1<F> {
    let i = find_index(x, x_new);

    if i == 0 {
        return y[0].clone();
    }

    if i >= x.len() {
        return y[x.len() - 1].clone();
    }

    let x0 = x[i - 1];
    let x1 = x[i];
    let y0 = &y[i - 1];
    let y1 = &y[i];
    let dy0 = &dy[i - 1];
    let dy1 = &dy[i];

    // Normalized time coordinate
    let h = x1 - x0;
    let t = (x_new - x0) / h;

    // Hermite basis functions
    let h00 =
        F::from_f64(2.0).unwrap() * t.powi(3) - F::from_f64(3.0).unwrap() * t.powi(2) + F::one();
    let h10 = t.powi(3) - F::from_f64(2.0).unwrap() * t.powi(2) + t;
    let h01 = F::from_f64(-2.0).unwrap() * t.powi(3) + F::from_f64(3.0).unwrap() * t.powi(2);
    let h11 = t.powi(3) - t.powi(2);

    // Compute interpolant: p(t) = h00*y0 + h10*h*dy0 + h01*y1 + h11*h*dy1
    let mut result = Array1::zeros(y0.dim());

    for i in 0..y0.len() {
        result[i] = h00 * y0[i] + h10 * h * dy0[i] + h01 * y1[i] + h11 * h * dy1[i];
    }

    result
}
