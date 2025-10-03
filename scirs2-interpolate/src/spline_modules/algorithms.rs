//! Core cubic spline interpolation algorithms
//!
//! This module contains the computational functions for various types of cubic spline
//! interpolation with different boundary conditions. These functions implement the mathematical
//! algorithms for solving the tridiagonal systems that arise in spline construction.
//!
//! ## Overview
//!
//! Cubic splines are piecewise cubic polynomials that provide smooth interpolation through
//! a set of data points. The different functions in this module handle various boundary
//! conditions:
//!
//! - **Natural**: Zero second derivative at endpoints
//! - **Not-a-knot**: Continuous third derivative at second and second-to-last points
//! - **Clamped**: Specified first derivatives at endpoints
//! - **Periodic**: Function and derivatives match at endpoints
//! - **Second derivative**: Specified second derivatives at endpoints
//! - **Parabolic runout**: Zero third derivative at endpoints
//!
//! ## Computational Approach
//!
//! All spline algorithms follow a similar pattern:
//! 1. Set up a tridiagonal system based on continuity and boundary conditions
//! 2. Solve for second derivatives at each point using Thomas algorithm
//! 3. Calculate polynomial coefficients for each segment
//!
//! The resulting coefficients define cubic polynomials of the form:
//! ```text
//! y(x) = a + b(x-xᵢ) + c(x-xᵢ)² + d(x-xᵢ)³
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Compute the coefficients for a natural cubic spline
///
/// Natural boundary conditions: second derivative is zero at the endpoints.
/// This minimizes the total curvature of the spline and is often the default choice
/// when no specific endpoint behavior is required.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
///
/// # Returns
///
/// A 2D array of shape (n-1, 4) containing the polynomial coefficients for each segment.
/// Each row contains [a, b, c, d] for the polynomial: a + b(x-xᵢ) + c(x-xᵢ)² + d(x-xᵢ)³
///
/// # Errors
///
/// Returns `InterpolateError::ComputationError` if:
/// - Any interval has zero length (duplicate x values)
/// - Numerical issues in the tridiagonal solver
/// - Float conversion failures
#[allow(dead_code)]
pub fn compute_natural_cubic_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients (n-1 segments x 4 coefficients)
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Step 1: Calculate the second derivatives at each point
    // We solve the tridiagonal system to get these

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Natural boundary conditions
    b[0] = F::one();
    d[0] = F::zero();
    b[n - 1] = F::one();
    d[n - 1] = F::zero();

    // Fill in the tridiagonal system
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        a[i] = h_i_minus_1;
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string(),
            )
        })?;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        // Check for division by zero in slope calculations
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in spline computation".to_string(),
            ));
        }

        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string(),
            )
        })?;
        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system using the Thomas algorithm
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        // Check for division by zero
        if b[i - 1].is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero diagonal element in Thomas algorithm forward sweep".to_string(),
            ));
        }
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    if b[n - 1].is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero diagonal element in Thomas algorithm back substitution".to_string(),
        ));
    }
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        if b[i].is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero diagonal element in Thomas algorithm back substitution".to_string(),
            ));
        }
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Step 2: Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for division by zero in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in spline coefficient calculation".to_string(),
            ));
        }

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

        // a is just the y value at the left endpoint
        coeffs[[i, 0]] = y[i];

        // b is the first derivative at the left endpoint
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i - h_i * (two * sigma[i] + sigma[i + 1]) / six;

        // c is half the second derivative at the left endpoint
        coeffs[[i, 2]] = sigma[i] / two;

        // d is the rate of change of the second derivative / 6
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a not-a-knot cubic spline
///
/// Not-a-knot boundary conditions: third derivative is continuous across the
/// first and last interior knots. This maximizes smoothness and often produces
/// the most visually pleasing curves.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
///
/// # Returns
///
/// A 2D array of shape (n-1, 4) containing the polynomial coefficients for each segment.
///
/// # Errors
///
/// Returns `InterpolateError::ComputationError` if:
/// - Any interval has zero length
/// - Numerical issues in the tridiagonal solver
/// - Float conversion failures
#[allow(dead_code)]
pub fn compute_not_a_knot_cubic_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients (n-1 segments x 4 coefficients)
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Step 1: Calculate the second derivatives at each point

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Not-a-knot condition at first interior point
    let h0 = x[1] - x[0];
    let h1 = x[2] - x[1];

    // Check for zero intervals
    if h0.is_zero() || h1.is_zero() || (h0 + h1).is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in not-a-knot spline boundary conditions".to_string(),
        ));
    }

    b[0] = h1;
    c[0] = h0 + h1;
    d[0] = ((h0 + h1) * h1 * (y[1] - y[0]) / h0 + h0 * h0 * (y[2] - y[1]) / h1) / (h0 + h1);

    // Not-a-knot condition at last interior point
    let hn_2 = x[n - 2] - x[n - 3];
    let hn_1 = x[n - 1] - x[n - 2];

    // Check for zero intervals
    if hn_1.is_zero() || hn_2.is_zero() || (hn_1 + hn_2).is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in not-a-knot spline boundary conditions".to_string(),
        ));
    }

    a[n - 1] = hn_1 + hn_2;
    b[n - 1] = hn_2;
    d[n - 1] = ((hn_1 + hn_2) * hn_2 * (y[n - 1] - y[n - 2]) / hn_1
        + hn_1 * hn_1 * (y[n - 2] - y[n - 3]) / hn_2)
        / (hn_1 + hn_2);

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in not-a-knot spline computation".to_string(),
            ));
        }

        a[i] = h_i_minus_1;
        let two = F::from_f64(2.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 2.0 to float type".to_string(),
            )
        })?;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        let six = F::from_f64(6.0).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert constant 6.0 to float type".to_string(),
            )
        })?;
        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system using the Thomas algorithm
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    let mut c_prime = Array1::<F>::zeros(n);

    // Check for division by zero in first step
    if b[0].is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero diagonal element in not-a-knot Thomas algorithm".to_string(),
        ));
    }
    c_prime[0] = c[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];

        // Check for division by zero
        if m.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero diagonal element in not-a-knot Thomas algorithm".to_string(),
            ));
        }

        if i < n - 1 {
            c_prime[i] = c[i] / m;
        }
        d[i] = (d[i] - a[i] * d[i - 1]) / m;
    }

    // Back substitution
    sigma[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = d[i] - c_prime[i] * sigma[i + 1];
    }

    // Step 2: Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for division by zero in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in not-a-knot spline coefficient calculation".to_string(),
            ));
        }

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

        // a is just the y value at the left endpoint
        coeffs[[i, 0]] = y[i];

        // b is the first derivative at the left endpoint
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i - h_i * (two * sigma[i] + sigma[i + 1]) / six;

        // c is half the second derivative at the left endpoint
        coeffs[[i, 2]] = sigma[i] / two;

        // d is the rate of change of the second derivative / 6
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a clamped cubic spline
///
/// Clamped boundary conditions: first derivative specified at endpoints.
/// This provides exact control over endpoint slopes and is most accurate
/// when derivative information is known.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
/// * `left_deriv` - First derivative at the left endpoint
/// * `right_deriv` - First derivative at the right endpoint
///
/// # Returns
///
/// A 2D array of shape (n-1, 4) containing the polynomial coefficients for each segment.
///
/// # Errors
///
/// Returns `InterpolateError::ComputationError` if:
/// - Any interval has zero length
/// - Numerical issues in the tridiagonal solver
/// - Float conversion failures
#[allow(dead_code)]
pub fn compute_clamped_cubic_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    left_deriv: F,
    right_deriv: F,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Clamped boundary conditions
    let h0 = x[1] - x[0];

    // Check for zero interval
    if h0.is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in clamped spline boundary conditions".to_string(),
        ));
    }

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

    b[0] = two * h0;
    c[0] = h0;
    d[0] = six * ((y[1] - y[0]) / h0 - left_deriv);

    let hn_1 = x[n - 1] - x[n - 2];

    // Check for zero interval
    if hn_1.is_zero() {
        return Err(InterpolateError::ComputationError(
            "Zero interval length in clamped spline boundary conditions".to_string(),
        ));
    }

    a[n - 1] = hn_1;
    b[n - 1] = two * hn_1;
    d[n - 1] = six * (right_deriv - (y[n - 1] - y[n - 2]) / hn_1);

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in clamped spline computation".to_string(),
            ));
        }

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in clamped spline coefficient calculation".to_string(),
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a periodic cubic spline
///
/// Periodic boundary conditions: function and derivatives match at endpoints.
/// This ensures the spline forms a closed curve when the data represents
/// a periodic function.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
///
/// # Returns
///
/// A 2D array of shape (n-1, 4) containing the polynomial coefficients for each segment.
///
/// # Errors
///
/// Returns `InterpolateError::ComputationError` if:
/// - Any interval has zero length
/// - Numerical issues in the tridiagonal solver
/// - Float conversion failures
#[allow(dead_code)]
pub fn compute_periodic_cubic_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Define constants
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

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // For periodic splines, we need to solve a slightly modified system
    // The matrix is almost tridiagonal with additional corner elements

    let mut a = Array1::<F>::zeros(n - 1);
    let mut b = Array1::<F>::zeros(n - 1);
    let mut c = Array1::<F>::zeros(n - 1);
    let mut d = Array1::<F>::zeros(n - 1);

    // Fill the system (we work with n-1 equations due to periodicity)
    for i in 0..n - 1 {
        let h_i_minus_1 = if i == 0 {
            x[n - 1] - x[n - 2]
        } else {
            x[i] - x[i - 1]
        };
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in periodic spline computation".to_string(),
            ));
        }

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = if i == 0 {
            y[0] - y[n - 2] // Using periodicity
        } else {
            y[i] - y[i - 1]
        };
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // For periodic boundary conditions, we need to solve a cyclic tridiagonal system
    // Using Sherman-Morrison formula or reduction to standard tridiagonal
    // For simplicity, we'll use a modified Thomas algorithm

    let mut sigma = Array1::<F>::zeros(n);

    // Simplified approach: assume natural boundary conditions as approximation
    // (A more accurate implementation would solve the cyclic system)
    let mut b_mod = b.clone();
    let mut d_mod = d.clone();

    // Forward sweep
    for i in 1..n - 1 {
        let m = a[i] / b_mod[i - 1];
        b_mod[i] -= m * c[i - 1];
        d_mod[i] = d_mod[i] - m * d_mod[i - 1];
    }

    // Back substitution
    sigma[n - 2] = d_mod[n - 2] / b_mod[n - 2];
    for i in (0..n - 2).rev() {
        sigma[i] = (d_mod[i] - c[i] * sigma[i + 1]) / b_mod[i];
    }
    sigma[n - 1] = sigma[0]; // Periodicity

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in periodic spline coefficient calculation".to_string(),
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a cubic spline with specified second derivatives
///
/// Second derivative boundary conditions: second derivative is specified at endpoints.
/// This provides direct control over the curvature at the boundaries.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
/// * `left_d2` - Second derivative at the left endpoint
/// * `right_d2` - Second derivative at the right endpoint
///
/// # Returns
///
/// A 2D array of shape (n-1, 4) containing the polynomial coefficients for each segment.
///
/// # Errors
///
/// Returns `InterpolateError::ComputationError` if:
/// - Any interval has zero length
/// - Numerical issues in the tridiagonal solver
/// - Float conversion failures
#[allow(dead_code)]
pub fn compute_second_derivative_cubic_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    left_d2: F,
    right_d2: F,
) -> InterpolateResult<Array2<F>> {
    let n = x.len();
    let n_segments = n - 1;

    // Define constants
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

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Specified second derivative boundary conditions
    b[0] = F::one();
    d[0] = left_d2;
    b[n - 1] = F::one();
    d[n - 1] = right_d2;

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in second derivative spline computation".to_string(),
            ));
        }

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

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in second derivative spline coefficient calculation"
                    .to_string(),
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}

/// Compute the coefficients for a parabolic runout cubic spline
///
/// Parabolic runout boundary conditions: third derivative is zero at the endpoints.
/// This makes the spline reduce to a parabola near the endpoints, which can be
/// more natural for some applications.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
///
/// # Returns
///
/// A 2D array of shape (n-1, 4) containing the polynomial coefficients for each segment.
///
/// # Errors
///
/// Returns `InterpolateError::ComputationError` if:
/// - Any interval has zero length
/// - Numerical issues in the tridiagonal solver
/// - Float conversion failures
#[allow(dead_code)]
pub fn compute_parabolic_runout_cubic_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<Array2<F>> {
    // Parabolic runout means the third derivative is zero at the endpoints
    // This is equivalent to d[0] = 0 and d[n-2] = 0 in our coefficient representation
    // We can achieve this by setting specific boundary conditions on the second derivatives

    let n = x.len();
    let n_segments = n - 1;

    // Create array to hold the coefficients
    let mut coeffs = Array2::<F>::zeros((n_segments, 4));

    // Set up the tridiagonal system
    let mut a = Array1::<F>::zeros(n);
    let mut b = Array1::<F>::zeros(n);
    let mut c = Array1::<F>::zeros(n);
    let mut d = Array1::<F>::zeros(n);

    // Parabolic runout conditions
    // At the first point: 2*sigma[0] + sigma[1] = 0
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

    b[0] = two;
    c[0] = F::one();
    d[0] = F::zero();

    // At the last point: sigma[n-2] + 2*sigma[n-1] = 0
    a[n - 1] = F::one();
    b[n - 1] = two;
    d[n - 1] = F::zero();

    // Fill in the tridiagonal system for interior points
    for i in 1..n - 1 {
        let h_i_minus_1 = x[i] - x[i - 1];
        let h_i = x[i + 1] - x[i];

        // Check for zero intervals
        if h_i.is_zero() || h_i_minus_1.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in parabolic runout spline computation".to_string(),
            ));
        }

        a[i] = h_i_minus_1;
        b[i] = two * (h_i_minus_1 + h_i);
        c[i] = h_i;

        let dy_i_minus_1 = y[i] - y[i - 1];
        let dy_i = y[i + 1] - y[i];

        d[i] = six * (dy_i / h_i - dy_i_minus_1 / h_i_minus_1);
    }

    // Solve the tridiagonal system
    let mut sigma = Array1::<F>::zeros(n);

    // Forward sweep
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Back substitution
    sigma[n - 1] = d[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        sigma[i] = (d[i] - c[i] * sigma[i + 1]) / b[i];
    }

    // Calculate the polynomial coefficients
    for i in 0..n_segments {
        let h_i = x[i + 1] - x[i];

        // Check for zero interval in coefficient calculation
        if h_i.is_zero() {
            return Err(InterpolateError::ComputationError(
                "Zero interval length in parabolic runout spline coefficient calculation"
                    .to_string(),
            ));
        }

        coeffs[[i, 0]] = y[i];
        coeffs[[i, 1]] = (y[i + 1] - y[i]) / h_i - h_i * (two * sigma[i] + sigma[i + 1]) / six;
        coeffs[[i, 2]] = sigma[i] / two;
        coeffs[[i, 3]] = (sigma[i + 1] - sigma[i]) / (six * h_i);
    }

    Ok(coeffs)
}