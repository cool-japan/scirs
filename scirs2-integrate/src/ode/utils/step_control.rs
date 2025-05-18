//! Step size control algorithms for adaptive methods.
//!
//! This module provides utility functions for controlling step sizes
//! in adaptive ODE solvers.

use crate::IntegrateFloat;
use ndarray::{Array1, ArrayView1};

/// Calculate the error norm based on relative and absolute tolerances
///
/// # Arguments
///
/// * `error` - Error estimate from the ODE step
/// * `y` - Solution at the current step
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Returns
///
/// The normalized error
pub fn error_norm<F: IntegrateFloat>(error: &Array1<F>, y: &Array1<F>, rtol: F, atol: F) -> F {
    // Calculate the denominator for normalization
    let scale = y
        .iter()
        .zip(error.iter())
        .map(|(y_i, _)| rtol * y_i.abs() + atol)
        .collect::<Array1<F>>();

    // Calculate RMS of scaled error
    let mut sum_sq = F::zero();
    for (e, s) in error.iter().zip(scale.iter()) {
        sum_sq += (*e / *s).powi(2);
    }

    let n = F::from_usize(error.len()).unwrap();
    (sum_sq / n).sqrt()
}

/// Calculate a new step size based on error estimate
///
/// # Arguments
///
/// * `h_current` - Current step size
/// * `error_norm` - Current error norm
/// * `error_order` - Order of the error estimator
/// * `safety` - Safety factor (typically 0.8-0.9)
///
/// # Returns
///
/// The suggested new step size
pub fn calculate_new_step_size<F: IntegrateFloat>(
    h_current: F,
    error_norm: F,
    error_order: usize,
    safety: F,
) -> F {
    // If error is zero, increase step size significantly but safely
    if error_norm == F::zero() {
        return h_current * F::from_f64(10.0).unwrap();
    }

    // Standard step size calculation based on error estimate
    let order = F::from_usize(error_order).unwrap();
    let error_ratio = F::one() / error_norm;

    // Calculate factor using the formula: safety * error_ratio^(1/order)
    let factor = safety * error_ratio.powf(F::one() / order);

    // Limit factor to reasonable bounds to prevent too large or small step sizes
    let factor_max = F::from_f64(10.0).unwrap();
    let factor_min = F::from_f64(0.1).unwrap();

    let factor = if factor > factor_max {
        factor_max
    } else if factor < factor_min {
        factor_min
    } else {
        factor
    };

    // Apply factor to current step size
    h_current * factor
}

/// Select an initial step size for ODE solving
///
/// # Arguments
///
/// * `f` - ODE function
/// * `t` - Initial time
/// * `y` - Initial state
/// * `direction` - Direction of integration (1.0 for forward, -1.0 for backward)
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Returns
///
/// Suggested initial step size
pub fn select_initial_step<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    direction: F,
    rtol: F,
    atol: F,
) -> F
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Calculate scale based on tolerances
    let scale = y
        .iter()
        .map(|y_i| rtol * y_i.abs() + atol)
        .collect::<Array1<F>>();

    // Initial derivatives
    let f0 = f(t, y.view());

    // Estimate using the derivatives
    let d0 = f0
        .iter()
        .zip(scale.iter())
        .map(|(f, s)| *f / *s)
        .fold(F::zero(), |acc, x| acc + x * x);

    let d0 = d0.sqrt() / F::from_f64(y.len() as f64).unwrap().sqrt();

    let step_size = if d0 < F::from_f64(1.0e-5).unwrap() {
        // If derivatives are very small, use a default small step
        F::from_f64(1.0e-6).unwrap()
    } else {
        // Otherwise, use a step size based on the derivatives
        F::from_f64(0.01).unwrap() / d0
    };

    // Return step size with the correct sign
    step_size * direction.signum()
}
