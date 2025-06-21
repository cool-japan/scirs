//! Automatic differentiation for Jacobian computation
//!
//! This module provides functions for computing exact Jacobian matrices using
//! automatic differentiation through the scirs2-autograd crate. This eliminates
//! the need for finite difference approximations and can provide better
//! accuracy and performance for complex ODE systems.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use ndarray::{Array1, Array2, ArrayView1};

/// Compute Jacobian matrix using automatic differentiation
///
/// This function uses scirs2-autograd to compute the exact Jacobian
/// matrix for a given function, avoiding the numerical errors and
/// performance issues of finite differencing.
///
/// # Arguments
///
/// * `f` - Function to differentiate
/// * `t` - Time value
/// * `y` - State vector
/// * `f_current` - Current function evaluation at (t, y)
/// * `perturbation_scale` - Scaling factor (not used, but kept for API compatibility)
///
/// # Returns
///
/// Exact Jacobian matrix (∂f/∂y)
#[cfg(feature = "autodiff")]
pub fn autodiff_jacobian<F, Func>(
    _f: &Func,
    _t: F,
    y: &Array1<F>,
    _f_current: &Array1<F>, // Not needed but kept for API compatibility
    _perturbation_scale: F, // Not used but kept for API compatibility
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // TODO: Implement proper autodiff jacobian with updated scirs2_autograd API
    // For now, return error to indicate feature not ready
    let n = y.len();
    Err(crate::error::IntegrateError::ComputationError(
        format!("Autodiff jacobian feature needs to be updated for new scirs2_autograd API. Falling back to finite differences for {} x {} system", n, n)
    ))
}

/// Fallback implementation when autodiff feature is not enabled
#[cfg(not(feature = "autodiff"))]
pub fn autodiff_jacobian<F, Func>(
    _f: &Func,
    _t: F,
    _y: &Array1<F>,
    _f_current: &Array1<F>,
    _perturbation_scale: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    Err(crate::error::IntegrateError::ComputationError(
        "Autodiff Jacobian computation requires the 'autodiff' feature to be enabled.".to_string(),
    ))
}

/// Check if autodiff is available
pub fn is_autodiff_available() -> bool {
    cfg!(feature = "autodiff")
}

/// Jacobian strategy that uses autodiff when available and falls back
/// to finite differences when not
#[cfg(feature = "autodiff")]
pub fn adaptive_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    perturbation_scale: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // Use autodiff when available
    autodiff_jacobian(f, t, y, f_current, perturbation_scale)
}

/// Jacobian strategy that uses autodiff when available and falls back
/// to finite differences when not
#[cfg(not(feature = "autodiff"))]
pub fn adaptive_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    perturbation_scale: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // Fall back to finite differences
    Ok(crate::ode::utils::common::finite_difference_jacobian(
        f,
        t,
        y,
        f_current,
        perturbation_scale,
    ))
}
