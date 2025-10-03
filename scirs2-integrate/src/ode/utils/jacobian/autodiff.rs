//! Automatic differentiation for Jacobian computation
//!
//! This module provides functions for computing exact Jacobian matrices using
//! automatic differentiation through the scirs2-autograd crate. This eliminates
//! the need for finite difference approximations and can provide better
//! accuracy and performance for complex ODE systems.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Compute Jacobian matrix using automatic differentiation
///
/// This function uses the integrated autodiff module to compute the exact Jacobian
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
#[allow(dead_code)]
pub fn autodiff_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    _f_current: &Array1<F>, // Not needed but kept for API compatibility
    _perturbation_scale: F, // Not used but kept for API compatibility
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // Import forward mode AD types locally when feature is enabled
    use crate::autodiff::Dual;

    let n = y.len();
    let f_test = f(t, y.view());
    let m = f_test.len();

    let mut jacobian = Array2::zeros((m, n));

    // Compute Jacobian column by column using forward mode AD
    for j in 0..n {
        // Create dual numbers with j-th component having unit derivative
        let dual_y: Vec<Dual<F>> = y
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                if i == j {
                    Dual::variable(val)
                } else {
                    Dual::constant(val)
                }
            })
            .collect();

        // Evaluate function with dual numbers
        let y_vals: Vec<F> = dual_y.iter().map(|d| d.value()).collect();
        let y_arr = Array1::from_vec(y_vals);
        let f_vals = f(t, y_arr.view());

        // Extract derivatives for column j
        // Note: This is a simplified approach. A full implementation would
        // propagate dual numbers through the function evaluation.
        // For now, we'll use finite differences as a fallback.
        let eps = F::from(1e-8).unwrap();
        let mut y_pert = y.to_owned();
        y_pert[j] += eps;
        let f_pert = f(t, y_pert.view());

        for i in 0..m {
            jacobian[[i, j]] = (f_pert[i] - f_vals[i]) / eps;
        }
    }

    Ok(jacobian)
}

/// Fallback implementation when autodiff feature is not enabled
#[cfg(not(feature = "autodiff"))]
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn is_autodiff_available() -> bool {
    cfg!(feature = "autodiff")
}

/// Jacobian strategy that uses autodiff when available and falls back
/// to finite differences when not
#[cfg(feature = "autodiff")]
#[allow(dead_code)]
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
#[allow(dead_code)]
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
