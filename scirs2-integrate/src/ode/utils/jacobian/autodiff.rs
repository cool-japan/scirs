//! Automatic differentiation for Jacobian computation
//!
//! This module provides functions for computing exact Jacobian matrices using
//! automatic differentiation through the scirs2-autograd crate. This eliminates
//! the need for finite difference approximations and can provide better
//! accuracy and performance for complex ODE systems.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
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
    f: &Func,
    t: F,
    y: &Array1<F>,
    _f_current: &Array1<F>, // Not needed but kept for API compatibility
    _perturbation_scale: F, // Not used but kept for API compatibility
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat + scirs2_autograd::Float,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    use ag::tensor_ops as T;
    use ag::{Graph, Tensor};
    use scirs2_autograd as ag;

    let n = y.len();
    let mut jacobian = Array2::<F>::zeros((n, n));

    // Create a new graph
    let mut graph = ag::Graph::<F>::new();

    // Create placeholder tensor for input state variables
    let x = graph.placeholder(&[n]);

    // Build the computation graph for f(t, y)
    let f_clone = f.clone();
    let t_clone = t;

    // Use tensor_map to apply f to x
    let mut wrapped_f = move |x_tensor: Tensor<F>| {
        // Convert Tensor to ndarray, apply f, and convert result back
        let y_ndarray = x_tensor.eval_to_ndarray();
        let result = f_clone(t_clone, y_ndarray.view());
        Tensor::constant(result, &graph)
    };

    // Apply f to x
    let y_tensor = wrapped_f(x);

    // For each output variable, compute gradient with respect to inputs
    for i in 0..n {
        // Extract the i-th component of the output
        let y_i = T::slice(y_tensor, &[i..i + 1], &[]);

        // Compute gradient of y_i with respect to all inputs
        let grads = T::grad(&[y_i], &[x]);

        // Extract gradient values
        let grad_i = grads[0].eval_to_ndarray();

        // Fill the i-th row of the Jacobian
        for j in 0..n {
            jacobian[[i, j]] = grad_i[j];
        }
    }

    Ok(jacobian)
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
    Err(IntegrateError::NotImplementedError(
        "Autodiff Jacobian computation requires the 'autodiff' feature to be enabled.".to_string(),
    ))
}

/// Check if autodiff is available
pub fn is_autodiff_available() -> bool {
    cfg!(feature = "autodiff")
}

/// Jacobian strategy that uses autodiff when available and falls back
/// to finite differences when not
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
    if is_autodiff_available() {
        #[cfg(feature = "autodiff")]
        {
            // Use autodiff when available
            autodiff_jacobian(f, t, y, f_current, perturbation_scale)
        }
        #[cfg(not(feature = "autodiff"))]
        {
            // Fall back to finite differences
            Err(IntegrateError::NotImplementedError(
                "Autodiff feature is not enabled".to_string(),
            ))
        }
    } else {
        // Use finite differences
        Ok(crate::ode::utils::common::finite_difference_jacobian(
            f,
            t,
            y,
            f_current,
            perturbation_scale,
        ))
    }
}
