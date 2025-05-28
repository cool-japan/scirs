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
    use ag::{run, Context, Tensor};
    use scirs2_autograd as ag;

    let n = y.len();
    let mut jacobian = Array2::<F>::zeros((n, n));

    // Clone values needed inside the closure
    let y_clone = y.clone();
    let f_clone = f.clone();

    // Use the run function to create a computation context
    run(|ctx: &mut Context<F>| {
        // Create variable tensor for input state
        let x_var = ctx.variable(y_clone.clone());

        // Convert variable to tensor for operations
        let x_tensor: Tensor<F> = x_var.into();

        // Apply the ODE function
        // We need to extract values, apply f, then convert back
        let x_values = x_tensor.data();
        let x_array = Array1::from_shape_vec(n, x_values.to_vec()).unwrap();
        let f_result = f_clone(t, x_array.view());

        // Convert result back to tensor
        let y_tensor = ctx.constant(f_result);

        // For each output variable, compute gradient with respect to inputs
        for i in 0..n {
            // Extract the i-th component of the output
            let indices = ndarray::arr1(&[i as isize]);
            let y_i = T::gather(&y_tensor, &indices, 0);

            // Compute gradient of y_i with respect to all inputs
            let grads = T::grad(&[&y_i], &[&x_tensor]);

            // Extract gradient values and fill the i-th row of the Jacobian
            let grad_data = grads[0].data();
            for j in 0..n {
                jacobian[[i, j]] = grad_data[j];
            }
        }
    });

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
#[cfg(feature = "autodiff")]
pub fn adaptive_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    perturbation_scale: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat + scirs2_autograd::Float,
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
