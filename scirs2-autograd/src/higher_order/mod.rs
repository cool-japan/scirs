//! Higher-order automatic differentiation
//!
//! This module provides support for computing higher-order derivatives including:
//! - Hessian-vector products
//! - Jacobian-vector products (JVP)
//! - Vector-Jacobian products (VJP)
//! - Full Hessian matrices (for small dimensions)

use crate::tensor_ops as T;
use crate::{error::AutogradError, tensor::Tensor, Context, Float, NdArray, Result};

pub mod extensions;
pub mod hessian;
pub mod jacobian;
pub mod vjp;

/// Compute Hessian-vector product efficiently
///
/// Given a scalar function f: R^n -> R and a vector v, computes H(f) * v
/// where H(f) is the Hessian matrix of f.
///
/// This is much more efficient than computing the full Hessian matrix,
/// with complexity O(n) instead of O(n²).
///
/// # Arguments
/// * `f` - Scalar output tensor (must have shape `[]` or `[1]`)
/// * `x` - Input tensor with respect to which to compute Hessian
/// * `v` - Vector to multiply with Hessian
/// * `ctx` - Autograd context
///
/// # Returns
/// Hessian-vector product H(f) * v
pub fn hessian_vector_product<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    v: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    // Validate inputs
    if f.shape().len() > 1 || (f.shape().len() == 1 && f.shape()[0] != 1) {
        return Err(AutogradError::shape_error(
            "Function f must be scalar (shape [] or [1])".to_string(),
        ));
    }

    if x.shape() != v.shape() {
        return Err(AutogradError::shape_error(format!(
            "Shapes of x {:?} and v {:?} must match",
            x.shape(),
            v.shape()
        )));
    }

    // Step 1: Compute gradient df/dx
    let grad_f = &crate::tensor_ops::grad(&[*f], &[*x])[0];

    // Step 2: Compute dot product of gradient with v
    let grad_f_flat = crate::tensor_ops::flatten(*grad_f);
    let v_flat = crate::tensor_ops::flatten(*v);
    let dot_product = crate::tensor_ops::reduction::sum_all(grad_f_flat * v_flat);

    // Step 3: Compute gradient of the dot product (this gives Hessian-vector product)
    let hvp = &crate::tensor_ops::grad(&[dot_product], &[*x])[0];

    Ok(*hvp)
}

/// Compute full Hessian matrix (expensive for large dimensions)
///
/// # Arguments
/// * `f` - Scalar output tensor
/// * `x` - Input tensor
/// * `ctx` - Autograd context
///
/// # Warning
/// This has O(n²) complexity and should only be used for small input dimensions.
/// For large dimensions, use `hessian_vector_product` instead.
pub fn hessian<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    // Validate inputs
    if f.shape().len() > 1 || (f.shape().len() == 1 && f.shape()[0] != 1) {
        return Err(AutogradError::shape_error(
            "Function f must be scalar".to_string(),
        ));
    }

    let x_shape = x.shape();
    let n: usize = x_shape.iter().product();

    // Warn for large dimensions
    if n > 1000 {
        eprintln!("Warning: Computing full Hessian for {} parameters. Consider using hessian_vector_product instead.", n);
    }

    // Compute gradient
    let grad_f = &crate::tensor_ops::grad(&[*f], &[*x])[0];
    let grad_f_flat = crate::tensor_ops::flatten(*grad_f);

    // Compute Hessian by differentiating each component of the gradient
    let mut hessian_rows = Vec::with_capacity(n);

    for i in 0..n {
        // Extract i-th component of gradient
        let grad_i = crate::tensor_ops::slice(grad_f_flat, [i as isize], [(i + 1) as isize]);

        // Compute gradient of grad_i with respect to x (i-th row of Hessian)
        let hessian_row = &crate::tensor_ops::grad(&[grad_i], &[*x])[0];
        let hessian_row_flat = crate::tensor_ops::flatten(*hessian_row);

        hessian_rows.push(hessian_row_flat);
    }

    // Stack rows to form Hessian matrix
    let hessian_tensor = crate::tensor_ops::linear_algebra::concat(&hessian_rows, 0);

    Ok(hessian_tensor)
}

/// Compute Jacobian-vector product (forward-mode AD)
///
/// Given a function f: R^n -> R^m and a vector v ∈ R^n,
/// computes J(f) * v where J(f) is the Jacobian matrix.
///
/// # Arguments
/// * `f` - Output tensor (can be multi-dimensional)
/// * `x` - Input tensor
/// * `v` - Vector to multiply with Jacobian
/// * `ctx` - Autograd context
pub fn jacobian_vector_product<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    v: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if x.shape() != v.shape() {
        return Err(AutogradError::shape_error(format!(
            "Shapes of x {:?} and v {:?} must match",
            x.shape(),
            v.shape()
        )));
    }

    // For now, use the definition: JVP = d/dε f(x + ε*v) at ε=0
    // This can be computed using forward-mode AD or by linearization

    // Compute directional derivative using gradient
    let grad_f = &crate::tensor_ops::grad(&[*f], &[*x])[0];

    // JVP is the dot product of gradient with v along input dimensions
    let jvp = *grad_f * *v;

    // Sum over input dimensions to get output dimensions
    // For now, we'll flatten and reduce along all axes
    // This is a simplified implementation
    let jvp_flat = crate::tensor_ops::flatten(jvp);
    let jvp = crate::tensor_ops::reduction::sum_all(jvp_flat);

    Ok(jvp)
}

/// Compute vector-Jacobian product (reverse-mode AD)
///
/// Given a function f: R^n -> R^m and a vector v ∈ R^m,
/// computes v^T * J(f) where J(f) is the Jacobian matrix.
///
/// This is the fundamental operation in reverse-mode automatic differentiation.
///
/// # Arguments
/// * `f` - Output tensor
/// * `x` - Input tensor
/// * `v` - Vector to multiply with Jacobian (cotangent)
/// * `ctx` - Autograd context
pub fn vector_jacobian_product<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    v: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if f.shape() != v.shape() {
        return Err(AutogradError::shape_error(format!(
            "Shapes of f {:?} and v {:?} must match",
            f.shape(),
            v.shape()
        )));
    }

    // VJP is computed by: v^T * J = grad(v^T * f)
    // This is the standard backward pass

    // Compute dot product of v with f
    let vf = crate::tensor_ops::reduction::sum_all(*v * *f);

    // Gradient gives VJP
    let vjp = &crate::tensor_ops::grad(&[vf], &[*x])[0];

    Ok(*vjp)
}

/// Compute mixed partial derivatives
///
/// Computes ∂²f / ∂x ∂y
pub fn mixed_partial<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    y: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    // Validate f is scalar
    if f.shape().len() > 1 || (f.shape().len() == 1 && f.shape()[0] != 1) {
        return Err(AutogradError::shape_error(
            "Function f must be scalar".to_string(),
        ));
    }

    // First derivative with respect to x
    let df_dx = &crate::tensor_ops::grad(&[*f], &[*x])[0];

    // Second derivative with respect to y
    let d2f_dxdy = &crate::tensor_ops::grad(&[*df_dx], &[*y])[0];

    Ok(*d2f_dxdy)
}

/// Higher-order gradient computation
///
/// Computes the n-th order gradient of f with respect to x.
///
/// # Arguments
/// * `f` - Scalar output tensor
/// * `x` - Input tensor
/// * `order` - Order of derivative (1 = gradient, 2 = Hessian diagonal, etc.)
/// * `ctx` - Autograd context
///
/// # Note
/// This repeatedly applies the gradient operator, which can be expensive.
/// For order > 2, consider whether you really need full higher-order derivatives.
pub fn nth_order_gradient<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    order: usize,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if order == 0 {
        return Ok(*f);
    }

    if f.shape().len() > 1 || (f.shape().len() == 1 && f.shape()[0] != 1) {
        return Err(AutogradError::shape_error(
            "Function f must be scalar".to_string(),
        ));
    }

    // Compute first derivative
    let mut current_grad = crate::tensor_ops::grad(&[*f], &[*x])[0];

    // Apply gradient operator (order - 1) more times
    for _ in 1..order {
        // For higher orders, we need to sum the gradient to get a scalar,
        // then differentiate again
        let scalar_grad = crate::tensor_ops::reduction::sum_all(current_grad);
        current_grad = crate::tensor_ops::grad(&[scalar_grad], &[*x])[0];
    }

    Ok(current_grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;

    #[test]
    fn test_hessian_vector_product() {
        crate::run(|ctx: &mut Context<f64>| {
            // f(x) = x^T * x = x1² + x2²
            let x = ctx.placeholder("x", &[2]);
            let axes = [0_isize];
            let f: crate::tensor::Tensor<f64> = reduce_sum(x * x, &axes, false);

            // v = [1, 1]
            let v = convert_to_tensor(
                scirs2_core::ndarray::Array1::from(vec![1.0, 1.0]).into_dyn(),
                ctx,
            );

            // Hessian of f is diag([2, 2])
            // H * v = [2, 2]
            let hvp = hessian_vector_product(&f, &x, &v, ctx).expect("Should compute HVP");

            // Feed x = [1, 1]
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 1.0]);
            let result = ctx
                .evaluator()
                .push(&hvp)
                .feed(x, x_val.view().into_dyn())
                .run();

            // Check result is approximately [2, 2]
            let result_data = result[0]
                .as_ref()
                .expect("Should evaluate")
                .as_slice()
                .expect("Should get slice");
            assert!((result_data[0] - 2.0).abs() < 1e-6);
            assert!((result_data[1] - 2.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_mixed_partial() {
        crate::run(|ctx: &mut Context<f64>| {
            // f(x, y) = x * y
            let x = ctx.placeholder("x", &[]);
            let y = ctx.placeholder("y", &[]);
            let f = x * y;

            // ∂²f / ∂x ∂y = 1
            let mixed = mixed_partial(&f, &x, &y, ctx).expect("Should compute mixed partial");

            // The placeholders must be fed: the gradient graph still references them,
            // even though the mixed partial of `x * y` is the constant 1.  (This used to
            // "work" unfed only because the evaluator silently substituted a dummy
            // scalar for any input it had failed to compute.)
            let xy_val = scirs2_core::ndarray::arr0(3.0);
            let result = ctx
                .evaluator()
                .push(&mixed)
                .feed(x, xy_val.view().into_dyn())
                .feed(y, xy_val.view().into_dyn())
                .run();
            let result_arr = result[0].as_ref().expect("Should evaluate");
            let result_val = result_arr.first().copied().unwrap_or(0.0);

            assert!(
                (result_val - 1.0).abs() < 1e-6,
                "d2(x*y)/dxdy must be 1, got {result_val}"
            );
        });
    }

    #[test]
    fn test_nth_order_gradient() {
        crate::run(|ctx: &mut Context<f64>| {
            // f(x) = x³
            let x = ctx.placeholder("x", &[]);
            let f = x * x * x;

            // First derivative: 3x²
            let grad1 = nth_order_gradient(&f, &x, 1, ctx).expect("Should compute 1st derivative");

            // Second derivative: 6x
            let grad2 = nth_order_gradient(&f, &x, 2, ctx).expect("Should compute 2nd derivative");

            // Feed x = 2
            let x_val = scirs2_core::ndarray::arr0(2.0);

            let result1 = ctx
                .evaluator()
                .push(&grad1)
                .feed(x, x_val.view().into_dyn())
                .run();
            let val1 = result1[0]
                .as_ref()
                .expect("Should evaluate")
                .first()
                .copied()
                .unwrap_or(0.0);
            assert!((val1 - 12.0).abs() < 1e-6); // 3 * 2² = 12

            let result2 = ctx
                .evaluator()
                .push(&grad2)
                .feed(x, x_val.view().into_dyn())
                .run();
            let val2 = result2[0]
                .as_ref()
                .expect("Should evaluate")
                .first()
                .copied()
                .unwrap_or(0.0);
            assert!((val2 - 12.0).abs() < 1e-6); // 6 * 2 = 12
        });
    }
}
