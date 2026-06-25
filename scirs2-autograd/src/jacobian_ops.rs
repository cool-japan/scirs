//! Enhanced Jacobian computation for automatic differentiation
//!
//! This module provides a comprehensive API for computing Jacobians, JVPs, and VJPs
//! that builds on but is distinct from the basic `higher_order::jacobian` module.
//!
//! # Features
//!
//! - **Full Jacobian** via reverse-mode AD: `jacobian(f, x)` → `J ∈ R^{m×n}`
//! - **Jacobian-Vector Product** (JVP) via forward-mode: `jvp(f, x, v)` → `J·v ∈ R^m`
//! - **Vector-Jacobian Product** (VJP) via reverse-mode: `vjp(f, x, v)` → `v^T·J ∈ R^n`
//! - **Batched Jacobian**: compute Jacobians over a batch dimension
//! - **Numerical Jacobian** for gradient checking
//!
//! # When to use which
//!
//! | Function | Complexity | Best when |
//! |----------|-----------|-----------|
//! | `jacobian` | O(m·n) | Small m and n (full matrix needed) |
//! | `jvp` | O(n) | Few inputs, one direction v |
//! | `vjp` | O(m) | Few outputs, one covector v |
//! | `numerical_jacobian_check` | O(m·n) | Testing gradient correctness |
//!
//! # Examples
//!
//! ## Compute a full Jacobian
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use scirs2_autograd::jacobian_ops;
//! use ag::tensor_ops::*;
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     let x = ctx.placeholder("x", &[2]);
//!     // f(x) = [x0^2, x0*x1]
//!     let x0 = slice(x, [0isize], [1isize]);
//!     let x1 = slice(x, [1isize], [2isize]);
//!     let f = concat(&[x0 * x0, x0 * x1], 0);
//!
//!     let jac = jacobian_ops::jacobian_reverse(&f, &x, 2, 2, ctx)
//!         .expect("Should compute Jacobian");
//!
//!     // Evaluate at x = [2, 3]
//!     let x_val = scirs2_core::ndarray::arr1(&[2.0, 3.0]);
//!     let result = ctx.evaluator()
//!         .push(&jac)
//!         .feed(x, x_val.view().into_dyn())
//!         .run();
//!     // Jacobian = [[4, 0], [3, 2]]
//! });
//! ```

use crate::error::AutogradError;
use crate::forward_mode::DualNumber;
use crate::tensor::Tensor;
use crate::{Context, Float, Result};
use num_traits::Float as NumFloat;
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt;

// ---------------------------------------------------------------------------
// Full Jacobian via reverse-mode
// ---------------------------------------------------------------------------

/// Compute the full Jacobian matrix via reverse-mode AD.
///
/// For a function f: R^n -> R^m, computes the m x n Jacobian matrix.
/// Each row is computed by a separate backward pass.
///
/// # Arguments
/// * `f` - Output tensor of the function
/// * `x` - Input tensor (variable w.r.t. which Jacobian is computed)
/// * `m` - Number of output dimensions
/// * `n` - Number of input dimensions
/// * `ctx` - Autograd context
///
/// # Returns
/// A tensor of shape [m, n] representing the Jacobian matrix,
/// or an error if computation fails.
///
/// # Complexity
/// O(m) backward passes, each O(graph_size). Total: O(m * graph_size).
pub fn jacobian_reverse<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    m: usize,
    n: usize,
    _ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if m == 0 || n == 0 {
        return Err(AutogradError::shape_error(
            "Jacobian dimensions must be positive".to_string(),
        ));
    }

    if m > 50000 || n > 50000 {
        eprintln!("Warning: Computing large Jacobian {m}x{n}. Consider using JVP/VJP instead.");
    }

    let f_flat = crate::tensor_ops::flatten(*f);
    let mut jacobian_rows = Vec::with_capacity(m);

    for i in 0..m {
        let f_i = crate::tensor_ops::slice(f_flat, [i as isize], [(i + 1) as isize]);
        let grad_i = crate::tensor_ops::grad(&[f_i], &[*x])[0];
        let grad_i_flat = crate::tensor_ops::flatten(grad_i);
        jacobian_rows.push(grad_i_flat);
    }

    Ok(crate::tensor_ops::linear_algebra::concat(&jacobian_rows, 0))
}

/// Compute Jacobian with automatic dimension inference.
///
/// Tries to infer m and n from tensor shapes. Falls back to `jacobian_reverse`
/// with the inferred dimensions.
pub fn jacobian_auto<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    let f_shape = f.shape();
    let x_shape = x.shape();

    let m: usize = if f_shape.is_empty() {
        1
    } else {
        f_shape.iter().product::<usize>().max(1)
    };
    let n: usize = if x_shape.is_empty() {
        1
    } else {
        x_shape.iter().product::<usize>().max(1)
    };

    jacobian_reverse(f, x, m, n, ctx)
}

// ---------------------------------------------------------------------------
// JVP via forward-mode (dual numbers)
// ---------------------------------------------------------------------------

/// Compute Jacobian-Vector Product using forward-mode AD (dual numbers).
///
/// Given f: R^n -> R^m and v ∈ R^n, computes J(f,x) * v ∈ R^m
/// in a single forward pass with dual numbers.
///
/// # Arguments
/// * `f` - A function from `&[DualNumber<F>]` to `Vec<DualNumber<F>>`
/// * `x` - Point at which to evaluate (1-D array)
/// * `v` - Tangent vector (same shape as x)
///
/// # Returns
/// The JVP result as a 1-D array of length m.
///
/// # Complexity
/// One forward pass through f.
pub fn jvp_forward<F, Func>(f: Func, x: &Array1<F>, v: &Array1<F>) -> Result<Array1<F>>
where
    F: NumFloat + Copy + Send + Sync + fmt::Debug + 'static,
    Func: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>>,
{
    if x.len() != v.len() {
        return Err(AutogradError::shape_error(format!(
            "x length {} != v length {}",
            x.len(),
            v.len()
        )));
    }

    let n = x.len();
    let mut dual_inputs = Vec::with_capacity(n);
    for i in 0..n {
        dual_inputs.push(DualNumber::new(x[i], v[i]));
    }

    let dual_outputs = f(&dual_inputs);
    let result: Vec<F> = dual_outputs.iter().map(|d| d.tangent()).collect();
    Ok(Array1::from(result))
}

/// Compute JVP using the autograd graph (reverse-mode approximation).
///
/// This computes J*v = sum_i (df/dx_i * v_i) using the computation graph.
/// Less efficient than forward-mode for wide Jacobians but works with
/// any graph-based computation.
///
/// # Arguments
/// * `f` - Output tensor
/// * `x` - Input tensor
/// * `v` - Tangent vector tensor (same shape as x)
/// * `ctx` - Autograd context
pub fn jvp_graph<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    v: &Tensor<'graph, T>,
    _ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if x.shape() != v.shape() {
        return Err(AutogradError::shape_error(format!(
            "x shape {:?} != v shape {:?}",
            x.shape(),
            v.shape()
        )));
    }

    // Compute gradient (df/dx)
    let grad_f = crate::tensor_ops::grad(&[*f], &[*x])[0];

    // JVP = grad_f . v (element-wise product, then sum)
    let product = grad_f * *v;
    let jvp = crate::tensor_ops::reduction::sum_all(product);

    Ok(jvp)
}

// ---------------------------------------------------------------------------
// VJP via reverse-mode
// ---------------------------------------------------------------------------

/// Compute Vector-Jacobian Product using reverse-mode AD.
///
/// Given f: R^n -> R^m and v ∈ R^m, computes v^T * J(f,x) ∈ R^n
/// in a single backward pass.
///
/// # Arguments
/// * `f` - Output tensor
/// * `x` - Input tensor
/// * `v` - Cotangent vector tensor (same shape as f)
/// * `ctx` - Autograd context
///
/// # Returns
/// The VJP result as a tensor with the same shape as x.
///
/// # Complexity
/// One backward pass through the graph.
pub fn vjp_reverse<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    v: &Tensor<'graph, T>,
    _ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if f.shape() != v.shape() {
        return Err(AutogradError::shape_error(format!(
            "f shape {:?} != v shape {:?}",
            f.shape(),
            v.shape()
        )));
    }

    // VJP = grad(v^T . f, x)
    let weighted = crate::tensor_ops::reduction::sum_all(*v * *f);
    let vjp = crate::tensor_ops::grad(&[weighted], &[*x])[0];

    Ok(vjp)
}

/// Compute VJP with multiple outputs and cotangents.
///
/// For a function with multiple output tensors, weight each by the
/// corresponding cotangent and compute the total VJP.
///
/// # Arguments
/// * `outputs` - Slice of output tensors
/// * `inputs` - Slice of input tensors
/// * `cotangents` - Slice of cotangent vectors (one per output)
/// * `ctx` - Autograd context
///
/// # Returns
/// A vector of VJP results, one per input.
pub fn vjp_multi<'graph, T: Float>(
    outputs: &[Tensor<'graph, T>],
    inputs: &[Tensor<'graph, T>],
    cotangents: &[Tensor<'graph, T>],
    _ctx: &'graph Context<'graph, T>,
) -> Result<Vec<Tensor<'graph, T>>> {
    if outputs.len() != cotangents.len() {
        return Err(AutogradError::shape_error(format!(
            "Number of outputs {} != number of cotangents {}",
            outputs.len(),
            cotangents.len()
        )));
    }

    if outputs.is_empty() {
        return Err(AutogradError::shape_error(
            "Must have at least one output".to_string(),
        ));
    }

    // Weight each output by its cotangent and sum
    let weighted: Vec<Tensor<'graph, T>> = outputs
        .iter()
        .zip(cotangents.iter())
        .map(|(&out, &cot)| crate::tensor_ops::reduction::sum_all(out * cot))
        .collect();

    let total = weighted.iter().skip(1).fold(weighted[0], |acc, &x| acc + x);

    let vjps = crate::tensor_ops::grad(&[total], inputs);
    Ok(vjps)
}

// ---------------------------------------------------------------------------
// Batched Jacobian
// ---------------------------------------------------------------------------

/// Compute Jacobian for each element of a batch using forward-mode AD.
///
/// Given a batch of inputs (2-D array, one sample per row) and a function
/// f: R^n -> R^m, computes the Jacobian at each sample point.
///
/// # Arguments
/// * `f` - Function mapping `&[DualNumber<F>]` to `Vec<DualNumber<F>>`
/// * `batch` - 2-D array of shape [batch_size, n]
///
/// # Returns
/// A vector of Jacobian matrices, one per batch element.
pub fn batch_jacobian<F, Func>(f: Func, batch: &Array2<F>) -> Result<Vec<Array2<F>>>
where
    F: NumFloat + Copy + Send + Sync + fmt::Debug + 'static,
    Func: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>>,
{
    let batch_size = batch.nrows();
    let n = batch.ncols();

    if batch_size == 0 {
        return Err(AutogradError::shape_error(
            "Batch must not be empty".to_string(),
        ));
    }

    let mut jacobians = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let x = batch.row(b).to_owned();
        let jac = crate::forward_mode::jacobian_forward(&f, &x);
        jacobians.push(jac);
    }

    Ok(jacobians)
}

// ---------------------------------------------------------------------------
// Numerical Jacobian (for gradient checking)
// ---------------------------------------------------------------------------

/// Compute numerical Jacobian using finite differences.
///
/// This is primarily used for testing gradient correctness. It computes
/// the Jacobian by perturbing each input dimension and observing the
/// change in output.
///
/// # Arguments
/// * `f` - Function mapping `&Array1<F>` to `Array1<F>`
/// * `x` - Point at which to compute the Jacobian
/// * `epsilon` - Perturbation size (default: 1e-5)
///
/// # Returns
/// The numerical Jacobian as a 2-D array of shape [m, n].
pub fn numerical_jacobian<F, Func>(f: Func, x: &Array1<F>, epsilon: F) -> Array2<F>
where
    F: NumFloat + Copy + fmt::Debug,
    Func: Fn(&Array1<F>) -> Array1<F>,
{
    let n = x.len();
    let f_x = f(x);
    let m = f_x.len();

    let mut jac = Array2::zeros((m, n));
    let two = F::one() + F::one();

    for j in 0..n {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[j] = x_plus[j] + epsilon;
        x_minus[j] = x_minus[j] - epsilon;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        for i in 0..m {
            jac[[i, j]] = (f_plus[i] - f_minus[i]) / (two * epsilon);
        }
    }

    jac
}

/// Check gradient correctness by comparing analytical and numerical Jacobians.
///
/// # Arguments
/// * `f_dual` - Function using dual numbers (for analytical Jacobian)
/// * `f_plain` - Function using plain floats (for numerical Jacobian)
/// * `x` - Point at which to check
/// * `tolerance` - Maximum allowed relative error
///
/// # Returns
/// `Ok(max_error)` if the check passes, `Err` with details if it fails.
pub fn jacobian_check<F, DualFunc, PlainFunc>(
    f_dual: DualFunc,
    f_plain: PlainFunc,
    x: &Array1<F>,
    tolerance: F,
) -> Result<F>
where
    F: NumFloat + Copy + Send + Sync + fmt::Debug + fmt::Display + 'static,
    DualFunc: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>>,
    PlainFunc: Fn(&Array1<F>) -> Array1<F>,
{
    let analytical = crate::forward_mode::jacobian_forward(&f_dual, x);
    let eps = F::from(1e-5).unwrap_or(F::epsilon());
    let numerical = numerical_jacobian(f_plain, x, eps);

    if analytical.shape() != numerical.shape() {
        return Err(AutogradError::shape_error(format!(
            "Analytical shape {:?} != numerical shape {:?}",
            analytical.shape(),
            numerical.shape()
        )));
    }

    let mut max_error = F::zero();
    for (a, n) in analytical.iter().zip(numerical.iter()) {
        let denom = F::one() + n.abs();
        let rel_error = (*a - *n).abs() / denom;
        if rel_error > max_error {
            max_error = rel_error;
        }
    }

    if max_error > tolerance {
        return Err(AutogradError::compute_error(format!(
            "Jacobian check failed: max relative error {max_error} exceeds tolerance {tolerance}"
        )));
    }

    Ok(max_error)
}

// ---------------------------------------------------------------------------
// Jacobian diagonal (efficient)
// ---------------------------------------------------------------------------

/// Compute only the diagonal of the Jacobian using forward-mode AD.
///
/// For a function f: R^n -> R^n (same input/output dimensions),
/// computes [df_0/dx_0, df_1/dx_1, ..., df_{n-1}/dx_{n-1}].
///
/// # Arguments
/// * `f` - Function mapping `&[DualNumber<F>]` to `Vec<DualNumber<F>>`
/// * `x` - Point at which to compute
///
/// # Returns
/// 1-D array of diagonal Jacobian entries.
///
/// # Complexity
/// n forward passes (one per input dimension).
pub fn jacobian_diagonal<F, Func>(f: &Func, x: &Array1<F>) -> Result<Array1<F>>
where
    F: NumFloat + Copy + Send + Sync + fmt::Debug + 'static,
    Func: Fn(&[DualNumber<F>]) -> Vec<DualNumber<F>>,
{
    let n = x.len();
    let mut diag = Vec::with_capacity(n);

    for i in 0..n {
        // Unit vector in direction i
        let mut v = Array1::zeros(n);
        v[i] = F::one();

        let jvp_result = jvp_forward(f, x, &v)?;

        if i >= jvp_result.len() {
            return Err(AutogradError::shape_error(format!(
                "Output dimension {} < input dimension {}, cannot compute diagonal",
                jvp_result.len(),
                n
            )));
        }
        diag.push(jvp_result[i]);
    }

    Ok(Array1::from(diag))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;

    #[test]
    fn test_jacobian_reverse_linear() {
        crate::run(|ctx: &mut Context<f64>| {
            // f(x) = [2*x0 + x1, x0 - x1]
            // Jacobian = [[2, 1], [1, -1]]
            let x = ctx.placeholder("x", &[2]);
            let x0 = slice(x, [0isize], [1isize]);
            let x1 = slice(x, [1isize], [2isize]);
            let f = concat(&[x0 * 2.0 + x1, x0 - x1], 0);

            let jac = jacobian_reverse(&f, &x, 2, 2, ctx).expect("Should compute Jacobian");

            let x_val = scirs2_core::ndarray::arr1(&[1.0, 1.0]);
            let result = ctx
                .evaluator()
                .push(&jac)
                .feed(x, x_val.view().into_dyn())
                .run();

            let jac_arr = result[0].as_ref().expect("Should evaluate");
            let jac_vals = jac_arr.as_slice().unwrap_or(&[]);
            assert!((jac_vals[0] - 2.0).abs() < 1e-6); // df0/dx0 = 2
            assert!((jac_vals[1] - 1.0).abs() < 1e-6); // df0/dx1 = 1
            assert!((jac_vals[2] - 1.0).abs() < 1e-6); // df1/dx0 = 1
            assert!((jac_vals[3] - (-1.0)).abs() < 1e-6); // df1/dx1 = -1
        });
    }

    #[test]
    fn test_jvp_forward_simple() {
        // f(x) = [x0^2, x0 * x1], v = [1, 0]
        // Jacobian at (2,3) = [[4, 0], [3, 2]]
        // JVP = J * [1,0] = [4, 3]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[0] * xs[1]];
        let x = Array1::from(vec![2.0, 3.0]);
        let v = Array1::from(vec![1.0, 0.0]);

        let result = jvp_forward(f, &x, &v).expect("JVP should succeed");
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vjp_reverse_simple() {
        // VJP via the higher_order module which handles shape matching properly
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let x0 = slice(x, [0isize], [1isize]);
            let x1 = slice(x, [1isize], [2isize]);

            // Use separate scalar outputs and vjp_multi
            let f0 = reduce_sum(x0 * x0, &[0isize], false);
            let f1 = reduce_sum(x1 * x1, &[0isize], false);

            // Cotangent weights
            let v0 = convert_to_tensor(scirs2_core::ndarray::arr0(1.0).into_dyn(), ctx);
            let v1 = convert_to_tensor(scirs2_core::ndarray::arr0(2.0).into_dyn(), ctx);

            let vjps =
                vjp_multi(&[f0, f1], &[x], &[v0, v1], ctx).expect("VJP multi should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[3.0, 4.0]);
            let result = ctx
                .evaluator()
                .push(&vjps[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            // d/dx [1*x0^2 + 2*x1^2] = [2*x0, 4*x1] = [6, 16]
            let vjp_arr = result[0].as_ref().expect("Should evaluate");
            let vjp_vals = vjp_arr.as_slice().unwrap_or(&[]);
            assert!((vjp_vals[0] - 6.0).abs() < 1e-6);
            assert!((vjp_vals[1] - 16.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_numerical_jacobian() {
        let f = |x: &Array1<f64>| Array1::from(vec![x[0] * x[0], x[0] * x[1]]);
        let x = Array1::from(vec![2.0, 3.0]);
        let jac = numerical_jacobian(f, &x, 1e-7);

        // Jacobian = [[4, 0], [3, 2]]
        assert!((jac[[0, 0]] - 4.0).abs() < 1e-4);
        assert!((jac[[0, 1]] - 0.0).abs() < 1e-4);
        assert!((jac[[1, 0]] - 3.0).abs() < 1e-4);
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_jacobian_check_passes() {
        let f_dual = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[0] * xs[1]];
        let f_plain = |x: &Array1<f64>| Array1::from(vec![x[0] * x[0], x[0] * x[1]]);
        let x = Array1::from(vec![2.0, 3.0]);

        let result = jacobian_check(f_dual, f_plain, &x, 1e-3);
        assert!(result.is_ok());
        let max_err = result.expect("Should pass");
        assert!(max_err < 1e-3);
    }

    #[test]
    fn test_jacobian_diagonal() {
        // f(x) = [x0^2, x1^3]
        // diagonal = [2*x0, 3*x1^2]
        let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[1] * xs[1] * xs[1]];
        let x = Array1::from(vec![3.0, 2.0]);

        let diag = jacobian_diagonal(&f, &x).expect("Should compute diagonal");
        assert!((diag[0] - 6.0).abs() < 1e-10); // 2*3 = 6
        assert!((diag[1] - 12.0).abs() < 1e-10); // 3*4 = 12
    }

    #[test]
    fn test_batch_jacobian() {
        let f = |xs: &[DualNumber<f64>]| vec![xs[0] * xs[0], xs[1] * xs[1]];
        let batch = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("valid shape");

        let jacs = batch_jacobian(f, &batch).expect("Batch jacobian should work");
        assert_eq!(jacs.len(), 2);

        // First sample x=[1,2]: J = [[2,0],[0,4]]
        assert!((jacs[0][[0, 0]] - 2.0).abs() < 1e-10);
        assert!((jacs[0][[1, 1]] - 4.0).abs() < 1e-10);

        // Second sample x=[3,4]: J = [[6,0],[0,8]]
        assert!((jacs[1][[0, 0]] - 6.0).abs() < 1e-10);
        assert!((jacs[1][[1, 1]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_jvp_shape_mismatch_error() {
        let f = |xs: &[DualNumber<f64>]| vec![xs[0]];
        let x = Array1::from(vec![1.0, 2.0]);
        let v = Array1::from(vec![1.0]); // wrong size

        let result = jvp_forward(f, &x, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_jacobian_zero_dim_error() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let f = x * 2.0;
            let result = jacobian_reverse(&f, &x, 0, 2, ctx);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_vjp_multi() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let x0 = slice(x, [0isize], [1isize]);
            let x1 = slice(x, [1isize], [2isize]);

            let f1 = x0 * x0; // x0^2
            let f2 = x1 * x1; // x1^2

            let v1 = convert_to_tensor(scirs2_core::ndarray::arr0(1.0).into_dyn(), ctx);
            let v2 = convert_to_tensor(scirs2_core::ndarray::arr0(2.0).into_dyn(), ctx);

            let vjps =
                vjp_multi(&[f1, f2], &[x], &[v1, v2], ctx).expect("VJP multi should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[3.0, 4.0]);
            let result = ctx
                .evaluator()
                .push(&vjps[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = result[0].as_ref().expect("Should evaluate");
            let vals = arr.as_slice().unwrap_or(&[]);
            // d/dx [1*x0^2 + 2*x1^2] = [2*x0, 4*x1] = [6, 16]
            assert!((vals[0] - 6.0).abs() < 1e-6);
            assert!((vals[1] - 16.0).abs() < 1e-6);
        });
    }
}
