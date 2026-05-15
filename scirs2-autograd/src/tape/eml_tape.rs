//! EML-backed tape operations: element-wise, Jacobian, and Hessian.
//!
//! This module extends the symbolic backend with three new operation types:
//!
//! - [`EmlElementWiseOp`]: applies a `LoweredOp` element-wise to a 1-D input tensor.
//! - [`EmlJacobianOp`]: evaluates a pre-computed symbolic Jacobian (2-D output).
//! - [`EmlHessianOp`]: evaluates a pre-computed symbolic Hessian (2-D output).
//!
//! All types are gated behind `#[cfg(feature = "symbolic")]` and use only
//! `scirs2_symbolic::eml` for symbolic differentiation, with zero finite-difference
//! approximation.
//!
//! # No-unwrap policy
//!
//! No bare `.unwrap()` or `.expect()` in production code. All fallbacks use
//! `.unwrap_or(...)` / `.unwrap_or_else(...)` consistent with the pattern in
//! `symbolic_backend.rs`.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::{Context, Float};
use scirs2_core::ndarray;
use scirs2_symbolic::eml::{
    eval_real, grad as sym_grad, hessian as sym_hessian, jacobian as sym_jacobian, EvalCtx,
    LoweredOp,
};
use std::sync::Arc;

// ============================================================================
// EmlElementWiseOp
// ============================================================================

/// Apply a `LoweredOp` element-wise to a 1-D input tensor.
///
/// The expression `op` must reference only `Var(0)`, which is bound to each
/// element of the input in turn. The output tensor has the same shape as the
/// input.
///
/// **Forward**: maps `eval_real(op, ctx_with_one_var)` over every element.
/// **Backward**: computes `sym_grad(op, 0)` symbolically (once), then applies
///   it element-wise and multiplies by the upstream gradient (chain rule).
pub struct EmlElementWiseOp {
    /// The symbolic expression; must use only `Var(0)`.
    pub(crate) op: Arc<LoweredOp>,
}

impl<F: Float> Op<F> for EmlElementWiseOp {
    fn name(&self) -> &'static str {
        "EmlElementWiseOp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let out: Vec<F> = input
            .iter()
            .map(|&v| {
                let x = v.to_f64().unwrap_or(0.0);
                let binding = [x];
                let eval_ctx = EvalCtx::new(&binding);
                let result = eval_real(&self.op, &eval_ctx).unwrap_or(f64::NAN);
                F::from(result).unwrap_or_else(F::nan)
            })
            .collect();

        let out_arr = ndarray::Array::from_shape_vec(input.raw_dim(), out)
            .map_err(|e| OpError::Other(format!("shape mismatch: {}", e)))?;
        ctx.append_output(out_arr);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut GradientContext<'a, 'a, F>) {
        // Symbolic derivative of the element-wise function w.r.t. Var(0).
        let g_op = Arc::new(sym_grad(&self.op, 0));
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();
        // Build a new EmlElementWiseOp for the derivative, then chain-rule multiply.
        let grad_tensor = Tensor::builder(g)
            .append_input(x, false)
            .build(EmlElementWiseOp { op: g_op });
        let gx = crate::tensor_ops::mul(grad_tensor, gy);
        ctx.append_input_grad(0, Some(gx));
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

// ============================================================================
// EmlJacobianOp
// ============================================================================

/// Evaluate a symbolic Jacobian at a point.
///
/// Given `n_outputs` scalar functions of `n_vars` variables (each stored as a
/// `LoweredOp`), the forward pass evaluates every `grad_ops[i][j] =
/// d(outputs[i]) / d(Var(j))` and writes the result as a 2-D array of shape
/// `[n_outputs, n_vars]`.
///
/// **Forward**: fills a row-major `[n_outputs × n_vars]` array.
/// **Backward**: returns `None` for all inputs (Jacobian is treated as a leaf
///   in the autograd graph — higher-order Jacobian-of-Jacobian is not
///   supported in this release).
pub struct EmlJacobianOp {
    /// Jacobian entries: `grad_ops[i][j]` = d(output_i) / d(Var(j)).
    pub(crate) grad_ops: Vec<Vec<Arc<LoweredOp>>>,
    /// Number of output functions.
    pub(crate) n_outputs: usize,
    /// Number of input variables.
    pub(crate) n_vars: usize,
}

impl<F: Float> Op<F> for EmlJacobianOp {
    fn name(&self) -> &'static str {
        "EmlJacobianOp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Collect variable bindings from scalar input tensors.
        let bindings: Vec<f64> = (0..self.n_vars)
            .map(|i| {
                ctx.input(i)
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or_else(F::zero)
                    .to_f64()
                    .unwrap_or(0.0)
            })
            .collect();

        let eval_ctx = EvalCtx::new(&bindings);
        let mut flat: Vec<F> = Vec::with_capacity(self.n_outputs * self.n_vars);

        for row in &self.grad_ops {
            for cell in row {
                let val = eval_real(cell, &eval_ctx).unwrap_or(f64::NAN);
                flat.push(F::from(val).unwrap_or_else(F::nan));
            }
        }

        let out_arr = ndarray::Array::from_shape_vec((self.n_outputs, self.n_vars), flat)
            .map_err(|e| OpError::Other(format!("jacobian shape error: {}", e)))?;
        ctx.append_output(out_arr.into_dyn());
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut GradientContext<'a, 'a, F>) {
        // Jacobian is a leaf: no higher-order gradients flow through it.
        for i in 0..self.n_vars {
            ctx.append_input_grad(i, None);
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

// ============================================================================
// EmlHessianOp
// ============================================================================

/// Evaluate the Hessian of a scalar `LoweredOp` at a point.
///
/// `hess_ops[i][j]` = d²f / (d Var(i) d Var(j)).  The output is a 2-D
/// array of shape `[n_vars, n_vars]` (row-major, symmetric by construction
/// for smooth f).
///
/// **Forward**: evaluates all `n_vars²` entries.
/// **Backward**: returns `None` for all inputs (Hessian is a leaf).
pub struct EmlHessianOp {
    /// Hessian entries: `hess_ops[i][j]` = d²f / d(Var(i)) d(Var(j)).
    pub(crate) hess_ops: Vec<Vec<Arc<LoweredOp>>>,
    /// Dimension of the Hessian (number of variables).
    pub(crate) n_vars: usize,
}

impl<F: Float> Op<F> for EmlHessianOp {
    fn name(&self) -> &'static str {
        "EmlHessianOp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let bindings: Vec<f64> = (0..self.n_vars)
            .map(|i| {
                ctx.input(i)
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or_else(F::zero)
                    .to_f64()
                    .unwrap_or(0.0)
            })
            .collect();

        let eval_ctx = EvalCtx::new(&bindings);
        let mut flat: Vec<F> = Vec::with_capacity(self.n_vars * self.n_vars);

        for row in &self.hess_ops {
            for cell in row {
                let val = eval_real(cell, &eval_ctx).unwrap_or(f64::NAN);
                flat.push(F::from(val).unwrap_or_else(F::nan));
            }
        }

        let out_arr = ndarray::Array::from_shape_vec((self.n_vars, self.n_vars), flat)
            .map_err(|e| OpError::Other(format!("hessian shape error: {}", e)))?;
        ctx.append_output(out_arr.into_dyn());
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut GradientContext<'a, 'a, F>) {
        for i in 0..self.n_vars {
            ctx.append_input_grad(i, None);
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

// ============================================================================
// Public constructor functions
// ============================================================================

/// Apply `op` element-wise to `input`.
///
/// `op` must reference only `Var(0)`, which is bound to each element in turn.
/// The returned tensor has the same shape as `input`.
///
/// # Example
///
/// ```ignore
/// use scirs2_autograd as ag;
/// use scirs2_symbolic::eml::LoweredOp;
/// use std::sync::Arc;
///
/// ag::run(|g: &mut ag::Context<f64>| {
///     let sin_op = Arc::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0))));
///     let x = g.placeholder("x", &[3]);
///     let y = ag::eml_elementwise(sin_op, x, g);
///     // y[i] = sin(x[i])
/// });
/// ```
pub fn eml_elementwise<'g, F: Float>(
    op: Arc<LoweredOp>,
    input: Tensor<'g, F>,
    g: &'g Context<F>,
) -> Tensor<'g, F> {
    Tensor::builder(g)
        .append_input(input, false)
        .build(EmlElementWiseOp { op })
}

/// Build a Jacobian tensor of shape `[n_outputs, n_inputs]`.
///
/// Each `output_ops[i]` is the symbolic expression for the i-th output
/// function. `inputs` must contain exactly `n_vars` scalar tensors, one
/// per `Var(j)` in the expressions.
///
/// The symbolic Jacobian is computed once via
/// `scirs2_symbolic::eml::jacobian` and cached in the op struct; evaluation
/// at the actual input values happens lazily when the tensor is evaluated.
///
/// # Panics (none)
///
/// All errors surface as `NaN` entries; the function itself does not panic.
pub fn eml_jacobian<'g, F: Float>(
    output_ops: Vec<Arc<LoweredOp>>,
    inputs: &[Tensor<'g, F>],
    g: &'g Context<F>,
) -> Tensor<'g, F> {
    let n_outputs = output_ops.len();
    let n_vars = inputs.len();

    // Build grad_ops[i][j] = d(output_ops[i]) / d(Var(j))
    let grad_ops: Vec<Vec<Arc<LoweredOp>>> = output_ops
        .iter()
        .map(|f_op| {
            sym_jacobian(f_op, n_vars)
                .into_iter()
                .map(Arc::new)
                .collect()
        })
        .collect();

    let mut builder = Tensor::builder(g);
    for inp in inputs {
        builder = builder.append_input(inp, false);
    }
    builder.build(EmlJacobianOp {
        grad_ops,
        n_outputs,
        n_vars,
    })
}

/// Build a Hessian tensor of shape `[n_vars, n_vars]`.
///
/// `f_op` is the scalar symbolic function; `inputs` has one scalar tensor
/// per `Var(j)`. The symbolic Hessian is computed via
/// `scirs2_symbolic::eml::hessian` and cached in the op struct.
pub fn eml_hessian<'g, F: Float>(
    f_op: Arc<LoweredOp>,
    inputs: &[Tensor<'g, F>],
    g: &'g Context<F>,
) -> Tensor<'g, F> {
    let n_vars = inputs.len();

    // hess_ops[i][j] = d²f / (d Var(i) d Var(j))
    let raw_hess = sym_hessian(&f_op, n_vars);
    let hess_ops: Vec<Vec<Arc<LoweredOp>>> = raw_hess
        .into_iter()
        .map(|row| row.into_iter().map(Arc::new).collect())
        .collect();

    let mut builder = Tensor::builder(g);
    for inp in inputs {
        builder = builder.append_input(inp, false);
    }
    builder.build(EmlHessianOp { hess_ops, n_vars })
}
