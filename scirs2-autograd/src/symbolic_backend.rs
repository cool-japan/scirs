//! Symbolic backend: wraps [`scirs2_symbolic::eml::LoweredOp`] as a differentiable
//! autograd operation using exact symbolic gradients.
//!
//! # Overview
//!
//! [`EmlOp`] bridges the EML (Expression-tree Meta-Language) from `scirs2-symbolic`
//! into `scirs2-autograd`'s computation graph. Forward evaluation uses
//! [`scirs2_symbolic::eml::eval_real`]; backward uses
//! [`fn@scirs2_symbolic::eml::grad`] to produce a new `EmlOp` wrapping the
//! symbolic partial derivative, achieving exact gradients without any
//! finite-difference approximation.
//!
//! The gradient is dispatched through `gradient.rs::compute_grad_for_input` by
//! matching op name `"EmlOp"` and downcasting via `Op::as_any()`.
//!
//! # Example
//!
//! ```ignore
//! use scirs2_autograd as ag;
//! use scirs2_autograd::tensor_ops as T;
//! use scirs2_symbolic::eml::LoweredOp;
//! use std::sync::Arc;
//!
//! ag::run(|g: &mut ag::Context<f64>| {
//!     // f(x) = x^2, df/dx = 2x
//!     let op = Arc::new(LoweredOp::Pow(
//!         Box::new(LoweredOp::Var(0)),
//!         Box::new(LoweredOp::Const(2.0)),
//!     ));
//!     let x = g.placeholder("x", &[]);
//!     let y = ag::eml_scalar_op(op, &[x], g);
//!     let dy_dx = &T::grad(&[y], &[x])[0];
//!     // ...
//! });
//! ```

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::{Context, Float};
use scirs2_symbolic::eml::{eval_real, grad as sym_grad, EvalCtx, LoweredOp};
use std::sync::Arc;

/// An autograd `Op` backed by a `LoweredOp` symbolic expression.
///
/// **Forward**: evaluates via `scirs2_symbolic::eml::eval_real`.
/// **Backward**: handled by `gradient.rs::compute_grad_for_input` which
///   matches op name `"EmlOp"` and downcasts to this struct via `as_any()`.
///
/// All inputs must be scalar (0-dimensional or single-element) tensors.
/// Variable indices in the stored `LoweredOp` must cover exactly `0..num_vars`.
pub struct EmlOp {
    /// The symbolic expression.
    pub(crate) op: Arc<LoweredOp>,
    /// Number of `Var(i)` inputs expected — equals `inputs.len()` at construction.
    pub(crate) num_vars: usize,
}

impl<F: Float> Op<F> for EmlOp {
    fn name(&self) -> &'static str {
        "EmlOp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Collect variable bindings in order from the input tensors.
        let bindings: Vec<f64> = (0..self.num_vars)
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
        let val = eval_real(&self.op, &eval_ctx).unwrap_or(f64::NAN);
        let out_val = F::from(val).unwrap_or_else(F::nan);
        ctx.append_output(scirs2_core::ndarray::arr0(out_val).into_dyn());
        Ok(())
    }

    /// Gradient is computed in `gradient.rs::compute_grad_for_input` via
    /// `as_any()` downcasting; this method body is never invoked by the
    /// current gradient engine but is kept for API completeness.
    fn grad<'a>(&self, _ctx: &mut GradientContext<'a, 'a, F>) {
        // No-op: the gradient engine dispatches EmlOp gradients via
        // gradient.rs::compute_grad_for_input using as_any() downcasting.
    }

    /// Expose `&self` as `&dyn Any` for downcasting in the gradient engine.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// Create a differentiable scalar tensor backed by a [`LoweredOp`] expression.
///
/// `op` must use only `Var(i)` for `i in 0..inputs.len()`. All inputs must be
/// scalar (0-dimensional or single-element) tensors. The return value is a scalar
/// tensor; its gradient flows back through the exact symbolic derivative.
///
/// # Example
///
/// ```ignore
/// use scirs2_autograd as ag;
/// use scirs2_autograd::tensor_ops as T;
/// use scirs2_symbolic::eml::LoweredOp;
/// use std::sync::Arc;
///
/// ag::run(|g: &mut ag::Context<f64>| {
///     let op = Arc::new(LoweredOp::Pow(
///         Box::new(LoweredOp::Var(0)),
///         Box::new(LoweredOp::Const(2.0)),
///     ));
///     let x = g.placeholder("x", &[]);
///     let y = ag::eml_scalar_op(op, &[x], g);
///     // y represents x^2; its gradient is exactly 2x.
/// });
/// ```
pub fn eml_scalar_op<'g, F: Float>(
    op: Arc<LoweredOp>,
    inputs: &[Tensor<'g, F>],
    g: &'g Context<F>,
) -> Tensor<'g, F> {
    let num_vars = inputs.len();
    let mut builder = Tensor::builder(g);
    for inp in inputs {
        builder = builder.append_input(inp, false);
    }
    builder.build(EmlOp { op, num_vars })
}
