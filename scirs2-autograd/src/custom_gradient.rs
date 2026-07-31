//! Custom gradient rules for automatic differentiation
//!
//! This module provides a `@custom_gradient` equivalent: the ability to register
//! user-defined backward (gradient) functions for custom forward computations.
//! This is essential for:
//!
//! - Numerically stabilized gradients (e.g., log-sum-exp)
//! - Gradients through non-differentiable operations (e.g., quantization with STE)
//! - Approximate gradients for expensive operations
//! - Stop-gradient annotations for selective gradient blocking
//!
//! # Architecture
//!
//! A custom gradient is defined by implementing the [`CustomGradientOp`] trait, which
//! has two methods:
//!
//! - `forward(&self, inputs) -> outputs`: the primal computation
//! - `backward(&self, output_grads, saved) -> input_grads`: the custom gradient rule
//!
//! The [`custom_op`] function wraps this into a standard autograd `Op`.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use scirs2_autograd::custom_gradient::{CustomGradientOp, custom_op};
//! use scirs2_autograd::tensor_ops;
//! use scirs2_core::ndarray;
//!
//! /// Straight-Through Estimator: forward rounds to nearest int,
//! /// backward passes gradient through unchanged.
//! struct StraightThroughEstimator;
//!
//! impl CustomGradientOp<f64> for StraightThroughEstimator {
//!     fn forward(
//!         &self,
//!         inputs: &[scirs2_core::ndarray::ArrayViewD<f64>],
//!     ) -> Result<scirs2_core::ndarray::ArrayD<f64>, ag::error::OpError> {
//!         let x = &inputs[0];
//!         Ok(x.mapv(|v| v.round()))
//!     }
//!
//!     fn backward<'g>(
//!         &self,
//!         output_grad: &ag::Tensor<'g, f64>,
//!         _saved_tensors: &[ag::Tensor<'g, f64>],
//!         _ctx: &'g ag::Graph<f64>,
//!     ) -> Vec<Option<ag::Tensor<'g, f64>>> {
//!         // STE: pass gradient through unchanged
//!         vec![Some(*output_grad)]
//!     }
//!
//!     fn num_inputs(&self) -> usize { 1 }
//!     fn name(&self) -> &'static str { "StraightThroughEstimator" }
//! }
//! ```

use crate::error::OpError;
use crate::op::{self, ComputeContext, GradientContext};
use crate::tensor::Tensor;
use crate::{Context, Float, NdArray};
use scirs2_core::ndarray::{ArrayD, ArrayViewD};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CustomGradientOp trait
// ---------------------------------------------------------------------------

/// Trait for defining custom differentiable operations with user-specified gradients.
///
/// Implement this trait to define both the forward computation and the backward
/// (gradient) computation for a custom operation. This is analogous to PyTorch's
/// `torch.autograd.Function` or TensorFlow's `@tf.custom_gradient`.
///
/// # Type Parameters
/// * `F` - The floating point type (f32, f64)
///
/// # Contract
/// - `forward` receives immutable views of input arrays and must produce an output array.
/// - `backward` receives the output gradient tensor, any saved tensors from the forward
///   pass, and the autograd context, and must return one `Option<Tensor>` per input.
///   Return `None` for inputs that don't need gradients.
pub trait CustomGradientOp<F: Float>: Send + Sync {
    /// Forward computation.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input array views
    ///
    /// # Returns
    /// The output array, or an error if computation fails.
    fn forward(&self, inputs: &[ArrayViewD<F>]) -> Result<ArrayD<F>, OpError>;

    /// Backward computation (custom gradient rule).
    ///
    /// # Arguments
    /// * `output_grad` - Gradient flowing from downstream
    /// * `saved_tensors` - Tensors saved during forward pass (inputs + output)
    /// * `ctx` - The autograd graph context
    ///
    /// # Returns
    /// A vector of optional gradient tensors, one per input. `None` means no
    /// gradient for that input.
    fn backward<'g>(
        &self,
        output_grad: &Tensor<'g, F>,
        saved_tensors: &[Tensor<'g, F>],
        ctx: &'g crate::graph::Graph<F>,
    ) -> Vec<Option<Tensor<'g, F>>>;

    /// Number of inputs this op expects.
    fn num_inputs(&self) -> usize;

    /// Human-readable name for debugging and visualization.
    fn name(&self) -> &'static str {
        "CustomGradientOp"
    }

    /// Whether this op saves its inputs for the backward pass.
    ///
    /// If `true`, all input tensors are available in `saved_tensors[0..num_inputs()]`
    /// during `backward`. If `false`, only the output is saved.
    fn saves_inputs(&self) -> bool {
        true
    }

    /// Whether this op saves its output for the backward pass.
    ///
    /// If `true`, the output tensor is available as the last element of `saved_tensors`.
    fn saves_output(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Internal Op wrapper
// ---------------------------------------------------------------------------

/// Internal wrapper that bridges `CustomGradientOp` to the autograd `Op` trait.
struct CustomGradientWrapper<F: Float> {
    inner: Arc<dyn CustomGradientOp<F>>,
}

impl<F: Float> op::Op<F> for CustomGradientWrapper<F> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input_views: Vec<ArrayViewD<F>> = ctx.inputs();
        let output = self.inner.forward(&input_views)?;
        ctx.append_output(output);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let output_grad = ctx.output_grad();
        let graph = ctx.graph();

        // Collect saved tensors: inputs first, then output
        let mut saved = Vec::new();
        if self.inner.saves_inputs() {
            for i in 0..ctx.num_inputs() {
                saved.push(*ctx.input(i));
            }
        }
        if self.inner.saves_output() {
            saved.push(*ctx.output());
        }

        let input_grads = self.inner.backward(output_grad, &saved, graph);

        for (i, grad) in input_grads.into_iter().enumerate() {
            ctx.append_input_grad(i, grad);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: custom_op
// ---------------------------------------------------------------------------

/// Create a tensor node with a custom gradient rule.
///
/// This is the primary entry point for using custom gradients. It takes a
/// [`CustomGradientOp`] implementation and one or more input tensors, and
/// returns a new tensor whose backward pass uses the custom gradient rule.
///
/// # Arguments
/// * `op` - The custom gradient operation (wrapped in `Arc` for shared ownership)
/// * `inputs` - Slice of input tensors
/// * `ctx` - The autograd context
///
/// # Returns
/// A new tensor representing the output of the custom operation.
///
/// # Example
/// ```rust
/// use scirs2_autograd as ag;
/// use scirs2_autograd::custom_gradient::{CustomGradientOp, custom_op};
/// use std::sync::Arc;
///
/// struct DoubleOp;
/// impl CustomGradientOp<f64> for DoubleOp {
///     fn forward(
///         &self,
///         inputs: &[scirs2_core::ndarray::ArrayViewD<f64>],
///     ) -> Result<scirs2_core::ndarray::ArrayD<f64>, ag::error::OpError> {
///         Ok(inputs[0].mapv(|v| v * 2.0))
///     }
///     fn backward<'g>(
///         &self,
///         output_grad: &ag::Tensor<'g, f64>,
///         _saved: &[ag::Tensor<'g, f64>],
///         _ctx: &'g ag::Graph<f64>,
///     ) -> Vec<Option<ag::Tensor<'g, f64>>> {
///         vec![Some(*output_grad * 2.0)]
///     }
///     fn num_inputs(&self) -> usize { 1 }
///     fn name(&self) -> &'static str { "DoubleOp" }
/// }
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[3]);
///     let op = Arc::new(DoubleOp);
///     let y = custom_op(op, &[x], ctx);
///     // y = 2*x, dy/dx = 2
/// });
/// ```
pub fn custom_op<'g, F: Float>(
    op: Arc<dyn CustomGradientOp<F>>,
    inputs: &[Tensor<'g, F>],
    ctx: &'g Context<'g, F>,
) -> Tensor<'g, F> {
    let wrapper = CustomGradientWrapper { inner: op };
    let mut builder = Tensor::builder(ctx);
    for input in inputs {
        builder = builder.append_input(input, false);
    }
    builder.build(wrapper)
}

// ---------------------------------------------------------------------------
// Convenience: custom_unary_op
// ---------------------------------------------------------------------------

/// Create a custom unary operation with a closure-based gradient.
///
/// This is a convenience wrapper for the common case of a single-input,
/// single-output operation where both forward and backward can be expressed
/// as closures.
///
/// The backward closure is called during backpropagation with
/// `(output_grad, input, output)` and returns the gradient w.r.t. the input, or
/// `None` to block the gradient entirely. It is applied **verbatim**: this function
/// makes no attempt to check the rule against the forward pass, which is the whole
/// point of a custom gradient (straight-through estimators, stabilized rules,
/// deliberately approximate rules). Use
/// [`crate::test_helper::gradient_check`] if you want the rule verified numerically.
///
/// # Arguments
/// * `name` - Name for debugging
/// * `forward_fn` - Closure computing the forward pass
/// * `backward_fn` - Closure computing the gradient given (output_grad, input, output)
/// * `input` - The input tensor
/// * `ctx` - The autograd context
///
/// # Closure signature
///
/// `backward_fn` must be generic over the graph lifetime (a higher-ranked bound):
/// annotate its parameters with an anonymous lifetime — `&Tensor<'_, f64>` — and build
/// the result out of the tensors it is handed rather than out of tensors captured from
/// the enclosing scope (a captured tensor is tied to one specific graph lifetime and
/// will not satisfy the bound).
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::tensor_ops as T;
/// use scirs2_core::ndarray::{array, ArrayViewD};
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = T::variable(array![0.5_f64, -1.5, 2.0], ctx);
///     // Forward: x^3.  Backward: 3 x^2 * gy.
///     let y = ag::custom_unary_op(
///         "cube",
///         |v: &ArrayViewD<f64>| v.mapv(|e| e * e * e),
///         |gy: &ag::Tensor<'_, f64>, x: &ag::Tensor<'_, f64>, _y: &ag::Tensor<'_, f64>| {
///             Some(T::mul(*gy, T::scalar_mul(T::square(*x), 3.0)))
///         },
///         x,
///         ctx,
///     );
///     let gx = T::grad(&[T::sum_all(y)], &[x])[0];
///     let g = gx.eval(ctx).expect("gradient");
///     // 3 * 0.5^2 = 0.75
///     assert!((g[[0]] - 0.75).abs() < 1e-10);
/// });
/// ```
pub fn custom_unary_op<'g, F, FwdFn, BwdFn>(
    name: &'static str,
    forward_fn: FwdFn,
    backward_fn: BwdFn,
    input: Tensor<'g, F>,
    ctx: &'g Context<'g, F>,
) -> Tensor<'g, F>
where
    F: Float,
    FwdFn: Fn(&ArrayViewD<F>) -> ArrayD<F> + Send + Sync + 'static,
    // The backward closure is **higher-ranked** over the graph lifetime.  An `Op` must be
    // `'static`, so it cannot store a closure tied to one particular `'g`; binding the
    // closure to the caller's `'g` is exactly what forced the previous implementation to
    // give up and pass the cotangent through unchanged, silently ignoring the backward
    // function the caller supplied.  Quantifying over the lifetime instead lets the op
    // hold the closure and call it with whatever graph lifetime the backward pass has.
    BwdFn: for<'graph> Fn(
            &Tensor<'graph, F>,
            &Tensor<'graph, F>,
            &Tensor<'graph, F>,
        ) -> Option<Tensor<'graph, F>>
        + Send
        + Sync
        + 'static,
{
    // We use a wrapper struct to hold the closures.
    //
    // `Send`/`Sync` are derived automatically from the `Fwd`/`Bwd` bounds below plus
    // `PhantomData<F>` (`F: Float` is itself `Send + Sync`), so the two hand-written
    // `unsafe impl`s that used to live here — which promised `Send` without requiring
    // `F: Send` — are gone.
    struct ClosureOp<F: Float, Fwd, Bwd> {
        name: &'static str,
        forward: Fwd,
        backward: Bwd,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float, Fwd, Bwd> op::Op<F> for ClosureOp<F, Fwd, Bwd>
    where
        Fwd: Fn(&ArrayViewD<F>) -> ArrayD<F> + Send + Sync + 'static,
        Bwd: for<'graph> Fn(
                &Tensor<'graph, F>,
                &Tensor<'graph, F>,
                &Tensor<'graph, F>,
            ) -> Option<Tensor<'graph, F>>
            + Send
            + Sync
            + 'static,
    {
        fn name(&self) -> &'static str {
            self.name
        }

        fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
            let input = ctx.input(0);
            let output = (self.forward)(&input);
            ctx.append_output(output);
            Ok(())
        }

        fn grad<'a, 'graph>(&self, ctx: &mut GradientContext<'a, 'graph, F>) {
            // Apply the user-registered backward closure.  This is the entire point of
            // `custom_unary_op`: without it the op reports `d f(x) / dx = 1`.
            let gy = ctx.output_grad();
            let x = ctx.input(0);
            let y = ctx.output();
            let gx = (self.backward)(gy, x, y);
            ctx.append_input_grad(0, gx);
        }
    }

    let op = ClosureOp {
        name,
        forward: forward_fn,
        backward: backward_fn,
        _phantom: std::marker::PhantomData,
    };

    Tensor::builder(ctx).append_input(input, false).build(op)
}

// ---------------------------------------------------------------------------
// SelectiveStopGradient
// ---------------------------------------------------------------------------

/// An operation that selectively stops gradient flow based on a mask.
///
/// Unlike a full stop-gradient (which blocks all gradient flow), this allows
/// gradients to flow through selected dimensions while blocking others.
/// The mask is a boolean array where `true` means "allow gradient" and
/// `false` means "block gradient".
pub struct SelectiveStopGradient {
    /// Per-element mask: true = allow gradient, false = block
    mask: Vec<bool>,
}

impl SelectiveStopGradient {
    /// Create a new selective stop gradient with the given mask.
    ///
    /// # Arguments
    /// * `mask` - Boolean mask. `true` allows gradient flow, `false` blocks it.
    pub fn new(mask: Vec<bool>) -> Self {
        Self { mask }
    }

    /// Create a mask that blocks gradients for specific indices.
    ///
    /// # Arguments
    /// * `size` - Total number of elements
    /// * `blocked_indices` - Indices where gradient should be blocked
    pub fn block_indices(size: usize, blocked_indices: &[usize]) -> Self {
        let mut mask = vec![true; size];
        for &idx in blocked_indices {
            if idx < size {
                mask[idx] = false;
            }
        }
        Self { mask }
    }

    /// Create a mask that only allows gradients for specific indices.
    ///
    /// # Arguments
    /// * `size` - Total number of elements
    /// * `allowed_indices` - Indices where gradient should flow
    pub fn allow_indices(size: usize, allowed_indices: &[usize]) -> Self {
        let mut mask = vec![false; size];
        for &idx in allowed_indices {
            if idx < size {
                mask[idx] = true;
            }
        }
        Self { mask }
    }
}

impl<F: Float> op::Op<F> for SelectiveStopGradient {
    fn name(&self) -> &'static str {
        "SelectiveStopGradient"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Forward pass: identity
        let input = ctx.input(0);
        ctx.append_output(input.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<'_, '_, F>) {
        let gy = ctx.output_grad();

        // Apply mask to gradient: zero out blocked dimensions
        let mask_vals: Vec<F> = self
            .mask
            .iter()
            .map(|&m| if m { F::one() } else { F::zero() })
            .collect();

        let mask_arr = scirs2_core::ndarray::Array1::from(mask_vals).into_dyn();
        let mask_tensor = crate::tensor_ops::convert_to_tensor(mask_arr, ctx.graph());
        let masked_grad = *gy * mask_tensor;

        ctx.append_input_grad(0, Some(masked_grad));
    }
}

/// Apply selective stop-gradient to a tensor.
///
/// # Arguments
/// * `input` - The input tensor
/// * `mask` - Boolean mask: `true` = allow gradient, `false` = block gradient
/// * `ctx` - The autograd context
pub fn selective_stop_gradient<'g, F: Float>(
    input: Tensor<'g, F>,
    mask: Vec<bool>,
    ctx: &'g Context<'g, F>,
) -> Tensor<'g, F> {
    let op = SelectiveStopGradient::new(mask);
    Tensor::builder(ctx).append_input(input, false).build(op)
}

// ---------------------------------------------------------------------------
// ScaleGradient: scale gradients by a constant factor
// ---------------------------------------------------------------------------

/// Operation that scales gradients by a constant factor during backprop.
///
/// This is useful for:
/// - Gradient reversal (factor = -1.0) for domain adaptation
/// - Gradient scaling for multi-task learning
/// - Soft stop-gradient (factor close to 0)
pub struct ScaleGradient<F: Float> {
    scale: F,
}

impl<F: Float> ScaleGradient<F> {
    /// Create a gradient scaling operation.
    ///
    /// # Arguments
    /// * `scale` - Factor to multiply gradients by. Use -1.0 for gradient reversal.
    pub fn new(scale: F) -> Self {
        Self { scale }
    }
}

impl<F: Float> op::Op<F> for ScaleGradient<F> {
    fn name(&self) -> &'static str {
        "ScaleGradient"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        ctx.append_output(input.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<'_, '_, F>) {
        let gy = ctx.output_grad();
        let scaled = *gy * self.scale;
        ctx.append_input_grad(0, Some(scaled));
    }
}

/// Scale gradients flowing through a tensor by a constant factor.
///
/// Forward pass is identity; backward pass multiplies gradient by `scale`.
///
/// # Arguments
/// * `input` - The input tensor
/// * `scale` - Factor to multiply gradients by
/// * `ctx` - The autograd context
///
/// # Common uses
/// - `scale_gradient(x, -1.0, ctx)` for gradient reversal
/// - `scale_gradient(x, 0.1, ctx)` for reduced gradient magnitude
/// - `scale_gradient(x, 0.0, ctx)` for stop-gradient (equivalent)
pub fn scale_gradient<'g, F: Float>(
    input: Tensor<'g, F>,
    scale: F,
    ctx: &'g Context<'g, F>,
) -> Tensor<'g, F> {
    let op = ScaleGradient::new(scale);
    Tensor::builder(ctx).append_input(input, false).build(op)
}

/// Apply gradient reversal to a tensor (for domain adaptation).
///
/// Forward: identity. Backward: negate gradient.
/// This is a convenience wrapper around `scale_gradient(input, -1.0, ctx)`.
pub fn gradient_reversal<'g, F: Float>(
    input: Tensor<'g, F>,
    ctx: &'g Context<'g, F>,
) -> Tensor<'g, F> {
    let neg_one = F::from(-1.0).unwrap_or_else(|| F::zero() - F::one());
    scale_gradient(input, neg_one, ctx)
}

// ---------------------------------------------------------------------------
// DetachOp: explicit graph-level stop gradient
// ---------------------------------------------------------------------------

/// Internal stop-gradient op (identity forward, None gradient backward).
struct DetachOp;

impl<F: Float> op::Op<F> for DetachOp {
    fn name(&self) -> &'static str {
        "Detach"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        ctx.append_output(input.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<'_, '_, F>) {
        ctx.append_input_grad(0, None);
    }
}

/// Detach a tensor from the computation graph, creating a new leaf node.
///
/// This is the graph-level equivalent of `stop_gradient`. The returned tensor
/// has the same value but no gradient connection to its inputs.
pub fn detach<'g, F: Float>(input: Tensor<'g, F>, ctx: &'g Context<'g, F>) -> Tensor<'g, F> {
    Tensor::builder(ctx)
        .append_input(input, false)
        .build(DetachOp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops;
    use crate::tensor_ops::*;
    use std::sync::Arc;

    // --- CustomGradientOp: identity with doubled gradient ---
    struct DoubledGradOp;

    impl CustomGradientOp<f64> for DoubledGradOp {
        fn forward(&self, inputs: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, OpError> {
            Ok(inputs[0].to_owned())
        }

        fn backward<'g>(
            &self,
            output_grad: &Tensor<'g, f64>,
            _saved: &[Tensor<'g, f64>],
            _ctx: &'g crate::graph::Graph<f64>,
        ) -> Vec<Option<Tensor<'g, f64>>> {
            vec![Some(*output_grad * 2.0)]
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "DoubledGrad"
        }
    }

    #[test]
    fn test_custom_op_forward() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = convert_to_tensor(scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0]).into_dyn(), ctx);
            let op = Arc::new(DoubledGradOp);
            let y = custom_op(op, &[x], ctx);

            let result = y.eval(ctx);
            match result {
                Ok(arr) => {
                    let vals = arr.as_slice().unwrap_or(&[]);
                    assert!((vals[0] - 1.0).abs() < 1e-10);
                    assert!((vals[1] - 2.0).abs() < 1e-10);
                    assert!((vals[2] - 3.0).abs() < 1e-10);
                }
                Err(e) => panic!("Forward eval failed: {e:?}"),
            }
        });
    }

    #[test]
    fn test_custom_op_backward_doubled() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let op = Arc::new(DoubledGradOp);
            let y = custom_op(op, &[x], ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate gradient");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            // `sum_all`'s cotangent is uniformly 1 and the registered backward doubles
            // it, so the gradient must be exactly 2 everywhere. A 1.0 here would mean
            // the wrapper fell back to passing the cotangent straight through instead
            // of calling `CustomGradientOp::backward`.
            assert_eq!(grad_vals.len(), 3, "gradient must have the input's shape");
            for &val in grad_vals {
                assert!(
                    (val - 2.0).abs() < 1e-12,
                    "expected exactly 2.0 from the doubling backward rule, got {val}"
                );
            }
        });
    }

    // --- CustomGradientOp: a backward rule that reads its saved tensors ---

    /// Forward is `x^2` and the backward recovers `2x` from `saved_tensors[0]`, so this
    /// verifies the *saved-tensor contract* (`saves_inputs`/`saves_output`), not just
    /// that some backward ran.
    struct SquareFromSavedOp;

    impl CustomGradientOp<f64> for SquareFromSavedOp {
        fn forward(&self, inputs: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, OpError> {
            Ok(inputs[0].mapv(|v| v * v))
        }

        fn backward<'g>(
            &self,
            output_grad: &Tensor<'g, f64>,
            saved: &[Tensor<'g, f64>],
            _ctx: &'g crate::graph::Graph<f64>,
        ) -> Vec<Option<Tensor<'g, f64>>> {
            assert_eq!(
                saved.len(),
                2,
                "saves_inputs() + saves_output() must deliver [x, y]"
            );
            let x = saved[0];
            vec![Some(*output_grad * (x * 2.0))]
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "SquareFromSaved"
        }
    }

    #[test]
    fn test_custom_op_backward_uses_saved_input_tensor() {
        crate::run(|ctx: &mut Context<f64>| {
            let x_arr = scirs2_core::ndarray::arr1(&[1.5, -2.0, 3.25]);
            let x = tensor_ops::variable(x_arr.clone().into_dyn(), ctx);
            let op = Arc::new(SquareFromSavedOp);
            let y = custom_op(op, &[x], ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let grad_arr = grads[0].eval(ctx).expect("Should evaluate gradient");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            assert_eq!(grad_vals.len(), 3);
            for (i, &val) in grad_vals.iter().enumerate() {
                let expected = 2.0 * x_arr[i];
                assert!(
                    (val - expected).abs() < 1e-12,
                    "d(x^2)/dx[{i}] = {val}, expected {expected}"
                );
            }
        });
    }

    // --- custom_unary_op: the closure-based convenience wrapper ---

    /// The registered backward closure must be invoked verbatim.
    ///
    /// The forward is `x^2`, whose true derivative is `2x`, and the registered backward
    /// is the deliberately unrelated `5 * gy`. Exactly 5 therefore proves the caller's
    /// closure ran: 1 would mean it was ignored (the old pass-through), `2x` would mean
    /// something else supplied the rule.
    #[test]
    fn test_custom_unary_op_applies_registered_backward() {
        crate::run(|ctx: &mut Context<f64>| {
            let x_arr = scirs2_core::ndarray::arr1(&[1.5, -2.0, 3.25]);
            let x = tensor_ops::variable(x_arr.clone().into_dyn(), ctx);
            let y = custom_unary_op(
                "square_with_five_times_backward",
                |v: &ArrayViewD<f64>| v.mapv(|e| e * e),
                |gy: &Tensor<'_, f64>, _x: &Tensor<'_, f64>, _y: &Tensor<'_, f64>| Some(*gy * 5.0),
                x,
                ctx,
            );

            // Forward is the plain square.
            let y_arr = y.eval(ctx).expect("forward should evaluate");
            for (i, &v) in y_arr.iter().enumerate() {
                assert!(
                    (v - x_arr[i] * x_arr[i]).abs() < 1e-12,
                    "forward[{i}] = {v}"
                );
            }

            let loss = crate::tensor_ops::reduction::sum_all(y);
            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let grad_arr = grads[0].eval(ctx).expect("Should evaluate gradient");
            assert_eq!(grad_arr.shape(), &[3]);
            for (i, &val) in grad_arr.iter().enumerate() {
                assert!(
                    (val - 5.0).abs() < 1e-12,
                    "custom_unary_op backward `5 * gy` must give exactly 5 at {i}, got {val}"
                );
            }
        });
    }

    /// A closure that reads the forward *output* it was handed.
    #[test]
    fn test_custom_unary_op_backward_reads_output() {
        crate::run(|ctx: &mut Context<f64>| {
            let x_arr = scirs2_core::ndarray::arr1(&[0.3, -0.7, 1.1]);
            let x = tensor_ops::variable(x_arr.clone().into_dyn(), ctx);
            // Forward exp(x); backward y * gy, i.e. the derivative expressed through
            // the output.
            let y = custom_unary_op(
                "exp_via_closure",
                |v: &ArrayViewD<f64>| v.mapv(|e| e.exp()),
                |gy: &Tensor<'_, f64>, _x: &Tensor<'_, f64>, y: &Tensor<'_, f64>| Some(*gy * *y),
                x,
                ctx,
            );
            let loss = crate::tensor_ops::reduction::sum_all(y);
            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let grad_arr = grads[0].eval(ctx).expect("Should evaluate gradient");
            for (i, &val) in grad_arr.iter().enumerate() {
                let expected = x_arr[i].exp();
                assert!(
                    (val - expected).abs() < 1e-10,
                    "d(exp(x))/dx[{i}] = {val}, expected {expected}"
                );
            }
        });
    }

    /// A closure returning `None` must block the gradient rather than fall back to the
    /// pass-through.
    #[test]
    fn test_custom_unary_op_backward_none_blocks_gradient() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = tensor_ops::variable(
                scirs2_core::ndarray::arr1(&[1.5, -2.0, 3.25]).into_dyn(),
                ctx,
            );
            let y = custom_unary_op(
                "square_with_no_backward",
                |v: &ArrayViewD<f64>| v.mapv(|e| e * e),
                |_gy: &Tensor<'_, f64>, _x: &Tensor<'_, f64>, _y: &Tensor<'_, f64>| None,
                x,
                ctx,
            );
            let loss = crate::tensor_ops::reduction::sum_all(y);
            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let grad_arr = grads[0].eval(ctx).expect("Should evaluate gradient");
            for (i, &val) in grad_arr.iter().enumerate() {
                assert!(
                    val.abs() < 1e-12,
                    "a `None` backward must block the gradient, got {val} at {i}"
                );
            }
        });
    }

    // --- Straight-Through Estimator ---
    struct StraightThroughEstimator;

    impl CustomGradientOp<f64> for StraightThroughEstimator {
        fn forward(&self, inputs: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, OpError> {
            Ok(inputs[0].mapv(|v| v.round()))
        }

        fn backward<'g>(
            &self,
            output_grad: &Tensor<'g, f64>,
            _saved: &[Tensor<'g, f64>],
            _ctx: &'g crate::graph::Graph<f64>,
        ) -> Vec<Option<Tensor<'g, f64>>> {
            vec![Some(*output_grad)]
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "STE"
        }
    }

    #[test]
    fn test_straight_through_estimator() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[4]);
            let op = Arc::new(StraightThroughEstimator);
            let y = custom_op(op, &[x], ctx);

            // Forward: round
            let x_val = scirs2_core::ndarray::arr1(&[0.3, 1.7, -0.5, 2.9]);
            let fwd_result = ctx
                .evaluator()
                .push(&y)
                .feed(x, x_val.view().into_dyn())
                .run();
            let fwd_arr = fwd_result[0].as_ref().expect("Forward should work");
            let fwd_vals = fwd_arr.as_slice().unwrap_or(&[]);
            // f64::round is half-away-from-zero: 0.3 -> 0, 1.7 -> 2, -0.5 -> -1, 2.9 -> 3.
            let expected_fwd = [0.0, 2.0, -1.0, 3.0];
            assert_eq!(fwd_vals.len(), 4);
            for (i, (&got, &want)) in fwd_vals.iter().zip(expected_fwd.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-10,
                    "round[{i}] = {got}, want {want}"
                );
            }

            // Backward: STE passes gradient through unchanged
            let loss = crate::tensor_ops::reduction::sum_all(y);
            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let grad_result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();
            let grad_arr = grad_result[0].as_ref().expect("Gradient should work");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            // The true derivative of `round` is 0 almost everywhere; the whole point of
            // the STE is to report exactly 1 instead, and `sum_all`'s cotangent is 1.
            assert_eq!(grad_vals.len(), 4);
            for &val in grad_vals {
                assert!(
                    (val - 1.0).abs() < 1e-12,
                    "the STE must pass the cotangent through unchanged, got {val}"
                );
            }
        });
    }

    #[test]
    fn test_selective_stop_gradient() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[4]);
            // Block gradient for indices 1 and 3
            let mask = vec![true, false, true, false];
            let y = selective_stop_gradient(x, mask, ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            // `sum_all`'s cotangent is uniformly 1, so the mask is reproduced verbatim:
            // allowed entries keep the 1, blocked entries are exactly 0.
            assert_eq!(grad_vals.len(), 4, "Should have 4 gradient elements");
            let expected = [1.0, 0.0, 1.0, 0.0];
            for (i, (&got, &want)) in grad_vals.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-12,
                    "selective_stop_gradient[{i}] = {got}, expected {want}"
                );
            }
        });
    }

    #[test]
    fn test_scale_gradient() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = scale_gradient(x, 0.5, ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            // Forward is the identity, so the cotangent of `sum_all` is 1 and the op
            // must scale it to exactly 0.5.
            assert_eq!(grad_vals.len(), 3);
            for &val in grad_vals {
                assert!(
                    (val - 0.5).abs() < 1e-12,
                    "scale_gradient(0.5) must yield exactly 0.5, got {val}"
                );
            }
        });
    }

    #[test]
    fn test_gradient_reversal() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = gradient_reversal(x, ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 2.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            // Forward is the identity, so the reversal must flip the cotangent 1 to
            // exactly -1 -- the sign is the entire content of this op.
            assert_eq!(grad_vals.len(), 2);
            for &val in grad_vals {
                assert!(
                    (val + 1.0).abs() < 1e-12,
                    "gradient_reversal must yield exactly -1, got {val}"
                );
            }
        });
    }

    #[test]
    fn test_detach() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0;
            let z = super::detach(y, ctx);
            // z has no gradient connection to x:
            //   d(loss)/dx through the z path = 0 (detached)
            //   d(loss)/dx through the direct x path = 1
            // so the total must be exactly 1, not the 3 it would be without `detach`.
            let loss = crate::tensor_ops::reduction::sum_all(z + x);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            assert_eq!(grad_vals.len(), 3);
            for &val in grad_vals {
                assert!(
                    (val - 1.0).abs() < 1e-12,
                    "detach must cut the 2*x path, leaving exactly 1, got {val}"
                );
            }
        });
    }

    #[test]
    fn test_block_indices() {
        let ssg = SelectiveStopGradient::block_indices(5, &[1, 3]);
        assert!(ssg.mask[0]);
        assert!(!ssg.mask[1]);
        assert!(ssg.mask[2]);
        assert!(!ssg.mask[3]);
        assert!(ssg.mask[4]);
    }

    #[test]
    fn test_allow_indices() {
        let ssg = SelectiveStopGradient::allow_indices(5, &[0, 2, 4]);
        assert!(ssg.mask[0]);
        assert!(!ssg.mask[1]);
        assert!(ssg.mask[2]);
        assert!(!ssg.mask[3]);
        assert!(ssg.mask[4]);
    }

    #[test]
    fn test_custom_op_name() {
        let op = DoubledGradOp;
        assert_eq!(op.name(), "DoubledGrad");
        assert!(op.saves_inputs());
        assert!(op.saves_output());
        assert_eq!(op.num_inputs(), 1);
    }

    // -----------------------------------------------------------------------
    // Dispatch-safety regression tests.
    //
    // `CustomGradientWrapper::name()` forwards the *user's* string verbatim (see
    // `impl op::Op<F> for CustomGradientWrapper` above). `gradient.rs`'s backward-pass
    // override table used to be keyed on `Op::name()`; a custom op named after a
    // built-in op (e.g. `"Cond"`, `"Rank"`) would then have been routed through that
    // built-in op's override arm instead of this op's own `backward()`. The table is
    // now keyed on `TypeId` (`Op::concrete_type_id`), which cannot be spoofed by
    // `name()`, so these collisions must have no effect on the computed gradient.
    // -----------------------------------------------------------------------

    /// Reuses the exact `name()` string of the built-in `RankOp` override arm
    /// (`"Rank"`). That arm returns `Some(vec![None; num_inputs])`
    /// *unconditionally* (no downcast at all), so under the old string-keyed table
    /// this collision would have forced the gradient to `None` (-> zero, via
    /// `tensor_ops::grad`'s non-differentiable fallback) for every input, regardless
    /// of what this op's own `backward` computes.
    struct FakeRankNamedOp;

    impl CustomGradientOp<f64> for FakeRankNamedOp {
        fn forward(&self, inputs: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, OpError> {
            Ok(inputs[0].to_owned())
        }

        fn backward<'g>(
            &self,
            output_grad: &Tensor<'g, f64>,
            _saved: &[Tensor<'g, f64>],
            _ctx: &'g crate::graph::Graph<f64>,
        ) -> Vec<Option<Tensor<'g, f64>>> {
            // A distinctive multiplier unrelated to any built-in op's gradient, so a
            // hijacked dispatch (forced `None` -> zero) is trivially distinguishable
            // from the correct answer.
            vec![Some(*output_grad * 3.0)]
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "Rank"
        }
    }

    #[test]
    fn test_custom_gradient_name_collision_with_rank_is_not_hijacked() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let op = Arc::new(FakeRankNamedOp);
            let y = custom_op(op, &[x], ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate gradient");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            // `sum_all`'s cotangent is uniformly 1, so this op's own `backward`
            // (tripling it) must yield uniformly 3.0. The pre-fix behaviour (hijacked
            // by the built-in `Rank` arm) would instead yield 0.0 everywhere.
            assert_eq!(grad_vals.len(), 3);
            for &val in grad_vals {
                assert!(
                    (val - 3.0).abs() < 1e-10,
                    "expected 3.0 from the custom op's own backward rule, got {val}"
                );
            }
        });
    }

    /// Reuses the exact `name()` string of the built-in `CondOp` override arm
    /// (`"Cond"`). That arm downcasts to `tensor_ops::CondOp` to recover the norm
    /// variant; under the old string-keyed table the downcast would have failed
    /// (this is not really a `CondOp`), sending the gradient down the "not the
    /// 2-norm variant" path -> `None` for every input.
    struct FakeCondNamedOp;

    impl CustomGradientOp<f64> for FakeCondNamedOp {
        fn forward(&self, inputs: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, OpError> {
            Ok(inputs[0].to_owned())
        }

        fn backward<'g>(
            &self,
            output_grad: &Tensor<'g, f64>,
            _saved: &[Tensor<'g, f64>],
            _ctx: &'g crate::graph::Graph<f64>,
        ) -> Vec<Option<Tensor<'g, f64>>> {
            vec![Some(*output_grad * 5.0)]
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "Cond"
        }
    }

    #[test]
    fn test_custom_gradient_name_collision_with_cond_is_not_hijacked() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[4]);
            let op = Arc::new(FakeCondNamedOp);
            let y = custom_op(op, &[x], ctx);
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let grads = crate::tensor_ops::grad(&[loss], &[x]);
            let x_val = scirs2_core::ndarray::arr1(&[1.0, -2.0, 3.0, -4.0]);
            let result = ctx
                .evaluator()
                .push(&grads[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            let grad_arr = result[0].as_ref().expect("Should evaluate gradient");
            let grad_vals = grad_arr.as_slice().unwrap_or(&[]);
            assert_eq!(grad_vals.len(), 4);
            for &val in grad_vals {
                assert!(
                    (val - 5.0).abs() < 1e-10,
                    "expected 5.0 from the custom op's own backward rule, got {val}"
                );
            }
        });
    }
}
