//! Symbolic activation function wrapping a [`LoweredOp`] expression.
//!
//! Forward pass evaluates via [`eval_real`] element-wise.
//! Backward pass evaluates a precomputed [`sym_grad`] expression element-wise.
//!
//! # Performance note
//!
//! Per-element interpretation is ~100× slower than SIMD-accelerated activations
//! for large batches. Use hardcoded activations (ReLU, Swish, etc.) for training
//! at scale. Symbolic activations are ideal for experimentation and correctness
//! verification.
//!
//! JIT compilation via `scirs2_symbolic::compile::to_jit` will close this gap
//! in v0.4.5.
//!
//! # Convention
//!
//! The `backward` method receives `input` (not output) as its second argument,
//! consistent with [`crate::activations::Swish`]. The expression must use
//! exactly one variable `Var(0)` representing the scalar input `x`.

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::grad;
use scirs2_symbolic::eml::op::LoweredOp;
use std::fmt::Debug;
use std::sync::Arc;

/// A symbolic activation function defined by a [`LoweredOp`] expression.
///
/// The expression uses a single variable `Var(0)` representing the scalar
/// input `x`. Both the forward value and the derivative are computed by
/// interpreting the expression tree element-wise.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "symbolic")]
/// # fn main() -> scirs2_neural::error::Result<()> {
/// use scirs2_neural::activations::symbolic::SymbolicActivation;
/// use scirs2_neural::activations::Activation;
/// use scirs2_symbolic::eml::op::LoweredOp;
/// use scirs2_core::ndarray::Array;
/// use std::sync::Arc;
///
/// // f(x) = x^2
/// let op = Arc::new(LoweredOp::Mul(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Var(0)),
/// ));
/// let activation = SymbolicActivation::new(op)?;
/// let input = Array::from_vec(vec![2.0_f64]).into_dyn();
/// let output = activation.forward(&input)?;
/// assert!((output[scirs2_core::ndarray::IxDyn(&[0])] - 4.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "symbolic"))]
/// # fn main() {}
/// ```
#[derive(Clone)]
pub struct SymbolicActivation {
    op: Arc<LoweredOp>,
    grad_op: Arc<LoweredOp>,
}

impl SymbolicActivation {
    /// Create a new `SymbolicActivation` from a scalar [`LoweredOp`] expression.
    ///
    /// The expression must use exactly one variable `Var(0)` representing the
    /// scalar input `x`. The symbolic gradient `d(op)/dx` is precomputed at
    /// construction time.
    ///
    /// # Errors
    ///
    /// This constructor always succeeds; the symbolic gradient is computed
    /// structurally and cannot fail. Errors are only raised at evaluation time.
    pub fn new(op: Arc<LoweredOp>) -> Result<Self> {
        let grad_op = grad(op.as_ref(), 0);
        Ok(Self {
            op,
            grad_op: Arc::new(grad_op),
        })
    }
}

impl Debug for SymbolicActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolicActivation").finish()
    }
}

impl<F: Float + Debug + NumAssign> Activation<F> for SymbolicActivation {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut output = Array::zeros(input.raw_dim());
        let mut error: Option<NeuralError> = None;
        Zip::from(&mut output).and(input).for_each(|out, &x| {
            if error.is_some() {
                return;
            }
            let xf = x.to_f64().unwrap_or(0.0);
            match eval_real(self.op.as_ref(), &EvalCtx::new(&[xf])) {
                Ok(v) => {
                    *out = F::from(v).unwrap_or_else(F::nan);
                }
                Err(e) => {
                    error = Some(NeuralError::ComputationError(e.to_string()));
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, IxDyn>,
        input: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let mut grad_input = Array::zeros(input.raw_dim());
        let mut error: Option<NeuralError> = None;
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|out, &gy, &x| {
                if error.is_some() {
                    return;
                }
                let xf = x.to_f64().unwrap_or(0.0);
                match eval_real(self.grad_op.as_ref(), &EvalCtx::new(&[xf])) {
                    Ok(dfdx) => {
                        let dfdx_f = F::from(dfdx).unwrap_or_else(F::nan);
                        *out = gy * dfdx_f;
                    }
                    Err(e) => {
                        error = Some(NeuralError::ComputationError(e.to_string()));
                    }
                }
            });
        if let Some(e) = error {
            return Err(e);
        }
        Ok(grad_input)
    }
}

impl<F: Float + Debug + ScalarOperand + NumAssign> Layer<F> for SymbolicActivation {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        <Self as Activation<F>>::forward(self, input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Layer::backward(input, grad_output) → Activation::backward(grad_output, input)
        <Self as Activation<F>>::backward(self, grad_output, input)
    }

    fn update(&mut self, _learningrate: F) -> Result<()> {
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "SymbolicActivation"
    }
}
