//! Symbolic loss function wrapping a [`LoweredOp`] expression.
//!
//! The expression uses two variables: `Var(0)` for the prediction and
//! `Var(1)` for the target. Forward returns the mean over all elements.
//! Backward returns `(1/N) * d(op)/d(Var(0))` evaluated element-wise.
//!
//! # Performance note
//!
//! Per-element interpretation — see [`SymbolicActivation`] doc for details.
//! JIT compilation will close the performance gap in v0.4.5.
//!
//! # Variable convention
//!
//! - `Var(0)`: prediction value
//! - `Var(1)`: target value

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use scirs2_core::ndarray::{Array, IxDyn, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::grad;
use scirs2_symbolic::eml::op::LoweredOp;
use std::fmt::Debug;
use std::sync::Arc;

/// A symbolic loss function defined by a two-variable [`LoweredOp`] expression.
///
/// `Var(0)` is the prediction; `Var(1)` is the target. Forward evaluates the
/// expression at each element pair and returns the mean. Backward evaluates
/// the symbolic gradient w.r.t. `Var(0)` and scales by `1/N`.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "symbolic")]
/// # fn main() -> scirs2_neural::error::Result<()> {
/// use scirs2_neural::losses::symbolic::SymbolicLoss;
/// use scirs2_neural::losses::Loss;
/// use scirs2_symbolic::eml::op::LoweredOp;
/// use scirs2_core::ndarray::Array;
/// use std::sync::Arc;
///
/// // L(p, t) = (p - t)^2
/// let diff = LoweredOp::Sub(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Var(1)),
/// );
/// let op = Arc::new(LoweredOp::Mul(Box::new(diff.clone()), Box::new(diff)));
/// let loss_fn = SymbolicLoss::new(op)?;
/// let preds = Array::from_vec(vec![2.0_f64, 3.0]).into_dyn();
/// let targets = Array::from_vec(vec![1.0_f64, 1.0]).into_dyn();
/// let loss = loss_fn.forward(&preds, &targets)?;
/// // mean((2-1)^2, (3-1)^2) = mean(1, 4) = 2.5
/// assert!((loss - 2.5).abs() < 1e-10);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "symbolic"))]
/// # fn main() {}
/// ```
#[derive(Clone)]
pub struct SymbolicLoss {
    op: Arc<LoweredOp>,
    grad_pred_op: Arc<LoweredOp>,
}

impl SymbolicLoss {
    /// Create a new `SymbolicLoss` from a scalar two-variable [`LoweredOp`].
    ///
    /// `Var(0)` is prediction, `Var(1)` is target. The gradient w.r.t.
    /// `Var(0)` is precomputed at construction time.
    ///
    /// # Errors
    ///
    /// This constructor always succeeds; the gradient is computed structurally.
    /// Errors surface at evaluation time.
    pub fn new(op: Arc<LoweredOp>) -> Result<Self> {
        let grad_pred_op = grad(op.as_ref(), 0);
        Ok(Self {
            op,
            grad_pred_op: Arc::new(grad_pred_op),
        })
    }
}

impl Debug for SymbolicLoss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolicLoss").finish()
    }
}

impl<F: Float + Debug + NumAssign> Loss<F> for SymbolicLoss {
    fn forward(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "SymbolicLoss shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n = predictions.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "empty prediction array in SymbolicLoss::forward".to_string(),
            ));
        }

        let mut sum = F::zero();
        let mut error: Option<NeuralError> = None;

        Zip::from(predictions).and(targets).for_each(|&p, &t| {
            if error.is_some() {
                return;
            }
            let pf = p.to_f64().unwrap_or(0.0);
            let tf = t.to_f64().unwrap_or(0.0);
            match eval_real(self.op.as_ref(), &EvalCtx::new(&[pf, tf])) {
                Ok(v) => {
                    let vf = F::from(v).unwrap_or_else(F::nan);
                    sum += vf;
                }
                Err(e) => {
                    error = Some(NeuralError::ComputationError(e.to_string()));
                }
            }
        });

        if let Some(e) = error {
            return Err(e);
        }

        let n_f = F::from(n).ok_or_else(|| {
            NeuralError::ComputationError("could not convert array length to float".to_string())
        })?;
        Ok(sum / n_f)
    }

    fn backward(
        &self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "SymbolicLoss shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n = predictions.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "empty prediction array in SymbolicLoss::backward".to_string(),
            ));
        }

        let n_f = F::from(n).ok_or_else(|| {
            NeuralError::ComputationError("could not convert array length to float".to_string())
        })?;

        let mut grad_out = Array::zeros(predictions.raw_dim());
        let mut error: Option<NeuralError> = None;

        Zip::from(&mut grad_out)
            .and(predictions)
            .and(targets)
            .for_each(|out, &p, &t| {
                if error.is_some() {
                    return;
                }
                let pf = p.to_f64().unwrap_or(0.0);
                let tf = t.to_f64().unwrap_or(0.0);
                match eval_real(self.grad_pred_op.as_ref(), &EvalCtx::new(&[pf, tf])) {
                    Ok(dldp) => {
                        let dldp_f = F::from(dldp).unwrap_or_else(F::nan);
                        *out = dldp_f / n_f;
                    }
                    Err(e) => {
                        error = Some(NeuralError::ComputationError(e.to_string()));
                    }
                }
            });

        if let Some(e) = error {
            return Err(e);
        }

        Ok(grad_out)
    }
}
