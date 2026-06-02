//! Symbolic-prior regularisation loss for time-series neural networks.
//!
//! This module provides [`SymbolicPriorLoss`], which combines a conventional
//! scalar loss value with a physics/structure-informed regularisation term
//! derived from a [`scirs2_symbolic::neural_priors::SeriesPrior`].
//!
//! # How it works
//!
//! During training, the neural model produces a scalar prediction for the
//! next time step. At the same time, the [`SeriesPrior`] contains a set of
//! discovered symbolic formulas evaluated on the same input window. The
//! regularisation term penalises the neural prediction for *disagreeing with
//! all* prior formulas simultaneously — it only needs to match one of them.
//!
//! ```text
//! total_loss = base_loss + lambda * min_k |neural_pred - prior_pred_k|^2
//! ```
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "symbolic")]
//! # fn main() -> scirs2_neural::error::Result<()> {
//! use scirs2_neural::losses::symbolic_prior_loss::SymbolicPriorLoss;
//! use scirs2_symbolic::neural_priors::SeriesPrior;
//! use scirs2_symbolic::LoweredOp;
//!
//! // Build a trivial prior: formula f(window) = window[0] (Var(0))
//! let prior = SeriesPrior {
//!     formulas: vec![(LoweredOp::Var(0), 1.0)],
//!     variable_names: vec!["y[t-1]".to_string()],
//!     lookback: 1,
//! };
//!
//! let loss_fn = SymbolicPriorLoss::new(prior, 1.0);
//!
//! // Neural pred = 2.0, window = [2.0] → prior_pred = 2.0 → reg = 0.
//! let total = loss_fn.total_loss(0.5, 2.0, &[2.0])?;
//! assert!((total - 0.5).abs() < 1e-10);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "symbolic"))]
//! # fn main() {}
//! ```

use crate::error::{NeuralError, Result};
use scirs2_symbolic::neural_priors::{eval_series_prior, series_prior_regularization, SeriesPrior};

/// Loss combinator that adds symbolic-prior regularisation to a scalar base loss.
///
/// # Fields
///
/// - `lambda` — regularisation strength (`0.0` = disabled, larger values
///   enforce stronger adherence to the symbolic prior).
/// - `prior`  — the discovered [`SeriesPrior`] containing symbolic formulas.
#[derive(Clone, Debug)]
pub struct SymbolicPriorLoss {
    /// Regularisation strength.
    pub lambda: f64,
    /// Symbolic prior discovered from time-series data.
    pub prior: SeriesPrior,
}

impl SymbolicPriorLoss {
    /// Create a new `SymbolicPriorLoss`.
    ///
    /// # Arguments
    ///
    /// - `prior`  — symbolic prior obtained from
    ///   [`scirs2_symbolic::neural_priors::discover_series_prior`].
    /// - `lambda` — regularisation strength (0 = no regularisation).
    pub fn new(prior: SeriesPrior, lambda: f64) -> Self {
        Self { lambda, prior }
    }

    /// Compute the symbolic-prior regularisation term for a single prediction.
    ///
    /// Evaluates all prior formulas at `input_window`, then returns
    /// `lambda * min_k |neural_pred - prior_pred_k|^2`.
    ///
    /// # Arguments
    ///
    /// - `neural_pred`  — scalar prediction produced by the neural model.
    /// - `input_window` — the most recent `prior.lookback` time-series values.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::ComputationError`] if `input_window` length does
    /// not match `prior.lookback`, or if a formula evaluation fails (domain
    /// error, unbound variable, etc.).
    pub fn regularization_term(&self, neural_pred: f64, input_window: &[f64]) -> Result<f64> {
        if self.lambda <= 0.0 || self.prior.formulas.is_empty() {
            return Ok(0.0);
        }
        let prior_preds = eval_series_prior(&self.prior, input_window).map_err(|e| {
            NeuralError::ComputationError(format!("symbolic prior eval failed: {e}"))
        })?;
        Ok(series_prior_regularization(
            neural_pred,
            &prior_preds,
            self.lambda,
        ))
    }

    /// Compute the total loss: `base_loss + regularization_term(neural_pred, input_window)`.
    ///
    /// # Arguments
    ///
    /// - `base_loss`    — the conventional training loss (MSE, MAE, etc.).
    /// - `neural_pred`  — scalar prediction from the neural model.
    /// - `input_window` — most recent `prior.lookback` time-series values.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::regularization_term`].
    pub fn total_loss(
        &self,
        base_loss: f64,
        neural_pred: f64,
        input_window: &[f64],
    ) -> Result<f64> {
        let reg = self.regularization_term(neural_pred, input_window)?;
        Ok(base_loss + reg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_symbolic::LoweredOp;

    fn ar1_prior(lookback: usize) -> SeriesPrior {
        // f(window) = window[lookback - 1] = Var(lookback - 1)
        let var_idx = lookback.saturating_sub(1);
        SeriesPrior {
            formulas: vec![(LoweredOp::Var(var_idx), 1.0)],
            variable_names: (0..lookback)
                .map(|j| format!("y[t-{}]", lookback - j))
                .collect(),
            lookback,
        }
    }

    #[test]
    fn regularization_zero_when_neural_matches_prior() {
        let prior = ar1_prior(2);
        let loss_fn = SymbolicPriorLoss::new(prior, 1.0);
        // window = [_, 3.5]; Var(1) = 3.5; neural_pred = 3.5 → reg = 0.
        let reg = loss_fn
            .regularization_term(3.5, &[1.0, 3.5])
            .expect("should succeed");
        assert!(reg < 1e-20, "reg={reg}");
    }

    #[test]
    fn regularization_penalizes_deviation() {
        let prior = ar1_prior(1); // f(window) = window[0] = Var(0)
        let loss_fn = SymbolicPriorLoss::new(prior, 1.0);
        // window = [2.0]; Var(0) = 2.0; neural_pred = 4.0 → |4-2|^2 * 1.0 = 4.0
        let reg = loss_fn
            .regularization_term(4.0, &[2.0])
            .expect("should succeed");
        assert!((reg - 4.0).abs() < 1e-12, "expected 4.0, got {reg}");
    }

    #[test]
    fn total_loss_adds_regularization() {
        let prior = ar1_prior(1); // Var(0)
        let loss_fn = SymbolicPriorLoss::new(prior, 1.0);
        // base_loss=0.5, neural=3.0, window=[2.0] → reg=(3-2)^2*1=1.0 → total=1.5
        let total = loss_fn
            .total_loss(0.5, 3.0, &[2.0])
            .expect("should succeed");
        assert!((total - 1.5).abs() < 1e-12, "expected 1.5, got {total}");
    }

    #[test]
    fn zero_lambda_gives_base_loss_only() {
        let prior = ar1_prior(1);
        let loss_fn = SymbolicPriorLoss::new(prior, 0.0);
        let total = loss_fn
            .total_loss(0.75, 999.0, &[0.0])
            .expect("should succeed");
        assert!((total - 0.75).abs() < 1e-14, "expected 0.75, got {total}");
    }
}
