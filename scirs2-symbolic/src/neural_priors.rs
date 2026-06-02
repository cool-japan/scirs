//! SR-as-prior: discover symbolic governing equations from time-series data
//! and use them as regularisation priors for neural network training.
//!
//! # Overview
//!
//! This module bridges [`fn@crate::regression::discover`] and time-series neural
//! models. Given a scalar time series, it:
//!
//! 1. Constructs a sliding-window feature matrix (each row = last `lookback`
//!    values; target = next value `target_lag` steps ahead).
//! 2. Runs symbolic regression to find candidate governing equations.
//! 3. Packages the top-K formulas into a [`SeriesPrior`] that downstream
//!    training loops can query for regularisation.
//!
//! # Variable convention
//!
//! Inside the discovered formulas, `Var(j)` corresponds to
//! `series[t - lookback + j]`, i.e. `Var(0)` is the oldest value in the
//! window and `Var(lookback - 1)` is the most recent. This aligns with the
//! column ordering of the feature matrix passed to [`fn@crate::regression::discover`].
//!
//! # Regularisation semantics
//!
//! [`series_prior_regularization`] returns
//! `lambda * min_k |neural_pred - prior_preds[k]|^2`, meaning the neural
//! model must agree with *at least one* discovered formula — it is not forced
//! to match all of them simultaneously.
//!
//! # Examples
//!
//! ```
//! use scirs2_symbolic::neural_priors::{
//!     discover_series_prior, eval_series_prior, series_prior_regularization,
//! };
//!
//! // Build a simple AR(1)-like series: y[t] = 0.9 * y[t-1]
//! let series: Vec<f64> = (0..100)
//!     .scan(1.0_f64, |state, _| {
//!         let v = *state;
//!         *state *= 0.9;
//!         Some(v)
//!     })
//!     .collect();
//!
//! let prior = discover_series_prior(&series, 3, 1, 3, None)
//!     .expect("discovery failed");
//! assert!(!prior.formulas.is_empty());
//!
//! let window = &series[97..100];
//! let preds = eval_series_prior(&prior, window).expect("eval failed");
//! assert_eq!(preds.len(), prior.formulas.len());
//!
//! // Zero penalty when neural matches the top formula exactly.
//! let reg = series_prior_regularization(preds[0], &preds, 1.0);
//! assert!(reg < 1e-20);
//! ```

use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::LoweredOp;
use crate::error::SymbolicError;
use crate::regression::{discover, SrConfig};
use ndarray::{Array1, Array2};

/// A symbolic prior discovered from a time series.
///
/// Holds the top-K formulas ranked by R² (descending), together with
/// the metadata needed to evaluate them on new windows.
#[derive(Clone, Debug)]
pub struct SeriesPrior {
    /// Top-K discovered formulas: `(formula, r_squared)`.
    ///
    /// Sorted by `r_squared` descending (best first).
    pub formulas: Vec<(LoweredOp, f64)>,
    /// Human-readable names for each input variable in the formulas.
    ///
    /// `variable_names[j]` corresponds to `Var(j)`, which maps to
    /// `series[t - lookback + j]` at evaluation time.
    pub variable_names: Vec<String>,
    /// The sliding-window width used during discovery.
    pub lookback: usize,
}

/// Configuration re-export so callers don't have to import two crates.
pub use crate::regression::SrConfig as SymRegConfig;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Discover a symbolic prior from a scalar time series.
///
/// Constructs a sliding-window feature matrix and runs symbolic regression
/// to find candidate governing equations. The top `max_formulas` results
/// (by R²) are returned as a [`SeriesPrior`].
///
/// # Arguments
///
/// - `series`        — scalar time series values (length T).
/// - `lookback`      — window size: how many past steps form the feature vector.
/// - `target_lag`    — how many steps ahead to predict (`1` = one-step-ahead).
/// - `max_formulas`  — maximum number of formulas to return (K).
/// - `config`        — optional [`SymRegConfig`]; defaults to
///   `SrConfig::default().with_top_n(max_formulas)`.
///
/// # Errors
///
/// Returns [`SymbolicError::DomainError`] if the series is too short to build
/// at least one training window: requires `series.len() >= lookback + target_lag + 1`.
pub fn discover_series_prior(
    series: &[f64],
    lookback: usize,
    target_lag: usize,
    max_formulas: usize,
    config: Option<SymRegConfig>,
) -> Result<SeriesPrior, SymbolicError> {
    // Validate minimum series length.
    let min_len = lookback + target_lag + 1;
    if series.len() < min_len {
        return Err(SymbolicError::DomainError(format!(
            "series too short: need at least {} samples for lookback={} target_lag={}, got {}",
            min_len,
            lookback,
            target_lag,
            series.len()
        )));
    }
    if lookback == 0 {
        return Err(SymbolicError::DomainError(
            "lookback must be at least 1".into(),
        ));
    }
    if target_lag == 0 {
        return Err(SymbolicError::DomainError(
            "target_lag must be at least 1".into(),
        ));
    }

    // Build the feature matrix and target vector.
    // Window t:  row = series[t .. t+lookback], target = series[t + lookback + target_lag - 1]
    let n_windows = series.len() - lookback - target_lag + 1;
    let mut feature_data: Vec<f64> = Vec::with_capacity(n_windows * lookback);
    let mut target_data: Vec<f64> = Vec::with_capacity(n_windows);

    for t in 0..n_windows {
        for j in 0..lookback {
            feature_data.push(series[t + j]);
        }
        target_data.push(series[t + lookback + target_lag - 1]);
    }

    let features = Array2::from_shape_vec((n_windows, lookback), feature_data)
        .map_err(|e| SymbolicError::DomainError(format!("feature matrix shape error: {e}")))?;
    let targets = Array1::from_vec(target_data);

    // Configure SR.
    let effective_max = max_formulas.max(1);
    let cfg = config.unwrap_or_else(|| SrConfig::default().with_top_n(effective_max));

    // Run symbolic regression.
    let discovered = discover(features.view(), targets.view(), &cfg);

    // Extract top-K by R² (SrConfig::top_n already limits the list, but we
    // respect max_formulas as an additional cap and sort by r_squared desc).
    let mut formulas: Vec<(LoweredOp, f64)> = discovered
        .into_iter()
        .take(effective_max)
        .map(|df| (df.op, df.fitness.r_squared))
        .collect();

    // Descending sort by R².
    formulas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build human-readable variable names: "y[t-lookback+j]".
    let variable_names: Vec<String> = (0..lookback)
        .map(|j| {
            let lag = lookback - j;
            format!("y[t-{}]", lag)
        })
        .collect();

    Ok(SeriesPrior {
        formulas,
        variable_names,
        lookback,
    })
}

/// Evaluate all formulas in a [`SeriesPrior`] at a given input window.
///
/// `window` must have length == `prior.lookback`. Returns one predicted value
/// per formula, in the same order as [`SeriesPrior::formulas`].
///
/// # Errors
///
/// Returns [`SymbolicError::DomainError`] if `window.len() != prior.lookback`.
/// Evaluation errors (domain violations, division by zero) from individual
/// formulas are reported as [`SymbolicError::DomainError`].
pub fn eval_series_prior(prior: &SeriesPrior, window: &[f64]) -> Result<Vec<f64>, SymbolicError> {
    if window.len() != prior.lookback {
        return Err(SymbolicError::DomainError(format!(
            "window length mismatch: expected {}, got {}",
            prior.lookback,
            window.len()
        )));
    }
    let ctx = EvalCtx::new(window);
    let mut results = Vec::with_capacity(prior.formulas.len());
    for (op, _r2) in &prior.formulas {
        let val = eval_real(op, &ctx)
            .map_err(|e| SymbolicError::DomainError(format!("formula evaluation error: {e}")))?;
        results.push(val);
    }
    Ok(results)
}

/// Compute a symbolic-prior regularisation loss.
///
/// Returns `lambda * min_k |neural_pred - prior_preds[k]|^2`.
///
/// The minimum is taken over all discovered formulas: the neural model must
/// agree with *at least one* formula, but is free to ignore the others. This
/// is more permissive than penalising against the mean of the priors and
/// avoids destructive interference when the formulas disagree.
///
/// Returns `0.0` if `prior_preds` is empty or `lambda <= 0.0`.
pub fn series_prior_regularization(neural_pred: f64, prior_preds: &[f64], lambda: f64) -> f64 {
    if prior_preds.is_empty() || lambda <= 0.0 {
        return 0.0;
    }
    let min_sq = prior_preds
        .iter()
        .map(|p| (neural_pred - p).powi(2))
        .fold(f64::INFINITY, f64::min);
    lambda * min_sq
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regularization_zero_when_matches() {
        let reg = series_prior_regularization(2.5, &[2.5, 10.0], 1.0);
        assert!(reg < 1e-20, "reg={reg}");
    }

    #[test]
    fn regularization_lambda_when_unit_deviation() {
        // |3.0 - 2.0|^2 = 1.0; lambda=1.0 => 1.0
        let reg = series_prior_regularization(3.0, &[2.0], 1.0);
        assert!((reg - 1.0).abs() < 1e-14, "reg={reg}");
    }

    #[test]
    fn regularization_zero_lambda() {
        let reg = series_prior_regularization(99.0, &[0.0, 1.0], 0.0);
        assert_eq!(reg, 0.0);
    }

    #[test]
    fn regularization_empty_prior() {
        let reg = series_prior_regularization(1.0, &[], 2.0);
        assert_eq!(reg, 0.0);
    }

    #[test]
    fn eval_window_length_mismatch() {
        let prior = SeriesPrior {
            formulas: vec![(LoweredOp::Var(0), 1.0)],
            variable_names: vec!["y[t-1]".into()],
            lookback: 2,
        };
        let result = eval_series_prior(&prior, &[1.0]); // wrong length
        assert!(result.is_err());
    }

    #[test]
    fn eval_ar1_prior_directly() {
        // Directly construct AR(1): Var(2) = y[t-1] (last element of window of 3)
        let prior = SeriesPrior {
            formulas: vec![(LoweredOp::Var(2), 1.0)],
            variable_names: vec!["y[t-3]".into(), "y[t-2]".into(), "y[t-1]".into()],
            lookback: 3,
        };
        let window = [1.0_f64, 2.0, 5.0];
        let preds = eval_series_prior(&prior, &window).expect("eval failed");
        assert_eq!(preds.len(), 1);
        assert!((preds[0] - 5.0).abs() < 1e-12, "got {}", preds[0]);
    }

    #[test]
    fn short_series_returns_error() {
        let short = vec![1.0, 2.0, 3.0];
        let result = discover_series_prior(&short, 5, 1, 3, None);
        assert!(result.is_err());
        if let Err(SymbolicError::DomainError(msg)) = result {
            assert!(msg.contains("too short"), "msg={msg}");
        } else {
            panic!("expected DomainError");
        }
    }
}
