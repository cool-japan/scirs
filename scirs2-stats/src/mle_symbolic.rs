//! Gradient descent MLE with exact symbolic gradient via `scirs2_symbolic::eml::grad`.
//!
//! # Variable convention
//!
//! The symbolic `neg_log_lik` expression uses the following variable index layout:
//!
//! - `Var(0)` .. `Var(n_data - 1)` — the `n_data` observed data points.
//! - `Var(n_data)` .. `Var(n_data + n_params - 1)` — the `n_params` parameters
//!   being estimated.
//!
//! The caller is responsible for constructing the *summed* scalar NLL
//! (i.e. `Σ_i term(x_i, θ)`); this module does not sum over data itself.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use scirs2_core::ndarray::array;
//! use scirs2_symbolic::eml::LoweredOp;
//! use scirs2_stats::mle_symbolic::{fit_mle_symbolic};
//!
//! // One-parameter quadratic NLL: (θ - 5)²  (minimum at θ = 5)
//! let nll = Arc::new(LoweredOp::Pow(
//!     Box::new(LoweredOp::Sub(
//!         Box::new(LoweredOp::Var(0)),
//!         Box::new(LoweredOp::Const(5.0)),
//!     )),
//!     Box::new(LoweredOp::Const(2.0)),
//! ));
//! // No data observations, one parameter at index 0
//! let data: scirs2_core::ndarray::Array1<f64> = scirs2_core::ndarray::Array1::zeros(0);
//! let result = fit_mle_symbolic(&nll, data.view(), array![0.0f64].view(), 200, 1e-6, 0.5)
//!     .expect("converge");
//! assert!((result.params[0] - 5.0).abs() < 1e-4);
//! ```

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_symbolic::eml::{eval_real, grad as sym_grad, EvalCtx, LoweredOp};
use std::sync::Arc;

// ─── Public types ────────────────────────────────────────────────────────────

/// Result returned by [`fit_mle_symbolic`] on success.
#[derive(Debug, Clone)]
pub struct MleSymbolicResult {
    /// Fitted parameter vector (length `n_params`).
    pub params: Array1<f64>,
    /// Negative log-likelihood at the returned parameter vector.
    pub nll_final: f64,
    /// Number of gradient-descent iterations performed.
    pub iters: usize,
    /// `true` when `‖∇NLL‖₂ < tol` was satisfied before `max_iter` was reached.
    pub converged: bool,
}

/// Errors from [`fit_mle_symbolic`].
#[derive(Debug)]
pub enum MleSymbolicError {
    /// An underlying `LoweredOp` evaluation failed (domain violation,
    /// unbound variable, division by zero, etc.).
    EvalError(String),
    /// The backtracking line search could not find a step that decreases the NLL.
    NotConverged,
    /// The length of `init_params` does not match `n_params`.
    DimMismatch {
        /// Expected number of parameters (from `n_params` argument).
        expected: usize,
        /// Actual length supplied.
        got: usize,
    },
}

impl std::fmt::Display for MleSymbolicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvalError(msg) => write!(f, "symbolic evaluation error: {}", msg),
            Self::NotConverged => write!(
                f,
                "backtracking line search failed: no descent step found after 20 halvings"
            ),
            Self::DimMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {} parameters, got {}",
                expected, got
            ),
        }
    }
}

impl std::error::Error for MleSymbolicError {}

// ─── Main function ────────────────────────────────────────────────────────────

/// Fit parameters by maximum likelihood estimation using symbolic gradient descent.
///
/// Minimises `neg_log_lik` (the negative log-likelihood) with respect to the
/// `n_params = init_params.len()` parameters, using exact symbolic gradients
/// and backtracking Armijo line search.
///
/// # Arguments
///
/// - `neg_log_lik` — symbolic NLL expression over variables `[x_0 .. x_{n_data-1},
///   θ_0 .. θ_{n_params-1}]`.
/// - `data` — observed data, bound to `Var(0)` .. `Var(n_data - 1)`.
/// - `init_params` — initial parameter guesses, length `n_params`.
/// - `max_iter` — maximum gradient descent iterations.
/// - `tol` — convergence tolerance on `‖∇NLL‖₂`.
/// - `learning_rate` — initial step size for backtracking line search.
///
/// # Errors
///
/// - [`MleSymbolicError::EvalError`] — symbolic expression evaluation failure.
/// - [`MleSymbolicError::NotConverged`] — backtracking line search failed
///   (all 20 halvings exhausted without NLL decrease).
/// - [`MleSymbolicError::DimMismatch`] — parameter dimension mismatch.
pub fn fit_mle_symbolic(
    neg_log_lik: &Arc<LoweredOp>,
    data: ArrayView1<f64>,
    init_params: ArrayView1<f64>,
    max_iter: usize,
    tol: f64,
    learning_rate: f64,
) -> Result<MleSymbolicResult, MleSymbolicError> {
    let n_data = data.len();
    let n_params = init_params.len();

    // Step 4 (spec): handle max_iter == 0 before any computation
    if max_iter == 0 {
        return Ok(MleSymbolicResult {
            params: init_params.to_owned(),
            nll_final: f64::NAN,
            iters: 0,
            converged: false,
        });
    }

    // Step 2 (spec): precompute symbolic gradient — once, before the loop
    let grad_ops: Vec<LoweredOp> = (0..n_params)
        .map(|k| sym_grad(neg_log_lik.as_ref(), n_data + k))
        .collect();

    // Binding buffer: [x_0 .. x_{n_data-1}, θ_0 .. θ_{n_params-1}]
    let mut bindings: Vec<f64> = vec![0.0; n_data + n_params];

    // Fill data slice once (it never changes)
    for (i, &xi) in data.iter().enumerate() {
        bindings[i] = xi;
    }

    // Mutable parameter vector
    let mut params: Vec<f64> = init_params.to_vec();

    let mut converged = false;
    let mut iters = 0_usize;
    let mut nll_val = f64::NAN;

    // Step 5 (spec): main gradient descent loop
    for _iter in 0..max_iter {
        iters = _iter + 1;

        // Copy current params into binding buffer
        for (k, &pk) in params.iter().enumerate() {
            bindings[n_data + k] = pk;
        }

        // Evaluate NLL
        nll_val = eval_real(neg_log_lik.as_ref(), &EvalCtx::new(&bindings))
            .map_err(|e| MleSymbolicError::EvalError(e.to_string()))?;

        // Guard: non-finite NLL signals a domain issue
        if !nll_val.is_finite() {
            return Err(MleSymbolicError::EvalError(format!(
                "non-finite NLL ({}) at current parameters",
                nll_val
            )));
        }

        // Evaluate gradient components
        let mut g: Vec<f64> = Vec::with_capacity(n_params);
        for grad_op in &grad_ops {
            let gk = eval_real(grad_op, &EvalCtx::new(&bindings))
                .map_err(|e| MleSymbolicError::EvalError(e.to_string()))?;
            g.push(gk);
        }

        // Convergence check: ‖g‖₂ < tol
        let grad_norm = g.iter().map(|&gk| gk * gk).sum::<f64>().sqrt();
        if grad_norm < tol {
            converged = true;
            break;
        }

        // Backtracking Armijo line search (up to 20 halvings).
        //
        // Domain errors (e.g. ln(σ) when σ ≤ 0) are treated as an infinite
        // candidate NLL — the step size is halved further. Only errors on the
        // *current* accepted point (evaluated above) are fatal.
        let mut alpha = learning_rate;
        let mut accepted = false;
        let mut new_params = vec![0.0_f64; n_params];

        for _halving in 0..20 {
            for k in 0..n_params {
                new_params[k] = params[k] - alpha * g[k];
                bindings[n_data + k] = new_params[k];
            }

            let candidate_nll = eval_real(neg_log_lik.as_ref(), &EvalCtx::new(&bindings));

            // A domain error means the candidate is outside the feasible region.
            // Treat it as non-improving and halve further.
            let new_nll = match candidate_nll {
                Ok(v) => v,
                Err(_) => f64::INFINITY,
            };

            if new_nll < nll_val {
                // Accept step
                nll_val = new_nll;
                accepted = true;
                break;
            }

            alpha /= 2.0;
        }

        if !accepted {
            return Err(MleSymbolicError::NotConverged);
        }

        // Commit updated parameters
        params = new_params;
    }

    // Restore final params into bindings so nll_val is consistent
    // (it was updated to the post-step value on the last accepted step)
    let result_params = Array1::from_vec(params);

    Ok(MleSymbolicResult {
        params: result_params,
        nll_final: nll_val,
        iters,
        converged,
    })
}
