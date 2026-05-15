//! Closed-form quadratic line-search for symbolic objectives.
//!
//! Wraps [`scirs2_symbolic::cas::closed_form_step`] to provide a
//! build-once / evaluate-many interface suitable for the inner loop of
//! gradient-descent and Newton-type optimizers in scirs2-optimize.
//!
//! # Sign convention
//!
//! The caller takes step `x ← x + α* · d` (not `x − α*·d`).
//! For gradient descent, pass `direction[i] = −∇f_i` to obtain `α* > 0`
//! on a strictly convex quadratic.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "symbolic")]
//! # {
//! use scirs2_optimize::symbolic::line_search::SymbolicLineSearch;
//! use scirs2_symbolic::eml::LoweredOp;
//!
//! // f(x) = (x - 5)²
//! let inner = LoweredOp::Sub(
//!     Box::new(LoweredOp::Var(0)),
//!     Box::new(LoweredOp::Const(5.0)),
//! );
//! let f = LoweredOp::Mul(Box::new(inner.clone()), Box::new(inner));
//!
//! // direction = +1 (ascent direction along x)
//! let ls = SymbolicLineSearch::new(&f, &[0], &[LoweredOp::Const(1.0)])
//!     .expect("build");
//! let alpha = ls.eval(&[0.0]).expect("eval"); // from x=0, step = 5.0
//! assert!((alpha - 5.0).abs() < 1e-10);
//! # }
//! ```

use scirs2_symbolic::cas::{closed_form_step, LineSearchError as SymLineSearchError};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::LoweredOp;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from [`SymbolicLineSearch`].
#[derive(Debug)]
pub enum OptLineSearchError {
    /// Error propagated from [`scirs2_symbolic::cas::closed_form_step`].
    Symbolic(SymLineSearchError),
    /// Evaluation of the symbolic `α*` expression failed (domain error,
    /// unbound variable, etc.).
    EvalError(String),
    /// The length of `x` at eval-time does not match the number of variables
    /// the `α*` expression was built for.
    DimensionMismatch {
        /// Expected number of variables (length of `x_vars` at build time).
        expected: usize,
        /// Length of `x` supplied to [`SymbolicLineSearch::eval`].
        got: usize,
    },
}

impl std::fmt::Display for OptLineSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Symbolic(e) => write!(f, "symbolic line-search error: {e}"),
            Self::EvalError(s) => write!(f, "line-search eval error: {s}"),
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: α* expression expects {expected} variables, x has {got}"
            ),
        }
    }
}

impl std::error::Error for OptLineSearchError {}

// ─────────────────────────────────────────────────────────────────────────────
// SymbolicLineSearch
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed symbolic line search for a quadratic objective.
///
/// Build once from `f`, `x_vars`, and `direction`; then call [`Self::eval`]
/// at each new point to get the concrete step length `α*`.
///
/// The symbolic expression for `α*` is built at construction time using
/// [`closed_form_step`]; repeated evaluations only call `eval_real`.
#[derive(Debug)]
pub struct SymbolicLineSearch {
    /// Symbolic expression for the optimal step length `α*(x_vars)`.
    alpha_expr: LoweredOp,
    /// Number of variables (`x_vars.len()`), kept for the dimension check.
    n_vars: usize,
}

impl SymbolicLineSearch {
    /// Build the symbolic `α*` expression from `f` and a fixed direction.
    ///
    /// # Arguments
    ///
    /// * `f`         — scalar objective as a `LoweredOp`
    /// * `x_vars`    — variable indices to differentiate
    /// * `direction` — one symbolic `LoweredOp` per entry of `x_vars`
    ///
    /// # Errors
    ///
    /// Propagates errors from [`closed_form_step`] wrapped in
    /// [`OptLineSearchError::Symbolic`].
    pub fn new(
        f: &LoweredOp,
        x_vars: &[usize],
        direction: &[LoweredOp],
    ) -> Result<Self, OptLineSearchError> {
        let alpha_expr =
            closed_form_step(f, x_vars, direction).map_err(OptLineSearchError::Symbolic)?;
        Ok(Self {
            alpha_expr,
            n_vars: x_vars.len(),
        })
    }

    /// Evaluate the step length `α*` at the concrete point `x`.
    ///
    /// `x` must have at least `n_vars` elements (the maximum variable index
    /// referenced by `x_vars` at build time, + 1).
    ///
    /// # Errors
    ///
    /// * [`OptLineSearchError::DimensionMismatch`] when `x.len() < self.n_vars`
    /// * [`OptLineSearchError::EvalError`] when the symbolic evaluation fails
    pub fn eval(&self, x: &[f64]) -> Result<f64, OptLineSearchError> {
        if x.len() < self.n_vars {
            return Err(OptLineSearchError::DimensionMismatch {
                expected: self.n_vars,
                got: x.len(),
            });
        }
        let ctx = EvalCtx::new(x);
        eval_real(&self.alpha_expr, &ctx).map_err(|e| OptLineSearchError::EvalError(e.to_string()))
    }

    /// Access the raw symbolic `α*` expression (for inspection or re-use).
    pub fn alpha_expr(&self) -> &LoweredOp {
        &self.alpha_expr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    // ── Test 1: f = (x−5)², direction = +1, from x=0 → α* = 5.0 ─────────────
    #[test]
    fn test_shifted_quadratic_step_from_origin() {
        // f(x) = (x - 5)^2
        let inner = LoweredOp::Sub(Box::new(var(0)), Box::new(c(5.0)));
        let f = LoweredOp::Mul(Box::new(inner.clone()), Box::new(inner));

        let ls = SymbolicLineSearch::new(&f, &[0], &[c(1.0)]).expect("build");
        let alpha = ls.eval(&[0.0]).expect("eval");

        // g = 2*(x-5); at x=0: g=-10; dᵀHd = 1*2*1 = 2; α* = -(-10)/2 = 5.0
        assert!((alpha - 5.0).abs() < 1e-10, "expected 5.0, got {alpha}");

        // Verify that taking the step lands at the minimum.
        // x_new = 0 + 5.0 * 1 = 5.0; f(5.0) = 0
        let x_new = 0.0 + alpha * 1.0;
        assert!(
            (x_new - 5.0).abs() < 1e-10,
            "step should land at x=5 (minimum), got x={x_new}"
        );
    }

    // ── Test 2: degenerate direction propagates Err ───────────────────────────
    #[test]
    fn test_degenerate_direction_propagates() {
        // f = x (linear — Hessian is zero everywhere)
        let f = var(0);
        let result = SymbolicLineSearch::new(&f, &[0], &[c(0.0)]);
        assert!(
            matches!(
                result,
                Err(OptLineSearchError::Symbolic(
                    SymLineSearchError::DegenerateDirection
                ))
            ),
            "expected DegenerateDirection, got: {result:?}"
        );
    }
}
