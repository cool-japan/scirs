//! Closed-form quadratic line-search step computation.
//!
//! For a locally quadratic objective `f`, the optimal step length along a
//! fixed direction `d` is available in closed form:
//!
//! ```text
//! α* = −(∇f · d) / (dᵀ H d)
//! ```
//!
//! where `∇f` is the symbolic gradient and `H` is the symbolic Hessian.
//!
//! The caller takes step `x ← x + α* · d` (not `x − α*·d`).  For gradient
//! descent, pass `d = −∇f` to obtain `α* > 0` (descent step length).
//!
//! # Algorithm
//!
//! 1. Compute partial derivatives `g_i = ∂f/∂x_vars[i]` symbolically.
//! 2. Form `g·d = Σᵢ g_i * d_i` as a `LoweredOp`.
//! 3. Compute the Hessian `H = hessian(f, max_var + 1)`.
//! 4. Form `dᵀHd = Σᵢ Σⱼ d_i * H[x_vars[i]][x_vars[j]] * d_j`.
//! 5. Canonicalize `dᵀHd`. If it collapses to `Const(0)` the direction is
//!    degenerate (flat quadratic — no finite minimizer along `d`).
//! 6. Return `canonicalize(−(g·d) / (dᵀHd))` as a `LoweredOp`.

use crate::cas::canonicalize::canonicalize;
use crate::eml::grad::{grad, hessian};
use crate::eml::op::LoweredOp;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`closed_form_step`].
#[derive(Debug)]
pub enum LineSearchError {
    /// The quadratic curvature along the direction is (symbolically) zero:
    /// `dᵀHd` canonicalised to `Const(0)`. No finite minimiser along `d`.
    DegenerateDirection,
    /// The length of `direction` does not match `x_vars.len()`.
    GradSizeMismatch {
        /// Length of `x_vars`.
        grad_len: usize,
        /// Length of `direction`.
        dir_len: usize,
    },
    /// The Hessian dimension was unexpected (internal guard, should not occur
    /// when the caller constructs inputs from the same `f`).
    HessianDimError,
}

impl std::fmt::Display for LineSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegenerateDirection => {
                write!(f, "degenerate direction: dᵀHd canonicalised to zero")
            }
            Self::GradSizeMismatch { grad_len, dir_len } => {
                write!(
                    f,
                    "dimension mismatch: x_vars has {grad_len} elements but direction has {dir_len}"
                )
            }
            Self::HessianDimError => write!(f, "Hessian dimension does not cover all x_vars"),
        }
    }
}

impl std::error::Error for LineSearchError {}

// ─────────────────────────────────────────────────────────────────────────────
// Core function
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the closed-form optimal step length `α*` for a locally quadratic
/// objective.
///
/// # Arguments
///
/// * `f`         — scalar cost function as a `LoweredOp`
/// * `x_vars`    — indices of the variables to differentiate (e.g. `&[0, 1]`)
/// * `direction` — one symbolic `LoweredOp` per entry of `x_vars` representing
///   the step direction `d`
///
/// # Returns
///
/// A `LoweredOp` representing `α*(x_vars)` symbolically.  Evaluate it with
/// [`crate::eml::eval::eval_real`] at the current point to get a concrete step
/// length.
///
/// # Sign convention
///
/// The caller takes step `x ← x + α* · d`.  For gradient descent, pass
/// `d_i = -g_i` so that `α* > 0` on a strictly convex objective.
///
/// # Errors
///
/// * [`LineSearchError::GradSizeMismatch`] when `direction.len() != x_vars.len()`
/// * [`LineSearchError::DegenerateDirection`] when `dᵀHd` canonicalises to zero
/// * [`LineSearchError::HessianDimError`] when the Hessian does not cover all
///   variable indices in `x_vars`
pub fn closed_form_step(
    f: &LoweredOp,
    x_vars: &[usize],
    direction: &[LoweredOp],
) -> Result<LoweredOp, LineSearchError> {
    let n = x_vars.len();

    // ── Dimension check ───────────────────────────────────────────────────────
    if direction.len() != n {
        return Err(LineSearchError::GradSizeMismatch {
            grad_len: n,
            dir_len: direction.len(),
        });
    }

    // ── Gradient: g_i = ∂f/∂x_vars[i] ───────────────────────────────────────
    let grad_ops: Vec<LoweredOp> = x_vars.iter().map(|&v| grad(f, v)).collect();

    // ── g·d = Σᵢ g_i * d_i ───────────────────────────────────────────────────
    // Build the sum iteratively to avoid recursion blow-up.
    let g_dot_d: LoweredOp = {
        // Each term: g_i * d_i
        let mut terms: Vec<LoweredOp> = grad_ops
            .iter()
            .zip(direction.iter())
            .map(|(g_i, d_i)| LoweredOp::Mul(Box::new(g_i.clone()), Box::new(d_i.clone())))
            .collect();
        // Fold: accumulate sum left-to-right
        let first = terms.remove(0);
        terms
            .into_iter()
            .fold(first, |acc, t| LoweredOp::Add(Box::new(acc), Box::new(t)))
    };

    // ── Hessian: H[i][j] = ∂²f / (∂x_vars[i] ∂x_vars[j]) ───────────────────
    // hessian(f, n_vars) builds an n_vars × n_vars matrix.
    // We need at least max(x_vars) + 1 rows/cols.
    let max_var = x_vars.iter().copied().max().unwrap_or(0);
    let hess_size = max_var + 1;
    let h = hessian(f, hess_size);

    // Guard: all x_vars must be within the Hessian.
    for &v in x_vars {
        if v >= h.len() || v >= h[v].len() {
            return Err(LineSearchError::HessianDimError);
        }
    }

    // ── dᵀHd = Σᵢ Σⱼ d_i * H[x_vars[i]][x_vars[j]] * d_j ───────────────────
    let d_h_d: LoweredOp = {
        let mut terms: Vec<LoweredOp> = Vec::with_capacity(n * n);
        for (i, &vi) in x_vars.iter().enumerate() {
            for (j, &vj) in x_vars.iter().enumerate() {
                let h_ij = h[vi][vj].clone();
                // d_i * H[vi][vj] * d_j
                let inner = LoweredOp::Mul(Box::new(direction[i].clone()), Box::new(h_ij));
                let term = LoweredOp::Mul(Box::new(inner), Box::new(direction[j].clone()));
                terms.push(term);
            }
        }
        // Sum all terms.
        let first = terms.remove(0);
        terms
            .into_iter()
            .fold(first, |acc, t| LoweredOp::Add(Box::new(acc), Box::new(t)))
    };

    // ── Canonicalize dᵀHd and check for degeneracy ────────────────────────────
    let d_h_d_canon = canonicalize(&d_h_d);
    match d_h_d_canon.op() {
        LoweredOp::Const(v) if v.abs() < 1e-14 => {
            return Err(LineSearchError::DegenerateDirection);
        }
        _ => {}
    }

    // ── α* = −(g·d) / (dᵀHd) ─────────────────────────────────────────────────
    let neg_g_dot_d = LoweredOp::Neg(Box::new(g_dot_d));
    let alpha_raw = LoweredOp::Div(Box::new(neg_g_dot_d), Box::new(d_h_d_canon.into_op()));
    let alpha_canon = canonicalize(&alpha_raw);
    Ok(alpha_canon.into_op())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    /// Helper: evaluate a `LoweredOp` at a point given by `bindings`.
    fn eval_at(op: &LoweredOp, bindings: &[f64]) -> f64 {
        let ctx = EvalCtx::new(bindings);
        eval_real(op, &ctx).expect("eval_at: unexpected evaluation error")
    }

    // ── Test 1: f = x², direction = [1], α* eval at x=2 → -2 ─────────────────
    #[test]
    fn test_x_squared_step() {
        // f = x_0^2 = Mul(Var(0), Var(0))
        let f = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        let alpha = closed_form_step(&f, &[0], &[c(1.0)]).expect("closed_form_step");
        // α* = -(2*x)/(2) = -x. At x=2: α* = -2.
        let val = eval_at(&alpha, &[2.0]);
        assert!((val - (-2.0)).abs() < 1e-10, "expected -2.0, got {val}");
    }

    // ── Test 2: f = x₀²+x₁², direction=[1,1], at (1,1) → α*=-1 ──────────────
    #[test]
    fn test_2d_unit_direction() {
        let f = LoweredOp::Add(
            Box::new(LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)))),
            Box::new(LoweredOp::Mul(Box::new(var(1)), Box::new(var(1)))),
        );
        let alpha = closed_form_step(&f, &[0, 1], &[c(1.0), c(1.0)]).expect("closed_form_step");
        // g·d = 2x+2y, dᵀHd = 1*2*1 + 1*2*1 = 4, α* = -(2+2)/4 = -1.
        let val = eval_at(&alpha, &[1.0, 1.0]);
        assert!((val - (-1.0)).abs() < 1e-10, "expected -1.0, got {val}");
    }

    // ── Test 3: f = (x-3)², at x=0, direction=[1] → α*=3 ────────────────────
    #[test]
    fn test_shifted_quadratic_step() {
        // (x-3)^2
        let inner = LoweredOp::Sub(Box::new(var(0)), Box::new(c(3.0)));
        let f = LoweredOp::Mul(Box::new(inner.clone()), Box::new(inner));
        let alpha = closed_form_step(&f, &[0], &[c(1.0)]).expect("closed_form_step");
        // g = 2*(x-3), at x=0: g=-6; dᵀHd=2; α* = -(-6)/2 = 3.
        let val = eval_at(&alpha, &[0.0]);
        assert!((val - 3.0).abs() < 1e-10, "expected 3.0, got {val}");
    }

    // ── Test 4: degenerate direction (direction=[0]) → Err ────────────────────
    #[test]
    fn test_degenerate_direction() {
        // f = x₀ (linear — Hessian is zero), direction = [0]
        let f = var(0);
        let result = closed_form_step(&f, &[0], &[c(0.0)]);
        assert!(
            matches!(result, Err(LineSearchError::DegenerateDirection)),
            "expected DegenerateDirection error, got: {result:?}"
        );
    }

    // ── Test 5: mixed f = x₀²+x₀x₁+x₁², verify descent ─────────────────────
    #[test]
    fn test_mixed_quadratic_descends() {
        // f = x0^2 + x0*x1 + x1^2
        let x0 = var(0);
        let x1 = var(1);
        let f = LoweredOp::Add(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Mul(Box::new(x0.clone()), Box::new(x0.clone()))),
                Box::new(LoweredOp::Mul(Box::new(x0.clone()), Box::new(x1.clone()))),
            )),
            Box::new(LoweredOp::Mul(Box::new(x1.clone()), Box::new(x1.clone()))),
        );

        // direction = [1, 0] from point (1, 0)
        let alpha = closed_form_step(&f, &[0, 1], &[c(1.0), c(0.0)]).expect("closed_form_step");
        let a = eval_at(&alpha, &[1.0, 0.0]);

        // f at start (1,0)
        let ctx_start = EvalCtx::new(&[1.0_f64, 0.0]);
        let f_start = eval_real(&f, &ctx_start).expect("f_start");

        // f at x + α*d = (1 + a*1, 0 + a*0) = (1+a, 0)
        let end_bindings = [1.0 + a, 0.0_f64];
        let ctx_end = EvalCtx::new(&end_bindings);
        let f_end = eval_real(&f, &ctx_end).expect("f_end");

        assert!(
            f_end < f_start,
            "step should descend: f_start={f_start}, f_end={f_end}, alpha={a}"
        );
    }

    // ── Test 6: canonicalize idempotent on result ──────────────────────────────
    #[test]
    fn test_result_is_canonical() {
        let f = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        let alpha = closed_form_step(&f, &[0], &[c(1.0)]).expect("closed_form_step");
        // Canonicalizing the result again should give the same hash.
        let c1 = canonicalize(&alpha);
        let c2 = canonicalize(c1.op());
        assert_eq!(
            c1.hash(),
            c2.hash(),
            "canonicalize should be idempotent on the result"
        );
    }

    // ── GradSizeMismatch error ────────────────────────────────────────────────
    #[test]
    fn test_size_mismatch_error() {
        let f = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        // x_vars has 1 element but direction has 2.
        let result = closed_form_step(&f, &[0], &[c(1.0), c(0.0)]);
        assert!(
            matches!(result, Err(LineSearchError::GradSizeMismatch { .. })),
            "expected GradSizeMismatch, got: {result:?}"
        );
    }
}
