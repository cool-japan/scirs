//! Symbolic series expansions: Taylor polynomial and Padé rational approximant.
//!
//! Both [`taylor`] and [`pade`] take a [`LoweredOp`] expression, differentiate
//! it symbolically via [`mod@crate::eml::grad`], evaluate derivatives at the
//! expansion centre numerically, and assemble the result as a new `LoweredOp`
//! in EML form that can be further simplified or canonicalized.
//!
//! # Accuracy and domain
//!
//! Taylor and Padé approximations are mathematically valid only in a
//! neighbourhood of `center`. This module does not attempt radius-of-convergence
//! analysis; that is a future Phase 2 concern.
//!
//! # Design notes
//!
//! - `taylor_coefficients` is a shared private helper to avoid computing
//!   the iterated symbolic gradient twice in `pade`.
//! - `eval_real` requires a bindings slice whose length is ≥ max_var_index + 1.
//!   We build `let mut bindings = vec![0.0; n_vars.max(var_idx + 1)]` to cover
//!   every variable the expression may reference.
//! - `grad` already calls `simplify_op` internally; we do not simplify between
//!   derivative iterations. We do call `simplify_op` once on the assembled result.

use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use crate::eml::simplify::simplify_op;

/// Maximum Taylor/Padé order accepted by this module.
pub const MAX_TAYLOR_ORDER: usize = 20;

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

/// Errors that can arise from series-expansion operations.
#[derive(Debug)]
pub enum SeriesError {
    /// Symbolic differentiation failed (e.g. non-differentiable op at this level).
    GradError(String),
    /// Evaluating a derivative at the expansion centre failed (e.g. singularity).
    EvalError(String),
    /// Padé coefficient matrix is singular — no [n/m] approximant exists.
    SingularSystem,
    /// The requested order exceeds [`MAX_TAYLOR_ORDER`].
    InvalidOrder {
        /// The requested order.
        order: usize,
        /// The maximum accepted order.
        max: usize,
    },
}

impl std::fmt::Display for SeriesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeriesError::GradError(msg) => write!(f, "gradient error: {msg}"),
            SeriesError::EvalError(msg) => write!(f, "evaluation error: {msg}"),
            SeriesError::SingularSystem => write!(f, "Padé coefficient matrix is singular"),
            SeriesError::InvalidOrder { order, max } => {
                write!(f, "order {order} exceeds maximum {max}")
            }
        }
    }
}

impl std::error::Error for SeriesError {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Taylor polynomial approximation of `f` around `center` to `order`-th degree.
///
/// Returns a [`LoweredOp`] representing the polynomial:
/// ```text
/// T_n(x) = Σ_{k=0}^{order}  (f^(k)(center) / k!)  * (x - center)^k
/// ```
/// where `var_idx` is the differentiation variable index (`Var(var_idx)`).
///
/// Uses [`mod@crate::eml::grad`] repeatedly to compute higher-order derivatives.
///
/// # Errors
/// - [`SeriesError::InvalidOrder`] — `order > MAX_TAYLOR_ORDER`
/// - [`SeriesError::EvalError`] — a derivative value is non-finite at `center`
pub fn taylor(
    f: &LoweredOp,
    var_idx: usize,
    center: f64,
    order: usize,
) -> Result<LoweredOp, SeriesError> {
    if order > MAX_TAYLOR_ORDER {
        return Err(SeriesError::InvalidOrder {
            order,
            max: MAX_TAYLOR_ORDER,
        });
    }

    let coeffs = taylor_coefficients(f, var_idx, center, order)?;
    Ok(build_polynomial(&coeffs, var_idx, center))
}

/// Padé rational approximant `[num_order/den_order]` of `f` around `center`.
///
/// Returns a [`LoweredOp`] representing `P(x)/Q(x)` where `P` has degree
/// `num_order` and `Q` has degree `den_order` with `Q(center) = 1`.
///
/// Uses the standard linear-algebra construction: assemble `n + m + 1` Taylor
/// coefficients then solve the Padé linear system via partial-pivoting Gaussian
/// elimination.
///
/// # Errors
/// - [`SeriesError::InvalidOrder`] — `num_order + den_order > MAX_TAYLOR_ORDER`
/// - [`SeriesError::EvalError`] — a Taylor coefficient cannot be evaluated
/// - [`SeriesError::SingularSystem`] — the coefficient matrix is singular
pub fn pade(
    f: &LoweredOp,
    var_idx: usize,
    center: f64,
    num_order: usize,
    den_order: usize,
) -> Result<LoweredOp, SeriesError> {
    let total = num_order + den_order;
    if total > MAX_TAYLOR_ORDER {
        return Err(SeriesError::InvalidOrder {
            order: total,
            max: MAX_TAYLOR_ORDER,
        });
    }

    // n+m+1 Taylor coefficients c_0 .. c_{n+m}
    let c = taylor_coefficients(f, var_idx, center, total)?;

    let n = num_order;
    let m = den_order;
    let size = n + m + 1; // number of unknowns: p_0..p_n, q_1..q_m

    // Build Padé linear system: unknowns = [p_0, ..., p_n, q_1, ..., q_m]
    // Equation k (0-indexed): p_k - Σ_{j=1}^{min(k,m)} c_{k-j} * q_j = c_k
    //   where p_k = 0 for k > n.
    let mut a: Vec<Vec<f64>> = vec![vec![0.0; size]; size];
    let mut b: Vec<f64> = vec![0.0; size];

    for k in 0..size {
        // p_k coefficient — only exists for k ≤ n
        if k <= n {
            a[k][k] = 1.0;
        }
        // q_j coefficients (columns n+1..n+m)
        let max_j = k.min(m);
        for j in 1..=max_j {
            // column for q_j is n + j (but our 0-indexed columns are n+j-1 offset by 1):
            // columns 0..=n hold p_0..p_n; columns n+1..n+m hold q_1..q_m.
            let col = n + j; // q_j → column n+j-1 … let's use n+(j-1) = n+j-1
                             // Actually: q_1 lives at index n+1-1 = n? No. Let's be explicit:
                             // Unknowns layout: index 0..=n → p_0..p_n; index n+1..n+m → q_1..q_m
                             // So q_j (j=1..m) lives at index n+j.
                             // Wait: n+m unknowns total if we put q at n+j: for j=m, index = n+m = size-1. ✓
            let _ = col; // shadow to avoid confusion — recalculate below cleanly
            let q_col = n + j; // q_j at column n+j (j starts at 1, so first q is at n+1)
                               // Equation: p_k + (−c_{k-j}) * q_j = c_k
            a[k][q_col] = -c[k - j];
        }
        b[k] = c[k];
    }

    let sol = solve_linear(&mut a, &mut b)?;

    // Extract p_0..p_n and q_1..q_m
    let p_coeffs: Vec<f64> = sol[0..=n].to_vec();
    let q_coeffs: Vec<f64> = sol[n + 1..].to_vec(); // length m

    // Build P(x) = Σ p_k * (x - center)^k
    let p_op = build_polynomial(&p_coeffs, var_idx, center);

    // Build Q(x) = 1 + Σ_{j=1}^{m} q_j * (x - center)^j
    let q_op = build_denominator(&q_coeffs, var_idx, center);

    // Return P / Q, simplified
    let result = LoweredOp::Div(Box::new(p_op), Box::new(q_op));
    Ok(simplify_op(&result))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the first `order+1` Taylor coefficients of `f` at `center`.
///
/// Returns `c[k] = f^{(k)}(center) / k!` for `k = 0..=order`.
///
/// Uses `var_idx` as the differentiation variable and pads `bindings` so that
/// every variable index referenced by `f` (or its derivatives) is in range.
fn taylor_coefficients(
    f: &LoweredOp,
    var_idx: usize,
    center: f64,
    order: usize,
) -> Result<Vec<f64>, SeriesError> {
    // Build a bindings slice wide enough to cover all referenced variables.
    let n_vars = f.count_vars().max(var_idx + 1);
    let mut bindings = vec![0.0_f64; n_vars];
    bindings[var_idx] = center;

    let mut coeffs: Vec<f64> = Vec::with_capacity(order + 1);
    let mut deriv = f.clone();

    for k in 0..=order {
        // Re-check binding width — each `grad` call can introduce new var indices
        // from deeper rewrites, but in practice count_vars is monotone-decreasing.
        // We rebuild the ctx fresh each iteration to stay safe.
        let needed = deriv.count_vars().max(var_idx + 1);
        if needed > bindings.len() {
            bindings.resize(needed, 0.0);
            // Reassert center value (resize fills with 0.0, which is fine for
            // extra slots; var_idx is already in range and its value is set).
        }
        bindings[var_idx] = center;

        let ctx = EvalCtx::new(&bindings);
        let val = eval_real(&deriv, &ctx).map_err(|e| SeriesError::EvalError(e.to_string()))?;

        if !val.is_finite() {
            return Err(SeriesError::EvalError(format!(
                "derivative order {k} evaluated to non-finite value {val} at center {center}"
            )));
        }

        let ck = val / factorial(k);
        coeffs.push(ck);

        // Compute next derivative (skip for last iteration to save work)
        if k < order {
            deriv = grad(&deriv, var_idx);
        }
    }

    Ok(coeffs)
}

/// Build `Σ coeffs[k] * (x - center)^k` as a `LoweredOp`.
///
/// - k=0: `Const(coeffs[0])`
/// - k=1: `Mul(Const(c1), Sub(Var(var_idx), Const(center)))`
/// - k≥2: `Mul(Const(ck), Pow(Sub(Var(var_idx), Const(center)), Const(k)))`
///
/// All terms are folded by `Add`. Final result is passed through `simplify_op`.
fn build_polynomial(coeffs: &[f64], var_idx: usize, center: f64) -> LoweredOp {
    // x - center (shared base for all terms of degree ≥ 1)
    let x_minus_c = || {
        if center == 0.0 {
            LoweredOp::Var(var_idx)
        } else {
            LoweredOp::Sub(
                Box::new(LoweredOp::Var(var_idx)),
                Box::new(LoweredOp::Const(center)),
            )
        }
    };

    // Build each term, filter near-zero coefficients to keep the tree lean
    let mut terms: Vec<LoweredOp> = Vec::with_capacity(coeffs.len());

    for (k, &ck) in coeffs.iter().enumerate() {
        if ck == 0.0 {
            // Include a Const(0) only for order 0 (ensures we return *something*)
            if k == 0 {
                terms.push(LoweredOp::Const(0.0));
            }
            continue;
        }

        let term = match k {
            0 => LoweredOp::Const(ck),
            1 => LoweredOp::Mul(Box::new(LoweredOp::Const(ck)), Box::new(x_minus_c())),
            _ => LoweredOp::Mul(
                Box::new(LoweredOp::Const(ck)),
                Box::new(LoweredOp::Pow(
                    Box::new(x_minus_c()),
                    Box::new(LoweredOp::Const(k as f64)),
                )),
            ),
        };
        terms.push(term);
    }

    // If all coefficients were zero (except degree-0 placeholder), terms holds
    // a single Const(0.0) — that is the correct answer.
    let raw = fold_sum(terms);
    simplify_op(&raw)
}

/// Build the Padé denominator `Q(x) = 1 + Σ_{j=1}^{m} q_coeffs[j-1] * (x - center)^j`.
///
/// `q_coeffs` is length `m`, representing `q_1 .. q_m`.
fn build_denominator(q_coeffs: &[f64], var_idx: usize, center: f64) -> LoweredOp {
    let x_minus_c = || {
        if center == 0.0 {
            LoweredOp::Var(var_idx)
        } else {
            LoweredOp::Sub(
                Box::new(LoweredOp::Var(var_idx)),
                Box::new(LoweredOp::Const(center)),
            )
        }
    };

    // Start with Q = 1
    let mut terms: Vec<LoweredOp> = vec![LoweredOp::Const(1.0)];

    for (idx, &qj) in q_coeffs.iter().enumerate() {
        let j = idx + 1; // q_1, q_2, ...
        if qj == 0.0 {
            continue;
        }
        let power_part = if j == 1 {
            x_minus_c()
        } else {
            LoweredOp::Pow(Box::new(x_minus_c()), Box::new(LoweredOp::Const(j as f64)))
        };
        let term = LoweredOp::Mul(Box::new(LoweredOp::Const(qj)), Box::new(power_part));
        terms.push(term);
    }

    let raw = fold_sum(terms);
    simplify_op(&raw)
}

/// Fold a `Vec<LoweredOp>` into a left-associative sum via `Add`.
///
/// Panics at compile time are impossible — we always have at least one term
/// because `build_polynomial` guarantees ≥1 element.
fn fold_sum(mut terms: Vec<LoweredOp>) -> LoweredOp {
    if terms.is_empty() {
        return LoweredOp::Const(0.0);
    }
    let first = terms.remove(0);
    terms
        .into_iter()
        .fold(first, |acc, t| LoweredOp::Add(Box::new(acc), Box::new(t)))
}

/// Partial-pivoting Gaussian elimination.
///
/// Mutates `a` (augmented matrix rows) and `b` (right-hand side) in place.
/// Returns the solution vector or [`SeriesError::SingularSystem`] if any
/// pivot is smaller than `1e-14` in absolute value.
///
/// # Implementation note
///
/// The inner loops use raw index pairs to access two different rows of `a`
/// simultaneously in the elimination step (`a[col][k]` and `a[row][k]`).
/// This is the canonical implementation pattern for in-place Gaussian
/// elimination and cannot be trivially reformulated with iterator adapters.
#[allow(clippy::needless_range_loop)]
fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, SeriesError> {
    let n = b.len();

    for col in 0..n {
        // Find pivot row with largest absolute value in this column
        let mut max_val = a[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(SeriesError::SingularSystem);
        }

        // Swap pivot row into position
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];

        // Eliminate below — reads a[col][k] while writing a[row][k];
        // the two rows are disjoint so this is always safe.
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            a[row][col] = 0.0;
            for k in (col + 1)..n {
                let delta = factor * a[col][k];
                a[row][k] -= delta;
            }
            let bv = factor * b[col];
            b[row] -= bv;
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for col in (row + 1)..n {
            s -= a[row][col] * x[col];
        }
        x[row] = s / a[row][row];
    }

    Ok(x)
}

/// Factorial as `f64`.
///
/// Returns `1.0` for `n = 0`. Safe and exact for `n ≤ 20` within `f64`
/// precision; matches the `MAX_TAYLOR_ORDER = 20` bound.
fn factorial(n: usize) -> f64 {
    // Table lookup — exact for 0..=20 and avoids any fp accumulation error.
    const TABLE: [f64; 21] = [
        1.0,                   // 0!
        1.0,                   // 1!
        2.0,                   // 2!
        6.0,                   // 3!
        24.0,                  // 4!
        120.0,                 // 5!
        720.0,                 // 6!
        5040.0,                // 7!
        40320.0,               // 8!
        362880.0,              // 9!
        3628800.0,             // 10!
        39916800.0,            // 11!
        479001600.0,           // 12!
        6227020800.0,          // 13!
        87178291200.0,         // 14!
        1307674368000.0,       // 15!
        20922789888000.0,      // 16!
        355687428096000.0,     // 17!
        6402373705728000.0,    // 18!
        121645100408832000.0,  // 19!
        2432902008176640000.0, // 20!
    ];
    if n < TABLE.len() {
        TABLE[n]
    } else {
        // Fallback for n > 20 (should not happen given InvalidOrder guard)
        (1..=n).map(|i| i as f64).product()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    /// Evaluate `op` at `var_idx = x_val`, all other vars = 0.
    fn eval_at(op: &LoweredOp, var_idx: usize, x_val: f64) -> f64 {
        let n_vars = op.count_vars().max(var_idx + 1);
        let mut bindings = vec![0.0_f64; n_vars];
        bindings[var_idx] = x_val;
        let ctx = EvalCtx::new(&bindings);
        eval_real(op, &ctx).expect("eval_at failed")
    }

    /// Helper: build `exp(Var(0))` as a `LoweredOp`.
    fn exp_x() -> LoweredOp {
        LoweredOp::Exp(Box::new(LoweredOp::Var(0)))
    }

    /// Helper: build `sin(Var(0))` as a `LoweredOp`.
    fn sin_x() -> LoweredOp {
        LoweredOp::Sin(Box::new(LoweredOp::Var(0)))
    }

    #[test]
    fn test_taylor_exp_at_0_order_4() {
        // Taylor T_4(x) of exp(x) around 0 — evaluate at x=0.5
        // Error ≈ 0.5^5/120 ≈ 2.6e-5; tolerance 1e-3
        let f = exp_x();
        let poly = taylor(&f, 0, 0.0, 4).expect("taylor should succeed");
        let approx = eval_at(&poly, 0, 0.5);
        let exact = 0.5_f64.exp();
        assert!(
            (approx - exact).abs() < 1e-3,
            "T_4(exp,0)(0.5) = {approx}, exact = {exact}, diff = {}",
            (approx - exact).abs()
        );
    }

    #[test]
    fn test_taylor_sin_at_0_order_5() {
        // Taylor T_5(x) of sin(x) around 0 — evaluate at x=0.3
        // Error ≈ 0.3^7/5040 ≈ 4e-9; tolerance 1e-5
        let f = sin_x();
        let poly = taylor(&f, 0, 0.0, 5).expect("taylor should succeed");
        let approx = eval_at(&poly, 0, 0.3);
        let exact = 0.3_f64.sin();
        assert!(
            (approx - exact).abs() < 1e-5,
            "T_5(sin,0)(0.3) = {approx}, exact = {exact}, diff = {}",
            (approx - exact).abs()
        );
    }

    #[test]
    fn test_taylor_const() {
        // Taylor of Const(42.0) around 0 to order 3 — all derivatives are 0
        // so result evaluates to 42.0 everywhere.
        let f = LoweredOp::Const(42.0);
        let poly = taylor(&f, 0, 0.0, 3).expect("taylor should succeed");
        let val = eval_at(&poly, 0, 1.5);
        assert!(
            (val - 42.0).abs() < 1e-12,
            "const taylor evaluated to {val}, expected 42.0"
        );
    }

    #[test]
    fn test_taylor_linear() {
        // Taylor of x (Var(0)) around 0 to order 1 — result is x.
        let f = LoweredOp::Var(0);
        let poly = taylor(&f, 0, 0.0, 1).expect("taylor should succeed");
        // Check at x=3.7: should give 3.7
        let val = eval_at(&poly, 0, 3.7);
        assert!(
            (val - 3.7).abs() < 1e-12,
            "linear taylor at 3.7 = {val}, expected 3.7"
        );
    }

    #[test]
    fn test_pade_exp_2_2() {
        // Padé [2/2] of exp(x) around 0 — evaluate at x=0.3
        // Theoretical error bound: x^5/720 ≈ 0.3^5/720 ≈ 3.4e-7; tolerance 1e-5
        // (Evaluating at 0.5 gives error ~7e-5, outside 1e-5 — use 0.3 instead.)
        let f = exp_x();
        let approx_op = pade(&f, 0, 0.0, 2, 2).expect("pade should succeed");
        let approx = eval_at(&approx_op, 0, 0.3);
        let exact = 0.3_f64.exp();
        assert!(
            (approx - exact).abs() < 1e-5,
            "Pade[2/2](exp,0)(0.3) = {approx}, exact = {exact}, diff = {}",
            (approx - exact).abs()
        );
    }

    #[test]
    fn test_pade_sin_3_2() {
        // Padé [3/2] of sin(x) around 0 — evaluate at x=1.0
        // Tolerance 1e-3 (as specified)
        let f = sin_x();
        let approx_op = pade(&f, 0, 0.0, 3, 2).expect("pade should succeed");
        let approx = eval_at(&approx_op, 0, 1.0);
        let exact = 1.0_f64.sin();
        assert!(
            (approx - exact).abs() < 1e-3,
            "Pade[3/2](sin,0)(1.0) = {approx}, exact = {exact}, diff = {}",
            (approx - exact).abs()
        );
    }

    #[test]
    fn test_taylor_invalid_order() {
        let f = exp_x();
        let result = taylor(&f, 0, 0.0, 25);
        match result {
            Err(SeriesError::InvalidOrder { order: 25, max: 20 }) => {} // expected
            other => panic!("expected InvalidOrder{{25, 20}}, got: {other:?}"),
        }
    }

    #[test]
    fn test_pade_singular() {
        // Padé [2/2] of Const(1.0) — all derivatives are 0, the q-block
        // of the coefficient matrix will be all-zeros → singular.
        let f = LoweredOp::Const(1.0);
        let result = pade(&f, 0, 0.0, 2, 2);
        // Must not panic; must return an error (SingularSystem or similar).
        assert!(
            result.is_err(),
            "pade of a constant with non-zero den_order should fail"
        );
    }
}
