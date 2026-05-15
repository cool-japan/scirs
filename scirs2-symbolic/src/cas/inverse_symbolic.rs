//! `cas::inverse_symbolic` — Inverse-Symbolic Calculator.
//!
//! Recovers candidate symbolic forms from an `f64` value using:
//! 1. **Continued-fraction expansion** — produces a rational approximation `p/q`.
//! 2. **Integer-relation detection** — searches combinations of well-known
//!    constants `[1, π, e, ln 2, √2, γ]` for small-integer linear combinations
//!    that reproduce the target value within tolerance.
//!
//! # Example
//!
//! ```
//! use scirs2_symbolic::cas::inverse_symbolic::{recover, RecoverOpts};
//!
//! let candidates = recover(std::f64::consts::PI, &RecoverOpts::default());
//! assert!(!candidates.is_empty());
//! assert!(candidates[0].residual <= 1e-12);
//! ```

use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::op::LoweredOp;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Euler–Mascheroni constant γ ≈ 0.5772156649015329.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Options for the inverse-symbolic recovery algorithm.
#[derive(Clone, Debug)]
pub struct RecoverOpts {
    /// CF truncation: stop when denominator exceeds this value (default 10,000).
    pub max_denominator: u64,
    /// Accept candidate if `|reconstruct − x| ≤ residual_tol` (default 1e-12).
    pub residual_tol: f64,
    /// Cap the output vector (default 10).
    pub max_candidates: usize,
    /// Include `[1, π, e, ln 2, √2, 1/√2, γ]` in the constants table (default true).
    pub use_constants_table: bool,
}

impl Default for RecoverOpts {
    fn default() -> Self {
        Self {
            max_denominator: 10_000,
            residual_tol: 1e-12,
            max_candidates: 10,
            use_constants_table: true,
        }
    }
}

/// A single inverse-symbolic candidate.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// The symbolic expression that reconstructs the input value.
    pub expr: LoweredOp,
    /// `|eval(expr) − x|`.
    pub residual: f64,
    /// Score = `−log10(residual + 1e-300) − 0.5 * tree_size(expr)`.
    ///
    /// Higher is better. A perfect rational or named constant scores ≈ 299.x;
    /// approximate CF rationals score lower.
    pub score: f64,
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Iterative tree-size counter: returns number of nodes (leaves + operators).
fn tree_size_iterative(op: &LoweredOp) -> usize {
    let mut count: usize = 0;
    let mut work: Vec<&LoweredOp> = vec![op];
    while let Some(node) = work.pop() {
        count += 1;
        match node {
            LoweredOp::Const(_) | LoweredOp::Var(_) => {}
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => {
                work.push(a);
                work.push(b);
            }
            LoweredOp::Neg(c)
            | LoweredOp::Exp(c)
            | LoweredOp::Ln(c)
            | LoweredOp::Sin(c)
            | LoweredOp::Cos(c)
            | LoweredOp::Tan(c)
            | LoweredOp::Sinh(c)
            | LoweredOp::Cosh(c)
            | LoweredOp::Tanh(c)
            | LoweredOp::Arcsin(c)
            | LoweredOp::Arccos(c)
            | LoweredOp::Arctan(c)
            | LoweredOp::Arcsinh(c)
            | LoweredOp::Arccosh(c)
            | LoweredOp::Arctanh(c)
            | LoweredOp::Sqrt(c)
            | LoweredOp::Abs(c) => {
                work.push(c);
            }
        }
    }
    count
}

/// Compute score for a candidate.
fn score_candidate(residual: f64, expr: &LoweredOp) -> f64 {
    let size = tree_size_iterative(expr) as f64;
    -residual.abs().log10().max(-300.0) - 0.5 * size
}

/// Build the constants table: returns `(f64 value, LoweredOp expression)` pairs.
///
/// Core set: `[1, π, e, ln 2, √2, γ]` (spec) plus derived reciprocal `1/√2`
/// which appears frequently in signal processing and probability.
fn constants_table() -> [(f64, LoweredOp); 7] {
    [
        (1.0_f64, LoweredOp::Const(1.0)),
        (std::f64::consts::PI, LoweredOp::Const(std::f64::consts::PI)),
        (std::f64::consts::E, LoweredOp::Const(std::f64::consts::E)),
        (2.0_f64.ln(), LoweredOp::Ln(Box::new(LoweredOp::Const(2.0)))),
        (
            2.0_f64.sqrt(),
            LoweredOp::Sqrt(Box::new(LoweredOp::Const(2.0))),
        ),
        // 1/√2 = FRAC_1_SQRT_2 ≈ 0.7071… — common in signal processing.
        (
            1.0 / 2.0_f64.sqrt(),
            LoweredOp::Div(
                Box::new(LoweredOp::Const(1.0)),
                Box::new(LoweredOp::Sqrt(Box::new(LoweredOp::Const(2.0)))),
            ),
        ),
        (EULER_MASCHERONI, LoweredOp::Const(EULER_MASCHERONI)),
    ]
}

/// Wrap an expression `c * base_expr` simplifying `c == 1 → base_expr`,
/// `c == -1 → Neg(base_expr)`, otherwise `Mul(Const(c), base_expr)`.
fn scale_expr(c: i64, base_expr: LoweredOp) -> LoweredOp {
    match c {
        1 => base_expr,
        -1 => LoweredOp::Neg(Box::new(base_expr)),
        _ => LoweredOp::Mul(Box::new(LoweredOp::Const(c as f64)), Box::new(base_expr)),
    }
}

/// Build `c_i * val_i + c_j * val_j` as a `LoweredOp`, collapsing negatives
/// to `Sub` when appropriate for a slightly simpler form.
fn pair_expr(ci: i64, oi: LoweredOp, cj: i64, oj: LoweredOp) -> LoweredOp {
    let ei = scale_expr(ci, oi);
    let ej = scale_expr(cj, oj);
    // Use Sub when the j-term is already negated (starts with Neg) to avoid
    // double negation: a + Neg(b) → a − b.
    match &ej {
        LoweredOp::Neg(inner) => LoweredOp::Sub(Box::new(ei), Box::new(*inner.clone())),
        _ => LoweredOp::Add(Box::new(ei), Box::new(ej)),
    }
}

// -----------------------------------------------------------------------
// Continued-fraction expansion
// -----------------------------------------------------------------------

/// Stern–Brocot / Euclidean continued-fraction expansion.
///
/// Iterative. Returns the best rational convergent `p/q` with `q ≤ max_denom`.
/// Returns `None` if the value itself is the best representation (i.e. `q == 1`
/// from the start).
fn cf_rational(x: f64, max_denom: u64) -> (i64, i64) {
    // Standard three-term recurrence for convergents:
    //   h_n = a_n * h_{n-1} + h_{n-2}
    //   k_n = a_n * k_{n-1} + k_{n-2}
    // Initial seeds: h_{-2}=0, h_{-1}=1, k_{-2}=1, k_{-1}=0.

    let sign: i64 = if x < 0.0 { -1 } else { 1 };
    let x = x.abs();

    let mut h_prev2: i64 = 0;
    let mut h_prev1: i64 = 1;
    let mut k_prev2: i64 = 1;
    let mut k_prev1: i64 = 0;

    // Best convergent so far.
    let mut best_p: i64 = x.round() as i64;
    let mut best_q: i64 = 1;

    let mut rem = x;
    let max_iter = 200usize;

    for _ in 0..max_iter {
        if rem.abs() < 1e-15 {
            break;
        }

        let a = rem.floor() as i64;
        let h_n = a * h_prev1 + h_prev2;
        let k_n = a * k_prev1 + k_prev2;

        if k_n < 0 || k_n as u64 > max_denom {
            break;
        }

        best_p = h_n;
        best_q = k_n.max(1);

        // Advance recurrence.
        h_prev2 = h_prev1;
        h_prev1 = h_n;
        k_prev2 = k_prev1;
        k_prev1 = k_n;

        let frac = rem - rem.floor();
        if frac < 1e-15 {
            break;
        }
        rem = 1.0 / frac;
    }

    (sign * best_p, best_q)
}

// -----------------------------------------------------------------------
// Main public entry point
// -----------------------------------------------------------------------

/// Recover candidate symbolic forms from an `f64` value.
///
/// Returns candidates sorted by score (descending), deduplicated by structural
/// hash, and truncated to `opts.max_candidates`. All returned candidates have
/// `residual ≤ opts.residual_tol`.
///
/// Returns `vec![]` for `NaN` or `Infinity` inputs.
pub fn recover(x: f64, opts: &RecoverOpts) -> Vec<Candidate> {
    // Guard: not a finite real number.
    if !x.is_finite() {
        return vec![];
    }

    let ctx = EvalCtx::new(&[]);
    let mut raw: Vec<LoweredOp> = Vec::new();

    // ------------------------------------------------------------------
    // Phase 1: Continued-fraction expansion → rational candidate.
    // ------------------------------------------------------------------
    {
        let (p, q) = cf_rational(x, opts.max_denominator);

        let candidate_expr = if q == 1 {
            LoweredOp::Const(x)
        } else {
            LoweredOp::Div(
                Box::new(LoweredOp::Const(p as f64)),
                Box::new(LoweredOp::Const(q as f64)),
            )
        };
        raw.push(candidate_expr);

        // Also push plain Const(x) if x rounds exactly to an integer.
        if x == x.round() && q != 1 {
            raw.push(LoweredOp::Const(x));
        }
    }

    // ------------------------------------------------------------------
    // Phase 2: Integer-relation detection over constants table.
    // ------------------------------------------------------------------
    if opts.use_constants_table {
        let table = constants_table();
        let n = table.len();

        // Singles: c_i * val_i, c_i ∈ [-8, 8] \ {0}.
        for (val_i, op_i) in table.iter().take(n) {
            for ci in -8i64..=8 {
                if ci == 0 {
                    continue;
                }
                let approx = (ci as f64) * val_i;
                if (approx - x).abs() <= opts.residual_tol {
                    raw.push(scale_expr(ci, op_i.clone()));
                }
            }
        }

        // Pairs: c_i * val_i + c_j * val_j, i < j, both ∈ [-8, 8] \ {0}.
        for i in 0..n {
            for j in (i + 1)..n {
                let (val_i, ref op_i) = table[i];
                let (val_j, ref op_j) = table[j];
                for ci in -8i64..=8 {
                    if ci == 0 {
                        continue;
                    }
                    for cj in -8i64..=8 {
                        if cj == 0 {
                            continue;
                        }
                        let approx = (ci as f64) * val_i + (cj as f64) * val_j;
                        if (approx - x).abs() <= opts.residual_tol {
                            let expr = pair_expr(ci, op_i.clone(), cj, op_j.clone());
                            raw.push(expr);
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: Score, filter, dedup, sort.
    // ------------------------------------------------------------------
    let mut candidates: Vec<Candidate> = raw
        .into_iter()
        .filter_map(|expr| {
            let val = eval_real(&expr, &ctx).ok()?;
            let residual = (val - x).abs();
            // Filter: only keep candidates within tolerance.
            if residual > opts.residual_tol {
                return None;
            }
            let score = score_candidate(residual, &expr);
            Some(Candidate {
                expr,
                residual,
                score,
            })
        })
        .collect();

    // Sort descending by score.
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    // Deduplicate by structural hash, keeping first seen (best score).
    let mut seen: HashSet<u128> = HashSet::new();
    candidates.retain(|c| seen.insert(c.expr.structural_hash()));

    // Truncate.
    candidates.truncate(opts.max_candidates);

    candidates
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum tolerance used in "exact" tests.
    const TOL: f64 = 1e-12;

    // ----------------------------------------------------------------
    // Test 1: 0.5 → rational 1/2
    // ----------------------------------------------------------------
    #[test]
    fn test_half_rational() {
        let candidates = recover(0.5, &Default::default());
        assert!(
            !candidates.is_empty(),
            "expected at least one candidate for 0.5"
        );
        assert!(
            candidates[0].residual <= TOL,
            "residual too large: {}",
            candidates[0].residual
        );
    }

    // ----------------------------------------------------------------
    // Test 2: 1/3 with limited denominator
    // ----------------------------------------------------------------
    #[test]
    fn test_one_third() {
        let opts = RecoverOpts {
            max_denominator: 100,
            ..Default::default()
        };
        let candidates = recover(1.0 / 3.0, &opts);
        assert!(!candidates.is_empty(), "expected a candidate for 1/3");
        // The best candidate should have residual within 1e-12.
        assert!(
            candidates[0].residual <= TOL,
            "residual for 1/3: {}",
            candidates[0].residual
        );
    }

    // ----------------------------------------------------------------
    // Test 3: π recovers the π constant
    // ----------------------------------------------------------------
    #[test]
    fn test_pi() {
        let candidates = recover(std::f64::consts::PI, &Default::default());
        assert!(!candidates.is_empty(), "expected candidates for π");
        assert!(
            candidates[0].residual <= TOL,
            "top candidate residual for π: {}",
            candidates[0].residual
        );
        // Verify the top expression evaluates to π.
        let ctx = EvalCtx::new(&[]);
        let val = eval_real(&candidates[0].expr, &ctx).expect("eval must succeed");
        assert!(
            (val - std::f64::consts::PI).abs() <= TOL,
            "evaluated value differs from π: {} vs {}",
            val,
            std::f64::consts::PI
        );
    }

    // ----------------------------------------------------------------
    // Test 4: 2π recovers 2·π
    // ----------------------------------------------------------------
    #[test]
    fn test_two_pi() {
        let candidates = recover(2.0 * std::f64::consts::PI, &Default::default());
        assert!(!candidates.is_empty(), "expected candidates for 2π");
        assert!(
            candidates[0].residual <= TOL,
            "top candidate residual for 2π: {}",
            candidates[0].residual
        );
        let ctx = EvalCtx::new(&[]);
        let val = eval_real(&candidates[0].expr, &ctx).expect("eval");
        assert!(
            (val - 2.0 * std::f64::consts::PI).abs() <= TOL,
            "value mismatch: {}",
            val
        );
    }

    // ----------------------------------------------------------------
    // Test 5: e recovers the Euler number
    // ----------------------------------------------------------------
    #[test]
    fn test_euler_e() {
        let candidates = recover(std::f64::consts::E, &Default::default());
        assert!(!candidates.is_empty(), "expected candidates for e");
        assert!(
            candidates[0].residual <= TOL,
            "top candidate residual for e: {}",
            candidates[0].residual
        );
        let ctx = EvalCtx::new(&[]);
        let val = eval_real(&candidates[0].expr, &ctx).expect("eval");
        assert!(
            (val - std::f64::consts::E).abs() <= TOL,
            "value mismatch: {}",
            val
        );
    }

    // ----------------------------------------------------------------
    // Test 6: ln(2) recovers Ln(Const(2.0))
    // ----------------------------------------------------------------
    #[test]
    fn test_ln2() {
        let candidates = recover(2.0_f64.ln(), &Default::default());
        assert!(!candidates.is_empty(), "expected candidates for ln 2");
        assert!(
            candidates[0].residual <= TOL,
            "top candidate residual for ln 2: {}",
            candidates[0].residual
        );
        let ctx = EvalCtx::new(&[]);
        let val = eval_real(&candidates[0].expr, &ctx).expect("eval");
        assert!((val - 2.0_f64.ln()).abs() <= TOL, "value mismatch: {}", val);
    }

    // ----------------------------------------------------------------
    // Test 7: 1/√2 — residual ≤ 1e-10
    // ----------------------------------------------------------------
    #[test]
    fn test_inv_sqrt2() {
        let target = 1.0 / 2.0_f64.sqrt();
        let candidates = recover(target, &Default::default());
        assert!(!candidates.is_empty(), "expected candidates for 1/√2");
        assert!(
            candidates[0].residual <= 1e-10,
            "top candidate residual for 1/√2: {}",
            candidates[0].residual
        );
    }

    // ----------------------------------------------------------------
    // Test 8: NaN → empty
    // ----------------------------------------------------------------
    #[test]
    fn test_nan_empty() {
        let result = recover(f64::NAN, &Default::default());
        assert!(result.is_empty(), "NaN must produce empty vec");
    }

    // ----------------------------------------------------------------
    // Test 9: Infinity → empty
    // ----------------------------------------------------------------
    #[test]
    fn test_infinity_empty() {
        let result = recover(f64::INFINITY, &Default::default());
        assert!(result.is_empty(), "Infinity must produce empty vec");
    }

    // ----------------------------------------------------------------
    // Test 10: 42.0 → Const(42.0), residual = 0
    // ----------------------------------------------------------------
    #[test]
    fn test_integer_42() {
        let candidates = recover(42.0, &Default::default());
        assert!(!candidates.is_empty(), "expected candidate for 42.0");
        // Top candidate must eval to 42 exactly.
        let ctx = EvalCtx::new(&[]);
        let val = eval_real(&candidates[0].expr, &ctx).expect("eval");
        assert!(
            (val - 42.0).abs() < f64::EPSILON * 100.0,
            "evaluated value: {}",
            val
        );
        assert!(
            candidates[0].residual <= TOL,
            "residual for 42.0: {}",
            candidates[0].residual
        );
    }

    // ----------------------------------------------------------------
    // Test 11: obscure decimal — all returned candidates within tol
    // ----------------------------------------------------------------
    #[test]
    fn test_obscure_decimal() {
        let opts = RecoverOpts {
            residual_tol: 1e-15,
            ..Default::default()
        };
        let result = recover(0.123_456_789_012_345, &opts);
        // Either empty or all within tolerance.
        assert!(
            result.iter().all(|c| c.residual <= 1e-15),
            "some candidate exceeds tol=1e-15"
        );
    }

    // ----------------------------------------------------------------
    // Test 12: 1.0 → recovers Const(1.0) or exact rational
    // ----------------------------------------------------------------
    #[test]
    fn test_one() {
        let candidates = recover(1.0, &Default::default());
        assert!(!candidates.is_empty(), "expected candidate for 1.0");
        assert!(
            candidates[0].residual <= TOL,
            "residual for 1.0: {}",
            candidates[0].residual
        );
        let ctx = EvalCtx::new(&[]);
        let val = eval_real(&candidates[0].expr, &ctx).expect("eval");
        assert!((val - 1.0).abs() < f64::EPSILON * 10.0, "value: {}", val);
    }
}
