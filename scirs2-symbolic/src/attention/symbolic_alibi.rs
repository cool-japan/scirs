//! Symbolic ALiBi (Attention with Linear Biases) wrapper.
//!
//! Expresses ALiBi biases as symbolic [`LoweredOp`] expression trees,
//! enabling symbolic manipulation, differentiation, and verification of the
//! ALiBi positional bias formula.
//!
//! # ALiBi Formula
//!
//! ALiBi (Press et al. 2021, "Train Short, Test Long") replaces positional
//! encodings with a bias added to attention logits before softmax:
//!
//! ```text
//! score(q_i, k_j) = q_i · k_j / sqrt(d) − m_h * |i − j|
//! ```
//!
//! where the head-specific slopes follow the geometric sequence
//! `m_h = 2^{-8h/H}` for `h = 1 … H` (1-based head index).
//!
//! # Variable convention
//!
//! In symbolic expressions returned by [`alibi_bias_expr`]:
//!
//! - The caller supplies `pos_i` and `pos_j` as any `LoweredOp` subtrees
//!   (typically `LoweredOp::Var(i)` nodes, but arbitrary sub-expressions work).
//! - The resulting tree encodes `-m_h * |pos_i - pos_j|` exactly.
//!
//! # Slope indexing
//!
//! To match the numerical implementation this module uses **1-based** head
//! indices internally: `m_h = 2^{-8h/H}` for `h = 1 … H`. The public API
//! accepts the conventional **0-based** head index `h ∈ [0, num_heads)` and
//! translates to `h+1` internally.

use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::LoweredOp;
use crate::error::SymbolicError;

// ─────────────────────────────────────────────────────────────────────────────
// Core slope computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the ALiBi slope for 0-indexed head `h` out of `num_heads`.
///
/// Uses the formula `m = 2^{-8*(h+1)/H}` where `h+1` is the 1-based head
/// index. This matches the numerical implementation in `scirs2-neural`:
/// the first head (`h=0`) gets slope `2^{-8/H}`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::attention::symbolic_alibi::alibi_slope;
/// use scirs2_symbolic::eml::LoweredOp;
///
/// // For 4 heads, head 0 → slope = 2^{-8*1/4} = 2^{-2} = 0.25
/// let s = alibi_slope(0, 4);
/// assert!(matches!(s, LoweredOp::Const(v) if (v - 0.25).abs() < 1e-12));
/// ```
///
/// # Panics
///
/// Does not panic; returns `LoweredOp::Const(1.0)` when `num_heads == 0`
/// (undefined input) and `LoweredOp::Const(0.0)` when `h >= num_heads`.
pub fn alibi_slope(h: usize, num_heads: usize) -> LoweredOp {
    if num_heads == 0 {
        // Undefined configuration — return a neutral constant.
        return LoweredOp::Const(1.0);
    }
    // 1-based index matching the numerical implementation.
    let slope = compute_slope_f64(h, num_heads);
    LoweredOp::Const(slope)
}

/// Compute the raw f64 ALiBi slope for 0-indexed head `h` out of `num_heads`.
///
/// Returns `0.0` when `num_heads == 0` to avoid division by zero.
fn compute_slope_f64(h: usize, num_heads: usize) -> f64 {
    if num_heads == 0 {
        return 0.0;
    }
    // 1-based: translate h → h+1 to match AlibiSlopes::compute_slopes
    2.0_f64.powf(-8.0 * (h + 1) as f64 / num_heads as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbolic bias expression
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic ALiBi bias for position pair `(pos_i, pos_j)` at head `h`.
///
/// Builds the symbolic expression `-m_h * |pos_i - pos_j|` as a
/// `LoweredOp` tree:
///
/// ```text
/// Mul(Neg(Const(m_h)), Abs(Sub(pos_i, pos_j)))
/// ```
///
/// The caller provides `pos_i` and `pos_j` as any `LoweredOp` subtrees
/// (typically `LoweredOp::Var(k)` nodes).
///
/// # Errors
///
/// Returns [`SymbolicError::DomainError`] when `num_heads == 0` or
/// `h >= num_heads`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::attention::symbolic_alibi::alibi_bias_expr;
/// use scirs2_symbolic::eml::{LoweredOp, eval_real, EvalCtx};
///
/// // Diagonal: |i - i| = 0 → bias = 0
/// let expr = alibi_bias_expr(0, 8, LoweredOp::Var(0), LoweredOp::Var(0))
///     .expect("valid params");
/// let ctx = EvalCtx::new(&[5.0_f64]);
/// let val = eval_real(&expr, &ctx).expect("eval");
/// assert!(val.abs() < 1e-12, "diagonal bias must be 0, got {val}");
/// ```
pub fn alibi_bias_expr(
    h: usize,
    num_heads: usize,
    pos_i: LoweredOp,
    pos_j: LoweredOp,
) -> Result<LoweredOp, SymbolicError> {
    if num_heads == 0 {
        return Err(SymbolicError::DomainError(
            "alibi_bias_expr: num_heads must be > 0".into(),
        ));
    }
    if h >= num_heads {
        return Err(SymbolicError::DomainError(format!(
            "alibi_bias_expr: head index h={h} out of range for num_heads={num_heads}"
        )));
    }

    let slope_val = compute_slope_f64(h, num_heads);

    // Build: Mul(Neg(Const(slope)), Abs(Sub(pos_i, pos_j)))
    let neg_slope = LoweredOp::Neg(Box::new(LoweredOp::Const(slope_val)));
    let diff = LoweredOp::Sub(Box::new(pos_i), Box::new(pos_j));
    let abs_diff = LoweredOp::Abs(Box::new(diff));
    let bias = LoweredOp::Mul(Box::new(neg_slope), Box::new(abs_diff));

    Ok(bias)
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbolic bias matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Build the symbolic ALiBi bias matrix for a given sequence length.
///
/// Returns a `seq_len × seq_len` matrix where each entry is a
/// `LoweredOp::Const` encoding the numeric bias value
/// `-m_{head_idx} * |i - j|` for positions `i` and `j`.
///
/// Positions are concrete at construction time so all entries evaluate to
/// numeric constants. The `LoweredOp::Const` wrapping keeps the symbolic
/// interface uniform (callers can compose or inspect as symbolic values).
///
/// # Layout
///
/// `result[i][j]` is the bias for query position `i` attending to key
/// position `j`.
///
/// # Errors
///
/// Returns [`SymbolicError::DomainError`] when `num_heads == 0` or
/// `head_idx >= num_heads`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::attention::symbolic_alibi::alibi_bias_matrix_symbolic;
/// use scirs2_symbolic::eml::LoweredOp;
///
/// let matrix = alibi_bias_matrix_symbolic(4, 0, 8).expect("valid params");
/// assert_eq!(matrix.len(), 4);
/// assert_eq!(matrix[0].len(), 4);
/// // Diagonal entries are 0.
/// for i in 0..4 {
///     assert!(matches!(matrix[i][i], LoweredOp::Const(v) if v.abs() < 1e-12));
/// }
/// ```
pub fn alibi_bias_matrix_symbolic(
    seq_len: usize,
    head_idx: usize,
    num_heads: usize,
) -> Result<Vec<Vec<LoweredOp>>, SymbolicError> {
    if num_heads == 0 {
        return Err(SymbolicError::DomainError(
            "alibi_bias_matrix_symbolic: num_heads must be > 0".into(),
        ));
    }
    if head_idx >= num_heads {
        return Err(SymbolicError::DomainError(format!(
            "alibi_bias_matrix_symbolic: head_idx={head_idx} out of range for num_heads={num_heads}"
        )));
    }

    let slope = compute_slope_f64(head_idx, num_heads);
    let mut matrix = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let mut row = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let dist = i.abs_diff(j) as f64;
            let bias_val = -slope * dist;
            row.push(LoweredOp::Const(bias_val));
        }
        matrix.push(row);
    }

    Ok(matrix)
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical verification
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that symbolic ALiBi evaluation matches the numerical computation.
///
/// Evaluates [`alibi_bias_matrix_symbolic`] at concrete positions for all
/// heads and compares each bias value against the reference formula
/// `-slope_h * |i - j|` (computed independently, without importing
/// `scirs2-neural`).
///
/// # Returns
///
/// The maximum absolute difference across all heads and all position pairs.
/// A value below `1e-10` indicates floating-point-accurate agreement.
///
/// # Errors
///
/// Returns [`SymbolicError::DomainError`] for invalid arguments or if
/// symbolic evaluation of a `LoweredOp::Const` fails unexpectedly.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::attention::symbolic_alibi::verify_symbolic_vs_numerical;
///
/// let max_diff = verify_symbolic_vs_numerical(8, 4).expect("verification ok");
/// assert!(max_diff < 1e-10, "max_diff={max_diff}");
/// ```
pub fn verify_symbolic_vs_numerical(
    seq_len: usize,
    num_heads: usize,
) -> Result<f64, SymbolicError> {
    if num_heads == 0 {
        return Err(SymbolicError::DomainError(
            "verify_symbolic_vs_numerical: num_heads must be > 0".into(),
        ));
    }

    let mut max_diff = 0.0_f64;
    // Empty bindings: all entries are Const, no Var nodes are needed.
    let ctx = EvalCtx::new(&[]);

    for h in 0..num_heads {
        let slope = compute_slope_f64(h, num_heads);
        let matrix = alibi_bias_matrix_symbolic(seq_len, h, num_heads)?;

        for (i, row) in matrix.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                let symbolic_val = eval_real(entry, &ctx).map_err(|e| {
                    SymbolicError::DomainError(format!(
                        "eval_real failed at head={h} i={i} j={j}: {e}"
                    ))
                })?;

                let dist = i.abs_diff(j) as f64;
                let reference_val = -slope * dist;

                let diff = (symbolic_val - reference_val).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }

    Ok(max_diff)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alibi_slope_is_const() {
        let s = alibi_slope(0, 4);
        assert!(
            matches!(s, LoweredOp::Const(_)),
            "expected Const, got {s:?}"
        );
    }

    #[test]
    fn alibi_slope_head0_four_heads() {
        // h=0 (1-based h=1): slope = 2^{-8/4} = 2^{-2} = 0.25
        let s = alibi_slope(0, 4);
        if let LoweredOp::Const(v) = s {
            assert!((v - 0.25).abs() < 1e-12, "expected 0.25 got {v}");
        } else {
            panic!("expected Const");
        }
    }

    #[test]
    fn alibi_slope_head0_eight_heads() {
        // h=0 (1-based h=1): slope = 2^{-8/8} = 2^{-1} = 0.5
        let s = alibi_slope(0, 8);
        if let LoweredOp::Const(v) = s {
            assert!((v - 0.5).abs() < 1e-12, "expected 0.5 got {v}");
        } else {
            panic!("expected Const");
        }
    }

    #[test]
    fn bias_expr_zero_for_equal_positions() {
        // When pos_i == pos_j (same Var index), bias must be 0.
        let expr =
            alibi_bias_expr(0, 8, LoweredOp::Var(0), LoweredOp::Var(0)).expect("valid params");
        let ctx = EvalCtx::new(&[5.0_f64]);
        let val = eval_real(&expr, &ctx).expect("eval ok");
        assert!(val.abs() < 1e-12, "diagonal bias must be 0, got {val}");
    }

    #[test]
    fn bias_expr_negative_for_distant_positions() {
        let expr =
            alibi_bias_expr(0, 8, LoweredOp::Var(0), LoweredOp::Var(1)).expect("valid params");
        let ctx = EvalCtx::new(&[2.0_f64, 5.0]);
        let val = eval_real(&expr, &ctx).expect("eval ok");
        assert!(val < 0.0, "off-diagonal bias must be negative, got {val}");
    }

    #[test]
    fn bias_matrix_diagonal_is_zero() {
        let mat = alibi_bias_matrix_symbolic(6, 0, 8).expect("valid params");
        for (i, row) in mat.iter().enumerate() {
            if let LoweredOp::Const(v) = row[i] {
                assert!(v.abs() < 1e-12, "diagonal [{i},{i}] should be 0, got {v}");
            } else {
                panic!("expected Const at [{i},{i}]");
            }
        }
    }

    #[test]
    fn verify_matches_numerical() {
        let max_diff = verify_symbolic_vs_numerical(16, 8).expect("verification ok");
        assert!(max_diff < 1e-10, "max_diff={max_diff} exceeds tolerance");
    }
}
