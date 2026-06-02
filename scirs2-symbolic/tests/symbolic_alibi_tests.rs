//! Integration tests for the symbolic ALiBi (Attention with Linear Biases) module.
//!
//! These tests verify that:
//! - Slopes are computed correctly (1-based head index matching `scirs2-neural`)
//! - Symbolic expressions evaluate correctly at concrete positions
//! - The symbolic matrix is symmetric in magnitude
//! - Symbolic and numerical values agree to floating-point precision

use scirs2_symbolic::attention::symbolic_alibi::{
    alibi_bias_expr, alibi_bias_matrix_symbolic, alibi_slope, verify_symbolic_vs_numerical,
};
use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};

/// Test 1: For num_heads=4 and h=0, slope = 2^{-8*1/4} = 2^{-2} = 0.25.
///
/// The task spec names this "head0_is_one_quarter" — for H=4, h=0 (1-based h=1)
/// yields 2^{-2} = 0.25.
#[test]
fn alibi_slope_head0_is_one_quarter() {
    let s = alibi_slope(0, 4);
    match s {
        LoweredOp::Const(v) => {
            assert!(
                (v - 0.25).abs() < 1e-12,
                "expected slope 0.25 for h=0, num_heads=4, got {v}"
            );
        }
        other => panic!("expected LoweredOp::Const, got {other:?}"),
    }
}

/// Test 2: For 8 heads the slopes form a geometric sequence.
///
/// `slope[h+1] / slope[h] = 2^{-1} = 0.5` for all consecutive heads.
/// This matches `AlibiSlopes::compute_slopes(8)` from scirs2-neural.
#[test]
fn alibi_slope_geometric_sequence() {
    let num_heads = 8;
    let slopes: Vec<f64> = (0..num_heads)
        .map(|h| {
            let s = alibi_slope(h, num_heads);
            match s {
                LoweredOp::Const(v) => v,
                other => panic!("expected Const, got {other:?}"),
            }
        })
        .collect();

    assert_eq!(slopes.len(), num_heads);

    // Each slope is half the previous (ratio = 2^{-8/8} = 2^{-1} = 0.5).
    for i in 1..num_heads {
        let ratio = slopes[i] / slopes[i - 1];
        assert!(
            (ratio - 0.5).abs() < 1e-12,
            "slope[{i}]/slope[{}] = {ratio:.14} expected 0.5",
            i - 1
        );
    }

    // Head 0 slope for 8 heads = 2^{-1} = 0.5.
    assert!(
        (slopes[0] - 0.5).abs() < 1e-12,
        "first slope for 8 heads should be 0.5, got {}",
        slopes[0]
    );
}

/// Test 3: alibi_bias_expr with pos_i = pos_j (same Var index) evaluates to 0.
///
/// |i - i| = 0 so the bias is -m_h * 0 = 0 for any head and any position value.
#[test]
fn alibi_bias_expr_zero_on_diagonal() {
    let h = 0;
    let num_heads = 8;

    // Both pos_i and pos_j refer to Var(0) — same variable index.
    let expr = alibi_bias_expr(h, num_heads, LoweredOp::Var(0), LoweredOp::Var(0))
        .expect("alibi_bias_expr with valid params should succeed");

    // Evaluate at position value 5.0.
    let ctx = EvalCtx::new(&[5.0_f64]);
    let val = eval_real(&expr, &ctx).expect("eval_real on Var(0)-Var(0) diagonal");

    assert!(
        val.abs() < 1e-12,
        "diagonal bias (|i-i|=0) must be exactly 0, got {val}"
    );
}

/// Test 4: alibi_bias_expr gives a negative value for distinct positions.
///
/// For i=2, j=5: bias = -m_h * |2 - 5| = -m_h * 3 < 0.
#[test]
fn alibi_bias_expr_negative_off_diagonal() {
    let h = 0;
    let num_heads = 8;

    // pos_i = Var(0), pos_j = Var(1)
    let expr = alibi_bias_expr(h, num_heads, LoweredOp::Var(0), LoweredOp::Var(1))
        .expect("alibi_bias_expr with valid params should succeed");

    // Evaluate with pos_i=2.0, pos_j=5.0 → distance = 3.
    let ctx = EvalCtx::new(&[2.0_f64, 5.0]);
    let val = eval_real(&expr, &ctx).expect("eval_real off-diagonal");

    assert!(
        val < 0.0,
        "off-diagonal bias must be strictly negative, got {val}"
    );

    // Also verify exact value: -0.5 * 3 = -1.5 for h=0, num_heads=8
    let slope = 0.5_f64; // 2^{-1}
    let expected = -slope * 3.0;
    assert!(
        (val - expected).abs() < 1e-12,
        "expected {expected}, got {val}"
    );
}

/// Test 5: Symbolic bias matrix is symmetric in magnitude.
///
/// |bias[i][j]| == |bias[j][i]| for all i, j because |i-j| == |j-i|.
#[test]
fn alibi_bias_matrix_symmetric_in_magnitude() {
    let seq_len = 6;
    let head_idx = 0;
    let num_heads = 8;

    let matrix = alibi_bias_matrix_symbolic(seq_len, head_idx, num_heads).expect("valid params");

    assert_eq!(matrix.len(), seq_len);
    for row in &matrix {
        assert_eq!(row.len(), seq_len);
    }

    let ctx = EvalCtx::new(&[]);

    for (i, row_i) in matrix.iter().enumerate() {
        for (j, entry_ij) in row_i.iter().enumerate() {
            let bij = eval_real(entry_ij, &ctx).expect("eval [i][j]");
            let bji = eval_real(&matrix[j][i], &ctx).expect("eval [j][i]");

            assert!(
                (bij.abs() - bji.abs()).abs() < 1e-12,
                "|bias[{i},{j}]| = {} differs from |bias[{j},{i}]| = {}",
                bij.abs(),
                bji.abs()
            );
        }
    }
}

/// Test 6: verify_symbolic_vs_numerical confirms max absolute diff < 1e-10.
///
/// Compares symbolic matrix entries against the independently-computed
/// reference formula for all heads and all position pairs.
#[test]
fn symbolic_vs_numerical_match() {
    let max_diff = verify_symbolic_vs_numerical(16, 8)
        .expect("verify_symbolic_vs_numerical should succeed for seq_len=16, num_heads=8");

    assert!(
        max_diff < 1e-10,
        "max absolute diff between symbolic and numerical ALiBi = {max_diff:.2e} exceeds 1e-10"
    );
}
