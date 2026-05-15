//! Integration tests for the `symbolic` feature of `scirs2-linalg`.

#![cfg(feature = "symbolic")]

use scirs2_core::ndarray::{arr1, arr2, Array2};
use scirs2_linalg::{
    condition_number_symbolic, det_symbolic, eigenvalues_symbolic_2x2, SymbolicLinalgError,
};
use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
use std::sync::Arc;

fn const_op(v: f64) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Const(v))
}
fn var_op(i: usize) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Var(i))
}

// ─────────────────────── det tests ───────────────────────────────────────

#[test]
fn det_2x2_diagonal_matches_product() {
    // [[a, 0], [0, d]] → det = a * d; at (2, 3) → 6
    let zero = const_op(0.0);
    let mat = Array2::from_shape_fn((2, 2), |(r, c)| match (r, c) {
        (0, 0) => var_op(0),
        (1, 1) => var_op(1),
        _ => Arc::clone(&zero),
    });
    let expr = det_symbolic(mat.view()).expect("det");
    let val = eval_real(&expr, &EvalCtx::new(&[2.0, 3.0])).expect("eval");
    assert!((val - 6.0).abs() < 1e-12, "expected 6.0, got {val}");
}

#[test]
fn det_2x2_general() {
    // [[a,b],[c,d]] with Var(0)..Var(3) at (1,2,3,4) → 1*4 - 2*3 = -2
    let mat = Array2::from_shape_fn((2, 2), |(r, c)| var_op(r * 2 + c));
    let expr = det_symbolic(mat.view()).expect("det");
    let val = eval_real(&expr, &EvalCtx::new(&[1.0, 2.0, 3.0, 4.0])).expect("eval");
    assert!((val - (-2.0)).abs() < 1e-12, "expected -2.0, got {val}");
}

#[test]
fn det_3x3_diagonal() {
    // [[2,0,0],[0,3,0],[0,0,5]] → det = 30
    let zero = const_op(0.0);
    let diag = [2.0_f64, 3.0, 5.0];
    let mat = Array2::from_shape_fn((3, 3), |(r, c)| {
        if r == c {
            const_op(diag[r])
        } else {
            Arc::clone(&zero)
        }
    });
    let expr = det_symbolic(mat.view()).expect("det");
    let val = eval_real(&expr, &EvalCtx::new(&[])).expect("eval");
    assert!((val - 30.0).abs() < 1e-10, "expected 30.0, got {val}");
}

#[test]
fn det_3x3_known() {
    // [[1,2,0],[3,4,0],[0,0,5]] → (1*4 - 2*3) * 5 = -10
    let entries = [[1.0_f64, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
    let mat = Array2::from_shape_fn((3, 3), |(r, c)| const_op(entries[r][c]));
    let expr = det_symbolic(mat.view()).expect("det");
    let val = eval_real(&expr, &EvalCtx::new(&[])).expect("eval");
    assert!((val - (-10.0)).abs() < 1e-10, "expected -10.0, got {val}");
}

#[test]
fn det_4x4_block_diagonal() {
    // Diagonal 4×4 with entries Var(0)..Var(3); at (2,3,4,5) → 2*3*4*5 = 120
    let zero = const_op(0.0);
    let mat = Array2::from_shape_fn(
        (4, 4),
        |(r, c)| {
            if r == c {
                var_op(r)
            } else {
                Arc::clone(&zero)
            }
        },
    );
    let expr = det_symbolic(mat.view()).expect("det");
    let val = eval_real(&expr, &EvalCtx::new(&[2.0, 3.0, 4.0, 5.0])).expect("eval");
    assert!((val - 120.0).abs() < 1e-8, "expected 120.0, got {val}");
}

#[test]
fn det_5x5_returns_unsupported() {
    let mat = Array2::from_elem((5, 5), const_op(1.0));
    match det_symbolic(mat.view()) {
        Err(SymbolicLinalgError::Unsupported { n: 5, max: 4 }) => {}
        other => panic!("expected Unsupported {{n:5, max:4}}, got {other:?}"),
    }
}

#[test]
fn det_non_square_returns_err() {
    let mat = Array2::from_elem((2, 3), const_op(1.0));
    match det_symbolic(mat.view()) {
        Err(SymbolicLinalgError::NotSquare { rows: 2, cols: 3 }) => {}
        other => panic!("expected NotSquare {{rows:2, cols:3}}, got {other:?}"),
    }
}

// ─────────────────── eigenvalue tests ────────────────────────────────────

#[test]
fn eigenvalues_2x2_symmetric() {
    // [[a,1],[1,a]] at a=3: λ+ = 4, λ- = 2
    let one = const_op(1.0);
    let mat = Array2::from_shape_fn(
        (2, 2),
        |(r, c)| {
            if r == c {
                var_op(0)
            } else {
                Arc::clone(&one)
            }
        },
    );
    let [lp, lm] = eigenvalues_symbolic_2x2(mat.view()).expect("eig");
    let ctx = EvalCtx::new(&[3.0]);
    let vp = eval_real(&lp, &ctx).expect("eval λ+");
    let vm = eval_real(&lm, &ctx).expect("eval λ-");
    assert!((vp - 4.0).abs() < 1e-10, "λ+ expected 4.0, got {vp}");
    assert!((vm - 2.0).abs() < 1e-10, "λ- expected 2.0, got {vm}");
}

#[test]
fn eigenvalues_2x2_complex_at_point_returns_err() {
    // [[0,-1],[1,0]] → discriminant = 0² - 4*1 = -4 → Sqrt(-4) → Err
    let mat = Array2::from_shape_fn((2, 2), |(r, c)| {
        let v = match (r, c) {
            (0, 1) => -1.0,
            (1, 0) => 1.0,
            _ => 0.0,
        };
        const_op(v)
    });
    let [lp, _lm] = eigenvalues_symbolic_2x2(mat.view()).expect("eig");
    let ctx = EvalCtx::new(&[]);
    assert!(
        eval_real(&lp, &ctx).is_err(),
        "expected Err when evaluating complex eigenvalue"
    );
}

#[test]
fn eigenvalues_non_2x2_returns_err() {
    let mat = Array2::from_elem((3, 3), const_op(1.0));
    match eigenvalues_symbolic_2x2(mat.view()) {
        Err(SymbolicLinalgError::Unsupported { n: 3, max: 2 }) => {}
        other => panic!("expected Unsupported {{n:3, max:2}}, got {other:?}"),
    }
}

// ─────────────────── condition number tests ──────────────────────────────

#[test]
fn condition_number_2x2_diagonal_known() {
    // [[a,1],[1,a]] at a=3 → eigenvalues 4 and 2 → cond_2 = 2.0
    let one = const_op(1.0);
    let mat = Array2::from_shape_fn(
        (2, 2),
        |(r, c)| {
            if r == c {
                var_op(0)
            } else {
                Arc::clone(&one)
            }
        },
    );
    let kappa = condition_number_symbolic(mat.view(), arr1(&[3.0_f64]).view()).expect("cond");
    assert!((kappa - 2.0).abs() < 1e-6, "expected 2.0, got {kappa}");
}

#[test]
fn condition_number_matches_numerical_baseline() {
    // [[Var(0), 0.5], [0.5, Var(1)]] at [2.0, 3.0] → [[2,0.5],[0.5,3]]
    let half = const_op(0.5);
    let mat = Array2::from_shape_fn((2, 2), |(r, c)| match (r, c) {
        (0, 0) => var_op(0),
        (1, 1) => var_op(1),
        _ => Arc::clone(&half),
    });
    let symbolic_kappa =
        condition_number_symbolic(mat.view(), arr1(&[2.0_f64, 3.0]).view()).expect("symbolic cond");

    let numeric = arr2(&[[2.0_f64, 0.5], [0.5, 3.0]]);
    let numeric_kappa = scirs2_linalg::cond(&numeric.view(), None, None).expect("numeric cond");

    assert!(
        (symbolic_kappa - numeric_kappa).abs() < 1e-8,
        "symbolic={symbolic_kappa}, numeric={numeric_kappa}"
    );
}
