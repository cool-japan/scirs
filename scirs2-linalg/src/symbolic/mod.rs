//! Symbolic linear algebra — exact symbolic expressions for determinants,
//! eigenvalues, and condition numbers.
//!
//! Functions in this module operate on matrices whose entries are
//! [`LoweredOp`] expression trees (from `scirs2-symbolic`). They produce new
//! `LoweredOp` trees that represent the mathematical result symbolically, so
//! the caller can differentiate, simplify, or evaluate the output at an
//! arbitrary point.
//!
//! # Feature gate
//!
//! This module is compiled only when the `symbolic` Cargo feature is enabled:
//! ```toml
//! scirs2-linalg = { version = "0.4.4", features = ["symbolic"] }
//! ```
//!
//! # Examples
//!
//! ```
//! use scirs2_linalg::symbolic::{det_symbolic, SymbolicLinalgError};
//! use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};
//! use scirs2_core::ndarray::{array, Array2};
//! use std::sync::Arc;
//!
//! // Determinant of [[a, 0], [0, b]] = a * b
//! let a = Arc::new(LoweredOp::Var(0));
//! let b = Arc::new(LoweredOp::Var(1));
//! let zero = Arc::new(LoweredOp::Const(0.0));
//! let mat: Array2<Arc<LoweredOp>> = Array2::from_shape_fn((2, 2), |(i, j)| {
//!     match (i, j) {
//!         (0, 0) => Arc::clone(&a),
//!         (1, 1) => Arc::clone(&b),
//!         _      => Arc::clone(&zero),
//!     }
//! });
//! let det_expr = det_symbolic(mat.view()).expect("det");
//! let val = eval_real(&det_expr, &EvalCtx::new(&[2.0, 3.0])).expect("eval");
//! assert!((val - 6.0).abs() < 1e-12);
//! ```

pub mod expm;
pub use expm::{expm_symbolic_2x2, expm_symbolic_3x3, ExpmSymbolicError};

pub mod recognize;
pub use recognize::{inverse_by_structure, recognize, StructureKind};

pub mod spectral;
pub use spectral::{
    eigenpairs_symmetric_2x2, eigenvalues_circulant, structured_eigenvalues, StructuredEig,
};

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{simplify_op, LoweredOp};
use std::sync::Arc;

// ─────────────────────────── error type ───────────────────────────────────

/// Errors returned by symbolic linear-algebra functions.
#[derive(Debug)]
pub enum SymbolicLinalgError {
    /// Matrix is not square.
    NotSquare {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },
    /// Matrix size exceeds the supported maximum for the requested operation.
    Unsupported {
        /// Actual matrix size (n×n).
        n: usize,
        /// Maximum supported size.
        max: usize,
    },
    /// An error occurred while evaluating a `LoweredOp` expression.
    EvalError(String),
    /// An error was returned by an underlying numeric linear algebra function.
    LinalgError(String),
}

impl std::fmt::Display for SymbolicLinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicLinalgError::NotSquare { rows, cols } => {
                write!(f, "matrix is not square ({rows}×{cols})")
            }
            SymbolicLinalgError::Unsupported { n, max } => {
                write!(
                    f,
                    "matrix size {n}×{n} exceeds maximum supported size {max}×{max}"
                )
            }
            SymbolicLinalgError::EvalError(msg) => write!(f, "evaluation error: {msg}"),
            SymbolicLinalgError::LinalgError(msg) => write!(f, "linalg error: {msg}"),
        }
    }
}

impl std::error::Error for SymbolicLinalgError {}

// ─────────────────────── internal helpers ─────────────────────────────────

/// Clone a `LoweredOp` out of an `Arc` cell.
#[inline]
fn cell(m: &ArrayView2<Arc<LoweredOp>>, r: usize, c: usize) -> LoweredOp {
    m[[r, c]].as_ref().clone()
}

/// Build a 2-argument `LoweredOp::Add`.
#[inline]
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

/// Build a 2-argument `LoweredOp::Sub`.
#[inline]
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

/// Build a 2-argument `LoweredOp::Mul`.
#[inline]
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

/// Build a 2-argument `LoweredOp::Pow`.
#[inline]
fn pow(base: LoweredOp, exp: LoweredOp) -> LoweredOp {
    LoweredOp::Pow(Box::new(base), Box::new(exp))
}

/// Build a `LoweredOp::Div`.
#[inline]
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Box::new(a), Box::new(b))
}

/// Build a `LoweredOp::Sqrt`.
#[inline]
fn sqrt(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sqrt(Box::new(a))
}

/// Build a `LoweredOp::Neg`.
#[inline]
fn neg(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Box::new(a))
}

/// Constant shorthand.
#[inline]
fn cnst(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

/// Extract the (n-1)×(n-1) minor of `matrix` by removing `row` and `col`.
///
/// Returns an owned `Array2<Arc<LoweredOp>>` so recursive calls can pass
/// `.view()` without lifetime complications.
fn minor(matrix: &ArrayView2<Arc<LoweredOp>>, row: usize, col: usize) -> Array2<Arc<LoweredOp>> {
    let n = matrix.nrows();
    debug_assert!(n >= 1);
    let m = n - 1;
    Array2::from_shape_fn((m, m), |(ri, ci)| {
        let src_r = if ri < row { ri } else { ri + 1 };
        let src_c = if ci < col { ci } else { ci + 1 };
        Arc::clone(&matrix[[src_r, src_c]])
    })
}

// ─────────────────────────── det_symbolic ─────────────────────────────────

/// Compute the determinant of a square matrix of symbolic expressions.
///
/// The determinant is computed by Leibniz cofactor expansion and returned as a
/// `LoweredOp` expression tree. A final simplification pass is applied before
/// returning, so common cancellations (zero coefficients, neutral elements) are
/// folded away.
///
/// # Size limits
///
/// | n | algorithm |
/// |---|-----------|
/// | 0 | returns `Const(1.0)` (empty product) |
/// | 1 | returns the single entry |
/// | 2 | `a·d − b·c` |
/// | 3 | cofactor expansion along row 0 |
/// | 4 | cofactor expansion along row 0 |
/// | ≥ 5 | returns [`SymbolicLinalgError::Unsupported`] |
///
/// # Errors
///
/// - [`SymbolicLinalgError::NotSquare`] — matrix is not square.
/// - [`SymbolicLinalgError::Unsupported`] — n ≥ 5.
pub fn det_symbolic(matrix: ArrayView2<Arc<LoweredOp>>) -> Result<LoweredOp, SymbolicLinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(SymbolicLinalgError::NotSquare { rows, cols });
    }
    let n = rows;

    let raw = det_recursive(&matrix, n)?;
    Ok(simplify_op(&raw))
}

/// Recursive cofactor expansion. Separated from the public function so that
/// recursive 3×3/4×4 calls skip the redundant square-check and final
/// simplification.
fn det_recursive(
    matrix: &ArrayView2<Arc<LoweredOp>>,
    n: usize,
) -> Result<LoweredOp, SymbolicLinalgError> {
    match n {
        0 => Ok(cnst(1.0)),
        1 => Ok(cell(matrix, 0, 0)),
        2 => {
            // det = a·d − b·c
            let a = cell(matrix, 0, 0);
            let b = cell(matrix, 0, 1);
            let c = cell(matrix, 1, 0);
            let d = cell(matrix, 1, 1);
            Ok(sub(mul(a, d), mul(b, c)))
        }
        3 | 4 => {
            // Cofactor expansion along row 0
            // term_j = (-1)^j * m[0,j] * det(minor(0, j))
            let mut terms: Vec<LoweredOp> = Vec::with_capacity(n);
            for j in 0..n {
                let m_sub = minor(matrix, 0, j);
                let sub_det = det_recursive(&m_sub.view(), n - 1)?;
                let entry = cell(matrix, 0, j);
                let product = mul(entry, sub_det);
                if j % 2 == 0 {
                    terms.push(product);
                } else {
                    terms.push(neg(product));
                }
            }
            // Fold: Add(Add(..., term_{n-2}), term_{n-1})
            let mut acc = terms.remove(0);
            for t in terms {
                acc = add(acc, t);
            }
            Ok(acc)
        }
        n => Err(SymbolicLinalgError::Unsupported { n, max: 4 }),
    }
}

// ───────────────────── eigenvalues_symbolic_2x2 ───────────────────────────

/// Compute the two eigenvalues of a 2×2 symbolic matrix in closed form.
///
/// Uses the quadratic formula on the characteristic polynomial:
///
/// ```text
/// λ = (tr ± sqrt(tr² − 4·det)) / 2
/// ```
///
/// The returned expressions are symbolic — they may involve a `Sqrt` of a
/// negative discriminant if the actual eigenvalues are complex. Such
/// expressions evaluate to an error via [`eval_real`]; callers that suspect
/// complex eigenvalues should use the complex evaluator or check the
/// discriminant before evaluating.
///
/// # Errors
///
/// - [`SymbolicLinalgError::Unsupported`] — matrix is not 2×2.
pub fn eigenvalues_symbolic_2x2(
    matrix: ArrayView2<Arc<LoweredOp>>,
) -> Result<[LoweredOp; 2], SymbolicLinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != 2 || cols != 2 {
        let n = rows;
        return Err(SymbolicLinalgError::Unsupported { n, max: 2 });
    }

    let a = cell(&matrix, 0, 0);
    let b = cell(&matrix, 0, 1);
    let c = cell(&matrix, 1, 0);
    let d = cell(&matrix, 1, 1);

    // tr = a + d
    let tr = add(a.clone(), d.clone());
    // det_val = a·d − b·c
    let det_val = sub(mul(a, d), mul(b, c));
    // discriminant = tr² − 4·det_val
    let tr_sq = pow(tr.clone(), cnst(2.0));
    let four_det = mul(cnst(4.0), det_val);
    let discriminant = sub(tr_sq, four_det);
    // sqrt_disc = sqrt(discriminant)
    let sqrt_disc = sqrt(discriminant);
    // λ± = (tr ± sqrt_disc) / 2
    let lambda_plus = div(add(tr.clone(), sqrt_disc.clone()), cnst(2.0));
    let lambda_minus = div(sub(tr, sqrt_disc), cnst(2.0));

    Ok([simplify_op(&lambda_plus), simplify_op(&lambda_minus)])
}

// ─────────────────── condition_number_symbolic ────────────────────────────

/// Evaluate a symbolic matrix at a point and compute its 2-norm condition
/// number numerically.
///
/// This function performs symbolic → numeric substitution by evaluating each
/// matrix entry with `eval_real`, then delegates to the existing numeric
/// [`crate::cond`] function (2-norm via SVD).
///
/// # Arguments
///
/// * `matrix` — symbolic matrix; must be square.
/// * `point`  — variable binding slice. `point[i]` binds `Var(i)`.
///
/// # Errors
///
/// - [`SymbolicLinalgError::NotSquare`] — matrix is not square.
/// - [`SymbolicLinalgError::EvalError`] — a symbolic entry failed to evaluate
///   at the given point (domain violation, unbound variable, etc.).
/// - [`SymbolicLinalgError::LinalgError`] — the underlying `cond` computation
///   failed (e.g., singular matrix with σ_min = 0 returns ∞, which is fine;
///   the error path is for structural failures).
pub fn condition_number_symbolic(
    matrix: ArrayView2<Arc<LoweredOp>>,
    point: ArrayView1<f64>,
) -> Result<f64, SymbolicLinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(SymbolicLinalgError::NotSquare { rows, cols });
    }
    let n = rows;

    // Build the evaluation context from the point slice.
    let bindings: Vec<f64> = point.to_vec();
    let ctx = EvalCtx::new(&bindings);

    // Evaluate every entry symbolically to build the numeric matrix.
    let mut numeric = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let entry = matrix[[i, j]].as_ref();
            let v = eval_real(entry, &ctx)
                .map_err(|e| SymbolicLinalgError::EvalError(e.to_string()))?;
            numeric[[i, j]] = v;
        }
    }

    // Delegate to the crate-internal `cond` function (2-norm via SVD).
    crate::cond(&numeric.view(), None, None)
        .map_err(|e| SymbolicLinalgError::LinalgError(e.to_string()))
}

// ─────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2, Array2};
    use scirs2_symbolic::eml::{eval_real, EvalCtx};

    fn c(v: f64) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Const(v))
    }
    fn v(i: usize) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(i))
    }

    // ── det tests ──

    #[test]
    fn det_2x2_diagonal_matches_product() {
        // [[a, 0], [0, d]] → det = a * d
        let zero = c(0.0);
        let mat = Array2::from_shape_fn((2, 2), |(r, c_)| match (r, c_) {
            (0, 0) => v(0),
            (1, 1) => v(1),
            _ => Arc::clone(&zero),
        });
        let expr = det_symbolic(mat.view()).expect("det");
        let val = eval_real(&expr, &EvalCtx::new(&[2.0, 3.0])).expect("eval");
        assert!((val - 6.0).abs() < 1e-12, "got {val}");
    }

    #[test]
    fn det_2x2_general() {
        // [[a,b],[c,d]] at (1,2,3,4) → 1*4 - 2*3 = -2
        let mat = Array2::from_shape_fn((2, 2), |(r, c_)| v(r * 2 + c_));
        let expr = det_symbolic(mat.view()).expect("det");
        let val = eval_real(&expr, &EvalCtx::new(&[1.0, 2.0, 3.0, 4.0])).expect("eval");
        assert!((val - (-2.0)).abs() < 1e-12, "got {val}");
    }

    #[test]
    fn det_3x3_diagonal() {
        // diagonal [[2,0,0],[0,3,0],[0,0,5]] → det = 30
        let zero = c(0.0);
        let mat = Array2::from_shape_fn((3, 3), |(r, c_)| {
            if r == c_ {
                c([2.0, 3.0, 5.0][r])
            } else {
                Arc::clone(&zero)
            }
        });
        let expr = det_symbolic(mat.view()).expect("det");
        let val = eval_real(&expr, &EvalCtx::new(&[])).expect("eval");
        assert!((val - 30.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn det_3x3_known() {
        // [[1,2,0],[3,4,0],[0,0,5]] → (1*4 - 2*3) * 5 = -10
        let entries = [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        let mat = Array2::from_shape_fn((3, 3), |(r, c_)| c(entries[r][c_]));
        let expr = det_symbolic(mat.view()).expect("det");
        let val = eval_real(&expr, &EvalCtx::new(&[])).expect("eval");
        assert!((val - (-10.0)).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn det_4x4_block_diagonal() {
        // Block-diagonal [[a,0,0,0],[0,b,0,0],[0,0,c,0],[0,0,0,d]]
        // det = a*b*c*d; at (2,3,4,5) → 120
        let zero = c(0.0);
        let mat = Array2::from_shape_fn(
            (4, 4),
            |(r, c_)| {
                if r == c_ {
                    v(r)
                } else {
                    Arc::clone(&zero)
                }
            },
        );
        let expr = det_symbolic(mat.view()).expect("det");
        let val = eval_real(&expr, &EvalCtx::new(&[2.0, 3.0, 4.0, 5.0])).expect("eval");
        assert!((val - 120.0).abs() < 1e-8, "got {val}");
    }

    #[test]
    fn det_5x5_returns_unsupported() {
        let mat = Array2::from_elem((5, 5), c(1.0));
        match det_symbolic(mat.view()) {
            Err(SymbolicLinalgError::Unsupported { n: 5, max: 4 }) => {}
            other => panic!("expected Unsupported(5,4), got {other:?}"),
        }
    }

    #[test]
    fn det_non_square_returns_err() {
        let mat = Array2::from_elem((2, 3), c(1.0));
        match det_symbolic(mat.view()) {
            Err(SymbolicLinalgError::NotSquare { rows: 2, cols: 3 }) => {}
            other => panic!("expected NotSquare(2,3), got {other:?}"),
        }
    }

    // ── eigenvalue tests ──

    #[test]
    fn eigenvalues_2x2_symmetric() {
        // [[a,1],[1,a]] at a=3: λ± = (6 ± sqrt(36-32))/2 = (6 ± 2)/2 = {4, 2}
        let one = c(1.0);
        let mat = Array2::from_shape_fn(
            (2, 2),
            |(r, c_)| {
                if r == c_ {
                    v(0)
                } else {
                    Arc::clone(&one)
                }
            },
        );
        let [lp, lm] = eigenvalues_symbolic_2x2(mat.view()).expect("eig");
        let ctx = EvalCtx::new(&[3.0]);
        let vp = eval_real(&lp, &ctx).expect("eval λ+");
        let vm = eval_real(&lm, &ctx).expect("eval λ-");
        assert!((vp - 4.0).abs() < 1e-10, "λ+ = {vp}");
        assert!((vm - 2.0).abs() < 1e-10, "λ- = {vm}");
    }

    #[test]
    fn eigenvalues_2x2_complex_at_point_returns_err() {
        // [[0,-1],[1,0]] → discriminant = 0 - 4*1 = -4 → Sqrt(-4) → Err
        let mat = Array2::from_shape_fn((2, 2), |(r, c_)| {
            let v = match (r, c_) {
                (0, 0) | (1, 1) => 0.0,
                (0, 1) => -1.0,
                _ => 1.0,
            };
            c(v)
        });
        let [lp, _lm] = eigenvalues_symbolic_2x2(mat.view()).expect("eig");
        let ctx = EvalCtx::new(&[]);
        // eval_real must fail for sqrt of negative
        assert!(
            eval_real(&lp, &ctx).is_err(),
            "expected Err for complex eigenvalue"
        );
    }

    #[test]
    fn eigenvalues_non_2x2_returns_err() {
        let mat = Array2::from_elem((3, 3), c(1.0));
        match eigenvalues_symbolic_2x2(mat.view()) {
            Err(SymbolicLinalgError::Unsupported { n: 3, max: 2 }) => {}
            other => panic!("expected Unsupported(3,2), got {other:?}"),
        }
    }

    // ── condition number tests ──

    #[test]
    fn condition_number_2x2_diagonal_known() {
        // [[a,1],[1,a]] at a=3 → eigenvalues 4 and 2 → cond_2 = 4/2 = 2
        let one = c(1.0);
        let mat = Array2::from_shape_fn(
            (2, 2),
            |(r, c_)| {
                if r == c_ {
                    v(0)
                } else {
                    Arc::clone(&one)
                }
            },
        );
        let kappa = condition_number_symbolic(mat.view(), arr1(&[3.0]).view()).expect("cond");
        assert!((kappa - 2.0).abs() < 1e-6, "cond = {kappa}");
    }

    #[test]
    fn condition_number_matches_numerical_baseline() {
        // [[Var(0), 0.5], [0.5, Var(1)]] at [2.0, 3.0] → numeric [[2,0.5],[0.5,3]]
        let half = c(0.5);
        let mat = Array2::from_shape_fn((2, 2), |(r, c_)| match (r, c_) {
            (0, 0) => v(0),
            (1, 1) => v(1),
            _ => Arc::clone(&half),
        });
        let symbolic_kappa =
            condition_number_symbolic(mat.view(), arr1(&[2.0, 3.0]).view()).expect("cond");

        // Compute numeric baseline
        let numeric = arr2(&[[2.0_f64, 0.5], [0.5, 3.0]]);
        let numeric_kappa = crate::cond(&numeric.view(), None, None).expect("cond baseline");

        assert!(
            (symbolic_kappa - numeric_kappa).abs() < 1e-8,
            "symbolic={symbolic_kappa}, numeric={numeric_kappa}"
        );
    }

    #[test]
    fn condition_number_non_square_returns_err() {
        let mat = Array2::from_elem((2, 3), c(1.0));
        match condition_number_symbolic(mat.view(), arr1(&[1.0, 2.0, 3.0]).view()) {
            Err(SymbolicLinalgError::NotSquare { rows: 2, cols: 3 }) => {}
            other => panic!("expected NotSquare, got {other:?}"),
        }
    }
}
