//! Closed-form symbolic matrix exponential for small matrices.
//!
//! Wraps [`scirs2_symbolic::cas::matrix_exp`] functions and converts between
//! `Array2<Arc<LoweredOp>>` (used by scirs2-linalg) and the fixed-size arrays
//! expected by the CAS layer.
//!
//! # Fast paths
//!
//! Both `expm_symbolic_2x2` and `expm_symbolic_3x3` try the diagonal fast path
//! first (via [`scirs2_symbolic::cas::matrix_exp::expm_diag_2x2`] /
//! `expm_diag_3x3`). Only when the matrix is not diagonal do they fall through
//! to the general closed-form formula.
//!
//! # 3×3 symbolic limitation
//!
//! The general 3×3 path is limited to matrices whose entries are all concrete
//! `Const` values (after canonicalization). Passing a matrix with symbolic
//! (non-`Const`) entries that is not diagonal returns
//! [`ExpmSymbolicError::CubicRootSymbolic`].

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_symbolic::cas::matrix_exp::{expm_2x2, expm_3x3, expm_diag_2x2, expm_diag_3x3};
use scirs2_symbolic::eml::op::LoweredOp;
use std::sync::Arc;

// ─────────────────────────── error type ──────────────────────────────────────

/// Errors returned by symbolic matrix-exponential functions.
#[derive(Debug)]
pub enum ExpmSymbolicError {
    /// Matrix dimensions don't match the expected size.
    WrongSize {
        /// Actual row count.
        got_rows: usize,
        /// Actual column count.
        got_cols: usize,
        /// Expected side length (n for an n×n matrix).
        expected: usize,
    },
    /// Matrix is not square.
    NotSquare {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },
    /// The 3×3 general case requires a cubic root that cannot be expressed
    /// symbolically (entries are not all concrete `Const` values and the matrix
    /// is not diagonal).
    CubicRootSymbolic,
}

impl std::fmt::Display for ExpmSymbolicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpmSymbolicError::WrongSize {
                got_rows,
                got_cols,
                expected,
            } => write!(
                f,
                "matrix is {got_rows}×{got_cols} but expm_symbolic_{expected}x{expected} \
                 requires a {expected}×{expected} matrix"
            ),
            ExpmSymbolicError::NotSquare { rows, cols } => {
                write!(f, "matrix is not square ({rows}×{cols})")
            }
            ExpmSymbolicError::CubicRootSymbolic => write!(
                f,
                "expm_symbolic_3x3 requires all entries to be constant (Const) \
                 or the matrix to be diagonal; symbolic off-diagonal entries \
                 are not supported in the general 3×3 case"
            ),
        }
    }
}

impl std::error::Error for ExpmSymbolicError {}

// ─────────────────── private conversion helpers ──────────────────────────────

/// Convert an `ArrayView2<Arc<LoweredOp>>` to a fixed `[[LoweredOp; 2]; 2]`.
///
/// Returns `Err` if the matrix is not exactly 2×2.
fn array2_to_fixed_2x2(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<[[LoweredOp; 2]; 2], ExpmSymbolicError> {
    let (r, c) = m.dim();
    if r != c {
        return Err(ExpmSymbolicError::NotSquare { rows: r, cols: c });
    }
    if r != 2 {
        return Err(ExpmSymbolicError::WrongSize {
            got_rows: r,
            got_cols: c,
            expected: 2,
        });
    }
    Ok([
        [m[[0, 0]].as_ref().clone(), m[[0, 1]].as_ref().clone()],
        [m[[1, 0]].as_ref().clone(), m[[1, 1]].as_ref().clone()],
    ])
}

/// Convert an `ArrayView2<Arc<LoweredOp>>` to a fixed `[[LoweredOp; 3]; 3]`.
///
/// Returns `Err` if the matrix is not exactly 3×3.
fn array2_to_fixed_3x3(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<[[LoweredOp; 3]; 3], ExpmSymbolicError> {
    let (r, c) = m.dim();
    if r != c {
        return Err(ExpmSymbolicError::NotSquare { rows: r, cols: c });
    }
    if r != 3 {
        return Err(ExpmSymbolicError::WrongSize {
            got_rows: r,
            got_cols: c,
            expected: 3,
        });
    }
    Ok([
        [
            m[[0, 0]].as_ref().clone(),
            m[[0, 1]].as_ref().clone(),
            m[[0, 2]].as_ref().clone(),
        ],
        [
            m[[1, 0]].as_ref().clone(),
            m[[1, 1]].as_ref().clone(),
            m[[1, 2]].as_ref().clone(),
        ],
        [
            m[[2, 0]].as_ref().clone(),
            m[[2, 1]].as_ref().clone(),
            m[[2, 2]].as_ref().clone(),
        ],
    ])
}

/// Convert a fixed `[[LoweredOp; 2]; 2]` to an owned `Array2<Arc<LoweredOp>>`.
fn fixed_2x2_to_array2(m: [[LoweredOp; 2]; 2]) -> Array2<Arc<LoweredOp>> {
    Array2::from_shape_fn((2, 2), |(i, j)| Arc::new(m[i][j].clone()))
}

/// Convert a fixed `[[LoweredOp; 3]; 3]` to an owned `Array2<Arc<LoweredOp>>`.
fn fixed_3x3_to_array2(m: [[LoweredOp; 3]; 3]) -> Array2<Arc<LoweredOp>> {
    Array2::from_shape_fn((3, 3), |(i, j)| Arc::new(m[i][j].clone()))
}

// ─────────────────────────── public API ──────────────────────────────────────

/// Compute the symbolic matrix exponential of a 2×2 matrix.
///
/// The computation proceeds in two stages:
///
/// 1. **Diagonal fast path** — if all off-diagonal entries canonicalize to
///    `Const(0)`, the result is `diag(exp(m[0][0]), exp(m[1][1]))`. The
///    diagonal entries may be arbitrary symbolic expressions.
///
/// 2. **General Cayley–Hamilton path** — uses the mean-shift formula
///    `exp(M) = exp(t) · [cosh(δ)·I + sinh(δ)/δ · M']` where
///    `t = trace(M)/2`, `M' = M − tI`, and `δ = sqrt(−det(M'))`.
///
/// All output entries are canonicalized before return.
///
/// # Errors
///
/// - [`ExpmSymbolicError::NotSquare`] — `m` is not square.
/// - [`ExpmSymbolicError::WrongSize`] — `m` is square but not 2×2.
pub fn expm_symbolic_2x2(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<Array2<Arc<LoweredOp>>, ExpmSymbolicError> {
    let fixed = array2_to_fixed_2x2(m)?;

    // Diagonal fast path.
    if let Some(result) = expm_diag_2x2(&fixed) {
        return Ok(fixed_2x2_to_array2(result));
    }

    // General Cayley-Hamilton path.
    let result = expm_2x2(&fixed);
    Ok(fixed_2x2_to_array2(result))
}

/// Compute the symbolic matrix exponential of a 3×3 matrix.
///
/// The computation proceeds in two stages:
///
/// 1. **Diagonal fast path** — if all off-diagonal entries canonicalize to
///    `Const(0)`, the result is
///    `diag(exp(m[0][0]), exp(m[1][1]), exp(m[2][2]))`. The diagonal entries
///    may be arbitrary symbolic expressions.
///
/// 2. **All-constant general path** — if every entry is a `LoweredOp::Const`
///    (after canonicalization), the matrix exponential is computed numerically
///    via scaling-and-squaring with a 20-term Taylor series, and the result is
///    returned as `Const` entries.
///
/// For matrices with symbolic (non-`Const`) off-diagonal entries, the function
/// returns [`ExpmSymbolicError::CubicRootSymbolic`].
///
/// # Errors
///
/// - [`ExpmSymbolicError::NotSquare`] — `m` is not square.
/// - [`ExpmSymbolicError::WrongSize`] — `m` is square but not 3×3.
/// - [`ExpmSymbolicError::CubicRootSymbolic`] — matrix has symbolic
///   off-diagonal entries and the general path cannot proceed.
pub fn expm_symbolic_3x3(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<Array2<Arc<LoweredOp>>, ExpmSymbolicError> {
    let fixed = array2_to_fixed_3x3(m)?;

    // Diagonal fast path.
    if let Some(result) = expm_diag_3x3(&fixed) {
        return Ok(fixed_3x3_to_array2(result));
    }

    // General all-constant path.
    let result = expm_3x3(&fixed).map_err(|_| ExpmSymbolicError::CubicRootSymbolic)?;
    Ok(fixed_3x3_to_array2(result))
}

// ─────────────────────────────── tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};

    /// Build an `Arc<LoweredOp::Const(v)>`.
    fn c(v: f64) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Const(v))
    }

    /// Build an `Arc<LoweredOp::Var(i)>`.
    fn var(i: usize) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(i))
    }

    /// Evaluate a constant `LoweredOp` expression (no variable bindings).
    fn eval(op: &LoweredOp) -> f64 {
        let ctx = EvalCtx::new(&[]);
        eval_real(op, &ctx).expect("expression must be constant for eval()")
    }

    // ── Test 1 ── zero matrix → identity ─────────────────────────────────────

    #[test]
    fn test_expm_2x2_zero_matrix_gives_identity() {
        // exp([[0,0],[0,0]]) = I
        let mat = Array2::from_shape_fn((2, 2), |_| c(0.0));
        let result = expm_symbolic_2x2(mat.view()).expect("expm_symbolic_2x2 zero");
        let tol = 1e-12;
        assert!(
            (eval(result[[0, 0]].as_ref()) - 1.0).abs() < tol,
            "result[0][0] should be 1, got {}",
            eval(result[[0, 0]].as_ref())
        );
        assert!(
            eval(result[[0, 1]].as_ref()).abs() < tol,
            "result[0][1] should be 0"
        );
        assert!(
            eval(result[[1, 0]].as_ref()).abs() < tol,
            "result[1][0] should be 0"
        );
        assert!(
            (eval(result[[1, 1]].as_ref()) - 1.0).abs() < tol,
            "result[1][1] should be 1"
        );
    }

    // ── Test 2 ── diagonal 2×2 → element-wise exp ────────────────────────────

    #[test]
    fn test_expm_2x2_diagonal_entries() {
        // exp([[1, 0],[0, 2]]) = [[e, 0],[0, e²]]
        let mat = Array2::from_shape_fn(
            (2, 2),
            |(r, col)| {
                if r == col {
                    c([1.0, 2.0][r])
                } else {
                    c(0.0)
                }
            },
        );
        let result = expm_symbolic_2x2(mat.view()).expect("expm_symbolic_2x2 diag");
        let e = std::f64::consts::E;
        let tol = 1e-10;
        assert!(
            (eval(result[[0, 0]].as_ref()) - e).abs() < tol,
            "result[0][0] = e¹"
        );
        assert!(
            (eval(result[[1, 1]].as_ref()) - e * e).abs() < tol,
            "result[1][1] = e²"
        );
        assert!(
            eval(result[[0, 1]].as_ref()).abs() < tol,
            "off-diagonal [0][1] = 0"
        );
        assert!(
            eval(result[[1, 0]].as_ref()).abs() < tol,
            "off-diagonal [1][0] = 0"
        );
    }

    // ── Test 3 ── general off-diagonal 2×2 with δ ≠ 0 ───────────────────────

    #[test]
    fn test_expm_2x2_general_off_diagonal() {
        // exp([[0, 1],[-1, 0]]) = [[cos(1), sin(1)],[-sin(1), cos(1)]]
        // t = 0, M' = M, δ² = (1)*(-1) - 0*0 = -1, δ = sqrt(-1) = i
        // But in symbolic form: δ = sqrt(-1), cosh(i) = cos(1), sinh(i)/i = sin(1)/1 = sin(1)
        // So result[0][0] = cos(1), result[0][1] = sin(1), etc.
        // Numerically: expm([[0,1],[-1,0]]) at entries (Const), δ²=-1,
        // sqrt(-1) is imaginary but the formula still works numerically for the cosh/sinh:
        // cosh(i) = cos(1) ≈ 0.5403, sinh(i) = i·sin(1) ≈ 0.8415i
        // sinh(i)/i = sin(1) ≈ 0.8415
        // The Cayley-Hamilton path is used (off-diagonal is non-zero).
        // Use a real skew-symmetric scaled to avoid imaginary δ:
        // M = [[1, 2],[0, 1]] → t=1, M'=[[0,2],[0,0]], δ²=0·0-0·2=0 (nilpotent again)
        // Use M = [[0, 2],[1, 0]] instead: t=0, M'=M, δ²=2·1-0·0=2, δ=sqrt(2)
        // exp(M) numerically: [[cosh(√2), √2·sinh(√2)/√2],[sinh(√2)/√2, cosh(√2)]]
        //                   = [[cosh(√2), sinh(√2)],[sinh(√2)/√2·1, cosh(√2)]]
        // Actually for M=[[0,a],[b,0]]: t=0, M'=M, δ²=a·b, δ=sqrt(ab)
        // exp(M) = [[cosh(√(ab)), a·sinh(√(ab))/√(ab)],
        //           [b·sinh(√(ab))/√(ab), cosh(√(ab))]]
        // With a=2, b=1: δ=√2, result = [[cosh(√2), 2·sinh(√2)/√2],[sinh(√2)/√2, cosh(√2)]]
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 1) => c(2.0),
            (1, 0) => c(1.0),
            _ => c(0.0),
        });
        let result = expm_symbolic_2x2(mat.view()).expect("expm_symbolic_2x2 off-diagonal general");
        let sqrt2 = 2.0_f64.sqrt();
        let cosh_sqrt2 = sqrt2.cosh();
        let sinh_sqrt2 = sqrt2.sinh();
        let tol = 1e-8;
        assert!(
            (eval(result[[0, 0]].as_ref()) - cosh_sqrt2).abs() < tol,
            "[0][0] = cosh(√2) = {cosh_sqrt2:.8}, got {}",
            eval(result[[0, 0]].as_ref())
        );
        // result[0][1] = exp(t=0) * sinh(δ)/δ * m'[0][1] = 1 * sinh(√2)/√2 * 2
        let expected_01 = 2.0 * sinh_sqrt2 / sqrt2;
        assert!(
            (eval(result[[0, 1]].as_ref()) - expected_01).abs() < tol,
            "[0][1] = 2·sinh(√2)/√2 = {expected_01:.8}, got {}",
            eval(result[[0, 1]].as_ref())
        );
        // result[1][0] = sinh(√2)/√2 * m'[1][0] = sinh(√2)/√2 * 1
        let expected_10 = sinh_sqrt2 / sqrt2;
        assert!(
            (eval(result[[1, 0]].as_ref()) - expected_10).abs() < tol,
            "[1][0] = sinh(√2)/√2 = {expected_10:.8}, got {}",
            eval(result[[1, 0]].as_ref())
        );
        assert!(
            (eval(result[[1, 1]].as_ref()) - cosh_sqrt2).abs() < tol,
            "[1][1] = cosh(√2)"
        );
    }

    // ── Test 4 ── round-trip: expm(M) · expm(-M) ≈ I ─────────────────────────

    #[test]
    fn test_expm_2x2_round_trip_inverse() {
        // M = [[1, 2],[3, 4]], expm(M) * expm(-M) should be ≈ I
        let entries = [[1.0_f64, 2.0], [3.0, 4.0]];
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| c(entries[r][col]));
        let neg_mat = Array2::from_shape_fn((2, 2), |(r, col)| c(-entries[r][col]));

        let em = expm_symbolic_2x2(mat.view()).expect("expm(M)");
        let enm = expm_symbolic_2x2(neg_mat.view()).expect("expm(-M)");

        // Extract f64 values.
        let em_f = [
            [eval(em[[0, 0]].as_ref()), eval(em[[0, 1]].as_ref())],
            [eval(em[[1, 0]].as_ref()), eval(em[[1, 1]].as_ref())],
        ];
        let enm_f = [
            [eval(enm[[0, 0]].as_ref()), eval(enm[[0, 1]].as_ref())],
            [eval(enm[[1, 0]].as_ref()), eval(enm[[1, 1]].as_ref())],
        ];

        // Numeric matmul: prod = em_f * enm_f
        let prod = [
            [
                em_f[0][0] * enm_f[0][0] + em_f[0][1] * enm_f[1][0],
                em_f[0][0] * enm_f[0][1] + em_f[0][1] * enm_f[1][1],
            ],
            [
                em_f[1][0] * enm_f[0][0] + em_f[1][1] * enm_f[1][0],
                em_f[1][0] * enm_f[0][1] + em_f[1][1] * enm_f[1][1],
            ],
        ];

        let tol = 1e-6;
        assert!(
            (prod[0][0] - 1.0).abs() < tol,
            "I[0][0] = {:.2e}",
            prod[0][0]
        );
        assert!(prod[0][1].abs() < tol, "I[0][1] = {:.2e}", prod[0][1]);
        assert!(prod[1][0].abs() < tol, "I[1][0] = {:.2e}", prod[1][0]);
        assert!(
            (prod[1][1] - 1.0).abs() < tol,
            "I[1][1] = {:.2e}",
            prod[1][1]
        );
    }

    // ── Test 5 ── diagonal 3×3 → element-wise exp ────────────────────────────

    #[test]
    fn test_expm_3x3_diagonal_entries() {
        // exp(diag(1,2,3)) = diag(e, e², e³)
        let mat = Array2::from_shape_fn((3, 3), |(r, col)| {
            if r == col {
                c([1.0, 2.0, 3.0][r])
            } else {
                c(0.0)
            }
        });
        let result = expm_symbolic_3x3(mat.view()).expect("expm_symbolic_3x3 diag");
        let e = std::f64::consts::E;
        let tol = 1e-10;
        assert!(
            (eval(result[[0, 0]].as_ref()) - e).abs() < tol,
            "diag[0] = e¹"
        );
        assert!(
            (eval(result[[1, 1]].as_ref()) - e * e).abs() < tol,
            "diag[1] = e²"
        );
        assert!(
            (eval(result[[2, 2]].as_ref()) - e * e * e).abs() < tol,
            "diag[2] = e³"
        );
        // Spot-check two off-diagonal entries.
        assert!(eval(result[[0, 1]].as_ref()).abs() < tol, "off[0][1] = 0");
        assert!(eval(result[[1, 2]].as_ref()).abs() < tol, "off[1][2] = 0");
    }

    // ── Test 6 ── symbolic 3×3 (Var entries) → Err(CubicRootSymbolic) ────────

    #[test]
    fn test_expm_3x3_symbolic_entries_returns_err() {
        // A 3×3 matrix with Var entries — non-diagonal and non-constant.
        // The general path cannot handle this and must return CubicRootSymbolic.
        let mat = Array2::from_shape_fn((3, 3), |(r, col)| var(r * 3 + col));
        let result = expm_symbolic_3x3(mat.view());
        assert!(
            matches!(result, Err(ExpmSymbolicError::CubicRootSymbolic)),
            "expected CubicRootSymbolic, got {result:?}"
        );
    }

    // ── Additional edge-case tests ────────────────────────────────────────────

    #[test]
    fn test_expm_2x2_wrong_size_returns_err() {
        let mat = Array2::from_shape_fn((3, 3), |_| c(0.0));
        let result = expm_symbolic_2x2(mat.view());
        assert!(
            matches!(
                result,
                Err(ExpmSymbolicError::WrongSize { expected: 2, .. })
            ),
            "expected WrongSize(2), got {result:?}"
        );
    }

    #[test]
    fn test_expm_2x2_non_square_returns_err() {
        let mat = Array2::from_shape_fn((2, 3), |_| c(0.0));
        let result = expm_symbolic_2x2(mat.view());
        assert!(
            matches!(
                result,
                Err(ExpmSymbolicError::NotSquare { rows: 2, cols: 3 })
            ),
            "expected NotSquare(2,3), got {result:?}"
        );
    }

    #[test]
    fn test_expm_3x3_non_square_returns_err() {
        let mat = Array2::from_shape_fn((3, 4), |_| c(0.0));
        let result = expm_symbolic_3x3(mat.view());
        assert!(
            matches!(
                result,
                Err(ExpmSymbolicError::NotSquare { rows: 3, cols: 4 })
            ),
            "expected NotSquare(3,4), got {result:?}"
        );
    }

    #[test]
    fn test_expm_3x3_constant_general_matrix() {
        // exp([[1,0,0],[0,2,0],[0,0,3]]) using the non-diagonal constant path
        // by constructing a non-diagonal constant matrix that expm_3x3 can handle.
        // Use a known result: exp of a rotation-like matrix.
        // [[0,1,0],[-1,0,0],[0,0,0]] — skew-symmetric, known result.
        let mat = Array2::from_shape_fn((3, 3), |(r, col)| {
            let v = match (r, col) {
                (0, 1) => 1.0,
                (1, 0) => -1.0,
                _ => 0.0,
            };
            c(v)
        });
        let result = expm_symbolic_3x3(mat.view())
            .expect("expm_symbolic_3x3 should succeed for all-constant matrix");
        // exp([[0,1,0],[-1,0,0],[0,0,0]]) = [[cos1, sin1, 0],[-sin1, cos1, 0],[0,0,1]]
        let cos1 = 1.0_f64.cos();
        let sin1 = 1.0_f64.sin();
        let tol = 1e-8;
        assert!(
            (eval(result[[0, 0]].as_ref()) - cos1).abs() < tol,
            "[0][0] = cos(1) = {cos1:.8}, got {}",
            eval(result[[0, 0]].as_ref())
        );
        assert!(
            (eval(result[[0, 1]].as_ref()) - sin1).abs() < tol,
            "[0][1] = sin(1) = {sin1:.8}, got {}",
            eval(result[[0, 1]].as_ref())
        );
        assert!(
            (eval(result[[2, 2]].as_ref()) - 1.0).abs() < tol,
            "[2][2] = 1"
        );
    }
}
