//! `cas::matrix_ops` — symbolic matrix operations for 2×2, 3×3, and 4×4 matrices.
//!
//! All matrices are represented as `[[LoweredOp; N]; N]` Rust arrays.
//! All operations clone entries (which is cheap — `LoweredOp::Const(f64)` has no heap).
//!
//! # Functions
//!
//! - [`trace_2x2`], [`trace_3x3`], [`trace_4x4`] — sum of diagonal entries
//! - [`det_2x2`], [`det_3x3`], [`det_4x4`] — symbolic determinant (cofactor expansion)
//! - [`cofactor_3x3`] — (−1)^(i+j) × det of the 2×2 minor at row i, col j
//! - [`adjugate_2x2`], [`adjugate_3x3`], [`adjugate_4x4`] — transpose of the cofactor matrix
//! - [`inverse_2x2`], [`inverse_3x3`], [`inverse_4x4`] — symbolic inverse (returns `InverseResult`)
//!
//! # Singularity detection
//!
//! The [`InverseResult::Singular`] variant is returned only when the determinant
//! canonicalises to a [`LoweredOp::Const`] with absolute value `< 1e-14`.
//! For fully symbolic determinants the function always returns `Invertible*`.
//!
//! # Canonicalization
//!
//! Every intermediate and final result passes through
//! [`crate::cas::canonicalize::canonicalize`] so that commutativity-equivalent
//! sub-expressions hash identically.
//!
//! # Iterativeness / no recursion
//!
//! All traversals are flat loop-based; there are no recursive calls.
//! `det_4x4` is implemented by expanding across row 0 and calling `det_3x3`
//! on each 3×3 minor (itself flat).

use crate::cas::canonicalize::canonicalize;
use crate::eml::op::LoweredOp;

// ─────────────────────────────────────────────────────────────────────────────
// Internal arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build `LoweredOp::Add(a, b)`.
#[inline]
fn add_ops(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

/// Build `LoweredOp::Sub(a, b)`.
#[inline]
fn sub_ops(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

/// Build `LoweredOp::Mul(a, b)`.
#[inline]
fn mul_ops(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

/// Build `LoweredOp::Neg(a)`.
#[inline]
fn neg_op(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Box::new(a))
}

/// Canonicalize a [`LoweredOp`] and extract the inner op.
#[inline]
fn canon(op: LoweredOp) -> LoweredOp {
    canonicalize(&op).into_op()
}

// ─────────────────────────────────────────────────────────────────────────────
// Trace
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic trace of a 2×2 matrix: `m[0][0] + m[1][1]`.
pub fn trace_2x2(m: &[[LoweredOp; 2]; 2]) -> LoweredOp {
    canon(add_ops(m[0][0].clone(), m[1][1].clone()))
}

/// Symbolic trace of a 3×3 matrix: `m[0][0] + m[1][1] + m[2][2]`.
pub fn trace_3x3(m: &[[LoweredOp; 3]; 3]) -> LoweredOp {
    let t = add_ops(m[0][0].clone(), m[1][1].clone());
    canon(add_ops(t, m[2][2].clone()))
}

/// Symbolic trace of a 4×4 matrix: `m[0][0] + m[1][1] + m[2][2] + m[3][3]`.
pub fn trace_4x4(m: &[[LoweredOp; 4]; 4]) -> LoweredOp {
    let t0 = add_ops(m[0][0].clone(), m[1][1].clone());
    let t1 = add_ops(m[2][2].clone(), m[3][3].clone());
    canon(add_ops(t0, t1))
}

// ─────────────────────────────────────────────────────────────────────────────
// 2×2 determinant
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic determinant of a 2×2 matrix: `ad − bc`.
///
/// Entries are cloned; result is canonicalized.
pub fn det_2x2(m: &[[LoweredOp; 2]; 2]) -> LoweredOp {
    let ad = mul_ops(m[0][0].clone(), m[1][1].clone());
    let bc = mul_ops(m[0][1].clone(), m[1][0].clone());
    canon(sub_ops(ad, bc))
}

// ─────────────────────────────────────────────────────────────────────────────
// 3×3 minor / cofactor
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the 2×2 minor of a 3×3 matrix by deleting row `ri` and column `ci`.
fn minor_2x2_from_3x3(m: &[[LoweredOp; 3]; 3], ri: usize, ci: usize) -> [[LoweredOp; 2]; 2] {
    let rows: Vec<usize> = (0..3).filter(|&r| r != ri).collect();
    let cols: Vec<usize> = (0..3).filter(|&c| c != ci).collect();
    // Safe: exactly 2 rows and 2 cols after filtering
    [
        [m[rows[0]][cols[0]].clone(), m[rows[0]][cols[1]].clone()],
        [m[rows[1]][cols[0]].clone(), m[rows[1]][cols[1]].clone()],
    ]
}

/// Cofactor `C_{i,j}` of a 3×3 matrix.
///
/// Returns `(−1)^(i+j) × det(M_{ij})` where `M_{ij}` is the 2×2 minor
/// obtained by deleting row `i` and column `j`.
///
/// Panics if `i >= 3` or `j >= 3`.
pub fn cofactor_3x3(m: &[[LoweredOp; 3]; 3], i: usize, j: usize) -> LoweredOp {
    assert!(i < 3 && j < 3, "cofactor_3x3: indices must be 0..2");
    let minor = minor_2x2_from_3x3(m, i, j);
    let d = det_2x2(&minor);
    if (i + j).is_multiple_of(2) {
        canon(d)
    } else {
        canon(neg_op(d))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3×3 determinant
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic determinant of a 3×3 matrix via cofactor expansion along row 0.
pub fn det_3x3(m: &[[LoweredOp; 3]; 3]) -> LoweredOp {
    // det = m[0][0]*C_{0,0} + m[0][1]*C_{0,1} + m[0][2]*C_{0,2}
    let t0 = mul_ops(m[0][0].clone(), cofactor_3x3(m, 0, 0));
    let t1 = mul_ops(m[0][1].clone(), cofactor_3x3(m, 0, 1));
    let t2 = mul_ops(m[0][2].clone(), cofactor_3x3(m, 0, 2));
    canon(add_ops(add_ops(t0, t1), t2))
}

// ─────────────────────────────────────────────────────────────────────────────
// 4×4 minor and determinant
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the 3×3 minor of a 4×4 matrix by deleting row `ri` and column `ci`.
fn minor_3x3_from_4x4(m: &[[LoweredOp; 4]; 4], ri: usize, ci: usize) -> [[LoweredOp; 3]; 3] {
    let rows: Vec<usize> = (0..4).filter(|&r| r != ri).collect();
    let cols: Vec<usize> = (0..4).filter(|&c| c != ci).collect();
    // Safe: exactly 3 rows and 3 cols after filtering
    [
        [
            m[rows[0]][cols[0]].clone(),
            m[rows[0]][cols[1]].clone(),
            m[rows[0]][cols[2]].clone(),
        ],
        [
            m[rows[1]][cols[0]].clone(),
            m[rows[1]][cols[1]].clone(),
            m[rows[1]][cols[2]].clone(),
        ],
        [
            m[rows[2]][cols[0]].clone(),
            m[rows[2]][cols[1]].clone(),
            m[rows[2]][cols[2]].clone(),
        ],
    ]
}

/// Symbolic determinant of a 4×4 matrix via cofactor expansion along row 0.
pub fn det_4x4(m: &[[LoweredOp; 4]; 4]) -> LoweredOp {
    // Expand along row 0: det = sum_{j=0}^{3} (-1)^j * m[0][j] * det(M_{0j})
    let mut terms: Vec<LoweredOp> = Vec::with_capacity(4);
    for j in 0..4 {
        let minor = minor_3x3_from_4x4(m, 0, j);
        let minor_det = det_3x3(&minor);
        let entry = m[0][j].clone();
        let product = mul_ops(entry, minor_det);
        let signed = if j.is_multiple_of(2) {
            product
        } else {
            neg_op(product)
        };
        terms.push(signed);
    }
    // Fold: terms[0] + terms[1] + terms[2] + terms[3]
    let sum01 = add_ops(terms.remove(0), terms.remove(0));
    let sum23 = add_ops(terms.remove(0), terms.remove(0));
    canon(add_ops(sum01, sum23))
}

// ─────────────────────────────────────────────────────────────────────────────
// Adjugate
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic adjugate (classical adjoint) of a 2×2 matrix.
///
/// For `[[a, b], [c, d]]` the adjugate is `[[d, -b], [-c, a]]`.
pub fn adjugate_2x2(m: &[[LoweredOp; 2]; 2]) -> [[LoweredOp; 2]; 2] {
    [
        [canon(m[1][1].clone()), canon(neg_op(m[0][1].clone()))],
        [canon(neg_op(m[1][0].clone())), canon(m[0][0].clone())],
    ]
}

/// Symbolic adjugate of a 3×3 matrix.
///
/// The adjugate is the transpose of the cofactor matrix:
/// `adj[i][j] = C_{j,i}`.
pub fn adjugate_3x3(m: &[[LoweredOp; 3]; 3]) -> [[LoweredOp; 3]; 3] {
    // adj[i][j] = cofactor(m, j, i)  — note the transposition
    [
        [
            cofactor_3x3(m, 0, 0),
            cofactor_3x3(m, 1, 0),
            cofactor_3x3(m, 2, 0),
        ],
        [
            cofactor_3x3(m, 0, 1),
            cofactor_3x3(m, 1, 1),
            cofactor_3x3(m, 2, 1),
        ],
        [
            cofactor_3x3(m, 0, 2),
            cofactor_3x3(m, 1, 2),
            cofactor_3x3(m, 2, 2),
        ],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse result type
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic adjugate of a 4×4 matrix.
///
/// The adjugate is the transpose of the cofactor matrix:
/// `adj[i][j] = (−1)^(j+i) × det(minor(m, j, i))`.
///
/// Each minor is a 3×3 matrix obtained by deleting row `j` and column `i`
/// from the original 4×4 matrix (note transposition: cofactor `C_{j,i}`
/// fills position `[i][j]` of the adjugate).
pub fn adjugate_4x4(m: &[[LoweredOp; 4]; 4]) -> [[LoweredOp; 4]; 4] {
    // adj[i][j] = cofactor of m at position (j, i) = (-1)^(j+i) * det(minor(m, j, i))
    let mut result: Vec<Vec<LoweredOp>> = Vec::with_capacity(4);
    for i in 0..4 {
        let mut row = Vec::with_capacity(4);
        for j in 0..4 {
            let minor = minor_3x3_from_4x4(m, j, i);
            let d = det_3x3(&minor);
            let entry = if (i + j).is_multiple_of(2) {
                canon(d)
            } else {
                canon(neg_op(d))
            };
            row.push(entry);
        }
        result.push(row);
    }
    // Reconstruct as fixed-size array
    [
        [
            result[0][0].clone(),
            result[0][1].clone(),
            result[0][2].clone(),
            result[0][3].clone(),
        ],
        [
            result[1][0].clone(),
            result[1][1].clone(),
            result[1][2].clone(),
            result[1][3].clone(),
        ],
        [
            result[2][0].clone(),
            result[2][1].clone(),
            result[2][2].clone(),
            result[2][3].clone(),
        ],
        [
            result[3][0].clone(),
            result[3][1].clone(),
            result[3][2].clone(),
            result[3][3].clone(),
        ],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a symbolic matrix inversion.
///
/// [`InverseResult::Singular`] is returned only when the determinant
/// canonicalises to a numeric [`LoweredOp::Const`] with absolute value
/// `< 1e-14`. For fully symbolic determinants the `Invertible*` variant
/// is always returned.
#[derive(Clone, Debug)]
pub enum InverseResult {
    /// The matrix is (numerically) singular — determinant canonicalises to
    /// a constant with |det| < 1e-14.
    Singular,
    /// The 2×2 matrix is invertible; each entry is `adj[i][j] / det`.
    Invertible2([[LoweredOp; 2]; 2]),
    /// The 3×3 matrix is invertible; each entry is `adj[i][j] / det`.
    Invertible3([[LoweredOp; 3]; 3]),
    /// The 4×4 matrix is invertible; each entry is `adj[i][j] / det`.
    Invertible4([[LoweredOp; 4]; 4]),
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic inverse of a 2×2 matrix.
///
/// Returns [`InverseResult::Singular`] when `det` canonicalises to a
/// numeric constant with `|det| < 1e-14`. Otherwise returns
/// [`InverseResult::Invertible2`] with entries `adj[i][j] / det`.
pub fn inverse_2x2(m: &[[LoweredOp; 2]; 2]) -> InverseResult {
    let det = det_2x2(m);
    if let LoweredOp::Const(v) = &det {
        if v.abs() < 1e-14 {
            return InverseResult::Singular;
        }
    }
    let adj = adjugate_2x2(m);
    let result = adj
        .map(|row| row.map(|entry| canon(LoweredOp::Div(Box::new(entry), Box::new(det.clone())))));
    InverseResult::Invertible2(result)
}

/// Symbolic inverse of a 3×3 matrix.
///
/// Returns [`InverseResult::Singular`] when `det` canonicalises to a
/// numeric constant with `|det| < 1e-14`. Otherwise returns
/// [`InverseResult::Invertible3`] with entries `adj[i][j] / det`.
pub fn inverse_3x3(m: &[[LoweredOp; 3]; 3]) -> InverseResult {
    let det = det_3x3(m);
    if let LoweredOp::Const(v) = &det {
        if v.abs() < 1e-14 {
            return InverseResult::Singular;
        }
    }
    let adj = adjugate_3x3(m);
    let result = adj
        .map(|row| row.map(|entry| canon(LoweredOp::Div(Box::new(entry), Box::new(det.clone())))));
    InverseResult::Invertible3(result)
}

/// Symbolic inverse of a 4×4 matrix.
///
/// Returns [`InverseResult::Singular`] when `det` canonicalises to a
/// numeric constant with `|det| < 1e-14`. Otherwise returns
/// [`InverseResult::Invertible4`] with entries `adj[i][j] / det`.
pub fn inverse_4x4(m: &[[LoweredOp; 4]; 4]) -> InverseResult {
    let det = det_4x4(m);
    if let LoweredOp::Const(v) = &det {
        if v.abs() < 1e-14 {
            return InverseResult::Singular;
        }
    }
    let adj = adjugate_4x4(m);
    let result = adj
        .map(|row| row.map(|entry| canon(LoweredOp::Div(Box::new(entry), Box::new(det.clone())))));
    InverseResult::Invertible4(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    // Convenience: build Var(i) and Const(v)
    fn v(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(x: f64) -> LoweredOp {
        LoweredOp::Const(x)
    }

    // Evaluate a symbolic LoweredOp at a given point (slice of f64).
    fn eval(op: &LoweredOp, vals: &[f64]) -> f64 {
        let ctx = EvalCtx::new(vals);
        eval_real(op, &ctx).expect("eval failed in test")
    }

    // ── Test 1: det_2x2 symbolic, evaluate at (2,3,1,4) → 5 ──────────────
    #[test]
    fn test_det_2x2_symbolic() {
        // a=Var(0), b=Var(1), c=Var(2), d=Var(3)
        let m = [[v(0), v(1)], [v(2), v(3)]];
        let det = det_2x2(&m);
        // eval at a=2,b=3,c=1,d=4 → 2*4 - 3*1 = 8 - 3 = 5
        let result = eval(&det, &[2.0, 3.0, 1.0, 4.0]);
        assert!((result - 5.0).abs() < 1e-10, "expected 5, got {result}");
    }

    // ── Test 2: trace_2x2 at numeric [[1,2],[3,4]] → 5 ───────────────────
    #[test]
    fn test_trace_2x2_numeric() {
        let m = [[c(1.0), c(2.0)], [c(3.0), c(4.0)]];
        let tr = trace_2x2(&m);
        let result = eval(&tr, &[]);
        assert!((result - 5.0).abs() < 1e-10, "expected 5, got {result}");
    }

    // ── Test 3: det_3x3 diagonal [[2,0,0],[0,3,0],[0,0,5]] → 30 ─────────
    #[test]
    fn test_det_3x3_diagonal() {
        let m = [
            [c(2.0), c(0.0), c(0.0)],
            [c(0.0), c(3.0), c(0.0)],
            [c(0.0), c(0.0), c(5.0)],
        ];
        let det = det_3x3(&m);
        let result = eval(&det, &[]);
        assert!((result - 30.0).abs() < 1e-10, "expected 30, got {result}");
    }

    // ── Test 4: cofactor_3x3([[1,2,3],[4,5,6],[7,8,9]], 0,0) → -3 ────────
    #[test]
    fn test_cofactor_3x3_00() {
        let m = [
            [c(1.0), c(2.0), c(3.0)],
            [c(4.0), c(5.0), c(6.0)],
            [c(7.0), c(8.0), c(9.0)],
        ];
        let cof = cofactor_3x3(&m, 0, 0);
        // minor = [[5,6],[8,9]], det = 5*9 - 6*8 = 45 - 48 = -3
        // (0+0) even → +minor_det = -3
        let result = eval(&cof, &[]);
        assert!((result - (-3.0)).abs() < 1e-10, "expected -3, got {result}");
    }

    // ── Test 5: cofactor sign at (0,1): should be -(5*9-6*8) = -(45-48) = 3 → -(-3)=3
    // Actually for all-ones 3x3:  cofactor(m,0,1) = (-1)^1 * det([[m10,m12],[m20,m22]])
    //                                              = -1 * det([[1,1],[1,1]]) = -1*0 = 0
    // But more informative test: use [[1,2,3],[4,5,6],[7,8,9]], cofactor(0,1)
    // minor removes row0 col1 → [[4,6],[7,9]], det = 4*9 - 6*7 = 36-42 = -6
    // sign (0+1)=1 → odd → negate → +6
    #[test]
    fn test_cofactor_3x3_01_sign() {
        let m = [
            [c(1.0), c(2.0), c(3.0)],
            [c(4.0), c(5.0), c(6.0)],
            [c(7.0), c(8.0), c(9.0)],
        ];
        let cof = cofactor_3x3(&m, 0, 1);
        // minor [[4,6],[7,9]], det = 36-42 = -6, sign (-1)^(0+1) = -1
        // cofactor = -1 * (-6) = 6
        let result = eval(&cof, &[]);
        assert!((result - 6.0).abs() < 1e-10, "expected 6, got {result}");
    }

    // ── Test 6: inverse_2x2([[1,2],[3,4]]) entry [0][0] → 4/(-2) = -2 ───
    #[test]
    fn test_inverse_2x2_entry() {
        let m = [[c(1.0), c(2.0)], [c(3.0), c(4.0)]];
        // det = 1*4 - 2*3 = 4 - 6 = -2
        // adj = [[4,-2],[-3,1]]
        // inv = adj/det: [0][0] = 4/(-2) = -2
        match inverse_2x2(&m) {
            InverseResult::Invertible2(inv) => {
                let val = eval(&inv[0][0], &[]);
                assert!((val - (-2.0)).abs() < 1e-9, "expected -2, got {val}");
            }
            InverseResult::Singular => panic!("expected invertible"),
            _ => panic!("unexpected variant"),
        }
    }

    // ── Test 7: inverse_2x2([[1,2],[2,4]]) → Singular (det = 0) ──────────
    #[test]
    fn test_inverse_2x2_singular() {
        let m = [[c(1.0), c(2.0)], [c(2.0), c(4.0)]];
        // det = 1*4 - 2*2 = 0
        assert!(
            matches!(inverse_2x2(&m), InverseResult::Singular),
            "expected Singular"
        );
    }

    // ── Test 8: adjugate_2x2 → verify entries at (1,2,3,4) ──────────────
    #[test]
    fn test_adjugate_2x2() {
        // [[a,b],[c,d]] = [[Var(0),Var(1)],[Var(2),Var(3)]]
        let m = [[v(0), v(1)], [v(2), v(3)]];
        let adj = adjugate_2x2(&m);
        // adj = [[d,-b],[-c,a]] = [[Var(3),-Var(1)],[-Var(2),Var(0)]]
        // At (1,2,3,4): adj = [[4,-2],[-3,1]]
        let vals = &[1.0_f64, 2.0, 3.0, 4.0];
        assert!(
            (eval(&adj[0][0], vals) - 4.0).abs() < 1e-10,
            "adj[0][0] expected 4"
        );
        assert!(
            (eval(&adj[0][1], vals) - (-2.0)).abs() < 1e-10,
            "adj[0][1] expected -2"
        );
        assert!(
            (eval(&adj[1][0], vals) - (-3.0)).abs() < 1e-10,
            "adj[1][0] expected -3"
        );
        assert!(
            (eval(&adj[1][1], vals) - 1.0).abs() < 1e-10,
            "adj[1][1] expected 1"
        );
    }

    // ── Test 9: 3×3 identity → inverse → identity ─────────────────────────
    #[test]
    fn test_inverse_3x3_identity() {
        let m = [
            [c(1.0), c(0.0), c(0.0)],
            [c(0.0), c(1.0), c(0.0)],
            [c(0.0), c(0.0), c(1.0)],
        ];
        match inverse_3x3(&m) {
            InverseResult::Invertible3(inv) => {
                // Diagonal entries should evaluate to 1, off-diagonal to 0
                assert!(
                    (eval(&inv[0][0], &[]) - 1.0).abs() < 1e-9,
                    "inv[0][0] expected 1"
                );
                assert!(
                    (eval(&inv[0][1], &[]) - 0.0).abs() < 1e-9,
                    "inv[0][1] expected 0"
                );
            }
            InverseResult::Singular => panic!("identity matrix is not singular"),
            _ => panic!("unexpected variant"),
        }
    }

    // ── Test 10: det_4x4 diagonal [[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]] → 24 ─
    #[test]
    fn test_det_4x4_diagonal() {
        let m = [
            [c(1.0), c(0.0), c(0.0), c(0.0)],
            [c(0.0), c(2.0), c(0.0), c(0.0)],
            [c(0.0), c(0.0), c(3.0), c(0.0)],
            [c(0.0), c(0.0), c(0.0), c(4.0)],
        ];
        let det = det_4x4(&m);
        let result = eval(&det, &[]);
        assert!((result - 24.0).abs() < 1e-9, "expected 24, got {result}");
    }

    // ── Test 11: trace_4x4 at (1,2,3,4) → 10 ─────────────────────────────
    #[test]
    fn test_trace_4x4_symbolic() {
        // diagonal matrix with vars
        let m = [
            [v(0), c(0.0), c(0.0), c(0.0)],
            [c(0.0), v(1), c(0.0), c(0.0)],
            [c(0.0), c(0.0), v(2), c(0.0)],
            [c(0.0), c(0.0), c(0.0), v(3)],
        ];
        let tr = trace_4x4(&m);
        let result = eval(&tr, &[1.0, 2.0, 3.0, 4.0]);
        assert!((result - 10.0).abs() < 1e-10, "expected 10, got {result}");
    }

    // ── Test 12: adjugate round-trip M * adj(M) [0][0] = det(M) ──────────
    // For M = [[2,1,0],[1,3,1],[0,1,2]]
    // det(M) = 2*(3*2-1*1) - 1*(1*2-1*0) + 0 = 2*5 - 1*2 = 10 - 2 = 8
    // (M * adj(M))[0][0] = row0(M) . col0(adj) = sum_k M[0][k]*adj[k][0]
    //                     = M[0][0]*C_{00} + M[0][1]*C_{01} + M[0][2]*C_{02}
    //                     = same as det expansion = det(M)
    #[test]
    fn test_adjugate_3x3_roundtrip() {
        let m = [
            [c(2.0), c(1.0), c(0.0)],
            [c(1.0), c(3.0), c(1.0)],
            [c(0.0), c(1.0), c(2.0)],
        ];
        let det = det_3x3(&m);
        let adj = adjugate_3x3(&m);

        // Compute (M * adj)[0][0] = sum_k m[0][k] * adj[k][0]
        let prod = {
            let t0 = mul_ops(m[0][0].clone(), adj[0][0].clone());
            let t1 = mul_ops(m[0][1].clone(), adj[1][0].clone());
            let t2 = mul_ops(m[0][2].clone(), adj[2][0].clone());
            canon(add_ops(add_ops(t0, t1), t2))
        };

        let det_val = eval(&det, &[]);
        let prod_val = eval(&prod, &[]);
        assert!(
            (prod_val - det_val).abs() < 1e-9,
            "M*adj[0][0]={prod_val} should equal det={det_val}"
        );
        assert!(
            (det_val - 8.0).abs() < 1e-9,
            "det should be 8, got {det_val}"
        );
    }

    // ── Test 13: canonicalization is idempotent for det_2x2 ───────────────
    #[test]
    fn test_det_2x2_canonical_idempotent() {
        let m = [[v(0), v(1)], [v(2), v(3)]];
        let det1 = det_2x2(&m);
        let det2 = canon(det1.clone());
        // The structural hash of the double-canonicalized form must equal the
        // single-canonicalized form (det_2x2 already calls canon once).
        let h1 = canonicalize(&det1).hash();
        let h2 = canonicalize(&det2).hash();
        assert_eq!(h1, h2, "canonicalization must be idempotent");
    }

    // ── Test 14: inverse_3x3([[1,2,1],[1,3,2],[1,1,2]]) entry [0][0] ─────
    // det = 1*(3*2-2*1) - 2*(1*2-2*1) + 1*(1*1-3*1)
    //     = 1*(6-2) - 2*(2-2) + 1*(1-3)
    //     = 4 - 0 - 2 = 2
    // adj[0][0] = cofactor(0,0) = det([[3,2],[1,2]]) = 6-2 = 4
    // inv[0][0] = adj[0][0]/det = 4/2 = 2
    #[test]
    fn test_inverse_3x3_entry() {
        let m = [
            [c(1.0), c(2.0), c(1.0)],
            [c(1.0), c(3.0), c(2.0)],
            [c(1.0), c(1.0), c(2.0)],
        ];
        match inverse_3x3(&m) {
            InverseResult::Invertible3(inv) => {
                let val = eval(&inv[0][0], &[]);
                // textbook: inv[0][0] = 4/2 = 2
                assert!((val - 2.0).abs() < 1e-9, "inv[0][0] expected 2, got {val}");
            }
            InverseResult::Singular => panic!("expected invertible"),
            _ => panic!("unexpected variant"),
        }
    }
}
