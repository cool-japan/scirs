//! Symbolic matrix structure recognition.
//!
//! Given a matrix whose entries are [`LoweredOp`] expression trees, this
//! module identifies high-level structure (diagonal, scalar, circulant,
//! rank-1 update, or general) and exploits that structure to produce compact
//! symbolic inverses.
//!
//! # Feature gate
//!
//! Compiled only under the `symbolic` Cargo feature (inherited from the
//! parent `symbolic` module).
//!
//! # Structure kinds
//!
//! | [`StructureKind`]  | Detection criterion |
//! |--------------------|---------------------|
//! | `Scalar`           | All entries share the same structural hash |
//! | `Diagonal`         | All off-diagonal entries canonicalize to `Const(0.0)` |
//! | `Circulant`        | `m[r,c]` hash == `m[(r+1)%n,(c+1)%n]` hash for all `(r,c)` |
//! | `LowRankUpdate`    | Every off-diagonal entry is a `Mul(u_i, v_j)` with consistent per-row left-arm and per-column right-arm hashes; diagonal entries are `Add(Const(1.0), Mul(u_i,v_i))` or the canonicalized form thereof |
//! | `General`          | None of the above matched |
//!
//! # Note on `Scalar` semantics
//!
//! `Scalar` does **not** mean "scalar multiple of identity". It means
//! *all entries share the same structural hash*. Such a matrix is
//! rank-1 (and singular for n ≥ 2), so `inverse_by_structure` falls
//! through to `General` whose `inverse_2x2`/`inverse_3x3` will
//! detect the zero determinant symbolically.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_symbolic::cas::{canonicalize, inverse_2x2, inverse_3x3, InverseResult};
use scirs2_symbolic::eml::{simplify_op, LoweredOp};
use std::sync::Arc;

use super::SymbolicLinalgError;

// ─────────────────────── structure kind ──────────────────────────────────────

/// High-level matrix structure inferred from symbolic entries.
#[derive(Debug, Clone, PartialEq)]
pub enum StructureKind {
    /// Every entry shares the same structural hash (all equal symbolically).
    ///
    /// Note: for n ≥ 2 this matrix is singular; `inverse_by_structure`
    /// delegates to the general path which will surface the singular
    /// determinant.
    Scalar,

    /// All off-diagonal entries canonicalize to `Const(0.0)`.
    Diagonal,

    /// Each entry satisfies `m[r,c] ≡ m[(r+1)%n,(c+1)%n]` by structural
    /// hash (only checked for n ≤ 8 to bound cost).
    Circulant {
        /// First row expression trees `m[0,0], m[0,1], …, m[0,n-1]`.
        first_row: Vec<LoweredOp>,
    },

    /// Matrix of the form `I + u·vᵀ` where `u` and `v` are column/row
    /// vectors represented as `LoweredOp` expression trees.
    ///
    /// The Sherman-Morrison formula yields an exact symbolic inverse.
    LowRankUpdate {
        /// Column vector `u` (n entries).
        u: Vec<LoweredOp>,
        /// Row vector `v` (n entries).
        v: Vec<LoweredOp>,
    },

    /// No special structure detected; generic CAS-based inverse is used.
    General,
}

// ─────────────────────── internal helpers ─────────────────────────────────────

/// Structural hash of `LoweredOp::Const(0.0)` used for the Diagonal check.
fn zero_hash() -> u128 {
    LoweredOp::Const(0.0).structural_hash()
}

/// Canonicalize a `LoweredOp` reference and return the canonical hash.
fn canon_hash(op: &LoweredOp) -> u128 {
    canonicalize(op).hash()
}

/// Build a `LoweredOp::Mul`.
#[inline]
fn lo_mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

/// Build a `LoweredOp::Add`.
#[inline]
fn lo_add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

/// Build a `LoweredOp::Sub`.
#[inline]
fn lo_sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

/// Build a `LoweredOp::Div`.
#[inline]
fn lo_div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Box::new(a), Box::new(b))
}

/// Build a `LoweredOp::Const`.
#[inline]
fn lo_const(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

/// Clone the underlying `LoweredOp` from a cell in the matrix.
#[inline]
fn cell_op(m: &ArrayView2<Arc<LoweredOp>>, r: usize, c: usize) -> LoweredOp {
    m[[r, c]].as_ref().clone()
}

// ─────────────────────────── recognize ───────────────────────────────────────

/// Identify the structure of a square symbolic matrix.
///
/// The matrix dimensions are assumed to be n×n; if the matrix is not square
/// the result is [`StructureKind::General`] (callers that need the error should
/// validate shape before calling).
///
/// Detection is tried in priority order:
/// `Scalar → Diagonal → Circulant → LowRankUpdate → General`.
pub fn recognize(m: ArrayView2<Arc<LoweredOp>>) -> StructureKind {
    let nrows = m.nrows();
    let ncols = m.ncols();
    if nrows != ncols || nrows == 0 {
        return StructureKind::General;
    }
    let n = nrows;

    // ── Scalar check ─────────────────────────────────────────────────────────
    // All entries share the same structural hash.
    let base_hash = m[[0, 0]].structural_hash();
    let all_equal = (0..n).all(|r| (0..n).all(|c| m[[r, c]].structural_hash() == base_hash));
    if all_equal {
        return StructureKind::Scalar;
    }

    // ── Diagonal check ───────────────────────────────────────────────────────
    // All off-diagonal entries must canonicalize to zero.
    let zero_h = zero_hash();
    let is_diagonal =
        (0..n).all(|r| (0..n).all(|c| r == c || canon_hash(m[[r, c]].as_ref()) == zero_h));
    if is_diagonal {
        return StructureKind::Diagonal;
    }

    // ── Circulant check ──────────────────────────────────────────────────────
    // Only check for n ≤ 8 to keep cost O(n²) but bounded.
    if n <= 8 {
        let is_circulant = (0..n).all(|r| {
            (0..n).all(|c| {
                m[[r, c]].structural_hash() == m[[(r + 1) % n, (c + 1) % n]].structural_hash()
            })
        });
        if is_circulant {
            let first_row: Vec<LoweredOp> = (0..n).map(|c| cell_op(&m, 0, c)).collect();
            return StructureKind::Circulant { first_row };
        }
    }

    // ── LowRankUpdate check ─────────────────────────────────────────────────
    // Check for I + u·vᵀ structure (only for n ≤ 8 to bound cost).
    // Each off-diagonal (i,j) entry should be `Mul(u_i, v_j)` where the
    // left-arm hash is consistent per-row and the right-arm hash is consistent
    // per-column.
    if n <= 8 {
        if let Some((u, v)) = try_extract_low_rank_update(&m, n) {
            return StructureKind::LowRankUpdate { u, v };
        }
    }

    StructureKind::General
}

/// Try to extract u and v from an `I + u·vᵀ` matrix.
///
/// Returns `Some((u, v))` if the structure matches, `None` otherwise.
fn try_extract_low_rank_update(
    m: &ArrayView2<Arc<LoweredOp>>,
    n: usize,
) -> Option<(Vec<LoweredOp>, Vec<LoweredOp>)> {
    // Build per-row and per-column hash signatures from off-diagonal entries.
    // For row r and column c (r≠c), if m[r,c] = Mul(u_r, v_c) we expect:
    //   - left-arm hash identical for all c ≠ r in row r  → u_r
    //   - right-arm hash identical for all r ≠ c in column c → v_c

    // Collect left/right arm hashes from Mul off-diagonals.
    let mut row_left_hash: Vec<Option<u128>> = vec![None; n];
    let mut row_left_op: Vec<Option<LoweredOp>> = (0..n).map(|_| None).collect();
    let mut col_right_hash: Vec<Option<u128>> = vec![None; n];
    let mut col_right_op: Vec<Option<LoweredOp>> = (0..n).map(|_| None).collect();

    for r in 0..n {
        for c in 0..n {
            if r == c {
                continue;
            }
            let entry = m[[r, c]].as_ref();
            // Must be a Mul node.
            let (left, right) = match entry {
                LoweredOp::Mul(l, r_arm) => (l.as_ref().clone(), r_arm.as_ref().clone()),
                _ => return None,
            };
            let lh = left.structural_hash();
            let rh = right.structural_hash();

            // Check per-row left-arm consistency.
            match row_left_hash[r] {
                None => {
                    row_left_hash[r] = Some(lh);
                    row_left_op[r] = Some(left);
                }
                Some(existing) => {
                    if existing != lh {
                        return None;
                    }
                }
            }
            // Check per-column right-arm consistency.
            match col_right_hash[c] {
                None => {
                    col_right_hash[c] = Some(rh);
                    col_right_op[c] = Some(right);
                }
                Some(existing) => {
                    if existing != rh {
                        return None;
                    }
                }
            }
        }
    }

    // All rows and columns must have a detected arm.
    let u: Vec<LoweredOp> = row_left_op.into_iter().collect::<Option<Vec<_>>>()?;
    let v: Vec<LoweredOp> = col_right_op.into_iter().collect::<Option<Vec<_>>>()?;

    // Verify diagonal: m[i,i] must be Const(1.0) + u[i]*v[i] (in canonical form).
    let expected_diag_hashes: Vec<u128> = (0..n)
        .map(|i| {
            let expected = lo_add(lo_const(1.0), lo_mul(u[i].clone(), v[i].clone()));
            canon_hash(&expected)
        })
        .collect();
    for i in 0..n {
        let actual_h = canon_hash(m[[i, i]].as_ref());
        if actual_h != expected_diag_hashes[i] {
            return None;
        }
    }

    Some((u, v))
}

// ─────────────────────────── inverse_by_structure ────────────────────────────

/// Compute the symbolic inverse of a square matrix by exploiting its structure.
///
/// Dispatches based on [`recognize`]:
///
/// | Structure | Method |
/// |-----------|--------|
/// | `Diagonal` | Element-wise `1/m[i,i]` on the diagonal, zeros elsewhere |
/// | `LowRankUpdate` | Sherman-Morrison formula: `(I+uvᵀ)⁻¹ = I − uvᵀ/(1+vᵀu)` |
/// | `General` / `Scalar` / `Circulant` | CAS-based `inverse_2x2` or `inverse_3x3` (n ≤ 3); returns `Unsupported` for n > 3 |
///
/// # Errors
///
/// - [`SymbolicLinalgError::NotSquare`] — matrix is not square.
/// - [`SymbolicLinalgError::Unsupported`] — structure is General/Scalar/Circulant
///   and n > 3 (CAS inverse only supports n ≤ 3).
/// - [`SymbolicLinalgError::EvalError`] — the matrix is symbolically singular
///   (zero determinant detected during CAS inverse computation).
pub fn inverse_by_structure(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<Array2<Arc<LoweredOp>>, SymbolicLinalgError> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    if nrows != ncols {
        return Err(SymbolicLinalgError::NotSquare {
            rows: nrows,
            cols: ncols,
        });
    }
    let n = nrows;

    match recognize(m) {
        StructureKind::Diagonal => inverse_diagonal(m, n),
        StructureKind::LowRankUpdate { u, v } => inverse_low_rank_update(u, v, n),
        // Scalar, Circulant, General all fall through to CAS-based inverse.
        _ => inverse_general_cas(m, n),
    }
}

/// Diagonal inverse: `inv[i,i] = 1 / m[i,i]`, zeros elsewhere.
fn inverse_diagonal(
    m: ArrayView2<Arc<LoweredOp>>,
    n: usize,
) -> Result<Array2<Arc<LoweredOp>>, SymbolicLinalgError> {
    let zero = Arc::new(lo_const(0.0));
    let result = Array2::from_shape_fn((n, n), |(r, c)| {
        if r == c {
            let diag_entry = cell_op(&m, r, c);
            Arc::new(simplify_op(&lo_div(lo_const(1.0), diag_entry)))
        } else {
            Arc::clone(&zero)
        }
    });
    Ok(result)
}

/// Sherman-Morrison inverse: `(I + u·vᵀ)⁻¹ = I − u·vᵀ / (1 + vᵀu)`.
fn inverse_low_rank_update(
    u: Vec<LoweredOp>,
    v: Vec<LoweredOp>,
    n: usize,
) -> Result<Array2<Arc<LoweredOp>>, SymbolicLinalgError> {
    // Compute scalar denominator: 1 + vᵀu = 1 + Σ v[i]*u[i]
    let dot: LoweredOp = (0..n).fold(lo_const(0.0), |acc, i| {
        lo_add(acc, lo_mul(v[i].clone(), u[i].clone()))
    });
    let denom = simplify_op(&lo_add(lo_const(1.0), dot));

    // Build the result matrix.
    let result = Array2::from_shape_fn((n, n), |(r, c)| {
        // off-diagonal component: -u[r]*v[c] / denom
        let correction = lo_div(lo_mul(u[r].clone(), v[c].clone()), denom.clone());
        let entry = if r == c {
            // diagonal: 1 - u[r]*v[r] / denom
            lo_sub(lo_const(1.0), correction)
        } else {
            // off-diagonal: 0 - u[r]*v[c] / denom = -u[r]*v[c] / denom
            lo_sub(lo_const(0.0), correction)
        };
        Arc::new(simplify_op(&entry))
    });
    Ok(result)
}

/// CAS-based inverse using `scirs2_symbolic::cas::{inverse_2x2, inverse_3x3}`.
fn inverse_general_cas(
    m: ArrayView2<Arc<LoweredOp>>,
    n: usize,
) -> Result<Array2<Arc<LoweredOp>>, SymbolicLinalgError> {
    match n {
        2 => {
            let arr: [[LoweredOp; 2]; 2] = [
                [cell_op(&m, 0, 0), cell_op(&m, 0, 1)],
                [cell_op(&m, 1, 0), cell_op(&m, 1, 1)],
            ];
            match inverse_2x2(&arr) {
                InverseResult::Invertible2(inv) => Ok(Array2::from_shape_fn((2, 2), |(r, c)| {
                    Arc::new(simplify_op(&inv[r][c]))
                })),
                InverseResult::Singular => Err(SymbolicLinalgError::EvalError(
                    "matrix is symbolically singular (zero determinant)".to_owned(),
                )),
                _ => Err(SymbolicLinalgError::EvalError(
                    "unexpected InverseResult variant for 2x2".to_owned(),
                )),
            }
        }
        3 => {
            let arr: [[LoweredOp; 3]; 3] = [
                [cell_op(&m, 0, 0), cell_op(&m, 0, 1), cell_op(&m, 0, 2)],
                [cell_op(&m, 1, 0), cell_op(&m, 1, 1), cell_op(&m, 1, 2)],
                [cell_op(&m, 2, 0), cell_op(&m, 2, 1), cell_op(&m, 2, 2)],
            ];
            match inverse_3x3(&arr) {
                InverseResult::Invertible3(inv) => Ok(Array2::from_shape_fn((3, 3), |(r, c)| {
                    Arc::new(simplify_op(&inv[r][c]))
                })),
                InverseResult::Singular => Err(SymbolicLinalgError::EvalError(
                    "matrix is symbolically singular (zero determinant)".to_owned(),
                )),
                _ => Err(SymbolicLinalgError::EvalError(
                    "unexpected InverseResult variant for 3x3".to_owned(),
                )),
            }
        }
        n => Err(SymbolicLinalgError::Unsupported { n, max: 3 }),
    }
}

// ─────────────────────────────── tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};

    fn c(v: f64) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Const(v))
    }
    fn var(i: usize) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(i))
    }
    fn mul_arc(a: Arc<LoweredOp>, b: Arc<LoweredOp>) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Mul(
            Box::new(a.as_ref().clone()),
            Box::new(b.as_ref().clone()),
        ))
    }
    fn add_arc(a: Arc<LoweredOp>, b: Arc<LoweredOp>) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Add(
            Box::new(a.as_ref().clone()),
            Box::new(b.as_ref().clone()),
        ))
    }

    // ── Test 1: diagonal matrix → Diagonal ──────────────────────────────────

    #[test]
    fn recognize_diagonal_2x2() {
        let zero = c(0.0);
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 0) => var(0),
            (1, 1) => var(1),
            _ => Arc::clone(&zero),
        });
        assert_eq!(recognize(mat.view()), StructureKind::Diagonal);
    }

    // ── Test 2: all-equal matrix → Scalar ────────────────────────────────────

    #[test]
    fn recognize_scalar_2x2() {
        let c0 = var(0);
        let mat = Array2::from_shape_fn((2, 2), |_| Arc::clone(&c0));
        assert_eq!(recognize(mat.view()), StructureKind::Scalar);
    }

    // ── Test 3: 2×2 circulant [[c0,c1],[c1,c0]] → Circulant ─────────────────

    #[test]
    fn recognize_circulant_2x2() {
        let c0 = var(0);
        let c1 = var(1);
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| {
            if r == col {
                Arc::clone(&c0)
            } else {
                Arc::clone(&c1)
            }
        });
        match recognize(mat.view()) {
            StructureKind::Circulant { first_row } => {
                assert_eq!(first_row.len(), 2);
                assert_eq!(first_row[0].structural_hash(), c0.structural_hash());
                assert_eq!(first_row[1].structural_hash(), c1.structural_hash());
            }
            other => panic!("expected Circulant, got {other:?}"),
        }
    }

    // ── Test 4: 3×3 circulant [[c0,c1,c2],[c2,c0,c1],[c1,c2,c0]] ───────────

    #[test]
    fn recognize_circulant_3x3() {
        // Toeplitz-circulant: m[r,c] = first_row[(c - r + n) % n]
        let first_row = [var(0), var(1), var(2)];
        let n = 3usize;
        let mat =
            Array2::from_shape_fn((n, n), |(r, col)| Arc::clone(&first_row[(col + n - r) % n]));
        match recognize(mat.view()) {
            StructureKind::Circulant { first_row: fr } => {
                assert_eq!(fr.len(), 3);
            }
            other => panic!("expected Circulant(3×3), got {other:?}"),
        }
    }

    // ── Test 5: I + u·vᵀ → LowRankUpdate ───────────────────────────────────

    #[test]
    fn recognize_low_rank_update_2x2() {
        // u = [Var(0), Var(1)], v = [Var(2), Var(3)]
        // m[i][j] = u[i]*v[j] for i≠j; m[i][i] = 1 + u[i]*v[i]
        let u0 = var(0);
        let u1 = var(1);
        let v0 = var(2);
        let v1 = var(3);
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 0) => add_arc(c(1.0), mul_arc(Arc::clone(&u0), Arc::clone(&v0))),
            (0, 1) => mul_arc(Arc::clone(&u0), Arc::clone(&v1)),
            (1, 0) => mul_arc(Arc::clone(&u1), Arc::clone(&v0)),
            (1, 1) => add_arc(c(1.0), mul_arc(Arc::clone(&u1), Arc::clone(&v1))),
            _ => unreachable!(),
        });
        match recognize(mat.view()) {
            StructureKind::LowRankUpdate { u, v } => {
                assert_eq!(u.len(), 2);
                assert_eq!(v.len(), 2);
            }
            other => panic!("expected LowRankUpdate, got {other:?}"),
        }
    }

    // ── Test 6: inverse of diagonal [[Var(0),0],[0,Var(1)]] ─────────────────

    #[test]
    fn inverse_diagonal_numeric_eval() {
        let zero = c(0.0);
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 0) => var(0),
            (1, 1) => var(1),
            _ => Arc::clone(&zero),
        });
        let inv = inverse_by_structure(mat.view()).expect("inverse");
        let ctx = EvalCtx::new(&[2.0, 3.0]);

        let inv00 = eval_real(inv[[0, 0]].as_ref(), &ctx).expect("eval");
        let inv11 = eval_real(inv[[1, 1]].as_ref(), &ctx).expect("eval");
        let inv01 = eval_real(inv[[0, 1]].as_ref(), &ctx).expect("eval");
        let inv10 = eval_real(inv[[1, 0]].as_ref(), &ctx).expect("eval");

        assert!((inv00 - 0.5).abs() < 1e-10, "inv[0,0]={inv00}");
        assert!((inv11 - 1.0 / 3.0).abs() < 1e-10, "inv[1,1]={inv11}");
        assert!(inv01.abs() < 1e-10, "inv[0,1]={inv01}");
        assert!(inv10.abs() < 1e-10, "inv[1,0]={inv10}");
    }

    // ── Test 7: Sherman-Morrison on I + u·vᵀ with Const entries ─────────────
    //
    // u = [Const(1), Const(2)], v = [Const(3), Const(4)]
    // m = [[1+1*3, 1*4], [2*3, 1+2*4]] = [[4, 4], [6, 9]]
    // vᵀu = 3*1 + 4*2 = 11  →  denom = 1 + 11 = 12
    // inv via SM: inv[0,0] = 1 - 1*3/12 = 9/12 = 0.75
    //             inv[0,1] = 0 - 1*4/12 = -1/3
    //             inv[1,0] = 0 - 2*3/12 = -1/2
    //             inv[1,1] = 1 - 2*4/12 = 4/12 = 1/3
    // Cross-check: det(m) = 4*9 - 4*6 = 36-24 = 12
    // adjugate: [[9,-4],[-6,4]] / 12 → [0.75, -0.333, -0.5, 0.333] ✓

    #[test]
    fn inverse_low_rank_update_numeric_eval() {
        let u = [c(1.0), c(2.0)];
        let v = [c(3.0), c(4.0)];
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 0) => add_arc(c(1.0), mul_arc(Arc::clone(&u[0]), Arc::clone(&v[0]))),
            (0, 1) => mul_arc(Arc::clone(&u[0]), Arc::clone(&v[1])),
            (1, 0) => mul_arc(Arc::clone(&u[1]), Arc::clone(&v[0])),
            (1, 1) => add_arc(c(1.0), mul_arc(Arc::clone(&u[1]), Arc::clone(&v[1]))),
            _ => unreachable!(),
        });

        // Confirm recognition first.
        assert!(
            matches!(recognize(mat.view()), StructureKind::LowRankUpdate { .. }),
            "should detect LowRankUpdate"
        );

        let inv = inverse_by_structure(mat.view()).expect("inverse");
        let ctx = EvalCtx::new(&[]); // all Const — no variable bindings needed

        let inv00 = eval_real(inv[[0, 0]].as_ref(), &ctx).expect("eval");
        let inv01 = eval_real(inv[[0, 1]].as_ref(), &ctx).expect("eval");
        let inv10 = eval_real(inv[[1, 0]].as_ref(), &ctx).expect("eval");
        let inv11 = eval_real(inv[[1, 1]].as_ref(), &ctx).expect("eval");

        assert!((inv00 - 0.75).abs() < 1e-10, "inv[0,0]={inv00}");
        assert!((inv01 - (-1.0 / 3.0)).abs() < 1e-10, "inv[0,1]={inv01}");
        assert!((inv10 - (-0.5)).abs() < 1e-10, "inv[1,0]={inv10}");
        assert!((inv11 - (1.0 / 3.0)).abs() < 1e-10, "inv[1,1]={inv11}");
    }

    // ── Test 8: general matrix → General ─────────────────────────────────────

    #[test]
    fn recognize_general_2x2() {
        // [[Var(0), Var(1)], [Var(2), Const(1.0)]] — no regular structure
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 0) => var(0),
            (0, 1) => var(1),
            (1, 0) => var(2),
            (1, 1) => c(1.0),
            _ => unreachable!(),
        });
        assert_eq!(recognize(mat.view()), StructureKind::General);
    }
}
