//! Structured spectral decomposition for special matrix families.
//!
//! Detects structure via [`recognize`] then dispatches to the matching
//! closed-form eigenvalue formula.
//!
//! # Feature gate
//!
//! Compiled only under the `symbolic` Cargo feature (inherited from the
//! parent `symbolic` module).

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_symbolic::eml::LoweredOp;
use std::f64::consts::PI;
use std::sync::Arc;

use super::recognize::{recognize, StructureKind};
use super::{eigenvalues_symbolic_2x2, SymbolicLinalgError};

// ─────────────────────── LoweredOp arithmetic helpers ─────────────────────────

#[inline]
fn lo_add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

#[inline]
fn lo_mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

#[inline]
fn lo_cos(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Box::new(a))
}

// ─────────────────────── public result type ───────────────────────────────────

/// Result of a structured eigendecomposition.
#[derive(Debug, Clone)]
pub struct StructuredEig {
    /// Eigenvalues as `LoweredOp` expressions.
    pub eigenvalues: Vec<LoweredOp>,
    /// Which structural pattern was detected.
    pub kind: StructureKind,
}

// ─────────────────────── circulant eigenvalues ────────────────────────────────

/// Compute eigenvalues of a real circulant matrix symbolically.
///
/// For an n×n circulant with first row `[c₀, c₁, …, c_{n-1}]`,
/// eigenvalue k is: `λₖ = Σⱼ cⱼ · cos(2π·j·k/n)` for k = 0..n-1.
///
/// This returns the real part only, which is exact when the circulant is
/// symmetric (`first_row[j] == first_row[n-j]`).
///
/// # Arguments
///
/// * `first_row` — a slice of Arc-wrapped `LoweredOp` entries representing
///   the first row of the circulant matrix.
pub fn eigenvalues_circulant(first_row: ArrayView1<Arc<LoweredOp>>) -> Vec<LoweredOp> {
    let n = first_row.len();
    if n == 0 {
        return Vec::new();
    }

    let mut eigenvalues = Vec::with_capacity(n);

    for k in 0..n {
        // λₖ = Σⱼ cⱼ · cos(2π·j·k/n)
        // j=0 contributes c₀·cos(0) = c₀·1 = c₀ (no cos needed).
        let mut sum = first_row[0].as_ref().clone();

        for j in 1..n {
            let angle = 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let cos_term = lo_cos(LoweredOp::Const(angle));
            let term = lo_mul(first_row[j].as_ref().clone(), cos_term);
            sum = lo_add(sum, term);
        }

        eigenvalues.push(sum);
    }

    eigenvalues
}

// ─────────────────────── symmetric 2×2 eigenpairs ────────────────────────────

/// Compute eigenvalues and eigenvectors of a symmetric 2×2 matrix symbolically.
///
/// Wraps [`scirs2_symbolic::cas::spectral_2x2::eig_symmetric_2x2`].
///
/// # Returns
///
/// `Ok((eigenvalues, eigenvectors))` where:
/// - `eigenvalues` is `[lambda_1, lambda_2]` with the larger eigenvalue first.
/// - `eigenvectors` is a 2×2 `Array2<Arc<LoweredOp>>` whose **columns** are
///   the two (unnormalized) eigenvectors. `result[[i, j]]` is component `i`
///   of eigenvector `j`.
///
/// # Errors
///
/// - [`SymbolicLinalgError::NotSquare`] — matrix is not 2×2 square.
/// - [`SymbolicLinalgError::Unsupported`] — matrix size is not 2×2.
pub fn eigenpairs_symmetric_2x2(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<(Vec<LoweredOp>, Array2<Arc<LoweredOp>>), SymbolicLinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(SymbolicLinalgError::NotSquare { rows, cols });
    }
    if rows != 2 {
        return Err(SymbolicLinalgError::Unsupported { n: rows, max: 2 });
    }

    // Extract entries.
    let a = m[[0, 0]].as_ref().clone();
    let b = m[[0, 1]].as_ref().clone();
    let c = m[[1, 0]].as_ref().clone();
    let d = m[[1, 1]].as_ref().clone();

    let fixed = [[a, b], [c, d]];
    let eig = scirs2_symbolic::cas::spectral_2x2::eig_symmetric_2x2(&fixed);

    // Convert eigenvalues to Vec.
    let eigenvalues: Vec<LoweredOp> = eig.eigenvalues.to_vec();

    // Convert eigenvectors to 2×2 Array2, columns = eigenvectors.
    // eig.eigenvectors[j][i] = component i of eigenvector j.
    // result[[i, j]] = component i of eigenvector j = eig.eigenvectors[j][i].
    let evecs = Array2::from_shape_fn((2, 2), |(i, j)| Arc::new(eig.eigenvectors[j][i].clone()));

    Ok((eigenvalues, evecs))
}

// ─────────────────────── structured_eigenvalues ───────────────────────────────

/// Dispatch to the appropriate spectral method based on detected structure.
///
/// | Structure      | Method                                                        |
/// |----------------|---------------------------------------------------------------|
/// | `Diagonal`     | Return diagonal entries as eigenvalues                        |
/// | `Circulant`    | Closed-form DFT sum via [`eigenvalues_circulant`]             |
/// | `LowRankUpdate`| Matrix-determinant-lemma: one eigenvalue = 1 + vᵀu, rest = 1 |
/// | `General` 2×2  | [`eigenvalues_symbolic_2x2`] (quadratic formula)             |
/// | `Scalar`       | Returns `Err(Unsupported)` — eigenvalue multiplicity needs analysis |
/// | `General` n>2  | Returns `Err(Unsupported)`                                  |
///
/// # Errors
///
/// - [`SymbolicLinalgError::NotSquare`] — matrix is not square.
/// - [`SymbolicLinalgError::Unsupported`] — structure or size not supported.
pub fn structured_eigenvalues(
    m: ArrayView2<Arc<LoweredOp>>,
) -> Result<StructuredEig, SymbolicLinalgError> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(SymbolicLinalgError::NotSquare { rows, cols });
    }
    let n = rows;

    let kind = recognize(m);

    match &kind {
        StructureKind::Diagonal => {
            // Eigenvalues are the diagonal entries.
            let eigenvalues: Vec<LoweredOp> = (0..n).map(|i| m[[i, i]].as_ref().clone()).collect();
            Ok(StructuredEig { eigenvalues, kind })
        }

        StructureKind::Circulant { first_row } => {
            // Wrap Vec<LoweredOp> into Array1<Arc<LoweredOp>> and call
            // eigenvalues_circulant.
            let arc_row: Array1<Arc<LoweredOp>> =
                Array1::from_iter(first_row.iter().map(|op| Arc::new(op.clone())));
            let eigenvalues = eigenvalues_circulant(arc_row.view());
            Ok(StructuredEig { eigenvalues, kind })
        }

        StructureKind::LowRankUpdate { u, v } => {
            // For I + u·vᵀ: vᵀu = Σ v[i]*u[i].
            // One special eigenvalue = 1 + vᵀu; the remaining n-1 are 1.
            // Build vᵀu as a LoweredOp sum.
            let dot: LoweredOp = u
                .iter()
                .zip(v.iter())
                .fold(LoweredOp::Const(0.0), |acc, (ui, vi)| {
                    lo_add(acc, lo_mul(vi.clone(), ui.clone()))
                });
            let special_ev = lo_add(LoweredOp::Const(1.0), dot);

            let mut eigenvalues: Vec<LoweredOp> = (0..n.saturating_sub(1))
                .map(|_| LoweredOp::Const(1.0))
                .collect();
            eigenvalues.push(special_ev);
            Ok(StructuredEig { eigenvalues, kind })
        }

        StructureKind::Scalar => {
            // A scalar matrix (all entries equal) is rank-1 for n >= 2 and
            // singular. Computing eigenvalue multiplicity requires knowing
            // whether entries equal zero. Delegate to General for 2×2,
            // otherwise unsupported.
            if n == 2 {
                let evs = eigenvalues_symbolic_2x2(m)?;
                Ok(StructuredEig {
                    eigenvalues: evs.to_vec(),
                    kind,
                })
            } else {
                Err(SymbolicLinalgError::Unsupported { n, max: 0 })
            }
        }

        StructureKind::General => {
            // Only support 2×2 for General.
            if n == 2 {
                let evs = eigenvalues_symbolic_2x2(m)?;
                Ok(StructuredEig {
                    eigenvalues: evs.to_vec(),
                    kind,
                })
            } else {
                Err(SymbolicLinalgError::Unsupported { n, max: 2 })
            }
        }
    }
}

// ─────────────────────────────── tests ────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_symbolic::eml::{eval_real, EvalCtx};

    fn c(v: f64) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Const(v))
    }

    fn ev(op: &LoweredOp) -> f64 {
        eval_real(op, &EvalCtx::new(&[])).expect("eval failed")
    }

    // ── Test 1: eigenvalues_circulant for identity-like first_row [1, 0] ──────
    //
    // Circulant([[1,0],[0,1]]) — identity matrix.
    // λ₀ = 1·cos(0) + 0·cos(0) = 1
    // λ₁ = 1·cos(0) + 0·cos(π) = 1
    #[test]
    fn circulant_eigenvalues_identity_row() {
        let row = Array1::from_vec(vec![c(1.0), c(0.0)]);
        let evals = eigenvalues_circulant(row.view());
        assert_eq!(evals.len(), 2, "should have 2 eigenvalues");
        let lam0 = ev(&evals[0]);
        let lam1 = ev(&evals[1]);
        assert!((lam0 - 1.0).abs() < 1e-10, "lam0={lam0}");
        assert!((lam1 - 1.0).abs() < 1e-10, "lam1={lam1}");
    }

    // ── Test 2: eigenvalues_circulant for first_row [2, 1] ───────────────────
    //
    // Circulant([[2,1],[1,2]]).
    // λ₀ = 2·cos(0) + 1·cos(0) = 3
    // λ₁ = 2·cos(0) + 1·cos(π) = 2 - 1 = 1
    #[test]
    fn circulant_eigenvalues_2x2_known() {
        let row = Array1::from_vec(vec![c(2.0), c(1.0)]);
        let evals = eigenvalues_circulant(row.view());
        assert_eq!(evals.len(), 2, "should have 2 eigenvalues");
        let lam0 = ev(&evals[0]);
        let lam1 = ev(&evals[1]);
        // λ₀ = 3, λ₁ = 1 (order follows DFT, not sorted by magnitude).
        assert!((lam0 - 3.0).abs() < 1e-10, "lam0={lam0} (expected 3)");
        assert!((lam1 - 1.0).abs() < 1e-10, "lam1={lam1} (expected 1)");
    }

    // ── Test 3: eigenpairs_symmetric_2x2 eigenvalues [[3,1],[1,3]] ───────────
    //
    // Eigenvalues: 4 and 2.
    #[test]
    fn eigenpairs_2x2_eigenvalues_known() {
        let one = c(1.0);
        let mat = Array2::from_shape_fn(
            (2, 2),
            |(r, col)| {
                if r == col {
                    c(3.0)
                } else {
                    Arc::clone(&one)
                }
            },
        );
        let (evals, _evecs) = eigenpairs_symmetric_2x2(mat.view()).expect("eig");
        assert_eq!(evals.len(), 2, "should have 2 eigenvalues");
        // Larger eigenvalue is first per SymmetricEig2 convention.
        let lam1 = ev(&evals[0]);
        let lam2 = ev(&evals[1]);
        assert!((lam1 - 4.0).abs() < 1e-10, "lam1={lam1} (expected 4)");
        assert!((lam2 - 2.0).abs() < 1e-10, "lam2={lam2} (expected 2)");
    }

    // ── Test 4: eigenpairs_symmetric_2x2 orthogonality ───────────────────────
    //
    // For [[3,1],[1,3]], the two eigenvectors must be orthogonal.
    #[test]
    fn eigenpairs_2x2_eigenvector_orthogonality() {
        let one = c(1.0);
        let mat = Array2::from_shape_fn(
            (2, 2),
            |(r, col)| {
                if r == col {
                    c(3.0)
                } else {
                    Arc::clone(&one)
                }
            },
        );
        let (_evals, evecs) = eigenpairs_symmetric_2x2(mat.view()).expect("eig");
        // evecs[[i, j]] = component i of eigenvector j.
        // v1 = column 0, v2 = column 1.
        let v1x = ev(evecs[[0, 0]].as_ref());
        let v1y = ev(evecs[[1, 0]].as_ref());
        let v2x = ev(evecs[[0, 1]].as_ref());
        let v2y = ev(evecs[[1, 1]].as_ref());
        let dot = v1x * v2x + v1y * v2y;
        assert!(dot.abs() < 1e-10, "<v1, v2> = {dot} (expected 0)");
    }

    // ── Test 5: structured_eigenvalues dispatches to Circulant ───────────────
    //
    // Build a 3×3 circulant from first_row [2, 1, 1].
    // Circulant: m[r,c] = first_row[(c - r + n) % n].
    #[test]
    fn structured_eigenvalues_circulant_3x3() {
        let first_row_entries = [c(2.0), c(1.0), c(1.0)];
        let n = 3usize;
        let mat = Array2::from_shape_fn((n, n), |(r, col)| {
            Arc::clone(&first_row_entries[(col + n - r) % n])
        });
        let result = structured_eigenvalues(mat.view()).expect("eig");
        assert!(
            matches!(result.kind, StructureKind::Circulant { .. }),
            "expected Circulant, got {:?}",
            result.kind
        );
        assert_eq!(result.eigenvalues.len(), 3, "3×3 should have 3 eigenvalues");
        // Eigenvalues of [[2,1,1],[1,2,1],[1,1,2]]:
        // Using circulant formula: λ₀ = 2+1+1 = 4, λ₁ = 2+cos(2π/3)+cos(4π/3),
        // λ₂ = 2+cos(4π/3)+cos(8π/3).
        // Actually λ₀=4, λ₁=λ₂=1 for this symmetric circulant.
        let lam0 = ev(&result.eigenvalues[0]);
        assert!((lam0 - 4.0).abs() < 1e-10, "lam0={lam0} (expected 4)");
    }

    // ── Test 6: structured_eigenvalues dispatches to Diagonal ─────────────────
    //
    // [[2,0],[0,5]] → eigenvalues [2, 5].
    #[test]
    fn structured_eigenvalues_diagonal_2x2() {
        let zero = c(0.0);
        let mat = Array2::from_shape_fn((2, 2), |(r, col)| match (r, col) {
            (0, 0) => c(2.0),
            (1, 1) => c(5.0),
            _ => Arc::clone(&zero),
        });
        let result = structured_eigenvalues(mat.view()).expect("eig");
        assert!(
            matches!(result.kind, StructureKind::Diagonal),
            "expected Diagonal, got {:?}",
            result.kind
        );
        assert_eq!(result.eigenvalues.len(), 2, "should have 2 eigenvalues");
        let lam0 = ev(&result.eigenvalues[0]);
        let lam1 = ev(&result.eigenvalues[1]);
        assert!((lam0 - 2.0).abs() < 1e-10, "lam0={lam0} (expected 2)");
        assert!((lam1 - 5.0).abs() < 1e-10, "lam1={lam1} (expected 5)");
    }

    // ── Test 7: structured_eigenvalues dispatches to LowRankUpdate ───────────
    //
    // I + u·vᵀ with u=[1,2], v=[3,4].
    // vᵀu = 1*3 + 2*4 = 11
    // Eigenvalues: [1.0, 1 + 11] = [1.0, 12.0].
    #[test]
    fn structured_eigenvalues_low_rank_update() {
        // Build I + u·vᵀ with concrete Const entries.
        let u = [c(1.0), c(2.0)];
        let v = [c(3.0), c(4.0)];

        let mat = Array2::from_shape_fn((2, 2), |(r, col)| {
            // diagonal: 1 + u[r]*v[r]
            // off-diagonal: u[r]*v[col]
            let uv = Arc::new(LoweredOp::Mul(
                Box::new(u[r].as_ref().clone()),
                Box::new(v[col].as_ref().clone()),
            ));
            if r == col {
                Arc::new(LoweredOp::Add(
                    Box::new(LoweredOp::Const(1.0)),
                    Box::new(uv.as_ref().clone()),
                ))
            } else {
                uv
            }
        });

        // Confirm recognition first.
        assert!(
            matches!(recognize(mat.view()), StructureKind::LowRankUpdate { .. }),
            "should detect LowRankUpdate"
        );

        let result = structured_eigenvalues(mat.view()).expect("eig");
        assert!(
            matches!(result.kind, StructureKind::LowRankUpdate { .. }),
            "expected LowRankUpdate, got {:?}",
            result.kind
        );
        assert_eq!(result.eigenvalues.len(), 2, "2×2 should have 2 eigenvalues");

        // The special eigenvalue is last (index 1), ordinary ones are 1.0.
        let ev_ordinary = ev(&result.eigenvalues[0]);
        let ev_special = ev(&result.eigenvalues[1]);
        assert!(
            (ev_ordinary - 1.0).abs() < 1e-10,
            "ordinary eigenvalue={ev_ordinary} (expected 1.0)"
        );
        assert!(
            (ev_special - 12.0).abs() < 1e-10,
            "special eigenvalue={ev_special} (expected 12.0)"
        );
    }
}
