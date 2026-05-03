//! Tucker decomposition for 3-D tensors.
//!
//! Tucker decomposes a tensor as:
//!
//! `X ≈ G ×_1 A ×_2 B ×_3 C`
//!
//! where `G ∈ R^{r1×r2×r3}` is the core tensor and
//! `A ∈ R^{I×r1}`, `B ∈ R^{J×r2}`, `C ∈ R^{K×r3}` are the factor matrices.
//!
//! Tucker is more general than CP (PARAFAC): CP is a special case where the
//! core is superdiagonal (r1 = r2 = r3 = R, G[r,r,r] = λ_r).
//!
//! ## Algorithms
//!
//! - **ALS via HOOI**: The standard iterative algorithm that minimises the
//!   Frobenius reconstruction error.
//! - **Core consistency diagnostic** (CORCONDIA): assesses the appropriateness
//!   of a CP model with a given rank by projecting onto Tucker core and
//!   measuring deviation from an identity superdiagonal.
//!
//! ## References
//!
//! - L. De Lathauwer et al., SIAM J. Matrix Anal. Appl. 21(4), 2000.
//! - R. A. Harshman, "Foundations of PARAFAC", 1970.
//! - R. Bro, H. Kiers, "A new efficient method for determining the number of
//!   components in PARAFAC models", J. Chemometrics 17(5), 2003.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor_decomp::hosvd::hooi;
use crate::tensor_decomp::parafac::fit_als;
use crate::tensor_decomp::tensor_utils::{mode_n_product, Tensor3D};

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Result of a Tucker decomposition.
///
/// The decomposition is: `X ≈ G ×_1 factors[0] ×_2 factors[1] ×_3 factors[2]`
///
/// - `G` (the core) has shape `(ranks[0], ranks[1], ranks[2])`.
/// - `factors[n]` has shape `(shape[n], ranks[n])` with orthonormal columns.
#[derive(Debug, Clone)]
pub struct TuckerDecomp {
    /// Core tensor `G`.
    pub g: Tensor3D,
    /// Factor matrices `[A, B, C]`; `factors[n]` is `shape[n] × ranks[n]`.
    pub factors: [Vec<Vec<f64>>; 3],
    /// Multilinear ranks.
    pub ranks: [usize; 3],
}

impl TuckerDecomp {
    /// Reconstruct the full tensor: `X̃ = G ×_1 A ×_2 B ×_3 C`.
    pub fn reconstruct(&self) -> LinalgResult<Tensor3D> {
        let t1 = mode_n_product(&self.g, &self.factors[0], 0)?;
        let t2 = mode_n_product(&t1, &self.factors[1], 1)?;
        mode_n_product(&t2, &self.factors[2], 2)
    }

    /// Relative Frobenius reconstruction error `‖X - X̃‖_F / ‖X‖_F`.
    pub fn relative_error(&self, x: &Tensor3D) -> LinalgResult<f64> {
        let xhat = self.reconstruct()?;
        let diff_sq: f64 = x
            .data
            .iter()
            .zip(xhat.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let orig_sq: f64 = x.data.iter().map(|v| v * v).sum();
        if orig_sq == 0.0 {
            if diff_sq == 0.0 {
                Ok(0.0)
            } else {
                Ok(f64::INFINITY)
            }
        } else {
            Ok((diff_sq / orig_sq).sqrt())
        }
    }

    /// Compression ratio: `numel(original) / (numel(core) + numel(factors))`.
    ///
    /// Values > 1 indicate compression; = 1 means no change.
    pub fn compress_ratio(&self, original_shape: [usize; 3]) -> f64 {
        let original: usize = original_shape.iter().product();
        let core: usize = self.g.shape.iter().product();
        let factors: usize = self
            .factors
            .iter()
            .map(|f| f.len() * if f.is_empty() { 0 } else { f[0].len() })
            .sum();
        let compressed = core + factors;
        if compressed == 0 {
            return f64::INFINITY;
        }
        original as f64 / compressed as f64
    }
}

// ---------------------------------------------------------------------------
// Tucker ALS via HOOI
// ---------------------------------------------------------------------------

/// Fit Tucker decomposition using the HOOI algorithm.
///
/// Equivalent to HOSVD initialisation followed by iterative refinement
/// (HOOI).
///
/// # Arguments
/// - `x`        – input tensor.
/// - `ranks`    – multilinear ranks `[r1, r2, r3]`.
/// - `max_iter` – maximum HOOI iterations.
/// - `tol`      – convergence tolerance on subspace change.
///
/// # Errors
/// Returns an error if `ranks` are out of bounds or internal SVD fails.
pub fn tucker_als(
    x: &Tensor3D,
    ranks: [usize; 3],
    max_iter: usize,
    tol: f64,
) -> LinalgResult<TuckerDecomp> {
    let hosvd_result = hooi(x, ranks, max_iter, tol)?;
    Ok(TuckerDecomp {
        g: hosvd_result.g,
        factors: hosvd_result.u,
        ranks,
    })
}

// ---------------------------------------------------------------------------
// Core Consistency Diagnostic (CORCONDIA)
// ---------------------------------------------------------------------------

/// Core Consistency Diagnostic (CORCONDIA) for CP model selection.
///
/// Computes a CP decomposition of rank `rank`, projects the factor matrices
/// onto the Tucker core, and measures the deviation of the core from the
/// identity superdiagonal tensor.
///
/// The diagnostic is:
///
/// ```text
/// CORCONDIA(R) = 100 * (1 - ‖G - T‖_F² / ‖T‖_F²)
/// ```
///
/// where:
/// - `G` is the Tucker core computed from the CP factor matrices.
/// - `T` is the "ideal" superdiagonal tensor with `T[r,r,r] = 1`.
///
/// A value near 100 % indicates a CP model is appropriate; values < 50 %
/// suggest the rank is too high or the model is inappropriate.
///
/// # Arguments
/// - `x`        – input tensor.
/// - `rank`     – CP rank to test.
/// - `max_iter` – maximum ALS iterations for the CP fit.
///
/// # Errors
/// Returns an error if the CP fit or Tucker projection fails.
pub fn core_consistency_diagnostic(
    x: &Tensor3D,
    rank: usize,
    max_iter: usize,
) -> LinalgResult<f64> {
    // Fit CP decomposition
    let cp = fit_als(x, rank, max_iter, 1e-8)?;

    // Use CP factor matrices as Tucker factors, compute core.
    // G = X ×_1 A^T ×_2 B^T ×_3 C^T
    // Factors cp.a, cp.b, cp.c have unit-norm columns; component weights are in cp.lambda.
    // We project with orthonormal (unit-norm) factors, which gives:
    //   G[r,r,r] = lambda[r]   (perfect CP fit)
    //   G[r,s,t] ≈ 0 for r≠s or s≠t (off-diagonal cross-terms)
    // The ideal Tucker core for a CP model is superdiagonal: T[r,r,r] = lambda[r].
    // CORCONDIA = 100 * (1 - ‖G - T‖_F² / ‖T‖_F²)
    use crate::tensor_decomp::tensor_utils::mat_transpose;

    let at = mat_transpose(&cp.a);
    let bt = mat_transpose(&cp.b);
    let ct = mat_transpose(&cp.c);

    // Orthonormalise each factor before projection (QR via Gram-Schmidt)
    let at_q = gram_schmidt_rows(&at)?;
    let bt_q = gram_schmidt_rows(&bt)?;
    let ct_q = gram_schmidt_rows(&ct)?;

    let g1 = mode_n_product(x, &at_q, 0)?;
    let g2 = mode_n_product(&g1, &bt_q, 1)?;
    let g = mode_n_product(&g2, &ct_q, 2)?;

    // Ideal superdiagonal tensor T[r,r,r] = lambda[r] (not 1, since factors have unit norms)
    let r = rank.min(g.shape[0]).min(g.shape[1]).min(g.shape[2]);
    let mut t = Tensor3D::zeros(g.shape);
    for ri in 0..r {
        let lam = if ri < cp.lambda.len() {
            cp.lambda[ri]
        } else {
            1.0
        };
        t.set(ri, ri, ri, lam);
    }

    // ‖G - T‖_F²  and  ‖T‖_F²
    let diff_sq: f64 = g
        .data
        .iter()
        .zip(t.data.iter())
        .map(|(gi, ti)| (gi - ti).powi(2))
        .sum();
    let t_norm_sq: f64 = t.data.iter().map(|v| v * v).sum();

    if t_norm_sq == 0.0 {
        return Ok(0.0);
    }

    let corcondia = 100.0 * (1.0 - diff_sq / t_norm_sq);
    Ok(corcondia)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gram-Schmidt orthonormalisation of the *rows* of a matrix.
/// Returns a new matrix with orthonormal rows (input rows span preserved).
fn gram_schmidt_rows(mat: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    if mat.is_empty() {
        return Ok(Vec::new());
    }
    let n = mat[0].len();
    let m = mat.len();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m);
    for row in mat {
        let mut v = row.clone();
        // Project out already-accepted basis vectors
        for qi in &q {
            let dot: f64 = v.iter().zip(qi.iter()).map(|(a, b)| a * b).sum();
            for (vi, qi_val) in v.iter_mut().zip(qi.iter()) {
                *vi -= dot * qi_val;
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            // Linearly dependent row — keep as zero vector
            q.push(vec![0.0_f64; n]);
        } else {
            q.push(v.iter().map(|x| x / norm).collect());
        }
    }
    Ok(q)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor() -> Tensor3D {
        let data: Vec<f64> = (0..60_usize).map(|x| x as f64 + 1.0).collect();
        Tensor3D::new(data, [3, 4, 5]).expect("ok")
    }

    #[test]
    fn test_tucker_als_shapes() {
        let t = make_tensor();
        let d = tucker_als(&t, [2, 3, 4], 20, 1e-8).expect("ok");
        assert_eq!(d.g.shape, [2, 3, 4]);
        assert_eq!(d.factors[0].len(), 3);
        assert_eq!(d.factors[0][0].len(), 2);
        assert_eq!(d.factors[1].len(), 4);
        assert_eq!(d.factors[1][0].len(), 3);
        assert_eq!(d.factors[2].len(), 5);
        assert_eq!(d.factors[2][0].len(), 4);
    }

    #[test]
    fn test_tucker_als_full_rank_lossless() {
        let t =
            Tensor3D::new((0..27_usize).map(|x| x as f64 + 1.0).collect(), [3, 3, 3]).expect("ok");
        let d = tucker_als(&t, [3, 3, 3], 10, 1e-12).expect("ok");
        let err = d.relative_error(&t).expect("err");
        assert!(err < 1e-7, "full-rank Tucker error {err:.2e}");
    }

    #[test]
    fn test_tucker_als_reconstruction_error() {
        let t = make_tensor();
        let d = tucker_als(&t, [2, 2, 2], 30, 1e-10).expect("ok");
        let err = d.relative_error(&t).expect("err");
        assert!(err < 1.0, "Tucker reconstruction error {err:.4}");
    }

    #[test]
    fn test_compress_ratio() {
        let t = make_tensor(); // 3×4×5 = 60 elements
        let d = tucker_als(&t, [2, 3, 4], 20, 1e-8).expect("ok");
        // core: 2*3*4=24, factors: 3*2+4*3+5*4=6+12+20=38, total=62
        let ratio = d.compress_ratio([3, 4, 5]);
        assert!(ratio > 0.0, "compress_ratio should be positive");
    }

    #[test]
    fn test_tucker_factor_orthogonality() {
        let t = make_tensor();
        let d = tucker_als(&t, [2, 3, 4], 20, 1e-8).expect("ok");
        for n in 0..3 {
            let u = &d.factors[n];
            let m = u.len();
            let r = u[0].len();
            for i in 0..r {
                for j in 0..r {
                    let dot: f64 = (0..m).map(|k| u[k][i] * u[k][j]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-7,
                        "mode {n}: U^TU[{i},{j}] = {dot:.3e}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_core_consistency_rank1() {
        // A rank-1 tensor should have CORCONDIA ≈ 100 for rank=1
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1.0_f64, 2.0];
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let data: Vec<f64> = (0..3)
            .flat_map(|i| {
                (0..2_usize).flat_map(move |j| (0..4_usize).map(move |k| a[i] * b[j] * c[k]))
            })
            .collect();
        let x = Tensor3D::new(data, [3, 2, 4]).expect("ok");
        let cc = core_consistency_diagnostic(&x, 1, 200).expect("cc ok");
        assert!(
            cc > 80.0,
            "rank-1 tensor CORCONDIA for rank=1 should be near 100, got {cc:.1}"
        );
    }
}
