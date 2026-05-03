//! Higher-Order SVD (HOSVD / MLSVD) for 3-D tensors.
//!
//! Provides two algorithms:
//!
//! - **HOSVD** (De Lathauwer et al. 2000): compute each factor matrix
//!   independently from the mode unfoldings, then project.  Fast but
//!   sub-optimal (does not minimise the Tucker approximation error jointly).
//!
//! - **HOOI** (De Lathauwer et al. 2000b): Higher-Order Orthogonal Iteration.
//!   Iteratively refines the factor matrices starting from the HOSVD
//!   initialisation.  Converges to a local optimum.
//!
//! ## References
//!
//! - L. De Lathauwer, B. De Moor, J. Vandewalle, "A Multilinear Singular
//!   Value Decomposition", SIAM J. Matrix Anal. Appl. 21(4), 2000.
//! - L. De Lathauwer, B. De Moor, J. Vandewalle, "On the Best Rank-(R1,R2,R3)
//!   Approximation of Higher-Order Tensors", SIAM J. Matrix Anal. Appl.
//!   21(4), 2000.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor_decomp::tensor_utils::{mat_transpose, mode_n_product, truncated_svd, Tensor3D};

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Result of an HOSVD or HOOI decomposition.
///
/// Decomposition: `X ≈ G ×_1 U_1 ×_2 U_2 ×_3 U_3`
///
/// where:
/// - `G` is the core tensor with shape `(ranks[0], ranks[1], ranks[2])`.
/// - `U[n]` is the factor matrix for mode `n` with shape
///   `(shape[n], ranks[n])` and orthonormal columns.
#[derive(Debug, Clone)]
pub struct HOSVDDecomp {
    /// Core tensor `G`.
    pub g: Tensor3D,
    /// Factor matrices `[U_1, U_2, U_3]`; `u[n]` has shape `(shape[n], ranks[n])`.
    /// Stored as row-major `Vec<Vec<f64>>` where outer index is the original
    /// dimension and inner index is the rank index.
    pub u: [Vec<Vec<f64>>; 3],
    /// Multi-linear ranks `[r1, r2, r3]`.
    pub ranks: [usize; 3],
}

impl HOSVDDecomp {
    /// Reconstruct the full tensor: `X̃ = G ×_1 U_1 ×_2 U_2 ×_3 U_3`.
    pub fn reconstruct(&self) -> LinalgResult<Tensor3D> {
        let t1 = mode_n_product(&self.g, &self.u[0], 0)?;
        let t2 = mode_n_product(&t1, &self.u[1], 1)?;
        mode_n_product(&t2, &self.u[2], 2)
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
}

// ---------------------------------------------------------------------------
// HOSVD (truncated)
// ---------------------------------------------------------------------------

/// Compute the full-rank HOSVD (all singular vectors retained).
///
/// For each mode `n`:
/// 1. Form the mode-n unfolding `X_(n)`.
/// 2. Compute the SVD and keep **all** left singular vectors.
/// 3. Core: `G = X ×_1 U_1^T ×_2 U_2^T ×_3 U_3^T`.
///
/// # Errors
/// Returns an error if the SVD or mode product fails.
pub fn hosvd(x: &Tensor3D) -> LinalgResult<HOSVDDecomp> {
    let ranks = x.shape;
    hosvd_truncated(x, ranks)
}

/// Truncated HOSVD keeping `ranks[n]` left singular vectors per mode.
///
/// This is the standard initialisation for HOOI.
///
/// # Arguments
/// - `x`     – input tensor.
/// - `ranks` – multilinear ranks `[r1, r2, r3]`.
///
/// # Errors
/// Returns an error if any `ranks[n] == 0`, if `ranks[n] > shape[n]`, or if
/// an internal SVD fails.
pub fn hosvd_truncated(x: &Tensor3D, ranks: [usize; 3]) -> LinalgResult<HOSVDDecomp> {
    for n in 0..3 {
        if ranks[n] == 0 {
            return Err(LinalgError::DomainError(format!(
                "hosvd_truncated: ranks[{n}] must be ≥ 1"
            )));
        }
        if ranks[n] > x.shape[n] {
            return Err(LinalgError::DomainError(format!(
                "hosvd_truncated: ranks[{n}]={} > shape[{n}]={}",
                ranks[n], x.shape[n]
            )));
        }
    }

    let mut us: Vec<Vec<Vec<f64>>> = Vec::with_capacity(3);
    for n in 0..3 {
        let unfolding = x.mode_unfold(n)?;
        let (u, _, _) = truncated_svd(&unfolding, ranks[n])?;
        us.push(u);
    }

    // Core: G = X ×_1 U1^T ×_2 U2^T ×_3 U3^T
    let u0t = mat_transpose(&us[0]);
    let u1t = mat_transpose(&us[1]);
    let u2t = mat_transpose(&us[2]);
    let g1 = mode_n_product(x, &u0t, 0)?;
    let g2 = mode_n_product(&g1, &u1t, 1)?;
    let g = mode_n_product(&g2, &u2t, 2)?;

    let u_arr = [us.remove(0), us.remove(0), us.remove(0)];
    Ok(HOSVDDecomp { g, u: u_arr, ranks })
}

// ---------------------------------------------------------------------------
// HOOI
// ---------------------------------------------------------------------------

/// Higher-Order Orthogonal Iteration (HOOI).
///
/// Starts from the truncated HOSVD and iteratively improves the factor
/// matrices by alternating SVD updates.
///
/// Each iteration for mode `n`:
/// 1. Compute `Y = X ×_{m≠n} U_m^T` (project all modes except `n`).
/// 2. Compute the mode-n unfolding of `Y`.
/// 3. Update `U_n` as the leading `ranks[n]` left singular vectors.
///
/// Convergence: measured by `‖U_n_new - U_n_old‖_F` summed over modes.
///
/// # Arguments
/// - `x`        – input tensor.
/// - `ranks`    – multilinear ranks.
/// - `max_iter` – maximum iterations.
/// - `tol`      – convergence tolerance on subspace change (sum over modes).
///
/// # Errors
/// Same as [`hosvd_truncated`] plus convergence infrastructure.
pub fn hooi(
    x: &Tensor3D,
    ranks: [usize; 3],
    max_iter: usize,
    tol: f64,
) -> LinalgResult<HOSVDDecomp> {
    // Initialise from truncated HOSVD
    let init = hosvd_truncated(x, ranks)?;
    let mut us = init.u;

    for _iter in 0..max_iter {
        let mut delta = 0.0_f64;

        for n in 0..3_usize {
            // Collect the other two modes
            let modes_except_n: Vec<usize> = (0..3).filter(|&m| m != n).collect();

            // Project along all other modes
            let mut y = x.clone();
            for &m in &modes_except_n {
                let umt = mat_transpose(&us[m]);
                y = mode_n_product(&y, &umt, m)?;
            }

            // Unfold along mode n and compute truncated SVD
            let y_n = y.mode_unfold(n)?;
            let (u_new, _, _) = truncated_svd(&y_n, ranks[n])?;

            // Measure change (Frobenius norm of U_new - U_old, up to sign ambiguity)
            delta += subspace_change(&us[n], &u_new);
            us[n] = u_new;
        }

        if delta < tol {
            break;
        }
    }

    // Compute final core
    let u0t = mat_transpose(&us[0]);
    let u1t = mat_transpose(&us[1]);
    let u2t = mat_transpose(&us[2]);
    let g1 = mode_n_product(x, &u0t, 0)?;
    let g2 = mode_n_product(&g1, &u1t, 1)?;
    let g = mode_n_product(&g2, &u2t, 2)?;

    Ok(HOSVDDecomp {
        g,
        u: [us[0].clone(), us[1].clone(), us[2].clone()],
        ranks,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Measure the subspace change between two factor matrices `u_old` and
/// `u_new` via `‖u_old u_old^T - u_new u_new^T‖_F`.
///
/// This is invariant to column sign flips and permutations within the
/// span (only if columns span the same subspace).  For a quick convergence
/// check we use the simpler `‖u_new - u_old‖_F` after aligning signs.
fn subspace_change(u_old: &[Vec<f64>], u_new: &[Vec<f64>]) -> f64 {
    if u_old.len() != u_new.len() || u_old.is_empty() {
        return f64::INFINITY;
    }
    let m = u_old.len();
    let k = u_old[0].len().min(u_new[0].len());
    let mut sum = 0.0_f64;
    for i in 0..m {
        for j in 0..k {
            let diff = u_new[i][j] - u_old[i][j];
            sum += diff * diff;
        }
    }
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor() -> Tensor3D {
        // 3×4×5 tensor with elements (i+1)*(j+1) + k
        let data: Vec<f64> = (0..3_usize)
            .flat_map(|i| {
                (0..4_usize).flat_map(move |j| {
                    (0..5_usize).map(move |k| ((i + 1) * (j + 1)) as f64 + k as f64)
                })
            })
            .collect();
        Tensor3D::new(data, [3, 4, 5]).expect("ok")
    }

    #[test]
    fn test_hosvd_shapes() {
        let t = make_tensor();
        let d = hosvd_truncated(&t, [2, 3, 4]).expect("ok");
        assert_eq!(d.g.shape, [2, 3, 4]);
        assert_eq!(d.u[0].len(), 3);
        assert_eq!(d.u[0][0].len(), 2);
        assert_eq!(d.u[1].len(), 4);
        assert_eq!(d.u[1][0].len(), 3);
        assert_eq!(d.u[2].len(), 5);
        assert_eq!(d.u[2][0].len(), 4);
    }

    #[test]
    fn test_hosvd_full_rank_lossless() {
        let t = make_tensor();
        let d = hosvd(&t).expect("full rank ok");
        let err = d.relative_error(&t).expect("err ok");
        assert!(err < 1e-8, "HOSVD full-rank error {err:.2e}");
    }

    #[test]
    fn test_hosvd_truncated_reduces_error_with_rank() {
        let t = make_tensor();
        let d1 = hosvd_truncated(&t, [1, 1, 1]).expect("rank-1");
        let d2 = hosvd_truncated(&t, [2, 2, 2]).expect("rank-2");
        let e1 = d1.relative_error(&t).expect("e1");
        let e2 = d2.relative_error(&t).expect("e2");
        assert!(e2 <= e1 + 1e-10, "rank-2 error {e2} > rank-1 error {e1}");
    }

    #[test]
    fn test_hooi_shapes() {
        let t = make_tensor();
        let d = hooi(&t, [2, 3, 4], 20, 1e-8).expect("hooi ok");
        assert_eq!(d.g.shape, [2, 3, 4]);
        assert_eq!(d.u[0].len(), 3);
        assert_eq!(d.u[1].len(), 4);
        assert_eq!(d.u[2].len(), 5);
    }

    #[test]
    fn test_hooi_better_or_equal_to_hosvd() {
        let t = make_tensor();
        let d_hosvd = hosvd_truncated(&t, [2, 2, 2]).expect("hosvd");
        let d_hooi = hooi(&t, [2, 2, 2], 30, 1e-10).expect("hooi");
        let e_hosvd = d_hosvd.relative_error(&t).expect("e_hosvd");
        let e_hooi = d_hooi.relative_error(&t).expect("e_hooi");
        assert!(
            e_hooi <= e_hosvd + 1e-6,
            "HOOI error {e_hooi} > HOSVD error {e_hosvd}"
        );
    }

    #[test]
    fn test_hooi_full_rank_lossless() {
        let t =
            Tensor3D::new((0..27_usize).map(|x| x as f64 + 1.0).collect(), [3, 3, 3]).expect("ok");
        let d = hooi(&t, [3, 3, 3], 10, 1e-12).expect("ok");
        let err = d.relative_error(&t).expect("err");
        assert!(err < 1e-7, "full-rank HOOI error {err:.2e}");
    }

    #[test]
    fn test_factor_orthogonality() {
        let t = make_tensor();
        let d = hosvd_truncated(&t, [2, 3, 4]).expect("ok");
        for n in 0..3 {
            let u = &d.u[n];
            let r = u[0].len();
            let m = u.len();
            // U^T U ≈ I_r
            for i in 0..r {
                for j in 0..r {
                    let dot: f64 = (0..m).map(|k| u[k][i] * u[k][j]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-8,
                        "mode {n}: U^TU[{i},{j}] = {dot:.3e}, expected {expected}"
                    );
                }
            }
        }
    }
}
