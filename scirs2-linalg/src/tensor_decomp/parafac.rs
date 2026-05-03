//! PARAFAC / CP (Canonical Polyadic) decomposition for 3-D tensors.
//!
//! Implements Alternating Least Squares (ALS) and its L2-regularised variant.
//!
//! ## Model
//!
//! `X ≈ sum_{r=1}^{R} λ_r * a_r ⊗ b_r ⊗ c_r`
//!
//! where:
//! - `A ∈ R^{I×R}`, `B ∈ R^{J×R}`, `C ∈ R^{K×R}` have unit-norm columns.
//! - `λ ∈ R^R` stores the component norms.
//!
//! ## References
//!
//! T. Kolda, B. Bader, "Tensor Decompositions and Applications",
//! SIAM Rev. 51(3), 2009.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor_decomp::tensor_utils::{
    gram, hadamard, mat_mul, mat_transpose, mode_n_product, solve_spd, truncated_svd, Tensor3D,
};

// ---------------------------------------------------------------------------
// Public structs
// ---------------------------------------------------------------------------

/// Result of a CP/PARAFAC decomposition.
///
/// The decomposition approximates `X ≈ sum_r λ_r * a_r ⊗ b_r ⊗ c_r` where:
/// - `A[:,r]`, `B[:,r]`, `C[:,r]` are unit-norm factor vectors.
/// - `lambda[r]` is the weight of component `r`.
#[derive(Debug, Clone)]
pub struct CPDecomp {
    /// Factor matrix `A ∈ R^{I×R}` (unit-norm columns).
    pub a: Vec<Vec<f64>>,
    /// Factor matrix `B ∈ R^{J×R}` (unit-norm columns).
    pub b: Vec<Vec<f64>>,
    /// Factor matrix `C ∈ R^{K×R}` (unit-norm columns).
    pub c: Vec<Vec<f64>>,
    /// Number of components.
    pub rank: usize,
    /// Component weights (column norms absorbed from `A`, `B`, `C`).
    pub lambda: Vec<f64>,
}

impl CPDecomp {
    /// Reconstruct the full tensor `X̃ = sum_r λ_r * a_r ⊗ b_r ⊗ c_r`.
    pub fn reconstruct(&self) -> LinalgResult<Tensor3D> {
        let i_dim = self.a.len();
        let j_dim = self.b.len();
        let k_dim = self.c.len();
        let mut result = Tensor3D::zeros([i_dim, j_dim, k_dim]);
        for r in 0..self.rank {
            let lam = self.lambda[r];
            for i in 0..i_dim {
                for j in 0..j_dim {
                    for k in 0..k_dim {
                        let v =
                            result.get(i, j, k) + lam * self.a[i][r] * self.b[j][r] * self.c[k][r];
                        result.set(i, j, k, v);
                    }
                }
            }
        }
        Ok(result)
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
// ALS algorithm
// ---------------------------------------------------------------------------

/// Fit a CP decomposition using Alternating Least Squares (ALS).
///
/// ## Algorithm
///
/// At each iteration the factor matrices are updated cyclically:
///
/// ```text
/// A ← X_(1) · (C ⊙ B) · (CᵀC * BᵀB)⁻¹
/// B ← X_(2) · (C ⊙ A) · (CᵀC * AᵀA)⁻¹
/// C ← X_(3) · (B ⊙ A) · (BᵀB * AᵀA)⁻¹
/// ```
///
/// where `⊙` is the Khatri-Rao product and `*` is element-wise multiplication.
///
/// After each update, columns are normalised and norms accumulated in `λ`.
///
/// # Arguments
/// - `x`        – input tensor.
/// - `rank`     – number of CP components.
/// - `max_iter` – maximum ALS iterations.
/// - `tol`      – convergence tolerance on the relative reconstruction error.
///
/// # Errors
/// Returns an error if `rank == 0`, `max_iter == 0`, or an internal linear
/// algebra operation fails.
pub fn fit_als(x: &Tensor3D, rank: usize, max_iter: usize, tol: f64) -> LinalgResult<CPDecomp> {
    fit_als_impl(x, rank, max_iter, tol, 0.0)
}

/// Fit a CP decomposition with L2 (ridge) regularisation.
///
/// The update for mode `n` adds `lambda_reg * I` to the Gram product before
/// solving, stabilising the ALS iteration for ill-conditioned tensors.
///
/// # Arguments
/// - `x`          – input tensor.
/// - `rank`       – number of CP components.
/// - `max_iter`   – maximum ALS iterations.
/// - `lambda_reg` – ridge-regularisation coefficient (≥ 0).
///
/// # Errors
/// Same as [`fit_als`].
pub fn fit_sparse_als(
    x: &Tensor3D,
    rank: usize,
    max_iter: usize,
    lambda_reg: f64,
) -> LinalgResult<CPDecomp> {
    fit_als_impl(x, rank, max_iter, 1e-10, lambda_reg)
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

fn fit_als_impl(
    x: &Tensor3D,
    rank: usize,
    max_iter: usize,
    tol: f64,
    lambda_reg: f64,
) -> LinalgResult<CPDecomp> {
    if rank == 0 {
        return Err(LinalgError::DomainError("CP rank must be ≥ 1".to_string()));
    }
    if max_iter == 0 {
        return Err(LinalgError::DomainError("max_iter must be ≥ 1".to_string()));
    }

    let [i_dim, j_dim, k_dim] = x.shape;
    let x_norm = x.frobenius_norm();

    // Initialise factor matrices via truncated SVD of mode unfoldings
    let x0 = x.mode_unfold(0)?;
    let (u0, _, _) = truncated_svd(&x0, rank)?;
    let x1 = x.mode_unfold(1)?;
    let (u1, _, _) = truncated_svd(&x1, rank)?;
    let x2 = x.mode_unfold(2)?;
    let (u2, _, _) = truncated_svd(&x2, rank)?;

    let mut a = pad_or_trim_columns(u0, i_dim, rank);
    let mut b = pad_or_trim_columns(u1, j_dim, rank);
    let mut c = pad_or_trim_columns(u2, k_dim, rank);
    // lambda is reset at the start of each iteration: it stores column norms
    // for the *current* iteration only. Accumulating across iterations causes
    // exponential blow-up (lambda → ∞) because normalise_columns multiplies
    // lambda by the column norm on every call.
    let mut lambda = vec![1.0_f64; rank];

    // Mode unfoldings (precomputed)
    let x_unfold_0 = x.mode_unfold(0)?; // [I, JK]
    let x_unfold_1 = x.mode_unfold(1)?; // [J, IK]
    let x_unfold_2 = x.mode_unfold(2)?; // [K, IJ]

    let mut prev_err = f64::INFINITY;

    for _iter in 0..max_iter {
        // Standard ALS updates: all three factors first, then normalize once.
        // Normalization happens AFTER all three updates so each update uses
        // the correctly-scaled factors from the previous iteration.

        // --- Update A ---
        // A ← X_(0) · (B ⊙ C) · (BᵀB * CᵀC)⁻¹
        let bc = Tensor3D::khatri_rao(&b, &c)?; // (JK × R)
        let gram_bc = gram_hadamard(&b, &c)?; // R × R
        let gram_bc_reg = add_ridge(&gram_bc, lambda_reg);
        let rhs_a = mat_mul(&x_unfold_0, &bc)?; // I × R
        a = solve_spd_rows(&gram_bc_reg, &rhs_a)?; // I × R

        // --- Update B ---
        // B ← X_(1) · (A ⊙ C) · (AᵀA * CᵀC)⁻¹
        let ac = Tensor3D::khatri_rao(&a, &c)?; // (IK × R)
        let gram_ac = gram_hadamard(&a, &c)?;
        let gram_ac_reg = add_ridge(&gram_ac, lambda_reg);
        let rhs_b = mat_mul(&x_unfold_1, &ac)?;
        b = solve_spd_rows(&gram_ac_reg, &rhs_b)?;

        // --- Update C ---
        // C ← X_(2) · (A ⊙ B) · (AᵀA * BᵀB)⁻¹
        let ab = Tensor3D::khatri_rao(&a, &b)?; // (IJ × R)
        let gram_ab = gram_hadamard(&a, &b)?;
        let gram_ab_reg = add_ridge(&gram_ab, lambda_reg);
        let rhs_c = mat_mul(&x_unfold_2, &ab)?;
        c = solve_spd_rows(&gram_ab_reg, &rhs_c)?;

        // --- Normalize all factors once per iteration ---
        // lambda[r] = ‖A[:,r]‖ * ‖B[:,r]‖ * ‖C[:,r]‖
        // Each factor is divided by its column norm; lambda absorbs the scale.
        let mut norm_a = vec![1.0_f64; rank];
        let mut norm_b = vec![1.0_f64; rank];
        let mut norm_c = vec![1.0_f64; rank];
        normalise_columns(&mut a, &mut norm_a);
        normalise_columns(&mut b, &mut norm_b);
        normalise_columns(&mut c, &mut norm_c);
        for r in 0..rank {
            lambda[r] = norm_a[r] * norm_b[r] * norm_c[r];
        }

        // --- Convergence check ---
        if tol > 0.0 && x_norm > 0.0 {
            let err = reconstruction_error_fast(x, &a, &b, &c, &lambda)?;
            let rel = err / x_norm;
            if (prev_err - rel).abs() < tol {
                break;
            }
            prev_err = rel;
        }
    }

    Ok(CPDecomp {
        a,
        b,
        c,
        rank,
        lambda,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pad or trim a matrix to have exactly `cols` columns.
fn pad_or_trim_columns(mut mat: Vec<Vec<f64>>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    // Ensure exactly `rows` rows
    mat.resize_with(rows, || vec![0.0_f64; cols]);
    for row in mat.iter_mut() {
        row.resize(cols, 0.0);
    }
    mat
}

/// `GᵀG` element-wise product (Gram-Hadamard): `V = (AᵀA) * (BᵀB)`.
fn gram_hadamard(a: &[Vec<f64>], b: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    let ga = gram(a)?;
    let gb = gram(b)?;
    hadamard(&ga, &gb)
}

/// Add `ridge * I` to a square matrix.
fn add_ridge(mat: &[Vec<f64>], ridge: f64) -> Vec<Vec<f64>> {
    let mut m = mat.to_vec();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] += ridge;
    }
    m
}

/// Solve `X · G = RHS` for each row of `RHS`, i.e. `X = RHS · G⁻¹`.
/// We solve `G · Xᵀ = RHSᵀ` and transpose.
fn solve_spd_rows(gram_mat: &[Vec<f64>], rhs: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    // gram_mat is R×R, rhs is (rows × R).
    // We want each row of the output to satisfy: out[i] · gram_mat = rhs[i].
    // Equivalently: gram_mat^T · out[i]^T = rhs[i]^T  (gram_mat is symmetric).
    let rhs_t = mat_transpose(rhs); // R × rows
    let sol_t = solve_spd(gram_mat, &rhs_t)?; // R × rows
    Ok(mat_transpose(&sol_t)) // rows × R
}

/// Normalise each column of `mat` and accumulate the norms multiplicatively
/// into `lambda`.
fn normalise_columns(mat: &mut [Vec<f64>], lambda: &mut [f64]) {
    let rows = mat.len();
    if rows == 0 {
        return;
    }
    let cols = mat[0].len();
    for c in 0..cols {
        let mut norm_sq = 0.0_f64;
        for r in 0..rows {
            norm_sq += mat[r][c] * mat[r][c];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            lambda[c] *= norm;
            for r in 0..rows {
                mat[r][c] /= norm;
            }
        }
    }
}

/// Efficient Frobenius reconstruction error without building the full tensor.
///
/// Uses `‖X‖_F² - 2 <X, X̃> + ‖X̃‖_F²` where the inner products are computed
/// mode-by-mode.  Falls back to direct reconstruction for small tensors.
fn reconstruction_error_fast(
    x: &Tensor3D,
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    lambda: &[f64],
) -> LinalgResult<f64> {
    let rank = lambda.len();
    let [i_dim, j_dim, k_dim] = x.shape;

    // Build the Gram-Hadamard product V = (AᵀA) * (BᵀB) * (CᵀC) (R×R)
    let ga = gram(a)?;
    let gb = gram(b)?;
    let gc = gram(c)?;
    let gabgc = hadamard(&hadamard(&ga, &gb)?, &gc)?;

    // ‖X̃‖_F² = λᵀ V λ  (where λ is a diagonal scaling)
    let mut xhat_norm_sq = 0.0_f64;
    for r in 0..rank {
        for s in 0..rank {
            xhat_norm_sq += lambda[r] * lambda[s] * gabgc[r][s];
        }
    }

    // <X, X̃> = sum_{r} λ_r * (Aᵀ X_(1) (C⊙B))_{rr}
    // Computed as sum over tensor elements
    let mut inner = 0.0_f64;
    for i in 0..i_dim {
        for j in 0..j_dim {
            for k in 0..k_dim {
                let xval = x.get(i, j, k);
                let mut approx = 0.0_f64;
                for r in 0..rank {
                    approx += lambda[r] * a[i][r] * b[j][r] * c[k][r];
                }
                inner += xval * approx;
            }
        }
    }

    let x_norm_sq: f64 = x.data.iter().map(|v| v * v).sum();
    let err_sq = (x_norm_sq - 2.0 * inner + xhat_norm_sq).max(0.0);
    Ok(err_sq.sqrt())
}

// ---------------------------------------------------------------------------
// Public wrapper kept for backward compat (re-exported in mod.rs)
// ---------------------------------------------------------------------------

/// Convenience: same as [`fit_als`].
pub fn cp_als(x: &Tensor3D, rank: usize, max_iter: usize, tol: f64) -> LinalgResult<CPDecomp> {
    fit_als(x, rank, max_iter, tol)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rank2_tensor() -> Tensor3D {
        // Build X = a1⊗b1⊗c1 + 2*a2⊗b2⊗c2
        let i = 4usize;
        let j = 5usize;
        let k = 6usize;
        let a1 = [1.0, 0.0, 0.0, 0.0];
        let b1 = [0.0, 1.0, 0.0, 0.0, 0.0];
        let c1 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let a2 = [0.0, 1.0, 0.0, 0.0];
        let b2 = [0.0, 0.0, 1.0, 0.0, 0.0];
        let c2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut data = vec![0.0_f64; i * j * k];
        for ii in 0..i {
            for jj in 0..j {
                for kk in 0..k {
                    data[ii * j * k + jj * k + kk] =
                        a1[ii] * b1[jj] * c1[kk] + 2.0 * a2[ii] * b2[jj] * c2[kk];
                }
            }
        }
        Tensor3D::new(data, [i, j, k]).expect("ok")
    }

    #[test]
    fn test_cp_reconstruct_rank2() {
        let x = make_rank2_tensor();
        let decomp = fit_als(&x, 2, 200, 1e-8).expect("als ok");
        let err = decomp.relative_error(&x).expect("err ok");
        assert!(err < 0.1, "CP reconstruction error {err:.4} >= 0.1");
    }

    #[test]
    fn test_cp_sparse_als() {
        let x = make_rank2_tensor();
        let decomp = fit_sparse_als(&x, 2, 200, 1e-4).expect("sparse als ok");
        let err = decomp.relative_error(&x).expect("err ok");
        assert!(err < 0.15, "CP sparse ALS error {err:.4} >= 0.15");
    }

    #[test]
    fn test_cp_rank_1() {
        // Rank-1 tensor: outer product of [1,2,3] ⊗ [4,5] ⊗ [6,7,8]
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0];
        let c = [6.0_f64, 7.0, 8.0];
        let data: Vec<f64> = (0..3)
            .flat_map(|i| (0..2).flat_map(move |j| (0..3).map(move |k| a[i] * b[j] * c[k])))
            .collect();
        let x = Tensor3D::new(data, [3, 2, 3]).expect("ok");
        let decomp = fit_als(&x, 1, 100, 1e-10).expect("als ok");
        let err = decomp.relative_error(&x).expect("err ok");
        assert!(err < 1e-6, "Rank-1 error {err:.2e} >= 1e-6");
    }

    #[test]
    fn test_cp_error_high_rank() {
        // Over-parameterised should have near-zero error
        let data: Vec<f64> = (0..60).map(|x| x as f64).collect();
        let x = Tensor3D::new(data, [3, 4, 5]).expect("ok");
        // rank >= 12 is an over-decomposition for a generic (3,4,5) tensor
        let decomp = fit_als(&x, 12, 300, 1e-8).expect("als ok");
        let err = decomp.relative_error(&x).expect("err ok");
        assert!(err < 0.05, "over-param error {err:.4} >= 0.05");
    }

    #[test]
    fn test_cp_lambda_positive() {
        let data: Vec<f64> = (0..24).map(|x| (x as f64) + 1.0).collect();
        let x = Tensor3D::new(data, [2, 3, 4]).expect("ok");
        let decomp = fit_als(&x, 3, 100, 1e-6).expect("als ok");
        for (r, &lam) in decomp.lambda.iter().enumerate() {
            assert!(lam >= 0.0, "lambda[{r}] = {lam} < 0");
        }
    }
}
