//! Canonical Polyadic (CP / PARAFAC) decomposition for 3-D tensors (concrete f64 API).
//!
//! This module provides a clean, Array3-based API for CP decomposition,
//! complementing the generic `tensor_contraction::cp` module.
//!
//! ## Algorithm
//!
//! **ALS** (Alternating Least Squares): at each step, fix all factor matrices
//! except one and solve the resulting linear least-squares problem.  The update
//! for mode `n` is:
//!
//! ```text
//! A^(n) ← X_(n) · (⊙_{k≠n} A^(k)) · (⊙_{k≠n} (A^(k)ᵀ A^(k)))⁻¹
//! ```
//!
//! where `X_(n)` is the mode-`n` unfolding of the tensor and `⊙` denotes the
//! Khatri-Rao product.
//!
//! ## Degeneracy / Swamping Detection
//!
//! CP-ALS can exhibit *swamping* (large cancelling components) or *degeneracy*
//! (unbounded factors).  After convergence this module checks:
//! - **Degeneracy**: whether any component norm is disproportionately large
//!   relative to the reconstruction.
//! - **Swamping**: whether factor column norms vary by more than three orders of
//!   magnitude (a practical heuristic).
//!
//! ## References
//!
//! T. Kolda, B. Bader, "Tensor Decompositions and Applications", SIAM Rev. 2009.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor_contractions::{khatri_rao_view, unfold_tensor_view};
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

// ---------------------------------------------------------------------------
// Public structs
// ---------------------------------------------------------------------------

/// Result of a CP/PARAFAC decomposition.
///
/// The decomposition approximates `T ≈ sum_{r=1}^{R} λ_r · a^(0)_r ⊗ a^(1)_r ⊗ a^(2)_r`
/// where:
/// - `lambdas[r]` is the weight (norm) of component `r`.
/// - `factors[n][:,r]` is the unit-norm factor vector for mode `n`, component `r`.
#[derive(Debug, Clone)]
pub struct CpDecomp {
    /// Factor matrices `[A₀, A₁, A₂]`; each `Aₙ` has shape `(Iₙ, rank)` with
    /// unit-norm columns.
    pub factors: [Array2<f64>; 3],
    /// Component weights (norms), length `rank`.
    pub lambdas: Array1<f64>,
}

/// Diagnostics returned alongside the decomposition.
#[derive(Debug, Clone)]
pub struct CpDiagnostics {
    /// Whether degeneracy was detected (large cancelling components).
    pub degenerate: bool,
    /// Whether swamping was detected (column norms varying by > 3 orders of magnitude).
    pub swamping: bool,
    /// Relative Frobenius reconstruction error `‖T - T̃‖_F / ‖T‖_F`.
    pub relative_error: f64,
    /// Number of ALS iterations executed.
    pub iterations: usize,
    /// Whether the ALS converged within tolerance before `max_iter`.
    pub converged: bool,
}

impl CpDecomp {
    /// Reconstruct the full tensor from the CP decomposition.
    pub fn reconstruct(&self) -> Array3<f64> {
        cp_reconstruct(self)
    }

    /// Relative Frobenius error vs. original tensor.
    pub fn relative_error(&self, original: &Array3<f64>) -> f64 {
        let reconstructed = self.reconstruct();
        let shape = original.shape();
        let mut diff_sq = 0.0_f64;
        let mut orig_sq = 0.0_f64;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let diff = original[[i, j, k]] - reconstructed[[i, j, k]];
                    diff_sq += diff * diff;
                    orig_sq += original[[i, j, k]] * original[[i, j, k]];
                }
            }
        }
        if orig_sq == 0.0 {
            if diff_sq == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (diff_sq / orig_sq).sqrt()
        }
    }

    /// Compute the Frobenius norms of each component (weight × factors).
    ///
    /// Returns an array of length `rank`; element `r` is `|λ_r|` because the
    /// factor columns are normalised to unit norm.
    pub fn cp_norms(&self) -> Array1<f64> {
        self.lambdas.mapv(f64::abs)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the CP/PARAFAC decomposition of a 3-D tensor via ALS.
///
/// # Arguments
///
/// * `tensor`   – Input tensor of shape `(I0, I1, I2)`.
/// * `rank`     – Number of components `R`.
/// * `max_iter` – Maximum number of ALS iterations.
/// * `tol`      – Relative convergence tolerance on reconstruction error.
///
/// # Returns
///
/// `(CpDecomp, CpDiagnostics)` on success.
///
/// # Errors
///
/// Returns `LinalgError::ValueError` if `rank == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::cp_decomposition::cp_decomp;
///
/// // Rank-1 tensor: outer product of [1,2] ⊗ [1,1,1] ⊗ [1,1]
/// let t = Array3::from_shape_fn((2, 3, 2), |(i, j, k)| {
///     (i + 1) as f64 * 1.0 * 1.0
/// });
/// let (decomp, diag) = cp_decomp(&t, 1, 200, 1e-8).expect("valid input");
/// assert_eq!(decomp.factors[0].shape(), &[2, 1]);
/// assert!(diag.relative_error < 0.5);
/// ```
pub fn cp_decomp(
    tensor: &Array3<f64>,
    rank: usize,
    max_iter: usize,
    tol: f64,
) -> LinalgResult<(CpDecomp, CpDiagnostics)> {
    cp_decomp_view(&tensor.view(), rank, max_iter, tol)
}

/// View-based variant of [`cp_decomp`].
pub fn cp_decomp_view(
    tensor: &ArrayView3<f64>,
    rank: usize,
    max_iter: usize,
    tol: f64,
) -> LinalgResult<(CpDecomp, CpDiagnostics)> {
    if rank == 0 {
        return Err(LinalgError::ValueError("rank must be >= 1".into()));
    }

    let shape = tensor.shape();
    let (i0, i1, i2) = (shape[0], shape[1], shape[2]);

    // Precompute unfoldings
    let x0 = unfold_tensor_view(tensor, 0)?;
    let x1 = unfold_tensor_view(tensor, 1)?;
    let x2 = unfold_tensor_view(tensor, 2)?;

    let mut factors: [Array2<f64>; 3] = [
        init_factor(i0, rank, 0),
        init_factor(i1, rank, 1),
        init_factor(i2, rank, 2),
    ];
    let unfoldings = [x0, x1, x2];

    let tensor_norm = frobenius_norm_3d(tensor);

    let mut prev_error = f64::INFINITY;
    let mut converged = false;
    let mut final_iter = max_iter;

    // Track lambdas across iterations — factors are unit-normalised inside the loop
    // so we must NOT call normalise_factors again after the loop (it would give lambda=1).
    let mut lambdas = Array1::<f64>::ones(rank);

    for iter in 0..max_iter {
        // Update all modes without normalization (standard ALS)
        for mode in 0..3_usize {
            // Khatri-Rao of all factors except `mode`
            let kr = khatri_rao_all_except(&factors, mode)?;
            // Gram matrix: V = ⊙_{k≠mode} (A_k^T A_k)  (Hadamard product)
            let v = gram_hadamard(&factors, mode);
            // Update: A_mode = X_(mode) · KR · V^{-1}
            let rhs = unfoldings[mode].dot(&kr);
            let v_inv = pseudo_inverse_small(&v)?;
            factors[mode] = rhs.dot(&v_inv);
        }

        // Normalise factors once per iteration; absorb norms into lambdas
        lambdas = normalise_factors(&mut factors);

        // Compute reconstruction error
        let error = compute_error_from_factors(tensor, &factors, &lambdas, tensor_norm);

        let rel_change = (prev_error - error).abs() / (prev_error.max(1e-30));
        // Declare convergence when relative change is small AND error is reasonable.
        if rel_change < tol && iter > 0 && error < 0.5 {
            final_iter = iter + 1;
            converged = true;

            let decomp = build_decomp(factors, lambdas);
            let diag = build_diagnostics(&decomp, tensor, tensor_norm, final_iter, converged);
            return Ok((decomp, diag));
        }
        prev_error = error;
    }

    // Use the lambdas from the final iteration — do NOT call normalise_factors again
    // because factors are already unit-norm and a second call would return lambdas = 1.
    let decomp = build_decomp(factors, lambdas);
    let diag = build_diagnostics(&decomp, tensor, tensor_norm, final_iter, converged);
    Ok((decomp, diag))
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct a 3-D tensor from a CP decomposition.
///
/// `T̃[i,j,k] = sum_r  λ_r * A₀[i,r] * A₁[j,r] * A₂[k,r]`
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::cp_decomposition::{cp_decomp, cp_reconstruct};
///
/// let t = Array3::from_shape_fn((2, 3, 2), |(i, j, k)| (i + j + k) as f64);
/// let (d, _) = cp_decomp(&t, 3, 100, 1e-6).expect("valid input");
/// let t2 = cp_reconstruct(&d);
/// assert_eq!(t2.shape(), t.shape());
/// ```
pub fn cp_reconstruct(decomp: &CpDecomp) -> Array3<f64> {
    let rank = decomp.lambdas.len();
    let i0 = decomp.factors[0].nrows();
    let i1 = decomp.factors[1].nrows();
    let i2 = decomp.factors[2].nrows();
    let mut result = Array3::<f64>::zeros((i0, i1, i2));
    for r in 0..rank {
        let lam = decomp.lambdas[r];
        for i in 0..i0 {
            for j in 0..i1 {
                for k in 0..i2 {
                    result[[i, j, k]] += lam
                        * decomp.factors[0][[i, r]]
                        * decomp.factors[1][[j, r]]
                        * decomp.factors[2][[k, r]];
                }
            }
        }
    }
    result
}

/// Compute the Frobenius norms of each CP component.
///
/// For a properly normalised decomposition with unit-norm factor columns,
/// the component norm is simply `|λ_r|`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::cp_decomposition::{cp_decomp, cp_norms};
///
/// let t = Array3::from_shape_fn((2, 3, 2), |(i, j, k)| (i + j + k) as f64 + 1.0);
/// let (d, _) = cp_decomp(&t, 2, 50, 1e-6).expect("valid input");
/// let norms = cp_norms(&d);
/// assert_eq!(norms.len(), 2);
/// ```
pub fn cp_norms(decomp: &CpDecomp) -> Array1<f64> {
    decomp.cp_norms()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Deterministic factor initialisation: entries `(i+1)*(r+1)/(rows*rank)` normalized.
fn init_factor(rows: usize, rank: usize, mode: usize) -> Array2<f64> {
    // Initialise factor matrix with unit-norm columns.
    // Use a "staggered uniform" pattern: each column r gets a vector derived
    // from [f(0,r), f(1,r), ..., f(rows-1,r)] where f(i,r) = cos(pi*(i+0.5+r*rows/rank)/rows).
    // This places the column directions evenly on the unit sphere's great circle,
    // ensuring no two columns are collinear even for rank == rows.
    let mut factor = Array2::<f64>::zeros((rows, rank));
    let pi = std::f64::consts::PI;
    for r in 0..rank {
        for i in 0..rows {
            // Use sine for mode=1 and cosine for others to diversify across modes
            let angle = pi
                * (i as f64 + 0.5 + (r * rows / rank.max(1)) as f64 + mode as f64 * 0.37)
                / rows as f64;
            factor[[i, r]] = angle.cos();
        }
        // Normalize column r
        let norm = col_norm(&factor.view(), r);
        if norm > 1e-30 {
            for i in 0..rows {
                factor[[i, r]] /= norm;
            }
        } else {
            factor[[r % rows, r]] = 1.0;
        }
    }
    factor
}

/// Frobenius norm of a 3-D tensor view.
fn frobenius_norm_3d(tensor: &ArrayView3<f64>) -> f64 {
    let shape = tensor.shape();
    let mut sq = 0.0_f64;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let v = tensor[[i, j, k]];
                sq += v * v;
            }
        }
    }
    sq.sqrt()
}

/// Column Euclidean norm.
fn col_norm(matrix: &ArrayView2<f64>, col: usize) -> f64 {
    let mut sq = 0.0_f64;
    for i in 0..matrix.nrows() {
        let v = matrix[[i, col]];
        sq += v * v;
    }
    sq.sqrt()
}

/// Normalise each column of every factor matrix in-place.
/// Returns the vector of per-component weights (product of column norms).
fn normalise_factors(factors: &mut [Array2<f64>; 3]) -> Array1<f64> {
    let rank = factors[0].ncols();
    let mut lambdas = Array1::ones(rank);
    for factor in factors.iter_mut() {
        for r in 0..rank {
            let norm = col_norm(&factor.view(), r);
            if norm > 1e-30 {
                lambdas[r] *= norm;
                for i in 0..factor.nrows() {
                    factor[[i, r]] /= norm;
                }
            }
        }
    }
    lambdas
}

/// Compute the Khatri-Rao product of all factor matrices except `skip`.
fn khatri_rao_all_except(factors: &[Array2<f64>; 3], skip: usize) -> LinalgResult<Array2<f64>> {
    // Modes in ascending order, excluding `skip`.
    let other_modes: Vec<usize> = (0..3).filter(|&m| m != skip).collect();
    let result = khatri_rao_view(
        &factors[other_modes[0]].view(),
        &factors[other_modes[1]].view(),
    )?;
    Ok(result)
}

/// Hadamard product of all `A_k^T A_k` for k != skip.
/// Returns an `(R, R)` matrix.
fn gram_hadamard(factors: &[Array2<f64>; 3], skip: usize) -> Array2<f64> {
    let rank = factors[0].ncols();
    let mut v = Array2::<f64>::ones((rank, rank));
    for (mode, factor) in factors.iter().enumerate() {
        if mode == skip {
            continue;
        }
        let ftf = factor.t().dot(factor);
        for i in 0..rank {
            for j in 0..rank {
                v[[i, j]] *= ftf[[i, j]];
            }
        }
    }
    v
}

/// Moore-Penrose pseudo-inverse for small square matrices (via SVD).
/// Uses SVD for numerical stability, falling back to regularized inverse.
fn pseudo_inverse_small(m: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    let n = m.nrows();
    // Attempt SVD-based pseudo-inverse for best numerical stability
    if let Ok((u, s, vt)) = crate::decomposition::svd_f64_lapack(&m.view(), false) {
        let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
        let thr = f64::EPSILON * (n as f64) * s_max;
        let k = s.len();
        let mut result = Array2::<f64>::zeros((n, n));
        for i in 0..k {
            if s[i] > thr {
                let inv_si = 1.0 / s[i];
                let v_i = vt.row(i); // length n
                let u_i = u.column(i); // length n
                for r in 0..n {
                    for c in 0..n {
                        result[[r, c]] += inv_si * v_i[r] * u_i[c];
                    }
                }
            }
        }
        return Ok(result);
    }
    // Fallback: regularised matrix inverse
    let reg = 1e-8_f64; // stronger regularization for stability
    let mut m_reg = m.clone();
    for i in 0..n {
        m_reg[[i, i]] += reg;
    }
    crate::inv(&m_reg.view(), None).or_else(|_| {
        // Last resort: diagonal pseudo-inverse
        let mut diag_inv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            if m_reg[[i, i]].abs() > 1e-30 {
                diag_inv[[i, i]] = 1.0 / m_reg[[i, i]];
            }
        }
        Ok(diag_inv)
    })
}

/// Compute the reconstruction error given factors (with temporarily absorbed norms).
fn compute_error_from_factors(
    tensor: &ArrayView3<f64>,
    factors: &[Array2<f64>; 3],
    lambdas: &Array1<f64>,
    tensor_norm: f64,
) -> f64 {
    let rank = lambdas.len();
    let shape = tensor.shape();
    let (i0, i1, i2) = (shape[0], shape[1], shape[2]);

    let mut diff_sq = 0.0_f64;
    for i in 0..i0 {
        for j in 0..i1 {
            for k in 0..i2 {
                let mut approx = 0.0_f64;
                for r in 0..rank {
                    approx +=
                        lambdas[r] * factors[0][[i, r]] * factors[1][[j, r]] * factors[2][[k, r]];
                }
                let diff = tensor[[i, j, k]] - approx;
                diff_sq += diff * diff;
            }
        }
    }
    let abs_err = diff_sq.sqrt();
    if tensor_norm > 0.0 {
        abs_err / tensor_norm
    } else {
        abs_err
    }
}

/// Construct a `CpDecomp` from raw factors and lambdas.
fn build_decomp(factors: [Array2<f64>; 3], lambdas: Array1<f64>) -> CpDecomp {
    CpDecomp { factors, lambdas }
}

/// Construct `CpDiagnostics`, including degeneracy / swamping checks.
fn build_diagnostics(
    decomp: &CpDecomp,
    tensor: &ArrayView3<f64>,
    tensor_norm: f64,
    iterations: usize,
    converged: bool,
) -> CpDiagnostics {
    let rank = decomp.lambdas.len();

    // Relative error
    let rel_err = compute_error_from_factors(tensor, &decomp.factors, &decomp.lambdas, tensor_norm);

    // Degeneracy: check if any component weight is much larger than the others
    let max_lambda: f64 = decomp.lambdas.iter().cloned().fold(0.0_f64, f64::max).abs();
    let sum_lambda: f64 = decomp.lambdas.iter().cloned().map(f64::abs).sum::<f64>();
    let degenerate = if rank > 1 && sum_lambda > 0.0 {
        max_lambda / (sum_lambda / rank as f64) > 1e3
    } else {
        false
    };

    // Swamping: large spread in per-mode column norms
    let swamping = check_swamping(decomp);

    CpDiagnostics {
        degenerate,
        swamping,
        relative_error: rel_err,
        iterations,
        converged,
    }
}

/// Check if any factor column norms vary by more than 1000× across the rank-1 components.
fn check_swamping(decomp: &CpDecomp) -> bool {
    let rank = decomp.lambdas.len();
    if rank <= 1 {
        return false;
    }
    let mut max_norm = 0.0_f64;
    let mut min_norm = f64::INFINITY;
    for r in 0..rank {
        let lambda = decomp.lambdas[r].abs();
        if lambda > max_norm {
            max_norm = lambda;
        }
        if lambda < min_norm {
            min_norm = lambda;
        }
    }
    if min_norm > 1e-30 {
        max_norm / min_norm > 1000.0
    } else {
        max_norm > 1e3
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array3;

    fn make_rank1_tensor() -> Array3<f64> {
        // outer product of [1.0, 2.0] ⊗ [1.0, 1.0, 1.0] ⊗ [2.0, 3.0]
        Array3::from_shape_fn((2, 3, 2), |(i, j, k)| {
            let a = [1.0_f64, 2.0][i];
            let b = 1.0_f64;
            let c = [2.0_f64, 3.0][k];
            let _ = j;
            a * b * c
        })
    }

    // --- cp_decomp basic shapes ---

    #[test]
    fn test_cp_decomp_shapes() {
        let t = Array3::<f64>::ones((2, 3, 4));
        let (d, _) = cp_decomp(&t, 2, 100, 1e-6).expect("cp_decomp ok");
        assert_eq!(d.factors[0].shape(), &[2, 2]);
        assert_eq!(d.factors[1].shape(), &[3, 2]);
        assert_eq!(d.factors[2].shape(), &[4, 2]);
        assert_eq!(d.lambdas.len(), 2);
    }

    #[test]
    fn test_cp_decomp_rank0_error() {
        let t = Array3::<f64>::ones((2, 3, 4));
        assert!(cp_decomp(&t, 0, 100, 1e-6).is_err());
    }

    // --- cp_reconstruct ---

    #[test]
    fn test_cp_reconstruct_shape() {
        let t = Array3::<f64>::ones((2, 3, 4));
        let (d, _) = cp_decomp(&t, 2, 50, 1e-4).expect("ok");
        let r = cp_reconstruct(&d);
        assert_eq!(r.shape(), t.shape());
    }

    #[test]
    fn test_cp_reconstruct_rank1() {
        // A rank-1 tensor should be recoverable exactly (rank == 1)
        let t = make_rank1_tensor();
        let (d, diag) = cp_decomp(&t, 1, 300, 1e-8).expect("rank1 ok");
        // Error may not be machine-zero due to ALS initialisation, but should be small
        assert!(diag.relative_error < 0.1, "error = {}", diag.relative_error);
        let t2 = d.reconstruct();
        assert_eq!(t2.shape(), t.shape());
    }

    // --- cp_norms ---

    #[test]
    fn test_cp_norms_positive() {
        let t = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i + j + k + 1) as f64);
        let (d, _) = cp_decomp(&t, 3, 50, 1e-4).expect("ok");
        let norms = cp_norms(&d);
        assert_eq!(norms.len(), 3);
        for n in norms.iter() {
            assert!(*n >= 0.0, "norm must be non-negative");
        }
    }

    // --- factor column orthonormality ---

    #[test]
    fn test_factor_columns_unit_norm() {
        let t = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| (i * 20 + j * 5 + k) as f64);
        let (d, _) = cp_decomp(&t, 3, 80, 1e-6).expect("ok");
        for factor in &d.factors {
            let rank = factor.ncols();
            for r in 0..rank {
                let norm = col_norm(&factor.view(), r);
                assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
            }
        }
    }

    // --- diagnostics fields ---

    #[test]
    fn test_diagnostics_fields_present() {
        let t = Array3::from_shape_fn((2, 3, 2), |(i, j, k)| (i + j + k) as f64 + 1.0);
        let (_, diag) = cp_decomp(&t, 2, 50, 1e-4).expect("ok");
        assert!(diag.relative_error.is_finite());
        assert!(diag.iterations > 0);
    }

    // --- degeneracy / swamping thresholds ---

    #[test]
    fn test_no_degeneracy_rank1() {
        let t = make_rank1_tensor();
        let (_, diag) = cp_decomp(&t, 1, 200, 1e-8).expect("ok");
        // Single component can't be degenerate (need rank > 1)
        assert!(!diag.degenerate);
    }

    // --- reconstruction consistent with relative_error ---

    #[test]
    fn test_relative_error_consistent() {
        let t = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i * 9 + j * 3 + k) as f64);
        let (d, diag) = cp_decomp(&t, 3, 100, 1e-6).expect("ok");
        let manual_error = d.relative_error(&t);
        assert_abs_diff_eq!(manual_error, diag.relative_error, epsilon = 1e-8);
    }

    // --- reconstruction decreases with higher rank ---

    #[test]
    fn test_higher_rank_lower_error() {
        let t = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| (i + j + k) as f64 + 1.0);
        let (_, diag1) = cp_decomp(&t, 1, 100, 1e-6).expect("rank1 ok");
        let (_, diag4) = cp_decomp(&t, 4, 100, 1e-6).expect("rank4 ok");
        assert!(
            diag4.relative_error <= diag1.relative_error + 1e-6,
            "rank-4 error {} > rank-1 error {}",
            diag4.relative_error,
            diag1.relative_error
        );
    }
}
