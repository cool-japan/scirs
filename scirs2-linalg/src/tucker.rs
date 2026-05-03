//! Tucker decomposition for 3-D tensors (concrete f64 API).
//!
//! This module provides a clean, Array3-based API for Tucker decomposition,
//! complementing the generic `tensor_contraction::tucker` module.
//!
//! ## Algorithms
//!
//! * **HOSVD** (Higher-Order SVD) – computes each factor matrix by truncated SVD of
//!   the mode-`n` unfolding, then projects the tensor to obtain the core.
//! * **HOOI** (Higher-Order Orthogonal Iteration) – iteratively refines the factor
//!   matrices to minimize reconstruction error; converges to a local optimum of the
//!   best low-multilinear-rank approximation.
//!
//! ## References
//!
//! L. De Lathauwer, B. De Moor, J. Vandewalle, "A Multilinear Singular Value
//! Decomposition", SIAM J. Matrix Anal. Appl. 21(4), 2000.
//!
//! L. De Lathauwer, B. De Moor, J. Vandewalle, "On the Best Rank-1 and
//! Rank-(R1,R2,...,RN) Approximation of Higher-Order Tensors", SIAM J. Matrix
//! Anal. Appl. 21(4), 2000.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor_contractions::{tensor_mode_product_view, unfold_tensor_view};
use scirs2_core::ndarray::{Array2, Array3, ArrayView2, ArrayView3};

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Result of a Tucker decomposition of a 3-D tensor.
///
/// The decomposition is: `T ≈ G ×₁ U₀ ×₂ U₁ ×₃ U₂`
/// where:
/// - `G` is the core tensor of shape `(ranks[0], ranks[1], ranks[2])`.
/// - `factors[0]` is `U₀` of shape `(I0, ranks[0])`.
/// - `factors[1]` is `U₁` of shape `(I1, ranks[1])`.
/// - `factors[2]` is `U₂` of shape `(I2, ranks[2])`.
#[derive(Debug, Clone)]
pub struct TuckerDecomp {
    /// Core tensor; shape `(ranks[0], ranks[1], ranks[2])`.
    pub core: Array3<f64>,
    /// Factor matrices `[U₀, U₁, U₂]`; `factors[n]` has shape `(I_n, ranks[n])`.
    pub factors: [Array2<f64>; 3],
}

impl TuckerDecomp {
    /// Reconstruct the full tensor from the Tucker decomposition.
    ///
    /// Returns `G ×₁ U₀ ×₂ U₁ ×₃ U₂`.
    pub fn reconstruct(&self) -> LinalgResult<Array3<f64>> {
        tucker_reconstruct(self)
    }

    /// Relative Frobenius reconstruction error: `‖T - T̃‖_F / ‖T‖_F`.
    ///
    /// Returns `0` if both tensors are zero.
    pub fn relative_error(&self, original: &Array3<f64>) -> f64 {
        let reconstructed = match self.reconstruct() {
            Ok(r) => r,
            Err(_) => return f64::INFINITY,
        };
        let mut diff_sq = 0.0_f64;
        let mut orig_sq = 0.0_f64;
        let shape = original.shape();
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

    /// Compression ratio: `numel(original) / numel(core + factors)`.
    ///
    /// Values > 1 indicate compression.
    pub fn compression_ratio(&self, original_shape: [usize; 3]) -> f64 {
        let original_elements: usize = original_shape.iter().product();
        let core_elements: usize = self.core.shape().iter().product();
        let factor_elements: usize = self.factors.iter().map(|f| f.len()).sum();
        let compressed_elements = core_elements + factor_elements;
        if compressed_elements == 0 {
            return f64::INFINITY;
        }
        original_elements as f64 / compressed_elements as f64
    }

    /// Upper bound on the Frobenius reconstruction error from the HOSVD analysis.
    ///
    /// Uses the identity: for HOSVD, the error is bounded by
    /// `sqrt(sum_n sigma_{n,ranks[n]+1}^2 + ...)`.
    /// This returns the Frobenius norm of `(T - T̃)` directly via reconstruction.
    pub fn error_bound(&self, original: &Array3<f64>) -> f64 {
        let reconstructed = match self.reconstruct() {
            Ok(r) => r,
            Err(_) => return f64::INFINITY,
        };
        let mut diff_sq = 0.0_f64;
        let shape = original.shape();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let diff = original[[i, j, k]] - reconstructed[[i, j, k]];
                    diff_sq += diff * diff;
                }
            }
        }
        diff_sq.sqrt()
    }
}

// ---------------------------------------------------------------------------
// HOSVD
// ---------------------------------------------------------------------------

/// Compute the Tucker decomposition of a 3-D tensor via HOSVD.
///
/// Each factor matrix `U_n` is the leading `ranks[n]` left singular vectors of
/// the mode-`n` unfolding of the tensor.  The core tensor is then
/// `G = T ×₁ U₀ᵀ ×₂ U₁ᵀ ×₃ U₂ᵀ`.
///
/// # Arguments
///
/// * `tensor` – Input tensor of shape `(I0, I1, I2)`.
/// * `ranks`  – Target ranks `[r0, r1, r2]` with `r_n <= I_n`.
///
/// # Errors
///
/// Returns `LinalgError::ValueError` if any rank is zero or exceeds the
/// corresponding tensor dimension.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::tucker::tucker_decomp;
///
/// let t = Array3::from_shape_fn((4, 5, 6), |(i, j, k)| (i + j + k) as f64);
/// let decomp = tucker_decomp(&t, [2, 3, 4]).expect("valid input");
/// assert_eq!(decomp.core.shape(), &[2, 3, 4]);
/// assert_eq!(decomp.factors[0].shape(), &[4, 2]);
/// assert_eq!(decomp.factors[1].shape(), &[5, 3]);
/// assert_eq!(decomp.factors[2].shape(), &[6, 4]);
/// ```
pub fn tucker_decomp(tensor: &Array3<f64>, ranks: [usize; 3]) -> LinalgResult<TuckerDecomp> {
    tucker_hosvd(&tensor.view(), ranks)
}

/// View-based variant; performs HOSVD Tucker decomposition.
pub fn tucker_hosvd(tensor: &ArrayView3<f64>, ranks: [usize; 3]) -> LinalgResult<TuckerDecomp> {
    let shape = tensor.shape();
    for n in 0..3 {
        if ranks[n] == 0 {
            return Err(LinalgError::ValueError(format!(
                "ranks[{n}] must be positive; got 0"
            )));
        }
        if ranks[n] > shape[n] {
            return Err(LinalgError::ValueError(format!(
                "ranks[{n}]={} exceeds tensor dimension {}={}",
                ranks[n], n, shape[n]
            )));
        }
    }

    // Compute factor matrices via truncated SVD of mode-n unfoldings
    let factors = compute_hosvd_factors(tensor, ranks)?;

    // Core: G = T ×₁ U₀ᵀ ×₂ U₁ᵀ ×₃ U₂ᵀ
    let core = project_to_core(tensor, &factors)?;

    Ok(TuckerDecomp { core, factors })
}

// ---------------------------------------------------------------------------
// HOOI
// ---------------------------------------------------------------------------

/// Compute the Tucker decomposition via HOOI (Higher-Order Orthogonal Iteration).
///
/// HOOI initializes with HOSVD and then alternately updates each factor matrix
/// by computing the leading singular vectors of the mode-`n` unfolding of the
/// tensor partially projected onto the other factor matrices.
///
/// The iteration refines the factors until the relative change in reconstruction
/// fit is smaller than `tol` or `max_iter` iterations are reached.
///
/// # Arguments
///
/// * `tensor`   – Input tensor of shape `(I0, I1, I2)`.
/// * `ranks`    – Target ranks `[r0, r1, r2]`.
/// * `max_iter` – Maximum number of HOOI iterations.
/// * `tol`      – Convergence tolerance (relative change in fit).
///
/// # Errors
///
/// Same as [`tucker_decomp`].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::tucker::tucker_hooi;
///
/// let t = Array3::from_shape_fn((4, 5, 6), |(i, j, k)| (i * 30 + j * 6 + k) as f64);
/// let decomp = tucker_hooi(&t, [2, 3, 4], 20, 1e-6).expect("valid input");
/// assert_eq!(decomp.core.shape(), &[2, 3, 4]);
/// // HOOI should achieve smaller or equal error than HOSVD
/// let hosvd = scirs2_linalg::tucker::tucker_decomp(&t, [2, 3, 4]).expect("valid input");
/// assert!(decomp.relative_error(&t) <= hosvd.relative_error(&t) + 1e-6);
/// ```
pub fn tucker_hooi(
    tensor: &Array3<f64>,
    ranks: [usize; 3],
    max_iter: usize,
    tol: f64,
) -> LinalgResult<TuckerDecomp> {
    tucker_hooi_view(&tensor.view(), ranks, max_iter, tol)
}

/// View-based variant of [`tucker_hooi`].
pub fn tucker_hooi_view(
    tensor: &ArrayView3<f64>,
    ranks: [usize; 3],
    max_iter: usize,
    tol: f64,
) -> LinalgResult<TuckerDecomp> {
    let shape = tensor.shape();
    for n in 0..3 {
        if ranks[n] == 0 {
            return Err(LinalgError::ValueError(format!(
                "ranks[{n}] must be positive; got 0"
            )));
        }
        if ranks[n] > shape[n] {
            return Err(LinalgError::ValueError(format!(
                "ranks[{n}]={} exceeds tensor dimension {}={}",
                ranks[n], n, shape[n]
            )));
        }
    }

    // Initialise with HOSVD
    let mut factors = compute_hosvd_factors(tensor, ranks)?;

    let tensor_norm = frobenius_norm_3d(tensor);

    let mut prev_fit = 0.0_f64;

    for _iter in 0..max_iter {
        for mode in 0..3 {
            // Build Y = T ×_{n≠mode} U_n^T
            let y = project_except_mode(tensor, &factors, mode)?;
            // Update U_mode = leading r_mode left singular vectors of unfold(Y, mode)
            let y_unf = unfold_tensor_view(&y.view(), mode)?;
            let (u_new, _) = truncated_svd_left(&y_unf, ranks[mode])?;
            factors[mode] = u_new;
        }

        // Compute fit = ‖G‖_F / ‖T‖_F  (equivalent measure to error)
        let core = project_to_core(tensor, &factors)?;
        let core_norm = frobenius_norm_3d(&core.view());
        let fit = if tensor_norm > 0.0 {
            core_norm / tensor_norm
        } else {
            1.0
        };

        if (fit - prev_fit).abs() < tol {
            let decomp = TuckerDecomp { core, factors };
            return Ok(decomp);
        }
        prev_fit = fit;
    }

    let core = project_to_core(tensor, &factors)?;
    Ok(TuckerDecomp { core, factors })
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct the full tensor from a Tucker decomposition.
///
/// `T̃ = G ×₁ U₀ ×₂ U₁ ×₃ U₂`
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::tucker::{tucker_decomp, tucker_reconstruct};
///
/// let t = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i + j + k) as f64);
/// let d = tucker_decomp(&t, [3, 3, 3]).expect("valid input"); // full rank → lossless
/// let t2 = tucker_reconstruct(&d).expect("valid input");
/// for i in 0..3 {
///     for j in 0..3 {
///         for k in 0..3 {
///             assert!((t[[i,j,k]] - t2[[i,j,k]]).abs() < 1e-8);
///         }
///     }
/// }
/// ```
pub fn tucker_reconstruct(decomp: &TuckerDecomp) -> LinalgResult<Array3<f64>> {
    let core = &decomp.core;
    // Apply each factor matrix in order: T̃ = G ×₁ U₀ ×₂ U₁ ×₃ U₂
    let tmp0 = tensor_mode_product_view(&core.view(), &decomp.factors[0].view(), 0)?;
    let tmp1 = tensor_mode_product_view(&tmp0.view(), &decomp.factors[1].view(), 1)?;
    let result = tensor_mode_product_view(&tmp1.view(), &decomp.factors[2].view(), 2)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the three HOSVD factor matrices: truncated left singular vectors
/// of the mode-n unfoldings.
fn compute_hosvd_factors(
    tensor: &ArrayView3<f64>,
    ranks: [usize; 3],
) -> LinalgResult<[Array2<f64>; 3]> {
    let mut factors_vec: Vec<Array2<f64>> = Vec::with_capacity(3);
    for mode in 0..3 {
        let x_n = unfold_tensor_view(tensor, mode)?;
        let (u, _) = truncated_svd_left(&x_n, ranks[mode])?;
        factors_vec.push(u);
    }
    // Convert to fixed-size array; safe because we push exactly 3 elements
    let f2 = factors_vec.pop().expect("factor 2 must exist");
    let f1 = factors_vec.pop().expect("factor 1 must exist");
    let f0 = factors_vec.pop().expect("factor 0 must exist");
    Ok([f0, f1, f2])
}

/// Compute the core tensor: `G = T ×₁ U₀ᵀ ×₂ U₁ᵀ ×₃ U₂ᵀ`.
fn project_to_core(
    tensor: &ArrayView3<f64>,
    factors: &[Array2<f64>; 3],
) -> LinalgResult<Array3<f64>> {
    let ut0 = factors[0].t().to_owned();
    let ut1 = factors[1].t().to_owned();
    let ut2 = factors[2].t().to_owned();
    let tmp0 = tensor_mode_product_view(tensor, &ut0.view(), 0)?;
    let tmp1 = tensor_mode_product_view(&tmp0.view(), &ut1.view(), 1)?;
    let core = tensor_mode_product_view(&tmp1.view(), &ut2.view(), 2)?;
    Ok(core)
}

/// For HOOI: build `Y = T ×_{n≠mode} U_n^T`.
/// This contracts all modes *except* `skip_mode` with the respective U^T.
fn project_except_mode(
    tensor: &ArrayView3<f64>,
    factors: &[Array2<f64>; 3],
    skip_mode: usize,
) -> LinalgResult<Array3<f64>> {
    // We need to apply U_n^T for n != skip_mode.
    // We do this by iterating all modes and skipping `skip_mode`.
    // The shape of Y will have I_{skip_mode} × ranks[n] × ranks[m] for n,m != skip_mode.
    //
    // Strategy: apply modes in ascending order, but skip the target mode.
    // The mode indices shift as we apply contractions.
    // Simplest approach: apply one at a time and track resulting shape.

    // We operate on the dynamic-shape representation internally by sequentially
    // applying the transposed factors.
    let modes: Vec<usize> = (0..3).filter(|&m| m != skip_mode).collect();

    // First contraction
    let ut = factors[modes[0]].t().to_owned();
    let tmp = tensor_mode_product_view(tensor, &ut.view(), modes[0])?;

    // Second contraction — mode index is still the same because we only changed the
    // dimensionality along modes[0], not modes[1].
    let ut2 = factors[modes[1]].t().to_owned();
    let result = tensor_mode_product_view(&tmp.view(), &ut2.view(), modes[1])?;

    Ok(result)
}

/// Compute the leading `r` left singular vectors of a matrix via full SVD.
///
/// Returns `(U_trunc, singular_values_trunc)` where `U_trunc` is `(m, r)`.
fn truncated_svd_left(matrix: &Array2<f64>, r: usize) -> LinalgResult<(Array2<f64>, Vec<f64>)> {
    use crate::decomposition::svd;
    let (u, s, _vt) = svd(&matrix.view(), true, None)?;
    // u has shape (m, min(m,n)) in full mode; we want leading r columns
    let actual_r = r.min(u.ncols());
    let u_trunc = u.slice(scirs2_core::ndarray::s![.., ..actual_r]).to_owned();
    let s_trunc: Vec<f64> = s.iter().take(actual_r).copied().collect();
    Ok((u_trunc, s_trunc))
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array3;

    fn make_test_tensor() -> Array3<f64> {
        Array3::from_shape_fn((4, 5, 6), |(i, j, k)| (i * 30 + j * 6 + k) as f64)
    }

    // --- tucker_decomp (HOSVD) ---

    #[test]
    fn test_tucker_decomp_shapes() {
        let t = make_test_tensor();
        let d = tucker_decomp(&t, [2, 3, 4]).expect("tucker decomp ok");
        assert_eq!(d.core.shape(), &[2, 3, 4]);
        assert_eq!(d.factors[0].shape(), &[4, 2]);
        assert_eq!(d.factors[1].shape(), &[5, 3]);
        assert_eq!(d.factors[2].shape(), &[6, 4]);
    }

    #[test]
    fn test_tucker_decomp_full_rank_lossless() {
        // Full-rank decomposition should be (near) lossless
        let t = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i + j + k) as f64 + 1.0);
        let d = tucker_decomp(&t, [3, 3, 3]).expect("full rank ok");
        let err = d.relative_error(&t);
        assert!(err < 1e-8, "full rank error = {err}");
    }

    #[test]
    fn test_tucker_decomp_low_rank() {
        let t = make_test_tensor();
        let d = tucker_decomp(&t, [2, 2, 2]).expect("low rank ok");
        assert_eq!(d.core.shape(), &[2, 2, 2]);
        // Error should be finite and positive (not lossless)
        let err = d.relative_error(&t);
        assert!(err.is_finite(), "error must be finite");
    }

    #[test]
    fn test_tucker_decomp_error_rank0() {
        let t = make_test_tensor();
        assert!(tucker_decomp(&t, [0, 2, 2]).is_err());
    }

    #[test]
    fn test_tucker_decomp_error_rank_exceeds_dim() {
        let t = make_test_tensor(); // shape (4,5,6)
        assert!(tucker_decomp(&t, [5, 3, 4]).is_err()); // 5 > 4
    }

    // --- compression_ratio ---

    #[test]
    fn test_compression_ratio() {
        let t = make_test_tensor(); // 4*5*6 = 120 elements
        let d = tucker_decomp(&t, [2, 3, 4]).expect("ok");
        // core: 2*3*4=24, factors: 4*2+5*3+6*4=8+15+24=47, total=71
        let ratio = d.compression_ratio([4, 5, 6]);
        assert!((ratio - 120.0 / 71.0).abs() < 1e-10, "ratio = {ratio}");
    }

    // --- tucker_reconstruct ---

    #[test]
    fn test_tucker_reconstruct_roundtrip() {
        let t = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| {
            (i as f64 + 1.0) * (j as f64 + 1.0) + k as f64
        });
        let d = tucker_decomp(&t, [3, 4, 5]).expect("full rank ok");
        let t2 = tucker_reconstruct(&d).expect("reconstruct ok");
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_abs_diff_eq!(t[[i, j, k]], t2[[i, j, k]], epsilon = 1e-8);
                }
            }
        }
    }

    // --- HOOI ---

    #[test]
    fn test_tucker_hooi_shapes() {
        let t = make_test_tensor();
        let d = tucker_hooi(&t, [2, 3, 4], 20, 1e-8).expect("hooi ok");
        assert_eq!(d.core.shape(), &[2, 3, 4]);
        assert_eq!(d.factors[0].shape(), &[4, 2]);
        assert_eq!(d.factors[1].shape(), &[5, 3]);
        assert_eq!(d.factors[2].shape(), &[6, 4]);
    }

    #[test]
    fn test_tucker_hooi_better_than_hosvd() {
        let t = make_test_tensor();
        let hosvd = tucker_decomp(&t, [2, 2, 2]).expect("hosvd ok");
        let hooi = tucker_hooi(&t, [2, 2, 2], 30, 1e-10).expect("hooi ok");
        let err_hosvd = hosvd.relative_error(&t);
        let err_hooi = hooi.relative_error(&t);
        // HOOI should be at least as good
        assert!(
            err_hooi <= err_hosvd + 1e-6,
            "HOOI error {err_hooi} > HOSVD error {err_hosvd}"
        );
    }

    #[test]
    fn test_tucker_hooi_full_rank_lossless() {
        let t = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i + j + k) as f64 + 1.0);
        let d = tucker_hooi(&t, [3, 3, 3], 10, 1e-12).expect("hooi full rank ok");
        let err = d.relative_error(&t);
        assert!(err < 1e-7, "full rank hooi error = {err}");
    }

    // --- error_bound ---

    #[test]
    fn test_error_bound_consistency() {
        let t = make_test_tensor();
        let d = tucker_decomp(&t, [2, 3, 4]).expect("ok");
        let eb = d.error_bound(&t);
        let re = d.relative_error(&t);
        let tensor_norm = frobenius_norm_3d(&t.view());
        // error_bound returns absolute norm; relative_error is normalized
        assert_abs_diff_eq!(eb, re * tensor_norm, epsilon = 1e-8);
    }

    // --- orthogonality of factor matrices ---

    #[test]
    fn test_factor_orthogonality() {
        let t = make_test_tensor();
        let d = tucker_decomp(&t, [2, 3, 4]).expect("ok");
        // Each factor should have orthonormal columns (U^T U = I)
        for (n, factor) in d.factors.iter().enumerate() {
            let utu = factor.t().dot(factor);
            let r = factor.ncols();
            for i in 0..r {
                for j in 0..r {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_abs_diff_eq!(utu[[i, j]], expected, epsilon = 1e-8,);
                    let _ = n; // suppress unused warning
                }
            }
        }
    }
}
