//! Higher-Order SVD (HOSVD) and Higher-Order Orthogonal Iteration (HOOI)
//! for dense N-way tensors.
//!
//! ## Algorithms
//!
//! ### HOSVD
//!
//! The *Higher-Order Singular Value Decomposition* (De Lathauwer et al. 2000)
//! computes a multilinear (Tucker) approximation by:
//!
//! 1. For each mode `n`, form the mode-n unfolding `T_(n)`.
//! 2. Compute a truncated SVD of `T_(n)` and keep the top `ranks[n]` left
//!    singular vectors as factor matrix `U_n`.
//! 3. Project the tensor into the factor subspaces:
//!    `G = T ×_1 U_1^T ×_2 U_2^T ×_3 … ×_N U_N^T`.
//!
//! ### HOOI
//!
//! *Higher-Order Orthogonal Iteration* (De Lathauwer et al. 2000b) refines
//! the HOSVD solution via alternating updates, converging to a stationary
//! point (local minimum) of the Tucker approximation error.  Each iteration
//! for mode `n` computes the top-`ranks[n]` SVD of
//!
//! ```text
//! Y_(n) = (T ×_{m≠n} U_m^T)_(n)
//! ```
//!
//! and updates `U_n`.
//!
//! ### truncated_hosvd
//!
//! Automatically selects ranks based on a relative energy threshold `eps`:
//! retains the smallest prefix of singular values whose squared sum accounts
//! for at least `(1 - eps^2)` of the total squared singular-value energy for
//! that mode.
//!
//! ## References
//!
//! - De Lathauwer, De Moor, Vandewalle (2000). "A Multilinear SVD".
//!   *SIAM J. Matrix Anal. Appl.* 21(4).
//! - De Lathauwer, De Moor, Vandewalle (2000b). "On the Best Rank-(R₁,…,Rₙ)
//!   Approximation". *SIAM J. Matrix Anal. Appl.* 21(4).

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::tensor::core::{Tensor, TensorScalar};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a Tucker / HOSVD decomposition.
///
/// The decomposition approximates the original tensor as
/// `T ≈ G ×_1 U_1 ×_2 U_2 ×_3 … ×_N U_N`
/// where:
/// - `core` (`G`) has shape `(ranks[0], ranks[1], …, ranks[N-1])`.
/// - `factors[n]` (`U_n`) has shape `(shape[n], ranks[n])` with orthonormal
///   columns.
#[derive(Debug, Clone)]
pub struct HOSVDResult<F> {
    /// Compressed core tensor `G`.
    pub core: Tensor<F>,
    /// Factor matrices `[U_1, …, U_N]`; `factors[n]` has shape
    /// `(shape[n], ranks[n])`.
    pub factors: Vec<Array2<F>>,
}

impl<F: TensorScalar> HOSVDResult<F> {
    /// Reconstruct the full tensor from this decomposition.
    ///
    /// Computes `G ×_1 U_1 ×_2 U_2 ×_N …`.
    pub fn reconstruct(&self) -> LinalgResult<Tensor<F>> {
        let mut result = self.core.clone();
        for (n, factor) in self.factors.iter().enumerate() {
            result = result.mode_product(factor, n)?;
        }
        Ok(result)
    }

    /// Relative Frobenius reconstruction error `‖T - T̃‖_F / ‖T‖_F`.
    ///
    /// Returns `F::zero()` when both tensors are zero.
    pub fn relative_error(&self, original: &Tensor<F>) -> LinalgResult<F> {
        let reconstructed = self.reconstruct()?;
        let orig_norm = original.frobenius_norm();
        if orig_norm == F::zero() {
            let diff_norm = {
                let diff_sq: F = original
                    .data
                    .iter()
                    .zip(reconstructed.data.iter())
                    .map(|(&a, &b)| {
                        let d = a - b;
                        d * d
                    })
                    .fold(F::zero(), |acc, x| acc + x);
                diff_sq.sqrt()
            };
            return Ok(if diff_norm == F::zero() {
                F::zero()
            } else {
                F::infinity()
            });
        }
        let diff_sq: F = original
            .data
            .iter()
            .zip(reconstructed.data.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .fold(F::zero(), |acc, x| acc + x);
        Ok((diff_sq / (orig_norm * orig_norm)).sqrt())
    }

    /// Compression ratio: `numel(original) / numel(core + factors)`.
    pub fn compression_ratio(&self, original_shape: &[usize]) -> F {
        let orig: usize = original_shape.iter().product();
        let core_n: usize = self.core.shape.iter().product();
        let factors_n: usize = self.factors.iter().map(|f| f.len()).sum();
        let compressed = core_n + factors_n;
        if compressed == 0 {
            return F::infinity();
        }
        F::from(orig).unwrap_or(F::zero()) / F::from(compressed).unwrap_or(F::one())
    }
}

// ---------------------------------------------------------------------------
// HOSVD
// ---------------------------------------------------------------------------

/// Compute the Higher-Order SVD (Tucker decomposition) of an N-way tensor.
///
/// # Arguments
///
/// * `tensor` – input tensor.
/// * `ranks`  – multilinear rank `(R_1, …, R_N)`; must satisfy
///   `0 < ranks[n] <= tensor.shape[n]`.
///
/// # Returns
///
/// A [`HOSVDResult`] containing the core tensor and factor matrices.
///
/// # Errors
///
/// Returns an error when `ranks` length doesn't match `ndim`, or any rank
/// is zero or exceeds the corresponding mode size.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
/// use scirs2_linalg::tensor::hosvd::hosvd;
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let tensor = Tensor::new(data, vec![2, 3, 4]).expect("valid");
/// let result = hosvd(&tensor, &[2, 2, 3]).expect("hosvd ok");
/// assert_eq!(result.core.shape, vec![2, 2, 3]);
/// ```
pub fn hosvd<F: TensorScalar>(tensor: &Tensor<F>, ranks: &[usize]) -> LinalgResult<HOSVDResult<F>> {
    validate_ranks(tensor, ranks)?;
    let ndim = tensor.ndim();
    let mut factors: Vec<Array2<F>> = Vec::with_capacity(ndim);

    for n in 0..ndim {
        let unfolded = tensor.unfold(n)?;
        let u = truncated_left_singular_vectors(&unfolded, ranks[n])?;
        factors.push(u);
    }

    // Core = T ×_1 U_1^T ×_2 U_2^T … ×_N U_N^T
    let core = compute_core(tensor, &factors)?;
    Ok(HOSVDResult { core, factors })
}

// ---------------------------------------------------------------------------
// HOOI
// ---------------------------------------------------------------------------

/// Higher-Order Orthogonal Iteration (HOOI) for Tucker decomposition.
///
/// Starts from the HOSVD initialisation and alternates mode-wise SVD updates
/// until convergence or `max_iter` is reached.
///
/// # Arguments
///
/// * `tensor`   – input tensor.
/// * `ranks`    – target multilinear rank.
/// * `max_iter` – maximum number of outer iterations (suggested: 100–500).
///
/// # Returns
///
/// A [`HOSVDResult`] that achieves a local minimum of `‖T - T̃‖_F`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
/// use scirs2_linalg::tensor::hosvd::hooi;
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let tensor = Tensor::new(data, vec![2, 3, 4]).expect("valid");
/// let result = hooi(&tensor, &[2, 2, 3], 50).expect("hooi ok");
/// assert_eq!(result.core.shape, vec![2, 2, 3]);
/// ```
pub fn hooi<F: TensorScalar>(
    tensor: &Tensor<F>,
    ranks: &[usize],
    max_iter: usize,
) -> LinalgResult<HOSVDResult<F>> {
    validate_ranks(tensor, ranks)?;
    let ndim = tensor.ndim();

    // Initialise from HOSVD
    let init = hosvd(tensor, ranks)?;
    let mut factors = init.factors;

    let tol = F::from(1e-10_f64).unwrap_or(F::zero());

    for _iter in 0..max_iter {
        let mut max_change = F::zero();

        for n in 0..ndim {
            // Compute Y = T ×_{m≠n} U_m^T
            let y = project_all_but_mode(tensor, &factors, n)?;
            // Y_(n): mode-n unfolding of Y
            let y_unfold = y.unfold(n)?;
            // New U_n = top-ranks[n] left singular vectors of Y_(n)
            let u_new = truncated_left_singular_vectors(&y_unfold, ranks[n])?;

            // Measure change in subspace (Frobenius distance between old & new)
            let u_old = &factors[n];
            let change = subspace_change(u_old, &u_new);
            if change > max_change {
                max_change = change;
            }
            factors[n] = u_new;
        }

        if max_change < tol {
            break;
        }
    }

    let core = compute_core(tensor, &factors)?;
    Ok(HOSVDResult { core, factors })
}

// ---------------------------------------------------------------------------
// Truncated HOSVD
// ---------------------------------------------------------------------------

/// Auto-rank HOSVD: select ranks based on energy threshold `eps`.
///
/// For each mode `n`, the rank is the smallest `r` such that:
///
/// ```text
/// sum_{k=1}^{r} sigma_k^2 >= (1 - eps^2) * sum_k sigma_k^2
/// ```
///
/// where `sigma_k` are the singular values of the mode-n unfolding.
///
/// # Arguments
///
/// * `tensor` – input tensor.
/// * `eps`    – relative energy to discard (e.g. `0.01` keeps 99% of energy).
///
/// # Returns
///
/// [`HOSVDResult`] with automatically selected ranks.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
/// use scirs2_linalg::tensor::hosvd::truncated_hosvd;
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let tensor = Tensor::new(data, vec![2, 3, 4]).expect("valid");
/// let result = truncated_hosvd(&tensor, 0.01_f64).expect("ok");
/// // ranks are automatically chosen
/// assert!(result.core.ndim() == 3);
/// ```
pub fn truncated_hosvd<F: TensorScalar>(
    tensor: &Tensor<F>,
    eps: F,
) -> LinalgResult<HOSVDResult<F>> {
    let ndim = tensor.ndim();
    let mut ranks = Vec::with_capacity(ndim);

    for n in 0..ndim {
        let unfolded = tensor.unfold(n)?;
        let rank = auto_rank_from_energy(&unfolded, eps)?;
        ranks.push(rank.max(1));
    }

    hosvd(tensor, &ranks)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that `ranks` is compatible with `tensor.shape`.
fn validate_ranks<F: TensorScalar>(tensor: &Tensor<F>, ranks: &[usize]) -> LinalgResult<()> {
    if ranks.len() != tensor.ndim() {
        return Err(LinalgError::DimensionError(format!(
            "ranks length {} != tensor ndim {}",
            ranks.len(),
            tensor.ndim()
        )));
    }
    for (n, (&r, &s)) in ranks.iter().zip(tensor.shape.iter()).enumerate() {
        if r == 0 {
            return Err(LinalgError::ValueError(format!("ranks[{n}] must be > 0")));
        }
        if r > s {
            return Err(LinalgError::ValueError(format!(
                "ranks[{n}]={r} exceeds tensor shape[{n}]={s}"
            )));
        }
    }
    Ok(())
}

/// Compute the truncated left singular vectors of a matrix using SVD.
///
/// Returns `U[:, 0..k]` (the `k` dominant left singular vectors), as an
/// `Array2` of shape `(m, k)`.
fn truncated_left_singular_vectors<F: TensorScalar>(
    matrix: &Array2<F>,
    k: usize,
) -> LinalgResult<Array2<F>> {
    // svd returns (U, s, Vt) with full_matrices option
    let (u_full, _s, _vt) = svd(&matrix.view(), false, None)?;
    // u_full has shape (m, min(m,n)); keep first k columns
    let m = u_full.nrows();
    let avail = u_full.ncols().min(k);
    let mut u = Array2::<F>::zeros((m, k));
    for i in 0..m {
        for j in 0..avail {
            u[[i, j]] = u_full[[i, j]];
        }
    }
    Ok(u)
}

/// Compute the core tensor `G = T ×_1 U_1^T ×_2 U_2^T … ×_N U_N^T`.
fn compute_core<F: TensorScalar>(
    tensor: &Tensor<F>,
    factors: &[Array2<F>],
) -> LinalgResult<Tensor<F>> {
    let mut g = tensor.clone();
    for (n, u) in factors.iter().enumerate() {
        // Multiply by U^T, i.e., transpose rows/cols of U
        let ut = u.t().to_owned();
        g = g.mode_product(&ut, n)?;
    }
    Ok(g)
}

/// Compute `Y = T ×_{m≠n} U_m^T` (project along all modes except `n`).
fn project_all_but_mode<F: TensorScalar>(
    tensor: &Tensor<F>,
    factors: &[Array2<F>],
    skip_mode: usize,
) -> LinalgResult<Tensor<F>> {
    let mut y = tensor.clone();
    for (m, u) in factors.iter().enumerate() {
        if m == skip_mode {
            continue;
        }
        let ut = u.t().to_owned();
        y = y.mode_product(&ut, m)?;
    }
    Ok(y)
}

/// Measure the change in a factor matrix as `‖U_new - U_old‖_F`.
fn subspace_change<F: TensorScalar>(u_old: &Array2<F>, u_new: &Array2<F>) -> F {
    let sq: F = u_old
        .iter()
        .zip(u_new.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .fold(F::zero(), |acc, x| acc + x);
    sq.sqrt()
}

/// Determine the minimum rank `r` that retains energy fraction `(1 - eps²)`.
fn auto_rank_from_energy<F: TensorScalar>(matrix: &Array2<F>, eps: F) -> LinalgResult<usize> {
    let (_u, s, _vt) = svd(&matrix.view(), false, None)?;
    let total_energy: F = s.iter().map(|&sv| sv * sv).fold(F::zero(), |a, b| a + b);
    if total_energy == F::zero() {
        return Ok(1);
    }
    let threshold = (F::one() - eps * eps) * total_energy;
    let mut cumulative = F::zero();
    for (k, &sv) in s.iter().enumerate() {
        cumulative += sv * sv;
        if cumulative >= threshold {
            return Ok(k + 1);
        }
    }
    Ok(s.len().max(1))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_tensor_234() -> Tensor<f64> {
        let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
        Tensor::new(data, vec![2, 3, 4]).expect("valid")
    }

    #[test]
    fn test_hosvd_shape() {
        let t = make_tensor_234();
        let r = hosvd(&t, &[2, 2, 3]).expect("ok");
        assert_eq!(r.core.shape, vec![2, 2, 3]);
        assert_eq!(r.factors[0].shape(), &[2, 2]);
        assert_eq!(r.factors[1].shape(), &[3, 2]);
        assert_eq!(r.factors[2].shape(), &[4, 3]);
    }

    #[test]
    fn test_hosvd_full_rank_lossless() {
        let t = make_tensor_234();
        let r = hosvd(&t, &[2, 3, 4]).expect("full rank");
        let err = r.relative_error(&t).expect("err ok");
        assert!(err < 1e-8, "full rank HOSVD error={err}");
    }

    #[test]
    fn test_hosvd_factor_orthogonality() {
        let t = make_tensor_234();
        let r = hosvd(&t, &[2, 2, 3]).expect("ok");
        for (n, factor) in r.factors.iter().enumerate() {
            let utu = factor.t().dot(factor);
            let rank = factor.ncols();
            for i in 0..rank {
                for j in 0..rank {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let diff = (utu[[i, j]] - expected).abs();
                    assert!(
                        diff < 1e-8,
                        "factor[{n}] not orthonormal at ({i},{j}): diff={diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_hooi_shape() {
        let t = make_tensor_234();
        let r = hooi(&t, &[2, 2, 3], 50).expect("ok");
        assert_eq!(r.core.shape, vec![2, 2, 3]);
    }

    #[test]
    fn test_hooi_better_than_hosvd() {
        let t = make_tensor_234();
        let r_hosvd = hosvd(&t, &[1, 2, 3]).expect("hosvd");
        let r_hooi = hooi(&t, &[1, 2, 3], 100).expect("hooi");
        let err_hosvd = r_hosvd.relative_error(&t).expect("e1");
        let err_hooi = r_hooi.relative_error(&t).expect("e2");
        assert!(
            err_hooi <= err_hosvd + 1e-6,
            "HOOI err {err_hooi} > HOSVD err {err_hosvd}"
        );
    }

    #[test]
    fn test_truncated_hosvd_auto_rank() {
        let t = make_tensor_234();
        let r = truncated_hosvd(&t, 0.0_f64).expect("ok"); // keep all energy
                                                           // At eps=0 we should keep all singular values
        let err = r.relative_error(&t).expect("err");
        assert!(err < 1e-6, "truncated (eps=0) err={err}");
    }

    #[test]
    fn test_compression_ratio() {
        // 4*5*6 = 120 elements; ranks [2,2,2] → core=8 + factors=4*2+5*2+6*2=30 → 38 < 120
        let data: Vec<f64> = (0..120).map(|x| x as f64 + 1.0).collect();
        let t = Tensor::new(data, vec![4, 5, 6]).expect("valid");
        let r = hosvd(&t, &[2, 2, 2]).expect("ok");
        let ratio = r.compression_ratio(&[4, 5, 6]);
        assert!(ratio > 1.0, "should compress: ratio={ratio}");
    }

    #[test]
    fn test_invalid_ranks_length() {
        let t = make_tensor_234();
        assert!(hosvd(&t, &[2, 2]).is_err());
    }

    #[test]
    fn test_invalid_rank_too_large() {
        let t = make_tensor_234();
        assert!(hosvd(&t, &[3, 3, 4]).is_err()); // shape[0]=2, rank=3 invalid
    }
}
