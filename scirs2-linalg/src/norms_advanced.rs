//! Advanced matrix norms and related numerical properties
//!
//! This module provides:
//!
//! - **Nuclear norm**: sum of singular values ‖A‖_* = Σ σ_i
//! - **Schatten p-norm**: (Σ σ_i^p)^{1/p}
//! - **Ky Fan k-norm**: sum of k largest singular values
//! - **Numerical rank**: number of singular values above threshold
//! - **Pseudoinverse (Moore-Penrose)**: via SVD
//! - **Condition number**: with respect to various p-norms
//! - **Stable rank**: ‖A‖_F² / ‖A‖_2²
//! - **Incoherence**: max column coherence with canonical basis

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

// -----------------------------------------------------------------------
// Nuclear norm  ‖A‖_* = Σ σ_i
// -----------------------------------------------------------------------

/// Compute the nuclear norm of a matrix (sum of singular values).
///
/// The nuclear norm equals the Schatten 1-norm and is the convex envelope
/// of the rank function.  It satisfies ‖A‖_* ≥ ‖A‖_F / √rank(A).
///
/// # Arguments
/// * `a` - Input matrix (m×n)
///
/// # Returns
/// Nuclear norm (f64)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::nuclear_norm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
/// let nn = nuclear_norm(&a.view()).expect("nuclear_norm failed");
/// assert!((nn - 7.0).abs() < 1e-10, "nuclear norm should be 7");
/// ```
pub fn nuclear_norm(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    let s = singular_values(a)?;
    Ok(s.iter().sum())
}

// -----------------------------------------------------------------------
// Schatten p-norm  (Σ σ_i^p)^{1/p}
// -----------------------------------------------------------------------

/// Compute the Schatten p-norm: (Σ σ_i^p)^{1/p}.
///
/// Special cases:
/// - p = 1: nuclear norm
/// - p = 2: Frobenius norm
/// - p → ∞: spectral norm (largest singular value)
///
/// # Arguments
/// * `a` - Input matrix
/// * `p` - Order p (must be ≥ 1)
///
/// # Returns
/// Schatten p-norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::schatten_norm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
/// // Schatten 2-norm = Frobenius norm = sqrt(9+16) = 5
/// let s2 = schatten_norm(&a.view(), 2.0).expect("schatten_norm p=2 failed");
/// assert!((s2 - 5.0).abs() < 1e-10);
/// ```
pub fn schatten_norm(a: &ArrayView2<f64>, p: f64) -> LinalgResult<f64> {
    if p < 1.0 {
        return Err(LinalgError::ValueError(format!(
            "Schatten p-norm requires p >= 1, got p = {p}"
        )));
    }
    let s = singular_values(a)?;
    let sum_p: f64 = s.iter().map(|&sv| sv.powf(p)).sum();
    Ok(sum_p.powf(1.0 / p))
}

// -----------------------------------------------------------------------
// Ky Fan k-norm  Σ_{i=1}^{k} σ_i
// -----------------------------------------------------------------------

/// Compute the Ky Fan k-norm: sum of k largest singular values.
///
/// The Ky Fan 1-norm is the spectral (operator) norm; the Ky Fan rank(A)-norm
/// is the nuclear norm.
///
/// # Arguments
/// * `a` - Input matrix
/// * `k` - Number of singular values to sum (must be ≥ 1)
///
/// # Returns
/// Ky Fan k-norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::ky_fan_norm;
///
/// let a = array![[3.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0]];
/// // σ₁=3, σ₂=2 → k=1 gives 3, k=2 gives 5
/// let kf1 = ky_fan_norm(&a.view(), 1).expect("ky_fan_norm k=1 failed");
/// assert!((kf1 - 3.0).abs() < 1e-10);
/// let kf2 = ky_fan_norm(&a.view(), 2).expect("ky_fan_norm k=2 failed");
/// assert!((kf2 - 5.0).abs() < 1e-10);
/// ```
pub fn ky_fan_norm(a: &ArrayView2<f64>, k: usize) -> LinalgResult<f64> {
    if k == 0 {
        return Err(LinalgError::ValueError("k must be >= 1".into()));
    }
    let mut s = singular_values(a)?;
    // Singular values from SVD are already in descending order, but sort to be sure
    s.iter_mut().for_each(|sv| {
        if *sv < 0.0 {
            *sv = 0.0;
        }
    });
    // Sort descending
    let mut sorted: Vec<f64> = s.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("NaN in singular values"));

    let k_eff = k.min(sorted.len());
    Ok(sorted[..k_eff].iter().sum())
}

// -----------------------------------------------------------------------
// Numerical rank
// -----------------------------------------------------------------------

/// Estimate the numerical rank of a matrix.
///
/// Counts singular values above a threshold.  Default threshold is
/// `max(m, n) * eps * σ_max` where eps = machine epsilon and σ_max is the
/// largest singular value.
///
/// # Arguments
/// * `a`   - Input matrix
/// * `tol` - Explicit threshold (None = use default)
///
/// # Returns
/// Estimated numerical rank
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::numerical_rank;
///
/// // Rank-2 matrix (3rd row = sum of first two)
/// let a = array![
///     [1.0_f64, 0.0],
///     [0.0,     1.0],
///     [1.0,     1.0],
/// ];
/// let r = numerical_rank(&a.view(), None).expect("numerical_rank failed");
/// assert_eq!(r, 2);
///
/// // Rank-1 matrix
/// let b = array![
///     [1.0_f64, 2.0, 3.0],
///     [2.0,     4.0, 6.0],
/// ];
/// let r1 = numerical_rank(&b.view(), None).expect("numerical_rank failed");
/// assert_eq!(r1, 1);
/// ```
pub fn numerical_rank(a: &ArrayView2<f64>, tol: Option<f64>) -> LinalgResult<usize> {
    let m = a.nrows();
    let n = a.ncols();
    let s = singular_values(a)?;
    let sigma_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = tol.unwrap_or_else(|| {
        let eps = f64::EPSILON;
        eps * (m.max(n) as f64) * sigma_max
    });
    Ok(s.iter().filter(|&&sv| sv > threshold).count())
}

// -----------------------------------------------------------------------
// Pseudoinverse (Moore-Penrose)
// -----------------------------------------------------------------------

/// Compute the Moore-Penrose pseudoinverse of a matrix via SVD.
///
/// For a matrix A = U Σ V^T, the pseudoinverse is A† = V Σ† U^T where Σ†
/// inverts all non-zero singular values.
///
/// # Arguments
/// * `a`   - Input matrix (m×n)
/// * `tol` - Threshold below which singular values are treated as zero
///   (None = machine-epsilon based default)
///
/// # Returns
/// Pseudoinverse matrix (n×m)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::pinv;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0]];
/// let ai = pinv(&a.view(), None).expect("pinv failed");
/// // A† A should be I₂
/// let prod = ai.dot(&a);
/// for i in 0..2 {
///     for j in 0..2 {
///         let expected = if i == j { 1.0 } else { 0.0 };
///         assert!((prod[[i,j]] - expected).abs() < 1e-10);
///     }
/// }
/// ```
pub fn pinv(a: &ArrayView2<f64>, tol: Option<f64>) -> LinalgResult<Array2<f64>> {
    let m = a.nrows();
    let n = a.ncols();

    // Use OxiBLAS SVD (reliable for degenerate cases including repeated singular values).
    // svd_f64_lapack returns (u, s, vt) where u may be m×m or m×k, vt may be k×n or n×n.
    // We use `svd(a, false, None)` (thin SVD) which gives u: m×k, s: k, vt: k×n,
    // where k = min(m, n).  This is sufficient for the pseudoinverse.
    let (u, s, vt) = crate::decomposition::svd_f64_lapack(a, false)?;

    let sigma_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = tol.unwrap_or_else(|| {
        let eps = f64::EPSILON;
        eps * (m.max(n) as f64) * sigma_max
    });

    let k = s.len(); // = min(m, n)

    // A† = V * Σ† * U^T  where V = Vt^T
    //   = sum_i (1/s_i) * v_i * u_i^T   for s_i > threshold
    // v_i = row i of Vt (length n)
    // u_i = column i of U (length m)
    // Result shape: A† is (n × m).

    let mut result = Array2::<f64>::zeros((n, m));
    for i in 0..k {
        if s[i] > threshold {
            let inv_si = 1.0 / s[i];
            // v_i = row i of Vt (length n = a.ncols())
            let v_i = vt.row(i);
            // u_i = column i of U (length m = a.nrows())
            let u_i = u.column(i);
            // outer product: result[r, c] += inv_si * v_i[r] * u_i[c]
            for r in 0..n {
                for c in 0..m {
                    result[[r, c]] += inv_si * v_i[r] * u_i[c];
                }
            }
        }
    }
    Ok(result)
}

// -----------------------------------------------------------------------
// Condition number w.r.t. p-norm
// -----------------------------------------------------------------------

/// Compute the condition number of a matrix with respect to a given p-norm.
///
/// For p = None or p = Some(2.0) uses the spectral condition number σ_max/σ_min.
/// For p = Some(1.0) uses ‖A‖₁ · ‖A†‖₁.
/// For p = Some(f64::INFINITY) uses ‖A‖_∞ · ‖A†‖_∞.
/// For p = Some(-1.0) returns the *minimum* singular value (useful for
/// near-singularity checking, not a true condition number).
///
/// # Arguments
/// * `a` - Input matrix (must be square for p≠2, or any shape for p=2)
/// * `p` - Norm order (None defaults to 2)
///
/// # Returns
/// Condition number
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::cond;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 1.0]];
/// let c = cond(&a.view(), None).expect("cond failed");
/// assert!((c - 2.0).abs() < 1e-10, "cond should be 2, got {c}");
/// ```
pub fn cond(a: &ArrayView2<f64>, p: Option<f64>) -> LinalgResult<f64> {
    let p_val = p.unwrap_or(2.0);

    match p_val {
        v if (v - 2.0).abs() < f64::EPSILON => {
            // Spectral condition number
            let s = singular_values(a)?;
            if s.is_empty() {
                return Err(LinalgError::ValueError("Empty matrix".into()));
            }
            let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
            let s_min = s.iter().cloned().fold(f64::INFINITY, f64::min);
            if s_min < f64::EPSILON * s_max {
                Ok(f64::INFINITY)
            } else {
                Ok(s_max / s_min)
            }
        }
        v if (v - 1.0).abs() < f64::EPSILON => {
            // 1-norm condition: ‖A‖₁ · ‖A†‖₁
            let norm_a = matrix_norm_1(a);
            let a_pinv = pinv(a, None)?;
            let norm_pinv = matrix_norm_1(&a_pinv.view());
            Ok(norm_a * norm_pinv)
        }
        v if v.is_infinite() && v > 0.0 => {
            // inf-norm condition: ‖A‖_∞ · ‖A†‖_∞
            let norm_a = matrix_norm_inf(a);
            let a_pinv = pinv(a, None)?;
            let norm_pinv = matrix_norm_inf(&a_pinv.view());
            Ok(norm_a * norm_pinv)
        }
        _ => Err(LinalgError::ValueError(format!(
            "Unsupported p for cond: {p_val}"
        ))),
    }
}

// -----------------------------------------------------------------------
// Stable rank  ‖A‖_F² / ‖A‖_2²
// -----------------------------------------------------------------------

/// Compute the stable rank of a matrix: ‖A‖_F² / ‖A‖_2².
///
/// The stable rank is a continuous relaxation of the rank.  It satisfies
/// 1 ≤ stable_rank(A) ≤ rank(A) and equals rank(A) for matrices with
/// equal singular values.
///
/// # Arguments
/// * `a` - Input matrix
///
/// # Returns
/// Stable rank (dimensionless, ≥ 1)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::stable_rank;
///
/// // Identity: all SVs = 1  => stable_rank = n / 1 = n
/// let a = scirs2_core::ndarray::Array2::<f64>::eye(3);
/// let sr = stable_rank(&a.view()).expect("stable_rank failed");
/// assert!((sr - 3.0).abs() < 1e-10);
/// ```
pub fn stable_rank(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    let s = singular_values(a)?;
    if s.is_empty() {
        return Err(LinalgError::ValueError(
            "Empty matrix for stable_rank".into(),
        ));
    }
    let frob_sq: f64 = s.iter().map(|&sv| sv * sv).sum();
    let spec_sq = s.iter().cloned().fold(0.0_f64, f64::max).powi(2);
    if spec_sq < f64::EPSILON {
        return Ok(0.0);
    }
    Ok(frob_sq / spec_sq)
}

// -----------------------------------------------------------------------
// Incoherence
// -----------------------------------------------------------------------

/// Compute the incoherence of a matrix.
///
/// Incoherence measures how "spread out" the row space is with respect to
/// the canonical basis.  Formally, given the compact SVD A = U Σ V^T with
/// U (m×r), the incoherence is:
///   μ(A) = (m / r) * max_i ‖U^T e_i‖²
/// where e_i are standard basis vectors.  A small incoherence (close to 1)
/// means the row space is spread evenly across all directions.
///
/// For a matrix that has not yet been factored, this function uses the left
/// singular vectors from the full compact SVD.
///
/// # Arguments
/// * `a` - Input matrix (m×n)
///
/// # Returns
/// Incoherence μ ∈ [1, m/r]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::norms_advanced::incoherence;
///
/// // Identity: maximally incoherent (μ = 1)
/// let a = scirs2_core::ndarray::Array2::<f64>::eye(3);
/// let mu = incoherence(&a.view()).expect("incoherence failed");
/// assert!((mu - 1.0).abs() < 1e-10, "identity incoherence should be 1, got {mu}");
/// ```
pub fn incoherence(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    let m = a.nrows();

    let (u, s, _) = crate::decomposition::svd(a, true, None)?;

    // Determine rank
    let sigma_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = f64::EPSILON * (m.max(a.ncols()) as f64) * sigma_max;
    let r = s.iter().filter(|&&sv| sv > threshold).count();

    if r == 0 {
        return Ok(1.0);
    }

    // U is m×m (full U from SVD); take first r columns
    // μ = (m/r) * max_i ‖U[i, 0..r]‖²
    let mu_unnorm: f64 = (0..m)
        .map(|i| (0..r).map(|j| u[[i, j]] * u[[i, j]]).sum::<f64>())
        .fold(0.0_f64, f64::max);

    Ok((m as f64 / r as f64) * mu_unnorm)
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Extract singular values (sorted descending) from SVD.
fn singular_values(a: &ArrayView2<f64>) -> LinalgResult<Array1<f64>> {
    let (_, s, _) = crate::decomposition::svd(a, false, None)?;
    Ok(s)
}

/// 1-norm of a matrix: max column sum of absolute values.
fn matrix_norm_1(a: &ArrayView2<f64>) -> f64 {
    (0..a.ncols())
        .map(|j| a.column(j).iter().map(|&v| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// Infinity-norm of a matrix: max row sum of absolute values.
fn matrix_norm_inf(a: &ArrayView2<f64>) -> f64 {
    (0..a.nrows())
        .map(|i| a.row(i).iter().map(|&v| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // ---- Nuclear norm ----

    #[test]
    fn test_nuclear_norm_diagonal() {
        // Diagonal matrix: nuclear norm = sum of |diag entries| = 3 + 4 = 7
        let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        assert!((nn - 7.0).abs() < 1e-10, "nuclear norm diagonal: {nn}");
    }

    #[test]
    fn test_nuclear_norm_identity() {
        // Identity n×n: nuclear norm = n
        let a = Array2::<f64>::eye(4);
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm eye");
        assert!((nn - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nuclear_norm_lower_bound() {
        // ‖A‖_* ≥ ‖A‖_F / √rank(A)
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        let frob_sq: f64 = a.iter().map(|&v| v * v).sum();
        let frob = frob_sq.sqrt();
        let rank = numerical_rank(&a.view(), None).expect("rank") as f64;
        let lower_bound = frob / rank.sqrt();
        assert!(
            nn >= lower_bound - 1e-10,
            "nuclear_norm {nn} < lower bound {lower_bound}"
        );
    }

    // ---- Schatten norm ----

    #[test]
    fn test_schatten_p1_equals_nuclear() {
        let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
        let s1 = schatten_norm(&a.view(), 1.0).expect("schatten p=1");
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        assert!((s1 - nn).abs() < 1e-10);
    }

    #[test]
    fn test_schatten_p2_equals_frobenius() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let s2 = schatten_norm(&a.view(), 2.0).expect("schatten p=2");
        let frob_sq: f64 = a.iter().map(|&v| v * v).sum();
        let frob = frob_sq.sqrt();
        assert!((s2 - frob).abs() < 1e-10, "schatten 2 vs frobenius");
    }

    #[test]
    fn test_schatten_invalid_p() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(schatten_norm(&a.view(), 0.5).is_err());
    }

    // ---- Ky Fan norm ----

    #[test]
    fn test_ky_fan_k1_equals_spectral() {
        let a = array![[3.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let kf1 = ky_fan_norm(&a.view(), 1).expect("ky_fan k=1");
        assert!((kf1 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ky_fan_k2_equals_nuclear_for_rank2() {
        let a = array![[3.0_f64, 0.0], [0.0, 2.0]];
        let kf2 = ky_fan_norm(&a.view(), 2).expect("ky_fan k=2");
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        assert!((kf2 - nn).abs() < 1e-10);
    }

    #[test]
    fn test_ky_fan_monotone() {
        let a = array![[5.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let kf1 = ky_fan_norm(&a.view(), 1).expect("kf1");
        let kf2 = ky_fan_norm(&a.view(), 2).expect("kf2");
        let kf3 = ky_fan_norm(&a.view(), 3).expect("kf3");
        assert!(kf1 <= kf2 && kf2 <= kf3);
    }

    // ---- Numerical rank ----

    #[test]
    fn test_numerical_rank_full() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = numerical_rank(&a.view(), None).expect("rank full");
        assert_eq!(r, 2);
    }

    #[test]
    fn test_numerical_rank_deficient() {
        let a = array![[1.0_f64, 2.0, 3.0], [2.0, 4.0, 6.0],];
        let r = numerical_rank(&a.view(), None).expect("rank deficient");
        assert_eq!(r, 1, "rank should be 1, got {r}");
    }

    #[test]
    fn test_numerical_rank_zero_matrix() {
        let a = Array2::<f64>::zeros((3, 3));
        let r = numerical_rank(&a.view(), None).expect("rank zero");
        assert_eq!(r, 0);
    }

    // ---- Pseudoinverse ----

    #[test]
    fn test_pinv_full_rank_square() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let ai = pinv(&a.view(), None).expect("pinv square");
        let prod = a.dot(&ai);
        let eye2 = Array2::<f64>::eye(2);
        let err: f64 = (&prod - &eye2).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(err < 1e-10, "pinv * A not I: {err}");
    }

    #[test]
    fn test_pinv_tall_matrix() {
        // A is 3×2 full rank => A† A = I₂
        let a = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let ai = pinv(&a.view(), None).expect("pinv tall");
        let prod = ai.dot(&a);
        let eye2 = Array2::<f64>::eye(2);
        let err: f64 = (&prod - &eye2).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(err < 1e-10, "A† A not I₂: {err}");
    }

    #[test]
    fn test_pinv_rank_deficient() {
        // Rank-1: [1,2;2,4] => A† A != I but A A† A = A
        let a = array![[1.0_f64, 2.0], [2.0, 4.0]];
        let ai = pinv(&a.view(), None).expect("pinv rank-def");
        // Check Moore-Penrose condition A A† A = A
        let recon = a.dot(&ai).dot(&a);
        let err: f64 = (&recon - &a).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(err < 1e-10, "pinv Moore-Penrose cond 1: {err}");
    }

    // ---- Condition number ----

    #[test]
    fn test_cond_identity() {
        let a = Array2::<f64>::eye(3);
        let c = cond(&a.view(), None).expect("cond identity");
        assert!((c - 1.0).abs() < 1e-10, "cond(I) should be 1");
    }

    #[test]
    fn test_cond_diagonal() {
        let a = array![[2.0_f64, 0.0], [0.0, 1.0]];
        let c = cond(&a.view(), Some(2.0)).expect("cond diagonal");
        assert!((c - 2.0).abs() < 1e-10, "cond diag: {c}");
    }

    #[test]
    fn test_cond_singular() {
        let a = array![[1.0_f64, 0.0], [0.0, 0.0]];
        let c = cond(&a.view(), None).expect("cond singular");
        assert!(c.is_infinite() || c > 1e10, "cond singular should be large");
    }

    // ---- Stable rank ----

    #[test]
    fn test_stable_rank_identity() {
        let a = Array2::<f64>::eye(3);
        let sr = stable_rank(&a.view()).expect("stable_rank eye");
        assert!((sr - 3.0).abs() < 1e-10, "stable_rank(I₃) = 3, got {sr}");
    }

    #[test]
    fn test_stable_rank_rank1() {
        // Rank-1 matrix: stable_rank = 1
        let a = array![[1.0_f64, 2.0], [2.0, 4.0]];
        let sr = stable_rank(&a.view()).expect("stable_rank rank1");
        assert!((sr - 1.0).abs() < 1e-10, "stable_rank rank-1 = 1, got {sr}");
    }

    #[test]
    fn test_stable_rank_bounds() {
        // Stable rank should be between 1 and rank
        let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
        let sr = stable_rank(&a.view()).expect("stable_rank bounds");
        assert!((1.0 - 1e-10..=2.0 + 1e-10).contains(&sr));
    }

    // ---- Incoherence ----

    #[test]
    fn test_incoherence_identity() {
        // Identity: all rows are canonical basis vectors => maximally incoherent
        // μ = (n/n) * max_i ‖e_i‖² = 1
        let a = Array2::<f64>::eye(3);
        let mu = incoherence(&a.view()).expect("incoherence identity");
        assert!((mu - 1.0).abs() < 1e-10, "incoherence(I) = 1, got {mu}");
    }

    #[test]
    fn test_incoherence_range() {
        // μ ≥ 1 always
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mu = incoherence(&a.view()).expect("incoherence range");
        assert!(mu >= 1.0 - 1e-10, "incoherence must be >= 1, got {mu}");
    }
}
