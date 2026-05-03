//! Extended matrix norm computations
//!
//! This module provides matrix norms beyond the basics already available in
//! `norm.rs` and `norms_advanced.rs`:
//!
//! - **Nuclear norm** (`nuclear_norm`): ‖A‖_* = Σ σ_i
//! - **Operator / spectral norm** (`operator_norm`): ‖A‖_2 = σ_max
//! - **(p,q)-norm** (`matrix_pq_norm`): ‖A‖_{p,q} = (Σ_j ‖a_j‖_p^q)^{1/q}
//! - **Schatten p-norm** (`schatten_norm`): (Σ σ_i^p)^{1/p}
//! - **Ky Fan k-norm** (`ky_fan_norm`): Σ_{i=1}^{k} σ_i (k largest SVs)
//!
//! All functions accept `f32` or `f64` via the `Float` bound.

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute all singular values of `a` in **descending** order.
fn sorted_singular_values<F>(a: &ArrayView2<F>) -> LinalgResult<Vec<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    if a.nrows() == 0 || a.ncols() == 0 {
        return Err(LinalgError::InvalidInputError(
            "matrix_norms: matrix must be non-empty".to_string(),
        ));
    }
    let (_u, s, _vt) = svd(a, false, None)?;
    // SVD already returns singular values in descending order, but we sort
    // explicitly to be robust against implementation differences.
    let mut values: Vec<F> = s.iter().cloned().collect();
    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(values)
}

// ---------------------------------------------------------------------------
// Nuclear norm
// ---------------------------------------------------------------------------

/// Compute the nuclear norm ‖A‖_* = Σ σ_i (sum of all singular values).
///
/// The nuclear norm is the Schatten 1-norm and serves as the convex relaxation
/// of the matrix rank.  It satisfies ‖A‖_* ≥ ‖A‖_F / √rank(A).
///
/// # Arguments
/// * `a` - Input matrix (m×n), any shape
///
/// # Returns
/// Nuclear norm ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_norms::nuclear_norm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
/// let nn = nuclear_norm(&a.view()).expect("nuclear_norm failed");
/// assert!((nn - 7.0).abs() < 1e-10, "expected 7, got {nn}");
/// ```
pub fn nuclear_norm<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let svs = sorted_singular_values(a)?;
    Ok(svs.iter().cloned().fold(F::zero(), |acc, s| acc + s))
}

// ---------------------------------------------------------------------------
// Operator / spectral norm
// ---------------------------------------------------------------------------

/// Compute the operator (spectral) norm ‖A‖_2 = σ_max(A).
///
/// This equals the largest singular value and is the induced ℓ²→ℓ² norm.
///
/// # Arguments
/// * `a` - Input matrix (m×n)
///
/// # Returns
/// Largest singular value ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_norms::operator_norm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
/// let on = operator_norm(&a.view()).expect("operator_norm failed");
/// assert!((on - 4.0).abs() < 1e-10, "expected 4, got {on}");
/// ```
pub fn operator_norm<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let svs = sorted_singular_values(a)?;
    Ok(svs.first().cloned().unwrap_or(F::zero()))
}

// ---------------------------------------------------------------------------
// (p,q)-norms
// ---------------------------------------------------------------------------

/// Compute the (p,q)-mixed matrix norm.
///
/// ‖A‖_{p,q} = ( Σ_j ‖a_j‖_p^q )^{1/q}
///
/// where a_j is the j-th **column** of A and ‖·‖_p is the vector p-norm.
///
/// Special cases:
/// - (1,1): sum of all |a_{ij}|  (entrywise ℓ¹ norm)
/// - (2,2): Frobenius norm
/// - (2,1): sum of ℓ²-norms of columns (group Lasso penalty)
/// - (∞,∞): max |a_{ij}| (Chebyshev norm); pass `p = f64::INFINITY`
///
/// # Arguments
/// * `a` - Input matrix (m×n)
/// * `p` - Row norm order (must be ≥ 1, or use `f64::INFINITY` for ∞-norm)
/// * `q` - Column aggregation order (must be ≥ 1, or ∞)
///
/// # Returns
/// (p,q)-norm value ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_norms::matrix_pq_norm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// // Frobenius norm via (2,2)-norm
/// let f = matrix_pq_norm(&a.view(), 2.0, 2.0).expect("pq_norm failed");
/// let fro = (1.0_f64 + 4.0 + 9.0 + 16.0_f64).sqrt();
/// assert!((f - fro).abs() < 1e-10);
/// ```
pub fn matrix_pq_norm<F>(a: &ArrayView2<F>, p: F, q: F) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let one = F::one();
    if p < one {
        let p_f64 = p.to_f64().unwrap_or(0.0);
        return Err(LinalgError::ValueError(format!(
            "matrix_pq_norm: p must be >= 1, got p = {p_f64}"
        )));
    }
    if q < one {
        let q_f64 = q.to_f64().unwrap_or(0.0);
        return Err(LinalgError::ValueError(format!(
            "matrix_pq_norm: q must be >= 1, got q = {q_f64}"
        )));
    }
    if a.nrows() == 0 || a.ncols() == 0 {
        return Err(LinalgError::InvalidInputError(
            "matrix_pq_norm: matrix must be non-empty".to_string(),
        ));
    }

    let m = a.nrows();
    let ncols = a.ncols();

    let inf = F::infinity();

    // Compute the p-norm of each column
    let col_norms: Vec<F> = (0..ncols)
        .map(|j| {
            let col = a.column(j);
            if p == inf {
                col.iter()
                    .cloned()
                    .map(|x| x.abs())
                    .fold(F::zero(), |acc, v| if v > acc { v } else { acc })
            } else {
                let sum_p: F = col.iter().cloned().map(|x| x.abs().powf(p)).sum();
                sum_p.powf(F::one() / p)
            }
        })
        .collect();

    // Now aggregate column norms with q-norm
    if q == inf {
        Ok(col_norms
            .iter()
            .cloned()
            .fold(F::zero(), |acc, v| if v > acc { v } else { acc }))
    } else {
        let sum_q: F = col_norms.iter().cloned().map(|v| v.powf(q)).sum();
        Ok(sum_q.powf(F::one() / q))
    }
}

// ---------------------------------------------------------------------------
// Schatten p-norm
// ---------------------------------------------------------------------------

/// Compute the Schatten p-norm: ( Σ σ_i^p )^{1/p}.
///
/// Special cases:
/// - p = 1: nuclear norm
/// - p = 2: Frobenius norm
/// - p → ∞: operator (spectral) norm (largest σ); pass `p = f64::INFINITY`
///
/// # Arguments
/// * `a` - Input matrix (m×n)
/// * `p` - Order p (must be ≥ 1, or ∞)
///
/// # Returns
/// Schatten p-norm ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_norms::schatten_norm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
/// // Schatten 1-norm = nuclear norm = 7
/// let s1 = schatten_norm(&a.view(), 1.0).expect("schatten p=1 failed");
/// assert!((s1 - 7.0).abs() < 1e-10);
///
/// // Schatten 2-norm = Frobenius norm = 5
/// let s2 = schatten_norm(&a.view(), 2.0).expect("schatten p=2 failed");
/// assert!((s2 - 5.0).abs() < 1e-10);
/// ```
pub fn schatten_norm<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let one = F::one();
    if p < one {
        let p_f64 = p.to_f64().unwrap_or(0.0);
        return Err(LinalgError::ValueError(format!(
            "schatten_norm: p must be >= 1, got p = {p_f64}"
        )));
    }

    let svs = sorted_singular_values(a)?;
    let inf = F::infinity();

    if p == inf {
        return Ok(svs.first().cloned().unwrap_or(F::zero()));
    }

    let sum_p: F = svs.iter().cloned().map(|s| s.powf(p)).sum();
    Ok(sum_p.powf(F::one() / p))
}

// ---------------------------------------------------------------------------
// Ky Fan k-norm
// ---------------------------------------------------------------------------

/// Compute the Ky Fan k-norm: Σ_{i=1}^{k} σ_i (sum of k largest singular values).
///
/// The Ky Fan 1-norm = operator norm = σ_max.
/// The Ky Fan rank(A)-norm = nuclear norm.
///
/// # Arguments
/// * `a` - Input matrix (m×n)
/// * `k` - Number of singular values to sum (1 ≤ k ≤ min(m,n))
///
/// # Returns
/// Ky Fan k-norm ≥ 0.
///
/// # Errors
/// Returns `LinalgError` if k = 0 or k > min(m, n).
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_norms::ky_fan_norm;
///
/// let a = array![[5.0_f64, 0.0], [0.0, 3.0]];
/// let kf1 = ky_fan_norm(&a.view(), 1).expect("kf1 failed");
/// assert!((kf1 - 5.0).abs() < 1e-10);
///
/// let kf2 = ky_fan_norm(&a.view(), 2).expect("kf2 failed");
/// assert!((kf2 - 8.0).abs() < 1e-10);
/// ```
pub fn ky_fan_norm<F>(a: &ArrayView2<F>, k: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    if k == 0 {
        return Err(LinalgError::ValueError(
            "ky_fan_norm: k must be >= 1".to_string(),
        ));
    }

    let svs = sorted_singular_values(a)?;

    if k > svs.len() {
        return Err(LinalgError::ValueError(format!(
            "ky_fan_norm: k={k} exceeds number of singular values {}",
            svs.len()
        )));
    }

    Ok(svs
        .iter()
        .take(k)
        .cloned()
        .fold(F::zero(), |acc, s| acc + s))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // ---- nuclear_norm ----

    #[test]
    fn test_nuclear_norm_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        assert!((nn - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_nuclear_norm_identity() {
        let a = Array2::<f64>::eye(4);
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm identity");
        assert!((nn - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nuclear_norm_rank1() {
        // Rank-1 matrix: only one non-zero singular value
        let a = array![[1.0_f64, 2.0], [2.0, 4.0]];
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm rank1");
        // σ₁ = √(1+4+4+16) = √25 = 5
        assert!((nn - 5.0).abs() < 1e-10, "expected 5, got {nn}");
    }

    // ---- operator_norm ----

    #[test]
    fn test_operator_norm_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
        let on = operator_norm(&a.view()).expect("operator_norm");
        assert!((on - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_operator_norm_identity() {
        let a = Array2::<f64>::eye(3);
        let on = operator_norm(&a.view()).expect("operator_norm identity");
        assert!((on - 1.0).abs() < 1e-10);
    }

    // ---- matrix_pq_norm ----

    #[test]
    fn test_pq_norm_22_is_frobenius() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let pq = matrix_pq_norm(&a.view(), 2.0, 2.0).expect("pq 2,2");
        let fro = (1.0_f64 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((pq - fro).abs() < 1e-10, "expected {fro}, got {pq}");
    }

    #[test]
    fn test_pq_norm_11_is_l1_entry() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let pq = matrix_pq_norm(&a.view(), 1.0, 1.0).expect("pq 1,1");
        // ‖a_1‖_1 + ‖a_2‖_1 = 4 + 6 = 10
        assert!((pq - 10.0).abs() < 1e-10, "expected 10, got {pq}");
    }

    #[test]
    fn test_pq_norm_invalid_p() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(matrix_pq_norm(&a.view(), 0.5, 2.0).is_err());
    }

    #[test]
    fn test_pq_norm_invalid_q() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(matrix_pq_norm(&a.view(), 2.0, 0.5).is_err());
    }

    // ---- schatten_norm ----

    #[test]
    fn test_schatten_1_equals_nuclear() {
        let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
        let s1 = schatten_norm(&a.view(), 1.0).expect("schatten 1");
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        assert!((s1 - nn).abs() < 1e-10);
    }

    #[test]
    fn test_schatten_2_equals_frobenius() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let s2 = schatten_norm(&a.view(), 2.0).expect("schatten 2");
        let fro: f64 = a.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!((s2 - fro).abs() < 1e-10, "schatten 2 vs frobenius");
    }

    #[test]
    fn test_schatten_invalid_p() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(schatten_norm(&a.view(), 0.5).is_err());
    }

    #[test]
    fn test_schatten_monotone_in_p() {
        // For a non-trivial matrix, Schatten p-norm decreases as p grows
        let a = array![[5.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let s1 = schatten_norm(&a.view(), 1.0).expect("s1");
        let s2 = schatten_norm(&a.view(), 2.0).expect("s2");
        let s4 = schatten_norm(&a.view(), 4.0).expect("s4");
        assert!(
            s1 >= s2 - 1e-10 && s2 >= s4 - 1e-10,
            "monotone: {s1} >= {s2} >= {s4}"
        );
    }

    // ---- ky_fan_norm ----

    #[test]
    fn test_ky_fan_1_is_spectral() {
        let a = array![[5.0_f64, 0.0], [0.0, 3.0]];
        let kf1 = ky_fan_norm(&a.view(), 1).expect("kf1");
        assert!((kf1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ky_fan_max_is_nuclear() {
        let a = array![[3.0_f64, 0.0], [0.0, 4.0]];
        let kf2 = ky_fan_norm(&a.view(), 2).expect("kf2");
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        assert!((kf2 - nn).abs() < 1e-10);
    }

    #[test]
    fn test_ky_fan_monotone() {
        let a = array![[5.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let kf1 = ky_fan_norm(&a.view(), 1).expect("kf1");
        let kf2 = ky_fan_norm(&a.view(), 2).expect("kf2");
        let kf3 = ky_fan_norm(&a.view(), 3).expect("kf3");
        assert!(kf1 <= kf2 + 1e-10 && kf2 <= kf3 + 1e-10);
    }

    #[test]
    fn test_ky_fan_k0_error() {
        let a = Array2::<f64>::eye(2);
        assert!(ky_fan_norm(&a.view(), 0).is_err());
    }

    #[test]
    fn test_ky_fan_k_too_large_error() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(ky_fan_norm(&a.view(), 3).is_err());
    }

    // ---- cross-module consistency ----

    #[test]
    fn test_nuclear_ge_operator() {
        // ‖A‖_* ≥ ‖A‖_2 always
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let nn = nuclear_norm(&a.view()).expect("nuclear_norm");
        let on = operator_norm(&a.view()).expect("operator_norm");
        assert!(nn >= on - 1e-10, "nuclear {nn} >= operator {on}");
    }
}
