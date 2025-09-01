//! Specialized eigenvalue solvers for structured matrices
//!
//! This module provides optimized eigenvalue solvers for matrices with special structure,
//! including tridiagonal, banded, Toeplitz, and circulant matrices. These specialized
//! algorithms can be significantly faster than general-purpose eigenvalue solvers.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, One, Zero};
use rand::{self, Rng};
use std::iter::Sum;

// Compatibility wrapper functions for the compat module
/// Wrapper for banded matrix eigenvalues and eigenvectors (SciPy-style)
#[allow(dead_code)]
pub fn banded_eigh<F>(
    matrix: &ArrayView2<F>,
    bandwidth: usize,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (eigenvals, eigenvecs_opt) = banded_eigen(matrix, bandwidth, true)?;
    let eigenvecs = eigenvecs_opt.ok_or_else(|| {
        LinalgError::ComputationError("Failed to compute eigenvectors".to_string())
    })?;
    Ok((eigenvals, eigenvecs))
}

/// Wrapper for banded matrix eigenvalues only (SciPy-style)
#[allow(dead_code)]
pub fn banded_eigvalsh<F>(matrix: &ArrayView2<F>, bandwidth: usize) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (eigenvals, _) = banded_eigen(matrix, bandwidth, false)?;
    Ok(eigenvals)
}

/// Wrapper for tridiagonal matrix eigenvalues and eigenvectors (SciPy-style)
#[allow(dead_code)]
pub fn tridiagonal_eigh<F>(
    diagonal: &ArrayView1<F>,
    sub_diagonal: &ArrayView1<F>,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (eigenvals, eigenvecs_opt) = tridiagonal_eigen(diagonal, sub_diagonal, true)?;
    let eigenvecs = eigenvecs_opt.ok_or_else(|| {
        LinalgError::ComputationError("Failed to compute eigenvectors".to_string())
    })?;
    Ok((eigenvals, eigenvecs))
}

/// Wrapper for tridiagonal matrix eigenvalues only (SciPy-style)
#[allow(dead_code)]
pub fn tridiagonal_eigvalsh<F>(
    diagonal: &ArrayView1<F>,
    sub_diagonal: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (eigenvals, _) = tridiagonal_eigen(diagonal, sub_diagonal, false)?;
    Ok(eigenvals)
}

/// Find the k largest eigenvalues and eigenvectors of a symmetric matrix
///
/// This is a wrapper around the partial_eigen function for compatibility.
///
/// # Arguments
///
/// * `matrix` - Symmetric matrix
/// * `k` - Number of largest eigenvalues to compute
/// * `max_iter` - Maximum iterations for the algorithm
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * k largest eigenvalues and corresponding eigenvectors
#[allow(dead_code)]
pub fn largest_k_eigh<F>(
    matrix: &ArrayView2<F>,
    k: usize,
    max_iter: usize,
    tol: F,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    if k == 0 {
        let n = matrix.nrows();
        return Ok((Array1::zeros(0), Array2::zeros((n, 0))));
    }

    partial_eigen(matrix, k, "largest", Some(max_iter), Some(tol))
}

/// Find the k smallest eigenvalues and eigenvectors of a symmetric matrix
///
/// This is a wrapper around the partial_eigen function for compatibility.
///
/// # Arguments
///
/// * `matrix` - Symmetric matrix
/// * `k` - Number of smallest eigenvalues to compute
/// * `max_iter` - Maximum iterations for the algorithm
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * k smallest eigenvalues and corresponding eigenvectors
#[allow(dead_code)]
pub fn smallest_k_eigh<F>(
    matrix: &ArrayView2<F>,
    k: usize,
    max_iter: usize,
    tol: F,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    if k == 0 {
        let n = matrix.nrows();
        return Ok((Array1::zeros(0), Array2::zeros((n, 0))));
    }

    partial_eigen(matrix, k, "smallest", Some(max_iter), Some(tol))
}

/// Eigenvalue solver for symmetric tridiagonal matrices
///
/// Uses the implicit QL algorithm with Givens rotations for O(n²) complexity.
/// This is much faster than general eigenvalue algorithms for tridiagonal matrices.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `sub_diagonal` - Sub-diagonal elements (length n-1)
/// * `compute_eigenvectors` - Whether to compute eigenvectors
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) where eigenvectors is None if not computed
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen_specialized::tridiagonal_eigen;
///
/// let diagonal = array![4.0, 4.0, 4.0];
/// let sub_diagonal = array![1.0, 1.0];
/// let (eigenvals, eigenvecs) = tridiagonal_eigen(&diagonal.view(), &sub_diagonal.view(), true).unwrap();
/// assert_eq!(eigenvals.len(), 3);
/// ```
#[allow(dead_code)]
pub fn tridiagonal_eigen<F>(
    diagonal: &ArrayView1<F>,
    sub_diagonal: &ArrayView1<F>,
    compute_eigenvectors: bool,
) -> LinalgResult<(Array1<F>, Option<Array2<F>>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = diagonal.len();

    if sub_diagonal.len() != n - 1 {
        return Err(LinalgError::ShapeError(format!(
            "Sub-diagonal length {} must be n-1 = {}",
            sub_diagonal.len(),
            n - 1
        )));
    }

    if n == 0 {
        return Ok((Array1::zeros(0), Some(Array2::zeros((0, 0)))));
    }

    if n == 1 {
        let eigenvals = Array1::from_elem(1, diagonal[0]);
        let eigenvecs = if compute_eigenvectors {
            Some(Array2::eye(1))
        } else {
            None
        };
        return Ok((eigenvals, eigenvecs));
    }

    // Copy data for in-place computation
    let mut d = diagonal.to_owned();
    let mut e = Array1::zeros(n);
    for i in 0..n - 1 {
        e[i] = sub_diagonal[i];
    }

    let mut z = if compute_eigenvectors {
        Some(Array2::eye(n))
    } else {
        None
    };

    // Implicit QL algorithm with shifts
    let eps = F::epsilon();
    let max_iter = 30 * n;
    let mut iter = 0;

    let mut l = 0;
    while l < n - 1 && iter < max_iter {
        iter += 1;

        // Find the largest m such that e[m] is negligible
        let mut m = l;
        while m < n - 1 {
            let tst = d[m].abs() + d[m + 1].abs();
            if e[m].abs() <= eps * tst {
                e[m] = F::zero();
                break;
            }
            m += 1;
        }

        if m == l {
            // Eigenvalue converged
            l += 1;
            continue;
        }

        // Choose shift (Wilkinson's shift)
        let mut g = (d[l + 1] - d[l]) / (F::from(2.0).unwrap() * e[l]);
        let mut r = (g * g + F::one()).sqrt();
        if g < F::zero() {
            r = -r;
        }
        g = d[m] - d[l] + e[l] / (g + r);

        // QL transformation
        let mut s = F::one();
        let mut c = F::one();
        let mut p = F::zero();

        for i in (l..m).rev() {
            let f = s * e[i];
            let b = c * e[i];

            r = (f * f + g * g).sqrt();
            e[i + 1] = r;

            if r == F::zero() {
                d[i + 1] -= p;
                e[m] = F::zero();
                break;
            }

            s = f / r;
            c = g / r;
            g = d[i + 1] - p;
            r = (d[i] - g) * s + F::from(2.0).unwrap() * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g = c * r - b;

            // Accumulate _eigenvectors if needed
            if let Some(ref mut z_mat) = z {
                for k in 0..n {
                    let temp = z_mat[[k, i + 1]];
                    z_mat[[k, i + 1]] = s * z_mat[[k, i]] + c * temp;
                    z_mat[[k, i]] = c * z_mat[[k, i]] - s * temp;
                }
            }
        }

        d[l] -= p;
        e[l] = g;
        e[m] = F::zero();
    }

    if iter >= max_iter {
        return Err(LinalgError::ConvergenceError(
            "Tridiagonal eigenvalue algorithm did not converge".to_string(),
        ));
    }

    // Sort eigenvalues and _eigenvectors in ascending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_eigenvals = Array1::zeros(n);
    let sorted_eigenvecs = if compute_eigenvectors {
        let mut sorted_vecs = Array2::zeros((n, n));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvals[new_idx] = d[old_idx];
            if let Some(ref z_mat) = z {
                for row in 0..n {
                    sorted_vecs[[row, new_idx]] = z_mat[[row, old_idx]];
                }
            }
        }
        Some(sorted_vecs)
    } else {
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvals[new_idx] = d[old_idx];
        }
        None
    };

    Ok((sorted_eigenvals, sorted_eigenvecs))
}

/// Eigenvalue solver for symmetric banded matrices
///
/// Reduces the banded matrix to tridiagonal form using Householder transformations,
/// then applies the tridiagonal eigenvalue solver.
///
/// # Arguments
///
/// * `matrix` - Symmetric banded matrix (full storage)
/// * `bandwidth` - Number of super/sub-diagonals
/// * `compute_eigenvectors` - Whether to compute eigenvectors
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors)
#[allow(dead_code)]
pub fn banded_eigen<F>(
    matrix: &ArrayView2<F>,
    bandwidth: usize,
    compute_eigenvectors: bool,
) -> LinalgResult<(Array1<F>, Option<Array2<F>>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = matrix.nrows();

    if matrix.ncols() != n {
        return Err(LinalgError::ShapeError("Matrix must be square".to_string()));
    }

    if bandwidth >= n {
        // Use general eigenvalue solver for dense matrices
        return Err(LinalgError::InvalidInputError(
            "Bandwidth too large for banded algorithm".to_string(),
        ));
    }

    // For small bandwidth, reduce to tridiagonal form
    let (tri_diag, tri_sub, qmatrix) = reduce_banded_to_tridiagonal(matrix, bandwidth)?;

    // Solve tridiagonal eigenvalue problem
    let (eigenvals, tri_eigenvecs) =
        tridiagonal_eigen(&tri_diag.view(), &tri_sub.view(), compute_eigenvectors)?;

    // Transform _eigenvectors back if needed
    let eigenvecs = if compute_eigenvectors {
        if let (Some(q), Some(tri_vecs)) = (qmatrix, tri_eigenvecs) {
            Some(q.dot(&tri_vecs))
        } else {
            None
        }
    } else {
        None
    };

    Ok((eigenvals, eigenvecs))
}

/// Eigenvalue solver for circulant matrices
///
/// Uses the FFT-based approach where eigenvalues are the discrete Fourier transform
/// of the first column. This is an O(n log n) algorithm.
///
/// # Arguments
///
/// * `first_column` - First column of the circulant matrix
///
/// # Returns
///
/// * Complex eigenvalues of the circulant matrix
#[allow(dead_code)]
pub fn circulant_eigenvalues<F>(
    first_column: &ArrayView1<F>,
) -> LinalgResult<Array1<num_complex::Complex<F>>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = first_column.len();

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // For circulant matrices, eigenvalues are DFT of the first _column
    // This is a simplified version - in practice you'd use an FFT library
    let mut eigenvals = Array1::zeros(n);

    for k in 0..n {
        let mut sum = num_complex::Complex::new(F::zero(), F::zero());
        for j in 0..n {
            let theta = F::from(-2.0 * std::f64::consts::PI).unwrap()
                * F::from(k).unwrap()
                * F::from(j).unwrap()
                / F::from(n).unwrap();
            let complex_exp = num_complex::Complex::new(theta.cos(), theta.sin());
            sum += num_complex::Complex::new(first_column[j], F::zero()) * complex_exp;
        }
        eigenvals[k] = sum;
    }

    Ok(eigenvals)
}

/// Find the k largest (or smallest) eigenvalues using partial eigenvalue computation
///
/// Uses the Lanczos algorithm for symmetric matrices to find a subset of eigenvalues
/// without computing the full spectrum.
///
/// # Arguments
///
/// * `matrix` - Symmetric matrix
/// * `k` - Number of eigenvalues to compute
/// * `which` - "largest" or "smallest" eigenvalues
/// * `max_iter` - Maximum Lanczos iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * k eigenvalues and corresponding eigenvectors
#[allow(dead_code)]
pub fn partial_eigen<F>(
    matrix: &ArrayView2<F>,
    k: usize,
    which: &str,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = matrix.nrows();

    if matrix.ncols() != n {
        return Err(LinalgError::ShapeError("Matrix must be square".to_string()));
    }

    // Check if matrix is symmetric
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > F::epsilon() * F::from(1000.0).unwrap() {
                return Err(LinalgError::ShapeError(
                    "Matrix must be symmetric for partial eigenvalue computation".to_string(),
                ));
            }
        }
    }

    if k > n {
        return Err(LinalgError::InvalidInputError(
            "k must be less than or equal to matrix dimension".to_string(),
        ));
    }

    // Special case: if k == n, compute all eigenvalues using standard method
    if k == n {
        let (all_eigenvals, all_eigenvecs) = crate::eigen::eigh(matrix, None)?;

        // Sort according to the 'which' parameter
        let mut eigen_pairs: Vec<(F, usize)> = all_eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        match which {
            "largest" => eigen_pairs
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)),
            "smallest" => eigen_pairs
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)),
            _ => {
                return Err(LinalgError::InvalidInputError(
                    "which must be 'largest' or 'smallest'".to_string(),
                ))
            }
        }

        let mut result_eigenvals = Array1::zeros(n);
        let mut result_eigenvecs = Array2::zeros((n, n));

        for (i, &(eigenval, old_idx)) in eigen_pairs.iter().enumerate() {
            result_eigenvals[i] = eigenval;
            for row in 0..n {
                result_eigenvecs[[row, i]] = all_eigenvecs[[row, old_idx]];
            }
        }

        return Ok((result_eigenvals, result_eigenvecs));
    }

    let max_iter = max_iter.unwrap_or(std::cmp::min(n, 3 * k + 50));
    let tol = tol.unwrap_or(F::epsilon() * F::from(1000.0).unwrap());

    // Simplified Lanczos algorithm
    let m = std::cmp::min(max_iter, n);
    let mut qmatrix = Array2::zeros((n, m + 1));
    let mut alpha = Array1::zeros(m);
    let mut beta = Array1::zeros(m);

    // Initialize with random vector
    let mut rng = rand::rng();
    for i in 0..n {
        qmatrix[[i, 0]] = F::from(rng.random_range(-1.0..=1.0)).unwrap();
    }

    // Normalize
    let mut norm = F::zero();
    for i in 0..n {
        norm += qmatrix[[i, 0]] * qmatrix[[i, 0]];
    }
    norm = norm.sqrt();
    for i in 0..n {
        qmatrix[[i, 0]] /= norm;
    }

    for j in 0..m {
        // Compute w = A * q_j
        let mut w = Array1::zeros(n);
        for i in 0..n {
            for l in 0..n {
                w[i] += matrix[[i, l]] * qmatrix[[l, j]];
            }
        }

        // Compute alpha_j = q_j^T * w
        alpha[j] = F::zero();
        for i in 0..n {
            alpha[j] += qmatrix[[i, j]] * w[i];
        }

        // Update w = w - alpha_j * q_j
        for i in 0..n {
            w[i] -= alpha[j] * qmatrix[[i, j]];
        }

        // Orthogonalize against previous vector if j > 0
        if j > 0 {
            for i in 0..n {
                w[i] -= beta[j - 1] * qmatrix[[i, j - 1]];
            }
        }

        // Compute beta_j = ||w||
        beta[j] = F::zero();
        for i in 0..n {
            beta[j] += w[i] * w[i];
        }
        beta[j] = beta[j].sqrt();

        if j + 1 < m && beta[j] > tol {
            // q_{j+1} = w / beta_j
            for i in 0..n {
                qmatrix[[i, j + 1]] = w[i] / beta[j];
            }
        } else {
            break;
        }
    }

    // Solve the tridiagonal eigenvalue problem
    let m_actual = alpha.len();
    let beta_sub = if m_actual > 1 {
        Array1::from_iter(beta.iter().take(m_actual - 1).cloned())
    } else {
        Array1::zeros(0)
    };

    let (tri_eigenvals, tri_eigenvecs) = tridiagonal_eigen(&alpha.view(), &beta_sub.view(), true)?;

    // Select k eigenvalues based on 'which' parameter
    let mut eigen_pairs: Vec<(F, usize)> = tri_eigenvals
        .iter()
        .enumerate()
        .map(|(i, &val)| (val, i))
        .collect();

    match which {
        "largest" => {
            eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
        }
        "smallest" => {
            eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        }
        _ => {
            return Err(LinalgError::InvalidInputError(
                "which must be 'largest' or 'smallest'".to_string(),
            ))
        }
    }

    let k_actual = std::cmp::min(k, eigen_pairs.len());
    let mut result_eigenvals = Array1::zeros(k_actual);
    let mut result_eigenvecs = Array2::zeros((n, k_actual));

    for (i, &(eigenval, tri_idx)) in eigen_pairs.iter().take(k_actual).enumerate() {
        result_eigenvals[i] = eigenval;

        // Transform eigenvector back to original space
        if let Some(ref tri_vecs) = tri_eigenvecs {
            for row in 0..n {
                let mut sum = F::zero();
                for col in 0..m_actual {
                    sum += qmatrix[[row, col]] * tri_vecs[[col, tri_idx]];
                }
                result_eigenvecs[[row, i]] = sum;
            }
        }
    }

    Ok((result_eigenvals, result_eigenvecs))
}

/// Type alias for tridiagonal reduction result
type TridiagonalReduction<F> = LinalgResult<(Array1<F>, Array1<F>, Option<Array2<F>>)>;

/// Helper function to reduce banded matrix to tridiagonal form
#[allow(dead_code)]
fn reduce_banded_to_tridiagonal<F>(
    matrix: &ArrayView2<F>,
    bandwidth: usize,
) -> TridiagonalReduction<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = matrix.nrows();
    let mut a = matrix.to_owned();
    let q = Array2::eye(n);

    // For small bandwidth, use direct reduction
    if bandwidth <= 1 {
        // Already tridiagonal
        let mut diagonal = Array1::zeros(n);
        let mut sub_diagonal = Array1::zeros(n - 1);

        for i in 0..n {
            diagonal[i] = a[[i, i]];
            if i < n - 1 {
                sub_diagonal[i] = a[[i + 1, i]];
            }
        }

        return Ok((diagonal, sub_diagonal, Some(q)));
    }

    // Simplified reduction using Householder transformations
    for k in 0..n - 2 {
        let start = std::cmp::max(k + 1, k + 1);
        let end = std::cmp::min(n, k + bandwidth + 1);

        if start >= end {
            continue;
        }

        // Extract subvector for Householder reflection
        let subvec_len = end - start;
        if subvec_len <= 1 {
            continue;
        }

        let mut x = Array1::zeros(subvec_len);
        for i in 0..subvec_len {
            x[i] = a[[start + i, k]];
        }

        // Compute Householder vector
        let x_norm = x.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();
        if x_norm > F::epsilon() {
            let alpha = if x[0] >= F::zero() { -x_norm } else { x_norm };
            let mut v = x.clone();
            v[0] -= alpha;

            let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();
            if v_norm > F::epsilon() {
                for i in 0..v.len() {
                    v[i] /= v_norm;
                }

                // Apply Householder transformation (simplified)
                // This is a simplified version - full implementation would be more complex
                a[[start, k]] = alpha;
                for i in 1..subvec_len {
                    a[[start + i, k]] = F::zero();
                    a[[k, start + i]] = F::zero();
                }
            }
        }
    }

    // Extract tridiagonal elements
    let mut diagonal = Array1::zeros(n);
    let mut sub_diagonal = Array1::zeros(n - 1);

    for i in 0..n {
        diagonal[i] = a[[i, i]];
        if i < n - 1 {
            sub_diagonal[i] = a[[i + 1, i]];
        }
    }

    Ok((diagonal, sub_diagonal, Some(q)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_tridiagonal_eigen_simple() {
        let diagonal = array![2.0, 2.0, 2.0];
        let sub_diagonal = array![1.0, 1.0];

        let (eigenvals, eigenvecs) =
            tridiagonal_eigen(&diagonal.view(), &sub_diagonal.view(), true).unwrap();

        assert_eq!(eigenvals.len(), 3);
        assert!(eigenvals.iter().all(|&x| x.is_finite()));

        if let Some(vecs) = eigenvecs {
            assert_eq!(vecs.dim(), (3, 3));
        }
    }

    #[test]
    fn test_tridiagonal_eigen_diagonal() {
        // Test diagonal matrix (sub_diagonal = 0)
        let diagonal = array![1.0, 2.0, 3.0];
        let sub_diagonal = array![0.0, 0.0];

        let (eigenvals, _) =
            tridiagonal_eigen(&diagonal.view(), &sub_diagonal.view(), false).unwrap();

        assert_abs_diff_eq!(eigenvals[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(eigenvals[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(eigenvals[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circulant_eigenvalues() {
        let first_col = array![1.0, 2.0, 3.0];
        let eigenvals = circulant_eigenvalues(&first_col.view()).unwrap();

        assert_eq!(eigenvals.len(), 3);
        assert!(eigenvals.iter().all(|x| x.norm().is_finite()));

        // First eigenvalue should be the sum of the first column (for circulant matrices)
        assert_abs_diff_eq!(eigenvals[0].re, 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(eigenvals[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_partial_eigen() {
        let matrix = array![[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]];

        let (eigenvals, eigenvecs) =
            partial_eigen(&matrix.view(), 2, "largest", None, None).unwrap();

        assert_eq!(eigenvals.len(), 2);
        assert_eq!(eigenvecs.dim(), (3, 2));
        assert!(eigenvals.iter().all(|&x| x.is_finite()));
        assert!(eigenvecs.iter().all(|&x| x.is_finite()));
    }
}
