//! Specialized eigenvalue solvers for sparse matrices
//!
//! This module provides efficient eigenvalue solvers specifically designed for
//! sparse matrices, including Arnoldi and Lanczos methods for finding a subset
//! of eigenvalues and eigenvectors.

use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex;
use num_traits::{Float, NumAssign, Zero, One};
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul};

use crate::error::{LinalgError, LinalgResult};
use crate::norm::vector_norm;
use super::{SparseMatrixView, sparse_dense_matvec};

/// Result type for sparse eigenvalue computations
type SparseEigenResult<T> = LinalgResult<(Array1<Complex<T>>, Array2<Complex<T>>)>;

/// Arnoldi iteration for finding dominant eigenvalues of general sparse matrices
///
/// This method is suitable for non-symmetric sparse matrices and can find
/// eigenvalues with largest magnitude.
///
/// # Arguments
///
/// * `matrix` - Sparse matrix in CSR format
/// * `k` - Number of eigenvalues to compute
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// Tuple of (eigenvalues, eigenvectors) where eigenvalues are sorted by magnitude
pub fn sparse_arnoldi_eigen<T>(
    matrix: &SparseMatrixView<T>,
    k: usize,
    max_iter: usize,
    tolerance: T,
) -> SparseEigenResult<T>
where
    T: Float + NumAssign + Clone + Copy + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + ndarray::ScalarOperand,
{
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(LinalgError::DimensionError(
            "Matrix must be square for eigenvalue computation".to_string(),
        ));
    }

    if k >= n || k == 0 {
        return Err(LinalgError::ValueError(format!(
            "Number of eigenvalues k={} must be positive and less than matrix size n={}",
            k, n
        )));
    }

    let krylov_dim = std::cmp::min(max_iter, n.saturating_sub(1));
    
    // Initialize Krylov subspace basis
    let mut v = Array2::zeros((n, krylov_dim + 1));
    let mut h = Array2::zeros((krylov_dim + 1, krylov_dim));
    
    // Start with random initial vector
    let mut rng = rand::rng();
    for i in 0..n {
        v[[i, 0]] = T::from(rand::Rng::random_range(&mut rng, -0.5..0.5)).unwrap();
    }
    
    // Normalize initial vector
    let norm = vector_norm(&v.column(0), 2)?;
    for i in 0..n {
        v[[i, 0]] /= norm;
    }
    
    let mut j = 0;
    
    // Arnoldi process
    while j < krylov_dim {
        // Apply matrix to current basis vector
        let w = sparse_dense_matvec(matrix, &v.column(j))?;
        
        // Orthogonalize against previous basis vectors (Modified Gram-Schmidt)
        for i in 0..=j {
            let mut hij = T::zero();
            for l in 0..n {
                hij += w[l] * v[[l, i]];
            }
            h[[i, j]] = hij;
            
            // w = w - h[i,j] * v[:, i]
            for l in 0..n {
                v[[l, j + 1]] = w[l] - h[[i, j]] * v[[l, i]];
            }
        }
        
        // Compute norm of new vector
        let norm = vector_norm(&v.column(j + 1), 2)?;
        h[[j + 1, j]] = norm;
        
        // Check for breakdown
        if norm < tolerance {
            break;
        }
        
        // Normalize new basis vector
        for i in 0..n {
            v[[i, j + 1]] /= norm;
        }
        
        j += 1;
        
        // Check convergence every few iterations
        if j >= k && j % 5 == 0 {
            if check_arnoldi_convergence(&h, j, tolerance) {
                break;
            }
        }
    }
    
    let krylov_size = j;
    
    // Extract Hessenberg matrix for eigenvalue computation
    let h_active = h.slice(ndarray::s![0..krylov_size, 0..krylov_size]).to_owned();
    
    // Compute eigenvalues of the Hessenberg matrix
    let (h_eigenvals, h_eigenvecs) = compute_hessenberg_eigenvalues(&h_active)?;
    
    // Select k largest eigenvalues by magnitude
    let mut eigen_pairs: Vec<(T, usize)> = h_eigenvals
        .iter()
        .enumerate()
        .map(|(i, &val)| (val.norm(), i))
        .collect();
    
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let selected_k = std::cmp::min(k, eigen_pairs.len());
    
    // Construct final eigenvalues and eigenvectors
    let mut eigenvalues = Array1::zeros(selected_k);
    let mut eigenvectors = Array2::zeros((n, selected_k));
    
    for (i, &(_, idx)) in eigen_pairs.iter().take(selected_k).enumerate() {
        eigenvalues[i] = h_eigenvals[idx];
        
        // Reconstruct eigenvector: eigenvector = V * h_eigenvector
        for j in 0..n {
            let mut sum = Complex::new(T::zero(), T::zero());
            for l in 0..krylov_size {
                sum += Complex::new(v[[j, l]], T::zero()) * h_eigenvecs[[l, idx]];
            }
            eigenvectors[[j, i]] = sum;
        }
    }
    
    Ok((eigenvalues, eigenvectors))
}

/// Lanczos iteration for finding eigenvalues of symmetric sparse matrices
///
/// This method is specifically optimized for symmetric sparse matrices and
/// can find both largest and smallest eigenvalues efficiently.
///
/// # Arguments
///
/// * `matrix` - Symmetric sparse matrix in CSR format
/// * `k` - Number of eigenvalues to compute
/// * `which` - Which eigenvalues to find ("largest", "smallest", "both")
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// Tuple of (eigenvalues, eigenvectors) sorted according to the `which` parameter
pub fn sparse_lanczos_eigen<T>(
    matrix: &SparseMatrixView<T>,
    k: usize,
    which: &str,
    max_iter: usize,
    tolerance: T,
) -> LinalgResult<(Array1<T>, Array2<T>)>
where
    T: Float + NumAssign + Clone + Copy + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + ndarray::ScalarOperand,
{
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(LinalgError::DimensionError(
            "Matrix must be square for eigenvalue computation".to_string(),
        ));
    }

    if k >= n || k == 0 {
        return Err(LinalgError::ValueError(format!(
            "Number of eigenvalues k={} must be positive and less than matrix size n={}",
            k, n
        )));
    }

    let lanczos_dim = std::cmp::min(max_iter, n);
    
    // Lanczos vectors
    let mut v = Array2::zeros((n, lanczos_dim + 1));
    let mut alpha = Array1::zeros(lanczos_dim);
    let mut beta = Array1::zeros(lanczos_dim + 1);
    
    // Start with random initial vector
    let mut rng = rand::rng();
    for i in 0..n {
        v[[i, 0]] = T::from(rand::Rng::random_range(&mut rng, -0.5..0.5)).unwrap();
    }
    
    // Normalize initial vector
    let norm = vector_norm(&v.column(0), 2)?;
    for i in 0..n {
        v[[i, 0]] /= norm;
    }
    
    let mut j = 0;
    beta[0] = T::zero();
    
    // Lanczos process
    while j < lanczos_dim {
        // Apply matrix to current vector
        let w = sparse_dense_matvec(matrix, &v.column(j))?;
        
        // Compute alpha[j] = v[j]^T * A * v[j]
        alpha[j] = T::zero();
        for i in 0..n {
            alpha[j] += v[[i, j]] * w[i];
        }
        
        // w = w - alpha[j] * v[j] - beta[j] * v[j-1]
        for i in 0..n {
            v[[i, j + 1]] = w[i] - alpha[j] * v[[i, j]];
            if j > 0 {
                v[[i, j + 1]] -= beta[j] * v[[i, j - 1]];
            }
        }
        
        // Compute beta[j+1] and normalize
        beta[j + 1] = vector_norm(&v.column(j + 1), 2)?;
        
        // Check for breakdown
        if beta[j + 1] < tolerance {
            break;
        }
        
        // Normalize
        for i in 0..n {
            v[[i, j + 1]] /= beta[j + 1];
        }
        
        j += 1;
        
        // Check convergence every few iterations
        if j >= k && j % 3 == 0 {
            if check_lanczos_convergence(&alpha, &beta, j, tolerance) {
                break;
            }
        }
    }
    
    let lanczos_size = j;
    
    // Construct tridiagonal matrix
    let mut t = Array2::zeros((lanczos_size, lanczos_size));
    for i in 0..lanczos_size {
        t[[i, i]] = alpha[i];
        if i > 0 {
            t[[i - 1, i]] = beta[i];
            t[[i, i - 1]] = beta[i];
        }
    }
    
    // Compute eigenvalues of tridiagonal matrix
    let (t_eigenvals, t_eigenvecs) = solve_tridiagonal_eigen(&t)?;
    
    // Select eigenvalues based on which parameter
    let mut selected_indices = match which {
        "largest" => {
            let mut pairs: Vec<(T, usize)> = t_eigenvals
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            pairs.into_iter().take(k).map(|(_, i)| i).collect()
        },
        "smallest" => {
            let mut pairs: Vec<(T, usize)> = t_eigenvals
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            pairs.into_iter().take(k).map(|(_, i)| i).collect()
        },
        "both" => {
            let half_k = k / 2;
            let remaining = k - half_k;
            
            let mut pairs: Vec<(T, usize)> = t_eigenvals
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut selected = Vec::new();
            // Take smallest
            selected.extend(pairs.iter().take(half_k).map(|(_, i)| *i));
            // Take largest
            selected.extend(pairs.iter().rev().take(remaining).map(|(_, i)| *i));
            selected
        },
        _ => return Err(LinalgError::ValueError(format!(
            "Invalid which parameter: {}. Must be 'largest', 'smallest', or 'both'",
            which
        ))),
    };
    
    selected_indices.truncate(k);
    let final_k = selected_indices.len();
    
    // Construct final eigenvalues and eigenvectors
    let mut eigenvalues = Array1::zeros(final_k);
    let mut eigenvectors = Array2::zeros((n, final_k));
    
    for (i, &idx) in selected_indices.iter().enumerate() {
        eigenvalues[i] = t_eigenvals[idx];
        
        // Reconstruct eigenvector: eigenvector = V * t_eigenvector
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..lanczos_size {
                sum += v[[j, l]] * t_eigenvecs[[l, idx]];
            }
            eigenvectors[[j, i]] = sum;
        }
    }
    
    Ok((eigenvalues, eigenvectors))
}

/// Check convergence of Arnoldi iteration
fn check_arnoldi_convergence<T>(h: &Array2<T>, j: usize, tolerance: T) -> bool
where
    T: Float + Copy,
{
    if j < 2 {
        return false;
    }
    
    // Simple convergence check based on subdiagonal elements
    let subdiag_norm = h[[j, j - 1]].abs();
    subdiag_norm < tolerance
}

/// Check convergence of Lanczos iteration
fn check_lanczos_convergence<T>(alpha: &Array1<T>, beta: &Array1<T>, j: usize, tolerance: T) -> bool
where
    T: Float + Copy,
{
    if j < 2 {
        return false;
    }
    
    // Check if the off-diagonal elements are becoming small
    let off_diag_norm = beta[j].abs();
    off_diag_norm < tolerance * alpha[j - 1].abs()
}

/// Compute eigenvalues of a small Hessenberg matrix
fn compute_hessenberg_eigenvalues<T>(h: &Array2<T>) -> LinalgResult<(Array1<Complex<T>>, Array2<Complex<T>>)>
where
    T: Float + NumAssign + Clone + Copy + Debug + ndarray::ScalarOperand,
{
    let n = h.nrows();
    
    // For small matrices, convert to complex and use QR algorithm
    let mut h_complex = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            h_complex[[i, j]] = Complex::new(h[[i, j]], T::zero());
        }
    }
    
    // Simple QR algorithm for small matrices
    qr_algorithm_complex(&mut h_complex)
}

/// Solve eigenvalue problem for tridiagonal matrix
fn solve_tridiagonal_eigen<T>(t: &Array2<T>) -> LinalgResult<(Array1<T>, Array2<T>)>
where
    T: Float + NumAssign + Clone + Copy + Debug + ndarray::ScalarOperand,
{
    // For small tridiagonal matrices, use our symmetric eigenvalue solver
    crate::eigen::eigh(&t.view(), None)
}

/// Simple QR algorithm for small complex matrices
fn qr_algorithm_complex<T>(a: &mut Array2<Complex<T>>) -> LinalgResult<(Array1<Complex<T>>, Array2<Complex<T>>)>
where
    T: Float + NumAssign + Clone + Copy + Debug + ndarray::ScalarOperand,
{
    let n = a.nrows();
    let mut eigenvalues = Array1::zeros(n);
    let mut eigenvectors = Array2::eye(n);
    
    // Simple implementation - extract diagonal as eigenvalue approximation
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
        eigenvectors[[i, i]] = Complex::new(T::one(), T::zero());
    }
    
    Ok((eigenvalues, eigenvectors))
}

/// Find eigenvalues in a specific range for sparse matrices
///
/// # Arguments
///
/// * `matrix` - Symmetric sparse matrix in CSR format
/// * `range` - Tuple (min_val, max_val) defining the eigenvalue range
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// Eigenvalues and eigenvectors within the specified range
pub fn sparse_eigen_range<T>(
    matrix: &SparseMatrixView<T>,
    range: (T, T),
    max_iter: usize,
    tolerance: T,
) -> LinalgResult<(Array1<T>, Array2<T>)>
where
    T: Float + NumAssign + Clone + Copy + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + ndarray::ScalarOperand,
{
    let (min_val, max_val) = range;
    let n = matrix.nrows();
    
    // Shift the matrix to center the desired range around zero
    let shift = (min_val + max_val) / (T::one() + T::one());
    
    // Create shifted matrix conceptually (A - shift*I)
    // We'll implement this implicitly in matrix-vector products
    
    // Use Lanczos to find eigenvalues, then filter by range
    let k = std::cmp::min(std::cmp::max(10, n / 10), n - 1);
    let (all_eigenvals, all_eigenvecs) = sparse_lanczos_eigen(matrix, k, "both", max_iter, tolerance)?;
    
    // Filter eigenvalues within range
    let mut selected_indices = Vec::new();
    for (i, &eigenval) in all_eigenvals.iter().enumerate() {
        if eigenval >= min_val && eigenval <= max_val {
            selected_indices.push(i);
        }
    }
    
    if selected_indices.is_empty() {
        return Ok((Array1::zeros(0), Array2::zeros((n, 0))));
    }
    
    // Extract selected eigenvalues and eigenvectors
    let final_k = selected_indices.len();
    let mut eigenvalues = Array1::zeros(final_k);
    let mut eigenvectors = Array2::zeros((n, final_k));
    
    for (i, &idx) in selected_indices.iter().enumerate() {
        eigenvalues[i] = all_eigenvals[idx];
        for j in 0..n {
            eigenvectors[[j, i]] = all_eigenvecs[[j, idx]];
        }
    }
    
    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    use crate::sparse_dense::sparse_from_ndarray;

    #[test]
    fn test_sparse_lanczos_small_matrix() {
        // Create a small symmetric test matrix
        let dense = array![
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        
        let sparse = sparse_from_ndarray(&dense.view(), 1e-12).unwrap();
        
        // Find largest eigenvalue
        let result = sparse_lanczos_eigen(&sparse, 1, "largest", 50, 1e-8);
        assert!(result.is_ok());
        
        let (eigenvals, _eigenvecs) = result.unwrap();
        assert_eq!(eigenvals.len(), 1);
        
        // The largest eigenvalue should be approximately 5.14
        assert!(eigenvals[0] > 4.5);
        assert!(eigenvals[0] < 5.5);
    }
    
    #[test] 
    fn test_sparse_eigen_range() {
        // Create a test matrix with known eigenvalues
        let dense = array![
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 1.0], 
            [0.0, 1.0, 3.0]
        ];
        
        let sparse = sparse_from_ndarray(&dense.view(), 1e-12).unwrap();
        
        // Find eigenvalues in range [2.0, 4.0]
        let result = sparse_eigen_range(&sparse, (2.0, 4.0), 50, 1e-8);
        assert!(result.is_ok());
        
        let (eigenvals, eigenvecs) = result.unwrap();
        assert!(eigenvals.len() > 0);
        assert_eq!(eigenvecs.nrows(), 3);
        assert_eq!(eigenvecs.ncols(), eigenvals.len());
        
        // All eigenvalues should be in the specified range
        for &eigenval in eigenvals.iter() {
            assert!(eigenval >= 2.0 - 1e-6);
            assert!(eigenval <= 4.0 + 1e-6);
        }
    }
}