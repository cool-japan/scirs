//! Utility functions for sparse matrices
//!
//! This module provides utility functions for sparse matrices.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use num_traits::Zero;

/// Create an identity matrix in CSR format
///
/// # Arguments
///
/// * `n` - Size of the matrix (n x n)
///
/// # Returns
///
/// * Identity matrix in CSR format
///
/// # Example
///
/// ```
/// use scirs2_sparse::utils::identity;
///
/// // Create a 3x3 identity matrix
/// let eye = identity(3).unwrap();
/// ```
#[allow(dead_code)]
pub fn identity(n: usize) -> SparseResult<CsrMatrix<f64>> {
    if n == 0 {
        return Err(SparseError::ValueError(
            "Matrix size must be positive".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(n);
    let mut row_indices = Vec::with_capacity(n);
    let mut col_indices = Vec::with_capacity(n);

    for i in 0..n {
        data.push(1.0);
        row_indices.push(i);
        col_indices.push(i);
    }

    CsrMatrix::new(data, row_indices, col_indices, (n, n))
}

/// Create a diagonal matrix in CSR format
///
/// # Arguments
///
/// * `diag` - Vector of diagonal elements
///
/// # Returns
///
/// * Diagonal matrix in CSR format
///
/// # Example
///
/// ```
/// use scirs2_sparse::utils::diag;
///
/// // Create a diagonal matrix with elements [1, 2, 3]
/// let d = diag(&[1.0, 2.0, 3.0]).unwrap();
/// ```
#[allow(dead_code)]
pub fn diag(diag: &[f64]) -> SparseResult<CsrMatrix<f64>> {
    if diag.is_empty() {
        return Err(SparseError::ValueError(
            "Diagonal vector must not be empty".to_string(),
        ));
    }

    let n = diag.len();
    let mut data = Vec::with_capacity(n);
    let mut row_indices = Vec::with_capacity(n);
    let mut col_indices = Vec::with_capacity(n);

    for (i, &val) in diag.iter().enumerate() {
        if val != 0.0 {
            data.push(val);
            row_indices.push(i);
            col_indices.push(i);
        }
    }

    CsrMatrix::new(data, row_indices, col_indices, (n, n))
}

/// Calculate the density of a sparse matrix
///
/// # Arguments
///
/// * `shape` - Matrix shape (rows, cols)
/// * `nnz` - Number of non-zero elements
///
/// # Returns
///
/// * Density (fraction of non-zero elements)
#[allow(dead_code)]
pub fn density(shape: (usize, usize), nnz: usize) -> f64 {
    let (rows, cols) = shape;
    if rows == 0 || cols == 0 {
        return 0.0;
    }

    nnz as f64 / (rows * cols) as f64
}

/// Check if a sparse matrix is symmetric
///
/// # Arguments
///
/// * `matrix` - Sparse matrix to check
///
/// # Returns
///
/// * true if the matrix is symmetric, false otherwise
#[allow(dead_code)]
pub fn is_symmetric(matrix: &CsrMatrix<f64>) -> bool {
    let (rows, cols) = matrix.shape();

    // Must be square
    if rows != cols {
        return false;
    }

    // Check if A = A^T
    let transposed = matrix.transpose();
    let a_dense = matrix.to_dense();
    let at_dense = transposed.to_dense();

    for i in 0..rows {
        for j in 0..cols {
            if (a_dense[i][j] - at_dense[i][j]).abs() > 1e-10 {
                return false;
            }
        }
    }

    true
}

/// Generate a random sparse matrix with given density
///
/// # Arguments
///
/// * `shape` - Matrix shape (rows, cols)
/// * `density` - Desired density (0.0 to 1.0)
///
/// # Returns
///
/// * Random sparse matrix in CSR format
#[allow(dead_code)]
pub fn random(shape: (usize, usize), density: f64) -> SparseResult<CsrMatrix<f64>> {
    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::ValueError(format!(
            "Density must be between 0 and 1, got {density}"
        )));
    }

    let (rows, cols) = shape;
    if rows == 0 || cols == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    // Calculate number of non-zero elements
    let nnz = (rows * cols) as f64 * density;
    let nnz = nnz.round() as usize;

    if nnz == 0 {
        // Return empty matrix
        return CsrMatrix::new(Vec::new(), Vec::new(), Vec::new(), shape);
    }

    // Generate random non-zero elements
    let mut data = Vec::with_capacity(nnz);
    let mut row_indices = Vec::with_capacity(nnz);
    let mut col_indices = Vec::with_capacity(nnz);

    // Use a simple approach: randomly select nnz cells
    // Note: this is not the most efficient approach for very sparse matrices
    let mut used = vec![vec![false; cols]; rows];
    let mut count = 0;

    use rand::Rng;
    let mut rng = rand::rng();

    while count < nnz {
        let i = rng.random_range(0..rows);
        let j = rng.random_range(0..cols);

        if !used[i][j] {
            used[i][j] = true;
            data.push(rng.random_range(-1.0..1.0));
            row_indices.push(i);
            col_indices.push(j);
            count += 1;
        }
    }

    CsrMatrix::new(data, row_indices, col_indices, shape)
}

/// Calculate the sparsity pattern of a matrix
///
/// # Arguments
///
/// * `matrix` - Sparse matrix
///
/// # Returns
///
/// * Vector of vectors representing the sparsity pattern (1 for non-zero, 0 for zero)
#[allow(dead_code)]
pub fn sparsity_pattern<T>(matrix: &CsrMatrix<T>) -> Vec<Vec<usize>>
where
    T: Clone + Copy + Zero + PartialEq,
{
    let (rows, cols) = matrix.shape();
    let dense = matrix.to_dense();

    let mut pattern = vec![vec![0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            if dense[i][j] != T::zero() {
                pattern[i][j] = 1;
            }
        }
    }

    pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let n = 3;
        let eye = identity(n).unwrap();

        assert_eq!(eye.shape(), (n, n));
        assert_eq!(eye.nnz(), n);

        let dense = eye.to_dense();
        for (i, row) in dense.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(value, expected);
            }
        }
    }

    #[test]
    fn test_diag() {
        let diag_elements = [1.0, 2.0, 3.0];
        let d = diag(&diag_elements).unwrap();

        assert_eq!(d.shape(), (3, 3));
        assert_eq!(d.nnz(), 3);

        let dense = d.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { diag_elements[i] } else { 0.0 };
                assert_eq!(dense[i][j], expected);
            }
        }
    }

    #[test]
    fn test_density() {
        // Matrix with 25% non-zero elements
        assert_relative_eq!(density((4, 4), 4), 0.25, epsilon = 1e-10);

        // Empty matrix
        assert_relative_eq!(density((10, 10), 0), 0.0, epsilon = 1e-10);

        // Full matrix
        assert_relative_eq!(density((5, 5), 25), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_symmetric() {
        // Create a symmetric matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 0, 2];
        let data = vec![1.0, 2.0, 2.0, 3.0, 0.0, 4.0]; // Note: explicitly setting a zero
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();

        // A symmetric matrix should have the same value at (i,j) and (j,i)
        assert!(is_symmetric(&matrix));

        // Create a non-symmetric matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 3.0, 0.0, 4.0]; // Changed 2.0 to 3.0
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();

        assert!(!is_symmetric(&matrix));
    }

    #[test]
    fn test_sparsity_pattern() {
        // Create a sparse matrix
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();

        // Calculate sparsity pattern
        let pattern = sparsity_pattern(&matrix);

        // Expected pattern:
        // [1 0 1]
        // [0 0 1]
        // [1 1 0]
        let expected = vec![vec![1, 0, 1], vec![0, 0, 1], vec![1, 1, 0]];

        assert_eq!(pattern, expected);
    }
}
