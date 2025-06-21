//! Special matrix functions
//!
//! This module provides specialized matrix functions for scientific computing, including:
//!
//! - Block diagonal matrices
//! - Matrix exponential, logarithm, and square root functions
//! - Matrix sign function and other matrix decompositions

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::matrix_functions;
use crate::solve::solve_multiple;

/// Construct a block diagonal matrix from provided matrices.
///
/// # Arguments
///
/// * `arrays` - A slice of matrices to be arranged on the diagonal
///
/// # Returns
///
/// * Block diagonal matrix with input matrices on the diagonal
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::special::block_diag;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = block_diag(&[&a.view(), &b.view()]).unwrap();
/// // c is a 4x4 matrix with a in the top-left and b in the bottom-right
/// ```
pub fn block_diag<F>(arrays: &[&ArrayView2<F>]) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    if arrays.is_empty() {
        return Err(LinalgError::ShapeError(
            "At least one array must be provided".to_string(),
        ));
    }

    // Calculate output dimensions
    let mut n_rows = 0;
    let mut n_cols = 0;
    for arr in arrays {
        n_rows += arr.nrows();
        n_cols += arr.ncols();
    }

    // Create output array filled with zeros
    let mut result = Array2::zeros((n_rows, n_cols));

    // Place each input matrix on the diagonal
    let mut r_idx = 0;
    let mut c_idx = 0;
    for arr in arrays {
        let rows = arr.nrows();
        let cols = arr.ncols();

        // Copy the array into the result at the current position
        for i in 0..rows {
            for j in 0..cols {
                result[[r_idx + i, c_idx + j]] = arr[[i, j]];
            }
        }

        // Update indices for the next array
        r_idx += rows;
        c_idx += cols;
    }

    Ok(result)
}

/// Compute the matrix exponential.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Matrix exponential of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::special;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]];
/// let result = special::expm(&a.view(), None);
/// assert!(result.is_ok());
/// ```
pub fn expm<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    // Redirect to the implementation in matrix_functions module
    matrix_functions::expm(a, workers)
}

/// Compute the matrix logarithm.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix logarithm of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::special::logm;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let result = logm(&a.view());
/// assert!(result.is_ok());
/// ```
pub fn logm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    // Redirect to the implementation in matrix_functions module
    matrix_functions::logm(a)
}

/// Compute the matrix square root.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix square root of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::special;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let result = special::sqrtm(&a.view());
/// assert!(result.is_ok());
/// ```
pub fn sqrtm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    // Redirect to the implementation in matrix_functions module with default parameters
    matrix_functions::sqrtm(a, 20, F::from(1e-10).unwrap())
}

/// Compute the matrix sign function using Newton's method.
///
/// For a matrix A with no eigenvalues on the imaginary axis, the sign function is defined as:
/// sign(A) = A(A²)^(-1/2)
///
/// The matrix sign function is useful for spectral decompositions and solving matrix equations.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Matrix sign function of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::special;
///
/// // Positive definite matrix - should yield the identity matrix
/// let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let result = special::signm(&a.view(), 20, 1e-10);
/// assert!(result.is_ok());
/// let sign_a = result.unwrap();
/// // All eigenvalues are positive, so sign(A) = I
/// assert!((sign_a[[0, 0]] - 1.0).abs() < 1e-8);
/// assert!((sign_a[[1, 1]] - 1.0).abs() < 1e-8);
/// ```
pub fn signm<F>(a: &ArrayView2<F>, max_iter: usize, tol: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute sign function, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let val = a[[0, 0]];
        let mut result = Array2::zeros((1, 1));
        if val == F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute sign of zero".to_string(),
            ));
        }
        result[[0, 0]] = if val > F::zero() { F::one() } else { -F::one() };
        return Ok(result);
    }

    // Newton's method for the matrix sign function
    // Xₙ₊₁ = 0.5 * (Xₙ + Xₙ⁻¹)
    let mut x = a.to_owned();
    let identity = Array2::eye(n);

    for _ in 0..max_iter {
        // Compute X_inv
        let x_inv = match solve_multiple(&x.view(), &identity.view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::InvalidInputError(
                    "Matrix is singular during sign function iteration".to_string(),
                ))
            }
        };

        // Newton iteration: X_{k+1} = 0.5 * (X_k + X_k^{-1})
        let mut x_next = Array2::zeros((n, n));
        let half = F::from(0.5).unwrap();

        for i in 0..n {
            for j in 0..n {
                x_next[[i, j]] = half * (x[[i, j]] + x_inv[[i, j]]);
            }
        }

        // Check for convergence
        let mut error = F::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = (x_next[[i, j]] - x[[i, j]]).abs();
                if diff > error {
                    error = diff;
                }
            }
        }

        x = x_next;

        if error < tol {
            return Ok(x);
        }
    }

    // Return the current approximation if max iterations reached
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_block_diag() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = block_diag(&[&a.view(), &b.view()]).unwrap();

        assert_eq!(result.shape(), &[4, 4]);

        // Check top-left block (matrix a)
        assert_relative_eq!(result[[0, 0]], 1.0);
        assert_relative_eq!(result[[0, 1]], 2.0);
        assert_relative_eq!(result[[1, 0]], 3.0);
        assert_relative_eq!(result[[1, 1]], 4.0);

        // Check bottom-right block (matrix b)
        assert_relative_eq!(result[[2, 2]], 5.0);
        assert_relative_eq!(result[[2, 3]], 6.0);
        assert_relative_eq!(result[[3, 2]], 7.0);
        assert_relative_eq!(result[[3, 3]], 8.0);

        // Check zeros in off-diagonal blocks
        assert_relative_eq!(result[[0, 2]], 0.0);
        assert_relative_eq!(result[[0, 3]], 0.0);
        assert_relative_eq!(result[[1, 2]], 0.0);
        assert_relative_eq!(result[[1, 3]], 0.0);
        assert_relative_eq!(result[[2, 0]], 0.0);
        assert_relative_eq!(result[[2, 1]], 0.0);
        assert_relative_eq!(result[[3, 0]], 0.0);
        assert_relative_eq!(result[[3, 1]], 0.0);
    }

    #[test]
    fn test_empty_block_diag() {
        let result = block_diag::<f64>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_function_redirects() {
        // Test that the special module functions correctly redirect to matrix_functions
        let a = array![[4.0, 0.0], [0.0, 9.0]];

        // Test sqrtm
        let sqrt_a = sqrtm(&a.view()).unwrap();
        assert_relative_eq!(sqrt_a[[0, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[1, 1]], 3.0, epsilon = 1e-10);

        // Test logm with identity matrix
        let id = array![[1.0, 0.0], [0.0, 1.0]];
        let log_id = logm(&id.view()).unwrap();
        assert!(log_id[[0, 0]].abs() < 1e-10);
        assert!(log_id[[1, 1]].abs() < 1e-10);

        // Test expm with zero matrix
        let zero = array![[0.0, 0.0], [0.0, 0.0]];
        let exp_zero = expm(&zero.view(), None).unwrap();
        assert_relative_eq!(exp_zero[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp_zero[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_sign_function() {
        // Test matrix sign function with various cases

        // Case 1: Positive definite matrix - sign(A) = I
        let a = array![[2.0, 0.0], [0.0, 3.0]];
        let sign_a = signm(&a.view(), 20, 1e-10).unwrap();
        assert_relative_eq!(sign_a[[0, 0]], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sign_a[[1, 1]], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sign_a[[0, 1]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(sign_a[[1, 0]], 0.0, epsilon = 1e-8);

        // Case 2: Negative definite matrix - sign(A) = -I
        let b = array![[-2.0, 0.0], [0.0, -3.0]];
        let sign_b = signm(&b.view(), 20, 1e-10).unwrap();
        assert_relative_eq!(sign_b[[0, 0]], -1.0, epsilon = 1e-8);
        assert_relative_eq!(sign_b[[1, 1]], -1.0, epsilon = 1e-8);
        assert_relative_eq!(sign_b[[0, 1]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(sign_b[[1, 0]], 0.0, epsilon = 1e-8);

        // Case 3: Mixed eigenvalues - diagonal case
        let c = array![[2.0, 0.0], [0.0, -3.0]];
        let sign_c = signm(&c.view(), 20, 1e-10).unwrap();
        assert_relative_eq!(sign_c[[0, 0]], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sign_c[[1, 1]], -1.0, epsilon = 1e-8);
        assert_relative_eq!(sign_c[[0, 1]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(sign_c[[1, 0]], 0.0, epsilon = 1e-8);
    }
}
