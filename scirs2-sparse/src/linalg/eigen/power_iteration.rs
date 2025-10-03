//! Power iteration method for sparse matrix eigenvalue computation
//!
//! This module implements the power iteration algorithm for finding the largest
//! eigenvalue and corresponding eigenvector of symmetric sparse matrices.

use crate::error::{SparseError, SparseResult};
use crate::sym_csr::SymCsrMatrix;
use crate::sym_ops::sym_csr_matvec;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Configuration options for the power iteration method
#[derive(Debug, Clone)]
pub struct PowerIterationOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to normalize at each iteration
    pub normalize: bool,
}

impl Default for PowerIterationOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            normalize: true,
        }
    }
}

/// Result of an eigenvalue computation
#[derive(Debug, Clone)]
pub struct EigenResult<T>
where
    T: Float + Debug + Copy,
{
    /// Converged eigenvalues
    pub eigenvalues: Array1<T>,
    /// Corresponding eigenvectors (if requested)
    pub eigenvectors: Option<Array2<T>>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Residual norms for each eigenpair
    pub residuals: Array1<T>,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Computes the largest eigenvalue and corresponding eigenvector of a symmetric
/// matrix using the power iteration method.
///
/// # Arguments
///
/// * `matrix` - The symmetric matrix
/// * `options` - Configuration options
/// * `initial_guess` - Initial guess for the eigenvector (optional)
///
/// # Returns
///
/// Result containing eigenvalue and eigenvector
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_sparse::{
///     sym_csr::SymCsrMatrix,
///     linalg::{power_iteration, PowerIterationOptions},
/// };
///
/// // Create a symmetric matrix
/// let data = vec![2.0, 1.0, 2.0, 3.0];
/// let indices = vec![0, 0, 1, 2];
/// let indptr = vec![0, 1, 3, 4];
/// let matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
///
/// // Configure options
/// let options = PowerIterationOptions {
///     max_iter: 100,
///     tol: 1e-8,
///     normalize: true,
/// };
///
/// // Compute the largest eigenvalue and eigenvector
/// let result = power_iteration(&matrix, &options, None).unwrap();
///
/// // Check the result
/// println!("Eigenvalue: {}", result.eigenvalues[0]);
/// println!("Converged in {} iterations", result.iterations);
/// println!("Final residual: {}", result.residuals[0]);
/// assert!(result.converged);
/// ```
#[allow(dead_code)]
pub fn power_iteration<T>(
    matrix: &SymCsrMatrix<T>,
    options: &PowerIterationOptions,
    initial_guess: Option<ArrayView1<T>>,
) -> SparseResult<EigenResult<T>>
where
    T: Float
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::iter::Sum
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync
        + 'static,
{
    let (n, _) = matrix.shape();

    // Initialize eigenvector
    let mut x = match initial_guess {
        Some(v) => {
            if v.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: v.len(),
                });
            }
            // Create a copy of the initial guess
            let mut x_arr = Array1::zeros(n);
            for i in 0..n {
                x_arr[i] = v[i];
            }
            x_arr
        }
        None => {
            // Random initialization
            let mut x_arr = Array1::zeros(n);
            x_arr[0] = T::one(); // Simple initialization with [1, 0, 0, ...]
            x_arr
        }
    };

    // Normalize the initial vector
    if options.normalize {
        let norm = (x.iter().map(|&v| v * v).sum::<T>()).sqrt();
        if !norm.is_zero() {
            for i in 0..n {
                x[i] = x[i] / norm;
            }
        }
    }

    let mut lambda = T::zero();
    let mut prev_lambda = T::zero();
    let mut converged = false;
    let mut iter = 0;

    // Power iteration loop
    while iter < options.max_iter {
        // Compute matrix-vector product: y = A * x
        let y = sym_csr_matvec(matrix, &x.view())?;

        // Compute Rayleigh quotient: lambda = (x^T * y) / (x^T * x)
        let rayleigh_numerator = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum::<T>();

        if options.normalize {
            lambda = rayleigh_numerator;

            // Check for convergence
            let diff = (lambda - prev_lambda).abs();
            if diff < T::from(options.tol).unwrap() {
                converged = true;
                break;
            }

            // Normalize y to get the next x
            let norm = (y.iter().map(|&v| v * v).sum::<T>()).sqrt();
            if !norm.is_zero() {
                for i in 0..n {
                    x[i] = y[i] / norm;
                }
            }
        } else {
            // If not normalizing at each iteration, just update x
            x = y;

            // Compute eigenvalue estimate
            let norm_x = (x.iter().map(|&v| v * v).sum::<T>()).sqrt();
            if !norm_x.is_zero() {
                lambda = rayleigh_numerator / (norm_x * norm_x);
            }

            // Check for convergence
            let diff = (lambda - prev_lambda).abs();
            if diff < T::from(options.tol).unwrap() {
                converged = true;
                break;
            }
        }

        prev_lambda = lambda;
        iter += 1;
    }

    // Compute final residual: ||Ax - λx||
    let ax = sym_csr_matvec(matrix, &x.view())?;
    let mut residual = Array1::zeros(n);
    for i in 0..n {
        residual[i] = ax[i] - lambda * x[i];
    }
    let residual_norm = (residual.iter().map(|&v| v * v).sum::<T>()).sqrt();

    // Prepare eigenvectors if needed
    let eigenvectors = {
        let mut vecs = Array2::zeros((n, 1));
        for i in 0..n {
            vecs[[i, 0]] = x[i];
        }
        Some(vecs)
    };

    // Prepare the result
    let result = EigenResult {
        eigenvalues: Array1::from_vec(vec![lambda]),
        eigenvectors,
        iterations: iter,
        residuals: Array1::from_vec(vec![residual_norm]),
        converged,
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sym_csr::SymCsrMatrix;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_power_iteration_simple() {
        // Create a simple 2x2 symmetric matrix [[2, 1], [1, 2]]
        // For symmetric matrix, store lower triangle: (0,0)=2, (1,0)=1, (1,1)=2
        let data = vec![2.0, 1.0, 2.0];
        let indices = vec![0, 0, 1]; // Column indices: row 0 has col 0, row 1 has cols 0,1
        let indptr = vec![0, 1, 3]; // Row 0 has 1 element, row 1 has 2 elements
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).unwrap();

        let options = PowerIterationOptions::default();
        let result = power_iteration(&matrix, &options, None).unwrap();

        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 1);
        // The largest eigenvalue of [[2, 1], [1, 2]] is 3.0
        assert!((result.eigenvalues[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_power_iteration_with_initial_guess() {
        // Matrix: [[4, 1], [1, 3]] stored as lower: [[4], [1, 3]]
        let data = vec![4.0, 1.0, 3.0];
        let indptr = vec![0, 1, 3];
        let indices = vec![0, 0, 1];
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).unwrap();

        let initial_guess = Array1::from_vec(vec![1.0, 1.0]);
        let options = PowerIterationOptions::default();
        let result = power_iteration(&matrix, &options, Some(initial_guess.view())).unwrap();

        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 1);
    }

    #[test]
    fn test_power_iteration_convergence() {
        // Matrix: [[5, 2], [2, 4]] stored as lower: [[5], [2, 4]]
        let data = vec![5.0, 2.0, 4.0];
        let indptr = vec![0, 1, 3];
        let indices = vec![0, 0, 1];
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).unwrap();

        let options = PowerIterationOptions {
            max_iter: 50,
            tol: 1e-10,
            normalize: true,
        };
        let result = power_iteration(&matrix, &options, None).unwrap();

        assert!(result.converged);
        assert!(result.iterations <= 50);
        assert!(result.residuals[0] < 1e-4);
    }
}
