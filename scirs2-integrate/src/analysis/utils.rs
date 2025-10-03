//! Utility functions for dynamical systems analysis
//!
//! This module contains helper functions and utilities used across
//! the analysis modules.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;

/// Compute determinant of a square matrix using LU decomposition
pub fn compute_determinant(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return 0.0; // Not square
    }

    let mut lu = matrix.clone();
    let mut determinant = 1.0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_idx = k;

        for i in (k + 1)..n {
            if lu[[i, k]].abs() > max_val {
                max_val = lu[[i, k]].abs();
                max_idx = i;
            }
        }

        // Swap rows if needed
        if max_idx != k {
            for j in 0..n {
                let temp = lu[[k, j]];
                lu[[k, j]] = lu[[max_idx, j]];
                lu[[max_idx, j]] = temp;
            }
            determinant *= -1.0; // Row swap changes sign
        }

        // Check for singular matrix
        if lu[[k, k]].abs() < 1e-14 {
            return 0.0;
        }

        determinant *= lu[[k, k]];

        // Eliminate
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / lu[[k, k]];
            for j in (k + 1)..n {
                lu[[i, j]] -= factor * lu[[k, j]];
            }
        }
    }

    determinant
}

/// Compute trace of a matrix (sum of diagonal elements)
pub fn compute_trace(matrix: &Array2<f64>) -> f64 {
    let n = std::cmp::min(matrix.nrows(), matrix.ncols());
    (0..n).map(|i| matrix[[i, i]]).sum()
}

/// Estimate the rank of a matrix using QR-like decomposition
pub fn estimate_matrix_rank(matrix: &Array2<f64>, tolerance: f64) -> usize {
    // Simplified rank estimation using QR decomposition
    let (m, n) = matrix.dim();
    let mut a = matrix.clone();
    let mut rank = 0;

    for k in 0..std::cmp::min(m, n) {
        // Find the column with maximum norm
        let mut max_norm = 0.0;
        let mut max_col = k;

        for j in k..n {
            let col_norm: f64 = (k..m).map(|i| a[[i, j]].powi(2)).sum::<f64>().sqrt();
            if col_norm > max_norm {
                max_norm = col_norm;
                max_col = j;
            }
        }

        // If maximum norm is below tolerance, we've found the rank
        if max_norm < tolerance {
            break;
        }

        // Swap columns
        if max_col != k {
            for i in 0..m {
                let temp = a[[i, k]];
                a[[i, k]] = a[[i, max_col]];
                a[[i, max_col]] = temp;
            }
        }

        rank += 1;

        // Normalize and orthogonalize
        for i in k..m {
            a[[i, k]] /= max_norm;
        }

        for j in (k + 1)..n {
            let dot_product: f64 = (k..m).map(|i| a[[i, k]] * a[[i, j]]).sum();
            for i in k..m {
                a[[i, j]] -= dot_product * a[[i, k]];
            }
        }
    }

    rank
}

/// Solve linear system Ax = b using LU decomposition with partial pivoting
pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> IntegrateResult<Array1<f64>> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut x = b.clone();

    // LU decomposition with partial pivoting
    let mut pivot = Array1::<usize>::zeros(n);
    for i in 0..n {
        pivot[i] = i;
    }

    for k in 0..n - 1 {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_idx = k;

        for i in k + 1..n {
            if lu[[i, k]].abs() > max_val {
                max_val = lu[[i, k]].abs();
                max_idx = i;
            }
        }

        // Swap rows
        if max_idx != k {
            for j in 0..n {
                let temp = lu[[k, j]];
                lu[[k, j]] = lu[[max_idx, j]];
                lu[[max_idx, j]] = temp;
            }
            pivot.swap(k, max_idx);
        }

        // Eliminate
        for i in k + 1..n {
            if lu[[k, k]].abs() < 1e-14 {
                return Err(IntegrateError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            let factor = lu[[i, k]] / lu[[k, k]];
            lu[[i, k]] = factor;

            for j in k + 1..n {
                lu[[i, j]] -= factor * lu[[k, j]];
            }
        }
    }

    // Apply row swaps to RHS
    for k in 0..n - 1 {
        x.swap(k, pivot[k]);
    }

    // Forward substitution
    for i in 1..n {
        for j in 0..i {
            x[i] -= lu[[i, j]] * x[j];
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lu[[i, j]] * x[j];
        }
        // Check for zero diagonal element
        if lu[[i, i]].abs() < 1e-14 {
            return Err(IntegrateError::ComputationError(
                "Zero diagonal element in back substitution".to_string(),
            ));
        }
        x[i] /= lu[[i, i]];
    }

    Ok(x)
}

/// Check if point is a bifurcation point based on eigenvalues crossing imaginary axis
pub fn is_bifurcation_point(eigenvalues: &[Complex64]) -> bool {
    // Check for eigenvalues crossing the imaginary axis
    eigenvalues.iter().any(|&eig| eig.re.abs() < 1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_compute_determinant() {
        // Test 2x2 matrix
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let det = compute_determinant(&matrix);
        assert_abs_diff_eq!(det, -2.0, epsilon = 1e-10);

        // Test 3x3 identity matrix
        let identity = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let det = compute_determinant(&identity);
        assert_abs_diff_eq!(det, 1.0, epsilon = 1e-10);

        // Test singular matrix
        let singular = array![[1.0, 2.0], [2.0, 4.0]];
        let det = compute_determinant(&singular);
        assert_abs_diff_eq!(det, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_trace() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let trace = compute_trace(&matrix);
        assert_abs_diff_eq!(trace, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_matrix_rank() {
        // Full rank matrix
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let rank = estimate_matrix_rank(&matrix, 1e-10);
        assert_eq!(rank, 2);

        // Rank-deficient matrix
        let matrix = array![[1.0, 2.0], [2.0, 4.0]];
        let rank = estimate_matrix_rank(&matrix, 1e-10);
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_solve_linear_system() {
        // Simple 2x2 system
        let a = array![[2.0, 1.0], [1.0, 1.0]];
        let b = array![3.0, 2.0];
        let result = solve_linear_system(&a, &b).unwrap();
        let expected = array![1.0, 1.0];

        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_is_bifurcation_point() {
        use scirs2_core::numeric::Complex64;

        // Eigenvalue on imaginary axis (bifurcation)
        let eigenvalues = vec![Complex64::new(0.0, 1.0), Complex64::new(-1.0, 0.0)];
        assert!(is_bifurcation_point(&eigenvalues));

        // All eigenvalues away from imaginary axis (no bifurcation)
        let eigenvalues = vec![Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0)];
        assert!(!is_bifurcation_point(&eigenvalues));
    }
}
