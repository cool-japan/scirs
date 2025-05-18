//! Utility functions for DAE solvers
//!
//! This module provides utility functions for use in DAE solvers.

use crate::IntegrateFloat;
use ndarray::{Array2, ArrayView2};
// use num_traits::{Float, FromPrimitive};

// Linear solvers
pub mod linear_solvers;

// Re-export useful utilities
pub use linear_solvers::{matrix_norm, solve_linear_system, solve_lu, vector_norm};

/// Compute the constraint Jacobian for a constraint function
pub fn compute_constraint_jacobian<F: IntegrateFloat>(
    g: &impl Fn(F, &[F], &[F]) -> Vec<F>,
    t: F,
    x: &[F],
    y: &[F],
) -> Array2<F> {
    let n = x.len();
    let m = y.len();
    let epsilon = F::from_f64(1e-8).unwrap();

    // Compute g at the current point
    let g0 = g(t, x, y);
    let ng = g0.len();

    let mut jacobian = Array2::zeros((ng, n + m));

    // Compute partial derivatives with respect to x
    for i in 0..n {
        let mut x_perturbed = x.to_vec();
        x_perturbed[i] += epsilon;
        let g_perturbed = g(t, &x_perturbed, y);

        for j in 0..ng {
            jacobian[(j, i)] = (g_perturbed[j] - g0[j]) / epsilon;
        }
    }

    // Compute partial derivatives with respect to y
    for i in 0..m {
        let mut y_perturbed = y.to_vec();
        y_perturbed[i] += epsilon;
        let g_perturbed = g(t, x, &y_perturbed);

        for j in 0..ng {
            jacobian[(j, n + i)] = (g_perturbed[j] - g0[j]) / epsilon;
        }
    }

    jacobian
}

/// Check if a matrix is singular
pub fn is_singular_matrix<F: IntegrateFloat>(matrix: ArrayView2<F>) -> bool {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return true; // Non-square matrices are considered singular
    }

    // Use a simple determinant check for small matrices
    if n == 1 {
        return matrix[(0, 0)].abs() < F::from_f64(1e-10).unwrap();
    }

    // For larger matrices, check condition number or use LU decomposition
    // For now, use a simple diagonal dominance check as a heuristic
    let epsilon = F::from_f64(1e-10).unwrap();

    for i in 0..n {
        let diagonal = matrix[(i, i)].abs();
        let mut off_diagonal_sum = F::zero();

        for j in 0..n {
            if i != j {
                off_diagonal_sum += matrix[(i, j)].abs();
            }
        }

        if diagonal < epsilon || diagonal < off_diagonal_sum * F::from_f64(0.1).unwrap() {
            return true;
        }
    }

    false
}
