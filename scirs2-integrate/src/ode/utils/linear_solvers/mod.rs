//! Linear solvers for ODE systems
//!
//! This module provides linear system solvers for use within ODE solvers.
//! These replace the need for external linear algebra libraries like ndarray-linalg.

use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Enum for different types of linear solvers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearSolverType {
    /// Direct solver using LU decomposition
    Direct,
    /// Iterative solver (GMRES, etc.)
    Iterative,
    /// Automatic selection based on problem size
    Auto,
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting
///
/// # Arguments
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
///
/// # Returns
/// * `Result<Array1<F>, IntegrateError>` - The solution vector x
pub fn solve_linear_system<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign,
{
    // Get dimensions
    let n = a.shape()[0];

    // Check that A is square
    if a.shape()[0] != a.shape()[1] {
        return Err(IntegrateError::ValueError(format!(
            "Matrix must be square to solve linear system, got shape {:?}",
            a.shape()
        )));
    }

    // Check that b has compatible dimensions
    if b.len() != n {
        return Err(IntegrateError::ValueError(
            format!("Right-hand side vector dimensions incompatible with matrix: matrix has {} rows but vector has {} elements", 
                n, b.len())
        ));
    }

    // Create copies of A and b that we can modify
    let mut a_copy = a.to_owned();
    let mut b_copy = b.to_owned();

    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut pivot_idx = k;
        let mut max_val = a_copy[[k, k]].abs();

        for i in (k + 1)..n {
            let val = a_copy[[i, k]].abs();
            if val > max_val {
                max_val = val;
                pivot_idx = i;
            }
        }

        // Check for singularity
        if max_val < F::from_f64(1e-14).unwrap() {
            return Err(IntegrateError::ValueError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if necessary
        if pivot_idx != k {
            // Swap rows in A
            for j in k..n {
                let temp = a_copy[[k, j]];
                a_copy[[k, j]] = a_copy[[pivot_idx, j]];
                a_copy[[pivot_idx, j]] = temp;
            }

            // Swap elements in b
            let temp = b_copy[k];
            b_copy[k] = b_copy[pivot_idx];
            b_copy[pivot_idx] = temp;
        }

        // Eliminate below the pivot
        for i in (k + 1)..n {
            let factor = a_copy[[i, k]] / a_copy[[k, k]];

            // Update the right-hand side
            b_copy[i] = b_copy[i] - factor * b_copy[k];

            // Update the matrix
            a_copy[[i, k]] = F::zero(); // Explicitly set to zero to avoid numerical issues

            for j in (k + 1)..n {
                a_copy[[i, j]] = a_copy[[i, j]] - factor * a_copy[[k, j]];
            }
        }
    }

    // Back-substitution
    let mut x = Array1::<F>::zeros(n);

    for i in (0..n).rev() {
        let mut sum = b_copy[i];

        for j in (i + 1)..n {
            sum -= a_copy[[i, j]] * x[j];
        }

        x[i] = sum / a_copy[[i, i]];
    }

    Ok(x)
}

/// Compute the norm of a vector
///
/// # Arguments
/// * `v` - The vector
///
/// # Returns
/// * The L2 norm of the vector
pub fn vector_norm<F>(v: &ArrayView1<F>) -> F
where
    F: Float,
{
    let mut sum = F::zero();
    for &val in v.iter() {
        sum = sum + val * val;
    }
    sum.sqrt()
}

/// Compute the Frobenius norm of a matrix
///
/// # Arguments
/// * `m` - The matrix
///
/// # Returns
/// * The Frobenius norm of the matrix
pub fn matrix_norm<F>(m: &ArrayView2<F>) -> F
where
    F: Float,
{
    let mut sum = F::zero();
    for val in m.iter() {
        sum = sum + (*val) * (*val);
    }
    sum.sqrt()
}

/// Solve a linear system using automatic method selection
pub fn auto_solve_linear_system<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    solver_type: LinearSolverType,
) -> IntegrateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign,
{
    match solver_type {
        LinearSolverType::Direct => solve_linear_system(a, b),
        LinearSolverType::Iterative => {
            // For now, use direct solver as iterative is not implemented
            solve_linear_system(a, b)
        }
        LinearSolverType::Auto => {
            // Use direct solver for small problems, iterative for large
            let n = a.shape()[0];
            if n < 100 {
                solve_linear_system(a, b)
            } else {
                // For now, use direct solver until iterative is implemented
                solve_linear_system(a, b)
            }
        }
    }
}

/// Solve a linear system using LU decomposition (alias for compatibility)
pub fn solve_lu<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign,
{
    solve_linear_system(a, b)
}
