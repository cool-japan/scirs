//! Linear solvers for ODE systems
//!
//! This module provides linear system solvers for use within ODE solvers.
//! These replace the need for external linear algebra libraries like ndarray-linalg.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
        + std::ops::MulAssign
        + std::default::Default
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + std::ops::DivAssign,
{
    match solver_type {
        LinearSolverType::Direct => solve_linear_system(a, b),
        LinearSolverType::Iterative => {
            // Use GMRES iterative solver
            solve_gmres(a, b, None, None, None)
        }
        LinearSolverType::Auto => {
            // Use direct solver for small problems, iterative for large
            let n = a.shape()[0];
            if n < 100 {
                solve_linear_system(a, b)
            } else {
                // Use GMRES for large systems
                solve_gmres(a, b, None, None, None)
            }
        }
    }
}

/// Solve a linear system using LU decomposition (alias for compatibility)
#[allow(dead_code)]
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

/// Solve a linear system using GMRES (Generalized Minimal Residual) method
///
/// GMRES is a robust iterative method for solving general linear systems.
///
/// # Arguments
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
/// * `max_iter` - Maximum number of iterations (default: min(n, 50))
/// * `tol` - Convergence tolerance (default: 1e-10)
/// * `restart` - Restart parameter for GMRES(m) (default: min(n, 20))
///
/// # Returns
/// * `Result<Array1<F>, IntegrateError>` - The solution vector x
#[allow(dead_code)]
pub fn solve_gmres<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
    restart: Option<usize>,
) -> IntegrateResult<Array1<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + Default
        + std::iter::Sum
        + scirs2_core::ndarray::ScalarOperand
        + std::ops::DivAssign,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(IntegrateError::ValueError(
            "Matrix must be square".to_string(),
        ));
    }
    if n != b.len() {
        return Err(IntegrateError::ValueError(
            "Matrix and vector dimensions must match".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(std::cmp::min(n, 50));
    let tol = tol.unwrap_or_else(|| F::from_f64(1e-10).unwrap());
    let restart = restart.unwrap_or(std::cmp::min(n, 20));

    // Initial guess: zero vector
    let mut x = Array1::<F>::zeros(n);

    // Compute initial residual: r0 = b - A*x0
    let mut r = b.to_owned();
    for i in 0..n {
        let mut ax_i = F::zero();
        for j in 0..n {
            ax_i += a[[i, j]] * x[j];
        }
        r[i] -= ax_i;
    }

    let initial_norm = (r.iter().map(|&x| x * x).sum::<F>()).sqrt();
    if initial_norm < tol {
        return Ok(x); // Already converged
    }

    let mut outer_iter = 0;
    while outer_iter < max_iter {
        // GMRES restart cycle
        let m = std::cmp::min(restart, max_iter - outer_iter);

        // Normalize r to get v1
        let beta = (r.iter().map(|&x| x * x).sum::<F>()).sqrt();
        if beta < tol {
            break; // Converged
        }

        let mut v = vec![Array1::<F>::zeros(n); m + 1];
        v[0] = &r / beta;

        let mut h = vec![vec![F::zero(); m]; m + 1];
        let mut g = vec![F::zero(); m + 1];
        g[0] = beta;

        let mut j = 0;
        while j < m {
            // Compute w = A * v[j]
            let mut w = Array1::<F>::zeros(n);
            for i in 0..n {
                for k in 0..n {
                    w[i] += a[[i, k]] * v[j][k];
                }
            }

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = v[i].dot(&w);
                for k in 0..n {
                    w[k] -= h[i][j] * v[i][k];
                }
            }

            h[j + 1][j] = (w.iter().map(|&x| x * x).sum::<F>()).sqrt();

            if h[j + 1][j] < F::from_f64(1e-14).unwrap() {
                // Linear dependence, stop early
                break;
            }

            v[j + 1] = &w / h[j + 1][j];

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let c = if i < g.len() - 1 {
                    h[i][j] / (h[i][j] * h[i][j] + h[i + 1][j] * h[i + 1][j]).sqrt()
                } else {
                    F::one()
                };
                let s = if i < g.len() - 1 {
                    h[i + 1][j] / (h[i][j] * h[i][j] + h[i + 1][j] * h[i + 1][j]).sqrt()
                } else {
                    F::zero()
                };

                let temp = c * h[i][j] + s * h[i + 1][j];
                h[i + 1][j] = -s * h[i][j] + c * h[i + 1][j];
                h[i][j] = temp;
            }

            // Compute new Givens rotation
            let c = h[j][j] / (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            let s = h[j + 1][j] / (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();

            // Apply new Givens rotation
            h[j][j] = c * h[j][j] + s * h[j + 1][j];
            h[j + 1][j] = F::zero();

            let temp = c * g[j];
            g[j + 1] = -s * g[j];
            g[j] = temp;

            // Check convergence
            if g[j + 1].abs() < tol * initial_norm {
                j += 1;
                break;
            }

            j += 1;
        }

        // Solve upper triangular system H*y = g
        let mut y = vec![F::zero(); j];
        for i in (0..j).rev() {
            let mut sum = g[i];
            for k in (i + 1)..j {
                sum -= h[i][k] * y[k];
            }
            y[i] = sum / h[i][i];
        }

        // Update solution: x = x + V*y
        for i in 0..n {
            for k in 0..j {
                x[i] += y[k] * v[k][i];
            }
        }

        // Compute new residual
        r = b.to_owned();
        for i in 0..n {
            let mut ax_i = F::zero();
            for k in 0..n {
                ax_i += a[[i, k]] * x[k];
            }
            r[i] -= ax_i;
        }

        let residual_norm = (r.iter().map(|&x| x * x).sum::<F>()).sqrt();
        if residual_norm < tol * initial_norm {
            break; // Converged
        }

        outer_iter += m;
    }

    Ok(x)
}
