//! Linear solvers for DAE systems
//!
//! This module provides linear system solvers for use within DAE solvers.
//! These replace the need for external linear algebra libraries like ndarray-linalg.

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
// use num_traits::{Float, FromPrimitive};

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
    F: IntegrateFloat,
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

/// Solve a linear system Ax = b using LU decomposition
///
/// # Arguments
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
///
/// # Returns
/// * `Result<Array1<F>, IntegrateError>` - The solution vector x
pub fn solve_lu<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
{
    // For small systems, just use Gaussian elimination
    if a.shape()[0] <= 10 {
        return solve_linear_system(a, b);
    }

    // Get dimensions
    let n = a.shape()[0];

    // Create copies of A that we can modify
    let mut a_copy = a.to_owned();

    // Arrays to store the LU decomposition
    let mut l = Array2::<F>::eye(n);
    let mut u = Array2::<F>::zeros((n, n));

    // Array to store permutation
    let mut p = vec![0; n];
    for (i, p_elem) in p.iter_mut().enumerate().take(n) {
        *p_elem = i;
    }

    // Perform LU decomposition with partial pivoting
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
            for j in 0..n {
                let temp = a_copy[[k, j]];
                a_copy[[k, j]] = a_copy[[pivot_idx, j]];
                a_copy[[pivot_idx, j]] = temp;
            }

            // Update permutation
            p.swap(k, pivot_idx);

            // If k > 0, swap rows in L for columns 0 to k-1
            if k > 0 {
                for j in 0..k {
                    let temp = l[[k, j]];
                    l[[k, j]] = l[[pivot_idx, j]];
                    l[[pivot_idx, j]] = temp;
                }
            }
        }

        // Compute elements of U
        for j in k..n {
            u[[k, j]] = a_copy[[k, j]];
            for p in 0..k {
                u[[k, j]] = u[[k, j]] - l[[k, p]] * u[[p, j]];
            }
        }

        // Compute elements of L
        for i in (k + 1)..n {
            if u[[k, k]].abs() < F::from_f64(1e-14).unwrap() {
                return Err(IntegrateError::ValueError(
                    "LU decomposition failed: division by zero".to_string(),
                ));
            }

            l[[i, k]] = a_copy[[i, k]];
            for p in 0..k {
                l[[i, k]] = l[[i, k]] - l[[i, p]] * u[[p, k]];
            }
            l[[i, k]] /= u[[k, k]];
        }
    }

    // Solve Ly = Pb
    let mut y = Array1::<F>::zeros(n);
    let mut pb = Array1::<F>::zeros(n);

    // Permute b
    for i in 0..n {
        pb[i] = b[p[i]];
    }

    // Forward substitution
    for i in 0..n {
        y[i] = pb[i];
        for j in 0..i {
            y[i] = y[i] - l[[i, j]] * y[j];
        }
    }

    // Solve Ux = y
    let mut x = Array1::<F>::zeros(n);

    // Back substitution
    for i in (0..n).rev() {
        if u[[i, i]].abs() < F::from_f64(1e-14).unwrap() {
            return Err(IntegrateError::ValueError(
                "LU decomposition: singular matrix detected during back substitution".to_string(),
            ));
        }

        x[i] = y[i];
        for j in (i + 1)..n {
            x[i] = x[i] - u[[i, j]] * x[j];
        }
        x[i] /= u[[i, i]];
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
    F: IntegrateFloat,
{
    let mut sum = F::zero();
    for &val in v.iter() {
        sum += val * val;
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
    F: IntegrateFloat,
{
    let mut sum = F::zero();
    for val in m.iter() {
        sum += (*val) * (*val);
    }
    sum.sqrt()
}
