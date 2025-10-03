//! Least Squares QR (LSQR) method for sparse linear systems
//!
//! LSQR is an iterative method for solving sparse least squares problems
//! and sparse linear systems. It can handle both overdetermined and
//! underdetermined systems.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Options for the LSQR solver
#[derive(Debug, Clone)]
pub struct LSQROptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance for the residual
    pub atol: f64,
    /// Convergence tolerance for the solution
    pub btol: f64,
    /// Condition number limit
    pub conlim: f64,
    /// Whether to compute standard errors
    pub calc_var: bool,
    /// Whether to store residual history
    pub store_residual_history: bool,
}

impl Default for LSQROptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            atol: 1e-8,
            btol: 1e-8,
            conlim: 1e8,
            calc_var: false,
            store_residual_history: true,
        }
    }
}

/// Result from LSQR solver
#[derive(Debug, Clone)]
pub struct LSQRResult<T> {
    /// Solution vector
    pub x: Array1<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ||Ax - b||
    pub residualnorm: T,
    /// Final solution norm ||x||
    pub solution_norm: T,
    /// Condition number estimate
    pub condition_number: T,
    /// Whether the solver converged
    pub converged: bool,
    /// Standard errors (if requested)
    pub standard_errors: Option<Array1<T>>,
    /// Residual history (if requested)
    pub residual_history: Option<Vec<T>>,
    /// Convergence reason
    pub convergence_reason: String,
}

/// LSQR algorithm for sparse least squares problems
///
/// Solves the least squares problem min ||Ax - b||_2 or the linear system Ax = b.
/// The method is based on the bidiagonalization of A.
///
/// # Arguments
///
/// * `matrix` - The coefficient matrix A (m x n)
/// * `b` - The right-hand side vector (length m)
/// * `x0` - Initial guess (optional, length n)
/// * `options` - Solver options
///
/// # Returns
///
/// An `LSQRResult` containing the solution and convergence information
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::{lsqr, LSQROptions};
/// use scirs2_core::ndarray::Array1;
///
/// // Create an overdetermined system
/// let rows = vec![0, 0, 1, 1, 2, 2];
/// let cols = vec![0, 1, 0, 1, 0, 1];
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).unwrap();
///
/// // Right-hand side
/// let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Solve using LSQR
/// let result = lsqr(&matrix, &b.view(), None, LSQROptions::default()).unwrap();
/// ```
#[allow(dead_code)]
pub fn lsqr<T, S>(
    matrix: &S,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    options: LSQROptions,
) -> SparseResult<LSQRResult<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();

    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    // Initialize solution vector
    let mut x = match x0 {
        Some(x0_val) => {
            if x0_val.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0_val.len(),
                });
            }
            x0_val.to_owned()
        }
        None => Array1::zeros(n),
    };

    // Compute initial residual
    let ax = matrix_vector_multiply(matrix, &x.view())?;
    let mut u = b - &ax;
    let beta = l2_norm(&u.view());

    if beta > T::zero() {
        for i in 0..m {
            u[i] = u[i] / beta;
        }
    }

    // Initialize variables
    let mut v = matrix_transpose_vector_multiply(matrix, &u.view())?;
    let mut alpha = l2_norm(&v.view());

    if alpha > T::zero() {
        for i in 0..n {
            v[i] = v[i] / alpha;
        }
    }

    let mut w = v.clone();
    let mut x_norm = T::zero();
    let mut dd_norm = T::zero();
    let mut res2 = beta;

    // Variables for QR factorization of bidiagonal matrix
    let mut rho_bar = alpha;
    let mut phi_bar = beta;

    // Tolerances
    let atol = T::from(options.atol).unwrap();
    let btol = T::from(options.btol).unwrap();
    let conlim = T::from(options.conlim).unwrap();

    let mut residual_history = if options.store_residual_history {
        Some(vec![beta])
    } else {
        None
    };

    let mut converged = false;
    let mut convergence_reason = String::new();
    let mut iter = 0;

    for k in 0..options.max_iter {
        iter = k + 1;

        // Bidiagonalization step: u := A*v - alpha*u
        let av = matrix_vector_multiply(matrix, &v.view())?;
        for i in 0..m {
            u[i] = av[i] - alpha * u[i];
        }
        let beta_new = l2_norm(&u.view());

        if beta_new > T::zero() {
            for i in 0..m {
                u[i] = u[i] / beta_new;
            }
        }

        // v := A^T*u - beta_new*v
        let atu = matrix_transpose_vector_multiply(matrix, &u.view())?;
        for i in 0..n {
            v[i] = atu[i] - beta_new * v[i];
        }
        let alpha_new = l2_norm(&v.view());

        if alpha_new > T::zero() {
            for i in 0..n {
                v[i] = v[i] / alpha_new;
            }
        }

        // QR factorization of the bidiagonal matrix
        let rho = (rho_bar * rho_bar + beta_new * beta_new).sqrt();
        let c = rho_bar / rho;
        let s = beta_new / rho;
        let theta = s * alpha_new;
        let rho_bar_new = -c * alpha_new;
        let phi = c * phi_bar;
        let phi_bar_new = s * phi_bar;

        // Update solution
        for i in 0..n {
            x[i] = x[i] + (phi / rho) * w[i];
            w[i] = v[i] - (theta / rho) * w[i];
        }

        // Update norms and residual estimate
        x_norm = (x_norm * x_norm + (phi / rho) * (phi / rho)).sqrt();
        dd_norm = dd_norm + (T::one() / rho) * (T::one() / rho);
        res2 = phi_bar_new.abs();

        if let Some(ref mut history) = residual_history {
            history.push(res2);
        }

        // Check convergence
        let r1_norm = res2;
        let r2_norm = if x_norm > T::zero() {
            alpha_new.abs() * x_norm
        } else {
            alpha_new.abs()
        };

        let test1 = r1_norm / (atol + btol * beta);
        let test2 = if x_norm > T::zero() {
            alpha_new.abs() / (atol + btol * x_norm)
        } else {
            alpha_new.abs() / atol
        };
        let test3 = T::one() / conlim;

        if test1 <= T::one() {
            converged = true;
            convergence_reason = "Residual tolerance satisfied".to_string();
            break;
        }

        if test2 <= T::one() {
            converged = true;
            convergence_reason = "Solution tolerance satisfied".to_string();
            break;
        }

        // Condition number estimate should be compared to limit, not x_norm to test3
        let condition_estimate = if dd_norm > T::zero() {
            x_norm / dd_norm.sqrt()
        } else {
            T::one()
        };

        if condition_estimate > conlim {
            converged = true;
            convergence_reason = "Condition number limit reached".to_string();
            break;
        }

        // Update for next iteration
        alpha = alpha_new;
        rho_bar = rho_bar_new;
        phi_bar = phi_bar_new;
    }

    if !converged {
        convergence_reason = "Maximum iterations reached".to_string();
    }

    // Compute final metrics
    let ax_final = matrix_vector_multiply(matrix, &x.view())?;
    let final_residual = b - &ax_final;
    let final_residualnorm = l2_norm(&final_residual.view());
    let final_solution_norm = l2_norm(&x.view());

    // Estimate condition number (simplified)
    let condition_number = if dd_norm > T::zero() {
        x_norm / dd_norm.sqrt()
    } else {
        T::one()
    };

    // Compute standard errors if requested
    let standard_errors = if options.calc_var {
        Some(compute_standard_errors(matrix, final_residualnorm, n)?)
    } else {
        None
    };

    Ok(LSQRResult {
        x,
        iterations: iter,
        residualnorm: final_residualnorm,
        solution_norm: final_solution_norm,
        condition_number,
        converged,
        standard_errors,
        residual_history,
        convergence_reason,
    })
}

/// Helper function for matrix-vector multiplication
#[allow(dead_code)]
fn matrix_vector_multiply<T, S>(matrix: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();
    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(rows);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[i] = result[i] + values[k] * x[j];
    }

    Ok(result)
}

/// Helper function for matrix transpose-vector multiplication
#[allow(dead_code)]
fn matrix_transpose_vector_multiply<T, S>(matrix: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();
    if x.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(cols);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[j] = result[j] + values[k] * x[i];
    }

    Ok(result)
}

/// Compute L2 norm of a vector
#[allow(dead_code)]
fn l2_norm<T>(x: &ArrayView1<T>) -> T
where
    T: Float + Debug + Copy,
{
    (x.iter().map(|&val| val * val).fold(T::zero(), |a, b| a + b)).sqrt()
}

/// Compute standard errors (simplified implementation)
#[allow(dead_code)]
fn compute_standard_errors<T, S>(matrix: &S, residualnorm: T, n: usize) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (m, _) = matrix.shape();

    // Simplified standard error computation
    // In practice, this should use the diagonal of (A^T A)^(-1)
    let variance = if m > n {
        residualnorm * residualnorm / T::from(m - n).unwrap()
    } else {
        residualnorm * residualnorm
    };

    let std_err = variance.sqrt();
    Ok(Array1::from_elem(n, std_err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    fn test_lsqr_square_system() {
        // Create a simple 3x3 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let result = lsqr(&matrix, &b.view(), None, LSQROptions::default()).unwrap();

        assert!(result.converged);

        // Verify solution by computing residual
        let ax = matrix_vector_multiply(&matrix, &result.x.view()).unwrap();
        let residual = &b - &ax;
        let residualnorm = l2_norm(&residual.view());

        assert!(residualnorm < 1e-6);
    }

    #[test]
    fn test_lsqr_overdetermined_system() {
        // Create an overdetermined 3x2 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).unwrap();

        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = lsqr(&matrix, &b.view(), None, LSQROptions::default()).unwrap();

        assert!(result.converged);
        assert_eq!(result.x.len(), 2);

        // For overdetermined systems, check that we get a reasonable least squares solution
        assert!(result.residualnorm < 2.0); // Should be a reasonable fit
    }

    #[test]
    fn test_lsqr_diagonal_system() {
        // Create a diagonal system
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![4.0, 9.0, 16.0]);
        let result = lsqr(&matrix, &b.view(), None, LSQROptions::default()).unwrap();

        assert!(result.converged);

        // For diagonal system, solution should be [2, 3, 4]
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[2], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lsqr_with_initial_guess() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![5.0, 6.0, 7.0]);
        let x0 = Array1::from_vec(vec![4.0, 5.0, 6.0]); // Close to solution

        let result = lsqr(&matrix, &b.view(), Some(&x0.view()), LSQROptions::default()).unwrap();

        assert!(result.converged);
        assert!(result.iterations <= 5); // Should converge quickly with good initial guess
    }

    #[test]
    fn test_lsqr_standard_errors() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let options = LSQROptions {
            calc_var: true,
            ..Default::default()
        };

        let result = lsqr(&matrix, &b.view(), None, options).unwrap();

        assert!(result.converged);
        assert!(result.standard_errors.is_some());

        let std_errs = result.standard_errors.unwrap();
        assert_eq!(std_errs.len(), 3);
    }
}
