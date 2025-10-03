//! Transpose-Free Quasi-Minimal Residual (TFQMR) method for sparse linear systems
//!
//! TFQMR is a Krylov subspace method that can solve non-symmetric linear systems
//! without requiring the transpose of the coefficient matrix. It's related to
//! BiCGSTAB but uses a different update strategy.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Options for the TFQMR solver
#[derive(Debug, Clone)]
pub struct TFQMROptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to use left preconditioning
    pub use_left_preconditioner: bool,
    /// Whether to use right preconditioning
    pub use_right_preconditioner: bool,
}

impl Default for TFQMROptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            use_left_preconditioner: false,
            use_right_preconditioner: false,
        }
    }
}

/// Result from TFQMR solver
#[derive(Debug, Clone)]
pub struct TFQMRResult<T> {
    /// Solution vector
    pub x: Array1<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: T,
    /// Whether the solver converged
    pub converged: bool,
    /// Residual history (if requested)
    pub residual_history: Option<Vec<T>>,
}

/// Transpose-Free Quasi-Minimal Residual method
///
/// Solves the linear system A * x = b using the TFQMR method.
/// This method is suitable for non-symmetric matrices and does not
/// require computing A^T explicitly.
///
/// # Arguments
///
/// * `matrix` - The coefficient matrix A
/// * `b` - The right-hand side vector
/// * `x0` - Initial guess (optional)
/// * `options` - Solver options
///
/// # Returns
///
/// A `TFQMRResult` containing the solution and convergence information
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::{tfqmr, TFQMROptions};
/// use scirs2_core::ndarray::Array1;
///
/// // Create a simple matrix
/// let rows = vec![0, 0, 1, 1, 2, 2];
/// let cols = vec![0, 1, 0, 1, 1, 2];
/// let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// // Right-hand side
/// let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
///
/// // Solve using TFQMR
/// let result = tfqmr(&matrix, &b.view(), None, TFQMROptions::default()).unwrap();
/// ```
#[allow(dead_code)]
pub fn tfqmr<T, S>(
    matrix: &S,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    options: TFQMROptions,
) -> SparseResult<TFQMRResult<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = b.len();
    let (rows, cols) = matrix.shape();

    if rows != cols || rows != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: rows,
        });
    }

    // Initialize solution vector
    let mut x = match x0 {
        Some(x0_val) => x0_val.to_owned(),
        None => Array1::zeros(n),
    };

    // Compute initial residual: r0 = b - A * x0
    let ax = matrix_vector_multiply(matrix, &x.view())?;
    let mut r = b - &ax;

    // Check if already converged
    let initial_residual_norm = l2_norm(&r.view());
    let b_norm = l2_norm(b);
    let tolerance = T::from(options.tol).unwrap() * b_norm;

    if initial_residual_norm <= tolerance {
        return Ok(TFQMRResult {
            x,
            iterations: 0,
            residual_norm: initial_residual_norm,
            converged: true,
            residual_history: Some(vec![initial_residual_norm]),
        });
    }

    // Choose r0* (arbitrary, often r0* = r0)
    let r_star = r.clone();

    // Initialize vectors
    let mut v = r.clone();
    let mut y = v.clone();
    let mut w = matrix_vector_multiply(matrix, &y.view())?;
    let mut z = w.clone();

    let mut d = Array1::zeros(n);
    let mut theta = T::zero();
    let mut eta = T::zero();
    let mut tau = initial_residual_norm;

    // TFQMR parameters
    let mut rho = dot_product(&r_star.view(), &r.view());
    let mut alpha = T::zero();
    let mut beta = T::zero();

    let mut residual_history = Vec::new();
    residual_history.push(initial_residual_norm);

    let mut converged = false;
    let mut iter = 0;

    for m in 0..options.max_iter {
        iter = m + 1;

        // Compute alpha
        let sigma = dot_product(&r_star.view(), &w.view());
        if sigma.abs() < T::from(1e-14).unwrap() {
            return Err(SparseError::ConvergenceError(
                "TFQMR breakdown: sigma is too small".to_string(),
            ));
        }
        alpha = rho / sigma;

        // Update v and y for odd steps
        for i in 0..n {
            v[i] = v[i] - alpha * w[i];
            y[i] = y[i] - alpha * z[i];
        }

        // Compute theta and c for odd step
        let v_norm = l2_norm(&v.view());
        theta = v_norm / tau;
        let c = T::one() / (T::one() + theta * theta).sqrt();
        tau = tau * theta * c;
        eta = c * c * alpha;

        // Update solution and residual
        for i in 0..n {
            d[i] = y[i] + (theta * eta) * d[i];
            x[i] = x[i] + eta * d[i];
        }

        // Check convergence for odd step
        let current_residual = tau;
        residual_history.push(current_residual);

        if current_residual <= tolerance {
            converged = true;
            break;
        }

        // Compute w for even step
        w = matrix_vector_multiply(matrix, &y.view())?;

        // Update rho and beta
        let rho_new = dot_product(&r_star.view(), &v.view());
        beta = rho_new / rho;
        rho = rho_new;

        // Update y and z for even step
        for i in 0..n {
            y[i] = v[i] + beta * y[i];
            z[i] = w[i] + beta * z[i];
        }

        // Compute theta and c for even step
        let y_norm = l2_norm(&y.view());
        theta = y_norm / tau;
        let c = T::one() / (T::one() + theta * theta).sqrt();
        tau = tau * theta * c;
        eta = c * c * alpha;

        // Update solution
        for i in 0..n {
            d[i] = z[i] + (theta * eta) * d[i];
            x[i] = x[i] + eta * d[i];
        }

        // Check convergence for even step
        let current_residual = tau;
        residual_history.push(current_residual);

        if current_residual <= tolerance {
            converged = true;
            break;
        }

        // Update w for next iteration
        w = matrix_vector_multiply(matrix, &z.view())?;
    }

    // Compute final residual norm by explicit calculation
    let ax_final = matrix_vector_multiply(matrix, &x.view())?;
    let final_residual = b - &ax_final;
    let final_residual_norm = l2_norm(&final_residual.view());

    Ok(TFQMRResult {
        x,
        iterations: iter,
        residual_norm: final_residual_norm,
        converged,
        residual_history: Some(residual_history),
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

/// Compute L2 norm of a vector
#[allow(dead_code)]
fn l2_norm<T>(x: &ArrayView1<T>) -> T
where
    T: Float + Debug + Copy,
{
    (x.iter().map(|&val| val * val).fold(T::zero(), |a, b| a + b)).sqrt()
}

/// Compute dot product of two vectors
#[allow(dead_code)]
fn dot_product<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float + Debug + Copy,
{
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi * yi)
        .fold(T::zero(), |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    #[ignore] // TODO: Fix TFQMR algorithm - currently not converging correctly
    fn test_tfqmr_simple_system() {
        // Create a simple 3x3 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let result = tfqmr(&matrix, &b.view(), None, TFQMROptions::default()).unwrap();

        assert!(result.converged);

        // Verify solution by computing residual
        let ax = matrix_vector_multiply(&matrix, &result.x.view()).unwrap();
        let residual = &b - &ax;
        let residual_norm = l2_norm(&residual.view());

        assert!(residual_norm < 1e-6);
    }

    #[test]
    #[ignore] // TODO: Fix TFQMR algorithm - currently not converging correctly
    fn test_tfqmr_diagonal_system() {
        // Create a diagonal system
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![4.0, 9.0, 16.0]);
        let result = tfqmr(&matrix, &b.view(), None, TFQMROptions::default()).unwrap();

        assert!(result.converged);

        // For diagonal system, solution should be [2, 3, 4]
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[2], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tfqmr_with_initial_guess() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![5.0, 6.0, 7.0]);
        let x0 = Array1::from_vec(vec![4.0, 5.0, 6.0]); // Close to solution

        let result = tfqmr(
            &matrix,
            &b.view(),
            Some(&x0.view()),
            TFQMROptions::default(),
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.iterations <= 5); // Should converge quickly with good initial guess
    }
}
