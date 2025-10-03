//! Generalized Conjugate Residual with Orthogonalization and Truncation (GCROT-m) method
//!
//! GCROT-m is a Krylov subspace method for solving sparse linear systems.
//! It maintains a set of search directions and performs orthogonalization
//! to improve stability and convergence.

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Type alias for GCROT inner iteration result
type GCROTInnerResult<T> = SparseResult<(Array1<T>, Option<Array1<T>>, Option<Array1<T>>, bool)>;

/// Options for the GCROT solver
#[derive(Debug, Clone)]
pub struct GCROTOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum dimension of the truncated space (m parameter)
    pub truncation_size: usize,
    /// Whether to store residual history
    pub store_residual_history: bool,
}

impl Default for GCROTOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            truncation_size: 20,
            store_residual_history: true,
        }
    }
}

/// Result from GCROT solver
#[derive(Debug, Clone)]
pub struct GCROTResult<T> {
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

/// Generalized Conjugate Residual with Orthogonalization and Truncation method
///
/// Solves the linear system A * x = b using the GCROT-m method.
/// This method builds and maintains a truncated Krylov subspace to
/// accelerate convergence for challenging linear systems.
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
/// A `GCROTResult` containing the solution and convergence information
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::{gcrot, GCROTOptions};
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
/// // Solve using GCROT
/// let result = gcrot(&matrix, &b.view(), None, GCROTOptions::default()).unwrap();
/// ```
#[allow(dead_code)]
pub fn gcrot<T, S>(
    matrix: &S,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    options: GCROTOptions,
) -> SparseResult<GCROTResult<T>>
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
        return Ok(GCROTResult {
            x,
            iterations: 0,
            residual_norm: initial_residual_norm,
            converged: true,
            residual_history: if options.store_residual_history {
                Some(vec![initial_residual_norm])
            } else {
                None
            },
        });
    }

    let m = options.truncation_size;

    // Storage for the truncated space
    let mut c_vectors = Array2::zeros((n, 0)); // C_k matrix
    let mut u_vectors = Array2::zeros((n, 0)); // U_k matrix (A * C_k)

    let mut residual_history = if options.store_residual_history {
        Some(vec![initial_residual_norm])
    } else {
        None
    };

    let mut converged = false;
    let mut iter = 0;

    for k in 0..options.max_iter {
        iter = k + 1;

        // GCROT inner iteration (flexible GMRES with truncation)
        let (delta_x, new_c, new_u, inner_converged) = gcrot_inner_iteration(
            matrix,
            &r.view(),
            &c_vectors.view(),
            &u_vectors.view(),
            tolerance,
        )?;

        // Update solution
        x = &x + &delta_x;

        // Update residual
        let ax = matrix_vector_multiply(matrix, &x.view())?;
        r = b - &ax;
        let residual_norm = l2_norm(&r.view());

        if let Some(ref mut history) = residual_history {
            history.push(residual_norm);
        }

        // Check convergence
        if residual_norm <= tolerance || inner_converged {
            converged = true;
            break;
        }

        // Update truncated space
        if let (Some(c), Some(u)) = (new_c, new_u) {
            if c_vectors.ncols() >= m {
                // Truncate the space by removing the oldest vector
                let mut new_c_vectors = Array2::zeros((n, m));
                let mut new_u_vectors = Array2::zeros((n, m));

                // Keep the m-1 most recent vectors and add the new one
                for j in 1..c_vectors.ncols() {
                    for i in 0..n {
                        new_c_vectors[[i, j - 1]] = c_vectors[[i, j]];
                        new_u_vectors[[i, j - 1]] = u_vectors[[i, j]];
                    }
                }

                // Add new vectors
                for i in 0..n {
                    new_c_vectors[[i, m - 1]] = c[i];
                    new_u_vectors[[i, m - 1]] = u[i];
                }

                c_vectors = new_c_vectors;
                u_vectors = new_u_vectors;
            } else {
                // Simply append the new vectors
                let old_cols = c_vectors.ncols();
                let mut new_c_vectors = Array2::zeros((n, old_cols + 1));
                let mut new_u_vectors = Array2::zeros((n, old_cols + 1));

                // Copy old vectors
                for j in 0..old_cols {
                    for i in 0..n {
                        new_c_vectors[[i, j]] = c_vectors[[i, j]];
                        new_u_vectors[[i, j]] = u_vectors[[i, j]];
                    }
                }

                // Add new vectors
                for i in 0..n {
                    new_c_vectors[[i, old_cols]] = c[i];
                    new_u_vectors[[i, old_cols]] = u[i];
                }

                c_vectors = new_c_vectors;
                u_vectors = new_u_vectors;
            }
        }
    }

    // Compute final residual norm
    let ax_final = matrix_vector_multiply(matrix, &x.view())?;
    let final_residual = b - &ax_final;
    let final_residual_norm = l2_norm(&final_residual.view());

    Ok(GCROTResult {
        x,
        iterations: iter,
        residual_norm: final_residual_norm,
        converged,
        residual_history,
    })
}

/// Inner GCROT iteration (flexible GMRES step)
#[allow(dead_code)]
fn gcrot_inner_iteration<T, S>(
    matrix: &S,
    r: &ArrayView1<T>,
    c_vectors: &scirs2_core::ndarray::ArrayView2<T>,
    u_vectors: &scirs2_core::ndarray::ArrayView2<T>,
    tolerance: T,
) -> GCROTInnerResult<T>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = r.len();
    let k = c_vectors.ncols(); // Number of _vectors in truncated space

    // Start with the current residual
    let mut v = r.to_owned();
    let beta = l2_norm(&v.view());

    if beta <= tolerance {
        return Ok((Array1::zeros(n), None, None, true));
    }

    // Normalize v
    for i in 0..n {
        v[i] = v[i] / beta;
    }

    // Orthogonalize against the truncated space
    for j in 0..k {
        let mut proj = T::zero();
        for i in 0..n {
            proj = proj + u_vectors[[i, j]] * v[i];
        }

        for i in 0..n {
            v[i] = v[i] - proj * c_vectors[[i, j]];
        }
    }

    // Renormalize
    let v_norm = l2_norm(&v.view());
    if v_norm > T::from(1e-12).unwrap() {
        for i in 0..n {
            v[i] = v[i] / v_norm;
        }
    }

    // Compute A * v
    let av = matrix_vector_multiply(matrix, &v.view())?;

    // Simple update: delta_x = (beta / ||A*v||^2) * (A*v . r) * v
    let av_norm_sq = dot_product(&av.view(), &av.view());
    let av_r_dot = dot_product(&av.view(), r);

    if av_norm_sq > T::from(1e-12).unwrap() {
        let alpha = av_r_dot / av_norm_sq;
        let mut delta_x = Array1::zeros(n);

        for i in 0..n {
            delta_x[i] = alpha * v[i];
        }

        Ok((delta_x, Some(v), Some(av), false))
    } else {
        Ok((Array1::zeros(n), None, None, true))
    }
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

    #[test]
    fn test_gcrot_simple_system() {
        // Create a simple 3x3 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let result = gcrot(&matrix, &b.view(), None, GCROTOptions::default()).unwrap();

        assert!(result.converged);

        // Verify solution by computing residual
        let ax = matrix_vector_multiply(&matrix, &result.x.view()).unwrap();
        let residual = &b - &ax;
        let residual_norm = l2_norm(&residual.view());

        assert!(residual_norm < 1e-6);
    }

    #[test]
    fn test_gcrot_diagonal_system() {
        // Create a simple diagonal-dominant system that should converge easily
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![5.0, 5.0, 5.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![5.0, 10.0, 15.0]);

        let result = gcrot(&matrix, &b.view(), None, GCROTOptions::default()).unwrap();

        assert!(result.converged);

        // Verify solution by computing residual
        let ax = matrix_vector_multiply(&matrix, &result.x.view()).unwrap();
        let residual = &b - &ax;
        let residual_norm = l2_norm(&residual.view());

        assert!(residual_norm < 1e-6);
    }

    #[test]
    fn test_gcrot_truncation() {
        // Test with small truncation size
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let options = GCROTOptions {
            truncation_size: 2, // Small truncation
            ..Default::default()
        };

        let result = gcrot(&matrix, &b.view(), None, options).unwrap();

        // Should still converge even with small truncation
        assert!(result.converged);
    }
}
