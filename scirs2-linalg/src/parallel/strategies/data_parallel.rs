//! Algorithm-specific parallel implementations
//!
//! This module provides data parallel implementations of linear algebra algorithms
//! optimized for performance and scalability across multiple CPU cores.

use super::super::{adaptive, WorkerConfig};
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use scirs2_core::parallel_ops::*;
use std::iter::Sum;

/// Parallel matrix-vector multiplication
///
/// This is a simpler and more effective parallelization that can be used
/// as a building block for more complex algorithms.
///
/// # Arguments
///
/// * `matrix` - Input matrix
/// * `vector` - Input vector
/// * `config` - Worker configuration
///
/// # Returns
///
/// * Result vector y = A * x
pub fn parallel_matvec<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    config: &WorkerConfig,
) -> LinalgResult<Array1<F>>
where
    F: Float + Send + Sync + Zero + Sum + 'static,
{
    let (m, n) = matrix.dim();
    if n != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix-vector dimensions incompatible: {}x{} * {}",
            m,
            n,
            vector.len()
        )));
    }

    let datasize = m * n;
    if !adaptive::should_use_parallel(datasize, config) {
        // Fall back to serial computation
        return Ok(matrix.dot(vector));
    }

    config.apply();

    // Parallel computation of each row
    let result_vec: Vec<F> = (0..m)
        .into_par_iter()
        .map(|i| {
            matrix
                .row(i)
                .iter()
                .zip(vector.iter())
                .map(|(&aij, &xj)| aij * xj)
                .sum()
        })
        .collect();

    Ok(Array1::from_vec(result_vec))
}

/// Parallel power iteration for dominant eigenvalue
///
/// This implementation uses parallel matrix-vector multiplications
/// in the power iteration method for computing dominant eigenvalues.
pub fn parallel_power_iteration<F>(
    matrix: &ArrayView2<F>,
    max_iter: usize,
    tolerance: F,
    config: &WorkerConfig,
) -> LinalgResult<(F, Array1<F>)>
where
    F: Float + Send + Sync + Zero + Sum + NumAssign + One + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    if m != n {
        return Err(LinalgError::ShapeError(
            "Power iteration requires square matrix".to_string(),
        ));
    }

    let datasize = m * n;
    if !adaptive::should_use_parallel(datasize, config) {
        // Fall back to serial power iteration
        return crate::eigen::power_iteration(&matrix.view(), max_iter, tolerance);
    }

    config.apply();

    // Initialize with simple vector
    let mut v = Array1::ones(n);
    let norm = v.iter().map(|&x| x * x).sum::<F>().sqrt();
    v /= norm;

    let mut eigenvalue = F::zero();

    for _iter in 0..max_iter {
        // Use the parallel matrix-vector multiplication
        let new_v = parallel_matvec(matrix, &v.view(), config)?;

        // Compute eigenvalue estimate (Rayleigh quotient)
        let new_eigenvalue = new_v
            .iter()
            .zip(v.iter())
            .map(|(&new_vi, &vi)| new_vi * vi)
            .sum::<F>();

        // Normalize
        let norm = new_v.iter().map(|&x| x * x).sum::<F>().sqrt();
        if norm < F::epsilon() {
            return Err(LinalgError::ComputationError(
                "Vector became zero during iteration".to_string(),
            ));
        }
        let normalized_v = new_v / norm;

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tolerance {
            return Ok((new_eigenvalue, normalized_v));
        }

        eigenvalue = new_eigenvalue;
        v = normalized_v;
    }

    Err(LinalgError::ComputationError(
        "Power iteration failed to converge".to_string(),
    ))
}

/// Parallel vector operations for linear algebra
///
/// This module provides basic parallel vector operations that serve as
/// building blocks for more complex algorithms.
pub mod vector_ops {
    use super::*;

    /// Parallel dot product of two vectors
    pub fn parallel_dot<F>(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<F>
    where
        F: Float + Send + Sync + Zero + Sum + 'static,
    {
        if x.len() != y.len() {
            return Err(LinalgError::ShapeError(
                "Vectors must have same length for dot product".to_string(),
            ));
        }

        let datasize = x.len();
        if !adaptive::should_use_parallel(datasize, config) {
            return Ok(x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum());
        }

        config.apply();

        let result = (0..x.len()).into_par_iter().map(|i| x[i] * y[i]).sum();

        Ok(result)
    }

    /// Parallel vector norm computation
    pub fn parallel_norm<F>(x: &ArrayView1<F>, config: &WorkerConfig) -> LinalgResult<F>
    where
        F: Float + Send + Sync + Zero + Sum + 'static,
    {
        let datasize = x.len();
        if !adaptive::should_use_parallel(datasize, config) {
            return Ok(x.iter().map(|&xi| xi * xi).sum::<F>().sqrt());
        }

        config.apply();

        let sum_squares = (0..x.len()).into_par_iter().map(|i| x[i] * x[i]).sum::<F>();

        Ok(sum_squares.sqrt())
    }

    /// Parallel AXPY operation: y = a*x + y
    ///
    /// Note: This function returns a new array rather than modifying in-place
    /// due to complications with parallel mutable iteration.
    pub fn parallel_axpy<F>(
        alpha: F,
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + 'static,
    {
        if x.len() != y.len() {
            return Err(LinalgError::ShapeError(
                "Vectors must have same length for AXPY".to_string(),
            ));
        }

        let datasize = x.len();
        if !adaptive::should_use_parallel(datasize, config) {
            let result = x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| alpha * xi + yi)
                .collect();
            return Ok(Array1::from_vec(result));
        }

        config.apply();

        let result_vec: Vec<F> = (0..x.len())
            .into_par_iter()
            .map(|i| alpha * x[i] + y[i])
            .collect();

        Ok(Array1::from_vec(result_vec))
    }
}

/// Parallel matrix multiplication (GEMM)
///
/// Implements parallel general matrix multiplication with block-based
/// parallelization for improved cache performance.
pub fn parallel_gemm<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    config: &WorkerConfig,
) -> LinalgResult<scirs2_core::ndarray::Array2<F>>
where
    F: Float + Send + Sync + Zero + Sum + NumAssign + 'static,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();

    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible for multiplication: {m}x{k} * {k2}x{n}"
        )));
    }

    let datasize = m * k * n;
    if !adaptive::should_use_parallel(datasize, config) {
        return Ok(a.dot(b));
    }

    config.apply();

    // Block size for cache-friendly computation
    let blocksize = config.chunksize;

    let mut result = scirs2_core::ndarray::Array2::zeros((m, n));

    // Parallel computation using blocks
    result
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                let mut sum = F::zero();
                for kb in (0..k).step_by(blocksize) {
                    let k_end = std::cmp::min(kb + blocksize, k);
                    for ki in kb..k_end {
                        sum += a[[i, ki]] * b[[ki, j]];
                    }
                }
                row[j] = sum;
            }
        });

    Ok(result)
}

/// Parallel conjugate gradient solver
///
/// Implements parallel conjugate gradient method for solving linear systems
/// with symmetric positive definite matrices.
pub fn parallel_conjugate_gradient<F>(
    matrix: &ArrayView2<F>,
    b: &ArrayView1<F>,
    max_iter: usize,
    tolerance: F,
    config: &WorkerConfig,
) -> LinalgResult<Array1<F>>
where
    F: Float + Send + Sync + Zero + Sum + One + NumAssign + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    if m != n {
        return Err(LinalgError::ShapeError(
            "CG requires square matrix".to_string(),
        ));
    }
    if n != b.len() {
        return Err(LinalgError::ShapeError(
            "Matrix and vector dimensions incompatible".to_string(),
        ));
    }

    let datasize = m * n;
    if !adaptive::should_use_parallel(datasize, config) {
        return crate::iterative_solvers::conjugate_gradient(
            &matrix.view(),
            &b.view(),
            max_iter,
            tolerance,
            None,
        );
    }

    config.apply();

    // Initialize
    let mut x = Array1::zeros(n);

    // r = b - A*x
    let ax = parallel_matvec(matrix, &x.view(), config)?;
    let mut r = b - &ax;
    let mut p = r.clone();
    let mut rsold = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

    for _iter in 0..max_iter {
        let ap = parallel_matvec(matrix, &p.view(), config)?;
        let alpha = rsold / vector_ops::parallel_dot(&p.view(), &ap.view(), config)?;

        x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
        r = vector_ops::parallel_axpy(-alpha, &ap.view(), &r.view(), config)?;

        let rsnew = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

        if rsnew.sqrt() < tolerance {
            return Ok(x);
        }

        let beta = rsnew / rsold;
        p = vector_ops::parallel_axpy(beta, &p.view(), &r.view(), config)?;
        rsold = rsnew;
    }

    Err(LinalgError::ComputationError(
        "Conjugate gradient failed to converge".to_string(),
    ))
}

/// Parallel Jacobi method
///
/// Implements parallel Jacobi iteration for solving linear systems.
/// This method is particularly well-suited for parallel execution.
pub fn parallel_jacobi<F>(
    matrix: &ArrayView2<F>,
    b: &ArrayView1<F>,
    max_iter: usize,
    tolerance: F,
    config: &WorkerConfig,
) -> LinalgResult<Array1<F>>
where
    F: Float + Send + Sync + Zero + Sum + One + NumAssign + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    if m != n {
        return Err(LinalgError::ShapeError(
            "Jacobi method requires square matrix".to_string(),
        ));
    }
    if n != b.len() {
        return Err(LinalgError::ShapeError(
            "Matrix and vector dimensions incompatible".to_string(),
        ));
    }

    let datasize = m * n;
    if !adaptive::should_use_parallel(datasize, config) {
        return crate::iterative_solvers::jacobi_method(
            &matrix.view(),
            &b.view(),
            max_iter,
            tolerance,
            None,
        );
    }

    config.apply();

    // Extract diagonal
    let diag: Vec<F> = (0..n)
        .into_par_iter()
        .map(|i| {
            if matrix[[i, i]].abs() < F::epsilon() {
                F::one() // Avoid division by zero
            } else {
                matrix[[i, i]]
            }
        })
        .collect();

    let mut x = Array1::zeros(n);

    for _iter in 0..max_iter {
        // Parallel update: x_new[i] = (b[i] - sum(A[i,j]*x[j] for j != i)) / A[i,i]
        let x_new_vec: Vec<F> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut sum = b[i];
                for j in 0..n {
                    if i != j {
                        sum -= matrix[[i, j]] * x[j];
                    }
                }
                sum / diag[i]
            })
            .collect();

        let x_new = Array1::from_vec(x_new_vec);

        // Check convergence
        let diff = &x_new - &x;
        let error = vector_ops::parallel_norm(&diff.view(), config)?;

        if error < tolerance {
            return Ok(x_new);
        }

        x = x_new.clone();
    }

    Err(LinalgError::ComputationError(
        "Jacobi method failed to converge".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_parallel_matvec() {
        let config = WorkerConfig::default();
        let matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let vector = arr1(&[1.0, 2.0]);
        
        let result = parallel_matvec(&matrix.view(), &vector.view(), &config).unwrap();
        
        assert_eq!(result, arr1(&[5.0, 11.0]));
    }

    #[test]
    fn test_parallel_dot() {
        let config = WorkerConfig::default();
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        
        let result = vector_ops::parallel_dot(&x.view(), &y.view(), &config).unwrap();
        
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_parallel_norm() {
        let config = WorkerConfig::default();
        let x = arr1(&[3.0, 4.0]);
        
        let result = vector_ops::parallel_norm(&x.view(), &config).unwrap();
        
        assert_eq!(result, 5.0); // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    }

    #[test]
    fn test_parallel_axpy() {
        let config = WorkerConfig::default();
        let x = arr1(&[1.0, 2.0]);
        let y = arr1(&[3.0, 4.0]);
        let alpha = 2.0;
        
        let result = vector_ops::parallel_axpy(alpha, &x.view(), &y.view(), &config).unwrap();
        
        assert_eq!(result, arr1(&[5.0, 8.0])); // 2*1 + 3 = 5, 2*2 + 4 = 8
    }

    #[test]
    fn test_parallel_gemm() {
        let config = WorkerConfig::default();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        
        let result = parallel_gemm(&a.view(), &b.view(), &config).unwrap();
        
        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        assert_eq!(result, arr2(&[[19.0, 22.0], [43.0, 50.0]]));
    }
}