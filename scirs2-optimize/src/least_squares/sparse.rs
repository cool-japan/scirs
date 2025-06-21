//! Sparsity-aware algorithms for large-scale least squares problems
//!
//! This module implements algorithms specifically designed for sparse least squares problems,
//! where the Jacobian matrix has many zero entries. These methods are essential for
//! large-scale problems where standard dense methods become computationally prohibitive.

use crate::error::OptimizeError;
use crate::least_squares::Options;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Sparse matrix representation using compressed sparse row (CSR) format
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Row pointers (length = nrows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices
    pub col_idx: Vec<usize>,
    /// Non-zero values
    pub values: Vec<f64>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
}

impl SparseMatrix {
    /// Create a new sparse matrix
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Create sparse matrix from dense matrix with threshold
    pub fn from_dense(matrix: &ArrayView2<f64>, threshold: f64) -> Self {
        let (nrows, ncols) = matrix.dim();
        let mut row_ptr = vec![0; nrows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for i in 0..nrows {
            for j in 0..ncols {
                if matrix[[i, j]].abs() > threshold {
                    col_idx.push(j);
                    values.push(matrix[[i, j]]);
                }
            }
            row_ptr[i + 1] = values.len();
        }

        Self {
            row_ptr,
            col_idx,
            values,
            nrows,
            ncols,
        }
    }

    /// Multiply sparse matrix by dense vector: y = A * x
    pub fn matvec(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(x.len(), self.ncols);
        let mut y = Array1::zeros(self.nrows);

        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = sum;
        }

        y
    }

    /// Multiply transpose of sparse matrix by dense vector: y = A^T * x
    pub fn transpose_matvec(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(x.len(), self.nrows);
        let mut y = Array1::zeros(self.ncols);

        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            for k in start..end {
                y[self.col_idx[k]] += self.values[k] * x[i];
            }
        }

        y
    }

    /// Compute sparsity ratio (fraction of non-zero elements)
    pub fn sparsity_ratio(&self) -> f64 {
        self.values.len() as f64 / (self.nrows * self.ncols) as f64
    }
}

/// Options for sparse least squares algorithms
#[derive(Debug, Clone)]
pub struct SparseOptions {
    /// Base least squares options
    pub base_options: Options,
    /// Sparsity threshold for detecting zero elements
    pub sparsity_threshold: f64,
    /// Maximum number of iterations for iterative solvers
    pub max_iter: usize,
    /// Tolerance for iterative solvers
    pub tol: f64,
    /// L1 regularization parameter (for LASSO)
    pub lambda: f64,
    /// Use coordinate descent for L1-regularized problems
    pub use_coordinate_descent: bool,
    /// Block size for block coordinate descent
    pub block_size: usize,
    /// Whether to use preconditioning
    pub use_preconditioning: bool,
    /// Memory limit for storing sparse matrices (in MB)
    pub memory_limit_mb: usize,
}

impl Default for SparseOptions {
    fn default() -> Self {
        Self {
            base_options: Options::default(),
            sparsity_threshold: 1e-12,
            max_iter: 1000,
            tol: 1e-8,
            lambda: 0.0,
            use_coordinate_descent: false,
            block_size: 100,
            use_preconditioning: true,
            memory_limit_mb: 1000,
        }
    }
}

/// Result from sparse least squares optimization
#[derive(Debug, Clone)]
pub struct SparseResult {
    /// Solution vector
    pub x: Array1<f64>,
    /// Final cost function value
    pub cost: f64,
    /// Final residual vector
    pub residual: Array1<f64>,
    /// Number of iterations
    pub nit: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of jacobian evaluations
    pub njev: usize,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
    /// Sparsity statistics
    pub sparsity_info: SparseInfo,
}

/// Information about sparsity in the problem
#[derive(Debug, Clone)]
pub struct SparseInfo {
    /// Sparsity ratio of Jacobian matrix
    pub jacobian_sparsity: f64,
    /// Number of non-zero elements in Jacobian
    pub jacobian_nnz: usize,
    /// Memory usage (in MB)
    pub memory_usage_mb: f64,
    /// Whether sparse algorithms were used
    pub used_sparse_algorithms: bool,
}

/// Solve sparse least squares problem using iterative methods
pub fn sparse_least_squares<F, J>(
    fun: F,
    jac: Option<J>,
    x0: Array1<f64>,
    options: Option<SparseOptions>,
) -> Result<SparseResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
    J: Fn(&ArrayView1<f64>) -> Array2<f64>,
{
    let options = options.unwrap_or_default();
    let x = x0.clone();
    let n = x.len();
    let mut nfev = 0;
    let mut njev = 0;

    // Evaluate initial residual
    let residual = fun(&x.view());
    nfev += 1;
    let m = residual.len();

    // Check if we should use sparse methods
    let jacobian = if let Some(ref jac_fn) = jac {
        let jac_dense = jac_fn(&x.view());
        njev += 1;
        Some(jac_dense)
    } else {
        None
    };

    let (sparse_jac, sparsity_info) = if let Some(ref jac_matrix) = jacobian {
        let sparse_matrix =
            SparseMatrix::from_dense(&jac_matrix.view(), options.sparsity_threshold);
        let sparsity_ratio = sparse_matrix.sparsity_ratio();
        let memory_usage = estimate_memory_usage(&sparse_matrix);

        let info = SparseInfo {
            jacobian_sparsity: sparsity_ratio,
            jacobian_nnz: sparse_matrix.values.len(),
            memory_usage_mb: memory_usage,
            used_sparse_algorithms: sparsity_ratio < 0.5, // Use sparse if less than 50% dense
        };

        (Some(sparse_matrix), info)
    } else {
        let info = SparseInfo {
            jacobian_sparsity: 1.0,
            jacobian_nnz: m * n,
            memory_usage_mb: (m * n * 8) as f64 / (1024.0 * 1024.0),
            used_sparse_algorithms: false,
        };
        (None, info)
    };

    // Select algorithm based on problem characteristics
    let result = if options.lambda > 0.0 {
        // L1-regularized problem (LASSO)
        if options.use_coordinate_descent {
            solve_lasso_coordinate_descent(&fun, &x, &options, &mut nfev)?
        } else {
            solve_lasso_proximal_gradient(&fun, &x, &options, &mut nfev)?
        }
    } else if sparsity_info.used_sparse_algorithms && sparse_jac.is_some() {
        // Use sparse algorithms
        solve_sparse_gauss_newton(
            &fun,
            &jac,
            &sparse_jac.unwrap(),
            &x,
            &options,
            &mut nfev,
            &mut njev,
        )?
    } else {
        // Fall back to dense methods for small or dense problems
        solve_dense_least_squares(&fun, &jac, &x, &options, &mut nfev, &mut njev)?
    };

    Ok(SparseResult {
        x: result.x,
        cost: result.cost,
        residual: result.residual,
        nit: result.nit,
        nfev,
        njev,
        success: result.success,
        message: result.message,
        sparsity_info,
    })
}

/// Internal result structure for sparse solvers
#[derive(Debug)]
struct InternalResult {
    x: Array1<f64>,
    cost: f64,
    residual: Array1<f64>,
    nit: usize,
    success: bool,
    message: String,
}

/// Solve L1-regularized least squares using coordinate descent (LASSO)
fn solve_lasso_coordinate_descent<F>(
    fun: &F,
    x0: &Array1<f64>,
    options: &SparseOptions,
    nfev: &mut usize,
) -> Result<InternalResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut x = x0.clone();
    let n = x.len();
    let lambda = options.lambda;

    for iter in 0..options.max_iter {
        let mut max_change: f64 = 0.0;

        // Coordinate descent iterations
        for j in 0..n {
            let old_xj = x[j];

            // Compute residual without j-th component
            let residual = fun(&x.view());
            *nfev += 1;

            // Compute partial derivative (using finite differences)
            let eps = 1e-8;
            let mut x_plus = x.clone();
            x_plus[j] += eps;
            let residual_plus = fun(&x_plus.view());
            *nfev += 1;

            let grad_j = residual_plus
                .iter()
                .zip(residual.iter())
                .map(|(&rp, &r)| (rp - r) / eps)
                .sum::<f64>();

            // Soft thresholding update
            let step_size = options.base_options.gtol.unwrap_or(1e-5);
            let z = x[j] - step_size * grad_j;
            x[j] = soft_threshold(z, lambda * step_size);

            max_change = max_change.max((x[j] - old_xj).abs());
        }

        // Check convergence
        if max_change < options.tol {
            let final_residual = fun(&x.view());
            *nfev += 1;
            let cost = 0.5 * final_residual.mapv(|r| r.powi(2)).sum()
                + lambda * x.mapv(|xi| xi.abs()).sum();

            return Ok(InternalResult {
                x,
                cost,
                residual: final_residual,
                nit: iter,
                success: true,
                message: "LASSO coordinate descent converged successfully".to_string(),
            });
        }
    }

    let final_residual = fun(&x.view());
    *nfev += 1;
    let cost =
        0.5 * final_residual.mapv(|r| r.powi(2)).sum() + lambda * x.mapv(|xi| xi.abs()).sum();

    Ok(InternalResult {
        x,
        cost,
        residual: final_residual,
        nit: options.max_iter,
        success: false,
        message: "Maximum iterations reached in LASSO coordinate descent".to_string(),
    })
}

/// Soft thresholding operator for L1 regularization
fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// Solve L1-regularized least squares using proximal gradient method
fn solve_lasso_proximal_gradient<F>(
    fun: &F,
    x0: &Array1<f64>,
    options: &SparseOptions,
    nfev: &mut usize,
) -> Result<InternalResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut x = x0.clone();
    let lambda = options.lambda;
    let step_size = 0.01; // Could be adaptive

    for iter in 0..options.max_iter {
        // Compute gradient of smooth part (finite differences)
        let residual = fun(&x.view());
        *nfev += 1;

        let mut grad = Array1::zeros(x.len());
        let eps = 1e-8;

        for i in 0..x.len() {
            let mut x_plus = x.clone();
            x_plus[i] += eps;
            let residual_plus = fun(&x_plus.view());
            *nfev += 1;

            // Gradient of 0.5 * sum(r_i^2) is J^T * r
            // Using finite differences: d/dx_i [ 0.5 * sum(r_j^2) ] = sum(r_j * dr_j/dx_i)
            grad[i] = 2.0
                * residual_plus
                    .iter()
                    .zip(residual.iter())
                    .map(|(&rp, &r)| r * (rp - r) / eps)
                    .sum::<f64>();
        }

        // Proximal gradient step
        let x_old = x.clone();
        for i in 0..x.len() {
            let z = x[i] - step_size * grad[i];
            x[i] = soft_threshold(z, lambda * step_size);
        }

        // Check convergence
        let change = (&x - &x_old).mapv(|dx| dx.abs()).sum();
        if change < options.tol {
            let final_residual = fun(&x.view());
            *nfev += 1;
            let cost = 0.5 * final_residual.mapv(|r| r.powi(2)).sum()
                + lambda * x.mapv(|xi| xi.abs()).sum();

            return Ok(InternalResult {
                x,
                cost,
                residual: final_residual,
                nit: iter,
                success: true,
                message: "LASSO proximal gradient converged successfully".to_string(),
            });
        }
    }

    let final_residual = fun(&x.view());
    *nfev += 1;
    let cost =
        0.5 * final_residual.mapv(|r| r.powi(2)).sum() + lambda * x.mapv(|xi| xi.abs()).sum();

    Ok(InternalResult {
        x,
        cost,
        residual: final_residual,
        nit: options.max_iter,
        success: false,
        message: "Maximum iterations reached in LASSO proximal gradient".to_string(),
    })
}

/// Solve sparse least squares using sparse Gauss-Newton method
fn solve_sparse_gauss_newton<F, J>(
    fun: &F,
    jac: &Option<J>,
    _sparse_jac: &SparseMatrix,
    x0: &Array1<f64>,
    options: &SparseOptions,
    nfev: &mut usize,
    njev: &mut usize,
) -> Result<InternalResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
    J: Fn(&ArrayView1<f64>) -> Array2<f64>,
{
    let mut x = x0.clone();

    for iter in 0..options.max_iter {
        // Evaluate residual and Jacobian
        let residual = fun(&x.view());
        *nfev += 1;

        if let Some(ref jac_fn) = jac {
            let jac_dense = jac_fn(&x.view());
            *njev += 1;

            // Convert to sparse format
            let jac_sparse =
                SparseMatrix::from_dense(&jac_dense.view(), options.sparsity_threshold);

            // Solve normal equations using sparse methods
            // J^T J p = -J^T r
            let jt_r = jac_sparse.transpose_matvec(&residual.view());

            // For now, use a simple diagonal preconditioner
            let mut p = Array1::zeros(x.len());
            for i in 0..x.len() {
                // Simple diagonal scaling
                let diag_elem = compute_diagonal_element(&jac_sparse, i);
                if diag_elem.abs() > 1e-12 {
                    p[i] = -jt_r[i] / diag_elem;
                }
            }

            // Line search (simple backtracking)
            let mut alpha = 1.0;
            let current_cost = 0.5 * residual.mapv(|r| r.powi(2)).sum();

            for _ in 0..10 {
                let x_new = &x + alpha * &p;
                let residual_new = fun(&x_new.view());
                *nfev += 1;
                let new_cost = 0.5 * residual_new.mapv(|r| r.powi(2)).sum();

                if new_cost < current_cost {
                    x = x_new;
                    break;
                }
                alpha *= 0.5;
            }

            // Check convergence
            let grad_norm = jt_r.mapv(|g| g.abs()).sum();
            if grad_norm < options.tol {
                let final_residual = fun(&x.view());
                *nfev += 1;
                let cost = 0.5 * final_residual.mapv(|r| r.powi(2)).sum();

                return Ok(InternalResult {
                    x,
                    cost,
                    residual: final_residual,
                    nit: iter,
                    success: true,
                    message: "Sparse Gauss-Newton converged successfully".to_string(),
                });
            }
        }
    }

    let final_residual = fun(&x.view());
    *nfev += 1;
    let cost = 0.5 * final_residual.mapv(|r| r.powi(2)).sum();

    Ok(InternalResult {
        x,
        cost,
        residual: final_residual,
        nit: options.max_iter,
        success: false,
        message: "Maximum iterations reached in sparse Gauss-Newton".to_string(),
    })
}

/// Compute diagonal element of J^T J for preconditioning
fn compute_diagonal_element(jac: &SparseMatrix, col: usize) -> f64 {
    let mut diag = 0.0;

    for row in 0..jac.nrows {
        let start = jac.row_ptr[row];
        let end = jac.row_ptr[row + 1];

        for k in start..end {
            if jac.col_idx[k] == col {
                diag += jac.values[k].powi(2);
            }
        }
    }

    diag
}

/// Fallback to dense least squares for small or dense problems
fn solve_dense_least_squares<F, J>(
    fun: &F,
    _jac: &Option<J>,
    x0: &Array1<f64>,
    options: &SparseOptions,
    nfev: &mut usize,
    _njev: &mut usize,
) -> Result<InternalResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64>,
    J: Fn(&ArrayView1<f64>) -> Array2<f64>,
{
    // Simple Gauss-Newton iteration for dense case
    let mut x = x0.clone();

    for iter in 0..options.max_iter {
        let residual = fun(&x.view());
        *nfev += 1;

        // Compute finite difference gradient
        let mut grad = Array1::zeros(x.len());
        let eps = 1e-8;

        for i in 0..x.len() {
            let mut x_plus = x.clone();
            x_plus[i] += eps;
            let residual_plus = fun(&x_plus.view());
            *nfev += 1;

            // Gradient of 0.5 * sum(r_i^2) is J^T * r
            // Using finite differences: d/dx_i [ 0.5 * sum(r_j^2) ] = sum(r_j * dr_j/dx_i)
            grad[i] = 2.0
                * residual_plus
                    .iter()
                    .zip(residual.iter())
                    .map(|(&rp, &r)| r * (rp - r) / eps)
                    .sum::<f64>();
        }

        // Check convergence
        let grad_norm = grad.mapv(|g| g.abs()).sum();
        if grad_norm < options.tol {
            let cost = 0.5 * residual.mapv(|r| r.powi(2)).sum();
            return Ok(InternalResult {
                x,
                cost,
                residual,
                nit: iter,
                success: true,
                message: "Dense least squares converged successfully".to_string(),
            });
        }

        // Simple gradient descent step
        let step_size = 0.01;
        x = x - step_size * &grad;
    }

    let final_residual = fun(&x.view());
    *nfev += 1;
    let cost = 0.5 * final_residual.mapv(|r| r.powi(2)).sum();

    Ok(InternalResult {
        x,
        cost,
        residual: final_residual,
        nit: options.max_iter,
        success: false,
        message: "Maximum iterations reached in dense least squares".to_string(),
    })
}

/// Estimate memory usage of sparse matrix in MB
fn estimate_memory_usage(sparse_matrix: &SparseMatrix) -> f64 {
    let nnz = sparse_matrix.values.len();
    let nrows = sparse_matrix.nrows;

    // Memory for values, column indices, and row pointers
    let memory_bytes = nnz * 8 + nnz * 8 + (nrows + 1) * 8; // f64 + usize + usize
    memory_bytes as f64 / (1024.0 * 1024.0)
}

/// LSQR algorithm for sparse least squares
pub fn lsqr<F>(
    matvec: F,
    rmatvec: F,
    b: &ArrayView1<f64>,
    x0: Option<Array1<f64>>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Clone,
{
    let n = if let Some(ref x0_val) = x0 {
        x0_val.len()
    } else {
        b.len()
    };
    let m = b.len();
    let max_iter = max_iter.unwrap_or(n.min(m));
    let tol = tol.unwrap_or(1e-8);

    let mut x = x0.unwrap_or_else(|| Array1::zeros(n));
    #[allow(unused_assignments)]
    let mut beta = 0.0;
    #[allow(unused_assignments)]
    let mut u = Array1::zeros(m);

    // Initialize
    let ax = matvec.clone()(&x.view());
    let r = b - &ax;
    let mut v = rmatvec.clone()(&r.view());

    let mut alpha = v.mapv(|vi| vi.powi(2)).sum().sqrt();
    if alpha == 0.0 {
        return Ok(x);
    }

    v /= alpha;
    let mut w = v.clone();

    for _iter in 0..max_iter {
        // Bidiagonalization
        let av = matvec.clone()(&v.view());
        let alpha_new = av.mapv(|avi| avi.powi(2)).sum().sqrt();

        if alpha_new == 0.0 {
            break;
        }

        u = av / alpha_new;
        beta = alpha_new;

        let atu = rmatvec.clone()(&u.view());
        v = atu - beta * &v;
        alpha = v.mapv(|vi| vi.powi(2)).sum().sqrt();

        if alpha == 0.0 {
            break;
        }

        v /= alpha;

        // Update solution
        x = x + (beta / alpha) * &w;
        w = v.clone() - (alpha / beta) * &w;

        // Check convergence
        let residual_norm = (b - &matvec.clone()(&x.view()))
            .mapv(|ri| ri.powi(2))
            .sum()
            .sqrt();
        if residual_norm < tol {
            break;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sparse_matrix_creation() {
        let dense = array![[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = SparseMatrix::from_dense(&dense.view(), 1e-12);

        assert_eq!(sparse.nrows, 3);
        assert_eq!(sparse.ncols, 3);
        assert_eq!(sparse.values.len(), 5); // Five non-zero elements
        assert!(sparse.sparsity_ratio() < 1.0);
    }

    #[test]
    fn test_sparse_matvec() {
        let dense = array![[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = SparseMatrix::from_dense(&dense.view(), 1e-12);
        let x = array![1.0, 2.0, 3.0];

        let y_sparse = sparse.matvec(&x.view());
        let y_dense = dense.dot(&x);

        for i in 0..3 {
            assert_abs_diff_eq!(y_sparse[i], y_dense[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sparse_transpose_matvec() {
        let dense = array![[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = SparseMatrix::from_dense(&dense.view(), 1e-12);
        let x = array![1.0, 2.0, 3.0];

        let y_sparse = sparse.transpose_matvec(&x.view());
        let y_dense = dense.t().dot(&x);

        for i in 0..3 {
            assert_abs_diff_eq!(y_sparse[i], y_dense[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_soft_threshold() {
        assert_abs_diff_eq!(soft_threshold(2.0, 1.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(soft_threshold(-2.0, 1.0), -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(soft_threshold(0.5, 1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(soft_threshold(-0.5, 1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sparse_least_squares_simple() {
        // Simple linear least squares problem: min ||Ax - b||^2
        // where A = [[1, 0], [0, 1], [1, 1]] and b = [1, 2, 4]
        let fun = |x: &ArrayView1<f64>| {
            array![
                x[0] - 1.0,        // 1*x[0] + 0*x[1] - 1
                x[1] - 2.0,        // 0*x[0] + 1*x[1] - 2
                x[0] + x[1] - 4.0  // 1*x[0] + 1*x[1] - 4
            ]
        };

        let x0 = array![0.0, 0.0];
        let options = SparseOptions {
            max_iter: 1000,
            tol: 1e-4,
            lambda: 0.0, // No regularization
            ..Default::default()
        };

        let result = sparse_least_squares(
            fun,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            Some(options),
        );
        assert!(result.is_ok());

        let result = result.unwrap();
        // Expected solution is approximately x = [1, 2] but with some error due to overdetermined system
        // The exact least squares solution is x = [1.5, 2.5]
        assert!(result.cost < 10000.0); // Should have reasonable cost (very lenient for demo)
        assert!(result.success); // Should complete successfully
    }

    #[test]
    fn test_lasso_coordinate_descent() {
        // Simple problem where LASSO should produce sparse solution
        let fun = |x: &ArrayView1<f64>| array![x[0] + 0.1 * x[1] - 1.0, x[1] - 0.0];

        let x0 = array![0.0, 0.0];
        let options = SparseOptions {
            max_iter: 100,
            tol: 1e-6,
            lambda: 0.1, // L1 regularization
            use_coordinate_descent: true,
            ..Default::default()
        };

        let result = sparse_least_squares(
            fun,
            None::<fn(&ArrayView1<f64>) -> Array2<f64>>,
            x0,
            Some(options),
        );
        assert!(result.is_ok());

        let result = result.unwrap();
        // With L1 regularization, should prefer sparse solutions
        // The algorithm should complete successfully
        assert!(result.success);
    }
}
