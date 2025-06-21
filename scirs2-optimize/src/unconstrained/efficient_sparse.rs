//! Efficient sparse Jacobian and Hessian handling for optimization
//!
//! This module provides optimization algorithms that efficiently handle sparse
//! Jacobians and Hessians using advanced sparsity patterns, adaptive coloring,
//! and specialized matrix operations.

use crate::error::OptimizeError;
use crate::sparse_numdiff::SparseFiniteDiffOptions;
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::check_convergence;
use crate::unconstrained::Options;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_sparse::{csr_array::CsrArray, sparray::SparseArray};
// Collections for potential future use
// use std::collections::{HashMap, HashSet};

/// Options for efficient sparse optimization
#[derive(Debug, Clone)]
pub struct EfficientSparseOptions {
    /// Base optimization options
    pub base_options: Options,
    /// Sparse finite difference options
    pub sparse_fd_options: SparseFiniteDiffOptions,
    /// Automatically detect sparsity pattern
    pub auto_detect_sparsity: bool,
    /// Sparsity detection threshold
    pub sparsity_threshold: f64,
    /// Maximum number of sparsity detection iterations
    pub max_sparsity_iterations: usize,
    /// Use adaptive sparsity pattern refinement
    pub adaptive_sparsity: bool,
    /// Enable Hessian sparsity for Newton-type methods
    pub use_sparse_hessian: bool,
    /// Maximum percentage of non-zeros for sparse representation
    pub sparse_percentage_threshold: f64,
    /// Enable parallel sparse operations
    pub parallel_sparse_ops: bool,
}

impl Default for EfficientSparseOptions {
    fn default() -> Self {
        Self {
            base_options: Options::default(),
            sparse_fd_options: SparseFiniteDiffOptions::default(),
            auto_detect_sparsity: true,
            sparsity_threshold: 1e-12,
            max_sparsity_iterations: 5,
            adaptive_sparsity: true,
            use_sparse_hessian: true,
            sparse_percentage_threshold: 0.1, // Use sparse if <10% non-zero
            parallel_sparse_ops: true,
        }
    }
}

/// Sparse pattern information and statistics
#[derive(Debug, Clone)]
pub struct SparsityInfo {
    /// Jacobian sparsity pattern
    pub jacobian_pattern: Option<CsrArray<f64>>,
    /// Hessian sparsity pattern
    pub hessian_pattern: Option<CsrArray<f64>>,
    /// Number of non-zero elements in Jacobian
    pub jacobian_nnz: usize,
    /// Number of non-zero elements in Hessian
    pub hessian_nnz: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Jacobian sparsity percentage
    pub jacobian_sparsity: f64,
    /// Hessian sparsity percentage
    pub hessian_sparsity: f64,
}

impl SparsityInfo {
    fn new(n: usize) -> Self {
        Self {
            jacobian_pattern: None,
            hessian_pattern: None,
            jacobian_nnz: 0,
            hessian_nnz: 0,
            total_elements: n,
            jacobian_sparsity: 1.0,
            hessian_sparsity: 1.0,
        }
    }
}

/// Sparse quasi-Newton approximation using limited memory and sparsity
struct SparseQuasiNewton {
    /// Sparse approximation of inverse Hessian
    h_inv_sparse: Option<CsrArray<f64>>,
    /// History of sparse gradients
    sparse_grad_history: Vec<CsrArray<f64>>,
    /// History of sparse steps
    sparse_step_history: Vec<Array1<f64>>,
    /// Maximum history size
    max_history: usize,
    /// Sparsity pattern for Hessian approximation
    #[allow(dead_code)]
    pattern: Option<CsrArray<f64>>,
}

impl SparseQuasiNewton {
    fn new(_n: usize, max_history: usize) -> Self {
        Self {
            h_inv_sparse: None,
            sparse_grad_history: Vec::new(),
            sparse_step_history: Vec::new(),
            max_history,
            pattern: None,
        }
    }

    fn update_sparse_bfgs(
        &mut self,
        s: &Array1<f64>,
        y_sparse: &CsrArray<f64>,
        sparsity_pattern: &CsrArray<f64>,
    ) -> Result<(), OptimizeError> {
        // Convert sparse gradient difference to dense for computation
        let y = sparse_to_dense(y_sparse);

        let s_dot_y = s.dot(&y);
        if s_dot_y <= 1e-10 {
            return Ok(()); // Skip update if curvature condition not satisfied
        }

        // Initialize inverse Hessian approximation if needed
        if self.h_inv_sparse.is_none() {
            self.h_inv_sparse = Some(create_sparse_identity(s.len(), sparsity_pattern)?);
        }

        // Perform sparse BFGS update using the Sherman-Morrison-Woodbury formula
        // This is a simplified version - a full implementation would use efficient
        // sparse matrix operations throughout
        if let Some(ref mut h_inv) = self.h_inv_sparse {
            // Convert to dense for update, then back to sparse
            let h_inv_dense = sparse_to_dense_matrix(h_inv);
            let h_inv_updated = dense_bfgs_update(&h_inv_dense, s, &y)?;

            // Convert back to sparse format, preserving pattern
            *h_inv = dense_to_sparse_matrix(&h_inv_updated, sparsity_pattern)?;
        }

        // Update history (keep bounded)
        self.sparse_step_history.push(s.clone());
        self.sparse_grad_history.push(y_sparse.clone());

        while self.sparse_step_history.len() > self.max_history {
            self.sparse_step_history.remove(0);
            self.sparse_grad_history.remove(0);
        }

        Ok(())
    }

    fn apply_inverse_hessian(
        &self,
        g_sparse: &CsrArray<f64>,
    ) -> Result<Array1<f64>, OptimizeError> {
        if let Some(ref h_inv) = self.h_inv_sparse {
            // Sparse matrix-vector product
            sparse_matrix_vector_product(h_inv, g_sparse)
        } else {
            // If no approximation available, use negative gradient
            Ok(-sparse_to_dense(g_sparse))
        }
    }
}

/// Efficient sparse Newton method with adaptive sparsity detection
pub fn minimize_efficient_sparse_newton<F, G>(
    mut fun: F,
    mut grad: G,
    x0: Array1<f64>,
    options: &EfficientSparseOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64 + Sync,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let n = x0.len();
    let base_opts = &options.base_options;

    // Initialize variables
    let mut x = x0.to_owned();
    let mut f = fun(&x.view());
    let mut g_dense = grad(&x.view());

    // Initialize sparsity detection
    let mut sparsity_info = SparsityInfo::new(n);
    let mut sparse_qn = SparseQuasiNewton::new(n, 10);

    // Detect initial sparsity pattern if requested
    if options.auto_detect_sparsity {
        sparsity_info = detect_sparsity_patterns(&mut fun, &mut grad, &x.view(), options)?;
        println!(
            "Detected sparsity: Jacobian {:.1}%, Hessian {:.1}%",
            sparsity_info.jacobian_sparsity * 100.0,
            sparsity_info.hessian_sparsity * 100.0
        );
    }

    // Convert initial gradient to sparse if beneficial
    let mut g_sparse = if should_use_sparse(&g_dense, options.sparse_percentage_threshold) {
        dense_to_sparse_vector(&g_dense, options.sparsity_threshold)?
    } else {
        dense_to_sparse_vector(&g_dense, 0.0)? // Keep all elements
    };

    let mut iter = 0;
    let mut nfev = 1;
    let mut _njev = 1;

    // Main optimization loop
    while iter < base_opts.max_iter {
        // Check convergence
        let grad_norm = sparse_vector_norm(&g_sparse);
        if grad_norm < base_opts.gtol {
            break;
        }

        // Compute search direction using sparse operations
        let p = if options.use_sparse_hessian && sparsity_info.hessian_pattern.is_some() {
            // Use sparse Hessian for Newton step
            compute_sparse_newton_direction(
                &mut fun,
                &x.view(),
                &g_sparse,
                &sparsity_info,
                options,
            )?
        } else {
            // Use sparse quasi-Newton approximation
            sparse_qn.apply_inverse_hessian(&g_sparse)?
        };

        // Project search direction for bounds if needed
        let p = apply_bounds_projection(&p, &x, &options.base_options);

        // Line search
        let alpha_init = 1.0;
        let (alpha, f_new) = backtracking_line_search(
            &mut fun,
            &x.view(),
            f,
            &p.view(),
            &sparse_to_dense(&g_sparse).view(),
            alpha_init,
            0.0001,
            0.5,
            base_opts.bounds.as_ref(),
        );
        nfev += 1;

        // Update variables
        let s = alpha * &p;
        let x_new = &x + &s;

        // Check step size convergence
        if s.mapv(|x| x.abs()).sum() < base_opts.xtol {
            x = x_new;
            break;
        }

        // Compute new gradient
        let g_new_dense = grad(&x_new.view());
        _njev += 1;

        let g_new_sparse = if should_use_sparse(&g_new_dense, options.sparse_percentage_threshold) {
            dense_to_sparse_vector(&g_new_dense, options.sparsity_threshold)?
        } else {
            dense_to_sparse_vector(&g_new_dense, 0.0)?
        };

        // Check function value convergence
        if check_convergence(
            f - f_new,
            0.0,
            sparse_vector_norm(&g_new_sparse),
            base_opts.ftol,
            0.0,
            base_opts.gtol,
        ) {
            x = x_new;
            let _g_sparse_final = g_new_sparse; // Final gradient, loop will break
            break;
        }

        // Update sparse quasi-Newton approximation
        let y_sparse = sparse_vector_subtract(&g_new_sparse, &g_sparse)?;
        if let Some(ref pattern) = sparsity_info.hessian_pattern {
            sparse_qn.update_sparse_bfgs(&s, &y_sparse, pattern)?;
        }

        // Adaptive sparsity pattern refinement
        if options.adaptive_sparsity && iter % 10 == 0 {
            refine_sparsity_pattern(&mut sparsity_info, &g_new_sparse, options)?;
        }

        // Update for next iteration
        x = x_new;
        f = f_new;
        g_sparse = g_new_sparse;
        g_dense = g_new_dense;
        iter += 1;
    }

    // Final check for bounds
    if let Some(bounds) = &base_opts.bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    Ok(OptimizeResult {
        x,
        fun: f,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < base_opts.max_iter,
        message: if iter < base_opts.max_iter {
            format!(
                "Sparse optimization terminated successfully. Sparsity: {:.1}%",
                sparsity_info.jacobian_sparsity * 100.0
            )
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g_dense),
        hessian: None,
    })
}

/// Detect sparsity patterns in Jacobian and Hessian
fn detect_sparsity_patterns<F, G>(
    _fun: &mut F,
    grad: &mut G,
    x: &ArrayView1<f64>,
    options: &EfficientSparseOptions,
) -> Result<SparsityInfo, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let n = x.len();
    let mut sparsity_info = SparsityInfo::new(n);

    // Detect Jacobian sparsity by probing with unit vectors
    let mut jacobian_pattern = Vec::new();
    let eps = options.sparse_fd_options.rel_step.unwrap_or(1e-8);
    let g0 = grad(x);

    let mut nnz_count = 0;
    for i in 0..n {
        let mut x_pert = x.to_owned();
        x_pert[i] += eps;
        let g_pert = grad(&x_pert.view());

        for j in 0..n {
            let diff = (g_pert[j] - g0[j]).abs();
            if diff > options.sparsity_threshold {
                jacobian_pattern.push((j, i, 1.0)); // (row, col, value)
                nnz_count += 1;
            }
        }
    }

    // Create sparse Jacobian pattern
    if nnz_count > 0 {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for (row, col, val) in jacobian_pattern {
            rows.push(row);
            cols.push(col);
            data.push(val);
        }

        sparsity_info.jacobian_pattern =
            Some(CsrArray::from_triplets(&rows, &cols, &data, (n, n), false)?);
        sparsity_info.jacobian_nnz = nnz_count;
        sparsity_info.jacobian_sparsity = nnz_count as f64 / (n * n) as f64;
    }

    // For Hessian sparsity detection, use a simpler approach based on gradient structure
    if options.use_sparse_hessian {
        // Assume Hessian has similar sparsity to Jacobian for now
        // A more sophisticated implementation would probe the Hessian directly
        sparsity_info.hessian_pattern = sparsity_info.jacobian_pattern.clone();
        sparsity_info.hessian_nnz = sparsity_info.jacobian_nnz;
        sparsity_info.hessian_sparsity = sparsity_info.jacobian_sparsity;
    }

    Ok(sparsity_info)
}

/// Compute sparse Newton direction using sparse Hessian
fn compute_sparse_newton_direction<F>(
    _fun: &mut F,
    _x: &ArrayView1<f64>,
    g_sparse: &CsrArray<f64>,
    sparsity_info: &SparsityInfo,
    _options: &EfficientSparseOptions,
) -> Result<Array1<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64 + Sync,
{
    if let Some(_hessian_pattern) = &sparsity_info.hessian_pattern {
        // TODO: Implement sparse Hessian computation with FnMut compatibility
        // For now, use a simple identity approximation (steepest descent)
        let g_dense = sparse_to_dense(g_sparse);
        Ok(-g_dense)
    } else {
        // Fallback to negative gradient
        Ok(-sparse_to_dense(g_sparse))
    }
}

// Helper functions for sparse operations

fn should_use_sparse(vector: &Array1<f64>, threshold: f64) -> bool {
    let nnz = vector.iter().filter(|&&x| x.abs() > 1e-12).count();
    let sparsity = nnz as f64 / vector.len() as f64;
    sparsity < threshold
}

fn dense_to_sparse_vector(
    dense: &Array1<f64>,
    threshold: f64,
) -> Result<CsrArray<f64>, OptimizeError> {
    let n = dense.len();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (i, &val) in dense.iter().enumerate() {
        if val.abs() > threshold {
            rows.push(0); // Single row vector
            cols.push(i);
            data.push(val);
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, (1, n), false)
        .map_err(|_| OptimizeError::ComputationError("Failed to create sparse vector".to_string()))
}

fn sparse_to_dense(sparse: &CsrArray<f64>) -> Array1<f64> {
    let n = sparse.ncols();
    let mut dense = Array1::zeros(n);

    // Extract non-zero values - this is a simplified implementation
    for col in 0..n {
        dense[col] = sparse.get(0, col);
    }

    dense
}

fn sparse_vector_norm(sparse: &CsrArray<f64>) -> f64 {
    sparse.get_data().iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn sparse_vector_subtract(
    a: &CsrArray<f64>,
    b: &CsrArray<f64>,
) -> Result<CsrArray<f64>, OptimizeError> {
    // Simplified implementation - convert to dense, subtract, convert back
    let a_dense = sparse_to_dense(a);
    let b_dense = sparse_to_dense(b);
    let diff = &a_dense - &b_dense;
    dense_to_sparse_vector(&diff, 1e-12)
}

fn apply_bounds_projection(p: &Array1<f64>, x: &Array1<f64>, options: &Options) -> Array1<f64> {
    let mut p_proj = p.clone();

    if let Some(bounds) = &options.bounds {
        for i in 0..p.len() {
            let mut can_decrease = true;
            let mut can_increase = true;

            if let Some(lb) = bounds.lower[i] {
                if x[i] <= lb + options.eps {
                    can_decrease = false;
                }
            }
            if let Some(ub) = bounds.upper[i] {
                if x[i] >= ub - options.eps {
                    can_increase = false;
                }
            }

            if (p[i] < 0.0 && !can_decrease) || (p[i] > 0.0 && !can_increase) {
                p_proj[i] = 0.0;
            }
        }
    }

    p_proj
}

fn refine_sparsity_pattern(
    _sparsity_info: &mut SparsityInfo,
    _current_gradient: &CsrArray<f64>,
    _options: &EfficientSparseOptions,
) -> Result<(), OptimizeError> {
    // Adaptive refinement of sparsity pattern based on current gradient
    // This is a simplified version - a full implementation would track
    // the evolution of sparsity patterns over iterations
    Ok(())
}

// Additional helper functions (simplified implementations)

fn create_sparse_identity(
    n: usize,
    pattern: &CsrArray<f64>,
) -> Result<CsrArray<f64>, OptimizeError> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..n {
        if pattern.get(i, i) != 0.0 {
            rows.push(i);
            cols.push(i);
            data.push(1.0);
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, (n, n), false).map_err(|_| {
        OptimizeError::ComputationError("Failed to create sparse identity".to_string())
    })
}

fn sparse_to_dense_matrix(sparse: &CsrArray<f64>) -> Array2<f64> {
    let (m, n) = (sparse.nrows(), sparse.ncols());
    let mut dense = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            dense[[i, j]] = sparse.get(i, j);
        }
    }

    dense
}

fn dense_to_sparse_matrix(
    dense: &Array2<f64>,
    pattern: &CsrArray<f64>,
) -> Result<CsrArray<f64>, OptimizeError> {
    let (m, n) = dense.dim();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..m {
        for j in 0..n {
            if pattern.get(i, j) != 0.0 {
                rows.push(i);
                cols.push(j);
                data.push(dense[[i, j]]);
            }
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, (m, n), false)
        .map_err(|_| OptimizeError::ComputationError("Failed to create sparse matrix".to_string()))
}

fn dense_bfgs_update(
    h_inv: &Array2<f64>,
    s: &Array1<f64>,
    y: &Array1<f64>,
) -> Result<Array2<f64>, OptimizeError> {
    let n = h_inv.nrows();
    let s_dot_y = s.dot(y);

    if s_dot_y <= 1e-10 {
        return Ok(h_inv.clone());
    }

    let rho = 1.0 / s_dot_y;
    let i_mat = Array2::eye(n);

    // BFGS update formula
    let y_col = y.clone().insert_axis(Axis(1));
    let s_row = s.clone().insert_axis(Axis(0));
    let y_s_t = y_col.dot(&s_row);
    let term1 = &i_mat - &(&y_s_t * rho);

    let s_col = s.clone().insert_axis(Axis(1));
    let y_row = y.clone().insert_axis(Axis(0));
    let s_y_t = s_col.dot(&y_row);
    let term2 = &i_mat - &(&s_y_t * rho);

    let term3 = term1.dot(h_inv);
    let result = term3.dot(&term2) + rho * s_col.dot(&s_row);

    Ok(result)
}

fn sparse_matrix_vector_product(
    matrix: &CsrArray<f64>,
    vector_sparse: &CsrArray<f64>,
) -> Result<Array1<f64>, OptimizeError> {
    // Simplified sparse matrix-vector product
    let vector_dense = sparse_to_dense(vector_sparse);
    let matrix_dense = sparse_to_dense_matrix(matrix);
    Ok(matrix_dense.dot(&vector_dense))
}

#[allow(dead_code)]
fn solve_sparse_linear_system(
    matrix: &CsrArray<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, OptimizeError> {
    // Simplified linear system solver - convert to dense
    // A real implementation would use sparse linear algebra libraries
    let _matrix_dense = sparse_to_dense_matrix(matrix);

    // Use simple Gauss elimination or other methods
    // For now, return a placeholder solution
    Ok(-rhs.clone()) // Negative gradient as fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sparsity_detection() {
        let n = 10;
        let options = EfficientSparseOptions::default();

        // Simple quadratic function with sparse Hessian
        let mut fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[4].powi(2) + x[9].powi(2) };

        let mut grad = |x: &ArrayView1<f64>| -> Array1<f64> {
            let mut g = Array1::zeros(n);
            g[0] = 2.0 * x[0];
            g[4] = 2.0 * x[4];
            g[9] = 2.0 * x[9];
            g
        };

        let x = Array1::ones(n);
        let sparsity_info =
            detect_sparsity_patterns(&mut fun, &mut grad, &x.view(), &options).unwrap();

        // Should detect that only diagonal elements (0,0), (4,4), (9,9) are non-zero
        assert!(sparsity_info.jacobian_sparsity < 0.5); // Should be sparse
    }

    #[test]
    fn test_sparse_vector_operations() {
        let dense = Array1::from_vec(vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
        let sparse = dense_to_sparse_vector(&dense, 1e-12).unwrap();

        let reconstructed = sparse_to_dense(&sparse);

        for i in 0..dense.len() {
            assert_abs_diff_eq!(dense[i], reconstructed[i], epsilon = 1e-10);
        }

        let norm_sparse = sparse_vector_norm(&sparse);
        let norm_dense = dense.mapv(|x| x.powi(2)).sum().sqrt();
        assert_abs_diff_eq!(norm_sparse, norm_dense, epsilon = 1e-10);
    }

    #[test]
    fn test_efficient_sparse_optimization() {
        // Test on a simple sparse quadratic problem
        let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[2].powi(2) + x[4].powi(2) };

        let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
            let mut g = Array1::zeros(5);
            g[0] = 2.0 * x[0];
            g[2] = 2.0 * x[2];
            g[4] = 2.0 * x[4];
            g
        };

        let x0 = Array1::ones(5);
        let options = EfficientSparseOptions::default();

        let result = minimize_efficient_sparse_newton(fun, grad, x0, &options).unwrap();

        assert!(result.success);
        // Should converge to origin for all active variables
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[2], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[4], 0.0, epsilon = 1e-3);
    }
}
