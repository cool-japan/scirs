//! Sparse optimization algorithms for high-dimensional problems
//!
//! This module provides optimization algorithms that leverage sparse numerical
//! differentiation and matrix operations for efficient handling of high-dimensional
//! optimization problems.

use crate::error::OptimizeError;
use crate::sparse_numdiff::{sparse_jacobian, SparseFiniteDiffOptions};
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::check_convergence;
use crate::unconstrained::Options;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_sparse::{csr_array::CsrArray, sparray::SparseArray};

/// Options for sparse optimization algorithms
#[derive(Debug, Clone)]
pub struct SparseOptimizationOptions {
    /// Basic optimization options
    pub base_options: Options,
    /// Sparse finite difference options
    pub sparse_options: SparseFiniteDiffOptions,
    /// Known sparsity pattern for gradient/Jacobian (if None, auto-detected)
    pub sparsity_pattern: Option<CsrArray<f64>>,
    /// Whether to use sparse matrix operations throughout
    pub use_sparse_operations: bool,
    /// Memory threshold for switching to sparse operations (number of variables)
    pub sparse_threshold: usize,
}

impl Default for SparseOptimizationOptions {
    fn default() -> Self {
        Self {
            base_options: Options::default(),
            sparse_options: SparseFiniteDiffOptions::default(),
            sparsity_pattern: None,
            use_sparse_operations: true,
            sparse_threshold: 100, // Use sparse operations for 100+ variables
        }
    }
}

/// Sparse BFGS algorithm for large-scale optimization
pub fn minimize_sparse_bfgs<F, S>(
    fun: F,
    x0: Array1<f64>,
    options: &SparseOptimizationOptions,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S + Clone + Sync,
    S: Into<f64> + Clone,
{
    let n = x0.len();
    let base_opts = &options.base_options;
    let sparse_opts = &options.sparse_options;

    // Decide whether to use sparse operations based on problem size
    let use_sparse = options.use_sparse_operations && n >= options.sparse_threshold;

    // Initialize variables
    let mut x = x0.to_owned();
    let bounds = base_opts.bounds.as_ref();

    // Ensure initial point is within bounds
    if let Some(bounds) = bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    let mut f = fun(&x.view()).into();

    // Calculate initial gradient
    let mut g = if use_sparse && options.sparsity_pattern.is_some() {
        // Use sparse gradient computation
        compute_sparse_gradient(&fun, &x.view(), sparse_opts)?
    } else {
        // Use dense gradient computation - create a mutable wrapper
        let mut fun_wrapper = |x_val: &ArrayView1<f64>| fun(x_val);
        crate::unconstrained::utils::finite_difference_gradient(
            &mut fun_wrapper,
            &x.view(),
            base_opts.eps,
        )?
    };

    // Initialize inverse Hessian approximation
    let mut h_inv = if use_sparse {
        // For sparse case, we could use L-BFGS-like limited memory approach
        // or a sparse approximation of the inverse Hessian
        None
    } else {
        Some(Array2::eye(n))
    };

    // For sparse case, use L-BFGS-like storage
    let mut s_history: Vec<Array1<f64>> = Vec::new();
    let mut y_history: Vec<Array1<f64>> = Vec::new();
    let max_history = 10; // L-BFGS memory

    // Initialize counters
    let mut iter = 0;
    let mut nfev = 1; // Initial function evaluation

    // Main loop
    while iter < base_opts.max_iter {
        // Check convergence on gradient
        if g.mapv(|gi| gi.abs()).sum() < base_opts.gtol {
            break;
        }

        // Compute search direction
        let mut p = if use_sparse && !s_history.is_empty() {
            // Use L-BFGS two-loop recursion for sparse case
            compute_lbfgs_direction(&g, &s_history, &y_history)
        } else if let Some(ref h) = h_inv {
            // Use dense BFGS
            -h.dot(&g)
        } else {
            // Fallback: steepest descent
            -&g
        };

        // Project search direction for bounded optimization
        if let Some(bounds) = bounds {
            for i in 0..n {
                let mut can_decrease = true;
                let mut can_increase = true;

                // Check if at boundary
                if let Some(lb) = bounds.lower[i] {
                    if x[i] <= lb + base_opts.eps {
                        can_decrease = false;
                    }
                }
                if let Some(ub) = bounds.upper[i] {
                    if x[i] >= ub - base_opts.eps {
                        can_increase = false;
                    }
                }

                // Project gradient component
                if (g[i] > 0.0 && !can_decrease) || (g[i] < 0.0 && !can_increase) {
                    p[i] = 0.0;
                }
            }

            // If no movement is possible, we're at a constrained optimum
            if p.mapv(|pi| pi.abs()).sum() < 1e-10 {
                break;
            }
        }

        // Line search
        let alpha_init = 1.0;
        let mut fun_wrapper = |x_val: &ArrayView1<f64>| fun(x_val);
        let (alpha, f_new) = backtracking_line_search(
            &mut fun_wrapper,
            &x.view(),
            f,
            &p.view(),
            &g.view(),
            alpha_init,
            0.0001,
            0.5,
            bounds,
        );

        nfev += 1;

        // Update position
        let s = alpha * &p;
        let x_new = &x + &s;

        // Check step size convergence
        if s.mapv(|si| si.abs()).sum() < base_opts.xtol {
            x = x_new;
            break;
        }

        // Calculate new gradient
        let g_new = if use_sparse && options.sparsity_pattern.is_some() {
            compute_sparse_gradient(&fun, &x_new.view(), sparse_opts)?
        } else {
            let mut fun_wrapper = |x_val: &ArrayView1<f64>| fun(x_val);
            crate::unconstrained::utils::finite_difference_gradient(
                &mut fun_wrapper,
                &x_new.view(),
                base_opts.eps,
            )?
        };

        nfev += n; // Approximate function evaluations for gradient

        // Gradient difference
        let y = &g_new - &g;

        // Check convergence on function value
        if check_convergence(
            f - f_new,
            0.0,
            g_new.mapv(|gi| gi.abs()).sum(),
            base_opts.ftol,
            0.0,
            base_opts.gtol,
        ) {
            x = x_new;
            g = g_new;
            break;
        }

        // Update approximation
        let s_dot_y = s.dot(&y);
        if s_dot_y > 1e-10 {
            if use_sparse {
                // Update L-BFGS history
                s_history.push(s.clone());
                y_history.push(y.clone());

                // Keep only recent history
                if s_history.len() > max_history {
                    s_history.remove(0);
                    y_history.remove(0);
                }
            } else if let Some(ref mut h) = h_inv {
                // Dense BFGS update
                let rho = 1.0 / s_dot_y;
                let i_mat = Array2::eye(n);

                let y_col = y.clone().insert_axis(Axis(1));
                let s_row = s.clone().insert_axis(Axis(0));
                let y_s_t = y_col.dot(&s_row);
                let term1 = &i_mat - &(&y_s_t * rho);

                let s_col = s.clone().insert_axis(Axis(1));
                let y_row = y.clone().insert_axis(Axis(0));
                let s_y_t = s_col.dot(&y_row);
                let term2 = &i_mat - &(&s_y_t * rho);

                let term3 = term1.dot(h);
                *h = term3.dot(&term2) + rho * s_col.dot(&s_row);
            }
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;
        iter += 1;
    }

    // Final check for bounds
    if let Some(bounds) = bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    // Use original function for final value
    let final_fun = fun(&x.view());

    Ok(OptimizeResult {
        x,
        fun: final_fun,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < base_opts.max_iter,
        message: if iter < base_opts.max_iter {
            "Sparse optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// Compute sparse gradient using sparse numerical differentiation
pub fn compute_sparse_gradient<F, S>(
    fun: &F,
    x: &ArrayView1<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<Array1<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S + Clone + Sync,
    S: Into<f64> + Clone,
{
    // Create wrapper that returns Array1<f64> instead of scalar
    let wrapper_fn = |x_val: &ArrayView1<f64>| -> Array1<f64> {
        let result: f64 = fun(x_val).into();
        Array1::from(vec![result])
    };

    // Create a 1x1 sparsity pattern (gradient of scalar function)
    let mut rows = vec![0];
    let mut cols = vec![];
    let mut data = vec![];

    for i in 0..x.len() {
        rows.push(0);
        cols.push(i);
        data.push(1.0);
    }
    rows.remove(0); // Remove the first element we added

    let sparsity =
        CsrArray::from_triplets(&rows, &cols, &data, (1, x.len()), false).map_err(|e| {
            OptimizeError::ValueError(format!("Failed to create sparsity pattern: {}", e))
        })?;

    // Compute sparse Jacobian (which gives us the gradient)
    let jacobian = sparse_jacobian(wrapper_fn, x, None, Some(&sparsity), Some(options.clone()))?;

    // Extract the gradient (first and only row of the Jacobian)
    let grad_array = jacobian.to_array();
    Ok(grad_array.row(0).to_owned())
}

/// Compute gradient using sparse finite differences with a scalar function
fn finite_difference_gradient_sparse<F, S>(
    fun: &mut F,
    x: &ArrayView1<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<Array1<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let n = x.len();
    let mut grad = Array1::<f64>::zeros(n);
    let step = options.rel_step.unwrap_or(1e-8);

    // Compute f0
    let f0 = fun(x).into();

    // Use finite differences to compute gradient
    let mut x_perturbed = x.to_owned();
    for i in 0..n {
        let h = step * (1.0 + x[i].abs());
        x_perturbed[i] = x[i] + h;

        let f_plus = fun(&x_perturbed.view()).into();

        if !f_plus.is_finite() {
            return Err(OptimizeError::ComputationError(
                "Function returned non-finite value during gradient computation".to_string(),
            ));
        }

        grad[i] = (f_plus - f0) / h;
        x_perturbed[i] = x[i]; // Reset
    }

    Ok(grad)
}

/// Compute L-BFGS search direction using two-loop recursion
fn compute_lbfgs_direction(
    g: &Array1<f64>,
    s_history: &[Array1<f64>],
    y_history: &[Array1<f64>],
) -> Array1<f64> {
    let m = s_history.len();
    if m == 0 {
        return -g; // Steepest descent if no history
    }

    let mut q = g.clone();
    let mut alpha = vec![0.0; m];

    // First loop: backward through history
    for i in (0..m).rev() {
        let rho_i = 1.0 / y_history[i].dot(&s_history[i]);
        alpha[i] = rho_i * s_history[i].dot(&q);
        q = &q - alpha[i] * &y_history[i];
    }

    // Apply initial Hessian approximation (simple scaling)
    let gamma = if m > 0 {
        s_history[m - 1].dot(&y_history[m - 1]) / y_history[m - 1].dot(&y_history[m - 1])
    } else {
        1.0
    };
    let mut r = gamma * q;

    // Second loop: forward through history
    for i in 0..m {
        let rho_i = 1.0 / y_history[i].dot(&s_history[i]);
        let beta = rho_i * y_history[i].dot(&r);
        r = &r + (alpha[i] - beta) * &s_history[i];
    }

    -r // Search direction
}

/// Auto-detect sparsity pattern by evaluating the function at multiple points
pub fn auto_detect_sparsity<F, S>(
    fun: F,
    x: &ArrayView1<f64>,
    num_samples: Option<usize>,
    threshold: Option<f64>,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    let n = x.len();
    let num_samples = num_samples.unwrap_or(std::cmp::min(n, 10));
    let threshold = threshold.unwrap_or(1e-10);

    // Sample points around x
    let mut sparsity_pattern = Array2::<f64>::zeros((1, n));

    for _sample in 0..num_samples {
        // Create a small random perturbation
        let mut x_pert = x.to_owned();
        for i in 0..n {
            x_pert[i] += 1e-6 * (rand::random::<f64>() - 0.5);
        }

        // Compute gradient at this point
        let options = SparseFiniteDiffOptions::default();
        if let Ok(grad) =
            finite_difference_gradient_sparse(&mut fun.clone(), &x_pert.view(), &options)
        {
            for i in 0..n {
                if grad[i].abs() > threshold {
                    sparsity_pattern[[0, i]] = 1.0;
                }
            }
        }
    }

    // Convert to sparse format
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for j in 0..n {
        if sparsity_pattern[[0, j]] > 0.0 {
            rows.push(0);
            cols.push(j);
            data.push(1.0);
        }
    }

    if rows.is_empty() {
        // If no sparsity detected, assume dense
        for j in 0..n {
            rows.push(0);
            cols.push(j);
            data.push(1.0);
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, (1, n), false)
        .map_err(|e| OptimizeError::ValueError(format!("Failed to create sparsity pattern: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sparse_bfgs_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 {
            // Simple quadratic: f(x) = sum(x_i^2)
            x.mapv(|xi| xi.powi(2)).sum()
        };

        let n = 50; // Medium-sized problem
        let x0 = Array1::ones(n);
        let mut options = SparseOptimizationOptions::default();
        options.sparse_threshold = 10; // Force sparse operations

        let result = minimize_sparse_bfgs(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // Should converge to origin
        for i in 0..n {
            assert_abs_diff_eq!(result.x[i], 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_auto_detect_sparsity() {
        let sparse_fn = |x: &ArrayView1<f64>| -> f64 {
            // Only depends on first two variables
            x[0].powi(2) + x[1].powi(2)
        };

        let x = Array1::zeros(10);
        let sparsity = auto_detect_sparsity(sparse_fn, &x.view(), Some(5), Some(1e-8)).unwrap();

        // Should detect that only first two variables matter
        let dense = sparsity.to_array();
        assert!(dense[[0, 0]] > 0.0);
        assert!(dense[[0, 1]] > 0.0);
        // Rest should be zero (or close to zero)
        for i in 2..10 {
            assert!(dense[[0, i]].abs() < 1e-8);
        }
    }
}
