//! Parallel computation utilities for optimization algorithms
//!
//! This module provides parallel implementations of computationally intensive
//! operations used in optimization algorithms, including:
//! - Parallel function evaluations
//! - Parallel gradient approximations
//! - Multi-start optimization with parallel workers
//!
//! # Example
//!
//! ```
//! use scirs2_optimize::parallel::{parallel_finite_diff_gradient, ParallelOptions};
//! use ndarray::{array, ArrayView1};
//!
//! fn objective(x: &ArrayView1<f64>) -> f64 {
//!     x.iter().map(|&xi| xi.powi(2)).sum()
//! }
//!
//! let x = array![1.0, 2.0, 3.0];
//! let options = ParallelOptions::default();
//! let gradient = parallel_finite_diff_gradient(objective, x.view(), &options);
//! ```

use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

/// Options for parallel computation
#[derive(Debug, Clone)]
pub struct ParallelOptions {
    /// Number of worker threads (None = use rayon default)
    pub num_workers: Option<usize>,

    /// Minimum problem size to enable parallelization
    pub min_parallel_size: usize,

    /// Chunk size for parallel operations
    pub chunk_size: usize,

    /// Enable parallel function evaluations
    pub parallel_evaluations: bool,

    /// Enable parallel gradient computation
    pub parallel_gradient: bool,
}

impl Default for ParallelOptions {
    fn default() -> Self {
        ParallelOptions {
            num_workers: None,
            min_parallel_size: 10,
            chunk_size: 1,
            parallel_evaluations: true,
            parallel_gradient: true,
        }
    }
}

/// Parallel finite difference gradient approximation
///
/// Computes the gradient using parallel function evaluations when
/// the dimension is large enough to benefit from parallelization.
pub fn parallel_finite_diff_gradient<F>(
    f: F,
    x: ArrayView1<f64>,
    options: &ParallelOptions,
) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    let n = x.len();
    let eps = 1e-8;

    // For small problems, use sequential computation
    if n < options.min_parallel_size || !options.parallel_gradient {
        return sequential_finite_diff_gradient(f, x, eps);
    }

    // Configure thread pool if specified
    if let Some(num_workers) = options.num_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .unwrap()
            .install(|| compute_parallel_gradient(&f, x, eps))
    } else {
        compute_parallel_gradient(&f, x, eps)
    }
}

/// Sequential finite difference gradient (fallback)
fn sequential_finite_diff_gradient<F>(f: F, x: ArrayView1<f64>, eps: f64) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    let fx = f(&x);

    for i in 0..n {
        let mut x_plus = x.to_owned();
        x_plus[i] += eps;
        let fx_plus = f(&x_plus.view());
        grad[i] = (fx_plus - fx) / eps;
    }

    grad
}

/// Compute gradient in parallel
fn compute_parallel_gradient<F>(f: &F, x: ArrayView1<f64>, eps: f64) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    let n = x.len();
    let fx = f(&x);

    // Parallel computation of gradient components
    let grad_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut x_plus = x.to_owned();
            x_plus[i] += eps;
            let fx_plus = f(&x_plus.view());
            (fx_plus - fx) / eps
        })
        .collect();

    Array1::from_vec(grad_vec)
}

/// Parallel evaluation of multiple points
///
/// Evaluates the objective function at multiple points in parallel.
pub fn parallel_evaluate_batch<F>(
    f: F,
    points: &[Array1<f64>],
    options: &ParallelOptions,
) -> Vec<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    if points.len() < options.min_parallel_size || !options.parallel_evaluations {
        // Sequential evaluation for small batches
        points.iter().map(|x| f(&x.view())).collect()
    } else {
        // Parallel evaluation
        points.par_iter().map(|x| f(&x.view())).collect()
    }
}

/// Parallel multi-start optimization
///
/// Runs multiple optimization instances from different starting points in parallel.
#[allow(dead_code)] // Bounds will be used in future implementation
pub struct ParallelMultiStart<F> {
    objective: F,
    bounds: Vec<(f64, f64)>,
    options: ParallelOptions,
}

impl<F> ParallelMultiStart<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync + Send,
{
    /// Create a new parallel multi-start optimizer
    pub fn new(objective: F, bounds: Vec<(f64, f64)>, options: ParallelOptions) -> Self {
        ParallelMultiStart {
            objective,
            bounds,
            options,
        }
    }

    /// Run optimization from multiple starting points
    pub fn run<O>(&self, starting_points: Vec<Array1<f64>>, optimizer: O) -> Vec<OptimizationResult>
    where
        O: Fn(&Array1<f64>, &F) -> OptimizationResult + Sync,
    {
        if starting_points.len() < self.options.min_parallel_size {
            // Sequential execution for small number of starts
            starting_points
                .iter()
                .map(|x0| optimizer(x0, &self.objective))
                .collect()
        } else {
            // Parallel execution
            starting_points
                .par_iter()
                .map(|x0| optimizer(x0, &self.objective))
                .collect()
        }
    }

    /// Find the best result among all runs
    pub fn best_result(results: &[OptimizationResult]) -> Option<&OptimizationResult> {
        results
            .iter()
            .min_by(|a, b| a.function_value.partial_cmp(&b.function_value).unwrap())
    }
}

/// Result from an optimization run
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub x: Array1<f64>,
    pub function_value: f64,
    pub iterations: usize,
    pub success: bool,
}

/// Parallel computation of Hessian matrix using finite differences
pub fn parallel_finite_diff_hessian<F>(
    f: &F,
    x: ArrayView1<f64>,
    gradient: Option<&Array1<f64>>,
    options: &ParallelOptions,
) -> Array1<f64>
// Returns upper triangle in row-major order
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    let n = x.len();
    let eps = 1e-8;

    // For small problems, use sequential computation
    if n < options.min_parallel_size {
        return sequential_finite_diff_hessian(f, x, gradient, eps);
    }

    // Compute gradient if not provided
    let _grad = match gradient {
        Some(g) => g.clone(),
        None => parallel_finite_diff_gradient(f, x, options),
    };

    // Number of elements in upper triangle (including diagonal)
    let num_elements = n * (n + 1) / 2;

    // Parallel computation of Hessian elements
    let hessian_elements: Vec<f64> = (0..num_elements)
        .into_par_iter()
        .map(|idx| {
            // Convert linear index to (i, j) coordinates
            let (i, j) = index_to_upper_triangle(idx, n);

            if i == j {
                // Diagonal element: ∂²f/∂xᵢ²
                let mut x_plus = x.to_owned();
                let mut x_minus = x.to_owned();
                x_plus[i] += eps;
                x_minus[i] -= eps;

                let f_plus = f(&x_plus.view());
                let f_minus = f(&x_minus.view());
                let f_center = f(&x);

                (f_plus - 2.0 * f_center + f_minus) / (eps * eps)
            } else {
                // Off-diagonal element: ∂²f/∂xᵢ∂xⱼ
                let mut x_pp = x.to_owned();
                let mut x_pm = x.to_owned();
                let mut x_mp = x.to_owned();
                let mut x_mm = x.to_owned();

                x_pp[i] += eps;
                x_pp[j] += eps;
                x_pm[i] += eps;
                x_pm[j] -= eps;
                x_mp[i] -= eps;
                x_mp[j] += eps;
                x_mm[i] -= eps;
                x_mm[j] -= eps;

                let f_pp = f(&x_pp.view());
                let f_pm = f(&x_pm.view());
                let f_mp = f(&x_mp.view());
                let f_mm = f(&x_mm.view());

                (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
            }
        })
        .collect();

    Array1::from_vec(hessian_elements)
}

/// Sequential Hessian computation (fallback)
fn sequential_finite_diff_hessian<F>(
    f: &F,
    x: ArrayView1<f64>,
    _gradient: Option<&Array1<f64>>,
    eps: f64,
) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let num_elements = n * (n + 1) / 2;
    let mut hessian = Array1::zeros(num_elements);

    for idx in 0..num_elements {
        let (i, j) = index_to_upper_triangle(idx, n);

        if i == j {
            // Diagonal element
            let mut x_plus = x.to_owned();
            let mut x_minus = x.to_owned();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let f_plus = f(&x_plus.view());
            let f_minus = f(&x_minus.view());
            let f_center = f(&x);

            hessian[idx] = (f_plus - 2.0 * f_center + f_minus) / (eps * eps);
        } else {
            // Off-diagonal element
            let mut x_pp = x.to_owned();
            let mut x_pm = x.to_owned();
            let mut x_mp = x.to_owned();
            let mut x_mm = x.to_owned();

            x_pp[i] += eps;
            x_pp[j] += eps;
            x_pm[i] += eps;
            x_pm[j] -= eps;
            x_mp[i] -= eps;
            x_mp[j] += eps;
            x_mm[i] -= eps;
            x_mm[j] -= eps;

            let f_pp = f(&x_pp.view());
            let f_pm = f(&x_pm.view());
            let f_mp = f(&x_mp.view());
            let f_mm = f(&x_mm.view());

            hessian[idx] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
        }
    }

    hessian
}

/// Convert linear index to (i, j) coordinates in upper triangle
fn index_to_upper_triangle(idx: usize, _n: usize) -> (usize, usize) {
    // Find row i such that i*(i+1)/2 <= idx < (i+1)*(i+2)/2
    let mut i = 0;
    let mut cumsum = 0;

    while cumsum + i < idx {
        i += 1;
        cumsum += i;
    }

    let j = idx - cumsum;
    (j, i)
}

/// Parallel line search
///
/// Evaluates multiple step sizes in parallel to find the best one.
pub fn parallel_line_search<F>(
    f: F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    step_sizes: &[f64],
    options: &ParallelOptions,
) -> (f64, f64)
// Returns (best_step, best_value)
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    let evaluations: Vec<(f64, f64)> = if step_sizes.len() < options.min_parallel_size {
        // Sequential evaluation
        step_sizes
            .iter()
            .map(|&alpha| {
                let x_new = x + alpha * direction;
                let value = f(&x_new.view());
                (alpha, value)
            })
            .collect()
    } else {
        // Parallel evaluation
        step_sizes
            .par_iter()
            .map(|&alpha| {
                let x_new = x + alpha * direction;
                let value = f(&x_new.view());
                (alpha, value)
            })
            .collect()
    };

    // Find the best step size
    evaluations
        .into_iter()
        .min_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
        .unwrap_or((0.0, f64::INFINITY))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn quadratic(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    #[test]
    fn test_parallel_gradient() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let options = ParallelOptions::default();

        let grad_parallel = parallel_finite_diff_gradient(quadratic, x.view(), &options);
        let grad_sequential = sequential_finite_diff_gradient(quadratic, x.view(), 1e-8);

        // Check that parallel and sequential give similar results
        for i in 0..x.len() {
            assert!((grad_parallel[i] - grad_sequential[i]).abs() < 1e-10);
            // For quadratic function, gradient should be 2*x
            assert!((grad_parallel[i] - 2.0 * x[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parallel_batch_evaluation() {
        let points = vec![
            array![1.0, 2.0],
            array![3.0, 4.0],
            array![5.0, 6.0],
            array![7.0, 8.0],
        ];

        let options = ParallelOptions {
            min_parallel_size: 2,
            ..Default::default()
        };

        let values = parallel_evaluate_batch(quadratic, &points, &options);

        // Verify results
        for (i, point) in points.iter().enumerate() {
            let expected = quadratic(&point.view());
            assert_eq!(values[i], expected);
        }
    }

    #[test]
    fn test_parallel_line_search() {
        let x = array![1.0, 1.0];
        let direction = array![-1.0, -1.0];
        let step_sizes: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();

        let options = ParallelOptions::default();
        let (best_step, best_value) =
            parallel_line_search(quadratic, &x, &direction, &step_sizes, &options);

        // For quadratic function starting at (1,1) going in direction (-1,-1),
        // the minimum should be at step size 1.0
        assert!((best_step - 1.0).abs() < 0.1);
        assert!(best_value < 0.1);
    }
}
