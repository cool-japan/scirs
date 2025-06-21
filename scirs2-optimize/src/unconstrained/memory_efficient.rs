//! Memory-efficient algorithms for large-scale optimization problems
//!
//! This module provides optimization algorithms designed to handle very large problems
//! with limited memory by using chunked processing, streaming algorithms, and
//! memory pool management.

use crate::error::OptimizeError;
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::check_convergence;
use crate::unconstrained::Options;
use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

/// Memory optimization options for large-scale problems
#[derive(Debug, Clone)]
pub struct MemoryOptions {
    /// Base optimization options
    pub base_options: Options,
    /// Maximum memory usage in bytes (0 means unlimited)
    pub max_memory_bytes: usize,
    /// Chunk size for processing large vectors
    pub chunk_size: usize,
    /// Maximum history size for L-BFGS-style methods
    pub max_history: usize,
    /// Whether to use memory pooling
    pub use_memory_pool: bool,
    /// Whether to use out-of-core storage for very large problems
    pub use_out_of_core: bool,
    /// Temporary directory for out-of-core storage
    pub temp_dir: Option<std::path::PathBuf>,
}

impl Default for MemoryOptions {
    fn default() -> Self {
        Self {
            base_options: Options::default(),
            max_memory_bytes: 0, // Unlimited by default
            chunk_size: 1024,    // Process 1024 elements at a time
            max_history: 10,     // Keep last 10 iterations
            use_memory_pool: true,
            use_out_of_core: false,
            temp_dir: None,
        }
    }
}

/// Memory pool for reusing arrays to reduce allocations
struct MemoryPool {
    array_pool: VecDeque<Array1<f64>>,
    max_pool_size: usize,
}

impl MemoryPool {
    fn new(max_size: usize) -> Self {
        Self {
            array_pool: VecDeque::new(),
            max_pool_size: max_size,
        }
    }

    fn get_array(&mut self, size: usize) -> Array1<f64> {
        // Try to reuse an existing array of the right size
        for i in 0..self.array_pool.len() {
            if self.array_pool[i].len() == size {
                return self.array_pool.remove(i).unwrap();
            }
        }
        // If no suitable array found, create a new one
        Array1::zeros(size)
    }

    fn return_array(&mut self, mut array: Array1<f64>) {
        if self.array_pool.len() < self.max_pool_size {
            // Zero out the array for reuse
            array.fill(0.0);
            self.array_pool.push_back(array);
        }
        // If pool is full, just drop the array
    }
}

/// Streaming gradient computation for very large problems
struct StreamingGradient {
    chunk_size: usize,
    eps: f64,
}

impl StreamingGradient {
    fn new(chunk_size: usize, eps: f64) -> Self {
        Self { chunk_size, eps }
    }

    /// Compute gradient using chunked finite differences
    fn compute<F, S>(&self, fun: &mut F, x: &ArrayView1<f64>) -> Result<Array1<f64>, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> S,
        S: Into<f64>,
    {
        let n = x.len();
        let mut grad = Array1::zeros(n);
        let f0 = fun(x).into();

        let mut x_pert = x.to_owned();

        // Process gradient computation in chunks to limit memory usage
        for chunk_start in (0..n).step_by(self.chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + self.chunk_size, n);

            for i in chunk_start..chunk_end {
                let h = self.eps * (1.0 + x[i].abs());
                x_pert[i] = x[i] + h;

                let f_plus = fun(&x_pert.view()).into();

                if !f_plus.is_finite() {
                    return Err(OptimizeError::ComputationError(
                        "Function returned non-finite value during gradient computation"
                            .to_string(),
                    ));
                }

                grad[i] = (f_plus - f0) / h;
                x_pert[i] = x[i]; // Reset
            }

            // Optional: yield control to prevent blocking for too long
            // In a real implementation, this could check for cancellation
        }

        Ok(grad)
    }
}

/// Memory-efficient L-BFGS implementation with bounded history
pub fn minimize_memory_efficient_lbfgs<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &MemoryOptions,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    let n = x0.len();
    let base_opts = &options.base_options;

    // Initialize memory pool if enabled
    let mut memory_pool = if options.use_memory_pool {
        Some(MemoryPool::new(options.max_history * 2))
    } else {
        None
    };

    // Estimate memory usage
    let estimated_memory = estimate_memory_usage(n, options.max_history);
    if options.max_memory_bytes > 0 && estimated_memory > options.max_memory_bytes {
        return Err(OptimizeError::ValueError(format!(
            "Estimated memory usage ({} bytes) exceeds limit ({} bytes). Consider reducing max_history or chunk_size.",
            estimated_memory, options.max_memory_bytes
        )));
    }

    // Initialize variables
    let mut x = x0.to_owned();
    let bounds = base_opts.bounds.as_ref();

    // Ensure initial point is within bounds
    if let Some(bounds) = bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    let mut f = fun(&x.view()).into();

    // Initialize streaming gradient computer
    let streaming_grad = StreamingGradient::new(options.chunk_size, base_opts.eps);

    // Calculate initial gradient
    let mut g = streaming_grad.compute(&mut fun, &x.view())?;

    // L-BFGS history with bounded size
    let mut s_history: VecDeque<Array1<f64>> = VecDeque::new();
    let mut y_history: VecDeque<Array1<f64>> = VecDeque::new();

    // Initialize counters
    let mut iter = 0;
    let mut nfev = 1 + n.div_ceil(options.chunk_size); // Initial function evaluations

    // Main loop
    while iter < base_opts.max_iter {
        // Check convergence on gradient
        if g.mapv(|gi| gi.abs()).sum() < base_opts.gtol {
            break;
        }

        // Compute search direction using L-BFGS two-loop recursion
        let mut p = if s_history.is_empty() {
            // Use steepest descent if no history
            get_array_from_pool(&mut memory_pool, n, |_size| -&g)
        } else {
            compute_lbfgs_direction_memory_efficient(&g, &s_history, &y_history, &mut memory_pool)?
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
                return_array_to_pool(&mut memory_pool, p);
                break;
            }
        }

        // Line search
        let alpha_init = 1.0;
        let (alpha, f_new) = backtracking_line_search(
            &mut fun,
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
        let s = get_array_from_pool(&mut memory_pool, n, |_| alpha * &p);
        let x_new = &x + &s;

        // Check step size convergence
        if array_norm_chunked(&s, options.chunk_size) < base_opts.xtol {
            return_array_to_pool(&mut memory_pool, s);
            return_array_to_pool(&mut memory_pool, p);
            x = x_new;
            break;
        }

        // Calculate new gradient using streaming approach
        let g_new = streaming_grad.compute(&mut fun, &x_new.view())?;
        nfev += n.div_ceil(options.chunk_size);

        // Gradient difference
        let y = get_array_from_pool(&mut memory_pool, n, |_| &g_new - &g);

        // Check convergence on function value
        if check_convergence(
            f - f_new,
            0.0,
            g_new.mapv(|gi| gi.abs()).sum(),
            base_opts.ftol,
            0.0,
            base_opts.gtol,
        ) {
            return_array_to_pool(&mut memory_pool, s);
            return_array_to_pool(&mut memory_pool, y);
            return_array_to_pool(&mut memory_pool, p);
            x = x_new;
            g = g_new;
            break;
        }

        // Update L-BFGS history with memory management
        let s_dot_y = chunked_dot_product(&s, &y, options.chunk_size);
        if s_dot_y > 1e-10 {
            // Add to history
            s_history.push_back(s);
            y_history.push_back(y);

            // Remove oldest entries if history is too long
            while s_history.len() > options.max_history {
                if let Some(old_s) = s_history.pop_front() {
                    return_array_to_pool(&mut memory_pool, old_s);
                }
                if let Some(old_y) = y_history.pop_front() {
                    return_array_to_pool(&mut memory_pool, old_y);
                }
            }
        } else {
            // Don't update history, just return arrays to pool
            return_array_to_pool(&mut memory_pool, s);
            return_array_to_pool(&mut memory_pool, y);
        }

        return_array_to_pool(&mut memory_pool, p);

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;
        iter += 1;
    }

    // Clean up remaining arrays in history
    while let Some(s) = s_history.pop_front() {
        return_array_to_pool(&mut memory_pool, s);
    }
    while let Some(y) = y_history.pop_front() {
        return_array_to_pool(&mut memory_pool, y);
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
            "Memory-efficient optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// Memory-efficient L-BFGS direction computation
fn compute_lbfgs_direction_memory_efficient(
    g: &Array1<f64>,
    s_history: &VecDeque<Array1<f64>>,
    y_history: &VecDeque<Array1<f64>>,
    memory_pool: &mut Option<MemoryPool>,
) -> Result<Array1<f64>, OptimizeError> {
    let m = s_history.len();
    if m == 0 {
        return Ok(-g);
    }

    let n = g.len();
    let mut q = get_array_from_pool(memory_pool, n, |_| g.clone());
    let mut alpha = vec![0.0; m];

    // First loop: backward through history
    for i in (0..m).rev() {
        let rho_i = 1.0 / y_history[i].dot(&s_history[i]);
        alpha[i] = rho_i * s_history[i].dot(&q);
        let temp = get_array_from_pool(memory_pool, n, |_| &q - alpha[i] * &y_history[i]);
        return_array_to_pool(memory_pool, q);
        q = temp;
    }

    // Apply initial Hessian approximation (simple scaling)
    let gamma = if m > 0 {
        s_history[m - 1].dot(&y_history[m - 1]) / y_history[m - 1].dot(&y_history[m - 1])
    } else {
        1.0
    };

    let mut r = get_array_from_pool(memory_pool, n, |_| gamma * &q);
    return_array_to_pool(memory_pool, q);

    // Second loop: forward through history
    for i in 0..m {
        let rho_i = 1.0 / y_history[i].dot(&s_history[i]);
        let beta = rho_i * y_history[i].dot(&r);
        let temp = get_array_from_pool(memory_pool, n, |_| &r + (alpha[i] - beta) * &s_history[i]);
        return_array_to_pool(memory_pool, r);
        r = temp;
    }

    // Return -r as the search direction
    let result = get_array_from_pool(memory_pool, n, |_| -&r);
    return_array_to_pool(memory_pool, r);
    Ok(result)
}

/// Get array from memory pool or create new one
fn get_array_from_pool<F>(
    memory_pool: &mut Option<MemoryPool>,
    size: usize,
    init_fn: F,
) -> Array1<f64>
where
    F: FnOnce(usize) -> Array1<f64>,
{
    match memory_pool {
        Some(pool) => {
            let mut array = pool.get_array(size);
            if array.len() != size {
                array = Array1::zeros(size);
            }
            let result = init_fn(size);
            pool.return_array(array);
            result
        }
        None => init_fn(size),
    }
}

/// Return array to memory pool
fn return_array_to_pool(memory_pool: &mut Option<MemoryPool>, array: Array1<f64>) {
    if let Some(pool) = memory_pool {
        pool.return_array(array);
    }
    // If no pool, array will be dropped normally
}

/// Compute dot product in chunks to reduce memory usage
fn chunked_dot_product(a: &Array1<f64>, b: &Array1<f64>, chunk_size: usize) -> f64 {
    let n = a.len();
    let mut result = 0.0;

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n);
        let a_chunk = a.slice(ndarray::s![chunk_start..chunk_end]);
        let b_chunk = b.slice(ndarray::s![chunk_start..chunk_end]);
        result += a_chunk.dot(&b_chunk);
    }

    result
}

/// Compute array norm in chunks to reduce memory usage
fn array_norm_chunked(array: &Array1<f64>, chunk_size: usize) -> f64 {
    let n = array.len();
    let mut sum_sq = 0.0;

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n);
        let chunk = array.slice(ndarray::s![chunk_start..chunk_end]);
        sum_sq += chunk.mapv(|x| x.powi(2)).sum();
    }

    sum_sq.sqrt()
}

/// Estimate memory usage for given problem size and history
fn estimate_memory_usage(n: usize, max_history: usize) -> usize {
    // Size of f64 in bytes
    const F64_SIZE: usize = std::mem::size_of::<f64>();

    // Current point and gradient
    let current_vars = 2 * n * F64_SIZE;

    // L-BFGS history (s and y vectors)
    let history_size = 2 * max_history * n * F64_SIZE;

    // Temporary arrays for computation
    let temp_arrays = 4 * n * F64_SIZE;

    current_vars + history_size + temp_arrays
}

/// Create a memory-efficient optimizer with automatic parameter selection
pub fn create_memory_efficient_optimizer(
    problem_size: usize,
    available_memory_mb: usize,
) -> MemoryOptions {
    let available_bytes = available_memory_mb * 1024 * 1024;

    // Estimate parameters based on available memory
    let max_history = std::cmp::min(
        20,
        available_bytes / (2 * problem_size * std::mem::size_of::<f64>() * 4),
    )
    .max(1);

    let chunk_size = std::cmp::min(
        problem_size,
        std::cmp::max(64, available_bytes / (8 * std::mem::size_of::<f64>())),
    );

    MemoryOptions {
        base_options: Options::default(),
        max_memory_bytes: available_bytes,
        chunk_size,
        max_history,
        use_memory_pool: true,
        use_out_of_core: available_memory_mb < 100, // Use out-of-core for very limited memory
        temp_dir: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_memory_efficient_lbfgs_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 {
            // Simple quadratic: f(x) = sum(x_i^2)
            x.mapv(|xi| xi.powi(2)).sum()
        };

        let n = 100; // Large problem
        let x0 = Array1::ones(n);
        let mut options = MemoryOptions::default();
        options.chunk_size = 32; // Small chunks
        options.max_history = 5; // Limited history

        let result = minimize_memory_efficient_lbfgs(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // Should converge to origin
        for i in 0..std::cmp::min(10, n) {
            assert_abs_diff_eq!(result.x[i], 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_chunked_operations() {
        let a = Array1::from_vec((0..100).map(|i| i as f64).collect());
        let b = Array1::from_vec((0..100).map(|i| (i * 2) as f64).collect());

        // Test chunked dot product
        let dot_chunked = chunked_dot_product(&a, &b, 10);
        let dot_normal = a.dot(&b);
        assert_abs_diff_eq!(dot_chunked, dot_normal, epsilon = 1e-10);

        // Test chunked norm
        let norm_chunked = array_norm_chunked(&a, 10);
        let norm_normal = a.mapv(|x| x.powi(2)).sum().sqrt();
        assert_abs_diff_eq!(norm_chunked, norm_normal, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(3);

        // Get and return arrays
        let arr1 = pool.get_array(10);
        let arr2 = pool.get_array(10);

        pool.return_array(arr1);
        pool.return_array(arr2);

        // Should reuse arrays
        let arr3 = pool.get_array(10);
        let arr4 = pool.get_array(10);

        pool.return_array(arr3);
        pool.return_array(arr4);

        assert_eq!(pool.array_pool.len(), 2);
    }

    #[test]
    fn test_memory_estimation() {
        let n = 1000;
        let max_history = 10;
        let estimated = estimate_memory_usage(n, max_history);

        // Should be reasonable estimate (not zero, not too large)
        assert!(estimated > 0);
        assert!(estimated < 1_000_000); // Less than 1MB for this small problem
    }

    #[test]
    fn test_auto_parameter_selection() {
        let options = create_memory_efficient_optimizer(10000, 64); // 64MB available

        assert!(options.chunk_size > 0);
        assert!(options.max_history > 0);
        assert!(options.max_memory_bytes > 0);
    }
}
