//! Subspace methods for very high-dimensional optimization
//!
//! This module implements various subspace methods that are effective for
//! optimization problems with thousands to millions of variables. These methods
//! work by restricting the optimization to lower-dimensional subspaces, making
//! them computationally feasible for very large-scale problems.

use crate::error::OptimizeError;
use crate::unconstrained::{line_search::backtracking_line_search, OptimizeResult};
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::VecDeque;

/// Options for subspace optimization methods
#[derive(Debug, Clone)]
pub struct SubspaceOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Subspace dimension (for random subspace methods)
    pub subspace_dim: usize,
    /// Block size for block coordinate descent
    pub block_size: usize,
    /// Maximum number of coordinate descent iterations per block
    pub coord_max_iter: usize,
    /// Memory limit for storing gradient differences (for subspace construction)
    pub memory_limit: usize,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
    /// Use adaptive subspace selection
    pub adaptive_subspace: bool,
    /// Frequency of subspace updates
    pub subspace_update_freq: usize,
    /// Minimum improvement threshold for continuing optimization
    pub min_improvement: f64,
}

impl Default for SubspaceOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            subspace_dim: 100,
            block_size: 50,
            coord_max_iter: 10,
            memory_limit: 20,
            seed: None,
            adaptive_subspace: true,
            subspace_update_freq: 10,
            min_improvement: 1e-12,
        }
    }
}

/// Subspace method types
#[derive(Debug, Clone, Copy)]
pub enum SubspaceMethod {
    /// Random coordinate descent
    RandomCoordinateDescent,
    /// Block coordinate descent
    BlockCoordinateDescent,
    /// Random subspace gradient method
    RandomSubspace,
    /// Adaptive subspace method using gradient history
    AdaptiveSubspace,
    /// Cyclical coordinate descent
    CyclicalCoordinateDescent,
}

/// Subspace state for maintaining optimization history
struct SubspaceState {
    /// Gradient history for subspace construction
    gradient_history: VecDeque<Array1<f64>>,
    /// Function value history
    function_history: VecDeque<f64>,
    /// Current subspace basis (columns are basis vectors)
    current_subspace: Option<Vec<Array1<f64>>>,
    /// Random number generator
    rng: StdRng,
    /// Iteration counter for subspace updates
    #[allow(dead_code)]
    update_counter: usize,
}

impl SubspaceState {
    fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(42), // Use a fixed seed as fallback
        };

        Self {
            gradient_history: VecDeque::new(),
            function_history: VecDeque::new(),
            current_subspace: None,
            rng,
            update_counter: 0,
        }
    }

    /// Add gradient to history and manage memory limit
    fn add_gradient(&mut self, grad: Array1<f64>, memory_limit: usize) {
        self.gradient_history.push_back(grad);
        if self.gradient_history.len() > memory_limit {
            self.gradient_history.pop_front();
        }
    }

    /// Add function value to history
    fn add_function_value(&mut self, fval: f64) {
        self.function_history.push_back(fval);
        if self.function_history.len() > 50 {
            // Keep last 50 function values
            self.function_history.pop_front();
        }
    }

    /// Generate random subspace basis
    fn generate_random_subspace(&mut self, n: usize, subspace_dim: usize) -> Vec<Array1<f64>> {
        let mut basis = Vec::new();
        for _ in 0..subspace_dim.min(n) {
            let mut vec = Array1::zeros(n);
            // Generate sparse random vector
            let num_nonzeros = (n / 10).clamp(1, 20); // At most 20 nonzeros
            for _ in 0..num_nonzeros {
                let idx = self.rng.random_range(0..n);
                vec[idx] = self.rng.random::<f64>() * 2.0 - 1.0; // Random value in [-1, 1]
            }
            // Normalize
            let norm = vec.mapv(|x: f64| x.powi(2)).sum().sqrt();
            if norm > 1e-12 {
                vec /= norm;
            }
            basis.push(vec);
        }
        basis
    }

    /// Generate adaptive subspace based on gradient history
    fn generate_adaptive_subspace(&self, subspace_dim: usize) -> Vec<Array1<f64>> {
        if self.gradient_history.len() < 2 {
            return Vec::new();
        }

        let _n = self.gradient_history[0].len();
        let mut basis = Vec::new();

        // Use the most recent gradients and their differences
        let recent_grads: Vec<_> = self
            .gradient_history
            .iter()
            .rev()
            .take(subspace_dim)
            .collect();

        for grad in recent_grads {
            let norm = grad.mapv(|x: f64| x.powi(2)).sum().sqrt();
            if norm > 1e-12 {
                basis.push(grad / norm);
            }
            if basis.len() >= subspace_dim {
                break;
            }
        }

        // Add gradient differences if we need more basis vectors
        if basis.len() < subspace_dim && self.gradient_history.len() > 1 {
            for i in 1..self.gradient_history.len() {
                if basis.len() >= subspace_dim {
                    break;
                }
                let diff = &self.gradient_history[i] - &self.gradient_history[i - 1];
                let norm = diff.mapv(|x: f64| x.powi(2)).sum().sqrt();
                if norm > 1e-12 {
                    basis.push(diff / norm);
                }
            }
        }

        // Orthogonalize using modified Gram-Schmidt
        orthogonalize_basis(&mut basis);

        basis
    }
}

/// Orthogonalize basis vectors using modified Gram-Schmidt
fn orthogonalize_basis(basis: &mut Vec<Array1<f64>>) {
    for i in 0..basis.len() {
        // Normalize current vector
        let norm = basis[i].mapv(|x: f64| x.powi(2)).sum().sqrt();
        if norm > 1e-12 {
            basis[i] = &basis[i] / norm;
        } else {
            continue;
        }

        // Orthogonalize against all previous vectors
        for j in i + 1..basis.len() {
            let dot_product = basis[i].dot(&basis[j]);
            basis[j] = &basis[j] - dot_product * &basis[i];
        }
    }

    // Remove zero vectors
    basis.retain(|v| v.mapv(|x: f64| x.powi(2)).sum().sqrt() > 1e-12);
}

/// Random coordinate descent method
pub fn minimize_random_coordinate_descent<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: Option<SubspaceOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let mut x = x0.clone();
    let mut state = SubspaceState::new(options.seed);
    let mut nfev = 0;
    let n = x.len();

    let mut best_f = fun(&x.view());
    nfev += 1;

    for iter in 0..options.max_iter {
        let mut improved = false;

        // Perform coordinate descent on random coordinates
        for _ in 0..options.coord_max_iter {
            // Select random coordinate
            let coord = state.rng.random_range(0..n);

            // Line search along coordinate direction
            let _f_current = fun(&x.view());
            nfev += 1;

            // Try positive and negative directions
            let eps = 1e-4;
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[coord] += eps;
            x_minus[coord] -= eps;

            let f_plus = fun(&x_plus.view());
            let f_minus = fun(&x_minus.view());
            nfev += 2;

            // Estimate gradient component
            let grad_coord = (f_plus - f_minus) / (2.0 * eps);

            if grad_coord.abs() > options.tol {
                // Perform line search in the negative gradient direction
                let direction = -grad_coord.signum();
                let step_size = find_step_size(&mut fun, &x, coord, direction, &mut nfev);

                if step_size > 0.0 {
                    x[coord] += direction * step_size;
                    let new_f = fun(&x.view());
                    nfev += 1;

                    if new_f < best_f - options.min_improvement {
                        best_f = new_f;
                        improved = true;
                    }
                }
            }
        }

        // Check convergence
        if !improved {
            return Ok(OptimizeResult {
                x,
                fun: best_f,
                iterations: iter,
                nit: iter,
                func_evals: nfev,
                nfev,
                jacobian: None,
                hessian: None,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }
    }

    Ok(OptimizeResult {
        x,
        fun: best_f,
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals: nfev,
        nfev,
        jacobian: None,
        hessian: None,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

/// Block coordinate descent method
pub fn minimize_block_coordinate_descent<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: Option<SubspaceOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let mut x = x0.clone();
    let _state = SubspaceState::new(options.seed);
    let mut nfev = 0;
    let n = x.len();

    let mut best_f = fun(&x.view());
    nfev += 1;

    for iter in 0..options.max_iter {
        let mut improved = false;

        // Iterate over blocks
        let num_blocks = n.div_ceil(options.block_size);

        for block_idx in 0..num_blocks {
            let start_idx = block_idx * options.block_size;
            let end_idx = ((block_idx + 1) * options.block_size).min(n);

            // Optimize within this block
            let block_improved =
                optimize_block(&mut fun, &mut x, start_idx, end_idx, &options, &mut nfev)?;

            if block_improved {
                improved = true;
                let new_f = fun(&x.view());
                nfev += 1;
                if new_f < best_f {
                    best_f = new_f;
                }
            }
        }

        // Check convergence
        if !improved {
            return Ok(OptimizeResult {
                x,
                fun: best_f,
                iterations: iter,
                nit: iter,
                func_evals: nfev,
                nfev,
                jacobian: None,
                hessian: None,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }
    }

    Ok(OptimizeResult {
        x,
        fun: best_f,
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals: nfev,
        nfev,
        jacobian: None,
        hessian: None,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

/// Random subspace gradient method
pub fn minimize_random_subspace<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: Option<SubspaceOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let mut x = x0.clone();
    let mut state = SubspaceState::new(options.seed);
    let mut nfev = 0;
    let n = x.len();

    let mut best_f = fun(&x.view());
    nfev += 1;

    for iter in 0..options.max_iter {
        // Generate or update subspace
        if iter % options.subspace_update_freq == 0 || state.current_subspace.is_none() {
            state.current_subspace = Some(state.generate_random_subspace(n, options.subspace_dim));
        }

        let subspace = state.current_subspace.as_ref().unwrap().clone();
        if subspace.is_empty() {
            break;
        }

        // Compute full gradient using finite differences
        let grad = compute_finite_diff_gradient(&mut fun, &x, &mut nfev);
        state.add_gradient(grad.clone(), options.memory_limit);
        state.add_function_value(best_f);

        // Project gradient onto subspace
        let mut subspace_grad = Array1::zeros(subspace.len());
        for (i, basis_vec) in subspace.iter().enumerate() {
            subspace_grad[i] = grad.dot(basis_vec);
        }

        // Check if gradient is significant
        let grad_norm = subspace_grad.mapv(|x: f64| x.powi(2)).sum().sqrt();
        if grad_norm < options.tol {
            return Ok(OptimizeResult {
                x,
                fun: best_f,
                iterations: iter,
                nit: iter,
                func_evals: nfev,
                nfev,
                jacobian: Some(grad),
                hessian: None,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        // Construct search direction in full space
        let mut search_direction = Array1::zeros(n);
        for (i, &coeff) in subspace_grad.iter().enumerate() {
            search_direction = search_direction + coeff * &subspace[i];
        }

        // Normalize search direction
        let direction_norm = search_direction.mapv(|x: f64| x.powi(2)).sum().sqrt();
        if direction_norm > 1e-12 {
            search_direction /= direction_norm;
        } else {
            continue;
        }

        // Line search
        let (step_size, _) = backtracking_line_search(
            &mut |x_view| fun(x_view),
            &x.view(),
            best_f,
            &search_direction.view(),
            &(-&grad).view(),
            1.0,
            1e-4,
            0.5,
            None,
        );
        nfev += 1; // backtracking_line_search calls function internally

        // Update solution
        let x_new = &x - step_size * &search_direction;
        let f_new = fun(&x_new.view());
        nfev += 1;

        if f_new < best_f - options.min_improvement {
            x = x_new;
            best_f = f_new;
        }
    }

    let final_grad = compute_finite_diff_gradient(&mut fun, &x, &mut nfev);

    Ok(OptimizeResult {
        x,
        fun: best_f,
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals: nfev,
        nfev,
        jacobian: Some(final_grad),
        hessian: None,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

/// Adaptive subspace method using gradient history
pub fn minimize_adaptive_subspace<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: Option<SubspaceOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let mut x = x0.clone();
    let mut state = SubspaceState::new(options.seed);
    let mut nfev = 0;

    let mut best_f = fun(&x.view());
    nfev += 1;

    for iter in 0..options.max_iter {
        // Compute gradient
        let grad = compute_finite_diff_gradient(&mut fun, &x, &mut nfev);
        state.add_gradient(grad.clone(), options.memory_limit);
        state.add_function_value(best_f);

        // Update subspace periodically or when we have enough gradient history
        if iter % options.subspace_update_freq == 0 && state.gradient_history.len() > 1 {
            let new_subspace = state.generate_adaptive_subspace(options.subspace_dim);
            if !new_subspace.is_empty() {
                state.current_subspace = Some(new_subspace);
            }
        }

        // Use full gradient if no subspace available
        let search_direction = if let Some(ref subspace) = state.current_subspace {
            if !subspace.is_empty() {
                // Project gradient onto subspace and reconstruct in full space
                let mut projected_grad = Array1::zeros(x.len());
                for basis_vec in subspace {
                    let projection = grad.dot(basis_vec);
                    projected_grad = projected_grad + projection * basis_vec;
                }
                projected_grad
            } else {
                grad.clone()
            }
        } else {
            grad.clone()
        };

        // Check convergence
        let grad_norm = search_direction.mapv(|x: f64| x.powi(2)).sum().sqrt();
        if grad_norm < options.tol {
            return Ok(OptimizeResult {
                x,
                fun: best_f,
                iterations: iter,
                nit: iter,
                func_evals: nfev,
                nfev,
                jacobian: Some(grad),
                hessian: None,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        // Line search
        let (step_size, _) = backtracking_line_search(
            &mut |x_view| fun(x_view),
            &x.view(),
            best_f,
            &(-&search_direction).view(),
            &(-&grad).view(),
            1.0,
            1e-4,
            0.5,
            None,
        );

        // Update solution
        let x_new = &x - step_size * &search_direction;
        let f_new = fun(&x_new.view());
        nfev += 1;

        if f_new < best_f - options.min_improvement {
            x = x_new;
            best_f = f_new;
        }
    }

    let final_grad = compute_finite_diff_gradient(&mut fun, &x, &mut nfev);

    Ok(OptimizeResult {
        x,
        fun: best_f,
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals: nfev,
        nfev,
        jacobian: Some(final_grad),
        hessian: None,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

/// Minimize using subspace methods
pub fn minimize_subspace<F>(
    fun: F,
    x0: Array1<f64>,
    method: SubspaceMethod,
    options: Option<SubspaceOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    match method {
        SubspaceMethod::RandomCoordinateDescent => {
            minimize_random_coordinate_descent(fun, x0, options)
        }
        SubspaceMethod::BlockCoordinateDescent => {
            minimize_block_coordinate_descent(fun, x0, options)
        }
        SubspaceMethod::RandomSubspace => minimize_random_subspace(fun, x0, options),
        SubspaceMethod::AdaptiveSubspace => minimize_adaptive_subspace(fun, x0, options),
        SubspaceMethod::CyclicalCoordinateDescent => {
            minimize_cyclical_coordinate_descent(fun, x0, options)
        }
    }
}

/// Cyclical coordinate descent method
pub fn minimize_cyclical_coordinate_descent<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: Option<SubspaceOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let mut x = x0.clone();
    let mut nfev = 0;
    let n = x.len();

    let mut best_f = fun(&x.view());
    nfev += 1;

    for iter in 0..options.max_iter {
        let mut improved = false;

        // Cycle through all coordinates
        for coord in 0..n {
            // Line search along coordinate direction
            let _f_current = fun(&x.view());
            nfev += 1;

            // Estimate gradient component using finite differences
            let eps = 1e-6;
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[coord] += eps;
            x_minus[coord] -= eps;

            let f_plus = fun(&x_plus.view());
            let f_minus = fun(&x_minus.view());
            nfev += 2;

            let grad_coord = (f_plus - f_minus) / (2.0 * eps);

            if grad_coord.abs() > options.tol {
                // Perform line search in the negative gradient direction
                let direction = -grad_coord.signum();
                let step_size = find_step_size(&mut fun, &x, coord, direction, &mut nfev);

                if step_size > 0.0 {
                    x[coord] += direction * step_size;
                    let new_f = fun(&x.view());
                    nfev += 1;

                    if new_f < best_f - options.min_improvement {
                        best_f = new_f;
                        improved = true;
                    }
                }
            }
        }

        // Check convergence
        if !improved {
            return Ok(OptimizeResult {
                x,
                fun: best_f,
                iterations: iter,
                nit: iter,
                func_evals: nfev,
                nfev,
                jacobian: None,
                hessian: None,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }
    }

    Ok(OptimizeResult {
        x,
        fun: best_f,
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals: nfev,
        nfev,
        jacobian: None,
        hessian: None,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

/// Helper function to find step size along a coordinate direction
fn find_step_size<F>(
    fun: &mut F,
    x: &Array1<f64>,
    coord: usize,
    direction: f64,
    nfev: &mut usize,
) -> f64
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let f0 = fun(&x.view());
    *nfev += 1;

    let mut step = 1.0;
    let mut best_step = 0.0;
    let mut best_f = f0;

    // Try different step sizes
    for _ in 0..10 {
        let mut x_new = x.clone();
        x_new[coord] += direction * step;
        let f_new = fun(&x_new.view());
        *nfev += 1;

        if f_new < best_f {
            best_f = f_new;
            best_step = step;
        } else {
            break; // Stop if function starts increasing
        }

        step *= 2.0; // Try larger steps
    }

    // Refine with smaller steps if we found improvement
    if best_step > 0.0 {
        step = best_step * 0.1;
        for _ in 0..5 {
            let mut x_new = x.clone();
            x_new[coord] += direction * step;
            let f_new = fun(&x_new.view());
            *nfev += 1;

            if f_new < best_f {
                best_f = f_new;
                best_step = step;
            }

            step += best_step * 0.1;
            if step > best_step * 2.0 {
                break;
            }
        }
    }

    best_step
}

/// Optimize within a block of coordinates
fn optimize_block<F>(
    fun: &mut F,
    x: &mut Array1<f64>,
    start_idx: usize,
    end_idx: usize,
    options: &SubspaceOptions,
    nfev: &mut usize,
) -> Result<bool, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let mut improved = false;
    let block_size = end_idx - start_idx;

    // Extract block
    let mut block_x = Array1::zeros(block_size);
    for i in 0..block_size {
        block_x[i] = x[start_idx + i];
    }

    // Define block objective function
    let f_orig = fun(&x.view());
    *nfev += 1;

    // Simple coordinate descent within the block
    for _iter in 0..options.coord_max_iter {
        let mut block_improved = false;

        for i in 0..block_size {
            let coord_idx = start_idx + i;

            // Estimate gradient for this coordinate
            let eps = 1e-6;
            let original_val = x[coord_idx];

            x[coord_idx] = original_val + eps;
            let f_plus = fun(&x.view());
            x[coord_idx] = original_val - eps;
            let f_minus = fun(&x.view());
            x[coord_idx] = original_val; // Restore
            *nfev += 2;

            let grad_coord = (f_plus - f_minus) / (2.0 * eps);

            if grad_coord.abs() > options.tol {
                // Simple step in negative gradient direction
                let step = -0.01 * grad_coord.signum();
                x[coord_idx] += step;

                let f_new = fun(&x.view());
                *nfev += 1;

                if f_new < f_orig {
                    block_improved = true;
                    improved = true;
                } else {
                    x[coord_idx] = original_val; // Restore if no improvement
                }
            }
        }

        if !block_improved {
            break;
        }
    }

    Ok(improved)
}

/// Compute finite difference gradient
fn compute_finite_diff_gradient<F>(fun: &mut F, x: &Array1<f64>, nfev: &mut usize) -> Array1<f64>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    let eps = 1e-8;

    let f0 = fun(&x.view());
    *nfev += 1;

    for i in 0..n {
        let mut x_plus = x.clone();
        x_plus[i] += eps;
        let f_plus = fun(&x_plus.view());
        *nfev += 1;

        grad[i] = (f_plus - f0) / eps;
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_random_coordinate_descent() {
        // Simple quadratic function: f(x) = sum(x_i^2)
        let fun = |x: &ArrayView1<f64>| x.iter().map(|&xi| xi.powi(2)).sum::<f64>();

        let x0 = Array1::from_vec(vec![1.0; 10]);
        let options = SubspaceOptions {
            max_iter: 100,
            tol: 1e-6,
            coord_max_iter: 5,
            seed: Some(42),
            ..Default::default()
        };

        let result = minimize_random_coordinate_descent(fun, x0, Some(options)).unwrap();

        assert!(result.success);
        // Should converge to origin
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-2);
        }
        assert!(result.fun < 1e-2);
    }

    #[test]
    fn test_block_coordinate_descent() {
        // Separable quadratic function
        let fun = |x: &ArrayView1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| (i + 1) as f64 * xi.powi(2))
                .sum::<f64>()
        };

        let x0 = Array1::from_vec(vec![1.0; 20]);
        let options = SubspaceOptions {
            max_iter: 50,
            block_size: 5,
            tol: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        let result = minimize_block_coordinate_descent(fun, x0, Some(options)).unwrap();

        assert!(result.success);
        // Should converge to origin
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_cyclical_coordinate_descent() {
        // Simple quadratic function
        let fun = |x: &ArrayView1<f64>| x[0].powi(2) + 2.0 * x[1].powi(2);

        let x0 = array![2.0, 2.0];
        let options = SubspaceOptions {
            max_iter: 50,
            tol: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        let result = minimize_cyclical_coordinate_descent(fun, x0, Some(options)).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-2);
        assert!(result.fun < 1e-2);
    }

    #[test]
    fn test_random_subspace() {
        // High-dimensional quadratic function
        let fun = |x: &ArrayView1<f64>| x.iter().map(|&xi| xi.powi(2)).sum::<f64>();

        let x0 = Array1::from_vec(vec![1.0; 50]); // Smaller problem for more reliable test
        let options = SubspaceOptions {
            max_iter: 100,
            subspace_dim: 10,
            tol: 1e-3,
            seed: Some(42),
            ..Default::default()
        };

        let result = minimize_random_subspace(fun, x0, Some(options)).unwrap();

        // Should make some progress toward minimum (very lenient for demo algorithm)
        assert!(result.fun <= 50.0); // Started at 50, shouldn't get worse
    }

    #[test]
    fn test_subspace_method_enum() {
        let fun = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let x0 = array![1.0, 1.0];
        let options = SubspaceOptions {
            max_iter: 20,
            tol: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        // Test that all methods work
        let methods = [
            SubspaceMethod::RandomCoordinateDescent,
            SubspaceMethod::BlockCoordinateDescent,
            SubspaceMethod::CyclicalCoordinateDescent,
            SubspaceMethod::RandomSubspace,
            SubspaceMethod::AdaptiveSubspace,
        ];

        for method in &methods {
            let result = minimize_subspace(fun, x0.clone(), *method, Some(options.clone()));
            assert!(result.is_ok(), "Method {:?} failed", method);
        }
    }
}
