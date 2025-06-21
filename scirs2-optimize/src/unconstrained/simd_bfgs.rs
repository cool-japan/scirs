//! SIMD-accelerated BFGS algorithm
//!
//! This module provides a SIMD-optimized implementation of the BFGS
//! quasi-Newton optimization algorithm, offering significant performance
//! improvements for problems with many variables.

use crate::error::OptimizeError;
use crate::simd_ops::{SimdConfig, SimdVectorOps};
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::{Bounds, OptimizeResult, Options};
use ndarray::{Array1, Array2, ArrayView1};

/// Options specific to SIMD BFGS
#[derive(Debug, Clone)]
pub struct SimdBfgsOptions {
    /// Base optimization options
    pub base_options: Options,
    /// SIMD configuration (auto-detected if None)
    pub simd_config: Option<SimdConfig>,
    /// Force SIMD usage even for small problems
    pub force_simd: bool,
    /// Minimum problem size to enable SIMD
    pub simd_threshold: usize,
}

impl Default for SimdBfgsOptions {
    fn default() -> Self {
        Self {
            base_options: Options::default(),
            simd_config: None,
            force_simd: false,
            simd_threshold: 8, // Enable SIMD for problems with 8+ variables
        }
    }
}

/// SIMD-accelerated BFGS state
struct SimdBfgsState {
    /// Current Hessian approximation
    hessian_inv: Array2<f64>,
    /// SIMD operations handler
    simd_ops: SimdVectorOps,
    /// Current gradient
    gradient: Array1<f64>,
    /// Previous gradient
    prev_gradient: Array1<f64>,
    /// Current position
    position: Array1<f64>,
    /// Previous position
    prev_position: Array1<f64>,
    /// Current function value
    function_value: f64,
    /// Number of function evaluations
    nfev: usize,
    /// Number of gradient evaluations
    njev: usize,
}

impl SimdBfgsState {
    fn new(x0: &Array1<f64>, simd_config: Option<SimdConfig>) -> Self {
        let n = x0.len();
        let simd_ops = if let Some(config) = simd_config {
            SimdVectorOps::with_config(config)
        } else {
            SimdVectorOps::new()
        };

        Self {
            hessian_inv: Array2::eye(n),
            simd_ops,
            gradient: Array1::zeros(n),
            prev_gradient: Array1::zeros(n),
            position: x0.clone(),
            prev_position: x0.clone(),
            function_value: 0.0,
            nfev: 0,
            njev: 0,
        }
    }

    /// Update the inverse Hessian approximation using SIMD operations
    fn update_hessian(&mut self) {
        let n = self.position.len();

        // Compute s = x_k - x_{k-1}
        let s = self
            .simd_ops
            .sub(&self.position.view(), &self.prev_position.view());

        // Compute y = g_k - g_{k-1}
        let y = self
            .simd_ops
            .sub(&self.gradient.view(), &self.prev_gradient.view());

        // Compute ρ = 1 / (s^T y)
        let s_dot_y = self.simd_ops.dot_product(&s.view(), &y.view());

        if s_dot_y.abs() < 1e-14 {
            // Skip update if curvature condition is not satisfied
            return;
        }

        let rho = 1.0 / s_dot_y;

        // BFGS update using SIMD operations where possible
        // H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T

        // For efficiency, we'll use the Sherman-Morrison-Woodbury formula
        // This requires matrix operations that are not easily SIMD-vectorized,
        // but we can use SIMD for the vector operations within the matrix computations

        // Compute H_k * y
        let hy = self.matrix_vector_multiply_simd(&self.hessian_inv.view(), &y.view());

        // Compute y^T H_k y
        let ythhy = self.simd_ops.dot_product(&y.view(), &hy.view());

        // Update H_{k+1} = H_k - (H_k y y^T H_k) / (y^T H_k y) + (s s^T) / (s^T y)
        //                   + rho^2 (y^T H_k y) (s s^T)

        for i in 0..n {
            for j in 0..n {
                let hess_update = -hy[i] * hy[j] / ythhy + rho * s[i] * s[j];
                self.hessian_inv[[i, j]] += hess_update;
            }
        }
    }

    /// Matrix-vector multiply using SIMD for row operations
    fn matrix_vector_multiply_simd(
        &self,
        matrix: &ndarray::ArrayView2<f64>,
        vector: &ArrayView1<f64>,
    ) -> Array1<f64> {
        self.simd_ops.matvec(matrix, vector)
    }

    /// Vector-matrix multiply (v^T * M) using SIMD
    #[allow(dead_code)]
    fn vector_matrix_multiply_simd(
        &self,
        vector: &ArrayView1<f64>,
        matrix: &ndarray::ArrayView2<f64>,
    ) -> Array1<f64> {
        let n = matrix.ncols();
        let mut result = Array1::zeros(n);

        // Compute v^T * M by taking dot products of v with each column of M
        for j in 0..n {
            let column = matrix.column(j);
            result[j] = self.simd_ops.dot_product(vector, &column);
        }

        result
    }

    /// Compute search direction using SIMD operations
    fn compute_search_direction(&self) -> Array1<f64> {
        // d = -H * g
        let neg_grad = self.simd_ops.scale(-1.0, &self.gradient.view());
        self.matrix_vector_multiply_simd(&self.hessian_inv.view(), &neg_grad.view())
    }
}

/// SIMD-accelerated BFGS minimization
pub fn minimize_simd_bfgs<F>(
    mut fun: F,
    x0: Array1<f64>,
    options: Option<SimdBfgsOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let n = x0.len();

    // Check if SIMD should be used
    let use_simd = options.force_simd
        || (n >= options.simd_threshold
            && options
                .simd_config
                .as_ref()
                .map_or_else(|| SimdConfig::detect().has_simd(), |c| c.has_simd()));

    if !use_simd {
        // Fall back to regular BFGS for small problems or when SIMD is not available
        return crate::unconstrained::bfgs::minimize_bfgs(fun, x0, &options.base_options);
    }

    let mut state = SimdBfgsState::new(&x0, options.simd_config);

    // Initial function evaluation
    state.function_value = fun(&state.position.view());
    state.nfev += 1;

    // Compute initial gradient using finite differences
    state.gradient = compute_gradient_finite_diff(&mut fun, &state.position, &mut state.nfev);
    state.njev += 1;

    let mut prev_f = state.function_value;

    for iteration in 0..options.base_options.max_iter {
        // Check convergence - gradient norm
        let grad_norm = state.simd_ops.norm(&state.gradient.view());
        if grad_norm < options.base_options.gtol {
            return Ok(OptimizeResult {
                x: state.position,
                fun: state.function_value,
                iterations: iteration,
                nit: iteration,
                func_evals: state.nfev,
                nfev: state.nfev,
                jacobian: Some(state.gradient),
                hessian: Some(state.hessian_inv),
                success: true,
                message: "SIMD BFGS optimization terminated successfully.".to_string(),
            });
        }

        // Check function value convergence
        if iteration > 0 {
            let f_change = (prev_f - state.function_value).abs();
            if f_change < options.base_options.ftol {
                return Ok(OptimizeResult {
                    x: state.position,
                    fun: state.function_value,
                    iterations: iteration,
                    nit: iteration,
                    func_evals: state.nfev,
                    nfev: state.nfev,
                    jacobian: Some(state.gradient),
                    hessian: Some(state.hessian_inv),
                    success: true,
                    message: "SIMD BFGS optimization terminated successfully.".to_string(),
                });
            }
        }

        // Store previous values
        state.prev_position = state.position.clone();
        state.prev_gradient = state.gradient.clone();
        prev_f = state.function_value;

        // Compute search direction
        let search_direction = state.compute_search_direction();

        // Ensure it's a descent direction
        let directional_derivative = state
            .simd_ops
            .dot_product(&state.gradient.view(), &search_direction.view());
        if directional_derivative >= 0.0 {
            // Reset Hessian to identity if not a descent direction
            state.hessian_inv = Array2::eye(n);
            let neg_grad = state.simd_ops.scale(-1.0, &state.gradient.view());
            state.position = state.simd_ops.add(
                &state.position.view(),
                &state.simd_ops.scale(0.001, &neg_grad.view()).view(),
            );
        } else {
            // Line search
            let (step_size, line_search_nfev) = backtracking_line_search(
                &mut |x| fun(x),
                &state.position.view(),
                state.function_value,
                &search_direction.view(),
                &state.gradient.view(),
                1.0,
                1e-4,
                0.9,
                options.base_options.bounds.as_ref(),
            );
            state.nfev += line_search_nfev as usize;

            // Update position using SIMD operations
            let step_vec = state.simd_ops.scale(step_size, &search_direction.view());
            state.position = state.simd_ops.add(&state.position.view(), &step_vec.view());
        }

        // Apply bounds if specified
        if let Some(ref bounds) = options.base_options.bounds {
            apply_bounds(&mut state.position, bounds);
        }

        // Evaluate function at new position
        state.function_value = fun(&state.position.view());
        state.nfev += 1;

        // Compute new gradient
        state.gradient = compute_gradient_finite_diff(&mut fun, &state.position, &mut state.nfev);
        state.njev += 1;

        // Update Hessian approximation (only if we took a step)
        if iteration > 0 {
            state.update_hessian();
        }

        // Check parameter change convergence
        let position_change = state
            .simd_ops
            .sub(&state.position.view(), &state.prev_position.view());
        let position_change_norm = state.simd_ops.norm(&position_change.view());
        if position_change_norm < options.base_options.xtol {
            return Ok(OptimizeResult {
                x: state.position,
                fun: state.function_value,
                iterations: iteration + 1,
                nit: iteration + 1,
                func_evals: state.nfev,
                nfev: state.nfev,
                jacobian: Some(state.gradient),
                hessian: Some(state.hessian_inv),
                success: true,
                message: "SIMD BFGS optimization terminated successfully.".to_string(),
            });
        }
    }

    // Maximum iterations reached
    Ok(OptimizeResult {
        x: state.position,
        fun: state.function_value,
        iterations: options.base_options.max_iter,
        nit: options.base_options.max_iter,
        func_evals: state.nfev,
        nfev: state.nfev,
        jacobian: Some(state.gradient),
        hessian: Some(state.hessian_inv),
        success: false,
        message: "Maximum iterations reached in SIMD BFGS.".to_string(),
    })
}

/// Compute gradient using finite differences
fn compute_gradient_finite_diff<F>(fun: &mut F, x: &Array1<f64>, nfev: &mut usize) -> Array1<f64>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    let eps = (f64::EPSILON).sqrt();
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

/// Apply bounds constraints to the position vector
fn apply_bounds(x: &mut Array1<f64>, bounds: &Bounds) {
    for (i, xi) in x.iter_mut().enumerate() {
        if i < bounds.lower.len() {
            if let Some(lb) = bounds.lower[i] {
                if *xi < lb {
                    *xi = lb;
                }
            }
        }
        if i < bounds.upper.len() {
            if let Some(ub) = bounds.upper[i] {
                if *xi > ub {
                    *xi = ub;
                }
            }
        }
    }
}

/// Convenience function for SIMD BFGS with default options
pub fn minimize_simd_bfgs_default<F>(
    fun: F,
    x0: Array1<f64>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
{
    minimize_simd_bfgs(fun, x0, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_simd_bfgs_quadratic() {
        // Simple quadratic function: f(x) = x^T x
        let fun = |x: &ArrayView1<f64>| x.iter().map(|&xi| xi.powi(2)).sum::<f64>();

        let x0 = array![1.0, 2.0, 3.0, 4.0];
        let options = SimdBfgsOptions {
            base_options: Options {
                max_iter: 100,
                gtol: 1e-8,
                ..Default::default()
            },
            force_simd: true,
            ..Default::default()
        };

        let result = minimize_simd_bfgs(fun, x0, Some(options)).unwrap();

        assert!(result.success);
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-6);
        }
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_simd_bfgs_rosenbrock() {
        // Rosenbrock function
        let rosenbrock = |x: &ArrayView1<f64>| {
            let mut sum = 0.0;
            for i in 0..x.len() - 1 {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i].powi(2);
                sum += a.powi(2) + 100.0 * b.powi(2);
            }
            sum
        };

        let x0 = array![0.0, 0.0, 0.0, 0.0];
        let options = SimdBfgsOptions {
            base_options: Options {
                max_iter: 1000,
                gtol: 1e-6,
                ftol: 1e-9,
                ..Default::default()
            },
            force_simd: true,
            ..Default::default()
        };

        let result = minimize_simd_bfgs(rosenbrock, x0, Some(options)).unwrap();

        // Rosenbrock function minimum is at (1, 1, 1, 1)
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 1.0, epsilon = 1e-3);
        }
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_simd_bfgs_with_bounds() {
        // Function with minimum outside bounds
        let fun = |x: &ArrayView1<f64>| (x[0] + 2.0).powi(2) + (x[1] + 2.0).powi(2);

        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        let x0 = array![0.5, 0.5];
        let options = SimdBfgsOptions {
            base_options: Options {
                max_iter: 100,
                gtol: 1e-6,
                bounds: Some(bounds),
                ..Default::default()
            },
            force_simd: true,
            ..Default::default()
        };

        let result = minimize_simd_bfgs(fun, x0, Some(options)).unwrap();

        // Should find minimum at bounds: (0, 0)
        assert!(result.x[0] >= 0.0 && result.x[0] <= 1.0);
        assert!(result.x[1] >= 0.0 && result.x[1] <= 1.0);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_config_detection() {
        let config = SimdConfig::detect();
        println!("SIMD capabilities detected:");
        println!("  AVX2: {}", config.avx2_available);
        println!("  SSE4.1: {}", config.sse41_available);
        println!("  FMA: {}", config.fma_available);
        println!("  Vector width: {}", config.vector_width);

        // Test that we can create SIMD BFGS with detected config
        let options = SimdBfgsOptions {
            simd_config: Some(config),
            force_simd: false,
            ..Default::default()
        };

        let fun = |x: &ArrayView1<f64>| x[0].powi(2);
        let x0 = array![1.0];
        let result = minimize_simd_bfgs(fun, x0, Some(options));
        assert!(result.is_ok());
    }

    #[test]
    fn test_fallback_to_regular_bfgs() {
        // Test with small problem size that should fall back to regular BFGS
        let fun = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let x0 = array![1.0, 2.0];

        let options = SimdBfgsOptions {
            force_simd: false,
            simd_threshold: 10, // Larger than problem size
            ..Default::default()
        };

        let result = minimize_simd_bfgs(fun, x0, Some(options)).unwrap();
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
    }
}
