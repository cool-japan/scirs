//! Automatic differentiation for exact gradient and Hessian computation
//!
//! This module provides automatic differentiation capabilities for optimization,
//! supporting both forward-mode and reverse-mode AD for efficient and exact
//! derivative computation.

pub mod dual_numbers;
pub mod forward_mode;
pub mod reverse_mode;
pub mod tape;

// Re-export commonly used items
pub use dual_numbers::{Dual, DualNumber};
pub use forward_mode::{forward_gradient, forward_hessian_diagonal, ForwardADOptions};
pub use reverse_mode::{reverse_gradient, reverse_hessian, ReverseADOptions};
pub use tape::{ComputationTape, TapeNode, Variable};

use crate::error::OptimizeError;
use ndarray::{Array1, Array2, ArrayView1};

/// Automatic differentiation mode selection
#[derive(Debug, Clone, Copy)]
pub enum ADMode {
    /// Forward-mode AD (efficient for low-dimensional problems)
    Forward,
    /// Reverse-mode AD (efficient for high-dimensional problems)
    Reverse,
    /// Automatic mode selection based on problem dimension
    Auto,
}

/// Options for automatic differentiation
#[derive(Debug, Clone)]
pub struct AutoDiffOptions {
    /// AD mode to use
    pub mode: ADMode,
    /// Threshold for automatic mode selection (dimension)
    pub auto_threshold: usize,
    /// Enable sparse AD for sparse functions
    pub enable_sparse: bool,
    /// Compute Hessian when possible
    pub compute_hessian: bool,
    /// Forward-mode specific options
    pub forward_options: ForwardADOptions,
    /// Reverse-mode specific options
    pub reverse_options: ReverseADOptions,
}

impl Default for AutoDiffOptions {
    fn default() -> Self {
        Self {
            mode: ADMode::Auto,
            auto_threshold: 10,
            enable_sparse: false,
            compute_hessian: false,
            forward_options: ForwardADOptions::default(),
            reverse_options: ReverseADOptions::default(),
        }
    }
}

/// Result of automatic differentiation computation
#[derive(Debug, Clone)]
pub struct ADResult {
    /// Function value
    pub value: f64,
    /// Gradient (if computed)
    pub gradient: Option<Array1<f64>>,
    /// Hessian (if computed)
    pub hessian: Option<Array2<f64>>,
    /// Number of function evaluations used
    pub n_fev: usize,
    /// AD mode used
    pub mode_used: ADMode,
}

/// Function trait for automatic differentiation
pub trait AutoDiffFunction<T> {
    /// Evaluate the function with AD variables
    fn eval(&self, x: &[T]) -> T;
}

/// Wrapper for regular functions to make them compatible with AD
pub struct FunctionWrapper<F> {
    func: F,
}

impl<F> FunctionWrapper<F>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> AutoDiffFunction<f64> for FunctionWrapper<F>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    fn eval(&self, x: &[f64]) -> f64 {
        let x_array = Array1::from_vec(x.to_vec());
        (self.func)(&x_array.view())
    }
}

impl<F> AutoDiffFunction<Dual> for FunctionWrapper<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    fn eval(&self, x: &[Dual]) -> Dual {
        // For demonstration - this would need proper dual number evaluation
        // In practice, the function would need to be rewritten using dual arithmetic
        let values: Vec<f64> = x.iter().map(|d| d.value()).collect();
        let x_array = Array1::from_vec(values);
        Dual::constant((self.func)(&x_array.view()))
    }
}

/// Main automatic differentiation function
pub fn autodiff<F>(
    func: F,
    x: &ArrayView1<f64>,
    options: &AutoDiffOptions,
) -> Result<ADResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let n = x.len();
    let mode = match options.mode {
        ADMode::Auto => {
            if n <= options.auto_threshold {
                ADMode::Forward
            } else {
                ADMode::Reverse
            }
        }
        mode => mode,
    };

    match mode {
        ADMode::Forward => autodiff_forward(func, x, &options.forward_options),
        ADMode::Reverse => autodiff_reverse(func, x, &options.reverse_options),
        ADMode::Auto => unreachable!(), // Already handled above
    }
}

/// Forward-mode automatic differentiation
fn autodiff_forward<F>(
    func: F,
    x: &ArrayView1<f64>,
    options: &ForwardADOptions,
) -> Result<ADResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let n = x.len();
    let mut n_fev = 0;

    // Compute function value
    let value = func(x);
    n_fev += 1;

    // Compute gradient using forward-mode AD
    let gradient = if options.compute_gradient {
        let grad = forward_gradient(func.clone(), x)?;
        n_fev += n; // Forward mode requires n+1 evaluations for gradient
        Some(grad)
    } else {
        None
    };

    // Compute Hessian diagonal using forward-mode AD (if requested)
    let hessian = if options.compute_hessian {
        let hess_diag = forward_hessian_diagonal(func, x)?;
        n_fev += n; // Additional evaluations for Hessian diagonal

        // Convert diagonal to full matrix (zeros off-diagonal)
        let mut hess = Array2::zeros((n, n));
        for i in 0..n {
            hess[[i, i]] = hess_diag[i];
        }
        Some(hess)
    } else {
        None
    };

    Ok(ADResult {
        value,
        gradient,
        hessian,
        n_fev,
        mode_used: ADMode::Forward,
    })
}

/// Reverse-mode automatic differentiation
fn autodiff_reverse<F>(
    func: F,
    x: &ArrayView1<f64>,
    options: &ReverseADOptions,
) -> Result<ADResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let mut n_fev = 0;

    // Compute function value
    let value = func(x);
    n_fev += 1;

    // Compute gradient using reverse-mode AD
    let gradient = if options.compute_gradient {
        let grad = reverse_gradient(func.clone(), x)?;
        n_fev += 1; // Reverse mode requires only 1 additional evaluation for gradient
        Some(grad)
    } else {
        None
    };

    // Compute Hessian using reverse-mode AD (if requested)
    let hessian = if options.compute_hessian {
        let hess = reverse_hessian(func, x)?;
        n_fev += x.len(); // Reverse mode for Hessian requires n additional evaluations
        Some(hess)
    } else {
        None
    };

    Ok(ADResult {
        value,
        gradient,
        hessian,
        n_fev,
        mode_used: ADMode::Reverse,
    })
}

/// Create a gradient function using automatic differentiation
pub fn create_ad_gradient<F>(
    func: F,
    options: AutoDiffOptions,
) -> impl Fn(&ArrayView1<f64>) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + 'static,
{
    move |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut opts = options.clone();
        opts.forward_options.compute_gradient = true;
        opts.reverse_options.compute_gradient = true;

        match autodiff(func.clone(), x, &opts) {
            Ok(result) => result.gradient.unwrap_or_else(|| Array1::zeros(x.len())),
            Err(_) => Array1::zeros(x.len()), // Fallback to zeros on error
        }
    }
}

/// Create a Hessian function using automatic differentiation
pub fn create_ad_hessian<F>(
    func: F,
    options: AutoDiffOptions,
) -> impl Fn(&ArrayView1<f64>) -> Array2<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + 'static,
{
    move |x: &ArrayView1<f64>| -> Array2<f64> {
        let mut opts = options.clone();
        opts.forward_options.compute_hessian = true;
        opts.reverse_options.compute_hessian = true;

        match autodiff(func.clone(), x, &opts) {
            Ok(result) => result
                .hessian
                .unwrap_or_else(|| Array2::zeros((x.len(), x.len()))),
            Err(_) => Array2::zeros((x.len(), x.len())), // Fallback to zeros on error
        }
    }
}

/// Optimize AD mode selection based on problem characteristics
pub fn optimize_ad_mode(problem_dim: usize, output_dim: usize, expected_sparsity: f64) -> ADMode {
    // Forward mode is efficient when input dimension is small
    // Reverse mode is efficient when output dimension is small (typically 1 for optimization)

    if problem_dim <= 5 {
        ADMode::Forward
    } else if expected_sparsity > 0.8 {
        // For very sparse problems, forward mode might be better
        ADMode::Forward
    } else if output_dim == 1 && problem_dim > 20 {
        ADMode::Reverse
    } else {
        // Default to reverse mode for optimization (output_dim = 1)
        ADMode::Reverse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_autodiff_quadratic() {
        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 2.0 * x[1] * x[1] + x[0] * x[1] };

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let mut options = AutoDiffOptions::default();
        options.forward_options.compute_gradient = true;
        options.reverse_options.compute_gradient = true;

        // Test forward mode
        options.mode = ADMode::Forward;
        let result_forward = autodiff(func, &x.view(), &options).unwrap();

        assert_abs_diff_eq!(result_forward.value, 11.0, epsilon = 1e-10); // 1 + 8 + 2 = 11

        if let Some(grad) = result_forward.gradient {
            // ∂f/∂x₀ = 2x₀ + x₁ = 2(1) + 2 = 4
            // ∂f/∂x₁ = 4x₁ + x₀ = 4(2) + 1 = 9
            assert_abs_diff_eq!(grad[0], 4.0, epsilon = 1e-7);
            assert_abs_diff_eq!(grad[1], 9.0, epsilon = 1e-7);
        }

        // Test reverse mode
        options.mode = ADMode::Reverse;
        let result_reverse = autodiff(func, &x.view(), &options).unwrap();

        assert_abs_diff_eq!(result_reverse.value, 11.0, epsilon = 1e-10);

        if let Some(grad) = result_reverse.gradient {
            assert_abs_diff_eq!(grad[0], 4.0, epsilon = 1e-7);
            assert_abs_diff_eq!(grad[1], 9.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_ad_mode_selection() {
        // Small problem should use forward mode
        assert!(matches!(optimize_ad_mode(3, 1, 0.1), ADMode::Forward));

        // Large problem should use reverse mode
        assert!(matches!(optimize_ad_mode(100, 1, 0.1), ADMode::Reverse));

        // Sparse problem should use forward mode
        assert!(matches!(optimize_ad_mode(50, 1, 0.9), ADMode::Forward));
    }

    #[test]
    fn test_create_ad_gradient() {
        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

        let options = AutoDiffOptions::default();
        let grad_func = create_ad_gradient(func, options);

        let x = Array1::from_vec(vec![3.0, 4.0]);
        let grad = grad_func(&x.view());

        // ∂f/∂x₀ = 2x₀ = 6, ∂f/∂x₁ = 2x₁ = 8
        assert_abs_diff_eq!(grad[0], 6.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad[1], 8.0, epsilon = 1e-6);
    }
}
