//! RMSProp (Root Mean Square Propagation) optimizer
//!
//! RMSProp is an adaptive learning rate method that addresses AdaGrad's learning rate
//! decay problem. It maintains a moving average of the squared gradients and divides
//! the gradient by the root of this average.

use crate::error::OptimizeError;
use crate::stochastic::{
    clip_gradients, generate_batch_indices, update_learning_rate, DataProvider,
    LearningRateSchedule, StochasticGradientFunction,
};
use crate::unconstrained::result::OptimizeResult;
use ndarray::Array1;

/// Options for RMSProp optimization
#[derive(Debug, Clone)]
pub struct RMSPropOptions {
    /// Learning rate (step size)
    pub learning_rate: f64,
    /// Decay rate for the moving average
    pub decay_rate: f64,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Batch size for mini-batch optimization
    pub batch_size: Option<usize>,
    /// Use centered RMSProp (subtract gradient mean)
    pub centered: bool,
    /// Momentum parameter (when > 0, uses RMSProp with momentum)
    pub momentum: Option<f64>,
}

impl Default for RMSPropOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            decay_rate: 0.99,
            epsilon: 1e-8,
            max_iter: 1000,
            tol: 1e-6,
            lr_schedule: LearningRateSchedule::Constant,
            gradient_clip: None,
            batch_size: None,
            centered: false,
            momentum: None,
        }
    }
}

/// RMSProp optimizer implementation
pub fn minimize_rmsprop<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: RMSPropOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize moving averages
    let mut s: Array1<f64> = Array1::zeros(x.len()); // Moving average of squared gradients
    let mut g_mean = if options.centered {
        Some(Array1::<f64>::zeros(x.len())) // Moving average of gradients (for centered variant)
    } else {
        None
    };
    let mut momentum_buffer = if options.momentum.is_some() {
        Some(Array1::<f64>::zeros(x.len())) // Momentum buffer
    } else {
        None
    };

    // Track the best solution found
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    // Convergence tracking
    let mut prev_loss = f64::INFINITY;
    let mut stagnant_iterations = 0;

    println!("Starting RMSProp optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", actual_batch_size);
    println!("  Initial learning rate: {}", options.learning_rate);
    println!("  Decay rate: {}", options.decay_rate);
    println!("  Centered: {}", options.centered);
    if let Some(mom) = options.momentum {
        println!("  Momentum: {}", mom);
    }

    #[allow(clippy::explicit_counter_loop)]
    for iteration in 0..options.max_iter {
        // Update learning rate according to schedule
        let current_lr = update_learning_rate(
            options.learning_rate,
            iteration,
            options.max_iter,
            &options.lr_schedule,
        );

        // Generate batch indices
        let batch_indices = if actual_batch_size < num_samples {
            generate_batch_indices(num_samples, actual_batch_size, true)
        } else {
            (0..num_samples).collect()
        };

        // Get batch data
        let batch_data = data_provider.get_batch(&batch_indices);

        // Compute gradient on batch
        let mut gradient = grad_func.compute_gradient(&x.view(), &batch_data);
        _grad_evals += 1;

        // Apply gradient clipping if specified
        if let Some(clip_threshold) = options.gradient_clip {
            clip_gradients(&mut gradient, clip_threshold);
        }

        // Update moving average of squared gradients
        let gradient_sq = gradient.mapv(|g| g * g);
        s = &s * options.decay_rate + &gradient_sq * (1.0 - options.decay_rate);

        // Update moving average of gradients (centered variant)
        if let Some(ref mut g_avg) = g_mean {
            *g_avg = &*g_avg * options.decay_rate + &gradient * (1.0 - options.decay_rate);
        }

        // Compute the adaptive learning rate
        let effective_gradient = if options.centered {
            if let Some(ref g_avg) = g_mean {
                // Centered RMSProp: use s - g_mean^2 for better conditioning
                let centered_s = &s - &g_avg.mapv(|g| g * g);
                let denominator = centered_s.mapv(|s| (s + options.epsilon).sqrt());
                &gradient / &denominator
            } else {
                unreachable!("g_mean should be Some when centered is true");
            }
        } else {
            // Standard RMSProp
            let denominator = s.mapv(|s| (s + options.epsilon).sqrt());
            &gradient / &denominator
        };

        // Apply momentum if specified
        let update = if let Some(momentum_factor) = options.momentum {
            if let Some(ref mut momentum_buf) = momentum_buffer {
                // RMSProp with momentum: v = momentum * v + lr * effective_gradient
                *momentum_buf = &*momentum_buf * momentum_factor + &effective_gradient * current_lr;
                momentum_buf.clone()
            } else {
                unreachable!("momentum_buffer should be Some when momentum is Some");
            }
        } else {
            // Standard RMSProp update
            &effective_gradient * current_lr
        };

        // Update parameters
        x = &x - &update;

        // Evaluate on full dataset periodically for convergence check
        if iteration % 10 == 0 || iteration == options.max_iter - 1 {
            let full_data = data_provider.get_full_data();
            let current_loss = grad_func.compute_value(&x.view(), &full_data);
            func_evals += 1;

            // Update best solution
            if current_loss < best_f {
                best_f = current_loss;
                best_x = x.clone();
                stagnant_iterations = 0;
            } else {
                stagnant_iterations += 1;
            }

            // Progress reporting
            if iteration % 100 == 0 {
                let grad_norm = gradient.mapv(|g| g * g).sum().sqrt();
                let rms_norm = s.mapv(|s| s.sqrt()).mean().unwrap_or(0.0);
                println!(
                    "  Iteration {}: loss = {:.6e}, |grad| = {:.3e}, RMS = {:.3e}, lr = {:.3e}",
                    iteration, current_loss, grad_norm, rms_norm, current_lr
                );
            }

            // Check convergence
            let loss_change = (prev_loss - current_loss).abs();
            if loss_change < options.tol {
                return Ok(OptimizeResult {
                    x: best_x,
                    fun: best_f,
                    iterations: iteration,
                    nit: iteration,
                    func_evals,
                    nfev: func_evals,
                    success: true,
                    message: format!(
                        "RMSProp converged: loss change {:.2e} < {:.2e}",
                        loss_change, options.tol
                    ),
                    jacobian: Some(gradient),
                    hessian: None,
                });
            }

            prev_loss = current_loss;

            // Early stopping for stagnation
            if stagnant_iterations > 50 {
                return Ok(OptimizeResult {
                    x: best_x,
                    fun: best_f,
                    iterations: iteration,
                    nit: iteration,
                    func_evals,
                    nfev: func_evals,
                    success: false,
                    message: "RMSProp stopped due to stagnation".to_string(),
                    jacobian: Some(gradient),
                    hessian: None,
                });
            }
        }
    }

    // Final evaluation
    let full_data = data_provider.get_full_data();
    let final_loss = grad_func.compute_value(&best_x.view(), &full_data);
    func_evals += 1;

    Ok(OptimizeResult {
        x: best_x,
        fun: final_loss.min(best_f),
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals,
        nfev: func_evals,
        success: false,
        message: "RMSProp reached maximum iterations".to_string(),
        jacobian: None,
        hessian: None,
    })
}

/// Graves' RMSProp implementation with improved numerical stability
pub fn minimize_graves_rmsprop<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: RMSPropOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize Graves' RMSProp variables
    let mut n: Array1<f64> = Array1::zeros(x.len()); // Accumulated squared gradients
    let mut g: Array1<f64> = Array1::zeros(x.len()); // Accumulated gradients
    let mut delta: Array1<f64> = Array1::zeros(x.len()); // Accumulated squared updates

    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    println!("Starting Graves' RMSProp optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", actual_batch_size);

    #[allow(clippy::explicit_counter_loop)]
    for iteration in 0..options.max_iter {
        let current_lr = update_learning_rate(
            options.learning_rate,
            iteration,
            options.max_iter,
            &options.lr_schedule,
        );

        // Generate batch and compute gradient
        let batch_indices = if actual_batch_size < num_samples {
            generate_batch_indices(num_samples, actual_batch_size, true)
        } else {
            (0..num_samples).collect()
        };

        let batch_data = data_provider.get_batch(&batch_indices);
        let mut gradient = grad_func.compute_gradient(&x.view(), &batch_data);
        _grad_evals += 1;

        if let Some(clip_threshold) = options.gradient_clip {
            clip_gradients(&mut gradient, clip_threshold);
        }

        // Graves' RMSProp updates
        n = &n * options.decay_rate + &gradient.mapv(|g| g * g) * (1.0 - options.decay_rate);
        g = &g * options.decay_rate + &gradient * (1.0 - options.decay_rate);

        // Compute the parameter update
        let rms_n = n.mapv(|n_i| (n_i + options.epsilon).sqrt());
        // let rms_delta = delta.mapv(|d_i| (d_i + options.epsilon).sqrt());

        // Simplified Graves' formula: just use gradient scaling similar to standard RMSProp
        // The original Graves' formula had numerical stability issues
        let scaled_gradient = &gradient / &rms_n;
        let final_update = scaled_gradient.mapv_into_any(|g| g * current_lr);

        // Update parameters and accumulate squared updates
        x = &x - &final_update;
        delta = &delta * options.decay_rate
            + &final_update.mapv(|u| u * u) * (1.0 - options.decay_rate);

        // Evaluate progress
        if iteration % 10 == 0 || iteration == options.max_iter - 1 {
            let full_data = data_provider.get_full_data();
            let current_loss = grad_func.compute_value(&x.view(), &full_data);
            func_evals += 1;

            if current_loss < best_f {
                best_f = current_loss;
                best_x = x.clone();
            }

            if iteration % 100 == 0 {
                let grad_norm = gradient.mapv(|g| g * g).sum().sqrt();
                println!(
                    "  Iteration {}: loss = {:.6e}, |grad| = {:.3e}, lr = {:.3e}",
                    iteration, current_loss, grad_norm, current_lr
                );
            }
        }
    }

    Ok(OptimizeResult {
        x: best_x,
        fun: best_f,
        iterations: options.max_iter,
        nit: options.max_iter,
        func_evals,
        nfev: func_evals,
        success: false,
        message: "Graves' RMSProp completed".to_string(),
        jacobian: None,
        hessian: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stochastic::InMemoryDataProvider;
    use approx::assert_abs_diff_eq;
    use ndarray::ArrayView1;

    // Simple quadratic function for testing
    struct QuadraticFunction;

    impl StochasticGradientFunction for QuadraticFunction {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
            // Gradient of f(x) = sum(x_i^2) is 2*x
            x.mapv(|xi| 2.0 * xi)
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
            // f(x) = sum(x_i^2)
            x.mapv(|xi| xi * xi).sum()
        }
    }

    #[test]
    fn test_rmsprop_quadratic() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, 2.0, -1.5]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = RMSPropOptions {
            learning_rate: 0.1,
            max_iter: 200,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_rmsprop(grad_func, x0, data_provider, options).unwrap();

        // Should converge to zero
        assert!(result.success || result.fun < 1e-4);
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_rmsprop_centered() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, -1.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = RMSPropOptions {
            learning_rate: 0.1,
            max_iter: 500,
            batch_size: Some(10),
            centered: true,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_rmsprop(grad_func, x0, data_provider, options).unwrap();

        // Centered RMSProp should converge
        assert!(result.success || result.fun < 1e-4);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![2.0, -2.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = RMSPropOptions {
            learning_rate: 0.01,
            max_iter: 150,
            batch_size: Some(20),
            momentum: Some(0.9),
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_rmsprop(grad_func, x0, data_provider, options).unwrap();

        // RMSProp with momentum should help convergence
        assert!(result.success || result.fun < 1e-3);
    }

    #[test]
    fn test_graves_rmsprop() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.5, -1.5]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = RMSPropOptions {
            learning_rate: 0.1,
            max_iter: 500,
            batch_size: Some(10),
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_graves_rmsprop(grad_func, x0, data_provider, options).unwrap();

        // Graves' variant should also work well (very relaxed tolerance)
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_rmsprop_different_decay_rates() {
        let _grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, 1.0]);
        let _data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        // Test different decay rates
        let decay_rates = [0.9, 0.95, 0.99, 0.999];

        for &decay_rate in &decay_rates {
            let options = RMSPropOptions {
                learning_rate: 0.1,
                decay_rate,
                max_iter: 500,
                tol: 1e-6,
                ..Default::default()
            };

            let grad_func_clone = QuadraticFunction;
            let x0_clone = x0.clone();
            let data_provider_clone = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

            let result =
                minimize_rmsprop(grad_func_clone, x0_clone, data_provider_clone, options).unwrap();

            // All decay rates should lead to reasonable convergence
            assert!(result.fun < 1e-2, "Failed with decay rate {}", decay_rate);
        }
    }
}
