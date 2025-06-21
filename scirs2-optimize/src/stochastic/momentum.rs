//! SGD with Momentum optimizer
//!
//! This module implements Stochastic Gradient Descent with momentum, including
//! both classical momentum and Nesterov accelerated gradient (NAG) variants.

use crate::error::OptimizeError;
use crate::stochastic::{
    clip_gradients, generate_batch_indices, update_learning_rate, DataProvider,
    LearningRateSchedule, StochasticGradientFunction,
};
use crate::unconstrained::result::OptimizeResult;
use ndarray::Array1;

/// Options for SGD with momentum optimization
#[derive(Debug, Clone)]
pub struct MomentumOptions {
    /// Learning rate (step size)
    pub learning_rate: f64,
    /// Momentum coefficient (typically 0.9)
    pub momentum: f64,
    /// Use Nesterov accelerated gradient
    pub nesterov: bool,
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
    /// Dampening factor for momentum
    pub dampening: f64,
}

impl Default for MomentumOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            nesterov: false,
            max_iter: 1000,
            tol: 1e-6,
            lr_schedule: LearningRateSchedule::Constant,
            gradient_clip: None,
            batch_size: None,
            dampening: 0.0,
        }
    }
}

/// SGD with momentum optimizer implementation
pub fn minimize_sgd_momentum<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: MomentumOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize momentum buffer
    let mut velocity: Array1<f64> = Array1::zeros(x.len());

    // Track the best solution found
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    // Convergence tracking
    let mut prev_loss = f64::INFINITY;
    let mut stagnant_iterations = 0;

    println!("Starting SGD with momentum optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", actual_batch_size);
    println!("  Initial learning rate: {}", options.learning_rate);
    println!("  Momentum: {}", options.momentum);
    println!("  Nesterov: {}", options.nesterov);
    println!("  Dampening: {}", options.dampening);

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

        // Compute gradient on batch at current point
        let mut gradient = grad_func.compute_gradient(&x.view(), &batch_data);
        _grad_evals += 1;

        // Apply gradient clipping if specified
        if let Some(clip_threshold) = options.gradient_clip {
            clip_gradients(&mut gradient, clip_threshold);
        }

        // Update velocity with momentum
        if iteration == 0 {
            // First iteration: initialize velocity
            velocity = gradient.clone();
        } else {
            // Subsequent iterations: apply momentum with optional dampening
            velocity = &velocity * options.momentum + &gradient * (1.0 - options.dampening);
        }

        // Update parameters
        if options.nesterov {
            // Nesterov accelerated gradient: x = x - lr * (momentum * v + grad)
            // Use standard Nesterov formulation to avoid double-counting momentum
            x = &x - (&velocity + &gradient * options.momentum) * current_lr;
        } else {
            // Classical momentum: x = x - lr * v
            x = &x - &velocity * current_lr;
        }

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
                let velocity_norm = velocity.mapv(|v| v * v).sum().sqrt();
                println!("  Iteration {}: loss = {:.6e}, |grad| = {:.3e}, |velocity| = {:.3e}, lr = {:.3e}",
                    iteration, current_loss, grad_norm, velocity_norm, current_lr);
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
                        "SGD with momentum converged: loss change {:.2e} < {:.2e}",
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
                    message: "SGD with momentum stopped due to stagnation".to_string(),
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
        message: "SGD with momentum reached maximum iterations".to_string(),
        jacobian: None,
        hessian: None,
    })
}

/// Heavy-ball momentum (Polyak momentum) implementation
pub fn minimize_heavy_ball<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: MomentumOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize for heavy-ball method
    let mut x_prev = x.clone();
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    println!("Starting Heavy-Ball momentum optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", actual_batch_size);
    println!("  Momentum: {}", options.momentum);

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

        // Heavy-ball update: x_new = x - lr * grad + momentum * (x - x_prev)
        let momentum_term = if iteration > 0 {
            (&x - &x_prev) * options.momentum
        } else {
            Array1::<f64>::zeros(x.len())
        };

        let x_new = &x - &gradient * current_lr + &momentum_term;

        // Update for next iteration
        x_prev = x.clone();
        x = x_new;

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
        message: "Heavy-ball momentum completed".to_string(),
        jacobian: None,
        hessian: None,
    })
}

/// Adaptive momentum based on gradient correlation
pub fn minimize_adaptive_momentum<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    mut options: MomentumOptions,
    correlation_window: usize,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize adaptive momentum variables
    let mut velocity: Array1<f64> = Array1::zeros(x.len());
    let mut gradient_history: Vec<Array1<f64>> = Vec::new();
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    println!("Starting Adaptive Momentum optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Correlation window: {}", correlation_window);

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

        // Store gradient in history
        gradient_history.push(gradient.clone());
        if gradient_history.len() > correlation_window {
            gradient_history.remove(0);
        }

        // Adapt momentum based on gradient correlation
        if gradient_history.len() >= 2 {
            let recent_grad = &gradient_history[gradient_history.len() - 1];
            let prev_grad = &gradient_history[gradient_history.len() - 2];

            // Compute cosine similarity
            let dot_product = recent_grad.dot(prev_grad);
            let norm_recent = recent_grad.mapv(|g| g * g).sum().sqrt();
            let norm_prev = prev_grad.mapv(|g| g * g).sum().sqrt();

            let cosine_similarity = if norm_recent > 1e-10 && norm_prev > 1e-10 {
                dot_product / (norm_recent * norm_prev)
            } else {
                0.0
            };

            // Adapt momentum: higher correlation → higher momentum
            options.momentum = (0.5 + 0.4 * cosine_similarity.max(0.0)).min(0.99);
        }

        // Update velocity and parameters
        velocity = &velocity * options.momentum + &gradient;
        x = &x - &velocity * current_lr;

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
                    "  Iteration {}: loss = {:.6e}, |grad| = {:.3e}, momentum = {:.3}, lr = {:.3e}",
                    iteration, current_loss, grad_norm, options.momentum, current_lr
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
        message: "Adaptive momentum completed".to_string(),
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
    fn test_sgd_momentum_quadratic() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, 2.0, -1.5]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = MomentumOptions {
            learning_rate: 0.1,
            momentum: 0.9,
            max_iter: 200,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_sgd_momentum(grad_func, x0, data_provider, options).unwrap();

        // Should converge to zero faster than plain SGD
        assert!(result.success || result.fun < 1e-4);
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_nesterov_momentum() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, -1.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = MomentumOptions {
            learning_rate: 0.01,
            momentum: 0.9,
            nesterov: true,
            max_iter: 500,
            batch_size: Some(10),
            tol: 1e-4,
            ..Default::default()
        };

        let result = minimize_sgd_momentum(grad_func, x0, data_provider, options).unwrap();

        // Nesterov should accelerate convergence
        assert!(result.success || result.fun < 1e-4);
    }

    #[test]
    fn test_momentum_dampening() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![2.0, -2.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = MomentumOptions {
            learning_rate: 0.01,
            momentum: 0.9,
            dampening: 0.1, // Add dampening to reduce oscillations
            max_iter: 200,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_sgd_momentum(grad_func, x0, data_provider, options).unwrap();

        // Dampening should still allow convergence
        assert!(result.success || result.fun < 1e-3);
    }

    #[test]
    fn test_heavy_ball_momentum() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.5, -1.5]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = MomentumOptions {
            learning_rate: 0.1,
            momentum: 0.8,
            max_iter: 500,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_heavy_ball(grad_func, x0, data_provider, options).unwrap();

        // Heavy-ball should converge
        assert!(result.fun < 1e-3);
    }

    #[test]
    fn test_adaptive_momentum() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![2.0, -2.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = MomentumOptions {
            learning_rate: 0.05,
            momentum: 0.5, // Will be adapted
            max_iter: 150,
            batch_size: Some(20),
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_adaptive_momentum(grad_func, x0, data_provider, options, 5).unwrap();

        // Adaptive momentum should help convergence
        assert!(result.fun < 1e-3);
    }

    #[test]
    fn test_momentum_learning_rate_schedules() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, 1.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        // Test with exponential decay
        let options = MomentumOptions {
            learning_rate: 0.1,
            momentum: 0.9,
            max_iter: 500,
            lr_schedule: LearningRateSchedule::ExponentialDecay { decay_rate: 0.95 },
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_sgd_momentum(grad_func, x0, data_provider, options).unwrap();

        // Should converge with decaying learning rate
        assert!(result.success || result.fun < 1e-3);
    }

    #[test]
    fn test_momentum_comparison() {
        let x0 = Array1::from_vec(vec![3.0, -3.0]);

        // Test standard momentum
        let grad_func1 = QuadraticFunction;
        let data_provider1 = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
        let options_standard = MomentumOptions {
            learning_rate: 0.01,
            momentum: 0.9,
            nesterov: false,
            max_iter: 500,
            ..Default::default()
        };
        let result_standard =
            minimize_sgd_momentum(grad_func1, x0.clone(), data_provider1, options_standard)
                .unwrap();

        // Test Nesterov momentum
        let grad_func2 = QuadraticFunction;
        let data_provider2 = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
        let options_nesterov = MomentumOptions {
            learning_rate: 0.01,
            momentum: 0.9,
            nesterov: true,
            max_iter: 500,
            ..Default::default()
        };
        let result_nesterov =
            minimize_sgd_momentum(grad_func2, x0, data_provider2, options_nesterov).unwrap();

        // Both should converge
        assert!(result_standard.fun < 1e-2);
        assert!(result_nesterov.fun < 1e-2);

        // Nesterov might converge faster for this problem
        // (Not guaranteed for all problems, but often the case)
    }
}
