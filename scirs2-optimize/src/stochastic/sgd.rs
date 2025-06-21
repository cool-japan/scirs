//! Stochastic Gradient Descent (SGD) optimization
//!
//! This module implements the basic SGD algorithm and its variants for
//! stochastic optimization problems.

use crate::error::OptimizeError;
use crate::stochastic::{
    clip_gradients, generate_batch_indices, update_learning_rate, DataProvider,
    LearningRateSchedule, StochasticGradientFunction,
};
use crate::unconstrained::result::OptimizeResult;
use ndarray::Array1;

/// Options for SGD optimization
#[derive(Debug, Clone)]
pub struct SGDOptions {
    /// Learning rate (step size)
    pub learning_rate: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Batch size for mini-batch SGD
    pub batch_size: Option<usize>,
}

impl Default for SGDOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            lr_schedule: LearningRateSchedule::Constant,
            gradient_clip: None,
            batch_size: None,
        }
    }
}

/// Stochastic Gradient Descent optimizer
pub fn minimize_sgd<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: SGDOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(num_samples);
    let actual_batch_size = batch_size.min(num_samples);

    // Track the best solution found
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    // Convergence tracking
    let mut prev_loss = f64::INFINITY;
    let mut stagnant_iterations = 0;

    println!("Starting SGD optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", actual_batch_size);
    println!("  Initial learning rate: {}", options.learning_rate);

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

        // SGD update: x = x - lr * gradient
        x = &x - &(&gradient * current_lr);

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
                println!(
                    "  Iteration {}: loss = {:.6e}, |grad| = {:.3e}, lr = {:.3e}",
                    iteration, current_loss, grad_norm, current_lr
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
                        "SGD converged: loss change {:.2e} < {:.2e}",
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
                    message: "SGD stopped due to stagnation".to_string(),
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
        message: "SGD reached maximum iterations".to_string(),
        jacobian: None,
        hessian: None,
    })
}

/// Variance-reduced SGD using SVRG (Stochastic Variance Reduced Gradient)
pub fn minimize_svrg<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: SGDOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(1);
    let update_frequency = num_samples / batch_size; // Full pass frequency

    // Compute full gradient initially
    let full_data = data_provider.get_full_data();
    let mut full_gradient = grad_func.compute_gradient(&x.view(), &full_data);
    _grad_evals += 1;

    let mut x_snapshot = x.clone();
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    println!("Starting SVRG optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", batch_size);
    println!("  Update frequency: {}", update_frequency);

    for epoch in 0..options.max_iter {
        let current_lr = update_learning_rate(
            options.learning_rate,
            epoch,
            options.max_iter,
            &options.lr_schedule,
        );

        // Inner loop: one pass through data
        for _inner_iter in 0..update_frequency {
            // Generate batch indices
            let batch_indices = generate_batch_indices(num_samples, batch_size, true);
            let batch_data = data_provider.get_batch(&batch_indices);

            // Compute stochastic gradient
            let stoch_grad = grad_func.compute_gradient(&x.view(), &batch_data);
            _grad_evals += 1;

            // Compute control variate gradient at snapshot
            let control_grad = grad_func.compute_gradient(&x_snapshot.view(), &batch_data);
            _grad_evals += 1;

            // SVRG gradient estimate: g_i - g_i(snapshot) + full_gradient
            let mut svrg_gradient = &stoch_grad - &control_grad + &full_gradient;

            // Apply gradient clipping
            if let Some(clip_threshold) = options.gradient_clip {
                clip_gradients(&mut svrg_gradient, clip_threshold);
            }

            // Update parameters
            x = &x - &(&svrg_gradient * current_lr);
        }

        // Update snapshot and full gradient
        x_snapshot = x.clone();
        full_gradient = grad_func.compute_gradient(&x_snapshot.view(), &full_data);
        _grad_evals += 1;

        // Evaluate progress
        let current_loss = grad_func.compute_value(&x.view(), &full_data);
        func_evals += 1;

        if current_loss < best_f {
            best_f = current_loss;
            best_x = x.clone();
        }

        if epoch % 10 == 0 {
            let grad_norm = full_gradient.mapv(|g| g * g).sum().sqrt();
            println!(
                "  Epoch {}: loss = {:.6e}, |grad| = {:.3e}, lr = {:.3e}",
                epoch, current_loss, grad_norm, current_lr
            );
        }

        // Check convergence
        let grad_norm = full_gradient.mapv(|g| g * g).sum().sqrt();
        if grad_norm < options.tol {
            return Ok(OptimizeResult {
                x: best_x,
                fun: best_f,
                iterations: epoch,
                nit: epoch,
                func_evals,
                nfev: func_evals,
                success: true,
                message: format!(
                    "SVRG converged: gradient norm {:.2e} < {:.2e}",
                    grad_norm, options.tol
                ),
                jacobian: Some(full_gradient),
                hessian: None,
            });
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
        message: "SVRG reached maximum iterations".to_string(),
        jacobian: Some(full_gradient),
        hessian: None,
    })
}

/// Mini-batch SGD with averaging for better convergence
pub fn minimize_mini_batch_sgd<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: SGDOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let batches_per_epoch = num_samples.div_ceil(batch_size);

    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    // Moving average for parameters (Polyak averaging)
    let mut x_avg = x.clone();
    let avg_start_epoch = options.max_iter / 4; // Start averaging after 25% of iterations

    println!("Starting Mini-batch SGD optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", batch_size);
    println!("  Batches per epoch: {}", batches_per_epoch);

    for epoch in 0..options.max_iter {
        let current_lr = update_learning_rate(
            options.learning_rate,
            epoch,
            options.max_iter,
            &options.lr_schedule,
        );

        // Shuffle data indices for this epoch
        let mut all_indices: Vec<usize> = (0..num_samples).collect();
        use rand::rng;
        use rand::seq::SliceRandom;
        all_indices.shuffle(&mut rng());

        let mut _epoch_loss = 0.0;
        let mut epoch_grad_norm = 0.0;

        // Process all batches in epoch
        for batch_idx in 0..batches_per_epoch {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(num_samples);
            let batch_indices = &all_indices[start_idx..end_idx];

            let batch_data = data_provider.get_batch(batch_indices);

            // Compute gradient on batch
            let mut gradient = grad_func.compute_gradient(&x.view(), &batch_data);
            _grad_evals += 1;

            // Apply gradient clipping
            if let Some(clip_threshold) = options.gradient_clip {
                clip_gradients(&mut gradient, clip_threshold);
            }

            // Update parameters
            x = &x - &(&gradient * current_lr);

            // Update running averages
            let grad_norm = gradient.mapv(|g| g * g).sum().sqrt();
            epoch_grad_norm += grad_norm;

            let batch_loss = grad_func.compute_value(&x.view(), &batch_data);
            func_evals += 1;
            _epoch_loss += batch_loss;
        }

        // Update Polyak averaging
        if epoch >= avg_start_epoch {
            let weight = 1.0 / (epoch - avg_start_epoch + 1) as f64;
            x_avg = &x_avg * (1.0 - weight) + &x * weight;
        }

        // Use averaged parameters for evaluation after averaging starts
        let eval_x = if epoch >= avg_start_epoch { &x_avg } else { &x };

        // Evaluate on full dataset
        let full_data = data_provider.get_full_data();
        let current_loss = grad_func.compute_value(&eval_x.view(), &full_data);
        func_evals += 1;

        if current_loss < best_f {
            best_f = current_loss;
            best_x = eval_x.clone();
        }

        // Progress reporting
        if epoch % 10 == 0 {
            let avg_grad_norm = epoch_grad_norm / batches_per_epoch as f64;
            println!(
                "  Epoch {}: loss = {:.6e}, avg |grad| = {:.3e}, lr = {:.3e}{}",
                epoch,
                current_loss,
                avg_grad_norm,
                current_lr,
                if epoch >= avg_start_epoch {
                    " (averaged)"
                } else {
                    ""
                }
            );
        }

        // Check convergence
        let avg_grad_norm = epoch_grad_norm / batches_per_epoch as f64;
        if avg_grad_norm < options.tol {
            return Ok(OptimizeResult {
                x: best_x,
                fun: best_f,
                iterations: epoch,
                nit: epoch,
                func_evals,
                nfev: func_evals,
                success: true,
                message: format!(
                    "Mini-batch SGD converged: avg gradient norm {:.2e} < {:.2e}",
                    avg_grad_norm, options.tol
                ),
                jacobian: None,
                hessian: None,
            });
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
        message: "Mini-batch SGD reached maximum iterations".to_string(),
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
    fn test_sgd_quadratic() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, 2.0, -1.5]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = SGDOptions {
            learning_rate: 0.1,
            max_iter: 100,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_sgd(grad_func, x0, data_provider, options).unwrap();

        // Should converge to zero
        assert!(result.success || result.fun < 1e-4);
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_svrg_quadratic() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, -1.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = SGDOptions {
            learning_rate: 0.05,
            max_iter: 50,
            batch_size: Some(5),
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_svrg(grad_func, x0, data_provider, options).unwrap();

        // SVRG should converge faster than regular SGD
        assert!(result.success || result.fun < 1e-4);
    }

    #[test]
    fn test_mini_batch_sgd() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![2.0, -2.0, 1.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 200]));

        let options = SGDOptions {
            learning_rate: 0.01,
            max_iter: 100,
            batch_size: Some(10),
            tol: 1e-6,
            lr_schedule: LearningRateSchedule::ExponentialDecay { decay_rate: 0.99 },
            ..Default::default()
        };

        let result = minimize_mini_batch_sgd(grad_func, x0, data_provider, options).unwrap();

        // Should converge with Polyak averaging
        assert!(result.success || result.fun < 1e-3);
    }
}
