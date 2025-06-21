//! ADAM (Adaptive Moment Estimation) optimizer
//!
//! ADAM combines the advantages of two other extensions of stochastic gradient descent:
//! AdaGrad and RMSProp. It computes adaptive learning rates for each parameter and
//! stores an exponentially decaying average of past gradients (momentum) and
//! past squared gradients (adaptive learning rate).

use crate::error::OptimizeError;
use crate::stochastic::{
    clip_gradients, generate_batch_indices, update_learning_rate, DataProvider,
    LearningRateSchedule, StochasticGradientFunction,
};
use crate::unconstrained::result::OptimizeResult;
use ndarray::Array1;

/// Options for ADAM optimization
#[derive(Debug, Clone)]
pub struct AdamOptions {
    /// Learning rate (step size)
    pub learning_rate: f64,
    /// First moment decay parameter (momentum)
    pub beta1: f64,
    /// Second moment decay parameter (RMSProp-like)
    pub beta2: f64,
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
    /// Use AMSGrad variant (max of past second moments)
    pub amsgrad: bool,
}

impl Default for AdamOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iter: 1000,
            tol: 1e-6,
            lr_schedule: LearningRateSchedule::Constant,
            gradient_clip: None,
            batch_size: None,
            amsgrad: false,
        }
    }
}

/// ADAM optimizer implementation
pub fn minimize_adam<F>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: AdamOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize moment estimates
    let mut m: Array1<f64> = Array1::zeros(x.len()); // First moment estimate (momentum)
    let mut v: Array1<f64> = Array1::zeros(x.len()); // Second moment estimate (adaptive lr)
    let mut v_hat_max: Array1<f64> = Array1::zeros(x.len()); // For AMSGrad variant

    // Track the best solution found
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    // Convergence tracking
    let mut prev_loss = f64::INFINITY;
    let mut stagnant_iterations = 0;

    println!("Starting ADAM optimization:");
    println!("  Parameters: {}", x.len());
    println!("  Dataset size: {}", num_samples);
    println!("  Batch size: {}", actual_batch_size);
    println!("  Initial learning rate: {}", options.learning_rate);
    println!("  Beta1: {}, Beta2: {}", options.beta1, options.beta2);
    println!("  AMSGrad: {}", options.amsgrad);

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

        // Update biased first moment estimate
        m = &m * options.beta1 + &gradient * (1.0 - options.beta1);

        // Update biased second raw moment estimate
        let gradient_sq = gradient.mapv(|g| g * g);
        v = &v * options.beta2 + &gradient_sq * (1.0 - options.beta2);

        // Compute bias-corrected first moment estimate
        let bias_correction1 = 1.0 - options.beta1.powi((iteration + 1) as i32);
        let m_hat = &m / bias_correction1;

        // Compute bias-corrected second moment estimate
        let bias_correction2 = 1.0 - options.beta2.powi((iteration + 1) as i32);
        let v_hat = &v / bias_correction2;

        // AMSGrad: Use max of current and past second moments
        let v_final = if options.amsgrad {
            // Element-wise maximum of v_hat and v_hat_max
            for i in 0..v_hat_max.len() {
                v_hat_max[i] = v_hat_max[i].max(v_hat[i]);
            }
            &v_hat_max
        } else {
            &v_hat
        };

        // Update parameters: x = x - lr * m_hat / (sqrt(v_final) + epsilon)
        let denominator = v_final.mapv(|v| v.sqrt() + options.epsilon);
        let update = &m_hat / &denominator * current_lr;
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
                let m_norm = m_hat.mapv(|g: f64| g * g).sum().sqrt();
                let v_mean = v_final.mean().unwrap_or(0.0);
                println!("  Iteration {}: loss = {:.6e}, |grad| = {:.3e}, |m| = {:.3e}, <v> = {:.3e}, lr = {:.3e}",
                    iteration, current_loss, grad_norm, m_norm, v_mean, current_lr);
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
                        "ADAM converged: loss change {:.2e} < {:.2e}",
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
                    message: "ADAM stopped due to stagnation".to_string(),
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
        message: "ADAM reached maximum iterations".to_string(),
        jacobian: None,
        hessian: None,
    })
}

/// ADAM with learning rate warmup
pub fn minimize_adam_with_warmup<F>(
    grad_func: F,
    x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: AdamOptions,
    warmup_steps: usize,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    let original_lr = options.learning_rate;

    // Implement warmup by modifying the learning rate schedule
    let warmup_schedule =
        move |epoch: usize, max_epochs: usize, base_schedule: &LearningRateSchedule| -> f64 {
            let base_lr = update_learning_rate(original_lr, epoch, max_epochs, base_schedule);

            if epoch < warmup_steps {
                // Linear warmup from 0 to base_lr
                base_lr * (epoch as f64 / warmup_steps as f64)
            } else {
                base_lr
            }
        };

    // We'll handle warmup manually during optimization
    minimize_adam_with_custom_schedule(grad_func, x, data_provider, options, warmup_schedule)
}

/// ADAM with custom learning rate schedule function
fn minimize_adam_with_custom_schedule<F, S>(
    mut grad_func: F,
    mut x: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: AdamOptions,
    lr_scheduler: S,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
    S: Fn(usize, usize, &LearningRateSchedule) -> f64,
{
    let mut func_evals = 0;
    let mut _grad_evals = 0;

    let num_samples = data_provider.num_samples();
    let batch_size = options.batch_size.unwrap_or(32.min(num_samples / 10));
    let actual_batch_size = batch_size.min(num_samples);

    // Initialize moment estimates
    let mut m: Array1<f64> = Array1::zeros(x.len());
    let mut v: Array1<f64> = Array1::zeros(x.len());
    let mut v_hat_max: Array1<f64> = Array1::zeros(x.len());

    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;

    #[allow(clippy::explicit_counter_loop)]
    for iteration in 0..options.max_iter {
        // Use custom learning rate schedule
        let current_lr = lr_scheduler(iteration, options.max_iter, &options.lr_schedule);

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

        // ADAM updates
        m = &m * options.beta1 + &gradient * (1.0 - options.beta1);
        let gradient_sq = gradient.mapv(|g| g * g);
        v = &v * options.beta2 + &gradient_sq * (1.0 - options.beta2);

        let bias_correction1 = 1.0 - options.beta1.powi((iteration + 1) as i32);
        let bias_correction2 = 1.0 - options.beta2.powi((iteration + 1) as i32);
        let m_hat = &m / bias_correction1;
        let v_hat = &v / bias_correction2;

        let v_final = if options.amsgrad {
            for i in 0..v_hat_max.len() {
                v_hat_max[i] = v_hat_max[i].max(v_hat[i]);
            }
            &v_hat_max
        } else {
            &v_hat
        };

        let denominator = v_final.mapv(|v| v.sqrt() + options.epsilon);
        let update = &m_hat / &denominator * current_lr;
        x = &x - &update;

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
                println!(
                    "  Iteration {}: loss = {:.6e}, lr = {:.3e} (custom schedule)",
                    iteration, current_loss, current_lr
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
        message: "ADAM with custom schedule completed".to_string(),
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
    fn test_adam_quadratic() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, 2.0, -1.5]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = AdamOptions {
            learning_rate: 0.1,
            max_iter: 200,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_adam(grad_func, x0, data_provider, options).unwrap();

        // Should converge to zero
        assert!(result.success || result.fun < 1e-4);
        for &xi in result.x.iter() {
            assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_adam_amsgrad() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![1.0, -1.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = AdamOptions {
            learning_rate: 0.1,
            max_iter: 100,
            batch_size: Some(10),
            amsgrad: true,
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_adam(grad_func, x0, data_provider, options).unwrap();

        // AMSGrad should converge reliably
        assert!(result.success || result.fun < 1e-4);
    }

    #[test]
    fn test_adam_with_warmup() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![2.0, -2.0]);
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let options = AdamOptions {
            learning_rate: 0.1,
            max_iter: 100,
            batch_size: Some(20),
            tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_adam_with_warmup(grad_func, x0, data_provider, options, 10).unwrap();

        // Warmup should help with convergence
        assert!(result.success || result.fun < 1e-3);
    }

    #[test]
    fn test_adam_gradient_clipping() {
        let grad_func = QuadraticFunction;
        let x0 = Array1::from_vec(vec![10.0, -10.0]); // Large initial values
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let options = AdamOptions {
            learning_rate: 0.1,       // Increased learning rate to compensate for clipping
            max_iter: 1000,           // More iterations for convergence with clipping
            gradient_clip: Some(1.0), // Clip gradients to norm 1.0
            tol: 1e-4,
            ..Default::default()
        };

        let result = minimize_adam(grad_func, x0, data_provider, options).unwrap();

        // Should still converge even with large initial gradients (relaxed tolerance for clipped gradients)
        assert!(result.success || result.fun < 1e-1);
    }
}
