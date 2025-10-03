//! Stochastic optimization methods for machine learning and large-scale problems
//!
//! This module provides stochastic optimization algorithms that are particularly
//! well-suited for machine learning, neural networks, and large-scale problems
//! where exact gradients are expensive or noisy.

pub mod adam;
pub mod adamw;
pub mod momentum;
pub mod rmsprop;
pub mod sgd;

// Re-export commonly used items
pub use adam::{minimize_adam, AdamOptions};
pub use adamw::{minimize_adamw, AdamWOptions};
pub use momentum::{minimize_sgd_momentum, MomentumOptions};
pub use rmsprop::{minimize_rmsprop, RMSPropOptions};
pub use sgd::{minimize_sgd, SGDOptions};

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::prelude::*;

/// Stochastic optimization method selection
#[derive(Debug, Clone, Copy)]
pub enum StochasticMethod {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with Momentum
    Momentum,
    /// Root Mean Square Propagation
    RMSProp,
    /// Adaptive Moment Estimation
    Adam,
    /// Adam with Weight Decay
    AdamW,
}

/// Common options for stochastic optimization
#[derive(Debug, Clone)]
pub struct StochasticOptions {
    /// Learning rate (step size)
    pub learning_rate: f64,
    /// Maximum number of iterations (epochs)
    pub max_iter: usize,
    /// Batch size for mini-batch optimization
    pub batch_size: Option<usize>,
    /// Tolerance for convergence
    pub tol: f64,
    /// Whether to use adaptive learning rate
    pub adaptive_lr: bool,
    /// Learning rate decay factor
    pub lr_decay: f64,
    /// Learning rate decay schedule
    pub lr_schedule: LearningRateSchedule,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}

impl Default for StochasticOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            max_iter: 1000,
            batch_size: None,
            tol: 1e-6,
            adaptive_lr: false,
            lr_decay: 0.99,
            lr_schedule: LearningRateSchedule::Constant,
            gradient_clip: None,
            early_stopping_patience: None,
        }
    }
}

/// Learning rate schedules
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Exponential decay: lr * decay^epoch
    ExponentialDecay { decay_rate: f64 },
    /// Step decay: lr * decay_factor every decay_steps
    StepDecay {
        decay_factor: f64,
        decay_steps: usize,
    },
    /// Linear decay: lr * (1 - epoch/max_epochs)
    LinearDecay,
    /// Cosine annealing: lr * 0.5 * (1 + cos(π * epoch/max_epochs))
    CosineAnnealing,
    /// Inverse time decay: lr / (1 + decay_rate * epoch)
    InverseTimeDecay { decay_rate: f64 },
}

/// Data provider trait for stochastic optimization
pub trait DataProvider {
    /// Get the total number of samples
    fn num_samples(&self) -> usize;

    /// Get a batch of samples
    fn get_batch(&self, indices: &[usize]) -> Vec<f64>;

    /// Get the full dataset
    fn get_full_data(&self) -> Vec<f64>;
}

/// Simple in-memory data provider
#[derive(Clone)]
pub struct InMemoryDataProvider {
    data: Vec<f64>,
}

impl InMemoryDataProvider {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }
}

impl DataProvider for InMemoryDataProvider {
    fn num_samples(&self) -> usize {
        self.data.len()
    }

    fn get_batch(&self, indices: &[usize]) -> Vec<f64> {
        indices.iter().map(|&i| self.data[i]).collect()
    }

    fn get_full_data(&self) -> Vec<f64> {
        self.data.clone()
    }
}

/// Stochastic gradient function trait
pub trait StochasticGradientFunction {
    /// Compute gradient on a batch of data
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, batchdata: &[f64]) -> Array1<f64>;

    /// Compute function value on a batch of data
    fn compute_value(&mut self, x: &ArrayView1<f64>, batchdata: &[f64]) -> f64;
}

/// Wrapper for regular gradient functions
pub struct BatchGradientWrapper<F, G> {
    func: F,
    grad: G,
}

impl<F, G> BatchGradientWrapper<F, G>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    pub fn new(func: F, grad: G) -> Self {
        Self { func, grad }
    }
}

impl<F, G> StochasticGradientFunction for BatchGradientWrapper<F, G>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> Array1<f64> {
        (self.grad)(x)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> f64 {
        (self.func)(x)
    }
}

/// Update learning rate according to schedule
#[allow(dead_code)]
pub fn update_learning_rate(
    initial_lr: f64,
    epoch: usize,
    max_epochs: usize,
    schedule: &LearningRateSchedule,
) -> f64 {
    match schedule {
        LearningRateSchedule::Constant => initial_lr,
        LearningRateSchedule::ExponentialDecay { decay_rate } => {
            initial_lr * decay_rate.powi(epoch as i32)
        }
        LearningRateSchedule::StepDecay {
            decay_factor,
            decay_steps,
        } => initial_lr * decay_factor.powi((epoch / decay_steps) as i32),
        LearningRateSchedule::LinearDecay => {
            initial_lr * (1.0 - epoch as f64 / max_epochs as f64).max(0.0)
        }
        LearningRateSchedule::CosineAnnealing => {
            initial_lr
                * 0.5
                * (1.0 + (std::f64::consts::PI * epoch as f64 / max_epochs as f64).cos())
        }
        LearningRateSchedule::InverseTimeDecay { decay_rate } => {
            initial_lr / (1.0 + decay_rate * epoch as f64)
        }
    }
}

/// Clip gradients to prevent exploding gradients
#[allow(dead_code)]
pub fn clip_gradients(gradient: &mut Array1<f64>, maxnorm: f64) {
    let grad_norm = gradient.mapv(|x| x * x).sum().sqrt();
    if grad_norm > maxnorm {
        let scale = maxnorm / grad_norm;
        gradient.mapv_inplace(|x| x * scale);
    }
}

/// Generate random batch indices
#[allow(dead_code)]
pub fn generate_batch_indices(_num_samples: usize, batchsize: usize, shuffle: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0.._num_samples).collect();

    if shuffle {
        use scirs2_core::random::seq::SliceRandom;
        indices.shuffle(&mut thread_rng());
    }

    indices.into_iter().take(batchsize).collect()
}

/// Main stochastic optimization function
#[allow(dead_code)]
pub fn minimize_stochastic<F>(
    method: StochasticMethod,
    grad_func: F,
    x0: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: StochasticOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: StochasticGradientFunction,
{
    match method {
        StochasticMethod::SGD => {
            let sgd_options = SGDOptions {
                learning_rate: options.learning_rate,
                max_iter: options.max_iter,
                tol: options.tol,
                lr_schedule: options.lr_schedule,
                gradient_clip: options.gradient_clip,
                batch_size: options.batch_size,
            };
            sgd::minimize_sgd(grad_func, x0, data_provider, sgd_options)
        }
        StochasticMethod::Momentum => {
            let momentum_options = MomentumOptions {
                learning_rate: options.learning_rate,
                momentum: 0.9, // Default momentum
                max_iter: options.max_iter,
                tol: options.tol,
                lr_schedule: options.lr_schedule,
                gradient_clip: options.gradient_clip,
                batch_size: options.batch_size,
                nesterov: false,
                dampening: 0.0,
            };
            momentum::minimize_sgd_momentum(grad_func, x0, data_provider, momentum_options)
        }
        StochasticMethod::RMSProp => {
            let rmsprop_options = RMSPropOptions {
                learning_rate: options.learning_rate,
                decay_rate: 0.99, // Default RMSProp decay
                epsilon: 1e-8,
                max_iter: options.max_iter,
                tol: options.tol,
                lr_schedule: options.lr_schedule,
                gradient_clip: options.gradient_clip,
                batch_size: options.batch_size,
                centered: false,
                momentum: None,
            };
            rmsprop::minimize_rmsprop(grad_func, x0, data_provider, rmsprop_options)
        }
        StochasticMethod::Adam => {
            let adam_options = AdamOptions {
                learning_rate: options.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                max_iter: options.max_iter,
                tol: options.tol,
                lr_schedule: options.lr_schedule,
                gradient_clip: options.gradient_clip,
                batch_size: options.batch_size,
                amsgrad: false,
            };
            adam::minimize_adam(grad_func, x0, data_provider, adam_options)
        }
        StochasticMethod::AdamW => {
            let adamw_options = AdamWOptions {
                learning_rate: options.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.01, // Default weight decay
                max_iter: options.max_iter,
                tol: options.tol,
                lr_schedule: options.lr_schedule,
                gradient_clip: options.gradient_clip,
                batch_size: options.batch_size,
                decouple_weight_decay: true,
            };
            adamw::minimize_adamw(grad_func, x0, data_provider, adamw_options)
        }
    }
}

/// Create stochastic options optimized for specific problem types
#[allow(dead_code)]
pub fn create_stochastic_options_for_problem(
    problem_type: &str,
    dataset_size: usize,
) -> StochasticOptions {
    match problem_type.to_lowercase().as_str() {
        "neural_network" => StochasticOptions {
            learning_rate: 0.001,
            max_iter: 1000,
            batch_size: Some(32.min(dataset_size / 10)),
            lr_schedule: LearningRateSchedule::ExponentialDecay { decay_rate: 0.99 },
            gradient_clip: Some(1.0),
            early_stopping_patience: Some(50),
            ..Default::default()
        },
        "linear_regression" => StochasticOptions {
            learning_rate: 0.01,
            max_iter: 500,
            batch_size: Some(64.min(dataset_size / 5)),
            lr_schedule: LearningRateSchedule::LinearDecay,
            ..Default::default()
        },
        "logistic_regression" => StochasticOptions {
            learning_rate: 0.01,
            max_iter: 200,
            batch_size: Some(32.min(dataset_size / 10)),
            lr_schedule: LearningRateSchedule::StepDecay {
                decay_factor: 0.9,
                decay_steps: 50,
            },
            ..Default::default()
        },
        "large_scale" => StochasticOptions {
            learning_rate: 0.001,
            max_iter: 2000,
            batch_size: Some(128.min(dataset_size / 20)),
            lr_schedule: LearningRateSchedule::CosineAnnealing,
            gradient_clip: Some(5.0),
            adaptive_lr: true,
            ..Default::default()
        },
        "noisy_gradients" => StochasticOptions {
            learning_rate: 0.01,
            max_iter: 1000,
            batch_size: Some(64.min(dataset_size / 5)),
            lr_schedule: LearningRateSchedule::InverseTimeDecay { decay_rate: 1.0 },
            gradient_clip: Some(2.0),
            early_stopping_patience: Some(100),
            ..Default::default()
        },
        _ => StochasticOptions::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_learning_rate_schedules() {
        let initial_lr = 0.1;
        let max_epochs = 100;

        // Test constant schedule
        let constant = LearningRateSchedule::Constant;
        assert_abs_diff_eq!(
            update_learning_rate(initial_lr, 50, max_epochs, &constant),
            initial_lr,
            epsilon = 1e-10
        );

        // Test exponential decay
        let exp_decay = LearningRateSchedule::ExponentialDecay { decay_rate: 0.9 };
        let lr_exp = update_learning_rate(initial_lr, 10, max_epochs, &exp_decay);
        assert_abs_diff_eq!(lr_exp, initial_lr * 0.9_f64.powi(10), epsilon = 1e-10);

        // Test linear decay
        let linear = LearningRateSchedule::LinearDecay;
        let lr_linear = update_learning_rate(initial_lr, 50, max_epochs, &linear);
        assert_abs_diff_eq!(lr_linear, initial_lr * 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut grad = Array1::from_vec(vec![3.0, 4.0]); // Norm = 5
        clip_gradients(&mut grad, 2.5);

        let clipped_norm = grad.mapv(|x| x * x).sum().sqrt();
        assert_abs_diff_eq!(clipped_norm, 2.5, epsilon = 1e-10);

        // Check direction is preserved
        assert_abs_diff_eq!(grad[0] / grad[1], 3.0 / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_indices_generation() {
        let indices = generate_batch_indices(100, 10, false);
        assert_eq!(indices.len(), 10);
        assert_eq!(indices, (0..10).collect::<Vec<usize>>());

        let shuffled = generate_batch_indices(100, 10, true);
        assert_eq!(shuffled.len(), 10);
        // All indices should be < 100
        assert!(shuffled.iter().all(|&i| i < 100));
    }

    #[test]
    fn test_in_memory_data_provider() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let provider = InMemoryDataProvider::new(data.clone());

        assert_eq!(provider.num_samples(), 5);
        assert_eq!(provider.get_full_data(), data);

        let batch = provider.get_batch(&[0, 2, 4]);
        assert_eq!(batch, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_problem_specific_options() {
        let nn_options = create_stochastic_options_for_problem("neural_network", 1000);
        assert_eq!(nn_options.learning_rate, 0.001);
        assert!(nn_options.batch_size.is_some());
        assert!(nn_options.gradient_clip.is_some());

        let lr_options = create_stochastic_options_for_problem("linear_regression", 500);
        assert_eq!(lr_options.learning_rate, 0.01);
        assert!(matches!(
            lr_options.lr_schedule,
            LearningRateSchedule::LinearDecay
        ));

        let large_options = create_stochastic_options_for_problem("large_scale", 10000);
        assert!(matches!(
            large_options.lr_schedule,
            LearningRateSchedule::CosineAnnealing
        ));
        assert_eq!(large_options.batch_size, Some(128));
    }
}
