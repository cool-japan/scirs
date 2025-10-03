//! Integration with scirs2-neural for machine learning optimization
//!
//! This module provides wrappers and utilities to use scirs2-optimize's
//! stochastic optimizers with scirs2-neural's neural network models.

use crate::error::OptimizeError;
use crate::stochastic::{StochasticMethod, StochasticOptions};
use scirs2_core::ndarray::{s, Array1, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Neural network parameter container
#[derive(Debug, Clone)]
pub struct NeuralParameters<F: Float + ScalarOperand> {
    /// Parameter vectors for each layer
    pub parameters: Vec<Array1<F>>,
    /// Gradient vectors for each layer
    pub gradients: Vec<Array1<F>>,
    /// Parameter names/ids for tracking
    pub names: Vec<String>,
}

impl<F: Float + ScalarOperand> Default for NeuralParameters<F> {
    fn default() -> Self {
        Self {
            parameters: Vec::new(),
            gradients: Vec::new(),
            names: Vec::new(),
        }
    }
}

impl<F: Float + ScalarOperand> NeuralParameters<F> {
    /// Create new neural parameters container
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter vector
    pub fn add_parameter(&mut self, name: String, param: Array1<F>) {
        self.names.push(name);
        self.gradients.push(Array1::zeros(param.raw_dim()));
        self.parameters.push(param);
    }

    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        self.parameters.iter().map(|p| p.len()).sum()
    }

    /// Flatten all parameters into a single vector
    pub fn flatten_parameters(&self) -> Array1<F> {
        let total_len = self.total_parameters();
        let mut flat = Array1::zeros(total_len);
        let mut offset = 0;

        for param in &self.parameters {
            let len = param.len();
            flat.slice_mut(s![offset..offset + len]).assign(param);
            offset += len;
        }

        flat
    }

    /// Flatten all gradients into a single vector
    pub fn flatten_gradients(&self) -> Array1<F> {
        let total_len = self.total_parameters();
        let mut flat = Array1::zeros(total_len);
        let mut offset = 0;

        for grad in &self.gradients {
            let len = grad.len();
            flat.slice_mut(s![offset..offset + len]).assign(grad);
            offset += len;
        }

        flat
    }

    /// Update parameters from flattened vector
    pub fn update_from_flat(&mut self, flat_params: &Array1<F>) {
        let mut offset = 0;

        for param in &mut self.parameters {
            let len = param.len();
            param.assign(&flat_params.slice(s![offset..offset + len]));
            offset += len;
        }
    }

    /// Update gradients from flattened vector
    pub fn update_gradients_from_flat(&mut self, flat_grads: &Array1<F>) {
        let mut offset = 0;

        for grad in &mut self.gradients {
            let len = grad.len();
            grad.assign(&flat_grads.slice(s![offset..offset + len]));
            offset += len;
        }
    }
}

/// Neural network optimizer that uses scirs2-optimize stochastic methods
pub struct NeuralOptimizer<F: Float + ScalarOperand> {
    method: StochasticMethod,
    options: StochasticOptions,
    /// Internal state for momentum-based optimizers
    momentum_buffers: HashMap<String, Array1<F>>,
    /// Internal state for Adam-family optimizers
    first_moment: HashMap<String, Array1<F>>,
    second_moment: HashMap<String, Array1<F>>,
    /// Step counter for bias correction
    step_count: usize,
}

impl<F: Float + ScalarOperand> NeuralOptimizer<F>
where
    F: 'static + Send + Sync,
{
    /// Create a new neural optimizer
    pub fn new(method: StochasticMethod, options: StochasticOptions) -> Self {
        Self {
            method,
            options,
            momentum_buffers: HashMap::new(),
            first_moment: HashMap::new(),
            second_moment: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create SGD optimizer for neural networks
    pub fn sgd(learning_rate: F, max_iter: usize) -> Self {
        let options = StochasticOptions {
            learning_rate: learning_rate.to_f64().unwrap_or(0.01),
            max_iter,
            batch_size: None,
            tol: 1e-6,
            adaptive_lr: false,
            lr_decay: 0.99,
            lr_schedule: crate::stochastic::LearningRateSchedule::Constant,
            gradient_clip: None,
            early_stopping_patience: None,
        };

        Self::new(StochasticMethod::SGD, options)
    }

    /// Create Adam optimizer for neural networks
    pub fn adam(learning_rate: F, max_iter: usize) -> Self {
        let options = StochasticOptions {
            learning_rate: learning_rate.to_f64().unwrap_or(0.001),
            max_iter,
            batch_size: None,
            tol: 1e-6,
            adaptive_lr: false,
            lr_decay: 0.99,
            lr_schedule: crate::stochastic::LearningRateSchedule::Constant,
            gradient_clip: Some(1.0),
            early_stopping_patience: None,
        };

        Self::new(StochasticMethod::Adam, options)
    }

    /// Create AdamW optimizer for neural networks
    pub fn adamw(learning_rate: F, max_iter: usize) -> Self {
        let options = StochasticOptions {
            learning_rate: learning_rate.to_f64().unwrap_or(0.001),
            max_iter,
            batch_size: None,
            tol: 1e-6,
            adaptive_lr: false,
            lr_decay: 0.99,
            lr_schedule: crate::stochastic::LearningRateSchedule::Constant,
            gradient_clip: Some(1.0),
            early_stopping_patience: None,
        };

        Self::new(StochasticMethod::AdamW, options)
    }

    /// Update neural network parameters using the selected optimizer
    pub fn step(&mut self, params: &mut NeuralParameters<F>) -> Result<(), OptimizeError> {
        self.step_count += 1;

        match self.method {
            StochasticMethod::SGD => self.sgd_step(params),
            StochasticMethod::Momentum => self.momentum_step(params),
            StochasticMethod::Adam => self.adam_step(params),
            StochasticMethod::AdamW => self.adamw_step(params),
            StochasticMethod::RMSProp => self.rmsprop_step(params),
        }
    }

    /// SGD parameter update
    fn sgd_step(&self, params: &mut NeuralParameters<F>) -> Result<(), OptimizeError> {
        let lr = F::from(self.options.learning_rate).unwrap_or_else(|| F::from(0.01).unwrap());

        for (param, grad) in params.parameters.iter_mut().zip(params.gradients.iter()) {
            *param = param.clone() - &(grad.clone() * lr);
        }

        Ok(())
    }

    /// Momentum SGD parameter update
    fn momentum_step(&mut self, params: &mut NeuralParameters<F>) -> Result<(), OptimizeError> {
        let lr = F::from(self.options.learning_rate).unwrap_or_else(|| F::from(0.01).unwrap());
        let momentum = F::from(0.9).unwrap();

        for (i, (param, grad)) in params
            .parameters
            .iter_mut()
            .zip(params.gradients.iter())
            .enumerate()
        {
            let param_name = format!("param_{}", i);

            // Initialize momentum buffer if not exists
            if !self.momentum_buffers.contains_key(&param_name) {
                self.momentum_buffers
                    .insert(param_name.clone(), Array1::zeros(param.raw_dim()));
            }

            let momentum_buffer = self.momentum_buffers.get_mut(&param_name).unwrap();

            // v = momentum * v + grad
            *momentum_buffer = momentum_buffer.clone() * momentum + grad;

            // param = param - lr * v
            *param = param.clone() - &(momentum_buffer.clone() * lr);
        }

        Ok(())
    }

    /// Adam parameter update
    fn adam_step(&mut self, params: &mut NeuralParameters<F>) -> Result<(), OptimizeError> {
        let lr = F::from(self.options.learning_rate).unwrap_or_else(|| F::from(0.001).unwrap());
        let beta1 = F::from(0.9).unwrap();
        let beta2 = F::from(0.999).unwrap();
        let epsilon = F::from(1e-8).unwrap();

        for (i, (param, grad)) in params
            .parameters
            .iter_mut()
            .zip(params.gradients.iter())
            .enumerate()
        {
            let param_name = format!("param_{}", i);

            // Initialize moment buffers if not exists
            if !self.first_moment.contains_key(&param_name) {
                self.first_moment
                    .insert(param_name.clone(), Array1::zeros(param.raw_dim()));
                self.second_moment
                    .insert(param_name.clone(), Array1::zeros(param.raw_dim()));
            }

            let m = self.first_moment.get_mut(&param_name).unwrap();
            let v = self.second_moment.get_mut(&param_name).unwrap();

            // m = beta1 * m + (1 - beta1) * grad
            *m = m.clone() * beta1 + &(grad.clone() * (F::one() - beta1));

            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mapv(|x| x * x);
            *v = v.clone() * beta2 + &(grad_squared * (F::one() - beta2));

            // Bias correction
            let step_f = F::from(self.step_count).unwrap();
            let m_hat = m.clone() / (F::one() - beta1.powf(step_f));
            let v_hat = v.clone() / (F::one() - beta2.powf(step_f));

            // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
            let denominator = v_hat.mapv(|x| x.sqrt()) + epsilon;
            let update = m_hat / denominator * lr;
            *param = param.clone() - &update;
        }

        Ok(())
    }

    /// AdamW parameter update (Adam with decoupled weight decay)
    fn adamw_step(&mut self, params: &mut NeuralParameters<F>) -> Result<(), OptimizeError> {
        let lr = F::from(self.options.learning_rate).unwrap_or_else(|| F::from(0.001).unwrap());
        let beta1 = F::from(0.9).unwrap();
        let beta2 = F::from(0.999).unwrap();
        let epsilon = F::from(1e-8).unwrap();
        let weight_decay = F::from(0.01).unwrap();

        for (i, (param, grad)) in params
            .parameters
            .iter_mut()
            .zip(params.gradients.iter())
            .enumerate()
        {
            let param_name = format!("param_{}", i);

            // Initialize moment buffers if not exists
            if !self.first_moment.contains_key(&param_name) {
                self.first_moment
                    .insert(param_name.clone(), Array1::zeros(param.raw_dim()));
                self.second_moment
                    .insert(param_name.clone(), Array1::zeros(param.raw_dim()));
            }

            let m = self.first_moment.get_mut(&param_name).unwrap();
            let v = self.second_moment.get_mut(&param_name).unwrap();

            // m = beta1 * m + (1 - beta1) * grad
            *m = m.clone() * beta1 + &(grad.clone() * (F::one() - beta1));

            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mapv(|x| x * x);
            *v = v.clone() * beta2 + &(grad_squared * (F::one() - beta2));

            // Bias correction
            let step_f = F::from(self.step_count).unwrap();
            let m_hat = m.clone() / (F::one() - beta1.powf(step_f));
            let v_hat = v.clone() / (F::one() - beta2.powf(step_f));

            // AdamW update: param = param - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * param)
            let denominator = v_hat.mapv(|x| x.sqrt()) + epsilon;
            let adam_update = m_hat / denominator;
            let weight_decay_update = param.clone() * weight_decay;
            let total_update = (adam_update + weight_decay_update) * lr;

            *param = param.clone() - &total_update;
        }

        Ok(())
    }

    /// RMSprop parameter update
    fn rmsprop_step(&mut self, params: &mut NeuralParameters<F>) -> Result<(), OptimizeError> {
        let lr = F::from(self.options.learning_rate).unwrap_or_else(|| F::from(0.001).unwrap());
        let alpha = F::from(0.99).unwrap(); // decay rate
        let epsilon = F::from(1e-8).unwrap();

        for (i, (param, grad)) in params
            .parameters
            .iter_mut()
            .zip(params.gradients.iter())
            .enumerate()
        {
            let param_name = format!("param_{}", i);

            // Initialize squared average buffer if not exists
            if !self.second_moment.contains_key(&param_name) {
                self.second_moment
                    .insert(param_name.clone(), Array1::zeros(param.raw_dim()));
            }

            let v = self.second_moment.get_mut(&param_name).unwrap();

            // v = alpha * v + (1 - alpha) * grad^2
            let grad_squared = grad.mapv(|x| x * x);
            *v = v.clone() * alpha + &(grad_squared * (F::one() - alpha));

            // param = param - lr * grad / (sqrt(v) + epsilon)
            let denominator = v.mapv(|x| x.sqrt()) + epsilon;
            let update = grad.clone() / denominator * lr;
            *param = param.clone() - &update;
        }

        Ok(())
    }

    /// Get current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.options.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.options.learning_rate = lr;
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.momentum_buffers.clear();
        self.first_moment.clear();
        self.second_moment.clear();
        self.step_count = 0;
    }

    /// Get optimizer method name
    pub fn method_name(&self) -> &'static str {
        match self.method {
            StochasticMethod::SGD => "SGD",
            StochasticMethod::Momentum => "SGD with Momentum",
            StochasticMethod::Adam => "Adam",
            StochasticMethod::AdamW => "AdamW",
            StochasticMethod::RMSProp => "RMSprop",
        }
    }
}

/// Neural network trainer that combines optimization with training loops
pub struct NeuralTrainer<F: Float + ScalarOperand> {
    optimizer: NeuralOptimizer<F>,
    loss_history: Vec<F>,
    early_stopping_patience: Option<usize>,
    best_loss: Option<F>,
    patience_counter: usize,
}

impl<F: Float + ScalarOperand> NeuralTrainer<F>
where
    F: 'static + Send + Sync + std::fmt::Display,
{
    /// Create a new neural trainer
    pub fn new(optimizer: NeuralOptimizer<F>) -> Self {
        Self {
            optimizer,
            loss_history: Vec::new(),
            early_stopping_patience: None,
            best_loss: None,
            patience_counter: 0,
        }
    }

    /// Set early stopping patience
    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = Some(patience);
        self
    }

    /// Train for one epoch
    pub fn train_epoch<LossFn, GradFn>(
        &mut self,
        params: &mut NeuralParameters<F>,
        loss_fn: &mut LossFn,
        grad_fn: &mut GradFn,
    ) -> Result<F, OptimizeError>
    where
        LossFn: FnMut(&NeuralParameters<F>) -> F,
        GradFn: FnMut(&NeuralParameters<F>) -> Vec<Array1<F>>,
    {
        // Compute gradients
        let gradients = grad_fn(params);
        params.gradients = gradients;

        // Apply gradient clipping if specified
        if let Some(max_norm) = self.optimizer.options.gradient_clip {
            self.clip_gradients(params, max_norm);
        }

        // Update parameters
        self.optimizer.step(params)?;

        // Compute loss
        let loss = loss_fn(params);
        self.loss_history.push(loss);

        // Check early stopping
        if let Some(_patience) = self.early_stopping_patience {
            if let Some(best_loss) = self.best_loss {
                if loss < best_loss {
                    self.best_loss = Some(loss);
                    self.patience_counter = 0;
                } else {
                    self.patience_counter += 1;
                }
            } else {
                self.best_loss = Some(loss);
            }
        }

        Ok(loss)
    }

    /// Check if training should stop early
    pub fn should_stop_early(&self) -> bool {
        if let Some(patience) = self.early_stopping_patience {
            self.patience_counter >= patience
        } else {
            false
        }
    }

    /// Get loss history
    pub fn loss_history(&self) -> &[F] {
        &self.loss_history
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.get_learning_rate()
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Clip gradients to prevent exploding gradients
    fn clip_gradients(&self, params: &mut NeuralParameters<F>, max_norm: f64) {
        let max_norm_f = F::from(max_norm).unwrap();

        // Compute total gradient norm
        let mut total_norm_sq = F::zero();
        for grad in &params.gradients {
            total_norm_sq = total_norm_sq + grad.mapv(|x| x * x).sum();
        }
        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm_f {
            let scale = max_norm_f / total_norm;
            for grad in &mut params.gradients {
                grad.mapv_inplace(|x| x * scale);
            }
        }
    }
}

/// Convenience functions for creating neural optimizers
pub mod optimizers {
    use super::*;

    /// Create SGD optimizer with default settings for neural networks
    pub fn sgd<F>(learning_rate: F) -> NeuralOptimizer<F>
    where
        F: Float + ScalarOperand + 'static + Send + Sync,
    {
        NeuralOptimizer::sgd(learning_rate, 1000)
    }

    /// Create Adam optimizer with default settings for neural networks
    pub fn adam<F>(learning_rate: F) -> NeuralOptimizer<F>
    where
        F: Float + ScalarOperand + 'static + Send + Sync,
    {
        NeuralOptimizer::adam(learning_rate, 1000)
    }

    /// Create AdamW optimizer with default settings for neural networks
    pub fn adamw<F>(learning_rate: F) -> NeuralOptimizer<F>
    where
        F: Float + ScalarOperand + 'static + Send + Sync,
    {
        NeuralOptimizer::adamw(learning_rate, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_neural_parameters() {
        let mut params = NeuralParameters::<f64>::new();

        // Add some parameters
        params.add_parameter("layer1".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
        params.add_parameter("layer2".to_string(), Array1::from_vec(vec![4.0, 5.0]));

        assert_eq!(params.total_parameters(), 5);

        // Test flattening
        let flat = params.flatten_parameters();
        assert_eq!(flat.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test updating from flat
        let new_flat = Array1::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0]);
        params.update_from_flat(&new_flat);

        assert_eq!(params.parameters[0].as_slice().unwrap(), &[6.0, 7.0, 8.0]);
        assert_eq!(params.parameters[1].as_slice().unwrap(), &[9.0, 10.0]);
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = NeuralOptimizer::sgd(0.1, 100);
        let mut params = NeuralParameters::<f64>::new();

        // Add parameter
        params.add_parameter("test".to_string(), Array1::from_vec(vec![1.0, 2.0]));
        // Set gradient
        params.gradients[0] = Array1::from_vec(vec![0.5, 1.0]);

        // Perform one step
        optimizer.step(&mut params).unwrap();

        // Check update: param = param - lr * grad
        let expected = [1.0 - 0.1 * 0.5, 2.0 - 0.1 * 1.0];
        assert_abs_diff_eq!(params.parameters[0][0], expected[0], epsilon = 1e-10);
        assert_abs_diff_eq!(params.parameters[0][1], expected[1], epsilon = 1e-10);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = NeuralOptimizer::adam(0.001, 100);
        let mut params = NeuralParameters::<f64>::new();

        // Add parameter
        params.add_parameter("test".to_string(), Array1::from_vec(vec![1.0, 2.0]));
        // Set gradient
        params.gradients[0] = Array1::from_vec(vec![0.1, 0.2]);

        let original_params = params.parameters[0].clone();

        // Perform one step
        optimizer.step(&mut params).unwrap();

        // Parameters should have changed
        assert_ne!(params.parameters[0][0], original_params[0]);
        assert_ne!(params.parameters[0][1], original_params[1]);

        // Parameters should have decreased (since gradients are positive)
        assert!(params.parameters[0][0] < original_params[0]);
        assert!(params.parameters[0][1] < original_params[1]);
    }

    #[test]
    fn test_neural_trainer() {
        let optimizer = NeuralOptimizer::sgd(0.1, 100);
        let mut trainer = NeuralTrainer::new(optimizer).with_early_stopping(5);

        let mut params = NeuralParameters::<f64>::new();
        params.add_parameter("test".to_string(), Array1::from_vec(vec![1.0]));
        params.gradients[0] = Array1::from_vec(vec![1.0]);

        // Simple quadratic loss function
        let mut loss_fn = |p: &NeuralParameters<f64>| p.parameters[0][0] * p.parameters[0][0];
        let mut grad_fn =
            |p: &NeuralParameters<f64>| vec![Array1::from_vec(vec![2.0 * p.parameters[0][0]])];

        // Train for one epoch
        let loss = trainer
            .train_epoch(&mut params, &mut loss_fn, &mut grad_fn)
            .unwrap();

        // Loss should be computed
        assert_eq!(trainer.loss_history().len(), 1);
        assert_eq!(trainer.loss_history()[0], loss);
    }

    #[test]
    fn test_optimizer_convenience_functions() {
        let sgd_opt = optimizers::sgd(0.01);
        assert_eq!(sgd_opt.method_name(), "SGD");

        let adam_opt = optimizers::adam(0.001);
        assert_eq!(adam_opt.method_name(), "Adam");

        let adamw_opt = optimizers::adamw(0.001);
        assert_eq!(adamw_opt.method_name(), "AdamW");
    }
}
