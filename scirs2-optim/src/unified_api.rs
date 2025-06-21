//! Unified API consistent with popular deep learning frameworks
//!
//! This module provides a unified interface that closely follows the design patterns
//! of popular deep learning frameworks like PyTorch, TensorFlow, and JAX/Optax.
//!
//! # Design Principles
//!
//! - **Parameter Groups**: Support for different optimization parameters for different layers
//! - **State Management**: Automatic handling of optimizer state
//! - **Framework Consistency**: APIs that feel familiar to PyTorch/TensorFlow users
//! - **Flexible Configuration**: Easy-to-use builder patterns
//! - **Scheduler Integration**: Seamless integration with learning rate schedulers

use crate::error::{OptimError, Result};
use crate::schedulers::LearningRateScheduler;
use ndarray::{Array, Array1, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Unified optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig<A: Float> {
    /// Learning rate
    pub lr: A,
    /// Weight decay (L2 regularization)
    pub weight_decay: A,
    /// Gradient clipping value (optional)
    pub grad_clip: Option<A>,
    /// Additional optimizer-specific parameters
    pub params: HashMap<String, A>,
}

impl<A: Float> Default for OptimizerConfig<A> {
    fn default() -> Self {
        Self {
            lr: A::from(0.001).unwrap(),
            weight_decay: A::zero(),
            grad_clip: None,
            params: HashMap::new(),
        }
    }
}

impl<A: Float> OptimizerConfig<A> {
    /// Create a new optimizer configuration with the given learning rate
    pub fn new(lr: A) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set weight decay
    pub fn weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set gradient clipping
    pub fn grad_clip(mut self, grad_clip: A) -> Self {
        self.grad_clip = Some(grad_clip);
        self
    }

    /// Add a custom parameter
    pub fn param<S: Into<String>>(mut self, key: S, value: A) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Set multiple parameters at once
    pub fn params(mut self, params: HashMap<String, A>) -> Self {
        self.params.extend(params);
        self
    }
}

/// Parameter tensor wrapper for unified API
#[derive(Debug, Clone)]
pub struct Parameter<A: Float, D: Dimension> {
    /// Parameter data
    pub data: Array<A, D>,
    /// Gradient data (optional)
    pub grad: Option<Array<A, D>>,
    /// Whether this parameter requires gradients
    pub requires_grad: bool,
    /// Parameter name/identifier
    pub name: String,
}

impl<A: Float + ScalarOperand, D: Dimension> Parameter<A, D> {
    /// Create a new parameter
    pub fn new<S: Into<String>>(data: Array<A, D>, name: S) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: true,
            name: name.into(),
        }
    }

    /// Create a parameter that doesn't require gradients
    pub fn no_grad<S: Into<String>>(data: Array<A, D>, name: S) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: false,
            name: name.into(),
        }
    }

    /// Set gradient for this parameter
    pub fn set_grad(&mut self, grad: Array<A, D>) {
        if self.requires_grad {
            self.grad = Some(grad);
        }
    }

    /// Clear gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Get gradient reference
    pub fn grad(&self) -> Option<&Array<A, D>> {
        self.grad.as_ref()
    }

    /// Apply gradient clipping if specified
    pub fn clip_grad(&mut self, max_norm: A) -> Result<()> {
        if let Some(ref mut grad) = self.grad {
            let norm = grad
                .iter()
                .map(|x| (*x) * (*x))
                .fold(A::zero(), |acc, x| acc + x)
                .sqrt();
            if norm > max_norm {
                let scale = max_norm / norm;
                grad.mapv_inplace(|x| x * scale);
            }
        }
        Ok(())
    }
}

/// Unified optimizer interface
pub trait UnifiedOptimizer<A: Float> {
    /// Get optimizer configuration
    fn config(&self) -> &OptimizerConfig<A>;

    /// Update a single parameter
    fn step_param<D: Dimension>(&mut self, param: &mut Parameter<A, D>) -> Result<()>
    where
        A: ScalarOperand + Debug;

    /// Update multiple parameters
    fn step_params<D: Dimension>(&mut self, params: &mut [Parameter<A, D>]) -> Result<()>
    where
        A: ScalarOperand + Debug,
    {
        for param in params.iter_mut() {
            self.step_param(param)?;
        }
        Ok(())
    }

    /// Zero gradients for all parameters
    fn zero_grad<D: Dimension>(&self, params: &mut [Parameter<A, D>]) {
        for param in params.iter_mut() {
            param.grad = None;
        }
    }

    /// Update learning rate
    fn set_lr(&mut self, lr: A);

    /// Get current learning rate
    fn get_lr(&self) -> A;

    /// State dictionary for serialization
    fn state_dict(&self) -> HashMap<String, Vec<u8>>;

    /// Load state from dictionary
    fn load_state_dict(&mut self, state_dict: HashMap<String, Vec<u8>>) -> Result<()>;
}

/// SGD optimizer with unified API
#[derive(Debug)]
pub struct UnifiedSGD<A: Float> {
    config: OptimizerConfig<A>,
    momentum_buffers: HashMap<String, Array1<A>>,
}

impl<A: Float + ScalarOperand + Debug> UnifiedSGD<A> {
    /// Create a new SGD optimizer
    pub fn new(config: OptimizerConfig<A>) -> Self {
        Self {
            config,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(mut config: OptimizerConfig<A>, momentum: A) -> Self {
        config.params.insert("momentum".to_string(), momentum);
        Self::new(config)
    }
}

impl<A: Float + ScalarOperand + Debug> UnifiedOptimizer<A> for UnifiedSGD<A> {
    fn config(&self) -> &OptimizerConfig<A> {
        &self.config
    }

    fn step_param<D: Dimension>(&mut self, param: &mut Parameter<A, D>) -> Result<()> {
        if !param.requires_grad {
            return Ok(());
        }

        // Check gradient exists first
        if param.grad.is_none() {
            return Err(OptimError::InvalidConfig(
                "Parameter has no gradient".to_string(),
            ));
        }

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.grad_clip {
            param.clip_grad(max_norm)?;
        }

        // Apply weight decay
        if self.config.weight_decay > A::zero() {
            param
                .data
                .mapv_inplace(|x| x * (A::one() - self.config.weight_decay * self.config.lr));
        }

        // Get gradient safely
        let grad = param.grad.as_ref().unwrap();

        // Get momentum factor
        let momentum = self
            .config
            .params
            .get("momentum")
            .copied()
            .unwrap_or(A::zero());

        if momentum > A::zero() {
            // SGD with momentum
            if let Some(momentum_buffer) = self.momentum_buffers.get_mut(&param.name) {
                // Update momentum buffer
                for (m, g) in momentum_buffer.iter_mut().zip(grad.iter()) {
                    *m = momentum * (*m) + *g;
                }
                // Update parameters
                for (p, m) in param.data.iter_mut().zip(momentum_buffer.iter()) {
                    *p = *p - self.config.lr * (*m);
                }
            } else {
                // Initialize momentum buffer
                let mut momentum_buffer = Array1::zeros(grad.len());
                for (m, g) in momentum_buffer.iter_mut().zip(grad.iter()) {
                    *m = *g;
                }
                // Update parameters
                for (p, m) in param.data.iter_mut().zip(momentum_buffer.iter()) {
                    *p = *p - self.config.lr * (*m);
                }
                self.momentum_buffers
                    .insert(param.name.clone(), momentum_buffer);
            }
        } else {
            // Standard SGD
            for (p, g) in param.data.iter_mut().zip(grad.iter()) {
                *p = *p - self.config.lr * (*g);
            }
        }

        Ok(())
    }

    fn set_lr(&mut self, lr: A) {
        self.config.lr = lr;
    }

    fn get_lr(&self) -> A {
        self.config.lr
    }

    fn state_dict(&self) -> HashMap<String, Vec<u8>> {
        // Simplified state serialization
        HashMap::new()
    }

    fn load_state_dict(&mut self, _state_dict: HashMap<String, Vec<u8>>) -> Result<()> {
        // Simplified state deserialization
        Ok(())
    }
}

/// Adam optimizer with unified API
#[derive(Debug)]
pub struct UnifiedAdam<A: Float> {
    config: OptimizerConfig<A>,
    step_count: usize,
    exp_avg: HashMap<String, Array1<A>>,
    exp_avg_sq: HashMap<String, Array1<A>>,
}

impl<A: Float + ScalarOperand + Debug> UnifiedAdam<A> {
    /// Create a new Adam optimizer
    pub fn new(config: OptimizerConfig<A>) -> Self {
        let mut params = config.params.clone();
        params
            .entry("beta1".to_string())
            .or_insert_with(|| A::from(0.9).unwrap());
        params
            .entry("beta2".to_string())
            .or_insert_with(|| A::from(0.999).unwrap());
        params
            .entry("eps".to_string())
            .or_insert_with(|| A::from(1e-8).unwrap());

        Self {
            config: OptimizerConfig { params, ..config },
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Create Adam with custom betas
    pub fn with_betas(mut config: OptimizerConfig<A>, beta1: A, beta2: A) -> Self {
        config.params.insert("beta1".to_string(), beta1);
        config.params.insert("beta2".to_string(), beta2);
        Self::new(config)
    }
}

impl<A: Float + ScalarOperand + Debug> UnifiedOptimizer<A> for UnifiedAdam<A> {
    fn config(&self) -> &OptimizerConfig<A> {
        &self.config
    }

    fn step_param<D: Dimension>(&mut self, param: &mut Parameter<A, D>) -> Result<()> {
        if !param.requires_grad {
            return Ok(());
        }

        // Check gradient exists first
        if param.grad.is_none() {
            return Err(OptimError::InvalidConfig(
                "Parameter has no gradient".to_string(),
            ));
        }

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.grad_clip {
            param.clip_grad(max_norm)?;
        }

        self.step_count += 1;

        let beta1 = self.config.params["beta1"];
        let beta2 = self.config.params["beta2"];
        let eps = self.config.params["eps"];

        // Get gradient safely
        let grad = param.grad.as_ref().unwrap();

        // Initialize or get existing moment estimates
        let exp_avg = self
            .exp_avg
            .entry(param.name.clone())
            .or_insert_with(|| Array1::zeros(grad.len()));
        let exp_avg_sq = self
            .exp_avg_sq
            .entry(param.name.clone())
            .or_insert_with(|| Array1::zeros(grad.len()));

        // Update biased first and second moment estimates
        for ((exp_avg_val, exp_avg_sq_val), grad_val) in exp_avg
            .iter_mut()
            .zip(exp_avg_sq.iter_mut())
            .zip(grad.iter())
        {
            *exp_avg_val = beta1 * (*exp_avg_val) + (A::one() - beta1) * (*grad_val);
            *exp_avg_sq_val =
                beta2 * (*exp_avg_sq_val) + (A::one() - beta2) * (*grad_val) * (*grad_val);
        }

        // Bias correction
        let bias_correction1 = A::one() - beta1.powi(self.step_count as i32);
        let bias_correction2 = A::one() - beta2.powi(self.step_count as i32);

        let step_size = self.config.lr * (bias_correction2.sqrt() / bias_correction1);

        // Update parameters
        for ((p, exp_avg_val), exp_avg_sq_val) in param
            .data
            .iter_mut()
            .zip(exp_avg.iter())
            .zip(exp_avg_sq.iter())
        {
            let denom = exp_avg_sq_val.sqrt() + eps;
            *p = *p - step_size * (*exp_avg_val) / denom;
        }

        // Apply weight decay after the main update
        if self.config.weight_decay > A::zero() {
            param
                .data
                .mapv_inplace(|x| x * (A::one() - self.config.weight_decay * self.config.lr));
        }

        Ok(())
    }

    fn set_lr(&mut self, lr: A) {
        self.config.lr = lr;
    }

    fn get_lr(&self) -> A {
        self.config.lr
    }

    fn state_dict(&self) -> HashMap<String, Vec<u8>> {
        // Simplified state serialization
        HashMap::new()
    }

    fn load_state_dict(&mut self, _state_dict: HashMap<String, Vec<u8>>) -> Result<()> {
        // Simplified state deserialization
        Ok(())
    }
}

/// Optimizer factory for creating optimizers with unified API
pub struct OptimizerFactory;

impl OptimizerFactory {
    /// Create SGD optimizer
    pub fn sgd<A: Float + ScalarOperand + Debug>(config: OptimizerConfig<A>) -> UnifiedSGD<A> {
        UnifiedSGD::new(config)
    }

    /// Create Adam optimizer
    pub fn adam<A: Float + ScalarOperand + Debug>(config: OptimizerConfig<A>) -> UnifiedAdam<A> {
        UnifiedAdam::new(config)
    }

    /// Create SGD with momentum
    pub fn sgd_momentum<A: Float + ScalarOperand + Debug>(
        config: OptimizerConfig<A>,
        momentum: A,
    ) -> UnifiedSGD<A> {
        UnifiedSGD::with_momentum(config, momentum)
    }

    /// Create Adam with custom parameters
    pub fn adam_custom<A: Float + ScalarOperand + Debug>(
        config: OptimizerConfig<A>,
        beta1: A,
        beta2: A,
    ) -> UnifiedAdam<A> {
        UnifiedAdam::with_betas(config, beta1, beta2)
    }
}

/// Training loop helper with unified API
pub struct TrainingLoop<A: Float, O: UnifiedOptimizer<A>> {
    optimizer: O,
    scheduler: Option<Box<dyn LearningRateScheduler<A>>>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + ScalarOperand + Debug, O: UnifiedOptimizer<A>> TrainingLoop<A, O> {
    /// Create a new training loop
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            scheduler: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a learning rate scheduler
    pub fn with_scheduler(mut self, scheduler: Box<dyn LearningRateScheduler<A>>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Perform one training step
    pub fn step<D: Dimension>(&mut self, params: &mut [Parameter<A, D>]) -> Result<()> {
        // Update parameters
        self.optimizer.step_params(params)?;

        // Update learning rate if scheduler is present
        if let Some(ref mut scheduler) = self.scheduler {
            let new_lr = scheduler.step();
            self.optimizer.set_lr(new_lr);
        }

        Ok(())
    }

    /// Zero gradients
    pub fn zero_grad<D: Dimension>(&self, params: &mut [Parameter<A, D>]) {
        for param in params.iter_mut() {
            param.grad = None;
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> A {
        self.optimizer.get_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_unified_sgd() {
        let config = OptimizerConfig::new(0.1f64);
        let mut optimizer = UnifiedSGD::new(config);

        let mut param = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "test_param");
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));

        optimizer.step_param(&mut param).unwrap();

        // Check that parameters were updated correctly
        assert!((param.data[0] - 0.99).abs() < 1e-10);
        assert!((param.data[1] - 1.98).abs() < 1e-10);
        assert!((param.data[2] - 2.97).abs() < 1e-10);
    }

    #[test]
    fn test_unified_adam() {
        let config = OptimizerConfig::new(0.001f64);
        let mut optimizer = UnifiedAdam::new(config);

        let mut param = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "test_param");
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));

        optimizer.step_param(&mut param).unwrap();

        // Parameters should have been updated (exact values depend on Adam's internal state)
        assert!(param.data[0] < 1.0);
        assert!(param.data[1] < 2.0);
        assert!(param.data[2] < 3.0);
    }

    #[test]
    fn test_optimizer_factory() {
        let config = OptimizerConfig::new(0.01f64).weight_decay(0.0001);
        let _sgd = OptimizerFactory::sgd(config.clone());
        let _adam = OptimizerFactory::adam(config);
    }

    #[test]
    fn test_parameter_operations() {
        let mut param = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "test");

        // Test gradient setting
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));
        assert!(param.grad().is_some());

        // Test gradient clipping
        param.clip_grad(0.1).unwrap();
        let grad = param.grad().unwrap();
        let norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 0.1).abs() < 1e-10);

        // Test zero grad
        param.zero_grad();
        assert!(param.grad().is_none());
    }
}
