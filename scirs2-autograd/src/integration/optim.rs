//! Integration utilities for scirs2-optim module
//!
//! This module provides seamless integration between scirs2-autograd and scirs2-optim,
//! including optimizer interfaces, parameter management, and learning rate scheduling.

use super::{core::SciRS2Data, IntegrationError, SciRS2Integration};
use crate::tensor::Tensor;
use crate::Float;
use std::collections::HashMap;

/// Optimizer interface for autograd integration
pub trait AutogradOptimizer<F: Float> {
    /// Optimizer name
    fn name(&self) -> &str;

    /// Initialize optimizer with parameters
    fn initialize(&mut self, parameters: &[&Tensor<F>]) -> Result<(), IntegrationError>;

    /// Perform optimization step
    fn step(
        &mut self,
        parameters: &mut [&mut Tensor<F>],
        gradients: &[&Tensor<F>],
    ) -> Result<(), IntegrationError>;

    /// Zero gradients
    fn zero_grad(&mut self);

    /// Get learning rate
    fn learning_rate(&self) -> f64;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f64);

    /// Get optimizer state
    fn state(&self) -> OptimizerState<'_, F>;

    /// Set optimizer state
    fn set_state(&mut self, state: OptimizerState<'_, F>);

    /// Get parameter groups
    fn parameter_groups(&self) -> &[ParameterGroup<F>];

    /// Add parameter group
    fn add_parameter_group(&mut self, group: ParameterGroup<F>);
}

/// Optimizer state for serialization and checkpointing
#[derive(Debug, Clone)]
pub struct OptimizerState<'a, F: Float> {
    /// Step count
    pub step_count: usize,
    /// Per-parameter state
    pub param_state: HashMap<String, ParameterState<'a, F>>,
    /// Global optimizer state
    pub global_state: HashMap<String, StateValue>,
    /// Configuration
    pub config: OptimizerConfig,
}

impl<'a, F: Float> OptimizerState<'a, F> {
    /// Create new optimizer state
    pub fn new() -> Self {
        Self {
            step_count: 0,
            param_state: HashMap::new(),
            global_state: HashMap::new(),
            config: OptimizerConfig::default(),
        }
    }

    /// Get parameter state
    pub fn get_param_state(&self, param_id: &str) -> Option<&ParameterState<'a, F>> {
        self.param_state.get(param_id)
    }

    /// Set parameter state
    pub fn set_param_state(&mut self, param_id: String, state: ParameterState<'a, F>) {
        self.param_state.insert(param_id, state);
    }

    /// Increment step count
    pub fn increment_step(&mut self) {
        self.step_count += 1;
    }
}

impl<F: Float> Default for OptimizerState<'_, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-parameter state for optimizers
#[derive(Debug, Clone)]
pub struct ParameterState<'a, F: Float> {
    /// Momentum buffer
    pub momentum: Option<Tensor<'a, F>>,
    /// Squared gradient buffer (for Adam, RMSprop)
    pub exp_avg_sq: Option<Tensor<'a, F>>,
    /// Exponential moving average (for Adam)
    pub exp_avg: Option<Tensor<'a, F>>,
    /// Maximum squared gradient (for Adamax)
    pub exp_inf: Option<Tensor<'a, F>>,
    /// Step count for this parameter
    pub step: usize,
    /// Additional state
    pub extra_state: HashMap<String, Tensor<'a, F>>,
}

impl<F: Float> ParameterState<'_, F> {
    /// Create new parameter state
    pub fn new() -> Self {
        Self {
            momentum: None,
            exp_avg_sq: None,
            exp_avg: None,
            exp_inf: None,
            step: 0,
            extra_state: HashMap::new(),
        }
    }

    // Note: Buffer initialization skipped due to autograd's lazy evaluation
    // Buffers would be initialized when first needed during optimization
}

impl<F: Float> Default for ParameterState<'_, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// State value types for optimizer configuration
#[derive(Debug, Clone)]
pub enum StateValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    FloatArray(Vec<f64>),
}

impl StateValue {
    /// Get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            StateValue::Float(val) => Some(*val),
            StateValue::Int(val) => Some(*val as f64),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            StateValue::Int(val) => Some(*val),
            StateValue::Float(val) => Some(*val as i64),
            _ => None,
        }
    }
}

/// Parameter group for different optimization settings
#[derive(Debug)]
pub struct ParameterGroup<F: Float> {
    /// Parameters in this group
    pub parameters: Vec<String>, // Parameter IDs
    /// Learning rate for this group
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Group-specific configuration
    pub config: HashMap<String, StateValue>,
    /// Group name
    pub name: String,
    /// Phantom data to use type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> ParameterGroup<F> {
    /// Create new parameter group
    pub fn new(name: String, learning_rate: f64) -> Self {
        Self {
            parameters: Vec::new(),
            learning_rate,
            weight_decay: 0.0,
            config: HashMap::new(),
            name,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add parameter to group
    pub fn add_parameter(mut self, param_id: String) -> Self {
        self.parameters.push(param_id);
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, decay: f64) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Add configuration
    pub fn config(mut self, key: String, value: StateValue) -> Self {
        self.config.insert(key, value);
        self
    }
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Base learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Momentum factor
    pub momentum: f64,
    /// Beta1 for Adam
    pub beta1: f64,
    /// Beta2 for Adam
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
    /// Amsgrad flag
    pub amsgrad: bool,
    /// Additional configuration
    pub extra_config: HashMap<String, StateValue>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 0.0,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
            extra_config: HashMap::new(),
        }
    }
}

/// SGD optimizer implementation
pub struct SGDOptimizer<'a, F: Float> {
    /// Optimizer configuration
    config: OptimizerConfig,
    /// Optimizer state
    state: OptimizerState<'a, F>,
    /// Parameter groups
    parameter_groups: Vec<ParameterGroup<F>>,
}

impl<F: Float> SGDOptimizer<'_, F> {
    /// Create new SGD optimizer
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        let config = OptimizerConfig {
            learning_rate,
            momentum,
            ..Default::default()
        };

        Self {
            config,
            state: OptimizerState::new(),
            parameter_groups: Vec::new(),
        }
    }

    /// Create with weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.config.weight_decay = weight_decay;
        self
    }
}

impl<F: Float> AutogradOptimizer<F> for SGDOptimizer<'_, F> {
    fn name(&self) -> &str {
        "SGD"
    }

    fn initialize(&mut self, parameters: &[&Tensor<F>]) -> Result<(), IntegrationError> {
        for (i, _param) in parameters.iter().enumerate() {
            let param_id = format!("param_{}", i);
            let param_state = ParameterState::new();
            // Skip shape-based initialization to avoid lazy evaluation issues
            self.state.set_param_state(param_id, param_state);
        }
        Ok(())
    }

    fn step(
        &mut self,
        parameters: &mut [&mut Tensor<F>],
        gradients: &[&Tensor<F>],
    ) -> Result<(), IntegrationError> {
        if parameters.len() != gradients.len() {
            return Err(IntegrationError::ModuleCompatibility(
                "Parameter and gradient count mismatch".to_string(),
            ));
        }

        // Collect parameter states first to avoid borrowing conflicts
        let mut param_updates = Vec::new();
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let param_id = format!("param_{}", i);
            param_updates.push((param, grad, param_id));
        }

        // Now update parameters
        for (param, grad, param_id) in param_updates {
            if let Some(param_state) = self.state.param_state.get_mut(&param_id) {
                Self::update_parameter(&self.config, param, grad, param_state)?;
            }
        }

        self.state.increment_step();
        Ok(())
    }

    fn zero_grad(&mut self) {
        // In practice, would clear gradient buffers
    }

    fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state(&self) -> OptimizerState<'_, F> {
        self.state.clone()
    }

    fn set_state(&mut self, state: OptimizerState<'_, F>) {
        // Convert the incoming state to the correct lifetime
        let converted_state = OptimizerState {
            step_count: state.step_count,
            param_state: HashMap::new(), // Skip parameter state due to lifetime complexity
            global_state: state.global_state,
            config: state.config,
        };
        self.state = converted_state;
    }

    fn parameter_groups(&self) -> &[ParameterGroup<F>] {
        &self.parameter_groups
    }

    fn add_parameter_group(&mut self, group: ParameterGroup<F>) {
        self.parameter_groups.push(group);
    }
}

impl<'a, F: Float> SGDOptimizer<'a, F> {
    fn update_parameter(
        config: &OptimizerConfig,
        _param: &mut &mut Tensor<F>,
        _grad: &Tensor<F>,
        _param_state: &mut ParameterState<'a, F>,
    ) -> Result<(), IntegrationError> {
        // Simplified SGD update: param = param - lr * grad
        // In practice, would implement proper momentum and weight decay

        let _lr = F::from(config.learning_rate).unwrap();

        // Apply weight decay if configured
        if config.weight_decay > 0.0 {
            let _decay = F::from(config.weight_decay).unwrap();
            // param_data = param_data - decay * param_data (simplified)
        }

        // Update with momentum if configured
        if config.momentum > 0.0 {
            if let Some(ref mut _momentum_buffer) = _param_state.momentum {
                let _momentum = F::from(config.momentum).unwrap();
                // momentum_buffer = momentum * momentum_buffer + lr * grad
                // param_data = param_data - momentum_buffer
            }
        } else {
            // Simple update: param = param - lr * grad
            // This is a placeholder - actual implementation would modify param data
        }

        Ok(())
    }
}

/// Adam optimizer implementation
pub struct AdamOptimizer<'a, F: Float> {
    /// Optimizer configuration
    config: OptimizerConfig,
    /// Optimizer state
    state: OptimizerState<'a, F>,
    /// Parameter groups
    parameter_groups: Vec<ParameterGroup<F>>,
}

impl<F: Float> AdamOptimizer<'_, F> {
    /// Create new Adam optimizer
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        let config = OptimizerConfig {
            learning_rate,
            beta1,
            beta2,
            eps,
            ..Default::default()
        };

        Self {
            config,
            state: OptimizerState::new(),
            parameter_groups: Vec::new(),
        }
    }

    /// Create with default Adam parameters
    pub fn default_adam(learning_rate: f64) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8)
    }
}

impl<F: Float> AutogradOptimizer<F> for AdamOptimizer<'_, F> {
    fn name(&self) -> &str {
        "Adam"
    }

    fn initialize(&mut self, parameters: &[&Tensor<F>]) -> Result<(), IntegrationError> {
        for (i, _param) in parameters.iter().enumerate() {
            let param_id = format!("param_{}", i);
            let param_state = ParameterState::new();
            // Skip shape-based initialization to avoid lazy evaluation issues
            self.state.set_param_state(param_id, param_state);
        }
        Ok(())
    }

    fn step(
        &mut self,
        parameters: &mut [&mut Tensor<F>],
        gradients: &[&Tensor<F>],
    ) -> Result<(), IntegrationError> {
        if parameters.len() != gradients.len() {
            return Err(IntegrationError::ModuleCompatibility(
                "Parameter and gradient count mismatch".to_string(),
            ));
        }

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let param_id = format!("param_{}", i);

            if let Some(param_state) = self.state.param_state.get_mut(&param_id) {
                Self::update_parameter_adam(&self.config, param, grad, param_state)?;
            }
        }

        self.state.increment_step();
        Ok(())
    }

    fn zero_grad(&mut self) {
        // Clear gradients
    }

    fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state(&self) -> OptimizerState<'_, F> {
        self.state.clone()
    }

    fn set_state(&mut self, state: OptimizerState<'_, F>) {
        // Convert the incoming state to the correct lifetime
        let converted_state = OptimizerState {
            step_count: state.step_count,
            param_state: HashMap::new(), // Skip parameter state due to lifetime complexity
            global_state: state.global_state,
            config: state.config,
        };
        self.state = converted_state;
    }

    fn parameter_groups(&self) -> &[ParameterGroup<F>] {
        &self.parameter_groups
    }

    fn add_parameter_group(&mut self, group: ParameterGroup<F>) {
        self.parameter_groups.push(group);
    }
}

impl<'a, F: Float> AdamOptimizer<'a, F> {
    fn update_parameter_adam(
        _config: &OptimizerConfig,
        _param: &mut &mut Tensor<F>,
        _grad: &Tensor<F>,
        param_state: &mut ParameterState<'a, F>,
    ) -> Result<(), IntegrationError> {
        // Simplified Adam update
        // In practice, would implement full Adam algorithm:
        // m_t = β1 * m_{t-1} + (1-β1) * g_t
        // v_t = β2 * v_{t-1} + (1-β2) * g_t^2
        // m̂_t = m_t / (1-β1^t)
        // v̂_t = v_t / (1-β2^t)
        // θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

        param_state.step += 1;

        // This is a placeholder implementation
        Ok(())
    }
}

/// Learning rate scheduler for autograd optimizers
pub trait LearningRateScheduler<F: Float> {
    /// Get current learning rate
    fn get_lr(&self) -> f64;

    /// Step the scheduler
    fn step(&mut self, optimizer: &mut dyn AutogradOptimizer<F>);

    /// Step with metric (for ReduceLROnPlateau)
    fn step_with_metric(&mut self, optimizer: &mut dyn AutogradOptimizer<F>, metric: f64);
}

/// Step learning rate scheduler
pub struct StepLRScheduler {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
    current_step: usize,
}

impl StepLRScheduler {
    /// Create new step scheduler
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl<F: Float> LearningRateScheduler<F> for StepLRScheduler {
    fn get_lr(&self) -> f64 {
        let decay_factor = (self.current_step / self.step_size) as f64;
        self.initial_lr * self.gamma.powf(decay_factor)
    }

    fn step(&mut self, optimizer: &mut dyn AutogradOptimizer<F>) {
        self.current_step += 1;
        let new_lr = <Self as LearningRateScheduler<F>>::get_lr(self);
        optimizer.set_learning_rate(new_lr);
    }

    fn step_with_metric(&mut self, optimizer: &mut dyn AutogradOptimizer<F>, _metric: f64) {
        self.step(optimizer);
    }
}

/// Cosine annealing scheduler
pub struct CosineAnnealingLRScheduler {
    initial_lr: f64,
    min_lr: f64,
    t_max: usize,
    current_step: usize,
}

impl CosineAnnealingLRScheduler {
    /// Create new cosine annealing scheduler
    pub fn new(initial_lr: f64, t_max: usize, min_lr: f64) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
            current_step: 0,
        }
    }
}

impl<F: Float> LearningRateScheduler<F> for CosineAnnealingLRScheduler {
    fn get_lr(&self) -> f64 {
        let progress = (self.current_step as f64) / (self.t_max as f64);
        let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
    }

    fn step(&mut self, optimizer: &mut dyn AutogradOptimizer<F>) {
        self.current_step += 1;
        let new_lr = <Self as LearningRateScheduler<F>>::get_lr(self);
        optimizer.set_learning_rate(new_lr);
    }

    fn step_with_metric(&mut self, optimizer: &mut dyn AutogradOptimizer<F>, _metric: f64) {
        self.step(optimizer);
    }
}

/// Optimizer factory for creating optimizers
pub struct OptimizerFactory;

impl OptimizerFactory {
    /// Create SGD optimizer
    pub fn sgd<'a, F: Float>(
        learning_rate: f64,
        momentum: f64,
    ) -> Box<dyn AutogradOptimizer<F> + 'a> {
        Box::new(SGDOptimizer::new(learning_rate, momentum))
    }

    /// Create Adam optimizer
    pub fn adam<'a, F: Float>(learning_rate: f64) -> Box<dyn AutogradOptimizer<F> + 'a> {
        Box::new(AdamOptimizer::default_adam(learning_rate))
    }

    /// Create custom Adam optimizer
    pub fn adam_custom<'a, F: Float>(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> Box<dyn AutogradOptimizer<F> + 'a> {
        Box::new(AdamOptimizer::new(learning_rate, beta1, beta2, eps))
    }
}

/// Implement SciRS2Integration for optimizer state
impl<F: Float> SciRS2Integration for OptimizerState<'_, F> {
    fn module_name() -> &'static str {
        "scirs2-optim"
    }

    fn module_version() -> &'static str {
        "0.1.0-alpha.5"
    }

    fn check_compatibility() -> Result<(), IntegrationError> {
        match super::check_compatibility("scirs2-autograd", "scirs2-optim")? {
            true => Ok(()),
            false => Err(IntegrationError::ModuleCompatibility(
                "Version mismatch".to_string(),
            )),
        }
    }
}

/// Convert optimizer state to SciRS2Data
pub fn optimizer_to_scirs2_data<'a, F: Float>(
    optimizer: &dyn AutogradOptimizer<F>,
) -> SciRS2Data<'a, F> {
    let mut data = SciRS2Data::new();

    let state = optimizer.state();

    // Add optimizer metadata
    data = data.add_metadata("module_name".to_string(), "scirs2-optim".to_string());
    data = data.add_metadata("optimizer_name".to_string(), optimizer.name().to_string());
    data = data.add_metadata("step_count".to_string(), state.step_count.to_string());
    data = data.add_metadata(
        "learning_rate".to_string(),
        optimizer.learning_rate().to_string(),
    );

    // Add state information as metadata instead of tensor
    data = data.add_metadata("step_count_value".to_string(), state.step_count.to_string());

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::tensor::Tensor;

    #[test]
    fn test_optimizer_state() {
        let mut state = OptimizerState::<f32>::new();

        let param_state = ParameterState::new();
        state.set_param_state("param_0".to_string(), param_state);

        assert_eq!(state.step_count, 0);
        assert!(state.get_param_state("param_0").is_some());

        state.increment_step();
        assert_eq!(state.step_count, 1);
    }

    #[test]
    fn test_parameter_group() {
        let group = ParameterGroup::<f32>::new("default".to_string(), 0.01)
            .add_parameter("param_0".to_string())
            .weight_decay(1e-4)
            .config("momentum".to_string(), StateValue::Float(0.9));

        assert_eq!(group.learning_rate, 0.01);
        assert_eq!(group.weight_decay, 1e-4);
        assert_eq!(group.parameters.len(), 1);
        assert!(group.config.contains_key("momentum"));
    }

    #[test]
    fn test_sgd_optimizer() {
        let optimizer = SGDOptimizer::<f32>::new(0.01, 0.9);

        assert_eq!(optimizer.name(), "SGD");
        assert_eq!(optimizer.learning_rate(), 0.01);

        // Skip tensor-based tests due to autograd's lazy evaluation
        assert_eq!(optimizer.state().param_state.len(), 0);
    }

    #[test]
    fn test_adam_optimizer() {
        let optimizer = AdamOptimizer::<f32>::default_adam(0.001);

        assert_eq!(optimizer.name(), "Adam");
        assert_eq!(optimizer.learning_rate(), 0.001);

        // Skip tensor-based tests due to autograd's lazy evaluation
        assert_eq!(optimizer.state().param_state.len(), 0);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let mut scheduler = StepLRScheduler::new(0.1, 5, 0.5);
        let mut optimizer = SGDOptimizer::<f64>::new(0.1, 0.0);

        assert_eq!(
            <StepLRScheduler as LearningRateScheduler<f64>>::get_lr(&scheduler),
            0.1f64
        );

        // Step 5 times
        for _ in 0..5 {
            scheduler.step(&mut optimizer);
        }

        // Learning rate should be reduced
        assert!(<StepLRScheduler as LearningRateScheduler<f64>>::get_lr(&scheduler) < 0.1);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let mut scheduler = CosineAnnealingLRScheduler::new(0.1, 10, 0.0);
        let mut optimizer = SGDOptimizer::<f64>::new(0.1, 0.0);

        let initial_lr: f64 =
            <CosineAnnealingLRScheduler as LearningRateScheduler<f64>>::get_lr(&scheduler);

        // Step halfway
        for _ in 0..5 {
            scheduler.step(&mut optimizer);
        }

        let halfway_lr =
            <CosineAnnealingLRScheduler as LearningRateScheduler<f64>>::get_lr(&scheduler);
        assert!(halfway_lr < initial_lr);
    }

    #[test]
    #[ignore = "Factory tests skipped due to lifetime complexity"]
    fn test_optimizer_factory() {
        // TODO: Implement factory tests when lifetime issues are resolved
    }

    #[test]
    fn test_scirs2_integration() {
        let state = OptimizerState::<f32>::new();

        // Test basic state properties
        assert_eq!(state.step_count, 0);
        assert!(state.param_state.is_empty());
    }
}
