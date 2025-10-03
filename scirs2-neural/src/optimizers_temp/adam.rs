//! Adam optimizer wrapper

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::Array;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
// Import from scirs2-optim
use scirs2_optim::optimizers as optim_optimizers;
/// Adam optimizer (Adaptive Moment Estimation).
///
/// Adam is an optimization algorithm that adapts the learning rate for each parameter
/// by maintaining first and second moment estimates of gradients.
/// It combines the advantages of:
/// - AdaGrad: handles sparse gradients well
/// - RMSProp: handles non-stationary objectives
/// This is a wrapper around the Adam implementation from scirs2-optim.
/// # Examples
/// ```
/// use scirs2_neural::optimizers::Adam;
/// use scirs2_neural::optimizers::Optimizer;
/// use scirs2_core::ndarray::Array;
/// // Create Adam optimizer with default hyperparameters
/// let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
/// // Initialize parameters and gradients
/// let mut params = vec![Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()];
/// let grads = vec![Array::from_vec(vec![0.1, 0.2, 0.3]).into_dyn()];
/// // Update parameters
/// adam.update(&mut params, &grads).unwrap();
pub struct Adam<F: Float + Debug> {
    /// Inner Adam optimizer from scirs2-optim
    inner: optim, optimizers: Adam<F>,
    /// Weight decay (L2 regularization)
    weight_decay: F,
}
impl<F: Float + Debug> Adam<F> {
    /// Create a new Adam optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate
    /// * `beta1` - Exponential decay rate for first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    pub fn new(_learningrate: F, beta1: F, beta2: F, epsilon: F) -> Self {
        Self {
            inner: optim, optimizers: Adam::new_with_config(
                learning_rate,
                beta1,
                beta2,
                epsilon,
                F::zero() // No weight decay by default
            ),
            weight_decay: F::zero(),
        }
    }
    /// Create a new Adam optimizer with default hyperparameters
    pub fn default_with_lr(_learningrate: F) -> Result<Self> {
        let beta1 = F::from(0.9).ok_or_else(|| 
            NeuralError::InvalidArchitecture("Failed to convert beta1".to_string()))?;
        let beta2 = F::from(0.999).ok_or_else(|| 
            NeuralError::InvalidArchitecture("Failed to convert beta2".to_string()))?;
        let epsilon = F::from(1e-8).ok_or_else(|| 
            NeuralError::InvalidArchitecture("Failed to convert epsilon".to_string()))?;
        
        Ok(Self::new(_learning_rate, beta1, beta2, epsilon))
    
    /// Create a new Adam optimizer with weight decay
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    /// * `weight_decay` - Weight decay factor (L2 regularization)
    pub fn new_with_decay(
        learning_rate: F, 
        beta1: F, 
        beta2: F, 
        epsilon: F,
        weight_decay: F
    ) -> Self {
                weight_decay
            weight_decay,
    /// Get beta1 parameter
    pub fn get_beta1(&self) -> F {
        self.inner.get_beta1()
    /// Set beta1 parameter
    pub fn set_beta1(&mut self, beta1: F) -> &mut Self {
        self.inner.set_beta1(beta1);
        self
    /// Get beta2 parameter
    pub fn get_beta2(&self) -> F {
        self.inner.get_beta2()
    /// Set beta2 parameter
    pub fn set_beta2(&mut self, beta2: F) -> &mut Self {
        self.inner.set_beta2(beta2);
    /// Get epsilon parameter
    pub fn get_epsilon(&self) -> F {
        self.inner.get_epsilon()
    /// Set epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: F) -> &mut Self {
        self.inner.set_epsilon(epsilon);
    /// Get weight decay parameter
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
    /// Set weight decay parameter
    pub fn set_weight_decay(&mut self, weightdecay: F) -> &mut Self {
        self.weight_decay = weight_decay;
        self.inner.set_weight_decay(weight_decay);
    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.inner.reset();
impl<F: Float + Debug> Default for Adam<F> {
    fn default() -> Self {
        // Safe default with learning rate 0.001
        let _learning_rate = F::from(0.001).unwrap_or(F::one() / F::from(1000.0).unwrap_or(F::one()));
        let beta1 = F::from(0.9).unwrap_or(F::one());
        let beta2 = F::from(0.999).unwrap_or(F::one());
        let epsilon = F::from(1e-8).unwrap_or(F::zero());
        Self::new(_learning_rate, beta1, beta2, epsilon)
impl<F: Float + Debug> Optimizer<F> for Adam<F> {
    fn update(&mut self, params: &mut [Array<F, scirs2_core::ndarray::IxDyn>], 
              grads: &[Array<F, scirs2_core::ndarray::IxDyn>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(NeuralError::TrainingError(format!(
                "Parameter and gradient counts do not match: {} vs {}",
                params.len(), grads.len()
            )));
        // Update each parameter with its gradient using the inner optimizer
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            // Create a mutable copy of the parameter for the update
            let mut param_copy = param.clone();
            
            // Use the inner optimizer to update the parameter
            match self.inner.step(&param_copy, grad) {
                Ok(updated_param) => {
                    // Copy the updated parameter back
                    *param = updated_param;
                },
                Err(e) => {
                    return Err(NeuralError::TrainingError(format!(
                        "Failed to update parameter: {}", e
                    )));
                }
            }
        Ok(())
    fn get_learning_rate(&self) -> F {
        self.inner.get_learning_rate()
    fn set_learning_rate(&mut self, lr: F) {
        self.inner.set_learning_rate(lr);
