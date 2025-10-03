//! Stochastic Gradient Descent (SGD) optimizer wrapper

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::Array;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
// Import from scirs2-optim
use scirs2_optim::optimizers as optim_optimizers;
/// Stochastic Gradient Descent (SGD) optimizer.
///
/// SGD is the simplest optimization algorithm, which updates parameters by
/// subtracting the gradient multiplied by the learning rate.
/// This implementation also supports momentum, which helps accelerate convergence
/// especially in the presence of high curvature or small but consistent gradients.
/// This is a wrapper around the SGD implementation from scirs2-optim.
/// # Examples
/// ```
/// use scirs2_neural::optimizers::SGD;
/// use scirs2_neural::optimizers::Optimizer;
/// use scirs2_core::ndarray::Array;
/// // Create SGD optimizer with learning rate 0.01 and momentum 0.9
/// let mut sgd = SGD::new(0.01, Some(0.9));
/// // Initialize parameters and gradients
/// let mut params = vec![Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()];
/// let grads = vec![Array::from_vec(vec![0.1, 0.2, 0.3]).into_dyn()];
/// // Update parameters
/// sgd.update(&mut params, &grads).unwrap();
pub struct SGD<F: Float + Debug> {
    /// Inner SGD optimizer from scirs2-optim
    inner: optim, optimizers: SGD<F>,
    /// Weight decay (L2 regularization)
    weight_decay: F,
}
impl<F: Float + Debug> SGD<F> {
    /// Create a new SGD optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate for the optimizer
    /// * `momentum` - Optional momentum factor (between 0 and 1)
    pub fn new(_learningrate: F, momentum: Option<F>) -> Self {
        let momentum_value = momentum.unwrap_or(F::zero());
        
        Self {
            inner: optim, optimizers: SGD::new_with_config(
                learning_rate,
                momentum_value,
                F::zero() // No weight decay by default
            ),
            weight_decay: F::zero(),
        }
    }
    
    /// Create a new SGD optimizer with weight decay
    /// * `momentum` - Momentum factor (between 0 and 1)
    /// * `weight_decay` - Weight decay factor (L2 regularization)
    pub fn new_with_decay(_learning_rate: F, momentum: F, weightdecay: F) -> Self {
                momentum,
                weight_decay
            weight_decay,
    /// Get the momentum value
    pub fn get_momentum(&self) -> F {
        self.inner.get_momentum()
    /// Set the momentum value
    pub fn set_momentum(&mut self, momentum: F) {
        self.inner.set_momentum(momentum);
    /// Get the weight decay value
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
    /// Set the weight decay value
    pub fn set_weight_decay(&mut self, weightdecay: F) {
        self.weight_decay = weight_decay;
        self.inner.set_weight_decay(weight_decay);
impl<F: Float + Debug> Optimizer<F> for SGD<F> {
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
