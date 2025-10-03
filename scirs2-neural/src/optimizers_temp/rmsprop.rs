//! RMSprop optimizer wrapper

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::Array;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
// Import from scirs2-optim
use scirs2_optim::optimizers as optim_optimizers;
/// RMSprop optimizer.
///
/// RMSprop is an optimization algorithm proposed by Geoffrey Hinton
/// that adapts the learning rate for each parameter based on the
/// recent history of gradients.
/// This is a wrapper around the RMSprop implementation from scirs2-optim.
/// # Examples
/// ```
/// use scirs2_neural::optimizers::RMSprop;
/// use scirs2_neural::optimizers::Optimizer;
/// use scirs2_core::ndarray::Array;
/// // Create RMSprop optimizer with learning rate 0.001
/// let mut rmsprop = RMSprop::new(0.001);
/// // Initialize parameters and gradients
/// let mut params = vec![Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()];
/// let grads = vec![Array::from_vec(vec![0.1, 0.2, 0.3]).into_dyn()];
/// // Update parameters
/// rmsprop.update(&mut params, &grads).unwrap();
pub struct RMSprop<F: Float + Debug> {
    /// Inner RMSprop optimizer from scirs2-optim
    inner: optim, optimizers: RMSprop<F>,
    /// Weight decay (L2 regularization)
    weight_decay: F,
}
impl<F: Float + Debug> RMSprop<F> {
    /// Create a new RMSprop optimizer with default parameters
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate
    pub fn new(_learningrate: F) -> Self {
        // Default values: rho=0.9, epsilon=1e-8, weight_decay=0.0
        let rho = F::from(0.9).unwrap_or(F::one());
        let epsilon = F::from(1e-8).unwrap_or(F::zero());
        let weight_decay = F::zero();
        
        Self {
            inner: optim, optimizers: RMSprop::new_with_config(
                learning_rate,
                rho,
                epsilon,
                weight_decay
            ),
            weight_decay,
        }
    }
    
    /// Create a new RMSprop optimizer with custom parameters
    /// * `rho` - Decay rate for moving average of squared gradients
    /// * `epsilon` - Small constant for numerical stability
    /// * `weight_decay` - Weight decay (L2 regularization)
    pub fn new_with_config(_learning_rate: F, rho: F, epsilon: F, weightdecay: F) -> Self {
    /// Get rho parameter (decay rate)
    pub fn get_rho(&self) -> F {
        self.inner.get_rho()
    /// Set rho parameter (decay rate)
    pub fn set_rho(&mut self, rho: F) -> &mut Self {
        self.inner.set_rho(rho);
        self
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
impl<F: Float + Debug> Optimizer<F> for RMSprop<F> {
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
