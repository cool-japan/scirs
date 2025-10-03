//! Momentum optimizer for neural networks

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Momentum optimizer
///
/// Implements SGD with momentum. This is essentially SGD but with momentum always enabled.
/// Formula:
/// v_t = momentum * v_{t-1} + learning_rate * (gradient + weight_decay * param)
/// param_t = param_{t-1} - v_t
/// # Examples
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_neural::optimizers::{MomentumOptimizer, Optimizer};
/// // Create a momentum optimizer with learning rate 0.01 and momentum 0.9
/// let mut momentum = MomentumOptimizer::new(0.01f64, 0.9);
/// // or with weight decay
/// let mut momentum_with_decay = MomentumOptimizer::new_with_weight_decay(0.01f64, 0.9, 0.0001);
#[derive(Debug, Clone)]
pub struct MomentumOptimizer<F: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: F,
    /// Momentum factor
    momentum: F,
    /// Weight decay factor (L2 regularization)
    weight_decay: F,
    /// Velocity (momentum state) for each parameter array
    velocity: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
}
impl<F: Float + ScalarOperand + Debug> MomentumOptimizer<F> {
    /// Creates a new momentum optimizer with the given learning rate and momentum
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `momentum` - The momentum factor (typically 0.9)
    pub fn new(_learningrate: F, momentum: F) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: F::zero(),
            velocity: Vec::new(),
        }
    }
    /// Creates a new momentum optimizer with weight decay
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn new_with_weight_decay(_learning_rate: F, momentum: F, weightdecay: F) -> Self {
            weight_decay,
    /// Sets the momentum factor
    /// * `momentum` - The momentum factor
    pub fn set_momentum(&mut self, momentum: F) -> &mut Self {
        self.momentum = momentum;
        self
    /// Gets the current momentum factor
    pub fn get_momentum(&self) -> F {
        self.momentum
    /// Sets the weight decay factor
    pub fn set_weight_decay(&mut self, weightdecay: F) -> &mut Self {
        self.weight_decay = weight_decay;
    /// Gets the current weight decay factor
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
impl<F: Float + ScalarOperand + Debug> Optimizer<F> for MomentumOptimizer<F> {
    fn update(
        &mut self,
        params: &mut [Array<F, scirs2_core::ndarray::IxDyn>],
        grads: &[Array<F, scirs2_core::ndarray::IxDyn>],
    ) -> Result<()> {
        if params.len() != grads.len() {
            return Err(NeuralError::TrainingError(format!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params.len(),
                grads.len()
            )));
        // Initialize velocity if it doesn't exist or has changed size
        if self.velocity.len() != params.len() {
            self.velocity = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
        // Update parameters for each param-grad pair
        for i in 0..params.len() {
            // Apply weight decay to gradients if needed
            let adjusted_grad = if self.weight_decay > F::zero() {
                &grads[i] + &(&params[i] * self.weight_decay)
            } else {
                grads[i].clone()
            };
            // Update velocity with momentum
            self.velocity[i] =
                &self.velocity[i] * self.momentum + &(&adjusted_grad * self.learning_rate);
            // Update parameters
            params[i] = &params[i] - &self.velocity[i];
        Ok(())
    fn get_learning_rate(&self) -> F {
        self.learning_rate
    fn set_learning_rate(&mut self, lr: F) {
        self.learning_rate = lr;
    fn reset(&mut self) {
        self.velocity.clear();
    fn name(&self) -> &'static str {
        "MomentumOptimizer"
