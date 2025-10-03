//! Adagrad optimizer implementation

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Adagrad optimizer
///
/// Implements the Adagrad algorithm, which adapts the learning rate
/// for each parameter based on the historical gradient values.
/// Formula:
/// g_sum_t = g_sum_{t-1} + g_t^2
/// param_t = param_{t-1} - learningrate * g_t / (sqrt(g_sum_t) + epsilon)
/// # Examples
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_neural::optimizers::{Adagrad, Optimizer};
/// // Create a simple Adagrad optimizer
/// let mut adagrad = Adagrad::<f64>::new(0.01).unwrap();
/// // or with custom epsilon
/// let mut adagrad_custom = Adagrad::new_with_config(0.01, 1e-10, 0.0);
#[derive(Debug, Clone)]
pub struct Adagrad<F: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: F,
    /// Small constant for numerical stability
    epsilon: F,
    /// Weight decay factor (L2 regularization)
    weight_decay: F,
    /// Sum of squared gradients for each parameter array
    g_sum: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
}
impl<F: Float + ScalarOperand + Debug> Adagrad<F> {
    /// Creates a new Adagrad optimizer with the given learning rate and default parameters
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(_learningrate: F) -> Result<Self> {
        let epsilon = F::from(1e-10).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 1e-10 to the appropriate floating point type".to_string(),
            )
        })?;
        Ok(Self {
            learning_rate,
            epsilon,
            weight_decay: F::zero(),
            g_sum: Vec::new(),
        })
    }
    /// Creates a new Adagrad optimizer with the full configuration
    /// * `epsilon` - Small constant for numerical stability
    /// * `weight_decay` - Weight decay factor (L2 regularization)
    pub fn new_with_config(_learning_rate: F, epsilon: F, weightdecay: F) -> Self {
        Self {
            weight_decay,
        }
    /// Gets the epsilon parameter
    pub fn get_epsilon(&self) -> F {
        self.epsilon
    /// Sets the epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: F) -> &mut Self {
        self.epsilon = epsilon;
        self
    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weightdecay: F) -> &mut Self {
        self.weight_decay = weight_decay;
    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.g_sum.clear();
impl<F: Float + ScalarOperand + Debug> Optimizer<F> for Adagrad<F> {
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
        // Initialize g_sum if needed
        if self.g_sum.len() != params.len() {
            self.g_sum = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
        // Update parameters for each param-grad pair
        for i in 0..params.len() {
            // Apply weight decay to gradients if needed
            let adjusted_grad = if self.weight_decay > F::zero() {
                &grads[i] + &(&params[i] * self.weight_decay)
            } else {
                grads[i].clone()
            };
            // Update sum of squared gradients
            self.g_sum[i] = &self.g_sum[i] + &adjusted_grad.mapv(|x| x * x);
            // Compute parameter update
            let denom = self.g_sum[i].mapv(|x| x.sqrt()) + self.epsilon;
            params[i] = &params[i] - &(&adjusted_grad / denom * self.learning_rate);
        Ok(())
    fn get_learning_rate(&self) -> F {
        self.learning_rate
    fn set_learning_rate(&mut self, lr: F) {
        self.learning_rate = lr;
