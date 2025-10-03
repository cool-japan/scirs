//! AdamW optimizer implementation for neural networks

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// AdamW optimizer for neural networks
///
/// Implements the AdamW optimizer from the paper:
/// "Decoupled Weight Decay Regularization" by Loshchilov and Hutter (2017).
/// AdamW is a variant of Adam that correctly implements weight decay regularization,
/// which helps improve generalization and training stability.
/// The key difference from Adam is that weight decay is applied directly to the weights
/// rather than to the gradients.
/// # Examples
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_neural::optimizers::{AdamW, Optimizer};
/// // Create a simple AdamW optimizer with default parameters
/// let mut adamw = AdamW::<f64>::default_with_lr(0.001).unwrap();
/// // or with custom configuration
/// let mut adamw_custom = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
#[derive(Debug, Clone)]
pub struct AdamW<F: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: F,
    /// Exponential decay rate for the first moment estimates
    beta1: F,
    /// Exponential decay rate for the second moment estimates
    beta2: F,
    /// Small constant for numerical stability
    epsilon: F,
    /// Weight decay factor (decoupled from gradient updates)
    weight_decay: F,
    /// First moment estimate for each parameter array
    m: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Second moment estimate for each parameter array
    v: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Current timestep
    t: usize,
}
impl<F: Float + ScalarOperand + Debug> AdamW<F> {
    /// Creates a new AdamW optimizer with the given hyperparameters
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor (default: 0.01)
    pub fn new(_learning_rate: F, beta1: F, beta2: F, epsilon: F, weightdecay: F) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    /// Creates a new AdamW optimizer with default hyperparameters
    pub fn default_with_lr(_learningrate: F) -> Result<Self> {
        let beta1 = F::from(0.9).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.9 to the appropriate floating point type".to_string(),
            )
        })?;
        let beta2 = F::from(0.999).ok_or_else(|| {
                "Failed to convert 0.999 to the appropriate floating point type".to_string(),
        let epsilon = F::from(1e-8).ok_or_else(|| {
                "Failed to convert 1e-8 to the appropriate floating point type".to_string(),
        let weight_decay = F::from(0.01).ok_or_else(|| {
                "Failed to convert 0.01 to the appropriate floating point type".to_string(),
        Ok(Self::new(
        ))
    /// Gets the beta1 parameter
    pub fn get_beta1(&self) -> F {
        self.beta1
    /// Sets the beta1 parameter
    pub fn set_beta1(&mut self, beta1: F) -> &mut Self {
        self.beta1 = beta1;
        self
    /// Gets the beta2 parameter
    pub fn get_beta2(&self) -> F {
        self.beta2
    /// Sets the beta2 parameter
    pub fn set_beta2(&mut self, beta2: F) -> &mut Self {
        self.beta2 = beta2;
    /// Gets the epsilon parameter
    pub fn get_epsilon(&self) -> F {
        self.epsilon
    /// Sets the epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: F) -> &mut Self {
        self.epsilon = epsilon;
    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weightdecay: F) -> &mut Self {
        self.weight_decay = weight_decay;
    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
impl<F: Float + ScalarOperand + Debug> Optimizer<F> for AdamW<F> {
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
        // Increment timestep
        self.t += 1;
        // Initialize moment estimates if needed
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
            self.v = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
        let one_minus_beta1 = F::one() - self.beta1;
        let one_minus_beta2 = F::one() - self.beta2;
        let beta1_pow_t = self.beta1.powi(self.t as i32);
        let beta2_pow_t = self.beta2.powi(self.t as i32);
        let bias_correction1 = F::one() - beta1_pow_t;
        let bias_correction2 = F::one() - beta2_pow_t;
        // Update parameters for each param-grad pair
        for i in 0..params.len() {
            // Update biased first moment estimate
            self.m[i] = &self.m[i] * self.beta1 + &(&grads[i] * one_minus_beta1);
            // Update biased second raw moment estimate
            self.v[i] = &self.v[i] * self.beta2 + &(grads[i].mapv(|x| x * x) * one_minus_beta2);
            // Compute bias-corrected first moment estimate
            let m_hat = &self.m[i] / bias_correction1;
            // Compute bias-corrected second raw moment estimate
            let v_hat = &self.v[i] / bias_correction2;
            // Perform parameter update with decoupled weight decay
            let denom = v_hat.mapv(|x| x.sqrt()) + self.epsilon;
            // The key difference from Adam:
            // 1. Apply weight decay directly to the parameter (not the gradient)
            // 2. Update the parameter using both the adaptive learning rate and weight decay
            params[i] = &params[i] * (F::one() - self.learning_rate * self.weight_decay)
                - &(m_hat / denom * self.learning_rate);
        Ok(())
    fn get_learning_rate(&self) -> F {
        self.learning_rate
    fn set_learning_rate(&mut self, lr: F) {
        self.learning_rate = lr;
// Enable direct usage of scirs2-optim's AdamW when the optim feature is enabled
#[cfg(feature = "optim")]
pub use scirs2_optim::AdamW as OptimAdamW;
