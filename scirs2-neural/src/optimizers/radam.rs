//! RAdam (Rectified Adam) optimizer implementation for neural networks

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// RAdam (Rectified Adam) optimizer for neural networks
///
/// Implements the RAdam optimization algorithm from the paper:
/// "On the Variance of the Adaptive Learning Rate and Beyond" by Liu et al. (2019).
/// RAdam is a variant of Adam that addresses the warm-up issue in Adam by:
/// 1. Correcting the adaptive learning rate bias
/// 2. Rectifying the variance of the adaptive learning rate
/// This generally leads to faster convergence and better generalization compared to
/// vanilla Adam.
/// # Examples
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_neural::optimizers::{RAdam, Optimizer};
/// // Create a simple RAdam optimizer with default parameters
/// let mut radam = RAdam::<f64>::default_with_lr(0.001).unwrap();
/// // or with custom configuration
/// let mut radam_custom = RAdam::new(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();
#[derive(Debug, Clone)]
pub struct RAdam<F: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: F,
    /// Exponential decay rate for the first moment estimates
    beta1: F,
    /// Exponential decay rate for the second moment estimates
    beta2: F,
    /// Small constant for numerical stability
    epsilon: F,
    /// Weight decay factor
    weight_decay: F,
    /// First moment estimate for each parameter array
    m: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Second moment estimate for each parameter array
    v: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Current timestep
    t: usize,
    /// RAdam-specific rho_inf value (calculated once)
    rho_inf: F,
}
impl<F: Float + ScalarOperand + Debug> RAdam<F> {
    /// Creates a new RAdam optimizer with the given hyperparameters
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor (default: 0.0)
    pub fn new(_learning_rate: F, beta1: F, beta2: F, epsilon: F, weightdecay: F) -> Result<Self> {
        // Calculate rho_inf (used in the rectification term)
        let two = F::from(2.0).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 2.0 to the appropriate floating point type".to_string(),
            )
        })?;
        let rho_inf = two / (F::one() - beta2) - F::one();
        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            rho_inf,
        })
    }
    /// Creates a new RAdam optimizer with default hyperparameters
    pub fn default_with_lr(_learningrate: F) -> Result<Self> {
        let beta1 = F::from(0.9).ok_or_else(|| {
                "Failed to convert 0.9 to the appropriate floating point type".to_string(),
        let beta2 = F::from(0.999).ok_or_else(|| {
                "Failed to convert 0.999 to the appropriate floating point type".to_string(),
        let epsilon = F::from(1e-8).ok_or_else(|| {
                "Failed to convert 1e-8 to the appropriate floating point type".to_string(),
        let weight_decay = F::from(0.0).ok_or_else(|| {
                "Failed to convert 0.0 to the appropriate floating point type".to_string(),
        Self::new(_learning_rate, beta1, beta2, epsilon, weight_decay)
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
        self
    }

    /// Gets the epsilon parameter
    pub fn get_epsilon(&self) -> F {
        self.epsilon
    }

    /// Sets the epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: F) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
    }

    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weightdecay: F) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

impl<F: Float + ScalarOperand + Debug> Optimizer<F> for RAdam<F> {
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
        }
        // Increment timestep
        self.t += 1;
        // Initialize moment estimates if needed
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
            self.v = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
        }

        // Prepare common terms
        let one_minus_beta1 = F::one() - self.beta1;
        let one_minus_beta2 = F::one() - self.beta2;
        let beta1_pow_t = self.beta1.powi(self.t as i32);
        let beta2_pow_t = self.beta2.powi(self.t as i32);
        let bias_correction1 = F::one() - beta1_pow_t;
        // RAdam-specific calculations
        let _two = F::from(2.0).ok_or_else(|| {
            NeuralError::InvalidArgument("Failed to convert 2.0 to floating point type".to_string())
        })?;
        let _four = F::from(4.0).ok_or_else(|| {
            NeuralError::InvalidArgument("Failed to convert 4.0 to floating point type".to_string())
        })?;
        #[allow(dead_code)]
        let _five = F::from(5.0).ok_or_else(|| {
            NeuralError::InvalidArgument("Failed to convert 5.0 to floating point type".to_string())
        })?; // Unused but keep for reference
        // Compute rho_t
        // All calculations in f64 first
        let rho_inf_f64 = self.rho_inf.to_f64().ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert rho_inf to f64".to_string())
        })?;
        let beta2_pow_t_f64 = beta2_pow_t.to_f64().ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert beta2_pow_t to f64".to_string())
        })?;
        let t_f64 = self.t as f64;
        let rho_t_f64 = rho_inf_f64 - 2.0 * t_f64 * beta2_pow_t_f64 / (1.0 - beta2_pow_t_f64);
        // Convert to F type
        let rho_t = F::from(rho_t_f64).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert rho_t from f64".to_string())
        // Rectification term calculation
        let rect_term;
        let use_adaptive_lr;
        if rho_t > four {
            // Compute the rectification term
            let rho_t_minus_4 = rho_t - four;
            let rho_t_minus_2 = rho_t - two;
            let rt = (rho_t_minus_2 / rho_t_minus_4)
                * ((rho_t - four) * (rho_t - two) / (rho_t * rho_t_minus_4)).sqrt();
            rect_term = rt;
            use_adaptive_lr = true;
        } else {
            rect_term = F::one();
            use_adaptive_lr = false;
        // Update parameters for each param-grad pair
        for i in 0..params.len() {
            // Apply weight decay to gradients if needed
            let adjusted_grad = if self.weight_decay > F::zero() {
                &grads[i] + &(&params[i] * self.weight_decay)
            } else {
                grads[i].clone()
            };
            // Update biased first moment estimate (same as Adam)
            self.m[i] = &self.m[i] * self.beta1 + &(&adjusted_grad * one_minus_beta1);
            // Update biased second raw moment estimate (same as Adam)
            self.v[i] =
                &self.v[i] * self.beta2 + &(adjusted_grad.mapv(|x| x * x) * one_minus_beta2);
            // Compute bias-corrected first moment estimate
            let m_hat = &self.m[i] / bias_correction1;
            // Parameter update differs based on whether we're in the adaptive phase or not
            if use_adaptive_lr {
                // Compute bias-corrected second raw moment estimate
                let v_hat = &self.v[i] * (F::one() / (F::one() - beta2_pow_t));
                // Compute the denominator with the rectification term
                let denom = v_hat.mapv(|x| x.sqrt()) + self.epsilon;
                // Apply the rectified update
                params[i] = &params[i] - &(m_hat / denom * self.learning_rate * rect_term);
                // In the early phase, use SGD with momentum instead of the adaptive learning rate
                params[i] = &params[i] - &(m_hat * self.learning_rate);
            }
        Ok(())
    fn get_learning_rate(&self) -> F {
        self.learning_rate
    fn set_learning_rate(&mut self, lr: F) {
        self.learning_rate = lr;
// Enable direct usage of scirs2-optim's RAdam when the optim feature is enabled
#[cfg(feature = "optim")]
pub use scirs2_optim::RAdam as OptimRAdam;
