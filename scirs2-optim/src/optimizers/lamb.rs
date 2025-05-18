//! LAMB optimizer implementation
//!
//! Based on the paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
//! by You et al. (2019).

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// LAMB (Layer-wise Adaptive Moments) optimizer
///
/// LAMB is designed for large batch optimization. It extends AdamW with layer-wise
/// adaptive learning rates, making it particularly effective for training large models
/// with high batch sizes.
///
/// Formula:
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)
/// v_hat_t = v_t / (1 - beta2^t)
/// r1 = ||theta_t||
/// g' = m_hat_t / (sqrt(v_hat_t) + epsilon) + lambda * theta_t
/// r2 = ||g'||
/// ratio = r1/r2 if r1 > 0 and r2 > 0, else 1.0
/// theta_t = theta_{t-1} - lr * ratio * g'
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{LAMB, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create a LAMB optimizer with default hyperparameters
/// let mut optimizer = LAMB::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LAMB<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Exponential decay rate for the first moment estimates
    beta1: A,
    /// Exponential decay rate for the second moment estimates
    beta2: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Whether to use bias correction
    bias_correction: bool,
    /// First moment vector
    m: Option<Vec<Array<A, ndarray::IxDyn>>>,
    /// Second moment vector
    v: Option<Vec<Array<A, ndarray::IxDyn>>>,
    /// Current timestep
    t: usize,
}

impl<A: Float + ScalarOperand + Debug> LAMB<A> {
    /// Creates a new LAMB optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-6).unwrap(),
            weight_decay: A::zero(),
            bias_correction: true,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Creates a new LAMB optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-6)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    /// * `bias_correction` - Whether to use bias correction (default: true)
    pub fn new_with_config(
        learning_rate: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        weight_decay: A,
        bias_correction: bool,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            bias_correction,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Sets the beta1 parameter
    pub fn set_beta1(&mut self, beta1: A) -> &mut Self {
        self.beta1 = beta1;
        self
    }

    /// Gets the beta1 parameter
    pub fn get_beta1(&self) -> A {
        self.beta1
    }

    /// Sets the beta2 parameter
    pub fn set_beta2(&mut self, beta2: A) -> &mut Self {
        self.beta2 = beta2;
        self
    }

    /// Gets the beta2 parameter
    pub fn get_beta2(&self) -> A {
        self.beta2
    }

    /// Sets the epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: A) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Gets the epsilon parameter
    pub fn get_epsilon(&self) -> A {
        self.epsilon
    }

    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the learning rate
    pub fn set_lr(&mut self, lr: A) {
        self.learning_rate = lr;
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.t = 0;
    }
}

impl<A, D> Optimizer<A, D> for LAMB<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to dynamic dimension for storage in state vectors
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        // Initialize state if this is the first step
        if self.m.is_none() {
            self.m = Some(vec![Array::zeros(params_dyn.raw_dim())]);
            self.v = Some(vec![Array::zeros(params_dyn.raw_dim())]);
            self.t = 0;
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Ensure we have state for this parameter set
        if m.is_empty() {
            m.push(Array::zeros(params_dyn.raw_dim()));
            v.push(Array::zeros(params_dyn.raw_dim()));
        } else if m[0].raw_dim() != params_dyn.raw_dim() {
            // If the parameter dimensions have changed, reset state
            m[0] = Array::zeros(params_dyn.raw_dim());
            v[0] = Array::zeros(params_dyn.raw_dim());
        }

        // Increment timestep
        self.t += 1;

        // Update biased first moment estimate
        m[0] = &m[0] * self.beta1 + &gradients_dyn * (A::one() - self.beta1);

        // Update biased second raw moment estimate
        v[0] = &v[0] * self.beta2 + &(&gradients_dyn * &gradients_dyn * (A::one() - self.beta2));

        // Compute bias-corrected moments if enabled
        let (m_hat, v_hat) = if self.bias_correction {
            let bias1 = A::one() - self.beta1.powi(self.t as i32);
            let bias2 = A::one() - self.beta2.powi(self.t as i32);
            (&m[0] / bias1, &v[0] / bias2)
        } else {
            (m[0].clone(), v[0].clone())
        };

        // Compute adaptive term (similar to Adam)
        let v_hat_sqrt = v_hat.mapv(|x| x.sqrt());
        let adaptive_term = &m_hat / &(&v_hat_sqrt + self.epsilon);

        // Apply weight decay to create the full gradient term
        let normalized_gradient = if self.weight_decay > A::zero() {
            &adaptive_term + &(&params_dyn * self.weight_decay)
        } else {
            adaptive_term
        };

        // Layer-wise adaptation (trust ratio)
        let weight_norm = {
            let norm_sq = params_dyn
                .iter()
                .map(|x| *x * *x)
                .fold(A::zero(), |acc, x| acc + x);
            norm_sq.sqrt()
        };
        let gradient_norm = {
            let norm_sq = normalized_gradient
                .iter()
                .map(|x| *x * *x)
                .fold(A::zero(), |acc, x| acc + x);
            norm_sq.sqrt()
        };

        let trust_ratio = if weight_norm > A::zero() && gradient_norm > A::zero() {
            weight_norm / gradient_norm
        } else {
            A::one()
        };

        // Update parameters with the trust ratio
        let step = &normalized_gradient * (self.learning_rate * trust_ratio);
        let updated_params = &params_dyn - step;

        // Convert back to original dimension
        Ok(updated_params.into_dimensionality::<D>().unwrap())
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_lamb_basic_creation() {
        let optimizer: LAMB<f64> = LAMB::new(0.001);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.001);
        assert_abs_diff_eq!(optimizer.get_beta1(), 0.9);
        assert_abs_diff_eq!(optimizer.get_beta2(), 0.999);
        assert_abs_diff_eq!(optimizer.get_epsilon(), 1e-6);
        assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.0);
        assert!(optimizer.bias_correction);
    }

    #[test]
    fn test_lamb_convergence() {
        let mut optimizer: LAMB<f64> = LAMB::new(0.1);

        // Minimize a simple quadratic function: f(x) = x^2 + y^2
        let mut params = Array1::from_vec(vec![5.0, 3.0]);

        for _ in 0..50 {
            // Gradient of x^2 + y^2 is (2x, 2y)
            let gradients = Array1::from_vec(vec![2.0 * params[0], 2.0 * params[1]]);
            params = optimizer.step(&params, &gradients).unwrap();
        }

        // Should converge towards (0, 0)
        assert!(params[0].abs() < 1.0);
        assert!(params[1].abs() < 1.0);
    }

    #[test]
    fn test_lamb_with_weight_decay() {
        let mut optimizer: LAMB<f64> = LAMB::new_with_config(
            0.1,   // learning_rate
            0.9,   // beta1
            0.999, // beta2
            1e-6,  // epsilon
            0.1,   // weight_decay
            true,  // bias_correction
        );

        // Start from (1.0, 1.0)
        let mut params = Array1::from_vec(vec![1.0, 1.0]);

        // Run optimization with small gradients
        for _ in 0..20 {
            let gradients = Array1::from_vec(vec![0.1, 0.1]);
            params = optimizer.step(&params, &gradients).unwrap();
        }

        // With weight decay, parameters should decrease
        assert!(params[0] < 1.0);
        assert!(params[1] < 1.0);
    }

    #[test]
    fn test_lamb_reset() {
        let mut optimizer: LAMB<f64> = LAMB::new(0.1);

        // Perform a step to initialize state
        let params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![0.5]);
        let _ = optimizer.step(&params, &gradients).unwrap();

        // State should exist
        assert!(optimizer.m.is_some());
        assert!(optimizer.v.is_some());
        assert_eq!(optimizer.t, 1);

        // Reset
        optimizer.reset();

        // State should be cleared
        assert!(optimizer.m.is_none());
        assert!(optimizer.v.is_none());
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_lamb_trust_ratio() {
        // Test with normal gradient and parameters
        let mut optimizer: LAMB<f64> = LAMB::new(0.1);
        let params = Array1::from_vec(vec![2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.4, 0.6]);

        let new_params = optimizer.step(&params, &gradients).unwrap();

        // Parameters should be updated
        assert_ne!(new_params[0], params[0]);
        assert_ne!(new_params[1], params[1]);

        // Check they moved in the right direction
        assert!(new_params[0] < params[0]); // gradient was positive
        assert!(new_params[1] < params[1]); // gradient was positive
    }
}
