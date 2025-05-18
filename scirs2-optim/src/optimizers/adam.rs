//! Adam optimizer implementation

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Adam optimizer
///
/// Implements the Adam optimization algorithm from the paper:
/// "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014).
///
/// Formula:
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)
/// v_hat_t = v_t / (1 - beta2^t)
/// theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{Adam, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an Adam optimizer with default hyperparameters
/// let mut optimizer = Adam::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Adam<A: Float + ScalarOperand + Debug> {
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
    /// First moment vector
    m: Option<Vec<Array<A, ndarray::IxDyn>>>,
    /// Second moment vector
    v: Option<Vec<Array<A, ndarray::IxDyn>>>,
    /// Current timestep
    t: usize,
}

impl<A: Float + ScalarOperand + Debug> Adam<A> {
    /// Creates a new Adam optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            weight_decay: A::zero(),
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Creates a new Adam optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(
        learning_rate: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        weight_decay: A,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
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

    /// Builder method to set beta1 and return self
    pub fn with_beta1(mut self, beta1: A) -> Self {
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

    /// Builder method to set beta2 and return self
    pub fn with_beta2(mut self, beta2: A) -> Self {
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

    /// Builder method to set epsilon and return self
    pub fn with_epsilon(mut self, epsilon: A) -> Self {
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

    /// Builder method to set weight decay and return self
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
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

impl<A, D> Optimizer<A, D> for Adam<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to dynamic dimension for storage in state vectors
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        // Apply weight decay to gradients if needed
        let adjusted_gradients = if self.weight_decay > A::zero() {
            &gradients_dyn + &(&params_dyn * self.weight_decay)
        } else {
            gradients_dyn
        };

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
        m[0] = &m[0] * self.beta1 + &(&adjusted_gradients * (A::one() - self.beta1));

        // Update biased second raw moment estimate
        v[0] = &v[0] * self.beta2
            + &(&adjusted_gradients * &adjusted_gradients * (A::one() - self.beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = &m[0] / (A::one() - self.beta1.powi(self.t as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &v[0] / (A::one() - self.beta2.powi(self.t as i32));

        // Compute square root of v_hat
        let v_hat_sqrt = v_hat.mapv(|x| x.sqrt());

        // Update parameters
        let step = &m_hat / &(&v_hat_sqrt + self.epsilon) * self.learning_rate;
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
