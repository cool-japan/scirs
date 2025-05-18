//! Stochastic Gradient Descent optimizer

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Stochastic Gradient Descent optimizer
///
/// Implements the classic SGD algorithm with support for momentum and weight decay.
///
/// Formula:
/// v_t = momentum * v_{t-1} + learning_rate * (gradient + weight_decay * param)
/// param_t = param_{t-1} - v_t
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{SGD, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an SGD optimizer with learning rate 0.01 and momentum 0.9
/// let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SGD<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Momentum factor (0.0 means no momentum)
    momentum: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Velocity (momentum state)
    velocity: Option<Vec<Array<A, ndarray::IxDyn>>>,
}

impl<A: Float + ScalarOperand + Debug> SGD<A> {
    /// Creates a new SGD optimizer with the given learning rate and no momentum/weight decay
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            momentum: A::zero(),
            weight_decay: A::zero(),
            velocity: None,
        }
    }

    /// Creates a new SGD optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `momentum` - The momentum factor (0.0 means no momentum)
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn new_with_config(learning_rate: A, momentum: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: None,
        }
    }

    /// Sets the momentum factor
    ///
    /// # Arguments
    ///
    /// * `momentum` - The momentum factor (0.0 means no momentum)
    pub fn set_momentum(&mut self, momentum: A) -> &mut Self {
        self.momentum = momentum;
        self
    }

    /// Builder method to set momentum and return self
    ///
    /// # Arguments
    ///
    /// * `momentum` - The momentum factor (0.0 means no momentum)
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = momentum;
        self
    }

    /// Gets the current momentum factor
    pub fn get_momentum(&self) -> A {
        self.momentum
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the weight decay factor
    ///
    /// # Arguments
    ///
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to set weight decay and return self
    ///
    /// # Arguments
    ///
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the current weight decay factor
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }
}

impl<A, D> Optimizer<A, D> for SGD<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to dynamic dimension for storage in velocity
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        // Initialize velocity if this is the first step
        if self.velocity.is_none() {
            self.velocity = Some(vec![Array::zeros(params_dyn.raw_dim())]);
        }

        let velocity = self.velocity.as_mut().unwrap();

        // Ensure we have velocity for this parameter set
        if velocity.is_empty() {
            velocity.push(Array::zeros(params_dyn.raw_dim()));
        } else if velocity[0].raw_dim() != params_dyn.raw_dim() {
            // If the parameter dimensions have changed, reset velocity
            velocity[0] = Array::zeros(params_dyn.raw_dim());
        }

        // Apply weight decay to gradients if needed
        let adjusted_gradients = if self.weight_decay > A::zero() {
            &gradients_dyn + &(&params_dyn * self.weight_decay)
        } else {
            gradients_dyn
        };

        // Update velocity with momentum
        if self.momentum > A::zero() {
            velocity[0] =
                &velocity[0] * self.momentum + &(&adjusted_gradients * self.learning_rate);
        } else {
            velocity[0] = &adjusted_gradients * self.learning_rate;
        }

        // Update parameters
        let updated_params = &params_dyn - &velocity[0];

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
