//! Optimization algorithms for machine learning
//!
//! This module provides various optimization algorithms commonly used in machine learning,
//! such as Stochastic Gradient Descent (SGD), Adam, RMSprop, and others.

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Trait that defines the interface for optimization algorithms
pub trait Optimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Updates parameters using the given gradients
    ///
    /// # Arguments
    ///
    /// * `params` - The current parameter values
    /// * `gradients` - The gradients of the parameters
    ///
    /// # Returns
    ///
    /// The updated parameters
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>>;

    /// Gets the current learning rate
    fn get_learning_rate(&self) -> A
    where
        Self: Sized;

    /// Sets a new learning rate
    fn set_learning_rate(&mut self, learning_rate: A)
    where
        Self: Sized;

    /// Updates multiple parameter arrays at once
    ///
    /// # Arguments
    ///
    /// * `params_list` - List of parameter arrays
    /// * `gradients_list` - List of gradient arrays corresponding to the parameters
    ///
    /// # Returns
    ///
    /// Updated parameter arrays
    fn step_list(
        &mut self,
        params_list: &[&Array<A, D>],
        gradients_list: &[&Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        if params_list.len() != gradients_list.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params_list.len(),
                gradients_list.len()
            )));
        }

        let mut results = Vec::with_capacity(params_list.len());
        for (params, grads) in params_list.iter().zip(gradients_list.iter()) {
            results.push(self.step(params, grads)?);
        }
        Ok(results)
    }
}

// Import specific optimizers
mod adagrad;
mod adam;
mod adamw;
mod radam;
mod rmsprop;
mod sgd;

// Re-export specific optimizers
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use sgd::SGD;
