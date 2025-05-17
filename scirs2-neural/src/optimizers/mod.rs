//! Neural network optimizers
//!
//! This module provides various optimization algorithms for neural networks
//! such as SGD, Adam, RMSProp, Adagrad, etc.
//!
//! The optimizers in this module are wrappers around the implementations in
//! the `scirs2-optim` crate, adapted to work with the neural network API.

use crate::error::Result;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

// Re-export scirs2-optim's optimizers when the feature is enabled
#[cfg(feature = "optim")]
pub use scirs2_optim::optimizers as optim_optimizers;

/// Trait for neural network optimizers
pub trait Optimizer<F: Float + Debug + ScalarOperand> {
    /// Update parameters based on gradients
    fn update(
        &mut self,
        params: &mut [Array<F, ndarray::IxDyn>],
        grads: &[Array<F, ndarray::IxDyn>],
    ) -> Result<()>;

    /// Get the current learning rate
    fn get_learning_rate(&self) -> F;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: F);
}

/// Extension methods for optimizers
pub trait OptimizerExt<F: Float + Debug + ScalarOperand>: Optimizer<F> {
    /// Step the optimizer with a model's parameters
    fn step<L: crate::layers::Layer<F> + ?Sized>(&mut self, model: &mut L) -> Result<()> {
        let params = model.params();
        let gradients = model.gradients();
        self.update(&mut params.clone(), &gradients)
    }
}

// Implement OptimizerExt for all Optimizer implementations
impl<F: Float + Debug + ScalarOperand, T: Optimizer<F> + ?Sized> OptimizerExt<F> for T {}

mod adagrad;
mod adam;
mod adamw;
mod lr_scheduler_wrapper;
mod radam;
mod rmsprop;
mod sgd;

pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use lr_scheduler_wrapper::{with_cosine_annealing, with_step_decay, LRSchedulerOptimizer};
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use sgd::SGD;

// Converters between scirs2-optim optimizers and neural network optimizers
#[cfg(feature = "optim")]
mod wrappers {
    use super::*;
    use crate::NeuralError;
    use ndarray::Dimension;
    use scirs2_optim::optimizers as optim;

    /// Wrapper for optim optimizers to conform to our Optimizer trait
    pub struct OptimOptimizerWrapper<F, D, O>
    where
        F: Float + Debug + ScalarOperand,
        D: Dimension,
        O: optim::Optimizer<F, D>,
    {
        optimizer: O,
        phantom: std::marker::PhantomData<(F, D)>,
    }

    impl<F, D, O> OptimOptimizerWrapper<F, D, O>
    where
        F: Float + Debug + ScalarOperand,
        D: Dimension,
        O: optim::Optimizer<F, D>,
    {
        pub fn new(optimizer: O) -> Self {
            Self {
                optimizer,
                phantom: std::marker::PhantomData,
            }
        }

        pub fn inner(&self) -> &O {
            &self.optimizer
        }

        pub fn inner_mut(&mut self) -> &mut O {
            &mut self.optimizer
        }
    }

    impl<F, D, O> super::Optimizer<F> for OptimOptimizerWrapper<F, D, O>
    where
        F: Float + Debug + ScalarOperand,
        D: Dimension,
        O: optim::Optimizer<F, D>,
    {
        fn update(
            &mut self,
            params: &mut [Array<F, ndarray::IxDyn>],
            grads: &[Array<F, ndarray::IxDyn>],
        ) -> Result<()> {
            if params.len() != grads.len() {
                return Err(NeuralError::TrainingError(format!(
                    "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                    params.len(),
                    grads.len()
                )));
            }

            // Convert to refs for optim library
            let params_refs: Vec<&Array<F, ndarray::IxDyn>> = params.iter().collect();
            let grads_refs: Vec<&Array<F, ndarray::IxDyn>> = grads.iter().collect();

            // Use the optim library's step_list method
            let updated_params = self
                .optimizer
                .step_list(&params_refs, &grads_refs)
                .map_err(|e| NeuralError::TrainingError(format!("Optimizer error: {}", e)))?;

            // Update the original params with the updated values
            for (i, param) in params.iter_mut().enumerate() {
                *param = updated_params[i].clone();
            }

            Ok(())
        }

        fn get_learning_rate(&self) -> F {
            self.optimizer.get_learning_rate()
        }

        fn set_learning_rate(&mut self, lr: F) {
            self.optimizer.set_learning_rate(lr);
        }
    }

    // Helper function to create wrapped SGD optimizer
    pub fn wrap_sgd<F: Float + Debug + ScalarOperand>(
        lr: F,
        momentum: F,
        weight_decay: F,
    ) -> OptimOptimizerWrapper<F, ndarray::IxDyn, optim::SGD<F>> {
        let sgd = optim::SGD::new_with_config(lr, momentum, weight_decay);
        OptimOptimizerWrapper::new(sgd)
    }

    // Helper function to create wrapped Adam optimizer
    pub fn wrap_adam<F: Float + Debug + ScalarOperand>(
        lr: F,
        beta1: F,
        beta2: F,
        epsilon: F,
        weight_decay: F,
    ) -> OptimOptimizerWrapper<F, ndarray::IxDyn, optim::Adam<F>> {
        let adam = optim::Adam::new_with_config(lr, beta1, beta2, epsilon, weight_decay);
        OptimOptimizerWrapper::new(adam)
    }

    // Helper function to create wrapped AdamW optimizer
    pub fn wrap_adamw<F: Float + Debug + ScalarOperand>(
        lr: F,
        beta1: F,
        beta2: F,
        epsilon: F,
        weight_decay: F,
    ) -> OptimOptimizerWrapper<F, ndarray::IxDyn, optim::AdamW<F>> {
        let adamw = optim::AdamW::new_with_config(lr, beta1, beta2, epsilon, weight_decay);
        OptimOptimizerWrapper::new(adamw)
    }

    // Helper function to create wrapped RAdam optimizer
    pub fn wrap_radam<F: Float + Debug + ScalarOperand>(
        lr: F,
        beta1: F,
        beta2: F,
        epsilon: F,
        weight_decay: F,
    ) -> OptimOptimizerWrapper<F, ndarray::IxDyn, optim::RAdam<F>> {
        let radam = optim::RAdam::new_with_config(lr, beta1, beta2, epsilon, weight_decay);
        OptimOptimizerWrapper::new(radam)
    }
}
