//! Neural network optimizers
//!
//! This module provides various optimization algorithms for neural networks
//! such as SGD, Adam, RMSProp, Adagrad, etc.
//! The optimizers in this module are wrappers around the implementations in
//! the `scirs2-optim` crate, adapted to work with the neural network API.

use crate::error::Result;
use crate::layers::ParamLayer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
// Re-export scirs2-optim's optimizers when the feature is enabled
// Note: scirs2_optim crate may not exist yet - commented out for compilation
// #[cfg(feature = "optim")]
// pub use scirs2_optim::optimizers as optim_optimizers;
/// Trait for neural network optimizers
pub trait Optimizer<F: Float + Debug + ScalarOperand> {
    /// Update parameters based on gradients
    fn update(
        &mut self,
        params: &mut [Array<F, scirs2_core::ndarray::IxDyn>],
        grads: &[Array<F, scirs2_core::ndarray::IxDyn>],
    ) -> Result<()>;
    /// Update model parameters using the optimizer (non-generic version for trait objects)
    fn step_model(&mut self, model: &mut dyn ParamLayer<F>) -> Result<()> {
        // Get parameters and gradients from the model
        let params_refs = model.get_parameters();
        let grads_refs = model.get_gradients();
        // Convert to owned arrays for update
        let mut params: Vec<Array<F, scirs2_core::ndarray::IxDyn>> =
            params_refs.iter().map(|p| (*p).clone()).collect();
        let grads: Vec<Array<F, scirs2_core::ndarray::IxDyn>> =
            grads_refs.iter().map(|g| (*g).clone()).collect();
        // Update parameters
        self.update(&mut params, &grads)?;
        // Set updated parameters back to the model
        model.set_parameters(params)?;
        Ok(())
    }
    /// Get the current learning rate
    fn get_learning_rate(&self) -> F;
    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: F);
    /// Reset the optimizer's internal state
    fn reset(&mut self) {}
    /// Get the optimizer's name
    fn name(&self) -> &'static str {
        "Unknown"
}
/// Extension trait for optimizers that can work with model layers
pub trait OptimizerStep<F: Float + Debug + ScalarOperand> {
    /// Update model parameters using the optimizer
    fn step<L: ParamLayer<F> + ?Sized>(&mut self, model: &mut L) -> Result<()>;
/// Blanket implementation for all optimizers
impl<F: Float + Debug + ScalarOperand, O: Optimizer<F>> OptimizerStep<F> for O {
    fn step<L: ParamLayer<F> + ?Sized>(&mut self, model: &mut L) -> Result<()> {
/// Implementation for trait objects
impl<F: Float + Debug + ScalarOperand> OptimizerStep<F> for dyn Optimizer<F> + Send + Sync {
// Standard optimizer implementations
pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod lr_scheduler_wrapper;
pub mod momentum;
pub mod radam;
pub mod rmsprop;
pub mod sgd;
// Re-export standard implementations
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use lr_scheduler_wrapper::{with_cosine_annealing, with_step_decay, LRSchedulerOptimizer};
pub use momentum::MomentumOptimizer;
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use sgd::SGD;
// Traits are already defined in this module
// Converters between scirs2-optim optimizers and neural network optimizers
// Note: wrappers module commented out due to missing scirs2_optim crate
// mod wrappers {
// use super::*;
// use crate::NeuralError;
// use scirs2_core::ndarray::Dimension;
// use scirs2_optim::optimizers as optim;
// /// Wrapper for optim optimizers to conform to our Optimizer trait
// pub struct OptimOptimizerWrapper<F, D, O>
// where
//     F: Float + Debug + ScalarOperand,
//     D: Dimension,
//     O: optim::Optimizer<F, D>,
// {
//     optimizer: O,
//     phantom: std::marker::PhantomData<(F, D)>,
// }
// impl<F, D, O> OptimOptimizerWrapper<F, D, O>
//     pub fn new(optimizer: O) -> Self {
//         Self {
//             optimizer,
//             phantom: std::marker::PhantomData,
//         }
//     }
//     pub fn inner(&self) -> &O {
//         &self.optimizer
//     pub fn inner_mut(&mut self) -> &mut O {
//         &mut self.optimizer
// impl<F, D, O> super::Optimizer<F> for OptimOptimizerWrapper<F, D, O>
//     fn update(
//         &mut self,
//         params: &mut [Array<F, scirs2_core::ndarray::IxDyn>],
//         grads: &[Array<F, scirs2_core::ndarray::IxDyn>],
//     ) -> Result<()> {
//         if params.len() != grads.len() {
//             return Err(NeuralError::TrainingError(format!(
//                 "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
//                 params.len(),
//                 grads.len()
//             )));
//         // Convert to refs for optim library
//         let params_refs: Vec<&Array<F, scirs2_core::ndarray::IxDyn>> = params.iter().collect();
//         let grads_refs: Vec<&Array<F, scirs2_core::ndarray::IxDyn>> = grads.iter().collect();
//         // Use the optim library's step_list method
//         let updated_params = self
//             .optimizer
//             .step_list(&params_refs, &grads_refs)
//             .map_err(|e| NeuralError::TrainingError(format!("Optimizer error: {}", e)))?;
//         // Update the original params with the updated values
//         for (i, param) in params.iter_mut().enumerate() {
//             *param = updated_params[i].clone();
//         Ok(())
//     fn get_learning_rate(&self) -> F {
//         self.optimizer.get_learning_rate()
//     fn set_learning_rate(&mut self, lr: F) {
//         self.optimizer.set_learning_rate(lr);
// // Helper function to create wrapped SGD optimizer
// pub fn wrap_sgd<F: Float + Debug + ScalarOperand>(
//     lr: F,
//     momentum: F,
//     weight_decay: F,
// ) -> OptimOptimizerWrapper<F, scirs2_core::ndarray::IxDyn, optim::SGD<F>> {
//     let sgd = optim::SGD::new_with_config(lr, momentum, weight_decay);
//     OptimOptimizerWrapper::new(sgd)
// // Helper function to create wrapped Adam optimizer
// pub fn wrap_adam<F: Float + Debug + ScalarOperand>(
//     beta1: F,
//     beta2: F,
//     epsilon: F,
// ) -> OptimOptimizerWrapper<F, scirs2_core::ndarray::IxDyn, optim::Adam<F>> {
//     let adam = optim::Adam::new_with_config(lr, beta1, beta2, epsilon, weight_decay);
//     OptimOptimizerWrapper::new(adam)
// // Helper function to create wrapped AdamW optimizer
// pub fn wrap_adamw<F: Float + Debug + ScalarOperand>(
// ) -> OptimOptimizerWrapper<F, scirs2_core::ndarray::IxDyn, optim::AdamW<F>> {
//     let adamw = optim::AdamW::new_with_config(lr, beta1, beta2, epsilon, weight_decay);
//     OptimOptimizerWrapper::new(adamw)
// // Helper function to create wrapped RAdam optimizer
// pub fn wrap_radam<F: Float + Debug + ScalarOperand>(
// ) -> OptimOptimizerWrapper<F, scirs2_core::ndarray::IxDyn, optim::RAdam<F>> {
//     let radam = optim::RAdam::new_with_config(lr, beta1, beta2, epsilon, weight_decay);
//     OptimOptimizerWrapper::new(radam)
