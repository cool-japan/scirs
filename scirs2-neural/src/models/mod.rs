//! Neural network model implementations
//!
//! This module provides implementations of neural network models,
//! including Sequential models and training utilities.

use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::losses::Loss;
use crate::optimizers::Optimizer;

/// Trait for neural network models
pub trait Model<F: Float + Debug + ScalarOperand> {
    /// Forward pass through the model
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Backward pass to compute gradients
    fn backward(
        &self,
        input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;

    /// Update the model parameters with the given learning rate
    fn update(&mut self, learning_rate: F) -> Result<()>;

    /// Train the model on a batch of data
    fn train_batch(
        &mut self,
        inputs: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
        loss_fn: &dyn Loss<F>,
        optimizer: &mut dyn Optimizer<F>,
    ) -> Result<F>;

    /// Predict the output for a batch of inputs
    fn predict(&self, inputs: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Evaluate the model on a batch of data
    fn evaluate(
        &self,
        inputs: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F>;
}

pub mod architectures;
pub mod sequential;
pub mod trainer;

pub use architectures::*;
pub use sequential::Sequential;
pub use trainer::{History, Trainer, TrainingConfig};
