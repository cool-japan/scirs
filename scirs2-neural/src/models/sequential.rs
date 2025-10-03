//! Sequential model implementation
//!
//! This module provides a sequential model implementation that chains
//! layers together in a linear sequence.

use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use crate::losses::Loss;
use crate::models::{History, Model, TrainingConfig};
use crate::optimizers::Optimizer;
/// A sequential model that chains layers together in a linear sequence
pub struct Sequential<F: Float + Debug + ScalarOperand + 'static> {
    layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    layer_outputs: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    input: Option<Array<F, scirs2_core::ndarray::IxDyn>>,
    history: History<F>,
}
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + 'static> Default
    for Sequential<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + 'static> Clone for Sequential<F> {
    fn clone(&self) -> Self {
        // Note: We can't clone the layer trait objects directly
        // This creates a new empty model - the Clone trait is mainly for testing
        Sequential {
            layers: Vec::new(), // Cannot clone trait objects
            layer_outputs: Vec::new(),
            input: None,
            history: History::default(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + 'static> Sequential<F> {
    /// Create a new empty sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            layer_outputs: Vec::new(),
            input: None,
            history: History::default(),
        }
    }

    /// Create a new sequential model from existing layers
    pub fn from_layers(layers: Vec<Box<dyn Layer<F> + Send + Sync>>) -> Self {
        Self {
            layers,
            layer_outputs: Vec::new(),
            input: None,
            history: History::default(),
        }
    }

    /// Add a layer to the model
    pub fn add_layer<L: Layer<F> + 'static + Send + Sync>(&mut self, layer: L) -> &mut Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Get the number of layers in the model
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    /// Get the layers in the model
    pub fn layers(&self) -> &[Box<dyn Layer<F> + Send + Sync>] {
        &self.layers
    /// Get a mutable reference to the layers in the model
    pub fn layers_mut(&mut self) -> &mut Vec<Box<dyn Layer<F> + Send + Sync>> {
        &mut self.layers
    /// Get the training history
    pub fn training_history(&self) -> &History<F> {
        &self.history
    /// Get a mutable reference to the training history
    pub fn training_history_mut(&mut self) -> &mut History<F> {
        &mut self.history
    /// Predict on batched input data
    pub fn predict_batched(
        &self,
        inputs: &Array<F, scirs2_core::ndarray::IxDyn>,
        batch_size: usize,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let inputshape = inputs.shape();
        let num_samples = inputshape[0];
        let mut outputs = Vec::new();
        for i in (0..num_samples).step_by(batch_size) {
            let end_idx = std::cmp::min(i + batch_size, num_samples);
            let batch = inputs
                .slice(scirs2_core::ndarray::s![i..end_idx, ..])
                .to_owned()
                .into_dyn();
            let batch_output = self.forward(&batch)?;
            outputs.push(batch_output);
        // Concatenate all batch outputs
        if outputs.len() == 1 {
            Ok(outputs.into_iter().next().unwrap())
        } else {
            // For multiple batches, concatenate along the first axis
            let mut concatenated = outputs[0].clone();
            for output in outputs.into_iter().skip(1) {
                concatenated = scirs2_core::ndarray::concatenate![scirs2_core::ndarray::Axis(0), concatenated, output];
            }
            Ok(concatenated)
    /// Fit the model to training data
    pub fn fit(
        &mut self,
        x_train: &Array<F, scirs2_core::ndarray::IxDyn>,
        y_train: &Array<F, scirs2_core::ndarray::IxDyn>,
        config: &TrainingConfig,
        loss_fn: &dyn Loss<F>,
        optimizer: &mut dyn Optimizer<F>,
    ) -> Result<()> {
        let num_samples = x_train.shape()[0];
        let val_split_idx = if config.validation_split > 0.0 {
            ((1.0 - config.validation_split) * num_samples as f64) as usize
            num_samples
        };
        // Split data into training and validation sets
        let (x_train_split, x_val) = if config.validation_split > 0.0 {
            let x_train_split = x_train
                .slice(scirs2_core::ndarray::s![0..val_split_idx, ..])
            let x_val = x_train
                .slice(scirs2_core::ndarray::s![val_split_idx.., ..])
            (x_train_split, Some(x_val))
            (x_train.clone(), None)
        let (y_train_split, y_val) = if config.validation_split > 0.0 {
            let y_train_split = y_train
            let y_val = y_train
            (y_train_split, Some(y_val))
            (y_train.clone(), None)
        // Training loop
        for epoch in 0..config.epochs {
            let mut epoch_loss = F::zero();
            let num_batches = x_train_split.shape()[0].div_ceil(config.batch_size);
            // Training phase
            for i in 0..num_batches {
                let start_idx = i * config.batch_size;
                let end_idx =
                    std::cmp::min(start_idx + config.batch_size, x_train_split.shape()[0]);
                let batch_x = x_train_split
                    .slice(scirs2_core::ndarray::s![start_idx..end_idx, ..])
                    .to_owned()
                    .into_dyn();
                let batch_y = y_train_split
                let batch_loss = self.train_batch(&batch_x, &batch_y, loss_fn, optimizer)?;
                epoch_loss = epoch_loss + batch_loss;
            let avg_train_loss =
                epoch_loss / F::from_usize(num_batches).unwrap_or_else(|| F::one());
            self.history.train_loss.push(avg_train_loss);
            // Validation phase
            if let (Some(x_val), Some(y_val)) = (&x_val, &y_val) {
                let val_loss = self.evaluate(x_val, y_val, loss_fn)?;
                self.history.val_loss.push(val_loss);
            } else {
                // If no validation split, use training loss as validation loss
                self.history.val_loss.push(avg_train_loss);
            if config.verbose > 0 {
                println!(
                    "Epoch {}/{} - loss: {:.4}",
                    epoch + 1,
                    config.epochs,
                    avg_train_loss
                );
        Ok(())
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Display + 'static> Model<F>
    fn forward(&self, input: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut current_output = input.clone();
        for layer in &self.layers {
            current_output = layer.forward(&current_output)?;
        Ok(current_output)
    fn backward(
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        if self.layer_outputs.is_empty() {
            return Err(NeuralError::InferenceError(
                "No forward pass performed before backward pass".to_string(),
            ));
        let mut grad_input = grad_output.clone();
        // Iterate through layers in reverse
        for (i, layer) in self.layers.iter().enumerate().rev() {
            // Get input for this layer (either from previous layer or the original input)
            let layer_input = if i > 0 {
                &self.layer_outputs[i - 1]
            } else if let Some(saved_input) = &self.input {
                saved_input
                // Fallback to the provided input if nothing is saved
                input
            };
            grad_input = layer.backward(layer_input, &grad_input)?;
        Ok(grad_input)
    fn update(&mut self, learningrate: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(learningrate)?;
    fn train_batch(
        targets: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<F> {
        // Forward pass
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        let mut current_output = inputs.clone();
            layer_outputs.push(current_output.clone());
        // Save outputs for backward pass
        self.input = Some(inputs.clone());
        self.layer_outputs = layer_outputs;
        // Compute loss
        let predictions = self
            .layer_outputs
            .last()
            .ok_or_else(|| NeuralError::InferenceError("No layers in model".to_string()))?;
        let loss = loss_fn.forward(predictions, targets)?;
        // Backward pass to compute gradients
        let loss_grad = loss_fn.backward(predictions, targets)?;
        let mut grad_input = loss_grad;
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                inputs
        // Update parameters using optimizer
        let mut all_params = Vec::new();
        let mut all_grads = Vec::new();
        let mut param_layers = Vec::new();
        // First, collect all parameters and gradients
        for (i, layer) in self.layers.iter().enumerate() {
            // Need to use concrete type instead of dyn trait for downcasting
            // Try to use concrete implementations
            if let Some(param_layer) = layer
                .as_any()
                .downcast_ref::<Box<dyn ParamLayer<F> + Send + Sync>>()
            {
                param_layers.push(i);
                for param in param_layer.get_parameters() {
                    all_params.push(param.clone());
                }
                for grad in param_layer.get_gradients() {
                    all_grads.push(grad.clone());
        optimizer.update(&mut all_params, &all_grads)?;
        // Update the layers with the optimized parameters
        let mut param_idx = 0;
        for i in param_layers {
            if let Some(param_layer) = self.layers[i]
                .as_any_mut()
                .downcast_mut::<Box<dyn ParamLayer<F> + Send + Sync>>()
                let num_params = param_layer.get_parameters().len();
                if param_idx + num_params <= all_params.len() {
                    let layer_params = all_params[param_idx..param_idx + num_params].to_vec();
                    param_layer.set_parameters(layer_params)?;
                    param_idx += num_params;
        Ok(loss)
    fn predict(&self, inputs: &Array<F, scirs2_core::ndarray::IxDyn>) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        self.forward(inputs)
    fn evaluate(
        let predictions = self.forward(inputs)?;
        loss_fn.forward(&predictions, targets)
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::layers::Dense;
//     use crate::losses::MeanSquaredError;
//     use crate::models::TrainingConfig;
//     use crate::optimizers::SGD;
//     use scirs2_core::ndarray::Array2;
//     use scirs2_core::random::rngs::StdRng;
//     use scirs2_core::random::SeedableRng;
//
//     // Note: Tests temporarily disabled due to rand version conflicts
//     // TODO: Fix rand version compatibility issues
//     #[allow(dead_code)]
//     fn _test_sequential_training() {
//         let mut rng = StdRng::seed_from_u64(42);
//         let mut model = Sequential::<f64>::new();
//         // Add layers
//         model.add_layer(Dense::new(4, 8, None, &mut rng).unwrap());
//         model.add_layer(Dense::new(8, 4, None, &mut rng).unwrap());
//         model.add_layer(Dense::new(4, 1, None, &mut rng).unwrap());
//         // Create dummy data
//         let train_x = Array2::<f64>::from_elem((100, 4), 0.5).into_dyn();
//         let train_y = Array2::<f64>::from_elem((100, 1), 1.0).into_dyn();
//         // Training configuration
//         let config = TrainingConfig {
//             batch_size: 10,
//             epochs: 5,
//             validation_split: 0.2,
//             verbose: 0,
//             ..Default::default()
//         };
//         let loss_fn = MeanSquaredError::new();
//         let mut optimizer = SGD::new(0.01);
//         // Train the model
//         model.fit(&train_x, &train_y, &config, &loss_fn, &mut optimizer).unwrap();
//         // Check that training history was recorded
//         assert!(!model.training_history().train_loss.is_empty());
//         assert!(!model.training_history().val_loss.is_empty());
//     }
//     #[test]
//     fn test_sequential_prediction() {
//         model.add_layer(Dense::new(3, 5, None, &mut rng).unwrap());
//         model.add_layer(Dense::new(5, 2, None, &mut rng).unwrap());
//         let input = Array2::<f64>::from_elem((2, 3), 0.5).into_dyn();
//         let output = model.predict(&input).unwrap();
//         assert_eq!(output.shape(), &[2, 2]);
//     fn test_batched_prediction() {
//         let input = Array2::<f64>::from_elem((25, 3), 0.5).into_dyn();
//         let output = model.predict_batched(&input, 10).unwrap();
//         assert_eq!(output.shape(), &[25, 2]);
// }
