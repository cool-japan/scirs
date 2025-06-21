//! Sequential model implementation
//!
//! This module provides a sequential model implementation that chains
//! layers together in a linear sequence.

use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use crate::losses::Loss;
use crate::models::Model;
use crate::optimizers::Optimizer;

/// A sequential model that chains layers together in a linear sequence
pub struct Sequential<F: Float + Debug + ScalarOperand + 'static> {
    layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    layer_outputs: Vec<Array<F, ndarray::IxDyn>>,
    input: Option<Array<F, ndarray::IxDyn>>,
}

impl<F: Float + Debug + ScalarOperand + 'static> Default for Sequential<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Clone for Sequential<F> {
    fn clone(&self) -> Self {
        // Note: We can't clone the layer trait objects directly
        // This creates a new empty model - the Clone trait is mainly for testing
        Sequential {
            layers: Vec::new(), // Cannot clone trait objects
            layer_outputs: Vec::new(),
            input: None,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Sequential<F> {
    /// Create a new empty sequential model
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            layer_outputs: Vec::new(),
            input: None,
        }
    }

    /// Create a new sequential model from existing layers
    pub fn from_layers(layers: Vec<Box<dyn Layer<F> + Send + Sync>>) -> Self {
        Sequential {
            layers,
            layer_outputs: Vec::new(),
            input: None,
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
    }

    /// Get the layers in the model
    pub fn layers(&self) -> &[Box<dyn Layer<F> + Send + Sync>] {
        &self.layers
    }

    /// Get a mutable reference to the layers in the model
    pub fn layers_mut(&mut self) -> &mut Vec<Box<dyn Layer<F> + Send + Sync>> {
        &mut self.layers
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Model<F> for Sequential<F> {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let mut current_output = input.clone();

        for layer in &self.layers {
            current_output = layer.forward(&current_output)?;
        }

        Ok(current_output)
    }

    fn backward(
        &self,
        input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        if self.layer_outputs.is_empty() {
            return Err(NeuralError::InferenceError(
                "No forward pass performed before backward pass".to_string(),
            ));
        }

        let mut grad_input = grad_output.clone();

        // Iterate through layers in reverse
        for (i, layer) in self.layers.iter().enumerate().rev() {
            // Get input for this layer (either from previous layer or the original input)
            let layer_input = if i > 0 {
                &self.layer_outputs[i - 1]
            } else if let Some(saved_input) = &self.input {
                saved_input
            } else {
                // Fallback to the provided input if nothing is saved
                input
            };

            grad_input = layer.backward(layer_input, &grad_input)?;
        }

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(learning_rate)?;
        }

        Ok(())
    }

    fn train_batch(
        &mut self,
        inputs: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
        loss_fn: &dyn Loss<F>,
        optimizer: &mut dyn Optimizer<F>,
    ) -> Result<F> {
        // Forward pass
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        let mut current_output = inputs.clone();

        for layer in &self.layers {
            current_output = layer.forward(&current_output)?;
            layer_outputs.push(current_output.clone());
        }

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

        // Iterate through layers in reverse
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            // Get input for this layer (either from previous layer or the original input)
            let layer_input = if i > 0 {
                &self.layer_outputs[i - 1]
            } else {
                inputs
            };

            grad_input = layer.backward(layer_input, &grad_input)?;
        }

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
                }
            }
        }

        // Update parameters using optimizer
        optimizer.update(&mut all_params, &all_grads)?;

        // Update the layers with the optimized parameters
        let mut param_idx = 0;
        for i in param_layers {
            // Need to use concrete type instead of dyn trait for downcasting
            if let Some(param_layer) = self.layers[i]
                .as_any_mut()
                .downcast_mut::<Box<dyn ParamLayer<F> + Send + Sync>>()
            {
                let num_params = param_layer.get_parameters().len();
                if param_idx + num_params <= all_params.len() {
                    let layer_params = all_params[param_idx..param_idx + num_params].to_vec();
                    param_layer.set_parameters(layer_params)?;
                    param_idx += num_params;
                }
            }
        }

        Ok(loss)
    }

    fn predict(&self, inputs: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        self.forward(inputs)
    }

    fn evaluate(
        &self,
        inputs: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F> {
        let predictions = self.forward(inputs)?;
        loss_fn.forward(&predictions, targets)
    }
}
