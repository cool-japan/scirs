//! Neural network layers implementation
//!
//! This module provides implementations of various neural network layers
//! such as dense (fully connected), attention, convolution, pooling, etc.

use crate::error::Result;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Base trait for neural network layers
pub trait Layer<F: Float + Debug + ScalarOperand> {
    /// Forward pass of the layer
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Backward pass of the layer to compute gradients
    fn backward(
        &self,
        input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;

    /// Update the layer parameters with the given gradients
    fn update(&mut self, learning_rate: F) -> Result<()>;

    /// Get the layer as a dyn Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the layer as a mutable dyn Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get the parameters of the layer
    fn params(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        Vec::new()
    }

    /// Get the gradients of the layer parameters
    fn gradients(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        Vec::new()
    }

    /// Set the gradients of the layer parameters
    fn set_gradients(&mut self, _gradients: &[Array<F, ndarray::IxDyn>]) -> Result<()> {
        Ok(())
    }

    /// Set the parameters of the layer
    fn set_params(&mut self, _params: &[Array<F, ndarray::IxDyn>]) -> Result<()> {
        Ok(())
    }

    /// Set the layer to training mode (true) or evaluation mode (false)
    fn set_training(&mut self, _training: bool) {
        // Default implementation: do nothing
    }

    /// Get the current training mode
    fn is_training(&self) -> bool {
        true // Default implementation: always in training mode
    }

    /// Get the type of the layer (e.g., "Dense", "Conv2D")
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Get the number of trainable parameters in this layer
    fn parameter_count(&self) -> usize {
        0
    }

    /// Get a detailed description of this layer
    fn layer_description(&self) -> String {
        format!("type:{}", self.layer_type())
    }
}

/// Trait for layers with parameters (weights, biases)
pub trait ParamLayer<F: Float + Debug + ScalarOperand>: Layer<F> {
    /// Get the parameters of the layer as a vector of arrays
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>>;

    /// Get the gradients of the parameters
    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>>;

    /// Set the parameters of the layer
    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()>;
}

mod attention;
mod conv;
mod dense;
mod dropout;
mod embedding;
mod normalization;
mod recurrent;
mod rnn_thread_safe;

// Re-export layer types
pub use attention::{AttentionConfig, AttentionMask, MultiHeadAttention, SelfAttention};
pub use conv::{Conv2D, GlobalAvgPool2D, MaxPool2D, PaddingMode};
pub use dense::Dense;
pub use dropout::Dropout;
pub use embedding::{Embedding, EmbeddingConfig, PatchEmbedding, PositionalEmbedding};
pub use normalization::{BatchNorm, LayerNorm, LayerNorm2D};
pub use recurrent::{
    Bidirectional, GRUConfig, LSTMConfig, RNNConfig, RecurrentActivation, GRU, LSTM, RNN,
};
pub use rnn_thread_safe::{
    RecurrentActivation as ThreadSafeRecurrentActivation, ThreadSafeBidirectional, ThreadSafeRNN,
};

// Configuration types
/// Configuration enum for different types of layers
#[derive(Debug, Clone)]
pub enum LayerConfig {
    /// Dense (fully connected) layer
    Dense,
    /// 2D Convolutional layer
    Conv2D,
    /// Recurrent Neural Network layer
    RNN,
    /// Long Short-Term Memory layer
    LSTM,
    /// Gated Recurrent Unit layer
    GRU,
    // Add other layer types as needed
}

/// Sequential container for neural network layers
///
/// A Sequential model is a linear stack of layers.
/// Layers are executed in sequence during forward and backward passes.
pub struct Sequential<F: Float + Debug + ScalarOperand> {
    layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    training: bool,
}

impl<F: Float + Debug + ScalarOperand> std::fmt::Debug for Sequential<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .field("training", &self.training)
            .finish()
    }
}

// We can't clone trait objects directly
// This is a minimal implementation that won't clone the actual layers
impl<F: Float + Debug + ScalarOperand + 'static> Clone for Sequential<F> {
    fn clone(&self) -> Self {
        // We can't clone the layers, so we just create an empty Sequential
        // with the same training flag
        Self {
            layers: Vec::new(),
            training: self.training,
        }
    }
}

impl<F: Float + Debug + ScalarOperand> Default for Sequential<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + ScalarOperand> Sequential<F> {
    /// Create a new Sequential container
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Add a layer to the container
    pub fn add<L: Layer<F> + Send + Sync + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// Get the number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if there are no layers
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for Sequential<F> {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output)?;
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // For simplicity, we'll just return the grad_output as-is
        // A real implementation would propagate through the layers in reverse
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(learning_rate)?;
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        let mut params = Vec::new();

        for layer in &self.layers {
            params.extend(layer.params());
        }

        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;

        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
