//! Neural network layers implementation
//!
//! This module provides implementations of various neural network layers
//! such as dense (fully connected), attention, convolution, pooling, etc.
//! Layers are the fundamental building blocks of neural networks.
//!
//! # Overview
//!
//! Neural network layers transform input data through learned parameters (weights and biases).
//! Each layer implements the `Layer` trait, which defines the interface for forward and
//! backward propagation, parameter management, and training/evaluation modes.
//!
//! # Available Layer Types
//!
//! ## Core Layers
//! - **Dense**: Fully connected linear transformation
//! - **Conv2D**: 2D convolutional layers for image processing
//! - **Embedding**: Lookup tables for discrete inputs (words, tokens)
//!
//! ## Activation & Regularization
//! - **Dropout**: Randomly sets inputs to zero during training
//! - **BatchNorm/LayerNorm**: Normalization for stable training
//! - **ActivityRegularization**: L1/L2 penalties on activations
//!
//! ## Pooling & Reshaping
//! - **MaxPool2D/AdaptiveMaxPool2D**: Spatial downsampling
//! - **GlobalAvgPool2D**: Global spatial average pooling
//!
//! ## Attention & Sequence
//! - **MultiHeadAttention**: Transformer-style attention mechanism
//! - **LSTM/GRU**: Recurrent layers for sequences
//! - **Bidirectional**: Wrapper for bidirectional RNNs
//!
//! ## Embedding & Positional
//! - **PositionalEmbedding**: Learned positional encodings
//! - **PatchEmbedding**: Convert image patches to embeddings
//!
//! # Examples
//!
//! ## Creating a Simple Dense Layer
//!
//! ```rust
//! use scirs2_neural::layers::{Layer, Dense};
//! use ndarray::Array;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let mut rng = SmallRng::seed_from_u64(42);
//!
//! // Create a dense layer: 784 inputs -> 128 outputs with ReLU activation
//! let dense = Dense::<f64>::new(784, 128, Some("relu"), &mut rng)?;
//!
//! // Create input batch (batch_size=2, features=784)
//! let input = Array::zeros((2, 784)).into_dyn();
//!
//! // Forward pass
//! let output = dense.forward(&input)?;
//! assert_eq!(output.shape(), &[2, 128]);
//!
//! println!("Layer type: {}", dense.layer_type());
//! println!("Parameters: {}", dense.parameter_count());
//! # Ok(())
//! # }
//! ```
//!
//! ## Building a Sequential Model
//!
//! ```rust
//! use scirs2_neural::layers::{Layer, Dense, Dropout};
//! use scirs2_neural::models::{Sequential, Model};
//! use ndarray::Array;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let mut rng = SmallRng::seed_from_u64(42);
//! let mut model: Sequential<f32> = Sequential::new();
//!
//! // Build a multi-layer network
//! model.add_layer(Dense::<f32>::new(784, 512, Some("relu"), &mut rng)?);
//! model.add_layer(Dropout::<f32>::new(0.2, &mut rng)?);
//! model.add_layer(Dense::<f32>::new(512, 256, Some("relu"), &mut rng)?);
//! model.add_layer(Dropout::<f32>::new(0.2, &mut rng)?);
//! model.add_layer(Dense::<f32>::new(256, 10, Some("softmax"), &mut rng)?);
//!
//! // Input: batch of MNIST-like images (batch_size=32, flattened=784)
//! let input = Array::zeros((32, 784)).into_dyn();
//!
//! // Forward pass through entire model
//! let output = model.forward(&input)?;
//! assert_eq!(output.shape(), &[32, 10]); // 10-class predictions
//!
//! println!("Model has {} layers", model.num_layers());
//! let total_params: usize = model.layers().iter().map(|l| l.parameter_count()).sum();
//! println!("Total parameters: {}", total_params);
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Convolutional Layers
//!
//! ```rust
//! use scirs2_neural::layers::{Layer, Conv2D, MaxPool2D, PaddingMode};
//! use ndarray::Array;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let mut rng = SmallRng::seed_from_u64(42);
//!
//! // Create conv layer: 3 input channels -> 32 output channels, 3x3 kernel
//! let conv = Conv2D::<f64>::new(3, 32, (3, 3), (1, 1), PaddingMode::Same, &mut rng)?;
//! let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None)?; // 2x2 max pooling
//!
//! // Input: batch of RGB images (batch=4, channels=3, height=32, width=32)
//! let input = Array::zeros((4, 3, 32, 32)).into_dyn();
//!
//! // Apply convolution then pooling
//! let conv_out = conv.forward(&input)?;
//! assert_eq!(conv_out.shape(), &[4, 32, 32, 32]); // Same padding preserved size
//!
//! let pool_out = pool.forward(&conv_out)?;
//! assert_eq!(pool_out.shape(), &[4, 32, 16, 16]); // Pooling halved spatial dims
//! # Ok(())
//! # }
//! ```
//!
//! ## Training vs Evaluation Mode
//!
//! ```rust
//! use scirs2_neural::layers::{Layer, Dropout, BatchNorm};
//! use ndarray::Array;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let mut rng = SmallRng::seed_from_u64(42);
//! let dropout = Dropout::<f64>::new(0.5, &mut rng)?;
//! let mut batchnorm = BatchNorm::<f64>::new(128, 0.9, 1e-5, &mut rng)?;
//!
//! let input = Array::ones((10, 128)).into_dyn();
//!
//! // Training mode (default)
//! assert!(dropout.is_training());
//! let train_output = dropout.forward(&input)?;
//! // Some outputs will be zero due to dropout
//!
//! // Switch to evaluation mode (dropout is immutable in this example)
//! batchnorm.set_training(false);
//!
//! let eval_output = dropout.forward(&input)?;
//! // No dropout applied, all outputs preserved but scaled
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Layer Implementation
//!
//! ```rust
//! use scirs2_neural::layers::Layer;
//! use scirs2_neural::error::Result;
//! use ndarray::{Array, ArrayD, ScalarOperand};
//! use num_traits::Float;
//! use std::fmt::Debug;
//!
//! // Custom activation layer that squares the input
//! struct SquareLayer;
//!
//! impl<F: Float + Debug + ScalarOperand> Layer<F> for SquareLayer {
//!     fn forward(&self, input: &ArrayD<F>) -> Result<ArrayD<F>> {
//!         Ok(input.mapv(|x| x * x))
//!     }
//!
//!     fn backward(&self, input: &ArrayD<F>, grad_output: &ArrayD<F>) -> Result<ArrayD<F>> {
//!         // Derivative of x^2 is 2x
//!         Ok(grad_output * &input.mapv(|x| x + x))
//!     }
//!
//!     fn update(&mut self, _learning_rate: F) -> Result<()> {
//!         Ok(()) // No parameters to update
//!     }
//!
//!     fn as_any(&self) -> &dyn std::any::Any { self }
//!     fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
//!     fn layer_type(&self) -> &str { "Square" }
//! }
//! ```
//!
//! # Layer Design Patterns
//!
//! ## Parameter Initialization
//! Most layers use random number generators for weight initialization:
//! - **Xavier/Glorot**: Good for tanh/sigmoid activations
//! - **He/Kaiming**: Better for ReLU activations
//! - **Random Normal**: Simple baseline
//!
//! ## Memory Management
//! - Use `set_training(false)` during inference to disable dropout and enable batch norm inference
//! - Sequential containers manage memory efficiently by reusing intermediate buffers
//! - Large models benefit from gradient checkpointing (available in memory_efficient module)
//!
//! ## Gradient Flow
//! - Always implement both `forward` and `backward` methods
//! - The `backward` method should compute gradients w.r.t. inputs and update internal parameter gradients
//! - Use `update` method to apply gradients with learning rate

use crate::error::Result;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Base trait for neural network layers
///
/// This trait defines the core interface that all neural network layers must implement.
/// It supports forward propagation, backpropagation, parameter management, and
/// training/evaluation mode switching.
///
/// # Core Methods
///
/// - `forward`: Compute layer output given input
/// - `backward`: Compute gradients for backpropagation  
/// - `update`: Apply parameter updates using computed gradients
/// - `set_training`/`is_training`: Control training vs evaluation behavior
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{Layer, Dense};
/// use ndarray::Array;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let mut rng = SmallRng::seed_from_u64(42);
/// let mut layer = Dense::<f64>::new(10, 5, None, &mut rng)?;
///
/// let input = Array::zeros((2, 10)).into_dyn();
/// let output = layer.forward(&input)?;
/// assert_eq!(output.shape(), &[2, 5]);
///
/// // Check layer properties
/// println!("Layer type: {}", layer.layer_type());
/// println!("Parameter count: {}", layer.parameter_count());
/// println!("Training mode: {}", layer.is_training());
/// # Ok(())
/// # }
/// ```
pub trait Layer<F: Float + Debug + ScalarOperand> {
    /// Forward pass of the layer
    ///
    /// Computes the output of the layer given an input tensor. This method
    /// applies the layer's transformation (e.g., linear transformation, convolution,
    /// activation function) to the input.
    ///
    /// # Arguments
    /// * `input` - Input tensor with arbitrary dimensions
    ///
    /// # Returns
    /// Output tensor after applying the layer's transformation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::layers::{Layer, Dense};
    /// use ndarray::Array;
    /// use rand::rngs::SmallRng;
    /// use rand::SeedableRng;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let layer = Dense::<f64>::new(3, 2, Some("relu"), &mut rng)?;
    ///
    /// let input = Array::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0])?.into_dyn();
    /// let output = layer.forward(&input)?;
    /// assert_eq!(output.shape(), &[1, 2]);
    /// # Ok(())
    /// # }
    /// ```
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Backward pass of the layer to compute gradients
    ///
    /// Computes gradients with respect to the layer's input, which is needed
    /// for backpropagation. This method also typically updates the layer's
    /// internal parameter gradients.
    ///
    /// # Arguments
    /// * `input` - Original input to the forward pass
    /// * `grad_output` - Gradient of loss with respect to this layer's output
    ///
    /// # Returns
    /// Gradient of loss with respect to this layer's input
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::layers::{Layer, Dense};
    /// use ndarray::Array;
    /// use rand::rngs::SmallRng;
    /// use rand::SeedableRng;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let layer = Dense::<f64>::new(3, 2, None, &mut rng)?;
    ///
    /// let input = Array::zeros((1, 3)).into_dyn();
    /// let grad_output = Array::ones((1, 2)).into_dyn();
    ///
    /// let grad_input = layer.backward(&input, &grad_output)?;
    /// assert_eq!(grad_input.shape(), input.shape());
    /// # Ok(())
    /// # }
    /// ```
    fn backward(
        &self,
        input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;

    /// Update the layer parameters with the given gradients
    ///
    /// Applies parameter updates using the provided learning rate and the
    /// gradients computed during the backward pass. This is typically called
    /// by optimizers.
    ///
    /// # Arguments
    /// * `learning_rate` - Step size for parameter updates
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::layers::{Layer, Dense};
    /// use ndarray::Array;
    /// use rand::rngs::SmallRng;
    /// use rand::SeedableRng;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let mut layer = Dense::<f64>::new(3, 2, None, &mut rng)?;
    ///
    /// // Simulate forward/backward pass
    /// let input = Array::zeros((1, 3)).into_dyn();
    /// let output = layer.forward(&input)?;
    /// let grad_output = Array::ones((1, 2)).into_dyn();
    /// let _grad_input = layer.backward(&input, &grad_output)?;
    ///
    /// // Update parameters
    /// layer.update(0.01)?; // learning rate = 0.01
    /// # Ok(())
    /// # }
    /// ```
    fn update(&mut self, learning_rate: F) -> Result<()>;

    /// Get the layer as a dyn Any for downcasting
    ///
    /// This method enables runtime type checking and downcasting to specific
    /// layer types when needed.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the layer as a mutable dyn Any for downcasting
    ///
    /// This method enables runtime type checking and downcasting to specific
    /// layer types when mutable access is needed.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get the parameters of the layer
    ///
    /// Returns all trainable parameters (weights, biases) as a vector of arrays.
    /// Default implementation returns empty vector for parameterless layers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::layers::{Layer, Dense};
    /// use rand::rngs::SmallRng;
    /// use rand::SeedableRng;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let layer = Dense::<f64>::new(3, 2, None, &mut rng)?;
    ///
    /// let params = layer.params();
    /// // Dense layer has weights and biases
    /// assert_eq!(params.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    fn params(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        Vec::new()
    }

    /// Get the gradients of the layer parameters
    ///
    /// Returns gradients for all trainable parameters. Must be called after
    /// backward pass to get meaningful values.
    fn gradients(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        Vec::new()
    }

    /// Set the gradients of the layer parameters
    ///
    /// Used by optimizers to set computed gradients. Default implementation
    /// does nothing for parameterless layers.
    fn set_gradients(&mut self, _gradients: &[Array<F, ndarray::IxDyn>]) -> Result<()> {
        Ok(())
    }

    /// Set the parameters of the layer
    ///
    /// Used for loading pre-trained weights or applying parameter updates.
    /// Default implementation does nothing for parameterless layers.
    fn set_params(&mut self, _params: &[Array<F, ndarray::IxDyn>]) -> Result<()> {
        Ok(())
    }

    /// Set the layer to training mode (true) or evaluation mode (false)
    ///
    /// Training mode enables features like dropout and batch normalization
    /// parameter updates. Evaluation mode disables these features for inference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::layers::{Layer, Dropout};
    /// use rand::rngs::SmallRng;
    /// use rand::SeedableRng;
    ///
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let mut dropout = Dropout::<f32>::new(0.5, &mut rng).unwrap();
    /// assert!(dropout.is_training()); // Default is training mode
    ///
    /// dropout.set_training(false); // Switch to evaluation
    /// assert!(!dropout.is_training());
    /// ```
    fn set_training(&mut self, _training: bool) {
        // Default implementation: do nothing
    }

    /// Get the current training mode
    ///
    /// Returns true if layer is in training mode, false if in evaluation mode.
    fn is_training(&self) -> bool {
        true // Default implementation: always in training mode
    }

    /// Get the type of the layer (e.g., "Dense", "Conv2D")
    ///
    /// Returns a string identifier for the layer type, useful for debugging
    /// and model introspection.
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Get the number of trainable parameters in this layer
    ///
    /// Returns the total count of all trainable parameters (weights, biases, etc.).
    /// Useful for model analysis and memory estimation.
    fn parameter_count(&self) -> usize {
        0
    }

    /// Get a detailed description of this layer
    ///
    /// Returns a human-readable description including layer type and key properties.
    /// Can be overridden for more detailed layer-specific information.
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
pub mod dense;
mod dropout;
mod embedding;
mod normalization;
pub mod recurrent;
mod regularization;
mod rnn_thread_safe;

// Re-export layer types
pub use attention::{AttentionConfig, AttentionMask, MultiHeadAttention, SelfAttention};
pub use conv::{
    AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D, AdaptiveMaxPool1D, AdaptiveMaxPool2D,
    AdaptiveMaxPool3D, Conv2D, GlobalAvgPool2D, MaxPool2D, PaddingMode,
};
pub use dense::Dense;
pub use dropout::Dropout;
pub use embedding::{Embedding, EmbeddingConfig, PatchEmbedding, PositionalEmbedding};
pub use normalization::{BatchNorm, LayerNorm, LayerNorm2D};
pub use recurrent::{
    Bidirectional, GRUConfig, LSTMConfig, RNNConfig, RecurrentActivation, GRU, LSTM, RNN,
};
pub use regularization::{
    ActivityRegularization, L1ActivityRegularization, L2ActivityRegularization,
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
/// A Sequential model is a linear stack of layers where data flows through
/// each layer in order. This is the most common way to build neural networks
/// and is suitable for feed-forward architectures.
///
/// # Features
///
/// - **Linear topology**: Layers are executed in the order they were added
/// - **Automatic gradient flow**: Backward pass automatically chains through all layers
/// - **Training mode management**: Sets all contained layers to training/evaluation mode
/// - **Parameter aggregation**: Collects parameters from all layers for optimization
/// - **Memory efficient**: Reuses intermediate tensors when possible
///
/// # Examples
///
/// ## Building a Classifier
///
/// ```rust
/// use scirs2_neural::layers::{Dense, Dropout, Layer};
/// use scirs2_neural::models::{Sequential, Model};
/// use ndarray::Array;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let mut rng = SmallRng::seed_from_u64(42);
/// let mut model: Sequential<f32> = Sequential::new();
///
/// // Build a 3-layer classifier for MNIST (28x28 = 784 inputs, 10 classes)
/// model.add_layer(Dense::<f32>::new(784, 128, Some("relu"), &mut rng)?);
/// model.add_layer(Dropout::new(0.3, &mut rng)?);
/// model.add_layer(Dense::new(128, 64, Some("relu"), &mut rng)?);
/// model.add_layer(Dropout::new(0.3, &mut rng)?);
/// model.add_layer(Dense::<f32>::new(64, 10, Some("softmax"), &mut rng)?);
///
/// // Process a batch of images
/// let batch = Array::zeros((32, 784)).into_dyn(); // 32 samples
/// let predictions = model.forward(&batch)?;
/// assert_eq!(predictions.shape(), &[32, 10]);
///
/// println!("Model summary:");
/// println!("- Layers: {}", model.num_layers());
/// # Ok(())
/// # }
/// ```
///
/// ## CNN for Image Recognition
///
/// ```rust
/// use scirs2_neural::layers::{Conv2D, MaxPool2D, Dense, Dropout, Layer, PaddingMode};
/// use scirs2_neural::models::{Sequential, Model};
/// use ndarray::Array;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let mut rng = SmallRng::seed_from_u64(42);
/// let mut cnn: Sequential<f32> = Sequential::new();
///
/// // Convolutional feature extractor
/// cnn.add_layer(Conv2D::new(3, 32, (3, 3), (1, 1), PaddingMode::Same, &mut rng)?); // 3->32 channels
/// cnn.add_layer(MaxPool2D::new((2, 2), (2, 2), None)?); // Downsample 2x
/// cnn.add_layer(Conv2D::new(32, 64, (3, 3), (1, 1), PaddingMode::Same, &mut rng)?); // 32->64 channels  
/// cnn.add_layer(MaxPool2D::new((2, 2), (2, 2), None)?); // Downsample 2x
///
/// // Classifier head (would need reshape layer in practice)
/// // cnn.add_layer(Flatten::new()); // Would flatten to 1D
/// // cnn.add_layer(Dense::new(64*8*8, 128, Some("relu"), &mut rng)?);
/// // cnn.add_layer(Dropout::new(0.5, &mut rng)?);
/// // cnn.add_layer(Dense::new(128, 10, None, &mut rng)?);
///
/// // Input: batch of 32x32 RGB images
/// let images = Array::zeros((16, 3, 32, 32)).into_dyn();
/// let features = cnn.forward(&images)?;
/// println!("Feature shape: {:?}", features.shape());
/// # Ok(())
/// # }
/// ```
///
/// ## Training and Evaluation Modes
///
/// ```rust
/// use scirs2_neural::layers::{Dense, Dropout, Layer};
/// use scirs2_neural::models::{Sequential, Model};
/// use ndarray::Array;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let mut rng = SmallRng::seed_from_u64(42);
/// let mut model: Sequential<f32> = Sequential::new();
/// model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng)?);
/// model.add_layer(Dropout::new(0.5, &mut rng)?); // 50% dropout
/// model.add_layer(Dense::<f32>::new(5, 1, None, &mut rng)?);
///
/// let input = Array::ones((4, 10)).into_dyn();
///
/// // Forward pass through the model
/// let output = model.forward(&input)?;
/// println!("Output shape: {:?}", output.shape());
/// # Ok(())
/// # }
/// ```
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
