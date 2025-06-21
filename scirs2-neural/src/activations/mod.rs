//! Activation functions for neural networks
//!
//! This module provides common activation functions used in neural networks.
//! Activation functions introduce non-linearity into neural networks, enabling
//! them to learn complex patterns and relationships.
//!
//! # Overview
//!
//! Activation functions are mathematical functions that determine whether a neuron
//! should be activated or not based on the input. They introduce non-linearity
//! to the network, allowing it to learn complex mappings between inputs and outputs.
//!
//! # Available Activation Functions
//!
//! - **ReLU** (Rectified Linear Unit): Most commonly used, simple and effective
//! - **Sigmoid**: Maps input to (0,1), useful for binary classification output layers
//! - **Tanh**: Maps input to (-1,1), often better than sigmoid for hidden layers
//! - **Softmax**: Converts logits to probability distribution, used in multi-class classification
//! - **GELU** (Gaussian Error Linear Unit): Smooth alternative to ReLU, used in transformers
//! - **Swish/SiLU**: Self-gated activation, often outperforms ReLU
//! - **Mish**: Smooth, non-monotonic activation function
//! - **Leaky ReLU**: Variant of ReLU that allows small negative values
//! - **ELU** (Exponential Linear Unit): Smooth variant of ReLU
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use scirs2_neural::activations::{Activation, ReLU, Sigmoid, Softmax};
//! use ndarray::Array;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! // Create activation functions
//! let relu = ReLU::new();
//! let sigmoid = Sigmoid::new();
//! let softmax = Softmax::new(0); // Apply softmax along axis 0
//!
//! // Create input data
//! let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0])
//!     .into_dyn();
//!
//! // Apply ReLU activation
//! let relu_output = relu.forward(&input)?;
//! // Output: [0.0, 0.0, 0.0, 1.0, 2.0]
//!
//! // Apply Sigmoid activation
//! let sigmoid_output = sigmoid.forward(&input)?;
//! // Output: [0.119, 0.269, 0.5, 0.731, 0.881] (approximately)
//!
//! // For softmax, typically used with 2D input (batch_size, num_classes)
//! let logits = Array::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0])?.into_dyn();
//! let probabilities = softmax.forward(&logits)?;
//! // Output: [[0.090, 0.245, 0.665]] (approximately, sums to 1.0)
//! # Ok(())
//! # }
//! ```
//!
//! ## Using in Forward and Backward Pass
//!
//! ```rust
//! use scirs2_neural::activations::{Activation, ReLU};
//! use ndarray::Array;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let relu = ReLU::new();
//! let input = Array::from_vec(vec![-1.0, 0.5, 2.0]).into_dyn();
//!
//! // Forward pass
//! let output = relu.forward(&input)?;
//! println!("ReLU output: {:?}", output);
//! // Output: [0.0, 0.5, 2.0]
//!
//! // Backward pass (computing gradients)
//! let grad_output = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();
//! let grad_input = relu.backward(&grad_output, &output)?;
//! println!("ReLU gradient: {:?}", grad_input);
//! // Output: [0.0, 1.0, 1.0] (gradient is 0 for negative inputs, 1 for positive)
//! # Ok(())
//! # }
//! ```
//!
//! ## Choosing the Right Activation Function
//!
//! ### For Hidden Layers:
//! - **ReLU**: Default choice, computationally efficient, prevents vanishing gradient
//! - **GELU**: Good for transformer architectures
//! - **Swish**: Often outperforms ReLU, especially in deep networks
//! - **Tanh**: When you need outputs centered around zero
//!
//! ### For Output Layers:
//! - **Sigmoid**: Binary classification (single output)
//! - **Softmax**: Multi-class classification (multiple outputs that sum to 1)
//! - **Linear (no activation)**: Regression tasks
//! - **Tanh**: When output should be in range (-1, 1)
//!
//! # Performance Considerations
//!
//! - **ReLU** and **Leaky ReLU**: Fastest to compute
//! - **Sigmoid** and **Tanh**: Require expensive exponential operations
//! - **Softmax**: Most expensive, but only used in output layer typically
//! - **GELU** and **Swish**: More expensive than ReLU but can provide better results

use crate::error::Result;
use ndarray::Array;
use num_traits::Float;

/// Trait for activation functions
///
/// This trait defines the interface for all activation functions in the neural network.
/// Activation functions must implement both forward and backward pass methods to support
/// automatic differentiation during training.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::activations::{Activation, ReLU};
/// use ndarray::Array;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let activation = ReLU::new();
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
///
/// // Forward pass
/// let output = activation.forward(&input)?;
/// assert_eq!(output.as_slice().unwrap(), &[0.0, 0.0, 1.0]);
///
/// // Backward pass
/// let grad_output = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();
/// let grad_input = activation.backward(&grad_output, &output)?;
/// assert_eq!(grad_input.as_slice().unwrap(), &[0.0, 0.0, 1.0]);
/// # Ok(())
/// # }
/// ```
pub trait Activation<F: Float> {
    /// Apply the activation function to the input
    ///
    /// # Arguments
    /// * `input` - Input tensor of arbitrary dimensionality
    ///
    /// # Returns
    /// The activated output tensor with the same shape as the input
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::activations::{Activation, Sigmoid};
    /// use ndarray::Array;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let sigmoid = Sigmoid::new();
    /// let input = Array::from_vec(vec![0.0]).into_dyn();
    /// let output = sigmoid.forward(&input)?;
    /// // Sigmoid(0) = 0.5
    /// assert!((output[ndarray::IxDyn(&[0])] - 0.5f64).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Compute the derivative of the activation function with respect to the input
    ///
    /// This method computes the gradient of the activation function, which is needed
    /// for backpropagation during training.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient from the next layer in the backpropagation chain
    /// * `output` - Output of the forward pass (used by some activations for efficiency)
    ///
    /// # Returns
    /// The gradient with respect to the input, same shape as the input
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::activations::{Activation, ReLU};
    /// use ndarray::Array;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let relu = ReLU::new();
    /// let input = Array::from_vec(vec![-1.0, 1.0]).into_dyn();
    /// let output = relu.forward(&input)?; // [0.0, 1.0]
    ///
    /// let grad_output = Array::from_vec(vec![1.0, 1.0]).into_dyn();
    /// let grad_input = relu.backward(&grad_output, &output)?;
    /// // ReLU gradient: 0 for negative inputs, 1 for positive inputs
    /// assert_eq!(grad_input.as_slice().unwrap(), &[0.0, 1.0]);
    /// # Ok(())
    /// # }
    /// ```
    fn backward(
        &self,
        grad_output: &Array<F, ndarray::IxDyn>,
        output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;
}

mod gelu;
mod mish;
mod relu;
mod sigmoid;
mod softmax;
mod swish;
mod tanh;

pub use gelu::GELU;
pub use mish::Mish;
pub use relu::{LeakyReLU, ReLU, ELU};
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use swish::Swish;
pub use tanh::Tanh;
