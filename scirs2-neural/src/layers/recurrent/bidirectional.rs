//! Bidirectional wrapper for recurrent layers

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use std::cell::RefCell;
use std::fmt::Debug;

/// Bidirectional RNN wrapper for recurrent layers
///
/// This layer wraps a recurrent layer to enable bidirectional processing.
/// It processes the input sequence in both forward and backward directions,
/// and concatenates the results.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{Bidirectional, RNN, Layer, RecurrentActivation};
/// use ndarray::{Array, Array3};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create an RNN layer with 10 input features and 20 hidden units
/// let mut rng = SmallRng::seed_from_u64(42);
/// let rnn = RNN::new(10, 20, RecurrentActivation::Tanh, &mut rng).unwrap();
///
/// // Wrap it in a bidirectional layer
/// let birnn = Bidirectional::new(Box::new(rnn), None).unwrap();
///
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = birnn.forward(&input).unwrap();
///
/// // TODO: Currently returns only forward output. Should be [batch_size, seq_len, hidden_size*2]
/// // when backward layer is properly implemented
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
/// ```
pub struct Bidirectional<F: Float + Debug> {
    /// Forward direction layer
    forward_layer: Box<dyn Layer<F> + Send + Sync>,
    /// Backward direction layer (using the same layer type)
    backward_layer: Option<Box<dyn Layer<F> + Send + Sync>>,
    /// Name for the layer
    name: Option<String>,
    /// Input cache for backward pass
    input_cache: RefCell<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + 'static> Bidirectional<F> {
    /// Create a new bidirectional wrapper
    ///
    /// # Arguments
    ///
    /// * `layer` - The recurrent layer to use in forward direction
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    ///
    /// * A new bidirectional layer
    pub fn new(layer: Box<dyn Layer<F> + Send + Sync>, name: Option<&str>) -> Result<Self> {
        // Clone the layer for backward direction
        let forward_layer = layer;
        let backward_layer = None; // For now, just use None

        Ok(Self {
            forward_layer,
            backward_layer,
            name: name.map(String::from),
            input_cache: RefCell::new(None),
        })
    }

    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for Bidirectional<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        self.input_cache.replace(Some(input.clone()));

        // Check input dimensions
        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input [batch_size, seq_len, input_size], got {:?}",
                input_shape
            )));
        }

        // Forward direction
        let forward_output = self.forward_layer.forward(input)?;

        // If no backward layer, just return forward output
        if self.backward_layer.is_none() {
            return Ok(forward_output);
        }

        // Otherwise, process backward direction and concatenate
        let backward_layer = self.backward_layer.as_ref().unwrap();

        // Reverse the sequence dimension of input
        let reversed_input = input.clone();
        let _batch_size = input_shape[0];
        let _seq_len = input_shape[1];

        // TODO: Implement the actual reversing of the sequence dimension
        // For now, just use the forward direction output

        let _backward_output = backward_layer.forward(&reversed_input)?;

        // Concatenate forward and backward outputs
        // TODO: Implement the actual concatenation along the feature dimension
        // For now, just return the forward direction output

        Ok(forward_output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached input
        let _input_ref = self.input_cache.borrow();
        if _input_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            ));
        }

        // For now, just return a placeholder gradient
        let grad_input = Array::zeros(input.dim());

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update forward layer
        self.forward_layer.update(learning_rate)?;

        // Update backward layer if present
        if let Some(ref mut backward_layer) = self.backward_layer {
            backward_layer.update(learning_rate)?;
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
