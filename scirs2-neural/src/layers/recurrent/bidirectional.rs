//! Bidirectional wrapper for recurrent layers

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{concatenate, Array, Axis, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Bidirectional RNN wrapper for recurrent layers
///
/// This layer wraps a recurrent layer to enable bidirectional processing.
/// It processes the input sequence in both forward and backward directions,
/// and concatenates the results.
///
/// # Examples
/// ```
/// use scirs2_neural::layers::{Layer, recurrent::{Bidirectional, RNN, rnn::RecurrentActivation}};
/// use scirs2_core::ndarray::{Array, Array3};
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
///
/// // Create RNN layers for forward and backward directions
/// let mut rng = scirs2_core::random::rng();
/// let forward_rnn = RNN::new(10, 20, RecurrentActivation::Tanh, &mut rng).expect("Operation failed");
/// let backward_rnn = RNN::new(10, 20, RecurrentActivation::Tanh, &mut rng).expect("Operation failed");
///
/// // Wrap them in a bidirectional layer
/// let birnn = Bidirectional::new(Box::new(forward_rnn), Some(Box::new(backward_rnn)), None).expect("Operation failed");
///
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = birnn.forward(&input).expect("Operation failed");
///
/// // Output should have dimensions [batch_size, seq_len, hidden_size*2]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 40]);
/// ```
pub struct Bidirectional<F: Float + Debug + Send + Sync + NumAssign> {
    /// Forward direction layer
    forward_layer: Box<dyn Layer<F> + Send + Sync>,
    /// Backward direction layer, with its own independent parameters
    backward_layer: Box<dyn Layer<F> + Send + Sync>,
    /// Name for the layer
    name: Option<String>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Bidirectional<F> {
    /// Create a new bidirectional wrapper
    ///
    /// # Arguments
    /// * `forward_layer` - The recurrent layer to use in forward direction
    /// * `backward_layer` - The recurrent layer for the backward direction
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    /// * A new bidirectional layer
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArchitecture`] when `backward_layer` is
    /// `None`. A bidirectional RNN needs an independent parameter set per
    /// direction: reusing one layer for both would make the two directions
    /// share weights, and the shared layer's forward cache and gradient
    /// accumulator can only hold one direction at a time, so the reported
    /// gradients would silently be wrong.
    pub fn new(
        forward_layer: Box<dyn Layer<F> + Send + Sync>,
        backward_layer: Option<Box<dyn Layer<F> + Send + Sync>>,
        name: Option<&str>,
    ) -> Result<Self> {
        let backward_layer = backward_layer.ok_or_else(|| {
            NeuralError::InvalidArchitecture(
                "Bidirectional requires a dedicated backward-direction layer; pass a second \
                 recurrent layer with its own parameters instead of None"
                    .to_string(),
            )
        })?;
        Ok(Self {
            forward_layer,
            backward_layer,
            name: name.map(String::from),
            input_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for Bidirectional<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        *self.input_cache.write().expect("Operation failed") = Some(input.clone());

        // Check input dimensions
        let inputshape = input.shape();
        if inputshape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input [batch_size, seq_len, input_size], got {inputshape:?}"
            )));
        }
        let _batch_size = inputshape[0];
        let seq_len = inputshape[1];

        // Forward direction
        let forward_output = self.forward_layer.forward(input)?;

        // Process backward direction
        let backward_layer = &self.backward_layer;

        // Reverse the sequence dimension of input
        // Create views for each time step and reverse their order
        let mut reversed_slices = Vec::new();
        for t in (0..seq_len).rev() {
            let slice = input.slice(scirs2_core::ndarray::s![.., t..t + 1, ..]);
            reversed_slices.push(slice);
        }

        // Concatenate the reversed slices along the time dimension
        let views: Vec<_> = reversed_slices.iter().map(|s| s.view()).collect();
        let reversed_input = concatenate(Axis(1), &views)?.into_dyn();

        // Process through backward layer
        let backward_output = backward_layer.forward(&reversed_input)?;

        // Reverse the backward output to align with forward output
        let mut backward_reversed_slices = Vec::new();
        for t in (0..seq_len).rev() {
            let slice = backward_output.slice(scirs2_core::ndarray::s![.., t..t + 1, ..]);
            backward_reversed_slices.push(slice);
        }
        let backward_views: Vec<_> = backward_reversed_slices.iter().map(|s| s.view()).collect();
        let backward_output_aligned = concatenate(Axis(1), &backward_views)?.into_dyn();

        // Concatenate forward and backward outputs along the feature dimension
        let forward_view = forward_output.view();
        let backward_view = backward_output_aligned.view();
        let output = concatenate(Axis(2), &[forward_view, backward_view])?.into_dyn();
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached _input
        let input_ref = self.input_cache.read().expect("Operation failed");
        if input_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached _input for backward pass. Call forward() first.".to_string(),
            ));
        }
        let cached_input = input_ref.as_ref().expect("Operation failed");

        // Check gradient dimensions
        let gradshape = grad_output.shape();
        if gradshape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D gradient [batch_size, seq_len, hidden_size*2], got {gradshape:?}"
            )));
        }
        let _batch_size = gradshape[0];
        let seq_len = gradshape[1];
        let total_hidden = gradshape[2];

        // Get the backward layer
        let backward_layer = &self.backward_layer;

        // Split gradient into forward and backward components
        let hidden_size = total_hidden / 2;
        let grad_forward = grad_output
            .slice(scirs2_core::ndarray::s![.., .., ..hidden_size])
            .to_owned()
            .into_dyn();
        let grad_backward = grad_output
            .slice(scirs2_core::ndarray::s![.., .., hidden_size..])
            .to_owned()
            .into_dyn();

        // Backward pass through forward layer
        let grad_input_forward = self.forward_layer.backward(cached_input, &grad_forward)?;

        // For backward layer, we need to reverse the gradient and _input
        // Reverse the gradient for backward layer
        let mut backward_grad_slices = Vec::new();
        for t in (0..seq_len).rev() {
            let slice = grad_backward.slice(scirs2_core::ndarray::s![.., t..t + 1, ..]);
            backward_grad_slices.push(slice);
        }
        let backward_grad_views: Vec<_> = backward_grad_slices.iter().map(|s| s.view()).collect();
        let grad_backward_reversed = concatenate(Axis(1), &backward_grad_views)?.into_dyn();

        // Reverse the _input for backward layer
        let mut input_slices = Vec::new();
        for t in (0..seq_len).rev() {
            let slice = cached_input.slice(scirs2_core::ndarray::s![.., t..t + 1, ..]);
            input_slices.push(slice);
        }
        let input_views: Vec<_> = input_slices.iter().map(|s| s.view()).collect();
        let input_reversed = concatenate(Axis(1), &input_views)?.into_dyn();

        // Backward pass through backward layer
        let grad_input_backward_reversed =
            backward_layer.backward(&input_reversed, &grad_backward_reversed)?;

        // Reverse the backward gradient back to original order
        let mut final_backward_slices = Vec::new();
        for t in (0..seq_len).rev() {
            let slice =
                grad_input_backward_reversed.slice(scirs2_core::ndarray::s![.., t..t + 1, ..]);
            final_backward_slices.push(slice);
        }
        let final_backward_views: Vec<_> = final_backward_slices.iter().map(|s| s.view()).collect();
        let grad_input_backward = concatenate(Axis(1), &final_backward_views)?.into_dyn();

        // Sum the gradients from forward and backward paths
        let grad_input = grad_input_forward + grad_input_backward;
        Ok(grad_input)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        self.forward_layer.update(learningrate)?;
        self.backward_layer.update(learningrate)?;
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = self.forward_layer.params();
        params.extend(self.backward_layer.params());
        params
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let mut grads = self.forward_layer.gradients();
        grads.extend(self.backward_layer.gradients());
        grads
    }

    fn layer_type(&self) -> &str {
        "Bidirectional"
    }

    fn parameter_count(&self) -> usize {
        self.forward_layer.parameter_count() + self.backward_layer.parameter_count()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
