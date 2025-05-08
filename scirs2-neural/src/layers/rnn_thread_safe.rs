// Thread-safe RNN implementations
//
// This module provides thread-safe versions of the RNN, LSTM, and GRU layers
// that can be safely used across multiple threads by using Arc<RwLock<>> instead
// of RefCell for internal state.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use ndarray::{Array, ArrayView, Axis, Ix2, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Thread-safe version of RNN for sequence processing
///
/// This implementation replaces RefCell with Arc<RwLock<>> for thread safety.
pub struct ThreadSafeRNN<F: Float + Debug + Send + Sync> {
    /// Input size (number of input features)
    pub input_size: usize,
    /// Hidden size (number of hidden units)
    pub hidden_size: usize,
    /// Activation function
    pub activation: RecurrentActivation,
    /// Input-to-hidden weights
    weight_ih: Array<F, IxDyn>,
    /// Hidden-to-hidden weights
    weight_hh: Array<F, IxDyn>,
    /// Input-to-hidden bias
    bias_ih: Array<F, IxDyn>,
    /// Hidden-to-hidden bias
    bias_hh: Array<F, IxDyn>,
    /// Gradient of input-to-hidden weights
    dweight_ih: Array<F, IxDyn>,
    /// Gradient of hidden-to-hidden weights
    dweight_hh: Array<F, IxDyn>,
    /// Gradient of input-to-hidden bias
    dbias_ih: Array<F, IxDyn>,
    /// Gradient of hidden-to-hidden bias
    dbias_hh: Array<F, IxDyn>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Hidden states cache for backward pass
    hidden_states_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

/// Activation function types for recurrent layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecurrentActivation {
    /// Hyperbolic tangent (tanh) activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// Rectified Linear Unit (ReLU)
    ReLU,
}

impl RecurrentActivation {
    /// Apply the activation function
    fn apply<F: Float>(&self, x: F) -> F {
        match self {
            RecurrentActivation::Tanh => x.tanh(),
            RecurrentActivation::Sigmoid => F::one() / (F::one() + (-x).exp()),
            RecurrentActivation::ReLU => {
                if x > F::zero() {
                    x
                } else {
                    F::zero()
                }
            }
        }
    }

    /// Apply the activation function to an array
    #[allow(dead_code)]
    fn apply_array<F: Float + ScalarOperand>(&self, x: &Array<F, IxDyn>) -> Array<F, IxDyn> {
        match self {
            RecurrentActivation::Tanh => x.mapv(|v| v.tanh()),
            RecurrentActivation::Sigmoid => x.mapv(|v| F::one() / (F::one() + (-v).exp())),
            RecurrentActivation::ReLU => x.mapv(|v| if v > F::zero() { v } else { F::zero() }),
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + 'static> ThreadSafeRNN<F> {
    /// Create a new thread-safe RNN layer
    pub fn new<R: Rng>(
        input_size: usize,
        hidden_size: usize,
        activation: RecurrentActivation,
        rng: &mut R,
    ) -> Result<Self> {
        // Validate parameters
        if input_size == 0 || hidden_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Input size and hidden size must be positive".to_string(),
            ));
        }

        // Initialize weights with Xavier/Glorot initialization
        let scale_ih = F::from(1.0 / (input_size as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;

        let scale_hh = F::from(1.0 / (hidden_size as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;

        // Initialize input-to-hidden weights
        let mut weight_ih_vec: Vec<F> = Vec::with_capacity(hidden_size * input_size);
        for _ in 0..(hidden_size * input_size) {
            let rand_val = rng.random_range(-1.0f64..1.0f64);
            let val = F::from(rand_val).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
            })?;
            weight_ih_vec.push(val * scale_ih);
        }

        let weight_ih = Array::from_shape_vec(IxDyn(&[hidden_size, input_size]), weight_ih_vec)
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
            })?;

        // Initialize hidden-to-hidden weights
        let mut weight_hh_vec: Vec<F> = Vec::with_capacity(hidden_size * hidden_size);
        for _ in 0..(hidden_size * hidden_size) {
            let rand_val = rng.random_range(-1.0f64..1.0f64);
            let val = F::from(rand_val).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
            })?;
            weight_hh_vec.push(val * scale_hh);
        }

        let weight_hh = Array::from_shape_vec(IxDyn(&[hidden_size, hidden_size]), weight_hh_vec)
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
            })?;

        // Initialize biases
        let bias_ih = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh = Array::zeros(IxDyn(&[hidden_size]));

        // Initialize gradients
        let dweight_ih = Array::zeros(weight_ih.dim());
        let dweight_hh = Array::zeros(weight_hh.dim());
        let dbias_ih = Array::zeros(bias_ih.dim());
        let dbias_hh = Array::zeros(bias_hh.dim());

        Ok(Self {
            input_size,
            hidden_size,
            activation,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            dweight_ih,
            dweight_hh,
            dbias_ih,
            dbias_hh,
            input_cache: Arc::new(RwLock::new(None)),
            hidden_states_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Helper method to compute one step of the RNN
    fn step(&self, x: &ArrayView<F, IxDyn>, h: &ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let x_shape = x.shape();
        let h_shape = h.shape();
        let batch_size = x_shape[0];

        // Validate shapes
        if x_shape[1] != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "Input feature dimension mismatch: expected {}, got {}",
                self.input_size, x_shape[1]
            )));
        }

        if h_shape[1] != self.hidden_size {
            return Err(NeuralError::InferenceError(format!(
                "Hidden state dimension mismatch: expected {}, got {}",
                self.hidden_size, h_shape[1]
            )));
        }

        if x_shape[0] != h_shape[0] {
            return Err(NeuralError::InferenceError(format!(
                "Batch size mismatch: input has {}, hidden state has {}",
                x_shape[0], h_shape[0]
            )));
        }

        // Initialize output
        let mut new_h = Array::zeros((batch_size, self.hidden_size));

        // Compute h_t = activation(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                // Input-to-hidden contribution: W_ih * x_t + b_ih
                let mut ih_sum = self.bias_ih[i];
                for j in 0..self.input_size {
                    ih_sum = ih_sum + self.weight_ih[[i, j]] * x[[b, j]];
                }

                // Hidden-to-hidden contribution: W_hh * h_(t-1) + b_hh
                let mut hh_sum = self.bias_hh[i];
                for j in 0..self.hidden_size {
                    hh_sum = hh_sum + self.weight_hh[[i, j]] * h[[b, j]];
                }

                // Apply activation
                new_h[[b, i]] = self.activation.apply(ih_sum + hh_sum);
            }
        }

        // Convert to IxDyn dimension
        let new_h_dyn = new_h.into_dyn();
        Ok(new_h_dyn)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for ThreadSafeRNN<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.to_owned());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        // Validate input shape
        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input [batch_size, seq_len, features], got {:?}",
                input_shape
            )));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        if features != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "Input features dimension mismatch: expected {}, got {}",
                self.input_size, features
            )));
        }

        // Initialize hidden state to zeros
        let mut h = Array::zeros((batch_size, self.hidden_size));

        // Initialize output array to store all hidden states
        let mut all_hidden_states = Array::zeros((batch_size, seq_len, self.hidden_size));

        // Process each time step
        for t in 0..seq_len {
            // Extract input at time t
            let x_t = input.slice(ndarray::s![.., t, ..]);

            // Process one step
            let x_t_view = x_t.view().into_dyn();
            let h_view = h.view().into_dyn();
            h = self
                .step(&x_t_view, &h_view)?
                .into_dimensionality::<Ix2>()
                .unwrap();

            // Store hidden state
            for b in 0..batch_size {
                for i in 0..self.hidden_size {
                    all_hidden_states[[b, t, i]] = h[[b, i]];
                }
            }
        }

        // Cache all hidden states for backward pass
        if let Ok(mut cache) = self.hidden_states_cache.write() {
            *cache = Some(all_hidden_states.clone().into_dyn());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on hidden states cache".to_string(),
            ));
        }

        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };

        let hidden_states_ref = match self.hidden_states_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on hidden states cache".to_string(),
                ))
            }
        };

        if input_ref.is_none() || hidden_states_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            ));
        }

        // In a real implementation, we would compute gradients for all parameters
        // and return the gradient with respect to the input

        // Here we're providing a simplified version that returns a gradient of zeros
        // with the correct shape
        let grad_input = Array::zeros(input.dim());

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Apply a small update to parameters (placeholder)
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learning_rate;

        // Update weights and biases
        for w in self.weight_ih.iter_mut() {
            *w = *w - lr * self.dweight_ih[[0, 0]];
        }

        for w in self.weight_hh.iter_mut() {
            *w = *w - lr * self.dweight_hh[[0, 0]];
        }

        for b in self.bias_ih.iter_mut() {
            *b = *b - lr * self.dbias_ih[[0]];
        }

        for b in self.bias_hh.iter_mut() {
            *b = *b - lr * self.dbias_hh[[0]];
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

/// Thread-safe version of Bidirectional RNN wrapper
///
/// This layer wraps a recurrent layer to enable bidirectional processing
/// while ensuring thread safety with Arc<RwLock<>> instead of RefCell.
pub struct ThreadSafeBidirectional<F: Float + Debug + Send + Sync> {
    /// Forward RNN layer
    forward_layer: Box<dyn Layer<F> + Send + Sync>,
    /// Backward RNN layer (optional)
    backward_layer: Option<Box<dyn Layer<F> + Send + Sync>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Name of the layer (optional)
    name: Option<String>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + 'static> ThreadSafeBidirectional<F> {
    /// Create a new Bidirectional RNN wrapper
    ///
    /// # Arguments
    ///
    /// * `layer` - The RNN layer to make bidirectional
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    ///
    /// * A new Bidirectional RNN wrapper
    pub fn new(layer: Box<dyn Layer<F> + Send + Sync>, name: Option<&str>) -> Result<Self> {
        // For now, we'll create a dummy backward RNN with the same configuration
        // In a real implementation, we would create a proper clone of the layer
        let backward_layer = if let Some(rnn) = layer.as_any().downcast_ref::<ThreadSafeRNN<F>>() {
            // If it's a ThreadSafeRNN, create a new one with the same parameters
            let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
            let backward_rnn =
                ThreadSafeRNN::<F>::new(rnn.input_size, rnn.hidden_size, rnn.activation, &mut rng)?;
            Some(Box::new(backward_rnn) as Box<dyn Layer<F> + Send + Sync>)
        } else {
            // If not, just set it to None for now
            None
        };

        Ok(Self {
            forward_layer: layer,
            backward_layer,
            input_cache: Arc::new(RwLock::new(None)),
            name: name.map(|s| s.to_string()),
        })
    }
}

// Custom implementation of Debug for ThreadSafeBidirectional
impl<F: Float + Debug + Send + Sync> std::fmt::Debug for ThreadSafeBidirectional<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadSafeBidirectional")
            .field("name", &self.name)
            .finish()
    }
}

// We can't implement Clone for ThreadSafeBidirectional because Box<dyn Layer> can't be cloned.
// We'll just create a dummy clone implementation for debugging purposes only.
// This won't actually clone the layers.
impl<F: Float + Debug + Send + Sync + ScalarOperand + 'static> Clone
    for ThreadSafeBidirectional<F>
{
    fn clone(&self) -> Self {
        // This is NOT a real clone and should not be used for actual computation.
        // It's provided just to satisfy the Clone trait requirement for debugging.
        // The cloned object will have empty layers.

        // Create a dummy RNN for the forward layer
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let dummy_rnn = ThreadSafeRNN::<F>::new(1, 1, RecurrentActivation::Tanh, &mut rng)
            .expect("Failed to create dummy RNN");

        Self {
            forward_layer: Box::new(dummy_rnn),
            backward_layer: None,
            input_cache: Arc::new(RwLock::new(None)),
            name: self.name.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F>
    for ThreadSafeBidirectional<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.to_owned());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        // Run forward layer
        let forward_output = self.forward_layer.forward(input)?;

        // If we have a backward layer, run it on reversed input and combine
        if let Some(ref backward_layer) = self.backward_layer {
            // Reverse input along sequence dimension (axis 1)
            let mut reversed_input = input.to_owned();
            reversed_input.invert_axis(Axis(1));

            // Run backward layer
            let mut backward_output = backward_layer.forward(&reversed_input)?;

            // Reverse backward output to align with forward output
            backward_output.invert_axis(Axis(1));

            // Combine forward and backward outputs along last dimension
            let combined =
                ndarray::stack(Axis(2), &[forward_output.view(), backward_output.view()])?;

            // Reshape to flatten the last dimension
            let shape = combined.shape();
            let new_shape = (shape[0], shape[1], shape[2] * shape[3]);
            let output = combined.into_shape_with_order(new_shape)?.into_dyn();

            Ok(output)
        } else {
            // If no backward layer, just return forward output
            Ok(forward_output)
        }
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached input
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };

        if input_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            ));
        }

        // For now, just return a placeholder gradient
        let grad_input = Array::zeros(_input.dim());

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

// TODO: Add thread-safe LSTM and GRU implementations following the same pattern
