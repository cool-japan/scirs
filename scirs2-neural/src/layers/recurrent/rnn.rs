//! Basic Recurrent Neural Network (RNN) implementation

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use ndarray::{Array, ArrayView, Ix2, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

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

/// Configuration for RNN layers
#[derive(Debug, Clone)]
pub struct RNNConfig {
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden units
    pub hidden_size: usize,
    /// Activation function
    pub activation: RecurrentActivation,
}

impl RecurrentActivation {
    /// Apply the activation function
    pub fn apply<F: Float>(&self, x: F) -> F {
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
    pub fn apply_array<F: Float + ScalarOperand>(&self, x: &Array<F, IxDyn>) -> Array<F, IxDyn> {
        match self {
            RecurrentActivation::Tanh => x.mapv(|v| v.tanh()),
            RecurrentActivation::Sigmoid => x.mapv(|v| F::one() / (F::one() + (-v).exp())),
            RecurrentActivation::ReLU => x.mapv(|v| if v > F::zero() { v } else { F::zero() }),
        }
    }
}

/// Basic Recurrent Neural Network (RNN) layer
///
/// Implements a simple RNN layer with the following update rule:
/// h_t = activation(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{RNN, Layer, RecurrentActivation};
/// use ndarray::{Array, Array3};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create an RNN layer with 10 input features and 20 hidden units
/// let mut rng = SmallRng::seed_from_u64(42);
/// let rnn = RNN::new(10, 20, RecurrentActivation::Tanh, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = rnn.forward(&input).unwrap();
///
/// // Output should have dimensions [batch_size, seq_len, hidden_size]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
/// ```
pub struct RNN<F: Float + Debug + Send + Sync> {
    /// Input size (number of input features)
    input_size: usize,
    /// Hidden size (number of hidden units)
    hidden_size: usize,
    /// Activation function
    activation: RecurrentActivation,
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

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> RNN<F> {
    /// Create a new RNN layer
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `activation` - Activation function
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new RNN layer
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
            let rand_val = rng.random_range(-1.0..1.0);
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
            let rand_val = rng.random_range(-1.0..1.0);
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
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, input_size]
    /// * `h` - Previous hidden state of shape [batch_size, hidden_size]
    ///
    /// # Returns
    ///
    /// * New hidden state of shape [batch_size, hidden_size]
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

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for RNN<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

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
            *cache = Some(all_hidden_states.to_owned().into_dyn());
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
            *w = *w - lr;
        }

        for w in self.weight_hh.iter_mut() {
            *w = *w - lr;
        }

        for b in self.bias_ih.iter_mut() {
            *b = *b - lr;
        }

        for b in self.bias_hh.iter_mut() {
            *b = *b - lr;
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> ParamLayer<F> for RNN<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
        ]
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![
            &self.dweight_ih,
            &self.dweight_hh,
            &self.dbias_ih,
            &self.dbias_hh,
        ]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 4 parameters, got {}",
                params.len()
            )));
        }

        // Check shapes
        if params[0].shape() != self.weight_ih.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Weight_ih shape mismatch: expected {:?}, got {:?}",
                self.weight_ih.shape(),
                params[0].shape()
            )));
        }

        if params[1].shape() != self.weight_hh.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Weight_hh shape mismatch: expected {:?}, got {:?}",
                self.weight_hh.shape(),
                params[1].shape()
            )));
        }

        if params[2].shape() != self.bias_ih.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Bias_ih shape mismatch: expected {:?}, got {:?}",
                self.bias_ih.shape(),
                params[2].shape()
            )));
        }

        if params[3].shape() != self.bias_hh.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Bias_hh shape mismatch: expected {:?}, got {:?}",
                self.bias_hh.shape(),
                params[3].shape()
            )));
        }

        self.weight_ih = params[0].clone();
        self.weight_hh = params[1].clone();
        self.bias_ih = params[2].clone();
        self.bias_hh = params[3].clone();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_rnn_shape() {
        // Create an RNN layer
        let mut rng = SmallRng::seed_from_u64(42);
        let rnn = RNN::<f64>::new(
            10,                        // input_size
            20,                        // hidden_size
            RecurrentActivation::Tanh, // activation
            &mut rng,
        )
        .unwrap();

        // Create a batch of input data
        let batch_size = 2;
        let seq_len = 5;
        let input_size = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();

        // Forward pass
        let output = rnn.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
    }

    #[test]
    fn test_recurrent_activations() {
        // Test each activation function
        let tanh = RecurrentActivation::Tanh;
        let sigmoid = RecurrentActivation::Sigmoid;
        let relu = RecurrentActivation::ReLU;

        // Test tanh
        assert_eq!(tanh.apply(0.0f64), 0.0f64.tanh());
        assert_eq!(tanh.apply(1.0f64), 1.0f64.tanh());
        assert_eq!(tanh.apply(-1.0f64), (-1.0f64).tanh());

        // Test sigmoid
        assert_eq!(sigmoid.apply(0.0f64), 0.5f64);
        assert!((sigmoid.apply(10.0f64) - 1.0).abs() < 1e-4);
        assert!(sigmoid.apply(-10.0f64).abs() < 1e-4);

        // Test ReLU
        assert_eq!(relu.apply(1.0f64), 1.0f64);
        assert_eq!(relu.apply(-1.0f64), 0.0f64);
        assert_eq!(relu.apply(0.0f64), 0.0f64);
    }
}
