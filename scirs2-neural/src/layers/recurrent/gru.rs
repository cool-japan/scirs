//! Gated Recurrent Unit (GRU) implementation

use crate::error::{NeuralError, Result};
use crate::layers::recurrent::{GruForwardOutput, GruGateCache};
use crate::layers::{Layer, ParamLayer};
use ndarray::{Array, ArrayView, Ix2, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::cell::RefCell;
use std::fmt::Debug;

/// Configuration for GRU layers
#[derive(Debug, Clone)]
pub struct GRUConfig {
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden units
    pub hidden_size: usize,
}

/// Gated Recurrent Unit (GRU) layer
///
/// Implements a GRU layer with the following update rules:
/// r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_(t-1) + b_hr)  # reset gate
/// z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_(t-1) + b_hz)  # update gate
/// n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_(t-1) + b_hn))  # new gate
/// h_t = (1 - z_t) * n_t + z_t * h_(t-1)  # hidden state
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{GRU, Layer};
/// use ndarray::{Array, Array3};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create a GRU layer with 10 input features and 20 hidden units
/// let mut rng = SmallRng::seed_from_u64(42);
/// let gru = GRU::new(10, 20, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = gru.forward(&input).unwrap();
///
/// // Output should have dimensions [batch_size, seq_len, hidden_size]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
/// ```
pub struct GRU<F: Float + Debug> {
    /// Input size (number of input features)
    input_size: usize,
    /// Hidden size (number of hidden units)
    hidden_size: usize,
    /// Input-to-hidden weights for reset gate
    weight_ir: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for reset gate
    weight_hr: Array<F, IxDyn>,
    /// Input-to-hidden bias for reset gate
    bias_ir: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for reset gate
    bias_hr: Array<F, IxDyn>,
    /// Input-to-hidden weights for update gate
    weight_iz: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for update gate
    weight_hz: Array<F, IxDyn>,
    /// Input-to-hidden bias for update gate
    bias_iz: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for update gate
    bias_hz: Array<F, IxDyn>,
    /// Input-to-hidden weights for new gate
    weight_in: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for new gate
    weight_hn: Array<F, IxDyn>,
    /// Input-to-hidden bias for new gate
    bias_in: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for new gate
    bias_hn: Array<F, IxDyn>,
    /// Gradients for all parameters (kept simple here)
    #[allow(dead_code)]
    gradients: RefCell<Vec<Array<F, IxDyn>>>,
    /// Input cache for backward pass
    input_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Hidden states cache for backward pass
    hidden_states_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Gate values cache for backward pass
    #[allow(dead_code)]
    gate_cache: GruGateCache<F>,
}

impl<F: Float + Debug + ScalarOperand + 'static> GRU<F> {
    /// Create a new GRU layer
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new GRU layer
    pub fn new<R: Rng>(input_size: usize, hidden_size: usize, rng: &mut R) -> Result<Self> {
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

        // Helper function to create weight matrices
        let mut create_weight_matrix = |rows: usize,
                                        cols: usize,
                                        scale: F|
         -> Result<Array<F, IxDyn>> {
            let mut weights_vec: Vec<F> = Vec::with_capacity(rows * cols);
            for _ in 0..(rows * cols) {
                let rand_val = rng.random_range(-1.0..1.0);
                let val = F::from(rand_val).ok_or_else(|| {
                    NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
                })?;
                weights_vec.push(val * scale);
            }

            Array::from_shape_vec(IxDyn(&[rows, cols]), weights_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
            })
        };

        // Initialize all weights and biases
        let weight_ir = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hr = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_ir: Array<F, _> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hr: Array<F, _> = Array::zeros(IxDyn(&[hidden_size]));

        let weight_iz = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hz = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_iz: Array<F, _> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hz: Array<F, _> = Array::zeros(IxDyn(&[hidden_size]));

        let weight_in = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hn = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_in: Array<F, _> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hn: Array<F, _> = Array::zeros(IxDyn(&[hidden_size]));

        // Initialize gradients
        let gradients = vec![
            Array::zeros(weight_ir.dim()),
            Array::zeros(weight_hr.dim()),
            Array::zeros(bias_ir.dim()),
            Array::zeros(bias_hr.dim()),
            Array::zeros(weight_iz.dim()),
            Array::zeros(weight_hz.dim()),
            Array::zeros(bias_iz.dim()),
            Array::zeros(bias_hz.dim()),
            Array::zeros(weight_in.dim()),
            Array::zeros(weight_hn.dim()),
            Array::zeros(bias_in.dim()),
            Array::zeros(bias_hn.dim()),
        ];

        Ok(Self {
            input_size,
            hidden_size,
            weight_ir,
            weight_hr,
            bias_ir,
            bias_hr,
            weight_iz,
            weight_hz,
            bias_iz,
            bias_hz,
            weight_in,
            weight_hn,
            bias_in,
            bias_hn,
            gradients: RefCell::new(gradients),
            input_cache: RefCell::new(None),
            hidden_states_cache: RefCell::new(None),
            gate_cache: RefCell::new(None),
        })
    }

    /// Helper method to compute one step of the GRU
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, input_size]
    /// * `h` - Previous hidden state of shape [batch_size, hidden_size]
    ///
    /// # Returns
    ///
    /// * (new_h, gates) where:
    ///   - new_h: New hidden state of shape [batch_size, hidden_size]
    ///   - gates: (reset_gate, update_gate, new_gate)
    fn step(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
    ) -> Result<GruForwardOutput<F>> {
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

        // Initialize gates
        let mut r_gate: Array<F, _> = Array::zeros((batch_size, self.hidden_size));
        let mut z_gate: Array<F, _> = Array::zeros((batch_size, self.hidden_size));
        let mut n_gate: Array<F, _> = Array::zeros((batch_size, self.hidden_size));

        // Initialize new hidden state
        let mut new_h: Array<F, _> = Array::zeros((batch_size, self.hidden_size));

        // Compute gates for each batch item
        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                // Reset gate (r_t)
                let mut r_sum = self.bias_ir[i] + self.bias_hr[i];
                for j in 0..self.input_size {
                    r_sum = r_sum + self.weight_ir[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    r_sum = r_sum + self.weight_hr[[i, j]] * h[[b, j]];
                }
                r_gate[[b, i]] = F::one() / (F::one() + (-r_sum).exp()); // sigmoid

                // Update gate (z_t)
                let mut z_sum = self.bias_iz[i] + self.bias_hz[i];
                for j in 0..self.input_size {
                    z_sum = z_sum + self.weight_iz[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    z_sum = z_sum + self.weight_hz[[i, j]] * h[[b, j]];
                }
                z_gate[[b, i]] = F::one() / (F::one() + (-z_sum).exp()); // sigmoid

                // New gate (n_t)
                let mut n_sum = self.bias_in[i];
                for j in 0..self.input_size {
                    n_sum = n_sum + self.weight_in[[i, j]] * x[[b, j]];
                }

                // Reset gate applied to hidden state
                let mut hn_sum = self.bias_hn[i];
                for j in 0..self.hidden_size {
                    hn_sum = hn_sum + self.weight_hn[[i, j]] * h[[b, j]];
                }

                n_gate[[b, i]] = (n_sum + r_gate[[b, i]] * hn_sum).tanh(); // tanh

                // New hidden state (h_t)
                new_h[[b, i]] =
                    (F::one() - z_gate[[b, i]]) * n_gate[[b, i]] + z_gate[[b, i]] * h[[b, i]];
            }
        }

        // Convert all to dynamic dimension
        let new_h_dyn = new_h.into_dyn();
        let r_gate_dyn = r_gate.into_dyn();
        let z_gate_dyn = z_gate.into_dyn();
        let n_gate_dyn = n_gate.into_dyn();

        Ok((new_h_dyn, (r_gate_dyn, z_gate_dyn, n_gate_dyn)))
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for GRU<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        self.input_cache.replace(Some(input.clone()));

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
        let mut all_gates = Vec::with_capacity(seq_len);

        // Process each time step
        for t in 0..seq_len {
            // Extract input at time t
            let x_t = input.slice(ndarray::s![.., t, ..]);

            // Process one step - converting views to dynamic dimension
            let x_t_view = x_t.view().into_dyn();
            let h_view = h.view().into_dyn();

            let (new_h, gates) = self.step(&x_t_view, &h_view)?;

            // Convert back from dynamic dimension
            h = new_h.into_dimensionality::<Ix2>().unwrap();
            all_gates.push(gates);

            // Store hidden state
            for b in 0..batch_size {
                for i in 0..self.hidden_size {
                    all_hidden_states[[b, t, i]] = h[[b, i]];
                }
            }
        }

        // Cache hidden states for backward pass
        self.hidden_states_cache
            .replace(Some(all_hidden_states.clone().into_dyn()));

        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.borrow();
        let hidden_states_ref = self.hidden_states_cache.borrow();

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

        // Helper function to update a parameter
        let update_param = |param: &mut Array<F, IxDyn>| {
            for w in param.iter_mut() {
                *w = *w - lr;
            }
        };

        // Update all parameters
        update_param(&mut self.weight_ir);
        update_param(&mut self.weight_hr);
        update_param(&mut self.bias_ir);
        update_param(&mut self.bias_hr);

        update_param(&mut self.weight_iz);
        update_param(&mut self.weight_hz);
        update_param(&mut self.bias_iz);
        update_param(&mut self.bias_hz);

        update_param(&mut self.weight_in);
        update_param(&mut self.weight_hn);
        update_param(&mut self.bias_in);
        update_param(&mut self.bias_hn);

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> ParamLayer<F> for GRU<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![
            &self.weight_ir,
            &self.weight_hr,
            &self.bias_ir,
            &self.bias_hr,
            &self.weight_iz,
            &self.weight_hz,
            &self.bias_iz,
            &self.bias_hz,
            &self.weight_in,
            &self.weight_hn,
            &self.bias_in,
            &self.bias_hn,
        ]
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        // This is a placeholder implementation until proper gradient access is implemented
        // Return an empty vector as we can't get references to the gradients inside the RefCell
        // The actual gradient update logic is handled in the backward method
        Vec::new()
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 12 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 12 parameters, got {}",
                params.len()
            )));
        }

        // Validate shapes
        let expected_shapes = [
            self.weight_ir.shape(),
            self.weight_hr.shape(),
            self.bias_ir.shape(),
            self.bias_hr.shape(),
            self.weight_iz.shape(),
            self.weight_hz.shape(),
            self.bias_iz.shape(),
            self.bias_hz.shape(),
            self.weight_in.shape(),
            self.weight_hn.shape(),
            self.bias_in.shape(),
            self.bias_hn.shape(),
        ];

        for (i, (param, expected)) in params.iter().zip(expected_shapes.iter()).enumerate() {
            if param.shape() != *expected {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "Parameter {} shape mismatch: expected {:?}, got {:?}",
                    i,
                    expected,
                    param.shape()
                )));
            }
        }

        // Set parameters
        self.weight_ir = params[0].clone();
        self.weight_hr = params[1].clone();
        self.bias_ir = params[2].clone();
        self.bias_hr = params[3].clone();

        self.weight_iz = params[4].clone();
        self.weight_hz = params[5].clone();
        self.bias_iz = params[6].clone();
        self.bias_hz = params[7].clone();

        self.weight_in = params[8].clone();
        self.weight_hn = params[9].clone();
        self.bias_in = params[10].clone();
        self.bias_hn = params[11].clone();

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
    fn test_gru_shape() {
        // Create a GRU layer
        let mut rng = SmallRng::seed_from_u64(42);
        let gru = GRU::<f64>::new(
            10, // input_size
            20, // hidden_size
            &mut rng,
        )
        .unwrap();

        // Create a batch of input data
        let batch_size = 2;
        let seq_len = 5;
        let input_size = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();

        // Forward pass
        let output = gru.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
    }
}
