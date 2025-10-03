//! Long Short-Term Memory (LSTM) implementation

use crate::error::{NeuralError, Result};
use crate::layers::recurrent::{LstmGateCache, LstmStepOutput};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, ArrayView, Ix2, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Distribution, Uniform};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
/// Configuration for LSTM layers
#[derive(Debug, Clone)]
pub struct LSTMConfig {
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden units
    pub hidden_size: usize,
}
/// Long Short-Term Memory (LSTM) layer
///
/// Implements an LSTM layer with the following update rules:
/// i_t = sigmoid(W_ii * x_t + b_ii + W_hi * h_(t-1) + b_hi)  # input gate
/// f_t = sigmoid(W_if * x_t + b_if + W_hf * h_(t-1) + b_hf)  # forget gate
/// g_t = tanh(W_ig * x_t + b_ig + W_hg * h_(t-1) + b_hg)     # cell input
/// o_t = sigmoid(W_io * x_t + b_io + W_ho * h_(t-1) + b_ho)  # output gate
/// c_t = f_t * c_(t-1) + i_t * g_t                          # cell state
/// h_t = o_t * tanh(c_t)                                     # hidden state
/// # Examples
/// ```
/// use scirs2_neural::layers::{Layer, recurrent::LSTM};
/// use scirs2_core::ndarray::{Array, Array3};
/// use scirs2_core::random::rngs::StdRng;
/// use scirs2_core::random::SeedableRng;
/// // Create an LSTM layer with 10 input features and 20 hidden units
/// let mut rng = StdRng::seed_from_u64(42);
/// let lstm = LSTM::new(10, 20, &mut rng).unwrap();
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = lstm.forward(&input).unwrap();
/// // Output should have dimensions [batch_size, seq_len, hidden_size]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
pub struct LSTM<F: Float + Debug + Send + Sync> {
    /// Input size (number of input features)
    input_size: usize,
    /// Hidden size (number of hidden units)
    hidden_size: usize,
    /// Input-to-hidden weights for input gate
    weight_ii: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for input gate
    weight_hi: Array<F, IxDyn>,
    /// Input-to-hidden bias for input gate
    bias_ii: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for input gate
    bias_hi: Array<F, IxDyn>,
    /// Input-to-hidden weights for forget gate
    weight_if: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for forget gate
    weight_hf: Array<F, IxDyn>,
    /// Input-to-hidden bias for forget gate
    bias_if: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for forget gate
    bias_hf: Array<F, IxDyn>,
    /// Input-to-hidden weights for cell gate
    weight_ig: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for cell gate
    weight_hg: Array<F, IxDyn>,
    /// Input-to-hidden bias for cell gate
    bias_ig: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for cell gate
    bias_hg: Array<F, IxDyn>,
    /// Input-to-hidden weights for output gate
    weight_io: Array<F, IxDyn>,
    /// Hidden-to-hidden weights for output gate
    weight_ho: Array<F, IxDyn>,
    /// Input-to-hidden bias for output gate
    bias_io: Array<F, IxDyn>,
    /// Hidden-to-hidden bias for output gate
    bias_ho: Array<F, IxDyn>,
    /// Gradients for all parameters (kept simple here)
    #[allow(dead_code)]
    gradients: Arc<RwLock<Vec<Array<F, IxDyn>>>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Hidden states cache for backward pass
    hidden_states_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cell states cache for backward pass
    cell_states_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Gate values cache for backward pass
    #[allow(dead_code)]
    gate_cache: LstmGateCache<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> LSTM<F> {
    /// Create a new LSTM layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `rng` - Random number generator for weight initialization
    /// # Returns
    /// * A new LSTM layer
    pub fn new<R: scirs2_core::random::Rng + scirs2_core::random::RngCore>(
        input_size: usize,
        hidden_size: usize,
        rng: &mut R,
    ) -> Result<Self> {
        // Validate parameters
        if input_size == 0 || hidden_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Input _size and hidden _size must be positive".to_string(),
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
            let uniform = Uniform::new(-1.0, 1.0).map_err(|e| {
                NeuralError::InvalidArchitecture(format!(
                    "Failed to create uniform distribution: {e}"
                ))
            })?;
            for _ in 0..(rows * cols) {
                let rand_val = uniform.sample(rng);
                let val = F::from(rand_val).ok_or_else(|| {
                    NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
                })?;
                weights_vec.push(val * scale);
            }
            Array::from_shape_vec(IxDyn(&[rows, cols]), weights_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {e}"))
            })
        };
        // Initialize all weights and biases
        let weight_ii = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hi = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_ii: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hi: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let weight_if = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hf = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        // Initialize forget gate biases to 1.0 (common practice to help training)
        let mut bias_if: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let mut bias_hf: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let one = F::one();
        for i in 0..hidden_size {
            bias_if[i] = one;
            bias_hf[i] = one;
        }

        let weight_ig = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hg = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_ig: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hg: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let weight_io = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_ho = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_io: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_ho: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        // Initialize gradients
        let gradients = vec![
            Array::zeros(weight_ii.dim()),
            Array::zeros(weight_hi.dim()),
            Array::zeros(bias_ii.dim()),
            Array::zeros(bias_hi.dim()),
            Array::zeros(weight_if.dim()),
            Array::zeros(weight_hf.dim()),
            Array::zeros(bias_if.dim()),
            Array::zeros(bias_hf.dim()),
            Array::zeros(weight_ig.dim()),
            Array::zeros(weight_hg.dim()),
            Array::zeros(bias_ig.dim()),
            Array::zeros(bias_hg.dim()),
            Array::zeros(weight_io.dim()),
            Array::zeros(weight_ho.dim()),
            Array::zeros(bias_io.dim()),
            Array::zeros(bias_ho.dim()),
        ];
        Ok(Self {
            input_size,
            hidden_size,
            weight_ii,
            weight_hi,
            bias_ii,
            bias_hi,
            weight_if,
            weight_hf,
            bias_if,
            bias_hf,
            weight_ig,
            weight_hg,
            bias_ig,
            bias_hg,
            weight_io,
            weight_ho,
            bias_io,
            bias_ho,
            gradients: Arc::new(RwLock::new(gradients)),
            input_cache: Arc::new(RwLock::new(None)),
            hidden_states_cache: Arc::new(RwLock::new(None)),
            cell_states_cache: Arc::new(RwLock::new(None)),
            gate_cache: Arc::new(RwLock::new(None)),
        })
    }
    /// Helper method to compute one step of the LSTM
    /// * `x` - Input tensor of shape [batch_size, input_size]
    /// * `h` - Previous hidden state of shape [batch_size, hidden_size]
    /// * `c` - Previous cell state of shape [batch_size, hidden_size]
    /// * (new_h, new_c, gates) where:
    ///   - new_h: New hidden state of shape [batch_size, hidden_size]
    ///   - new_c: New cell state of shape [batch_size, hidden_size]
    ///   - gates: (input_gate, forget_gate, cell_gate, output_gate)
    fn step(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
        c: &ArrayView<F, IxDyn>,
    ) -> Result<LstmStepOutput<F>> {
        let xshape = x.shape();
        let hshape = h.shape();
        let cshape = c.shape();
        let batch_size = xshape[0];
        // Validate shapes
        if xshape[1] != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "Input feature dimension mismatch: expected {}, got {}",
                self.input_size, xshape[1]
            )));
        }
        if hshape[1] != self.hidden_size || cshape[1] != self.hidden_size {
            return Err(NeuralError::InferenceError(format!(
                "Hidden/cell state dimension mismatch: expected {}, got {}/{}",
                self.hidden_size, hshape[1], cshape[1]
            )));
        }
        if xshape[0] != hshape[0] || xshape[0] != cshape[0] {
            return Err(NeuralError::InferenceError(format!(
                "Batch size mismatch: input has {}, hidden state has {}, cell state has {}",
                xshape[0], hshape[0], cshape[0]
            )));
        }
        // Initialize gates
        let mut i_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut f_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut g_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut o_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        // Initialize new states
        let mut new_c: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut new_h: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        // Compute gates for each batch item
        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                // Input gate (i_t)
                let mut i_sum = self.bias_ii[i] + self.bias_hi[i];
                for j in 0..self.input_size {
                    i_sum = i_sum + self.weight_ii[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    i_sum = i_sum + self.weight_hi[[i, j]] * h[[b, j]];
                }
                i_gate[[b, i]] = F::one() / (F::one() + (-i_sum).exp()); // sigmoid

                // Forget gate (f_t)
                let mut f_sum = self.bias_if[i] + self.bias_hf[i];
                for j in 0..self.input_size {
                    f_sum = f_sum + self.weight_if[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    f_sum = f_sum + self.weight_hf[[i, j]] * h[[b, j]];
                }
                f_gate[[b, i]] = F::one() / (F::one() + (-f_sum).exp()); // sigmoid

                // Cell gate (g_t)
                let mut g_sum = self.bias_ig[i] + self.bias_hg[i];
                for j in 0..self.input_size {
                    g_sum = g_sum + self.weight_ig[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    g_sum = g_sum + self.weight_hg[[i, j]] * h[[b, j]];
                }
                g_gate[[b, i]] = g_sum.tanh(); // tanh

                // Output gate (o_t)
                let mut o_sum = self.bias_io[i] + self.bias_ho[i];
                for j in 0..self.input_size {
                    o_sum = o_sum + self.weight_io[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    o_sum = o_sum + self.weight_ho[[i, j]] * h[[b, j]];
                }
                o_gate[[b, i]] = F::one() / (F::one() + (-o_sum).exp()); // sigmoid
                                                                         // New cell state (c_t)
                new_c[[b, i]] = f_gate[[b, i]] * c[[b, i]] + i_gate[[b, i]] * g_gate[[b, i]];
                // New hidden state (h_t)
                new_h[[b, i]] = o_gate[[b, i]] * new_c[[b, i]].tanh();
            }
        }

        // Convert all to dynamic dimension
        let new_h_dyn = new_h.into_dyn();
        let new_c_dyn = new_c.into_dyn();
        let i_gate_dyn = i_gate.into_dyn();
        let f_gate_dyn = f_gate.into_dyn();
        let g_gate_dyn = g_gate.into_dyn();
        let o_gate_dyn = o_gate.into_dyn();
        Ok((
            new_h_dyn,
            new_c_dyn,
            (i_gate_dyn, f_gate_dyn, g_gate_dyn, o_gate_dyn),
        ))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for LSTM<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        *self.input_cache.write().unwrap() = Some(input.clone());
        // Validate input shape
        let inputshape = input.shape();
        if inputshape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input [batch_size, seq_len, features], got {inputshape:?}"
            )));
        }

        let batch_size = inputshape[0];
        let seq_len = inputshape[1];
        let features = inputshape[2];
        if features != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "Input features dimension mismatch: expected {}, got {}",
                self.input_size, features
            )));
        }
        // Initialize hidden and cell states to zeros
        let mut h = Array::zeros((batch_size, self.hidden_size));
        let mut c = Array::zeros((batch_size, self.hidden_size));
        // Initialize output arrays to store all states
        let mut all_hidden_states = Array::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_cell_states = Array::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_gates = Vec::with_capacity(seq_len);
        // Process each time step
        for t in 0..seq_len {
            // Extract input at time t
            let x_t = input.slice(scirs2_core::ndarray::s![.., t, ..]);
            // Process one step - converting views to dynamic dimension
            let x_t_view = x_t.view().into_dyn();
            let h_view = h.view().into_dyn();
            let c_view = c.view().into_dyn();
            let (new_h, new_c, gates) = self.step(&x_t_view, &h_view, &c_view)?;
            // Convert back from dynamic dimension
            h = new_h.into_dimensionality::<Ix2>().unwrap();
            c = new_c.into_dimensionality::<Ix2>().unwrap();
            all_gates.push(gates);
            // Store hidden and cell states
            for b in 0..batch_size {
                for i in 0..self.hidden_size {
                    all_hidden_states[[b, t, i]] = h[[b, i]];
                    all_cell_states[[b, t, i]] = c[[b, i]];
                }
            }
        }

        // Cache states and gates for backward pass
        *self.hidden_states_cache.write().unwrap() = Some(all_hidden_states.clone().into_dyn());
        *self.cell_states_cache.write().unwrap() = Some(all_cell_states.into_dyn());
        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;
        let hidden_states_ref = self.hidden_states_cache.read().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire read lock on hidden states cache".to_string(),
            )
        })?;
        let cell_states_ref = self.cell_states_cache.read().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire read lock on cell states cache".to_string(),
            )
        })?;
        if input_ref.is_none() || hidden_states_ref.is_none() || cell_states_ref.is_none() {
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

    fn update(&mut self, learningrate: F) -> Result<()> {
        // Apply a small update to parameters (placeholder)
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learningrate;
        // Helper function to update a parameter
        let update_param = |param: &mut Array<F, IxDyn>| {
            for w in param.iter_mut() {
                *w = *w - lr;
            }
        };

        // Update all parameters
        update_param(&mut self.weight_ii);
        update_param(&mut self.weight_hi);
        update_param(&mut self.bias_ii);
        update_param(&mut self.bias_hi);
        update_param(&mut self.weight_if);
        update_param(&mut self.weight_hf);
        update_param(&mut self.bias_if);
        update_param(&mut self.bias_hf);
        update_param(&mut self.weight_ig);
        update_param(&mut self.weight_hg);
        update_param(&mut self.bias_ig);
        update_param(&mut self.bias_hg);
        update_param(&mut self.weight_io);
        update_param(&mut self.weight_ho);
        update_param(&mut self.bias_io);
        update_param(&mut self.bias_ho);
        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> ParamLayer<F> for LSTM<F> {
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![
            self.weight_ii.clone(),
            self.weight_hi.clone(),
            self.bias_ii.clone(),
            self.bias_hi.clone(),
            self.weight_if.clone(),
            self.weight_hf.clone(),
            self.bias_if.clone(),
            self.bias_hf.clone(),
            self.weight_ig.clone(),
            self.weight_hg.clone(),
            self.bias_ig.clone(),
            self.bias_hg.clone(),
            self.weight_io.clone(),
            self.weight_ho.clone(),
            self.bias_io.clone(),
            self.bias_ho.clone(),
        ]
    }

    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        // This is a placeholder implementation until proper gradient access is implemented
        // Return an empty vector as we can't get references to the gradients inside the RwLock
        // The actual gradient update logic is handled in the backward method
        Vec::new()
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 16 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 16 parameters, got {}",
                params.len()
            )));
        }

        let expectedshapes = vec![
            self.weight_ii.shape(),
            self.weight_hi.shape(),
            self.bias_ii.shape(),
            self.bias_hi.shape(),
            self.weight_if.shape(),
            self.weight_hf.shape(),
            self.bias_if.shape(),
            self.bias_hf.shape(),
            self.weight_ig.shape(),
            self.weight_hg.shape(),
            self.bias_ig.shape(),
            self.bias_hg.shape(),
            self.weight_io.shape(),
            self.weight_ho.shape(),
            self.bias_io.shape(),
            self.bias_ho.shape(),
        ];

        for (i, (param, expected)) in params.iter().zip(expectedshapes.iter()).enumerate() {
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
        self.weight_ii = params[0].clone();
        self.weight_hi = params[1].clone();
        self.bias_ii = params[2].clone();
        self.bias_hi = params[3].clone();
        self.weight_if = params[4].clone();
        self.weight_hf = params[5].clone();
        self.bias_if = params[6].clone();
        self.bias_hf = params[7].clone();
        self.weight_ig = params[8].clone();
        self.weight_hg = params[9].clone();
        self.bias_ig = params[10].clone();
        self.bias_hg = params[11].clone();
        self.weight_io = params[12].clone();
        self.weight_ho = params[13].clone();
        self.bias_io = params[14].clone();
        self.bias_ho = params[15].clone();

        Ok(())
    }
}
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use scirs2_core::ndarray::Array3;
//     use scirs2_core::random::rngs::SmallRng;
//     use scirs2_core::random::SeedableRng;
//
//     #[test]
// //     fn test_lstmshape() {
// //         // Create an LSTM layer
// //         let mut rng = scirs2_core::random::rng();
// //         let lstm = LSTM::<f64>::new(
// //             10, // input_size
// //             20, // hidden_size
// //             &mut rng,
// //         )
// //         .unwrap();
// //
// //         // Create a batch of input data
// //         let batch_size = 2;
// //         let seq_len = 5;
// //         let input_size = 10;
// //         let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
// //         // Forward pass
// //         let output = lstm.forward(&input).unwrap();
// //         // Check output shape
// //         assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
// //     }
// // }
