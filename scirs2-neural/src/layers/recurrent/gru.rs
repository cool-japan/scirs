//! Gated Recurrent Unit (GRU) implementation

use crate::error::{NeuralError, Result};
use crate::layers::recurrent::{GruForwardOutput, GruGateSeqCache};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, ArrayView, ArrayView1, Ix2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Uniform};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Threshold for using SIMD-accelerated GRU step
const GRU_SIMD_THRESHOLD: usize = 32;
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
/// # Examples
/// ```
/// use scirs2_neural::layers::{Layer, recurrent::GRU};
/// use scirs2_core::ndarray::{Array, Array3};
/// use scirs2_core::random::rngs::StdRng;
/// use scirs2_core::random::SeedableRng;
/// // Create a GRU layer with 10 input features and 20 hidden units
/// let mut rng = StdRng::seed_from_u64(42);
/// let gru = GRU::new(10, 20, &mut rng).expect("Operation failed");
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = gru.forward(&input).expect("Operation failed");
/// // Output should have dimensions [batch_size, seq_len, hidden_size]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
pub struct GRU<F: Float + Debug + NumAssign> {
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
    /// Gradients for all 12 parameters, in the order reported by
    /// [`ParamLayer::get_parameters`]; filled in by `backward`
    gradients: RwLock<Vec<Array<F, IxDyn>>>,
    /// Input cache for backward pass
    input_cache: RwLock<Option<Array<F, IxDyn>>>,
    /// Hidden states cache for backward pass
    hidden_states_cache: RwLock<Option<Array<F, IxDyn>>>,
    /// Per-time-step gate activations cached by `forward` for use by BPTT
    gate_cache: GruGateSeqCache<F>,
}

/// Index of each GRU parameter inside the flat gradient/parameter vector
mod param_index {
    /// Input-to-hidden weights of the reset gate
    pub const W_IR: usize = 0;
    /// Hidden-to-hidden weights of the reset gate
    pub const W_HR: usize = 1;
    /// Input-to-hidden bias of the reset gate
    pub const B_IR: usize = 2;
    /// Hidden-to-hidden bias of the reset gate
    pub const B_HR: usize = 3;
    /// Input-to-hidden weights of the update gate
    pub const W_IZ: usize = 4;
    /// Hidden-to-hidden weights of the update gate
    pub const W_HZ: usize = 5;
    /// Input-to-hidden bias of the update gate
    pub const B_IZ: usize = 6;
    /// Hidden-to-hidden bias of the update gate
    pub const B_HZ: usize = 7;
    /// Input-to-hidden weights of the candidate ("new") gate
    pub const W_IN: usize = 8;
    /// Hidden-to-hidden weights of the candidate ("new") gate
    pub const W_HN: usize = 9;
    /// Input-to-hidden bias of the candidate ("new") gate
    pub const B_IN: usize = 10;
    /// Hidden-to-hidden bias of the candidate ("new") gate
    pub const B_HN: usize = 11;
    /// Total number of GRU parameter tensors
    pub const COUNT: usize = 12;
}

impl<F: Float + Debug + ScalarOperand + SimdUnifiedOps + 'static + NumAssign> GRU<F> {
    /// Create a new GRU layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `rng` - Random number generator for weight initialization
    /// # Returns
    /// * A new GRU layer
    pub fn new<R: scirs2_core::random::Rng>(
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
            NeuralError::InvalidArchitecture("Failed to convert hidden _size scale".to_string())
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
        let weight_ir = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hr = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_ir: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hr: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let weight_iz = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hz = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_iz: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hz: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let weight_in = create_weight_matrix(hidden_size, input_size, scale_ih)?;
        let weight_hn = create_weight_matrix(hidden_size, hidden_size, scale_hh)?;
        let bias_in: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hn: Array<F, IxDyn> = Array::zeros(IxDyn(&[hidden_size]));
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
            gradients: RwLock::new(gradients),
            input_cache: RwLock::new(None),
            hidden_states_cache: RwLock::new(None),
            gate_cache: Arc::new(RwLock::new(None)),
        })
    }
    /// Check if SIMD path should be used
    fn should_use_simd(&self) -> bool {
        self.input_size + self.hidden_size >= GRU_SIMD_THRESHOLD
    }

    /// Helper method to compute one step of the GRU
    /// * `x` - Input tensor of shape [batch_size, input_size]
    /// * `h` - Previous hidden state of shape [batch_size, hidden_size]
    /// * (new_h, gates) where:
    ///   - new_h: New hidden state of shape [batch_size, hidden_size]
    ///   - gates: (reset_gate, update_gate, new_gate)
    fn step(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
    ) -> Result<GruForwardOutput<F>> {
        if self.should_use_simd() {
            self.step_simd(x, h)
        } else {
            self.step_naive(x, h)
        }
    }

    /// SIMD-accelerated step using simd_dot for gate computations
    fn step_simd(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
    ) -> Result<GruForwardOutput<F>> {
        let xshape = x.shape();
        let hshape = h.shape();
        let batch_size = xshape[0];

        if xshape[1] != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "Input feature dimension mismatch: expected {}, got {}",
                self.input_size, xshape[1]
            )));
        }
        if hshape[1] != self.hidden_size {
            return Err(NeuralError::InferenceError(format!(
                "Hidden state dimension mismatch: expected {}, got {}",
                self.hidden_size, hshape[1]
            )));
        }
        if xshape[0] != hshape[0] {
            return Err(NeuralError::InferenceError(format!(
                "Batch size mismatch: input has {}, hidden state has {}",
                xshape[0], hshape[0]
            )));
        }

        let mut r_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut z_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut n_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut new_h: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));

        for b in 0..batch_size {
            let x_b = x.slice(scirs2_core::ndarray::s![b, ..]);
            let x_view: ArrayView1<F> = x_b.into_dimensionality().expect("Operation failed");
            let h_b = h.slice(scirs2_core::ndarray::s![b, ..]);
            let h_view: ArrayView1<F> = h_b.into_dimensionality().expect("Operation failed");

            for i in 0..self.hidden_size {
                // Get weight rows for SIMD dot products
                let wir_row = self.weight_ir.slice(scirs2_core::ndarray::s![i, ..]);
                let wir_view: ArrayView1<F> =
                    wir_row.into_dimensionality().expect("Operation failed");
                let whr_row = self.weight_hr.slice(scirs2_core::ndarray::s![i, ..]);
                let whr_view: ArrayView1<F> =
                    whr_row.into_dimensionality().expect("Operation failed");

                let wiz_row = self.weight_iz.slice(scirs2_core::ndarray::s![i, ..]);
                let wiz_view: ArrayView1<F> =
                    wiz_row.into_dimensionality().expect("Operation failed");
                let whz_row = self.weight_hz.slice(scirs2_core::ndarray::s![i, ..]);
                let whz_view: ArrayView1<F> =
                    whz_row.into_dimensionality().expect("Operation failed");

                let win_row = self.weight_in.slice(scirs2_core::ndarray::s![i, ..]);
                let win_view: ArrayView1<F> =
                    win_row.into_dimensionality().expect("Operation failed");
                let whn_row = self.weight_hn.slice(scirs2_core::ndarray::s![i, ..]);
                let whn_view: ArrayView1<F> =
                    whn_row.into_dimensionality().expect("Operation failed");

                // Reset gate with simd_dot
                let r_sum = self.bias_ir[i]
                    + self.bias_hr[i]
                    + F::simd_dot(&wir_view, &x_view)
                    + F::simd_dot(&whr_view, &h_view);
                r_gate[[b, i]] = F::one() / (F::one() + (-r_sum).exp());

                // Update gate
                let z_sum = self.bias_iz[i]
                    + self.bias_hz[i]
                    + F::simd_dot(&wiz_view, &x_view)
                    + F::simd_dot(&whz_view, &h_view);
                z_gate[[b, i]] = F::one() / (F::one() + (-z_sum).exp());

                // New gate
                let n_sum = self.bias_in[i] + F::simd_dot(&win_view, &x_view);
                let hn_sum = self.bias_hn[i] + F::simd_dot(&whn_view, &h_view);
                n_gate[[b, i]] = (n_sum + r_gate[[b, i]] * hn_sum).tanh();

                // New hidden state
                new_h[[b, i]] =
                    (F::one() - z_gate[[b, i]]) * n_gate[[b, i]] + z_gate[[b, i]] * h[[b, i]];
            }
        }

        Ok((
            new_h.into_dyn(),
            (r_gate.into_dyn(), z_gate.into_dyn(), n_gate.into_dyn()),
        ))
    }

    /// Naive (scalar) step implementation for small dimensions
    fn step_naive(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
    ) -> Result<GruForwardOutput<F>> {
        let xshape = x.shape();
        let hshape = h.shape();
        let batch_size = xshape[0];

        if xshape[1] != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "Input feature dimension mismatch: expected {}, got {}",
                self.input_size, xshape[1]
            )));
        }
        if hshape[1] != self.hidden_size {
            return Err(NeuralError::InferenceError(format!(
                "Hidden state dimension mismatch: expected {}, got {}",
                self.hidden_size, hshape[1]
            )));
        }
        if xshape[0] != hshape[0] {
            return Err(NeuralError::InferenceError(format!(
                "Batch size mismatch: input has {}, hidden state has {}",
                xshape[0], hshape[0]
            )));
        }

        let mut r_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut z_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut n_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut new_h: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));

        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                let mut r_sum = self.bias_ir[i] + self.bias_hr[i];
                for j in 0..self.input_size {
                    r_sum += self.weight_ir[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    r_sum += self.weight_hr[[i, j]] * h[[b, j]];
                }
                r_gate[[b, i]] = F::one() / (F::one() + (-r_sum).exp());

                let mut z_sum = self.bias_iz[i] + self.bias_hz[i];
                for j in 0..self.input_size {
                    z_sum += self.weight_iz[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    z_sum += self.weight_hz[[i, j]] * h[[b, j]];
                }
                z_gate[[b, i]] = F::one() / (F::one() + (-z_sum).exp());

                let mut n_sum = self.bias_in[i];
                for j in 0..self.input_size {
                    n_sum += self.weight_in[[i, j]] * x[[b, j]];
                }
                let mut hn_sum = self.bias_hn[i];
                for j in 0..self.hidden_size {
                    hn_sum += self.weight_hn[[i, j]] * h[[b, j]];
                }
                n_gate[[b, i]] = (n_sum + r_gate[[b, i]] * hn_sum).tanh();

                new_h[[b, i]] =
                    (F::one() - z_gate[[b, i]]) * n_gate[[b, i]] + z_gate[[b, i]] * h[[b, i]];
            }
        }

        Ok((
            new_h.into_dyn(),
            (r_gate.into_dyn(), z_gate.into_dyn(), n_gate.into_dyn()),
        ))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign> Layer<F>
    for GRU<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        *self.input_cache.write().expect("Operation failed") = Some(input.clone());
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
        // Initialize hidden state to zeros
        let mut h = Array::zeros((batch_size, self.hidden_size));
        // Initialize output array to store all hidden states
        let mut all_hidden_states = Array::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_gates = Vec::with_capacity(seq_len);
        // Process each time step
        for t in 0..seq_len {
            // Extract input at time t
            let x_t = input.slice(scirs2_core::ndarray::s![.., t, ..]);
            // Process one step - converting views to dynamic dimension
            let x_t_view = x_t.view().into_dyn();
            let h_view = h.view().into_dyn();
            let step_result = self.step(&x_t_view, &h_view)?;
            let new_h = step_result.0;
            let gates = step_result.1;
            // Convert back from dynamic dimension
            h = new_h
                .into_dimensionality::<Ix2>()
                .expect("Operation failed");
            all_gates.push(gates);
            // Store hidden state
            for b in 0..batch_size {
                for i in 0..self.hidden_size {
                    all_hidden_states[[b, t, i]] = h[[b, i]];
                }
            }
        }
        // Cache hidden states and gate activations for backward pass
        *self.hidden_states_cache.write().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire write lock on hidden states cache".to_string(),
            )
        })? = Some(all_hidden_states.clone().into_dyn());
        *self.gate_cache.write().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire write lock on gate cache".to_string())
        })? = Some(all_gates);
        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    }

    /// Backpropagation through time for the whole cached sequence.
    ///
    /// `grad_output` holds the gradient of the loss with respect to every
    /// hidden state emitted by [`Layer::forward`] (shape
    /// `[batch, seq_len, hidden]`). Gradients of all twelve parameters are
    /// accumulated over the batch and the sequence and stored internally for
    /// [`Layer::update`] / [`ParamLayer::get_gradients`]; the returned array is
    /// the gradient with respect to the layer input.
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
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
        let gate_ref = self.gate_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on gate cache".to_string())
        })?;

        let missing = || {
            NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            )
        };
        let cached_input = input_ref.as_ref().ok_or_else(missing)?;
        let hidden_states = hidden_states_ref.as_ref().ok_or_else(missing)?;
        let gates = gate_ref.as_ref().ok_or_else(missing)?;

        if cached_input.shape() != input.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Backward input shape {:?} does not match the cached forward input shape {:?}",
                input.shape(),
                cached_input.shape()
            )));
        }

        let batch_size = cached_input.shape()[0];
        let seq_len = cached_input.shape()[1];
        let hidden_size = self.hidden_size;
        let input_size = self.input_size;

        if grad_output.shape() != [batch_size, seq_len, hidden_size] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected output gradient of shape [{batch_size}, {seq_len}, {hidden_size}], got {:?}",
                grad_output.shape()
            )));
        }
        if gates.len() != seq_len {
            return Err(NeuralError::InferenceError(format!(
                "Cached gate activations cover {} steps but the sequence has {seq_len}",
                gates.len()
            )));
        }

        let mut grads: Vec<Array<F, IxDyn>> = vec![
            Array::zeros(self.weight_ir.dim()),
            Array::zeros(self.weight_hr.dim()),
            Array::zeros(self.bias_ir.dim()),
            Array::zeros(self.bias_hr.dim()),
            Array::zeros(self.weight_iz.dim()),
            Array::zeros(self.weight_hz.dim()),
            Array::zeros(self.bias_iz.dim()),
            Array::zeros(self.bias_hz.dim()),
            Array::zeros(self.weight_in.dim()),
            Array::zeros(self.weight_hn.dim()),
            Array::zeros(self.bias_in.dim()),
            Array::zeros(self.bias_hn.dim()),
        ];

        let mut grad_input: Array<F, IxDyn> = Array::zeros(cached_input.dim());
        let mut dh_next: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));

        // Pre-activation gradients of one sample, plus the gradient of the
        // hidden-to-hidden candidate pre-sum `q = b_hn + W_hn h_{t-1}`.
        let mut da_r = vec![F::zero(); hidden_size];
        let mut da_z = vec![F::zero(); hidden_size];
        let mut da_n = vec![F::zero(); hidden_size];
        let mut d_q = vec![F::zero(); hidden_size];

        for t in (0..seq_len).rev() {
            let (r_gate, z_gate, n_gate) = &gates[t];
            let mut dh_prev: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));

            for b in 0..batch_size {
                for i in 0..hidden_size {
                    let r_t = r_gate[[b, i]];
                    let z_t = z_gate[[b, i]];
                    let n_t = n_gate[[b, i]];
                    let h_prev_i = if t == 0 {
                        F::zero()
                    } else {
                        hidden_states[[b, t - 1, i]]
                    };

                    // q_t = b_hn + W_hn h_{t-1}: recomputed from the cached
                    // previous hidden state (it is not itself cached).
                    let mut q_t = self.bias_hn[i];
                    if t > 0 {
                        for j in 0..hidden_size {
                            q_t += self.weight_hn[[i, j]] * hidden_states[[b, t - 1, j]];
                        }
                    }

                    // h_t = (1 - z_t) n_t + z_t h_{t-1}
                    let dh = grad_output[[b, t, i]] + dh_next[[b, i]];
                    let d_n = dh * (F::one() - z_t);
                    let d_z = dh * (h_prev_i - n_t);
                    // Direct path h_{t-1} -> h_t through the update gate.
                    dh_prev[[b, i]] = dh * z_t;

                    // n_t = tanh(W_in x + b_in + r_t * q_t)
                    let a_n = d_n * (F::one() - n_t * n_t);
                    da_n[i] = a_n;
                    d_q[i] = a_n * r_t;
                    let d_r = a_n * q_t;

                    da_r[i] = d_r * r_t * (F::one() - r_t);
                    da_z[i] = d_z * z_t * (F::one() - z_t);
                }

                for i in 0..hidden_size {
                    let (ar, az, an, dq) = (da_r[i], da_z[i], da_n[i], d_q[i]);
                    grads[param_index::B_IR][i] += ar;
                    grads[param_index::B_HR][i] += ar;
                    grads[param_index::B_IZ][i] += az;
                    grads[param_index::B_HZ][i] += az;
                    grads[param_index::B_IN][i] += an;
                    grads[param_index::B_HN][i] += dq;

                    for j in 0..input_size {
                        let x = cached_input[[b, t, j]];
                        grads[param_index::W_IR][[i, j]] += ar * x;
                        grads[param_index::W_IZ][[i, j]] += az * x;
                        grads[param_index::W_IN][[i, j]] += an * x;
                    }
                    for j in 0..hidden_size {
                        let h_prev = if t == 0 {
                            F::zero()
                        } else {
                            hidden_states[[b, t - 1, j]]
                        };
                        grads[param_index::W_HR][[i, j]] += ar * h_prev;
                        grads[param_index::W_HZ][[i, j]] += az * h_prev;
                        grads[param_index::W_HN][[i, j]] += dq * h_prev;
                    }
                }

                // Gradient with respect to x_t.
                for j in 0..input_size {
                    let mut sum = F::zero();
                    for i in 0..hidden_size {
                        sum += da_r[i] * self.weight_ir[[i, j]]
                            + da_z[i] * self.weight_iz[[i, j]]
                            + da_n[i] * self.weight_in[[i, j]];
                    }
                    grad_input[[b, t, j]] = sum;
                }

                // Remaining gradient with respect to h_{t-1} (through the gates).
                for j in 0..hidden_size {
                    let mut sum = F::zero();
                    for i in 0..hidden_size {
                        sum += da_r[i] * self.weight_hr[[i, j]]
                            + da_z[i] * self.weight_hz[[i, j]]
                            + d_q[i] * self.weight_hn[[i, j]];
                    }
                    dh_prev[[b, j]] += sum;
                }
            }

            dh_next = dh_prev;
        }

        *self.gradients.write().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire write lock on gradients".to_string())
        })? = grads;

        Ok(grad_input)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        let grads = {
            let guard = self.gradients.read().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on gradients".to_string())
            })?;
            guard.clone()
        };
        if grads.len() != param_index::COUNT {
            return Err(NeuralError::InferenceError(format!(
                "Expected {} parameter gradients, found {}",
                param_index::COUNT,
                grads.len()
            )));
        }

        let mut params: [&mut Array<F, IxDyn>; param_index::COUNT] = [
            &mut self.weight_ir,
            &mut self.weight_hr,
            &mut self.bias_ir,
            &mut self.bias_hr,
            &mut self.weight_iz,
            &mut self.weight_hz,
            &mut self.bias_iz,
            &mut self.bias_hz,
            &mut self.weight_in,
            &mut self.weight_hn,
            &mut self.bias_in,
            &mut self.bias_hn,
        ];

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            if param.shape() != grad.shape() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Parameter shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
            scirs2_core::ndarray::Zip::from(&mut **param)
                .and(grad)
                .for_each(|w, &g| *w -= learningrate * g);
        }

        Ok(())
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        match self.gradients.read() {
            Ok(guard) => guard.clone(),
            Err(_) => Vec::new(),
        }
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        ParamLayer::get_parameters(self)
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        ParamLayer::set_parameters(self, params.to_vec())
    }

    fn layer_type(&self) -> &str {
        "GRU"
    }

    fn parameter_count(&self) -> usize {
        3 * (self.hidden_size * self.input_size
            + self.hidden_size * self.hidden_size
            + 2 * self.hidden_size)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign>
    ParamLayer<F> for GRU<F>
{
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![
            self.weight_ir.clone(),
            self.weight_hr.clone(),
            self.bias_ir.clone(),
            self.bias_hr.clone(),
            self.weight_iz.clone(),
            self.weight_hz.clone(),
            self.bias_iz.clone(),
            self.bias_hz.clone(),
            self.weight_in.clone(),
            self.weight_hn.clone(),
            self.bias_in.clone(),
            self.bias_hn.clone(),
        ]
    }

    /// Gradients of all 12 parameters, in the same order as
    /// [`ParamLayer::get_parameters`].
    ///
    /// They are zero until [`Layer::backward`] has run at least once.
    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        match self.gradients.read() {
            Ok(guard) => guard.clone(),
            Err(_) => Vec::new(),
        }
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
        if params.len() != param_index::COUNT {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected {} parameters, got {}",
                param_index::COUNT,
                params.len()
            )));
        }

        let expectedshapes = [
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
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_grushape() {
        // Create a GRU layer
        let mut rng = SmallRng::from_seed([42; 32]);
        let gru = GRU::<f64>::new(
            10, // input_size
            20, // hidden_size
            &mut rng,
        )
        .expect("Operation failed");

        // Create a batch of input data
        let batch_size = 2;
        let seq_len = 5;
        let input_size = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
        // Forward pass
        let output = gru.forward(&input).expect("Operation failed");
        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
    }
}
