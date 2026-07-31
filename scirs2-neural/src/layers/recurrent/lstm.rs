//! Long Short-Term Memory (LSTM) implementation

use crate::error::{NeuralError, Result};
use crate::layers::recurrent::{LstmGateSeqCache, LstmStepOutput};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, ArrayView, ArrayView1, Ix2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Uniform};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Threshold for using SIMD-accelerated LSTM step
/// When input_size + hidden_size >= threshold, use SIMD path
const LSTM_SIMD_THRESHOLD: usize = 32;
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
/// let lstm = LSTM::new(10, 20, &mut rng).expect("Operation failed");
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = lstm.forward(&input).expect("Operation failed");
/// // Output should have dimensions [batch_size, seq_len, hidden_size]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
pub struct LSTM<F: Float + Debug + Send + Sync + NumAssign> {
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
    /// Gradients for all 16 parameters, in the order reported by
    /// [`ParamLayer::get_parameters`]; filled in by `backward`
    gradients: Arc<RwLock<Vec<Array<F, IxDyn>>>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Hidden states cache for backward pass
    hidden_states_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cell states cache for backward pass
    cell_states_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Per-time-step gate activations cached by `forward` for use by BPTT
    gate_cache: LstmGateSeqCache<F>,
}

/// Index of each LSTM parameter inside the flat gradient/parameter vector
mod param_index {
    /// Input-to-hidden weights of the input gate
    pub const W_II: usize = 0;
    /// Hidden-to-hidden weights of the input gate
    pub const W_HI: usize = 1;
    /// Input-to-hidden bias of the input gate
    pub const B_II: usize = 2;
    /// Hidden-to-hidden bias of the input gate
    pub const B_HI: usize = 3;
    /// Input-to-hidden weights of the forget gate
    pub const W_IF: usize = 4;
    /// Hidden-to-hidden weights of the forget gate
    pub const W_HF: usize = 5;
    /// Input-to-hidden bias of the forget gate
    pub const B_IF: usize = 6;
    /// Hidden-to-hidden bias of the forget gate
    pub const B_HF: usize = 7;
    /// Input-to-hidden weights of the cell gate
    pub const W_IG: usize = 8;
    /// Hidden-to-hidden weights of the cell gate
    pub const W_HG: usize = 9;
    /// Input-to-hidden bias of the cell gate
    pub const B_IG: usize = 10;
    /// Hidden-to-hidden bias of the cell gate
    pub const B_HG: usize = 11;
    /// Input-to-hidden weights of the output gate
    pub const W_IO: usize = 12;
    /// Hidden-to-hidden weights of the output gate
    pub const W_HO: usize = 13;
    /// Input-to-hidden bias of the output gate
    pub const B_IO: usize = 14;
    /// Hidden-to-hidden bias of the output gate
    pub const B_HO: usize = 15;
    /// Total number of LSTM parameter tensors
    pub const COUNT: usize = 16;
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    LSTM<F>
{
    /// Create a new LSTM layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `rng` - Random number generator for weight initialization
    /// # Returns
    /// * A new LSTM layer
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

    /// Check if SIMD path should be used
    fn should_use_simd(&self) -> bool {
        self.input_size + self.hidden_size >= LSTM_SIMD_THRESHOLD
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
        // Route to SIMD or naive implementation
        if self.should_use_simd() {
            self.step_simd(x, h, c)
        } else {
            self.step_naive(x, h, c)
        }
    }

    /// SIMD-accelerated step using simd_dot for gate computations
    fn step_simd(
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
        let mut new_c: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut new_h: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));

        // SIMD-accelerated gate computation using simd_dot
        for b in 0..batch_size {
            let x_b = x.slice(scirs2_core::ndarray::s![b, ..]);
            let x_view: ArrayView1<F> = x_b.into_dimensionality().expect("Operation failed");
            let h_b = h.slice(scirs2_core::ndarray::s![b, ..]);
            let h_view: ArrayView1<F> = h_b.into_dimensionality().expect("Operation failed");

            for i in 0..self.hidden_size {
                // Get weight rows for SIMD dot products
                let wii_row = self.weight_ii.slice(scirs2_core::ndarray::s![i, ..]);
                let wii_view: ArrayView1<F> =
                    wii_row.into_dimensionality().expect("Operation failed");
                let whi_row = self.weight_hi.slice(scirs2_core::ndarray::s![i, ..]);
                let whi_view: ArrayView1<F> =
                    whi_row.into_dimensionality().expect("Operation failed");

                let wif_row = self.weight_if.slice(scirs2_core::ndarray::s![i, ..]);
                let wif_view: ArrayView1<F> =
                    wif_row.into_dimensionality().expect("Operation failed");
                let whf_row = self.weight_hf.slice(scirs2_core::ndarray::s![i, ..]);
                let whf_view: ArrayView1<F> =
                    whf_row.into_dimensionality().expect("Operation failed");

                let wig_row = self.weight_ig.slice(scirs2_core::ndarray::s![i, ..]);
                let wig_view: ArrayView1<F> =
                    wig_row.into_dimensionality().expect("Operation failed");
                let whg_row = self.weight_hg.slice(scirs2_core::ndarray::s![i, ..]);
                let whg_view: ArrayView1<F> =
                    whg_row.into_dimensionality().expect("Operation failed");

                let wio_row = self.weight_io.slice(scirs2_core::ndarray::s![i, ..]);
                let wio_view: ArrayView1<F> =
                    wio_row.into_dimensionality().expect("Operation failed");
                let who_row = self.weight_ho.slice(scirs2_core::ndarray::s![i, ..]);
                let who_view: ArrayView1<F> =
                    who_row.into_dimensionality().expect("Operation failed");

                // Input gate with simd_dot
                let i_sum = self.bias_ii[i]
                    + self.bias_hi[i]
                    + F::simd_dot(&wii_view, &x_view)
                    + F::simd_dot(&whi_view, &h_view);
                i_gate[[b, i]] = F::one() / (F::one() + (-i_sum).exp());

                // Forget gate
                let f_sum = self.bias_if[i]
                    + self.bias_hf[i]
                    + F::simd_dot(&wif_view, &x_view)
                    + F::simd_dot(&whf_view, &h_view);
                f_gate[[b, i]] = F::one() / (F::one() + (-f_sum).exp());

                // Cell gate
                let g_sum = self.bias_ig[i]
                    + self.bias_hg[i]
                    + F::simd_dot(&wig_view, &x_view)
                    + F::simd_dot(&whg_view, &h_view);
                g_gate[[b, i]] = g_sum.tanh();

                // Output gate
                let o_sum = self.bias_io[i]
                    + self.bias_ho[i]
                    + F::simd_dot(&wio_view, &x_view)
                    + F::simd_dot(&who_view, &h_view);
                o_gate[[b, i]] = F::one() / (F::one() + (-o_sum).exp());

                // Cell and hidden state updates
                new_c[[b, i]] = f_gate[[b, i]] * c[[b, i]] + i_gate[[b, i]] * g_gate[[b, i]];
                new_h[[b, i]] = o_gate[[b, i]] * new_c[[b, i]].tanh();
            }
        }

        Ok((
            new_h.into_dyn(),
            new_c.into_dyn(),
            (
                i_gate.into_dyn(),
                f_gate.into_dyn(),
                g_gate.into_dyn(),
                o_gate.into_dyn(),
            ),
        ))
    }

    /// Naive (scalar) step implementation for small dimensions
    fn step_naive(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
        c: &ArrayView<F, IxDyn>,
    ) -> Result<LstmStepOutput<F>> {
        let xshape = x.shape();
        let hshape = h.shape();
        let cshape = c.shape();
        let batch_size = xshape[0];

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

        let mut i_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut f_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut g_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut o_gate: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut new_c: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));
        let mut new_h: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, self.hidden_size]));

        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                let mut i_sum = self.bias_ii[i] + self.bias_hi[i];
                for j in 0..self.input_size {
                    i_sum += self.weight_ii[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    i_sum += self.weight_hi[[i, j]] * h[[b, j]];
                }
                i_gate[[b, i]] = F::one() / (F::one() + (-i_sum).exp());

                let mut f_sum = self.bias_if[i] + self.bias_hf[i];
                for j in 0..self.input_size {
                    f_sum += self.weight_if[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    f_sum += self.weight_hf[[i, j]] * h[[b, j]];
                }
                f_gate[[b, i]] = F::one() / (F::one() + (-f_sum).exp());

                let mut g_sum = self.bias_ig[i] + self.bias_hg[i];
                for j in 0..self.input_size {
                    g_sum += self.weight_ig[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    g_sum += self.weight_hg[[i, j]] * h[[b, j]];
                }
                g_gate[[b, i]] = g_sum.tanh();

                let mut o_sum = self.bias_io[i] + self.bias_ho[i];
                for j in 0..self.input_size {
                    o_sum += self.weight_io[[i, j]] * x[[b, j]];
                }
                for j in 0..self.hidden_size {
                    o_sum += self.weight_ho[[i, j]] * h[[b, j]];
                }
                o_gate[[b, i]] = F::one() / (F::one() + (-o_sum).exp());

                new_c[[b, i]] = f_gate[[b, i]] * c[[b, i]] + i_gate[[b, i]] * g_gate[[b, i]];
                new_h[[b, i]] = o_gate[[b, i]] * new_c[[b, i]].tanh();
            }
        }

        Ok((
            new_h.into_dyn(),
            new_c.into_dyn(),
            (
                i_gate.into_dyn(),
                f_gate.into_dyn(),
                g_gate.into_dyn(),
                o_gate.into_dyn(),
            ),
        ))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for LSTM<F>
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
            h = new_h
                .into_dimensionality::<Ix2>()
                .expect("Operation failed");
            c = new_c
                .into_dimensionality::<Ix2>()
                .expect("Operation failed");
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
        *self.hidden_states_cache.write().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire write lock on hidden states cache".to_string(),
            )
        })? = Some(all_hidden_states.clone().into_dyn());
        *self.cell_states_cache.write().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire write lock on cell states cache".to_string(),
            )
        })? = Some(all_cell_states.into_dyn());
        *self.gate_cache.write().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire write lock on gate cache".to_string())
        })? = Some(all_gates);
        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    }

    /// Backpropagation through time for the whole cached sequence.
    ///
    /// `grad_output` is the gradient of the loss with respect to every hidden
    /// state emitted by [`Layer::forward`] (shape `[batch, seq_len, hidden]`).
    /// The gradients of all sixteen parameters are accumulated over the batch
    /// and the sequence and stored internally so that [`Layer::update`] (or an
    /// external optimizer reading [`ParamLayer::get_gradients`]) can apply them.
    /// The returned array is the gradient with respect to the layer input.
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
        let cell_states_ref = self.cell_states_cache.read().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire read lock on cell states cache".to_string(),
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
        let cell_states = cell_states_ref.as_ref().ok_or_else(missing)?;
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

        // Parameter gradient accumulators (same order as `get_parameters`).
        let mut grads: Vec<Array<F, IxDyn>> = vec![
            Array::zeros(self.weight_ii.dim()),
            Array::zeros(self.weight_hi.dim()),
            Array::zeros(self.bias_ii.dim()),
            Array::zeros(self.bias_hi.dim()),
            Array::zeros(self.weight_if.dim()),
            Array::zeros(self.weight_hf.dim()),
            Array::zeros(self.bias_if.dim()),
            Array::zeros(self.bias_hf.dim()),
            Array::zeros(self.weight_ig.dim()),
            Array::zeros(self.weight_hg.dim()),
            Array::zeros(self.bias_ig.dim()),
            Array::zeros(self.bias_hg.dim()),
            Array::zeros(self.weight_io.dim()),
            Array::zeros(self.weight_ho.dim()),
            Array::zeros(self.bias_io.dim()),
            Array::zeros(self.bias_ho.dim()),
        ];

        let mut grad_input: Array<F, IxDyn> = Array::zeros(cached_input.dim());
        // Gradient flowing back from the *next* time step.
        let mut dh_next: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));
        let mut dc_next: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));

        // Scratch buffers for the gate pre-activation gradients of one sample.
        let mut da_i = vec![F::zero(); hidden_size];
        let mut da_f = vec![F::zero(); hidden_size];
        let mut da_g = vec![F::zero(); hidden_size];
        let mut da_o = vec![F::zero(); hidden_size];

        for t in (0..seq_len).rev() {
            let (i_gate, f_gate, g_gate, o_gate) = &gates[t];
            let mut dh_prev: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));

            for b in 0..batch_size {
                for i in 0..hidden_size {
                    let i_t = i_gate[[b, i]];
                    let f_t = f_gate[[b, i]];
                    let g_t = g_gate[[b, i]];
                    let o_t = o_gate[[b, i]];
                    let c_t = cell_states[[b, t, i]];
                    let c_prev = if t == 0 {
                        F::zero()
                    } else {
                        cell_states[[b, t - 1, i]]
                    };

                    let tanh_c = c_t.tanh();
                    // h_t = o_t * tanh(c_t)
                    let dh = grad_output[[b, t, i]] + dh_next[[b, i]];
                    let d_o = dh * tanh_c;
                    // c_t = f_t * c_{t-1} + i_t * g_t
                    let dc = dh * o_t * (F::one() - tanh_c * tanh_c) + dc_next[[b, i]];
                    let d_f = dc * c_prev;
                    let d_i = dc * g_t;
                    let d_g = dc * i_t;
                    // Gradient carried to the previous cell state.
                    dc_next[[b, i]] = dc * f_t;

                    // Through the gate non-linearities (sigmoid / tanh).
                    da_i[i] = d_i * i_t * (F::one() - i_t);
                    da_f[i] = d_f * f_t * (F::one() - f_t);
                    da_g[i] = d_g * (F::one() - g_t * g_t);
                    da_o[i] = d_o * o_t * (F::one() - o_t);
                }

                // Parameter gradients for this (batch element, time step).
                for i in 0..hidden_size {
                    let (ai, af, ag, ao) = (da_i[i], da_f[i], da_g[i], da_o[i]);
                    grads[param_index::B_II][i] += ai;
                    grads[param_index::B_HI][i] += ai;
                    grads[param_index::B_IF][i] += af;
                    grads[param_index::B_HF][i] += af;
                    grads[param_index::B_IG][i] += ag;
                    grads[param_index::B_HG][i] += ag;
                    grads[param_index::B_IO][i] += ao;
                    grads[param_index::B_HO][i] += ao;

                    for j in 0..input_size {
                        let x = cached_input[[b, t, j]];
                        grads[param_index::W_II][[i, j]] += ai * x;
                        grads[param_index::W_IF][[i, j]] += af * x;
                        grads[param_index::W_IG][[i, j]] += ag * x;
                        grads[param_index::W_IO][[i, j]] += ao * x;
                    }
                    for j in 0..hidden_size {
                        let h_prev = if t == 0 {
                            F::zero()
                        } else {
                            hidden_states[[b, t - 1, j]]
                        };
                        grads[param_index::W_HI][[i, j]] += ai * h_prev;
                        grads[param_index::W_HF][[i, j]] += af * h_prev;
                        grads[param_index::W_HG][[i, j]] += ag * h_prev;
                        grads[param_index::W_HO][[i, j]] += ao * h_prev;
                    }
                }

                // Gradient with respect to x_t.
                for j in 0..input_size {
                    let mut sum = F::zero();
                    for i in 0..hidden_size {
                        sum += da_i[i] * self.weight_ii[[i, j]]
                            + da_f[i] * self.weight_if[[i, j]]
                            + da_g[i] * self.weight_ig[[i, j]]
                            + da_o[i] * self.weight_io[[i, j]];
                    }
                    grad_input[[b, t, j]] = sum;
                }

                // Gradient with respect to h_{t-1}.
                for j in 0..hidden_size {
                    let mut sum = F::zero();
                    for i in 0..hidden_size {
                        sum += da_i[i] * self.weight_hi[[i, j]]
                            + da_f[i] * self.weight_hf[[i, j]]
                            + da_g[i] * self.weight_hg[[i, j]]
                            + da_o[i] * self.weight_ho[[i, j]];
                    }
                    dh_prev[[b, j]] = sum;
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
            &mut self.weight_ii,
            &mut self.weight_hi,
            &mut self.bias_ii,
            &mut self.bias_hi,
            &mut self.weight_if,
            &mut self.weight_hf,
            &mut self.bias_if,
            &mut self.bias_hf,
            &mut self.weight_ig,
            &mut self.weight_hg,
            &mut self.bias_ig,
            &mut self.bias_hg,
            &mut self.weight_io,
            &mut self.weight_ho,
            &mut self.bias_io,
            &mut self.bias_ho,
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
        "LSTM"
    }

    fn parameter_count(&self) -> usize {
        4 * (self.hidden_size * self.input_size
            + self.hidden_size * self.hidden_size
            + 2 * self.hidden_size)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    ParamLayer<F> for LSTM<F>
{
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

    /// Gradients of all 16 parameters, in the same order as
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
// //         .expect("operation should succeed");
// //
// //         // Create a batch of input data
// //         let batch_size = 2;
// //         let seq_len = 5;
// //         let input_size = 10;
// //         let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
// //         // Forward pass
// //         let output = lstm.forward(&input).expect("Operation failed");
// //         // Check output shape
// //         assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
// //     }
// // }
