//! Basic Recurrent Neural Network (RNN) implementation

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, ArrayView, ArrayView1, Ix2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Threshold for using SIMD-accelerated RNN step
const RNN_SIMD_THRESHOLD: usize = 32;
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
    /// Derivative of the activation, expressed in terms of its own output
    ///
    /// For all three supported activations the derivative at `x` can be
    /// recovered from `y = f(x)` alone, which lets backpropagation reuse the
    /// hidden states cached by the forward pass instead of re-deriving the
    /// pre-activations:
    /// * `tanh`: `1 - y^2`
    /// * `sigmoid`: `y (1 - y)`
    /// * `relu`: `1` where `y > 0`, else `0` (`y > 0` iff `x > 0`)
    pub fn derivative_from_output<F: Float>(&self, y: F) -> F {
        match self {
            RecurrentActivation::Tanh => F::one() - y * y,
            RecurrentActivation::Sigmoid => y * (F::one() - y),
            RecurrentActivation::ReLU => {
                if y > F::zero() {
                    F::one()
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
/// # Examples
/// ```
/// use scirs2_neural::layers::{Layer, recurrent::{RNN, rnn::RecurrentActivation}};
/// use scirs2_core::ndarray::{Array, Array3};
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
/// // Create an RNN layer with 10 input features and 20 hidden units
/// let mut rng = scirs2_core::random::rng();
/// let rnn = RNN::new(10, 20, RecurrentActivation::Tanh, &mut rng).expect("Operation failed");
/// // Forward pass with a batch of 2 samples, sequence length 5, and 10 features
/// let batch_size = 2;
/// let seq_len = 5;
/// let input_size = 10;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
/// let output = rnn.forward(&input).expect("Operation failed");
/// // Output should have dimensions [batch_size, seq_len, hidden_size]
/// assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
pub struct RNN<F: Float + Debug + Send + Sync + NumAssign> {
    /// Input size (number of input features)
    input_size: usize,
    /// Hidden size (number of hidden units)
    hidden_size: usize,
    activation: RecurrentActivation,
    /// Input-to-hidden weights
    weight_ih: Array<F, IxDyn>,
    /// Hidden-to-hidden weights
    weight_hh: Array<F, IxDyn>,
    /// Input-to-hidden bias
    bias_ih: Array<F, IxDyn>,
    /// Hidden-to-hidden bias
    bias_hh: Array<F, IxDyn>,
    /// Gradient of input-to-hidden weights, written by `backward`
    dweight_ih: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of hidden-to-hidden weights, written by `backward`
    dweight_hh: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of input-to-hidden bias, written by `backward`
    dbias_ih: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of hidden-to-hidden bias, written by `backward`
    dbias_hh: Arc<RwLock<Array<F, IxDyn>>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Hidden states cache for backward pass
    hidden_states_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign> RNN<F> {
    /// Create a new RNN layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `activation` - Activation function
    /// * `rng` - Random number generator for weight initialization
    /// # Returns
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
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {e}"))
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
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {e}"))
            })?;
        // Initialize biases
        let bias_ih = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh = Array::zeros(IxDyn(&[hidden_size]));
        // Initialize gradients
        let dweight_ih = Arc::new(RwLock::new(Array::zeros(weight_ih.dim())));
        let dweight_hh = Arc::new(RwLock::new(Array::zeros(weight_hh.dim())));
        let dbias_ih = Arc::new(RwLock::new(Array::zeros(bias_ih.dim())));
        let dbias_hh = Arc::new(RwLock::new(Array::zeros(bias_hh.dim())));
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
    /// Check if SIMD path should be used
    fn should_use_simd(&self) -> bool {
        self.input_size + self.hidden_size >= RNN_SIMD_THRESHOLD
    }

    /// Helper method to compute one step of the RNN
    /// * `x` - Input tensor of shape [batch_size, input_size]
    /// * `h` - Previous hidden state of shape [batch_size, hidden_size]
    /// * New hidden state of shape [batch_size, hidden_size]
    fn step(&self, x: &ArrayView<F, IxDyn>, h: &ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if self.should_use_simd() {
            self.step_simd(x, h)
        } else {
            self.step_naive(x, h)
        }
    }

    /// SIMD-accelerated step using simd_dot for weight-vector products
    fn step_simd(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
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

        let mut new_h = Array::zeros((batch_size, self.hidden_size));

        for b in 0..batch_size {
            let x_b = x.slice(scirs2_core::ndarray::s![b, ..]);
            let x_view: ArrayView1<F> = x_b.into_dimensionality().expect("Operation failed");
            let h_b = h.slice(scirs2_core::ndarray::s![b, ..]);
            let h_view: ArrayView1<F> = h_b.into_dimensionality().expect("Operation failed");

            for i in 0..self.hidden_size {
                let wih_row = self.weight_ih.slice(scirs2_core::ndarray::s![i, ..]);
                let wih_view: ArrayView1<F> =
                    wih_row.into_dimensionality().expect("Operation failed");
                let whh_row = self.weight_hh.slice(scirs2_core::ndarray::s![i, ..]);
                let whh_view: ArrayView1<F> =
                    whh_row.into_dimensionality().expect("Operation failed");

                // SIMD dot products for weight-vector multiplication
                let ih_sum = self.bias_ih[i] + F::simd_dot(&wih_view, &x_view);
                let hh_sum = self.bias_hh[i] + F::simd_dot(&whh_view, &h_view);

                new_h[[b, i]] = self.activation.apply(ih_sum + hh_sum);
            }
        }

        Ok(new_h.into_dyn())
    }

    /// Naive (scalar) step implementation for small dimensions
    fn step_naive(
        &self,
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
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

        let mut new_h = Array::zeros((batch_size, self.hidden_size));

        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                let mut ih_sum = self.bias_ih[i];
                for j in 0..self.input_size {
                    ih_sum += self.weight_ih[[i, j]] * x[[b, j]];
                }
                let mut hh_sum = self.bias_hh[i];
                for j in 0..self.hidden_size {
                    hh_sum += self.weight_hh[[i, j]] * h[[b, j]];
                }
                new_h[[b, i]] = self.activation.apply(ih_sum + hh_sum);
            }
        }

        Ok(new_h.into_dyn())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign> Layer<F>
    for RNN<F>
{
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
        // Process each time step
        for t in 0..seq_len {
            // Extract input at time t
            let x_t = input.slice(scirs2_core::ndarray::s![.., t, ..]);
            // Process one step
            let x_t_view = x_t.view().into_dyn();
            let h_view = h.view().into_dyn();
            h = self
                .step(&x_t_view, &h_view)?
                .into_dimensionality::<Ix2>()
                .expect("Operation failed");
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
        // Return all hidden states
        Ok(all_hidden_states.into_dyn())
    }

    /// Backpropagation through time for the whole cached sequence.
    ///
    /// `grad_output` holds the gradient of the loss with respect to every
    /// hidden state emitted by [`Layer::forward`] (shape
    /// `[batch, seq_len, hidden]`). Weight and bias gradients are accumulated
    /// over the batch and the sequence and stored internally for
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

        let missing = || {
            NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            )
        };
        let cached_input = input_ref.as_ref().ok_or_else(missing)?;
        let hidden_states = hidden_states_ref.as_ref().ok_or_else(missing)?;

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

        let mut dweight_ih: Array<F, IxDyn> = Array::zeros(self.weight_ih.dim());
        let mut dweight_hh: Array<F, IxDyn> = Array::zeros(self.weight_hh.dim());
        let mut dbias_ih: Array<F, IxDyn> = Array::zeros(self.bias_ih.dim());
        let mut dbias_hh: Array<F, IxDyn> = Array::zeros(self.bias_hh.dim());

        let mut grad_input: Array<F, IxDyn> = Array::zeros(cached_input.dim());
        let mut dh_next: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));
        // Pre-activation gradient of a single sample.
        let mut dz = vec![F::zero(); hidden_size];

        for t in (0..seq_len).rev() {
            let mut dh_prev: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));

            for b in 0..batch_size {
                for i in 0..hidden_size {
                    let h_t = hidden_states[[b, t, i]];
                    let dh = grad_output[[b, t, i]] + dh_next[[b, i]];
                    dz[i] = dh * self.activation.derivative_from_output(h_t);
                }

                for i in 0..hidden_size {
                    let g = dz[i];
                    // Both bias vectors are added to the same pre-activation,
                    // so they receive identical gradients.
                    dbias_ih[i] += g;
                    dbias_hh[i] += g;
                    for j in 0..input_size {
                        dweight_ih[[i, j]] += g * cached_input[[b, t, j]];
                    }
                    for j in 0..hidden_size {
                        let h_prev = if t == 0 {
                            F::zero()
                        } else {
                            hidden_states[[b, t - 1, j]]
                        };
                        dweight_hh[[i, j]] += g * h_prev;
                    }
                }

                for j in 0..input_size {
                    let mut sum = F::zero();
                    for (i, &g) in dz.iter().enumerate() {
                        sum += g * self.weight_ih[[i, j]];
                    }
                    grad_input[[b, t, j]] = sum;
                }
                for j in 0..hidden_size {
                    let mut sum = F::zero();
                    for (i, &g) in dz.iter().enumerate() {
                        sum += g * self.weight_hh[[i, j]];
                    }
                    dh_prev[[b, j]] = sum;
                }
            }

            dh_next = dh_prev;
        }

        let lock_err =
            || NeuralError::InferenceError("Failed to acquire write lock on gradients".to_string());
        *self.dweight_ih.write().map_err(|_| lock_err())? = dweight_ih;
        *self.dweight_hh.write().map_err(|_| lock_err())? = dweight_hh;
        *self.dbias_ih.write().map_err(|_| lock_err())? = dbias_ih;
        *self.dbias_hh.write().map_err(|_| lock_err())? = dbias_hh;

        Ok(grad_input)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        let lock_err =
            || NeuralError::InferenceError("Failed to acquire read lock on gradients".to_string());
        let dweight_ih = self.dweight_ih.read().map_err(|_| lock_err())?.clone();
        let dweight_hh = self.dweight_hh.read().map_err(|_| lock_err())?.clone();
        let dbias_ih = self.dbias_ih.read().map_err(|_| lock_err())?.clone();
        let dbias_hh = self.dbias_hh.read().map_err(|_| lock_err())?.clone();

        for (param, grad) in [
            (&mut self.weight_ih, &dweight_ih),
            (&mut self.weight_hh, &dweight_hh),
            (&mut self.bias_ih, &dbias_ih),
            (&mut self.bias_hh, &dbias_hh),
        ] {
            if param.shape() != grad.shape() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Parameter shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
            scirs2_core::ndarray::Zip::from(&mut *param)
                .and(grad)
                .for_each(|w, &g| *w -= learningrate * g);
        }

        Ok(())
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        ParamLayer::get_gradients(self)
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        ParamLayer::get_parameters(self)
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        ParamLayer::set_parameters(self, params.to_vec())
    }

    fn layer_type(&self) -> &str {
        "RNN"
    }

    fn parameter_count(&self) -> usize {
        self.hidden_size * self.input_size
            + self.hidden_size * self.hidden_size
            + 2 * self.hidden_size
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign>
    ParamLayer<F> for RNN<F>
{
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }

    /// Gradients of `[weight_ih, weight_hh, bias_ih, bias_hh]`, in the same
    /// order as [`ParamLayer::get_parameters`].
    ///
    /// They are zero until [`Layer::backward`] has run at least once.
    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        let read = |cell: &Arc<RwLock<Array<F, IxDyn>>>| match cell.read() {
            Ok(guard) => guard.clone(),
            Err(_) => Array::zeros(IxDyn(&[0])),
        };
        vec![
            read(&self.dweight_ih),
            read(&self.dweight_hh),
            read(&self.dbias_ih),
            read(&self.dbias_hh),
        ]
    }
    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
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
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::SeedableRng;
    #[test]
    fn test_rnnshape() {
        // Create an RNN layer
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let rnn = RNN::<f64>::new(
            10,                        // input_size
            20,                        // hidden_size
            RecurrentActivation::Tanh, // activation
            &mut rng,
        )
        .expect("Operation failed");
        // Create a batch of input data
        let batch_size = 2;
        let seq_len = 5;
        let input_size = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, input_size), 0.1).into_dyn();
        // Forward pass
        let output = rnn.forward(&input).expect("Operation failed");
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
