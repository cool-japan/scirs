// Thread-safe RNN implementations
//
// This module provides thread-safe versions of the RNN, LSTM, and GRU layers
// that can be safely used across multiple threads by using Arc<RwLock<>> instead
// of RefCell for internal state.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, ArrayView, Axis, Ix2, IxDyn, ScalarOperand, Slice};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt, SeedableRng};
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

impl RecurrentActivation {
    /// Apply the activation function
    fn apply<F: Float + NumAssign>(&self, x: F) -> F {
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

    /// Derivative of the activation expressed in terms of its own output
    ///
    /// Lets backpropagation reuse the hidden states cached by the forward pass
    /// instead of re-deriving the pre-activations.
    fn derivative_from_output<F: Float + NumAssign>(&self, y: F) -> F {
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
    fn apply_array<F: Float + NumAssign + ScalarOperand>(
        &self,
        x: &Array<F, IxDyn>,
    ) -> Array<F, IxDyn> {
        match self {
            RecurrentActivation::Tanh => x.mapv(|v| v.tanh()),
            RecurrentActivation::Sigmoid => x.mapv(|v| F::one() / (F::one() + (-v).exp())),
            RecurrentActivation::ReLU => x.mapv(|v| if v > F::zero() { v } else { F::zero() }),
        }
    }
}

/// Thread-safe version of RNN for sequence processing
///
/// This implementation replaces RefCell with Arc<RwLock<>> for thread safety.
pub struct ThreadSafeRNN<F: Float + Debug + Send + Sync + NumAssign> {
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

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Clone
    for ThreadSafeRNN<F>
{
    /// Deep-copies the learned parameters, the accumulated gradients and the
    /// forward-pass caches, producing a fully independent layer.
    fn clone(&self) -> Self {
        let clone_grad = |cell: &Arc<RwLock<Array<F, IxDyn>>>| match cell.read() {
            Ok(guard) => Arc::new(RwLock::new(guard.clone())),
            Err(poisoned) => Arc::new(RwLock::new(poisoned.into_inner().clone())),
        };
        let clone_cache = |cell: &Arc<RwLock<Option<Array<F, IxDyn>>>>| match cell.read() {
            Ok(guard) => Arc::new(RwLock::new(guard.clone())),
            Err(poisoned) => Arc::new(RwLock::new(poisoned.into_inner().clone())),
        };
        Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            activation: self.activation,
            weight_ih: self.weight_ih.clone(),
            weight_hh: self.weight_hh.clone(),
            bias_ih: self.bias_ih.clone(),
            bias_hh: self.bias_hh.clone(),
            dweight_ih: clone_grad(&self.dweight_ih),
            dweight_hh: clone_grad(&self.dweight_hh),
            dbias_ih: clone_grad(&self.dbias_ih),
            dbias_hh: clone_grad(&self.dbias_hh),
            input_cache: clone_cache(&self.input_cache),
            hidden_states_cache: clone_cache(&self.hidden_states_cache),
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> ThreadSafeRNN<F> {
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
            NeuralError::InvalidArchitecture("Failed to convert hidden size scale".to_string())
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

    /// Helper method to compute one step of the RNN
    fn step(&self, x: &ArrayView<F, IxDyn>, h: &ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let xshape = x.shape();
        let hshape = h.shape();
        let batch_size = xshape[0];

        // Validate shapes
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

        // Initialize output
        let mut new_h = Array::zeros((batch_size, self.hidden_size));

        // Compute h_t = activation(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                // Input-to-hidden contribution: W_ih * x_t + b_ih
                let mut ih_sum = self.bias_ih[i];
                for j in 0..self.input_size {
                    ih_sum += self.weight_ih[[i, j]] * x[[b, j]];
                }

                // Hidden-to-hidden contribution: W_hh * h_(t-1) + b_hh
                let mut hh_sum = self.bias_hh[i];
                for j in 0..self.hidden_size {
                    hh_sum += self.weight_hh[[i, j]] * h[[b, j]];
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

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for ThreadSafeRNN<F>
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

        // Validate input shape
        let inputshape = input.shape();
        if inputshape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input [batch_size, seq_len, features], got {:?}",
                inputshape
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
            *cache = Some(all_hidden_states.clone().into_dyn());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on hidden states cache".to_string(),
            ));
        }

        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    }

    /// Backpropagation through time for the whole cached sequence.
    ///
    /// `grad_output` holds the gradient of the loss with respect to every
    /// hidden state emitted by [`Layer::forward`] (shape
    /// `[batch, seq_len, hidden]`). Weight and bias gradients are accumulated
    /// over the batch and the sequence and stored internally for
    /// [`Layer::update`]; the returned array is the gradient with respect to
    /// the layer input.
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
                "Expected output gradient of shape [{}, {}, {}], got {:?}",
                batch_size,
                seq_len,
                hidden_size,
                grad_output.shape()
            )));
        }

        let mut dweight_ih: Array<F, IxDyn> = Array::zeros(self.weight_ih.dim());
        let mut dweight_hh: Array<F, IxDyn> = Array::zeros(self.weight_hh.dim());
        let mut dbias_ih: Array<F, IxDyn> = Array::zeros(self.bias_ih.dim());
        let mut dbias_hh: Array<F, IxDyn> = Array::zeros(self.bias_hh.dim());

        let mut grad_input: Array<F, IxDyn> = Array::zeros(cached_input.dim());
        let mut dh_next: Array<F, IxDyn> = Array::zeros(IxDyn(&[batch_size, hidden_size]));
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
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

    fn layer_type(&self) -> &str {
        "ThreadSafeRNN"
    }

    fn parameter_count(&self) -> usize {
        self.hidden_size * self.input_size
            + self.hidden_size * self.hidden_size
            + 2 * self.hidden_size
    }
}

/// Thread-safe version of Bidirectional RNN wrapper
/// This layer wraps a recurrent layer to enable bidirectional processing
/// while ensuring thread safety with Arc<RwLock<>> instead of RefCell.
pub struct ThreadSafeBidirectional<F: Float + Debug + Send + Sync + NumAssign> {
    /// RNN reading the sequence left to right
    forward_layer: ThreadSafeRNN<F>,
    /// RNN reading the sequence right to left (its own independent parameters)
    backward_layer: ThreadSafeRNN<F>,
    /// Name of the layer (optional)
    name: Option<String>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    ThreadSafeBidirectional<F>
{
    /// Create a new Bidirectional RNN wrapper
    ///
    /// The wrapped layer becomes the forward direction; a second RNN with the
    /// same architecture but its own freshly initialised parameters is created
    /// for the backward direction, as a bidirectional RNN requires.
    ///
    /// # Arguments
    /// * `layer` - The RNN layer to make bidirectional
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    /// * A new Bidirectional RNN wrapper
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArchitecture`] when `layer` is not a
    /// [`ThreadSafeRNN`]: the backward direction has to be built from the
    /// wrapped layer's architecture, which cannot be recovered from an opaque
    /// `dyn Layer` trait object.
    pub fn new(layer: Box<dyn Layer<F> + Send + Sync>, name: Option<&str>) -> Result<Self> {
        let mut rng = scirs2_core::random::rng();
        Self::new_with_rng(layer, name, &mut rng)
    }

    /// Same as [`ThreadSafeBidirectional::new`] but with a caller-supplied RNG,
    /// so the backward direction's initialisation is reproducible.
    pub fn new_with_rng<R: Rng>(
        layer: Box<dyn Layer<F> + Send + Sync>,
        name: Option<&str>,
        rng: &mut R,
    ) -> Result<Self> {
        let forward_layer = layer
            .as_any()
            .downcast_ref::<ThreadSafeRNN<F>>()
            .ok_or_else(|| {
                NeuralError::InvalidArchitecture(format!(
                    "ThreadSafeBidirectional requires a ThreadSafeRNN inner layer so that the \
                     backward direction can be built from the same architecture, got a '{}' layer",
                    layer.layer_type()
                ))
            })?
            .clone();

        let backward_layer = ThreadSafeRNN::<F>::new(
            forward_layer.input_size,
            forward_layer.hidden_size,
            forward_layer.activation,
            rng,
        )?;

        Ok(Self {
            forward_layer,
            backward_layer,
            name: name.map(|s| s.to_string()),
        })
    }

    /// Reference to the forward-direction RNN
    pub fn forward_layer(&self) -> &ThreadSafeRNN<F> {
        &self.forward_layer
    }

    /// Reference to the backward-direction RNN
    pub fn backward_layer(&self) -> &ThreadSafeRNN<F> {
        &self.backward_layer
    }
}

// Custom implementation of Debug for ThreadSafeBidirectional
impl<F: Float + Debug + Send + Sync + NumAssign> std::fmt::Debug for ThreadSafeBidirectional<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadSafeBidirectional")
            .field("name", &self.name)
            .finish()
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Clone
    for ThreadSafeBidirectional<F>
{
    /// Deep-copies both directions, including their learned parameters, so the
    /// clone reproduces the original's outputs exactly.
    fn clone(&self) -> Self {
        Self {
            forward_layer: self.forward_layer.clone(),
            backward_layer: self.backward_layer.clone(),
            name: self.name.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for ThreadSafeBidirectional<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Run forward layer
        let forward_output = self.forward_layer.forward(input)?;

        // Reverse input along sequence dimension (axis 1)
        let mut reversed_input = input.to_owned();
        reversed_input.invert_axis(Axis(1));

        // Run backward layer
        let mut backward_output = self.backward_layer.forward(&reversed_input)?;

        // Reverse backward output to align with forward output
        backward_output.invert_axis(Axis(1));

        // Concatenate along the feature axis: the first half is the forward
        // direction, the second half the backward direction. This is built
        // element-wise rather than via `stack` + reshape because
        // `invert_axis` leaves `backward_output` with a negative stride, which
        // makes a flattening reshape layout-dependent.
        let fshape = forward_output.shape().to_vec();
        if fshape.len() != 3 || backward_output.shape() != fshape.as_slice() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Bidirectional directions must produce matching 3D outputs, got {:?} and {:?}",
                fshape,
                backward_output.shape()
            )));
        }
        let (batch_size, seq_len, hidden) = (fshape[0], fshape[1], fshape[2]);
        let mut output = Array::zeros(IxDyn(&[batch_size, seq_len, hidden * 2]));
        for b in 0..batch_size {
            for t in 0..seq_len {
                for i in 0..hidden {
                    output[[b, t, i]] = forward_output[[b, t, i]];
                    output[[b, t, hidden + i]] = backward_output[[b, t, i]];
                }
            }
        }
        Ok(output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let backward_layer = &self.backward_layer;

        // The forward pass stacked [forward_out | backward_out] along the last
        // axis, so its length must be even: the first half is the forward
        // layer's gradient, the second half the backward layer's.
        let ndim = grad_output.ndim();
        if ndim < 2 {
            return Err(NeuralError::InferenceError(
                "ThreadSafeBidirectional expects at least a (batch, seq, ..) gradient".to_string(),
            ));
        }
        let last_axis = Axis(ndim - 1);
        let combined = grad_output.len_of(last_axis);
        if !combined.is_multiple_of(2) {
            return Err(NeuralError::ShapeMismatch(format!(
                "Bidirectional gradient last dimension ({combined}) must be even"
            )));
        }
        let hidden = (combined / 2) as isize;

        let grad_forward = grad_output
            .slice_axis(last_axis, Slice::new(0, Some(hidden), 1))
            .to_owned();
        let mut grad_backward = grad_output
            .slice_axis(last_axis, Slice::new(hidden, Some(combined as isize), 1))
            .to_owned();

        // Forward branch: straightforward backprop.
        let grad_input_forward = self.forward_layer.backward(input, &grad_forward)?;

        // Backward branch: the forward pass reversed the sequence axis (axis 1)
        // *after* the backward layer's forward call, so undo that reversal on the
        // incoming gradient, backprop through the backward layer on the reversed
        // input, then re-reverse the resulting input gradient.
        grad_backward.invert_axis(Axis(1));
        let mut reversed_input = input.to_owned();
        reversed_input.invert_axis(Axis(1));
        let mut grad_input_backward = backward_layer.backward(&reversed_input, &grad_backward)?;
        grad_input_backward.invert_axis(Axis(1));

        Ok(grad_input_forward + grad_input_backward)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        self.forward_layer.update(learningrate)?;
        self.backward_layer.update(learningrate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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
        "ThreadSafeBidirectional"
    }

    fn parameter_count(&self) -> usize {
        self.forward_layer.parameter_count() + self.backward_layer.parameter_count()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}
