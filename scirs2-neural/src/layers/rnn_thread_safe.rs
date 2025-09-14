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
            RecurrentActivation::Tanh => x.mapv(|v| v.tanh()),
            RecurrentActivation::Sigmoid => x.mapv(|v| F::one() / (F::one() + (-v).exp())),
            RecurrentActivation::ReLU => x.mapv(|v| if v >, F::zero() { v } else { F::zero() }),
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
        // Initialize weights with Xavier/Glorot initialization
        let scale_ih = F::from(1.0 / (input_size as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;
        let scale_hh = F::from(1.0 / (hidden_size as f64).sqrt()).ok_or_else(|| {
        // Initialize input-to-hidden weights
        let mut weight_ih_vec: Vec<F> = Vec::with_capacity(hidden_size * input_size);
        for _ in 0..(hidden_size * input_size) {
            let rand_val = rng.gen_range(-1.0f64..1.0f64);
            let val = F::from(rand_val).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
            })?;
            weight_ih_vec.push(val * scale_ih);
        let weight_ih = Array::from_shape_vec(IxDyn(&[hidden_size..input_size]), weight_ih_vec)
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
        // Initialize hidden-to-hidden weights
        let mut weight_hh_vec: Vec<F> = Vec::with_capacity(hidden_size * hidden_size);
        for _ in 0..(hidden_size * hidden_size) {
            weight_hh_vec.push(val * scale_hh);
        let weight_hh = Array::from_shape_vec(IxDyn(&[hidden_size, hidden_size]), weight_hh_vec)
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
        if hshape[1] != self.hidden_size {
                "Hidden state dimension mismatch: expected {}, got {}",
                self.hidden_size, hshape[1]
        if xshape[0] != hshape[0] {
                "Batch size mismatch: input has {}, hidden state has {}",
                xshape[0], hshape[0]
        // Initialize output
        let mut new_h = Array::zeros((batch_size, self.hidden_size));
        // Compute h_t = activation(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
        for b in 0..batch_size {
            for i in 0..self.hidden_size {
                // Input-to-hidden contribution: W_ih * x_t + b_ih
                let mut ih_sum = self.bias_ih[i];
                for j in 0..self.input_size {
                    ih_sum = ih_sum + self.weight_ih[[i, j]] * x[[b, j]];
                // Hidden-to-hidden contribution: W_hh * h_(t-1) + b_hh
                let mut hh_sum = self.bias_hh[i];
                for j in 0..self.hidden_size {
                    hh_sum = hh_sum + self.weight_hh[[i, j]] * h[[b, j]];
                // Apply activation
                new_h[[b, i]] = self.activation.apply(ih_sum + hh_sum);
        // Convert to IxDyn dimension
        let new_h_dyn = new_h.into_dyn();
        Ok(new_h_dyn)
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for ThreadSafeRNN<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.to_owned());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
        // Validate input shape
        let inputshape = input.shape();
        if inputshape.len() != 3 {
                "Expected 3D input [batch_size, seq_len, features], got {:?}",
                inputshape
        let batch_size = inputshape[0];
        let seq_len = inputshape[1];
        let features = inputshape[2];
        if features != self.input_size {
                "Input features dimension mismatch: expected {}, got {}",
                self.input_size, features
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
        // Cache all hidden states for backward pass
        if let Ok(mut cache) = self.hidden_states_cache.write() {
            *cache = Some(all_hidden_states.clone().into_dyn());
                "Failed to acquire write lock on hidden states cache".to_string(),
        // Return with correct dynamic dimension
        Ok(all_hidden_states.into_dyn())
    fn backward(
        &self,
        input: &Array<F, IxDyn>, _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
        };
        let hidden_states_ref = match self.hidden_states_cache.read() {
                    "Failed to acquire read lock on hidden states cache".to_string(),
        if input_ref.is_none() || hidden_states_ref.is_none() {
                "No cached values for backward pass. Call forward() first.".to_string(),
        // In a real implementation, we would compute gradients for all parameters
        // and return the gradient with respect to the input
        // Here we're providing a simplified version that returns a gradient of zeros
        // with the correct shape
        let grad_input = Array::zeros(input.dim());
        Ok(grad_input)
    fn update(&mut self, learningrate: F) -> Result<()> {
        // Apply a small update to parameters (placeholder)
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learning_rate;
        // Update weights and biases
        for w in self.weight_ih.iter_mut() {
            *w = *w - lr * self.dweight_ih[[0, 0]];
        for w in self.weight_hh.iter_mut() {
            *w = *w - lr * self.dweight_hh[[0, 0]];
        for b in self.bias_ih.iter_mut() {
            *b = *b - lr * self.dbias_ih[[0]];
        for b in self.bias_hh.iter_mut() {
            *b = *b - lr * self.dbias_hh[[0]];
        Ok(())
    fn as_any(&self) -> &dyn std::any::Any {
        self
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
/// Thread-safe version of Bidirectional RNN wrapper
/// This layer wraps a recurrent layer to enable bidirectional processing
/// while ensuring thread safety with Arc<RwLock<>> instead of RefCell.
pub struct ThreadSafeBidirectional<F: Float + Debug + Send + Sync> {
    /// Forward RNN layer
    forward_layer: Box<dyn Layer<F> + Send + Sync>,
    /// Backward RNN layer (optional)
    backward_layer: Option<Box<dyn Layer<F> + Send + Sync>>,
    /// Name of the layer (optional)
    name: Option<String>,
impl<F: Float + Debug + Send + Sync + ScalarOperand + 'static> ThreadSafeBidirectional<F> {
    /// Create a new Bidirectional RNN wrapper
    ///
    /// # Arguments
    /// * `layer` - The RNN layer to make bidirectional
    /// * `name` - Optional name for the layer
    /// # Returns
    /// * A new Bidirectional RNN wrapper
    pub fn new(layer: Box<dyn Layer<F> + Send + Sync>, name: Option<&str>) -> Result<Self> {
        // For now, we'll create a dummy backward RNN with the same configuration
        // In a real implementation, we would create a proper clone of the _layer
        let backward_layer = if let Some(rnn) = layer.as_any().downcast_ref::<ThreadSafeRNN<F>>() {
            // If it's a ThreadSafeRNN, create a new one with the same parameters
            let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
            let backward_rnn =
                ThreadSafeRNN::<F>::new(rnn.input_size, rnn.hidden_size, rnn.activation, &mut rng)?;
            Some(Box::new(backward_rnn) as Box<dyn Layer<F> + Send + Sync>)
            // If not, just set it to None for now
            None
            forward_layer: layer,
            backward_layer,
            name: name.map(|s| s.to_string()),
// Custom implementation of Debug for ThreadSafeBidirectional
impl<F: Float + Debug + Send + Sync> + std::fmt::Debug for ThreadSafeBidirectional<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadSafeBidirectional")
            .field("name", &self.name)
            .finish()
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
        // Create a dummy RNN for the forward _layer
        let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
        let dummy_rnn = ThreadSafeRNN::<F>::new(1, 1, RecurrentActivation::Tanh, &mut rng)
            .expect("Failed to create dummy RNN");
        Self {
            forward_layer: Box::new(dummy_rnn),
            backward_layer: None,
            name: self.name.clone(),
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F>
        // Run forward _layer
        let forward_output = self.forward_layer.forward(input)?;
        // If we have a backward layer, run it on reversed input and combine
        if let Some(ref backward_layer) = self.backward_layer {
            // Reverse input along sequence dimension (axis 1)
            let mut reversed_input = input.to_owned();
            reversed_input.invert_axis(Axis(1));
            // Run backward _layer
            let mut backward_output = backward_layer.forward(&reversed_input)?;
            // Reverse backward output to align with forward output
            backward_output.invert_axis(Axis(1));
            // Combine forward and backward outputs along last dimension
            let combined =
                ndarray::stack(Axis(2), &[forward_output.view(), backward_output.view()])?;
            // Reshape to flatten the last dimension
            let shape = combined.shape();
            let newshape = (shape[0], shape[1], shape[2] * shape[3]);
            let output = combined.into_shape_with_order(newshape)?.into_dyn();
            Ok(output)
            // If no backward layer, just return forward output
            Ok(forward_output)
        _input: &Array<F, IxDyn>,
        // Retrieve cached input
        if input_ref.is_none() {
                "No cached input for backward pass. Call forward() first.".to_string(),
        // For now, just return a placeholder gradient
        let grad_input = Array::zeros(_input.dim());
        // Update forward _layer
        self.forward_layer.update(learningrate)?;
        // Update backward _layer if present
        if let Some(ref mut backward_layer) = self.backward_layer {
            backward_layer.update(learningrate)?;
/// Thread-safe version of LSTM for sequence processing
/// This implementation uses Arc<RwLock<>> for thread safety.
#[allow(dead_code)]
pub struct ThreadSafeLSTM<F: Float + Debug + Send + Sync> {
    /// Input gate weights and biases
    weight_ih_i: Array<F, IxDyn>, // Input to input gate
    weight_hh_i: Array<F, IxDyn>, // Hidden to input gate
    bias_ih_i: Array<F, IxDyn>,
    bias_hh_i: Array<F, IxDyn>,
    /// Forget gate weights and biases
    weight_ih_f: Array<F, IxDyn>, // Input to forget gate
    weight_hh_f: Array<F, IxDyn>, // Hidden to forget gate
    bias_ih_f: Array<F, IxDyn>,
    bias_hh_f: Array<F, IxDyn>,
    /// Cell gate weights and biases
    weight_ih_g: Array<F, IxDyn>, // Input to cell gate
    weight_hh_g: Array<F, IxDyn>, // Hidden to cell gate
    bias_ih_g: Array<F, IxDyn>,
    bias_hh_g: Array<F, IxDyn>,
    /// Output gate weights and biases
    weight_ih_o: Array<F, IxDyn>, // Input to output gate
    weight_hh_o: Array<F, IxDyn>, // Hidden to output gate
    bias_ih_o: Array<F, IxDyn>,
    bias_hh_o: Array<F, IxDyn>,
    /// Gradients for all parameters
    dweight_ih_i: Array<F, IxDyn>,
    dweight_hh_i: Array<F, IxDyn>,
    dbias_ih_i: Array<F, IxDyn>,
    dbias_hh_i: Array<F, IxDyn>,
    dweight_ih_f: Array<F, IxDyn>,
    dweight_hh_f: Array<F, IxDyn>,
    dbias_ih_f: Array<F, IxDyn>,
    dbias_hh_f: Array<F, IxDyn>,
    dweight_ih_g: Array<F, IxDyn>,
    dweight_hh_g: Array<F, IxDyn>,
    dbias_ih_g: Array<F, IxDyn>,
    dbias_hh_g: Array<F, IxDyn>,
    dweight_ih_o: Array<F, IxDyn>,
    dweight_hh_o: Array<F, IxDyn>,
    dbias_ih_o: Array<F, IxDyn>,
    dbias_hh_o: Array<F, IxDyn>,
    /// Hidden and cell states cache for backward pass
    #[allow(clippy::type_complexity)]
    states_cache: Arc<RwLock<Option<(Array<F, IxDyn>, Array<F, IxDyn>)>>>,
impl<F: Float + Debug + Send + Sync + ScalarOperand + 'static> ThreadSafeLSTM<F> {
    /// Create a new thread-safe LSTM _layer
    pub fn new<R: Rng>(_input, size: usize, hiddensize: usize, rng: &mut R) -> Result<Self> {
        // Helper function to create weights
        let create_weight = |rows: usize,
                             cols: usize,
                             scale: F,
                             rng: &mut R|
         -> Result<Array<F, IxDyn>> {
            let mut weight_vec: Vec<F> = Vec::with_capacity(rows * cols);
            for _ in 0..(rows * cols) {
                let rand_val = rng.gen_range(-1.0f64..1.0f64);
                let val = F::from(rand_val).ok_or_else(|| {
                    NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
                })?;
                weight_vec.push(val * scale);
            Array::from_shape_vec(IxDyn(&[rows..cols]), weight_vec).map_err(|e| {
            })
        // Create all LSTM weights
        let weight_ih_i = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_i = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        let weight_ih_f = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_f = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        let weight_ih_g = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_g = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        let weight_ih_o = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_o = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        // Initialize biases (forget gate bias initialized to 1)
        let bias_ih_i = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_i = Array::zeros(IxDyn(&[hidden_size]));
        let mut bias_ih_f = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_f = Array::zeros(IxDyn(&[hidden_size]));
        let bias_ih_g = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_g = Array::zeros(IxDyn(&[hidden_size]));
        let bias_ih_o = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_o = Array::zeros(IxDyn(&[hidden_size]));
        // Initialize forget gate bias to 1 (common practice in LSTM)
        bias_ih_f.fill(F::one());
        let dweight_ih_i = Array::zeros(weight_ih_i.dim());
        let dweight_hh_i = Array::zeros(weight_hh_i.dim());
        let dbias_ih_i = Array::zeros(bias_ih_i.dim());
        let dbias_hh_i = Array::zeros(bias_hh_i.dim());
        let dweight_ih_f = Array::zeros(weight_ih_f.dim());
        let dweight_hh_f = Array::zeros(weight_hh_f.dim());
        let dbias_ih_f = Array::zeros(bias_ih_f.dim());
        let dbias_hh_f = Array::zeros(bias_hh_f.dim());
        let dweight_ih_g = Array::zeros(weight_ih_g.dim());
        let dweight_hh_g = Array::zeros(weight_hh_g.dim());
        let dbias_ih_g = Array::zeros(bias_ih_g.dim());
        let dbias_hh_g = Array::zeros(bias_hh_g.dim());
        let dweight_ih_o = Array::zeros(weight_ih_o.dim());
        let dweight_hh_o = Array::zeros(weight_hh_o.dim());
        let dbias_ih_o = Array::zeros(bias_ih_o.dim());
        let dbias_hh_o = Array::zeros(bias_hh_o.dim());
            weight_ih_i,
            weight_hh_i,
            bias_ih_i,
            bias_hh_i,
            weight_ih_f,
            weight_hh_f,
            bias_ih_f,
            bias_hh_f,
            weight_ih_g,
            weight_hh_g,
            bias_ih_g,
            bias_hh_g,
            weight_ih_o,
            weight_hh_o,
            bias_ih_o,
            bias_hh_o,
            dweight_ih_i,
            dweight_hh_i,
            dbias_ih_i,
            dbias_hh_i,
            dweight_ih_f,
            dweight_hh_f,
            dbias_ih_f,
            dbias_hh_f,
            dweight_ih_g,
            dweight_hh_g,
            dbias_ih_g,
            dbias_hh_g,
            dweight_ih_o,
            dweight_hh_o,
            dbias_ih_o,
            dbias_hh_o,
            states_cache: Arc::new(RwLock::new(None)),
    /// Sigmoid activation function
    fn sigmoid(x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    /// Helper method to compute one step of the LSTM
    fn step(
        x: &ArrayView<F, IxDyn>,
        h: &ArrayView<F, IxDyn>,
        c: &ArrayView<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let cshape = c.shape();
        if hshape[1] != self.hidden_size || cshape[1] != self.hidden_size {
                "Hidden/Cell state dimension mismatch: expected {}, got {}/{}",
                self.hidden_size, hshape[1], cshape[1]
        // Initialize outputs
        let mut new_c = Array::zeros((batch_size, self.hidden_size));
        // Compute LSTM gates
                // Input gate: i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
                let mut i_gate = self.bias_ih_i[i] + self.bias_hh_i[i];
                    i_gate = i_gate + self.weight_ih_i[[i, j]] * x[[b, j]];
                    i_gate = i_gate + self.weight_hh_i[[i, j]] * h[[b, j]];
                let i_t = Self::sigmoid(i_gate);
                // Forget gate: f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
                let mut f_gate = self.bias_ih_f[i] + self.bias_hh_f[i];
                    f_gate = f_gate + self.weight_ih_f[[i, j]] * x[[b, j]];
                    f_gate = f_gate + self.weight_hh_f[[i, j]] * h[[b, j]];
                let f_t = Self::sigmoid(f_gate);
                // Cell gate: g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
                let mut g_gate = self.bias_ih_g[i] + self.bias_hh_g[i];
                    g_gate = g_gate + self.weight_ih_g[[i, j]] * x[[b, j]];
                    g_gate = g_gate + self.weight_hh_g[[i, j]] * h[[b, j]];
                let g_t = g_gate.tanh();
                // Output gate: o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
                let mut o_gate = self.bias_ih_o[i] + self.bias_hh_o[i];
                    o_gate = o_gate + self.weight_ih_o[[i, j]] * x[[b, j]];
                    o_gate = o_gate + self.weight_hh_o[[i, j]] * h[[b, j]];
                let o_t = Self::sigmoid(o_gate);
                // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
                new_c[[b, i]] = f_t * c[[b, i]] + i_t * g_t;
                // Update hidden state: h_t = o_t * tanh(c_t)
                new_h[[b, i]] = o_t * new_c[[b, i]].tanh();
        let new_c_dyn = new_c.into_dyn();
        Ok((new_h_dyn, new_c_dyn))
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for ThreadSafeLSTM<F> {
        // Initialize hidden and cell states to zeros
        let mut c = Array::zeros((batch_size, self.hidden_size));
            let c_view = c.view().into_dyn();
            let (new_h, new_c) = self.step(&x_t_view, &h_view, &c_view)?;
            h = new_h.into_dimensionality::<Ix2>().unwrap();
            c = new_c.into_dimensionality::<Ix2>().unwrap();
        // Cache final states for backward pass
        if let Ok(mut cache) = self.states_cache.write() {
            *cache = Some((h.clone().into_dyn(), c.clone().into_dyn()));
                "Failed to acquire write lock on states cache".to_string(),
        let states_ref = match self.states_cache.read() {
                    "Failed to acquire read lock on states cache".to_string(),
        if input_ref.is_none() || states_ref.is_none() {
        // Apply gradients to all LSTM parameters
        // Update all weights and biases
        for w in self.weight_ih_i.iter_mut() {
            *w = *w - lr * small_change;
        for w in self.weight_hh_i.iter_mut() {
        for w in self.weight_ih_f.iter_mut() {
        for w in self.weight_hh_f.iter_mut() {
        for w in self.weight_ih_g.iter_mut() {
        for w in self.weight_hh_g.iter_mut() {
        for w in self.weight_ih_o.iter_mut() {
        for w in self.weight_hh_o.iter_mut() {
        // Update biases
        for b in self.bias_ih_i.iter_mut() {
            *b = *b - lr * small_change;
        for b in self.bias_hh_i.iter_mut() {
        for b in self.bias_ih_f.iter_mut() {
        for b in self.bias_hh_f.iter_mut() {
        for b in self.bias_ih_g.iter_mut() {
        for b in self.bias_hh_g.iter_mut() {
        for b in self.bias_ih_o.iter_mut() {
        for b in self.bias_hh_o.iter_mut() {
/// Thread-safe version of GRU for sequence processing
pub struct ThreadSafeGRU<F: Float + Debug + Send + Sync> {
    /// Reset gate weights and biases
    weight_ih_r: Array<F, IxDyn>, // Input to reset gate
    weight_hh_r: Array<F, IxDyn>, // Hidden to reset gate
    bias_ih_r: Array<F, IxDyn>,
    bias_hh_r: Array<F, IxDyn>,
    /// Update gate weights and biases  
    weight_ih_z: Array<F, IxDyn>, // Input to update gate
    weight_hh_z: Array<F, IxDyn>, // Hidden to update gate
    bias_ih_z: Array<F, IxDyn>,
    bias_hh_z: Array<F, IxDyn>,
    /// New gate weights and biases
    weight_ih_n: Array<F, IxDyn>, // Input to new gate
    weight_hh_n: Array<F, IxDyn>, // Hidden to new gate
    bias_ih_n: Array<F, IxDyn>,
    bias_hh_n: Array<F, IxDyn>,
    dweight_ih_r: Array<F, IxDyn>,
    dweight_hh_r: Array<F, IxDyn>,
    dbias_ih_r: Array<F, IxDyn>,
    dbias_hh_r: Array<F, IxDyn>,
    dweight_ih_z: Array<F, IxDyn>,
    dweight_hh_z: Array<F, IxDyn>,
    dbias_ih_z: Array<F, IxDyn>,
    dbias_hh_z: Array<F, IxDyn>,
    dweight_ih_n: Array<F, IxDyn>,
    dweight_hh_n: Array<F, IxDyn>,
    dbias_ih_n: Array<F, IxDyn>,
    dbias_hh_n: Array<F, IxDyn>,
impl<F: Float + Debug + Send + Sync + ScalarOperand + 'static> ThreadSafeGRU<F> {
    /// Create a new thread-safe GRU layer
        // Create all GRU weights
        let weight_ih_r = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_r = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        let weight_ih_z = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_z = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        let weight_ih_n = create_weight(hidden_size, input_size, scale_ih, rng)?;
        let weight_hh_n = create_weight(hidden_size, hidden_size, scale_hh, rng)?;
        let bias_ih_r = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_r = Array::zeros(IxDyn(&[hidden_size]));
        let bias_ih_z = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_z = Array::zeros(IxDyn(&[hidden_size]));
        let bias_ih_n = Array::zeros(IxDyn(&[hidden_size]));
        let bias_hh_n = Array::zeros(IxDyn(&[hidden_size]));
        let dweight_ih_r = Array::zeros(weight_ih_r.dim());
        let dweight_hh_r = Array::zeros(weight_hh_r.dim());
        let dbias_ih_r = Array::zeros(bias_ih_r.dim());
        let dbias_hh_r = Array::zeros(bias_hh_r.dim());
        let dweight_ih_z = Array::zeros(weight_ih_z.dim());
        let dweight_hh_z = Array::zeros(weight_hh_z.dim());
        let dbias_ih_z = Array::zeros(bias_ih_z.dim());
        let dbias_hh_z = Array::zeros(bias_hh_z.dim());
        let dweight_ih_n = Array::zeros(weight_ih_n.dim());
        let dweight_hh_n = Array::zeros(weight_hh_n.dim());
        let dbias_ih_n = Array::zeros(bias_ih_n.dim());
        let dbias_hh_n = Array::zeros(bias_hh_n.dim());
            weight_ih_r,
            weight_hh_r,
            bias_ih_r,
            bias_hh_r,
            weight_ih_z,
            weight_hh_z,
            bias_ih_z,
            bias_hh_z,
            weight_ih_n,
            weight_hh_n,
            bias_ih_n,
            bias_hh_n,
            dweight_ih_r,
            dweight_hh_r,
            dbias_ih_r,
            dbias_hh_r,
            dweight_ih_z,
            dweight_hh_z,
            dbias_ih_z,
            dbias_hh_z,
            dweight_ih_n,
            dweight_hh_n,
            dbias_ih_n,
            dbias_hh_n,
    /// Helper method to compute one step of the GRU
        // Compute GRU gates
                // Reset gate: r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
                let mut r_gate = self.bias_ih_r[i] + self.bias_hh_r[i];
                    r_gate = r_gate + self.weight_ih_r[[i, j]] * x[[b, j]];
                    r_gate = r_gate + self.weight_hh_r[[i, j]] * h[[b, j]];
                let r_t = Self::sigmoid(r_gate);
                // Update gate: z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
                let mut z_gate = self.bias_ih_z[i] + self.bias_hh_z[i];
                    z_gate = z_gate + self.weight_ih_z[[i, j]] * x[[b, j]];
                    z_gate = z_gate + self.weight_hh_z[[i, j]] * h[[b, j]];
                let z_t = Self::sigmoid(z_gate);
                // New gate: n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
                let mut n_gate = self.bias_ih_n[i];
                    n_gate = n_gate + self.weight_ih_n[[i, j]] * x[[b, j]];
                let mut h_part = self.bias_hh_n[i];
                    h_part = h_part + self.weight_hh_n[[i, j]] * h[[b, j]];
                n_gate = n_gate + r_t * h_part;
                let n_t = n_gate.tanh();
                // Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
                new_h[[b, i]] = (F::one() - z_t) * n_t + z_t * h[[b, i]];
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for ThreadSafeGRU<F> {
        // Cache final hidden states for backward pass
        // Apply gradients to all GRU parameters
        for w in self.weight_ih_r.iter_mut() {
        for w in self.weight_hh_r.iter_mut() {
        for w in self.weight_ih_z.iter_mut() {
        for w in self.weight_hh_z.iter_mut() {
        for w in self.weight_ih_n.iter_mut() {
        for w in self.weight_hh_n.iter_mut() {
        for b in self.bias_ih_r.iter_mut() {
        for b in self.bias_hh_r.iter_mut() {
        for b in self.bias_ih_z.iter_mut() {
        for b in self.bias_hh_z.iter_mut() {
        for b in self.bias_ih_n.iter_mut() {
        for b in self.bias_hh_n.iter_mut() {
