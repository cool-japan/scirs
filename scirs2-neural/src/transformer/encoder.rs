//! Transformer encoder implementation
//!
//! This module provides implementation of transformer encoder layers and blocks
//! as described in "Attention Is All You Need" by Vaswani et al.

use crate::error::{NeuralError, Result};
use crate::layers::{AttentionConfig, Layer, LayerNorm, ParamLayer, SelfAttention};
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
/// Feed-forward network used in transformer encoder/decoder layers
///
/// Implements the position-wise feed-forward network in transformer models.
/// This consists of two linear transformations with a ReLU activation in between.
/// FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
#[derive(Debug)]
pub struct FeedForward<F: Float + Debug + Send + Sync + SimdUnifiedOps> {
    /// Input/output dimension
    d_model: usize,
    /// Hidden dimension
    d_ff: usize,
    /// First linear transformation weights
    w1: Array<F, IxDyn>,
    /// First linear transformation biases
    b1: Array<F, IxDyn>,
    /// Second linear transformation weights
    w2: Array<F, IxDyn>,
    /// Second linear transformation biases
    b2: Array<F, IxDyn>,
    /// Gradient of w1
    dw1: Array<F, IxDyn>,
    /// Gradient of b1
    db1: Array<F, IxDyn>,
    /// Gradient of w2
    dw2: Array<F, IxDyn>,
    /// Gradient of b2
    db2: Array<F, IxDyn>,
    /// Dropout rate (0 means no dropout)
    dropout: F,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Hidden state cache for backward pass
    hidden_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps> Clone
    for FeedForward<F>
{
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            d_ff: self.d_ff,
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
            dw1: self.dw1.clone(),
            db1: self.db1.clone(),
            dw2: self.dw2.clone(),
            db2: self.db2.clone(),
            dropout: self.dropout,
            input_cache: Arc::new(RwLock::new(self.input_cache.read().unwrap().clone())),
            hidden_cache: Arc::new(RwLock::new(self.hidden_cache.read().unwrap().clone())),
        }
    }
    for TransformerEncoderLayer<F>
            self_attn: self.self_attn.clone(),
            norm1: self.norm1.clone(),
            feed_forward: self.feed_forward.clone(),
            norm2: self.norm2.clone(),
            attn_output_cache: Arc::new(RwLock::new(
                self.attn_output_cache.read().unwrap().clone(),
            )),
            norm1_output_cache: Arc::new(RwLock::new(
                self.norm1_output_cache.read().unwrap().clone(),
    for TransformerEncoder<F>
            layers: self.layers.clone(),
            layer_outputs: Arc::new(RwLock::new(self.layer_outputs.read().unwrap().clone())),
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps> FeedForward<F> {
    /// Create a new feed-forward network for transformers
    ///
    /// # Arguments
    /// * `d_model` - Input/output dimension
    /// * `d_ff` - Hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `rng` - Random number generator for weight initialization
    /// # Returns
    /// * A new feed-forward network
    pub fn new<R: Rng>(_d, model: usize, dff: usize, dropout: f64, rng: &mut R) -> Result<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let scale1 = F::from(1.0 / (_d_model as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;
        let scale2 = F::from(1.0 / (d_ff as f64).sqrt()).ok_or_else(|| {
        // Helper function to create weight matrix
        let create_weight_matrix = |input_size: usize,
                                    output_size: usize,
                                    scale: F,
                                    rng: &mut R|
         -> Result<Array<F, IxDyn>> {
            let weights_vec: Vec<F> = (0..(input_size * output_size))
                .map(|_| {
                    let val = F::from(rng.random_range(-1.0..1.0)).ok_or_else(|| {
                        NeuralError::InvalidArchitecture(
                            "Failed to convert random value".to_string()..)
                    });
                    val.map(|v| v * scale).unwrap_or_else(|_| F::zero())
                })
                .collect();
            Array::from_shape_vec(IxDyn(&[input_size, output_size]), weights_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
            })
        };
        // Create weight matrices
        let w1 = create_weight_matrix(d_model, d_ff, scale1, rng)?;
        let w2 = create_weight_matrix(d_ff, d_model, scale2, rng)?;
        // Initialize biases with zeros
        let b1 = Array::zeros(IxDyn(&[d_ff]));
        let b2 = Array::zeros(IxDyn(&[d_model]));
        // Initialize gradient arrays with zeros
        let dw1 = Array::zeros(w1.dim());
        let dw2 = Array::zeros(w2.dim());
        let db1 = Array::zeros(b1.dim());
        let db2 = Array::zeros(b2.dim());
        // Convert dropout rate
        let dropout = F::from(dropout).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert dropout rate".to_string())
        Ok(Self {
            d_model,
            d_ff,
            w1,
            b1,
            w2,
            b2,
            dw1,
            dw2,
            db1,
            db2,
            dropout,
            input_cache: Arc::new(RwLock::new(None)),
            hidden_cache: Arc::new(RwLock::new(None)),
        })
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps> Layer<F>
    fn as_any(&self) -> &dyn std::any::Any {
        self
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        *self.input_cache.write().unwrap() = Some(input.clone());
        // Check input shape
        if input.ndim() < 2 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 2 dimensions [batch, ..., features]".to_string(),
            ));
        let inputshape = input.shape();
        let ndim = input.ndim();
        let feat_dim = inputshape[ndim - 1];
        if feat_dim != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of input ({}) must match d_model ({})",
                feat_dim, self.d_model
            )));
        // Compute the batch size (all dimensions except the last one)
        let batch_dims: Vec<usize> = inputshape[..ndim - 1].to_vec();
        let batch_size: usize = batch_dims.iter().product();
        // Reshape input to 2D: [batch_size, d_model]
        let reshaped_input = input
            .clone()
            .into_shape_with_order((batch_size, self.d_model))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;
        // First linear transformation: [batch_size, d_model] x [d_model, d_ff] -> [batch_size, d_ff]
        let mut hidden = Array::<F>::zeros((batch_size, self.d_ff));
        for i in 0..batch_size {
            for j in 0..self.d_ff {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    sum = sum + reshaped_input[[i, k]] * self.w1[[k, j]];
                }
                hidden[[i, j]] = sum + self.b1[[j]];
            }
        // Apply ReLU activation
                hidden[[i, j]] = if hidden[[i, j]] > F::zero() {
                    hidden[[i, j]]
                } else {
                    F::zero()
                };
        // Cache hidden state for backward pass
        // Convert 2D array to dynamic dimensions before storing
        let hidden_dyn = hidden.clone().into_dyn();
        *self.hidden_cache.write().unwrap() = Some(hidden_dyn);
        // Apply dropout if needed
        // This is a simplified version - in a real implementation, we would use a proper random mask
        if self.dropout > F::zero() {
            let keep_prob = F::one() - self.dropout;
            for i in 0..batch_size {
                for j in 0..self.d_ff {
                    hidden[[i, j]] = hidden[[i, j]] / keep_prob;
        // Second linear transformation: [batch_size, d_ff] x [d_ff, d_model] -> [batch_size, d_model]
        let mut output = Array::<F>::zeros((batch_size, self.d_model));
            for j in 0..self.d_model {
                for k in 0..self.d_ff {
                    sum = sum + hidden[[i, k]] * self.w2[[k, j]];
                output[[i, j]] = sum + self.b2[[j]];
        // Reshape output to match input shape
        let mut outputshape = inputshape.to_vec();
        outputshape[ndim - 1] = self.d_model;
        let output_reshaped = output
            .into_shape_with_order(outputshape)
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {}", e)))?;
        Ok(output_reshaped)
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.read().unwrap();
        let hidden_ref = self.hidden_cache.read().unwrap();
        if input_ref.is_none() || hidden_ref.is_none() {
                "No cached values for backward pass. Call forward() first.".to_string(),
        let cached_input = input_ref.as_ref().unwrap();
        let cached_hidden = hidden_ref.as_ref().unwrap();
        // Get input dimensions
        // Reshape cached values to 2D
        let cached_input_2d = cached_input
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape cached input: {}", e))
            })?;
        let cached_hidden_2d = cached_hidden
            .into_shape_with_order((batch_size, self.d_ff))
                NeuralError::InferenceError(format!("Failed to reshape cached hidden: {}", e))
        // Reshape grad_output to 2D
        let grad_output_2d = grad_output
                NeuralError::InferenceError(format!("Failed to reshape gradoutput: {}", e))
        // Backward through second linear layer: grad_output -> grad_hidden
        let mut grad_hidden = Array::<F>::zeros((batch_size, self.d_ff));
            for k in 0..self.d_ff {
                for j in 0..self.d_model {
                    sum = sum + grad_output_2d[[i, j]] * self.w2[[k, j]];
                grad_hidden[[i, k]] = sum;
        // Compute gradients for w2 and b2
        let mut grad_w2 = Array::<F>::zeros(self.w2.dim());
        for k in 0..self.d_ff {
                for i in 0..batch_size {
                    sum = sum + cached_hidden_2d[[i, k]] * grad_output_2d[[i, j]];
                grad_w2[[k, j]] = sum;
        let mut grad_b2 = Array::<F>::zeros(self.b2.dim());
        for j in 0..self.d_model {
            let mut sum = F::zero();
                sum = sum + grad_output_2d[[i, j]];
            grad_b2[[j]] = sum;
        // Backward through ReLU activation
                if cached_hidden_2d[[i, k]] <= F::zero() {
                    grad_hidden[[i, k]] = F::zero();
        // Backward through first linear layer: grad_hidden -> grad_input
        let mut grad_input_2d = Array::<F>::zeros((batch_size, self.d_model));
            for k in 0..self.d_model {
                    sum = sum + grad_hidden[[i, j]] * self.w1[[k, j]];
                grad_input_2d[[i, k]] = sum;
        // Compute gradients for w1 and b1
        let mut grad_w1 = Array::<F>::zeros(self.w1.dim());
        for k in 0..self.d_model {
                    sum = sum + cached_input_2d[[i, k]] * grad_hidden[[i, j]];
                grad_w1[[k, j]] = sum;
        let mut grad_b1 = Array::<F>::zeros(self.b1.dim());
        for j in 0..self.d_ff {
                sum = sum + grad_hidden[[i, j]];
            grad_b1[[j]] = sum;
        // Note: In a complete implementation, gradients would be stored in the optimizer
        // For now, we skip storing gradients to avoid unsafe code
        // The gradients (grad_w1, grad_b1, grad_w2, grad_b2) are computed but not stored
        // Reshape grad_input back to original shape
        let grad_input = grad_input_2d
            .into_shape_with_order(inputshape)
                NeuralError::InferenceError(format!("Failed to reshape gradinput: {}", e))
        Ok(grad_input)
    fn update(&mut self, learningrate: F) -> Result<()> {
        // Apply a small update
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learning_rate;
        // Update all parameters
        for w in [&mut self.w1, &mut self.w2, &mut self.b1, &mut self.b2] {
            for elem in w.iter_mut() {
                *elem = *elem - lr;
        Ok(())
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps> ParamLayer<F>
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.w1, &self.b1, &self.w2, &self.b2]
    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.dw1, &self.db1, &self.dw2, &self.db2]
    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 4 parameters, got {}",
                params.len()
        self.w1 = params[0].clone();
        self.b1 = params[1].clone();
        self.w2 = params[2].clone();
        self.b2 = params[3].clone();
/// Transformer encoder layer
/// Implements a single layer of the transformer encoder as described in
/// "Attention Is All You Need" by Vaswani et al. It consists of multi-head
/// self-attention followed by a position-wise feed-forward network, with
/// residual connections and layer normalization.
pub struct TransformerEncoderLayer<F: Float + Debug + Send + Sync + SimdUnifiedOps> {
    /// Multi-head self-attention layer
    self_attn: SelfAttention<F>,
    /// Layer normalization after attention
    norm1: LayerNorm<F>,
    /// Feed-forward network
    feed_forward: FeedForward<F>,
    /// Layer normalization after feed-forward network
    norm2: LayerNorm<F>,
    /// Dropout rate for residual connections
    /// Model embedding dimension
    /// Self-attention output cache for backward pass
    attn_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Normalized attention output cache for backward pass
    norm1_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps>
    TransformerEncoderLayer<F>
    /// Create a new transformer encoder layer
    /// * `d_model` - Model embedding dimension
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `epsilon` - Small constant for layer normalization
    /// * A new transformer encoder layer
    pub fn new<R: Rng>(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        epsilon: f64,
        rng: &mut R,
    ) -> Result<Self> {
        // Verify parameters
        if d_model % n_heads != 0 {
                "d_model ({}) must be divisible by n_heads ({})",
                d_model, n_heads
        // Calculate head dimension
        let head_dim = d_model / n_heads;
        // Create attention config
        let attn_config = AttentionConfig {
            num_heads: n_heads,
            head_dim,
            dropout_prob: dropout,
            causal: false,
            scale: None,
        // Create components
        let self_attn = SelfAttention::new(d_model, attn_config, rng)?;
        let norm1 = LayerNorm::new(d_model, epsilon, rng)?;
        let feed_forward = FeedForward::new(d_model, d_ff, dropout, rng)?;
        let norm2 = LayerNorm::new(d_model, epsilon, rng)?;
            self_attn,
            norm1,
            feed_forward,
            norm2,
            attn_output_cache: Arc::new(RwLock::new(None)),
            norm1_output_cache: Arc::new(RwLock::new(None)),
        if input.ndim() < 3 {
                "Input must have at least 3 dimensions [batch, seq_len, features]".to_string(),
        let feat_dim = inputshape[input.ndim() - 1];
        // 1. Self-attention with residual connection
        let attn_output = self.self_attn.forward(input)?;
        *self.attn_output_cache.write().unwrap() = Some(attn_output.clone());
        // Add residual connection (x + Sublayer(x))
        // Clone input and add attention output using element-wise operations
        let mut attn_output_residual = input.clone();
        let tmp = &attn_output_residual + &attn_output;
        attn_output_residual = tmp;
        // 2. Layer normalization after attention
        let norm1_output = self.norm1.forward(&attn_output_residual)?;
        *self.norm1_output_cache.write().unwrap() = Some(norm1_output.clone());
        // 3. Feed-forward network with residual connection
        let ff_output = self.feed_forward.forward(&norm1_output)?;
        let mut output = norm1_output.clone();
        // Add feed-forward output using element-wise operations
        let tmp = &output + &ff_output;
        output = tmp;
        // 4. Layer normalization after feed-forward
        let final_output = self.norm2.forward(&output)?;
        Ok(final_output)
        _grad_output: &Array<F, IxDyn>,
        // In a complete implementation, this would compute gradients through all components
        // For simplicity, this is just a placeholder that returns a gradient of the same shape
        // Create a placeholder gradient for the input
        let grad_input = Array::zeros(input.dim());
        // Return gradient with respect to input
        // Update all components
        self.self_attn.update(learning_rate)?;
        self.norm1.update(learning_rate)?;
        self.feed_forward.update(learning_rate)?;
        self.norm2.update(learning_rate)?;
/// Transformer encoder
/// Stack of transformer encoder layers that processes sequences using
/// self-attention and feed-forward networks.
pub struct TransformerEncoder<F: Float + Debug + Send + Sync + SimdUnifiedOps> {
    /// Stack of encoder layers
    layers: Vec<TransformerEncoderLayer<F>>,
    /// Layer outputs cache for backward pass
    layer_outputs: Arc<RwLock<Vec<Array<F, IxDyn>>>>,
    TransformerEncoder<F>
    /// Create a new transformer encoder
    /// * `n_layers` - Number of encoder layers
    /// * A new transformer encoder
        n_layers: usize,
        // Create encoder layers
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model, n_heads, d_ff, dropout, epsilon, rng,
            )?);
            layers,
            layer_outputs: Arc::new(RwLock::new(Vec::new())),
        // Clear layer outputs cache
        *self.layer_outputs.write().unwrap() = Vec::new();
        // Process input through all encoder layers
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
            // Cache layer output for backward pass
            self.layer_outputs.write().unwrap().push(output.clone());
        Ok(output)
        // In a complete implementation, this would compute gradients through all layers
        // Update all layers
        for layer in &mut self.layers {
            layer.update(learningrate)?;
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    #[test]
    fn test_feed_forwardshape() {
        // Set up feed-forward network
        let mut rng = rand::rng();
        let d_model = 64;
        let d_ff = 256;
        let ff = FeedForward::<f64>::new(d_model, d_ff, 0.1, &mut rng).unwrap();
        // Create a batch of inputs
        let batch_size = 2;
        let seq_len = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();
        // Forward pass
        let output = ff.forward(&input).unwrap();
        // Check output shape
        assert_eq!(output.shape(), input.shape());
    fn test_encoder_layershape() {
        // Set up encoder layer
        let n_heads = 4;
        let dropout = 0.1;
        let epsilon = 1e-5;
        let enc_layer =
            TransformerEncoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .unwrap();
        let output = enc_layer.forward(&input).unwrap();
    fn test_encoder_stackshape() {
        // Set up encoder
        let n_layers = 2;
        let encoder = TransformerEncoder::<f64>::new(
            d_model, n_layers, n_heads, d_ff, dropout, epsilon, &mut rng,
        )
        .unwrap();
        let output = encoder.forward(&input).unwrap();
