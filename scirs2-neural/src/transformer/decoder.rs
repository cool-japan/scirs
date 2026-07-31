//! Transformer decoder implementation
//!
//! This module provides implementation of transformer decoder layers and blocks
//! as described in "Attention Is All You Need" by Vaswani et al.

use crate::error::{NeuralError, Result};
use crate::layers::{AttentionConfig, Layer, LayerNorm, MultiHeadAttention, SelfAttention};
use crate::transformer::encoder::FeedForward;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Transformer decoder layer
///
/// Implements a single layer of the transformer decoder as described in
/// "Attention Is All You Need" by Vaswani et al. It consists of masked multi-head
/// self-attention, multi-head cross-attention over encoder output, and a position-wise
/// feed-forward network, with residual connections and layer normalization.
pub struct TransformerDecoderLayer<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Masked multi-head self-attention layer
    self_attn: SelfAttention<F>,
    /// Layer normalization after self-attention
    norm1: LayerNorm<F>,
    /// Multi-head cross-attention layer
    cross_attn: MultiHeadAttention<F>,
    /// Layer normalization after cross-attention
    norm2: LayerNorm<F>,
    /// Feed-forward network
    feed_forward: FeedForward<F>,
    /// Layer normalization after feed-forward network
    norm3: LayerNorm<F>,
    /// Dropout rate for residual connections
    #[allow(dead_code)]
    dropout: F,
    /// Model embedding dimension
    d_model: usize,
    /// Self-attention output cache for backward pass
    self_attn_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Normalized self-attention output cache for backward pass
    norm1_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cross-attention output cache for backward pass
    cross_attn_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Normalized cross-attention output cache for backward pass
    norm2_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cache of the tensor fed into the final layer normalization
    /// (`norm_input + FFN(norm_input)`), needed by the backward pass
    norm3_input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Whether the last forward pass went through cross-attention
    /// (`forward_with_encoder`) or the encoder-free `Layer::forward` path
    used_cross_attention: Arc<RwLock<bool>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for TransformerDecoderLayer<F>
{
    fn clone(&self) -> Self {
        Self {
            self_attn: self.self_attn.clone(),
            norm1: self.norm1.clone(),
            cross_attn: self.cross_attn.clone(),
            norm2: self.norm2.clone(),
            feed_forward: self.feed_forward.clone(),
            norm3: self.norm3.clone(),
            dropout: self.dropout,
            d_model: self.d_model,
            self_attn_output_cache: Arc::new(RwLock::new(None)),
            norm1_output_cache: Arc::new(RwLock::new(None)),
            cross_attn_output_cache: Arc::new(RwLock::new(None)),
            norm2_output_cache: Arc::new(RwLock::new(None)),
            norm3_input_cache: Arc::new(RwLock::new(None)),
            used_cross_attention: Arc::new(RwLock::new(false)),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    TransformerDecoderLayer<F>
{
    /// Create a new transformer decoder layer
    ///
    /// # Arguments
    /// * `d_model` - Model embedding dimension
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `epsilon` - Small constant for layer normalization
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    /// * A new transformer decoder layer
    pub fn new<R: Rng>(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        epsilon: f64,
        rng: &mut R,
    ) -> Result<Self> {
        // Verify parameters
        if !d_model.is_multiple_of(n_heads) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                d_model, n_heads
            )));
        }

        // Calculate head dimension
        let head_dim = d_model / n_heads;

        // Create self-attention config (with causal masking)
        let self_attn_config = AttentionConfig {
            num_heads: n_heads,
            head_dim,
            dropout_prob: dropout,
            causal: true, // Use causal masking for self-attention in decoder
            scale: None,
        };

        // Create cross-attention config (no causal masking)
        let cross_attn_config = AttentionConfig {
            num_heads: n_heads,
            head_dim,
            dropout_prob: dropout,
            causal: false, // No causal masking for cross-attention
            scale: None,
        };

        // Create components
        let self_attn = SelfAttention::new(d_model, self_attn_config, rng)?;
        let norm1 = LayerNorm::new(d_model, epsilon, rng)?;
        let cross_attn = MultiHeadAttention::new(d_model, cross_attn_config, rng)?;
        let norm2 = LayerNorm::new(d_model, epsilon, rng)?;
        let feed_forward = FeedForward::new(d_model, d_ff, dropout, rng)?;
        let norm3 = LayerNorm::new(d_model, epsilon, rng)?;

        // Convert dropout rate
        let dropout = F::from(dropout).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert dropout rate".to_string())
        })?;

        Ok(Self {
            self_attn,
            norm1,
            cross_attn,
            norm2,
            feed_forward,
            norm3,
            dropout,
            d_model,
            self_attn_output_cache: Arc::new(RwLock::new(None)),
            norm1_output_cache: Arc::new(RwLock::new(None)),
            cross_attn_output_cache: Arc::new(RwLock::new(None)),
            norm2_output_cache: Arc::new(RwLock::new(None)),
            norm3_input_cache: Arc::new(RwLock::new(None)),
            used_cross_attention: Arc::new(RwLock::new(false)),
        })
    }

    /// Forward pass with encoder output
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, tgt_len, d_model]
    /// * `encoder_output` - Encoder output tensor [batch, src_len, d_model]
    ///
    /// # Returns
    /// * Output tensor [batch, tgt_len, d_model]
    pub fn forward_with_encoder(
        &self,
        input: &Array<F, IxDyn>,
        encoder_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Check input shape
        if input.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 3 dimensions [batch, tgt_len, features]".to_string(),
            ));
        }

        let input_shape = input.shape();
        let feat_dim = input_shape[input.ndim() - 1];
        if feat_dim != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of input ({}) must match d_model ({})",
                feat_dim, self.d_model
            )));
        }

        // Check encoder output shape
        if encoder_output.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Encoder output must have at least 3 dimensions [batch, src_len, features]"
                    .to_string(),
            ));
        }

        let encoder_shape = encoder_output.shape();
        let encoder_feat_dim = encoder_shape[encoder_output.ndim() - 1];
        if encoder_feat_dim != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of encoder output ({}) must match d_model ({})",
                encoder_feat_dim, self.d_model
            )));
        }

        // 1. Self-attention with residual connection
        let self_attn_output = self.self_attn.forward(input)?;
        *self
            .self_attn_output_cache
            .write()
            .expect("Operation failed") = Some(self_attn_output.clone());

        // Add residual connection (x + Sublayer(x))
        let self_attn_output_residual = input + &self_attn_output;

        // 2. Layer normalization after self-attention
        let norm1_output = self.norm1.forward(&self_attn_output_residual)?;
        *self.norm1_output_cache.write().expect("Operation failed") = Some(norm1_output.clone());

        // 3. Cross-attention: queries come from the decoder, keys and values
        // from the encoder output.
        let cross_attn_output = self
            .cross_attn
            .forward_with_kv(&norm1_output, encoder_output)?;
        *self
            .cross_attn_output_cache
            .write()
            .expect("Operation failed") = Some(cross_attn_output.clone());

        // Add residual connection
        let cross_attn_output_residual = &norm1_output + &cross_attn_output;

        // 4. Layer normalization after cross-attention
        let norm2_output = self.norm2.forward(&cross_attn_output_residual)?;
        *self.norm2_output_cache.write().expect("Operation failed") = Some(norm2_output.clone());

        // 5. Feed-forward network with residual connection
        let ff_output = self.feed_forward.forward(&norm2_output)?;

        // Add residual connection
        let output = &norm2_output + &ff_output;
        *self.norm3_input_cache.write().expect("Operation failed") = Some(output.clone());
        *self.used_cross_attention.write().expect("Operation failed") = true;

        // 6. Layer normalization after feed-forward
        let final_output = self.norm3.forward(&output)?;

        Ok(final_output)
    }

    /// Backpropagate through [`TransformerDecoderLayer::forward_with_encoder`].
    ///
    /// # Returns
    /// `(grad_input, grad_encoder_output)` - the gradients with respect to the
    /// decoder input and to the encoder output that fed cross-attention.
    pub fn backward_with_encoder(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let cached = |cell: &Arc<RwLock<Option<Array<F, IxDyn>>>>, what: &str| {
            cell.read()
                .map_err(|_| {
                    NeuralError::InferenceError(format!("Failed to read the {what} cache"))
                })?
                .clone()
                .ok_or_else(|| {
                    NeuralError::InferenceError(
                        "No cached values for backward pass. Call forward_with_encoder() first."
                            .to_string(),
                    )
                })
        };
        let self_attn_output = cached(&self.self_attn_output_cache, "self-attention")?;
        let norm1_output = cached(&self.norm1_output_cache, "norm1")?;
        let cross_attn_output = cached(&self.cross_attn_output_cache, "cross-attention")?;
        let norm2_output = cached(&self.norm2_output_cache, "norm2")?;
        let norm3_input = cached(&self.norm3_input_cache, "norm3 input")?;

        // 6. final layer normalization
        let grad_norm3_input = self.norm3.backward(&norm3_input, grad_output)?;

        // 5. feed-forward residual branch
        let grad_ff = self
            .feed_forward
            .backward(&norm2_output, &grad_norm3_input)?;
        let grad_norm2_output = &grad_norm3_input + &grad_ff;

        // 4. layer normalization after cross-attention
        let cross_residual = &norm1_output + &cross_attn_output;
        let grad_cross_residual = self.norm2.backward(&cross_residual, &grad_norm2_output)?;

        // 3. cross-attention: split the gradient between the decoder query path
        // and the encoder key/value path.
        let (grad_cross_query, grad_encoder) =
            self.cross_attn.backward_with_kv(&grad_cross_residual)?;
        let grad_norm1_output = &grad_cross_residual + &grad_cross_query;

        // 2. layer normalization after self-attention
        let self_residual = input + &self_attn_output;
        let grad_self_residual = self.norm1.backward(&self_residual, &grad_norm1_output)?;

        // 1. self-attention residual branch
        let grad_self_input = self.self_attn.backward(input, &grad_self_residual)?;
        Ok((&grad_self_residual + &grad_self_input, grad_encoder))
    }

    /// Get the model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Layer<F>
    for TransformerDecoderLayer<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This is a simplified forward pass that just applies self-attention and feed-forward
        // without cross-attention. For full decoder functionality, use forward_with_encoder.

        // Check input shape
        if input.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 3 dimensions [batch, seq_len, features]".to_string(),
            ));
        }

        let input_shape = input.shape();
        let feat_dim = input_shape[input.ndim() - 1];
        if feat_dim != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of input ({}) must match d_model ({})",
                feat_dim, self.d_model
            )));
        }

        // 1. Self-attention with residual connection
        let self_attn_output = self.self_attn.forward(input)?;
        *self
            .self_attn_output_cache
            .write()
            .expect("Operation failed") = Some(self_attn_output.clone());
        let self_attn_output_residual = input + &self_attn_output;

        // 2. Layer normalization after self-attention
        let norm1_output = self.norm1.forward(&self_attn_output_residual)?;
        *self.norm1_output_cache.write().expect("Operation failed") = Some(norm1_output.clone());

        // 3. Feed-forward network with residual connection
        let ff_output = self.feed_forward.forward(&norm1_output)?;
        let output = &norm1_output + &ff_output;
        *self.norm3_input_cache.write().expect("Operation failed") = Some(output.clone());
        *self.used_cross_attention.write().expect("Operation failed") = false;

        // 4. Apply final normalization
        let final_output = self.norm3.forward(&output)?;

        Ok(final_output)
    }

    /// Backpropagate through whichever forward path was last executed.
    ///
    /// After [`TransformerDecoderLayer::forward_with_encoder`] this returns the
    /// gradient with respect to the decoder input only; use
    /// [`TransformerDecoderLayer::backward_with_encoder`] when the encoder-side
    /// gradient is needed as well.
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let used_cross = *self.used_cross_attention.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read the decoder forward mode".to_string())
        })?;
        if used_cross {
            return Ok(self.backward_with_encoder(input, grad_output)?.0);
        }

        let cached = |cell: &Arc<RwLock<Option<Array<F, IxDyn>>>>, what: &str| {
            cell.read()
                .map_err(|_| {
                    NeuralError::InferenceError(format!("Failed to read the {what} cache"))
                })?
                .clone()
                .ok_or_else(|| {
                    NeuralError::InferenceError(
                        "No cached values for backward pass. Call forward() first.".to_string(),
                    )
                })
        };
        let self_attn_output = cached(&self.self_attn_output_cache, "self-attention")?;
        let norm1_output = cached(&self.norm1_output_cache, "norm1")?;
        let norm3_input = cached(&self.norm3_input_cache, "norm3 input")?;

        // 4. final layer normalization
        let grad_norm3_input = self.norm3.backward(&norm3_input, grad_output)?;

        // 3. feed-forward residual branch
        let grad_ff = self
            .feed_forward
            .backward(&norm1_output, &grad_norm3_input)?;
        let grad_norm1_output = &grad_norm3_input + &grad_ff;

        // 2. layer normalization after self-attention
        let self_residual = input + &self_attn_output;
        let grad_self_residual = self.norm1.backward(&self_residual, &grad_norm1_output)?;

        // 1. self-attention residual branch
        let grad_self_input = self.self_attn.backward(input, &grad_self_residual)?;
        Ok(&grad_self_residual + &grad_self_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all components
        self.self_attn.update(learning_rate)?;
        self.norm1.update(learning_rate)?;
        self.cross_attn.update(learning_rate)?;
        self.norm2.update(learning_rate)?;
        self.feed_forward.update(learning_rate)?;
        self.norm3.update(learning_rate)?;

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = self.self_attn.params();
        params.extend(self.norm1.params());
        params.extend(self.cross_attn.params());
        params.extend(self.norm2.params());
        params.extend(self.feed_forward.params());
        params.extend(self.norm3.params());
        params
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let mut grads = self.self_attn.gradients();
        grads.extend(self.norm1.gradients());
        grads.extend(self.cross_attn.gradients());
        grads.extend(self.norm2.gradients());
        grads.extend(self.feed_forward.gradients());
        grads.extend(self.norm3.gradients());
        grads
    }

    fn layer_type(&self) -> &str {
        "TransformerDecoderLayer"
    }

    fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.norm1.parameter_count()
            + self.cross_attn.parameter_count()
            + self.norm2.parameter_count()
            + self.feed_forward.parameter_count()
            + self.norm3.parameter_count()
    }
}

/// Transformer decoder
///
/// Stack of transformer decoder layers that processes target sequences using
/// masked self-attention, cross-attention with encoder output, and feed-forward networks.
pub struct TransformerDecoder<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Stack of decoder layers
    layers: Vec<TransformerDecoderLayer<F>>,
    /// Layer outputs cache for backward pass
    layer_outputs: Arc<RwLock<Vec<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for TransformerDecoder<F>
{
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            layer_outputs: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    TransformerDecoder<F>
{
    /// Create a new transformer decoder
    ///
    /// # Arguments
    /// * `d_model` - Model embedding dimension
    /// * `n_layers` - Number of decoder layers
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `epsilon` - Small constant for layer normalization
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    /// * A new transformer decoder
    pub fn new<R: Rng>(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        epsilon: f64,
        rng: &mut R,
    ) -> Result<Self> {
        // Create decoder layers
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerDecoderLayer::new(
                d_model, n_heads, d_ff, dropout, epsilon, rng,
            )?);
        }

        Ok(Self {
            layers,
            layer_outputs: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Forward pass with encoder output
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, tgt_len, d_model]
    /// * `encoder_output` - Encoder output tensor [batch, src_len, d_model]
    ///
    /// # Returns
    /// * Output tensor [batch, tgt_len, d_model]
    pub fn forward_with_encoder(
        &self,
        input: &Array<F, IxDyn>,
        encoder_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Clear layer outputs cache
        *self.layer_outputs.write().expect("Operation failed") = Vec::new();

        // Process input through all decoder layers
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward_with_encoder(&output, encoder_output)?;
            // Cache layer output for backward pass
            self.layer_outputs
                .write()
                .expect("Operation failed")
                .push(output.clone());
        }

        Ok(output)
    }

    /// Backpropagate through the stack executed by
    /// [`TransformerDecoder::forward_with_encoder`].
    ///
    /// # Returns
    /// `(grad_input, grad_encoder_output)`; the encoder-side gradient is the
    /// sum of every layer's cross-attention contribution.
    pub fn backward_with_encoder(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let outputs = self
            .layer_outputs
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on layer outputs".to_string(),
                )
            })?
            .clone();
        if outputs.len() != self.layers.len() {
            return Err(NeuralError::InferenceError(format!(
                "Cached {} layer outputs for {} decoder layers. Call forward_with_encoder() first.",
                outputs.len(),
                self.layers.len()
            )));
        }

        let mut grad = grad_output.clone();
        let mut grad_encoder: Option<Array<F, IxDyn>> = None;
        for (idx, layer) in self.layers.iter().enumerate().rev() {
            let layer_input = if idx == 0 { input } else { &outputs[idx - 1] };
            let (grad_in, grad_enc) = layer.backward_with_encoder(layer_input, &grad)?;
            grad = grad_in;
            grad_encoder = Some(match grad_encoder {
                Some(acc) => acc + grad_enc,
                None => grad_enc,
            });
        }
        let grad_encoder = grad_encoder.ok_or_else(|| {
            NeuralError::InferenceError(
                "A transformer decoder needs at least one layer".to_string(),
            )
        })?;
        Ok((grad, grad_encoder))
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get a reference to the layers
    pub fn layers(&self) -> &[TransformerDecoderLayer<F>] {
        &self.layers
    }

    /// Get a mutable reference to the layers
    pub fn layers_mut(&mut self) -> &mut [TransformerDecoderLayer<F>] {
        &mut self.layers
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Layer<F>
    for TransformerDecoder<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Clear layer outputs cache
        *self.layer_outputs.write().expect("Operation failed") = Vec::new();

        // Process input through all decoder layers (simplified, no cross-attention)
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
            // Cache layer output for backward pass
            self.layer_outputs
                .write()
                .expect("Operation failed")
                .push(output.clone());
        }

        Ok(output)
    }

    /// Backpropagate through the stack, feeding each layer the input it saw
    /// during the forward pass (the previous layer's cached output).
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let outputs = self
            .layer_outputs
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on layer outputs".to_string(),
                )
            })?
            .clone();
        if outputs.len() != self.layers.len() {
            return Err(NeuralError::InferenceError(format!(
                "Cached {} layer outputs for {} decoder layers. Call forward() first.",
                outputs.len(),
                self.layers.len()
            )));
        }

        let mut grad = grad_output.clone();
        for (idx, layer) in self.layers.iter().enumerate().rev() {
            let layer_input = if idx == 0 { input } else { &outputs[idx - 1] };
            grad = layer.backward(layer_input, &grad)?;
        }
        Ok(grad)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all layers
        for layer in &mut self.layers {
            layer.update(learning_rate)?;
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        self.layers.iter().flat_map(|l| l.gradients()).collect()
    }

    fn layer_type(&self) -> &str {
        "TransformerDecoder"
    }

    fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_decoder_layer_shape() {
        // Set up decoder layer
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let dec_layer =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .expect("Operation failed");

        // Create a batch of inputs
        let batch_size = 2;
        let tgt_seq_len = 8;
        let src_seq_len = 10;

        let decoder_input =
            Array3::<f64>::from_elem((batch_size, tgt_seq_len, d_model), 0.1).into_dyn();
        let encoder_output =
            Array3::<f64>::from_elem((batch_size, src_seq_len, d_model), 0.1).into_dyn();

        // Forward pass with encoder output
        let output = dec_layer
            .forward_with_encoder(&decoder_input, &encoder_output)
            .expect("Operation failed");

        // Check output shape
        assert_eq!(output.shape(), decoder_input.shape());
    }

    #[test]
    fn test_decoder_stack_shape() {
        // Set up decoder
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;
        let n_layers = 2;

        let decoder = TransformerDecoder::<f64>::new(
            d_model, n_layers, n_heads, d_ff, dropout, epsilon, &mut rng,
        )
        .expect("Operation failed");

        // Create a batch of inputs
        let batch_size = 2;
        let tgt_seq_len = 8;
        let src_seq_len = 10;

        let decoder_input =
            Array3::<f64>::from_elem((batch_size, tgt_seq_len, d_model), 0.1).into_dyn();
        let encoder_output =
            Array3::<f64>::from_elem((batch_size, src_seq_len, d_model), 0.1).into_dyn();

        // Forward pass with encoder output
        let output = decoder
            .forward_with_encoder(&decoder_input, &encoder_output)
            .expect("Operation failed");

        // Check output shape
        assert_eq!(output.shape(), decoder_input.shape());
    }

    #[test]
    fn test_decoder_causal_attention() {
        // Set up decoder layer with causal masking
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.0; // No dropout for deterministic test
        let epsilon = 1e-5;

        let dec_layer =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .expect("Operation failed");

        // Create a batch with clear position signals
        let batch_size = 1;
        let tgt_seq_len = 3;
        let src_seq_len = 3;

        // Create a target input where positions are clearly marked
        let mut decoder_input = Array3::<f64>::zeros((batch_size, tgt_seq_len, d_model));
        for i in 0..tgt_seq_len {
            let start_idx = i * 10;
            let end_idx = start_idx + 10;
            for j in start_idx..end_idx {
                if j < d_model {
                    decoder_input[[0, i, j]] = 1.0;
                }
            }
        }

        // Create a simple encoder output
        let encoder_output =
            Array3::<f64>::from_elem((batch_size, src_seq_len, d_model), 0.1).into_dyn();

        // Convert to dyn
        let decoder_input_dyn = decoder_input.into_dyn();

        // Forward pass
        let output = dec_layer
            .forward_with_encoder(&decoder_input_dyn, &encoder_output)
            .expect("Operation failed");

        // The output should have the right shape
        assert_eq!(output.shape(), decoder_input_dyn.shape());
    }

    #[test]
    fn test_decoder_simplified_forward() {
        // Test the simplified forward (without cross-attention)
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let dec_layer =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .expect("Operation failed");

        // Create input
        let batch_size = 2;
        let seq_len = 8;

        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        // Forward pass using Layer trait
        let output = dec_layer.forward(&input).expect("Operation failed");

        // Check output shape
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_decoder_clone() {
        // Test Clone implementation
        let mut rng = scirs2_core::random::rng();
        let d_model = 32;
        let n_heads = 2;
        let d_ff = 128;
        let dropout = 0.1;
        let epsilon = 1e-5;
        let n_layers = 2;

        let decoder = TransformerDecoder::<f64>::new(
            d_model, n_layers, n_heads, d_ff, dropout, epsilon, &mut rng,
        )
        .expect("Operation failed");

        // Clone the decoder
        let decoder_clone = decoder.clone();

        // Verify cloned decoder has same structure
        assert_eq!(decoder.num_layers(), decoder_clone.num_layers());

        // Test that both produce outputs
        let input = Array3::<f64>::from_elem((1, 4, d_model), 0.1).into_dyn();
        let output1 = decoder.forward(&input).expect("Operation failed");
        let output2 = decoder_clone.forward(&input).expect("Operation failed");

        // Both should have the same shape
        assert_eq!(output1.shape(), output2.shape());
    }

    #[test]
    fn test_decoder_invalid_input() {
        // Test error handling for invalid inputs
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let dec_layer =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .expect("Operation failed");

        // Test with wrong dimensions (2D instead of 3D)
        let wrong_input =
            scirs2_core::ndarray::Array2::<f64>::from_elem((4, d_model), 0.1).into_dyn();
        let result = dec_layer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong feature dimension
        let wrong_dim_input = Array3::<f64>::from_elem((2, 4, d_model + 10), 0.1).into_dyn();
        let result = dec_layer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_d_model_divisibility() {
        // Test that d_model must be divisible by n_heads
        let mut rng = scirs2_core::random::rng();
        let d_model = 65; // Not divisible by 4
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let result =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng);
        assert!(result.is_err());
    }
}
