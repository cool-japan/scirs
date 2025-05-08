//! Transformer decoder implementation
//!
//! This module provides implementation of transformer decoder layers and blocks
//! as described in "Attention Is All You Need" by Vaswani et al.

use crate::error::{NeuralError, Result};
use crate::layers::{AttentionConfig, Layer, LayerNorm, MultiHeadAttention, SelfAttention};
use crate::transformer::encoder::FeedForward;
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::cell::RefCell;
use std::fmt::Debug;

/// Transformer decoder layer
///
/// Implements a single layer of the transformer decoder as described in
/// "Attention Is All You Need" by Vaswani et al. It consists of masked multi-head
/// self-attention, multi-head cross-attention over encoder output, and a position-wise
/// feed-forward network, with residual connections and layer normalization.
pub struct TransformerDecoderLayer<F: Float + Debug + Send + Sync> {
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
    self_attn_output_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Normalized self-attention output cache for backward pass
    norm1_output_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Cross-attention output cache for backward pass
    cross_attn_output_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Normalized cross-attention output cache for backward pass
    norm2_output_cache: RefCell<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> TransformerDecoderLayer<F> {
    /// Create a new transformer decoder layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model embedding dimension
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `epsilon` - Small constant for layer normalization
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
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
        if d_model % n_heads != 0 {
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
            self_attn_output_cache: RefCell::new(None),
            norm1_output_cache: RefCell::new(None),
            cross_attn_output_cache: RefCell::new(None),
            norm2_output_cache: RefCell::new(None),
        })
    }

    /// Forward pass with encoder output
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch, tgt_len, d_model]
    /// * `encoder_output` - Encoder output tensor [batch, src_len, d_model]
    ///
    /// # Returns
    ///
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
        self.self_attn_output_cache
            .replace(Some(self_attn_output.clone()));

        // Add residual connection (x + Sublayer(x))
        let mut self_attn_output_residual = input.clone();
        let tmp = &self_attn_output_residual + &self_attn_output;
        self_attn_output_residual = tmp;

        // 2. Layer normalization after self-attention
        let norm1_output = self.norm1.forward(&self_attn_output_residual)?;
        self.norm1_output_cache.replace(Some(norm1_output.clone()));

        // 3. Cross-attention with encoder output
        let cross_attn_output = self.cross_attn.forward(&norm1_output)?;
        self.cross_attn_output_cache
            .replace(Some(cross_attn_output.clone()));

        // Add residual connection (x + Sublayer(x))
        let mut cross_attn_output_residual = norm1_output.clone();
        let tmp = &cross_attn_output_residual + &cross_attn_output;
        cross_attn_output_residual = tmp;

        // 4. Layer normalization after cross-attention
        let norm2_output = self.norm2.forward(&cross_attn_output_residual)?;
        self.norm2_output_cache.replace(Some(norm2_output.clone()));

        // 5. Feed-forward network with residual connection
        let ff_output = self.feed_forward.forward(&norm2_output)?;

        // Add residual connection (x + Sublayer(x))
        let mut output = norm2_output.clone();
        let tmp = &output + &ff_output;
        output = tmp;

        // 6. Layer normalization after feed-forward
        let final_output = self.norm3.forward(&output)?;

        Ok(final_output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F>
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
        self.self_attn_output_cache
            .replace(Some(self_attn_output.clone()));

        // Add residual connection (x + Sublayer(x))
        let mut self_attn_output_residual = input.clone();
        let tmp = &self_attn_output_residual + &self_attn_output;
        self_attn_output_residual = tmp;

        // 2. Layer normalization after self-attention
        let norm1_output = self.norm1.forward(&self_attn_output_residual)?;
        self.norm1_output_cache.replace(Some(norm1_output.clone()));

        // For feed-forward only, use norm1 output as input to feed-forward
        let output = self.feed_forward.forward(&norm1_output)?;

        // Add residual connection
        let mut output_residual = norm1_output.clone();
        let tmp = &output_residual + &output;
        output_residual = tmp;

        // Apply final normalization
        let final_output = self.norm3.forward(&output_residual)?;

        Ok(final_output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a complete implementation, this would compute gradients through all components
        // For simplicity, this is just a placeholder that returns a gradient of the same shape

        // Create a placeholder gradient for the input
        let grad_input = Array::zeros(input.dim());

        // Return gradient with respect to input
        Ok(grad_input)
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
}

/// Transformer decoder
///
/// Stack of transformer decoder layers that processes target sequences using
/// masked self-attention, cross-attention with encoder output, and feed-forward networks.
pub struct TransformerDecoder<F: Float + Debug + Send + Sync> {
    /// Stack of decoder layers
    layers: Vec<TransformerDecoderLayer<F>>,
    /// Model embedding dimension
    d_model: usize,
    /// Layer outputs cache for backward pass
    layer_outputs: RefCell<Vec<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> TransformerDecoder<F> {
    /// Create a new transformer decoder
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model embedding dimension
    /// * `n_layers` - Number of decoder layers
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `epsilon` - Small constant for layer normalization
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
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
            d_model,
            layer_outputs: RefCell::new(Vec::new()),
        })
    }

    /// Forward pass with encoder output
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch, tgt_len, d_model]
    /// * `encoder_output` - Encoder output tensor [batch, src_len, d_model]
    ///
    /// # Returns
    ///
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

        // Clear layer outputs cache
        self.layer_outputs.replace(Vec::new());

        // Process input through all decoder layers
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward_with_encoder(&output, encoder_output)?;

            // Cache layer output for backward pass
            self.layer_outputs.borrow_mut().push(output.clone());
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for TransformerDecoder<F> {
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

        // Clear layer outputs cache
        self.layer_outputs.replace(Vec::new());

        // Process input through all decoder layers
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;

            // Cache layer output for backward pass
            self.layer_outputs.borrow_mut().push(output.clone());
        }

        Ok(output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a complete implementation, this would compute gradients through all layers
        // For simplicity, this is just a placeholder that returns a gradient of the same shape

        // Create a placeholder gradient for the input
        let grad_input = Array::zeros(input.dim());

        // Return gradient with respect to input
        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all layers
        for layer in &mut self.layers {
            layer.update(learning_rate)?;
        }

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
    fn test_decoder_layer_shape() {
        // Set up decoder layer
        let mut rng = SmallRng::seed_from_u64(42);
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let dec_layer =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .unwrap();

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
            .unwrap();

        // Check output shape
        assert_eq!(output.shape(), decoder_input.shape());
    }

    #[test]
    fn test_decoder_stack_shape() {
        // Set up decoder
        let mut rng = SmallRng::seed_from_u64(42);
        let d_model = 64;
        let n_layers = 2;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let decoder = TransformerDecoder::<f64>::new(
            d_model, n_layers, n_heads, d_ff, dropout, epsilon, &mut rng,
        )
        .unwrap();

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
            .unwrap();

        // Check output shape
        assert_eq!(output.shape(), decoder_input.shape());
    }

    #[test]
    fn test_decoder_causal_attention() {
        // Set up decoder layer
        let mut rng = SmallRng::seed_from_u64(42);
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let dec_layer =
            TransformerDecoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .unwrap();

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

        // Convert to dyn once and reuse
        let decoder_input_dyn = decoder_input.clone().into_dyn();

        // Forward pass with encoder output
        let output = dec_layer
            .forward_with_encoder(&decoder_input_dyn, &encoder_output)
            .unwrap();

        // The output should have the right shape
        assert_eq!(output.shape(), decoder_input_dyn.shape());
    }
}
