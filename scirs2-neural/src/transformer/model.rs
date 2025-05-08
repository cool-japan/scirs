//! Complete transformer model implementation
//!
//! This module provides a complete implementation of the transformer model
//! as described in "Attention Is All You Need" by Vaswani et al.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use crate::transformer::{TransformerDecoder, TransformerEncoder};
use crate::utils::{PositionalEncoding, PositionalEncodingFactory, PositionalEncodingType};
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::cell::RefCell;
use std::fmt::Debug;

/// Configuration for transformer models
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Embedding dimension
    pub d_model: usize,
    /// Number of encoder layers
    pub n_encoder_layers: usize,
    /// Number of decoder layers
    pub n_decoder_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Feed-forward network hidden dimension
    pub d_ff: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Type of positional encoding to use
    pub pos_encoding_type: PositionalEncodingType,
    /// Small constant for layer normalization
    pub epsilon: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            n_heads: 8,
            d_ff: 2048,
            max_seq_len: 512,
            dropout: 0.1,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            epsilon: 1e-5,
        }
    }
}

/// Complete transformer model with encoder and decoder
///
/// This implements the full transformer architecture from
/// "Attention Is All You Need", combining encoder and decoder
/// stacks with positional encoding.
pub struct Transformer<F: Float + Debug + Send + Sync> {
    /// Transformer encoder stack
    encoder: TransformerEncoder<F>,
    /// Transformer decoder stack
    decoder: TransformerDecoder<F>,
    /// Positional encoding for input embeddings
    pos_encoding: Box<dyn PositionalEncoding<F>>,
    /// Model configuration
    config: TransformerConfig,
    /// Encoder output cache for backward pass
    encoder_output_cache: RefCell<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Transformer<F> {
    /// Create a new transformer model
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new transformer model
    pub fn new<R: Rng>(config: TransformerConfig, rng: &mut R) -> Result<Self> {
        // Create encoder
        let encoder = TransformerEncoder::new(
            config.d_model,
            config.n_encoder_layers,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.epsilon,
            rng,
        )?;

        // Create decoder
        let decoder = TransformerDecoder::new(
            config.d_model,
            config.n_decoder_layers,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.epsilon,
            rng,
        )?;

        // Create positional encoding
        let pos_encoding = PositionalEncodingFactory::create(
            config.pos_encoding_type,
            config.max_seq_len,
            config.d_model,
        )?;

        Ok(Self {
            encoder,
            decoder,
            pos_encoding,
            config,
            encoder_output_cache: RefCell::new(None),
        })
    }

    /// Forward pass with encoder and decoder
    ///
    /// # Arguments
    ///
    /// * `src` - Source sequences [batch, src_len, d_model]
    /// * `tgt` - Target sequences [batch, tgt_len, d_model]
    ///
    /// # Returns
    ///
    /// * Output tensor [batch, tgt_len, d_model]
    pub fn forward_train(
        &self,
        src: &Array<F, IxDyn>,
        tgt: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Check source shape
        if src.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Source must have at least 3 dimensions [batch, src_len, features]".to_string(),
            ));
        }

        let src_shape = src.shape();
        let src_feat_dim = src_shape[src.ndim() - 1];

        if src_feat_dim != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of source ({}) must match d_model ({})",
                src_feat_dim, self.config.d_model
            )));
        }

        // Check target shape
        if tgt.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Target must have at least 3 dimensions [batch, tgt_len, features]".to_string(),
            ));
        }

        let tgt_shape = tgt.shape();
        let tgt_feat_dim = tgt_shape[tgt.ndim() - 1];

        if tgt_feat_dim != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of target ({}) must match d_model ({})",
                tgt_feat_dim, self.config.d_model
            )));
        }

        // Add positional encoding to source
        let src_pos = self.pos_encoding.forward(src)?;

        // Add positional encoding to target
        let tgt_pos = self.pos_encoding.forward(tgt)?;

        // Encode source
        let encoder_output = self.encoder.forward(&src_pos)?;
        self.encoder_output_cache
            .replace(Some(encoder_output.clone()));

        // Decode target with encoder output
        let decoder_output = self
            .decoder
            .forward_with_encoder(&tgt_pos, &encoder_output)?;

        Ok(decoder_output)
    }

    /// Forward pass for inference (without target)
    ///
    /// # Arguments
    ///
    /// * `src` - Source sequences [batch, src_len, d_model]
    /// * `tgt` - Target sequences so far [batch, tgt_len, d_model]
    ///
    /// # Returns
    ///
    /// * Output tensor [batch, tgt_len, d_model]
    pub fn forward_inference(
        &self,
        src: &Array<F, IxDyn>,
        tgt: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Check source shape
        if src.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Source must have at least 3 dimensions [batch, src_len, features]".to_string(),
            ));
        }

        let src_shape = src.shape();
        let src_feat_dim = src_shape[src.ndim() - 1];

        if src_feat_dim != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of source ({}) must match d_model ({})",
                src_feat_dim, self.config.d_model
            )));
        }

        // Check target shape
        if tgt.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Target must have at least 3 dimensions [batch, tgt_len, features]".to_string(),
            ));
        }

        let tgt_shape = tgt.shape();
        let tgt_feat_dim = tgt_shape[tgt.ndim() - 1];

        if tgt_feat_dim != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of target ({}) must match d_model ({})",
                tgt_feat_dim, self.config.d_model
            )));
        }

        // Add positional encoding to source
        let src_pos = self.pos_encoding.forward(src)?;

        // Add positional encoding to target
        let tgt_pos = self.pos_encoding.forward(tgt)?;

        // Encode source
        let encoder_output = self.encoder.forward(&src_pos)?;
        self.encoder_output_cache
            .replace(Some(encoder_output.clone()));

        // Decode target with encoder output
        let decoder_output = self
            .decoder
            .forward_with_encoder(&tgt_pos, &encoder_output)?;

        Ok(decoder_output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for Transformer<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This is a simplified forward pass that just applies the encoder
        // For full transformer functionality, use forward_train or forward_inference

        // Check input shape
        if input.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 3 dimensions [batch, seq_len, features]".to_string(),
            ));
        }

        let input_shape = input.shape();
        let feat_dim = input_shape[input.ndim() - 1];

        if feat_dim != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of input ({}) must match d_model ({})",
                feat_dim, self.config.d_model
            )));
        }

        // Add positional encoding
        let input_pos = self.pos_encoding.forward(input)?;

        // Apply encoder
        let encoder_output = self.encoder.forward(&input_pos)?;
        self.encoder_output_cache
            .replace(Some(encoder_output.clone()));

        Ok(encoder_output)
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
        self.encoder.update(learning_rate)?;
        self.decoder.update(learning_rate)?;
        self.pos_encoding.update(learning_rate)?;

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
    fn test_transformer_train() {
        // Set up transformer with small configuration for testing
        let mut rng = SmallRng::seed_from_u64(42);
        let config = TransformerConfig {
            d_model: 64,
            n_encoder_layers: 2,
            n_decoder_layers: 2,
            n_heads: 4,
            d_ff: 128,
            max_seq_len: 100,
            dropout: 0.1,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            epsilon: 1e-5,
        };

        let transformer = Transformer::<f64>::new(config, &mut rng).unwrap();

        // Create sample inputs
        let batch_size = 2;
        let src_len = 8;
        let tgt_len = 6;
        let d_model = 64;

        let src = Array3::<f64>::from_elem((batch_size, src_len, d_model), 0.1).into_dyn();
        let tgt = Array3::<f64>::from_elem((batch_size, tgt_len, d_model), 0.1).into_dyn();

        // Forward pass for training
        let output = transformer.forward_train(&src, &tgt).unwrap();

        // Check output shape
        assert_eq!(output.shape(), tgt.shape());
    }

    #[test]
    fn test_transformer_inference() {
        // Set up transformer with small configuration for testing
        let mut rng = SmallRng::seed_from_u64(42);
        let config = TransformerConfig {
            d_model: 64,
            n_encoder_layers: 1,
            n_decoder_layers: 1,
            n_heads: 4,
            d_ff: 128,
            max_seq_len: 100,
            dropout: 0.1,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            epsilon: 1e-5,
        };

        let transformer = Transformer::<f64>::new(config, &mut rng).unwrap();

        // Create sample inputs
        let batch_size = 2;
        let src_len = 8;
        let tgt_len = 1; // Single token for autoregressive generation
        let d_model = 64;

        let src = Array3::<f64>::from_elem((batch_size, src_len, d_model), 0.1).into_dyn();
        let tgt = Array3::<f64>::from_elem((batch_size, tgt_len, d_model), 0.1).into_dyn();

        // Forward pass for inference
        let output = transformer.forward_inference(&src, &tgt).unwrap();

        // Check output shape
        assert_eq!(output.shape(), tgt.shape());
    }

    #[test]
    fn test_encoder_only() {
        // Set up transformer with small configuration for testing
        let mut rng = SmallRng::seed_from_u64(42);
        let config = TransformerConfig {
            d_model: 64,
            n_encoder_layers: 1,
            n_decoder_layers: 1,
            n_heads: 4,
            d_ff: 128,
            max_seq_len: 100,
            dropout: 0.1,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            epsilon: 1e-5,
        };

        let transformer = Transformer::<f64>::new(config, &mut rng).unwrap();

        // Create sample inputs
        let batch_size = 2;
        let src_len = 8;
        let d_model = 64;

        let src = Array3::<f64>::from_elem((batch_size, src_len, d_model), 0.1).into_dyn();

        // Forward pass for encoder only
        let output = transformer.forward(&src).unwrap();

        // Check output shape
        assert_eq!(output.shape(), src.shape());
    }
}
