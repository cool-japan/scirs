//! Complete transformer model implementation
//!
//! This module provides a complete implementation of the transformer model
//! as described in "Attention Is All You Need" by Vaswani et al.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use crate::transformer::{TransformerDecoder, TransformerEncoder};
use crate::utils::{PositionalEncoding, PositionalEncodingFactory, PositionalEncodingType};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

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
pub struct Transformer<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Transformer encoder stack
    encoder: TransformerEncoder<F>,
    /// Transformer decoder stack
    decoder: TransformerDecoder<F>,
    /// Positional encoding for input embeddings
    pos_encoding: Box<dyn PositionalEncoding<F> + Send + Sync>,
    /// Model configuration
    config: TransformerConfig,
    /// Encoder output cache for backward pass
    encoder_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for Transformer<F>
{
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            pos_encoding: self.pos_encoding.clone_box(),
            config: self.config.clone(),
            encoder_output_cache: Arc::new(RwLock::new(None)),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    Transformer<F>
{
    /// Create a new transformer model
    ///
    /// # Arguments
    /// * `config` - Transformer configuration
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
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
            config.d_model,
            config.max_seq_len,
            rng,
        );

        Ok(Self {
            encoder,
            decoder,
            pos_encoding,
            config,
            encoder_output_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Forward pass with encoder and decoder
    ///
    /// # Arguments
    /// * `src` - Source sequences [batch, src_len, d_model]
    /// * `tgt` - Target sequences [batch, tgt_len, d_model]
    ///
    /// # Returns
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
        *self.encoder_output_cache.write().expect("Operation failed") =
            Some(encoder_output.clone());

        // Decode target with encoder output
        let decoder_output = self
            .decoder
            .forward_with_encoder(&tgt_pos, &encoder_output)?;

        Ok(decoder_output)
    }

    /// Forward pass for inference (without computing new encoder output)
    ///
    /// # Arguments
    /// * `src` - Source sequences [batch, src_len, d_model]
    /// * `tgt` - Target sequences so far [batch, tgt_len, d_model]
    ///
    /// # Returns
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

        // Encode source (for inference, we might cache this for efficiency)
        let encoder_output = self.encoder.forward(&src_pos)?;
        *self.encoder_output_cache.write().expect("Operation failed") =
            Some(encoder_output.clone());

        // Decode target with encoder output
        let decoder_output = self
            .decoder
            .forward_with_encoder(&tgt_pos, &encoder_output)?;

        Ok(decoder_output)
    }

    /// Get the model configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Get a reference to the encoder
    pub fn encoder(&self) -> &TransformerEncoder<F> {
        &self.encoder
    }

    /// Get a reference to the decoder
    pub fn decoder(&self) -> &TransformerDecoder<F> {
        &self.decoder
    }

    /// Get a mutable reference to the encoder
    pub fn encoder_mut(&mut self) -> &mut TransformerEncoder<F> {
        &mut self.encoder
    }

    /// Get a mutable reference to the decoder
    pub fn decoder_mut(&mut self) -> &mut TransformerDecoder<F> {
        &mut self.decoder
    }

    /// Backpropagate through the encoder-decoder pass performed by
    /// [`Transformer::forward_train`].
    ///
    /// # Arguments
    /// * `src` - the source tensor passed to `forward_train`
    /// * `tgt` - the target tensor passed to `forward_train`
    /// * `grad_output` - gradient with respect to the decoder output
    ///
    /// # Returns
    /// `(grad_src, grad_tgt)` - gradients with respect to both inputs. The
    /// source gradient combines the encoder path with the decoder's
    /// cross-attention contribution.
    pub fn backward_train(
        &self,
        src: &Array<F, IxDyn>,
        tgt: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        // Presence of the cached encoder output proves a forward pass ran.
        if self
            .encoder_output_cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on the encoder output cache".to_string(),
                )
            })?
            .is_none()
        {
            return Err(NeuralError::InferenceError(
                "No cached encoder output. Call forward_train() first.".to_string(),
            ));
        }

        let src_pos = self.pos_encoding.forward(src)?;
        let tgt_pos = self.pos_encoding.forward(tgt)?;

        let (grad_tgt_pos, grad_encoder_output) =
            self.decoder.backward_with_encoder(&tgt_pos, grad_output)?;
        let grad_src_pos = self.encoder.backward(&src_pos, &grad_encoder_output)?;

        let grad_src = self.pos_encoding.backward(&grad_src_pos)?;
        let grad_tgt = self.pos_encoding.backward(&grad_tgt_pos)?;
        Ok((grad_src, grad_tgt))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Layer<F>
    for Transformer<F>
{
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

        Ok(encoder_output)
    }

    /// Backpropagate through the encoder-only pass performed by
    /// [`Layer::forward`] (positional encoding followed by the encoder stack).
    ///
    /// Use [`Transformer::backward_train`] for the full encoder-decoder path.
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let input_pos = self.pos_encoding.forward(input)?;
        let grad_input_pos = self.encoder.backward(&input_pos, grad_output)?;
        self.pos_encoding.backward(&grad_input_pos)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all components
        self.encoder.update(learning_rate)?;
        self.decoder.update(learning_rate)?;
        self.pos_encoding.update(learning_rate)?;

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = self.encoder.params();
        params.extend(self.decoder.params());
        params
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let mut grads = self.encoder.gradients();
        grads.extend(self.decoder.gradients());
        grads
    }

    fn layer_type(&self) -> &str {
        "Transformer"
    }

    fn parameter_count(&self) -> usize {
        self.encoder.parameter_count() + self.decoder.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_transformer_config_default() {
        let config = TransformerConfig::default();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_encoder_layers, 6);
        assert_eq!(config.n_decoder_layers, 6);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.d_ff, 2048);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn test_transformer_creation() {
        let mut rng = scirs2_core::random::rng();
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

        let result = Transformer::<f64>::new(config, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_train() {
        // Set up transformer with small configuration for testing
        let mut rng = scirs2_core::random::rng();
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

        let transformer = Transformer::<f64>::new(config, &mut rng).expect("Operation failed");

        // Create sample inputs
        let batch_size = 2;
        let src_len = 8;
        let tgt_len = 6;
        let d_model = 64;

        let src = Array3::<f64>::from_elem((batch_size, src_len, d_model), 0.1).into_dyn();
        let tgt = Array3::<f64>::from_elem((batch_size, tgt_len, d_model), 0.1).into_dyn();

        // Forward pass for training
        let output = transformer
            .forward_train(&src, &tgt)
            .expect("Operation failed");

        // Check output shape
        assert_eq!(output.shape(), tgt.shape());
    }

    #[test]
    fn test_transformer_inference() {
        let mut rng = scirs2_core::random::rng();
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

        let transformer = Transformer::<f64>::new(config, &mut rng).expect("Operation failed");

        let batch_size = 2;
        let src_len = 8;
        let tgt_len = 1; // Single token for autoregressive generation
        let d_model = 64;

        let src = Array3::<f64>::from_elem((batch_size, src_len, d_model), 0.1).into_dyn();
        let tgt = Array3::<f64>::from_elem((batch_size, tgt_len, d_model), 0.1).into_dyn();

        // Forward pass for inference
        let output = transformer
            .forward_inference(&src, &tgt)
            .expect("Operation failed");

        // Check output shape
        assert_eq!(output.shape(), tgt.shape());
    }

    #[test]
    fn test_encoder_only() {
        let mut rng = scirs2_core::random::rng();
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

        let transformer = Transformer::<f64>::new(config, &mut rng).expect("Operation failed");

        let batch_size = 2;
        let src_len = 8;
        let d_model = 64;

        let src = Array3::<f64>::from_elem((batch_size, src_len, d_model), 0.1).into_dyn();

        // Forward pass for encoder only
        let output = transformer.forward(&src).expect("Operation failed");

        // Check output shape
        assert_eq!(output.shape(), src.shape());
    }

    #[test]
    fn test_transformer_clone() {
        let mut rng = scirs2_core::random::rng();
        let config = TransformerConfig {
            d_model: 32,
            n_encoder_layers: 1,
            n_decoder_layers: 1,
            n_heads: 2,
            d_ff: 64,
            max_seq_len: 50,
            dropout: 0.1,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            epsilon: 1e-5,
        };

        let transformer = Transformer::<f64>::new(config, &mut rng).expect("Operation failed");
        let transformer_clone = transformer.clone();

        // Both should produce outputs
        let src = Array3::<f64>::from_elem((1, 4, 32), 0.1).into_dyn();
        let output1 = transformer.forward(&src).expect("Operation failed");
        let output2 = transformer_clone.forward(&src).expect("Operation failed");

        assert_eq!(output1.shape(), output2.shape());
    }

    #[test]
    fn test_transformer_invalid_input() {
        let mut rng = scirs2_core::random::rng();
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

        let transformer = Transformer::<f64>::new(config, &mut rng).expect("Operation failed");

        // Test with wrong dimensions (2D instead of 3D)
        let wrong_input = scirs2_core::ndarray::Array2::<f64>::from_elem((4, 64), 0.1).into_dyn();
        let result = transformer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong feature dimension
        let wrong_dim_input = Array3::<f64>::from_elem((2, 4, 32), 0.1).into_dyn(); // 32 != 64
        let result = transformer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_accessors() {
        let mut rng = scirs2_core::random::rng();
        let config = TransformerConfig {
            d_model: 64,
            n_encoder_layers: 2,
            n_decoder_layers: 3,
            n_heads: 4,
            d_ff: 128,
            max_seq_len: 100,
            dropout: 0.1,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            epsilon: 1e-5,
        };

        let transformer =
            Transformer::<f64>::new(config.clone(), &mut rng).expect("Operation failed");

        // Test config accessor
        assert_eq!(transformer.config().d_model, 64);
        assert_eq!(transformer.config().n_encoder_layers, 2);
        assert_eq!(transformer.config().n_decoder_layers, 3);

        // Test encoder/decoder accessors
        assert_eq!(transformer.encoder().num_layers(), 2);
        assert_eq!(transformer.decoder().num_layers(), 3);
    }
}
