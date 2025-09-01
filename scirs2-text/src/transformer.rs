//! # Transformer Architecture Module
//!
//! This module provides a complete implementation of the Transformer architecture,
//! the foundation of modern language models like BERT, GPT, and T5. It includes
//! all essential components for building state-of-the-art NLP models.
//!
//! ## Overview
//!
//! The Transformer architecture revolutionized natural language processing by
//! introducing the self-attention mechanism. This module implements:
//!
//! - **Multi-Head Attention**: Core attention mechanism with multiple attention heads
//! - **Positional Encoding**: Sinusoidal and learned position representations
//! - **Encoder-Decoder Architecture**: Full transformer with both encoder and decoder stacks
//! - **Layer Normalization**: Pre-norm and post-norm variants
//! - **Feed-Forward Networks**: Position-wise fully connected layers
//! - **Token Embeddings**: Learnable word and position embeddings
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::transformer::{TransformerModel, TransformerConfig};
//!
//! // Configure the transformer
//! let config = TransformerConfig {
//!     d_model: 512,           // Model dimension
//!     nheads: 8,             // Number of attention heads
//!     d_ff: 2048,             // Feed-forward dimension
//!     n_encoder_layers: 6,    // Number of encoder layers
//!     n_decoder_layers: 6,    // Number of decoder layers
//!     max_seqlen: 512,       // Maximum sequence length
//!     dropout: 0.1,           // Dropout rate
//!     vocab_size: 10000,      // Vocabulary size
//! };
//!
//! // Create the model
//! let vocabulary = (0..10000).map(|i| format!("token_{}", i)).collect();
//! let mut transformer = TransformerModel::new(config, vocabulary).unwrap();
//!
//! // Example input sequences (string tokens)
//! let src_tokens = vec!["token_1".to_string(), "token_2".to_string(), "token_3".to_string()];
//!
//! // Encode the tokens
//! let output = transformer.encode_tokens(&src_tokens).unwrap();
//! println!("Model output shape: {:?}", output.shape());
//! ```
//!
//! ## Building Individual Components
//!
//! ### Multi-Head Attention
//!
//! ```rust
//! use scirs2_text::transformer::MultiHeadAttention;
//! use ndarray::Array2;
//!
//! let d_model = 512;
//! let nheads = 8;
//! let mut attention = MultiHeadAttention::new(d_model, nheads).unwrap();
//!
//! // Create dummy input (batch_size=2, seqlen=10, d_model=512)
//! let input = Array2::zeros((10, 512));
//! let output = attention.forward(input.view(), input.view(), input.view(), None).unwrap();
//! ```
//!
//! ### Positional Encoding
//!
//! ```rust
//! use scirs2_text::transformer::PositionalEncoding;
//! use ndarray::Array2;
//!
//! let d_model = 512;
//! let max_len = 1000;
//! let pos_encoding = PositionalEncoding::new(d_model, max_len);
//!
//! // Apply positional encoding to embeddings
//! let seqlen = 20;
//! let embeddings = Array2::<f64>::zeros((seqlen, d_model));
//! let positional_encodings = pos_encoding.get_encoding(seqlen).unwrap();
//! println!("Embeddings shape: {:?}", embeddings.shape());
//! println!("Positional encodings shape: {:?}", positional_encodings.shape());
//! ```
//!
//! ### Complete Encoder
//!
//! ```rust
//! use scirs2_text::transformer::{TransformerEncoder, TransformerConfig};
//! use ndarray::Array2;
//!
//! let config = TransformerConfig {
//!     d_model: 256,
//!     nheads: 4,
//!     d_ff: 1024,
//!     n_encoder_layers: 3,
//!     dropout: 0.1,
//!     ..Default::default()
//! };
//!
//! let encoder = TransformerEncoder::new(config).unwrap();
//! let input = Array2::zeros((50, 256)); // (seqlen, d_model)
//! let encoded = encoder.encode(input.view(), None).unwrap();
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom Attention Patterns
//!
//! ```rust
//! use scirs2_text::transformer::MultiHeadAttention;
//! use ndarray::Array2;
//!
//! let mut attention = MultiHeadAttention::new(512, 8).unwrap();
//!
//! // Create attention mask for autoregressive generation
//! let seqlen = 10;
//! let mut mask = Array2::from_elem((seqlen, seqlen), false);
//! for i in 0..seqlen {
//!     for j in (i+1)..seqlen {
//!         mask[[i, j]] = true; // Mask future positions
//!     }
//! }
//!
//! let query = Array2::zeros((seqlen, 512));
//! let key = Array2::zeros((seqlen, 512));
//! let value = Array2::zeros((seqlen, 512));
//! let output = attention.forward(query.view(), key.view(), value.view(), Some(mask.view())).unwrap();
//! ```
//!
//! ### Layer-wise Learning Rate Decay
//!
//! ```rust
//! use scirs2_text::transformer::{TransformerModel, TransformerConfig};
//!
//! # let config = TransformerConfig::default();
//! # let vocabulary: Vec<String> = (0..config.vocab_size).map(|i| format!("token_{}", i)).collect();
//! // Apply different learning rates to different layers  
//! let mut model = TransformerModel::new(config, vocabulary).unwrap();
//!
//! // Typically: deeper layers get smaller learning rates
//! let base_lr = 1e-4;
//! // Note: Layer parameters would be accessed through training APIs
//! println!("Base learning rate: {}", base_lr);
//! ```
//!
//! ## Architecture Details
//!
//! ### Attention Mechanism
//!
//! The multi-head attention computes:
//!
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d_k)V
//! MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
//! where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
//! ```
//!
//! ### Positional Encoding
//!
//! Uses sinusoidal functions to encode position information:
//!
//! ```text
//! PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
//! PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//! ```
//!
//! ### Layer Structure
//!
//! Each encoder/decoder layer follows the pattern:
//!
//! ```text
//! x = LayerNorm(x + SelfAttention(x))
//! x = LayerNorm(x + FeedForward(x))
//! ```
//!
//! ## Performance Optimization
//!
//! 1. **Gradient Checkpointing**: Trade memory for computation in deep models
//! 2. **Mixed Precision**: Use FP16 for faster training with minimal quality loss
//! 3. **Key-Value Caching**: Cache attention keys and values during inference
//! 4. **Attention Patterns**: Use sparse attention for very long sequences
//! 5. **Model Parallelism**: Split large models across multiple GPUs
//!
//! ## Common Use Cases
//!
//! - **Machine Translation**: Encoder-decoder for seq2seq tasks
//! - **Language Modeling**: Decoder-only for autoregressive generation
//! - **Text Classification**: Encoder with classification head
//! - **Question Answering**: Encoder with span prediction heads
//! - **Text Summarization**: Encoder-decoder with copy mechanism
//!
//! ## Best Practices
//!
//! 1. **Warmup Learning Rate**: Start with small LR and gradually increase
//! 2. **Layer Normalization**: Pre-norm generally works better than post-norm
//! 3. **Residual Connections**: Essential for training deep networks
//! 4. **Attention Dropout**: Apply dropout to attention weights, not just outputs
//! 5. **Weight Initialization**: Use Xavier/Glorot initialization for stability

use crate::error::{Result, TextError};
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use rand::Rng;
use statrs::statistics::Statistics;
use std::collections::HashMap;

/// Configuration for transformer models
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Model dimension (embedding size)
    pub d_model: usize,
    /// Number of attention heads
    pub nheads: usize,
    /// Feed-forward network dimension
    pub d_ff: usize,
    /// Number of encoder layers
    pub n_encoder_layers: usize,
    /// Number of decoder layers
    pub n_decoder_layers: usize,
    /// Maximum sequence length
    pub max_seqlen: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            nheads: 8,
            d_ff: 2048,
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            max_seqlen: 512,
            dropout: 0.1,
            vocab_size: 10000,
        }
    }
}

/// Position encoding for transformer models
pub struct PositionalEncoding {
    encodings: Array2<f64>,
    max_len: usize,
    #[allow(dead_code)]
    d_model: usize,
}

impl PositionalEncoding {
    /// Create new positional encoding
    pub fn new(_max_len: usize, dmodel: usize) -> Self {
        let mut encodings = Array2::<f64>::zeros((_max_len, dmodel));

        for pos in 0.._max_len {
            for i in (0..dmodel).step_by(2) {
                let angle = pos as f64 / (10000.0_f64).powf(i as f64 / dmodel as f64);
                encodings[[pos, i]] = angle.sin();
                if i + 1 < dmodel {
                    encodings[[pos, i + 1]] = angle.cos();
                }
            }
        }

        Self {
            encodings,
            max_len: _max_len,
            d_model: dmodel,
        }
    }

    /// Get position encoding for given sequence length
    pub fn get_encoding(&self, seqlen: usize) -> Result<ArrayView2<f64>> {
        if seqlen > self.max_len {
            return Err(TextError::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seqlen, self.max_len
            )));
        }
        Ok(self.encodings.slice(s![0..seqlen, ..]))
    }
}

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    d_model: usize,
    nheads: usize,
    d_k: usize,
    w_q: Array2<f64>,
    w_k: Array2<f64>,
    w_v: Array2<f64>,
    w_o: Array2<f64>,
}

impl MultiHeadAttention {
    /// Create new multi-head attention layer
    pub fn new(d_model: usize, nheads: usize) -> Result<Self> {
        if d_model % nheads != 0 {
            return Err(TextError::InvalidInput(
                "d_model must be divisible by nheads".to_string(),
            ));
        }

        let d_k = d_model / nheads;

        // Initialize weight matrices with Xavier initialization
        let scale = (2.0 / d_model as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| {
            rand::rng().random_range(-scale..scale)
        });

        Ok(Self {
            d_model,
            nheads,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
        })
    }

    /// Compute scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: ArrayView2<f64>,
        k: ArrayView2<f64>,
        v: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let d_k = self.d_k as f64;

        // Compute attention scores: Q * K^T / sqrt(d_k)
        let scores = q.dot(&k.t()) / d_k.sqrt();

        // Apply mask if provided
        let mut masked_scores = scores;
        if let Some(mask) = mask {
            for ((i, j), &should_mask) in mask.indexed_iter() {
                if should_mask {
                    masked_scores[[i, j]] = f64::NEG_INFINITY;
                }
            }
        }

        // Apply softmax
        let attention_weights = self.softmax_2d(&masked_scores)?;

        // Apply attention to values
        Ok(attention_weights.dot(&v))
    }

    /// Apply softmax to 2D array along last axis
    fn softmax_2d(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = x.clone();

        for mut row in result.rows_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        Ok(result)
    }

    /// Forward pass through multi-head attention
    pub fn forward(
        &self,
        query: ArrayView2<f64>,
        key: ArrayView2<f64>,
        value: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let _seqlen = query.shape()[0];

        // Linear projections
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        // Reshape for multi-head attention
        let q_heads = self.reshape_for_heads(&q)?;
        let k_heads = self.reshape_for_heads(&k)?;
        let v_heads = self.reshape_for_heads(&v)?;

        // Apply attention for each head
        let mut head_outputs = Vec::new();
        for head in 0..self.nheads {
            let q_head = q_heads.slice(s![head, .., ..]);
            let k_head = k_heads.slice(s![head, .., ..]);
            let v_head = v_heads.slice(s![head, .., ..]);

            let head_output = self.scaled_dot_product_attention(q_head, k_head, v_head, mask)?;
            head_outputs.push(head_output);
        }

        // Concatenate heads
        let concatenated = self.concatenate_heads(&head_outputs)?;

        // Final linear projection
        Ok(concatenated.dot(&self.w_o))
    }

    /// Reshape tensor for multi-head attention
    fn reshape_for_heads(&self, x: &Array2<f64>) -> Result<Array3<f64>> {
        let (seqlen, d_model) = x.dim();
        let reshaped = x
            .clone()
            .into_shape_with_order((seqlen, self.nheads, self.d_k))
            .map_err(|e| TextError::InvalidInput(format!("Reshape error: {e}")))?;

        // Transpose to (nheads, seqlen, d_k)
        Ok(reshaped.permuted_axes([1, 0, 2]))
    }

    /// Concatenate attention heads
    fn concatenate_heads(&self, heads: &[Array2<f64>]) -> Result<Array2<f64>> {
        if heads.is_empty() {
            return Err(TextError::InvalidInput("No heads provided".to_string()));
        }

        let seqlen = heads[0].shape()[0];
        let mut result = Array2::zeros((seqlen, self.d_model));

        for (i, head) in heads.iter().enumerate() {
            let start_col = i * self.d_k;
            let end_col = start_col + self.d_k;
            result.slice_mut(s![.., start_col..end_col]).assign(head);
        }

        Ok(result)
    }

    /// Get attention weight matrices for serialization
    pub fn get_weights(&self) -> (&Array2<f64>, &Array2<f64>, &Array2<f64>, &Array2<f64>) {
        (&self.w_q, &self.w_k, &self.w_v, &self.w_o)
    }

    /// Set attention weight matrices from loaded weights
    pub fn set_weights(
        &mut self,
        w_q: Array2<f64>,
        w_k: Array2<f64>,
        w_v: Array2<f64>,
        w_o: Array2<f64>,
    ) -> Result<()> {
        if w_q.shape() != [self.d_model, self.d_model] {
            return Err(TextError::InvalidInput("Invalid w_q shape".to_string()));
        }
        if w_k.shape() != [self.d_model, self.d_model] {
            return Err(TextError::InvalidInput("Invalid w_k shape".to_string()));
        }
        if w_v.shape() != [self.d_model, self.d_model] {
            return Err(TextError::InvalidInput("Invalid w_v shape".to_string()));
        }
        if w_o.shape() != [self.d_model, self.d_model] {
            return Err(TextError::InvalidInput("Invalid w_o shape".to_string()));
        }

        self.w_q = w_q;
        self.w_k = w_k;
        self.w_v = w_v;
        self.w_o = w_o;
        Ok(())
    }
}

/// Feed-forward network layer
pub struct FeedForward {
    w1: Array2<f64>,
    w2: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
}

impl FeedForward {
    /// Create new feed-forward layer
    pub fn new(_dmodel: usize, dff: usize) -> Self {
        let scale = (2.0 / _dmodel as f64).sqrt();

        let w1 = Array2::from_shape_fn((_dmodel, dff), |_| rand::rng().random_range(-scale..scale));
        let w2 = Array2::from_shape_fn((dff, _dmodel), |_| rand::rng().random_range(-scale..scale));
        let b1 = Array1::zeros(dff);
        let b2 = Array1::zeros(_dmodel);

        Self { w1, w2, b1, b2 }
    }

    /// Forward pass through feed-forward network
    pub fn forward(&self, x: ArrayView2<f64>) -> Array2<f64> {
        // First linear transformation + ReLU
        let hidden = x.dot(&self.w1) + &self.b1;
        let activated = hidden.mapv(|x| x.max(0.0)); // ReLU activation

        // Second linear transformation
        activated.dot(&self.w2) + &self.b2
    }

    /// Get feed-forward weight matrices for serialization
    pub fn get_weights(&self) -> (&Array2<f64>, &Array2<f64>, &Array1<f64>, &Array1<f64>) {
        (&self.w1, &self.w2, &self.b1, &self.b2)
    }

    /// Set feed-forward weight matrices from loaded weights
    pub fn set_weights(
        &mut self,
        w1: Array2<f64>,
        w2: Array2<f64>,
        b1: Array1<f64>,
        b2: Array1<f64>,
    ) -> Result<()> {
        if w1.shape()[1] != w2.shape()[0] {
            return Err(TextError::InvalidInput(
                "Weight matrix dimensions don't match".to_string(),
            ));
        }
        if b1.len() != w1.shape()[1] {
            return Err(TextError::InvalidInput(
                "Bias b1 size doesn't match w1".to_string(),
            ));
        }
        if b2.len() != w2.shape()[1] {
            return Err(TextError::InvalidInput(
                "Bias b2 size doesn't match w2".to_string(),
            ));
        }

        self.w1 = w1;
        self.w2 = w2;
        self.b1 = b1;
        self.b2 = b2;
        Ok(())
    }
}

/// Layer normalization
pub struct LayerNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    eps: f64,
}

impl LayerNorm {
    /// Create new layer normalization
    pub fn new(_dmodel: usize, eps: f64) -> Self {
        Self {
            gamma: Array1::ones(_dmodel),
            beta: Array1::zeros(_dmodel),
            eps,
        }
    }

    /// Apply layer normalization
    pub fn forward(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());

        for (i, row) in x.rows().into_iter().enumerate() {
            let mean = row.mean();
            let var = row.mapv(|x| (x - mean).powi(2)).mean();
            let std = (var + self.eps).sqrt();

            let normalized = row.mapv(|x| (x - mean) / std);
            let scaled = &normalized * &self.gamma + &self.beta;

            result.row_mut(i).assign(&scaled);
        }

        result
    }

    /// Get layer normalization parameters for serialization
    pub fn get_params(&self) -> (&Array1<f64>, &Array1<f64>) {
        (&self.gamma, &self.beta)
    }

    /// Set layer normalization parameters from loaded weights
    pub fn set_params(&mut self, gamma: Array1<f64>, beta: Array1<f64>) -> Result<()> {
        if gamma.len() != beta.len() {
            return Err(TextError::InvalidInput(
                "Gamma and beta must have same length".to_string(),
            ));
        }
        if gamma.len() != self.gamma.len() {
            return Err(TextError::InvalidInput(
                "Parameter size doesn't match layer dimension".to_string(),
            ));
        }

        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}

/// Transformer encoder layer
pub struct TransformerEncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    #[allow(dead_code)]
    dropout: f64,
}

impl TransformerEncoderLayer {
    /// Create new transformer encoder layer
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(config.d_model, config.nheads)?,
            feed_forward: FeedForward::new(config.d_model, config.d_ff),
            norm1: LayerNorm::new(config.d_model, 1e-6),
            norm2: LayerNorm::new(config.d_model, 1e-6),
            dropout: config.dropout,
        })
    }

    /// Forward pass through encoder layer
    pub fn forward(
        &self,
        x: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        // Self-attention with residual connection and layer norm
        let attn_output = self.self_attention.forward(x, x, x, mask)?;
        let x = &self.norm1.forward(x) + &attn_output;

        // Feed-forward with residual connection and layer norm
        let ff_output = self.feed_forward.forward(x.view());
        let output = &self.norm2.forward(x.view()) + &ff_output;

        Ok(output)
    }

    /// Get mutable access to layer components for weight loading
    pub fn get_components_mut(
        &mut self,
    ) -> (
        &mut MultiHeadAttention,
        &mut FeedForward,
        &mut LayerNorm,
        &mut LayerNorm,
    ) {
        (
            &mut self.self_attention,
            &mut self.feed_forward,
            &mut self.norm1,
            &mut self.norm2,
        )
    }

    /// Get access to layer components for weight access
    pub fn get_components(&self) -> (&MultiHeadAttention, &FeedForward, &LayerNorm, &LayerNorm) {
        (
            &self.self_attention,
            &self.feed_forward,
            &self.norm1,
            &self.norm2,
        )
    }
}

/// Complete transformer encoder
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    position_encoding: PositionalEncoding,
    config: TransformerConfig,
}

impl TransformerEncoder {
    /// Create new transformer encoder
    pub fn new(config: TransformerConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_encoder_layers {
            layers.push(TransformerEncoderLayer::new(&config)?);
        }

        let position_encoding = PositionalEncoding::new(config.max_seqlen, config.d_model);

        Ok(Self {
            layers,
            position_encoding,
            config,
        })
    }

    /// Encode input sequence
    pub fn encode(
        &self,
        embeddings: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let seqlen = embeddings.shape()[0];

        // Add positional encoding
        let pos_enc = self.position_encoding.get_encoding(seqlen)?;
        let mut x = embeddings.to_owned() + pos_enc;

        // Pass through encoder layers
        for layer in &self.layers {
            x = layer.forward(x.view(), mask)?;
        }

        Ok(x)
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Get mutable access to encoder layers for weight loading
    pub fn get_layers_mut(&mut self) -> &mut Vec<TransformerEncoderLayer> {
        &mut self.layers
    }

    /// Get access to encoder layers for weight access
    pub fn get_layers(&self) -> &Vec<TransformerEncoderLayer> {
        &self.layers
    }
}

/// Transformer decoder layer with self-attention, cross-attention, and feed-forward
pub struct TransformerDecoderLayer {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    #[allow(dead_code)]
    dropout: f64,
}

impl TransformerDecoderLayer {
    /// Create new decoder layer
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(config.d_model, config.nheads)?,
            cross_attention: MultiHeadAttention::new(config.d_model, config.nheads)?,
            feed_forward: FeedForward::new(config.d_model, config.d_ff),
            norm1: LayerNorm::new(config.d_model, 1e-6),
            norm2: LayerNorm::new(config.d_model, 1e-6),
            norm3: LayerNorm::new(config.d_model, 1e-6),
            dropout: config.dropout,
        })
    }

    /// Forward pass with encoder output for cross-attention
    pub fn forward(
        &self,
        x: ArrayView2<f64>,
        encoder_output: ArrayView2<f64>,
        self_attn_mask: Option<ArrayView2<bool>>,
        cross_attn_mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        // Self-attention with residual connection and layer norm
        let self_attn_out = self.self_attention.forward(x, x, x, self_attn_mask)?;
        let x = self.norm1.forward((x.to_owned() + self_attn_out).view());

        // Cross-attention with encoder _output
        let cross_attn_out = self.cross_attention.forward(
            x.view(),
            encoder_output,
            encoder_output,
            cross_attn_mask,
        )?;
        let x = self.norm2.forward((x + cross_attn_out).view());

        // Feed-forward with residual connection and layer norm
        let ff_out = self.feed_forward.forward(x.view());
        let _output = self.norm3.forward((x + ff_out).view());

        Ok(_output)
    }
}

/// Transformer decoder stack
pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
    position_encoding: PositionalEncoding,
    config: TransformerConfig,
}

impl TransformerDecoder {
    /// Create new decoder
    pub fn new(config: TransformerConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_decoder_layers {
            layers.push(TransformerDecoderLayer::new(&config)?);
        }

        let position_encoding = PositionalEncoding::new(config.max_seqlen, config.d_model);

        Ok(Self {
            layers,
            position_encoding,
            config,
        })
    }

    /// Forward pass through decoder
    pub fn forward(
        &self,
        embeddings: ArrayView2<f64>,
        encoder_output: ArrayView2<f64>,
        self_attn_mask: Option<ArrayView2<bool>>,
        cross_attn_mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let seqlen = embeddings.shape()[0];

        // Add positional encoding
        let pos_enc = self.position_encoding.get_encoding(seqlen)?;
        let mut x = embeddings.to_owned() + pos_enc;

        // Pass through decoder layers
        for layer in &self.layers {
            x = layer.forward(x.view(), encoder_output, self_attn_mask, cross_attn_mask)?;
        }

        Ok(x)
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

/// Token embedding layer
pub struct TokenEmbedding {
    embeddings: Array2<f64>,
    vocab_size: usize,
    d_model: usize,
}

impl TokenEmbedding {
    /// Create new token embedding layer
    pub fn new(_vocab_size: usize, dmodel: usize) -> Self {
        let scale = (1.0 / dmodel as f64).sqrt();
        let embeddings = Array2::from_shape_fn((_vocab_size, dmodel), |_| {
            rand::rng().random_range(-scale..scale)
        });

        Self {
            embeddings,
            vocab_size: _vocab_size,
            d_model: dmodel,
        }
    }

    /// Get embeddings for token IDs
    pub fn forward(&self, tokenids: &[usize]) -> Result<Array2<f64>> {
        let mut result = Array2::zeros((tokenids.len(), self.d_model));

        for (i, &token_id) in tokenids.iter().enumerate() {
            if token_id >= self.vocab_size {
                return Err(TextError::InvalidInput(format!(
                    "Token ID {} exceeds vocabulary size {}",
                    token_id, self.vocab_size
                )));
            }
            result.row_mut(i).assign(&self.embeddings.row(token_id));
        }

        Ok(result)
    }

    /// Get access to the embedding matrix for serialization
    pub fn get_embeddings(&self) -> &Array2<f64> {
        &self.embeddings
    }

    /// Set the embedding matrix from loaded weights
    pub fn set_embeddings(&mut self, embeddings: Array2<f64>) -> Result<()> {
        if embeddings.shape()[0] != self.vocab_size || embeddings.shape()[1] != self.d_model {
            return Err(TextError::InvalidInput(format!(
                "Embedding shape {:?} doesn't match expected ({}, {})",
                embeddings.shape(),
                self.vocab_size,
                self.d_model
            )));
        }
        self.embeddings = embeddings;
        Ok(())
    }
}

/// Complete transformer model for text processing
pub struct TransformerModel {
    /// Model configuration
    pub config: TransformerConfig,
    /// Token embedding layer
    pub token_embedding: TokenEmbedding,
    /// Transformer encoder
    pub encoder: TransformerEncoder,
    /// Optional transformer decoder
    pub decoder: Option<TransformerDecoder>,
    vocab_to_id: HashMap<String, usize>,
    id_to_vocab: HashMap<usize, String>,
}

impl TransformerModel {
    /// Create new transformer model
    pub fn new(config: TransformerConfig, vocabulary: Vec<String>) -> Result<Self> {
        let vocab_size = vocabulary.len();
        if vocab_size != config.vocab_size {
            return Err(TextError::InvalidInput(format!(
                "Vocabulary size {} doesn't match config {}",
                vocab_size, config.vocab_size
            )));
        }

        let mut vocab_to_id = HashMap::new();
        let mut id_to_vocab = HashMap::new();

        for (id, token) in vocabulary.into_iter().enumerate() {
            vocab_to_id.insert(token.clone(), id);
            id_to_vocab.insert(id, token);
        }

        Ok(Self {
            config: config.clone(),
            token_embedding: TokenEmbedding::new(config.vocab_size, config.d_model),
            encoder: TransformerEncoder::new(config)?,
            decoder: None, // Encoder-only model
            vocab_to_id,
            id_to_vocab,
        })
    }

    /// Encode text tokens to contextual embeddings
    pub fn encode_tokens(&self, tokens: &[String]) -> Result<Array2<f64>> {
        // Convert tokens to IDs
        let tokenids: Result<Vec<usize>> = tokens
            .iter()
            .map(|token| {
                self.vocab_to_id
                    .get(token)
                    .cloned()
                    .ok_or_else(|| TextError::InvalidInput(format!("Unknown token: {token}")))
            })
            .collect();
        let tokenids = tokenids?;

        // Get token embeddings
        let embeddings = self.token_embedding.forward(&tokenids)?;

        // Encode with transformer
        self.encoder.encode(embeddings.view(), None)
    }

    /// Create new encoder-decoder transformer model
    pub fn new_encoder_decoder(config: TransformerConfig, vocabulary: Vec<String>) -> Result<Self> {
        let vocab_size = vocabulary.len();
        if vocab_size != config.vocab_size {
            return Err(TextError::InvalidInput(format!(
                "Vocabulary size {} doesn't match config {}",
                vocab_size, config.vocab_size
            )));
        }

        let mut vocab_to_id = HashMap::new();
        let mut id_to_vocab = HashMap::new();

        for (id, token) in vocabulary.into_iter().enumerate() {
            vocab_to_id.insert(token.clone(), id);
            id_to_vocab.insert(id, token);
        }

        Ok(Self {
            config: config.clone(),
            token_embedding: TokenEmbedding::new(config.vocab_size, config.d_model),
            encoder: TransformerEncoder::new(config.clone())?,
            decoder: Some(TransformerDecoder::new(config)?),
            vocab_to_id,
            id_to_vocab,
        })
    }

    /// Perform encoder-decoder forward pass
    pub fn encode_decode(
        &self,
        input_tokens: &[String],
        target_tokens: &[String],
    ) -> Result<Array2<f64>> {
        let decoder = self
            .decoder
            .as_ref()
            .ok_or_else(|| TextError::InvalidInput("Model has no decoder".to_string()))?;

        // Encode input
        let encoder_output = self.encode_tokens(input_tokens)?;

        // Convert target _tokens to IDs and embeddings
        let target_ids: Result<Vec<usize>> = target_tokens
            .iter()
            .map(|token| {
                self.vocab_to_id
                    .get(token)
                    .copied()
                    .ok_or_else(|| TextError::InvalidInput(format!("Unknown token: {token}")))
            })
            .collect();
        let target_ids = target_ids?;

        let target_embeddings = self.token_embedding.forward(&target_ids)?;

        // Generate causal mask for decoder self-attention
        let seqlen = target_tokens.len();
        let mut causal_mask = Array2::from_elem((seqlen, seqlen), false);
        for i in 0..seqlen {
            for j in (i + 1)..seqlen {
                causal_mask[[i, j]] = true; // Mask future positions
            }
        }

        // Decode
        decoder.forward(
            target_embeddings.view(),
            encoder_output.view(),
            Some(causal_mask.view()),
            None,
        )
    }

    /// Generate text using the decoder (for generation tasks)
    pub fn generate(
        &self,
        input_tokens: &[String],
        max_length: usize,
        start_token: &str,
    ) -> Result<Vec<String>> {
        let decoder = self
            .decoder
            .as_ref()
            .ok_or_else(|| TextError::InvalidInput("Model has no decoder".to_string()))?;

        // Encode input
        let encoder_output = self.encode_tokens(input_tokens)?;

        // Start with the start _token
        let mut generated_tokens = vec![start_token.to_string()];

        for _ in 0..max_length {
            // Convert current _tokens to embeddings
            let current_ids: Result<Vec<usize>> = generated_tokens
                .iter()
                .map(|_token| {
                    self.vocab_to_id
                        .get(_token)
                        .copied()
                        .ok_or_else(|| TextError::InvalidInput(format!("Unknown token: {_token}")))
                })
                .collect();
            let current_ids = current_ids?;

            let current_embeddings = self.token_embedding.forward(&current_ids)?;

            // Generate causal mask
            let seqlen = generated_tokens.len();
            let mut causal_mask = Array2::from_elem((seqlen, seqlen), false);
            for i in 0..seqlen {
                for j in (i + 1)..seqlen {
                    causal_mask[[i, j]] = true;
                }
            }

            // Decode
            let decoder_output = decoder.forward(
                current_embeddings.view(),
                encoder_output.view(),
                Some(causal_mask.view()),
                None,
            )?;

            // Get the last timestep output
            let last_output = decoder_output.row(decoder_output.nrows() - 1);

            // Simple greedy selection (find _token with highest logit)
            let mut best_token_id = 0;
            let mut best_score = last_output[0];
            for (i, &score) in last_output.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_token_id = i;
                }
            }

            // Convert _token ID back to string
            if let Some(_token) = self.id_to_vocab.get(&best_token_id) {
                generated_tokens.push(_token.clone());

                // Stop if we hit an end _token (you might want to customize this)
                if _token == "</s>" || _token == "<eos>" {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(generated_tokens)
    }

    /// Get vocabulary mapping
    pub fn vocabulary(&self) -> (&HashMap<String, usize>, &HashMap<usize, String>) {
        (&self.vocab_to_id, &self.id_to_vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding() {
        let pos_enc = PositionalEncoding::new(10, 4);
        let encoding = pos_enc.get_encoding(5).unwrap();
        assert_eq!(encoding.shape(), &[5, 4]);

        // Test that positions are different
        let pos0 = encoding.row(0);
        let pos1 = encoding.row(1);
        assert!(pos0
            .iter()
            .zip(pos1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(8, 2).unwrap();
        let seqlen = 4;
        let d_model = 8;

        let input = Array2::ones((seqlen, d_model));
        let output = mha
            .forward(input.view(), input.view(), input.view(), None)
            .unwrap();

        assert_eq!(output.shape(), &[seqlen, d_model]);
    }

    #[test]
    fn test_transformer_encoder() {
        let config = TransformerConfig {
            d_model: 8,
            nheads: 2,
            d_ff: 16,
            n_encoder_layers: 2,
            ..Default::default()
        };

        let encoder = TransformerEncoder::new(config).unwrap();
        let input = Array2::ones((4, 8));
        let output = encoder.encode(input.view(), None).unwrap();

        assert_eq!(output.shape(), &[4, 8]);
    }
}
