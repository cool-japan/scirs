//! BERT implementation
//!
//! BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based
//! model designed to pretrain deep bidirectional representations from unlabeled text.
//! Reference: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", Devlin et al. (2018)
//! <https://arxiv.org/abs/1810.04805>

use crate::error::{NeuralError, Result};
use crate::layers::{
    Dense, Dropout, Embedding, EmbeddingConfig, Layer, LayerNorm, MultiHeadAttention,
};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::SeedableRng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Configuration for a BERT model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size in feed-forward networks
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f64,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f64,
    /// Type vocabulary size (usually 2 for sentence pair tasks)
    pub type_vocab_size: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
}

impl BertConfig {
    /// Create a BERT-Base configuration
    pub fn bert_base_uncased() -> Self {
        Self {
            vocab_size: 30522,
            max_position_embeddings: 512,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
        }
    }

    /// Create a BERT-Large configuration
    pub fn bert_large_uncased() -> Self {
        Self {
            vocab_size: 30522,
            max_position_embeddings: 512,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
        }
    }

    /// Create a custom BERT configuration
    pub fn custom(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
    ) -> Self {
        Self {
            vocab_size,
            max_position_embeddings: 512,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size: hidden_size * 4,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
        }
    }
}

/// BERT embeddings combining token, position, and token type embeddings
struct BertEmbeddings<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static>
where
    F: SimdUnifiedOps,
{
    /// Token embeddings
    word_embeddings: Embedding<F>,
    /// Position embeddings
    position_embeddings: Embedding<F>,
    /// Token type embeddings
    token_type_embeddings: Embedding<F>,
    /// Layer normalization
    layer_norm: LayerNorm<F>,
    /// Dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertEmbeddings<F>
{
    fn clone(&self) -> Self {
        Self {
            word_embeddings: self.word_embeddings.clone(),
            position_embeddings: self.position_embeddings.clone(),
            token_type_embeddings: self.token_type_embeddings.clone(),
            layer_norm: self.layer_norm.clone(),
            dropout: self.dropout.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertEmbeddings<F>
{
    /// Create BERT embeddings
    pub fn new(config: &BertConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(EmbeddingConfig {
            num_embeddings: config.vocab_size,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
        })?;

        let position_embeddings = Embedding::new(EmbeddingConfig {
            num_embeddings: config.max_position_embeddings,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
        })?;

        let token_type_embeddings = Embedding::new(EmbeddingConfig {
            num_embeddings: config.type_vocab_size,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
        })?;

        let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([45; 32]);
        let layer_norm = LayerNorm::new(config.hidden_size, config.layer_norm_eps, &mut rng4)?;

        let mut rng5 = scirs2_core::random::rngs::SmallRng::from_seed([46; 32]);
        let dropout = Dropout::new(config.hidden_dropout_prob, &mut rng5)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertEmbeddings<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Input should be of shape [batch_size, seq_len] and contain token IDs
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, seq_len], got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];

        // Get word embeddings
        let inputs_embeds = self.word_embeddings.forward(input)?;

        // Create position IDs
        let mut position_ids = Array::zeros(IxDyn(&[batch_size, seq_len]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                position_ids[[b, s]] = F::from(s).expect("Failed to convert to float");
            }
        }

        // Get position embeddings
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Create token type IDs (all zeros for single sequence)
        let token_type_ids = Array::zeros(IxDyn(&[batch_size, seq_len]));

        // Get token type embeddings
        let token_type_embeds = self.token_type_embeddings.forward(&token_type_ids)?;

        // Combine embeddings
        let embeddings = &inputs_embeds + &position_embeds + &token_type_embeds;

        // Apply layer normalization
        let embeddings = self.layer_norm.forward(&embeddings)?;

        // Apply dropout
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.word_embeddings.update(learning_rate)?;
        self.position_embeddings.update(learning_rate)?;
        self.token_type_embeddings.update(learning_rate)?;
        self.layer_norm.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT self-attention layer
struct BertSelfAttention<
    F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// Multi-head attention layer
    attention: MultiHeadAttention<F>,
    /// Output dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertSelfAttention<F>
{
    fn clone(&self) -> Self {
        Self {
            attention: self.attention.clone(),
            dropout: self.dropout.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertSelfAttention<F>
{
    /// Create BERT self-attention layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let attn_config = crate::layers::AttentionConfig {
            num_heads: config.num_attention_heads,
            head_dim,
            dropout_prob: config.attention_probs_dropout_prob,
            causal: false,
            scale: None,
        };

        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([47; 32]);
        let attention = MultiHeadAttention::new(config.hidden_size, attn_config, &mut rng)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([48; 32]);
        let dropout = Dropout::new(config.hidden_dropout_prob, &mut rng2)?;

        Ok(Self { attention, dropout })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertSelfAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let attention_output = self.attention.forward(input)?;
        let attention_output = self.dropout.forward(&attention_output)?;
        Ok(attention_output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.attention.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT feed-forward network (intermediate + output)
struct BertFeedForward<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static>
where
    F: SimdUnifiedOps,
{
    /// Intermediate dense layer
    intermediate_dense: Dense<F>,
    /// Output dense layer
    output_dense: Dense<F>,
    /// Layer normalization
    layer_norm: LayerNorm<F>,
    /// Dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertFeedForward<F>
{
    fn clone(&self) -> Self {
        Self {
            intermediate_dense: self.intermediate_dense.clone(),
            output_dense: self.output_dense.clone(),
            layer_norm: self.layer_norm.clone(),
            dropout: self.dropout.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertFeedForward<F>
{
    /// Create BERT feed-forward layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([49; 32]);
        let intermediate_dense = Dense::new(
            config.hidden_size,
            config.intermediate_size,
            None,
            &mut rng1,
        )?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([50; 32]);
        let output_dense = Dense::new(
            config.intermediate_size,
            config.hidden_size,
            None,
            &mut rng2,
        )?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([51; 32]);
        let layer_norm = LayerNorm::new(config.hidden_size, config.layer_norm_eps, &mut rng3)?;

        let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([52; 32]);
        let dropout = Dropout::new(config.hidden_dropout_prob, &mut rng4)?;

        Ok(Self {
            intermediate_dense,
            output_dense,
            layer_norm,
            dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertFeedForward<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // The Dense layers operate on 2D `[rows, features]` tensors, so flatten
        // the 3D `[batch, seq, hidden]` activations to `[batch * seq, hidden]`
        // before applying them and restore the original shape afterwards.
        let original_shape = input.shape().to_vec();
        let needs_flatten = original_shape.len() > 2;
        let feature_dim = *original_shape
            .last()
            .ok_or_else(|| NeuralError::InferenceError("Empty feed-forward input".to_string()))?;

        let input_2d = if needs_flatten {
            let rows: usize = original_shape[..original_shape.len() - 1].iter().product();
            input
                .clone()
                .into_shape_with_order(IxDyn(&[rows, feature_dim]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!(
                        "Failed to flatten feed-forward input: {e}"
                    ))
                })?
        } else {
            input.clone()
        };

        // Intermediate layer with GELU activation
        let hidden = self.intermediate_dense.forward(&input_2d)?;
        let hidden = hidden.mapv(|v: F| {
            // GELU approximation
            let x3 = v * v * v;
            v * F::from(0.5).expect("Failed to convert constant to float")
                * (F::one()
                    + (v + F::from(0.044715).expect("Failed to convert constant to float") * x3)
                        .tanh())
        });

        // Output layer
        let mut output = self.output_dense.forward(&hidden)?;
        if needs_flatten {
            output = output
                .into_shape_with_order(IxDyn(&original_shape))
                .map_err(|e| {
                    NeuralError::InferenceError(format!(
                        "Failed to restore feed-forward output shape: {e}"
                    ))
                })?;
        }
        let output = self.dropout.forward(&output)?;

        // Add residual and layer norm
        let output = input + &output;
        let output = self.layer_norm.forward(&output)?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.intermediate_dense.update(learning_rate)?;
        self.output_dense.update(learning_rate)?;
        self.layer_norm.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT layer (attention + feed-forward)
struct BertLayer<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Self attention
    attention: BertSelfAttention<F>,
    /// Attention output layer norm
    attention_layer_norm: LayerNorm<F>,
    /// Feed-forward network
    feed_forward: BertFeedForward<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertLayer<F>
{
    fn clone(&self) -> Self {
        Self {
            attention: self.attention.clone(),
            attention_layer_norm: self.attention_layer_norm.clone(),
            feed_forward: self.feed_forward.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertLayer<F>
{
    /// Create BERT layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        let attention = BertSelfAttention::new(config)?;

        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([53; 32]);
        let attention_layer_norm =
            LayerNorm::new(config.hidden_size, config.layer_norm_eps, &mut rng)?;

        let feed_forward = BertFeedForward::new(config)?;

        Ok(Self {
            attention,
            attention_layer_norm,
            feed_forward,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Self-attention with residual and layer norm
        let attention_output = self.attention.forward(input)?;
        let attention_output = input + &attention_output;
        let attention_output = self.attention_layer_norm.forward(&attention_output)?;

        // Feed-forward with residual and layer norm
        let layer_output = self.feed_forward.forward(&attention_output)?;

        Ok(layer_output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.attention.update(learning_rate)?;
        self.attention_layer_norm.update(learning_rate)?;
        self.feed_forward.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT encoder
struct BertEncoder<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// BERT layers
    layers: Vec<BertLayer<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertEncoder<F>
{
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertEncoder<F>
{
    /// Create BERT encoder
    pub fn new(config: &BertConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config)?);
        }

        Ok(Self { layers })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertEncoder<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut hidden_states = input.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(learning_rate)?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT pooler (for classification tasks)
struct BertPooler<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// Dense layer
    dense: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertPooler<F>
{
    fn clone(&self) -> Self {
        Self {
            dense: self.dense.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertPooler<F>
{
    /// Create BERT pooler
    pub fn new(config: &BertConfig) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([54; 32]);
        let dense = Dense::new(config.hidden_size, config.hidden_size, None, &mut rng)?;

        Ok(Self { dense })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertPooler<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Take the first token ([CLS]) representation
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, seq_len, hidden_size], got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let hidden_size = shape[2];

        // Extract [CLS] token (first token)
        let mut cls_tokens = Array::zeros(IxDyn(&[batch_size, hidden_size]));
        for b in 0..batch_size {
            for i in 0..hidden_size {
                cls_tokens[[b, i]] = input[[b, 0, i]];
            }
        }

        // Apply dense layer
        let pooled_output = self.dense.forward(&cls_tokens)?;

        // Apply tanh activation
        let pooled_output = pooled_output.mapv(|x: F| x.tanh());

        Ok(pooled_output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.dense.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT model implementation
pub struct BertModel<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Embeddings layer
    embeddings: BertEmbeddings<F>,
    /// Encoder
    encoder: BertEncoder<F>,
    /// Pooler
    pooler: BertPooler<F>,
    /// Model configuration
    config: BertConfig,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for BertModel<F>
{
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            encoder: self.encoder.clone(),
            pooler: self.pooler.clone(),
            config: self.config.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    BertModel<F>
{
    /// Create a new BERT model
    pub fn new(config: BertConfig) -> Result<Self> {
        let embeddings = BertEmbeddings::new(&config)?;
        let encoder = BertEncoder::new(&config)?;
        let pooler = BertPooler::new(&config)?;

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            config,
        })
    }

    /// Create a BERT-Base-Uncased model
    pub fn bert_base_uncased() -> Result<Self> {
        let config = BertConfig::bert_base_uncased();
        Self::new(config)
    }

    /// Create a BERT-Large-Uncased model
    pub fn bert_large_uncased() -> Result<Self> {
        let config = BertConfig::bert_large_uncased();
        Self::new(config)
    }

    /// Create a custom BERT model
    pub fn custom(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
    ) -> Result<Self> {
        let config = BertConfig::custom(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
        );
        Self::new(config)
    }

    /// Get sequence output (last layer hidden states)
    pub fn get_sequence_output(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let embedding_output = self.embeddings.forward(input)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }

    /// Get pooled output (for classification tasks)
    pub fn get_pooled_output(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let sequence_output = self.get_sequence_output(input)?;
        let pooled_output = self.pooler.forward(&sequence_output)?;
        Ok(pooled_output)
    }

    /// Get the model configuration
    pub fn config(&self) -> &BertConfig {
        &self.config
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for BertModel<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // By default, return the full sequence output
        self.get_sequence_output(input)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.embeddings.update(learning_rate)?;
        self.encoder.update(learning_rate)?;
        self.pooler.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + ToPrimitive
            + FromPrimitive
            + 'static,
    > BertModel<F>
{
    /// Extract all named parameters in HuggingFace-compatible format.
    ///
    /// Parameter names mirror the official HuggingFace BERT naming:
    /// - `embeddings.word_embeddings.weight`
    /// - `embeddings.LayerNorm.weight`, `embeddings.LayerNorm.bias`
    /// - `encoder.layer.N.attention.self.query.weight`
    /// - `encoder.layer.N.attention.output.dense.weight`
    /// - `encoder.layer.N.intermediate.dense.weight`
    /// - `encoder.layer.N.output.dense.weight`
    /// - `pooler.dense.weight`, `pooler.dense.bias`
    pub fn extract_named_params(&self) -> Result<Vec<(String, Array<F, IxDyn>)>> {
        let mut result = Vec::new();

        // Embeddings
        for p in self.embeddings.word_embeddings.params().iter() {
            result.push(("embeddings.word_embeddings.weight".to_string(), p.clone()));
        }
        for p in self.embeddings.position_embeddings.params().iter() {
            result.push((
                "embeddings.position_embeddings.weight".to_string(),
                p.clone(),
            ));
        }
        for p in self.embeddings.token_type_embeddings.params().iter() {
            result.push((
                "embeddings.token_type_embeddings.weight".to_string(),
                p.clone(),
            ));
        }
        let ln_params = self.embeddings.layer_norm.params();
        if !ln_params.is_empty() {
            result.push((
                "embeddings.LayerNorm.weight".to_string(),
                ln_params[0].clone(),
            ));
        }
        if ln_params.len() >= 2 {
            result.push((
                "embeddings.LayerNorm.bias".to_string(),
                ln_params[1].clone(),
            ));
        }

        // Encoder layers
        for (layer_idx, bert_layer) in self.encoder.layers.iter().enumerate() {
            let prefix = format!("encoder.layer.{layer_idx}");

            // Self-attention: MultiHeadAttention has 4 params: w_query, w_key, w_value, w_output
            let attn_params = bert_layer.attention.attention.params();
            if attn_params.len() >= 4 {
                result.push((
                    format!("{prefix}.attention.self.query.weight"),
                    attn_params[0].clone(),
                ));
                result.push((
                    format!("{prefix}.attention.self.key.weight"),
                    attn_params[1].clone(),
                ));
                result.push((
                    format!("{prefix}.attention.self.value.weight"),
                    attn_params[2].clone(),
                ));
                result.push((
                    format!("{prefix}.attention.output.dense.weight"),
                    attn_params[3].clone(),
                ));
            } else if attn_params.len() == 3 {
                result.push((
                    format!("{prefix}.attention.self.query.weight"),
                    attn_params[0].clone(),
                ));
                result.push((
                    format!("{prefix}.attention.self.key.weight"),
                    attn_params[1].clone(),
                ));
                result.push((
                    format!("{prefix}.attention.self.value.weight"),
                    attn_params[2].clone(),
                ));
            }

            // Attention output layer norm
            let attn_ln_params = bert_layer.attention_layer_norm.params();
            if !attn_ln_params.is_empty() {
                result.push((
                    format!("{prefix}.attention.output.LayerNorm.weight"),
                    attn_ln_params[0].clone(),
                ));
            }
            if attn_ln_params.len() >= 2 {
                result.push((
                    format!("{prefix}.attention.output.LayerNorm.bias"),
                    attn_ln_params[1].clone(),
                ));
            }

            // Feed-forward intermediate dense
            let ff_inter_params = bert_layer.feed_forward.intermediate_dense.params();
            if !ff_inter_params.is_empty() {
                result.push((
                    format!("{prefix}.intermediate.dense.weight"),
                    ff_inter_params[0].clone(),
                ));
            }
            if ff_inter_params.len() >= 2 {
                result.push((
                    format!("{prefix}.intermediate.dense.bias"),
                    ff_inter_params[1].clone(),
                ));
            }

            // Feed-forward output dense
            let ff_out_params = bert_layer.feed_forward.output_dense.params();
            if !ff_out_params.is_empty() {
                result.push((
                    format!("{prefix}.output.dense.weight"),
                    ff_out_params[0].clone(),
                ));
            }
            if ff_out_params.len() >= 2 {
                result.push((
                    format!("{prefix}.output.dense.bias"),
                    ff_out_params[1].clone(),
                ));
            }

            // Feed-forward layer norm
            let ff_ln_params = bert_layer.feed_forward.layer_norm.params();
            if !ff_ln_params.is_empty() {
                result.push((
                    format!("{prefix}.output.LayerNorm.weight"),
                    ff_ln_params[0].clone(),
                ));
            }
            if ff_ln_params.len() >= 2 {
                result.push((
                    format!("{prefix}.output.LayerNorm.bias"),
                    ff_ln_params[1].clone(),
                ));
            }
        }

        // Pooler
        let pooler_params = self.pooler.dense.params();
        if !pooler_params.is_empty() {
            result.push(("pooler.dense.weight".to_string(), pooler_params[0].clone()));
        }
        if pooler_params.len() >= 2 {
            result.push(("pooler.dense.bias".to_string(), pooler_params[1].clone()));
        }

        Ok(result)
    }

    /// Load named parameters from a map (by name).
    ///
    /// Unknown parameter names are silently ignored, enabling graceful
    /// forward/backward compatibility between model versions.
    pub fn load_named_params(
        &mut self,
        params_map: &HashMap<String, Array<F, IxDyn>>,
    ) -> Result<()> {
        // Embeddings
        if let Some(p) = params_map.get("embeddings.word_embeddings.weight") {
            self.embeddings
                .word_embeddings
                .set_params(std::slice::from_ref(p))?;
        }
        if let Some(p) = params_map.get("embeddings.position_embeddings.weight") {
            self.embeddings
                .position_embeddings
                .set_params(std::slice::from_ref(p))?;
        }
        if let Some(p) = params_map.get("embeddings.token_type_embeddings.weight") {
            self.embeddings
                .token_type_embeddings
                .set_params(std::slice::from_ref(p))?;
        }
        {
            let mut ln_ps = Vec::new();
            if let Some(p) = params_map.get("embeddings.LayerNorm.weight") {
                ln_ps.push(p.clone());
            }
            if let Some(p) = params_map.get("embeddings.LayerNorm.bias") {
                ln_ps.push(p.clone());
            }
            if !ln_ps.is_empty() {
                self.embeddings.layer_norm.set_params(&ln_ps)?;
            }
        }

        // Encoder layers
        for (layer_idx, bert_layer) in self.encoder.layers.iter_mut().enumerate() {
            let prefix = format!("encoder.layer.{layer_idx}");

            // Self-attention weights
            let mut attn_ps = Vec::new();
            if let Some(p) = params_map.get(&format!("{prefix}.attention.self.query.weight")) {
                attn_ps.push(p.clone());
            }
            if let Some(p) = params_map.get(&format!("{prefix}.attention.self.key.weight")) {
                attn_ps.push(p.clone());
            }
            if let Some(p) = params_map.get(&format!("{prefix}.attention.self.value.weight")) {
                attn_ps.push(p.clone());
            }
            if let Some(p) = params_map.get(&format!("{prefix}.attention.output.dense.weight")) {
                attn_ps.push(p.clone());
            }
            if !attn_ps.is_empty() {
                bert_layer.attention.attention.set_params(&attn_ps)?;
            }

            // Attention output layer norm
            {
                let mut ln_ps = Vec::new();
                if let Some(p) =
                    params_map.get(&format!("{prefix}.attention.output.LayerNorm.weight"))
                {
                    ln_ps.push(p.clone());
                }
                if let Some(p) =
                    params_map.get(&format!("{prefix}.attention.output.LayerNorm.bias"))
                {
                    ln_ps.push(p.clone());
                }
                if !ln_ps.is_empty() {
                    bert_layer.attention_layer_norm.set_params(&ln_ps)?;
                }
            }

            // Feed-forward intermediate dense
            {
                let mut ff_ps = Vec::new();
                if let Some(p) = params_map.get(&format!("{prefix}.intermediate.dense.weight")) {
                    ff_ps.push(p.clone());
                }
                if let Some(p) = params_map.get(&format!("{prefix}.intermediate.dense.bias")) {
                    ff_ps.push(p.clone());
                }
                if !ff_ps.is_empty() {
                    bert_layer
                        .feed_forward
                        .intermediate_dense
                        .set_params(&ff_ps)?;
                }
            }

            // Feed-forward output dense
            {
                let mut ff_ps = Vec::new();
                if let Some(p) = params_map.get(&format!("{prefix}.output.dense.weight")) {
                    ff_ps.push(p.clone());
                }
                if let Some(p) = params_map.get(&format!("{prefix}.output.dense.bias")) {
                    ff_ps.push(p.clone());
                }
                if !ff_ps.is_empty() {
                    bert_layer.feed_forward.output_dense.set_params(&ff_ps)?;
                }
            }

            // Feed-forward layer norm
            {
                let mut ln_ps = Vec::new();
                if let Some(p) = params_map.get(&format!("{prefix}.output.LayerNorm.weight")) {
                    ln_ps.push(p.clone());
                }
                if let Some(p) = params_map.get(&format!("{prefix}.output.LayerNorm.bias")) {
                    ln_ps.push(p.clone());
                }
                if !ln_ps.is_empty() {
                    bert_layer.feed_forward.layer_norm.set_params(&ln_ps)?;
                }
            }
        }

        // Pooler
        {
            let mut ps = Vec::new();
            if let Some(p) = params_map.get("pooler.dense.weight") {
                ps.push(p.clone());
            }
            if let Some(p) = params_map.get("pooler.dense.bias") {
                ps.push(p.clone());
            }
            if !ps.is_empty() {
                self.pooler.dense.set_params(&ps)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_config_base() {
        let config = BertConfig::bert_base_uncased();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
    }

    #[test]
    fn test_bert_config_large() {
        let config = BertConfig::bert_large_uncased();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
    }

    #[test]
    fn test_bert_config_custom() {
        let config = BertConfig::custom(10000, 256, 4, 4);
        assert_eq!(config.vocab_size, 10000);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.intermediate_size, 1024); // 256 * 4
    }
}
