//! BERT implementation
//!
//! BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based
//! model designed to pretrain deep bidirectional representations from unlabeled text.
//!
//! Reference: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", Devlin et al. (2018)
//! https://arxiv.org/abs/1810.04805

use crate::error::{NeuralError, Result};
use crate::layers::{
    Dense, Dropout, Embedding, EmbeddingConfig, Layer, LayerNorm, MultiHeadAttention,
};
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt::Debug;

/// Configuration for a BERT model
#[derive(Debug, Clone)]
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
struct BertEmbeddings<F: Float + Debug + ScalarOperand + Send + Sync> {
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

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertEmbeddings<F> {
    /// Create BERT embeddings
    pub fn new(config: &BertConfig) -> Result<Self> {
        // Word embeddings
        let word_embedding_config = EmbeddingConfig {
            num_embeddings: config.vocab_size,
            embedding_dim: config.hidden_size,
            padding_idx: Some(0),
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };
        let word_embeddings = Embedding::new(word_embedding_config)?;

        // Position embeddings
        let position_embedding_config = EmbeddingConfig {
            num_embeddings: config.max_position_embeddings,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };
        let position_embeddings = Embedding::new(position_embedding_config)?;

        // Token type embeddings
        let token_type_embedding_config = EmbeddingConfig {
            num_embeddings: config.type_vocab_size,
            embedding_dim: config.hidden_size,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };
        let token_type_embeddings = Embedding::new(token_type_embedding_config)?;

        // Layer normalization
        let layer_norm_eps = F::from(config.layer_norm_eps).unwrap();
        let mut local_rng = SmallRng::seed_from_u64(42);
        let layer_norm = LayerNorm::new(
            config.hidden_size,
            layer_norm_eps.to_f64().unwrap(),
            &mut local_rng,
        )?;

        // Dropout
        let dropout_prob = config.hidden_dropout_prob;
        let mut rng_dropout = SmallRng::seed_from_u64(42);
        let dropout = Dropout::new(dropout_prob, &mut rng_dropout)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertEmbeddings<F> {
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
                position_ids[[b, s]] = F::from(s).unwrap();
            }
        }

        // Get position embeddings
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        // Create token type IDs (all zeros for now, in real implementation this would be input)
        let token_type_ids = Array::zeros(IxDyn(&[batch_size, seq_len]));

        // Get token type embeddings
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        // Combine embeddings
        let mut embeddings = inputs_embeds.clone();

        // Add position embeddings
        for i in 0..embeddings.len() {
            embeddings[i] = embeddings[i] + position_embeddings[i];
        }

        // Add token type embeddings
        for i in 0..embeddings.len() {
            embeddings[i] = embeddings[i] + token_type_embeddings[i];
        }

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
struct BertSelfAttention<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Number of attention heads
    #[allow(dead_code)]
    num_attention_heads: usize,
    /// Size of each attention head
    #[allow(dead_code)]
    attention_head_size: usize,
    /// Multi-head attention layer
    attention: MultiHeadAttention<F>,
    /// Output dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertSelfAttention<F> {
    /// Create BERT self-attention layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        // Multi-head attention
        // Import needed only in this scope
        use crate::layers::AttentionConfig;

        let mut rng_attn = SmallRng::seed_from_u64(42);
        let attention_config = AttentionConfig {
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            dropout_prob: config.attention_probs_dropout_prob,
            causal: false,
            scale: None,
        };

        let attention =
            MultiHeadAttention::new(config.hidden_size, attention_config, &mut rng_attn)?;

        // Output dropout
        let dropout_prob = config.hidden_dropout_prob;
        let mut rng_dropout = SmallRng::seed_from_u64(42);
        let dropout = Dropout::new(dropout_prob, &mut rng_dropout)?;

        Ok(Self {
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            attention,
            dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertSelfAttention<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Use multi-head attention component
        let attention_output = self.attention.forward(input)?;

        // Apply dropout
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

/// BERT feed-forward network
struct BertIntermediate<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Dense layer
    dense: Dense<F>,
    /// Activation function
    activation_fn: Box<dyn Fn(F) -> F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertIntermediate<F> {
    /// Create BERT intermediate layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        // Dense layer
        let mut rng_dense = SmallRng::seed_from_u64(42);
        let dense = Dense::new(
            config.hidden_size,
            config.intermediate_size,
            None,
            &mut rng_dense,
        )?;

        // Activation function
        let activation_fn: Box<dyn Fn(F) -> F> = match config.hidden_act.as_str() {
            "gelu" => Box::new(|x: F| {
                // Approximation of GELU
                let x3 = x * x * x;
                x * F::from(0.5).unwrap()
                    * (F::one() + (x + F::from(0.044715).unwrap() * x3).tanh())
            }),
            "relu" => Box::new(|x: F| x.max(F::zero())),
            _ => {
                return Err(NeuralError::InferenceError(format!(
                    "Unsupported activation function: {}",
                    config.hidden_act
                )))
            }
        };

        Ok(Self {
            dense,
            activation_fn,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertIntermediate<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Apply dense layer
        let hidden_states = self.dense.forward(input)?;

        // Apply activation function
        let hidden_states = hidden_states.mapv(|x| (self.activation_fn)(x));

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

/// BERT output layer
struct BertOutput<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Dense layer
    dense: Dense<F>,
    /// Layer normalization
    layer_norm: LayerNorm<F>,
    /// Dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertOutput<F> {
    /// Create BERT output layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        // Dense layer
        let mut rng_dense = SmallRng::seed_from_u64(42);
        let dense = Dense::new(
            config.intermediate_size,
            config.hidden_size,
            None,
            &mut rng_dense,
        )?;

        // Layer normalization
        let layer_norm_eps = config.layer_norm_eps;
        let mut rng_ln = SmallRng::seed_from_u64(42);
        let layer_norm = LayerNorm::new(config.hidden_size, layer_norm_eps, &mut rng_ln)?;

        // Dropout
        let dropout_prob = config.hidden_dropout_prob;
        let mut rng_dropout = SmallRng::seed_from_u64(42);
        let dropout = Dropout::new(dropout_prob, &mut rng_dropout)?;

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertOutput<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This is a simplified version without the residual connection
        // In a real implementation, we'd need to pass both the input and the residual

        // Apply dense layer
        let mut hidden_states = self.dense.forward(input)?;

        // Apply dropout
        hidden_states = self.dropout.forward(&hidden_states)?;

        // Apply layer normalization
        hidden_states = self.layer_norm.forward(&hidden_states)?;

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
        self.dense.update(learning_rate)?;
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

/// BERT attention block
struct BertAttention<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Self attention
    self_attention: BertSelfAttention<F>,
    /// Output layer
    output: BertOutput<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertAttention<F> {
    /// Create BERT attention block
    pub fn new(config: &BertConfig) -> Result<Self> {
        let self_attention = BertSelfAttention::new(config)?;
        let output = BertOutput::new(config)?;

        Ok(Self {
            self_attention,
            output,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertAttention<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let attention_output = self.self_attention.forward(input)?;
        let layer_output = self.output.forward(&attention_output)?;

        // In a real implementation, we would add the residual connection here
        // For now, we'll manually add the residual in the simplified output.forward()

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
        self.self_attention.update(learning_rate)?;
        self.output.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// BERT layer
struct BertLayer<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Attention block
    attention: BertAttention<F>,
    /// Intermediate layer
    intermediate: BertIntermediate<F>,
    /// Output layer
    output: BertOutput<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertLayer<F> {
    /// Create BERT layer
    pub fn new(config: &BertConfig) -> Result<Self> {
        let attention = BertAttention::new(config)?;
        let intermediate = BertIntermediate::new(config)?;
        let output = BertOutput::new(config)?;

        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertLayer<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let attention_output = self.attention.forward(input)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;

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
        self.intermediate.update(learning_rate)?;
        self.output.update(learning_rate)?;
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
struct BertEncoder<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// BERT layers
    layers: Vec<BertLayer<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertEncoder<F> {
    /// Create BERT encoder
    pub fn new(config: &BertConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config)?);
        }

        Ok(Self { layers })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertEncoder<F> {
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

/// BERT pooler
struct BertPooler<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Dense layer
    dense: Dense<F>,
    /// Activation function
    activation_fn: Box<dyn Fn(F) -> F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertPooler<F> {
    /// Create BERT pooler
    pub fn new(config: &BertConfig) -> Result<Self> {
        // Dense layer
        let mut rng_dense = SmallRng::seed_from_u64(42);
        let dense = Dense::new(config.hidden_size, config.hidden_size, None, &mut rng_dense)?;

        // Activation function (tanh)
        let activation_fn = Box::new(|x: F| x.tanh());

        Ok(Self {
            dense,
            activation_fn,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertPooler<F> {
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

        // Apply activation function
        let pooled_output = pooled_output.mapv(|x| (self.activation_fn)(x));

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
pub struct BertModel<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Embeddings layer
    embeddings: BertEmbeddings<F>,
    /// Encoder
    encoder: BertEncoder<F>,
    /// Pooler
    pooler: BertPooler<F>,
    /// Model configuration
    #[allow(dead_code)]
    config: BertConfig,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BertModel<F> {
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
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BertModel<F> {
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
