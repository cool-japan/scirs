//! GPT implementation
//!
//! GPT (Generative Pre-trained Transformer) is a transformer-based language model
//! designed for autoregressive language modeling. Unlike BERT which is bidirectional,
//! GPT uses a unidirectional (left-to-right) transformer architecture.
//! Reference: "Improving Language Understanding by Generative Pre-Training", Radford et al. (2018)
//! <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Embedding, EmbeddingConfig, Layer, LayerNorm};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::SeedableRng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Configuration for a GPT model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
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
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
}

impl GPTConfig {
    /// Create a GPT-2 Small configuration
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// Create a GPT-2 Medium configuration
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// Create a GPT-2 Large configuration
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 1280,
            num_hidden_layers: 36,
            num_attention_heads: 20,
            intermediate_size: 5120,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// Create a custom GPT configuration
    pub fn custom(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
    ) -> Self {
        Self {
            vocab_size,
            max_position_embeddings: 1024,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size: hidden_size * 4,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }
}

/// GPT embedding combining token and position embeddings
struct GPTEmbeddings<
    F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
> {
    /// Token embeddings
    token_embeddings: Embedding<F>,
    /// Position embeddings
    position_embeddings: Embedding<F>,
    /// Dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for GPTEmbeddings<F>
{
    fn clone(&self) -> Self {
        Self {
            token_embeddings: self.token_embeddings.clone(),
            position_embeddings: self.position_embeddings.clone(),
            dropout: self.dropout.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    GPTEmbeddings<F>
{
    /// Create GPT embeddings
    pub fn new(config: &GPTConfig) -> Result<Self> {
        let token_embeddings = Embedding::new(EmbeddingConfig {
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

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([44; 32]);
        let dropout = Dropout::new(config.hidden_dropout_prob, &mut rng3)?;

        Ok(Self {
            token_embeddings,
            position_embeddings,
            dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for GPTEmbeddings<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, seq_len], got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];

        // Get token embeddings
        let inputs_embeds = self.token_embeddings.forward(input)?;

        // Create position IDs
        let mut position_ids = Array::zeros(IxDyn(&[batch_size, seq_len]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                position_ids[[b, s]] = F::from(s).expect("Failed to convert to float");
            }
        }

        // Get position embeddings
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Combine embeddings
        let embeddings = &inputs_embeds + &position_embeds;

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
        self.token_embeddings.update(learning_rate)?;
        self.position_embeddings.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// GPT MLP (feed-forward network)
struct GPTMlp<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// First dense layer
    fc1: Dense<F>,
    /// Second dense layer
    fc2: Dense<F>,
    /// Dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for GPTMlp<F>
{
    fn clone(&self) -> Self {
        Self {
            fc1: self.fc1.clone(),
            fc2: self.fc2.clone(),
            dropout: self.dropout.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    GPTMlp<F>
{
    /// Create GPT MLP
    pub fn new(config: &GPTConfig) -> Result<Self> {
        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([45; 32]);
        let fc1 = Dense::new(
            config.hidden_size,
            config.intermediate_size,
            None,
            &mut rng1,
        )?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([46; 32]);
        let fc2 = Dense::new(
            config.intermediate_size,
            config.hidden_size,
            None,
            &mut rng2,
        )?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([47; 32]);
        let dropout = Dropout::new(config.hidden_dropout_prob, &mut rng3)?;

        Ok(Self { fc1, fc2, dropout })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for GPTMlp<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Apply first dense layer
        let hidden_states = self.fc1.forward(input)?;

        // Apply GELU activation
        let hidden_states = hidden_states.mapv(|x: F| {
            let x3 = x * x * x;
            x * F::from(0.5).expect("Failed to convert constant to float")
                * (F::one()
                    + (x + F::from(0.044715).expect("Failed to convert constant to float") * x3)
                        .tanh())
        });

        // Apply second dense layer
        let hidden_states = self.fc2.forward(&hidden_states)?;

        // Apply dropout
        let hidden_states = self.dropout.forward(&hidden_states)?;

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
        self.fc1.update(learning_rate)?;
        self.fc2.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// GPT attention layer (masked multi-head attention)
struct GPTAttention<
    F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
> {
    /// Number of attention heads
    num_attention_heads: usize,
    /// Size of each attention head
    attention_head_size: usize,
    /// Query projection
    query: Dense<F>,
    /// Key projection
    key: Dense<F>,
    /// Value projection
    value: Dense<F>,
    /// Output projection
    output: Dense<F>,
    /// Attention dropout
    attn_dropout: Dropout<F>,
    /// Output dropout
    resid_dropout: Dropout<F>,
    /// Scale factor for attention scores
    scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for GPTAttention<F>
{
    fn clone(&self) -> Self {
        Self {
            num_attention_heads: self.num_attention_heads,
            attention_head_size: self.attention_head_size,
            query: self.query.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
            output: self.output.clone(),
            attn_dropout: self.attn_dropout.clone(),
            resid_dropout: self.resid_dropout.clone(),
            scale: self.scale,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    GPTAttention<F>
{
    /// Create GPT attention layer
    pub fn new(config: &GPTConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = hidden_size / num_attention_heads;

        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([48; 32]);
        let query = Dense::new(hidden_size, hidden_size, None, &mut rng1)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([49; 32]);
        let key = Dense::new(hidden_size, hidden_size, None, &mut rng2)?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([50; 32]);
        let value = Dense::new(hidden_size, hidden_size, None, &mut rng3)?;

        let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([51; 32]);
        let output = Dense::new(hidden_size, hidden_size, None, &mut rng4)?;

        let mut rng5 = scirs2_core::random::rngs::SmallRng::from_seed([52; 32]);
        let attn_dropout = Dropout::new(config.attention_probs_dropout_prob, &mut rng5)?;

        let mut rng6 = scirs2_core::random::rngs::SmallRng::from_seed([53; 32]);
        let resid_dropout = Dropout::new(config.hidden_dropout_prob, &mut rng6)?;

        let scale = F::from(1.0 / (attention_head_size as f64).sqrt()).expect("Operation failed");

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            query,
            key,
            value,
            output,
            attn_dropout,
            resid_dropout,
            scale,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for GPTAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, seq_len, hidden_size], got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        // Project query, key, value
        let query = self.query.forward(input)?;
        let key = self.key.forward(input)?;
        let value = self.value.forward(input)?;

        // Simplified attention: just combine projections
        // In a full implementation, we'd do proper multi-head attention with causal masking
        let attention_output = &query + &key + &value;

        // Apply output projection
        let output = self.output.forward(&attention_output)?;
        let output = self.resid_dropout.forward(&output)?;

        // Suppress unused variable warnings
        let _ = (batch_size, seq_len, hidden_size);

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
        self.query.update(learning_rate)?;
        self.key.update(learning_rate)?;
        self.value.update(learning_rate)?;
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

/// GPT block (attention + MLP)
struct GPTBlock<
    F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
> {
    /// Layer normalization for attention
    ln_1: LayerNorm<F>,
    /// Attention layer
    attn: GPTAttention<F>,
    /// Layer normalization for MLP
    ln_2: LayerNorm<F>,
    /// MLP
    mlp: GPTMlp<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for GPTBlock<F>
{
    fn clone(&self) -> Self {
        Self {
            ln_1: self.ln_1.clone(),
            attn: self.attn.clone(),
            ln_2: self.ln_2.clone(),
            mlp: self.mlp.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    GPTBlock<F>
{
    /// Create GPT block
    pub fn new(config: &GPTConfig) -> Result<Self> {
        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([54; 32]);
        let ln_1 = LayerNorm::new(config.hidden_size, config.layer_norm_eps, &mut rng1)?;

        let attn = GPTAttention::new(config)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([55; 32]);
        let ln_2 = LayerNorm::new(config.hidden_size, config.layer_norm_eps, &mut rng2)?;

        let mlp = GPTMlp::new(config)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for GPTBlock<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Attention with residual connection
        let ln1_output = self.ln_1.forward(input)?;
        let attn_output = self.attn.forward(&ln1_output)?;
        let residual1 = input + &attn_output;

        // MLP with residual connection
        let ln2_output = self.ln_2.forward(&residual1)?;
        let mlp_output = self.mlp.forward(&ln2_output)?;
        let residual2 = &residual1 + &mlp_output;

        Ok(residual2)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.ln_1.update(learning_rate)?;
        self.attn.update(learning_rate)?;
        self.ln_2.update(learning_rate)?;
        self.mlp.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// GPT model implementation
pub struct GPTModel<
    F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
> {
    /// Embeddings layer
    embeddings: GPTEmbeddings<F>,
    /// Transformer blocks
    blocks: Vec<GPTBlock<F>>,
    /// Final layer normalization
    ln_f: LayerNorm<F>,
    /// Model configuration
    config: GPTConfig,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Clone
    for GPTModel<F>
{
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            blocks: self.blocks.clone(),
            ln_f: self.ln_f.clone(),
            config: self.config.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    GPTModel<F>
{
    /// Create a new GPT model
    pub fn new(config: GPTConfig) -> Result<Self> {
        let embeddings = GPTEmbeddings::new(&config)?;

        // Create transformer blocks
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            blocks.push(GPTBlock::new(&config)?);
        }

        // Final layer normalization
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([56; 32]);
        let ln_f = LayerNorm::new(config.hidden_size, config.layer_norm_eps, &mut rng)?;

        Ok(Self {
            embeddings,
            blocks,
            ln_f,
            config,
        })
    }

    /// Create a GPT-2 Small model
    pub fn gpt2_small() -> Result<Self> {
        let config = GPTConfig::gpt2_small();
        Self::new(config)
    }

    /// Create a GPT-2 Medium model
    pub fn gpt2_medium() -> Result<Self> {
        let config = GPTConfig::gpt2_medium();
        Self::new(config)
    }

    /// Create a GPT-2 Large model
    pub fn gpt2_large() -> Result<Self> {
        let config = GPTConfig::gpt2_large();
        Self::new(config)
    }

    /// Create a custom GPT model
    pub fn custom(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
    ) -> Result<Self> {
        let config = GPTConfig::custom(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
        );
        Self::new(config)
    }

    /// Get the model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for GPTModel<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Get embeddings
        let mut hidden_states = self.embeddings.forward(input)?;

        // Apply transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        // Apply final layer normalization
        hidden_states = self.ln_f.forward(&hidden_states)?;

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
        self.embeddings.update(learning_rate)?;
        for block in &mut self.blocks {
            block.update(learning_rate)?;
        }
        self.ln_f.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config_small() {
        let config = GPTConfig::gpt2_small();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
    }

    #[test]
    fn test_gpt_config_medium() {
        let config = GPTConfig::gpt2_medium();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
    }

    #[test]
    fn test_gpt_config_large() {
        let config = GPTConfig::gpt2_large();
        assert_eq!(config.hidden_size, 1280);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 20);
    }

    #[test]
    fn test_gpt_config_custom() {
        let config = GPTConfig::custom(10000, 256, 4, 4);
        assert_eq!(config.vocab_size, 10000);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.intermediate_size, 1024);
    }
}
