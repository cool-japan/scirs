//! Hugging Face configuration structures and utilities
//!
//! This module provides configuration structures that map Hugging Face
//! model and tokenizer configurations to SciRS2 internal formats.

use crate::error::{Result, TextError};
use crate::transformer::TransformerConfig;
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde-support")]
use serde_json;

/// Hugging Face model configuration format
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct HfConfig {
    /// Model architecture type
    pub architectures: Vec<String>,
    /// Model type (e.g., "bert", "gpt2", "roberta")
    pub model_type: String,
    /// Number of attention heads
    pub num_attention_heads: Option<usize>,
    /// Hidden size
    pub hidden_size: Option<usize>,
    /// Intermediate size
    pub intermediate_size: Option<usize>,
    /// Number of hidden layers
    pub num_hidden_layers: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Maximum position embeddings
    pub max_position_embeddings: Option<usize>,
    /// Additional configuration parameters
    #[cfg(feature = "serde-support")]
    pub extraconfig: HashMap<String, serde_json::Value>,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["BertModel".to_string()],
            model_type: "bert".to_string(),
            num_attention_heads: Some(12),
            hidden_size: Some(768),
            intermediate_size: Some(3072),
            num_hidden_layers: Some(12),
            vocab_size: Some(30522),
            max_position_embeddings: Some(512),
            #[cfg(feature = "serde-support")]
            extraconfig: HashMap::new(),
        }
    }
}

impl HfConfig {
    /// Convert to SciRS2 transformer config
    pub fn to_transformer_config(&self) -> Result<TransformerConfig> {
        Ok(TransformerConfig {
            d_model: self.hidden_size.unwrap_or(768),
            nheads: self.num_attention_heads.unwrap_or(12),
            d_ff: self.intermediate_size.unwrap_or(3072),
            n_encoder_layers: self.num_hidden_layers.unwrap_or(12),
            n_decoder_layers: self.num_hidden_layers.unwrap_or(12),
            max_seqlen: self.max_position_embeddings.unwrap_or(512),
            dropout: 0.1,
            vocab_size: self.vocab_size.unwrap_or(30522),
        })
    }

    /// Create from transformer config
    pub fn from_transformer_config(config: &TransformerConfig) -> Self {
        Self {
            architectures: vec!["TransformerModel".to_string()],
            model_type: "transformer".to_string(),
            num_attention_heads: Some(config.nheads),
            hidden_size: Some(config.d_model),
            intermediate_size: Some(config.d_ff),
            num_hidden_layers: Some(config.n_encoder_layers),
            vocab_size: Some(config.vocab_size),
            max_position_embeddings: Some(config.max_seqlen),
            #[cfg(feature = "serde-support")]
            extraconfig: HashMap::new(),
        }
    }
}

/// Hugging Face tokenizer configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct HfTokenizerConfig {
    /// Tokenizer type
    pub tokenizer_type: String,
    /// Vocabulary file path
    pub vocab_file: Option<PathBuf>,
    /// Merges file path (for BPE)
    pub merges_file: Option<PathBuf>,
    /// Special tokens
    pub special_tokens: HashMap<String, String>,
    /// Maximum sequence length
    pub max_len: usize,
    /// Padding token
    pub pad_token: String,
    /// Unknown token
    pub unk_token: String,
    /// Start of sequence token
    pub bos_token: Option<String>,
    /// End of sequence token
    pub eos_token: Option<String>,
}

impl Default for HfTokenizerConfig {
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("[CLS]".to_string(), "cls_token".to_string());
        special_tokens.insert("[SEP]".to_string(), "sep_token".to_string());
        special_tokens.insert("[PAD]".to_string(), "pad_token".to_string());
        special_tokens.insert("[UNK]".to_string(), "unk_token".to_string());
        special_tokens.insert("[MASK]".to_string(), "mask_token".to_string());

        Self {
            tokenizer_type: "WordPiece".to_string(),
            vocab_file: None,
            merges_file: None,
            special_tokens,
            max_len: 512,
            pad_token: "[PAD]".to_string(),
            unk_token: "[UNK]".to_string(),
            bos_token: Some("[CLS]".to_string()),
            eos_token: Some("[SEP]".to_string()),
        }
    }
}
