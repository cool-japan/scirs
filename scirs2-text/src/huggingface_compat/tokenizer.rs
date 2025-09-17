//! Hugging Face compatible tokenizer implementations
//!
//! This module provides tokenizer wrappers that are compatible with
//! Hugging Face tokenizer formats and APIs.

use super::config::HfTokenizerConfig;
use crate::error::Result;
use crate::tokenize::Tokenizer;
use std::collections::HashMap;

/// Hugging Face compatible tokenizer wrapper
pub struct HfTokenizer {
    /// Underlying tokenizer
    tokenizer: Box<dyn Tokenizer>,
    /// Tokenizer configuration
    config: HfTokenizerConfig,
    /// Vocabulary mapping
    vocab: HashMap<String, usize>,
    /// Reverse vocabulary mapping
    reverse_vocab: HashMap<usize, String>,
}

impl HfTokenizer {
    /// Create new HF-compatible tokenizer
    pub fn new(tokenizer: Box<dyn Tokenizer>, config: HfTokenizerConfig) -> Self {
        // Create basic vocabulary (in practice, this would be loaded from files)
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add special tokens
        for (token_id, token) in config.special_tokens.keys().enumerate() {
            vocab.insert(token.clone(), token_id);
            reverse_vocab.insert(token_id, token.clone());
        }

        Self {
            tokenizer,
            config,
            vocab,
            reverse_vocab,
        }
    }

    /// Tokenize text with HF-compatible output
    pub fn encode(&self, text: &str, add_specialtokens: bool) -> Result<HfEncodedInput> {
        let mut tokens = self.tokenizer.tokenize(text)?;

        // Add special tokens if requested
        if add_specialtokens {
            if let Some(bos_token) = &self.config.bos_token {
                tokens.insert(0, bos_token.clone());
            }
            if let Some(eos_token) = &self.config.eos_token {
                tokens.push(eos_token.clone());
            }
        }

        // Convert tokens to IDs
        let input_ids: Vec<usize> = tokens
            .iter()
            .map(|token| {
                self.vocab
                    .get(token)
                    .copied()
                    .unwrap_or(self.vocab.get(&self.config.unk_token).copied().unwrap_or(0))
            })
            .collect();

        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask = vec![1; input_ids.len()];

        // Token type IDs (all 0 for single sentence)
        let token_type_ids = vec![0; input_ids.len()];

        Ok(HfEncodedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
            tokens,
        })
    }

    /// Batch encode multiple texts
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<HfEncodedInput>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize], skip_specialtokens: bool) -> Result<String> {
        let tokens: Vec<String> = token_ids
            .iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .filter(|token| {
                if skip_specialtokens {
                    !self.config.special_tokens.contains_key(*token)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        Ok(tokens.join(" "))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// HF-compatible encoded input format
#[derive(Debug, Clone)]
pub struct HfEncodedInput {
    /// Token IDs
    pub input_ids: Vec<usize>,
    /// Attention mask
    pub attention_mask: Vec<i32>,
    /// Token type IDs (for multi-sentence tasks)
    pub token_type_ids: Option<Vec<usize>>,
    /// Original tokens
    pub tokens: Vec<String>,
}
