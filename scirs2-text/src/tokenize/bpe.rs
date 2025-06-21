//! Byte Pair Encoding (BPE) tokenizer implementation
//!
//! This module provides a BPE tokenizer which can learn and apply
//! subword tokenization based on the most frequent byte pairs.

use crate::error::{Result, TextError};
use crate::tokenize::Tokenizer;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// A pair of tokens
type TokenPair = (String, String);

/// A vocabulary for BPE tokenization
#[derive(Clone)]
pub struct BpeVocabulary {
    /// Token to ID mapping
    pub token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    pub id_to_token: HashMap<usize, String>,
    /// Merge rules (token pair -> merged token)
    pub merges: HashMap<TokenPair, String>,
}

impl fmt::Debug for BpeVocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BpeVocabulary")
            .field("vocab_size", &self.token_to_id.len())
            .field("num_merges", &self.merges.len())
            .finish()
    }
}

impl BpeVocabulary {
    /// Create a new empty BPE vocabulary
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: HashMap::new(),
        }
    }

    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }

        let id = self.token_to_id.len();
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }

    /// Add a merge rule to the vocabulary
    pub fn add_merge(&mut self, pair: TokenPair, merged: String) {
        self.merges.insert(pair, merged);
    }

    /// Save the vocabulary to a file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        // Write the vocabulary size
        writeln!(writer, "{}", self.token_to_id.len())
            .map_err(|e| TextError::IoError(e.to_string()))?;

        // Write the tokens and their IDs
        for (token, id) in &self.token_to_id {
            writeln!(writer, "{}\t{}", token, id).map_err(|e| TextError::IoError(e.to_string()))?;
        }

        // Write the number of merges
        writeln!(writer, "{}", self.merges.len()).map_err(|e| TextError::IoError(e.to_string()))?;

        // Write the merge rules
        for ((first, second), merged) in &self.merges {
            writeln!(writer, "{}\t{}\t{}", first, second, merged)
                .map_err(|e| TextError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    /// Load a vocabulary from a file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .map_err(|e| TextError::IoError(e.to_string()))?;

        let mut lines = content.lines();

        // Read the vocabulary size
        let vocab_size: usize = lines
            .next()
            .ok_or_else(|| TextError::IoError("Unexpected end of file".to_string()))?
            .parse()
            .map_err(|e| TextError::IoError(format!("Invalid vocabulary size: {}", e)))?;

        let mut vocabulary = Self::new();

        // Read the tokens and their IDs
        for _ in 0..vocab_size {
            let line = lines
                .next()
                .ok_or_else(|| TextError::IoError("Unexpected end of file".to_string()))?;
            let parts: Vec<&str> = line.split('\t').collect();

            if parts.len() != 2 {
                return Err(TextError::IoError(format!(
                    "Invalid vocabulary entry: {}",
                    line
                )));
            }

            let token = parts[0].to_string();
            let id: usize = parts[1]
                .parse()
                .map_err(|e| TextError::IoError(format!("Invalid token ID: {}", e)))?;

            vocabulary.token_to_id.insert(token.clone(), id);
            vocabulary.id_to_token.insert(id, token);
        }

        // Read the number of merges
        let num_merges: usize = lines
            .next()
            .ok_or_else(|| TextError::IoError("Unexpected end of file".to_string()))?
            .parse()
            .map_err(|e| TextError::IoError(format!("Invalid number of merges: {}", e)))?;

        // Read the merge rules
        for _ in 0..num_merges {
            let line = lines
                .next()
                .ok_or_else(|| TextError::IoError("Unexpected end of file".to_string()))?;
            let parts: Vec<&str> = line.split('\t').collect();

            if parts.len() != 3 {
                return Err(TextError::IoError(format!("Invalid merge rule: {}", line)));
            }

            let first = parts[0].to_string();
            let second = parts[1].to_string();
            let merged = parts[2].to_string();

            vocabulary.merges.insert((first, second), merged);
        }

        Ok(vocabulary)
    }
}

impl Default for BpeVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for BPE tokenizer
#[derive(Debug, Clone)]
pub struct BpeConfig {
    /// The maximum vocabulary size
    pub vocab_size: usize,
    /// The minimum frequency for a token to be included in the vocabulary
    pub min_frequency: usize,
    /// Special tokens to add to the vocabulary
    pub special_tokens: Vec<String>,
    /// Whether to treat characters as the base tokens
    pub character_level: bool,
    /// Whether to lowercase the input text
    pub lowercase: bool,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            special_tokens: vec![],
            character_level: true,
            lowercase: true,
        }
    }
}

/// A Byte Pair Encoding (BPE) tokenizer
///
/// BPE is a subword tokenization algorithm that iteratively merges the most
/// frequent pairs of tokens (bytes or characters) to form new tokens.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Tokenizer configuration
    config: BpeConfig,
    /// The vocabulary for the tokenizer
    vocabulary: Option<BpeVocabulary>,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer with the given configuration
    pub fn new(config: BpeConfig) -> Self {
        Self {
            config,
            vocabulary: Some(BpeVocabulary::new()),
        }
    }

    /// Create a new BPE tokenizer with default configuration
    pub fn with_defaults() -> Self {
        Self::new(BpeConfig::default())
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        match &self.vocabulary {
            Some(vocab) => vocab.token_to_id.len(),
            None => 0,
        }
    }

    /// Check if the tokenizer has a vocabulary
    pub fn has_vocabulary(&self) -> bool {
        self.vocabulary.is_some()
    }

    /// Get a reference to the tokenizer's vocabulary
    pub fn vocabulary(&self) -> Option<&BpeVocabulary> {
        self.vocabulary.as_ref()
    }

    /// Set the tokenizer's vocabulary
    pub fn set_vocabulary(&mut self, vocabulary: BpeVocabulary) {
        self.vocabulary = Some(vocabulary);
    }

    /// Save the tokenizer's vocabulary to a file
    pub fn save_vocabulary(&self, path: impl AsRef<Path>) -> Result<()> {
        match &self.vocabulary {
            Some(vocab) => vocab.save(path),
            None => Err(TextError::TokenizationError(
                "No vocabulary available to save".to_string(),
            )),
        }
    }

    /// Load the tokenizer's vocabulary from a file
    pub fn load_vocabulary(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vocabulary = Some(BpeVocabulary::load(path)?);
        Ok(())
    }

    /// Train the BPE tokenizer on a corpus
    pub fn train(&mut self, corpus: &[&str]) -> Result<()> {
        if corpus.is_empty() {
            return Err(TextError::TokenizationError(
                "Cannot train on empty corpus".to_string(),
            ));
        }

        let mut vocabulary = BpeVocabulary::new();

        // Initialize vocabulary with special tokens
        for token in &self.config.special_tokens {
            vocabulary.add_token(token);
        }

        // Count initial tokens (characters or words)
        let mut token_counts = HashMap::new();
        let mut all_tokens = Vec::new();

        for text in corpus {
            let processed_text = if self.config.lowercase {
                text.to_lowercase()
            } else {
                text.to_string()
            };

            // For character-level tokenization, we operate directly on characters
            // For word-level tokenization, we need to process each word separately
            if self.config.character_level {
                // Character-level tokenization
                let initial_tokens: Vec<String> =
                    processed_text.chars().map(|c| c.to_string()).collect();
                // Add character sequence directly
                for token in &initial_tokens {
                    *token_counts.entry(token.clone()).or_insert(0) += 1;
                }
                all_tokens.push(initial_tokens);
            } else {
                // Word-level tokenization with characters as base tokens
                for word in processed_text.split_whitespace() {
                    let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                    // Count individual characters
                    for token in &chars {
                        *token_counts.entry(token.clone()).or_insert(0) += 1;
                    }
                    all_tokens.push(chars);
                }
            };

            // The token counting is now handled in the previous block
        }

        // Add initial tokens to vocabulary
        for (token, &count) in &token_counts {
            if count >= self.config.min_frequency {
                vocabulary.add_token(token);
            }
        }

        // Train BPE on the corpus
        let mut merges = Vec::new();
        let max_merges = self.config.vocab_size - vocabulary.token_to_id.len();

        for _ in 0..max_merges {
            // Count token pairs
            let mut pair_counts = HashMap::new();
            let mut pair_to_merged = HashMap::new();

            for tokens in &all_tokens {
                for window in tokens.windows(2) {
                    if window.len() < 2 {
                        continue;
                    }

                    let pair = (window[0].clone(), window[1].clone());
                    let merged = format!("{}{}", pair.0, pair.1);
                    *pair_counts.entry(pair.clone()).or_insert(0) += 1;
                    pair_to_merged.insert(pair, merged);
                }
            }

            // Find the most frequent pair
            let best_pair = pair_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| pair.clone());

            if let Some(pair) = best_pair {
                let merged = pair_to_merged[&pair].clone();

                // Add the merged token to the vocabulary
                vocabulary.add_token(&merged);

                // Add the merge rule
                vocabulary.add_merge(pair.clone(), merged.clone());
                merges.push((pair.clone(), merged.clone()));

                // Update tokens by applying the merge
                for tokens in &mut all_tokens {
                    let mut i = 0;
                    while i < tokens.len() - 1 {
                        if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                            tokens[i] = merged.clone();
                            tokens.remove(i + 1);
                        } else {
                            i += 1;
                        }
                    }
                }
            } else {
                // No more pairs to merge
                break;
            }
        }

        self.vocabulary = Some(vocabulary);
        Ok(())
    }

    /// Apply BPE to tokenize a word
    fn tokenize_word(&self, word: &str) -> Result<Vec<String>> {
        let vocab = match &self.vocabulary {
            Some(v) => v,
            None => {
                return Err(TextError::TokenizationError(
                    "Tokenizer vocabulary not initialized. Call train() first".to_string(),
                ))
            }
        };

        // Split word into characters
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply merges
        let mut has_changes = true;
        while has_changes {
            has_changes = false;

            let mut i = 0;
            while i < tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(merged) = vocab.merges.get(&pair) {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                    has_changes = true;
                } else {
                    i += 1;
                }
            }
        }

        Ok(tokens)
    }
}

impl Tokenizer for BpeTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        if !self.has_vocabulary() {
            return Err(TextError::TokenizationError(
                "Tokenizer vocabulary not initialized. Call train() first".to_string(),
            ));
        }

        let processed_text = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let mut tokens = Vec::new();

        if self.config.character_level {
            // Tokenize as a single sequence
            tokens = self.tokenize_word(&processed_text)?;
        } else {
            // Tokenize each word separately
            for word in processed_text.split_whitespace() {
                let word_tokens = self.tokenize_word(word)?;
                tokens.extend(word_tokens);
            }
        }

        Ok(tokens)
    }

    fn clone_box(&self) -> Box<dyn Tokenizer + Send + Sync> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_bpe_tokenizer_train() {
        let corpus = [
            "this is a test",
            "another test",
            "more tests for testing",
            "test the tokenizer",
        ];

        let mut tokenizer = BpeTokenizer::with_defaults();
        tokenizer.train(&corpus).unwrap();

        assert!(tokenizer.has_vocabulary());
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_bpe_tokenizer_tokenize() {
        let corpus = [
            "this is a test",
            "another test",
            "more tests for testing",
            "test the tokenizer",
        ];

        let mut tokenizer = BpeTokenizer::with_defaults();
        tokenizer.train(&corpus).unwrap();

        let tokens = tokenizer.tokenize("this is a tokenizer test").unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_vocabulary_save_load() {
        let corpus = [
            "this is a test",
            "another test",
            "more tests for testing",
            "test the tokenizer",
        ];

        let mut tokenizer = BpeTokenizer::with_defaults();
        tokenizer.train(&corpus).unwrap();

        // Create a temporary directory for the test
        let temp_dir = tempdir().unwrap();
        let vocab_path = temp_dir.path().join("vocab.bpe");

        // Save the vocabulary
        tokenizer.save_vocabulary(&vocab_path).unwrap();

        // Create a new tokenizer and load the vocabulary
        let mut new_tokenizer = BpeTokenizer::with_defaults();
        new_tokenizer.load_vocabulary(&vocab_path).unwrap();

        // Both tokenizers should produce the same tokens
        let text = "this is a tokenizer test";
        let tokens1 = tokenizer.tokenize(text).unwrap();
        let tokens2 = new_tokenizer.tokenize(text).unwrap();

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_bpe_tokenizer_with_special_tokens() {
        let config = BpeConfig {
            special_tokens: vec!["<pad>".to_string(), "<unk>".to_string()],
            ..Default::default()
        };

        let corpus = [
            "this is a test",
            "another test",
            "more tests for testing",
            "test the tokenizer",
        ];

        let mut tokenizer = BpeTokenizer::new(config);
        tokenizer.train(&corpus).unwrap();

        let vocab = tokenizer.vocabulary().unwrap();
        assert!(vocab.token_to_id.contains_key("<pad>"));
        assert!(vocab.token_to_id.contains_key("<unk>"));
    }

    #[test]
    fn test_bpe_tokenizer_empty_text() {
        let corpus = ["this is a test"];
        let mut tokenizer = BpeTokenizer::with_defaults();
        tokenizer.train(&corpus).unwrap();

        let tokens = tokenizer.tokenize("").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_bpe_tokenizer_case_sensitivity() {
        let corpus = ["This IS a TEST"];

        // Test with lowercase=true (default)
        let mut tokenizer1 = BpeTokenizer::with_defaults();
        tokenizer1.train(&corpus).unwrap();
        let tokens1 = tokenizer1.tokenize("THIS is A test").unwrap();

        // Test with lowercase=false
        let config = BpeConfig {
            lowercase: false,
            ..Default::default()
        };
        let mut tokenizer2 = BpeTokenizer::new(config);
        tokenizer2.train(&corpus).unwrap();
        let tokens2 = tokenizer2.tokenize("THIS is A test").unwrap();

        // The lowercase tokenizer should produce fewer tokens as it's case-insensitive
        assert!(tokens1.len() <= tokens2.len());
    }

    #[test]
    fn test_bpe_tokenizer_no_vocabulary() {
        // Create tokenizer with no vocabulary (vocabulary set to None)
        let mut tokenizer = BpeTokenizer::with_defaults();
        tokenizer.vocabulary = None;

        // No vocabulary is initialized, so this should fail
        let result = tokenizer.tokenize("test");
        assert!(result.is_err()); // This should be an error
    }
}
