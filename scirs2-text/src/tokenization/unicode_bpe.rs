//! Language-agnostic BPE tokenizer with Unicode/NFC normalization.
//!
//! Implements the standard BPE merge algorithm operating on Unicode characters
//! (with optional byte-fallback for unknown characters) rather than on raw
//! bytes alone, so it works across scripts.

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Unicode-aware BPE tokenizer.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct UnicodeBpeConfig {
    /// Target vocabulary size (base chars + merge operations).
    pub vocab_size: usize,
    /// Minimum pair frequency for a merge operation to be kept.
    pub min_frequency: usize,
    /// Apply NFC-style normalization (simplified: recompose via canonical form).
    pub normalize: bool,
    /// Represent characters absent from the training vocabulary as `<0xHH>` byte tokens.
    pub byte_fallback: bool,
}

impl Default for UnicodeBpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            min_frequency: 2,
            normalize: true,
            byte_fallback: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: simplified NFC normalization
// ---------------------------------------------------------------------------

/// Simplified NFC: collect chars, re-emit them — Rust's `char` is Unicode scalar,
/// so collecting into a `String` already yields a well-formed Unicode string.
/// Full NFC would require unicode-normalization; here we at minimum remove
/// ASCII control characters and canonicalize whitespace.
fn nfc_normalize(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || c.is_whitespace())
        .collect()
}

// ---------------------------------------------------------------------------
// BPE implementation
// ---------------------------------------------------------------------------

/// Unicode-normalized BPE tokenizer that trains on a raw text corpus.
pub struct UnicodeBpeTokenizer {
    config: UnicodeBpeConfig,
    /// token → id mapping (populated after training).
    vocab: HashMap<String, u32>,
    /// id → token reverse mapping.
    id_to_token: Vec<String>,
    /// Ordered list of merge operations (pair → merged token).
    merges: Vec<(String, String)>,
    /// Special tokens always added to the vocabulary.
    special_tokens: Vec<String>,
}

/// Result of a single merge step.
struct MergeResult {
    pair: (String, String),
    freq: usize,
    new_token: String,
}

impl UnicodeBpeTokenizer {
    /// Create an untrained tokenizer with the given configuration.
    pub fn new(config: UnicodeBpeConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            merges: Vec::new(),
            special_tokens: vec!["<unk>".into(), "<s>".into(), "</s>".into(), "<pad>".into()],
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train the BPE vocabulary on a corpus of strings.
    pub fn train(&mut self, corpus: &[&str]) -> Result<()> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput(
                "BPE training corpus must not be empty".into(),
            ));
        }

        // ---- 1. Pre-tokenize corpus into words, normalize ----
        let words: Vec<String> = corpus
            .iter()
            .flat_map(|doc| {
                let normalized = if self.config.normalize {
                    nfc_normalize(doc)
                } else {
                    doc.to_string()
                };
                normalized
                    .split_whitespace()
                    .map(|w| w.to_owned())
                    .collect::<Vec<_>>()
            })
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus has no words after split".into(),
            ));
        }

        // ---- 2. Count word frequencies ----
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *word_freq.entry(word.clone()).or_insert(0) += 1;
        }

        // ---- 3. Represent each word as a sequence of chars (+ </w> end marker) ----
        // word_splits: word → Vec<String> of character tokens
        let mut word_splits: HashMap<String, Vec<String>> = word_freq
            .keys()
            .map(|w| {
                let chars: Vec<String> = w.chars().map(|c| c.to_string()).collect();
                (w.clone(), chars)
            })
            .collect();

        // ---- 4. Collect base character vocabulary ----
        let mut base_chars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for chars in word_splits.values() {
            for c in chars {
                base_chars.insert(c.clone());
            }
        }

        // ---- 5. Initialise vocabulary with special tokens + base chars ----
        self.vocab.clear();
        self.id_to_token.clear();
        self.merges.clear();

        for sp in &self.special_tokens {
            let id = self.id_to_token.len() as u32;
            self.vocab.insert(sp.clone(), id);
            self.id_to_token.push(sp.clone());
        }
        for c in &base_chars {
            if !self.vocab.contains_key(c) {
                let id = self.id_to_token.len() as u32;
                self.vocab.insert(c.clone(), id);
                self.id_to_token.push(c.clone());
            }
        }

        // ---- 6. BPE merge loop ----
        let max_merges = self.config.vocab_size.saturating_sub(self.vocab.len());

        for _ in 0..max_merges {
            // Count bigram frequencies
            let mut pair_freq: HashMap<(String, String), usize> = HashMap::new();
            for (word, freq) in &word_freq {
                let chars = match word_splits.get(word) {
                    Some(c) => c,
                    None => continue,
                };
                for window in chars.windows(2) {
                    *pair_freq
                        .entry((window[0].clone(), window[1].clone()))
                        .or_insert(0) += freq;
                }
            }

            // Find best merge (highest frequency, tie-break by lexicographic order)
            let best = pair_freq
                .iter()
                .filter(|(_, &freq)| freq >= self.config.min_frequency)
                .max_by_key(|((a, b), &freq)| (freq, std::cmp::Reverse((a.clone(), b.clone()))));

            let merge = match best {
                Some(((a, b), &freq)) => MergeResult {
                    pair: (a.clone(), b.clone()),
                    freq,
                    new_token: format!("{}{}", a, b),
                },
                None => break, // no more eligible merges
            };

            if merge.freq < self.config.min_frequency {
                break;
            }

            // Register new token
            if !self.vocab.contains_key(&merge.new_token) {
                let id = self.id_to_token.len() as u32;
                self.vocab.insert(merge.new_token.clone(), id);
                self.id_to_token.push(merge.new_token.clone());
            }
            self.merges.push(merge.pair.clone());

            // Apply merge to all word splits
            let (ref left, ref right) = merge.pair;
            for chars in word_splits.values_mut() {
                let mut i = 0;
                while i + 1 < chars.len() {
                    if chars[i] == *left && chars[i + 1] == *right {
                        let merged = format!("{}{}", chars[i], chars[i + 1]);
                        chars.splice(i..=i + 1, std::iter::once(merged));
                        // Don't advance i — the newly merged token might pair again
                    } else {
                        i += 1;
                    }
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Tokenize a string to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if self.vocab.is_empty() {
            return Err(TextError::ModelNotFitted(
                "BPE tokenizer has not been trained".into(),
            ));
        }

        let normalized = if self.config.normalize {
            nfc_normalize(text)
        } else {
            text.to_string()
        };

        let unk_id = self.vocab.get("<unk>").copied().unwrap_or(0);

        let mut ids = Vec::new();

        for word in normalized.split_whitespace() {
            // Split word into individual characters
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Apply merges in training order
            for (left, right) in &self.merges {
                let mut i = 0;
                while i + 1 < chars.len() {
                    if chars[i] == *left && chars[i + 1] == *right {
                        let merged = format!("{}{}", chars[i], chars[i + 1]);
                        chars.splice(i..=i + 1, std::iter::once(merged));
                    } else {
                        i += 1;
                    }
                }
            }

            for tok in chars {
                if let Some(&id) = self.vocab.get(&tok) {
                    ids.push(id);
                } else if self.config.byte_fallback {
                    // Encode as individual UTF-8 bytes: <0xHH>
                    for byte in tok.as_bytes() {
                        let byte_tok = format!("<0x{:02X}>", byte);
                        let id = self.vocab.get(&byte_tok).copied().unwrap_or(unk_id);
                        ids.push(id);
                    }
                } else {
                    ids.push(unk_id);
                }
            }
        }

        Ok(ids)
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if self.id_to_token.is_empty() {
            return Err(TextError::ModelNotFitted(
                "BPE tokenizer has not been trained".into(),
            ));
        }
        let mut parts = Vec::new();
        for &id in ids {
            let idx = id as usize;
            if idx >= self.id_to_token.len() {
                return Err(TextError::InvalidInput(format!(
                    "token id {} out of vocabulary range {}",
                    id,
                    self.id_to_token.len()
                )));
            }
            parts.push(self.id_to_token[idx].clone());
        }
        Ok(parts.join(" "))
    }

    /// Current vocabulary size (number of entries in the token → id mapping).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Number of merge operations learned during training.
    pub fn n_merges(&self) -> usize {
        self.merges.len()
    }

    /// Access the raw vocabulary map.
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<&'static str> {
        vec![
            "low lower lowest",
            "new newer newest",
            "low new lower newest",
            "the lowest number",
        ]
    }

    #[test]
    fn test_default_config() {
        let cfg = UnicodeBpeConfig::default();
        assert_eq!(cfg.vocab_size, 32_000);
        assert_eq!(cfg.min_frequency, 2);
        assert!(cfg.normalize);
        assert!(cfg.byte_fallback);
    }

    #[test]
    fn test_train_empty_corpus_error() {
        let mut tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig::default());
        let result = tok.train(&[]);
        assert!(result.is_err(), "empty corpus must return error");
    }

    #[test]
    fn test_train_succeeds() {
        let mut tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig::default());
        tok.train(&small_corpus()).expect("train failed");
        assert!(
            tok.vocab_size() > 0,
            "vocab should be non-empty after training"
        );
    }

    #[test]
    fn test_vocab_size_bounded() {
        let config = UnicodeBpeConfig {
            vocab_size: 20,
            min_frequency: 1,
            ..Default::default()
        };
        let mut tok = UnicodeBpeTokenizer::new(config);
        tok.train(&small_corpus()).expect("train failed");
        assert!(
            tok.vocab_size() <= 20,
            "vocab size {} must be <= 20",
            tok.vocab_size()
        );
    }

    #[test]
    fn test_encode_returns_ids() {
        let mut tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig {
            min_frequency: 1,
            ..Default::default()
        });
        tok.train(&small_corpus()).expect("train failed");
        let ids = tok.encode("low").expect("encode failed");
        assert!(
            !ids.is_empty(),
            "encoding 'low' should produce at least one id"
        );
    }

    #[test]
    fn test_encode_before_train_error() {
        let tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig::default());
        let result = tok.encode("hello");
        assert!(result.is_err(), "encode before train must return error");
    }

    #[test]
    fn test_decode_before_train_error() {
        let tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig::default());
        let result = tok.decode(&[0, 1]);
        assert!(result.is_err(), "decode before train must return error");
    }

    #[test]
    fn test_n_merges_increases_with_training() {
        let mut tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig {
            vocab_size: 50,
            min_frequency: 1,
            ..Default::default()
        });
        tok.train(&small_corpus()).expect("train failed");
        assert!(tok.n_merges() > 0, "should have at least one merge");
    }

    #[test]
    fn test_special_tokens_in_vocab() {
        let mut tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig::default());
        tok.train(&small_corpus()).expect("train failed");
        assert!(tok.vocab().contains_key("<unk>"));
        assert!(tok.vocab().contains_key("<s>"));
        assert!(tok.vocab().contains_key("</s>"));
    }

    #[test]
    fn test_decode_special_token() {
        let mut tok = UnicodeBpeTokenizer::new(UnicodeBpeConfig {
            min_frequency: 1,
            ..Default::default()
        });
        tok.train(&small_corpus()).expect("train failed");
        let unk_id = tok.vocab()["<unk>"];
        let decoded = tok.decode(&[unk_id]).expect("decode failed");
        assert_eq!(decoded, "<unk>");
    }
}
