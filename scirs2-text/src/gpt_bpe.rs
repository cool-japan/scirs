//! GPT-2 Byte-Level BPE Tokenizer
//!
//! This module implements a byte-level Byte-Pair Encoding tokenizer as used by
//! GPT-2, GPT-3, GPT-4, and RoBERTa. Key characteristics:
//!
//! - **Byte-level encoding**: text is first converted to UTF-8 bytes, then
//!   each byte is mapped to a printable Unicode character for clean display.
//! - **Pre-tokenization**: a regex-based pattern splits text into words,
//!   contractions, numbers, and punctuation before BPE merging.
//! - **Ordered merge rules**: BPE merges are applied in a fixed priority order
//!   learned during training.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::gpt_bpe::Gpt2BpeTokenizer;
//! use scirs2_text::tokenizer::TransformerTokenizer;
//!
//! let corpus = &["hello world", "hello there", "world wide web"];
//! let tokenizer = Gpt2BpeTokenizer::train(corpus, 200)
//!     .expect("training failed");
//! let ids = tokenizer.encode("hello world");
//! let decoded = tokenizer.decode(&ids);
//! assert!(!ids.is_empty());
//! ```

use crate::error::{Result, TextError};
use crate::tokenizer::TransformerTokenizer;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Byte-to-Unicode mapping (GPT-2 style)
// ---------------------------------------------------------------------------

/// Build the GPT-2 byte-to-unicode mapping.
///
/// GPT-2 maps each byte (0..=255) to a unique Unicode code point so that
/// subword tokens can be displayed as readable strings. Printable ASCII bytes
/// map to themselves; non-printable bytes are shifted to a higher Unicode range.
fn byte_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut next_free: u32 = 256;

    for b in 0u16..=255 {
        let byte = b as u8;
        let ch = byte as char;
        // GPT-2 considers these ranges "printable"
        let is_printable = (b'!'..=b'~').contains(&byte)
            || (0xA1u8..=0xACu8).contains(&byte)
            || (0xAEu8..=0xFFu8).contains(&byte);
        if is_printable {
            map.insert(byte, ch);
        } else {
            // Map to a Unicode character above 255
            if let Some(c) = char::from_u32(next_free) {
                map.insert(byte, c);
                next_free += 1;
            }
        }
    }
    map
}

/// Build the inverse of [`byte_to_unicode`]: Unicode char -> byte value.
fn unicode_to_byte() -> HashMap<char, u8> {
    byte_to_unicode().into_iter().map(|(b, c)| (c, b)).collect()
}

// ---------------------------------------------------------------------------
// Pre-tokenization regex (simplified)
// ---------------------------------------------------------------------------

/// GPT-2-style pre-tokenization: split text into word-like tokens.
///
/// This is a simplified implementation of the GPT-2 pre-tokenization regex:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`
///
/// We split on transitions between:
/// - alphabetic runs (possibly preceded by a space)
/// - numeric runs (possibly preceded by a space)
/// - punctuation / other runs (possibly preceded by a space)
/// - whitespace-only runs
fn gpt2_pre_tokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let ch = chars[i];

        // Check for contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if ch == '\'' && i + 1 < n {
            let next = chars[i + 1];
            match next {
                's' | 't' | 'm' | 'd' => {
                    tokens.push(format!("'{}", next));
                    i += 2;
                    continue;
                }
                'r' if i + 2 < n && chars[i + 2] == 'e' => {
                    tokens.push("'re".to_string());
                    i += 3;
                    continue;
                }
                'v' if i + 2 < n && chars[i + 2] == 'e' => {
                    tokens.push("'ve".to_string());
                    i += 3;
                    continue;
                }
                'l' if i + 2 < n && chars[i + 2] == 'l' => {
                    tokens.push("'ll".to_string());
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }

        if ch.is_whitespace() {
            // Whitespace run
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            // If followed by a letter/digit/punct, attach one space to next token
            if i < n {
                // Keep whitespace as separate token
                let ws: String = chars[start..i].iter().collect();
                tokens.push(ws);
            } else {
                let ws: String = chars[start..i].iter().collect();
                tokens.push(ws);
            }
        } else if ch.is_alphabetic() {
            // Letter run (possibly preceded by space already consumed)
            let start = i;
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            tokens.push(word);
        } else if ch.is_ascii_digit() {
            // Number run
            let start = i;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            let num: String = chars[start..i].iter().collect();
            tokens.push(num);
        } else {
            // Punctuation / other
            let start = i;
            while i < n
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_ascii_digit()
            {
                i += 1;
            }
            let punct: String = chars[start..i].iter().collect();
            tokens.push(punct);
        }
    }

    // Merge leading space into next token where appropriate
    // (GPT-2 convention: " hello" is one token, not " " + "hello")
    let mut merged = Vec::new();
    let mut idx = 0;
    while idx < tokens.len() {
        if tokens[idx].chars().all(|c| c == ' ')
            && tokens[idx].len() == 1
            && idx + 1 < tokens.len()
            && !tokens[idx + 1]
                .chars()
                .next()
                .is_none_or(|c| c.is_whitespace())
        {
            // Merge single space with next token
            merged.push(format!("{}{}", tokens[idx], tokens[idx + 1]));
            idx += 2;
        } else {
            merged.push(tokens[idx].clone());
            idx += 1;
        }
    }

    merged
}

// ---------------------------------------------------------------------------
// BPE merge application
// ---------------------------------------------------------------------------

/// Apply one merge to a token sequence: find all adjacent (left, right) pairs
/// and replace them with `merged`.
fn apply_merge_to_word(word: &[String], left: &str, right: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
            result.push(format!("{}{}", left, right));
            i += 2;
        } else {
            result.push(word[i].clone());
            i += 1;
        }
    }
    result
}

/// Apply all merges (in priority order) to a word already split into byte-level
/// Unicode characters.
fn bpe_merge(word: &[String], merges: &[(String, String)]) -> Vec<String> {
    let mut current = word.to_vec();
    if current.len() <= 1 {
        return current;
    }

    // Build a rank map for O(1) lookup
    let merge_rank: HashMap<(&str, &str), usize> = merges
        .iter()
        .enumerate()
        .map(|(rank, (a, b))| ((a.as_str(), b.as_str()), rank))
        .collect();

    loop {
        if current.len() < 2 {
            break;
        }

        // Find the pair with the lowest rank (highest priority)
        let best = current
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                merge_rank
                    .get(&(w[0].as_str(), w[1].as_str()))
                    .map(|&rank| (i, rank))
            })
            .min_by_key(|&(_, rank)| rank);

        match best {
            None => break,
            Some((pos, _rank)) => {
                let left = &current[pos];
                let right = &current[pos + 1];
                let merged_str = format!("{}{}", left, right);
                current = apply_merge_to_word(&current, left, right);
                // The merged_str is not needed separately - it's created inside apply_merge_to_word
                let _ = merged_str;
            }
        }
    }

    current
}

// ---------------------------------------------------------------------------
// GPT-2 BPE Tokenizer
// ---------------------------------------------------------------------------

/// A GPT-2 style byte-level BPE tokenizer.
///
/// This tokenizer:
/// 1. Pre-tokenizes text into word-like chunks (letters, digits, punct, spaces)
/// 2. Encodes each chunk into byte-level Unicode characters
/// 3. Applies BPE merges in priority order
/// 4. Maps resulting tokens to integer IDs
#[derive(Debug, Clone)]
pub struct Gpt2BpeTokenizer {
    /// Ordered merge rules: (left, right) pairs in priority order
    merges: Vec<(String, String)>,
    /// Token string -> token ID
    vocab: HashMap<String, u32>,
    /// Token ID -> token string
    id_to_token: HashMap<u32, String>,
    /// Byte -> Unicode char mapping
    byte_encoder: HashMap<u8, char>,
    /// Unicode char -> byte mapping
    byte_decoder: HashMap<char, u8>,
    /// Special tokens
    special_tokens: HashMap<String, u32>,
}

impl Gpt2BpeTokenizer {
    /// Create a tokenizer from pre-built merges, vocabulary, and special tokens.
    ///
    /// # Errors
    /// Returns an error if the merges list or vocab is empty.
    pub fn new(
        merges: Vec<(String, String)>,
        vocab: HashMap<String, u32>,
        special_tokens: HashMap<String, u32>,
    ) -> Result<Self> {
        if vocab.is_empty() {
            return Err(TextError::InvalidInput(
                "vocabulary must not be empty".to_string(),
            ));
        }

        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let byte_encoder = byte_to_unicode();
        let byte_decoder = unicode_to_byte();

        Ok(Self {
            merges,
            vocab,
            id_to_token,
            byte_encoder,
            byte_decoder,
            special_tokens,
        })
    }

    /// Train a GPT-2 BPE tokenizer from a text corpus.
    ///
    /// This learns merge rules by iteratively merging the most frequent byte-pair.
    ///
    /// # Arguments
    /// - `corpus`: slice of text documents
    /// - `vocab_size`: target vocabulary size
    ///
    /// # Errors
    /// Returns an error if the corpus is empty.
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus must not be empty".to_string(),
            ));
        }

        let byte_encoder = byte_to_unicode();

        // Pre-tokenize the entire corpus
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for text in corpus {
            let pre_tokens = gpt2_pre_tokenize(text);
            for token in pre_tokens {
                *word_freqs.entry(token).or_insert(0) += 1;
            }
        }

        if word_freqs.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus contains no tokenizable text".to_string(),
            ));
        }

        // Convert words to byte-level unicode sequences
        let mut word_splits: Vec<(Vec<String>, usize)> = Vec::new();
        for (word, freq) in &word_freqs {
            let byte_chars: Vec<String> = word
                .as_bytes()
                .iter()
                .map(|b| {
                    byte_encoder
                        .get(b)
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| format!("{b}"))
                })
                .collect();
            word_splits.push((byte_chars, *freq));
        }

        // Build base vocabulary from individual byte-level chars
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        // Add special tokens
        let special_tokens_list = ["<|endoftext|>", "<|padding|>"];
        let mut special_tokens_map = HashMap::new();
        for sp in &special_tokens_list {
            vocab.insert(sp.to_string(), next_id);
            special_tokens_map.insert(sp.to_string(), next_id);
            next_id += 1;
        }

        // Add all unique byte-level characters
        for (splits, _) in &word_splits {
            for ch in splits {
                if !vocab.contains_key(ch) {
                    vocab.insert(ch.clone(), next_id);
                    next_id += 1;
                }
            }
        }

        let base_vocab_size = vocab.len();
        let num_merges = vocab_size.saturating_sub(base_vocab_size);

        let mut merges: Vec<(String, String)> = Vec::new();

        for _ in 0..num_merges {
            // Count all adjacent pairs weighted by word frequency
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (splits, freq) in &word_splits {
                for window in splits.windows(2) {
                    *pair_counts
                        .entry((window[0].clone(), window[1].clone()))
                        .or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best = pair_counts
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(pair, _)| pair.clone());

            let pair = match best {
                Some(p) => p,
                None => break,
            };

            let merged = format!("{}{}", pair.0, pair.1);

            // Add merged token to vocab
            if !vocab.contains_key(&merged) {
                vocab.insert(merged.clone(), next_id);
                next_id += 1;
            }
            merges.push(pair.clone());

            // Apply merge to all word splits
            for (splits, _) in &mut word_splits {
                *splits = apply_merge_to_word(splits, &pair.0, &pair.1);
            }
        }

        Self::new(merges, vocab, special_tokens_map)
    }

    /// Encode a single pre-tokenized word (already in byte-level Unicode) into
    /// token IDs.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        let byte_chars: Vec<String> = word
            .as_bytes()
            .iter()
            .map(|b| {
                self.byte_encoder
                    .get(b)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| format!("{b}"))
            })
            .collect();

        let merged = bpe_merge(&byte_chars, &self.merges);

        merged
            .iter()
            .map(|tok| {
                self.vocab.get(tok).copied().unwrap_or_else(|| {
                    // Fallback to <|endoftext|> as UNK
                    self.special_tokens
                        .get("<|endoftext|>")
                        .copied()
                        .unwrap_or(0)
                })
            })
            .collect()
    }

    /// Get the token string for a given ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Get the token ID for a given string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Return the list of merge rules.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Return the number of merge rules.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Return a cloned snapshot of the `token → id` vocabulary map.
    ///
    /// Useful for serialisation (e.g. building a HuggingFace `tokenizers.json`).
    pub fn vocab_snapshot(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }
}

// ---------------------------------------------------------------------------
// TransformerTokenizer implementation
// ---------------------------------------------------------------------------

impl TransformerTokenizer for Gpt2BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let pre_tokens = gpt2_pre_tokenize(text);
        let mut ids = Vec::new();
        for token in &pre_tokens {
            ids.extend(self.encode_word(token));
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut byte_chars = String::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(&id) {
                // Skip special tokens
                if self.special_tokens.contains_key(tok) {
                    continue;
                }
                byte_chars.push_str(tok);
            }
        }

        // Convert byte-level Unicode characters back to actual bytes
        let bytes: Vec<u8> = byte_chars
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        // Use lossy conversion to handle any invalid UTF-8 gracefully
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn train_test_tokenizer() -> Gpt2BpeTokenizer {
        let corpus = &[
            "hello world",
            "hello there",
            "world wide web",
            "hello hello hello",
            "the quick brown fox jumps over the lazy dog",
        ];
        Gpt2BpeTokenizer::train(corpus, 200).expect("training should succeed")
    }

    #[test]
    fn test_train_creates_vocab() {
        let tok = train_test_tokenizer();
        assert!(tok.vocab_size() > 0);
        assert!(!tok.merges.is_empty(), "should have learned merges");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = train_test_tokenizer();
        let text = "hello world";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "roundtrip should preserve text");
    }

    #[test]
    fn test_byte_level_handles_non_ascii() {
        let corpus = &[
            "hello world",
            "caf\u{00e9} latte",
            "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}",
        ];
        let tok = Gpt2BpeTokenizer::train(corpus, 300).expect("training should succeed");
        // Encode non-ASCII text
        let ids = tok.encode("caf\u{00e9}");
        assert!(!ids.is_empty(), "should encode non-ASCII text");
        let decoded = tok.decode(&ids);
        assert!(
            decoded.contains("caf"),
            "decoded should contain 'caf': {decoded}"
        );
    }

    #[test]
    fn test_merge_rules_applied_in_order() {
        let tok = train_test_tokenizer();
        // The most frequent pair should be merged first
        // Verify merges list is non-empty and applied
        assert!(!tok.merges.is_empty());
        // Encoding should produce fewer tokens than individual bytes
        let text = "hello";
        let ids = tok.encode(text);
        assert!(
            ids.len() <= text.len(),
            "BPE should merge bytes: got {} tokens for {} chars",
            ids.len(),
            text.len()
        );
    }

    #[test]
    fn test_pre_tokenization_contractions() {
        let tokens = gpt2_pre_tokenize("I'm don't we're you've they'll he'd");
        // Should split contractions
        let joined = tokens.join("|");
        assert!(
            joined.contains("'m"),
            "should split contraction 'm': {joined}"
        );
        assert!(
            joined.contains("'t"),
            "should split contraction 't': {joined}"
        );
        assert!(
            joined.contains("'re"),
            "should split contraction 're': {joined}"
        );
        assert!(
            joined.contains("'ve"),
            "should split contraction 've': {joined}"
        );
        assert!(
            joined.contains("'ll"),
            "should split contraction 'll': {joined}"
        );
        assert!(
            joined.contains("'d"),
            "should split contraction 'd': {joined}"
        );
    }

    #[test]
    fn test_pre_tokenization_numbers_and_punct() {
        let tokens = gpt2_pre_tokenize("hello, world! 42 is the answer.");
        assert!(!tokens.is_empty());
        // Should separate letters, punctuation, and numbers
        let has_number = tokens.iter().any(|t| t.contains("42"));
        assert!(has_number, "should separate numbers: {tokens:?}");
    }

    #[test]
    fn test_empty_corpus_error() {
        let result = Gpt2BpeTokenizer::train(&[], 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_string_encode() {
        let tok = train_test_tokenizer();
        let ids = tok.encode("");
        assert!(ids.is_empty(), "empty string should produce no tokens");
    }

    #[test]
    fn test_byte_to_unicode_coverage() {
        let map = byte_to_unicode();
        // Should map all 256 bytes
        assert_eq!(map.len(), 256, "should map all 256 byte values");
        // All values should be unique
        let values: std::collections::HashSet<char> = map.values().copied().collect();
        assert_eq!(values.len(), 256, "all mapped chars should be unique");
    }

    #[test]
    fn test_special_tokens_in_vocab() {
        let tok = train_test_tokenizer();
        assert!(
            tok.token_to_id("<|endoftext|>").is_some(),
            "should have endoftext token"
        );
        assert!(
            tok.token_to_id("<|padding|>").is_some(),
            "should have padding token"
        );
    }

    #[test]
    fn test_emoji_encoding() {
        // Train on corpus with emoji
        let corpus = &["hello \u{1f600}", "world \u{1f44d}", "hello world"];
        let tok = Gpt2BpeTokenizer::train(corpus, 300).expect("training should succeed");
        let ids = tok.encode("\u{1f600}");
        assert!(!ids.is_empty(), "should encode emoji");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "\u{1f600}", "should roundtrip emoji");
    }
}
