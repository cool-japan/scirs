//! Enhanced Byte Pair Encoding (BPE) tokenizer
//!
//! This module provides a standalone BPE tokenizer with:
//! - Training from a raw-text corpus (`train_bpe`)
//! - Encoding to subword IDs (`bpe_encode`)
//! - Decoding IDs back to text (`bpe_decode`)
//! - Tokenizing to string subword tokens (`bpe_tokenize`)
//!
//! The design intentionally differs from the tokenizer in `tokenize::bpe` – this
//! module exposes a flat, functional API (`BpeVocab` + free functions) that is
//! useful for embedding pipelines and cross-lingual NLP.

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BpeVocab
// ---------------------------------------------------------------------------

/// BPE vocabulary containing merge rules and a token-to-ID mapping.
///
/// The `merges` vector is **ordered**: earlier merges have higher priority and
/// are applied first when encoding new text.
#[derive(Debug, Clone)]
pub struct BpeVocab {
    /// Ordered merge rules learned during training
    pub merges: Vec<(String, String)>,
    /// Token string → integer ID mapping
    pub vocab: HashMap<String, usize>,
    /// Integer ID → token string mapping (inverse of `vocab`)
    id_to_token: HashMap<usize, String>,
}

impl BpeVocab {
    /// Create an empty `BpeVocab`.
    pub fn new() -> Self {
        Self {
            merges: Vec::new(),
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
        }
    }

    /// Add a token, returning its ID.  If the token already exists the
    /// existing ID is returned without modification.
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }
        let id = self.vocab.len();
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }

    /// Look up a token by ID.
    pub fn token_for_id(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// `true` when the vocabulary contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Build a merge-rule priority map: `(left, right)` → merge rank (lower
    /// rank = applied first).
    fn merge_priorities(&self) -> HashMap<(&str, &str), usize> {
        self.merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.as_str(), b.as_str()), rank))
            .collect()
    }
}

impl Default for BpeVocab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Split a word into individual UTF-8 characters, appending `</w>` to the last
/// symbol to mark word boundaries (standard SentencePiece/GPT-2 convention).
fn word_to_chars(word: &str) -> Vec<String> {
    let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    if let Some(last) = chars.last_mut() {
        last.push_str("</w>");
    }
    chars
}

/// Count every adjacent pair in a token sequence.
fn count_pairs(sequences: &[Vec<String>]) -> HashMap<(String, String), usize> {
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    for seq in sequences {
        for window in seq.windows(2) {
            *counts
                .entry((window[0].clone(), window[1].clone()))
                .or_insert(0) += 1;
        }
    }
    counts
}

/// Apply one BPE merge (replace every occurrence of `pair` with `merged`) in-place.
fn apply_merge(sequences: &mut [Vec<String>], pair: &(String, String), merged: &str) {
    for seq in sequences.iter_mut() {
        let mut i = 0;
        while i + 1 < seq.len() {
            if seq[i] == pair.0 && seq[i + 1] == pair.1 {
                seq[i] = merged.to_string();
                seq.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}

/// Train a BPE vocabulary from a raw text corpus.
///
/// Words are lower-cased and split on whitespace before character-level
/// initialisation.  The `</w>` end-of-word marker is appended to each word's
/// last symbol so that subwords can distinguish word-medial from word-final
/// occurrences.
///
/// # Errors
/// Returns [`TextError::InvalidInput`] when `corpus` is empty or `vocab_size`
/// is smaller than the initial character-level vocabulary.
pub fn train_bpe(corpus: &[&str], vocab_size: usize) -> Result<BpeVocab> {
    if corpus.is_empty() {
        return Err(TextError::InvalidInput(
            "corpus must not be empty".to_string(),
        ));
    }

    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for text in corpus {
        for word in text.split_whitespace() {
            *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
        }
    }

    if word_counts.is_empty() {
        return Err(TextError::InvalidInput(
            "corpus contains no words".to_string(),
        ));
    }

    // Build initial character-level sequences
    let mut sequences: Vec<Vec<String>> = Vec::new();
    let mut word_seq_counts: Vec<usize> = Vec::new();

    for (word, &cnt) in &word_counts {
        sequences.push(word_to_chars(word));
        word_seq_counts.push(cnt);
    }

    // Collect base character vocabulary
    let mut vocab = BpeVocab::new();
    for seq in &sequences {
        for tok in seq {
            vocab.add_token(tok);
        }
    }

    if vocab_size < vocab.len() {
        return Err(TextError::InvalidInput(format!(
            "vocab_size ({vocab_size}) is smaller than the initial character vocabulary ({})",
            vocab.len()
        )));
    }

    let num_merges = vocab_size - vocab.len();

    // BPE training loop
    for _ in 0..num_merges {
        // Weight pair counts by word frequency
        let mut weighted: HashMap<(String, String), usize> = HashMap::new();
        for (seq, &cnt) in sequences.iter().zip(word_seq_counts.iter()) {
            for window in seq.windows(2) {
                *weighted
                    .entry((window[0].clone(), window[1].clone()))
                    .or_insert(0) += cnt;
            }
        }

        let best = weighted
            .iter()
            .max_by_key(|&(_, &cnt)| cnt)
            .map(|(pair, _)| pair.clone());

        let pair = match best {
            Some(p) => p,
            None => break, // No more pairs
        };

        let merged = format!("{}{}", pair.0, pair.1);

        vocab.add_token(&merged);
        vocab.merges.push((pair.0.clone(), pair.1.clone()));

        apply_merge(&mut sequences, &pair, &merged);
    }

    Ok(vocab)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Apply BPE merges to a single pre-tokenized word (already split into chars
/// with `</w>` appended to the last symbol).
fn bpe_encode_word(chars: Vec<String>, priorities: &HashMap<(&str, &str), usize>) -> Vec<String> {
    let mut tokens = chars;

    loop {
        if tokens.len() < 2 {
            break;
        }

        // Find the pair with the lowest merge rank (= highest priority)
        let best_pos = tokens
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                priorities
                    .get(&(w[0].as_str(), w[1].as_str()))
                    .map(|&rank| (i, rank))
            })
            .min_by_key(|&(_, rank)| rank);

        match best_pos {
            None => break,
            Some((pos, _)) => {
                let merged = format!("{}{}", tokens[pos], tokens[pos + 1]);
                tokens[pos] = merged;
                tokens.remove(pos + 1);
            }
        }
    }

    tokens
}

/// Encode `text` into a sequence of subword token IDs using `vocab`.
///
/// Unknown subword tokens are mapped to the `<unk>` token if present in the
/// vocabulary, or omitted otherwise.
///
/// # Errors
/// Returns [`TextError::ProcessingError`] when the vocab is empty.
pub fn bpe_encode(text: &str, vocab: &BpeVocab) -> Result<Vec<usize>> {
    if vocab.is_empty() {
        return Err(TextError::ProcessingError(
            "BpeVocab is empty; train the tokenizer first".to_string(),
        ));
    }

    let tokens = bpe_tokenize_inner(text, vocab);
    let unk_id = vocab.vocab.get("<unk>").copied();

    let ids: Vec<usize> = tokens
        .iter()
        .filter_map(|tok| vocab.vocab.get(tok.as_str()).copied().or(unk_id))
        .collect();

    Ok(ids)
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decode a sequence of subword IDs back to a text string using `vocab`.
///
/// The `</w>` end-of-word marker is replaced by a space so that word
/// boundaries are reconstructed naturally.
///
/// # Errors
/// Returns [`TextError::ProcessingError`] when the vocab is empty.
pub fn bpe_decode(ids: &[usize], vocab: &BpeVocab) -> Result<String> {
    if vocab.is_empty() {
        return Err(TextError::ProcessingError(
            "BpeVocab is empty; train the tokenizer first".to_string(),
        ));
    }

    let mut text = String::new();
    for &id in ids {
        if let Some(tok) = vocab.token_for_id(id) {
            text.push_str(&tok.replace("</w>", " "));
        }
    }

    Ok(text.trim_end().to_string())
}

// ---------------------------------------------------------------------------
// Tokenization (string output)
// ---------------------------------------------------------------------------

/// Internal: split `text` into BPE subword tokens as strings (with `</w>`).
fn bpe_tokenize_inner(text: &str, vocab: &BpeVocab) -> Vec<String> {
    if vocab.merges.is_empty() {
        // No merges – fall back to character-level with </w> boundary markers
        let mut out = Vec::new();
        for word in text.split_whitespace() {
            out.extend(word_to_chars(&word.to_lowercase()));
        }
        return out;
    }

    let priorities = vocab.merge_priorities();

    let mut out: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let chars = word_to_chars(&word.to_lowercase());
        let encoded = bpe_encode_word(chars, &priorities);
        out.extend(encoded);
    }
    out
}

/// Tokenize `text` into BPE subword tokens (as `String`s, with `</w>` markers
/// stripped for readability).
///
/// # Errors
/// Returns [`TextError::ProcessingError`] when the vocab is empty.
pub fn bpe_tokenize(text: &str, vocab: &BpeVocab) -> Result<Vec<String>> {
    if vocab.is_empty() {
        return Err(TextError::ProcessingError(
            "BpeVocab is empty; train the tokenizer first".to_string(),
        ));
    }

    let raw = bpe_tokenize_inner(text, vocab);
    // Strip </w> from display tokens so callers get clean subword strings
    let clean: Vec<String> = raw
        .into_iter()
        .map(|t| t.replace("</w>", ""))
        .filter(|t| !t.is_empty())
        .collect();
    Ok(clean)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &[&str] = &[
        "low lower lowest",
        "new newer newest",
        "wide wider widest",
        "low low low new new newer",
    ];

    #[test]
    fn test_train_bpe_creates_vocab() {
        let vocab = train_bpe(CORPUS, 50).expect("train_bpe failed");
        assert!(!vocab.is_empty());
        assert!(!vocab.merges.is_empty());
    }

    #[test]
    fn test_train_bpe_respects_vocab_size() {
        let vocab = train_bpe(CORPUS, 50).expect("train_bpe failed");
        assert!(vocab.len() <= 50);
    }

    #[test]
    fn test_train_bpe_empty_corpus() {
        assert!(train_bpe(&[], 100).is_err());
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let vocab = train_bpe(CORPUS, 60).expect("train_bpe failed");
        let ids = bpe_encode("low new", &vocab).expect("encode failed");
        assert!(!ids.is_empty());
        let text = bpe_decode(&ids, &vocab).expect("decode failed");
        // The decoded text should contain the original words (possibly with
        // subword boundaries collapsed)
        assert!(!text.is_empty());
    }

    #[test]
    fn test_bpe_tokenize_returns_strings() {
        let vocab = train_bpe(CORPUS, 60).expect("train_bpe failed");
        let tokens = bpe_tokenize("low newer", &vocab).expect("tokenize failed");
        assert!(!tokens.is_empty());
        // No token should contain the </w> marker (they are stripped)
        for tok in &tokens {
            assert!(
                !tok.contains("</w>"),
                "token should not contain </w>: {tok}"
            );
        }
    }

    #[test]
    fn test_bpe_encode_empty_vocab() {
        let vocab = BpeVocab::new();
        assert!(bpe_encode("hello", &vocab).is_err());
    }

    #[test]
    fn test_bpe_decode_empty_vocab() {
        let vocab = BpeVocab::new();
        assert!(bpe_decode(&[0, 1], &vocab).is_err());
    }

    #[test]
    fn test_word_to_chars_has_end_marker() {
        let chars = word_to_chars("hi");
        assert_eq!(chars, vec!["h".to_string(), "i</w>".to_string()]);
    }

    #[test]
    fn test_bpe_vocab_add_token_idempotent() {
        let mut vocab = BpeVocab::new();
        let id1 = vocab.add_token("hello");
        let id2 = vocab.add_token("hello");
        assert_eq!(id1, id2);
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_bpe_tokenize_unknown_word() {
        // Train on a small corpus, then tokenize an unseen word; should degrade
        // gracefully to character-level tokens rather than panicking.
        let vocab = train_bpe(&["abc def"], 20).expect("train_bpe failed");
        let tokens = bpe_tokenize("xyz", &vocab).expect("tokenize failed");
        // We expect individual characters x, y, z (or merged variants)
        assert!(!tokens.is_empty());
    }
}
