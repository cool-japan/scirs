//! Tokenizer implementations: BPE (Byte Pair Encoding), character-level, and word-level.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_neural::nlp::tokenizer::{CharTokenizer, WordTokenizer};
//!
//! let texts = ["hello world", "goodbye world"];
//! let tok = CharTokenizer::from_corpus(&texts);
//! let ids = tok.encode("hello");
//! let decoded = tok.decode(&ids);
//! assert_eq!(decoded, "hello");
//! ```

use crate::error::NeuralError;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BPE Tokenizer
// ---------------------------------------------------------------------------

/// Byte Pair Encoding tokenizer.
///
/// Trains a BPE vocabulary on a text corpus and provides encode/decode operations.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// token string → token id
    vocab: HashMap<String, usize>,
    /// id → token string
    id_to_token: Vec<String>,
    /// merge rules in priority order (pair to merge → merged token)
    merges: Vec<(String, String)>,
    /// special tokens
    special_tokens: HashMap<String, usize>,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeTokenizer {
    /// Create an empty BPE tokenizer with only special tokens.
    pub fn new() -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::new();
        let mut special_tokens: HashMap<String, usize> = HashMap::new();

        for (i, tok) in ["<pad>", "<unk>", "<bos>", "<eos>"].iter().enumerate() {
            vocab.insert(tok.to_string(), i);
            id_to_token.push(tok.to_string());
            special_tokens.insert(tok.to_string(), i);
        }

        Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            special_tokens,
        }
    }

    /// Train BPE on `corpus` until the vocabulary reaches `vocab_size`.
    ///
    /// The algorithm:
    /// 1. Start with character-level vocabulary (one char per token).
    /// 2. Repeatedly find the most frequent adjacent pair.
    /// 3. Merge that pair into a new token; update all sequences.
    /// 4. Stop when vocab_size is reached or no pairs remain.
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self, NeuralError> {
        if vocab_size < 4 {
            return Err(NeuralError::InvalidArgument(
                "vocab_size must be at least 4 (special tokens)".to_string(),
            ));
        }

        let mut tokenizer = Self::new();

        // Tokenize corpus into word → frequency, each word split into chars + </w>
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        for text in corpus {
            for raw_word in text.split_whitespace() {
                let mut chars: Vec<String> = raw_word.chars().map(|c| c.to_string()).collect();
                chars.push("</w>".to_string());
                *word_freqs.entry(chars).or_insert(0) += 1;
            }
        }

        // Add all initial characters to vocab
        for word_tokens in word_freqs.keys() {
            for tok in word_tokens {
                if !tokenizer.vocab.contains_key(tok) {
                    let id = tokenizer.id_to_token.len();
                    tokenizer.vocab.insert(tok.clone(), id);
                    tokenizer.id_to_token.push(tok.clone());
                }
            }
        }

        // Perform merges until we reach vocab_size
        while tokenizer.vocab.len() < vocab_size {
            // Count pair frequencies
            let pair_freqs = count_pairs(&word_freqs);
            if pair_freqs.is_empty() {
                break;
            }

            // Find most frequent pair (tie-break by lexicographic order for determinism)
            let best_pair = pair_freqs
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(k, _)| k.clone());

            let (left, right) = match best_pair {
                Some(p) => p,
                None => break,
            };

            let merged = format!("{left}{right}");

            // Add merged token to vocab
            if !tokenizer.vocab.contains_key(&merged) {
                let id = tokenizer.id_to_token.len();
                tokenizer.vocab.insert(merged.clone(), id);
                tokenizer.id_to_token.push(merged.clone());
            }

            // Record merge rule
            tokenizer.merges.push((left.clone(), right.clone()));

            // Apply merge to all words
            let new_word_freqs: HashMap<Vec<String>, usize> = word_freqs
                .into_iter()
                .map(|(tokens, freq)| (apply_merge(&tokens, &left, &right, &merged), freq))
                .collect();
            word_freqs = new_word_freqs;
        }

        Ok(tokenizer)
    }

    /// Tokenize a string into token IDs using the trained BPE vocabulary.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let unk_id = *self.special_tokens.get("<unk>").unwrap_or(&1);
        let mut ids = Vec::new();

        for raw_word in text.split_whitespace() {
            // Split into characters + end-of-word marker
            let mut tokens: Vec<String> = raw_word.chars().map(|c| c.to_string()).collect();
            tokens.push("</w>".to_string());

            // Apply merge rules in order
            for (left, right) in &self.merges {
                let merged = format!("{left}{right}");
                tokens = apply_merge(&tokens, left, right, &merged);
            }

            // Convert tokens to IDs
            for tok in tokens {
                let id = self.vocab.get(&tok).copied().unwrap_or(unk_id);
                ids.push(id);
            }
        }

        ids
    }

    /// Convert token IDs back to a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        let mut first = true;

        for &id in ids {
            let tok = self
                .id_to_token
                .get(id)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");

            // Skip special tokens in output
            if self.special_tokens.contains_key(tok) {
                continue;
            }

            if let Some(word_part) = tok.strip_suffix("</w>") {
                // End of word token: strip </w>, add space before if not first
                if !first {
                    result.push(' ');
                }
                result.push_str(word_part);
                first = false;
            } else {
                result.push_str(tok);
            }
        }

        result
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Serialise vocab + merges to a JSON string.
    pub fn to_json(&self) -> Result<String, NeuralError> {
        // Build a serialisable representation
        let merges_list: Vec<(String, String)> = self.merges.clone();
        let vocab_list: Vec<(String, usize)> =
            self.vocab.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let special_list: Vec<(String, usize)> = self
            .special_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        let obj = serde_json::json!({
            "vocab": vocab_list,
            "merges": merges_list,
            "special_tokens": special_list,
        });

        serde_json::to_string(&obj)
            .map_err(|e| NeuralError::SerializationError(format!("BPE serialization failed: {e}")))
    }

    /// Deserialise from a JSON string produced by [`to_json`].
    pub fn from_json(s: &str) -> Result<Self, NeuralError> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| {
            NeuralError::DeserializationError(format!("BPE JSON parse failed: {e}"))
        })?;

        let vocab_arr = v["vocab"]
            .as_array()
            .ok_or_else(|| NeuralError::DeserializationError("missing 'vocab' field".into()))?;
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut max_id = 0usize;
        for item in vocab_arr {
            let pair = item
                .as_array()
                .ok_or_else(|| NeuralError::DeserializationError("vocab entry not array".into()))?;
            let token = pair[0]
                .as_str()
                .ok_or_else(|| NeuralError::DeserializationError("vocab token not str".into()))?
                .to_string();
            let id = pair[1]
                .as_u64()
                .ok_or_else(|| NeuralError::DeserializationError("vocab id not u64".into()))?
                as usize;
            max_id = max_id.max(id);
            vocab.insert(token, id);
        }

        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in &vocab {
            if id < id_to_token.len() {
                id_to_token[id] = token.clone();
            }
        }

        let merges_arr = v["merges"]
            .as_array()
            .ok_or_else(|| NeuralError::DeserializationError("missing 'merges' field".into()))?;
        let mut merges: Vec<(String, String)> = Vec::new();
        for item in merges_arr {
            let pair = item
                .as_array()
                .ok_or_else(|| NeuralError::DeserializationError("merge entry not array".into()))?;
            let l = pair[0]
                .as_str()
                .ok_or_else(|| NeuralError::DeserializationError("merge left not str".into()))?
                .to_string();
            let r = pair[1]
                .as_str()
                .ok_or_else(|| NeuralError::DeserializationError("merge right not str".into()))?
                .to_string();
            merges.push((l, r));
        }

        let special_arr = v["special_tokens"].as_array().ok_or_else(|| {
            NeuralError::DeserializationError("missing 'special_tokens' field".into())
        })?;
        let mut special_tokens: HashMap<String, usize> = HashMap::new();
        for item in special_arr {
            let pair = item.as_array().ok_or_else(|| {
                NeuralError::DeserializationError("special entry not array".into())
            })?;
            let token = pair[0]
                .as_str()
                .ok_or_else(|| NeuralError::DeserializationError("special token not str".into()))?
                .to_string();
            let id = pair[1]
                .as_u64()
                .ok_or_else(|| NeuralError::DeserializationError("special id not u64".into()))?
                as usize;
            special_tokens.insert(token, id);
        }

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
        })
    }

    /// Return the id for the special `<unk>` token.
    pub fn unk_id(&self) -> usize {
        self.special_tokens.get("<unk>").copied().unwrap_or(1)
    }

    /// Return the id for the special `<pad>` token.
    pub fn pad_id(&self) -> usize {
        self.special_tokens.get("<pad>").copied().unwrap_or(0)
    }

    /// Return the id for the special `<bos>` token.
    pub fn bos_id(&self) -> usize {
        self.special_tokens.get("<bos>").copied().unwrap_or(2)
    }

    /// Return the id for the special `<eos>` token.
    pub fn eos_id(&self) -> usize {
        self.special_tokens.get("<eos>").copied().unwrap_or(3)
    }
}

// ---------------------------------------------------------------------------
// BPE helpers
// ---------------------------------------------------------------------------

/// Count the frequency of each adjacent pair across all words.
fn count_pairs(word_freqs: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
    let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
    for (tokens, &freq) in word_freqs {
        for window in tokens.windows(2) {
            *pair_freqs
                .entry((window[0].clone(), window[1].clone()))
                .or_insert(0) += freq;
        }
    }
    pair_freqs
}

/// Replace every occurrence of `(left, right)` in `tokens` with `merged`.
fn apply_merge(tokens: &[String], left: &str, right: &str, merged: &str) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
            result.push(merged.to_string());
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Character-level tokenizer
// ---------------------------------------------------------------------------

/// Character-level tokenizer.
///
/// Token id 0 is always `<pad>`, id 1 is always `<unk>`.
/// Character ids start at 2.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: Vec<Option<char>>,
}

impl CharTokenizer {
    /// Construct from an explicit character set.
    ///
    /// id 0 → `<pad>`, id 1 → `<unk>`, ids 2..N → chars in order.
    pub fn new(chars: &[char]) -> Self {
        let mut char_to_id: HashMap<char, usize> = HashMap::new();
        // 0 = <pad>, 1 = <unk>
        let mut id_to_char: Vec<Option<char>> = vec![None, None];

        for &c in chars {
            if let std::collections::hash_map::Entry::Vacant(e) = char_to_id.entry(c) {
                let id = id_to_char.len();
                e.insert(id);
                id_to_char.push(Some(c));
            }
        }

        Self {
            char_to_id,
            id_to_char,
        }
    }

    /// Build from a collection of texts.
    pub fn from_corpus(texts: &[&str]) -> Self {
        let mut chars: Vec<char> = texts
            .iter()
            .flat_map(|t| t.chars())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort_unstable();
        Self::new(&chars)
    }

    /// Encode text into character token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| self.char_to_id.get(&c).copied().unwrap_or(1)) // unk = 1
            .collect()
    }

    /// Decode character token ids back to a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_char.get(id).and_then(|opt| *opt))
            .collect()
    }

    /// Return the total vocabulary size (including `<pad>` and `<unk>`).
    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    /// Id for `<pad>` (always 0).
    pub fn pad_id(&self) -> usize {
        0
    }

    /// Id for `<unk>` (always 1).
    pub fn unk_id(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Word-level tokenizer
// ---------------------------------------------------------------------------

/// Word-level (whitespace) tokenizer.
///
/// id 0 → `<pad>`, id 1 → `<unk>`, then words sorted by descending frequency.
#[derive(Debug, Clone)]
pub struct WordTokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
    unk_id: usize,
}

impl WordTokenizer {
    /// Build from a corpus.
    ///
    /// Words are ordered by descending frequency; ties broken lexicographically.
    /// `max_vocab` limits the vocabulary size (including `<pad>` and `<unk>`).
    pub fn from_corpus(texts: &[&str], max_vocab: Option<usize>) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for word in text.split_whitespace() {
                *freq.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Sort by (descending frequency, ascending lexicographic order)
        let mut words: Vec<(String, usize)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut id_to_word: Vec<String> = vec!["<pad>".to_string(), "<unk>".to_string()];
        let unk_id = 1usize;

        let max_words = max_vocab.unwrap_or(usize::MAX).saturating_sub(2);
        for (word, _) in words.into_iter().take(max_words) {
            id_to_word.push(word);
        }

        let word_to_id: HashMap<String, usize> = id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        Self {
            word_to_id,
            id_to_word,
            unk_id,
        }
    }

    /// Encode whitespace-tokenized text into word ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| self.word_to_id.get(w).copied().unwrap_or(self.unk_id))
            .collect()
    }

    /// Decode word ids back to a space-separated string.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_word.get(id))
            .filter(|w| *w != "<pad>")
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Return the total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    /// Id for `<unk>`.
    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    /// Id for `<pad>`.
    pub fn pad_id(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BPE tests ---

    #[test]
    fn test_bpe_new_has_special_tokens() {
        let tok = BpeTokenizer::new();
        assert!(tok.vocab.contains_key("<pad>"));
        assert!(tok.vocab.contains_key("<unk>"));
        assert!(tok.vocab.contains_key("<bos>"));
        assert!(tok.vocab.contains_key("<eos>"));
        assert_eq!(tok.vocab_size(), 4);
    }

    #[test]
    fn test_bpe_train_grows_vocab() {
        let corpus = ["hello world", "hello there", "world peace"];
        let tok = BpeTokenizer::train(&corpus, 30).expect("BPE training failed");
        // Vocab should be between initial chars and vocab_size
        assert!(tok.vocab_size() > 4, "vocab grew beyond special tokens");
        assert!(tok.vocab_size() <= 30, "vocab did not exceed target");
    }

    #[test]
    fn test_bpe_train_too_small_vocab() {
        let corpus = ["hello"];
        assert!(BpeTokenizer::train(&corpus, 3).is_err());
    }

    #[test]
    fn test_bpe_encode_returns_ids_in_vocab() {
        let corpus = ["hello world", "world cup"];
        let tok = BpeTokenizer::train(&corpus, 40).expect("training");
        let ids = tok.encode("hello world");
        assert!(!ids.is_empty());
        for &id in &ids {
            assert!(id < tok.vocab_size(), "id {id} out of range");
        }
    }

    #[test]
    fn test_bpe_encode_unknown_char_gives_unk() {
        let corpus = ["abc def"];
        let tok = BpeTokenizer::train(&corpus, 20).expect("training");
        // '☺' definitely not in corpus
        let ids = tok.encode("☺");
        assert_eq!(ids[0], tok.unk_id());
    }

    #[test]
    fn test_bpe_decode_round_trip_single_word() {
        let corpus = ["hello world", "hello there"];
        let tok = BpeTokenizer::train(&corpus, 40).expect("training");
        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_decode_round_trip_multi_word() {
        let corpus = [
            "the cat sat on the mat",
            "the cat is on the mat",
            "the dog sat on the rug",
        ];
        let tok = BpeTokenizer::train(&corpus, 60).expect("training");
        let text = "the cat";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_special_token_ids_are_small() {
        let tok = BpeTokenizer::new();
        assert!(tok.pad_id() < 4);
        assert!(tok.unk_id() < 4);
        assert!(tok.bos_id() < 4);
        assert!(tok.eos_id() < 4);
    }

    #[test]
    fn test_bpe_to_json_and_from_json_round_trip() {
        let corpus = ["hello world", "bye world"];
        let tok = BpeTokenizer::train(&corpus, 30).expect("training");
        let json = tok.to_json().expect("serialise");
        let tok2 = BpeTokenizer::from_json(&json).expect("deserialise");
        assert_eq!(tok.vocab_size(), tok2.vocab_size());
        assert_eq!(tok.merges.len(), tok2.merges.len());
    }

    #[test]
    fn test_bpe_from_json_bad_input() {
        assert!(BpeTokenizer::from_json("not json").is_err());
        assert!(BpeTokenizer::from_json("{}").is_err());
    }

    #[test]
    fn test_bpe_encode_decode_consistency_after_json() {
        let corpus = ["natural language processing"];
        let tok = BpeTokenizer::train(&corpus, 40).expect("training");
        let json = tok.to_json().expect("serialise");
        let tok2 = BpeTokenizer::from_json(&json).expect("deserialise");
        let ids1 = tok.encode("natural");
        let ids2 = tok2.encode("natural");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn test_bpe_empty_corpus_still_has_special_tokens() {
        // Empty corpus → only special tokens in initial vocab; no merges possible
        let tok = BpeTokenizer::train(&[], 10).expect("training with empty corpus");
        assert!(tok.vocab_size() >= 4);
    }

    #[test]
    fn test_bpe_merge_count_increases_with_more_merges() {
        let corpus = ["aaaa bbbb cccc", "aaaa bbbb dddd", "aaaa cccc dddd"];
        let tok_small = BpeTokenizer::train(&corpus, 15).expect("small");
        let tok_large = BpeTokenizer::train(&corpus, 25).expect("large");
        assert!(tok_large.merges.len() >= tok_small.merges.len());
    }

    #[test]
    fn test_bpe_single_char_corpus() {
        let corpus = ["a a a a a"];
        let tok = BpeTokenizer::train(&corpus, 10).expect("training");
        assert!(tok.vocab_size() > 0);
    }

    // --- CharTokenizer tests ---

    #[test]
    fn test_char_tokenizer_from_corpus_vocab_size() {
        let texts = ["hello", "world"];
        let tok = CharTokenizer::from_corpus(&texts);
        // unique chars in "helloworld" = {h,e,l,o,w,r,d} = 7  + pad + unk = 9
        assert_eq!(tok.vocab_size(), 9);
    }

    #[test]
    fn test_char_tokenizer_encode_decode_round_trip() {
        let texts = ["hello world", "foo bar"];
        let tok = CharTokenizer::from_corpus(&texts);
        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_char_tokenizer_unk_id_is_1() {
        let tok = CharTokenizer::new(&['a', 'b']);
        assert_eq!(tok.unk_id(), 1);
    }

    #[test]
    fn test_char_tokenizer_pad_id_is_0() {
        let tok = CharTokenizer::new(&['a', 'b']);
        assert_eq!(tok.pad_id(), 0);
    }

    #[test]
    fn test_char_tokenizer_unknown_char_maps_to_unk() {
        let tok = CharTokenizer::new(&['a', 'b', 'c']);
        let ids = tok.encode("z");
        assert_eq!(ids, vec![tok.unk_id()]);
    }

    #[test]
    fn test_char_tokenizer_empty_text() {
        let tok = CharTokenizer::from_corpus(&["hello"]);
        let ids = tok.encode("");
        assert!(ids.is_empty());
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_char_tokenizer_new_explicit() {
        let tok = CharTokenizer::new(&['x', 'y', 'z']);
        // pad=0, unk=1, x=2, y=3, z=4
        assert_eq!(tok.vocab_size(), 5);
        assert_eq!(tok.encode("x"), vec![2]);
    }

    // --- WordTokenizer tests ---

    #[test]
    fn test_word_tokenizer_vocab_size() {
        let texts = ["the cat sat", "the cat ran", "a dog ran"];
        let tok = WordTokenizer::from_corpus(&texts, None);
        // words: the(2), cat(2), ran(2), sat(1), a(1), dog(1) + pad + unk = 8
        assert_eq!(tok.vocab_size(), 8);
    }

    #[test]
    fn test_word_tokenizer_encode_decode_round_trip() {
        let texts = ["hello world", "goodbye world"];
        let tok = WordTokenizer::from_corpus(&texts, None);
        let ids = tok.encode("hello world");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_word_tokenizer_unk_for_oov() {
        let texts = ["hello world"];
        let tok = WordTokenizer::from_corpus(&texts, None);
        let ids = tok.encode("unknown_word");
        assert_eq!(ids, vec![tok.unk_id()]);
    }

    #[test]
    fn test_word_tokenizer_max_vocab_limits_size() {
        let texts = ["a b c d e f g h i j"];
        let tok = WordTokenizer::from_corpus(&texts, Some(5));
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn test_word_tokenizer_pad_excluded_from_decode() {
        let texts = ["hello world"];
        let tok = WordTokenizer::from_corpus(&texts, None);
        // decode with pad id (0) should skip it
        let decoded = tok.decode(&[0]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_word_tokenizer_empty_text() {
        let texts = ["hello"];
        let tok = WordTokenizer::from_corpus(&texts, None);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }
}
