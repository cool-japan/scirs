//! Byte-Pair Encoding (BPE) tokenizer — training, encoding, decoding.
//!
//! This module provides a *struct-oriented* BPE tokenizer that wraps a
//! [`BpeVocab`] with special-token support and serialisation via a
//! JSON-like text format.  It complements the flat functional API in
//! [`crate::bpe_tokenizer`].

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ─── BpeVocab ────────────────────────────────────────────────────────────────

/// BPE vocabulary: token ↔ ID mapping together with ordered merge rules.
#[derive(Debug, Clone, Default)]
pub struct BpeVocab {
    /// token string → integer ID
    pub token_to_id: HashMap<String, u32>,
    /// integer ID → token string
    pub id_to_token: Vec<String>,
    /// Ordered merge operations learned during training.
    /// Earlier entries have higher priority.
    pub merges: Vec<(String, String)>,
}

impl BpeVocab {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a token, returning its ID.  Idempotent.
    pub fn add_token(&mut self, token: impl Into<String>) -> u32 {
        let token = token.into();
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.push(token);
        id
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

// ─── BpeTokenizer ────────────────────────────────────────────────────────────

/// A trained BPE tokenizer with special-token support.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: BpeVocab,
    /// Well-known special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`.
    special_tokens: HashMap<String, u32>,
    /// The UNK token ID (used when a character is out of vocabulary).
    unk_id: u32,
    max_vocab_size: usize,
}

impl BpeTokenizer {
    // ── Internal helpers ──────────────────────────────────────────────────

    /// Split `word` into individual UTF-8 characters, appending `</w>` to the
    /// last symbol as the standard GPT-2 word-boundary marker.
    fn word_to_symbols(word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(last) = syms.last_mut() {
            last.push_str("</w>");
        }
        syms
    }

    /// Apply the learned merge rules to a symbol sequence, returning the
    /// merged (tokenised) form.
    fn apply_merges<'a>(
        symbols: &[String],
        merge_priorities: &HashMap<(&'a str, &'a str), usize>,
        merges: &'a [(String, String)],
    ) -> Vec<String> {
        // We need owned Strings because we mutate during merge
        let mut syms: Vec<String> = symbols.to_vec();
        loop {
            let mut best_rank = usize::MAX;
            let mut best_pos = usize::MAX;

            for i in 0..syms.len().saturating_sub(1) {
                // Build a temporary key that lives long enough for the lookup
                if let Some(&rank) = merge_priorities.get(&(syms[i].as_str(), syms[i + 1].as_str()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }

            if best_pos == usize::MAX {
                break;
            }

            // Merge syms[best_pos] and syms[best_pos + 1]
            let merged = format!("{}{}", merges[best_rank].0, merges[best_rank].1);
            syms[best_pos] = merged;
            syms.remove(best_pos + 1);
        }
        syms
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Train a BPE tokenizer from a raw-text corpus.
    ///
    /// # Arguments
    /// * `corpus` – slice of text strings to train on.
    /// * `vocab_size` – target vocabulary size (including special tokens and
    ///   initial character alphabet).
    /// * `min_frequency` – pairs occurring fewer than this many times are
    ///   never merged.
    pub fn train(corpus: &[&str], vocab_size: usize, min_frequency: usize) -> Result<Self> {
        if vocab_size < 4 {
            return Err(TextError::InvalidInput(
                "vocab_size must be at least 4 (special tokens + alphabet)".to_string(),
            ));
        }

        // ── Step 1: count word frequencies ───────────────────────────────
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        for text in corpus {
            for word in text.split_whitespace() {
                let syms = Self::word_to_symbols(word);
                *word_freqs.entry(syms).or_insert(0) += 1;
            }
        }

        // ── Step 2: build initial character alphabet ──────────────────────
        let mut vocab = BpeVocab::new();
        let special: Vec<(&str, u32)> = vec![
            ("<pad>", vocab.add_token("<pad>")),
            ("<unk>", vocab.add_token("<unk>")),
            ("<bos>", vocab.add_token("<bos>")),
            ("<eos>", vocab.add_token("<eos>")),
        ];
        let mut special_tokens: HashMap<String, u32> =
            special.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let unk_id = special_tokens["<unk>"];

        // Add all individual characters to vocab
        let mut char_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for symbols in word_freqs.keys() {
            for s in symbols {
                char_set.insert(s.clone());
            }
        }
        for c in &char_set {
            vocab.add_token(c.as_str());
        }

        // ── Step 3: iteratively compute & apply best merge ────────────────
        let num_merges = vocab_size.saturating_sub(vocab.len());
        let merges = compute_merges(&word_freqs, num_merges, min_frequency);

        for (a, b) in &merges {
            let merged = format!("{}{}", a, b);
            vocab.add_token(merged.as_str());
        }
        vocab.merges = merges;

        Ok(BpeTokenizer {
            vocab,
            special_tokens,
            unk_id,
            max_vocab_size: vocab_size,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Build merge priority map once
        let merge_priorities: HashMap<(&str, &str), usize> = self
            .vocab
            .merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.as_str(), b.as_str()), rank))
            .collect();

        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let syms = Self::word_to_symbols(word);
            let merged = Self::apply_merges(&syms, &merge_priorities, &self.vocab.merges);
            for tok in &merged {
                let id = self
                    .vocab
                    .token_to_id
                    .get(tok.as_str())
                    .copied()
                    .unwrap_or(self.unk_id);
                ids.push(id);
            }
        }
        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut first = true;
        for &id in ids {
            if let Some(tok) = self.vocab.id_to_token.get(id as usize) {
                // Skip special tokens in decoded output
                if self.special_tokens.values().any(|&sid| sid == id) {
                    continue;
                }
                if tok.ends_with("</w>") {
                    if !first {
                        out.push(' ');
                    }
                    out.push_str(&tok[..tok.len() - 4]);
                    first = false;
                } else {
                    out.push_str(tok.as_str());
                }
            }
        }
        out
    }

    /// Tokenize text, returning subword tokens as strings.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let merge_priorities: HashMap<(&str, &str), usize> = self
            .vocab
            .merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.as_str(), b.as_str()), rank))
            .collect();

        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            let syms = Self::word_to_symbols(word);
            let merged = Self::apply_merges(&syms, &merge_priorities, &self.vocab.merges);
            tokens.extend(merged);
        }
        tokens
    }

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Serialise the tokenizer to a simple JSON-like string.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");

        // vocab
        out.push_str("  \"vocab\": {\n");
        let mut pairs: Vec<_> = self.vocab.token_to_id.iter().collect();
        pairs.sort_by_key(|(_, &id)| id);
        for (i, (tok, id)) in pairs.iter().enumerate() {
            let comma = if i + 1 < pairs.len() { "," } else { "" };
            out.push_str(&format!(
                "    \"{}\": {}{}\\n",
                tok.replace('\\', "\\\\").replace('"', "\\\""),
                id,
                comma
            ));
        }
        out.push_str("  },\n");

        // merges
        out.push_str("  \"merges\": [\n");
        for (i, (a, b)) in self.vocab.merges.iter().enumerate() {
            let comma = if i + 1 < self.vocab.merges.len() {
                ","
            } else {
                ""
            };
            out.push_str(&format!(
                "    [\"{}\", \"{}\"]{}\\n",
                a.replace('"', "\\\""),
                b.replace('"', "\\\""),
                comma
            ));
        }
        out.push_str("  ],\n");

        // special_tokens
        out.push_str("  \"special_tokens\": {\n");
        let mut sp: Vec<_> = self.special_tokens.iter().collect();
        sp.sort_by_key(|(k, _)| (*k).clone());
        for (i, (k, v)) in sp.iter().enumerate() {
            let comma = if i + 1 < sp.len() { "," } else { "" };
            out.push_str(&format!("    \"{}\": {}{}\\n", k, v, comma));
        }
        out.push_str("  }\n}\n");

        out
    }

    /// Deserialise a tokenizer from the format produced by [`to_json`].
    ///
    /// This is a lightweight parser that does not depend on `serde_json`.
    pub fn from_json(json: &str) -> Result<Self> {
        // Very lightweight key-value extraction — sufficient for round-trip
        // with our own to_json output.

        // Extract vocab block
        let vocab_map = extract_json_string_u32_map(json, "vocab")?;
        let mut vocab = BpeVocab::new();
        let mut sorted_vocab: Vec<_> = vocab_map.iter().collect();
        sorted_vocab.sort_by_key(|(_, &id)| id);
        for (tok, _) in sorted_vocab {
            vocab.add_token(tok.as_str());
        }

        // Extract merges
        let merges = extract_json_merge_pairs(json)?;
        vocab.merges = merges;

        // Extract special tokens
        let sp_map = extract_json_string_u32_map(json, "special_tokens")?;
        let unk_id = sp_map.get("<unk>").copied().unwrap_or(1);

        Ok(BpeTokenizer {
            max_vocab_size: vocab.len(),
            vocab,
            special_tokens: sp_map,
            unk_id,
        })
    }

    /// Access the underlying vocabulary (read-only).
    pub fn vocab(&self) -> &BpeVocab {
        &self.vocab
    }

    /// Access special tokens map.
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}

// ─── BPE merge computation ────────────────────────────────────────────────────

/// Compute up to `num_merges` BPE merge operations from word-frequency counts.
///
/// Each entry in the returned `Vec` is `(left_symbol, right_symbol)` in the
/// order they should be applied.
pub fn compute_merges(
    word_freqs: &HashMap<Vec<String>, usize>,
    num_merges: usize,
    min_frequency: usize,
) -> Vec<(String, String)> {
    // Make a mutable working copy
    let mut seqs: Vec<(Vec<String>, usize)> =
        word_freqs.iter().map(|(k, &v)| (k.clone(), v)).collect();

    let mut merges: Vec<(String, String)> = Vec::with_capacity(num_merges);

    for _ in 0..num_merges {
        // Count pair frequencies
        let mut pair_freqs: HashMap<(&str, &str), usize> = HashMap::new();
        for (syms, freq) in &seqs {
            for w in syms.windows(2) {
                *pair_freqs.entry((&w[0], &w[1])).or_insert(0) += freq;
            }
        }

        // Find the best pair (highest frequency; tie-break: lexicographic)
        let best = pair_freqs
            .iter()
            .filter(|(_, &f)| f >= min_frequency)
            .max_by(|(k1, &f1), (k2, &f2)| f1.cmp(&f2).then_with(|| k1.cmp(k2).reverse()));

        let ((left, right), _) = match best {
            Some(entry) => entry,
            None => break,
        };

        let left = left.to_string();
        let right = right.to_string();
        let merged = format!("{}{}", left, right);

        // Apply the merge to all sequences
        for (syms, _) in &mut seqs {
            let mut new_syms: Vec<String> = Vec::with_capacity(syms.len());
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len() && syms[i] == left && syms[i + 1] == right {
                    new_syms.push(merged.clone());
                    i += 2;
                } else {
                    new_syms.push(syms[i].clone());
                    i += 1;
                }
            }
            *syms = new_syms;
        }

        merges.push((left, right));
    }

    merges
}

// ─── Lightweight JSON helpers ─────────────────────────────────────────────────

/// Extract a `{"key": u32, ...}` block from the JSON string.
fn extract_json_string_u32_map(json: &str, block_name: &str) -> Result<HashMap<String, u32>> {
    let needle = format!("\"{}\":", block_name);
    let start = json
        .find(&needle)
        .ok_or_else(|| TextError::IoError(format!("missing block '{}'", block_name)))?
        + needle.len();

    // Find opening brace
    let block_start = json[start..]
        .find('{')
        .ok_or_else(|| TextError::IoError(format!("malformed block '{}'", block_name)))?
        + start;
    // Find matching closing brace (simple depth counting)
    let block_end = find_matching_brace(json, block_start)?;
    let block = &json[block_start + 1..block_end];

    let mut map = HashMap::new();
    // Each line is like: "token": 42,
    for line in block.lines() {
        let line = line.trim().trim_end_matches(',');
        if line.is_empty() || line == "{" || line == "}" {
            continue;
        }
        // Split at the colon separating key and value
        if let Some(colon) = line.rfind(':') {
            let key_part = line[..colon].trim();
            let val_part = line[colon + 1..].trim();
            if key_part.starts_with('"') && key_part.ends_with('"') {
                let key = key_part[1..key_part.len() - 1].to_string();
                if let Ok(val) = val_part.parse::<u32>() {
                    map.insert(key, val);
                }
            }
        }
    }
    Ok(map)
}

/// Extract merge pairs `[["a","b"], ...]` from the JSON string.
fn extract_json_merge_pairs(json: &str) -> Result<Vec<(String, String)>> {
    let needle = "\"merges\":";
    let start = json
        .find(needle)
        .ok_or_else(|| TextError::IoError("missing 'merges' block".to_string()))?
        + needle.len();

    let arr_start = json[start..]
        .find('[')
        .ok_or_else(|| TextError::IoError("malformed 'merges' block".to_string()))?
        + start;

    let arr_end = find_matching_bracket(json, arr_start)?;
    let block = &json[arr_start + 1..arr_end];

    let mut pairs = Vec::new();
    // Each line is like: ["a", "b"],
    for line in block.lines() {
        let line = line.trim().trim_end_matches(',');
        if !line.starts_with('[') {
            continue;
        }
        let inner = line.trim_start_matches('[').trim_end_matches(']');
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() >= 2 {
            let a = parts[0].trim().trim_matches('"').to_string();
            let b = parts[1].trim().trim_matches('"').to_string();
            pairs.push((a, b));
        }
    }
    Ok(pairs)
}

fn find_matching_brace(s: &str, open: usize) -> Result<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    for i in open..bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }
    Err(TextError::IoError("unmatched '{' in JSON".to_string()))
}

fn find_matching_bracket(s: &str, open: usize) -> Result<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    for i in open..bytes.len() {
        match bytes[i] {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }
    Err(TextError::IoError("unmatched '[' in JSON".to_string()))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_corpus() -> Vec<&'static str> {
        vec![
            "low lower lowest",
            "newer newest new",
            "lower lowest newer newest",
            "low low low low newer newer",
        ]
    }

    #[test]
    fn test_bpe_train_basic() {
        let corpus = toy_corpus();
        let tok = BpeTokenizer::train(&corpus, 50, 1).expect("train failed");
        assert!(tok.vocab_size() > 4); // at least special tokens + alphabet + some merges
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let corpus = toy_corpus();
        let tok = BpeTokenizer::train(&corpus, 60, 1).expect("train failed");
        let text = "low lower";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_tokenize() {
        let corpus = toy_corpus();
        let tok = BpeTokenizer::train(&corpus, 60, 1).expect("train failed");
        let tokens = tok.tokenize("newer");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_special_tokens() {
        let corpus = toy_corpus();
        let tok = BpeTokenizer::train(&corpus, 50, 1).expect("train failed");
        assert!(tok.special_tokens().contains_key("<pad>"));
        assert!(tok.special_tokens().contains_key("<unk>"));
        assert!(tok.special_tokens().contains_key("<bos>"));
        assert!(tok.special_tokens().contains_key("<eos>"));
    }

    #[test]
    fn test_bpe_min_frequency() {
        let corpus = vec!["rare word appears once only"];
        // With min_frequency=2, the rare pair should not be merged
        let tok = BpeTokenizer::train(&corpus, 40, 2).expect("train failed");
        assert!(tok.vocab_size() >= 4); // at least special tokens
    }

    #[test]
    fn test_compute_merges() {
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        word_freqs.insert(vec!["l".into(), "o".into(), "w".into(), "</w>".into()], 5);
        word_freqs.insert(
            vec![
                "l".into(),
                "o".into(),
                "w".into(),
                "e".into(),
                "r".into(),
                "</w>".into(),
            ],
            3,
        );
        let merges = compute_merges(&word_freqs, 4, 1);
        assert!(!merges.is_empty());
    }
}
