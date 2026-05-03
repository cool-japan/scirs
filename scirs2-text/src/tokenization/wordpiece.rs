//! WordPiece tokenizer — the subword tokenization used by BERT.
//!
//! Implements:
//! - [`BasicTokenizer`]: whitespace + punctuation splitting with optional
//!   lower-casing and accent stripping.
//! - [`WordPieceTokenizer`]: greedy longest-match subword tokenisation.

use std::collections::HashMap;

use crate::error::{Result, TextError};

// ─── BasicTokenizer ───────────────────────────────────────────────────────────

/// BERT-style basic tokenizer: split on whitespace and punctuation, optionally
/// lower-case and strip accent marks.
#[derive(Debug, Clone)]
pub struct BasicTokenizer {
    /// Lowercase all characters before tokenising.
    pub do_lower_case: bool,
    /// Strip Unicode combining characters (accents / diacritics).
    pub strip_accents: bool,
}

impl BasicTokenizer {
    /// Create a new [`BasicTokenizer`].
    pub fn new(do_lower_case: bool, strip_accents: bool) -> Self {
        BasicTokenizer {
            do_lower_case,
            strip_accents,
        }
    }

    /// Tokenize `text` into a list of token strings.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.do_lower_case {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Strip accent marks (combining Unicode characters, category Mn)
        let text = if self.strip_accents {
            strip_accents_str(&text)
        } else {
            text
        };

        // Insert whitespace around punctuation and split
        let mut spaced = String::with_capacity(text.len() + 16);
        for ch in text.chars() {
            if ch.is_whitespace() {
                spaced.push(' ');
            } else if is_punctuation(ch) || is_chinese_char(ch) {
                spaced.push(' ');
                spaced.push(ch);
                spaced.push(' ');
            } else {
                spaced.push(ch);
            }
        }

        spaced
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        BasicTokenizer::new(true, true)
    }
}

/// Return `true` for characters in Unicode general category Mn (non-spacing
/// combining marks).  We use an approximate range check.
fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    // Combining Diacritical Marks U+0300–U+036F
    // Combining Diacritical Marks Supplement U+1DC0–U+1DFF
    // Combining Diacritical Marks Extended U+1AB0–U+1AFF
    (0x0300..=0x036F).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
}

/// Decompose `s` to NFD then drop combining characters.
fn strip_accents_str(s: &str) -> String {
    // Manual NFD-like decomposition: use unicode_normalization if available, else
    // do a best-effort strip of common combining marks.
    use unicode_normalization::UnicodeNormalization;
    s.nfd().filter(|&ch| !is_combining_mark(ch)).collect()
}

/// `true` for ASCII punctuation and Unicode punctuation categories.
fn is_punctuation(ch: char) -> bool {
    if (ch as u32) <= 47
        || (58..=64).contains(&(ch as u32))
        || (91..=96).contains(&(ch as u32))
        || (123..=126).contains(&(ch as u32))
    {
        return true;
    }
    ch.is_ascii_punctuation() || ch == '。' || ch == '，'
}

/// `true` for CJK Unified Ideograph ranges.
fn is_chinese_char(ch: char) -> bool {
    let cp = ch as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

// ─── WordPieceTokenizer ───────────────────────────────────────────────────────

/// BERT-style WordPiece tokenizer.
///
/// Words are split by [`BasicTokenizer`] first, then each word is broken into
/// the longest matching subwords from the vocabulary.  Continuation subwords
/// are prefixed with `##`.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    unk_id: u32,
    max_input_chars_per_word: usize,
    basic: BasicTokenizer,
}

impl WordPieceTokenizer {
    // Special token constants
    const UNK_TOKEN: &'static str = "[UNK]";
    const CLS_TOKEN: &'static str = "[CLS]";
    const SEP_TOKEN: &'static str = "[SEP]";
    const PAD_TOKEN: &'static str = "[PAD]";
    const MASK_TOKEN: &'static str = "[MASK]";

    /// Build from an existing `token → id` vocabulary map.
    ///
    /// The vocabulary must contain at least `[UNK]`.  If `[UNK]` is missing
    /// it is added with ID `vocab.len()`.
    pub fn from_vocab(mut vocab: HashMap<String, u32>) -> Self {
        // Ensure [UNK] exists
        if !vocab.contains_key(Self::UNK_TOKEN) {
            let next_id = vocab.len() as u32;
            vocab.insert(Self::UNK_TOKEN.to_string(), next_id);
        }
        let unk_id = vocab[Self::UNK_TOKEN];

        // Build id→token from max ID
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (tok, &id) in &vocab {
            if let Some(slot) = id_to_token.get_mut(id as usize) {
                *slot = tok.clone();
            }
        }

        WordPieceTokenizer {
            vocab,
            id_to_token,
            unk_id,
            max_input_chars_per_word: 200,
            basic: BasicTokenizer::default(),
        }
    }

    /// Build a minimal tokenizer from a plain vocabulary list (one token per
    /// entry; ID = index).
    pub fn from_vocab_list(tokens: &[impl AsRef<str>]) -> Self {
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_ref().to_string(), i as u32))
            .collect();
        Self::from_vocab(vocab)
    }

    /// Set the maximum number of input characters per word before falling back
    /// to `[UNK]`.
    pub fn with_max_input_chars(mut self, n: usize) -> Self {
        self.max_input_chars_per_word = n;
        self
    }

    // ── Core subword splitting ──────────────────────────────────────────

    /// Greedy longest-match WordPiece segmentation of a single `word`.
    fn wordpiece_word(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return vec![Self::UNK_TOKEN.to_string()];
        }

        let mut sub_tokens: Vec<String> = Vec::new();
        let mut start = 0usize;
        let n = chars.len();
        let mut is_bad = false;

        while start < n {
            let mut end = n;
            let mut found: Option<String> = None;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start == 0 {
                    substr.clone()
                } else {
                    format!("##{}", substr)
                };

                if self.vocab.contains_key(&candidate) {
                    found = Some(candidate);
                    break;
                }
                if end == start + 1 {
                    // Single character not in vocab → whole word is unknown
                    is_bad = true;
                    break;
                }
                end -= 1;
            }

            if is_bad {
                break;
            }

            match found {
                Some(tok) => {
                    sub_tokens.push(tok);
                    start = end;
                }
                None => {
                    is_bad = true;
                    break;
                }
            }
        }

        if is_bad {
            vec![Self::UNK_TOKEN.to_string()]
        } else {
            sub_tokens
        }
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Tokenize `text` to subword token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenize_to_strings(text)
            .iter()
            .map(|tok| self.vocab.get(tok.as_str()).copied().unwrap_or(self.unk_id))
            .collect()
    }

    /// Tokenize `text` to subword token strings.
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        let words = self.basic.tokenize(text);
        words.iter().flat_map(|w| self.wordpiece_word(w)).collect()
    }

    /// Decode a sequence of token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let tok = self
                .id_to_token
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("[UNK]");

            // Skip padding
            if tok == Self::PAD_TOKEN {
                continue;
            }

            if tok.starts_with("##") {
                out.push_str(&tok[2..]);
            } else if !out.is_empty() && tok != Self::CLS_TOKEN && tok != Self::SEP_TOKEN {
                out.push(' ');
                out.push_str(tok);
            } else {
                out.push_str(tok);
            }
        }
        out
    }

    /// Encode `text` with optional special tokens, padding/truncation to
    /// `max_length`.
    ///
    /// Returns `(input_ids, attention_mask)` where `attention_mask[i] = 1`
    /// for real tokens and `0` for padding.
    pub fn encode(
        &self,
        text: &str,
        max_length: usize,
        add_special_tokens: bool,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        if max_length == 0 {
            return Err(TextError::InvalidInput(
                "max_length must be > 0".to_string(),
            ));
        }

        let cls_id = self
            .vocab
            .get(Self::CLS_TOKEN)
            .copied()
            .unwrap_or(self.unk_id);
        let sep_id = self
            .vocab
            .get(Self::SEP_TOKEN)
            .copied()
            .unwrap_or(self.unk_id);
        let pad_id = self
            .vocab
            .get(Self::PAD_TOKEN)
            .copied()
            .unwrap_or(self.unk_id);

        let token_ids = self.tokenize(text);

        // Reserve space for [CLS] and [SEP] when using special tokens
        let reserve = if add_special_tokens { 2 } else { 0 };
        let content_budget = max_length.saturating_sub(reserve);
        let truncated: Vec<u32> = token_ids.into_iter().take(content_budget).collect();

        let mut ids: Vec<u32> = Vec::with_capacity(max_length);
        if add_special_tokens {
            ids.push(cls_id);
        }
        ids.extend_from_slice(&truncated);
        if add_special_tokens {
            ids.push(sep_id);
        }

        let real_len = ids.len();
        // Pad to max_length
        while ids.len() < max_length {
            ids.push(pad_id);
        }

        let mut mask: Vec<u8> = vec![0u8; max_length];
        for m in mask.iter_mut().take(real_len) {
            *m = 1;
        }

        Ok((ids, mask))
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Return a cloned snapshot of the `token → id` vocabulary map.
    ///
    /// Useful for serialisation (e.g. building a HuggingFace `tokenizers.json`).
    pub fn vocab_snapshot(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn mini_vocab() -> HashMap<String, u32> {
        let mut v = HashMap::new();
        for (i, tok) in [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "he", "llo", "##llo", "world", "##world",
            "want", "##ed", "to", "un", "##want", "##ed", "low", "##er", "##est", "new", "##er",
            "##est", "h", "e", "l", "o", "w", "r", "d",
        ]
        .iter()
        .enumerate()
        {
            v.entry(tok.to_string()).or_insert(i as u32);
        }
        v
    }

    #[test]
    fn test_basic_tokenizer_lower() {
        let tok = BasicTokenizer::new(true, false);
        let tokens = tok.tokenize("Hello, World!");
        assert!(tokens.iter().any(|t| t == "hello"));
        assert!(tokens.iter().any(|t| t == "world"));
        assert!(tokens.iter().any(|t| t == ","));
        assert!(tokens.iter().any(|t| t == "!"));
    }

    #[test]
    fn test_basic_tokenizer_no_lower() {
        let tok = BasicTokenizer::new(false, false);
        let tokens = tok.tokenize("Hello World");
        assert!(tokens.iter().any(|t| t == "Hello"));
        assert!(tokens.iter().any(|t| t == "World"));
    }

    #[test]
    fn test_wordpiece_tokenize_to_strings_known() {
        let vocab = mini_vocab();
        let wp = WordPieceTokenizer::from_vocab(vocab);
        // Words fully in vocab should not become [UNK]
        let tokens = wp.tokenize_to_strings("low");
        assert!(!tokens.iter().any(|t| t == "[UNK]"), "got {:?}", tokens);
    }

    #[test]
    fn test_wordpiece_encode_length() {
        let vocab = mini_vocab();
        let wp = WordPieceTokenizer::from_vocab(vocab);
        let (ids, mask) = wp.encode("low", 8, true).expect("encode failed");
        assert_eq!(ids.len(), 8);
        assert_eq!(mask.len(), 8);
        // First mask value should be 1 ([CLS])
        assert_eq!(mask[0], 1);
    }

    #[test]
    fn test_wordpiece_encode_truncation() {
        let vocab = mini_vocab();
        let wp = WordPieceTokenizer::from_vocab(vocab);
        let (ids, mask) = wp
            .encode("low low low low", 4, true)
            .expect("encode failed");
        assert_eq!(ids.len(), 4);
        assert_eq!(mask.len(), 4);
    }

    #[test]
    fn test_wordpiece_encode_no_special_tokens() {
        let vocab = mini_vocab();
        let wp = WordPieceTokenizer::from_vocab(vocab);
        let (ids, mask) = wp.encode("low", 4, false).expect("encode failed");
        assert_eq!(ids.len(), 4);
        // Real tokens + padding
        assert!(mask[0] == 1);
    }

    #[test]
    fn test_wordpiece_decode_strips_double_hash() {
        let vocab = mini_vocab();
        let wp = WordPieceTokenizer::from_vocab(vocab);
        // low ##er should decode to "lower"
        let low_id = *wp.vocab.get("low").unwrap();
        let er_id = *wp.vocab.get("##er").unwrap();
        let decoded = wp.decode(&[low_id, er_id]);
        assert_eq!(decoded, "lower");
    }

    #[test]
    fn test_basic_tokenizer_punctuation_isolation() {
        let tok = BasicTokenizer::new(false, false);
        let tokens = tok.tokenize("It's fine.");
        // Period should be its own token
        assert!(tokens.contains(&".".to_string()));
    }
}
