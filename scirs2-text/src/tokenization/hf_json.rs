//! HuggingFace `tokenizers.json` format serialisation / deserialisation.
//!
//! The HuggingFace tokenizers library persists tokenizer configurations as
//! a single JSON file.  This module can read and write that format so that
//! tokenizers trained with [`super::wordpiece::WordPieceTokenizer`] or
//! [`crate::gpt_bpe::Gpt2BpeTokenizer`] can be exchanged with the HF
//! ecosystem.
//!
//! # Format summary
//!
//! ```json
//! {
//!   "version": "1.0",
//!   "model": {
//!     "type": "WordPiece",
//!     "unk_token": "[UNK]",
//!     "continuing_subword_prefix": "##",
//!     "max_input_chars_per_word": 100,
//!     "vocab": { "[PAD]": 0, "[UNK]": 1, ... }
//!   },
//!   "added_tokens": [
//!     { "id": 0, "content": "[PAD]", "special": true },
//!     ...
//!   ],
//!   "normalizer": null,
//!   "pre_tokenizer": null,
//!   "post_processor": null,
//!   "decoder": null,
//!   "truncation": null,
//!   "padding": null
//! }
//! ```

use std::collections::HashMap;

use crate::error::{Result, TextError};
use crate::gpt_bpe::Gpt2BpeTokenizer;
use crate::tokenization::wordpiece::WordPieceTokenizer;

// ── Model-type enum ──────────────────────────────────────────────────────────

/// The tokenizer model type encoded in a HuggingFace tokenizers JSON file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HfModelType {
    /// BERT WordPiece model.
    WordPiece,
    /// Byte-Pair Encoding model (GPT-2, RoBERTa, …).
    Bpe,
    /// SentencePiece Unigram Language Model.
    Unigram,
    /// Unknown or unrecognised model type.
    Unknown(String),
}

impl HfModelType {
    /// Convert to the string used in the JSON `model.type` field.
    pub fn as_str(&self) -> &str {
        match self {
            HfModelType::WordPiece => "WordPiece",
            HfModelType::Bpe => "BPE",
            HfModelType::Unigram => "Unigram",
            HfModelType::Unknown(s) => s.as_str(),
        }
    }

    /// Parse from the JSON string representation.
    pub fn parse(s: &str) -> Self {
        match s {
            "WordPiece" | "wordpiece" | "WORDPIECE" => HfModelType::WordPiece,
            "BPE" | "Bpe" | "bpe" => HfModelType::Bpe,
            "Unigram" | "unigram" | "UNIGRAM" => HfModelType::Unigram,
            other => HfModelType::Unknown(other.to_string()),
        }
    }
}

// ── HfAddedToken ─────────────────────────────────────────────────────────────

/// A special / added token entry in the HF JSON.
#[derive(Debug, Clone)]
pub struct HfAddedToken {
    /// Integer ID in the vocabulary.
    pub id: u32,
    /// Token surface string.
    pub content: String,
    /// Whether this token is a "special" control token.
    pub special: bool,
    /// Whether the token should not be split during pre-tokenisation.
    pub single_word: bool,
    /// Whether this token strips leading whitespace.
    pub lstrip: bool,
    /// Whether this token strips trailing whitespace.
    pub rstrip: bool,
    /// Whether the token is normalised.
    pub normalized: bool,
}

impl HfAddedToken {
    /// Construct a simple special token entry.
    pub fn special(id: u32, content: impl Into<String>) -> Self {
        HfAddedToken {
            id,
            content: content.into(),
            special: true,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
        }
    }

    /// Render to a JSON object string (no trailing newline).
    fn to_json_object(&self) -> String {
        format!(
            r#"{{"id":{},"content":{},"single_word":{},"lstrip":{},"rstrip":{},"normalized":{},"special":{}}}"#,
            self.id,
            json_string(&self.content),
            self.single_word,
            self.lstrip,
            self.rstrip,
            self.normalized,
            self.special,
        )
    }

    /// Parse from a JSON object string (best-effort).
    fn from_json_object(obj: &str) -> Option<Self> {
        let id = parse_u32_field(obj, "id")?;
        let content = parse_string_field(obj, "content")?;
        let special = parse_bool_field(obj, "special").unwrap_or(false);
        let single_word = parse_bool_field(obj, "single_word").unwrap_or(false);
        let lstrip = parse_bool_field(obj, "lstrip").unwrap_or(false);
        let rstrip = parse_bool_field(obj, "rstrip").unwrap_or(false);
        let normalized = parse_bool_field(obj, "normalized").unwrap_or(false);
        Some(HfAddedToken {
            id,
            content,
            special,
            single_word,
            lstrip,
            rstrip,
            normalized,
        })
    }
}

// ── HfModel ──────────────────────────────────────────────────────────────────

/// The `model` object inside a HuggingFace tokenizers JSON.
#[derive(Debug, Clone)]
pub struct HfModel {
    /// The `"type"` discriminant string (e.g. `"WordPiece"`, `"BPE"`).
    pub model_type: String,
    /// Token-string → integer-ID vocabulary mapping.
    pub vocab: HashMap<String, u32>,
    /// Ordered BPE merge rules as `"A B"` strings (BPE only).
    pub merges: Option<Vec<String>>,
    /// The UNK token string.
    pub unk_token: Option<String>,
    /// The continuation subword prefix (WordPiece: `"##"`).
    pub continuing_subword_prefix: Option<String>,
    /// Maximum number of input characters per word before falling back to UNK.
    pub max_input_chars_per_word: Option<u32>,
}

impl HfModel {
    /// Serialise to a JSON object string.
    fn to_json_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        parts.push(format!(r#""type":{}"#, json_string(&self.model_type)));

        if let Some(ref unk) = self.unk_token {
            parts.push(format!(r#""unk_token":{}"#, json_string(unk)));
        }

        if let Some(ref pfx) = self.continuing_subword_prefix {
            parts.push(format!(
                r#""continuing_subword_prefix":{}"#,
                json_string(pfx)
            ));
        }

        if let Some(max_chars) = self.max_input_chars_per_word {
            parts.push(format!(r#""max_input_chars_per_word":{}"#, max_chars));
        }

        // vocab: sort for determinism
        let vocab_entries = {
            let mut sorted: Vec<(&String, &u32)> = self.vocab.iter().collect();
            sorted.sort_by_key(|(_, &id)| id);
            sorted
                .iter()
                .map(|(tok, id)| format!("{}:{}", json_string(tok), id))
                .collect::<Vec<_>>()
                .join(",")
        };
        parts.push(format!(r#""vocab":{{{}}}"#, vocab_entries));

        if let Some(ref merges) = self.merges {
            let merge_strs = merges
                .iter()
                .map(|m| json_string(m))
                .collect::<Vec<_>>()
                .join(",");
            parts.push(format!(r#""merges":[{}]"#, merge_strs));
        }

        format!("{{{}}}", parts.join(","))
    }

    /// Deserialise from the JSON string of a model object.
    fn from_json_str(s: &str) -> Result<Self> {
        let model_type = parse_string_field(s, "type").ok_or_else(|| {
            TextError::InvalidInput("HF JSON: missing model.type field".to_string())
        })?;

        let unk_token = parse_string_field(s, "unk_token");
        let continuing_subword_prefix = parse_string_field(s, "continuing_subword_prefix");
        let max_input_chars_per_word = parse_u32_field(s, "max_input_chars_per_word");

        let vocab = parse_vocab_object(s)?;
        let merges = parse_string_array_field(s, "merges");

        Ok(HfModel {
            model_type,
            vocab,
            merges,
            unk_token,
            continuing_subword_prefix,
            max_input_chars_per_word,
        })
    }
}

// ── HfTokenizerJson ───────────────────────────────────────────────────────────

/// A complete HuggingFace `tokenizers.json` document.
#[derive(Debug, Clone)]
pub struct HfTokenizerJson {
    /// Format version (typically `"1.0"`).
    pub version: String,
    /// The core tokenizer model (vocab, merges, etc.).
    pub model: HfModel,
    /// Extra special tokens added on top of the base vocabulary.
    pub added_tokens: Vec<HfAddedToken>,
    /// Raw JSON for the normalizer component (null if absent).
    pub normalizer_json: Option<String>,
    /// Raw JSON for the pre-tokenizer component.
    pub pre_tokenizer_json: Option<String>,
    /// Raw JSON for the post-processor component.
    pub post_processor_json: Option<String>,
    /// Raw JSON for the decoder component.
    pub decoder_json: Option<String>,
}

impl HfTokenizerJson {
    // ── Constructors ───────────────────────────────────────────────────────

    /// Build a [`HfTokenizerJson`] from a trained [`WordPieceTokenizer`].
    ///
    /// Extracts the vocabulary and annotates the standard BERT special tokens
    /// as `added_tokens`.
    pub fn from_wordpiece(wp: &WordPieceTokenizer) -> Self {
        let vocab: HashMap<String, u32> = wp.vocab_snapshot();

        // Standard BERT special token IDs (use vocab lookup with fallbacks)
        let get = |tok: &str, fallback: u32| -> u32 { vocab.get(tok).copied().unwrap_or(fallback) };

        let added_tokens = vec![
            HfAddedToken::special(get("[PAD]", 0), "[PAD]"),
            HfAddedToken::special(get("[UNK]", 1), "[UNK]"),
            HfAddedToken::special(get("[CLS]", 101), "[CLS]"),
            HfAddedToken::special(get("[SEP]", 102), "[SEP]"),
            HfAddedToken::special(get("[MASK]", 103), "[MASK]"),
        ];

        let model = HfModel {
            model_type: "WordPiece".to_string(),
            vocab,
            merges: None,
            unk_token: Some("[UNK]".to_string()),
            continuing_subword_prefix: Some("##".to_string()),
            max_input_chars_per_word: Some(100),
        };

        HfTokenizerJson {
            version: "1.0".to_string(),
            model,
            added_tokens,
            normalizer_json: None,
            pre_tokenizer_json: None,
            post_processor_json: None,
            decoder_json: None,
        }
    }

    /// Build a [`HfTokenizerJson`] from a trained [`Gpt2BpeTokenizer`].
    pub fn from_gpt2_bpe(bpe: &Gpt2BpeTokenizer) -> Self {
        let vocab: HashMap<String, u32> = bpe.vocab_snapshot();
        let merges: Vec<String> = bpe
            .merges()
            .iter()
            .map(|(a, b)| format!("{} {}", a, b))
            .collect();

        let model = HfModel {
            model_type: "BPE".to_string(),
            vocab,
            merges: Some(merges),
            unk_token: None,
            continuing_subword_prefix: None,
            max_input_chars_per_word: None,
        };

        HfTokenizerJson {
            version: "1.0".to_string(),
            model,
            added_tokens: vec![],
            normalizer_json: None,
            pre_tokenizer_json: None,
            post_processor_json: None,
            decoder_json: None,
        }
    }

    // ── Serialisation ──────────────────────────────────────────────────────

    /// Serialise to a JSON string.
    ///
    /// When the `serde-support` feature is enabled this delegates to
    /// `serde_json`; otherwise a manual serialiser is used.
    pub fn to_json_string(&self) -> String {
        let added_tokens_str = self
            .added_tokens
            .iter()
            .map(|t| t.to_json_object())
            .collect::<Vec<_>>()
            .join(",");

        let null_or =
            |opt: &Option<String>| -> String { opt.as_deref().unwrap_or("null").to_string() };

        format!(
            r#"{{"version":{},"truncation":null,"padding":null,"added_tokens":[{}],"normalizer":{},"pre_tokenizer":{},"post_processor":{},"decoder":{},"model":{}}}"#,
            json_string(&self.version),
            added_tokens_str,
            null_or(&self.normalizer_json),
            null_or(&self.pre_tokenizer_json),
            null_or(&self.post_processor_json),
            null_or(&self.decoder_json),
            self.model.to_json_string(),
        )
    }

    // ── Deserialisation ────────────────────────────────────────────────────

    /// Parse a HuggingFace `tokenizers.json` string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let version = parse_string_field(s, "version").unwrap_or_else(|| "1.0".to_string());

        // Extract the model object
        let model_str = extract_object_field(s, "model").ok_or_else(|| {
            TextError::InvalidInput("HF JSON: missing 'model' object".to_string())
        })?;
        let model = HfModel::from_json_str(model_str)?;

        // Extract added_tokens array
        let added_tokens = extract_array_field(s, "added_tokens")
            .unwrap_or_default()
            .iter()
            .filter_map(|obj| HfAddedToken::from_json_object(obj))
            .collect();

        let normalizer_json = extract_object_field(s, "normalizer").map(|o| o.to_string());
        let pre_tokenizer_json = extract_object_field(s, "pre_tokenizer").map(|o| o.to_string());
        let post_processor_json = extract_object_field(s, "post_processor").map(|o| o.to_string());
        let decoder_json = extract_object_field(s, "decoder").map(|o| o.to_string());

        Ok(HfTokenizerJson {
            version,
            model,
            added_tokens,
            normalizer_json,
            pre_tokenizer_json,
            post_processor_json,
            decoder_json,
        })
    }

    // ── Utilities ──────────────────────────────────────────────────────────

    /// Check that the tokenizer round-trips through JSON without data loss.
    ///
    /// Returns `true` when the vocab sizes before and after serialisation
    /// match and the model type is preserved.
    pub fn wordpiece_roundtrip_check(wp: &WordPieceTokenizer) -> bool {
        let original = Self::from_wordpiece(wp);
        let json = original.to_json_string();
        match Self::from_json_str(&json) {
            Ok(restored) => {
                restored.model.vocab.len() == original.model.vocab.len()
                    && restored.model.model_type == original.model.model_type
            }
            Err(_) => false,
        }
    }
}

// ── Free function ─────────────────────────────────────────────────────────────

/// Peek at a HuggingFace tokenizers JSON string and return the model type.
pub fn detect_model_type(json: &str) -> Result<HfModelType> {
    let model_str = extract_object_field(json, "model").ok_or_else(|| {
        TextError::InvalidInput("HF JSON: could not locate 'model' object".to_string())
    })?;
    let type_str = parse_string_field(model_str, "type")
        .ok_or_else(|| TextError::InvalidInput("HF JSON: missing model.type field".to_string()))?;
    Ok(HfModelType::parse(&type_str))
}

// Note: vocab_snapshot() methods are defined directly on WordPieceTokenizer
// (in tokenization/wordpiece.rs) and Gpt2BpeTokenizer (in gpt_bpe.rs).
// Both return HashMap<String, u32>.

// ── Minimal JSON helpers ─────────────────────────────────────────────────────

/// Encode a string as a JSON string literal (with escaping).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Extract the raw value (string, number, or object) for a JSON key in a
/// *flat* (non-nested) context.  Returns `None` when the key is absent or the
/// value is `null`.
///
/// This is intentionally simple: it searches for `"key":` then extracts the
/// next token.  It works for the HF JSON format because top-level keys are not
/// repeated in nested structures at the same parse depth.
fn extract_json_value<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":", key);
    let pos = json.find(needle.as_str())?;
    let after_key = json[pos + needle.len()..].trim_start();

    if after_key.starts_with("null") {
        return None;
    }

    // Return a slice covering the raw value (string, number, bool, or {…}/[…])
    Some(after_key)
}

/// Parse a JSON string field from a JSON object string.
fn parse_string_field(json: &str, key: &str) -> Option<String> {
    let raw = extract_json_value(json, key)?;
    if !raw.starts_with('"') {
        return None;
    }
    // Walk forward to find the closing unescaped quote
    let mut chars = raw.char_indices().skip(1); // skip opening "
    let mut result = String::new();
    loop {
        match chars.next() {
            None => return None,
            Some((_, '"')) => break,
            Some((_, '\\')) => {
                match chars.next() {
                    Some((_, '"')) => result.push('"'),
                    Some((_, '\\')) => result.push('\\'),
                    Some((_, 'n')) => result.push('\n'),
                    Some((_, 'r')) => result.push('\r'),
                    Some((_, 't')) => result.push('\t'),
                    Some((_, 'u')) => {
                        // \uXXXX
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if let Some((_, c)) = chars.next() {
                                hex.push(c);
                            }
                        }
                        if let Ok(n) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(n) {
                                result.push(c);
                            }
                        }
                    }
                    Some((_, c)) => result.push(c),
                    None => return None,
                }
            }
            Some((_, c)) => result.push(c),
        }
    }
    Some(result)
}

/// Parse a JSON boolean field.
fn parse_bool_field(json: &str, key: &str) -> Option<bool> {
    let raw = extract_json_value(json, key)?;
    if raw.starts_with("true") {
        Some(true)
    } else if raw.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Parse a JSON unsigned-integer field.
fn parse_u32_field(json: &str, key: &str) -> Option<u32> {
    let raw = extract_json_value(json, key)?;
    let num: String = raw.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// Extract the inner text of a JSON object `{...}` for a given key.
/// Returns `None` when the value is `null` or not an object.
fn extract_object_field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let raw = extract_json_value(json, key)?;
    if !raw.starts_with('{') {
        return None;
    }
    // Find matching closing brace
    let end = find_matching_brace(raw, '{', '}')?;
    Some(&raw[..=end])
}

/// Extract each element of a JSON array `[...]` for a given key.
/// Only handles arrays of objects `[{...},{...}]`.
fn extract_array_field(json: &str, key: &str) -> Option<Vec<String>> {
    let raw = extract_json_value(json, key)?;
    if !raw.starts_with('[') {
        return None;
    }
    let end = find_matching_brace(raw, '[', ']')?;
    let inner = &raw[1..end]; // strip [ and ]
    Some(split_json_array_objects(inner))
}

/// Parse a JSON string-array field (e.g. the `merges` list).
fn parse_string_array_field(json: &str, key: &str) -> Option<Vec<String>> {
    let raw = extract_json_value(json, key)?;
    if !raw.starts_with('[') {
        return None;
    }
    let end = find_matching_brace(raw, '[', ']')?;
    let inner = &raw[1..end];

    let mut result = Vec::new();
    let mut remainder = inner.trim();
    while !remainder.is_empty() {
        if remainder.starts_with('"') {
            // Scan for the end of the string
            let mut chars = remainder.char_indices().skip(1);
            let mut s = String::new();
            let mut end_pos = 0;
            let mut found = false;
            loop {
                match chars.next() {
                    None => break,
                    Some((i, '"')) => {
                        end_pos = i;
                        found = true;
                        break;
                    }
                    Some((_, '\\')) => match chars.next() {
                        Some((_, c)) => s.push(c),
                        None => break,
                    },
                    Some((_, c)) => s.push(c),
                }
            }
            if found {
                result.push(s);
                remainder = remainder[end_pos + 1..].trim_start_matches(',').trim();
            } else {
                break;
            }
        } else {
            // Skip non-string element
            let skip = remainder
                .find(',')
                .map(|i| i + 1)
                .unwrap_or(remainder.len());
            remainder = &remainder[skip..];
        }
    }
    Some(result)
}

/// Parse the `vocab` object (`{"token": id, ...}`) embedded within a model
/// object string.
fn parse_vocab_object(json: &str) -> Result<HashMap<String, u32>> {
    // Find the vocab sub-object
    let vocab_raw = extract_object_field(json, "vocab").ok_or_else(|| {
        TextError::InvalidInput("HF JSON: missing model.vocab object".to_string())
    })?;

    let inner = &vocab_raw[1..vocab_raw.len() - 1]; // strip { }
    let mut map = HashMap::new();

    // Parse "token": id pairs separated by commas
    let mut remainder = inner.trim();
    while !remainder.is_empty() {
        if remainder.starts_with('"') {
            // Parse key string
            let key = match parse_json_string_at_start(remainder) {
                Some((s, consumed)) => {
                    remainder = &remainder[consumed..];
                    s
                }
                None => break,
            };
            remainder = remainder.trim_start();
            if !remainder.starts_with(':') {
                break;
            }
            remainder = remainder[1..].trim_start();
            // Parse numeric value
            let num_str: String = remainder
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if num_str.is_empty() {
                break;
            }
            if let Ok(id) = num_str.parse::<u32>() {
                map.insert(key, id);
            }
            remainder = &remainder[num_str.len()..];
            remainder = remainder.trim_start();
            if remainder.starts_with(',') {
                remainder = remainder[1..].trim_start();
            }
        } else {
            // Skip unexpected characters
            remainder = &remainder[1..];
        }
    }

    Ok(map)
}

/// Parse a JSON string at the start of `s`, returning `(value, bytes_consumed)`.
fn parse_json_string_at_start(s: &str) -> Option<(String, usize)> {
    if !s.starts_with('"') {
        return None;
    }
    let mut result = String::new();
    let mut chars = s.char_indices().skip(1);
    loop {
        match chars.next() {
            None => return None,
            Some((i, '"')) => return Some((result, i + '"'.len_utf8())),
            Some((_, '\\')) => match chars.next() {
                Some((_, '"')) => result.push('"'),
                Some((_, '\\')) => result.push('\\'),
                Some((_, 'n')) => result.push('\n'),
                Some((_, 'r')) => result.push('\r'),
                Some((_, 't')) => result.push('\t'),
                Some((_, 'u')) => {
                    let mut hex = String::new();
                    for _ in 0..4 {
                        if let Some((_, c)) = chars.next() {
                            hex.push(c);
                        }
                    }
                    if let Ok(n) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(n) {
                            result.push(c);
                        }
                    }
                }
                Some((_, c)) => result.push(c),
                None => return None,
            },
            Some((_, c)) => result.push(c),
        }
    }
}

/// Find the byte offset of the closing bracket/brace matching the opening
/// bracket/brace at position 0 of `s`.
fn find_matching_brace(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut prev_escape = false;

    for (i, ch) in s.char_indices() {
        if prev_escape {
            prev_escape = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                prev_escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Split a JSON array's inner text `{...},{...},...` into object strings.
fn split_json_array_objects(inner: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut remainder = inner.trim();
    while !remainder.is_empty() {
        if remainder.starts_with('{') {
            match find_matching_brace(remainder, '{', '}') {
                Some(end) => {
                    result.push(remainder[..=end].to_string());
                    remainder = remainder[end + 1..].trim_start_matches(',').trim();
                }
                None => break,
            }
        } else {
            // Skip unexpected content
            let skip = remainder.find('{').unwrap_or(remainder.len());
            if skip == remainder.len() {
                break;
            }
            remainder = &remainder[skip..];
        }
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenization::wordpiece::WordPieceTokenizer;

    fn minimal_wp() -> WordPieceTokenizer {
        let tokens = vec![
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##ing", "foo",
        ];
        WordPieceTokenizer::from_vocab_list(&tokens)
    }

    #[test]
    fn from_wordpiece_model_type() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        assert_eq!(hf.model.model_type, "WordPiece");
    }

    #[test]
    fn to_json_string_contains_vocab() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        let s = hf.to_json_string();
        assert!(s.contains("\"vocab\""), "JSON must contain vocab key");
    }

    #[test]
    fn roundtrip_from_json_str() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        let json = hf.to_json_string();
        let restored = HfTokenizerJson::from_json_str(&json).expect("parse failed");
        assert_eq!(restored.model.model_type, "WordPiece");
    }

    #[test]
    fn detect_model_type_wordpiece() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        let json = hf.to_json_string();
        let mt = detect_model_type(&json).expect("detect failed");
        assert_eq!(mt, HfModelType::WordPiece);
    }

    #[test]
    fn detect_model_type_bpe() {
        // Build a minimal BPE-type JSON by hand
        let json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"hello":0},"merges":["h e"]},"added_tokens":[]}"#;
        let mt = detect_model_type(json).expect("detect failed");
        assert_eq!(mt, HfModelType::Bpe);
    }

    #[test]
    fn added_tokens_contains_cls() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        let has_cls = hf.added_tokens.iter().any(|t| t.content == "[CLS]");
        assert!(has_cls, "added_tokens must contain [CLS]");
    }

    #[test]
    fn vocab_size_matches_input() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        assert_eq!(hf.model.vocab.len(), wp.vocab_size());
    }

    #[test]
    fn empty_vocab_serialises_without_panic() {
        let tokens: &[&str] = &[];
        let wp = WordPieceTokenizer::from_vocab_list(tokens);
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        let json = hf.to_json_string();
        assert!(json.contains("WordPiece"));
    }

    #[test]
    fn hf_model_type_variants_accessible() {
        let _ = HfModelType::WordPiece;
        let _ = HfModelType::Bpe;
        let _ = HfModelType::Unigram;
        let _ = HfModelType::Unknown("X".to_string());
    }

    #[test]
    fn invalid_json_returns_err() {
        let result = HfTokenizerJson::from_json_str("not json at all }{");
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_check_helper() {
        let wp = minimal_wp();
        assert!(HfTokenizerJson::wordpiece_roundtrip_check(&wp));
    }

    #[test]
    fn version_field_preserved() {
        let wp = minimal_wp();
        let hf = HfTokenizerJson::from_wordpiece(&wp);
        let json = hf.to_json_string();
        let restored = HfTokenizerJson::from_json_str(&json).unwrap();
        assert_eq!(restored.version, "1.0");
    }
}
