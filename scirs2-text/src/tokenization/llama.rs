//! LLaMA-style SentencePiece BPE tokenizer with byte fallback.
//!
//! LLaMA uses a SentencePiece model trained with the BPE algorithm, augmented
//! with:
//!
//! - A `▁` (U+2581 LOWER ONE EIGHTH BLOCK) whitespace marker prepended to
//!   each word.
//! - Byte-level fallback tokens `<0x00>`..`<0xFF>` for any codepoint not
//!   directly present in the vocabulary.
//! - BOS (`<s>`, ID 1) and EOS (`</s>`, ID 2) control tokens.
//!
//! This implementation is *inference-only*: it accepts a pre-built vocabulary
//! and merge list and performs greedy BPE tokenisation.  Training is outside
//! the scope of this module.
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_text::tokenization::llama::{LlamaTokenizer, LlamaTokenizerConfig};
//!
//! let config = LlamaTokenizerConfig::default();
//! let tokenizer = LlamaTokenizer::new_minimal(512, config);
//!
//! let ids = tokenizer.encode("hello world");
//! assert!(!ids.is_empty());
//!
//! let text = tokenizer.decode(&ids);
//! assert!(text.contains("hello"));
//! ```

use std::collections::HashMap;

use crate::error::{Result, TextError};

// ── Constants ─────────────────────────────────────────────────────────────────

/// The whitespace-prefix character used by SentencePiece / LLaMA.
const SPIECE_UNDERLINE: char = '▁'; // U+2581

/// Number of byte tokens in the vocabulary.
const NUM_BYTE_TOKENS: usize = 256;

/// Starting ID for byte tokens in the minimal vocabulary.
const BYTE_TOKEN_OFFSET: u32 = 3;

/// Starting ID for single-character alphabetic tokens.
const ALPHA_TOKEN_OFFSET: u32 = BYTE_TOKEN_OFFSET + NUM_BYTE_TOKENS as u32;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`LlamaTokenizer`].
#[derive(Debug, Clone)]
pub struct LlamaTokenizerConfig {
    /// ID of the unknown token `<unk>`.
    pub unk_id: u32,
    /// ID of the beginning-of-sequence token `<s>`.
    pub bos_id: u32,
    /// ID of the end-of-sequence token `</s>`.
    pub eos_id: u32,
    /// ID of the padding token.  `-1` means no dedicated pad token.
    pub pad_id: i64,
    /// Prepend BOS token to every encoded sequence.
    pub add_bos: bool,
    /// Append EOS token to every encoded sequence.
    pub add_eos: bool,
    /// Normalise whitespace: prepend `▁` and replace spaces with `▁`.
    pub normalize_whitespace: bool,
}

impl Default for LlamaTokenizerConfig {
    fn default() -> Self {
        LlamaTokenizerConfig {
            unk_id: 0,
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            add_bos: true,
            add_eos: false,
            normalize_whitespace: true,
        }
    }
}

// ── LlamaTokenizer ────────────────────────────────────────────────────────────

/// A LLaMA-style SentencePiece BPE tokenizer with byte-level fallback.
///
/// This struct is constructed either from an external vocabulary / merge list
/// or via [`LlamaTokenizer::new_minimal`] which creates a self-contained
/// demo vocabulary.
#[derive(Debug, Clone)]
pub struct LlamaTokenizer {
    /// Piece string → integer ID.
    pub vocab: HashMap<String, u32>,
    /// Integer ID → piece string.
    id_to_piece: HashMap<u32, String>,
    /// BPE merge rules as (left_id, right_id) pairs.
    ///
    /// Earlier entries have *higher* priority (lower rank).
    pub merges: Vec<(u32, u32)>,
    /// Byte-fallback token IDs: `byte_tokens[b]` is the vocabulary ID for
    /// the byte `b`.
    pub byte_tokens: [u32; NUM_BYTE_TOKENS],
    /// Tokenizer configuration.
    pub config: LlamaTokenizerConfig,
}

impl LlamaTokenizer {
    // ── Construction ──────────────────────────────────────────────────────

    /// Build a minimal tokenizer with a demo vocabulary.
    ///
    /// The vocabulary contains:
    /// - ID 0: `<unk>`
    /// - ID 1: `<s>` (BOS)
    /// - ID 2: `</s>` (EOS)
    /// - IDs 3..=258: byte tokens `<0x00>`..`<0xFF>`
    /// - IDs 259..=310: lowercase and uppercase ASCII letters `a`..`z`, `A`..`Z`
    /// - ID 311: the whitespace marker `▁`
    /// - Further IDs: common digrams to demonstrate BPE merging
    ///
    /// `vocab_size` controls how many additional tokens are allocated; values
    /// below 312 are silently clamped to 312 (the minimum useful size).
    pub fn new_minimal(vocab_size: usize, config: LlamaTokenizerConfig) -> Self {
        let min_size = 312usize; // 3 special + 256 bytes + 52 alpha + 1 marker
        let effective_size = vocab_size.max(min_size);

        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(effective_size);
        let mut id_to_piece: HashMap<u32, String> = HashMap::with_capacity(effective_size);

        // ── special tokens ────────────────────────────────────────────────
        let specials = [
            (config.unk_id, "<unk>"),
            (config.bos_id, "<s>"),
            (config.eos_id, "</s>"),
        ];
        for (id, tok) in &specials {
            vocab.insert(tok.to_string(), *id);
            id_to_piece.insert(*id, tok.to_string());
        }

        // ── byte tokens ───────────────────────────────────────────────────
        let mut byte_tokens = [0u32; NUM_BYTE_TOKENS];
        for b in 0u32..256 {
            let piece = format!("<0x{:02X}>", b);
            let id = BYTE_TOKEN_OFFSET + b;
            vocab.insert(piece.clone(), id);
            id_to_piece.insert(id, piece);
            byte_tokens[b as usize] = id;
        }

        // ── alphabetic unigrams ───────────────────────────────────────────
        let mut alpha_id = ALPHA_TOKEN_OFFSET;
        for ch in ('a'..='z').chain('A'..='Z') {
            let piece = ch.to_string();
            vocab.insert(piece.clone(), alpha_id);
            id_to_piece.insert(alpha_id, piece);
            alpha_id += 1;
        }

        // ── whitespace marker ─────────────────────────────────────────────
        let spiece_id = alpha_id; // 311
        let spiece_str = SPIECE_UNDERLINE.to_string();
        vocab.insert(spiece_str.clone(), spiece_id);
        id_to_piece.insert(spiece_id, spiece_str.clone());
        let mut next_id = spiece_id + 1;

        // ── common digrams as BPE merges ──────────────────────────────────
        // Pre-define common English digrams so that BPE merging actually
        // produces multi-character tokens in tests.
        let common_digrams: &[(&str, &str)] = &[
            ("h", "e"),   // "he"
            ("l", "l"),   // "ll"
            ("he", "ll"), // "hell"
            ("o", "w"),   // "ow"
            ("w", "o"),   // "wo"
            ("▁", "h"),   // "▁h"
            ("▁", "w"),   // "▁w"
            ("r", "l"),   // "rl"
            ("l", "d"),   // "ld"
            ("o", "r"),   // "or"
        ];

        let mut merges: Vec<(u32, u32)> = Vec::new();

        for (left_str, right_str) in common_digrams {
            if next_id as usize >= effective_size {
                break;
            }
            let left_id = vocab.get(*left_str).copied();
            let right_id = vocab.get(*right_str).copied();
            let merged = format!("{}{}", left_str, right_str);

            if let (Some(lid), Some(rid)) = (left_id, right_id) {
                if !vocab.contains_key(&merged) {
                    vocab.insert(merged.clone(), next_id);
                    id_to_piece.insert(next_id, merged);
                    merges.push((lid, rid));
                    next_id += 1;
                } else {
                    // Still record the merge even if the token already exists
                    merges.push((lid, rid));
                }
            }
        }

        LlamaTokenizer {
            vocab,
            id_to_piece,
            merges,
            byte_tokens,
            config,
        }
    }

    /// Build a tokenizer from a pre-existing vocabulary and merge rules.
    ///
    /// - `vocab`: piece string → ID mapping (must include `<unk>`, `<s>`, `</s>`,
    ///   and all `<0xXX>` byte tokens).
    /// - `merges`: ordered list of `(left_piece, right_piece)` merge pairs.
    /// - `config`: tokenizer configuration.
    ///
    /// Returns an error when a byte token is missing from the vocabulary.
    pub fn from_vocab_and_merges(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        config: LlamaTokenizerConfig,
    ) -> Result<Self> {
        // Build id → piece reverse map
        let id_to_piece: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        // Resolve merge pairs to IDs
        let merges_ids: Vec<(u32, u32)> = merges
            .iter()
            .filter_map(|(l, r)| {
                let lid = vocab.get(l.as_str()).copied()?;
                let rid = vocab.get(r.as_str()).copied()?;
                Some((lid, rid))
            })
            .collect();

        // Build byte-token table
        let mut byte_tokens = [0u32; NUM_BYTE_TOKENS];
        for b in 0usize..NUM_BYTE_TOKENS {
            let piece = format!("<0x{:02X}>", b);
            byte_tokens[b] = vocab.get(&piece).copied().ok_or_else(|| {
                TextError::VocabularyError(format!(
                    "LlamaTokenizer: byte token '{}' missing from vocab",
                    piece
                ))
            })?;
        }

        Ok(LlamaTokenizer {
            vocab,
            id_to_piece,
            merges: merges_ids,
            byte_tokens,
            config,
        })
    }

    // ── Encoding ──────────────────────────────────────────────────────────

    /// Encode `text` into a sequence of token IDs.
    ///
    /// Steps:
    /// 1. Optionally prepend `▁` and replace spaces with `▁`.
    /// 2. Split into individual Unicode codepoints.
    /// 3. Look up each codepoint in the vocabulary; unknown codepoints are
    ///    encoded as their UTF-8 byte tokens.
    /// 4. Apply BPE merges greedily (lowest-rank pair first).
    /// 5. Optionally prepend BOS and/or append EOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return if self.config.add_bos {
                vec![self.config.bos_id]
            } else {
                vec![]
            };
        }

        // ── normalise whitespace ──────────────────────────────────────────
        let normalised: String = if self.config.normalize_whitespace {
            let replaced = text.replace(' ', &SPIECE_UNDERLINE.to_string());
            format!("{}{}", SPIECE_UNDERLINE, replaced)
        } else {
            text.to_string()
        };

        // ── codepoint → initial token IDs ─────────────────────────────────
        let mut token_ids: Vec<u32> = self.text_to_initial_ids(&normalised);

        // ── BPE merges ────────────────────────────────────────────────────
        if !self.merges.is_empty() {
            token_ids = self.apply_bpe_merges(token_ids);
        }

        // ── special tokens ────────────────────────────────────────────────
        let mut result = Vec::with_capacity(token_ids.len() + 2);
        if self.config.add_bos {
            result.push(self.config.bos_id);
        }
        result.extend(token_ids);
        if self.config.add_eos {
            result.push(self.config.eos_id);
        }

        result
    }

    /// Decode a sequence of token IDs back to a string.
    ///
    /// BOS and EOS tokens are skipped.  Byte tokens `<0xXX>` are converted to
    /// their actual byte values and then decoded as UTF-8 (with lossy
    /// substitution for malformed sequences).  The `▁` whitespace marker is
    /// replaced with a plain space, and a leading space is stripped.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut byte_buf: Vec<u8> = Vec::new();

        for &id in ids {
            // Skip BOS / EOS
            if id == self.config.bos_id || id == self.config.eos_id {
                continue;
            }

            if let Some(piece) = self.id_to_piece.get(&id) {
                // Check if this is a byte token <0xXX>
                if let Some(b) = parse_byte_token(piece) {
                    byte_buf.push(b);
                } else {
                    // Flush pending bytes first
                    // Regular piece: encode the ▁ as space for proper UTF-8 output
                    let text_piece = piece.replace(SPIECE_UNDERLINE, " ");
                    byte_buf.extend_from_slice(text_piece.as_bytes());
                }
            }
        }

        let decoded = String::from_utf8_lossy(&byte_buf).into_owned();

        // Strip a leading space that was introduced by the ▁ prefix
        if decoded.starts_with(' ') {
            decoded[1..].to_string()
        } else {
            decoded
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Total number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up the ID for a piece string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Look up the piece string for a given ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_piece.get(&id).map(|s| s.as_str())
    }

    // ── HF JSON export ────────────────────────────────────────────────────

    /// Export this tokenizer in HuggingFace `tokenizers.json` format.
    ///
    /// Uses `model.type = "BPE"` since LLaMA is fundamentally a BPE model.
    pub fn to_hf_json(&self) -> String {
        use super::hf_json::{HfAddedToken, HfModel, HfTokenizerJson};
        use std::collections::HashMap;

        // Reconstruct string merge rules from ID pairs
        let merges_strs: Vec<String> = self
            .merges
            .iter()
            .filter_map(|(lid, rid)| {
                let l = self.id_to_piece.get(lid)?;
                let r = self.id_to_piece.get(rid)?;
                Some(format!("{} {}", l, r))
            })
            .collect();

        let model = HfModel {
            model_type: "BPE".to_string(),
            vocab: self.vocab.clone(),
            merges: Some(merges_strs),
            unk_token: Some("<unk>".to_string()),
            continuing_subword_prefix: None,
            max_input_chars_per_word: None,
        };

        // Add BOS/EOS as special tokens
        let mut added_tokens = Vec::new();
        for (id, name) in [
            (self.config.unk_id, "<unk>"),
            (self.config.bos_id, "<s>"),
            (self.config.eos_id, "</s>"),
        ] {
            added_tokens.push(HfAddedToken::special(id, name));
        }

        let hf = HfTokenizerJson {
            version: "1.0".to_string(),
            model,
            added_tokens,
            normalizer_json: None,
            pre_tokenizer_json: None,
            post_processor_json: None,
            decoder_json: None,
        };

        hf.to_json_string()
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Convert text into a sequence of token IDs by looking up each codepoint.
    ///
    /// Unknown codepoints fall back to byte tokens.
    fn text_to_initial_ids(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        // Iterate over characters; attempt vocab lookup; fallback to bytes
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let piece = chars[i].to_string();
            if let Some(&id) = self.vocab.get(&piece) {
                ids.push(id);
                i += 1;
            } else {
                // Encode as UTF-8 bytes
                let mut buf = [0u8; 4];
                let encoded = chars[i].encode_utf8(&mut buf);
                for &b in encoded.as_bytes() {
                    ids.push(self.byte_tokens[b as usize]);
                }
                i += 1;
            }
        }
        ids
    }

    /// Apply BPE merge rules to a sequence of token IDs.
    ///
    /// Each pass finds the pair (left_id, right_id) with the lowest merge rank
    /// and replaces all occurrences.  This is repeated until no more merges
    /// are applicable.
    fn apply_bpe_merges(&self, mut ids: Vec<u32>) -> Vec<u32> {
        if ids.len() <= 1 {
            return ids;
        }

        // Build a lookup: (left_id, right_id) → (rank, merged_id)
        // The merged token ID is the token obtained by concatenating the pieces.
        let merge_lookup: HashMap<(u32, u32), (usize, u32)> = self
            .merges
            .iter()
            .enumerate()
            .filter_map(|(rank, (lid, rid))| {
                let left_piece = self.id_to_piece.get(lid)?;
                let right_piece = self.id_to_piece.get(rid)?;
                let merged_piece = format!("{}{}", left_piece, right_piece);
                let merged_id = self.vocab.get(&merged_piece).copied()?;
                Some(((*lid, *rid), (rank, merged_id)))
            })
            .collect();

        loop {
            if ids.len() < 2 {
                break;
            }

            // Find the best (lowest-rank) adjacent pair
            let best = ids
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    merge_lookup
                        .get(&(w[0], w[1]))
                        .map(|&(rank, mid)| (i, rank, mid))
                })
                .min_by_key(|&(_, rank, _)| rank);

            match best {
                None => break,
                Some((pos, _, merged_id)) => {
                    // Replace ids[pos] and ids[pos+1] with merged_id
                    let mut new_ids = Vec::with_capacity(ids.len() - 1);
                    new_ids.extend_from_slice(&ids[..pos]);
                    new_ids.push(merged_id);
                    new_ids.extend_from_slice(&ids[pos + 2..]);
                    ids = new_ids;
                }
            }
        }

        ids
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a byte-token piece like `<0x4A>` into its byte value.
/// Returns `None` for any other string.
fn parse_byte_token(piece: &str) -> Option<u8> {
    // Pattern: <0xXX>  (exactly 6 chars)
    if piece.len() != 6 {
        return None;
    }
    let bytes = piece.as_bytes();
    if bytes[0] != b'<' || bytes[1] != b'0' || bytes[2] != b'x' || bytes[5] != b'>' {
        return None;
    }
    let hex = &piece[3..5];
    u8::from_str_radix(hex, 16).ok()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> LlamaTokenizer {
        LlamaTokenizer::new_minimal(512, LlamaTokenizerConfig::default())
    }

    #[test]
    fn new_minimal_special_tokens() {
        let tok = make_tokenizer();
        assert_eq!(tok.token_to_id("<unk>"), Some(0));
        assert_eq!(tok.token_to_id("<s>"), Some(1));
        assert_eq!(tok.token_to_id("</s>"), Some(2));
    }

    #[test]
    fn new_minimal_byte_tokens_at_correct_ids() {
        let tok = make_tokenizer();
        assert_eq!(tok.token_to_id("<0x00>"), Some(BYTE_TOKEN_OFFSET));
        assert_eq!(tok.token_to_id("<0xFF>"), Some(BYTE_TOKEN_OFFSET + 255));
    }

    #[test]
    fn encode_returns_nonempty() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello");
        assert!(!ids.is_empty());
    }

    #[test]
    fn encode_prepends_bos_when_configured() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello");
        assert_eq!(ids[0], 1, "first token should be BOS=1");
    }

    #[test]
    fn encode_empty_string_with_bos() {
        let tok = make_tokenizer();
        let ids = tok.encode("");
        assert_eq!(ids, vec![1], "empty + add_bos → [BOS]");
    }

    #[test]
    fn encode_empty_string_without_bos() {
        let config = LlamaTokenizerConfig {
            add_bos: false,
            ..Default::default()
        };
        let tok = LlamaTokenizer::new_minimal(512, config);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn encode_no_bos_no_bos_prefix() {
        let config = LlamaTokenizerConfig {
            add_bos: false,
            ..Default::default()
        };
        let tok = LlamaTokenizer::new_minimal(512, config);
        let ids = tok.encode("hello");
        // First token should NOT be BOS (1)
        assert_ne!(ids[0], 1);
    }

    #[test]
    fn ascii_roundtrip() {
        let tok = make_tokenizer();
        let text = "hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "roundtrip failed: got '{}'", decoded);
    }

    #[test]
    fn cjk_byte_fallback_no_panic() {
        let tok = make_tokenizer();
        // 'こ' is not in the minimal vocab → must fall back to byte tokens
        let ids = tok.encode("こんにちは");
        assert!(!ids.is_empty(), "encoding CJK should produce tokens");
    }

    #[test]
    fn vocab_size_gt_3() {
        let tok = make_tokenizer();
        assert!(tok.vocab_size() > 3);
    }

    #[test]
    fn unk_token_lookup() {
        let tok = make_tokenizer();
        assert_eq!(tok.token_to_id("<unk>"), Some(0));
    }

    #[test]
    fn id_to_token_bos() {
        let tok = make_tokenizer();
        let s = tok.id_to_token(1);
        assert_eq!(s, Some("<s>"));
    }

    #[test]
    fn all_256_bytes_have_ids() {
        let tok = make_tokenizer();
        for b in 0usize..256 {
            let piece = format!("<0x{:02X}>", b);
            assert!(
                tok.token_to_id(&piece).is_some(),
                "byte token {} not found",
                piece
            );
        }
    }

    #[test]
    fn decode_byte_tokens_reconstructs_bytes() {
        let tok = make_tokenizer();
        // ASCII 'A' = 0x41
        let byte_id = tok.token_to_id("<0x41>").expect("byte token missing");
        let decoded = tok.decode(&[byte_id]);
        assert_eq!(decoded, "A", "expected 'A', got '{}'", decoded);
    }

    #[test]
    fn to_hf_json_contains_bpe() {
        let tok = make_tokenizer();
        let json = tok.to_hf_json();
        assert!(json.contains("BPE"), "HF JSON must contain BPE model type");
    }

    #[test]
    fn config_defaults() {
        let cfg = LlamaTokenizerConfig::default();
        assert_eq!(cfg.unk_id, 0);
        assert_eq!(cfg.bos_id, 1);
        assert_eq!(cfg.eos_id, 2);
        assert_eq!(cfg.pad_id, -1);
        assert!(cfg.add_bos);
        assert!(!cfg.add_eos);
        assert!(cfg.normalize_whitespace);
    }
}
