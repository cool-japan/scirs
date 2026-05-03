//! Byte-Level BPE tokenizer — GPT-2/GPT-4 style.
//!
//! Encodes each byte as a unicode character using the GPT-2 byte→unicode table,
//! then applies BPE merges on top of that representation.
//!
//! The mapping is bijective: 256 bytes → 256 distinct unicode code points.

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

/// Type alias for the four-tuple returned by `init_base`.
type InitBaseMaps = (
    HashMap<u8, char>,
    HashMap<char, u8>,
    HashMap<String, u32>,
    Vec<String>,
);

// ─── GPT-2 byte-to-unicode table ─────────────────────────────────────────────

/// Build the GPT-2 byte→unicode bijection.
///
/// Bytes that are already printable ASCII (33-126) or in the
/// Latin-1 supplement printable range (161-172, 174-255) map to themselves.
/// The remaining 68 bytes (0-32, 127-160, 173) map to the
/// consecutive block starting at U+0100 (LATIN CAPITAL LETTER A WITH MACRON).
pub fn bytes_to_unicode() -> HashMap<u8, char> {
    // Collect the "nice" printable bytes first
    let mut bs: Vec<u8> = (b'!'..=b'~').collect(); // 33-126
    bs.extend(b'\xa1'..=b'\xac'); // 161-172
    bs.extend(b'\xae'..=b'\xff'); // 174-255

    // The remaining bytes need remapping — assign them unicode codepoints
    // starting at U+0100 in order of their byte value.
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0u32; // offset counter into the extension block
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            // U+0100 + n
            let cp = 0x0100u32 + n;
            cs.push(char::from_u32(cp).unwrap_or('\u{0100}'));
            n += 1;
        }
    }

    bs.into_iter().zip(cs).collect()
}

// ─── ByteLevelBpeConfig ───────────────────────────────────────────────────────

/// Configuration for [`ByteLevelBpeTokenizer`] training.
#[derive(Debug, Clone)]
pub struct ByteLevelBpeConfig {
    /// Target vocabulary size (includes the 256 byte-level base tokens).
    pub vocab_size: usize,
    /// Minimum pair frequency required for a merge to be created.
    pub min_frequency: usize,
    /// Whether to add a space prefix (Ġ) before each word except the first.
    pub add_prefix_space: bool,
}

impl Default for ByteLevelBpeConfig {
    fn default() -> Self {
        ByteLevelBpeConfig {
            vocab_size: 50257,
            min_frequency: 2,
            add_prefix_space: true,
        }
    }
}

// ─── ByteLevelBpeTokenizer ────────────────────────────────────────────────────

/// GPT-2-style byte-level BPE tokenizer.
///
/// Every input byte is first mapped to a unique unicode character via the
/// GPT-2 byte→unicode table, so the BPE algorithm operates on unicode
/// character sequences.  This makes the tokenizer vocabulary guaranteed to be
/// lossless and eliminates any `[UNK]` token for arbitrary UTF-8 input.
#[derive(Debug, Clone)]
pub struct ByteLevelBpeTokenizer {
    /// token string → integer id
    pub vocab: HashMap<String, u32>,
    /// integer id → token string
    pub id_to_token: Vec<String>,
    /// ordered merge rules (left_piece, right_piece)
    pub merges: Vec<(String, String)>,
    /// byte → unicode char
    pub byte_encoder: HashMap<u8, char>,
    /// unicode char → byte  (inverse of byte_encoder)
    pub byte_decoder: HashMap<char, u8>,
}

// Internal helpers
impl ByteLevelBpeTokenizer {
    /// Build encoder/decoder maps and seed base vocabulary from 256 bytes.
    fn init_base() -> InitBaseMaps {
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::new();
        // Add all 256 byte-level characters in byte-value order
        for b in 0u8..=255u8 {
            let ch = byte_encoder[&b];
            let tok = ch.to_string();
            if !vocab.contains_key(&tok) {
                let id = id_to_token.len() as u32;
                vocab.insert(tok.clone(), id);
                id_to_token.push(tok);
            }
        }
        (byte_encoder, byte_decoder, vocab, id_to_token)
    }

    /// Encode a single `word` string (already byte-encoded) into a list of
    /// individual character tokens, then apply all known merge rules.
    fn apply_merges(&self, chars: Vec<String>) -> Vec<String> {
        let mut word = chars;
        // Build a fast merge-priority map
        let merge_rank: HashMap<(&str, &str), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| (a.as_str(), b.as_str()))
            // We can't borrow from a temporary this way — collect differently
            .enumerate()
            .map(|(i, _)| (("", ""), i)) // placeholder, rebuilt below
            .collect();
        // Rebuild properly using indices
        let merge_rank: HashMap<(String, String), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        loop {
            if word.len() < 2 {
                break;
            }
            // Find the highest-priority (lowest rank) adjacent pair
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_idx == usize::MAX {
                break; // no more merges possible
            }
            // Merge at best_idx
            let merged = format!("{}{}", word[best_idx], word[best_idx + 1]);
            word.remove(best_idx + 1);
            word[best_idx] = merged;
        }
        word
    }

    /// Byte-encode a string: each UTF-8 byte is mapped to its unicode char.
    fn byte_encode_str(&self, s: &str) -> Vec<String> {
        s.bytes()
            .map(|b| {
                self.byte_encoder
                    .get(&b)
                    .copied()
                    .unwrap_or('\u{FFFD}')
                    .to_string()
            })
            .collect()
    }
}

// ─── Training ────────────────────────────────────────────────────────────────

impl ByteLevelBpeTokenizer {
    /// Train a new [`ByteLevelBpeTokenizer`] from raw text slices.
    ///
    /// Pre-tokenises on whitespace boundaries and prepends `Ġ` (U+0120) to
    /// every word that is **not** at the beginning of the pre-tokenised
    /// sequence.
    pub fn train(texts: &[&str], config: ByteLevelBpeConfig) -> Self {
        let (byte_encoder, byte_decoder, mut vocab, mut id_to_token) = Self::init_base();

        // Count word frequencies after byte-encoding.
        // IMPORTANT: the Ġ prefix is the *byte-level* representation of byte 0x20
        // (space).  We must NOT format `"\u{0120}word"` and then call `.bytes()`
        // because that would encode the UTF-8 bytes of Ġ (0xC4 0xA0) rather than
        // the single byte-level token Ġ.  Instead, prepend the encoded form of
        // byte 0x20 directly.
        let space_char = byte_encoder
            .get(&0x20u8)
            .copied()
            .unwrap_or('\u{0120}')
            .to_string();

        let mut word_freq: HashMap<Vec<String>, usize> = HashMap::new();
        for text in texts {
            // simple whitespace pre-tokenisation
            let mut first = true;
            for word in text.split_whitespace() {
                // byte-encode the word's raw UTF-8 bytes
                let mut encoded: Vec<String> = word
                    .bytes()
                    .map(|b| {
                        byte_encoder
                            .get(&b)
                            .copied()
                            .unwrap_or('\u{FFFD}')
                            .to_string()
                    })
                    .collect();
                // Prepend the byte-level space token to non-first words
                if !first && config.add_prefix_space {
                    encoded.insert(0, space_char.clone());
                }
                first = false;
                *word_freq.entry(encoded).or_insert(0) += 1;
            }
        }

        let mut merges: Vec<(String, String)> = Vec::new();

        // BPE merge loop
        while vocab.len() < config.vocab_size {
            // Count pair frequencies weighted by word frequency
            let mut pair_freq: HashMap<(String, String), usize> = HashMap::new();
            for (word, &freq) in &word_freq {
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *pair_freq.entry(pair).or_insert(0) += freq;
                }
            }

            // Find best pair
            let best = pair_freq
                .iter()
                .filter(|(_, &f)| f >= config.min_frequency)
                .max_by_key(|(_, &f)| f);

            let (left, right) = match best {
                Some(((l, r), _)) => (l.clone(), r.clone()),
                None => break,
            };

            // Record merge
            merges.push((left.clone(), right.clone()));
            let merged = format!("{}{}", left, right);
            let new_id = id_to_token.len() as u32;
            vocab.insert(merged.clone(), new_id);
            id_to_token.push(merged.clone());

            // Apply merge to all words
            let updated: HashMap<Vec<String>, usize> = word_freq
                .into_iter()
                .map(|(word, freq)| {
                    let new_word = merge_pair_in_word(word, &left, &right);
                    (new_word, freq)
                })
                .collect();
            word_freq = updated;
        }

        ByteLevelBpeTokenizer {
            vocab,
            id_to_token,
            merges,
            byte_encoder,
            byte_decoder,
        }
    }
}

/// Merge all occurrences of (left, right) adjacent pair in `word`.
fn merge_pair_in_word(word: Vec<String>, left: &str, right: &str) -> Vec<String> {
    let mut result = Vec::with_capacity(word.len());
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

// ─── Encoding / Decoding ─────────────────────────────────────────────────────

impl ByteLevelBpeTokenizer {
    /// Encode `text` to a sequence of token IDs.
    ///
    /// Applies the same whitespace pre-tokenisation + byte-encoding + BPE
    /// merges as during training.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        // The byte-level token for space (byte 0x20) is used as a word-prefix
        // marker — it is the *character* that byte 0x20 maps to in the
        // byte_encoder (i.e. Ġ = U+0120).  We prepend it directly (without
        // going through byte_encode_str again) so that the Ġ character itself
        // ends up as a single token rather than its two UTF-8 bytes.
        let space_tok = self
            .byte_encoder
            .get(&0x20u8)
            .copied()
            .unwrap_or('\u{0120}')
            .to_string();

        let mut first = true;
        for word in text.split_whitespace() {
            // Byte-encode only the word's own UTF-8 bytes
            let mut chars = self.byte_encode_str(word);
            // Prepend the byte-level space token for non-first words
            if !first {
                chars.insert(0, space_tok.clone());
            }
            first = false;
            let merged = self.apply_merges(chars);
            for tok in merged {
                if let Some(&id) = self.vocab.get(&tok) {
                    ids.push(id);
                }
                // With byte-level encoding, every byte maps to a valid base token.
                // Unknown tokens should not occur, but we silently skip them if they do.
            }
        }
        ids
    }

    /// Decode a sequence of token IDs back to a UTF-8 string.
    ///
    /// This is lossless: `decode(encode(text)) == text` for any valid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> String {
        // Map ids → token strings → bytes
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id as usize) {
                for ch in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ─── Serialisation ───────────────────────────────────────────────────────────

impl ByteLevelBpeTokenizer {
    /// Save vocabulary (HuggingFace JSON format) and merge rules to separate files.
    ///
    /// The vocab file is a JSON object mapping token strings to integer IDs.
    /// The merges file contains one merge rule per line: `left right`.
    pub fn save_vocab(&self, vocab_path: &str, merges_path: &str) -> Result<()> {
        // Write vocab JSON
        {
            let mut f =
                std::fs::File::create(vocab_path).map_err(|e| TextError::IoError(e.to_string()))?;
            // Manually write JSON to avoid external dependency
            write!(f, "{{").map_err(|e| TextError::IoError(e.to_string()))?;
            let mut pairs: Vec<(&String, &u32)> = self.vocab.iter().collect();
            pairs.sort_by_key(|(_, &id)| id);
            for (i, (tok, id)) in pairs.iter().enumerate() {
                let escaped = escape_json_string(tok);
                if i + 1 < pairs.len() {
                    write!(f, "\"{}\": {}, ", escaped, id)
                        .map_err(|e| TextError::IoError(e.to_string()))?;
                } else {
                    write!(f, "\"{}\": {}", escaped, id)
                        .map_err(|e| TextError::IoError(e.to_string()))?;
                }
            }
            writeln!(f, "}}").map_err(|e| TextError::IoError(e.to_string()))?;
        }

        // Write merges
        {
            let mut f = std::fs::File::create(merges_path)
                .map_err(|e| TextError::IoError(e.to_string()))?;
            writeln!(f, "#version: 0.2").map_err(|e| TextError::IoError(e.to_string()))?;
            for (left, right) in &self.merges {
                writeln!(f, "{} {}", left, right).map_err(|e| TextError::IoError(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Load a tokenizer from a HuggingFace-format vocab JSON and merges text file.
    pub fn load(vocab_path: &str, merges_path: &str) -> Result<Self> {
        // Parse vocab JSON (minimal, no external dep)
        let vocab_content =
            std::fs::read_to_string(vocab_path).map_err(|e| TextError::IoError(e.to_string()))?;
        let vocab = parse_vocab_json(&vocab_content)?;

        // Build id_to_token
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (tok, &id) in &vocab {
            if let Some(slot) = id_to_token.get_mut(id as usize) {
                *slot = tok.clone();
            }
        }

        // Parse merges
        let merges_file =
            std::fs::File::open(merges_path).map_err(|e| TextError::IoError(e.to_string()))?;
        let reader = BufReader::new(merges_file);
        let mut merges = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| TextError::IoError(e.to_string()))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }

        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();

        Ok(ByteLevelBpeTokenizer {
            vocab,
            id_to_token,
            merges,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up the token string for an ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Look up the ID for a token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Minimal JSON string escaping.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Minimal JSON object parser that only handles `{"key": number, ...}`.
///
/// Uses a string-aware comma splitter so that tokens containing `"` or `,`
/// are handled correctly.
fn parse_vocab_json(s: &str) -> Result<HashMap<String, u32>> {
    let s = s.trim();
    let inner = s
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| TextError::IoError("Invalid vocab JSON: missing braces".to_string()))?;

    let mut vocab = HashMap::new();
    // Split on `,` only when outside a JSON string (tracks in-string state).
    let chars: Vec<char> = inner.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut start = 0;

    while i <= n {
        let at_end = i == n;

        if at_end {
            // Flush the final entry
            let entry: String = chars[start..i].iter().collect();
            let entry = entry.trim();
            if !entry.is_empty() {
                parse_vocab_entry(entry, &mut vocab)?;
            }
            break;
        }

        let ch = chars[i];

        if ch == '"' {
            // Skip over the whole quoted string including any escaped characters
            i += 1;
            while i < n {
                let sc = chars[i];
                i += 1;
                if sc == '\\' {
                    // skip the escaped character
                    i += 1;
                } else if sc == '"' {
                    break;
                }
            }
            // After closing quote, continue the outer loop without incrementing i
            continue;
        }

        if ch == ',' {
            let entry: String = chars[start..i].iter().collect();
            let entry = entry.trim();
            if !entry.is_empty() {
                parse_vocab_entry(entry, &mut vocab)?;
            }
            start = i + 1;
        }

        i += 1;
    }

    Ok(vocab)
}

fn parse_vocab_entry(entry: &str, vocab: &mut HashMap<String, u32>) -> Result<()> {
    // Format: `"token": id`
    let colon_pos = find_colon_outside_string(entry)
        .ok_or_else(|| TextError::IoError(format!("Invalid vocab entry (no colon): {}", entry)))?;
    let key_part = entry[..colon_pos].trim();
    let val_part = entry[colon_pos + 1..].trim();

    let key = key_part
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .map(unescape_json_string)
        .ok_or_else(|| TextError::IoError(format!("Invalid vocab key: {}", key_part)))?;

    let id: u32 = val_part
        .parse()
        .map_err(|_| TextError::IoError(format!("Invalid vocab id: {}", val_part)))?;

    vocab.insert(key, id);
    Ok(())
}

fn find_colon_outside_string(s: &str) -> Option<usize> {
    let mut in_str = false;
    let mut escaped = false;
    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_str {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_str = !in_str;
            continue;
        }
        if ch == ':' && !in_str {
            return Some(i);
        }
    }
    None
}

fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(n) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(n) {
                            out.push(c);
                        }
                    }
                }
                Some(c) => out.push(c),
                None => {}
            }
        } else {
            out.push(ch);
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode_count() {
        let map = bytes_to_unicode();
        assert_eq!(map.len(), 256, "should have exactly 256 entries");
    }

    #[test]
    fn test_bytes_to_unicode_bijective() {
        let map = bytes_to_unicode();
        let mut chars: Vec<char> = map.values().copied().collect();
        chars.sort();
        chars.dedup();
        assert_eq!(
            chars.len(),
            256,
            "all unicode chars must be distinct (bijection)"
        );
    }

    #[test]
    fn test_bytes_to_unicode_ascii_identity() {
        let map = bytes_to_unicode();
        // printable ASCII chars should map to themselves
        for b in b'!'..=b'~' {
            let ch = map[&b];
            assert_eq!(
                ch as u32, b as u32,
                "byte {} should map to itself, got {}",
                b, ch as u32
            );
        }
    }

    #[test]
    fn test_train_vocab_size() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "hello world hello rust hello tokenizer",
            "byte level bpe tokenizer training test data for vocabulary",
            "more text data to train the byte level bpe model properly",
        ];
        let config = ByteLevelBpeConfig {
            vocab_size: 300,
            min_frequency: 1,
            add_prefix_space: true,
        };
        let tok = ByteLevelBpeTokenizer::train(&texts, config);
        assert!(
            tok.vocab_size() <= 300,
            "vocab size should not exceed requested"
        );
        assert!(
            tok.vocab_size() >= 256,
            "should have at least base 256 tokens"
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let texts = [
            "hello world",
            "the quick brown fox",
            "rust programming language",
            "byte level encoding test",
        ];
        let config = ByteLevelBpeConfig {
            vocab_size: 500,
            min_frequency: 1,
            add_prefix_space: true,
        };
        let tok = ByteLevelBpeTokenizer::train(&texts, config);
        let input = "hello world";
        let ids = tok.encode(input);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, input, "encode/decode roundtrip should be lossless");
    }

    #[test]
    fn test_gword_prefix() {
        // Non-first words should be prefixed with Ġ (U+0120)
        let texts = ["hello world test"];
        let config = ByteLevelBpeConfig {
            vocab_size: 300,
            min_frequency: 1,
            add_prefix_space: true,
        };
        let tok = ByteLevelBpeTokenizer::train(&texts, config);
        // The tokenizer vocab should contain a Ġ-prefixed token
        let has_g_prefix = tok.vocab.keys().any(|k| k.starts_with('\u{0120}'));
        assert!(has_g_prefix, "vocabulary should contain Ġ-prefixed tokens");
    }

    #[test]
    fn test_hello_token() {
        let texts = ["hello world hello hello hello"];
        let config = ByteLevelBpeConfig {
            vocab_size: 300,
            min_frequency: 1,
            add_prefix_space: false,
        };
        let tok = ByteLevelBpeTokenizer::train(&texts, config);
        // After training, "hello" should appear as a merged token
        // (since it's frequent enough)
        assert!(
            tok.vocab.contains_key("hello"),
            "hello should be in vocabulary after training on repeated hello"
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let texts = [
            "hello world",
            "test tokenizer save load",
            "byte level bpe tokenizer",
        ];
        let config = ByteLevelBpeConfig {
            vocab_size: 350,
            min_frequency: 1,
            add_prefix_space: true,
        };
        let tok = ByteLevelBpeTokenizer::train(&texts, config);

        let dir = std::env::temp_dir();
        let vocab_path = dir
            .join("test_bpe_vocab.json")
            .to_string_lossy()
            .into_owned();
        let merges_path = dir
            .join("test_bpe_merges.txt")
            .to_string_lossy()
            .into_owned();

        tok.save_vocab(&vocab_path, &merges_path)
            .expect("save failed");
        let loaded = ByteLevelBpeTokenizer::load(&vocab_path, &merges_path).expect("load failed");

        assert_eq!(tok.vocab_size(), loaded.vocab_size());
        assert_eq!(tok.merges.len(), loaded.merges.len());
    }
}
