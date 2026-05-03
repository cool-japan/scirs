//! Language-agnostic Unicode tokenizer following UAX #29 word boundaries.
//!
//! This tokenizer handles mixed-script text (Latin, CJK, Cyrillic, Arabic, etc.)
//! by using Unicode word-boundary segmentation for non-CJK text and per-character
//! splitting for CJK runs. NFC normalization and optional lowercasing are supported.
//!
//! # Example
//!
//! ```
//! use scirs2_text::tokenization::language_agnostic::LanguageAgnosticTokenizer;
//!
//! let t = LanguageAgnosticTokenizer::new();
//! let tokens = t.tokenize_str("Hello 你好 World");
//! assert!(tokens.iter().any(|s| s == "Hello"));
//! assert!(tokens.iter().any(|s| s == "你" || s == "你好"));
//! ```

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

use crate::error::Result;
use crate::tokenize::Tokenizer;

// ── CJK block detection ────────────────────────────────────────────────────────

/// Returns `true` if the character belongs to a CJK-related Unicode block.
///
/// Covered blocks:
/// - CJK Unified Ideographs (U+4E00–U+9FFF)
/// - CJK Extension A (U+3400–U+4DBF)
/// - CJK Extension B (U+20000–U+2A6DF)
/// - CJK Compatibility Ideographs (U+F900–U+FAFF)
/// - CJK Compatibility Supplement (U+2F800–U+2FA1F)
/// - CJK Symbols and Punctuation (U+3000–U+303F)
/// - Hiragana (U+3040–U+309F)
/// - Katakana (U+30A0–U+30FF)
/// - Katakana Phonetic Extensions (U+31F0–U+31FF)
/// - Hangul Syllables (U+AC00–U+D7AF)
fn is_cjk_char(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF    | // CJK Unified Ideographs
        0x3400..=0x4DBF    | // CJK Extension A
        0x20000..=0x2A6DF  | // CJK Extension B
        0xF900..=0xFAFF    | // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F  | // CJK Compatibility Supplement
        0x3000..=0x303F    | // CJK Symbols and Punctuation
        0x3040..=0x309F    | // Hiragana
        0x30A0..=0x30FF    | // Katakana
        0x31F0..=0x31FF    | // Katakana Phonetic Extensions
        0xAC00..=0xD7AF      // Hangul Syllables
    )
}

/// Returns `true` if the word boundary segment contains only whitespace.
fn is_whitespace_segment(s: &str) -> bool {
    s.chars().all(|c| c.is_whitespace())
}

/// Returns `true` if the word boundary segment is purely punctuation (no letters/digits).
fn is_pure_punctuation(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| !c.is_alphanumeric() && !c.is_whitespace())
}

// ── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the language-agnostic tokenizer.
///
/// Default configuration: NFC normalization on, CJK per-character splitting on,
/// punctuation preserved, no lowercasing, no length limit.
///
/// # Example
///
/// ```
/// use scirs2_text::tokenization::language_agnostic::LanguageAgnosticTokenizer;
///
/// let t = LanguageAgnosticTokenizer {
///     normalize: true,
///     lowercase: true,
///     split_cjk_by_char: true,
///     preserve_punctuation: false,
///     max_token_len: Some(32),
/// };
/// let tokens = t.tokenize_str("Hello, WORLD!");
/// assert!(tokens.iter().any(|s| s == "hello"));
/// ```
#[derive(Debug, Clone)]
pub struct LanguageAgnosticTokenizer {
    /// Apply NFC normalization before tokenizing (recommended: `true`).
    pub normalize: bool,
    /// Convert all tokens to lowercase.
    pub lowercase: bool,
    /// Emit each CJK character as an individual token instead of grouping runs.
    pub split_cjk_by_char: bool,
    /// Keep punctuation as separate tokens; when `false`, punctuation-only segments are dropped.
    pub preserve_punctuation: bool,
    /// Truncate tokens longer than this many *characters*. `None` disables the limit.
    pub max_token_len: Option<usize>,
}

impl Default for LanguageAgnosticTokenizer {
    fn default() -> Self {
        Self {
            normalize: true,
            lowercase: false,
            split_cjk_by_char: true,
            preserve_punctuation: true,
            max_token_len: None,
        }
    }
}

impl LanguageAgnosticTokenizer {
    /// Create a tokenizer with the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenize `text` and return the token list directly (infallible).
    ///
    /// This is a convenience method; for the trait-compatible fallible version,
    /// use `<Self as crate::tokenize::Tokenizer>::tokenize`.
    pub fn tokenize_str(&self, text: &str) -> Vec<String> {
        // ── Step 1: NFC normalization ──────────────────────────────────────────
        let normalized: String = if self.normalize {
            text.nfc().collect()
        } else {
            text.to_owned()
        };

        // ── Step 2: case folding ───────────────────────────────────────────────
        let processed: String = if self.lowercase {
            normalized.to_lowercase()
        } else {
            normalized
        };

        // ── Step 3: word-boundary segmentation (UAX #29) ──────────────────────
        // `split_word_bounds` returns every segment including whitespace and
        // punctuation, giving us full control over which segments to keep.
        // Note: per UAX #29, CJK characters are each their own word boundary
        // segment. When `split_cjk_by_char=false` we coalesce consecutive CJK
        // segments back into a single token.
        let mut tokens: Vec<String> = Vec::new();

        // Accumulator for consecutive CJK segments when not splitting by char.
        let mut cjk_run: String = String::new();

        let flush_cjk_run = |run: &mut String, tokens: &mut Vec<String>, max: Option<usize>| {
            if !run.is_empty() {
                let s = std::mem::take(run);
                match max {
                    Some(max_len) if s.chars().count() > max_len => {
                        let truncated: String = s.chars().take(max_len).collect();
                        tokens.push(truncated);
                    }
                    _ => tokens.push(s),
                }
            }
        };

        for segment in processed.split_word_bounds() {
            // Drop whitespace-only segments unconditionally.
            if is_whitespace_segment(segment) {
                // Flush any pending CJK run on whitespace boundary.
                flush_cjk_run(&mut cjk_run, &mut tokens, self.max_token_len);
                continue;
            }

            // Drop pure-punctuation segments when preservation is off.
            if is_pure_punctuation(segment) && !self.preserve_punctuation {
                flush_cjk_run(&mut cjk_run, &mut tokens, self.max_token_len);
                continue;
            }

            // Detect whether this segment contains CJK characters.
            let has_cjk = segment.chars().any(is_cjk_char);

            if has_cjk {
                if self.split_cjk_by_char {
                    // Flush any accumulated non-CJK run first (shouldn't happen
                    // since CJK boundaries are always single-char UAX#29 segments,
                    // but be safe).
                    flush_cjk_run(&mut cjk_run, &mut tokens, self.max_token_len);
                    // Emit each CJK codepoint individually.
                    for ch in segment.chars() {
                        let ch_str = ch.to_string();
                        if !ch_str.trim().is_empty() {
                            self.push_token(&mut tokens, ch_str);
                        }
                    }
                } else {
                    // Accumulate into the CJK run.
                    cjk_run.push_str(segment);
                }
            } else {
                // Non-CJK segment: flush pending CJK run first, then emit.
                flush_cjk_run(&mut cjk_run, &mut tokens, self.max_token_len);
                self.push_token(&mut tokens, segment.to_owned());
            }
        }

        // Flush any trailing CJK run.
        flush_cjk_run(&mut cjk_run, &mut tokens, self.max_token_len);

        tokens
    }

    /// Push a token to the list, applying `max_token_len` truncation if configured.
    fn push_token(&self, tokens: &mut Vec<String>, token: String) {
        match self.max_token_len {
            Some(max_len) if token.chars().count() > max_len => {
                let truncated: String = token.chars().take(max_len).collect();
                tokens.push(truncated);
            }
            _ => tokens.push(token),
        }
    }
}

// ── Tokenizer trait implementation ─────────────────────────────────────────────

impl Tokenizer for LanguageAgnosticTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.tokenize_str(text))
    }

    fn clone_box(&self) -> Box<dyn Tokenizer + Send + Sync> {
        Box::new(self.clone())
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_mixed_latin_cjk() {
        let t = LanguageAgnosticTokenizer::new();
        let tokens = t.tokenize_str("Hello 你好 World");
        assert!(
            tokens.iter().any(|s| s == "Hello"),
            "missing 'Hello': {tokens:?}"
        );
        assert!(
            tokens.iter().any(|s| s == "你" || s == "你好"),
            "missing CJK token: {tokens:?}"
        );
        assert!(
            tokens.iter().any(|s| s == "World"),
            "missing 'World': {tokens:?}"
        );
    }

    #[test]
    fn unit_cjk_split_by_char() {
        let t = LanguageAgnosticTokenizer {
            split_cjk_by_char: true,
            ..Default::default()
        };
        let tokens = t.tokenize_str("日本語");
        assert!(
            tokens.len() >= 3,
            "expected 3 individual CJK chars, got: {tokens:?}"
        );
        assert!(tokens.contains(&"日".to_string()), "tokens: {tokens:?}");
        assert!(tokens.contains(&"本".to_string()), "tokens: {tokens:?}");
        assert!(tokens.contains(&"語".to_string()), "tokens: {tokens:?}");
    }

    #[test]
    fn unit_cjk_no_split() {
        let t = LanguageAgnosticTokenizer {
            split_cjk_by_char: false,
            ..Default::default()
        };
        let tokens = t.tokenize_str("日本語");
        // Without per-char split, the entire run is one segment.
        assert_eq!(
            tokens.len(),
            1,
            "expected single CJK token, got: {tokens:?}"
        );
        assert_eq!(tokens[0], "日本語");
    }

    #[test]
    fn unit_empty_string() {
        let t = LanguageAgnosticTokenizer::new();
        assert_eq!(t.tokenize_str(""), Vec::<String>::new());
    }

    #[test]
    fn unit_whitespace_only() {
        let t = LanguageAgnosticTokenizer::new();
        assert_eq!(t.tokenize_str("   \t\n  "), Vec::<String>::new());
    }

    #[test]
    fn unit_lowercase() {
        let t = LanguageAgnosticTokenizer {
            lowercase: true,
            ..Default::default()
        };
        let tokens = t.tokenize_str("Hello World");
        assert!(
            tokens.iter().any(|s| s == "hello"),
            "expected lowercase: {tokens:?}"
        );
        assert!(
            tokens.iter().any(|s| s == "world"),
            "expected lowercase: {tokens:?}"
        );
    }

    #[test]
    fn unit_preserve_punctuation_true() {
        let t = LanguageAgnosticTokenizer {
            preserve_punctuation: true,
            ..Default::default()
        };
        let tokens = t.tokenize_str("hello, world!");
        let has_comma = tokens.iter().any(|s| s.contains(','));
        let has_excl = tokens.iter().any(|s| s.contains('!'));
        assert!(
            has_comma || has_excl,
            "expected punctuation preserved: {tokens:?}"
        );
    }

    #[test]
    fn unit_preserve_punctuation_false() {
        let t = LanguageAgnosticTokenizer {
            preserve_punctuation: false,
            ..Default::default()
        };
        let tokens = t.tokenize_str("hello, world!");
        let has_punc = tokens.iter().any(|s| is_pure_punctuation(s));
        assert!(!has_punc, "unexpected punctuation token: {tokens:?}");
        assert!(
            tokens.iter().any(|s| s == "hello"),
            "missing 'hello': {tokens:?}"
        );
    }

    #[test]
    fn unit_max_token_len() {
        let t = LanguageAgnosticTokenizer {
            max_token_len: Some(3),
            ..Default::default()
        };
        let tokens = t.tokenize_str("superlongword");
        assert!(
            tokens.iter().all(|s| s.chars().count() <= 3),
            "token exceeded max_len=3: {tokens:?}"
        );
    }

    #[test]
    fn unit_nfc_normalization_idempotent() {
        let t = LanguageAgnosticTokenizer {
            normalize: true,
            ..Default::default()
        };
        // NFD café vs NFC café — after normalization, results must match.
        let nfd_cafe = "cafe\u{0301}"; // NFD: e + combining accent
        let nfc_cafe = "caf\u{00E9}"; // NFC: é as single char
        let t1 = t.tokenize_str(nfd_cafe);
        let t2 = t.tokenize_str(nfc_cafe);
        assert_eq!(t1, t2, "NFC normalization not idempotent: {t1:?} vs {t2:?}");
    }

    #[test]
    fn unit_trait_tokenize_result() {
        let t = LanguageAgnosticTokenizer::new();
        let result = <LanguageAgnosticTokenizer as Tokenizer>::tokenize(&t, "hello world");
        assert!(result.is_ok());
        let tokens = result.unwrap_or_default();
        assert!(tokens.iter().any(|s| s == "hello"), "tokens: {tokens:?}");
    }
}
