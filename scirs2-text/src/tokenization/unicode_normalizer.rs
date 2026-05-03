//! Unicode normalization and language-agnostic tokenization utilities.
//!
//! Provides:
//! - [`Script`]: Unicode script detection for individual characters.
//! - [`UnicodeNormalizer`]: NFC/NFD normalization, accent stripping, case folding.
//! - Language-agnostic tokenization that handles CJK character segmentation.

use unicode_normalization::UnicodeNormalization;

// ─── Script detection ─────────────────────────────────────────────────────────

/// Unicode script classification for a single character.
///
/// Used to determine whether whitespace should be inserted around individual
/// characters (e.g. CJK) or whether a word-based tokenization strategy is
/// appropriate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    /// Latin characters (includes Latin Extended).
    Latin,
    /// CJK Unified Ideographs and related blocks.
    Cjk,
    /// Cyrillic script.
    Cyrillic,
    /// Arabic script.
    Arabic,
    /// Devanagari script (used for Hindi, Sanskrit, etc.).
    Devanagari,
    /// Hebrew script.
    Hebrew,
    /// Any script not listed above.
    Other,
}

/// Detect the [`Script`] for a single Unicode character.
///
/// Uses Unicode block ranges.  Characters that straddle multiple blocks
/// (e.g. punctuation) fall into [`Script::Other`].
pub fn detect_script(c: char) -> Script {
    let cp = c as u32;

    // CJK ranges
    if (0x4E00..=0x9FFF).contains(&cp)   // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&cp)  // CJK Extension A
        || (0x20000..=0x2A6DF).contains(&cp) // CJK Extension B
        || (0x2A700..=0x2B73F).contains(&cp) // CJK Extension C
        || (0x2B740..=0x2B81F).contains(&cp) // CJK Extension D
        || (0x2B820..=0x2CEAF).contains(&cp) // CJK Extension E
        || (0xF900..=0xFAFF).contains(&cp)  // CJK Compatibility Ideographs
        || (0x2F800..=0x2FA1F).contains(&cp) // CJK Compatibility Supplement
        || (0x3000..=0x303F).contains(&cp)  // CJK Symbols and Punctuation
        || (0x3040..=0x309F).contains(&cp)  // Hiragana
        || (0x30A0..=0x30FF).contains(&cp)
    // Katakana
    {
        return Script::Cjk;
    }

    // Cyrillic U+0400–U+04FF
    if (0x0400..=0x04FF).contains(&cp) {
        return Script::Cyrillic;
    }

    // Arabic U+0600–U+06FF
    if (0x0600..=0x06FF).contains(&cp) {
        return Script::Arabic;
    }

    // Devanagari U+0900–U+097F
    if (0x0900..=0x097F).contains(&cp) {
        return Script::Devanagari;
    }

    // Hebrew U+0590–U+05FF
    if (0x0590..=0x05FF).contains(&cp) {
        return Script::Hebrew;
    }

    // Latin: Basic Latin letters + Latin-1 Supplement + Latin Extended-A/B
    if (0x0041..=0x005A).contains(&cp)   // A-Z
        || (0x0061..=0x007A).contains(&cp) // a-z
        || (0x00C0..=0x00D6).contains(&cp)
        || (0x00D8..=0x00F6).contains(&cp)
        || (0x00F8..=0x024F).contains(&cp)
    // Latin Extended-A and B
    {
        return Script::Latin;
    }

    Script::Other
}

// ─── NormForm ─────────────────────────────────────────────────────────────────

/// Unicode normalization form.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormForm {
    /// Canonical Decomposition, followed by Canonical Composition (NFC).
    Nfc,
    /// Canonical Decomposition (NFD).
    Nfd,
}

// ─── UnicodeNormalizerConfig ───────────────────────────────────────────────────

/// Configuration for [`UnicodeNormalizer`].
#[derive(Debug, Clone)]
pub struct UnicodeNormalizerConfig {
    /// Normalization form to apply.
    pub form: NormForm,
    /// Strip combining diacritical marks (accent removal).
    pub strip_accents: bool,
    /// Fold all characters to lowercase.
    pub lowercase: bool,
    /// Insert whitespace around CJK characters to facilitate word splitting.
    pub tokenize_cjk: bool,
}

impl Default for UnicodeNormalizerConfig {
    fn default() -> Self {
        UnicodeNormalizerConfig {
            form: NormForm::Nfc,
            strip_accents: false,
            lowercase: false,
            tokenize_cjk: true,
        }
    }
}

// ─── UnicodeNormalizer ────────────────────────────────────────────────────────

/// Unicode-aware text normalizer.
///
/// Supports NFC/NFD normalization, accent stripping, case folding, and
/// language-agnostic CJK tokenization.
///
/// # Example
///
/// ```rust
/// use scirs2_text::tokenization::unicode_normalizer::{UnicodeNormalizer, UnicodeNormalizerConfig, NormForm};
///
/// let config = UnicodeNormalizerConfig {
///     form: NormForm::Nfc,
///     strip_accents: true,
///     lowercase: true,
///     tokenize_cjk: true,
/// };
/// let normalizer = UnicodeNormalizer::new(config);
/// let tokens = normalizer.tokenize_language_agnostic("Héllo 世界");
/// assert!(tokens.len() >= 3); // "hello", "世", "界"
/// ```
#[derive(Debug, Clone)]
pub struct UnicodeNormalizer {
    config: UnicodeNormalizerConfig,
}

impl UnicodeNormalizer {
    /// Create a new [`UnicodeNormalizer`] with the given configuration.
    pub fn new(config: UnicodeNormalizerConfig) -> Self {
        UnicodeNormalizer { config }
    }

    /// Create a normalizer with default settings.
    pub fn default_normalizer() -> Self {
        UnicodeNormalizer::new(UnicodeNormalizerConfig::default())
    }

    /// Normalize `text` according to the configuration.
    ///
    /// Steps applied in order:
    /// 1. Lowercase (if configured)
    /// 2. NFD decomposition + accent stripping (if configured)
    /// 3. NFC composition (if configured, after potential NFD strip)
    pub fn normalize(&self, text: &str) -> String {
        // Step 1: Lowercase
        let s = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_owned()
        };

        // Step 2 & 3: Normalize form + optional accent strip
        match self.config.form {
            NormForm::Nfd => {
                if self.config.strip_accents {
                    // NFD then remove combining marks
                    s.nfd().filter(|&c| !is_combining_diacritic(c)).collect()
                } else {
                    s.nfd().collect()
                }
            }
            NormForm::Nfc => {
                if self.config.strip_accents {
                    // NFD decompose → strip accents → NFC recompose
                    let stripped: String =
                        s.nfd().filter(|&c| !is_combining_diacritic(c)).collect();
                    stripped.nfc().collect()
                } else {
                    s.nfc().collect()
                }
            }
        }
    }

    /// Tokenize `text` in a language-agnostic manner.
    ///
    /// Algorithm:
    /// 1. Normalize the text.
    /// 2. Insert whitespace around CJK characters (when `tokenize_cjk` is set).
    /// 3. Split on Unicode whitespace.
    /// 4. Filter empty tokens.
    ///
    /// This approach works across scripts without any language-specific logic.
    pub fn tokenize_language_agnostic(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize(text);

        let mut spaced = String::with_capacity(normalized.len() * 2);
        for ch in normalized.chars() {
            if self.config.tokenize_cjk && is_cjk_character(ch) {
                // Surround each CJK character with spaces so it becomes its own token
                spaced.push(' ');
                spaced.push(ch);
                spaced.push(' ');
            } else {
                spaced.push(ch);
            }
        }

        spaced
            .split(|c: char| c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_owned())
            .collect()
    }

    /// Return the configuration.
    pub fn config(&self) -> &UnicodeNormalizerConfig {
        &self.config
    }
}

impl Default for UnicodeNormalizer {
    fn default() -> Self {
        UnicodeNormalizer::new(UnicodeNormalizerConfig::default())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return `true` for Unicode combining diacritical marks (U+0300–U+036F and
/// related blocks).
fn is_combining_diacritic(ch: char) -> bool {
    let cp = ch as u32;
    // Combining Diacritical Marks
    (0x0300..=0x036F).contains(&cp)
    // Combining Diacritical Marks Supplement
    || (0x1DC0..=0x1DFF).contains(&cp)
    // Combining Diacritical Marks Extended
    || (0x1AB0..=0x1AFF).contains(&cp)
    // Combining Half Marks
    || (0xFE20..=0xFE2F).contains(&cp)
}

/// Return `true` for CJK characters that should be individually tokenized.
fn is_cjk_character(ch: char) -> bool {
    let cp = ch as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
        || (0x3040..=0x309F).contains(&cp) // Hiragana
        || (0x30A0..=0x30FF).contains(&cp) // Katakana
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_script ───────────────────────────────────────────────────

    #[test]
    fn test_detect_script_latin() {
        assert_eq!(detect_script('a'), Script::Latin);
        assert_eq!(detect_script('Z'), Script::Latin);
        assert_eq!(detect_script('é'), Script::Latin); // U+00E9 Latin Small Letter E with Acute
    }

    #[test]
    fn test_detect_script_cjk() {
        assert_eq!(detect_script('中'), Script::Cjk); // U+4E2D
        assert_eq!(detect_script('日'), Script::Cjk); // U+65E5
        assert_eq!(detect_script('語'), Script::Cjk); // U+8A9E
    }

    #[test]
    fn test_detect_script_cyrillic() {
        assert_eq!(detect_script('А'), Script::Cyrillic); // U+0410
        assert_eq!(detect_script('я'), Script::Cyrillic); // U+044F
    }

    #[test]
    fn test_detect_script_arabic() {
        assert_eq!(detect_script('ع'), Script::Arabic); // U+0639
        assert_eq!(detect_script('م'), Script::Arabic); // U+0645
    }

    #[test]
    fn test_detect_script_devanagari() {
        assert_eq!(detect_script('क'), Script::Devanagari); // U+0915
        assert_eq!(detect_script('ा'), Script::Devanagari); // U+093E
    }

    #[test]
    fn test_detect_script_hebrew() {
        assert_eq!(detect_script('א'), Script::Hebrew); // U+05D0
        assert_eq!(detect_script('ש'), Script::Hebrew); // U+05E9
    }

    #[test]
    fn test_detect_script_other() {
        assert_eq!(detect_script('!'), Script::Other);
        assert_eq!(detect_script(' '), Script::Other);
        assert_eq!(detect_script('1'), Script::Other);
    }

    // ── UnicodeNormalizer::normalize ────────────────────────────────────

    #[test]
    fn test_normalize_lowercase() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            lowercase: true,
            ..Default::default()
        });
        assert_eq!(n.normalize("Hello WORLD"), "hello world");
    }

    #[test]
    fn test_normalize_no_lowercase() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            lowercase: false,
            ..Default::default()
        });
        assert_eq!(n.normalize("Hello WORLD"), "Hello WORLD");
    }

    #[test]
    fn test_normalize_strip_accents_nfc() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            form: NormForm::Nfc,
            strip_accents: true,
            lowercase: false,
            tokenize_cjk: false,
        });
        // "café" → "cafe"
        let result = n.normalize("café");
        assert_eq!(result, "cafe");
    }

    #[test]
    fn test_normalize_strip_accents_nfd() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            form: NormForm::Nfd,
            strip_accents: true,
            lowercase: false,
            tokenize_cjk: false,
        });
        let result = n.normalize("résumé");
        assert_eq!(result, "resume");
    }

    #[test]
    fn test_normalize_nfc_idempotent_on_ascii() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            form: NormForm::Nfc,
            strip_accents: false,
            lowercase: false,
            tokenize_cjk: false,
        });
        let text = "hello world 123";
        assert_eq!(n.normalize(text), text);
    }

    // ── tokenize_language_agnostic ───────────────────────────────────────

    #[test]
    fn test_cjk_chars_split() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            tokenize_cjk: true,
            lowercase: false,
            strip_accents: false,
            form: NormForm::Nfc,
        });
        let tokens = n.tokenize_language_agnostic("Hello世界");
        // "Hello" should be one token; "世" and "界" should each be their own
        assert!(tokens.contains(&"Hello".to_string()), "got: {:?}", tokens);
        assert!(tokens.contains(&"世".to_string()), "got: {:?}", tokens);
        assert!(tokens.contains(&"界".to_string()), "got: {:?}", tokens);
    }

    #[test]
    fn test_cjk_split_mixed_text() {
        let n = UnicodeNormalizer::default();
        let tokens = n.tokenize_language_agnostic("我 love Rust");
        // "我" is CJK and should be its own token
        assert!(tokens.iter().any(|t| t == "我"), "got: {:?}", tokens);
        assert!(tokens.iter().any(|t| t == "love"), "got: {:?}", tokens);
        assert!(tokens.iter().any(|t| t == "Rust"), "got: {:?}", tokens);
    }

    #[test]
    fn test_tokenize_latin_only() {
        let n = UnicodeNormalizer::default();
        let tokens = n.tokenize_language_agnostic("the quick brown fox");
        assert_eq!(tokens, vec!["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let n = UnicodeNormalizer::default();
        let tokens = n.tokenize_language_agnostic("   ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_with_lowercase_and_accent_strip() {
        let n = UnicodeNormalizer::new(UnicodeNormalizerConfig {
            form: NormForm::Nfc,
            strip_accents: true,
            lowercase: true,
            tokenize_cjk: true,
        });
        let tokens = n.tokenize_language_agnostic("Héllo Wörld");
        assert!(tokens.iter().any(|t| t == "hello"), "got: {:?}", tokens);
        assert!(tokens.iter().any(|t| t == "world"), "got: {:?}", tokens);
    }

    #[test]
    fn test_combining_mark_detection() {
        // U+0301 is COMBINING ACUTE ACCENT — a combining diacritic
        assert!(is_combining_diacritic('\u{0301}'));
        assert!(is_combining_diacritic('\u{0300}'));
        assert!(is_combining_diacritic('\u{036F}'));
        // Regular ASCII should not be diacritics
        assert!(!is_combining_diacritic('a'));
        assert!(!is_combining_diacritic('é')); // precomposed — single codepoint
    }

    #[test]
    fn test_cjk_character_detection() {
        assert!(is_cjk_character('中'));
        assert!(is_cjk_character('日'));
        assert!(is_cjk_character('あ')); // Hiragana
        assert!(is_cjk_character('ア')); // Katakana
        assert!(!is_cjk_character('a'));
        assert!(!is_cjk_character('1'));
        assert!(!is_cjk_character(' '));
    }
}
