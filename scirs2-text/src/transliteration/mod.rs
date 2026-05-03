//! Transliteration utilities: convert non-Latin scripts to Latin characters.
//!
//! # Overview
//!
//! This module provides two tiers of API:
//!
//! ## Trait-based API (new)
//! High-level, script-specific transliterators each implementing the
//! [`Transliterator`] trait:
//!
//! | Type | Script | Notes |
//! |------|--------|-------|
//! | [`HepburnTransliterator`] | Japanese kana | Modified Hepburn, yōon, long vowels |
//! | [`CyrillicTransliterator`] | Cyrillic | GOST 2005, BGN/PCGN 1947, ALA-LC |
//! | [`PinyinTransliterator`] | Simplified Chinese | HSK 1-6, tone marks or numbered |
//!
//! ## Struct-based API (legacy)
//! [`ScriptTransliterator`] accepts a [`Script`] enum at call time and dispatches
//! to embedded tables for Cyrillic (ISO 9), Greek (ALA-LC), Hiragana and Katakana.
//! This API is kept for backwards-compatibility.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_text::transliteration::{
//!     Transliterator, HepburnTransliterator, CyrillicTransliterator,
//!     CyrillicScheme, PinyinTransliterator, PinyinStyle,
//! };
//!
//! // Japanese kana → romaji
//! let hepburn = HepburnTransliterator::new();
//! assert_eq!(hepburn.transliterate("さくら"), "sakura");
//!
//! // Cyrillic → Latin (BGN/PCGN)
//! let cyrillic = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
//! let r = cyrillic.transliterate("Москва");
//! assert!(r.to_lowercase().contains("moskva"));
//!
//! // Chinese → Pinyin
//! let pinyin = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
//! let r = pinyin.transliterate("你好");
//! assert!(r.contains("nǐ") || r.contains("ni"));
//! ```

use unicode_normalization::UnicodeNormalization;

pub mod cyrillic;
pub mod hepburn;
pub mod pinyin;

pub use cyrillic::{CyrillicScheme, CyrillicTransliterator};
pub use hepburn::HepburnTransliterator;
pub use pinyin::{PinyinStyle, PinyinTransliterator};

// ─── Core trait ───────────────────────────────────────────────────────────────

/// Common interface for all script-specific transliterators.
///
/// Each implementation converts its source script to a Latin representation.
/// Characters that are not part of the source script (e.g. Latin letters,
/// digits, punctuation) are passed through unchanged.
pub trait Transliterator {
    /// Transliterate `input` and return the Latin representation.
    fn transliterate(&self, input: &str) -> String;
}

// ─── Legacy struct-based API ──────────────────────────────────────────────────

/// Japanese writing system variant.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum JapaneseScript {
    /// Hiragana syllabary (U+3040–U+309F).
    Hiragana,
    /// Katakana syllabary (U+30A0–U+30FF).
    Katakana,
    /// Latin romaji representation.
    Romaji,
}

/// Chinese romanisation system.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChineseSystem {
    /// Hanyu Pinyin (mainland standard).
    Pinyin,
    /// Wade-Giles system.
    #[allow(non_camel_case_types)]
    Wade_Giles,
}

/// Writing script identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Script {
    /// Cyrillic script (U+0400–U+04FF).
    Cyrillic,
    /// Greek script (U+0370–U+03FF).
    Greek,
    /// Arabic script (U+0600–U+06FF).
    Arabic,
    /// Hebrew script (U+0590–U+05FF).
    Hebrew,
    /// Japanese (Hiragana / Katakana).
    Japanese(JapaneseScript),
    /// Korean Hangul (U+AC00–U+D7AF).
    Korean,
    /// Chinese characters (U+4E00–U+9FFF).
    Chinese(ChineseSystem),
    /// Latin / ASCII script.
    Latin,
}

/// Configuration for the [`ScriptTransliterator`].
#[derive(Debug, Clone)]
pub struct TranslitConfig {
    /// If `true`, preserve case through mapping of uppercase source characters.
    pub preserve_case: bool,
    /// If `true`, strip diacritics from the output string.
    pub strip_diacritics: bool,
}

impl Default for TranslitConfig {
    fn default() -> Self {
        Self {
            preserve_case: true,
            strip_diacritics: false,
        }
    }
}

// ─── Static transliteration tables (legacy) ───────────────────────────────────

/// Cyrillic → Latin (ISO 9).
pub static CYRILLIC_TO_LATIN: &[(&str, &str)] = &[
    // Lowercase
    ("а", "a"),
    ("б", "b"),
    ("в", "v"),
    ("г", "g"),
    ("д", "d"),
    ("е", "je"),
    ("ё", "jo"),
    ("ж", "zh"),
    ("з", "z"),
    ("и", "i"),
    ("й", "j"),
    ("к", "k"),
    ("л", "l"),
    ("м", "m"),
    ("н", "n"),
    ("о", "o"),
    ("п", "p"),
    ("р", "r"),
    ("с", "s"),
    ("т", "t"),
    ("у", "u"),
    ("ф", "f"),
    ("х", "h"),
    ("ц", "c"),
    ("ч", "ch"),
    ("ш", "sh"),
    ("щ", "shh"),
    ("ъ", "\u{2033}"), // double prime ″
    ("ы", "y"),
    ("ь", "\u{2032}"), // prime ′
    ("э", "eh"),
    ("ю", "ju"),
    ("я", "ja"),
    // Uppercase
    ("А", "A"),
    ("Б", "B"),
    ("В", "V"),
    ("Г", "G"),
    ("Д", "D"),
    ("Е", "Je"),
    ("Ё", "Jo"),
    ("Ж", "Zh"),
    ("З", "Z"),
    ("И", "I"),
    ("Й", "J"),
    ("К", "K"),
    ("Л", "L"),
    ("М", "M"),
    ("Н", "N"),
    ("О", "O"),
    ("П", "P"),
    ("Р", "R"),
    ("С", "S"),
    ("Т", "T"),
    ("У", "U"),
    ("Ф", "F"),
    ("Х", "H"),
    ("Ц", "C"),
    ("Ч", "Ch"),
    ("Ш", "Sh"),
    ("Щ", "Shh"),
    ("Ъ", "\u{2033}"),
    ("Ы", "Y"),
    ("Ь", "\u{2032}"),
    ("Э", "Eh"),
    ("Ю", "Ju"),
    ("Я", "Ja"),
];

/// Greek → Latin (ALA-LC).
pub static GREEK_TO_LATIN: &[(&str, &str)] = &[
    ("α", "a"),
    ("β", "b"),
    ("γ", "g"),
    ("δ", "d"),
    ("ε", "e"),
    ("ζ", "z"),
    ("η", "\u{0113}"), // ē
    ("θ", "th"),
    ("ι", "i"),
    ("κ", "k"),
    ("λ", "l"),
    ("μ", "m"),
    ("ν", "n"),
    ("ξ", "x"),
    ("ο", "o"),
    ("π", "p"),
    ("ρ", "r"),
    ("σ", "s"),
    ("ς", "s"), // final sigma
    ("τ", "t"),
    ("υ", "y"),
    ("φ", "ph"),
    ("χ", "ch"),
    ("ψ", "ps"),
    ("ω", "\u{014D}"), // ō
    // Uppercase
    ("Α", "A"),
    ("Β", "B"),
    ("Γ", "G"),
    ("Δ", "D"),
    ("Ε", "E"),
    ("Ζ", "Z"),
    ("Η", "\u{0112}"), // Ē
    ("Θ", "Th"),
    ("Ι", "I"),
    ("Κ", "K"),
    ("Λ", "L"),
    ("Μ", "M"),
    ("Ν", "N"),
    ("Ξ", "X"),
    ("Ο", "O"),
    ("Π", "P"),
    ("Ρ", "R"),
    ("Σ", "S"),
    ("Τ", "T"),
    ("Υ", "Y"),
    ("Φ", "Ph"),
    ("Χ", "Ch"),
    ("Ψ", "Ps"),
    ("Ω", "\u{014C}"), // Ō
];

/// Hiragana → Romaji (Hepburn).
pub static HIRAGANA_TO_ROMAJI: &[(&str, &str)] = &[
    ("あ", "a"),
    ("い", "i"),
    ("う", "u"),
    ("え", "e"),
    ("お", "o"),
    ("か", "ka"),
    ("き", "ki"),
    ("く", "ku"),
    ("け", "ke"),
    ("こ", "ko"),
    ("さ", "sa"),
    ("し", "shi"),
    ("す", "su"),
    ("せ", "se"),
    ("そ", "so"),
    ("た", "ta"),
    ("ち", "chi"),
    ("つ", "tsu"),
    ("て", "te"),
    ("と", "to"),
    ("な", "na"),
    ("に", "ni"),
    ("ぬ", "nu"),
    ("ね", "ne"),
    ("の", "no"),
    ("は", "ha"),
    ("ひ", "hi"),
    ("ふ", "fu"),
    ("へ", "he"),
    ("ほ", "ho"),
    ("ま", "ma"),
    ("み", "mi"),
    ("む", "mu"),
    ("め", "me"),
    ("も", "mo"),
    ("や", "ya"),
    ("ゆ", "yu"),
    ("よ", "yo"),
    ("ら", "ra"),
    ("り", "ri"),
    ("る", "ru"),
    ("れ", "re"),
    ("ろ", "ro"),
    ("わ", "wa"),
    ("を", "wo"),
    ("ん", "n"),
    // Voiced consonants
    ("が", "ga"),
    ("ぎ", "gi"),
    ("ぐ", "gu"),
    ("げ", "ge"),
    ("ご", "go"),
    ("ざ", "za"),
    ("じ", "ji"),
    ("ず", "zu"),
    ("ぜ", "ze"),
    ("ぞ", "zo"),
    ("だ", "da"),
    ("ぢ", "di"),
    ("づ", "du"),
    ("で", "de"),
    ("ど", "do"),
    ("ば", "ba"),
    ("び", "bi"),
    ("ぶ", "bu"),
    ("べ", "be"),
    ("ぼ", "bo"),
    // Semi-voiced
    ("ぱ", "pa"),
    ("ぴ", "pi"),
    ("ぷ", "pu"),
    ("ぺ", "pe"),
    ("ぽ", "po"),
    // Small vowels / combination starters
    ("ぁ", "xa"),
    ("ぃ", "xi"),
    ("ぅ", "xu"),
    ("ぇ", "xe"),
    ("ぉ", "xo"),
];

/// Katakana → Romaji (Hepburn).
pub static KATAKANA_TO_ROMAJI: &[(&str, &str)] = &[
    ("ア", "a"),
    ("イ", "i"),
    ("ウ", "u"),
    ("エ", "e"),
    ("オ", "o"),
    ("カ", "ka"),
    ("キ", "ki"),
    ("ク", "ku"),
    ("ケ", "ke"),
    ("コ", "ko"),
    ("サ", "sa"),
    ("シ", "shi"),
    ("ス", "su"),
    ("セ", "se"),
    ("ソ", "so"),
    ("タ", "ta"),
    ("チ", "chi"),
    ("ツ", "tsu"),
    ("テ", "te"),
    ("ト", "to"),
    ("ナ", "na"),
    ("ニ", "ni"),
    ("ヌ", "nu"),
    ("ネ", "ne"),
    ("ノ", "no"),
    ("ハ", "ha"),
    ("ヒ", "hi"),
    ("フ", "fu"),
    ("ヘ", "he"),
    ("ホ", "ho"),
    ("マ", "ma"),
    ("ミ", "mi"),
    ("ム", "mu"),
    ("メ", "me"),
    ("モ", "mo"),
    ("ヤ", "ya"),
    ("ユ", "yu"),
    ("ヨ", "yo"),
    ("ラ", "ra"),
    ("リ", "ri"),
    ("ル", "ru"),
    ("レ", "re"),
    ("ロ", "ro"),
    ("ワ", "wa"),
    ("ヲ", "wo"),
    ("ン", "n"),
    // Voiced
    ("ガ", "ga"),
    ("ギ", "gi"),
    ("グ", "gu"),
    ("ゲ", "ge"),
    ("ゴ", "go"),
    ("ザ", "za"),
    ("ジ", "ji"),
    ("ズ", "zu"),
    ("ゼ", "ze"),
    ("ゾ", "zo"),
    ("ダ", "da"),
    ("ヂ", "di"),
    ("ヅ", "du"),
    ("デ", "de"),
    ("ド", "do"),
    ("バ", "ba"),
    ("ビ", "bi"),
    ("ブ", "bu"),
    ("ベ", "be"),
    ("ボ", "bo"),
    // Semi-voiced
    ("パ", "pa"),
    ("ピ", "pi"),
    ("プ", "pu"),
    ("ペ", "pe"),
    ("ポ", "po"),
];

// ─── ScriptTransliterator (legacy, struct-based) ──────────────────────────────

/// Stateful transliterator (legacy struct-based API).
///
/// For new code, prefer the trait-based API with [`HepburnTransliterator`],
/// [`CyrillicTransliterator`], or [`PinyinTransliterator`].
pub struct ScriptTransliterator {
    config: TranslitConfig,
}

impl ScriptTransliterator {
    /// Create a new `ScriptTransliterator` with the given configuration.
    pub fn new(config: TranslitConfig) -> Self {
        Self { config }
    }

    /// Transliterate `text` from `from` script to Latin.
    ///
    /// Characters that have no entry in the table are passed through unchanged.
    pub fn transliterate(&self, text: &str, from: &Script) -> String {
        let table: &[(&str, &str)] = match from {
            Script::Cyrillic => CYRILLIC_TO_LATIN,
            Script::Greek => GREEK_TO_LATIN,
            Script::Japanese(JapaneseScript::Hiragana) => HIRAGANA_TO_ROMAJI,
            Script::Japanese(JapaneseScript::Katakana) => KATAKANA_TO_ROMAJI,
            Script::Japanese(JapaneseScript::Romaji) | Script::Latin => {
                // Already Latin — just pass through (optionally strip diacritics).
                return if self.config.strip_diacritics {
                    strip_diacritics(text)
                } else {
                    text.to_string()
                };
            }
            _ => {
                // Arabic, Hebrew, Korean, Chinese, etc. — no table yet, return as-is.
                return text.to_string();
            }
        };

        let mut result = String::with_capacity(text.len() * 2);
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        'outer: while i < chars.len() {
            // Try to match the longest table entry starting at position i.
            // Build a candidate string starting at chars[i].
            let mut candidate = String::new();
            for &ch in &chars[i..] {
                candidate.push(ch);
                // Check if any table entry starts with this prefix.
                let any_prefix = table
                    .iter()
                    .any(|(src, _)| src.starts_with(candidate.as_str()));
                if !any_prefix {
                    break;
                }
            }
            // Now try decreasing lengths to find an exact match.
            let remaining: String = chars[i..].iter().collect();
            for (src, dst) in table.iter() {
                if remaining.starts_with(src) {
                    result.push_str(dst);
                    i += src.chars().count();
                    continue 'outer;
                }
            }
            // No match: emit the character unchanged.
            result.push(chars[i]);
            i += 1;
        }

        if self.config.strip_diacritics {
            strip_diacritics(&result)
        } else {
            result
        }
    }

    /// Detect the predominant writing script of `text` based on Unicode block ranges.
    ///
    /// Returns `Script::Latin` if no recognised non-Latin characters are found.
    pub fn detect_script(text: &str) -> Script {
        let mut cyrillic = 0usize;
        let mut greek = 0usize;
        let mut arabic = 0usize;
        let mut hebrew = 0usize;
        let mut hiragana = 0usize;
        let mut katakana = 0usize;
        let mut hangul = 0usize;
        let mut cjk = 0usize;

        for ch in text.chars() {
            let cp = ch as u32;
            if (0x0400..=0x04FF).contains(&cp) {
                cyrillic += 1;
            } else if (0x0370..=0x03FF).contains(&cp) {
                greek += 1;
            } else if (0x0600..=0x06FF).contains(&cp) {
                arabic += 1;
            } else if (0x0590..=0x05FF).contains(&cp) {
                hebrew += 1;
            } else if (0x3040..=0x309F).contains(&cp) {
                hiragana += 1;
            } else if (0x30A0..=0x30FF).contains(&cp) {
                katakana += 1;
            } else if (0xAC00..=0xD7AF).contains(&cp) {
                hangul += 1;
            } else if (0x4E00..=0x9FFF).contains(&cp) {
                cjk += 1;
            }
        }

        // Return the script with the most characters; fall back to Latin if none.
        let scores: [(usize, fn() -> Script); 8] = [
            (cyrillic, || Script::Cyrillic),
            (greek, || Script::Greek),
            (arabic, || Script::Arabic),
            (hebrew, || Script::Hebrew),
            (hiragana, || Script::Japanese(JapaneseScript::Hiragana)),
            (katakana, || Script::Japanese(JapaneseScript::Katakana)),
            (hangul, || Script::Korean),
            (cjk, || Script::Chinese(ChineseSystem::Pinyin)),
        ];

        let best = scores.iter().max_by_key(|(count, _)| *count);

        match best {
            Some((count, make_script)) if *count > 0 => make_script(),
            _ => Script::Latin,
        }
    }
}

/// Strip diacritical combining marks (U+0300–U+036F) from a string.
///
/// The string is first NFD-decomposed, then all combining characters in the
/// diacritics range are removed, and the result is NFC-recomposed.
pub fn strip_diacritics(s: &str) -> String {
    s.nfd()
        .filter(|ch| {
            let cp = *ch as u32;
            !(0x0300..=0x036F).contains(&cp)
        })
        .nfc()
        .collect()
}

// ─── Tests (legacy API) ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cyrillic() {
        assert_eq!(
            ScriptTransliterator::detect_script("Привет"),
            Script::Cyrillic
        );
    }

    #[test]
    fn test_detect_greek() {
        assert_eq!(ScriptTransliterator::detect_script("αβγδ"), Script::Greek);
    }

    #[test]
    fn test_detect_hiragana() {
        let s = ScriptTransliterator::detect_script("あいうえお");
        assert_eq!(s, Script::Japanese(JapaneseScript::Hiragana));
    }

    #[test]
    fn test_detect_katakana() {
        let s = ScriptTransliterator::detect_script("アイウエオ");
        assert_eq!(s, Script::Japanese(JapaneseScript::Katakana));
    }

    #[test]
    fn test_detect_latin_fallback() {
        assert_eq!(
            ScriptTransliterator::detect_script("hello world"),
            Script::Latin
        );
    }

    #[test]
    fn test_transliterate_cyrillic() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        let result = t.transliterate("привет", &Script::Cyrillic);
        // "привет" → "p"+"r"+"i"+"v"+"je"+"t" = "privjet"
        assert!(
            result
                .chars()
                .all(|c| c.is_ascii() || c == '\u{2032}' || c == '\u{2033}'),
            "Cyrillic should transliterate to Latin-like chars, got: {}",
            result
        );
        assert!(!result.is_empty());
    }

    #[test]
    fn test_transliterate_cyrillic_known() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        assert_eq!(t.transliterate("а", &Script::Cyrillic), "a");
        assert_eq!(t.transliterate("б", &Script::Cyrillic), "b");
        assert_eq!(t.transliterate("ш", &Script::Cyrillic), "sh");
    }

    #[test]
    fn test_transliterate_hiragana_aiu() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        let result = t.transliterate("あいう", &Script::Japanese(JapaneseScript::Hiragana));
        assert_eq!(result, "aiu");
    }

    #[test]
    fn test_transliterate_hiragana_full_word() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        // "さくら" (sakura)
        let result = t.transliterate("さくら", &Script::Japanese(JapaneseScript::Hiragana));
        assert_eq!(result, "sakura");
    }

    #[test]
    fn test_transliterate_katakana() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        let result = t.transliterate("アイウ", &Script::Japanese(JapaneseScript::Katakana));
        assert_eq!(result, "aiu");
    }

    #[test]
    fn test_transliterate_greek() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        let result = t.transliterate("αβγ", &Script::Greek);
        assert_eq!(result, "abg");
    }

    #[test]
    fn test_strip_diacritics() {
        // "café" → "cafe"
        let s = strip_diacritics("café");
        assert_eq!(s, "cafe");
    }

    #[test]
    fn test_strip_diacritics_config() {
        let t = ScriptTransliterator::new(TranslitConfig {
            strip_diacritics: true,
            ..Default::default()
        });
        // Greek η transliterates to ē (with macron); with strip_diacritics it becomes e.
        let result = t.transliterate("η", &Script::Greek);
        assert_eq!(result, "e");
    }

    #[test]
    fn test_no_match_passthrough() {
        let t = ScriptTransliterator::new(TranslitConfig::default());
        // ASCII should pass through unchanged for Cyrillic transliterator.
        let result = t.transliterate("abc", &Script::Cyrillic);
        assert_eq!(result, "abc");
    }
}
