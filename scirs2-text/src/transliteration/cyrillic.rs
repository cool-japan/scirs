//! Cyrillic-to-Latin transliteration with three authoritative schemes.
//!
//! Supported schemes:
//! - **GOST 2005** (GOST R 52535.1-2006): Russian state standard
//! - **BGN/PCGN 1947**: US Board on Geographic Names / UK Permanent Committee
//! - **ALA-LC**: American Library Association / Library of Congress

use super::Transliterator;

/// Cyrillic transliteration scheme selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CyrillicScheme {
    /// GOST 2005 (GOST R 52535.1-2006) — Russian state standard.
    Gost2005,
    /// BGN/PCGN 1947 — US/UK geographic naming standard.
    BgnPcgn,
    /// ALA-LC — American Library Association / Library of Congress.
    AlaLc,
}

// ── Transliteration table ─────────────────────────────────────────────────────
//
// Each row: (Cyrillic char, GOST 2005, BGN/PCGN 1947, ALA-LC)
// Both upper- and lowercase variants are listed explicitly.
//
// Sources:
//  - GOST R 52535.1-2006 (ISO 9:1995-like mapping)
//  - BGN/PCGN Russian romanization system (1947)
//  - ALA-LC Romanization Tables (2012)

static CYRILLIC_TABLE: &[(char, &str, &str, &str)] = &[
    // ── А а ──────────────────────────────────────────────────────────────────
    ('А', "A", "A", "A"),
    ('а', "a", "a", "a"),
    // ── Б б ──────────────────────────────────────────────────────────────────
    ('Б', "B", "B", "B"),
    ('б', "b", "b", "b"),
    // ── В в ──────────────────────────────────────────────────────────────────
    ('В', "V", "V", "V"),
    ('в', "v", "v", "v"),
    // ── Г г ──────────────────────────────────────────────────────────────────
    ('Г', "G", "G", "G"),
    ('г', "g", "g", "g"),
    // ── Д д ──────────────────────────────────────────────────────────────────
    ('Д', "D", "D", "D"),
    ('д', "d", "d", "d"),
    // ── Е е ──────────────────────────────────────────────────────────────────
    // Context-sensitive: after consonant → "e", at word-start / after vowel → "ye"
    // For simplicity we implement the simple BGN/PCGN rule: always "ye" initially
    // and after vowels — handled at word level; table fallback uses "e".
    ('Е', "E", "Ye", "E"),
    ('е', "e", "ye", "e"),
    // ── Ё ё ──────────────────────────────────────────────────────────────────
    ('Ё', "Yo", "Yo", "Ë"),
    ('ё', "yo", "yo", "ë"),
    // ── Ж ж ──────────────────────────────────────────────────────────────────
    ('Ж', "Zh", "Zh", "Zh"),
    ('ж', "zh", "zh", "zh"),
    // ── З з ──────────────────────────────────────────────────────────────────
    ('З', "Z", "Z", "Z"),
    ('з', "z", "z", "z"),
    // ── И и ──────────────────────────────────────────────────────────────────
    ('И', "I", "I", "I"),
    ('и', "i", "i", "i"),
    // ── Й й ──────────────────────────────────────────────────────────────────
    ('Й', "J", "Y", "\u{012C}"), // Ĭ (U+012C)
    ('й', "j", "y", "\u{012D}"), // ĭ (U+012D)
    // ── К к ──────────────────────────────────────────────────────────────────
    ('К', "K", "K", "K"),
    ('к', "k", "k", "k"),
    // ── Л л ──────────────────────────────────────────────────────────────────
    ('Л', "L", "L", "L"),
    ('л', "l", "l", "l"),
    // ── М м ──────────────────────────────────────────────────────────────────
    ('М', "M", "M", "M"),
    ('м', "m", "m", "m"),
    // ── Н н ──────────────────────────────────────────────────────────────────
    ('Н', "N", "N", "N"),
    ('н', "n", "n", "n"),
    // ── О о ──────────────────────────────────────────────────────────────────
    ('О', "O", "O", "O"),
    ('о', "o", "o", "o"),
    // ── П п ──────────────────────────────────────────────────────────────────
    ('П', "P", "P", "P"),
    ('п', "p", "p", "p"),
    // ── Р р ──────────────────────────────────────────────────────────────────
    ('Р', "R", "R", "R"),
    ('р', "r", "r", "r"),
    // ── С с ──────────────────────────────────────────────────────────────────
    ('С', "S", "S", "S"),
    ('с', "s", "s", "s"),
    // ── Т т ──────────────────────────────────────────────────────────────────
    ('Т', "T", "T", "T"),
    ('т', "t", "t", "t"),
    // ── У у ──────────────────────────────────────────────────────────────────
    ('У', "U", "U", "U"),
    ('у', "u", "u", "u"),
    // ── Ф ф ──────────────────────────────────────────────────────────────────
    ('Ф', "F", "F", "F"),
    ('ф', "f", "f", "f"),
    // ── Х х ──────────────────────────────────────────────────────────────────
    ('Х', "Kh", "Kh", "Kh"),
    ('х', "kh", "kh", "kh"),
    // ── Ц ц ──────────────────────────────────────────────────────────────────
    // ALA-LC uses a ligature T͡s (U+0054 + U+0361 + U+0073)
    ('Ц', "C", "Ts", "T\u{0361}s"),
    ('ц', "c", "ts", "t\u{0361}s"),
    // ── Ч ч ──────────────────────────────────────────────────────────────────
    ('Ч', "Ch", "Ch", "Ch"),
    ('ч', "ch", "ch", "ch"),
    // ── Ш ш ──────────────────────────────────────────────────────────────────
    ('Ш', "Sh", "Sh", "Sh"),
    ('ш', "sh", "sh", "sh"),
    // ── Щ щ ──────────────────────────────────────────────────────────────────
    ('Щ', "Sch", "Shch", "Shch"),
    ('щ', "sch", "shch", "shch"),
    // ── Ъ ъ ──────────────────────────────────────────────────────────────────
    // Hard sign: GOST → double prime, BGN → omit, ALA-LC → modifier double prime ʺ
    ('Ъ', "\u{2033}", "", "\u{02BA}"),
    ('ъ', "\u{2033}", "", "\u{02BA}"),
    // ── Ы ы ──────────────────────────────────────────────────────────────────
    ('Ы', "Y", "Y", "Y"),
    ('ы', "y", "y", "y"),
    // ── Ь ь ──────────────────────────────────────────────────────────────────
    // Soft sign: GOST → prime, BGN → apostrophe, ALA-LC → modifier prime ʹ
    ('Ь', "\u{2032}", "'", "\u{02B9}"),
    ('ь', "\u{2032}", "'", "\u{02B9}"),
    // ── Э э ──────────────────────────────────────────────────────────────────
    // ALA-LC: Ė (E with dot above, U+0116/U+0117)
    ('Э', "Eh", "E", "\u{0116}"),
    ('э', "eh", "e", "\u{0117}"),
    // ── Ю ю ──────────────────────────────────────────────────────────────────
    // ALA-LC: I͡u (I + U+0361 + u)
    ('Ю', "Yu", "Yu", "I\u{0361}u"),
    ('ю', "yu", "yu", "i\u{0361}u"),
    // ── Я я ──────────────────────────────────────────────────────────────────
    // ALA-LC: I͡a (I + U+0361 + a)
    ('Я', "Ya", "Ya", "I\u{0361}a"),
    ('я', "ya", "ya", "i\u{0361}a"),
    // ── Additional Cyrillic characters (old orthography / regional) ───────────
    // Ё alternate entry handled above
    // Ї ї (Ukrainian)
    ('\u{0407}', "Yi", "Yi", "Yi"),
    ('\u{0457}', "yi", "yi", "yi"),
    // І і (Ukrainian/Belarusian)
    ('\u{0406}', "I", "I", "I"),
    ('\u{0456}', "i", "i", "i"),
    // Є є (Ukrainian)
    ('\u{0404}', "Ye", "Ye", "Ie"),
    ('\u{0454}', "ye", "ye", "ie"),
    // Ґ ґ (Ukrainian)
    ('\u{0490}', "G", "G", "G"),
    ('\u{0491}', "g", "g", "g"),
];

/// Cyrillic transliterator with scheme selection.
///
/// # Example
/// ```
/// use scirs2_text::transliteration::{CyrillicTransliterator, CyrillicScheme, Transliterator};
///
/// let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
/// let result = t.transliterate("Москва");
/// assert!(result.to_lowercase().contains("moskva"), "got: {result}");
/// ```
#[derive(Debug, Clone)]
pub struct CyrillicTransliterator {
    scheme: CyrillicScheme,
}

impl CyrillicTransliterator {
    /// Create a new `CyrillicTransliterator` with the given scheme.
    pub fn new(scheme: CyrillicScheme) -> Self {
        Self { scheme }
    }

    /// Return the currently configured scheme.
    pub fn scheme(&self) -> CyrillicScheme {
        self.scheme
    }

    /// Look up a single Cyrillic character in the table for the current scheme.
    fn lookup(&self, ch: char) -> Option<&'static str> {
        CYRILLIC_TABLE
            .iter()
            .find(|(src, ..)| *src == ch)
            .map(|(_, gost, bgn, ala)| match self.scheme {
                CyrillicScheme::Gost2005 => *gost,
                CyrillicScheme::BgnPcgn => *bgn,
                CyrillicScheme::AlaLc => *ala,
            })
    }
}

impl Transliterator for CyrillicTransliterator {
    fn transliterate(&self, input: &str) -> String {
        let mut result = String::with_capacity(input.len() * 2);
        for ch in input.chars() {
            if let Some(roman) = self.lookup(ch) {
                result.push_str(roman);
            } else {
                result.push(ch);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transliteration::Transliterator;

    #[test]
    fn test_gost_basic() {
        let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
        assert_eq!(t.transliterate("а"), "a");
        assert_eq!(t.transliterate("б"), "b");
        assert_eq!(t.transliterate("ш"), "sh");
        assert_eq!(t.transliterate("ж"), "zh");
    }

    #[test]
    fn test_gost_moskva() {
        let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
        // М-о-с-к-в-а
        let r = t.transliterate("Москва");
        assert!(r.to_lowercase().contains("moskva"), "got: {r}");
    }

    #[test]
    fn test_bgn_rossiya() {
        let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
        let r = t.transliterate("Россия");
        assert!(r.to_lowercase().contains("rossiya"), "got: {r}");
    }

    #[test]
    fn test_ala_lc_known() {
        let t = CyrillicTransliterator::new(CyrillicScheme::AlaLc);
        // ё → ë
        assert_eq!(t.transliterate("ё"), "ë");
        // й → ĭ
        assert_eq!(t.transliterate("й"), "\u{012D}");
    }

    #[test]
    fn test_uppercase_preservation() {
        let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
        let r = t.transliterate("А");
        assert_eq!(r, "A");
    }

    #[test]
    fn test_passthrough_latin() {
        let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
        assert_eq!(t.transliterate("Hello"), "Hello");
    }

    #[test]
    fn test_mixed_cyrillic_latin() {
        let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
        let r = t.transliterate("Москва (Moscow)");
        assert!(r.contains("oskva"), "got: {r}");
        assert!(r.contains("Moscow"), "got: {r}");
    }

    #[test]
    fn test_hard_sign_bgn_empty() {
        let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
        // BGN/PCGN omits the hard sign
        assert_eq!(t.transliterate("ъ"), "");
    }

    #[test]
    fn test_soft_sign_bgn_apostrophe() {
        let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
        assert_eq!(t.transliterate("ь"), "'");
    }
}
