//! Hepburn romanisation for Japanese hiragana and katakana.
//!
//! Implements the Modified Hepburn system (the most widely-used standard).
//! Yōon (combination sounds) are matched before single kana so that e.g.
//! "きゃ" maps to "kya" rather than "ki"+"ya".
//! Long vowel marks (ー) and small-tsu doubled consonants are handled
//! context-sensitively when `use_macron` is true.

use super::Transliterator;

// ── Yōon (combination sounds) hiragana ───────────────────────────────────────
// Must come BEFORE single-kana entries so longest-match wins.
static HIRAGANA_YOON: &[(&str, &str)] = &[
    ("きゃ", "kya"),
    ("きゅ", "kyu"),
    ("きょ", "kyo"),
    ("しゃ", "sha"),
    ("しゅ", "shu"),
    ("しょ", "sho"),
    ("ちゃ", "cha"),
    ("ちゅ", "chu"),
    ("ちょ", "cho"),
    ("にゃ", "nya"),
    ("にゅ", "nyu"),
    ("にょ", "nyo"),
    ("ひゃ", "hya"),
    ("ひゅ", "hyu"),
    ("ひょ", "hyo"),
    ("みゃ", "mya"),
    ("みゅ", "myu"),
    ("みょ", "myo"),
    ("りゃ", "rya"),
    ("りゅ", "ryu"),
    ("りょ", "ryo"),
    ("ぎゃ", "gya"),
    ("ぎゅ", "gyu"),
    ("ぎょ", "gyo"),
    ("じゃ", "ja"),
    ("じゅ", "ju"),
    ("じょ", "jo"),
    ("びゃ", "bya"),
    ("びゅ", "byu"),
    ("びょ", "byo"),
    ("ぴゃ", "pya"),
    ("ぴゅ", "pyu"),
    ("ぴょ", "pyo"),
    ("ちゃ", "cha"),
    ("ちゅ", "chu"),
    ("ちょ", "cho"),
    ("にゃ", "nya"),
    ("にゅ", "nyu"),
    ("にょ", "nyo"),
];

// ── Base hiragana ─────────────────────────────────────────────────────────────
static HIRAGANA_BASE: &[(&str, &str)] = &[
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
    ("ゐ", "i"),
    ("ゑ", "e"),
    ("を", "wo"),
    ("ん", "n"),
    // Voiced (dakuten)
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
    ("ぢ", "ji"),
    ("づ", "zu"),
    ("で", "de"),
    ("ど", "do"),
    ("ば", "ba"),
    ("び", "bi"),
    ("ぶ", "bu"),
    ("べ", "be"),
    ("ぼ", "bo"),
    // Semi-voiced (handakuten)
    ("ぱ", "pa"),
    ("ぴ", "pi"),
    ("ぷ", "pu"),
    ("ぺ", "pe"),
    ("ぽ", "po"),
    // Small vowels (treated as regular vowels in output)
    ("ぁ", "a"),
    ("ぃ", "i"),
    ("ぅ", "u"),
    ("ぇ", "e"),
    ("ぉ", "o"),
    // Small ya/yu/yo (appear after yōon combos are consumed)
    ("ゃ", "ya"),
    ("ゅ", "yu"),
    ("ょ", "yo"),
    // Small wa
    ("ゎ", "wa"),
    // Special
    ("ゔ", "vu"),
];

// ── Yōon katakana ─────────────────────────────────────────────────────────────
static KATAKANA_YOON: &[(&str, &str)] = &[
    ("キャ", "kya"),
    ("キュ", "kyu"),
    ("キョ", "kyo"),
    ("シャ", "sha"),
    ("シュ", "shu"),
    ("ショ", "sho"),
    ("チャ", "cha"),
    ("チュ", "chu"),
    ("チョ", "cho"),
    ("ニャ", "nya"),
    ("ニュ", "nyu"),
    ("ニョ", "nyo"),
    ("ヒャ", "hya"),
    ("ヒュ", "hyu"),
    ("ヒョ", "hyo"),
    ("ミャ", "mya"),
    ("ミュ", "myu"),
    ("ミョ", "myo"),
    ("リャ", "rya"),
    ("リュ", "ryu"),
    ("リョ", "ryo"),
    ("ギャ", "gya"),
    ("ギュ", "gyu"),
    ("ギョ", "gyo"),
    ("ジャ", "ja"),
    ("ジュ", "ju"),
    ("ジョ", "jo"),
    ("ビャ", "bya"),
    ("ビュ", "byu"),
    ("ビョ", "byo"),
    ("ピャ", "pya"),
    ("ピュ", "pyu"),
    ("ピョ", "pyo"),
    // Extended katakana for foreign words
    ("ファ", "fa"),
    ("フィ", "fi"),
    ("フェ", "fe"),
    ("フォ", "fo"),
    ("ウィ", "wi"),
    ("ウェ", "we"),
    ("ウォ", "wo"),
    ("ティ", "ti"),
    ("ディ", "di"),
    ("ツァ", "tsa"),
    ("ツェ", "tse"),
    ("ツォ", "tso"),
    ("チェ", "che"),
    ("ジェ", "je"),
    ("シェ", "she"),
    ("イェ", "ye"),
    ("ヴァ", "va"),
    ("ヴィ", "vi"),
    ("ヴェ", "ve"),
    ("ヴォ", "vo"),
];

// ── Base katakana ─────────────────────────────────────────────────────────────
static KATAKANA_BASE: &[(&str, &str)] = &[
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
    // Voiced (dakuten)
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
    ("ヂ", "ji"),
    ("ヅ", "zu"),
    ("デ", "de"),
    ("ド", "do"),
    ("バ", "ba"),
    ("ビ", "bi"),
    ("ブ", "bu"),
    ("ベ", "be"),
    ("ボ", "bo"),
    // Semi-voiced (handakuten)
    ("パ", "pa"),
    ("ピ", "pi"),
    ("プ", "pu"),
    ("ペ", "pe"),
    ("ポ", "po"),
    // Small vowels
    ("ァ", "a"),
    ("ィ", "i"),
    ("ゥ", "u"),
    ("ェ", "e"),
    ("ォ", "o"),
    // Small ya/yu/yo
    ("ャ", "ya"),
    ("ュ", "yu"),
    ("ョ", "yo"),
    // Small tsu handled separately
    // Small wa
    ("ヮ", "wa"),
    // Vu
    ("ヴ", "vu"),
];

// Unicode code points for special characters
const HIRAGANA_SMALL_TSU: char = 'っ';
const KATAKANA_SMALL_TSU: char = 'ッ';
const KATAKANA_LONG_VOWEL: char = 'ー';

// Macron-extended vowels for long vowel representation
const fn long_vowel_macron(romaji: &str) -> Option<&'static str> {
    match romaji.as_bytes() {
        b"a" => Some("ā"),
        b"i" => Some("ī"),
        b"u" => Some("ū"),
        b"e" => Some("ē"),
        b"o" => Some("ō"),
        _ => None,
    }
}

/// Hepburn romanisation transliterator for Japanese.
///
/// Converts hiragana and katakana to Hepburn romaji.
/// Non-Japanese characters are passed through unchanged.
///
/// # Long vowels
/// When `use_macron` is `true` (default), the katakana long-vowel mark `ー`
/// extends the preceding vowel using a macron (ā ī ū ē ō).
/// When `false`, `ー` is rendered as a repeated vowel.
///
/// # Doubled consonants
/// The small tsu `っ`/`ッ` is rendered by doubling the initial consonant of
/// the following syllable (e.g. `っき` → `kki`).
#[derive(Debug, Clone)]
pub struct HepburnTransliterator {
    use_macron: bool,
}

impl HepburnTransliterator {
    /// Create a new `HepburnTransliterator` with macron-based long vowels enabled.
    pub fn new() -> Self {
        Self { use_macron: true }
    }

    /// Create a new `HepburnTransliterator` with explicit macron setting.
    pub fn with_macron(use_macron: bool) -> Self {
        Self { use_macron }
    }
}

impl Default for HepburnTransliterator {
    fn default() -> Self {
        Self::new()
    }
}

impl Transliterator for HepburnTransliterator {
    fn transliterate(&self, input: &str) -> String {
        let chars: Vec<char> = input.chars().collect();
        let mut result = String::with_capacity(input.len() * 2);
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            // Handle small tsu (doubled consonant)
            if ch == HIRAGANA_SMALL_TSU || ch == KATAKANA_SMALL_TSU {
                // Peek ahead: get the romaji of the next syllable
                if i + 1 < chars.len() {
                    let next_romaji = transliterate_single_char_or_combo(&chars, i + 1);
                    // The initial consonant of the next syllable is doubled
                    if let Some(first_byte) = next_romaji
                        .chars()
                        .next()
                        .filter(|c| c.is_ascii_alphabetic())
                    {
                        result.push(first_byte);
                        // Don't advance i; let the next iteration handle i+1 normally
                        i += 1;
                        continue;
                    }
                }
                // Fallback: emit nothing (small tsu with no following kana)
                i += 1;
                continue;
            }

            // Handle katakana long vowel mark ー
            if ch == KATAKANA_LONG_VOWEL {
                if self.use_macron {
                    // Find the last vowel in result and apply macron
                    let prev_vowel = get_last_vowel_char(&result);
                    if let Some(v) = prev_vowel.and_then(long_vowel_macron) {
                        // Replace last character (the short vowel) with macron version
                        replace_last_vowel_with_macron(&mut result, v);
                    } else {
                        result.push(ch); // pass through if no prior vowel
                    }
                } else {
                    // Repeat the last vowel
                    if let Some(v) = get_last_vowel_char(&result) {
                        let v_char = v.chars().next().unwrap_or('ー');
                        result.push(v_char);
                    } else {
                        result.push(ch);
                    }
                }
                i += 1;
                continue;
            }

            // Try yōon (2-char combo) first for both hiragana and katakana
            if i + 1 < chars.len() {
                let two_char: String = chars[i..=i + 1].iter().collect();
                if let Some(romaji) = lookup_table(HIRAGANA_YOON, &two_char)
                    .or_else(|| lookup_table(KATAKANA_YOON, &two_char))
                {
                    result.push_str(romaji);
                    i += 2;
                    continue;
                }
            }

            // Try single hiragana
            let one_char: String = std::iter::once(ch).collect();
            if let Some(romaji) = lookup_table(HIRAGANA_BASE, &one_char)
                .or_else(|| lookup_table(KATAKANA_BASE, &one_char))
            {
                result.push_str(romaji);
                i += 1;
                continue;
            }

            // Pass through unchanged (Latin, CJK, punctuation, etc.)
            result.push(ch);
            i += 1;
        }

        result
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Look up a string in a static table; returns the romanisation or `None`.
fn lookup_table<'a>(table: &'a [(&str, &str)], key: &str) -> Option<&'a str> {
    table
        .iter()
        .find(|(src, _)| *src == key)
        .map(|(_, dst)| *dst)
}

/// Get the romaji of the syllable starting at index `idx` in `chars`,
/// used for peeking ahead to double the consonant after small-tsu.
fn transliterate_single_char_or_combo(chars: &[char], idx: usize) -> String {
    if idx >= chars.len() {
        return String::new();
    }
    // Try two-char combo first
    if idx + 1 < chars.len() {
        let two: String = chars[idx..=idx + 1].iter().collect();
        if let Some(r) =
            lookup_table(HIRAGANA_YOON, &two).or_else(|| lookup_table(KATAKANA_YOON, &two))
        {
            return r.to_string();
        }
    }
    // Single char
    let one: String = std::iter::once(chars[idx]).collect();
    lookup_table(HIRAGANA_BASE, &one)
        .or_else(|| lookup_table(KATAKANA_BASE, &one))
        .map(|s| s.to_string())
        .unwrap_or_default()
}

/// Return a static string slice for the last vowel in `s`, if any.
fn get_last_vowel_char(s: &str) -> Option<&'static str> {
    let last_char = s.chars().next_back()?;
    match last_char {
        'a' => Some("a"),
        'i' => Some("i"),
        'u' => Some("u"),
        'e' => Some("e"),
        'o' => Some("o"),
        // Already has a macron? Handle repeated ー
        'ā' => Some("a"),
        'ī' => Some("i"),
        'ū' => Some("u"),
        'ē' => Some("e"),
        'ō' => Some("o"),
        _ => None,
    }
}

/// Replace the last short vowel (or macron vowel) in `s` with the macron form.
fn replace_last_vowel_with_macron(s: &mut String, macron: &str) {
    // The last character might be a multi-byte char (macron vowel)
    // Pop the last char and replace with macron
    if s.is_empty() {
        s.push_str(macron);
        return;
    }
    // Find the byte offset of the last character
    let last_char_len = s.chars().next_back().map(|c| c.len_utf8()).unwrap_or(0);
    let new_len = s.len() - last_char_len;
    s.truncate(new_len);
    s.push_str(macron);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transliteration::Transliterator;

    #[test]
    fn test_hiragana_basic() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("あいうえお"), "aiueo");
    }

    #[test]
    fn test_hiragana_sakura() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("さくら"), "sakura");
    }

    #[test]
    fn test_hiragana_konnichiwa() {
        let t = HepburnTransliterator::new();
        // こんにちは — "ko" + "n" + "ni" + "chi" + "ha"
        assert_eq!(t.transliterate("こんにちは"), "konnichiha");
    }

    #[test]
    fn test_hiragana_yoon() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("きゃ"), "kya");
        assert_eq!(t.transliterate("しゃ"), "sha");
        assert_eq!(t.transliterate("ちゃ"), "cha");
    }

    #[test]
    fn test_small_tsu_doubled_consonant() {
        let t = HepburnTransliterator::new();
        // っき → kki
        assert_eq!(t.transliterate("っき"), "kki");
        // にっき → nikki
        assert_eq!(t.transliterate("にっき"), "nikki");
    }

    #[test]
    fn test_katakana_basic() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("アイウエオ"), "aiueo");
    }

    #[test]
    fn test_katakana_long_vowel_macron() {
        let t = HepburnTransliterator::new();
        // トーキョー → tōkyō
        let result = t.transliterate("トーキョー");
        assert!(
            result.contains('ō') || result.contains("oo"),
            "expected long-o: got {result}"
        );
    }

    #[test]
    fn test_katakana_long_vowel_no_macron() {
        let t = HepburnTransliterator::with_macron(false);
        // トー → "too"
        let result = t.transliterate("トー");
        assert_eq!(result, "too");
    }

    #[test]
    fn test_latin_passthrough() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("Hello"), "Hello");
    }

    #[test]
    fn test_mixed_latin_hiragana() {
        let t = HepburnTransliterator::new();
        let result = t.transliterate("Hello こんにちは World");
        assert!(result.contains("Hello"), "got: {result}");
        assert!(result.contains("World"), "got: {result}");
        assert!(result.contains("konnichiha"), "got: {result}");
    }

    #[test]
    fn test_dakuten_voiced() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("が"), "ga");
        assert_eq!(t.transliterate("じ"), "ji");
        assert_eq!(t.transliterate("ぼ"), "bo");
    }

    #[test]
    fn test_handakuten_semivoiced() {
        let t = HepburnTransliterator::new();
        assert_eq!(t.transliterate("ぱ"), "pa");
        assert_eq!(t.transliterate("ぽ"), "po");
    }
}
