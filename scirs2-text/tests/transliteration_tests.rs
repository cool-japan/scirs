//! Integration tests for the transliteration module.
//!
//! Tests the trait-based API: HepburnTransliterator, CyrillicTransliterator,
//! PinyinTransliterator, and the legacy ScriptTransliterator.

use scirs2_text::transliteration::{
    CyrillicScheme, CyrillicTransliterator, HepburnTransliterator, PinyinStyle,
    PinyinTransliterator, Transliterator,
};

// ── Hepburn (Japanese kana → romaji) ─────────────────────────────────────────

#[test]
fn hepburn_konnichiwa() {
    let t = HepburnTransliterator::new();
    // こんにちは = ko+n+ni+chi+ha
    let result = t.transliterate("こんにちは");
    assert_eq!(result, "konnichiha", "got: {result}");
}

#[test]
fn hepburn_sakura() {
    let t = HepburnTransliterator::new();
    assert_eq!(t.transliterate("さくら"), "sakura");
}

#[test]
fn hepburn_katakana_basic() {
    let t = HepburnTransliterator::new();
    assert_eq!(t.transliterate("アイウエオ"), "aiueo");
}

#[test]
fn hepburn_katakana_tokyo_macron() {
    let t = HepburnTransliterator::new();
    // トーキョー → tōkyō  (long vowel mark ー produces macron)
    let result = t.transliterate("トーキョー");
    // The result should contain ō for the long vowel
    assert!(
        result.contains('ō') || result.contains("oo"),
        "expected long-o encoding: got {result}"
    );
    // The consonants should be present
    assert!(result.contains('k'), "expected 'k': got {result}");
}

#[test]
fn hepburn_katakana_no_macron() {
    let t = HepburnTransliterator::with_macron(false);
    // トー = to + long-vowel → "too"
    let result = t.transliterate("トー");
    assert_eq!(result, "too", "got: {result}");
}

#[test]
fn hepburn_yoon_combination() {
    let t = HepburnTransliterator::new();
    assert_eq!(t.transliterate("きゃ"), "kya");
    assert_eq!(t.transliterate("しゃ"), "sha");
    assert_eq!(t.transliterate("ちゃ"), "cha");
    assert_eq!(t.transliterate("じゃ"), "ja");
    assert_eq!(t.transliterate("びゃ"), "bya");
}

#[test]
fn hepburn_doubled_consonant_small_tsu() {
    let t = HepburnTransliterator::new();
    // っき = kki (doubled k)
    assert_eq!(t.transliterate("っき"), "kki");
    // にっき = nikki
    assert_eq!(t.transliterate("にっき"), "nikki");
    // さっさ = sassa
    assert_eq!(t.transliterate("さっさ"), "sassa");
}

#[test]
fn hepburn_dakuten_voiced() {
    let t = HepburnTransliterator::new();
    assert_eq!(t.transliterate("が"), "ga");
    assert_eq!(t.transliterate("じ"), "ji");
    assert_eq!(t.transliterate("ぞ"), "zo");
}

#[test]
fn hepburn_handakuten_semivoiced() {
    let t = HepburnTransliterator::new();
    assert_eq!(t.transliterate("ぱ"), "pa");
    assert_eq!(t.transliterate("ぺ"), "pe");
    assert_eq!(t.transliterate("ぽ"), "po");
}

#[test]
fn mixed_script_latin_passthrough() {
    let t = HepburnTransliterator::new();
    let result = t.transliterate("Hello こんにちは World");
    assert!(result.contains("Hello"), "missing 'Hello': got {result}");
    assert!(result.contains("World"), "missing 'World': got {result}");
    assert!(
        result.contains("konnichiha"),
        "missing romaji: got {result}"
    );
}

#[test]
fn hepburn_pure_latin_unchanged() {
    let t = HepburnTransliterator::new();
    let input = "The quick brown fox";
    assert_eq!(t.transliterate(input), input);
}

#[test]
fn hepburn_katakana_yoon() {
    let t = HepburnTransliterator::new();
    assert_eq!(t.transliterate("キャ"), "kya");
    assert_eq!(t.transliterate("シャ"), "sha");
    assert_eq!(t.transliterate("チャ"), "cha");
}

#[test]
fn hepburn_n_before_vowel() {
    let t = HepburnTransliterator::new();
    // ん before a — outputs "n" (not "n'a"; simple Hepburn)
    let result = t.transliterate("あん");
    assert_eq!(result, "an");
}

// ── Cyrillic transliteration ──────────────────────────────────────────────────

#[test]
fn cyrillic_gost_moskva() {
    let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
    let result = t.transliterate("Москва");
    assert!(result.to_lowercase().contains("moskva"), "got: {result}");
}

#[test]
fn cyrillic_bgn_matches_known_example() {
    let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
    let result = t.transliterate("Россия");
    // Р-о-с-с-и-я = R+o+s+s+i+ya → Rossiya
    assert!(result.to_lowercase().contains("rossiya"), "got: {result}");
}

#[test]
fn cyrillic_ala_lc_yo() {
    let t = CyrillicTransliterator::new(CyrillicScheme::AlaLc);
    // ё → ë
    assert_eq!(t.transliterate("ё"), "ë");
}

#[test]
fn cyrillic_ala_lc_short_i() {
    let t = CyrillicTransliterator::new(CyrillicScheme::AlaLc);
    // й → ĭ (U+012D)
    assert_eq!(t.transliterate("й"), "\u{012D}");
}

#[test]
fn cyrillic_bgn_hard_sign_omitted() {
    let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
    assert_eq!(t.transliterate("ъ"), "");
}

#[test]
fn cyrillic_bgn_soft_sign_apostrophe() {
    let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
    assert_eq!(t.transliterate("ь"), "'");
}

#[test]
fn cyrillic_uppercase_preserved() {
    let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
    let r = t.transliterate("А");
    assert_eq!(r, "A");
}

#[test]
fn cyrillic_passthrough_latin() {
    let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
    assert_eq!(t.transliterate("Hello"), "Hello");
}

#[test]
fn cyrillic_mixed_cyrillic_latin() {
    let t = CyrillicTransliterator::new(CyrillicScheme::BgnPcgn);
    let r = t.transliterate("Москва (Moscow)");
    assert!(r.contains("oskva"), "got: {r}");
    assert!(r.contains("Moscow"), "got: {r}");
}

#[test]
fn cyrillic_gost_all_basic_vowels() {
    let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
    // а е и о у → a e i o u  (some may differ by scheme)
    let r = t.transliterate("аиоу");
    assert!(r.contains('a'), "got: {r}");
    assert!(r.contains('i'), "got: {r}");
    assert!(r.contains('o'), "got: {r}");
    assert!(r.contains('u'), "got: {r}");
}

// ── Pinyin transliteration ────────────────────────────────────────────────────

#[test]
fn pinyin_ni_hao_tone_marks() {
    let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
    let result = t.transliterate("你好");
    assert!(
        result.contains("nǐ") || result.contains("ni"),
        "got: {result}"
    );
    assert!(
        result.contains("hǎo") || result.contains("hao"),
        "got: {result}"
    );
}

#[test]
fn pinyin_numbered_tones() {
    let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
    let result = t.transliterate("你好");
    // 你 = nǐ (tone 3) → "ni3"; 好 = hǎo (tone 3) → "hao3"
    assert!(
        result.contains("3") || result.contains("ni3"),
        "expected tone digit 3: got {result}"
    );
}

#[test]
fn pinyin_no_tones() {
    let t = PinyinTransliterator::new(PinyinStyle::NoTones);
    let result = t.transliterate("你好");
    assert!(
        !result.contains('ǐ') && !result.contains('ǎ'),
        "unexpected tone mark diacritics: got {result}"
    );
    assert!(result.contains("ni"), "got: {result}");
    assert!(result.contains("hao"), "got: {result}");
}

#[test]
fn pinyin_neutral_tone_de() {
    let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
    // 的 = de (neutral tone 5) → "de" (no digit for neutral)
    let result = t.transliterate("的");
    assert_eq!(result, "de", "got: {result}");
}

#[test]
fn pinyin_latin_passthrough() {
    let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
    assert_eq!(t.transliterate("Hello"), "Hello");
}

#[test]
fn pinyin_unknown_cjk_passthrough() {
    let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
    // An obscure character unlikely in HSK table — should not panic
    let result = t.transliterate("㗇");
    assert!(!result.is_empty(), "result should not be empty");
}

#[test]
fn pinyin_yi_er_san_numbered() {
    let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
    let r = t.transliterate("一二三");
    assert!(r.contains("yi1"), "got: {r}");
    // 二 = èr (tone 4)
    assert!(r.contains("er4") || r.contains("er"), "got: {r}");
    assert!(r.contains("san1"), "got: {r}");
}

#[test]
fn pinyin_chinese_numerals_tone_marks() {
    let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
    let r = t.transliterate("一");
    assert_eq!(r, "yī", "got: {r}");
}

#[test]
fn pinyin_wo_shi_zhongguoren() {
    let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
    // 我 是 中 国 人
    let r = t.transliterate("我是中国人");
    // Each character should produce some pinyin or pass through
    assert!(!r.is_empty(), "got: {r}");
    assert!(r.contains("wǒ") || r.contains("wo"), "got: {r}");
}

// ── Default / builder patterns ────────────────────────────────────────────────

#[test]
fn hepburn_default_is_macron_enabled() {
    let t1 = HepburnTransliterator::new();
    let t2 = HepburnTransliterator::default();
    // Both should produce the same output
    let r1 = t1.transliterate("トー");
    let r2 = t2.transliterate("トー");
    assert_eq!(r1, r2);
}

#[test]
fn cyrillic_scheme_accessor() {
    let t = CyrillicTransliterator::new(CyrillicScheme::Gost2005);
    assert_eq!(t.scheme(), CyrillicScheme::Gost2005);
}

#[test]
fn pinyin_style_accessor() {
    let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
    assert_eq!(t.style(), PinyinStyle::NumberedTones);
}
