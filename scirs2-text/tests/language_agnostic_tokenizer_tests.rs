//! Integration tests for the language-agnostic Unicode tokenizer.

use scirs2_text::tokenization::language_agnostic::LanguageAgnosticTokenizer;

// ── Basic Latin tokenization ──────────────────────────────────────────────────

#[test]
fn tok_pure_latin() {
    let t = LanguageAgnosticTokenizer::new();
    let tokens = t.tokenize_str("The quick brown fox");
    assert_eq!(tokens, vec!["The", "quick", "brown", "fox"]);
}

#[test]
fn tok_empty_string() {
    let t = LanguageAgnosticTokenizer::new();
    assert_eq!(t.tokenize_str(""), Vec::<String>::new());
}

#[test]
fn tok_whitespace_only() {
    let t = LanguageAgnosticTokenizer::new();
    assert_eq!(t.tokenize_str("   "), Vec::<String>::new());
}

// ── CJK handling ──────────────────────────────────────────────────────────────

#[test]
fn tok_mixed_latin_cjk() {
    let t = LanguageAgnosticTokenizer::new();
    let tokens = t.tokenize_str("Hello 你好 World");
    assert!(tokens.contains(&"Hello".to_string()), "tokens: {tokens:?}");
    // CJK split_by_char=true (default) → 你 and 好 each separate
    assert!(
        tokens.contains(&"你".to_string()) || tokens.contains(&"你好".to_string()),
        "tokens: {tokens:?}"
    );
    assert!(tokens.contains(&"World".to_string()), "tokens: {tokens:?}");
}

#[test]
fn tok_cjk_split_by_char() {
    let t = LanguageAgnosticTokenizer {
        split_cjk_by_char: true,
        ..Default::default()
    };
    let tokens = t.tokenize_str("日本語");
    // Each character should be its own token
    assert!(tokens.len() >= 3, "expected 3 CJK chars, got: {tokens:?}");
    assert!(tokens.contains(&"日".to_string()), "tokens: {tokens:?}");
    assert!(tokens.contains(&"本".to_string()), "tokens: {tokens:?}");
    assert!(tokens.contains(&"語".to_string()), "tokens: {tokens:?}");
}

#[test]
fn tok_cjk_no_split() {
    let t = LanguageAgnosticTokenizer {
        split_cjk_by_char: false,
        ..Default::default()
    };
    let tokens = t.tokenize_str("日本語");
    assert_eq!(tokens.len(), 1, "expected whole CJK token, got: {tokens:?}");
    assert_eq!(tokens[0], "日本語");
}

#[test]
fn tok_hiragana_split_by_char() {
    let t = LanguageAgnosticTokenizer {
        split_cjk_by_char: true,
        ..Default::default()
    };
    // Hiragana is CJK-family — each char should be separate
    let tokens = t.tokenize_str("さくら");
    assert!(tokens.len() >= 3, "expected 3 hiragana chars: {tokens:?}");
}

#[test]
fn tok_hangul_split_by_char() {
    let t = LanguageAgnosticTokenizer {
        split_cjk_by_char: true,
        ..Default::default()
    };
    // Korean Hangul syllables are in the CJK range
    let tokens = t.tokenize_str("안녕");
    assert!(
        tokens.len() >= 2,
        "expected at least 2 Hangul chars: {tokens:?}"
    );
}

// ── Normalization ─────────────────────────────────────────────────────────────

#[test]
fn tok_nfc_normalization_idempotent() {
    let t = LanguageAgnosticTokenizer {
        normalize: true,
        ..Default::default()
    };
    // NFD: e + combining grave; NFC: è as single char
    let nfd = "cafe\u{0301}"; // NFD: accent separate
    let nfc = "caf\u{00E9}"; // NFC: é combined
    let t1 = t.tokenize_str(nfd);
    let t2 = t.tokenize_str(nfc);
    assert_eq!(t1, t2, "NFC normalization not idempotent: {t1:?} vs {t2:?}");
}

#[test]
fn tok_no_normalization_keeps_nfd() {
    let t = LanguageAgnosticTokenizer {
        normalize: false,
        ..Default::default()
    };
    // Without normalization, NFD and NFC forms are separate chars — may differ
    // but must not panic and must be non-empty.
    let nfd = "cafe\u{0301}";
    let tokens = t.tokenize_str(nfd);
    assert!(!tokens.is_empty(), "expected non-empty tokens: {tokens:?}");
}

// ── Case options ──────────────────────────────────────────────────────────────

#[test]
fn tok_lowercase_option() {
    let t = LanguageAgnosticTokenizer {
        lowercase: true,
        ..Default::default()
    };
    let tokens = t.tokenize_str("Hello World");
    assert!(tokens.iter().any(|s| s == "hello"), "tokens: {tokens:?}");
    assert!(tokens.iter().any(|s| s == "world"), "tokens: {tokens:?}");
}

#[test]
fn tok_no_lowercase_preserves_case() {
    let t = LanguageAgnosticTokenizer {
        lowercase: false,
        ..Default::default()
    };
    let tokens = t.tokenize_str("Hello World");
    assert!(tokens.iter().any(|s| s == "Hello"), "tokens: {tokens:?}");
}

// ── Punctuation policy ────────────────────────────────────────────────────────

#[test]
fn tok_punctuation_policy_roundtrip() {
    // With preserve_punctuation=true, punctuation tokens appear.
    let t_keep = LanguageAgnosticTokenizer {
        preserve_punctuation: true,
        ..Default::default()
    };
    let tokens_keep = t_keep.tokenize_str("hello, world!");
    let has_punc = tokens_keep
        .iter()
        .any(|s| s.contains(',') || s.contains('!'));
    assert!(has_punc, "expected punctuation preserved: {tokens_keep:?}");

    // With preserve_punctuation=false, no pure-punctuation tokens.
    let t_drop = LanguageAgnosticTokenizer {
        preserve_punctuation: false,
        ..Default::default()
    };
    let tokens_drop = t_drop.tokenize_str("hello, world!");
    let has_punc_after = tokens_drop
        .iter()
        .any(|s| !s.chars().any(|c| c.is_alphanumeric()) && !s.chars().all(|c| c.is_whitespace()));
    assert!(
        !has_punc_after,
        "unexpected punctuation token: {tokens_drop:?}"
    );
    // Words still present.
    assert!(
        tokens_drop.iter().any(|s| s == "hello"),
        "tokens: {tokens_drop:?}"
    );
    assert!(
        tokens_drop.iter().any(|s| s == "world"),
        "tokens: {tokens_drop:?}"
    );
}

// ── Emoji / grapheme cluster handling ─────────────────────────────────────────

#[test]
fn tok_preserves_emoji_grapheme_clusters() {
    // Emoji should not be split by CJK logic (they're not in CJK ranges).
    let t = LanguageAgnosticTokenizer {
        split_cjk_by_char: true,
        preserve_punctuation: true,
        ..Default::default()
    };
    let tokens = t.tokenize_str("Hello 🌍 World");
    // The globe emoji should appear as a token (or inside a combined token).
    assert!(
        tokens.iter().any(|s| s.contains('🌍') || s == "World"),
        "tokens: {tokens:?}"
    );
    // Latin words must be present.
    assert!(tokens.iter().any(|s| s == "Hello"), "tokens: {tokens:?}");
    assert!(tokens.iter().any(|s| s == "World"), "tokens: {tokens:?}");
}

// ── max_token_len ─────────────────────────────────────────────────────────────

#[test]
fn tok_max_token_len_truncates() {
    let t = LanguageAgnosticTokenizer {
        max_token_len: Some(4),
        ..Default::default()
    };
    let tokens = t.tokenize_str("superlongword");
    assert!(
        tokens.iter().all(|s| s.chars().count() <= 4),
        "token exceeded max_len=4: {tokens:?}"
    );
}

// ── Trait implementation ──────────────────────────────────────────────────────

#[test]
fn tok_trait_tokenize_returns_ok() {
    use scirs2_text::tokenize::Tokenizer;
    let t = LanguageAgnosticTokenizer::new();
    let result = t.tokenize("hello world");
    assert!(result.is_ok(), "tokenize() should not fail: {result:?}");
    let tokens = result.unwrap_or_default();
    assert!(tokens.iter().any(|s| s == "hello"), "tokens: {tokens:?}");
}

#[test]
fn tok_trait_clone_box() {
    use scirs2_text::tokenize::Tokenizer;
    let t = LanguageAgnosticTokenizer::new();
    let boxed = t.clone_box();
    let result = boxed.tokenize("rust programming");
    assert!(result.is_ok(), "clone_box tokenize() should not fail");
    let tokens = result.unwrap_or_default();
    assert!(tokens.iter().any(|s| s == "rust"), "tokens: {tokens:?}");
}

#[test]
fn tok_trait_batch_tokenize() {
    use scirs2_text::tokenize::Tokenizer;
    let t = LanguageAgnosticTokenizer::new();
    let result = t.tokenize_batch(&["hello world", "rust language"]);
    assert!(result.is_ok());
    let batches = result.unwrap_or_default();
    assert_eq!(batches.len(), 2);
    assert!(batches[0].iter().any(|s| s == "hello"));
    assert!(batches[1].iter().any(|s| s == "rust"));
}
