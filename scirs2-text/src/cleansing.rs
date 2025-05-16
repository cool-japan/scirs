//! Advanced text cleansing utilities
//!
//! This module provides functions for cleaning text data including
//! HTML stripping, URL handling, and various normalization operations.

use crate::error::Result;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    // HTML/XML tag pattern
    static ref HTML_TAG_PATTERN: Regex = Regex::new(r"<[^>]+>").unwrap();

    // URL pattern
    static ref URL_PATTERN: Regex = Regex::new(
        r"(?i)https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
    ).unwrap();

    // Email pattern
    static ref EMAIL_PATTERN: Regex = Regex::new(
        r"(?i)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ).unwrap();

    // Phone number pattern (simplified, US-style)
    static ref PHONE_PATTERN: Regex = Regex::new(
        r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"
    ).unwrap();

    // Common contractions mapping
    static ref CONTRACTIONS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("can't", "cannot");
        m.insert("won't", "will not");
        m.insert("n't", " not");
        m.insert("'re", " are");
        m.insert("'ve", " have");
        m.insert("'ll", " will");
        m.insert("'d", " would");
        m.insert("'m", " am");
        m.insert("'s", " is");
        m.insert("let's", "let us");
        m.insert("it's", "it is");
        m.insert("that's", "that is");
        m.insert("what's", "what is");
        m.insert("where's", "where is");
        m.insert("who's", "who is");
        m.insert("there's", "there is");
        m.insert("here's", "here is");
        m
    };

    // Emoji pattern
    static ref EMOJI_PATTERN: Regex = Regex::new(
        concat!(
            "[",
            "\u{1F600}-\u{1F64F}", // Emoticons
            "\u{1F300}-\u{1F5FF}", // Symbols & Pictographs
            "\u{1F680}-\u{1F6FF}", // Transport & Map
            "\u{1F700}-\u{1F77F}", // Alchemical Symbols
            "\u{1F780}-\u{1F7FF}", // Geometric Shapes Extended
            "\u{1F800}-\u{1F8FF}", // Supplemental Arrows-C
            "\u{2600}-\u{26FF}",   // Miscellaneous Symbols
            "\u{2700}-\u{27BF}",   // Dingbats
            "]"
        )
    ).unwrap();
}

/// Advanced text cleaner with various cleaning options
#[derive(Debug, Clone)]
pub struct AdvancedTextCleaner {
    strip_html: bool,
    replace_urls: bool,
    replace_emails: bool,
    replace_phone_numbers: bool,
    expand_contractions: bool,
    remove_emojis: bool,
    normalize_unicode: bool,
    preserve_case: bool,
    url_placeholder: String,
    email_placeholder: String,
    phone_placeholder: String,
}

impl AdvancedTextCleaner {
    /// Create a new advanced text cleaner with default settings
    pub fn new() -> Self {
        Self {
            strip_html: true,
            replace_urls: true,
            replace_emails: true,
            replace_phone_numbers: false,
            expand_contractions: true,
            remove_emojis: false,
            normalize_unicode: true,
            preserve_case: false,
            url_placeholder: "[URL]".to_string(),
            email_placeholder: "[EMAIL]".to_string(),
            phone_placeholder: "[PHONE]".to_string(),
        }
    }

    /// Create a cleaner for privacy-sensitive text
    pub fn privacy_focused() -> Self {
        Self {
            strip_html: true,
            replace_urls: true,
            replace_emails: true,
            replace_phone_numbers: true,
            expand_contractions: true,
            remove_emojis: false,
            normalize_unicode: true,
            preserve_case: false,
            url_placeholder: "[URL]".to_string(),
            email_placeholder: "[EMAIL]".to_string(),
            phone_placeholder: "[PHONE]".to_string(),
        }
    }

    /// Create a minimal cleaner that preserves most content
    pub fn minimal() -> Self {
        Self {
            strip_html: true,
            replace_urls: false,
            replace_emails: false,
            replace_phone_numbers: false,
            expand_contractions: false,
            remove_emojis: false,
            normalize_unicode: true,
            preserve_case: true,
            url_placeholder: "[URL]".to_string(),
            email_placeholder: "[EMAIL]".to_string(),
            phone_placeholder: "[PHONE]".to_string(),
        }
    }

    /// Set whether to strip HTML tags
    pub fn set_strip_html(mut self, value: bool) -> Self {
        self.strip_html = value;
        self
    }

    /// Set whether to replace URLs
    pub fn set_replace_urls(mut self, value: bool) -> Self {
        self.replace_urls = value;
        self
    }

    /// Set whether to replace emails
    pub fn set_replace_emails(mut self, value: bool) -> Self {
        self.replace_emails = value;
        self
    }

    /// Set whether to replace phone numbers
    pub fn set_replace_phone_numbers(mut self, value: bool) -> Self {
        self.replace_phone_numbers = value;
        self
    }

    /// Set whether to expand contractions
    pub fn set_expand_contractions(mut self, value: bool) -> Self {
        self.expand_contractions = value;
        self
    }

    /// Set whether to remove emojis
    pub fn set_remove_emojis(mut self, value: bool) -> Self {
        self.remove_emojis = value;
        self
    }

    /// Set custom placeholders
    pub fn set_placeholders(
        mut self,
        url: Option<String>,
        email: Option<String>,
        phone: Option<String>,
    ) -> Self {
        if let Some(u) = url {
            self.url_placeholder = u;
        }
        if let Some(e) = email {
            self.email_placeholder = e;
        }
        if let Some(p) = phone {
            self.phone_placeholder = p;
        }
        self
    }

    /// Clean the text according to the configured options
    pub fn clean(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Strip HTML tags
        if self.strip_html {
            cleaned = strip_html_tags(&cleaned);
        }

        // Replace URLs
        if self.replace_urls {
            cleaned = URL_PATTERN
                .replace_all(&cleaned, &self.url_placeholder)
                .to_string();
        }

        // Replace emails
        if self.replace_emails {
            cleaned = EMAIL_PATTERN
                .replace_all(&cleaned, &self.email_placeholder)
                .to_string();
        }

        // Replace phone numbers
        if self.replace_phone_numbers {
            cleaned = PHONE_PATTERN
                .replace_all(&cleaned, &self.phone_placeholder)
                .to_string();
        }

        // Expand contractions
        if self.expand_contractions {
            cleaned = expand_contractions(&cleaned);
        }

        // Remove emojis
        if self.remove_emojis {
            cleaned = EMOJI_PATTERN.replace_all(&cleaned, " ").to_string();
        }

        // Normalize unicode
        if self.normalize_unicode {
            cleaned = normalize_unicode(&cleaned)?;
        }

        // Handle case
        if !self.preserve_case {
            cleaned = cleaned.to_lowercase();
        }

        // Normalize whitespace
        cleaned = normalize_whitespace(&cleaned);

        Ok(cleaned)
    }
}

impl Default for AdvancedTextCleaner {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip HTML and XML tags from text
pub fn strip_html_tags(text: &str) -> String {
    HTML_TAG_PATTERN.replace_all(text, " ").to_string()
}

/// Replace URLs with a placeholder
pub fn replace_urls(text: &str, placeholder: &str) -> String {
    URL_PATTERN.replace_all(text, placeholder).to_string()
}

/// Replace email addresses with a placeholder
pub fn replace_emails(text: &str, placeholder: &str) -> String {
    EMAIL_PATTERN.replace_all(text, placeholder).to_string()
}

/// Replace phone numbers with a placeholder
pub fn replace_phone_numbers(text: &str, placeholder: &str) -> String {
    PHONE_PATTERN.replace_all(text, placeholder).to_string()
}

/// Expand common contractions
pub fn expand_contractions(text: &str) -> String {
    let mut result = text.to_string();

    // Sort contractions by length (descending) to avoid partial replacements
    let mut contractions: Vec<_> = CONTRACTIONS.iter().collect();
    contractions.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));

    for (contraction, expansion) in contractions {
        let pattern = format!(r"\b{}\b", regex::escape(contraction));
        if let Ok(re) = Regex::new(&pattern) {
            result = re.replace_all(&result, *expansion).to_string();
        }
    }

    result
}

/// Normalize Unicode text (NFD -> NFC)
pub fn normalize_unicode(text: &str) -> Result<String> {
    use unicode_normalization::UnicodeNormalization;
    Ok(text.nfc().collect())
}

/// Normalize whitespace (multiple spaces to single space, trim)
pub fn normalize_whitespace(text: &str) -> String {
    lazy_static! {
        static ref WHITESPACE_PATTERN: Regex = Regex::new(r"\s+").unwrap();
    }

    WHITESPACE_PATTERN.replace_all(text.trim(), " ").to_string()
}

/// Remove accents from text
pub fn remove_accents(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    text.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect()
}

/// Convert various dash types to regular hyphen
pub fn normalize_dashes(text: &str) -> String {
    lazy_static! {
        static ref DASH_PATTERN: Regex = Regex::new(r"[\u{2010}-\u{2015}\u{2212}]").unwrap();
    }

    DASH_PATTERN.replace_all(text, "-").to_string()
}

/// Convert various quote types to regular quotes
pub fn normalize_quotes(text: &str) -> String {
    lazy_static! {
        static ref SINGLE_QUOTE_PATTERN: Regex =
            Regex::new(r"[\u{2018}\u{2019}\u{201A}\u{201B}]").unwrap();
        static ref DOUBLE_QUOTE_PATTERN: Regex =
            Regex::new(r"[\u{201C}\u{201D}\u{201E}\u{201F}]").unwrap();
    }

    let text = SINGLE_QUOTE_PATTERN.replace_all(text, "'");
    DOUBLE_QUOTE_PATTERN.replace_all(&text, "\"").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        let html = "<p>Hello <b>world</b>!</p>";
        let cleaned = strip_html_tags(html);
        assert_eq!(cleaned, " Hello  world ! ");
    }

    #[test]
    fn test_replace_urls() {
        let text = "Check out https://www.example.com for more info";
        let replaced = replace_urls(text, "[URL]");
        assert_eq!(replaced, "Check out [URL] for more info");
    }

    #[test]
    fn test_replace_emails() {
        let text = "Contact us at support@example.com for help";
        let replaced = replace_emails(text, "[EMAIL]");
        assert_eq!(replaced, "Contact us at [EMAIL] for help");
    }

    #[test]
    fn test_expand_contractions() {
        let text = "I can't believe it's working! They'll be happy.";
        let expanded = expand_contractions(text);
        assert_eq!(
            expanded,
            "I cannot believe it is working! They will be happy."
        );
    }

    #[test]
    fn test_normalize_whitespace() {
        let text = "  Hello   world  \n\t  test  ";
        let normalized = normalize_whitespace(text);
        assert_eq!(normalized, "Hello world test");
    }

    #[test]
    fn test_remove_accents() {
        let text = "Héllo wörld café";
        let cleaned = remove_accents(text);
        assert_eq!(cleaned, "Hello world cafe");
    }

    #[test]
    fn test_advanced_cleaner() {
        let cleaner = AdvancedTextCleaner::new();
        let text = "<p>Check out https://example.com! Email: test@example.com</p>";
        let cleaned = cleaner.clean(text).unwrap();
        assert_eq!(cleaned, "check out [url]! email: [email]");
    }

    #[test]
    fn test_privacy_focused_cleaner() {
        let cleaner = AdvancedTextCleaner::privacy_focused();
        let text = "Call me at (555) 123-4567 or email john@example.com";
        let cleaned = cleaner.clean(text).unwrap();
        assert_eq!(cleaned, "call me at [phone] or email [email]");
    }
}
