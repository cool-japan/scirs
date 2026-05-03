//! # Text Preprocessing Pipeline
//!
//! A configurable text preprocessing pipeline with the following capabilities:
//!
//! - **HTML tag removal**: Strip HTML/XML tags from text
//! - **URL/email/mention extraction**: Detect and extract or remove URLs, emails, @mentions
//! - **Number normalization**: Convert numbers to tokens or normalized forms
//! - **Contraction expansion**: Expand English contractions (e.g., "can't" -> "cannot")
//! - **Spell checking**: Edit-distance-based spelling correction with dictionary lookup
//! - **Text normalization**: Unicode normalization, diacritics removal
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::text_preprocess::{TextPreprocessor, PreprocessConfig};
//!
//! let config = PreprocessConfig::default();
//! let preprocessor = TextPreprocessor::new(config);
//!
//! let text = "<p>I can't believe it's only $9.99!</p>";
//! let result = preprocessor.process(text).unwrap();
//! assert!(!result.text.contains("<p>"));
//! ```

use crate::error::{Result, TextError};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the text preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Remove HTML/XML tags.
    pub strip_html: bool,
    /// Remove or replace URLs.
    pub handle_urls: UrlHandling,
    /// Remove or replace email addresses.
    pub handle_emails: EmailHandling,
    /// Remove or replace @mentions.
    pub handle_mentions: MentionHandling,
    /// Normalize numbers.
    pub normalize_numbers: bool,
    /// Number replacement token.
    pub number_token: String,
    /// Expand contractions.
    pub expand_contractions: bool,
    /// Enable spell checking.
    pub spell_check: bool,
    /// Maximum edit distance for spell checking.
    pub max_edit_distance: usize,
    /// Remove diacritics/accents.
    pub remove_diacritics: bool,
    /// Perform unicode normalization (NFC).
    pub unicode_normalize: bool,
    /// Convert to lowercase.
    pub lowercase: bool,
    /// Remove extra whitespace.
    pub normalize_whitespace: bool,
    /// Remove punctuation.
    pub remove_punctuation: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            strip_html: true,
            handle_urls: UrlHandling::Remove,
            handle_emails: EmailHandling::Remove,
            handle_mentions: MentionHandling::Remove,
            normalize_numbers: false,
            number_token: "<NUM>".to_string(),
            expand_contractions: true,
            spell_check: false,
            max_edit_distance: 2,
            remove_diacritics: false,
            unicode_normalize: true,
            lowercase: false,
            normalize_whitespace: true,
            remove_punctuation: false,
        }
    }
}

/// How to handle URLs.
#[derive(Debug, Clone, PartialEq)]
pub enum UrlHandling {
    /// Leave URLs as-is.
    Keep,
    /// Remove URLs entirely.
    Remove,
    /// Replace with a token.
    Replace(String),
}

/// How to handle email addresses.
#[derive(Debug, Clone, PartialEq)]
pub enum EmailHandling {
    /// Leave emails as-is.
    Keep,
    /// Remove emails entirely.
    Remove,
    /// Replace with a token.
    Replace(String),
}

/// How to handle @mentions.
#[derive(Debug, Clone, PartialEq)]
pub enum MentionHandling {
    /// Leave mentions as-is.
    Keep,
    /// Remove mentions entirely.
    Remove,
    /// Replace with a token.
    Replace(String),
}

// ---------------------------------------------------------------------------
// Preprocessing result
// ---------------------------------------------------------------------------

/// Result of text preprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// The preprocessed text.
    pub text: String,
    /// URLs extracted from the text.
    pub extracted_urls: Vec<String>,
    /// Email addresses extracted from the text.
    pub extracted_emails: Vec<String>,
    /// @mentions extracted from the text.
    pub extracted_mentions: Vec<String>,
    /// Numbers found in the text.
    pub extracted_numbers: Vec<String>,
    /// Spelling corrections made (original -> corrected).
    pub spelling_corrections: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Text Preprocessor
// ---------------------------------------------------------------------------

/// The main text preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    config: PreprocessConfig,
    /// Spell check dictionary.
    dictionary: HashSet<String>,
    /// Contraction mapping.
    contractions: HashMap<String, String>,
}

impl TextPreprocessor {
    /// Create a new preprocessor with the given configuration.
    pub fn new(config: PreprocessConfig) -> Self {
        let contractions = build_contraction_map();
        Self {
            config,
            dictionary: HashSet::new(),
            contractions,
        }
    }

    /// Set a custom dictionary for spell checking.
    pub fn with_dictionary(mut self, words: impl IntoIterator<Item = String>) -> Self {
        self.dictionary = words.into_iter().collect();
        self
    }

    /// Add words to the spell checking dictionary.
    pub fn add_dictionary_words(&mut self, words: impl IntoIterator<Item = String>) {
        self.dictionary.extend(words);
    }

    /// Load a basic English dictionary (common words).
    pub fn with_basic_dictionary(mut self) -> Self {
        self.dictionary = build_basic_dictionary();
        self
    }

    /// Process a text through the preprocessing pipeline.
    pub fn process(&self, text: &str) -> Result<PreprocessResult> {
        let mut result = PreprocessResult {
            text: text.to_string(),
            extracted_urls: Vec::new(),
            extracted_emails: Vec::new(),
            extracted_mentions: Vec::new(),
            extracted_numbers: Vec::new(),
            spelling_corrections: Vec::new(),
        };

        // 1. Unicode normalization
        if self.config.unicode_normalize {
            result.text = unicode_nfc_normalize(&result.text);
        }

        // 2. Strip HTML
        if self.config.strip_html {
            result.text = strip_html_tags(&result.text);
        }

        // 3. Handle URLs
        let (text_after_urls, urls) =
            extract_and_handle_urls(&result.text, &self.config.handle_urls);
        result.text = text_after_urls;
        result.extracted_urls = urls;

        // 4. Handle emails
        let (text_after_emails, emails) =
            extract_and_handle_emails(&result.text, &self.config.handle_emails);
        result.text = text_after_emails;
        result.extracted_emails = emails;

        // 5. Handle mentions
        let (text_after_mentions, mentions) =
            extract_and_handle_mentions(&result.text, &self.config.handle_mentions);
        result.text = text_after_mentions;
        result.extracted_mentions = mentions;

        // 6. Expand contractions
        if self.config.expand_contractions {
            result.text = self.expand_contractions_text(&result.text);
        }

        // 7. Normalize numbers
        if self.config.normalize_numbers {
            let (text, numbers) = normalize_numbers(&result.text, &self.config.number_token);
            result.text = text;
            result.extracted_numbers = numbers;
        }

        // 8. Remove diacritics
        if self.config.remove_diacritics {
            result.text = remove_diacritics_from_text(&result.text);
        }

        // 9. Lowercase
        if self.config.lowercase {
            result.text = result.text.to_lowercase();
        }

        // 10. Remove punctuation
        if self.config.remove_punctuation {
            result.text = remove_punctuation(&result.text);
        }

        // 11. Spell check
        if self.config.spell_check && !self.dictionary.is_empty() {
            let (text, corrections) =
                self.spell_check_text(&result.text, self.config.max_edit_distance);
            result.text = text;
            result.spelling_corrections = corrections;
        }

        // 12. Normalize whitespace (always last for clean output)
        if self.config.normalize_whitespace {
            result.text = normalize_whitespace(&result.text);
        }

        Ok(result)
    }

    /// Expand contractions in text.
    fn expand_contractions_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Sort contractions by length descending to match longer ones first
        let mut sorted_contractions: Vec<(&String, &String)> = self.contractions.iter().collect();
        sorted_contractions.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));

        for (contraction, expansion) in &sorted_contractions {
            // Case-insensitive replacement
            let lower = result.to_lowercase();
            let contraction_lower = contraction.to_lowercase();
            let mut new_result = String::with_capacity(result.len());
            let mut search_from = 0;

            loop {
                let lower_slice = &lower[search_from..];
                match lower_slice.find(&contraction_lower) {
                    Some(pos) => {
                        new_result.push_str(&result[search_from..search_from + pos]);
                        new_result.push_str(expansion);
                        search_from += pos + contraction.len();
                    }
                    None => {
                        new_result.push_str(&result[search_from..]);
                        break;
                    }
                }
            }
            result = new_result;
        }
        result
    }

    /// Spell check text and return corrected text with corrections list.
    fn spell_check_text(&self, text: &str, max_distance: usize) -> (String, Vec<(String, String)>) {
        let mut corrections = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result_words = Vec::with_capacity(words.len());

        for word in &words {
            let clean_word = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();

            if clean_word.is_empty() || self.dictionary.contains(&clean_word) {
                result_words.push(word.to_string());
                continue;
            }

            // Find closest dictionary word
            if let Some(correction) = find_closest_word(&clean_word, &self.dictionary, max_distance)
            {
                corrections.push((clean_word.clone(), correction.clone()));
                // Preserve original casing pattern for the replacement
                let corrected = transfer_casing(word, &correction);
                result_words.push(corrected);
            } else {
                result_words.push(word.to_string());
            }
        }

        (result_words.join(" "), corrections)
    }
}

// ---------------------------------------------------------------------------
// HTML stripping
// ---------------------------------------------------------------------------

/// Remove HTML/XML tags from text.
///
/// Handles self-closing tags, attributes, and HTML entities.
pub fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '<' {
            in_tag = true;
            i += 1;
            continue;
        }
        if chars[i] == '>' && in_tag {
            in_tag = false;
            i += 1;
            continue;
        }
        if !in_tag {
            // Handle HTML entities
            if chars[i] == '&' {
                if let Some(entity_result) = try_decode_entity(&chars, i) {
                    result.push(entity_result.0);
                    i = entity_result.1;
                    continue;
                }
            }
            result.push(chars[i]);
        }
        i += 1;
    }
    result
}

/// Try to decode an HTML entity starting at position `start`.
/// Returns (decoded_char, next_position) or None.
fn try_decode_entity(chars: &[char], start: usize) -> Option<(char, usize)> {
    // Find the semicolon
    let mut end = start + 1;
    while end < chars.len() && end - start < 10 {
        if chars[end] == ';' {
            let entity: String = chars[start..=end].iter().collect();
            let decoded = match entity.as_str() {
                "&amp;" => Some('&'),
                "&lt;" => Some('<'),
                "&gt;" => Some('>'),
                "&quot;" => Some('"'),
                "&apos;" => Some('\''),
                "&nbsp;" => Some(' '),
                _ => {
                    // Numeric entities
                    if entity.starts_with("&#x") || entity.starts_with("&#X") {
                        let hex_str: String = entity[3..entity.len() - 1].to_string();
                        u32::from_str_radix(&hex_str, 16)
                            .ok()
                            .and_then(char::from_u32)
                    } else if entity.starts_with("&#") {
                        let num_str: String = entity[2..entity.len() - 1].to_string();
                        num_str.parse::<u32>().ok().and_then(char::from_u32)
                    } else {
                        None
                    }
                }
            };
            if let Some(c) = decoded {
                return Some((c, end + 1));
            }
            return None;
        }
        end += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// URL extraction and handling
// ---------------------------------------------------------------------------

/// Extract and handle URLs in text.
fn extract_and_handle_urls(text: &str, handling: &UrlHandling) -> (String, Vec<String>) {
    let mut urls = Vec::new();

    match handling {
        UrlHandling::Keep => (text.to_string(), urls),
        UrlHandling::Remove | UrlHandling::Replace(_) => {
            let replacement = match handling {
                UrlHandling::Replace(token) => token.as_str(),
                _ => "",
            };
            let result =
                replace_pattern_simple(text, is_url_start, find_url_end, replacement, &mut urls);
            (result, urls)
        }
    }
}

/// Check if a URL starts at this position.
fn is_url_start(text: &str, pos: usize) -> bool {
    let remaining = &text[pos..];
    remaining.starts_with("http://")
        || remaining.starts_with("https://")
        || remaining.starts_with("ftp://")
        || remaining.starts_with("www.")
}

/// Find end of URL starting at `start`.
fn find_url_end(text: &str, start: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = start;
    while end < bytes.len() {
        let b = bytes[end];
        // URL ends at whitespace, or certain punctuation at end
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == b'>' || b == b'"' {
            break;
        }
        end += 1;
    }
    // Trim trailing punctuation that's unlikely part of URL
    while end > start {
        let b = bytes[end - 1];
        if b == b'.'
            || b == b','
            || b == b')'
            || b == b']'
            || b == b';'
            || b == b':'
            || b == b'!'
            || b == b'?'
        {
            end -= 1;
        } else {
            break;
        }
    }
    end
}

// ---------------------------------------------------------------------------
// Email extraction and handling
// ---------------------------------------------------------------------------

/// Extract and handle email addresses.
fn extract_and_handle_emails(text: &str, handling: &EmailHandling) -> (String, Vec<String>) {
    let mut emails = Vec::new();

    match handling {
        EmailHandling::Keep => (text.to_string(), emails),
        EmailHandling::Remove | EmailHandling::Replace(_) => {
            let replacement = match handling {
                EmailHandling::Replace(token) => token.as_str(),
                _ => "",
            };
            let result = find_and_replace_emails(text, replacement, &mut emails);
            (result, emails)
        }
    }
}

/// Find and replace email addresses in text.
fn find_and_replace_emails(text: &str, replacement: &str, extracted: &mut Vec<String>) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Look for @ sign and work backwards/forwards
        if chars[i] == '@' && i > 0 {
            // Find local part (before @)
            let mut local_start = i;
            while local_start > 0 {
                let c = chars[local_start - 1];
                if c.is_alphanumeric() || c == '.' || c == '_' || c == '+' || c == '-' || c == '%' {
                    local_start -= 1;
                } else {
                    break;
                }
            }

            // Find domain part (after @)
            let mut domain_end = i + 1;
            let mut has_dot = false;
            while domain_end < chars.len() {
                let c = chars[domain_end];
                if c.is_alphanumeric() || c == '.' || c == '-' {
                    if c == '.' {
                        has_dot = true;
                    }
                    domain_end += 1;
                } else {
                    break;
                }
            }

            if local_start < i && domain_end > i + 1 && has_dot {
                let email: String = chars[local_start..domain_end].iter().collect();
                extracted.push(email);

                // Remove what we already wrote for the local part
                let already_written = i - local_start;
                for _ in 0..already_written {
                    result.pop();
                }

                result.push_str(replacement);
                i = domain_end;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Mention extraction and handling
// ---------------------------------------------------------------------------

/// Extract and handle @mentions.
fn extract_and_handle_mentions(text: &str, handling: &MentionHandling) -> (String, Vec<String>) {
    let mut mentions = Vec::new();

    match handling {
        MentionHandling::Keep => (text.to_string(), mentions),
        MentionHandling::Remove | MentionHandling::Replace(_) => {
            let replacement = match handling {
                MentionHandling::Replace(token) => token.as_str(),
                _ => "",
            };
            let result = find_and_replace_mentions(text, replacement, &mut mentions);
            (result, mentions)
        }
    }
}

/// Find and replace @mentions.
fn find_and_replace_mentions(text: &str, replacement: &str, extracted: &mut Vec<String>) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '@' {
            // Check that @ is preceded by whitespace or start of string
            let preceded_by_space = i == 0 || chars[i - 1].is_whitespace();
            if preceded_by_space {
                let mut end = i + 1;
                while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
                    end += 1;
                }
                if end > i + 1 {
                    let mention: String = chars[i..end].iter().collect();
                    extracted.push(mention);
                    result.push_str(replacement);
                    i = end;
                    continue;
                }
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Number normalization
// ---------------------------------------------------------------------------

/// Normalize numbers in text by replacing them with a token.
fn normalize_numbers(text: &str, token: &str) -> (String, Vec<String>) {
    let mut numbers = Vec::new();
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i].is_ascii_digit()
            || (chars[i] == '-'
                && i + 1 < chars.len()
                && chars[i + 1].is_ascii_digit()
                && (i == 0 || chars[i - 1].is_whitespace()))
        {
            let start = i;
            if chars[i] == '-' {
                i += 1;
            }
            // Consume digits with optional commas and decimal point
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            // Check for comma-separated groups
            while i + 1 < chars.len() && chars[i] == ',' && chars[i + 1].is_ascii_digit() {
                i += 1; // skip comma
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            // Decimal part
            if i < chars.len()
                && chars[i] == '.'
                && i + 1 < chars.len()
                && chars[i + 1].is_ascii_digit()
            {
                i += 1; // skip dot
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            // Scientific notation
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let save = i;
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                if i < chars.len() && chars[i].is_ascii_digit() {
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save; // not valid scientific notation
                }
            }

            let num: String = chars[start..i].iter().collect();
            numbers.push(num);
            result.push_str(token);
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    (result, numbers)
}

// ---------------------------------------------------------------------------
// Diacritics removal
// ---------------------------------------------------------------------------

/// Remove diacritics/accents from text.
///
/// Uses Unicode decomposition to separate base characters from combining marks.
pub fn remove_diacritics_from_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    text.nfd().filter(|c| !is_combining_mark(*c)).collect()
}

/// Check if a character is a Unicode combining mark.
fn is_combining_mark(c: char) -> bool {
    let code = c as u32;
    // Combining Diacritical Marks: U+0300 to U+036F
    // Combining Diacritical Marks Extended: U+1AB0 to U+1AFF
    // Combining Diacritical Marks Supplement: U+1DC0 to U+1DFF
    // Combining Half Marks: U+FE20 to U+FE2F
    (0x0300..=0x036F).contains(&code)
        || (0x1AB0..=0x1AFF).contains(&code)
        || (0x1DC0..=0x1DFF).contains(&code)
        || (0xFE20..=0xFE2F).contains(&code)
}

// ---------------------------------------------------------------------------
// Unicode normalization
// ---------------------------------------------------------------------------

/// Apply Unicode NFC normalization.
fn unicode_nfc_normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfc().collect()
}

// ---------------------------------------------------------------------------
// Whitespace normalization
// ---------------------------------------------------------------------------

/// Normalize whitespace: collapse multiple spaces, trim.
pub fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_space = true; // true to trim leading

    for c in text.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(c);
            last_was_space = false;
        }
    }

    // Trim trailing space
    if result.ends_with(' ') {
        result.pop();
    }
    result
}

// ---------------------------------------------------------------------------
// Punctuation removal
// ---------------------------------------------------------------------------

/// Remove punctuation from text.
fn remove_punctuation(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect()
}

// ---------------------------------------------------------------------------
// Spell checking
// ---------------------------------------------------------------------------

/// Compute edit distance (Levenshtein) between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Find the closest word in the dictionary within max_distance.
fn find_closest_word(
    word: &str,
    dictionary: &HashSet<String>,
    max_distance: usize,
) -> Option<String> {
    let mut best: Option<(String, usize)> = None;

    for dict_word in dictionary {
        // Quick length filter
        let len_diff = if word.len() > dict_word.len() {
            word.len() - dict_word.len()
        } else {
            dict_word.len() - word.len()
        };
        if len_diff > max_distance {
            continue;
        }

        let dist = edit_distance(word, dict_word);
        if dist <= max_distance {
            match &best {
                None => best = Some((dict_word.clone(), dist)),
                Some((_, best_dist)) => {
                    if dist < *best_dist {
                        best = Some((dict_word.clone(), dist));
                    }
                }
            }
        }
    }

    best.map(|(w, _)| w)
}

/// Transfer casing pattern from source to target.
fn transfer_casing(source: &str, target: &str) -> String {
    let source_chars: Vec<char> = source.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();

    if source_chars.iter().all(|c| c.is_uppercase()) {
        return target.to_uppercase();
    }

    if source_chars
        .first()
        .map(|c| c.is_uppercase())
        .unwrap_or(false)
    {
        let mut result: String = target_chars
            .first()
            .map(|c| c.to_uppercase().to_string())
            .unwrap_or_default();
        for &c in &target_chars[1..] {
            result.push(c);
        }
        return result;
    }

    target.to_string()
}

// ---------------------------------------------------------------------------
// Contraction map
// ---------------------------------------------------------------------------

/// Build the English contraction mapping.
fn build_contraction_map() -> HashMap<String, String> {
    let mut m = HashMap::new();
    let pairs = [
        ("can't", "cannot"),
        ("won't", "will not"),
        ("don't", "do not"),
        ("doesn't", "does not"),
        ("didn't", "did not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("hasn't", "has not"),
        ("haven't", "have not"),
        ("hadn't", "had not"),
        ("wouldn't", "would not"),
        ("couldn't", "could not"),
        ("shouldn't", "should not"),
        ("mustn't", "must not"),
        ("needn't", "need not"),
        ("shan't", "shall not"),
        ("mightn't", "might not"),
        ("it's", "it is"),
        ("that's", "that is"),
        ("what's", "what is"),
        ("where's", "where is"),
        ("who's", "who is"),
        ("there's", "there is"),
        ("here's", "here is"),
        ("let's", "let us"),
        ("i'm", "i am"),
        ("you're", "you are"),
        ("we're", "we are"),
        ("they're", "they are"),
        ("i've", "i have"),
        ("you've", "you have"),
        ("we've", "we have"),
        ("they've", "they have"),
        ("i'll", "i will"),
        ("you'll", "you will"),
        ("he'll", "he will"),
        ("she'll", "she will"),
        ("it'll", "it will"),
        ("we'll", "we will"),
        ("they'll", "they will"),
        ("i'd", "i would"),
        ("you'd", "you would"),
        ("he'd", "he would"),
        ("she'd", "she would"),
        ("we'd", "we would"),
        ("they'd", "they would"),
    ];
    for (contraction, expansion) in &pairs {
        m.insert(contraction.to_string(), expansion.to_string());
    }
    m
}

// ---------------------------------------------------------------------------
// Basic dictionary
// ---------------------------------------------------------------------------

/// Build a basic English dictionary of common words.
fn build_basic_dictionary() -> HashSet<String> {
    let words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "great",
        "world",
        "very",
        "much",
        "been",
        "hello",
        "world",
        "computer",
        "science",
        "data",
        "machine",
        "learning",
        "algorithm",
        "programming",
        "software",
        "system",
        "network",
        "internet",
        "technology",
        "digital",
        "information",
        "process",
        "language",
        "text",
    ];
    words.iter().map(|w| w.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Generic pattern replacement helper
// ---------------------------------------------------------------------------

/// Replace patterns in text identified by start/end detector functions.
fn replace_pattern_simple(
    text: &str,
    is_start: fn(&str, usize) -> bool,
    find_end: fn(&str, usize) -> usize,
    replacement: &str,
    extracted: &mut Vec<String>,
) -> String {
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if is_start(text, i) {
            let end = find_end(text, i);
            if end > i {
                extracted.push(text[i..end].to_string());
                result.push_str(replacement);
                i = end;
                continue;
            }
        }
        // Push one UTF-8 character
        let c = text[i..].chars().next().unwrap_or(' ');
        result.push(c);
        i += c.len_utf8();
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_basic() {
        assert_eq!(strip_html_tags("<p>hello</p>"), "hello");
        assert_eq!(strip_html_tags("<b>bold</b> text"), "bold text");
    }

    #[test]
    fn test_strip_html_nested() {
        assert_eq!(strip_html_tags("<div><p>nested</p></div>"), "nested");
    }

    #[test]
    fn test_strip_html_entities() {
        assert_eq!(strip_html_tags("a &amp; b"), "a & b");
        assert_eq!(strip_html_tags("a &lt; b"), "a < b");
        assert_eq!(strip_html_tags("a &gt; b"), "a > b");
    }

    #[test]
    fn test_strip_html_no_tags() {
        assert_eq!(strip_html_tags("no tags here"), "no tags here");
    }

    #[test]
    fn test_url_detection() {
        let (text, urls) =
            extract_and_handle_urls("visit https://example.com for info", &UrlHandling::Remove);
        assert!(!text.contains("https://"));
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://example.com");
    }

    #[test]
    fn test_url_replacement() {
        let (text, urls) = extract_and_handle_urls(
            "check https://example.com now",
            &UrlHandling::Replace("<URL>".to_string()),
        );
        assert!(text.contains("<URL>"));
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn test_url_keep() {
        let (text, urls) = extract_and_handle_urls("see https://example.com", &UrlHandling::Keep);
        assert!(text.contains("https://example.com"));
        assert!(urls.is_empty());
    }

    #[test]
    fn test_email_detection() {
        let (text, emails) =
            extract_and_handle_emails("contact user@example.com for help", &EmailHandling::Remove);
        assert!(!text.contains("user@example.com"));
        assert_eq!(emails.len(), 1);
        assert_eq!(emails[0], "user@example.com");
    }

    #[test]
    fn test_mention_detection() {
        let (text, mentions) =
            extract_and_handle_mentions("hello @user123 how are you", &MentionHandling::Remove);
        assert!(!text.contains("@user123"));
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0], "@user123");
    }

    #[test]
    fn test_mention_replacement() {
        let (text, _) = extract_and_handle_mentions(
            "hi @alice and @bob",
            &MentionHandling::Replace("<MENTION>".to_string()),
        );
        assert!(text.contains("<MENTION>"));
        assert!(!text.contains("@alice"));
    }

    #[test]
    fn test_number_normalization() {
        let (text, numbers) = normalize_numbers("I have 42 cats and 3.14 dogs", "<NUM>");
        assert!(text.contains("<NUM>"));
        assert_eq!(numbers.len(), 2);
        assert!(numbers.contains(&"42".to_string()));
        assert!(numbers.contains(&"3.14".to_string()));
    }

    #[test]
    fn test_number_with_commas() {
        let (text, numbers) = normalize_numbers("population: 1,234,567", "<NUM>");
        assert!(text.contains("<NUM>"));
        assert_eq!(numbers.len(), 1);
    }

    #[test]
    fn test_contraction_expansion() {
        let preprocessor = TextPreprocessor::new(PreprocessConfig {
            strip_html: false,
            expand_contractions: true,
            ..Default::default()
        });
        let result = preprocessor
            .process("I can't do this")
            .expect("process failed");
        assert!(result.text.contains("cannot"));
    }

    #[test]
    fn test_contraction_wont() {
        let preprocessor = TextPreprocessor::new(PreprocessConfig {
            strip_html: false,
            expand_contractions: true,
            ..Default::default()
        });
        let result = preprocessor.process("I won't go").expect("process failed");
        assert!(result.text.contains("will not"));
    }

    #[test]
    fn test_diacritics_removal() {
        let result = remove_diacritics_from_text("cafe\u{0301}"); // café
        assert_eq!(result, "cafe");
    }

    #[test]
    fn test_diacritics_spanish() {
        let result = remove_diacritics_from_text("ni\u{00f1}o"); // niño
        assert_eq!(result, "nino");
    }

    #[test]
    fn test_whitespace_normalization() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("a\t\nb"), "a b");
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("abc", ""), 3);
    }

    #[test]
    fn test_spell_check() {
        let mut dictionary = HashSet::new();
        dictionary.insert("hello".to_string());
        dictionary.insert("world".to_string());
        dictionary.insert("computer".to_string());

        let closest = find_closest_word("helo", &dictionary, 2);
        assert_eq!(closest, Some("hello".to_string()));
    }

    #[test]
    fn test_spell_check_no_match() {
        let mut dictionary = HashSet::new();
        dictionary.insert("hello".to_string());

        let closest = find_closest_word("zzzzz", &dictionary, 1);
        assert!(closest.is_none());
    }

    #[test]
    fn test_full_pipeline() {
        let config = PreprocessConfig {
            strip_html: true,
            handle_urls: UrlHandling::Replace("<URL>".to_string()),
            handle_emails: EmailHandling::Replace("<EMAIL>".to_string()),
            handle_mentions: MentionHandling::Replace("<MENTION>".to_string()),
            normalize_numbers: true,
            expand_contractions: true,
            unicode_normalize: true,
            normalize_whitespace: true,
            ..Default::default()
        };

        let preprocessor = TextPreprocessor::new(config);
        let text = "<p>I can't believe https://example.com has @user with 42 items!</p>";
        let result = preprocessor.process(text).expect("process failed");

        assert!(!result.text.contains("<p>"));
        assert!(result.text.contains("cannot"));
        assert!(result.text.contains("<URL>"));
        assert!(result.text.contains("<MENTION>"));
        assert!(!result.extracted_urls.is_empty());
        assert!(!result.extracted_mentions.is_empty());
    }

    #[test]
    fn test_pipeline_defaults() {
        let preprocessor = TextPreprocessor::new(PreprocessConfig::default());
        let result = preprocessor.process("Hello World").expect("process failed");
        assert_eq!(result.text, "Hello World");
    }

    #[test]
    fn test_punctuation_removal() {
        let text = remove_punctuation("Hello, world! How are you?");
        assert!(!text.contains(','));
        assert!(!text.contains('!'));
        assert!(!text.contains('?'));
    }

    #[test]
    fn test_transfer_casing() {
        assert_eq!(transfer_casing("Hello", "world"), "World");
        assert_eq!(transfer_casing("HELLO", "world"), "WORLD");
        assert_eq!(transfer_casing("hello", "WORLD"), "WORLD");
    }

    #[test]
    fn test_basic_dictionary() {
        let dict = build_basic_dictionary();
        assert!(dict.contains("the"));
        assert!(dict.contains("hello"));
    }

    #[test]
    fn test_spell_check_integration() {
        let config = PreprocessConfig {
            strip_html: false,
            expand_contractions: false,
            spell_check: true,
            max_edit_distance: 2,
            normalize_whitespace: true,
            ..Default::default()
        };

        let preprocessor = TextPreprocessor::new(config).with_basic_dictionary();
        let result = preprocessor.process("helo wrld").expect("process failed");
        // Should correct "helo" -> "hello" and "wrld" -> "world"
        assert!(!result.spelling_corrections.is_empty());
    }

    #[test]
    fn test_numeric_entity_decode() {
        assert_eq!(strip_html_tags("&#65;"), "A");
        assert_eq!(strip_html_tags("&#x41;"), "A");
    }

    #[test]
    fn test_empty_input() {
        let preprocessor = TextPreprocessor::new(PreprocessConfig::default());
        let result = preprocessor.process("").expect("process failed");
        assert_eq!(result.text, "");
    }

    #[test]
    fn test_multiple_urls() {
        let (text, urls) =
            extract_and_handle_urls("see https://a.com and https://b.com", &UrlHandling::Remove);
        assert_eq!(urls.len(), 2);
        assert!(!text.contains("https://"));
    }

    #[test]
    fn test_lowercase() {
        let config = PreprocessConfig {
            strip_html: false,
            expand_contractions: false,
            lowercase: true,
            ..Default::default()
        };
        let preprocessor = TextPreprocessor::new(config);
        let result = preprocessor.process("Hello WORLD").expect("process failed");
        assert_eq!(result.text, "hello world");
    }

    #[test]
    fn test_scientific_notation() {
        let (text, numbers) = normalize_numbers("value is 1.5e10 and 2E-3", "<NUM>");
        assert_eq!(numbers.len(), 2);
        assert!(text.contains("<NUM>"));
    }

    #[test]
    fn test_negative_numbers() {
        let (text, numbers) = normalize_numbers("temperature: -42 degrees", "<NUM>");
        assert!(numbers.contains(&"-42".to_string()));
        assert!(text.contains("<NUM>"));
    }

    #[test]
    fn test_html_self_closing() {
        assert_eq!(strip_html_tags("before<br/>after"), "beforeafter");
        assert_eq!(strip_html_tags("a<img src='x'/>b"), "ab");
    }

    #[test]
    fn test_email_no_email() {
        let (text, emails) = extract_and_handle_emails("no email here", &EmailHandling::Remove);
        assert_eq!(text, "no email here");
        assert!(emails.is_empty());
    }

    #[test]
    fn test_mention_not_at_word_boundary() {
        // @ in the middle of a word should not be treated as mention
        let (text, mentions) =
            extract_and_handle_mentions("test@notamention", &MentionHandling::Remove);
        // Since @ is not preceded by whitespace or start, should keep as-is
        assert!(mentions.is_empty());
        assert!(text.contains("test@notamention"));
    }
}
