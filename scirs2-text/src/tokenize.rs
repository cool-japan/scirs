//! Text tokenization utilities
//!
//! This module provides functionality for tokenizing text into
//! words, sentences, or characters.

pub mod bpe;

use crate::error::{Result, TextError};
use lazy_static::lazy_static;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

pub use bpe::{BpeConfig, BpeTokenizer, BpeVocabulary};

lazy_static! {
    static ref WORD_PATTERN: Regex = Regex::new(r"\b\w+\b").unwrap();
    static ref SENTENCE_PATTERN: Regex = Regex::new(r"[^.!?]+[.!?]").unwrap();
}

/// Trait for tokenizing text
pub trait Tokenizer {
    /// Tokenize the input text into tokens
    fn tokenize(&self, text: &str) -> Result<Vec<String>>;

    /// Tokenize batch of text
    fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<Vec<String>>> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }
}

/// Tokenizer for splitting text into words
#[derive(Debug, Clone)]
pub struct WordTokenizer {
    lowercase: bool,
    pattern: Option<Regex>,
}

impl WordTokenizer {
    /// Create a new word tokenizer
    pub fn new(lowercase: bool) -> Self {
        Self {
            lowercase,
            pattern: None,
        }
    }

    /// Create a new word tokenizer with a custom pattern
    pub fn with_pattern(lowercase: bool, pattern: &str) -> Result<Self> {
        match Regex::new(pattern) {
            Ok(regex) => Ok(Self {
                lowercase,
                pattern: Some(regex),
            }),
            Err(e) => Err(TextError::TokenizationError(format!(
                "Invalid regex pattern: {}",
                e
            ))),
        }
    }
}

impl Default for WordTokenizer {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let tokens = match &self.pattern {
            Some(pattern) => pattern
                .find_iter(&text)
                .map(|m| m.as_str().to_string())
                .collect(),
            None => WORD_PATTERN
                .find_iter(&text)
                .map(|m| m.as_str().to_string())
                .collect(),
        };

        Ok(tokens)
    }
}

/// Tokenizer for splitting text into sentences
#[derive(Debug, Clone)]
pub struct SentenceTokenizer {
    pattern: Option<Regex>,
}

impl SentenceTokenizer {
    /// Create a new sentence tokenizer
    pub fn new() -> Self {
        Self { pattern: None }
    }

    /// Create a new sentence tokenizer with a custom pattern
    pub fn with_pattern(pattern: &str) -> Result<Self> {
        match Regex::new(pattern) {
            Ok(regex) => Ok(Self {
                pattern: Some(regex),
            }),
            Err(e) => Err(TextError::TokenizationError(format!(
                "Invalid regex pattern: {}",
                e
            ))),
        }
    }
}

impl Default for SentenceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SentenceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let tokens = match &self.pattern {
            Some(pattern) => pattern
                .find_iter(text)
                .map(|m| m.as_str().trim().to_string())
                .collect(),
            None => SENTENCE_PATTERN
                .find_iter(text)
                .map(|m| m.as_str().trim().to_string())
                .collect(),
        };

        Ok(tokens)
    }
}

/// Tokenizer for splitting text into characters or grapheme clusters
#[derive(Debug, Clone)]
pub struct CharacterTokenizer {
    use_grapheme_clusters: bool,
}

impl CharacterTokenizer {
    /// Create a new character tokenizer
    pub fn new(use_grapheme_clusters: bool) -> Self {
        Self {
            use_grapheme_clusters,
        }
    }
}

impl Default for CharacterTokenizer {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Tokenizer for CharacterTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let tokens = if self.use_grapheme_clusters {
            text.graphemes(true).map(|g| g.to_string()).collect()
        } else {
            text.chars().map(|c| c.to_string()).collect()
        };

        Ok(tokens)
    }
}

/// Tokenizer for extracting n-grams from text
#[derive(Debug, Clone)]
pub struct NgramTokenizer {
    n: usize,
    min_n: usize,
    only_alphanumeric: bool,
    separator: String,
}

impl NgramTokenizer {
    /// Create a new n-gram tokenizer
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(TextError::TokenizationError(
                "N-gram size must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            n,
            min_n: n,
            only_alphanumeric: false,
            separator: " ".to_string(),
        })
    }

    /// Create an n-gram tokenizer with a range of n values
    pub fn with_range(min_n: usize, max_n: usize) -> Result<Self> {
        if min_n == 0 || max_n < min_n {
            return Err(TextError::TokenizationError(
                "Invalid n-gram range".to_string(),
            ));
        }

        Ok(Self {
            n: max_n,
            min_n,
            only_alphanumeric: false,
            separator: " ".to_string(),
        })
    }

    /// Set whether to only include alphanumeric tokens
    pub fn only_alphanumeric(mut self, value: bool) -> Self {
        self.only_alphanumeric = value;
        self
    }

    /// Set the separator for n-grams
    pub fn with_separator(mut self, separator: String) -> Self {
        self.separator = separator;
        self
    }

    /// Extract n-grams from a sequence of tokens
    fn extract_ngrams(&self, tokens: &[String], n: usize) -> Vec<String> {
        if tokens.len() < n {
            return Vec::new();
        }

        tokens
            .windows(n)
            .map(|window| window.join(&self.separator))
            .collect()
    }
}

impl Tokenizer for NgramTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        // First tokenize into words
        let word_tokenizer = WordTokenizer::new(true);
        let words = word_tokenizer.tokenize(text)?;

        let filtered_words = if self.only_alphanumeric {
            words
                .into_iter()
                .filter(|w| w.chars().all(|c| c.is_alphanumeric()))
                .collect()
        } else {
            words
        };

        let mut ngrams = Vec::new();

        // Extract n-grams for each n in the range
        for n in self.min_n..=self.n {
            ngrams.extend(self.extract_ngrams(&filtered_words, n));
        }

        Ok(ngrams)
    }
}

/// Regular expression based tokenizer
#[derive(Debug, Clone)]
pub struct RegexTokenizer {
    pattern: Regex,
    gaps: bool,
}

impl RegexTokenizer {
    /// Create a new regex tokenizer
    ///
    /// # Arguments
    /// * `pattern` - The regex pattern to use
    /// * `gaps` - If true, the pattern matches token separators. If false, it matches tokens.
    pub fn new(pattern: &str, gaps: bool) -> Result<Self> {
        match Regex::new(pattern) {
            Ok(regex) => Ok(Self {
                pattern: regex,
                gaps,
            }),
            Err(e) => Err(TextError::TokenizationError(format!(
                "Invalid regex pattern: {}",
                e
            ))),
        }
    }
}

impl Tokenizer for RegexTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let tokens = if self.gaps {
            // Pattern matches separators
            self.pattern
                .split(text)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            // Pattern matches tokens
            self.pattern
                .find_iter(text)
                .map(|m| m.as_str().to_string())
                .collect()
        };

        Ok(tokens)
    }
}

/// Whitespace tokenizer that splits on any whitespace character
#[derive(Debug, Clone)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        Ok(text.split_whitespace().map(|s| s.to_string()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_tokenizer() {
        let tokenizer = WordTokenizer::default();
        let text = "Hello, world! This is a test.";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_word_tokenizer_custom_pattern() {
        let tokenizer = WordTokenizer::with_pattern(false, r"\w+").unwrap();
        let text = "Hello, world! This is a test.";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["Hello", "world", "This", "is", "a", "test"]);
    }

    #[test]
    fn test_sentence_tokenizer() {
        let tokenizer = SentenceTokenizer::default();
        let text = "Hello, world! This is a test. How are you today?";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(
            tokens,
            vec!["Hello, world!", "This is a test.", "How are you today?"]
        );
    }

    #[test]
    fn test_character_tokenizer() {
        let tokenizer = CharacterTokenizer::new(false);
        let text = "Hello";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["H", "e", "l", "l", "o"]);
    }

    #[test]
    fn test_grapheme_tokenizer() {
        let tokenizer = CharacterTokenizer::default();
        let text = "café";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["c", "a", "f", "é"]);
    }

    #[test]
    fn test_ngram_tokenizer() {
        let tokenizer = NgramTokenizer::new(2).unwrap();
        let text = "hello world test";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["hello world", "world test"]);
    }

    #[test]
    fn test_ngram_tokenizer_range() {
        let tokenizer = NgramTokenizer::with_range(1, 2).unwrap();
        let text = "hello world";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["hello", "world", "hello world"]);
    }

    #[test]
    fn test_ngram_tokenizer_alphanumeric() {
        let tokenizer = NgramTokenizer::new(2).unwrap().only_alphanumeric(true);
        let text = "hello, world! test123";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["hello world", "world test123"]);
    }

    #[test]
    fn test_regex_tokenizer_matches() {
        let tokenizer = RegexTokenizer::new(r"\b\w+\b", false).unwrap();
        let text = "Hello, world! Test 123.";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["Hello", "world", "Test", "123"]);
    }

    #[test]
    fn test_regex_tokenizer_gaps() {
        let tokenizer = RegexTokenizer::new(r"\s*,\s*", true).unwrap();
        let text = "apple, banana, cherry";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let text = "hello   world\ttest\nline";
        let tokens = tokenizer.tokenize(text).unwrap();
        assert_eq!(tokens, vec!["hello", "world", "test", "line"]);
    }
}
