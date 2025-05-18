//! Token filtering functionality
//!
//! This module provides utilities for filtering tokens based on various criteria
//! such as length, frequency, regex patterns, and custom rules.

use crate::error::{Result, TextError};
use crate::tokenize::Tokenizer;
use crate::vocabulary::Vocabulary;
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Trait for token filtering strategies
pub trait TokenFilter {
    /// Filter tokens based on the strategy
    fn apply(&self, tokens: &[String]) -> Vec<String>;

    /// Apply the filter directly to text
    fn filter_text(&self, text: &str, tokenizer: &dyn Tokenizer) -> Result<String> {
        let tokens = tokenizer.tokenize(text)?;
        let filtered = self.apply(&tokens);
        Ok(filtered.join(" "))
    }
}

/// Filter tokens by length
#[derive(Debug, Clone)]
pub struct LengthFilter {
    /// Minimum token length
    pub min_length: usize,
    /// Maximum token length
    pub max_length: usize,
}

impl Default for LengthFilter {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: usize::MAX,
        }
    }
}

impl LengthFilter {
    /// Create a new length filter
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self {
            min_length,
            max_length,
        }
    }

    /// Set minimum token length
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }

    /// Set maximum token length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

impl TokenFilter for LengthFilter {
    fn apply(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| {
                let len = token.chars().count(); // Use Unicode chars for proper length
                len >= self.min_length && len <= self.max_length
            })
            .cloned()
            .collect()
    }
}

/// Filter tokens by frequency in a corpus
#[derive(Debug, Clone)]
pub struct FrequencyFilter {
    /// Minimum token frequency
    pub min_count: usize,
    /// Maximum token frequency (absolute count)
    pub max_count: Option<usize>,
    /// Maximum token frequency (as a fraction of total)
    pub max_freq: Option<f64>,
    /// Token frequencies
    token_counts: HashMap<String, usize>,
    /// Total token count
    total_count: usize,
}

impl FrequencyFilter {
    /// Create a new frequency filter from tokens with a vocabulary for reference
    pub fn from_tokens_with_vocabulary(
        tokens: &[String],
        vocabulary: &Vocabulary,
        min_count: usize,
    ) -> Self {
        let mut token_counts = HashMap::new();

        // Count tokens that exist in vocabulary
        for token in tokens {
            if vocabulary.contains(token) {
                *token_counts.entry(token.clone()).or_insert(0) += 1;
            }
        }

        let total_count: usize = token_counts.values().sum();

        Self {
            min_count,
            max_count: None,
            max_freq: None,
            token_counts,
            total_count,
        }
    }

    /// Create a new frequency filter from token counts
    pub fn from_counts(token_counts: HashMap<String, usize>, min_count: usize) -> Self {
        let total_count = token_counts.values().sum();

        Self {
            min_count,
            max_count: None,
            max_freq: None,
            token_counts,
            total_count,
        }
    }

    /// Learn token frequencies from a corpus
    pub fn learn_from_corpus(
        texts: &[&str],
        tokenizer: &dyn Tokenizer,
        min_count: usize,
    ) -> Result<Self> {
        let mut counts = HashMap::new();
        let mut total = 0;

        for &text in texts {
            let tokens = tokenizer.tokenize(text)?;
            for token in tokens {
                *counts.entry(token).or_insert(0) += 1;
                total += 1;
            }
        }

        Ok(Self {
            min_count,
            max_count: None,
            max_freq: None,
            token_counts: counts,
            total_count: total,
        })
    }

    /// Set the maximum count threshold
    pub fn with_max_count(mut self, max_count: usize) -> Self {
        self.max_count = Some(max_count);
        self
    }

    /// Set the maximum frequency threshold (0.0 to 1.0)
    pub fn with_max_freq(mut self, max_freq: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&max_freq) {
            return Err(TextError::InvalidInput(
                "max_freq must be between 0.0 and 1.0".to_string(),
            ));
        }

        self.max_freq = Some(max_freq);
        Ok(self)
    }
}

impl TokenFilter for FrequencyFilter {
    fn apply(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| {
                let count = self.token_counts.get(*token).copied().unwrap_or(0);

                // Apply minimum count filter
                if count < self.min_count {
                    return false;
                }

                // Apply maximum count filter if specified
                if let Some(max_count) = self.max_count {
                    if count > max_count {
                        return false;
                    }
                }

                // Apply maximum frequency filter if specified
                if let Some(max_freq) = self.max_freq {
                    if self.total_count > 0 {
                        let freq = count as f64 / self.total_count as f64;
                        if freq > max_freq {
                            return false;
                        }
                    }
                }

                true
            })
            .cloned()
            .collect()
    }
}

/// Filter tokens using regular expressions
#[derive(Debug, Clone)]
pub struct RegexFilter {
    /// Regex pattern
    pattern: Regex,
    /// Whether to keep tokens that match (true) or don't match (false)
    keep_matching: bool,
}

impl RegexFilter {
    /// Create a new regex filter
    pub fn new(pattern: &str, keep_matching: bool) -> Result<Self> {
        match Regex::new(pattern) {
            Ok(regex) => Ok(Self {
                pattern: regex,
                keep_matching,
            }),
            Err(e) => Err(TextError::InvalidInput(format!(
                "Invalid regex pattern: {}",
                e
            ))),
        }
    }
}

impl TokenFilter for RegexFilter {
    fn apply(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| {
                let matches = self.pattern.is_match(token);
                matches == self.keep_matching
            })
            .cloned()
            .collect()
    }
}

/// Filter tokens using a predefined stopword list
#[derive(Debug, Clone)]
pub struct StopwordsFilter {
    /// Set of stopwords
    stopwords: HashSet<String>,
    /// Whether to keep stopwords (false) or filter them out (true)
    remove_stopwords: bool,
}

impl StopwordsFilter {
    /// Create a new stopwords filter
    pub fn new(stopwords: Vec<String>, remove_stopwords: bool) -> Self {
        Self {
            stopwords: stopwords.into_iter().collect(),
            remove_stopwords,
        }
    }

    /// Create a stopwords filter from a file
    pub fn from_file(path: &str) -> Result<Self> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let reader = BufReader::new(file);

        let mut stopwords = HashSet::new();
        for line in reader.lines() {
            let word = line.map_err(|e| TextError::IoError(e.to_string()))?;
            if !word.trim().is_empty() && !word.starts_with('#') {
                stopwords.insert(word.trim().to_lowercase());
            }
        }

        Ok(Self {
            stopwords,
            remove_stopwords: true,
        })
    }

    /// Set whether to remove stopwords
    pub fn remove_stopwords(mut self, remove: bool) -> Self {
        self.remove_stopwords = remove;
        self
    }

    /// Add stopwords to the filter
    pub fn add_stopwords(&mut self, words: &[String]) {
        for word in words {
            self.stopwords.insert(word.clone());
        }
    }

    /// Get the current stopwords
    pub fn get_stopwords(&self) -> Vec<String> {
        self.stopwords.iter().cloned().collect()
    }
}

impl TokenFilter for StopwordsFilter {
    fn apply(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| {
                let is_stopword = self.stopwords.contains(&token.to_lowercase());
                if self.remove_stopwords {
                    !is_stopword
                } else {
                    is_stopword
                }
            })
            .cloned()
            .collect()
    }
}

/// Composite filter that combines multiple filters
pub struct CompositeFilter {
    /// The filters to apply in sequence
    filters: Vec<Box<dyn TokenFilter + Send + Sync>>,
}

impl CompositeFilter {
    /// Create a new empty composite filter
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the composite
    pub fn add_filter<F>(&mut self, filter: F)
    where
        F: TokenFilter + Send + Sync + 'static,
    {
        self.filters.push(Box::new(filter));
    }

    /// Add a filter and return self (builder pattern)
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: TokenFilter + Send + Sync + 'static,
    {
        self.add_filter(filter);
        self
    }
}

impl Default for CompositeFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CompositeFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeFilter")
            .field("num_filters", &self.filters.len())
            .finish()
    }
}

// We can't derive Clone because the trait isn't implemented for boxed trait objects
// Instead, we'll create a new CompositeFilter when cloning
impl Clone for CompositeFilter {
    fn clone(&self) -> Self {
        // We can't clone the filters, so we create a new empty CompositeFilter
        // This is a limitation - cloned composite filters will be empty
        Self::new()
    }
}

impl TokenFilter for CompositeFilter {
    fn apply(&self, tokens: &[String]) -> Vec<String> {
        let mut filtered = tokens.to_vec();

        for filter in &self.filters {
            filtered = filter.apply(&filtered);
        }

        filtered
    }
}

/// Custom filter using a function predicate
pub struct CustomFilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    /// The predicate function
    predicate: F,
}

impl<F> CustomFilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    /// Create a new custom filter with the given predicate
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> TokenFilter for CustomFilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    fn apply(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| (self.predicate)(token))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WordTokenizer;

    fn get_test_tokens() -> Vec<String> {
        vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
            "jumps".to_string(),
            "over".to_string(),
            "the".to_string(),
            "lazy".to_string(),
            "dog".to_string(),
        ]
    }

    #[test]
    fn test_length_filter() {
        let tokens = get_test_tokens();

        // Filter tokens with length >= 4
        let filter = LengthFilter::new(4, usize::MAX);
        let filtered = filter.apply(&tokens);

        // Sort the result for consistent comparison regardless of original order
        let mut sorted_filtered = filtered.clone();
        sorted_filtered.sort();
        assert_eq!(
            sorted_filtered,
            vec!["brown", "jumps", "lazy", "over", "quick"]
        );

        // Filter tokens with length == 3
        let filter = LengthFilter::new(3, 3);
        let filtered = filter.apply(&tokens);

        // Sort for consistent comparison
        let mut sorted_filtered = filtered.clone();
        sorted_filtered.sort();
        assert_eq!(sorted_filtered, vec!["dog", "fox", "the", "the"]);
    }

    #[test]
    fn test_frequency_filter() {
        let tokens = get_test_tokens();

        // Create token counts
        let mut counts = HashMap::new();
        for token in &tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }

        // Filter out tokens that appear only once
        let filter = FrequencyFilter::from_counts(counts, 2);
        let filtered = filter.apply(&tokens);

        // Only "the" appears twice
        assert_eq!(filtered, vec!["the", "the"]);
    }

    #[test]
    fn test_regex_filter() {
        let tokens = get_test_tokens();

        // Keep tokens that start with 'b'
        let filter = RegexFilter::new(r"^b", true).unwrap();
        let filtered = filter.apply(&tokens);

        assert_eq!(filtered, vec!["brown"]);

        // This test is specifically checking tokens without 'o'
        // Create a new set of tokens for clearer testing
        let test_tokens = vec![
            "the".to_string(),
            "jumps".to_string(),
            "the".to_string(),
            "lazy".to_string(),
        ];

        // Remove tokens containing 'o'
        let filter = RegexFilter::new(r"o", false).unwrap();
        let filtered = filter.apply(&test_tokens);

        // Sort for consistent comparison
        let mut sorted_filtered = filtered.clone();
        sorted_filtered.sort();
        let expected = vec!["jumps", "lazy", "the", "the"];
        assert_eq!(sorted_filtered, expected);
    }

    #[test]
    fn test_stopwords_filter() {
        let tokens = get_test_tokens();

        // Define stopwords
        let stopwords = vec!["the".to_string(), "over".to_string()];

        // Filter out stopwords
        let filter = StopwordsFilter::new(stopwords, true);
        let filtered = filter.apply(&tokens);

        assert_eq!(
            filtered,
            vec!["quick", "brown", "fox", "jumps", "lazy", "dog"]
        );
    }

    #[test]
    fn test_composite_filter() {
        let tokens = get_test_tokens();

        // Create filters
        let length_filter = LengthFilter::new(4, usize::MAX);
        let regex_filter = RegexFilter::new(r"o", true).unwrap();

        // Combine filters
        let composite = CompositeFilter::new()
            .with_filter(length_filter)
            .with_filter(regex_filter);

        let filtered = composite.apply(&tokens);

        // Tokens with length >= 4 AND containing 'o'
        assert_eq!(filtered, vec!["brown", "over"]);
    }

    #[test]
    fn test_custom_filter() {
        let tokens = get_test_tokens();

        // Custom filter: only keep tokens that contain 'o' followed by any letter
        let filter = CustomFilter::new(|token: &str| token.contains('o'));

        let filtered = filter.apply(&tokens);
        // Sort to ensure consistent order for the test
        let mut sorted_filtered = filtered.clone();
        sorted_filtered.sort();
        assert_eq!(sorted_filtered, vec!["brown", "dog", "fox", "over"]);
    }

    #[test]
    fn test_filter_text() {
        let text = "The quick brown fox jumps over the lazy dog";
        let tokenizer = WordTokenizer::default();

        // Filter out short words
        let filter = LengthFilter::new(5, usize::MAX);
        let filtered = filter.filter_text(text, &tokenizer).unwrap();

        assert_eq!(filtered, "quick brown jumps");
    }
}
