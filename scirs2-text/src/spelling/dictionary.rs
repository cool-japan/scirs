//! Dictionary-based spelling correction
//!
//! This module provides a dictionary-based approach to spelling correction
//! using string similarity metrics to find the closest match for a
//! potentially misspelled word.
//!
//! # Key Components
//!
//! - `DictionaryCorrector`: Main implementation of dictionary-based spelling correction
//! - `DictionaryCorrectorConfig`: Configuration options for the corrector
//!
//! # Example
//!
//! ```
//! use scirs2_text::spelling::{DictionaryCorrector, SpellingCorrector};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a dictionary-based spelling corrector with default settings
//! let corrector = DictionaryCorrector::default();
//!
//! // Correct a misspelled word
//! let corrected = corrector.correct("recieve")?;
//! assert_eq!(corrected, "receive");
//!
//! // Check if a word is correct
//! assert!(corrector.is_correct("computer"));
//! assert!(!corrector.is_correct("computre"));
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TextError};
use crate::string_metrics::{DamerauLevenshteinMetric, StringMetric};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use super::utils::{dictionary_contains, is_within_length_threshold, normalize_string};
use super::SpellingCorrector;

/// Configuration for the dictionary-based spelling corrector
#[derive(Debug, Clone)]
pub struct DictionaryCorrectorConfig {
    /// Maximum edit distance to consider for corrections
    pub max_edit_distance: usize,
    /// Whether to use case-sensitive matching
    pub case_sensitive: bool,
    /// Maximum number of suggestions to consider
    pub max_suggestions: usize,
    /// Minimum word frequency to consider for suggestions
    pub min_frequency: usize,
    /// Whether to prioritize suggestions by word frequency
    pub prioritize_by_frequency: bool,
}

impl Default for DictionaryCorrectorConfig {
    fn default() -> Self {
        Self {
            max_edit_distance: 2,
            case_sensitive: false,
            max_suggestions: 5,
            min_frequency: 1,
            prioritize_by_frequency: true,
        }
    }
}

/// Dictionary-based spelling corrector
pub struct DictionaryCorrector {
    /// Dictionary of words and their frequencies
    pub(crate) dictionary: HashMap<String, usize>,
    /// Configuration for the corrector
    pub(crate) config: DictionaryCorrectorConfig,
    /// Metric to use for string similarity
    pub(crate) metric: Arc<dyn StringMetric + Send + Sync>,
}

impl Clone for DictionaryCorrector {
    fn clone(&self) -> Self {
        Self {
            dictionary: self.dictionary.clone(),
            config: self.config.clone(),
            metric: self.metric.clone(),
        }
    }
}

impl std::fmt::Debug for DictionaryCorrector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DictionaryCorrector")
            .field("dictionary", &format!("<{} words>", self.dictionary.len()))
            .field("config", &self.config)
            .field("metric", &"<StringMetric>")
            .finish()
    }
}

impl Default for DictionaryCorrector {
    fn default() -> Self {
        // Create a default dictionary with common English words
        let mut dictionary = HashMap::new();

        // Add some common English words
        let common_words = vec![
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "I",
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
            "information",
            "computer",
            "system",
            "data",
            "software",
            "program",
            "application",
            "hardware",
            "network",
            "user",
            "file",
            "memory",
            "process",
            "code",
            "function",
            "algorithm",
            "interface",
            "method",
            "language",
            "programming",
            "library",
            "class",
            "object",
            "variable",
            "value",
            "type",
            "reference",
            "pointer",
            "array",
            "string",
            "receive",
            "believe",
            "achieve",
            "field",
            "friend",
            "science",
            "weight",
            "eight",
            "neighbor",
            "height",
            "weird",
            "foreign",
            "sovereign",
            "ceiling",
            "leisure",
            "neither",
            "protein",
            "caffeine",
            "seize",
            "receipt",
            "perceive",
            "conceive",
            "deceive",
            "except",
            "accept",
            "desert",
            "dessert",
            "principal",
            "principle",
            "stationary",
            "stationery",
            "complement",
            "compliment",
            "affect",
            "effect",
            "lose",
            "loose",
            "than",
            "then",
            "your",
            "you're",
            "its",
            "it's",
            "there",
            "their",
            "they're",
            "weather",
            "whether",
            "hear",
            "here",
            "too",
            "to",
            "two",
        ];

        for word in common_words {
            dictionary.insert(word.to_string(), 100);
        }

        Self {
            dictionary,
            config: DictionaryCorrectorConfig::default(),
            metric: Arc::new(DamerauLevenshteinMetric::new()),
        }
    }
}

impl DictionaryCorrector {
    /// Create a new dictionary-based spelling corrector with the given configuration
    pub fn new(config: DictionaryCorrectorConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Create a new dictionary-based spelling corrector with a custom dictionary
    pub fn with_dictionary(dictionary: HashMap<String, usize>) -> Self {
        Self {
            dictionary,
            config: DictionaryCorrectorConfig::default(),
            metric: Arc::new(DamerauLevenshteinMetric::new()),
        }
    }

    /// Create a new dictionary-based spelling corrector with a custom metric
    pub fn with_metric<M: StringMetric + Send + Sync + 'static>(metric: M) -> Self {
        Self {
            dictionary: HashMap::new(),
            config: DictionaryCorrectorConfig::default(),
            metric: Arc::new(metric),
        }
    }

    /// Load a dictionary from a file
    ///
    /// The file should contain one word per line, optionally followed by a frequency count.
    /// If no frequency is provided, a default value of 1 is used.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| TextError::IoError(format!("Failed to open dictionary file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut dictionary = HashMap::new();

        for line in reader.lines() {
            let line = line.map_err(|e| {
                TextError::IoError(format!("Failed to read line from dictionary file: {}", e))
            })?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Parse the line (word + optional frequency)
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.len() {
                1 => {
                    // Just a word, use default frequency of 1
                    dictionary.insert(parts[0].to_string(), 1);
                }
                2 => {
                    // Word and frequency
                    let word = parts[0];
                    let frequency = parts[1].parse::<usize>().map_err(|e| {
                        TextError::Other(format!("Failed to parse frequency as integer: {}", e))
                    })?;

                    dictionary.insert(word.to_string(), frequency);
                }
                _ => {
                    // Invalid format, skip this line
                    continue;
                }
            }
        }

        Ok(Self {
            dictionary,
            config: DictionaryCorrectorConfig::default(),
            metric: Arc::new(DamerauLevenshteinMetric::new()),
        })
    }

    /// Add a word to the dictionary
    pub fn add_word(&mut self, word: &str, frequency: usize) {
        self.dictionary.insert(word.to_string(), frequency);
    }

    /// Remove a word from the dictionary
    pub fn remove_word(&mut self, word: &str) {
        self.dictionary.remove(word);
    }

    /// Set the string metric to use for similarity calculations
    pub fn set_metric<M: StringMetric + Send + Sync + 'static>(&mut self, metric: M) {
        self.metric = Arc::new(metric);
    }

    /// Set the configuration
    pub fn set_config(&mut self, config: DictionaryCorrectorConfig) {
        self.config = config;
    }

    /// Get the total number of words in the dictionary
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }
}

impl SpellingCorrector for DictionaryCorrector {
    fn correct(&self, word: &str) -> Result<String> {
        // If the word is already correct, return it as is
        if self.is_correct(word) {
            return Ok(word.to_string());
        }

        // Get suggestions and return the best one
        let suggestions = self.get_suggestions(word, 1)?;

        if suggestions.is_empty() {
            // No suggestions found, return the original word
            Ok(word.to_string())
        } else {
            // Return the best suggestion
            Ok(suggestions[0].clone())
        }
    }

    fn get_suggestions(&self, word: &str, limit: usize) -> Result<Vec<String>> {
        // If the word is already correct, return it as the only suggestion
        if self.is_correct(word) {
            return Ok(vec![word.to_string()]);
        }

        let word_to_check = normalize_string(word, self.config.case_sensitive);

        // Find candidates with edit distance less than the threshold
        let mut candidates: Vec<(String, usize, usize)> = Vec::new(); // (word, edit_distance, frequency)

        for (dict_word, frequency) in &self.dictionary {
            if *frequency < self.config.min_frequency {
                continue;
            }

            let dict_word_normalized = normalize_string(dict_word, self.config.case_sensitive);

            // Skip words that are too different in length
            if !is_within_length_threshold(
                &dict_word_normalized,
                &word_to_check,
                self.config.max_edit_distance,
            ) {
                continue;
            }

            // Calculate edit distance
            if let Ok(distance) = self.metric.distance(&word_to_check, &dict_word_normalized) {
                // Convert to usize and check if it's within the threshold
                let distance_usize = distance.round() as usize;
                if distance_usize <= self.config.max_edit_distance {
                    candidates.push((dict_word.clone(), distance_usize, *frequency));
                }
            }
        }

        // Sort candidates by edit distance and optionally by frequency
        if self.config.prioritize_by_frequency {
            candidates.sort_by(|a, b| {
                let (_, dist_a, freq_a) = a;
                let (_, dist_b, freq_b) = b;

                // First, compare by distance
                dist_a.cmp(dist_b)
                    // Then by frequency (higher frequency is better)
                    .then_with(|| freq_b.cmp(freq_a))
            });
        } else {
            candidates.sort_by(|a, b| {
                let (_, dist_a, _) = a;
                let (_, dist_b, _) = b;

                // Sort only by distance
                dist_a.cmp(dist_b)
            });
        }

        // Return the top suggestions, limited by the requested count
        let actual_limit = std::cmp::min(limit, candidates.len());
        let suggestions = candidates[0..actual_limit]
            .iter()
            .map(|(word, _, _)| word.clone())
            .collect();

        Ok(suggestions)
    }

    fn is_correct(&self, word: &str) -> bool {
        dictionary_contains(&self.dictionary, word, self.config.case_sensitive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_corrector_basics() {
        let corrector = DictionaryCorrector::default();

        // Test some common misspellings
        assert_eq!(corrector.correct("recieve").unwrap(), "receive");
        assert_eq!(corrector.correct("freind").unwrap(), "friend");
        assert_eq!(corrector.correct("belive").unwrap(), "believe");

        // Test correct words
        assert_eq!(corrector.correct("computer").unwrap(), "computer");
        assert_eq!(corrector.correct("programming").unwrap(), "programming");

        // Test is_correct
        assert!(corrector.is_correct("computer"));
        assert!(!corrector.is_correct("computre"));
    }

    #[test]
    fn test_dictionary_corrector_with_custom_dictionary() {
        let mut dictionary = HashMap::new();

        // Add some domain-specific terms
        dictionary.insert("rust".to_string(), 100);
        dictionary.insert("cargo".to_string(), 80);
        dictionary.insert("crate".to_string(), 75);
        dictionary.insert("trait".to_string(), 60);
        dictionary.insert("struct".to_string(), 55);
        dictionary.insert("enum".to_string(), 50);

        let corrector = DictionaryCorrector::with_dictionary(dictionary);

        // Test domain-specific corrections
        assert_eq!(corrector.correct("rsut").unwrap(), "rust");
        assert_eq!(corrector.correct("crago").unwrap(), "cargo");
        assert_eq!(corrector.correct("crat").unwrap(), "crate");

        // Test is_correct with domain-specific terms
        assert!(corrector.is_correct("rust"));
        assert!(corrector.is_correct("cargo"));
        assert!(!corrector.is_correct("python")); // Not in dictionary
    }

    #[test]
    fn test_dictionary_corrector_with_custom_config() {
        let config = DictionaryCorrectorConfig {
            max_edit_distance: 1, // More strict
            case_sensitive: true, // Case-sensitive
            max_suggestions: 3,
            min_frequency: 10,
            prioritize_by_frequency: true,
        };

        let mut corrector = DictionaryCorrector::new(config);

        // Add the word deliberately for testing
        corrector.add_word("recieve", 100); // Misspelled version

        // Should return "recieve" itself since it's in the dictionary (even though misspelled)
        assert_eq!(corrector.correct("recieve").unwrap(), "recieve");

        // Test case sensitivity
        assert!(corrector.is_correct("I")); // In dictionary as uppercase
        assert!(!corrector.is_correct("i")); // Not found as lowercase
    }

    #[test]
    fn test_dictionary_corrector_get_suggestions() {
        let mut corrector = DictionaryCorrector::default();

        // Add programming explicitly to ensure test consistency
        corrector.add_word("programming", 100);

        // Test getting multiple suggestions
        let suggestions = corrector.get_suggestions("programing", 3).unwrap();
        assert!(suggestions.contains(&"programming".to_string()));

        // Test with an artificial word that should have no suggestions
        let suggestions = corrector.get_suggestions("xyzabc123", 3).unwrap();
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_dictionary_corrector_correct_text() {
        let mut corrector = DictionaryCorrector::default();

        // Add required words to the dictionary with specific corrections
        corrector.add_word("believe", 100);
        corrector.add_word("received", 100);
        corrector.add_word("was", 100);
        corrector.add_word("correct", 100);

        let text = "I beleive the recieved information was corect.";

        // Create a custom corrector for the test to ensure consistent behavior
        let corrected = corrector.correct_text(text).unwrap();

        // Check each word individually to be more robust
        assert!(corrected.contains("believe"));
        assert!(corrected.contains("received"));
        assert!(corrected.contains("was"));
        assert!(corrected.contains("correct"));
    }
}
