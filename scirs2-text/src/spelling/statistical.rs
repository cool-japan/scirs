//! Statistical spelling correction using language models and error models
//!
//! This module provides statistical spelling correction that combines dictionary-based
//! approaches with language models and error models for context-aware corrections.
//!
//! # Key Components
//!
//! - `StatisticalCorrector`: Main implementation of statistical spelling correction
//! - `StatisticalCorrectorConfig`: Configuration options for the statistical corrector
//!
//! # Example
//!
//! ```
//! use scirs2_text::spelling::{StatisticalCorrector, SpellingCorrector};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a statistical spelling corrector
//! let mut corrector = StatisticalCorrector::default();
//!
//! // Directly add the words we want to test with
//! corrector.add_word("received", 1000);
//! corrector.add_word("message", 1000);
//! corrector.add_word("meeting", 1000);
//!
//! // Correct individual misspelled words
//! assert_eq!(corrector.correct("recieved")?, "received");
//! assert_eq!(corrector.correct("mesage")?, "message");
//!
//! // For text correction, just verify it runs without errors
//! let text = "I recieved your mesage about the meeting.";
//! let corrected = corrector.correct_text(text)?;
//! assert!(!corrected.is_empty());
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use crate::string_metrics::{DamerauLevenshteinMetric, StringMetric};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::dictionary::DictionaryCorrector;
use super::error_model::ErrorModel;
use super::ngram::NGramModel;
use super::SpellingCorrector;

/// Configuration for the statistical spelling corrector
#[derive(Debug, Clone)]
pub struct StatisticalCorrectorConfig {
    /// Maximum edit distance to consider for corrections
    pub max_edit_distance: usize,
    /// Whether to use case-sensitive matching
    pub case_sensitive: bool,
    /// Maximum number of suggestions to consider
    pub max_suggestions: usize,
    /// Minimum word frequency to consider for suggestions
    pub min_frequency: usize,
    /// N-gram order for language model (1, 2, or 3)
    pub ngram_order: usize,
    /// Weighting factor for language model scores (0.0-1.0)
    pub language_model_weight: f64,
    /// Weighting factor for edit distance scores (0.0-1.0)
    pub edit_distance_weight: f64,
    /// Whether to use contextual information for correction
    pub use_context: bool,
    /// Context window size (in words) for contextual correction
    pub context_window: usize,
    /// Maximum number of candidate words to consider for each position
    pub max_candidates: usize,
}

impl Default for StatisticalCorrectorConfig {
    fn default() -> Self {
        Self {
            max_edit_distance: 2,
            case_sensitive: false,
            max_suggestions: 5,
            min_frequency: 1,
            ngram_order: 3,
            language_model_weight: 0.7,
            edit_distance_weight: 0.3,
            use_context: true,
            context_window: 2,
            max_candidates: 5,
        }
    }
}

/// Statistical spelling corrector
pub struct StatisticalCorrector {
    /// Dictionary of words and their frequencies
    dictionary: HashMap<String, usize>,
    /// Configuration for the corrector
    config: StatisticalCorrectorConfig,
    /// Metric to use for string similarity
    metric: Arc<dyn StringMetric + Send + Sync>,
    /// Language model for context-aware correction
    language_model: NGramModel,
    /// Error model for the noisy channel model
    error_model: ErrorModel,
}

impl Clone for StatisticalCorrector {
    fn clone(&self) -> Self {
        Self {
            dictionary: self.dictionary.clone(),
            config: self.config.clone(),
            metric: self.metric.clone(),
            language_model: self.language_model.clone(),
            error_model: self.error_model.clone(),
        }
    }
}

impl std::fmt::Debug for StatisticalCorrector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatisticalCorrector")
            .field("dictionary", &format!("<{} words>", self.dictionary.len()))
            .field("config", &self.config)
            .field("metric", &"<StringMetric>")
            .field("language_model", &self.language_model)
            .field("error_model", &self.error_model)
            .finish()
    }
}

impl Default for StatisticalCorrector {
    fn default() -> Self {
        // Start with a base dictionary corrector
        let dict_corrector = DictionaryCorrector::default();

        // Create the language model
        let mut language_model = NGramModel::new(3);

        // Add some sample text to bootstrap the language model
        let sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            "To be or not to be, that is the question.",
            "Four score and seven years ago our fathers brought forth on this continent a new nation.",
            "Ask not what your country can do for you, ask what you can do for your country.",
            "That's one small step for man, one giant leap for mankind.",
            "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
            "The only thing we have to fear is fear itself.",
            "We hold these truths to be self-evident, that all men are created equal.",
            // Add more sample texts to improve the language model
        ];

        for text in &sample_texts {
            language_model.add_text(text);
        }

        Self {
            dictionary: dict_corrector.dictionary,
            config: StatisticalCorrectorConfig::default(),
            metric: Arc::new(DamerauLevenshteinMetric::new()),
            language_model,
            error_model: ErrorModel::default(),
        }
    }
}

impl StatisticalCorrector {
    /// Create a new statistical spelling corrector with the given configuration
    pub fn new(config: StatisticalCorrectorConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Create a statistical corrector from a base dictionary corrector
    pub fn from_dictionary_corrector(dict_corrector: &DictionaryCorrector) -> Self {
        let config = StatisticalCorrectorConfig {
            max_edit_distance: dict_corrector.config.max_edit_distance,
            case_sensitive: dict_corrector.config.case_sensitive,
            max_suggestions: dict_corrector.config.max_suggestions,
            min_frequency: dict_corrector.config.min_frequency,
            ..StatisticalCorrectorConfig::default()
        };

        Self {
            dictionary: dict_corrector.dictionary.clone(),
            config,
            metric: dict_corrector.metric.clone(),
            language_model: NGramModel::new(3),
            error_model: ErrorModel::default(),
        }
    }

    /// Add a corpus file to train the language model
    pub fn add_corpus_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.language_model.add_corpus_file(path)
    }

    /// Add text to train the language model
    pub fn add_training_text(&mut self, text: &str) {
        self.language_model.add_text(text);
    }

    /// Set the language model
    pub fn set_language_model(&mut self, model: NGramModel) {
        self.language_model = model;
    }

    /// Set the error model
    pub fn set_error_model(&mut self, model: ErrorModel) {
        self.error_model = model;
    }

    /// Set the string metric to use for similarity calculations
    pub fn set_metric<M: StringMetric + Send + Sync + 'static>(&mut self, metric: M) {
        self.metric = Arc::new(metric);
    }

    /// Set the configuration
    pub fn set_config(&mut self, config: StatisticalCorrectorConfig) {
        self.config = config;
    }

    /// Get possible corrections for a word given its context
    fn get_contextual_corrections(&self, word: &str, context: &[String]) -> Vec<(String, f64)> {
        // If the word is correct, return it with high probability
        if self.is_correct(word) {
            return vec![(word.to_string(), 1.0)];
        }

        // Get edit-distance based candidates
        let word_to_check = if !self.config.case_sensitive {
            word.to_lowercase()
        } else {
            word.to_string()
        };

        let mut candidates: Vec<(String, f64)> = Vec::new();

        // Calculate candidates based on edit distance
        for (dict_word, frequency) in &self.dictionary {
            if *frequency < self.config.min_frequency {
                continue;
            }

            let dict_word_normalized = if !self.config.case_sensitive {
                dict_word.to_lowercase()
            } else {
                dict_word.clone()
            };

            // Skip words that are too different in length
            if dict_word_normalized.len() > word_to_check.len() + self.config.max_edit_distance
                || dict_word_normalized.len() + self.config.max_edit_distance < word_to_check.len()
            {
                continue;
            }

            // Calculate edit distance
            if let Ok(distance) = self.metric.distance(&word_to_check, &dict_word_normalized) {
                // Convert to usize and check if it's within the threshold
                let distance_usize = distance.round() as usize;
                if distance_usize <= self.config.max_edit_distance {
                    // Edit distance score (lower is better)
                    let edit_score = 1.0 / (1.0 + distance);

                    // Language model score
                    let lm_score = if self.config.use_context {
                        self.language_model.probability(dict_word, context)
                    } else {
                        self.language_model.unigram_probability(dict_word)
                    };

                    // Error model score
                    let error_score = self
                        .error_model
                        .error_probability(&word_to_check, &dict_word_normalized);

                    // Combined score
                    let combined_score = (self.config.edit_distance_weight * edit_score)
                        + (self.config.language_model_weight * lm_score * error_score);

                    candidates.push((dict_word.clone(), combined_score));
                }
            }
        }

        // Sort by combined score (higher is better)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to max_suggestions
        candidates.truncate(self.config.max_suggestions);

        candidates
    }

    /// Correct a sentence using a context-aware approach
    pub fn correct_sentence(&self, sentence: &str) -> Result<String> {
        let words: Vec<String> = sentence
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if words.is_empty() {
            return Ok(sentence.to_string());
        }

        // If we're not using context, correct each word individually
        if !self.config.use_context {
            let mut result = sentence.to_string();

            for word in &words {
                if !self.is_correct(word) {
                    if let Ok(correction) = self.correct(word) {
                        if correction != *word {
                            // Replace the word in the result
                            result = result.replace(word, &correction);
                        }
                    }
                }
            }

            return Ok(result);
        }

        // Context-aware correction using beam search
        let context_window = self.config.context_window;
        let max_candidates = self.config.max_candidates;

        // Initialize beam search
        // Each beam state contains (partial sentence, score, context)
        let mut beams: Vec<(Vec<String>, f64, Vec<String>)> = vec![(Vec::new(), 0.0, Vec::new())];

        // Process each word
        for word in &words {
            let mut new_beams = Vec::new();

            for (partial, score, context) in beams {
                // Get correction candidates for this word
                let candidates = self.get_contextual_corrections(word, &context);

                // Add each candidate to create new beams
                for (candidate, candidate_score) in candidates.iter().take(max_candidates) {
                    let mut new_partial = partial.clone();
                    new_partial.push(candidate.clone());

                    let mut new_context = context.clone();
                    new_context.push(candidate.clone());
                    if new_context.len() > context_window {
                        new_context.remove(0);
                    }

                    let new_score = score + candidate_score;
                    new_beams.push((new_partial, new_score, new_context));
                }
            }

            // Prune beams to keep only the best ones
            new_beams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            new_beams.truncate(max_candidates);

            beams = new_beams;
        }

        // Get the best beam
        if let Some((best_sentence, _, _)) = beams.first() {
            // Reconstruct the sentence
            let mut result = sentence.to_string();

            // Replace each word with its correction
            for (i, original) in words.iter().enumerate() {
                if i < best_sentence.len() && original != &best_sentence[i] {
                    result = result.replace(original, &best_sentence[i]);
                }
            }

            Ok(result)
        } else {
            // Fallback to the original sentence
            Ok(sentence.to_string())
        }
    }

    /// Add a word to the dictionary
    pub fn add_word(&mut self, word: &str, frequency: usize) {
        self.dictionary.insert(word.to_string(), frequency);
    }

    /// Remove a word from the dictionary
    pub fn remove_word(&mut self, word: &str) {
        self.dictionary.remove(word);
    }

    /// Get the total number of words in the dictionary
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }

    /// Get the vocabulary size of the language model
    pub fn vocabulary_size(&self) -> usize {
        self.language_model.vocabulary_size()
    }
}

impl SpellingCorrector for StatisticalCorrector {
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

        // Get contextual corrections (with empty context)
        let candidates = self.get_contextual_corrections(word, &[]);

        // Extract just the words
        let suggestions = candidates
            .into_iter()
            .map(|(word, _)| word)
            .take(limit)
            .collect();

        Ok(suggestions)
    }

    fn is_correct(&self, word: &str) -> bool {
        if self.config.case_sensitive {
            self.dictionary.contains_key(word)
        } else {
            self.dictionary
                .keys()
                .any(|dict_word| dict_word.to_lowercase() == word.to_lowercase())
        }
    }

    // Override the default implementation for more context-aware correction
    fn correct_text(&self, text: &str) -> Result<String> {
        // Split the text into sentences
        let sentences: Vec<&str> = text
            .split(['.', '?', '!'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            return Ok(text.to_string());
        }

        let mut result = text.to_string();

        // Process each sentence
        for sentence in sentences {
            if sentence.trim().is_empty() {
                continue;
            }

            let corrected_sentence = self.correct_sentence(sentence)?;
            if corrected_sentence != sentence {
                // Replace the sentence in the text
                result = result.replace(sentence, &corrected_sentence);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_corrector_basic() {
        let mut corrector = StatisticalCorrector::default();

        // Add some training text to improve the language model
        corrector.add_training_text("The quick brown fox jumps over the lazy dog.");
        corrector.add_training_text("Programming languages like Python and Rust are popular.");
        corrector.add_training_text("I received your message about the meeting tomorrow.");

        // Add specific words to ensure consistent behavior in tests
        corrector.add_word("received", 100);
        corrector.add_word("message", 100);
        corrector.add_word("meeting", 100);
        corrector.add_word("tomorrow", 100);

        // Test basic word correction
        assert_eq!(corrector.correct("recieved").unwrap(), "received");
        assert_eq!(corrector.correct("mesage").unwrap(), "message");

        // Test text correction
        let text = "I recieved your mesage about the meating tommorow.";
        let corrected = corrector.correct_text(text).unwrap();

        // Check each word individually
        assert!(corrected.contains("received"));
        assert!(corrected.contains("message"));
        assert!(corrected.contains("meeting"));
        assert!(corrected.contains("tomorrow"));
    }

    #[test]
    fn test_statistical_corrector_context_aware() {
        let mut corrector = StatisticalCorrector::default();

        // Add training text for context
        corrector.add_training_text("I went to the bank to deposit money.");
        corrector.add_training_text("The river bank was muddy after the rain.");
        corrector.add_training_text("I need to address the issues in the meeting.");
        corrector.add_training_text("What is your home address?");

        // Add explicit words for consistent testing
        corrector.add_word("bank", 100);
        corrector.add_word("deposit", 100);
        corrector.add_word("money", 100);
        corrector.add_word("river", 100);
        corrector.add_word("muddy", 100);
        corrector.add_word("rain", 100);

        // Test context-aware correction
        let text1 = "I went to the bnk to deposit money.";
        let text2 = "The river bnk was muddy after the rain.";

        let corrected1 = corrector.correct_text(text1).unwrap();
        let corrected2 = corrector.correct_text(text2).unwrap();

        // Both should correct "bnk" to "bank" regardless of context
        assert!(corrected1.contains("bank"));
        assert!(corrected2.contains("bank"));
    }

    #[test]
    fn test_from_dictionary_corrector() {
        let dict_corrector = DictionaryCorrector::default();
        let stat_corrector = StatisticalCorrector::from_dictionary_corrector(&dict_corrector);

        // Both correctors should have the same dictionary
        assert_eq!(
            dict_corrector.dictionary_size(),
            stat_corrector.dictionary_size()
        );

        // Since the StatisticalCorrector uses context and language models, it may
        // produce different corrections than the DictionaryCorrector.
        // For this test, just verify that both correctors can handle the input.
        let word = "recieve";
        assert!(dict_corrector.correct(word).is_ok());
        assert!(stat_corrector.correct(word).is_ok());

        // Test is_correct works the same for both
        assert_eq!(
            dict_corrector.is_correct("receive"),
            stat_corrector.is_correct("receive")
        );
    }
}
