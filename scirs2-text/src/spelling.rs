//! Spelling correction algorithms
//!
//! This module provides functionality for detecting and correcting misspelled words
//! using dictionary-based and statistical approaches.
//!
//! # Dictionary-based Correction
//!
//! The dictionary-based approach uses a reference word list and string similarity
//! metrics to find the closest match for a potentially misspelled word.
//!
//! # Statistical Correction
//!
//! The statistical approach uses n-gram language models and the noisy channel model
//! to find the most likely correction for a misspelled word based on context.
//!
//! # Features
//!
//! - Dictionary-based correction with configurable similarity metrics
//! - Statistical correction using n-gram language models
//! - Support for custom dictionaries and language models
//! - Context-aware suggestion prioritization
//! - Customizable edit distance thresholds
//!
//! # Examples
//!
//! ## Dictionary-based Correction
//!
//! ```
//! use scirs2_text::spelling::{DictionaryCorrector, SpellingCorrector};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a dictionary-based spelling corrector
//! let corrector = DictionaryCorrector::default();
//!
//! // Correct a misspelled word
//! let corrected = corrector.correct("recieve")?;
//! assert_eq!(corrected, "receive");
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Statistical Correction
//!
//! ```
//! use scirs2_text::spelling::{StatisticalCorrector, SpellingCorrector};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a statistical spelling corrector with added dictionary entries
//! let mut corrector = StatisticalCorrector::default();
//!
//! // Explicitly add words to the dictionary to ensure consistent behavior in tests
//! corrector.add_word("received", 100);
//! corrector.add_word("message", 100);
//!
//! // Correct a misspelled word in context
//! let text = "I recieved your mesage";
//! let corrected = corrector.correct_text(text)?;
//! // Check for corrections
//! assert!(corrected.contains("received"));
//! assert!(corrected.contains("message"));
//!
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TextError};
use crate::string_metrics::{DamerauLevenshteinMetric, StringMetric};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

/// Trait for spelling correction algorithms
pub trait SpellingCorrector {
    /// Correct a potentially misspelled word
    fn correct(&self, word: &str) -> Result<String>;

    /// Get a list of suggestions for a potentially misspelled word
    fn get_suggestions(&self, word: &str, limit: usize) -> Result<Vec<String>>;

    /// Check if a word is spelled correctly
    fn is_correct(&self, word: &str) -> bool;

    /// Correct all words in a text
    fn correct_text(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Extract words for processing
        let words: Vec<(usize, &str)> = text
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .enumerate()
            .filter(|(_, s)| !s.is_empty())
            .collect();

        // Process each word
        for (_, word) in words {
            if !self.is_correct(word) {
                // Replace the misspelled word with the correction
                let correction = self.correct(word)?;
                if correction != word {
                    // Replace the word in the text
                    // This is a simplistic approach - in a real implementation,
                    // we would need to be more careful about word boundaries
                    result = result.replace(word, &correction);
                }
            }
        }

        Ok(result)
    }
}

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

/// Dictionary-based spelling corrector
pub struct DictionaryCorrector {
    /// Dictionary of words and their frequencies
    dictionary: HashMap<String, usize>,
    /// Configuration for the corrector
    config: DictionaryCorrectorConfig,
    /// Metric to use for string similarity
    metric: Arc<dyn StringMetric + Send + Sync>,
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
            // Add more words as needed
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

        let word_to_check = if !self.config.case_sensitive {
            word.to_lowercase()
        } else {
            word.to_string()
        };

        // Find candidates with edit distance less than the threshold
        let mut candidates: Vec<(String, usize, usize)> = Vec::new(); // (word, edit_distance, frequency)

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
        if self.config.case_sensitive {
            self.dictionary.contains_key(word)
        } else {
            self.dictionary
                .keys()
                .any(|dict_word| dict_word.to_lowercase() == word.to_lowercase())
        }
    }
}

/// N-gram language model for statistical spelling correction
#[derive(Clone)]
pub struct NGramModel {
    /// Unigram counts
    unigrams: HashMap<String, usize>,
    /// Bigram counts
    bigrams: HashMap<(String, String), usize>,
    /// Trigram counts
    trigrams: HashMap<(String, String, String), usize>,
    /// Total number of words in training data
    total_words: usize,
    /// Order of the n-gram model
    order: usize,
    /// Start of sentence token
    start_token: String,
    /// End of sentence token
    end_token: String,
}

impl NGramModel {
    /// Create a new n-gram model with the specified order
    pub fn new(order: usize) -> Self {
        if order > 3 {
            // Warn but limit to 3
            eprintln!("Warning: NGramModel only supports orders up to 3. Using order=3.");
        }

        Self {
            unigrams: HashMap::new(),
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            total_words: 0,
            order: order.clamp(1, 3),
            start_token: "<s>".to_string(),
            end_token: "</s>".to_string(),
        }
    }

    /// Add a text to the language model
    pub fn add_text(&mut self, text: &str) {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|s| {
                s.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|s| !s.is_empty())
            .collect();

        if words.is_empty() {
            return;
        }

        // Process sentences (separated by punctuation)
        let mut current_sentence = Vec::new();

        for word in words {
            // Check if this is end of sentence
            let is_end = word.ends_with('.') || word.ends_with('?') || word.ends_with('!');

            // Add word to current sentence
            let clean_word = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            if !clean_word.is_empty() {
                current_sentence.push(clean_word);
                self.total_words += 1;
            }

            // Process sentence if we're at the end
            if is_end && !current_sentence.is_empty() {
                self.process_sentence(&current_sentence);
                current_sentence.clear();
            }
        }

        // Process any remaining words as a sentence
        if !current_sentence.is_empty() {
            self.process_sentence(&current_sentence);
        }
    }

    /// Process a single sentence to add to the language model
    fn process_sentence(&mut self, sentence: &[String]) {
        // Add start and end tokens
        let mut words = Vec::with_capacity(sentence.len() + 2);
        words.push(self.start_token.clone());
        words.extend(sentence.iter().cloned());
        words.push(self.end_token.clone());

        // Update unigram counts
        for word in &words {
            *self.unigrams.entry(word.clone()).or_insert(0) += 1;
        }

        // Update bigram counts if order >= 2
        if self.order >= 2 {
            for i in 0..words.len() - 1 {
                let bigram = (words[i].clone(), words[i + 1].clone());
                *self.bigrams.entry(bigram).or_insert(0) += 1;
            }
        }

        // Update trigram counts if order >= 3
        if self.order >= 3 {
            for i in 0..words.len() - 2 {
                let trigram = (words[i].clone(), words[i + 1].clone(), words[i + 2].clone());
                *self.trigrams.entry(trigram).or_insert(0) += 1;
            }
        }
    }

    /// Add a corpus file to the language model
    pub fn add_corpus_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let file = File::open(path)
            .map_err(|e| TextError::IoError(format!("Failed to open corpus file: {}", e)))?;

        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.map_err(|e| {
                TextError::IoError(format!("Failed to read line from corpus file: {}", e))
            })?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            self.add_text(&line);
        }

        Ok(())
    }

    /// Generate a probability estimate for a word given its context
    pub fn probability(&self, word: &str, context: &[String]) -> f64 {
        match self.order {
            1 => self.unigram_probability(word),
            2 => self.bigram_probability(word, context),
            3 => self.trigram_probability(word, context),
            _ => self.unigram_probability(word), // Default fallback
        }
    }

    /// Calculate unigram probability P(word)
    fn unigram_probability(&self, word: &str) -> f64 {
        let word_count = self.unigrams.get(word).copied().unwrap_or(0);

        // Add-one smoothing (Laplace smoothing)
        let vocabulary_size = self.unigrams.len();
        (word_count as f64 + 1.0) / (self.total_words as f64 + vocabulary_size as f64)
    }

    /// Calculate bigram probability P(word | previous)
    fn bigram_probability(&self, word: &str, context: &[String]) -> f64 {
        if context.is_empty() {
            return self.unigram_probability(word);
        }

        let previous = &context[context.len() - 1];

        let bigram_count = self
            .bigrams
            .get(&(previous.clone(), word.to_string()))
            .copied()
            .unwrap_or(0);

        let previous_count = self.unigrams.get(previous).copied().unwrap_or(0);

        if previous_count == 0 {
            return self.unigram_probability(word);
        }

        // Add-one smoothing
        let vocabulary_size = self.unigrams.len();
        (bigram_count as f64 + 1.0) / (previous_count as f64 + vocabulary_size as f64)
    }

    /// Calculate trigram probability P(word | previous1, previous2)
    fn trigram_probability(&self, word: &str, context: &[String]) -> f64 {
        if context.len() < 2 {
            return self.bigram_probability(word, context);
        }

        let previous1 = &context[context.len() - 2];
        let previous2 = &context[context.len() - 1];

        let trigram_count = self
            .trigrams
            .get(&(previous1.clone(), previous2.clone(), word.to_string()))
            .copied()
            .unwrap_or(0);

        let bigram_count = self
            .bigrams
            .get(&(previous1.clone(), previous2.clone()))
            .copied()
            .unwrap_or(0);

        if bigram_count == 0 {
            return self.bigram_probability(word, &[previous2.clone()]);
        }

        // Add-one smoothing
        let vocabulary_size = self.unigrams.len();
        (trigram_count as f64 + 1.0) / (bigram_count as f64 + vocabulary_size as f64)
    }

    /// Calculate perplexity on a test text
    pub fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|s| {
                s.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|s| !s.is_empty())
            .collect();

        if words.is_empty() {
            return f64::INFINITY;
        }

        let mut log_prob_sum = 0.0;
        let mut context = Vec::new();

        for word in words.iter() {
            let prob = self.probability(word, &context);
            log_prob_sum += (prob + 1e-10).log2(); // Add small epsilon to avoid log(0)

            // Update context for next word
            context.push(word.clone());
            if context.len() > self.order {
                context.remove(0);
            }
        }

        // Perplexity = 2^(-average log probability)
        2.0f64.powf(-log_prob_sum / words.len() as f64)
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.unigrams.len()
    }

    /// Get total words processed
    pub fn total_words(&self) -> usize {
        self.total_words
    }

    /// Get the frequency of a word
    pub fn word_frequency(&self, word: &str) -> usize {
        self.unigrams.get(word).copied().unwrap_or(0)
    }

    /// Generate potential single-edit typos for a word
    pub fn generate_typos(&self, word: &str, num_typos: usize) -> Vec<String> {
        let mut typos = HashSet::new();
        let word = word.to_lowercase();
        let chars: Vec<char> = word.chars().collect();

        // Deletion errors (removing one character)
        for i in 0..chars.len() {
            let mut new_word = String::new();
            for (j, &c) in chars.iter().enumerate() {
                if j != i {
                    new_word.push(c);
                }
            }
            typos.insert(new_word);
        }

        // Transposition errors (swapping adjacent characters)
        for i in 0..chars.len() - 1 {
            let mut new_chars = chars.clone();
            new_chars.swap(i, i + 1);
            typos.insert(new_chars.iter().collect());
        }

        // Insertion errors (adding one character)
        for i in 0..=chars.len() {
            for c in 'a'..='z' {
                let mut new_chars = chars.clone();
                new_chars.insert(i, c);
                typos.insert(new_chars.iter().collect());
            }
        }

        // Replacement errors (changing one character)
        for i in 0..chars.len() {
            for c in 'a'..='z' {
                if chars[i] != c {
                    let mut new_chars = chars.clone();
                    new_chars[i] = c;
                    typos.insert(new_chars.iter().collect());
                }
            }
        }

        // Convert to Vec and limit by frequency
        let mut typos_vec: Vec<_> = typos.into_iter().collect();

        // Sort by word frequency in our model
        typos_vec.sort_by(|a, b| {
            let freq_a = self.word_frequency(a);
            let freq_b = self.word_frequency(b);
            freq_b.cmp(&freq_a) // Higher frequency first
        });

        // Limit to requested number
        typos_vec.truncate(num_typos);

        typos_vec
    }
}

impl std::fmt::Debug for NGramModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NGramModel")
            .field("order", &self.order)
            .field("vocabulary_size", &self.vocabulary_size())
            .field("total_words", &self.total_words)
            .field("unigrams", &format!("<{} entries>", self.unigrams.len()))
            .field("bigrams", &format!("<{} entries>", self.bigrams.len()))
            .field("trigrams", &format!("<{} entries>", self.trigrams.len()))
            .finish()
    }
}

/// Error model for the noisy channel model
#[derive(Debug, Clone)]
pub struct ErrorModel {
    /// Probability of deletion errors
    pub p_deletion: f64,
    /// Probability of insertion errors
    pub p_insertion: f64,
    /// Probability of substitution errors
    pub p_substitution: f64,
    /// Probability of transposition errors
    pub p_transposition: f64,
    /// Character confusion matrix
    _char_confusion: HashMap<(char, char), f64>,
}

impl Default for ErrorModel {
    fn default() -> Self {
        Self {
            p_deletion: 0.25,
            p_insertion: 0.25,
            p_substitution: 0.25,
            p_transposition: 0.25,
            _char_confusion: HashMap::new(),
        }
    }
}

impl ErrorModel {
    /// Create a new error model with custom error probabilities
    pub fn new(
        p_deletion: f64,
        p_insertion: f64,
        p_substitution: f64,
        p_transposition: f64,
    ) -> Self {
        // Normalize probabilities to sum to 1.0
        let total = p_deletion + p_insertion + p_substitution + p_transposition;
        Self {
            p_deletion: p_deletion / total,
            p_insertion: p_insertion / total,
            p_substitution: p_substitution / total,
            p_transposition: p_transposition / total,
            _char_confusion: HashMap::new(),
        }
    }

    /// Calculate the error probability P(typo | correct)
    pub fn error_probability(&self, typo: &str, correct: &str) -> f64 {
        // Special case: identical words
        if typo == correct {
            return 1.0;
        }

        // Simple edit distance-based probability
        let edit_distance = self.min_edit_operations(typo, correct);

        match edit_distance.len() {
            0 => 1.0, // No edits needed
            1 => {
                // Single edit
                match edit_distance[0] {
                    EditOp::Delete(_) => self.p_deletion,
                    EditOp::Insert(_) => self.p_insertion,
                    EditOp::Substitute(_, _) => self.p_substitution,
                    EditOp::Transpose(_, _) => self.p_transposition,
                }
            }
            n => {
                // Multiple edits - calculate product of probabilities, with decay
                let base_prob = 0.1f64.powi(n as i32 - 1);
                let mut prob = base_prob;

                for op in &edit_distance {
                    match op {
                        EditOp::Delete(_) => prob *= self.p_deletion,
                        EditOp::Insert(_) => prob *= self.p_insertion,
                        EditOp::Substitute(_, _) => prob *= self.p_substitution,
                        EditOp::Transpose(_, _) => prob *= self.p_transposition,
                    }
                }

                prob
            }
        }
    }

    /// Find the minimum edit operations to transform correct into typo
    fn min_edit_operations(&self, typo: &str, correct: &str) -> Vec<EditOp> {
        let typo_chars: Vec<char> = typo.chars().collect();
        let correct_chars: Vec<char> = correct.chars().collect();

        // Try to detect the type of error
        if correct_chars.len() == typo_chars.len() + 1 {
            // Possible deletion
            for i in 0..correct_chars.len() {
                let mut test_chars = correct_chars.clone();
                test_chars.remove(i);
                if test_chars == typo_chars {
                    return vec![EditOp::Delete(correct_chars[i])];
                }
            }
        } else if correct_chars.len() + 1 == typo_chars.len() {
            // Possible insertion
            for i in 0..typo_chars.len() {
                let mut test_chars = typo_chars.clone();
                test_chars.remove(i);
                if test_chars == correct_chars {
                    return vec![EditOp::Insert(typo_chars[i])];
                }
            }
        } else if correct_chars.len() == typo_chars.len() {
            // Possible substitution or transposition
            let mut diff_positions = Vec::new();

            for i in 0..correct_chars.len() {
                if correct_chars[i] != typo_chars[i] {
                    diff_positions.push(i);
                }
            }

            if diff_positions.len() == 1 {
                // Single substitution
                let i = diff_positions[0];
                return vec![EditOp::Substitute(correct_chars[i], typo_chars[i])];
            } else if diff_positions.len() == 2 && diff_positions[0] + 1 == diff_positions[1] {
                let i = diff_positions[0];

                // Check if it's a transposition
                if correct_chars[i] == typo_chars[i + 1] && correct_chars[i + 1] == typo_chars[i] {
                    return vec![EditOp::Transpose(correct_chars[i], correct_chars[i + 1])];
                }
            }
        }

        // Fallback: use Levenshtein to determine general edit distance
        // This is a simplified approach - a real implementation would track the actual operations
        let mut operations = Vec::new();
        let _distance = self.levenshtein_with_ops(correct, typo, &mut operations);
        operations
    }

    /// Levenshtein distance with operations tracking
    fn levenshtein_with_ops(&self, s1: &str, s2: &str, operations: &mut Vec<EditOp>) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        // Create distance matrix
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }

        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill matrix and track operations
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

                matrix[i][j] = std::cmp::min(
                    matrix[i - 1][j] + 1, // deletion
                    std::cmp::min(
                        matrix[i][j - 1] + 1,        // insertion
                        matrix[i - 1][j - 1] + cost, // substitution
                    ),
                );

                // Check for transposition (if possible)
                if i > 1
                    && j > 1
                    && chars1[i - 1] == chars2[j - 2]
                    && chars1[i - 2] == chars2[j - 1]
                {
                    matrix[i][j] = std::cmp::min(
                        matrix[i][j],
                        matrix[i - 2][j - 2] + 1, // transposition
                    );
                }
            }
        }

        // Backtrack to find operations
        let mut i = len1;
        let mut j = len2;

        // Use a temporary vector to store operations in correct order
        let mut temp_ops = Vec::new();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && chars1[i - 1] == chars2[j - 1] {
                // No operation (match)
                i -= 1;
                j -= 1;
            } else if i > 1
                && j > 1
                && chars1[i - 1] == chars2[j - 2]
                && chars1[i - 2] == chars2[j - 1]
                && matrix[i][j] == matrix[i - 2][j - 2] + 1
            {
                // Transposition
                temp_ops.push(EditOp::Transpose(chars1[i - 2], chars1[i - 1]));
                i -= 2;
                j -= 2;
            } else if i > 0 && j > 0 && matrix[i][j] == matrix[i - 1][j - 1] + 1 {
                // Substitution
                temp_ops.push(EditOp::Substitute(chars1[i - 1], chars2[j - 1]));
                i -= 1;
                j -= 1;
            } else if i > 0 && matrix[i][j] == matrix[i - 1][j] + 1 {
                // Deletion
                temp_ops.push(EditOp::Delete(chars1[i - 1]));
                i -= 1;
            } else if j > 0 && matrix[i][j] == matrix[i][j - 1] + 1 {
                // Insertion
                temp_ops.push(EditOp::Insert(chars2[j - 1]));
                j -= 1;
            } else {
                // Should not reach here
                break;
            }
        }

        // Reverse operations to get correct order
        temp_ops.reverse();
        operations.extend(temp_ops);

        matrix[len1][len2]
    }
}

/// Edit operations for the error model
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// Delete a character
    Delete(char),
    /// Insert a character
    Insert(char),
    /// Substitute one character for another
    Substitute(char, char),
    /// Transpose two adjacent characters
    Transpose(char, char),
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

    #[test]
    fn test_ngram_model_basics() {
        let mut model = NGramModel::new(3);

        // Add some training data
        model.add_text("The quick brown fox jumps over the lazy dog.");

        // Test unigram probabilities
        let p_the = model.unigram_probability("the");
        let p_quick = model.unigram_probability("quick");
        let p_unknown = model.unigram_probability("unknown");

        // The should be more frequent than quick
        assert!(p_the > p_quick);

        // Unknown words should have non-zero probability due to smoothing
        assert!(p_unknown > 0.0);

        // Test bigram probabilities
        let p_quick_given_the = model.bigram_probability("quick", &["the".to_string()]);
        let p_brown_given_quick = model.bigram_probability("brown", &["quick".to_string()]);

        // These specific bigrams should exist in the training data
        assert!(p_quick_given_the > 0.0);
        assert!(p_brown_given_quick > 0.0);

        // Test trigram model
        let p_fox_given_quick_brown =
            model.trigram_probability("fox", &["quick".to_string(), "brown".to_string()]);

        // This specific trigram should exist in the training data
        assert!(p_fox_given_quick_brown > 0.0);
    }

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
    fn test_error_model() {
        let error_model = ErrorModel::default();

        // Test error probability calculations
        let p_deletion = error_model.error_probability("cat", "cart"); // 'r' deleted
        let p_insertion = error_model.error_probability("cart", "cat"); // 'r' inserted
        let p_substitution = error_model.error_probability("cat", "cut"); // 'a' -> 'u'
        let p_transposition = error_model.error_probability("form", "from"); // 'or' -> 'ro'

        // Each type of error should have non-zero probability
        assert!(p_deletion > 0.0);
        assert!(p_insertion > 0.0);
        assert!(p_substitution > 0.0);
        assert!(p_transposition > 0.0);

        // For identical words, probability should be 1.0
        assert_eq!(error_model.error_probability("word", "word"), 1.0);
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
}
