//! N-gram language model for text processing and spelling correction
//!
//! This module provides an n-gram language model implementation that can be used
//! for context-aware spelling correction, text generation, and other natural language
//! processing tasks.
//!
//! # Key Components
//!
//! - `NGramModel`: A language model that supports unigrams, bigrams, and trigrams
//!
//! # Example
//!
//! ```
//! use scirs2_text::spelling::NGramModel;
//!
//! # fn main() {
//! // Create a new trigram language model
//! let mut model = NGramModel::new(3);
//!
//! // Train the model with some text
//! model.add_text("The quick brown fox jumps over the lazy dog.");
//! model.add_text("Programming languages like Python and Rust are popular.");
//!
//! // Get probability of a word given its context
//! let context = vec!["quick".to_string(), "brown".to_string()];
//! let prob = model.probability("fox", &context);
//!
//! // Higher probability for words that appeared in the training text
//! assert!(prob > model.probability("cat", &context));
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{Result, TextError};

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
    pub fn unigram_probability(&self, word: &str) -> f64 {
        let word_count = self.unigrams.get(word).copied().unwrap_or(0);

        // Add-one smoothing (Laplace smoothing)
        let vocabulary_size = self.unigrams.len();
        (word_count as f64 + 1.0) / (self.total_words as f64 + vocabulary_size as f64)
    }

    /// Calculate bigram probability P(word | previous)
    pub fn bigram_probability(&self, word: &str, context: &[String]) -> f64 {
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
    pub fn trigram_probability(&self, word: &str, context: &[String]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
