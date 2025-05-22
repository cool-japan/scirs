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

mod dictionary;
mod error_model;
mod ngram;
mod statistical;
mod utils;

pub use dictionary::{DictionaryCorrector, DictionaryCorrectorConfig};
pub use error_model::{EditOp, ErrorModel};
pub use ngram::NGramModel;
pub use statistical::{StatisticalCorrector, StatisticalCorrectorConfig};
// Re-export utility functions that might be useful for users building custom correctors
pub use utils::{extract_words, normalize_string, split_sentences};

use crate::error::Result;

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

#[cfg(test)]
mod tests {
    // Integration tests for the spelling module
}
