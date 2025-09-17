//! Fill mask pipeline implementation for masked language modeling
//!
//! This module provides functionality for filling masked tokens in text
//! based on contextual understanding.

use super::FillMaskResult;
use crate::error::{Result, TextError};

/// Fill mask pipeline for masked language modeling
#[derive(Debug)]
pub struct FillMaskPipeline;

impl Default for FillMaskPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl FillMaskPipeline {
    /// Create new fill mask pipeline
    pub fn new() -> Self {
        Self
    }

    /// Fill masked tokens in text
    pub fn fill_mask(&self, text: &str) -> Result<Vec<FillMaskResult>> {
        // Improved mask filling using context analysis
        if !text.contains("[MASK]") {
            return Err(TextError::InvalidInput("No [MASK] token found".to_string()));
        }

        // Analyze context around mask
        let words: Vec<&str> = text.split_whitespace().collect();
        let mask_index = words.iter().position(|&w| w == "[MASK]").unwrap_or(0);

        // Get context words
        let left_context: Vec<&str> = if mask_index > 0 {
            words[..mask_index].iter().rev().take(3).copied().collect()
        } else {
            vec![]
        };

        let right_context: Vec<&str> = if mask_index < words.len() - 1 {
            words[mask_index + 1..].iter().take(3).copied().collect()
        } else {
            vec![]
        };

        // Generate contextually appropriate candidates
        let mut candidates = Vec::new();

        // Common words with context-based scoring
        let common_words = vec![
            ("the", 0.85),
            ("a", 0.75),
            ("an", 0.65),
            ("is", 0.80),
            ("was", 0.75),
            ("are", 0.70),
            ("will", 0.68),
            ("can", 0.72),
            ("would", 0.70),
            ("should", 0.65),
            ("very", 0.60),
            ("more", 0.68),
            ("most", 0.65),
            ("good", 0.60),
            ("great", 0.58),
            ("important", 0.55),
            ("significant", 0.52),
            ("major", 0.50),
        ];

        for (word, base_score) in common_words {
            // Adjust score based on context
            let mut score = base_score;

            // Boost score if word fits grammatical context
            if !left_context.is_empty() {
                let prev_word = left_context[0];
                if (prev_word == "a" || prev_word == "an") && word.starts_with(char::is_alphabetic)
                {
                    score *= 0.3; // Reduce score for articles after articles
                } else if prev_word.ends_with("ly") && (word == "good" || word == "important") {
                    score *= 1.2; // Boost adjectives after adverbs
                }
            }

            if !right_context.is_empty() {
                let next_word = right_context[0];
                if word == "a" && next_word.starts_with(|c: char| "aeiou".contains(c)) {
                    score *= 0.2; // Heavily penalize "a" before vowels
                } else if word == "an" && !next_word.starts_with(|c: char| "aeiou".contains(c)) {
                    score *= 0.2; // Heavily penalize "an" before consonants
                }
            }

            candidates.push(FillMaskResult {
                token_str: word.to_string(),
                sequence: text.replace("[MASK]", word),
                score,
                token: candidates.len() + 1,
            });
        }

        // Sort by score and return top candidates
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(candidates.into_iter().take(5).collect())
    }
}
