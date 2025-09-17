//! Summarization pipeline implementation
//!
//! This module provides functionality for text summarization using
//! extractive and abstractive approaches.

use super::SummarizationResult;
use crate::error::Result;

/// Summarization pipeline
#[derive(Debug)]
pub struct SummarizationPipeline {
    /// Maximum summary length
    max_length: usize,
    /// Minimum summary length
    min_length: usize,
}

impl Default for SummarizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl SummarizationPipeline {
    /// Create new summarization pipeline
    pub fn new() -> Self {
        Self {
            max_length: 150,
            min_length: 30,
        }
    }

    /// Summarize text
    pub fn summarize(&self, text: &str) -> Result<SummarizationResult> {
        // Simple extractive summarization
        let sentences: Vec<&str> = text
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            return Ok(SummarizationResult {
                summary_text: text.to_string(),
            });
        }

        // Score sentences by length and position
        let mut scored_sentences: Vec<(f64, &str)> = sentences
            .iter()
            .enumerate()
            .map(|(i, &sentence)| {
                let position_score = if i == 0 { 0.8 } else { 0.5 };
                let length_score = (sentence.len() as f64 / 100.0).min(1.0);
                let score = position_score + length_score;
                (score, sentence)
            })
            .collect();

        // Sort by score and take top sentences
        scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut summary_length = 0;
        let mut summary_parts = Vec::new();

        for (_, sentence) in scored_sentences {
            if summary_length + sentence.len() <= self.max_length {
                summary_parts.push(sentence);
                summary_length += sentence.len();
            }
            if summary_length >= self.min_length {
                break;
            }
        }

        let summary_text = if summary_parts.is_empty() {
            sentences[0].to_string()
        } else {
            summary_parts.join(". ") + "."
        };

        Ok(SummarizationResult { summary_text })
    }

    /// Set summarization parameters
    pub fn set_max_length(&mut self, max_length: usize) {
        self.max_length = max_length;
    }

    pub fn set_min_length(&mut self, min_length: usize) {
        self.min_length = min_length;
    }
}
