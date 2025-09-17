//! Question answering pipeline implementation
//!
//! This module provides functionality for extracting answers from context
//! given questions, following the Hugging Face API format.

use super::QuestionAnsweringResult;
use crate::error::Result;

/// Question answering pipeline
#[derive(Debug)]
pub struct QuestionAnsweringPipeline;

impl Default for QuestionAnsweringPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl QuestionAnsweringPipeline {
    /// Create new question answering pipeline
    pub fn new() -> Self {
        Self
    }

    /// Answer question based on context
    pub fn answer(&self, question: &str, context: &str) -> Result<QuestionAnsweringResult> {
        // Simple keyword-based question answering
        let question_lower = question.to_lowercase();
        let question_words: Vec<&str> = question_lower.split_whitespace().collect();
        let context_words: Vec<&str> = context.split_whitespace().collect();

        // Find most relevant sentence or phrase
        let sentences: Vec<&str> = context
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut best_sentence = context;
        let mut best_score = 0.0;
        let mut answer_start = 0;
        let mut answer_end = context.len();

        for sentence in &sentences {
            let sentence_lower = sentence.to_lowercase();
            let sentence_words: Vec<&str> = sentence_lower.split_whitespace().collect();

            // Calculate overlap score
            let overlap = question_words
                .iter()
                .filter(|qw| sentence_words.contains(qw))
                .count();

            let score = overlap as f64 / question_words.len().max(1) as f64;

            if score > best_score {
                best_score = score;
                best_sentence = sentence;

                // Find position in original context
                if let Some(pos) = context.find(sentence) {
                    answer_start = pos;
                    answer_end = pos + sentence.len();
                }
            }
        }

        // Extract answer from best sentence
        let answer = if best_score > 0.2 {
            best_sentence.trim().to_string()
        } else {
            // Fallback: find first relevant phrase
            let words_with_positions: Vec<(usize, &str)> = context_words
                .iter()
                .scan(0, |acc, word| {
                    let start = *acc;
                    *acc += word.len() + 1; // +1 for space
                    Some((start, *word))
                })
                .collect();

            for (start_pos, word) in &words_with_positions {
                if question_words
                    .iter()
                    .any(|qw| qw.to_lowercase() == word.to_lowercase())
                {
                    answer_start = *start_pos;
                    answer_end = start_pos + word.len();
                    break;
                }
            }

            context[answer_start..answer_end].to_string()
        };

        Ok(QuestionAnsweringResult {
            answer,
            start: answer_start,
            end: answer_end,
            score: best_score.max(0.1),
        })
    }
}
