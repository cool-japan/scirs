//! Types and data structures for POS tagging
//!
//! This module contains the core data structures used throughout
//! the POS tagging system.

use crate::stemming::PosTag;

/// Configuration for the POS tagger
#[derive(Debug, Clone)]
pub struct PosTaggerConfig {
    /// Whether to use contextual information (HMM-like approach)
    pub use_context: bool,
    /// Smoothing factor for unknown words (0.0 to 1.0)
    pub smoothing_factor: f64,
    /// Whether to use morphological patterns for disambiguation
    pub use_morphology: bool,
    /// Whether to consider capitalization patterns
    pub use_capitalization: bool,
}

impl Default for PosTaggerConfig {
    fn default() -> Self {
        Self {
            use_context: true,
            smoothing_factor: 0.001,
            use_morphology: true,
            use_capitalization: true,
        }
    }
}

/// Result of POS tagging for a single word
#[derive(Debug, Clone, PartialEq)]
pub struct PosTagResult {
    /// The word that was tagged
    pub word: String,
    /// The assigned POS tag
    pub tag: PosTag,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Result of POS tagging for a sentence or text
#[derive(Debug, Clone)]
pub struct PosTaggingResult {
    /// Original tokens
    pub tokens: Vec<String>,
    /// POS tags for each token
    pub tags: Vec<PosTag>,
    /// Confidence scores for each tag
    pub confidences: Vec<f64>,
}

impl PosTaggingResult {
    /// Create a new tagging result
    pub fn new(tokens: Vec<String>, tags: Vec<PosTag>, confidences: Vec<f64>) -> Self {
        Self {
            tokens,
            tags,
            confidences,
        }
    }

    /// Get the tag for a specific token index
    pub fn get_tag(&self, index: usize) -> Option<&PosTag> {
        self.tags.get(index)
    }

    /// Get the confidence for a specific token index
    pub fn get_confidence(&self, index: usize) -> Option<f64> {
        self.confidences.get(index).copied()
    }

    /// Get the number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if the result is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get an iterator over (token, tag, confidence) tuples
    pub fn iter(&self) -> impl Iterator<Item = (&String, &PosTag, f64)> {
        self.tokens
            .iter()
            .zip(&self.tags)
            .zip(&self.confidences)
            .map(|((token, tag), &confidence)| (token, tag, confidence))
    }
}
