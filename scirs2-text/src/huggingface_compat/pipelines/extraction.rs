//! Feature extraction pipeline implementation
//!
//! This module provides feature extraction capabilities for converting
//! text into numerical representations suitable for downstream tasks.

use crate::error::Result;
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use scirs2_core::ndarray::Array2;

/// Feature extraction pipeline
#[derive(Debug)]
pub struct FeatureExtractionPipeline;

impl Default for FeatureExtractionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractionPipeline {
    /// Create new feature extraction pipeline
    pub fn new() -> Self {
        Self
    }

    /// Extract features from text
    pub fn extract_features(&self, text: &str) -> Result<Array2<f64>> {
        // Extract meaningful text features using actual text processing
        let feature_dim = 768;
        let words: Vec<&str> = text.split_whitespace().collect();
        let seq_len = words.len().max(1);

        // Create TF-IDF vectorizer for feature extraction
        let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
        let documents = [text.to_string()];
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let tfidf_matrix = vectorizer.fit_transform(&doc_refs)?;

        // Create feature matrix by extending TF-IDF features
        let mut features = Array2::zeros((seq_len, feature_dim));

        for (i, word) in words.iter().enumerate() {
            // Extract various text features for each word
            let word_len = word.len() as f64;
            let is_upper = if word.chars().all(|c| c.is_uppercase()) {
                1.0
            } else {
                0.0
            };
            let is_title = if word.chars().next().is_some_and(|c| c.is_uppercase()) {
                1.0
            } else {
                0.0
            };
            let has_digits = if word.chars().any(|c| c.is_ascii_digit()) {
                1.0
            } else {
                0.0
            };
            let has_punct = if word.chars().any(|c| c.is_ascii_punctuation()) {
                1.0
            } else {
                0.0
            };

            // Fill feature vector with computed features
            features[[i, 0]] = word_len;
            features[[i, 1]] = is_upper;
            features[[i, 2]] = is_title;
            features[[i, 3]] = has_digits;
            features[[i, 4]] = has_punct;

            // Add character-level features
            for (j, c) in word.chars().take(200).enumerate() {
                if j + 5 < feature_dim {
                    features[[i, j + 5]] = c as u8 as f64 / 255.0;
                }
            }

            // Add TF-IDF features if available
            if i == 0 {
                // Apply TF-IDF to all tokens equally for simplicity
                let tfidf_row = tfidf_matrix.row(0);
                for (k, &value) in tfidf_row.iter().take(feature_dim - 300).enumerate() {
                    if k + 300 < feature_dim {
                        features[[i, k + 300]] = value;
                    }
                }
            }
        }

        Ok(features)
    }

    /// Extract features from multiple texts
    pub fn extract_features_batch(&self, texts: &[&str]) -> Result<Vec<Array2<f64>>> {
        texts
            .iter()
            .map(|text| self.extract_features(text))
            .collect()
    }
}
