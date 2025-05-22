//! Text vectorization utilities
//!
//! This module provides functionality for converting text into
//! numerical vector representations.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use ndarray::{Array1, Array2, Axis};
use scirs2_core::parallel;
use std::collections::HashMap;

/// Trait for text vectorizers
pub trait Vectorizer: Clone {
    /// Fit the vectorizer on a corpus of texts
    fn fit(&mut self, texts: &[&str]) -> Result<()>;

    /// Transform a text into a vector
    fn transform(&self, text: &str) -> Result<Array1<f64>>;

    /// Transform a batch of texts into a matrix where each row is a document vector
    fn transform_batch(&self, texts: &[&str]) -> Result<Array2<f64>>;

    /// Fit on a corpus and then transform a batch of texts
    fn fit_transform(&mut self, texts: &[&str]) -> Result<Array2<f64>> {
        self.fit(texts)?;
        self.transform_batch(texts)
    }
}

/// Count vectorizer that uses a bag-of-words representation
#[derive(Clone)]
pub struct CountVectorizer {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    vocabulary: Vocabulary,
    binary: bool, // If true, all non-zero counts are set to 1
}

impl Clone for Box<dyn Tokenizer + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl CountVectorizer {
    /// Create a new count vectorizer
    pub fn new(binary: bool) -> Self {
        Self {
            tokenizer: Box::new(WordTokenizer::default()),
            vocabulary: Vocabulary::new(),
            binary,
        }
    }

    /// Create a count vectorizer with a custom tokenizer
    pub fn with_tokenizer(tokenizer: Box<dyn Tokenizer + Send + Sync>, binary: bool) -> Self {
        Self {
            tokenizer,
            vocabulary: Vocabulary::new(),
            binary,
        }
    }

    /// Get a reference to the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Vectorizer for CountVectorizer {
    fn fit(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for fitting".into(),
            ));
        }

        // Clear any existing vocabulary
        self.vocabulary = Vocabulary::new();

        // Process all documents to build vocabulary
        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            for token in tokens {
                self.vocabulary.add_token(&token);
            }
        }

        Ok(())
    }

    fn transform(&self, text: &str) -> Result<Array1<f64>> {
        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "Vocabulary is empty. Call fit() first".into(),
            ));
        }

        let vocab_size = self.vocabulary.len();
        let mut vector = Array1::zeros(vocab_size);

        // Tokenize the text
        let tokens = self.tokenizer.tokenize(text)?;

        // Count tokens
        for token in tokens {
            if let Some(idx) = self.vocabulary.get_index(&token) {
                vector[idx] += 1.0;
            }
        }

        // Make binary if requested
        if self.binary {
            for val in vector.iter_mut() {
                if *val > 0.0 {
                    *val = 1.0;
                }
            }
        }

        Ok(vector)
    }

    fn transform_batch(&self, texts: &[&str]) -> Result<Array2<f64>> {
        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "Vocabulary is empty. Call fit() first".into(),
            ));
        }

        if texts.is_empty() {
            return Ok(Array2::zeros((0, self.vocabulary.len())));
        }

        // Use scirs2-core::parallel for parallel processing
        // Clone data to avoid lifetime issues
        let texts_owned: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
        let self_clone = self.clone();

        let vectors = parallel::parallel_map(&texts_owned, move |text| {
            self_clone.transform(text).map_err(|e| {
                // Convert TextError to CoreError
                scirs2_core::CoreError::ComputationError(scirs2_core::error::ErrorContext::new(
                    format!("Text vectorization error: {}", e),
                ))
            })
        })?;

        // Convert to 2D array
        let n_samples = vectors.len();
        let n_features = self.vocabulary.len();

        let mut matrix = Array2::zeros((n_samples, n_features));
        for (i, vec) in vectors.iter().enumerate() {
            matrix.row_mut(i).assign(vec);
        }

        Ok(matrix)
    }
}

/// TF-IDF vectorizer that computes term frequency-inverse document frequency
#[derive(Clone)]
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    idf: Option<Array1<f64>>,
    smooth_idf: bool,
    norm: Option<String>, // None, "l1", "l2"
}

impl TfidfVectorizer {
    /// Create a new TF-IDF vectorizer
    pub fn new(binary: bool, smooth_idf: bool, norm: Option<String>) -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(binary),
            idf: None,
            smooth_idf,
            norm,
        }
    }

    /// Create a TF-IDF vectorizer with a custom tokenizer
    pub fn with_tokenizer(
        tokenizer: Box<dyn Tokenizer + Send + Sync>,
        binary: bool,
        smooth_idf: bool,
        norm: Option<String>,
    ) -> Self {
        Self {
            count_vectorizer: CountVectorizer::with_tokenizer(tokenizer, binary),
            idf: None,
            smooth_idf,
            norm,
        }
    }

    /// Get a reference to the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        self.count_vectorizer.vocabulary()
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.count_vectorizer.vocabulary_size()
    }

    /// Compute IDF values from document frequencies
    fn compute_idf(&mut self, df: &Array1<f64>, n_documents: f64) -> Result<()> {
        let n_features = df.len();

        let mut idf = Array1::zeros(n_features);

        for (i, &df_i) in df.iter().enumerate() {
            if df_i > 0.0 {
                if self.smooth_idf {
                    // log((n_documents + 1) / (df + 1)) + 1
                    idf[i] = ((n_documents + 1.0) / (df_i + 1.0)).ln() + 1.0;
                } else {
                    // log(n_documents / df)
                    idf[i] = (n_documents / df_i).ln();
                }
            } else if self.smooth_idf {
                idf[i] = ((n_documents + 1.0) / 1.0).ln() + 1.0;
            } else {
                // For features that aren't present in the corpus, set IDF to a high value
                idf[i] = 0.0;
            }
        }

        self.idf = Some(idf);
        Ok(())
    }

    /// Apply normalization to a document vector
    fn normalize_vector(&self, vector: &mut Array1<f64>) -> Result<()> {
        if let Some(ref norm) = self.norm {
            match norm.as_str() {
                "l1" => {
                    let sum = vector.sum();
                    if sum > 0.0 {
                        vector.mapv_inplace(|x| x / sum);
                    }
                }
                "l2" => {
                    let squared_sum: f64 = vector.iter().map(|&x| x * x).sum();
                    if squared_sum > 0.0 {
                        let norm = squared_sum.sqrt();
                        vector.mapv_inplace(|x| x / norm);
                    }
                }
                _ => {
                    return Err(TextError::InvalidInput(format!(
                        "Unknown normalization: {}",
                        norm
                    )))
                }
            }
        }

        Ok(())
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new(false, true, Some("l2".to_string()))
    }
}

impl Vectorizer for TfidfVectorizer {
    fn fit(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for fitting".into(),
            ));
        }

        // First, fit the count vectorizer to build the vocabulary
        self.count_vectorizer.fit(texts)?;

        let n_documents = texts.len() as f64;
        let n_features = self.count_vectorizer.vocabulary_size();

        // Get document frequency for each term
        let mut df = Array1::zeros(n_features);

        for &text in texts {
            let tokens = self.count_vectorizer.tokenizer.tokenize(text)?;
            let mut seen_tokens = HashMap::new();

            // Count each token only once per document
            for token in tokens {
                if let Some(idx) = self.count_vectorizer.vocabulary.get_index(&token) {
                    seen_tokens.insert(idx, true);
                }
            }

            // Update document frequencies
            for idx in seen_tokens.keys() {
                df[*idx] += 1.0;
            }
        }

        // Compute IDF
        self.compute_idf(&df, n_documents)?;

        Ok(())
    }

    fn transform(&self, text: &str) -> Result<Array1<f64>> {
        if self.idf.is_none() {
            return Err(TextError::VocabularyError(
                "IDF values not computed. Call fit() first".into(),
            ));
        }

        // Get count vector
        let mut count_vector = self.count_vectorizer.transform(text)?;

        // Apply TF-IDF transformation
        let idf = self.idf.as_ref().unwrap();
        for i in 0..count_vector.len() {
            count_vector[i] *= idf[i];
        }

        // Apply normalization if requested
        self.normalize_vector(&mut count_vector)?;

        Ok(count_vector)
    }

    fn transform_batch(&self, texts: &[&str]) -> Result<Array2<f64>> {
        if self.idf.is_none() {
            return Err(TextError::VocabularyError(
                "IDF values not computed. Call fit() first".into(),
            ));
        }

        if texts.is_empty() {
            return Ok(Array2::zeros((0, self.count_vectorizer.vocabulary_size())));
        }

        // Get count vectors
        let mut count_matrix = self.count_vectorizer.transform_batch(texts)?;

        // Apply TF-IDF transformation
        let idf = self.idf.as_ref().unwrap();
        for mut row in count_matrix.axis_iter_mut(Axis(0)) {
            for i in 0..row.len() {
                row[i] *= idf[i];
            }

            // Apply normalization if requested
            if let Some(ref norm) = self.norm {
                match norm.as_str() {
                    "l1" => {
                        let sum = row.sum();
                        if sum > 0.0 {
                            row.mapv_inplace(|x| x / sum);
                        }
                    }
                    "l2" => {
                        let squared_sum: f64 = row.iter().map(|&x| x * x).sum();
                        if squared_sum > 0.0 {
                            let norm = squared_sum.sqrt();
                            row.mapv_inplace(|x| x / norm);
                        }
                    }
                    _ => {
                        return Err(TextError::InvalidInput(format!(
                            "Unknown normalization: {}",
                            norm
                        )))
                    }
                }
            }
        }

        Ok(count_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_vectorizer() {
        let mut vectorizer = CountVectorizer::default();
        let corpus = [
            "This is the first document.",
            "This document is the second document.",
        ];

        // Fit the vectorizer
        vectorizer.fit(&corpus).unwrap();

        // Check vocabulary
        assert_eq!(vectorizer.vocabulary_size(), 6);

        // Transform a document
        let vec = vectorizer.transform(corpus[0]).unwrap();
        assert_eq!(vec.len(), 6);

        // Check that document frequencies are correct
        let vec_sum: f64 = vec.iter().sum();
        assert_eq!(vec_sum, 5.0); // 5 tokens in the first document
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let mut vectorizer = TfidfVectorizer::default();
        let corpus = [
            "This is the first document.",
            "This document is the second document.",
        ];

        // Fit the vectorizer
        vectorizer.fit(&corpus).unwrap();

        // Transform a document
        let vec = vectorizer.transform(corpus[0]).unwrap();
        assert_eq!(vec.len(), 6);

        // Check that the vector is normalized (using L2 norm)
        let norm: f64 = vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_vectorizer() {
        let mut vectorizer = CountVectorizer::new(true);
        let corpus = ["this this this is a document", "this is another document"];

        // Fit and transform
        let matrix = vectorizer.fit_transform(&corpus).unwrap();

        // First document should have binary values (all 1.0 or 0.0)
        for val in matrix.row(0).iter() {
            assert!(*val == 0.0 || *val == 1.0);
        }
    }
}
