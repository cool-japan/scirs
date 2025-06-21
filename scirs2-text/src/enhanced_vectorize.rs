//! Enhanced text vectorization with n-gram support
//!
//! This module provides enhanced vectorizers with n-gram support,
//! additional preprocessing options, and IDF smoothing.

use crate::error::{Result, TextError};
use crate::tokenize::{NgramTokenizer, Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Enhanced count vectorizer with n-gram support
pub struct EnhancedCountVectorizer {
    vocabulary: Vocabulary,
    binary: bool,
    ngram_range: (usize, usize),
    max_features: Option<usize>,
    min_df: f64,
    max_df: f64,
    lowercase: bool,
}

impl EnhancedCountVectorizer {
    /// Create a new enhanced count vectorizer
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            binary: false,
            ngram_range: (1, 1),
            max_features: None,
            min_df: 0.0,
            max_df: 1.0,
            lowercase: true,
        }
    }

    /// Set whether to produce binary vectors
    pub fn set_binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Set the n-gram range (min_n, max_n)
    pub fn set_ngram_range(mut self, range: (usize, usize)) -> Result<Self> {
        if range.0 == 0 || range.1 < range.0 {
            return Err(TextError::InvalidInput(
                "Invalid n-gram range. Must have min_n > 0 and max_n >= min_n".to_string(),
            ));
        }
        self.ngram_range = range;
        Ok(self)
    }

    /// Set the maximum number of features
    pub fn set_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the minimum document frequency
    pub fn set_min_df(mut self, min_df: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&min_df) {
            return Err(TextError::InvalidInput(
                "min_df must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.min_df = min_df;
        Ok(self)
    }

    /// Set the maximum document frequency
    pub fn set_max_df(mut self, max_df: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&max_df) {
            return Err(TextError::InvalidInput(
                "max_df must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.max_df = max_df;
        Ok(self)
    }

    /// Set whether to lowercase text
    pub fn set_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Fit the vectorizer on a corpus
    pub fn fit(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for fitting".to_string(),
            ));
        }

        // Clear existing vocabulary
        self.vocabulary = Vocabulary::new();

        // Track document frequencies
        let mut doc_frequencies: HashMap<String, usize> = HashMap::new();
        let total_docs = texts.len();

        // Process each document
        for text in texts {
            let mut seen_in_doc: HashMap<String, bool> = HashMap::new();

            // Extract all n-grams in the range
            let all_tokens = self.extract_ngrams(text)?;

            // Count document frequencies
            for token in all_tokens {
                if !seen_in_doc.contains_key(&token) {
                    *doc_frequencies.entry(token.clone()).or_insert(0) += 1;
                    seen_in_doc.insert(token.clone(), true);
                }

                // Add to vocabulary (will handle max_size internally)
                self.vocabulary.add_token(&token);
            }
        }

        // Filter by document frequency
        let min_count = (self.min_df * total_docs as f64).ceil() as usize;
        let max_count = (self.max_df * total_docs as f64).floor() as usize;

        let mut filtered_tokens: Vec<(String, usize)> = doc_frequencies
            .into_iter()
            .filter(|(_, count)| *count >= min_count && *count <= max_count)
            .collect();

        // Sort by frequency and limit features if needed
        filtered_tokens.sort_by(|a, b| b.1.cmp(&a.1));

        if let Some(max_features) = self.max_features {
            filtered_tokens.truncate(max_features);
        }

        // Rebuild vocabulary with filtered tokens
        self.vocabulary = Vocabulary::with_max_size(self.max_features.unwrap_or(usize::MAX));
        for (token, _) in filtered_tokens {
            self.vocabulary.add_token(&token);
        }

        Ok(())
    }

    /// Extract n-grams from text based on the configured range
    fn extract_ngrams(&self, text: &str) -> Result<Vec<String>> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // If range is (1, 1), just use word tokenizer
        let all_ngrams = if self.ngram_range == (1, 1) {
            let tokenizer = WordTokenizer::new(false);
            tokenizer.tokenize(&text)?
        } else {
            // Use n-gram tokenizer for the range
            let ngram_tokenizer =
                NgramTokenizer::with_range(self.ngram_range.0, self.ngram_range.1)?;
            ngram_tokenizer.tokenize(&text)?
        };

        Ok(all_ngrams)
    }

    /// Transform text into a count vector
    pub fn transform(&self, text: &str) -> Result<Array1<f64>> {
        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "Vocabulary is empty. Call fit() first".to_string(),
            ));
        }

        let vocab_size = self.vocabulary.len();
        let mut vector = Array1::zeros(vocab_size);

        // Extract n-grams
        let tokens = self.extract_ngrams(text)?;

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

    /// Transform multiple texts into a count matrix
    pub fn transform_batch(&self, texts: &[&str]) -> Result<Array2<f64>> {
        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "Vocabulary is empty. Call fit() first".to_string(),
            ));
        }

        let n_samples = texts.len();
        let vocab_size = self.vocabulary.len();
        let mut matrix = Array2::zeros((n_samples, vocab_size));

        for (i, text) in texts.iter().enumerate() {
            let vector = self.transform(text)?;
            matrix.row_mut(i).assign(&vector);
        }

        Ok(matrix)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, texts: &[&str]) -> Result<Array2<f64>> {
        self.fit(texts)?;
        self.transform_batch(texts)
    }
}

impl Default for EnhancedCountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced TF-IDF vectorizer with IDF smoothing options
pub struct EnhancedTfidfVectorizer {
    count_vectorizer: EnhancedCountVectorizer,
    use_idf: bool,
    smooth_idf: bool,
    sublinear_tf: bool,
    norm: Option<String>,
    idf_: Option<Array1<f64>>,
}

impl EnhancedTfidfVectorizer {
    /// Create a new enhanced TF-IDF vectorizer
    pub fn new() -> Self {
        Self {
            count_vectorizer: EnhancedCountVectorizer::new(),
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            norm: Some("l2".to_string()),
            idf_: None,
        }
    }

    /// Set whether to use IDF weighting
    pub fn set_use_idf(mut self, use_idf: bool) -> Self {
        self.use_idf = use_idf;
        self
    }

    /// Set whether to smooth IDF weights
    pub fn set_smooth_idf(mut self, smooth_idf: bool) -> Self {
        self.smooth_idf = smooth_idf;
        self
    }

    /// Set whether to use sublinear TF scaling
    pub fn set_sublinear_tf(mut self, sublinear_tf: bool) -> Self {
        self.sublinear_tf = sublinear_tf;
        self
    }

    /// Set the normalization method (None, "l1", or "l2")
    pub fn set_norm(mut self, norm: Option<String>) -> Result<Self> {
        if let Some(ref n) = norm {
            if n != "l1" && n != "l2" {
                return Err(TextError::InvalidInput(
                    "Norm must be 'l1', 'l2', or None".to_string(),
                ));
            }
        }
        self.norm = norm;
        Ok(self)
    }

    /// Set n-gram range
    pub fn set_ngram_range(mut self, range: (usize, usize)) -> Result<Self> {
        self.count_vectorizer = self.count_vectorizer.set_ngram_range(range)?;
        Ok(self)
    }

    /// Set maximum features
    pub fn set_max_features(mut self, max_features: Option<usize>) -> Self {
        self.count_vectorizer = self.count_vectorizer.set_max_features(max_features);
        self
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        self.count_vectorizer.vocabulary()
    }

    /// Fit the vectorizer on a corpus
    pub fn fit(&mut self, texts: &[&str]) -> Result<()> {
        // Fit the count vectorizer
        self.count_vectorizer.fit(texts)?;

        if self.use_idf {
            // Calculate IDF weights
            self.calculate_idf(texts)?;
        }

        Ok(())
    }

    /// Calculate IDF weights
    fn calculate_idf(&mut self, texts: &[&str]) -> Result<()> {
        let vocab_size = self.count_vectorizer.vocabulary().len();
        let mut df: Array1<f64> = Array1::zeros(vocab_size);
        let n_samples = texts.len() as f64;

        // Count document frequencies
        for text in texts {
            let count_vec = self.count_vectorizer.transform(text)?;
            for (idx, &count) in count_vec.iter().enumerate() {
                if count > 0.0 {
                    df[idx] += 1.0;
                }
            }
        }

        // Calculate IDF
        let mut idf = Array1::zeros(vocab_size);
        for (idx, &doc_freq) in df.iter().enumerate() {
            if self.smooth_idf {
                idf[idx] = (1.0 + n_samples) / (1.0 + doc_freq);
            } else {
                idf[idx] = n_samples / doc_freq.max(1.0);
            }
            idf[idx] = idf[idx].ln() + 1.0;
        }

        self.idf_ = Some(idf);
        Ok(())
    }

    /// Transform text into a TF-IDF vector
    pub fn transform(&self, text: &str) -> Result<Array1<f64>> {
        // Get count vector
        let mut vector = self.count_vectorizer.transform(text)?;

        // Apply sublinear TF scaling if requested
        if self.sublinear_tf {
            for val in vector.iter_mut() {
                if *val > 0.0 {
                    *val = 1.0 + (*val).ln();
                }
            }
        }

        // Apply IDF weighting
        if self.use_idf {
            if let Some(ref idf) = self.idf_ {
                vector *= idf;
            } else {
                return Err(TextError::VocabularyError(
                    "IDF weights not calculated. Call fit() first".to_string(),
                ));
            }
        }

        // Apply normalization
        if let Some(ref norm) = self.norm {
            match norm.as_str() {
                "l1" => {
                    let norm_val = vector.iter().map(|x| x.abs()).sum::<f64>();
                    if norm_val > 0.0 {
                        vector /= norm_val;
                    }
                }
                "l2" => {
                    let norm_val = vector.dot(&vector).sqrt();
                    if norm_val > 0.0 {
                        vector /= norm_val;
                    }
                }
                _ => {}
            }
        }

        Ok(vector)
    }

    /// Transform multiple texts into a TF-IDF matrix
    pub fn transform_batch(&self, texts: &[&str]) -> Result<Array2<f64>> {
        let n_samples = texts.len();
        let vocab_size = self.count_vectorizer.vocabulary().len();
        let mut matrix = Array2::zeros((n_samples, vocab_size));

        for (i, text) in texts.iter().enumerate() {
            let vector = self.transform(text)?;
            matrix.row_mut(i).assign(&vector);
        }

        Ok(matrix)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, texts: &[&str]) -> Result<Array2<f64>> {
        self.fit(texts)?;
        self.transform_batch(texts)
    }
}

impl Default for EnhancedTfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_count_vectorizer_unigrams() {
        let mut vectorizer = EnhancedCountVectorizer::new();

        let documents = vec![
            "this is a test",
            "this is another test",
            "something different here",
        ];

        vectorizer.fit(&documents).unwrap();

        let vector = vectorizer.transform("this is a test").unwrap();
        assert!(!vector.is_empty());
    }

    #[test]
    fn test_enhanced_count_vectorizer_ngrams() {
        let mut vectorizer = EnhancedCountVectorizer::new()
            .set_ngram_range((1, 2))
            .unwrap();

        let documents = vec!["hello world", "hello there", "world peace"];

        vectorizer.fit(&documents).unwrap();

        // Should include both unigrams and bigrams
        let vocab = vectorizer.vocabulary();
        assert!(vocab.len() > 3); // More than just unigrams
    }

    #[test]
    fn test_enhanced_tfidf_vectorizer() {
        let mut vectorizer = EnhancedTfidfVectorizer::new()
            .set_smooth_idf(true)
            .set_norm(Some("l2".to_string()))
            .unwrap();

        let documents = vec![
            "this is a test",
            "this is another test",
            "something different here",
        ];

        vectorizer.fit(&documents).unwrap();

        let vector = vectorizer.transform("this is a test").unwrap();

        // Check L2 normalization
        let norm = vector.dot(&vector).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_features() {
        let mut vectorizer = EnhancedCountVectorizer::new().set_max_features(Some(5));

        let documents = vec![
            "one two three four five six seven eight",
            "one two three four five six seven eight nine ten",
        ];

        vectorizer.fit(&documents).unwrap();

        // Should only keep top 5 features
        assert_eq!(vectorizer.vocabulary().len(), 5);
    }

    #[test]
    fn test_document_frequency_filtering() {
        let mut vectorizer = EnhancedCountVectorizer::new().set_min_df(0.5).unwrap(); // Token must appear in at least 50% of docs

        let documents = vec![
            "common word rare",
            "common word unique",
            "common another distinct",
        ];

        vectorizer.fit(&documents).unwrap();

        // Only "common" should remain (appears in all docs)
        let vocab = vectorizer.vocabulary();
        assert!(vocab.contains("common"));
        assert!(!vocab.contains("rare"));
        assert!(!vocab.contains("unique"));
    }
}
