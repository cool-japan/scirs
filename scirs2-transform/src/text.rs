//! Text processing transformers for feature extraction
//!
//! This module provides utilities for converting text data into numerical features
//! suitable for machine learning algorithms.

use ahash::AHasher;
use regex::Regex;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::error::{Result, TransformError};

/// Count vectorizer for converting text documents to term frequency vectors
pub struct CountVectorizer {
    /// Vocabulary mapping from terms to indices
    vocabulary: HashMap<String, usize>,
    /// Inverse vocabulary mapping from indices to terms
    feature_names: Vec<String>,
    /// Maximum number of features
    max_features: Option<usize>,
    /// Minimum document frequency
    min_df: f64,
    /// Maximum document frequency
    max_df: f64,
    /// Whether to convert to lowercase
    lowercase: bool,
    /// Token pattern regex
    token_pattern: Regex,
    /// Set of stop words to exclude
    stop_words: HashSet<String>,
    /// Whether the vectorizer has been fitted
    fitted: bool,
}

impl CountVectorizer {
    /// Create a new count vectorizer
    pub fn new() -> Self {
        CountVectorizer {
            vocabulary: HashMap::new(),
            feature_names: Vec::new(),
            max_features: None,
            min_df: 1.0,
            max_df: 1.0,
            lowercase: true,
            token_pattern: Regex::new(r"\b\w+\b").unwrap(),
            stop_words: HashSet::new(),
            fitted: false,
        }
    }

    /// Set maximum number of features
    #[allow(dead_code)]
    pub fn with_max_features(mut self, maxfeatures: usize) -> Self {
        self.max_features = Some(maxfeatures);
        self
    }

    /// Set minimum document frequency
    #[allow(dead_code)]
    pub fn with_min_df(mut self, mindf: f64) -> Self {
        self.min_df = mindf;
        self
    }

    /// Set maximum document frequency
    #[allow(dead_code)]
    pub fn with_max_df(mut self, maxdf: f64) -> Self {
        self.max_df = maxdf;
        self
    }

    /// Set whether to convert to lowercase
    #[allow(dead_code)]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set custom token pattern
    #[allow(dead_code)]
    pub fn with_token_pattern(mut self, pattern: &str) -> Result<Self> {
        self.token_pattern = Regex::new(pattern)
            .map_err(|e| TransformError::InvalidInput(format!("Invalid regex pattern: {e}")))?;
        Ok(self)
    }

    /// Set stop words
    #[allow(dead_code)]
    pub fn with_stop_words(mut self, stopwords: Vec<String>) -> Self {
        self.stop_words = stopwords.into_iter().collect();
        self
    }

    /// Tokenize a document
    fn tokenize(&self, doc: &str) -> Vec<String> {
        let text = if self.lowercase {
            doc.to_lowercase()
        } else {
            doc.to_string()
        };

        self.token_pattern
            .find_iter(&text)
            .map(|m| m.as_str().to_string())
            .filter(|token| !self.stop_words.contains(token))
            .collect()
    }

    /// Fit the vectorizer on a collection of documents
    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        if documents.is_empty() {
            return Err(TransformError::InvalidInput(
                "Empty document collection".into(),
            ));
        }

        // Count term frequencies across all documents
        let mut term_doc_freq: HashMap<String, usize> = HashMap::new();
        let n_docs = documents.len();

        for doc in documents {
            let tokens: HashSet<String> = self.tokenize(doc).into_iter().collect();
            for token in tokens {
                *term_doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let min_df_count = if self.min_df <= 1.0 {
            self.min_df as usize
        } else {
            (self.min_df * n_docs as f64).ceil() as usize
        };

        let max_df_count = if self.max_df <= 1.0 {
            (self.max_df * n_docs as f64).floor() as usize
        } else {
            self.max_df as usize
        };

        let mut filtered_terms: Vec<(String, usize)> = term_doc_freq
            .into_iter()
            .filter(|(_, freq)| *freq >= min_df_count && *freq <= max_df_count)
            .collect();

        // Sort by document frequency (descending) for max_features selection
        filtered_terms.sort_by(|a, b| b.1.cmp(&a.1));

        // Limit to max_features if specified
        if let Some(max_feat) = self.max_features {
            filtered_terms.truncate(max_feat);
        }

        // Build vocabulary
        self.vocabulary.clear();
        self.feature_names.clear();

        for (idx, (term, _freq)) in filtered_terms.into_iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
            self.feature_names.push(term);
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform documents to count vectors
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(TransformError::NotFitted(
                "CountVectorizer must be fitted before transform".into(),
            ));
        }

        let n_samples = documents.len();
        let n_features = self.vocabulary.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, doc) in documents.iter().enumerate() {
            let tokens = self.tokenize(doc);
            for token in tokens {
                if let Some(&idx) = self.vocabulary.get(&token) {
                    result[[i, idx]] += 1.0;
                }
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Get feature names
    #[allow(dead_code)]
    pub fn get_feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// TF-IDF vectorizer for converting text to TF-IDF features
pub struct TfidfVectorizer {
    /// Underlying count vectorizer
    count_vectorizer: CountVectorizer,
    /// IDF values for each feature
    idf: Array1<f64>,
    /// Whether to use IDF weighting
    use_idf: bool,
    /// Whether to apply L2 normalization
    norm: bool,
    /// Whether to add 1 to document frequencies
    smooth_idf: bool,
    /// Whether to subtract 1 from IDF
    sublinear_tf: bool,
}

impl TfidfVectorizer {
    /// Create a new TF-IDF vectorizer
    pub fn new() -> Self {
        TfidfVectorizer {
            count_vectorizer: CountVectorizer::new(),
            idf: Array1::zeros(0),
            use_idf: true,
            norm: true,
            smooth_idf: true,
            sublinear_tf: false,
        }
    }

    /// Set whether to use IDF weighting
    #[allow(dead_code)]
    pub fn with_use_idf(mut self, useidf: bool) -> Self {
        self.use_idf = useidf;
        self
    }

    /// Set whether to apply L2 normalization
    #[allow(dead_code)]
    pub fn with_norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }

    /// Set whether to smooth IDF weights
    #[allow(dead_code)]
    pub fn with_smooth_idf(mut self, smoothidf: bool) -> Self {
        self.smooth_idf = smoothidf;
        self
    }

    /// Set whether to use sublinear term frequency
    #[allow(dead_code)]
    pub fn with_sublinear_tf(mut self, sublineartf: bool) -> Self {
        self.sublinear_tf = sublineartf;
        self
    }

    /// Configure the underlying count vectorizer
    #[allow(dead_code)]
    pub fn configure_count_vectorizer<F>(mut self, f: F) -> Self
    where
        F: FnOnce(CountVectorizer) -> CountVectorizer,
    {
        self.count_vectorizer = f(self.count_vectorizer);
        self
    }

    /// Fit the vectorizer on documents
    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        // Fit count vectorizer
        self.count_vectorizer.fit(documents)?;

        if self.use_idf {
            // Calculate IDF values
            let n_samples = documents.len() as f64;
            let n_features = self.count_vectorizer.vocabulary.len();
            let mut df = Array1::zeros(n_features);

            // Count document frequencies
            for doc in documents {
                let tokens: HashSet<String> =
                    self.count_vectorizer.tokenize(doc).into_iter().collect();
                for token in tokens {
                    if let Some(&idx) = self.count_vectorizer.vocabulary.get(&token) {
                        df[idx] += 1.0;
                    }
                }
            }

            // Calculate IDF
            if self.smooth_idf {
                self.idf = df.mapv(|d: f64| ((n_samples + 1.0) / (d + 1.0)).ln() + 1.0);
            } else {
                self.idf = df.mapv(|d: f64| (n_samples / d).ln() + 1.0);
            }
        }

        Ok(())
    }

    /// Transform documents to TF-IDF vectors
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        // Get count vectors
        let mut x = self.count_vectorizer.transform(documents)?;

        // Apply sublinear TF scaling
        if self.sublinear_tf {
            x.mapv_inplace(|v| if v > 0.0 { 1.0 + v.ln() } else { 0.0 });
        }

        // Apply IDF weighting
        if self.use_idf {
            for i in 0..x.shape()[0] {
                for j in 0..x.shape()[1] {
                    x[[i, j]] *= self.idf[j];
                }
            }
        }

        // Apply L2 normalization
        if self.norm {
            for i in 0..x.shape()[0] {
                let row = x.row(i);
                let norm = row.dot(&row).sqrt();
                if norm > 0.0 {
                    x.row_mut(i).mapv_inplace(|v| v / norm);
                }
            }
        }

        Ok(x)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    /// Get feature names
    #[allow(dead_code)]
    pub fn get_feature_names(&self) -> &[String] {
        self.count_vectorizer.get_feature_names()
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Hashing vectorizer for memory-efficient text vectorization
pub struct HashingVectorizer {
    /// Number of features (hash space size)
    n_features: usize,
    /// Whether to convert to lowercase
    lowercase: bool,
    /// Token pattern regex
    token_pattern: Regex,
    /// Whether to use binary occurrence instead of counts
    binary: bool,
    /// Norm to use for normalization
    norm: Option<String>,
}

impl HashingVectorizer {
    /// Create a new hashing vectorizer
    pub fn new(_nfeatures: usize) -> Self {
        HashingVectorizer {
            n_features: _nfeatures,
            lowercase: true,
            token_pattern: Regex::new(r"\b\w+\b").unwrap(),
            binary: false,
            norm: Some("l2".to_string()),
        }
    }

    /// Set whether to use binary occurrence
    #[allow(dead_code)]
    pub fn with_binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Set normalization method
    #[allow(dead_code)]
    pub fn with_norm(mut self, norm: Option<String>) -> Self {
        self.norm = norm;
        self
    }

    /// Set whether to convert to lowercase
    #[allow(dead_code)]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Hash a token to a feature index
    fn hash_token(&self, token: &str) -> usize {
        let mut hasher = AHasher::default();
        token.hash(&mut hasher);
        (hasher.finish() as usize) % self.n_features
    }

    /// Tokenize a document
    fn tokenize(&self, doc: &str) -> Vec<String> {
        let text = if self.lowercase {
            doc.to_lowercase()
        } else {
            doc.to_string()
        };

        self.token_pattern
            .find_iter(&text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Transform documents to hashed feature vectors
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let n_samples = documents.len();
        let mut result = Array2::zeros((n_samples, self.n_features));

        for (i, doc) in documents.iter().enumerate() {
            let tokens = self.tokenize(doc);

            if self.binary {
                let unique_indices: HashSet<usize> =
                    tokens.iter().map(|token| self.hash_token(token)).collect();

                for idx in unique_indices {
                    result[[i, idx]] = 1.0;
                }
            } else {
                for token in tokens {
                    let idx = self.hash_token(&token);
                    result[[i, idx]] += 1.0;
                }
            }

            // Apply normalization
            if let Some(ref norm_type) = self.norm {
                let row = result.row(i).to_owned();
                let norm_value = match norm_type.as_str() {
                    "l1" => row.iter().map(|v: &f64| v.abs()).sum::<f64>(),
                    "l2" => row.dot(&row).sqrt(),
                    _ => continue,
                };

                if norm_value > 0.0 {
                    result.row_mut(i).mapv_inplace(|v| v / norm_value);
                }
            }
        }

        Ok(result)
    }
}

/// Streaming count vectorizer that can learn vocabulary incrementally
pub struct StreamingCountVectorizer {
    /// Current vocabulary
    vocabulary: HashMap<String, usize>,
    /// Document frequency counts
    doc_freq: HashMap<String, usize>,
    /// Number of documents seen
    n_docs_seen: usize,
    /// Maximum vocabulary size
    max_features: Option<usize>,
    /// Whether to convert to lowercase
    lowercase: bool,
    /// Token pattern regex
    token_pattern: Regex,
}

impl StreamingCountVectorizer {
    /// Create a new streaming count vectorizer
    pub fn new() -> Self {
        StreamingCountVectorizer {
            vocabulary: HashMap::new(),
            doc_freq: HashMap::new(),
            n_docs_seen: 0,
            max_features: None,
            lowercase: true,
            token_pattern: Regex::new(r"\b\w+\b").unwrap(),
        }
    }

    /// Set maximum vocabulary size
    #[allow(dead_code)]
    pub fn with_max_features(mut self, maxfeatures: usize) -> Self {
        self.max_features = Some(maxfeatures);
        self
    }

    /// Tokenize a document
    fn tokenize(&self, doc: &str) -> Vec<String> {
        let text = if self.lowercase {
            doc.to_lowercase()
        } else {
            doc.to_string()
        };

        self.token_pattern
            .find_iter(&text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Update vocabulary with new documents
    pub fn partial_fit(&mut self, documents: &[String]) -> Result<()> {
        for doc in documents {
            self.n_docs_seen += 1;
            let tokens: HashSet<String> = self.tokenize(doc).into_iter().collect();

            for token in tokens {
                *self.doc_freq.entry(token.clone()).or_insert(0) += 1;

                if !self.vocabulary.contains_key(&token) {
                    if let Some(max_feat) = self.max_features {
                        if self.vocabulary.len() >= max_feat {
                            // Find least frequent term to replace
                            if let Some((min_token_, _)) = self
                                .vocabulary
                                .iter()
                                .min_by_key(|(t, _)| self.doc_freq.get(*t).unwrap_or(&0))
                            {
                                let min_token = min_token_.clone();
                                let min_freq = self.doc_freq.get(&min_token).unwrap_or(&0);
                                let new_freq = self.doc_freq.get(&token).unwrap_or(&0);

                                if new_freq > min_freq {
                                    let old_idx = self.vocabulary.remove(&min_token).unwrap();
                                    self.vocabulary.insert(token, old_idx);
                                }
                            }
                        } else {
                            self.vocabulary.insert(token, self.vocabulary.len());
                        }
                    } else {
                        self.vocabulary.insert(token, self.vocabulary.len());
                    }
                }
            }
        }

        Ok(())
    }

    /// Transform documents using current vocabulary
    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let n_samples = documents.len();
        let n_features = self.vocabulary.len();

        if n_features == 0 {
            return Err(TransformError::NotFitted(
                "No vocabulary learned yet".into(),
            ));
        }

        let mut result = Array2::zeros((n_samples, n_features));

        for (i, doc) in documents.iter().enumerate() {
            let tokens = self.tokenize(doc);
            for token in tokens {
                if let Some(&idx) = self.vocabulary.get(&token) {
                    result[[i, idx]] += 1.0;
                }
            }
        }

        Ok(result)
    }
}

impl Default for StreamingCountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}
