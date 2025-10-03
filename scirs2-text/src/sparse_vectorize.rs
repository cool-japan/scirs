//! Sparse vectorization for memory-efficient text representation
//!
//! This module provides sparse implementations of text vectorizers
//! that use memory-efficient sparse matrix representations.

use crate::error::{Result, TextError};
use crate::sparse::{CsrMatrix, SparseMatrixBuilder, SparseVector};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Sparse count vectorizer using CSR matrix representation
pub struct SparseCountVectorizer {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    vocabulary: Vocabulary,
    binary: bool,
}

impl Clone for SparseCountVectorizer {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone_box(),
            vocabulary: self.vocabulary.clone(),
            binary: self.binary,
        }
    }
}

impl SparseCountVectorizer {
    /// Create a new sparse count vectorizer
    pub fn new(binary: bool) -> Self {
        Self {
            tokenizer: Box::new(WordTokenizer::default()),
            vocabulary: Vocabulary::new(),
            binary,
        }
    }

    /// Create with a custom tokenizer
    pub fn with_tokenizer(tokenizer: Box<dyn Tokenizer + Send + Sync>, binary: bool) -> Self {
        Self {
            tokenizer,
            vocabulary: Vocabulary::new(),
            binary,
        }
    }

    /// Fit the vectorizer on a corpus
    pub fn fit(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for fitting".into(),
            ));
        }

        self.vocabulary = Vocabulary::new();

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            for token in tokens {
                self.vocabulary.add_token(&token);
            }
        }

        Ok(())
    }

    /// Transform a single text into a sparse vector
    pub fn transform(&self, text: &str) -> Result<SparseVector> {
        let tokens = self.tokenizer.tokenize(text)?;
        let mut counts: HashMap<usize, f64> = HashMap::new();

        for token in tokens {
            if let Some(idx) = self.vocabulary.get_index(&token) {
                *counts.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        // Sort indices for efficient sparse operations
        let mut indices: Vec<usize> = counts.keys().copied().collect();
        indices.sort_unstable();

        let values: Vec<f64> = if self.binary {
            indices.iter().map(|_| 1.0).collect()
        } else {
            indices.iter().map(|&idx| counts[&idx]).collect()
        };

        let sparse_vec = SparseVector::fromindices_values(indices, values, self.vocabulary.len());

        Ok(sparse_vec)
    }

    /// Transform a batch of texts into a sparse matrix
    pub fn transform_batch(&self, texts: &[&str]) -> Result<CsrMatrix> {
        let n_cols = self.vocabulary.len();
        let mut builder = SparseMatrixBuilder::new(n_cols);

        for &text in texts {
            let sparse_vec = self.transform(text)?;
            builder.add_row(sparse_vec)?;
        }

        Ok(builder.build())
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, texts: &[&str]) -> Result<CsrMatrix> {
        self.fit(texts)?;
        self.transform_batch(texts)
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
}

/// Sparse TF-IDF vectorizer
#[derive(Clone)]
pub struct SparseTfidfVectorizer {
    count_vectorizer: SparseCountVectorizer,
    idf: Option<Array1<f64>>,
    useidf: bool,
    norm: Option<String>,
}

impl SparseTfidfVectorizer {
    /// Create a new sparse TF-IDF vectorizer
    pub fn new() -> Self {
        Self {
            count_vectorizer: SparseCountVectorizer::new(false),
            idf: None,
            useidf: true,
            norm: Some("l2".to_string()),
        }
    }

    /// Create with custom settings
    pub fn with_settings(useidf: bool, norm: Option<String>) -> Self {
        Self {
            count_vectorizer: SparseCountVectorizer::new(false),
            idf: None,
            useidf,
            norm,
        }
    }

    /// Create with a custom tokenizer
    pub fn with_tokenizer(tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        Self {
            count_vectorizer: SparseCountVectorizer::with_tokenizer(tokenizer, false),
            idf: None,
            useidf: true,
            norm: Some("l2".to_string()),
        }
    }

    /// Fit the vectorizer on a corpus
    pub fn fit(&mut self, texts: &[&str]) -> Result<()> {
        self.count_vectorizer.fit(texts)?;

        if self.useidf {
            // Calculate IDF values
            let n_docs = texts.len() as f64;
            let vocab_size = self.count_vectorizer.vocabulary_size();
            let mut doc_freq = vec![0.0; vocab_size];

            // Count document frequencies
            for &text in texts {
                let sparse_vec = self.count_vectorizer.transform(text)?;
                for &idx in sparse_vec.indices() {
                    doc_freq[idx] += 1.0;
                }
            }

            // Calculate IDF values: log(n_docs / df) + 1
            let mut idf_values = Array1::zeros(vocab_size);
            for (idx, &df) in doc_freq.iter().enumerate() {
                if df > 0.0 {
                    idf_values[idx] = (n_docs / df).ln() + 1.0;
                } else {
                    idf_values[idx] = 1.0;
                }
            }

            self.idf = Some(idf_values);
        }

        Ok(())
    }

    /// Transform a single text into a sparse TF-IDF vector
    pub fn transform(&self, text: &str) -> Result<SparseVector> {
        let mut sparse_vec = self.count_vectorizer.transform(text)?;

        // Apply IDF weighting if enabled
        if self.useidf {
            if let Some(ref idf) = self.idf {
                let indices_copy: Vec<usize> = sparse_vec.indices().to_vec();
                let values = sparse_vec.values_mut();
                for (i, &idx) in indices_copy.iter().enumerate() {
                    values[i] *= idf[idx];
                }
            }
        }

        // Apply normalization if specified
        if let Some(ref norm_type) = self.norm {
            match norm_type.as_str() {
                "l2" => {
                    let norm = sparse_vec.norm();
                    if norm > 0.0 {
                        sparse_vec.scale(1.0 / norm);
                    }
                }
                "l1" => {
                    let sum: f64 = sparse_vec.values().iter().map(|x| x.abs()).sum();
                    if sum > 0.0 {
                        sparse_vec.scale(1.0 / sum);
                    }
                }
                _ => {
                    return Err(TextError::InvalidInput(format!(
                        "Unknown normalization type: {norm_type}"
                    )));
                }
            }
        }

        Ok(sparse_vec)
    }

    /// Transform a batch of texts into a sparse TF-IDF matrix
    pub fn transform_batch(&self, texts: &[&str]) -> Result<CsrMatrix> {
        let n_cols = self.count_vectorizer.vocabulary_size();
        let mut builder = SparseMatrixBuilder::new(n_cols);

        for &text in texts {
            let sparse_vec = self.transform(text)?;
            builder.add_row(sparse_vec)?;
        }

        Ok(builder.build())
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, texts: &[&str]) -> Result<CsrMatrix> {
        self.fit(texts)?;
        self.transform_batch(texts)
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.count_vectorizer.vocabulary_size()
    }

    /// Get the vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        self.count_vectorizer.vocabulary()
    }

    /// Get the IDF values
    pub fn idf_values(&self) -> Option<&Array1<f64>> {
        self.idf.as_ref()
    }
}

impl Default for SparseTfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between sparse vectors
#[allow(dead_code)]
pub fn sparse_cosine_similarity(v1: &SparseVector, v2: &SparseVector) -> Result<f64> {
    if v1.size() != v2.size() {
        return Err(TextError::InvalidInput(format!(
            "Vector dimensions don't match: {} vs {}",
            v1.size(),
            v2.size()
        )));
    }

    let dot = v1.dotsparse(v2)?;
    let norm1 = v1.norm();
    let norm2 = v2.norm();

    if norm1 == 0.0 || norm2 == 0.0 {
        Ok(if norm1 == norm2 { 1.0 } else { 0.0 })
    } else {
        Ok(dot / (norm1 * norm2))
    }
}

/// Memory usage statistics for sparse representation
pub struct MemoryStats {
    /// Memory used by sparse representation in bytes
    pub sparse_bytes: usize,
    /// Memory that would be used by dense representation in bytes
    pub dense_bytes: usize,
    /// Compression ratio (dense_bytes / sparse_bytes)
    pub compression_ratio: f64,
    /// Sparsity ratio (number of zeros / total elements)
    pub sparsity: f64,
}

impl MemoryStats {
    /// Calculate memory statistics for a sparse matrix
    pub fn from_sparse_matrix(sparse: &CsrMatrix) -> Self {
        let (n_rows, n_cols) = sparse.shape();
        let dense_bytes = n_rows * n_cols * std::mem::size_of::<f64>();
        let sparse_bytes = sparse.memory_usage();
        let total_elements = n_rows * n_cols;
        let nnz = sparse.nnz();

        Self {
            sparse_bytes,
            dense_bytes,
            compression_ratio: dense_bytes as f64 / sparse_bytes as f64,
            sparsity: 1.0 - (nnz as f64 / total_elements as f64),
        }
    }

    /// Print memory statistics
    pub fn print_stats(&self) {
        println!("Memory Usage Statistics:");
        println!("  Sparse representation: {} bytes", self.sparse_bytes);
        println!("  Dense representation: {} bytes", self.dense_bytes);
        println!("  Compression ratio: {:.2}x", self.compression_ratio);
        println!("  Sparsity: {:.1}%", self.sparsity * 100.0);
        println!(
            "  Memory saved: {:.1}%",
            (1.0 - 1.0 / self.compression_ratio) * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_count_vectorizer() {
        // Use larger, sparser data to ensure compression benefits
        let texts = vec![
            "this is a test document with some unique words",
            "this is another test document with different vocabulary",
            "yet another example document with more text content",
            "completely different text with various other terms",
            "final document in the test set with distinct words",
        ];

        let mut vectorizer = SparseCountVectorizer::new(false);
        let sparse_matrix = vectorizer.fit_transform(&texts).unwrap();

        assert_eq!(sparse_matrix.shape().0, 5); // 5 documents
        assert!(sparse_matrix.nnz() > 0);

        // Check memory efficiency - with larger vocabulary, sparse should be more efficient
        let stats = MemoryStats::from_sparse_matrix(&sparse_matrix);
        // For small test data, just verify it's calculated properly
        assert!(stats.compression_ratio > 0.0);
        assert!(stats.sparsity >= 0.0);
    }

    #[test]
    fn test_sparse_tfidf_vectorizer() {
        let texts = vec!["the quick brown fox", "the lazy dog", "brown fox jumps"];

        let mut vectorizer = SparseTfidfVectorizer::new();
        let sparse_matrix = vectorizer.fit_transform(&texts).unwrap();

        assert_eq!(sparse_matrix.shape().0, 3);

        // Verify TF-IDF properties
        let first_doc = sparse_matrix.get_row(0).unwrap();
        assert!(first_doc.norm() > 0.0);

        // With L2 normalization, the norm should be approximately 1
        assert!((first_doc.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_cosine_similarity() {
        let v1 = SparseVector::fromindices_values(vec![0, 2, 3], vec![1.0, 2.0, 3.0], 5);

        let v2 = SparseVector::fromindices_values(vec![1, 2, 4], vec![1.0, 2.0, 1.0], 5);

        let similarity = sparse_cosine_similarity(&v1, &v2).unwrap();

        // Only index 2 overlaps with value 2.0 in both
        // v1 dot v2 = 2.0 * 2.0 = 4.0
        // |v1| = sqrt(1 + 4 + 9) = sqrt(14)
        // |v2| = sqrt(1 + 4 + 1) = sqrt(6)
        // cos = 4 / (sqrt(14) * sqrt(6))
        let expected = 4.0 / (14.0_f64.sqrt() * 6.0_f64.sqrt());
        assert!((similarity - expected).abs() < 1e-10);
    }

    #[test]
    fn test_memory_efficiency_large() {
        // Create a large corpus with sparse content
        let texts: Vec<String> = (0..100)
            .map(|i| {
                let word_idx = i % 10;
                format!("document {i} contains word{word_idx}")
            })
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

        let mut vectorizer = SparseCountVectorizer::new(false);
        let sparse_matrix = vectorizer.fit_transform(&text_refs).unwrap();

        let stats = MemoryStats::from_sparse_matrix(&sparse_matrix);
        stats.print_stats();

        // Should achieve significant compression for sparse data
        assert!(stats.compression_ratio > 5.0);
        assert!(stats.sparsity > 0.8);
    }
}
