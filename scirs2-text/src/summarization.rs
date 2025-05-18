//! Text summarization module
//!
//! This module provides various algorithms for automatic text summarization.

use crate::error::{Result, TextError};
use crate::tokenize::Tokenizer;
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use ndarray::{Array1, Array2};
use std::collections::HashSet;

/// TextRank algorithm for extractive summarization
pub struct TextRank {
    /// Number of sentences to extract
    num_sentences: usize,
    /// Damping factor (usually 0.85)
    damping_factor: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence threshold
    threshold: f64,
    /// Tokenizer for sentence splitting
    sentence_tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl TextRank {
    /// Create a new TextRank summarizer
    pub fn new(num_sentences: usize) -> Self {
        Self {
            num_sentences,
            damping_factor: 0.85,
            max_iterations: 100,
            threshold: 0.0001,
            sentence_tokenizer: Box::new(crate::tokenize::SentenceTokenizer::new()),
        }
    }

    /// Set the damping factor
    pub fn with_damping_factor(mut self, damping_factor: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&damping_factor) {
            return Err(TextError::InvalidInput(
                "Damping factor must be between 0 and 1".to_string(),
            ));
        }
        self.damping_factor = damping_factor;
        Ok(self)
    }

    /// Extract summary from text
    pub fn summarize(&self, text: &str) -> Result<String> {
        let sentences: Vec<String> = self.sentence_tokenizer.tokenize(text)?;

        if sentences.is_empty() {
            return Ok(String::new());
        }

        if sentences.len() <= self.num_sentences {
            return Ok(text.to_string());
        }

        // Build similarity matrix
        let similarity_matrix = self.build_similarity_matrix(&sentences)?;

        // Apply PageRank algorithm
        let scores = self.page_rank(&similarity_matrix)?;

        // Select top sentences
        let selected_indices = self.select_top_sentences(&scores);

        // Reconstruct summary maintaining original order
        let summary = self.reconstruct_summary(&sentences, &selected_indices);

        Ok(summary)
    }

    /// Build similarity matrix between sentences
    fn build_similarity_matrix(&self, sentences: &[String]) -> Result<Array2<f64>> {
        let n = sentences.len();
        let mut matrix = Array2::zeros((n, n));

        // Use TF-IDF for sentence representation
        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_ref()).collect();
        let mut vectorizer = TfidfVectorizer::default();
        vectorizer.fit(&sentence_refs)?;
        let vectors = vectorizer.transform_batch(&sentence_refs)?;

        // Calculate cosine similarity between all pairs
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[[i, j]] = 0.0; // No self-loops
                } else {
                    let similarity = self
                        .cosine_similarity(vectors.row(i).to_owned(), vectors.row(j).to_owned());
                    matrix[[i, j]] = similarity;
                }
            }
        }

        Ok(matrix)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, vec1: Array1<f64>, vec2: Array1<f64>) -> f64 {
        let dot_product = vec1.dot(&vec2);
        let norm1 = vec1.dot(&vec1).sqrt();
        let norm2 = vec2.dot(&vec2).sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Apply PageRank algorithm
    fn page_rank(&self, matrix: &Array2<f64>) -> Result<Array1<f64>> {
        let n = matrix.nrows();
        let mut scores = Array1::from_elem(n, 1.0 / n as f64);

        // Normalize rows of similarity matrix
        let mut normalized_matrix = matrix.clone();
        for i in 0..n {
            let row_sum: f64 = matrix.row(i).sum();
            if row_sum > 0.0 {
                normalized_matrix.row_mut(i).mapv_inplace(|x| x / row_sum);
            }
        }

        // Iterate until convergence
        for _ in 0..self.max_iterations {
            let new_scores = Array1::from_elem(n, (1.0 - self.damping_factor) / n as f64)
                + self.damping_factor * normalized_matrix.t().dot(&scores);

            // Check convergence
            let diff = (&new_scores - &scores).mapv(f64::abs).sum();
            scores = new_scores;

            if diff < self.threshold {
                break;
            }
        }

        Ok(scores)
    }

    /// Select top scoring sentences
    fn select_top_sentences(&self, scores: &Array1<f64>) -> Vec<usize> {
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed_scores
            .iter()
            .take(self.num_sentences)
            .map(|&(idx, _)| idx)
            .collect()
    }

    /// Reconstruct summary maintaining original order
    fn reconstruct_summary(&self, sentences: &[String], indices: &[usize]) -> String {
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();

        sorted_indices
            .iter()
            .map(|&idx| sentences[idx].clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Centroid-based summarization
pub struct CentroidSummarizer {
    /// Number of sentences to extract
    num_sentences: usize,
    /// Topic threshold
    topic_threshold: f64,
    /// Redundancy threshold
    redundancy_threshold: f64,
    /// Sentence tokenizer
    sentence_tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl CentroidSummarizer {
    /// Create a new centroid summarizer
    pub fn new(num_sentences: usize) -> Self {
        Self {
            num_sentences,
            topic_threshold: 0.1,
            redundancy_threshold: 0.95,
            sentence_tokenizer: Box::new(crate::tokenize::SentenceTokenizer::new()),
        }
    }

    /// Summarize text using centroid method
    pub fn summarize(&self, text: &str) -> Result<String> {
        let sentences: Vec<String> = self.sentence_tokenizer.tokenize(text)?;

        if sentences.is_empty() {
            return Ok(String::new());
        }

        if sentences.len() <= self.num_sentences {
            return Ok(text.to_string());
        }

        // Create TF-IDF vectors
        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_ref()).collect();
        let mut vectorizer = TfidfVectorizer::default();
        vectorizer.fit(&sentence_refs)?;
        let vectors = vectorizer.transform_batch(&sentence_refs)?;

        // Calculate centroid
        let centroid = self.calculate_centroid(&vectors);

        // Select sentences closest to centroid
        let selected_indices = self.select_sentences(&vectors, &centroid);

        // Reconstruct summary
        let summary = self.reconstruct_summary(&sentences, &selected_indices);

        Ok(summary)
    }

    /// Calculate document centroid
    fn calculate_centroid(&self, vectors: &Array2<f64>) -> Array1<f64> {
        let _n_docs = vectors.nrows();
        let mut centroid = vectors.mean_axis(ndarray::Axis(0)).unwrap();

        // Apply topic threshold
        centroid.mapv_inplace(|x| if x > self.topic_threshold { x } else { 0.0 });

        centroid
    }

    /// Select sentences based on centroid similarity
    fn select_sentences(&self, vectors: &Array2<f64>, centroid: &Array1<f64>) -> Vec<usize> {
        let mut selected = Vec::new();
        let mut used_sentences = HashSet::new();

        // Calculate similarities to centroid
        let mut similarities: Vec<(usize, f64)> = Vec::new();
        for i in 0..vectors.nrows() {
            let similarity = self.cosine_similarity(vectors.row(i).to_owned(), centroid.clone());
            similarities.push((i, similarity));
        }

        // Sort by similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select sentences avoiding redundancy
        for (idx, _) in similarities {
            if selected.len() >= self.num_sentences {
                break;
            }

            // Check redundancy with already selected sentences
            let mut is_redundant = false;
            for &selected_idx in &selected {
                let sim = self.cosine_similarity(
                    vectors.row(idx).to_owned(),
                    vectors.row(selected_idx).to_owned(),
                );
                if sim > self.redundancy_threshold {
                    is_redundant = true;
                    break;
                }
            }

            if !is_redundant {
                selected.push(idx);
                used_sentences.insert(idx);
            }
        }

        selected
    }

    /// Calculate cosine similarity
    fn cosine_similarity(&self, vec1: Array1<f64>, vec2: Array1<f64>) -> f64 {
        let dot_product = vec1.dot(&vec2);
        let norm1 = vec1.dot(&vec1).sqrt();
        let norm2 = vec2.dot(&vec2).sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Reconstruct summary maintaining original order
    fn reconstruct_summary(&self, sentences: &[String], indices: &[usize]) -> String {
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();

        sorted_indices
            .iter()
            .map(|&idx| sentences[idx].clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Keyword extraction using TF-IDF
pub struct KeywordExtractor {
    /// Number of keywords to extract
    num_keywords: usize,
    /// Minimum document frequency
    #[allow(dead_code)]
    min_df: f64,
    /// Maximum document frequency
    #[allow(dead_code)]
    max_df: f64,
    /// N-gram range
    ngram_range: (usize, usize),
}

impl KeywordExtractor {
    /// Create a new keyword extractor
    pub fn new(num_keywords: usize) -> Self {
        Self {
            num_keywords,
            min_df: 0.01, // Unused but kept for API compatibility
            max_df: 0.95, // Unused but kept for API compatibility
            ngram_range: (1, 3),
        }
    }

    /// Configure n-gram range
    pub fn with_ngram_range(mut self, min_n: usize, max_n: usize) -> Result<Self> {
        if min_n > max_n || min_n == 0 {
            return Err(TextError::InvalidInput("Invalid n-gram range".to_string()));
        }
        self.ngram_range = (min_n, max_n);
        Ok(self)
    }

    /// Extract keywords from text
    pub fn extract_keywords(&self, text: &str) -> Result<Vec<(String, f64)>> {
        // Split into sentences for better TF-IDF
        let sentence_tokenizer = crate::tokenize::SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(text)?;

        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_ref()).collect();

        // Create enhanced TF-IDF vectorizer with n-grams
        // Create vectorizer with ngram range configuration
        let mut vectorizer = crate::enhanced_vectorize::EnhancedTfidfVectorizer::new()
            .set_ngram_range((self.ngram_range.0, self.ngram_range.1))?;

        vectorizer.fit(&sentence_refs)?;
        let tfidf_matrix = vectorizer.transform_batch(&sentence_refs)?;

        // Calculate average TF-IDF scores across documents
        let avg_tfidf = tfidf_matrix.mean_axis(ndarray::Axis(0)).unwrap();

        // Get terms from the tokenizer directly
        let all_words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();

        // Create keyword-score pairs (use top scoring features)
        let mut keyword_scores: Vec<(String, f64)> = avg_tfidf
            .iter()
            .enumerate()
            .take(self.num_keywords * 2) // Get more than needed to filter
            .map(|(i, &score)| {
                let term = if i < all_words.len() {
                    all_words[i].clone()
                } else {
                    format!("term_{}", i)
                };
                (term, score)
            })
            .collect();

        // Sort by score
        keyword_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top keywords
        Ok(keyword_scores.into_iter().take(self.num_keywords).collect())
    }

    /// Extract keywords with position information
    pub fn extract_keywords_with_positions(
        &self,
        text: &str,
    ) -> Result<Vec<(String, f64, Vec<usize>)>> {
        let keywords = self.extract_keywords(text)?;
        let mut results = Vec::new();

        for (keyword, score) in keywords {
            let positions = self.find_keyword_positions(text, &keyword);
            results.push((keyword, score, positions));
        }

        Ok(results)
    }

    /// Find positions of a keyword in text
    fn find_keyword_positions(&self, text: &str, keyword: &str) -> Vec<usize> {
        let mut positions = Vec::new();
        let text_lower = text.to_lowercase();
        let keyword_lower = keyword.to_lowercase();

        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(&keyword_lower) {
            positions.push(start + pos);
            start += pos + keyword.len();
        }

        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_textrank_summarizer() {
        let summarizer = TextRank::new(2);
        let text = "Machine learning is a subset of artificial intelligence. \
                    It enables computers to learn from data. \
                    Deep learning is a subset of machine learning. \
                    Neural networks are used in deep learning. \
                    These technologies are transforming many industries.";

        let summary = summarizer.summarize(text).unwrap();
        assert!(!summary.is_empty());
        assert!(summary.len() < text.len());
    }

    #[test]
    fn test_centroid_summarizer() {
        let summarizer = CentroidSummarizer::new(2);
        let text = "Natural language processing is important. \
                    It helps computers understand human language. \
                    Many applications use NLP technology. \
                    Chatbots and translation are examples. \
                    NLP continues to evolve rapidly.";

        let summary = summarizer.summarize(text).unwrap();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_keyword_extraction() {
        let extractor = KeywordExtractor::new(5);
        let text = "Machine learning algorithms are essential for artificial intelligence. \
                    Deep learning models use neural networks. \
                    These models can process complex data patterns.";

        let keywords = extractor.extract_keywords(text).unwrap();
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);

        // Check that scores are in descending order
        for i in 1..keywords.len() {
            assert!(keywords[i - 1].1 >= keywords[i].1);
        }
    }

    #[test]
    fn test_keyword_positions() {
        let extractor = KeywordExtractor::new(3);
        let text = "Machine learning is great. Machine learning transforms industries.";

        let keywords_with_pos = extractor.extract_keywords_with_positions(text).unwrap();

        // Should find positions for repeated keywords
        for (keyword, _, positions) in keywords_with_pos {
            if keyword.to_lowercase().contains("machine learning") {
                assert!(positions.len() >= 2);
            }
        }
    }

    #[test]
    fn test_empty_text() {
        let textrank = TextRank::new(3);
        let centroid = CentroidSummarizer::new(3);
        let keywords = KeywordExtractor::new(5);

        assert_eq!(textrank.summarize("").unwrap(), "");
        assert_eq!(centroid.summarize("").unwrap(), "");
        assert_eq!(keywords.extract_keywords("").unwrap().len(), 0);
    }

    #[test]
    fn test_short_text() {
        let summarizer = TextRank::new(5);
        let short_text = "This is a short text.";

        let summary = summarizer.summarize(short_text).unwrap();
        assert_eq!(summary, short_text);
    }
}
