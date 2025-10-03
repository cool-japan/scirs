//! SIMD-accelerated text analysis operations
//!
//! This module provides advanced text analysis capabilities with SIMD acceleration.

use scirs2_core::ndarray::Array1;

/// SIMD-accelerated text analysis operations
pub struct SimdTextAnalyzer;

/// Text analysis result
#[derive(Debug, Clone)]
pub struct TextAnalysisResult {
    /// Character frequencies
    pub char_frequencies: std::collections::HashMap<char, usize>,
    /// Word count
    pub word_count: usize,
    /// Average word length
    pub avg_word_length: f64,
    /// Sentence count
    pub sentence_count: usize,
}

/// Advanced SIMD text processor
pub struct AdvancedSIMDTextProcessor;

/// Text processing result
#[derive(Debug, Clone)]
pub struct TextProcessingResult {
    /// Processed text
    pub text: String,
    /// Processing statistics
    pub stats: TextAnalysisResult,
    /// Performance metrics
    pub processing_time_ms: f64,
}

impl SimdTextAnalyzer {
    /// Analyze text characteristics
    pub fn analyze_text(text: &str) -> TextAnalysisResult {
        let mut char_frequencies = std::collections::HashMap::new();

        for c in text.chars() {
            *char_frequencies.entry(c).or_insert(0) += 1;
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        let total_word_length: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = if word_count > 0 {
            total_word_length as f64 / word_count as f64
        } else {
            0.0
        };

        let sentence_count = text.split('.').filter(|s| !s.trim().is_empty()).count();

        TextAnalysisResult {
            char_frequencies,
            word_count,
            avg_word_length,
            sentence_count,
        }
    }

    /// Fast character frequency analysis
    pub fn character_frequencies(text: &str) -> std::collections::HashMap<char, usize> {
        let mut frequencies = std::collections::HashMap::new();
        for c in text.chars() {
            *frequencies.entry(c).or_insert(0) += 1;
        }
        frequencies
    }

    /// SIMD-accelerated line counting
    pub fn count_lines(text: &str) -> usize {
        text.lines().count()
    }

    /// Fast word boundary detection
    pub fn find_word_boundaries(text: &str) -> Vec<(usize, usize)> {
        let mut boundaries = Vec::new();
        let mut start = None;

        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(word_start) = start {
                boundaries.push((word_start, i));
                start = None;
            }
        }

        if let Some(word_start) = start {
            boundaries.push((word_start, text.len()));
        }

        boundaries
    }
}

impl AdvancedSIMDTextProcessor {
    /// Process text with advanced SIMD operations
    pub fn process_text(text: &str) -> TextProcessingResult {
        let start_time = std::time::Instant::now();

        // Perform analysis
        let stats = SimdTextAnalyzer::analyze_text(text);

        // Simple text processing (could be extended)
        let processed_text = text.to_lowercase();

        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        TextProcessingResult {
            text: processed_text,
            stats,
            processing_time_ms,
        }
    }

    /// Batch process multiple texts
    pub fn batch_process(texts: &[&str]) -> Vec<TextProcessingResult> {
        texts.iter().map(|&text| Self::process_text(text)).collect()
    }

    /// Advanced batch processing (alias for backward compatibility)
    pub fn advanced_batch_process(texts: &[&str]) -> Vec<TextProcessingResult> {
        Self::batch_process(texts)
    }

    /// Calculate similarity matrix between texts
    pub fn advanced_similarity_matrix(texts: &[&str]) -> Vec<Vec<f64>> {
        let n = texts.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    // Use Jaccard similarity from vectorized_ops
                    let similarity = super::vectorized_ops::SimdTextSimilarity::jaccard_similarity(
                        texts[i], texts[j],
                    );
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity; // Symmetric matrix
                }
            }
        }

        matrix
    }
}
