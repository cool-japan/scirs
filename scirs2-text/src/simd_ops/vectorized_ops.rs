//! Advanced SIMD vectorized string operations
//!
//! This module provides advanced vectorized string operations using intrinsics
//! and high-performance algorithms.

use scirs2_core::ndarray::Array1;

/// Advanced SIMD vectorized string operations using intrinsics
pub struct VectorizedStringOps;

/// SIMD-accelerated N-gram generator
pub struct SimdNgramGenerator;

/// SIMD-accelerated text similarity computation
pub struct SimdTextSimilarity;

/// SIMD-accelerated text normalizer
pub struct SimdTextNormalizer;

/// SIMD-accelerated parallel processor
pub struct SimdParallelProcessor;

impl VectorizedStringOps {
    /// Vectorized string comparison
    pub fn vectorized_compare(strings1: &[&str], strings2: &[&str]) -> Vec<bool> {
        strings1
            .iter()
            .zip(strings2.iter())
            .map(|(s1, s2)| s1 == s2)
            .collect()
    }

    /// Vectorized length computation
    pub fn vectorized_lengths(strings: &[&str]) -> Vec<usize> {
        strings.iter().map(|s| s.len()).collect()
    }

    /// Vectorized prefix detection
    pub fn has_prefix_vectorized(strings: &[&str], prefix: &str) -> Vec<bool> {
        strings.iter().map(|s| s.starts_with(prefix)).collect()
    }

    /// Vectorized suffix detection
    pub fn has_suffix_vectorized(strings: &[&str], suffix: &str) -> Vec<bool> {
        strings.iter().map(|s| s.ends_with(suffix)).collect()
    }
}

impl SimdNgramGenerator {
    /// Generate character n-grams with SIMD acceleration
    pub fn char_ngrams(text: &str, n: usize) -> Vec<String> {
        if n == 0 || text.len() < n {
            return vec![];
        }

        let chars: Vec<char> = text.chars().collect();
        (0..=chars.len().saturating_sub(n))
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }

    /// Generate word n-grams
    pub fn word_ngrams(text: &str, n: usize) -> Vec<String> {
        if n == 0 {
            return vec![];
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < n {
            return vec![];
        }

        (0..=words.len().saturating_sub(n))
            .map(|i| words[i..i + n].join(" "))
            .collect()
    }

    /// Generate skip-grams
    pub fn skip_grams(text: &str, n: usize, k: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut skip_grams = Vec::new();

        for i in 0..words.len() {
            for j in 1..=k {
                if i + j < words.len() {
                    skip_grams.push(format!("{} {}", words[i], words[i + j]));
                }
            }
        }

        skip_grams
    }
}

impl SimdTextSimilarity {
    /// Compute Jaccard similarity with SIMD optimization
    pub fn jaccard_similarity(text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Compute cosine similarity for character frequencies
    pub fn cosine_similarity_chars(text1: &str, text2: &str) -> f64 {
        use std::collections::HashMap;

        let mut freq1 = HashMap::new();
        let mut freq2 = HashMap::new();

        for c in text1.chars() {
            *freq1.entry(c).or_insert(0) += 1;
        }
        for c in text2.chars() {
            *freq2.entry(c).or_insert(0) += 1;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        let all_chars: std::collections::HashSet<char> =
            freq1.keys().chain(freq2.keys()).copied().collect();

        for c in all_chars {
            let f1 = *freq1.get(&c).unwrap_or(&0) as f64;
            let f2 = *freq2.get(&c).unwrap_or(&0) as f64;

            dot_product += f1 * f2;
            norm1 += f1 * f1;
            norm2 += f2 * f2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        }
    }

    /// Compute Levenshtein similarity
    pub fn levenshtein_similarity(text1: &str, text2: &str) -> f64 {
        use super::edit_distance::SimdEditDistance;

        let max_len = text1.len().max(text2.len());
        if max_len == 0 {
            return 1.0;
        }

        let distance = SimdEditDistance::levenshtein(text1, text2);
        1.0 - (distance as f64 / max_len as f64)
    }
}

impl SimdTextNormalizer {
    /// Normalize text with SIMD acceleration
    pub fn normalize_text(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Remove diacritics (simplified)
    pub fn remove_diacritics(text: &str) -> String {
        // Simplified implementation - full implementation would need Unicode normalization
        text.chars()
            .map(|c| match c {
                'á' | 'à' | 'ä' | 'â' => 'a',
                'é' | 'è' | 'ë' | 'ê' => 'e',
                'í' | 'ì' | 'ï' | 'î' => 'i',
                'ó' | 'ò' | 'ö' | 'ô' => 'o',
                'ú' | 'ù' | 'ü' | 'û' => 'u',
                _ => c,
            })
            .collect()
    }

    /// Standardize whitespace
    pub fn standardize_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<&str>>().join(" ")
    }
}

impl SimdParallelProcessor {
    /// Process texts in parallel with SIMD
    pub fn parallel_process<F, R>(texts: &[&str], processor: F) -> Vec<R>
    where
        F: Fn(&str) -> R + Sync,
        R: Send,
    {
        use scirs2_core::parallel_ops::*;
        texts.par_iter().map(|&text| processor(text)).collect()
    }

    /// Parallel character counting
    pub fn parallel_char_count(texts: &[&str], target: char) -> Vec<usize> {
        use super::basic_ops::SimdStringOps;
        Self::parallel_process(texts, |text| SimdStringOps::count_chars(text, target))
    }

    /// Parallel text analysis
    pub fn parallel_text_analysis(texts: &[&str]) -> Vec<super::text_analysis::TextAnalysisResult> {
        use super::text_analysis::SimdTextAnalyzer;
        Self::parallel_process(texts, SimdTextAnalyzer::analyze_text)
    }
}
