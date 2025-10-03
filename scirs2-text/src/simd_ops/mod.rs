//! SIMD-accelerated string operations for text processing
//!
//! This module provides SIMD-accelerated implementations of common string operations
//! using scirs2-core's SIMD infrastructure.

pub mod basic_ops;
pub mod edit_distance;
pub mod pattern_matching;
pub mod text_analysis;
pub mod vectorized_ops;

// Re-export main components
pub use basic_ops::SimdStringOps;
pub use edit_distance::SimdEditDistance;
pub use pattern_matching::SimdPatternMatcher;
pub use text_analysis::{
    AdvancedSIMDTextProcessor, SimdTextAnalyzer, TextAnalysisResult, TextProcessingResult,
};
pub use vectorized_ops::{
    SimdNgramGenerator, SimdParallelProcessor, SimdTextNormalizer, SimdTextSimilarity,
    VectorizedStringOps,
};

use scirs2_core::ndarray::Array1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_string_ops_count_chars() {
        let text = "hello world hello";
        let count = SimdStringOps::count_chars(text, 'l');
        assert_eq!(count, 5);
    }

    #[test]
    fn test_simd_string_ops_find_whitespace() {
        let text = "hello world test";
        let positions = SimdStringOps::find_whitespace_positions(text);
        assert_eq!(positions, vec![5, 11]);
    }

    #[test]
    fn test_simd_string_ops_to_lowercase() {
        let text = "HELLO World";
        let result = SimdStringOps::to_lowercase_ascii(text);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_simd_string_ops_find_substring() {
        let haystack = "hello world";
        let needle = "world";
        let pos = SimdStringOps::find_substring(haystack, needle);
        assert_eq!(pos, Some(6));
    }

    #[test]
    fn test_simd_hamming_distance() {
        let s1 = "hello";
        let s2 = "hallo";
        let distance = SimdStringOps::hamming_distance(s1, s2);
        assert_eq!(distance, Some(1));
    }

    #[test]
    fn test_simd_edit_distance() {
        let distance = SimdEditDistance::levenshtein("kitten", "sitting");
        assert_eq!(distance, 3);
    }
}
