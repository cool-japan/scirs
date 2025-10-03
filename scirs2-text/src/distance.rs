//! Text distance and similarity measures
//!
//! This module provides functions for computing distances and similarities
//! between texts or sequences.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::{HashMap, HashSet};

/// Compute the Levenshtein edit distance between two strings
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string into another.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// * The Levenshtein distance between the two strings
#[allow(dead_code)]
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    // Use SIMD-accelerated version when available
    #[cfg(feature = "simd")]
    {
        crate::simd_ops::SimdEditDistance::levenshtein(s1, s2)
    }

    #[cfg(not(feature = "simd"))]
    {
        if s1.is_empty() {
            return s2.chars().count();
        }
        if s2.is_empty() {
            return s1.chars().count();
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let m = s1_chars.len();
        let n = s2_chars.len();

        // Create distance matrix
        let mut matrix = vec![vec![0; n + 1]; m + 1];

        // Initialize first row and column
        for (i, row) in matrix.iter_mut().enumerate().take(m + 1) {
            row[0] = i;
        }
        if let Some(first_row) = matrix.first_mut() {
            for (j, cell) in first_row.iter_mut().enumerate().take(n + 1) {
                *cell = j;
            }
        }

        // Fill the matrix
        for i in 1..=m {
            for j in 1..=n {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1, // deletion
                        matrix[i][j - 1] + 1, // insertion
                    ),
                    matrix[i - 1][j - 1] + cost, // substitution
                );
            }
        }

        matrix[m][n]
    }
}

/// Compute the normalized Levenshtein distance between two strings
///
/// The normalized distance is the Levenshtein distance divided by
/// the maximum length of the two strings, resulting in a value in [0, 1].
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// * The normalized Levenshtein distance between the two strings
#[allow(dead_code)]
pub fn normalized_levenshtein_distance(s1: &str, s2: &str) -> f64 {
    let distance = levenshtein_distance(s1, s2) as f64;
    let max_length = std::cmp::max(s1.chars().count(), s2.chars().count()) as f64;

    if max_length == 0.0 {
        return 0.0;
    }

    distance / max_length
}

/// Compute the Jaccard similarity between two strings
///
/// The Jaccard similarity is the size of the intersection divided by
/// the size of the union of the two sets of tokens.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
/// * `tokenizer` - Optional tokenizer to use (defaults to word tokenizer)
///
/// # Returns
///
/// * Result containing the Jaccard similarity between the two strings
#[allow(dead_code)]
pub fn jaccard_similarity(s1: &str, s2: &str, tokenizer: Option<&dyn Tokenizer>) -> Result<f64> {
    // Use the provided tokenizer or default to word tokenizer
    let tokenizer = match tokenizer {
        Some(t) => t,
        None => &WordTokenizer::default(),
    };

    // Tokenize both strings
    let tokens1 = tokenizer.tokenize(s1)?;
    let tokens2 = tokenizer.tokenize(s2)?;

    // Convert to sets
    let set1: HashSet<&String> = tokens1.iter().collect();
    let set2: HashSet<&String> = tokens2.iter().collect();

    // Calculate intersection and union
    let intersection_size = set1.intersection(&set2).count();
    let union_size = set1.union(&set2).count();

    if union_size == 0 {
        return Ok(1.0); // Both strings are empty or have no common tokens
    }

    Ok(intersection_size as f64 / union_size as f64)
}

/// Compute the cosine similarity between two vectors
///
/// The cosine similarity is the cosine of the angle between the two vectors.
///
/// # Arguments
///
/// * `v1` - First vector
/// * `v2` - Second vector
///
/// # Returns
///
/// * Result containing the cosine similarity between the two vectors
#[allow(dead_code)]
pub fn cosine_similarity(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> Result<f64> {
    if v1.len() != v2.len() {
        return Err(TextError::DistanceError(format!(
            "Vectors must have the same dimension, got {} and {}",
            v1.len(),
            v2.len()
        )));
    }

    #[cfg(feature = "simd")]
    {
        use scirs2_core::simd_ops::SimdUnifiedOps;

        // Use SIMD operations for dot product and norms
        let dot_product = f64::simd_dot(&v1, &v2);
        let norm1 = f64::simd_norm(&v1);
        let norm2 = f64::simd_norm(&v2);

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(if norm1 == norm2 { 1.0 } else { 0.0 });
        }

        Ok(dot_product / (norm1 * norm2))
    }

    #[cfg(not(feature = "simd"))]
    {
        // Calculate dot product manually since direct multiplication isn't implemented for ArrayView1
        let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();

        // Calculate norms manually
        let norm1 = v1.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(if norm1 == norm2 { 1.0 } else { 0.0 });
        }

        Ok(dot_product / (norm1 * norm2))
    }
}

/// Compute the cosine similarity between two texts based on bag-of-words vectors
///
/// # Arguments
///
/// * `s1` - First text
/// * `s2` - Second text
/// * `tokenizer` - Optional tokenizer to use (defaults to word tokenizer)
///
/// # Returns
///
/// * Result containing the cosine similarity between the two texts
#[allow(dead_code)]
pub fn text_cosine_similarity(
    s1: &str,
    s2: &str,
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<f64> {
    // Use the provided tokenizer or default to word tokenizer
    let tokenizer = match tokenizer {
        Some(t) => t,
        None => &WordTokenizer::default(),
    };

    // Tokenize both strings
    let tokens1 = tokenizer.tokenize(s1)?;
    let tokens2 = tokenizer.tokenize(s2)?;

    // Get unique tokens from both texts to create a vocabulary
    let mut all_tokens = HashSet::new();
    for token in tokens1.iter().chain(tokens2.iter()) {
        all_tokens.insert(token.clone());
    }

    // Sort tokens to ensure consistent order
    let mut sorted_tokens: Vec<String> = all_tokens.into_iter().collect();
    sorted_tokens.sort();

    // Create a mapping from token to index
    let token_to_idx: HashMap<String, usize> = sorted_tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), i))
        .collect();

    // Create vectors for each text
    let dimension = token_to_idx.len();
    let mut v1 = Array1::zeros(dimension);
    let mut v2 = Array1::zeros(dimension);

    // Fill vectors with token counts
    for token in &tokens1 {
        if let Some(&idx) = token_to_idx.get(token) {
            v1[idx] += 1.0;
        }
    }

    for token in &tokens2 {
        if let Some(&idx) = token_to_idx.get(token) {
            v2[idx] += 1.0;
        }
    }

    // Compute cosine similarity
    cosine_similarity(v1.view(), v2.view())
}

/// Compute the Jaro-Winkler similarity between two strings
///
/// The Jaro-Winkler similarity is a variant of the Jaro distance metric,
/// which gives higher scores to strings that match from the beginning.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// * The Jaro-Winkler similarity between the two strings (0.0 to 1.0)
#[allow(dead_code)]
pub fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    // Compute Jaro similarity first
    let jaro_sim = jaro_similarity(s1, s2);

    // Calculate the common prefix length (up to 4 characters)
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let mut prefix_len = 0;
    let max_prefix = std::cmp::min(4, std::cmp::min(s1_chars.len(), s2_chars.len()));

    for i in 0..max_prefix {
        if s1_chars[i] == s2_chars[i] {
            prefix_len += 1;
        } else {
            break;
        }
    }

    // Apply the Winkler adjustment (p = 0.1 is standard)
    let p = 0.1;
    jaro_sim + (prefix_len as f64 * p * (1.0 - jaro_sim))
}

/// Compute the Jaro similarity between two strings
///
/// The Jaro distance is the weighted sum of percentage of matched characters
/// from each string and transposed characters.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// * The Jaro similarity between the two strings (0.0 to 1.0)
#[allow(dead_code)]
fn jaro_similarity(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }

    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let match_distance = std::cmp::max(s1_chars.len(), s2_chars.len()) / 2 - 1;
    let match_distance = std::cmp::max(0, match_distance);

    // Flags for matched characters
    let mut s1_matches = vec![false; s1_chars.len()];
    let mut s2_matches = vec![false; s2_chars.len()];

    // Count matching characters within match_distance
    let mut matching_chars = 0;
    for (i, &c1) in s1_chars.iter().enumerate() {
        let start = i.saturating_sub(match_distance);
        let end = std::cmp::min(i + match_distance + 1, s2_chars.len());

        for j in start..end {
            if !s2_matches[j] && s2_chars[j] == c1 {
                s1_matches[i] = true;
                s2_matches[j] = true;
                matching_chars += 1;
                break;
            }
        }
    }

    if matching_chars == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut transpositions = 0;
    let mut j = 0;

    for (i, &matched) in s1_matches.iter().enumerate() {
        if matched {
            while !s2_matches[j] {
                j += 1;
            }

            if s1_chars[i] != s2_chars[j] {
                transpositions += 1;
            }

            j += 1;
        }
    }

    // Half the number of character transpositions
    let transpositions = transpositions as f64 / 2.0;

    // Calculate Jaro similarity
    let m = matching_chars as f64;

    1.0 / 3.0 * (m / s1_chars.len() as f64 + m / s2_chars.len() as f64 + (m - transpositions) / m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn test_normalized_levenshtein_distance() {
        assert_eq!(
            normalized_levenshtein_distance("kitten", "sitting"),
            3.0 / 7.0
        );
        assert_eq!(normalized_levenshtein_distance("hello", "hello"), 0.0);
        assert_eq!(normalized_levenshtein_distance("", "hello"), 1.0);
        assert_eq!(normalized_levenshtein_distance("hello", ""), 1.0);
        assert_eq!(normalized_levenshtein_distance("", ""), 0.0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let result = jaccard_similarity("this is a test", "this is another test", None).unwrap();
        assert!(result > 0.5 && result < 1.0);

        let result = jaccard_similarity("identical", "identical", None).unwrap();
        assert_eq!(result, 1.0);

        let result = jaccard_similarity("completely different", "not the same", None).unwrap();
        assert_eq!(result, 0.0);

        let result = jaccard_similarity("", "", None).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = array![1.0, 0.0, 0.0];
        let v2 = array![0.0, 1.0, 0.0];
        let result = cosine_similarity(v1.view(), v2.view()).unwrap();
        assert_eq!(result, 0.0);

        let v1 = array![1.0, 1.0, 1.0];
        let v2 = array![1.0, 1.0, 1.0];
        let result = cosine_similarity(v1.view(), v2.view()).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        let v1 = array![1.0, 2.0, 3.0];
        let v2 = array![4.0, 5.0, 6.0];
        let result = cosine_similarity(v1.view(), v2.view()).unwrap();
        assert!(result > 0.9 && result < 1.0);
    }

    #[test]
    fn testtext_cosine_similarity() {
        let result =
            text_cosine_similarity("this is a test", "this is another test", None).unwrap();
        assert!(result > 0.5 && result < 1.0);

        let result = text_cosine_similarity("identical", "identical", None).unwrap();
        assert_eq!(result, 1.0);

        let result = text_cosine_similarity("completely different", "not the same", None).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_jaro_winkler_similarity() {
        assert!(jaro_winkler_similarity("DWAYNE", "DUANE") > 0.8);
        assert!(jaro_winkler_similarity("MARTHA", "MARHTA") > 0.9);

        // Prefix should boost the score
        let jaro = jaro_similarity("PREFIX", "PREFIXX");
        let jaro_winkler = jaro_winkler_similarity("PREFIX", "PREFIXX");
        assert!(jaro_winkler > jaro);

        // Same strings should have score 1.0
        assert_eq!(jaro_winkler_similarity("SAME", "SAME"), 1.0);
    }
}
