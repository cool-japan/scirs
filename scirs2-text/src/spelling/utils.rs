//! Utility functions for spelling correction
//!
//! This module provides shared utility functions used across different
//! spelling correction implementations.

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Normalizes a string based on case sensitivity settings
#[inline]
pub fn normalize_string(text: &str, case_sensitive: bool) -> String {
    if !case_sensitive {
        text.to_lowercase()
    } else {
        text.to_string()
    }
}

/// Extract words from text, normalizing and filtering empty words
pub fn extract_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Split text into sentences
pub fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['.', '?', '!'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Check if two words are within the edit distance threshold based on length
#[inline]
pub fn is_within_length_threshold(word1: &str, word2: &str, max_edit_distance: usize) -> bool {
    word1.len() <= word2.len() + max_edit_distance && word1.len() + max_edit_distance >= word2.len()
}

/// Check if a word exists in a dictionary with optional case sensitivity
#[inline]
pub fn dictionary_contains(
    dictionary: &HashMap<String, usize>,
    word: &str,
    case_sensitive: bool,
) -> bool {
    if case_sensitive {
        dictionary.contains_key(word)
    } else {
        let word_lower = word.to_lowercase();
        dictionary
            .keys()
            .any(|dict_word| dict_word.to_lowercase() == word_lower)
    }
}

/// Load data from a file line by line with a custom processor
#[allow(dead_code)]
pub fn load_from_file<P, F, T>(path: P, mut line_processor: F) -> Result<T>
where
    P: AsRef<Path>,
    F: FnMut(&str) -> Result<T>,
    T: Default,
{
    let file =
        File::open(path).map_err(|e| TextError::IoError(format!("Failed to open file: {}", e)))?;

    let reader = BufReader::new(file);
    let mut result = T::default();

    for line in reader.lines() {
        let line =
            line.map_err(|e| TextError::IoError(format!("Failed to read line from file: {}", e)))?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        result = line_processor(&line)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_string() {
        assert_eq!(normalize_string("Hello", false), "hello");
        assert_eq!(normalize_string("Hello", true), "Hello");
    }

    #[test]
    fn test_extract_words() {
        let text = "Hello, world! This is a test.";
        let words = extract_words(text);
        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello, world! This is a test. Another sentence.";
        let sentences = split_sentences(text);
        assert_eq!(
            sentences,
            vec!["Hello, world", "This is a test", "Another sentence"]
        );
    }

    #[test]
    fn test_is_within_length_threshold() {
        assert!(is_within_length_threshold("hello", "hello", 2));
        assert!(is_within_length_threshold("hello", "hell", 2));
        assert!(is_within_length_threshold("hello", "helloo", 2));
        assert!(!is_within_length_threshold("hello", "hi", 2));
        assert!(!is_within_length_threshold("hello", "hello world", 2));
    }

    #[test]
    fn test_dictionary_contains() {
        let mut dictionary = HashMap::new();
        dictionary.insert("Hello".to_string(), 10);
        dictionary.insert("World".to_string(), 20);

        // Case-sensitive checks
        assert!(dictionary_contains(&dictionary, "Hello", true));
        assert!(!dictionary_contains(&dictionary, "hello", true));

        // Case-insensitive checks
        assert!(dictionary_contains(&dictionary, "hello", false));
        assert!(dictionary_contains(&dictionary, "WORLD", false));
        assert!(!dictionary_contains(&dictionary, "test", false));
    }
}
