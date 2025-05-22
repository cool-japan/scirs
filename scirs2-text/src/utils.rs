//! Utility functions for text processing
//!
//! This module provides utility functions for text processing operations.

use crate::error::{Result, TextError};
use crate::tokenize::Tokenizer;
use scirs2_core::parallel;
use std::collections::HashMap;

/// Count the frequency of tokens in a text
///
/// # Arguments
///
/// * `text` - The text to analyze
/// * `tokenizer` - The tokenizer to use
///
/// # Returns
///
/// * Result containing a HashMap of token frequencies
pub fn count_tokens(text: &str, tokenizer: &dyn Tokenizer) -> Result<HashMap<String, usize>> {
    let tokens = tokenizer.tokenize(text)?;
    let mut counts = HashMap::new();

    for token in tokens {
        *counts.entry(token).or_insert(0) += 1;
    }

    Ok(counts)
}

/// Count the frequency of tokens in a batch of texts
///
/// # Arguments
///
/// * `texts` - The texts to analyze
/// * `tokenizer` - The tokenizer to use
///
/// # Returns
///
/// * Result containing a HashMap of token frequencies
pub fn count_tokens_batch(
    texts: &[&str],
    tokenizer: &dyn Tokenizer,
) -> Result<HashMap<String, usize>> {
    // Process texts sequentially (for thread safety)
    let mut total_counts = HashMap::new();

    for &text in texts {
        let counts = count_tokens(text, tokenizer)?;
        for (token, count) in counts {
            *total_counts.entry(token).or_insert(0) += count;
        }
    }

    Ok(total_counts)
}

/// Count the frequency of tokens in a batch of texts (parallel version)
///
/// # Arguments
///
/// * `texts` - The texts to analyze
/// * `tokenizer` - The tokenizer to use (must be Send + Sync)
///
/// # Returns
///
/// * Result containing a HashMap of token frequencies
pub fn count_tokens_batch_parallel<T>(
    texts: &[&str],
    tokenizer: &T,
) -> Result<HashMap<String, usize>>
where
    T: Tokenizer + Send + Sync,
{
    // Process texts in parallel using scirs2-core::parallel
    // Clone data to avoid lifetime issues
    let texts_owned: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
    let tokenizer_boxed = tokenizer.clone_box();

    let token_counts = parallel::parallel_map(&texts_owned, move |text| {
        count_tokens(text, &*tokenizer_boxed).map_err(|e| {
            // Convert TextError to CoreError
            scirs2_core::CoreError::ComputationError(scirs2_core::error::ErrorContext::new(
                format!("Text processing error: {}", e),
            ))
        })
    })?;

    // Merge all counts
    let mut total_counts = HashMap::new();
    for counts in token_counts {
        for (token, count) in counts {
            *total_counts.entry(token).or_insert(0) += count;
        }
    }

    Ok(total_counts)
}

/// Remove tokens from a text based on a predicate function
///
/// # Arguments
///
/// * `text` - The text to filter
/// * `tokenizer` - The tokenizer to use
/// * `predicate` - Function that returns true for tokens to keep
///
/// # Returns
///
/// * Result containing the filtered text
pub fn filter_tokens<F>(text: &str, tokenizer: &dyn Tokenizer, predicate: F) -> Result<String>
where
    F: Fn(&str) -> bool,
{
    let tokens = tokenizer.tokenize(text)?;
    let filtered_tokens: Vec<String> = tokens
        .iter()
        .filter(|token| predicate(token))
        .cloned()
        .collect();

    Ok(filtered_tokens.join(" "))
}

/// Extract n-grams from a text
///
/// # Arguments
///
/// * `text` - The text to process
/// * `tokenizer` - The tokenizer to use
/// * `n` - The n-gram size
///
/// # Returns
///
/// * Result containing a vector of n-grams
pub fn extract_ngrams(text: &str, tokenizer: &dyn Tokenizer, n: usize) -> Result<Vec<String>> {
    if n == 0 {
        return Err(TextError::InvalidInput(
            "n-gram size must be greater than 0".to_string(),
        ));
    }

    let tokens = tokenizer.tokenize(text)?;

    if tokens.is_empty() || tokens.len() < n {
        return Ok(Vec::new());
    }

    let ngrams: Vec<String> = (0..=(tokens.len() - n))
        .map(|i| tokens[i..(i + n)].to_vec().join(" "))
        .collect();

    Ok(ngrams)
}

/// Extract collocations (frequently co-occurring words) from a text
///
/// # Arguments
///
/// * `text` - The text to process
/// * `tokenizer` - The tokenizer to use
/// * `window_size` - The window size for considering co-occurrence
/// * `min_count` - Minimum count for a collocation to be included
///
/// # Returns
///
/// * Result containing a HashMap of collocations and their frequencies
pub fn extract_collocations(
    text: &str,
    tokenizer: &dyn Tokenizer,
    window_size: usize,
    min_count: usize,
) -> Result<HashMap<(String, String), usize>> {
    let tokens = tokenizer.tokenize(text)?;
    let mut collocations = HashMap::new();

    if tokens.len() < 2 {
        return Ok(collocations);
    }

    // Count co-occurrences within the window
    for i in 0..tokens.len() {
        let end = std::cmp::min(i + window_size + 1, tokens.len());

        for j in (i + 1)..end {
            let pair = (tokens[i].clone(), tokens[j].clone());
            *collocations.entry(pair).or_insert(0) += 1;
        }
    }

    // Filter by minimum count
    collocations.retain(|_, &mut count| count >= min_count);

    Ok(collocations)
}

/// Split text into training and testing sets
///
/// # Arguments
///
/// * `texts` - The texts to split
/// * `test_size` - The proportion of the dataset to use for testing (0.0 to 1.0)
/// * `random_seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * `(Vec<String>, Vec<String>)` - Training and testing sets
pub fn train_test_split(
    texts: &[String],
    test_size: f64,
    random_seed: Option<u64>,
) -> Result<(Vec<String>, Vec<String>)> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    if !(0.0..=1.0).contains(&test_size) {
        return Err(TextError::InvalidInput(
            "test_size must be between 0.0 and 1.0".to_string(),
        ));
    }

    if texts.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Use the random seed if provided
    let mut rng = match random_seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => {
            let mut temp_rng = rand::rng();
            rand::rngs::StdRng::from_rng(&mut temp_rng)
        }
    };

    // Shuffle the texts
    let mut texts_copy = texts.to_vec();
    texts_copy.shuffle(&mut rng);

    // Split into training and testing sets
    let test_count = (texts.len() as f64 * test_size).round() as usize;
    let train_count = texts.len() - test_count;

    let train_texts = texts_copy.iter().take(train_count).cloned().collect();
    let test_texts = texts_copy.iter().skip(train_count).cloned().collect();

    Ok((train_texts, test_texts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WordTokenizer;

    #[test]
    fn test_count_tokens() {
        let tokenizer = WordTokenizer::default();
        let text = "this is a test this is only a test";
        let counts = count_tokens(text, &tokenizer).unwrap();

        assert_eq!(counts.get("this").unwrap(), &2);
        assert_eq!(counts.get("is").unwrap(), &2);
        assert_eq!(counts.get("a").unwrap(), &2);
        assert_eq!(counts.get("test").unwrap(), &2);
        assert_eq!(counts.get("only").unwrap(), &1);
    }

    #[test]
    fn test_filter_tokens() {
        let tokenizer = WordTokenizer::default();
        let text = "this is a test this is only a test";

        // Filter out common words
        let predicate = |token: &str| !["this", "is", "a"].contains(&token);
        let filtered = filter_tokens(text, &tokenizer, predicate).unwrap();

        assert_eq!(filtered, "test only test");
    }

    #[test]
    fn test_extract_ngrams() {
        let tokenizer = WordTokenizer::default();
        let text = "this is a simple test";

        // Extract bigrams
        let bigrams = extract_ngrams(text, &tokenizer, 2).unwrap();
        assert_eq!(bigrams, vec!["this is", "is a", "a simple", "simple test"]);

        // Extract trigrams
        let trigrams = extract_ngrams(text, &tokenizer, 3).unwrap();
        assert_eq!(trigrams, vec!["this is a", "is a simple", "a simple test"]);
    }

    #[test]
    fn test_extract_collocations() {
        let tokenizer = WordTokenizer::default();
        let text = "machine learning is a subset of artificial intelligence that provides systems with the ability to learn";

        let collocations = extract_collocations(text, &tokenizer, 2, 1).unwrap();

        // Check some expected collocations
        assert!(collocations.contains_key(&("machine".to_string(), "learning".to_string())));
        assert!(collocations.contains_key(&("artificial".to_string(), "intelligence".to_string())));
    }

    #[test]
    fn test_train_test_split() {
        let texts = vec![
            "text 1".to_string(),
            "text 2".to_string(),
            "text 3".to_string(),
            "text 4".to_string(),
            "text 5".to_string(),
        ];

        // Split with a fixed seed for reproducibility
        let (train, test) = train_test_split(&texts, 0.4, Some(42)).unwrap();

        assert_eq!(train.len(), 3);
        assert_eq!(test.len(), 2);

        // All texts should be present exactly once in either train or test
        for text in &texts {
            assert_eq!(
                train.iter().filter(|&t| t == text).count()
                    + test.iter().filter(|&t| t == text).count(),
                1
            );
        }
    }
}
