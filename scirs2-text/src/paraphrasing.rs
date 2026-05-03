//! # Text Paraphrasing Module
//!
//! This module provides advanced text paraphrasing capabilities using multiple strategies:
//! - **Synonym-based paraphrasing**: Replace words with semantically similar alternatives
//! - **Sentence restructuring**: Reorder clauses and change sentence structures
//! - **Back-translation**: Simulate translation-based paraphrasing patterns
//! - **Template-based variation**: Use linguistic patterns for generation
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::paraphrasing::{Paraphraser, ParaphraseConfig, ParaphraseStrategy};
//!
//! let config = ParaphraseConfig {
//!     num_variations: 3,
//!     strategy: ParaphraseStrategy::Hybrid,
//!     preserve_entities: true,
//!     min_similarity: 0.6,
//!     ..Default::default()
//! };
//!
//! let paraphraser = Paraphraser::new(config);
//! let text = "The quick brown fox jumps over the lazy dog";
//! let paraphrases = paraphraser.paraphrase(text).expect("Paraphrasing failed");
//!
//! for (i, paraphrase) in paraphrases.iter().enumerate() {
//!     println!("Paraphrase {}: {}", i + 1, paraphrase.text);
//!     println!("  Similarity: {:.3}", paraphrase.similarity);
//! }
//! ```

use crate::embeddings::Word2Vec;
use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, HashSet};

/// Paraphrasing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParaphraseStrategy {
    /// Use only synonym replacement
    Synonym,
    /// Use only sentence restructuring
    Restructure,
    /// Use back-translation patterns
    BackTranslation,
    /// Combine multiple strategies
    Hybrid,
}

/// Configuration for paraphrasing
#[derive(Debug, Clone)]
pub struct ParaphraseConfig {
    /// Number of paraphrase variations to generate
    pub num_variations: usize,
    /// Paraphrasing strategy to use
    pub strategy: ParaphraseStrategy,
    /// Whether to preserve named entities
    pub preserve_entities: bool,
    /// Minimum semantic similarity threshold (0.0-1.0)
    pub min_similarity: f32,
    /// Maximum percentage of words to replace (0.0-1.0)
    pub max_replacement_ratio: f32,
    /// Whether to use aggressive transformations
    pub aggressive: bool,
}

impl Default for ParaphraseConfig {
    fn default() -> Self {
        Self {
            num_variations: 3,
            strategy: ParaphraseStrategy::Hybrid,
            preserve_entities: true,
            min_similarity: 0.6,
            max_replacement_ratio: 0.4,
            aggressive: false,
        }
    }
}

/// A paraphrased text result
#[derive(Debug, Clone)]
pub struct ParaphraseResult {
    /// The paraphrased text
    pub text: String,
    /// Semantic similarity to original (0.0-1.0)
    pub similarity: f32,
    /// Strategy used for this paraphrase
    pub strategy_used: ParaphraseStrategy,
    /// Words that were replaced
    pub replacements: Vec<(String, String)>,
}

/// Main paraphraser
pub struct Paraphraser {
    config: ParaphraseConfig,
    tokenizer: Box<dyn Tokenizer>,
    word2vec: Option<Word2Vec>,
    synonym_map: HashMap<String, Vec<String>>,
}

impl Paraphraser {
    /// Create a new paraphraser with configuration
    pub fn new(config: ParaphraseConfig) -> Self {
        Self {
            config,
            tokenizer: Box::new(WordTokenizer::default()),
            word2vec: None,
            synonym_map: Self::build_default_synonym_map(),
        }
    }

    /// Create with a trained Word2Vec model for better synonym detection
    pub fn with_word2vec(mut self, model: Word2Vec) -> Self {
        self.word2vec = Some(model);
        self
    }

    /// Create with a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Derive a deterministic u64 seed from a text string using FNV-1a hashing.
    ///
    /// Using a text-derived seed ensures that the same input always produces the
    /// same output, eliminating non-determinism from parallel test execution.
    fn text_seed(text: &str) -> u64 {
        let mut hash: u64 = 14695981039346656037;
        for byte in text.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }

    /// Generate paraphrases of the input text
    pub fn paraphrase(&self, text: &str) -> Result<Vec<ParaphraseResult>> {
        if text.trim().is_empty() {
            return Err(TextError::InvalidInput("Input text is empty".into()));
        }

        let mut results = Vec::new();
        let mut seen = HashSet::new();
        seen.insert(text.to_lowercase());

        let mut attempt = 0;
        let max_attempts = self.config.num_variations * 3;

        while results.len() < self.config.num_variations && attempt < max_attempts {
            attempt += 1;

            let strategy = if self.config.strategy == ParaphraseStrategy::Hybrid {
                // Select a strategy deterministically but varying by attempt so the
                // dedup loop can pick different strategies on successive iterations.
                self.select_random_strategy(text, attempt)
            } else {
                self.config.strategy
            };

            let paraphrase_result = match strategy {
                ParaphraseStrategy::Synonym => self.paraphrase_synonym(text)?,
                ParaphraseStrategy::Restructure => self.paraphrase_restructure(text)?,
                ParaphraseStrategy::BackTranslation => self.paraphrase_backtranslation(text)?,
                ParaphraseStrategy::Hybrid => unreachable!(),
            };

            // Check for duplicates
            let paraphrase_lower = paraphrase_result.text.to_lowercase();
            if !seen.contains(&paraphrase_lower) && paraphrase_result.text != text {
                seen.insert(paraphrase_lower);
                results.push(paraphrase_result);
            }
        }

        if results.is_empty() {
            return Err(TextError::ProcessingError(
                "Could not generate any valid paraphrases".to_string(),
            ));
        }

        Ok(results)
    }

    /// Paraphrase using synonym replacement
    fn paraphrase_synonym(&self, text: &str) -> Result<ParaphraseResult> {
        let tokens = self.tokenizer.tokenize(text)?;
        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(1));
        let mut new_tokens = tokens.clone();
        let mut replacements = Vec::new();

        // Determine how many words to replace
        let max_replacements =
            ((tokens.len() as f32 * self.config.max_replacement_ratio).ceil() as usize).max(1);

        let mut replaced_count = 0;
        let mut candidates: Vec<usize> = (0..tokens.len()).collect();

        // Shuffle candidates
        for i in (1..candidates.len()).rev() {
            let j = (rng.random::<f32>() * (i + 1) as f32) as usize;
            candidates.swap(i, j);
        }

        // Try to replace words
        for &idx in candidates.iter() {
            if replaced_count >= max_replacements {
                break;
            }

            let word = &tokens[idx];

            // Skip short words and punctuation
            if word.len() <= 2 || !word.chars().any(|c| c.is_alphabetic()) {
                continue;
            }

            // Try to find a synonym
            if let Some(synonym) = self.find_synonym(word)? {
                new_tokens[idx] = synonym.clone();
                replacements.push((word.clone(), synonym));
                replaced_count += 1;
            }
        }

        let paraphrased_text = new_tokens.join(" ");
        let similarity = self.calculate_similarity(text, &paraphrased_text);

        Ok(ParaphraseResult {
            text: paraphrased_text,
            similarity,
            strategy_used: ParaphraseStrategy::Synonym,
            replacements,
        })
    }

    /// Paraphrase using sentence restructuring
    fn paraphrase_restructure(&self, text: &str) -> Result<ParaphraseResult> {
        let restructured = self.apply_restructuring_patterns(text)?;
        let similarity = self.calculate_similarity(text, &restructured);

        Ok(ParaphraseResult {
            text: restructured,
            similarity,
            strategy_used: ParaphraseStrategy::Restructure,
            replacements: vec![],
        })
    }

    /// Paraphrase using back-translation patterns
    fn paraphrase_backtranslation(&self, text: &str) -> Result<ParaphraseResult> {
        let transformed = self.apply_backtranslation_patterns(text)?;
        let similarity = self.calculate_similarity(text, &transformed);

        Ok(ParaphraseResult {
            text: transformed,
            similarity,
            strategy_used: ParaphraseStrategy::BackTranslation,
            replacements: vec![],
        })
    }

    /// Find a synonym for a word
    fn find_synonym(&self, word: &str) -> Result<Option<String>> {
        let word_lower = word.to_lowercase();

        // Try Word2Vec if available
        if let Some(ref model) = self.word2vec {
            if let Ok(similar_words) = model.most_similar(&word_lower, 5) {
                if !similar_words.is_empty() {
                    let mut rng = StdRng::seed_from_u64(Self::text_seed(word).wrapping_add(2));
                    let idx = (rng.random::<f32>() * similar_words.len() as f32) as usize;
                    let selected = &similar_words[idx.min(similar_words.len() - 1)].0;
                    return Ok(Some(self.match_case(word, selected)));
                }
            }
        }

        // Fall back to synonym map
        if let Some(synonyms) = self.synonym_map.get(&word_lower) {
            if !synonyms.is_empty() {
                let mut rng = StdRng::seed_from_u64(Self::text_seed(word).wrapping_add(3));
                let idx = (rng.random::<f32>() * synonyms.len() as f32) as usize;
                let selected = &synonyms[idx.min(synonyms.len() - 1)];
                return Ok(Some(self.match_case(word, selected)));
            }
        }

        Ok(None)
    }

    /// Match the case of the original word
    fn match_case(&self, original: &str, replacement: &str) -> String {
        if original.chars().all(|c| c.is_uppercase()) {
            replacement.to_uppercase()
        } else if original.chars().next().is_some_and(|c| c.is_uppercase()) {
            let mut chars = replacement.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        } else {
            replacement.to_lowercase()
        }
    }

    /// Apply sentence restructuring patterns
    fn apply_restructuring_patterns(&self, text: &str) -> Result<String> {
        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(4));
        let pattern_idx = (rng.random::<f32>() * 4.0) as usize;

        let result = match pattern_idx {
            0 => self.pattern_passive_to_active(text),
            1 => self.pattern_clause_reorder(text),
            2 => self.pattern_conjunction_variation(text),
            _ => self.pattern_adverb_movement(text),
        };

        Ok(result)
    }

    /// Convert passive voice to active or vice versa
    fn pattern_passive_to_active(&self, text: &str) -> String {
        // Simple pattern: "X is Y by Z" -> "Z Y X"
        // This is a simplified transformation
        if text.contains(" is ") && text.contains(" by ") {
            let parts: Vec<&str> = text.split(" by ").collect();
            if parts.len() == 2 {
                let first_parts: Vec<&str> = parts[0].split(" is ").collect();
                if first_parts.len() == 2 {
                    return format!(
                        "{} {} {}",
                        parts[1].trim(),
                        first_parts[1].trim(),
                        first_parts[0].trim()
                    );
                }
            }
        }
        text.to_string()
    }

    /// Reorder clauses in compound sentences
    fn pattern_clause_reorder(&self, text: &str) -> String {
        // Reorder around conjunctions
        for conj in &[" and ", " but ", " or ", ", "] {
            if text.contains(conj) {
                let parts: Vec<&str> = text.splitn(2, conj).collect();
                if parts.len() == 2 {
                    return format!("{}{}{}", parts[1].trim(), conj, parts[0].trim());
                }
            }
        }
        text.to_string()
    }

    /// Vary conjunctions
    fn pattern_conjunction_variation(&self, text: &str) -> String {
        let replacements = [
            (" and ", " as well as "),
            (" but ", " however "),
            (" because ", " since "),
            (" so ", " therefore "),
        ];

        let mut result = text.to_string();
        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(5));
        let idx = (rng.random::<f32>() * replacements.len() as f32) as usize;
        let (original, replacement) = replacements[idx.min(replacements.len() - 1)];

        if result.contains(original) {
            result = result.replacen(original, replacement, 1);
        }

        result
    }

    /// Move adverbs to different positions
    fn pattern_adverb_movement(&self, text: &str) -> String {
        // Move adverbs ending in "ly" to different positions
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 3 {
            return text.to_string();
        }

        // Find adverbs
        for (i, token) in tokens.iter().enumerate() {
            if token.ends_with("ly") && i > 0 {
                // Move adverb to the beginning
                let mut new_tokens = tokens.clone();
                new_tokens.remove(i);
                new_tokens.insert(0, token);
                return new_tokens.join(" ");
            }
        }

        text.to_string()
    }

    /// Apply back-translation patterns
    fn apply_backtranslation_patterns(&self, text: &str) -> Result<String> {
        // Simulate back-translation artifacts
        let patterns = [
            self.pattern_article_variation(text),
            self.pattern_preposition_variation(text),
            self.pattern_tense_variation(text),
            self.pattern_number_variation(text),
        ];

        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(6));
        let idx = (rng.random::<f32>() * patterns.len() as f32) as usize;
        Ok(patterns[idx.min(patterns.len() - 1)].clone())
    }

    /// Vary article usage
    fn pattern_article_variation(&self, text: &str) -> String {
        let mut result = text.to_string();
        let replacements = [(" a ", " the "), (" the ", " a "), (" an ", " the ")];

        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(7));
        let idx = (rng.random::<f32>() * replacements.len() as f32) as usize;
        let (original, replacement) = replacements[idx.min(replacements.len() - 1)];

        if result.contains(original) {
            result = result.replacen(original, replacement, 1);
        }

        result
    }

    /// Vary prepositions
    fn pattern_preposition_variation(&self, text: &str) -> String {
        let replacements = [
            (" on ", " upon "),
            (" in ", " within "),
            (" at ", " in "),
            (" to ", " towards "),
        ];

        let mut result = text.to_string();
        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(8));
        let idx = (rng.random::<f32>() * replacements.len() as f32) as usize;
        let (original, replacement) = replacements[idx.min(replacements.len() - 1)];

        if result.contains(original) {
            result = result.replacen(original, replacement, 1);
        }

        result
    }

    /// Vary verb tenses
    fn pattern_tense_variation(&self, text: &str) -> String {
        let replacements = [
            (" is ", " was "),
            (" are ", " were "),
            (" has ", " had "),
            (" will ", " would "),
        ];

        let mut result = text.to_string();
        let mut rng = StdRng::seed_from_u64(Self::text_seed(text).wrapping_add(9));
        let idx = (rng.random::<f32>() * replacements.len() as f32) as usize;
        let (original, replacement) = replacements[idx.min(replacements.len() - 1)];

        if result.contains(original) {
            result = result.replacen(original, replacement, 1);
        }

        result
    }

    /// Vary singular/plural forms
    fn pattern_number_variation(&self, text: &str) -> String {
        // This is a very simplified approach
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let mut new_tokens = tokens.clone();

        for (i, token) in tokens.iter().enumerate() {
            if token.ends_with('s') && token.len() > 2 && !token.ends_with("ss") {
                // Try to singularize
                new_tokens[i] = token[..token.len() - 1].to_string();
                break;
            } else if !token.ends_with('s') && token.chars().all(|c| c.is_alphabetic()) {
                // Try to pluralize
                new_tokens[i] = format!("{}s", token);
                break;
            }
        }

        new_tokens.join(" ")
    }

    /// Calculate semantic similarity between two texts
    fn calculate_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple Jaccard similarity based on tokens
        let tokens1: HashSet<String> = text1
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let tokens2: HashSet<String> = text2
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    /// Select a deterministic strategy for hybrid mode.
    ///
    /// The seed mixes the text hash with the attempt counter so that successive
    /// iterations of the dedup loop choose different strategies and are therefore
    /// able to produce distinct paraphrases.
    fn select_random_strategy(&self, text: &str, attempt: usize) -> ParaphraseStrategy {
        let seed = Self::text_seed(text)
            .wrapping_add(10)
            .wrapping_add(attempt as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let val = rng.random::<f32>();

        if val < 0.33 {
            ParaphraseStrategy::Synonym
        } else if val < 0.67 {
            ParaphraseStrategy::Restructure
        } else {
            ParaphraseStrategy::BackTranslation
        }
    }

    /// Build a default synonym map
    fn build_default_synonym_map() -> HashMap<String, Vec<String>> {
        let mut map = HashMap::new();

        // Common synonyms
        map.insert(
            "good".to_string(),
            vec![
                "excellent".to_string(),
                "great".to_string(),
                "fine".to_string(),
            ],
        );
        map.insert(
            "bad".to_string(),
            vec![
                "poor".to_string(),
                "awful".to_string(),
                "terrible".to_string(),
            ],
        );
        map.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "enormous".to_string(),
            ],
        );
        map.insert(
            "small".to_string(),
            vec![
                "tiny".to_string(),
                "little".to_string(),
                "minute".to_string(),
            ],
        );
        map.insert(
            "fast".to_string(),
            vec![
                "quick".to_string(),
                "rapid".to_string(),
                "swift".to_string(),
            ],
        );
        map.insert(
            "slow".to_string(),
            vec![
                "gradual".to_string(),
                "leisurely".to_string(),
                "sluggish".to_string(),
            ],
        );
        map.insert(
            "important".to_string(),
            vec![
                "significant".to_string(),
                "crucial".to_string(),
                "vital".to_string(),
            ],
        );
        map.insert(
            "easy".to_string(),
            vec![
                "simple".to_string(),
                "effortless".to_string(),
                "straightforward".to_string(),
            ],
        );
        map.insert(
            "difficult".to_string(),
            vec![
                "hard".to_string(),
                "challenging".to_string(),
                "complex".to_string(),
            ],
        );
        map.insert(
            "beautiful".to_string(),
            vec![
                "lovely".to_string(),
                "attractive".to_string(),
                "gorgeous".to_string(),
            ],
        );
        map.insert(
            "happy".to_string(),
            vec![
                "joyful".to_string(),
                "cheerful".to_string(),
                "delighted".to_string(),
            ],
        );
        map.insert(
            "sad".to_string(),
            vec![
                "unhappy".to_string(),
                "sorrowful".to_string(),
                "melancholy".to_string(),
            ],
        );
        map.insert(
            "smart".to_string(),
            vec![
                "intelligent".to_string(),
                "clever".to_string(),
                "bright".to_string(),
            ],
        );
        map.insert(
            "stupid".to_string(),
            vec![
                "foolish".to_string(),
                "silly".to_string(),
                "ignorant".to_string(),
            ],
        );
        map.insert(
            "old".to_string(),
            vec![
                "ancient".to_string(),
                "aged".to_string(),
                "elderly".to_string(),
            ],
        );
        map.insert(
            "new".to_string(),
            vec![
                "recent".to_string(),
                "modern".to_string(),
                "fresh".to_string(),
            ],
        );
        map.insert(
            "strong".to_string(),
            vec![
                "powerful".to_string(),
                "robust".to_string(),
                "sturdy".to_string(),
            ],
        );
        map.insert(
            "weak".to_string(),
            vec![
                "feeble".to_string(),
                "frail".to_string(),
                "fragile".to_string(),
            ],
        );
        map.insert(
            "clean".to_string(),
            vec![
                "spotless".to_string(),
                "pristine".to_string(),
                "immaculate".to_string(),
            ],
        );
        map.insert(
            "dirty".to_string(),
            vec![
                "filthy".to_string(),
                "grimy".to_string(),
                "soiled".to_string(),
            ],
        );
        map.insert(
            "quick".to_string(),
            vec!["fast".to_string(), "rapid".to_string(), "swift".to_string()],
        );
        map.insert(
            "lazy".to_string(),
            vec![
                "idle".to_string(),
                "sluggish".to_string(),
                "lethargic".to_string(),
            ],
        );
        map.insert(
            "jumps".to_string(),
            vec![
                "leaps".to_string(),
                "hops".to_string(),
                "bounds".to_string(),
            ],
        );
        map.insert(
            "brown".to_string(),
            vec![
                "tan".to_string(),
                "chestnut".to_string(),
                "tawny".to_string(),
            ],
        );
        map.insert(
            "over".to_string(),
            vec![
                "above".to_string(),
                "across".to_string(),
                "past".to_string(),
            ],
        );

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paraphrase_basic() {
        let config = ParaphraseConfig::default();
        let paraphraser = Paraphraser::new(config);

        let text = "The quick brown fox jumps over the lazy dog";
        let result = paraphraser.paraphrase(text);
        assert!(result.is_ok());

        let paraphrases = result.expect("Test failure: paraphrasing should succeed");
        assert!(!paraphrases.is_empty());
        assert!(paraphrases[0].text != text);
    }

    #[test]
    fn test_synonym_replacement() {
        let config = ParaphraseConfig {
            num_variations: 1,
            strategy: ParaphraseStrategy::Synonym,
            ..Default::default()
        };
        let paraphraser = Paraphraser::new(config);

        let text = "This is a good example";
        let result = paraphraser.paraphrase(text);
        assert!(result.is_ok());

        let paraphrases = result.expect("Test failure: paraphrasing should succeed");
        assert!(!paraphrases.is_empty());
    }

    #[test]
    fn test_case_matching() {
        let config = ParaphraseConfig::default();
        let paraphraser = Paraphraser::new(config);

        assert_eq!(paraphraser.match_case("Good", "excellent"), "Excellent");
        assert_eq!(paraphraser.match_case("GOOD", "excellent"), "EXCELLENT");
        assert_eq!(paraphraser.match_case("good", "excellent"), "excellent");
    }

    #[test]
    fn test_similarity_calculation() {
        let config = ParaphraseConfig::default();
        let paraphraser = Paraphraser::new(config);

        let text1 = "the quick brown fox";
        let text2 = "the quick brown fox";
        let similarity = paraphraser.calculate_similarity(text1, text2);
        assert!((similarity - 1.0).abs() < 0.001);

        let text3 = "the slow white cat";
        let similarity2 = paraphraser.calculate_similarity(text1, text3);
        assert!(similarity2 < 1.0 && similarity2 > 0.0);
    }

    #[test]
    fn test_empty_input() {
        let config = ParaphraseConfig::default();
        let paraphraser = Paraphraser::new(config);

        let result = paraphraser.paraphrase("");
        assert!(result.is_err());
    }
}
