//! Part-of-Speech (POS) tagging for English text
//!
//! This module provides statistical and rule-based POS tagging functionality
//! that integrates with the stemming/lemmatization system to improve accuracy.

pub mod core;
pub mod lexicon;
pub mod patterns;
pub mod types;

// Re-export main components
pub use core::PosTagger;
pub use types::{PosTagResult, PosTaggerConfig, PosTaggingResult};

// Simplified implementations of the specialized components
use crate::error::Result;
use crate::stemming::{PosTag, RuleLemmatizer};
use crate::tokenize::Tokenizer;
use scirs2_core::parallel_ops;

/// Morphological analyzer for POS disambiguation
#[derive(Debug)]
pub struct MorphologicalAnalyzer {
    /// Whether to use advanced morphological rules
    use_advanced_rules: bool,
}

impl MorphologicalAnalyzer {
    /// Create new morphological analyzer
    pub fn new() -> Self {
        Self {
            use_advanced_rules: true,
        }
    }

    /// Predict POS tag from morphological features
    pub fn predict_pos(&self, word: &str) -> Option<(PosTag, f64)> {
        patterns::PatternMatcher::predict_from_morphology(word)
            .map(|tag| (tag, 0.7))
            .or_else(|| {
                patterns::PatternMatcher::predict_from_capitalization(word).map(|tag| (tag, 0.6))
            })
    }

    /// Analyze morphological features
    pub fn analyze_features(&self, word: &str) -> Vec<String> {
        let mut features = Vec::new();

        if patterns::PatternMatcher::matches_pattern(word, "verb") {
            features.push("VERB_PATTERN".to_string());
        }
        if patterns::PatternMatcher::matches_pattern(word, "noun") {
            features.push("NOUN_PATTERN".to_string());
        }
        if patterns::PatternMatcher::matches_pattern(word, "adjective") {
            features.push("ADJ_PATTERN".to_string());
        }
        if patterns::PatternMatcher::matches_pattern(word, "adverb") {
            features.push("ADV_PATTERN".to_string());
        }

        features
    }
}

impl Default for MorphologicalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Contextual disambiguator for POS tagging
#[derive(Debug)]
pub struct ContextualDisambiguator {
    /// Context window size
    window_size: usize,
}

impl ContextualDisambiguator {
    /// Create new contextual disambiguator
    pub fn new() -> Self {
        Self { window_size: 3 }
    }

    /// Disambiguate POS tag using context
    pub fn disambiguate(
        &self,
        tokens: &[String],
        index: usize,
        initial_tag: PosTag,
    ) -> (PosTag, f64) {
        if tokens.is_empty() || index >= tokens.len() {
            return (initial_tag, 0.5);
        }

        // Simple context-based rules
        let word = &tokens[index];

        // Check for common disambiguation patterns
        if index > 0 {
            let prev_word = &tokens[index - 1];
            if (prev_word == "the" || prev_word == "a" || prev_word == "an")
                && initial_tag == PosTag::Verb
            {
                return (PosTag::Noun, 0.8); // "the running" -> "running" is likely a noun
            }
        }

        if index + 1 < tokens.len() {
            let next_word = &tokens[index + 1];
            if (next_word == "is" || next_word == "was" || next_word == "are")
                && word.ends_with("ing")
            {
                return (PosTag::Verb, 0.8); // "running is" -> "running" is likely a verb
            }
        }

        (initial_tag, 0.6)
    }

    /// Set context window size
    pub fn set_window_size(&mut self, size: usize) {
        self.window_size = size;
    }
}

impl Default for ContextualDisambiguator {
    fn default() -> Self {
        Self::new()
    }
}

/// POS-aware lemmatizer that uses POS tags to improve lemmatization
#[derive(Debug)]
pub struct PosAwareLemmatizer {
    /// Base lemmatizer rules
    use_pos_rules: bool,
    /// Underlying rule-based lemmatizer for exceptions and dictionary lookups
    rule_lemmatizer: RuleLemmatizer,
}

impl PosAwareLemmatizer {
    /// Create new POS-aware lemmatizer
    pub fn new() -> Self {
        Self {
            use_pos_rules: true,
            rule_lemmatizer: RuleLemmatizer::new(),
        }
    }

    /// Create POS-aware lemmatizer with custom configurations
    pub fn with_configs(
        _pos_config: types::PosTaggerConfig,
        _lemma_config: crate::stemming::LemmatizerConfig,
    ) -> Self {
        Self {
            use_pos_rules: true,
            rule_lemmatizer: RuleLemmatizer::with_config(_lemma_config),
        }
    }

    /// Lemmatize word using POS tag information
    pub fn lemmatize(&self, word: &str, pos_tag: &PosTag) -> String {
        let lower_word = word.to_lowercase();

        // First consult the rule-based lemmatizer (covers exceptions and dictionary)
        let rl_result = self.rule_lemmatizer.lemmatize(word, Some(pos_tag.clone()));
        if rl_result.to_lowercase() != lower_word {
            return rl_result;
        }

        // Fallback to simple rule-based transformations when no exception/dict match
        match pos_tag {
            PosTag::Verb => self.lemmatize_verb(&lower_word),
            PosTag::Noun => self.lemmatize_noun(&lower_word),
            PosTag::Adjective => self.lemmatize_adjective(&lower_word),
            PosTag::Adverb => self.lemmatize_adverb(&lower_word),
            PosTag::Other => lower_word,
        }
    }

    fn lemmatize_verb(&self, word: &str) -> String {
        // Simple verb lemmatization rules
        if word.ends_with("ing") && word.len() > 4 {
            let stem = &word[..word.len() - 3];
            if stem.ends_with(&stem[stem.len() - 1..]) {
                // Double consonant
                return stem[..stem.len() - 1].to_string();
            }
            return stem.to_string();
        }

        if word.ends_with("ed") && word.len() > 3 {
            let stem = &word[..word.len() - 2];
            if stem.ends_with("i") {
                return format!("{}y", &stem[..stem.len() - 1]);
            }
            return stem.to_string();
        }

        if word.ends_with("s") && word.len() > 2 {
            return word[..word.len() - 1].to_string();
        }

        word.to_string()
    }

    fn lemmatize_noun(&self, word: &str) -> String {
        // Simple noun lemmatization (plurals)
        if word.ends_with("ies") && word.len() > 4 {
            return format!("{}y", &word[..word.len() - 3]);
        }

        if word.ends_with("es") && word.len() > 3 {
            let stem = &word[..word.len() - 2];
            if stem.ends_with("s")
                || stem.ends_with("x")
                || stem.ends_with("z")
                || stem.ends_with("ch")
                || stem.ends_with("sh")
            {
                return stem.to_string();
            }
        }

        if word.ends_with("s") && word.len() > 2 && !word.ends_with("ss") {
            return word[..word.len() - 1].to_string();
        }

        word.to_string()
    }

    fn lemmatize_adjective(&self, word: &str) -> String {
        // Simple adjective lemmatization
        if word.ends_with("er") && word.len() > 3 {
            return word[..word.len() - 2].to_string();
        }

        if word.ends_with("est") && word.len() > 4 {
            return word[..word.len() - 3].to_string();
        }

        word.to_string()
    }

    fn lemmatize_adverb(&self, word: &str) -> String {
        // Simple adverb lemmatization
        if word.ends_with("ly") && word.len() > 3 {
            let stem = &word[..word.len() - 2];
            if stem.ends_with("i") {
                return format!("{}y", &stem[..stem.len() - 1]);
            }
            return stem.to_string();
        }

        word.to_string()
    }
}

impl Default for PosAwareLemmatizer {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Stemmer trait for PosAwareLemmatizer for compatibility
impl crate::stemming::Stemmer for PosAwareLemmatizer {
    fn stem(&self, word: &str) -> Result<String> {
        // Try to detect if it's a verb based on simple heuristics
        let lower_word = word.to_lowercase();
        let pos_tag = if lower_word.ends_with("ing")
            || lower_word.ends_with("ed")
            || lower_word.ends_with("es")
        {
            PosTag::Verb
        } else {
            PosTag::Noun
        };
        Ok(self.lemmatize(word, &pos_tag))
    }
}

/// Parallel processing utilities for POS tagging
impl PosTagger {
    /// Tag multiple texts in parallel
    pub fn tagtexts_parallel(
        &self,
        texts: &[&str],
        tokenizer: &dyn Tokenizer,
    ) -> Result<Vec<PosTaggingResult>> {
        // Sequential implementation for now - parallel would require Send+Sync tokenizer
        texts
            .iter()
            .map(|text| self.tagtext(text, tokenizer))
            .collect()
    }

    /// Tag multiple token sequences in parallel
    pub fn tag_sequences_parallel(&self, token_sequences: &[Vec<String>]) -> Vec<PosTaggingResult> {
        // Sequential implementation - can be made parallel since no shared tokenizer
        token_sequences
            .iter()
            .map(|tokens| self.tag_sequence(tokens))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WhitespaceTokenizer;

    #[test]
    fn test_pos_tagger_basic() {
        let tagger = PosTagger::new();
        let result = tagger.tag_word("running");
        assert_eq!(result.word, "running");
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_pos_tagger_sequence() {
        let tagger = PosTagger::new();
        let tokens = vec![
            "The".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];
        let result = tagger.tag_sequence(&tokens);
        assert_eq!(result.tokens.len(), 4);
        assert_eq!(result.tags.len(), 4);
        assert_eq!(result.confidences.len(), 4);
    }

    #[test]
    fn test_morphological_analyzer() {
        let analyzer = MorphologicalAnalyzer::new();
        let result = analyzer.predict_pos("running");
        assert!(result.is_some());

        let features = analyzer.analyze_features("quickly");
        assert!(!features.is_empty());
    }

    #[test]
    fn test_contextual_disambiguator() {
        let disambiguator = ContextualDisambiguator::new();
        let tokens = vec![
            "the".to_string(),
            "running".to_string(),
            "water".to_string(),
        ];
        let (tag, conf) = disambiguator.disambiguate(&tokens, 1, PosTag::Verb);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_pos_aware_lemmatizer() {
        let lemmatizer = PosAwareLemmatizer::new();

        let verb_result = lemmatizer.lemmatize("running", &PosTag::Verb);
        assert_eq!(verb_result, "run");

        let noun_result = lemmatizer.lemmatize("cats", &PosTag::Noun);
        assert_eq!(noun_result, "cat");

        let adj_result = lemmatizer.lemmatize("faster", &PosTag::Adjective);
        assert_eq!(adj_result, "fast");
    }

    #[test]
    fn test_parallel_tagging() {
        let tagger = PosTagger::new();
        let tokenizer = WhitespaceTokenizer::new();
        let texts = vec!["Hello world", "The quick brown fox"];

        let results = tagger.tagtexts_parallel(&texts, &tokenizer).unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results[0].tokens.is_empty());
        assert!(!results[1].tokens.is_empty());
    }
}
