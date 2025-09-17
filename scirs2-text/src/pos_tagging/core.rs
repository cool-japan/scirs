//! Core POS tagging implementation
//!
//! This module contains the main PosTagger struct and its core
//! tagging algorithms including Viterbi-like sequence tagging.

use super::lexicon::initialize_lexicon;
use super::patterns::{
    ADJ_PATTERNS, ADV_PATTERNS, ALL_CAPS_PATTERN, NOUN_PATTERNS, PROPER_NOUN_PATTERN, VERB_PATTERNS,
};
use super::types::{PosTagResult, PosTaggerConfig, PosTaggingResult};
use crate::error::Result;
use crate::stemming::PosTag;
use crate::tokenize::Tokenizer;
use std::collections::HashMap;

/// A context-aware POS tagger that uses statistical and rule-based approaches
#[derive(Debug, Clone)]
pub struct PosTagger {
    /// Dictionary of word -> most likely POS tag
    lexicon: HashMap<String, PosTag>,
    /// Transition probabilities between POS tags
    transition_probs: HashMap<(PosTag, PosTag), f64>,
    /// Emission probabilities for word given POS tag
    #[allow(dead_code)]
    emission_probs: HashMap<(String, PosTag), f64>,
    /// Use contextual information for disambiguation
    use_context: bool,
    /// Smoothing factor for unknown words
    smoothing_factor: f64,
}

impl PosTagger {
    /// Create a new POS tagger with default configuration
    pub fn new() -> Self {
        let mut tagger = Self {
            lexicon: HashMap::new(),
            transition_probs: HashMap::new(),
            emission_probs: HashMap::new(),
            use_context: true,
            smoothing_factor: 0.001,
        };

        tagger.lexicon = initialize_lexicon();
        tagger.initialize_transition_probs();
        tagger
    }

    /// Create a new POS tagger with custom configuration
    pub fn with_config(config: PosTaggerConfig) -> Self {
        let mut tagger = Self {
            lexicon: HashMap::new(),
            transition_probs: HashMap::new(),
            emission_probs: HashMap::new(),
            use_context: config.use_context,
            smoothing_factor: config.smoothing_factor,
        };

        tagger.lexicon = initialize_lexicon();
        tagger.initialize_transition_probs();
        tagger
    }

    /// Initialize transition probabilities between POS tags
    fn initialize_transition_probs(&mut self) {
        // These probabilities are based on common English grammar patterns
        let transitions = [
            // From Noun
            ((PosTag::Noun, PosTag::Verb), 0.3),
            ((PosTag::Noun, PosTag::Noun), 0.2),
            ((PosTag::Noun, PosTag::Other), 0.25), // Prepositions, conjunctions, etc.
            ((PosTag::Noun, PosTag::Adjective), 0.15),
            ((PosTag::Noun, PosTag::Adverb), 0.1),
            // From Verb
            ((PosTag::Verb, PosTag::Noun), 0.4),
            ((PosTag::Verb, PosTag::Adjective), 0.2),
            ((PosTag::Verb, PosTag::Adverb), 0.2),
            ((PosTag::Verb, PosTag::Verb), 0.1),
            ((PosTag::Verb, PosTag::Other), 0.1),
            // From Adjective
            ((PosTag::Adjective, PosTag::Noun), 0.6),
            ((PosTag::Adjective, PosTag::Verb), 0.15),
            ((PosTag::Adjective, PosTag::Adjective), 0.1),
            ((PosTag::Adjective, PosTag::Other), 0.1),
            ((PosTag::Adjective, PosTag::Adverb), 0.05),
            // From Adverb
            ((PosTag::Adverb, PosTag::Verb), 0.4),
            ((PosTag::Adverb, PosTag::Adjective), 0.3),
            ((PosTag::Adverb, PosTag::Adverb), 0.15),
            ((PosTag::Adverb, PosTag::Noun), 0.1),
            ((PosTag::Adverb, PosTag::Other), 0.05),
            // From Other (determiners, prepositions, etc.)
            ((PosTag::Other, PosTag::Noun), 0.5),
            ((PosTag::Other, PosTag::Adjective), 0.2),
            ((PosTag::Other, PosTag::Verb), 0.15),
            ((PosTag::Other, PosTag::Adverb), 0.1),
            ((PosTag::Other, PosTag::Other), 0.05),
        ];

        for ((from, to), prob) in transitions {
            self.transition_probs.insert((from, to), prob);
        }
    }

    /// Tag a single word using lexicon lookup and morphological patterns
    pub fn tag_word(&self, word: &str) -> PosTagResult {
        let lower_word = word.to_lowercase();

        // Check lexicon first
        if let Some(tag) = self.lexicon.get(&lower_word) {
            return PosTagResult {
                word: word.to_string(),
                tag: tag.clone(),
                confidence: 0.9, // High confidence for known words
            };
        }

        // Use morphological patterns for unknown words
        let (tag, confidence) = self.tag_by_morphology(word);

        PosTagResult {
            word: word.to_string(),
            tag,
            confidence,
        }
    }

    /// Tag words using morphological patterns
    fn tag_by_morphology(&self, word: &str) -> (PosTag, f64) {
        let lower_word = word.to_lowercase();

        // Check capitalization patterns first
        if PROPER_NOUN_PATTERN.is_match(word) || ALL_CAPS_PATTERN.is_match(word) {
            return (PosTag::Noun, 0.7); // Likely proper noun
        }

        // Check suffix patterns
        if ADV_PATTERNS.is_match(&lower_word) {
            return (PosTag::Adverb, 0.8);
        }

        if ADJ_PATTERNS.is_match(&lower_word) {
            return (PosTag::Adjective, 0.7);
        }

        if NOUN_PATTERNS.is_match(&lower_word) {
            return (PosTag::Noun, 0.6);
        }

        if VERB_PATTERNS.is_match(&lower_word) {
            return (PosTag::Verb, 0.6);
        }

        // Default to noun if no pattern matches (nouns are most common)
        (PosTag::Noun, 0.3)
    }

    /// Tag a sequence of tokens with contextual information
    pub fn tag_sequence(&self, tokens: &[String]) -> PosTaggingResult {
        if tokens.is_empty() {
            return PosTaggingResult {
                tokens: Vec::new(),
                tags: Vec::new(),
                confidences: Vec::new(),
            };
        }

        if !self.use_context || tokens.len() == 1 {
            // Simple word-by-word tagging without context
            let mut tags = Vec::new();
            let mut confidences = Vec::new();

            for token in tokens {
                let result = self.tag_word(token);
                tags.push(result.tag);
                confidences.push(result.confidence);
            }

            return PosTaggingResult {
                tokens: tokens.to_vec(),
                tags,
                confidences,
            };
        }

        // Use Viterbi-like algorithm for contextual tagging
        self.viterbi_tag(tokens)
    }

    /// Viterbi-like algorithm for sequence tagging with context
    fn viterbi_tag(&self, tokens: &[String]) -> PosTaggingResult {
        let n = tokens.len();
        let pos_tags = [
            PosTag::Noun,
            PosTag::Verb,
            PosTag::Adjective,
            PosTag::Adverb,
            PosTag::Other,
        ];

        // Initialize DP table: dp[i][j] = (probability, backpointer)
        let mut dp = vec![vec![(0.0f64, 0usize); pos_tags.len()]; n];
        let mut path = vec![vec![0usize; pos_tags.len()]; n];

        // Initialize first word
        for (j, tag) in pos_tags.iter().enumerate() {
            let word_result = self.tag_word(&tokens[0]);
            let emission_prob = if &word_result.tag == tag {
                word_result.confidence
            } else {
                self.smoothing_factor
            };
            dp[0][j] = (emission_prob, 0);
        }

        // Forward pass
        for i in 1..n {
            for (j, tag) in pos_tags.iter().enumerate() {
                let word_result = self.tag_word(&tokens[i]);
                let emission_prob = if &word_result.tag == tag {
                    word_result.confidence
                } else {
                    self.smoothing_factor
                };

                let mut best_prob = 0.0;
                let mut best_prev = 0;

                for (k, prev_tag) in pos_tags.iter().enumerate() {
                    let transition_prob = self
                        .transition_probs
                        .get(&(prev_tag.clone(), tag.clone()))
                        .copied()
                        .unwrap_or(self.smoothing_factor);

                    let prob = dp[i - 1][k].0 * transition_prob * emission_prob;

                    if prob > best_prob {
                        best_prob = prob;
                        best_prev = k;
                    }
                }

                dp[i][j] = (best_prob, best_prev);
                path[i][j] = best_prev;
            }
        }

        // Find best final state
        let mut best_final_prob = 0.0;
        let mut best_final_state = 0;

        for (j, _) in pos_tags.iter().enumerate() {
            if dp[n - 1][j].0 > best_final_prob {
                best_final_prob = dp[n - 1][j].0;
                best_final_state = j;
            }
        }

        // Backward pass to reconstruct path
        let mut tags = vec![PosTag::Other; n];
        let mut confidences = vec![0.0; n];
        let mut current_state = best_final_state;

        for i in (0..n).rev() {
            tags[i] = pos_tags[current_state].clone();
            confidences[i] = dp[i][current_state].0.min(1.0); // Cap at 1.0

            if i > 0 {
                current_state = path[i][current_state];
            }
        }

        // Normalize confidences to reasonable values
        let max_conf = confidences.iter().fold(0.0f64, |a, &b| a.max(b));
        if max_conf > 0.0 {
            for conf in &mut confidences {
                *conf = (*conf / max_conf).clamp(0.1, 1.0); // Keep between 0.1 and 1.0
            }
        }

        PosTaggingResult {
            tokens: tokens.to_vec(),
            tags,
            confidences,
        }
    }

    /// Tag text after tokenization
    pub fn tagtext(&self, text: &str, tokenizer: &dyn Tokenizer) -> Result<PosTaggingResult> {
        let tokens = tokenizer.tokenize(text)?;
        Ok(self.tag_sequence(&tokens))
    }

    /// Get the transition probability between two POS tags
    pub fn get_transition_probability(&self, from: &PosTag, to: &PosTag) -> f64 {
        self.transition_probs
            .get(&(from.clone(), to.clone()))
            .copied()
            .unwrap_or(self.smoothing_factor)
    }

    /// Add a custom word to the lexicon
    pub fn add_word(&mut self, word: &str, tag: PosTag) {
        self.lexicon.insert(word.to_lowercase(), tag);
    }

    /// Get the lexicon size
    pub fn lexicon_size(&self) -> usize {
        self.lexicon.len()
    }
}

impl Default for PosTagger {
    fn default() -> Self {
        Self::new()
    }
}
