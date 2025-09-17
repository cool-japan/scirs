//! Pattern matching for POS tagging
//!
//! This module contains regex patterns and morphological rules
//! used for POS tag disambiguation.

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    // Common word patterns for POS disambiguation
    pub static ref VERB_PATTERNS: Regex = Regex::new(r"(?i)(ing|ed|s)$").unwrap();
    pub static ref NOUN_PATTERNS: Regex = Regex::new(r"(?i)(tion|sion|ness|ment|ship|hood|ity|cy|th|ing|er|or|ar|ist|ism|age|al|ance|ence|dom|tude|ure|ery|ary|ory|ly)$").unwrap();
    pub static ref ADJ_PATTERNS: Regex = Regex::new(r"(?i)(ful|less|ous|ious|eous|ary|ory|ic|ical|al|able|ible|ive|ative|itive|ent|ant|ed|ing|er|est|ward)$").unwrap();
    pub static ref ADV_PATTERNS: Regex = Regex::new(r"(?i)(ly|ward|wise|like)$").unwrap();

    // Capitalization patterns
    pub static ref PROPER_NOUN_PATTERN: Regex = Regex::new(r"^[A-Z][a-z]+$").unwrap();
    pub static ref ALL_CAPS_PATTERN: Regex = Regex::new(r"^[A-Z]{2,}$").unwrap();
}

/// Pattern-based POS tag predictor
pub struct PatternMatcher;

impl PatternMatcher {
    /// Predict POS tag based on morphological patterns
    pub fn predict_from_morphology(word: &str) -> Option<crate::stemming::PosTag> {
        use crate::stemming::PosTag;

        // Check for adjective patterns first (more specific)
        if ADJ_PATTERNS.is_match(word) {
            return Some(PosTag::Adjective);
        }

        // Check for adverb patterns
        if ADV_PATTERNS.is_match(word) {
            return Some(PosTag::Adverb);
        }

        // Check for noun patterns
        if NOUN_PATTERNS.is_match(word) {
            return Some(PosTag::Noun);
        }

        // Check for verb patterns (less specific, check last)
        if VERB_PATTERNS.is_match(word) {
            return Some(PosTag::Verb);
        }

        None
    }

    /// Predict POS tag based on capitalization patterns
    pub fn predict_from_capitalization(word: &str) -> Option<crate::stemming::PosTag> {
        use crate::stemming::PosTag;

        if PROPER_NOUN_PATTERN.is_match(word) {
            Some(PosTag::Noun) // Proper nouns
        } else if ALL_CAPS_PATTERN.is_match(word) {
            Some(PosTag::Noun) // Acronyms are typically nouns
        } else {
            None
        }
    }

    /// Check if word matches a specific pattern type
    pub fn matches_pattern(word: &str, pattern_type: &str) -> bool {
        match pattern_type {
            "verb" => VERB_PATTERNS.is_match(word),
            "noun" => NOUN_PATTERNS.is_match(word),
            "adjective" => ADJ_PATTERNS.is_match(word),
            "adverb" => ADV_PATTERNS.is_match(word),
            "proper_noun" => PROPER_NOUN_PATTERN.is_match(word),
            "all_caps" => ALL_CAPS_PATTERN.is_match(word),
            _ => false,
        }
    }
}
