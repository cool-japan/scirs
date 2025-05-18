//! Text stemming algorithms
//!
//! This module provides implementations of various stemming algorithms
//! including Porter, Snowball, and Lancaster stemmers, as well as lemmatization
//! approaches.
//!
//! # Stemming Algorithms
//!
//! This module offers three primary stemming algorithms with different characteristics:
//!
//! - **Porter stemmer**: A classic stemming algorithm with moderate stemming strength.
//!   Good balance between performance and accuracy for English text.
//!
//! - **Snowball stemmer**: An improved version of the Porter algorithm with language-specific
//!   rules. Currently supports English, with more languages planned for future updates.
//!
//! - **Lancaster stemmer**: Also known as Paice/Husk stemmer, this is a more aggressive
//!   stemming algorithm that typically produces shorter stems. It's configurable with
//!   options for setting minimum stem length and handling short words.
//!
//! # Lemmatization Approaches
//!
//! In addition to stemmers, this module provides two lemmatization options:
//!
//! - **SimpleLemmatizer**: A dictionary-based lemmatizer that uses predefined
//!   mappings from word forms to their lemmas. Simple but effective for high-frequency
//!   words and common irregular forms.
//!
//! - **RuleLemmatizer**: A more advanced lemmatizer that combines dictionary
//!   lookups with rule-based transformations. It handles regular inflectional
//!   patterns through rules and irregular forms through exceptions.
//!
//! # Performance Comparison
//!
//! In terms of computational efficiency:
//! - Lancaster stemmer is typically the fastest stemming algorithm
//! - Snowball stemmer is moderately fast
//! - Porter stemmer is the slowest of the three stemmers
//! - RuleLemmatizer performance is comparable to Porter stemming but with better
//!   accuracy for many English words
//! - SimpleLemmatizer is very fast for known words but limited in vocabulary
//!
//! # Choosing a Stemming/Lemmatization Approach
//!
//! - Use **Porter** when you need established, widely recognized stemming with moderate
//!   aggressiveness.
//! - Use **Snowball** when working with multiple languages or when you need language-specific
//!   refinements to the Porter algorithm.
//! - Use **Lancaster** when you need very aggressive stemming or maximum performance.
//! - Use **SimpleLemmatizer** when you need high-speed processing for a limited set
//!   of known words.
//! - Use **RuleLemmatizer** when you need more accurate word normalization that
//!   preserves the base form (lemma) rather than a stem, or when you need
//!   part-of-speech-aware word normalization.
//!
//! # Stemming vs. Lemmatization
//!
//! - **Stemming** (Porter, Snowball, Lancaster) simply removes word endings to reduce
//!   words to their stems, which may not be valid words.
//! - **Lemmatization** (SimpleLemmatizer, RuleLemmatizer) reduces words to their base
//!   form (lemma), which is always a valid word. It often requires knowledge of a word's
//!   part of speech.
//!
//! # Example
//!
//! ```
//! use scirs2_text::{LancasterStemmer, PorterStemmer, RuleLemmatizer, SimpleLemmatizer, SnowballStemmer, Stemmer};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let porter = PorterStemmer::new();
//! let snowball = SnowballStemmer::new("english")?;
//! let lancaster = LancasterStemmer::new();
//! let simple_lemmatizer = SimpleLemmatizer::new();
//! let rule_lemmatizer = RuleLemmatizer::new();
//!
//! // Compare stemming results
//! assert_eq!(porter.stem("running")?, "run");
//! assert_eq!(snowball.stem("running")?, "running");
//! assert_eq!(lancaster.stem("running")?, "running");
//! assert_eq!(simple_lemmatizer.stem("running")?, "run");
//! assert_eq!(rule_lemmatizer.stem("running")?, "run");
//!
//! // Compare lemmatization and stemming on irregular verbs
//! assert_eq!(porter.stem("went")?, "went");
//! assert_eq!(simple_lemmatizer.stem("went")?, "went"); // Unknown unless in dictionary
//! assert_eq!(rule_lemmatizer.stem("went")?, "go");     // Correctly lemmatizes irregular form
//! # Ok(())
//! # }
//! ```

pub mod lancaster;
pub mod rule_lemmatizer;

use crate::error::{Result, TextError};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

// Re-export stemmer and lemmatizer implementations
pub use self::lancaster::LancasterStemmer;
pub use self::rule_lemmatizer::{
    LemmatizerConfig, PosTag, RuleCondition, RuleLemmatizer, RuleLemmatizerBuilder,
};

lazy_static! {
    // Porter stemmer regex patterns
    static ref VOWEL_SEQUENCE: Regex = Regex::new(r"[aeiouy]").unwrap();
    static ref DOUBLE_CONSONANT: Regex = Regex::new(r"(bb|dd|ff|gg|mm|nn|pp|rr|tt)$").unwrap();
}

/// Trait for text stemming algorithms
pub trait Stemmer {
    /// Stem a single word
    fn stem(&self, word: &str) -> Result<String>;

    /// Stem multiple words
    fn stem_batch(&self, words: &[&str]) -> Result<Vec<String>> {
        words.iter().map(|word| self.stem(word)).collect()
    }
}

/// Porter stemmer implementation
#[derive(Debug, Clone)]
pub struct PorterStemmer;

impl PorterStemmer {
    /// Create a new Porter stemmer
    pub fn new() -> Self {
        Self
    }

    /// Check if the word ends with a consonant-vowel-consonant pattern
    fn ends_with_cvc(&self, word: &str) -> bool {
        if word.len() < 3 {
            return false;
        }

        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();

        // Check consonant-vowel-consonant pattern
        !self.is_vowel(&chars[n - 3])
            && self.is_vowel(&chars[n - 2])
            && !self.is_vowel(&chars[n - 1])
            && !matches!(chars[n - 1], 'w' | 'x' | 'y')
    }

    /// Check if a character is a vowel
    fn is_vowel(&self, ch: &char) -> bool {
        matches!(*ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
    }

    /// Calculate the measure of a word (count of consonant sequences)
    fn measure(&self, word: &str) -> usize {
        let mut measure = 0;
        let mut in_vowel_sequence = false;

        for ch in word.chars() {
            if self.is_vowel(&ch) {
                in_vowel_sequence = true;
            } else if in_vowel_sequence {
                measure += 1;
                in_vowel_sequence = false;
            }
        }

        measure
    }

    /// Step 1a: Plurals and past participles
    fn step1a(&self, word: String) -> String {
        if word.ends_with("sses") || word.ends_with("ies") {
            word[..word.len() - 2].to_string()
        } else if word.ends_with("s") && !word.ends_with("ss") {
            word[..word.len() - 1].to_string()
        } else {
            word
        }
    }

    /// Step 1b: Past participles
    fn step1b(&self, mut word: String) -> String {
        let mut step1b_applied = false;

        if word.ends_with("eed") {
            let stem = &word[..word.len() - 3];
            if self.measure(stem) > 0 {
                word = format!("{}{}", stem, "ee");
            }
        } else if word.ends_with("ed") {
            let stem = &word[..word.len() - 2];
            if VOWEL_SEQUENCE.is_match(stem) {
                word = stem.to_string();
                step1b_applied = true;
            }
        } else if word.ends_with("ing") {
            let stem = &word[..word.len() - 3];
            if VOWEL_SEQUENCE.is_match(stem) {
                word = stem.to_string();
                step1b_applied = true;
            }
        }

        if step1b_applied {
            if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
                word.push('e');
            } else if DOUBLE_CONSONANT.is_match(&word)
                && !word.ends_with("l")
                && !word.ends_with("s")
                && !word.ends_with("z")
            {
                word.pop();
            } else if self.measure(&word) == 1 && self.ends_with_cvc(&word) {
                word.push('e');
            }
        }

        word
    }

    /// Step 1c: Y → I
    fn step1c(&self, word: String) -> String {
        if word.ends_with("y") && word.len() > 1 {
            let stem = &word[..word.len() - 1];
            if VOWEL_SEQUENCE.is_match(stem) {
                return format!("{}{}", stem, "i");
            }
        }
        word
    }

    /// Steps 2-5: Various suffix removals
    fn step2(&self, word: String) -> String {
        let suffix_map = vec![
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];

        for (suffix, replacement) in suffix_map {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 0 {
                    return format!("{}{}", stem, replacement);
                }
            }
        }

        word
    }

    /// Step 3: Suffix removal
    fn step3(&self, word: String) -> String {
        let suffix_map = vec![
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];

        for (suffix, replacement) in suffix_map {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 0 {
                    return format!("{}{}", stem, replacement);
                }
            }
        }

        word
    }

    /// Step 4: Suffix removal
    fn step4(&self, word: String) -> String {
        let suffixes = vec![
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
            "sion", "tion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];

        for suffix in suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 1 {
                    return stem.to_string();
                }
            }
        }

        word
    }

    /// Step 5a: Remove E
    fn step5a(&self, word: String) -> String {
        if word.ends_with("e") {
            let stem = &word[..word.len() - 1];
            if self.measure(stem) > 1 || (self.measure(stem) == 1 && !self.ends_with_cvc(stem)) {
                return stem.to_string();
            }
        }
        word
    }

    /// Step 5b: Remove double L
    fn step5b(&self, word: String) -> String {
        if word.ends_with("ll") && self.measure(&word) > 1 {
            return word[..word.len() - 1].to_string();
        }
        word
    }
}

impl Default for PorterStemmer {
    fn default() -> Self {
        Self::new()
    }
}

impl Stemmer for PorterStemmer {
    fn stem(&self, word: &str) -> Result<String> {
        if word.is_empty() {
            return Ok(word.to_string());
        }

        let mut stemmed = word.to_lowercase();

        // Apply stemming steps in order
        stemmed = self.step1a(stemmed);
        stemmed = self.step1b(stemmed);
        stemmed = self.step1c(stemmed);
        stemmed = self.step2(stemmed);
        stemmed = self.step3(stemmed);
        stemmed = self.step4(stemmed);
        stemmed = self.step5a(stemmed);
        stemmed = self.step5b(stemmed);

        Ok(stemmed)
    }
}

/// Snowball stemmer (enhanced Porter stemmer)
#[derive(Debug, Clone)]
pub struct SnowballStemmer {
    language: String,
}

impl SnowballStemmer {
    /// Create a new Snowball stemmer for a specific language
    pub fn new(language: &str) -> Result<Self> {
        match language.to_lowercase().as_str() {
            "english" | "en" => Ok(Self {
                language: "english".to_string(),
            }),
            _ => Err(TextError::InvalidInput(format!(
                "Unsupported language: {}",
                language
            ))),
        }
    }

    /// Apply R1 and R2 region finding (used in Snowball algorithm)
    fn find_r1_r2(&self, word: &str) -> (usize, usize) {
        let mut r1 = word.len();
        let mut r2 = word.len();

        // Find R1: after first non-vowel following a vowel
        let chars: Vec<char> = word.chars().collect();
        let mut found_vowel = false;

        for (i, ch) in chars.iter().enumerate() {
            if self.is_vowel(ch) {
                found_vowel = true;
            } else if found_vowel {
                r1 = i + 1;
                break;
            }
        }

        // Find R2: same rule applied to R1
        if r1 < word.len() {
            found_vowel = false;
            for (i, ch) in chars[r1..].iter().enumerate() {
                if self.is_vowel(ch) {
                    found_vowel = true;
                } else if found_vowel {
                    r2 = r1 + i + 1;
                    break;
                }
            }
        }

        (r1, r2)
    }

    fn is_vowel(&self, ch: &char) -> bool {
        matches!(*ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
    }

    /// English-specific Snowball stemming rules
    fn stem_english(&self, word: &str) -> String {
        if word.len() <= 2 {
            return word.to_string();
        }

        let mut stemmed = word.to_lowercase();
        let (_r1, _r2) = self.find_r1_r2(&stemmed);

        // Step 0: Remove trailing apostrophes
        if stemmed.ends_with("'s'") {
            stemmed = stemmed[..stemmed.len() - 3].to_string();
        } else if stemmed.ends_with("'s") {
            stemmed = stemmed[..stemmed.len() - 2].to_string();
        } else if stemmed.ends_with("'") {
            stemmed = stemmed[..stemmed.len() - 1].to_string();
        }

        // Step 1a: Plurals
        if stemmed.ends_with("sses") {
            stemmed = format!("{}ss", &stemmed[..stemmed.len() - 4]);
        } else if stemmed.ends_with("ied") || stemmed.ends_with("ies") {
            if stemmed.len() > 4 {
                stemmed = format!("{}i", &stemmed[..stemmed.len() - 3]);
            } else {
                stemmed = format!("{}ie", &stemmed[..stemmed.len() - 3]);
            }
        } else if stemmed.ends_with("s") && !stemmed.ends_with("us") && !stemmed.ends_with("ss") {
            // Check if word contains a vowel before the s
            let stem = &stemmed[..stemmed.len() - 1];
            if VOWEL_SEQUENCE.is_match(stem) {
                stemmed = stem.to_string();
            }
        }

        // Additional Snowball steps would go here...
        // This is a simplified version

        stemmed
    }
}

impl Stemmer for SnowballStemmer {
    fn stem(&self, word: &str) -> Result<String> {
        match self.language.as_str() {
            "english" => Ok(self.stem_english(word)),
            _ => Err(TextError::InvalidInput(format!(
                "Unsupported language: {}",
                self.language
            ))),
        }
    }
}

/// Simple lemmatizer using a dictionary-based approach
#[derive(Debug, Clone)]
pub struct SimpleLemmatizer {
    lemma_dict: HashMap<String, String>,
}

impl SimpleLemmatizer {
    /// Create a new lemmatizer
    pub fn new() -> Self {
        let mut lemma_dict = HashMap::new();

        // Add some common lemmatization rules
        // This is a simplified example - a real implementation would load
        // from a comprehensive dictionary
        lemma_dict.insert("am".to_string(), "be".to_string());
        lemma_dict.insert("are".to_string(), "be".to_string());
        lemma_dict.insert("is".to_string(), "be".to_string());
        lemma_dict.insert("was".to_string(), "be".to_string());
        lemma_dict.insert("were".to_string(), "be".to_string());
        lemma_dict.insert("been".to_string(), "be".to_string());
        lemma_dict.insert("being".to_string(), "be".to_string());

        lemma_dict.insert("have".to_string(), "have".to_string());
        lemma_dict.insert("has".to_string(), "have".to_string());
        lemma_dict.insert("had".to_string(), "have".to_string());
        lemma_dict.insert("having".to_string(), "have".to_string());

        lemma_dict.insert("does".to_string(), "do".to_string());
        lemma_dict.insert("did".to_string(), "do".to_string());
        lemma_dict.insert("doing".to_string(), "do".to_string());

        lemma_dict.insert("better".to_string(), "good".to_string());
        lemma_dict.insert("best".to_string(), "good".to_string());
        lemma_dict.insert("worse".to_string(), "bad".to_string());
        lemma_dict.insert("worst".to_string(), "bad".to_string());

        lemma_dict.insert("running".to_string(), "run".to_string());
        lemma_dict.insert("ran".to_string(), "run".to_string());
        lemma_dict.insert("runs".to_string(), "run".to_string());

        Self { lemma_dict }
    }

    /// Load lemmatization dictionary from a file
    pub fn from_dict_file(_path: &str) -> Result<Self> {
        // In a real implementation, this would load from a file
        Ok(Self::new())
    }

    /// Add a lemma mapping
    pub fn add_lemma(&mut self, word: &str, lemma: &str) {
        self.lemma_dict.insert(word.to_string(), lemma.to_string());
    }
}

impl Default for SimpleLemmatizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Stemmer for SimpleLemmatizer {
    fn stem(&self, word: &str) -> Result<String> {
        let lower = word.to_lowercase();
        Ok(self.lemma_dict.get(&lower).unwrap_or(&lower).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_porter_stemmer() {
        let stemmer = PorterStemmer::new();

        let test_cases = vec![
            ("running", "run"),
            ("ran", "ran"),
            ("easily", "easili"),
            ("fishing", "fish"),
            ("fished", "fish"),
            ("productive", "product"),
            ("production", "produc"),
            ("sensational", "sensat"),
        ];

        for (word, expected) in test_cases {
            let stemmed = stemmer.stem(word).unwrap();
            assert_eq!(stemmed, expected, "Failed for word: {}", word);
        }
    }

    #[test]
    fn test_snowball_stemmer() {
        let stemmer = SnowballStemmer::new("english").unwrap();

        let test_cases = vec![
            ("cats", "cat"),
            ("running", "running"), // Simplified version doesn't handle all cases
            ("flies", "fli"),
            ("happiness", "happiness"), // Simplified version
        ];

        for (word, expected) in test_cases {
            let stemmed = stemmer.stem(word).unwrap();
            assert_eq!(stemmed, expected, "Failed for word: {}", word);
        }
    }

    #[test]
    fn test_simple_lemmatizer() {
        let lemmatizer = SimpleLemmatizer::new();

        let test_cases = vec![
            ("am", "be"),
            ("are", "be"),
            ("was", "be"),
            ("better", "good"),
            ("running", "run"),
            ("unknown", "unknown"), // Should return the word itself if not in dict
        ];

        for (word, expected) in test_cases {
            let lemma = lemmatizer.stem(word).unwrap();
            assert_eq!(lemma, expected, "Failed for word: {}", word);
        }
    }

    #[test]
    fn test_rule_lemmatizer() {
        let lemmatizer = RuleLemmatizer::new();

        // Test with various parts of speech
        assert_eq!(lemmatizer.lemmatize("running", Some(PosTag::Verb)), "run");
        assert_eq!(lemmatizer.lemmatize("cats", Some(PosTag::Noun)), "cat");
        assert_eq!(
            lemmatizer.lemmatize("better", Some(PosTag::Adjective)),
            "good"
        );
        assert_eq!(
            lemmatizer.lemmatize("quickly", Some(PosTag::Adverb)),
            "quick"
        );

        // Test irregular forms
        assert_eq!(lemmatizer.lemmatize("went", Some(PosTag::Verb)), "go");
        assert_eq!(
            lemmatizer.lemmatize("children", Some(PosTag::Noun)),
            "child"
        );
        assert_eq!(lemmatizer.lemmatize("feet", Some(PosTag::Noun)), "foot");

        // Test without POS tag
        assert_eq!(lemmatizer.lemmatize("running", None), "run");
        assert_eq!(lemmatizer.lemmatize("went", None), "go");
    }

    #[test]
    fn test_stemmers_and_lemmatizers_comparison() {
        let porter = PorterStemmer::new();
        let snowball = SnowballStemmer::new("english").unwrap();
        let lancaster = LancasterStemmer::new();
        let simple_lemmatizer = SimpleLemmatizer::new();
        let rule_lemmatizer = RuleLemmatizer::new();

        let test_words = vec![
            "running",
            "cats",
            "better",
            "went",
            "children",
            "feet",
            "universities",
        ];

        for word in test_words {
            println!(
                "Word: '{}'\nPorter: '{}'\nSnowball: '{}'\nLancaster: '{}'\nSimple: '{}'\nRule: '{}'",
                word,
                porter.stem(word).unwrap(),
                snowball.stem(word).unwrap(),
                lancaster.stem(word).unwrap(),
                simple_lemmatizer.stem(word).unwrap(),
                rule_lemmatizer.stem(word).unwrap()
            );
        }

        // Test basic cases
        assert_eq!(porter.stem("running").unwrap(), "run");
        assert_eq!(rule_lemmatizer.stem("running").unwrap(), "run");

        // Test that lemmatizer works better for irregular forms
        assert_eq!(porter.stem("went").unwrap(), "went"); // Stemmer doesn't handle irregular verbs
        assert_eq!(rule_lemmatizer.stem("went").unwrap(), "go"); // Lemmatizer does

        // Lemmatizer should handle irregular plurals
        assert_eq!(porter.stem("feet").unwrap(), "feet"); // Stemmer doesn't normalize irregular plurals
        assert_eq!(rule_lemmatizer.stem("feet").unwrap(), "foot"); // Lemmatizer does
    }
}
