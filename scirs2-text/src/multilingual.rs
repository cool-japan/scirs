//! Multilingual text processing and language detection
//!
//! This module provides functionality for detecting languages
//! and processing text in multiple languages.

use crate::error::{Result, TextError};
use std::collections::HashMap;

/// Supported languages for detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// English
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Italian
    Italian,
    /// Portuguese
    Portuguese,
    /// Dutch
    Dutch,
    /// Russian
    Russian,
    /// Chinese
    Chinese,
    /// Japanese
    Japanese,
    /// Korean
    Korean,
    /// Arabic
    Arabic,
    /// Unknown language
    Unknown,
}

impl Language {
    /// Get the ISO 639-1 code for the language
    pub fn iso_code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Dutch => "nl",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Unknown => "und",
        }
    }

    /// Get the language from ISO 639-1 code
    pub fn from_iso_code(code: &str) -> Self {
        match code.to_lowercase().as_str() {
            "en" => Language::English,
            "es" => Language::Spanish,
            "fr" => Language::French,
            "de" => Language::German,
            "it" => Language::Italian,
            "pt" => Language::Portuguese,
            "nl" => Language::Dutch,
            "ru" => Language::Russian,
            "zh" => Language::Chinese,
            "ja" => Language::Japanese,
            "ko" => Language::Korean,
            "ar" => Language::Arabic,
            _ => Language::Unknown,
        }
    }

    /// Get the full name of the language
    pub fn name(&self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Italian => "Italian",
            Language::Portuguese => "Portuguese",
            Language::Dutch => "Dutch",
            Language::Russian => "Russian",
            Language::Chinese => "Chinese",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::Arabic => "Arabic",
            Language::Unknown => "Unknown",
        }
    }
}

/// Result of language detection
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    /// The detected language
    pub language: Language,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Alternative language candidates with scores
    pub alternatives: Vec<(Language, f64)>,
}

/// Language detector using character n-gram profiles
pub struct LanguageDetector {
    /// Character n-gram profiles for each language
    profiles: HashMap<Language, HashMap<String, f64>>,
    /// N-gram size (typically 2 or 3)
    n_gram_size: usize,
}

impl LanguageDetector {
    /// Create a new language detector with default profiles
    pub fn new() -> Self {
        let mut detector = Self {
            profiles: HashMap::new(),
            n_gram_size: 3,
        };
        detector.initialize_default_profiles();
        detector
    }

    /// Create a language detector with custom n-gram size
    pub fn with_ngram_size(n_gram_size: usize) -> Result<Self> {
        if !(1..=5).contains(&n_gram_size) {
            return Err(TextError::InvalidInput(
                "N-gram size must be between 1 and 5".to_string(),
            ));
        }
        let mut detector = Self {
            profiles: HashMap::new(),
            n_gram_size,
        };
        detector.initialize_default_profiles();
        Ok(detector)
    }

    /// Initialize default language profiles with common n-grams
    fn initialize_default_profiles(&mut self) {
        // English profile
        let mut english_profile = HashMap::new();
        for (ngram, freq) in &[
            ("the", 0.05),
            ("and", 0.03),
            ("ing", 0.025),
            ("ion", 0.02),
            ("tio", 0.018),
            ("ent", 0.015),
            ("ati", 0.013),
            ("her", 0.012),
            ("for", 0.011),
            ("ter", 0.01),
            ("hat", 0.009),
            ("tha", 0.009),
            ("ere", 0.008),
            ("ate", 0.008),
            ("ver", 0.007),
            ("his", 0.007),
        ] {
            english_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::English, english_profile);

        // Spanish profile
        let mut spanish_profile = HashMap::new();
        for (ngram, freq) in &[
            ("que", 0.04),
            ("de_", 0.035),
            ("la_", 0.03),
            ("el_", 0.025),
            ("es_", 0.02),
            ("los", 0.018),
            ("las", 0.015),
            ("ión", 0.013),
            ("ado", 0.012),
            ("nte", 0.011),
            ("con", 0.01),
            ("par", 0.009),
            ("ara", 0.008),
            ("una", 0.008),
            ("por", 0.007),
            ("est", 0.007),
        ] {
            spanish_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Spanish, spanish_profile);

        // French profile
        let mut french_profile = HashMap::new();
        for (ngram, freq) in &[
            ("de_", 0.05),
            ("le_", 0.04),
            ("que", 0.03),
            ("les", 0.025),
            ("la_", 0.02),
            ("des", 0.018),
            ("ent", 0.015),
            ("ion", 0.013),
            ("est", 0.012),
            ("ait", 0.011),
            ("pour", 0.01),
            ("ais", 0.009),
            ("ans", 0.008),
            ("ont", 0.008),
            ("une", 0.007),
            ("qui", 0.007),
        ] {
            french_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::French, french_profile);

        // German profile
        let mut german_profile = HashMap::new();
        for (ngram, freq) in &[
            ("der", 0.05),
            ("die", 0.04),
            ("und", 0.03),
            ("den", 0.025),
            ("das", 0.02),
            ("ein", 0.018),
            ("ich", 0.015),
            ("ist", 0.013),
            ("sch", 0.012),
            ("cht", 0.011),
            ("ung", 0.01),
            ("gen", 0.009),
            ("eit", 0.008),
            ("ver", 0.008),
            ("ber", 0.007),
            ("ten", 0.007),
        ] {
            german_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::German, german_profile);
    }

    /// Detect the language of a text
    pub fn detect(&self, text: &str) -> Result<LanguageDetectionResult> {
        if text.trim().is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot detect language of empty text".to_string(),
            ));
        }

        // Extract n-grams from the text
        let text_profile = self.create_text_profile(text);

        // Score each language profile
        let mut scores: Vec<(Language, f64)> = self
            .profiles
            .iter()
            .map(|(lang, profile)| {
                let score = self.calculate_similarity(&text_profile, profile);
                (*lang, score)
            })
            .collect();

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if scores.is_empty() {
            return Ok(LanguageDetectionResult {
                language: Language::Unknown,
                confidence: 0.0,
                alternatives: vec![],
            });
        }

        let best_score = scores[0].1;
        let best_language = scores[0].0;

        // Calculate confidence based on the difference between top scores
        let confidence = if scores.len() > 1 {
            let second_score = scores[1].1;
            let diff = best_score - second_score;
            // Normalize confidence to [0, 1]
            (diff / best_score).clamp(0.0, 1.0)
        } else {
            best_score
        };

        Ok(LanguageDetectionResult {
            language: best_language,
            confidence,
            alternatives: scores.into_iter().skip(1).take(3).collect(),
        })
    }

    /// Create n-gram profile for a text
    fn create_text_profile(&self, text: &str) -> HashMap<String, f64> {
        let mut profile = HashMap::new();
        let text_lower = text.to_lowercase();
        let chars: Vec<char> = text_lower.chars().collect();
        let total_ngrams = chars.len().saturating_sub(self.n_gram_size - 1) as f64;

        if total_ngrams <= 0.0 {
            return profile;
        }

        // Count n-grams
        let mut ngram_counts: HashMap<String, usize> = HashMap::new();
        for i in 0..=chars.len().saturating_sub(self.n_gram_size) {
            let ngram: String = chars[i..i + self.n_gram_size].iter().collect();
            // Replace spaces with underscores for consistency
            let ngram = ngram.replace(' ', "_");
            *ngram_counts.entry(ngram).or_insert(0) += 1;
        }

        // Convert counts to frequencies
        for (ngram, count) in ngram_counts {
            profile.insert(ngram, count as f64 / total_ngrams);
        }

        profile
    }

    /// Calculate similarity between two n-gram profiles
    fn calculate_similarity(
        &self,
        profile1: &HashMap<String, f64>,
        profile2: &HashMap<String, f64>,
    ) -> f64 {
        let mut similarity = 0.0;
        let mut total_weight = 0.0;

        // Use cosine similarity
        for (ngram, freq1) in profile1 {
            if let Some(freq2) = profile2.get(ngram) {
                similarity += freq1 * freq2;
            }
            total_weight += freq1 * freq1;
        }

        if total_weight > 0.0 {
            similarity / total_weight.sqrt()
        } else {
            0.0
        }
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> Vec<Language> {
        self.profiles.keys().copied().collect()
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Language-specific stop words
pub struct StopWords {
    /// Stop words organized by language
    stop_words: HashMap<Language, Vec<String>>,
}

impl StopWords {
    /// Create a new stop words collection
    pub fn new() -> Self {
        let mut stop_words = HashMap::new();

        // English stop words
        stop_words.insert(
            Language::English,
            vec![
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
                "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "you",
                "your", "this", "have", "had", "been", "but", "not", "they", "were", "what",
                "when", "where", "who", "which", "their", "them", "these", "those", "there",
                "here", "than",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Spanish stop words
        stop_words.insert(
            Language::Spanish,
            vec![
                "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
                "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
                "ellas", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres", "es",
                "esa", "esas", "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos",
                "fue", "fueron", "fui", "la", "las", "lo", "los", "más", "mi", "mis", "mucho",
                "muchos", "muy", "ni", "no", "nos", "nosotras", "nosotros", "o", "otra", "otras",
                "otro", "otros", "para", "pero", "por", "porque", "que", "quien", "quienes", "se",
                "si", "sin", "sobre", "su", "sus", "también", "tanto", "te", "tu", "tus", "un",
                "una", "uno", "unos", "y", "ya", "yo",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // French stop words
        stop_words.insert(
            Language::French,
            vec![
                "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et",
                "eux", "il", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "même",
                "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas",
                "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes",
                "toi", "ton", "tu", "un", "une", "vos", "votre", "vous",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        Self { stop_words }
    }

    /// Get stop words for a specific language
    pub fn get(&self, language: Language) -> Option<&Vec<String>> {
        self.stop_words.get(&language)
    }

    /// Check if a word is a stop word in a specific language
    pub fn is_stop_word(&self, word: &str, language: Language) -> bool {
        if let Some(words) = self.stop_words.get(&language) {
            words.iter().any(|sw| sw == &word.to_lowercase())
        } else {
            false
        }
    }

    /// Remove stop words from a list of tokens
    pub fn remove_stop_words(&self, tokens: &[String], language: Language) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| !self.is_stop_word(token, language))
            .cloned()
            .collect()
    }
}

impl Default for StopWords {
    fn default() -> Self {
        Self::new()
    }
}

/// Language-specific text processor
pub struct MultilingualProcessor {
    /// Language detector
    detector: LanguageDetector,
    /// Stop words collection
    stop_words: StopWords,
}

impl MultilingualProcessor {
    /// Create a new multilingual processor
    pub fn new() -> Self {
        Self {
            detector: LanguageDetector::new(),
            stop_words: StopWords::new(),
        }
    }

    /// Process text with automatic language detection
    pub fn process(&self, text: &str) -> Result<ProcessedText> {
        // Detect language
        let detection = self.detector.detect(text)?;

        // Tokenize (simple whitespace tokenization for now)
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        // Remove stop words
        let filtered_tokens = self
            .stop_words
            .remove_stop_words(&tokens, detection.language);

        Ok(ProcessedText {
            original: text.to_string(),
            language: detection.language,
            confidence: detection.confidence,
            tokens,
            filtered_tokens,
        })
    }
}

impl Default for MultilingualProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of multilingual text processing
#[derive(Debug, Clone)]
pub struct ProcessedText {
    /// Original text
    pub original: String,
    /// Detected language
    pub language: Language,
    /// Language detection confidence
    pub confidence: f64,
    /// All tokens
    pub tokens: Vec<String>,
    /// Tokens after stop word removal
    pub filtered_tokens: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_enum() {
        assert_eq!(Language::English.iso_code(), "en");
        assert_eq!(Language::Spanish.name(), "Spanish");
        assert_eq!(Language::from_iso_code("fr"), Language::French);
        assert_eq!(Language::from_iso_code("unknown"), Language::Unknown);
    }

    #[test]
    fn test_language_detection() {
        let detector = LanguageDetector::new();

        // Test English detection with more text
        let result = detector.detect("The quick brown fox jumps over the lazy dog. This is definitely an English sentence with many common words.").unwrap();
        assert_eq!(result.language, Language::English);

        // Test with empty text
        let empty_result = detector.detect("");
        assert!(empty_result.is_err());
    }

    #[test]
    fn test_stop_words() {
        let stop_words = StopWords::new();

        // Test English stop words
        assert!(stop_words.is_stop_word("the", Language::English));
        assert!(stop_words.is_stop_word("and", Language::English));
        assert!(!stop_words.is_stop_word("hello", Language::English));

        // Test stop word removal
        let tokens = vec![
            "the".to_string(),
            "cat".to_string(),
            "is".to_string(),
            "happy".to_string(),
        ];
        let filtered = stop_words.remove_stop_words(&tokens, Language::English);
        assert_eq!(filtered, vec!["cat", "happy"]);
    }

    #[test]
    fn test_multilingual_processor() {
        let processor = MultilingualProcessor::new();

        let result = processor.process("The quick brown fox jumps over the lazy dog. This sentence has many English words.").unwrap();
        assert_eq!(result.language, Language::English);
        assert!(!result.tokens.is_empty());
        assert!(result.filtered_tokens.len() < result.tokens.len());
    }

    #[test]
    fn test_create_text_profile() {
        let detector = LanguageDetector::new();
        let profile = detector.create_text_profile("hello world");

        // Check that profile contains some n-grams
        assert!(!profile.is_empty());
        assert!(profile.contains_key("hel") || profile.contains_key("llo"));
    }
}
