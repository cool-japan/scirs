//! Sentiment analysis functionality
//!
//! This module provides sentiment analysis capabilities including
//! lexicon-based and rule-based sentiment analysis.

use crate::error::Result;
use crate::tokenize::{Tokenizer, WordTokenizer};
use std::collections::HashMap;

/// Sentiment polarity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sentiment {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
}

impl std::fmt::Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentiment::Positive => write!(f, "Positive"),
            Sentiment::Negative => write!(f, "Negative"),
            Sentiment::Neutral => write!(f, "Neutral"),
        }
    }
}

impl Sentiment {
    /// Convert sentiment to a numerical score
    pub fn to_score(&self) -> f64 {
        match self {
            Sentiment::Positive => 1.0,
            Sentiment::Neutral => 0.0,
            Sentiment::Negative => -1.0,
        }
    }

    /// Convert a numerical score to sentiment
    pub fn from_score(score: f64) -> Self {
        if score > 0.0 {
            Sentiment::Positive
        } else if score < 0.0 {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        }
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// The overall sentiment
    pub sentiment: Sentiment,
    /// The raw sentiment score
    pub score: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Breakdown of positive and negative word counts
    pub word_counts: SentimentWordCounts,
}

/// Word counts for sentiment analysis
#[derive(Debug, Clone, Default)]
pub struct SentimentWordCounts {
    /// Number of positive words
    pub positive_words: usize,
    /// Number of negative words
    pub negative_words: usize,
    /// Number of neutral words
    pub neutral_words: usize,
    /// Total number of words analyzed
    pub total_words: usize,
}

/// A sentiment lexicon mapping words to sentiment scores
#[derive(Debug, Clone)]
pub struct SentimentLexicon {
    /// Word to sentiment score mapping
    lexicon: HashMap<String, f64>,
    /// Default score for unknown words
    default_score: f64,
}

impl SentimentLexicon {
    /// Create a new sentiment lexicon
    pub fn new() -> Self {
        Self {
            lexicon: HashMap::new(),
            default_score: 0.0,
        }
    }

    /// Create a basic sentiment lexicon with common words
    pub fn with_basic_lexicon() -> Self {
        let mut lexicon = HashMap::new();

        // Positive words
        let positive_words = [
            ("good", 1.0),
            ("great", 2.0),
            ("excellent", 3.0),
            ("amazing", 3.0),
            ("wonderful", 2.5),
            ("fantastic", 2.5),
            ("love", 2.0),
            ("like", 1.0),
            ("happy", 2.0),
            ("joy", 2.0),
            ("pleased", 1.5),
            ("satisfied", 1.0),
            ("positive", 1.0),
            ("perfect", 3.0),
            ("best", 2.5),
            ("awesome", 2.5),
            ("beautiful", 2.0),
            ("brilliant", 2.5),
            ("superb", 2.5),
            ("nice", 1.0),
        ];

        // Negative words
        let negative_words = [
            ("bad", -1.0),
            ("terrible", -2.5),
            ("awful", -2.5),
            ("horrible", -3.0),
            ("hate", -2.5),
            ("dislike", -1.5),
            ("sad", -2.0),
            ("unhappy", -2.0),
            ("disappointed", -2.0),
            ("negative", -1.0),
            ("worst", -3.0),
            ("poor", -1.5),
            ("disgusting", -3.0),
            ("ugly", -2.0),
            ("nasty", -2.5),
            ("stupid", -2.0),
            ("pathetic", -2.5),
            ("failure", -2.0),
            ("fail", -2.0),
            ("sucks", -2.0),
        ];

        for (word, score) in &positive_words {
            lexicon.insert(word.to_string(), *score);
        }

        for (word, score) in &negative_words {
            lexicon.insert(word.to_string(), *score);
        }

        Self {
            lexicon,
            default_score: 0.0,
        }
    }

    /// Add a word to the lexicon
    pub fn add_word(&mut self, word: String, score: f64) {
        self.lexicon.insert(word.to_lowercase(), score);
    }

    /// Get the sentiment score for a word
    pub fn get_score(&self, word: &str) -> f64 {
        self.lexicon
            .get(&word.to_lowercase())
            .copied()
            .unwrap_or(self.default_score)
    }

    /// Check if a word is in the lexicon
    pub fn contains(&self, word: &str) -> bool {
        self.lexicon.contains_key(&word.to_lowercase())
    }

    /// Get the size of the lexicon
    pub fn len(&self) -> usize {
        self.lexicon.len()
    }

    /// Check if the lexicon is empty
    pub fn is_empty(&self) -> bool {
        self.lexicon.is_empty()
    }
}

impl Default for SentimentLexicon {
    fn default() -> Self {
        Self::new()
    }
}

/// Lexicon-based sentiment analyzer
pub struct LexiconSentimentAnalyzer {
    /// The sentiment lexicon
    lexicon: SentimentLexicon,
    /// The tokenizer to use
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Negation words that reverse sentiment
    negation_words: Vec<String>,
    /// Window size for negation detection
    negation_window: usize,
}

impl LexiconSentimentAnalyzer {
    /// Create a new lexicon-based sentiment analyzer
    pub fn new(lexicon: SentimentLexicon) -> Self {
        let negation_words = vec![
            "not".to_string(),
            "no".to_string(),
            "never".to_string(),
            "neither".to_string(),
            "nobody".to_string(),
            "nothing".to_string(),
            "nowhere".to_string(),
            "n't".to_string(),
            "cannot".to_string(),
            "without".to_string(),
        ];

        Self {
            lexicon,
            tokenizer: Box::new(WordTokenizer::default()),
            negation_words,
            negation_window: 3,
        }
    }

    /// Create an analyzer with a basic lexicon
    pub fn with_basic_lexicon() -> Self {
        Self::new(SentimentLexicon::with_basic_lexicon())
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Analyze the sentiment of a text
    pub fn analyze(&self, text: &str) -> Result<SentimentResult> {
        let tokens = self.tokenizer.tokenize(text)?;

        if tokens.is_empty() {
            return Ok(SentimentResult {
                sentiment: Sentiment::Neutral,
                score: 0.0,
                confidence: 0.0,
                word_counts: SentimentWordCounts {
                    positive_words: 0,
                    negative_words: 0,
                    neutral_words: 0,
                    total_words: 0,
                },
            });
        }

        let mut total_score = 0.0;
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;

        // Analyze each token
        for (i, token) in tokens.iter().enumerate() {
            let token_lower = token.to_lowercase();
            let mut score = self.lexicon.get_score(&token_lower);

            // Check for negation
            if score != 0.0 {
                for j in 1..=self.negation_window.min(i) {
                    let prev_token = &tokens[i - j].to_lowercase();
                    if self.negation_words.contains(prev_token) {
                        score *= -1.0;
                        break;
                    }
                }
            }

            total_score += score;

            if score > 0.0 {
                positive_count += 1;
            } else if score < 0.0 {
                negative_count += 1;
            } else {
                neutral_count += 1;
            }
        }

        let total_words = tokens.len();
        let sentiment = Sentiment::from_score(total_score);

        // Calculate confidence based on the proportion of sentiment-bearing words
        let sentiment_words = positive_count + negative_count;
        let confidence = if total_words > 0 {
            (sentiment_words as f64 / total_words as f64).min(1.0)
        } else {
            0.0
        };

        Ok(SentimentResult {
            sentiment,
            score: total_score,
            confidence,
            word_counts: SentimentWordCounts {
                positive_words: positive_count,
                negative_words: negative_count,
                neutral_words: neutral_count,
                total_words,
            },
        })
    }

    /// Analyze sentiment for multiple texts
    pub fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<SentimentResult>> {
        texts.iter().map(|&text| self.analyze(text)).collect()
    }
}

/// Rule-based sentiment modifications
#[derive(Debug, Clone)]
pub struct SentimentRules {
    /// Intensifier words that increase sentiment magnitude
    intensifiers: HashMap<String, f64>,
    /// Diminisher words that decrease sentiment magnitude
    diminishers: HashMap<String, f64>,
}

impl Default for SentimentRules {
    fn default() -> Self {
        let mut intensifiers = HashMap::new();
        intensifiers.insert("very".to_string(), 1.5);
        intensifiers.insert("extremely".to_string(), 2.0);
        intensifiers.insert("incredibly".to_string(), 2.0);
        intensifiers.insert("really".to_string(), 1.3);
        intensifiers.insert("so".to_string(), 1.3);
        intensifiers.insert("absolutely".to_string(), 2.0);

        let mut diminishers = HashMap::new();
        diminishers.insert("somewhat".to_string(), 0.5);
        diminishers.insert("slightly".to_string(), 0.5);
        diminishers.insert("barely".to_string(), 0.3);
        diminishers.insert("hardly".to_string(), 0.3);
        diminishers.insert("a little".to_string(), 0.5);

        Self {
            intensifiers,
            diminishers,
        }
    }
}

impl SentimentRules {
    /// Apply rules to modify a sentiment score
    pub fn apply(&self, tokens: &[String], base_scores: &[f64]) -> Vec<f64> {
        let mut modified_scores = base_scores.to_vec();

        for (i, score) in modified_scores.iter_mut().enumerate() {
            if *score == 0.0 {
                continue;
            }

            // Check for intensifiers/diminishers in the preceding words
            for j in 1..=2.min(i) {
                let prev_token = &tokens[i - j].to_lowercase();

                if let Some(&multiplier) = self.intensifiers.get(prev_token) {
                    *score *= multiplier;
                    break;
                } else if let Some(&multiplier) = self.diminishers.get(prev_token) {
                    *score *= multiplier;
                    break;
                }
            }
        }

        modified_scores
    }
}

/// Advanced rule-based sentiment analyzer
pub struct RuleBasedSentimentAnalyzer {
    /// The base analyzer
    base_analyzer: LexiconSentimentAnalyzer,
    /// Sentiment modification rules
    rules: SentimentRules,
}

impl RuleBasedSentimentAnalyzer {
    /// Create a new rule-based sentiment analyzer
    pub fn new(lexicon: SentimentLexicon) -> Self {
        Self {
            base_analyzer: LexiconSentimentAnalyzer::new(lexicon),
            rules: SentimentRules::default(),
        }
    }

    /// Create an analyzer with a basic lexicon
    pub fn with_basic_lexicon() -> Self {
        Self::new(SentimentLexicon::with_basic_lexicon())
    }

    /// Analyze sentiment with rule modifications
    pub fn analyze(&self, text: &str) -> Result<SentimentResult> {
        let tokens = self.base_analyzer.tokenizer.tokenize(text)?;

        if tokens.is_empty() {
            return self.base_analyzer.analyze(text);
        }

        // Get base scores for each token
        let base_scores: Vec<f64> = tokens
            .iter()
            .map(|token| self.base_analyzer.lexicon.get_score(token))
            .collect();

        // Apply rules to modify scores
        let modified_scores = self.rules.apply(&tokens, &base_scores);

        // Calculate final sentiment
        let total_score: f64 = modified_scores.iter().sum();
        let sentiment = Sentiment::from_score(total_score);

        // Count sentiment words
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;

        for &score in &modified_scores {
            if score > 0.0 {
                positive_count += 1;
            } else if score < 0.0 {
                negative_count += 1;
            } else {
                neutral_count += 1;
            }
        }

        let total_words = tokens.len();
        let sentiment_words = positive_count + negative_count;
        let confidence = if total_words > 0 {
            (sentiment_words as f64 / total_words as f64).min(1.0)
        } else {
            0.0
        };

        Ok(SentimentResult {
            sentiment,
            score: total_score,
            confidence,
            word_counts: SentimentWordCounts {
                positive_words: positive_count,
                negative_words: negative_count,
                neutral_words: neutral_count,
                total_words,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_lexicon() {
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_word("happy".to_string(), 2.0);
        lexicon.add_word("sad".to_string(), -2.0);

        assert_eq!(lexicon.get_score("happy"), 2.0);
        assert_eq!(lexicon.get_score("sad"), -2.0);
        assert_eq!(lexicon.get_score("unknown"), 0.0);
    }

    #[test]
    fn test_basic_sentiment_analysis() {
        let analyzer = LexiconSentimentAnalyzer::with_basic_lexicon();

        let positive_result = analyzer.analyze("This is a wonderful day!").unwrap();
        assert_eq!(positive_result.sentiment, Sentiment::Positive);
        assert!(positive_result.score > 0.0);

        let negative_result = analyzer.analyze("This is terrible and awful").unwrap();
        assert_eq!(negative_result.sentiment, Sentiment::Negative);
        assert!(negative_result.score < 0.0);

        let neutral_result = analyzer.analyze("This is a book").unwrap();
        assert_eq!(neutral_result.sentiment, Sentiment::Neutral);
        assert_eq!(neutral_result.score, 0.0);
    }

    #[test]
    fn test_negation_handling() {
        let analyzer = LexiconSentimentAnalyzer::with_basic_lexicon();

        let negated_result = analyzer.analyze("This is not good").unwrap();
        assert_eq!(negated_result.sentiment, Sentiment::Negative);
        assert!(negated_result.score < 0.0);
    }

    #[test]
    fn test_rule_based_sentiment() {
        let analyzer = RuleBasedSentimentAnalyzer::with_basic_lexicon();

        let intensified_result = analyzer.analyze("This is very good").unwrap();
        let normal_result = analyzer.analyze("This is good").unwrap();

        assert!(intensified_result.score > normal_result.score);
    }

    #[test]
    fn test_sentiment_batch_analysis() {
        let analyzer = LexiconSentimentAnalyzer::with_basic_lexicon();
        let texts = vec!["I love this", "I hate this", "This is okay"];

        let results = analyzer.analyze_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].sentiment, Sentiment::Positive);
        assert_eq!(results[1].sentiment, Sentiment::Negative);
    }
}
