//! Text classification pipeline implementations
//!
//! This module provides text classification pipelines including
//! standard binary/multi-class classification and zero-shot classification.

use super::ClassificationResult;
use crate::error::Result;

/// Text classification pipeline
#[derive(Debug)]
pub struct TextClassificationPipeline {
    /// Labels for classification
    #[allow(dead_code)]
    labels: Vec<String>,
}

impl Default for TextClassificationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl TextClassificationPipeline {
    /// Create new text classification pipeline
    pub fn new() -> Self {
        Self {
            labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
        }
    }

    /// Run classification on text
    pub fn predict(&self, text: &str) -> Result<Vec<ClassificationResult>> {
        // Use the existing sentiment analysis functionality for more realistic predictions
        use crate::sentiment::{LexiconSentimentAnalyzer, Sentiment, SentimentLexicon};

        let analyzer = LexiconSentimentAnalyzer::new(SentimentLexicon::with_basiclexicon());
        let sentiment_result = analyzer.analyze(text)?;

        // Convert sentiment result to classification format
        let (label, confidence) = match sentiment_result.sentiment {
            Sentiment::Positive => ("POSITIVE", sentiment_result.confidence),
            Sentiment::Negative => ("NEGATIVE", sentiment_result.confidence),
            Sentiment::Neutral => {
                // For binary classification, lean towards positive for neutral based on word counts
                let positive_ratio = sentiment_result.word_counts.positive_words as f64
                    / (sentiment_result.word_counts.total_words as f64).max(1.0);
                let negative_ratio = sentiment_result.word_counts.negative_words as f64
                    / (sentiment_result.word_counts.total_words as f64).max(1.0);

                if positive_ratio >= negative_ratio {
                    ("POSITIVE", 0.5 + (positive_ratio - negative_ratio) / 2.0)
                } else {
                    ("NEGATIVE", 0.5 + (negative_ratio - positive_ratio) / 2.0)
                }
            }
        };

        // Also provide the alternative label with lower confidence
        let alternative_label = if label == "POSITIVE" {
            "NEGATIVE"
        } else {
            "POSITIVE"
        };
        let alternative_confidence = 1.0 - confidence;

        Ok(vec![
            ClassificationResult {
                label: label.to_string(),
                score: confidence,
            },
            ClassificationResult {
                label: alternative_label.to_string(),
                score: alternative_confidence,
            },
        ])
    }
}

/// Zero-shot classification pipeline
#[derive(Debug)]
pub struct ZeroShotClassificationPipeline {
    /// Hypothesis template
    hypothesis_template: String,
}

impl Default for ZeroShotClassificationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroShotClassificationPipeline {
    /// Create new zero-shot classification pipeline
    pub fn new() -> Self {
        Self {
            hypothesis_template: "This example is {}.".to_string(),
        }
    }

    /// Classify text against multiple labels
    pub fn classify(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<ClassificationResult>> {
        let mut results = Vec::new();

        // Enhanced zero-shot classification using text similarity and keyword matching
        use crate::distance::cosine_similarity;
        use crate::tokenize::WhitespaceTokenizer;
        use crate::vectorize::{CountVectorizer, Vectorizer};

        let tokenizer = WhitespaceTokenizer::new();
        let mut vectorizer = CountVectorizer::with_tokenizer(Box::new(tokenizer), false);

        // Create corpus with text and hypotheses for each label
        let mut corpus = vec![text];
        let hypotheses: Vec<String> = candidate_labels
            .iter()
            .map(|label| self.hypothesis_template.replace("{}", label))
            .collect();
        corpus.extend(hypotheses.iter().map(|h| h.as_str()));

        // Vectorize the corpus
        if let Ok(vectors) = vectorizer.fit_transform(&corpus) {
            let text_vector = vectors.row(0);

            for (i, &label) in candidate_labels.iter().enumerate() {
                let hypothesis_vector = vectors.row(i + 1);

                // Calculate cosine similarity between text and hypothesis
                let similarity = cosine_similarity(text_vector, hypothesis_vector).unwrap_or(0.0);

                // Enhance with keyword matching
                let text_lower = text.to_lowercase();
                let label_lower = label.to_lowercase();
                let keyword_bonus = if text_lower.contains(&label_lower) {
                    0.2
                } else {
                    0.0
                };

                let score = (similarity + keyword_bonus).clamp(0.0, 1.0);

                results.push(ClassificationResult {
                    label: label.to_string(),
                    score,
                });
            }
        } else {
            // Fallback to simple keyword matching if vectorization fails
            for &label in candidate_labels {
                let text_lower = text.to_lowercase();
                let label_lower = label.to_lowercase();

                let score = if text_lower.contains(&label_lower) {
                    0.8
                } else {
                    // Basic similarity based on common words
                    let text_words: std::collections::HashSet<_> =
                        text_lower.split_whitespace().collect();
                    let label_words: std::collections::HashSet<_> =
                        label_lower.split_whitespace().collect();
                    let common_words = text_words.intersection(&label_words).count();
                    let total_words = text_words.union(&label_words).count();

                    if total_words > 0 {
                        common_words as f64 / total_words as f64
                    } else {
                        0.1
                    }
                };

                results.push(ClassificationResult {
                    label: label.to_string(),
                    score,
                });
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    /// Set hypothesis template
    pub fn set_hypothesis_template(&mut self, template: String) {
        self.hypothesis_template = template;
    }
}
