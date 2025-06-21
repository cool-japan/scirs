//! Natural Language Processing domain metrics
//!
//! This module provides specialized metric collections for NLP tasks including
//! text generation, classification, similarity, and sequence labeling.

use crate::classification::{accuracy_score, f1_score, precision_score, recall_score};
use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::Array1;
use std::collections::{HashMap, HashSet};

/// Text generation evaluation results
#[derive(Debug, Clone)]
pub struct TextGenerationResults {
    /// BLEU-1 score
    pub bleu_1: f64,
    /// BLEU-4 score
    pub bleu_4: f64,
    /// ROUGE-L score
    pub rouge_l: f64,
    /// ROUGE-1 score
    pub rouge_1: f64,
    /// ROUGE-2 score
    pub rouge_2: f64,
    /// Perplexity (if probabilities available)
    pub perplexity: Option<f64>,
    /// METEOR score (simplified)
    pub meteor: f64,
}

/// Text classification evaluation results
#[derive(Debug, Clone)]
pub struct TextClassificationResults {
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro-averaged F1 score
    pub macro_f1: f64,
    /// Micro-averaged F1 score
    pub micro_f1: f64,
    /// Weighted F1 score
    pub weighted_f1: f64,
    /// Per-class metrics
    pub per_class_metrics: HashMap<i32, ClassMetrics>,
}

/// Per-class metrics for text classification
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

/// Named Entity Recognition evaluation results
#[derive(Debug, Clone)]
pub struct NERResults {
    /// Entity-level precision
    pub entity_precision: f64,
    /// Entity-level recall
    pub entity_recall: f64,
    /// Entity-level F1 score
    pub entity_f1: f64,
    /// Token-level precision
    pub token_precision: f64,
    /// Token-level recall
    pub token_recall: f64,
    /// Token-level F1 score
    pub token_f1: f64,
    /// Per-entity-type metrics
    pub per_type_metrics: HashMap<String, ClassMetrics>,
}

/// Sentiment analysis evaluation results
#[derive(Debug, Clone)]
pub struct SentimentResults {
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro F1 score
    pub macro_f1: f64,
    /// Positive class precision
    pub positive_precision: f64,
    /// Positive class recall
    pub positive_recall: f64,
    /// Negative class precision
    pub negative_precision: f64,
    /// Negative class recall
    pub negative_recall: f64,
}

/// Text generation metrics calculator
pub struct TextGenerationMetrics {
    smooth_bleu: bool,
    max_ngram: usize,
}

impl Default for TextGenerationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGenerationMetrics {
    /// Create new text generation metrics calculator
    pub fn new() -> Self {
        Self {
            smooth_bleu: true,
            max_ngram: 4,
        }
    }

    /// Enable/disable BLEU smoothing
    pub fn with_bleu_smoothing(mut self, smooth: bool) -> Self {
        self.smooth_bleu = smooth;
        self
    }

    /// Set maximum n-gram for BLEU calculation
    pub fn with_max_ngram(mut self, max_ngram: usize) -> Self {
        self.max_ngram = max_ngram;
        self
    }

    /// Evaluate text generation quality
    pub fn evaluate_generation(
        &self,
        references: &[String],
        candidates: &[String],
    ) -> Result<TextGenerationResults> {
        if references.len() != candidates.len() {
            return Err(MetricsError::InvalidInput(
                "References and candidates must have same length".to_string(),
            ));
        }

        let mut bleu_1_scores = Vec::new();
        let mut bleu_4_scores = Vec::new();
        let mut rouge_l_scores = Vec::new();
        let mut rouge_1_scores = Vec::new();
        let mut rouge_2_scores = Vec::new();
        let mut meteor_scores = Vec::new();

        for (reference, candidate) in references.iter().zip(candidates.iter()) {
            // Tokenize sentences
            let ref_tokens = self.tokenize(reference);
            let cand_tokens = self.tokenize(candidate);

            // Calculate BLEU scores
            bleu_1_scores.push(self.calculate_bleu(&ref_tokens, &cand_tokens, 1));
            bleu_4_scores.push(self.calculate_bleu(&ref_tokens, &cand_tokens, 4));

            // Calculate ROUGE scores
            rouge_l_scores.push(self.calculate_rouge_l(&ref_tokens, &cand_tokens));
            rouge_1_scores.push(self.calculate_rouge_n(&ref_tokens, &cand_tokens, 1));
            rouge_2_scores.push(self.calculate_rouge_n(&ref_tokens, &cand_tokens, 2));

            // Calculate METEOR score (simplified)
            meteor_scores.push(self.calculate_meteor(&ref_tokens, &cand_tokens));
        }

        Ok(TextGenerationResults {
            bleu_1: bleu_1_scores.iter().sum::<f64>() / bleu_1_scores.len() as f64,
            bleu_4: bleu_4_scores.iter().sum::<f64>() / bleu_4_scores.len() as f64,
            rouge_l: rouge_l_scores.iter().sum::<f64>() / rouge_l_scores.len() as f64,
            rouge_1: rouge_1_scores.iter().sum::<f64>() / rouge_1_scores.len() as f64,
            rouge_2: rouge_2_scores.iter().sum::<f64>() / rouge_2_scores.len() as f64,
            perplexity: None, // Would need language model probabilities
            meteor: meteor_scores.iter().sum::<f64>() / meteor_scores.len() as f64,
        })
    }

    /// Simple tokenization (split by whitespace and punctuation)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Calculate BLEU score
    fn calculate_bleu(&self, reference: &[String], candidate: &[String], n: usize) -> f64 {
        if candidate.is_empty() || reference.is_empty() {
            return 0.0;
        }

        let mut precisions = Vec::new();

        for i in 1..=n {
            let ref_ngrams = self.get_ngrams(reference, i);
            let cand_ngrams = self.get_ngrams(candidate, i);

            if cand_ngrams.is_empty() {
                precisions.push(0.0);
                continue;
            }

            let mut matches = 0;
            for ngram in &cand_ngrams {
                if ref_ngrams.contains(ngram) {
                    matches += 1;
                }
            }

            let precision = matches as f64 / cand_ngrams.len() as f64;
            precisions.push(precision);
        }

        // Geometric mean of precisions
        let geo_mean = if precisions.iter().all(|&p| p > 0.0) {
            precisions
                .iter()
                .product::<f64>()
                .powf(1.0 / precisions.len() as f64)
        } else {
            0.0
        };

        // Brevity penalty
        let bp = if candidate.len() < reference.len() {
            (1.0 - reference.len() as f64 / candidate.len() as f64).exp()
        } else {
            1.0
        };

        bp * geo_mean
    }

    /// Calculate ROUGE-L score (Longest Common Subsequence)
    fn calculate_rouge_l(&self, reference: &[String], candidate: &[String]) -> f64 {
        let lcs_length = self.lcs_length(reference, candidate);

        if reference.is_empty() && candidate.is_empty() {
            return 1.0;
        }

        if reference.is_empty() || candidate.is_empty() {
            return 0.0;
        }

        let precision = lcs_length as f64 / candidate.len() as f64;
        let recall = lcs_length as f64 / reference.len() as f64;

        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    /// Calculate ROUGE-N score
    fn calculate_rouge_n(&self, reference: &[String], candidate: &[String], n: usize) -> f64 {
        let ref_ngrams = self.get_ngrams(reference, n);
        let cand_ngrams = self.get_ngrams(candidate, n);

        if ref_ngrams.is_empty() {
            return if cand_ngrams.is_empty() { 1.0 } else { 0.0 };
        }

        let mut matches = 0;
        for ngram in &cand_ngrams {
            if ref_ngrams.contains(ngram) {
                matches += 1;
            }
        }

        matches as f64 / ref_ngrams.len() as f64
    }

    /// Calculate simplified METEOR score
    fn calculate_meteor(&self, reference: &[String], candidate: &[String]) -> f64 {
        if reference.is_empty() && candidate.is_empty() {
            return 1.0;
        }

        if reference.is_empty() || candidate.is_empty() {
            return 0.0;
        }

        // Count exact matches
        let ref_set: HashSet<_> = reference.iter().collect();
        let cand_set: HashSet<_> = candidate.iter().collect();
        let matches = ref_set.intersection(&cand_set).count();

        let precision = matches as f64 / candidate.len() as f64;
        let recall = matches as f64 / reference.len() as f64;

        if precision + recall > 0.0 {
            // Simplified METEOR (without stemming, synonyms, paraphrase matching)
            10.0 * precision * recall / (9.0 * precision + recall)
        } else {
            0.0
        }
    }

    /// Get n-grams from a sequence
    fn get_ngrams(&self, tokens: &[String], n: usize) -> Vec<Vec<String>> {
        if tokens.len() < n {
            return vec![];
        }

        (0..=tokens.len() - n)
            .map(|i| tokens[i..i + n].to_vec())
            .collect()
    }

    /// Calculate Longest Common Subsequence length
    fn lcs_length(&self, a: &[String], b: &[String]) -> usize {
        let mut dp = vec![vec![0; b.len() + 1]; a.len() + 1];

        for i in 1..=a.len() {
            for j in 1..=b.len() {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[a.len()][b.len()]
    }
}

/// Text classification metrics calculator
pub struct TextClassificationMetrics;

impl Default for TextClassificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TextClassificationMetrics {
    /// Create new text classification metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Evaluate text classification performance
    pub fn evaluate_classification(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<TextClassificationResults> {
        let accuracy = accuracy_score(y_true, y_pred)?;

        // Get unique classes
        let classes: Vec<i32> = {
            let mut classes = y_true
                .iter()
                .chain(y_pred.iter())
                .copied()
                .collect::<Vec<_>>();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        // Calculate per-class metrics
        let mut per_class_metrics = HashMap::new();
        let mut class_f1_scores = Vec::new();
        let mut class_weights = Vec::new();

        for &class in &classes {
            let precision = precision_score(y_true, y_pred, class)?;
            let recall = recall_score(y_true, y_pred, class)?;
            let f1 = f1_score(y_true, y_pred, class)?;
            let support = y_true.iter().filter(|&&label| label == class).count();

            per_class_metrics.insert(
                class,
                ClassMetrics {
                    precision,
                    recall,
                    f1_score: f1,
                    support,
                },
            );

            class_f1_scores.push(f1);
            class_weights.push(support);
        }

        // Calculate macro F1 (unweighted average)
        let macro_f1 = if !class_f1_scores.is_empty() {
            class_f1_scores.iter().sum::<f64>() / class_f1_scores.len() as f64
        } else {
            0.0
        };

        // Calculate micro F1 (globally computed)
        let micro_f1 = accuracy; // For multi-class, micro F1 equals accuracy

        // Calculate weighted F1
        let total_support: usize = class_weights.iter().sum();
        let weighted_f1 = if total_support > 0 {
            class_f1_scores
                .iter()
                .zip(class_weights.iter())
                .map(|(f1, weight)| f1 * (*weight as f64))
                .sum::<f64>()
                / total_support as f64
        } else {
            0.0
        };

        Ok(TextClassificationResults {
            accuracy,
            macro_f1,
            micro_f1,
            weighted_f1,
            per_class_metrics,
        })
    }
}

/// Named Entity Recognition metrics calculator
pub struct NERMetrics;

impl Default for NERMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl NERMetrics {
    /// Create new NER metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Evaluate NER performance
    ///
    /// Entity format: (start_idx, end_idx, entity_type)
    pub fn evaluate_ner(
        &self,
        true_entities: &[(usize, usize, String)],
        pred_entities: &[(usize, usize, String)],
        num_tokens: usize,
    ) -> Result<NERResults> {
        // Entity-level evaluation
        let true_entity_set: HashSet<_> = true_entities.iter().collect();
        let pred_entity_set: HashSet<_> = pred_entities.iter().collect();

        let entity_tp = true_entity_set.intersection(&pred_entity_set).count();
        let entity_fp = pred_entities.len() - entity_tp;
        let entity_fn = true_entities.len() - entity_tp;

        let entity_precision = if entity_tp + entity_fp > 0 {
            entity_tp as f64 / (entity_tp + entity_fp) as f64
        } else {
            0.0
        };

        let entity_recall = if entity_tp + entity_fn > 0 {
            entity_tp as f64 / (entity_tp + entity_fn) as f64
        } else {
            0.0
        };

        let entity_f1 = if entity_precision + entity_recall > 0.0 {
            2.0 * entity_precision * entity_recall / (entity_precision + entity_recall)
        } else {
            0.0
        };

        // Token-level evaluation (BIO tagging simulation)
        let mut true_bio = vec!["O".to_string(); num_tokens];
        let mut pred_bio = vec!["O".to_string(); num_tokens];

        // Convert entities to BIO tags
        for (start, end, entity_type) in true_entities {
            if *start < num_tokens {
                true_bio[*start] = format!("B-{}", entity_type);
                for i in start + 1..=*end {
                    if i < num_tokens {
                        true_bio[i] = format!("I-{}", entity_type);
                    }
                }
            }
        }

        for (start, end, entity_type) in pred_entities {
            if *start < num_tokens {
                pred_bio[*start] = format!("B-{}", entity_type);
                for i in start + 1..=*end {
                    if i < num_tokens {
                        pred_bio[i] = format!("I-{}", entity_type);
                    }
                }
            }
        }

        // Calculate token-level metrics
        let token_correct = true_bio
            .iter()
            .zip(pred_bio.iter())
            .filter(|(t, p)| t == p)
            .count();
        let token_precision = token_correct as f64 / num_tokens as f64;
        let token_recall = token_precision; // Same for token-level
        let token_f1 = token_precision;

        // Per-entity-type metrics
        let mut entity_types = HashSet::new();
        for (_, _, entity_type) in true_entities.iter().chain(pred_entities.iter()) {
            entity_types.insert(entity_type.clone());
        }

        let mut per_type_metrics = HashMap::new();
        for entity_type in entity_types {
            let true_type_entities: Vec<_> = true_entities
                .iter()
                .filter(|(_, _, t)| t == &entity_type)
                .collect();
            let pred_type_entities: Vec<_> = pred_entities
                .iter()
                .filter(|(_, _, t)| t == &entity_type)
                .collect();

            let type_tp = true_type_entities
                .iter()
                .filter(|e| pred_type_entities.contains(e))
                .count();
            let type_fp = pred_type_entities.len() - type_tp;
            let type_fn = true_type_entities.len() - type_tp;

            let type_precision = if type_tp + type_fp > 0 {
                type_tp as f64 / (type_tp + type_fp) as f64
            } else {
                0.0
            };

            let type_recall = if type_tp + type_fn > 0 {
                type_tp as f64 / (type_tp + type_fn) as f64
            } else {
                0.0
            };

            let type_f1 = if type_precision + type_recall > 0.0 {
                2.0 * type_precision * type_recall / (type_precision + type_recall)
            } else {
                0.0
            };

            per_type_metrics.insert(
                entity_type,
                ClassMetrics {
                    precision: type_precision,
                    recall: type_recall,
                    f1_score: type_f1,
                    support: true_type_entities.len(),
                },
            );
        }

        Ok(NERResults {
            entity_precision,
            entity_recall,
            entity_f1,
            token_precision,
            token_recall,
            token_f1,
            per_type_metrics,
        })
    }
}

/// Sentiment analysis metrics calculator
pub struct SentimentMetrics;

impl Default for SentimentMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentMetrics {
    /// Create new sentiment analysis metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Evaluate sentiment analysis performance
    /// Assumes binary sentiment: 0 = negative, 1 = positive
    pub fn evaluate_sentiment(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<SentimentResults> {
        let accuracy = accuracy_score(y_true, y_pred)?;

        // Calculate per-class metrics
        let positive_precision = precision_score(y_true, y_pred, 1)?;
        let positive_recall = recall_score(y_true, y_pred, 1)?;
        let negative_precision = precision_score(y_true, y_pred, 0)?;
        let negative_recall = recall_score(y_true, y_pred, 0)?;

        // Calculate macro F1
        let positive_f1 = if positive_precision + positive_recall > 0.0 {
            2.0 * positive_precision * positive_recall / (positive_precision + positive_recall)
        } else {
            0.0
        };

        let negative_f1 = if negative_precision + negative_recall > 0.0 {
            2.0 * negative_precision * negative_recall / (negative_precision + negative_recall)
        } else {
            0.0
        };

        let macro_f1 = (positive_f1 + negative_f1) / 2.0;

        Ok(SentimentResults {
            accuracy,
            macro_f1,
            positive_precision,
            positive_recall,
            negative_precision,
            negative_recall,
        })
    }
}

/// Complete NLP metrics suite
pub struct NLPSuite {
    text_generation: TextGenerationMetrics,
    classification: TextClassificationMetrics,
    ner: NERMetrics,
    sentiment: SentimentMetrics,
}

impl Default for NLPSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl NLPSuite {
    /// Create a new NLP metrics suite
    pub fn new() -> Self {
        Self {
            text_generation: TextGenerationMetrics::new(),
            classification: TextClassificationMetrics::new(),
            ner: NERMetrics::new(),
            sentiment: SentimentMetrics::new(),
        }
    }

    /// Get text generation metrics calculator
    pub fn text_generation(&self) -> &TextGenerationMetrics {
        &self.text_generation
    }

    /// Get text classification metrics calculator
    pub fn classification(&self) -> &TextClassificationMetrics {
        &self.classification
    }

    /// Get NER metrics calculator
    pub fn ner(&self) -> &NERMetrics {
        &self.ner
    }

    /// Get sentiment analysis metrics calculator
    pub fn sentiment(&self) -> &SentimentMetrics {
        &self.sentiment
    }
}

impl DomainMetrics for NLPSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Natural Language Processing"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "text_generation_bleu_4",
            "text_generation_rouge_l",
            "text_classification_accuracy",
            "text_classification_macro_f1",
            "ner_entity_f1",
            "ner_token_f1",
            "sentiment_accuracy",
            "sentiment_macro_f1",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "text_generation_bleu_4",
            "BLEU-4 score for text generation quality",
        );
        descriptions.insert(
            "text_generation_rouge_l",
            "ROUGE-L score for text generation quality",
        );
        descriptions.insert(
            "text_classification_accuracy",
            "Accuracy for text classification",
        );
        descriptions.insert(
            "text_classification_macro_f1",
            "Macro-averaged F1 for text classification",
        );
        descriptions.insert(
            "ner_entity_f1",
            "Entity-level F1 score for named entity recognition",
        );
        descriptions.insert(
            "ner_token_f1",
            "Token-level F1 score for named entity recognition",
        );
        descriptions.insert("sentiment_accuracy", "Accuracy for sentiment analysis");
        descriptions.insert(
            "sentiment_macro_f1",
            "Macro F1 score for sentiment analysis",
        );
        descriptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bleu_calculation() {
        let metrics = TextGenerationMetrics::new();
        let reference = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];
        let candidate = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];

        let bleu_1 = metrics.calculate_bleu(&reference, &candidate, 1);
        assert_eq!(bleu_1, 1.0); // Perfect match

        let candidate2 = vec!["the".to_string(), "dog".to_string(), "ran".to_string()];
        let bleu_1_partial = metrics.calculate_bleu(&reference, &candidate2, 1);
        assert!(bleu_1_partial > 0.0 && bleu_1_partial < 1.0);
    }

    #[test]
    fn test_rouge_l_calculation() {
        let metrics = TextGenerationMetrics::new();
        let reference = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];
        let candidate = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];

        let rouge_l = metrics.calculate_rouge_l(&reference, &candidate);
        assert_eq!(rouge_l, 1.0); // Perfect match
    }

    #[test]
    fn test_text_generation_evaluation() {
        let metrics = TextGenerationMetrics::new();

        let references = vec![
            "The cat sat on the mat".to_string(),
            "A quick brown fox jumps".to_string(),
        ];

        let candidates = vec![
            "The cat sits on the mat".to_string(),
            "A quick brown fox jumped".to_string(),
        ];

        let results = metrics
            .evaluate_generation(&references, &candidates)
            .unwrap();

        assert!(results.bleu_1 >= 0.0 && results.bleu_1 <= 1.0);
        assert!(results.bleu_4 >= 0.0 && results.bleu_4 <= 1.0);
        assert!(results.rouge_l >= 0.0 && results.rouge_l <= 1.0);
    }

    #[test]
    fn test_text_classification_evaluation() {
        let metrics = TextClassificationMetrics::new();

        let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
        let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2]);

        let results = metrics.evaluate_classification(&y_true, &y_pred).unwrap();

        assert!(results.accuracy >= 0.0 && results.accuracy <= 1.0);
        assert!(results.macro_f1 >= 0.0 && results.macro_f1 <= 1.0);
        assert!(results.micro_f1 >= 0.0 && results.micro_f1 <= 1.0);
        assert!(results.weighted_f1 >= 0.0 && results.weighted_f1 <= 1.0);
        assert_eq!(results.per_class_metrics.len(), 3);
    }

    #[test]
    fn test_ner_evaluation() {
        let metrics = NERMetrics::new();

        let true_entities = vec![(0, 2, "PERSON".to_string()), (5, 7, "ORG".to_string())];

        let pred_entities = vec![
            (0, 2, "PERSON".to_string()), // Correct
            (5, 6, "ORG".to_string()),    // Partial match (different span)
            (10, 12, "LOC".to_string()),  // False positive
        ];

        let results = metrics
            .evaluate_ner(&true_entities, &pred_entities, 15)
            .unwrap();

        assert!(results.entity_precision >= 0.0 && results.entity_precision <= 1.0);
        assert!(results.entity_recall >= 0.0 && results.entity_recall <= 1.0);
        assert!(results.entity_f1 >= 0.0 && results.entity_f1 <= 1.0);
        assert!(!results.per_type_metrics.is_empty());
    }

    #[test]
    fn test_sentiment_evaluation() {
        let metrics = SentimentMetrics::new();

        let y_true = Array1::from_vec(vec![0, 1, 1, 0, 1, 0]); // 0=negative, 1=positive
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0, 1, 1]);

        let results = metrics.evaluate_sentiment(&y_true, &y_pred).unwrap();

        assert!(results.accuracy >= 0.0 && results.accuracy <= 1.0);
        assert!(results.macro_f1 >= 0.0 && results.macro_f1 <= 1.0);
        assert!(results.positive_precision >= 0.0 && results.positive_precision <= 1.0);
        assert!(results.negative_precision >= 0.0 && results.negative_precision <= 1.0);
    }

    #[test]
    fn test_nlp_suite() {
        let suite = NLPSuite::new();

        assert_eq!(suite.domain_name(), "Natural Language Processing");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
