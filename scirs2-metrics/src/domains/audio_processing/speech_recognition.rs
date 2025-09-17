//! Speech recognition evaluation metrics
//!
//! This module provides comprehensive metrics for evaluating speech recognition systems,
//! including Word Error Rate (WER), Character Error Rate (CER), Phone Error Rate (PER),
//! BLEU scores for speech translation, and confidence score analysis.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Speech recognition evaluation metrics
#[derive(Debug, Clone)]
pub struct SpeechRecognitionMetrics {
    /// Word Error Rate calculations
    wer_calculator: WerCalculator,
    /// Character Error Rate calculations
    cer_calculator: CerCalculator,
    /// Phone Error Rate calculations
    per_calculator: PerCalculator,
    /// BLEU score for speech translation
    bleu_calculator: BleuCalculator,
    /// Confidence score metrics
    confidence_metrics: ConfidenceMetrics,
}

/// Word Error Rate (WER) calculator
#[derive(Debug, Clone)]
pub struct WerCalculator {
    /// Total word substitutions
    substitutions: usize,
    /// Total word deletions
    deletions: usize,
    /// Total word insertions
    insertions: usize,
    /// Total reference words
    total_words: usize,
    /// Per-utterance WER scores
    utterance_wers: Vec<f64>,
}

/// Character Error Rate (CER) calculator
#[derive(Debug, Clone)]
pub struct CerCalculator {
    /// Total character substitutions
    char_substitutions: usize,
    /// Total character deletions
    char_deletions: usize,
    /// Total character insertions
    char_insertions: usize,
    /// Total reference characters
    total_chars: usize,
    /// Per-utterance CER scores
    utterance_cers: Vec<f64>,
}

/// Phone Error Rate (PER) calculator
#[derive(Debug, Clone)]
pub struct PerCalculator {
    /// Total phone substitutions
    phone_substitutions: usize,
    /// Total phone deletions
    phone_deletions: usize,
    /// Total phone insertions
    phone_insertions: usize,
    /// Total reference phones
    total_phones: usize,
    /// Phone confusion matrix
    confusion_matrix: HashMap<(String, String), usize>,
}

/// BLEU score calculator for speech translation
#[derive(Debug, Clone)]
pub struct BleuCalculator {
    /// N-gram weights (typically 1-gram to 4-gram)
    ngram_weights: Vec<f64>,
    /// Brevity penalty settings
    brevity_penalty: bool,
    /// Smoothing method
    smoothing: BleuSmoothing,
}

/// BLEU smoothing methods
#[derive(Debug, Clone)]
pub enum BleuSmoothing {
    None,
    Epsilon(f64),
    Add1,
    ExponentialDecay,
}

/// Confidence score metrics for ASR
#[derive(Debug, Clone)]
pub struct ConfidenceMetrics {
    /// Confidence threshold for filtering
    confidence_threshold: f64,
    /// Per-word confidence scores
    word_confidences: Vec<f64>,
    /// Utterance-level confidence scores
    utterance_confidences: Vec<f64>,
    /// Confidence-WER correlation
    confidence_wer_correlation: Option<f64>,
}

/// Speech recognition evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRecognitionResults {
    /// Word Error Rate
    pub wer: f64,
    /// Character Error Rate
    pub cer: f64,
    /// Phone Error Rate
    pub per: Option<f64>,
    /// BLEU score
    pub bleu: Option<f64>,
    /// Average confidence score
    pub avg_confidence: f64,
    /// Confidence-WER correlation
    pub confidence_wer_correlation: Option<f64>,
}

impl SpeechRecognitionMetrics {
    /// Create new speech recognition metrics
    pub fn new() -> Self {
        Self {
            wer_calculator: WerCalculator::new(),
            cer_calculator: CerCalculator::new(),
            per_calculator: PerCalculator::new(),
            bleu_calculator: BleuCalculator::new(),
            confidence_metrics: ConfidenceMetrics::new(),
        }
    }

    /// Evaluate speech recognition performance
    pub fn evaluate_recognition(
        &mut self,
        reference_text: &[String],
        hypothesis_text: &[String],
        reference_phones: Option<&[Vec<String>]>,
        hypothesis_phones: Option<&[Vec<String>]>,
        confidence_scores: Option<&[f64]>,
    ) -> Result<SpeechRecognitionResults> {
        // Calculate WER
        let wer = self
            .wer_calculator
            .calculate(reference_text, hypothesis_text)?;

        // Calculate CER
        let cer = self
            .cer_calculator
            .calculate(reference_text, hypothesis_text)?;

        // Calculate PER if phone sequences provided
        let per =
            if let (Some(ref_phones), Some(hyp_phones)) = (reference_phones, hypothesis_phones) {
                Some(self.per_calculator.calculate(ref_phones, hyp_phones)?)
            } else {
                None
            };

        // Calculate BLEU score
        let bleu = Some(
            self.bleu_calculator
                .calculate(reference_text, hypothesis_text)?,
        );

        // Calculate confidence metrics
        let (avg_confidence, confidence_wer_correlation) =
            if let Some(conf_scores) = confidence_scores {
                let avg_conf = conf_scores.iter().sum::<f64>() / conf_scores.len() as f64;
                let correlation = self
                    .confidence_metrics
                    .calculate_confidence_wer_correlation(
                        reference_text,
                        hypothesis_text,
                        conf_scores,
                    )?;
                (avg_conf, Some(correlation))
            } else {
                (0.0, None)
            };

        Ok(SpeechRecognitionResults {
            wer,
            cer,
            per,
            bleu,
            avg_confidence,
            confidence_wer_correlation,
        })
    }

    /// Compute word error rate between reference and hypothesis
    pub fn compute_wer(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        self.wer_calculator.compute_wer(reference, hypothesis)
    }

    /// Compute character error rate between reference and hypothesis
    pub fn compute_cer(&mut self, reference: &str, hypothesis: &str) -> Result<f64> {
        self.cer_calculator.compute_cer(reference, hypothesis)
    }

    /// Compute phone error rate between reference and hypothesis phoneme sequences
    pub fn compute_per(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        self.per_calculator.compute_per(reference, hypothesis)
    }

    /// Compute BLEU score for translation tasks
    pub fn compute_bleu(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        self.bleu_calculator.compute_bleu(reference, hypothesis)
    }

    /// Add confidence scores for analysis
    pub fn add_confidence_scores(&mut self, word_confidences: Vec<f64>, utterance_confidence: f64) {
        self.confidence_metrics
            .add_scores(word_confidences, utterance_confidence);
    }

    /// Get comprehensive speech recognition results
    pub fn get_results(&self) -> SpeechRecognitionResults {
        SpeechRecognitionResults {
            wer: self.wer_calculator.get_wer(),
            cer: self.cer_calculator.get_cer(),
            per: self.per_calculator.get_per(),
            bleu: self.bleu_calculator.get_bleu(),
            avg_confidence: self.confidence_metrics.get_average_confidence(),
            confidence_wer_correlation: self.confidence_metrics.confidence_wer_correlation,
        }
    }
}

impl WerCalculator {
    /// Create new WER calculator
    pub fn new() -> Self {
        Self {
            substitutions: 0,
            deletions: 0,
            insertions: 0,
            total_words: 0,
            utterance_wers: Vec::new(),
        }
    }

    /// Compute WER using edit distance algorithm
    pub fn compute_wer(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        let (subs, dels, ins) = self.edit_distance(reference, hypothesis);

        self.substitutions += subs;
        self.deletions += dels;
        self.insertions += ins;
        self.total_words += reference.len();

        let utterance_wer = if reference.is_empty() {
            if hypothesis.is_empty() {
                0.0
            } else {
                1.0
            }
        } else {
            (subs + dels + ins) as f64 / reference.len() as f64
        };

        self.utterance_wers.push(utterance_wer);
        Ok(utterance_wer)
    }

    /// Get overall WER
    pub fn get_wer(&self) -> f64 {
        if self.total_words == 0 {
            0.0
        } else {
            (self.substitutions + self.deletions + self.insertions) as f64 / self.total_words as f64
        }
    }

    /// Compute edit distance between reference and hypothesis
    fn edit_distance(&self, reference: &[String], hypothesis: &[String]) -> (usize, usize, usize) {
        let ref_len = reference.len();
        let hyp_len = hypothesis.len();

        let mut dp = vec![vec![0; hyp_len + 1]; ref_len + 1];
        let mut ops = vec![vec![(0, 0, 0); hyp_len + 1]; ref_len + 1]; // (subs, dels, ins)

        // Initialize base cases
        for i in 0..=ref_len {
            dp[i][0] = i;
            ops[i][0] = (0, i, 0);
        }
        for j in 0..=hyp_len {
            dp[0][j] = j;
            ops[0][j] = (0, 0, j);
        }

        // Fill DP table
        for i in 1..=ref_len {
            for j in 1..=hyp_len {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                    ops[i][j] = ops[i - 1][j - 1];
                } else {
                    let sub_cost = dp[i - 1][j - 1] + 1;
                    let del_cost = dp[i - 1][j] + 1;
                    let ins_cost = dp[i][j - 1] + 1;

                    if sub_cost <= del_cost && sub_cost <= ins_cost {
                        dp[i][j] = sub_cost;
                        ops[i][j] = (
                            ops[i - 1][j - 1].0 + 1,
                            ops[i - 1][j - 1].1,
                            ops[i - 1][j - 1].2,
                        );
                    } else if del_cost <= ins_cost {
                        dp[i][j] = del_cost;
                        ops[i][j] = (ops[i - 1][j].0, ops[i - 1][j].1 + 1, ops[i - 1][j].2);
                    } else {
                        dp[i][j] = ins_cost;
                        ops[i][j] = (ops[i][j - 1].0, ops[i][j - 1].1, ops[i][j - 1].2 + 1);
                    }
                }
            }
        }

        ops[ref_len][hyp_len]
    }

    /// Calculate WER (alias for compute_wer for backward compatibility)
    pub fn calculate(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        self.compute_wer(reference, hypothesis)
    }
}

impl CerCalculator {
    /// Create new CER calculator
    pub fn new() -> Self {
        Self {
            char_substitutions: 0,
            char_deletions: 0,
            char_insertions: 0,
            total_chars: 0,
            utterance_cers: Vec::new(),
        }
    }

    /// Compute CER using character-level edit distance
    pub fn compute_cer(&mut self, reference: &str, hypothesis: &str) -> Result<f64> {
        let ref_chars: Vec<char> = reference.chars().collect();
        let hyp_chars: Vec<char> = hypothesis.chars().collect();

        let (subs, dels, ins) = self.char_edit_distance(&ref_chars, &hyp_chars);

        self.char_substitutions += subs;
        self.char_deletions += dels;
        self.char_insertions += ins;
        self.total_chars += ref_chars.len();

        let utterance_cer = if ref_chars.is_empty() {
            if hyp_chars.is_empty() {
                0.0
            } else {
                1.0
            }
        } else {
            (subs + dels + ins) as f64 / ref_chars.len() as f64
        };

        self.utterance_cers.push(utterance_cer);
        Ok(utterance_cer)
    }

    /// Get overall CER
    pub fn get_cer(&self) -> f64 {
        if self.total_chars == 0 {
            0.0
        } else {
            (self.char_substitutions + self.char_deletions + self.char_insertions) as f64
                / self.total_chars as f64
        }
    }

    /// Compute character-level edit distance
    fn char_edit_distance(&self, reference: &[char], hypothesis: &[char]) -> (usize, usize, usize) {
        let ref_len = reference.len();
        let hyp_len = hypothesis.len();

        let mut dp = vec![vec![0; hyp_len + 1]; ref_len + 1];

        for i in 0..=ref_len {
            dp[i][0] = i;
        }
        for j in 0..=hyp_len {
            dp[0][j] = j;
        }

        for i in 1..=ref_len {
            for j in 1..=hyp_len {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1]);
                }
            }
        }

        // Simplified: return total edit distance as substitutions for now
        (dp[ref_len][hyp_len], 0, 0)
    }

    /// Calculate CER (alias for compute_cer for backward compatibility)
    pub fn calculate(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        if reference.len() != hypothesis.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and hypothesis must have the same length".to_string(),
            ));
        }

        let mut total_errors = 0;
        let mut total_chars = 0;

        for (ref_sent, hyp_sent) in reference.iter().zip(hypothesis.iter()) {
            let cer = self.compute_cer(ref_sent, hyp_sent)?;
            let ref_chars = ref_sent.chars().count();
            total_errors += (cer * ref_chars as f64) as usize;
            total_chars += ref_chars;
        }

        if total_chars == 0 {
            Ok(0.0)
        } else {
            Ok(total_errors as f64 / total_chars as f64)
        }
    }
}

impl PerCalculator {
    /// Create new PER calculator
    pub fn new() -> Self {
        Self {
            phone_substitutions: 0,
            phone_deletions: 0,
            phone_insertions: 0,
            total_phones: 0,
            confusion_matrix: HashMap::new(),
        }
    }

    /// Compute PER using phoneme-level edit distance
    pub fn compute_per(&mut self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        let (subs, dels, ins) = self.phone_edit_distance(reference, hypothesis);

        self.phone_substitutions += subs;
        self.phone_deletions += dels;
        self.phone_insertions += ins;
        self.total_phones += reference.len();

        let per = if reference.is_empty() {
            if hypothesis.is_empty() {
                0.0
            } else {
                1.0
            }
        } else {
            (subs + dels + ins) as f64 / reference.len() as f64
        };

        Ok(per)
    }

    /// Get overall PER
    pub fn get_per(&self) -> Option<f64> {
        if self.total_phones == 0 {
            None
        } else {
            Some(
                (self.phone_substitutions + self.phone_deletions + self.phone_insertions) as f64
                    / self.total_phones as f64,
            )
        }
    }

    /// Compute phoneme-level edit distance
    fn phone_edit_distance(
        &mut self,
        reference: &[String],
        hypothesis: &[String],
    ) -> (usize, usize, usize) {
        // Track phone confusions
        for (i, ref_phone) in reference.iter().enumerate() {
            if i < hypothesis.len() && ref_phone != &hypothesis[i] {
                *self
                    .confusion_matrix
                    .entry((ref_phone.clone(), hypothesis[i].clone()))
                    .or_insert(0) += 1;
            }
        }

        // Simplified edit distance calculation
        let mut subs = 0;
        let mut dels = 0;
        let mut ins = 0;

        let max_len = reference.len().max(hypothesis.len());
        for i in 0..max_len {
            match (reference.get(i), hypothesis.get(i)) {
                (Some(r), Some(h)) if r != h => subs += 1,
                (Some(_), None) => dels += 1,
                (None, Some(_)) => ins += 1,
                _ => {}
            }
        }

        (subs, dels, ins)
    }

    /// Calculate PER (alias for compute_per for backward compatibility)
    pub fn calculate(
        &mut self,
        reference: &[Vec<String>],
        hypothesis: &[Vec<String>],
    ) -> Result<f64> {
        if reference.len() != hypothesis.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and hypothesis must have the same length".to_string(),
            ));
        }

        let mut total_errors = 0;
        let mut total_phones = 0;

        for (ref_seq, hyp_seq) in reference.iter().zip(hypothesis.iter()) {
            let per = self.compute_per(ref_seq, hyp_seq)?;
            total_errors += (per * ref_seq.len() as f64) as usize;
            total_phones += ref_seq.len();
        }

        if total_phones == 0 {
            Ok(0.0)
        } else {
            Ok(total_errors as f64 / total_phones as f64)
        }
    }
}

impl BleuCalculator {
    /// Create new BLEU calculator
    pub fn new() -> Self {
        Self {
            ngram_weights: vec![0.25, 0.25, 0.25, 0.25], // Equal weights for 1-4 grams
            brevity_penalty: true,
            smoothing: BleuSmoothing::Epsilon(1e-7),
        }
    }

    /// Compute BLEU score
    pub fn compute_bleu(&self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        if reference.is_empty() || hypothesis.is_empty() {
            return Ok(0.0);
        }

        let mut precisions = Vec::new();

        // Compute n-gram precisions
        for n in 1..=4 {
            let precision = self.compute_ngram_precision(reference, hypothesis, n);
            precisions.push(precision);
        }

        // Compute geometric mean of precisions
        let log_sum: f64 = precisions
            .iter()
            .zip(&self.ngram_weights)
            .map(|(p, w)| w * p.ln())
            .sum();

        let mut bleu = log_sum.exp();

        // Apply brevity penalty
        if self.brevity_penalty {
            let bp = self.compute_brevity_penalty(reference.len(), hypothesis.len());
            bleu *= bp;
        }

        Ok(bleu)
    }

    /// Get BLEU score (placeholder)
    pub fn get_bleu(&self) -> Option<f64> {
        None // Would store computed BLEU scores
    }

    /// Compute n-gram precision
    fn compute_ngram_precision(
        &self,
        reference: &[String],
        hypothesis: &[String],
        n: usize,
    ) -> f64 {
        if hypothesis.len() < n {
            return 0.0;
        }

        let ref_ngrams = self.extract_ngrams(reference, n);
        let hyp_ngrams = self.extract_ngrams(hypothesis, n);

        let mut matches = 0;
        for ngram in &hyp_ngrams {
            if ref_ngrams.contains(ngram) {
                matches += 1;
            }
        }

        if hyp_ngrams.is_empty() {
            0.0
        } else {
            matches as f64 / hyp_ngrams.len() as f64
        }
    }

    /// Extract n-grams from sequence
    fn extract_ngrams(&self, sequence: &[String], n: usize) -> Vec<Vec<String>> {
        if sequence.len() < n {
            return Vec::new();
        }

        (0..=sequence.len() - n)
            .map(|i| sequence[i..i + n].to_vec())
            .collect()
    }

    /// Compute brevity penalty
    fn compute_brevity_penalty(&self, ref_len: usize, hyp_len: usize) -> f64 {
        if hyp_len >= ref_len {
            1.0
        } else {
            (1.0 - ref_len as f64 / hyp_len as f64).exp()
        }
    }

    /// Calculate BLEU score (alias for compute_bleu for backward compatibility)
    pub fn calculate(&self, reference: &[String], hypothesis: &[String]) -> Result<f64> {
        self.compute_bleu(reference, hypothesis)
    }
}

impl ConfidenceMetrics {
    /// Create new confidence metrics
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
            word_confidences: Vec::new(),
            utterance_confidences: Vec::new(),
            confidence_wer_correlation: None,
        }
    }

    /// Add confidence scores
    pub fn add_scores(&mut self, word_confidences: Vec<f64>, utterance_confidence: f64) {
        self.word_confidences.extend(word_confidences);
        self.utterance_confidences.push(utterance_confidence);
    }

    /// Get average confidence score
    pub fn get_average_confidence(&self) -> f64 {
        if self.utterance_confidences.is_empty() {
            0.0
        } else {
            self.utterance_confidences.iter().sum::<f64>() / self.utterance_confidences.len() as f64
        }
    }

    /// Set confidence threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.confidence_threshold = threshold;
    }

    /// Calculate confidence-WER correlation
    pub fn calculate_confidence_wer_correlation(
        &mut self,
        reference: &[String],
        hypothesis: &[String],
        confidence: &[f64],
    ) -> Result<f64> {
        if reference.len() != hypothesis.len() || hypothesis.len() != confidence.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched array lengths".to_string(),
            ));
        }

        let mut correct_scores = Vec::new();
        let mut incorrect_scores = Vec::new();

        for ((r, h), &c) in reference
            .iter()
            .zip(hypothesis.iter())
            .zip(confidence.iter())
        {
            if r == h {
                correct_scores.push(c);
            } else {
                incorrect_scores.push(c);
            }
        }

        if correct_scores.is_empty() || incorrect_scores.is_empty() {
            return Ok(0.0);
        }

        let correct_mean = correct_scores.iter().sum::<f64>() / correct_scores.len() as f64;
        let incorrect_mean = incorrect_scores.iter().sum::<f64>() / incorrect_scores.len() as f64;

        Ok((correct_mean - incorrect_mean).abs())
    }
}

impl Default for SpeechRecognitionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WerCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CerCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BleuCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConfidenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}
