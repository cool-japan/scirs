//! Contrastive learning metrics
//!
//! This module provides evaluation metrics for contrastive learning
//! including uniformity, alignment, and InfoNCE loss computation.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::iter::Sum;

use super::results::InfoNCEResult;

/// Contrastive learning metrics
pub struct ContrastiveLearningMetrics<F: Float> {
    /// Temperature parameter for contrastive loss
    pub temperature: F,
    /// Number of negative samples
    pub n_negatives: usize,
    /// Enable hard negative mining
    pub enable_hard_negatives: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand> Default
    for ContrastiveLearningMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + Sum + ndarray::ScalarOperand>
    ContrastiveLearningMetrics<F>
{
    /// Create new contrastive learning metrics
    pub fn new() -> Self {
        Self {
            temperature: F::from(0.1).unwrap(),
            n_negatives: 1024,
            enable_hard_negatives: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set temperature parameter
    pub fn with_temperature(mut self, temp: F) -> Self {
        self.temperature = temp;
        self
    }

    /// Set number of negatives
    pub fn with_negatives(mut self, n: usize) -> Self {
        self.n_negatives = n;
        self
    }

    /// Enable hard negative mining
    pub fn with_hard_negatives(mut self, enable: bool) -> Self {
        self.enable_hard_negatives = enable;
        self
    }

    /// Compute uniformity of representations
    pub fn uniformity(&self, representations: &Array2<F>, t: F) -> Result<F> {
        if representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        let n_samples = representations.nrows();
        if n_samples < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Normalize representations to unit sphere
        let normalized = self.l2_normalize(representations)?;

        let mut sum_exp = F::zero();
        let mut count = 0;

        // Compute pairwise similarities and uniformity
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let similarity = self.cosine_similarity(
                    &normalized.row(i).to_owned(),
                    &normalized.row(j).to_owned(),
                )?;

                sum_exp = sum_exp + (t * similarity).exp();
                count += 1;
            }
        }

        if count == 0 {
            return Ok(F::zero());
        }

        let uniformity = (sum_exp / F::from(count).unwrap()).ln() / t;
        Ok(uniformity)
    }

    /// Compute alignment between positive pairs
    pub fn alignment(
        &self,
        anchor_representations: &Array2<F>,
        positive_representations: &Array2<F>,
        alpha: F,
    ) -> Result<F> {
        if anchor_representations.nrows() != positive_representations.nrows() {
            return Err(MetricsError::InvalidInput(
                "Mismatched number of pairs".to_string(),
            ));
        }

        if anchor_representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        // Normalize representations
        let anchor_norm = self.l2_normalize(anchor_representations)?;
        let positive_norm = self.l2_normalize(positive_representations)?;

        let mut sum_distance = F::zero();
        let n_pairs = anchor_norm.nrows();

        for i in 0..n_pairs {
            let distance_sq = self.squared_euclidean_distance(
                &anchor_norm.row(i).to_owned(),
                &positive_norm.row(i).to_owned(),
            )?;

            sum_distance = sum_distance + distance_sq.powf(alpha);
        }

        let alignment = sum_distance / F::from(n_pairs).unwrap();
        Ok(alignment)
    }

    /// Compute InfoNCE loss
    pub fn infonce_loss(
        &self,
        anchor_representations: &Array2<F>,
        positive_representations: &Array2<F>,
        negative_representations: &Array2<F>,
    ) -> Result<InfoNCEResult<F>> {
        if anchor_representations.nrows() != positive_representations.nrows() {
            return Err(MetricsError::InvalidInput(
                "Mismatched anchor-positive pairs".to_string(),
            ));
        }

        let n_pairs = anchor_representations.nrows();
        let n_negatives = negative_representations.nrows();

        if n_pairs == 0 || n_negatives == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        // Normalize all representations
        let anchor_norm = self.l2_normalize(anchor_representations)?;
        let positive_norm = self.l2_normalize(positive_representations)?;
        let negative_norm = self.l2_normalize(negative_representations)?;

        let mut total_loss = F::zero();
        let mut correct_predictions = 0;

        for i in 0..n_pairs {
            // Compute positive similarity
            let pos_sim = self.cosine_similarity(
                &anchor_norm.row(i).to_owned(),
                &positive_norm.row(i).to_owned(),
            )?;
            let pos_logit = pos_sim / self.temperature;

            // Compute negative similarities
            let mut neg_logits = Vec::with_capacity(n_negatives);
            for j in 0..n_negatives {
                let neg_sim = self.cosine_similarity(
                    &anchor_norm.row(i).to_owned(),
                    &negative_norm.row(j).to_owned(),
                )?;
                neg_logits.push(neg_sim / self.temperature);
            }

            // Compute softmax denominator
            let mut exp_sum = pos_logit.exp();
            for &neg_logit in &neg_logits {
                exp_sum = exp_sum + neg_logit.exp();
            }

            // InfoNCE loss for this sample
            let sample_loss = -pos_logit + exp_sum.ln();
            total_loss = total_loss + sample_loss;

            // Check if positive is the highest scoring
            let max_neg_logit = neg_logits
                .iter()
                .copied()
                .fold(neg_logits[0], |a, b| a.max(b));
            if pos_logit > max_neg_logit {
                correct_predictions += 1;
            }
        }

        let mean_loss = total_loss / F::from(n_pairs).unwrap();
        let accuracy = F::from(correct_predictions).unwrap() / F::from(n_pairs).unwrap();

        Ok(InfoNCEResult {
            loss: mean_loss,
            accuracy,
            n_pairs,
            temperature: self.temperature,
        })
    }

    /// L2 normalize representations
    fn l2_normalize(&self, representations: &Array2<F>) -> Result<Array2<F>> {
        let mut normalized = representations.clone();

        for mut row in normalized.rows_mut() {
            let norm = (row.mapv(|x| x * x).sum()).sqrt();
            if norm > F::zero() {
                for val in row.iter_mut() {
                    *val = *val / norm;
                }
            }
        }

        Ok(normalized)
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let dot_product: F = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a = (a.mapv(|x| x * x).sum()).sqrt();
        let norm_b = (b.mapv(|x| x * x).sum()).sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Ok(F::zero());
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Compute squared Euclidean distance
    fn squared_euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let distance_sq: F = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        Ok(distance_sq)
    }
}
