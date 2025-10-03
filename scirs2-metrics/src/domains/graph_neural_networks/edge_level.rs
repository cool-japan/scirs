//! Edge-level task evaluation metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating edge-level tasks such as
//! link prediction, edge classification, edge regression, and temporal edge prediction.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{ClassMetrics, EdgeId, RankingMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Edge-level task evaluation metrics
#[derive(Debug, Clone)]
pub struct EdgeLevelMetrics {
    /// Link prediction metrics
    pub link_prediction: LinkPredictionMetrics,
    /// Edge classification metrics
    pub edge_classification: EdgeClassificationMetrics,
    /// Edge weight prediction metrics
    pub edge_regression: EdgeRegressionMetrics,
    /// Temporal edge prediction metrics
    pub temporal_metrics: TemporalEdgeMetrics,
}

/// Link prediction evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPredictionMetrics {
    /// Area Under ROC Curve
    pub auc_roc: f64,
    /// Area Under Precision-Recall Curve
    pub auc_pr: f64,
    /// Average Precision
    pub average_precision: f64,
    /// Hits@K metrics
    pub hits_at_k: HashMap<usize, f64>, // k -> hits@k
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Precision@K
    pub precision_at_k: HashMap<usize, f64>,
    /// Recall@K
    pub recall_at_k: HashMap<usize, f64>,
}

impl Default for LinkPredictionMetrics {
    fn default() -> Self {
        Self {
            auc_roc: 0.0,
            auc_pr: 0.0,
            average_precision: 0.0,
            hits_at_k: HashMap::new(),
            mrr: 0.0,
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
        }
    }
}

impl LinkPredictionMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute link prediction metrics from edge scores and ground truth
    pub fn compute_link_prediction_metrics<F: Float>(
        &mut self,
        edge_scores: &[(EdgeId, F)], // (edge, score) pairs
        positive_edges: &[EdgeId],
        negative_edges: &[EdgeId],
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        if edge_scores.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Edge scores cannot be empty".to_string(),
            ));
        }

        // Create ground truth labels
        let mut labeled_scores: Vec<(f64, bool)> = Vec::new();

        for (edge, score) in edge_scores {
            let score_f64 = score.to_f64().unwrap_or(0.0);
            if positive_edges.contains(edge) {
                labeled_scores.push((score_f64, true));
            } else if negative_edges.contains(edge) {
                labeled_scores.push((score_f64, false));
            }
        }

        if labeled_scores.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No labeled edges found".to_string(),
            ));
        }

        // Sort by score descending
        labeled_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute AUC-ROC
        self.auc_roc = self.compute_auc_roc(&labeled_scores)?;

        // Compute AUC-PR
        self.auc_pr = self.compute_auc_pr(&labeled_scores)?;

        // Compute ranking metrics
        self.compute_ranking_metrics(&labeled_scores)?;

        Ok(())
    }

    fn compute_auc_roc(&self, labeled_scores: &[(f64, bool)]) -> Result<f64> {
        let mut tp = 0;
        let mut fp = 0;
        let mut auc = 0.0;
        let mut prev_fpr = 0.0;

        let total_positives = labeled_scores.iter().filter(|(_, label)| *label).count() as f64;
        let total_negatives = labeled_scores.iter().filter(|(_, label)| !*label).count() as f64;

        if total_positives == 0.0 || total_negatives == 0.0 {
            return Ok(0.5); // Random performance
        }

        for (_, is_positive) in labeled_scores {
            if *is_positive {
                tp += 1;
            } else {
                fp += 1;
                let tpr = tp as f64 / total_positives;
                let fpr = fp as f64 / total_negatives;
                auc += tpr * (fpr - prev_fpr);
                prev_fpr = fpr;
            }
        }

        Ok(auc)
    }

    fn compute_auc_pr(&self, labeled_scores: &[(f64, bool)]) -> Result<f64> {
        let mut tp = 0;
        let mut fp = 0;
        let mut auc = 0.0;
        let mut prev_recall = 0.0;

        let total_positives = labeled_scores.iter().filter(|(_, label)| *label).count() as f64;

        if total_positives == 0.0 {
            return Ok(0.0);
        }

        for (_, is_positive) in labeled_scores {
            if *is_positive {
                tp += 1;
            } else {
                fp += 1;
            }

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = tp as f64 / total_positives;

            auc += precision * (recall - prev_recall);
            prev_recall = recall;
        }

        Ok(auc)
    }

    fn compute_ranking_metrics(&mut self, labeled_scores: &[(f64, bool)]) -> Result<()> {
        let positive_indices: Vec<usize> = labeled_scores.iter()
            .enumerate()
            .filter(|(_, (_, label))| *label)
            .map(|(i, _)| i + 1) // 1-based ranking
            .collect();

        if positive_indices.is_empty() {
            return Ok(());
        }

        // Compute MRR
        self.mrr = positive_indices
            .iter()
            .map(|&rank| 1.0 / rank as f64)
            .sum::<f64>()
            / positive_indices.len() as f64;

        // Compute Hits@K and Precision@K for various K values
        for k in [1, 3, 5, 10, 20, 50, 100] {
            let hits = positive_indices.iter().filter(|&&rank| rank <= k).count() as f64;

            self.hits_at_k
                .insert(k, hits / positive_indices.len() as f64);

            let precision = if k > 0 { hits / k as f64 } else { 0.0 };
            self.precision_at_k.insert(k, precision);

            let recall = hits / positive_indices.len() as f64;
            self.recall_at_k.insert(k, recall);
        }

        Ok(())
    }
}

/// Edge classification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeClassificationMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro-averaged F1
    pub macro_f1: f64,
    /// Micro-averaged F1
    pub micro_f1: f64,
    /// Per-edge-type metrics
    pub per_type_metrics: HashMap<String, ClassMetrics>,
}

impl Default for EdgeClassificationMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            macro_f1: 0.0,
            micro_f1: 0.0,
            per_type_metrics: HashMap::new(),
        }
    }
}

impl EdgeClassificationMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute edge classification metrics
    pub fn compute_classification_metrics<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
        edge_types: &[String],
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n = predictions.len();
        if n != ground_truth.len() {
            return Err(MetricsError::DimensionMismatch(
                "Predictions and ground truth must have same length".to_string(),
            ));
        }

        // Compute overall accuracy
        let correct = (0..n)
            .filter(|&i| (predictions[i] - ground_truth[i]).abs() < F::from(0.5).unwrap())
            .count();
        self.accuracy = correct as f64 / n as f64;

        // Compute per-type metrics
        let mut type_metrics = HashMap::new();

        for (type_idx, edge_type) in edge_types.iter().enumerate() {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;

            for i in 0..n {
                let pred_class = predictions[i].to_usize().unwrap_or(0);
                let true_class = ground_truth[i].to_usize().unwrap_or(0);

                match (pred_class == type_idx, true_class == type_idx) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_count += 1,
                    (false, false) => {} // tn
                }
            }

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            type_metrics.insert(
                edge_type.clone(),
                ClassMetrics {
                    precision,
                    recall,
                    f1_score: f1,
                    support: tp + fn_count,
                },
            );
        }

        // Compute macro and micro F1
        self.macro_f1 =
            type_metrics.values().map(|m| m.f1_score).sum::<f64>() / type_metrics.len() as f64;

        let total_tp: usize = type_metrics
            .values()
            .map(|m| (m.recall * m.support as f64) as usize)
            .sum();
        let total_support: usize = type_metrics.values().map(|m| m.support).sum();
        self.micro_f1 = if total_support > 0 {
            total_tp as f64 / total_support as f64
        } else {
            0.0
        };

        self.per_type_metrics = type_metrics;
        Ok(())
    }
}

/// Edge weight/attribute regression metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRegressionMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared score
    pub r2_score: f64,
    /// Spearman correlation
    pub spearman_correlation: f64,
    /// Pearson correlation
    pub pearson_correlation: f64,
}

impl Default for EdgeRegressionMetrics {
    fn default() -> Self {
        Self {
            mse: 0.0,
            mae: 0.0,
            r2_score: 0.0,
            spearman_correlation: 0.0,
            pearson_correlation: 0.0,
        }
    }
}

impl EdgeRegressionMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute edge regression metrics
    pub fn compute_regression_metrics<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n = predictions.len();
        if n != ground_truth.len() {
            return Err(MetricsError::DimensionMismatch(
                "Predictions and ground truth must have same length".to_string(),
            ));
        }

        if n == 0 {
            return Ok(());
        }

        // Convert to f64 for calculations
        let pred_vec: Vec<f64> = (0..n)
            .map(|i| predictions[i].to_f64().unwrap_or(0.0))
            .collect();
        let true_vec: Vec<f64> = (0..n)
            .map(|i| ground_truth[i].to_f64().unwrap_or(0.0))
            .collect();

        // MSE
        self.mse = pred_vec
            .iter()
            .zip(true_vec.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64;

        // MAE
        self.mae = pred_vec
            .iter()
            .zip(true_vec.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n as f64;

        // R²
        let true_mean = true_vec.iter().sum::<f64>() / n as f64;
        let ss_tot = true_vec
            .iter()
            .map(|t| (t - true_mean).powi(2))
            .sum::<f64>();
        let ss_res = pred_vec
            .iter()
            .zip(true_vec.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum::<f64>();

        self.r2_score = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        // Pearson correlation
        let pred_mean = pred_vec.iter().sum::<f64>() / n as f64;
        let numerator = pred_vec
            .iter()
            .zip(true_vec.iter())
            .map(|(p, t)| (p - pred_mean) * (t - true_mean))
            .sum::<f64>();

        let pred_var = pred_vec
            .iter()
            .map(|p| (p - pred_mean).powi(2))
            .sum::<f64>();
        let true_var = true_vec
            .iter()
            .map(|t| (t - true_mean).powi(2))
            .sum::<f64>();

        self.pearson_correlation = if pred_var > 0.0 && true_var > 0.0 {
            numerator / (pred_var.sqrt() * true_var.sqrt())
        } else {
            0.0
        };

        // Spearman correlation (rank-based)
        self.spearman_correlation = self.compute_spearman_correlation(&pred_vec, &true_vec);

        Ok(())
    }

    fn compute_spearman_correlation(&self, pred: &[f64], true_vals: &[f64]) -> f64 {
        let n = pred.len();
        if n < 2 {
            return 0.0;
        }

        // Create rankings
        let pred_ranks = self.compute_ranks(pred);
        let true_ranks = self.compute_ranks(true_vals);

        // Compute Pearson correlation of ranks
        let pred_rank_mean = pred_ranks.iter().sum::<f64>() / n as f64;
        let true_rank_mean = true_ranks.iter().sum::<f64>() / n as f64;

        let numerator = pred_ranks
            .iter()
            .zip(true_ranks.iter())
            .map(|(p, t)| (p - pred_rank_mean) * (t - true_rank_mean))
            .sum::<f64>();

        let pred_var = pred_ranks
            .iter()
            .map(|p| (p - pred_rank_mean).powi(2))
            .sum::<f64>();
        let true_var = true_ranks
            .iter()
            .map(|t| (t - true_rank_mean).powi(2))
            .sum::<f64>();

        if pred_var > 0.0 && true_var > 0.0 {
            numerator / (pred_var.sqrt() * true_var.sqrt())
        } else {
            0.0
        }
    }

    fn compute_ranks(&self, values: &[f64]) -> Vec<f64> {
        let mut indexed_values: Vec<(usize, f64)> =
            values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; values.len()];
        for (rank, (original_idx, _)) in indexed_values.iter().enumerate() {
            ranks[*original_idx] = (rank + 1) as f64;
        }

        ranks
    }
}

/// Temporal edge prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdgeMetrics {
    /// Time-aware AUC
    pub temporal_auc: f64,
    /// Temporal precision@K
    pub temporal_precision_at_k: HashMap<usize, f64>,
    /// Link persistence accuracy
    pub persistence_accuracy: f64,
    /// New link prediction accuracy
    pub new_link_accuracy: f64,
}

impl Default for TemporalEdgeMetrics {
    fn default() -> Self {
        Self {
            temporal_auc: 0.0,
            temporal_precision_at_k: HashMap::new(),
            persistence_accuracy: 0.0,
            new_link_accuracy: 0.0,
        }
    }
}

impl TemporalEdgeMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute temporal edge prediction metrics
    pub fn compute_temporal_metrics<F: Float>(
        &mut self,
        edge_scores: &[(EdgeId, F, f64)], // (edge, score, timestamp)
        persistent_edges: &[EdgeId],
        new_edges: &[EdgeId],
        disappeared_edges: &[EdgeId],
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        if edge_scores.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Edge scores cannot be empty".to_string(),
            ));
        }

        // Sort by timestamp
        let mut sorted_scores = edge_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Compute time-weighted AUC
        self.temporal_auc =
            self.compute_temporal_auc(&sorted_scores, persistent_edges, new_edges)?;

        // Compute persistence accuracy
        self.persistence_accuracy =
            self.compute_persistence_accuracy(&sorted_scores, persistent_edges, disappeared_edges)?;

        // Compute new link prediction accuracy
        self.new_link_accuracy = self.compute_new_link_accuracy(&sorted_scores, new_edges)?;

        // Compute temporal precision@K
        self.compute_temporal_precision_at_k(&sorted_scores, new_edges)?;

        Ok(())
    }

    fn compute_temporal_auc<F: Float>(
        &self,
        sorted_scores: &[(EdgeId, F, f64)],
        persistent_edges: &[EdgeId],
        new_edges: &[EdgeId],
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Create time-weighted labels
        let mut labeled_scores: Vec<(f64, bool, f64)> = Vec::new(); // (score, is_positive, timestamp)

        for (edge, score, timestamp) in sorted_scores {
            let score_f64 = score.to_f64().unwrap_or(0.0);
            if persistent_edges.contains(edge) || new_edges.contains(edge) {
                labeled_scores.push((score_f64, true, *timestamp));
            } else {
                labeled_scores.push((score_f64, false, *timestamp));
            }
        }

        if labeled_scores.is_empty() {
            return Ok(0.5);
        }

        // Sort by score descending
        labeled_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute time-weighted AUC (giving more weight to recent predictions)
        let max_time = labeled_scores
            .iter()
            .map(|(_, _, t)| *t)
            .fold(0.0, f64::max);
        let min_time = labeled_scores
            .iter()
            .map(|(_, _, t)| *t)
            .fold(f64::INFINITY, f64::min);
        let time_range = max_time - min_time;

        if time_range <= 0.0 {
            return Ok(0.5);
        }

        let mut weighted_tp = 0.0;
        let mut weighted_fp = 0.0;
        let mut auc = 0.0;
        let mut prev_weighted_fpr = 0.0;

        let total_weighted_positives: f64 = labeled_scores
            .iter()
            .filter(|(_, label, _)| *label)
            .map(|(_, _, t)| 1.0 + (t - min_time) / time_range)
            .sum();

        let total_weighted_negatives: f64 = labeled_scores
            .iter()
            .filter(|(_, label, _)| !*label)
            .map(|(_, _, t)| 1.0 + (t - min_time) / time_range)
            .sum();

        if total_weighted_positives <= 0.0 || total_weighted_negatives <= 0.0 {
            return Ok(0.5);
        }

        for (_, is_positive, timestamp) in &labeled_scores {
            let time_weight = 1.0 + (timestamp - min_time) / time_range;

            if *is_positive {
                weighted_tp += time_weight;
            } else {
                weighted_fp += time_weight;
                let weighted_tpr = weighted_tp / total_weighted_positives;
                let weighted_fpr = weighted_fp / total_weighted_negatives;
                auc += weighted_tpr * (weighted_fpr - prev_weighted_fpr);
                prev_weighted_fpr = weighted_fpr;
            }
        }

        Ok(auc)
    }

    fn compute_persistence_accuracy<F: Float>(
        &self,
        sorted_scores: &[(EdgeId, F, f64)],
        persistent_edges: &[EdgeId],
        disappeared_edges: &[EdgeId],
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let mut correct = 0;
        let mut total = 0;

        for (edge, score, _) in sorted_scores {
            let score_f64 = score.to_f64().unwrap_or(0.0);
            let predicted_persistent = score_f64 > 0.5;

            if persistent_edges.contains(edge) {
                total += 1;
                if predicted_persistent {
                    correct += 1;
                }
            } else if disappeared_edges.contains(edge) {
                total += 1;
                if !predicted_persistent {
                    correct += 1;
                }
            }
        }

        Ok(if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        })
    }

    fn compute_new_link_accuracy<F: Float>(
        &self,
        sorted_scores: &[(EdgeId, F, f64)],
        new_edges: &[EdgeId],
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let mut correct = 0;
        let mut total = 0;

        for (edge, score, _) in sorted_scores {
            let score_f64 = score.to_f64().unwrap_or(0.0);
            let predicted_new = score_f64 > 0.5;

            if new_edges.contains(edge) {
                total += 1;
                if predicted_new {
                    correct += 1;
                }
            }
        }

        Ok(if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        })
    }

    fn compute_temporal_precision_at_k<F: Float>(
        &mut self,
        sorted_scores: &[(EdgeId, F, f64)],
        new_edges: &[EdgeId],
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Sort by score descending
        let mut score_sorted: Vec<_> = sorted_scores.iter().collect();
        score_sorted.sort_by(|a, b| {
            b.1.to_f64()
                .unwrap_or(0.0)
                .partial_cmp(&a.1.to_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for k in [1, 3, 5, 10, 20, 50] {
            let top_k_edges: Vec<EdgeId> = score_sorted
                .iter()
                .take(k)
                .map(|(edge, _, _)| *edge)
                .collect();
            let correct = top_k_edges
                .iter()
                .filter(|edge| new_edges.contains(edge))
                .count();
            let precision = if k > 0 {
                correct as f64 / k.min(top_k_edges.len()) as f64
            } else {
                0.0
            };
            self.temporal_precision_at_k.insert(k, precision);
        }

        Ok(())
    }
}

impl EdgeLevelMetrics {
    /// Create new edge-level metrics
    pub fn new() -> Self {
        Self {
            link_prediction: LinkPredictionMetrics::new(),
            edge_classification: EdgeClassificationMetrics::new(),
            edge_regression: EdgeRegressionMetrics::new(),
            temporal_metrics: TemporalEdgeMetrics::new(),
        }
    }
}

impl Default for EdgeLevelMetrics {
    fn default() -> Self {
        Self::new()
    }
}
