//! Node-level task evaluation metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating node-level tasks such as
//! node classification, regression, embedding quality, and fairness.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{CalibrationMetrics, ClassMetrics, GroupFairnessMetrics, NodeId};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node-level task evaluation metrics
#[derive(Debug, Clone)]
pub struct NodeLevelMetrics {
    /// Standard classification/regression metrics
    pub classification_metrics: NodeClassificationMetrics,
    /// Node embedding quality metrics
    pub embedding_metrics: NodeEmbeddingMetrics,
    /// Homophily and heterophily aware metrics
    pub homophily_metrics: HomophilyAwareMetrics,
    /// Fairness metrics for node predictions
    pub fairness_metrics: NodeFairnessMetrics,
}

/// Node classification specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeClassificationMetrics {
    /// Accuracy considering graph structure
    pub structure_aware_accuracy: f64,
    /// Macro F1 score
    pub macro_f1: f64,
    /// Micro F1 score
    pub micro_f1: f64,
    /// Per-class metrics
    pub per_class_metrics: HashMap<String, ClassMetrics>,
    /// Confidence calibration metrics
    pub calibration_metrics: CalibrationMetrics,
}

impl Default for NodeClassificationMetrics {
    fn default() -> Self {
        Self {
            structure_aware_accuracy: 0.0,
            macro_f1: 0.0,
            micro_f1: 0.0,
            per_class_metrics: HashMap::new(),
            calibration_metrics: CalibrationMetrics::default(),
        }
    }
}

impl NodeClassificationMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute structure-aware accuracy considering graph connectivity
    pub fn compute_structure_aware_accuracy<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n = predictions.len();
        if n != ground_truth.len() || adjacency_matrix.nrows() != n || adjacency_matrix.ncols() != n
        {
            return Err(MetricsError::DimensionMismatch(
                "Predictions, ground truth, and adjacency matrix dimensions must match".to_string(),
            ));
        }

        let mut correct_predictions = 0.0;
        let mut total_predictions = 0.0;

        for i in 0..n {
            // Weight prediction correctness by node degree
            let degree = adjacency_matrix.row(i).sum().to_f64().unwrap_or(1.0);
            let weight = (degree + 1.0).ln(); // Log-weighted by degree

            if (predictions[i] - ground_truth[i]).abs() < F::from(0.5).unwrap() {
                correct_predictions += weight;
            }
            total_predictions += weight;
        }

        let accuracy = if total_predictions > 0.0 {
            correct_predictions / total_predictions
        } else {
            0.0
        };

        self.structure_aware_accuracy = accuracy;
        Ok(accuracy)
    }

    /// Compute macro and micro F1 scores
    pub fn compute_f1_scores<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
        class_labels: &[String],
    ) -> Result<(f64, f64)>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n = predictions.len();
        if n != ground_truth.len() {
            return Err(MetricsError::DimensionMismatch(
                "Predictions and ground truth dimensions must match".to_string(),
            ));
        }

        let mut class_metrics = HashMap::new();

        for class_label in class_labels {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;
            let mut tn = 0;

            for i in 0..n {
                let pred_class = predictions[i].to_usize().unwrap_or(0);
                let true_class = ground_truth[i].to_usize().unwrap_or(0);
                let current_class_idx = class_labels
                    .iter()
                    .position(|x| x == class_label)
                    .unwrap_or(0);

                match (
                    pred_class == current_class_idx,
                    true_class == current_class_idx,
                ) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_count += 1,
                    (false, false) => tn += 1,
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

            class_metrics.insert(
                class_label.clone(),
                ClassMetrics {
                    precision,
                    recall,
                    f1_score: f1,
                    support: (tp + fn_count),
                },
            );
        }

        // Compute macro F1
        let macro_f1 = class_metrics
            .values()
            .map(|metrics| metrics.f1_score)
            .sum::<f64>()
            / class_metrics.len() as f64;

        // Compute micro F1
        let total_tp: usize = class_metrics
            .values()
            .map(|m| (m.recall * m.support as f64) as usize)
            .sum();
        let total_support: usize = class_metrics.values().map(|m| m.support).sum();
        let micro_f1 = if total_support > 0 {
            total_tp as f64 / total_support as f64
        } else {
            0.0
        };

        self.macro_f1 = macro_f1;
        self.micro_f1 = micro_f1;
        self.per_class_metrics = class_metrics;

        Ok((macro_f1, micro_f1))
    }
}

/// Node embedding quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEmbeddingMetrics {
    /// Silhouette score for embeddings
    pub silhouette_score: f64,
    /// Intra-cluster cohesion
    pub intra_cluster_cohesion: f64,
    /// Inter-cluster separation
    pub inter_cluster_separation: f64,
    /// Embedding alignment with graph structure
    pub structure_alignment: f64,
    /// Neighborhood preservation score
    pub neighborhood_preservation: f64,
}

impl Default for NodeEmbeddingMetrics {
    fn default() -> Self {
        Self {
            silhouette_score: 0.0,
            intra_cluster_cohesion: 0.0,
            inter_cluster_separation: 0.0,
            structure_alignment: 0.0,
            neighborhood_preservation: 0.0,
        }
    }
}

impl NodeEmbeddingMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute embedding quality metrics
    pub fn compute_embedding_quality<F: Float>(
        &mut self,
        embeddings: &ArrayView2<F>,
        adjacency_matrix: &ArrayView2<F>,
        node_labels: Option<&ArrayView1<F>>,
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = embeddings.nrows();
        if adjacency_matrix.nrows() != n_nodes || adjacency_matrix.ncols() != n_nodes {
            return Err(MetricsError::DimensionMismatch(
                "Embeddings and adjacency matrix dimensions must match".to_string(),
            ));
        }

        // Compute neighborhood preservation
        self.neighborhood_preservation =
            self.compute_neighborhood_preservation(embeddings, adjacency_matrix)?;

        // Compute structure alignment
        self.structure_alignment =
            self.compute_structure_alignment(embeddings, adjacency_matrix)?;

        // If labels are provided, compute cluster-based metrics
        if let Some(labels) = node_labels {
            self.silhouette_score = self.compute_silhouette_score(embeddings, labels)?;
        }

        Ok(())
    }

    fn compute_neighborhood_preservation<F: Float>(
        &self,
        embeddings: &ArrayView2<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = embeddings.nrows();
        let mut preservation_sum = 0.0;

        for i in 0..n_nodes {
            let neighbors: Vec<usize> = (0..n_nodes)
                .filter(|&j| i != j && adjacency_matrix[(i, j)] > F::zero())
                .collect();

            if neighbors.is_empty() {
                continue;
            }

            // Compute distances to all other nodes in embedding space
            let mut distances: Vec<(usize, f64)> = (0..n_nodes)
                .filter(|&j| i != j)
                .map(|j| {
                    let dist = (0..embeddings.ncols())
                        .map(|k| (embeddings[(i, k)] - embeddings[(j, k)]).powi(2))
                        .sum::<F>()
                        .sqrt()
                        .to_f64()
                        .unwrap_or(0.0);
                    (j, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Count how many graph neighbors are among k-nearest neighbors in embedding space
            let k = neighbors.len().min(10); // Use top-k neighbors
            let top_k_nodes: HashSet<usize> =
                distances.iter().take(k).map(|(idx, _)| *idx).collect();
            let preserved = neighbors
                .iter()
                .filter(|&&n| top_k_nodes.contains(&n))
                .count();

            preservation_sum += preserved as f64 / neighbors.len() as f64;
        }

        Ok(preservation_sum / n_nodes as f64)
    }

    fn compute_structure_alignment<F: Float>(
        &self,
        embeddings: &ArrayView2<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = embeddings.nrows();
        let mut alignment_sum = 0.0;
        let mut pair_count = 0;

        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                let graph_connected = adjacency_matrix[(i, j)] > F::zero();

                // Compute embedding distance
                let emb_distance = (0..embeddings.ncols())
                    .map(|k| (embeddings[(i, k)] - embeddings[(j, k)]).powi(2))
                    .sum::<F>()
                    .sqrt()
                    .to_f64()
                    .unwrap_or(0.0);

                // Connected nodes should be closer in embedding space
                if graph_connected {
                    alignment_sum += (-emb_distance).exp(); // Higher for smaller distances
                } else {
                    alignment_sum += emb_distance.min(1.0); // Higher for larger distances
                }
                pair_count += 1;
            }
        }

        Ok(alignment_sum / pair_count as f64)
    }

    fn compute_silhouette_score<F: Float>(
        &self,
        embeddings: &ArrayView2<F>,
        labels: &ArrayView1<F>,
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = embeddings.nrows();
        if n_nodes != labels.len() {
            return Err(MetricsError::DimensionMismatch(
                "Embeddings and labels dimensions must match".to_string(),
            ));
        }

        let mut silhouette_sum = 0.0;

        for i in 0..n_nodes {
            let node_label = labels[i];

            // Compute average distance to nodes in same cluster
            let same_cluster_distances: Vec<f64> = (0..n_nodes)
                .filter(|&j| i != j && (labels[j] - node_label).abs() < F::from(0.1).unwrap())
                .map(|j| {
                    (0..embeddings.ncols())
                        .map(|k| (embeddings[(i, k)] - embeddings[(j, k)]).powi(2))
                        .sum::<F>()
                        .sqrt()
                        .to_f64()
                        .unwrap_or(0.0)
                })
                .collect();

            let a = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };

            // Compute minimum average distance to nodes in other clusters
            let mut other_labels: Vec<F> = (0..n_nodes)
                .map(|j| labels[j])
                .filter(|&label| (label - node_label).abs() >= F::from(0.1).unwrap())
                .collect();
            other_labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            other_labels.dedup_by(|a, b| (*a - *b).abs() < F::from(0.1).unwrap());

            let b = other_labels
                .iter()
                .map(|&other_label| {
                    let distances: Vec<f64> = (0..n_nodes)
                        .filter(|&j| (labels[j] - other_label).abs() < F::from(0.1).unwrap())
                        .map(|j| {
                            (0..embeddings.ncols())
                                .map(|k| (embeddings[(i, k)] - embeddings[(j, k)]).powi(2))
                                .sum::<F>()
                                .sqrt()
                                .to_f64()
                                .unwrap_or(0.0)
                        })
                        .collect();

                    if distances.is_empty() {
                        f64::INFINITY
                    } else {
                        distances.iter().sum::<f64>() / distances.len() as f64
                    }
                })
                .fold(f64::INFINITY, f64::min);

            let silhouette = if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };

            silhouette_sum += silhouette;
        }

        Ok(silhouette_sum / n_nodes as f64)
    }
}

/// Homophily-aware evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomophilyAwareMetrics {
    /// Homophily ratio of the graph
    pub homophily_ratio: f64,
    /// Performance on homophilic edges
    pub homophilic_performance: f64,
    /// Performance on heterophilic edges
    pub heterophilic_performance: f64,
    /// Difference in performance
    pub performance_gap: f64,
    /// Local homophily scores
    pub local_homophily: HashMap<usize, f64>, // node_id -> local homophily
}

impl Default for HomophilyAwareMetrics {
    fn default() -> Self {
        Self {
            homophily_ratio: 0.0,
            homophilic_performance: 0.0,
            heterophilic_performance: 0.0,
            performance_gap: 0.0,
            local_homophily: HashMap::new(),
        }
    }
}

impl HomophilyAwareMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute homophily-aware metrics
    pub fn compute_homophily_metrics<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = predictions.len();
        if n_nodes != ground_truth.len() || adjacency_matrix.nrows() != n_nodes {
            return Err(MetricsError::DimensionMismatch(
                "All inputs must have matching dimensions".to_string(),
            ));
        }

        // Compute global homophily ratio
        self.homophily_ratio = self.compute_global_homophily(ground_truth, adjacency_matrix)?;

        // Compute local homophily for each node
        self.local_homophily = self.compute_local_homophily(ground_truth, adjacency_matrix)?;

        // Compute performance on homophilic vs heterophilic edges
        self.compute_edge_performance(predictions, ground_truth, adjacency_matrix)?;

        Ok(())
    }

    fn compute_global_homophily<F: Float>(
        &self,
        labels: &ArrayView1<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<f64>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = labels.len();
        let mut homophilic_edges = 0;
        let mut total_edges = 0;

        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                if adjacency_matrix[(i, j)] > F::zero() {
                    total_edges += 1;
                    if (labels[i] - labels[j]).abs() < F::from(0.1).unwrap() {
                        homophilic_edges += 1;
                    }
                }
            }
        }

        Ok(if total_edges > 0 {
            homophilic_edges as f64 / total_edges as f64
        } else {
            0.0
        })
    }

    fn compute_local_homophily<F: Float>(
        &self,
        labels: &ArrayView1<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<HashMap<usize, f64>>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = labels.len();
        let mut local_homophily = HashMap::new();

        for i in 0..n_nodes {
            let neighbors: Vec<usize> = (0..n_nodes)
                .filter(|&j| i != j && adjacency_matrix[(i, j)] > F::zero())
                .collect();

            if neighbors.is_empty() {
                local_homophily.insert(i, 0.0);
                continue;
            }

            let same_label_neighbors = neighbors
                .iter()
                .filter(|&&j| (labels[i] - labels[j]).abs() < F::from(0.1).unwrap())
                .count();

            let homophily = same_label_neighbors as f64 / neighbors.len() as f64;
            local_homophily.insert(i, homophily);
        }

        Ok(local_homophily)
    }

    fn compute_edge_performance<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
        adjacency_matrix: &ArrayView2<F>,
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        let n_nodes = predictions.len();
        let mut homophilic_correct = 0;
        let mut homophilic_total = 0;
        let mut heterophilic_correct = 0;
        let mut heterophilic_total = 0;

        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                if adjacency_matrix[(i, j)] > F::zero() {
                    let same_true_label =
                        (ground_truth[i] - ground_truth[j]).abs() < F::from(0.1).unwrap();
                    let correct_i =
                        (predictions[i] - ground_truth[i]).abs() < F::from(0.5).unwrap();
                    let correct_j =
                        (predictions[j] - ground_truth[j]).abs() < F::from(0.5).unwrap();
                    let edge_correct = correct_i && correct_j;

                    if same_true_label {
                        homophilic_total += 1;
                        if edge_correct {
                            homophilic_correct += 1;
                        }
                    } else {
                        heterophilic_total += 1;
                        if edge_correct {
                            heterophilic_correct += 1;
                        }
                    }
                }
            }
        }

        self.homophilic_performance = if homophilic_total > 0 {
            homophilic_correct as f64 / homophilic_total as f64
        } else {
            0.0
        };

        self.heterophilic_performance = if heterophilic_total > 0 {
            heterophilic_correct as f64 / heterophilic_total as f64
        } else {
            0.0
        };

        self.performance_gap = (self.homophilic_performance - self.heterophilic_performance).abs();

        Ok(())
    }
}

/// Fairness metrics for node-level predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFairnessMetrics {
    /// Demographic parity difference
    pub demographic_parity: f64,
    /// Equalized odds difference
    pub equalized_odds: f64,
    /// Individual fairness score
    pub individual_fairness: f64,
    /// Group fairness metrics
    pub group_fairness: HashMap<String, GroupFairnessMetrics>,
}

impl Default for NodeFairnessMetrics {
    fn default() -> Self {
        Self {
            demographic_parity: 0.0,
            equalized_odds: 0.0,
            individual_fairness: 0.0,
            group_fairness: HashMap::new(),
        }
    }
}

impl NodeFairnessMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute fairness metrics
    pub fn compute_fairness_metrics<F: Float>(
        &mut self,
        predictions: &ArrayView1<F>,
        ground_truth: &ArrayView1<F>,
        sensitive_attributes: &HashMap<usize, String>,
    ) -> Result<()>
    where
        F: std::iter::Sum + std::fmt::Debug,
    {
        // Compute group-specific metrics
        let mut group_metrics = HashMap::new();
        let groups: std::collections::HashSet<_> = sensitive_attributes.values().cloned().collect();

        for group in groups {
            let group_indices: Vec<usize> = sensitive_attributes
                .iter()
                .filter(|(_, g)| *g == &group)
                .map(|(idx, _)| *idx)
                .collect();

            if group_indices.is_empty() {
                continue;
            }

            let mut tp = 0;
            let mut fp = 0;
            let mut tn = 0;
            let mut fn_count = 0;

            for &idx in &group_indices {
                let pred = predictions[idx].to_f64().unwrap_or(0.0) > 0.5;
                let truth = ground_truth[idx].to_f64().unwrap_or(0.0) > 0.5;

                match (pred, truth) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, false) => tn += 1,
                    (false, true) => fn_count += 1,
                }
            }

            let tpr = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            let fpr = if fp + tn > 0 {
                fp as f64 / (fp + tn) as f64
            } else {
                0.0
            };
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let selection_rate = if group_indices.len() > 0 {
                group_indices
                    .iter()
                    .map(|&idx| {
                        if predictions[idx].to_f64().unwrap_or(0.0) > 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / group_indices.len() as f64
            } else {
                0.0
            };

            group_metrics.insert(
                group.clone(),
                GroupFairnessMetrics {
                    tpr,
                    fpr,
                    precision,
                    selection_rate,
                },
            );
        }

        // Compute overall fairness metrics
        if group_metrics.len() >= 2 {
            let group_names: Vec<_> = group_metrics.keys().collect();
            let group1 = group_metrics.get(group_names[0]).unwrap();
            let group2 = group_metrics.get(group_names[1]).unwrap();

            self.demographic_parity = (group1.selection_rate - group2.selection_rate).abs();
            self.equalized_odds =
                ((group1.tpr - group2.tpr).abs() + (group1.fpr - group2.fpr).abs()) / 2.0;
        }

        self.group_fairness = group_metrics;
        Ok(())
    }
}

impl NodeLevelMetrics {
    /// Create new node-level metrics
    pub fn new() -> Self {
        Self {
            classification_metrics: NodeClassificationMetrics::new(),
            embedding_metrics: NodeEmbeddingMetrics::new(),
            homophily_metrics: HomophilyAwareMetrics::new(),
            fairness_metrics: NodeFairnessMetrics::new(),
        }
    }
}

impl Default for NodeLevelMetrics {
    fn default() -> Self {
        Self::new()
    }
}

use std::collections::HashSet;
