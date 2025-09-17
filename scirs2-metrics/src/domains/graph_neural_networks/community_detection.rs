//! Community detection evaluation metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating community detection and clustering
//! performance in graph neural networks.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{CommunityId, NodeId};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Community detection evaluation metrics
#[derive(Debug, Clone)]
pub struct CommunityDetectionMetrics {
    /// Modularity score
    pub modularity: f64,
    /// Normalized Mutual Information
    pub nmi: f64,
    /// Adjusted Rand Index
    pub ari: f64,
    /// Community conductance
    pub conductance: f64,
    /// Overlapping community metrics
    pub overlapping_metrics: OverlappingCommunityMetrics,
}

/// Overlapping community detection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlappingCommunityMetrics {
    /// Overlapping NMI
    pub overlapping_nmi: f64,
    /// Community coverage
    pub coverage: f64,
    /// Community performance
    pub performance: f64,
}

impl Default for CommunityDetectionMetrics {
    fn default() -> Self {
        Self {
            modularity: 0.0,
            nmi: 0.0,
            ari: 0.0,
            conductance: 0.0,
            overlapping_metrics: OverlappingCommunityMetrics::default(),
        }
    }
}

impl Default for OverlappingCommunityMetrics {
    fn default() -> Self {
        Self {
            overlapping_nmi: 0.0,
            coverage: 0.0,
            performance: 0.0,
        }
    }
}

impl CommunityDetectionMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute modularity score
    pub fn compute_modularity(
        &mut self,
        communities: &HashMap<NodeId, CommunityId>,
        adjacency_matrix: &[Vec<f64>],
    ) -> Result<f64> {
        let n = adjacency_matrix.len();
        if n == 0 {
            return Ok(0.0);
        }

        // Calculate total edge weight
        let total_weight: f64 = adjacency_matrix.iter().flatten().sum::<f64>() / 2.0; // Divide by 2 for undirected graphs

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        // Calculate node degrees
        let degrees: Vec<f64> = adjacency_matrix
            .iter()
            .map(|row| row.iter().sum())
            .collect();

        let mut modularity = 0.0;

        for i in 0..n {
            for j in 0..n {
                if let (Some(&comm_i), Some(&comm_j)) = (communities.get(&i), communities.get(&j)) {
                    if comm_i == comm_j {
                        let expected = degrees[i] * degrees[j] / (2.0 * total_weight);
                        modularity += adjacency_matrix[i][j] - expected;
                    }
                }
            }
        }

        self.modularity = modularity / (2.0 * total_weight);
        Ok(self.modularity)
    }

    /// Compute Normalized Mutual Information
    pub fn compute_nmi(
        &mut self,
        predicted_communities: &HashMap<NodeId, CommunityId>,
        true_communities: &HashMap<NodeId, CommunityId>,
    ) -> Result<f64> {
        if predicted_communities.is_empty() || true_communities.is_empty() {
            return Ok(0.0);
        }

        // Get all nodes
        let all_nodes: HashSet<NodeId> = predicted_communities
            .keys()
            .chain(true_communities.keys())
            .cloned()
            .collect();

        let n = all_nodes.len() as f64;
        if n == 0.0 {
            return Ok(0.0);
        }

        // Build confusion matrix
        let mut confusion_matrix: HashMap<(CommunityId, CommunityId), usize> = HashMap::new();
        let mut pred_counts: HashMap<CommunityId, usize> = HashMap::new();
        let mut true_counts: HashMap<CommunityId, usize> = HashMap::new();

        for node in &all_nodes {
            if let (Some(&pred_comm), Some(&true_comm)) =
                (predicted_communities.get(node), true_communities.get(node))
            {
                *confusion_matrix.entry((pred_comm, true_comm)).or_insert(0) += 1;
                *pred_counts.entry(pred_comm).or_insert(0) += 1;
                *true_counts.entry(true_comm).or_insert(0) += 1;
            }
        }

        // Calculate entropies
        let h_pred = pred_counts
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        let h_true = true_counts
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        // Calculate mutual information
        let mutual_info = confusion_matrix
            .values()
            .zip(confusion_matrix.keys())
            .map(|(&n_ij, &(pred_comm, true_comm))| {
                let p_ij = n_ij as f64 / n;
                let p_i = *pred_counts.get(&pred_comm).unwrap_or(&0) as f64 / n;
                let p_j = *true_counts.get(&true_comm).unwrap_or(&0) as f64 / n;

                if p_ij > 0.0 && p_i > 0.0 && p_j > 0.0 {
                    p_ij * (p_ij / (p_i * p_j)).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        // Calculate NMI
        self.nmi = if h_pred + h_true > 0.0 {
            2.0 * mutual_info / (h_pred + h_true)
        } else {
            0.0
        };

        Ok(self.nmi)
    }

    /// Compute Adjusted Rand Index
    pub fn compute_ari(
        &mut self,
        predicted_communities: &HashMap<NodeId, CommunityId>,
        true_communities: &HashMap<NodeId, CommunityId>,
    ) -> Result<f64> {
        if predicted_communities.is_empty() || true_communities.is_empty() {
            return Ok(0.0);
        }

        let all_nodes: HashSet<NodeId> = predicted_communities
            .keys()
            .chain(true_communities.keys())
            .cloned()
            .collect();

        let n = all_nodes.len();
        if n == 0 {
            return Ok(0.0);
        }

        // Build confusion matrix
        let mut confusion_matrix: HashMap<(CommunityId, CommunityId), usize> = HashMap::new();
        let mut pred_counts: HashMap<CommunityId, usize> = HashMap::new();
        let mut true_counts: HashMap<CommunityId, usize> = HashMap::new();

        for node in &all_nodes {
            if let (Some(&pred_comm), Some(&true_comm)) =
                (predicted_communities.get(node), true_communities.get(node))
            {
                *confusion_matrix.entry((pred_comm, true_comm)).or_insert(0) += 1;
                *pred_counts.entry(pred_comm).or_insert(0) += 1;
                *true_counts.entry(true_comm).or_insert(0) += 1;
            }
        }

        // Calculate ARI components
        let index = confusion_matrix
            .values()
            .map(|&count| {
                if count >= 2 {
                    count * (count - 1) / 2
                } else {
                    0
                }
            })
            .sum::<usize>() as f64;

        let expected_index = {
            let sum_pred = pred_counts
                .values()
                .map(|&count| {
                    if count >= 2 {
                        count * (count - 1) / 2
                    } else {
                        0
                    }
                })
                .sum::<usize>() as f64;

            let sum_true = true_counts
                .values()
                .map(|&count| {
                    if count >= 2 {
                        count * (count - 1) / 2
                    } else {
                        0
                    }
                })
                .sum::<usize>() as f64;

            let total_pairs = if n >= 2 { n * (n - 1) / 2 } else { 0 } as f64;

            if total_pairs > 0.0 {
                sum_pred * sum_true / total_pairs
            } else {
                0.0
            }
        };

        let max_index = {
            let sum_pred = pred_counts
                .values()
                .map(|&count| {
                    if count >= 2 {
                        count * (count - 1) / 2
                    } else {
                        0
                    }
                })
                .sum::<usize>() as f64;

            let sum_true = true_counts
                .values()
                .map(|&count| {
                    if count >= 2 {
                        count * (count - 1) / 2
                    } else {
                        0
                    }
                })
                .sum::<usize>() as f64;

            (sum_pred + sum_true) / 2.0
        };

        self.ari = if max_index - expected_index > 0.0 {
            (index - expected_index) / (max_index - expected_index)
        } else {
            0.0
        };

        Ok(self.ari)
    }

    /// Compute community conductance
    pub fn compute_conductance(
        &mut self,
        communities: &HashMap<NodeId, CommunityId>,
        adjacency_matrix: &[Vec<f64>],
    ) -> Result<f64> {
        let n = adjacency_matrix.len();
        if n == 0 || communities.is_empty() {
            return Ok(0.0);
        }

        let unique_communities: HashSet<CommunityId> = communities.values().cloned().collect();
        let mut total_conductance = 0.0;

        for community in &unique_communities {
            let community_nodes: HashSet<NodeId> = communities
                .iter()
                .filter(|(_, &comm)| comm == *community)
                .map(|(&node, _)| node)
                .collect();

            if community_nodes.is_empty() {
                continue;
            }

            let mut internal_edges = 0.0;
            let mut external_edges = 0.0;

            for &node in &community_nodes {
                if node >= n {
                    continue;
                }

                for (neighbor, &weight) in adjacency_matrix[node].iter().enumerate() {
                    if weight > 0.0 {
                        if community_nodes.contains(&neighbor) {
                            internal_edges += weight;
                        } else {
                            external_edges += weight;
                        }
                    }
                }
            }

            // Avoid double counting internal edges
            internal_edges /= 2.0;

            let volume = internal_edges + external_edges;
            let conductance = if volume > 0.0 {
                external_edges / volume
            } else {
                0.0
            };

            total_conductance += conductance;
        }

        self.conductance = if unique_communities.len() > 0 {
            total_conductance / unique_communities.len() as f64
        } else {
            0.0
        };

        Ok(self.conductance)
    }
}

impl OverlappingCommunityMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute overlapping community metrics
    pub fn compute_overlapping_metrics(
        &mut self,
        predicted_memberships: &HashMap<NodeId, Vec<CommunityId>>,
        true_memberships: &HashMap<NodeId, Vec<CommunityId>>,
    ) -> Result<()> {
        if predicted_memberships.is_empty() || true_memberships.is_empty() {
            return Ok(());
        }

        // Compute coverage
        let all_nodes: HashSet<NodeId> = predicted_memberships
            .keys()
            .chain(true_memberships.keys())
            .cloned()
            .collect();

        let mut covered_nodes = 0;
        for node in &all_nodes {
            if let Some(pred_comms) = predicted_memberships.get(node) {
                if !pred_comms.is_empty() {
                    covered_nodes += 1;
                }
            }
        }

        self.coverage = if all_nodes.len() > 0 {
            covered_nodes as f64 / all_nodes.len() as f64
        } else {
            0.0
        };

        // Compute performance (simplified version)
        let mut correct_overlaps = 0;
        let mut total_comparisons = 0;

        for node in &all_nodes {
            if let (Some(pred_comms), Some(true_comms)) =
                (predicted_memberships.get(node), true_memberships.get(node))
            {
                let pred_set: HashSet<CommunityId> = pred_comms.iter().cloned().collect();
                let true_set: HashSet<CommunityId> = true_comms.iter().cloned().collect();

                let intersection = pred_set.intersection(&true_set).count();
                let union = pred_set.union(&true_set).count();

                if union > 0 {
                    correct_overlaps += intersection;
                    total_comparisons += union;
                }
            }
        }

        self.performance = if total_comparisons > 0 {
            correct_overlaps as f64 / total_comparisons as f64
        } else {
            0.0
        };

        Ok(())
    }
}
