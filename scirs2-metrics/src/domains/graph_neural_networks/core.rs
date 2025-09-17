//! Core types and shared utilities for Graph Neural Network metrics
//!
//! This module provides fundamental types and utilities that are shared
//! across different GNN evaluation components.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Individual class metrics used across classification tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    /// Precision for this class
    pub precision: f64,
    /// Recall for this class
    pub recall: f64,
    /// F1 score for this class
    pub f1_score: f64,
    /// Support (number of instances)
    pub support: usize,
}

impl Default for ClassMetrics {
    fn default() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            support: 0,
        }
    }
}

impl ClassMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Calibration metrics for prediction confidence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (ECE)
    pub ece: f64,
    /// Maximum Calibration Error (MCE)
    pub mce: f64,
    /// Brier score
    pub brier_score: f64,
    /// Reliability diagram data
    pub reliability_diagram: Vec<(f64, f64, usize)>, // (confidence, accuracy, count)
}

impl CalibrationMetrics {
    pub fn new() -> Self {
        Self {
            ece: 0.0,
            mce: 0.0,
            brier_score: 0.0,
            reliability_diagram: Vec::new(),
        }
    }
}

/// Group-specific fairness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupFairnessMetrics {
    /// True Positive Rate for this group
    pub tpr: f64,
    /// False Positive Rate for this group
    pub fpr: f64,
    /// Precision for this group
    pub precision: f64,
    /// Selection rate for this group
    pub selection_rate: f64,
}

impl Default for GroupFairnessMetrics {
    fn default() -> Self {
        Self {
            tpr: 0.0,
            fpr: 0.0,
            precision: 0.0,
            selection_rate: 0.0,
        }
    }
}

impl GroupFairnessMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Node identifier type
pub type NodeId = usize;

/// Edge identifier type
pub type EdgeId = (NodeId, NodeId);

/// Community identifier type
pub type CommunityId = usize;

/// Graph structure representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStructure {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Adjacency list representation
    pub adjacency_list: HashMap<NodeId, HashSet<NodeId>>,
    /// Edge weights (if weighted graph)
    pub edge_weights: Option<HashMap<EdgeId, f64>>,
    /// Node features dimensions
    pub node_feature_dim: Option<usize>,
    /// Edge features dimensions
    pub edge_feature_dim: Option<usize>,
}

impl Default for GraphStructure {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            adjacency_list: HashMap::new(),
            edge_weights: None,
            node_feature_dim: None,
            edge_feature_dim: None,
        }
    }
}

impl GraphStructure {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: Option<f64>) {
        self.adjacency_list
            .entry(from)
            .or_insert_with(HashSet::new)
            .insert(to);
        self.adjacency_list
            .entry(to)
            .or_insert_with(HashSet::new)
            .insert(from);

        if let Some(w) = weight {
            if self.edge_weights.is_none() {
                self.edge_weights = Some(HashMap::new());
            }
            self.edge_weights.as_mut().unwrap().insert((from, to), w);
            self.edge_weights.as_mut().unwrap().insert((to, from), w);
        }
    }

    /// Get degree of a node
    pub fn degree(&self, node: NodeId) -> usize {
        self.adjacency_list
            .get(&node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.adjacency_list
            .get(&node)
            .map_or(Vec::new(), |neighbors| neighbors.iter().cloned().collect())
    }

    /// Check if two nodes are connected
    pub fn are_connected(&self, node1: NodeId, node2: NodeId) -> bool {
        self.adjacency_list
            .get(&node1)
            .map_or(false, |neighbors| neighbors.contains(&node2))
    }
}

/// Triple representation for knowledge graphs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    /// Head entity
    pub head: String,
    /// Relation
    pub relation: String,
    /// Tail entity
    pub tail: String,
}

impl Triple {
    pub fn new(head: String, relation: String, tail: String) -> Self {
        Self {
            head,
            relation,
            tail,
        }
    }
}

/// Molecular structure representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularStructure {
    /// SMILES string representation
    pub smiles: String,
    /// Number of atoms
    pub num_atoms: usize,
    /// Number of bonds
    pub num_bonds: usize,
    /// Atom types
    pub atom_types: Vec<String>,
    /// Bond types
    pub bond_types: Vec<String>,
    /// Molecular weight
    pub molecular_weight: Option<f64>,
    /// 3D coordinates (if available)
    pub coordinates_3d: Option<Vec<(f64, f64, f64)>>,
}

impl Default for MolecularStructure {
    fn default() -> Self {
        Self {
            smiles: String::new(),
            num_atoms: 0,
            num_bonds: 0,
            atom_types: Vec::new(),
            bond_types: Vec::new(),
            molecular_weight: None,
            coordinates_3d: None,
        }
    }
}

impl MolecularStructure {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_smiles(smiles: String) -> Self {
        Self {
            smiles,
            ..Default::default()
        }
    }
}

/// Ranking metrics for link prediction and similar tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingMetrics {
    /// Mean Rank
    pub mean_rank: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Hits@1
    pub hits_at_1: f64,
    /// Hits@3
    pub hits_at_3: f64,
    /// Hits@10
    pub hits_at_10: f64,
    /// Hits@K for various K values
    pub hits_at_k: BTreeMap<usize, f64>,
}

impl Default for RankingMetrics {
    fn default() -> Self {
        Self {
            mean_rank: 0.0,
            mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            hits_at_k: BTreeMap::new(),
        }
    }
}

impl RankingMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Distance metrics for various evaluation tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceMetrics {
    /// Euclidean distance
    pub euclidean: f64,
    /// Manhattan distance
    pub manhattan: f64,
    /// Cosine distance
    pub cosine: f64,
    /// Jaccard distance
    pub jaccard: f64,
    /// Edit distance (for sequences)
    pub edit_distance: Option<usize>,
}

impl Default for DistanceMetrics {
    fn default() -> Self {
        Self {
            euclidean: 0.0,
            manhattan: 0.0,
            cosine: 0.0,
            jaccard: 0.0,
            edit_distance: None,
        }
    }
}

impl DistanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Spectral properties of graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProperties {
    /// Largest eigenvalue
    pub largest_eigenvalue: f64,
    /// Second largest eigenvalue
    pub second_largest_eigenvalue: f64,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Trace of the adjacency matrix
    pub trace: f64,
    /// Number of connected components
    pub num_components: usize,
}

impl Default for SpectralProperties {
    fn default() -> Self {
        Self {
            largest_eigenvalue: 0.0,
            second_largest_eigenvalue: 0.0,
            spectral_gap: 0.0,
            trace: 0.0,
            num_components: 1,
        }
    }
}

impl SpectralProperties {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Statistics for various distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStatistics {
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median
    pub median: f64,
    /// Percentiles
    pub percentiles: BTreeMap<u8, f64>,
}

impl Default for DistributionStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            percentiles: BTreeMap::new(),
        }
    }
}

impl DistributionStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}
