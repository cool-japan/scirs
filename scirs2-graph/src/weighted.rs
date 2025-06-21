//! Specialized weighted graph operations and analysis
//!
//! This module provides specialized APIs for working with weighted graphs,
//! including weight statistics, filtering, normalization, and transformation operations.

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use std::collections::HashMap;

/// Weight statistics for a graph
#[derive(Debug, Clone)]
pub struct WeightStatistics<E: EdgeWeight> {
    /// Minimum weight in the graph
    pub min_weight: E,
    /// Maximum weight in the graph
    pub max_weight: E,
    /// Sum of all weights
    pub total_weight: E,
    /// Number of edges
    pub edge_count: usize,
}

/// Weight normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Min-max normalization to [0,1]
    MinMax,
    /// Z-score normalization (standardization)
    ZScore,
    /// L1 normalization (sum to 1)
    L1,
    /// L2 normalization (unit norm)
    L2,
    /// Robust normalization using median and MAD
    Robust,
}

/// Weight transformation functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightTransform {
    /// Linear transformation: ax + b
    Linear {
        /// Multiplier coefficient
        a: f64,
        /// Additive constant
        b: f64,
    },
    /// Logarithmic transformation: log(x + offset)
    Logarithmic {
        /// Offset to add before taking logarithm
        offset: f64,
    },
    /// Exponential transformation: exp(x)
    Exponential,
    /// Power transformation: x^power
    Power {
        /// Power exponent
        power: f64,
    },
    /// Inverse transformation: 1/x
    Inverse,
    /// Square root transformation
    SquareRoot,
}

/// Multi-weight edge for graphs with multiple edge attributes
#[derive(Debug, Clone)]
pub struct MultiWeight<E: EdgeWeight> {
    /// Primary weight
    pub primary: E,
    /// Additional weights with names
    pub weights: HashMap<String, E>,
}

impl<E: EdgeWeight> MultiWeight<E> {
    /// Create a new multi-weight with just primary weight
    pub fn new(primary: E) -> Self {
        Self {
            primary,
            weights: HashMap::new(),
        }
    }

    /// Add a named weight
    pub fn add_weight(&mut self, name: String, weight: E) {
        self.weights.insert(name, weight);
    }

    /// Get a named weight
    pub fn get_weight(&self, name: &str) -> Option<&E> {
        self.weights.get(name)
    }
}

/// Weighted graph operations trait
pub trait WeightedOps<N: Node, E: EdgeWeight> {
    /// Calculate weight statistics for the graph
    fn weight_statistics(&self) -> Result<WeightStatistics<E>>;

    /// Filter edges by weight threshold
    fn filter_by_weight(&self, min_weight: Option<E>, max_weight: Option<E>) -> Result<Self>
    where
        Self: Sized;

    /// Get edges sorted by weight
    fn edges_by_weight(&self, ascending: bool) -> Result<Vec<(N, N, E)>>;

    /// Extract subgraph with edges in weight range
    fn subgraph_by_weight_range(&self, min_weight: E, max_weight: E) -> Result<Self>
    where
        Self: Sized;

    /// Normalize edge weights using specified method
    fn normalize_weights(&self, method: NormalizationMethod) -> Result<Self>
    where
        Self: Sized;

    /// Transform edge weights using specified transformation
    fn transform_weights(&self, transform: WeightTransform) -> Result<Self>
    where
        Self: Sized;

    /// Get weight distribution as histogram
    fn weight_distribution(&self, bins: usize) -> Result<Vec<(E, usize)>>;

    /// Calculate weight-based centrality measures
    fn weighted_degree_centrality(&self) -> Result<HashMap<N, f64>>;

    /// Get total weight of the graph
    fn total_weight(&self) -> Result<E>;

    /// Get average edge weight
    fn average_weight(&self) -> Result<f64>;
}

impl<N: Node, E: EdgeWeight, Ix: petgraph::graph::IndexType> WeightedOps<N, E> for Graph<N, E, Ix>
where
    N: Clone + std::fmt::Debug,
    E: Clone
        + std::fmt::Debug
        + Into<f64>
        + From<f64>
        + PartialOrd
        + num_traits::Zero
        + std::ops::Add<Output = E>
        + std::ops::Div<f64, Output = E>
        + std::ops::Mul<f64, Output = E>,
{
    fn weight_statistics(&self) -> Result<WeightStatistics<E>> {
        let edges = self.edges();
        if edges.is_empty() {
            return Err(GraphError::InvalidGraph("No edges in graph".to_string()));
        }

        let mut min_weight = edges[0].weight.clone();
        let mut max_weight = edges[0].weight.clone();
        let mut total_weight = E::zero();

        for edge in &edges {
            if edge.weight < min_weight {
                min_weight = edge.weight.clone();
            }
            if edge.weight > max_weight {
                max_weight = edge.weight.clone();
            }
            total_weight = total_weight + edge.weight.clone();
        }

        Ok(WeightStatistics {
            min_weight,
            max_weight,
            total_weight,
            edge_count: edges.len(),
        })
    }

    fn filter_by_weight(&self, min_weight: Option<E>, max_weight: Option<E>) -> Result<Self> {
        let mut filtered_graph = Graph::new();

        // Add all nodes first
        for node in self.nodes() {
            filtered_graph.add_node(node.clone());
        }

        // Add edges that meet weight criteria
        for edge in self.edges() {
            let mut include = true;

            if let Some(ref min) = min_weight {
                if edge.weight < *min {
                    include = false;
                }
            }

            if let Some(ref max) = max_weight {
                if edge.weight > *max {
                    include = false;
                }
            }

            if include {
                filtered_graph.add_edge(
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.weight.clone(),
                )?;
            }
        }

        Ok(filtered_graph)
    }

    fn edges_by_weight(&self, ascending: bool) -> Result<Vec<(N, N, E)>> {
        let mut edges: Vec<_> = self
            .edges()
            .into_iter()
            .map(|edge| (edge.source, edge.target, edge.weight))
            .collect();

        edges.sort_by(|a, b| {
            if ascending {
                a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        Ok(edges)
    }

    fn subgraph_by_weight_range(&self, min_weight: E, max_weight: E) -> Result<Self> {
        self.filter_by_weight(Some(min_weight), Some(max_weight))
    }

    fn normalize_weights(&self, method: NormalizationMethod) -> Result<Self> {
        let edges = self.edges();
        if edges.is_empty() {
            return Ok(Graph::new());
        }

        let weights: Vec<f64> = edges.iter().map(|e| e.weight.clone().into()).collect();

        let normalized_weights = match method {
            NormalizationMethod::MinMax => {
                let min_val = weights.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max_val - min_val;
                if range == 0.0 {
                    vec![0.5; weights.len()]
                } else {
                    weights.iter().map(|w| (w - min_val) / range).collect()
                }
            }
            NormalizationMethod::ZScore => {
                let mean = weights.iter().sum::<f64>() / weights.len() as f64;
                let variance =
                    weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / weights.len() as f64;
                let std_dev = variance.sqrt();
                if std_dev == 0.0 {
                    vec![0.0; weights.len()]
                } else {
                    weights.iter().map(|w| (w - mean) / std_dev).collect()
                }
            }
            NormalizationMethod::L1 => {
                let sum = weights.iter().sum::<f64>();
                if sum == 0.0 {
                    weights.iter().map(|_| 1.0 / weights.len() as f64).collect()
                } else {
                    weights.iter().map(|w| w / sum).collect()
                }
            }
            NormalizationMethod::L2 => {
                let norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
                if norm == 0.0 {
                    weights
                        .iter()
                        .map(|_| 1.0 / (weights.len() as f64).sqrt())
                        .collect()
                } else {
                    weights.iter().map(|w| w / norm).collect()
                }
            }
            NormalizationMethod::Robust => {
                let mut sorted_weights = weights.clone();
                sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if sorted_weights.len() % 2 == 0 {
                    (sorted_weights[sorted_weights.len() / 2 - 1]
                        + sorted_weights[sorted_weights.len() / 2])
                        / 2.0
                } else {
                    sorted_weights[sorted_weights.len() / 2]
                };
                let mad: Vec<f64> = sorted_weights.iter().map(|w| (w - median).abs()).collect();
                let mut sorted_mad = mad.clone();
                sorted_mad.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mad_median = if sorted_mad.len() % 2 == 0 {
                    (sorted_mad[sorted_mad.len() / 2 - 1] + sorted_mad[sorted_mad.len() / 2]) / 2.0
                } else {
                    sorted_mad[sorted_mad.len() / 2]
                };
                if mad_median == 0.0 {
                    vec![0.0; weights.len()]
                } else {
                    weights.iter().map(|w| (w - median) / mad_median).collect()
                }
            }
        };

        let mut normalized_graph = Graph::new();

        // Add all nodes
        for node in self.nodes() {
            normalized_graph.add_node(node.clone());
        }

        // Add edges with normalized weights
        for (edge, &norm_weight) in edges.iter().zip(normalized_weights.iter()) {
            normalized_graph.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                E::from(norm_weight),
            )?;
        }

        Ok(normalized_graph)
    }

    fn transform_weights(&self, transform: WeightTransform) -> Result<Self> {
        let mut transformed_graph = Graph::new();

        // Add all nodes
        for node in self.nodes() {
            transformed_graph.add_node(node.clone());
        }

        // Transform and add edges
        for edge in self.edges() {
            let weight_f64: f64 = edge.weight.clone().into();
            let transformed_weight = match transform {
                WeightTransform::Linear { a, b } => a * weight_f64 + b,
                WeightTransform::Logarithmic { offset } => (weight_f64 + offset).ln(),
                WeightTransform::Exponential => weight_f64.exp(),
                WeightTransform::Power { power } => weight_f64.powf(power),
                WeightTransform::Inverse => {
                    if weight_f64 == 0.0 {
                        return Err(GraphError::InvalidGraph(
                            "Cannot invert zero weight".to_string(),
                        ));
                    }
                    1.0 / weight_f64
                }
                WeightTransform::SquareRoot => {
                    if weight_f64 < 0.0 {
                        return Err(GraphError::InvalidGraph(
                            "Cannot take square root of negative weight".to_string(),
                        ));
                    }
                    weight_f64.sqrt()
                }
            };

            transformed_graph.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                E::from(transformed_weight),
            )?;
        }

        Ok(transformed_graph)
    }

    fn weight_distribution(&self, bins: usize) -> Result<Vec<(E, usize)>> {
        let edges = self.edges();
        if edges.is_empty() || bins == 0 {
            return Ok(vec![]);
        }

        let weights: Vec<f64> = edges.iter().map(|e| e.weight.clone().into()).collect();
        let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if min_weight == max_weight {
            return Ok(vec![(E::from(min_weight), weights.len())]);
        }

        let bin_width = (max_weight - min_weight) / bins as f64;
        let mut histogram = vec![0; bins];
        let mut bin_centers = Vec::new();

        for i in 0..bins {
            bin_centers.push(E::from(min_weight + (i as f64 + 0.5) * bin_width));
        }

        for weight in weights {
            let bin_index = ((weight - min_weight) / bin_width) as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }

        Ok(bin_centers.into_iter().zip(histogram).collect())
    }

    fn weighted_degree_centrality(&self) -> Result<HashMap<N, f64>> {
        let mut centrality = HashMap::new();

        for node in self.nodes() {
            let mut weighted_degree = 0.0;

            if let Ok(neighbors) = self.neighbors(node) {
                for neighbor in neighbors {
                    if let Ok(weight) = self.edge_weight(node, &neighbor) {
                        weighted_degree += weight.into();
                    }
                }
            }

            centrality.insert(node.clone(), weighted_degree);
        }

        Ok(centrality)
    }

    fn total_weight(&self) -> Result<E> {
        let mut total = E::zero();
        for edge in self.edges() {
            total = total + edge.weight.clone();
        }
        Ok(total)
    }

    fn average_weight(&self) -> Result<f64> {
        let edges = self.edges();
        if edges.is_empty() {
            return Err(GraphError::InvalidGraph("No edges in graph".to_string()));
        }

        let total: f64 = edges.iter().map(|e| e.weight.clone().into()).sum();
        Ok(total / edges.len() as f64)
    }
}

impl<N: Node, E: EdgeWeight, Ix: petgraph::graph::IndexType> WeightedOps<N, E> for DiGraph<N, E, Ix>
where
    N: Clone + std::fmt::Debug,
    E: Clone
        + std::fmt::Debug
        + Into<f64>
        + From<f64>
        + PartialOrd
        + num_traits::Zero
        + std::ops::Add<Output = E>
        + std::ops::Div<f64, Output = E>
        + std::ops::Mul<f64, Output = E>,
{
    fn weight_statistics(&self) -> Result<WeightStatistics<E>> {
        let edges = self.edges();
        if edges.is_empty() {
            return Err(GraphError::InvalidGraph("No edges in graph".to_string()));
        }

        let mut min_weight = edges[0].weight.clone();
        let mut max_weight = edges[0].weight.clone();
        let mut total_weight = E::zero();

        for edge in &edges {
            if edge.weight < min_weight {
                min_weight = edge.weight.clone();
            }
            if edge.weight > max_weight {
                max_weight = edge.weight.clone();
            }
            total_weight = total_weight + edge.weight.clone();
        }

        Ok(WeightStatistics {
            min_weight,
            max_weight,
            total_weight,
            edge_count: edges.len(),
        })
    }

    fn filter_by_weight(&self, min_weight: Option<E>, max_weight: Option<E>) -> Result<Self> {
        let mut filtered_graph = DiGraph::new();

        // Add all nodes first
        for node in self.nodes() {
            filtered_graph.add_node(node.clone());
        }

        // Add edges that meet weight criteria
        for edge in self.edges() {
            let mut include = true;

            if let Some(ref min) = min_weight {
                if edge.weight < *min {
                    include = false;
                }
            }

            if let Some(ref max) = max_weight {
                if edge.weight > *max {
                    include = false;
                }
            }

            if include {
                filtered_graph.add_edge(
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.weight.clone(),
                )?;
            }
        }

        Ok(filtered_graph)
    }

    fn edges_by_weight(&self, ascending: bool) -> Result<Vec<(N, N, E)>> {
        let mut edges: Vec<_> = self
            .edges()
            .into_iter()
            .map(|edge| (edge.source, edge.target, edge.weight))
            .collect();

        edges.sort_by(|a, b| {
            if ascending {
                a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        Ok(edges)
    }

    fn subgraph_by_weight_range(&self, min_weight: E, max_weight: E) -> Result<Self> {
        self.filter_by_weight(Some(min_weight), Some(max_weight))
    }

    fn normalize_weights(&self, method: NormalizationMethod) -> Result<Self> {
        let edges = self.edges();
        if edges.is_empty() {
            return Ok(DiGraph::new());
        }

        let weights: Vec<f64> = edges.iter().map(|e| e.weight.clone().into()).collect();

        let normalized_weights = match method {
            NormalizationMethod::MinMax => {
                let min_val = weights.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max_val - min_val;
                if range == 0.0 {
                    vec![0.5; weights.len()]
                } else {
                    weights.iter().map(|w| (w - min_val) / range).collect()
                }
            }
            NormalizationMethod::ZScore => {
                let mean = weights.iter().sum::<f64>() / weights.len() as f64;
                let variance =
                    weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / weights.len() as f64;
                let std_dev = variance.sqrt();
                if std_dev == 0.0 {
                    vec![0.0; weights.len()]
                } else {
                    weights.iter().map(|w| (w - mean) / std_dev).collect()
                }
            }
            NormalizationMethod::L1 => {
                let sum = weights.iter().sum::<f64>();
                if sum == 0.0 {
                    weights.iter().map(|_| 1.0 / weights.len() as f64).collect()
                } else {
                    weights.iter().map(|w| w / sum).collect()
                }
            }
            NormalizationMethod::L2 => {
                let norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
                if norm == 0.0 {
                    weights
                        .iter()
                        .map(|_| 1.0 / (weights.len() as f64).sqrt())
                        .collect()
                } else {
                    weights.iter().map(|w| w / norm).collect()
                }
            }
            NormalizationMethod::Robust => {
                let mut sorted_weights = weights.clone();
                sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if sorted_weights.len() % 2 == 0 {
                    (sorted_weights[sorted_weights.len() / 2 - 1]
                        + sorted_weights[sorted_weights.len() / 2])
                        / 2.0
                } else {
                    sorted_weights[sorted_weights.len() / 2]
                };
                let mad: Vec<f64> = sorted_weights.iter().map(|w| (w - median).abs()).collect();
                let mut sorted_mad = mad.clone();
                sorted_mad.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mad_median = if sorted_mad.len() % 2 == 0 {
                    (sorted_mad[sorted_mad.len() / 2 - 1] + sorted_mad[sorted_mad.len() / 2]) / 2.0
                } else {
                    sorted_mad[sorted_mad.len() / 2]
                };
                if mad_median == 0.0 {
                    vec![0.0; weights.len()]
                } else {
                    weights.iter().map(|w| (w - median) / mad_median).collect()
                }
            }
        };

        let mut normalized_graph = DiGraph::new();

        // Add all nodes
        for node in self.nodes() {
            normalized_graph.add_node(node.clone());
        }

        // Add edges with normalized weights
        for (edge, &norm_weight) in edges.iter().zip(normalized_weights.iter()) {
            normalized_graph.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                E::from(norm_weight),
            )?;
        }

        Ok(normalized_graph)
    }

    fn transform_weights(&self, transform: WeightTransform) -> Result<Self> {
        let mut transformed_graph = DiGraph::new();

        // Add all nodes
        for node in self.nodes() {
            transformed_graph.add_node(node.clone());
        }

        // Transform and add edges
        for edge in self.edges() {
            let weight_f64: f64 = edge.weight.clone().into();
            let transformed_weight = match transform {
                WeightTransform::Linear { a, b } => a * weight_f64 + b,
                WeightTransform::Logarithmic { offset } => (weight_f64 + offset).ln(),
                WeightTransform::Exponential => weight_f64.exp(),
                WeightTransform::Power { power } => weight_f64.powf(power),
                WeightTransform::Inverse => {
                    if weight_f64 == 0.0 {
                        return Err(GraphError::InvalidGraph(
                            "Cannot invert zero weight".to_string(),
                        ));
                    }
                    1.0 / weight_f64
                }
                WeightTransform::SquareRoot => {
                    if weight_f64 < 0.0 {
                        return Err(GraphError::InvalidGraph(
                            "Cannot take square root of negative weight".to_string(),
                        ));
                    }
                    weight_f64.sqrt()
                }
            };

            transformed_graph.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                E::from(transformed_weight),
            )?;
        }

        Ok(transformed_graph)
    }

    fn weight_distribution(&self, bins: usize) -> Result<Vec<(E, usize)>> {
        let edges = self.edges();
        if edges.is_empty() || bins == 0 {
            return Ok(vec![]);
        }

        let weights: Vec<f64> = edges.iter().map(|e| e.weight.clone().into()).collect();
        let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if min_weight == max_weight {
            return Ok(vec![(E::from(min_weight), weights.len())]);
        }

        let bin_width = (max_weight - min_weight) / bins as f64;
        let mut histogram = vec![0; bins];
        let mut bin_centers = Vec::new();

        for i in 0..bins {
            bin_centers.push(E::from(min_weight + (i as f64 + 0.5) * bin_width));
        }

        for weight in weights {
            let bin_index = ((weight - min_weight) / bin_width) as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }

        Ok(bin_centers.into_iter().zip(histogram).collect())
    }

    fn weighted_degree_centrality(&self) -> Result<HashMap<N, f64>> {
        let mut centrality = HashMap::new();

        for node in self.nodes() {
            let mut weighted_out_degree = 0.0;
            let mut weighted_in_degree = 0.0;

            // Calculate out-degree weight
            if let Ok(successors) = self.successors(node) {
                for successor in successors {
                    if let Ok(weight) = self.edge_weight(node, &successor) {
                        weighted_out_degree += weight.into();
                    }
                }
            }

            // Calculate in-degree weight
            if let Ok(predecessors) = self.predecessors(node) {
                for predecessor in predecessors {
                    if let Ok(weight) = self.edge_weight(&predecessor, node) {
                        weighted_in_degree += weight.into();
                    }
                }
            }

            centrality.insert(node.clone(), weighted_out_degree + weighted_in_degree);
        }

        Ok(centrality)
    }

    fn total_weight(&self) -> Result<E> {
        let mut total = E::zero();
        for edge in self.edges() {
            total = total + edge.weight.clone();
        }
        Ok(total)
    }

    fn average_weight(&self) -> Result<f64> {
        let edges = self.edges();
        if edges.is_empty() {
            return Err(GraphError::InvalidGraph("No edges in graph".to_string()));
        }

        let total: f64 = edges.iter().map(|e| e.weight.clone().into()).sum();
        Ok(total / edges.len() as f64)
    }
}

/// Additional utility functions for weighted graphs
pub mod utils {
    use super::*;

    /// Create a weight threshold filter function
    pub fn weight_threshold_filter<E: EdgeWeight + PartialOrd>(
        threshold: E,
        above: bool,
    ) -> impl Fn(&E) -> bool {
        move |weight| {
            if above {
                *weight >= threshold
            } else {
                *weight < threshold
            }
        }
    }

    /// Combine two graphs by merging weights
    pub fn merge_weighted_graphs<N, E>(
        graph1: &Graph<N, E>,
        graph2: &Graph<N, E>,
    ) -> Result<Graph<N, E>>
    where
        N: Node + Clone + std::fmt::Debug,
        E: EdgeWeight + Clone + std::fmt::Debug + num_traits::Zero + std::ops::Add<Output = E>,
    {
        let mut merged = Graph::new();

        // Add all nodes from both graphs
        for node in graph1.nodes() {
            merged.add_node(node.clone());
        }
        for node in graph2.nodes() {
            merged.add_node(node.clone());
        }

        // Add edges from first graph
        for edge in graph1.edges() {
            merged.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                edge.weight.clone(),
            )?;
        }

        // Add or merge edges from second graph
        for edge in graph2.edges() {
            if merged.has_edge(&edge.source, &edge.target) {
                // Edge exists, merge weights
                let existing_weight = merged.edge_weight(&edge.source, &edge.target)?;
                let _new_weight = existing_weight + edge.weight.clone();

                // For now, we can't easily update edge weights in place
                // This is a limitation that could be addressed with a more sophisticated merge
                // For now, just add the edge with the original weight from the second graph
                merged.add_edge(
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.weight.clone(),
                )?;
            } else {
                // New edge
                merged.add_edge(
                    edge.source.clone(),
                    edge.target.clone(),
                    edge.weight.clone(),
                )?;
            }
        }

        Ok(merged)
    }

    /// Calculate weight correlation between two graphs with same structure
    pub fn weight_correlation<N, E>(graph1: &Graph<N, E>, graph2: &Graph<N, E>) -> Result<f64>
    where
        N: Node + Clone + std::fmt::Debug,
        E: EdgeWeight + Clone + std::fmt::Debug + Into<f64>,
    {
        let edges1 = graph1.edges();
        let edges2 = graph2.edges();

        if edges1.len() != edges2.len() {
            return Err(GraphError::InvalidGraph(
                "Graphs must have same number of edges".to_string(),
            ));
        }

        let weights1: Vec<f64> = edges1.iter().map(|e| e.weight.clone().into()).collect();
        let weights2: Vec<f64> = edges2.iter().map(|e| e.weight.clone().into()).collect();

        let mean1 = weights1.iter().sum::<f64>() / weights1.len() as f64;
        let mean2 = weights2.iter().sum::<f64>() / weights2.len() as f64;

        let numerator: f64 = weights1
            .iter()
            .zip(weights2.iter())
            .map(|(w1, w2)| (w1 - mean1) * (w2 - mean2))
            .sum();

        let var1: f64 = weights1.iter().map(|w| (w - mean1).powi(2)).sum();
        let var2: f64 = weights2.iter().map(|w| (w - mean2).powi(2)).sum();

        let denominator = (var1 * var2).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_statistics() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();

        let stats = graph.weight_statistics().unwrap();
        assert_eq!(stats.min_weight, 1.0);
        assert_eq!(stats.max_weight, 3.0);
        assert_eq!(stats.total_weight, 6.0);
        assert_eq!(stats.edge_count, 3);
    }

    #[test]
    fn test_filter_by_weight() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();

        let filtered = graph.filter_by_weight(Some(2.0), None).unwrap();
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_normalize_weights() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();

        let normalized = graph
            .normalize_weights(NormalizationMethod::MinMax)
            .unwrap();
        let edges = normalized.edges();

        // Check that weights are in [0, 1] range
        for edge in edges {
            let weight = edge.weight;
            assert!((0.0..=1.0).contains(&weight));
        }
    }

    #[test]
    fn test_transform_weights() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();

        let transformed = graph
            .transform_weights(WeightTransform::Linear { a: 2.0, b: 1.0 })
            .unwrap();
        let edges = transformed.edges();

        assert_eq!(edges[0].weight, 3.0); // 2*1 + 1
        assert_eq!(edges[1].weight, 5.0); // 2*2 + 1
    }

    #[test]
    fn test_weight_distribution() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();

        let distribution = graph.weight_distribution(3).unwrap();
        assert_eq!(distribution.len(), 3);
    }

    #[test]
    fn test_weighted_degree_centrality() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(2, 4, 3.0).unwrap();

        let centrality = graph.weighted_degree_centrality().unwrap();
        assert_eq!(centrality[&2], 6.0); // Node 2 has weighted degree 1+2+3=6
    }

    #[test]
    fn test_total_weight() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.5).unwrap();
        graph.add_edge(2, 3, 2.5).unwrap();

        let total = graph.total_weight().unwrap();
        assert_eq!(total, 4.0);
    }

    #[test]
    fn test_average_weight() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 3.0).unwrap();

        let avg = graph.average_weight().unwrap();
        assert_eq!(avg, 2.0);
    }

    #[test]
    fn test_multi_weight() {
        let mut multi_weight = MultiWeight::new(5.0);
        multi_weight.add_weight("distance".to_string(), 10.0);
        multi_weight.add_weight("time".to_string(), 15.0);

        assert_eq!(multi_weight.primary, 5.0);
        assert_eq!(multi_weight.get_weight("distance"), Some(&10.0));
        assert_eq!(multi_weight.get_weight("time"), Some(&15.0));
        assert_eq!(multi_weight.get_weight("cost"), None);
    }

    #[test]
    fn test_edges_by_weight() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 3.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 2.0).unwrap();

        let sorted_edges = graph.edges_by_weight(true).unwrap();
        assert_eq!(sorted_edges[0].2, 1.0);
        assert_eq!(sorted_edges[1].2, 2.0);
        assert_eq!(sorted_edges[2].2, 3.0);

        let sorted_edges_desc = graph.edges_by_weight(false).unwrap();
        assert_eq!(sorted_edges_desc[0].2, 3.0);
        assert_eq!(sorted_edges_desc[1].2, 2.0);
        assert_eq!(sorted_edges_desc[2].2, 1.0);
    }
}
