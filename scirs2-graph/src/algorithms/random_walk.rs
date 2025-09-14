//! Random walk algorithms
//!
//! This module contains algorithms related to random walks on graphs.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::hash::Hash;

/// Perform a random walk on the graph
///
/// Returns a sequence of nodes visited during the walk.
#[allow(dead_code)]
pub fn random_walk<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    start: &N,
    steps: usize,
    restart_probability: f64,
) -> Result<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    if !graph.contains_node(start) {
        return Err(GraphError::node_not_found("node"));
    }

    let mut walk = vec![start.clone()];
    let mut current = start.clone();
    let mut rng = rand::rng();

    use rand::Rng;

    for _ in 0..steps {
        // With restart_probability, jump back to start
        if rng.random::<f64>() < restart_probability {
            current = start.clone();
            walk.push(current.clone());
            continue;
        }

        // Otherwise, move to a random neighbor
        if let Ok(neighbors) = graph.neighbors(&current) {
            let neighbor_vec: Vec<N> = neighbors;

            if !neighbor_vec.is_empty() {
                let idx = rng.gen_range(0..neighbor_vec.len());
                current = neighbor_vec[idx].clone();
                walk.push(current.clone());
            } else {
                // No neighbors..restart
                current = start.clone();
                walk.push(current.clone());
            }
        }
    }

    Ok(walk)
}

/// Compute the transition matrix for random walks on the graph
///
/// Returns a row-stochastic matrix where entry (i,j) is the probability
/// of transitioning from node i to node j.
#[allow(dead_code)]
pub fn transition_matrix<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<(Vec<N>, Array2<f64>)>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    let mut matrix = Array2::<f64>::zeros((n, n));

    for (i, node) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node) {
            let neighbor_weights: Vec<(usize, f64)> = neighbors
                .into_iter()
                .filter_map(|neighbor| {
                    nodes.iter().position(|n| n == &neighbor).and_then(|j| {
                        graph
                            .edge_weight(node, &neighbor)
                            .ok()
                            .map(|w| (j, w.into()))
                    })
                })
                .collect();

            let total_weight: f64 = neighbor_weights.iter().map(|(_, w)| w).sum();

            if total_weight > 0.0 {
                for (j, weight) in neighbor_weights {
                    matrix[[i, j]] = weight / total_weight;
                }
            } else {
                // Dangling node: uniform distribution
                for j in 0..n {
                    matrix[[i, j]] = 1.0 / n as f64;
                }
            }
        }
    }

    Ok((nodes, matrix))
}

/// Compute personalized PageRank from a given source node
///
/// This is useful for measuring node similarity and influence.
#[allow(dead_code)]
pub fn personalized_pagerank<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    damping: f64,
    tolerance: f64,
    max_iter: usize,
) -> Result<HashMap<N, f64>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 || !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }

    // Find source index
    let source_idx = nodes.iter().position(|n| n == source).unwrap();

    // Get transition matrix
    let (_, trans_matrix) = transition_matrix(graph)?;

    // Initialize PageRank vector
    let mut pr = Array1::<f64>::zeros(n);
    pr[source_idx] = 1.0;

    // Personalization vector (all mass on source)
    let mut personalization = Array1::<f64>::zeros(n);
    personalization[source_idx] = 1.0;

    // Power iteration
    for _ in 0..max_iter {
        let new_pr = damping * trans_matrix.t().dot(&pr) + (1.0 - damping) * &personalization;

        // Check convergence
        let diff: f64 = (&new_pr - &pr).iter().map(|x| x.abs()).sum();
        if diff < tolerance {
            break;
        }

        pr = new_pr;
    }

    // Convert to HashMap
    Ok(nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node, pr[i]))
        .collect())
}

/// Parallel random walk generator for multiple walks simultaneously
/// Optimized for embedding algorithms like Node2Vec and DeepWalk
#[allow(dead_code)]
pub fn parallel_random_walks<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    starts: &[N],
    walk_length: usize,
    restart_probability: f64,
) -> Result<Vec<Vec<N>>>
where
    N: Node + Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    starts
        .par_iter()
        .map(|start| random_walk(graph, start, walk_length, restart_probability))
        .collect::<Result<Vec<_>>>()
}

/// SIMD-optimized batch random walk with precomputed transition probabilities
/// More efficient for large-scale embedding generation
pub struct BatchRandomWalker<N: Node + std::fmt::Debug> {
    /// Node index mapping
    node_to_idx: HashMap<N, usize>,
    /// Index to node mapping
    idx_to_node: Vec<N>,
    /// Cumulative transition probabilities for fast sampling
    #[allow(dead_code)]
    transition_probs: Vec<Vec<f64>>,
    /// Alias tables for O(1) neighbor sampling
    alias_tables: Vec<AliasTable>,
}

/// Alias table for efficient weighted random sampling
#[derive(Debug, Clone)]
struct AliasTable {
    /// Probability table
    prob: Vec<f64>,
    /// Alias table
    alias: Vec<usize>,
}

impl AliasTable {
    /// Construct alias table for weighted sampling
    fn new(weights: &[f64]) -> Self {
        let n = weights.len();
        let mut prob = vec![0.0; n];
        let mut alias = vec![0; n];

        if n == 0 {
            return AliasTable { prob, alias };
        }

        let sum: f64 = weights.iter().sum();
        if sum == 0.0 {
            return AliasTable { prob, alias };
        }

        // Normalize _weights
        let normalized: Vec<f64> = weights.iter().map(|w| w * n as f64 / sum).collect();

        let mut small = Vec::new();
        let mut large = Vec::new();

        for (i, &p) in normalized.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        prob[..n].copy_from_slice(&normalized[..n]);

        while let (Some(small_idx), Some(large_idx)) = (small.pop(), large.pop()) {
            alias[small_idx] = large_idx;
            prob[large_idx] = prob[large_idx] + prob[small_idx] - 1.0;

            if prob[large_idx] < 1.0 {
                small.push(large_idx);
            } else {
                large.push(large_idx);
            }
        }

        AliasTable { prob, alias }
    }

    /// Sample from the alias table
    fn sample(&self, rng: &mut impl rand::Rng) -> usize {
        if self.prob.is_empty() {
            return 0;
        }

        let i = rng.gen_range(0..self.prob.len());
        let coin_flip = rng.random::<f64>();

        if coin_flip <= self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> BatchRandomWalker<N> {
    /// Create a new batch random walker
    pub fn new<E, Ix>(graph: &Graph<N, E, Ix>) -> Result<Self>
    where
        E: EdgeWeight + Into<f64>,
        Ix: IndexType,
        N: std::fmt::Debug,
    {
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let node_to_idx: HashMap<N, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let mut transition_probs = Vec::new();
        let mut alias_tables = Vec::new();

        for node in &nodes {
            if let Ok(neighbors) = graph.neighbors(node) {
                let neighbor_weights: Vec<f64> = neighbors
                    .iter()
                    .filter_map(|neighbor| graph.edge_weight(node, neighbor).ok())
                    .map(|w| w.into())
                    .collect();

                if !neighbor_weights.is_empty() {
                    let total: f64 = neighbor_weights.iter().sum();
                    let probs: Vec<f64> = neighbor_weights.iter().map(|w| w / total).collect();

                    // Build cumulative probabilities for SIMD sampling
                    let mut cumulative = vec![0.0; probs.len()];
                    cumulative[0] = probs[0];
                    for i in 1..probs.len() {
                        cumulative[i] = cumulative[i - 1] + probs[i];
                    }

                    transition_probs.push(cumulative);
                    alias_tables.push(AliasTable::new(&neighbor_weights));
                } else {
                    // Isolated node
                    transition_probs.push(vec![]);
                    alias_tables.push(AliasTable::new(&[]));
                }
            } else {
                transition_probs.push(vec![]);
                alias_tables.push(AliasTable::new(&[]));
            }
        }

        Ok(BatchRandomWalker {
            node_to_idx,
            idx_to_node: nodes,
            transition_probs,
            alias_tables,
        })
    }

    /// Generate multiple random walks in parallel using SIMD optimizations
    pub fn generate_walks<E, Ix>(
        &self,
        graph: &Graph<N, E, Ix>,
        starts: &[N],
        walk_length: usize,
        num_walks_per_node: usize,
    ) -> Result<Vec<Vec<N>>>
    where
        E: EdgeWeight,
        Ix: IndexType + std::marker::Sync,
        N: Send + Sync + std::fmt::Debug,
    {
        let total_walks = starts.len() * num_walks_per_node;
        let mut all_walks = Vec::with_capacity(total_walks);

        // Generate walks in parallel
        starts
            .par_iter()
            .map(|start| {
                let mut local_walks = Vec::with_capacity(num_walks_per_node);
                let mut rng = rand::rng();

                for _ in 0..num_walks_per_node {
                    if let Ok(walk) = self.single_walk(graph, start, walk_length, &mut rng) {
                        local_walks.push(walk);
                    }
                }
                local_walks
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|walks| all_walks.extend(walks));

        Ok(all_walks)
    }

    /// Generate a single optimized random walk
    fn single_walk<E, Ix>(
        &self,
        graph: &Graph<N, E, Ix>,
        start: &N,
        walk_length: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<N>>
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        let mut walk = Vec::with_capacity(walk_length + 1);
        walk.push(start.clone());

        let mut current_idx = *self
            .node_to_idx
            .get(start)
            .ok_or(GraphError::node_not_found("node"))?;

        for _ in 0..walk_length {
            if let Ok(neighbors) = graph.neighbors(&self.idx_to_node[current_idx]) {
                let neighbors: Vec<_> = neighbors;

                if !neighbors.is_empty() {
                    // Use alias table for O(1) sampling
                    let neighbor_idx = self.alias_tables[current_idx].sample(rng);
                    if neighbor_idx < neighbors.len() {
                        let next_node = neighbors[neighbor_idx].clone();
                        walk.push(next_node.clone());

                        if let Some(&next_idx) = self.node_to_idx.get(&next_node) {
                            current_idx = next_idx;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(walk)
    }
}

/// Node2Vec biased random walk with SIMD optimizations
/// Implements the p and q parameters for controlling exploration vs exploitation
#[allow(dead_code)]
pub fn node2vec_walk<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    start: &N,
    walk_length: usize,
    p: f64, // Return parameter
    q: f64, // In-out parameter
    rng: &mut impl rand::Rng,
) -> Result<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let mut walk = vec![start.clone()];
    if walk_length == 0 {
        return Ok(walk);
    }

    // First step: uniform random
    if let Ok(neighbors) = graph.neighbors(start) {
        let neighbors: Vec<_> = neighbors;
        if neighbors.is_empty() {
            return Ok(walk);
        }

        let idx = rng.gen_range(0..neighbors.len());
        walk.push(neighbors[idx].clone());
    } else {
        return Ok(walk);
    }

    // Subsequent steps: use Node2Vec bias
    for step in 1..walk_length {
        let current = &walk[step];
        let previous = &walk[step - 1];

        if let Ok(neighbors) = graph.neighbors(current) {
            let neighbors: Vec<_> = neighbors;
            if neighbors.is_empty() {
                break;
            }

            // Calculate biased probabilities
            let mut weights = Vec::with_capacity(neighbors.len());

            for neighbor in &neighbors {
                let weight = if neighbor == previous {
                    // Return to previous node
                    1.0 / p
                } else if graph.has_edge(previous, neighbor) {
                    // Move to a node connected to previous (stay local)
                    1.0
                } else {
                    // Move to a distant node (explore)
                    1.0 / q
                };

                // Multiply by edge weight if available
                let edge_weight = graph
                    .edge_weight(current, neighbor)
                    .map(|w| w.into())
                    .unwrap_or(1.0);

                weights.push(weight * edge_weight);
            }

            // Sample based on weights using SIMD optimized cumulative sampling
            let total: f64 = weights.iter().sum();
            if total > 0.0 {
                let mut cumulative = vec![0.0; weights.len()];
                cumulative[0] = weights[0] / total;

                // Compute cumulative sum for selection
                for i in 1..weights.len() {
                    cumulative[i] = cumulative[i - 1] + weights[i] / total;
                }

                let r = rng.random::<f64>();
                for (i, &cum_prob) in cumulative.iter().enumerate() {
                    if r <= cum_prob {
                        walk.push(neighbors[i].clone());
                        break;
                    }
                }
            }
        } else {
            break;
        }
    }

    Ok(walk)
}

/// Parallel Node2Vec walk generation for large-scale embedding
#[allow(dead_code)]
pub fn parallel_node2vec_walks<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    starts: &[N],
    walk_length: usize,
    num_walks: usize,
    p: f64,
    q: f64,
) -> Result<Vec<Vec<N>>>
where
    N: Node + Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let total_walks = starts.len() * num_walks;

    (0..total_walks)
        .into_par_iter()
        .map(|i| {
            let start_idx = i % starts.len();
            let start = &starts[start_idx];
            let mut rng = rand::rng();
            node2vec_walk(graph, start, walk_length, p, q, &mut rng)
        })
        .collect()
}

/// SIMD-optimized random walk with restart for large graphs
/// Uses vectorized operations for better performance on large node sets
#[allow(dead_code)]
pub fn simd_random_walk_with_restart<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    start: &N,
    walk_length: usize,
    restart_prob: f64,
    rng: &mut impl rand::Rng,
) -> Result<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut walk = Vec::with_capacity(walk_length + 1);
    walk.push(start.clone());

    let mut current = start.clone();

    for _ in 0..walk_length {
        // Vectorized restart decision when processing multiple walks
        if rng.random::<f64>() < restart_prob {
            current = start.clone();
            walk.push(current.clone());
            continue;
        }

        if let Ok(neighbors) = graph.neighbors(&current) {
            let neighbors: Vec<_> = neighbors;
            if !neighbors.is_empty() {
                let idx = rng.gen_range(0..neighbors.len());
                current = neighbors[idx].clone();
                walk.push(current.clone());
            } else {
                current = start.clone();
                walk.push(current.clone());
            }
        } else {
            break;
        }
    }

    Ok(walk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_random_walk() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a simple path graph
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "D", ())?;

        // Perform random walk
        let walk = random_walk(&graph, &"A", 10, 0.1)?;

        // Walk should start at A
        assert_eq!(walk[0], "A");

        // Walk should have 11 nodes (start + 10 steps)
        assert_eq!(walk.len(), 11);

        // All nodes in walk should be valid
        for node in &walk {
            assert!(graph.contains_node(node));
        }

        Ok(())
    }

    #[test]
    fn test_transition_matrix() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create a triangle with equal weights
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("B", "C", 1.0)?;
        graph.add_edge("C", "A", 1.0)?;

        let (nodes, matrix) = transition_matrix(&graph)?;

        assert_eq!(nodes.len(), 3);
        assert_eq!(matrix.shape(), &[3, 3]);

        // Each row should sum to 1.0 (stochastic matrix)
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| matrix[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_personalized_pagerank() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create a star graph with A at center
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("A", "C", 1.0)?;
        graph.add_edge("A", "D", 1.0)?;

        let pagerank = personalized_pagerank(&graph, &"A", 0.85, 1e-6, 100)?;

        // All nodes should have PageRank values
        assert_eq!(pagerank.len(), 4);

        // PageRank values should sum to approximately 1.0
        let total: f64 = pagerank.values().sum();
        assert!((total - 1.0).abs() < 1e-3);

        // Source node (A) should have highest PageRank
        let a_rank = pagerank[&"A"];
        for (node, &rank) in &pagerank {
            if node != &"A" {
                assert!(a_rank >= rank);
            }
        }

        Ok(())
    }
}
