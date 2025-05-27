//! Random walk algorithms
//!
//! This module contains algorithms related to random walks on graphs.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::hash::Hash;

/// Perform a random walk on the graph
///
/// Returns a sequence of nodes visited during the walk.
pub fn random_walk<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    start: &N,
    steps: usize,
    restart_probability: f64,
) -> Result<Vec<N>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    if !graph.contains_node(start) {
        return Err(GraphError::NodeNotFound);
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
                let idx = rng.random_range(0..neighbor_vec.len());
                current = neighbor_vec[idx].clone();
                walk.push(current.clone());
            } else {
                // No neighbors, restart
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
pub fn transition_matrix<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<(Vec<N>, Array2<f64>)>
where
    N: Node + Clone,
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
pub fn personalized_pagerank<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    damping: f64,
    tolerance: f64,
    max_iter: usize,
) -> Result<HashMap<N, f64>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 || !graph.contains_node(source) {
        return Err(GraphError::NodeNotFound);
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
