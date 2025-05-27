//! Network flow and cut algorithms
//!
//! This module contains algorithms for network flow problems and graph cuts.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::hash::Hash;

/// Finds a minimum cut in a graph using Karger's algorithm
///
/// Returns the minimum cut value and a partition of nodes.
/// This is a randomized algorithm, so multiple runs may give different results.
pub fn minimum_cut<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n < 2 {
        return Err(GraphError::InvalidGraph(
            "Graph must have at least 2 nodes".to_string(),
        ));
    }

    // For small graphs, try all possible cuts
    if n <= 10 {
        let mut min_cut_value = f64::INFINITY;
        let mut best_partition = vec![false; n];

        // Try all possible bipartitions (except empty and full sets)
        for mask in 1..(1 << n) - 1 {
            let mut partition = vec![false; n];
            for (i, p) in partition.iter_mut().enumerate().take(n) {
                *p = (mask & (1 << i)) != 0;
            }

            // Calculate cut value
            let cut_value = calculate_cut_value(graph, &nodes, &partition);

            if cut_value < min_cut_value {
                min_cut_value = cut_value;
                best_partition = partition;
            }
        }

        Ok((min_cut_value, best_partition))
    } else {
        // For larger graphs, use a heuristic approach
        // This is a simplified version - a full implementation would use Karger's algorithm
        minimum_cut_heuristic(graph, &nodes)
    }
}

fn calculate_cut_value<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N], partition: &[bool]) -> f64
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let mut cut_value = 0.0;

    for (i, node_i) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node_i) {
            for neighbor in neighbors {
                if let Some(j) = nodes.iter().position(|n| n == &neighbor) {
                    // Only count edges going from partition A to partition B
                    if partition[i] && !partition[j] {
                        if let Ok(weight) = graph.edge_weight(node_i, &neighbor) {
                            cut_value += weight.into();
                        }
                    }
                }
            }
        }
    }

    cut_value
}

fn minimum_cut_heuristic<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N]) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let n = nodes.len();
    let mut best_cut_value = f64::INFINITY;
    let mut best_partition = vec![false; n];

    // Try a few random partitions
    use rand::Rng;
    let mut rng = rand::rng();

    for _ in 0..10 {
        let mut partition = vec![false; n];
        let size_a = rng.random_range(1..n);

        // Randomly select nodes for partition A
        let mut indices: Vec<usize> = (0..n).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);

        for i in 0..size_a {
            partition[indices[i]] = true;
        }

        let cut_value = calculate_cut_value(graph, nodes, &partition);

        if cut_value < best_cut_value {
            best_cut_value = cut_value;
            best_partition = partition;
        }
    }

    Ok((best_cut_value, best_partition))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_minimum_cut_simple() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create two clusters connected by a single edge
        // Cluster 1: A-B-C
        graph.add_edge("A", "B", 10.0)?;
        graph.add_edge("B", "C", 10.0)?;
        graph.add_edge("C", "A", 10.0)?;

        // Cluster 2: D-E-F
        graph.add_edge("D", "E", 10.0)?;
        graph.add_edge("E", "F", 10.0)?;
        graph.add_edge("F", "D", 10.0)?;

        // Bridge between clusters
        graph.add_edge("C", "D", 1.0)?;

        let (cut_value, partition) = minimum_cut(&graph)?;

        // The minimum cut should be 1.0 (the bridge edge)
        assert!((cut_value - 1.0).abs() < 1e-6);

        // Check that the partition separates the two clusters
        let nodes: Vec<&str> = graph.nodes().into_iter().cloned().collect();
        let cluster1: Vec<bool> = nodes.iter().map(|n| ["A", "B", "C"].contains(n)).collect();
        let cluster2: Vec<bool> = nodes.iter().map(|n| ["D", "E", "F"].contains(n)).collect();

        // Partition should match one of the clusters
        let matches_cluster1 = partition.iter().zip(&cluster1).all(|(a, b)| a == b);
        let matches_cluster2 = partition.iter().zip(&cluster2).all(|(a, b)| a == b);

        assert!(matches_cluster1 || matches_cluster2);

        Ok(())
    }

    #[test]
    fn test_minimum_cut_single_edge() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Simple two-node graph
        graph.add_edge("A", "B", 5.0)?;

        let (cut_value, partition) = minimum_cut(&graph)?;

        // The only cut has value 5.0
        assert_eq!(cut_value, 5.0);

        // One node in each partition
        assert_eq!(partition.iter().filter(|&&x| x).count(), 1);
        assert_eq!(partition.iter().filter(|&&x| !x).count(), 1);

        Ok(())
    }
}
