//! Graph similarity algorithms
//!
//! This module contains algorithms for measuring similarity between nodes and graphs.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::HashSet;
use std::hash::Hash;

/// Compute the Jaccard similarity between two nodes based on their neighbors
///
/// Jaccard similarity is the size of the intersection divided by the size of the union
/// of the neighbor sets.
pub fn jaccard_similarity<N, E, Ix>(graph: &Graph<N, E, Ix>, node1: &N, node2: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    if !graph.contains_node(node1) || !graph.contains_node(node2) {
        return Err(GraphError::NodeNotFound);
    }

    let neighbors1: HashSet<N> = graph
        .neighbors(node1)
        .unwrap_or_default()
        .into_iter()
        .collect();
    let neighbors2: HashSet<N> = graph
        .neighbors(node2)
        .unwrap_or_default()
        .into_iter()
        .collect();

    if neighbors1.is_empty() && neighbors2.is_empty() {
        return Ok(1.0); // Both have no neighbors, consider them similar
    }

    let intersection = neighbors1.intersection(&neighbors2).count();
    let union = neighbors1.union(&neighbors2).count();

    Ok(intersection as f64 / union as f64)
}

/// Compute the cosine similarity between two nodes based on their adjacency vectors
pub fn cosine_similarity<N, E, Ix>(graph: &Graph<N, E, Ix>, node1: &N, node2: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    if !graph.contains_node(node1) || !graph.contains_node(node2) {
        return Err(GraphError::NodeNotFound);
    }

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    // Build adjacency vectors
    let mut vec1 = vec![0.0; n];
    let mut vec2 = vec![0.0; n];

    for (i, node) in nodes.iter().enumerate() {
        if let Ok(weight) = graph.edge_weight(node1, node) {
            vec1[i] = weight.into();
        }
        if let Ok(weight) = graph.edge_weight(node2, node) {
            vec2[i] = weight.into();
        }
    }

    // Compute cosine similarity
    let dot_product: f64 = vec1.iter().zip(&vec2).map(|(a, b)| a * b).sum();
    let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (norm1 * norm2))
}

/// Compute the graph edit distance between two graphs
///
/// This is a simplified version that counts the number of edge additions/deletions
/// needed to transform one graph into another.
pub fn graph_edit_distance<N, E, Ix>(graph1: &Graph<N, E, Ix>, graph2: &Graph<N, E, Ix>) -> usize
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Get all edges from both graphs
    let edges1: HashSet<(N, N)> = graph1
        .edges()
        .into_iter()
        .map(|edge| (edge.source, edge.target))
        .collect();

    let edges2: HashSet<(N, N)> = graph2
        .edges()
        .into_iter()
        .map(|edge| (edge.source, edge.target))
        .collect();

    // Count edges only in graph1 (need to be deleted)
    let edges_to_delete = edges1.difference(&edges2).count();

    // Count edges only in graph2 (need to be added)
    let edges_to_add = edges2.difference(&edges1).count();

    edges_to_delete + edges_to_add
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_jaccard_similarity() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a graph where A and B share some neighbors
        graph.add_edge("A", "C", ())?;
        graph.add_edge("A", "D", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("B", "E", ())?;

        // A's neighbors: {C, D}
        // B's neighbors: {C, E}
        // Intersection: {C} (size 1)
        // Union: {C, D, E} (size 3)
        // Jaccard similarity: 1/3
        let similarity = jaccard_similarity(&graph, &"A", &"B")?;
        assert!((similarity - 1.0 / 3.0).abs() < 1e-6);

        // Node with itself should have similarity 1.0
        let self_similarity = jaccard_similarity(&graph, &"A", &"A")?;
        assert_eq!(self_similarity, 1.0);

        Ok(())
    }

    #[test]
    fn test_cosine_similarity() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create a weighted graph
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("A", "C", 2.0)?;
        graph.add_edge("B", "C", 3.0)?;

        // Test cosine similarity between connected nodes
        let similarity = cosine_similarity(&graph, &"A", &"B")?;
        assert!((0.0..=1.0).contains(&similarity));

        // Node with itself should have similarity 1.0
        let self_similarity = cosine_similarity(&graph, &"A", &"A")?;
        assert!((self_similarity - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_graph_edit_distance() -> GraphResult<()> {
        let mut graph1 = create_graph::<&str, ()>();
        graph1.add_edge("A", "B", ())?;
        graph1.add_edge("B", "C", ())?;

        let mut graph2 = create_graph::<&str, ()>();
        graph2.add_edge("A", "B", ())?;
        graph2.add_edge("B", "C", ())?;
        graph2.add_edge("C", "D", ())?;

        // Graph2 has one additional edge
        let distance = graph_edit_distance(&graph1, &graph2);
        assert_eq!(distance, 1);

        // Same graph should have distance 0
        let self_distance = graph_edit_distance(&graph1, &graph1);
        assert_eq!(self_distance, 0);

        Ok(())
    }
}
