//! Graph motif finding algorithms
//!
//! This module contains algorithms for finding small recurring subgraph patterns (motifs).

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::hash::Hash;

/// Finds all occurrences of a specific motif pattern in a graph
///
/// A motif is a small recurring subgraph pattern. This function finds all
/// instances of common motifs like triangles, squares, or stars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MotifType {
    /// Triangle (3-cycle)
    Triangle,
    /// Square (4-cycle)
    Square,
    /// Star with 3 leaves
    Star3,
    /// Clique of size 4
    Clique4,
}

/// Find all occurrences of a specified motif in the graph
pub fn find_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>, motif_type: MotifType) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    match motif_type {
        MotifType::Triangle => find_triangles(graph),
        MotifType::Square => find_squares(graph),
        MotifType::Star3 => find_star3s(graph),
        MotifType::Clique4 => find_clique4s(graph),
    }
}

fn find_triangles<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut triangles = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each triplet of nodes, check if they form a triangle
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if graph.has_edge(&nodes[i], &nodes[j]) {
                for k in j + 1..nodes.len() {
                    if graph.has_edge(&nodes[i], &nodes[k]) && graph.has_edge(&nodes[j], &nodes[k])
                    {
                        triangles.push(vec![nodes[i].clone(), nodes[j].clone(), nodes[k].clone()]);
                    }
                }
            }
        }
    }

    triangles
}

fn find_squares<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut squares = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each quadruplet of nodes, check if they form a square
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !graph.has_edge(&nodes[i], &nodes[j]) {
                continue;
            }
            for k in j + 1..nodes.len() {
                if !graph.has_edge(&nodes[j], &nodes[k]) {
                    continue;
                }
                for l in k + 1..nodes.len() {
                    if graph.has_edge(&nodes[k], &nodes[l])
                        && graph.has_edge(&nodes[l], &nodes[i])
                        && !graph.has_edge(&nodes[i], &nodes[k])
                        && !graph.has_edge(&nodes[j], &nodes[l])
                    {
                        squares.push(vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[l].clone(),
                        ]);
                    }
                }
            }
        }
    }

    squares
}

fn find_star3s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut stars = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each node as center, find if it has exactly 3 neighbors that aren't connected
    for center in &nodes {
        if let Ok(neighbors) = graph.neighbors(center) {
            let neighbor_list: Vec<N> = neighbors;

            if neighbor_list.len() >= 3 {
                // Check all combinations of 3 neighbors
                for i in 0..neighbor_list.len() {
                    for j in i + 1..neighbor_list.len() {
                        for k in j + 1..neighbor_list.len() {
                            // Check that the neighbors aren't connected to each other
                            if !graph.has_edge(&neighbor_list[i], &neighbor_list[j])
                                && !graph.has_edge(&neighbor_list[j], &neighbor_list[k])
                                && !graph.has_edge(&neighbor_list[i], &neighbor_list[k])
                            {
                                stars.push(vec![
                                    center.clone(),
                                    neighbor_list[i].clone(),
                                    neighbor_list[j].clone(),
                                    neighbor_list[k].clone(),
                                ]);
                            }
                        }
                    }
                }
            }
        }
    }

    stars
}

fn find_clique4s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut cliques = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each quadruplet of nodes, check if they form a complete graph
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !graph.has_edge(&nodes[i], &nodes[j]) {
                continue;
            }
            for k in j + 1..nodes.len() {
                if !graph.has_edge(&nodes[i], &nodes[k]) || !graph.has_edge(&nodes[j], &nodes[k]) {
                    continue;
                }
                for l in k + 1..nodes.len() {
                    if graph.has_edge(&nodes[i], &nodes[l])
                        && graph.has_edge(&nodes[j], &nodes[l])
                        && graph.has_edge(&nodes[k], &nodes[l])
                    {
                        cliques.push(vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[l].clone(),
                        ]);
                    }
                }
            }
        }
    }

    cliques
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_find_triangles() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a triangle ABC
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "A", ())?;

        // Add another node D connected to A (not forming new triangles)
        graph.add_edge("A", "D", ())?;

        let triangles = find_motifs(&graph, MotifType::Triangle);
        assert_eq!(triangles.len(), 1);

        // The triangle should contain A, B, and C
        let triangle = &triangles[0];
        assert_eq!(triangle.len(), 3);
        assert!(triangle.contains(&"A"));
        assert!(triangle.contains(&"B"));
        assert!(triangle.contains(&"C"));

        Ok(())
    }

    #[test]
    fn test_find_squares() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a square ABCD
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "D", ())?;
        graph.add_edge("D", "A", ())?;

        let squares = find_motifs(&graph, MotifType::Square);
        assert_eq!(squares.len(), 1);

        let square = &squares[0];
        assert_eq!(square.len(), 4);

        Ok(())
    }

    #[test]
    fn test_find_star3() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a star with center A and leaves B, C, D
        graph.add_edge("A", "B", ())?;
        graph.add_edge("A", "C", ())?;
        graph.add_edge("A", "D", ())?;

        let stars = find_motifs(&graph, MotifType::Star3);
        assert_eq!(stars.len(), 1);

        let star = &stars[0];
        assert_eq!(star.len(), 4);
        assert!(star.contains(&"A")); // Center should be included

        Ok(())
    }

    #[test]
    fn test_find_clique4() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a complete graph K4
        let nodes = ["A", "B", "C", "D"];
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], ())?;
            }
        }

        let cliques = find_motifs(&graph, MotifType::Clique4);
        assert_eq!(cliques.len(), 1);

        let clique = &cliques[0];
        assert_eq!(clique.len(), 4);

        Ok(())
    }
}
