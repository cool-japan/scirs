//! Graph traversal algorithms
//!
//! This module provides breadth-first search (BFS) and depth-first search (DFS)
//! algorithms for both directed and undirected graphs.

use std::collections::{HashSet, VecDeque};

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Performs breadth-first search (BFS) from a given starting node
///
/// # Arguments
/// * `graph` - The graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in BFS order
pub fn breadth_first_search<N, E, Ix>(graph: &Graph<N, E, Ix>, start: &N) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    queue.push_back(start_idx);
    visited.insert(start_idx);

    while let Some(current_idx) = queue.pop_front() {
        result.push(graph.inner()[current_idx].clone());

        // Visit all unvisited neighbors
        for neighbor_idx in graph.inner().neighbors(current_idx) {
            if !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);
                queue.push_back(neighbor_idx);
            }
        }
    }

    Ok(result)
}

/// Performs breadth-first search (BFS) from a given starting node in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in BFS order
pub fn breadth_first_search_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    start: &N,
) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    queue.push_back(start_idx);
    visited.insert(start_idx);

    while let Some(current_idx) = queue.pop_front() {
        result.push(graph.inner()[current_idx].clone());

        // Visit all unvisited neighbors (outgoing edges only for directed graph)
        for neighbor_idx in graph
            .inner()
            .neighbors_directed(current_idx, petgraph::Direction::Outgoing)
        {
            if !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);
                queue.push_back(neighbor_idx);
            }
        }
    }

    Ok(result)
}

/// Performs depth-first search (DFS) from a given starting node
///
/// # Arguments
/// * `graph` - The graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in DFS order
pub fn depth_first_search<N, E, Ix>(graph: &Graph<N, E, Ix>, start: &N) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    stack.push(start_idx);

    while let Some(current_idx) = stack.pop() {
        if !visited.contains(&current_idx) {
            visited.insert(current_idx);
            result.push(graph.inner()[current_idx].clone());

            // Add all unvisited neighbors to the stack (in reverse order for consistent traversal)
            let mut neighbors: Vec<_> = graph.inner().neighbors(current_idx).collect();
            neighbors.reverse();
            for neighbor_idx in neighbors {
                if !visited.contains(&neighbor_idx) {
                    stack.push(neighbor_idx);
                }
            }
        }
    }

    Ok(result)
}

/// Performs depth-first search (DFS) from a given starting node in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to traverse
/// * `start` - The starting node
///
/// # Returns
/// * `Result<Vec<N>>` - The nodes visited in DFS order
pub fn depth_first_search_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>, start: &N) -> Result<Vec<N>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.has_node(start) {
        return Err(GraphError::InvalidGraph(format!(
            "Start node {:?} not found",
            start
        )));
    }

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    let mut result = Vec::new();

    // Find the starting node index
    let start_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *start)
        .unwrap();

    stack.push(start_idx);

    while let Some(current_idx) = stack.pop() {
        if !visited.contains(&current_idx) {
            visited.insert(current_idx);
            result.push(graph.inner()[current_idx].clone());

            // Add all unvisited neighbors to the stack (outgoing edges only for directed graph)
            let mut neighbors: Vec<_> = graph
                .inner()
                .neighbors_directed(current_idx, petgraph::Direction::Outgoing)
                .collect();
            neighbors.reverse();
            for neighbor_idx in neighbors {
                if !visited.contains(&neighbor_idx) {
                    stack.push(neighbor_idx);
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breadth_first_search() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a simple graph: 1 -- 2 -- 3
        //                             |
        //                             4
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = breadth_first_search(&graph, &1).unwrap();

        // BFS should visit nodes level by level
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        // 3 and 4 can be in any order as they're at the same level
        assert!(result.contains(&3));
        assert!(result.contains(&4));
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_breadth_first_search_digraph() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        // Create a directed graph: 1 -> 2 -> 3
        //                               |
        //                               v
        //                               4
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = breadth_first_search_digraph(&graph, &1).unwrap();

        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert!(result.contains(&3));
        assert!(result.contains(&4));
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_depth_first_search() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a simple graph
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(1, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = depth_first_search(&graph, &1).unwrap();

        // DFS should visit nodes depth-first
        assert_eq!(result[0], 1);
        assert_eq!(result.len(), 4);
        assert!(result.contains(&2));
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }

    #[test]
    fn test_depth_first_search_digraph() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        // Create a directed graph
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(1, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        let result = depth_first_search_digraph(&graph, &1).unwrap();

        assert_eq!(result[0], 1);
        assert_eq!(result.len(), 4);
        assert!(result.contains(&2));
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }
}
