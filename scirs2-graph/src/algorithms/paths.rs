//! Path algorithms for graphs
//!
//! This module contains algorithms for finding special paths in graphs,
//! including Eulerian and Hamiltonian paths.

use crate::base::{EdgeWeight, Graph, Node};
use std::collections::{HashSet, VecDeque};

/// Result of Eulerian path/circuit check
#[derive(Debug, Clone, PartialEq)]
pub enum EulerianType {
    /// Graph has an Eulerian circuit (closed path visiting every edge exactly once)
    Circuit,
    /// Graph has an Eulerian path (open path visiting every edge exactly once)
    Path,
    /// Graph has neither Eulerian circuit nor path
    None,
}

/// Checks if a graph has an Eulerian path or circuit
///
/// An Eulerian circuit exists if all vertices have even degree.
/// An Eulerian path exists if exactly 0 or 2 vertices have odd degree.
///
/// # Arguments
/// * `graph` - The undirected graph to check
///
/// # Returns
/// * The type of Eulerian structure in the graph
pub fn eulerian_type<N, E, Ix>(graph: &Graph<N, E, Ix>) -> EulerianType
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    // First check if the graph is connected (ignoring isolated vertices)
    let non_isolated: Vec<_> = graph
        .inner()
        .node_indices()
        .filter(|&idx| graph.inner().edges(idx).count() > 0)
        .collect();

    if non_isolated.is_empty() {
        return EulerianType::Circuit; // Empty graph technically has a circuit
    }

    // Check connectivity among non-isolated vertices
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(non_isolated[0]);
    visited.insert(non_isolated[0]);

    while let Some(node) = queue.pop_front() {
        for neighbor in graph.inner().neighbors(node) {
            if !visited.contains(&neighbor) && non_isolated.contains(&neighbor) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    if visited.len() != non_isolated.len() {
        return EulerianType::None; // Graph is not connected
    }

    // Count vertices with odd degree
    let mut odd_degree_count = 0;
    for node in graph.inner().node_indices() {
        let degree = graph.inner().edges(node).count();
        if degree % 2 == 1 {
            odd_degree_count += 1;
        }
    }

    match odd_degree_count {
        0 => EulerianType::Circuit,
        2 => EulerianType::Path,
        _ => EulerianType::None,
    }
}

/// Checks if a graph has a Hamiltonian path (visiting every vertex exactly once)
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// * `bool` - True if a Hamiltonian path exists
pub fn has_hamiltonian_path<N, E, Ix>(graph: &Graph<N, E, Ix>) -> bool
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n == 0 {
        return true;
    }

    let nodes: Vec<_> = graph.inner().node_indices().collect();

    // Try starting from each node
    for &start in &nodes {
        let mut visited = vec![false; n];
        visited[start.index()] = true;

        if hamiltonian_path_dfs(graph, start, &mut visited, 1, n) {
            return true;
        }
    }

    false
}

/// DFS helper for Hamiltonian path
fn hamiltonian_path_dfs<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    current: petgraph::graph::NodeIndex<Ix>,
    visited: &mut Vec<bool>,
    count: usize,
    n: usize,
) -> bool
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if count == n {
        return true;
    }

    for neighbor in graph.inner().neighbors(current) {
        if !visited[neighbor.index()] {
            visited[neighbor.index()] = true;

            if hamiltonian_path_dfs(graph, neighbor, visited, count + 1, n) {
                return true;
            }

            visited[neighbor.index()] = false;
        }
    }

    false
}

/// Checks if a graph has a Hamiltonian circuit (cycle visiting every vertex exactly once)
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// * `bool` - True if a Hamiltonian circuit exists
pub fn has_hamiltonian_circuit<N, E, Ix>(graph: &Graph<N, E, Ix>) -> bool
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n == 0 {
        return true;
    }
    if n < 3 {
        return false;
    }

    let nodes: Vec<_> = graph.inner().node_indices().collect();
    let start = nodes[0];

    let mut visited = vec![false; n];
    visited[start.index()] = true;

    hamiltonian_circuit_dfs(graph, start, start, &mut visited, 1, n)
}

/// DFS helper for Hamiltonian circuit
fn hamiltonian_circuit_dfs<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    current: petgraph::graph::NodeIndex<Ix>,
    start: petgraph::graph::NodeIndex<Ix>,
    visited: &mut Vec<bool>,
    count: usize,
    n: usize,
) -> bool
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if count == n {
        // Check if we can return to start
        return graph.inner().contains_edge(current, start);
    }

    for neighbor in graph.inner().neighbors(current) {
        if !visited[neighbor.index()] {
            visited[neighbor.index()] = true;

            if hamiltonian_circuit_dfs(graph, neighbor, start, visited, count + 1, n) {
                return true;
            }

            visited[neighbor.index()] = false;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_eulerian_circuit() -> GraphResult<()> {
        // Create a square (all vertices have even degree)
        let mut graph = create_graph::<i32, ()>();

        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 3, ())?;
        graph.add_edge(3, 0, ())?;

        assert_eq!(eulerian_type(&graph), EulerianType::Circuit);

        Ok(())
    }

    #[test]
    fn test_eulerian_path() -> GraphResult<()> {
        // Create a path graph (2 vertices with odd degree)
        let mut graph = create_graph::<i32, ()>();

        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;

        assert_eq!(eulerian_type(&graph), EulerianType::Path);

        Ok(())
    }

    #[test]
    fn test_no_eulerian() -> GraphResult<()> {
        // Create a graph with 4 vertices of odd degree (no Eulerian path/circuit possible)
        let mut graph = create_graph::<i32, ()>();

        // Create a graph like: 0-1-2
        //                         X
        //                       3-4-5
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(1, 4, ())?; // Bridge
        graph.add_edge(3, 4, ())?;
        graph.add_edge(4, 5, ())?;

        assert_eq!(eulerian_type(&graph), EulerianType::None);

        Ok(())
    }

    #[test]
    fn test_hamiltonian_path() -> GraphResult<()> {
        // Create a path graph (has Hamiltonian path)
        let mut graph = create_graph::<i32, ()>();

        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 3, ())?;

        assert!(has_hamiltonian_path(&graph));

        Ok(())
    }

    #[test]
    fn test_hamiltonian_circuit() -> GraphResult<()> {
        // Create a square (has Hamiltonian circuit)
        let mut graph = create_graph::<i32, ()>();

        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 3, ())?;
        graph.add_edge(3, 0, ())?;

        assert!(has_hamiltonian_circuit(&graph));

        // Star graph (no Hamiltonian circuit)
        let mut star = create_graph::<i32, ()>();

        star.add_edge(0, 1, ())?;
        star.add_edge(0, 2, ())?;
        star.add_edge(0, 3, ())?;

        assert!(!has_hamiltonian_circuit(&star));

        Ok(())
    }
}
