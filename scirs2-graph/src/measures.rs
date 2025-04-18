//! Graph measures and metrics
//!
//! This module provides functions for measuring various properties
//! of graphs, including centrality measures, clustering coefficients,
//! and other graph metrics.

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};

use crate::algorithms::{shortest_path, shortest_path_digraph};
use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Centrality measure type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentralityType {
    /// Degree centrality: number of connections
    Degree,
    /// Betweenness centrality: number of shortest paths going through a node
    Betweenness,
    /// Closeness centrality: inverse of the sum of shortest paths to all other nodes
    Closeness,
    /// Eigenvector centrality: nodes connected to important nodes are important
    Eigenvector,
}

/// Calculates centrality measures for nodes in an undirected graph
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `centrality_type` - The type of centrality to calculate
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - A map from nodes to their centrality values
pub fn centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    centrality_type: CentralityType,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    match centrality_type {
        CentralityType::Degree => degree_centrality(graph),
        CentralityType::Betweenness => betweenness_centrality(graph),
        CentralityType::Closeness => closeness_centrality(graph),
        CentralityType::Eigenvector => eigenvector_centrality(graph),
    }
}

/// Calculates centrality measures for nodes in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
/// * `centrality_type` - The type of centrality to calculate
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - A map from nodes to their centrality values
pub fn centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    centrality_type: CentralityType,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    match centrality_type {
        CentralityType::Degree => degree_centrality_digraph(graph),
        CentralityType::Betweenness => betweenness_centrality_digraph(graph),
        CentralityType::Closeness => closeness_centrality_digraph(graph),
        CentralityType::Eigenvector => eigenvector_centrality_digraph(graph),
    }
}

/// Calculates degree centrality for nodes in an undirected graph
fn degree_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len() as f64;
    let degrees = graph.degree_vector();

    let mut centrality = HashMap::new();
    let normalization = if n <= 1.0 { 1.0 } else { n - 1.0 };

    for (i, node) in nodes.iter().enumerate() {
        let degree = degrees[i] as f64;
        centrality.insert((*node).clone(), degree / normalization);
    }

    Ok(centrality)
}

/// Calculates degree centrality for nodes in a directed graph
fn degree_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len() as f64;
    let in_degrees = graph.in_degree_vector();
    let out_degrees = graph.out_degree_vector();

    let mut centrality = HashMap::new();
    let normalization = if n <= 1.0 { 1.0 } else { n - 1.0 };

    for (i, node) in nodes.iter().enumerate() {
        // We use the sum of in-degree and out-degree as the total degree
        let degree = (in_degrees[i] + out_degrees[i]) as f64;
        centrality.insert((*node).clone(), degree / normalization);
    }

    Ok(centrality)
}

/// Calculates betweenness centrality for nodes in an undirected graph
fn betweenness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 0.0);
        }
        return Ok(result);
    }

    let mut betweenness = HashMap::new();
    for node in nodes.iter() {
        betweenness.insert((*node).clone(), 0.0);
    }

    // For each pair of nodes, find the shortest paths
    for (i, &s) in nodes.iter().enumerate() {
        for (j, &t) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }

            // Find shortest path from s to t
            if let Ok(Some(path)) = shortest_path(graph, s, t) {
                // Skip source and target nodes
                for node in &path.nodes[1..path.nodes.len() - 1] {
                    *betweenness.entry(node.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    // Normalize by the number of possible paths (excluding the node itself)
    let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
    for val in betweenness.values_mut() {
        *val *= scale;
    }

    Ok(betweenness)
}

/// Calculates betweenness centrality for nodes in a directed graph
fn betweenness_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 0.0);
        }
        return Ok(result);
    }

    let mut betweenness = HashMap::new();
    for node in nodes.iter() {
        betweenness.insert((*node).clone(), 0.0);
    }

    // For each pair of nodes, find the shortest paths
    for (i, &s) in nodes.iter().enumerate() {
        for (j, &t) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }

            // Find shortest path from s to t
            if let Ok(Some(path)) = shortest_path_digraph(graph, s, t) {
                // Skip source and target nodes
                for node in &path.nodes[1..path.nodes.len() - 1] {
                    *betweenness.entry(node.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    // Normalize by the number of possible paths (excluding the node itself)
    let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
    for val in betweenness.values_mut() {
        *val *= scale;
    }

    Ok(betweenness)
}

/// Calculates closeness centrality for nodes in an undirected graph
fn closeness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 1.0);
        }
        return Ok(result);
    }

    let mut closeness = HashMap::new();

    // For each node, calculate the sum of shortest paths to all other nodes
    for &node in nodes.iter() {
        let mut sum_distances = 0.0;
        let mut reachable_nodes = 0;

        for &other in nodes.iter() {
            if node == other {
                continue;
            }

            // Find shortest path from node to other
            if let Ok(Some(path)) = shortest_path(graph, node, other) {
                sum_distances += path.total_weight.into();
                reachable_nodes += 1;
            }
        }

        // If the node can reach other nodes
        if reachable_nodes > 0 {
            // Closeness is 1 / average path length
            let closeness_value = reachable_nodes as f64 / sum_distances;
            // Normalize by the fraction of nodes that are reachable
            let normalized = closeness_value * (reachable_nodes as f64 / (n - 1) as f64);
            closeness.insert(node.clone(), normalized);
        } else {
            closeness.insert(node.clone(), 0.0);
        }
    }

    Ok(closeness)
}

/// Calculates closeness centrality for nodes in a directed graph
fn closeness_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 1.0);
        }
        return Ok(result);
    }

    let mut closeness = HashMap::new();

    // For each node, calculate the sum of shortest paths to all other nodes
    for &node in nodes.iter() {
        let mut sum_distances = 0.0;
        let mut reachable_nodes = 0;

        for &other in nodes.iter() {
            if node == other {
                continue;
            }

            // Find shortest path from node to other
            if let Ok(Some(path)) = shortest_path_digraph(graph, node, other) {
                sum_distances += path.total_weight.into();
                reachable_nodes += 1;
            }
        }

        // If the node can reach other nodes
        if reachable_nodes > 0 {
            // Closeness is 1 / average path length
            let closeness_value = reachable_nodes as f64 / sum_distances;
            // Normalize by the fraction of nodes that are reachable
            let normalized = closeness_value * (reachable_nodes as f64 / (n - 1) as f64);
            closeness.insert(node.clone(), normalized);
        } else {
            closeness.insert(node.clone(), 0.0);
        }
    }

    Ok(closeness)
}

/// Calculates eigenvector centrality for nodes in an undirected graph
///
/// Uses the power iteration method.
fn eigenvector_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node,
    E: EdgeWeight + num_traits::Zero + num_traits::One + PartialOrd + Into<f64> + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix with weights converted to f64
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Start with uniform vector
    let mut x = Array1::<f64>::ones(n);
    x.mapv_inplace(|v| v / (n as f64).sqrt()); // Normalize

    // Power iteration
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // y = A * x
        let y = adj_f64.dot(&x);

        // Compute the norm
        let norm = (y.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-10 {
            // If the norm is too small, we have a zero vector
            return Err(GraphError::AlgorithmError(
                "Eigenvector centrality computation failed: zero eigenvalue".to_string(),
            ));
        }

        // Normalize y
        let y_norm = y / norm;

        // Check for convergence
        let diff = (&y_norm - &x).iter().map(|v| v.abs()).sum::<f64>();
        if diff < tol {
            // We've converged, create the result
            let mut result = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                result.insert(node.clone(), y_norm[i]);
            }
            return Ok(result);
        }

        // Update x for next iteration
        x = y_norm;
    }

    // We've reached max iterations without converging
    Err(GraphError::AlgorithmError(
        "Eigenvector centrality computation did not converge".to_string(),
    ))
}

/// Calculates eigenvector centrality for nodes in a directed graph
///
/// Uses the power iteration method.
fn eigenvector_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node,
    E: EdgeWeight + num_traits::Zero + num_traits::One + PartialOrd + Into<f64> + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix with weights converted to f64
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Start with uniform vector
    let mut x = Array1::<f64>::ones(n);
    x.mapv_inplace(|v| v / (n as f64).sqrt()); // Normalize

    // Power iteration
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // y = A * x
        let y = adj_f64.dot(&x);

        // Compute the norm
        let norm = (y.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-10 {
            // If the norm is too small, we have a zero vector
            return Err(GraphError::AlgorithmError(
                "Eigenvector centrality computation failed: zero eigenvalue".to_string(),
            ));
        }

        // Normalize y
        let y_norm = y / norm;

        // Check for convergence
        let diff = (&y_norm - &x).iter().map(|v| v.abs()).sum::<f64>();
        if diff < tol {
            // We've converged, create the result
            let mut result = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                result.insert(node.clone(), y_norm[i]);
            }
            return Ok(result);
        }

        // Update x for next iteration
        x = y_norm;
    }

    // We've reached max iterations without converging
    Err(GraphError::AlgorithmError(
        "Eigenvector centrality computation did not converge".to_string(),
    ))
}

/// Calculates the local clustering coefficient for each node in an undirected graph
///
/// The clustering coefficient measures how close a node's neighbors are to being a clique.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - A map from nodes to their clustering coefficients
pub fn clustering_coefficient<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut coefficients = HashMap::new();

    for (idx, &node) in graph.nodes().iter().enumerate() {
        // Get the neighbors of this node
        let node_idx = petgraph::graph::NodeIndex::new(idx);
        let neighbors: HashSet<_> = graph.inner().neighbors(node_idx).collect();

        let k = neighbors.len();

        // If the node has less than 2 neighbors, its clustering coefficient is 0
        if k < 2 {
            coefficients.insert(node.clone(), 0.0);
            continue;
        }

        // Count the number of edges between neighbors
        let mut edge_count = 0;

        for &n1 in &neighbors {
            for &n2 in &neighbors {
                if n1 != n2 && graph.inner().contains_edge(n1, n2) {
                    edge_count += 1;
                }
            }
        }

        // Each edge is counted twice (once from each direction)
        edge_count /= 2;

        // Calculate the clustering coefficient
        let possible_edges = k * (k - 1) / 2;
        let coefficient = edge_count as f64 / possible_edges as f64;

        coefficients.insert(node.clone(), coefficient);
    }

    Ok(coefficients)
}

/// Calculates the global clustering coefficient (transitivity) of a graph
///
/// This is the ratio of the number of closed triplets to the total number of triplets.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * `Result<f64>` - The global clustering coefficient
pub fn graph_density<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<f64>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n <= 1 {
        return Err(GraphError::InvalidGraph(
            "Graph density undefined for graphs with 0 or 1 nodes".to_string(),
        ));
    }

    let m = graph.edge_count();
    let possible_edges = n * (n - 1) / 2;

    Ok(m as f64 / possible_edges as f64)
}

/// Calculates the density of a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
///
/// # Returns
/// * `Result<f64>` - The graph density
pub fn graph_density_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<f64>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n <= 1 {
        return Err(GraphError::InvalidGraph(
            "Graph density undefined for graphs with 0 or 1 nodes".to_string(),
        ));
    }

    let m = graph.edge_count();
    let possible_edges = n * (n - 1); // In directed graphs, there can be n(n-1) edges

    Ok(m as f64 / possible_edges as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_centrality() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a star graph with A in the center
        // A -- B
        // |
        // |
        // C -- D

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();
        graph.add_edge('C', 'D', 1.0).unwrap();

        let centrality = centrality(&graph, CentralityType::Degree).unwrap();

        // A has 2 connections out of 3 possible, so centrality = 2/3
        // B has 1 connection out of 3 possible, so centrality = 1/3
        // C has 2 connections out of 3 possible, so centrality = 2/3
        // D has 1 connection out of 3 possible, so centrality = 1/3

        assert_eq!(centrality[&'A'], 2.0 / 3.0);
        assert_eq!(centrality[&'B'], 1.0 / 3.0);
        assert_eq!(centrality[&'C'], 2.0 / 3.0);
        assert_eq!(centrality[&'D'], 1.0 / 3.0);
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a graph:
        // 1 -- 2 -- 3
        // |         |
        // +----4----+
        //      |
        //      5

        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();
        graph.add_edge(4, 1, 1.0).unwrap();
        graph.add_edge(4, 5, 1.0).unwrap();

        let coefficients = clustering_coefficient(&graph).unwrap();

        // Node 1 has neighbors 2 and 4, and they're not connected, so coefficient = 0/1 = 0
        assert_eq!(coefficients[&1], 0.0);

        // Node 2 has neighbors 1 and 3, and they're not connected, so coefficient = 0/1 = 0
        assert_eq!(coefficients[&2], 0.0);

        // Node 3 has neighbors 2 and 4, and they're not connected, so coefficient = 0/1 = 0
        assert_eq!(coefficients[&3], 0.0);

        // Node 4 has neighbors 1, 3, and 5.
        // Neighbors 1 and 3 are not directly connected, and 5 is not connected to either 1 or 3
        // So coefficient = 0/3 = 0
        assert_eq!(coefficients[&4], 0.0);

        // Node 5 has only one neighbor (4), so coefficient = 0
        assert_eq!(coefficients[&5], 0.0);

        // Now let's add an edge to create a triangle
        graph.add_edge(1, 3, 1.0).unwrap();

        let coefficients = clustering_coefficient(&graph).unwrap();

        // Now nodes 1, 3, and 4 form a triangle
        // Node 4 has 3 neighbors (1, 3, 5) with 1 edge between them (1-3)
        // Possible edges: 3 choose 2 = 3
        // So coefficient = 1/3
        assert!((coefficients[&4] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_graph_density() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Empty graph should error
        assert!(graph_density(&graph).is_err());

        // Add one node
        graph.add_node(1);

        // Graph with one node should error
        assert!(graph_density(&graph).is_err());

        // Create a graph with 4 nodes and 3 edges
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);

        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();

        // 3 edges out of 6 possible edges (4 choose 2 = 6)
        let density = graph_density(&graph).unwrap();
        assert_eq!(density, 0.5);

        // Add 3 more edges to make a complete graph
        graph.add_edge(1, 3, 1.0).unwrap();
        graph.add_edge(1, 4, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        // 6 edges out of 6 possible edges
        let density = graph_density(&graph).unwrap();
        assert_eq!(density, 1.0);
    }
}
