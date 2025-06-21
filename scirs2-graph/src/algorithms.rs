//! Graph algorithms
//!
//! This module provides various algorithms for graph analysis and manipulation.
//! The algorithms are organized into submodules by category:
//!
//! - `traversal`: BFS, DFS, and other traversal algorithms
//! - `shortest_path`: Dijkstra, A*, Floyd-Warshall, etc.
//! - `connectivity`: Connected components, articulation points, bridges
//! - `flow`: Network flow and cut algorithms
//! - `matching`: Bipartite matching algorithms
//! - `coloring`: Graph coloring algorithms
//! - `paths`: Eulerian and Hamiltonian path algorithms
//! - `community`: Community detection algorithms
//! - `decomposition`: Graph decomposition algorithms
//! - `isomorphism`: Graph isomorphism and subgraph matching
//! - `motifs`: Motif finding algorithms
//! - `random_walk`: Random walk and PageRank algorithms
//! - `similarity`: Node and graph similarity measures
//! - `properties`: Graph properties like diameter, radius, center

pub mod coloring;
pub mod community;
pub mod connectivity;
pub mod decomposition;
pub mod flow;
pub mod hypergraph;
pub mod isomorphism;
pub mod matching;
pub mod motifs;
pub mod paths;
pub mod properties;
pub mod random_walk;
pub mod shortest_path;
pub mod similarity;
pub mod transformations;
pub mod traversal;

// Re-export all public items for convenience
pub use coloring::*;
pub use community::{
    fluid_communities, greedy_modularity_optimization, hierarchical_communities,
    infomap_communities, label_propagation, louvain_communities, modularity,
    modularity_optimization, CommunityStructure, InfomapResult,
};
pub use connectivity::*;
pub use decomposition::*;
pub use flow::{dinic_max_flow, minimum_cut, push_relabel_max_flow};
pub use hypergraph::*;
pub use isomorphism::*;
pub use matching::*;
pub use motifs::*;
pub use paths::*;
pub use properties::*;
pub use random_walk::*;
pub use shortest_path::*;
pub use similarity::*;
pub use transformations::*;
pub use traversal::*;

// Additional algorithms that haven't been moved to submodules yet

use crate::base::{DiGraph, EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use ndarray::{Array1, Array2};
use petgraph::algo::toposort as petgraph_toposort;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

/// Kruskal's algorithm for finding minimum spanning tree
///
/// Returns a vector of edges that form the minimum spanning tree.
/// Only works on undirected graphs.
pub fn minimum_spanning_tree<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
) -> Result<Vec<crate::base::Edge<N, E>>>
where
    N: Node,
    E: EdgeWeight + Into<f64> + std::cmp::PartialOrd,
    Ix: petgraph::graph::IndexType,
{
    // Get all edges and sort by weight
    let mut edges: Vec<_> = graph
        .inner()
        .edge_references()
        .map(|e| {
            let source = graph.inner()[e.source()].clone();
            let target = graph.inner()[e.target()].clone();
            let weight = e.weight().clone();
            (source, target, weight)
        })
        .collect();

    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Use Union-Find to detect cycles
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let mut parent: HashMap<N, N> = nodes.iter().map(|n| (n.clone(), n.clone())).collect();
    let mut rank: HashMap<N, usize> = nodes.iter().map(|n| (n.clone(), 0)).collect();

    fn find<N: Node>(parent: &mut HashMap<N, N>, node: &N) -> N {
        if parent[node] != *node {
            let root = find(parent, &parent[node].clone());
            parent.insert(node.clone(), root.clone());
        }
        parent[node].clone()
    }

    fn union<N: Node>(
        parent: &mut HashMap<N, N>,
        rank: &mut HashMap<N, usize>,
        x: &N,
        y: &N,
    ) -> bool {
        let root_x = find(parent, x);
        let root_y = find(parent, y);

        if root_x == root_y {
            return false; // Already in same set
        }

        // Union by rank
        match rank[&root_x].cmp(&rank[&root_y]) {
            Ordering::Less => {
                parent.insert(root_x, root_y);
            }
            Ordering::Greater => {
                parent.insert(root_y, root_x);
            }
            Ordering::Equal => {
                parent.insert(root_y, root_x.clone());
                *rank.get_mut(&root_x).unwrap() += 1;
            }
        }
        true
    }

    let mut mst = Vec::new();

    for (source, target, weight) in edges {
        if union(&mut parent, &mut rank, &source, &target) {
            mst.push(crate::base::Edge {
                source,
                target,
                weight,
            });
        }
    }

    Ok(mst)
}

/// Topological sort for directed acyclic graphs
///
/// Returns nodes in topological order if the graph is a DAG,
/// otherwise returns an error indicating a cycle was found.
pub fn topological_sort<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<Vec<N>>
where
    N: Node,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Use petgraph's topological sort
    match petgraph_toposort(graph.inner(), None) {
        Ok(indices) => {
            let sorted_nodes = indices
                .into_iter()
                .map(|idx| graph.inner()[idx].clone())
                .collect();
            Ok(sorted_nodes)
        }
        Err(_) => Err(GraphError::CycleDetected),
    }
}

/// PageRank algorithm for computing node importance
///
/// Returns a map from nodes to their PageRank scores.
pub fn pagerank<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    damping_factor: f64,
    tolerance: f64,
    max_iterations: usize,
) -> HashMap<N, f64>
where
    N: Node,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<_> = graph.inner().node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return HashMap::new();
    }

    // Initialize PageRank values
    let mut pr = vec![1.0 / n as f64; n];
    let mut new_pr = vec![0.0; n];

    // Create node index mapping
    let node_to_idx: HashMap<_, _> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    // Iterate until convergence
    for _ in 0..max_iterations {
        // Reset new PageRank values
        for pr in new_pr.iter_mut().take(n) {
            *pr = (1.0 - damping_factor) / n as f64;
        }

        // Calculate contributions from incoming edges
        for (i, &node_idx) in nodes.iter().enumerate() {
            let out_degree = graph
                .inner()
                .edges_directed(node_idx, Direction::Outgoing)
                .count();

            if out_degree > 0 {
                let contribution = damping_factor * pr[i] / out_degree as f64;

                for edge in graph.inner().edges_directed(node_idx, Direction::Outgoing) {
                    if let Some(&j) = node_to_idx.get(&edge.target()) {
                        new_pr[j] += contribution;
                    }
                }
            } else {
                // Dangling node: distribute equally to all nodes
                let contribution = damping_factor * pr[i] / n as f64;
                for pr_val in new_pr.iter_mut().take(n) {
                    *pr_val += contribution;
                }
            }
        }

        // Check convergence
        let diff: f64 = pr
            .iter()
            .zip(&new_pr)
            .map(|(old, new)| (old - new).abs())
            .sum();

        // Swap vectors
        std::mem::swap(&mut pr, &mut new_pr);

        if diff < tolerance {
            break;
        }
    }

    // Convert to HashMap
    nodes
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| (graph.inner()[node_idx].clone(), pr[i]))
        .collect()
}

/// Betweenness centrality for nodes
///
/// Measures the extent to which a node lies on paths between other nodes.
pub fn betweenness_centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    normalized: bool,
) -> HashMap<N, f64>
where
    N: Node,
    E: EdgeWeight,
    Ix: IndexType,
{
    let node_indices: Vec<_> = graph.inner().node_indices().collect();
    let nodes: Vec<N> = node_indices
        .iter()
        .map(|&idx| graph.inner()[idx].clone())
        .collect();
    let n = nodes.len();
    let mut centrality: HashMap<N, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

    // For each source node
    for s in &nodes {
        // Single-source shortest paths
        let mut stack = Vec::new();
        let mut paths: HashMap<N, Vec<N>> = HashMap::new();
        let mut sigma: HashMap<N, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();
        let mut dist: HashMap<N, Option<f64>> = nodes.iter().map(|n| (n.clone(), None)).collect();

        sigma.insert(s.clone(), 1.0);
        dist.insert(s.clone(), Some(0.0));

        let mut queue = VecDeque::new();
        queue.push_back(s.clone());

        // BFS
        while let Some(v) = queue.pop_front() {
            stack.push(v.clone());

            if let Ok(neighbors) = graph.neighbors(&v) {
                for w in neighbors {
                    // First time we reach w?
                    if dist[&w].is_none() {
                        dist.insert(w.clone(), Some(dist[&v].unwrap() + 1.0));
                        queue.push_back(w.clone());
                    }

                    // Shortest path to w via v?
                    if dist[&w] == Some(dist[&v].unwrap() + 1.0) {
                        *sigma.get_mut(&w).unwrap() += sigma[&v];
                        paths.entry(w.clone()).or_default().push(v.clone());
                    }
                }
            }
        }

        // Accumulation
        let mut delta: HashMap<N, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

        while let Some(w) = stack.pop() {
            if let Some(predecessors) = paths.get(&w) {
                for v in predecessors {
                    *delta.get_mut(v).unwrap() += (sigma[v] / sigma[&w]) * (1.0 + delta[&w]);
                }
            }

            if w != *s {
                *centrality.get_mut(&w).unwrap() += delta[&w];
            }
        }
    }

    // Normalization
    if normalized && n > 2 {
        let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
        for value in centrality.values_mut() {
            *value *= scale;
        }
    }

    centrality
}

/// Closeness centrality for nodes
///
/// Measures how close a node is to all other nodes in the graph.
pub fn closeness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>, normalized: bool) -> HashMap<N, f64>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: IndexType,
{
    let node_indices: Vec<_> = graph.inner().node_indices().collect();
    let nodes: Vec<N> = node_indices
        .iter()
        .map(|&idx| graph.inner()[idx].clone())
        .collect();
    let n = nodes.len();
    let mut centrality = HashMap::new();

    for node in &nodes {
        let mut total_distance = 0.0;
        let mut reachable_count = 0;

        // Calculate shortest paths to all other nodes
        for other in &nodes {
            if node != other {
                if let Ok(Some(path)) = shortest_path(graph, node, other) {
                    let distance: f64 = path.total_weight.into();
                    total_distance += distance;
                    reachable_count += 1;
                }
            }
        }

        if reachable_count > 0 {
            let closeness = reachable_count as f64 / total_distance;
            let value = if normalized && n > 1 {
                closeness * (reachable_count as f64 / (n - 1) as f64)
            } else {
                closeness
            };
            centrality.insert(node.clone(), value);
        } else {
            centrality.insert(node.clone(), 0.0);
        }
    }

    centrality
}

/// Eigenvector centrality
///
/// Computes the eigenvector centrality of nodes using power iteration.
pub fn eigenvector_centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iter: usize,
    tolerance: f64,
) -> Result<HashMap<N, f64>>
where
    N: Node,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let node_indices: Vec<_> = graph.inner().node_indices().collect();
    let nodes: Vec<N> = node_indices
        .iter()
        .map(|&idx| graph.inner()[idx].clone())
        .collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(HashMap::new());
    }

    // Create adjacency matrix
    let mut adj_matrix = Array2::<f64>::zeros((n, n));
    for (i, node_i) in nodes.iter().enumerate() {
        for (j, node_j) in nodes.iter().enumerate() {
            if let Ok(weight) = graph.edge_weight(node_i, node_j) {
                adj_matrix[[i, j]] = weight.into();
            }
        }
    }

    // Initialize eigenvector
    let mut x = Array1::<f64>::from_elem(n, 1.0 / (n as f64).sqrt());
    let mut converged = false;

    // Power iteration
    for _ in 0..max_iter {
        let x_new = adj_matrix.dot(&x);

        // Normalize
        let norm = x_new.dot(&x_new).sqrt();
        if norm == 0.0 {
            return Err(GraphError::ComputationError(
                "Eigenvector computation failed".to_string(),
            ));
        }

        let x_normalized = x_new / norm;

        // Check convergence
        let diff = (&x_normalized - &x).mapv(f64::abs).sum();
        if diff < tolerance {
            converged = true;
            x = x_normalized;
            break;
        }

        x = x_normalized;
    }

    if !converged {
        return Err(GraphError::ComputationError(
            "Eigenvector centrality did not converge".to_string(),
        ));
    }

    // Convert to HashMap
    Ok(nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node, x[i]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::create_graph;

    #[test]
    fn test_minimum_spanning_tree() {
        let mut graph = create_graph::<&str, f64>();
        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();
        graph.add_edge("A", "C", 3.0).unwrap();
        graph.add_edge("C", "D", 1.0).unwrap();

        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have n-1 edges
        assert_eq!(mst.len(), 3);

        // Total weight should be 4.0 (AB: 1, BC: 2, CD: 1)
        let total_weight: f64 = mst.iter().map(|e| e.weight).sum();
        assert_eq!(total_weight, 4.0);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = crate::generators::create_digraph::<&str, ()>();
        graph.add_edge("A", "B", ()).unwrap();
        graph.add_edge("A", "C", ()).unwrap();
        graph.add_edge("B", "D", ()).unwrap();
        graph.add_edge("C", "D", ()).unwrap();

        let sorted = topological_sort(&graph).unwrap();

        // A should come before B and C
        let a_pos = sorted.iter().position(|n| n == &"A").unwrap();
        let b_pos = sorted.iter().position(|n| n == &"B").unwrap();
        let c_pos = sorted.iter().position(|n| n == &"C").unwrap();
        let d_pos = sorted.iter().position(|n| n == &"D").unwrap();

        assert!(a_pos < b_pos);
        assert!(a_pos < c_pos);
        assert!(b_pos < d_pos);
        assert!(c_pos < d_pos);
    }

    #[test]
    fn test_pagerank() {
        let mut graph = crate::generators::create_digraph::<&str, ()>();
        graph.add_edge("A", "B", ()).unwrap();
        graph.add_edge("A", "C", ()).unwrap();
        graph.add_edge("B", "C", ()).unwrap();
        graph.add_edge("C", "A", ()).unwrap();

        let pr = pagerank(&graph, 0.85, 1e-6, 100);

        // All nodes should have positive PageRank
        assert!(pr.values().all(|&v| v > 0.0));

        // Sum should be approximately 1.0
        let sum: f64 = pr.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
