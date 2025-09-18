//! Base graph structures and operations
//!
//! This module provides the core graph data structures and interfaces
//! for representing and working with graphs.

pub mod graph;
pub mod types;

// Re-export main types
pub use graph::Graph;
pub use types::{Edge, EdgeWeight, Node};

// Re-export petgraph IndexType
pub use petgraph::graph::IndexType;

// Legacy imports for backward compatibility - include remaining graph types here
use ndarray::{Array1, Array2};
use petgraph::graph::{Graph as PetGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Undirected};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};

/// A directed graph structure
pub struct DiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Directed, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// A multigraph structure (allows multiple edges between same nodes)
pub struct MultiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, Vec<E>, Undirected, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// A directed multigraph structure
pub struct MultiDiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, Vec<E>, Directed, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// A bipartite graph structure
pub struct BipartiteGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Undirected, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
    left_nodes: std::collections::HashSet<N>,
    right_nodes: std::collections::HashSet<N>,
}

/// A hypergraph structure
pub struct Hypergraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    nodes: HashMap<N, NodeIndex<Ix>>,
    hyperedges: Vec<Hyperedge<N, E>>,
    _phantom: std::marker::PhantomData<Ix>,
}

/// Represents a hyperedge in a hypergraph
#[derive(Debug, Clone)]
pub struct Hyperedge<N: Node, E: EdgeWeight> {
    /// Nodes connected by this hyperedge
    pub nodes: Vec<N>,
    /// Weight of the hyperedge
    pub weight: E,
    /// Unique identifier for the hyperedge
    pub id: usize,
}

// Simplified implementations for backward compatibility
impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Default for DiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> DiGraph<N, E, Ix> {
    /// Create a new empty directed graph
    pub fn new() -> Self {
        DiGraph {
            graph: PetGraph::default(),
            node_indices: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) -> NodeIndex<Ix> {
        if let Some(idx) = self.node_indices.get(&node) {
            return *idx;
        }

        let idx = self.graph.add_node(node.clone());
        self.node_indices.insert(node, idx);
        idx
    }

    /// Add a directed edge from source to target with a given weight
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        let source_idx = self.add_node(source);
        let target_idx = self.add_node(target);

        self.graph.add_edge(source_idx, target_idx, weight);
        Ok(())
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.node_weights().collect()
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph contains a specific node
    pub fn contains_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Check if the graph has a specific node (alias for contains_node)
    pub fn has_node(&self, node: &N) -> bool {
        self.contains_node(node)
    }

    /// Get the internal petgraph structure for more advanced operations
    pub fn inner(&self) -> &PetGraph<N, E, Directed, Ix> {
        &self.graph
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if let Some(&idx) = self.node_indices.get(node) {
            let neighbors: Vec<N> = self
                .graph
                .neighbors(idx)
                .map(|neighbor_idx| self.graph[neighbor_idx].clone())
                .collect();
            Ok(neighbors)
        } else {
            Err(GraphError::node_not_found("unknown node"))
        }
    }

    /// Get successors (outgoing neighbors) of a node
    pub fn successors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if let Some(&idx) = self.node_indices.get(node) {
            let successors: Vec<N> = self
                .graph
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .map(|neighbor_idx| self.graph[neighbor_idx].clone())
                .collect();
            Ok(successors)
        } else {
            Err(GraphError::node_not_found("unknown node"))
        }
    }

    /// Get predecessors (incoming neighbors) of a node
    pub fn predecessors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if let Some(&idx) = self.node_indices.get(node) {
            let predecessors: Vec<N> = self
                .graph
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .map(|neighbor_idx| self.graph[neighbor_idx].clone())
                .collect();
            Ok(predecessors)
        } else {
            Err(GraphError::node_not_found("unknown node"))
        }
    }

    /// Get the weight of an edge between two nodes
    pub fn edge_weight(&self, source: &N, target: &N) -> Result<E>
    where
        E: Clone,
    {
        if let (Some(&src_idx), Some(&tgt_idx)) =
            (self.node_indices.get(source), self.node_indices.get(target))
        {
            if let Some(edge_ref) = self.graph.find_edge(src_idx, tgt_idx) {
                Ok(self.graph[edge_ref].clone())
            } else {
                Err(GraphError::edge_not_found("unknown", "unknown"))
            }
        } else {
            Err(GraphError::node_not_found("unknown node"))
        }
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        let mut result = Vec::new();
        let node_map: HashMap<NodeIndex<Ix>, &N> = self
            .graph
            .node_indices()
            .map(|idx| (idx, self.graph.node_weight(idx).unwrap()))
            .collect();

        for edge in self.graph.edge_references() {
            let source = node_map[&edge.source()].clone();
            let target = node_map[&edge.target()].clone();
            let weight = edge.weight().clone();

            result.push(Edge {
                source,
                target,
                weight,
            });
        }

        result
    }

    /// Check if an edge exists between two nodes
    pub fn has_edge(&self, source: &N, target: &N) -> bool {
        if let (Some(&src_idx), Some(&tgt_idx)) =
            (self.node_indices.get(source), self.node_indices.get(target))
        {
            self.graph.contains_edge(src_idx, tgt_idx)
        } else {
            false
        }
    }

    /// Get the degree of a node (total number of incident edges)
    pub fn degree(&self, node: &N) -> usize {
        if let Some(idx) = self.node_indices.get(node) {
            self.graph.neighbors(*idx).count()
        } else {
            0
        }
    }

    /// Get the node index for a specific node
    pub fn node_index(&self, node: &N) -> Option<NodeIndex<Ix>> {
        self.node_indices.get(node).copied()
    }

    /// Get the adjacency matrix of the graph
    pub fn adjacency_matrix(&self) -> Array2<f64>
    where
        E: Clone + Into<f64>,
    {
        let n = self.node_count();
        let mut matrix = Array2::zeros((n, n));

        // Create mapping from NodeIndex to matrix index
        let mut node_to_idx = HashMap::new();
        for (i, node_idx) in self.graph.node_indices().enumerate() {
            node_to_idx.insert(node_idx, i);
        }

        // Fill the adjacency matrix
        for edge in self.graph.edge_references() {
            let src_idx = node_to_idx[&edge.source()];
            let tgt_idx = node_to_idx[&edge.target()];
            let weight: f64 = edge.weight().clone().into();
            matrix[[src_idx, tgt_idx]] = weight;
        }

        matrix
    }

    /// Get the out-degree vector (number of outgoing edges for each node)
    pub fn out_degree_vector(&self) -> Array1<usize> {
        let n = self.node_count();
        let mut degrees = Array1::zeros(n);

        // Create mapping from NodeIndex to array index
        let mut node_to_idx = HashMap::new();
        for (i, node_idx) in self.graph.node_indices().enumerate() {
            node_to_idx.insert(node_idx, i);
        }

        // Count out-degrees
        for node_idx in self.graph.node_indices() {
            let out_degree = self
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .count();
            let idx = node_to_idx[&node_idx];
            degrees[idx] = out_degree;
        }

        degrees
    }

    /// Get the in-degree vector (number of incoming edges for each node)
    pub fn in_degree_vector(&self) -> Array1<usize> {
        let n = self.node_count();
        let mut degrees = Array1::zeros(n);

        // Create mapping from NodeIndex to array index
        let mut node_to_idx = HashMap::new();
        for (i, node_idx) in self.graph.node_indices().enumerate() {
            node_to_idx.insert(node_idx, i);
        }

        // Count in-degrees
        for node_idx in self.graph.node_indices() {
            let in_degree = self
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .count();
            let idx = node_to_idx[&node_idx];
            degrees[idx] = in_degree;
        }

        degrees
    }
}

// Placeholder implementations for other graph types to maintain compilation
impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Default for MultiGraph<N, E, Ix> {
    fn default() -> Self {
        MultiGraph {
            graph: PetGraph::default(),
            node_indices: HashMap::new(),
        }
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Default for MultiDiGraph<N, E, Ix> {
    fn default() -> Self {
        MultiDiGraph {
            graph: PetGraph::default(),
            node_indices: HashMap::new(),
        }
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Default for BipartiteGraph<N, E, Ix> {
    fn default() -> Self {
        BipartiteGraph {
            graph: PetGraph::default(),
            node_indices: HashMap::new(),
            left_nodes: std::collections::HashSet::new(),
            right_nodes: std::collections::HashSet::new(),
        }
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Hypergraph<N, E, Ix> {
    /// Create a new empty hypergraph
    pub fn new() -> Self {
        Hypergraph {
            nodes: HashMap::new(),
            hyperedges: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get all nodes in the hypergraph
    pub fn nodes(&self) -> impl Iterator<Item = &N> {
        self.nodes.keys()
    }

    /// Get all hyperedges in the hypergraph
    pub fn hyperedges(&self) -> &Vec<Hyperedge<N, E>> {
        &self.hyperedges
    }

    /// Check if the hypergraph has a node
    pub fn has_node(&self, node: &N) -> bool {
        self.nodes.contains_key(node)
    }

    /// Add a node to the hypergraph
    pub fn add_node(&mut self, node: N) -> NodeIndex<Ix> {
        if let Some(&idx) = self.nodes.get(&node) {
            return idx;
        }

        let idx = NodeIndex::new(self.nodes.len());
        self.nodes.insert(node, idx);
        idx
    }

    /// Add a hyperedge from a vector of nodes
    pub fn add_hyperedge_from_vec(&mut self, nodes: Vec<N>, weight: E) -> Result<usize>
    where
        N: Clone,
    {
        // Add all nodes to the hypergraph
        for node in &nodes {
            self.add_node(node.clone());
        }

        let id = self.hyperedges.len();
        self.hyperedges.push(Hyperedge { nodes, weight, id });
        Ok(id)
    }

    /// Add a hyperedge
    pub fn add_hyperedge(&mut self, nodes: Vec<N>, weight: E) -> Result<usize>
    where
        N: Clone,
    {
        self.add_hyperedge_from_vec(nodes, weight)
    }

    /// Check if two nodes are connected through any hyperedge
    pub fn are_connected(&self, source: &N, target: &N) -> bool
    where
        N: PartialEq,
    {
        for hyperedge in &self.hyperedges {
            if hyperedge.nodes.contains(source) && hyperedge.nodes.contains(target) {
                return true;
            }
        }
        false
    }

    /// Get hyperedges that connect two nodes
    pub fn connecting_hyperedges(&self, source: &N, target: &N) -> Vec<&Hyperedge<N, E>>
    where
        N: PartialEq,
    {
        self.hyperedges
            .iter()
            .filter(|hyperedge| {
                hyperedge.nodes.contains(source) && hyperedge.nodes.contains(target)
            })
            .collect()
    }

    /// Get all hyperedges containing a specific node
    pub fn hyperedges_containing_node(&self, node: &N) -> Vec<&Hyperedge<N, E>>
    where
        N: PartialEq,
    {
        self.hyperedges
            .iter()
            .filter(|hyperedge| hyperedge.nodes.contains(node))
            .collect()
    }

    /// Get neighbors of a node (all nodes connected through hyperedges)
    pub fn neighbors(&self, node: &N) -> Vec<N>
    where
        N: Clone + PartialEq,
    {
        let mut neighbors = std::collections::HashSet::new();

        for hyperedge in &self.hyperedges {
            if hyperedge.nodes.contains(node) {
                for neighbor in &hyperedge.nodes {
                    if neighbor != node {
                        neighbors.insert(neighbor.clone());
                    }
                }
            }
        }

        neighbors.into_iter().collect()
    }

    /// Convert hypergraph to a regular graph (2-section)
    pub fn to_graph(&self) -> Graph<N, E, Ix>
    where
        N: Clone,
        E: Clone + Default,
    {
        let mut graph = Graph::new();

        // Add all nodes
        for node in self.nodes() {
            graph.add_node(node.clone());
        }

        // Add edges between nodes that are in the same hyperedge
        for hyperedge in &self.hyperedges {
            for i in 0..hyperedge.nodes.len() {
                for j in (i + 1)..hyperedge.nodes.len() {
                    let _ = graph.add_edge(
                        hyperedge.nodes[i].clone(),
                        hyperedge.nodes[j].clone(),
                        hyperedge.weight.clone(),
                    );
                }
            }
        }

        graph
    }

    /// Get the number of nodes in the hypergraph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of hyperedges in the hypergraph
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedges.len()
    }

    /// Get the degree of a node (number of hyperedges it belongs to)
    pub fn degree(&self, node: &N) -> usize
    where
        N: PartialEq,
    {
        self.hyperedges
            .iter()
            .filter(|hyperedge| hyperedge.nodes.contains(node))
            .count()
    }

    /// Remove a hyperedge by its ID
    pub fn remove_hyperedge(&mut self, hyperedge_id: usize) -> Result<Hyperedge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        if hyperedge_id < self.hyperedges.len() {
            Ok(self.hyperedges.remove(hyperedge_id))
        } else {
            Err(GraphError::Other("Hyperedge ID out of bounds".to_string()))
        }
    }

    /// Get the incidence matrix of the hypergraph
    /// Returns a matrix where rows represent nodes and columns represent hyperedges
    /// A value of 1.0 indicates the node is in the hyperedge, 0.0 otherwise
    pub fn incidence_matrix(&self) -> Vec<Vec<f64>>
    where
        N: Clone + PartialEq,
    {
        let nodes: Vec<N> = self.nodes().cloned().collect();
        let mut matrix = vec![vec![0.0; self.hyperedges.len()]; nodes.len()];

        for (edge_idx, hyperedge) in self.hyperedges.iter().enumerate() {
            for node in &hyperedge.nodes {
                if let Some(node_idx) = nodes.iter().position(|n| n == node) {
                    matrix[node_idx][edge_idx] = 1.0;
                }
            }
        }

        matrix
    }

    /// Find maximal cliques in the hypergraph
    /// This is a simplified implementation that finds connected components
    /// A more sophisticated algorithm would be needed for true maximal cliques
    pub fn maximal_cliques(&self) -> Vec<Vec<N>>
    where
        N: Clone + PartialEq,
    {
        let mut cliques = Vec::new();
        let mut visited_nodes = std::collections::HashSet::new();

        for hyperedge in &self.hyperedges {
            let mut clique = Vec::new();
            for node in &hyperedge.nodes {
                if !visited_nodes.contains(node) {
                    clique.push(node.clone());
                    visited_nodes.insert(node.clone());
                }
            }
            if !clique.is_empty() {
                cliques.push(clique);
            }
        }

        cliques
    }

    /// Get statistics about hyperedge sizes
    /// Returns (min_size, max_size, avg_size)
    pub fn hyperedge_size_stats(&self) -> (usize, usize, f64) {
        if self.hyperedges.is_empty() {
            return (0, 0, 0.0);
        }

        let sizes: Vec<usize> = self.hyperedges.iter().map(|e| e.nodes.len()).collect();
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let max_size = *sizes.iter().max().unwrap_or(&0);
        let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;

        (min_size, max_size, avg_size)
    }

    /// Check if the hypergraph is uniform (all hyperedges have the same size)
    pub fn is_uniform(&self) -> bool {
        if self.hyperedges.is_empty() {
            return true;
        }

        let first_size = self.hyperedges[0].nodes.len();
        self.hyperedges.iter().all(|e| e.nodes.len() == first_size)
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Default for Hypergraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}
