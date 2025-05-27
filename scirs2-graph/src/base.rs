//! Base graph structures and operations
//!
//! This module provides the core graph data structures and interfaces
//! for representing and working with graphs.

use ndarray::{Array1, Array2};
pub use petgraph::graph::IndexType;
use petgraph::graph::{Graph as PetGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Undirected};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};

/// A trait representing a node in a graph
pub trait Node: Clone + Eq + Hash + Send + Sync {}

/// Implements Node for common types
impl<T: Clone + Eq + Hash + Send + Sync> Node for T {}

/// A trait for edge weights in a graph
pub trait EdgeWeight: Clone + PartialOrd + Send + Sync {}

/// Implements EdgeWeight for common types
impl<T: Clone + PartialOrd + Send + Sync> EdgeWeight for T {}

/// An undirected graph structure
pub struct Graph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Undirected, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// A directed graph structure
pub struct DiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Directed, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

/// Represents an edge in a graph
#[derive(Debug, Clone)]
pub struct Edge<N: Node, E: EdgeWeight> {
    /// Source node
    pub source: N,
    /// Target node
    pub target: N,
    /// Edge weight
    pub weight: E,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for Graph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Graph<N, E, Ix> {
    /// Create a new empty undirected graph
    pub fn new() -> Self {
        Graph {
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

    /// Add an edge between two nodes with a given weight
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        let source_idx = self.add_node(source);
        let target_idx = self.add_node(target);

        self.graph.add_edge(source_idx, target_idx, weight);
        Ok(())
    }

    /// Get the adjacency matrix representation of the graph
    pub fn adjacency_matrix(&self) -> Array2<E>
    where
        E: num_traits::Zero + num_traits::One + Copy,
    {
        let n = self.graph.node_count();
        let mut adj_mat = Array2::zeros((n, n));

        for edge in self.graph.edge_references() {
            let (src, tgt) = (edge.source().index(), edge.target().index());
            adj_mat[[src, tgt]] = *edge.weight();
            adj_mat[[tgt, src]] = *edge.weight(); // Undirected graph
        }

        adj_mat
    }

    /// Get the degree vector of the graph
    pub fn degree_vector(&self) -> Array1<usize> {
        let n = self.graph.node_count();
        let mut degrees = Array1::zeros(n);

        for (idx, node) in self.graph.node_indices().enumerate() {
            degrees[idx] = self.graph.neighbors(node).count();
        }

        degrees
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.node_weights().collect()
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

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph has a node
    pub fn has_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
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
            Err(GraphError::NodeNotFound)
        }
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
                Err(GraphError::EdgeNotFound)
            }
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Check if the graph contains a specific node
    pub fn contains_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get the internal petgraph structure for more advanced operations
    pub fn inner(&self) -> &PetGraph<N, E, Undirected, Ix> {
        &self.graph
    }

    /// Get a mutable reference to the internal petgraph structure
    pub fn inner_mut(&mut self) -> &mut PetGraph<N, E, Undirected, Ix> {
        &mut self.graph
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for DiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> DiGraph<N, E, Ix> {
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

    /// Get the adjacency matrix representation of the graph
    pub fn adjacency_matrix(&self) -> Array2<E>
    where
        E: num_traits::Zero + num_traits::One + Copy,
    {
        let n = self.graph.node_count();
        let mut adj_mat = Array2::zeros((n, n));

        for edge in self.graph.edge_references() {
            let (src, tgt) = (edge.source().index(), edge.target().index());
            adj_mat[[src, tgt]] = *edge.weight();
        }

        adj_mat
    }

    /// Get the in-degree vector of the graph
    pub fn in_degree_vector(&self) -> Array1<usize> {
        let n = self.graph.node_count();
        let mut degrees = Array1::zeros(n);

        for (idx, node) in self.graph.node_indices().enumerate() {
            degrees[idx] = self
                .graph
                .neighbors_directed(node, petgraph::Direction::Incoming)
                .count();
        }

        degrees
    }

    /// Get the out-degree vector of the graph
    pub fn out_degree_vector(&self) -> Array1<usize> {
        let n = self.graph.node_count();
        let mut degrees = Array1::zeros(n);

        for (idx, node) in self.graph.node_indices().enumerate() {
            degrees[idx] = self
                .graph
                .neighbors_directed(node, petgraph::Direction::Outgoing)
                .count();
        }

        degrees
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.node_weights().collect()
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

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph has a node
    pub fn has_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
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
            Err(GraphError::NodeNotFound)
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
            Err(GraphError::NodeNotFound)
        }
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
                Err(GraphError::EdgeNotFound)
            }
        } else {
            Err(GraphError::NodeNotFound)
        }
    }

    /// Check if the graph contains a specific node
    pub fn contains_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get the internal petgraph structure for more advanced operations
    pub fn inner(&self) -> &PetGraph<N, E, Directed, Ix> {
        &self.graph
    }

    /// Get a mutable reference to the internal petgraph structure
    pub fn inner_mut(&mut self) -> &mut PetGraph<N, E, Directed, Ix> {
        &mut self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undirected_graph_creation() {
        let mut graph: Graph<&str, f64> = Graph::new();

        graph.add_node("A");
        graph.add_node("B");
        graph.add_node("C");

        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_node(&"A"));
        assert!(graph.has_node(&"B"));
        assert!(graph.has_node(&"C"));
    }

    #[test]
    fn test_directed_graph_creation() {
        let mut graph: DiGraph<i32, f64> = DiGraph::new();

        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);

        graph.add_edge(1, 2, 1.5).unwrap();
        graph.add_edge(2, 3, 2.5).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_node(&1));
        assert!(graph.has_node(&2));
        assert!(graph.has_node(&3));
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut graph: Graph<u8, f64> = Graph::new();

        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);

        graph.add_edge(0, 1, 1.0).unwrap();
        graph.add_edge(1, 2, 2.0).unwrap();

        let adj_mat = graph.adjacency_matrix();

        // Expected matrix:
        // [[0.0, 1.0, 0.0],
        //  [1.0, 0.0, 2.0],
        //  [0.0, 2.0, 0.0]]

        assert_eq!(adj_mat.shape(), &[3, 3]);
        assert_eq!(adj_mat[[0, 1]], 1.0);
        assert_eq!(adj_mat[[1, 0]], 1.0);
        assert_eq!(adj_mat[[1, 2]], 2.0);
        assert_eq!(adj_mat[[2, 1]], 2.0);
        assert_eq!(adj_mat[[0, 2]], 0.0);
        assert_eq!(adj_mat[[2, 0]], 0.0);
    }

    #[test]
    fn test_degree_vector() {
        let mut graph: Graph<char, f64> = Graph::new();

        graph.add_node('A');
        graph.add_node('B');
        graph.add_node('C');
        graph.add_node('D');

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();
        graph.add_edge('C', 'D', 1.0).unwrap();
        graph.add_edge('D', 'A', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();

        let degrees = graph.degree_vector();

        // A connects to B, D, C = 3
        // B connects to A, C = 2
        // C connects to B, D, A = 3
        // D connects to C, A = 2

        assert_eq!(degrees, Array1::from_vec(vec![3, 2, 3, 2]));
    }
}
