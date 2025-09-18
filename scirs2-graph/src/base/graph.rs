//! Basic undirected graph implementation

use ndarray::{Array1, Array2};
use petgraph::graph::{Graph as PetGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Undirected;
use std::collections::HashMap;

use super::types::{Edge, EdgeWeight, Node};
use crate::error::{GraphError, Result};
pub use petgraph::graph::IndexType;

/// An undirected graph structure
pub struct Graph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    graph: PetGraph<N, E, Undirected, Ix>,
    node_indices: HashMap<N, NodeIndex<Ix>>,
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Default for Graph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: IndexType> Graph<N, E, Ix> {
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
            Err(GraphError::node_not_found_with_context(
                format!("{node:?}"),
                self.node_count(),
                "neighbors",
            ))
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
                Err(GraphError::edge_not_found("unknown", "unknown"))
            }
        } else {
            Err(GraphError::node_not_found("unknown node"))
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

    /// Check if the graph contains a specific node
    pub fn contains_node(&self, node: &N) -> bool {
        self.node_indices.contains_key(node)
    }

    /// Get the node index for a specific node
    pub fn node_index(&self, node: &N) -> Option<NodeIndex<Ix>> {
        self.node_indices.get(node).copied()
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
