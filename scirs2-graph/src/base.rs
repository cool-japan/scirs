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

/// A multi-graph structure that supports parallel edges
///
/// Unlike Graph, MultiGraph allows multiple edges between the same pair of nodes.
/// This is useful for modeling scenarios where multiple connections of different types
/// or weights can exist between nodes.
pub struct MultiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// Adjacency list representation: node -> list of (neighbor, edge_weight, edge_id)
    adjacency: HashMap<N, Vec<(N, E, usize)>>,
    /// All nodes in the graph
    nodes: std::collections::HashSet<N>,
    /// Edge counter for unique edge IDs
    edge_id_counter: usize,
    /// All edges in the graph with their IDs
    edges: HashMap<usize, Edge<N, E>>,
    /// Phantom data for index type
    _phantom: std::marker::PhantomData<Ix>,
}

/// A directed multi-graph structure that supports parallel edges
pub struct MultiDiGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// Outgoing adjacency list: node -> list of (target, edge_weight, edge_id)
    out_adjacency: HashMap<N, Vec<(N, E, usize)>>,
    /// Incoming adjacency list: node -> list of (source, edge_weight, edge_id)
    in_adjacency: HashMap<N, Vec<(N, E, usize)>>,
    /// All nodes in the graph
    nodes: std::collections::HashSet<N>,
    /// Edge counter for unique edge IDs
    edge_id_counter: usize,
    /// All edges in the graph with their IDs
    edges: HashMap<usize, Edge<N, E>>,
    /// Phantom data for index type
    _phantom: std::marker::PhantomData<Ix>,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for MultiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> MultiGraph<N, E, Ix> {
    /// Create a new empty multi-graph
    pub fn new() -> Self {
        MultiGraph {
            adjacency: HashMap::new(),
            nodes: std::collections::HashSet::new(),
            edge_id_counter: 0,
            edges: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) {
        if !self.nodes.contains(&node) {
            self.nodes.insert(node.clone());
            self.adjacency.insert(node, Vec::new());
        }
    }

    /// Add an edge between two nodes with a given weight
    /// Returns the edge ID for reference
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> usize
    where
        N: Clone,
        E: Clone,
    {
        // Ensure both nodes exist
        self.add_node(source.clone());
        self.add_node(target.clone());

        let edge_id = self.edge_id_counter;
        self.edge_id_counter += 1;

        // Add to adjacency lists (undirected, so add both directions)
        self.adjacency
            .get_mut(&source)
            .unwrap()
            .push((target.clone(), weight.clone(), edge_id));

        if source != target {
            self.adjacency.get_mut(&target).unwrap().push((
                source.clone(),
                weight.clone(),
                edge_id,
            ));
        }

        // Store edge information
        self.edges.insert(
            edge_id,
            Edge {
                source,
                target,
                weight,
            },
        );

        edge_id
    }

    /// Get all parallel edges between two nodes
    pub fn get_edges_between(&self, source: &N, target: &N) -> Vec<(usize, &E)>
    where
        E: Clone,
    {
        let mut result = Vec::new();

        if let Some(neighbors) = self.adjacency.get(source) {
            for (neighbor, weight, edge_id) in neighbors {
                if neighbor == target {
                    result.push((*edge_id, weight));
                }
            }
        }

        result
    }

    /// Remove an edge by its ID
    pub fn remove_edge(&mut self, edge_id: usize) -> Result<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        if let Some(edge) = self.edges.remove(&edge_id) {
            // Remove from adjacency lists
            if let Some(neighbors) = self.adjacency.get_mut(&edge.source) {
                neighbors.retain(|(_, _, id)| *id != edge_id);
            }

            if edge.source != edge.target {
                if let Some(neighbors) = self.adjacency.get_mut(&edge.target) {
                    neighbors.retain(|(_, _, id)| *id != edge_id);
                }
            }

            Ok(edge)
        } else {
            Err(GraphError::EdgeNotFound)
        }
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> std::collections::hash_set::Iter<'_, N> {
        self.nodes.iter()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> std::collections::hash_map::Values<'_, usize, Edge<N, E>> {
        self.edges.values()
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get neighbors of a node with edge weights and IDs
    pub fn neighbors_with_edges(&self, node: &N) -> Option<&Vec<(N, E, usize)>> {
        self.adjacency.get(node)
    }

    /// Get simple neighbors (without edge information)
    pub fn neighbors(&self, node: &N) -> Vec<&N> {
        if let Some(neighbors) = self.adjacency.get(node) {
            neighbors.iter().map(|(neighbor, _, _)| neighbor).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if the graph contains a node
    pub fn has_node(&self, node: &N) -> bool {
        self.nodes.contains(node)
    }

    /// Get the degree of a node (total number of incident edges)
    pub fn degree(&self, node: &N) -> usize {
        self.adjacency
            .get(node)
            .map_or(0, |neighbors| neighbors.len())
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for MultiDiGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> MultiDiGraph<N, E, Ix> {
    /// Create a new empty directed multi-graph
    pub fn new() -> Self {
        MultiDiGraph {
            out_adjacency: HashMap::new(),
            in_adjacency: HashMap::new(),
            nodes: std::collections::HashSet::new(),
            edge_id_counter: 0,
            edges: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: N) {
        if !self.nodes.contains(&node) {
            self.nodes.insert(node.clone());
            self.out_adjacency.insert(node.clone(), Vec::new());
            self.in_adjacency.insert(node, Vec::new());
        }
    }

    /// Add an edge from source to target with given weight
    /// Returns the edge ID for reference
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> usize
    where
        N: Clone,
        E: Clone,
    {
        // Ensure both nodes exist
        self.add_node(source.clone());
        self.add_node(target.clone());

        let edge_id = self.edge_id_counter;
        self.edge_id_counter += 1;

        // Add to outgoing adjacency list
        self.out_adjacency.get_mut(&source).unwrap().push((
            target.clone(),
            weight.clone(),
            edge_id,
        ));

        // Add to incoming adjacency list
        self.in_adjacency
            .get_mut(&target)
            .unwrap()
            .push((source.clone(), weight.clone(), edge_id));

        // Store edge information
        self.edges.insert(
            edge_id,
            Edge {
                source,
                target,
                weight,
            },
        );

        edge_id
    }

    /// Get all parallel edges between two nodes
    pub fn get_edges_between(&self, source: &N, target: &N) -> Vec<(usize, &E)> {
        let mut result = Vec::new();

        if let Some(neighbors) = self.out_adjacency.get(source) {
            for (neighbor, weight, edge_id) in neighbors {
                if neighbor == target {
                    result.push((*edge_id, weight));
                }
            }
        }

        result
    }

    /// Remove an edge by its ID
    pub fn remove_edge(&mut self, edge_id: usize) -> Result<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        if let Some(edge) = self.edges.remove(&edge_id) {
            // Remove from outgoing adjacency list
            if let Some(neighbors) = self.out_adjacency.get_mut(&edge.source) {
                neighbors.retain(|(_, _, id)| *id != edge_id);
            }

            // Remove from incoming adjacency list
            if let Some(neighbors) = self.in_adjacency.get_mut(&edge.target) {
                neighbors.retain(|(_, _, id)| *id != edge_id);
            }

            Ok(edge)
        } else {
            Err(GraphError::EdgeNotFound)
        }
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> std::collections::hash_set::Iter<'_, N> {
        self.nodes.iter()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> std::collections::hash_map::Values<'_, usize, Edge<N, E>> {
        self.edges.values()
    }

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get outgoing neighbors of a node with edge weights and IDs
    pub fn successors_with_edges(&self, node: &N) -> Option<&Vec<(N, E, usize)>> {
        self.out_adjacency.get(node)
    }

    /// Get incoming neighbors of a node with edge weights and IDs
    pub fn predecessors_with_edges(&self, node: &N) -> Option<&Vec<(N, E, usize)>> {
        self.in_adjacency.get(node)
    }

    /// Get simple successors (without edge information)
    pub fn successors(&self, node: &N) -> Vec<&N> {
        if let Some(neighbors) = self.out_adjacency.get(node) {
            neighbors.iter().map(|(neighbor, _, _)| neighbor).collect()
        } else {
            Vec::new()
        }
    }

    /// Get simple predecessors (without edge information)
    pub fn predecessors(&self, node: &N) -> Vec<&N> {
        if let Some(neighbors) = self.in_adjacency.get(node) {
            neighbors.iter().map(|(neighbor, _, _)| neighbor).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if the graph contains a node
    pub fn has_node(&self, node: &N) -> bool {
        self.nodes.contains(node)
    }

    /// Get the out-degree of a node
    pub fn out_degree(&self, node: &N) -> usize {
        self.out_adjacency
            .get(node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Get the in-degree of a node
    pub fn in_degree(&self, node: &N) -> usize {
        self.in_adjacency
            .get(node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Get the total degree of a node (in-degree + out-degree)
    pub fn degree(&self, node: &N) -> usize {
        self.in_degree(node) + self.out_degree(node)
    }
}

/// A specialized bipartite graph structure
///
/// A bipartite graph is a graph whose vertices can be divided into two disjoint sets
/// such that no two vertices within the same set are adjacent. This implementation
/// enforces the bipartite property and provides optimized operations.
pub struct BipartiteGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// The underlying undirected graph
    graph: Graph<N, E, Ix>,
    /// Set A of the bipartition
    set_a: std::collections::HashSet<N>,
    /// Set B of the bipartition
    set_b: std::collections::HashSet<N>,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for BipartiteGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> BipartiteGraph<N, E, Ix> {
    /// Create a new empty bipartite graph
    pub fn new() -> Self {
        BipartiteGraph {
            graph: Graph::new(),
            set_a: std::collections::HashSet::new(),
            set_b: std::collections::HashSet::new(),
        }
    }

    /// Create a bipartite graph from a regular graph if it's bipartite
    ///
    /// # Arguments
    /// * `graph` - The input graph to convert
    ///
    /// # Returns
    /// * `Result<BipartiteGraph<N, E, Ix>>` - The bipartite graph if conversion is successful
    pub fn from_graph(graph: Graph<N, E, Ix>) -> Result<Self>
    where
        N: Clone,
        E: Clone,
    {
        // Check if the graph is bipartite using the existing algorithm
        let bipartite_result = crate::algorithms::connectivity::is_bipartite(&graph);

        if !bipartite_result.is_bipartite {
            return Err(GraphError::InvalidGraph(
                "Input graph is not bipartite".to_string(),
            ));
        }

        let mut set_a = std::collections::HashSet::new();
        let mut set_b = std::collections::HashSet::new();

        // Partition nodes based on coloring
        for (node, &color) in &bipartite_result.coloring {
            if color == 0 {
                set_a.insert(node.clone());
            } else {
                set_b.insert(node.clone());
            }
        }

        Ok(BipartiteGraph {
            graph,
            set_a,
            set_b,
        })
    }

    /// Add a node to set A of the bipartition
    pub fn add_node_to_set_a(&mut self, node: N) {
        if !self.set_b.contains(&node) {
            self.graph.add_node(node.clone());
            self.set_a.insert(node);
        }
    }

    /// Add a node to set B of the bipartition
    pub fn add_node_to_set_b(&mut self, node: N) {
        if !self.set_a.contains(&node) {
            self.graph.add_node(node.clone());
            self.set_b.insert(node);
        }
    }

    /// Add an edge between nodes from different sets
    ///
    /// # Arguments
    /// * `source` - Source node (must be in one set)
    /// * `target` - Target node (must be in the other set)
    /// * `weight` - Edge weight
    ///
    /// # Returns
    /// * `Result<()>` - Success or error if nodes are in the same set
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()>
    where
        N: Clone,
    {
        // Validate bipartite property
        let source_in_a = self.set_a.contains(&source);
        let source_in_b = self.set_b.contains(&source);
        let target_in_a = self.set_a.contains(&target);
        let target_in_b = self.set_b.contains(&target);

        // Check if both nodes exist in the graph
        if (!source_in_a && !source_in_b) || (!target_in_a && !target_in_b) {
            return Err(GraphError::NodeNotFound);
        }

        // Check bipartite constraint: nodes must be in different sets
        if (source_in_a && target_in_a) || (source_in_b && target_in_b) {
            return Err(GraphError::InvalidGraph(
                "Cannot add edge between nodes in the same partition".to_string(),
            ));
        }

        self.graph.add_edge(source, target, weight)
    }

    /// Get all nodes in set A
    pub fn set_a(&self) -> &std::collections::HashSet<N> {
        &self.set_a
    }

    /// Get all nodes in set B
    pub fn set_b(&self) -> &std::collections::HashSet<N> {
        &self.set_b
    }

    /// Get the size of set A
    pub fn set_a_size(&self) -> usize {
        self.set_a.len()
    }

    /// Get the size of set B
    pub fn set_b_size(&self) -> usize {
        self.set_b.len()
    }

    /// Check which set a node belongs to
    ///
    /// # Arguments
    /// * `node` - The node to check
    ///
    /// # Returns
    /// * `Some(0)` if node is in set A, `Some(1)` if in set B, `None` if not found
    pub fn node_set(&self, node: &N) -> Option<u8> {
        if self.set_a.contains(node) {
            Some(0)
        } else if self.set_b.contains(node) {
            Some(1)
        } else {
            None
        }
    }

    /// Get neighbors of a node (always from the opposite set)
    pub fn neighbors(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        self.graph.neighbors(node)
    }

    /// Get neighbors in set A for a node in set B
    pub fn neighbors_in_a(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if !self.set_b.contains(node) {
            return Err(GraphError::InvalidGraph(
                "Node must be in set B to get neighbors in set A".to_string(),
            ));
        }

        let all_neighbors = self.graph.neighbors(node)?;
        Ok(all_neighbors
            .into_iter()
            .filter(|n| self.set_a.contains(n))
            .collect())
    }

    /// Get neighbors in set B for a node in set A
    pub fn neighbors_in_b(&self, node: &N) -> Result<Vec<N>>
    where
        N: Clone,
    {
        if !self.set_a.contains(node) {
            return Err(GraphError::InvalidGraph(
                "Node must be in set A to get neighbors in set B".to_string(),
            ));
        }

        let all_neighbors = self.graph.neighbors(node)?;
        Ok(all_neighbors
            .into_iter()
            .filter(|n| self.set_b.contains(n))
            .collect())
    }

    /// Get the degree of a node
    pub fn degree(&self, node: &N) -> usize {
        self.graph.degree(node)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&N> {
        self.graph.nodes()
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<Edge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        self.graph.edges()
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
        self.graph.has_node(node)
    }

    /// Check if an edge exists between two nodes
    pub fn has_edge(&self, source: &N, target: &N) -> bool {
        self.graph.has_edge(source, target)
    }

    /// Get the weight of an edge between two nodes
    pub fn edge_weight(&self, source: &N, target: &N) -> Result<E>
    where
        E: Clone,
    {
        self.graph.edge_weight(source, target)
    }

    /// Get the adjacency matrix representation of the bipartite graph
    ///
    /// Returns a matrix where rows correspond to set A and columns to set B
    pub fn biadjacency_matrix(&self) -> Array2<E>
    where
        E: num_traits::Zero + Copy,
        N: Clone,
    {
        let a_size = self.set_a.len();
        let b_size = self.set_b.len();
        let mut biadj_mat = Array2::zeros((a_size, b_size));

        // Create mappings from nodes to indices
        let a_nodes: Vec<&N> = self.set_a.iter().collect();
        let b_nodes: Vec<&N> = self.set_b.iter().collect();

        let a_to_idx: HashMap<&N, usize> =
            a_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let b_to_idx: HashMap<&N, usize> =
            b_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        // Fill the biadjacency matrix
        for edge in self.graph.edges() {
            let (a_idx, b_idx) = if self.set_a.contains(&edge.source) {
                (a_to_idx[&edge.source], b_to_idx[&edge.target])
            } else {
                (a_to_idx[&edge.target], b_to_idx[&edge.source])
            };
            biadj_mat[[a_idx, b_idx]] = edge.weight;
        }

        biadj_mat
    }

    /// Convert to a regular graph
    pub fn to_graph(self) -> Graph<N, E, Ix> {
        self.graph
    }

    /// Get a reference to the underlying graph
    pub fn as_graph(&self) -> &Graph<N, E, Ix> {
        &self.graph
    }

    /// Check if the graph is complete bipartite (all possible edges exist)
    pub fn is_complete(&self) -> bool {
        let expected_edges = self.set_a.len() * self.set_b.len();
        self.edge_count() == expected_edges
    }

    /// Get the maximum possible number of edges for this bipartite graph
    pub fn max_edges(&self) -> usize {
        self.set_a.len() * self.set_b.len()
    }

    /// Get the density of the bipartite graph (actual edges / max possible edges)
    pub fn density(&self) -> f64 {
        if self.max_edges() == 0 {
            0.0
        } else {
            self.edge_count() as f64 / self.max_edges() as f64
        }
    }
}

/// A hypergraph structure where hyperedges can connect any number of vertices
///
/// A hypergraph is a generalization of a graph where edges (called hyperedges)
/// can connect any number of vertices, not just two. This is useful for modeling
/// complex relationships where multiple entities interact simultaneously.
pub struct Hypergraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// All nodes in the hypergraph
    nodes: std::collections::HashSet<N>,
    /// Hyperedges stored as (hyperedge_id, nodes, weight)
    hyperedges: HashMap<usize, (std::collections::HashSet<N>, E)>,
    /// Node to hyperedges mapping: node -> set of hyperedge IDs containing this node
    node_to_hyperedges: HashMap<N, std::collections::HashSet<usize>>,
    /// Counter for unique hyperedge IDs
    hyperedge_id_counter: usize,
    /// Phantom data for index type
    _phantom: std::marker::PhantomData<Ix>,
}

/// Represents a hyperedge in a hypergraph
#[derive(Debug, Clone)]
pub struct Hyperedge<N: Node, E: EdgeWeight> {
    /// Unique identifier for this hyperedge
    pub id: usize,
    /// Set of nodes connected by this hyperedge
    pub nodes: std::collections::HashSet<N>,
    /// Weight/value associated with this hyperedge
    pub weight: E,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for Hypergraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Hypergraph<N, E, Ix> {
    /// Create a new empty hypergraph
    pub fn new() -> Self {
        Hypergraph {
            nodes: std::collections::HashSet::new(),
            hyperedges: HashMap::new(),
            node_to_hyperedges: HashMap::new(),
            hyperedge_id_counter: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a node to the hypergraph
    pub fn add_node(&mut self, node: N) {
        if !self.nodes.contains(&node) {
            self.nodes.insert(node.clone());
            self.node_to_hyperedges
                .insert(node, std::collections::HashSet::new());
        }
    }

    /// Add a hyperedge connecting a set of nodes with a given weight
    ///
    /// # Arguments
    /// * `nodes` - Set of nodes to connect
    /// * `weight` - Weight of the hyperedge
    ///
    /// # Returns
    /// * The ID of the created hyperedge
    pub fn add_hyperedge(
        &mut self,
        nodes: std::collections::HashSet<N>,
        weight: E,
    ) -> Result<usize> {
        if nodes.is_empty() {
            return Err(GraphError::InvalidGraph(
                "Hyperedge must connect at least one node".to_string(),
            ));
        }

        // Ensure all nodes exist in the hypergraph
        for node in &nodes {
            self.add_node(node.clone());
        }

        let hyperedge_id = self.hyperedge_id_counter;
        self.hyperedge_id_counter += 1;

        // Store the hyperedge
        self.hyperedges
            .insert(hyperedge_id, (nodes.clone(), weight));

        // Update node-to-hyperedges mapping
        for node in &nodes {
            self.node_to_hyperedges
                .get_mut(node)
                .unwrap()
                .insert(hyperedge_id);
        }

        Ok(hyperedge_id)
    }

    /// Add a hyperedge from a vector of nodes (convenience method)
    pub fn add_hyperedge_from_vec(&mut self, nodes: Vec<N>, weight: E) -> Result<usize> {
        let node_set: std::collections::HashSet<N> = nodes.into_iter().collect();
        self.add_hyperedge(node_set, weight)
    }

    /// Remove a hyperedge by its ID
    pub fn remove_hyperedge(&mut self, hyperedge_id: usize) -> Result<Hyperedge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        if let Some((nodes, weight)) = self.hyperedges.remove(&hyperedge_id) {
            // Remove from node-to-hyperedges mapping
            for node in &nodes {
                if let Some(hyperedge_set) = self.node_to_hyperedges.get_mut(node) {
                    hyperedge_set.remove(&hyperedge_id);
                }
            }

            Ok(Hyperedge {
                id: hyperedge_id,
                nodes,
                weight,
            })
        } else {
            Err(GraphError::EdgeNotFound)
        }
    }

    /// Get all nodes in the hypergraph
    pub fn nodes(&self) -> std::collections::hash_set::Iter<'_, N> {
        self.nodes.iter()
    }

    /// Get all hyperedges in the hypergraph
    pub fn hyperedges(&self) -> Vec<Hyperedge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        self.hyperedges
            .iter()
            .map(|(&id, (nodes, weight))| Hyperedge {
                id,
                nodes: nodes.clone(),
                weight: weight.clone(),
            })
            .collect()
    }

    /// Get a specific hyperedge by its ID
    pub fn get_hyperedge(&self, hyperedge_id: usize) -> Option<Hyperedge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        self.hyperedges
            .get(&hyperedge_id)
            .map(|(nodes, weight)| Hyperedge {
                id: hyperedge_id,
                nodes: nodes.clone(),
                weight: weight.clone(),
            })
    }

    /// Get all hyperedges that contain a specific node
    pub fn hyperedges_containing_node(&self, node: &N) -> Vec<Hyperedge<N, E>>
    where
        N: Clone,
        E: Clone,
    {
        let mut result = Vec::new();

        if let Some(hyperedge_ids) = self.node_to_hyperedges.get(node) {
            for &hyperedge_id in hyperedge_ids {
                if let Some(hyperedge) = self.get_hyperedge(hyperedge_id) {
                    result.push(hyperedge);
                }
            }
        }

        result
    }

    /// Get all nodes that are connected to a given node through some hyperedge
    ///
    /// Returns all nodes that share at least one hyperedge with the given node
    pub fn neighbors(&self, node: &N) -> std::collections::HashSet<N>
    where
        N: Clone,
    {
        let mut neighbors = std::collections::HashSet::new();

        if let Some(hyperedge_ids) = self.node_to_hyperedges.get(node) {
            for &hyperedge_id in hyperedge_ids {
                if let Some((nodes, _)) = self.hyperedges.get(&hyperedge_id) {
                    for neighbor in nodes {
                        if neighbor != node {
                            neighbors.insert(neighbor.clone());
                        }
                    }
                }
            }
        }

        neighbors
    }

    /// Number of nodes in the hypergraph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of hyperedges in the hypergraph
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedges.len()
    }

    /// Check if the hypergraph contains a specific node
    pub fn has_node(&self, node: &N) -> bool {
        self.nodes.contains(node)
    }

    /// Check if the hypergraph contains a specific hyperedge
    pub fn has_hyperedge(&self, hyperedge_id: usize) -> bool {
        self.hyperedges.contains_key(&hyperedge_id)
    }

    /// Get the degree of a node (number of hyperedges it participates in)
    pub fn degree(&self, node: &N) -> usize {
        self.node_to_hyperedges
            .get(node)
            .map_or(0, |hyperedges| hyperedges.len())
    }

    /// Get the size of a hyperedge (number of nodes it connects)
    pub fn hyperedge_size(&self, hyperedge_id: usize) -> Option<usize> {
        self.hyperedges
            .get(&hyperedge_id)
            .map(|(nodes, _)| nodes.len())
    }

    /// Check if two nodes are connected (share at least one hyperedge)
    pub fn are_connected(&self, node1: &N, node2: &N) -> bool {
        if let Some(hyperedge_ids) = self.node_to_hyperedges.get(node1) {
            for &hyperedge_id in hyperedge_ids {
                if let Some((nodes, _)) = self.hyperedges.get(&hyperedge_id) {
                    if nodes.contains(node2) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get the hyperedges that connect two specific nodes
    pub fn connecting_hyperedges(&self, node1: &N, node2: &N) -> Vec<usize> {
        let mut connecting = Vec::new();

        if let Some(hyperedge_ids) = self.node_to_hyperedges.get(node1) {
            for &hyperedge_id in hyperedge_ids {
                if let Some((nodes, _)) = self.hyperedges.get(&hyperedge_id) {
                    if nodes.contains(node2) {
                        connecting.push(hyperedge_id);
                    }
                }
            }
        }

        connecting
    }

    /// Convert to a regular graph by creating edges between all pairs of nodes
    /// that are connected by the same hyperedge
    ///
    /// This creates the 2-section (or clique expansion) of the hypergraph
    pub fn to_graph(&self) -> Graph<N, E, Ix>
    where
        N: Clone,
        E: Clone + num_traits::Zero + std::ops::Add<Output = E>,
    {
        let mut graph = Graph::new();

        // Add all nodes
        for node in &self.nodes {
            graph.add_node(node.clone());
        }

        // For each hyperedge, connect all pairs of nodes
        for (nodes, weight) in self.hyperedges.values() {
            let node_vec: Vec<&N> = nodes.iter().collect();
            for i in 0..node_vec.len() {
                for j in (i + 1)..node_vec.len() {
                    let node1 = node_vec[i];
                    let node2 = node_vec[j];

                    // If edge already exists, add to its weight
                    if graph.has_edge(node1, node2) {
                        if let Ok(existing_weight) = graph.edge_weight(node1, node2) {
                            let new_weight: E = existing_weight + weight.clone();
                            // Remove and re-add with new weight
                            // Note: This is a simplified approach. In practice, you might want
                            // a more efficient way to update edge weights
                            let mut new_graph = Graph::new();
                            for node in graph.nodes() {
                                new_graph.add_node(node.clone());
                            }
                            for edge in graph.edges() {
                                if (edge.source == *node1 && edge.target == *node2)
                                    || (edge.source == *node2 && edge.target == *node1)
                                {
                                    new_graph
                                        .add_edge(edge.source, edge.target, new_weight.clone())
                                        .unwrap();
                                } else {
                                    new_graph
                                        .add_edge(edge.source, edge.target, edge.weight)
                                        .unwrap();
                                }
                            }
                            graph = new_graph;
                        }
                    } else {
                        graph
                            .add_edge(node1.clone(), node2.clone(), weight.clone())
                            .unwrap();
                    }
                }
            }
        }

        graph
    }

    /// Get the incidence matrix of the hypergraph
    ///
    /// Returns a matrix where rows represent nodes and columns represent hyperedges.
    /// Entry (i,j) is 1 if node i is in hyperedge j, 0 otherwise.
    pub fn incidence_matrix(&self) -> Array2<u8>
    where
        N: Clone + Ord,
    {
        let mut sorted_nodes: Vec<&N> = self.nodes.iter().collect();
        sorted_nodes.sort();

        let mut sorted_hyperedges: Vec<usize> = self.hyperedges.keys().cloned().collect();
        sorted_hyperedges.sort();

        let mut matrix = Array2::zeros((sorted_nodes.len(), sorted_hyperedges.len()));

        let node_to_idx: HashMap<&N, usize> = sorted_nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        let hyperedge_to_idx: HashMap<usize, usize> = sorted_hyperedges
            .iter()
            .enumerate()
            .map(|(j, &he_id)| (he_id, j))
            .collect();

        for (&hyperedge_id, (nodes, _)) in &self.hyperedges {
            let j = hyperedge_to_idx[&hyperedge_id];
            for node in nodes {
                if let Some(&i) = node_to_idx.get(node) {
                    matrix[[i, j]] = 1;
                }
            }
        }

        matrix
    }

    /// Check if the hypergraph is uniform (all hyperedges have the same size)
    pub fn is_uniform(&self) -> bool {
        if self.hyperedges.is_empty() {
            return true;
        }

        let first_size = self.hyperedges.values().next().unwrap().0.len();
        self.hyperedges
            .values()
            .all(|(nodes, _)| nodes.len() == first_size)
    }

    /// Get the uniformity of the hypergraph (the common size if uniform, None otherwise)
    pub fn uniformity(&self) -> Option<usize> {
        if self.is_uniform() && !self.hyperedges.is_empty() {
            Some(self.hyperedges.values().next().unwrap().0.len())
        } else {
            None
        }
    }

    /// Get statistics about hyperedge sizes
    pub fn hyperedge_size_stats(&self) -> (usize, usize, f64) {
        if self.hyperedges.is_empty() {
            return (0, 0, 0.0);
        }

        let sizes: Vec<usize> = self
            .hyperedges
            .values()
            .map(|(nodes, _)| nodes.len())
            .collect();

        let min_size = *sizes.iter().min().unwrap();
        let max_size = *sizes.iter().max().unwrap();
        let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;

        (min_size, max_size, avg_size)
    }

    /// Find all maximal cliques in the hypergraph
    ///
    /// A maximal clique is a set of nodes that are all connected to each other
    /// and cannot be extended by adding another node while maintaining this property.
    /// In hypergraphs, this corresponds to finding maximal sets of nodes that
    /// all participate in some common hyperedge.
    pub fn maximal_cliques(&self) -> Vec<std::collections::HashSet<N>>
    where
        N: Clone,
    {
        let mut cliques = Vec::new();

        // For each hyperedge, the nodes form a clique
        for (nodes, _) in self.hyperedges.values() {
            let mut is_maximal = true;

            // Check if this clique is contained in any existing clique
            for existing_clique in &cliques {
                if nodes.is_subset(existing_clique) {
                    is_maximal = false;
                    break;
                }
            }

            if is_maximal {
                // Remove any existing cliques that are subsets of this one
                cliques.retain(|existing_clique| !existing_clique.is_subset(nodes));
                cliques.push(nodes.clone());
            }
        }

        cliques
    }

    /// Create a hypergraph from a regular graph where each edge becomes a 2-uniform hyperedge
    pub fn from_graph(graph: &Graph<N, E, Ix>) -> Self
    where
        N: Clone,
        E: Clone,
    {
        let mut hypergraph = Hypergraph::new();

        // Add all nodes
        for node in graph.nodes() {
            hypergraph.add_node(node.clone());
        }

        // Convert each edge to a 2-uniform hyperedge
        for edge in graph.edges() {
            let mut nodes = std::collections::HashSet::new();
            nodes.insert(edge.source);
            nodes.insert(edge.target);
            hypergraph.add_hyperedge(nodes, edge.weight).unwrap();
        }

        hypergraph
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

    #[test]
    fn test_multigraph_parallel_edges() {
        let mut graph: MultiGraph<&str, f64> = MultiGraph::new();

        // Add parallel edges between A and B
        let _edge1 = graph.add_edge("A", "B", 1.0);
        let edge2 = graph.add_edge("A", "B", 2.0);
        let _edge3 = graph.add_edge("A", "B", 3.0);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 3);

        // Check that we can get all parallel edges
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 3);

        // Check edge weights
        let weights: Vec<f64> = edges_ab.iter().map(|(_, &weight)| weight).collect();
        assert!(weights.contains(&1.0));
        assert!(weights.contains(&2.0));
        assert!(weights.contains(&3.0));

        // Remove one edge
        let removed_edge = graph.remove_edge(edge2).unwrap();
        assert_eq!(removed_edge.weight, 2.0);
        assert_eq!(graph.edge_count(), 2);

        // Check remaining edges
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 2);
    }

    #[test]
    fn test_multidigraph_parallel_edges() {
        let mut graph: MultiDiGraph<&str, f64> = MultiDiGraph::new();

        // Add parallel directed edges from A to B
        let edge1 = graph.add_edge("A", "B", 1.0);
        let _edge2 = graph.add_edge("A", "B", 2.0);

        // Add edge in opposite direction
        let _edge3 = graph.add_edge("B", "A", 3.0);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 3);

        // Check outgoing edges from A
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 2);

        // Check outgoing edges from B
        let edges_ba = graph.get_edges_between(&"B", &"A");
        assert_eq!(edges_ba.len(), 1);

        // Check degrees
        assert_eq!(graph.out_degree(&"A"), 2);
        assert_eq!(graph.in_degree(&"A"), 1);
        assert_eq!(graph.out_degree(&"B"), 1);
        assert_eq!(graph.in_degree(&"B"), 2);

        // Remove edge and check
        graph.remove_edge(edge1).unwrap();
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.out_degree(&"A"), 1);
        assert_eq!(graph.in_degree(&"B"), 1);
    }

    #[test]
    fn test_bipartite_graph_creation() {
        let mut bipartite: BipartiteGraph<&str, f64> = BipartiteGraph::new();

        // Add nodes to different sets
        bipartite.add_node_to_set_a("A1");
        bipartite.add_node_to_set_a("A2");
        bipartite.add_node_to_set_b("B1");
        bipartite.add_node_to_set_b("B2");

        assert_eq!(bipartite.set_a_size(), 2);
        assert_eq!(bipartite.set_b_size(), 2);
        assert_eq!(bipartite.node_count(), 4);

        // Add valid edges (between different sets)
        assert!(bipartite.add_edge("A1", "B1", 1.0).is_ok());
        assert!(bipartite.add_edge("A2", "B2", 2.0).is_ok());

        assert_eq!(bipartite.edge_count(), 2);
    }

    #[test]
    fn test_bipartite_graph_invalid_edges() {
        let mut bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a(1);
        bipartite.add_node_to_set_a(2);
        bipartite.add_node_to_set_b(3);
        bipartite.add_node_to_set_b(4);

        // Try to add edge within same set (should fail)
        assert!(bipartite.add_edge(1, 2, 1.0).is_err());
        assert!(bipartite.add_edge(3, 4, 1.0).is_err());

        // Valid edge should work
        assert!(bipartite.add_edge(1, 3, 1.0).is_ok());
    }

    #[test]
    fn test_bipartite_graph_from_regular_graph() {
        // Create a bipartite graph (square)
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();
        graph.add_edge(4, 1, 4.0).unwrap();

        // Convert to bipartite graph
        let bipartite = BipartiteGraph::from_graph(graph).unwrap();

        assert_eq!(bipartite.node_count(), 4);
        assert_eq!(bipartite.edge_count(), 4);
        assert_eq!(bipartite.set_a_size() + bipartite.set_b_size(), 4);

        // Check that nodes are properly partitioned
        assert!(bipartite.set_a_size() == 2);
        assert!(bipartite.set_b_size() == 2);
    }

    #[test]
    fn test_bipartite_graph_from_non_bipartite_graph() {
        // Create a non-bipartite graph (triangle)
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 1, 3.0).unwrap();

        // Should fail to convert
        assert!(BipartiteGraph::from_graph(graph).is_err());
    }

    #[test]
    fn test_bipartite_graph_node_set_identification() {
        let mut bipartite: BipartiteGraph<char, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a('A');
        bipartite.add_node_to_set_b('B');

        assert_eq!(bipartite.node_set(&'A'), Some(0));
        assert_eq!(bipartite.node_set(&'B'), Some(1));
        assert_eq!(bipartite.node_set(&'C'), None);
    }

    #[test]
    fn test_bipartite_graph_neighbors() {
        let mut bipartite: BipartiteGraph<&str, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a("A1");
        bipartite.add_node_to_set_a("A2");
        bipartite.add_node_to_set_b("B1");
        bipartite.add_node_to_set_b("B2");

        bipartite.add_edge("A1", "B1", 1.0).unwrap();
        bipartite.add_edge("A1", "B2", 2.0).unwrap();
        bipartite.add_edge("A2", "B1", 3.0).unwrap();

        // Test neighbors_in_b for nodes in set A
        let a1_neighbors = bipartite.neighbors_in_b(&"A1").unwrap();
        assert_eq!(a1_neighbors.len(), 2);
        assert!(a1_neighbors.contains(&"B1"));
        assert!(a1_neighbors.contains(&"B2"));

        // Test neighbors_in_a for nodes in set B
        let b1_neighbors = bipartite.neighbors_in_a(&"B1").unwrap();
        assert_eq!(b1_neighbors.len(), 2);
        assert!(b1_neighbors.contains(&"A1"));
        assert!(b1_neighbors.contains(&"A2"));

        // Test invalid neighbor queries
        assert!(bipartite.neighbors_in_b(&"B1").is_err()); // B1 is not in set A
        assert!(bipartite.neighbors_in_a(&"A1").is_err()); // A1 is not in set B
    }

    #[test]
    fn test_bipartite_graph_biadjacency_matrix() {
        let mut bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a(1);
        bipartite.add_node_to_set_a(2);
        bipartite.add_node_to_set_b(3);
        bipartite.add_node_to_set_b(4);

        bipartite.add_edge(1, 3, 5.0).unwrap();
        bipartite.add_edge(2, 4, 7.0).unwrap();

        let biadj = bipartite.biadjacency_matrix();
        assert_eq!(biadj.shape(), &[2, 2]);

        // Check that the matrix has the expected structure
        // Note: exact positions may vary based on hash set iteration order
        let total_sum: f64 = biadj.iter().sum();
        assert_eq!(total_sum, 12.0); // 5.0 + 7.0 = 12.0
    }

    #[test]
    fn test_bipartite_graph_completeness() {
        let mut bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a(1);
        bipartite.add_node_to_set_a(2);
        bipartite.add_node_to_set_b(3);
        bipartite.add_node_to_set_b(4);

        // Not complete initially
        assert!(!bipartite.is_complete());
        assert_eq!(bipartite.max_edges(), 4); // 2 * 2 = 4
        assert_eq!(bipartite.density(), 0.0);

        // Add all possible edges
        bipartite.add_edge(1, 3, 1.0).unwrap();
        bipartite.add_edge(1, 4, 1.0).unwrap();
        bipartite.add_edge(2, 3, 1.0).unwrap();
        bipartite.add_edge(2, 4, 1.0).unwrap();

        // Now it should be complete
        assert!(bipartite.is_complete());
        assert_eq!(bipartite.density(), 1.0);
    }

    #[test]
    fn test_bipartite_graph_conversion() {
        let mut bipartite: BipartiteGraph<&str, f64> = BipartiteGraph::new();

        bipartite.add_node_to_set_a("A");
        bipartite.add_node_to_set_b("B");
        bipartite.add_edge("A", "B", 3.15).unwrap();

        // Convert to regular graph
        let regular_graph = bipartite.to_graph();
        assert_eq!(regular_graph.node_count(), 2);
        assert_eq!(regular_graph.edge_count(), 1);
        assert!(regular_graph.has_edge(&"A", &"B"));
    }

    #[test]
    fn test_hypergraph_creation() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        // Add some nodes
        hypergraph.add_node("A");
        hypergraph.add_node("B");
        hypergraph.add_node("C");

        assert_eq!(hypergraph.node_count(), 3);
        assert_eq!(hypergraph.hyperedge_count(), 0);
        assert!(hypergraph.has_node(&"A"));
        assert!(hypergraph.has_node(&"B"));
        assert!(hypergraph.has_node(&"C"));
    }

    #[test]
    fn test_hypergraph_add_hyperedge() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Create a 3-uniform hyperedge
        let nodes = vec![1, 2, 3];
        let hyperedge_id = hypergraph.add_hyperedge_from_vec(nodes, 1.5).unwrap();

        assert_eq!(hypergraph.node_count(), 3);
        assert_eq!(hypergraph.hyperedge_count(), 1);
        assert!(hypergraph.has_hyperedge(hyperedge_id));
        assert_eq!(hypergraph.hyperedge_size(hyperedge_id), Some(3));

        // Check that nodes are properly connected
        assert!(hypergraph.are_connected(&1, &2));
        assert!(hypergraph.are_connected(&2, &3));
        assert!(hypergraph.are_connected(&1, &3));

        // Check degrees
        assert_eq!(hypergraph.degree(&1), 1);
        assert_eq!(hypergraph.degree(&2), 1);
        assert_eq!(hypergraph.degree(&3), 1);
    }

    #[test]
    fn test_hypergraph_neighbors() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        // Add two hyperedges: {A, B, C} and {B, C, D}
        hypergraph
            .add_hyperedge_from_vec(vec!["A", "B", "C"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["B", "C", "D"], 2.0)
            .unwrap();

        // Check neighbors of B
        let b_neighbors = hypergraph.neighbors(&"B");
        assert_eq!(b_neighbors.len(), 3);
        assert!(b_neighbors.contains(&"A"));
        assert!(b_neighbors.contains(&"C"));
        assert!(b_neighbors.contains(&"D"));

        // Check neighbors of A
        let a_neighbors = hypergraph.neighbors(&"A");
        assert_eq!(a_neighbors.len(), 2);
        assert!(a_neighbors.contains(&"B"));
        assert!(a_neighbors.contains(&"C"));
    }

    #[test]
    fn test_hypergraph_remove_hyperedge() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        let hyperedge_id = hypergraph
            .add_hyperedge_from_vec(vec![1, 2, 3], 5.0)
            .unwrap();
        assert_eq!(hypergraph.hyperedge_count(), 1);

        let removed = hypergraph.remove_hyperedge(hyperedge_id).unwrap();
        assert_eq!(removed.weight, 5.0);
        assert_eq!(removed.nodes.len(), 3);
        assert_eq!(hypergraph.hyperedge_count(), 0);

        // Nodes should still exist but have no connections
        assert!(hypergraph.has_node(&1));
        assert_eq!(hypergraph.degree(&1), 0);
        assert!(!hypergraph.are_connected(&1, &2));
    }

    #[test]
    fn test_hypergraph_from_graph() {
        let mut graph: Graph<&str, f64> = Graph::new();
        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();

        let hypergraph = Hypergraph::from_graph(&graph);

        assert_eq!(hypergraph.node_count(), 3);
        assert_eq!(hypergraph.hyperedge_count(), 2);
        assert!(hypergraph.is_uniform());
        assert_eq!(hypergraph.uniformity(), Some(2));

        // Each edge should become a 2-uniform hyperedge
        let hyperedges = hypergraph.hyperedges();
        for hyperedge in hyperedges {
            assert_eq!(hyperedge.nodes.len(), 2);
        }
    }

    #[test]
    fn test_hypergraph_to_graph() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Add a 3-uniform hyperedge {1, 2, 3}
        hypergraph
            .add_hyperedge_from_vec(vec![1, 2, 3], 1.0)
            .unwrap();

        let graph = hypergraph.to_graph();

        // Should create edges (1,2), (1,3), (2,3) all with weight 1.0
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        assert!(graph.has_edge(&1, &2));
        assert!(graph.has_edge(&1, &3));
        assert!(graph.has_edge(&2, &3));
    }

    #[test]
    fn test_hypergraph_incidence_matrix() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        hypergraph.add_hyperedge_from_vec(vec![1, 2], 1.0).unwrap();
        hypergraph.add_hyperedge_from_vec(vec![2, 3], 2.0).unwrap();

        let matrix = hypergraph.incidence_matrix();
        assert_eq!(matrix.shape(), &[3, 2]);

        // Each row should have exactly one or two 1s (depending on node participation)
        for row in matrix.rows() {
            let sum: u8 = row.iter().sum();
            assert!(sum >= 1);
        }
    }

    #[test]
    fn test_hypergraph_uniformity() {
        let mut uniform_hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Add 3-uniform hyperedges
        uniform_hypergraph
            .add_hyperedge_from_vec(vec![1, 2, 3], 1.0)
            .unwrap();
        uniform_hypergraph
            .add_hyperedge_from_vec(vec![4, 5, 6], 2.0)
            .unwrap();

        assert!(uniform_hypergraph.is_uniform());
        assert_eq!(uniform_hypergraph.uniformity(), Some(3));

        // Add a 2-uniform hyperedge to make it non-uniform
        uniform_hypergraph
            .add_hyperedge_from_vec(vec![1, 4], 3.0)
            .unwrap();

        assert!(!uniform_hypergraph.is_uniform());
        assert_eq!(uniform_hypergraph.uniformity(), None);
    }

    #[test]
    fn test_hypergraph_size_stats() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        hypergraph
            .add_hyperedge_from_vec(vec!["A", "B"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["B", "C", "D"], 2.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["A", "B", "C", "D", "E"], 3.0)
            .unwrap();

        let (min, max, avg) = hypergraph.hyperedge_size_stats();
        assert_eq!(min, 2);
        assert_eq!(max, 5);
        assert!((avg - 10.0 / 3.0).abs() < 1e-6); // (2 + 3 + 5) / 3 = 10/3 ≈ 3.33
    }

    #[test]
    fn test_hypergraph_maximal_cliques() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Add overlapping hyperedges
        hypergraph
            .add_hyperedge_from_vec(vec![1, 2, 3], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec![2, 3, 4], 2.0)
            .unwrap();
        hypergraph.add_hyperedge_from_vec(vec![1, 2], 3.0).unwrap(); // Subset of first

        let cliques = hypergraph.maximal_cliques();

        // Should have 2 maximal cliques: {1,2,3} and {2,3,4}
        // The {1,2} hyperedge should be filtered out as it's a subset
        assert_eq!(cliques.len(), 2);

        let clique_sizes: Vec<usize> = cliques.iter().map(|c| c.len()).collect();
        assert!(clique_sizes.contains(&3));
    }

    #[test]
    fn test_hypergraph_connecting_hyperedges() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        let he1 = hypergraph
            .add_hyperedge_from_vec(vec!["A", "B", "C"], 1.0)
            .unwrap();
        let he2 = hypergraph
            .add_hyperedge_from_vec(vec!["B", "C", "D"], 2.0)
            .unwrap();
        let _he3 = hypergraph
            .add_hyperedge_from_vec(vec!["A", "E"], 3.0)
            .unwrap();

        let connecting_ab = hypergraph.connecting_hyperedges(&"A", &"B");
        assert_eq!(connecting_ab.len(), 1);
        assert!(connecting_ab.contains(&he1));

        let connecting_bc = hypergraph.connecting_hyperedges(&"B", &"C");
        assert_eq!(connecting_bc.len(), 2);
        assert!(connecting_bc.contains(&he1));
        assert!(connecting_bc.contains(&he2));

        let connecting_ad = hypergraph.connecting_hyperedges(&"A", &"D");
        assert_eq!(connecting_ad.len(), 0);
    }

    #[test]
    fn test_hypergraph_empty() {
        let hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        assert_eq!(hypergraph.node_count(), 0);
        assert_eq!(hypergraph.hyperedge_count(), 0);
        assert!(hypergraph.is_uniform()); // Vacuously true
        assert_eq!(hypergraph.uniformity(), None);
        assert_eq!(hypergraph.hyperedge_size_stats(), (0, 0, 0.0));
        assert!(hypergraph.maximal_cliques().is_empty());
    }

    #[test]
    fn test_hypergraph_invalid_hyperedge() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Try to add empty hyperedge
        let result = hypergraph.add_hyperedge_from_vec(vec![], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_bipartite_graph_empty() {
        let bipartite: BipartiteGraph<i32, f64> = BipartiteGraph::new();

        assert_eq!(bipartite.node_count(), 0);
        assert_eq!(bipartite.edge_count(), 0);
        assert_eq!(bipartite.set_a_size(), 0);
        assert_eq!(bipartite.set_b_size(), 0);
        assert_eq!(bipartite.max_edges(), 0);
        assert_eq!(bipartite.density(), 0.0);
        assert!(bipartite.is_complete()); // Vacuously true for empty graph
    }

    #[test]
    fn test_multigraph_self_loops() {
        let mut graph: MultiGraph<i32, f64> = MultiGraph::new();

        // Add self loops
        let _edge1 = graph.add_edge(1, 1, 10.0);
        let _edge2 = graph.add_edge(1, 1, 20.0);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 2);

        // Self loops should appear in neighbors
        let neighbors = graph.neighbors(&1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.iter().all(|&&n| n == 1));

        // Degree should count self loops
        assert_eq!(graph.degree(&1), 2);

        // Check self-loop edges
        let self_edges = graph.get_edges_between(&1, &1);
        assert_eq!(self_edges.len(), 2);
    }
}
