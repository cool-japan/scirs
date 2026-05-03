//! Attributed graphs with typed node and edge features
//!
//! This module provides `AttributedGraph<N, E>` – a graph where every node
//! carries data of type `N` and every edge carries data of type `E`.  It
//! differs from the attribute-map approach in [`crate::attributes`] in that
//! the feature types are *statically known*, enabling zero-cost extraction of
//! dense feature matrices via a user-supplied projection function.
//!
//! ## Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! [`AttributedGraph<N,E>`]  | Core graph container |
//! [`AttributedGraphBuilder<N,E>`] | Ergonomic builder |
//! [`NeighborInfo<N,E>`] | Neighbour + data bundle returned by [`attributed_neighbors`] |
//!
//! ## Free functions
//!
//! - [`node_feature_matrix`] – project node data into an `Array2<f64>`
//! - [`edge_feature_matrix`] – project edge data into an `Array2<f64>`
//! - [`attributed_neighbors`] – enumerate neighbours with their data
//! - [`dijkstra_attributed`] – generalised Dijkstra using a caller-supplied cost function

use std::collections::HashMap;

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};

// ============================================================================
// NodeId
// ============================================================================

/// Stable integer node identifier.
///
/// Assigned sequentially by [`AttributedGraph::add_node`].  Remains valid as
/// long as the graph is not cleared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

// ============================================================================
// AttributedGraph
// ============================================================================

/// A directed graph whose nodes carry data of type `N` and whose edges carry
/// data of type `E`.
///
/// Internally, the graph is represented as an adjacency list over [`NodeId`]
/// indices.  Parallel edges (same source and target) are supported; the first
/// matching edge is returned by lookup helpers.
///
/// # Example
///
/// ```
/// use scirs2_graph::attributed_graph::{AttributedGraph, NodeId};
///
/// #[derive(Debug, Clone)]
/// struct Person { name: String, age: u32 }
///
/// #[derive(Debug, Clone)]
/// struct Knows { since: u32 }
///
/// let mut g: AttributedGraph<Person, Knows> = AttributedGraph::new();
/// let alice = g.add_node(Person { name: "Alice".into(), age: 30 });
/// let bob   = g.add_node(Person { name: "Bob".into(),   age: 25 });
/// g.add_edge(alice, bob, Knows { since: 2020 }).unwrap();
///
/// assert_eq!(g.node_count(), 2);
/// assert_eq!(g.edge_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct AttributedGraph<N, E> {
    /// Node data indexed by internal position (= NodeId.0)
    nodes: Vec<N>,
    /// Adjacency list: `adj[src_idx]` holds `(target_idx, edge_idx)` pairs
    adj: Vec<Vec<(usize, usize)>>,
    /// Edge storage: `(src_idx, dst_idx, E)`
    edges: Vec<(usize, usize, E)>,
}

impl<N, E> Default for AttributedGraph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> AttributedGraph<N, E> {
    /// Create an empty attributed graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            adj: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Create an empty graph with pre-allocated capacity.
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(node_capacity),
            adj: Vec::with_capacity(node_capacity),
            edges: Vec::with_capacity(edge_capacity),
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Add a node with the given data; returns its [`NodeId`].
    pub fn add_node(&mut self, data: N) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(data);
        self.adj.push(Vec::new());
        NodeId(id)
    }

    /// Add a directed edge from `src` to `dst` with data `edge_data`.
    ///
    /// Both nodes must already exist.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if either node is absent.
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, edge_data: E) -> Result<()> {
        self.validate_node(src)?;
        self.validate_node(dst)?;
        let edge_idx = self.edges.len();
        self.edges.push((src.0, dst.0, edge_data));
        self.adj[src.0].push((dst.0, edge_idx));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of (directed) edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Retrieve node data by id.
    ///
    /// Returns `None` if the id is out of range.
    pub fn node_data(&self, id: NodeId) -> Option<&N> {
        self.nodes.get(id.0)
    }

    /// Retrieve mutable node data by id.
    pub fn node_data_mut(&mut self, id: NodeId) -> Option<&mut N> {
        self.nodes.get_mut(id.0)
    }

    /// Iterate over all node ids and their data.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &N)> {
        self.nodes.iter().enumerate().map(|(i, n)| (NodeId(i), n))
    }

    /// Iterate over all edges as `(src, dst, &E)`.
    pub fn edges_iter(&self) -> impl Iterator<Item = (NodeId, NodeId, &E)> {
        self.edges
            .iter()
            .map(|(s, d, e)| (NodeId(*s), NodeId(*d), e))
    }

    /// Return the direct out-neighbours of `node` as `(target, &E)` pairs.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if the node is absent.
    pub fn out_neighbors(&self, node: NodeId) -> Result<Vec<(NodeId, &E)>> {
        self.validate_node(node)?;
        let result = self.adj[node.0]
            .iter()
            .map(|&(dst, eidx)| (NodeId(dst), &self.edges[eidx].2))
            .collect();
        Ok(result)
    }

    /// Return whether an edge from `src` to `dst` exists.
    pub fn has_edge(&self, src: NodeId, dst: NodeId) -> bool {
        if src.0 >= self.nodes.len() || dst.0 >= self.nodes.len() {
            return false;
        }
        self.adj[src.0].iter().any(|&(d, _)| d == dst.0)
    }

    /// Retrieve the data of the first edge from `src` to `dst`.
    ///
    /// Returns `None` if no such edge exists.
    pub fn edge_data(&self, src: NodeId, dst: NodeId) -> Option<&E> {
        if src.0 >= self.nodes.len() {
            return None;
        }
        self.adj[src.0]
            .iter()
            .find(|&&(d, _)| d == dst.0)
            .map(|&(_, eidx)| &self.edges[eidx].2)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn validate_node(&self, id: NodeId) -> Result<()> {
        if id.0 < self.nodes.len() {
            Ok(())
        } else {
            Err(GraphError::node_not_found_with_context(
                id.0,
                self.nodes.len(),
                "AttributedGraph node validation",
            ))
        }
    }
}

// ============================================================================
// AttributedGraphBuilder
// ============================================================================

/// Builder for [`AttributedGraph`] that provides a fluent API.
///
/// # Example
///
/// ```
/// use scirs2_graph::attributed_graph::{AttributedGraphBuilder, NodeId};
///
/// let mut builder: AttributedGraphBuilder<f64, f32> = AttributedGraphBuilder::new();
/// let a = builder.node(1.0);
/// let b = builder.node(2.0);
/// builder.edge(a, b, 0.5_f32);
/// let graph = builder.build().expect("build failed");
/// assert_eq!(graph.node_count(), 2);
/// assert_eq!(graph.edge_count(), 1);
/// ```
#[derive(Debug)]
pub struct AttributedGraphBuilder<N, E> {
    graph: AttributedGraph<N, E>,
    /// Accumulated errors from `edge()` calls
    errors: Vec<GraphError>,
}

impl<N, E> Default for AttributedGraphBuilder<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> AttributedGraphBuilder<N, E> {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            graph: AttributedGraph::new(),
            errors: Vec::new(),
        }
    }

    /// Create a builder with capacity hints.
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            graph: AttributedGraph::with_capacity(node_capacity, edge_capacity),
            errors: Vec::new(),
        }
    }

    /// Add a node; returns the assigned [`NodeId`].
    pub fn node(&mut self, data: N) -> NodeId {
        self.graph.add_node(data)
    }

    /// Add a directed edge.  Any error is deferred until [`build`][Self::build].
    pub fn edge(&mut self, src: NodeId, dst: NodeId, data: E) -> &mut Self {
        if let Err(e) = self.graph.add_edge(src, dst, data) {
            self.errors.push(e);
        }
        self
    }

    /// Consume the builder and return the graph.
    ///
    /// # Errors
    ///
    /// Returns the first deferred error, if any.
    pub fn build(self) -> Result<AttributedGraph<N, E>> {
        if let Some(err) = self.errors.into_iter().next() {
            return Err(err);
        }
        Ok(self.graph)
    }

    /// Consume the builder, returning the graph without checking errors.
    ///
    /// Silently drops any accumulated errors.
    pub fn build_unchecked(self) -> AttributedGraph<N, E> {
        self.graph
    }
}

// ============================================================================
// NeighborInfo
// ============================================================================

/// A neighbour together with its node data and the connecting edge data.
///
/// Returned by [`attributed_neighbors`].
#[derive(Debug)]
pub struct NeighborInfo<'a, N, E> {
    /// The neighbour's id.
    pub id: NodeId,
    /// Reference to the neighbour's node data.
    pub node_data: &'a N,
    /// Reference to the edge data connecting source → neighbour.
    pub edge_data: &'a E,
}

// ============================================================================
// Free functions
// ============================================================================

/// Construct a dense node-feature matrix from an attributed graph.
///
/// `feature_fn` is called for each node in insertion order and must return a
/// `Vec<f64>` of the same length for every node.  The resulting matrix has
/// shape `(node_count, feature_dim)`.
///
/// # Errors
///
/// Returns [`GraphError::InvalidParameter`] if the node count is zero or if
/// `feature_fn` returns vectors of inconsistent length.
///
/// # Example
///
/// ```
/// use scirs2_graph::attributed_graph::{AttributedGraph, node_feature_matrix};
///
/// let mut g = AttributedGraph::<[f64; 2], ()>::new();
/// g.add_node([1.0, 0.5]);
/// g.add_node([0.0, 1.0]);
/// let mat = node_feature_matrix(&g, |n| n.to_vec()).unwrap();
/// assert_eq!(mat.shape(), &[2, 2]);
/// ```
pub fn node_feature_matrix<N, E, F>(
    graph: &AttributedGraph<N, E>,
    feature_fn: F,
) -> Result<Array2<f64>>
where
    F: Fn(&N) -> Vec<f64>,
{
    let n = graph.node_count();
    if n == 0 {
        return Err(GraphError::invalid_parameter(
            "graph",
            "empty graph",
            "at least one node",
        ));
    }

    // Compute features for all nodes
    let features: Vec<Vec<f64>> = graph.nodes.iter().map(feature_fn).collect();

    let dim = features[0].len();
    if dim == 0 {
        return Err(GraphError::invalid_parameter(
            "feature_fn",
            "zero-length feature vector",
            "non-empty feature vector",
        ));
    }

    // Validate uniform dimensionality
    for (i, fv) in features.iter().enumerate() {
        if fv.len() != dim {
            return Err(GraphError::InvalidParameter {
                param: "feature_fn".to_string(),
                value: format!("node {i} returned dim={}", fv.len()),
                expected: format!("uniform dim={dim}"),
                context: "node_feature_matrix".to_string(),
            });
        }
    }

    let mut mat = Array2::zeros((n, dim));
    for (i, fv) in features.iter().enumerate() {
        for (j, &v) in fv.iter().enumerate() {
            mat[[i, j]] = v;
        }
    }
    Ok(mat)
}

/// Construct a dense edge-feature matrix from an attributed graph.
///
/// `feature_fn` is called for each edge in insertion order and must return a
/// `Vec<f64>` of the same length for every edge.  The resulting matrix has
/// shape `(edge_count, feature_dim)`.
///
/// # Errors
///
/// Returns [`GraphError::InvalidParameter`] if the edge count is zero or if
/// `feature_fn` returns vectors of inconsistent length.
pub fn edge_feature_matrix<N, E, F>(
    graph: &AttributedGraph<N, E>,
    feature_fn: F,
) -> Result<Array2<f64>>
where
    F: Fn(&E) -> Vec<f64>,
{
    let m = graph.edge_count();
    if m == 0 {
        return Err(GraphError::invalid_parameter(
            "graph",
            "graph has no edges",
            "at least one edge",
        ));
    }

    let features: Vec<Vec<f64>> = graph.edges.iter().map(|(_, _, e)| feature_fn(e)).collect();

    let dim = features[0].len();
    if dim == 0 {
        return Err(GraphError::invalid_parameter(
            "feature_fn",
            "zero-length feature vector",
            "non-empty feature vector",
        ));
    }

    for (i, fv) in features.iter().enumerate() {
        if fv.len() != dim {
            return Err(GraphError::InvalidParameter {
                param: "feature_fn".to_string(),
                value: format!("edge {i} returned dim={}", fv.len()),
                expected: format!("uniform dim={dim}"),
                context: "edge_feature_matrix".to_string(),
            });
        }
    }

    let mut mat = Array2::zeros((m, dim));
    for (i, fv) in features.iter().enumerate() {
        for (j, &v) in fv.iter().enumerate() {
            mat[[i, j]] = v;
        }
    }
    Ok(mat)
}

/// Return the out-neighbours of `node` together with their node data and the
/// connecting edge data.
///
/// # Errors
///
/// Returns [`GraphError::NodeNotFound`] if `node` is not in the graph.
///
/// # Example
///
/// ```
/// use scirs2_graph::attributed_graph::{AttributedGraph, attributed_neighbors};
///
/// let mut g = AttributedGraph::<&str, f64>::new();
/// let a = g.add_node("Alice");
/// let b = g.add_node("Bob");
/// g.add_edge(a, b, 1.5).unwrap();
///
/// let nbrs = attributed_neighbors(a, &g).unwrap();
/// assert_eq!(nbrs.len(), 1);
/// assert_eq!(*nbrs[0].node_data, "Bob");
/// assert_eq!(*nbrs[0].edge_data, 1.5);
/// ```
pub fn attributed_neighbors<'a, N, E>(
    node: NodeId,
    graph: &'a AttributedGraph<N, E>,
) -> Result<Vec<NeighborInfo<'a, N, E>>> {
    graph.validate_node(node)?;
    let result = graph.adj[node.0]
        .iter()
        .map(|&(dst, eidx)| NeighborInfo {
            id: NodeId(dst),
            node_data: &graph.nodes[dst],
            edge_data: &graph.edges[eidx].2,
        })
        .collect();
    Ok(result)
}

// ============================================================================
// Dijkstra
// ============================================================================

/// Priority queue entry used internally by Dijkstra.
#[derive(Debug, Clone, PartialEq)]
struct DijkstraEntry {
    cost: f64,
    node: usize,
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed for min-heap behaviour with BinaryHeap (max-heap)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Generalised Dijkstra shortest-path algorithm on an attributed graph.
///
/// `cost_fn` is called for each edge and must return a non-negative `f64`
/// cost.  Negative costs will produce incorrect results (no check is
/// performed for performance reasons).
///
/// Returns a map from each reachable [`NodeId`] to its minimum total cost
/// from `source`.  Unreachable nodes are absent from the map.
///
/// # Errors
///
/// Returns [`GraphError::NodeNotFound`] if `source` is not in the graph.
///
/// # Example
///
/// ```
/// use scirs2_graph::attributed_graph::{AttributedGraph, dijkstra_attributed, NodeId};
///
/// let mut g = AttributedGraph::<(), f64>::new();
/// let a = g.add_node(());
/// let b = g.add_node(());
/// let c = g.add_node(());
/// g.add_edge(a, b, 2.0).unwrap();
/// g.add_edge(b, c, 3.0).unwrap();
/// g.add_edge(a, c, 10.0).unwrap();
///
/// let dist = dijkstra_attributed(&g, a, |e| *e).unwrap();
/// assert_eq!(dist[&b], 2.0);
/// assert_eq!(dist[&c], 5.0); // 2 + 3, not 10
/// ```
pub fn dijkstra_attributed<N, E, F>(
    graph: &AttributedGraph<N, E>,
    source: NodeId,
    cost_fn: F,
) -> Result<HashMap<NodeId, f64>>
where
    F: Fn(&E) -> f64,
{
    use std::collections::BinaryHeap;

    graph.validate_node(source)?;

    let n = graph.node_count();
    let mut dist = vec![f64::INFINITY; n];
    dist[source.0] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(DijkstraEntry {
        cost: 0.0,
        node: source.0,
    });

    while let Some(DijkstraEntry { cost, node }) = heap.pop() {
        // Skip stale entries
        if cost > dist[node] {
            continue;
        }

        for &(dst, eidx) in &graph.adj[node] {
            let edge_cost = cost_fn(&graph.edges[eidx].2);
            let new_cost = cost + edge_cost;
            if new_cost < dist[dst] {
                dist[dst] = new_cost;
                heap.push(DijkstraEntry {
                    cost: new_cost,
                    node: dst,
                });
            }
        }
    }

    let result = dist
        .into_iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .map(|(i, d)| (NodeId(i), d))
        .collect();

    Ok(result)
}

// ============================================================================
// Additional graph analysis functions
// ============================================================================

/// Compute the in-degree of each node.
///
/// Returns a `Vec<usize>` of length `graph.node_count()` where entry `i` is
/// the number of edges *targeting* node `i`.
pub fn in_degrees<N, E>(graph: &AttributedGraph<N, E>) -> Vec<usize> {
    let mut deg = vec![0usize; graph.node_count()];
    for (_, dst, _) in &graph.edges {
        deg[*dst] += 1;
    }
    deg
}

/// Compute the out-degree of each node.
///
/// Returns a `Vec<usize>` of length `graph.node_count()`.
pub fn out_degrees<N, E>(graph: &AttributedGraph<N, E>) -> Vec<usize> {
    graph.adj.iter().map(|nbrs| nbrs.len()).collect()
}

/// Filter nodes by a predicate on their data, returning matching [`NodeId`]s.
///
/// # Example
///
/// ```
/// use scirs2_graph::attributed_graph::{AttributedGraph, filter_nodes};
///
/// let mut g = AttributedGraph::<i32, ()>::new();
/// g.add_node(10);
/// g.add_node(20);
/// g.add_node(30);
///
/// let big = filter_nodes(&g, |v| *v > 15);
/// assert_eq!(big.len(), 2);
/// ```
pub fn filter_nodes<N, E, P>(graph: &AttributedGraph<N, E>, predicate: P) -> Vec<NodeId>
where
    P: Fn(&N) -> bool,
{
    graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| if predicate(n) { Some(NodeId(i)) } else { None })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    #[derive(Debug, Clone, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Rel {
        weight: f64,
    }

    fn make_graph() -> AttributedGraph<Person, Rel> {
        let mut g = AttributedGraph::new();
        let alice = g.add_node(Person {
            name: "Alice".into(),
            age: 30,
        });
        let bob = g.add_node(Person {
            name: "Bob".into(),
            age: 25,
        });
        let charlie = g.add_node(Person {
            name: "Charlie".into(),
            age: 35,
        });
        g.add_edge(alice, bob, Rel { weight: 1.0 }).unwrap();
        g.add_edge(bob, charlie, Rel { weight: 2.0 }).unwrap();
        g.add_edge(alice, charlie, Rel { weight: 5.0 }).unwrap();
        g
    }

    // ------------------------------------------------------------------
    // Basic construction
    // ------------------------------------------------------------------

    #[test]
    fn test_basic_construction() {
        let g = make_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_node_data_access() {
        let g = make_graph();
        let alice = NodeId(0);
        let data = g.node_data(alice).unwrap();
        assert_eq!(data.name, "Alice");
        assert_eq!(data.age, 30);
    }

    #[test]
    fn test_edge_data_access() {
        let g = make_graph();
        let alice = NodeId(0);
        let bob = NodeId(1);
        let ed = g.edge_data(alice, bob).unwrap();
        assert!((ed.weight - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_has_edge() {
        let g = make_graph();
        assert!(g.has_edge(NodeId(0), NodeId(1)));
        assert!(!g.has_edge(NodeId(1), NodeId(0)));
        assert!(!g.has_edge(NodeId(2), NodeId(0)));
    }

    #[test]
    fn test_invalid_node_add_edge() {
        let mut g = AttributedGraph::<i32, ()>::new();
        g.add_node(1);
        let err = g.add_edge(NodeId(0), NodeId(5), ());
        assert!(err.is_err());
    }

    // ------------------------------------------------------------------
    // Builder
    // ------------------------------------------------------------------

    #[test]
    fn test_builder() {
        let mut b = AttributedGraphBuilder::<i32, f64>::new();
        let a = b.node(1);
        let bb = b.node(2);
        b.edge(a, bb, std::f64::consts::PI);
        let g = b.build().unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_builder_bad_edge_deferred() {
        let mut b = AttributedGraphBuilder::<i32, f64>::new();
        b.node(1);
        // NodeId(99) doesn't exist
        b.edge(NodeId(0), NodeId(99), 1.0);
        assert!(b.build().is_err());
    }

    // ------------------------------------------------------------------
    // Feature matrices
    // ------------------------------------------------------------------

    #[test]
    fn test_node_feature_matrix() {
        let g = make_graph();
        let mat = node_feature_matrix(&g, |p| vec![p.age as f64]).unwrap();
        assert_eq!(mat.shape(), &[3, 1]);
        assert!((mat[[0, 0]] - 30.0).abs() < 1e-12);
        assert!((mat[[1, 0]] - 25.0).abs() < 1e-12);
        assert!((mat[[2, 0]] - 35.0).abs() < 1e-12);
    }

    #[test]
    fn test_edge_feature_matrix() {
        let g = make_graph();
        let mat = edge_feature_matrix(&g, |r| vec![r.weight]).unwrap();
        assert_eq!(mat.shape(), &[3, 1]);
        // Edges in insertion order: Alice→Bob(1.0), Bob→Charlie(2.0), Alice→Charlie(5.0)
        assert!((mat[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((mat[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((mat[[2, 0]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_node_feature_matrix_multi_dim() {
        let g = make_graph();
        let mat = node_feature_matrix(&g, |p| vec![p.age as f64, p.name.len() as f64]).unwrap();
        assert_eq!(mat.shape(), &[3, 2]);
    }

    #[test]
    fn test_node_feature_matrix_empty_graph() {
        let g = AttributedGraph::<i32, ()>::new();
        let result = node_feature_matrix(&g, |v| vec![*v as f64]);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_feature_matrix_no_edges() {
        let mut g = AttributedGraph::<i32, f64>::new();
        g.add_node(1);
        let result = edge_feature_matrix(&g, |v| vec![*v]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // Attributed neighbours
    // ------------------------------------------------------------------

    #[test]
    fn test_attributed_neighbors() {
        let g = make_graph();
        let nbrs = attributed_neighbors(NodeId(0), &g).unwrap();
        assert_eq!(nbrs.len(), 2);
        // Bob
        assert_eq!(nbrs[0].node_data.name, "Bob");
        assert!((nbrs[0].edge_data.weight - 1.0).abs() < 1e-12);
        // Charlie (via direct edge)
        assert_eq!(nbrs[1].node_data.name, "Charlie");
        assert!((nbrs[1].edge_data.weight - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_attributed_neighbors_unknown_node() {
        let g = make_graph();
        assert!(attributed_neighbors(NodeId(99), &g).is_err());
    }

    // ------------------------------------------------------------------
    // Dijkstra
    // ------------------------------------------------------------------

    #[test]
    fn test_dijkstra_simple() {
        let g = make_graph();
        // Alice=0, Bob=1, Charlie=2
        // 0→1 (1.0), 1→2 (2.0), 0→2 (5.0)
        let dist = dijkstra_attributed(&g, NodeId(0), |r| r.weight).unwrap();
        assert!((dist[&NodeId(0)] - 0.0).abs() < 1e-12);
        assert!((dist[&NodeId(1)] - 1.0).abs() < 1e-12);
        assert!((dist[&NodeId(2)] - 3.0).abs() < 1e-12); // 1+2 beats direct 5
    }

    #[test]
    fn test_dijkstra_unreachable() {
        let mut g = AttributedGraph::<(), f64>::new();
        g.add_node(());
        g.add_node(());
        // No edges; node 1 is unreachable from node 0
        let dist = dijkstra_attributed(&g, NodeId(0), |e| *e).unwrap();
        assert!(dist.contains_key(&NodeId(0)));
        assert!(!dist.contains_key(&NodeId(1)));
    }

    #[test]
    fn test_dijkstra_invalid_source() {
        let g = make_graph();
        assert!(dijkstra_attributed(&g, NodeId(99), |r| r.weight).is_err());
    }

    #[test]
    fn test_dijkstra_single_node() {
        let mut g = AttributedGraph::<(), ()>::new();
        g.add_node(());
        let dist = dijkstra_attributed(&g, NodeId(0), |_| 1.0).unwrap();
        assert_eq!(dist.len(), 1);
        assert_eq!(dist[&NodeId(0)], 0.0);
    }

    // ------------------------------------------------------------------
    // Utility functions
    // ------------------------------------------------------------------

    #[test]
    fn test_in_out_degrees() {
        let g = make_graph();
        let out = out_degrees(&g);
        // Alice: 2, Bob: 1, Charlie: 0
        assert_eq!(out[0], 2);
        assert_eq!(out[1], 1);
        assert_eq!(out[2], 0);

        let inn = in_degrees(&g);
        // Alice: 0, Bob: 1, Charlie: 2
        assert_eq!(inn[0], 0);
        assert_eq!(inn[1], 1);
        assert_eq!(inn[2], 2);
    }

    #[test]
    fn test_filter_nodes() {
        let g = make_graph();
        let young = filter_nodes(&g, |p| p.age < 31);
        // Alice(30), Bob(25)
        assert_eq!(young.len(), 2);
    }

    #[test]
    fn test_nodes_iterator() {
        let g = make_graph();
        let names: Vec<&str> = g.nodes().map(|(_, p)| p.name.as_str()).collect();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
    }

    #[test]
    fn test_edges_iterator() {
        let g = make_graph();
        let edges: Vec<(NodeId, NodeId, f64)> =
            g.edges_iter().map(|(s, d, e)| (s, d, e.weight)).collect();
        assert_eq!(edges.len(), 3);
    }
}
