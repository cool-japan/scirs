//! Heterogeneous graphs with multiple node and edge types
//!
//! A *heterogeneous* (or *typed*) graph contains nodes and edges belonging to
//! distinct *types*.  Each node type represents a different entity class (e.g.
//! `"user"`, `"item"`, `"category"`) and each edge type captures a specific
//! relation between two entity classes (e.g. `"user" --buys--> "item"`).
//!
//! This representation is standard in *relational* machine learning and
//! *knowledge graphs*.  See also [`crate::knowledge_graph`] for specialised
//! KGE embedding models.
//!
//! ## Architecture
//!
//! ```text
//!   HeteroGraph
//!   ├── node_types : HashMap<String, Vec<NodeId>>
//!   └── edge_types : HashMap<HeteroEdgeType, Vec<(NodeId, NodeId)>>
//!                         (src_type, relation, dst_type)
//! ```
//!
//! Node identifiers are *global* (unique across all types in the same graph).
//! The type membership is stored separately in `node_types`.
//!
//! ## Key operations
//!
//! | Function | Description |
//! |----------|-------------|
//! [`HeteroGraph::add_node`] | Register a node under a type |
//! [`HeteroGraph::add_edge`] | Register a typed relation |
//! [`type_adjacency`] | Build a [`CsrMatrix`] for one edge type |
//! [`meta_path_adjacency`] | Chain edge types via meta-path multiplication |
//! [`hetero_message_passing`] | Aggregate neighbour representations per type |

use std::collections::{HashMap, HashSet};

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};
use crate::gnn::CsrMatrix;

// Re-export NodeId from attributed_graph for convenience
pub use crate::attributed_graph::NodeId;

// ============================================================================
// HeteroEdgeType
// ============================================================================

/// Describes a typed, directed edge as a `(source_type, relation, destination_type)` triple.
///
/// This mirrors the *canonical form* used in heterogeneous GNN literature (HAN,
/// HGT, etc.) and knowledge-graph reasoning.
///
/// # Example
///
/// ```
/// use scirs2_graph::heterogeneous::HeteroEdgeType;
///
/// let et = HeteroEdgeType::new("user", "rates", "item");
/// assert_eq!(et.relation, "rates");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HeteroEdgeType {
    /// Entity type of the source node.
    pub source_type: String,
    /// Name of the relation.
    pub relation: String,
    /// Entity type of the destination node.
    pub destination_type: String,
}

impl HeteroEdgeType {
    /// Create a new edge type.
    pub fn new(
        source_type: impl Into<String>,
        relation: impl Into<String>,
        destination_type: impl Into<String>,
    ) -> Self {
        Self {
            source_type: source_type.into(),
            relation: relation.into(),
            destination_type: destination_type.into(),
        }
    }

    /// Canonical string key `"src_type/relation/dst_type"`.
    pub fn key(&self) -> String {
        format!(
            "{}/{}/{}",
            self.source_type, self.relation, self.destination_type
        )
    }

    /// Return the reversed edge type (swaps source and destination).
    pub fn reversed(&self) -> Self {
        Self {
            source_type: self.destination_type.clone(),
            relation: format!("rev_{}", self.relation),
            destination_type: self.source_type.clone(),
        }
    }
}

impl std::fmt::Display for HeteroEdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) --{}--> ({})",
            self.source_type, self.relation, self.destination_type
        )
    }
}

// ============================================================================
// HeteroGraph
// ============================================================================

/// A heterogeneous graph with multiple node and edge types.
///
/// ## Node identifiers
///
/// All nodes share a single global [`NodeId`] namespace.  Use
/// [`HeteroGraph::add_node`] to assign a node to a particular type; the
/// returned [`NodeId`] is globally unique.
///
/// ## Edge types
///
/// Edges are grouped by [`HeteroEdgeType`].  Each group is an unordered list
/// of `(src_id, dst_id)` pairs; duplicate edges *are* allowed (useful for
/// multigraphs).
///
/// # Example
///
/// ```
/// use scirs2_graph::heterogeneous::HeteroGraph;
///
/// let mut g = HeteroGraph::new();
/// let u0 = g.add_node("user", 0).unwrap();
/// let i0 = g.add_node("item", 0).unwrap();
/// g.add_edge("user", "buys", "item", u0, i0).unwrap();
///
/// assert_eq!(g.node_count(), 2);
/// assert_eq!(g.edge_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    /// Next globally unique node id counter.
    next_node_id: usize,
    /// `node_types["user"]` → list of global NodeIds belonging to "user" type.
    node_types: HashMap<String, Vec<NodeId>>,
    /// Reverse mapping: NodeId → type name.
    node_type_of: HashMap<NodeId, String>,
    /// `edge_types[HeteroEdgeType]` → ordered list of (src_id, dst_id) pairs.
    edge_types: HashMap<HeteroEdgeType, Vec<(NodeId, NodeId)>>,
}

impl Default for HeteroGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl HeteroGraph {
    /// Create an empty heterogeneous graph.
    pub fn new() -> Self {
        Self {
            next_node_id: 0,
            node_types: HashMap::new(),
            node_type_of: HashMap::new(),
            edge_types: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Register a new node of type `type_name`.
    ///
    /// `_hint` is an optional user-supplied integer label (e.g. a database
    /// primary key); it is stored purely for user convenience and does **not**
    /// affect the global [`NodeId`] that is returned.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidParameter`] if `type_name` is empty.
    pub fn add_node(&mut self, type_name: impl Into<String>, _hint: usize) -> Result<NodeId> {
        let type_name = type_name.into();
        if type_name.is_empty() {
            return Err(GraphError::invalid_parameter(
                "type_name",
                "<empty>",
                "non-empty node type",
            ));
        }
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.node_types
            .entry(type_name.clone())
            .or_default()
            .push(id);
        self.node_type_of.insert(id, type_name);
        Ok(id)
    }

    /// Add a directed typed edge `src_id --relation--> dst_id`.
    ///
    /// Both nodes must already be present in the graph and must belong to
    /// `src_type` and `dst_type` respectively.
    ///
    /// # Errors
    ///
    /// * [`GraphError::NodeNotFound`]  – node not registered.
    /// * [`GraphError::InvalidParameter`] – node belongs to the wrong type.
    pub fn add_edge(
        &mut self,
        src_type: impl Into<String>,
        relation: impl Into<String>,
        dst_type: impl Into<String>,
        src_id: NodeId,
        dst_id: NodeId,
    ) -> Result<()> {
        let src_type = src_type.into();
        let dst_type = dst_type.into();
        let relation = relation.into();

        // Validate node existence
        let actual_src_type = self
            .node_type_of
            .get(&src_id)
            .ok_or_else(|| GraphError::node_not_found(src_id.0))?;

        let actual_dst_type = self
            .node_type_of
            .get(&dst_id)
            .ok_or_else(|| GraphError::node_not_found(dst_id.0))?;

        // Validate type consistency
        if actual_src_type != &src_type {
            return Err(GraphError::InvalidParameter {
                param: "src_type".to_string(),
                value: format!("node {} has type '{}'", src_id.0, actual_src_type),
                expected: format!("'{src_type}'"),
                context: "HeteroGraph::add_edge".to_string(),
            });
        }
        if actual_dst_type != &dst_type {
            return Err(GraphError::InvalidParameter {
                param: "dst_type".to_string(),
                value: format!("node {} has type '{}'", dst_id.0, actual_dst_type),
                expected: format!("'{dst_type}'"),
                context: "HeteroGraph::add_edge".to_string(),
            });
        }

        let et = HeteroEdgeType::new(src_type, relation, dst_type);
        self.edge_types
            .entry(et)
            .or_default()
            .push((src_id, dst_id));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Total number of registered nodes (across all types).
    pub fn node_count(&self) -> usize {
        self.next_node_id
    }

    /// Total number of registered edges (across all edge types).
    pub fn edge_count(&self) -> usize {
        self.edge_types.values().map(|v| v.len()).sum()
    }

    /// List all node types present in the graph.
    pub fn node_type_names(&self) -> Vec<&str> {
        self.node_types.keys().map(String::as_str).collect()
    }

    /// List all edge types present in the graph.
    pub fn edge_type_list(&self) -> Vec<&HeteroEdgeType> {
        self.edge_types.keys().collect()
    }

    /// Return the nodes belonging to `type_name`.
    pub fn nodes_of_type(&self, type_name: &str) -> &[NodeId] {
        self.node_types
            .get(type_name)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return the type name of a node.
    pub fn type_of(&self, node: NodeId) -> Option<&str> {
        self.node_type_of.get(&node).map(String::as_str)
    }

    /// Return all edges of a specific [`HeteroEdgeType`].
    ///
    /// Returns an empty slice if the edge type has no edges.
    pub fn edges_of_type(&self, et: &HeteroEdgeType) -> &[(NodeId, NodeId)] {
        self.edge_types.get(et).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Return the out-neighbours of `node` under a specific edge type.
    ///
    /// Complexity: O(number of edges of that type).
    pub fn out_neighbors_typed(&self, node: NodeId, et: &HeteroEdgeType) -> Vec<NodeId> {
        self.edge_types
            .get(et)
            .map(|edges| {
                edges
                    .iter()
                    .filter_map(|&(s, d)| if s == node { Some(d) } else { None })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return all (edge_type, neighbours) pairs for `node`.
    ///
    /// Useful for heterogeneous message passing.
    pub fn all_out_neighbors_typed(&self, node: NodeId) -> Vec<(&HeteroEdgeType, Vec<NodeId>)> {
        self.edge_types
            .iter()
            .filter_map(|(et, edges)| {
                let nbrs: Vec<NodeId> = edges
                    .iter()
                    .filter_map(|&(s, d)| if s == node { Some(d) } else { None })
                    .collect();
                if nbrs.is_empty() {
                    None
                } else {
                    Some((et, nbrs))
                }
            })
            .collect()
    }

    /// Check whether a node is registered in the graph.
    pub fn contains_node(&self, node: NodeId) -> bool {
        self.node_type_of.contains_key(&node)
    }

    /// Check whether a typed edge exists.
    pub fn has_typed_edge(&self, et: &HeteroEdgeType, src: NodeId, dst: NodeId) -> bool {
        self.edge_types
            .get(et)
            .map(|edges| edges.contains(&(src, dst)))
            .unwrap_or(false)
    }
}

// ============================================================================
// type_adjacency
// ============================================================================

/// Build the adjacency matrix (as a [`CsrMatrix`]) for a specific edge type.
///
/// The matrix has dimensions `(|src_nodes|, |dst_nodes|)` where the row and
/// column orderings follow the insertion order of the respective node-type
/// lists.
///
/// # Arguments
///
/// * `graph` – the heterogeneous graph.
/// * `edge_type` – which typed edge to materialise.
///
/// # Errors
///
/// Returns [`GraphError::InvalidParameter`] if the source or destination type
/// referenced by `edge_type` does not exist in the graph.
///
/// # Example
///
/// ```
/// use scirs2_graph::heterogeneous::{HeteroGraph, HeteroEdgeType, type_adjacency};
///
/// let mut g = HeteroGraph::new();
/// let u0 = g.add_node("user", 0).unwrap();
/// let i0 = g.add_node("item", 0).unwrap();
/// let i1 = g.add_node("item", 1).unwrap();
/// g.add_edge("user", "buys", "item", u0, i0).unwrap();
/// g.add_edge("user", "buys", "item", u0, i1).unwrap();
///
/// let et = HeteroEdgeType::new("user", "buys", "item");
/// let adj = type_adjacency(&g, &et).unwrap();
/// assert_eq!(adj.n_rows, 1);   // 1 user
/// assert_eq!(adj.n_cols, 2);   // 2 items
/// ```
pub fn type_adjacency(graph: &HeteroGraph, edge_type: &HeteroEdgeType) -> Result<CsrMatrix> {
    let src_nodes = graph.nodes_of_type(&edge_type.source_type);
    let dst_nodes = graph.nodes_of_type(&edge_type.destination_type);

    if src_nodes.is_empty() {
        return Err(GraphError::InvalidParameter {
            param: "edge_type.source_type".to_string(),
            value: edge_type.source_type.clone(),
            expected: "a type with at least one node".to_string(),
            context: "type_adjacency".to_string(),
        });
    }
    if dst_nodes.is_empty() {
        return Err(GraphError::InvalidParameter {
            param: "edge_type.destination_type".to_string(),
            value: edge_type.destination_type.clone(),
            expected: "a type with at least one node".to_string(),
            context: "type_adjacency".to_string(),
        });
    }

    // Build local index maps for row and column lookups
    let src_index: HashMap<NodeId, usize> = src_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();
    let dst_index: HashMap<NodeId, usize> = dst_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    let edges = graph.edges_of_type(edge_type);

    // Collect COO triples
    let mut coo: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len());
    for &(src, dst) in edges {
        let r = match src_index.get(&src) {
            Some(&i) => i,
            None => continue, // edge references node not in expected type; skip
        };
        let c = match dst_index.get(&dst) {
            Some(&j) => j,
            None => continue,
        };
        coo.push((r, c, 1.0));
    }

    CsrMatrix::from_coo(src_nodes.len(), dst_nodes.len(), &coo).map_err(|e| {
        GraphError::InvalidGraph(format!("type_adjacency CsrMatrix::from_coo failed: {e}"))
    })
}

// ============================================================================
// meta_path_adjacency
// ============================================================================

/// Compute the meta-path similarity matrix for a sequence of node types.
///
/// A *meta-path* is a sequence of node types connected by edges, e.g.
/// `["user", "item", "user"]` captures the "users who bought the same item"
/// relation.  The resulting matrix contains path counts (before row
/// normalisation) of shape `(|first_type|, |last_type|)`.
///
/// Internally, each consecutive pair of types yields an adjacency matrix which
/// is multiplied together to produce the final result.
///
/// # Arguments
///
/// * `graph` – the heterogeneous graph.
/// * `meta_path` – ordered sequence of node-type names of length ≥ 2.
///
/// # Errors
///
/// * [`GraphError::InvalidParameter`] – meta-path has fewer than 2 types.
/// * [`GraphError::InvalidParameter`] – a required edge type is absent.
/// * [`GraphError::InvalidParameter`] – no edge type connects adjacent types in the path.
///
/// # Notes
///
/// When the same source–destination type pair is connected by multiple
/// relations, the function sums over **all** of them.  If you need to select a
/// specific relation, restrict the graph to that edge type first.
///
/// # Example
///
/// ```no_run
/// use scirs2_graph::heterogeneous::{HeteroGraph, meta_path_adjacency};
///
/// let mut g = HeteroGraph::new();
/// let u0 = g.add_node("user", 0).unwrap();
/// let u1 = g.add_node("user", 1).unwrap();
/// let i0 = g.add_node("item", 0).unwrap();
/// g.add_edge("user", "buys", "item", u0, i0).unwrap();
/// g.add_edge("user", "buys", "item", u1, i0).unwrap();
///
/// // Meta-path user→item→user: both users bought item 0, so they share paths
/// let sim = meta_path_adjacency(&g, &["user", "item", "user"]).unwrap();
/// assert_eq!(sim.shape(), &[2, 2]);
/// ```
pub fn meta_path_adjacency(graph: &HeteroGraph, meta_path: &[&str]) -> Result<Array2<f64>> {
    if meta_path.len() < 2 {
        return Err(GraphError::InvalidParameter {
            param: "meta_path".to_string(),
            value: format!("length={}", meta_path.len()),
            expected: "at least 2 node types".to_string(),
            context: "meta_path_adjacency".to_string(),
        });
    }

    // Collect all edge types, grouped by (src_type, dst_type)
    let mut type_pair_edges: HashMap<(&str, &str), Vec<&HeteroEdgeType>> = HashMap::new();
    for et in graph.edge_types.keys() {
        type_pair_edges
            .entry((et.source_type.as_str(), et.destination_type.as_str()))
            .or_default()
            .push(et);
    }

    // Build the adjacency matrix for one step of the meta-path.
    // Returns a dense (n_src × n_dst) matrix, summing over all edge types
    // that connect the given type pair.
    let step_matrix = |src_type: &str, dst_type: &str| -> Result<Array2<f64>> {
        let src_nodes = graph.nodes_of_type(src_type);
        let dst_nodes = graph.nodes_of_type(dst_type);

        if src_nodes.is_empty() || dst_nodes.is_empty() {
            // Return zero matrix
            return Ok(Array2::zeros((
                src_nodes.len().max(1),
                dst_nodes.len().max(1),
            )));
        }

        let src_index: HashMap<NodeId, usize> =
            src_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let dst_index: HashMap<NodeId, usize> =
            dst_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        let mut mat = Array2::<f64>::zeros((src_nodes.len(), dst_nodes.len()));

        let ets = type_pair_edges.get(&(src_type, dst_type));
        if let Some(edge_types) = ets {
            for &et in edge_types {
                for &(s, d) in graph.edges_of_type(et) {
                    if let (Some(&r), Some(&c)) = (src_index.get(&s), dst_index.get(&d)) {
                        mat[[r, c]] += 1.0;
                    }
                }
            }
        }
        Ok(mat)
    };

    // Initial matrix for first step
    let mut result = step_matrix(meta_path[0], meta_path[1])?;

    // Multiply through remaining steps
    for window in meta_path.windows(2).skip(1) {
        let next = step_matrix(window[0], window[1])?;
        // result: (n_first × n_mid), next: (n_mid × n_next)
        // product: (n_first × n_next)
        let (r_rows, r_cols) = (result.shape()[0], result.shape()[1]);
        let (n_rows, n_cols) = (next.shape()[0], next.shape()[1]);
        if r_cols != n_rows {
            return Err(GraphError::InvalidParameter {
                param: "meta_path".to_string(),
                value: format!(
                    "dimension mismatch: {} cols vs {} rows at step",
                    r_cols, n_rows
                ),
                expected: "matching intermediate dimensions".to_string(),
                context: "meta_path_adjacency matrix multiply".to_string(),
            });
        }
        let mut product = Array2::<f64>::zeros((r_rows, n_cols));
        for i in 0..r_rows {
            for k in 0..r_cols {
                let rv = result[[i, k]];
                if rv == 0.0 {
                    continue;
                }
                for j in 0..n_cols {
                    product[[i, j]] += rv * next[[k, j]];
                }
            }
        }
        result = product;
    }

    Ok(result)
}

// ============================================================================
// Heterogeneous message passing
// ============================================================================

/// Message-passing result for a single edge type.
#[derive(Debug, Clone)]
pub struct TypedMessageResult {
    /// The edge type this result corresponds to.
    pub edge_type: HeteroEdgeType,
    /// Aggregated features for each destination node (in type-order).
    /// Shape: `(n_dst_nodes, feature_dim)`.
    pub aggregated: Array2<f64>,
}

/// Propagate and aggregate node feature vectors across all edge types in one
/// message-passing step.
///
/// For every registered edge type `(src_type, rel, dst_type)`:
///
/// 1. Look up the feature rows for all source nodes (from `node_features`).
/// 2. Aggregate (sum) incoming features for each destination node.
/// 3. Return an [`Array2<f64>`] of shape `(n_dst_nodes, feature_dim)`.
///
/// # Arguments
///
/// * `graph` – the heterogeneous graph.
/// * `node_features` – map from [`NodeId`] to a fixed-length feature vector.
///   Nodes absent from this map contribute zero vectors.
/// * `feature_dim` – length of each feature vector.
///
/// # Errors
///
/// Returns [`GraphError::InvalidParameter`] if any feature vector has a
/// different length from `feature_dim`.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use scirs2_graph::heterogeneous::{HeteroGraph, hetero_message_passing};
/// use scirs2_graph::attributed_graph::NodeId;
///
/// let mut g = HeteroGraph::new();
/// let u0 = g.add_node("user", 0).unwrap();
/// let i0 = g.add_node("item", 0).unwrap();
/// g.add_edge("user", "buys", "item", u0, i0).unwrap();
///
/// let mut features = HashMap::new();
/// features.insert(u0, vec![1.0, 2.0]);
///
/// let results = hetero_message_passing(&g, &features, 2).unwrap();
/// assert_eq!(results.len(), 1);
/// // Aggregated features for "item" node 0: [1.0, 2.0] (from u0)
/// assert!((results[0].aggregated[[0, 0]] - 1.0).abs() < 1e-9);
/// ```
pub fn hetero_message_passing(
    graph: &HeteroGraph,
    node_features: &HashMap<NodeId, Vec<f64>>,
    feature_dim: usize,
) -> Result<Vec<TypedMessageResult>> {
    // Validate all provided feature vectors
    for (nid, fv) in node_features {
        if fv.len() != feature_dim {
            return Err(GraphError::InvalidParameter {
                param: format!("node_features[{}]", nid.0),
                value: format!("len={}", fv.len()),
                expected: format!("feature_dim={feature_dim}"),
                context: "hetero_message_passing".to_string(),
            });
        }
    }

    let zero_feat = vec![0.0f64; feature_dim];
    let mut results = Vec::new();

    for (et, edges) in &graph.edge_types {
        if edges.is_empty() {
            continue;
        }

        let dst_nodes = graph.nodes_of_type(&et.destination_type);
        if dst_nodes.is_empty() {
            continue;
        }

        let dst_index: HashMap<NodeId, usize> =
            dst_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        let mut aggregated = Array2::<f64>::zeros((dst_nodes.len(), feature_dim));

        for &(src, dst) in edges {
            let feat = node_features
                .get(&src)
                .map(Vec::as_slice)
                .unwrap_or(zero_feat.as_slice());

            if let Some(&dst_row) = dst_index.get(&dst) {
                for (j, &v) in feat.iter().enumerate() {
                    aggregated[[dst_row, j]] += v;
                }
            }
        }

        results.push(TypedMessageResult {
            edge_type: et.clone(),
            aggregated,
        });
    }

    Ok(results)
}

// ============================================================================
// Utility helpers
// ============================================================================

/// Compute the degree (number of outgoing edges) for each node under a
/// specific edge type.
///
/// Returns a `HashMap<NodeId, usize>` containing only nodes with degree ≥ 1.
pub fn typed_out_degree(graph: &HeteroGraph, edge_type: &HeteroEdgeType) -> HashMap<NodeId, usize> {
    let mut deg: HashMap<NodeId, usize> = HashMap::new();
    for &(src, _dst) in graph.edges_of_type(edge_type) {
        *deg.entry(src).or_insert(0) += 1;
    }
    deg
}

/// Compute the in-degree for each node under a specific edge type.
pub fn typed_in_degree(graph: &HeteroGraph, edge_type: &HeteroEdgeType) -> HashMap<NodeId, usize> {
    let mut deg: HashMap<NodeId, usize> = HashMap::new();
    for &(_src, dst) in graph.edges_of_type(edge_type) {
        *deg.entry(dst).or_insert(0) += 1;
    }
    deg
}

/// Return all unique node types reachable from `start_type` following the
/// given edge types.
///
/// This performs a BFS over the *type graph*.
pub fn reachable_types(graph: &HeteroGraph, start_type: &str) -> HashSet<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start_type.to_string());

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current.clone()) {
            continue;
        }
        for et in graph.edge_types.keys() {
            if et.source_type == current && !visited.contains(&et.destination_type) {
                queue.push_back(et.destination_type.clone());
            }
        }
    }
    visited
}

/// Convert the heterogeneous graph to a homogeneous `Vec<(usize, usize)>` edge list
/// by stripping type information.
pub fn to_homogeneous_edge_list(graph: &HeteroGraph) -> Vec<(usize, usize)> {
    graph
        .edge_types
        .values()
        .flat_map(|edges| edges.iter().map(|&(s, d)| (s.0, d.0)))
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

    /// Build the canonical user-item-tag knowledge graph:
    ///
    /// ```text
    /// u0 --buys-->  i0 --has_tag--> t0
    /// u1 --buys-->  i0
    /// u0 --buys-->  i1 --has_tag--> t1
    /// ```
    fn make_graph() -> (HeteroGraph, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut g = HeteroGraph::new();
        let u0 = g.add_node("user", 0).unwrap();
        let u1 = g.add_node("user", 1).unwrap();
        let i0 = g.add_node("item", 0).unwrap();
        let i1 = g.add_node("item", 1).unwrap();
        let t0 = g.add_node("tag", 0).unwrap();
        g.add_edge("user", "buys", "item", u0, i0).unwrap();
        g.add_edge("user", "buys", "item", u1, i0).unwrap();
        g.add_edge("user", "buys", "item", u0, i1).unwrap();
        g.add_edge("item", "has_tag", "tag", i0, t0).unwrap();
        (g, u0, u1, i0, i1, t0)
    }

    // ------------------------------------------------------------------
    // HeteroEdgeType
    // ------------------------------------------------------------------

    #[test]
    fn test_edge_type_key() {
        let et = HeteroEdgeType::new("user", "buys", "item");
        assert_eq!(et.key(), "user/buys/item");
    }

    #[test]
    fn test_edge_type_reversed() {
        let et = HeteroEdgeType::new("user", "buys", "item");
        let rev = et.reversed();
        assert_eq!(rev.source_type, "item");
        assert_eq!(rev.relation, "rev_buys");
        assert_eq!(rev.destination_type, "user");
    }

    // ------------------------------------------------------------------
    // HeteroGraph construction
    // ------------------------------------------------------------------

    #[test]
    fn test_basic_construction() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_nodes_of_type() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        assert_eq!(g.nodes_of_type("user").len(), 2);
        assert_eq!(g.nodes_of_type("item").len(), 2);
        assert_eq!(g.nodes_of_type("tag").len(), 1);
        assert_eq!(g.nodes_of_type("nonexistent").len(), 0);
    }

    #[test]
    fn test_add_node_empty_type_fails() {
        let mut g = HeteroGraph::new();
        assert!(g.add_node("", 0).is_err());
    }

    #[test]
    fn test_add_edge_wrong_type_fails() {
        let mut g = HeteroGraph::new();
        let u0 = g.add_node("user", 0).unwrap();
        let i0 = g.add_node("item", 0).unwrap();
        // Declare wrong src type
        assert!(g.add_edge("item", "buys", "item", u0, i0).is_err());
    }

    #[test]
    fn test_add_edge_unknown_node_fails() {
        let mut g = HeteroGraph::new();
        g.add_node("user", 0).unwrap();
        let ghost = NodeId(999);
        assert!(g
            .add_edge("user", "buys", "item", ghost, NodeId(0))
            .is_err());
    }

    #[test]
    fn test_type_of_node() {
        let (g, u0, u1, i0, _i1, t0) = make_graph();
        assert_eq!(g.type_of(u0), Some("user"));
        assert_eq!(g.type_of(u1), Some("user"));
        assert_eq!(g.type_of(i0), Some("item"));
        assert_eq!(g.type_of(t0), Some("tag"));
        assert_eq!(g.type_of(NodeId(999)), None);
    }

    #[test]
    fn test_has_typed_edge() {
        let (g, u0, u1, i0, i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("user", "buys", "item");
        assert!(g.has_typed_edge(&et, u0, i0));
        assert!(g.has_typed_edge(&et, u1, i0));
        assert!(g.has_typed_edge(&et, u0, i1));
        assert!(!g.has_typed_edge(&et, u1, i1)); // this edge does not exist
    }

    #[test]
    fn test_out_neighbors_typed() {
        let (g, u0, _u1, i0, i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("user", "buys", "item");
        let mut nbrs = g.out_neighbors_typed(u0, &et);
        nbrs.sort_by_key(|n| n.0);
        assert_eq!(nbrs, vec![i0, i1]);
    }

    // ------------------------------------------------------------------
    // type_adjacency
    // ------------------------------------------------------------------

    #[test]
    fn test_type_adjacency_shape() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("user", "buys", "item");
        let adj = type_adjacency(&g, &et).unwrap();
        // 2 users × 2 items
        assert_eq!(adj.n_rows, 2);
        assert_eq!(adj.n_cols, 2);
    }

    #[test]
    fn test_type_adjacency_values() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("user", "buys", "item");
        let adj = type_adjacency(&g, &et).unwrap();

        // Convert to dense for inspection
        let mut dense = Array2::<f64>::zeros((adj.n_rows, adj.n_cols));
        for row in 0..adj.n_rows {
            let start = adj.indptr[row];
            let end = adj.indptr[row + 1];
            for idx in start..end {
                let col = adj.indices[idx];
                dense[[row, col]] += adj.data[idx];
            }
        }

        // u0 (row 0) buys i0 and i1: two 1s in row 0
        let row0_sum: f64 = dense.row(0).sum();
        assert!((row0_sum - 2.0).abs() < 1e-12);

        // u1 (row 1) only buys i0: one 1 in row 1
        let row1_sum: f64 = dense.row(1).sum();
        assert!((row1_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_type_adjacency_missing_type() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("ghost", "buys", "item");
        assert!(type_adjacency(&g, &et).is_err());
    }

    // ------------------------------------------------------------------
    // meta_path_adjacency
    // ------------------------------------------------------------------

    #[test]
    fn test_meta_path_too_short() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        assert!(meta_path_adjacency(&g, &["user"]).is_err());
    }

    #[test]
    fn test_meta_path_user_item_user() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        // user→item→user via "buys"
        // We need the reverse edge type as well
        let mut g2 = g.clone();
        // Add reverse "bought_by" edges
        let users: Vec<NodeId> = g2.nodes_of_type("user").to_vec();
        let items: Vec<NodeId> = g2.nodes_of_type("item").to_vec();
        let buys_et = HeteroEdgeType::new("user", "buys", "item");
        let buys_edges: Vec<(NodeId, NodeId)> = g2.edges_of_type(&buys_et).to_vec();
        for (src, dst) in buys_edges {
            // user → item forward already in g2
            // add item → user reverse
            if users.contains(&src) && items.contains(&dst) {
                g2.add_edge("item", "bought_by", "user", dst, src).unwrap();
            }
        }

        // Meta-path: user --buys--> item --bought_by--> user
        let sim = meta_path_adjacency(&g2, &["user", "item", "user"]).unwrap();
        assert_eq!(sim.shape()[0], 2); // 2 users
        assert_eq!(sim.shape()[1], 2); // 2 users
                                       // u0 and u1 both bought i0; so there is 1 path u0→i0→u1 and 1 path u1→i0→u0
        let u0_u1 = sim[[0, 1]];
        let u1_u0 = sim[[1, 0]];
        assert!(u0_u1 >= 1.0, "Expected at least 1 shared path, got {u0_u1}");
        assert!(u1_u0 >= 1.0, "Expected at least 1 shared path, got {u1_u0}");
    }

    #[test]
    fn test_meta_path_two_steps() {
        let (g, _u0, _u1, _i0, _i1, t0) = make_graph();
        // user→item→tag (len=3 path)
        let sim = meta_path_adjacency(&g, &["user", "item", "tag"]).unwrap();
        // 2 users, 1 tag
        assert_eq!(sim.shape(), &[2, 1]);
        // u0 bought i0 (has t0) and i1 (no tag) → 1 path to t0
        assert!((sim[[0, 0]] - 1.0).abs() < 1e-9, "u0 paths={}", sim[[0, 0]]);
        // u1 bought i0 (has t0) → 1 path to t0
        assert!((sim[[1, 0]] - 1.0).abs() < 1e-9, "u1 paths={}", sim[[1, 0]]);
        // t0's id is still referenced indirectly
        let _ = t0;
    }

    // ------------------------------------------------------------------
    // hetero_message_passing
    // ------------------------------------------------------------------

    #[test]
    fn test_hetero_message_passing_basic() {
        let (g, u0, u1, _i0, _i1, _t0) = make_graph();
        let mut feats: HashMap<NodeId, Vec<f64>> = HashMap::new();
        feats.insert(u0, vec![1.0, 0.0]);
        feats.insert(u1, vec![0.0, 1.0]);

        let results = hetero_message_passing(&g, &feats, 2).unwrap();
        // Should have one result per edge type with at least one edge
        // buys: user→item;   has_tag: item→tag
        assert!(!results.is_empty());

        // Find the "buys" result
        let buys_result = results
            .iter()
            .find(|r| r.edge_type.relation == "buys")
            .expect("should have buys result");

        // 2 item nodes; i0 received from u0 and u1, i1 only from u0
        assert_eq!(buys_result.aggregated.shape()[0], 2); // 2 items
        assert_eq!(buys_result.aggregated.shape()[1], 2); // 2 features
    }

    #[test]
    fn test_hetero_message_passing_wrong_dim() {
        let (g, u0, _u1, _i0, _i1, _t0) = make_graph();
        let mut feats: HashMap<NodeId, Vec<f64>> = HashMap::new();
        feats.insert(u0, vec![1.0, 2.0, 3.0]); // 3-dim but feature_dim=2
        assert!(hetero_message_passing(&g, &feats, 2).is_err());
    }

    // ------------------------------------------------------------------
    // Utility helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_typed_out_degree() {
        let (g, u0, u1, _i0, _i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("user", "buys", "item");
        let deg = typed_out_degree(&g, &et);
        assert_eq!(deg[&u0], 2); // u0 buys 2 items
        assert_eq!(deg[&u1], 1); // u1 buys 1 item
    }

    #[test]
    fn test_typed_in_degree() {
        let (g, _u0, _u1, i0, i1, _t0) = make_graph();
        let et = HeteroEdgeType::new("user", "buys", "item");
        let deg = typed_in_degree(&g, &et);
        assert_eq!(deg[&i0], 2); // i0 bought by 2 users
        assert_eq!(deg[&i1], 1); // i1 bought by 1 user
    }

    #[test]
    fn test_reachable_types() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        let reachable = reachable_types(&g, "user");
        assert!(reachable.contains("user"));
        assert!(reachable.contains("item"));
        assert!(reachable.contains("tag"));
        // No node type "ghost"
        assert!(!reachable.contains("ghost"));
    }

    #[test]
    fn test_to_homogeneous_edge_list() {
        let (g, _u0, _u1, _i0, _i1, _t0) = make_graph();
        let edges = to_homogeneous_edge_list(&g);
        assert_eq!(edges.len(), 4); // 3 buys + 1 has_tag
    }

    #[test]
    fn test_all_out_neighbors_typed() {
        let (g, u0, _u1, _i0, _i1, _t0) = make_graph();
        let nbrs = g.all_out_neighbors_typed(u0);
        assert_eq!(nbrs.len(), 1); // only "buys" edges from u0
        assert_eq!(nbrs[0].0.relation, "buys");
        assert_eq!(nbrs[0].1.len(), 2); // u0 buys 2 items
    }

    #[test]
    fn test_contains_node() {
        let (g, u0, _u1, _i0, _i1, _t0) = make_graph();
        assert!(g.contains_node(u0));
        assert!(!g.contains_node(NodeId(999)));
    }
}
