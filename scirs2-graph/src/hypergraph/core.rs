//! Core hypergraph data structures and expansions.
//!
//! Provides two complementary representations:
//!
//! * [`IndexedHypergraph`] – usize-indexed, weight-carrying structure used by
//!   spectral and algorithmic routines (the original workhorse).
//! * [`Hypergraph`] – generic `<N, E>` structure whose nodes carry arbitrary
//!   data and whose hyperedges carry metadata of type `E`.
//!
//! Both expose clique expansion, star expansion, and bipartite representation.

use crate::base::Hypergraph as BaseHypergraph;
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Hyperedge (used by IndexedHypergraph)
// ============================================================================

/// A hyperedge in an indexed hypergraph.
///
/// Nodes within a hyperedge are stored in **sorted order** so that equality
/// checks and set operations are efficient.
#[derive(Debug, Clone, PartialEq)]
pub struct Hyperedge {
    /// Unique identifier (sequential, assigned at insertion time)
    pub id: usize,
    /// Nodes belonging to this hyperedge (sorted, deduplicated)
    pub nodes: Vec<usize>,
    /// Non-negative weight (default 1.0)
    pub weight: f64,
}

impl Hyperedge {
    /// Create a new hyperedge with automatic deduplication and sorting.
    pub fn new(id: usize, mut nodes: Vec<usize>, weight: f64) -> Self {
        nodes.sort_unstable();
        nodes.dedup();
        Hyperedge { id, nodes, weight }
    }

    /// Cardinality (number of nodes in this hyperedge).
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Check whether `node` is a member of this hyperedge.
    pub fn contains(&self, node: usize) -> bool {
        self.nodes.binary_search(&node).is_ok()
    }

    /// Number of nodes shared with another hyperedge.
    pub fn intersection_size(&self, other: &Hyperedge) -> usize {
        let mut i = 0;
        let mut j = 0;
        let mut count = 0;
        while i < self.nodes.len() && j < other.nodes.len() {
            match self.nodes[i].cmp(&other.nodes[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        count
    }
}

// ============================================================================
// IndexedHypergraph – usize-indexed, algorithm-friendly
// ============================================================================

/// A hypergraph with usize-indexed nodes and rich algorithmic support.
///
/// ## Example
/// ```
/// use scirs2_graph::hypergraph::{IndexedHypergraph, clique_expansion};
///
/// let mut hg = IndexedHypergraph::new(5);
/// hg.add_hyperedge(vec![0, 1, 2], 1.0).unwrap();
/// hg.add_hyperedge(vec![2, 3, 4], 1.0).unwrap();
///
/// let g = clique_expansion(&hg);
/// assert!(g.edge_count() >= 3);
/// ```
#[derive(Debug, Clone)]
pub struct IndexedHypergraph {
    /// Total number of nodes (fixed at construction time)
    n_nodes: usize,
    /// All hyperedges in insertion order
    hyperedges: Vec<Hyperedge>,
    /// Inverted index: node → list of hyperedge ids containing that node
    node_to_hyperedges: Vec<Vec<usize>>,
}

impl IndexedHypergraph {
    /// Create an empty hypergraph with `n_nodes` nodes indexed `0..n_nodes`.
    pub fn new(n_nodes: usize) -> Self {
        IndexedHypergraph {
            n_nodes,
            hyperedges: Vec::new(),
            node_to_hyperedges: vec![Vec::new(); n_nodes],
        }
    }

    /// Number of nodes (constant after construction).
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of hyperedges.
    pub fn n_hyperedges(&self) -> usize {
        self.hyperedges.len()
    }

    /// Immutable reference to all hyperedges.
    pub fn hyperedges(&self) -> &[Hyperedge] {
        &self.hyperedges
    }

    /// Add a new hyperedge.  Nodes that exceed `n_nodes` are rejected.
    ///
    /// Returns the newly assigned hyperedge id.
    pub fn add_hyperedge(&mut self, nodes: Vec<usize>, weight: f64) -> Result<usize> {
        if weight < 0.0 {
            return Err(GraphError::InvalidGraph(
                "hyperedge weight must be non-negative".to_string(),
            ));
        }
        for &n in &nodes {
            if n >= self.n_nodes {
                return Err(GraphError::InvalidGraph(format!(
                    "node {n} out of range (n_nodes = {})",
                    self.n_nodes
                )));
            }
        }
        let id = self.hyperedges.len();
        let he = Hyperedge::new(id, nodes, weight);
        for &n in &he.nodes {
            self.node_to_hyperedges[n].push(id);
        }
        self.hyperedges.push(he);
        Ok(id)
    }

    /// Return all hyperedge ids that contain `node`.
    pub fn hyperedges_of_node(&self, node: usize) -> Option<&[usize]> {
        if node < self.n_nodes {
            Some(&self.node_to_hyperedges[node])
        } else {
            None
        }
    }

    /// Degree of a node = number of hyperedges it belongs to (unweighted).
    pub fn degree(&self, node: usize) -> usize {
        self.node_to_hyperedges
            .get(node)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Weighted degree of a node = sum of weights of incident hyperedges.
    pub fn weighted_degree(&self, node: usize) -> f64 {
        self.node_to_hyperedges
            .get(node)
            .map(|ids| ids.iter().map(|&id| self.hyperedges[id].weight).sum())
            .unwrap_or(0.0)
    }

    /// Compute the **node–hyperedge incidence matrix** B of shape `(n_nodes × n_hyperedges)`.
    ///
    /// `B[i, e] = sqrt(w_e)` when node `i` is in hyperedge `e`, else `0`.
    pub fn incidence_matrix(&self) -> Array2<f64> {
        let m = self.n_nodes;
        let e = self.hyperedges.len();
        let mut b = Array2::<f64>::zeros((m, e));
        for (eid, he) in self.hyperedges.iter().enumerate() {
            let sw = he.weight.sqrt();
            for &n in &he.nodes {
                b[[n, eid]] = sw;
            }
        }
        b
    }

    /// Compute the raw (unweighted) incidence matrix B where `B[i,e] ∈ {0,1}`.
    pub fn incidence_matrix_binary(&self) -> Array2<f64> {
        let m = self.n_nodes;
        let e = self.hyperedges.len();
        let mut b = Array2::<f64>::zeros((m, e));
        for (eid, he) in self.hyperedges.iter().enumerate() {
            for &n in &he.nodes {
                b[[n, eid]] = 1.0;
            }
        }
        b
    }

    /// Build the **star expansion** of this hypergraph.
    ///
    /// Each hyperedge `e` is replaced by a new auxiliary node (indexed
    /// `n_nodes + e`).  The auxiliary node is connected to every member of
    /// `e` with weight `w_e`.  The resulting graph has `n_nodes + n_hyperedges`
    /// nodes.
    pub fn star_expansion(&self) -> Graph<usize, f64> {
        let total = self.n_nodes + self.hyperedges.len();
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..total {
            g.add_node(i);
        }
        for (eid, he) in self.hyperedges.iter().enumerate() {
            let aux = self.n_nodes + eid;
            for &n in &he.nodes {
                let _ = g.add_edge(n, aux, he.weight);
            }
        }
        g
    }

    /// Build the **bipartite representation** of this hypergraph.
    ///
    /// Returns a hypergraph where the left partition contains original
    /// nodes (indices 0..n_nodes) and the right partition contains hyperedge
    /// auxiliary nodes (indices n_nodes..n_nodes+n_hyperedges).  Each
    /// 2-element hyperedge {original_node, aux_hyperedge_node} encodes a
    /// membership relation.
    pub fn bipartite_representation(&self) -> BaseHypergraph<usize, f64> {
        let mut bg: BaseHypergraph<usize, f64> = BaseHypergraph::new();
        // Add original nodes
        for i in 0..self.n_nodes {
            bg.add_node(i);
        }
        // Add hyperedge auxiliary nodes and membership hyperedges
        for (eid, he) in self.hyperedges.iter().enumerate() {
            let he_node = self.n_nodes + eid;
            bg.add_node(he_node);
            for &n in &he.nodes {
                let _ = bg.add_hyperedge_from_vec(vec![n, he_node], he.weight);
            }
        }
        bg
    }
}

// ============================================================================
// Generic Hypergraph<N, E>
// ============================================================================

/// Generic hypergraph whose nodes carry data of type `N` and hyperedges carry
/// metadata of type `E`.
///
/// Node and hyperedge indices are plain `usize` values assigned at insertion
/// time (monotonically increasing).
///
/// ## Example
/// ```
/// use scirs2_graph::hypergraph::Hypergraph;
///
/// let mut hg: Hypergraph<&str, &str> = Hypergraph::new();
/// let a = hg.add_node("Alice");
/// let b = hg.add_node("Bob");
/// let c = hg.add_node("Carol");
/// let e = hg.add_hyperedge("team", vec![a, b, c]);
/// assert_eq!(hg.node_degree(a), 1);
/// assert_eq!(hg.incident_edges(b), vec![e]);
/// ```
#[derive(Debug, Clone)]
pub struct Hypergraph<N, E> {
    /// Node data store (index = node id)
    nodes: Vec<N>,
    /// Hyperedge store: (metadata, sorted node indices)
    hyperedges: Vec<(E, Vec<usize>)>,
    /// Inverted index: node id → hyperedge ids
    node_to_edges: Vec<Vec<usize>>,
}

impl<N: Clone, E: Clone> Default for Hypergraph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Clone, E: Clone> Hypergraph<N, E> {
    /// Create an empty generic hypergraph.
    pub fn new() -> Self {
        Hypergraph {
            nodes: Vec::new(),
            hyperedges: Vec::new(),
            node_to_edges: Vec::new(),
        }
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, data: N) -> usize {
        let id = self.nodes.len();
        self.nodes.push(data);
        self.node_to_edges.push(Vec::new());
        id
    }

    /// Add a hyperedge and return its index.
    ///
    /// Duplicate nodes within `nodes` are silently deduplicated; order is
    /// normalised to sorted.  Panics if any node index is out of range (use
    /// [`try_add_hyperedge`] for the fallible version).
    pub fn add_hyperedge(&mut self, data: E, mut nodes: Vec<usize>) -> usize {
        nodes.sort_unstable();
        nodes.dedup();
        let id = self.hyperedges.len();
        for &n in &nodes {
            if n < self.node_to_edges.len() {
                self.node_to_edges[n].push(id);
            }
        }
        self.hyperedges.push((data, nodes));
        id
    }

    /// Fallible version of [`add_hyperedge`] that returns an error when a node
    /// index is out of range.
    pub fn try_add_hyperedge(&mut self, data: E, mut nodes: Vec<usize>) -> Result<usize> {
        nodes.sort_unstable();
        nodes.dedup();
        for &n in &nodes {
            if n >= self.nodes.len() {
                return Err(GraphError::InvalidGraph(format!(
                    "node index {n} out of range (n_nodes = {})",
                    self.nodes.len()
                )));
            }
        }
        let id = self.hyperedges.len();
        for &n in &nodes {
            self.node_to_edges[n].push(id);
        }
        self.hyperedges.push((data, nodes));
        Ok(id)
    }

    /// Return the ids of all hyperedges incident to `node`.
    pub fn incident_edges(&self, node: usize) -> Vec<usize> {
        self.node_to_edges.get(node).cloned().unwrap_or_default()
    }

    /// Return the degree (number of incident hyperedges) of a node.
    pub fn node_degree(&self, node: usize) -> usize {
        self.node_to_edges.get(node).map(|v| v.len()).unwrap_or(0)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of hyperedges.
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedges.len()
    }

    /// Access node data by index.
    pub fn node_data(&self, id: usize) -> Option<&N> {
        self.nodes.get(id)
    }

    /// Access hyperedge data and members by index.
    pub fn hyperedge_data(&self, id: usize) -> Option<(&E, &[usize])> {
        self.hyperedges.get(id).map(|(e, ns)| (e, ns.as_slice()))
    }

    /// Return the member node indices of hyperedge `id`.
    pub fn hyperedge_nodes(&self, id: usize) -> Option<&[usize]> {
        self.hyperedges.get(id).map(|(_, ns)| ns.as_slice())
    }
}

impl<N: Clone + std::fmt::Debug + std::hash::Hash + Eq + Ord, E: Clone> Hypergraph<N, E> {
    /// Build the **clique expansion** (2-section) of this hypergraph as a
    /// `Graph<usize, f64>` where nodes are identified by their index.
    ///
    /// Edge weight between `u` and `v` accumulates across all hyperedges that
    /// contain both.
    pub fn clique_expansion_indexed(&self) -> Graph<usize, f64> {
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..self.nodes.len() {
            g.add_node(i);
        }
        let mut weights: HashMap<(usize, usize), f64> = HashMap::new();
        for (_, nodes) in &self.hyperedges {
            let k = nodes.len();
            if k < 2 {
                continue;
            }
            let contrib = 1.0 / (k - 1) as f64;
            for i in 0..k {
                for j in (i + 1)..k {
                    *weights.entry((nodes[i], nodes[j])).or_insert(0.0) += contrib;
                }
            }
        }
        for ((u, v), w) in weights {
            let _ = g.add_edge(u, v, w);
        }
        g
    }
}

// ============================================================================
// clique_expansion (free function operating on IndexedHypergraph)
// ============================================================================

/// Build the **2-section graph** (clique expansion) of an [`IndexedHypergraph`].
///
/// For every hyperedge `e = {v_1, …, v_k, weight w_e}`, add a clique edge
/// `(v_i, v_j)` with weight `w_e / (k - 1)`.  When multiple hyperedges
/// connect the same pair their contributions are summed.
pub fn clique_expansion(hg: &IndexedHypergraph) -> Graph<usize, f64> {
    let mut graph: Graph<usize, f64> = Graph::new();
    for i in 0..hg.n_nodes() {
        graph.add_node(i);
    }
    let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
    for he in &hg.hyperedges {
        let k = he.nodes.len();
        if k < 2 {
            continue;
        }
        let contrib = he.weight / (k - 1) as f64;
        for i in 0..k {
            for j in (i + 1)..k {
                let key = (he.nodes[i], he.nodes[j]);
                *edge_weights.entry(key).or_insert(0.0) += contrib;
            }
        }
    }
    for ((u, v), w) in edge_weights {
        let _ = graph.add_edge(u, v, w);
    }
    graph
}

// ============================================================================
// line_graph
// ============================================================================

/// Build the **line graph** of an [`IndexedHypergraph`].
///
/// Each hyperedge becomes a node.  Two hyperedge-nodes are connected iff their
/// corresponding hyperedges share at least one node; the edge weight equals the
/// number of shared nodes.
pub fn line_graph(hg: &IndexedHypergraph) -> Graph<usize, f64> {
    let e = hg.n_hyperedges();
    let mut graph: Graph<usize, f64> = Graph::new();
    for i in 0..e {
        graph.add_node(i);
    }
    for i in 0..e {
        for j in (i + 1)..e {
            let shared = hg.hyperedges[i].intersection_size(&hg.hyperedges[j]);
            if shared > 0 {
                let _ = graph.add_edge(i, j, shared as f64);
            }
        }
    }
    graph
}

// ============================================================================
// hypergraph_random_walk
// ============================================================================

/// Perform a random walk on an [`IndexedHypergraph`].
///
/// At each step the walker at node `v`:
/// 1. Selects a hyperedge `e` containing `v` with probability ∝ `w_e / deg_w(v)`.
/// 2. Moves uniformly to one of the other nodes in `e` (stays if singleton).
///
/// Returns the visited node sequence of length `n_steps + 1`.
pub fn hypergraph_random_walk<R: Rng>(
    hg: &IndexedHypergraph,
    start: usize,
    n_steps: usize,
    rng: &mut R,
) -> Result<Vec<usize>> {
    if hg.n_nodes() == 0 {
        return Err(GraphError::InvalidGraph(
            "hypergraph has no nodes".to_string(),
        ));
    }
    if start >= hg.n_nodes() {
        return Err(GraphError::InvalidGraph(format!(
            "start node {start} >= n_nodes {}",
            hg.n_nodes()
        )));
    }
    let mut path = Vec::with_capacity(n_steps + 1);
    let mut current = start;
    path.push(current);

    for _ in 0..n_steps {
        let ids = match hg.hyperedges_of_node(current) {
            Some(ids) if !ids.is_empty() => ids,
            _ => {
                path.push(current);
                continue;
            }
        };
        let total_w: f64 = ids.iter().map(|&id| hg.hyperedges[id].weight).sum();
        let threshold = rng.random::<f64>() * total_w;
        let mut accum = 0.0;
        // Safety: ids is non-empty so last() always returns Some
        let mut chosen_id = ids[ids.len() - 1];
        for &id in ids {
            accum += hg.hyperedges[id].weight;
            if accum >= threshold {
                chosen_id = id;
                break;
            }
        }
        let he = &hg.hyperedges[chosen_id];
        let candidates: Vec<usize> = he.nodes.iter().copied().filter(|&n| n != current).collect();
        if candidates.is_empty() {
            path.push(current);
        } else {
            let idx = rng.random_range(0..candidates.len());
            current = candidates[idx];
            path.push(current);
        }
    }
    Ok(path)
}

/// Convenience wrapper: run [`hypergraph_random_walk`] with a seeded RNG.
pub fn hypergraph_random_walk_seeded(
    hg: &IndexedHypergraph,
    start: usize,
    n_steps: usize,
    seed: u64,
) -> Result<Vec<usize>> {
    use scirs2_core::random::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    hypergraph_random_walk(hg, start, n_steps, &mut rng)
}

// ============================================================================
// hyperedge_centrality
// ============================================================================

/// Compute **hyperedge centrality** via power iteration on `B^T D_v^{-1} B D_e^{-1}`.
///
/// Returns a vector of length `n_hyperedges` normalised to sum to 1.
pub fn hyperedge_centrality(hg: &IndexedHypergraph) -> Vec<f64> {
    let m = hg.n_nodes();
    let e = hg.n_hyperedges();
    if e == 0 || m == 0 {
        return Vec::new();
    }
    let dv: Vec<f64> = (0..m).map(|i| hg.weighted_degree(i)).collect();
    let de: Vec<f64> = hg.hyperedges.iter().map(|h| h.nodes.len() as f64).collect();

    let mut c = vec![1.0 / e as f64; e];
    let max_iter = 1000;
    let tol = 1e-9;

    for _ in 0..max_iter {
        let x: Vec<f64> = c
            .iter()
            .enumerate()
            .map(|(eid, &cv)| if de[eid] > 0.0 { cv / de[eid] } else { 0.0 })
            .collect();
        let mut y = vec![0.0f64; m];
        for (eid, he) in hg.hyperedges.iter().enumerate() {
            let sw = he.weight.sqrt();
            for &n in &he.nodes {
                y[n] += sw * x[eid];
            }
        }
        let z: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &yi)| if dv[i] > 0.0 { yi / dv[i] } else { 0.0 })
            .collect();
        let mut c_new = vec![0.0f64; e];
        for (eid, he) in hg.hyperedges.iter().enumerate() {
            let sw = he.weight.sqrt();
            for &n in &he.nodes {
                c_new[eid] += sw * z[n];
            }
        }
        let norm: f64 = c_new.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return vec![0.0; e];
        }
        for v in &mut c_new {
            *v /= norm;
        }
        let diff: f64 = c_new.iter().zip(c.iter()).map(|(a, b)| (a - b).abs()).sum();
        c = c_new;
        if diff < tol {
            break;
        }
    }

    let total: f64 = c.iter().map(|v| v.abs()).sum();
    if total > 0.0 {
        c.iter().map(|v| v.abs() / total).collect()
    } else {
        vec![0.0; e]
    }
}

// ============================================================================
// hypergraph_clustering_coefficient
// ============================================================================

/// Compute the **local clustering coefficient** of a node in an [`IndexedHypergraph`].
///
/// Defined as the fraction of neighbour pairs (in the 2-section) that are also
/// connected.  Returns `0.0` for isolated nodes or nodes with fewer than 2 neighbours.
pub fn hypergraph_clustering_coefficient(node: usize, hg: &IndexedHypergraph) -> f64 {
    if node >= hg.n_nodes() {
        return 0.0;
    }
    let mut neighbour_set: HashSet<usize> = HashSet::new();
    if let Some(ids) = hg.hyperedges_of_node(node) {
        for &eid in ids {
            for &n in &hg.hyperedges[eid].nodes {
                if n != node {
                    neighbour_set.insert(n);
                }
            }
        }
    }
    let neighbours: Vec<usize> = neighbour_set.into_iter().collect();
    let k = neighbours.len();
    if k < 2 {
        return 0.0;
    }
    let mut connected_pairs: usize = 0;
    let total_pairs = k * (k - 1) / 2;
    for i in 0..k {
        for j in (i + 1)..k {
            let u = neighbours[i];
            let v = neighbours[j];
            let u_edges: HashSet<usize> = hg
                .hyperedges_of_node(u)
                .unwrap_or(&[])
                .iter()
                .copied()
                .collect();
            let v_edges = hg.hyperedges_of_node(v).unwrap_or(&[]);
            if v_edges.iter().any(|id| u_edges.contains(id)) {
                connected_pairs += 1;
            }
        }
    }
    connected_pairs as f64 / total_pairs as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn triangle_hg() -> IndexedHypergraph {
        let mut hg = IndexedHypergraph::new(3);
        hg.add_hyperedge(vec![0, 1], 1.0).expect("add ok");
        hg.add_hyperedge(vec![1, 2], 1.0).expect("add ok");
        hg.add_hyperedge(vec![0, 2], 1.0).expect("add ok");
        hg
    }

    #[test]
    fn test_hyperedge_dedup() {
        let he = Hyperedge::new(0, vec![3, 1, 2, 1], 1.0);
        assert_eq!(he.nodes, vec![1, 2, 3]);
    }

    #[test]
    fn test_hyperedge_contains() {
        let he = Hyperedge::new(0, vec![0, 2, 4], 1.0);
        assert!(he.contains(0));
        assert!(!he.contains(1));
    }

    #[test]
    fn test_intersection_size() {
        let a = Hyperedge::new(0, vec![0, 1, 2], 1.0);
        let b = Hyperedge::new(1, vec![1, 2, 3], 1.0);
        assert_eq!(a.intersection_size(&b), 2);
    }

    #[test]
    fn test_add_hyperedge_invalid_node() {
        let mut hg = IndexedHypergraph::new(3);
        assert!(hg.add_hyperedge(vec![0, 5], 1.0).is_err());
    }

    #[test]
    fn test_add_hyperedge_negative_weight() {
        let mut hg = IndexedHypergraph::new(3);
        assert!(hg.add_hyperedge(vec![0, 1], -1.0).is_err());
    }

    #[test]
    fn test_degree_and_weighted_degree() {
        let mut hg = IndexedHypergraph::new(3);
        hg.add_hyperedge(vec![0, 1], 2.0).expect("ok");
        hg.add_hyperedge(vec![0, 2], 3.0).expect("ok");
        assert_eq!(hg.degree(0), 2);
        assert_relative_eq!(hg.weighted_degree(0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_incidence_matrix_shape() {
        let hg = triangle_hg();
        assert_eq!(hg.incidence_matrix().shape(), &[3, 3]);
    }

    #[test]
    fn test_clique_expansion_triangle() {
        let hg = triangle_hg();
        let g = clique_expansion(&hg);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_star_expansion_node_count() {
        let hg = triangle_hg(); // 3 nodes + 3 hyperedges
        let g = hg.star_expansion();
        assert_eq!(g.node_count(), 6);
        // Each 2-node hyperedge contributes 2 edges → 6 total
        assert_eq!(g.edge_count(), 6);
    }

    #[test]
    fn test_bipartite_representation() {
        let mut hg = IndexedHypergraph::new(3);
        hg.add_hyperedge(vec![0, 1], 1.0).expect("ok");
        let bg = hg.bipartite_representation();
        // 3 original nodes + 1 hyperedge aux node = 4
        assert_eq!(bg.node_count(), 4);
    }

    #[test]
    fn test_generic_hypergraph_degree() {
        let mut hg: Hypergraph<&str, &str> = Hypergraph::new();
        let a = hg.add_node("a");
        let b = hg.add_node("b");
        let c = hg.add_node("c");
        hg.add_hyperedge("e1", vec![a, b, c]);
        hg.add_hyperedge("e2", vec![a, b]);
        assert_eq!(hg.node_degree(a), 2);
        assert_eq!(hg.node_degree(c), 1);
    }

    #[test]
    fn test_generic_try_add_hyperedge_oob() {
        let mut hg: Hypergraph<i32, i32> = Hypergraph::new();
        hg.add_node(0);
        assert!(hg.try_add_hyperedge(99, vec![0, 5]).is_err());
    }

    #[test]
    fn test_line_graph_triangle() {
        let hg = triangle_hg();
        let lg = line_graph(&hg);
        assert_eq!(lg.node_count(), 3);
        assert_eq!(lg.edge_count(), 3);
    }

    #[test]
    fn test_random_walk_length() {
        let hg = triangle_hg();
        let path = hypergraph_random_walk_seeded(&hg, 0, 10, 99).expect("ok");
        assert_eq!(path.len(), 11);
    }

    #[test]
    fn test_hyperedge_centrality_sums_to_one() {
        let hg = triangle_hg();
        let c = hyperedge_centrality(&hg);
        let sum: f64 = c.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_clustering_coeff_full_triangle() {
        let hg = triangle_hg();
        assert_relative_eq!(
            hypergraph_clustering_coefficient(0, &hg),
            1.0,
            epsilon = 1e-10
        );
    }
}
