//! # NetworkX Migration Utilities
//!
//! This module provides a NetworkX-compatible API surface to ease migration of
//! Python scientific graph code to SciRS2. The functions here work with the
//! native [`Graph`] type but expose names and signatures that closely mirror
//! their `networkx` counterparts.
//!
//! ## Covered NetworkX Functions
//!
//! | NetworkX | SciRS2 compat | Notes |
//! |----------|---------------|-------|
//! | `nx.from_edgelist` | [`from_edge_list`] | Build from `(u, v, weight)` triples |
//! | `nx.to_numpy_array` | [`to_adjacency_matrix`] | Dense adjacency matrix |
//! | `nx.to_edgelist` | [`to_edge_list`] | Export `(source, target, weight)` |
//! | `nx.pagerank` | [`pagerank`] | Iterative power method |
//! | `nx.betweenness_centrality` | [`betweenness_centrality`] | Brandes algorithm |
//! | `nx.clustering` | [`clustering_coefficient`] | Local clustering coefficients |
//! | `nx.average_clustering` | [`global_clustering_coefficient`] | Global (average) clustering |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_graph::compat::{from_edge_list, pagerank, betweenness_centrality, clustering_coefficient};
//!
//! // Build a triangle graph from edge list
//! let edges = vec![(0u32, 1u32, 1.0f64), (1, 2, 1.0), (2, 0, 1.0)];
//! let graph = from_edge_list(&edges).expect("build graph");
//!
//! // PageRank
//! let pr = pagerank(&graph, 0.85, 100, 1e-6).expect("pagerank");
//! assert_eq!(pr.len(), 3);
//!
//! // Betweenness centrality
//! let bc = betweenness_centrality(&graph).expect("betweenness");
//! assert_eq!(bc.len(), 3);
//!
//! // Clustering coefficient
//! let cc = clustering_coefficient(&graph).expect("clustering");
//! assert!(cc.values().all(|&v| v >= 0.0 && v <= 1.0));
//! ```

use crate::base::graph::Graph;
use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

// ============================================================================
// from_edge_list
// ============================================================================

/// Build an undirected [`Graph`] from a slice of `(source, target, weight)` triples.
///
/// Equivalent to `networkx.from_edgelist(edges, create_using=nx.Graph())` with
/// numeric weights.
///
/// Duplicate edges between the same node pair are all added (the underlying
/// `petgraph` allows multi-edges; whether they appear in algorithms depends on
/// the operation).
///
/// # Errors
///
/// Returns an error if any edge operation fails internally.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::from_edge_list;
///
/// let edges = vec![(0u32, 1u32, 2.0f64), (1, 2, 3.0), (0, 2, 1.0)];
/// let g = from_edge_list(&edges).expect("build graph");
/// assert_eq!(g.node_count(), 3);
/// assert_eq!(g.edge_count(), 3);
/// ```
pub fn from_edge_list<N, E>(edges: &[(N, N, E)]) -> Result<Graph<N, E>>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
    E: Clone
        + Copy
        + PartialOrd
        + Debug
        + Send
        + Sync
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + 'static,
{
    let mut graph = Graph::<N, E>::new();
    for (src, tgt, weight) in edges {
        graph.add_edge(src.clone(), tgt.clone(), *weight)?;
    }
    Ok(graph)
}

// ============================================================================
// to_adjacency_matrix
// ============================================================================

/// Produce a dense `n × n` adjacency matrix from the graph.
///
/// The node ordering in the matrix follows the internal petgraph node-index
/// order, which is insertion order (first node added = row/column 0).
///
/// Zero-weight entries stay at zero; edges with explicit weight are placed at
/// their `[i, j]` and `[j, i]` positions (symmetric, since the graph is
/// undirected).
///
/// Equivalent to `networkx.to_numpy_array(G)`.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::{from_edge_list, to_adjacency_matrix};
///
/// let edges = vec![(0u32, 1u32, 1.0f64), (1, 2, 1.0)];
/// let g = from_edge_list(&edges).expect("build");
/// let adj = to_adjacency_matrix(&g).expect("adj");
/// assert_eq!(adj.shape(), &[3, 3]);
/// assert_eq!(adj[[0, 1]], 1.0f64);
/// assert_eq!(adj[[1, 0]], 1.0f64);  // symmetric
/// assert_eq!(adj[[0, 2]], 0.0f64);
/// ```
pub fn to_adjacency_matrix<N, E>(graph: &Graph<N, E>) -> Result<Array2<E>>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
    E: Clone
        + Copy
        + PartialOrd
        + Debug
        + Send
        + Sync
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + 'static,
{
    Ok(graph.adjacency_matrix())
}

// ============================================================================
// to_edge_list
// ============================================================================

/// Export the graph as a `Vec<(source, target, weight)>`.
///
/// Equivalent to `list(networkx.to_edgelist(G))`.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::{from_edge_list, to_edge_list};
///
/// let input = vec![(0u32, 1u32, 1.5f64)];
/// let g = from_edge_list(&input).expect("build");
/// let edges = to_edge_list(&g).expect("export");
/// assert_eq!(edges.len(), 1);
/// ```
pub fn to_edge_list<N, E>(graph: &Graph<N, E>) -> Result<Vec<(N, N, E)>>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
    E: Clone
        + Copy
        + PartialOrd
        + Debug
        + Send
        + Sync
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + 'static,
{
    let edges = graph.edges();
    Ok(edges
        .into_iter()
        .map(|e| (e.source, e.target, e.weight))
        .collect())
}

// ============================================================================
// pagerank  (iterative power method)
// ============================================================================

/// Compute PageRank for all nodes using the iterative power method.
///
/// This mirrors `networkx.pagerank(G, alpha, max_iter, tol)`.
///
/// Returns a `HashMap` mapping each node to its PageRank score (scores sum to 1).
///
/// # Arguments
///
/// - `graph`       – the undirected input graph (treated as directed with
///   symmetric adjacency for the purposes of PageRank).
/// - `damping`     – damping factor `α` (typically 0.85).
/// - `max_iter`    – maximum number of power-method iterations.
/// - `tolerance`   – convergence threshold (L1 norm of update).
///
/// # Errors
///
/// Returns [`GraphError::ComputationError`] if the graph has no nodes.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::{from_edge_list, pagerank};
///
/// let edges = vec![(0u32, 1u32, 1.0f64), (1, 2, 1.0), (2, 0, 1.0)];
/// let g = from_edge_list(&edges).expect("build");
/// let pr = pagerank(&g, 0.85, 100, 1e-6).expect("pagerank");
/// let total: f64 = pr.values().sum();
/// assert!((total - 1.0).abs() < 1e-9, "scores must sum to 1");
/// ```
pub fn pagerank<N>(
    graph: &Graph<N, f64>,
    damping: f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<HashMap<N, f64>>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
{
    let n = graph.node_count();
    if n == 0 {
        return Err(GraphError::ComputationError(
            "pagerank: graph has no nodes".to_string(),
        ));
    }

    // Build node list in stable order (sorted for determinism)
    let mut node_list: Vec<N> = graph.nodes().into_iter().cloned().collect();
    node_list.sort();

    // Index lookup
    let node_idx: HashMap<N, usize> = node_list
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build adjacency list (neighbor sets) for transition matrix
    let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in node_list.iter().enumerate() {
        let nbrs = graph.neighbors(node).unwrap_or_default();
        for nbr in nbrs {
            if let Some(&j) = node_idx.get(&nbr) {
                out_neighbors[i].push(j);
            }
        }
    }

    // Initial uniform rank
    let init_rank = 1.0 / n as f64;
    let mut rank: Vec<f64> = vec![init_rank; n];
    let dangling_weight = 1.0 / n as f64;

    for _ in 0..max_iter {
        let mut new_rank = vec![0.0f64; n];

        // Dangling nodes (no out-edges) spread their mass uniformly
        let dangling_sum: f64 = out_neighbors
            .iter()
            .enumerate()
            .filter(|(_, nbrs)| nbrs.is_empty())
            .map(|(i, _)| rank[i])
            .sum();

        for i in 0..n {
            // Teleportation + dangling redistribution
            new_rank[i] += (1.0 - damping) / n as f64 + damping * dangling_sum * dangling_weight;

            // Link contribution
            let nbrs = &out_neighbors[i];
            if !nbrs.is_empty() {
                let share = rank[i] / nbrs.len() as f64;
                for &j in nbrs {
                    new_rank[j] += damping * share;
                }
            }
        }

        // Normalize
        let total: f64 = new_rank.iter().sum();
        if total > 0.0 {
            for v in new_rank.iter_mut() {
                *v /= total;
            }
        }

        // Check convergence (L1)
        let delta: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        rank = new_rank;
        if delta < tolerance {
            break;
        }
    }

    Ok(node_list.into_iter().zip(rank).collect())
}

// ============================================================================
// betweenness_centrality  (Brandes 2001)
// ============================================================================

/// Compute betweenness centrality for all nodes using the Brandes algorithm.
///
/// Equivalent to `networkx.betweenness_centrality(G, normalized=True)`.
///
/// The score for node `v` is the fraction of all shortest paths between all
/// pairs of nodes `(s, t)` that pass through `v`.
///
/// # Normalization
///
/// By default scores are normalized by `1 / ((n-1)*(n-2))` for undirected graphs,
/// matching NetworkX's default.  For graphs with fewer than 3 nodes all scores
/// are 0.
///
/// # Complexity
///
/// O(V · E) — same as the classic Brandes algorithm for unweighted graphs.
///
/// # Errors
///
/// Returns [`GraphError::ComputationError`] if the graph has no nodes.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::{from_edge_list, betweenness_centrality};
///
/// // Star graph: center node (0) should have highest centrality
/// let edges = vec![(0u32, 1u32, 1.0f64), (0, 2, 1.0), (0, 3, 1.0)];
/// let g = from_edge_list(&edges).expect("build");
/// let bc = betweenness_centrality(&g).expect("bc");
/// assert!(bc[&0] > bc[&1]);
/// ```
pub fn betweenness_centrality<N>(graph: &Graph<N, f64>) -> Result<HashMap<N, f64>>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
{
    let n = graph.node_count();
    if n == 0 {
        return Err(GraphError::ComputationError(
            "betweenness_centrality: graph has no nodes".to_string(),
        ));
    }

    let mut node_list: Vec<N> = graph.nodes().into_iter().cloned().collect();
    node_list.sort();

    let node_idx: HashMap<N, usize> = node_list
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.clone(), i))
        .collect();

    // Adjacency list
    let adj: Vec<Vec<usize>> = node_list
        .iter()
        .map(|node| {
            graph
                .neighbors(node)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|nbr| node_idx.get(&nbr).copied())
                .collect()
        })
        .collect();

    let mut centrality = vec![0.0f64; n];

    // Brandes algorithm
    for s in 0..n {
        let mut stack: Vec<usize> = Vec::with_capacity(n);
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0i64; n];
        let mut dist: Vec<i64> = vec![-1; n];

        sigma[s] = 1;
        dist[s] = 0;

        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(s);

        // BFS from s
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &adj[v] {
                // First visit
                if dist[w] < 0 {
                    queue.push_back(w);
                    dist[w] = dist[v] + 1;
                }
                // Shortest path to w via v?
                if dist[w] == dist[v] + 1 {
                    sigma[w] = sigma[w].saturating_add(sigma[v]);
                    pred[w].push(v);
                }
            }
        }

        // Accumulation (back-propagation)
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                let coeff = sigma[v] as f64 / sigma[w] as f64 * (1.0 + delta[w]);
                delta[v] += coeff;
            }
            if w != s {
                centrality[w] += delta[w];
            }
        }
    }

    // Normalize: undirected graph -> divide by (n-1)(n-2)
    let norm = if n > 2 {
        1.0 / ((n as f64 - 1.0) * (n as f64 - 2.0))
    } else {
        1.0
    };
    for v in centrality.iter_mut() {
        *v *= norm;
    }

    Ok(node_list.into_iter().zip(centrality).collect())
}

// ============================================================================
// clustering_coefficient  (local)
// ============================================================================

/// Compute the local clustering coefficient for every node.
///
/// For an undirected graph the local clustering coefficient of node `v` with
/// degree `k` is:
///
/// ```text
/// C(v) = 2 * triangles(v) / (k * (k - 1))
/// ```
///
/// Nodes with fewer than 2 neighbors have a clustering coefficient of 0.
///
/// Equivalent to `networkx.clustering(G)`.
///
/// # Errors
///
/// Returns [`GraphError::ComputationError`] if the graph has no nodes.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::{from_edge_list, clustering_coefficient};
///
/// // Triangle: all nodes have clustering coefficient 1.0
/// let edges = vec![(0u32, 1u32, 1.0f64), (1, 2, 1.0), (2, 0, 1.0)];
/// let g = from_edge_list(&edges).expect("build");
/// let cc = clustering_coefficient(&g).expect("cc");
/// for (_, &v) in &cc {
///     assert!((v - 1.0).abs() < 1e-10);
/// }
/// ```
pub fn clustering_coefficient<N>(graph: &Graph<N, f64>) -> Result<HashMap<N, f64>>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
{
    let n = graph.node_count();
    if n == 0 {
        return Err(GraphError::ComputationError(
            "clustering_coefficient: graph has no nodes".to_string(),
        ));
    }

    let mut node_list: Vec<N> = graph.nodes().into_iter().cloned().collect();
    node_list.sort();

    let node_idx: HashMap<N, usize> = node_list
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.clone(), i))
        .collect();

    // Build neighbor sets (as HashSet for O(1) lookup)
    let neighbor_sets: Vec<HashSet<usize>> = node_list
        .iter()
        .map(|node| {
            graph
                .neighbors(node)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|nbr| node_idx.get(&nbr).copied())
                .collect::<HashSet<_>>()
        })
        .collect();

    let mut result = HashMap::with_capacity(n);

    for (i, node) in node_list.iter().enumerate() {
        let nbrs = &neighbor_sets[i];
        let k = nbrs.len();
        if k < 2 {
            result.insert(node.clone(), 0.0f64);
            continue;
        }

        // Count triangles: for each pair (j, l) in neighbors of i,
        // check if (j, l) is an edge (l ∈ neighbors[j]).
        let nbr_vec: Vec<usize> = nbrs.iter().copied().collect();
        let mut triangles = 0usize;
        for a in 0..nbr_vec.len() {
            for b in (a + 1)..nbr_vec.len() {
                let j = nbr_vec[a];
                let l = nbr_vec[b];
                if neighbor_sets[j].contains(&l) {
                    triangles += 1;
                }
            }
        }

        let coeff = 2.0 * triangles as f64 / (k as f64 * (k as f64 - 1.0));
        result.insert(node.clone(), coeff);
    }

    Ok(result)
}

// ============================================================================
// global_clustering_coefficient
// ============================================================================

/// Compute the global (average) clustering coefficient.
///
/// This is the mean of all local clustering coefficients.
/// Equivalent to `networkx.average_clustering(G)`.
///
/// # Errors
///
/// Returns an error if there are no nodes.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::compat::{from_edge_list, global_clustering_coefficient};
///
/// // Complete triangle: global cc = 1.0
/// let edges = vec![(0u32, 1u32, 1.0f64), (1, 2, 1.0), (2, 0, 1.0)];
/// let g = from_edge_list(&edges).expect("build");
/// let gcc = global_clustering_coefficient(&g).expect("gcc");
/// assert!((gcc - 1.0).abs() < 1e-10);
/// ```
pub fn global_clustering_coefficient<N>(graph: &Graph<N, f64>) -> Result<f64>
where
    N: Clone + Hash + Eq + Ord + Debug + Send + Sync + 'static,
{
    let local = clustering_coefficient(graph)?;
    let n = local.len();
    if n == 0 {
        return Ok(0.0);
    }
    let sum: f64 = local.values().sum();
    Ok(sum / n as f64)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> Graph<u32, f64> {
        from_edge_list(&[(0u32, 1u32, 1.0f64), (1, 2, 1.0), (2, 0, 1.0)]).expect("triangle graph")
    }

    fn path_graph(n: u32) -> Graph<u32, f64> {
        let edges: Vec<(u32, u32, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        from_edge_list(&edges).expect("path graph")
    }

    fn star_graph(spokes: u32) -> Graph<u32, f64> {
        let edges: Vec<(u32, u32, f64)> = (1..=spokes).map(|i| (0u32, i, 1.0)).collect();
        from_edge_list(&edges).expect("star graph")
    }

    // ---- from_edge_list ----

    #[test]
    fn test_from_edge_list_node_count() {
        let g = triangle_graph();
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_from_edge_list_edge_count() {
        let g = triangle_graph();
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_from_edge_list_empty() {
        let g: Graph<u32, f64> = from_edge_list(&[]).expect("empty graph");
        assert_eq!(g.node_count(), 0);
    }

    // ---- to_adjacency_matrix ----

    #[test]
    fn test_adjacency_matrix_shape() {
        let g = triangle_graph();
        let adj = to_adjacency_matrix(&g).expect("adj");
        assert_eq!(adj.shape(), &[3, 3]);
    }

    #[test]
    fn test_adjacency_matrix_symmetric() {
        let g = path_graph(4);
        let adj = to_adjacency_matrix(&g).expect("adj");
        let n = adj.shape()[0];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (adj[[i, j]] - adj[[j, i]]).abs() < 1e-12,
                    "adjacency matrix must be symmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_adjacency_matrix_zero_diagonal() {
        let g = triangle_graph();
        let adj = to_adjacency_matrix(&g).expect("adj");
        for i in 0..3 {
            assert_eq!(adj[[i, i]], 0.0);
        }
    }

    // ---- to_edge_list ----

    #[test]
    fn test_to_edge_list_roundtrip() {
        let input: Vec<(u32, u32, f64)> = vec![(0, 1, 2.5), (1, 2, 3.0)];
        let g = from_edge_list(&input).expect("build");
        let exported = to_edge_list(&g).expect("export");
        assert_eq!(exported.len(), 2);
    }

    #[test]
    fn test_to_edge_list_weights_preserved() {
        let g = from_edge_list(&[(0u32, 1u32, 7.5f64)]).expect("build");
        let edges = to_edge_list(&g).expect("export");
        assert_eq!(edges[0].2, 7.5);
    }

    // ---- pagerank ----

    #[test]
    fn test_pagerank_scores_sum_to_one() {
        let g = triangle_graph();
        let pr = pagerank(&g, 0.85, 100, 1e-8).expect("pagerank");
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-9, "sum was {total}");
    }

    #[test]
    fn test_pagerank_uniform_on_regular_graph() {
        // All nodes in a triangle should have equal PageRank
        let g = triangle_graph();
        let pr = pagerank(&g, 0.85, 200, 1e-10).expect("pagerank");
        let vals: Vec<f64> = pr.values().copied().collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        for &v in &vals {
            assert!((v - mean).abs() < 1e-6, "v={v} mean={mean}");
        }
    }

    #[test]
    fn test_pagerank_empty_graph_error() {
        let g: Graph<u32, f64> = Graph::new();
        assert!(pagerank(&g, 0.85, 100, 1e-6).is_err());
    }

    #[test]
    fn test_pagerank_single_node() {
        let mut g: Graph<u32, f64> = Graph::new();
        g.add_node(0);
        let pr = pagerank(&g, 0.85, 100, 1e-6).expect("pagerank single node");
        assert!((pr[&0] - 1.0).abs() < 1e-9);
    }

    // ---- betweenness_centrality ----

    #[test]
    fn test_betweenness_path_graph() {
        // On a path 0-1-2-3, node 1 and 2 should have higher centrality than 0 and 3
        let g = path_graph(4);
        let bc = betweenness_centrality(&g).expect("betweenness");
        assert!(
            bc[&1] > bc[&0],
            "interior nodes should have higher centrality"
        );
        assert!(
            bc[&2] > bc[&3],
            "interior nodes should have higher centrality"
        );
    }

    #[test]
    fn test_betweenness_triangle_all_equal() {
        let g = triangle_graph();
        let bc = betweenness_centrality(&g).expect("betweenness");
        // In a complete triangle, all betweenness scores are equal (all 0.0 with
        // the normalized formula since no shortest path passes through an
        // intermediate node when all pairs are directly connected).
        let vals: Vec<f64> = bc.values().copied().collect();
        let first = vals[0];
        for &v in &vals {
            assert!((v - first).abs() < 1e-9);
        }
    }

    #[test]
    fn test_betweenness_star_center_highest() {
        let g = star_graph(4);
        let bc = betweenness_centrality(&g).expect("betweenness");
        let center = bc[&0];
        for i in 1u32..=4 {
            assert!(center > bc[&i], "center must dominate spokes");
        }
    }

    #[test]
    fn test_betweenness_normalized_max_le_one() {
        let g = path_graph(5);
        let bc = betweenness_centrality(&g).expect("betweenness");
        for &v in bc.values() {
            assert!(
                v <= 1.0 + 1e-9,
                "normalized centrality must be ≤ 1; got {v}"
            );
        }
    }

    #[test]
    fn test_betweenness_empty_graph_error() {
        let g: Graph<u32, f64> = Graph::new();
        assert!(betweenness_centrality(&g).is_err());
    }

    // ---- clustering_coefficient ----

    #[test]
    fn test_clustering_triangle_all_ones() {
        let g = triangle_graph();
        let cc = clustering_coefficient(&g).expect("cc");
        for &v in cc.values() {
            assert!((v - 1.0).abs() < 1e-10, "triangle cc must be 1.0; got {v}");
        }
    }

    #[test]
    fn test_clustering_path_graph_all_zeros() {
        // In a path, no node has a pair of neighbors that are adjacent
        let g = path_graph(5);
        let cc = clustering_coefficient(&g).expect("cc");
        for &v in cc.values() {
            assert!((v).abs() < 1e-10, "path cc must be 0.0; got {v}");
        }
    }

    #[test]
    fn test_clustering_range_zero_to_one() {
        let edges = vec![(0u32, 1u32, 1.0f64), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0)];
        let g = from_edge_list(&edges).expect("build");
        let cc = clustering_coefficient(&g).expect("cc");
        for &v in cc.values() {
            assert!((0.0..=1.0 + 1e-10).contains(&v), "cc out of range: {v}");
        }
    }

    #[test]
    fn test_clustering_single_node_zero() {
        let mut g: Graph<u32, f64> = Graph::new();
        g.add_node(42);
        let cc = clustering_coefficient(&g).expect("cc");
        assert_eq!(cc[&42], 0.0);
    }

    #[test]
    fn test_clustering_empty_graph_error() {
        let g: Graph<u32, f64> = Graph::new();
        assert!(clustering_coefficient(&g).is_err());
    }

    // ---- global_clustering_coefficient ----

    #[test]
    fn test_global_clustering_triangle() {
        let g = triangle_graph();
        let gcc = global_clustering_coefficient(&g).expect("gcc");
        assert!((gcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_clustering_path() {
        let g = path_graph(5);
        let gcc = global_clustering_coefficient(&g).expect("gcc");
        assert!((gcc).abs() < 1e-10);
    }

    #[test]
    fn test_global_clustering_bounds() {
        let edges = vec![(0u32, 1u32, 1.0f64), (1, 2, 1.0), (0, 3, 1.0)];
        let g = from_edge_list(&edges).expect("build");
        let gcc = global_clustering_coefficient(&g).expect("gcc");
        assert!((0.0..=1.0 + 1e-10).contains(&gcc), "gcc={gcc}");
    }
}
