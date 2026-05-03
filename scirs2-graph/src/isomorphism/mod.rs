//! Graph isomorphism algorithms — WL hash and canonical labelling
//!
//! This module augments the existing VF2-based isomorphism routines in
//! `algorithms::isomorphism` with two complementary tools:
//!
//! * **Weisfeiler–Lehman (WL) graph hash** — a fast, graph-level fingerprint.
//!   Non-isomorphic graphs almost always produce different hashes; graphs that
//!   *are* isomorphic are *guaranteed* to produce the same hash.  The hash can
//!   therefore be used to *rule out* isomorphism cheaply before running VF2.
//!
//! * **Canonical certificate** — a canonical string encoding of the graph that
//!   is identical for isomorphic graphs and different for non-isomorphic ones
//!   (subject to the same WL-completeness caveats).  Certificates can be
//!   compared, stored, or used as hash-map keys.
//!
//! The module also re-exports the full VF2 family so callers can do everything
//! through a single `use scirs2_graph::isomorphism::*` import.
//!
//! # Algorithm Notes
//!
//! The WL graph hash uses the **1-dimensional Weisfeiler–Lehman colour-
//! refinement** procedure (also known as the WL subtree kernel iteration):
//!
//! 1. Initialise each node's label as its degree.
//! 2. Iterate `k` times:
//!    a. For each node *v* collect the sorted multiset of its neighbours' labels.
//!    b. Hash `(current_label, sorted_neighbour_labels)` to a new integer label.
//! 3. The graph hash is the hash of the *sorted* multiset of all final labels.
//!
//! The canonical certificate extends this by also encoding the edge structure
//! in a deterministic node-relabelling step after WL stabilisation.
//!
//! # References
//!
//! - Weisfeiler, B. & Lehman, A. (1968). A reduction of a graph to a canonical
//!   form and an algebra arising during this reduction. *Nauchno-Technicheskaya
//!   Informatsia*, 2(9), 12–16.
//! - Shervashidze, N. et al. (2011). Weisfeiler–Lehman graph kernels.
//!   *JMLR*, 12, 2539–2561.

use crate::algorithms::isomorphism::{
    are_graphs_isomorphic, are_graphs_isomorphic_enhanced, find_isomorphism, find_isomorphism_vf2,
};
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::HashMap;

// Re-export VF2 routines so callers can do everything from this module.
pub use crate::algorithms::isomorphism::{
    are_graphs_isomorphic as are_isomorphic,
    are_graphs_isomorphic_enhanced as are_isomorphic_enhanced,
    find_isomorphism as find_isomorphism_mapping,
    find_isomorphism_vf2 as find_isomorphism_vf2_mapping, find_subgraph_matches,
};
pub use crate::algorithms::motifs::vf2_subgraph_isomorphism;

// ─────────────────────────────────────────────────────────────────────────────
// Weisfeiler–Lehman graph hash
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Weisfeiler–Lehman (WL) graph hash.
///
/// The hash is a `u64` fingerprint that is **identical** for isomorphic graphs
/// and **almost certainly different** for non-isomorphic ones.  (There exist
/// regular graphs that are non-isomorphic yet WL-equivalent, but they are rare
/// and typically require very large node counts.)
///
/// # Arguments
///
/// * `graph` – the graph to hash.
/// * `iterations` – number of WL refinement steps (2–5 is usually sufficient;
///   more iterations distinguish subtler structural differences).
///
/// # Type Constraints
///
/// The node type `N` must implement `Ord` so that neighbour labels can be
/// deterministically sorted.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::base::Graph;
/// use scirs2_graph::isomorphism::weisfeiler_lehman_hash;
///
/// let mut g1: Graph<usize, f64> = Graph::new();
/// g1.add_node(0); g1.add_node(1); g1.add_node(2);
/// g1.add_edge(0, 1, 1.0).unwrap();
/// g1.add_edge(1, 2, 1.0).unwrap();
///
/// let mut g2: Graph<usize, f64> = Graph::new();
/// g2.add_node(10); g2.add_node(20); g2.add_node(30);
/// g2.add_edge(10, 20, 1.0).unwrap();
/// g2.add_edge(20, 30, 1.0).unwrap();
///
/// // Isomorphic graphs → same hash
/// assert_eq!(
///     weisfeiler_lehman_hash(&g1, 3).unwrap(),
///     weisfeiler_lehman_hash(&g2, 3).unwrap(),
/// );
/// ```
pub fn weisfeiler_lehman_hash<N, E, Ix>(graph: &Graph<N, E, Ix>, iterations: usize) -> Result<u64>
where
    N: Node + Ord + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    if iterations == 0 {
        return Err(GraphError::InvalidGraph(
            "weisfeiler_lehman_hash: iterations must be ≥ 1".to_string(),
        ));
    }

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    if nodes.is_empty() {
        return Ok(hash_u64_value(0u64));
    }

    // Map each node to a compact integer index for array-based label updates
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let n = nodes.len();
    // Initialise labels by degree
    let mut labels: Vec<u64> = nodes
        .iter()
        .map(|node| {
            graph
                .neighbors(node)
                .map(|nbrs| nbrs.len() as u64)
                .unwrap_or(0)
        })
        .collect();

    // Pre-build neighbour index lists
    let adj: Vec<Vec<usize>> = nodes
        .iter()
        .map(|node| {
            graph
                .neighbors(node)
                .map(|nbrs| {
                    nbrs.iter()
                        .filter_map(|nb| node_to_idx.get(nb).copied())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();

    let mut new_labels = vec![0u64; n];

    for _ in 0..iterations {
        for v in 0..n {
            // Collect and sort neighbour labels
            let mut nbr_labels: Vec<u64> = adj[v].iter().map(|&nb| labels[nb]).collect();
            nbr_labels.sort_unstable();

            // Hash (current_label, sorted_neighbour_labels)
            let combined = combine_hash_sequence(labels[v], &nbr_labels);
            new_labels[v] = combined;
        }
        labels.clone_from(&new_labels);
    }

    // Graph hash = hash of the sorted multiset of all node labels
    labels.sort_unstable();
    let graph_hash = combine_hash_sequence(nodes.len() as u64, &labels);
    Ok(graph_hash)
}

/// Efficient, stable FNV-1a–inspired hash combiner.
#[inline]
fn hash_u64_value(v: u64) -> u64 {
    // FNV-1a 64-bit on the bytes of v
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut h = FNV_OFFSET;
    for byte in v.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash a seed combined with a sequence of u64 values.
fn combine_hash_sequence(seed: u64, values: &[u64]) -> u64 {
    let mut h = hash_u64_value(seed);
    for &v in values {
        // Mix-in each value
        let vh = hash_u64_value(v);
        h ^= vh
            .wrapping_add(0x9e37_79b9_7f4a_7c15_u64)
            .wrapping_add(h << 6)
            .wrapping_add(h >> 2);
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical certificate
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a canonical certificate (string) of a graph.
///
/// The certificate is identical for isomorphic graphs and different for
/// non-isomorphic ones (subject to WL-completeness limits).  It can be
/// used as a hash-map key, stored, and compared efficiently.
///
/// The algorithm:
/// 1. Run `iterations` rounds of WL label refinement.
/// 2. Assign canonical node IDs by sorting nodes by their WL label
///    (ties broken by degree, then neighbour-label tuple).
/// 3. Re-encode the graph as a sorted edge list using canonical IDs.
/// 4. Serialise as a deterministic ASCII string.
///
/// # Arguments
///
/// * `graph` – the graph to label.
/// * `iterations` – WL refinement steps (default 3 is good for most graphs).
///
/// # Example
///
/// ```rust
/// use scirs2_graph::base::Graph;
/// use scirs2_graph::isomorphism::canonical_form;
///
/// let mut g1: Graph<usize, f64> = Graph::new();
/// g1.add_node(0); g1.add_node(1); g1.add_node(2);
/// g1.add_edge(0, 1, 1.0).unwrap();
/// g1.add_edge(1, 2, 1.0).unwrap();
///
/// let mut g2: Graph<usize, f64> = Graph::new();
/// g2.add_node(5); g2.add_node(6); g2.add_node(7);
/// g2.add_edge(6, 5, 1.0).unwrap();
/// g2.add_edge(7, 6, 1.0).unwrap();
///
/// assert_eq!(
///     canonical_form(&g1, 3).unwrap(),
///     canonical_form(&g2, 3).unwrap(),
/// );
/// ```
pub fn canonical_form<N, E, Ix>(graph: &Graph<N, E, Ix>, iterations: usize) -> Result<String>
where
    N: Node + Ord + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    if iterations == 0 {
        return Err(GraphError::InvalidGraph(
            "canonical_form: iterations must be ≥ 1".to_string(),
        ));
    }

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    if nodes.is_empty() {
        return Ok("n=0|e=0|".to_string());
    }

    // Map each node to a compact index
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let n = nodes.len();

    // Build neighbour index lists
    let adj: Vec<Vec<usize>> = nodes
        .iter()
        .map(|node| {
            graph
                .neighbors(node)
                .map(|nbrs| {
                    nbrs.iter()
                        .filter_map(|nb| node_to_idx.get(nb).copied())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();

    // Initialise labels by degree
    let mut labels: Vec<u64> = adj.iter().map(|nbrs| nbrs.len() as u64).collect();
    let mut new_labels = vec![0u64; n];

    for _ in 0..iterations {
        for v in 0..n {
            let mut nbr_labels: Vec<u64> = adj[v].iter().map(|&nb| labels[nb]).collect();
            nbr_labels.sort_unstable();
            new_labels[v] = combine_hash_sequence(labels[v], &nbr_labels);
        }
        labels.clone_from(&new_labels);
    }

    // Compute canonical ordering: sort nodes by (wl_label, degree, sorted_nbr_labels)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| {
        let mut nbr_l: Vec<u64> = adj[v].iter().map(|&nb| labels[nb]).collect();
        nbr_l.sort_unstable();
        (labels[v], adj[v].len() as u64, nbr_l)
    });

    // Build canonical_id[original_index] = canonical_position
    let mut canonical_id = vec![0usize; n];
    for (canon_pos, &orig_v) in order.iter().enumerate() {
        canonical_id[orig_v] = canon_pos;
    }

    // Collect edges in canonical IDs, sort, and deduplicate
    let mut edge_set: Vec<(usize, usize)> = Vec::new();
    for v in 0..n {
        for &nb in &adj[v] {
            let cu = canonical_id[v];
            let cv = canonical_id[nb];
            let (a, b) = if cu <= cv { (cu, cv) } else { (cv, cu) };
            edge_set.push((a, b));
        }
    }
    edge_set.sort_unstable();
    edge_set.dedup();

    // Serialise
    let mut cert = format!("n={}|e={}|", n, edge_set.len());
    for (a, b) in &edge_set {
        cert.push_str(&format!("{}-{},", a, b));
    }

    Ok(cert)
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: check isomorphism via WL filter then VF2
// ─────────────────────────────────────────────────────────────────────────────

/// Check two graphs for isomorphism using a two-phase approach.
///
/// Phase 1 (fast filter): compute and compare WL hashes.  If they differ the
/// graphs are definitively **not** isomorphic.  This check is O(n·k) where k
/// is the number of WL iterations.
///
/// Phase 2 (exact): run the VF2 algorithm to confirm or deny isomorphism
/// in cases where WL hashes agree.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::base::Graph;
/// use scirs2_graph::isomorphism::check_isomorphic;
///
/// let mut g1: Graph<usize, f64> = Graph::new();
/// g1.add_node(0); g1.add_node(1);
/// g1.add_edge(0, 1, 1.0).unwrap();
///
/// let mut g2: Graph<usize, f64> = Graph::new();
/// g2.add_node(5); g2.add_node(6);
/// g2.add_edge(5, 6, 1.0).unwrap();
///
/// assert!(check_isomorphic(&g1, &g2, 3).unwrap());
/// ```
pub fn check_isomorphic<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    wl_iterations: usize,
) -> Result<bool>
where
    N1: Node + Ord + Clone + std::fmt::Debug,
    N2: Node + Ord + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    if wl_iterations == 0 {
        return Err(GraphError::InvalidGraph(
            "check_isomorphic: wl_iterations must be ≥ 1".to_string(),
        ));
    }

    // Quick structural filters
    if graph1.node_count() != graph2.node_count() {
        return Ok(false);
    }
    if graph1.edge_count() != graph2.edge_count() {
        return Ok(false);
    }

    // WL hash filter
    let h1 = weisfeiler_lehman_hash(graph1, wl_iterations)?;
    let h2 = weisfeiler_lehman_hash(graph2, wl_iterations)?;
    if h1 != h2 {
        return Ok(false);
    }

    // Exact VF2 check
    Ok(are_graphs_isomorphic(graph1, graph2))
}

/// Find the isomorphism mapping between two graphs (or return `None`).
///
/// Uses WL hashing as a fast pre-filter before running VF2.
///
/// Returns `Some(mapping)` where `mapping[i]` is the node index in `graph2`
/// that corresponds to node index `i` in `graph1`.
pub fn find_graph_isomorphism<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    wl_iterations: usize,
) -> Result<Option<HashMap<N1, N2>>>
where
    N1: Node + Ord + Clone + std::fmt::Debug,
    N2: Node + Ord + Clone + std::fmt::Debug + std::hash::Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    if wl_iterations == 0 {
        return Err(GraphError::InvalidGraph(
            "find_graph_isomorphism: wl_iterations must be ≥ 1".to_string(),
        ));
    }

    // Quick structural filters
    if graph1.node_count() != graph2.node_count() {
        return Ok(None);
    }
    if graph1.edge_count() != graph2.edge_count() {
        return Ok(None);
    }

    // WL hash filter
    let h1 = weisfeiler_lehman_hash(graph1, wl_iterations)?;
    let h2 = weisfeiler_lehman_hash(graph2, wl_iterations)?;
    if h1 != h2 {
        return Ok(None);
    }

    // Exact VF2 isomorphism search
    Ok(find_isomorphism(graph1, graph2))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> Graph<usize, f64> {
        let mut g = Graph::new();
        for i in 0..n {
            g.add_node(i);
        }
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1, 1.0).expect("add_edge failed");
        }
        g
    }

    fn cycle_graph(n: usize) -> Graph<usize, f64> {
        let mut g = Graph::new();
        for i in 0..n {
            g.add_node(i);
        }
        for i in 0..n {
            g.add_edge(i, (i + 1) % n, 1.0).expect("add_edge failed");
        }
        g
    }

    fn complete_graph(n: usize) -> Graph<usize, f64> {
        let mut g = Graph::new();
        for i in 0..n {
            g.add_node(i);
        }
        for i in 0..n {
            for j in i + 1..n {
                g.add_edge(i, j, 1.0).expect("add_edge failed");
            }
        }
        g
    }

    // ── WL hash tests ──────────────────────────────────────────────────────

    #[test]
    fn test_wl_hash_empty_graph() {
        let g: Graph<usize, f64> = Graph::new();
        let h = weisfeiler_lehman_hash(&g, 3).expect("WL hash should succeed on empty graph");
        // Empty graph → hash of 0
        assert_eq!(h, hash_u64_value(0u64));
    }

    #[test]
    fn test_wl_hash_isomorphic_paths() {
        let g1 = path_graph(5);
        // Build an isomorphic path with different node labels
        let mut g2: Graph<usize, f64> = Graph::new();
        for i in 10..15 {
            g2.add_node(i);
        }
        for i in 0..4 {
            g2.add_edge(10 + i, 10 + i + 1, 1.0)
                .expect("add_edge failed");
        }
        let h1 = weisfeiler_lehman_hash(&g1, 3).expect("WL hash failed");
        let h2 = weisfeiler_lehman_hash(&g2, 3).expect("WL hash failed");
        assert_eq!(h1, h2, "Isomorphic paths should have the same WL hash");
    }

    #[test]
    fn test_wl_hash_non_isomorphic() {
        let path = path_graph(5);
        let cycle = cycle_graph(5);
        let h_path = weisfeiler_lehman_hash(&path, 3).expect("WL hash failed");
        let h_cycle = weisfeiler_lehman_hash(&cycle, 3).expect("WL hash failed");
        assert_ne!(
            h_path, h_cycle,
            "Path5 and Cycle5 must have different WL hashes"
        );
    }

    #[test]
    fn test_wl_hash_zero_iterations_error() {
        let g = path_graph(4);
        assert!(weisfeiler_lehman_hash(&g, 0).is_err());
    }

    #[test]
    fn test_wl_hash_complete_graphs() {
        let k4a = complete_graph(4);
        let mut k4b: Graph<usize, f64> = Graph::new();
        for i in 100..104 {
            k4b.add_node(i);
        }
        for i in 100..104 {
            for j in (i + 1)..104 {
                k4b.add_edge(i, j, 1.0).expect("add_edge failed");
            }
        }
        let h1 = weisfeiler_lehman_hash(&k4a, 3).expect("WL hash failed");
        let h2 = weisfeiler_lehman_hash(&k4b, 3).expect("WL hash failed");
        assert_eq!(h1, h2);
    }

    // ── canonical form tests ───────────────────────────────────────────────

    #[test]
    fn test_canonical_form_isomorphic_paths() {
        let g1 = path_graph(4);
        let mut g2: Graph<usize, f64> = Graph::new();
        for i in 20..24 {
            g2.add_node(i);
        }
        for i in 0..3 {
            g2.add_edge(20 + i, 20 + i + 1, 1.0)
                .expect("add_edge failed");
        }
        let c1 = canonical_form(&g1, 3).expect("canonical_form failed");
        let c2 = canonical_form(&g2, 3).expect("canonical_form failed");
        assert_eq!(
            c1, c2,
            "Isomorphic paths should have the same canonical form"
        );
    }

    #[test]
    fn test_canonical_form_empty() {
        let g: Graph<usize, f64> = Graph::new();
        let cert = canonical_form(&g, 3).expect("canonical_form failed");
        assert!(cert.starts_with("n=0|"));
    }

    #[test]
    fn test_canonical_form_non_isomorphic() {
        let path = path_graph(5);
        let cycle = cycle_graph(5);
        let c_path = canonical_form(&path, 3).expect("canonical_form failed");
        let c_cycle = canonical_form(&cycle, 3).expect("canonical_form failed");
        assert_ne!(c_path, c_cycle);
    }

    #[test]
    fn test_canonical_form_zero_iterations_error() {
        let g = path_graph(3);
        assert!(canonical_form(&g, 0).is_err());
    }

    // ── check_isomorphic tests ─────────────────────────────────────────────

    #[test]
    fn test_check_isomorphic_same_graph() {
        let g = path_graph(5);
        let result = check_isomorphic(&g, &g, 3).expect("check_isomorphic failed");
        assert!(result);
    }

    #[test]
    fn test_check_isomorphic_isomorphic_pair() {
        let g1 = path_graph(4);
        let mut g2: Graph<usize, f64> = Graph::new();
        for i in 50..54 {
            g2.add_node(i);
        }
        for i in 0..3 {
            g2.add_edge(50 + i, 50 + i + 1, 1.0)
                .expect("add_edge failed");
        }
        let result = check_isomorphic(&g1, &g2, 3).expect("check_isomorphic failed");
        assert!(result, "Path4 graphs should be isomorphic");
    }

    #[test]
    fn test_check_isomorphic_different_sizes() {
        let g1 = path_graph(4);
        let g2 = path_graph(5);
        let result = check_isomorphic(&g1, &g2, 3).expect("check_isomorphic failed");
        assert!(!result);
    }

    #[test]
    fn test_check_isomorphic_non_isomorphic() {
        let path = path_graph(5);
        let cycle = cycle_graph(5);
        let result = check_isomorphic(&path, &cycle, 3).expect("check_isomorphic failed");
        assert!(!result);
    }

    #[test]
    fn test_check_isomorphic_zero_iterations_error() {
        let g = path_graph(3);
        assert!(check_isomorphic(&g, &g, 0).is_err());
    }

    // ── find_graph_isomorphism tests ───────────────────────────────────────

    #[test]
    fn test_find_graph_isomorphism_path() {
        let g1 = path_graph(3);
        let mut g2: Graph<usize, f64> = Graph::new();
        g2.add_node(7);
        g2.add_node(8);
        g2.add_node(9);
        g2.add_edge(7, 8, 1.0).expect("add_edge failed");
        g2.add_edge(8, 9, 1.0).expect("add_edge failed");

        let mapping = find_graph_isomorphism(&g1, &g2, 3).expect("find_graph_isomorphism failed");
        assert!(mapping.is_some(), "Should find an isomorphism");
    }

    #[test]
    fn test_find_graph_isomorphism_no_match() {
        let path = path_graph(5);
        let cycle = cycle_graph(5);
        let mapping =
            find_graph_isomorphism(&path, &cycle, 3).expect("find_graph_isomorphism failed");
        assert!(
            mapping.is_none(),
            "Path5 and Cycle5 should not be isomorphic"
        );
    }

    // ── hash uniqueness smoke test ─────────────────────────────────────────

    #[test]
    fn test_wl_hash_distinguishes_small_graphs() {
        // Build all non-isomorphic connected graphs on 4 nodes (there are 6).
        // At least verify our hash doesn't collide for obviously different graphs.
        let g_k4 = complete_graph(4);
        let g_path4 = path_graph(4);
        let g_cycle4 = cycle_graph(4);

        let h_k4 = weisfeiler_lehman_hash(&g_k4, 4).expect("WL hash failed");
        let h_p4 = weisfeiler_lehman_hash(&g_path4, 4).expect("WL hash failed");
        let h_c4 = weisfeiler_lehman_hash(&g_cycle4, 4).expect("WL hash failed");

        // K4, Path4, Cycle4 are all non-isomorphic
        assert_ne!(h_k4, h_p4);
        assert_ne!(h_k4, h_c4);
        assert_ne!(h_p4, h_c4);
    }
}
