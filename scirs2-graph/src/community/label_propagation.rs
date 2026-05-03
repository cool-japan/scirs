//! Label Propagation Algorithm (LPA) for community detection.
//!
//! The Label Propagation Algorithm (Raghavan et al. 2007) is a near-linear-time
//! community detection method. Each node iteratively adopts the most frequent
//! label among its neighbours, with ties broken randomly. The algorithm converges
//! when no node changes its label.
//!
//! Two variants are provided:
//! - **Synchronous** (`label_propagation_edge_list`): all nodes are updated using
//!   labels from the previous iteration.
//! - **Asynchronous** (`async_label_propagation`): each node's update immediately
//!   affects subsequent nodes in the same pass (faster convergence).
//!
//! ## Reference
//! Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm
//! to detect community structures in large-scale networks. *Physical Review E*,
//! 76(3), 036106.

use std::collections::HashMap;

use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use super::louvain::compact_communities;
use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Internal adjacency
// ─────────────────────────────────────────────────────────────────────────────

/// Build a weighted adjacency list from an edge list.
fn build_adj(edges: &[(usize, usize, f64)], n: usize) -> Vec<Vec<(usize, f64)>> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for &(u, v, w) in edges {
        if u < n && v < n {
            adj[u].push((v, w));
            if u != v {
                adj[v].push((u, w));
            }
        }
    }
    adj
}

/// Pick the plurality label among neighbours.
///
/// When multiple labels are tied for the highest weight, the node's own current
/// label (`current_label`) is preferred if it appears in the tie set.  This
/// prevents oscillation in the synchronous variant where all nodes in a clique
/// would otherwise simultaneously switch labels each round.
///
/// Returns `None` if the node is isolated (no neighbours).
fn plurality_label(
    node: usize,
    adj: &[Vec<(usize, f64)>],
    labels: &[usize],
    current_label: usize,
    rng: &mut impl Rng,
) -> Option<usize> {
    let mut label_weight: HashMap<usize, f64> = HashMap::new();
    for &(nbr, w) in &adj[node] {
        if nbr < labels.len() {
            *label_weight.entry(labels[nbr]).or_insert(0.0) += w;
        }
    }
    if label_weight.is_empty() {
        return None;
    }
    let max_w = label_weight
        .values()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut best: Vec<usize> = label_weight
        .iter()
        .filter(|(_, &w)| (w - max_w).abs() < 1e-12)
        .map(|(&l, _)| l)
        .collect();
    best.sort_unstable();
    if best.len() == 1 {
        Some(best[0])
    } else if best.contains(&current_label) {
        // Break ties in favour of the current label to avoid oscillation
        Some(current_label)
    } else {
        Some(best[rng.random_range(0..best.len())])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synchronous LPA
// ─────────────────────────────────────────────────────────────────────────────

/// Synchronous Label Propagation Algorithm.
///
/// All labels are read from the **previous iteration** before any update is
/// applied. This prevents order-of-update effects but may oscillate on bipartite
/// subgraphs. The `max_iter` limit is used as a safeguard.
///
/// # Arguments
/// * `adj`      – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`  – Total number of nodes.
/// * `max_iter` – Maximum number of propagation rounds.
///
/// # Returns
/// Community assignment vector (0-indexed, densely numbered).
pub fn label_propagation_edge_list(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    max_iter: usize,
) -> Result<Vec<usize>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "label_propagation: n_nodes must be > 0".into(),
        ));
    }

    let graph = build_adj(adj, n_nodes);
    let mut labels: Vec<usize> = (0..n_nodes).collect();
    let mut rng = StdRng::seed_from_u64(0x1234567890abcdef);

    for _ in 0..max_iter {
        // Synchronous: snapshot current labels
        let prev_labels = labels.clone();
        let mut changed = false;

        // Randomised update order
        let mut order: Vec<usize> = (0..n_nodes).collect();
        for i in (1..n_nodes).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }

        for &node in &order {
            let cur = prev_labels[node];
            if let Some(chosen) = plurality_label(node, &graph, &prev_labels, cur, &mut rng) {
                if chosen != labels[node] {
                    labels[node] = chosen;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    compact_communities(&mut labels);
    Ok(labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Asynchronous LPA
// ─────────────────────────────────────────────────────────────────────────────

/// Asynchronous Label Propagation Algorithm.
///
/// Each node update immediately uses the **latest** labels of already-updated
/// neighbours in the same pass. This typically converges faster than the
/// synchronous version.
///
/// # Arguments
/// * `adj`      – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`  – Total number of nodes.
/// * `max_iter` – Maximum number of propagation rounds.
///
/// # Returns
/// Community assignment vector (0-indexed, densely numbered).
pub fn async_label_propagation(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    max_iter: usize,
) -> Result<Vec<usize>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "async_label_propagation: n_nodes must be > 0".into(),
        ));
    }

    let graph = build_adj(adj, n_nodes);
    let mut labels: Vec<usize> = (0..n_nodes).collect();
    let mut rng = StdRng::seed_from_u64(0xfedcba9876543210);

    for _ in 0..max_iter {
        let mut changed = false;

        // Randomised update order
        let mut order: Vec<usize> = (0..n_nodes).collect();
        for i in (1..n_nodes).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }

        for &node in &order {
            // Asynchronous: read current labels (which may already be updated this pass)
            let cur = labels[node];
            if let Some(chosen) = plurality_label(node, &graph, &labels, cur, &mut rng) {
                if chosen != labels[node] {
                    labels[node] = chosen;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    compact_communities(&mut labels);
    Ok(labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_clique_edges(k: usize) -> (Vec<(usize, usize, f64)>, usize) {
        let n = 2 * k;
        let mut edges = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((i, j, 1.0));
                edges.push((k + i, k + j, 1.0));
            }
        }
        // Weak bridge
        edges.push((0, k, 0.05));
        (edges, n)
    }

    #[test]
    fn test_sync_lpa_two_cliques() {
        let (edges, n) = two_clique_edges(4);
        let labels = label_propagation_edge_list(&edges, n, 50).expect("sync lpa");
        assert_eq!(labels.len(), 8);
        // All nodes in clique 1 should share one label
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0, "clique1 node {} has wrong label", i);
        }
        let l1 = labels[4];
        for i in 5..8 {
            assert_eq!(labels[i], l1, "clique2 node {} has wrong label", i);
        }
        assert_ne!(l0, l1, "the two cliques must have different labels");
    }

    #[test]
    fn test_async_lpa_two_cliques() {
        let (edges, n) = two_clique_edges(4);
        let labels = async_label_propagation(&edges, n, 50).expect("async lpa");
        assert_eq!(labels.len(), 8);
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0, "clique1 node {} has wrong label", i);
        }
        let l1 = labels[4];
        for i in 5..8 {
            assert_eq!(labels[i], l1, "clique2 node {} has wrong label", i);
        }
        assert_ne!(l0, l1);
    }

    #[test]
    fn test_lpa_single_node() {
        let labels = label_propagation_edge_list(&[], 1, 10).expect("single node");
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_async_lpa_single_node() {
        let labels = async_label_propagation(&[], 1, 10).expect("async single node");
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_lpa_empty_error() {
        assert!(label_propagation_edge_list(&[], 0, 10).is_err());
        assert!(async_label_propagation(&[], 0, 10).is_err());
    }

    #[test]
    fn test_lpa_isolated_nodes() {
        // No edges: each node keeps its own label
        let labels = label_propagation_edge_list(&[], 5, 10).expect("isolated");
        assert_eq!(labels.len(), 5);
        // All labels should be distinct (each node isolated)
        let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(unique.len(), 5);
    }
}
