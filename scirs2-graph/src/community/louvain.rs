//! Louvain community detection algorithm (Blondel et al. 2008) for edge-list graphs.
//!
//! This module provides a modularity-maximizing community detection algorithm that
//! accepts weighted edge lists `(src, dst, weight)` directly, without requiring a
//! dense adjacency matrix.
//!
//! ## Algorithm
//! Two alternating phases:
//! - **Phase 1 (Local moves)**: Each node is moved to the neighbour community that
//!   maximises the modularity gain ΔQ.
//! - **Phase 2 (Aggregation)**: Communities become super-nodes and the edge list is
//!   rebuilt for the next iteration.
//!
//! ## Reference
//! Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
//! Fast unfolding of communities in large networks. *Journal of Statistical
//! Mechanics: Theory and Experiment*, 2008(10), P10008.

use std::collections::HashMap;

use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Louvain community detection algorithm.
#[derive(Debug, Clone)]
pub struct LouvainCommunity {
    /// Community assignment for each node (0-indexed, dense).
    pub assignments: Vec<usize>,
    /// Newman-Girvan modularity Q of the found partition.
    pub modularity: f64,
    /// Number of distinct communities discovered.
    pub n_communities: usize,
    /// Number of outer (phase-1 + phase-2) iterations completed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal adjacency structures
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse adjacency representation for efficient Louvain iterations.
struct SparseAdj {
    /// Adjacency list: neighbours and weights.
    adj: Vec<Vec<(usize, f64)>>,
    /// Weighted degree of each node.
    degree: Vec<f64>,
    /// Total weight in graph (= 2m for undirected).
    two_m: f64,
}

impl SparseAdj {
    fn from_edge_list(edges: &[(usize, usize, f64)], n_nodes: usize) -> Self {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n_nodes];
        let mut degree = vec![0.0f64; n_nodes];
        let mut two_m = 0.0f64;

        for &(u, v, w) in edges {
            if u >= n_nodes || v >= n_nodes {
                continue;
            }
            adj[u].push((v, w));
            adj[v].push((u, w));
            degree[u] += w;
            degree[v] += w;
            two_m += 2.0 * w;
        }
        Self { adj, degree, two_m }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Modularity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Newman-Girvan modularity Q for a given partition of an edge-list graph.
///
/// `Q = 1/(2m) * Σ_{i,j} [A_{ij} - k_i·k_j/(2m)] · δ(c_i, c_j)`
///
/// where `m` is the total edge weight, `k_i` the weighted degree of node `i`,
/// and `c_i` the community of node `i`.
///
/// # Arguments
/// * `edges`    – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`  – Total number of nodes.
/// * `communities` – Community assignment for each node.
pub fn modularity(edges: &[(usize, usize, f64)], n_nodes: usize, communities: &[usize]) -> f64 {
    if n_nodes == 0 || communities.len() != n_nodes {
        return 0.0;
    }
    let g = SparseAdj::from_edge_list(edges, n_nodes);
    if g.two_m == 0.0 {
        return 0.0;
    }

    // Intra-community edge weight sum
    let mut intra = 0.0f64;
    for &(u, v, w) in edges {
        if communities[u] == communities[v] {
            intra += 2.0 * w; // undirected: count both directions
        }
    }

    // Sum of squared community degrees
    let n_comms = *communities.iter().max().unwrap_or(&0) + 1;
    let mut comm_degree = vec![0.0f64; n_comms];
    for i in 0..n_nodes {
        if communities[i] < n_comms {
            comm_degree[communities[i]] += g.degree[i];
        }
    }
    let sq_sum: f64 = comm_degree.iter().map(|&d| d * d).sum();

    (intra / g.two_m) - (sq_sum / (g.two_m * g.two_m))
}

// ─────────────────────────────────────────────────────────────────────────────
// Louvain
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Louvain community detection algorithm on a weighted edge-list graph.
///
/// # Arguments
/// * `adj`        – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`    – Total number of nodes (nodes are `0..n_nodes`).
/// * `resolution` – Resolution parameter γ; values > 1 produce smaller communities.
///
/// # Returns
/// [`LouvainCommunity`] containing node assignments and modularity.
pub fn louvain(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    resolution: f64,
) -> Result<LouvainCommunity> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "louvain: n_nodes must be > 0".into(),
        ));
    }

    // Use fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0x9e3779b9_7f4a7c15);

    // Build initial sparse adjacency
    let g = SparseAdj::from_edge_list(adj, n_nodes);
    if g.two_m == 0.0 {
        return Ok(LouvainCommunity {
            assignments: (0..n_nodes).collect(),
            modularity: 0.0,
            n_communities: n_nodes,
            iterations: 0,
        });
    }

    // Start: each node in its own community
    let mut assignments: Vec<usize> = (0..n_nodes).collect();
    let max_outer = 100usize;
    let mut iterations = 0usize;

    // Keep the original edge list to compute modularity at the end
    let original_adj = adj.to_vec();
    let original_n = n_nodes;

    // For the iterative aggregation we work with a meta-graph
    let mut current_edges: Vec<(usize, usize, f64)> = adj.to_vec();
    let mut current_n = n_nodes;
    // Maps meta-node → original nodes (for translating assignments back)
    let mut meta_to_original: Vec<Vec<usize>> = (0..n_nodes).map(|i| vec![i]).collect();

    for _outer in 0..max_outer {
        iterations += 1;
        let sparse = SparseAdj::from_edge_list(&current_edges, current_n);

        // Phase 1: local moves
        let mut local_assign: Vec<usize> = (0..current_n).collect();
        let improved = louvain_phase1(
            &sparse,
            &mut local_assign,
            sparse.two_m,
            resolution,
            &mut rng,
        );
        if !improved {
            break;
        }
        compact_communities(&mut local_assign);

        // Translate local assignments back to original nodes
        let new_n_comms = *local_assign.iter().max().unwrap_or(&0) + 1;
        // Build new meta-to-original mapping
        let mut new_meta: Vec<Vec<usize>> = vec![vec![]; new_n_comms];
        for (meta_node, &comm) in local_assign.iter().enumerate() {
            for &orig in &meta_to_original[meta_node] {
                new_meta[comm].push(orig);
            }
        }
        // Update original node assignments
        for comm in 0..new_n_comms {
            for &orig in &new_meta[comm] {
                assignments[orig] = comm;
            }
        }
        meta_to_original = new_meta;

        // Phase 2: aggregate into meta-graph
        let (agg_edges, agg_n) = aggregate_graph(&current_edges, &local_assign, new_n_comms);
        current_edges = agg_edges;
        current_n = agg_n;

        if current_n == 1 {
            break;
        }
    }

    let q = modularity(&original_adj, original_n, &assignments);
    let n_communities = *assignments.iter().max().unwrap_or(&0) + 1;

    Ok(LouvainCommunity {
        assignments,
        modularity: q,
        n_communities,
        iterations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: local moves
// ─────────────────────────────────────────────────────────────────────────────

/// Single phase-1 pass. Returns `true` if any assignment changed.
fn louvain_phase1(
    g: &SparseAdj,
    assignments: &mut [usize],
    two_m: f64,
    resolution: f64,
    rng: &mut impl Rng,
) -> bool {
    let n = g.adj.len();
    if n == 0 {
        return false;
    }

    // sigma_tot[c] = sum of degrees of all nodes in community c
    let n_comms_init = *assignments.iter().max().unwrap_or(&0) + 1;
    let mut sigma_tot: Vec<f64> = vec![0.0; n_comms_init + n];
    for i in 0..n {
        let c = assignments[i];
        if c < sigma_tot.len() {
            sigma_tot[c] += g.degree[i];
        }
    }

    let mut improved = false;
    // Randomised order
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }

    for &node in &order {
        let current_comm = assignments[node];
        let k_i = g.degree[node];

        // Accumulate weights to each neighbouring community
        let mut comm_weights: HashMap<usize, f64> = HashMap::new();
        for &(nbr, w) in &g.adj[node] {
            let c = assignments[nbr];
            *comm_weights.entry(c).or_insert(0.0) += w;
        }

        // ΔQ for removing node from its current community
        let k_i_in_cur = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
        let remove_gain = k_i_in_cur / two_m
            - resolution * (sigma_tot[current_comm] - k_i) * k_i / (two_m * two_m);

        // Find best neighbouring community
        let mut best_comm = current_comm;
        let mut best_gain = 0.0f64;
        for (&c, &k_i_in_c) in &comm_weights {
            if c == current_comm {
                continue;
            }
            // Ensure sigma_tot is large enough
            let sigma_c = if c < sigma_tot.len() {
                sigma_tot[c]
            } else {
                0.0
            };
            let gain =
                k_i_in_c / two_m - resolution * sigma_c * k_i / (two_m * two_m) - remove_gain;
            if gain > best_gain {
                best_gain = gain;
                best_comm = c;
            }
        }

        if best_comm != current_comm {
            // Ensure sigma_tot is large enough
            if best_comm >= sigma_tot.len() {
                sigma_tot.resize(best_comm + 1, 0.0);
            }
            sigma_tot[current_comm] -= k_i;
            sigma_tot[best_comm] += k_i;
            assignments[node] = best_comm;
            improved = true;
        }
    }
    improved
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: graph aggregation
// ─────────────────────────────────────────────────────────────────────────────

/// Build the meta-graph by merging nodes in the same community.
/// Self-loops (intra-community edges) are preserved as loop weights.
fn aggregate_graph(
    edges: &[(usize, usize, f64)],
    assignments: &[usize],
    n_comms: usize,
) -> (Vec<(usize, usize, f64)>, usize) {
    let mut meta_weights: HashMap<(usize, usize), f64> = HashMap::new();
    for &(u, v, w) in edges {
        let cu = assignments[u];
        let cv = assignments[v];
        let key = if cu <= cv { (cu, cv) } else { (cv, cu) };
        *meta_weights.entry(key).or_insert(0.0) += w;
    }
    let agg_edges: Vec<(usize, usize, f64)> = meta_weights
        .into_iter()
        .filter(|&(_, w)| w > 0.0)
        .map(|((u, v), w)| (u, v, w))
        .collect();
    (agg_edges, n_comms)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Renumber community IDs to be contiguous `0..n_comms`.
pub(crate) fn compact_communities(assignments: &mut [usize]) {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    for a in assignments.iter_mut() {
        let new_id = mapping.entry(*a).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *a = *new_id;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build two cliques of size `k` connected by a single weak bridge.
    fn two_clique_edges(k: usize) -> (Vec<(usize, usize, f64)>, usize) {
        let n = 2 * k;
        let mut edges = Vec::new();
        // Clique 1: nodes 0..k
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((i, j, 1.0));
            }
        }
        // Clique 2: nodes k..2k
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((k + i, k + j, 1.0));
            }
        }
        // Weak bridge
        edges.push((0, k, 0.05));
        (edges, n)
    }

    #[test]
    fn test_louvain_two_cliques() {
        let (edges, n) = two_clique_edges(4);
        let result = louvain(&edges, n, 1.0).expect("louvain");
        assert!(
            result.modularity > 0.0,
            "modularity should be positive: {}",
            result.modularity
        );
        assert_eq!(result.n_communities, 2, "should find 2 communities");
        // All nodes in clique 1 should have the same community
        let c0 = result.assignments[0];
        for i in 1..4 {
            assert_eq!(
                result.assignments[i], c0,
                "clique1 node {} wrong community",
                i
            );
        }
        let c1 = result.assignments[4];
        for i in 5..8 {
            assert_eq!(
                result.assignments[i], c1,
                "clique2 node {} wrong community",
                i
            );
        }
        assert_ne!(c0, c1, "the two cliques must be in different communities");
    }

    #[test]
    fn test_modularity_perfect_partition() {
        let (edges, n) = two_clique_edges(3);
        let assignments: Vec<usize> = (0..6).map(|i| if i < 3 { 0 } else { 1 }).collect();
        let q = modularity(&edges, n, &assignments);
        assert!(
            q > 0.0,
            "modularity for perfect partition should be positive: {q}"
        );
    }

    #[test]
    fn test_louvain_empty_graph() {
        assert!(louvain(&[], 0, 1.0).is_err());
    }

    #[test]
    fn test_louvain_no_edges() {
        // Isolated nodes (n_nodes > 0, no edges): each node forms its own community.
        // The function returns Ok (error only for n_nodes == 0).
        let result = louvain(&[], 5, 1.0).expect("isolated nodes should succeed");
        assert_eq!(
            result.n_communities, 5,
            "each isolated node is its own community"
        );
        assert_eq!(result.modularity, 0.0, "no edges → zero modularity");
        // Each assignment index should be distinct
        let unique: std::collections::HashSet<usize> = result.assignments.iter().cloned().collect();
        assert_eq!(unique.len(), 5, "all nodes have distinct community IDs");
    }

    #[test]
    fn test_louvain_zero_nodes_error() {
        assert!(louvain(&[], 0, 1.0).is_err());
    }

    #[test]
    fn test_compact_communities() {
        let mut a = vec![3, 3, 7, 7, 3];
        compact_communities(&mut a);
        assert_eq!(a[0], a[1]);
        assert_eq!(a[1], a[4]);
        assert_ne!(a[0], a[2]);
        assert_eq!(a[2], a[3]);
    }
}
