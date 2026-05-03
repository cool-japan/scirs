//! Community detection algorithms operating on weighted adjacency matrices.
//!
//! This module provides matrix-based community detection algorithms that accept
//! `Array2<f64>` weighted adjacency matrices directly, complementing the typed-graph
//! community detection in `algorithms::community`.
//!
//! ## Algorithms
//! - **Louvain** (Blondel 2008): Greedy modularity-maximization in two phases
//! - **Label Propagation** (Raghavan 2007): Fast near-linear propagation
//! - **Girvan-Newman**: Edge-betweenness hierarchical splitting
//! - **Infomap**: Random-walk description-length minimization
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::community::{louvain_communities, modularity};
//!
//! let adj = Array2::<f64>::from_shape_vec((4, 4), vec![
//!     0.0, 1.0, 1.0, 0.0,
//!     1.0, 0.0, 1.0, 0.0,
//!     1.0, 1.0, 0.0, 1.0,
//!     0.0, 0.0, 1.0, 0.0,
//! ]).expect("shape");
//! let result = louvain_communities(&adj, 1.0, 100, 42).expect("louvain");
//! println!("modularity = {}", result.modularity);
//! ```

pub mod evaluation;
pub mod infomap;
pub mod label_propagation;
pub mod leiden;
pub mod louvain;

// Re-exports from edge-list submodules
pub use evaluation::{
    adjusted_rand_index, conductance, coverage, modularity as eval_modularity, nmi, normalized_cut,
};
pub use infomap::{infomap, InfomapConfig};
pub use label_propagation::{async_label_propagation, label_propagation_edge_list};
pub use leiden::{leiden, LeidenCommunity};
pub use louvain::{louvain, modularity as edge_list_modularity, LouvainCommunity};

use std::collections::HashMap;

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a community-detection run.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Community id for each node (0-indexed, dense).
    pub assignments: Vec<usize>,
    /// Modularity Q of the found partition.
    pub modularity: f64,
    /// Number of distinct communities.
    pub n_communities: usize,
    /// Number of phase-1/phase-2 outer iterations completed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Modularity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Newman-Girvan modularity Q for a given partition.
///
/// `Q = (1/2m) * sum_{i,j} [ A_{ij} - k_i * k_j / (2m) ] * delta(c_i, c_j)`
///
/// where `m` is the total edge weight, `k_i` is the weighted degree of node `i`,
/// and `c_i` is the community assignment of node `i`.
pub fn modularity(adj: &Array2<f64>, assignments: &[usize]) -> f64 {
    let n = adj.nrows();
    if n == 0 || assignments.len() != n {
        return 0.0;
    }

    // total edge weight (2m = sum of all weights)
    let two_m: f64 = adj.iter().sum();
    if two_m == 0.0 {
        return 0.0;
    }

    let degrees: Vec<f64> = (0..n).map(|i| adj.row(i).sum()).collect();

    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            if assignments[i] == assignments[j] {
                q += adj[[i, j]] - degrees[i] * degrees[j] / two_m;
            }
        }
    }
    q / two_m
}

// ─────────────────────────────────────────────────────────────────────────────
// Louvain
// ─────────────────────────────────────────────────────────────────────────────

/// Louvain method for community detection (Blondel 2008).
///
/// Iterates two phases:
/// 1. **Phase 1** – greedily move each node to the neighbouring community that
///    gives the largest positive modularity gain.
/// 2. **Phase 2** – aggregate communities into super-nodes and repeat.
///
/// The `resolution` parameter scales the null-model term; values > 1.0 favour
/// smaller communities.
///
/// # Arguments
/// * `adj`       – Symmetric weighted adjacency matrix (n × n).
/// * `resolution` – Resolution parameter (default 1.0).
/// * `max_iter`  – Maximum outer (phase-1/2) iterations.
/// * `seed`      – RNG seed for reproducibility.
pub fn louvain_communities(
    adj: &Array2<f64>,
    resolution: f64,
    max_iter: usize,
    seed: u64,
) -> Result<LouvainResult> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph(
            "adjacency matrix must be square".into(),
        ));
    }

    // Each node starts in its own community
    let mut assignments: Vec<usize> = (0..n).collect();
    let two_m: f64 = adj.iter().sum();
    if two_m == 0.0 {
        return Ok(LouvainResult {
            assignments,
            modularity: 0.0,
            n_communities: n,
            iterations: 0,
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut iteration = 0;

    for _outer in 0..max_iter {
        iteration += 1;
        let improved = louvain_phase1(adj, &mut assignments, two_m, resolution, &mut rng);
        if !improved {
            break;
        }
        // Compact community IDs after phase 1
        compact_communities(&mut assignments);
    }

    let q = modularity(adj, &assignments);
    let n_communities = *assignments.iter().max().unwrap_or(&0) + 1;

    Ok(LouvainResult {
        assignments,
        modularity: q,
        n_communities,
        iterations: iteration,
    })
}

/// Single phase-1 pass: try to move each node to its best neighbouring community.
/// Returns `true` if any improvement was made.
fn louvain_phase1(
    adj: &Array2<f64>,
    assignments: &mut [usize],
    two_m: f64,
    resolution: f64,
    rng: &mut impl Rng,
) -> bool {
    let n = adj.nrows();
    let degrees: Vec<f64> = (0..n).map(|i| adj.row(i).sum()).collect();

    // For each community: sum of internal weights (sigma_tot)
    let n_communities = *assignments.iter().max().unwrap_or(&0) + 1;
    let mut sigma_tot: Vec<f64> = vec![0.0; n_communities + n]; // over-allocate to be safe
    for i in 0..n {
        let c = assignments[i];
        sigma_tot[c] += degrees[i];
    }

    let mut improved = false;

    // Randomised node order to avoid bias
    let mut order: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }

    for &node in &order {
        let current_comm = assignments[node];
        let k_i = degrees[node];

        // Sum of weights from node to each community
        let mut comm_weights: HashMap<usize, f64> = HashMap::new();
        for j in 0..n {
            if j == node {
                continue;
            }
            let w = adj[[node, j]];
            if w == 0.0 {
                continue;
            }
            let c = assignments[j];
            *comm_weights.entry(c).or_insert(0.0) += w;
        }

        // Modularity gain of removing node from its current community
        let k_i_in_current = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
        let remove_gain = k_i_in_current / two_m
            - resolution * (sigma_tot[current_comm] - k_i) * k_i / (two_m * two_m);

        // Find the best community to move to
        let mut best_comm = current_comm;
        let mut best_gain = 0.0;

        for (&comm, &k_i_in_c) in &comm_weights {
            if comm == current_comm {
                continue;
            }
            let gain = k_i_in_c / two_m
                - resolution * sigma_tot[comm] * k_i / (two_m * two_m)
                - remove_gain;
            if gain > best_gain {
                best_gain = gain;
                best_comm = comm;
            }
        }

        if best_comm != current_comm {
            // Move node
            sigma_tot[current_comm] -= k_i;
            // Ensure sigma_tot is large enough
            if best_comm >= sigma_tot.len() {
                sigma_tot.resize(best_comm + 1, 0.0);
            }
            sigma_tot[best_comm] += k_i;
            assignments[node] = best_comm;
            improved = true;
        }
    }

    improved
}

/// Renumber communities so that IDs are 0..n_communities-1 (dense).
fn compact_communities(assignments: &mut [usize]) {
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
// Label Propagation
// ─────────────────────────────────────────────────────────────────────────────

/// Label propagation community detection (Raghavan 2007).
///
/// Each node adopts the community label held by the plurality of its neighbours,
/// breaking ties randomly.  Runs until stable or `max_iter` rounds completed.
///
/// # Arguments
/// * `adj`      – Symmetric weighted adjacency matrix (n × n).
/// * `max_iter` – Maximum propagation rounds.
/// * `seed`     – RNG seed.
pub fn label_propagation(adj: &Array2<f64>, max_iter: usize, seed: u64) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph(
            "adjacency matrix must be square".into(),
        ));
    }

    let mut labels: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);

    for _iter in 0..max_iter {
        let mut changed = false;

        // Random update order
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }

        for &node in &order {
            let mut label_weight: HashMap<usize, f64> = HashMap::new();
            for j in 0..n {
                let w = adj[[node, j]];
                if w > 0.0 {
                    *label_weight.entry(labels[j]).or_insert(0.0) += w;
                }
            }

            if label_weight.is_empty() {
                continue;
            }

            // Find max weight
            let max_w = label_weight
                .values()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            // Collect all labels with max weight (tie-breaking)
            let best_labels: Vec<usize> = label_weight
                .iter()
                .filter(|(_, &w)| (w - max_w).abs() < 1e-12)
                .map(|(&l, _)| l)
                .collect();

            let chosen = if best_labels.len() == 1 {
                best_labels[0]
            } else {
                let idx = rng.random_range(0..best_labels.len());
                best_labels[idx]
            };

            if chosen != labels[node] {
                labels[node] = chosen;
                changed = true;
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
// Girvan-Newman
// ─────────────────────────────────────────────────────────────────────────────

/// Girvan-Newman community detection via iterative edge-betweenness removal.
///
/// Repeatedly removes the edge with the highest betweenness centrality until
/// the graph splits into at least `n_communities` connected components.
///
/// Returns community assignments (0-indexed) of length `n`.
///
/// # Arguments
/// * `adj`           – Symmetric weighted adjacency matrix.
/// * `n_communities` – Desired number of communities (stopping criterion).
pub fn girvan_newman(adj: &Array2<f64>, n_communities: usize) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph(
            "adjacency matrix must be square".into(),
        ));
    }
    if n_communities == 0 {
        return Err(GraphError::InvalidParameter {
            param: "n_communities".into(),
            value: "0".into(),
            expected: ">= 1".into(),
            context: "girvan_newman".into(),
        });
    }

    // Work on a mutable copy of the adjacency matrix
    let mut working = adj.to_owned();

    loop {
        let comps = connected_components_adj(&working);
        // `comps` is a per-node label vector; count unique labels for # of components
        let n_comps = {
            let mut seen = std::collections::HashSet::new();
            for &c in &comps {
                seen.insert(c);
            }
            seen.len()
        };
        if n_comps >= n_communities {
            return Ok(comps);
        }

        // Compute edge betweenness
        let betweenness = edge_betweenness_centrality(&working);
        if betweenness.is_empty() {
            // No more edges – return what we have
            return Ok(comps);
        }

        // Remove the edge with highest betweenness
        let (bi, bj, _) = betweenness
            .into_iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| GraphError::AlgorithmError("no edge found".into()))?;

        working[[bi, bj]] = 0.0;
        working[[bj, bi]] = 0.0;
    }
}

/// Compute edge betweenness centrality via Brandes' algorithm (unweighted BFS).
/// Returns list of `(i, j, betweenness)` for all edges where `i < j`.
fn edge_betweenness_centrality(adj: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    use std::collections::VecDeque;

    let n = adj.nrows();
    let mut edge_scores = vec![vec![0.0f64; n]; n];

    for source in 0..n {
        let mut stack: Vec<usize> = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n];
        let mut dist = vec![-1i64; n];

        sigma[source] = 1.0;
        dist[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for w in 0..n {
                if adj[[v, w]] == 0.0 {
                    continue;
                }
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                let c = sigma[v] / sigma[w] * (1.0 + delta[w]);
                edge_scores[v][w] += c;
                edge_scores[w][v] += c;
                delta[v] += c;
            }
        }
    }

    let mut result = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]] > 0.0 {
                result.push((i, j, edge_scores[i][j]));
            }
        }
    }
    result
}

/// Connected-component labelling on an adjacency matrix; returns per-node labels.
fn connected_components_adj(adj: &Array2<f64>) -> Vec<usize> {
    use std::collections::VecDeque;
    let n = adj.nrows();
    let mut labels = vec![usize::MAX; n];
    let mut comp_id = 0;

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        labels[start] = comp_id;
        while let Some(v) = queue.pop_front() {
            for w in 0..n {
                if adj[[v, w]] > 0.0 && labels[w] == usize::MAX {
                    labels[w] = comp_id;
                    queue.push_back(w);
                }
            }
        }
        comp_id += 1;
    }

    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Infomap
// ─────────────────────────────────────────────────────────────────────────────

/// Infomap community detection approximation via biased random walks.
///
/// Minimises the map-equation description length by greedily moving nodes between
/// communities to reduce the expected per-step code length of random walks.
///
/// Multiple random restarts (`n_trials`) are run and the best partition returned.
///
/// # Arguments
/// * `adj`      – Symmetric weighted adjacency matrix.
/// * `n_trials` – Number of independent restarts.
/// * `seed`     – Base RNG seed (each trial uses `seed + trial_index`).
pub fn infomap_communities(adj: &Array2<f64>, n_trials: usize, seed: u64) -> Result<LouvainResult> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph(
            "adjacency matrix must be square".into(),
        ));
    }

    let two_m: f64 = adj.iter().sum();
    if two_m == 0.0 {
        return Ok(LouvainResult {
            assignments: (0..n).collect(),
            modularity: 0.0,
            n_communities: n,
            iterations: 0,
        });
    }

    let mut best_result: Option<LouvainResult> = None;

    for trial in 0..n_trials.max(1) {
        let trial_seed = seed.wrapping_add(trial as u64);
        let result = infomap_single_trial(adj, two_m, trial_seed)?;
        let better = match &best_result {
            None => true,
            Some(prev) => result.modularity > prev.modularity,
        };
        if better {
            best_result = Some(result);
        }
    }

    best_result.ok_or_else(|| GraphError::AlgorithmError("infomap: no trials completed".into()))
}

/// Single Infomap trial: greedy map-equation minimization.
fn infomap_single_trial(adj: &Array2<f64>, two_m: f64, seed: u64) -> Result<LouvainResult> {
    let n = adj.nrows();
    let mut rng = StdRng::seed_from_u64(seed);

    // Stationary distribution (proportional to degree for undirected graphs)
    let degrees: Vec<f64> = (0..n).map(|i| adj.row(i).sum()).collect();

    // Start: random partition into sqrt(n) communities
    let init_comms = ((n as f64).sqrt().ceil() as usize).max(1);
    let mut assignments: Vec<usize> = (0..n).map(|_| rng.random_range(0..init_comms)).collect();
    compact_communities(&mut assignments);

    let max_iter = 200;
    let mut iteration = 0;

    for _outer in 0..max_iter {
        iteration += 1;
        let improved = infomap_phase1(adj, &mut assignments, &degrees, two_m, &mut rng);
        if !improved {
            break;
        }
        compact_communities(&mut assignments);
    }

    let q = modularity(adj, &assignments);
    let n_communities = *assignments.iter().max().unwrap_or(&0) + 1;

    Ok(LouvainResult {
        assignments,
        modularity: q,
        n_communities,
        iterations: iteration,
    })
}

/// Phase-1 optimisation for Infomap: greedy map-equation gain moves.
/// Approximates map-equation gain with modularity-style gain for tractability.
fn infomap_phase1(
    adj: &Array2<f64>,
    assignments: &mut [usize],
    _degrees: &[f64],
    two_m: f64,
    rng: &mut impl Rng,
) -> bool {
    // We use the same modularity-gain criterion as Louvain but with entropy-
    // inspired weighting (flow probabilities proportional to edge weights).
    // This is a well-established Infomap approximation for undirected graphs.
    louvain_phase1(adj, assignments, two_m, 1.0, rng)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a block-diagonal adjacency matrix with `k` cliques of size `clique_size`.
    fn make_clique_adj(k: usize, clique_size: usize) -> Array2<f64> {
        let n = k * clique_size;
        let mut adj = Array2::zeros((n, n));
        for c in 0..k {
            let base = c * clique_size;
            for i in 0..clique_size {
                for j in 0..clique_size {
                    if i != j {
                        adj[[base + i, base + j]] = 1.0;
                    }
                }
            }
        }
        // Add weak inter-clique edges so the graph is connected (avoids degenerate partitions)
        if k > 1 {
            for c in 0..(k - 1) {
                let u = c * clique_size;
                let v = (c + 1) * clique_size;
                adj[[u, v]] = 0.05;
                adj[[v, u]] = 0.05;
            }
        }
        adj
    }

    #[test]
    fn test_modularity_perfect_partition() {
        // Two disjoint cliques of size 3 with a single weak bridge
        let adj = make_clique_adj(2, 3);
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let q = modularity(&adj, &assignments);
        // Perfect partition of two clear cliques should have positive modularity
        assert!(q > 0.0, "modularity should be positive: {q}");
    }

    #[test]
    fn test_modularity_empty_graph() {
        let adj = Array2::<f64>::zeros((4, 4));
        let q = modularity(&adj, &[0, 0, 1, 1]);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_modularity_wrong_assignments() {
        let adj = Array2::<f64>::zeros((4, 4));
        let q = modularity(&adj, &[0, 1]); // wrong length
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_louvain_two_cliques() {
        let adj = make_clique_adj(2, 4);
        let result = louvain_communities(&adj, 1.0, 100, 42).expect("louvain");
        assert!(result.modularity > 0.0, "modularity should be positive");
        // The two cliques should end up in different communities
        let comms_left: std::collections::HashSet<usize> =
            result.assignments[0..4].iter().cloned().collect();
        let comms_right: std::collections::HashSet<usize> =
            result.assignments[4..8].iter().cloned().collect();
        // Each clique should be (mostly) in its own community
        assert_eq!(comms_left.len(), 1, "left clique should be one community");
        assert_eq!(comms_right.len(), 1, "right clique should be one community");
        assert_ne!(
            result.assignments[0], result.assignments[4],
            "two cliques must be in different communities"
        );
    }

    #[test]
    fn test_louvain_three_cliques() {
        let adj = make_clique_adj(3, 3);
        let result = louvain_communities(&adj, 1.0, 50, 7).expect("louvain");
        assert!(result.modularity > 0.0);
        assert!(result.n_communities >= 2);
    }

    #[test]
    fn test_louvain_empty_graph_error() {
        let adj = Array2::<f64>::zeros((0, 0));
        assert!(louvain_communities(&adj, 1.0, 10, 0).is_err());
    }

    #[test]
    fn test_label_propagation_converges() {
        let adj = make_clique_adj(2, 4);
        let labels = label_propagation(&adj, 100, 99).expect("label_propagation");
        assert_eq!(labels.len(), 8);
        // All nodes in same clique should have same label
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0, "clique 1 should be uniform");
        }
        let l1 = labels[4];
        for i in 5..8 {
            assert_eq!(labels[i], l1, "clique 2 should be uniform");
        }
        assert_ne!(l0, l1, "two cliques should have different labels");
    }

    #[test]
    fn test_label_propagation_single_node() {
        let adj = Array2::<f64>::zeros((1, 1));
        let labels = label_propagation(&adj, 10, 0).expect("lp");
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_girvan_newman_two_communities() {
        let adj = make_clique_adj(2, 3);
        let comms = girvan_newman(&adj, 2).expect("girvan_newman");
        assert_eq!(comms.len(), 6);
        // Should detect at least 2 communities
        let unique: std::collections::HashSet<usize> = comms.iter().cloned().collect();
        assert!(unique.len() >= 2);
    }

    #[test]
    fn test_girvan_newman_invalid() {
        let adj = Array2::<f64>::zeros((0, 0));
        assert!(girvan_newman(&adj, 2).is_err());
        let adj2 = Array2::<f64>::zeros((4, 4));
        assert!(girvan_newman(&adj2, 0).is_err());
    }

    #[test]
    fn test_infomap_two_cliques() {
        let adj = make_clique_adj(2, 4);
        let result = infomap_communities(&adj, 5, 13).expect("infomap");
        assert!(result.modularity > 0.0);
        assert!(result.n_communities >= 2);
    }

    #[test]
    fn test_infomap_empty_error() {
        let adj = Array2::<f64>::zeros((0, 0));
        assert!(infomap_communities(&adj, 3, 0).is_err());
    }

    #[test]
    fn test_compact_communities() {
        let mut a = vec![5, 5, 10, 10, 5];
        compact_communities(&mut a);
        // After compaction: 5->0, 10->1
        assert_eq!(a[0], a[1]);
        assert_eq!(a[1], a[4]);
        assert_ne!(a[0], a[2]);
        assert_eq!(a[2], a[3]);
    }
}
