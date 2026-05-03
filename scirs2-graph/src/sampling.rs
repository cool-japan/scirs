//! Graph Sampling Algorithms
//!
//! This module provides a comprehensive suite of graph sampling methods including:
//!
//! - **Random walks**: Uniform random walk, Node2Vec biased random walk
//! - **Graph sampling**: Frontier sampling, forest-fire sampling, snowball sampling
//! - **Subgraph operations**: Induced subgraph extraction
//!
//! All algorithms operate on adjacency-list representations for efficiency.
//!
//! ## References
//! - Leskovec & Faloutsos (2006): Sampling from Large Graphs. KDD 2006.
//! - Grover & Leskovec (2016): node2vec: Scalable Feature Learning for Networks. KDD 2016.
//! - Stumpf et al. (2005): Subnets of scale-free networks are not scale-free. PNAS.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{GraphError, Result};

/// Return type for [`induced_subgraph`]: `(subgraph_adjacency, original_indices)`.
type InducedSubgraphResult = (Vec<Vec<(usize, f64)>>, Vec<usize>);

// ─────────────────────────────────────────────────────────────────────────────
// Minimal LCG-based PRNG (avoids external rand dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// A fast, seedable linear-congruential pseudo-random number generator.
///
/// Uses the parameters from Knuth's MMIX (64-bit LCG).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid degenerate seed=0 by mixing in a constant.
        Self {
            state: seed.wrapping_add(6364136223846793005),
        }
    }

    /// Advance the state and return the next u64.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Return a uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for the mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a uniform usize in 0..n (exclusive). Panics if n == 0.
    fn next_usize(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "n must be > 0");
        (self.next_u64() as usize) % n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random Walk
// ─────────────────────────────────────────────────────────────────────────────

/// Perform a uniform random walk on an unweighted graph.
///
/// Starting from `start_node`, at each step a uniformly random neighbour is
/// chosen.  If the current node has no neighbours the walk terminates early.
///
/// # Parameters
/// - `adjacency`   – adjacency list (unweighted); `adjacency[u]` contains the
///   neighbours of node `u`.
/// - `start_node`  – index of the walk's first node.
/// - `walk_length` – desired total number of nodes in the walk (including the
///   starting node).
/// - `rng_seed`    – seed for the internal pseudo-random number generator.
///
/// # Returns
/// A `Vec<usize>` of length ≤ `walk_length` with the visited node sequence.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if `start_node` is out of range.
pub fn random_walk(
    adjacency: &[Vec<usize>],
    start_node: usize,
    walk_length: usize,
    rng_seed: u64,
) -> Result<Vec<usize>> {
    let n = adjacency.len();
    if start_node >= n {
        return Err(GraphError::invalid_parameter(
            "start_node",
            start_node,
            format!("must be < n_nodes ({})", n),
        ));
    }
    if walk_length == 0 {
        return Ok(Vec::new());
    }

    let mut rng = Lcg::new(rng_seed);
    let mut walk = Vec::with_capacity(walk_length);
    walk.push(start_node);

    let mut current = start_node;
    for _ in 1..walk_length {
        let neighbours = &adjacency[current];
        if neighbours.is_empty() {
            break;
        }
        current = neighbours[rng.next_usize(neighbours.len())];
        walk.push(current);
    }

    Ok(walk)
}

// ─────────────────────────────────────────────────────────────────────────────
// Node2Vec Biased Random Walk
// ─────────────────────────────────────────────────────────────────────────────

/// Perform a Node2Vec biased random walk on a weighted graph.
///
/// Node2Vec generalises DeepWalk by interpolating between BFS-like (p<1) and
/// DFS-like (q<1) exploration using the *return* parameter `p` and the
/// *in-out* parameter `q`.
///
/// The transition probability from node `v` to neighbour `x` (when the
/// previous node was `t`) is proportional to:
/// - `1/p` if `x == t`  (backtrack)
/// - `1`   if `x` is also a neighbour of `t`  (same distance)
/// - `1/q` otherwise    (explore further)
///
/// # Parameters
/// - `adjacency`   – weighted adjacency list; `adjacency[u]` is a list of
///   `(neighbour, weight)` pairs.
/// - `start_node`  – starting node index.
/// - `walk_length` – desired walk length (≥ 1).
/// - `p`           – return parameter (> 0). Higher values discourage backtracking.
/// - `q`           – in-out parameter (> 0). < 1 favours DFS-like walks; > 1
///   favours BFS-like walks.
/// - `rng_seed`    – PRNG seed.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] for out-of-range `start_node` or
/// non-positive `p`/`q`.
pub fn node2vec_walk(
    adjacency: &[Vec<(usize, f64)>],
    start_node: usize,
    walk_length: usize,
    p: f64,
    q: f64,
    rng_seed: u64,
) -> Result<Vec<usize>> {
    let n = adjacency.len();
    if start_node >= n {
        return Err(GraphError::invalid_parameter(
            "start_node",
            start_node,
            format!("must be < n_nodes ({})", n),
        ));
    }
    if p <= 0.0 {
        return Err(GraphError::invalid_parameter(
            "p",
            p,
            "must be strictly positive",
        ));
    }
    if q <= 0.0 {
        return Err(GraphError::invalid_parameter(
            "q",
            q,
            "must be strictly positive",
        ));
    }
    if walk_length == 0 {
        return Ok(Vec::new());
    }

    // Pre-build a fast neighbour-set lookup for bias computation.
    // neighbour_set[u] is the set of indices adjacent to u.
    let neighbour_sets: Vec<HashSet<usize>> = adjacency
        .iter()
        .map(|nbrs| nbrs.iter().map(|&(v, _)| v).collect())
        .collect();

    let mut rng = Lcg::new(rng_seed);
    let mut walk: Vec<usize> = Vec::with_capacity(walk_length);
    walk.push(start_node);

    // First step: uniform over neighbours (no previous node).
    if walk_length == 1 || adjacency[start_node].is_empty() {
        return Ok(walk);
    }
    let first_idx = rng.next_usize(adjacency[start_node].len());
    let first_next = adjacency[start_node][first_idx].0;
    walk.push(first_next);

    // Subsequent steps: biased by p and q relative to previous node.
    for _ in 2..walk_length {
        let prev = walk[walk.len() - 2];
        let curr = walk[walk.len() - 1];

        let nbrs = &adjacency[curr];
        if nbrs.is_empty() {
            break;
        }

        // Compute unnormalised weights for each candidate.
        let prev_set = &neighbour_sets[prev];
        let weights: Vec<f64> = nbrs
            .iter()
            .map(|&(x, edge_w)| {
                let bias = if x == prev {
                    1.0 / p
                } else if prev_set.contains(&x) {
                    1.0
                } else {
                    1.0 / q
                };
                (edge_w.max(0.0)) * bias
            })
            .collect();

        let total: f64 = weights.iter().sum();
        let next_node = if total <= 0.0 {
            // Fallback to uniform if all weights are zero.
            nbrs[rng.next_usize(nbrs.len())].0
        } else {
            let threshold = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut chosen = nbrs.last().map(|&(v, _)| v).unwrap_or(curr);
            for (idx, &w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative >= threshold {
                    chosen = nbrs[idx].0;
                    break;
                }
            }
            chosen
        };

        walk.push(next_node);
    }

    Ok(walk)
}

// ─────────────────────────────────────────────────────────────────────────────
// Frontier Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Frontier-based graph sampling.
///
/// Maintains a *frontier* set of nodes and at each step:
/// 1. Picks a random frontier node `u`.
/// 2. Picks a random neighbour `v` of `u`.
/// 3. If `v` is not yet sampled, adds it to the sample and the frontier; if
///    `v` already sampled, reinserts `u` into the frontier (Frontier Sampling
///    per Stumpf et al. / Leskovec & Faloutsos 2006).
///
/// Frontier sampling preserves degree distribution better than naive random
/// node or random edge sampling.
///
/// # Parameters
/// - `adjacency`   – unweighted adjacency list.
/// - `n_nodes`     – total number of nodes (= `adjacency.len()`).
/// - `sample_size` – desired number of nodes in the sample.
/// - `rng_seed`    – PRNG seed.
///
/// # Returns
/// Sorted `Vec<usize>` of sampled node indices (length ≤ `sample_size`).
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if `n_nodes` is 0 or
/// `sample_size > n_nodes`.
pub fn frontier_sampling(
    adjacency: &[Vec<usize>],
    n_nodes: usize,
    sample_size: usize,
    rng_seed: u64,
) -> Result<Vec<usize>> {
    if n_nodes == 0 {
        return Err(GraphError::invalid_parameter(
            "n_nodes",
            0usize,
            "must be > 0",
        ));
    }
    if sample_size > n_nodes {
        return Err(GraphError::invalid_parameter(
            "sample_size",
            sample_size,
            format!("must be ≤ n_nodes ({})", n_nodes),
        ));
    }
    if sample_size == 0 {
        return Ok(Vec::new());
    }

    let mut rng = Lcg::new(rng_seed);
    let mut sampled: HashSet<usize> = HashSet::with_capacity(sample_size);
    let mut frontier: Vec<usize> = Vec::new();

    // Seed with a random starting node.
    let seed = rng.next_usize(n_nodes);
    sampled.insert(seed);
    frontier.push(seed);

    let mut iters = 0usize;
    let max_iters = sample_size * n_nodes.max(100) * 10;

    while sampled.len() < sample_size && !frontier.is_empty() && iters < max_iters {
        iters += 1;
        // Pick random frontier node.
        let fi = rng.next_usize(frontier.len());
        let u = frontier[fi];

        let nbrs = &adjacency[u];
        if nbrs.is_empty() {
            // Dead-end: remove u from frontier.
            frontier.swap_remove(fi);
            continue;
        }

        let v = nbrs[rng.next_usize(nbrs.len())];
        if sampled.insert(v) {
            // New node: add to frontier.
            frontier.push(v);
        }
        // Whether new or not, keep u in frontier (it may have other unvisited neighbours).
    }

    // If graph is disconnected and we haven't reached sample_size, inject random unsampled nodes.
    if sampled.len() < sample_size {
        for candidate in 0..n_nodes {
            if sampled.len() >= sample_size {
                break;
            }
            sampled.insert(candidate);
        }
    }

    let mut result: Vec<usize> = sampled.into_iter().collect();
    result.sort_unstable();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Forest-Fire Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Forest-fire graph sampling.
///
/// Mimics a "fire spreading" process: from each burning node, a geometrically
/// distributed number of unvisited neighbours are "burned" with forward
/// probability `forward_prob`.  The process regenerates from a new random seed
/// when all fires die out.
///
/// Forest-fire sampling is known to preserve heavy-tail degree distributions
/// and densification patterns (Leskovec et al. 2005).
///
/// # Parameters
/// - `adjacency`    – unweighted adjacency list.
/// - `n_nodes`      – total number of nodes.
/// - `sample_size`  – target number of sampled nodes.
/// - `forward_prob` – probability of spreading to each neighbour (0 < p < 1).
/// - `rng_seed`     – PRNG seed.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] for invalid inputs.
pub fn forest_fire_sampling(
    adjacency: &[Vec<usize>],
    n_nodes: usize,
    sample_size: usize,
    forward_prob: f64,
    rng_seed: u64,
) -> Result<Vec<usize>> {
    if n_nodes == 0 {
        return Err(GraphError::invalid_parameter(
            "n_nodes",
            0usize,
            "must be > 0",
        ));
    }
    if sample_size > n_nodes {
        return Err(GraphError::invalid_parameter(
            "sample_size",
            sample_size,
            format!("must be ≤ n_nodes ({})", n_nodes),
        ));
    }
    if forward_prob <= 0.0 || forward_prob >= 1.0 {
        return Err(GraphError::invalid_parameter(
            "forward_prob",
            forward_prob,
            "must be in (0, 1)",
        ));
    }
    if sample_size == 0 {
        return Ok(Vec::new());
    }

    let mut rng = Lcg::new(rng_seed);
    let mut sampled: HashSet<usize> = HashSet::with_capacity(sample_size);
    // Queue of currently-burning nodes.
    let mut burning: VecDeque<usize> = VecDeque::new();

    // Helper: geometric-distributed number of links to burn.
    // Draw from Geometric(1 - forward_prob): # of successes before first failure.
    let geometric_draw = |rng: &mut Lcg| -> usize {
        let mut count = 0usize;
        while rng.next_f64() < forward_prob {
            count += 1;
        }
        count
    };

    while sampled.len() < sample_size {
        // Light a new fire from a random unsampled node.
        if burning.is_empty() {
            // Find an unsampled node.
            let start = rng.next_usize(n_nodes);
            let mut found = false;
            for offset in 0..n_nodes {
                let candidate = (start + offset) % n_nodes;
                if sampled.insert(candidate) {
                    burning.push_back(candidate);
                    found = true;
                    break;
                }
            }
            if !found {
                break; // All nodes sampled.
            }
        }

        // Spread the fire.
        while let Some(u) = burning.pop_front() {
            if sampled.len() >= sample_size {
                break;
            }
            let nbrs = &adjacency[u];
            if nbrs.is_empty() {
                continue;
            }

            // Number of neighbours to burn (capped by available).
            let n_burn = geometric_draw(&mut rng).min(nbrs.len());
            if n_burn == 0 {
                continue;
            }

            // Pick n_burn distinct unsampled neighbours (reservoir sample).
            // Shuffle first n_burn positions of a candidate list.
            let mut candidates: Vec<usize> = nbrs.clone();
            for i in 0..n_burn {
                let j = i + rng.next_usize(candidates.len() - i);
                candidates.swap(i, j);
            }
            for &v in candidates.iter().take(n_burn) {
                if sampled.len() >= sample_size {
                    break;
                }
                if sampled.insert(v) {
                    burning.push_back(v);
                }
            }
        }
    }

    let mut result: Vec<usize> = sampled.into_iter().collect();
    result.sort_unstable();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Snowball Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Snowball (BFS-neighbourhood) sampling.
///
/// Starting from the given `seed_nodes`, collects all nodes reachable within
/// `n_hops` hops.  This is equivalent to an ego-network expansion.
///
/// # Parameters
/// - `adjacency`  – unweighted adjacency list.
/// - `seed_nodes` – starting node indices.
/// - `n_hops`     – number of BFS expansion steps (0 = seed nodes only).
///
/// # Returns
/// Sorted `Vec<usize>` of all nodes within `n_hops` hops of any seed node.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if any seed node index is
/// out of range or if the adjacency list is empty.
pub fn snowball_sampling(
    adjacency: &[Vec<usize>],
    seed_nodes: &[usize],
    n_hops: usize,
) -> Result<Vec<usize>> {
    let n = adjacency.len();
    if n == 0 {
        return Err(GraphError::invalid_parameter(
            "adjacency",
            "empty",
            "graph must have at least one node",
        ));
    }
    for &s in seed_nodes {
        if s >= n {
            return Err(GraphError::invalid_parameter(
                "seed_node",
                s,
                format!("must be < n_nodes ({})", n),
            ));
        }
    }

    let mut visited: HashSet<usize> = seed_nodes.iter().cloned().collect();
    let mut frontier: Vec<usize> = seed_nodes.to_vec();

    for _ in 0..n_hops {
        let mut next_frontier: Vec<usize> = Vec::new();
        for &u in &frontier {
            for &v in &adjacency[u] {
                if visited.insert(v) {
                    next_frontier.push(v);
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    let mut result: Vec<usize> = visited.into_iter().collect();
    result.sort_unstable();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Induced Subgraph
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the induced subgraph on a set of nodes.
///
/// Given a weighted adjacency list and a set of node indices, returns:
/// - A new weighted adjacency list on the *re-indexed* subgraph (nodes are
///   re-numbered 0..node_set.len() in the order they appear after sorting).
/// - A mapping `original_indices[i]` = original node index of subgraph node `i`.
///
/// Only edges where **both** endpoints are in `node_set` are retained.
///
/// # Parameters
/// - `adjacency` – weighted adjacency list of the full graph.
/// - `node_set`  – node indices to include (may contain duplicates; duplicates
///   are silently deduplicated).
///
/// # Returns
/// `(subgraph_adjacency, original_indices)` where:
/// - `subgraph_adjacency[i]` is a list of `(j, weight)` pairs in subgraph
///   coordinates.
/// - `original_indices[i]` is the original node index for subgraph node `i`.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if any node index in `node_set`
/// is out of range.
pub fn induced_subgraph(
    adjacency: &[Vec<(usize, f64)>],
    node_set: &[usize],
) -> Result<InducedSubgraphResult> {
    let n = adjacency.len();
    for &v in node_set {
        if v >= n {
            return Err(GraphError::invalid_parameter(
                "node_set entry",
                v,
                format!("must be < n_nodes ({})", n),
            ));
        }
    }

    // Deduplicate and sort to get a stable ordering.
    let mut original_indices: Vec<usize> = {
        let mut s: Vec<usize> = node_set.to_vec();
        s.sort_unstable();
        s.dedup();
        s
    };
    original_indices.sort_unstable();

    let sub_n = original_indices.len();

    // Build reverse map: original_index → subgraph_index.
    let mut rev_map: HashMap<usize, usize> = HashMap::with_capacity(sub_n);
    for (sub_i, &orig_i) in original_indices.iter().enumerate() {
        rev_map.insert(orig_i, sub_i);
    }

    // Build subgraph adjacency.
    let mut sub_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); sub_n];
    for (sub_i, &orig_i) in original_indices.iter().enumerate() {
        for &(orig_j, w) in &adjacency[orig_i] {
            if let Some(&sub_j) = rev_map.get(&orig_j) {
                sub_adj[sub_i].push((sub_j, w));
            }
        }
    }

    Ok((sub_adj, original_indices))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────

    /// Path graph  0 – 1 – 2 – … – (n-1)  (unweighted)
    fn path_adj(n: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![vec![]; n];
        for i in 0..n.saturating_sub(1) {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        adj
    }

    /// Two cliques of size k connected by a single bridge (unweighted)
    fn two_clique_adj(k: usize) -> Vec<Vec<usize>> {
        let n = 2 * k;
        let mut adj = vec![vec![]; n];
        for i in 0..k {
            for j in (i + 1)..k {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
        for i in 0..k {
            for j in (i + 1)..k {
                adj[k + i].push(k + j);
                adj[k + j].push(k + i);
            }
        }
        // Bridge: 0 — k
        adj[0].push(k);
        adj[k].push(0);
        adj
    }

    /// Weighted cycle  0–1–2–…–(n-1)–0
    fn weighted_cycle(n: usize) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![vec![]; n];
        for i in 0..n {
            let j = (i + 1) % n;
            adj[i].push((j, 1.0));
            adj[j].push((i, 1.0));
        }
        adj
    }

    // ── random_walk ────────────────────────────────────────────────────────

    #[test]
    fn test_random_walk_length() {
        let adj = path_adj(10);
        let walk = random_walk(&adj, 0, 8, 42).expect("random_walk");
        assert!(walk.len() <= 8, "walk too long: {}", walk.len());
        assert_eq!(walk[0], 0, "must start at start_node");
    }

    #[test]
    fn test_random_walk_all_valid_nodes() {
        let adj = two_clique_adj(5);
        let walk = random_walk(&adj, 0, 20, 7).expect("random_walk");
        let n = adj.len();
        for &node in &walk {
            assert!(node < n, "node {} out of range", node);
        }
    }

    #[test]
    fn test_random_walk_isolated_node_stops_early() {
        // Node 0 has no neighbours.
        let adj: Vec<Vec<usize>> = vec![vec![], vec![0]];
        let walk = random_walk(&adj, 0, 5, 0).expect("random_walk");
        // Should stop after the first step (no neighbours).
        assert_eq!(walk, vec![0]);
    }

    #[test]
    fn test_random_walk_zero_length() {
        let adj = path_adj(5);
        let walk = random_walk(&adj, 0, 0, 0).expect("random_walk");
        assert!(walk.is_empty());
    }

    #[test]
    fn test_random_walk_invalid_start() {
        let adj = path_adj(5);
        assert!(random_walk(&adj, 99, 5, 0).is_err());
    }

    #[test]
    fn test_random_walk_consecutive_valid_edges() {
        // Every consecutive pair in the walk must be an edge.
        let adj = two_clique_adj(4);
        let walk = random_walk(&adj, 0, 30, 123).expect("random_walk");
        for window in walk.windows(2) {
            let u = window[0];
            let v = window[1];
            assert!(
                adj[u].contains(&v),
                "edge ({}, {}) does not exist in adjacency list",
                u,
                v
            );
        }
    }

    // ── node2vec_walk ──────────────────────────────────────────────────────

    #[test]
    fn test_node2vec_walk_length() {
        let adj = weighted_cycle(8);
        let walk = node2vec_walk(&adj, 0, 10, 1.0, 1.0, 42).expect("node2vec_walk");
        assert!(walk.len() <= 10);
        assert_eq!(walk[0], 0);
    }

    #[test]
    fn test_node2vec_walk_all_valid_nodes() {
        let adj = weighted_cycle(6);
        let n = adj.len();
        let walk = node2vec_walk(&adj, 2, 20, 2.0, 0.5, 77).expect("node2vec_walk");
        for &v in &walk {
            assert!(v < n, "invalid node index {}", v);
        }
    }

    #[test]
    fn test_node2vec_walk_consecutive_edges() {
        let adj = weighted_cycle(6);
        let walk = node2vec_walk(&adj, 0, 15, 1.0, 1.0, 0).expect("node2vec_walk");
        let unweighted: Vec<Vec<usize>> = adj
            .iter()
            .map(|nbrs| nbrs.iter().map(|&(v, _)| v).collect())
            .collect();
        for w in walk.windows(2) {
            let u = w[0];
            let v = w[1];
            assert!(unweighted[u].contains(&v), "({}, {}) not an edge", u, v);
        }
    }

    #[test]
    fn test_node2vec_walk_invalid_p() {
        let adj = weighted_cycle(4);
        assert!(node2vec_walk(&adj, 0, 5, 0.0, 1.0, 0).is_err());
        assert!(node2vec_walk(&adj, 0, 5, -1.0, 1.0, 0).is_err());
    }

    #[test]
    fn test_node2vec_walk_invalid_q() {
        let adj = weighted_cycle(4);
        assert!(node2vec_walk(&adj, 0, 5, 1.0, 0.0, 0).is_err());
    }

    #[test]
    fn test_node2vec_walk_zero_length() {
        let adj = weighted_cycle(4);
        let walk = node2vec_walk(&adj, 0, 0, 1.0, 1.0, 0).expect("node2vec_walk");
        assert!(walk.is_empty());
    }

    #[test]
    fn test_node2vec_walk_length_one() {
        let adj = weighted_cycle(4);
        let walk = node2vec_walk(&adj, 1, 1, 1.0, 1.0, 0).expect("node2vec_walk");
        assert_eq!(walk, vec![1]);
    }

    // ── frontier_sampling ──────────────────────────────────────────────────

    #[test]
    fn test_frontier_sampling_basic() {
        let adj = two_clique_adj(5);
        let n = adj.len();
        let sample = frontier_sampling(&adj, n, 6, 42).expect("frontier_sampling");
        assert_eq!(sample.len(), 6);
        // All returned nodes must be valid.
        for &v in &sample {
            assert!(v < n);
        }
        // No duplicates.
        let set: HashSet<usize> = sample.iter().cloned().collect();
        assert_eq!(set.len(), sample.len());
    }

    #[test]
    fn test_frontier_sampling_full_graph() {
        let adj = path_adj(5);
        let sample = frontier_sampling(&adj, 5, 5, 0).expect("frontier_sampling");
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_frontier_sampling_zero_size() {
        let adj = path_adj(5);
        let sample = frontier_sampling(&adj, 5, 0, 0).expect("frontier_sampling");
        assert!(sample.is_empty());
    }

    #[test]
    fn test_frontier_sampling_invalid_n_nodes() {
        let adj: Vec<Vec<usize>> = vec![];
        assert!(frontier_sampling(&adj, 0, 1, 0).is_err());
    }

    #[test]
    fn test_frontier_sampling_sample_exceeds_n() {
        let adj = path_adj(3);
        assert!(frontier_sampling(&adj, 3, 5, 0).is_err());
    }

    #[test]
    fn test_frontier_sampling_sorted_output() {
        let adj = two_clique_adj(4);
        let n = adj.len();
        let sample = frontier_sampling(&adj, n, 5, 99).expect("frontier_sampling");
        let mut sorted = sample.clone();
        sorted.sort_unstable();
        assert_eq!(sample, sorted, "output must be sorted");
    }

    // ── forest_fire_sampling ───────────────────────────────────────────────

    #[test]
    fn test_forest_fire_basic() {
        let adj = two_clique_adj(5);
        let n = adj.len();
        let sample = forest_fire_sampling(&adj, n, 6, 0.7, 42).expect("forest_fire");
        assert_eq!(sample.len(), 6);
        for &v in &sample {
            assert!(v < n);
        }
        let set: HashSet<usize> = sample.iter().cloned().collect();
        assert_eq!(set.len(), sample.len());
    }

    #[test]
    fn test_forest_fire_full_graph() {
        let adj = path_adj(4);
        let sample = forest_fire_sampling(&adj, 4, 4, 0.5, 0).expect("forest_fire");
        assert_eq!(sample.len(), 4);
    }

    #[test]
    fn test_forest_fire_zero_size() {
        let adj = path_adj(5);
        let sample = forest_fire_sampling(&adj, 5, 0, 0.5, 0).expect("forest_fire");
        assert!(sample.is_empty());
    }

    #[test]
    fn test_forest_fire_invalid_prob() {
        let adj = path_adj(5);
        assert!(forest_fire_sampling(&adj, 5, 3, 0.0, 0).is_err());
        assert!(forest_fire_sampling(&adj, 5, 3, 1.0, 0).is_err());
        assert!(forest_fire_sampling(&adj, 5, 3, -0.5, 0).is_err());
    }

    #[test]
    fn test_forest_fire_sorted_output() {
        let adj = two_clique_adj(4);
        let n = adj.len();
        let sample = forest_fire_sampling(&adj, n, 5, 0.6, 13).expect("forest_fire");
        let mut sorted = sample.clone();
        sorted.sort_unstable();
        assert_eq!(sample, sorted);
    }

    // ── snowball_sampling ──────────────────────────────────────────────────

    #[test]
    fn test_snowball_sampling_zero_hops() {
        let adj = path_adj(8);
        let sample = snowball_sampling(&adj, &[3], 0).expect("snowball");
        assert_eq!(sample, vec![3]);
    }

    #[test]
    fn test_snowball_sampling_one_hop_path() {
        let adj = path_adj(6);
        // From node 3: neighbours are 2 and 4.
        let sample = snowball_sampling(&adj, &[3], 1).expect("snowball");
        let set: HashSet<usize> = sample.iter().cloned().collect();
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(set.contains(&4));
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_snowball_sampling_two_hops_path() {
        let adj = path_adj(7);
        // From node 3, 2 hops: nodes 1, 2, 3, 4, 5.
        let sample = snowball_sampling(&adj, &[3], 2).expect("snowball");
        let set: HashSet<usize> = sample.iter().cloned().collect();
        for v in [1, 2, 3, 4, 5] {
            assert!(set.contains(&v), "node {} missing", v);
        }
    }

    #[test]
    fn test_snowball_sampling_multiple_seeds() {
        let adj = path_adj(10);
        // Seeds 0 and 9 (endpoints) with 1 hop each.
        let sample = snowball_sampling(&adj, &[0, 9], 1).expect("snowball");
        let set: HashSet<usize> = sample.iter().cloned().collect();
        // From 0: {0, 1}; From 9: {8, 9}.
        assert!(set.contains(&0) && set.contains(&1));
        assert!(set.contains(&8) && set.contains(&9));
    }

    #[test]
    fn test_snowball_sampling_empty_adj() {
        let adj: Vec<Vec<usize>> = vec![];
        assert!(snowball_sampling(&adj, &[0], 1).is_err());
    }

    #[test]
    fn test_snowball_sampling_out_of_range_seed() {
        let adj = path_adj(4);
        assert!(snowball_sampling(&adj, &[99], 1).is_err());
    }

    #[test]
    fn test_snowball_sampling_sorted_no_duplicates() {
        let adj = two_clique_adj(4);
        let sample = snowball_sampling(&adj, &[0, 1], 2).expect("snowball");
        let mut sorted = sample.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sample, sorted, "output must be sorted with no duplicates");
    }

    // ── induced_subgraph ───────────────────────────────────────────────────

    #[test]
    fn test_induced_subgraph_basic() {
        //  0 ─ 1 ─ 2 ─ 3  (path graph, weighted)
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0), (3, 1.0)],
            vec![(2, 1.0)],
        ];
        // Take nodes {1, 2}.
        let (sub, orig) = induced_subgraph(&adj, &[1, 2]).expect("induced_subgraph");
        assert_eq!(orig, vec![1, 2]);
        assert_eq!(sub.len(), 2);
        // Subgraph node 0 (original 1) → subgraph node 1 (original 2) with w=1.0.
        assert_eq!(sub[0].len(), 1);
        assert_eq!(sub[0][0], (1, 1.0));
        // Subgraph node 1 (original 2) → subgraph node 0 (original 1).
        assert_eq!(sub[1].len(), 1);
        assert_eq!(sub[1][0], (0, 1.0));
    }

    #[test]
    fn test_induced_subgraph_no_internal_edges() {
        // Star graph centred at 0.
        let adj = vec![
            vec![(1, 1.0), (2, 1.0), (3, 1.0)],
            vec![(0, 1.0)],
            vec![(0, 1.0)],
            vec![(0, 1.0)],
        ];
        // Take leaves only: {1, 2, 3}. No edges among them.
        let (sub, orig) = induced_subgraph(&adj, &[1, 2, 3]).expect("induced_subgraph");
        assert_eq!(orig, vec![1, 2, 3]);
        for nbrs in &sub {
            assert!(
                nbrs.is_empty(),
                "leaves should have no edges among themselves"
            );
        }
    }

    #[test]
    fn test_induced_subgraph_full_graph() {
        let adj = vec![vec![(1, 2.0)], vec![(0, 2.0), (2, 3.0)], vec![(1, 3.0)]];
        let (sub, orig) = induced_subgraph(&adj, &[0, 1, 2]).expect("induced_subgraph");
        assert_eq!(orig, vec![0, 1, 2]);
        // Subgraph should equal the original.
        assert_eq!(sub, adj);
    }

    #[test]
    fn test_induced_subgraph_duplicates_in_node_set() {
        let adj = vec![vec![(1, 1.0)], vec![(0, 1.0), (2, 1.0)], vec![(1, 1.0)]];
        // Passing duplicates: {0, 0, 1} → should give sub on {0, 1}.
        let (sub, orig) = induced_subgraph(&adj, &[0, 0, 1]).expect("induced_subgraph");
        assert_eq!(orig, vec![0, 1]);
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn test_induced_subgraph_out_of_range() {
        let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        assert!(induced_subgraph(&adj, &[0, 99]).is_err());
    }

    #[test]
    fn test_induced_subgraph_empty_node_set() {
        let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let (sub, orig) = induced_subgraph(&adj, &[]).expect("induced_subgraph");
        assert!(sub.is_empty());
        assert!(orig.is_empty());
    }

    #[test]
    fn test_induced_subgraph_preserves_weights() {
        //  0 ──(5.0)── 1 ──(3.0)── 2
        let adj = vec![vec![(1, 5.0)], vec![(0, 5.0), (2, 3.0)], vec![(1, 3.0)]];
        let (sub, _) = induced_subgraph(&adj, &[0, 1]).expect("induced_subgraph");
        // sub[0] should contain (1, 5.0) in subgraph coords.
        assert_eq!(sub[0], vec![(1, 5.0)]);
        assert_eq!(sub[1], vec![(0, 5.0)]);
    }
}
