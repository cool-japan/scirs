//! Leiden community detection algorithm (Traag et al. 2019).
//!
//! The Leiden algorithm improves upon Louvain by guaranteeing that all communities
//! are internally well-connected (no disconnected communities). It introduces a
//! refinement phase between the local move and aggregation phases.
//!
//! ## Phases
//! 1. **Local move phase** – Same as Louvain: move nodes to neighbour communities
//!    that increase modularity.
//! 2. **Refinement phase** – Each community is refined by sub-partitioning it into
//!    well-connected subsets. The parameter `theta` controls the acceptance
//!    probability of random merge moves (higher theta = more randomness).
//! 3. **Aggregation phase** – Build a meta-graph from the refined partition.
//!
//! ## Reference
//! Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
//! guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233.

use std::collections::{HashMap, HashSet, VecDeque};

use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use super::louvain::compact_communities;
use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Leiden community detection algorithm.
#[derive(Debug, Clone)]
pub struct LeidenCommunity {
    /// Community assignment for each node (0-indexed, dense).
    pub assignments: Vec<usize>,
    /// Newman-Girvan modularity Q of the found partition.
    pub modularity: f64,
    /// Number of distinct communities discovered.
    pub n_communities: usize,
    /// Number of outer iterations completed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal sparse adjacency
// ─────────────────────────────────────────────────────────────────────────────

struct SparseAdj {
    adj: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>,
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
            if u != v {
                adj[v].push((u, w));
            }
            degree[u] += w;
            if u != v {
                degree[v] += w;
            }
            two_m += 2.0 * w;
        }
        Self { adj, degree, two_m }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Leiden community detection algorithm on a weighted edge-list graph.
///
/// # Arguments
/// * `adj`        – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`    – Total number of nodes.
/// * `resolution` – Resolution parameter γ (> 0; default 1.0).
/// * `theta`      – Refinement randomness parameter (0 < theta ≤ 1; default 0.01).
///   Small theta = deterministic refinement; large theta = random.
///
/// # Returns
/// [`LeidenCommunity`] with node assignments and modularity score.
pub fn leiden(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    resolution: f64,
    theta: f64,
) -> Result<LeidenCommunity> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "leiden: n_nodes must be > 0".into(),
        ));
    }
    if resolution <= 0.0 {
        return Err(GraphError::InvalidParameter {
            param: "resolution".into(),
            value: format!("{resolution}"),
            expected: "> 0".into(),
            context: "leiden".into(),
        });
    }
    if theta <= 0.0 || theta > 1.0 {
        return Err(GraphError::InvalidParameter {
            param: "theta".into(),
            value: format!("{theta}"),
            expected: "(0, 1]".into(),
            context: "leiden".into(),
        });
    }

    let mut rng = StdRng::seed_from_u64(0xdeadbeef_cafef00d);
    let original_adj = adj.to_vec();
    let original_n = n_nodes;

    let g = SparseAdj::from_edge_list(adj, n_nodes);
    if g.two_m == 0.0 {
        return Ok(LeidenCommunity {
            assignments: (0..n_nodes).collect(),
            modularity: 0.0,
            n_communities: n_nodes,
            iterations: 0,
        });
    }

    // Initial partition: each node in its own community
    let mut assignments: Vec<usize> = (0..n_nodes).collect();
    // meta-to-original mapping for translating aggregated community IDs back
    let mut meta_to_original: Vec<Vec<usize>> = (0..n_nodes).map(|i| vec![i]).collect();

    let mut current_edges: Vec<(usize, usize, f64)> = adj.to_vec();
    let mut current_n = n_nodes;
    let max_outer = 100usize;
    let mut iterations = 0usize;

    for _outer in 0..max_outer {
        iterations += 1;
        let sparse = SparseAdj::from_edge_list(&current_edges, current_n);

        // Phase 1: local moves (same as Louvain)
        let mut local_assign: Vec<usize> = (0..current_n).collect();
        let improved = local_move_phase(
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

        // Phase 2: refinement — split communities into well-connected subsets
        let mut refined_assign = local_assign.clone();
        refinement_phase(
            &sparse,
            &local_assign,
            &mut refined_assign,
            resolution,
            theta,
            &mut rng,
        );
        compact_communities(&mut refined_assign);

        // Translate refined assignments back to original nodes
        let new_n_comms = *refined_assign.iter().max().unwrap_or(&0) + 1;
        let mut new_meta: Vec<Vec<usize>> = vec![vec![]; new_n_comms];
        for (meta_node, &comm) in refined_assign.iter().enumerate() {
            for &orig in &meta_to_original[meta_node] {
                new_meta[comm].push(orig);
            }
        }
        for comm in 0..new_n_comms {
            for &orig in &new_meta[comm] {
                assignments[orig] = comm;
            }
        }
        meta_to_original = new_meta;

        // Phase 3: aggregation
        let (agg_edges, agg_n) = aggregate_graph(&current_edges, &refined_assign, new_n_comms);
        current_edges = agg_edges;
        current_n = agg_n;

        if current_n == 1 {
            break;
        }
    }

    let q = super::louvain::modularity(&original_adj, original_n, &assignments);
    let n_communities = *assignments.iter().max().unwrap_or(&0) + 1;

    Ok(LeidenCommunity {
        assignments,
        modularity: q,
        n_communities,
        iterations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: local move
// ─────────────────────────────────────────────────────────────────────────────

fn local_move_phase(
    g: &SparseAdj,
    assignments: &mut [usize],
    two_m: f64,
    resolution: f64,
    rng: &mut impl Rng,
) -> bool {
    let n = g.adj.len();
    let n_comms_init = *assignments.iter().max().unwrap_or(&0) + 1;
    let mut sigma_tot: Vec<f64> = vec![0.0; n_comms_init + n];
    for i in 0..n {
        let c = assignments[i];
        if c < sigma_tot.len() {
            sigma_tot[c] += g.degree[i];
        }
    }

    let mut improved = false;
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }

    for &node in &order {
        let current_comm = assignments[node];
        let k_i = g.degree[node];

        let mut comm_weights: HashMap<usize, f64> = HashMap::new();
        for &(nbr, w) in &g.adj[node] {
            let c = assignments[nbr];
            *comm_weights.entry(c).or_insert(0.0) += w;
        }

        let k_i_in_cur = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
        let remove_gain = k_i_in_cur / two_m
            - resolution * (sigma_tot[current_comm] - k_i) * k_i / (two_m * two_m);

        let mut best_comm = current_comm;
        let mut best_gain = 0.0f64;
        for (&c, &k_i_in_c) in &comm_weights {
            if c == current_comm {
                continue;
            }
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
// Phase 2: refinement
// ─────────────────────────────────────────────────────────────────────────────

/// Refinement phase: within each community from `coarse_assign`, try to
/// further split into well-connected sub-communities.
///
/// The key Leiden guarantee: each resulting sub-community is a connected
/// subset of the coarse community. We enforce this by only allowing merges
/// between nodes that are connected within the coarse community.
fn refinement_phase(
    g: &SparseAdj,
    coarse_assign: &[usize],
    refined_assign: &mut [usize],
    resolution: f64,
    theta: f64,
    rng: &mut impl Rng,
) {
    let n = g.adj.len();
    let n_coarse = *coarse_assign.iter().max().unwrap_or(&0) + 1;

    // Group nodes by coarse community
    let mut coarse_groups: Vec<Vec<usize>> = vec![vec![]; n_coarse];
    for i in 0..n {
        if coarse_assign[i] < n_coarse {
            coarse_groups[coarse_assign[i]].push(i);
        }
    }

    // We assign new fine-grained IDs starting from the next available ID
    // (refined_assign starts as a copy of coarse_assign)
    let mut next_id = n_coarse;

    for group in &coarse_groups {
        if group.len() <= 1 {
            continue;
        }
        // Try to split this community via random singleton moves
        refine_community(
            g,
            group,
            refined_assign,
            resolution,
            theta,
            rng,
            &mut next_id,
        );
    }
}

/// Refine a single community by attempting to move singleton nodes to sub-communities.
fn refine_community(
    g: &SparseAdj,
    group: &[usize],
    refined_assign: &mut [usize],
    resolution: f64,
    theta: f64,
    rng: &mut impl Rng,
    next_id: &mut usize,
) {
    let two_m: f64 = g.two_m;
    if two_m == 0.0 {
        return;
    }

    // Each node in this group starts in its own singleton sub-community
    let group_set: HashSet<usize> = group.iter().cloned().collect();

    // Local community IDs for the refinement (using next_id pool)
    let mut local_comm: HashMap<usize, usize> = HashMap::new();
    for &node in group {
        local_comm.insert(node, *next_id);
        *next_id += 1;
    }
    for &node in group {
        refined_assign[node] = local_comm.get(&node).copied().unwrap_or(node);
    }

    // sigma_tot for sub-communities (indexed by sub-community ID)
    let mut sigma_tot: HashMap<usize, f64> = HashMap::new();
    for &node in group {
        let c = local_comm[&node];
        *sigma_tot.entry(c).or_insert(0.0) += g.degree[node];
    }

    // Greedy merge pass with random acceptance (theta controls randomness)
    let max_iter = group.len() * 2;
    for _ in 0..max_iter {
        let node_idx = rng.random_range(0..group.len());
        let node = group[node_idx];
        let current_comm = local_comm[&node];
        let k_i = g.degree[node];

        // Collect weights to neighbouring sub-communities (within the same group)
        let mut comm_weights: HashMap<usize, f64> = HashMap::new();
        for &(nbr, w) in &g.adj[node] {
            if group_set.contains(&nbr) {
                let c = local_comm[&nbr];
                *comm_weights.entry(c).or_insert(0.0) += w;
            }
        }

        if comm_weights.is_empty() {
            continue;
        }

        // Compute gains for each candidate community
        let k_i_in_cur = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
        let cur_sigma = *sigma_tot.get(&current_comm).unwrap_or(&0.0);
        let remove_gain =
            k_i_in_cur / two_m - resolution * (cur_sigma - k_i) * k_i / (two_m * two_m);

        let mut candidates: Vec<(usize, f64)> = Vec::new();
        for (&c, &k_i_in_c) in &comm_weights {
            if c == current_comm {
                continue;
            }
            let sigma_c = *sigma_tot.get(&c).unwrap_or(&0.0);
            let gain =
                k_i_in_c / two_m - resolution * sigma_c * k_i / (two_m * two_m) - remove_gain;
            if gain > 0.0 {
                candidates.push((c, gain));
            }
        }

        if candidates.is_empty() {
            continue;
        }

        // Select best or random based on theta
        let chosen_comm = if rng.random::<f64>() < theta {
            // Random selection among positive-gain communities
            let idx = rng.random_range(0..candidates.len());
            candidates[idx].0
        } else {
            // Deterministic: pick best gain
            candidates
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&(c, _)| c)
                .unwrap_or(current_comm)
        };

        if chosen_comm != current_comm {
            *sigma_tot.entry(current_comm).or_insert(0.0) -= k_i;
            *sigma_tot.entry(chosen_comm).or_insert(0.0) += k_i;
            local_comm.insert(node, chosen_comm);
            refined_assign[node] = chosen_comm;
        }
    }

    // Ensure all sub-communities are connected within the original graph
    enforce_connectivity(g, group, &group_set, refined_assign);
}

/// Enforce that each sub-community within a group is connected.
/// If disconnected, split into connected components.
fn enforce_connectivity(
    g: &SparseAdj,
    group: &[usize],
    group_set: &HashSet<usize>,
    refined_assign: &mut [usize],
) {
    // Group nodes by their current refined community
    let mut sub_comms: HashMap<usize, Vec<usize>> = HashMap::new();
    for &node in group {
        sub_comms
            .entry(refined_assign[node])
            .or_default()
            .push(node);
    }

    let mut max_id = *refined_assign.iter().max().unwrap_or(&0);

    for nodes in sub_comms.values() {
        if nodes.len() <= 1 {
            continue;
        }
        // BFS within this sub-community to find connected components
        let node_set: HashSet<usize> = nodes.iter().cloned().collect();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut comp_id = refined_assign[nodes[0]]; // first component keeps current ID

        for &start in nodes {
            if visited.contains(&start) {
                continue;
            }
            // BFS
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited.insert(start);
            while let Some(v) = queue.pop_front() {
                refined_assign[v] = comp_id;
                for &(nbr, _) in &g.adj[v] {
                    if node_set.contains(&nbr)
                        && group_set.contains(&nbr)
                        && !visited.contains(&nbr)
                    {
                        visited.insert(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
            // Next component gets a new ID
            max_id += 1;
            comp_id = max_id;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: aggregation
// ─────────────────────────────────────────────────────────────────────────────

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
        edges.push((0, k, 0.05));
        (edges, n)
    }

    #[test]
    fn test_leiden_two_cliques() {
        let (edges, n) = two_clique_edges(4);
        let result = leiden(&edges, n, 1.0, 0.01).expect("leiden");
        assert!(
            result.modularity > 0.0,
            "modularity should be positive: {}",
            result.modularity
        );
        assert!(result.n_communities >= 2, "should find >= 2 communities");
        // Verify that nodes in the same clique are in the same community
        let c0 = result.assignments[0];
        for i in 1..4 {
            assert_eq!(result.assignments[i], c0, "clique1 node {} wrong", i);
        }
        let c1 = result.assignments[4];
        for i in 5..8 {
            assert_eq!(result.assignments[i], c1, "clique2 node {} wrong", i);
        }
        assert_ne!(c0, c1, "two cliques must be in different communities");
    }

    #[test]
    fn test_leiden_invalid_params() {
        let edges = vec![(0usize, 1usize, 1.0f64)];
        assert!(leiden(&edges, 2, 0.0, 0.5).is_err()); // resolution <= 0
        assert!(leiden(&edges, 2, 1.0, 0.0).is_err()); // theta <= 0
        assert!(leiden(&edges, 2, 1.0, 1.5).is_err()); // theta > 1
        assert!(leiden(&edges, 0, 1.0, 0.5).is_err()); // n_nodes == 0
    }

    #[test]
    fn test_leiden_single_node() {
        let result = leiden(&[], 1, 1.0, 0.5).expect("leiden single node");
        assert_eq!(result.n_communities, 1);
    }

    #[test]
    fn test_leiden_modularity_positive() {
        let (edges, n) = two_clique_edges(5);
        let result = leiden(&edges, n, 1.0, 0.05).expect("leiden modularity");
        assert!(result.modularity > 0.0);
    }
}
