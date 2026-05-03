//! Infomap community detection algorithm (Rosvall & Bergstrom 2008).
//!
//! Infomap uses the **map equation** to find the partition of a network that
//! minimises the expected description length of a random walk trajectory.
//!
//! The map equation is:
//!
//! ```text
//! L(M) = q_↷ · H(Q) + Σ_i  p_i^↺ · H(P_i)
//! ```
//!
//! where:
//! - `q_↷` is the probability of exiting a module in one step,
//! - `H(Q)` is the entropy of the module-transition process,
//! - `p_i^↺` is the fraction of time spent inside module `i`,
//! - `H(P_i)` is the entropy of within-module movements.
//!
//! ## Implementation
//! This implementation uses a greedy optimisation with multiple random restarts
//! (`n_trials`). In each trial:
//! 1. Initialise with a random partition.
//! 2. Iteratively move nodes to neighbour modules if the map equation decreases.
//! 3. Compact community IDs and evaluate the final code length.
//!
//! The trial with the lowest map equation (best compression) is returned.
//!
//! ## Reference
//! Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex
//! networks reveal community structure. *Proceedings of the National Academy
//! of Sciences*, 105(4), 1118–1123.

use std::collections::HashMap;

use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use super::louvain::compact_communities;
use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Infomap algorithm.
#[derive(Debug, Clone)]
pub struct InfomapConfig {
    /// Number of independent restarts (best result is returned).
    pub n_trials: usize,
    /// Maximum iterations per trial.
    pub max_iter: usize,
    /// Convergence tolerance on the map equation.
    pub tol: f64,
}

impl Default for InfomapConfig {
    fn default() -> Self {
        Self {
            n_trials: 10,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal structures
// ─────────────────────────────────────────────────────────────────────────────

struct SparseAdj {
    adj: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>,
    two_m: f64,
}

impl SparseAdj {
    fn from_edge_list(edges: &[(usize, usize, f64)], n: usize) -> Self {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        let mut degree = vec![0.0f64; n];
        let mut two_m = 0.0f64;

        for &(u, v, w) in edges {
            if u >= n || v >= n {
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

    /// Compute stationary distribution (proportional to degree for undirected graphs).
    fn stationary(&self) -> Vec<f64> {
        let total = self.two_m;
        if total == 0.0 {
            let n = self.adj.len();
            return vec![1.0 / n.max(1) as f64; n];
        }
        self.degree.iter().map(|&d| d / total).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Map equation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the map equation code length for a given partition.
///
/// L(M) = q_↷ · H(Q) + Σ_i p_i^↺ · H(P_i)
fn map_equation(g: &SparseAdj, assignments: &[usize]) -> f64 {
    let n = g.adj.len();
    if n == 0 || g.two_m == 0.0 {
        return 0.0;
    }

    let pi = g.stationary(); // stationary distribution
    let n_comms = assignments.iter().max().copied().unwrap_or(0) + 1;

    // For each module i: visit rate, exit rate
    let mut module_visit: Vec<f64> = vec![0.0; n_comms];
    let mut module_exit: Vec<f64> = vec![0.0; n_comms];

    for node in 0..n {
        let c = assignments[node];
        if c >= n_comms {
            continue;
        }
        module_visit[c] += pi[node];
        // Exit rate: fraction of random walk steps leaving module c from node
        let mut exit_w = 0.0f64;
        let mut total_w = 0.0f64;
        for &(nbr, w) in &g.adj[node] {
            total_w += w;
            if nbr < assignments.len() && assignments[nbr] != c {
                exit_w += w;
            }
        }
        if total_w > 0.0 {
            module_exit[c] += pi[node] * exit_w / total_w;
        }
    }

    let q_total: f64 = module_exit.iter().sum();

    // H(Q): entropy of module exit process
    let h_q = if q_total > 0.0 {
        -module_exit
            .iter()
            .filter(|&&q| q > 0.0)
            .map(|&q| {
                let p = q / q_total;
                p * p.ln()
            })
            .sum::<f64>()
    } else {
        0.0
    };

    // H(P_i): within-module entropy for each module i
    let mut h_modules = 0.0f64;
    for i in 0..n_comms {
        let p_stay = module_visit[i] - module_exit[i];
        let p_total = module_visit[i] + module_exit[i]; // total flow through module
        if p_total <= 0.0 {
            continue;
        }
        // Within-module distribution: node visits + self-exit
        let mut within: Vec<f64> = Vec::new();
        for node in 0..n {
            if assignments[node] == i {
                within.push(pi[node]);
            }
        }
        within.push(module_exit[i]); // "exit codeword"

        let entropy: f64 = within
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let frac = v / p_total;
                if frac > 0.0 {
                    -frac * frac.ln()
                } else {
                    0.0
                }
            })
            .sum();
        h_modules += (p_total) * entropy;
        let _ = p_stay;
    }

    q_total * h_q + h_modules
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Infomap community detection algorithm.
///
/// # Arguments
/// * `adj`    – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes` – Total number of nodes.
/// * `config` – Algorithm configuration (`n_trials`, `max_iter`, `tol`).
///
/// # Returns
/// Community assignment vector (0-indexed, densely numbered).
pub fn infomap(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    config: &InfomapConfig,
) -> Result<Vec<usize>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "infomap: n_nodes must be > 0".into(),
        ));
    }

    let g = SparseAdj::from_edge_list(adj, n_nodes);
    if g.two_m == 0.0 {
        // Isolated graph: each node is its own community
        return Ok((0..n_nodes).collect());
    }

    let mut best_assignments: Option<Vec<usize>> = None;
    let mut best_code_length = f64::INFINITY;

    for trial in 0..config.n_trials.max(1) {
        let seed = 0xabcdef01_u64.wrapping_add(trial as u64 * 0x9e3779b9);
        let result = infomap_trial(&g, config, seed)?;
        let code_len = map_equation(&g, &result);
        if code_len < best_code_length {
            best_code_length = code_len;
            best_assignments = Some(result);
        }
    }

    let mut assignments = best_assignments
        .ok_or_else(|| GraphError::AlgorithmError("infomap: no trials completed".into()))?;
    compact_communities(&mut assignments);
    Ok(assignments)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single trial
// ─────────────────────────────────────────────────────────────────────────────

fn infomap_trial(g: &SparseAdj, config: &InfomapConfig, seed: u64) -> Result<Vec<usize>> {
    let n = g.adj.len();
    let mut rng = StdRng::seed_from_u64(seed);

    // Initialise with random partition into ceil(sqrt(n)) modules
    let init_comms = ((n as f64).sqrt().ceil() as usize).max(1).min(n);
    let mut assignments: Vec<usize> = (0..n).map(|_| rng.random_range(0..init_comms)).collect();

    let mut prev_code_len = map_equation(g, &assignments);

    for _iter in 0..config.max_iter {
        let improved = infomap_move_phase(g, &mut assignments, &mut rng);
        if !improved {
            break;
        }
        compact_communities(&mut assignments);

        let code_len = map_equation(g, &assignments);
        if (prev_code_len - code_len).abs() < config.tol {
            break;
        }
        prev_code_len = code_len;
    }

    Ok(assignments)
}

// ─────────────────────────────────────────────────────────────────────────────
// Move phase
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy node-move phase: for each node, try moving to each neighbour's module.
/// Accept the move that most reduces the map equation.
fn infomap_move_phase(g: &SparseAdj, assignments: &mut [usize], rng: &mut impl Rng) -> bool {
    let n = g.adj.len();
    let mut improved = false;

    // Randomised order
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }

    let current_code = map_equation(g, assignments);
    let mut running_code = current_code;

    for &node in &order {
        let orig_comm = assignments[node];

        // Candidate modules: current module + all neighbour modules
        let mut candidate_comms: Vec<usize> = vec![orig_comm];
        for &(nbr, _) in &g.adj[node] {
            let c = assignments[nbr];
            if !candidate_comms.contains(&c) {
                candidate_comms.push(c);
            }
        }

        let mut best_comm = orig_comm;
        let mut best_code = running_code;

        for c in candidate_comms {
            if c == orig_comm {
                continue;
            }
            assignments[node] = c;
            let new_code = map_equation(g, assignments);
            if new_code < best_code {
                best_code = new_code;
                best_comm = c;
            }
        }

        assignments[node] = best_comm;
        if best_comm != orig_comm {
            running_code = best_code;
            improved = true;
        }
    }
    improved
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
    fn test_infomap_two_cliques() {
        let (edges, n) = two_clique_edges(4);
        let config = InfomapConfig {
            n_trials: 5,
            max_iter: 50,
            tol: 1e-6,
        };
        let labels = infomap(&edges, n, &config).expect("infomap");
        assert_eq!(labels.len(), 8);
        // Clique 1
        let l0 = labels[0];
        for i in 1..4 {
            assert_eq!(labels[i], l0, "clique1 node {} wrong", i);
        }
        // Clique 2
        let l1 = labels[4];
        for i in 5..8 {
            assert_eq!(labels[i], l1, "clique2 node {} wrong", i);
        }
        assert_ne!(l0, l1, "different communities expected");
    }

    #[test]
    fn test_infomap_empty_error() {
        let config = InfomapConfig::default();
        assert!(infomap(&[], 0, &config).is_err());
    }

    #[test]
    fn test_infomap_no_edges() {
        let config = InfomapConfig {
            n_trials: 1,
            max_iter: 10,
            tol: 1e-6,
        };
        let labels = infomap(&[], 4, &config).expect("infomap no edges");
        // Each isolated node in its own community
        assert_eq!(labels.len(), 4);
        let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_map_equation_perfect_partition() {
        let (edges, n) = two_clique_edges(3);
        let g = SparseAdj::from_edge_list(&edges, n);
        let perfect: Vec<usize> = (0..6).map(|i| if i < 3 { 0 } else { 1 }).collect();
        let single: Vec<usize> = vec![0; 6];
        let code_perfect = map_equation(&g, &perfect);
        let code_single = map_equation(&g, &single);
        // Two-community partition should have shorter (lower) code length
        assert!(
            code_perfect <= code_single + 1e-9,
            "perfect partition code={code_perfect}, single={code_single}"
        );
    }

    #[test]
    fn test_default_config() {
        let cfg = InfomapConfig::default();
        assert_eq!(cfg.n_trials, 10);
        assert_eq!(cfg.max_iter, 200);
        assert!(cfg.tol < 1e-5);
    }
}
