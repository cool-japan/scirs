//! Influence Maximization algorithms
//!
//! This module provides algorithms for finding the top-k seed nodes that
//! maximise information spread under a given diffusion model:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`greedy_influence_max`] | Greedy hill-climbing with Monte-Carlo estimates (Kempe 2003) |
//! | [`celf_influence_max`] | CELF – lazy evaluation cuts MC calls dramatically |
//! | [`celf_plus_plus`] | CELF++ – one additional optimisation over CELF |
//! | [`degree_heuristic`] | Fast O(n log n) heuristic: pick highest-degree nodes |
//! | [`pagerank_heuristic`] | PageRank-based seed selection (directed influence proxy) |
//!
//! # References
//! - Kempe, Kleinberg & Tardos (2003) – *KDD 2003*
//! - Leskovec et al. (2007) – CELF, *KDD 2007*
//! - Goyal, Lu & Lakshmanan (2011) – CELF++, *WWW 2011*

use std::collections::{BinaryHeap, HashMap};

use crate::diffusion::models::{simulate_ic, AdjList};
use crate::error::{GraphError, Result};

// ---------------------------------------------------------------------------
// Configuration & result types
// ---------------------------------------------------------------------------

/// Configuration for influence maximization algorithms.
#[derive(Debug, Clone)]
pub struct InfluenceMaxConfig {
    /// Number of Monte-Carlo simulations used to estimate spread.
    pub num_simulations: usize,
    /// Diffusion model: `"ic"` (Independent Cascade) or `"lt"` (Linear Threshold).
    pub model: DiffusionModel,
}

/// Selector for which diffusion model to use during IM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionModel {
    /// Independent Cascade model.
    IC,
    /// Linear Threshold model.
    LT,
}

impl Default for InfluenceMaxConfig {
    fn default() -> Self {
        InfluenceMaxConfig {
            num_simulations: 100,
            model: DiffusionModel::IC,
        }
    }
}

/// Result returned by influence maximization routines.
#[derive(Debug, Clone)]
pub struct InfluenceMaxResult {
    /// Selected seed nodes in the order they were chosen.
    pub seeds: Vec<usize>,
    /// Estimated expected spread of the seed set.
    pub estimated_spread: f64,
    /// Number of oracle (Monte-Carlo) calls made during the run.
    pub oracle_calls: usize,
}

// ---------------------------------------------------------------------------
// Internal spread estimator
// ---------------------------------------------------------------------------

/// Estimate expected spread for a candidate seed set.
///
/// Uses Monte-Carlo averaging with the IC model (LT support can be added
/// similarly).  Returns `(spread_estimate, oracle_call_count)`.
fn estimate_spread(
    adjacency: &AdjList,
    num_nodes: usize,
    seeds: &[usize],
    config: &InfluenceMaxConfig,
) -> Result<(f64, usize)> {
    let n = config.num_simulations;
    if n == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_simulations".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "estimate_spread".to_string(),
        });
    }

    let spread = match config.model {
        DiffusionModel::IC => {
            let mut total = 0.0_f64;
            for _ in 0..n {
                total += simulate_ic(adjacency, seeds)?.spread as f64;
            }
            total / n as f64
        }
        DiffusionModel::LT => {
            use crate::diffusion::models::simulate_lt;
            let mut total = 0.0_f64;
            for _ in 0..n {
                total += simulate_lt(adjacency, num_nodes, seeds, None)?.spread as f64;
            }
            total / n as f64
        }
    };

    Ok((spread, n))
}

// ---------------------------------------------------------------------------
// Greedy (Kempe et al. 2003)
// ---------------------------------------------------------------------------

/// Greedy influence maximization using Monte-Carlo spread estimates.
///
/// At each of the `k` iterations the algorithm evaluates every non-seed node
/// as a candidate addition and picks the one with the highest *marginal gain*.
/// This is the algorithm of Kempe, Kleinberg & Tardos (KDD 2003) with a
/// `(1 – 1/e)`-approximation guarantee for submodular diffusion models.
///
/// **Complexity**: `O(k · n · num_simulations)` MC simulations.
///
/// # Arguments
/// * `adjacency` — directed adjacency list with propagation probabilities.
/// * `num_nodes` — total number of nodes.
/// * `k` — desired seed set size.
/// * `config` — number of MC simulations and model choice.
///
/// # Errors
/// Returns an error when `k > num_nodes` or `num_simulations == 0`.
pub fn greedy_influence_max(
    adjacency: &AdjList,
    num_nodes: usize,
    k: usize,
    config: &InfluenceMaxConfig,
) -> Result<InfluenceMaxResult> {
    if k == 0 {
        return Ok(InfluenceMaxResult {
            seeds: Vec::new(),
            estimated_spread: 0.0,
            oracle_calls: 0,
        });
    }
    if k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "greedy_influence_max".to_string(),
        });
    }

    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    let mut current_spread = 0.0_f64;
    let mut oracle_calls = 0_usize;
    let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for _round in 0..k {
        let mut best_node = None;
        let mut best_gain = f64::NEG_INFINITY;

        for candidate in 0..num_nodes {
            if selected.contains(&candidate) {
                continue;
            }
            let mut trial_seeds = seeds.clone();
            trial_seeds.push(candidate);
            let (spread, calls) = estimate_spread(adjacency, num_nodes, &trial_seeds, config)?;
            oracle_calls += calls;

            let gain = spread - current_spread;
            if gain > best_gain {
                best_gain = gain;
                best_node = Some((candidate, spread));
            }
        }

        match best_node {
            Some((node, spread)) => {
                seeds.push(node);
                selected.insert(node);
                current_spread = spread;
            }
            None => break,
        }
    }

    Ok(InfluenceMaxResult {
        estimated_spread: current_spread,
        seeds,
        oracle_calls,
    })
}

// ---------------------------------------------------------------------------
// CELF (lazy evaluation)
// ---------------------------------------------------------------------------

/// CELF entry in the priority queue.
#[derive(Debug, Clone)]
struct CelfEntry {
    node: usize,
    marginal_gain: f64,
    /// Round in which `marginal_gain` was last computed.
    round: usize,
    /// Flag used by CELF++ to avoid one extra re-evaluation per round.
    prev_best: bool,
}

impl PartialEq for CelfEntry {
    fn eq(&self, other: &Self) -> bool {
        self.marginal_gain == other.marginal_gain && self.node == other.node
    }
}

impl Eq for CelfEntry {}

impl PartialOrd for CelfEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CelfEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.marginal_gain
            .partial_cmp(&other.marginal_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.node.cmp(&other.node))
    }
}

/// CELF influence maximization (lazy evaluation).
///
/// CELF exploits the *submodularity* of influence spread: the marginal gain of
/// a node can only decrease as the seed set grows.  A node whose stored
/// marginal gain is an upper-bound (from a previous round) does not need
/// re-evaluation until it reaches the top of the heap.
///
/// **Expected complexity**: `O(k · num_simulations)` MC calls (empirically
/// much fewer than the naive greedy).
///
/// # Arguments
/// * `adjacency` — directed adjacency list.
/// * `num_nodes` — total number of nodes.
/// * `k` — seed set size.
/// * `config` — MC count and model.
pub fn celf_influence_max(
    adjacency: &AdjList,
    num_nodes: usize,
    k: usize,
    config: &InfluenceMaxConfig,
) -> Result<InfluenceMaxResult> {
    if k == 0 {
        return Ok(InfluenceMaxResult {
            seeds: Vec::new(),
            estimated_spread: 0.0,
            oracle_calls: 0,
        });
    }
    if k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "celf_influence_max".to_string(),
        });
    }

    let mut oracle_calls = 0_usize;

    // Initialise heap with marginal gains of singletons
    let mut heap: BinaryHeap<CelfEntry> = BinaryHeap::new();
    for node in 0..num_nodes {
        let (gain, calls) = estimate_spread(adjacency, num_nodes, &[node], config)?;
        oracle_calls += calls;
        heap.push(CelfEntry {
            node,
            marginal_gain: gain,
            round: 0,
            prev_best: false,
        });
    }

    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    let mut current_spread = 0.0_f64;
    let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();

    let mut round = 0_usize;
    while seeds.len() < k {
        let entry = loop {
            let top = heap.pop().ok_or_else(|| GraphError::AlgorithmFailure {
                algorithm: "celf_influence_max".to_string(),
                reason: "priority queue exhausted before k seeds selected".to_string(),
                iterations: seeds.len(),
                tolerance: 0.0,
            })?;

            if selected.contains(&top.node) {
                continue;
            }

            if top.round == round {
                // Already evaluated in this round — guaranteed optimal by submodularity
                break top;
            }

            // Re-evaluate marginal gain
            let mut trial = seeds.clone();
            trial.push(top.node);
            let (new_spread, calls) = estimate_spread(adjacency, num_nodes, &trial, config)?;
            oracle_calls += calls;

            let updated = CelfEntry {
                node: top.node,
                marginal_gain: new_spread - current_spread,
                round,
                prev_best: false,
            };
            heap.push(updated);
        };

        seeds.push(entry.node);
        selected.insert(entry.node);
        current_spread += entry.marginal_gain;
        round += 1;
    }

    // Final spread estimate with the full seed set
    let (final_spread, calls) = estimate_spread(adjacency, num_nodes, &seeds, config)?;
    oracle_calls += calls;

    Ok(InfluenceMaxResult {
        seeds,
        estimated_spread: final_spread,
        oracle_calls,
    })
}

// ---------------------------------------------------------------------------
// CELF++ (Goyal et al. 2011)
// ---------------------------------------------------------------------------

/// CELF++ influence maximization.
///
/// CELF++ adds one optimisation over CELF: within each round it tracks the
/// *second* node that was the best in the previous iteration.  If a node at
/// the top of the heap was also the best in the previous round (flag
/// `prev_best`), its marginal gain with the current seed set has already been
/// computed and can be used without re-evaluation.
///
/// In practice this reduces oracle calls by roughly 35–55 % compared to CELF.
pub fn celf_plus_plus(
    adjacency: &AdjList,
    num_nodes: usize,
    k: usize,
    config: &InfluenceMaxConfig,
) -> Result<InfluenceMaxResult> {
    if k == 0 {
        return Ok(InfluenceMaxResult {
            seeds: Vec::new(),
            estimated_spread: 0.0,
            oracle_calls: 0,
        });
    }
    if k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "celf_plus_plus".to_string(),
        });
    }

    let mut oracle_calls = 0_usize;

    // ------ initialise heap with singleton marginal gains ------
    let mut heap: BinaryHeap<CelfEntry> = BinaryHeap::new();
    // Also track per-node cached gains for the CELF++ prev_best optimisation
    let mut cached_gain: HashMap<usize, f64> = HashMap::new();

    for node in 0..num_nodes {
        let (gain, calls) = estimate_spread(adjacency, num_nodes, &[node], config)?;
        oracle_calls += calls;
        cached_gain.insert(node, gain);
        heap.push(CelfEntry {
            node,
            marginal_gain: gain,
            round: 0,
            prev_best: false,
        });
    }

    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    let mut current_spread = 0.0_f64;
    let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut prev_best_node: Option<usize> = None;

    let mut round = 0_usize;
    while seeds.len() < k {
        // ------ find best candidate for this round ------
        let chosen = loop {
            let top = heap.pop().ok_or_else(|| GraphError::AlgorithmFailure {
                algorithm: "celf_plus_plus".to_string(),
                reason: "priority queue exhausted".to_string(),
                iterations: seeds.len(),
                tolerance: 0.0,
            })?;

            if selected.contains(&top.node) {
                continue;
            }

            // CELF++ optimisation: if this node was the best in the previous
            // round AND its gain was already re-evaluated w.r.t. the *current*
            // seed set, skip re-evaluation.
            if top.prev_best && top.round == round {
                break top;
            }

            if top.round == round {
                // Already updated this round
                break top;
            }

            // Re-evaluate marginal gain w.r.t. current seed set
            let mut trial = seeds.clone();
            trial.push(top.node);
            let (new_spread, calls) = estimate_spread(adjacency, num_nodes, &trial, config)?;
            oracle_calls += calls;

            let gain = new_spread - current_spread;
            *cached_gain.entry(top.node).or_insert(gain) = gain;

            // CELF++ optimisation: also evaluate w.r.t. seeds + prev_best
            let is_prev_best = prev_best_node.map(|pb| pb == top.node).unwrap_or(false);
            let prev_best_flag = if let Some(pb) = prev_best_node {
                if !selected.contains(&pb) && !is_prev_best {
                    let mut trial2 = seeds.clone();
                    trial2.push(pb);
                    trial2.push(top.node);
                    let (spread2, calls2) = estimate_spread(adjacency, num_nodes, &trial2, config)?;
                    oracle_calls += calls2;
                    let gain2 =
                        spread2 - current_spread - cached_gain.get(&pb).cloned().unwrap_or(0.0);
                    // If gain2 >= gain the node is still best even after adding prev_best
                    gain2 >= gain
                } else {
                    false
                }
            } else {
                false
            };

            let updated = CelfEntry {
                node: top.node,
                marginal_gain: gain,
                round,
                prev_best: prev_best_flag,
            };
            heap.push(updated);
        };

        prev_best_node = Some(chosen.node);
        seeds.push(chosen.node);
        selected.insert(chosen.node);
        current_spread += chosen.marginal_gain;
        round += 1;
    }

    let (final_spread, calls) = estimate_spread(adjacency, num_nodes, &seeds, config)?;
    oracle_calls += calls;

    Ok(InfluenceMaxResult {
        seeds,
        estimated_spread: final_spread,
        oracle_calls,
    })
}

// ---------------------------------------------------------------------------
// Degree heuristic
// ---------------------------------------------------------------------------

/// High-degree seed selection heuristic.
///
/// Selects the `k` nodes with highest out-degree as the seed set.  This is a
/// fast `O(n log n)` heuristic that often performs surprisingly well in
/// practice.
///
/// # Arguments
/// * `adjacency` — directed adjacency list.
/// * `num_nodes` — total number of nodes.
/// * `k` — seed set size.
/// * `config` — used only to compute the spread estimate at the end.
pub fn degree_heuristic(
    adjacency: &AdjList,
    num_nodes: usize,
    k: usize,
    config: &InfluenceMaxConfig,
) -> Result<InfluenceMaxResult> {
    if k == 0 {
        return Ok(InfluenceMaxResult {
            seeds: Vec::new(),
            estimated_spread: 0.0,
            oracle_calls: 0,
        });
    }
    if k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "degree_heuristic".to_string(),
        });
    }

    // Compute out-degree for every node
    let mut degrees: Vec<(usize, usize)> = (0..num_nodes)
        .map(|node| {
            let deg = adjacency.get(&node).map(|nbrs| nbrs.len()).unwrap_or(0);
            (node, deg)
        })
        .collect();

    // Sort descending by degree
    degrees.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let seeds: Vec<usize> = degrees.iter().take(k).map(|&(node, _)| node).collect();

    let (estimated_spread, oracle_calls) = estimate_spread(adjacency, num_nodes, &seeds, config)?;

    Ok(InfluenceMaxResult {
        seeds,
        estimated_spread,
        oracle_calls,
    })
}

// ---------------------------------------------------------------------------
// PageRank heuristic
// ---------------------------------------------------------------------------

/// PageRank-based seed selection heuristic.
///
/// Runs a lightweight power-iteration PageRank on the propagation graph and
/// selects the `k` highest-ranked nodes as seeds.  PageRank captures both
/// degree and structural position, making it a stronger proxy for influence
/// than raw degree.
///
/// # Arguments
/// * `adjacency` — directed adjacency list.
/// * `num_nodes` — total number of nodes.
/// * `k` — seed set size.
/// * `config` — MC config used for the spread estimate.
/// * `damping` — PageRank damping factor (typically 0.85).
/// * `max_iter` — maximum power-iteration steps.
/// * `tol` — convergence tolerance (L1 norm of score delta).
pub fn pagerank_heuristic(
    adjacency: &AdjList,
    num_nodes: usize,
    k: usize,
    config: &InfluenceMaxConfig,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> Result<InfluenceMaxResult> {
    if k == 0 {
        return Ok(InfluenceMaxResult {
            seeds: Vec::new(),
            estimated_spread: 0.0,
            oracle_calls: 0,
        });
    }
    if k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "pagerank_heuristic".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&damping) {
        return Err(GraphError::InvalidParameter {
            param: "damping".to_string(),
            value: damping.to_string(),
            expected: "[0, 1]".to_string(),
            context: "pagerank_heuristic".to_string(),
        });
    }

    // ------- compute out-degree for normalisation -------
    let out_degree: Vec<f64> = (0..num_nodes)
        .map(|n| adjacency.get(&n).map(|v| v.len() as f64).unwrap_or(0.0))
        .collect();

    // ------- power iteration -------
    let base_score = (1.0 - damping) / num_nodes as f64;
    let mut scores: Vec<f64> = vec![1.0 / num_nodes as f64; num_nodes];

    for _ in 0..max_iter {
        let mut new_scores: Vec<f64> = vec![base_score; num_nodes];

        // Dangling nodes contribute uniformly
        let dangling_sum: f64 = (0..num_nodes)
            .filter(|&n| out_degree[n] == 0.0)
            .map(|n| scores[n])
            .sum::<f64>()
            * damping
            / num_nodes as f64;

        for n in 0..num_nodes {
            new_scores[n] += dangling_sum;
        }

        // Regular contributions
        for (src, nbrs) in adjacency {
            let contrib = damping * scores[*src] / out_degree[*src];
            for &(tgt, _) in nbrs {
                if tgt < num_nodes {
                    new_scores[tgt] += contrib;
                }
            }
        }

        // Convergence check
        let delta: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        scores = new_scores;
        if delta < tol {
            break;
        }
    }

    // ------- select top-k -------
    let mut ranked: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let seeds: Vec<usize> = ranked.iter().take(k).map(|&(node, _)| node).collect();

    let (estimated_spread, oracle_calls) = estimate_spread(adjacency, num_nodes, &seeds, config)?;

    Ok(InfluenceMaxResult {
        seeds,
        estimated_spread,
        oracle_calls,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a path graph 0→1→2→3→4 with probability `p`.
    fn path_adj(n: usize, p: f64) -> AdjList {
        let mut adj: AdjList = HashMap::new();
        for i in 0..(n - 1) {
            adj.entry(i).or_default().push((i + 1, p));
        }
        adj
    }

    /// Star: hub 0 with spokes to 1..n, probability `p`.
    fn star_adj(n: usize, p: f64) -> AdjList {
        let mut adj: AdjList = HashMap::new();
        for i in 1..n {
            adj.entry(0).or_default().push((i, p));
        }
        adj
    }

    #[test]
    fn test_greedy_k1_selects_hub() {
        let adj = star_adj(6, 1.0);
        let config = InfluenceMaxConfig {
            num_simulations: 20,
            model: DiffusionModel::IC,
        };
        let result = greedy_influence_max(&adj, 6, 1, &config).expect("greedy");
        assert_eq!(result.seeds.len(), 1);
        // Hub (node 0) should be selected since it activates all 5 spokes
        assert_eq!(result.seeds[0], 0);
    }

    #[test]
    fn test_greedy_k0() {
        let adj = star_adj(5, 1.0);
        let config = InfluenceMaxConfig::default();
        let result = greedy_influence_max(&adj, 5, 0, &config).expect("k=0");
        assert!(result.seeds.is_empty());
        assert_eq!(result.estimated_spread, 0.0);
    }

    #[test]
    fn test_greedy_k_too_large() {
        let adj = star_adj(3, 1.0);
        let config = InfluenceMaxConfig::default();
        let err = greedy_influence_max(&adj, 3, 10, &config);
        assert!(err.is_err());
    }

    #[test]
    fn test_celf_selects_hub() {
        let adj = star_adj(6, 1.0);
        let config = InfluenceMaxConfig {
            num_simulations: 20,
            model: DiffusionModel::IC,
        };
        let result = celf_influence_max(&adj, 6, 1, &config).expect("celf");
        assert_eq!(result.seeds.len(), 1);
        assert_eq!(result.seeds[0], 0);
    }

    #[test]
    fn test_celf_pp_selects_hub() {
        let adj = star_adj(6, 1.0);
        let config = InfluenceMaxConfig {
            num_simulations: 20,
            model: DiffusionModel::IC,
        };
        let result = celf_plus_plus(&adj, 6, 1, &config).expect("celf++");
        assert_eq!(result.seeds.len(), 1);
        assert_eq!(result.seeds[0], 0);
    }

    #[test]
    fn test_degree_heuristic() {
        let adj = star_adj(6, 0.5);
        let config = InfluenceMaxConfig::default();
        let result = degree_heuristic(&adj, 6, 1, &config).expect("degree heuristic");
        // Node 0 has degree 5, all others 0
        assert_eq!(result.seeds[0], 0);
    }

    #[test]
    fn test_pagerank_heuristic() {
        let adj = star_adj(6, 1.0);
        let config = InfluenceMaxConfig {
            num_simulations: 20,
            model: DiffusionModel::IC,
        };
        let result =
            pagerank_heuristic(&adj, 6, 1, &config, 0.85, 100, 1e-6).expect("pagerank heuristic");
        assert_eq!(result.seeds.len(), 1);
    }

    #[test]
    fn test_degree_heuristic_k2() {
        // Two hubs: 0 has 4 spokes, 1 has 3 spokes
        let mut adj: AdjList = HashMap::new();
        for i in 2..6 {
            adj.entry(0).or_default().push((i, 0.5));
        }
        for i in 6..9 {
            adj.entry(1).or_default().push((i, 0.5));
        }
        let config = InfluenceMaxConfig::default();
        let result = degree_heuristic(&adj, 9, 2, &config).expect("degree k=2");
        assert_eq!(result.seeds.len(), 2);
        assert!(result.seeds.contains(&0));
        assert!(result.seeds.contains(&1));
    }

    #[test]
    fn test_greedy_path_k2() {
        let adj = path_adj(10, 1.0);
        let config = InfluenceMaxConfig {
            num_simulations: 30,
            model: DiffusionModel::IC,
        };
        let result = greedy_influence_max(&adj, 10, 2, &config).expect("greedy path");
        assert_eq!(result.seeds.len(), 2);
        // With prob 1.0, node 0 activates entire chain; node 0 should be chosen first
        assert!(result.seeds.contains(&0));
    }
}
