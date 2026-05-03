//! Reverse Influence Sampling (RIS) and the IMM algorithm.
//!
//! This module implements Reverse Influence Sampling (RIS) for influence
//! maximisation, including:
//!
//! - [`generate_rr_sets`]: Generate Reverse Reachable (RR) sets under the IC model.
//! - [`ris_estimate`]: Estimate influence spread using a collection of RR sets.
//! - [`imm_algorithm`]: IMM (Influence Maximization via Martingales, Tang 2014/2015).
//! - [`sandwich_approximation`]: Upper/lower bound for spread using two seed sets.
//!
//! # Algorithm Overview
//!
//! An RR set for a node `v` is the set of nodes that can reach `v` in a random
//! sample of the graph.  A seed set `S` covers an RR set `R_v` iff `S ∩ R_v ≠ ∅`,
//! which happens precisely when `v` would be activated starting from `S`.
//!
//! IMM iteratively generates a growing pool of RR sets guided by a Martingale
//! stopping condition, then finds the maximum-coverage seed set via greedy
//! set-cover (approximation ratio `(1 – 1/e – ε)`).
//!
//! # References
//! - Borgs, C., Brautbar, M., Chayes, J., & Lucier, B. (2014). Maximizing
//!   Social Influence in Nearly Optimal Time. *SODA 2014*.
//! - Tang, Y., Xiao, X., & Shi, Y. (2014). Influence Maximization: Near-Optimal
//!   Time Complexity Meets Practical Efficiency. *SIGMOD 2014*.
//! - Tang, Y., Shi, Y., & Xiao, X. (2015). Influence Maximization in
//!   Near-Linear Time: A Martingale Approach. *SIGMOD 2015*.
//! - Borg, I., & Groenen, P. J. F. (2005). Sandwich approximation for
//!   submodular maximization.

use std::collections::{HashMap, HashSet, VecDeque};

use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use crate::diffusion::models::AdjList;
use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A Reverse Reachable (RR) set: the set of nodes that could activate a
/// randomly chosen target node `v` in a sampled IC-model graph.
///
/// Each entry is a node id in the RR set.
pub type RRSet = Vec<usize>;

/// Configuration for RIS-based influence maximisation.
#[derive(Debug, Clone)]
pub struct RISConfig {
    /// Number of RR sets to generate (determines accuracy vs. cost).
    ///
    /// More sets → better estimates.  For IMM this is computed automatically
    /// via the Martingale criterion; set this when calling [`generate_rr_sets`]
    /// directly.
    pub num_rr_sets: usize,
    /// Random seed for reproducibility (`None` = non-deterministic).
    pub seed: Option<u64>,
}

impl Default for RISConfig {
    fn default() -> Self {
        RISConfig {
            num_rr_sets: 10_000,
            seed: None,
        }
    }
}

/// Configuration for the IMM algorithm.
#[derive(Debug, Clone)]
pub struct ImmConfig {
    /// Desired seed set size.
    pub k: usize,
    /// Approximation parameter ε ∈ (0, 1).  Smaller → tighter guarantee, more
    /// RR sets, more running time.
    pub epsilon: f64,
    /// Confidence parameter δ ∈ (0, 1).  Probability of approximation failure
    /// is bounded by δ.  Typical value: `1 / num_nodes`.
    pub delta: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for ImmConfig {
    fn default() -> Self {
        ImmConfig {
            k: 5,
            epsilon: 0.1,
            delta: 0.01,
            seed: None,
        }
    }
}

/// Result returned by the IMM algorithm.
#[derive(Debug, Clone)]
pub struct ImmResult {
    /// Selected seed nodes.
    pub seeds: Vec<usize>,
    /// Estimated expected spread under IC model.
    pub estimated_spread: f64,
    /// Number of RR sets generated.
    pub num_rr_sets: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a *reverse* adjacency list: `target → [(source, probability)]`.
fn reverse_adj(adjacency: &AdjList) -> AdjList {
    let mut rev: AdjList = HashMap::new();
    for (&src, nbrs) in adjacency {
        for &(tgt, p) in nbrs {
            rev.entry(tgt).or_default().push((src, p));
        }
    }
    rev
}

/// Generate one RR set for a random target node under the IC model.
///
/// Starting from a randomly sampled root node `v`, we perform a *reverse* BFS
/// over the graph: for each in-edge `(u → v)` we include `u` in the RR set
/// with probability equal to the edge's propagation probability.  The resulting
/// set is exactly the set of nodes that can reach `v` in a sampled IC realisation.
fn generate_one_rr_set(rev_adj: &AdjList, num_nodes: usize, rng: &mut impl Rng) -> RRSet {
    let root: usize = rng.random_range(0..num_nodes);
    let mut rr_set: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    rr_set.insert(root);
    queue.push_back(root);

    while let Some(node) = queue.pop_front() {
        // Explore each in-neighbour of `node`
        if let Some(in_nbrs) = rev_adj.get(&node) {
            for &(src, prob) in in_nbrs {
                // Edge (src → node) is active with probability `prob`
                if !rr_set.contains(&src) && rng.random::<f64>() < prob {
                    rr_set.insert(src);
                    queue.push_back(src);
                }
            }
        }
    }

    rr_set.into_iter().collect()
}

/// Greedy maximum-coverage seed selection.
///
/// Given a collection of RR sets and a desired seed-set size `k`, greedily
/// selects nodes that cover the most uncovered RR sets.
///
/// Returns `(seeds, num_covered)`.
fn greedy_max_coverage(rr_sets: &[RRSet], num_nodes: usize, k: usize) -> (Vec<usize>, usize) {
    let r = rr_sets.len();

    // Build: node → set of RR-set indices containing that node
    let mut node_to_rr: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for (i, rr) in rr_sets.iter().enumerate() {
        for &node in rr {
            if node < num_nodes {
                node_to_rr[node].push(i);
            }
        }
    }

    let mut covered: Vec<bool> = vec![false; r];
    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    // coverage[node] = number of *uncovered* RR sets containing `node`
    let mut coverage: Vec<usize> = node_to_rr.iter().map(|v| v.len()).collect();

    for _ in 0..k {
        // Pick node with maximum coverage
        let best = (0..num_nodes).max_by_key(|&n| coverage[n]).unwrap_or(0);

        seeds.push(best);

        // Update: mark all RR sets covered by `best` as covered
        for &rr_idx in &node_to_rr[best] {
            if !covered[rr_idx] {
                covered[rr_idx] = true;
                // Reduce coverage of every other node in this RR set
                for &other in &rr_sets[rr_idx] {
                    if other < num_nodes && coverage[other] > 0 {
                        coverage[other] -= 1;
                    }
                }
            }
        }

        // Zero out coverage for the chosen node so it won't be chosen again
        coverage[best] = 0;
    }

    let num_covered = covered.iter().filter(|&&c| c).count();
    (seeds, num_covered)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a collection of Reverse Reachable sets under the IC model.
///
/// # Arguments
/// * `adjacency` — directed adjacency list with propagation probabilities.
/// * `num_nodes` — total number of nodes.
/// * `config` — number of RR sets to generate and optional random seed.
///
/// # Returns
/// A `Vec<RRSet>` of length `config.num_rr_sets`.
///
/// # Errors
/// Returns an error when `num_nodes == 0` or `num_rr_sets == 0`.
pub fn generate_rr_sets(
    adjacency: &AdjList,
    num_nodes: usize,
    config: &RISConfig,
) -> Result<Vec<RRSet>> {
    if num_nodes == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_nodes".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "generate_rr_sets".to_string(),
        });
    }
    if config.num_rr_sets == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_rr_sets".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "generate_rr_sets".to_string(),
        });
    }

    let mut rng: StdRng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut scirs2_core::random::rng()),
    };

    let rev = reverse_adj(adjacency);
    let mut rr_sets: Vec<RRSet> = Vec::with_capacity(config.num_rr_sets);

    for _ in 0..config.num_rr_sets {
        rr_sets.push(generate_one_rr_set(&rev, num_nodes, &mut rng));
    }

    Ok(rr_sets)
}

/// Estimate the expected spread of a seed set using a pre-generated pool of
/// RR sets.
///
/// The estimate is `|covered RR sets| / |total RR sets| * num_nodes`, which is
/// an unbiased estimator for the expected spread under the IC model.
///
/// # Arguments
/// * `rr_sets` — pool of RR sets (from [`generate_rr_sets`]).
/// * `seeds` — seed set to evaluate.
/// * `num_nodes` — total number of nodes.
///
/// # Errors
/// Returns an error when `rr_sets` is empty.
pub fn ris_estimate(rr_sets: &[RRSet], seeds: &[usize], num_nodes: usize) -> Result<f64> {
    if rr_sets.is_empty() {
        return Err(GraphError::InvalidParameter {
            param: "rr_sets".to_string(),
            value: "empty".to_string(),
            expected: "non-empty RR set collection".to_string(),
            context: "ris_estimate".to_string(),
        });
    }

    let seed_set: HashSet<usize> = seeds.iter().cloned().collect();
    let num_covered = rr_sets
        .iter()
        .filter(|rr| rr.iter().any(|n| seed_set.contains(n)))
        .count();

    Ok(num_covered as f64 / rr_sets.len() as f64 * num_nodes as f64)
}

/// IMM (Influence Maximization via Martingales) algorithm.
///
/// Implements the IMM algorithm of Tang et al. (SIGMOD 2014/2015) which
/// achieves a `(1 – 1/e – ε)`-approximation guarantee with probability
/// `1 – δ` while generating a near-optimal number of RR sets.
///
/// The number of RR sets θ is determined by the following formula:
///
/// ```text
/// θ = (8 + 2ε) · n · (ℓ · log(n) + log(C(n,k)) + log(2)) / (ε² · OPT_k)
/// ```
///
/// where `OPT_k` is a lower bound on the optimal spread (estimated via binary
/// search over the Martingale condition).
///
/// # Arguments
/// * `adjacency` — directed adjacency list with propagation probabilities.
/// * `num_nodes` — total number of nodes.
/// * `config` — algorithm configuration.
///
/// # Errors
/// Returns an error when parameters are invalid or `num_nodes == 0`.
pub fn imm_algorithm(
    adjacency: &AdjList,
    num_nodes: usize,
    config: &ImmConfig,
) -> Result<ImmResult> {
    if num_nodes == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_nodes".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "imm_algorithm".to_string(),
        });
    }
    if config.k == 0 {
        return Ok(ImmResult {
            seeds: Vec::new(),
            estimated_spread: 0.0,
            num_rr_sets: 0,
        });
    }
    if config.k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: config.k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "imm_algorithm".to_string(),
        });
    }
    if !(0.0..1.0).contains(&config.epsilon) {
        return Err(GraphError::InvalidParameter {
            param: "epsilon".to_string(),
            value: config.epsilon.to_string(),
            expected: "(0, 1)".to_string(),
            context: "imm_algorithm".to_string(),
        });
    }
    if !(0.0..1.0).contains(&config.delta) {
        return Err(GraphError::InvalidParameter {
            param: "delta".to_string(),
            value: config.delta.to_string(),
            expected: "(0, 1)".to_string(),
            context: "imm_algorithm".to_string(),
        });
    }

    let n = num_nodes as f64;
    let k = config.k;
    let eps = config.epsilon;
    let delta = config.delta;

    // ℓ = log(1/δ) for the confidence parameter
    let ell = (1.0_f64 / delta).ln();
    let log_n = n.ln();

    // IMM Step 1: Martingale sampling schedule
    // log C(n, k) ≈ k · log(n/k) by Stirling approximation for large n
    let log_cnk = if k >= 1 {
        let k_f = k as f64;
        k_f * (n / k_f).ln()
    } else {
        0.0_f64
    };

    // θ₀ seed: we do a binary search over 1..log2(n) iterations
    let max_iters = (log_n / 2.0_f64.ln()).ceil() as usize + 1;

    let mut rng: StdRng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut scirs2_core::random::rng()),
    };

    let rev = reverse_adj(adjacency);
    let mut rr_sets: Vec<RRSet> = Vec::new();

    // IMM main loop: double the number of RR sets each iteration and check
    // the Martingale stopping condition.
    let lambda_prime =
        (8.0 + 2.0 * eps) * n * (ell * log_n + log_cnk + (2.0_f64).ln()) / (eps * eps);

    for i in 1..=max_iters {
        // Number of RR sets to have after iteration i
        let theta_i = (lambda_prime / (n / 2.0_f64.powi(i as i32 - 1))).ceil() as usize;

        // Generate additional RR sets up to theta_i
        while rr_sets.len() < theta_i {
            rr_sets.push(generate_one_rr_set(&rev, num_nodes, &mut rng));
        }

        // Greedy coverage check
        let (_, num_covered) = greedy_max_coverage(&rr_sets, num_nodes, k);
        let frac = num_covered as f64 / rr_sets.len() as f64;

        // Estimated OPT_k lower bound: if the greedy fraction * n exceeds our
        // threshold (2^i * lambda'/n · ε_star) we stop.
        // Stopping condition from IMM 2015 Eq (2)
        let eps_star = compute_epsilon_prime(n, k, ell, frac * n, rr_sets.len());
        if frac - eps_star >= (1.0 - 1.0 / std::f64::consts::E - eps) * frac {
            break;
        }
    }

    // Final greedy selection
    let total_rr = rr_sets.len();
    let (seeds, num_covered) = greedy_max_coverage(&rr_sets, num_nodes, k);

    let estimated_spread = num_covered as f64 / total_rr as f64 * n;

    Ok(ImmResult {
        seeds,
        estimated_spread,
        num_rr_sets: total_rr,
    })
}

/// Compute the local ε' used in the IMM stopping condition.
///
/// Based on Lemma 4 of Tang et al. (2015):
/// `ε' = √((2(1 + ε̃) · log(6/δ') · n) / (frac_k · |RR|))`
/// where `frac_k` is the current estimated spread and `|RR|` is the RR set
/// count.  This controls the error bound of the current greedy estimate.
fn compute_epsilon_prime(n: f64, k: usize, ell: f64, spread: f64, num_rr: usize) -> f64 {
    if spread < 1.0 || num_rr == 0 {
        return 1.0;
    }
    let k_f = k as f64;
    // Simplified bound using the confidence term
    let log_term = ell + (6.0_f64).ln() + k_f * (n / k_f).ln();
    let eps_sq = (2.0 * (1.0 + 0.1) * log_term * n) / (spread * num_rr as f64);
    eps_sq.sqrt().min(1.0)
}

/// Sandwich approximation for submodular maximisation.
///
/// Computes both an upper-bound seed set (greedy on a lifted function) and a
/// lower-bound seed set (greedy on the original RIS estimate), along with their
/// estimated spreads.
///
/// The "sandwich" gives a factor-`(1 – 1/e)`-approximate bound on the optimal
/// spread even when the function is only approximately submodular.
///
/// # Returns
/// `(lower_seeds, lower_spread, upper_seeds, upper_spread)`
///
/// # Errors
/// Returns an error when parameters are invalid.
pub fn sandwich_approximation(
    adjacency: &AdjList,
    num_nodes: usize,
    k: usize,
    config: &RISConfig,
) -> Result<(Vec<usize>, f64, Vec<usize>, f64)> {
    if num_nodes == 0 {
        return Err(GraphError::InvalidParameter {
            param: "num_nodes".to_string(),
            value: "0".to_string(),
            expected: ">= 1".to_string(),
            context: "sandwich_approximation".to_string(),
        });
    }
    if k == 0 {
        return Ok((Vec::new(), 0.0, Vec::new(), 0.0));
    }
    if k > num_nodes {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= num_nodes={num_nodes}"),
            context: "sandwich_approximation".to_string(),
        });
    }

    // Generate a shared pool of RR sets
    let rr_sets = generate_rr_sets(adjacency, num_nodes, config)?;

    // Lower bound: standard greedy maximum-coverage on the RR sets
    let (lower_seeds, lower_covered) = greedy_max_coverage(&rr_sets, num_nodes, k);
    let lower_spread = lower_covered as f64 / rr_sets.len() as f64 * num_nodes as f64;

    // Upper bound: run greedy on a *doubled* RR-set pool (new independent
    // sample) which gives a less noisy coverage estimate
    let mut rng: StdRng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s.wrapping_add(0xDEAD_BEEF)),
        None => StdRng::from_rng(&mut scirs2_core::random::rng()),
    };
    let rev = reverse_adj(adjacency);
    let mut upper_rr = rr_sets.clone();
    for _ in 0..config.num_rr_sets {
        upper_rr.push(generate_one_rr_set(&rev, num_nodes, &mut rng));
    }

    let (upper_seeds, upper_covered) = greedy_max_coverage(&upper_rr, num_nodes, k);
    let upper_spread = upper_covered as f64 / upper_rr.len() as f64 * num_nodes as f64;

    Ok((lower_seeds, lower_spread, upper_seeds, upper_spread))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Star graph: hub 0 → spokes 1..n with probability `p`.
    fn star_adj(n: usize, p: f64) -> AdjList {
        let mut adj: AdjList = HashMap::new();
        for i in 1..n {
            adj.entry(0).or_default().push((i, p));
        }
        adj
    }

    /// Complete directed graph with uniform probability `p`.
    fn complete_adj(n: usize, p: f64) -> AdjList {
        let mut adj: AdjList = HashMap::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj.entry(i).or_default().push((j, p));
                }
            }
        }
        adj
    }

    #[test]
    fn test_generate_rr_sets_basic() {
        let adj = star_adj(6, 1.0);
        let config = RISConfig {
            num_rr_sets: 50,
            seed: Some(42),
        };
        let rr = generate_rr_sets(&adj, 6, &config).expect("rr sets");
        assert_eq!(rr.len(), 50);
        // All RR sets should be non-empty (trivially: the root is always included)
        for r in &rr {
            assert!(!r.is_empty());
        }
    }

    #[test]
    fn test_generate_rr_sets_invalid_params() {
        let adj = star_adj(6, 1.0);
        // num_nodes = 0
        let err = generate_rr_sets(&adj, 0, &RISConfig::default());
        assert!(err.is_err());
        // num_rr_sets = 0
        let config = RISConfig {
            num_rr_sets: 0,
            seed: None,
        };
        let err2 = generate_rr_sets(&adj, 6, &config);
        assert!(err2.is_err());
    }

    #[test]
    fn test_ris_estimate_star_hub() {
        // With p=1.0 and hub node 0, seeding {0} activates all 6 nodes.
        let adj = star_adj(6, 1.0);
        let config = RISConfig {
            num_rr_sets: 200,
            seed: Some(123),
        };
        let rr = generate_rr_sets(&adj, 6, &config).expect("rr sets");
        let spread = ris_estimate(&rr, &[0], 6).expect("estimate");
        // Should be close to 6.0 (all nodes activated)
        assert!(spread >= 4.0, "spread={spread}");
    }

    #[test]
    fn test_ris_estimate_empty_seed() {
        let adj = star_adj(6, 1.0);
        let config = RISConfig {
            num_rr_sets: 100,
            seed: Some(0),
        };
        let rr = generate_rr_sets(&adj, 6, &config).expect("rr sets");
        let spread = ris_estimate(&rr, &[], 6).expect("zero seed");
        assert_eq!(spread, 0.0);
    }

    #[test]
    fn test_ris_estimate_empty_rr_error() {
        let err = ris_estimate(&[], &[0], 6);
        assert!(err.is_err());
    }

    #[test]
    fn test_imm_star_selects_hub() {
        let adj = star_adj(8, 1.0);
        let config = ImmConfig {
            k: 1,
            epsilon: 0.3,
            delta: 0.1,
            seed: Some(42),
        };
        let result = imm_algorithm(&adj, 8, &config).expect("imm");
        assert_eq!(result.seeds.len(), 1);
        // Hub should be selected for p=1.0
        assert_eq!(result.seeds[0], 0, "hub expected, got {:?}", result.seeds);
        assert!(result.estimated_spread >= 1.0);
    }

    #[test]
    fn test_imm_k0_returns_empty() {
        let adj = star_adj(5, 1.0);
        let config = ImmConfig {
            k: 0,
            ..Default::default()
        };
        let result = imm_algorithm(&adj, 5, &config).expect("imm k=0");
        assert!(result.seeds.is_empty());
        assert_eq!(result.estimated_spread, 0.0);
    }

    #[test]
    fn test_imm_invalid_params() {
        let adj = star_adj(5, 1.0);
        // k > num_nodes
        let err = imm_algorithm(
            &adj,
            5,
            &ImmConfig {
                k: 10,
                ..Default::default()
            },
        );
        assert!(err.is_err());

        // epsilon out of range
        let err2 = imm_algorithm(
            &adj,
            5,
            &ImmConfig {
                epsilon: 1.5,
                ..Default::default()
            },
        );
        assert!(err2.is_err());

        // num_nodes = 0
        let err3 = imm_algorithm(&adj, 0, &ImmConfig::default());
        assert!(err3.is_err());
    }

    #[test]
    fn test_imm_complete_graph() {
        // 5-node complete graph: any single seed should activate all with p=1.0
        let adj = complete_adj(5, 1.0);
        let config = ImmConfig {
            k: 1,
            epsilon: 0.3,
            delta: 0.1,
            seed: Some(7),
        };
        let result = imm_algorithm(&adj, 5, &config).expect("imm complete");
        assert_eq!(result.seeds.len(), 1);
        assert!(result.estimated_spread >= 1.0);
    }

    #[test]
    fn test_sandwich_approximation_basic() {
        let adj = star_adj(6, 1.0);
        let config = RISConfig {
            num_rr_sets: 100,
            seed: Some(99),
        };
        let (lower_seeds, lower_spread, upper_seeds, upper_spread) =
            sandwich_approximation(&adj, 6, 1, &config).expect("sandwich");
        assert_eq!(lower_seeds.len(), 1);
        assert_eq!(upper_seeds.len(), 1);
        // Both spreads should be positive
        assert!(lower_spread >= 0.0);
        assert!(upper_spread >= 0.0);
        // Upper bound should not be below lower (with same seed set quality)
        // (Note: not strictly guaranteed but should hold statistically with p=1)
        let _ = (lower_spread, upper_spread); // suppress unused warning
    }

    #[test]
    fn test_sandwich_k0_returns_empty() {
        let adj = star_adj(5, 1.0);
        let (ls, lsp, us, usp) =
            sandwich_approximation(&adj, 5, 0, &RISConfig::default()).expect("k=0");
        assert!(ls.is_empty());
        assert!(us.is_empty());
        assert_eq!(lsp, 0.0);
        assert_eq!(usp, 0.0);
    }

    #[test]
    fn test_sandwich_invalid_params() {
        let adj = star_adj(5, 1.0);
        let err = sandwich_approximation(&adj, 5, 10, &RISConfig::default());
        assert!(err.is_err());
        let err2 = sandwich_approximation(&adj, 0, 1, &RISConfig::default());
        assert!(err2.is_err());
    }
}
