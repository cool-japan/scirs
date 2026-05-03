//! Network reliability algorithms: Monte Carlo estimation, exact BDD computation,
//! inclusion-exclusion polynomial, and component failure enumeration.
//!
//! ## Algorithms
//!
//! - **Monte Carlo two-terminal reliability**: sample edge subsets, check connectivity.
//! - **Monte Carlo all-terminal reliability**: check full graph connectivity.
//! - **ReliabilityPolynomial**: exact polynomial coefficients via inclusion-exclusion
//!   on spanning trees / path enumeration (feasible for |E| ≤ 20).
//! - **BDD** (Binary Decision Diagram): exact reliability for small networks via
//!   ordered BDDs over edge variables.
//! - **ComponentFailureTree**: ball-tree style structure for enumerating
//!   minimal cuts / paths.

use std::collections::{HashMap, VecDeque};

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the edge list and per-edge survival probabilities from an adjacency matrix.
///
/// For undirected graphs the matrix is treated as symmetric; each undirected
/// edge `{i, j}` (i < j) is included once.
fn extract_edges(adj: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let n = adj.nrows();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let w = adj[[i, j]];
            if w > 0.0 {
                // Clamp weight to [0,1] as survival probability
                edges.push((i, j, w.clamp(0.0, 1.0)));
            }
        }
    }
    edges
}

/// BFS reachability check: can node `s` reach node `t` through active edges?
///
/// `active[e]` is true iff edge `e` is functioning.
fn can_reach(n: usize, edges: &[(usize, usize, f64)], active: &[bool], s: usize, t: usize) -> bool {
    if s == t {
        return true;
    }
    // Build adjacency list for active edges
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, &(u, v, _)) in edges.iter().enumerate() {
        if active[idx] {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut visited = vec![false; n];
    visited[s] = true;
    let mut queue = VecDeque::new();
    queue.push_back(s);
    while let Some(node) = queue.pop_front() {
        for &nb in &adj[node] {
            if nb == t {
                return true;
            }
            if !visited[nb] {
                visited[nb] = true;
                queue.push_back(nb);
            }
        }
    }
    false
}

/// BFS connectivity check: are all nodes reachable from node 0?
fn is_fully_connected(n: usize, edges: &[(usize, usize, f64)], active: &[bool]) -> bool {
    if n <= 1 {
        return true;
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, &(u, v, _)) in edges.iter().enumerate() {
        if active[idx] {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut visited = vec![false; n];
    visited[0] = true;
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    let mut count = 1usize;
    while let Some(node) = queue.pop_front() {
        for &nb in &adj[node] {
            if !visited[nb] {
                visited[nb] = true;
                count += 1;
                queue.push_back(nb);
            }
        }
    }
    count == n
}

// ─────────────────────────────────────────────────────────────────────────────
// NetworkReliability — two-terminal Monte Carlo
// ─────────────────────────────────────────────────────────────────────────────

/// Two-terminal network reliability estimator.
///
/// Estimates P(source `s` can reach terminal `t`) under independent edge
/// failures via Monte Carlo sampling.
///
/// Each simulation trial:
/// 1. For each edge `(u, v, p)`, keep the edge with probability `p`.
/// 2. Run BFS to check whether `s` can reach `t`.
///
/// The estimate converges at rate `O(1/√N)` where `N` is the number of trials.
#[derive(Debug, Clone)]
pub struct NetworkReliability {
    /// Source node index.
    pub source: usize,
    /// Terminal (target) node index.
    pub terminal: usize,
}

impl NetworkReliability {
    /// Create a two-terminal reliability estimator.
    pub fn new(source: usize, terminal: usize) -> Self {
        Self { source, terminal }
    }

    /// Estimate two-terminal reliability via Monte Carlo simulation.
    ///
    /// # Arguments
    /// * `adj` — weighted adjacency matrix; weights are edge survival probabilities ∈ (0,1].
    /// * `num_trials` — number of Monte Carlo samples.
    /// * `seed` — optional RNG seed for reproducibility.
    ///
    /// # Returns
    /// Estimated probability ∈ [0, 1].
    pub fn monte_carlo(
        &self,
        adj: &Array2<f64>,
        num_trials: usize,
        seed: Option<u64>,
    ) -> Result<f64> {
        let n = adj.nrows();
        if self.source >= n {
            return Err(GraphError::InvalidParameter {
                param: "source".into(),
                value: self.source.to_string(),
                expected: format!("< {n}"),
                context: "NetworkReliability::monte_carlo".into(),
            });
        }
        if self.terminal >= n {
            return Err(GraphError::InvalidParameter {
                param: "terminal".into(),
                value: self.terminal.to_string(),
                expected: format!("< {n}"),
                context: "NetworkReliability::monte_carlo".into(),
            });
        }
        if num_trials == 0 {
            return Ok(0.0);
        }

        let edges = extract_edges(adj);
        let m = edges.len();
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut scirs2_core::random::rng()),
        };

        let mut successes = 0u64;
        let mut active = vec![false; m];

        for _ in 0..num_trials {
            for (idx, &(_, _, p)) in edges.iter().enumerate() {
                active[idx] = rng.random::<f64>() < p;
            }
            if can_reach(n, &edges, &active, self.source, self.terminal) {
                successes += 1;
            }
        }

        Ok(successes as f64 / num_trials as f64)
    }

    /// Compute a confidence interval for the Monte Carlo estimate.
    ///
    /// Returns `(estimate, half_width)` where the 95% CI is
    /// `[estimate − half_width, estimate + half_width]`.
    pub fn monte_carlo_with_ci(
        &self,
        adj: &Array2<f64>,
        num_trials: usize,
        seed: Option<u64>,
    ) -> Result<(f64, f64)> {
        let p_hat = self.monte_carlo(adj, num_trials, seed)?;
        // Wilson-like bound: half-width ≈ 1.96 * sqrt(p(1-p)/n)
        let half_width = if num_trials > 0 {
            1.96 * (p_hat * (1.0 - p_hat) / num_trials as f64).sqrt()
        } else {
            1.0
        };
        Ok((p_hat, half_width))
    }

    /// Compute the exact two-terminal reliability for small graphs by
    /// exhaustive enumeration of all 2^|E| edge subsets.
    ///
    /// Only feasible for `|E| ≤ 25`.
    pub fn exact(&self, adj: &Array2<f64>) -> Result<f64> {
        let n = adj.nrows();
        if self.source >= n || self.terminal >= n {
            return Err(GraphError::InvalidParameter {
                param: "source/terminal".into(),
                value: format!("{}/{}", self.source, self.terminal),
                expected: format!("< {n}"),
                context: "NetworkReliability::exact".into(),
            });
        }
        let edges = extract_edges(adj);
        let m = edges.len();
        if m > 25 {
            return Err(GraphError::InvalidParameter {
                param: "num_edges".into(),
                value: m.to_string(),
                expected: "<= 25 for exact computation".into(),
                context: "NetworkReliability::exact".into(),
            });
        }

        let mut total = 0.0_f64;
        for mask in 0u32..(1u32 << m) {
            let active: Vec<bool> = (0..m).map(|i| (mask >> i) & 1 == 1).collect();
            // Probability of this configuration
            let prob: f64 = edges
                .iter()
                .enumerate()
                .map(|(i, &(_, _, p))| if active[i] { p } else { 1.0 - p })
                .product();
            if can_reach(n, &edges, &active, self.source, self.terminal) {
                total += prob;
            }
        }
        Ok(total)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AllTerminalReliability — all-terminal Monte Carlo
// ─────────────────────────────────────────────────────────────────────────────

/// All-terminal network reliability estimator.
///
/// Estimates P(all nodes mutually reachable) under independent edge failures.
#[derive(Debug, Clone, Default)]
pub struct AllTerminalReliability;

impl AllTerminalReliability {
    /// Create an all-terminal reliability estimator.
    pub fn new() -> Self {
        Self
    }

    /// Estimate all-terminal reliability via Monte Carlo simulation.
    pub fn monte_carlo(
        &self,
        adj: &Array2<f64>,
        num_trials: usize,
        seed: Option<u64>,
    ) -> Result<f64> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency".into()));
        }
        if num_trials == 0 {
            return Ok(0.0);
        }

        let edges = extract_edges(adj);
        let m = edges.len();
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut scirs2_core::random::rng()),
        };

        let mut successes = 0u64;
        let mut active = vec![false; m];

        for _ in 0..num_trials {
            for (idx, &(_, _, p)) in edges.iter().enumerate() {
                active[idx] = rng.random::<f64>() < p;
            }
            if is_fully_connected(n, &edges, &active) {
                successes += 1;
            }
        }

        Ok(successes as f64 / num_trials as f64)
    }

    /// Exact all-terminal reliability (exhaustive; |E| ≤ 20).
    pub fn exact(&self, adj: &Array2<f64>) -> Result<f64> {
        let n = adj.nrows();
        let edges = extract_edges(adj);
        let m = edges.len();
        if m > 20 {
            return Err(GraphError::InvalidParameter {
                param: "num_edges".into(),
                value: m.to_string(),
                expected: "<= 20 for exact computation".into(),
                context: "AllTerminalReliability::exact".into(),
            });
        }
        let mut total = 0.0_f64;
        for mask in 0u32..(1u32 << m) {
            let active: Vec<bool> = (0..m).map(|i| (mask >> i) & 1 == 1).collect();
            let prob: f64 = edges
                .iter()
                .enumerate()
                .map(|(i, &(_, _, p))| if active[i] { p } else { 1.0 - p })
                .product();
            if is_fully_connected(n, &edges, &active) {
                total += prob;
            }
        }
        Ok(total)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReliabilityPolynomial — exact polynomial coefficients
// ─────────────────────────────────────────────────────────────────────────────

/// Exact reliability polynomial for small networks (|E| ≤ 20).
///
/// For a network where all edges have the same survival probability `p`, the
/// all-terminal reliability is a polynomial in `p`:
///
///   R(p) = Σ_{k=0}^{m} c_k p^k (1−p)^{m−k}
///
/// where `c_k` is the number of edge subsets of size `k` that make the graph
/// connected.  This struct computes the coefficient vector `[c_0, c_1, …, c_m]`
/// by exhaustive enumeration and stores it for fast evaluation at any `p`.
#[derive(Debug, Clone)]
pub struct ReliabilityPolynomial {
    /// Coefficients: `coeffs[k]` = number of connected edge subsets of size `k`.
    pub coeffs: Vec<u64>,
    /// Total number of edges.
    pub num_edges: usize,
    /// Number of nodes.
    pub num_nodes: usize,
}

impl ReliabilityPolynomial {
    /// Compute the reliability polynomial for a graph where all edges have
    /// equal survival probability.
    ///
    /// Only feasible for `|E| ≤ 20`.
    pub fn compute(adj: &Array2<f64>) -> Result<Self> {
        let n = adj.nrows();
        // Treat the adjacency matrix as unweighted for the polynomial
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| {
                (i + 1..n).filter_map(move |j| {
                    if adj[[i, j]] > 0.0 {
                        Some((i, j))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let m = edges.len();
        if m > 20 {
            return Err(GraphError::InvalidParameter {
                param: "num_edges".into(),
                value: m.to_string(),
                expected: "<= 20 for polynomial computation".into(),
                context: "ReliabilityPolynomial::compute".into(),
            });
        }

        let mut coeffs = vec![0u64; m + 1];
        // Enumerate all 2^m subsets
        for mask in 0u32..(1u32 << m) {
            let k = mask.count_ones() as usize;
            let active: Vec<bool> = (0..m).map(|i| (mask >> i) & 1 == 1).collect();
            // Build edge list with p=1 for active check
            let active_edges: Vec<(usize, usize, f64)> =
                edges.iter().map(|&(u, v)| (u, v, 1.0)).collect();
            let active_flags: Vec<bool> = (0..m).map(|i| active[i]).collect();
            if is_fully_connected(n, &active_edges, &active_flags) {
                coeffs[k] += 1;
            }
        }

        Ok(Self {
            coeffs,
            num_edges: m,
            num_nodes: n,
        })
    }

    /// Evaluate the reliability polynomial at survival probability `p`.
    ///
    /// R(p) = Σ_k c_k * p^k * (1−p)^{m−k}
    pub fn evaluate(&self, p: f64) -> f64 {
        let m = self.num_edges;
        let q = 1.0 - p;
        self.coeffs
            .iter()
            .enumerate()
            .map(|(k, &c)| {
                if c == 0 {
                    0.0
                } else {
                    c as f64 * p.powi(k as i32) * q.powi((m - k) as i32)
                }
            })
            .sum()
    }

    /// Return the minimum cut size (the lowest `k` with `coeffs[k] > 0`).
    pub fn min_connected_edges(&self) -> usize {
        self.coeffs
            .iter()
            .position(|&c| c > 0)
            .unwrap_or(self.num_edges)
    }

    /// Return the total number of spanning subgraphs (sum of all coefficients).
    pub fn total_connected_subgraphs(&self) -> u64 {
        self.coeffs.iter().sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BDD — Binary Decision Diagram for exact reliability
// ─────────────────────────────────────────────────────────────────────────────

/// BDD node type.
#[derive(Debug, Clone)]
enum BddNode {
    /// Terminal node: 0 = failure, 1 = success.
    Terminal(bool),
    /// Internal node: variable index, low child, high child.
    Internal { var: usize, low: usize, high: usize },
}

/// Binary Decision Diagram (BDD) for exact network reliability computation.
///
/// Represents the reliability function as an ordered BDD over edge boolean
/// variables.  Each variable `x_e = 1` means edge `e` is functioning.
///
/// The BDD is built by Shannon expansion on each edge variable in order:
/// - Set `x_e = 1` (edge alive): recurse on remaining problem.
/// - Set `x_e = 0` (edge failed): recurse.
/// - Merge isomorphic subgraphs (unique table) for compactness.
///
/// Reliability is computed by a single bottom-up traversal weighting
/// each path by `p_e^{x_e} (1−p_e)^{1−x_e}`.
#[derive(Debug)]
pub struct BDD {
    nodes: Vec<BddNode>,
    /// Map (var, low_idx, high_idx) → node index (unique table)
    unique: HashMap<(usize, usize, usize), usize>,
    /// Total number of nodes (edges) in the network.
    num_edges: usize,
    /// Number of network nodes.
    num_nodes: usize,
    /// Edge list.
    edges: Vec<(usize, usize)>,
    /// Root node index.
    root: usize,
}

impl BDD {
    /// Build a BDD for all-terminal reliability.
    ///
    /// Only feasible for `|E| ≤ 20`.
    pub fn build_all_terminal(adj: &Array2<f64>) -> Result<Self> {
        let n = adj.nrows();
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| {
                (i + 1..n).filter_map(move |j| {
                    if adj[[i, j]] > 0.0 {
                        Some((i, j))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let m = edges.len();
        if m > 20 {
            return Err(GraphError::InvalidParameter {
                param: "num_edges".into(),
                value: m.to_string(),
                expected: "<= 20 for BDD".into(),
                context: "BDD::build_all_terminal".into(),
            });
        }

        let mut bdd = BDD {
            nodes: Vec::new(),
            unique: HashMap::new(),
            num_edges: m,
            num_nodes: n,
            edges: edges.clone(),
            root: 0,
        };

        // Terminal nodes: index 0 = False, index 1 = True
        bdd.nodes.push(BddNode::Terminal(false));
        bdd.nodes.push(BddNode::Terminal(true));

        let active_mask = (1u32 << m) - 1; // all edges initially unknown
        let root = bdd.build_node(0, active_mask, n, &edges);
        bdd.root = root;
        Ok(bdd)
    }

    /// Recursively build a BDD node for `var_idx`-th edge variable.
    ///
    /// `active_mask` represents which edges are currently "forced on" (1-bit).
    /// We actually use a different approach: Shannon expansion.
    fn build_node(
        &mut self,
        var: usize,
        forced_on: u32,
        n_nodes: usize,
        edges: &[(usize, usize)],
    ) -> usize {
        let m = edges.len();
        if var == m {
            // All variables assigned; check if forced-on edges form connected graph
            let active: Vec<bool> = (0..m).map(|i| (forced_on >> i) & 1 == 1).collect();
            let edge_data: Vec<(usize, usize, f64)> =
                edges.iter().map(|&(u, v)| (u, v, 1.0)).collect();
            let connected = is_fully_connected(n_nodes, &edge_data, &active);
            return if connected { 1 } else { 0 };
        }

        // Check if already computed
        // For BDD with Shannon expansion we memoize on (var, forced_on)
        // Using forced_on as a compact state representation
        let key = (var, forced_on as usize, 0);
        if let Some(&idx) = self.unique.get(&key) {
            return idx;
        }

        // Shannon expansion on edge `var`
        // Low child: edge `var` = 0 (failed)
        let low_forced = forced_on & !(1u32 << var);
        let low = self.build_node(var + 1, low_forced, n_nodes, edges);

        // High child: edge `var` = 1 (alive)
        let high_forced = forced_on | (1u32 << var);
        let high = self.build_node(var + 1, high_forced, n_nodes, edges);

        // If both children are the same, no need for new node (reduction rule)
        if low == high {
            self.unique.insert(key, low);
            return low;
        }

        let idx = self.nodes.len();
        self.nodes.push(BddNode::Internal { var, low, high });
        self.unique.insert(key, idx);
        idx
    }

    /// Compute the all-terminal reliability R = E[connected(G_p)] using the BDD.
    ///
    /// # Arguments
    /// * `probs` — survival probability for each edge, in the same order as the
    ///   adjacency matrix edge enumeration (upper-triangle, row-major).
    pub fn reliability(&self, probs: &[f64]) -> Result<f64> {
        if probs.len() != self.num_edges {
            return Err(GraphError::InvalidParameter {
                param: "probs.len()".into(),
                value: probs.len().to_string(),
                expected: self.num_edges.to_string(),
                context: "BDD::reliability".into(),
            });
        }
        Ok(self.eval_node(self.root, probs))
    }

    fn eval_node(&self, node_idx: usize, probs: &[f64]) -> f64 {
        match &self.nodes[node_idx] {
            BddNode::Terminal(t) => {
                if *t {
                    1.0
                } else {
                    0.0
                }
            }
            BddNode::Internal { var, low, high } => {
                let p = probs[*var];
                let q = 1.0 - p;
                q * self.eval_node(*low, probs) + p * self.eval_node(*high, probs)
            }
        }
    }

    /// Return the number of BDD nodes (size of the diagram).
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges in the underlying network.
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Return the number of nodes in the underlying network.
    pub fn num_network_nodes(&self) -> usize {
        self.num_nodes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComponentFailureTree — ball-tree style enumeration of failure modes
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the component failure tree.
#[derive(Debug, Clone)]
pub struct FailureNode {
    /// The set of failed edge indices at this tree node.
    pub failed_edges: Vec<usize>,
    /// Whether this failure set constitutes a disconnecting cut.
    pub is_cut: bool,
    /// Probability of this exact failure set.
    pub probability: f64,
    /// Children nodes (further edge failures).
    pub children: Vec<FailureNode>,
}

/// Ball-tree style structure for enumerating component failure modes.
///
/// Builds a tree where each level introduces one additional edge failure.
/// The tree branches at each node for each remaining edge that could fail.
/// This is useful for computing minimal cuts and identifying the most
/// likely failure scenarios.
#[derive(Debug)]
pub struct ComponentFailureTree {
    /// Root of the failure tree.
    pub root: FailureNode,
    /// All minimal cuts found during tree construction.
    pub minimal_cuts: Vec<Vec<usize>>,
    /// Number of edges.
    pub num_edges: usize,
    /// Number of nodes.
    pub num_nodes: usize,
}

impl ComponentFailureTree {
    /// Build the component failure tree for a network up to `max_depth` edge failures.
    ///
    /// # Arguments
    /// * `adj` — weighted adjacency matrix.
    /// * `max_depth` — maximum number of simultaneous edge failures to consider.
    ///
    /// `max_depth` should be kept small (≤ 5) for efficiency.
    pub fn build(adj: &Array2<f64>, max_depth: usize) -> Result<Self> {
        let n = adj.nrows();
        let edges: Vec<(usize, usize, f64)> = extract_edges(adj);
        let m = edges.len();

        let mut minimal_cuts = Vec::new();
        let root_active = vec![true; m];
        let is_cut = !is_fully_connected(n, &edges, &root_active);

        let root = FailureNode {
            failed_edges: Vec::new(),
            is_cut,
            probability: 1.0,
            children: Vec::new(),
        };

        let mut tree = ComponentFailureTree {
            root,
            minimal_cuts: Vec::new(),
            num_edges: m,
            num_nodes: n,
        };

        // Build tree via DFS
        let failed: Vec<usize> = Vec::new();
        let edge_probs: Vec<f64> = edges.iter().map(|&(_, _, p)| p).collect();
        failure_tree_expand_node(
            &mut tree.root.children,
            &failed,
            0,
            max_depth,
            n,
            &edges,
            &edge_probs,
            m,
            &mut minimal_cuts,
        );
        tree.minimal_cuts = minimal_cuts;

        Ok(tree)
    }

    /// Return all minimal cuts found during tree construction.
    pub fn minimal_cuts(&self) -> &[Vec<usize>] {
        &self.minimal_cuts
    }

    /// Compute total probability of disconnection up to `max_depth` failures.
    ///
    /// This sums probabilities of all failure sets that are cuts, but avoids
    /// double-counting by only summing *minimal* cuts' exact set probabilities.
    pub fn unreliability_upper_bound(&self) -> f64 {
        Self::sum_cut_probs(&self.root)
    }

    fn sum_cut_probs(node: &FailureNode) -> f64 {
        let self_contribution = if node.is_cut { node.probability } else { 0.0 };
        let child_sum: f64 = node.children.iter().map(Self::sum_cut_probs).sum();
        self_contribution + child_sum
    }
}

#[allow(clippy::too_many_arguments)]
fn failure_tree_expand_node(
    children: &mut Vec<FailureNode>,
    parent_failed: &[usize],
    start_edge: usize,
    remaining_depth: usize,
    n: usize,
    edges: &[(usize, usize, f64)],
    edge_probs: &[f64],
    m: usize,
    minimal_cuts: &mut Vec<Vec<usize>>,
) {
    if remaining_depth == 0 {
        return;
    }
    for e in start_edge..m {
        let mut failed = parent_failed.to_vec();
        failed.push(e);

        // Probability of this failure set (prob that exactly these edges fail)
        let prob: f64 = (0..m)
            .map(|i| {
                if failed.contains(&i) {
                    1.0 - edge_probs[i]
                } else {
                    edge_probs[i]
                }
            })
            .product();

        let active: Vec<bool> = (0..m).map(|i| !failed.contains(&i)).collect();
        let is_cut = !is_fully_connected(n, edges, &active);

        // Check if this is a minimal cut: is_cut AND parent (without e) was NOT a cut
        if is_cut {
            // Check if parent failure set already disconnects
            let parent_active: Vec<bool> = (0..m).map(|i| !parent_failed.contains(&i)).collect();
            let parent_is_cut = !is_fully_connected(n, edges, &parent_active);
            if !parent_is_cut {
                minimal_cuts.push(failed.clone());
            }
        }

        let mut node = FailureNode {
            failed_edges: failed.clone(),
            is_cut,
            probability: prob,
            children: Vec::new(),
        };

        if !is_cut && remaining_depth > 1 {
            failure_tree_expand_node(
                &mut node.children,
                &failed,
                e + 1,
                remaining_depth - 1,
                n,
                edges,
                edge_probs,
                m,
                minimal_cuts,
            );
        }

        children.push(node);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_adj(p: f64) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = p;
        adj[[1, 0]] = p;
        adj[[1, 2]] = p;
        adj[[2, 1]] = p;
        adj[[0, 2]] = p;
        adj[[2, 0]] = p;
        adj
    }

    fn path_adj_p(n: usize, p: f64) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = p;
            adj[[i + 1, i]] = p;
        }
        adj
    }

    #[test]
    fn test_two_terminal_exact_vs_mc() {
        // Path graph 0-1-2: two-terminal reliability (0 to 2) = p^2
        let p = 0.8;
        let adj = path_adj_p(3, p);
        let rel = NetworkReliability::new(0, 2);
        let exact = rel.exact(&adj).unwrap();
        // For path 0-1-2, need both edges: p^2
        assert!((exact - p * p).abs() < 1e-9, "Exact: {exact} vs {}", p * p);
        let mc = rel.monte_carlo(&adj, 50000, Some(99)).unwrap();
        assert!((mc - exact).abs() < 0.02, "MC: {mc} vs exact: {exact}");
    }

    #[test]
    fn test_all_terminal_triangle_exact() {
        let p = 0.9;
        let adj = triangle_adj(p);
        let rel = AllTerminalReliability::new();
        let exact = rel.exact(&adj).unwrap();
        // All-terminal reliability of triangle: at least 2 of 3 edges must be present
        // = C(3,2)*p^2*(1-p) + C(3,3)*p^3 = 3p²(1-p) + p³
        let expected = 3.0 * p * p * (1.0 - p) + p * p * p;
        assert!(
            (exact - expected).abs() < 1e-9,
            "Exact: {exact} vs {expected}"
        );
        let mc = rel.monte_carlo(&adj, 50000, Some(7)).unwrap();
        assert!((mc - exact).abs() < 0.02);
    }

    #[test]
    fn test_reliability_polynomial() {
        let adj = triangle_adj(1.0); // all edges present (weights = 1, so p=1 clamp)
        let poly = ReliabilityPolynomial::compute(&adj).unwrap();
        assert_eq!(poly.num_edges, 3);
        // At p=1 all subsets with ≥2 edges should connect
        let r1 = poly.evaluate(1.0);
        assert!((r1 - 1.0).abs() < 1e-9, "R(1)={r1}");
        let r0 = poly.evaluate(0.0);
        assert!((r0 - 0.0).abs() < 1e-9, "R(0)={r0}");
    }

    #[test]
    fn test_bdd_vs_exact() {
        let p = 0.75;
        let adj = triangle_adj(p);
        let bdd = BDD::build_all_terminal(&adj).unwrap();
        let probs = vec![p; 3];
        let bdd_result = bdd.reliability(&probs).unwrap();
        let exact = AllTerminalReliability::new().exact(&adj).unwrap();
        assert!(
            (bdd_result - exact).abs() < 1e-9,
            "BDD: {bdd_result} vs exact: {exact}"
        );
    }

    #[test]
    fn test_component_failure_tree() {
        let adj = triangle_adj(0.9);
        let tree = ComponentFailureTree::build(&adj, 2).unwrap();
        // A triangle has minimal cuts of size 2 (any 2 edges incident to a node)
        assert!(!tree.minimal_cuts().is_empty());
    }

    #[test]
    fn test_reliability_ci() {
        let adj = path_adj_p(3, 0.9);
        let rel = NetworkReliability::new(0, 2);
        let (p_hat, hw) = rel.monte_carlo_with_ci(&adj, 10000, Some(1)).unwrap();
        assert!((0.0..=1.0).contains(&p_hat));
        assert!((0.0..=0.1).contains(&hw));
    }
}
