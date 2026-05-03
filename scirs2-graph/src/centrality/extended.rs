//! Extended centrality measures for graph analysis.
//!
//! This module provides advanced centrality algorithms beyond the standard
//! degree, betweenness, closeness, and eigenvector centralities found in
//! `measures.rs`.
//!
//! ## Algorithms
//!
//! | Struct / Function | Description |
//! |-------------------|-------------|
//! | [`VoteRankCentrality`] | VoteRank — iterative influential spreader identification |
//! | [`HITSCentrality`] | HITS (Hyperlink-Induced Topic Search) — hubs and authorities |
//! | [`TrustCentrality`] | Trust/distrust centrality for signed networks |
//! | [`CoreDecomposition`] | k-core decomposition and k-shell index |
//! | [`EffectiveResistance`] | Effective resistance distance matrix (via Laplacian pseudo-inverse) |
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::centrality::extended::{HITSCentrality, CoreDecomposition, EffectiveResistance};
//!
//! let mut adj = Array2::<f64>::zeros((4, 4));
//! adj[[0,1]] = 1.0; adj[[1,0]] = 1.0;
//! adj[[1,2]] = 1.0; adj[[2,1]] = 1.0;
//! adj[[2,3]] = 1.0; adj[[3,2]] = 1.0;
//! adj[[0,3]] = 1.0; adj[[3,0]] = 1.0;
//!
//! let hits = HITSCentrality::new(100, 1e-8);
//! let (hubs, auths) = hits.compute(&adj).unwrap();
//!
//! let core = CoreDecomposition::compute(&adj);
//! println!("Core numbers: {:?}", core.core_numbers);
//!
//! let er = EffectiveResistance::compute(&adj).unwrap();
//! println!("Resistance(0,2) = {:.4}", er.resistance(0, 2).unwrap());
//! ```

use std::collections::VecDeque;

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// VoteRankCentrality
// ─────────────────────────────────────────────────────────────────────────────

/// VoteRank algorithm for identifying influential spreaders.
///
/// VoteRank iteratively selects the most influential spreader by a voting
/// process: each non-selected node votes for its neighbors, and the node
/// with the highest vote score is chosen as the next spreader.  After
/// selection, the voting ability of the spreader's neighbors is reduced by
/// `1 / degree(selected)`.
///
/// # Reference
/// Zhang et al. (2016). "Identifying a set of influential spreaders in
/// complex networks." *Scientific Reports*, 6, 27823.
///
/// The algorithm produces an ordered list of `k` influential nodes,
/// where the first node is the single most influential spreader.
#[derive(Debug, Clone)]
pub struct VoteRankCentrality {
    /// Number of spreaders to identify.
    pub k: usize,
}

impl VoteRankCentrality {
    /// Create a VoteRank instance to find `k` influential spreaders.
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Run VoteRank on an undirected graph (adjacency matrix).
    ///
    /// Returns the ordered list of selected node indices (most influential first).
    pub fn compute(&self, adj: &Array2<f64>) -> Result<Vec<usize>> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
        }
        let k = self.k.min(n);

        // Compute degree of each node
        let degree: Vec<f64> = (0..n)
            .map(|i| adj.row(i).iter().filter(|&&x| x != 0.0).count() as f64)
            .collect();

        // Voting ability of each node (starts at 1.0, reduced after neighbours are selected)
        let mut vote_ability = vec![1.0_f64; n];

        // Track selected nodes
        let mut selected = Vec::with_capacity(k);
        let mut is_selected = vec![false; n];

        for _ in 0..k {
            // Compute vote scores: score[i] = sum of vote_ability[j] for all j ~ i
            let mut scores = vec![0.0_f64; n];
            for i in 0..n {
                if is_selected[i] {
                    continue;
                }
                for j in 0..n {
                    if adj[[i, j]] != 0.0 && !is_selected[j] {
                        scores[i] += vote_ability[j];
                    }
                }
            }

            // Select node with highest score (ties broken by lowest index)
            let best = (0..n).filter(|&i| !is_selected[i]).max_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            match best {
                Some(node) if scores[node] > 0.0 => {
                    is_selected[node] = true;
                    selected.push(node);
                    // Reduce voting ability of neighbours
                    let deg_inv = if degree[node] > 0.0 {
                        1.0 / degree[node]
                    } else {
                        0.0
                    };
                    for j in 0..n {
                        if adj[[node, j]] != 0.0 {
                            vote_ability[j] = (vote_ability[j] - deg_inv).max(0.0);
                        }
                    }
                }
                _ => break, // No more non-trivial nodes to select
            }
        }

        Ok(selected)
    }

    /// Compute the VoteRank score (vote count at time of selection) for each
    /// selected node, returned as `Vec<(node_index, score)>`.
    pub fn compute_with_scores(&self, adj: &Array2<f64>) -> Result<Vec<(usize, f64)>> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
        }
        let k = self.k.min(n);

        let degree: Vec<f64> = (0..n)
            .map(|i| adj.row(i).iter().filter(|&&x| x != 0.0).count() as f64)
            .collect();
        let mut vote_ability = vec![1.0_f64; n];
        let mut selected = Vec::with_capacity(k);
        let mut is_selected = vec![false; n];

        for _ in 0..k {
            let mut scores = vec![0.0_f64; n];
            for i in 0..n {
                if is_selected[i] {
                    continue;
                }
                for j in 0..n {
                    if adj[[i, j]] != 0.0 && !is_selected[j] {
                        scores[i] += vote_ability[j];
                    }
                }
            }
            let best = (0..n).filter(|&i| !is_selected[i]).max_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            match best {
                Some(node) if scores[node] > 0.0 => {
                    is_selected[node] = true;
                    selected.push((node, scores[node]));
                    let deg_inv = if degree[node] > 0.0 {
                        1.0 / degree[node]
                    } else {
                        0.0
                    };
                    for j in 0..n {
                        if adj[[node, j]] != 0.0 {
                            vote_ability[j] = (vote_ability[j] - deg_inv).max(0.0);
                        }
                    }
                }
                _ => break,
            }
        }

        Ok(selected)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HITSCentrality
// ─────────────────────────────────────────────────────────────────────────────

/// HITS (Hyperlink-Induced Topic Search) centrality — hubs and authorities.
///
/// HITS assigns two scores to each node:
/// - **Hub score** `h_i`: node `i` is a good hub if it points to many good authorities.
/// - **Authority score** `a_i`: node `i` is a good authority if it is pointed to by many good hubs.
///
/// Update rules (power iteration):
///   `a_i ← Σ_{j: j→i} h_j`
///   `h_i ← Σ_{j: i→j} a_j`
///
/// Followed by L2 normalisation of both vectors at each iteration.
///
/// For undirected graphs, the adjacency matrix is symmetric, so hubs and
/// authorities converge to proportional values (both equal the principal
/// eigenvector of `A`).
///
/// # Reference
/// Kleinberg (1999). "Authoritative sources in a hyperlinked environment."
/// *J. ACM*, 46(5), 604–632.
#[derive(Debug, Clone)]
pub struct HITSCentrality {
    /// Maximum number of power iterations.
    pub max_iter: usize,
    /// Convergence tolerance (L∞ norm of successive hub/auth changes).
    pub tol: f64,
}

impl HITSCentrality {
    /// Create a HITS instance.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Compute HITS hub and authority scores.
    ///
    /// # Returns
    /// `(hub_scores, authority_scores)` — both normalised to unit L2 norm.
    pub fn compute(&self, adj: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency".into()));
        }

        // Initialise: uniform hubs
        let mut h = Array1::from_elem(n, 1.0_f64 / n as f64);
        let mut a = Array1::<f64>::zeros(n);

        for iter in 0..self.max_iter {
            // Authority update: a = A^T h  (for directed; A^T[i,j] = A[j,i])
            let mut new_a = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    // edge j → i means adj[j,i] != 0
                    new_a[i] += adj[[j, i]] * h[j];
                }
            }
            // Normalise a
            let a_norm = new_a.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if a_norm > 1e-14 {
                new_a.iter_mut().for_each(|x| *x /= a_norm);
            }

            // Hub update: h = A a  (for directed: h_i = sum_j A[i,j] a_j)
            let mut new_h = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    new_h[i] += adj[[i, j]] * new_a[j];
                }
            }
            // Normalise h
            let h_norm = new_h.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if h_norm > 1e-14 {
                new_h.iter_mut().for_each(|x| *x /= h_norm);
            }

            // Check convergence
            let h_diff = h
                .iter()
                .zip(new_h.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let a_diff = a
                .iter()
                .zip(new_a.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            a = new_a;
            h = new_h;

            if h_diff < self.tol && a_diff < self.tol && iter > 0 {
                break;
            }
        }

        Ok((h, a))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TrustCentrality — signed network centrality
// ─────────────────────────────────────────────────────────────────────────────

/// Trust-based centrality for signed networks.
///
/// In signed networks, edges have positive (trust) or negative (distrust) weights.
/// Trust centrality measures the net trust flow to a node:
///
///   `trust_centrality(i) = Σ_{j: w_{ji} > 0} |w_{ji}| − Σ_{j: w_{ji} < 0} |w_{ji}|`
///
/// Extended measures:
/// - **Net trust**: incoming trust minus incoming distrust.
/// - **Positive centrality**: betweenness/closeness restricted to positive edges.
/// - **Balance score**: proportion of triangles that are structurally balanced.
///
/// # Reference
/// Guha et al. (2004). "Propagation of trust and distrust."
/// *WWW '04*, 403–412.
#[derive(Debug, Clone)]
pub struct TrustCentrality;

/// Results from trust centrality computation.
#[derive(Debug, Clone)]
pub struct TrustCentralityResult {
    /// Net trust score per node: positive = trusted, negative = distrusted.
    pub net_trust: Array1<f64>,
    /// Positive in-degree (number of trusting neighbours).
    pub positive_in_degree: Vec<usize>,
    /// Negative in-degree (number of distrusting neighbours).
    pub negative_in_degree: Vec<usize>,
    /// Trust ratio = positive_in_degree / (positive_in_degree + negative_in_degree).
    pub trust_ratio: Array1<f64>,
    /// Structural balance score of the entire network ∈ [0, 1].
    pub global_balance_score: f64,
}

impl TrustCentrality {
    /// Compute trust centrality measures for a signed adjacency matrix.
    ///
    /// # Arguments
    /// * `adj` — signed adjacency matrix; positive entries = trust, negative = distrust.
    ///   The magnitude is the weight; sign is the sentiment.
    pub fn compute(adj: &Array2<f64>) -> Result<TrustCentralityResult> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency".into()));
        }

        let mut net_trust = Array1::<f64>::zeros(n);
        let mut pos_in = vec![0usize; n];
        let mut neg_in = vec![0usize; n];

        for i in 0..n {
            for j in 0..n {
                let w = adj[[j, i]]; // incoming to i from j
                if w > 0.0 {
                    net_trust[i] += w;
                    pos_in[i] += 1;
                } else if w < 0.0 {
                    net_trust[i] += w; // subtracts
                    neg_in[i] += 1;
                }
            }
        }

        let trust_ratio = Array1::from_iter((0..n).map(|i| {
            let total = pos_in[i] + neg_in[i];
            if total == 0 {
                0.5 // neutral
            } else {
                pos_in[i] as f64 / total as f64
            }
        }));

        // Structural balance score: proportion of signed triangles that are balanced.
        // A triangle (i,j,k) is balanced if the product of the three signs is positive.
        let mut balanced = 0u64;
        let mut total_triangles = 0u64;
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let w_ij = adj[[i, j]];
                    let w_jk = adj[[j, k]];
                    let w_ik = adj[[i, k]];
                    // Only count if all three edges exist (non-zero)
                    if w_ij != 0.0 && w_jk != 0.0 && w_ik != 0.0 {
                        total_triangles += 1;
                        let product = w_ij.signum() * w_jk.signum() * w_ik.signum();
                        if product > 0.0 {
                            balanced += 1;
                        }
                    }
                }
            }
        }
        let global_balance_score = if total_triangles > 0 {
            balanced as f64 / total_triangles as f64
        } else {
            1.0 // no triangles → trivially balanced
        };

        Ok(TrustCentralityResult {
            net_trust,
            positive_in_degree: pos_in,
            negative_in_degree: neg_in,
            trust_ratio,
            global_balance_score,
        })
    }

    /// Compute the propagated trust from a set of source nodes using a
    /// random-walk-with-restart trust propagation model.
    ///
    /// The trust score of node `i` measures how much aggregate trust flows
    /// to `i` starting from the `sources` via positive edges (negative edges
    /// act as barriers / trust killers).
    ///
    /// # Arguments
    /// * `adj` — signed adjacency matrix.
    /// * `sources` — node indices that act as trusted seeds.
    /// * `alpha` — restart probability (teleportation back to sources) ∈ (0, 1).
    /// * `max_iter` — maximum power iterations.
    /// * `tol` — convergence tolerance.
    pub fn propagated_trust(
        adj: &Array2<f64>,
        sources: &[usize],
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Array1<f64>> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency".into()));
        }
        for &s in sources {
            if s >= n {
                return Err(GraphError::InvalidParameter {
                    param: "source node".into(),
                    value: s.to_string(),
                    expected: format!("< {n}"),
                    context: "TrustCentrality::propagated_trust".into(),
                });
            }
        }

        // Build positive-only row-normalised transition matrix
        // t_ij = adj+[i,j] / sum_k adj+[i,k]
        let mut trans = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| adj[[i, j]].max(0.0)).sum();
            if row_sum > 1e-14 {
                for j in 0..n {
                    if adj[[i, j]] > 0.0 {
                        trans[[i, j]] = adj[[i, j]] / row_sum;
                    }
                }
            }
        }

        // Personalised PageRank-style trust propagation
        let mut personalization = Array1::<f64>::zeros(n);
        for &s in sources {
            personalization[s] = 1.0;
        }
        let src_sum = personalization.sum();
        if src_sum > 0.0 {
            personalization.iter_mut().for_each(|x| *x /= src_sum);
        }

        let mut trust = personalization.clone();
        for _ in 0..max_iter {
            // new_trust = (1 - alpha) * T^T * trust + alpha * personalization
            let mut new_trust = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    new_trust[i] += trans[[j, i]] * trust[j];
                }
                new_trust[i] = (1.0 - alpha) * new_trust[i] + alpha * personalization[i];
            }
            let diff = trust
                .iter()
                .zip(new_trust.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            trust = new_trust;
            if diff < tol {
                break;
            }
        }

        Ok(trust)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CoreDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// k-core decomposition and k-shell index for undirected graphs.
///
/// The **k-core** of a graph is the maximal subgraph in which every node has
/// degree at least `k`.  The **core number** (or coreness) of a node is the
/// largest `k` for which the node belongs to the k-core.
///
/// The **k-shell** consists of nodes with core number exactly `k`
/// (in the k-core but not in the (k+1)-core).
///
/// ## Algorithm
/// Iterative degree peeling: repeatedly remove nodes with degree < k,
/// starting from k=1 upwards.
///
/// ## Time complexity
/// O(m) where m is the number of edges.
#[derive(Debug, Clone)]
pub struct CoreDecomposition {
    /// Core number (coreness) of each node.
    pub core_numbers: Vec<usize>,
    /// Shell index = core number (same as core number; kept separately for clarity).
    pub shell_index: Vec<usize>,
    /// Maximum core number (degeneracy of the graph).
    pub max_core: usize,
    /// Number of nodes.
    pub num_nodes: usize,
}

impl CoreDecomposition {
    /// Compute k-core decomposition from an adjacency matrix.
    pub fn compute(adj: &Array2<f64>) -> Self {
        let n = adj.nrows();
        if n == 0 {
            return Self {
                core_numbers: Vec::new(),
                shell_index: Vec::new(),
                max_core: 0,
                num_nodes: 0,
            };
        }

        // Build adjacency list (unweighted; any non-zero edge counts)
        let mut degree: Vec<usize> = (0..n)
            .map(|i| (0..n).filter(|&j| adj[[i, j]] != 0.0).count())
            .collect();
        let mut core_numbers = vec![0usize; n];
        let mut processed = vec![false; n];

        // Iterative degree-peeling (Batagelj & Zaversnik 2003).
        //
        // Canonical implementation: nodes are stored in a sorted array `vert[0..n]`
        // ordered by degree, with `bin[d]` marking the first index of bin d and
        // `pos[v]` giving each node's position in `vert`.  When a node's degree is
        // decremented the algorithm swaps it to the front of its current bin
        // (no duplicates are ever inserted).
        let max_deg = degree.iter().copied().max().unwrap_or(0);

        // bin[d] = first index in `vert` of the degree-d bucket.
        // Build by counting nodes per degree, then computing prefix sums.
        let mut bin = vec![0usize; max_deg + 2];
        for &d in &degree {
            bin[d] += 1;
        }
        // Convert counts to start positions (bin[d] becomes the first index)
        let mut start = 0usize;
        for d in 0..=max_deg {
            let cnt = bin[d];
            bin[d] = start;
            start += cnt;
        }
        bin[max_deg + 1] = n;

        // vert[i] = node at position i (sorted by degree ascending)
        // pos[v]  = position of node v in vert
        let mut vert = vec![0usize; n];
        let mut pos = vec![0usize; n];
        let mut tmp_bin = bin.clone();
        for v in 0..n {
            let d = degree[v];
            pos[v] = tmp_bin[d];
            vert[tmp_bin[d]] = v;
            tmp_bin[d] += 1;
        }

        for i in 0..n {
            let v = vert[i];
            core_numbers[v] = degree[v];
            processed[v] = true;

            for u in 0..n {
                if adj[[v, u]] == 0.0 || processed[u] {
                    continue;
                }
                // Only decrement u's degree if it is strictly higher than v's
                // core number.  This is the canonical Batagelj-Zaversnik condition:
                // if deg[u] == deg[v] they are in the same k-core and u keeps its
                // current effective degree.
                if degree[u] <= core_numbers[v] {
                    continue;
                }
                // Move u to the front of its current degree-bucket (swap with the
                // node at bin[du]) then advance bin[du], placing u into the next
                // lower bin.  This keeps `vert` sorted by current degree.
                let du = degree[u];
                let pu = pos[u];
                let pw = bin[du]; // first position of bin du (may be within processed range)
                let w = vert[pw]; // node currently at that position
                if u != w {
                    vert[pu] = w;
                    vert[pw] = u;
                    pos[w] = pu;
                    pos[u] = pw;
                }
                bin[du] += 1;
                degree[u] -= 1;
            }
        }

        let max_core = core_numbers.iter().copied().max().unwrap_or(0);
        let shell_index = core_numbers.clone();

        Self {
            core_numbers,
            shell_index,
            max_core,
            num_nodes: n,
        }
    }

    /// Return the set of node indices in the k-core (core number >= k).
    pub fn k_core_nodes(&self, k: usize) -> Vec<usize> {
        (0..self.num_nodes)
            .filter(|&i| self.core_numbers[i] >= k)
            .collect()
    }

    /// Return the set of node indices in the k-shell (core number == k).
    pub fn k_shell_nodes(&self, k: usize) -> Vec<usize> {
        (0..self.num_nodes)
            .filter(|&i| self.core_numbers[i] == k)
            .collect()
    }

    /// Return the main core (nodes with maximum core number).
    pub fn main_core_nodes(&self) -> Vec<usize> {
        self.k_core_nodes(self.max_core)
    }

    /// Compute the degeneracy ordering (nodes ordered by core number ascending).
    pub fn degeneracy_ordering(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.num_nodes).collect();
        order.sort_by_key(|&i| self.core_numbers[i]);
        order
    }

    /// Compute the coreness distribution: number of nodes at each shell level.
    pub fn shell_distribution(&self) -> Vec<(usize, usize)> {
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &c in &self.core_numbers {
            *counts.entry(c).or_insert(0) += 1;
        }
        let mut dist: Vec<(usize, usize)> = counts.into_iter().collect();
        dist.sort_by_key(|&(k, _)| k);
        dist
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EffectiveResistance
// ─────────────────────────────────────────────────────────────────────────────

/// Effective resistance (resistance distance) matrix.
///
/// The effective resistance between nodes `i` and `j` in an undirected weighted
/// graph is the electrical resistance between `i` and `j` when each edge `(u,v)`
/// with weight `w` is replaced by a resistor of resistance `1/w`.
///
/// ## Computation
///
/// For a connected graph with Laplacian `L`, the Moore-Penrose pseudo-inverse
/// `L⁺` satisfies:
///
///   `R(i,j) = L⁺[i,i] + L⁺[j,j] − 2 L⁺[i,j]`
///
/// where `L⁺ = (L + J/n)⁻¹ − J/n` with `J` the all-ones matrix.
///
/// ## Properties
/// - `R(i,j) ≥ 0` with equality iff `i = j`.
/// - The triangle inequality holds: `R(i,j) ≤ R(i,k) + R(k,j)` (metric).
/// - Captures both path length and the number of parallel paths (redundancy).
///
/// ## Reference
/// Klein & Randić (1993). "Resistance distance." *J. Math. Chem.*, 12, 81–95.
#[derive(Debug, Clone)]
pub struct EffectiveResistance {
    /// Pseudo-inverse of the Laplacian, shape `(n, n)`.
    pub l_plus: Array2<f64>,
    /// Pre-computed resistance matrix (R[i,j] = effective resistance), shape `(n, n)`.
    pub resistance_matrix: Array2<f64>,
    /// Number of nodes.
    pub num_nodes: usize,
}

impl EffectiveResistance {
    /// Compute the effective resistance matrix from a weighted adjacency matrix.
    ///
    /// # Note
    /// The graph must be connected.  If it is not, the result is only valid
    /// within connected components.
    pub fn compute(adj: &Array2<f64>) -> Result<Self> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency".into()));
        }

        // Build Laplacian L = D - A
        let mut lap = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let deg: f64 = adj.row(i).iter().map(|&x| x.abs()).sum();
            lap[[i, i]] = deg;
            for j in 0..n {
                if i != j {
                    lap[[i, j]] = -adj[[i, j]];
                }
            }
        }

        // Compute pseudo-inverse: L⁺ = (L + J/n)⁻¹ − J/n
        // where J is the all-ones matrix.
        let mut m = lap.clone();
        let inv_n = 1.0 / n as f64;
        for i in 0..n {
            for j in 0..n {
                m[[i, j]] += inv_n;
            }
        }

        // Invert m via Gaussian elimination with partial pivoting
        let m_inv = invert_matrix(&m)?;

        // L⁺ = m_inv - J/n
        let mut l_plus = m_inv;
        for i in 0..n {
            for j in 0..n {
                l_plus[[i, j]] -= inv_n;
            }
        }

        // Compute resistance matrix: R[i,j] = L⁺[i,i] + L⁺[j,j] - 2*L⁺[i,j]
        let mut r = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                r[[i, j]] = (l_plus[[i, i]] + l_plus[[j, j]] - 2.0 * l_plus[[i, j]]).max(0.0);
            }
        }

        Ok(Self {
            l_plus,
            resistance_matrix: r,
            num_nodes: n,
        })
    }

    /// Return the effective resistance between nodes `i` and `j`.
    pub fn resistance(&self, i: usize, j: usize) -> Result<f64> {
        let n = self.num_nodes;
        if i >= n || j >= n {
            return Err(GraphError::InvalidParameter {
                param: "i or j".into(),
                value: format!("{i}, {j}"),
                expected: format!("< {n}"),
                context: "EffectiveResistance::resistance".into(),
            });
        }
        Ok(self.resistance_matrix[[i, j]])
    }

    /// Compute the Kirchhoff index (sum of all pairwise effective resistances).
    ///
    /// `Kf = Σ_{i<j} R(i,j)`
    pub fn kirchhoff_index(&self) -> f64 {
        let n = self.num_nodes;
        let mut kf = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                kf += self.resistance_matrix[[i, j]];
            }
        }
        kf
    }

    /// Return the node with minimum average effective resistance
    /// (the "resistance centre" of the graph).
    pub fn resistance_centre(&self) -> usize {
        let n = self.num_nodes;
        let avg_resistance: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.resistance_matrix[[i, j]]).sum::<f64>() / n as f64)
            .collect();
        avg_resistance
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return the effective resistance diameter (maximum pairwise resistance).
    pub fn resistance_diameter(&self) -> f64 {
        self.resistance_matrix
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: matrix inversion via Gaussian elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Invert an `n × n` matrix via Gaussian elimination with partial pivoting.
fn invert_matrix(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    // Augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(GraphError::LinAlgError {
                operation: "matrix_inversion".into(),
                details: "matrix is singular or near-singular".into(),
            });
        }
        aug.swap(col, max_row);

        // Eliminate
        let pivot = aug[col][col];
        for k in col..(2 * n) {
            aug[col][k] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for k in col..(2 * n) {
                    let val = aug[col][k];
                    aug[row][k] -= factor * val;
                }
            }
        }
    }

    // Extract the right half (inverse)
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[i][n + j];
        }
    }
    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn complete4() -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    adj[[i, j]] = 1.0;
                }
            }
        }
        adj
    }

    fn path4() -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;
        adj
    }

    #[test]
    fn test_voterank_returns_k_nodes() {
        let adj = complete4();
        let vr = VoteRankCentrality::new(3);
        let spreaders = vr.compute(&adj).unwrap();
        assert_eq!(spreaders.len(), 3);
        // All selected nodes should be distinct
        let mut uniq = spreaders.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 3);
    }

    #[test]
    fn test_voterank_with_scores() {
        let adj = complete4();
        let vr = VoteRankCentrality::new(2);
        let results = vr.compute_with_scores(&adj).unwrap();
        assert_eq!(results.len(), 2);
        for (_, score) in &results {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_hits_undirected_symmetric() {
        // For undirected graph, hubs ~ authorities (both = principal eigenvector)
        let adj = complete4();
        let hits = HITSCentrality::new(100, 1e-10);
        let (h, a) = hits.compute(&adj).unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(a.len(), 4);
        // All nodes should have equal hub/authority score on K4
        let h_vals: Vec<f64> = h.to_vec();
        let max_h = h_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_h = h_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (max_h - min_h).abs() < 1e-6,
            "Hub scores should be equal on K4: max={max_h}, min={min_h}"
        );
    }

    #[test]
    fn test_hits_directed_hub_auth() {
        // Star graph: node 0 → 1, 2, 3 (directed out-star)
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[0, 3]] = 1.0;
        let hits = HITSCentrality::new(100, 1e-10);
        let (h, a) = hits.compute(&adj).unwrap();
        // Node 0 should be the top hub; nodes 1,2,3 top authorities
        assert!(h[0] > h[1].max(h[2]).max(h[3]));
        assert!(a[1] > a[0]);
    }

    #[test]
    fn test_trust_centrality_balanced_triangle() {
        // Balanced signed triangle: all positive
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[2, 0]] = 1.0;
        let res = TrustCentrality::compute(&adj).unwrap();
        assert_eq!(res.global_balance_score, 1.0); // all positive → balanced
        for &nt in res.net_trust.iter() {
            assert!(nt > 0.0);
        }
    }

    #[test]
    fn test_trust_centrality_imbalanced() {
        // Unbalanced triangle: two positive, one negative
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = -1.0;
        adj[[2, 0]] = -1.0;
        let res = TrustCentrality::compute(&adj).unwrap();
        // One negative edge → some distrust
        assert!((res.global_balance_score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_core_decomposition_path() {
        let adj = path4();
        let core = CoreDecomposition::compute(&adj);
        // In a path graph, all nodes have core number 1
        for &c in &core.core_numbers {
            assert_eq!(c, 1, "All path nodes should have core number 1");
        }
        assert_eq!(core.max_core, 1);
    }

    #[test]
    fn test_core_decomposition_complete() {
        let adj = complete4();
        let core = CoreDecomposition::compute(&adj);
        // K4: all nodes have core number 3 (= n-1)
        for &c in &core.core_numbers {
            assert_eq!(c, 3);
        }
        assert_eq!(core.max_core, 3);
        assert_eq!(core.main_core_nodes().len(), 4);
    }

    #[test]
    fn test_effective_resistance_path() {
        let adj = path4();
        let er = EffectiveResistance::compute(&adj).unwrap();
        // For a path graph, R(0,k) = k (one resistor per hop, weight=1)
        let r01 = er.resistance(0, 1).unwrap();
        let r02 = er.resistance(0, 2).unwrap();
        let r03 = er.resistance(0, 3).unwrap();
        assert!((r01 - 1.0).abs() < 1e-8, "R(0,1)={r01}");
        assert!((r02 - 2.0).abs() < 1e-8, "R(0,2)={r02}");
        assert!((r03 - 3.0).abs() < 1e-8, "R(0,3)={r03}");
    }

    #[test]
    fn test_effective_resistance_complete() {
        let adj = complete4();
        let er = EffectiveResistance::compute(&adj).unwrap();
        // In K_n, R(i,j) = 2/n for i != j
        let r01 = er.resistance(0, 1).unwrap();
        assert!((r01 - 2.0 / 4.0).abs() < 1e-8, "R(0,1) in K4 = {r01}");
    }

    #[test]
    fn test_kirchhoff_index() {
        let adj = path4();
        let er = EffectiveResistance::compute(&adj).unwrap();
        let kf = er.kirchhoff_index();
        // Kirchhoff index of path P_4: known to be Kf = sum R(i,j) for i<j
        // R(0,1)=1, R(0,2)=2, R(0,3)=3, R(1,2)=1, R(1,3)=2, R(2,3)=1 → Kf = 10
        assert!((kf - 10.0).abs() < 1e-7, "Kf(P4) = {kf}");
    }

    #[test]
    fn test_resistance_centre_path() {
        let adj = path4();
        let er = EffectiveResistance::compute(&adj).unwrap();
        let centre = er.resistance_centre();
        // For path graph, nodes 1 and 2 are more central than 0 and 3
        assert!(centre == 1 || centre == 2);
    }

    #[test]
    fn test_propagated_trust() {
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 2]] = 0.8;
        adj[[2, 3]] = 0.6;
        let trust = TrustCentrality::propagated_trust(&adj, &[0], 0.15, 100, 1e-9).unwrap();
        // Trust should be highest at source and decay outwards
        assert!(trust[0] > trust[3], "Source trust should be > distal trust");
    }
}
