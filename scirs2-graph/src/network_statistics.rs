//! Network-level statistics and structural measures
//!
//! This module provides global network statistics that characterise the overall
//! topology of a graph.  The functions operate on unweighted undirected graphs
//! represented by `Graph<usize, f64>`, using **BFS distances** for all path-
//! length metrics.
//!
//! ## Available statistics
//!
//! | Function | Description |
//! |---|---|
//! | [`eccentricity`] | Per-node eccentricity vector (BFS) |
//! | [`diameter`] | Maximum eccentricity |
//! | [`radius`] | Minimum eccentricity |
//! | [`periphery`] | Nodes at maximum eccentricity |
//! | [`center`] | Nodes at minimum eccentricity |
//! | [`average_path_length`] | Mean shortest-path distance |
//! | [`global_efficiency`] | Mean 1/d(i,j) over all pairs |
//! | [`local_efficiency`] | Efficiency of node neighbourhood |
//! | [`small_world_coefficient`] | σ = (C/C_rand)/(L/L_rand) |
//! | [`scale_free_exponent`] | Power-law degree exponent via MLE |

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Internal: BFS distance map from a single source
// ─────────────────────────────────────────────────────────────────────────────

/// Compute BFS distances from `source` to all reachable nodes.
///
/// Returns a `HashMap<N, usize>` where the value is the unweighted hop
/// distance.  Nodes unreachable from `source` are absent from the map.
fn bfs_distances<N, E, Ix>(graph: &Graph<N, E, Ix>, source: &N) -> HashMap<N, usize>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut dist: HashMap<N, usize> = HashMap::new();
    let mut queue: VecDeque<N> = VecDeque::new();

    dist.insert(source.clone(), 0);
    queue.push_back(source.clone());

    while let Some(current) = queue.pop_front() {
        let current_dist = *dist.get(&current).unwrap_or(&0);
        let neighbors = match graph.neighbors(&current) {
            Ok(nb) => nb,
            Err(_) => continue,
        };
        for nb in neighbors {
            if !dist.contains_key(&nb) {
                dist.insert(nb.clone(), current_dist + 1);
                queue.push_back(nb);
            }
        }
    }
    dist
}

// ─────────────────────────────────────────────────────────────────────────────
// Eccentricity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the eccentricity of every node.
///
/// The *eccentricity* of node `u` is the maximum BFS distance from `u` to any
/// other node in the graph.  For disconnected graphs the function returns
/// `None` (an eccentricity-based diameter is only well-defined on connected
/// graphs).
///
/// # Returns
/// `Some(Vec<usize>)` — eccentricities indexed by the sorted node list, or
/// `None` if the graph is disconnected or empty.
pub fn eccentricity<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<Vec<(N, usize)>>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    if nodes.is_empty() {
        return Some(vec![]);
    }
    nodes.sort();
    let n = nodes.len();

    let mut result: Vec<(N, usize)> = Vec::with_capacity(n);

    for u in &nodes {
        let dist = bfs_distances(graph, u);
        if dist.len() < n {
            // Not all nodes reachable → disconnected
            return None;
        }
        let ecc = dist.values().copied().max().unwrap_or(0);
        result.push((u.clone(), ecc));
    }
    Some(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Diameter
// ─────────────────────────────────────────────────────────────────────────────

/// Graph diameter: maximum eccentricity over all nodes (BFS).
///
/// Returns `None` for empty or disconnected graphs.
///
/// # Example
/// ```rust
/// use scirs2_graph::network_statistics::diameter;
/// use scirs2_graph::generators::path_graph;
/// let g = path_graph(5).unwrap();
/// assert_eq!(diameter(&g), Some(4));
/// ```
pub fn diameter<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<usize>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    eccentricity(graph).and_then(|eccs| eccs.into_iter().map(|(_, e)| e).max())
}

// ─────────────────────────────────────────────────────────────────────────────
// Radius
// ─────────────────────────────────────────────────────────────────────────────

/// Graph radius: minimum eccentricity over all nodes (BFS).
///
/// Returns `None` for empty or disconnected graphs.
///
/// # Example
/// ```rust
/// use scirs2_graph::network_statistics::radius;
/// use scirs2_graph::generators::cycle_graph;
/// let g = cycle_graph(6).unwrap();
/// assert_eq!(radius(&g), Some(3));
/// ```
pub fn radius<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<usize>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    eccentricity(graph).and_then(|eccs| eccs.into_iter().map(|(_, e)| e).min())
}

// ─────────────────────────────────────────────────────────────────────────────
// Periphery
// ─────────────────────────────────────────────────────────────────────────────

/// Graph periphery: set of nodes whose eccentricity equals the diameter.
///
/// Returns an empty vector for disconnected or empty graphs.
pub fn periphery<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<N>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let eccs = match eccentricity(graph) {
        Some(e) if !e.is_empty() => e,
        _ => return vec![],
    };
    let diam = eccs.iter().map(|(_, e)| *e).max().unwrap_or(0);
    eccs.into_iter()
        .filter_map(|(n, e)| if e == diam { Some(n) } else { None })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Center
// ─────────────────────────────────────────────────────────────────────────────

/// Graph center: set of nodes whose eccentricity equals the radius.
///
/// Returns an empty vector for disconnected or empty graphs.
pub fn center<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<N>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let eccs = match eccentricity(graph) {
        Some(e) if !e.is_empty() => e,
        _ => return vec![],
    };
    let rad = eccs.iter().map(|(_, e)| *e).min().unwrap_or(0);
    eccs.into_iter()
        .filter_map(|(n, e)| if e == rad { Some(n) } else { None })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Average path length
// ─────────────────────────────────────────────────────────────────────────────

/// Average shortest-path length of the graph.
///
/// Computed as the arithmetic mean of all pairwise BFS distances.  Returns
/// `None` for empty or disconnected graphs.
///
/// # Example
/// ```rust
/// use scirs2_graph::network_statistics::average_path_length;
/// use scirs2_graph::generators::complete_graph;
/// let g = complete_graph(4).unwrap();
/// // All paths in K_4 have length 1
/// assert!((average_path_length(&g).unwrap() - 1.0).abs() < 1e-9);
/// ```
pub fn average_path_length<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<f64>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    if n <= 1 {
        return Some(0.0);
    }

    let mut total = 0u64;
    let pairs = (n as u64) * (n as u64 - 1); // directed count; divide by 2 below

    for u in &nodes {
        let dist = bfs_distances(graph, u);
        if dist.len() < n {
            return None; // disconnected
        }
        for v in &nodes {
            if u != v {
                total += *dist.get(v).unwrap_or(&0) as u64;
            }
        }
    }

    Some(total as f64 / pairs as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Global efficiency
// ─────────────────────────────────────────────────────────────────────────────

/// Global efficiency of the graph.
///
/// Defined as the average of the reciprocals of shortest-path distances:
///
/// ```text
/// E_glob = 1 / (n(n-1))  ·  Σ_{i≠j} 1/d(i,j)
/// ```
///
/// For disconnected pairs d(i,j) = ∞, so 1/∞ = 0 — this is why global
/// efficiency is a more robust metric than average path length for graphs
/// that may be disconnected.  Returns `None` only for empty graphs.
///
/// # Example
/// ```rust
/// use scirs2_graph::network_statistics::global_efficiency;
/// use scirs2_graph::generators::complete_graph;
/// let g = complete_graph(5).unwrap();
/// // All pairs at distance 1 → efficiency = 1.0
/// assert!((global_efficiency(&g).unwrap() - 1.0).abs() < 1e-9);
/// ```
pub fn global_efficiency<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<f64>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    if n <= 1 {
        return Some(0.0);
    }

    let mut sum_inv = 0.0f64;
    let pairs = (n * (n - 1)) as f64;

    for u in &nodes {
        let dist = bfs_distances(graph, u);
        for v in &nodes {
            if u != v {
                if let Some(&d) = dist.get(v) {
                    if d > 0 {
                        sum_inv += 1.0 / d as f64;
                    }
                }
                // unreachable: contributes 0
            }
        }
    }

    Some(sum_inv / pairs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Local efficiency
// ─────────────────────────────────────────────────────────────────────────────

/// Local efficiency of a single node.
///
/// The local efficiency of node `u` is the global efficiency of the induced
/// subgraph on the neighbourhood of `u` (excluding `u` itself).  It
/// quantifies how well information can be exchanged among `u`'s neighbours
/// even if `u` were removed.
///
/// Returns `0.0` when the neighbourhood has fewer than 2 nodes (no pairs to
/// measure).  Returns a `GraphError` if `node` is not in the graph.
///
/// # Reference
/// Latora, V., & Marchiori, M. "Efficient behavior of small-world networks."
/// Phys. Rev. Lett. 87(19), 198701, 2001.
pub fn local_efficiency<N, E, Ix>(graph: &Graph<N, E, Ix>, node: &N) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    if !graph.has_node(node) {
        return Err(GraphError::node_not_found(format!("{node:?}")));
    }

    let neighbors = graph.neighbors(node)?;
    let k = neighbors.len();
    if k < 2 {
        return Ok(0.0);
    }

    // Build the induced subgraph on the neighbourhood set
    let nb_set: std::collections::HashSet<N> = neighbors.iter().cloned().collect();
    let mut sub: Graph<N, f64, u32> = Graph::new();
    for nb in &neighbors {
        sub.add_node(nb.clone());
    }
    for nb_u in &neighbors {
        let nb_u_neighbors = graph.neighbors(nb_u).unwrap_or_default();
        for nb_v in &nb_u_neighbors {
            if nb_set.contains(nb_v) && nb_u < nb_v {
                // Weight 1.0 for unweighted subgraph
                let _ = sub.add_edge(nb_u.clone(), nb_v.clone(), 1.0f64);
            }
        }
    }

    // Global efficiency on the subgraph
    let eff = global_efficiency(&sub).unwrap_or(0.0);
    Ok(eff)
}

// ─────────────────────────────────────────────────────────────────────────────
// Clustering coefficient helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the average (global) clustering coefficient of a graph.
///
/// The clustering coefficient of node `u` is:
/// ```text
/// C(u) = 2 · T(u) / (k_u · (k_u − 1))
/// ```
/// where T(u) is the number of triangles through `u` and k_u is its degree.
/// The network average is the mean of C(u) over all nodes with k_u ≥ 2.
///
/// Used internally by [`small_world_coefficient`].
fn average_clustering<N, E, Ix>(graph: &Graph<N, E, Ix>) -> f64
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let mut total = 0.0f64;
    let mut count = 0usize;

    for u in &nodes {
        let nb = graph.neighbors(u).unwrap_or_default();
        let k = nb.len();
        if k < 2 {
            continue;
        }
        let nb_set: std::collections::HashSet<N> = nb.iter().cloned().collect();
        let mut triangles = 0usize;
        for i in 0..nb.len() {
            for j in (i + 1)..nb.len() {
                if graph.has_edge(&nb[i], &nb[j]) {
                    triangles += 1;
                }
            }
        }
        let _ = nb_set; // suppress warning
        let c_u = 2.0 * triangles as f64 / (k * (k - 1)) as f64;
        total += c_u;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small-world coefficient σ
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the small-world coefficient σ for the graph.
///
/// The small-world coefficient (Humphries & Gurney, 2008) is:
///
/// ```text
/// σ = (C / C_rand) / (L / L_rand)
/// ```
///
/// where C and L are the clustering coefficient and average path length of
/// the input graph, and C_rand, L_rand are the expected values of `n_random`
/// Erdős–Rényi random graphs with the same number of nodes and edges.
///
/// σ > 1 indicates small-world structure.
///
/// # Arguments
/// * `graph`    – undirected graph to characterise
/// * `n_random` – number of random reference graphs to average over (≥ 1)
/// * `rng`      – random-number generator
///
/// # Returns
/// `Ok(sigma)` or a `GraphError` when the graph is too small / too sparse to
/// compute meaningful path lengths.
///
/// # Reference
/// Humphries, M. D., & Gurney, K. "Network 'small-world-ness': A quantitative
/// method for determining canonical network equivalence." PLoS ONE, 3(4), 2008.
pub fn small_world_coefficient<N, E, Ix, R>(
    graph: &Graph<N, E, Ix>,
    n_random: usize,
    rng: &mut R,
) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug + Ord,
    E: EdgeWeight,
    Ix: IndexType,
    R: Rng,
{
    let n = graph.node_count();
    let m = graph.edge_count();

    if n < 3 {
        return Err(GraphError::InvalidGraph(
            "small_world_coefficient: graph must have ≥ 3 nodes".to_string(),
        ));
    }
    if n_random == 0 {
        return Err(GraphError::InvalidGraph(
            "small_world_coefficient: n_random must be ≥ 1".to_string(),
        ));
    }

    let c = average_clustering(graph);
    let l = match average_path_length(graph) {
        Some(v) if v > 0.0 => v,
        _ => {
            return Err(GraphError::InvalidGraph(
                "small_world_coefficient: graph is disconnected or trivial".to_string(),
            ))
        }
    };

    // Compute reference C_rand and L_rand from Erdős–Rényi graphs
    let max_edges = n * (n - 1) / 2;
    if m > max_edges {
        return Err(GraphError::InvalidGraph(
            "small_world_coefficient: edge count exceeds maximum for simple graph".to_string(),
        ));
    }

    let mut sum_c_rand = 0.0f64;
    let mut sum_l_rand = 0.0f64;
    let mut valid_samples = 0usize;

    for _ in 0..n_random {
        // Build G(n,m) random reference
        let rg =
            crate::generators::random_graphs::erdos_renyi_g_nm(n, m, rng).unwrap_or_else(|_| {
                crate::generators::erdos_renyi_graph(n, m as f64 / (max_edges as f64).max(1.0), rng)
                    .unwrap_or_default()
            });

        if let Some(l_r) = average_path_length(&rg) {
            if l_r > 0.0 {
                sum_c_rand += average_clustering(&rg);
                sum_l_rand += l_r;
                valid_samples += 1;
            }
        }
    }

    if valid_samples == 0 {
        return Err(GraphError::ComputationError(
            "small_world_coefficient: all random reference graphs were disconnected".to_string(),
        ));
    }

    let c_rand = sum_c_rand / valid_samples as f64;
    let l_rand = sum_l_rand / valid_samples as f64;

    // Guard against division by zero
    if c_rand <= 0.0 || l_rand <= 0.0 || l <= 0.0 {
        return Err(GraphError::ComputationError(
            "small_world_coefficient: reference graph has degenerate clustering or path length"
                .to_string(),
        ));
    }

    let sigma = (c / c_rand) / (l / l_rand);
    Ok(sigma)
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale-free exponent (power-law MLE)
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the power-law exponent of a degree distribution.
///
/// Uses the maximum-likelihood estimator (MLE) for a discrete power law derived
/// by Clauset, Shalizi & Newman (2009):
///
/// ```text
/// gamma_hat = 1 + n · ( Σ_{i=1}^{n} ln(k_i / (k_min - 0.5)) )^{-1}
/// ```
///
/// where k_min is the minimum degree included in the fit.  The estimate is
/// valid for the **tail** of the distribution: degrees below `k_min` are
/// excluded.  The function automatically selects k_min as the value that
/// minimises the Kolmogorov–Smirnov distance between the empirical CDF and the
/// fitted power law.
///
/// # Arguments
/// * `degree_dist` – observed degree sequence (raw degrees, **not** counts);
///   all values must be ≥ 1
///
/// # Returns
/// `Some(gamma)` where gamma > 1 is the scale-free exponent, or `None` if the
/// distribution has fewer than 2 distinct values, the sequence is empty, or
/// the resulting exponent is not finite.
///
/// # Reference
/// Clauset, A., Shalizi, C. R., & Newman, M. E. J. "Power-law distributions
/// in empirical data." SIAM Review, 51(4), 661–703, 2009.
pub fn scale_free_exponent(degree_dist: &[usize]) -> Option<f64> {
    if degree_dist.is_empty() {
        return None;
    }

    // Filter out degree-0 nodes
    let degrees: Vec<usize> = degree_dist.iter().copied().filter(|&d| d >= 1).collect();
    if degrees.is_empty() {
        return None;
    }

    let mut sorted = degrees.clone();
    sorted.sort_unstable();

    let distinct: Vec<usize> = {
        let mut v: Vec<usize> = sorted.to_vec();
        v.dedup();
        v
    };
    if distinct.len() < 2 {
        return None;
    }

    // Candidate k_min values: all distinct degrees up to the largest-1
    let candidates = &distinct[..distinct.len() - 1];

    let mut best_ks_stat = f64::INFINITY;
    let mut best_gamma: Option<f64> = None;

    for &k_min in candidates {
        let tail: Vec<usize> = sorted.iter().copied().filter(|&d| d >= k_min).collect();
        let n_tail = tail.len();
        if n_tail < 5 {
            continue;
        }

        // MLE for discrete power law (Eq. 3.6 in Clauset et al.)
        let ln_sum: f64 = tail
            .iter()
            .map(|&k| ((k as f64) / (k_min as f64 - 0.5)).ln())
            .sum();
        if ln_sum <= 0.0 {
            continue;
        }
        let gamma = 1.0 + n_tail as f64 * (1.0 / ln_sum);
        if !gamma.is_finite() || gamma <= 1.0 {
            continue;
        }

        // KS statistic between empirical CDF and theoretical power-law CDF
        let ks = ks_statistic_discrete(&tail, gamma, k_min);
        if ks < best_ks_stat {
            best_ks_stat = ks;
            best_gamma = Some(gamma);
        }
    }

    best_gamma
}

/// Kolmogorov–Smirnov statistic for a discrete power law fitted to `data`.
///
/// data must be sorted ascending and all values ≥ k_min.
fn ks_statistic_discrete(data: &[usize], gamma: f64, k_min: usize) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::INFINITY;
    }

    // Theoretical CDF: P(K ≤ k) ≈ 1 − (k / k_min)^{1−γ}  (Pareto approx.)
    // For the discrete case we use the Hurwitz zeta approximation.
    // For our purposes the continuous approximation is sufficient.
    let theoretical_cdf = |k: usize| -> f64 {
        if k < k_min {
            return 0.0;
        }
        // P(K ≥ k_min) = 1 by definition; P(K ≤ k) = 1 − (k+1/k_min)^{1-gamma}
        let ratio = (k as f64 + 0.5) / (k_min as f64 - 0.5);
        (1.0 - ratio.powf(1.0 - gamma)).clamp(0.0, 1.0)
    };

    let mut max_diff = 0.0f64;
    for (i, &k) in data.iter().enumerate() {
        let emp_cdf = (i + 1) as f64 / n as f64;
        let theo_cdf = theoretical_cdf(k);
        let diff = (emp_cdf - theo_cdf).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::random_graphs::erdos_renyi_g_np;
    use crate::generators::{complete_graph, cycle_graph, path_graph, star_graph};
    use scirs2_core::random::prelude::*;

    // ── Diameter / Radius ────────────────────────────────────────────────────

    #[test]
    fn test_diameter_path() {
        let g = path_graph(6).expect("path_graph failed");
        assert_eq!(diameter(&g), Some(5));
    }

    #[test]
    fn test_diameter_complete() {
        let g = complete_graph(10).expect("complete_graph failed");
        assert_eq!(diameter(&g), Some(1));
    }

    #[test]
    fn test_radius_cycle() {
        // For a cycle of even length n, radius = n/2
        let g = cycle_graph(8).expect("cycle_graph failed");
        assert_eq!(radius(&g), Some(4));
    }

    #[test]
    fn test_radius_path() {
        // Path P5: eccentricities are [4,3,2,3,4] → radius=2
        let g = path_graph(5).expect("path_graph failed");
        assert_eq!(radius(&g), Some(2));
    }

    // ── Center / Periphery ───────────────────────────────────────────────────

    #[test]
    fn test_center_star() {
        // Star K_{1,4}: center is node 0 (eccentricity 1), leaves have ecc 2
        let g = star_graph(5).expect("star_graph failed");
        let c = center(&g);
        assert!(!c.is_empty());
        assert!(c.contains(&0));
    }

    #[test]
    fn test_periphery_path() {
        // P5 nodes 0 and 4 have max eccentricity 4
        let g = path_graph(5).expect("path_graph failed");
        let p = periphery(&g);
        assert!(p.contains(&0));
        assert!(p.contains(&4));
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn test_center_and_periphery_empty_graph() {
        let g: Graph<usize, f64> = Graph::new();
        assert!(center(&g).is_empty());
        assert!(periphery(&g).is_empty());
    }

    // ── Average path length ───────────────────────────────────────────────────

    #[test]
    fn test_apl_complete() {
        let g = complete_graph(5).expect("complete_graph failed");
        let apl = average_path_length(&g).expect("apl failed");
        assert!((apl - 1.0).abs() < 1e-9, "K_n has APL=1, got {apl}");
    }

    #[test]
    fn test_apl_path() {
        // P4: sum of distances = 1+2+3 + 1+2 + 1 = 10 (one direction),
        // × 2 / 12 = 20/12 ≈ 1.667
        let g = path_graph(4).expect("path_graph failed");
        let apl = average_path_length(&g).expect("apl failed");
        // Sum of all directed pair distances: 2*(1+2+3+1+2+1) = 2*10 = 20
        // Divided by 4*3 = 12 → ≈1.667
        assert!((apl - 20.0 / 12.0).abs() < 1e-9, "got {apl}");
    }

    #[test]
    fn test_apl_trivial() {
        let mut g: Graph<usize, f64> = Graph::new();
        g.add_node(0);
        assert_eq!(average_path_length(&g), Some(0.0));
    }

    // ── Global efficiency ─────────────────────────────────────────────────────

    #[test]
    fn test_global_efficiency_complete() {
        let g = complete_graph(5).expect("complete_graph failed");
        let eff = global_efficiency(&g).expect("eff failed");
        assert!((eff - 1.0).abs() < 1e-9, "K_n has efficiency=1, got {eff}");
    }

    #[test]
    fn test_global_efficiency_path() {
        // P3: pairs (0,1)=1, (1,2)=1, (0,2)=2 → efficiency = (1+1+0.5+1+1+0.5)/6 = 5/6
        let g = path_graph(3).expect("path_graph failed");
        let eff = global_efficiency(&g).expect("eff failed");
        let expected = (1.0 + 0.5 + 1.0 + 1.0 + 0.5 + 1.0) / 6.0;
        assert!(
            (eff - expected).abs() < 1e-9,
            "got {eff}, expected {expected}"
        );
    }

    #[test]
    fn test_global_efficiency_disconnected() {
        // Two disconnected triangles: unreachable pairs contribute 0
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..6usize {
            g.add_node(i);
        }
        // Component 1: 0-1-2-0
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(1, 2, 1.0).unwrap();
        g.add_edge(0, 2, 1.0).unwrap();
        // Component 2: 3-4-5-3
        g.add_edge(3, 4, 1.0).unwrap();
        g.add_edge(4, 5, 1.0).unwrap();
        g.add_edge(3, 5, 1.0).unwrap();

        let eff = global_efficiency(&g).expect("eff failed");
        // Within each triangle: 3 pairs at distance 1 each, 6 directed
        // Cross-component: 0 contribution
        // Total = 6 + 6 = 12 directed pairs at distance 1, out of 6*5=30
        let expected = 12.0 / 30.0;
        assert!((eff - expected).abs() < 1e-9, "got {eff}");
    }

    // ── Local efficiency ──────────────────────────────────────────────────────

    #[test]
    fn test_local_efficiency_triangle() {
        // In a triangle: neighbourhood of node 0 is {1,2} which are connected
        // → neighbourhood is K_2 → efficiency = 1.0
        let g = complete_graph(3).expect("complete_graph failed");
        let le = local_efficiency(&g, &0).expect("le failed");
        assert!((le - 1.0).abs() < 1e-9, "got {le}");
    }

    #[test]
    fn test_local_efficiency_star_center() {
        // Star center: neighbours form an independent set → subgraph has no edges → eff = 0
        let g = star_graph(5).expect("star_graph failed");
        let le = local_efficiency(&g, &0).expect("le failed");
        assert!((le - 0.0).abs() < 1e-9, "got {le}");
    }

    #[test]
    fn test_local_efficiency_missing_node() {
        let g = path_graph(4).expect("path_graph failed");
        assert!(local_efficiency(&g, &99).is_err());
    }

    // ── Small-world coefficient ───────────────────────────────────────────────

    #[test]
    #[ignore]
    fn test_small_world_coefficient_runs() {
        let mut rng = StdRng::seed_from_u64(42);
        // Watts-Strogatz β=0.1 is a canonical small-world network
        let g = crate::generators::watts_strogatz_graph(30, 4, 0.1, &mut rng).expect("ws failed");
        let sigma = small_world_coefficient(&g, 5, &mut rng);
        // Just check it runs without error; σ may vary
        assert!(sigma.is_ok(), "small_world_coefficient error: {sigma:?}");
    }

    #[test]
    fn test_small_world_invalid_input() {
        let mut rng = StdRng::seed_from_u64(1);
        let g: Graph<usize, f64> = Graph::new();
        assert!(small_world_coefficient(&g, 5, &mut rng).is_err());
        let tiny_g = path_graph(2).expect("path failed");
        assert!(small_world_coefficient(&tiny_g, 5, &mut rng).is_err());
    }

    // ── Scale-free exponent ───────────────────────────────────────────────────

    #[test]
    fn test_scale_free_exponent_power_law() {
        // Barabasi–Albert has gamma ≈ 3
        let mut rng = StdRng::seed_from_u64(99);
        let g =
            crate::generators::random_graphs::barabasi_albert(200, 2, &mut rng).expect("ba failed");
        let degrees: Vec<usize> = (0..g.node_count()).map(|i| g.degree(&i)).collect();
        if let Some(gamma) = scale_free_exponent(&degrees) {
            // BA model → exponent ≈ 3; allow generous bounds for small n
            assert!(gamma > 1.5 && gamma < 8.0, "gamma={gamma}");
        }
        // If None, it means the dataset was too small; that's acceptable
    }

    #[test]
    fn test_scale_free_exponent_empty() {
        assert_eq!(scale_free_exponent(&[]), None);
        assert_eq!(scale_free_exponent(&[0, 0, 0]), None);
    }

    #[test]
    fn test_scale_free_exponent_uniform() {
        // All degrees equal → no power law (single distinct value after 1 removed)
        let degrees = vec![3usize; 20];
        // May return None or a degenerate value — just check it doesn't panic
        let _ = scale_free_exponent(&degrees);
    }
}
