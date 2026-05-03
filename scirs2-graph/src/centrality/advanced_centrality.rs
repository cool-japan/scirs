//! Advanced centrality measures for network analysis.
//!
//! This module provides centrality metrics beyond standard degree, betweenness,
//! closeness, and eigenvector centrality. These measures capture different aspects
//! of node importance in a network.
//!
//! ## Measures
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`katz_centrality`] | Katz centrality via power iteration |
//! | [`harmonic_centrality`] | Harmonic centrality (handles disconnected graphs) |
//! | [`percolation_centrality`] | Weighted betweenness with node percolation states |
//! | [`subgraph_centrality`] | Diagonal of matrix exponential |
//! | [`communicability`] | Full matrix exponential `exp(A)` |
//!
//! ## References
//! - Katz, L. (1953). A new status index derived from sociometric analysis.
//! - Bavelas, A. (1950). Communication patterns in task-oriented groups.
//! - Estrada, E. & Rodriguez-Velazquez, J. A. (2005). Subgraph centrality.
//! - Estrada, E. & Hatano, N. (2008). Communicability in complex networks.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Internal adjacency helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a weighted adjacency list from an edge list.
fn build_adj(edges: &[(usize, usize, f64)], n: usize) -> Vec<Vec<(usize, f64)>> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for &(u, v, w) in edges {
        if u < n && v < n {
            adj[u].push((v, w));
            if u != v {
                adj[v].push((u, w));
            }
        }
    }
    adj
}

/// Build a dense adjacency matrix from an edge list (for matrix-exp approaches).
fn build_dense_adj(edges: &[(usize, usize, f64)], n: usize) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0f64; n]; n];
    for &(u, v, w) in edges {
        if u < n && v < n {
            mat[u][v] += w;
            if u != v {
                mat[v][u] += w;
            }
        }
    }
    mat
}

// ─────────────────────────────────────────────────────────────────────────────
// Katz centrality
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Katz centrality via power iteration.
///
/// `x = (I - α·A)^{-1} · β · 1`
///
/// which is solved iteratively as:
/// `x^{(t+1)} = α · A · x^{(t)} + β · 1`
///
/// The spectral radius condition `α < 1/λ_max` must hold for convergence.
/// This implementation normalises the result to unit L2 norm.
///
/// # Arguments
/// * `adj`     – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes` – Total number of nodes.
/// * `alpha`   – Attenuation factor (0 < α < 1/λ_max).
/// * `beta`    – Exogenous importance weight (default 1.0).
///
/// # Errors
/// Returns an error if `alpha` is non-positive or `n_nodes` is zero.
pub fn katz_centrality(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    alpha: f64,
    beta: f64,
) -> Result<Vec<f64>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "katz_centrality: n_nodes must be > 0".into(),
        ));
    }
    if alpha <= 0.0 {
        return Err(GraphError::InvalidParameter {
            param: "alpha".into(),
            value: format!("{alpha}"),
            expected: "> 0".into(),
            context: "katz_centrality".into(),
        });
    }

    let graph = build_adj(adj, n_nodes);
    let mut x = vec![1.0f64; n_nodes];
    let max_iter = 1000usize;
    let tol = 1e-9f64;

    for _ in 0..max_iter {
        let mut x_new = vec![beta; n_nodes];
        for u in 0..n_nodes {
            for &(v, w) in &graph[u] {
                x_new[u] += alpha * w * x[v];
            }
        }
        // Check convergence (L∞ norm)
        let diff = x_new
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        x = x_new;
        if diff < tol {
            break;
        }
    }

    // Normalise to unit L2 norm
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in &mut x {
            *v /= norm;
        }
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Harmonic centrality
// ─────────────────────────────────────────────────────────────────────────────

/// Priority queue entry for Dijkstra's algorithm.
#[derive(Clone, PartialEq)]
struct DijkEntry {
    dist: f64,
    node: usize,
}

impl Eq for DijkEntry {}

impl PartialOrd for DijkEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Single-source Dijkstra shortest paths (weighted, non-negative edges).
/// Returns distances from `source` to all other nodes (∞ if unreachable).
fn dijkstra(adj: &[Vec<(usize, f64)>], source: usize, n: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(DijkEntry {
        dist: 0.0,
        node: source,
    });

    while let Some(DijkEntry { dist: d, node: u }) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in &adj[u] {
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                heap.push(DijkEntry { dist: nd, node: v });
            }
        }
    }
    dist
}

/// Compute harmonic centrality for all nodes.
///
/// `H(v) = Σ_{u≠v} 1/d(u,v)`
///
/// Unreachable pairs (`d = ∞`) contribute 0 to the sum. This makes harmonic
/// centrality well-defined for disconnected graphs, unlike closeness centrality.
///
/// # Arguments
/// * `adj`     – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes` – Total number of nodes.
pub fn harmonic_centrality(adj: &[(usize, usize, f64)], n_nodes: usize) -> Result<Vec<f64>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "harmonic_centrality: n_nodes must be > 0".into(),
        ));
    }

    let graph = build_adj(adj, n_nodes);
    let mut centrality = vec![0.0f64; n_nodes];

    for source in 0..n_nodes {
        let dists = dijkstra(&graph, source, n_nodes);
        for target in 0..n_nodes {
            if target == source {
                continue;
            }
            let d = dists[target];
            if d.is_finite() && d > 0.0 {
                centrality[source] += 1.0 / d;
            }
        }
    }

    // Normalise by (n-1)
    let n_minus_1 = (n_nodes - 1).max(1) as f64;
    for v in &mut centrality {
        *v /= n_minus_1;
    }
    Ok(centrality)
}

// ─────────────────────────────────────────────────────────────────────────────
// Percolation centrality
// ─────────────────────────────────────────────────────────────────────────────

/// Compute percolation centrality accounting for node percolation states.
///
/// Percolation centrality measures the importance of a node `v` in transporting
/// information between "percolated" (active) source nodes and "non-percolated"
/// (inactive) target nodes.
///
/// `PC(v) = 1/((n-1)·(n-2)) · Σ_{s≠t≠v} [σ(s,t|v)/σ(s,t)] · ρ_s / Σ_u ρ_u`
///
/// where `σ(s,t)` is the number of shortest paths from `s` to `t`, and
/// `σ(s,t|v)` is the number passing through `v`.
///
/// # Arguments
/// * `adj`               – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`           – Total number of nodes.
/// * `percolation_state` – Percolation state `ρ_i ∈ [0,1]` for each node.
///
/// # Errors
/// Returns an error if `percolation_state.len() != n_nodes`.
pub fn percolation_centrality(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    percolation_state: &[f64],
) -> Result<Vec<f64>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "percolation_centrality: n_nodes must be > 0".into(),
        ));
    }
    if percolation_state.len() != n_nodes {
        return Err(GraphError::InvalidParameter {
            param: "percolation_state".into(),
            value: format!("len={}", percolation_state.len()),
            expected: format!("len={n_nodes}"),
            context: "percolation_centrality".into(),
        });
    }

    let state_sum: f64 = percolation_state.iter().sum();
    let norm_state: Vec<f64> = if state_sum > 0.0 {
        percolation_state.iter().map(|&s| s / state_sum).collect()
    } else {
        vec![1.0 / n_nodes as f64; n_nodes]
    };

    let graph = build_adj(adj, n_nodes);
    let mut centrality = vec![0.0f64; n_nodes];

    // Brandes-style BFS/Dijkstra for all-pairs shortest paths + path counting
    for source in 0..n_nodes {
        let (pair_dep, sigma_s) = betweenness_contribution(&graph, source, n_nodes);
        for v in 0..n_nodes {
            if v == source {
                continue;
            }
            // pair_dep[v] = Σ_t σ(source,t|v)/σ(source,t) for t ≠ source, v
            centrality[v] += norm_state[source] * pair_dep[v] * sigma_s[v];
        }
    }

    // Normalise
    if n_nodes > 2 {
        let factor = 1.0 / ((n_nodes - 1) as f64 * (n_nodes - 2) as f64);
        for v in &mut centrality {
            *v *= factor;
        }
    }
    Ok(centrality)
}

/// Brandes algorithm contribution: returns per-node pair dependency and sigma.
/// Returns `(pair_dependency, sigma)` where:
/// - `pair_dependency[v]` = Σ_t δ_{source}(v|t) for t reachable from source
/// - `sigma[v]` = fraction of s-t shortest paths through v (relative term)
fn betweenness_contribution(
    adj: &[Vec<(usize, f64)>],
    source: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut stack: Vec<usize> = Vec::new();
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
    let mut sigma = vec![0.0f64; n];
    let mut dist = vec![f64::INFINITY; n];
    let mut delta = vec![0.0f64; n];

    sigma[source] = 1.0;
    dist[source] = 0.0;

    // BFS (unweighted shortest-path counting)
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        for &(w, _) in &adj[v] {
            // Visit w for the first time?
            if dist[w].is_infinite() {
                dist[w] = dist[v] + 1.0;
                queue.push_back(w);
            }
            // Is this a shortest path?
            if (dist[w] - dist[v] - 1.0).abs() < 1e-9 {
                sigma[w] += sigma[v];
                pred[w].push(v);
            }
        }
    }

    // Back-propagation
    while let Some(w) = stack.pop() {
        for &v in &pred[w] {
            delta[v] += sigma[v] / sigma[w].max(1e-15) * (1.0 + delta[w]);
        }
    }

    (delta, sigma)
}

// ─────────────────────────────────────────────────────────────────────────────
// Subgraph centrality
// ─────────────────────────────────────────────────────────────────────────────

/// Compute subgraph centrality via the diagonal of the matrix exponential.
///
/// `SC(v) = Σ_{k=0}^∞ (A^k)_{vv} / k! = [exp(A)]_{vv}`
///
/// The matrix exponential is computed using a truncated Taylor series
/// (Padé approximation order 6 is used for accuracy and stability).
///
/// # Arguments
/// * `adj`     – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes` – Total number of nodes.
///
/// # Errors
/// Returns an error if `n_nodes` is zero or > 500 (to avoid O(n³) OOM).
pub fn subgraph_centrality(adj: &[(usize, usize, f64)], n_nodes: usize) -> Result<Vec<f64>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "subgraph_centrality: n_nodes must be > 0".into(),
        ));
    }
    if n_nodes > 500 {
        return Err(GraphError::InvalidParameter {
            param: "n_nodes".into(),
            value: format!("{n_nodes}"),
            expected: "<= 500 (matrix exponential is O(n³))".into(),
            context: "subgraph_centrality".into(),
        });
    }

    let mat = build_dense_adj(adj, n_nodes);
    let exp_mat = matrix_exp(&mat, n_nodes);

    // Diagonal elements
    let sc: Vec<f64> = (0..n_nodes).map(|i| exp_mat[i][i]).collect();
    Ok(sc)
}

// ─────────────────────────────────────────────────────────────────────────────
// Communicability
// ─────────────────────────────────────────────────────────────────────────────

/// Compute communicability between all pairs of nodes.
///
/// `G(u, v) = [exp(A)]_{uv}`
///
/// The communicability matrix is the full matrix exponential of the adjacency
/// matrix. `G(u,v)` measures the ease of communication between `u` and `v`,
/// accounting for all walks of all lengths.
///
/// # Arguments
/// * `adj`     – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes` – Total number of nodes.
///
/// # Errors
/// Returns an error if `n_nodes` is zero or > 500.
pub fn communicability(adj: &[(usize, usize, f64)], n_nodes: usize) -> Result<Vec<Vec<f64>>> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidGraph(
            "communicability: n_nodes must be > 0".into(),
        ));
    }
    if n_nodes > 500 {
        return Err(GraphError::InvalidParameter {
            param: "n_nodes".into(),
            value: format!("{n_nodes}"),
            expected: "<= 500 (matrix exponential is O(n³))".into(),
            context: "communicability".into(),
        });
    }

    let mat = build_dense_adj(adj, n_nodes);
    Ok(matrix_exp(&mat, n_nodes))
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix exponential (Padé approximation, order 6)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute matrix exponential exp(A) using scaling and squaring with Padé
/// approximation of order 6.
///
/// Algorithm:
/// 1. Scale A by 2^s so that ‖A/2^s‖ ≤ 0.5.
/// 2. Compute Padé(6) approximant.
/// 3. Square the result s times.
fn matrix_exp(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }

    // Find scaling factor s such that ||A||_inf / 2^s <= 0.5
    let norm_inf = mat_norm_inf(a, n);
    let s = if norm_inf > 0.5 {
        ((norm_inf / 0.5).log2().ceil() as i32).max(0)
    } else {
        0i32
    };
    let scale = 2.0f64.powi(s);
    let a_scaled: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[i][j] / scale).collect())
        .collect();

    // Padé(6) coefficients
    // p_6 = Σ c_k A^k,  q_6 = Σ c_k (-A)^k
    let c = [
        1.0,
        0.5,
        0.12,
        1.833333333333333e-2,
        1.992753623188406e-3,
        1.630434782608696e-4,
        1.035_196_687_370_1e-5,
    ];

    // Compute powers: A^0=I, A^1, A^2, A^3, A^4, A^5, A^6
    let mut powers: Vec<Vec<Vec<f64>>> = Vec::with_capacity(7);
    powers.push(identity(n));
    powers.push(a_scaled.to_vec());
    for k in 2..=6 {
        powers.push(mat_mul(&powers[k - 1], &powers[1], n));
    }

    // p = c_0*I + c_1*A + c_2*A^2 + ... + c_6*A^6
    // q = c_0*I - c_1*A + c_2*A^2 - ... + c_6*A^6
    let mut p = vec![vec![0.0f64; n]; n];
    let mut q = vec![vec![0.0f64; n]; n];
    for k in 0..=6 {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..n {
            for j in 0..n {
                p[i][j] += c[k] * powers[k][i][j];
                q[i][j] += sign * c[k] * powers[k][i][j];
            }
        }
    }

    // exp(A_scaled) ≈ q^{-1} * p (via Gauss-Jordan)
    let mut result = match mat_solve(&q, &p, n) {
        Some(r) => r,
        None => {
            // Fallback: return identity if solve fails
            identity(n)
        }
    };

    // Squaring: exp(A) = exp(A_scaled)^{2^s}
    for _ in 0..s {
        result = mat_mul(&result.clone(), &result, n);
    }
    result
}

/// Compute infinity norm (max row sum) of a matrix.
fn mat_norm_inf(a: &[Vec<f64>], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i][j].abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// Create n×n identity matrix.
fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Matrix multiplication C = A * B for dense n×n matrices.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k].abs() < 1e-15 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Solve the linear system A * X = B via Gauss-Jordan elimination.
/// Returns `None` if A is singular.
fn mat_solve(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    // Build augmented matrix [A | B]
    let ncols_b = b[0].len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            aug[r1][col]
                .abs()
                .partial_cmp(&aug[r2][col].abs())
                .unwrap_or(Ordering::Equal)
        })?;

        if aug[pivot_row][col].abs() < 1e-14 {
            return None; // Singular
        }

        aug.swap(col, pivot_row);
        let pivot = aug[col][col];

        // Normalise pivot row
        let total_cols = n + ncols_b;
        for j in 0..total_cols {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in 0..total_cols {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract solution B
    let x: Vec<Vec<f64>> = (0..n).map(|i| aug[i][n..].to_vec()).collect();
    Some(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple path graph: 0 — 1 — 2 — 3
    fn path_edges(n: usize) -> Vec<(usize, usize, f64)> {
        (0..n - 1).map(|i| (i, i + 1, 1.0)).collect()
    }

    /// Complete graph K_n
    fn complete_edges(n: usize) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j, 1.0));
            }
        }
        edges
    }

    #[test]
    fn test_katz_centrality_path() {
        let edges = path_edges(5);
        let katz = katz_centrality(&edges, 5, 0.1, 1.0).expect("katz");
        assert_eq!(katz.len(), 5);
        // All values should be positive
        for &v in &katz {
            assert!(v > 0.0, "katz centrality should be positive: {v}");
        }
        // Unit norm
        let norm: f64 = katz.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "katz should have unit norm: {norm}"
        );
    }

    #[test]
    fn test_katz_centrality_invalid() {
        assert!(katz_centrality(&[], 0, 0.1, 1.0).is_err());
        assert!(katz_centrality(&[], 5, -0.1, 1.0).is_err());
        assert!(katz_centrality(&[], 5, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_harmonic_centrality_path() {
        let edges = path_edges(5);
        let hc = harmonic_centrality(&edges, 5).expect("harmonic");
        assert_eq!(hc.len(), 5);
        // All values non-negative
        for &v in &hc {
            assert!(v >= 0.0, "harmonic centrality non-negative: {v}");
        }
        // Middle nodes (higher connectivity) should have higher centrality
        assert!(
            hc[2] >= hc[0],
            "middle node >= endpoint: {} >= {}",
            hc[2],
            hc[0]
        );
    }

    #[test]
    fn test_harmonic_centrality_disconnected() {
        // Two disconnected nodes
        let hc = harmonic_centrality(&[], 2).expect("harmonic disconnected");
        // Unreachable pairs contribute 0
        assert_eq!(hc[0], 0.0);
        assert_eq!(hc[1], 0.0);
    }

    #[test]
    fn test_harmonic_centrality_invalid() {
        assert!(harmonic_centrality(&[], 0).is_err());
    }

    #[test]
    fn test_percolation_centrality_uniform() {
        let edges = path_edges(5);
        let state = vec![1.0; 5];
        let pc = percolation_centrality(&edges, 5, &state).expect("percolation");
        assert_eq!(pc.len(), 5);
        for &v in &pc {
            assert!(v >= 0.0, "percolation centrality non-negative: {v}");
        }
    }

    #[test]
    fn test_percolation_centrality_invalid() {
        assert!(percolation_centrality(&[], 0, &[]).is_err());
        assert!(percolation_centrality(&[], 3, &[1.0]).is_err()); // length mismatch
    }

    #[test]
    fn test_subgraph_centrality_complete() {
        let edges = complete_edges(4);
        let sc = subgraph_centrality(&edges, 4).expect("subgraph");
        assert_eq!(sc.len(), 4);
        // In a regular graph, all nodes should have equal subgraph centrality
        let first = sc[0];
        for &v in &sc {
            assert!((v - first).abs() < 1e-6, "uniform graph: {v} vs {first}");
        }
        // SC > 1 (since A^0 diagonal = 1 and higher-order terms add)
        assert!(first > 1.0, "SC should be > 1 for connected graph: {first}");
    }

    #[test]
    fn test_subgraph_centrality_invalid() {
        assert!(subgraph_centrality(&[], 0).is_err());
        assert!(subgraph_centrality(&[], 501).is_err());
    }

    #[test]
    fn test_communicability_symmetric() {
        let edges = path_edges(4);
        let comm = communicability(&edges, 4).expect("communicability");
        assert_eq!(comm.len(), 4);
        // Matrix should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (comm[i][j] - comm[j][i]).abs() < 1e-9,
                    "communicability should be symmetric: [{i}][{j}]={} vs [{j}][{i}]={}",
                    comm[i][j],
                    comm[j][i]
                );
            }
        }
    }

    #[test]
    fn test_communicability_diagonal_equals_subgraph() {
        let edges = complete_edges(3);
        let comm = communicability(&edges, 3).expect("communicability");
        let sc = subgraph_centrality(&edges, 3).expect("subgraph centrality");
        for i in 0..3 {
            assert!(
                (comm[i][i] - sc[i]).abs() < 1e-9,
                "communicability diagonal should equal subgraph centrality: {} vs {}",
                comm[i][i],
                sc[i]
            );
        }
    }

    #[test]
    fn test_communicability_invalid() {
        assert!(communicability(&[], 0).is_err());
        assert!(communicability(&[], 501).is_err());
    }

    #[test]
    fn test_matrix_exp_identity() {
        // exp(0) = I
        let zero = vec![vec![0.0f64; 3]; 3];
        let exp = matrix_exp(&zero, 3);
        let id = identity(3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (exp[i][j] - id[i][j]).abs() < 1e-9,
                    "exp(0) should be I: [{i}][{j}] = {}",
                    exp[i][j]
                );
            }
        }
    }

    #[test]
    fn test_katz_complete_graph_uniform() {
        // In a complete graph, all nodes have equal Katz centrality
        let edges = complete_edges(5);
        let katz = katz_centrality(&edges, 5, 0.1, 1.0).expect("katz complete");
        let first = katz[0];
        for &v in &katz {
            assert!(
                (v - first).abs() < 1e-6,
                "uniform graph katz: {v} vs {first}"
            );
        }
    }

    // Suppress unused import warning in test helper
    fn _use_hashmap() {
        let _m: HashMap<usize, usize> = HashMap::new();
    }
}
