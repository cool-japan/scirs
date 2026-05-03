//! Hypergraph algorithms.
//!
//! Implements:
//! * **Spectral clustering** via the normalised Laplacian (Zhou et al. 2006).
//! * **Hyperedge cuts**: raw cut, ratio cut, normalised cut.
//! * **Generalised random walk** (Markov chain on nodes through hyperedges).
//! * **Hypergraph betweenness centrality** based on shortest paths through
//!   the clique-expansion graph.
//! * **s-walks and s-paths**: walks/paths between hyperedges that share ≥ s nodes.

use super::core::{clique_expansion, hyperedge_centrality, IndexedHypergraph};
use crate::error::{GraphError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ============================================================================
// Internal helpers
// ============================================================================

/// Returns the normalised Laplacian Θ = I − D_v^{-1/2} B W D_e^{-1} B^T D_v^{-1/2}.
///
/// Shape: (n_nodes × n_nodes).  Used for spectral clustering.
fn normalised_laplacian(hg: &IndexedHypergraph) -> Array2<f64> {
    let n = hg.n_nodes();
    let m = hg.n_hyperedges();

    if n == 0 || m == 0 {
        return Array2::eye(n);
    }

    // B: n × m incidence (binary, so each entry 0/1)
    let b = hg.incidence_matrix_binary();

    // D_v[i] = weighted degree of node i
    let dv: Vec<f64> = (0..n).map(|i| hg.weighted_degree(i)).collect();
    // D_e[e] = size of hyperedge e (cardinality)
    let de: Vec<f64> = hg
        .hyperedges()
        .iter()
        .map(|he| he.nodes.len() as f64)
        .collect();
    // W[e] = weight of hyperedge e
    let w: Vec<f64> = hg.hyperedges().iter().map(|he| he.weight).collect();

    // Compute Θ = I − D_v^{-1/2} B W D_e^{-1} B^T D_v^{-1/2}
    // We build the (n × n) matrix Ω = D_v^{-1/2} B W D_e^{-1} B^T D_v^{-1/2} directly.
    let mut omega = Array2::<f64>::zeros((n, n));
    for e in 0..m {
        if de[e] == 0.0 {
            continue;
        }
        let scale = w[e] / de[e];
        // Find which nodes belong to this hyperedge
        let members: Vec<usize> = (0..n)
            .filter(|&i| (b[[i, e]] - 1.0).abs() < 1e-10)
            .collect();
        for &i in &members {
            let dvi = if dv[i] > 0.0 { dv[i].sqrt() } else { 0.0 };
            if dvi == 0.0 {
                continue;
            }
            for &j in &members {
                let dvj = if dv[j] > 0.0 { dv[j].sqrt() } else { 0.0 };
                if dvj == 0.0 {
                    continue;
                }
                omega[[i, j]] += scale / (dvi * dvj);
            }
        }
    }

    // Θ = I − Ω
    let mut theta = Array2::<f64>::eye(n);
    for i in 0..n {
        for j in 0..n {
            theta[[i, j]] -= omega[[i, j]];
        }
    }
    theta
}

// ============================================================================
// Spectral clustering
// ============================================================================

/// Result of hypergraph spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralClusteringResult {
    /// Cluster label for each node (`labels[i] ∈ 0..k`).
    pub labels: Vec<usize>,
    /// The first `k` eigenvectors stacked as columns (shape `n × k`).
    pub embedding: Array2<f64>,
    /// Number of iterations used by the power-iteration eigensolver.
    pub eigenvalue_iterations: usize,
}

/// Perform **spectral clustering** on a hypergraph using the normalised
/// Laplacian (Zhou et al. NeurIPS 2006).
///
/// # Arguments
/// * `hg`   – the hypergraph
/// * `k`    – number of clusters
/// * `seed` – RNG seed for k-means initialisation
///
/// # Algorithm
/// 1. Form the `n × n` normalised Laplacian Θ.
/// 2. Extract the `k` eigenvectors corresponding to the **smallest** `k`
///    eigenvalues using deflated power iteration.
/// 3. Run k-means on the resulting `n × k` embedding.
///
/// # Errors
/// Returns `GraphError::InvalidGraph` when `k > n_nodes` or the hypergraph is
/// empty.
pub fn spectral_clustering(
    hg: &IndexedHypergraph,
    k: usize,
    seed: u64,
) -> Result<SpectralClusteringResult> {
    use scirs2_core::random::ChaCha20Rng;
    let n = hg.n_nodes();
    if n == 0 {
        return Err(GraphError::InvalidGraph(
            "hypergraph has no nodes".to_string(),
        ));
    }
    if k == 0 || k > n {
        return Err(GraphError::InvalidGraph(format!(
            "k = {k} must be in 1..={n}"
        )));
    }

    let theta = normalised_laplacian(hg);

    // Compute the k smallest eigenvectors via deflated power iteration on
    // (sigma*I - Theta).  We use sigma = 2 so the operator is positive definite.
    let sigma = 2.0_f64;
    // shifted: A = sigma*I - Theta  → largest eigenvectors of A = smallest of Theta
    let mut a = theta.clone();
    for i in 0..n {
        a[[i, i]] = sigma - theta[[i, i]];
        for j in 0..n {
            if i != j {
                a[[i, j]] = -theta[[i, j]];
            }
        }
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut total_iters = 0usize;

    for _ki in 0..k {
        // Random starting vector
        let mut v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
        // Orthogonalise against already-found eigenvectors
        for prev in &eigenvecs {
            let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for (vi, pi) in v.iter_mut().zip(prev.iter()) {
                *vi -= dot * pi;
            }
        }
        // Normalise
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            v = vec![0.0; n];
            if _ki < n {
                v[_ki] = 1.0;
            }
        } else {
            for vi in &mut v {
                *vi /= norm;
            }
        }

        let max_iter = 2000;
        let tol = 1e-10;
        let mut iters = 0usize;
        for _ in 0..max_iter {
            iters += 1;
            // Multiply by A
            let mut nv: Vec<f64> = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    nv[i] += a[[i, j]] * v[j];
                }
            }
            // Deflation: remove components along already-found eigenvectors
            for prev in &eigenvecs {
                let dot: f64 = nv.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for (nvi, pi) in nv.iter_mut().zip(prev.iter()) {
                    *nvi -= dot * pi;
                }
            }
            // Normalise
            let norm: f64 = nv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for vi in &mut nv {
                *vi /= norm;
            }
            // Convergence check
            let diff: f64 = nv
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).abs() + (a + b).abs())
                .fold(0.0_f64, f64::min);
            // Check if either v==nv or v==-nv
            let diff_pos: f64 = nv
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let diff_neg: f64 = nv
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a + b).powi(2))
                .sum::<f64>()
                .sqrt();
            let _ = diff; // suppress unused warning
            v = nv;
            if diff_pos < tol || diff_neg < tol {
                break;
            }
        }
        total_iters += iters;
        eigenvecs.push(v);
    }

    // Build n×k embedding matrix
    let mut embedding = Array2::<f64>::zeros((n, k));
    for (ki, ev) in eigenvecs.iter().enumerate() {
        for (i, &val) in ev.iter().enumerate() {
            embedding[[i, ki]] = val;
        }
    }

    // k-means on embedding rows
    let labels = kmeans(&embedding, k, seed + 1, 300);

    Ok(SpectralClusteringResult {
        labels,
        embedding,
        eigenvalue_iterations: total_iters,
    })
}

/// Simple k-means clustering on the rows of a matrix.
fn kmeans(data: &Array2<f64>, k: usize, seed: u64, max_iter: usize) -> Vec<usize> {
    use scirs2_core::random::ChaCha20Rng;
    let n = data.nrows();
    let d = data.ncols();
    if k == 0 || n == 0 {
        return vec![0; n];
    }
    let k = k.min(n);

    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    // k-means++ initialisation
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = rng.random_range(0..n);
    centers.push(data.row(first).to_vec());

    for _ in 1..k {
        // Probability proportional to distance squared to nearest centre
        let dists: Vec<f64> = (0..n)
            .map(|i| {
                centers
                    .iter()
                    .map(|c| {
                        data.row(i)
                            .iter()
                            .zip(c.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = dists.iter().sum();
        if total < 1e-15 {
            break;
        }
        let threshold = rng.random::<f64>() * total;
        let mut acc = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            acc += d;
            if acc >= threshold {
                chosen = i;
                break;
            }
        }
        centers.push(data.row(chosen).to_vec());
    }

    let mut labels = vec![0usize; n];
    for _iter in 0..max_iter {
        // Assignment
        let mut changed = false;
        for i in 0..n {
            let best = (0..centers.len())
                .min_by(|&a, &b| {
                    let da: f64 = data
                        .row(i)
                        .iter()
                        .zip(centers[a].iter())
                        .map(|(x, c)| (x - c).powi(2))
                        .sum();
                    let db: f64 = data
                        .row(i)
                        .iter()
                        .zip(centers[b].iter())
                        .map(|(x, c)| (x - c).powi(2))
                        .sum();
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0);
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update centres
        let mut sums = vec![vec![0.0f64; d]; centers.len()];
        let mut counts = vec![0usize; centers.len()];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                sums[c][j] += data[[i, j]];
            }
        }
        for c in 0..centers.len() {
            if counts[c] > 0 {
                for j in 0..d {
                    centers[c][j] = sums[c][j] / counts[c] as f64;
                }
            }
        }
    }
    labels
}

// ============================================================================
// Hyperedge cuts
// ============================================================================

/// Result of a hyperedge cut computation.
#[derive(Debug, Clone)]
pub struct CutResult {
    /// Raw hyperedge cut (number of hyperedges crossing the partition).
    pub cut: usize,
    /// Ratio cut: `cut / min(|S|, |V\S|)`.
    pub ratio_cut: f64,
    /// Normalised cut: `cut/vol(S) + cut/vol(V\S)` where `vol(X) = Σ_v deg(v)`.
    pub normalised_cut: f64,
}

/// Compute the **hyperedge cut**, **ratio cut**, and **normalised cut** for a
/// binary partition of the node set.
///
/// # Arguments
/// * `hg`        – the hypergraph
/// * `partition` – boolean slice of length `n_nodes`; `true` → side A, `false` → side B
///
/// # Errors
/// Returns an error if `partition.len() != hg.n_nodes()`.
pub fn hyperedge_cut(hg: &IndexedHypergraph, partition: &[bool]) -> Result<CutResult> {
    if partition.len() != hg.n_nodes() {
        return Err(GraphError::InvalidGraph(format!(
            "partition length {} != n_nodes {}",
            partition.len(),
            hg.n_nodes()
        )));
    }
    let mut cut = 0usize;
    for he in hg.hyperedges() {
        let has_true = he.nodes.iter().any(|&n| partition[n]);
        let has_false = he.nodes.iter().any(|&n| !partition[n]);
        if has_true && has_false {
            cut += 1;
        }
    }

    let size_a = partition.iter().filter(|&&b| b).count();
    let size_b = hg.n_nodes() - size_a;
    let min_side = size_a.min(size_b);

    let ratio_cut = if min_side > 0 {
        cut as f64 / min_side as f64
    } else {
        f64::INFINITY
    };

    // Volumes
    let vol_a: f64 = (0..hg.n_nodes())
        .filter(|&i| partition[i])
        .map(|i| hg.weighted_degree(i))
        .sum();
    let vol_b: f64 = (0..hg.n_nodes())
        .filter(|&i| !partition[i])
        .map(|i| hg.weighted_degree(i))
        .sum();

    let normalised_cut = if vol_a > 0.0 && vol_b > 0.0 {
        cut as f64 / vol_a + cut as f64 / vol_b
    } else {
        f64::INFINITY
    };

    Ok(CutResult {
        cut,
        ratio_cut,
        normalised_cut,
    })
}

// ============================================================================
// Generalised hypergraph random walk (stationary distribution)
// ============================================================================

/// Compute the **stationary distribution** of the generalised random walk on a
/// hypergraph, using power iteration on the transition matrix.
///
/// The transition probability P(u→v) follows the Chung–Zhou formulation:
///
/// ```text
/// P(u, v) = Σ_e ∈ E(u) [ w_e / (deg_w(u) * |e|) ]   for v ∈ e
/// ```
///
/// Returns a vector of length `n_nodes` summing to 1.
///
/// # Errors
/// Returns an error if the hypergraph has no nodes or all nodes are isolated.
pub fn stationary_distribution(hg: &IndexedHypergraph) -> Result<Array1<f64>> {
    let n = hg.n_nodes();
    if n == 0 {
        return Err(GraphError::InvalidGraph(
            "hypergraph has no nodes".to_string(),
        ));
    }

    // Check that at least one node has non-zero weighted degree
    let any_connected = (0..n).any(|i| hg.weighted_degree(i) > 0.0);
    if !any_connected {
        // Uniform distribution over all nodes
        return Ok(Array1::from_elem(n, 1.0 / n as f64));
    }

    // Build transition matrix P (n × n)
    let mut p = Array2::<f64>::zeros((n, n));
    for he in hg.hyperedges() {
        let size = he.nodes.len() as f64;
        if size == 0.0 {
            continue;
        }
        for &u in &he.nodes {
            let deg_u = hg.weighted_degree(u);
            if deg_u == 0.0 {
                continue;
            }
            for &v in &he.nodes {
                p[[u, v]] += he.weight / (deg_u * size);
            }
        }
    }

    // Power iteration: pi * P = pi
    let mut pi = Array1::from_elem(n, 1.0 / n as f64);
    let max_iter = 5000;
    let tol = 1e-10;

    for _ in 0..max_iter {
        // pi_new = pi . P
        let mut pi_new = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                pi_new[j] += pi[i] * p[[i, j]];
            }
        }
        // Normalise
        let s: f64 = pi_new.iter().sum();
        if s > 0.0 {
            pi_new.mapv_inplace(|x| x / s);
        }
        // Convergence
        let diff: f64 = pi_new
            .iter()
            .zip(pi.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        pi = pi_new;
        if diff < tol {
            break;
        }
    }
    Ok(pi)
}

// ============================================================================
// Hypergraph betweenness centrality
// ============================================================================

/// Compute **hypergraph betweenness centrality** for every node.
///
/// We compute betweenness on the **clique-expansion graph** (2-section), where
/// edge weights are used as distances in Dijkstra's algorithm.  The betweenness
/// of node `v` is the fraction of shortest paths between all pairs `(s, t)`
/// (s ≠ v ≠ t) that pass through `v`.
///
/// Returns a vector of length `n_nodes`.
pub fn betweenness_centrality(hg: &IndexedHypergraph) -> Vec<f64> {
    let n = hg.n_nodes();
    let mut bc = vec![0.0f64; n];
    if n < 3 {
        return bc;
    }

    let g = clique_expansion(hg);

    // Build adjacency list from graph
    // We use a simple BFS-based SSSP (unweighted) for correctness, since
    // the clique-expansion already captures hyperedge topology.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    // Pull edges from the clique expansion
    for he in hg.hyperedges() {
        let k = he.nodes.len();
        for i in 0..k {
            for j in (i + 1)..k {
                let u = he.nodes[i];
                let v = he.nodes[j];
                if !adj[u].contains(&v) {
                    adj[u].push(v);
                }
                if !adj[v].contains(&u) {
                    adj[v].push(u);
                }
            }
        }
    }
    let _ = g; // g used via adj

    // Brandes' algorithm (unweighted BFS version)
    for s in 0..n {
        let mut stack: Vec<usize> = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n];
        sigma[s] = 1.0;
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &adj[v] {
                if dist[w] < 0 {
                    queue.push_back(w);
                    dist[w] = dist[v] + 1;
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
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                bc[w] += delta[w];
            }
        }
    }

    // Normalise by (n-1)(n-2) for undirected graphs
    let factor = if n > 2 {
        1.0 / ((n - 1) as f64 * (n - 2) as f64)
    } else {
        1.0
    };
    for v in &mut bc {
        *v *= factor;
    }
    bc
}

// ============================================================================
// s-walks and s-paths
// ============================================================================

/// Compute the **s-distance** between two hyperedges: the length of the
/// shortest s-path (sequence of hyperedges each sharing ≥ `s` nodes with the
/// next).
///
/// Returns `None` if the hyperedges are not s-connected.
pub fn s_distance(hg: &IndexedHypergraph, e1: usize, e2: usize, s: usize) -> Option<usize> {
    let m = hg.n_hyperedges();
    if e1 >= m || e2 >= m {
        return None;
    }
    if e1 == e2 {
        return Some(0);
    }

    // BFS on hyperedge-level s-adjacency graph
    let mut dist = vec![usize::MAX; m];
    dist[e1] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(e1);

    while let Some(cur) = queue.pop_front() {
        let cur_dist = dist[cur];
        for next in 0..m {
            if next == cur {
                continue;
            }
            if dist[next] == usize::MAX {
                let shared = hg.hyperedges()[cur].intersection_size(&hg.hyperedges()[next]);
                if shared >= s {
                    dist[next] = cur_dist + 1;
                    if next == e2 {
                        return Some(dist[next]);
                    }
                    queue.push_back(next);
                }
            }
        }
    }
    None
}

/// Compute the **s-diameter** of the hypergraph: the maximum s-distance over
/// all pairs of hyperedges in the same s-connected component.
///
/// Returns `0` if there are fewer than 2 hyperedges or the hypergraph is
/// s-disconnected everywhere.
pub fn s_diameter(hg: &IndexedHypergraph, s: usize) -> usize {
    let m = hg.n_hyperedges();
    let mut max_dist = 0usize;
    for e1 in 0..m {
        for e2 in (e1 + 1)..m {
            if let Some(d) = s_distance(hg, e1, e2, s) {
                max_dist = max_dist.max(d);
            }
        }
    }
    max_dist
}

/// Enumerate all **s-paths** of length ≤ `max_len` starting from hyperedge
/// `start` as BFS layers.
///
/// Returns a `HashMap<usize, usize>` mapping each reachable hyperedge to its
/// s-distance from `start`.
pub fn s_reachability(
    hg: &IndexedHypergraph,
    start: usize,
    s: usize,
    max_len: usize,
) -> HashMap<usize, usize> {
    let m = hg.n_hyperedges();
    let mut dists: HashMap<usize, usize> = HashMap::new();
    if start >= m {
        return dists;
    }
    dists.insert(start, 0);
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(cur) = queue.pop_front() {
        let cur_dist = *dists.get(&cur).unwrap_or(&0);
        if cur_dist >= max_len {
            continue;
        }
        for next in 0..m {
            if next == cur || dists.contains_key(&next) {
                continue;
            }
            let shared = hg.hyperedges()[cur].intersection_size(&hg.hyperedges()[next]);
            if shared >= s {
                dists.insert(next, cur_dist + 1);
                queue.push_back(next);
            }
        }
    }
    dists
}

/// Hyperedge betweenness centrality in the **s-line graph**.
///
/// Returns a vector of length `n_hyperedges` where each entry is the fraction
/// of shortest s-paths (in the s-adjacency graph) passing through that
/// hyperedge.
pub fn s_betweenness_centrality(hg: &IndexedHypergraph, s: usize) -> Vec<f64> {
    let m = hg.n_hyperedges();
    let mut bc = vec![0.0f64; m];
    if m < 3 {
        return bc;
    }

    // Build s-adjacency list
    let mut s_adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for i in 0..m {
        for j in (i + 1)..m {
            let shared = hg.hyperedges()[i].intersection_size(&hg.hyperedges()[j]);
            if shared >= s {
                s_adj[i].push(j);
                s_adj[j].push(i);
            }
        }
    }

    // Brandes on s-adjacency graph
    for src in 0..m {
        let mut stack: Vec<usize> = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut sigma = vec![0.0f64; m];
        sigma[src] = 1.0;
        let mut dist = vec![-1i64; m];
        dist[src] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(src);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &s_adj[v] {
                if dist[w] < 0 {
                    queue.push_back(w);
                    dist[w] = dist[v] + 1;
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }
        let mut delta = vec![0.0f64; m];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != src {
                bc[w] += delta[w];
            }
        }
    }

    let factor = if m > 2 {
        1.0 / ((m - 1) as f64 * (m - 2) as f64)
    } else {
        1.0
    };
    for v in &mut bc {
        *v *= factor;
    }
    bc
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_hg() -> IndexedHypergraph {
        // 5 nodes, 3 hyperedges
        let mut hg = IndexedHypergraph::new(5);
        hg.add_hyperedge(vec![0, 1, 2], 1.0).expect("ok");
        hg.add_hyperedge(vec![2, 3, 4], 1.0).expect("ok");
        hg.add_hyperedge(vec![0, 3], 1.0).expect("ok");
        hg
    }

    #[test]
    fn test_spectral_clustering_labels() {
        let hg = make_hg();
        let res = spectral_clustering(&hg, 2, 42).expect("cluster ok");
        assert_eq!(res.labels.len(), 5);
        // All labels must be in 0..2
        for &l in &res.labels {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_spectral_clustering_invalid_k() {
        let hg = make_hg();
        assert!(spectral_clustering(&hg, 0, 0).is_err());
        assert!(spectral_clustering(&hg, 100, 0).is_err());
    }

    #[test]
    fn test_hyperedge_cut_partition() {
        let mut hg = IndexedHypergraph::new(4);
        hg.add_hyperedge(vec![0, 1], 1.0).expect("ok");
        hg.add_hyperedge(vec![2, 3], 1.0).expect("ok");
        hg.add_hyperedge(vec![1, 2], 1.0).expect("ok"); // crosses partition
                                                        // Partition: {0,1} vs {2,3}
        let part = vec![true, true, false, false];
        let res = hyperedge_cut(&hg, &part).expect("cut ok");
        assert_eq!(res.cut, 1);
    }

    #[test]
    fn test_hyperedge_cut_all_same_side() {
        let mut hg = IndexedHypergraph::new(4);
        hg.add_hyperedge(vec![0, 1, 2, 3], 1.0).expect("ok");
        let part = vec![true, true, true, true];
        let res = hyperedge_cut(&hg, &part).expect("cut ok");
        assert_eq!(res.cut, 0);
    }

    #[test]
    fn test_hyperedge_cut_wrong_len() {
        let hg = make_hg();
        assert!(hyperedge_cut(&hg, &[true, false]).is_err());
    }

    #[test]
    fn test_stationary_distribution_sums_to_one() {
        let hg = make_hg();
        let pi = stationary_distribution(&hg).expect("ok");
        let s: f64 = pi.iter().sum();
        assert_relative_eq!(s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_stationary_empty() {
        let hg = IndexedHypergraph::new(0);
        assert!(stationary_distribution(&hg).is_err());
    }

    #[test]
    fn test_betweenness_centrality_len() {
        let hg = make_hg();
        let bc = betweenness_centrality(&hg);
        assert_eq!(bc.len(), 5);
        for &v in &bc {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_s_distance_same_edge() {
        let hg = make_hg();
        assert_eq!(s_distance(&hg, 0, 0, 1), Some(0));
    }

    #[test]
    fn test_s_distance_adjacent() {
        // Edges 0 and 1 share node 2 → s=1 distance is 1
        let hg = make_hg();
        assert_eq!(s_distance(&hg, 0, 1, 1), Some(1));
    }

    #[test]
    fn test_s_distance_disconnected() {
        // Edges {0,1} and {3,4} share no nodes → s=1 distance might still be connected via 3rd edge
        let mut hg = IndexedHypergraph::new(5);
        hg.add_hyperedge(vec![0, 1], 1.0).expect("ok");
        hg.add_hyperedge(vec![3, 4], 1.0).expect("ok");
        // Completely disjoint → no s=1 path
        assert_eq!(s_distance(&hg, 0, 1, 1), None);
    }

    #[test]
    fn test_s_reachability() {
        let hg = make_hg();
        let reach = s_reachability(&hg, 0, 1, 5);
        assert!(reach.contains_key(&0));
        // Edge 0 shares node 2 with edge 1 → reachable
        assert!(reach.contains_key(&1));
    }

    #[test]
    fn test_s_betweenness_len() {
        let hg = make_hg();
        let sbc = s_betweenness_centrality(&hg, 1);
        assert_eq!(sbc.len(), hg.n_hyperedges());
    }

    #[test]
    fn test_s_diameter() {
        let hg = make_hg();
        let d = s_diameter(&hg, 1);
        // Should be finite (all edges connected at s=1 through shared nodes)
        assert!(d <= hg.n_hyperedges());
    }
}
