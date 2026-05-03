//! Community quality evaluation metrics.
//!
//! This module provides standard metrics to assess the quality of a community
//! partition, both intrinsically (without ground truth) and extrinsically
//! (compared to a reference labelling).
//!
//! ## Intrinsic metrics
//! - [`modularity`]: Newman-Girvan modularity Q
//! - [`conductance`]: Cut fraction relative to minimum volume
//! - [`coverage`]: Fraction of intra-community edges
//! - [`normalized_cut`]: Normalised cut value
//!
//! ## Extrinsic metrics
//! - [`nmi`]: Normalised Mutual Information
//! - [`adjusted_rand_index`]: Adjusted Rand Index

use std::collections::HashMap;

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate edge-list statistics: returns
/// `(total_weight, intra_weight, degree[n], n_comms)`.
fn graph_stats(
    edges: &[(usize, usize, f64)],
    n_nodes: usize,
    communities: &[usize],
) -> (f64, f64, Vec<f64>, usize) {
    let n_comms = communities.iter().max().copied().unwrap_or(0) + 1;
    let mut degree = vec![0.0f64; n_nodes];
    let mut total_w = 0.0f64;
    let mut intra_w = 0.0f64;

    for &(u, v, w) in edges {
        if u >= n_nodes || v >= n_nodes {
            continue;
        }
        degree[u] += w;
        if u != v {
            degree[v] += w;
        }
        total_w += 2.0 * w;
        if communities[u] == communities[v] {
            intra_w += 2.0 * w;
        }
    }
    (total_w, intra_w, degree, n_comms)
}

// ─────────────────────────────────────────────────────────────────────────────
// Modularity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Newman-Girvan modularity Q for a partition of an edge-list graph.
///
/// `Q = 1/(2m) · Σ_{i,j} [A_{ij} − k_i·k_j/(2m)] · δ(c_i, c_j)`
///
/// # Arguments
/// * `edges`       – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`     – Total number of nodes.
/// * `communities` – Community assignment for each node.
pub fn modularity(edges: &[(usize, usize, f64)], n_nodes: usize, communities: &[usize]) -> f64 {
    if n_nodes == 0 || communities.len() != n_nodes {
        return 0.0;
    }
    let (two_m, intra_w, degree, n_comms) = graph_stats(edges, n_nodes, communities);
    if two_m == 0.0 {
        return 0.0;
    }

    // Sum of squared community degrees
    let mut comm_degree = vec![0.0f64; n_comms];
    for i in 0..n_nodes {
        if communities[i] < n_comms {
            comm_degree[communities[i]] += degree[i];
        }
    }
    let sq_sum: f64 = comm_degree.iter().map(|&d| d * d).sum();

    (intra_w / two_m) - (sq_sum / (two_m * two_m))
}

// ─────────────────────────────────────────────────────────────────────────────
// Conductance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the conductance of a single community (set of node indices).
///
/// `φ(S) = cut(S, S̄) / min(vol(S), vol(S̄))`
///
/// where `vol(S) = Σ_{i∈S} k_i` is the volume of the set.
///
/// # Arguments
/// * `edges`     – Weighted edge list `(src, dst, weight)`.
/// * `n_nodes`   – Total number of nodes.
/// * `community` – Set of node indices forming the community.
///
/// # Returns
/// Conductance in `[0, 1]`; returns 1.0 if the community is empty or has zero volume.
pub fn conductance(
    edges: &[(usize, usize, f64)],
    n_nodes: usize,
    community: &[usize],
) -> Result<f64> {
    if community.is_empty() {
        return Ok(1.0);
    }
    let in_community: std::collections::HashSet<usize> = community.iter().cloned().collect();

    let mut degree = vec![0.0f64; n_nodes];
    let mut cut = 0.0f64;

    for &(u, v, w) in edges {
        if u >= n_nodes || v >= n_nodes {
            continue;
        }
        degree[u] += w;
        if u != v {
            degree[v] += w;
        }
        // Undirected: count both u→v and v→u
        let u_in = in_community.contains(&u);
        let v_in = in_community.contains(&v);
        if u_in != v_in {
            cut += w; // count once for undirected
        }
    }

    let vol_s: f64 = community
        .iter()
        .filter(|&&n| n < n_nodes)
        .map(|&n| degree[n])
        .sum();
    let vol_total: f64 = degree.iter().sum();
    let vol_s_bar = vol_total - vol_s;

    let min_vol = vol_s.min(vol_s_bar);
    if min_vol == 0.0 {
        return Ok(1.0);
    }
    Ok(cut / min_vol)
}

// ─────────────────────────────────────────────────────────────────────────────
// Coverage
// ─────────────────────────────────────────────────────────────────────────────

/// Compute coverage: the fraction of edge weight inside communities.
///
/// `coverage = Σ_{intra} w_{ij} / Σ_{all} w_{ij}`
///
/// # Arguments
/// * `edges`       – Weighted edge list.
/// * `n_nodes`     – Total number of nodes.
/// * `communities` – Community assignment for each node.
pub fn coverage(edges: &[(usize, usize, f64)], n_nodes: usize, communities: &[usize]) -> f64 {
    if edges.is_empty() || communities.len() != n_nodes {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut intra = 0.0f64;
    for &(u, v, w) in edges {
        if u >= n_nodes || v >= n_nodes {
            continue;
        }
        total += w;
        if communities[u] == communities[v] {
            intra += w;
        }
    }
    if total == 0.0 {
        0.0
    } else {
        intra / total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalised cut
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the normalised cut value for a partition.
///
/// `NCut(M) = Σ_i  cut(i, Ī) / vol(i)`
///
/// where `cut(i, Ī)` is the weight of edges leaving community `i` and
/// `vol(i)` is the sum of degrees of all nodes in community `i`.
///
/// # Arguments
/// * `edges`       – Weighted edge list.
/// * `n_nodes`     – Total number of nodes.
/// * `communities` – Community assignment for each node.
pub fn normalized_cut(edges: &[(usize, usize, f64)], n_nodes: usize, communities: &[usize]) -> f64 {
    if communities.len() != n_nodes {
        return 0.0;
    }
    let n_comms = communities.iter().max().copied().unwrap_or(0) + 1;

    let mut degree = vec![0.0f64; n_nodes];
    for &(u, v, w) in edges {
        if u < n_nodes && v < n_nodes {
            degree[u] += w;
            if u != v {
                degree[v] += w;
            }
        }
    }

    let mut vol_comm = vec![0.0f64; n_comms];
    for i in 0..n_nodes {
        if communities[i] < n_comms {
            vol_comm[communities[i]] += degree[i];
        }
    }

    let mut cut_comm = vec![0.0f64; n_comms];
    for &(u, v, w) in edges {
        if u >= n_nodes || v >= n_nodes {
            continue;
        }
        if communities[u] != communities[v] {
            cut_comm[communities[u]] += w;
            cut_comm[communities[v]] += w;
        }
    }

    let ncut: f64 = (0..n_comms)
        .filter(|&c| vol_comm[c] > 0.0)
        .map(|c| cut_comm[c] / vol_comm[c])
        .sum();
    ncut
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalised Mutual Information
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Normalised Mutual Information (NMI) between two label vectors.
///
/// `NMI(X, Y) = 2 · I(X;Y) / [H(X) + H(Y)]`
///
/// Returns a value in `[0, 1]` where 1.0 indicates perfect agreement.
///
/// # Arguments
/// * `true_labels` – Ground-truth community labels.
/// * `pred_labels` – Predicted community labels.
pub fn nmi(true_labels: &[usize], pred_labels: &[usize]) -> Result<f64> {
    let n = true_labels.len();
    if n != pred_labels.len() {
        return Err(GraphError::InvalidParameter {
            param: "pred_labels".into(),
            value: format!("len={}", pred_labels.len()),
            expected: format!("len={n}"),
            context: "nmi".into(),
        });
    }
    if n == 0 {
        return Ok(1.0);
    }

    let fn64 = n as f64;

    // Count contingency table
    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut true_counts: HashMap<usize, usize> = HashMap::new();
    let mut pred_counts: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        *contingency
            .entry((true_labels[i], pred_labels[i]))
            .or_insert(0) += 1;
        *true_counts.entry(true_labels[i]).or_insert(0) += 1;
        *pred_counts.entry(pred_labels[i]).or_insert(0) += 1;
    }

    // Mutual information
    let mi: f64 = contingency
        .iter()
        .map(|(&(t, p), &cnt)| {
            let n_tp = cnt as f64;
            let n_t = *true_counts.get(&t).unwrap_or(&1) as f64;
            let n_p = *pred_counts.get(&p).unwrap_or(&1) as f64;
            if n_tp > 0.0 {
                n_tp / fn64 * (n_tp * fn64 / (n_t * n_p)).ln()
            } else {
                0.0
            }
        })
        .sum();

    // Entropies
    let h_true: f64 = true_counts
        .values()
        .map(|&c| {
            let p = c as f64 / fn64;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let h_pred: f64 = pred_counts
        .values()
        .map(|&c| {
            let p = c as f64 / fn64;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let denom = h_true + h_pred;
    if denom == 0.0 {
        Ok(1.0)
    } else {
        Ok((2.0 * mi / denom).clamp(0.0, 1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adjusted Rand Index
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Adjusted Rand Index (ARI) between two label vectors.
///
/// `ARI = (RI - E[RI]) / (max(RI) - E[RI])`
///
/// Returns a value in `[-1, 1]` where 1.0 is perfect, 0.0 is random.
///
/// # Arguments
/// * `true_labels` – Ground-truth community labels.
/// * `pred_labels` – Predicted community labels.
pub fn adjusted_rand_index(true_labels: &[usize], pred_labels: &[usize]) -> Result<f64> {
    let n = true_labels.len();
    if n != pred_labels.len() {
        return Err(GraphError::InvalidParameter {
            param: "pred_labels".into(),
            value: format!("len={}", pred_labels.len()),
            expected: format!("len={n}"),
            context: "adjusted_rand_index".into(),
        });
    }
    if n == 0 {
        return Ok(1.0);
    }

    // Build contingency table
    let n_true = true_labels.iter().max().copied().unwrap_or(0) + 1;
    let n_pred = pred_labels.iter().max().copied().unwrap_or(0) + 1;
    let mut contingency = vec![vec![0u64; n_pred]; n_true];
    for i in 0..n {
        let t = true_labels[i];
        let p = pred_labels[i];
        if t < n_true && p < n_pred {
            contingency[t][p] += 1;
        }
    }

    // Row sums, column sums
    let a: Vec<u64> = (0..n_true).map(|i| contingency[i].iter().sum()).collect();
    let b: Vec<u64> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| contingency[i][j]).sum())
        .collect();

    // C(n_{ij}, 2), C(a_i, 2), C(b_j, 2)
    let comb2 = |x: u64| -> f64 { (x * x.saturating_sub(1)) as f64 / 2.0 };

    let sum_comb_c: f64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| comb2(x))
        .sum();
    let sum_comb_a: f64 = a.iter().map(|&x| comb2(x)).sum();
    let sum_comb_b: f64 = b.iter().map(|&x| comb2(x)).sum();
    let comb_n = comb2(n as u64);

    if comb_n == 0.0 {
        // Only one element
        return Ok(1.0);
    }

    let expected = sum_comb_a * sum_comb_b / comb_n;
    let max_val = (sum_comb_a + sum_comb_b) / 2.0;
    let denom = max_val - expected;
    if denom.abs() < 1e-15 {
        // Perfect agreement
        return Ok(1.0);
    }

    Ok(((sum_comb_c - expected) / denom).clamp(-1.0, 1.0))
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
    fn test_modularity_perfect() {
        let (edges, n) = two_clique_edges(4);
        let perfect: Vec<usize> = (0..8).map(|i| if i < 4 { 0 } else { 1 }).collect();
        let q = modularity(&edges, n, &perfect);
        assert!(q > 0.0, "modularity should be positive: {q}");
    }

    #[test]
    fn test_modularity_single_community() {
        let (edges, n) = two_clique_edges(3);
        let single = vec![0usize; 6];
        let q = modularity(&edges, n, &single);
        assert!(
            q <= 0.0 + 1e-10,
            "single-community modularity should be ≤ 0: {q}"
        );
    }

    #[test]
    fn test_conductance_perfect_cut() {
        let (edges, n) = two_clique_edges(4);
        let community: Vec<usize> = (0..4).collect();
        let phi = conductance(&edges, n, &community).expect("conductance");
        // The only inter-community edge is the weak bridge (weight 0.05)
        assert!(phi < 1.0, "conductance should be < 1: {phi}");
        assert!(phi >= 0.0);
    }

    #[test]
    fn test_conductance_empty_community() {
        let phi = conductance(&[], 4, &[]).expect("conductance empty");
        assert_eq!(phi, 1.0);
    }

    #[test]
    fn test_coverage_perfect_partition() {
        let (edges, n) = two_clique_edges(4);
        let perfect: Vec<usize> = (0..8).map(|i| if i < 4 { 0 } else { 1 }).collect();
        let cov = coverage(&edges, n, &perfect);
        // Almost all weight is intra-community
        assert!(cov > 0.9, "coverage should be high: {cov}");
    }

    #[test]
    fn test_coverage_single_community() {
        let (edges, n) = two_clique_edges(3);
        let single = vec![0usize; 6];
        let cov = coverage(&edges, n, &single);
        assert!(
            (cov - 1.0).abs() < 1e-9,
            "single community coverage = 1: {cov}"
        );
    }

    #[test]
    fn test_normalized_cut_perfect_partition() {
        let (edges, n) = two_clique_edges(4);
        let perfect: Vec<usize> = (0..8).map(|i| if i < 4 { 0 } else { 1 }).collect();
        let ncut = normalized_cut(&edges, n, &perfect);
        assert!(ncut >= 0.0);
        assert!(
            ncut < 1.0,
            "normalized cut for near-perfect partition should be small: {ncut}"
        );
    }

    #[test]
    fn test_nmi_perfect_agreement() {
        let labels = vec![0, 0, 1, 1, 2, 2];
        let nmi_val = nmi(&labels, &labels).expect("nmi perfect");
        assert!(
            (nmi_val - 1.0).abs() < 1e-9,
            "NMI perfect agreement = 1: {nmi_val}"
        );
    }

    #[test]
    fn test_nmi_length_mismatch() {
        assert!(nmi(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn test_nmi_empty() {
        let v: Vec<usize> = vec![];
        let val = nmi(&v, &v).expect("nmi empty");
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_ari_perfect_agreement() {
        let labels = vec![0, 0, 0, 1, 1, 1];
        let ari = adjusted_rand_index(&labels, &labels).expect("ari perfect");
        assert!((ari - 1.0).abs() < 1e-9, "ARI perfect = 1: {ari}");
    }

    #[test]
    fn test_ari_random_labels() {
        // With random assignments, ARI should be close to 0
        let true_l = vec![0, 0, 1, 1, 2, 2];
        let pred_l = vec![0, 1, 2, 0, 1, 2]; // random-looking assignment
        let ari = adjusted_rand_index(&true_l, &pred_l).expect("ari random");
        assert!(
            ari < 0.5,
            "ARI for dissimilar partitions should be small: {ari}"
        );
    }

    #[test]
    fn test_ari_length_mismatch() {
        assert!(adjusted_rand_index(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn test_ari_empty() {
        let v: Vec<usize> = vec![];
        let ari = adjusted_rand_index(&v, &v).expect("ari empty");
        assert_eq!(ari, 1.0);
    }
}
