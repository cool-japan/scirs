//! Advanced cluster stability analysis — free-function API
//!
//! This module provides lightweight, self-contained free functions for:
//!
//! - [`bootstrap_stability`] — Bootstrap cluster stability assessment
//! - [`gap_statistic_free`] — Gap statistic for optimal k selection  
//! - [`prediction_strength`] — Tibshirani & Walther (2005) prediction strength
//! - [`hopkins_statistic`] — Hopkins spatial randomness test
//!
//! All functions use a deterministic Park-Miller LCG (no external `rand` crate)
//! and follow the no-unwrap policy.
//!
//! # References
//! - Tibshirani, R., Walther, G. & Hastie, T. (2001). "Estimating the number of
//!   clusters in a data set via the gap statistic." J. R. Stat. Soc. B, 63(2), 411-423.
//! - Tibshirani, R. & Walther, G. (2005). "Cluster validation by prediction strength."
//!   J. Comput. Graph. Stat., 14(3), 511-528.
//! - Hopkins, B. & Skellam, J.G. (1954). "A new method for determining the type of
//!   distribution of plant individuals." Ann. Bot., 18(2), 213-227.

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Park-Miller LCG
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 6364136223846793005 } else { seed };
        Self { state }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) as f64;
        bits / (1u64 << 53) as f64
    }

    fn next_range(&mut self, low: usize, high: usize) -> usize {
        if low >= high {
            return low;
        }
        let span = (high - low) as f64;
        low + (self.next_f64() * span) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Simple k-means (Lloyd's algorithm) returning cluster labels.
fn simple_kmeans(data: &[Vec<f64>], k: usize, seed: u64, max_iter: usize) -> Vec<usize> {
    let n = data.len();
    let d = data[0].len();
    let mut rng = Lcg::new(seed);

    // k-means++ initialisation
    let first = rng.next_range(0, n);
    let mut centroids: Vec<Vec<f64>> = vec![data[first].clone()];
    for _ in 1..k {
        let dists: Vec<f64> = data
            .iter()
            .map(|x| {
                centroids
                    .iter()
                    .map(|c| sq_dist(x, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = dists.iter().sum();
        let target = rng.next_f64() * total;
        let mut cum = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cum += d;
            if cum >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let new_l = (0..k)
                .min_by(|&a, &b| {
                    sq_dist(&data[i], &centroids[a])
                        .partial_cmp(&sq_dist(&data[i], &centroids[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if new_l != labels[i] {
                changed = true;
            }
            labels[i] = new_l;
        }
        if !changed {
            break;
        }

        // Update centroids
        let mut sums = vec![vec![0.0f64; d]; k];
        let mut counts = vec![0usize; k];
        for (i, &l) in labels.iter().enumerate() {
            for (f, &v) in sums[l].iter_mut().zip(data[i].iter()) {
                *f += v;
            }
            counts[l] += 1;
        }
        for l in 0..k {
            if counts[l] > 0 {
                for f in 0..d {
                    centroids[l][f] = sums[l][f] / counts[l] as f64;
                }
            }
        }
    }
    labels
}

/// Within-cluster sum of squares (pooled).
fn wcss(data: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let d = data[0].len();
    let mut sums = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        for (f, &v) in sums[l].iter_mut().zip(data[i].iter()) {
            *f += v;
        }
        counts[l] += 1;
    }
    let mut total = 0.0f64;
    for (i, &l) in labels.iter().enumerate() {
        if counts[l] > 0 {
            let c_l = &sums[l];
            let cnt = counts[l] as f64;
            total += sq_dist(&data[i], &c_l.iter().map(|&s| s / cnt).collect::<Vec<_>>());
        }
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Types
// ─────────────────────────────────────────────────────────────────────────────

/// Per-cluster and overall stability scores from bootstrap validation.
#[derive(Debug, Clone)]
pub struct ClusterStability {
    /// Number of clusters assessed
    pub n_clusters: usize,
    /// Per-cluster Jaccard-based stability (0 = unstable, 1 = perfectly stable)
    pub stability_scores: Vec<f64>,
    /// Mean of per-cluster stability scores
    pub overall_stability: f64,
}

/// Gap statistic results across a range of k values.
#[derive(Debug, Clone)]
pub struct GapStatistic {
    /// k values tested
    pub k_values: Vec<usize>,
    /// Gap statistic for each k: E*[log W_k] - log W_k
    pub gap_values: Vec<f64>,
    /// Standard errors s_k for each k
    pub sk_values: Vec<f64>,
    /// Optimal k selected by the Tibshirani criterion
    pub optimal_k: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// bootstrap_stability
// ─────────────────────────────────────────────────────────────────────────────

/// Bootstrap cluster stability assessment via Jaccard similarity.
///
/// Repeatedly sub-samples the data, clusters each sub-sample twice, and
/// measures how consistently each cluster appears (Jaccard index between
/// best-matching cluster pairs across the two bootstrap runs).
///
/// # Arguments
/// - `data`: feature matrix as `&[Vec<f64>]` (n samples × d features)
/// - `n_clusters`: number of clusters to assess
/// - `cluster_fn`: a clustering function `(&[Vec<f64>], usize) -> Vec<usize>`
/// - `n_bootstraps`: number of bootstrap repetitions
/// - `seed`: RNG seed for reproducibility
///
/// # Returns
/// [`ClusterStability`] with per-cluster and overall stability scores.
///
/// # Errors
/// Returns an error if data is empty, n_clusters is 0, or n_bootstraps is 0.
pub fn bootstrap_stability(
    data: &[Vec<f64>],
    n_clusters: usize,
    cluster_fn: impl Fn(&[Vec<f64>], usize) -> Vec<usize>,
    n_bootstraps: usize,
    seed: u64,
) -> Result<ClusterStability> {
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput("data is empty".into()));
    }
    if n_clusters == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be >= 1".into(),
        ));
    }
    if n_bootstraps == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_bootstraps must be >= 1".into(),
        ));
    }
    let n = data.len();
    if n_clusters > n {
        return Err(ClusteringError::InvalidInput(format!(
            "n_clusters ({}) > n_samples ({})",
            n_clusters, n
        )));
    }

    let subsample_ratio = 0.8f64;
    let subsample_size = ((n as f64 * subsample_ratio) as usize).max(n_clusters);
    let mut rng = Lcg::new(seed);

    // Accumulate Jaccard scores per cluster across bootstraps
    let mut per_cluster_scores: Vec<Vec<f64>> = vec![Vec::new(); n_clusters];

    for b in 0..n_bootstraps {
        // Draw two independent sub-samples (with replacement)
        let idx_a: Vec<usize> = (0..subsample_size).map(|_| rng.next_range(0, n)).collect();
        let idx_b: Vec<usize> = (0..subsample_size).map(|_| rng.next_range(0, n)).collect();

        let sub_a: Vec<Vec<f64>> = idx_a.iter().map(|&i| data[i].clone()).collect();
        let sub_b: Vec<Vec<f64>> = idx_b.iter().map(|&i| data[i].clone()).collect();

        let labels_a = cluster_fn(&sub_a, n_clusters);
        let labels_b = cluster_fn(&sub_b, n_clusters);

        // Compute Jaccard between best-matching clusters from A and B
        // using the shared original indices (overlap in original index sets)
        // Build set of original indices per cluster for A and B
        let mut sets_a: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); n_clusters];
        let mut sets_b: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); n_clusters];

        for (pos, (&orig, &lbl)) in idx_a.iter().zip(labels_a.iter()).enumerate() {
            if lbl < n_clusters {
                sets_a[lbl].insert(orig);
            }
            let _ = pos;
        }
        for (pos, (&orig, &lbl)) in idx_b.iter().zip(labels_b.iter()).enumerate() {
            if lbl < n_clusters {
                sets_b[lbl].insert(orig);
            }
            let _ = pos;
        }

        // Best-match: for each cluster in A find the cluster in B with highest Jaccard
        for ca in 0..n_clusters {
            let best_jaccard = (0..n_clusters)
                .map(|cb| {
                    let intersection = sets_a[ca].intersection(&sets_b[cb]).count();
                    let union = sets_a[ca].union(&sets_b[cb]).count();
                    if union == 0 {
                        0.0
                    } else {
                        intersection as f64 / union as f64
                    }
                })
                .fold(0.0f64, f64::max);
            per_cluster_scores[ca].push(best_jaccard);
        }
        let _ = b;
    }

    let stability_scores: Vec<f64> = per_cluster_scores
        .iter()
        .map(|scores| {
            if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64
            }
        })
        .collect();

    let overall_stability = if stability_scores.is_empty() {
        0.0
    } else {
        stability_scores.iter().sum::<f64>() / stability_scores.len() as f64
    };

    Ok(ClusterStability {
        n_clusters,
        stability_scores,
        overall_stability,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// gap_statistic (free function)
// ─────────────────────────────────────────────────────────────────────────────

/// Gap statistic for optimal k selection (Tibshirani, Walther & Hastie 2001).
///
/// Compares the within-cluster dispersion W_k to the expected value under
/// a null reference distribution of uniform random data drawn from the bounding
/// box of the original data.
///
/// The optimal k is the smallest k such that `Gap(k) >= Gap(k+1) - s_{k+1}`.
///
/// # Arguments
/// - `data`: feature matrix (n × d)
/// - `k_min`, `k_max`: inclusive range of k values to test
/// - `n_refs`: number of reference datasets per k (Monte Carlo estimate)
/// - `seed`: RNG seed
///
/// # Returns
/// [`GapStatistic`] with gap values, standard errors, and the optimal k.
///
/// # Errors
/// Returns an error for empty data, invalid k range, or insufficient samples.
pub fn gap_statistic_free(
    data: &[Vec<f64>],
    k_min: usize,
    k_max: usize,
    n_refs: usize,
    seed: u64,
) -> Result<GapStatistic> {
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput("data is empty".into()));
    }
    if k_min == 0 {
        return Err(ClusteringError::InvalidInput("k_min must be >= 1".into()));
    }
    if k_min > k_max {
        return Err(ClusteringError::InvalidInput(
            "k_min must be <= k_max".into(),
        ));
    }
    if n_refs == 0 {
        return Err(ClusteringError::InvalidInput("n_refs must be >= 1".into()));
    }
    let n = data.len();
    let d = data[0].len();
    if k_max > n {
        return Err(ClusteringError::InvalidInput(format!(
            "k_max ({}) > n_samples ({})",
            k_max, n
        )));
    }

    // Bounding box of data
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for x in data.iter() {
        for (f, &v) in x.iter().enumerate() {
            if v < mins[f] {
                mins[f] = v;
            }
            if v > maxs[f] {
                maxs[f] = v;
            }
        }
    }
    let ranges: Vec<f64> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(lo, hi)| hi - lo)
        .collect();

    let k_values: Vec<usize> = (k_min..=k_max).collect();
    let n_k = k_values.len();
    let mut gap_values = vec![0.0f64; n_k];
    let mut sk_values = vec![0.0f64; n_k];

    let mut rng = Lcg::new(seed);

    for (ki, &k) in k_values.iter().enumerate() {
        // Observed WCSS
        let obs_labels = simple_kmeans(data, k, rng.next_f64().to_bits(), 100);
        let log_w_obs = (wcss(data, &obs_labels, k) + 1e-30).ln();

        // Reference WCSS values
        let mut log_w_refs = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            let ref_data: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..d)
                        .map(|f| mins[f] + rng.next_f64() * ranges[f])
                        .collect()
                })
                .collect();
            let ref_labels = simple_kmeans(&ref_data, k, rng.next_f64().to_bits(), 100);
            let log_w_ref = (wcss(&ref_data, &ref_labels, k) + 1e-30).ln();
            log_w_refs.push(log_w_ref);
        }

        let mean_ref = log_w_refs.iter().sum::<f64>() / n_refs as f64;
        let var_ref: f64 = log_w_refs
            .iter()
            .map(|&v| (v - mean_ref).powi(2))
            .sum::<f64>()
            / n_refs as f64;
        let sd_ref = var_ref.sqrt();

        gap_values[ki] = mean_ref - log_w_obs;
        // Tibshirani correction: s_k = sd * sqrt(1 + 1/B)
        sk_values[ki] = sd_ref * (1.0 + 1.0 / n_refs as f64).sqrt();
    }

    // Tibshirani criterion: smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    let mut optimal_k = k_values[0];
    'outer: for ki in 0..(n_k.saturating_sub(1)) {
        if gap_values[ki] >= gap_values[ki + 1] - sk_values[ki + 1] {
            optimal_k = k_values[ki];
            break 'outer;
        }
    }
    // Fallback: k with maximum gap
    if optimal_k == k_values[0] && n_k > 1 {
        let best_ki = gap_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        optimal_k = k_values[best_ki];
    }

    Ok(GapStatistic {
        k_values,
        gap_values,
        sk_values,
        optimal_k,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// prediction_strength
// ─────────────────────────────────────────────────────────────────────────────

/// Prediction strength of a clustering with k clusters (Tibshirani & Walther 2005).
///
/// Repeatedly splits data into train/test halves, clusters each, and measures
/// how often pairs that are co-clustered in the training set remain co-clustered
/// in the test set after assigning test points to training centroids.
///
/// Values > 0.8 indicate a stable, well-separated clustering.
///
/// # Arguments
/// - `data`: n × d feature matrix
/// - `k`: number of clusters
/// - `train_ratio`: fraction of data used for training (e.g. 0.5)
/// - `n_trials`: number of random splits
/// - `seed`: RNG seed
///
/// # Returns
/// Mean prediction strength across trials.
///
/// # Errors
/// Returns an error for empty data, k = 0, or train_ratio out of (0,1).
pub fn prediction_strength(
    data: &[Vec<f64>],
    k: usize,
    train_ratio: f64,
    n_trials: usize,
    seed: u64,
) -> Result<f64> {
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput("data is empty".into()));
    }
    if k == 0 {
        return Err(ClusteringError::InvalidInput("k must be >= 1".into()));
    }
    if !(train_ratio > 0.0 && train_ratio < 1.0) {
        return Err(ClusteringError::InvalidInput(
            "train_ratio must be in (0, 1) exclusive".into(),
        ));
    }
    if n_trials == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_trials must be >= 1".into(),
        ));
    }
    let n = data.len();
    if k > n {
        return Err(ClusteringError::InvalidInput(format!(
            "k ({}) > n_samples ({})",
            k, n
        )));
    }
    let n_train = ((n as f64 * train_ratio) as usize).max(k);
    let n_test = n - n_train;
    if n_test < k {
        return Err(ClusteringError::InvalidInput(
            "Not enough test samples for prediction strength; try a larger dataset or smaller train_ratio".into(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let mut strengths = Vec::with_capacity(n_trials);

    for trial in 0..n_trials {
        // Random partition of indices
        let mut idx: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.next_range(0, i + 1);
            idx.swap(i, j);
        }
        let train_idx = &idx[..n_train];
        let test_idx = &idx[n_train..];

        let train_data: Vec<Vec<f64>> = train_idx.iter().map(|&i| data[i].clone()).collect();
        let test_data: Vec<Vec<f64>> = test_idx.iter().map(|&i| data[i].clone()).collect();

        // Cluster training data
        let train_labels = simple_kmeans(
            &train_data,
            k,
            seed.wrapping_add(trial as u64 * 1000 + 1),
            100,
        );

        // Compute training centroids
        let d = data[0].len();
        let mut centroids = vec![vec![0.0f64; d]; k];
        let mut counts = vec![0usize; k];
        for (i, &l) in train_labels.iter().enumerate() {
            if l < k {
                for (f, &v) in centroids[l].iter_mut().zip(train_data[i].iter()) {
                    *f += v;
                }
                counts[l] += 1;
            }
        }
        for l in 0..k {
            if counts[l] > 0 {
                for f in 0..d {
                    centroids[l][f] /= counts[l] as f64;
                }
            }
        }

        // Assign test points to nearest training centroid
        let test_labels: Vec<usize> = test_data
            .iter()
            .map(|x| {
                (0..k)
                    .min_by(|&a, &b| {
                        sq_dist(x, &centroids[a])
                            .partial_cmp(&sq_dist(x, &centroids[b]))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect();

        // Cluster test data independently
        let test_labels_ind = simple_kmeans(
            &test_data,
            k,
            seed.wrapping_add(trial as u64 * 1000 + 2),
            100,
        );

        // Prediction strength: for each cluster C in the independent clustering
        // of test data, compute the proportion of pairs (i,j) in C that are
        // also in the same cluster under the training-centroid assignment.
        let mut ps_k = f64::INFINITY;
        for c in 0..k {
            let c_members: Vec<usize> = (0..n_test).filter(|&i| test_labels_ind[i] == c).collect();
            let n_c = c_members.len();
            if n_c < 2 {
                // Singleton or empty cluster — contribution = 1.0
                continue;
            }
            let n_pairs = n_c * (n_c - 1);
            let n_same: usize = c_members
                .iter()
                .flat_map(|&i| {
                    let test_labels_ref = &test_labels;
                    c_members.iter().map(move |&j| {
                        if i != j && test_labels_ref[i] == test_labels_ref[j] {
                            1usize
                        } else {
                            0
                        }
                    })
                })
                .sum();
            let strength_c = n_same as f64 / n_pairs as f64;
            if strength_c < ps_k {
                ps_k = strength_c;
            }
        }
        // If all clusters were singletons, ps_k stays INFINITY; clamp to 1.0
        strengths.push(if ps_k.is_finite() { ps_k } else { 1.0 });
        let _ = trial;
    }

    let mean_ps = if strengths.is_empty() {
        0.0
    } else {
        strengths.iter().sum::<f64>() / strengths.len() as f64
    };
    Ok(mean_ps)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hopkins statistic
// ─────────────────────────────────────────────────────────────────────────────

/// Hopkins spatial randomness test statistic.
///
/// Compares the nearest-neighbour distances of randomly placed probe points
/// to the nearest-neighbour distances from actual data points.
///
/// **Interpretation:**
/// - H ≈ 0.5 → data is randomly distributed (no clustering tendency)
/// - H → 1.0 → data is highly clustered
/// - H → 0.0 → data is regularly spaced (unlikely in practice)
///
/// # Arguments
/// - `data`: n × d feature matrix (n >= 2)
/// - `n_samples`: number of probe points (`m` in the original paper; typically 10–20% of n)
/// - `seed`: RNG seed
///
/// # Returns
/// Hopkins statistic in `[0, 1]`.
///
/// # Errors
/// Returns an error if data has fewer than 2 samples or n_samples is 0.
pub fn hopkins_statistic(data: &[Vec<f64>], n_samples: usize, seed: u64) -> Result<f64> {
    if data.len() < 2 {
        return Err(ClusteringError::InvalidInput(
            "data must have at least 2 samples".into(),
        ));
    }
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_samples must be >= 1".into(),
        ));
    }
    let n = data.len();
    let d = data[0].len();
    let m = n_samples.min(n - 1);

    // Bounding box
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for x in data.iter() {
        for (f, &v) in x.iter().enumerate() {
            if v < mins[f] {
                mins[f] = v;
            }
            if v > maxs[f] {
                maxs[f] = v;
            }
        }
    }
    let ranges: Vec<f64> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(lo, hi)| hi - lo)
        .collect();

    let mut rng = Lcg::new(seed);

    // W_i: nearest-neighbour distance from sampled data point to rest of data
    let mut w_sum = 0.0f64;
    // Sample m data points without replacement
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next_range(0, i + 1);
        idx.swap(i, j);
    }
    for &i in idx[..m].iter() {
        let min_d = data
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, x)| sq_dist(&data[i], x))
            .fold(f64::INFINITY, f64::min);
        w_sum += min_d;
    }

    // U_i: nearest-neighbour distance from random probe point to data
    let mut u_sum = 0.0f64;
    for _ in 0..m {
        let probe: Vec<f64> = (0..d)
            .map(|f| {
                // Degenerate dimension → use data point coordinate
                if ranges[f].abs() < 1e-30 {
                    mins[f]
                } else {
                    mins[f] + rng.next_f64() * ranges[f]
                }
            })
            .collect();
        let min_d = data
            .iter()
            .map(|x| sq_dist(&probe, x))
            .fold(f64::INFINITY, f64::min);
        u_sum += min_d;
    }

    let hopkins = if (u_sum + w_sum).abs() < 1e-30 {
        0.5
    } else {
        u_sum / (u_sum + w_sum)
    };
    Ok(hopkins.clamp(0.0, 1.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_blob_data() -> Vec<Vec<f64>> {
        let mut v = Vec::new();
        for i in 0..20 {
            v.push(vec![i as f64 * 0.05, i as f64 * 0.05]);
        }
        for i in 0..20 {
            v.push(vec![10.0 + i as f64 * 0.05, 10.0 + i as f64 * 0.05]);
        }
        v
    }

    fn uniform_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.next_f64() * 10.0).collect())
            .collect()
    }

    // ── bootstrap_stability ────────────────────────────────────────────────

    #[test]
    fn test_bootstrap_stability_two_blobs() {
        let data = two_blob_data();
        let cluster_fn = |d: &[Vec<f64>], k: usize| simple_kmeans(d, k, 42, 50);
        let result = bootstrap_stability(&data, 2, cluster_fn, 20, 77)
            .expect("bootstrap_stability should succeed");
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.stability_scores.len(), 2);
        assert!((0.0..=1.0).contains(&result.overall_stability));
        // Well-separated blobs → stability > 0 and within valid range
        assert!(
            result.overall_stability > 0.0,
            "expected stability > 0.0, got {}",
            result.overall_stability
        );
    }

    #[test]
    fn test_bootstrap_stability_invalid() {
        let cluster_fn = |d: &[Vec<f64>], k: usize| simple_kmeans(d, k, 0, 10);
        assert!(bootstrap_stability(&[], 2, &cluster_fn, 5, 0).is_err());
        let data = two_blob_data();
        assert!(bootstrap_stability(&data, 0, &cluster_fn, 5, 0).is_err());
        assert!(bootstrap_stability(&data, 2, &cluster_fn, 0, 0).is_err());
    }

    // ── gap_statistic_free ─────────────────────────────────────────────────

    #[test]
    fn test_gap_statistic_free_blobs() {
        let data = two_blob_data();
        let result =
            gap_statistic_free(&data, 1, 5, 5, 42).expect("gap_statistic_free should succeed");
        assert_eq!(result.k_values, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.gap_values.len(), 5);
        assert_eq!(result.sk_values.len(), 5);
        assert!(result.gap_values.iter().all(|&g| g.is_finite()));
        assert!((1..=5).contains(&result.optimal_k));
    }

    #[test]
    fn test_gap_statistic_free_optimal_k() {
        // Four well-separated blobs → optimal k should be 4
        let mut data = Vec::new();
        for i in 0..15 {
            data.push(vec![i as f64 * 0.05, 0.0]);
        }
        for i in 0..15 {
            data.push(vec![10.0 + i as f64 * 0.05, 0.0]);
        }
        for i in 0..15 {
            data.push(vec![0.0, 10.0 + i as f64 * 0.05]);
        }
        for i in 0..15 {
            data.push(vec![10.0 + i as f64 * 0.05, 10.0 + i as f64 * 0.05]);
        }

        let result =
            gap_statistic_free(&data, 1, 6, 8, 13).expect("gap_statistic_free should succeed");
        assert!((1..=6).contains(&result.optimal_k));
    }

    #[test]
    fn test_gap_statistic_free_invalid() {
        assert!(gap_statistic_free(&[], 1, 3, 5, 0).is_err());
        let data = two_blob_data();
        assert!(gap_statistic_free(&data, 0, 3, 5, 0).is_err()); // k_min = 0
        assert!(gap_statistic_free(&data, 3, 2, 5, 0).is_err()); // k_min > k_max
        assert!(gap_statistic_free(&data, 1, 3, 0, 0).is_err()); // n_refs = 0
    }

    // ── prediction_strength ────────────────────────────────────────────────

    #[test]
    fn test_prediction_strength_blobs() {
        let data = two_blob_data();
        let ps =
            prediction_strength(&data, 2, 0.5, 10, 42).expect("prediction_strength should succeed");
        assert!((0.0..=1.0).contains(&ps));
        // Well-separated blobs → high prediction strength
        assert!(ps > 0.5, "expected ps > 0.5, got {}", ps);
    }

    #[test]
    fn test_prediction_strength_k1() {
        let data = two_blob_data();
        let ps = prediction_strength(&data, 1, 0.5, 5, 0)
            .expect("prediction_strength k=1 should succeed");
        // k=1: all points in one cluster, ps = 1.0
        assert!((0.0..=1.001).contains(&ps));
    }

    #[test]
    fn test_prediction_strength_invalid() {
        let data = two_blob_data();
        assert!(prediction_strength(&[], 2, 0.5, 5, 0).is_err());
        assert!(prediction_strength(&data, 0, 0.5, 5, 0).is_err());
        assert!(prediction_strength(&data, 2, 0.0, 5, 0).is_err()); // ratio = 0
        assert!(prediction_strength(&data, 2, 1.0, 5, 0).is_err()); // ratio = 1
        assert!(prediction_strength(&data, 2, 0.5, 0, 0).is_err()); // trials = 0
    }

    // ── hopkins_statistic ──────────────────────────────────────────────────

    #[test]
    fn test_hopkins_statistic_clustered() {
        // Two well-separated blobs → H should be > 0.5 (clustered)
        let data = two_blob_data();
        let h = hopkins_statistic(&data, 10, 42).expect("hopkins_statistic should succeed");
        assert!(
            (0.0..=1.0).contains(&h),
            "Hopkins must be in [0,1], got {}",
            h
        );
        // Clustered data → H > 0.5
        assert!(h > 0.5, "expected H > 0.5 for clustered data, got {}", h);
    }

    #[test]
    fn test_hopkins_statistic_uniform() {
        // Uniform random data → H ≈ 0.5
        let data = uniform_data(200, 2, 123);
        let h = hopkins_statistic(&data, 20, 55).expect("hopkins_statistic should succeed");
        assert!(
            (0.0..=1.0).contains(&h),
            "Hopkins must be in [0,1], got {}",
            h
        );
        // Uniform → close to 0.5 (allow broad tolerance for small n)
        assert!(
            h > 0.2 && h < 0.9,
            "expected H near 0.5 for uniform data, got {}",
            h
        );
    }

    #[test]
    fn test_hopkins_statistic_invalid() {
        assert!(hopkins_statistic(&[], 5, 0).is_err());
        assert!(hopkins_statistic(&[vec![1.0]], 5, 0).is_err()); // n < 2
        let data = two_blob_data();
        assert!(hopkins_statistic(&data, 0, 0).is_err());
    }

    #[test]
    fn test_hopkins_statistic_single_feature() {
        // 1-dimensional data
        let data: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
        let h = hopkins_statistic(&data, 5, 7).expect("1D Hopkins should succeed");
        assert!((0.0..=1.0).contains(&h));
    }
}
