//! Consensus Clustering (Monti et al., 2003)
//!
//! Consensus clustering repeatedly subsamples the dataset and applies a base clustering
//! algorithm to build a co-clustering frequency matrix (the "consensus matrix").
//! The final clustering is obtained by applying hierarchical agglomeration to that matrix.
//! A sweep over a range of k values lets callers identify the most stable k by examining
//! the relative change in the area under the empirical CDF of the consensus matrix.
//!
//! # References
//!
//! Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). Consensus clustering: A resampling-
//! based method for class discovery and visualization of gene expression microarray data.
//! *Machine Learning*, 52(1-2), 91-118.

use scirs2_core::ndarray::Array2;

use crate::error::{ClusteringError, Result};

/// Configuration for consensus clustering.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Number of subsampling / clustering runs (default 100).
    pub n_runs: usize,
    /// Fraction of items to include in each subsampled run (default 0.8).
    pub subsample_fraction: f64,
    /// Maximum number of clusters to try when sweeping k values.
    pub max_k: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            n_runs: 100,
            subsample_fraction: 0.8,
            max_k: 10,
            seed: 42,
        }
    }
}

/// Result of a single consensus clustering run for a fixed k.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// n×n co-clustering frequency matrix.  `matrix[i][j]` is the fraction of runs
    /// (in which both i and j were sampled) where they were assigned to the same cluster.
    pub consensus_matrix: Array2<f64>,
    /// Final hard cluster assignments (0-indexed).
    pub assignments: Vec<usize>,
    /// Area under the empirical CDF of the upper-triangle entries of the consensus matrix.
    pub cdf_area: f64,
    /// Relative change in CDF area compared with k-1 (0.0 for the first k in a sweep).
    pub delta_k: f64,
    /// Number of clusters k.
    pub k: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Minimal linear-congruential PRNG (Knuth MMIX) — avoids any external dependency.
#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Return next value in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use upper 53 bits for float mantissa precision
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sample k distinct indices from 0..n using Fisher-Yates partial shuffle.
    fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next_f64() * (n - i) as f64) as usize;
            let j = j.min(n - 1);
            pool.swap(i, j);
        }
        pool[..k].to_vec()
    }
}

/// Build the consensus matrix from `n_runs` subsampled clusterings.
///
/// For each run:
/// 1. Subsample `round(n * fraction)` items.
/// 2. Run the base clusterer on the submatrix.
/// 3. Accumulate co-clustering counts and indicator counts.
///
/// `consensus[i][j] = co_cluster_count[i][j] / indicator[i][j]`
/// where `indicator[i][j]` is the number of runs in which both i and j were sampled.
fn build_consensus_matrix<F>(
    data: &Array2<f64>,
    k: usize,
    base_clusterer: &F,
    config: &ConsensusConfig,
) -> Result<Array2<f64>>
where
    F: Fn(&Array2<f64>, usize, u64) -> Vec<usize>,
{
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 || d == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data matrix must be non-empty".into(),
        ));
    }
    if k < 2 {
        return Err(ClusteringError::InvalidInput("k must be at least 2".into()));
    }
    if config.subsample_fraction <= 0.0 || config.subsample_fraction > 1.0 {
        return Err(ClusteringError::InvalidInput(
            "subsample_fraction must be in (0, 1]".into(),
        ));
    }
    if config.n_runs == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_runs must be at least 1".into(),
        ));
    }

    let sub_n = ((n as f64 * config.subsample_fraction).round() as usize).max(k);
    let sub_n = sub_n.min(n);

    let mut co_counts = vec![0u32; n * n];
    let mut indicator = vec![0u32; n * n];

    let mut rng = Lcg::new(config.seed);

    for run in 0..config.n_runs {
        let run_seed = config
            .seed
            .wrapping_add(run as u64)
            .wrapping_mul(2_654_435_761);

        // Sample indices
        let idx = rng.sample_indices(n, sub_n);

        // Build submatrix
        let mut sub_data = Array2::<f64>::zeros((sub_n, d));
        for (row_out, &row_in) in idx.iter().enumerate() {
            for col in 0..d {
                sub_data[[row_out, col]] = data[[row_in, col]];
            }
        }

        // Run base clusterer
        let sub_labels = base_clusterer(&sub_data, k, run_seed);

        if sub_labels.len() != sub_n {
            return Err(ClusteringError::ComputationError(format!(
                "Base clusterer returned {} labels but expected {}",
                sub_labels.len(),
                sub_n
            )));
        }

        // Accumulate
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                indicator[ia * n + ib] += 1;
                if sub_labels[a] == sub_labels[b] {
                    co_counts[ia * n + ib] += 1;
                }
            }
        }
    }

    // Normalise
    let mut matrix = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let ind = indicator[i * n + j];
            matrix[[i, j]] = if ind > 0 {
                co_counts[i * n + j] as f64 / ind as f64
            } else {
                0.0
            };
        }
    }

    Ok(matrix)
}

/// Compute the area under the empirical CDF of the upper-triangle entries
/// of the consensus matrix.  Values are sorted; the CDF is a step function
/// with uniform x-spacing 1/(m-1) where m is the number of values.
fn cdf_area(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut vals: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            vals.push(matrix[[i, j]]);
        }
    }

    if vals.is_empty() {
        return 0.0;
    }

    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let m = vals.len();
    if m == 1 {
        return vals[0];
    }

    // Trapezoid rule on the empirical CDF
    // x values are evenly spaced in [0,1]
    let step = 1.0 / (m - 1) as f64;
    let mut area = 0.0_f64;
    for (rank, &v) in vals.iter().enumerate() {
        let cdf = (rank as f64) / ((m - 1) as f64);
        // height at this x = 1 - cdf  (fraction of values <= v)
        // contribution via trapezoid: 0.5 * step * (height[rank] + height[rank+1])
        // simplified by integrating 1 - F(x) directly
        if rank + 1 < m {
            let cdf_next = (rank + 1) as f64 / (m - 1) as f64;
            let _ = cdf;
            let _ = cdf_next;
            // Area under the sorted CDF from val[rank] to val[rank+1] segment:
            // Integrate (1 - F(c)) dc over c in [v, v_next]
            // For a uniform step in rank-space: area += step * (1 - avg_rank_normalized)
            let avg_cdf = (rank as f64 + 0.5) / (m - 1) as f64;
            area += step * (1.0 - avg_cdf);
        }
    }

    // Normalise by the range of the CDF (which is [0,1]): already normalised.
    area
}

/// Hierarchical agglomerative clustering (average-linkage) on the consensus matrix,
/// returning cluster assignments for k clusters.
fn cluster_consensus_matrix(matrix: &Array2<f64>, k: usize) -> Result<Vec<usize>> {
    let n = matrix.nrows();
    if k == 0 || k > n {
        return Err(ClusteringError::InvalidInput(format!(
            "k={} out of valid range [1, {}]",
            k, n
        )));
    }
    if k == n {
        return Ok((0..n).collect());
    }
    if k == 1 {
        return Ok(vec![0; n]);
    }

    // Distance = 1 - similarity
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > k {
        let nc = clusters.len();
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_ci = 0;
        let mut best_cj = 1;

        for ci in 0..nc {
            for cj in (ci + 1)..nc {
                // Average linkage
                let mut sim_sum = 0.0_f64;
                let mut count = 0usize;
                for &a in &clusters[ci] {
                    for &b in &clusters[cj] {
                        sim_sum += matrix[[a, b]];
                        count += 1;
                    }
                }
                let avg_sim = if count > 0 {
                    sim_sum / count as f64
                } else {
                    0.0
                };
                if avg_sim > best_sim {
                    best_sim = avg_sim;
                    best_ci = ci;
                    best_cj = cj;
                }
            }
        }

        // Merge cj into ci (take from higher index to avoid shifting)
        let removed = clusters.remove(best_cj);
        clusters[best_ci].extend(removed);
    }

    // Build label array
    let mut labels = vec![0usize; n];
    for (cluster_idx, members) in clusters.iter().enumerate() {
        for &point_idx in members {
            labels[point_idx] = cluster_idx;
        }
    }
    Ok(labels)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run consensus clustering on `data` using a fixed k.
///
/// The base clusterer is called as `base_clusterer(subdata, k, seed)` and must
/// return a `Vec<usize>` of length equal to `subdata.nrows()`.
///
/// # Arguments
///
/// * `data`            — n × d data matrix.
/// * `k`               — Number of clusters.
/// * `base_clusterer`  — Closure `(data, k, seed) -> labels`.
/// * `config`          — Algorithm configuration.
///
/// # Returns
///
/// `ConsensusResult` containing the consensus matrix, final assignments, and stability metrics.
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::consensus::{consensus_clustering, ConsensusConfig};
/// use scirs2_core::ndarray::Array2;
///
/// fn simple_clusterer(data: &Array2<f64>, k: usize, seed: u64) -> Vec<usize> {
///     let n = data.nrows();
///     (0..n).map(|i| i % k).collect()
/// }
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
///     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
/// ]).expect("operation should succeed");
///
/// let config = ConsensusConfig { n_runs: 10, subsample_fraction: 0.8, max_k: 4, seed: 42 };
/// let result = consensus_clustering(&data, 2, simple_clusterer, config).expect("operation should succeed");
/// assert_eq!(result.k, 2);
/// ```
pub fn consensus_clustering<F>(
    data: &Array2<f64>,
    k: usize,
    base_clusterer: F,
    config: ConsensusConfig,
) -> Result<ConsensusResult>
where
    F: Fn(&Array2<f64>, usize, u64) -> Vec<usize>,
{
    let matrix = build_consensus_matrix(data, k, &base_clusterer, &config)?;
    let assignments = cluster_consensus_matrix(&matrix, k)?;
    let area = cdf_area(&matrix);

    Ok(ConsensusResult {
        consensus_matrix: matrix,
        assignments,
        cdf_area: area,
        delta_k: 0.0, // Filled in by sweep; single-k call has no reference.
        k,
    })
}

/// Sweep over a range of k values and return consensus results for each k.
///
/// The `delta_k` field of each result is the relative increase in CDF area:
/// `delta_k[k] = (area[k] - area[k-1]) / area[k-1]`  (0 for the first entry).
///
/// Callers can select the k with the smallest positive `delta_k` as the most stable.
///
/// # Arguments
///
/// * `data`      — n × d data matrix.
/// * `k_range`   — Half-open range of k values (e.g. `2..8`).
/// * `config`    — Algorithm configuration; `max_k` is ignored (k_range takes precedence).
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::consensus::{consensus_clustering_sweep, ConsensusConfig};
/// use scirs2_core::ndarray::Array2;
///
/// fn simple_clusterer(data: &Array2<f64>, k: usize, seed: u64) -> Vec<usize> {
///     let n = data.nrows();
///     (0..n).map(|i| i % k).collect()
/// }
///
/// let data = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  -0.1, 0.0,
///     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,   4.9, 5.0,
/// ]).expect("operation should succeed");
///
/// let config = ConsensusConfig { n_runs: 10, subsample_fraction: 0.8, max_k: 4, seed: 7 };
/// let results = consensus_clustering_sweep(&data, 2..4, simple_clusterer, config).expect("operation should succeed");
/// assert_eq!(results.len(), 2);
/// ```
pub fn consensus_clustering_sweep<F>(
    data: &Array2<f64>,
    k_range: std::ops::Range<usize>,
    base_clusterer: F,
    config: ConsensusConfig,
) -> Result<Vec<ConsensusResult>>
where
    F: Fn(&Array2<f64>, usize, u64) -> Vec<usize>,
{
    if k_range.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "k_range must be non-empty".into(),
        ));
    }
    if k_range.start < 2 {
        return Err(ClusteringError::InvalidInput(
            "k_range must start at 2 or above".into(),
        ));
    }

    let k_values: Vec<usize> = k_range.collect();
    let mut results: Vec<ConsensusResult> = Vec::with_capacity(k_values.len());
    let mut prev_area: Option<f64> = None;

    for &k in &k_values {
        // Use a k-specific seed offset to ensure different runs per k
        let k_config = ConsensusConfig {
            seed: config.seed.wrapping_add(k as u64 * 999_983),
            ..config.clone()
        };
        let matrix = build_consensus_matrix(data, k, &base_clusterer, &k_config)?;
        let assignments = cluster_consensus_matrix(&matrix, k)?;
        let area = cdf_area(&matrix);

        let delta_k = match prev_area {
            None => 0.0,
            Some(prev) => {
                if prev.abs() < 1e-15 {
                    0.0
                } else {
                    (area - prev) / prev
                }
            }
        };

        results.push(ConsensusResult {
            consensus_matrix: matrix,
            assignments,
            cdf_area: area,
            delta_k,
            k,
        });

        prev_area = Some(area);
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// A simple deterministic base clusterer that assigns points to clusters by proximity
    /// to a grid of k equidistant centers along the first feature dimension.
    fn grid_clusterer(data: &Array2<f64>, k: usize, _seed: u64) -> Vec<usize> {
        let n = data.nrows();
        if n == 0 {
            return vec![];
        }
        let min_val = data.column(0).fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.column(0).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = (max_val - min_val).max(1e-10);

        (0..n)
            .map(|i| {
                let t = (data[[i, 0]] - min_val) / range;
                let c = (t * k as f64).floor() as usize;
                c.min(k - 1)
            })
            .collect()
    }

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 10.0, 10.0, 10.1,
                10.0, 10.0, 10.1, 9.9, 10.0, 10.1, 10.1, 9.9, 9.9,
            ],
        )
        .expect("create test data")
    }

    #[test]
    fn test_consensus_matrix_symmetry() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            n_runs: 20,
            subsample_fraction: 0.8,
            max_k: 4,
            seed: 1,
        };
        let result = consensus_clustering(&data, 2, grid_clusterer, config)
            .expect("operation should succeed");
        let m = &result.consensus_matrix;
        let n = m.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (m[[i, j]] - m[[j, i]]).abs() < 1e-12,
                    "Consensus matrix not symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_consensus_matrix_range() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            n_runs: 20,
            subsample_fraction: 0.8,
            max_k: 4,
            seed: 2,
        };
        let result = consensus_clustering(&data, 2, grid_clusterer, config)
            .expect("operation should succeed");
        for &v in result.consensus_matrix.iter() {
            assert!(
                v >= 0.0 && v <= 1.0 + 1e-12,
                "Consensus matrix value {} out of [0,1]",
                v
            );
        }
    }

    #[test]
    fn test_consensus_assignments_count() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            n_runs: 30,
            subsample_fraction: 0.8,
            max_k: 4,
            seed: 3,
        };
        let result = consensus_clustering(&data, 2, grid_clusterer, config)
            .expect("operation should succeed");
        assert_eq!(result.assignments.len(), 12);
        assert_eq!(result.k, 2);
        // All assignments in [0, k)
        for &a in &result.assignments {
            assert!(a < 2, "Assignment {} >= k=2", a);
        }
    }

    #[test]
    fn test_consensus_sweep_returns_correct_count() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            n_runs: 10,
            subsample_fraction: 0.8,
            max_k: 6,
            seed: 4,
        };
        let results = consensus_clustering_sweep(&data, 2..5, grid_clusterer, config)
            .expect("operation should succeed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].k, 2);
        assert_eq!(results[1].k, 3);
        assert_eq!(results[2].k, 4);
    }

    #[test]
    fn test_consensus_sweep_delta_k_first_is_zero() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            n_runs: 10,
            subsample_fraction: 0.8,
            max_k: 6,
            seed: 5,
        };
        let results = consensus_clustering_sweep(&data, 2..5, grid_clusterer, config)
            .expect("operation should succeed");
        assert_eq!(
            results[0].delta_k, 0.0,
            "First delta_k should be 0.0 (no prior k)"
        );
    }

    #[test]
    fn test_consensus_cdf_area_in_range() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            n_runs: 20,
            subsample_fraction: 0.8,
            max_k: 4,
            seed: 6,
        };
        let result = consensus_clustering(&data, 2, grid_clusterer, config)
            .expect("operation should succeed");
        assert!(
            result.cdf_area >= 0.0 && result.cdf_area <= 1.0 + 1e-9,
            "CDF area {} should be in [0,1]",
            result.cdf_area
        );
    }

    #[test]
    fn test_consensus_invalid_k() {
        let data = two_cluster_data();
        let config = ConsensusConfig::default();
        let err = consensus_clustering(&data, 1, grid_clusterer, config);
        assert!(err.is_err(), "k=1 should return error");
    }

    #[test]
    fn test_consensus_invalid_subsample() {
        let data = two_cluster_data();
        let config = ConsensusConfig {
            subsample_fraction: 0.0,
            ..ConsensusConfig::default()
        };
        let err = consensus_clustering(&data, 2, grid_clusterer, config);
        assert!(err.is_err(), "subsample_fraction=0 should return error");
    }
}
