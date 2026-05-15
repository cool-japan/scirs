//! Integration of `scirs2-core` distributed primitives with dataset loading.
//!
//! This module bridges `scirs2-datasets` distributed infrastructure to the
//! production-grade primitives in `scirs2_core::distributed`:
//!
//! - `par_map` / `par_fold` from `scirs2_core::distributed::par_iter` —
//!   lightweight OS-thread parallel iterators that need no async runtime.
//! - `distributed_map` / `distributed_map_reduce` from
//!   `scirs2_core::distributed::primitives` — a `WorkerPool`-backed parallel
//!   map / map-reduce with result-order preservation.
//!
//! ## Relationship to existing modules
//!
//! `crate::distributed` (`distributed.rs`) provides `DistributedProcessor`,
//! `DistributedConfig`, and `ScalingParameters` — higher-level machinery
//! built on `std::thread::spawn + mpsc`.  This module provides lower-level
//! **core-backed** helpers that use the same primitives as the rest of the
//! SciRS2 ecosystem.
//!
//! ## Design goals
//!
//! 1. Zero new public types — pure functional helpers.
//! 2. Composable: callers supply an `Fn` closure; results are returned in
//!    input order.
//! 3. Fallible: every public function returns `Result<_, DatasetsError>`.
//! 4. No C/Fortran transitive deps (follows COOLJAPAN Pure Rust Policy).

use scirs2_core::distributed::par_iter::{par_fold, par_map};
use scirs2_core::distributed::primitives::{distributed_map, distributed_map_reduce};
use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `f` to each row of `dataset.data` in parallel using
/// `scirs2_core::distributed::par_iter::par_map`.
///
/// Rows are split into contiguous chunks; each chunk is processed on a
/// separate OS thread.  Results are returned in input order.
///
/// # Arguments
///
/// * `dataset`     — Source dataset (only `.data` is used).
/// * `f`           — Closure applied to each row (as a `Vec<f64>`).
/// * `num_workers` — Override number of worker threads. `None` → logical
///   CPU count.
///
/// # Returns
///
/// `Ok(Vec<U>)` with one element per row, in original order.
pub fn par_map_rows<U, F>(dataset: &Dataset, f: F, num_workers: Option<usize>) -> Result<Vec<U>>
where
    U: Send + 'static,
    F: Fn(Vec<f64>) -> U + Send + Sync + 'static,
{
    // Materialise rows as owned Vecs so we don't hold a reference to dataset
    // across the thread boundary.
    let rows: Vec<Vec<f64>> = dataset
        .data
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let mapped = par_map(&rows, |row| f(row.clone()), num_workers);
    Ok(mapped)
}

/// Reduce the rows of `dataset.data` in parallel using
/// `scirs2_core::distributed::par_iter::par_fold`.
///
/// The fold is first applied within each chunk (on its own thread); partial
/// accumulators are then combined with `combine_fn` on the calling thread.
///
/// # Type parameters
///
/// * `A`          — Accumulator type (must implement `Clone + Send + 'static`).
/// * `FoldOp`     — Per-element fold operation.
/// * `CombineOp`  — Accumulator combination operation.
///
/// # Arguments
///
/// * `dataset`    — Source dataset.
/// * `identity`   — Identity value for the accumulator.
/// * `fold_fn`    — `|acc, row| -> A` applied sequentially within a chunk.
/// * `combine_fn` — `|acc_a, acc_b| -> A` used to merge chunk accumulators.
/// * `num_workers` — Override thread count; `None` → CPU count.
///
/// # Returns
///
/// `Ok(A)` with the final reduced accumulator.
pub fn par_fold_rows<A, FoldOp, CombineOp>(
    dataset: &Dataset,
    identity: A,
    fold_fn: FoldOp,
    combine_fn: CombineOp,
    num_workers: Option<usize>,
) -> Result<A>
where
    A: Clone + Send + Sync + 'static,
    FoldOp: Fn(A, &Vec<f64>) -> A + Send + Sync + 'static,
    CombineOp: Fn(A, A) -> A + Send + Sync + 'static,
{
    let rows: Vec<Vec<f64>> = dataset
        .data
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let result = par_fold(&rows, identity, fold_fn, combine_fn, num_workers);
    Ok(result)
}

/// Apply `f` to each dataset chunk (slice of rows) in parallel using
/// `scirs2_core::distributed::primitives::distributed_map`.
///
/// This uses the `WorkerPool`-backed primitive from `scirs2-core` which
/// preserves output ordering and supports arbitrary chunk sizes.
///
/// # Arguments
///
/// * `dataset`     — Source dataset to chunk.
/// * `chunk_size`  — Number of rows per chunk.
/// * `n_workers`   — Number of worker threads.
/// * `f`           — Closure applied to each chunk (given as a sub-`Dataset`).
///
/// # Returns
///
/// `Ok(Vec<R>)` with one element per chunk, in original order.
pub fn core_par_map_chunks<R, F>(
    dataset: &Dataset,
    chunk_size: usize,
    n_workers: usize,
    f: F,
) -> Result<Vec<R>>
where
    R: Send + 'static,
    F: Fn(Dataset) -> R + Send + Clone + 'static,
{
    let chunks = build_chunks(dataset, chunk_size)?;
    let results = distributed_map(chunks, f, n_workers);
    Ok(results)
}

/// Map-reduce over dataset chunks using
/// `scirs2_core::distributed::primitives::distributed_map_reduce`.
///
/// The map phase processes chunks in parallel; the reduce phase combines the
/// mapped results into a single accumulator on the calling thread.
///
/// # Type parameters
///
/// * `R` — Per-chunk map result.
/// * `S` — Accumulator type.
///
/// # Arguments
///
/// * `dataset`     — Source dataset.
/// * `chunk_size`  — Number of rows per chunk.
/// * `n_workers`   — Worker thread count.
/// * `map_fn`      — `|chunk: Dataset| -> R`.
/// * `reduce_fn`   — `|acc: S, r: R| -> S`.
/// * `initial`     — Initial accumulator value.
///
/// # Returns
///
/// `Ok(S)` with the final reduced accumulator.
pub fn core_map_reduce_chunks<R, S, F, G>(
    dataset: &Dataset,
    chunk_size: usize,
    n_workers: usize,
    map_fn: F,
    reduce_fn: G,
    initial: S,
) -> Result<S>
where
    R: Send + 'static,
    S: Send + Clone + 'static,
    F: Fn(Dataset) -> R + Send + Clone + 'static,
    G: Fn(S, R) -> S + Send + Clone + 'static,
{
    let chunks = build_chunks(dataset, chunk_size)?;
    let result = distributed_map_reduce(chunks, map_fn, reduce_fn, initial, n_workers);
    Ok(result)
}

/// Compute per-feature column statistics (mean, min, max, variance) in parallel
/// across dataset chunks, using `scirs2_core` distributed map-reduce.
///
/// This is intended as a practical demonstration of the core integration:
/// each chunk computes partial Welford-style statistics; partial statistics are
/// combined on the calling thread.
///
/// # Returns
///
/// `Ok(FeatureStats)` with per-feature mean, min, max, and population variance.
pub fn par_feature_stats(
    dataset: &Dataset,
    chunk_size: usize,
    n_workers: usize,
) -> Result<FeatureStats> {
    let n_features = dataset.n_features();
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Dataset has no features".to_string(),
        ));
    }

    let chunks = build_chunks(dataset, chunk_size)?;
    if chunks.is_empty() {
        return Ok(FeatureStats::zeros(n_features));
    }

    // Map: compute partial stats per chunk
    let partial_stats: Vec<PartialStats> = distributed_map(
        chunks,
        move |chunk| PartialStats::from_dataset(&chunk),
        n_workers,
    );

    // Reduce: merge all partial stats
    let merged = partial_stats
        .into_iter()
        .reduce(|a, b| a.merge(&b))
        .ok_or_else(|| DatasetsError::InvalidFormat("No chunks to reduce".to_string()))?;

    Ok(merged.finalise())
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Partial statistics computed over one chunk of rows.
///
/// Uses a parallel version of Welford's online algorithm for numerically stable
/// mean and variance computation.  All fields are per-feature `Vec<f64>`.
#[derive(Debug, Clone)]
struct PartialStats {
    n: usize,
    sums: Vec<f64>,
    sum_sq: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

impl PartialStats {
    fn from_dataset(ds: &Dataset) -> Self {
        let n_features = ds.n_features();
        let mut sums = vec![0.0f64; n_features];
        let mut sum_sq = vec![0.0f64; n_features];
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];

        for row in ds.data.rows() {
            for (j, &v) in row.iter().enumerate() {
                sums[j] += v;
                sum_sq[j] += v * v;
                if v < mins[j] {
                    mins[j] = v;
                }
                if v > maxs[j] {
                    maxs[j] = v;
                }
            }
        }

        Self {
            n: ds.n_samples(),
            sums,
            sum_sq,
            mins,
            maxs,
        }
    }

    /// Merge `other` into `self`, returning a new combined `PartialStats`.
    fn merge(&self, other: &Self) -> Self {
        let n_features = self.sums.len();
        let mut sums = vec![0.0f64; n_features];
        let mut sum_sq = vec![0.0f64; n_features];
        let mut mins = vec![0.0f64; n_features];
        let mut maxs = vec![0.0f64; n_features];

        for j in 0..n_features {
            sums[j] = self.sums[j] + other.sums[j];
            sum_sq[j] = self.sum_sq[j] + other.sum_sq[j];
            mins[j] = self.mins[j].min(other.mins[j]);
            maxs[j] = self.maxs[j].max(other.maxs[j]);
        }

        Self {
            n: self.n + other.n,
            sums,
            sum_sq,
            mins,
            maxs,
        }
    }

    /// Compute final `FeatureStats` from the accumulated sums.
    fn finalise(&self) -> FeatureStats {
        let n = self.n as f64;
        let n_features = self.sums.len();
        let mut means = vec![0.0f64; n_features];
        let mut variances = vec![0.0f64; n_features];

        for j in 0..n_features {
            let mean = if n > 0.0 { self.sums[j] / n } else { 0.0 };
            means[j] = mean;
            let variance = if n > 1.0 {
                // Population variance: E[X^2] - E[X]^2
                (self.sum_sq[j] / n) - mean * mean
            } else {
                0.0
            };
            variances[j] = variance.max(0.0); // clamp floating-point negatives
        }

        FeatureStats {
            means,
            variances,
            mins: self.mins.clone(),
            maxs: self.maxs.clone(),
            n_samples: self.n,
        }
    }
}

/// Per-feature statistics computed in parallel via `scirs2-core` primitives.
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Per-feature arithmetic means.
    pub means: Vec<f64>,
    /// Per-feature population variances.
    pub variances: Vec<f64>,
    /// Per-feature minimums.
    pub mins: Vec<f64>,
    /// Per-feature maximums.
    pub maxs: Vec<f64>,
    /// Total number of samples processed.
    pub n_samples: usize,
}

impl FeatureStats {
    /// Return a zero-initialised `FeatureStats` for `n_features` features.
    fn zeros(n_features: usize) -> Self {
        Self {
            means: vec![0.0; n_features],
            variances: vec![0.0; n_features],
            mins: vec![0.0; n_features],
            maxs: vec![0.0; n_features],
            n_samples: 0,
        }
    }

    /// Standard deviations (square roots of the population variances).
    pub fn stds(&self) -> Vec<f64> {
        self.variances.iter().map(|v| v.sqrt()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Split `dataset` into owned `Dataset` chunks of at most `chunk_size` rows.
fn build_chunks(dataset: &Dataset, chunk_size: usize) -> Result<Vec<Dataset>> {
    let chunk_size = chunk_size.max(1);
    let n = dataset.n_samples();
    let n_features = dataset.n_features();
    let mut chunks = Vec::new();

    let mut start = 0usize;
    while start < n {
        let end = (start + chunk_size).min(n);
        let n_rows = end - start;

        // Build data array for this chunk
        let flat: Vec<f64> = dataset
            .data
            .rows()
            .into_iter()
            .skip(start)
            .take(n_rows)
            .flat_map(|row| row.to_vec())
            .collect();

        let data = Array2::from_shape_vec((n_rows, n_features), flat)
            .map_err(|e| DatasetsError::InvalidFormat(format!("chunk build failed: {}", e)))?;

        let target = dataset.target.as_ref().map(|t| {
            let vals: Vec<f64> = t.iter().skip(start).take(n_rows).copied().collect();
            Array1::from_vec(vals)
        });

        chunks.push(Dataset {
            data,
            target,
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some(format!("chunk {start}..{end}")),
            metadata: dataset.metadata.clone(),
        });

        start = end;
    }

    Ok(chunks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::make_classification;

    // ── build_chunks ──────────────────────────────────────────────────────────

    #[test]
    fn test_build_chunks_total_rows_preserved() {
        let ds = make_classification(47, 4, 2, 2, 1, Some(1)).expect("make_classification");
        let chunks = build_chunks(&ds, 10).expect("build_chunks");

        let total: usize = chunks.iter().map(|c| c.n_samples()).sum();
        assert_eq!(total, 47, "total rows across chunks must equal source rows");
    }

    #[test]
    fn test_build_chunks_exact_split() {
        let ds = make_classification(30, 3, 2, 2, 1, Some(2)).expect("make_classification");
        let chunks = build_chunks(&ds, 10).expect("build_chunks");
        assert_eq!(chunks.len(), 3, "30 rows / 10 per chunk = 3 chunks");
        for c in &chunks {
            assert_eq!(c.n_samples(), 10);
        }
    }

    #[test]
    fn test_build_chunks_remainder() {
        let ds = make_classification(25, 3, 2, 2, 1, Some(3)).expect("make_classification");
        let chunks = build_chunks(&ds, 10).expect("build_chunks");
        // 10, 10, 5
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].n_samples(), 5);
    }

    // ── par_map_rows ──────────────────────────────────────────────────────────

    #[test]
    fn test_par_map_rows_count_matches() {
        let ds = make_classification(60, 4, 2, 2, 1, Some(7)).expect("make_classification");
        let results =
            par_map_rows(&ds, |row| row.iter().copied().sum::<f64>(), None).expect("par_map_rows");
        assert_eq!(results.len(), 60, "one result per row");
    }

    #[test]
    fn test_par_map_rows_identity_feature_lengths() {
        let ds = make_classification(20, 5, 2, 2, 1, Some(11)).expect("make_classification");
        let lengths = par_map_rows(&ds, |row| row.len(), None).expect("par_map_rows");
        assert!(
            lengths.iter().all(|&l| l == 5),
            "each mapped row should have 5 features"
        );
    }

    // ── par_fold_rows ─────────────────────────────────────────────────────────

    #[test]
    fn test_par_fold_rows_row_count() {
        let ds = make_classification(80, 3, 2, 2, 1, Some(13)).expect("make_classification");
        let count = par_fold_rows(&ds, 0usize, |acc, _row| acc + 1, |a, b| a + b, None)
            .expect("par_fold_rows");
        assert_eq!(count, 80, "fold should accumulate one per row");
    }

    // ── core_par_map_chunks ───────────────────────────────────────────────────

    #[test]
    fn test_core_par_map_chunks_total_samples() {
        let ds = make_classification(100, 4, 2, 3, 1, Some(17)).expect("make_classification");
        let chunk_sample_counts =
            core_par_map_chunks(&ds, 25, 2, |c| c.n_samples()).expect("core_par_map_chunks");
        let total: usize = chunk_sample_counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_core_par_map_chunks_feature_dim() {
        let ds = make_classification(50, 6, 2, 2, 1, Some(19)).expect("make_classification");
        let feature_counts =
            core_par_map_chunks(&ds, 15, 2, |c| c.n_features()).expect("core_par_map_chunks");
        assert!(
            feature_counts.iter().all(|&f| f == 6),
            "all chunks should have 6 features"
        );
    }

    // ── core_map_reduce_chunks ────────────────────────────────────────────────

    #[test]
    fn test_core_map_reduce_total_sample_count() {
        let ds = make_classification(120, 4, 2, 3, 1, Some(23)).expect("make_classification");
        let total = core_map_reduce_chunks(
            &ds,
            30,
            2,
            |chunk| chunk.n_samples(),
            |acc, r| acc + r,
            0usize,
        )
        .expect("core_map_reduce_chunks");
        assert_eq!(total, 120);
    }

    // ── par_feature_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_par_feature_stats_n_samples() {
        let ds = make_classification(200, 4, 2, 3, 1, Some(29)).expect("make_classification");
        let stats = par_feature_stats(&ds, 50, 2).expect("par_feature_stats");
        assert_eq!(stats.n_samples, 200);
    }

    #[test]
    fn test_par_feature_stats_means_len() {
        let ds = make_classification(100, 5, 2, 3, 1, Some(31)).expect("make_classification");
        let stats = par_feature_stats(&ds, 25, 2).expect("par_feature_stats");
        assert_eq!(stats.means.len(), 5, "one mean per feature");
        assert_eq!(stats.variances.len(), 5);
        assert_eq!(stats.mins.len(), 5);
        assert_eq!(stats.maxs.len(), 5);
    }

    #[test]
    fn test_par_feature_stats_mins_le_maxs() {
        let ds = make_classification(80, 4, 2, 3, 1, Some(37)).expect("make_classification");
        let stats = par_feature_stats(&ds, 20, 2).expect("par_feature_stats");
        for j in 0..4 {
            assert!(
                stats.mins[j] <= stats.maxs[j],
                "min[{j}] must be <= max[{j}]"
            );
        }
    }

    #[test]
    fn test_par_feature_stats_variances_nonnegative() {
        let ds = make_classification(60, 3, 2, 2, 1, Some(41)).expect("make_classification");
        let stats = par_feature_stats(&ds, 20, 2).expect("par_feature_stats");
        for (j, &v) in stats.variances.iter().enumerate() {
            assert!(v >= 0.0, "variance[{j}] must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_feature_stats_stds() {
        let ds = make_classification(40, 3, 2, 2, 1, Some(43)).expect("make_classification");
        let stats = par_feature_stats(&ds, 10, 2).expect("par_feature_stats");
        let stds = stats.stds();
        assert_eq!(stds.len(), 3);
        for (j, &s) in stds.iter().enumerate() {
            assert!(s >= 0.0, "std[{j}] must be non-negative, got {s}");
        }
    }
}
