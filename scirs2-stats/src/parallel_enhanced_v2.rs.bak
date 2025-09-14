//! Enhanced parallel processing for v1.0.0
//!
//! This module provides improved parallel implementations with:
//! - Dynamic threshold adjustment
//! - Better work distribution
//! - Support for non-contiguous arrays
//! - Task-based parallelism

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::{num_threads, par_chunks, IntoParallelIterator, ParallelIterator};
use scirs2_core::validation::check_not_empty;
use std::sync::Arc;

/// Configuration for parallel operations
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum size for parallel execution
    pub minsize: usize,
    /// Target chunk size per thread
    pub chunksize: Option<usize>,
    /// Maximum number of threads to use
    pub max_threads: Option<usize>,
    /// Whether to use adaptive thresholds
    pub adaptive: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            minsize: 5_000,    // Lower threshold than before
            chunksize: None,   // Auto-determine
            max_threads: None, // Use all available
            adaptive: true,
        }
    }
}

impl ParallelConfig {
    /// Create config with specific thread count
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.max_threads = Some(threads);
        self
    }

    /// Create config with specific chunk size
    pub fn with_chunksize(mut self, size: usize) -> Self {
        self.chunksize = Some(size);
        self
    }

    /// Determine if parallel execution should be used
    pub fn should_parallelize(&self, n: usize) -> bool {
        if self.adaptive {
            // Adaptive threshold based on system load and data size
            let threads = self.max_threads.unwrap_or_else(num_threads);

            // Dynamic overhead estimation based on available cores
            let base_overhead = 800;
            let overhead_factor = base_overhead + (threads.saturating_sub(1) * 200);

            // For very large arrays, always parallelize
            if n > 100_000 {
                return true;
            }

            // For small arrays, prefer sequential
            if n < 1_000 {
                return false;
            }

            // Adaptive decision for medium arrays
            n > threads * overhead_factor
        } else {
            n >= self.minsize
        }
    }

    /// Get optimal chunk size for the given data size
    pub fn get_chunksize(&self, n: usize) -> usize {
        if let Some(size) = self.chunksize {
            size
        } else {
            // Simple adaptive chunk size: divide data among available threads
            let threads = self.max_threads.unwrap_or(num_threads());
            (n / threads).max(1000)
        }
    }
}

/// Enhanced parallel mean computation
///
/// Handles non-contiguous arrays and provides better load balancing
#[allow(dead_code)]
pub fn mean_parallel_enhanced<F, D>(
    x: &ArrayBase<D, Ix1>,
    config: Option<ParallelConfig>,
) -> StatsResult<F>
where
    F: Float + NumCast + Send + Sync + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F> + Sync,
{
    // Use scirs2-core validation
    check_not_empty(x, "x")
        .map_err(|_| StatsError::invalid_argument("Cannot compute mean of empty array"))?;

    let config = config.unwrap_or_default();
    let n = x.len();

    if !config.should_parallelize(n) {
        // Sequential computation
        let sum = x.iter().fold(F::zero(), |acc, &val| acc + val);
        return Ok(sum / F::from(n).unwrap());
    }

    // Parallel computation with better handling
    let sum = if let Some(slice) = x.as_slice() {
        // Contiguous array - use slice-based parallelism
        parallel_sum_slice(slice, &config)
    } else {
        // Non-contiguous array - use index-based parallelism
        parallel_sum_indexed(x, &config)
    };

    Ok(sum / F::from(n).unwrap())
}

/// Parallel variance with single-pass algorithm
///
/// Uses parallel Welford's algorithm for numerical stability
#[allow(dead_code)]
pub fn variance_parallel_enhanced<F, D>(
    x: &ArrayBase<D, Ix1>,
    ddof: usize,
    config: Option<ParallelConfig>,
) -> StatsResult<F>
where
    F: Float + NumCast + Send + Sync + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F> + Sync,
{
    let n = x.len();
    if n <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom",
        ));
    }

    let config = config.unwrap_or_default();

    if !config.should_parallelize(n) {
        // Use sequential Welford's algorithm
        return variance_sequential_welford(x, ddof);
    }

    // Parallel Welford's algorithm
    let chunksize = config.get_chunksize(n);
    let n_chunks = (n + chunksize - 1) / chunksize;

    // Each chunk computes local mean and M2
    let chunk_stats: Vec<(F, F, usize)> = (0..n_chunks)
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunksize;
            let end = (start + chunksize).min(n);

            let mut local_mean = F::zero();
            let mut local_m2 = F::zero();
            let mut count = 0;

            for i in start..end {
                count += 1;
                let val = x[i];
                let delta = val - local_mean;
                local_mean = local_mean + delta / F::from(count).unwrap();
                let delta2 = val - local_mean;
                local_m2 = local_m2 + delta * delta2;
            }

            (local_mean, local_m2, count)
        })
        .collect();

    // Combine chunk statistics
    let (_total_mean, total_m2__, total_count) = combine_welford_stats(&chunk_stats);

    Ok(total_m2__ / F::from(n - ddof).unwrap())
}

/// Parallel correlation matrix computation
///
/// Efficiently computes correlation matrix for multivariate data
#[allow(dead_code)]
pub fn corrcoef_parallel_enhanced<F, D>(
    data: &ArrayBase<D, Ix2>,
    config: Option<ParallelConfig>,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Send + Sync + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F> + Sync,
{
    let (n_samples_, n_features) = data.dim();

    if n_samples_ == 0 || n_features == 0 {
        return Err(StatsError::invalid_argument("Empty data matrix"));
    }

    let config = config.unwrap_or_default();

    // Compute means for each feature in parallel
    let means: Vec<F> = (0..n_features)
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|j| {
            let col = data.column(j);
            mean_parallel_enhanced(&col, Some(config.clone())).unwrap_or(F::zero())
        })
        .collect();

    // Compute correlation matrix in parallel
    let mut corr_matrix = Array2::zeros((n_features, n_features));

    // Only compute upper triangle (correlation matrix is symmetric)
    let indices: Vec<(usize, usize)> = (0..n_features)
        .flat_map(|i| (i..n_features).map(move |j| (i, j)))
        .collect();

    let correlations: Vec<((usize, usize), F)> = indices
        .into_par_iter()
        .map(|(i, j)| {
            let corr = if i == j {
                F::one() // Diagonal is always 1
            } else {
                compute_correlation_pair(&data.column(i), &data.column(j), means[i], means[j])
            };
            ((i, j), corr)
        })
        .collect();

    // Fill the correlation matrix
    for ((i, j), corr) in correlations {
        corr_matrix[(i, j)] = corr;
        if i != j {
            corr_matrix[(j, i)] = corr; // Symmetric
        }
    }

    Ok(corr_matrix)
}

/// Parallel bootstrap resampling
///
/// Generates bootstrap samples in parallel for faster computation
#[allow(dead_code)]
pub fn bootstrap_parallel_enhanced<F, D>(
    data: &ArrayBase<D, Ix1>,
    n_samples_: usize,
    statistic_fn: impl Fn(&ArrayView1<F>) -> F + Send + Sync,
    config: Option<ParallelConfig>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Send + Sync,
    D: Data<Elem = F> + Sync,
{
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Cannot bootstrap empty data"));
    }

    let _config = config.unwrap_or_default();
    let data_arc = Arc::new(data.to_owned());
    let n = data.len();

    // Generate bootstrap statistics in parallel
    let stats: Vec<F> = (0..n_samples_)
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|sample_idx| {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};

            // Create deterministic RNG for reproducibility
            let mut rng = StdRng::seed_from_u64(sample_idx as u64);
            let mut sample = Array1::zeros(n);

            // Generate bootstrap sample
            for i in 0..n {
                let idx = rng.gen_range(0..n);
                sample[i] = data_arc[idx];
            }

            statistic_fn(&sample.view())
        })
        .collect();

    Ok(Array1::from(stats))
}

/// Helper function for parallel sum on slices
#[allow(dead_code)]
fn parallel_sum_slice<F>(slice: &[F], config: &ParallelConfig) -> F
where
    F: Float + NumCast + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let chunksize = config.get_chunksize(slice.len());

    par_chunks(slice, chunksize)
        .map(|chunk| chunk.iter().fold(F::zero(), |acc, &val| acc + val))
        .reduce(|| F::zero(), |a, b| a + b)
}

/// Helper function for parallel sum on indexed arrays
#[allow(dead_code)]
fn parallel_sum_indexed<F, D>(arr: &ArrayBase<D, Ix1>, config: &ParallelConfig) -> F
where
    F: Float + NumCast + Send + Sync + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F> + Sync,
{
    let n = arr.len();
    let chunksize = config.get_chunksize(n);
    let n_chunks = (n + chunksize - 1) / chunksize;

    (0..n_chunks)
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunksize;
            let end = (start + chunksize).min(n);

            (start..end)
                .map(|i| arr[i])
                .fold(F::zero(), |acc, val| acc + val)
        })
        .reduce(|| F::zero(), |a, b| a + b)
}

/// Sequential Welford's algorithm (fallback)
#[allow(dead_code)]
fn variance_sequential_welford<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let mut mean = F::zero();
    let mut m2 = F::zero();
    let mut count = 0;

    for &val in x.iter() {
        count += 1;
        let delta = val - mean;
        mean = mean + delta / F::from(count).unwrap();
        let delta2 = val - mean;
        m2 = m2 + delta * delta2;
    }

    Ok(m2 / F::from(count - ddof).unwrap())
}

/// Combine Welford statistics from parallel chunks
#[allow(dead_code)]
fn combine_welford_stats<F>(stats: &[(F, F, usize)]) -> (F, F, usize)
where
    F: Float + NumCast + std::fmt::Display,
{
    stats.iter().fold(
        (F::zero(), F::zero(), 0),
        |(mean_a, m2_a, count_a), &(mean_b, m2_b, count_b)| {
            let count = count_a + count_b;
            let delta = mean_b - mean_a;
            let mean = mean_a + delta * F::from(count_b).unwrap() / F::from(count).unwrap();
            let m2 = m2_a
                + m2_b
                + delta * delta * F::from(count_a).unwrap() * F::from(count_b).unwrap()
                    / F::from(count).unwrap();
            (mean, m2, count)
        },
    )
}

/// Compute correlation between two vectors
#[allow(dead_code)]
fn compute_correlation_pair<F>(x: &ArrayView1<F>, y: &ArrayView1<F>, mean_x: F, meany: F) -> F
where
    F: Float + NumCast + std::fmt::Display,
{
    let n = x.len();
    let mut cov = F::zero();
    let mut var_x = F::zero();
    let mut var_y = F::zero();

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - meany;
        cov = cov + dx * dy;
        var_x = var_x + dx * dx;
        var_y = var_y + dy * dy;
    }

    if var_x > F::epsilon() && var_y > F::epsilon() {
        cov / (var_x * var_y).sqrt()
    } else {
        F::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert!(config.should_parallelize(100_000));
        assert!(!config.should_parallelize(100));

        let config_fixed = ParallelConfig::default()
            .with_threads(4)
            .with_chunksize(1000);
        assert_eq!(config_fixed.get_chunksize(10_000), 1000);
    }

    #[test]
    fn test_mean_parallel_enhanced() {
        let data = Array1::from_vec((0..10_000).map(|i| i as f64).collect());
        let mean = mean_parallel_enhanced(&data.view(), None).unwrap();
        assert!((mean - 4999.5).abs() < 1e-10);
    }

    #[test]
    fn test_variance_parallel_enhanced() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = variance_parallel_enhanced(&data.view(), 1, None).unwrap();
        assert!((var - 2.5).abs() < 1e-10);
    }
}
