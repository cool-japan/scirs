//! Enhanced parallel processing for statistical operations (v4)
//!
//! This module provides comprehensive parallel processing capabilities that
//! fully leverage scirs2-core's unified parallel operations infrastructure.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumCast, One, Zero};
use rand::Rng;
use scirs2_core::{parallel_ops::*, simd_ops::SimdUnifiedOps, validation::*};
use std::sync::Arc;

/// Enhanced parallel configuration
#[derive(Debug, Clone)]
pub struct EnhancedParallelConfig {
    /// Number of threads to use (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Minimum chunk size for parallel processing
    pub min_chunksize: usize,
    /// Maximum number of chunks
    pub max_chunks: usize,
    /// Enable NUMA-aware processing
    pub numa_aware: bool,
    /// Enable work stealing
    pub work_stealing: bool,
}

impl Default for EnhancedParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            min_chunksize: 1000,
            max_chunks: num_cpus::get() * 4,
            numa_aware: true,
            work_stealing: true,
        }
    }
}

/// Enhanced parallel statistics processor
pub struct EnhancedParallelProcessor<F> {
    config: EnhancedParallelConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F> EnhancedParallelProcessor<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>
        + num_traits::FromPrimitive,
{
    /// Create new enhanced parallel processor
    pub fn new() -> Self {
        Self {
            config: EnhancedParallelConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EnhancedParallelConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Parallel mean computation with optimal chunking
    pub fn mean_parallel_enhanced(&self, data: &ArrayView1<F>) -> StatsResult<F> {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        let n = data.len();

        if n < self.config.min_chunksize {
            // Use sequential computation for small datasets
            return Ok(data.mean().unwrap());
        }

        // Use scirs2-core's parallel operations
        let chunksize = self.calculate_optimal_chunksize(n);
        let result = data
            .as_slice()
            .unwrap()
            .par_chunks(chunksize)
            .map(|chunk| {
                // Process each chunk (map phase)
                let sum: F = chunk.iter().copied().sum();
                let count = chunk.len();
                (sum, count)
            })
            .reduce(
                || (F::zero(), 0),
                |(sum1, count1), (sum2, count2)| {
                    // Combine results (reduce phase)
                    (sum1 + sum2, count1 + count2)
                },
            );

        let (total_sum, total_count) = result;
        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Parallel variance computation with Welford's algorithm
    pub fn variance_parallel_enhanced(&self, data: &ArrayView1<F>, ddof: usize) -> StatsResult<F> {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        let n = data.len();

        if n < self.config.min_chunksize {
            // Use sequential computation for small datasets
            let mean = data.mean().unwrap();
            let sum_sq_diff: F = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
            return Ok(sum_sq_diff / F::from(n.saturating_sub(ddof)).unwrap());
        }

        // First pass: compute mean in parallel
        let mean = self.mean_parallel_enhanced(data)?;

        // Second pass: compute variance in parallel
        let chunksize = self.calculate_optimal_chunksize(n);
        let result = data
            .as_slice()
            .unwrap()
            .par_chunks(chunksize)
            .map(|chunk| {
                // Process each chunk (map phase)
                let sum_sq_diff: F = chunk.iter().map(|&x| (x - mean) * (x - mean)).sum();
                let count = chunk.len();
                (sum_sq_diff, count)
            })
            .reduce(
                || (F::zero(), 0),
                |(sum1, count1), (sum2, count2)| {
                    // Combine results (reduce phase)
                    (sum1 + sum2, count1 + count2)
                },
            );

        let (total_sum_sq_diff, total_count) = result;
        let denominator = total_count.saturating_sub(ddof);

        if denominator == 0 {
            return Err(StatsError::InvalidArgument(
                "Insufficient degrees of freedom".to_string(),
            ));
        }

        Ok(total_sum_sq_diff / F::from(denominator).unwrap())
    }

    /// Parallel correlation matrix computation
    pub fn correlation_matrix_parallel(&self, matrix: &ArrayView2<F>) -> StatsResult<Array2<F>> {
        checkarray_finite(matrix, "matrix")?;

        let (_n_samples_, n_features) = matrix.dim();

        if n_features < 2 {
            return Err(StatsError::InvalidArgument(
                "At least 2 features required for correlation matrix".to_string(),
            ));
        }

        // Compute means in parallel
        let means = parallel_map_collect(0..n_features, |i| {
            let col = matrix.column(i);
            self.mean_parallel_enhanced(&col).unwrap()
        });

        // Compute correlation coefficients in parallel
        let mut corr_matrix = Array2::zeros((n_features, n_features));
        let pairs: Vec<(usize, usize)> = (0..n_features)
            .flat_map(|i| (i..n_features).map(move |j| (i, j)))
            .collect();

        let correlations = parallel_map_collect(&pairs, |&(i, j)| {
            if i == j {
                (i, j, F::one())
            } else {
                let col_i = matrix.column(i);
                let col_j = matrix.column(j);
                let corr = self
                    .correlation_coefficient(&col_i, &col_j, means[i], means[j])
                    .unwrap();
                (i, j, corr)
            }
        });

        // Fill correlation matrix
        for (i, j, corr) in correlations {
            corr_matrix[[i, j]] = corr;
            if i != j {
                corr_matrix[[j, i]] = corr;
            }
        }

        Ok(corr_matrix)
    }

    /// Parallel bootstrap sampling
    pub fn bootstrap_parallel_enhanced(
        &self,
        data: &ArrayView1<F>,
        n_bootstrap: usize,
        statistic_fn: impl Fn(&ArrayView1<F>) -> F + Send + Sync,
        seed: Option<u64>,
    ) -> StatsResult<Array1<F>> {
        checkarray_finite(data, "data")?;
        check_positive(n_bootstrap, "n_bootstrap")?;

        let statistic_fn = Arc::new(statistic_fn);
        let data_arc = Arc::new(data.to_owned());

        let results = parallel_map_collect(0..n_bootstrap, |i| {
            use scirs2_core::random::Random;
            let mut rng = match seed {
                Some(s) => Random::seed(s.wrapping_add(i as u64)),
                None => Random::seed(i as u64), // Use index as seed for determinism in parallel
            };

            // Generate _bootstrap sample
            let n = data_arc.len();
            let mut bootstrap_sample = Array1::zeros(n);
            for j in 0..n {
                let idx = rng.gen_range(0..n);
                bootstrap_sample[j] = data_arc[idx];
            }

            // Apply statistic function
            statistic_fn(&bootstrap_sample.view())
        });

        Ok(Array1::from_vec(results))
    }

    /// Parallel matrix operations with optimal memory access patterns
    pub fn matrix_operations_parallel(
        &self,
        matrix: &ArrayView2<F>,
    ) -> StatsResult<MatrixParallelResult<F>> {
        checkarray_finite(matrix, "matrix")?;

        let (rows, cols) = matrix.dim();

        // Compute row statistics in parallel
        let row_means = parallel_map_collect(0..rows, |i| {
            let row = matrix.row(i);
            self.mean_parallel_enhanced(&row).unwrap()
        });

        let row_vars = parallel_map_collect(0..rows, |i| {
            let row = matrix.row(i);
            self.variance_parallel_enhanced(&row, 1).unwrap()
        });

        // Compute column statistics in parallel
        let col_means = parallel_map_collect(0..cols, |j| {
            let col = matrix.column(j);
            self.mean_parallel_enhanced(&col).unwrap()
        });

        let col_vars = parallel_map_collect(0..cols, |j| {
            let col = matrix.column(j);
            self.variance_parallel_enhanced(&col, 1).unwrap()
        });

        // Compute overall statistics
        let flattened = matrix.iter().copied().collect::<Array1<F>>();
        let overall_mean = self.mean_parallel_enhanced(&flattened.view())?;
        let overall_var = self.variance_parallel_enhanced(&flattened.view(), 1)?;

        Ok(MatrixParallelResult {
            row_means: Array1::from_vec(row_means),
            row_vars: Array1::from_vec(row_vars),
            col_means: Array1::from_vec(col_means),
            col_vars: Array1::from_vec(col_vars),
            overall_mean,
            overall_var,
            shape: (rows, cols),
        })
    }

    /// Parallel quantile computation using parallel sorting
    pub fn quantiles_parallel(
        &self,
        data: &ArrayView1<F>,
        quantiles: &[F],
    ) -> StatsResult<Array1<F>> {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        for &q in quantiles {
            if q < F::zero() || q > F::one() {
                return Err(StatsError::InvalidArgument(
                    "Quantiles must be in [0, 1]".to_string(),
                ));
            }
        }

        // Create owned copy for sorting
        let mut sorteddata = data.to_owned();

        // Use parallel sorting if data is large enough
        if sorteddata.len() >= self.config.min_chunksize {
            sorteddata
                .as_slice_mut()
                .unwrap()
                .par_sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            sorteddata
                .as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        // Compute quantiles
        let n = sorteddata.len();
        let results = quantiles
            .iter()
            .map(|&q| {
                let index = (q * F::from(n - 1).unwrap()).to_f64().unwrap();
                let lower = index.floor() as usize;
                let upper = index.ceil() as usize;
                let weight = F::from(index - index.floor()).unwrap();

                if lower == upper {
                    sorteddata[lower]
                } else {
                    sorteddata[lower] * (F::one() - weight) + sorteddata[upper] * weight
                }
            })
            .collect::<Vec<F>>();

        Ok(Array1::from_vec(results))
    }

    /// Helper: Calculate optimal chunk size
    fn calculate_optimal_chunksize(&self, datalen: usize) -> usize {
        let num_threads = self.config.num_threads.unwrap_or_else(num_cpus::get);
        let ideal_chunks = num_threads * 2; // Allow for load balancing
        let chunksize = (datalen / ideal_chunks).max(self.config.min_chunksize);
        chunksize.min(datalen)
    }

    /// Helper: Compute correlation coefficient
    fn correlation_coefficient(
        &self,
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        mean_x: F,
        mean_y: F,
    ) -> StatsResult<F> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(
                "Arrays must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n < 2 {
            return Ok(F::zero());
        }

        let chunksize = self.calculate_optimal_chunksize(n);
        let result = parallel_map_reduce_indexed(
            0..n,
            chunksize,
            |indices| {
                let mut sum_xy = F::zero();
                let mut sum_x2 = F::zero();
                let mut sum_y2 = F::zero();

                for &i in indices {
                    let dx = x[i] - mean_x;
                    let dy = y[i] - mean_y;
                    sum_xy = sum_xy + dx * dy;
                    sum_x2 = sum_x2 + dx * dx;
                    sum_y2 = sum_y2 + dy * dy;
                }

                (sum_xy, sum_x2, sum_y2)
            },
            |(xy1, x2_1, y2_1), (xy2, x2_2, y2_2)| (xy1 + xy2, x2_1 + x2_2, y2_1 + y2_2),
        );

        let (sum_xy, sum_x2, sum_y2) = result;
        let denom = (sum_x2 * sum_y2).sqrt();

        if denom > F::zero() {
            Ok(sum_xy / denom)
        } else {
            Ok(F::zero())
        }
    }
}

/// Result structure for parallel matrix operations
#[derive(Debug, Clone)]
pub struct MatrixParallelResult<F> {
    pub row_means: Array1<F>,
    pub row_vars: Array1<F>,
    pub col_means: Array1<F>,
    pub col_vars: Array1<F>,
    pub overall_mean: F,
    pub overall_var: F,
    pub shape: (usize, usize),
}

/// High-level convenience functions
#[allow(dead_code)]
pub fn mean_parallel_advanced<F>(data: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>
        + num_traits::FromPrimitive,
{
    let processor = EnhancedParallelProcessor::<F>::new();
    processor.mean_parallel_enhanced(data)
}

#[allow(dead_code)]
pub fn variance_parallel_advanced<F>(data: &ArrayView1<F>, ddof: usize) -> StatsResult<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>
        + num_traits::FromPrimitive,
{
    let processor = EnhancedParallelProcessor::<F>::new();
    processor.variance_parallel_enhanced(data, ddof)
}

#[allow(dead_code)]
pub fn correlation_matrix_parallel_advanced<F>(matrix: &ArrayView2<F>) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>
        + num_traits::FromPrimitive,
{
    let processor = EnhancedParallelProcessor::<F>::new();
    processor.correlation_matrix_parallel(matrix)
}

#[allow(dead_code)]
pub fn bootstrap_parallel_advanced<F>(
    data: &ArrayView1<F>,
    n_bootstrap: usize,
    statistic_fn: impl Fn(&ArrayView1<F>) -> F + Send + Sync,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>
        + num_traits::FromPrimitive,
{
    let processor = EnhancedParallelProcessor::<F>::new();
    processor.bootstrap_parallel_enhanced(data, n_bootstrap, statistic_fn, seed)
}
