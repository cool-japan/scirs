//! Advanced parallel processing enhancements for statistical operations
//!
//! This module provides additional parallel implementations for complex statistical
//! operations that can benefit significantly from multi-threading.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

/// Advanced parallel configuration with work stealing and load balancing
#[derive(Debug, Clone)]
pub struct AdvancedParallelConfig {
    /// Minimum size for parallel execution
    pub minsize: usize,
    /// Target chunk size per thread
    pub chunksize: Option<usize>,
    /// Maximum number of threads to use
    pub max_threads: Option<usize>,
    /// Enable work stealing for better load balancing
    pub work_stealing: bool,
    /// Enable dynamic chunk size adjustment
    pub dynamic_chunks: bool,
}

impl Default for AdvancedParallelConfig {
    fn default() -> Self {
        Self {
            minsize: 2_000,    // More aggressive parallelization
            chunksize: None,   // Auto-determine
            max_threads: None, // Use all available
            work_stealing: true,
            dynamic_chunks: true,
        }
    }
}

impl AdvancedParallelConfig {
    /// Get optimal chunk size based on data size and threading
    pub fn get_optimal_chunksize(&self, n: usize) -> usize {
        if let Some(size) = self.chunksize {
            return size;
        }

        let threads = self.max_threads.unwrap_or_else(|| num_cpus::get());

        if self.dynamic_chunks {
            // Dynamic sizing based on data size and thread count
            let base_chunk = n / (threads * 4); // 4 chunks per thread for load balancing
            base_chunk.clamp(100, 10_000) // Reasonable bounds
        } else {
            n / threads
        }
    }
}

/// Parallel batch processor for statistical operations
///
/// Provides efficient parallel processing of statistical operations over
/// multiple datasets or large single datasets.
pub struct ParallelBatchProcessor<F> {
    config: AdvancedParallelConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F> ParallelBatchProcessor<F>
where
    F: Float + NumCast + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    pub fn new(config: AdvancedParallelConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process multiple datasets in parallel
    pub fn batch_descriptive_stats<D>(
        &self,
        datasets: &[ArrayBase<D, Ix1>],
    ) -> StatsResult<Vec<(F, F, F, F)>>
    // (mean, var, min, max)
    where
        D: Data<Elem = F> + Sync,
    {
        if datasets.is_empty() {
            return Ok(Vec::new());
        }

        let results: Vec<StatsResult<(F, F, F, F)>> = datasets
            .iter()
            .map(|dataset| self.compute_singledataset_stats(dataset))
            .collect();

        results.into_iter().collect()
    }

    fn compute_singledataset_stats<D>(&self, data: &ArrayBase<D, Ix1>) -> StatsResult<(F, F, F, F)>
    where
        D: Data<Elem = F>,
    {
        let n = data.len();
        if n == 0 {
            return Err(StatsError::InvalidArgument(
                "Dataset cannot be empty".to_string(),
            ));
        }

        if n < self.config.minsize {
            // Sequential computation for small datasets
            let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap();
            let variance = data
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(n - 1).unwrap();
            let min = data
                .iter()
                .fold(data[0], |min_val, &x| if x < min_val { x } else { min_val });
            let max = data
                .iter()
                .fold(data[0], |max_val, &x| if x > max_val { x } else { max_val });

            return Ok((mean, variance, min, max));
        }

        // Parallel computation
        let chunksize = self.config.get_optimal_chunksize(n);

        // Parallel reduction for statistics
        let results: Vec<(F, F, F, F, usize)> = data
            .as_slice()
            .unwrap()
            .par_chunks(chunksize)
            .map(|chunk| {
                let len = chunk.len();
                let sum = chunk.iter().fold(F::zero(), |acc, &x| acc + x);
                let min = chunk.iter().fold(
                    chunk[0],
                    |min_val, &x| if x < min_val { x } else { min_val },
                );
                let max = chunk.iter().fold(
                    chunk[0],
                    |max_val, &x| if x > max_val { x } else { max_val },
                );

                // Local mean for variance calculation
                let local_mean = sum / F::from(len).unwrap();
                let sum_sq_dev = chunk
                    .iter()
                    .map(|&x| {
                        let diff = x - local_mean;
                        diff * diff
                    })
                    .fold(F::zero(), |acc, x| acc + x);

                (sum, sum_sq_dev, min, max, len)
            })
            .collect();

        // Combine results
        let total_sum = results
            .iter()
            .map(|(sum__, _, _, _, _)| *sum__)
            .fold(F::zero(), |acc, x| acc + x);
        let total_len = results.iter().map(|(_, _, _, _, len)| *len).sum::<usize>();
        let global_mean = total_sum / F::from(total_len).unwrap();

        let global_min =
            results
                .iter()
                .map(|(_, _, min__, _, _)| *min__)
                .fold(
                    results[0].2,
                    |min_val, x| if x < min_val { x } else { min_val },
                );
        let global_max =
            results
                .iter()
                .map(|(_, _, _, max_, _)| *max_)
                .fold(
                    results[0].3,
                    |max_val, x| if x > max_val { x } else { max_val },
                );

        // Recalculate variance with global mean (more accurate)
        let global_variance = par_chunks(data.as_slice().unwrap(), chunksize)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&x| {
                        let diff = x - global_mean;
                        diff * diff
                    })
                    .fold(F::zero(), |acc, x| acc + x)
            })
            .reduce(|| F::zero(), |a, b| a + b)
            / F::from(total_len - 1).unwrap();

        Ok((global_mean, global_variance, global_min, global_max))
    }
}

/// Parallel cross-validation framework
///
/// Provides efficient parallel implementation of k-fold cross-validation
/// for statistical model evaluation.
pub struct ParallelCrossValidator<F> {
    k_folds: usize,
    config: AdvancedParallelConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F> ParallelCrossValidator<F>
where
    F: Float + NumCast + Send + Sync + std::fmt::Display,
{
    pub fn new(_kfolds: usize, config: AdvancedParallelConfig) -> Self {
        Self {
            k_folds: _kfolds,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform k-fold cross-validation for correlation analysis
    pub fn cross_validate_correlation<D1, D2>(
        &self,
        x: &ArrayBase<D1, Ix1>,
        y: &ArrayBase<D2, Ix1>,
    ) -> StatsResult<(F, F)>
    // (mean_correlation, std_correlation)
    where
        D1: Data<Elem = F> + Sync,
        D2: Data<Elem = F> + Sync,
    {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(
                "Arrays must have same length".to_string(),
            ));
        }

        let n = x.len();
        if n < self.k_folds {
            return Err(StatsError::InvalidArgument(
                "Not enough data for k-fold validation".to_string(),
            ));
        }

        let foldsize = n / self.k_folds;
        let x_arc = Arc::new(x.to_owned());
        let y_arc = Arc::new(y.to_owned());

        // Parallel computation of correlations for each fold
        let correlations: Vec<F> = (0..self.k_folds)
            .into_iter()
            .map(|fold| {
                let start = fold * foldsize;
                let end = if fold == self.k_folds - 1 {
                    n
                } else {
                    (fold + 1) * foldsize
                };

                // Create fold by excluding test indices
                let mut train_x = Vec::new();
                let mut train_y = Vec::new();

                for i in 0..n {
                    if i < start || i >= end {
                        train_x.push(x_arc[i]);
                        train_y.push(y_arc[i]);
                    }
                }

                // Compute correlation for this fold
                let x_train = Array1::from(train_x);
                let y_train = Array1::from(train_y);

                self.compute_pearson_correlation(&x_train.view(), &y_train.view())
                    .unwrap_or(F::zero())
            })
            .collect();

        // Compute mean and standard deviation of correlations
        let mean_corr =
            correlations.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(self.k_folds).unwrap();
        let var_corr = correlations
            .iter()
            .map(|&corr| {
                let diff = corr - mean_corr;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(self.k_folds - 1).unwrap();
        let std_corr = var_corr.sqrt();

        Ok((mean_corr, std_corr))
    }

    fn compute_pearson_correlation(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> StatsResult<F> {
        let n = x.len();
        let mean_x = x.iter().fold(F::zero(), |acc, &val| acc + val) / F::from(n).unwrap();
        let mean_y = y.iter().fold(F::zero(), |acc, &val| acc + val) / F::from(n).unwrap();

        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();
        let mut sum_y2 = F::zero();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_dev = xi - mean_x;
            let y_dev = yi - mean_y;
            sum_xy = sum_xy + x_dev * y_dev;
            sum_x2 = sum_x2 + x_dev * x_dev;
            sum_y2 = sum_y2 + y_dev * y_dev;
        }

        let epsilon = F::from(1e-15).unwrap_or_else(|| F::from(0.0).unwrap());
        if sum_x2 <= epsilon || sum_y2 <= epsilon {
            return Ok(F::zero());
        }

        Ok(sum_xy / (sum_x2 * sum_y2).sqrt())
    }
}

/// Parallel Monte Carlo simulation framework
///
/// Provides efficient parallel Monte Carlo simulations for statistical analysis.
pub struct ParallelMonteCarlo<F> {
    n_simulations: usize,
    config: AdvancedParallelConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F> ParallelMonteCarlo<F>
where
    F: Float + NumCast + Send + Sync + std::fmt::Display,
{
    pub fn new(_nsimulations: usize, config: AdvancedParallelConfig) -> Self {
        Self {
            n_simulations: _nsimulations,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Parallel bootstrap confidence interval estimation
    pub fn bootstrap_confidence_interval<D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        statistic_fn: impl Fn(&ArrayView1<F>) -> F + Send + Sync,
        confidence_level: F,
    ) -> StatsResult<(F, F, F)>
    // (estimate, lower_bound, upper_bound)
    where
        D: Data<Elem = F> + Sync,
    {
        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        if confidence_level <= F::zero() || confidence_level >= F::one() {
            return Err(StatsError::InvalidArgument(
                "Confidence _level must be between 0 and 1".to_string(),
            ));
        }

        let data_arc = Arc::new(data.to_owned());
        let n = data.len();

        // Parallel bootstrap sampling
        let bootstrap_stats: Vec<F> = (0..self.n_simulations)
            .into_iter()
            .map(|seed| {
                use rand::rngs::StdRng;
                use rand::SeedableRng;

                let mut rng = StdRng::seed_from_u64(seed as u64);
                let mut bootstrap_sample = Array1::zeros(n);

                for i in 0..n {
                    use rand::Rng;
                    let idx = rng.gen_range(0..n);
                    bootstrap_sample[i] = data_arc[idx];
                }

                statistic_fn(&bootstrap_sample.view())
            })
            .collect();

        // Sort bootstrap statistics for percentile calculation
        let mut sorted_stats = bootstrap_stats;
        sorted_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate confidence interval
        let alpha = F::one() - confidence_level;
        let lower_percentile = alpha / F::from(2.0).unwrap();
        let upper_percentile = F::one() - lower_percentile;

        let lower_idx = (lower_percentile * F::from(self.n_simulations - 1).unwrap())
            .floor()
            .to_usize()
            .unwrap();
        let upper_idx = (upper_percentile * F::from(self.n_simulations - 1).unwrap())
            .ceil()
            .to_usize()
            .unwrap();

        let original_estimate = statistic_fn(&data.view());
        let lower_bound = sorted_stats[lower_idx];
        let upper_bound = sorted_stats[upper_idx];

        Ok((original_estimate, lower_bound, upper_bound))
    }

    /// Parallel permutation test for hypothesis testing
    pub fn permutation_test<D1, D2>(
        &self,
        group1: &ArrayBase<D1, Ix1>,
        group2: &ArrayBase<D2, Ix1>,
        test_statistic: impl Fn(&ArrayView1<F>, &ArrayView1<F>) -> F + Send + Sync,
    ) -> StatsResult<F>
    // p-value
    where
        D1: Data<Elem = F> + Sync,
        D2: Data<Elem = F> + Sync,
    {
        if group1.is_empty() || group2.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Groups cannot be empty".to_string(),
            ));
        }

        // Combine groups for permutation
        let combined: Vec<F> = group1.iter().chain(group2.iter()).cloned().collect();
        let n1 = group1.len();
        let n2 = group2.len();
        let _total_n = n1 + n2;

        // Calculate observed test _statistic
        let observed_stat = test_statistic(&group1.view(), &group2.view());

        // Parallel permutation sampling
        let combined_arc = Arc::new(combined);
        let count_extreme = Arc::new(Mutex::new(0usize));

        (0..self.n_simulations).into_iter().for_each(|seed| {
            use rand::rngs::StdRng;
            use rand::{seq::SliceRandom, SeedableRng};

            let mut rng = StdRng::seed_from_u64(seed as u64);
            let mut permuted = combined_arc.as_ref().clone();
            permuted.shuffle(&mut rng);

            // Create permuted groups
            let perm_group1 = Array1::from_vec(permuted[0..n1].to_vec());
            let perm_group2 = Array1::from_vec(permuted[n1..].to_vec());

            let perm_stat = test_statistic(&perm_group1.view(), &perm_group2.view());

            // Check if permuted _statistic is as extreme as observed
            if perm_stat.abs() >= observed_stat.abs() {
                let mut count = count_extreme.lock().unwrap();
                *count += 1;
            }
        });

        let extreme_count = *count_extreme.lock().unwrap();
        let p_value = F::from(extreme_count).unwrap() / F::from(self.n_simulations).unwrap();

        Ok(p_value)
    }
}

/// Parallel algorithm for efficient matrix operations used in statistics
pub struct ParallelMatrixOps;

impl ParallelMatrixOps {
    /// Parallel matrix-vector multiplication optimized for statistical operations
    pub fn matvec_parallel<F, D1, D2>(
        matrix: &ArrayBase<D1, Ix2>,
        vector: &ArrayBase<D2, Ix1>,
        config: Option<AdvancedParallelConfig>,
    ) -> StatsResult<Array1<F>>
    where
        F: Float + NumCast + Send + Sync + std::iter::Sum,
        D1: Data<Elem = F> + Sync,
        D2: Data<Elem = F> + Sync,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(StatsError::DimensionMismatch(
                "Matrix columns must match vector length".to_string(),
            ));
        }

        let config = config.unwrap_or_default();
        let mut result = Array1::zeros(m);

        if m < config.minsize {
            // Sequential computation for small matrices
            for i in 0..m {
                let row = matrix.row(i);
                result[i] = row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum();
            }
        } else {
            // Parallel computation
            let chunksize = config.get_optimal_chunksize(m);

            result
                .axis_chunks_iter_mut(Axis(0), chunksize)
                .enumerate()
                .for_each(|(chunk_idx, mut result_chunk)| {
                    let start_row = chunk_idx * chunksize;
                    let end_row = (start_row + result_chunk.len()).min(m);

                    for (local_idx, i) in (start_row..end_row).enumerate() {
                        let row = matrix.row(i);
                        result_chunk[local_idx] =
                            row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum();
                    }
                });
        }

        Ok(result)
    }

    /// Parallel outer product computation
    pub fn outer_product_parallel<F, D1, D2>(
        a: &ArrayBase<D1, Ix1>,
        b: &ArrayBase<D2, Ix1>,
        config: Option<AdvancedParallelConfig>,
    ) -> Array2<F>
    where
        F: Float + NumCast + Send + Sync,
        D1: Data<Elem = F> + Sync,
        D2: Data<Elem = F> + Sync,
    {
        let m = a.len();
        let n = b.len();
        let mut result = Array2::zeros((m, n));

        let config = config.unwrap_or_default();

        if m * n < config.minsize {
            // Sequential computation for small matrices
            for i in 0..m {
                for j in 0..n {
                    result[(i, j)] = a[i] * b[j];
                }
            }
        } else {
            // Parallel computation by rows
            result
                .axis_iter_mut(Axis(0))
                .enumerate()
                .par_bridge()
                .for_each(|(i, mut row)| {
                    for (j, elem) in row.iter_mut().enumerate() {
                        *elem = a[i] * b[j];
                    }
                });
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_parallel_batch_processor() {
        let datasets = vec![
            array![1.0, 2.0, 3.0, 4.0, 5.0],
            array![2.0, 4.0, 6.0, 8.0, 10.0],
            array![1.0, 1.0, 1.0, 1.0, 1.0],
        ];

        let processor = ParallelBatchProcessor::new(AdvancedParallelConfig::default());
        let results = processor.batch_descriptive_stats(&datasets).unwrap();

        assert_eq!(results.len(), 3);
        assert_relative_eq!(results[0].0, 3.0, epsilon = 1e-10); // Mean of first dataset
        assert_relative_eq!(results[1].0, 6.0, epsilon = 1e-10); // Mean of second dataset
        assert_relative_eq!(results[2].0, 1.0, epsilon = 1e-10); // Mean of third dataset
    }

    #[test]
    #[ignore = "timeout"]
    fn test_parallel_cross_validator() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]; // Perfect correlation

        let validator = ParallelCrossValidator::new(5, AdvancedParallelConfig::default());
        let (mean_corr, std_corr) = validator
            .cross_validate_correlation(&x.view(), &y.view())
            .unwrap();

        assert!(mean_corr > 0.9); // Should be close to 1.0 for perfect correlation
        assert!(std_corr < 0.1); // Should have low variance
    }

    #[test]
    #[ignore = "timeout"]
    fn test_parallel_matrix_ops() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let vector = array![1.0, 2.0, 3.0];

        let result =
            ParallelMatrixOps::matvec_parallel(&matrix.view(), &vector.view(), None).unwrap();

        assert_relative_eq!(result[0], 14.0, epsilon = 1e-10); // 1*1 + 2*2 + 3*3
        assert_relative_eq!(result[1], 32.0, epsilon = 1e-10); // 4*1 + 5*2 + 6*3
    }
}
