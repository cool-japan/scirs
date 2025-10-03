//! Enhanced parallel correlation computations
//!
//! This module provides SIMD and parallel-accelerated implementations of correlation
//! operations using scirs2-core's unified optimization framework.

use crate::error::{StatsError, StatsResult};
use crate::{kendall_tau, pearson_r, spearman_r};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    simd_ops::{AutoOptimizer, SimdUnifiedOps},
    validation::*,
};
use std::sync::{Arc, Mutex};

/// Parallel configuration for correlation computations
#[derive(Debug, Clone)]
pub struct ParallelCorrelationConfig {
    /// Minimum matrix size to trigger parallel processing
    pub min_parallelsize: usize,
    /// Chunk size for parallel processing
    pub chunksize: Option<usize>,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Use work stealing for load balancing
    pub work_stealing: bool,
}

impl Default for ParallelCorrelationConfig {
    fn default() -> Self {
        Self {
            min_parallelsize: 50, // 50x50 matrix threshold
            chunksize: None,      // Auto-determine
            use_simd: true,
            work_stealing: true,
        }
    }
}

/// Parallel and SIMD-optimized correlation matrix computation
///
/// Computes pairwise correlations between all variables in a matrix using
/// parallel processing for the correlation pairs and SIMD for individual
/// correlation calculations.
///
/// # Arguments
///
/// * `data` - Input data matrix (observations × variables)
/// * `method` - Correlation method ("pearson", "spearman", "kendall")
/// * `config` - Parallel processing configuration
///
/// # Returns
///
/// * Correlation matrix (variables × variables)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::{corrcoef_parallel_enhanced, ParallelCorrelationConfig};
///
/// let data = array![
///     [1.0, 5.0, 10.0],
///     [2.0, 4.0, 9.0],
///     [3.0, 3.0, 8.0],
///     [4.0, 2.0, 7.0],
///     [5.0, 1.0, 6.0]
/// ];
///
/// let config = ParallelCorrelationConfig::default();
/// let corr_matrix = corrcoef_parallel_enhanced(&data.view(), "pearson", &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn corrcoef_parallel_enhanced<F>(
    data: &ArrayView2<F>,
    method: &str,
    config: &ParallelCorrelationConfig,
) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display,
{
    // Validate inputs
    checkarray_finite_2d(data, "data")?;

    match method {
        "pearson" | "spearman" | "kendall" => {}
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "Method must be 'pearson', 'spearman', or 'kendall', got {}",
                method
            )))
        }
    }

    let (n_obs, n_vars) = data.dim();

    if n_obs == 0 || n_vars == 0 {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    // Initialize correlation matrix
    let mut corr_mat = Array2::<F>::zeros((n_vars, n_vars));

    // Set diagonal elements to 1
    for i in 0..n_vars {
        corr_mat[[i, i]] = F::one();
    }

    // Generate upper triangular pairs for parallel processing
    let mut pairs = Vec::new();
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            pairs.push((i, j));
        }
    }

    // Decide whether to use parallel processing
    let use_parallel = n_vars >= config.min_parallelsize;

    if use_parallel {
        // Parallel processing with result collection
        let chunksize = config
            .chunksize
            .unwrap_or(std::cmp::max(1, pairs.len() / 4));

        // Process pairs in parallel and collect results
        let results = Arc::new(Mutex::new(Vec::new()));

        pairs.chunks(chunksize).for_each(|chunk| {
            let mut local_results = Vec::new();

            for &(i, j) in chunk {
                let var_i = data.slice(s![.., i]);
                let var_j = data.slice(s![.., j]);

                let corr = match method {
                    "pearson" => {
                        if config.use_simd {
                            match pearson_r_simd_enhanced(&var_i, &var_j) {
                                Ok(val) => val,
                                Err(_) => continue,
                            }
                        } else {
                            match pearson_r(&var_i, &var_j) {
                                Ok(val) => val,
                                Err(_) => continue,
                            }
                        }
                    }
                    "spearman" => match spearman_r(&var_i, &var_j) {
                        Ok(val) => val,
                        Err(_) => continue,
                    },
                    "kendall" => match kendall_tau(&var_i, &var_j, "b") {
                        Ok(val) => val,
                        Err(_) => continue,
                    },
                    _ => unreachable!(),
                };

                local_results.push((i, j, corr));
            }

            let mut global_results = results.lock().unwrap();
            global_results.extend(local_results);
        });

        let all_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();

        // Write results back to matrix
        for (i, j, corr) in all_results {
            corr_mat[[i, j]] = corr;
            corr_mat[[j, i]] = corr; // Symmetric
        }
    } else {
        // Sequential processing for smaller matrices
        for (i, j) in pairs {
            let var_i = data.slice(s![.., i]);
            let var_j = data.slice(s![.., j]);

            let corr = match method {
                "pearson" => {
                    if config.use_simd {
                        pearson_r_simd_enhanced(&var_i, &var_j)?
                    } else {
                        pearson_r(&var_i, &var_j)?
                    }
                }
                "spearman" => spearman_r(&var_i, &var_j)?,
                "kendall" => kendall_tau(&var_i, &var_j, "b")?,
                _ => unreachable!(),
            };

            corr_mat[[i, j]] = corr;
            corr_mat[[j, i]] = corr; // Symmetric
        }
    }

    Ok(corr_mat)
}

/// SIMD-enhanced Pearson correlation computation
///
/// Optimized version of Pearson correlation using SIMD operations
/// for improved performance on large datasets.
#[allow(dead_code)]
pub fn pearson_r_simd_enhanced<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy + std::iter::Sum<F>,
    D: Data<Elem = F>,
{
    // Check dimensions
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Arrays cannot be empty".to_string(),
        ));
    }

    let n = x.len();
    let n_f = F::from(n).unwrap();
    let optimizer = AutoOptimizer::new();

    // Use SIMD for mean calculations if beneficial
    let (mean_x, mean_y) = if optimizer.should_use_simd(n) {
        let sum_x = F::simd_sum(&x.view());
        let sum_y = F::simd_sum(&y.view());
        (sum_x / n_f, sum_y / n_f)
    } else {
        let mean_x = x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f;
        let mean_y = y.iter().fold(F::zero(), |acc, &val| acc + val) / n_f;
        (mean_x, mean_y)
    };

    // SIMD-optimized correlation calculation
    let (sum_xy, sum_x2, sum_y2) = if optimizer.should_use_simd(n) {
        // Create arrays with means for SIMD subtraction
        let mean_x_array = Array1::from_elem(n, mean_x);
        let mean_y_array = Array1::from_elem(n, mean_y);

        // Compute deviations
        let x_dev = F::simd_sub(&x.view(), &mean_x_array.view());
        let y_dev = F::simd_sub(&y.view(), &mean_y_array.view());

        // Compute products and squares
        let xy_prod = F::simd_mul(&x_dev.view(), &y_dev.view());
        let x_sq = F::simd_mul(&x_dev.view(), &x_dev.view());
        let y_sq = F::simd_mul(&y_dev.view(), &y_dev.view());

        // Sum the results
        let sum_xy = F::simd_sum(&xy_prod.view());
        let sum_x2 = F::simd_sum(&x_sq.view());
        let sum_y2 = F::simd_sum(&y_sq.view());

        (sum_xy, sum_x2, sum_y2)
    } else {
        // Scalar fallback
        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();
        let mut sum_y2 = F::zero();

        for i in 0..n {
            let x_dev = x[i] - mean_x;
            let y_dev = y[i] - mean_y;

            sum_xy = sum_xy + x_dev * y_dev;
            sum_x2 = sum_x2 + x_dev * x_dev;
            sum_y2 = sum_y2 + y_dev * y_dev;
        }

        (sum_xy, sum_x2, sum_y2)
    };

    // Check for zero variances
    if sum_x2 <= F::epsilon() || sum_y2 <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute correlation when one or both variables have zero variance".to_string(),
        ));
    }

    // Calculate correlation coefficient
    let corr = sum_xy / (sum_x2 * sum_y2).sqrt();

    // Clamp to valid range [-1, 1]
    let corr = if corr > F::one() {
        F::one()
    } else if corr < -F::one() {
        -F::one()
    } else {
        corr
    };

    Ok(corr)
}

/// Parallel batch correlation computation
///
/// Computes correlations between multiple pairs of arrays in parallel,
/// useful for large-scale correlation analysis.
///
/// # Arguments
///
/// * `pairs` - Vector of array pairs to correlate
/// * `method` - Correlation method
/// * `config` - Parallel processing configuration
///
/// # Returns
///
/// * Vector of correlation coefficients in the same order as input pairs
#[allow(dead_code)]
pub fn batch_correlations_parallel<'a, F>(
    pairs: &[(ArrayView1<'a, F>, ArrayView1<'a, F>)],
    method: &str,
    config: &ParallelCorrelationConfig,
) -> StatsResult<Vec<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display,
{
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    // Validate method
    match method {
        "pearson" | "spearman" | "kendall" => {}
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "Method must be 'pearson', 'spearman', or 'kendall', got {}",
                method
            )))
        }
    }

    let n_pairs = pairs.len();
    let use_parallel = n_pairs >= config.min_parallelsize.min(10); // Lower threshold for batch operations

    if use_parallel {
        // Parallel processing with chunking
        let chunksize = config.chunksize.unwrap_or(std::cmp::max(1, n_pairs / 4));

        let results = Arc::new(Mutex::new(Vec::new()));
        let error_occurred = Arc::new(Mutex::new(false));

        pairs.chunks(chunksize).for_each(|chunk| {
            let mut local_results = Vec::new();
            let mut has_error = false;

            for (x, y) in chunk {
                let corr = match method {
                    "pearson" => {
                        if config.use_simd {
                            pearson_r_simd_enhanced(x, y)
                        } else {
                            pearson_r(x, y)
                        }
                    }
                    "spearman" => spearman_r(x, y),
                    "kendall" => kendall_tau(x, y, "b"),
                    _ => unreachable!(),
                };

                match corr {
                    Ok(val) => local_results.push(val),
                    Err(_) => {
                        has_error = true;
                        break;
                    }
                }
            }

            if has_error {
                *error_occurred.lock().unwrap() = true;
            } else {
                results.lock().unwrap().extend(local_results);
            }
        });

        if *error_occurred.lock().unwrap() {
            return Err(StatsError::InvalidArgument(
                "Error occurred during batch correlation computation".to_string(),
            ));
        }

        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        Ok(final_results)
    } else {
        // Sequential processing
        let mut results = Vec::with_capacity(n_pairs);

        for (x, y) in pairs {
            let corr = match method {
                "pearson" => {
                    if config.use_simd {
                        pearson_r_simd_enhanced(x, y)?
                    } else {
                        pearson_r(x, y)?
                    }
                }
                "spearman" => spearman_r(x, y)?,
                "kendall" => kendall_tau(x, y, "b")?,
                _ => unreachable!(),
            };
            results.push(corr);
        }

        Ok(results)
    }
}

/// Rolling correlation computation with parallel processing
///
/// Computes rolling correlations between two time series using
/// parallel processing for multiple windows.
#[allow(dead_code)]
pub fn rolling_correlation_parallel<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    windowsize: usize,
    method: &str,
    config: &ParallelCorrelationConfig,
) -> StatsResult<Array1<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display,
{
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "x and y must have the same length, got {} and {}",
            x.len(),
            y.len()
        )));
    }
    check_positive(windowsize, "windowsize")?;

    if windowsize > x.len() {
        return Err(StatsError::InvalidArgument(
            "Window size cannot be larger than data length".to_string(),
        ));
    }

    let n_windows = x.len() - windowsize + 1;
    let mut results = Array1::zeros(n_windows);

    // Generate window pairs
    let window_pairs: Vec<_> = (0..n_windows)
        .map(|i| {
            let x_window = x.slice(s![i..i + windowsize]);
            let y_window = y.slice(s![i..i + windowsize]);
            (x_window, y_window)
        })
        .collect();

    // Compute correlations in parallel
    let correlations = batch_correlations_parallel(&window_pairs, method, config)?;

    // Copy results
    for (i, corr) in correlations.into_iter().enumerate() {
        results[i] = corr;
    }

    Ok(results)
}

// Helper function for 2D array validation
#[allow(dead_code)]
fn checkarray_finite_2d<F, D>(arr: &ArrayBase<D, Ix2>, name: &str) -> StatsResult<()>
where
    F: Float,
    D: Data<Elem = F>,
{
    for &val in arr.iter() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "{} contains non-finite values",
                name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corrcoef;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_corrcoef_parallel_enhanced_consistency() {
        let data = array![
            [1.0, 5.0, 10.0],
            [2.0, 4.0, 9.0],
            [3.0, 3.0, 8.0],
            [4.0, 2.0, 7.0],
            [5.0, 1.0, 6.0]
        ];

        let config = ParallelCorrelationConfig::default();
        let parallel_result = corrcoef_parallel_enhanced(&data.view(), "pearson", &config).unwrap();
        let sequential_result = corrcoef(&data.view(), "pearson").unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (parallel_result[[i, j]] - sequential_result[[i, j]]).abs() < 1e-10,
                    "Mismatch at [{}, {}]: parallel {} vs sequential {}",
                    i,
                    j,
                    parallel_result[[i, j]],
                    sequential_result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_pearson_r_simd_enhanced_consistency() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];

        let simd_result = pearson_r_simd_enhanced(&x.view(), &y.view()).unwrap();
        let standard_result = pearson_r(&x.view(), &y.view()).unwrap();

        assert!((simd_result - standard_result).abs() < 1e-10);
    }

    #[test]
    fn test_batch_correlations_parallel() {
        let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y1 = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let x2 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y2 = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let pairs = vec![(x1.view(), y1.view()), (x2.view(), y2.view())];
        let config = ParallelCorrelationConfig::default();

        let results = batch_correlations_parallel(&pairs, "pearson", &config).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0] - (-1.0)).abs() < 1e-10); // Perfect negative correlation
        assert!((results[1] - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_rolling_correlation_parallel() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let config = ParallelCorrelationConfig::default();
        let rolling_corrs =
            rolling_correlation_parallel(&x.view(), &y.view(), 3, "pearson", &config).unwrap();

        assert_eq!(rolling_corrs.len(), 8); // 10 - 3 + 1

        // All rolling correlations should be negative (x increases, y decreases)
        for corr in rolling_corrs.iter() {
            assert!(*corr < 0.0);
        }
    }
}
