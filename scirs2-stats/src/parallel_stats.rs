//! Parallel statistical operations for large datasets
//!
//! This module provides parallel implementations of statistical functions
//! that can significantly improve performance on multi-core systems.

use crate::error::{StatsError, StatsResult};
use crate::error_standardization::ErrorMessages;
use crate::{mean, quantile, var, QuantileInterpolation};
use scirs2_core::ndarray::{s, Array1, ArrayBase, ArrayView1, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::{num_threads, par_chunks, parallel_map, ParallelIterator};

/// Threshold for using parallel operations (number of elements)
const PARALLEL_THRESHOLD: usize = 10_000;

/// Compute mean in parallel for large arrays
///
/// This function automatically switches to parallel computation for arrays
/// larger than the threshold.
///
/// # Arguments
///
/// * `x` - Input data array
///
/// # Returns
///
/// The arithmetic mean of the input data
#[allow(dead_code)]
pub fn mean_parallel<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    let n = x.len();

    if n < PARALLEL_THRESHOLD {
        // Use sequential version for small arrays
        return mean(&x.view());
    }

    // Parallel sum using chunk processing
    let chunksize = (n / num_threads()).max(1000);
    let sum: F = if let Some(slice) = x.as_slice() {
        // Array is contiguous, use efficient parallel processing
        par_chunks(slice, chunksize)
            .map(|chunk| chunk.iter().fold(F::zero(), |acc, &val| acc + val))
            .reduce(|| F::zero(), |a, b| a + b)
    } else {
        // Array is not contiguous, fall back to sequential processing
        x.iter().fold(F::zero(), |acc, &val| acc + val)
    };

    Ok(sum / F::from(n).unwrap())
}

/// Compute variance in parallel for large arrays
///
/// # Arguments
///
/// * `x` - Input data array
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// The variance of the input data
#[allow(dead_code)]
pub fn variance_parallel<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    let n = x.len();
    if n <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom",
        ));
    }

    if n < PARALLEL_THRESHOLD {
        return var(&x.view(), ddof, None);
    }

    // First compute the mean
    let mean_val = mean_parallel(x)?;

    // Parallel computation of sum of squared deviations
    let chunksize = (n / num_threads()).max(1000);
    let sum_sq_dev: F = par_chunks(x.as_slice().unwrap(), chunksize)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&val| {
                    let dev = val - mean_val;
                    dev * dev
                })
                .fold(F::zero(), |acc, val| acc + val)
        })
        .reduce(|| F::zero(), |a, b| a + b);

    Ok(sum_sq_dev / F::from(n - ddof).unwrap())
}

/// Compute multiple quantiles in parallel
///
/// This function efficiently computes multiple quantiles by sorting once
/// and then extracting all requested quantiles.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `quantiles` - Array of quantiles to compute (values between 0 and 1)
/// * `method` - Interpolation method
///
/// # Returns
///
/// Array of computed quantile values
#[allow(dead_code)]
pub fn quantiles_parallel<F, D>(
    x: &ArrayBase<D, Ix1>,
    quantiles: &[F],
    method: QuantileInterpolation,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Send + Sync + std::fmt::Display,
    D: Data<Elem = F> + Sync,
{
    if x.is_empty() {
        return Err(StatsError::invalid_argument(
            "Cannot compute quantiles of empty array",
        ));
    }

    // Validate quantiles
    for &q in quantiles {
        if q < F::zero() || q > F::one() {
            return Err(StatsError::domain("Quantiles must be between 0 and 1"));
        }
    }

    let n = x.len();

    if n < PARALLEL_THRESHOLD || quantiles.len() < 4 {
        // Use sequential version for small arrays or few quantiles
        let mut results = Array1::zeros(quantiles.len());
        for (i, &q) in quantiles.iter().enumerate() {
            results[i] = quantile(&x.view(), q, method)?;
        }
        return Ok(results);
    }

    // Sort array once (this is the expensive operation)
    let mut sorted = x.to_owned();
    sorted
        .as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute all quantiles in parallel
    let results: Vec<F> = parallel_map(quantiles, |&q| {
        // Direct quantile computation on sorted array
        let pos = q * F::from(n - 1).unwrap();
        let idx = pos.floor();
        let frac = pos - idx;

        let idx_usize: usize = NumCast::from(idx).unwrap();

        if frac == F::zero() {
            sorted[idx_usize]
        } else {
            let lower = sorted[idx_usize];
            let upper = sorted[idx_usize + 1];
            lower + frac * (upper - lower)
        }
    });

    Ok(Array1::from_vec(results))
}

/// Compute row-wise statistics in parallel
///
/// This function computes statistics for each row of a 2D array in parallel.
///
/// # Arguments
///
/// * `data` - 2D input array
/// * `stat_fn` - Function to compute statistic for each row
///
/// # Returns
///
/// Array of statistics, one per row
#[allow(dead_code)]
pub fn row_statistics_parallel<F, D, S>(
    data: &ArrayBase<D, Ix2>,
    stat_fn: S,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Send + Sync,
    D: Data<Elem = F> + Sync,
    S: Fn(&ArrayView1<F>) -> StatsResult<F> + Send + Sync + std::fmt::Display,
{
    let nrows = data.nrows();

    if nrows < PARALLEL_THRESHOLD / data.ncols() {
        // Sequential processing for small number of rows
        let mut results = Vec::with_capacity(nrows);
        for i in 0..nrows {
            results.push(stat_fn(&data.slice(s![i, ..]).view())?);
        }
        return Ok(Array1::from_vec(results));
    }

    // Process rows in parallel
    let row_indices: Vec<usize> = (0..nrows).collect();
    let results: Result<Vec<F>, StatsError> =
        parallel_map(&row_indices, |&i| stat_fn(&data.slice(s![i, ..]).view()))
            .into_iter()
            .collect();

    Ok(Array1::from_vec(results?))
}

/// Compute correlation matrix in parallel
///
/// This function computes all pairwise correlations in parallel,
/// which can significantly speed up the computation for large matrices.
///
/// # Arguments
///
/// * `data` - 2D array where columns are variables
///
/// # Returns
///
/// Correlation matrix
#[allow(dead_code)]
pub fn corrcoef_parallel<F, D>(
    data: &ArrayBase<D, Ix2>,
) -> StatsResult<scirs2_core::ndarray::Array2<F>>
where
    F: Float
        + NumCast
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D: Data<Elem = F> + Sync,
{
    use crate::pearson_r;

    let n_vars = data.ncols();

    if n_vars * n_vars < PARALLEL_THRESHOLD {
        // Use sequential version for small matrices
        return crate::corrcoef(&data.view(), "pearson");
    }

    let mut corr_matrix = scirs2_core::ndarray::Array2::zeros((n_vars, n_vars));

    // Generate all pairs (i, j) where i < j
    let pairs: Vec<(usize, usize)> = (0..n_vars)
        .flat_map(|i| ((i + 1)..n_vars).map(move |j| (i, j)))
        .collect();

    // Compute correlations in parallel
    let correlations: Vec<((usize, usize), F)> = parallel_map(&pairs, |&(i, j)| {
        let var_i = data.slice(s![.., i]);
        let var_j = data.slice(s![.., j]);
        let corr = pearson_r(&var_i, &var_j)?;
        Ok(((i, j), corr))
    })
    .into_iter()
    .collect::<StatsResult<Vec<_>>>()?;

    // Fill the correlation matrix
    for i in 0..n_vars {
        corr_matrix[(i, i)] = F::one();
    }

    for ((i, j), corr) in correlations {
        corr_matrix[(i, j)] = corr;
        corr_matrix[(j, i)] = corr; // Symmetric
    }

    Ok(corr_matrix)
}

/// Bootstrap resampling in parallel
///
/// Performs bootstrap resampling with parallel computation of statistics.
///
/// # Arguments
///
/// * `data` - Input data array
/// * `n_samples_` - Number of bootstrap samples
/// * `statistic` - Function to compute statistic on each sample
/// * `seed` - Random seed
///
/// # Returns
///
/// Array of bootstrap statistics
#[allow(dead_code)]
pub fn bootstrap_parallel<F, S>(
    data: &Array1<F>,
    n_samples_: usize,
    statistic: S,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Send + Sync,
    S: Fn(&ArrayBase<scirs2_core::ndarray::ViewRepr<&F>, Ix1>) -> StatsResult<F>
        + Send
        + Sync
        + std::fmt::Display,
{
    use crate::sampling::bootstrap;

    if n_samples_ < PARALLEL_THRESHOLD / data.len() {
        // Sequential bootstrap for small number of _samples
        let samples = bootstrap(&data.view(), n_samples_, seed)?;
        let mut results = Array1::zeros(n_samples_);
        for (i, sample) in samples.outer_iter().enumerate() {
            results[i] = statistic(&sample)?;
        }
        return Ok(results);
    }

    // Generate seeds for parallel random number generation
    let base_seed = seed.unwrap_or(42);
    let seeds: Vec<u64> = (0..n_samples_)
        .map(|i| base_seed.wrapping_add(i as u64))
        .collect();

    // Compute bootstrap statistics in parallel
    let results: Vec<F> = parallel_map(&seeds, |&seed| {
        let sample = bootstrap(&data.view(), 1, Some(seed))?;
        statistic(&sample.slice(s![0, ..]))
    })
    .into_iter()
    .collect::<StatsResult<Vec<_>>>()?;

    Ok(Array1::from_vec(results))
}
