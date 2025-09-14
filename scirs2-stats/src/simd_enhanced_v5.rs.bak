//! Advanced-advanced SIMD optimizations for statistical operations (v5)
//!
//! This module provides state-of-the-art SIMD optimizations for advanced statistical
//! operations, building upon v4 with additional functionality and improved algorithms.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{Float, NumCast, One, Zero};
use rand::Rng;
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use statrs::statistics::Statistics;

/// SIMD-optimized rolling statistics with configurable functions
///
/// Computes rolling statistics (mean, variance, min, max, custom functions) efficiently
/// using SIMD operations and optimized sliding window algorithms.
#[allow(dead_code)]
pub fn rolling_statistics_simd<F>(
    data: &ArrayView1<F>,
    windowsize: usize,
    statistics: &[RollingStatistic],
) -> StatsResult<RollingStatsResult<F>>
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
        + std::iter::Sum<F>,
{
    checkarray_finite(data, "data")?;
    check_positive(windowsize, "windowsize")?;

    if windowsize > data.len() {
        return Err(StatsError::InvalidArgument(
            "Window size cannot be larger than data length".to_string(),
        ));
    }

    let n_windows = data.len() - windowsize + 1;
    let mut results = RollingStatsResult::new(n_windows, statistics);

    // Sequential processing for all datasets (parallel version needs different implementation)
    for i in 0..n_windows {
        let window = data.slice(ndarray::s![i..i + windowsize]);
        compute_window_statistics(&window, &statistics, &mut results, i);
    }

    Ok(results)
}

/// Types of rolling statistics that can be computed
#[derive(Debug, Clone, PartialEq)]
pub enum RollingStatistic {
    Mean,
    Variance,
    Min,
    Max,
    Median,
    Skewness,
    Kurtosis,
    Range,
    StandardDeviation,
    MeanAbsoluteDeviation,
}

/// Results container for rolling statistics
#[derive(Debug, Clone)]
pub struct RollingStatsResult<F> {
    pub means: Option<Array1<F>>,
    pub variances: Option<Array1<F>>,
    pub mins: Option<Array1<F>>,
    pub maxs: Option<Array1<F>>,
    pub medians: Option<Array1<F>>,
    pub skewness: Option<Array1<F>>,
    pub kurtosis: Option<Array1<F>>,
    pub ranges: Option<Array1<F>>,
    pub std_devs: Option<Array1<F>>,
    pub mad: Option<Array1<F>>,
}

impl<F: Zero + Clone> RollingStatsResult<F> {
    fn new(nwindows: usize, statistics: &[RollingStatistic]) -> Self {
        let mut result = RollingStatsResult {
            means: None,
            variances: None,
            mins: None,
            maxs: None,
            medians: None,
            skewness: None,
            kurtosis: None,
            ranges: None,
            std_devs: None,
            mad: None,
        };

        for stat in statistics {
            match stat {
                RollingStatistic::Mean => result.means = Some(Array1::zeros(nwindows)),
                RollingStatistic::Variance => result.variances = Some(Array1::zeros(nwindows)),
                RollingStatistic::Min => result.mins = Some(Array1::zeros(nwindows)),
                RollingStatistic::Max => result.maxs = Some(Array1::zeros(nwindows)),
                RollingStatistic::Median => result.medians = Some(Array1::zeros(nwindows)),
                RollingStatistic::Skewness => result.skewness = Some(Array1::zeros(nwindows)),
                RollingStatistic::Kurtosis => result.kurtosis = Some(Array1::zeros(nwindows)),
                RollingStatistic::Range => result.ranges = Some(Array1::zeros(nwindows)),
                RollingStatistic::StandardDeviation => {
                    result.std_devs = Some(Array1::zeros(nwindows))
                }
                RollingStatistic::MeanAbsoluteDeviation => {
                    result.mad = Some(Array1::zeros(nwindows))
                }
            }
        }

        result
    }
}

#[allow(dead_code)]
fn compute_window_statistics<F>(
    window: &ArrayView1<F>,
    statistics: &[RollingStatistic],
    results: &mut RollingStatsResult<F>,
    window_idx: usize,
) where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    let windowsize = window.len();
    let windowsize_f = F::from(windowsize).unwrap();

    // Compute basic statistics that might be needed for derived ones
    let (sum, sum_sq, min_val, max_val) = if windowsize > 16 {
        // SIMD path
        let sum = F::simd_sum(window);
        let sqdata = F::simd_mul(window, window);
        let sum_sq = F::simd_sum(&sqdata.view());
        let min_val = F::simd_min_element(window);
        let max_val = F::simd_max_element(window);
        (sum, sum_sq, min_val, max_val)
    } else {
        // Scalar fallback
        let sum = window.iter().fold(F::zero(), |acc, &x| acc + x);
        let sum_sq = window.iter().fold(F::zero(), |acc, &x| acc + x * x);
        let min_val = window
            .iter()
            .fold(window[0], |acc, &x| if x < acc { x } else { acc });
        let max_val = window
            .iter()
            .fold(window[0], |acc, &x| if x > acc { x } else { acc });
        (sum, sum_sq, min_val, max_val)
    };

    let mean = sum / windowsize_f;
    let variance = if windowsize > 1 {
        let n_minus_1 = F::from(windowsize - 1).unwrap();
        (sum_sq - sum * sum / windowsize_f) / n_minus_1
    } else {
        F::zero()
    };

    // Store requested statistics
    for stat in statistics {
        match stat {
            RollingStatistic::Mean => {
                if let Some(ref mut means) = results.means {
                    means[window_idx] = mean;
                }
            }
            RollingStatistic::Variance => {
                if let Some(ref mut variances) = results.variances {
                    variances[window_idx] = variance;
                }
            }
            RollingStatistic::Min => {
                if let Some(ref mut mins) = results.mins {
                    mins[window_idx] = min_val;
                }
            }
            RollingStatistic::Max => {
                if let Some(ref mut maxs) = results.maxs {
                    maxs[window_idx] = max_val;
                }
            }
            RollingStatistic::Range => {
                if let Some(ref mut ranges) = results.ranges {
                    ranges[window_idx] = max_val - min_val;
                }
            }
            RollingStatistic::StandardDeviation => {
                if let Some(ref mut std_devs) = results.std_devs {
                    std_devs[window_idx] = variance.sqrt();
                }
            }
            RollingStatistic::Median => {
                if let Some(ref mut medians) = results.medians {
                    let mut sorted_window = window.to_owned();
                    sorted_window
                        .as_slice_mut()
                        .unwrap()
                        .sort_by(|a, b| a.partial_cmp(b).unwrap());
                    medians[window_idx] = if windowsize % 2 == 1 {
                        sorted_window[windowsize / 2]
                    } else {
                        let mid1 = sorted_window[windowsize / 2 - 1];
                        let mid2 = sorted_window[windowsize / 2];
                        (mid1 + mid2) / F::from(2.0).unwrap()
                    };
                }
            }
            RollingStatistic::Skewness => {
                if let Some(ref mut skewness) = results.skewness {
                    if variance > F::zero() {
                        let std_dev = variance.sqrt();
                        let skew_sum = if windowsize > 16 {
                            let mean_vec = Array1::from_elem(windowsize, mean);
                            let centered = F::simd_sub(window, &mean_vec.view());
                            let cubed = F::simd_mul(
                                &F::simd_mul(&centered.view(), &centered.view()).view(),
                                &centered.view(),
                            );
                            F::simd_sum(&cubed.view())
                        } else {
                            window
                                .iter()
                                .map(|&x| {
                                    let dev = x - mean;
                                    dev * dev * dev
                                })
                                .fold(F::zero(), |acc, x| acc + x)
                        };
                        skewness[window_idx] = skew_sum / (windowsize_f * std_dev.powi(3));
                    } else {
                        skewness[window_idx] = F::zero();
                    }
                }
            }
            RollingStatistic::Kurtosis => {
                if let Some(ref mut kurtosis) = results.kurtosis {
                    if variance > F::zero() {
                        let kurt_sum = if windowsize > 16 {
                            let mean_vec = Array1::from_elem(windowsize, mean);
                            let centered = F::simd_sub(window, &mean_vec.view());
                            let squared = F::simd_mul(&centered.view(), &centered.view());
                            let fourth = F::simd_mul(&squared.view(), &squared.view());
                            F::simd_sum(&fourth.view())
                        } else {
                            window
                                .iter()
                                .map(|&x| {
                                    let dev = x - mean;
                                    let dev_sq = dev * dev;
                                    dev_sq * dev_sq
                                })
                                .fold(F::zero(), |acc, x| acc + x)
                        };
                        kurtosis[window_idx] = (kurt_sum / (windowsize_f * variance * variance))
                            - F::from(3.0).unwrap();
                    } else {
                        kurtosis[window_idx] = F::zero();
                    }
                }
            }
            RollingStatistic::MeanAbsoluteDeviation => {
                if let Some(ref mut mad) = results.mad {
                    let mad_sum = if windowsize > 16 {
                        let mean_vec = Array1::from_elem(windowsize, mean);
                        let centered = F::simd_sub(window, &mean_vec.view());
                        let abs_centered = F::simd_abs(&centered.view());
                        F::simd_sum(&abs_centered.view())
                    } else {
                        window
                            .iter()
                            .map(|&x| (x - mean).abs())
                            .fold(F::zero(), |acc, x| acc + x)
                    };
                    mad[window_idx] = mad_sum / windowsize_f;
                }
            }
        }
    }
}

/// SIMD-optimized matrix-wise statistical operations
///
/// Computes statistics along specified axes of 2D matrices using SIMD operations
/// and parallel processing for optimal performance.
#[allow(dead_code)]
pub fn matrix_statistics_simd<F>(
    data: &ArrayView2<F>,
    axis: Option<usize>,
    operations: &[MatrixOperation],
) -> StatsResult<MatrixStatsResult<F>>
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
        + std::iter::Sum<F>,
{
    checkarray_finite(data, "data")?;

    let (n_rows, n_cols) = data.dim();

    if n_rows == 0 || n_cols == 0 {
        return Err(StatsError::InvalidArgument(
            "Data matrix cannot be empty".to_string(),
        ));
    }

    match axis {
        None => compute_global_matrix_stats(data, operations),
        Some(0) => compute_column_wise_stats(data, operations),
        Some(1) => compute_row_wise_stats(data, operations),
        Some(_) => Err(StatsError::InvalidArgument(
            "Axis must be None, 0, or 1 for 2D arrays".to_string(),
        )),
    }
}

/// Types of matrix operations that can be computed
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixOperation {
    Mean,
    Variance,
    StandardDeviation,
    Min,
    Max,
    Sum,
    Product,
    Median,
    Quantile(f64),
    L1Norm,
    L2Norm,
    FrobeniusNorm,
}

/// Results container for matrix statistics
#[derive(Debug, Clone)]
pub struct MatrixStatsResult<F> {
    pub means: Option<Array1<F>>,
    pub variances: Option<Array1<F>>,
    pub std_devs: Option<Array1<F>>,
    pub mins: Option<Array1<F>>,
    pub maxs: Option<Array1<F>>,
    pub sums: Option<Array1<F>>,
    pub products: Option<Array1<F>>,
    pub medians: Option<Array1<F>>,
    pub quantiles: Option<Array1<F>>,
    pub l1_norms: Option<Array1<F>>,
    pub l2_norms: Option<Array1<F>>,
    pub frobenius_norm: Option<F>,
}

#[allow(dead_code)]
fn compute_column_wise_stats<F>(
    data: &ArrayView2<F>,
    operations: &[MatrixOperation],
) -> StatsResult<MatrixStatsResult<F>>
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
        + std::iter::Sum<F>,
{
    let (_n_rows, n_cols) = data.dim();
    let mut results = MatrixStatsResult::new_column_wise(n_cols, operations);

    // Process columns sequentially (parallel version would need different data structure)
    for j in 0..n_cols {
        let column = data.column(j);
        compute_column_statistics(&column, operations, &mut results, j);
    }

    Ok(results)
}

#[allow(dead_code)]
fn compute_row_wise_stats<F>(
    data: &ArrayView2<F>,
    operations: &[MatrixOperation],
) -> StatsResult<MatrixStatsResult<F>>
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
        + std::iter::Sum<F>,
{
    let (n_rows, n_cols) = data.dim();
    let mut results = MatrixStatsResult::new_row_wise(n_rows, operations);

    // Process rows sequentially (parallel version would need different data structure)
    for i in 0..n_rows {
        let row = data.row(i);
        compute_row_statistics(&row, operations, &mut results, i);
    }

    Ok(results)
}

#[allow(dead_code)]
fn compute_global_matrix_stats<F>(
    data: &ArrayView2<F>,
    operations: &[MatrixOperation],
) -> StatsResult<MatrixStatsResult<F>>
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
        + std::iter::Sum<F>,
{
    let mut results = MatrixStatsResult::new_global(operations);

    for operation in operations {
        match operation {
            MatrixOperation::FrobeniusNorm => {
                // Flatten to 1D for SIMD operations since simd_mul expects 1D arrays
                let flattened = Array1::from_iter(data.iter().cloned());
                let squared_sum = if flattened.len() > 1000 {
                    // Use sequential chunked processing for large matrices
                    let mut sum = F::zero();
                    let chunksize = 1000;
                    for i in (0..flattened.len()).step_by(chunksize) {
                        let end = (i + chunksize).min(flattened.len());
                        let chunk = flattened.slice(ndarray::s![i..end]);
                        let squared = F::simd_mul(&chunk, &chunk);
                        sum = sum + F::simd_sum(&squared.view());
                    }
                    sum
                } else {
                    let squared = F::simd_mul(&flattened.view(), &flattened.view());
                    F::simd_sum(&squared.view())
                };
                results.frobenius_norm = Some(squared_sum.sqrt());
            }
            _ => {
                // For other operations, flatten and compute
                let flattened = Array1::from_iter(data.iter().cloned());
                compute_vector_operation(&flattened.view(), operation, &mut results, 0);
            }
        }
    }

    Ok(results)
}

#[allow(dead_code)]
fn compute_column_statistics<F>(
    column: &ArrayView1<F>,
    operations: &[MatrixOperation],
    results: &mut MatrixStatsResult<F>,
    col_idx: usize,
) where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    for operation in operations {
        compute_vector_operation(column, operation, results, col_idx);
    }
}

#[allow(dead_code)]
fn compute_row_statistics<F>(
    row: &ArrayView1<F>,
    operations: &[MatrixOperation],
    results: &mut MatrixStatsResult<F>,
    row_idx: usize,
) where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    for operation in operations {
        compute_vector_operation(row, operation, results, row_idx);
    }
}

#[allow(dead_code)]
fn compute_vector_operation<F>(
    data: &ArrayView1<F>,
    operation: &MatrixOperation,
    results: &mut MatrixStatsResult<F>,
    idx: usize,
) where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    let n = data.len();
    let n_f = F::from(n).unwrap();
    let use_simd = n > 16;

    match operation {
        MatrixOperation::Mean => {
            if let Some(ref mut means) = results.means {
                means[idx] = if use_simd {
                    F::simd_sum(data) / n_f
                } else {
                    data.iter().copied().sum::<F>() / n_f
                };
            }
        }
        MatrixOperation::Sum => {
            if let Some(ref mut sums) = results.sums {
                sums[idx] = if use_simd {
                    F::simd_sum(data)
                } else {
                    data.iter().copied().sum::<F>()
                };
            }
        }
        MatrixOperation::Min => {
            if let Some(ref mut mins) = results.mins {
                mins[idx] = if use_simd {
                    F::simd_min_element(&data.view())
                } else {
                    data.iter().copied().fold(data[0], F::min)
                };
            }
        }
        MatrixOperation::Max => {
            if let Some(ref mut maxs) = results.maxs {
                maxs[idx] = if use_simd {
                    F::simd_max_element(&data.view())
                } else {
                    data.iter().copied().fold(data[0], F::max)
                };
            }
        }
        MatrixOperation::Variance => {
            if let Some(ref mut variances) = results.variances {
                if use_simd {
                    let sum = F::simd_sum(data);
                    let mean = sum / n_f;
                    let mean_vec = Array1::from_elem(n, mean);
                    let centered = F::simd_sub(data, &mean_vec.view());
                    let squared = F::simd_mul(&centered.view(), &centered.view());
                    let sum_sq = F::simd_sum(&squared.view());
                    variances[idx] = if n > 1 {
                        sum_sq / F::from(n - 1).unwrap()
                    } else {
                        F::zero()
                    };
                } else {
                    let mean = data.iter().copied().sum::<F>() / n_f;
                    let sum_sq = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>();
                    variances[idx] = if n > 1 {
                        sum_sq / F::from(n - 1).unwrap()
                    } else {
                        F::zero()
                    };
                }
            }
        }
        MatrixOperation::StandardDeviation => {
            if let Some(ref mut std_devs) = results.std_devs {
                // Reuse variance computation logic
                let variance = if use_simd {
                    let sum = F::simd_sum(data);
                    let mean = sum / n_f;
                    let mean_vec = Array1::from_elem(n, mean);
                    let centered = F::simd_sub(data, &mean_vec.view());
                    let squared = F::simd_mul(&centered.view(), &centered.view());
                    let sum_sq = F::simd_sum(&squared.view());
                    if n > 1 {
                        sum_sq / F::from(n - 1).unwrap()
                    } else {
                        F::zero()
                    }
                } else {
                    let mean = data.iter().copied().sum::<F>() / n_f;
                    let sum_sq = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>();
                    if n > 1 {
                        sum_sq / F::from(n - 1).unwrap()
                    } else {
                        F::zero()
                    }
                };
                std_devs[idx] = variance.sqrt();
            }
        }
        MatrixOperation::L1Norm => {
            if let Some(ref mut l1_norms) = results.l1_norms {
                l1_norms[idx] = if use_simd {
                    let absdata = F::simd_abs(data);
                    F::simd_sum(&absdata.view())
                } else {
                    data.iter().map(|&x| x.abs()).sum::<F>()
                };
            }
        }
        MatrixOperation::L2Norm => {
            if let Some(ref mut l2_norms) = results.l2_norms {
                l2_norms[idx] = if use_simd {
                    let squared = F::simd_mul(data, data);
                    F::simd_sum(&squared.view()).sqrt()
                } else {
                    data.iter().map(|&x| x * x).sum::<F>().sqrt()
                };
            }
        }
        MatrixOperation::Product => {
            if let Some(ref mut products) = results.products {
                products[idx] = data.iter().copied().fold(F::one(), |acc, x| acc * x);
            }
        }
        MatrixOperation::Median => {
            if let Some(ref mut medians) = results.medians {
                let mut sorteddata = data.to_owned();
                sorteddata
                    .as_slice_mut()
                    .unwrap()
                    .sort_by(|a, b| a.partial_cmp(b).unwrap());
                medians[idx] = if n % 2 == 1 {
                    sorteddata[n / 2]
                } else {
                    let mid1 = sorteddata[n / 2 - 1];
                    let mid2 = sorteddata[n / 2];
                    (mid1 + mid2) / F::from(2.0).unwrap()
                };
            }
        }
        MatrixOperation::Quantile(q) => {
            if let Some(ref mut quantiles) = results.quantiles {
                let mut sorteddata = data.to_owned();
                sorteddata
                    .as_slice_mut()
                    .unwrap()
                    .sort_by(|a, b| a.partial_cmp(b).unwrap());
                let pos = q * (n - 1) as f64;
                let lower_idx = pos.floor() as usize;
                let upper_idx = (lower_idx + 1).min(n - 1);
                let weight = F::from(pos - lower_idx as f64).unwrap();
                let lower_val = sorteddata[lower_idx];
                let upper_val = sorteddata[upper_idx];
                quantiles[idx] = lower_val + weight * (upper_val - lower_val);
            }
        }
        MatrixOperation::FrobeniusNorm => {
            // This should only be called for global operations
        }
    }
}

impl<F: Zero + Clone> MatrixStatsResult<F> {
    fn new_column_wise(ncols: usize, operations: &[MatrixOperation]) -> Self {
        let mut result = MatrixStatsResult {
            means: None,
            variances: None,
            std_devs: None,
            mins: None,
            maxs: None,
            sums: None,
            products: None,
            medians: None,
            quantiles: None,
            l1_norms: None,
            l2_norms: None,
            frobenius_norm: None,
        };

        for operation in operations {
            Self::allocate_for_operation(&mut result, operation, ncols);
        }

        result
    }

    fn new_row_wise(nrows: usize, operations: &[MatrixOperation]) -> Self {
        let mut result = MatrixStatsResult {
            means: None,
            variances: None,
            std_devs: None,
            mins: None,
            maxs: None,
            sums: None,
            products: None,
            medians: None,
            quantiles: None,
            l1_norms: None,
            l2_norms: None,
            frobenius_norm: None,
        };

        for operation in operations {
            Self::allocate_for_operation(&mut result, operation, nrows);
        }

        result
    }

    fn new_global(operations: &[MatrixOperation]) -> Self {
        let mut result = MatrixStatsResult {
            means: None,
            variances: None,
            std_devs: None,
            mins: None,
            maxs: None,
            sums: None,
            products: None,
            medians: None,
            quantiles: None,
            l1_norms: None,
            l2_norms: None,
            frobenius_norm: None,
        };

        for operation in operations {
            Self::allocate_for_operation(&mut result, operation, 1);
        }

        result
    }

    fn allocate_for_operation(result: &mut Self, operation: &MatrixOperation, size: usize) {
        match operation {
            MatrixOperation::Mean => result.means = Some(Array1::zeros(size)),
            MatrixOperation::Variance => result.variances = Some(Array1::zeros(size)),
            MatrixOperation::StandardDeviation => result.std_devs = Some(Array1::zeros(size)),
            MatrixOperation::Min => result.mins = Some(Array1::zeros(size)),
            MatrixOperation::Max => result.maxs = Some(Array1::zeros(size)),
            MatrixOperation::Sum => result.sums = Some(Array1::zeros(size)),
            MatrixOperation::Product => result.products = Some(Array1::zeros(size)),
            MatrixOperation::Median => result.medians = Some(Array1::zeros(size)),
            MatrixOperation::Quantile(_) => result.quantiles = Some(Array1::zeros(size)),
            MatrixOperation::L1Norm => result.l1_norms = Some(Array1::zeros(size)),
            MatrixOperation::L2Norm => result.l2_norms = Some(Array1::zeros(size)),
            MatrixOperation::FrobeniusNorm => {} // Will be handled separately
        }
    }
}

/// SIMD-optimized bootstrap sampling with confidence intervals
///
/// Performs bootstrap resampling with SIMD-accelerated statistic computation
/// and efficient confidence interval estimation.
#[allow(dead_code)]
pub fn bootstrap_confidence_interval_simd<F>(
    data: &ArrayView1<F>,
    statistic_fn: BootstrapStatistic,
    n_bootstrap: usize,
    confidence_level: f64,
    random_seed: Option<u64>,
) -> StatsResult<BootstrapResult<F>>
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
    checkarray_finite(data, "data")?;
    check_positive(n_bootstrap, "n_bootstrap")?;
    check_probability(confidence_level, "confidence_level")?;

    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    use scirs2_core::random::Random;

    let mut rng = match random_seed {
        Some(_seed) => Random::seed(_seed),
        None => Random::seed(42), // Use default _seed
    };

    let _ndata = data.len();
    let mut bootstrap_stats = Array1::zeros(n_bootstrap);

    // Bootstrap sampling (sequential for thread safety)
    for i in 0..n_bootstrap {
        let bootstrap_sample = generate_bootstrap_sample(data, &mut rng);
        bootstrap_stats[i] = compute_bootstrap_statistic(&bootstrap_sample.view(), &statistic_fn);
    }

    // Sort _bootstrap statistics for confidence interval computation
    bootstrap_stats
        .as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute confidence interval
    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    let lower_bound = bootstrap_stats[lower_idx.min(n_bootstrap - 1)];
    let upper_bound = bootstrap_stats[upper_idx.min(n_bootstrap - 1)];

    // Compute original statistic
    let original_stat = compute_bootstrap_statistic(data, &statistic_fn);

    // Compute _bootstrap statistics
    let bootstrap_mean = bootstrap_stats.mean().unwrap();
    let bootstrap_std = bootstrap_stats.var(F::from(1.0).unwrap()).sqrt(); // ddof=1

    Ok(BootstrapResult {
        original_statistic: original_stat,
        bootstrap_mean,
        bootstrap_std,
        confidence_interval: (lower_bound, upper_bound),
        confidence_level,
        n_bootstrap,
        bootstrap_statistics: bootstrap_stats,
    })
}

/// Types of statistics that can be bootstrapped
#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapStatistic {
    Mean,
    Median,
    StandardDeviation,
    Variance,
    Skewness,
    Kurtosis,
    Min,
    Max,
    Range,
    InterquartileRange,
    Quantile(f64),
}

/// Results from bootstrap confidence interval estimation
#[derive(Debug, Clone)]
pub struct BootstrapResult<F> {
    pub original_statistic: F,
    pub bootstrap_mean: F,
    pub bootstrap_std: F,
    pub confidence_interval: (F, F),
    pub confidence_level: f64,
    pub n_bootstrap: usize,
    pub bootstrap_statistics: Array1<F>,
}

#[allow(dead_code)]
fn generate_bootstrap_sample<F, R>(data: &ArrayView1<F>, rng: &mut R) -> Array1<F>
where
    F: Copy + Zero,
    R: Rng,
{
    let n = data.len();
    let mut sample = Array1::zeros(n);

    for i in 0..n {
        let idx = rng.gen_range(0..n);
        sample[i] = data[idx];
    }

    sample
}

#[allow(dead_code)]
fn compute_bootstrap_statistic<F>(data: &ArrayView1<F>, statistic: &BootstrapStatistic) -> F
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    let n = data.len();
    let n_f = F::from(n).unwrap();
    let use_simd = n > 16;

    match statistic {
        BootstrapStatistic::Mean => {
            if use_simd {
                F::simd_sum(data) / n_f
            } else {
                data.iter().copied().sum::<F>() / n_f
            }
        }
        BootstrapStatistic::StandardDeviation => {
            let variance = if use_simd {
                let sum = F::simd_sum(data);
                let mean = sum / n_f;
                let mean_vec = Array1::from_elem(n, mean);
                let centered = F::simd_sub(data, &mean_vec.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let sum_sq = F::simd_sum(&squared.view());
                if n > 1 {
                    sum_sq / F::from(n - 1).unwrap()
                } else {
                    F::zero()
                }
            } else {
                let mean = data.iter().copied().sum::<F>() / n_f;
                let sum_sq = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>();
                if n > 1 {
                    sum_sq / F::from(n - 1).unwrap()
                } else {
                    F::zero()
                }
            };
            variance.sqrt()
        }
        BootstrapStatistic::Variance => {
            if use_simd {
                let sum = F::simd_sum(data);
                let mean = sum / n_f;
                let mean_vec = Array1::from_elem(n, mean);
                let centered = F::simd_sub(data, &mean_vec.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let sum_sq = F::simd_sum(&squared.view());
                if n > 1 {
                    sum_sq / F::from(n - 1).unwrap()
                } else {
                    F::zero()
                }
            } else {
                let mean = data.iter().copied().sum::<F>() / n_f;
                let sum_sq = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>();
                if n > 1 {
                    sum_sq / F::from(n - 1).unwrap()
                } else {
                    F::zero()
                }
            }
        }
        BootstrapStatistic::Min => {
            if use_simd {
                F::simd_min_element(&data.view())
            } else {
                data.iter().copied().fold(data[0], F::min)
            }
        }
        BootstrapStatistic::Max => {
            if use_simd {
                F::simd_max_element(&data.view())
            } else {
                data.iter().copied().fold(data[0], F::max)
            }
        }
        BootstrapStatistic::Range => {
            let (min_val, max_val) = if use_simd {
                (
                    F::simd_min_element(&data.view()),
                    F::simd_max_element(&data.view()),
                )
            } else {
                let min_val = data.iter().copied().fold(data[0], F::min);
                let max_val = data.iter().copied().fold(data[0], F::max);
                (min_val, max_val)
            };
            max_val - min_val
        }
        BootstrapStatistic::Median => {
            let mut sorteddata = data.to_owned();
            sorteddata
                .as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
            if n % 2 == 1 {
                sorteddata[n / 2]
            } else {
                let mid1 = sorteddata[n / 2 - 1];
                let mid2 = sorteddata[n / 2];
                (mid1 + mid2) / F::from(2.0).unwrap()
            }
        }
        BootstrapStatistic::Quantile(q) => {
            let mut sorteddata = data.to_owned();
            sorteddata
                .as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos = q * (n - 1) as f64;
            let lower_idx = pos.floor() as usize;
            let upper_idx = (lower_idx + 1).min(n - 1);
            let weight = F::from(pos - lower_idx as f64).unwrap();
            let lower_val = sorteddata[lower_idx];
            let upper_val = sorteddata[upper_idx];
            lower_val + weight * (upper_val - lower_val)
        }
        BootstrapStatistic::InterquartileRange => {
            let mut sorteddata = data.to_owned();
            sorteddata
                .as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1_pos = 0.25 * (n - 1) as f64;
            let q3_pos = 0.75 * (n - 1) as f64;

            let q1_lower = q1_pos.floor() as usize;
            let q1_upper = (q1_lower + 1).min(n - 1);
            let q1_weight = F::from(q1_pos - q1_lower as f64).unwrap();
            let q1 =
                sorteddata[q1_lower] + q1_weight * (sorteddata[q1_upper] - sorteddata[q1_lower]);

            let q3_lower = q3_pos.floor() as usize;
            let q3_upper = (q3_lower + 1).min(n - 1);
            let q3_weight = F::from(q3_pos - q3_lower as f64).unwrap();
            let q3 =
                sorteddata[q3_lower] + q3_weight * (sorteddata[q3_upper] - sorteddata[q3_lower]);

            q3 - q1
        }
        BootstrapStatistic::Skewness => {
            let sum = if use_simd {
                F::simd_sum(data)
            } else {
                data.iter().copied().sum::<F>()
            };
            let mean = sum / n_f;

            let (variance, skew_sum) = if use_simd {
                let mean_vec = Array1::from_elem(n, mean);
                let centered = F::simd_sub(data, &mean_vec.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let cubed = F::simd_mul(&squared.view(), &centered.view());
                let variance = F::simd_sum(&squared.view()) / F::from(n - 1).unwrap();
                let skew_sum = F::simd_sum(&cubed.view());
                (variance, skew_sum)
            } else {
                let mut variance_sum = F::zero();
                let mut skew_sum = F::zero();
                for &x in data.iter() {
                    let dev = x - mean;
                    let dev_sq = dev * dev;
                    variance_sum = variance_sum + dev_sq;
                    skew_sum = skew_sum + dev_sq * dev;
                }
                let variance = variance_sum / F::from(n - 1).unwrap();
                (variance, skew_sum)
            };

            if variance > F::zero() {
                let std_dev = variance.sqrt();
                skew_sum / (n_f * std_dev.powi(3))
            } else {
                F::zero()
            }
        }
        BootstrapStatistic::Kurtosis => {
            let sum = if use_simd {
                F::simd_sum(data)
            } else {
                data.iter().copied().sum::<F>()
            };
            let mean = sum / n_f;

            let (variance, kurt_sum) = if use_simd {
                let mean_vec = Array1::from_elem(n, mean);
                let centered = F::simd_sub(data, &mean_vec.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let fourth = F::simd_mul(&squared.view(), &squared.view());
                let variance = F::simd_sum(&squared.view()) / F::from(n - 1).unwrap();
                let kurt_sum = F::simd_sum(&fourth.view());
                (variance, kurt_sum)
            } else {
                let mut variance_sum = F::zero();
                let mut kurt_sum = F::zero();
                for &x in data.iter() {
                    let dev = x - mean;
                    let dev_sq = dev * dev;
                    variance_sum = variance_sum + dev_sq;
                    kurt_sum = kurt_sum + dev_sq * dev_sq;
                }
                let variance = variance_sum / F::from(n - 1).unwrap();
                (variance, kurt_sum)
            };

            if variance > F::zero() {
                (kurt_sum / (n_f * variance * variance)) - F::from(3.0).unwrap()
            } else {
                F::zero()
            }
        }
    }
}

/// SIMD-optimized kernel density estimation
///
/// Computes kernel density estimates using SIMD-accelerated operations
/// for improved performance on large datasets.
#[allow(dead_code)]
pub fn kernel_density_estimation_simd<F>(
    data: &ArrayView1<F>,
    eval_points: &ArrayView1<F>,
    bandwidth: Option<F>,
    kernel: KernelType,
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
        + std::iter::Sum<F>,
{
    checkarray_finite(data, "data")?;
    checkarray_finite(eval_points, "eval_points")?;

    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    let ndata = data.len();
    let n_eval = eval_points.len();

    // Compute bandwidth using Scott's rule if not provided
    let h = match bandwidth {
        Some(bw) => {
            check_positive(bw.to_f64().unwrap(), "bandwidth")?;
            bw
        }
        None => {
            // Scott's rule: h = n^(-1/5) * std(data)
            let std_dev = if ndata > 16 {
                let sum = F::simd_sum(data);
                let mean = sum / F::from(ndata).unwrap();
                let mean_vec = Array1::from_elem(ndata, mean);
                let centered = F::simd_sub(data, &mean_vec.view());
                let squared = F::simd_mul(&centered.view(), &centered.view());
                let variance = F::simd_sum(&squared.view()) / F::from(ndata - 1).unwrap();
                variance.sqrt()
            } else {
                let mean = data.iter().copied().sum::<F>() / F::from(ndata).unwrap();
                let variance = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
                    / F::from(ndata - 1).unwrap();
                variance.sqrt()
            };

            let scott_factor = F::from(ndata as f64).unwrap().powf(F::from(-0.2).unwrap());
            std_dev * scott_factor
        }
    };

    let mut density = Array1::zeros(n_eval);
    let ndata_f = F::from(ndata).unwrap();
    let normalization = F::one() / (ndata_f * h);

    // Sequential processing for all datasets (parallel version has data race issues)
    {
        for (eval_idx, &eval_point) in eval_points.iter().enumerate() {
            let mut density_sum = F::zero();

            if ndata > 16 {
                // SIMD path
                let eval_vec = Array1::from_elem(ndata, eval_point);
                let differences = F::simd_sub(data, &eval_vec.view());
                let h_vec = Array1::from_elem(ndata, h);
                let standardized = F::simd_div(&differences.view(), &h_vec.view());

                for &z in standardized.iter() {
                    density_sum = density_sum + kernel_function(z, &kernel);
                }
            } else {
                // Scalar fallback
                for &data_point in data.iter() {
                    let z = (data_point - eval_point) / h;
                    density_sum = density_sum + kernel_function(z, &kernel);
                }
            }

            density[eval_idx] = density_sum * normalization;
        }
    }

    Ok(density)
}

/// Types of kernel functions for density estimation
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
    Uniform,
    Triangle,
    Biweight,
    Triweight,
    Cosine,
}

#[allow(dead_code)]
fn kernel_function<F>(z: F, kernel: &KernelType) -> F
where
    F: Float + NumCast + std::fmt::Display,
{
    match kernel {
        KernelType::Gaussian => {
            // (1/√(2π)) * exp(-z²/2)
            let sqrt_2pi = F::from(2.5066282746310005).unwrap(); // √(2π)
            let z_squared = z * z;
            let exp_term = (-z_squared / F::from(2.0).unwrap()).exp();
            exp_term / sqrt_2pi
        }
        KernelType::Epanechnikov => {
            // (3/4) * (1 - z²) for |z| ≤ 1, 0 otherwise
            let abs_z = z.abs();
            if abs_z <= F::one() {
                let z_squared = z * z;
                F::from(0.75).unwrap() * (F::one() - z_squared)
            } else {
                F::zero()
            }
        }
        KernelType::Uniform => {
            // 1/2 for |z| ≤ 1, 0 otherwise
            let abs_z = z.abs();
            if abs_z <= F::one() {
                F::from(0.5).unwrap()
            } else {
                F::zero()
            }
        }
        KernelType::Triangle => {
            // (1 - |z|) for |z| ≤ 1, 0 otherwise
            let abs_z = z.abs();
            if abs_z <= F::one() {
                F::one() - abs_z
            } else {
                F::zero()
            }
        }
        KernelType::Biweight => {
            // (15/16) * (1 - z²)² for |z| ≤ 1, 0 otherwise
            let abs_z = z.abs();
            if abs_z <= F::one() {
                let z_squared = z * z;
                let term = F::one() - z_squared;
                F::from(15.0 / 16.0).unwrap() * term * term
            } else {
                F::zero()
            }
        }
        KernelType::Triweight => {
            // (35/32) * (1 - z²)³ for |z| ≤ 1, 0 otherwise
            let abs_z = z.abs();
            if abs_z <= F::one() {
                let z_squared = z * z;
                let term = F::one() - z_squared;
                F::from(35.0 / 32.0).unwrap() * term * term * term
            } else {
                F::zero()
            }
        }
        KernelType::Cosine => {
            // (π/4) * cos(πz/2) for |z| ≤ 1, 0 otherwise
            let abs_z = z.abs();
            if abs_z <= F::one() {
                let pi = F::from(std::f64::consts::PI).unwrap();
                let pi_4 = pi / F::from(4.0).unwrap();
                let pi_z_2 = pi * z / F::from(2.0).unwrap();
                pi_4 * pi_z_2.cos()
            } else {
                F::zero()
            }
        }
    }
}
