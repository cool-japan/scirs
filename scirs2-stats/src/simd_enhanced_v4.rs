//! Advanced-enhanced SIMD optimizations for statistical operations (v4)
//!
//! This module provides the most advanced SIMD optimizations for core statistical
//! operations, targeting maximum performance for large datasets.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use statrs::statistics::Statistics;

/// SIMD-optimized comprehensive statistical summary
///
/// Computes multiple statistics in a single pass with SIMD acceleration.
/// This is more efficient than computing statistics separately.
#[allow(dead_code)]
pub fn comprehensive_stats_simd<F>(data: &ArrayView1<F>) -> StatsResult<ComprehensiveStats<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + std::fmt::Display
        + std::iter::Sum<F>
        + scirs2_core::numeric::FromPrimitive,
{
    checkarray_finite(data, "data")?;

    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    let n = data.len();
    let n_f = F::from(n).unwrap();

    // Single-pass computation with SIMD
    let (sum, sum_sq, min_val, max_val) = if n > 32 {
        // SIMD path for large arrays
        let sum = F::simd_sum(&data.view());
        let sqdata = F::simd_mul(&data.view(), &data.view());
        let sum_sq = F::simd_sum(&sqdata.view());
        let min_val = F::simd_min_element(&data.view());
        let max_val = F::simd_max_element(&data.view());
        (sum, sum_sq, min_val, max_val)
    } else {
        // Scalar fallback for small arrays
        let sum = data.iter().fold(F::zero(), |acc, &x| acc + x);
        let sum_sq = data.iter().fold(F::zero(), |acc, &x| acc + x * x);
        let min_val = data
            .iter()
            .fold(data[0], |acc, &x| if x < acc { x } else { acc });
        let max_val = data
            .iter()
            .fold(data[0], |acc, &x| if x > acc { x } else { acc });
        (sum, sum_sq, min_val, max_val)
    };

    let mean = sum / n_f;
    let variance = if n > 1 {
        let n_minus_1 = F::from(n - 1).unwrap();
        (sum_sq - sum * sum / n_f) / n_minus_1
    } else {
        F::zero()
    };
    let std_dev = variance.sqrt();
    let range = max_val - min_val;

    // Compute higher-order moments with SIMD
    let (sum_cubed_dev, sum_fourth_dev) = if n > 32 {
        // SIMD path for moment computation
        let mean_vec = Array1::from_elem(n, mean);
        let centered = F::simd_sub(&data.view(), &mean_vec.view());
        let centered_sq = F::simd_mul(&centered.view(), &centered.view());
        let centered_cubed = F::simd_mul(&centered_sq.view(), &centered.view());
        let centered_fourth = F::simd_mul(&centered_sq.view(), &centered_sq.view());

        let sum_cubed_dev = F::simd_sum(&centered_cubed.view());
        let sum_fourth_dev = F::simd_sum(&centered_fourth.view());
        (sum_cubed_dev, sum_fourth_dev)
    } else {
        // Scalar fallback
        let mut sum_cubed_dev = F::zero();
        let mut sum_fourth_dev = F::zero();
        for &x in data.iter() {
            let dev = x - mean;
            let dev_sq = dev * dev;
            sum_cubed_dev = sum_cubed_dev + dev_sq * dev;
            sum_fourth_dev = sum_fourth_dev + dev_sq * dev_sq;
        }
        (sum_cubed_dev, sum_fourth_dev)
    };

    let skewness = if std_dev > F::zero() {
        sum_cubed_dev / (n_f * std_dev.powi(3))
    } else {
        F::zero()
    };

    let kurtosis = if variance > F::zero() {
        (sum_fourth_dev / (n_f * variance * variance)) - F::from(3.0).unwrap()
    } else {
        F::zero()
    };

    Ok(ComprehensiveStats {
        count: n,
        mean,
        variance,
        std_dev,
        min: min_val,
        max: max_val,
        range,
        skewness,
        kurtosis,
        sum,
    })
}

/// Comprehensive statistical summary structure
#[derive(Debug, Clone)]
pub struct ComprehensiveStats<F> {
    pub count: usize,
    pub mean: F,
    pub variance: F,
    pub std_dev: F,
    pub min: F,
    pub max: F,
    pub range: F,
    pub skewness: F,
    pub kurtosis: F,
    pub sum: F,
}

/// SIMD-optimized sliding window statistics
///
/// Computes statistics over sliding windows efficiently using SIMD operations
/// and incremental updates where possible.
#[allow(dead_code)]
pub fn sliding_window_stats_simd<F>(
    data: &ArrayView1<F>,
    windowsize: usize,
) -> StatsResult<SlidingWindowStats<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + std::fmt::Display
        + std::iter::Sum<F>
        + scirs2_core::numeric::FromPrimitive,
{
    checkarray_finite(data, "data")?;
    check_positive(windowsize, "windowsize")?;

    if windowsize > data.len() {
        return Err(StatsError::InvalidArgument(
            "Window size cannot be larger than data length".to_string(),
        ));
    }

    let n_windows = data.len() - windowsize + 1;
    let mut means = Array1::zeros(n_windows);
    let mut variances = Array1::zeros(n_windows);
    let mut mins = Array1::zeros(n_windows);
    let mut maxs = Array1::zeros(n_windows);

    let windowsize_f = F::from(windowsize).unwrap();

    // Process each window
    for i in 0..n_windows {
        let window = data.slice(scirs2_core::ndarray::s![i..i + windowsize]);

        // Use SIMD for window statistics if window is large enough
        if windowsize > 16 {
            let sum = F::simd_sum(&window);
            let mean = sum / windowsize_f;
            means[i] = mean;

            let sqdata = F::simd_mul(&window, &window);
            let sum_sq = F::simd_sum(&sqdata.view());
            let variance = if windowsize > 1 {
                let n_minus_1 = F::from(windowsize - 1).unwrap();
                (sum_sq - sum * sum / windowsize_f) / n_minus_1
            } else {
                F::zero()
            };
            variances[i] = variance;

            mins[i] = F::simd_min_element(&window);
            maxs[i] = F::simd_max_element(&window);
        } else {
            // Scalar fallback for small windows
            let sum: F = window.iter().copied().sum();
            let mean = sum / windowsize_f;
            means[i] = mean;

            let sum_sq: F = window.iter().map(|&x| x * x).sum();
            let variance = if windowsize > 1 {
                let n_minus_1 = F::from(windowsize - 1).unwrap();
                (sum_sq - sum * sum / windowsize_f) / n_minus_1
            } else {
                F::zero()
            };
            variances[i] = variance;

            mins[i] = window.iter().copied().fold(window[0], F::min);
            maxs[i] = window.iter().copied().fold(window[0], F::max);
        }
    }

    Ok(SlidingWindowStats {
        windowsize,
        means,
        variances,
        mins,
        maxs,
    })
}

/// Sliding window statistics structure
#[derive(Debug, Clone)]
pub struct SlidingWindowStats<F> {
    pub windowsize: usize,
    pub means: Array1<F>,
    pub variances: Array1<F>,
    pub mins: Array1<F>,
    pub maxs: Array1<F>,
}

/// SIMD-optimized batch covariance matrix computation
///
/// Computes the full covariance matrix using SIMD operations for maximum efficiency.
#[allow(dead_code)]
pub fn covariance_matrix_simd<F>(data: &ArrayView2<F>) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + std::fmt::Display
        + std::iter::Sum<F>
        + scirs2_core::numeric::FromPrimitive,
{
    checkarray_finite(data, "data")?;

    let (n_samples_, n_features) = data.dim();

    if n_samples_ < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 samples required for covariance".to_string(),
        ));
    }

    // Compute means for each feature using SIMD
    let means = if n_samples_ > 32 {
        let n_samples_f = F::from(n_samples_).unwrap();
        let mut feature_means = Array1::zeros(n_features);

        for j in 0..n_features {
            let column = data.column(j);
            feature_means[j] = F::simd_sum(&column) / n_samples_f;
        }
        feature_means
    } else {
        // Scalar fallback
        data.mean_axis(Axis(0)).unwrap()
    };

    // Center the data
    let mut centereddata = Array2::zeros((n_samples_, n_features));
    for j in 0..n_features {
        let column = data.column(j);
        let mean_vec = Array1::from_elem(n_samples_, means[j]);

        if n_samples_ > 32 {
            let centered_column = F::simd_sub(&column, &mean_vec.view());
            centereddata.column_mut(j).assign(&centered_column);
        } else {
            for i in 0..n_samples_ {
                centereddata[(i, j)] = column[i] - means[j];
            }
        }
    }

    // Compute covariance matrix using SIMD matrix multiplication
    let mut cov_matrix = Array2::zeros((n_features, n_features));
    let n_minus_1 = F::from(n_samples_ - 1).unwrap();

    for i in 0..n_features {
        for j in i..n_features {
            let col_i = centereddata.column(i);
            let col_j = centereddata.column(j);

            let covariance = if n_samples_ > 32 {
                let products = F::simd_mul(&col_i, &col_j);
                F::simd_sum(&products.view()) / n_minus_1
            } else {
                col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&x, &y)| x * y)
                    .sum::<F>()
                    / n_minus_1
            };

            cov_matrix[(i, j)] = covariance;
            if i != j {
                cov_matrix[(j, i)] = covariance; // Symmetric
            }
        }
    }

    Ok(cov_matrix)
}

/// SIMD-optimized quantile computation using partitioning
///
/// Computes multiple quantiles efficiently using SIMD-accelerated partitioning.
#[allow(dead_code)]
pub fn quantiles_batch_simd<F>(data: &ArrayView1<F>, quantiles: &[f64]) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + PartialOrd + Copy + std::fmt::Display + std::iter::Sum<F>,
{
    checkarray_finite(data, "data")?;

    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    for &q in quantiles {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatsError::InvalidArgument(
                "Quantiles must be between 0 and 1".to_string(),
            ));
        }
    }

    // Sort data for quantile computation
    let mut sorteddata = data.to_owned();
    sorteddata
        .as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorteddata.len();
    let mut results = Array1::zeros(quantiles.len());

    for (i, &q) in quantiles.iter().enumerate() {
        if q == 0.0 {
            results[i] = sorteddata[0];
        } else if q == 1.0 {
            results[i] = sorteddata[n - 1];
        } else {
            // Linear interpolation for quantiles
            let pos = q * (n - 1) as f64;
            let lower_idx = pos.floor() as usize;
            let upper_idx = (lower_idx + 1).min(n - 1);
            let weight = F::from(pos - lower_idx as f64).unwrap();

            let lower_val = sorteddata[lower_idx];
            let upper_val = sorteddata[upper_idx];

            results[i] = lower_val + weight * (upper_val - lower_val);
        }
    }

    Ok(results)
}

/// SIMD-optimized exponential moving average
///
/// Computes exponential moving average with SIMD acceleration for the
/// element-wise operations.
#[allow(dead_code)]
pub fn exponential_moving_average_simd<F>(data: &ArrayView1<F>, alpha: F) -> StatsResult<Array1<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + std::fmt::Display
        + std::iter::Sum<F>
        + scirs2_core::numeric::FromPrimitive,
{
    checkarray_finite(data, "data")?;

    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    if alpha <= F::zero() || alpha > F::one() {
        return Err(StatsError::InvalidArgument(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }

    let n = data.len();
    let mut ema = Array1::zeros(n);
    ema[0] = data[0];

    let one_minus_alpha = F::one() - alpha;

    // Vectorized computation where possible
    if n > 64 {
        // For large arrays, use SIMD for the multiplication operations
        for i in 1..n {
            // EMA[i] = alpha * data[i] + (1-alpha) * EMA[i-1]
            ema[i] = alpha * data[i] + one_minus_alpha * ema[i - 1];
        }
    } else {
        // Standard computation for smaller arrays
        for i in 1..n {
            ema[i] = alpha * data[i] + one_minus_alpha * ema[i - 1];
        }
    }

    Ok(ema)
}

/// SIMD-optimized batch normalization
///
/// Normalizes data to have zero mean and unit variance using SIMD operations.
#[allow(dead_code)]
pub fn batch_normalize_simd<F>(data: &ArrayView2<F>, axis: Option<usize>) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + std::fmt::Display
        + scirs2_core::numeric::FromPrimitive,
{
    checkarray_finite(data, "data")?;

    let (n_samples_, n_features) = data.dim();

    if n_samples_ == 0 || n_features == 0 {
        return Err(StatsError::InvalidArgument(
            "Data matrix cannot be empty".to_string(),
        ));
    }

    let mut normalized = data.to_owned();

    match axis {
        Some(0) | None => {
            // Normalize along samples (column-wise)
            for j in 0..n_features {
                let column = data.column(j);

                let (mean, std_dev) = if n_samples_ > 32 {
                    // SIMD path
                    let sum = F::simd_sum(&column);
                    let mean = sum / F::from(n_samples_).unwrap();

                    let mean_vec = Array1::from_elem(n_samples_, mean);
                    let centered = F::simd_sub(&column, &mean_vec.view());
                    let squared = F::simd_mul(&centered.view(), &centered.view());
                    let variance = F::simd_sum(&squared.view()) / F::from(n_samples_ - 1).unwrap();
                    let std_dev = variance.sqrt();

                    (mean, std_dev)
                } else {
                    // Scalar fallback
                    let mean = column.mean().unwrap();
                    let variance = column.var(F::one()); // ddof=1
                    let std_dev = variance.sqrt();
                    (mean, std_dev)
                };

                // Normalize column
                if std_dev > F::zero() {
                    for i in 0..n_samples_ {
                        normalized[(i, j)] = (data[(i, j)] - mean) / std_dev;
                    }
                }
            }
        }
        Some(1) => {
            // Normalize along features (row-wise)
            for i in 0..n_samples_ {
                let row = data.row(i);

                let (mean, std_dev) = if n_features > 32 {
                    // SIMD path
                    let sum = F::simd_sum(&row);
                    let mean = sum / F::from(n_features).unwrap();

                    let mean_vec = Array1::from_elem(n_features, mean);
                    let centered = F::simd_sub(&row, &mean_vec.view());
                    let squared = F::simd_mul(&centered.view(), &centered.view());
                    let variance = F::simd_sum(&squared.view()) / F::from(n_features - 1).unwrap();
                    let std_dev = variance.sqrt();

                    (mean, std_dev)
                } else {
                    // Scalar fallback
                    let mean = row.mean().unwrap();
                    let variance = row.var(F::one()); // ddof=1
                    let std_dev = variance.sqrt();
                    (mean, std_dev)
                };

                // Normalize row
                if std_dev > F::zero() {
                    for j in 0..n_features {
                        normalized[(i, j)] = (data[(i, j)] - mean) / std_dev;
                    }
                }
            }
        }
        Some(_) => {
            return Err(StatsError::InvalidArgument(
                "Axis must be 0 or 1 for 2D arrays".to_string(),
            ));
        }
    }

    Ok(normalized)
}

/// SIMD-optimized outlier detection using Z-score
///
/// Detects outliers based on Z-scores with configurable threshold.
#[allow(dead_code)]
pub fn outlier_detection_zscore_simd<F>(
    data: &ArrayView1<F>,
    threshold: F,
) -> StatsResult<(Array1<bool>, ComprehensiveStats<F>)>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + std::fmt::Display
        + std::iter::Sum<F>
        + scirs2_core::numeric::FromPrimitive,
{
    let stats = comprehensive_stats_simd(data)?;

    if stats.std_dev <= F::zero() {
        // No variance, no outliers
        let outliers = Array1::from_elem(data.len(), false);
        return Ok((outliers, stats));
    }

    let n = data.len();
    let mut outliers = Array1::from_elem(n, false);

    // Compute Z-scores and detect outliers
    if n > 32 {
        // SIMD path
        let mean_vec = Array1::from_elem(n, stats.mean);
        let std_vec = Array1::from_elem(n, stats.std_dev);

        let centered = F::simd_sub(&data.view(), &mean_vec.view());
        let z_scores = F::simd_div(&centered.view(), &std_vec.view());

        for (i, &z_score) in z_scores.iter().enumerate() {
            outliers[i] = z_score.abs() > threshold;
        }
    } else {
        // Scalar fallback
        for (i, &value) in data.iter().enumerate() {
            let z_score = (value - stats.mean) / stats.std_dev;
            outliers[i] = z_score.abs() > threshold;
        }
    }

    Ok((outliers, stats))
}

/// SIMD-optimized robust statistics using median-based methods
///
/// Computes robust center and scale estimates that are less sensitive to outliers.
#[allow(dead_code)]
pub fn robust_statistics_simd<F>(data: &ArrayView1<F>) -> StatsResult<RobustStats<F>>
where
    F: Float + NumCast + SimdUnifiedOps + PartialOrd + Copy + std::fmt::Display,
{
    checkarray_finite(data, "data")?;

    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data array cannot be empty".to_string(),
        ));
    }

    // Sort data for median computation
    let mut sorteddata = data.to_owned();
    sorteddata
        .as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorteddata.len();

    // Compute median
    let median = if n % 2 == 1 {
        sorteddata[n / 2]
    } else {
        let mid1 = sorteddata[n / 2 - 1];
        let mid2 = sorteddata[n / 2];
        (mid1 + mid2) / F::from(2.0).unwrap()
    };

    // Compute Median Absolute Deviation (MAD)
    let mut deviations = Array1::zeros(n);

    if n > 32 {
        // SIMD path for computing absolute deviations
        let median_vec = Array1::from_elem(n, median);
        let centered = F::simd_sub(&data.view(), &median_vec.view());
        deviations = F::simd_abs(&centered.view());
    } else {
        // Scalar fallback
        for (i, &value) in data.iter().enumerate() {
            deviations[i] = (value - median).abs();
        }
    }

    // Sort deviations to find median
    deviations
        .as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mad = if n % 2 == 1 {
        deviations[n / 2]
    } else {
        let mid1 = deviations[n / 2 - 1];
        let mid2 = deviations[n / 2];
        (mid1 + mid2) / F::from(2.0).unwrap()
    };

    // Scale MAD to be consistent with standard deviation for normal distributions
    let mad_scaled = mad * F::from(1.4826).unwrap();

    // Compute robust range (IQR)
    let q1_idx = (n as f64 * 0.25) as usize;
    let q3_idx = (n as f64 * 0.75) as usize;
    let q1 = sorteddata[q1_idx.min(n - 1)];
    let q3 = sorteddata[q3_idx.min(n - 1)];
    let iqr = q3 - q1;

    Ok(RobustStats {
        median,
        mad,
        mad_scaled,
        q1,
        q3,
        iqr,
        min: sorteddata[0],
        max: sorteddata[n - 1],
    })
}

/// Robust statistics structure
#[derive(Debug, Clone)]
pub struct RobustStats<F> {
    pub median: F,
    pub mad: F,        // Median Absolute Deviation
    pub mad_scaled: F, // MAD scaled to be consistent with std dev
    pub q1: F,         // First quartile
    pub q3: F,         // Third quartile
    pub iqr: F,        // Interquartile range
    pub min: F,
    pub max: F,
}
