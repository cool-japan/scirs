//! Window-based aggregation features for time series analysis
//!
//! This module provides comprehensive sliding window analysis including multi-scale
//! statistical features, cross-window correlations, change detection, rolling statistics,
//! technical indicators, and normalized features for financial and statistical analysis.

use ndarray::{s, Array1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::WindowConfig;
use super::utils::{calculate_pearson_correlation, linear_fit};
use crate::error::Result;

/// Window-based aggregation features for time series analysis
///
/// This struct contains comprehensive features computed over sliding windows
/// of various sizes, enabling multi-scale statistical analysis.
#[derive(Debug, Clone)]
pub struct WindowBasedFeatures<F> {
    /// Features from small windows (high temporal resolution)
    pub small_window_features: WindowFeatures<F>,
    /// Features from medium windows (balanced resolution)
    pub medium_window_features: WindowFeatures<F>,
    /// Features from large windows (low temporal resolution)
    pub large_window_features: WindowFeatures<F>,
    /// Multi-scale variance features
    pub multi_scale_variance: Vec<F>,
    /// Multi-scale trend features
    pub multi_scale_trends: Vec<F>,
    /// Cross-window correlation features
    pub cross_window_correlations: CrossWindowFeatures<F>,
    /// Window-based change detection features
    pub change_detection_features: ChangeDetectionFeatures<F>,
    /// Rolling aggregation statistics
    pub rolling_statistics: RollingStatistics<F>,
}

impl<F> Default for WindowBasedFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            small_window_features: WindowFeatures::default(),
            medium_window_features: WindowFeatures::default(),
            large_window_features: WindowFeatures::default(),
            multi_scale_variance: Vec::new(),
            multi_scale_trends: Vec::new(),
            cross_window_correlations: CrossWindowFeatures::default(),
            change_detection_features: ChangeDetectionFeatures::default(),
            rolling_statistics: RollingStatistics::default(),
        }
    }
}

/// Features computed over a specific window size
#[derive(Debug, Clone)]
pub struct WindowFeatures<F> {
    /// Window size used for computation
    pub window_size: usize,
    /// Rolling means across all windows
    pub rolling_means: Vec<F>,
    /// Rolling standard deviations
    pub rolling_stds: Vec<F>,
    /// Rolling minimums
    pub rolling_mins: Vec<F>,
    /// Rolling maximums
    pub rolling_maxs: Vec<F>,
    /// Rolling medians
    pub rolling_medians: Vec<F>,
    /// Rolling skewness values
    pub rolling_skewness: Vec<F>,
    /// Rolling kurtosis values
    pub rolling_kurtosis: Vec<F>,
    /// Rolling quantiles (25%, 75%)
    pub rolling_quantiles: Vec<(F, F)>,
    /// Rolling ranges (max - min)
    pub rolling_ranges: Vec<F>,
    /// Rolling coefficient of variation
    pub rolling_cv: Vec<F>,
    /// Summary statistics of rolling features
    pub summary_stats: WindowSummaryStats<F>,
}

impl<F> Default for WindowFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            window_size: 0,
            rolling_means: Vec::new(),
            rolling_stds: Vec::new(),
            rolling_mins: Vec::new(),
            rolling_maxs: Vec::new(),
            rolling_medians: Vec::new(),
            rolling_skewness: Vec::new(),
            rolling_kurtosis: Vec::new(),
            rolling_quantiles: Vec::new(),
            rolling_ranges: Vec::new(),
            rolling_cv: Vec::new(),
            summary_stats: WindowSummaryStats::default(),
        }
    }
}

/// Summary statistics of rolling window features
#[derive(Debug, Clone)]
pub struct WindowSummaryStats<F> {
    /// Mean of rolling means
    pub mean_of_means: F,
    /// Standard deviation of rolling means
    pub std_of_means: F,
    /// Mean of rolling standard deviations
    pub mean_of_stds: F,
    /// Standard deviation of rolling standard deviations
    pub std_of_stds: F,
    /// Maximum range observed
    pub max_range: F,
    /// Minimum range observed
    pub min_range: F,
    /// Mean range
    pub mean_range: F,
    /// Trend in rolling means (slope)
    pub trend_in_means: F,
    /// Trend in rolling standard deviations
    pub trend_in_stds: F,
    /// Variability index (CV of CVs)
    pub variability_index: F,
}

impl<F> Default for WindowSummaryStats<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            mean_of_means: F::zero(),
            std_of_means: F::zero(),
            mean_of_stds: F::zero(),
            std_of_stds: F::zero(),
            max_range: F::zero(),
            min_range: F::zero(),
            mean_range: F::zero(),
            trend_in_means: F::zero(),
            trend_in_stds: F::zero(),
            variability_index: F::zero(),
        }
    }
}

/// Cross-window analysis features
#[derive(Debug, Clone)]
pub struct CrossWindowFeatures<F> {
    /// Correlation between small and medium window means
    pub small_medium_correlation: F,
    /// Correlation between medium and large window means
    pub medium_large_correlation: F,
    /// Correlation between small and large window means
    pub small_large_correlation: F,
    /// Phase difference between different window scales
    pub scale_phase_differences: Vec<F>,
    /// Cross-scale consistency measure
    pub cross_scale_consistency: F,
    /// Multi-scale coherence
    pub multi_scale_coherence: F,
}

impl<F> Default for CrossWindowFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            small_medium_correlation: F::zero(),
            medium_large_correlation: F::zero(),
            small_large_correlation: F::zero(),
            scale_phase_differences: Vec::new(),
            cross_scale_consistency: F::zero(),
            multi_scale_coherence: F::zero(),
        }
    }
}

/// Change detection features from window analysis
#[derive(Debug, Clone)]
pub struct ChangeDetectionFeatures<F> {
    /// Number of significant mean changes detected
    pub mean_change_points: usize,
    /// Number of significant variance changes detected
    pub variance_change_points: usize,
    /// CUSUM (Cumulative Sum) statistics for mean changes
    pub cusum_mean_changes: Vec<F>,
    /// CUSUM statistics for variance changes
    pub cusum_variance_changes: Vec<F>,
    /// Maximum CUSUM value for mean
    pub max_cusum_mean: F,
    /// Maximum CUSUM value for variance
    pub max_cusum_variance: F,
    /// Window-based stability measure
    pub stability_measure: F,
    /// Relative change magnitude
    pub relative_change_magnitude: F,
}

impl<F> Default for ChangeDetectionFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            mean_change_points: 0,
            variance_change_points: 0,
            cusum_mean_changes: Vec::new(),
            cusum_variance_changes: Vec::new(),
            max_cusum_mean: F::zero(),
            max_cusum_variance: F::zero(),
            stability_measure: F::zero(),
            relative_change_magnitude: F::zero(),
        }
    }
}

/// Rolling aggregation statistics
#[derive(Debug, Clone)]
pub struct RollingStatistics<F> {
    /// Exponentially weighted moving average (EWMA)
    pub ewma: Vec<F>,
    /// Exponentially weighted moving variance
    pub ewmv: Vec<F>,
    /// Bollinger band features (upper, lower, width)
    pub bollinger_bands: BollingerBandFeatures<F>,
    /// Moving average convergence divergence (MACD) features
    pub macd_features: MACDFeatures<F>,
    /// Relative strength index (RSI) over windows
    pub rsi_values: Vec<F>,
    /// Z-score normalized rolling features
    pub normalized_features: NormalizedRollingFeatures<F>,
}

impl<F> Default for RollingStatistics<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            ewma: Vec::new(),
            ewmv: Vec::new(),
            bollinger_bands: BollingerBandFeatures::default(),
            macd_features: MACDFeatures::default(),
            rsi_values: Vec::new(),
            normalized_features: NormalizedRollingFeatures::default(),
        }
    }
}

/// Bollinger band features
#[derive(Debug, Clone)]
pub struct BollingerBandFeatures<F> {
    /// Upper Bollinger band values
    pub upper_band: Vec<F>,
    /// Lower Bollinger band values
    pub lower_band: Vec<F>,
    /// Band width (upper - lower)
    pub band_width: Vec<F>,
    /// Percentage above upper band
    pub percent_above_upper: F,
    /// Percentage below lower band
    pub percent_below_lower: F,
    /// Mean band width
    pub mean_band_width: F,
    /// Band squeeze periods (low width)
    pub squeeze_periods: usize,
}

impl<F> Default for BollingerBandFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            upper_band: Vec::new(),
            lower_band: Vec::new(),
            band_width: Vec::new(),
            percent_above_upper: F::zero(),
            percent_below_lower: F::zero(),
            mean_band_width: F::zero(),
            squeeze_periods: 0,
        }
    }
}

/// MACD (Moving Average Convergence Divergence) features
#[derive(Debug, Clone)]
pub struct MACDFeatures<F> {
    /// MACD line (fast EMA - slow EMA)
    pub macd_line: Vec<F>,
    /// Signal line (EMA of MACD)
    pub signal_line: Vec<F>,
    /// MACD histogram (MACD - Signal)
    pub histogram: Vec<F>,
    /// Number of bullish crossovers
    pub bullish_crossovers: usize,
    /// Number of bearish crossovers
    pub bearish_crossovers: usize,
    /// Mean histogram value
    pub mean_histogram: F,
    /// MACD divergence measure
    pub divergence_measure: F,
}

impl<F> Default for MACDFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            macd_line: Vec::new(),
            signal_line: Vec::new(),
            histogram: Vec::new(),
            bullish_crossovers: 0,
            bearish_crossovers: 0,
            mean_histogram: F::zero(),
            divergence_measure: F::zero(),
        }
    }
}

/// Normalized rolling features
#[derive(Debug, Clone)]
pub struct NormalizedRollingFeatures<F> {
    /// Z-score normalized rolling means
    pub normalized_means: Vec<F>,
    /// Z-score normalized rolling stds
    pub normalized_stds: Vec<F>,
    /// Percentile rank of rolling values
    pub percentile_ranks: Vec<F>,
    /// Outlier detection based on rolling statistics
    pub outlier_scores: Vec<F>,
    /// Number of rolling outliers detected
    pub outlier_count: usize,
}

impl<F> Default for NormalizedRollingFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            normalized_means: Vec::new(),
            normalized_stds: Vec::new(),
            percentile_ranks: Vec::new(),
            outlier_scores: Vec::new(),
            outlier_count: 0,
        }
    }
}

// =============================================================================
// Main Calculation Function
// =============================================================================

/// Calculate comprehensive window-based features
///
/// This function performs extensive sliding window analysis including multi-scale
/// statistical features, cross-window correlations, change detection, and technical
/// indicators commonly used in financial analysis.
///
/// # Arguments
///
/// * `ts` - Input time series data
/// * `config` - Window analysis configuration
///
/// # Returns
///
/// WindowBasedFeatures containing comprehensive windowed analysis
pub fn calculate_window_based_features<F>(
    ts: &Array1<F>,
    config: &WindowConfig,
) -> Result<WindowBasedFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + ndarray::ScalarOperand + std::iter::Sum,
{
    let n = ts.len();

    // Validate window sizes
    let small_size = config.small_window_size.max(3).min(n / 4);
    let medium_size = config.medium_window_size.max(small_size + 1).min(n / 2);
    let large_size = config.large_window_size.max(medium_size + 1).min(n - 1);

    if n < small_size + 2 {
        return Ok(WindowBasedFeatures::default());
    }

    // Calculate features for each window size
    let small_features = calculate_window_features(ts, small_size)?;
    let medium_features = calculate_window_features(ts, medium_size)?;
    let large_features = calculate_window_features(ts, large_size)?;

    // Multi-scale variance analysis
    let multi_scale_variance =
        calculate_multi_scale_variance(ts, &[small_size, medium_size, large_size])?;

    // Multi-scale trend analysis
    let multi_scale_trends =
        calculate_multi_scale_trends(ts, &[small_size, medium_size, large_size])?;

    // Cross-window correlations
    let cross_correlations = if config.calculate_cross_correlations {
        calculate_cross_window_correlations(&small_features, &medium_features, &large_features)?
    } else {
        CrossWindowFeatures::default()
    };

    // Change detection features
    let change_features = if config.detect_changes {
        calculate_change_detection_features(ts, &medium_features, config)?
    } else {
        ChangeDetectionFeatures::default()
    };

    // Rolling statistics
    let rolling_stats = calculate_rolling_statistics(ts, config)?;

    Ok(WindowBasedFeatures {
        small_window_features: small_features,
        medium_window_features: medium_features,
        large_window_features: large_features,
        multi_scale_variance,
        multi_scale_trends,
        cross_window_correlations: cross_correlations,
        change_detection_features: change_features,
        rolling_statistics: rolling_stats,
    })
}

// =============================================================================
// Window Feature Calculation
// =============================================================================

/// Calculate features for a specific window size
fn calculate_window_features<F>(ts: &Array1<F>, window_size: usize) -> Result<WindowFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < window_size {
        return Ok(WindowFeatures::default());
    }

    let num_windows = n - window_size + 1;
    let mut rolling_means = Vec::with_capacity(num_windows);
    let mut rolling_stds = Vec::with_capacity(num_windows);
    let mut rolling_mins = Vec::with_capacity(num_windows);
    let mut rolling_maxs = Vec::with_capacity(num_windows);
    let mut rolling_medians = Vec::with_capacity(num_windows);
    let mut rolling_skewness = Vec::with_capacity(num_windows);
    let mut rolling_kurtosis = Vec::with_capacity(num_windows);
    let mut rolling_quantiles = Vec::with_capacity(num_windows);
    let mut rolling_ranges = Vec::with_capacity(num_windows);
    let mut rolling_cv = Vec::with_capacity(num_windows);

    // Calculate rolling statistics
    for i in 0..num_windows {
        let window = ts.slice(s![i..i + window_size]);

        // Basic statistics
        let mean = window.sum() / F::from(window_size).unwrap();
        rolling_means.push(mean);

        let variance = window.mapv(|x| (x - mean).powi(2)).sum() / F::from(window_size).unwrap();
        let std = variance.sqrt();
        rolling_stds.push(std);

        let min = window.iter().fold(F::infinity(), |a, &b| a.min(b));
        let max = window.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        rolling_mins.push(min);
        rolling_maxs.push(max);
        rolling_ranges.push(max - min);

        // Coefficient of variation
        let cv = if mean != F::zero() {
            std / mean.abs()
        } else {
            F::zero()
        };
        rolling_cv.push(cv);

        // Median and quantiles
        let mut sorted_window: Vec<F> = window.iter().cloned().collect();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_idx = window_size / 2;
        let median = if window_size % 2 == 0 {
            (sorted_window[median_idx - 1] + sorted_window[median_idx]) / F::from(2.0).unwrap()
        } else {
            sorted_window[median_idx]
        };
        rolling_medians.push(median);

        let q1_idx = window_size / 4;
        let q3_idx = 3 * window_size / 4;
        let q1 = sorted_window[q1_idx];
        let q3 = sorted_window[q3_idx.min(window_size - 1)];
        rolling_quantiles.push((q1, q3));

        // Higher-order moments (skewness and kurtosis)
        if std != F::zero() {
            let sum_cube = window.mapv(|x| ((x - mean) / std).powi(3)).sum();
            let sum_quad = window.mapv(|x| ((x - mean) / std).powi(4)).sum();

            let skewness = sum_cube / F::from(window_size).unwrap();
            let kurtosis = sum_quad / F::from(window_size).unwrap() - F::from(3.0).unwrap();

            rolling_skewness.push(skewness);
            rolling_kurtosis.push(kurtosis);
        } else {
            rolling_skewness.push(F::zero());
            rolling_kurtosis.push(F::zero());
        }
    }

    // Calculate summary statistics
    let summary_stats = calculate_window_summary_stats(
        &rolling_means,
        &rolling_stds,
        &rolling_ranges,
        &rolling_cv,
    )?;

    Ok(WindowFeatures {
        window_size,
        rolling_means,
        rolling_stds,
        rolling_mins,
        rolling_maxs,
        rolling_medians,
        rolling_skewness,
        rolling_kurtosis,
        rolling_quantiles,
        rolling_ranges,
        rolling_cv,
        summary_stats,
    })
}

/// Calculate summary statistics for window features
fn calculate_window_summary_stats<F>(
    rolling_means: &[F],
    rolling_stds: &[F],
    rolling_ranges: &[F],
    rolling_cv: &[F],
) -> Result<WindowSummaryStats<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = rolling_means.len();
    if n == 0 {
        return Ok(WindowSummaryStats::default());
    }

    let n_f = F::from(n).unwrap();

    // Mean and std of rolling means
    let mean_of_means = rolling_means.iter().fold(F::zero(), |acc, &x| acc + x) / n_f;
    let std_of_means = if n > 1 {
        let variance = rolling_means.iter().fold(F::zero(), |acc, &x| {
            acc + (x - mean_of_means) * (x - mean_of_means)
        }) / F::from(n - 1).unwrap();
        variance.sqrt()
    } else {
        F::zero()
    };

    // Mean and std of rolling stds
    let mean_of_stds = rolling_stds.iter().fold(F::zero(), |acc, &x| acc + x) / n_f;
    let std_of_stds = if n > 1 {
        let variance = rolling_stds.iter().fold(F::zero(), |acc, &x| {
            acc + (x - mean_of_stds) * (x - mean_of_stds)
        }) / F::from(n - 1).unwrap();
        variance.sqrt()
    } else {
        F::zero()
    };

    // Range statistics
    let max_range = rolling_ranges
        .iter()
        .fold(F::neg_infinity(), |a, &b| a.max(b));
    let min_range = rolling_ranges.iter().fold(F::infinity(), |a, &b| a.min(b));
    let mean_range = rolling_ranges.iter().fold(F::zero(), |acc, &x| acc + x) / n_f;

    // Trend calculations (linear regression slope)
    let indices: Vec<F> = (0..n).map(|i| F::from(i).unwrap()).collect();
    let (trend_in_means, _) = linear_fit(&indices, rolling_means);
    let (trend_in_stds, _) = linear_fit(&indices, rolling_stds);

    // Variability index (CV of CVs)
    let mean_cv = rolling_cv.iter().fold(F::zero(), |acc, &x| acc + x) / n_f;
    let variability_index = if mean_cv != F::zero() && n > 1 {
        let cv_variance = rolling_cv
            .iter()
            .fold(F::zero(), |acc, &x| acc + (x - mean_cv) * (x - mean_cv))
            / F::from(n - 1).unwrap();
        cv_variance.sqrt() / mean_cv
    } else {
        F::zero()
    };

    Ok(WindowSummaryStats {
        mean_of_means,
        std_of_means,
        mean_of_stds,
        std_of_stds,
        max_range,
        min_range,
        mean_range,
        trend_in_means,
        trend_in_stds,
        variability_index,
    })
}

// =============================================================================
// Multi-Scale Analysis
// =============================================================================

/// Calculate multi-scale variance features
fn calculate_multi_scale_variance<F>(ts: &Array1<F>, window_sizes: &[usize]) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let mut variances = Vec::with_capacity(window_sizes.len());

    for &window_size in window_sizes {
        let mut scale_variances = Vec::new();
        let num_windows = ts.len().saturating_sub(window_size).saturating_add(1);

        for i in 0..num_windows {
            let window = ts.slice(s![i..i + window_size]);
            let mean = window.sum() / F::from(window_size).unwrap();
            let variance =
                window.mapv(|x| (x - mean) * (x - mean)).sum() / F::from(window_size).unwrap();
            scale_variances.push(variance);
        }

        let mean_variance = if !scale_variances.is_empty() {
            scale_variances.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from(scale_variances.len()).unwrap()
        } else {
            F::zero()
        };

        variances.push(mean_variance);
    }

    Ok(variances)
}

/// Calculate multi-scale trend features
fn calculate_multi_scale_trends<F>(ts: &Array1<F>, window_sizes: &[usize]) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let mut trends = Vec::with_capacity(window_sizes.len());

    for &window_size in window_sizes {
        let mut scale_trends = Vec::new();
        let num_windows = ts.len().saturating_sub(window_size).saturating_add(1);

        for i in 0..num_windows {
            let window = ts.slice(s![i..i + window_size]);
            let indices: Vec<F> = (0..window_size).map(|j| F::from(j).unwrap()).collect();
            let values: Vec<F> = window.iter().cloned().collect();
            let (slope, _) = linear_fit(&indices, &values);
            scale_trends.push(slope);
        }

        let mean_trend = if !scale_trends.is_empty() {
            scale_trends.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from(scale_trends.len()).unwrap()
        } else {
            F::zero()
        };

        trends.push(mean_trend);
    }

    Ok(trends)
}

// =============================================================================
// Cross-Window Analysis
// =============================================================================

/// Calculate cross-window correlations
fn calculate_cross_window_correlations<F>(
    small: &WindowFeatures<F>,
    medium: &WindowFeatures<F>,
    large: &WindowFeatures<F>,
) -> Result<CrossWindowFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone + std::iter::Sum + ndarray::ScalarOperand,
{
    // Align arrays by taking common length
    let min_len = small
        .rolling_means
        .len()
        .min(medium.rolling_means.len())
        .min(large.rolling_means.len());

    if min_len < 2 {
        return Ok(CrossWindowFeatures::default());
    }

    // Calculate correlations between different window sizes
    let small_array = Array1::from_vec(small.rolling_means[..min_len].to_vec());
    let medium_array = Array1::from_vec(medium.rolling_means[..min_len].to_vec());
    let large_array = Array1::from_vec(large.rolling_means[..min_len].to_vec());

    let small_medium_correlation = calculate_pearson_correlation(&small_array, &medium_array)?;
    let medium_large_correlation = calculate_pearson_correlation(&medium_array, &large_array)?;
    let small_large_correlation = calculate_pearson_correlation(&small_array, &large_array)?;

    // Calculate phase differences (simplified as mean differences)
    let mut scale_phase_differences = Vec::new();
    for i in 0..min_len {
        let small_val = small.rolling_means[i];
        let medium_val = medium.rolling_means[i];
        let large_val = large.rolling_means[i];

        let phase_diff_sm = (small_val - medium_val).abs();
        let phase_diff_ml = (medium_val - large_val).abs();
        scale_phase_differences.push(phase_diff_sm);
        scale_phase_differences.push(phase_diff_ml);
    }

    // Cross-scale consistency
    let correlations = [
        small_medium_correlation,
        medium_large_correlation,
        small_large_correlation,
    ];
    let mean_correlation =
        correlations.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(3.0).unwrap();
    let cross_scale_consistency = mean_correlation;

    // Multi-scale coherence (average of absolute correlations)
    let multi_scale_coherence =
        correlations.iter().fold(F::zero(), |acc, &x| acc + x.abs()) / F::from(3.0).unwrap();

    Ok(CrossWindowFeatures {
        small_medium_correlation,
        medium_large_correlation,
        small_large_correlation,
        scale_phase_differences,
        cross_scale_consistency,
        multi_scale_coherence,
    })
}

// =============================================================================
// Change Detection
// =============================================================================

/// Calculate change detection features
fn calculate_change_detection_features<F>(
    ts: &Array1<F>,
    window_features: &WindowFeatures<F>,
    config: &WindowConfig,
) -> Result<ChangeDetectionFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < 10 {
        return Ok(ChangeDetectionFeatures::default());
    }

    // CUSUM for mean changes
    let target_mean = ts.sum() / F::from(n).unwrap();
    let mut cusum_mean = F::zero();
    let mut cusum_mean_changes = Vec::new();
    let mut mean_change_points = 0;

    for &value in ts.iter() {
        cusum_mean = cusum_mean + (value - target_mean);
        cusum_mean_changes.push(cusum_mean);

        if cusum_mean.abs() > F::from(config.change_detection_threshold).unwrap() {
            mean_change_points += 1;
            cusum_mean = F::zero(); // Reset after detection
        }
    }

    // CUSUM for variance changes
    let rolling_stds = &window_features.rolling_stds;
    let target_std = if !rolling_stds.is_empty() {
        rolling_stds.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(rolling_stds.len()).unwrap()
    } else {
        F::zero()
    };

    let mut cusum_variance = F::zero();
    let mut cusum_variance_changes = Vec::new();
    let mut variance_change_points = 0;

    for &std_val in rolling_stds.iter() {
        cusum_variance = cusum_variance + (std_val - target_std);
        cusum_variance_changes.push(cusum_variance);

        if cusum_variance.abs() > F::from(config.change_detection_threshold).unwrap() {
            variance_change_points += 1;
            cusum_variance = F::zero(); // Reset after detection
        }
    }

    // Maximum CUSUM values
    let max_cusum_mean = cusum_mean_changes
        .iter()
        .fold(F::neg_infinity(), |a, &b| a.max(b.abs()));
    let max_cusum_variance = cusum_variance_changes
        .iter()
        .fold(F::neg_infinity(), |a, &b| a.max(b.abs()));

    // Stability measure
    let total_changes = mean_change_points + variance_change_points;
    let stability_measure = F::one() - F::from(total_changes).unwrap() / F::from(n).unwrap();

    // Relative change magnitude
    let data_range = ts.iter().fold(F::neg_infinity(), |a, &b| a.max(b))
        - ts.iter().fold(F::infinity(), |a, &b| a.min(b));
    let relative_change_magnitude = if data_range > F::zero() {
        max_cusum_mean / data_range
    } else {
        F::zero()
    };

    Ok(ChangeDetectionFeatures {
        mean_change_points,
        variance_change_points,
        cusum_mean_changes,
        cusum_variance_changes,
        max_cusum_mean,
        max_cusum_variance,
        stability_measure,
        relative_change_magnitude,
    })
}

// =============================================================================
// Rolling Statistics and Technical Indicators
// =============================================================================

/// Calculate rolling statistics including technical indicators
fn calculate_rolling_statistics<F>(
    ts: &Array1<F>,
    config: &WindowConfig,
) -> Result<RollingStatistics<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Calculate EWMA
    let ewma = calculate_ewma(ts, config.ewma_alpha)?;

    // Calculate EWMV
    let ewmv = calculate_ewmv(ts, &ewma, config.ewma_alpha)?;

    // Calculate Bollinger Bands
    let bollinger_bands = calculate_bollinger_bands(ts, config)?;

    // Calculate MACD
    let macd_features = calculate_macd_features(ts, config)?;

    // Calculate RSI
    let rsi_values = calculate_rsi(ts, config.rsi_period)?;

    // Calculate normalized features
    let normalized_features = calculate_normalized_features(ts, config)?;

    Ok(RollingStatistics {
        ewma,
        ewmv,
        bollinger_bands,
        macd_features,
        rsi_values,
        normalized_features,
    })
}

/// Calculate Exponentially Weighted Moving Average
fn calculate_ewma<F>(ts: &Array1<F>, alpha: f64) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Clone,
{
    let alpha_f = F::from(alpha).unwrap();
    let one_minus_alpha = F::one() - alpha_f;
    let mut ewma = Vec::with_capacity(ts.len());

    if ts.is_empty() {
        return Ok(ewma);
    }

    ewma.push(ts[0]);

    for i in 1..ts.len() {
        let new_val = alpha_f * ts[i] + one_minus_alpha * ewma[i - 1];
        ewma.push(new_val);
    }

    Ok(ewma)
}

/// Calculate Exponentially Weighted Moving Variance
fn calculate_ewmv<F>(ts: &Array1<F>, ewma: &[F], alpha: f64) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Clone,
{
    let alpha_f = F::from(alpha).unwrap();
    let one_minus_alpha = F::one() - alpha_f;
    let mut ewmv = Vec::with_capacity(ts.len());

    if ts.is_empty() || ewma.is_empty() {
        return Ok(ewmv);
    }

    ewmv.push(F::zero());

    for i in 1..ts.len() {
        let diff = ts[i] - ewma[i];
        let new_var = alpha_f * diff * diff + one_minus_alpha * ewmv[i - 1];
        ewmv.push(new_var);
    }

    Ok(ewmv)
}

/// Calculate Bollinger Bands
fn calculate_bollinger_bands<F>(
    ts: &Array1<F>,
    config: &WindowConfig,
) -> Result<BollingerBandFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let window_size = config.bollinger_window;
    let multiplier = F::from(config.bollinger_multiplier).unwrap();
    let n = ts.len();

    if n < window_size {
        return Ok(BollingerBandFeatures::default());
    }

    let mut upper_band = Vec::new();
    let mut lower_band = Vec::new();
    let mut band_width = Vec::new();

    // Calculate rolling mean and std for Bollinger Bands
    for i in 0..=(n - window_size) {
        let window = ts.slice(s![i..i + window_size]);
        let mean = window.sum() / F::from(window_size).unwrap();
        let variance =
            window.mapv(|x| (x - mean) * (x - mean)).sum() / F::from(window_size).unwrap();
        let std = variance.sqrt();

        let upper = mean + multiplier * std;
        let lower = mean - multiplier * std;
        let width = upper - lower;

        upper_band.push(upper);
        lower_band.push(lower);
        band_width.push(width);
    }

    // Calculate additional Bollinger Band features
    let mut above_upper = 0;
    let mut below_lower = 0;

    for (i, &value) in ts.iter().enumerate() {
        if i < upper_band.len() {
            if value > upper_band[i] {
                above_upper += 1;
            }
            if value < lower_band[i] {
                below_lower += 1;
            }
        }
    }

    let percent_above_upper = F::from(above_upper).unwrap() / F::from(upper_band.len()).unwrap();
    let percent_below_lower = F::from(below_lower).unwrap() / F::from(lower_band.len()).unwrap();

    let mean_band_width = if !band_width.is_empty() {
        band_width.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(band_width.len()).unwrap()
    } else {
        F::zero()
    };

    // Count squeeze periods (when band width is unusually low)
    let min_width = band_width.iter().fold(F::infinity(), |a, &b| a.min(b));
    let squeeze_threshold = min_width * F::from(1.2).unwrap();
    let squeeze_periods = band_width
        .iter()
        .filter(|&&w| w <= squeeze_threshold)
        .count();

    Ok(BollingerBandFeatures {
        upper_band,
        lower_band,
        band_width,
        percent_above_upper,
        percent_below_lower,
        mean_band_width,
        squeeze_periods,
    })
}

/// Calculate MACD features
fn calculate_macd_features<F>(ts: &Array1<F>, config: &WindowConfig) -> Result<MACDFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let fast_period = config.macd_fast_period;
    let slow_period = config.macd_slow_period;
    let signal_period = config.macd_signal_period;

    // Calculate EMAs
    let fast_alpha = 2.0 / (fast_period as f64 + 1.0);
    let slow_alpha = 2.0 / (slow_period as f64 + 1.0);
    let signal_alpha = 2.0 / (signal_period as f64 + 1.0);

    let fast_ema = calculate_ewma(ts, fast_alpha)?;
    let slow_ema = calculate_ewma(ts, slow_alpha)?;

    // Calculate MACD line
    let macd_line: Vec<F> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(&fast, &slow)| fast - slow)
        .collect();

    // Calculate signal line (EMA of MACD)
    let macd_array = Array1::from_vec(macd_line.clone());
    let signal_line = calculate_ewma(&macd_array, signal_alpha)?;

    // Calculate histogram
    let histogram: Vec<F> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(&macd, &signal)| macd - signal)
        .collect();

    // Count crossovers
    let mut bullish_crossovers = 0;
    let mut bearish_crossovers = 0;

    for i in 1..histogram.len() {
        if histogram[i - 1] <= F::zero() && histogram[i] > F::zero() {
            bullish_crossovers += 1;
        } else if histogram[i - 1] >= F::zero() && histogram[i] < F::zero() {
            bearish_crossovers += 1;
        }
    }

    let mean_histogram = if !histogram.is_empty() {
        histogram.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(histogram.len()).unwrap()
    } else {
        F::zero()
    };

    // Calculate divergence measure (simplified)
    let divergence_measure = histogram.iter().fold(F::zero(), |acc, &x| acc + x.abs())
        / F::from(histogram.len().max(1)).unwrap();

    Ok(MACDFeatures {
        macd_line,
        signal_line,
        histogram,
        bullish_crossovers,
        bearish_crossovers,
        mean_histogram,
        divergence_measure,
    })
}

/// Calculate Relative Strength Index (RSI)
fn calculate_rsi<F>(ts: &Array1<F>, period: usize) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < period + 1 {
        return Ok(Vec::new());
    }

    let mut rsi_values = Vec::new();

    // Calculate price changes
    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..n {
        let change = ts[i] - ts[i - 1];
        if change > F::zero() {
            gains.push(change);
            losses.push(F::zero());
        } else {
            gains.push(F::zero());
            losses.push(-change);
        }
    }

    // Calculate RSI for each window
    for i in 0..=(gains.len() - period) {
        let avg_gain = gains[i..i + period]
            .iter()
            .fold(F::zero(), |acc, &x| acc + x)
            / F::from(period).unwrap();
        let avg_loss = losses[i..i + period]
            .iter()
            .fold(F::zero(), |acc, &x| acc + x)
            / F::from(period).unwrap();

        let rs = if avg_loss != F::zero() {
            avg_gain / avg_loss
        } else {
            F::from(100.0).unwrap()
        };

        let rsi = F::from(100.0).unwrap() - (F::from(100.0).unwrap() / (F::one() + rs));
        rsi_values.push(rsi);
    }

    Ok(rsi_values)
}

/// Calculate normalized features
fn calculate_normalized_features<F>(
    ts: &Array1<F>,
    config: &WindowConfig,
) -> Result<NormalizedRollingFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let window_size = config.normalization_window;
    let n = ts.len();

    if n < window_size {
        return Ok(NormalizedRollingFeatures::default());
    }

    let mut normalized_means = Vec::new();
    let mut normalized_stds = Vec::new();
    let mut percentile_ranks = Vec::new();
    let mut outlier_scores = Vec::new();
    let mut outlier_count = 0;

    for i in 0..=(n - window_size) {
        let window = ts.slice(s![i..i + window_size]);
        let mean = window.sum() / F::from(window_size).unwrap();
        let variance =
            window.mapv(|x| (x - mean) * (x - mean)).sum() / F::from(window_size).unwrap();
        let std = variance.sqrt();

        // Z-score normalization
        let current_value = ts[i + window_size - 1];
        let z_score_mean = if std != F::zero() {
            (current_value - mean) / std
        } else {
            F::zero()
        };
        normalized_means.push(z_score_mean);

        let z_score_std = if mean != F::zero() {
            std / mean.abs()
        } else {
            F::zero()
        };
        normalized_stds.push(z_score_std);

        // Percentile rank
        let rank = window.iter().filter(|&&x| x <= current_value).count();
        let percentile = F::from(rank).unwrap() / F::from(window_size).unwrap();
        percentile_ranks.push(percentile);

        // Outlier detection (values beyond 2 standard deviations)
        let outlier_score = z_score_mean.abs();
        outlier_scores.push(outlier_score);

        if outlier_score > F::from(2.0).unwrap() {
            outlier_count += 1;
        }
    }

    Ok(NormalizedRollingFeatures {
        normalized_means,
        normalized_stds,
        percentile_ranks,
        outlier_scores,
        outlier_count,
    })
}
