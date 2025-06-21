//! Multi-seasonal decomposition methods
//!
//! This module provides implementations for decomposing time series with multiple
//! nested seasonal patterns, including automatic period detection.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::DecompositionModel;
use crate::error::{Result, TimeSeriesError};
use crate::utils::autocorrelation;

/// Result of multi-seasonal decomposition
#[derive(Debug, Clone)]
pub struct MultiSeasonalResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal components (one for each detected period)
    pub seasonal_components: Vec<Array1<F>>,
    /// Combined seasonal component
    pub seasonal: Array1<F>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
    /// Detected periods
    pub periods: Vec<usize>,
}

/// Configuration for multi-seasonal decomposition
#[derive(Debug, Clone)]
pub struct MultiSeasonalConfig {
    /// Model type (additive or multiplicative)
    pub model: DecompositionModel,
    /// Maximum number of seasonal periods to detect
    pub max_periods: usize,
    /// Minimum period length to consider
    pub min_period: usize,
    /// Maximum period length to consider  
    pub max_period: usize,
    /// Threshold for seasonal strength to accept a period
    pub seasonal_strength_threshold: f64,
    /// Maximum iterations for decomposition
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for MultiSeasonalConfig {
    fn default() -> Self {
        Self {
            model: DecompositionModel::Additive,
            max_periods: 3,
            min_period: 4,
            max_period: 0, // Will be set to n/3 if 0
            seasonal_strength_threshold: 0.1,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Performs multi-seasonal decomposition with automatic period detection
///
/// This function decomposes a time series with multiple nested seasonal patterns.
/// It automatically detects the most significant seasonal periods and performs
/// decomposition using an iterative approach.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `config` - Configuration for the decomposition
///
/// # Returns
///
/// * Multi-seasonal decomposition result
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{decompose_multi_seasonal, MultiSeasonalConfig};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0];
/// let config = MultiSeasonalConfig::default();
/// let result = decompose_multi_seasonal(&ts, &config).unwrap();
/// ```
pub fn decompose_multi_seasonal<F>(
    ts: &Array1<F>,
    config: &MultiSeasonalConfig,
) -> Result<MultiSeasonalResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    if n < 12 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 12 observations for multi-seasonal decomposition"
                .to_string(),
        ));
    }

    // Detect seasonal periods automatically
    let mut periods = detect_seasonal_periods(ts, config)?;

    if periods.is_empty() {
        return Err(TimeSeriesError::DecompositionError(
            "No significant seasonal periods detected".to_string(),
        ));
    }

    // Sort periods in ascending order
    periods.sort_unstable();

    // Limit to max_periods
    if periods.len() > config.max_periods {
        periods.truncate(config.max_periods);
    }

    // Initialize components
    let mut trend = Array1::zeros(n);
    let mut seasonal_components = vec![Array1::zeros(n); periods.len()];
    let mut residual = ts.clone();

    // Iterative decomposition
    for _iter in 0..config.max_iterations {
        let old_trend = trend.clone();
        let old_seasonal_components = seasonal_components.clone();

        // Update trend component
        trend = extract_multi_seasonal_trend(&residual, &periods, config)?;

        // Update seasonal components
        let detrended = match config.model {
            DecompositionModel::Additive => ts - &trend,
            DecompositionModel::Multiplicative => {
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    if trend[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    result[i] = ts[i] / trend[i];
                }
                result
            }
        };

        seasonal_components = extract_seasonal_components(&detrended, &periods, config)?;

        // Update residual
        let combined_seasonal = combine_seasonal_components(&seasonal_components, config.model)?;
        residual = match config.model {
            DecompositionModel::Additive => ts - &trend - &combined_seasonal,
            DecompositionModel::Multiplicative => {
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    let denominator = trend[i] * combined_seasonal[i];
                    if denominator == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    result[i] = ts[i] / denominator;
                }
                result
            }
        };

        // Check convergence
        let trend_change = calculate_l2_norm(&(&trend - &old_trend))?;
        let seasonal_change = {
            let mut total_change = F::zero();
            for (old, new) in old_seasonal_components
                .iter()
                .zip(seasonal_components.iter())
            {
                total_change = total_change + calculate_l2_norm(&(new - old))?;
            }
            total_change
        };

        if trend_change < F::from_f64(config.tolerance).unwrap()
            && seasonal_change < F::from_f64(config.tolerance).unwrap()
        {
            break;
        }
    }

    let combined_seasonal = combine_seasonal_components(&seasonal_components, config.model)?;

    Ok(MultiSeasonalResult {
        trend,
        seasonal_components,
        seasonal: combined_seasonal,
        residual,
        original: ts.clone(),
        periods,
    })
}

/// Detects seasonal periods automatically using multiple methods
fn detect_seasonal_periods<F>(ts: &Array1<F>, config: &MultiSeasonalConfig) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let max_period = if config.max_period == 0 {
        n / 3
    } else {
        config.max_period
    };

    let mut period_scores = Vec::new();

    // Method 1: Autocorrelation-based detection
    let acf_periods = detect_periods_acf(ts, config.min_period, max_period)?;
    for period in acf_periods {
        let strength = calculate_seasonal_strength(ts, period, &config.model)?;
        if strength > config.seasonal_strength_threshold {
            period_scores.push((period, strength));
        }
    }

    // Method 2: Periodogram-based detection
    let pgram_periods = detect_periods_periodogram(ts, config.min_period, max_period)?;
    for period in pgram_periods {
        let strength = calculate_seasonal_strength(ts, period, &config.model)?;
        if strength > config.seasonal_strength_threshold {
            // Check if already detected
            if !period_scores.iter().any(|(p, _)| *p == period) {
                period_scores.push((period, strength));
            }
        }
    }

    // Method 3: Multiple autocorrelation peaks
    let multi_acf_periods = detect_multiple_acf_peaks(ts, config.min_period, max_period)?;
    for period in multi_acf_periods {
        let strength = calculate_seasonal_strength(ts, period, &config.model)?;
        if strength > config.seasonal_strength_threshold
            && !period_scores.iter().any(|(p, _)| *p == period)
        {
            period_scores.push((period, strength));
        }
    }

    // Sort by seasonal strength (descending)
    period_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Extract periods and filter out harmonics
    let mut periods = Vec::new();
    for (period, _) in period_scores {
        // Check if this period is not a harmonic of an existing period
        let is_harmonic = periods
            .iter()
            .any(|&existing| period % existing == 0 || existing % period == 0);

        if !is_harmonic {
            periods.push(period);
        }
    }

    Ok(periods)
}

/// Detects periods using autocorrelation function
fn detect_periods_acf<F>(ts: &Array1<F>, min_period: usize, max_period: usize) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let acf = autocorrelation(ts, Some(max_period))?;
    let mut periods = Vec::new();

    // Find local maxima in ACF
    for i in min_period..max_period.min(acf.len() - 1) {
        if acf[i] > acf[i - 1] && acf[i] > acf[i + 1] {
            let threshold = F::from_f64(0.1).unwrap(); // 10% threshold
            if acf[i] > threshold {
                periods.push(i);
            }
        }
    }

    Ok(periods)
}

/// Detects periods using periodogram analysis
fn detect_periods_periodogram<F>(
    ts: &Array1<F>,
    min_period: usize,
    max_period: usize,
) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut periods = Vec::new();

    // Simple periodogram computation
    for period in min_period..=max_period.min(n / 2) {
        let mut sum_cos = F::zero();
        let mut sum_sin = F::zero();

        for i in 0..n {
            let angle = F::from_f64(2.0 * std::f64::consts::PI).unwrap()
                * F::from_usize(i).unwrap()
                / F::from_usize(period).unwrap();
            sum_cos = sum_cos + ts[i] * angle.cos();
            sum_sin = sum_sin + ts[i] * angle.sin();
        }

        let power = sum_cos * sum_cos + sum_sin * sum_sin;
        let threshold = F::from_f64(0.1).unwrap() * F::from_usize(n).unwrap().powi(2);

        if power > threshold {
            periods.push(period);
        }
    }

    Ok(periods)
}

/// Detects multiple peaks in autocorrelation function
fn detect_multiple_acf_peaks<F>(
    ts: &Array1<F>,
    min_period: usize,
    max_period: usize,
) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug,
{
    let acf = autocorrelation(ts, Some(max_period))?;
    let mut periods = Vec::new();

    // Calculate moving average for smoothing
    let window = 5;
    let mut smoothed_acf = Vec::new();
    for i in 0..acf.len() {
        let start = if i >= window / 2 { i - window / 2 } else { 0 };
        let end = (i + window / 2 + 1).min(acf.len());
        let slice = acf.slice(ndarray::s![start..end]);
        let sum: F = slice.iter().fold(F::zero(), |acc, &x| acc + x);
        smoothed_acf.push(sum / F::from_usize(end - start).unwrap());
    }

    // Find peaks with prominence
    let max_index = if smoothed_acf.len() > 1 {
        max_period.min(smoothed_acf.len() - 1)
    } else {
        return Ok(periods);
    };
    for i in min_period..max_index {
        if smoothed_acf[i] > smoothed_acf[i - 1] && smoothed_acf[i] > smoothed_acf[i + 1] {
            // Check prominence
            let left_min = smoothed_acf[i - min_period.min(i)..i]
                .iter()
                .fold(smoothed_acf[i], |acc, &x| if x < acc { x } else { acc });
            let right_min = smoothed_acf[i + 1..(i + min_period).min(smoothed_acf.len())]
                .iter()
                .fold(smoothed_acf[i], |acc, &x| if x < acc { x } else { acc });

            let prominence = smoothed_acf[i] - left_min.max(right_min);

            if prominence > F::from_f64(0.05).unwrap() {
                // 5% prominence threshold
                periods.push(i);
            }
        }
    }

    Ok(periods)
}

/// Calculates seasonal strength for a given period
fn calculate_seasonal_strength<F>(
    ts: &Array1<F>,
    period: usize,
    _model: &DecompositionModel,
) -> Result<f64>
where
    F: Float + FromPrimitive + Debug,
{
    if period >= ts.len() {
        return Ok(0.0);
    }

    // Extract seasonal indices for this period
    let mut seasonal_values = vec![Vec::new(); period];
    for (i, &value) in ts.iter().enumerate() {
        seasonal_values[i % period].push(value);
    }

    // Calculate variance within and between seasons
    let mut within_season_var = F::zero();
    let mut between_season_var = F::zero();
    let mut season_means = Vec::new();

    // Calculate season means and within-season variance
    for season_data in &seasonal_values {
        if season_data.is_empty() {
            continue;
        }

        let mean = season_data.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from_usize(season_data.len()).unwrap();
        season_means.push(mean);

        let var = season_data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from_usize(season_data.len()).unwrap();
        within_season_var = within_season_var + var;
    }

    // Calculate overall mean and between-season variance
    let overall_mean =
        ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(ts.len()).unwrap();

    for &season_mean in &season_means {
        between_season_var = between_season_var + (season_mean - overall_mean).powi(2);
    }

    if !season_means.is_empty() {
        between_season_var = between_season_var / F::from_usize(season_means.len()).unwrap();
    }

    // Calculate seasonal strength as ratio
    let total_var = within_season_var + between_season_var;
    if total_var == F::zero() {
        return Ok(0.0);
    }

    let strength = between_season_var / total_var;
    Ok(strength.to_f64().unwrap_or(0.0))
}

/// Extracts trend component considering multiple seasonal patterns
fn extract_multi_seasonal_trend<F>(
    ts: &Array1<F>,
    periods: &[usize],
    _config: &MultiSeasonalConfig,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut trend = Array1::zeros(n);

    // Use largest period for window size
    let window_size = periods.iter().max().unwrap_or(&12) * 2 + 1;

    for i in 0..n {
        let start = if i >= window_size / 2 {
            i - window_size / 2
        } else {
            0
        };
        let end = if i + window_size / 2 < n {
            i + window_size / 2 + 1
        } else {
            n
        };

        let window_data: Vec<F> = ts.slice(ndarray::s![start..end]).to_vec();
        trend[i] = robust_mean(&window_data);
    }

    Ok(trend)
}

/// Extracts seasonal components for each period
fn extract_seasonal_components<F>(
    detrended: &Array1<F>,
    periods: &[usize],
    config: &MultiSeasonalConfig,
) -> Result<Vec<Array1<F>>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = detrended.len();
    let mut seasonal_components = vec![Array1::zeros(n); periods.len()];
    let mut remaining = detrended.clone();

    // Extract each seasonal component in order of period length (shortest first)
    let mut sorted_indices: Vec<usize> = (0..periods.len()).collect();
    sorted_indices.sort_by_key(|&i| periods[i]);

    for &idx in &sorted_indices {
        let period = periods[idx];
        let mut seasonal_pattern = Array1::zeros(period);

        // Calculate average for each position in the pattern
        for pos in 0..period {
            let mut values = Vec::new();
            for i in (pos..n).step_by(period) {
                values.push(remaining[i]);
            }
            seasonal_pattern[pos] = robust_mean(&values);
        }

        // Normalize based on model type
        match config.model {
            DecompositionModel::Additive => {
                let mean_seasonal = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from_usize(period).unwrap();
                for i in 0..period {
                    seasonal_pattern[i] = seasonal_pattern[i] - mean_seasonal;
                }
            }
            DecompositionModel::Multiplicative => {
                let mean_seasonal = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                    / F::from_usize(period).unwrap();
                if mean_seasonal != F::zero() {
                    for i in 0..period {
                        seasonal_pattern[i] = seasonal_pattern[i] / mean_seasonal;
                    }
                }
            }
        }

        // Replicate pattern across the series
        for i in 0..n {
            seasonal_components[idx][i] = seasonal_pattern[i % period];
        }

        // Remove this seasonal component from remaining
        for i in 0..n {
            remaining[i] = match config.model {
                DecompositionModel::Additive => remaining[i] - seasonal_components[idx][i],
                DecompositionModel::Multiplicative => {
                    if seasonal_components[idx][i] != F::zero() {
                        remaining[i] / seasonal_components[idx][i]
                    } else {
                        remaining[i]
                    }
                }
            };
        }
    }

    Ok(seasonal_components)
}

/// Combines multiple seasonal components into a single seasonal component
fn combine_seasonal_components<F>(
    seasonal_components: &[Array1<F>],
    model: DecompositionModel,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if seasonal_components.is_empty() {
        return Err(TimeSeriesError::DecompositionError(
            "No seasonal components to combine".to_string(),
        ));
    }

    let n = seasonal_components[0].len();
    let mut combined = match model {
        DecompositionModel::Additive => Array1::zeros(n),
        DecompositionModel::Multiplicative => Array1::ones(n),
    };

    for component in seasonal_components {
        for i in 0..n {
            combined[i] = match model {
                DecompositionModel::Additive => combined[i] + component[i],
                DecompositionModel::Multiplicative => combined[i] * component[i],
            };
        }
    }

    Ok(combined)
}

/// Calculates robust mean using median
fn robust_mean<F>(values: &[F]) -> F
where
    F: Float + FromPrimitive + Debug,
{
    if values.is_empty() {
        return F::zero();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    if len % 2 == 0 {
        let mid1 = sorted[len / 2 - 1];
        let mid2 = sorted[len / 2];
        (mid1 + mid2) / (F::one() + F::one())
    } else {
        sorted[len / 2]
    }
}

fn calculate_l2_norm<F>(arr: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let sum_squares = arr.iter().fold(F::zero(), |acc, &x| acc + x * x);
    Ok(sum_squares.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multi_seasonal_basic() {
        // Create a time series with two seasonal patterns (period 4 and 12)
        let mut ts = Array1::zeros(48);
        for i in 0..48 {
            let t = i as f64;
            ts[i] = 10.0 + t * 0.1  // trend
                  + 2.0 * (2.0 * std::f64::consts::PI * t / 4.0).sin()  // period 4
                  + 3.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin(); // period 12
        }

        let config = MultiSeasonalConfig {
            model: DecompositionModel::Additive,
            max_periods: 2,
            min_period: 3,
            max_period: 20,
            seasonal_strength_threshold: 0.05,
            max_iterations: 50,
            tolerance: 1e-6,
        };

        let result = decompose_multi_seasonal(&ts, &config).unwrap();
        assert!(!result.periods.is_empty());
        assert_eq!(result.seasonal_components.len(), result.periods.len());
    }

    #[test]
    fn test_period_detection() {
        let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let config = MultiSeasonalConfig::default();

        let periods = detect_seasonal_periods(&ts, &config).unwrap();
        assert!(periods.contains(&4) || !periods.is_empty());
    }

    #[test]
    fn test_seasonal_strength() {
        let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let strength = calculate_seasonal_strength(&ts, 4, &DecompositionModel::Additive).unwrap();
        assert!(strength > 0.0);
    }

    #[test]
    fn test_combine_seasonal_components() {
        let comp1 = array![1.0, 2.0, 1.0, 2.0];
        let comp2 = array![0.5, 0.5, 0.5, 0.5];
        let components = vec![comp1, comp2];

        let combined =
            combine_seasonal_components(&components, DecompositionModel::Additive).unwrap();
        assert_eq!(combined[0], 1.5);
        assert_eq!(combined[1], 2.5);
    }
}
