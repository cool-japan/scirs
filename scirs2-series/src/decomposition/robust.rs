//! Robust time series decomposition methods
//!
//! This module provides robust variants of time series decomposition that are
//! resistant to outliers and extreme values.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::{DecompositionModel, DecompositionResult};
use crate::error::{Result, TimeSeriesError};

/// Robust seasonal decomposition using M-estimators and iterative approach
///
/// This is an outlier-resistant version of classical seasonal decomposition that
/// uses robust statistics to handle extreme values.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `model` - Decomposition model (additive or multiplicative)
/// * `max_iter` - Maximum number of iterations for convergence
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{decompose_robust_seasonal, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = decompose_robust_seasonal(&ts, 4, DecompositionModel::Additive, 50, 1e-6).unwrap();
/// ```
pub fn decompose_robust_seasonal<F>(
    ts: &Array1<F>,
    period: usize,
    model: DecompositionModel,
    max_iter: usize,
    tolerance: F,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    let n = ts.len();
    let mut seasonal = Array1::zeros(n);
    let mut residual = ts.clone();

    // Initialize trend with simple moving average
    let mut trend = robust_trend_initial(ts, period)?;

    for iter in 0..max_iter {
        let old_trend = trend.clone();
        let old_seasonal = seasonal.clone();

        // Update seasonal component using robust averaging
        seasonal = update_seasonal_robust(&residual, period, model)?;

        // Update trend component using robust smoother
        let deseasonalized = match model {
            DecompositionModel::Additive => ts - &seasonal,
            DecompositionModel::Multiplicative => {
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    if seasonal[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    result[i] = ts[i] / seasonal[i];
                }
                result
            }
        };

        trend = robust_trend_smoother(&deseasonalized, period)?;

        // Update residual
        for i in 0..n {
            residual[i] = match model {
                DecompositionModel::Additive => ts[i] - trend[i] - seasonal[i],
                DecompositionModel::Multiplicative => {
                    if trend[i] == F::zero() || seasonal[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    ts[i] / (trend[i] * seasonal[i])
                }
            };
        }

        // Check convergence
        let trend_change = calculate_l2_norm(&(&trend - &old_trend))?;
        let seasonal_change = calculate_l2_norm(&(&seasonal - &old_seasonal))?;

        if trend_change < tolerance && seasonal_change < tolerance {
            break;
        }

        if iter == max_iter - 1 {
            eprintln!(
                "Warning: Robust decomposition did not converge in {} iterations",
                max_iter
            );
        }
    }

    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original: ts.clone(),
    })
}

/// Robust LOESS-based decomposition (R-LOESS)
///
/// This implements a robust variant of LOESS decomposition that downweights outliers.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `trend_bandwidth` - Bandwidth for trend smoothing (0.0 to 1.0)
/// * `seasonal_bandwidth` - Bandwidth for seasonal smoothing (0.0 to 1.0)
/// * `max_iter` - Maximum iterations
/// * `tolerance` - Convergence tolerance
///
pub fn decompose_robust_loess<F>(
    ts: &Array1<F>,
    period: usize,
    trend_bandwidth: F,
    seasonal_bandwidth: F,
    max_iter: usize,
    tolerance: F,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    let n = ts.len();
    let mut trend = Array1::zeros(n);
    let mut seasonal = Array1::zeros(n);
    let mut weights = Array1::ones(n);

    for _iter in 0..max_iter {
        let old_trend = trend.clone();
        let old_seasonal = seasonal.clone();

        // Update seasonal component with robust LOESS
        let detrended = ts - &trend;
        seasonal = robust_loess_seasonal(&detrended, period, seasonal_bandwidth, &weights)?;

        // Update trend component with robust LOESS
        let deseasonalized = ts - &seasonal;
        trend = robust_loess_trend(&deseasonalized, trend_bandwidth, &weights)?;

        // Calculate residuals and update weights
        let residual: Array1<F> = ts - &trend - &seasonal;
        weights = calculate_robust_weights(&residual)?;

        // Check convergence
        let trend_change = calculate_l2_norm(&(&trend - &old_trend))?;
        let seasonal_change = calculate_l2_norm(&(&seasonal - &old_seasonal))?;

        if trend_change < tolerance && seasonal_change < tolerance {
            break;
        }
    }

    let residual = ts - &trend - &seasonal;

    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original: ts.clone(),
    })
}

/// M-estimator based robust decomposition
///
/// Uses M-estimators with Huber or Tukey bisquare loss functions for robustness.
///
pub fn decompose_m_estimator<F>(
    ts: &Array1<F>,
    period: usize,
    model: DecompositionModel,
    loss_type: RobustLossType,
    max_iter: usize,
    tolerance: F,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    let n = ts.len();
    let mut seasonal = Array1::zeros(n);

    // Initialize with simple decomposition
    let mut trend = robust_trend_initial(ts, period)?;

    for iter in 0..max_iter {
        let old_trend = trend.clone();
        let old_seasonal = seasonal.clone();

        // Update seasonal using M-estimator
        seasonal = update_seasonal_m_estimator(ts, &trend, period, model, loss_type)?;

        // Update trend using M-estimator
        let deseasonalized = match model {
            DecompositionModel::Additive => ts - &seasonal,
            DecompositionModel::Multiplicative => {
                let mut result = Array1::zeros(n);
                for i in 0..n {
                    if seasonal[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    result[i] = ts[i] / seasonal[i];
                }
                result
            }
        };

        trend = update_trend_m_estimator(&deseasonalized, period, loss_type)?;

        // Check convergence
        let trend_change = calculate_l2_norm(&(&trend - &old_trend))?;
        let seasonal_change = calculate_l2_norm(&(&seasonal - &old_seasonal))?;

        if trend_change < tolerance && seasonal_change < tolerance {
            break;
        }

        if iter == max_iter - 1 {
            eprintln!(
                "Warning: M-estimator decomposition did not converge in {} iterations",
                max_iter
            );
        }
    }

    // Calculate final residuals
    let mut residual = Array1::zeros(n);
    for i in 0..n {
        residual[i] = match model {
            DecompositionModel::Additive => ts[i] - trend[i] - seasonal[i],
            DecompositionModel::Multiplicative => {
                if trend[i] == F::zero() || seasonal[i] == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model".to_string(),
                    ));
                }
                ts[i] / (trend[i] * seasonal[i])
            }
        };
    }

    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original: ts.clone(),
    })
}

/// Robust loss function types for M-estimators
#[derive(Debug, Clone, Copy)]
pub enum RobustLossType {
    /// Huber loss function
    Huber,
    /// Tukey bisquare loss function  
    TukeyBisquare,
    /// Andrews sine loss function
    Andrews,
}

// Helper functions

fn robust_trend_initial<F>(ts: &Array1<F>, period: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut trend = Array1::zeros(n);
    let window = period;

    for i in 0..n {
        let start = if i >= window / 2 { i - window / 2 } else { 0 };
        let end = if i + window / 2 < n {
            i + window / 2 + 1
        } else {
            n
        };

        let window_data: Vec<F> = ts.slice(ndarray::s![start..end]).to_vec();
        trend[i] = median(&window_data);
    }

    Ok(trend)
}

fn update_seasonal_robust<F>(
    residual: &Array1<F>,
    period: usize,
    model: DecompositionModel,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = residual.len();
    let mut seasonal = Array1::zeros(n);
    let mut seasonal_pattern = Array1::zeros(period);

    // Calculate robust seasonal pattern
    for pos in 0..period {
        let mut values = Vec::new();
        for i in (pos..n).step_by(period) {
            values.push(residual[i]);
        }
        seasonal_pattern[pos] = median(&values);
    }

    // Normalize seasonal pattern
    match model {
        DecompositionModel::Additive => {
            let median_val = median(&seasonal_pattern.to_vec());
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] - median_val;
            }
        }
        DecompositionModel::Multiplicative => {
            let median_val = median(&seasonal_pattern.to_vec());
            if median_val == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero in multiplicative seasonal normalization".to_string(),
                ));
            }
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] / median_val;
            }
        }
    }

    // Replicate pattern
    for i in 0..n {
        seasonal[i] = seasonal_pattern[i % period];
    }

    Ok(seasonal)
}

fn robust_trend_smoother<F>(ts: &Array1<F>, window: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut trend = Array1::zeros(n);

    for i in 0..n {
        let start = if i >= window / 2 { i - window / 2 } else { 0 };
        let end = if i + window / 2 < n {
            i + window / 2 + 1
        } else {
            n
        };

        let window_data: Vec<F> = ts.slice(ndarray::s![start..end]).to_vec();
        trend[i] = median(&window_data);
    }

    Ok(trend)
}

fn robust_loess_seasonal<F>(
    ts: &Array1<F>,
    _period: usize,
    bandwidth: F,
    weights: &Array1<F>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut seasonal = Array1::zeros(n);
    let window_size = ((bandwidth * F::from_usize(n).unwrap())
        .round()
        .to_usize()
        .unwrap())
    .max(1);

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

        let mut weighted_values = Vec::new();
        for j in start..end {
            let weight = weights[j];
            for _ in 0..(weight * F::from_f64(100.0).unwrap())
                .round()
                .to_usize()
                .unwrap_or(1)
            {
                weighted_values.push(ts[j]);
            }
        }

        seasonal[i] = if weighted_values.is_empty() {
            F::zero()
        } else {
            median(&weighted_values)
        };
    }

    Ok(seasonal)
}

fn robust_loess_trend<F>(ts: &Array1<F>, bandwidth: F, weights: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut trend = Array1::zeros(n);
    let window_size = ((bandwidth * F::from_usize(n).unwrap())
        .round()
        .to_usize()
        .unwrap())
    .max(1);

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

        let mut weighted_values = Vec::new();
        for j in start..end {
            let weight = weights[j];
            for _ in 0..(weight * F::from_f64(100.0).unwrap())
                .round()
                .to_usize()
                .unwrap_or(1)
            {
                weighted_values.push(ts[j]);
            }
        }

        trend[i] = if weighted_values.is_empty() {
            F::zero()
        } else {
            median(&weighted_values)
        };
    }

    Ok(trend)
}

fn calculate_robust_weights<F>(residuals: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = residuals.len();
    let mut weights = Array1::ones(n);

    // Calculate median absolute deviation (MAD)
    let residual_vec: Vec<F> = residuals.to_vec();
    let median_residual = median(&residual_vec);
    let abs_deviations: Vec<F> = residual_vec
        .iter()
        .map(|&r| (r - median_residual).abs())
        .collect();

    let mad = median(&abs_deviations) * F::from_f64(1.4826).unwrap(); // 1.4826 for normal distribution

    if mad == F::zero() {
        return Ok(weights);
    }

    // Calculate Tukey bisquare weights
    let c = F::from_f64(4.685).unwrap(); // Tukey constant
    for i in 0..n {
        let u = (residuals[i] - median_residual).abs() / mad;
        if u <= c {
            let ratio = u / c;
            weights[i] = (F::one() - ratio * ratio).powi(2);
        } else {
            weights[i] = F::zero();
        }
    }

    Ok(weights)
}

fn update_seasonal_m_estimator<F>(
    ts: &Array1<F>,
    trend: &Array1<F>,
    period: usize,
    model: DecompositionModel,
    loss_type: RobustLossType,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut seasonal = Array1::zeros(n);
    let mut seasonal_pattern = Array1::zeros(period);

    // Detrend the series
    let detrended = match model {
        DecompositionModel::Additive => ts - trend,
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

    // Calculate robust seasonal estimates for each position
    for pos in 0..period {
        let mut values = Vec::new();
        for i in (pos..n).step_by(period) {
            values.push(detrended[i]);
        }
        seasonal_pattern[pos] = m_estimator(&values, loss_type)?;
    }

    // Normalize seasonal pattern
    match model {
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
            if mean_seasonal == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero in multiplicative seasonal normalization".to_string(),
                ));
            }
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] / mean_seasonal;
            }
        }
    }

    // Replicate pattern
    for i in 0..n {
        seasonal[i] = seasonal_pattern[i % period];
    }

    Ok(seasonal)
}

fn update_trend_m_estimator<F>(
    ts: &Array1<F>,
    window: usize,
    loss_type: RobustLossType,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut trend = Array1::zeros(n);

    for i in 0..n {
        let start = if i >= window / 2 { i - window / 2 } else { 0 };
        let end = if i + window / 2 < n {
            i + window / 2 + 1
        } else {
            n
        };

        let window_data: Vec<F> = ts.slice(ndarray::s![start..end]).to_vec();
        trend[i] = m_estimator(&window_data, loss_type)?;
    }

    Ok(trend)
}

fn m_estimator<F>(values: &[F], loss_type: RobustLossType) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if values.is_empty() {
        return Ok(F::zero());
    }

    match loss_type {
        RobustLossType::Huber => huber_estimator(values),
        RobustLossType::TukeyBisquare => tukey_estimator(values),
        RobustLossType::Andrews => andrews_estimator(values),
    }
}

fn huber_estimator<F>(values: &[F]) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_val = median(&sorted_values);
    let mad = {
        let abs_deviations: Vec<F> = sorted_values
            .iter()
            .map(|&v| (v - median_val).abs())
            .collect::<Vec<_>>();
        let mut sorted_abs = abs_deviations;
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        median(&sorted_abs) * F::from_f64(1.4826).unwrap()
    };

    if mad == F::zero() {
        return Ok(median_val);
    }

    let c = F::from_f64(1.345).unwrap(); // Huber constant
    let threshold = c * mad;

    // Iteratively reweighted least squares
    let mut estimate = median_val;
    for _ in 0..20 {
        // Max iterations
        let mut sum_weighted = F::zero();
        let mut sum_weights = F::zero();

        for &value in values {
            let residual = (value - estimate).abs();
            let weight = if residual <= threshold {
                F::one()
            } else {
                threshold / residual
            };

            sum_weighted = sum_weighted + weight * value;
            sum_weights = sum_weights + weight;
        }

        let new_estimate = if sum_weights == F::zero() {
            estimate
        } else {
            sum_weighted / sum_weights
        };

        if (new_estimate - estimate).abs() < F::from_f64(1e-8).unwrap() {
            break;
        }
        estimate = new_estimate;
    }

    Ok(estimate)
}

fn tukey_estimator<F>(values: &[F]) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let sorted_values = {
        let mut vals = values.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals
    };

    let median_val = median(&sorted_values);
    let mad = {
        let abs_deviations: Vec<F> = sorted_values
            .iter()
            .map(|&v| (v - median_val).abs())
            .collect::<Vec<_>>();
        let mut sorted_abs = abs_deviations;
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        median(&sorted_abs) * F::from_f64(1.4826).unwrap()
    };

    if mad == F::zero() {
        return Ok(median_val);
    }

    let c = F::from_f64(4.685).unwrap(); // Tukey constant

    // Iteratively reweighted least squares
    let mut estimate = median_val;
    for _ in 0..20 {
        // Max iterations
        let mut sum_weighted = F::zero();
        let mut sum_weights = F::zero();

        for &value in values {
            let u = (value - estimate).abs() / mad;
            let weight = if u <= c {
                let ratio = u / c;
                (F::one() - ratio * ratio).powi(2)
            } else {
                F::zero()
            };

            sum_weighted = sum_weighted + weight * value;
            sum_weights = sum_weights + weight;
        }

        let new_estimate = if sum_weights == F::zero() {
            estimate
        } else {
            sum_weighted / sum_weights
        };

        if (new_estimate - estimate).abs() < F::from_f64(1e-8).unwrap() {
            break;
        }
        estimate = new_estimate;
    }

    Ok(estimate)
}

fn andrews_estimator<F>(values: &[F]) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let sorted_values = {
        let mut vals = values.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals
    };

    let median_val = median(&sorted_values);
    let mad = {
        let abs_deviations: Vec<F> = sorted_values
            .iter()
            .map(|&v| (v - median_val).abs())
            .collect::<Vec<_>>();
        let mut sorted_abs = abs_deviations;
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        median(&sorted_abs) * F::from_f64(1.4826).unwrap()
    };

    if mad == F::zero() {
        return Ok(median_val);
    }

    let c = F::from_f64(1.339).unwrap(); // Andrews constant

    // Iteratively reweighted least squares
    let mut estimate = median_val;
    for _ in 0..20 {
        // Max iterations
        let mut sum_weighted = F::zero();
        let mut sum_weights = F::zero();

        for &value in values {
            let u = (value - estimate).abs() / mad;
            let weight = if u <= c {
                let pi_val = F::from_f64(std::f64::consts::PI).unwrap();
                ((u * pi_val / c).sin() / (u * pi_val / c)).abs()
            } else {
                F::zero()
            };

            sum_weighted = sum_weighted + weight * value;
            sum_weights = sum_weights + weight;
        }

        let new_estimate = if sum_weights == F::zero() {
            estimate
        } else {
            sum_weighted / sum_weights
        };

        if (new_estimate - estimate).abs() < F::from_f64(1e-8).unwrap() {
            break;
        }
        estimate = new_estimate;
    }

    Ok(estimate)
}

fn median<F>(values: &[F]) -> F
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
