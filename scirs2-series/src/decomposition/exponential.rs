//! Exponential smoothing decomposition for time series

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::{min, DecompositionModel, DecompositionResult};
use crate::error::{Result, TimeSeriesError};

/// Performs exponential smoothing decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `alpha` - Level smoothing parameter (0 < alpha < 1)
/// * `beta` - Trend smoothing parameter (0 < beta < 1)
/// * `gamma` - Seasonal smoothing parameter (0 < gamma < 1)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing level, trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{exponential_decomposition, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = exponential_decomposition(&ts, 4, 0.2, 0.1, 0.3,
///                                         DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn exponential_decomposition<F>(
    ts: &Array1<F>,
    period: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    model: DecompositionModel,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < period + 1 {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be greater than the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    if alpha <= 0.0 || alpha >= 1.0 || beta <= 0.0 || beta >= 1.0 || gamma <= 0.0 || gamma >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Smoothing parameters must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    let alpha = F::from_f64(alpha).unwrap();
    let beta = F::from_f64(beta).unwrap();
    let gamma = F::from_f64(gamma).unwrap();

    let n = ts.len();

    // Initialize components
    let mut level = Array1::zeros(n + 1);
    let mut trend = Array1::zeros(n + 1);
    let mut seasonal = Array1::zeros(n + period);
    let mut residual = Array1::zeros(n);

    // Initialize level, trend, and seasonal components
    let initial_level = ts[0]; // Could also use average of first few observations
    level[0] = initial_level;

    // Initialize trend (average of first differences)
    if n > 1 {
        let mut sum = F::zero();
        for i in 1..min(n, 10) {
            sum = sum + (ts[i] - ts[i - 1]);
        }
        trend[0] = sum / F::from_usize(min(n - 1, 9)).unwrap();
    }

    // Initialize seasonal (average deviation from level for each season)
    for i in 0..min(period, n) {
        let pos = i % period;
        let expected = level[0] + F::from_usize(i).unwrap() * trend[0];
        match model {
            DecompositionModel::Additive => {
                seasonal[pos] = ts[i] - expected;
            }
            DecompositionModel::Multiplicative => {
                if expected == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model initialization".to_string(),
                    ));
                }
                seasonal[pos] = ts[i] / expected;
            }
        }
    }

    // Normalize initial seasonal component
    match model {
        DecompositionModel::Additive => {
            let mean = seasonal
                .iter()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            for i in 0..period {
                seasonal[i] = seasonal[i] - mean;
            }
        }
        DecompositionModel::Multiplicative => {
            let mean = seasonal
                .iter()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            if mean == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero normalizing multiplicative seasonal component".to_string(),
                ));
            }
            for i in 0..period {
                seasonal[i] = seasonal[i] / mean;
            }
        }
    }

    // Exponential smoothing recursion
    for i in 0..n {
        let s = i % period; // Current season
        let expected = match model {
            DecompositionModel::Additive => level[i] + trend[i],
            DecompositionModel::Multiplicative => level[i] * trend[i],
        };

        // Calculate residual
        match model {
            DecompositionModel::Additive => {
                residual[i] = ts[i] - expected - seasonal[s];
            }
            DecompositionModel::Multiplicative => {
                if expected == F::zero() || seasonal[s] == F::zero() {
                    residual[i] = F::zero(); // Avoid division by zero
                } else {
                    residual[i] = ts[i] / (expected * seasonal[s]);
                }
            }
        }

        // Update level, trend, and seasonal components
        match model {
            DecompositionModel::Additive => {
                level[i + 1] =
                    alpha * (ts[i] - seasonal[s]) + (F::one() - alpha) * (level[i] + trend[i]);
                trend[i + 1] = beta * (level[i + 1] - level[i]) + (F::one() - beta) * trend[i];
                seasonal[s + period] =
                    gamma * (ts[i] - level[i + 1]) + (F::one() - gamma) * seasonal[s];
            }
            DecompositionModel::Multiplicative => {
                if seasonal[s] == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model update".to_string(),
                    ));
                }
                level[i + 1] =
                    alpha * (ts[i] / seasonal[s]) + (F::one() - alpha) * (level[i] * trend[i]);

                if level[i] == F::zero() {
                    trend[i + 1] = trend[i]; // Avoid division by zero
                } else {
                    trend[i + 1] = beta * (level[i + 1] / level[i]) + (F::one() - beta) * trend[i];
                }

                if level[i + 1] == F::zero() {
                    seasonal[s + period] = seasonal[s]; // Avoid division by zero
                } else {
                    seasonal[s + period] =
                        gamma * (ts[i] / level[i + 1]) + (F::one() - gamma) * seasonal[s];
                }
            }
        }
    }

    // Prepare final components
    let trend_component = Array1::from_iter(level.iter().take(n).cloned());
    let seasonal_component = Array1::from_iter((0..n).map(|i| seasonal[i % period]));
    let original = ts.clone();

    Ok(DecompositionResult {
        trend: trend_component,
        seasonal: seasonal_component,
        residual,
        original,
    })
}
