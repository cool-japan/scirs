//! Classical seasonal decomposition methods

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::{DecompositionModel, DecompositionResult};
use crate::error::{Result, TimeSeriesError};
use crate::utils::moving_average;

/// Performs classical seasonal decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period (e.g., 12 for monthly data with yearly seasonality)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{decompose_seasonal, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = decompose_seasonal(&ts, 4, DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn decompose_seasonal<F>(
    ts: &Array1<F>,
    period: usize,
    model: DecompositionModel,
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

    // 1. Calculate centered moving average (trend)
    let window_size = if period % 2 == 0 { period + 1 } else { period };
    let trend = moving_average(ts, window_size)?;

    // Pad trend with NaN values at the beginning and end
    let half_window = window_size / 2;
    let mut padded_trend = Array1::from_elem(ts.len(), F::nan());
    let offset = half_window;

    // Ensure we don't go out of bounds
    for (i, &val) in trend.iter().enumerate() {
        if i + offset < padded_trend.len() {
            padded_trend[i + offset] = val;
        }
    }

    // 2. Remove trend to get detrended series
    let mut detrended = Array1::zeros(ts.len());
    for i in 0..ts.len() {
        if padded_trend[i].is_nan() {
            detrended[i] = F::zero(); // Set detrended to zero where trend is NaN
        } else {
            match model {
                DecompositionModel::Additive => {
                    detrended[i] = ts[i] - padded_trend[i];
                }
                DecompositionModel::Multiplicative => {
                    if padded_trend[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    detrended[i] = ts[i] / padded_trend[i];
                }
            }
        }
    }

    // 3. Calculate seasonal component by averaging values for each season
    let mut seasonal = Array1::zeros(ts.len());
    let mut seasonal_pattern = Array1::zeros(period);
    let mut counts = vec![0; period];

    // Calculate average for each position in the seasonal pattern
    for i in 0..ts.len() {
        let pos = i % period;
        if !detrended[i].is_nan() {
            seasonal_pattern[pos] = seasonal_pattern[pos] + detrended[i];
            counts[pos] += 1;
        }
    }

    // Normalize seasonal pattern
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_pattern[i] = seasonal_pattern[i] / F::from_usize(counts[i]).unwrap();
        }
    }

    // Normalize to ensure seasonal components sum to zero (additive) or average to one (multiplicative)
    match model {
        DecompositionModel::Additive => {
            let mean = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] - mean;
            }
        }
        DecompositionModel::Multiplicative => {
            let mean = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            if mean == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero normalizing multiplicative seasonal pattern".to_string(),
                ));
            }
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] / mean;
            }
        }
    }

    // Apply seasonal pattern to the whole series
    for i in 0..ts.len() {
        seasonal[i] = seasonal_pattern[i % period];
    }

    // 4. Calculate residual component
    let mut residual = Array1::zeros(ts.len());
    for i in 0..ts.len() {
        if padded_trend[i].is_nan() {
            residual[i] = F::nan();
        } else {
            match model {
                DecompositionModel::Additive => {
                    residual[i] = ts[i] - padded_trend[i] - seasonal[i];
                }
                DecompositionModel::Multiplicative => {
                    if padded_trend[i] == F::zero() || seasonal[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model residual calculation"
                                .to_string(),
                        ));
                    }
                    residual[i] = ts[i] / (padded_trend[i] * seasonal[i]);
                }
            }
        }
    }

    // Create result struct
    let original = ts.clone();

    Ok(DecompositionResult {
        trend: padded_trend,
        seasonal,
        residual,
        original,
    })
}
