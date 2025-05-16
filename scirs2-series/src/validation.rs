//! Validation utilities for time series module
//!
//! Provides centralized validation functions for parameters and data

use ndarray::{ArrayBase, Data, Ix1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Display;

use crate::error::{Result, TimeSeriesError};

/// Validate that a value is positive
pub fn check_positive<F: Float + Display>(value: F, name: &str) -> Result<()> {
    if value <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: name.to_string(),
            message: format!("Must be positive, got {}", value),
        });
    }
    Ok(())
}

/// Validate that a value is non-negative
pub fn check_non_negative<F: Float + Display>(value: F, name: &str) -> Result<()> {
    if value < F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: name.to_string(),
            message: format!("Must be non-negative, got {}", value),
        });
    }
    Ok(())
}

/// Validate that a value is in range [0, 1]
pub fn check_probability<F: Float + Display>(value: F, name: &str) -> Result<()> {
    if value < F::zero() || value > F::one() {
        return Err(TimeSeriesError::InvalidParameter {
            name: name.to_string(),
            message: format!("Must be in [0, 1], got {}", value),
        });
    }
    Ok(())
}

/// Validate that a value is in a given range
pub fn check_in_range<F: Float + Display>(value: F, min: F, max: F, name: &str) -> Result<()> {
    if value < min || value > max {
        return Err(TimeSeriesError::InvalidParameter {
            name: name.to_string(),
            message: format!("Must be in [{}, {}], got {}", min, max, value),
        });
    }
    Ok(())
}

/// Validate that an array has sufficient length
pub fn check_array_length<S, F>(
    data: &ArrayBase<S, Ix1>,
    min_length: usize,
    operation: &str,
) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float,
{
    if data.len() < min_length {
        return Err(TimeSeriesError::InsufficientData {
            message: format!("for {}", operation),
            required: min_length,
            actual: data.len(),
        });
    }
    Ok(())
}

/// Validate that two arrays have the same length
pub fn check_same_length<S1, S2, F>(
    arr1: &ArrayBase<S1, Ix1>,
    arr2: &ArrayBase<S2, Ix1>,
    _name1: &str,
    _name2: &str,
) -> Result<()>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float,
{
    if arr1.len() != arr2.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: arr1.len(),
            actual: arr2.len(),
        });
    }
    Ok(())
}

/// Validate ARIMA orders
pub fn validate_arima_orders(p: usize, d: usize, q: usize) -> Result<()> {
    if p > 10 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "p".to_string(),
            message: format!("AR order too large: {}", p),
        });
    }
    if d > 3 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "d".to_string(),
            message: format!("Differencing order too large: {}", d),
        });
    }
    if q > 10 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "q".to_string(),
            message: format!("MA order too large: {}", q),
        });
    }
    Ok(())
}

/// Validate seasonal ARIMA orders
pub fn validate_seasonal_arima_orders(
    p: usize,
    d: usize,
    q: usize,
    p_seasonal: usize,
    d_seasonal: usize,
    q_seasonal: usize,
    period: usize,
) -> Result<()> {
    validate_arima_orders(p, d, q)?;

    if p_seasonal > 5 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "p_seasonal".to_string(),
            message: format!("Seasonal AR order too large: {}", p_seasonal),
        });
    }
    if d_seasonal > 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "d_seasonal".to_string(),
            message: format!("Seasonal differencing order too large: {}", d_seasonal),
        });
    }
    if q_seasonal > 5 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "q_seasonal".to_string(),
            message: format!("Seasonal MA order too large: {}", q_seasonal),
        });
    }
    if period < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: format!("Period must be at least 2, got {}", period),
        });
    }
    if period > 365 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: format!("Period too large: {}", period),
        });
    }

    Ok(())
}

/// Validate forecast horizon
pub fn validate_forecast_horizon(steps: usize, max_reasonable: Option<usize>) -> Result<()> {
    if steps == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "steps".to_string(),
            message: "Forecast horizon must be positive".to_string(),
        });
    }

    let max = max_reasonable.unwrap_or(10000);
    if steps > max {
        return Err(TimeSeriesError::InvalidParameter {
            name: "steps".to_string(),
            message: format!("Forecast horizon too large: {}", steps),
        });
    }

    Ok(())
}

/// Validate window size for rolling operations
pub fn validate_window_size(window: usize, data_length: usize) -> Result<()> {
    if window == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window".to_string(),
            message: "Window size must be positive".to_string(),
        });
    }

    if window > data_length {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window".to_string(),
            message: format!("Window size {} exceeds data length {}", window, data_length),
        });
    }

    Ok(())
}

/// Validate lag for time series operations
pub fn validate_lag(lag: usize, data_length: usize) -> Result<()> {
    if lag >= data_length {
        return Err(TimeSeriesError::InvalidParameter {
            name: "lag".to_string(),
            message: format!("Lag {} must be less than data length {}", lag, data_length),
        });
    }
    Ok(())
}

/// Check if array has no missing values
pub fn check_no_missing<S, F>(data: &ArrayBase<S, Ix1>) -> Result<()>
where
    S: Data<Elem = F>,
    F: Float,
{
    for (i, &x) in data.iter().enumerate() {
        if x.is_nan() || x.is_infinite() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Non-finite value at index {}",
                i
            )));
        }
    }
    Ok(())
}

/// Check if array is stationary (basic check)
pub fn check_stationarity_basic<S, F>(data: &ArrayBase<S, Ix1>) -> Result<bool>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    check_array_length(data, 10, "stationarity check")?;

    // Split data into two halves
    let mid = data.len() / 2;
    let first_half = data.slice(ndarray::s![..mid]);
    let second_half = data.slice(ndarray::s![mid..]);

    // Compare means and variances
    let mean1 = first_half.mean().unwrap_or(F::zero());
    let mean2 = second_half.mean().unwrap_or(F::zero());

    let var1 = first_half
        .mapv(|x| (x - mean1) * (x - mean1))
        .mean()
        .unwrap_or(F::zero());
    let var2 = second_half
        .mapv(|x| (x - mean2) * (x - mean2))
        .mean()
        .unwrap_or(F::zero());

    // Check if means and variances are similar
    let mean_diff = (mean1 - mean2).abs();
    let var_ratio = if var1 > F::zero() && var2 > F::zero() {
        (var1 / var2).max(var2 / var1)
    } else {
        F::one()
    };

    // Rough thresholds
    let mean_threshold =
        F::from(0.2).unwrap() * (var1.sqrt() + var2.sqrt()) / F::from(2.0).unwrap();
    let var_threshold = F::from(2.0).unwrap();

    Ok(mean_diff < mean_threshold && var_ratio < var_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_check_positive() {
        assert!(check_positive(1.0, "value").is_ok());
        assert!(check_positive(0.0, "value").is_err());
        assert!(check_positive(-1.0, "value").is_err());
    }

    #[test]
    fn test_check_probability() {
        assert!(check_probability(0.5, "prob").is_ok());
        assert!(check_probability(0.0, "prob").is_ok());
        assert!(check_probability(1.0, "prob").is_ok());
        assert!(check_probability(1.1, "prob").is_err());
        assert!(check_probability(-0.1, "prob").is_err());
    }

    #[test]
    fn test_check_array_length() {
        let arr = array![1.0, 2.0, 3.0];
        assert!(check_array_length(&arr, 3, "test").is_ok());
        assert!(check_array_length(&arr, 4, "test").is_err());
    }

    #[test]
    fn test_validate_arima_orders() {
        assert!(validate_arima_orders(2, 1, 2).is_ok());
        assert!(validate_arima_orders(11, 1, 1).is_err());
        assert!(validate_arima_orders(1, 4, 1).is_err());
        assert!(validate_arima_orders(1, 1, 11).is_err());
    }
}
