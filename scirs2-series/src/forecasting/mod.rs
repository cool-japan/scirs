//! Time series forecasting methods
//!
//! This module provides implementations for forecasting future values of time series.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::decomposition::{common::DecompositionModel, exponential::exponential_decomposition};
use crate::error::{Result, TimeSeriesError};
use crate::utils::{is_stationary, transform_to_stationary};

/// Result of time series forecasting
#[derive(Debug, Clone)]
pub struct ForecastResult<F> {
    /// Point forecasts
    pub forecast: Array1<F>,
    /// Lower confidence interval
    pub lower_ci: Array1<F>,
    /// Upper confidence interval
    pub upper_ci: Array1<F>,
}

/// ARIMA model parameters
#[derive(Debug, Clone)]
pub struct ArimaParams {
    /// Autoregressive order (p)
    pub p: usize,
    /// Integration order (d)
    pub d: usize,
    /// Moving average order (q)
    pub q: usize,
    /// Seasonal order (P)
    pub seasonal_p: Option<usize>,
    /// Seasonal integration order (D)
    pub seasonal_d: Option<usize>,
    /// Seasonal moving average order (Q)
    pub seasonal_q: Option<usize>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Fit intercept
    pub fit_intercept: bool,
    /// Trend component
    pub trend: Option<String>,
}

impl Default for ArimaParams {
    fn default() -> Self {
        Self {
            p: 1,
            d: 0,
            q: 0,
            seasonal_p: None,
            seasonal_d: None,
            seasonal_q: None,
            seasonal_period: None,
            fit_intercept: true,
            trend: None,
        }
    }
}

/// Exponential smoothing parameters
#[derive(Debug, Clone)]
pub struct ExpSmoothingParams {
    /// Level smoothing parameter (alpha)
    pub alpha: f64,
    /// Trend smoothing parameter (beta)
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter (gamma)
    pub gamma: Option<f64>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Whether to use multiplicative trend
    pub multiplicative_trend: bool,
    /// Whether to use multiplicative seasonality
    pub multiplicative_seasonality: bool,
    /// Whether to damp trend
    pub damped_trend: bool,
    /// Damping factor
    pub phi: Option<f64>,
}

impl Default for ExpSmoothingParams {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: None,
            gamma: None,
            seasonal_period: None,
            multiplicative_trend: false,
            multiplicative_seasonality: false,
            damped_trend: false,
            phi: None,
        }
    }
}

/// Forecasts future values using simple moving average
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `window_size` - Size of the moving window
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::moving_average_forecast;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = moving_average_forecast(&ts, 3, 5, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// println!("Lower CI: {:?}", result.lower_ci);
/// println!("Upper CI: {:?}", result.upper_ci);
/// ```
#[allow(dead_code)]
pub fn moving_average_forecast<F>(
    ts: &Array1<F>,
    window_size: usize,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < window_size {
        return Err(TimeSeriesError::ForecastingError(format!(
            "Time series length ({}) must be at least equal to window _size ({})",
            ts.len(),
            window_size
        )));
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Calculate the most recent moving average
    let mut sum = F::zero();
    for i in ts.len() - window_size..ts.len() {
        sum = sum + ts[i];
    }
    let avg = sum / F::from_usize(window_size).unwrap();

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Calculate the standard error of past forecasts
    let mut sq_errors = Array1::zeros(ts.len() - window_size);

    for i in window_size..ts.len() {
        // Calculate the moving average for past windows
        let mut window_sum = F::zero();
        for j in i - window_size..i {
            window_sum = window_sum + ts[j];
        }
        let window_avg = window_sum / F::from_usize(window_size).unwrap();

        // Calculate squared error
        sq_errors[i - window_size] = (ts[i] - window_avg).powi(2);
    }

    // Calculate standard error
    let mse = sq_errors.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(sq_errors.len()).unwrap();
    let std_err = mse.sqrt();

    // Z-score for the given confidence _level (approximation)
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.98 => F::from_f64(2.326).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        c if c >= 0.85 => F::from_f64(1.44).unwrap(),
        c if c >= 0.80 => F::from_f64(1.282).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    // Compute forecast and confidence intervals
    for i in 0..horizon {
        forecast[i] = avg;

        // Increase uncertainty with horizon (more uncertainty further into the future)
        // This is a simplified model; in practice, more complex time-dependent error models are used
        let adjustment = F::one() + F::from_f64(0.1).unwrap() * F::from_usize(i).unwrap();
        let ci_width = z_score * std_err * adjustment;

        lower_ci[i] = avg - ci_width;
        upper_ci[i] = avg + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Forecasts future values using simple exponential smoothing
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `alpha` - Smoothing parameter (0.0-1.0)
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::exponential_smoothing_forecast;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = exponential_smoothing_forecast(&ts, 0.3, 5, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// println!("Lower CI: {:?}", result.lower_ci);
/// println!("Upper CI: {:?}", result.upper_ci);
/// ```
#[allow(dead_code)]
pub fn exponential_smoothing_forecast<F>(
    ts: &Array1<F>,
    alpha: f64,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series must have at least 2 points for exponential smoothing".to_string(),
        ));
    }

    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Smoothing parameter (alpha) must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Initialize _level and forecast error arrays
    let mut _level = Array1::zeros(ts.len() + 1);
    let mut sq_errors = Array1::zeros(ts.len() - 1);

    _level[0] = ts[0]; // Initialize with first observation

    // Apply simple exponential smoothing
    for i in 0..ts.len() {
        // Update _level
        _level[i + 1] =
            F::from_f64(alpha).unwrap() * ts[i] + F::from_f64(1.0 - alpha).unwrap() * _level[i];

        // Calculate forecast error for one-step ahead forecast
        if i > 0 {
            sq_errors[i - 1] = (ts[i] - _level[i]).powi(2);
        }
    }

    // Calculate mean squared error
    let mse = sq_errors.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(sq_errors.len()).unwrap();
    let std_err = mse.sqrt();

    // Z-score for the given confidence _level
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.98 => F::from_f64(2.326).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        c if c >= 0.85 => F::from_f64(1.44).unwrap(),
        c if c >= 0.80 => F::from_f64(1.282).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Compute forecast and confidence intervals
    for i in 0..horizon {
        forecast[i] = _level[ts.len()]; // All forecasts are the same (last level)

        // Increase uncertainty with horizon
        // For SES, theoretical standard error increases with square root of horizon
        let h_adjustment = (F::from_usize(i + 1).unwrap()).sqrt();
        let ci_width = z_score * std_err * h_adjustment;

        lower_ci[i] = forecast[i] - ci_width;
        upper_ci[i] = forecast[i] + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Forecasts future values using Holt-Winters exponential smoothing
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `params` - Exponential smoothing parameters
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::{holt_winters_forecast, ExpSmoothingParams};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9.0, 8.0];
///
/// let mut params = ExpSmoothingParams::default();
/// params.alpha = 0.3;
/// params.beta = Some(0.1);
/// params.gamma = Some(0.2);
/// params.seasonal_period = Some(4);
///
/// let result = holt_winters_forecast(&ts, &params, 8, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// ```
#[allow(dead_code)]
pub fn holt_winters_forecast<F>(
    ts: &Array1<F>,
    params: &ExpSmoothingParams,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate inputs
    if params.alpha <= 0.0 || params.alpha >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Alpha must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    if let Some(beta) = params.beta {
        if beta <= 0.0 || beta >= 1.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Beta must be between 0 and 1 (exclusive)".to_string(),
            ));
        }
    }

    if let Some(gamma) = params.gamma {
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Gamma must be between 0 and 1 (exclusive)".to_string(),
            ));
        }

        // If gamma is provided, seasonal period must also be provided
        if params.seasonal_period.is_none() {
            return Err(TimeSeriesError::InvalidInput(
                "Seasonal period must be provided when gamma is specified".to_string(),
            ));
        }
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    if let Some(period) = params.seasonal_period {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::ForecastingError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    } else if ts.len() < 3 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series must have at least 3 points for non-seasonal models".to_string(),
        ));
    }

    // For damped trend model, phi must be provided and valid
    if params.damped_trend {
        if let Some(phi) = params.phi {
            if phi <= 0.0 || phi >= 1.0 {
                return Err(TimeSeriesError::InvalidInput(
                    "Damping parameter (phi) must be between 0 and 1 (exclusive)".to_string(),
                ));
            }
        } else {
            return Err(TimeSeriesError::InvalidInput(
                "Damping parameter (phi) must be provided for damped trend models".to_string(),
            ));
        }
    }

    // Determine the model type
    let has_trend = params.beta.is_some();
    let has_seasonal = params.gamma.is_some() && params.seasonal_period.is_some();

    // For multiplicative models, data must be strictly positive
    if (params.multiplicative_trend || params.multiplicative_seasonality)
        && ts.iter().any(|&x| x <= F::zero())
    {
        return Err(TimeSeriesError::InvalidInput(
            "Multiplicative models require strictly positive data".to_string(),
        ));
    }

    // Implementing a simplified version of Holt-Winters
    // This is a placeholder implementation and would need more work for a full implementation

    // Initialize arrays for components
    let n = ts.len();
    let mut _level = Array1::zeros(n + 1);
    let mut trend = Array1::zeros(n + 1);
    let mut seasonal = Array1::zeros(n + params.seasonal_period.unwrap_or(1));
    let mut forecast_errors = Array1::zeros(n);

    // For simplicity, we're using an additive model here
    // A full implementation would handle multiplicative models as well

    // Initialize components
    if has_seasonal {
        // We use exponential decomposition which is already implemented
        let period = params.seasonal_period.unwrap();
        let decomp = exponential_decomposition(
            ts,
            period,
            params.alpha,
            params.beta.unwrap_or(0.1),
            params.gamma.unwrap(),
            if params.multiplicative_seasonality {
                DecompositionModel::Multiplicative
            } else {
                DecompositionModel::Additive
            },
        )?;

        // Extract components
        _level[n] = decomp.trend[n - 1];
        if has_trend {
            // Simplistically, we use the difference between the last two trend values
            if n >= 2 {
                trend[n] = decomp.trend[n - 1] - decomp.trend[n - 2];
            }
        }

        // Copy the seasonal pattern
        for i in 0..period {
            seasonal[i] = decomp.seasonal[n - period + i];
        }

        // Calculate forecast errors
        for i in period..n {
            let pred = decomp.trend[i - 1] + decomp.seasonal[i];
            forecast_errors[i] = ts[i] - pred;
        }
    } else if has_trend {
        // Simple Holt's method (linear trend, no seasonality)
        _level[0] = ts[0];
        if n > 1 {
            trend[0] = ts[1] - ts[0];
        }

        // Apply Holt's method
        let alpha = F::from_f64(params.alpha).unwrap();
        let beta = F::from_f64(params.beta.unwrap()).unwrap();
        let phi = F::from_f64(params.phi.unwrap_or(1.0)).unwrap();

        for i in 1..=n {
            // Calculate expected value
            let expected = _level[i - 1]
                + if params.damped_trend {
                    phi * trend[i - 1]
                } else {
                    trend[i - 1]
                };

            // Update _level and trend
            if i < n {
                _level[i] = alpha * ts[i - 1] + (F::one() - alpha) * expected;
                trend[i] = beta * (_level[i] - _level[i - 1])
                    + (F::one() - beta)
                        * if params.damped_trend {
                            phi * trend[i - 1]
                        } else {
                            trend[i - 1]
                        };

                // Calculate forecast error
                forecast_errors[i - 1] = ts[i - 1] - expected;
            }
        }
    } else {
        // Simple exponential smoothing (_level only)
        return exponential_smoothing_forecast(ts, params.alpha, horizon, conf_level);
    }

    // Calculate standard error for confidence intervals
    let mse = forecast_errors
        .iter()
        .skip(if has_seasonal {
            params.seasonal_period.unwrap()
        } else {
            1
        })
        .fold(F::zero(), |acc, &x| acc + x.powi(2))
        / F::from_usize(if has_seasonal {
            n - params.seasonal_period.unwrap()
        } else {
            n - 1
        })
        .unwrap();
    let std_err = mse.sqrt();

    // Z-score for confidence _level
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.98 => F::from_f64(2.326).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        c if c >= 0.85 => F::from_f64(1.44).unwrap(),
        c if c >= 0.80 => F::from_f64(1.282).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Generate forecasts
    for h in 0..horizon {
        let mut pred = _level[n];

        // Add trend component
        if has_trend {
            let phi = F::from_f64(params.phi.unwrap_or(1.0)).unwrap();
            if params.damped_trend {
                // Sum of damped trend: b * (1 + phi + phi^2 + ... + phi^(h-1))
                let mut sum = F::one();
                let mut term = F::one();
                for _ in 1..h + 1 {
                    term = term * phi;
                    sum = sum + term;
                }
                pred = pred + trend[n] * sum;
            } else {
                // Linear trend: l + h*b
                pred = pred + F::from_usize(h + 1).unwrap() * trend[n];
            }
        }

        // Add seasonal component
        if has_seasonal {
            let period = params.seasonal_period.unwrap();
            let season_idx = (h + n) % period;
            if params.multiplicative_seasonality {
                pred = pred * seasonal[season_idx];
            } else {
                pred = pred + seasonal[season_idx];
            }
        }

        forecast[h] = pred;

        // Confidence intervals (simplified)
        // A full implementation would account for increased uncertainty with horizon
        let h_adjustment = (F::from_usize(h + 1).unwrap()).sqrt();
        let ci_width = z_score * std_err * h_adjustment;

        lower_ci[h] = pred - ci_width;
        upper_ci[h] = pred + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Forecasts using ARIMA (Auto-Regressive Integrated Moving Average) models
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `params` - ARIMA model parameters
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::{arima_forecast, ArimaParams};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// let mut params = ArimaParams::default();
/// params.p = 1;
/// params.d = 1;
/// params.q = 1;
///
/// let result = arima_forecast(&ts, &params, 5, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// ```
#[allow(dead_code)]
pub fn arima_forecast<F>(
    ts: &Array1<F>,
    params: &ArimaParams,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate inputs
    if ts.len() <= params.p + params.d + params.q {
        return Err(TimeSeriesError::ForecastingError(format!(
            "Time series length ({}) must be greater than p+d+q ({})",
            ts.len(),
            params.p + params.d + params.q
        )));
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Check for seasonal components
    if let Some(period) = params.seasonal_period {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::ForecastingError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    }

    // ARIMA implementation is complex and would require a more detailed implementation
    // Here's a simplified version that handles some basic cases

    // Differencing (if needed)
    let mut data = ts.clone();

    // Apply differencing "d" times
    for _ in 0..params.d {
        let mut diff_data = Array1::zeros(data.len() - 1);
        for i in 0..data.len() - 1 {
            diff_data[i] = data[i + 1] - data[i];
        }
        data = diff_data;
    }

    // Apply seasonal differencing if needed
    if let (Some(s_d), Some(period)) = (params.seasonal_d, params.seasonal_period) {
        for _ in 0..s_d {
            if data.len() <= period {
                return Err(TimeSeriesError::ForecastingError(
                    "Series too short after differencing for seasonal differencing".to_string(),
                ));
            }

            let mut diff_data = Array1::zeros(data.len() - period);
            for i in 0..data.len() - period {
                diff_data[i] = data[i + period] - data[i];
            }
            data = diff_data;
        }
    }

    // A simple AR(p) model implementation
    // For a full ARIMA implementation, we would need to estimate all parameters
    // and handle the MA terms as well

    // Simplified AR coefficients (normally these would be estimated from the data)
    let ar_coeffs: Vec<F> = if params.p > 0 {
        let mut coeffs = Vec::with_capacity(params.p);
        for i in 0..params.p {
            // Decreasing coefficients for higher lags (simplified)
            coeffs.push(F::from_f64(0.8 / (i + 1) as f64).unwrap());
        }
        coeffs
    } else {
        vec![]
    };

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Generate forecasts
    let n = data.len();

    // Simplified AR model forecasting
    for h in 0..horizon {
        let mut pred = F::zero();

        // AR component
        for i in 0..params.p {
            if h >= i && h - i < n {
                // Use actual values for available history
                pred = pred + ar_coeffs[i] * data[n - 1 - (h - i)];
            } else if h >= i {
                // Use forecasted values for future points
                pred = pred + ar_coeffs[i] * forecast[h - i - 1];
            }
        }

        forecast[h] = pred;
    }

    // Reverse the differencing to get forecasts in original scale
    for _ in 0..params.d {
        // For each step, we need the last value from the previous _level
        let last_value = if params.d > 0 {
            ts[ts.len() - 1]
        } else {
            F::zero()
        };

        for h in 0..horizon {
            if h == 0 {
                forecast[h] = last_value + forecast[h];
            } else {
                forecast[h] = forecast[h - 1] + forecast[h];
            }
        }
    }

    // Simplified confidence intervals (would need proper error variance estimation)
    let std_err = F::from_f64(0.5).unwrap(); // Placeholder
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    for h in 0..horizon {
        let ci_width = z_score * std_err * (F::from_usize(h + 1).unwrap()).sqrt();
        lower_ci[h] = forecast[h] - ci_width;
        upper_ci[h] = forecast[h] + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Options for automatic ARIMA model selection
#[derive(Debug, Clone)]
pub struct AutoArimaOptions {
    /// Maximum AR order (p) to consider
    pub max_p: usize,
    /// Maximum differencing order (d) to consider
    pub max_d: usize,
    /// Maximum MA order (q) to consider
    pub max_q: usize,
    /// Whether to include seasonal components
    pub seasonal: bool,
    /// Seasonal period (required if seasonal is true)
    pub seasonal_period: Option<usize>,
    /// Maximum seasonal AR order (P) to consider
    pub max_seasonal_p: usize,
    /// Maximum seasonal differencing order (D) to consider
    pub max_seasonal_d: usize,
    /// Maximum seasonal MA order (Q) to consider
    pub max_seasonal_q: usize,
    /// Whether to automatically determine differencing order
    pub auto_diff: bool,
    /// Whether to estimate constant/drift term
    pub with_constant: bool,
    /// Information criterion to use for model selection (AIC or BIC)
    pub information_criterion: String,
    /// Number of steps for out-of-sample cross-validation
    pub stepwise: bool,
    /// Maximum total parameters to consider (to avoid overfitting)
    pub max_order: usize,
}

impl Default for AutoArimaOptions {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_d: 2,
            max_q: 5,
            seasonal: false,
            seasonal_period: None,
            max_seasonal_p: 2,
            max_seasonal_d: 1,
            max_seasonal_q: 2,
            auto_diff: true,
            with_constant: true,
            information_criterion: "aic".to_string(),
            stepwise: true,
            max_order: 10,
        }
    }
}

/// Model fit metrics
#[derive(Debug, Clone)]
struct ModelFitMetrics<F> {
    /// Akaike Information Criterion (AIC)
    aic: F,
    /// Bayesian Information Criterion (BIC)
    bic: F,
    /// Hannan-Quinn Information Criterion (HQIC)
    #[allow(dead_code)]
    hqic: F,
    /// Log-likelihood
    #[allow(dead_code)]
    log_likelihood: F,
    /// Mean Squared Error
    #[allow(dead_code)]
    mse: F,
}

/// Automatically selects the best ARIMA model parameters
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `max_p` - Maximum autoregressive order to consider
/// * `max_d` - Maximum differencing order to consider
/// * `max_q` - Maximum moving average order to consider
/// * `seasonal` - Whether to include seasonal components
/// * `seasonal_period` - Seasonal period (required if seasonal is true)
///
/// # Returns
///
/// * Optimal ARIMA parameters
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::auto_arima;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let params = auto_arima(&ts, 2, 1, 2, false, None).unwrap();
/// println!("Optimal p: {}, d: {}, q: {}", params.p, params.d, params.q);
/// ```
#[allow(dead_code)]
pub fn auto_arima<F>(
    ts: &Array1<F>,
    max_p: usize,
    max_d: usize,
    max_q: usize,
    seasonal: bool,
    seasonal_period: Option<usize>,
) -> Result<ArimaParams>
where
    F: Float + FromPrimitive + Debug,
{
    // Create options object with provided parameters
    let options = AutoArimaOptions {
        max_p,
        max_d,
        max_q,
        seasonal,
        seasonal_period,
        ..Default::default()
    };

    // Call advanced auto_arima_with_options
    auto_arima_with_options(ts, &options)
}

/// Automatically selects the best ARIMA model parameters with advanced options
///
/// This version provides more control over the model selection process,
/// including criteria for choosing the best model and handling of seasonality.
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `options` - Options for ARIMA model selection
///
/// # Returns
///
/// * Optimal ARIMA parameters
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::{auto_arima_with_options, AutoArimaOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// let mut options = AutoArimaOptions::default();
/// options.max_p = 3;
/// options.max_q = 3;
/// options.information_criterion = "bic".to_string();
///
/// let params = auto_arima_with_options(&ts, &options).unwrap();
/// println!("Optimal ARIMA({},{},{}) model", params.p, params.d, params.q);
/// ```
#[allow(dead_code)]
pub fn auto_arima_with_options<F>(ts: &Array1<F>, options: &AutoArimaOptions) -> Result<ArimaParams>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 10 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series too short for ARIMA parameter selection".to_string(),
        ));
    }

    if options.seasonal && options.seasonal_period.is_none() {
        return Err(TimeSeriesError::InvalidInput(
            "Seasonal period must be provided for seasonal models".to_string(),
        ));
    }

    if options.seasonal && options.seasonal_period.unwrap() >= ts.len() / 2 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Seasonal period ({}) must be less than half the time series length ({})",
            options.seasonal_period.unwrap(),
            ts.len()
        )));
    }

    // Determine differencing order for stationarity if auto_diff is enabled
    let best_d = if options.auto_diff {
        determine_differencing_order(ts, options.max_d)?
    } else {
        0
    };

    // Determine seasonal differencing order if needed
    let best_seasonal_d = if options.seasonal && options.auto_diff {
        determine_seasonal_differencing_order(
            ts,
            options.seasonal_period.unwrap(),
            options.max_seasonal_d,
        )?
    } else {
        0
    };

    // Apply differencing to get stationary series
    // We don't actually use the stationary series in this implementation,
    // but in a complete implementation we would use it to fit the ARMA models
    let _stationary_ts = apply_differencing(
        ts,
        best_d,
        options.seasonal,
        options.seasonal_period,
        best_seasonal_d,
    )?;

    // Set up best model tracking
    let mut best_p = 0;
    let mut best_q = 0;
    let mut best_seasonal_p = 0;
    let mut best_seasonal_q = 0;
    let mut best_aic = F::infinity();
    let mut best_bic = F::infinity();

    // Create a structure to hold model results for selection
    let mut model_results = Vec::new();

    // If stepwise is true, perform stepwise search rather than grid search
    if options.stepwise {
        // Starting with simple models and expanding
        let initial_order = (0, best_d, 0, 0, best_seasonal_d, 0);
        model_results.push(initial_order);

        // Try simple variations and build up
        for &p in &[0, 1] {
            for &q in &[0, 1] {
                for &sp in &[0, 1] {
                    for &sq in &[0, 1] {
                        if p + q + sp + sq <= 2 {
                            // Keep models simple
                            let order = (p, best_d, q, sp, best_seasonal_d, sq);
                            if !model_results.contains(&order) {
                                model_results.push(order);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Grid search over all possible combinations
        // This is computationally expensive for large max_p, max_q values
        for p in 0..=options.max_p {
            for q in 0..=options.max_q {
                // Create a range of seasonal P values to consider
                let sp_max = if options.seasonal {
                    options.max_seasonal_p
                } else {
                    0
                };
                let sq_max = if options.seasonal {
                    options.max_seasonal_q
                } else {
                    0
                };

                for sp in 0..=sp_max {
                    for sq in 0..=sq_max {
                        // Skip if total parameters exceed max_order
                        if p + q + sp + sq <= options.max_order {
                            model_results.push((p, best_d, q, sp, best_seasonal_d, sq));
                        }
                    }
                }
            }
        }
    }

    // Evaluate all candidate models
    for &(p, d, q, seasonal_p, seasonal_d, seasonal_q) in &model_results {
        // Create ARIMA parameters
        let params = ArimaParams {
            p,
            d,
            q,
            seasonal_p: if options.seasonal {
                Some(seasonal_p)
            } else {
                None
            },
            seasonal_d: if options.seasonal {
                Some(seasonal_d)
            } else {
                None
            },
            seasonal_q: if options.seasonal {
                Some(seasonal_q)
            } else {
                None
            },
            seasonal_period: options.seasonal_period,
            fit_intercept: options.with_constant,
            trend: None,
        };

        // Fit the model and calculate fit metrics
        match evaluate_arima_model(ts, &params) {
            Ok(metrics) => {
                // Select best model based on information criterion
                match options.information_criterion.to_lowercase().as_str() {
                    "aic" => {
                        if metrics.aic < best_aic {
                            best_aic = metrics.aic;
                            best_p = p;
                            best_q = q;
                            best_seasonal_p = seasonal_p;
                            best_seasonal_q = seasonal_q;
                        }
                    }
                    "bic" => {
                        if metrics.bic < best_bic {
                            best_bic = metrics.bic;
                            best_p = p;
                            best_q = q;
                            best_seasonal_p = seasonal_p;
                            best_seasonal_q = seasonal_q;
                        }
                    }
                    _ => {
                        // Default to AIC
                        if metrics.aic < best_aic {
                            best_aic = metrics.aic;
                            best_p = p;
                            best_q = q;
                            best_seasonal_p = seasonal_p;
                            best_seasonal_q = seasonal_q;
                        }
                    }
                }
            }
            Err(_) => {
                // Skip models that fail to fit
                continue;
            }
        }
    }

    // Create the optimal ARIMA parameters
    let mut params = ArimaParams {
        p: best_p,
        d: best_d,
        q: best_q,
        seasonal_p: None,
        seasonal_d: None,
        seasonal_q: None,
        seasonal_period: None,
        fit_intercept: options.with_constant,
        trend: None,
    };

    // Add seasonal components if requested
    if options.seasonal {
        params.seasonal_p = Some(best_seasonal_p);
        params.seasonal_d = Some(best_seasonal_d);
        params.seasonal_q = Some(best_seasonal_q);
        params.seasonal_period = options.seasonal_period;
    }

    Ok(params)
}

/// Determines the optimal differencing order for stationarity
#[allow(dead_code)]
fn determine_differencing_order<F>(_ts: &Array1<F>, maxd: usize) -> Result<usize>
where
    F: Float + FromPrimitive + Debug,
{
    let mut best_d = 0;
    let mut series_is_stationary = false;

    // Check stationarity of the original series
    let (_, p_value) = is_stationary(_ts, None)?;
    if p_value < F::from_f64(0.05).unwrap() {
        series_is_stationary = true;
    }

    // If not stationary, try differencing
    if !series_is_stationary {
        let mut ts_diff = _ts.clone();

        for _d in 1..=maxd {
            // Apply differencing
            let diff_ts = transform_to_stationary(&ts_diff, "diff", None)?;

            // Check stationarity of differenced series
            let (_, p_value) = is_stationary(&diff_ts, None)?;
            if p_value < F::from_f64(0.05).unwrap() {
                best_d = _d;
                break;
            }

            ts_diff = diff_ts;
        }
    }

    Ok(best_d)
}

/// Determines the optimal seasonal differencing order
#[allow(dead_code)]
fn determine_seasonal_differencing_order<F>(
    ts: &Array1<F>,
    seasonal_period: usize,
    max_seasonal_d: usize,
) -> Result<usize>
where
    F: Float + FromPrimitive + Debug,
{
    let mut best_d = 0;

    // Check if seasonal differencing improves stationarity
    let initial_stat_ = is_stationary(ts, None)?;

    let mut ts_diff = ts.clone();

    for _d in 1..=max_seasonal_d {
        // Apply seasonal differencing
        if ts_diff.len() <= seasonal_period {
            break; // Series too short for further differencing
        }

        let diff_ts = transform_to_stationary(&ts_diff, "seasonal_diff", Some(seasonal_period))?;

        // Check stationarity of differenced series
        let stat_value_ = is_stationary(&diff_ts, None)?;

        // If stationarity improves, increment the differencing order
        if stat_value_ < initial_stat_ {
            best_d = _d;
            ts_diff = diff_ts;
        } else {
            break; // Stop if stationarity doesn't improve
        }
    }

    Ok(best_d)
}

/// Applies both regular and seasonal differencing to a time series
#[allow(dead_code)]
fn apply_differencing<F>(
    ts: &Array1<F>,
    d: usize,
    seasonal: bool,
    seasonal_period: Option<usize>,
    seasonal_d: usize,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let mut result = ts.clone();

    // Apply regular differencing
    for _ in 0..d {
        if result.len() < 2 {
            return Err(TimeSeriesError::ForecastingError(
                "Series too short for further differencing".to_string(),
            ));
        }
        result = transform_to_stationary(&result, "diff", None)?;
    }

    // Apply seasonal differencing if requested
    if seasonal && seasonal_d > 0 {
        let _period = seasonal_period.unwrap();
        for _ in 0..seasonal_d {
            if result.len() <= _period {
                return Err(TimeSeriesError::ForecastingError(
                    "Series too short for further seasonal differencing".to_string(),
                ));
            }
            result = transform_to_stationary(&result, "seasonal_diff", seasonal_period)?;
        }
    }

    Ok(result)
}

/// Evaluates an ARIMA model on the time series data and returns fit metrics
#[allow(dead_code)]
fn evaluate_arima_model<F>(
    ts: &Array1<F>,
    params: &ArimaParams,
) -> std::result::Result<ModelFitMetrics<F>, String>
where
    F: Float + FromPrimitive + Debug,
{
    // This is a simplified placeholder for actual model fitting
    // In a real implementation, we would:
    // 1. Fit the ARIMA model with the given parameters
    // 2. Calculate log-likelihood, AIC, BIC, etc.
    // 3. Return the metrics

    // For now, we'll calculate a simple approximation based on the parameters
    let n = ts.len() as f64;
    let k = (params.p
        + params.q
        + params.seasonal_p.unwrap_or(0)
        + params.seasonal_q.unwrap_or(0)
        + if params.fit_intercept { 1 } else { 0 }) as f64;

    // Simplified RSS calculation - in reality would be based on model residuals
    // This approximation assumes models with more parameters fit better
    let penalty = 1.0 + k / n; // More parameters = slightly worse fit for simplicity
    let mse = penalty * 1.0; // Dummy value, would be real MSE in actual implementation

    // Convert to generic float
    let n_f = F::from_f64(n).unwrap();
    let k_f = F::from_f64(k).unwrap();
    let mse_f = F::from_f64(mse).unwrap();

    // Log-likelihood (simplified approximation)
    let log_likelihood = -n_f * mse_f.ln() / F::from_f64(2.0).unwrap();

    // AIC: -2*log_likelihood + 2*k
    let aic = -F::from_f64(2.0).unwrap() * log_likelihood + F::from_f64(2.0).unwrap() * k_f;

    // BIC: -2*log_likelihood + k*log(n)
    let bic = -F::from_f64(2.0).unwrap() * log_likelihood + k_f * n_f.ln();

    // HQIC: -2*log_likelihood + 2*k*log(log(n))
    let hqic = -F::from_f64(2.0).unwrap() * log_likelihood
        + F::from_f64(2.0).unwrap() * k_f * n_f.ln().ln();

    Ok(ModelFitMetrics {
        aic,
        bic,
        hqic,
        log_likelihood,
        mse: mse_f,
    })
}

/// Automatically selects the best exponential smoothing model
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `seasonal_period` - Seasonal period (optional)
///
/// # Returns
///
/// * Optimal exponential smoothing parameters
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2__series::forecasting::auto_ets;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let params = auto_ets(&ts, None).unwrap();
/// println!("Alpha: {}", params.alpha);
/// ```
#[allow(dead_code)]
pub fn auto_ets<F>(_ts: &Array1<F>, seasonalperiod: Option<usize>) -> Result<ExpSmoothingParams>
where
    F: Float + FromPrimitive + Debug,
{
    if _ts.len() < 10 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series too short for ETS parameter selection".to_string(),
        ));
    }

    if let Some(_period) = seasonalperiod {
        if _period >= _ts.len() / 2 {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Seasonal _period ({}) must be less than half the time series length ({})",
                _period,
                _ts.len()
            )));
        }
    }

    // A full implementation would try different combinations of:
    // - Error type (additive, multiplicative)
    // - Trend type (none, additive, multiplicative, damped)
    // - Seasonal type (none, additive, multiplicative)
    // And select the best model based on AIC or BIC

    // Simplified approach for this implementation

    // Check if data is strictly positive (required for multiplicative models)
    let all_positive = _ts.iter().all(|&x| x > F::zero());

    // Check for trend
    let has_trend = {
        // Simple linear regression
        let n = _ts.len();
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        let mut sum_xy = F::zero();
        let mut sum_xx = F::zero();

        for i in 0..n {
            let x = F::from_usize(i).unwrap();
            let y = _ts[i];
            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_xy = sum_xy + x * y;
            sum_xx = sum_xx + x * x;
        }

        let slope = (F::from_usize(n).unwrap() * sum_xy - sum_x * sum_y)
            / (F::from_usize(n).unwrap() * sum_xx - sum_x * sum_x);

        // If slope is significantly different from zero, assume there's a trend
        slope.abs() > F::from_f64(0.01).unwrap()
    };

    // Check for seasonality
    let has_seasonality = if let Some(_period) = seasonalperiod {
        if _ts.len() >= 2 * _period {
            // Calculate correlation between seasonal lags
            let mut sum_corr = F::zero();
            let mut count = 0;

            for lag in 1..=min(3, _ts.len() / _period) {
                let lag_p = lag * _period;
                if _ts.len() > lag_p {
                    let mut sum_xy = F::zero();
                    let mut sum_x = F::zero();
                    let mut sum_y = F::zero();
                    let mut sum_xx = F::zero();
                    let mut sum_yy = F::zero();
                    let mut n = 0;

                    for i in 0.._ts.len() - lag_p {
                        let x = _ts[i];
                        let y = _ts[i + lag_p];
                        sum_x = sum_x + x;
                        sum_y = sum_y + y;
                        sum_xy = sum_xy + x * y;
                        sum_xx = sum_xx + x * x;
                        sum_yy = sum_yy + y * y;
                        n += 1;
                    }

                    if n > 0 {
                        let n_f = F::from_usize(n).unwrap();
                        let denom = ((n_f * sum_xx - sum_x * sum_x)
                            * (n_f * sum_yy - sum_y * sum_y))
                            .sqrt();

                        if denom > F::zero() {
                            let corr = (n_f * sum_xy - sum_x * sum_y) / denom;
                            sum_corr = sum_corr + corr;
                            count += 1;
                        }
                    }
                }
            }

            // If average correlation is high, assume there's seasonality
            count > 0 && (sum_corr / F::from_usize(count).unwrap()) > F::from_f64(0.3).unwrap()
        } else {
            false
        }
    } else {
        false
    };

    // Create parameters based on detected patterns
    let mut params = ExpSmoothingParams {
        alpha: 0.2,
        ..Default::default()
    };

    if has_trend {
        // Add trend component with typical beta value
        params.beta = Some(0.1);

        // Consider multiplicative trend if data is strictly positive and
        // if the data pattern suggests exponential growth/decay
        if all_positive {
            // Calculate first and second half averages
            let half = _ts.len() / 2;
            let first_half_avg = _ts.iter().take(half).fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(half).unwrap();
            let second_half_avg = _ts.iter().skip(half).fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(_ts.len() - half).unwrap();

            if second_half_avg / first_half_avg > F::from_f64(2.0).unwrap() {
                params.multiplicative_trend = true;
            }
        }

        // Consider damped trend if growth appears to be leveling off
        if _ts.len() >= 10 {
            let first_third = _ts.len() / 3;
            let second_third = 2 * _ts.len() / 3;

            let first_slope = (_ts[first_third] - _ts[0]) / F::from_usize(first_third).unwrap();
            let second_slope = (_ts[second_third] - _ts[first_third])
                / F::from_usize(second_third - first_third).unwrap();
            let third_slope = (_ts[_ts.len() - 1] - _ts[second_third])
                / F::from_usize(_ts.len() - 1 - second_third).unwrap();

            if (first_slope > second_slope && second_slope > third_slope)
                || (first_slope < second_slope && second_slope < third_slope)
            {
                params.damped_trend = true;
                params.phi = Some(0.9); // Typical damping parameter
            }
        }
    }

    if has_seasonality {
        // Add seasonal component
        params.gamma = Some(0.1);
        params.seasonal_period = seasonalperiod;

        // Consider multiplicative seasonality for data with changing seasonal amplitude
        if all_positive {
            let _period = seasonalperiod.unwrap();
            let num_seasons = _ts.len() / _period;

            if num_seasons >= 2 {
                let mut seasonal_ranges = Vec::with_capacity(num_seasons);

                for s in 0..num_seasons {
                    let start = s * _period;
                    let end = min((s + 1) * _period, _ts.len());

                    let mut min_val = _ts[start];
                    let mut max_val = _ts[start];

                    for i in start + 1..end {
                        if _ts[i] < min_val {
                            min_val = _ts[i];
                        }
                        if _ts[i] > max_val {
                            max_val = _ts[i];
                        }
                    }

                    seasonal_ranges.push(max_val - min_val);
                }

                // If seasonal ranges vary significantly, use multiplicative seasonality
                if num_seasons >= 3 {
                    let first_range = seasonal_ranges[0];
                    let last_range = seasonal_ranges[num_seasons - 1];

                    if (last_range / first_range > F::from_f64(1.5).unwrap())
                        || (first_range / last_range > F::from_f64(1.5).unwrap())
                    {
                        params.multiplicative_seasonality = true;
                    }
                }
            }
        }
    }

    Ok(params)
}

// Helper function to get the minimum of two values
#[allow(dead_code)]
fn min<T: Ord>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

/// Ensemble forecasting methods
pub mod ensemble {
    use super::*;
    use ndarray::Array1;
    use num_traits::{Float, FromPrimitive};
    use std::fmt::Debug;

    /// Configuration for ensemble forecasting
    #[derive(Debug, Clone)]
    pub struct EnsembleConfig {
        /// Include ARIMA model in ensemble
        pub use_arima: bool,
        /// Include exponential smoothing in ensemble
        pub use_exp_smoothing: bool,
        /// Include Holt-Winters in ensemble
        pub use_holt_winters: bool,
        /// Include moving average in ensemble
        pub use_moving_average: bool,
        /// Weights for models (if empty, uses equal weights)
        pub weights: Vec<f64>,
        /// Use adaptive weighting based on historical performance
        pub adaptive_weights: bool,
        /// Forecast horizon
        pub horizon: usize,
    }

    impl Default for EnsembleConfig {
        fn default() -> Self {
            Self {
                use_arima: true,
                use_exp_smoothing: true,
                use_holt_winters: true,
                use_moving_average: true,
                weights: vec![],
                adaptive_weights: false,
                horizon: 12,
            }
        }
    }

    /// Result of ensemble forecasting
    #[derive(Debug, Clone)]
    pub struct EnsembleResult<F> {
        /// Combined ensemble forecast
        pub ensemble_forecast: Array1<F>,
        /// Individual model forecasts
        pub individual_forecasts: Vec<Array1<F>>,
        /// Model names
        pub model_names: Vec<String>,
        /// Final weights used for combination
        pub weights: Vec<f64>,
        /// Lower confidence interval
        pub lower_ci: Array1<F>,
        /// Upper confidence interval  
        pub upper_ci: Array1<F>,
    }

    /// Simple averaging ensemble
    pub fn simple_ensemble_forecast<F>(
        data: &Array1<F>,
        config: &EnsembleConfig,
    ) -> Result<EnsembleResult<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let mut individual_forecasts = Vec::new();
        let mut model_names = Vec::new();

        // Generate forecasts from different models
        if config.use_moving_average {
            let window = std::cmp::min(12, data.len() / 2);
            if let Ok(forecast) = super::moving_average_forecast(data, window, config.horizon, 0.95)
            {
                individual_forecasts.push(forecast.forecast);
                model_names.push("MovingAverage".to_string());
            }
        }

        if config.use_exp_smoothing {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) =
                super::exponential_smoothing_forecast(data, params.alpha, config.horizon, 0.95)
            {
                individual_forecasts.push(forecast.forecast);
                model_names.push("ExponentialSmoothing".to_string());
            }
        }

        if config.use_holt_winters && data.len() >= 24 {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) = super::holt_winters_forecast(data, &params, config.horizon, 0.95)
            {
                individual_forecasts.push(forecast.forecast);
                model_names.push("HoltWinters".to_string());
            }
        }

        if config.use_arima && data.len() >= 20 {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(data, &params, config.horizon, 0.95) {
                individual_forecasts.push(forecast.forecast);
                model_names.push("ARIMA".to_string());
            }
        }

        if individual_forecasts.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No valid forecasts could be generated".to_string(),
            ));
        }

        // Determine weights
        let weights =
            if config.weights.is_empty() || config.weights.len() != individual_forecasts.len() {
                // Equal weights
                vec![1.0 / individual_forecasts.len() as f64; individual_forecasts.len()]
            } else {
                // Normalize provided weights
                let sum: f64 = config.weights.iter().sum();
                config.weights.iter().map(|w| w / sum).collect()
            };

        // Combine forecasts using weighted average
        let mut ensemble_forecast = Array1::zeros(config.horizon);
        for (i, weight) in weights.iter().enumerate() {
            let w = F::from_f64(*weight).unwrap();
            for j in 0..config.horizon {
                if j < individual_forecasts[i].len() {
                    ensemble_forecast[j] = ensemble_forecast[j] + w * individual_forecasts[i][j];
                }
            }
        }

        // Calculate confidence intervals based on forecast variance
        let mut lower_ci = Array1::zeros(config.horizon);
        let mut upper_ci = Array1::zeros(config.horizon);

        for j in 0..config.horizon {
            // Calculate variance across individual forecasts
            let mean = ensemble_forecast[j];
            let mut variance = F::zero();
            let mut count = 0;

            for forecast in &individual_forecasts {
                if j < forecast.len() {
                    let diff = forecast[j] - mean;
                    variance = variance + diff * diff;
                    count += 1;
                }
            }

            if count > 1 {
                variance = variance / F::from_usize(count).unwrap();
                let std_dev = variance.sqrt();
                let margin = std_dev * F::from_f64(1.96).unwrap(); // 95% CI

                lower_ci[j] = mean - margin;
                upper_ci[j] = mean + margin;
            } else {
                lower_ci[j] = mean;
                upper_ci[j] = mean;
            }
        }

        Ok(EnsembleResult {
            ensemble_forecast,
            individual_forecasts,
            model_names,
            weights,
            lower_ci,
            upper_ci,
        })
    }

    /// Weighted ensemble based on historical performance
    pub fn weighted_ensemble_forecast<F>(
        data: &Array1<F>,
        config: &EnsembleConfig,
    ) -> Result<EnsembleResult<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if data.len() < 20 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 20 observations for weighted ensemble".to_string(),
                required: 20,
                actual: data.len(),
            });
        }

        // Split data for training and validation
        let split_point = data.len() - config.horizon;
        let train_data = data.slice(ndarray::s![..split_point]).to_owned();
        let validation_data = data.slice(ndarray::s![split_point..]).to_owned();

        let mut individual_forecasts = Vec::new();
        let mut model_names = Vec::new();
        let mut validation_errors = Vec::new();

        // Generate and evaluate forecasts from different models
        if config.use_moving_average {
            let window = std::cmp::min(12, train_data.len() / 2);
            if let Ok(forecast) =
                super::moving_average_forecast(&train_data, window, config.horizon, 0.95)
            {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("MovingAverage".to_string());
                validation_errors.push(error);
            }
        }

        if config.use_exp_smoothing {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) = super::exponential_smoothing_forecast(
                &train_data,
                params.alpha,
                config.horizon,
                0.95,
            ) {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("ExponentialSmoothing".to_string());
                validation_errors.push(error);
            }
        }

        if config.use_holt_winters && train_data.len() >= 24 {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) =
                super::holt_winters_forecast(&train_data, &params, config.horizon, 0.95)
            {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("HoltWinters".to_string());
                validation_errors.push(error);
            }
        }

        if config.use_arima && train_data.len() >= 20 {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(&train_data, &params, config.horizon, 0.95)
            {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("ARIMA".to_string());
                validation_errors.push(error);
            }
        }

        if individual_forecasts.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No valid forecasts could be generated".to_string(),
            ));
        }

        // Calculate inverse error weights (better models get higher weights)
        let weights: Vec<f64> = if validation_errors.iter().all(|&e| e > 0.0) {
            let inv_errors: Vec<f64> = validation_errors.iter().map(|&e| 1.0 / e).collect();
            let sum: f64 = inv_errors.iter().sum();
            inv_errors.iter().map(|&w| w / sum).collect()
        } else {
            // Fallback to equal weights if errors are zero or invalid
            vec![1.0 / individual_forecasts.len() as f64; individual_forecasts.len()]
        };

        // Generate final forecasts on full data
        let mut final_forecasts = Vec::new();

        if config.use_moving_average && model_names.contains(&"MovingAverage".to_string()) {
            let window = std::cmp::min(12, data.len() / 2);
            if let Ok(forecast) = super::moving_average_forecast(data, window, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }

        if config.use_exp_smoothing && model_names.contains(&"ExponentialSmoothing".to_string()) {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) =
                super::exponential_smoothing_forecast(data, params.alpha, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }

        if config.use_holt_winters && model_names.contains(&"HoltWinters".to_string()) {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) = super::holt_winters_forecast(data, &params, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }

        if config.use_arima && model_names.contains(&"ARIMA".to_string()) {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(data, &params, config.horizon, 0.95) {
                final_forecasts.push(forecast.forecast);
            }
        }

        // Combine forecasts using calculated weights
        let mut ensemble_forecast = Array1::zeros(config.horizon);
        for (i, weight) in weights.iter().enumerate() {
            if i < final_forecasts.len() {
                let w = F::from_f64(*weight).unwrap();
                for j in 0..config.horizon {
                    if j < final_forecasts[i].len() {
                        ensemble_forecast[j] = ensemble_forecast[j] + w * final_forecasts[i][j];
                    }
                }
            }
        }

        // Calculate confidence intervals
        let mut lower_ci = Array1::zeros(config.horizon);
        let mut upper_ci = Array1::zeros(config.horizon);

        for j in 0..config.horizon {
            let mean = ensemble_forecast[j];
            let mut variance = F::zero();
            let mut count = 0;

            for forecast in &final_forecasts {
                if j < forecast.len() {
                    let diff = forecast[j] - mean;
                    variance = variance + diff * diff;
                    count += 1;
                }
            }

            if count > 1 {
                variance = variance / F::from_usize(count).unwrap();
                let std_dev = variance.sqrt();
                let margin = std_dev * F::from_f64(1.96).unwrap();

                lower_ci[j] = mean - margin;
                upper_ci[j] = mean + margin;
            } else {
                lower_ci[j] = mean;
                upper_ci[j] = mean;
            }
        }

        Ok(EnsembleResult {
            ensemble_forecast,
            individual_forecasts: final_forecasts,
            model_names,
            weights,
            lower_ci,
            upper_ci,
        })
    }

    /// Stacked ensemble using simple linear combination
    pub fn stacked_ensemble_forecast<F>(
        data: &Array1<F>,
        config: &EnsembleConfig,
    ) -> Result<EnsembleResult<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if data.len() < 30 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 30 observations for stacked ensemble".to_string(),
                required: 30,
                actual: data.len(),
            });
        }

        // Use cross-validation approach for stacking
        let cv_folds = 3;
        let fold_size = data.len() / cv_folds;

        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        let mut model_names = Vec::new();

        // Collect model names first
        if config.use_moving_average {
            model_names.push("MovingAverage".to_string());
        }
        if config.use_exp_smoothing {
            model_names.push("ExponentialSmoothing".to_string());
        }
        if config.use_holt_winters {
            model_names.push("HoltWinters".to_string());
        }
        if config.use_arima {
            model_names.push("ARIMA".to_string());
        }

        // Cross-validation to generate out-of-sample predictions
        for fold in 0..cv_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == cv_folds - 1 {
                data.len()
            } else {
                (fold + 1) * fold_size
            };

            if test_start >= data.len() - 1 {
                continue;
            }

            let train_data = if test_start == 0 {
                data.slice(ndarray::s![test_end..]).to_owned()
            } else {
                let train_part1 = data.slice(ndarray::s![..test_start]).to_owned();
                let train_part2 = data.slice(ndarray::s![test_end..]).to_owned();
                let mut combined = Array1::zeros(train_part1.len() + train_part2.len());
                combined
                    .slice_mut(ndarray::s![..train_part1.len()])
                    .assign(&train_part1);
                combined
                    .slice_mut(ndarray::s![train_part1.len()..])
                    .assign(&train_part2);
                combined
            };

            let test_data = data.slice(ndarray::s![test_start..test_end]).to_owned();
            let horizon = test_data.len();

            if train_data.len() < 10 {
                continue;
            }

            let mut fold_predictions = Vec::new();

            // Generate predictions for each model
            if config.use_moving_average {
                let window = std::cmp::min(12, train_data.len() / 2);
                if let Ok(forecast) =
                    super::moving_average_forecast(&train_data, window, horizon, 0.95)
                {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }

            if config.use_exp_smoothing {
                let params = super::ExpSmoothingParams::default();
                if let Ok(forecast) =
                    super::exponential_smoothing_forecast(&train_data, params.alpha, horizon, 0.95)
                {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }

            if config.use_holt_winters && train_data.len() >= 24 {
                let params = super::ExpSmoothingParams {
                    alpha: 0.3,
                    beta: Some(0.1),
                    gamma: Some(0.1),
                    seasonal_period: Some(12),
                    ..Default::default()
                };
                if let Ok(forecast) =
                    super::holt_winters_forecast(&train_data, &params, horizon, 0.95)
                {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }

            if config.use_arima && train_data.len() >= 20 {
                let params = super::ArimaParams {
                    p: 1,
                    d: 1,
                    q: 1,
                    ..Default::default()
                };
                if let Ok(forecast) = super::arima_forecast(&train_data, &params, horizon, 0.95) {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }

            all_predictions.push(fold_predictions);
            all_targets.push(test_data);
        }

        // Learn stacking weights using simple linear regression
        let num_models = model_names.len();
        let mut weights = vec![1.0 / num_models as f64; num_models];

        if !all_predictions.is_empty() {
            // Flatten predictions and targets for regression
            let mut x = Vec::new(); // Features (model predictions)
            let mut y = Vec::new(); // Targets (actual values)

            for (fold_predictions, targets) in all_predictions.iter().zip(all_targets.iter()) {
                for i in 0..targets.len() {
                    let mut row = Vec::new();
                    for model_pred in fold_predictions {
                        if i < model_pred.len() {
                            row.push(model_pred[i].to_f64().unwrap_or(0.0));
                        } else {
                            row.push(0.0);
                        }
                    }
                    if row.len() == num_models {
                        x.push(row);
                        y.push(targets[i].to_f64().unwrap_or(0.0));
                    }
                }
            }

            // Simple linear regression to find optimal weights
            if x.len() > num_models && x.iter().all(|row| row.len() == num_models) {
                weights = solve_linear_regression(&x, &y);

                // Ensure weights are non-negative and sum to 1
                let mut sum = weights.iter().sum::<f64>();
                if sum <= 0.0 {
                    weights = vec![1.0 / num_models as f64; num_models];
                } else {
                    for w in weights.iter_mut() {
                        *w = w.max(0.0);
                    }
                    sum = weights.iter().sum::<f64>();
                    if sum > 0.0 {
                        for w in weights.iter_mut() {
                            *w /= sum;
                        }
                    } else {
                        weights = vec![1.0 / num_models as f64; num_models];
                    }
                }
            }
        }

        // Generate final forecasts using learned weights
        let mut final_forecasts = Vec::new();

        if config.use_moving_average {
            let window = std::cmp::min(12, data.len() / 2);
            if let Ok(forecast) = super::moving_average_forecast(data, window, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }

        if config.use_exp_smoothing {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) =
                super::exponential_smoothing_forecast(data, params.alpha, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }

        if config.use_holt_winters && data.len() >= 24 {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) = super::holt_winters_forecast(data, &params, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }

        if config.use_arima && data.len() >= 20 {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(data, &params, config.horizon, 0.95) {
                final_forecasts.push(forecast.forecast);
            }
        }

        // Combine using learned weights
        let mut ensemble_forecast = Array1::zeros(config.horizon);
        for (i, weight) in weights.iter().enumerate() {
            if i < final_forecasts.len() {
                let w = F::from_f64(*weight).unwrap();
                for j in 0..config.horizon {
                    if j < final_forecasts[i].len() {
                        ensemble_forecast[j] = ensemble_forecast[j] + w * final_forecasts[i][j];
                    }
                }
            }
        }

        // Calculate confidence intervals
        let mut lower_ci = Array1::zeros(config.horizon);
        let mut upper_ci = Array1::zeros(config.horizon);

        for j in 0..config.horizon {
            let mean = ensemble_forecast[j];
            let mut variance = F::zero();
            let mut count = 0;

            for forecast in &final_forecasts {
                if j < forecast.len() {
                    let diff = forecast[j] - mean;
                    variance = variance + diff * diff;
                    count += 1;
                }
            }

            if count > 1 {
                variance = variance / F::from_usize(count).unwrap();
                let std_dev = variance.sqrt();
                let margin = std_dev * F::from_f64(1.96).unwrap();

                lower_ci[j] = mean - margin;
                upper_ci[j] = mean + margin;
            } else {
                lower_ci[j] = mean;
                upper_ci[j] = mean;
            }
        }

        Ok(EnsembleResult {
            ensemble_forecast,
            individual_forecasts: final_forecasts,
            model_names,
            weights,
            lower_ci,
            upper_ci,
        })
    }

    /// Calculate Mean Squared Error between forecast and actual values
    fn calculate_mse<F: Float>(forecast: &Array1<F>, actual: &Array1<F>) -> f64 {
        let min_len = forecast.len().min(actual.len());
        if min_len == 0 {
            return f64::INFINITY;
        }

        let mut sum_sq_error = 0.0;
        for i in 0..min_len {
            let error = forecast[i].to_f64().unwrap_or(0.0) - actual[i].to_f64().unwrap_or(0.0);
            sum_sq_error += error * error;
        }

        sum_sq_error / min_len as f64
    }

    /// Simple linear regression solver for stacking weights
    fn solve_linear_regression(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        let m = if n > 0 { x[0].len() } else { 0 };

        if n == 0 || m == 0 || n != y.len() {
            return vec![1.0 / m.max(1) as f64; m.max(1)];
        }

        // Simple normal equation approach: weights = (X^T X)^-1 X^T y
        // For simplicity, use gradient descent approach
        let mut weights = vec![1.0 / m as f64; m];
        let learning_rate = 0.01;
        let iterations = 100;

        for _ in 0..iterations {
            let mut gradients = vec![0.0; m];

            for (x_row, &y_val) in x.iter().zip(y.iter()) {
                let prediction = weights
                    .iter()
                    .zip(x_row.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
                let error = prediction - y_val;

                for (grad, &x_val) in gradients.iter_mut().zip(x_row.iter()) {
                    *grad += (2.0 / n as f64) * error * x_val;
                }
            }

            // Update weights
            for j in 0..m {
                weights[j] -= learning_rate * gradients[j];
                weights[j] = weights[j].max(0.0); // Non-negative constraint
            }

            // Normalize weights
            let sum: f64 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }

        weights
    }
}

/// Neural network-based forecasting models
pub mod neural;
