//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::functions::const_f64;
use super::functions_3::evaluate_arima_model;
use super::types::{ArimaParams, AutoArimaOptions, ForecastResult};
use crate::utils::{is_stationary, transform_to_stationary};

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
/// use scirs2_core::ndarray::array;
/// use scirs2_series::forecasting::{arima_forecast, ArimaParams};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// let mut params = ArimaParams::default();
/// params.p = 1;
/// params.d = 1;
/// params.q = 1;
///
/// let result = arima_forecast(&ts, &params, 5, 0.95).expect("Example/test failed");
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
    if let Some(period) = params.seasonal_period {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::ForecastingError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    }
    let mut data = ts.clone();
    for _ in 0..params.d {
        let mut diff_data = Array1::zeros(data.len() - 1);
        for i in 0..data.len() - 1 {
            diff_data[i] = data[i + 1] - data[i];
        }
        data = diff_data;
    }
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
    let ar_coeffs: Vec<F> = if params.p > 0 {
        let mut coeffs = Vec::with_capacity(params.p);
        for i in 0..params.p {
            coeffs.push(
                F::from_f64(0.8 / (i + 1) as f64).expect("Failed to convert coefficient to float"),
            );
        }
        coeffs
    } else {
        vec![]
    };
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);
    let n = data.len();
    for h in 0..horizon {
        let mut pred = F::zero();
        for i in 0..params.p {
            if h >= i && h - i < n {
                pred = pred + ar_coeffs[i] * data[n - 1 - (h - i)];
            } else if h >= i {
                pred = pred + ar_coeffs[i] * forecast[h - i - 1];
            }
        }
        forecast[h] = pred;
    }
    for _ in 0..params.d {
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
    let std_err = const_f64::<F>(0.5);
    let z_score = match conf_level {
        c if c >= 0.99 => const_f64::<F>(2.576),
        c if c >= 0.95 => const_f64::<F>(1.96),
        c if c >= 0.90 => const_f64::<F>(1.645),
        _ => const_f64::<F>(1.0),
    };
    for h in 0..horizon {
        let ci_width = z_score
            * std_err
            * (F::from_usize(h + 1).expect("Failed to convert usize to float")).sqrt();
        lower_ci[h] = forecast[h] - ci_width;
        upper_ci[h] = forecast[h] + ci_width;
    }
    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
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
/// use scirs2_series::forecasting::auto_arima;
///
/// let ts: scirs2_core::ndarray::Array1<f64> = (1..=50).map(|i| i as f64 + (i as f64 * 0.3).sin()).collect();
/// let params = auto_arima(&ts, 2, 1, 2, false, None).expect("Example/test failed");
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
    F: Float + FromPrimitive + Debug + 'static,
{
    let options = AutoArimaOptions {
        max_p,
        max_d,
        max_q,
        seasonal,
        seasonal_period,
        ..Default::default()
    };
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
/// use scirs2_series::forecasting::{auto_arima_with_options, AutoArimaOptions};
///
/// let ts: scirs2_core::ndarray::Array1<f64> = (1..=50).map(|i| i as f64 + (i as f64 * 0.3).sin()).collect();
///
/// let mut options = AutoArimaOptions::default();
/// options.max_p = 3;
/// options.max_q = 3;
/// options.information_criterion = "bic".to_string();
///
/// let params = auto_arima_with_options(&ts, &options).expect("Example/test failed");
/// println!("Optimal ARIMA({},{},{}) model", params.p, params.d, params.q);
/// ```
#[allow(dead_code)]
pub fn auto_arima_with_options<F>(ts: &Array1<F>, options: &AutoArimaOptions) -> Result<ArimaParams>
where
    F: Float + FromPrimitive + Debug + 'static,
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
    if options.seasonal
        && options.seasonal_period.expect("Seasonal period required") >= ts.len() / 2
    {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Seasonal period ({}) must be less than half the time series length ({})",
            options.seasonal_period.expect("Seasonal period required"),
            ts.len()
        )));
    }
    let best_d = if options.auto_diff {
        determine_differencing_order(ts, options.max_d)?
    } else {
        0
    };
    let best_seasonal_d = if options.seasonal && options.auto_diff {
        determine_seasonal_differencing_order(
            ts,
            options.seasonal_period.expect("Seasonal period required"),
            options.max_seasonal_d,
        )?
    } else {
        0
    };
    let _stationary_ts = apply_differencing(
        ts,
        best_d,
        options.seasonal,
        options.seasonal_period,
        best_seasonal_d,
    )?;
    let mut best_p = 0;
    let mut best_q = 0;
    let mut best_seasonal_p = 0;
    let mut best_seasonal_q = 0;
    let mut best_aic = F::infinity();
    let mut best_bic = F::infinity();
    let mut model_results = Vec::new();
    if options.stepwise {
        let initial_order = (0, best_d, 0, 0, best_seasonal_d, 0);
        model_results.push(initial_order);
        for &p in &[0, 1] {
            for &q in &[0, 1] {
                for &sp in &[0, 1] {
                    for &sq in &[0, 1] {
                        if p + q + sp + sq <= 2 {
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
        for p in 0..=options.max_p {
            for q in 0..=options.max_q {
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
                        if p + q + sp + sq <= options.max_order {
                            model_results.push((p, best_d, q, sp, best_seasonal_d, sq));
                        }
                    }
                }
            }
        }
    }
    for &(p, d, q, seasonal_p, seasonal_d, seasonal_q) in &model_results {
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
        match evaluate_arima_model(ts, &params) {
            Ok(metrics) => match options.information_criterion.to_lowercase().as_str() {
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
                    if metrics.aic < best_aic {
                        best_aic = metrics.aic;
                        best_p = p;
                        best_q = q;
                        best_seasonal_p = seasonal_p;
                        best_seasonal_q = seasonal_q;
                    }
                }
            },
            Err(_) => {
                continue;
            }
        }
    }
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
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut best_d = 0;
    let mut series_is_stationary = false;
    let (_, p_value) = is_stationary(_ts, None)?;
    if p_value < const_f64::<F>(0.05) {
        series_is_stationary = true;
    }
    if !series_is_stationary {
        let mut ts_diff = _ts.clone();
        for _d in 1..=maxd {
            let diff_ts = transform_to_stationary(&ts_diff, "diff", None)?;
            let (_, p_value) = is_stationary(&diff_ts, None)?;
            if p_value < const_f64::<F>(0.05) {
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
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut best_d = 0;
    let initial_stat_ = is_stationary(ts, None)?;
    let mut ts_diff = ts.clone();
    for _d in 1..=max_seasonal_d {
        if ts_diff.len() <= seasonal_period {
            break;
        }
        let diff_ts = transform_to_stationary(&ts_diff, "seasonal_diff", Some(seasonal_period))?;
        let stat_value_ = is_stationary(&diff_ts, None)?;
        if stat_value_ < initial_stat_ {
            best_d = _d;
            ts_diff = diff_ts;
        } else {
            break;
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
    for _ in 0..d {
        if result.len() < 2 {
            return Err(TimeSeriesError::ForecastingError(
                "Series too short for further differencing".to_string(),
            ));
        }
        result = transform_to_stationary(&result, "diff", None)?;
    }
    if seasonal && seasonal_d > 0 {
        let _period = seasonal_period.expect("Seasonal period required");
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
