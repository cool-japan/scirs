//! Forecast Evaluation Metrics
//!
//! Comprehensive forecast accuracy measures and statistical tests for comparing
//! forecast performance. Implements standard metrics from the forecasting literature:
//!
//! - **Point forecast metrics**: MAE, MAPE, sMAPE, MASE, RMSE, WAPE
//! - **Interval forecast metrics**: Coverage probability, Winkler score
//! - **Comparative tests**: Diebold-Mariano test for equal predictive accuracy
//!
//! # References
//!
//! - Hyndman, R.J. & Koehler, A.B. (2006) "Another look at measures of forecast accuracy"
//! - Diebold, F.X. & Mariano, R.S. (1995) "Comparing Predictive Accuracy"
//! - Winkler, R.L. (1972) "A Decision-Theoretic Approach to Interval Estimation"

use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Point forecast accuracy metrics
// ---------------------------------------------------------------------------

/// Compute Mean Absolute Error (MAE)
///
/// MAE = (1/n) * sum(|actual - forecast|)
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast` - Predicted values
///
/// # Returns
/// Mean absolute error (non-negative)
pub fn mae<S1, S2, F>(actual: &ArrayBase<S1, Ix1>, forecast: &ArrayBase<S2, Ix1>) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "MAE")?;

    let n = F::from_usize(actual.len())
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;

    let sum: F = actual
        .iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| (a - f).abs())
        .fold(F::zero(), |acc, x| acc + x);

    Ok(sum / n)
}

/// Compute Mean Absolute Percentage Error (MAPE)
///
/// MAPE = (100/n) * sum(|actual - forecast| / |actual|)
///
/// Note: MAPE is undefined when actual values are zero. Such observations
/// are skipped and a warning is emitted via the count of valid observations.
///
/// # Arguments
/// * `actual` - Observed values (must be non-zero for meaningful results)
/// * `forecast` - Predicted values
///
/// # Returns
/// MAPE as a percentage (0-100+)
pub fn mape<S1, S2, F>(actual: &ArrayBase<S1, Ix1>, forecast: &ArrayBase<S2, Ix1>) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "MAPE")?;

    let hundred = F::from_f64(100.0).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert constant".to_string())
    })?;

    let mut sum = F::zero();
    let mut valid_count = 0usize;

    for (&a, &f) in actual.iter().zip(forecast.iter()) {
        let abs_a = a.abs();
        if abs_a > F::epsilon() {
            sum = sum + (a - f).abs() / abs_a;
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return Err(TimeSeriesError::ComputationError(
            "MAPE undefined: all actual values are zero".to_string(),
        ));
    }

    let n = F::from_usize(valid_count)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert count".to_string()))?;

    Ok(hundred * sum / n)
}

/// Compute Symmetric Mean Absolute Percentage Error (sMAPE)
///
/// sMAPE = (200/n) * sum(|actual - forecast| / (|actual| + |forecast|))
///
/// sMAPE avoids the asymmetry of standard MAPE and handles zero values
/// better. Range: [0, 200].
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast` - Predicted values
///
/// # Returns
/// sMAPE as a percentage (0-200)
pub fn smape<S1, S2, F>(actual: &ArrayBase<S1, Ix1>, forecast: &ArrayBase<S2, Ix1>) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "sMAPE")?;

    let two_hundred = F::from_f64(200.0).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert constant".to_string())
    })?;

    let mut sum = F::zero();
    let mut valid_count = 0usize;

    for (&a, &f) in actual.iter().zip(forecast.iter()) {
        let denom = a.abs() + f.abs();
        if denom > F::epsilon() {
            sum = sum + (a - f).abs() / denom;
            valid_count += 1;
        }
        // Both zero => perfect forecast, contributes 0
    }

    if valid_count == 0 {
        // All pairs are (0, 0) => perfect forecasts
        return Ok(F::zero());
    }

    let n = F::from_usize(actual.len())
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert count".to_string()))?;

    Ok(two_hundred * sum / n)
}

/// Compute Mean Absolute Scaled Error (MASE)
///
/// MASE = MAE / MAE_naive
///
/// where MAE_naive is the in-sample MAE of the naive (random walk) forecast.
/// A MASE < 1 indicates the forecast outperforms the naive method.
///
/// For seasonal series, use `mase_seasonal` with the appropriate period.
///
/// # Arguments
/// * `actual` - Observed (out-of-sample) values
/// * `forecast` - Predicted values
/// * `training` - In-sample (training) data used to compute the naive scale
///
/// # Returns
/// MASE value (positive; < 1 is better than naive)
pub fn mase<S1, S2, S3, F>(
    actual: &ArrayBase<S1, Ix1>,
    forecast: &ArrayBase<S2, Ix1>,
    training: &ArrayBase<S3, Ix1>,
) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "MASE")?;
    check_minimum_length(training.len(), 2, "MASE (training)")?;

    // Naive in-sample MAE: average |y_t - y_{t-1}|
    let naive_scale = naive_mae(training, 1)?;

    if naive_scale <= F::epsilon() {
        return Err(TimeSeriesError::ComputationError(
            "MASE undefined: training data has zero naive MAE (constant series)".to_string(),
        ));
    }

    let forecast_mae = mae(actual, forecast)?;
    Ok(forecast_mae / naive_scale)
}

/// Compute seasonal MASE
///
/// Uses seasonal differences (lag = period) instead of first differences
/// for the naive scaling factor.
///
/// # Arguments
/// * `actual` - Observed (out-of-sample) values
/// * `forecast` - Predicted values
/// * `training` - In-sample training data
/// * `period` - Seasonal period (e.g. 12 for monthly data with yearly seasonality)
pub fn mase_seasonal<S1, S2, S3, F>(
    actual: &ArrayBase<S1, Ix1>,
    forecast: &ArrayBase<S2, Ix1>,
    training: &ArrayBase<S3, Ix1>,
    period: usize,
) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "seasonal MASE")?;

    if period == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: "Seasonal period must be positive".to_string(),
        });
    }

    check_minimum_length(training.len(), period + 1, "seasonal MASE (training)")?;

    let naive_scale = naive_mae(training, period)?;

    if naive_scale <= F::epsilon() {
        return Err(TimeSeriesError::ComputationError(
            "Seasonal MASE undefined: training series has zero seasonal naive MAE".to_string(),
        ));
    }

    let forecast_mae = mae(actual, forecast)?;
    Ok(forecast_mae / naive_scale)
}

/// Compute Root Mean Squared Error (RMSE)
///
/// RMSE = sqrt((1/n) * sum((actual - forecast)^2))
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast` - Predicted values
///
/// # Returns
/// RMSE (non-negative, same units as data)
pub fn rmse<S1, S2, F>(actual: &ArrayBase<S1, Ix1>, forecast: &ArrayBase<S2, Ix1>) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "RMSE")?;

    let n = F::from_usize(actual.len())
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;

    let sum_sq: F = actual
        .iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| {
            let e = a - f;
            e * e
        })
        .fold(F::zero(), |acc, x| acc + x);

    Ok((sum_sq / n).sqrt())
}

/// Compute Mean Squared Error (MSE)
///
/// MSE = (1/n) * sum((actual - forecast)^2)
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast` - Predicted values
///
/// # Returns
/// MSE (non-negative)
pub fn mse<S1, S2, F>(actual: &ArrayBase<S1, Ix1>, forecast: &ArrayBase<S2, Ix1>) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "MSE")?;

    let n = F::from_usize(actual.len())
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;

    let sum_sq: F = actual
        .iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| {
            let e = a - f;
            e * e
        })
        .fold(F::zero(), |acc, x| acc + x);

    Ok(sum_sq / n)
}

/// Compute Weighted Absolute Percentage Error (WAPE)
///
/// WAPE = sum(|actual - forecast|) / sum(|actual|)
///
/// Also known as weighted MAPE. Unlike MAPE, WAPE is not distorted by
/// small actual values because it uses the sum of absolute actuals as denominator.
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast` - Predicted values
///
/// # Returns
/// WAPE as a ratio (multiply by 100 for percentage)
pub fn wape<S1, S2, F>(actual: &ArrayBase<S1, Ix1>, forecast: &ArrayBase<S2, Ix1>) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    check_lengths(actual.len(), forecast.len())?;
    check_minimum_length(actual.len(), 1, "WAPE")?;

    let sum_abs_actual: F = actual
        .iter()
        .map(|&a| a.abs())
        .fold(F::zero(), |acc, x| acc + x);

    if sum_abs_actual <= F::epsilon() {
        return Err(TimeSeriesError::ComputationError(
            "WAPE undefined: sum of absolute actual values is zero".to_string(),
        ));
    }

    let sum_abs_error: F = actual
        .iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| (a - f).abs())
        .fold(F::zero(), |acc, x| acc + x);

    Ok(sum_abs_error / sum_abs_actual)
}

// ---------------------------------------------------------------------------
// Interval forecast metrics
// ---------------------------------------------------------------------------

/// Compute coverage probability for prediction intervals
///
/// Coverage = (1/n) * sum(lower <= actual <= upper)
///
/// Measures the fraction of actual observations that fall within the
/// prediction interval. A well-calibrated 95% interval should have
/// coverage close to 0.95.
///
/// # Arguments
/// * `actual` - Observed values
/// * `lower` - Lower bounds of prediction intervals
/// * `upper` - Upper bounds of prediction intervals
///
/// # Returns
/// Coverage probability in [0, 1]
pub fn coverage_probability<S1, S2, S3, F>(
    actual: &ArrayBase<S1, Ix1>,
    lower: &ArrayBase<S2, Ix1>,
    upper: &ArrayBase<S3, Ix1>,
) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    let n = actual.len();
    check_minimum_length(n, 1, "coverage probability")?;

    if lower.len() != n || upper.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: if lower.len() != n {
                lower.len()
            } else {
                upper.len()
            },
        });
    }

    let count: usize = actual
        .iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .filter(|((&a, &lo), &hi)| a >= lo && a <= hi)
        .count();

    let n_f = F::from_usize(n)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;
    let count_f = F::from_usize(count)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert count".to_string()))?;

    Ok(count_f / n_f)
}

/// Compute Winkler score for interval forecasts
///
/// The Winkler score penalizes both the width of the interval and
/// observations falling outside. For a (1-alpha)*100% interval:
///
/// S_t = (upper - lower) + (2/alpha)*(lower - actual)  if actual < lower
/// S_t = (upper - lower)                                if lower <= actual <= upper
/// S_t = (upper - lower) + (2/alpha)*(actual - upper)  if actual > upper
///
/// Average Winkler Score = (1/n) * sum(S_t)
///
/// Lower is better.
///
/// # Arguments
/// * `actual` - Observed values
/// * `lower` - Lower bounds of prediction intervals
/// * `upper` - Upper bounds of prediction intervals
/// * `alpha` - Significance level (e.g. 0.05 for 95% interval)
///
/// # Returns
/// Average Winkler score (lower is better)
pub fn winkler_score<S1, S2, S3, F>(
    actual: &ArrayBase<S1, Ix1>,
    lower: &ArrayBase<S2, Ix1>,
    upper: &ArrayBase<S3, Ix1>,
    alpha: F,
) -> Result<F>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display,
{
    let n = actual.len();
    check_minimum_length(n, 1, "Winkler score")?;

    if lower.len() != n || upper.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: if lower.len() != n {
                lower.len()
            } else {
                upper.len()
            },
        });
    }

    if alpha <= F::zero() || alpha >= F::one() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "alpha".to_string(),
            message: format!("Must be in (0, 1), got {alpha}"),
        });
    }

    let two = F::from_f64(2.0).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert constant".to_string())
    })?;
    let penalty_factor = two / alpha;

    let mut total_score = F::zero();

    for ((&a, &lo), &hi) in actual.iter().zip(lower.iter()).zip(upper.iter()) {
        let width = hi - lo;
        let penalty = if a < lo {
            penalty_factor * (lo - a)
        } else if a > hi {
            penalty_factor * (a - hi)
        } else {
            F::zero()
        };
        total_score = total_score + width + penalty;
    }

    let n_f = F::from_usize(n)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;

    Ok(total_score / n_f)
}

// ---------------------------------------------------------------------------
// Diebold-Mariano test
// ---------------------------------------------------------------------------

/// Result of the Diebold-Mariano test
#[derive(Debug, Clone)]
pub struct DieboldMarianoResult<F: Float> {
    /// DM test statistic (follows standard normal under H0)
    pub statistic: F,
    /// Two-sided p-value
    pub p_value: F,
    /// Mean loss differential (d_bar)
    pub mean_loss_diff: F,
    /// Long-run variance estimate of loss differential
    pub long_run_variance: F,
    /// Horizon used for HAC variance estimation
    pub horizon: usize,
}

/// Loss function for the Diebold-Mariano test
#[derive(Debug, Clone, Copy)]
pub enum DMLossFunction {
    /// Squared error loss: (actual - forecast)^2
    SquaredError,
    /// Absolute error loss: |actual - forecast|
    AbsoluteError,
}

/// Perform the Diebold-Mariano test for equal predictive accuracy
///
/// Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) is the loss
/// differential between two forecasts. Uses a HAC (Newey-West) variance
/// estimator to account for serial correlation in multi-step-ahead forecasts.
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast1` - Forecast from model 1
/// * `forecast2` - Forecast from model 2
/// * `loss` - Loss function to use
/// * `horizon` - Forecast horizon h (for HAC bandwidth; 1 for one-step-ahead)
///
/// # Returns
/// DieboldMarianoResult with test statistic, p-value, etc.
///
/// # References
/// Diebold, F.X. & Mariano, R.S. (1995)
pub fn diebold_mariano<S1, S2, S3, F>(
    actual: &ArrayBase<S1, Ix1>,
    forecast1: &ArrayBase<S2, Ix1>,
    forecast2: &ArrayBase<S3, Ix1>,
    loss: DMLossFunction,
    horizon: usize,
) -> Result<DieboldMarianoResult<F>>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display,
{
    let n = actual.len();
    check_minimum_length(n, 2, "Diebold-Mariano test")?;

    if forecast1.len() != n || forecast2.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: if forecast1.len() != n {
                forecast1.len()
            } else {
                forecast2.len()
            },
        });
    }

    if horizon == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "horizon".to_string(),
            message: "Forecast horizon must be at least 1".to_string(),
        });
    }

    // Compute loss differentials
    let loss_diff: Vec<F> = actual
        .iter()
        .zip(forecast1.iter())
        .zip(forecast2.iter())
        .map(|((&a, &f1), &f2)| {
            let l1 = compute_loss(a, f1, loss);
            let l2 = compute_loss(a, f2, loss);
            l1 - l2
        })
        .collect();

    let n_f = F::from_usize(n)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;

    // Mean loss differential
    let d_bar = loss_diff.iter().copied().fold(F::zero(), |a, x| a + x) / n_f;

    // HAC (Newey-West) variance estimator
    // Bandwidth = h - 1 for h-step-ahead forecasts
    let bandwidth = if horizon > 1 { horizon - 1 } else { 0 };

    let long_run_var = newey_west_variance(&loss_diff, d_bar, bandwidth)?;

    if long_run_var <= F::epsilon() {
        return Err(TimeSeriesError::ComputationError(
            "Diebold-Mariano test: zero variance in loss differentials (forecasts may be identical)"
                .to_string(),
        ));
    }

    // DM statistic
    let dm_stat = d_bar / (long_run_var / n_f).sqrt();

    // Two-sided p-value using normal approximation
    let p_value = two_sided_normal_p(dm_stat);

    Ok(DieboldMarianoResult {
        statistic: dm_stat,
        p_value,
        mean_loss_diff: d_bar,
        long_run_variance: long_run_var,
        horizon,
    })
}

/// Compute a comprehensive evaluation report
///
/// Calculates all point forecast metrics at once.
///
/// # Arguments
/// * `actual` - Observed values
/// * `forecast` - Predicted values
/// * `training` - Optional training data (needed for MASE)
///
/// # Returns
/// EvaluationReport with all metrics
pub fn evaluation_report<S1, S2, F>(
    actual: &ArrayBase<S1, Ix1>,
    forecast: &ArrayBase<S2, Ix1>,
    training: Option<&Array1<F>>,
) -> Result<EvaluationReport<F>>
where
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display,
{
    let mae_val = mae(actual, forecast)?;
    let rmse_val = rmse(actual, forecast)?;
    let mse_val = mse(actual, forecast)?;
    let mape_val = mape(actual, forecast).ok();
    let smape_val = smape(actual, forecast)?;
    let wape_val = wape(actual, forecast).ok();

    let mase_val = if let Some(train) = training {
        mase(actual, forecast, train).ok()
    } else {
        None
    };

    Ok(EvaluationReport {
        mae: mae_val,
        rmse: rmse_val,
        mse: mse_val,
        mape: mape_val,
        smape: smape_val,
        wape: wape_val,
        mase: mase_val,
    })
}

/// Comprehensive evaluation report for point forecasts
#[derive(Debug, Clone)]
pub struct EvaluationReport<F: Float> {
    /// Mean Absolute Error
    pub mae: F,
    /// Root Mean Squared Error
    pub rmse: F,
    /// Mean Squared Error
    pub mse: F,
    /// Mean Absolute Percentage Error (None if undefined)
    pub mape: Option<F>,
    /// Symmetric MAPE
    pub smape: F,
    /// Weighted Absolute Percentage Error (None if undefined)
    pub wape: Option<F>,
    /// Mean Absolute Scaled Error (None if no training data provided)
    pub mase: Option<F>,
}

impl<F: Float + Display> std::fmt::Display for EvaluationReport<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Forecast Evaluation Report")?;
        writeln!(f, "=========================")?;
        writeln!(f, "MAE:   {:.6}", self.mae)?;
        writeln!(f, "RMSE:  {:.6}", self.rmse)?;
        writeln!(f, "MSE:   {:.6}", self.mse)?;
        if let Some(m) = self.mape {
            writeln!(f, "MAPE:  {:.4}%", m)?;
        }
        writeln!(f, "sMAPE: {:.4}", self.smape)?;
        if let Some(w) = self.wape {
            writeln!(f, "WAPE:  {:.6}", w)?;
        }
        if let Some(m) = self.mase {
            writeln!(f, "MASE:  {:.6}", m)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Naive MAE for MASE scaling: average |y_t - y_{t-lag}|
fn naive_mae<S, F>(data: &ArrayBase<S, Ix1>, lag: usize) -> Result<F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    let n = data.len();
    if n <= lag {
        return Err(TimeSeriesError::InsufficientData {
            message: "for naive MAE computation".to_string(),
            required: lag + 1,
            actual: n,
        });
    }

    let diffs = n - lag;
    let n_diffs = F::from_usize(diffs)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert count".to_string()))?;

    let sum: F = (lag..n)
        .map(|i| (data[i] - data[i - lag]).abs())
        .fold(F::zero(), |acc, x| acc + x);

    Ok(sum / n_diffs)
}

/// Compute a single loss value
fn compute_loss<F: Float>(actual: F, forecast: F, loss: DMLossFunction) -> F {
    match loss {
        DMLossFunction::SquaredError => {
            let e = actual - forecast;
            e * e
        }
        DMLossFunction::AbsoluteError => (actual - forecast).abs(),
    }
}

/// Newey-West HAC variance estimator
fn newey_west_variance<F: Float + FromPrimitive>(
    data: &[F],
    mean: F,
    bandwidth: usize,
) -> Result<F> {
    let n = data.len();
    let n_f = F::from_usize(n)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert length".to_string()))?;

    // Gamma(0) - variance at lag 0
    let gamma0: F = data
        .iter()
        .map(|&d| {
            let dev = d - mean;
            dev * dev
        })
        .fold(F::zero(), |acc, x| acc + x)
        / n_f;

    let mut lrv = gamma0;

    // Add weighted autocovariances for lags 1..=bandwidth
    for lag in 1..=bandwidth {
        if lag >= n {
            break;
        }

        let lag_f = F::from_usize(lag).ok_or_else(|| {
            TimeSeriesError::ComputationError("Failed to convert lag".to_string())
        })?;
        let bw_f = F::from_usize(bandwidth + 1).ok_or_else(|| {
            TimeSeriesError::ComputationError("Failed to convert bandwidth".to_string())
        })?;

        // Bartlett kernel weight
        let weight = F::one() - lag_f / bw_f;

        // Autocovariance at this lag
        let gamma_lag: F = (lag..n)
            .map(|i| (data[i] - mean) * (data[i - lag] - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / n_f;

        // Add both sides (symmetric)
        let two = F::from_f64(2.0).ok_or_else(|| {
            TimeSeriesError::ComputationError("Failed to convert constant".to_string())
        })?;
        lrv = lrv + two * weight * gamma_lag;
    }

    // Ensure non-negative
    if lrv < F::zero() {
        Ok(F::zero())
    } else {
        Ok(lrv)
    }
}

/// Two-sided p-value from standard normal using rational approximation
/// (Abramowitz & Stegun approximation for the normal CDF)
fn two_sided_normal_p<F: Float + FromPrimitive>(z: F) -> F {
    let abs_z = z.abs();
    let one = F::one();
    let two = F::from_f64(2.0).unwrap_or(one + one);

    // For very large |z|, return 0
    let threshold = F::from_f64(8.0).unwrap_or(F::from_f64(8.0).unwrap_or(one));
    if abs_z > threshold {
        return F::zero();
    }

    // Abramowitz & Stegun 26.2.17 approximation
    let p = F::from_f64(0.2316419).unwrap_or(F::zero());
    let b1 = F::from_f64(0.319381530).unwrap_or(F::zero());
    let b2 = F::from_f64(-0.356563782).unwrap_or(F::zero());
    let b3 = F::from_f64(1.781477937).unwrap_or(F::zero());
    let b4 = F::from_f64(-1.821255978).unwrap_or(F::zero());
    let b5 = F::from_f64(1.330274429).unwrap_or(F::zero());

    let inv_sqrt_2pi = F::from_f64(0.39894228040143268).unwrap_or(F::zero());

    let t = one / (one + p * abs_z);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5;
    let pdf = inv_sqrt_2pi * (-abs_z * abs_z / two).exp();

    let tail_prob = pdf * poly;

    // Two-sided
    two * tail_prob
}

/// Check that two arrays have the same length
fn check_lengths(len1: usize, len2: usize) -> Result<()> {
    if len1 != len2 {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: len1,
            actual: len2,
        });
    }
    Ok(())
}

/// Check minimum array length
fn check_minimum_length(len: usize, min: usize, operation: &str) -> Result<()> {
    if len < min {
        return Err(TimeSeriesError::InsufficientData {
            message: format!("for {operation}"),
            required: min,
            actual: len,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    const TOL: f64 = 1e-10;
    const LOOSE_TOL: f64 = 1e-4;

    #[test]
    fn test_mae_basic() {
        let actual = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let forecast = array![1.1, 2.2, 2.8, 4.3, 4.7];
        let result = mae(&actual, &forecast).expect("MAE should succeed");
        // |0.1| + |0.2| + |0.2| + |0.3| + |0.3| = 1.1 / 5 = 0.22
        assert!((result - 0.22).abs() < TOL);
    }

    #[test]
    fn test_mae_perfect() {
        let actual = array![1.0, 2.0, 3.0];
        let forecast = array![1.0, 2.0, 3.0];
        let result = mae(&actual, &forecast).expect("MAE should succeed");
        assert!(result.abs() < TOL);
    }

    #[test]
    fn test_mae_length_mismatch() {
        let actual = array![1.0, 2.0];
        let forecast = array![1.0, 2.0, 3.0];
        assert!(mae(&actual, &forecast).is_err());
    }

    #[test]
    fn test_mae_empty() {
        let actual: Array1<f64> = array![];
        let forecast: Array1<f64> = array![];
        assert!(mae(&actual, &forecast).is_err());
    }

    #[test]
    fn test_mape_basic() {
        let actual = array![100.0, 200.0, 300.0, 400.0];
        let forecast = array![110.0, 190.0, 280.0, 420.0];
        let result = mape(&actual, &forecast).expect("MAPE should succeed");
        // (10/100 + 10/200 + 20/300 + 20/400) * 100 / 4
        let expected = (0.10 + 0.05 + 20.0 / 300.0 + 0.05) * 100.0 / 4.0;
        assert!((result - expected).abs() < TOL);
    }

    #[test]
    fn test_mape_zero_actual() {
        let actual = array![0.0, 0.0, 0.0];
        let forecast = array![1.0, 2.0, 3.0];
        assert!(mape(&actual, &forecast).is_err());
    }

    #[test]
    fn test_smape_basic() {
        let actual = array![100.0, 200.0, 300.0];
        let forecast = array![110.0, 190.0, 310.0];
        let result = smape(&actual, &forecast).expect("sMAPE should succeed");
        // 200/3 * (10/210 + 10/390 + 10/610)
        let expected = 200.0 / 3.0 * (10.0 / 210.0 + 10.0 / 390.0 + 10.0 / 610.0);
        assert!((result - expected).abs() < LOOSE_TOL);
    }

    #[test]
    fn test_smape_both_zero() {
        // Both zero => perfect, contributes 0
        let actual = array![0.0, 0.0];
        let forecast = array![0.0, 0.0];
        let result = smape(&actual, &forecast).expect("sMAPE should succeed");
        assert!(result.abs() < TOL);
    }

    #[test]
    fn test_mase_basic() {
        let training = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let actual = array![11.0, 12.0, 13.0];
        let forecast = array![10.5, 12.5, 12.0];
        let result = mase(&actual, &forecast, &training).expect("MASE should succeed");
        // Naive MAE of training: |2-1|+|3-2|+...+|10-9| = 9 differences of 1.0 each => 1.0
        // Forecast MAE: (0.5 + 0.5 + 1.0) / 3 = 2/3
        // MASE = (2/3) / 1.0 = 2/3
        assert!((result - 2.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn test_mase_constant_training() {
        let training = array![5.0, 5.0, 5.0, 5.0];
        let actual = array![6.0];
        let forecast = array![5.5];
        // Naive MAE = 0 => MASE undefined
        assert!(mase(&actual, &forecast, &training).is_err());
    }

    #[test]
    fn test_mase_seasonal() {
        // Period-4 seasonal data
        let training = array![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0];
        let actual = array![4.0, 5.0, 6.0, 7.0];
        let forecast = array![3.5, 5.5, 5.5, 7.5];
        let result =
            mase_seasonal(&actual, &forecast, &training, 4).expect("Seasonal MASE should succeed");
        // Seasonal naive MAE: |2-1|+|3-2|+|4-3|+|5-4|+|3-2|+|4-3|+|5-4|+|6-5| = 8*1 / 8 = 1.0
        // Forecast MAE: (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        // MASE = 0.5 / 1.0 = 0.5
        assert!((result - 0.5).abs() < TOL);
    }

    #[test]
    fn test_rmse_basic() {
        let actual = array![1.0, 2.0, 3.0, 4.0];
        let forecast = array![1.1, 1.8, 3.2, 3.7];
        let result = rmse(&actual, &forecast).expect("RMSE should succeed");
        // Errors: 0.1, -0.2, 0.2, -0.3 => sq: 0.01, 0.04, 0.04, 0.09 => mean = 0.045 => sqrt = 0.2121...
        let expected = (0.045_f64).sqrt();
        assert!((result - expected).abs() < TOL);
    }

    #[test]
    fn test_mse_basic() {
        let actual = array![3.0, 5.0, 2.5];
        let forecast = array![2.5, 5.0, 3.0];
        let result = mse(&actual, &forecast).expect("MSE should succeed");
        // Squared errors: 0.25, 0, 0.25 => mean = 0.5/3
        assert!((result - 0.5 / 3.0).abs() < TOL);
    }

    #[test]
    fn test_wape_basic() {
        let actual = array![100.0, 200.0, 300.0];
        let forecast = array![110.0, 190.0, 280.0];
        let result = wape(&actual, &forecast).expect("WAPE should succeed");
        // sum_abs_error = 10 + 10 + 20 = 40
        // sum_abs_actual = 100 + 200 + 300 = 600
        // WAPE = 40/600
        assert!((result - 40.0 / 600.0).abs() < TOL);
    }

    #[test]
    fn test_coverage_basic() {
        let actual = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let lower = array![0.5, 1.5, 2.0, 3.5, 6.0];
        let upper = array![1.5, 2.5, 4.0, 4.5, 7.0];
        // Covered: 1.0 in [0.5,1.5] yes, 2.0 in [1.5,2.5] yes, 3.0 in [2.0,4.0] yes,
        // 4.0 in [3.5,4.5] yes, 5.0 in [6.0,7.0] no => 4/5 = 0.8
        let result =
            coverage_probability(&actual, &lower, &upper).expect("Coverage should succeed");
        assert!((result - 0.8).abs() < TOL);
    }

    #[test]
    fn test_coverage_perfect() {
        let actual = array![1.0, 2.0, 3.0];
        let lower = array![0.0, 0.0, 0.0];
        let upper = array![10.0, 10.0, 10.0];
        let result =
            coverage_probability(&actual, &lower, &upper).expect("Coverage should succeed");
        assert!((result - 1.0).abs() < TOL);
    }

    #[test]
    fn test_winkler_score_all_covered() {
        let actual = array![2.0, 3.0, 4.0];
        let lower = array![1.0, 2.0, 3.0];
        let upper = array![3.0, 4.0, 5.0];
        let result =
            winkler_score(&actual, &lower, &upper, 0.05).expect("Winkler score should succeed");
        // All covered => score = average width = (2+2+2)/3 = 2.0
        assert!((result - 2.0).abs() < TOL);
    }

    #[test]
    fn test_winkler_score_with_penalty() {
        let actual = array![0.0, 5.0];
        let lower = array![1.0, 2.0];
        let upper = array![3.0, 4.0];
        let alpha = 0.1_f64;
        let result =
            winkler_score(&actual, &lower, &upper, alpha).expect("Winkler score should succeed");
        // Obs 0: actual=0 < lower=1 => penalty = (2/0.1)*(1-0) = 20, width=2, total=22
        // Obs 1: actual=5 > upper=4 => penalty = (2/0.1)*(5-4) = 20, width=2, total=22
        // Average = (22+22)/2 = 22
        assert!((result - 22.0).abs() < TOL);
    }

    #[test]
    fn test_winkler_invalid_alpha() {
        let actual = array![1.0];
        let lower = array![0.0];
        let upper = array![2.0];
        assert!(winkler_score(&actual, &lower, &upper, 0.0_f64).is_err());
        assert!(winkler_score(&actual, &lower, &upper, 1.0_f64).is_err());
        assert!(winkler_score(&actual, &lower, &upper, -0.1_f64).is_err());
    }

    #[test]
    fn test_diebold_mariano_identical() {
        let actual = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let f1 = array![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];
        let f2 = f1.clone();
        // Identical forecasts => zero variance => error
        assert!(diebold_mariano(&actual, &f1, &f2, DMLossFunction::SquaredError, 1).is_err());
    }

    #[test]
    fn test_diebold_mariano_different() {
        // Use alternating errors so that squared loss differentials vary (non-zero variance)
        // f1 alternates over/under-shoot; f2 alternates under/over-shoot
        // => loss_diff alternates +0.21 / -0.21 => mean ~ 0, non-zero variance
        let actual = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let f1 = array![1.5, 1.8, 3.5, 3.8, 5.5, 5.8, 7.5, 7.8, 9.5, 9.8];
        let f2 = array![1.2, 2.5, 2.8, 4.5, 4.8, 6.5, 6.8, 8.5, 8.8, 10.5];
        let result = diebold_mariano(&actual, &f1, &f2, DMLossFunction::SquaredError, 1)
            .expect("DM test should succeed");
        // Alternating loss diffs => mean loss diff is ~ 0
        assert!(result.mean_loss_diff.abs() < 0.1);
        // p-value should be large (fail to reject null hypothesis of equal accuracy)
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_diebold_mariano_clearly_better() {
        let actual = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0
        ];
        // f1 is much worse with varying errors (cycling through 0.5, 1.0, 1.5, 2.0, 2.5)
        let f1 = array![
            1.5, 3.0, 4.5, 6.0, 7.5, 7.0, 8.5, 10.0, 11.5, 13.0, 12.5, 14.0, 15.5, 17.0, 18.5,
            17.0, 18.5, 20.0, 21.5, 23.0
        ];
        // f2 is close with small varying errors (cycling through 0.1, 0.2, 0.3)
        let f2 = array![
            1.1, 2.2, 3.3, 4.1, 5.2, 6.3, 7.1, 8.2, 9.3, 10.1, 11.2, 12.3, 13.1, 14.2, 15.3, 16.1,
            17.2, 18.3, 19.1, 20.2
        ];
        let result = diebold_mariano(&actual, &f1, &f2, DMLossFunction::SquaredError, 1)
            .expect("DM test should succeed");
        // f1 has much larger varying errors => positive mean loss diff
        assert!(result.mean_loss_diff > 0.0_f64);
        // Statistic should be positive (f1 is worse)
        assert!(result.statistic > 0.0_f64);
    }

    type F64 = f64;

    #[test]
    fn test_diebold_mariano_absolute_loss() {
        // f1 has alternating errors 1.0 and 1.5; f2 has alternating errors 0.1 and 0.2
        // => loss diffs alternate between 0.9 and 1.3, mean = 1.1, non-zero variance
        let actual = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let f1 = array![2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0, 9.5, 10.0, 11.5];
        let f2 = array![1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9];
        let result = diebold_mariano(&actual, &f1, &f2, DMLossFunction::AbsoluteError, 1)
            .expect("DM test should succeed");
        // f1 errors are larger than f2 errors => mean loss diff > 0
        assert!(result.mean_loss_diff > 0.0);
        // Mean loss diff should be close to 1.11 (average of alternating 0.9, 1.4, 0.8, 1.3)
        assert!((result.mean_loss_diff - 1.11).abs() < 0.01);
    }

    #[test]
    fn test_evaluation_report() {
        let actual = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let forecast = array![11.0, 19.0, 31.0, 38.0, 52.0];
        let training = array![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let report =
            evaluation_report(&actual, &forecast, Some(&training)).expect("Report should succeed");
        assert!(report.mae > 0.0);
        assert!(report.rmse > 0.0);
        assert!(report.mse > 0.0);
        assert!(report.smape > 0.0);
        assert!(report.mape.is_some());
        assert!(report.wape.is_some());
        assert!(report.mase.is_some());
    }

    #[test]
    fn test_evaluation_report_no_training() {
        let actual = array![1.0, 2.0, 3.0];
        let forecast = array![1.0, 2.0, 3.0];
        let report = evaluation_report::<_, _, f64>(&actual, &forecast, None)
            .expect("Report should succeed");
        assert!(report.mase.is_none());
    }

    #[test]
    fn test_rmse_ge_mae() {
        // RMSE >= MAE always holds (Cauchy-Schwarz)
        let actual = array![1.0, 5.0, 3.0, 8.0, 2.0];
        let forecast = array![2.0, 4.0, 5.0, 6.0, 3.0];
        let mae_val: f64 = mae(&actual, &forecast).expect("MAE should work");
        let rmse_val: f64 = rmse(&actual, &forecast).expect("RMSE should work");
        assert!(rmse_val >= mae_val);
    }

    #[test]
    fn test_smape_symmetric() {
        let a = array![100.0, 200.0];
        let b = array![200.0, 100.0];
        let smape_ab: f64 = smape(&a, &b).expect("smape ab");
        let smape_ba: f64 = smape(&b, &a).expect("smape ba");
        // sMAPE with swapped roles should give same result
        assert!((smape_ab - smape_ba).abs() < TOL);
    }

    #[test]
    fn test_coverage_boundary_inclusion() {
        // Test that boundaries are inclusive
        let actual = array![1.0, 3.0];
        let lower = array![1.0, 2.0];
        let upper = array![2.0, 3.0];
        let cov: f64 = coverage_probability(&actual, &lower, &upper).expect("Coverage should work");
        assert!((cov - 1.0).abs() < TOL);
    }

    #[test]
    fn test_winkler_alpha_bounds() {
        let actual = array![1.0];
        let lower = array![0.0];
        let upper = array![2.0];
        // Valid alpha
        assert!(winkler_score(&actual, &lower, &upper, 0.05_f64).is_ok());
        assert!(winkler_score(&actual, &lower, &upper, 0.5_f64).is_ok());
    }

    #[test]
    fn test_f32_support() {
        let actual: Array1<f32> = array![1.0f32, 2.0, 3.0];
        let forecast: Array1<f32> = array![1.1f32, 2.2, 2.8];
        let result = mae(&actual, &forecast);
        assert!(result.is_ok());
    }
}
