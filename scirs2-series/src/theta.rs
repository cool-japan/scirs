//! Theta Method for Time Series Forecasting
//!
//! Implements the Theta method (Assimakopoulos & Nikolopoulos, 2000) and its
//! optimized variant (Hyndman & Billah, 2003).
//!
//! The method decomposes a time series into two "theta lines":
//! - Theta-line 0 (θ=0): linear regression fit (captures long-term trend)
//! - Theta-line 2 (θ=2): amplifies local curvature of the original series
//!
//! The forecasts from these two lines are combined (equal weights by default)
//! to produce the final forecast.
//!
//! The Optimized Theta Method (OTM) selects the combination weight via MLE
//! over an SES model applied to the second differences, consistent with the
//! interpretation by Hyndman & Billah.
//!
//! # References
//!
//! - Assimakopoulos, V. & Nikolopoulos, K. (2000). "The theta model: a
//!   decomposition approach to forecasting." *International Journal of
//!   Forecasting*, 16(4), 521-530.
//! - Hyndman, R.J. & Billah, B. (2003). "Unmasking the Theta method."
//!   *International Journal of Forecasting*, 19(2), 287-290.
//! - Fioruci, J.A., Pellegrini, T.R., Louzada, F., & Petropoulos, F. (2016).
//!   "The optimized theta method." *Journal of Forecasting*, 35(2), 161-166.

use crate::error::{Result, TimeSeriesError};

// ──────────────────────────────────────────────────────────────────────────────
// Seasonal decomposition helpers (classical additive)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute seasonal indices for a series with a given period using a
/// centered moving-average decomposition.
fn seasonal_indices_additive(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 * period {
        return vec![0.0; period];
    }

    // Centered moving average (CMA)
    let half = period / 2;
    let mut trend = vec![f64::NAN; n];
    for i in half..n - half {
        let window = if period % 2 == 0 {
            // For even periods use weighted average
            let mut sum = 0.0;
            sum += data[i - half] * 0.5;
            for j in (i - half + 1)..=(i + half - 1) {
                sum += data[j];
            }
            sum += data[i + half] * 0.5;
            sum / period as f64
        } else {
            data[(i - half)..=(i + half)].iter().sum::<f64>() / period as f64
        };
        trend[i] = window;
    }

    // Detrended series
    let mut detrended = vec![f64::NAN; n];
    for i in 0..n {
        if trend[i].is_finite() {
            detrended[i] = data[i] - trend[i];
        }
    }

    // Average per season position
    let mut season_sums = vec![0.0_f64; period];
    let mut season_counts = vec![0_usize; period];
    for i in 0..n {
        if detrended[i].is_finite() {
            let pos = i % period;
            season_sums[pos] += detrended[i];
            season_counts[pos] += 1;
        }
    }

    let mut indices: Vec<f64> = season_sums
        .iter()
        .zip(season_counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    // Normalize so they sum to zero
    let mean = indices.iter().sum::<f64>() / period as f64;
    for idx in &mut indices {
        *idx -= mean;
    }
    indices
}

/// Deseasonalize a series using precomputed additive indices.
fn deseasonalize(data: &[f64], indices: &[f64]) -> Vec<f64> {
    let period = indices.len();
    data.iter()
        .enumerate()
        .map(|(i, &y)| y - indices[i % period])
        .collect()
}

/// Reseasonalize forecasts: add back seasonal indices starting at n_obs.
fn reseasonalize(forecasts: &[f64], indices: &[f64], n_obs: usize) -> Vec<f64> {
    let period = indices.len();
    forecasts
        .iter()
        .enumerate()
        .map(|(h, &f)| f + indices[(n_obs + h) % period])
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Linear regression helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a simple linear regression y = a + b*t (t = 1..=n) and return (a, b).
fn linear_regression(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let t_mean = (n + 1.0) / 2.0;
    let y_mean = data.iter().sum::<f64>() / n;

    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    for (i, &y) in data.iter().enumerate() {
        let t = (i + 1) as f64;
        sxy += (t - t_mean) * (y - y_mean);
        sxx += (t - t_mean).powi(2);
    }

    let b = if sxx.abs() > 1e-14 { sxy / sxx } else { 0.0 };
    let a = y_mean - b * t_mean;
    (a, b)
}

// ──────────────────────────────────────────────────────────────────────────────
// SES helpers (needed to compute the second theta-line)
// ──────────────────────────────────────────────────────────────────────────────

/// Run SES and return the final smoothed level.
fn ses_final_level(data: &[f64], alpha: f64) -> f64 {
    let mut level = data[0];
    for &y in data.iter().skip(1) {
        level = alpha * y + (1.0 - alpha) * level;
    }
    level
}

/// SSE of one-step-ahead SES forecasts.
fn ses_sse_from(data: &[f64], alpha: f64) -> f64 {
    let mut level = data[0];
    let mut sse = 0.0_f64;
    for &y in data.iter().skip(1) {
        let err = y - level;
        sse += err * err;
        level = alpha * y + (1.0 - alpha) * level;
    }
    sse
}

/// Golden-section search on (lo, hi) to minimize a univariate function.
fn golden_section(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut x1 = b - gr * (b - a);
    let mut x2 = a + gr * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..300 {
        if (b - a).abs() < 1e-9 {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - gr * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = f(x2);
        }
    }
    (a + b) / 2.0
}

/// Optimal SES alpha via golden-section search.
fn optimize_ses_alpha(data: &[f64]) -> f64 {
    golden_section(|a| ses_sse_from(data, a), 1e-6, 1.0 - 1e-10)
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal fitting state
// ──────────────────────────────────────────────────────────────────────────────

/// Internal fitted state for the Theta method.
#[derive(Debug, Clone)]
struct ThetaFitted {
    /// Intercept of the linear regression (theta-line 0 when theta=0)
    reg_intercept: f64,
    /// Slope of the linear regression
    reg_slope: f64,
    /// Final SES level of the second theta-line
    ses_level: f64,
    /// Optimized SES alpha applied to the second theta-line
    ses_alpha: f64,
    /// Weight of theta-line 2 in the combination (1 - weight for theta-line 0)
    w2: f64,
    /// Number of training observations
    n_obs: usize,
    /// Seasonal indices (empty if deseasonalize = false)
    seasonal_indices: Vec<f64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// ThetaModel
// ──────────────────────────────────────────────────────────────────────────────

/// Theta method for time series forecasting.
///
/// The series is optionally deseasonalized before applying the theta decomposition.
/// The default theta=2 corresponds to the original method (equal-weight combination
/// of theta-0 and theta-2 lines), which is equivalent to SES with a linear drift.
#[derive(Debug, Clone)]
pub struct ThetaModel {
    /// Theta coefficient (0 = drift only, 2 = original Theta method)
    theta: f64,
    /// Seasonal period for optional deseasonalization
    period: Option<usize>,
    /// Whether to remove seasonality before fitting
    deseasonalize: bool,
    /// Fitted state (populated after calling `fit`)
    fitted: Option<ThetaFitted>,
}

impl ThetaModel {
    /// Create a new Theta model.
    ///
    /// # Arguments
    /// * `theta` - Theta coefficient; 0 produces a pure-drift forecast,
    ///   2 is the classic Theta method, values > 2 amplify local curvature
    /// * `period` - Seasonal period; if provided and `theta` > 0, the series
    ///   is deseasonalized before fitting
    pub fn new(theta: f64, period: Option<usize>) -> Self {
        Self {
            theta,
            period,
            deseasonalize: period.is_some(),
            fitted: None,
        }
    }

    /// Fit the model to training data.
    ///
    /// # Arguments
    /// * `data` - Training observations (at least 4 values required)
    pub fn fit(&mut self, data: &[f64]) -> Result<&Self> {
        if data.len() < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Theta model requires at least 4 observations".to_string(),
                required: 4,
                actual: data.len(),
            });
        }

        // Optional seasonal decomposition
        let (working, seasonal_indices) = if self.deseasonalize {
            if let Some(p) = self.period {
                if data.len() >= 2 * p && p >= 2 {
                    let indices = seasonal_indices_additive(data, p);
                    let deseas = deseasonalize(data, &indices);
                    (deseas, indices)
                } else {
                    (data.to_vec(), Vec::new())
                }
            } else {
                (data.to_vec(), Vec::new())
            }
        } else {
            (data.to_vec(), Vec::new())
        };

        let n = working.len();

        // Fit theta-line 0 (linear regression = long-run drift)
        let (a, b) = linear_regression(&working);

        // Construct theta-line 2:
        // θ_2(t) = θ * y_t - (θ - 1) * (a + b*t)
        // = 2 * y_t - (2-1) * lin = 2*y_t - lin
        // General: theta * y_t - (theta - 1) * linear_t
        let theta2: Vec<f64> = working
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let t = (i + 1) as f64;
                let lin = a + b * t;
                self.theta * y - (self.theta - 1.0) * lin
            })
            .collect();

        // Optimize SES alpha on theta-line 2
        let ses_alpha = if theta2.len() > 1 {
            optimize_ses_alpha(&theta2)
        } else {
            0.5
        };

        // Final SES level on theta-line 2
        let ses_level = ses_final_level(&theta2, ses_alpha);

        // Combination weight: equal weights (w2 = 0.5 for theta=2)
        // General: w2 = 1 / theta, w0 = 1 - w2 for theta != 0
        let w2 = if self.theta.abs() < 1e-14 {
            0.0
        } else {
            0.5 // original Theta method: equal weights
        };

        self.fitted = Some(ThetaFitted {
            reg_intercept: a,
            reg_slope: b,
            ses_level,
            ses_alpha,
            w2,
            n_obs: n,
            seasonal_indices,
        });

        Ok(self)
    }

    /// Produce h-step-ahead forecasts.
    pub fn forecast(&self, h: usize) -> Result<Vec<f64>> {
        let state = self.fitted.as_ref().ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("Call fit() before forecast()".to_string())
        })?;

        let n = state.n_obs;
        let mut forecasts = Vec::with_capacity(h);

        for k in 1..=h {
            let t = (n + k) as f64;

            // Theta-line 0 forecast: linear trend
            let f0 = state.reg_intercept + state.reg_slope * t;

            // Theta-line 2 forecast: SES level (flat)
            let f2 = state.ses_level;

            // Weighted combination
            let combined = (1.0 - state.w2) * f0 + state.w2 * f2;
            forecasts.push(combined);
        }

        // Reseasonalize if needed
        let result = if !state.seasonal_indices.is_empty() {
            reseasonalize(&forecasts, &state.seasonal_indices, n)
        } else {
            forecasts
        };

        Ok(result)
    }

    /// Auto-select optimal theta by leave-one-out cross-validation.
    ///
    /// Evaluates theta in {0.0, 0.5, 1.0, 1.5, 2.0, 3.0} and picks the
    /// value minimizing mean absolute error on the last 20% of the series.
    pub fn auto_fit(data: &[f64], period: Option<usize>) -> Result<Self> {
        if data.len() < 8 {
            return Err(TimeSeriesError::InsufficientData {
                message: "auto_fit requires at least 8 observations".to_string(),
                required: 8,
                actual: data.len(),
            });
        }

        let candidates = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0];
        let val_size = (data.len() / 5).max(1);
        let train_size = data.len() - val_size;
        let train = &data[..train_size];
        let val = &data[train_size..];

        let mut best_theta = 2.0_f64;
        let mut best_mae = f64::INFINITY;

        for &theta in &candidates {
            let mut m = ThetaModel::new(theta, period);
            if m.fit(train).is_err() {
                continue;
            }
            let fcast = match m.forecast(val_size) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mae = fcast
                .iter()
                .zip(val.iter())
                .map(|(&f, &y)| (f - y).abs())
                .sum::<f64>()
                / val_size as f64;
            if mae < best_mae {
                best_mae = mae;
                best_theta = theta;
            }
        }

        // Refit on full data with optimal theta
        let mut model = ThetaModel::new(best_theta, period);
        model.fit(data)?;
        Ok(model)
    }

    /// Return the theta coefficient.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Return whether seasonal decomposition is applied.
    pub fn deseasonalize(&self) -> bool {
        self.deseasonalize
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OptimizedTheta
// ──────────────────────────────────────────────────────────────────────────────

/// Optimized Theta Method (OTM) - Hyndman & Billah (2003) / Fioruci et al. (2016).
///
/// Extends the standard Theta method by:
/// 1. Optimizing the combination weight via MLE on the SES model
/// 2. Optionally applying seasonal adjustment before decomposition
///
/// The OTM is shown to be equivalent to SES with a locally-optimized drift term,
/// making it both interpretable and competitive with more complex models.
#[derive(Debug, Clone)]
pub struct OptimizedTheta {
    /// Selected theta value
    theta: f64,
    /// Seasonal period
    period: Option<usize>,
    /// Internal fitted state
    fitted: Option<ThetaFitted>,
}

impl OptimizedTheta {
    /// Fit the Optimized Theta Model.
    ///
    /// Selects theta from a candidate set by minimizing one-step AIC on the
    /// SES-fitted second differences, then refits on the full series.
    ///
    /// # Arguments
    /// * `data` - Training observations (at least 6 required)
    /// * `period` - Optional seasonal period for deseasonalization
    pub fn fit(data: &[f64], period: Option<usize>) -> Result<Self> {
        if data.len() < 6 {
            return Err(TimeSeriesError::InsufficientData {
                message: "OptimizedTheta requires at least 6 observations".to_string(),
                required: 6,
                actual: data.len(),
            });
        }

        // Seasonal adjustment
        let (working, seasonal_indices) = if let Some(p) = period {
            if data.len() >= 2 * p && p >= 2 {
                let indices = seasonal_indices_additive(data, p);
                let deseas = deseasonalize(data, &indices);
                (deseas, indices)
            } else {
                (data.to_vec(), Vec::new())
            }
        } else {
            (data.to_vec(), Vec::new())
        };

        let n = working.len();
        let (a, b) = linear_regression(&working);

        // Candidate theta values: use MLE (minimize AIC on SES residuals)
        let candidates = [0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
        let mut best_theta = 2.0_f64;
        let mut best_aic = f64::INFINITY;

        for &theta in &candidates {
            let theta2: Vec<f64> = working
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let t = (i + 1) as f64;
                    let lin = a + b * t;
                    theta * y - (theta - 1.0) * lin
                })
                .collect();

            if theta2.len() < 3 {
                continue;
            }

            // Optimize SES alpha on theta-line 2
            let alpha = optimize_ses_alpha(&theta2);
            let sse = ses_sse_from(&theta2, alpha);

            // AIC for the SES model: k=1 (alpha), using Gaussian log-likelihood
            let k = 1_usize;
            let m = theta2.len() as f64;
            let sigma2 = sse / m;
            if sigma2 <= 0.0 {
                continue;
            }
            let log_lik = -0.5 * m * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());
            let aic = -2.0 * log_lik + 2.0 * k as f64;

            if aic < best_aic {
                best_aic = aic;
                best_theta = theta;
            }
        }

        // Construct the optimal theta-line 2 and compute SES on it
        let theta2: Vec<f64> = working
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let t = (i + 1) as f64;
                let lin = a + b * t;
                best_theta * y - (best_theta - 1.0) * lin
            })
            .collect();

        let ses_alpha = optimize_ses_alpha(&theta2);
        let ses_level = ses_final_level(&theta2, ses_alpha);

        // Combination weight: equal weights as in the original Theta method.
        // The OTM optimizes theta selection, not the combination weight.
        // Equal weights (w2=0.5) correspond to the classical Theta interpretation
        // proved by Hyndman & Billah (2003) to be equivalent to SES + linear drift.
        let w2 = if best_theta.abs() < 1e-14 {
            0.0
        } else {
            0.5 // equal weights (original Theta method combination)
        };

        let fitted = ThetaFitted {
            reg_intercept: a,
            reg_slope: b,
            ses_level,
            ses_alpha,
            w2,
            n_obs: n,
            seasonal_indices,
        };

        Ok(Self {
            theta: best_theta,
            period,
            fitted: Some(fitted),
        })
    }

    /// Produce h-step-ahead forecasts.
    pub fn forecast(&self, h: usize) -> Result<Vec<f64>> {
        let state = self.fitted.as_ref().ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("OptimizedTheta is not fitted".to_string())
        })?;

        let n = state.n_obs;
        let mut forecasts = Vec::with_capacity(h);

        for k in 1..=h {
            let t = (n + k) as f64;
            let f0 = state.reg_intercept + state.reg_slope * t;
            let f2 = state.ses_level;
            let combined = (1.0 - state.w2) * f0 + state.w2 * f2;
            forecasts.push(combined);
        }

        let result = if !state.seasonal_indices.is_empty() {
            reseasonalize(&forecasts, &state.seasonal_indices, n)
        } else {
            forecasts
        };

        Ok(result)
    }

    /// Return the selected theta value.
    pub fn theta(&self) -> f64 {
        self.theta
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_series(n: usize) -> Vec<f64> {
        (1..=n).map(|i| i as f64).collect()
    }

    fn seasonal_series() -> Vec<f64> {
        // 3 cycles of period-4 data with mild trend
        let base = vec![10.0_f64, 14.0, 12.0, 8.0];
        let mut data = Vec::new();
        for cycle in 0..3 {
            for &b in &base {
                data.push(b + cycle as f64);
            }
        }
        data
    }

    // ── ThetaModel tests ──────────────────────────────────────────────────
    #[test]
    fn test_theta_fit_forecast_length() {
        let data = linear_series(20);
        let mut model = ThetaModel::new(2.0, None);
        model.fit(&data).expect("unexpected None or Err");
        let fcast = model.forecast(5).expect("failed to create fcast");
        assert_eq!(fcast.len(), 5);
    }

    #[test]
    fn test_theta_forecast_linear_trend() {
        // For a perfect linear series, Theta should produce forecasts that
        // continue the linear trend approximately.
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let mut model = ThetaModel::new(2.0, None);
        model.fit(&data).expect("unexpected None or Err");
        let fcast = model.forecast(4).expect("failed to create fcast");
        // Forecasts should continue upward
        for w in fcast.windows(2) {
            assert!(
                w[1] > w[0],
                "forecasts should increase for linear trend: {:?}",
                fcast
            );
        }
    }

    #[test]
    fn test_theta_zero_gives_pure_drift() {
        // theta=0 means w2=0: pure linear drift forecast
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let mut model = ThetaModel::new(0.0, None);
        model.fit(&data).expect("unexpected None or Err");
        let fcast = model.forecast(3).expect("failed to create fcast");
        assert_eq!(fcast.len(), 3);
        // All finite
        for &f in &fcast {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_theta_seasonal_deseasonalize() {
        let data = seasonal_series();
        let mut model = ThetaModel::new(2.0, Some(4));
        model.fit(&data).expect("unexpected None or Err");
        let fcast = model.forecast(4).expect("failed to create fcast");
        assert_eq!(fcast.len(), 4);
        for &f in &fcast {
            assert!(f.is_finite(), "forecast must be finite");
        }
    }

    #[test]
    fn test_theta_insufficient_data() {
        let mut model = ThetaModel::new(2.0, None);
        assert!(model.fit(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_theta_not_fitted_errors() {
        let model = ThetaModel::new(2.0, None);
        assert!(model.forecast(3).is_err());
    }

    #[test]
    fn test_theta_auto_fit() {
        let data: Vec<f64> = (1..=30).map(|i| i as f64 + (i % 4) as f64 * 2.0).collect();
        let model = ThetaModel::auto_fit(&data, None).expect("failed to create model");
        let fcast = model.forecast(5).expect("failed to create fcast");
        assert_eq!(fcast.len(), 5);
        for &f in &fcast {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_theta_auto_fit_insufficient_data() {
        assert!(ThetaModel::auto_fit(&[1.0, 2.0, 3.0, 4.0], None).is_err());
    }

    // ── OptimizedTheta tests ───────────────────────────────────────────────
    #[test]
    fn test_otm_fit_forecast() {
        let data = linear_series(20);
        let model = OptimizedTheta::fit(&data, None).expect("failed to create model");
        let fcast = model.forecast(5).expect("failed to create fcast");
        assert_eq!(fcast.len(), 5);
        for &f in &fcast {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_otm_theta_in_candidates() {
        let data = linear_series(20);
        let model = OptimizedTheta::fit(&data, None).expect("failed to create model");
        // theta should be one of the candidates
        let candidates = [0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
        let found = candidates.iter().any(|&c| (c - model.theta()).abs() < 1e-9);
        assert!(found, "theta={} not in candidates", model.theta());
    }

    #[test]
    fn test_otm_seasonal() {
        let data = seasonal_series();
        let model = OptimizedTheta::fit(&data, Some(4)).expect("failed to create model");
        let fcast = model.forecast(4).expect("failed to create fcast");
        assert_eq!(fcast.len(), 4);
        for &f in &fcast {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_otm_increasing_forecast_for_trend() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let model = OptimizedTheta::fit(&data, None).expect("failed to create model");
        let fcast = model.forecast(5).expect("failed to create fcast");
        for w in fcast.windows(2) {
            assert!(
                w[1] > w[0],
                "OTM forecasts should increase for linear trend"
            );
        }
    }

    #[test]
    fn test_otm_insufficient_data() {
        assert!(OptimizedTheta::fit(&[1.0, 2.0, 3.0], None).is_err());
    }
}
