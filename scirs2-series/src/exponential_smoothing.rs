//! Exponential Smoothing Suite
//!
//! Implements a family of exponential smoothing forecasting methods:
//! - Simple Exponential Smoothing (SES): level only
//! - Holt's Double Exponential Smoothing: level + trend
//! - Holt-Winters Triple Exponential Smoothing: level + trend + seasonality
//!
//! These are classic time series forecasting methods that form the basis
//! of the ETS (Error-Trend-Seasonal) state space framework.
//!
//! # References
//!
//! - Hyndman, R.J. & Athanasopoulos, G. (2021) "Forecasting: Principles and Practice", 3rd ed.
//! - Gardner Jr, E.S. (1985) "Exponential smoothing: The state of the art"
//! - Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2008) "Forecasting with
//!   Exponential Smoothing: The State Space Approach"

use crate::error::{Result, TimeSeriesError};

/// Seasonal component type for Holt-Winters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalType {
    /// Additive seasonality: season affects level by addition/subtraction
    Additive,
    /// Multiplicative seasonality: season scales the level by a factor
    Multiplicative,
}

/// Fitted result from Simple Exponential Smoothing
///
/// Implements the basic exponential smoothing recursion:
/// l_t = alpha * y_t + (1 - alpha) * l_{t-1}
///
/// Forecasts beyond the training set are constant (flat forecast).
#[derive(Debug, Clone)]
pub struct SimpleExponentialSmoothing {
    /// Smoothing parameter (0 < alpha <= 1)
    alpha: f64,
    /// Final level (used for forecasting)
    initial_level: f64,
}

impl SimpleExponentialSmoothing {
    /// Fit SES to data, optionally providing alpha.
    ///
    /// When `alpha` is `None`, the parameter is optimized by minimizing
    /// the sum of squared one-step-ahead forecast errors via golden-section search.
    ///
    /// # Arguments
    /// * `data` - Training observations (must have at least 2 values)
    /// * `alpha` - Smoothing parameter; if `None`, auto-optimized
    pub fn fit(data: &[f64], alpha: Option<f64>) -> Result<Self> {
        if data.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "SES requires at least 2 observations".to_string(),
                required: 2,
                actual: data.len(),
            });
        }

        let alpha = match alpha {
            Some(a) => {
                if !(0.0 < a && a <= 1.0) {
                    return Err(TimeSeriesError::InvalidParameter {
                        name: "alpha".to_string(),
                        message: "alpha must be in (0, 1]".to_string(),
                    });
                }
                a
            }
            None => optimize_alpha_ses(data)?,
        };

        // Run the recursion to obtain the final level
        let level = compute_ses_level(data, alpha, data[0]);
        Ok(Self {
            alpha,
            initial_level: level,
        })
    }

    /// Produce h-step-ahead flat forecasts.
    ///
    /// SES forecasts are constant: every future value equals the last level.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        vec![self.initial_level; h]
    }

    /// Compute in-sample one-step-ahead fitted values.
    pub fn fitted_values(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut fitted = Vec::with_capacity(data.len());
        let mut level = data[0];
        // The first fitted value is y[0] itself (no prior level)
        fitted.push(level);
        for &y in data.iter().take(data.len() - 1) {
            level = self.alpha * y + (1.0 - self.alpha) * level;
            fitted.push(level);
        }
        fitted
    }

    /// Return the smoothing parameter alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the final fitted level.
    pub fn level(&self) -> f64 {
        self.initial_level
    }
}

/// Compute the final level after running the SES recursion.
fn compute_ses_level(data: &[f64], alpha: f64, l0: f64) -> f64 {
    let mut level = l0;
    for &y in data {
        level = alpha * y + (1.0 - alpha) * level;
    }
    level
}

/// Sum of squared one-step forecast errors for SES with given alpha.
fn ses_sse(data: &[f64], alpha: f64) -> f64 {
    let mut level = data[0];
    let mut sse = 0.0;
    for &y in data.iter().skip(1) {
        let err = y - level;
        sse += err * err;
        level = alpha * y + (1.0 - alpha) * level;
    }
    sse
}

/// Golden-section search on (0, 1) to minimize SES SSE.
fn optimize_alpha_ses(data: &[f64]) -> Result<f64> {
    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = 1e-4_f64;
    let mut b = 1.0_f64 - 1e-10;
    let tol = 1e-7;

    let mut x1 = b - golden_ratio * (b - a);
    let mut x2 = a + golden_ratio * (b - a);
    let mut f1 = ses_sse(data, x1);
    let mut f2 = ses_sse(data, x2);

    for _ in 0..200 {
        if (b - a).abs() < tol {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - golden_ratio * (b - a);
            f1 = ses_sse(data, x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + golden_ratio * (b - a);
            f2 = ses_sse(data, x2);
        }
    }
    let best = (a + b) / 2.0;
    if best <= 0.0 || best > 1.0 {
        return Err(TimeSeriesError::OptimizationError(
            "SES alpha optimization produced out-of-range value".to_string(),
        ));
    }
    Ok(best)
}

// ──────────────────────────────────────────────────────────────────────────────
// Holt's Double Exponential Smoothing
// ──────────────────────────────────────────────────────────────────────────────

/// Holt's Double (Linear) Exponential Smoothing with optional damped trend.
///
/// State equations:
/// ```text
/// l_t = alpha * y_t  + (1 - alpha) * (l_{t-1} + b_{t-1})
/// b_t = beta  * (l_t - l_{t-1}) + (1 - beta)  * b_{t-1}
/// ```
///
/// Damped forecast: ŷ_{t+h} = l_t + phi * (1 - phi^h) / (1 - phi) * b_t
#[derive(Debug, Clone)]
pub struct HoltLinear {
    /// Level smoothing parameter
    alpha: f64,
    /// Trend smoothing parameter
    beta: f64,
    /// Final fitted level
    level: f64,
    /// Final fitted trend
    trend: f64,
}

impl HoltLinear {
    /// Fit Holt's linear model to data.
    ///
    /// Auto-optimizes parameters by minimizing SSE when `alpha` or `beta` is `None`.
    pub fn fit(data: &[f64], alpha: Option<f64>, beta: Option<f64>) -> Result<Self> {
        if data.len() < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Holt linear requires at least 4 observations".to_string(),
                required: 4,
                actual: data.len(),
            });
        }

        validate_smoothing_param("alpha", alpha)?;
        validate_smoothing_param("beta", beta)?;

        let (opt_alpha, opt_beta) = match (alpha, beta) {
            (Some(a), Some(b)) => (a, b),
            _ => optimize_holt_params(data, alpha, beta)?,
        };

        let (l, b) = holt_final_state(data, opt_alpha, opt_beta);
        Ok(Self {
            alpha: opt_alpha,
            beta: opt_beta,
            level: l,
            trend: b,
        })
    }

    /// Produce h-step-ahead forecasts with linear extrapolation.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        (1..=h)
            .map(|k| self.level + (k as f64) * self.trend)
            .collect()
    }

    /// Produce h-step-ahead forecasts with damped trend.
    ///
    /// # Arguments
    /// * `h` - Forecast horizon
    /// * `phi` - Damping factor (0 < phi < 1); values near 1 give near-linear trend
    pub fn damped_forecast(&self, h: usize, phi: f64) -> Vec<f64> {
        if !(0.0 < phi && phi < 1.0) {
            // Fall back to undamped if phi is out of range
            return self.forecast(h);
        }
        (1..=h)
            .map(|k| {
                // Sum_{i=1}^{k} phi^i = phi * (1 - phi^k) / (1 - phi)
                let phi_sum = phi * (1.0 - phi.powi(k as i32)) / (1.0 - phi);
                self.level + phi_sum * self.trend
            })
            .collect()
    }

    /// In-sample fitted values.
    pub fn fitted_values(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < 2 {
            return data.to_vec();
        }
        let mut level = data[0];
        let mut trend = data[1] - data[0];
        let mut fitted = Vec::with_capacity(data.len());
        fitted.push(level + trend);
        for &y in data.iter().skip(1) {
            let l_prev = level;
            level = self.alpha * y + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - l_prev) + (1.0 - self.beta) * trend;
            fitted.push(level + trend);
        }
        fitted
    }

    /// Return the level smoothing parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the trend smoothing parameter.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

/// Validate that an optional smoothing parameter is in (0, 1].
fn validate_smoothing_param(name: &str, val: Option<f64>) -> Result<()> {
    if let Some(v) = val {
        if !(0.0 < v && v <= 1.0) {
            return Err(TimeSeriesError::InvalidParameter {
                name: name.to_string(),
                message: format!("{name} must be in (0, 1], got {v}"),
            });
        }
    }
    Ok(())
}

/// Run Holt recursion and return final (level, trend).
fn holt_final_state(data: &[f64], alpha: f64, beta: f64) -> (f64, f64) {
    let mut level = data[0];
    let mut trend = data[1] - data[0];
    for &y in data.iter().skip(1) {
        let l_prev = level;
        level = alpha * y + (1.0 - alpha) * (level + trend);
        trend = beta * (level - l_prev) + (1.0 - beta) * trend;
    }
    (level, trend)
}

/// Sum of squared forecast errors for Holt model.
fn holt_sse(data: &[f64], alpha: f64, beta: f64) -> f64 {
    if data.len() < 2 {
        return f64::INFINITY;
    }
    let mut level = data[0];
    let mut trend = data[1] - data[0];
    let mut sse = 0.0;
    for &y in data.iter().skip(1) {
        let forecast = level + trend;
        let err = y - forecast;
        sse += err * err;
        let l_prev = level;
        level = alpha * y + (1.0 - alpha) * (level + trend);
        trend = beta * (level - l_prev) + (1.0 - beta) * trend;
    }
    sse
}

/// Optimize Holt alpha and beta via grid-coordinate descent.
fn optimize_holt_params(
    data: &[f64],
    fixed_alpha: Option<f64>,
    fixed_beta: Option<f64>,
) -> Result<(f64, f64)> {
    let grid: Vec<f64> = (1..=20).map(|i| i as f64 * 0.05).collect();

    let mut best_alpha = fixed_alpha.unwrap_or(0.3);
    let mut best_beta = fixed_beta.unwrap_or(0.1);
    let mut best_sse = holt_sse(data, best_alpha, best_beta);

    // Grid search + local refinement
    if fixed_alpha.is_none() || fixed_beta.is_none() {
        for &a in &grid {
            for &b in &grid {
                let alpha = fixed_alpha.unwrap_or(a);
                let beta = fixed_beta.unwrap_or(b);
                let sse = holt_sse(data, alpha, beta);
                if sse < best_sse {
                    best_sse = sse;
                    best_alpha = alpha;
                    best_beta = beta;
                }
            }
        }
    }

    // Refine with golden-section on the free parameter(s)
    if fixed_alpha.is_none() {
        best_alpha = golden_section_1d(|a| holt_sse(data, a, best_beta), 1e-4, 1.0 - 1e-10);
    }
    if fixed_beta.is_none() {
        best_beta = golden_section_1d(|b| holt_sse(data, best_alpha, b), 1e-4, 1.0 - 1e-10);
    }

    Ok((best_alpha, best_beta))
}

/// Golden-section search minimizing a univariate function on [lo, hi].
fn golden_section_1d(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut x1 = b - gr * (b - a);
    let mut x2 = a + gr * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..300 {
        if (b - a).abs() < 1e-8 {
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

// ──────────────────────────────────────────────────────────────────────────────
// Holt-Winters Triple Exponential Smoothing
// ──────────────────────────────────────────────────────────────────────────────

/// Holt-Winters Triple Exponential Smoothing with trend and seasonality.
///
/// Supports both additive and multiplicative seasonal decomposition.
///
/// State equations (additive):
/// ```text
/// l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
/// b_t = beta  * (l_t - l_{t-1}) + (1 - beta)  * b_{t-1}
/// s_t = gamma * (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) * s_{t-m}
/// ŷ_{t+h} = l_t + h * b_t + s_{t+h-m*(floor((h-1)/m)+1)}
/// ```
///
/// State equations (multiplicative):
/// ```text
/// l_t = alpha * (y_t / s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
/// b_t = beta  * (l_t - l_{t-1}) + (1 - beta)  * b_{t-1}
/// s_t = gamma * (y_t / (l_{t-1} + b_{t-1})) + (1 - gamma) * s_{t-m}
/// ŷ_{t+h} = (l_t + h * b_t) * s_{t+h-m*(floor((h-1)/m)+1)}
/// ```
#[derive(Debug, Clone)]
pub struct HoltWinters {
    /// Level smoothing parameter
    alpha: f64,
    /// Trend smoothing parameter
    beta: f64,
    /// Seasonal smoothing parameter
    gamma: f64,
    /// Seasonal period (m)
    period: usize,
    /// Additive or multiplicative seasonality
    seasonal_type: SeasonalType,
    /// Final fitted level
    level: f64,
    /// Final fitted trend
    trend: f64,
    /// Final fitted seasonal indices (length = period)
    seasonals: Vec<f64>,
    /// Number of training observations
    n_obs: usize,
}

impl HoltWinters {
    /// Fit Holt-Winters to data.
    ///
    /// Parameters are optimized via coordinate descent when not provided.
    /// For multiplicative seasonality, all data values must be strictly positive.
    ///
    /// # Arguments
    /// * `data` - Training series (length >= 2 * period)
    /// * `period` - Seasonal period m (e.g., 12 for monthly, 4 for quarterly)
    /// * `seasonal` - Additive or Multiplicative seasonality
    /// * `alpha`, `beta`, `gamma` - Smoothing parameters; `None` = auto-optimize
    pub fn fit(
        data: &[f64],
        period: usize,
        seasonal: SeasonalType,
        alpha: Option<f64>,
        beta: Option<f64>,
        gamma: Option<f64>,
    ) -> Result<Self> {
        if period < 2 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "period".to_string(),
                message: "Seasonal period must be >= 2".to_string(),
            });
        }
        if data.len() < 2 * period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Holt-Winters requires at least 2 full seasonal cycles".to_string(),
                required: 2 * period,
                actual: data.len(),
            });
        }
        if seasonal == SeasonalType::Multiplicative {
            if data.iter().any(|&v| v <= 0.0) {
                return Err(TimeSeriesError::InvalidInput(
                    "Multiplicative Holt-Winters requires all positive data values".to_string(),
                ));
            }
        }

        validate_smoothing_param("alpha", alpha)?;
        validate_smoothing_param("beta", beta)?;
        validate_smoothing_param("gamma", gamma)?;

        let (opt_alpha, opt_beta, opt_gamma) =
            optimize_hw_params(data, period, seasonal, alpha, beta, gamma)?;

        let (l, b, s) = hw_final_state(data, period, seasonal, opt_alpha, opt_beta, opt_gamma);
        Ok(Self {
            alpha: opt_alpha,
            beta: opt_beta,
            gamma: opt_gamma,
            period,
            seasonal_type: seasonal,
            level: l,
            trend: b,
            seasonals: s,
            n_obs: data.len(),
        })
    }

    /// Produce h-step-ahead point forecasts.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        let m = self.period;
        (1..=h)
            .map(|k| {
                let s_idx = (self.n_obs - m + (k - 1) % m) % m;
                let s = self.seasonals[s_idx];
                match self.seasonal_type {
                    SeasonalType::Additive => self.level + (k as f64) * self.trend + s,
                    SeasonalType::Multiplicative => (self.level + (k as f64) * self.trend) * s,
                }
            })
            .collect()
    }

    /// Compute in-sample one-step-ahead fitted values.
    pub fn fitted_values(&self, data: &[f64]) -> Vec<f64> {
        let m = self.period;
        let init = hw_initial_components(data, m, self.seasonal_type);
        let mut level = init.0;
        let mut trend = init.1;
        let mut seasonals = init.2;

        let mut fitted = Vec::with_capacity(data.len());
        for (t, &y) in data.iter().enumerate() {
            let s_lag = seasonals[t % m];
            let yhat = match self.seasonal_type {
                SeasonalType::Additive => level + trend + s_lag,
                SeasonalType::Multiplicative => (level + trend) * s_lag,
            };
            fitted.push(yhat);
            let l_prev = level;
            match self.seasonal_type {
                SeasonalType::Additive => {
                    level = self.alpha * (y - s_lag) + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta * (level - l_prev) + (1.0 - self.beta) * trend;
                    seasonals[t % m] = self.gamma * (y - level) + (1.0 - self.gamma) * s_lag;
                }
                SeasonalType::Multiplicative => {
                    let safe_s = if s_lag.abs() < 1e-12 { 1e-12 } else { s_lag };
                    level = self.alpha * (y / safe_s) + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta * (level - l_prev) + (1.0 - self.beta) * trend;
                    let denom = level + trend;
                    let safe_denom = if denom.abs() < 1e-12 { 1e-12 } else { denom };
                    seasonals[t % m] = self.gamma * (y / safe_denom) + (1.0 - self.gamma) * s_lag;
                }
            }
        }
        fitted
    }

    /// Akaike Information Criterion for the fitted model.
    pub fn aic(&self, data: &[f64]) -> f64 {
        let fitted = self.fitted_values(data);
        let n = data.len() as f64;
        let sse: f64 = data
            .iter()
            .zip(fitted.iter())
            .map(|(&y, &yhat)| (y - yhat).powi(2))
            .sum();
        let sigma2 = sse / n;
        // k = alpha + beta + gamma + initial level + initial trend + period seasonal indices
        let k = 3 + 2 + self.period;
        let log_lik = -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());
        -2.0 * log_lik + 2.0 * k as f64
    }

    /// Return the level smoothing parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the trend smoothing parameter.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Return the seasonal smoothing parameter.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Return the seasonal period.
    pub fn period(&self) -> usize {
        self.period
    }
}

/// Compute initial level, trend, and seasonal indices using
/// the classical decomposition approach.
fn hw_initial_components(
    data: &[f64],
    period: usize,
    seasonal_type: SeasonalType,
) -> (f64, f64, Vec<f64>) {
    // Level: average of first complete cycle
    let level: f64 = data[..period].iter().sum::<f64>() / period as f64;

    // Trend: average of trend computed over initial two full cycles
    let trend = if data.len() >= 2 * period {
        let second_avg: f64 = data[period..2 * period].iter().sum::<f64>() / period as f64;
        (second_avg - level) / period as f64
    } else {
        0.0
    };

    // Seasonal indices: average ratio/deviation per season position
    let n_complete = data.len() / period;
    let mut seasonals = vec![0.0_f64; period];
    for i in 0..period {
        let sum: f64 = (0..n_complete)
            .map(|j| {
                let idx = j * period + i;
                if idx < data.len() {
                    match seasonal_type {
                        SeasonalType::Additive => data[idx] - (level + (idx as f64 + 1.0) * trend),
                        SeasonalType::Multiplicative => {
                            let denom = level + (idx as f64 + 1.0) * trend;
                            if denom.abs() < 1e-12 {
                                1.0
                            } else {
                                data[idx] / denom
                            }
                        }
                    }
                } else {
                    match seasonal_type {
                        SeasonalType::Additive => 0.0,
                        SeasonalType::Multiplicative => 1.0,
                    }
                }
            })
            .sum();
        seasonals[i] = sum / n_complete as f64;
    }

    // Normalize multiplicative indices so they sum to period
    if seasonal_type == SeasonalType::Multiplicative {
        let mean_s = seasonals.iter().sum::<f64>() / period as f64;
        if mean_s.abs() > 1e-12 {
            for s in &mut seasonals {
                *s /= mean_s;
            }
        }
    }

    (level, trend, seasonals)
}

/// Run Holt-Winters recursion and return final (level, trend, seasonals).
fn hw_final_state(
    data: &[f64],
    period: usize,
    seasonal_type: SeasonalType,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> (f64, f64, Vec<f64>) {
    let (mut level, mut trend, mut seasonals) = hw_initial_components(data, period, seasonal_type);

    for (t, &y) in data.iter().enumerate() {
        let s_lag = seasonals[t % period];
        let l_prev = level;
        match seasonal_type {
            SeasonalType::Additive => {
                level = alpha * (y - s_lag) + (1.0 - alpha) * (level + trend);
                trend = beta * (level - l_prev) + (1.0 - beta) * trend;
                seasonals[t % period] = gamma * (y - level) + (1.0 - gamma) * s_lag;
            }
            SeasonalType::Multiplicative => {
                let safe_s = if s_lag.abs() < 1e-12 { 1e-12 } else { s_lag };
                level = alpha * (y / safe_s) + (1.0 - alpha) * (level + trend);
                trend = beta * (level - l_prev) + (1.0 - beta) * trend;
                let denom = level + trend;
                let safe_denom = if denom.abs() < 1e-12 { 1e-12 } else { denom };
                seasonals[t % period] = gamma * (y / safe_denom) + (1.0 - gamma) * s_lag;
            }
        }
    }
    (level, trend, seasonals)
}

/// SSE for Holt-Winters with given parameters.
fn hw_sse(
    data: &[f64],
    period: usize,
    seasonal_type: SeasonalType,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> f64 {
    let (mut level, mut trend, mut seasonals) = hw_initial_components(data, period, seasonal_type);

    let mut sse = 0.0;
    for (t, &y) in data.iter().enumerate() {
        let s_lag = seasonals[t % period];
        let yhat = match seasonal_type {
            SeasonalType::Additive => level + trend + s_lag,
            SeasonalType::Multiplicative => (level + trend) * s_lag,
        };
        let err = y - yhat;
        sse += err * err;

        let l_prev = level;
        match seasonal_type {
            SeasonalType::Additive => {
                level = alpha * (y - s_lag) + (1.0 - alpha) * (level + trend);
                trend = beta * (level - l_prev) + (1.0 - beta) * trend;
                seasonals[t % period] = gamma * (y - level) + (1.0 - gamma) * s_lag;
            }
            SeasonalType::Multiplicative => {
                let safe_s = if s_lag.abs() < 1e-12 { 1e-12 } else { s_lag };
                level = alpha * (y / safe_s) + (1.0 - alpha) * (level + trend);
                trend = beta * (level - l_prev) + (1.0 - beta) * trend;
                let denom = level + trend;
                let safe_denom = if denom.abs() < 1e-12 { 1e-12 } else { denom };
                seasonals[t % period] = gamma * (y / safe_denom) + (1.0 - gamma) * s_lag;
            }
        }
    }
    sse
}

/// Optimize Holt-Winters parameters via grid search + coordinate refinement.
fn optimize_hw_params(
    data: &[f64],
    period: usize,
    seasonal_type: SeasonalType,
    fixed_alpha: Option<f64>,
    fixed_beta: Option<f64>,
    fixed_gamma: Option<f64>,
) -> Result<(f64, f64, f64)> {
    // Coarse grid
    let coarse: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8];

    let mut best_alpha = fixed_alpha.unwrap_or(0.3);
    let mut best_beta = fixed_beta.unwrap_or(0.1);
    let mut best_gamma = fixed_gamma.unwrap_or(0.2);
    let mut best_sse = hw_sse(
        data,
        period,
        seasonal_type,
        best_alpha,
        best_beta,
        best_gamma,
    );

    // Grid search over free parameters
    let alpha_candidates: Vec<f64> = if fixed_alpha.is_some() {
        {
            let v = fixed_alpha.expect("fixed_alpha is Some (checked above)");
            vec![v]
        }
    } else {
        coarse.clone()
    };
    let beta_candidates: Vec<f64> = if fixed_beta.is_some() {
        {
            let v = fixed_beta.expect("fixed_beta is Some (checked above)");
            vec![v]
        }
    } else {
        coarse.clone()
    };
    let gamma_candidates: Vec<f64> = if fixed_gamma.is_some() {
        {
            let v = fixed_gamma.expect("fixed_gamma is Some (checked above)");
            vec![v]
        }
    } else {
        coarse.clone()
    };

    for &a in &alpha_candidates {
        for &b in &beta_candidates {
            for &g in &gamma_candidates {
                let sse = hw_sse(data, period, seasonal_type, a, b, g);
                if sse < best_sse {
                    best_sse = sse;
                    best_alpha = a;
                    best_beta = b;
                    best_gamma = g;
                }
            }
        }
    }

    // Coordinate-descent refinement
    for _ in 0..5 {
        if fixed_alpha.is_none() {
            best_alpha = golden_section_1d(
                |a| hw_sse(data, period, seasonal_type, a, best_beta, best_gamma),
                1e-4,
                1.0 - 1e-10,
            );
        }
        if fixed_beta.is_none() {
            best_beta = golden_section_1d(
                |b| hw_sse(data, period, seasonal_type, best_alpha, b, best_gamma),
                1e-4,
                1.0 - 1e-10,
            );
        }
        if fixed_gamma.is_none() {
            best_gamma = golden_section_1d(
                |g| hw_sse(data, period, seasonal_type, best_alpha, best_beta, g),
                1e-4,
                1.0 - 1e-10,
            );
        }
    }

    Ok((best_alpha, best_beta, best_gamma))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seasonal_data_additive() -> Vec<f64> {
        // 3 cycles of period-4 seasonal data with slight linear trend
        vec![
            10.0, 14.0, 12.0, 8.0, 11.0, 15.0, 13.0, 9.0, 12.0, 16.0, 14.0, 10.0,
        ]
    }

    fn seasonal_data_multiplicative() -> Vec<f64> {
        // 3 cycles of positive multiplicative seasonal data
        vec![
            100.0, 140.0, 120.0, 80.0, 110.0, 154.0, 132.0, 88.0, 121.0, 169.4, 145.2, 96.8,
        ]
    }

    // ── SES tests ──────────────────────────────────────────────────────────
    #[test]
    fn test_ses_fit_auto() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64 + 0.1 * (i % 3) as f64).collect();
        let model = SimpleExponentialSmoothing::fit(&data, None).expect("failed to create model");
        assert!(
            model.alpha() > 0.0 && model.alpha() <= 1.0,
            "alpha out of range"
        );
    }

    #[test]
    fn test_ses_fit_fixed_alpha() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model =
            SimpleExponentialSmoothing::fit(&data, Some(0.3)).expect("failed to create model");
        assert!((model.alpha() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_ses_forecast_constant() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model =
            SimpleExponentialSmoothing::fit(&data, Some(0.5)).expect("failed to create model");
        let fcast = model.forecast(5);
        // All forecasts should equal the last level
        for &f in &fcast {
            assert!((f - fcast[0]).abs() < 1e-10, "SES forecast should be flat");
        }
    }

    #[test]
    fn test_ses_fitted_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let model =
            SimpleExponentialSmoothing::fit(&data, Some(0.4)).expect("failed to create model");
        let fitted = model.fitted_values(&data);
        assert_eq!(fitted.len(), data.len());
    }

    #[test]
    fn test_ses_converges_on_constant() {
        // For constant data, optimal alpha should drive level toward that constant
        let data = vec![5.0_f64; 30];
        let model = SimpleExponentialSmoothing::fit(&data, None).expect("failed to create model");
        let fcast = model.forecast(1);
        assert!(
            (fcast[0] - 5.0).abs() < 1e-6,
            "SES should converge on constant series"
        );
    }

    #[test]
    fn test_ses_invalid_alpha() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(SimpleExponentialSmoothing::fit(&data, Some(0.0)).is_err());
        assert!(SimpleExponentialSmoothing::fit(&data, Some(1.5)).is_err());
    }

    #[test]
    fn test_ses_insufficient_data() {
        assert!(SimpleExponentialSmoothing::fit(&[1.0], None).is_err());
    }

    // ── HoltLinear tests ───────────────────────────────────────────────────
    #[test]
    fn test_holt_fit_and_forecast() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let model = HoltLinear::fit(&data, None, None).expect("failed to create model");
        let fcast = model.forecast(5);
        assert_eq!(fcast.len(), 5);
        // For a pure linear trend, forecasts should also be increasing
        for w in fcast.windows(2) {
            assert!(w[1] > w[0], "linear trend forecasts must be increasing");
        }
    }

    #[test]
    fn test_holt_damped_forecast() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let model = HoltLinear::fit(&data, None, None).expect("failed to create model");
        let undamped = model.forecast(10);
        let damped = model.damped_forecast(10, 0.9);
        assert_eq!(damped.len(), 10);
        // Damped forecasts should be below undamped ones for positive trend
        let total_undamped: f64 = undamped.iter().sum();
        let total_damped: f64 = damped.iter().sum();
        assert!(
            total_damped < total_undamped,
            "damped should be less than undamped"
        );
    }

    #[test]
    fn test_holt_insufficient_data() {
        assert!(HoltLinear::fit(&[1.0, 2.0, 3.0], None, None).is_err());
    }

    // ── HoltWinters tests ──────────────────────────────────────────────────
    #[test]
    fn test_hw_additive_fit() {
        let data = seasonal_data_additive();
        let model = HoltWinters::fit(&data, 4, SeasonalType::Additive, None, None, None)
            .expect("failed to create model");
        assert!(model.alpha() > 0.0 && model.alpha() <= 1.0);
        assert!(model.beta() > 0.0 && model.beta() <= 1.0);
        assert!(model.gamma() > 0.0 && model.gamma() <= 1.0);
    }

    #[test]
    fn test_hw_additive_forecast_length() {
        let data = seasonal_data_additive();
        let model = HoltWinters::fit(
            &data,
            4,
            SeasonalType::Additive,
            Some(0.3),
            Some(0.1),
            Some(0.2),
        )
        .expect("unexpected None or Err");
        let fcast = model.forecast(8);
        assert_eq!(fcast.len(), 8);
    }

    #[test]
    fn test_hw_additive_fitted_values() {
        let data = seasonal_data_additive();
        let model = HoltWinters::fit(
            &data,
            4,
            SeasonalType::Additive,
            Some(0.4),
            Some(0.1),
            Some(0.3),
        )
        .expect("unexpected None or Err");
        let fitted = model.fitted_values(&data);
        assert_eq!(fitted.len(), data.len());
    }

    #[test]
    fn test_hw_multiplicative_fit() {
        let data = seasonal_data_multiplicative();
        let model = HoltWinters::fit(&data, 4, SeasonalType::Multiplicative, None, None, None)
            .expect("unexpected None or Err");
        let fcast = model.forecast(4);
        assert_eq!(fcast.len(), 4);
        for &f in &fcast {
            assert!(f.is_finite(), "forecast must be finite");
            assert!(
                f > 0.0,
                "multiplicative forecast must be positive for positive data"
            );
        }
    }

    #[test]
    fn test_hw_aic() {
        let data = seasonal_data_additive();
        let model = HoltWinters::fit(
            &data,
            4,
            SeasonalType::Additive,
            Some(0.3),
            Some(0.1),
            Some(0.2),
        )
        .expect("unexpected None or Err");
        let aic = model.aic(&data);
        assert!(aic.is_finite(), "AIC must be finite");
    }

    #[test]
    fn test_hw_period_too_small() {
        let data = seasonal_data_additive();
        assert!(HoltWinters::fit(&data, 1, SeasonalType::Additive, None, None, None).is_err());
    }

    #[test]
    fn test_hw_insufficient_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // Only 1 period
        assert!(HoltWinters::fit(&data, 4, SeasonalType::Additive, None, None, None).is_err());
    }

    #[test]
    fn test_hw_multiplicative_requires_positive() {
        let data = vec![1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(
            HoltWinters::fit(&data, 4, SeasonalType::Multiplicative, None, None, None).is_err()
        );
    }

    #[test]
    fn test_hw_seasonal_pattern_preserved() {
        // Data with strong seasonal pattern: high in Q1, low in Q3
        let data = vec![
            20.0, 10.0, 5.0, 15.0, 22.0, 11.0, 6.0, 16.0, 24.0, 12.0, 7.0, 17.0,
        ];
        let model = HoltWinters::fit(
            &data,
            4,
            SeasonalType::Additive,
            Some(0.3),
            Some(0.1),
            Some(0.3),
        )
        .expect("unexpected None or Err");
        let fcast = model.forecast(4);
        // Q1 should be the highest, Q3 the lowest
        assert!(
            fcast[0] > fcast[2],
            "Q1 forecast should be greater than Q3 forecast: {:.2} vs {:.2}",
            fcast[0],
            fcast[2]
        );
    }
}
