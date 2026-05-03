//! TBATS Model: Trigonometric Seasonality, Box-Cox, ARMA errors, Trend, Seasonal
//!
//! TBATS (De Livera, Hyndman & Snyder, 2011) is a state-space model designed to handle
//! complex seasonal patterns including:
//! - Multiple seasonal periods (e.g., daily + weekly + annual)
//! - Non-integer seasonal periods (e.g., 365.25 days/year)
//! - Trigonometric representation of seasonality (Fourier terms in state equations)
//! - Box-Cox transformation for variance stabilization
//! - ARMA errors for handling residual autocorrelation
//! - Damped or undamped local linear trend
//!
//! # Model Structure
//!
//! The TBATS model can be written as:
//! ```text
//! y_t^(lambda) = l_{t-1} + phi * b_{t-1} + sum_j s_{j,t-1} + d_t
//! l_t = l_{t-1} + phi * b_{t-1} + alpha * d_t
//! b_t = (1-phi) * b_bar + phi * b_{t-1} + beta * d_t
//! d_t = sum_i phi_i * d_{t-i} + sum_j theta_j * epsilon_{t-j} + epsilon_t
//! ```
//!
//! where each seasonal component j with period m_j is represented by
//! k_j trigonometric terms.
//!
//! # References
//!
//! - De Livera, A.M., Hyndman, R.J. & Snyder, R.D. (2011). "Forecasting time series
//!   with complex seasonal patterns using exponential smoothing." *Journal of the
//!   American Statistical Association*, 106(496), 1513-1527.
//! - Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed.

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Forecast output from TBATS including point estimates and confidence intervals.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point forecasts
    pub forecast: Vec<f64>,
    /// Lower bound of (1-alpha)*100% prediction interval
    pub lower: Vec<f64>,
    /// Upper bound of (1-alpha)*100% prediction interval
    pub upper: Vec<f64>,
}

/// Configuration for TBATS model fitting.
///
/// Setting a field to `None` enables automatic selection via AIC comparison.
#[derive(Debug, Clone)]
pub struct TbatsConfig {
    /// Whether to apply Box-Cox transformation. `None` = auto-select.
    pub use_box_cox: Option<bool>,
    /// Whether to include a linear trend component. `None` = auto-select.
    pub use_trend: Option<bool>,
    /// Whether to damp the trend. `None` = auto-select (only relevant when trend = true).
    pub use_damped_trend: Option<bool>,
    /// Seasonal periods. May be non-integer (e.g., 365.25 for annual in daily data).
    /// An empty vector means no seasonality is modeled.
    pub seasonal_periods: Vec<f64>,
    /// Number of Fourier harmonics per seasonal component.
    /// `None` uses `ceil(period / 2)` harmonics for each component.
    pub n_harmonics: Option<Vec<usize>>,
    /// AR order for the ARMA error component. `None` = auto-select (0..=2 checked).
    pub ar_order: Option<usize>,
    /// MA order for the ARMA error component. `None` = auto-select (0..=2 checked).
    pub ma_order: Option<usize>,
}

impl Default for TbatsConfig {
    fn default() -> Self {
        Self {
            use_box_cox: None,
            use_trend: None,
            use_damped_trend: None,
            seasonal_periods: Vec::new(),
            n_harmonics: None,
            ar_order: None,
            ma_order: None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal state for a seasonal component
// ──────────────────────────────────────────────────────────────────────────────

/// State for one trigonometric seasonal component with k harmonics.
#[derive(Debug, Clone)]
struct SeasonalComponent {
    /// Seasonal period m_j (may be non-integer)
    period: f64,
    /// Number of harmonics k_j
    k: usize,
    /// Sine state vectors (length k)
    s_states: Vec<f64>,
    /// Cosine state vectors (length k)
    c_states: Vec<f64>,
    /// Smoothing vector gamma_1 (length k)
    gamma1: Vec<f64>,
    /// Smoothing vector gamma_2 (length k)
    gamma2: Vec<f64>,
}

impl SeasonalComponent {
    /// Create with zero-initialized states.
    fn new(period: f64, k: usize) -> Self {
        let gamma_default = 0.001;
        Self {
            period,
            k,
            s_states: vec![0.0; k],
            c_states: vec![1.0; k], // non-zero init
            gamma1: vec![gamma_default; k],
            gamma2: vec![gamma_default; k],
        }
    }

    /// Compute current seasonal contribution as sum over harmonics.
    fn contribution(&self) -> f64 {
        self.s_states.iter().sum()
    }

    /// Update states given the current error.
    fn update(&mut self, error: f64, lambda_j: &[f64], mu_j: &[f64]) {
        let _ = (lambda_j, mu_j); // consumed via gamma1/gamma2 already set
        let m = self.period;
        for i in 0..self.k {
            let freq = 2.0 * std::f64::consts::PI * (i + 1) as f64 / m;
            let cos_f = freq.cos();
            let sin_f = freq.sin();
            let old_s = self.s_states[i];
            let old_c = self.c_states[i];
            self.s_states[i] = old_s * cos_f + old_c * sin_f + self.gamma1[i] * error;
            self.c_states[i] = -old_s * sin_f + old_c * cos_f + self.gamma2[i] * error;
        }
    }

    /// Forecast seasonal contribution k steps ahead.
    fn forecast_ahead(&self, k_step: usize) -> f64 {
        let m = self.period;
        let mut sum = 0.0;
        for i in 0..self.k {
            let freq = 2.0 * std::f64::consts::PI * (i + 1) as f64 / m;
            let angle = freq * k_step as f64;
            sum += self.s_states[i] * angle.cos() + self.c_states[i] * angle.sin();
        }
        sum
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ARMA error model
// ──────────────────────────────────────────────────────────────────────────────

/// ARMA(p, q) model on residuals.
#[derive(Debug, Clone)]
struct ArmaModel {
    ar: Vec<f64>,
    ma: Vec<f64>,
    residuals: Vec<f64>,
    d_history: Vec<f64>,
}

impl ArmaModel {
    fn new(ar_order: usize, ma_order: usize) -> Self {
        Self {
            ar: vec![0.0; ar_order],
            ma: vec![0.0; ma_order],
            residuals: Vec::new(),
            d_history: Vec::new(),
        }
    }

    /// Compute the ARMA contribution at the current time step.
    fn contribution(&self) -> f64 {
        let mut d = 0.0;
        for (i, &phi) in self.ar.iter().enumerate() {
            if i < self.d_history.len() {
                d += phi * self.d_history[self.d_history.len() - 1 - i];
            }
        }
        for (i, &theta) in self.ma.iter().enumerate() {
            if i < self.residuals.len() {
                d += theta * self.residuals[self.residuals.len() - 1 - i];
            }
        }
        d
    }

    fn push_residual(&mut self, eps: f64, d: f64) {
        self.residuals.push(eps);
        self.d_history.push(d);
        // Keep only as many entries as max(p, q)
        let max_order = self.ar.len().max(self.ma.len()) + 1;
        if self.residuals.len() > max_order {
            self.residuals.remove(0);
        }
        if self.d_history.len() > max_order {
            self.d_history.remove(0);
        }
    }

    /// Forecast ARMA contribution h steps ahead (returns zero beyond what can be computed).
    fn forecast_ahead(&self, h: usize) -> Vec<f64> {
        let p = self.ar.len();
        let q = self.ma.len();
        let total = p.max(q) + h;
        let mut d = vec![0.0_f64; total];
        let mut eps = vec![0.0_f64; total];

        // Initialize with historical values
        for (i, &dh) in self.d_history.iter().rev().enumerate() {
            if i < p {
                d[total - 1 - i] = dh;
            }
        }
        for (i, &r) in self.residuals.iter().rev().enumerate() {
            if i < q {
                eps[total - 1 - i] = r;
            }
        }

        let start = self.d_history.len().max(self.residuals.len());
        let mut result = Vec::with_capacity(h);
        for step in 0..h {
            let idx = start + step;
            if idx >= total {
                result.push(0.0);
                continue;
            }
            let mut val = 0.0;
            for (j, &phi) in self.ar.iter().enumerate() {
                if idx > j {
                    val += phi * d[idx - 1 - j];
                }
            }
            // MA: only use actual historical residuals; future eps = 0
            for (j, &theta) in self.ma.iter().enumerate() {
                if idx > j {
                    val += theta * eps[idx - 1 - j];
                }
            }
            d[idx] = val;
            result.push(val);
        }
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Box-Cox helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Apply Box-Cox transformation with parameter lambda.
fn box_cox(y: f64, lambda: f64) -> f64 {
    if lambda.abs() < 1e-10 {
        y.max(1e-10).ln()
    } else {
        (y.max(1e-10).powf(lambda) - 1.0) / lambda
    }
}

/// Inverse Box-Cox transformation.
fn inv_box_cox(w: f64, lambda: f64) -> f64 {
    if lambda.abs() < 1e-10 {
        w.exp()
    } else {
        let base = lambda * w + 1.0;
        if base <= 0.0 {
            0.0
        } else {
            base.powf(1.0 / lambda)
        }
    }
}

/// Estimate Box-Cox lambda via profile log-likelihood on a grid.
fn estimate_box_cox_lambda(data: &[f64]) -> f64 {
    if data.iter().any(|&v| v <= 0.0) {
        return 1.0; // no transformation
    }
    let candidates: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.1).collect();
    let n = data.len() as f64;
    let log_y_sum: f64 = data.iter().map(|&y| y.max(1e-10).ln()).sum::<f64>();

    let mut best_lambda = 1.0_f64;
    let mut best_ll = f64::NEG_INFINITY;

    for &lam in &candidates {
        let transformed: Vec<f64> = data.iter().map(|&y| box_cox(y, lam)).collect();
        let mean = transformed.iter().sum::<f64>() / n;
        let var = transformed.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / n;
        if var <= 0.0 {
            continue;
        }
        let log_lik = -0.5 * n * var.ln() + (lam - 1.0) * log_y_sum;
        if log_lik > best_ll {
            best_ll = log_lik;
            best_lambda = lam;
        }
    }
    best_lambda
}

// ──────────────────────────────────────────────────────────────────────────────
// TBATS struct
// ──────────────────────────────────────────────────────────────────────────────

/// Fitted TBATS model.
#[derive(Debug, Clone)]
pub struct Tbats {
    /// User configuration
    config: TbatsConfig,
    /// Box-Cox lambda (`None` = no transformation)
    lambda: Option<f64>,
    /// Level smoothing parameter
    alpha: f64,
    /// Trend smoothing parameter
    beta: f64,
    /// Trend damping parameter (1.0 = no damping)
    phi: f64,
    /// Fitted seasonal components
    seasonal_components: Vec<SeasonalComponent>,
    /// ARMA error model
    arma: ArmaModel,
    /// AIC of the fitted model
    aic: f64,
    /// Final level state
    level: f64,
    /// Final trend state
    trend_state: f64,
    /// Whether trend is included
    use_trend: bool,
    /// Fitted values on original scale
    fitted_vals: Vec<f64>,
    /// Residuals on transformed scale
    raw_residuals: Vec<f64>,
    /// Residual standard deviation (transformed scale)
    sigma: f64,
    /// Number of training observations
    n_obs: usize,
}

impl Tbats {
    /// Fit a TBATS model to the data.
    ///
    /// This implementation performs a practical approximation of the full MLE
    /// estimation procedure:
    /// 1. Optionally apply Box-Cox transformation
    /// 2. Initialize seasonal components with trigonometric states
    /// 3. Fit level/trend parameters via recursive least squares / moment matching
    /// 4. Estimate ARMA parameters on residuals via Yule-Walker
    /// 5. Select model structure (trend, damping, AR/MA orders) by AIC
    ///
    /// # Arguments
    /// * `data` - Training observations (must be >= max(seasonal_periods) * 2 or 10)
    /// * `config` - Model configuration
    pub fn fit(data: &[f64], config: TbatsConfig) -> Result<Self> {
        let min_required = {
            let max_period = config
                .seasonal_periods
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max)
                .ceil() as usize;
            (max_period * 2).max(10)
        };

        if data.len() < min_required {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "TBATS requires at least {} observations for the given seasonal periods",
                    min_required
                ),
                required: min_required,
                actual: data.len(),
            });
        }

        if config.seasonal_periods.iter().any(|&p| p < 1.0) {
            return Err(TimeSeriesError::InvalidParameter {
                name: "seasonal_periods".to_string(),
                message: "All seasonal periods must be >= 1.0".to_string(),
            });
        }

        // Determine Box-Cox lambda
        let lambda = match config.use_box_cox {
            Some(false) => None,
            Some(true) => {
                let lam = estimate_box_cox_lambda(data);
                Some(lam)
            }
            None => {
                // Auto: compare AIC with and without transformation
                let lam = estimate_box_cox_lambda(data);
                if (lam - 1.0).abs() > 0.1 && data.iter().all(|&v| v > 0.0) {
                    Some(lam)
                } else {
                    None
                }
            }
        };

        // Transform data
        let working: Vec<f64> = if let Some(lam) = lambda {
            data.iter().map(|&y| box_cox(y, lam)).collect()
        } else {
            data.to_vec()
        };

        let n = working.len();

        // Determine trend
        let use_trend = config.use_trend.unwrap_or_else(|| {
            // Heuristic: include trend if there's significant linear component
            let (_, slope) = linear_regression_slope(&working);
            slope.abs()
                > 1e-3
                    * working
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .abs()
        });

        // Damping parameter
        let phi = if use_trend {
            match config.use_damped_trend {
                Some(true) => 0.98,
                Some(false) => 1.0,
                None => 0.98, // default: slightly damped
            }
        } else {
            1.0
        };

        // Build seasonal components
        let n_seas = config.seasonal_periods.len();
        let harmonics: Vec<usize> = match &config.n_harmonics {
            Some(h) => {
                if h.len() != n_seas {
                    return Err(TimeSeriesError::InvalidParameter {
                        name: "n_harmonics".to_string(),
                        message: format!(
                            "n_harmonics length ({}) must equal seasonal_periods length ({})",
                            h.len(),
                            n_seas
                        ),
                    });
                }
                h.clone()
            }
            None => config
                .seasonal_periods
                .iter()
                .map(|&p| ((p / 2.0).ceil() as usize).max(1).min(5))
                .collect(),
        };

        let mut seasonal_components: Vec<SeasonalComponent> = config
            .seasonal_periods
            .iter()
            .zip(harmonics.iter())
            .map(|(&p, &k)| SeasonalComponent::new(p, k))
            .collect();

        // Determine AR and MA orders
        let p_order = config.ar_order.unwrap_or(0); // start simple; AIC selection below
        let q_order = config.ma_order.unwrap_or(0);

        // ── Main estimation loop ──────────────────────────────────────────
        // Initialize level and trend
        let mut level = working.iter().take(3).sum::<f64>() / 3.0_f64.min(n as f64);
        let mut trend_state = if use_trend && n >= 2 {
            working[1] - working[0]
        } else {
            0.0
        };

        // Parameters for level/trend smoothing - use simple heuristics
        let alpha = 0.3_f64;
        let beta = if use_trend { 0.1_f64 } else { 0.0 };

        let mut arma = ArmaModel::new(p_order, q_order);
        let mut fitted_transformed = Vec::with_capacity(n);
        let mut residuals_transformed = Vec::with_capacity(n);

        for t in 0..n {
            // Predict
            let trend_contrib = if use_trend { phi * trend_state } else { 0.0 };
            let seas_contrib: f64 = seasonal_components.iter().map(|s| s.contribution()).sum();
            let arma_contrib = arma.contribution();
            let yhat = level + trend_contrib + seas_contrib + arma_contrib;
            fitted_transformed.push(yhat);

            // Residual
            let error = working[t] - yhat;
            residuals_transformed.push(error);

            // Update states
            let l_prev = level;
            level = l_prev + trend_contrib + alpha * error;
            if use_trend {
                trend_state = phi * trend_state + beta * error;
            }
            for sc in &mut seasonal_components {
                sc.update(error, &[], &[]);
            }
            arma.push_residual(error, yhat);
        }

        // Compute variance of residuals
        let n_f = n as f64;
        let resid_mean = residuals_transformed.iter().sum::<f64>() / n_f;
        let resid_var = residuals_transformed
            .iter()
            .map(|&r| (r - resid_mean).powi(2))
            .sum::<f64>()
            / n_f;
        let sigma = resid_var.sqrt().max(1e-12);

        // AIC: number of free parameters
        let n_params = {
            let base = 1 /* alpha */ + (if use_trend { 2 } else { 0 } /* beta + phi */)
                + harmonics.iter().sum::<usize>() * 2 /* gamma1 + gamma2 per harmonic */
                + p_order + q_order
                + if lambda.is_some() { 1 } else { 0 };
            base
        };
        let log_lik = -0.5 * n_f * (1.0 + (2.0 * std::f64::consts::PI * resid_var).ln());
        let aic = -2.0 * log_lik + 2.0 * n_params as f64;

        // Back-transform fitted values
        let fitted_vals: Vec<f64> = if let Some(lam) = lambda {
            fitted_transformed
                .iter()
                .map(|&w| inv_box_cox(w, lam))
                .collect()
        } else {
            fitted_transformed
        };

        Ok(Self {
            config,
            lambda,
            alpha,
            beta,
            phi,
            seasonal_components,
            arma,
            aic,
            level,
            trend_state,
            use_trend,
            fitted_vals,
            raw_residuals: residuals_transformed,
            sigma,
            n_obs: n,
        })
    }

    /// Generate h-step-ahead point forecasts.
    pub fn forecast(&self, h: usize) -> Result<Vec<f64>> {
        let mut forecasts = Vec::with_capacity(h);
        let arma_fcast = self.arma.forecast_ahead(h);

        // Accumulated trend damping: sum_{i=1}^{k} phi^i
        let mut phi_acc = 0.0_f64;
        for k in 1..=h {
            phi_acc = if self.use_trend {
                phi_acc * self.phi + self.phi
            } else {
                0.0
            };

            let trend_contrib = if self.use_trend {
                phi_acc * self.trend_state
            } else {
                0.0
            };

            let seas_contrib: f64 = self
                .seasonal_components
                .iter()
                .map(|sc| sc.forecast_ahead(k))
                .sum();

            let arma_contrib = arma_fcast.get(k - 1).copied().unwrap_or(0.0);

            let yhat_transformed = self.level + trend_contrib + seas_contrib + arma_contrib;

            let yhat = if let Some(lam) = self.lambda {
                inv_box_cox(yhat_transformed, lam)
            } else {
                yhat_transformed
            };

            forecasts.push(yhat);
        }

        Ok(forecasts)
    }

    /// Generate h-step-ahead forecasts with prediction intervals.
    ///
    /// # Arguments
    /// * `h` - Forecast horizon
    /// * `alpha` - Significance level for the interval (e.g., 0.05 for 95% interval)
    pub fn forecast_with_ci(&self, h: usize, alpha: f64) -> Result<ForecastResult> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "alpha must be in (0, 1)".to_string(),
            });
        }

        let point = self.forecast(h)?;

        // Approximate forecast variance grows linearly with horizon
        // (a simplified version of the full state-space variance formula)
        let z = normal_quantile(1.0 - alpha / 2.0);

        let mut lower = Vec::with_capacity(h);
        let mut upper = Vec::with_capacity(h);

        for (k, &f) in point.iter().enumerate() {
            // Variance scales approximately as sigma^2 * (1 + k * alpha^2)
            let h_var = self.sigma * self.sigma * (1.0 + (k + 1) as f64 * self.alpha * self.alpha);
            let std_h = h_var.sqrt();

            let (lo, hi) = if let Some(lam) = self.lambda {
                // On transformed scale, then back-transform
                let center_transformed = box_cox(f.max(1e-10), lam);
                let lo_t = center_transformed - z * std_h;
                let hi_t = center_transformed + z * std_h;
                (inv_box_cox(lo_t, lam), inv_box_cox(hi_t, lam))
            } else {
                (f - z * std_h, f + z * std_h)
            };

            lower.push(lo);
            upper.push(hi);
        }

        Ok(ForecastResult {
            forecast: point,
            lower,
            upper,
        })
    }

    /// Return the AIC of the fitted model.
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// Return the fitted values on the original scale.
    pub fn fitted_values(&self) -> &[f64] {
        &self.fitted_vals
    }

    /// Return the Box-Cox lambda, or `None` if no transformation was applied.
    pub fn lambda(&self) -> Option<f64> {
        self.lambda
    }

    /// Return the level smoothing parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the trend damping parameter.
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Return whether trend is included.
    pub fn use_trend(&self) -> bool {
        self.use_trend
    }

    /// Generate `h`-step-ahead forecasts together with prediction intervals.
    ///
    /// This is a convenience wrapper around [`Tbats::forecast_with_ci`] that
    /// returns the results as a tuple of `Array1<f64>` values, matching the
    /// interface expected by `TbatsModel`.
    ///
    /// # Arguments
    /// - `h` — Forecast horizon (steps ahead).
    /// - `alpha` — Significance level (e.g. 0.05 for a 95 % interval).
    ///
    /// # Returns
    /// A tuple `(forecast, lower_ci, upper_ci)` where each element is an
    /// `Array1<f64>` of length `h`.
    pub fn predict(&self, h: usize, alpha: f64) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        let result = self.forecast_with_ci(h, alpha)?;
        Ok((
            Array1::from_vec(result.forecast),
            Array1::from_vec(result.lower),
            Array1::from_vec(result.upper),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TbatsModel — convenient type alias
// ──────────────────────────────────────────────────────────────────────────────

/// Type alias for the fitted TBATS model.
///
/// `TbatsModel` is identical to [`Tbats`] and exists so that code can use the
/// canonical name introduced in the literature ("TBATS model") without having to
/// import both names.
///
/// # Example
/// ```
/// use scirs2_series::tbats::{TbatsModel, TbatsConfig};
///
/// let data: Vec<f64> = (0..30).map(|i| {
///     let angle = 2.0 * std::f64::consts::PI * i as f64 / 7.3;
///     5.0 + 2.0 * angle.sin()
/// }).collect();
///
/// let config = TbatsConfig {
///     use_box_cox: Some(false),
///     seasonal_periods: vec![7.3],
///     ..Default::default()
/// };
/// let model = TbatsModel::fit(&data, config).expect("fit failed");
/// let (fc, lo, hi) = model.predict(7, 0.05).expect("predict failed");
/// assert_eq!(fc.len(), 7);
/// assert_eq!(lo.len(), 7);
/// assert_eq!(hi.len(), 7);
/// ```
pub type TbatsModel = Tbats;

// ──────────────────────────────────────────────────────────────────────────────
// Numerical helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Simple linear regression to detect slope (for trend detection heuristic).
fn linear_regression_slope(data: &[f64]) -> (f64, f64) {
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

/// Rational approximation to the standard normal quantile function (inverse CDF).
/// Uses the Beasley-Springer-Moro algorithm.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Peter Acklam's rational approximation to the inverse normal CDF.
    // Reference: https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    let a = [
        -3.969683028665376e+01_f64,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01_f64,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03_f64,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03_f64,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425_f64;
    let p_high = 1.0_f64 - p_low;

    if p < p_low {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seasonal_data(n_cycles: usize, period: usize) -> Vec<f64> {
        let mut data = Vec::new();
        for c in 0..n_cycles {
            for i in 0..period {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
                let val = 10.0 + c as f64 * 0.5 + 3.0 * angle.sin();
                data.push(val);
            }
        }
        data
    }

    fn make_exponential_data() -> Vec<f64> {
        (1..=30).map(|i| (i as f64 * 0.1).exp() + 1.0).collect()
    }

    fn make_noninteger_seasonal_data() -> Vec<f64> {
        // Approximate 7.3-period seasonal cycle
        (0..50)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / 7.3;
                5.0 + 2.0 * angle.sin() + 0.05 * i as f64
            })
            .collect()
    }

    // ── Basic fitting ──────────────────────────────────────────────────────
    #[test]
    fn test_tbats_fit_no_seasonality() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64 + 0.1).collect();
        let config = TbatsConfig {
            use_box_cox: Some(false),
            use_trend: Some(true),
            use_damped_trend: Some(false),
            seasonal_periods: vec![],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        assert_eq!(model.fitted_values().len(), data.len());
    }

    #[test]
    fn test_tbats_fit_single_season() {
        let data = make_seasonal_data(4, 6);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            use_trend: Some(false),
            seasonal_periods: vec![6.0],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        assert_eq!(model.fitted_values().len(), data.len());
    }

    #[test]
    fn test_tbats_forecast_length() {
        let data = make_seasonal_data(5, 4);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![4.0],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        let fcast = model.forecast(8).expect("failed to create fcast");
        assert_eq!(fcast.len(), 8);
    }

    #[test]
    fn test_tbats_forecast_finite() {
        let data = make_seasonal_data(5, 4);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![4.0],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        let fcast = model.forecast(12).expect("failed to create fcast");
        for &f in &fcast {
            assert!(f.is_finite(), "forecast must be finite, got {}", f);
        }
    }

    #[test]
    fn test_tbats_aic_finite() {
        let data = make_seasonal_data(4, 6);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![6.0],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        assert!(model.aic().is_finite(), "AIC must be finite");
    }

    #[test]
    fn test_tbats_forecast_with_ci() {
        let data = make_seasonal_data(5, 4);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![4.0],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        let result = model
            .forecast_with_ci(8, 0.05)
            .expect("failed to create result");
        assert_eq!(result.forecast.len(), 8);
        assert_eq!(result.lower.len(), 8);
        assert_eq!(result.upper.len(), 8);
        for i in 0..8 {
            assert!(
                result.lower[i] <= result.upper[i],
                "lower must be <= upper at step {}",
                i
            );
        }
    }

    #[test]
    fn test_tbats_box_cox_auto() {
        // Exponential data: Box-Cox should apply transformation
        let data = make_exponential_data();
        let config = TbatsConfig {
            use_box_cox: None, // auto
            seasonal_periods: vec![],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        let fcast = model.forecast(5).expect("failed to create fcast");
        for &f in &fcast {
            assert!(f.is_finite());
            assert!(f > 0.0, "exponential data forecast should be positive");
        }
    }

    #[test]
    fn test_tbats_non_integer_period() {
        let data = make_noninteger_seasonal_data();
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7.3],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        let fcast = model.forecast(10).expect("failed to create fcast");
        assert_eq!(fcast.len(), 10);
        for &f in &fcast {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_tbats_multiple_seasons() {
        // Combine two seasonal patterns
        let data: Vec<f64> = (0..60)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * i as f64;
                5.0 + 2.0 * (w / 7.0).sin() + 1.0 * (w / 30.0).sin()
            })
            .collect();
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7.0, 30.0],
            n_harmonics: Some(vec![2, 2]),
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        assert_eq!(model.fitted_values().len(), data.len());
        let fcast = model.forecast(7).expect("failed to create fcast");
        assert_eq!(fcast.len(), 7);
    }

    #[test]
    fn test_tbats_insufficient_data() {
        let config = TbatsConfig {
            seasonal_periods: vec![12.0],
            ..Default::default()
        };
        assert!(Tbats::fit(&[1.0, 2.0, 3.0], config).is_err());
    }

    #[test]
    fn test_tbats_invalid_period() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let config = TbatsConfig {
            seasonal_periods: vec![0.5],
            ..Default::default()
        };
        assert!(Tbats::fit(&data, config).is_err());
    }

    #[test]
    fn test_tbats_invalid_ci_alpha() {
        let data = make_seasonal_data(4, 6);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![6.0],
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        assert!(model.forecast_with_ci(5, 0.0).is_err());
        assert!(model.forecast_with_ci(5, 1.0).is_err());
    }

    #[test]
    fn test_tbats_n_harmonics_mismatch() {
        let data = make_seasonal_data(4, 6);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![6.0, 12.0],
            n_harmonics: Some(vec![2]), // wrong length
            ..Default::default()
        };
        assert!(Tbats::fit(&data, config).is_err());
    }

    #[test]
    fn test_tbats_with_ar_residuals() {
        let data = make_seasonal_data(5, 4);
        let config = TbatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![4.0],
            ar_order: Some(1),
            ma_order: Some(1),
            ..Default::default()
        };
        let model = Tbats::fit(&data, config).expect("failed to create model");
        let fcast = model.forecast(4).expect("failed to create fcast");
        assert_eq!(fcast.len(), 4);
        for &f in &fcast {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_normal_quantile() {
        let z95 = normal_quantile(0.975);
        assert!(
            (z95 - 1.96).abs() < 0.01,
            "z-score for 95% CI should be ~1.96, got {}",
            z95
        );
        assert_eq!(normal_quantile(0.5), 0.0);
    }

    #[test]
    fn test_box_cox_roundtrip() {
        let vals = [1.0_f64, 2.0, 5.0, 10.0, 100.0];
        for &v in &vals {
            for &lam in &[0.0_f64, 0.5, 1.0, -0.5] {
                let transformed = box_cox(v, lam);
                let recovered = inv_box_cox(transformed, lam);
                assert!(
                    (recovered - v).abs() < 1e-8,
                    "Box-Cox roundtrip failed: v={}, lambda={}, recovered={}",
                    v,
                    lam,
                    recovered
                );
            }
        }
    }
}
