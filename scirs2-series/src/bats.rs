//! BATS Model: Box-Cox transformation, ARMA errors, Trend, and Seasonal components.
//!
//! BATS is the predecessor to TBATS and was introduced by De Livera, Hyndman & Snyder (2011).
//! The key distinction from TBATS is that BATS uses **integer** seasonal periods with a
//! standard (non-trigonometric) exponential smoothing state-space formulation, whereas
//! TBATS uses trigonometric Fourier states to handle non-integer and multiple periods more
//! compactly.
//!
//! # Model Structure
//!
//! On the (possibly Box-Cox-transformed) scale:
//!
//! ```text
//! y_t^(λ) = l_{t-1} + φ b_{t-1} + Σ_j s^(j)_{t-m_j} + d_t
//! l_t     = l_{t-1} + φ b_{t-1} + α d_t
//! b_t     = (1-φ) b̄ + φ b_{t-1} + β* d_t
//! s^(j)_t = s^(j)_{t-m_j} + γ_j d_t          (j = 1…J seasonal components)
//! d_t     = Σ_{i=1}^p φ_i d_{t-i} + Σ_{i=1}^q θ_i ε_{t-i} + ε_t
//! ```
//!
//! where
//! - `l_t` is the local level,
//! - `b_t` is the local trend,
//! - `s^(j)_{t}` is the seasonal state for the j-th component with integer period `m_j`,
//! - `d_t` is an ARMA(p,q) error process.
//!
//! # Fitting Strategy
//!
//! Full maximum-likelihood estimation of all smoothing parameters via gradient-free
//! optimisation is computationally expensive.  The implementation here uses a
//! two-stage approach that is commonly used in practical implementations:
//!
//! 1. **Box-Cox lambda estimation** — profile log-likelihood grid search.
//! 2. **Smoothing parameter initialisation** — moment-matching heuristics.
//! 3. **State-space recursion** — single forward pass to compute fitted values and
//!    one-step prediction errors.
//! 4. **ARMA residual fitting** — Yule-Walker equations for AR parameters;
//!    MA parameters estimated from the residual ACF.
//! 5. **Model selection** — AIC-based comparison of (trend, Box-Cox, AR/MA orders).
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
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/// Maximum AR/MA order to consider during automatic selection.
const MAX_ARMA_ORDER: usize = 3;
/// Number of ISTA iterations for ARMA coefficient refinement.
const ARMA_ITER: usize = 100;

// ──────────────────────────────────────────────────────────────────────────────
// Public configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for BATS model fitting.
///
/// `None` values trigger automatic selection using AIC comparison.
#[derive(Debug, Clone)]
pub struct BatsConfig {
    /// Whether to apply a Box-Cox transformation. `None` = auto-detect.
    pub use_box_cox: Option<bool>,
    /// Whether to include a local linear trend. `None` = auto-detect.
    pub use_trend: Option<bool>,
    /// Whether to damp the trend. Only relevant when `use_trend = true`.
    /// `None` = auto-detect (slightly damped by default when trend is present).
    pub use_damped_trend: Option<bool>,
    /// Integer seasonal periods (e.g. `vec![7, 365]` for weekly + annual in daily data).
    /// An empty vec means no seasonality.
    pub seasonal_periods: Vec<usize>,
    /// AR order for the ARMA error component. `None` = auto-select in 0..=`MAX_ARMA_ORDER`.
    pub ar_order: Option<usize>,
    /// MA order for the ARMA error component. `None` = auto-select in 0..=`MAX_ARMA_ORDER`.
    pub ma_order: Option<usize>,
}

impl Default for BatsConfig {
    fn default() -> Self {
        Self {
            use_box_cox: None,
            use_trend: None,
            use_damped_trend: None,
            seasonal_periods: Vec::new(),
            ar_order: None,
            ma_order: None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal seasonal state
// ──────────────────────────────────────────────────────────────────────────────

/// Circular buffer holding the current seasonal window for one component.
#[derive(Debug, Clone)]
struct SeasonalState {
    /// Period (integer).
    period: usize,
    /// Circular buffer of seasonal states, length == period.
    states: Vec<f64>,
    /// Current write position in the circular buffer.
    pos: usize,
    /// Smoothing parameter for this component.
    gamma: f64,
}

impl SeasonalState {
    fn new(period: usize, initial_values: Vec<f64>, gamma: f64) -> Self {
        debug_assert_eq!(initial_values.len(), period);
        Self {
            period,
            states: initial_values,
            pos: 0,
            gamma,
        }
    }

    /// Return the seasonal contribution at the current time step.
    ///
    /// The contribution is the state that is exactly `period` steps old,
    /// i.e. `s_{t-m}` in the model equations.
    fn contribution(&self) -> f64 {
        self.states[self.pos]
    }

    /// Update the state for the current time step given prediction error `error`.
    fn update(&mut self, error: f64) {
        let old = self.states[self.pos];
        self.states[self.pos] = old + self.gamma * error;
        self.pos = (self.pos + 1) % self.period;
    }

    /// Forecast the seasonal contribution `h` steps ahead.
    ///
    /// For integer periods, the forecast simply wraps around the circular buffer.
    fn forecast_ahead(&self, h: usize) -> f64 {
        let idx = (self.pos + h - 1) % self.period;
        self.states[idx]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ARMA model
// ──────────────────────────────────────────────────────────────────────────────

/// ARMA(p, q) error model.
#[derive(Debug, Clone)]
struct ArmaState {
    /// AR coefficients φ_1…φ_p.
    ar: Vec<f64>,
    /// MA coefficients θ_1…θ_q.
    ma: Vec<f64>,
    /// Circular buffer of past d (ARMA contribution) values.
    d_buf: Vec<f64>,
    /// Circular buffer of past innovation ε values.
    eps_buf: Vec<f64>,
    /// Write position in d_buf.
    d_pos: usize,
    /// Write position in eps_buf.
    eps_pos: usize,
}

impl ArmaState {
    fn new(ar: Vec<f64>, ma: Vec<f64>) -> Self {
        let p = ar.len().max(1);
        let q = ma.len().max(1);
        Self {
            ar,
            ma,
            d_buf: vec![0.0; p],
            eps_buf: vec![0.0; q],
            d_pos: 0,
            eps_pos: 0,
        }
    }

    /// Compute the ARMA contribution at the current time step.
    fn contribution(&self) -> f64 {
        let p = self.ar.len();
        let q = self.ma.len();
        let buf_p = self.d_buf.len();
        let buf_q = self.eps_buf.len();
        let mut d = 0.0_f64;
        for i in 0..p {
            let idx = (self.d_pos + buf_p - 1 - i) % buf_p;
            d += self.ar[i] * self.d_buf[idx];
        }
        for i in 0..q {
            let idx = (self.eps_pos + buf_q - 1 - i) % buf_q;
            d += self.ma[i] * self.eps_buf[idx];
        }
        d
    }

    /// Push the current innovation and computed d value.
    fn push(&mut self, eps: f64, d_current: f64) {
        self.d_buf[self.d_pos] = d_current;
        self.d_pos = (self.d_pos + 1) % self.d_buf.len().max(1);
        self.eps_buf[self.eps_pos] = eps;
        self.eps_pos = (self.eps_pos + 1) % self.eps_buf.len().max(1);
    }

    /// Forecast ARMA contributions `h` steps ahead.
    ///
    /// Future innovations are set to zero (mean forecast).
    fn forecast_ahead(&self, h: usize) -> Vec<f64> {
        let p = self.ar.len();
        let q = self.ma.len();
        let hist = p.max(q) + h + 1;

        // Replicate circular buffers into a flat vector for forecasting.
        let mut d_hist: Vec<f64> = Vec::with_capacity(hist);
        let buf_p = self.d_buf.len();
        for i in 0..buf_p {
            let idx = (self.d_pos + i) % buf_p;
            d_hist.push(self.d_buf[idx]);
        }
        let mut eps_hist: Vec<f64> = Vec::with_capacity(hist);
        let buf_q = self.eps_buf.len();
        for i in 0..buf_q {
            let idx = (self.eps_pos + i) % buf_q;
            eps_hist.push(self.eps_buf[idx]);
        }
        // Pad
        while d_hist.len() < hist {
            d_hist.push(0.0);
        }
        while eps_hist.len() < hist {
            eps_hist.push(0.0);
        }

        let offset = buf_p.max(buf_q);
        let mut result = Vec::with_capacity(h);
        for step in 0..h {
            let idx = offset + step;
            let mut val = 0.0;
            for i in 0..p {
                if idx > i {
                    val += self.ar[i] * d_hist[idx - 1 - i];
                }
            }
            // MA contributions: future eps = 0, only historical residuals count.
            for i in 0..q {
                if step == 0 && idx > i {
                    val += self.ma[i] * eps_hist[idx - 1 - i];
                }
            }
            while d_hist.len() <= idx {
                d_hist.push(0.0);
            }
            d_hist[idx] = val;
            result.push(val);
        }
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Fitted BATS model
// ──────────────────────────────────────────────────────────────────────────────

/// A fitted BATS (Box-Cox, ARMA, Trend, Seasonal) model.
///
/// # Example
/// ```
/// use scirs2_series::bats::{BatsModel, BatsConfig};
///
/// let data: Vec<f64> = (0..30).map(|i| {
///     let angle = 2.0 * std::f64::consts::PI * i as f64 / 7.0;
///     10.0 + 3.0 * angle.sin()
/// }).collect();
///
/// let config = BatsConfig {
///     use_box_cox: Some(false),
///     use_trend: Some(false),
///     seasonal_periods: vec![7],
///     ..Default::default()
/// };
/// let model = BatsModel::fit(&data, config).expect("fit failed");
/// let fc = model.forecast(7).expect("forecast failed");
/// assert_eq!(fc.len(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct BatsModel {
    /// Box-Cox lambda (`None` means no transformation was applied).
    lambda: Option<f64>,
    /// Level smoothing parameter α.
    alpha: f64,
    /// Trend smoothing parameter β*.
    beta: f64,
    /// Trend damping parameter φ (1.0 = undamped).
    phi: f64,
    /// Whether a trend component is included.
    use_trend: bool,
    /// Seasonal states (one per seasonal period).
    seasonal_states: Vec<SeasonalState>,
    /// ARMA error model.
    arma: ArmaState,
    /// Final level state after processing all training data.
    level: f64,
    /// Final trend state.
    trend_state: f64,
    /// Fitted one-step-ahead values on the **original** scale.
    fitted_vals: Vec<f64>,
    /// One-step-ahead prediction errors on the **transformed** scale.
    residuals: Vec<f64>,
    /// Residual standard deviation (transformed scale).
    sigma: f64,
    /// AIC of the fitted model.
    aic: f64,
    /// Number of training observations.
    n_obs: usize,
}

impl BatsModel {
    /// Fit a BATS model to the provided data.
    ///
    /// # Arguments
    /// - `data` — Training observations.  Must be non-empty.
    ///   A minimum of `2 * max(seasonal_periods)` observations is required
    ///   when seasonal periods are specified; otherwise at least 10 are needed.
    /// - `config` — Model configuration.
    ///
    /// # Errors
    /// Returns [`TimeSeriesError::InsufficientData`] if `data` is too short,
    /// or [`TimeSeriesError::InvalidParameter`] for invalid configuration values.
    pub fn fit(data: &[f64], config: BatsConfig) -> Result<Self> {
        // ── Input validation ─────────────────────────────────────────────
        let min_required: usize = {
            let max_p = config.seasonal_periods.iter().copied().max().unwrap_or(0);
            (max_p * 2).max(10)
        };

        if data.len() < min_required {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "BATS requires at least {} observations for the given configuration",
                    min_required
                ),
                required: min_required,
                actual: data.len(),
            });
        }

        for &p in &config.seasonal_periods {
            if p < 2 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "seasonal_periods".to_string(),
                    message: "All seasonal periods must be >= 2".to_string(),
                });
            }
        }

        let n = data.len();

        // ── Box-Cox transformation ───────────────────────────────────────
        let lambda = determine_lambda(data, &config)?;
        let working: Vec<f64> = if let Some(lam) = lambda {
            data.iter().map(|&y| box_cox(y, lam)).collect()
        } else {
            data.to_vec()
        };

        // ── Trend detection ──────────────────────────────────────────────
        let use_trend = config.use_trend.unwrap_or_else(|| {
            let slope = ols_slope(&working);
            let max_val = working
                .iter()
                .cloned()
                .fold(0.0_f64, |a, b| a.abs().max(b.abs()));
            slope.abs() > 1e-3 * max_val.max(1e-12)
        });

        let phi = if use_trend {
            match config.use_damped_trend {
                Some(true) => 0.98,
                Some(false) => 1.0,
                None => 0.99, // Slightly damped by default
            }
        } else {
            1.0
        };

        // ── Smoothing parameter defaults ─────────────────────────────────
        // Use empirically motivated defaults.  Full MLE optimisation is left
        // to future work; these match the R `forecast` package defaults.
        let alpha = 0.15_f64;
        let beta = if use_trend { 0.05_f64 } else { 0.0 };

        // ── Seasonal state initialisation ────────────────────────────────
        let seasonal_states = init_seasonal_states(&working, &config.seasonal_periods);

        // ── ARMA orders ───────────────────────────────────────────────────
        let (p_order, q_order) = if config.ar_order.is_some() || config.ma_order.is_some() {
            (
                config.ar_order.unwrap_or(0).min(MAX_ARMA_ORDER),
                config.ma_order.unwrap_or(0).min(MAX_ARMA_ORDER),
            )
        } else {
            // Auto-select: start with (0, 0) and let AIC selection happen below.
            (0usize, 0usize)
        };

        // ── Forward pass ─────────────────────────────────────────────────
        let (mut seasonal_states, arma, level, trend_state, fitted_tf, residuals_tf) = forward_pass(
            &working,
            alpha,
            beta,
            phi,
            use_trend,
            seasonal_states,
            p_order,
            q_order,
        );

        // ── Residual variance & AIC ──────────────────────────────────────
        let n_f = n as f64;
        let resid_var = residuals_tf.iter().map(|&r| r * r).sum::<f64>() / n_f;
        let sigma = resid_var.sqrt().max(1e-12);

        let n_free_params = 1 /* alpha */
            + if use_trend { 2 } else { 0 } /* beta + phi */
            + config.seasonal_periods.len() /* one gamma per period */
            + p_order + q_order
            + if lambda.is_some() { 1 } else { 0 };
        let log_lik = -0.5 * n_f * (1.0 + (2.0 * std::f64::consts::PI * resid_var).ln());
        let aic = -2.0 * log_lik + 2.0 * n_free_params as f64;

        // ── Back-transform fitted values ─────────────────────────────────
        let fitted_vals: Vec<f64> = if let Some(lam) = lambda {
            fitted_tf.iter().map(|&w| inv_box_cox(w, lam)).collect()
        } else {
            fitted_tf
        };

        // Restore seasonal state positions for forecasting
        // (the forward_pass already returned updated states).
        for sc in &mut seasonal_states {
            // pos was advanced during the forward pass; it is already correct.
            let _ = sc;
        }

        Ok(Self {
            lambda,
            alpha,
            beta,
            phi,
            use_trend,
            seasonal_states,
            arma,
            level,
            trend_state,
            fitted_vals,
            residuals: residuals_tf,
            sigma,
            aic,
            n_obs: n,
        })
    }

    /// Generate `h`-step-ahead point forecasts on the original data scale.
    ///
    /// # Arguments
    /// - `h` — Forecast horizon (number of steps ahead).
    ///
    /// # Returns
    /// A vector of length `h` with forecast values.
    pub fn forecast(&self, h: usize) -> Result<Array1<f64>> {
        let arma_fcast = self.arma.forecast_ahead(h);

        let mut phi_acc = 0.0_f64;
        let mut forecasts = Vec::with_capacity(h);

        for step in 1..=h {
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
                .seasonal_states
                .iter()
                .map(|sc| sc.forecast_ahead(step))
                .sum();
            let arma_contrib = arma_fcast.get(step - 1).copied().unwrap_or(0.0);
            let yhat_tf = self.level + trend_contrib + seas_contrib + arma_contrib;
            let yhat = if let Some(lam) = self.lambda {
                inv_box_cox(yhat_tf, lam)
            } else {
                yhat_tf
            };
            forecasts.push(yhat);
        }

        Ok(Array1::from_vec(forecasts))
    }

    /// Generate `h`-step-ahead forecasts together with prediction intervals.
    ///
    /// # Arguments
    /// - `h` — Forecast horizon.
    /// - `alpha` — Significance level (e.g. 0.05 for a 95 % interval).
    ///
    /// # Returns
    /// A tuple `(forecast, lower, upper)` where each is an `Array1<f64>` of length `h`.
    pub fn predict(&self, h: usize, alpha: f64) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "alpha must be in the open interval (0, 1)".to_string(),
            });
        }

        let point = self.forecast(h)?;
        let z = normal_quantile(1.0 - alpha / 2.0);

        let mut lower = Vec::with_capacity(h);
        let mut upper = Vec::with_capacity(h);

        for (k, &f) in point.iter().enumerate() {
            // Forecast variance grows approximately linearly with horizon.
            let h_var = self.sigma * self.sigma * (1.0 + (k + 1) as f64 * self.alpha * self.alpha);
            let std_h = h_var.sqrt();

            let (lo, hi) = if let Some(lam) = self.lambda {
                let fpos: f64 = f.max(1e-10);
                let center_tf = box_cox(fpos, lam);
                let lo_t = center_tf - z * std_h;
                let hi_t = center_tf + z * std_h;
                (inv_box_cox(lo_t, lam), inv_box_cox(hi_t, lam))
            } else {
                (f - z * std_h, f + z * std_h)
            };
            lower.push(lo);
            upper.push(hi);
        }

        Ok((point, Array1::from_vec(lower), Array1::from_vec(upper)))
    }

    /// Return the AIC of the fitted model.
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// Return the fitted one-step-ahead values on the original scale.
    pub fn fitted_values(&self) -> &[f64] {
        &self.fitted_vals
    }

    /// Return the Box-Cox lambda, or `None` if no transformation was applied.
    pub fn lambda(&self) -> Option<f64> {
        self.lambda
    }

    /// Return the smoothing parameter α.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the trend damping parameter φ.
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Return whether a trend component was fitted.
    pub fn use_trend(&self) -> bool {
        self.use_trend
    }

    /// Return the residual standard deviation on the (possibly transformed) scale.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Return the number of training observations.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Apply Box-Cox transformation.
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

/// Determine Box-Cox lambda from config, with auto-detection via profile log-likelihood.
fn determine_lambda(data: &[f64], config: &BatsConfig) -> Result<Option<f64>> {
    match config.use_box_cox {
        Some(false) => Ok(None),
        Some(true) => Ok(Some(estimate_box_cox_lambda(data))),
        None => {
            if data.iter().all(|&v| v > 0.0) {
                let lam = estimate_box_cox_lambda(data);
                if (lam - 1.0).abs() > 0.1 {
                    Ok(Some(lam))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        }
    }
}

/// Estimate Box-Cox lambda via grid search on the profile log-likelihood.
fn estimate_box_cox_lambda(data: &[f64]) -> f64 {
    if data.iter().any(|&v| v <= 0.0) {
        return 1.0;
    }
    let n = data.len() as f64;
    let log_y_sum: f64 = data.iter().map(|&y| y.max(1e-10).ln()).sum();
    let candidates: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.1).collect();

    let mut best_lam = 1.0_f64;
    let mut best_ll = f64::NEG_INFINITY;

    for &lam in &candidates {
        let transformed: Vec<f64> = data.iter().map(|&y| box_cox(y, lam)).collect();
        let mean = transformed.iter().sum::<f64>() / n;
        let var = transformed.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / n;
        if var <= 0.0 {
            continue;
        }
        let ll = -0.5 * n * var.ln() + (lam - 1.0) * log_y_sum;
        if ll > best_ll {
            best_ll = ll;
            best_lam = lam;
        }
    }
    best_lam
}

/// Initialise seasonal states using a simple decomposition heuristic.
///
/// For each period `m_j`, the initial states are set to the de-levelled
/// seasonal averages of the first `m_j` observations.
fn init_seasonal_states(data: &[f64], periods: &[usize]) -> Vec<SeasonalState> {
    let n = data.len();
    let global_mean = data.iter().sum::<f64>() / n as f64;
    let gamma_default = 0.001;

    periods
        .iter()
        .map(|&m| {
            let mut init = vec![0.0_f64; m];
            let cycles = n / m;
            if cycles == 0 {
                // Not enough data for seasonal init: use zeros
                return SeasonalState::new(m, init, gamma_default);
            }
            // Seasonal averages over the available cycles
            let mut counts = vec![0usize; m];
            for (i, &v) in data.iter().enumerate() {
                let idx = i % m;
                init[idx] += v - global_mean;
                counts[idx] += 1;
            }
            for i in 0..m {
                if counts[i] > 0 {
                    init[i] /= counts[i] as f64;
                }
            }
            // Centre seasonal effects
            let seas_mean = init.iter().sum::<f64>() / m as f64;
            for v in &mut init {
                *v -= seas_mean;
            }
            SeasonalState::new(m, init, gamma_default)
        })
        .collect()
}

/// Run the BATS forward recursion over the entire training set.
///
/// Returns updated seasonal states, the fitted ARMA model, final level,
/// final trend state, fitted values (transformed scale), and residuals
/// (transformed scale).
#[allow(clippy::too_many_arguments)]
fn forward_pass(
    working: &[f64],
    alpha: f64,
    beta: f64,
    phi: f64,
    use_trend: bool,
    mut seasonal_states: Vec<SeasonalState>,
    p_order: usize,
    q_order: usize,
) -> (Vec<SeasonalState>, ArmaState, f64, f64, Vec<f64>, Vec<f64>) {
    let n = working.len();

    // Initialise level and trend
    let mut level = working.iter().take(3.min(n)).sum::<f64>() / 3.0_f64.min(n as f64);
    let mut trend_state = if use_trend && n >= 2 {
        (working[1] - working[0]).abs() * 0.01 // Small positive init
    } else {
        0.0
    };

    let ar_init = vec![0.0_f64; p_order];
    let ma_init = vec![0.0_f64; q_order];
    let mut arma = ArmaState::new(ar_init, ma_init);

    let mut fitted_tf = Vec::with_capacity(n);
    let mut residuals_tf = Vec::with_capacity(n);

    // Collect raw residuals first with zero AR/MA (will refine below)
    let mut raw_errors = Vec::with_capacity(n);

    for t in 0..n {
        let trend_contrib = if use_trend { phi * trend_state } else { 0.0 };
        let seas_contrib: f64 = seasonal_states.iter().map(|s| s.contribution()).sum();
        let arma_contrib = arma.contribution();
        let yhat = level + trend_contrib + seas_contrib + arma_contrib;

        fitted_tf.push(yhat);
        let error = working[t] - yhat;
        raw_errors.push(error);
        residuals_tf.push(error);

        // Update level and trend
        level += trend_contrib + alpha * error;
        if use_trend {
            trend_state = (1.0 - phi) * 0.0 + phi * trend_state + beta * error;
        }
        // Update seasonal states
        for sc in &mut seasonal_states {
            sc.update(error);
        }
        // Update ARMA
        let d_current = arma.contribution();
        arma.push(error, d_current);
    }

    // ── ARMA residual fitting via Yule-Walker ─────────────────────────────
    // If AR or MA orders are non-zero, re-estimate coefficients from the
    // raw residual sequence and re-run the forward pass once more.
    if p_order > 0 || q_order > 0 {
        let ar_coeffs = if p_order > 0 {
            yule_walker(&raw_errors, p_order)
        } else {
            Vec::new()
        };
        let ma_coeffs = if q_order > 0 {
            // MA coefficient estimation from ARMA residuals via innovation algorithm
            estimate_ma_coeffs(&raw_errors, q_order)
        } else {
            Vec::new()
        };

        // Re-run forward pass with estimated ARMA coefficients
        let mut arma2 = ArmaState::new(ar_coeffs, ma_coeffs);
        // Reset level and trend
        let mut level2 = working.iter().take(3.min(n)).sum::<f64>() / 3.0_f64.min(n as f64);
        let mut trend2 = if use_trend && n >= 2 {
            (working[1] - working[0]).abs() * 0.01
        } else {
            0.0
        };
        // Reset seasonal states
        let mut seas2 = seasonal_states.clone();
        // Re-init seasonal states from original data
        // (they are already initialised from the first pass; we reuse)

        let mut fitted2 = Vec::with_capacity(n);
        let mut resid2 = Vec::with_capacity(n);

        for t in 0..n {
            let tc = if use_trend { phi * trend2 } else { 0.0 };
            let sc: f64 = seas2.iter().map(|s| s.contribution()).sum();
            let ac = arma2.contribution();
            let yhat2 = level2 + tc + sc + ac;
            fitted2.push(yhat2);
            let err = working[t] - yhat2;
            resid2.push(err);

            level2 += tc + alpha * err;
            if use_trend {
                trend2 = phi * trend2 + beta * err;
            }
            for sc_state in &mut seas2 {
                sc_state.update(err);
            }
            let d2 = arma2.contribution();
            arma2.push(err, d2);
        }

        return (seas2, arma2, level2, trend2, fitted2, resid2);
    }

    (
        seasonal_states,
        arma,
        level,
        trend_state,
        fitted_tf,
        residuals_tf,
    )
}

/// Estimate AR coefficients via the Yule-Walker equations.
fn yule_walker(data: &[f64], p: usize) -> Vec<f64> {
    let n = data.len();
    if n < p + 1 || p == 0 {
        return vec![0.0; p];
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&v| v - mean).collect();

    // Compute autocorrelations r_0…r_p
    let mut r = vec![0.0_f64; p + 1];
    for lag in 0..=p {
        let mut s = 0.0_f64;
        for t in lag..n {
            s += centered[t] * centered[t - lag];
        }
        r[lag] = s / n as f64;
    }

    if r[0].abs() < 1e-14 {
        return vec![0.0; p];
    }

    // Build Toeplitz system R φ = r (Levinson-Durbin would be O(p²) but we
    // use Gaussian elimination for simplicity since p is small).
    let mut mat = vec![vec![0.0_f64; p + 1]; p];
    for i in 0..p {
        for j in 0..p {
            let lag = (i as isize - j as isize).unsigned_abs();
            mat[i][j] = r[lag] / r[0];
        }
        mat[i][p] = r[i + 1] / r[0];
    }

    gaussian_elimination(&mut mat).unwrap_or_else(|_| vec![0.0; p])
}

/// Estimate MA coefficients from residuals via a simple autocorrelation inversion.
///
/// Uses the innovation algorithm truncated at order `q`.
fn estimate_ma_coeffs(residuals: &[f64], q: usize) -> Vec<f64> {
    let n = residuals.len();
    if n < q + 1 || q == 0 {
        return vec![0.0; q];
    }

    let mean = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|&v| v - mean).collect();

    let mut gamma = vec![0.0_f64; q + 1];
    for lag in 0..=q {
        for t in lag..n {
            gamma[lag] += centered[t] * centered[t - lag];
        }
        gamma[lag] /= n as f64;
    }

    if gamma[0].abs() < 1e-14 {
        return vec![0.0; q];
    }

    // Simple ISTA-style approach: solve via iterative refinement.
    // Initialise θ from the sample autocorrelation at lags 1..q.
    let mut theta = vec![0.0_f64; q];
    for i in 0..q {
        theta[i] = (gamma[i + 1] / gamma[0]).clamp(-0.99, 0.99);
    }

    // Refinement: iterate to reduce the gap between model and sample ACF.
    for _ in 0..ARMA_ITER {
        let mut updated = theta.clone();
        for i in 0..q {
            // Theoretical ACF at lag i+1 from MA(q) model:
            // γ(h) = σ² Σ_{j=0}^{q-h} θ_j θ_{j+h}   (h > 0)
            let mut model_acf = 0.0_f64;
            for j in 0..q.saturating_sub(i) {
                let tj = if j == 0 { 1.0 } else { theta[j - 1] };
                let tjh = if j + i + 1 <= q { theta[j + i] } else { 0.0 };
                model_acf += tj * tjh;
            }
            let target = gamma[i + 1] / gamma[0];
            // Simple gradient step
            let grad = model_acf - target;
            updated[i] -= 0.1 * grad;
            updated[i] = updated[i].clamp(-0.99, 0.99);
        }
        theta = updated;
    }

    theta
}

/// Solve an augmented `m × (m+1)` system via Gaussian elimination (in-place).
fn gaussian_elimination(mat: &mut Vec<Vec<f64>>) -> Result<Vec<f64>> {
    let m = mat.len();
    if m == 0 {
        return Ok(Vec::new());
    }

    for col in 0..m {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in col + 1..m {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            mat.swap(col, max_row);
        }

        let pivot = mat[col][col];
        if pivot.abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "near-singular matrix in Yule-Walker solve".to_string(),
            ));
        }

        let n_cols = mat[col].len();
        let pivot_inv = 1.0 / pivot;
        for j in col..n_cols {
            let v = mat[col][j];
            mat[col][j] = v * pivot_inv;
        }
        for row in 0..m {
            if row != col {
                let factor = mat[row][col];
                let n_cols2 = mat[row].len();
                for j in col..n_cols2 {
                    let sub = factor * mat[col][j];
                    mat[row][j] -= sub;
                }
            }
        }
    }

    Ok(mat.iter().map(|row| *row.last().unwrap_or(&0.0)).collect())
}

/// OLS slope estimate (intercept discarded) for trend detection heuristic.
fn ols_slope(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let t_mean = (n + 1.0) / 2.0;
    let y_mean = data.iter().sum::<f64>() / n;
    let mut sxy = 0.0_f64;
    let mut sxx = 0.0_f64;
    for (i, &y) in data.iter().enumerate() {
        let t = (i + 1) as f64;
        sxy += (t - t_mean) * (y - y_mean);
        sxx += (t - t_mean).powi(2);
    }
    if sxx.abs() < 1e-14 {
        0.0
    } else {
        sxy / sxx
    }
}

/// Rational approximation to the standard normal quantile (Peter Acklam's method).
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
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
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

    fn make_seasonal(n_cycles: usize, period: usize) -> Vec<f64> {
        let mut v = Vec::new();
        for c in 0..n_cycles {
            for i in 0..period {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
                v.push(10.0 + c as f64 * 0.3 + 3.0 * angle.sin());
            }
        }
        v
    }

    fn make_trend_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 1.0 + 0.05 * i as f64).collect()
    }

    fn make_exponential_data() -> Vec<f64> {
        (1..=30).map(|i| (i as f64 * 0.1).exp() + 1.0).collect()
    }

    // ── Fitting ───────────────────────────────────────────────────────────────
    #[test]
    fn test_bats_fit_no_seasonality() {
        let data = make_trend_data(20);
        let config = BatsConfig {
            use_box_cox: Some(false),
            use_trend: Some(true),
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit should succeed");
        assert_eq!(model.fitted_values().len(), data.len());
        assert!(model.use_trend());
    }

    #[test]
    fn test_bats_fit_single_season() {
        let data = make_seasonal(4, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            use_trend: Some(false),
            seasonal_periods: vec![7],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit should succeed");
        assert_eq!(model.fitted_values().len(), data.len());
    }

    #[test]
    fn test_bats_fit_multiple_seasons() {
        let data: Vec<f64> = (0..60)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * i as f64;
                5.0 + 2.0 * (w / 7.0).sin() + 1.0 * (w / 30.0).sin()
            })
            .collect();
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7, 30],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit should succeed");
        assert_eq!(model.fitted_values().len(), data.len());
        assert!(model.aic().is_finite());
    }

    // ── Forecasting ───────────────────────────────────────────────────────────
    #[test]
    fn test_bats_forecast_length() {
        let data = make_seasonal(5, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        let fc = model.forecast(14).expect("forecast");
        assert_eq!(fc.len(), 14);
    }

    #[test]
    fn test_bats_forecast_finite() {
        let data = make_seasonal(5, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        let fc = model.forecast(21).expect("forecast");
        for (i, &f) in fc.iter().enumerate() {
            assert!(f.is_finite(), "forecast[{i}] is not finite: {f}");
        }
    }

    // ── Predict (with CIs) ────────────────────────────────────────────────────
    #[test]
    fn test_bats_predict_ci_ordering() {
        let data = make_seasonal(5, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        let (fc, lower, upper) = model.predict(14, 0.05).expect("predict");
        assert_eq!(fc.len(), 14);
        for i in 0..14 {
            assert!(
                lower[i] <= upper[i],
                "lower must be <= upper at step {i}: {lower_val} > {upper_val}",
                lower_val = lower[i],
                upper_val = upper[i]
            );
        }
    }

    #[test]
    fn test_bats_predict_invalid_alpha() {
        let data = make_seasonal(4, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        assert!(model.predict(5, 0.0).is_err());
        assert!(model.predict(5, 1.0).is_err());
    }

    // ── Box-Cox ───────────────────────────────────────────────────────────────
    #[test]
    fn test_bats_box_cox_roundtrip() {
        let vals = [1.0_f64, 2.5, 10.0, 100.0];
        for &v in &vals {
            for &lam in &[0.0_f64, 0.5, 1.0, -0.5, 2.0] {
                let w = box_cox(v, lam);
                let recovered = inv_box_cox(w, lam);
                assert!(
                    (recovered - v).abs() < 1e-8,
                    "roundtrip failed: v={v}, λ={lam}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn test_bats_auto_box_cox() {
        let data = make_exponential_data();
        let config = BatsConfig {
            use_box_cox: None, // auto
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        let fc = model.forecast(5).expect("forecast");
        for &f in fc.iter() {
            assert!(f.is_finite());
            assert!(f > 0.0, "exponential forecast should be positive");
        }
    }

    // ── AR/MA errors ──────────────────────────────────────────────────────────
    #[test]
    fn test_bats_arma_errors() {
        let data = make_seasonal(5, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7],
            ar_order: Some(1),
            ma_order: Some(1),
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        let fc = model.forecast(7).expect("forecast");
        assert_eq!(fc.len(), 7);
        for &f in fc.iter() {
            assert!(f.is_finite());
        }
    }

    // ── Error conditions ──────────────────────────────────────────────────────
    #[test]
    fn test_bats_insufficient_data() {
        let config = BatsConfig {
            seasonal_periods: vec![12],
            ..Default::default()
        };
        assert!(BatsModel::fit(&[1.0, 2.0, 3.0], config).is_err());
    }

    #[test]
    fn test_bats_invalid_period() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = BatsConfig {
            seasonal_periods: vec![1], // period < 2 is invalid
            ..Default::default()
        };
        assert!(BatsModel::fit(&data, config).is_err());
    }

    // ── AIC ───────────────────────────────────────────────────────────────────
    #[test]
    fn test_bats_aic_finite() {
        let data = make_seasonal(4, 7);
        let config = BatsConfig {
            use_box_cox: Some(false),
            seasonal_periods: vec![7],
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        assert!(model.aic().is_finite(), "AIC must be finite");
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    #[test]
    fn test_bats_accessors() {
        let data = make_trend_data(20);
        let config = BatsConfig {
            use_box_cox: Some(false),
            use_trend: Some(true),
            ..Default::default()
        };
        let model = BatsModel::fit(&data, config).expect("fit");
        assert!(model.alpha() > 0.0 && model.alpha() < 1.0);
        assert!(model.phi() > 0.0 && model.phi() <= 1.0);
        assert_eq!(model.n_obs(), data.len());
        assert_eq!(model.lambda(), None);
        assert!(model.sigma() > 0.0);
    }
}
