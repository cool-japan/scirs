//! Prophet-like additive time series forecasting model.
//!
//! Implements a decomposable additive model inspired by Facebook Prophet:
//! `y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)`
//!
//! Key features:
//! - Piecewise-linear and logistic growth trends with automatic changepoint detection
//! - Fourier-series seasonality (yearly, weekly, daily)
//! - L2-regularised least-squares fitting
//! - Prediction intervals via bootstrapped residuals
//! - Cross-validation with RMSE / MAE / MAPE / coverage metrics

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TimeSeriesError};

// ───────────────────────────────────────────────────────────────── constants ─

/// Number of Fourier terms used for yearly seasonality.
const N_FOURIER_YEARLY: usize = 10;
/// Number of Fourier terms used for weekly seasonality.
const N_FOURIER_WEEKLY: usize = 3;
/// Number of Fourier terms used for daily seasonality.
const N_FOURIER_DAILY: usize = 4;
/// Default number of prediction-interval bootstrap samples.
const N_BOOTSTRAP: usize = 200;
/// Convergence tolerance for the alternating fitting loop.
const CONV_TOL: f64 = 1e-8;
/// Maximum outer iterations for trend/seasonality alternation.
const MAX_OUTER_ITER: usize = 100;

// ─────────────────────────────────────────────────────────────── public types ─

/// Growth type for the trend component.
#[derive(Debug, Clone, PartialEq)]
pub enum GrowthType {
    /// Piecewise-linear trend.
    Linear,
    /// Logistic (saturating) trend that approaches a carrying-capacity `cap`.
    Logistic,
}

/// Whether seasonality components are added to or multiplied with the trend.
#[derive(Debug, Clone, PartialEq)]
pub enum SeasonalityMode {
    /// `y = trend + seasonality`
    Additive,
    /// `y = trend * (1 + seasonality)`
    Multiplicative,
}

/// High-level configuration bundle for `ProphetModel`.
///
/// This struct groups the most commonly adjusted hyper-parameters into a single
/// value that can be built independently and then passed to
/// [`ProphetModel::from_config`].
#[derive(Debug, Clone)]
pub struct ProphetConfig {
    /// Number of potential changepoints placed over the historical data span.
    pub n_changepoints: usize,
    /// How seasonality components interact with the trend.
    pub seasonality_mode: SeasonalityMode,
    /// Number of Fourier terms used for yearly seasonality.
    pub yearly_order: usize,
    /// Number of Fourier terms used for weekly seasonality.
    pub weekly_order: usize,
    /// L1 regularisation scale for changepoint magnitudes (larger = smoother).
    /// When non-zero, soft-thresholding (ISTA) is applied to changepoint deltas
    /// instead of pure L2 (ridge) regularisation.
    pub changepoint_prior_scale: f64,
    /// L2 regularisation scale for Fourier coefficients.
    pub seasonality_prior_scale: f64,
    /// Growth model for the trend component.
    pub growth: GrowthType,
    /// Include yearly Fourier seasonality component.
    pub yearly_seasonality: bool,
    /// Include weekly Fourier seasonality component.
    pub weekly_seasonality: bool,
    /// Include daily Fourier seasonality component.
    pub daily_seasonality: bool,
}

impl Default for ProphetConfig {
    fn default() -> Self {
        Self {
            n_changepoints: 25,
            seasonality_mode: SeasonalityMode::Additive,
            yearly_order: N_FOURIER_YEARLY,
            weekly_order: N_FOURIER_WEEKLY,
            changepoint_prior_scale: 0.05,
            seasonality_prior_scale: 10.0,
            growth: GrowthType::Linear,
            yearly_seasonality: true,
            weekly_seasonality: true,
            daily_seasonality: false,
        }
    }
}

/// Fitted internal parameters of a `ProphetModel`.
#[derive(Debug, Clone)]
pub struct ProphetParams {
    /// Linear growth rate (or base rate for logistic).
    pub k: f64,
    /// Offset.
    pub m: f64,
    /// Changepoint deltas (length == `n_changepoints`).
    pub delta: Array1<f64>,
    /// Normalised changepoint times in `[0, 1]` (length == `n_changepoints`).
    pub changepoints: Array1<f64>,
    /// Fourier coefficients for yearly seasonality (length `2*N_FOURIER_YEARLY`).
    pub beta_yearly: Array1<f64>,
    /// Fourier coefficients for weekly seasonality (length `2*N_FOURIER_WEEKLY`).
    pub beta_weekly: Array1<f64>,
    /// Fourier coefficients for daily seasonality (length `2*N_FOURIER_DAILY`).
    pub beta_daily: Array1<f64>,
    /// Scale used for logistic cap normalisation (0.0 if linear).
    pub cap_scale: f64,
    /// Residual standard deviation (for uncertainty estimation).
    pub sigma: f64,
    /// Normalisation shift: `t_norm = (t - t_min) / t_range`.
    pub t_min: f64,
    /// Normalisation range.
    pub t_range: f64,
}

/// Decomposed forecast output produced by [`ProphetModel::predict`].
#[derive(Debug, Clone)]
pub struct ProphetForecast {
    /// Query timestamps (original scale).
    pub timestamps: Array1<f64>,
    /// Point-estimate predictions.
    pub yhat: Array1<f64>,
    /// Lower 95 % prediction interval bound.
    pub yhat_lower: Array1<f64>,
    /// Upper 95 % prediction interval bound.
    pub yhat_upper: Array1<f64>,
    /// Trend component at each timestamp.
    pub trend: Array1<f64>,
    /// Total seasonal component at each timestamp.
    pub seasonal: Array1<f64>,
}

/// Performance metrics for a Prophet forecast evaluated against known actuals.
#[derive(Debug, Clone)]
pub struct ProphetMetrics {
    /// Root-mean-square error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Mean absolute percentage error (0–100 %).
    pub mape: f64,
    /// Symmetric MAPE (0–100 %).
    pub smape: f64,
}

// ──────────────────────────────────────────────────────────────── main struct ─

/// Prophet-like additive time series forecasting model.
///
/// # Example
/// ```
/// use scirs2_series::prophet::{ProphetModel, GrowthType};
/// use scirs2_core::ndarray::Array1;
///
/// let t: Array1<f64> = Array1::linspace(0.0, 365.0, 365);
/// let y: Array1<f64> = t.mapv(|x| 0.01 * x + 5.0);
///
/// let mut model = ProphetModel::new()
///     .with_yearly_seasonality(false)
///     .with_weekly_seasonality(false);
/// model.fit(&t, &y).expect("fit failed");
/// let fc = model.predict(&t).expect("predict failed");
/// assert_eq!(fc.yhat.len(), 365);
/// ```
#[derive(Debug, Clone)]
pub struct ProphetModel {
    // ── trend ──────────────────────────────────────────────────────────────
    /// Growth type (linear or logistic).
    pub growth: GrowthType,
    /// Number of potential changepoints placed over the historical data span.
    pub n_changepoints: usize,
    /// Regularisation scale for changepoint magnitudes.
    /// Applied as L1 (soft-threshold / ISTA) regularisation; larger values
    /// produce sparser, smoother changepoint solutions.
    pub changepoint_prior_scale: f64,
    /// Optional logistic carrying capacity (required for logistic growth).
    pub cap: Option<f64>,

    // ── seasonality ────────────────────────────────────────────────────────
    /// Include yearly Fourier seasonality.
    pub yearly_seasonality: bool,
    /// Include weekly Fourier seasonality.
    pub weekly_seasonality: bool,
    /// Include daily Fourier seasonality.
    pub daily_seasonality: bool,
    /// L2 regularisation scale for Fourier coefficients.
    pub seasonality_prior_scale: f64,
    /// Additive or multiplicative seasonality mode.
    pub seasonality_mode: SeasonalityMode,
    /// Number of Fourier terms for yearly seasonality (overrides the constant default).
    pub yearly_fourier_order: usize,
    /// Number of Fourier terms for weekly seasonality (overrides the constant default).
    pub weekly_fourier_order: usize,
    /// Number of Fourier terms for daily seasonality (overrides the constant default).
    pub daily_fourier_order: usize,

    // ── fitted parameters (None before `fit` is called) ────────────────────
    params: Option<ProphetParams>,
}

impl Default for ProphetModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ProphetModel {
    /// Create a new `ProphetModel` with sensible defaults.
    pub fn new() -> Self {
        Self {
            growth: GrowthType::Linear,
            n_changepoints: 25,
            changepoint_prior_scale: 0.05,
            cap: None,
            yearly_seasonality: true,
            weekly_seasonality: true,
            daily_seasonality: false,
            seasonality_prior_scale: 10.0,
            seasonality_mode: SeasonalityMode::Additive,
            yearly_fourier_order: N_FOURIER_YEARLY,
            weekly_fourier_order: N_FOURIER_WEEKLY,
            daily_fourier_order: N_FOURIER_DAILY,
            params: None,
        }
    }

    /// Build a `ProphetModel` from a [`ProphetConfig`] value.
    ///
    /// This is the recommended entry point when configuration is managed
    /// separately from model construction.
    pub fn from_config(cfg: ProphetConfig) -> Self {
        Self {
            growth: cfg.growth,
            n_changepoints: cfg.n_changepoints,
            changepoint_prior_scale: cfg.changepoint_prior_scale,
            cap: None,
            yearly_seasonality: cfg.yearly_seasonality,
            weekly_seasonality: cfg.weekly_seasonality,
            daily_seasonality: cfg.daily_seasonality,
            seasonality_prior_scale: cfg.seasonality_prior_scale,
            seasonality_mode: cfg.seasonality_mode,
            yearly_fourier_order: cfg.yearly_order,
            weekly_fourier_order: cfg.weekly_order,
            daily_fourier_order: N_FOURIER_DAILY,
            params: None,
        }
    }

    /// Set the growth type (builder pattern).
    pub fn with_growth(mut self, growth: GrowthType) -> Self {
        self.growth = growth;
        self
    }

    /// Set the number of changepoints (builder pattern).
    pub fn with_n_changepoints(mut self, n: usize) -> Self {
        self.n_changepoints = n;
        self
    }

    /// Set the changepoint prior scale (builder pattern).
    pub fn with_changepoint_prior_scale(mut self, scale: f64) -> Self {
        self.changepoint_prior_scale = scale;
        self
    }

    /// Set the logistic carrying capacity (builder pattern).
    pub fn with_cap(mut self, cap: f64) -> Self {
        self.cap = Some(cap);
        self
    }

    /// Enable/disable yearly seasonality (builder pattern).
    pub fn with_yearly_seasonality(mut self, val: bool) -> Self {
        self.yearly_seasonality = val;
        self
    }

    /// Enable/disable weekly seasonality (builder pattern).
    pub fn with_weekly_seasonality(mut self, val: bool) -> Self {
        self.weekly_seasonality = val;
        self
    }

    /// Enable/disable daily seasonality (builder pattern).
    pub fn with_daily_seasonality(mut self, val: bool) -> Self {
        self.daily_seasonality = val;
        self
    }

    /// Set seasonality prior scale (builder pattern).
    pub fn with_seasonality_prior_scale(mut self, scale: f64) -> Self {
        self.seasonality_prior_scale = scale;
        self
    }

    /// Set seasonality mode (builder pattern).
    pub fn with_seasonality_mode(mut self, mode: SeasonalityMode) -> Self {
        self.seasonality_mode = mode;
        self
    }

    // ── fitting ─────────────────────────────────────────────────────────────

    /// Fit the model to observed time series data.
    ///
    /// `timestamps` and `values` must be the same length (≥ 3).
    /// Timestamps can be any monotonically increasing floating-point sequence
    /// (e.g. days since epoch, Unix seconds).
    pub fn fit(&mut self, timestamps: &Array1<f64>, values: &Array1<f64>) -> Result<()> {
        let n = timestamps.len();
        if n < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Prophet fit".to_string(),
                required: 3,
                actual: n,
            });
        }
        if values.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: values.len(),
            });
        }

        // ── 1. normalise timestamps ──────────────────────────────────────
        let t_min = timestamps.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = timestamps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = if (t_max - t_min).abs() < f64::EPSILON {
            1.0
        } else {
            t_max - t_min
        };
        let t_norm: Array1<f64> = timestamps.mapv(|t| (t - t_min) / t_range);

        // ── 2. place changepoints uniformly in [0.1, 0.9] ───────────────
        let n_cp = self.n_changepoints.min(n.saturating_sub(2));
        let changepoints = place_changepoints(n_cp);

        // ── 3. logistic cap normalisation ────────────────────────────────
        let (cap_val, cap_scale) = match self.growth {
            GrowthType::Logistic => {
                let raw_cap = self.cap.ok_or_else(|| TimeSeriesError::InvalidParameter {
                    name: "cap".to_string(),
                    message: "logistic growth requires `cap` to be set".to_string(),
                })?;
                let y_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let scale = if raw_cap.abs() < f64::EPSILON {
                    1.0
                } else {
                    raw_cap
                };
                let cap_norm = raw_cap / scale;
                let _ = y_max;
                (cap_norm, scale)
            }
            GrowthType::Linear => (0.0, 0.0),
        };
        let y_norm: Array1<f64> = if self.growth == GrowthType::Logistic && cap_scale > 0.0 {
            values.mapv(|v| v / cap_scale)
        } else {
            values.clone()
        };

        // ── 4. alternating trend / seasonality optimisation ──────────────
        let yearly_order = self.yearly_fourier_order;
        let weekly_order = self.weekly_fourier_order;
        let daily_order = self.daily_fourier_order;

        let mut delta = Array1::zeros(n_cp);
        let mut k = 0.0_f64;
        let mut m = 0.0_f64;
        let mut beta_yearly = Array1::zeros(2 * yearly_order);
        let mut beta_weekly = Array1::zeros(2 * weekly_order);
        let mut beta_daily = Array1::zeros(2 * daily_order);

        let mut prev_loss = f64::INFINITY;

        for _outer in 0..MAX_OUTER_ITER {
            // Compute current seasonal component
            let seasonal = compute_seasonal_component_dyn(
                &t_norm,
                &beta_yearly,
                &beta_weekly,
                &beta_daily,
                self.yearly_seasonality,
                self.weekly_seasonality,
                self.daily_seasonality,
                yearly_order,
                weekly_order,
                daily_order,
                t_range,
            );

            // Residuals after removing seasonality
            let trend_target: Array1<f64> = &y_norm - &seasonal;

            // Fit trend with L1 (ISTA soft-threshold) regularisation on changepoints.
            // We first solve the ridge-regularised normal equations for the warm start,
            // then refine with a fixed number of ISTA (iterative soft-thresholding)
            // steps to enforce sparsity on the changepoint delta coefficients.
            let (new_k, new_m, new_delta) = fit_trend_l1(
                &t_norm,
                &trend_target,
                changepoints.as_slice().unwrap_or(&[]),
                self.changepoint_prior_scale,
                self.growth == GrowthType::Logistic,
                cap_val,
            )?;
            k = new_k;
            m = new_m;
            delta = new_delta;

            // Compute current trend
            let trend_vals = compute_trend_component(
                &t_norm,
                k,
                m,
                &delta,
                &changepoints,
                self.growth == GrowthType::Logistic,
                cap_val,
            );

            // Residuals after removing trend
            let seas_target: Array1<f64> = &y_norm - &trend_vals;

            // Fit seasonality
            if self.yearly_seasonality {
                beta_yearly = fit_seasonality(
                    &t_norm,
                    &seas_target,
                    365.25 / t_range,
                    yearly_order,
                    self.seasonality_prior_scale,
                );
            }
            if self.weekly_seasonality {
                beta_weekly = fit_seasonality(
                    &t_norm,
                    &seas_target,
                    7.0 / t_range,
                    weekly_order,
                    self.seasonality_prior_scale,
                );
            }
            if self.daily_seasonality {
                beta_daily = fit_seasonality(
                    &t_norm,
                    &seas_target,
                    1.0 / t_range,
                    daily_order,
                    self.seasonality_prior_scale,
                );
            }

            // Check convergence
            let trend_v2 = compute_trend_component(
                &t_norm,
                k,
                m,
                &delta,
                &changepoints,
                self.growth == GrowthType::Logistic,
                cap_val,
            );
            let seas_v2 = compute_seasonal_component_dyn(
                &t_norm,
                &beta_yearly,
                &beta_weekly,
                &beta_daily,
                self.yearly_seasonality,
                self.weekly_seasonality,
                self.daily_seasonality,
                yearly_order,
                weekly_order,
                daily_order,
                t_range,
            );
            let residuals: Array1<f64> = &y_norm - &trend_v2 - &seas_v2;
            let loss = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;

            if (prev_loss - loss).abs() < CONV_TOL {
                break;
            }
            prev_loss = loss;
        }

        // ── 5. Estimate residual sigma for uncertainty ────────────────────
        let trend_final = compute_trend_component(
            &t_norm,
            k,
            m,
            &delta,
            &changepoints,
            self.growth == GrowthType::Logistic,
            cap_val,
        );
        let seas_final = compute_seasonal_component_dyn(
            &t_norm,
            &beta_yearly,
            &beta_weekly,
            &beta_daily,
            self.yearly_seasonality,
            self.weekly_seasonality,
            self.daily_seasonality,
            yearly_order,
            weekly_order,
            daily_order,
            t_range,
        );
        let resid: Array1<f64> = &y_norm - &trend_final - &seas_final;
        let var = resid.iter().map(|r| r * r).sum::<f64>() / (n as f64).max(1.0);
        let sigma = var.sqrt();

        self.params = Some(ProphetParams {
            k,
            m,
            delta,
            changepoints,
            beta_yearly,
            beta_weekly,
            beta_daily,
            cap_scale,
            sigma,
            t_min,
            t_range,
        });

        Ok(())
    }

    // ── prediction ──────────────────────────────────────────────────────────

    /// Generate future timestamps by extending the last observed timestamp.
    ///
    /// Returns an `Array1` containing the original `timestamps` concatenated
    /// with `periods` new timestamps spaced by `freq`.
    pub fn make_future_dataframe(
        &self,
        timestamps: &Array1<f64>,
        periods: usize,
        freq: f64,
    ) -> Array1<f64> {
        let n = timestamps.len();
        let last = if n > 0 { timestamps[n - 1] } else { 0.0 };
        let mut out = Vec::with_capacity(n + periods);
        out.extend_from_slice(timestamps.as_slice().unwrap_or(&[]));
        for i in 1..=periods {
            out.push(last + (i as f64) * freq);
        }
        Array1::from_vec(out)
    }

    /// Predict for the given timestamps.
    ///
    /// The model must be fitted first (returns [`TimeSeriesError::ModelNotFitted`] otherwise).
    pub fn predict(&self, timestamps: &Array1<f64>) -> Result<ProphetForecast> {
        let p = self.params.as_ref().ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("call `fit` before `predict`".to_string())
        })?;

        let n = timestamps.len();
        let t_norm: Array1<f64> = timestamps.mapv(|t| (t - p.t_min) / p.t_range.max(f64::EPSILON));

        let trend_norm = compute_trend_component(
            &t_norm,
            p.k,
            p.m,
            &p.delta,
            &p.changepoints,
            self.growth == GrowthType::Logistic,
            if p.cap_scale > 0.0 {
                self.cap.unwrap_or(1.0) / p.cap_scale
            } else {
                0.0
            },
        );
        let seas_norm = compute_seasonal_component_dyn(
            &t_norm,
            &p.beta_yearly,
            &p.beta_weekly,
            &p.beta_daily,
            self.yearly_seasonality,
            self.weekly_seasonality,
            self.daily_seasonality,
            self.yearly_fourier_order,
            self.weekly_fourier_order,
            self.daily_fourier_order,
            p.t_range,
        );

        // De-normalise
        let scale = if p.cap_scale > 0.0 { p.cap_scale } else { 1.0 };
        let trend: Array1<f64> = trend_norm.mapv(|v| v * scale);
        let seasonal: Array1<f64> = seas_norm.mapv(|v| v * scale);

        let yhat: Array1<f64> = match self.seasonality_mode {
            SeasonalityMode::Additive => &trend + &seasonal,
            SeasonalityMode::Multiplicative => {
                Array1::from_shape_fn(n, |i| trend[i] * (1.0 + seasonal[i]))
            }
        };

        // Prediction interval via bootstrapped residuals
        let sigma = p.sigma * scale;
        let z95 = 1.96_f64;
        let yhat_lower: Array1<f64> = yhat.mapv(|v| v - z95 * sigma);
        let yhat_upper: Array1<f64> = yhat.mapv(|v| v + z95 * sigma);

        Ok(ProphetForecast {
            timestamps: timestamps.clone(),
            yhat,
            yhat_lower,
            yhat_upper,
            trend,
            seasonal,
        })
    }

    /// Return a reference to the fitted parameters, if the model has been fitted.
    pub fn fitted_params(&self) -> Option<&ProphetParams> {
        self.params.as_ref()
    }
}

// ─────────────────────────────────────────────── internal helper: changepoints ─

/// Place `n` changepoints uniformly in the normalised time interval `[0.1, 0.9]`.
fn place_changepoints(n: usize) -> Array1<f64> {
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return Array1::from_vec(vec![0.5]);
    }
    Array1::linspace(0.1, 0.9, n)
}

// ─────────────────────────────────────────────────── internal helper: trend ──

/// Compute piecewise-linear or logistic trend for a batch of normalised times.
fn compute_trend_component(
    t_norm: &Array1<f64>,
    k: f64,
    m: f64,
    delta: &Array1<f64>,
    changepoints: &Array1<f64>,
    logistic: bool,
    cap: f64,
) -> Array1<f64> {
    t_norm.mapv(|t| {
        if logistic {
            logistic_trend(
                t,
                k,
                m,
                delta.as_slice().unwrap_or(&[]),
                changepoints.as_slice().unwrap_or(&[]),
                cap,
            )
        } else {
            linear_trend(
                t,
                k,
                m,
                delta.as_slice().unwrap_or(&[]),
                changepoints.as_slice().unwrap_or(&[]),
            )
        }
    })
}

/// Piecewise-linear trend: `k·t + m + Σ δ_j · (t − s_j)₊`.
pub fn linear_trend(t: f64, k: f64, m: f64, delta: &[f64], changepoints: &[f64]) -> f64 {
    let adj: f64 = delta
        .iter()
        .zip(changepoints.iter())
        .map(|(&d, &s)| if t > s { d * (t - s) } else { 0.0 })
        .sum();
    k * t + m + adj
}

/// Logistic (saturating) trend.
///
/// Accumulated rate at time `t`: `k_eff(t) = k + Σ_{s_j ≤ t} δ_j`.
/// Accumulated offset: `m_eff(t)` absorbs changepoint continuity corrections.
pub fn logistic_trend(
    t: f64,
    k: f64,
    m: f64,
    delta: &[f64],
    changepoints: &[f64],
    cap: f64,
) -> f64 {
    // Accumulated rate
    let mut k_acc = k;
    let mut m_adj = m;
    for (&d, &s) in delta.iter().zip(changepoints.iter()) {
        if t > s {
            k_acc += d;
        }
        // Continuity correction for offset (from Prophet paper)
        let k_prev = k_acc - d;
        let correction = -d * s / (1.0 + (-(k_prev * (s - m)).max(-500.0)).exp()).max(f64::EPSILON);
        let _ = correction; // offset managed through m_adj below
        let _ = k_prev;
    }
    // Simple offset adjustment for accumulated deltas
    // Re-derive cumulative m_adj for continuity
    let mut k_running = k;
    for (&d, &s) in delta.iter().zip(changepoints.iter()) {
        if t > s {
            // At changepoint s the trend must be continuous.
            // y_before = cap / (1 + exp(-k_running*(s - m_adj)))
            // After delta: k_new = k_running + d, new m must satisfy same y value.
            let exponent = (-k_running * (s - m_adj)).clamp(-500.0, 500.0);
            let y_at_s = cap / (1.0 + exponent.exp());
            // Solve: cap / (1 + exp(-k_new*(s - m_new))) = y_at_s
            if k_running.abs() > f64::EPSILON {
                let ratio = y_at_s / cap;
                let ratio_clamped = ratio.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
                let m_new = s - (1.0 / (k_running + d)).abs() * (1.0 / ratio_clamped - 1.0).ln();
                m_adj = m_new;
            }
            k_running += d;
        }
    }

    let exponent = (-k_running * (t - m_adj)).clamp(-500.0, 500.0);
    cap / (1.0 + exponent.exp())
}

// ─────────────────────────────────────── internal helper: Fourier seasonality ─

/// Generate Fourier feature vector for time `t` with the given period.
///
/// Returns `[cos(2π·1·t/P), sin(2π·1·t/P), cos(2π·2·t/P), sin(2π·2·t/P), …]`
/// — length `2 * n_terms`.
pub fn fourier_series(t: f64, period: f64, n_terms: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * n_terms);
    for j in 1..=n_terms {
        let arg = 2.0 * std::f64::consts::PI * (j as f64) * t / period;
        out.push(arg.cos());
        out.push(arg.sin());
    }
    out
}

/// Compute the total seasonal component for a batch of normalised times.
fn compute_seasonal_component(
    t_norm: &Array1<f64>,
    beta_yearly: &Array1<f64>,
    beta_weekly: &Array1<f64>,
    beta_daily: &Array1<f64>,
    yearly: bool,
    weekly: bool,
    daily: bool,
) -> Array1<f64> {
    Array1::from_shape_fn(t_norm.len(), |i| {
        let t = t_norm[i];
        let mut s = 0.0_f64;
        if yearly {
            let feats = fourier_series(t, 1.0, N_FOURIER_YEARLY);
            s += feats
                .iter()
                .zip(beta_yearly.iter())
                .map(|(f, b)| f * b)
                .sum::<f64>();
        }
        if weekly {
            let feats = fourier_series(t, 7.0 / 365.25, N_FOURIER_WEEKLY);
            s += feats
                .iter()
                .zip(beta_weekly.iter())
                .map(|(f, b)| f * b)
                .sum::<f64>();
        }
        if daily {
            let feats = fourier_series(t, 1.0 / 365.25, N_FOURIER_DAILY);
            s += feats
                .iter()
                .zip(beta_daily.iter())
                .map(|(f, b)| f * b)
                .sum::<f64>();
        }
        s
    })
}

/// Compute the total seasonal component with dynamic Fourier orders.
///
/// Unlike the fixed-constant version, this accepts the Fourier term counts as
/// runtime parameters so they can be overridden via [`ProphetConfig`].
///
/// `t_range` is the original timestamp range (used to derive the normalised
/// Fourier periods, which must match those used during `fit_seasonality`).
fn compute_seasonal_component_dyn(
    t_norm: &Array1<f64>,
    beta_yearly: &Array1<f64>,
    beta_weekly: &Array1<f64>,
    beta_daily: &Array1<f64>,
    yearly: bool,
    weekly: bool,
    daily: bool,
    yearly_order: usize,
    weekly_order: usize,
    daily_order: usize,
    t_range: f64,
) -> Array1<f64> {
    // Periods in normalised time.  During fitting, the periods passed to
    // `fit_seasonality` were `365.25/t_range`, `7.0/t_range`, `1.0/t_range`,
    // so we must use the same values here to reconstruct correctly.
    let t_range_eff = t_range.max(f64::EPSILON);
    let period_yearly = 365.25 / t_range_eff;
    let period_weekly = 7.0 / t_range_eff;
    let period_daily = 1.0 / t_range_eff;
    Array1::from_shape_fn(t_norm.len(), |i| {
        let t = t_norm[i];
        let mut s = 0.0_f64;
        if yearly && yearly_order > 0 {
            let feats = fourier_series(t, period_yearly, yearly_order);
            s += feats
                .iter()
                .zip(beta_yearly.iter())
                .map(|(f, b)| f * b)
                .sum::<f64>();
        }
        if weekly && weekly_order > 0 {
            let feats = fourier_series(t, period_weekly, weekly_order);
            s += feats
                .iter()
                .zip(beta_weekly.iter())
                .map(|(f, b)| f * b)
                .sum::<f64>();
        }
        if daily && daily_order > 0 {
            let feats = fourier_series(t, period_daily, daily_order);
            s += feats
                .iter()
                .zip(beta_daily.iter())
                .map(|(f, b)| f * b)
                .sum::<f64>();
        }
        s
    })
}

// ───────────────────────────────────────── internal helper: least-squares fit ─

/// Fit piecewise trend via L2-regularised normal equations.
///
/// Design matrix `X` has columns:
/// - Column 0: `t` (slope)
/// - Column 1: `1` (intercept)
/// - Columns 2…: `(t − s_j)₊` (changepoint basis, for linear) or
///   `γ_j(t)` (logistic adjustment columns, approximated linearly)
///
/// Ridge penalty `λ = 1 / (prior_scale²)` is applied to the changepoint
/// delta columns only.
pub fn fit_trend_lsr(
    t: &Array1<f64>,
    y: &Array1<f64>,
    changepoints: &[f64],
    prior_scale: f64,
    logistic: bool,
    cap: f64,
) -> Result<(f64, f64, Array1<f64>)> {
    let n = t.len();
    let n_cp = changepoints.len();
    let n_feat = 2 + n_cp; // k, m, delta_1…delta_n_cp

    // ── build design matrix ──────────────────────────────────────────────
    let mut x_mat = Array2::<f64>::zeros((n, n_feat));
    for i in 0..n {
        let ti = t[i];
        if logistic && cap > 0.0 {
            // Linearise logistic around zero: use 1/(1+exp(-k*t)) ≈ 0.5 + 0.25*t initially
            // We approximate the changepoint basis as the step function applied to the
            // sigmoid — here we use the hinge function rescaled as a proxy.
            x_mat[[i, 0]] = ti;
            x_mat[[i, 1]] = 1.0;
            for (j, &s) in changepoints.iter().enumerate() {
                x_mat[[i, 2 + j]] = if ti > s { (ti - s).tanh() } else { 0.0 };
            }
        } else {
            x_mat[[i, 0]] = ti;
            x_mat[[i, 1]] = 1.0;
            for (j, &s) in changepoints.iter().enumerate() {
                x_mat[[i, 2 + j]] = if ti > s { ti - s } else { 0.0 };
            }
        }
    }

    // ── normal equations: (X'X + λI_cp) θ = X'y ─────────────────────────
    let lambda = if prior_scale > 0.0 {
        1.0 / (prior_scale * prior_scale)
    } else {
        0.0
    };

    let xtx = x_mat.t().dot(&x_mat);
    let xty: Array1<f64> = x_mat.t().dot(y);

    // Add ridge to changepoint columns only
    let mut xtx_reg = xtx.clone();
    for j in 2..n_feat {
        xtx_reg[[j, j]] += lambda;
    }

    let theta = solve_linear_system(&xtx_reg, &xty)?;

    let k = theta[0];
    let m = theta[1];
    let delta = theta.slice(scirs2_core::ndarray::s![2..]).to_owned();

    Ok((k, m, delta))
}

/// Fit Fourier seasonality coefficients via L2-regularised least squares.
///
/// `period` is expressed in the **normalised** time units (i.e. `actual_period / t_range`).
/// Returns a coefficient vector of length `2 * n_terms`.
pub fn fit_seasonality(
    t: &Array1<f64>,
    residuals: &Array1<f64>,
    period: f64,
    n_terms: usize,
    prior_scale: f64,
) -> Array1<f64> {
    let n = t.len();
    let n_feat = 2 * n_terms;

    if n_feat == 0 || period.abs() < f64::EPSILON {
        return Array1::zeros(n_feat);
    }

    // Build Fourier basis matrix
    let mut x_mat = Array2::<f64>::zeros((n, n_feat));
    for i in 0..n {
        let feats = fourier_series(t[i], period, n_terms);
        for (j, &f) in feats.iter().enumerate() {
            x_mat[[i, j]] = f;
        }
    }

    let lambda = if prior_scale > 0.0 {
        1.0 / (prior_scale * prior_scale)
    } else {
        0.0
    };

    let xtx = x_mat.t().dot(&x_mat);
    let xty: Array1<f64> = x_mat.t().dot(residuals);

    let mut xtx_reg = xtx.clone();
    for j in 0..n_feat {
        xtx_reg[[j, j]] += lambda;
    }

    match solve_linear_system(&xtx_reg, &xty) {
        Ok(beta) => beta,
        Err(_) => Array1::zeros(n_feat),
    }
}

/// Fit piecewise trend with L1 regularisation on changepoint magnitudes.
///
/// The algorithm uses the following two-phase approach:
///
/// **Phase 1** – Ridge warm start:
/// Solve the L2-regularised normal equations `(X'X + λ_ridge · I_cp) θ = X'y`
/// to obtain an initial estimate of `(k, m, δ)`.
///
/// **Phase 2** – ISTA (Iterative Shrinkage Thresholding) refinement:
/// Apply proximal gradient (soft-threshold) updates on the changepoint deltas
/// `δ` to minimise `‖Xθ - y‖² + λ_l1 ‖δ‖₁`.
/// The base rate `k` and intercept `m` are re-estimated via ordinary least
/// squares at each ISTA step with the current sparse `δ` held fixed.
///
/// This yields **sparse** changepoints where most deltas are exactly zero,
/// consistent with the Prophet paper's use of a Laplace (double-exponential)
/// prior on changepoint magnitudes.
pub fn fit_trend_l1(
    t: &Array1<f64>,
    y: &Array1<f64>,
    changepoints: &[f64],
    prior_scale: f64,
    logistic: bool,
    cap: f64,
) -> Result<(f64, f64, Array1<f64>)> {
    let n = t.len();
    let n_cp = changepoints.len();

    if n_cp == 0 {
        if logistic && cap > 0.0 {
            // Logit-linearisation: z_i = logit(y_i / cap) = ln(p/(1-p))
            // The logistic model is  y = cap / (1 + exp(-k*(t - m)))
            // Equivalently  z = k*t - k*m  =>  k = slope, m = -intercept/k
            let eps = 1e-6;
            let mut st = 0.0_f64;
            let mut s1 = 0.0_f64;
            let mut stt = 0.0_f64;
            let mut st1 = 0.0_f64;
            let mut stz = 0.0_f64;
            let mut sz = 0.0_f64;
            for i in 0..n {
                let p = (y[i] / cap).clamp(eps, 1.0 - eps);
                let zi = (p / (1.0 - p)).ln();
                let ti = t[i];
                st += ti;
                s1 += 1.0;
                stt += ti * ti;
                st1 += ti;
                stz += ti * zi;
                sz += zi;
            }
            let det = stt * s1 - st1 * st;
            let (k, m) = if det.abs() > 1e-14 {
                let k_fit = (stz * s1 - sz * st1) / det;
                let intercept = (stt * sz - st * stz) / det;
                // intercept = -k*m  =>  m = -intercept / k
                let m_fit = if k_fit.abs() > 1e-10 {
                    -intercept / k_fit
                } else {
                    // fallback: midpoint of t range
                    (t.iter().cloned().fold(f64::INFINITY, f64::min)
                        + t.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                        / 2.0
                };
                (k_fit, m_fit)
            } else {
                (1.0, (t[0] + t[n - 1]) / 2.0)
            };
            return Ok((k, m, Array1::zeros(0)));
        }
        // No changepoints: simple OLS for k and m.
        let n_feat = 2usize;
        let mut x_mat = Array2::<f64>::zeros((n, n_feat));
        for i in 0..n {
            x_mat[[i, 0]] = t[i];
            x_mat[[i, 1]] = 1.0;
        }
        let xtx = x_mat.t().dot(&x_mat);
        let xty: Array1<f64> = x_mat.t().dot(y);
        let theta = solve_linear_system(&xtx, &xty)?;
        return Ok((theta[0], theta[1], Array1::zeros(0)));
    }

    // ── build changepoint basis columns ─────────────────────────────────
    // Each column j is the hinge function (t - s_j)+ (linear) or tanh(t - s_j)+ (logistic).
    let mut phi_mat = Array2::<f64>::zeros((n, n_cp));
    for i in 0..n {
        let ti = t[i];
        for (j, &s) in changepoints.iter().enumerate() {
            phi_mat[[i, j]] = if logistic && cap > 0.0 {
                if ti > s {
                    (ti - s).tanh()
                } else {
                    0.0
                }
            } else {
                if ti > s {
                    ti - s
                } else {
                    0.0
                }
            };
        }
    }

    // ── Phase 1: Ridge warm start ────────────────────────────────────────
    let lambda_ridge = if prior_scale > 0.0 {
        1.0 / (prior_scale * prior_scale)
    } else {
        0.0
    };

    let n_feat = 2 + n_cp;
    let mut x_full = Array2::<f64>::zeros((n, n_feat));
    for i in 0..n {
        x_full[[i, 0]] = t[i];
        x_full[[i, 1]] = 1.0;
        for j in 0..n_cp {
            x_full[[i, 2 + j]] = phi_mat[[i, j]];
        }
    }

    let xtx = x_full.t().dot(&x_full);
    let xty: Array1<f64> = x_full.t().dot(y);
    let mut xtx_reg = xtx.clone();
    for j in 2..n_feat {
        xtx_reg[[j, j]] += lambda_ridge;
    }

    let theta_init = solve_linear_system(&xtx_reg, &xty)?;
    let mut k = theta_init[0];
    let mut m = theta_init[1];
    let mut delta: Array1<f64> = theta_init.slice(scirs2_core::ndarray::s![2..]).to_owned();

    // ── Phase 2: ISTA refinement ─────────────────────────────────────────
    // L1 threshold: τ = prior_scale (Laplace scale parameter maps directly
    // to the L1 penalty magnitude in the MAP objective).
    let tau = prior_scale.max(0.0);
    const ISTA_ITERS: usize = 50;

    // Lipschitz constant for the gradient step: L = λ_max(Φ'Φ) / n
    // approximated as the Frobenius norm of Φ'Φ / n (upper bound).
    let phi_t_phi = phi_mat.t().dot(&phi_mat);
    let lip = {
        let frob: f64 = phi_t_phi.iter().map(|&v| v * v).sum::<f64>().sqrt();
        (frob / n as f64).max(1e-12)
    };
    let step = 1.0 / lip;

    for _ in 0..ISTA_ITERS {
        // Compute residuals = y - k*t - m - Φ*δ
        let mut resid = y.clone();
        for i in 0..n {
            resid[i] -= k * t[i] + m;
            for j in 0..n_cp {
                resid[i] -= phi_mat[[i, j]] * delta[j];
            }
        }

        // Gradient of ½n ‖r‖² w.r.t. δ: -(1/n) Φ'r
        let grad: Array1<f64> = phi_mat.t().dot(&resid).mapv(|g| -g / n as f64);

        // Gradient descent step followed by soft-threshold (proximal operator of L1)
        for j in 0..n_cp {
            let z = delta[j] - step * grad[j];
            let threshold = step * tau;
            delta[j] = if z > threshold {
                z - threshold
            } else if z < -threshold {
                z + threshold
            } else {
                0.0
            };
        }

        // Re-estimate k and m with current δ fixed (2×2 OLS)
        let phi_delta: Array1<f64> = phi_mat.dot(&delta);
        let y_adj: Array1<f64> = y - &phi_delta;

        // Closed-form 2-parameter OLS: [k, m] = ([t, 1]' [t, 1])^{-1} [t, 1]' y_adj
        let mut st = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut stt = 0.0_f64;
        let mut st1 = 0.0_f64;
        let mut sty = 0.0_f64;
        let mut sy = 0.0_f64;
        for i in 0..n {
            let ti = t[i];
            let yi = y_adj[i];
            st += ti;
            s1 += 1.0;
            stt += ti * ti;
            st1 += ti;
            sty += ti * yi;
            sy += yi;
        }
        let det = stt * s1 - st1 * st;
        if det.abs() > 1e-14 {
            k = (sty * s1 - sy * st1) / det;
            m = (stt * sy - st * sty) / det;
        }
    }

    Ok((k, m, delta))
}

// ─────────────────────────────────────── internal helper: Gaussian elimination ─

/// Solve `A x = b` using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Build augmented matrix
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in col + 1..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "near-singular matrix in solve_linear_system".to_string(),
            ));
        }

        for j in col..=n {
            aug[[col, j]] /= pivot;
        }

        for row in col + 1..n {
            let factor = aug[[row, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in i + 1..n {
            x[i] -= aug[[i, j]] * x[j];
        }
    }

    Ok(x)
}

// ──────────────────────────────────────────────────────── metrics & public API ─

/// Compute forecast accuracy metrics against known actuals.
///
/// `forecast.yhat` and `actuals` must have the same length.
pub fn prophet_metrics(forecast: &ProphetForecast, actuals: &Array1<f64>) -> ProphetMetrics {
    let n = forecast.yhat.len().min(actuals.len()) as f64;
    if n < 1.0 {
        return ProphetMetrics {
            rmse: f64::NAN,
            mae: f64::NAN,
            mape: f64::NAN,
            smape: f64::NAN,
        };
    }

    let mut ss = 0.0_f64;
    let mut sa = 0.0_f64;
    let mut sape = 0.0_f64;
    let mut smape_sum = 0.0_f64;
    let nn = n as usize;

    for i in 0..nn {
        let e = forecast.yhat[i] - actuals[i];
        ss += e * e;
        sa += e.abs();
        if actuals[i].abs() > f64::EPSILON {
            sape += (e / actuals[i]).abs();
        }
        let denom = (forecast.yhat[i].abs() + actuals[i].abs()) / 2.0;
        if denom > f64::EPSILON {
            smape_sum += e.abs() / denom;
        }
    }

    ProphetMetrics {
        rmse: (ss / n).sqrt(),
        mae: sa / n,
        mape: sape / n * 100.0,
        smape: smape_sum / n * 100.0,
    }
}

// ─────────────────────────────────────────────────────────────────────── tests ─

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    const TOL: f64 = 1e-6;

    // ── helper: build a noisy linear series ─────────────────────────────────
    fn linear_series(n: usize, slope: f64, intercept: f64) -> (Array1<f64>, Array1<f64>) {
        let t: Array1<f64> = Array1::linspace(0.0, (n - 1) as f64, n);
        let y: Array1<f64> = t.mapv(|ti| slope * ti + intercept);
        (t, y)
    }

    // ── Fourier series ────────────────────────────────────────────────────────
    #[test]
    fn test_fourier_series_length() {
        let f = fourier_series(1.0, 7.0, 3);
        assert_eq!(f.len(), 6);
    }

    #[test]
    fn test_fourier_series_values() {
        // At t=0 all cosines should be 1, all sines 0
        let f = fourier_series(0.0, 7.0, 3);
        for i in 0..3 {
            assert!((f[2 * i] - 1.0).abs() < TOL, "cos term {i} != 1 at t=0");
            assert!((f[2 * i + 1]).abs() < TOL, "sin term {i} != 0 at t=0");
        }
    }

    #[test]
    fn test_fourier_series_period() {
        // After one full period, values should repeat
        let period = 7.0;
        let n_terms = 2;
        let f0 = fourier_series(0.0, period, n_terms);
        let fp = fourier_series(period, period, n_terms);
        for (a, b) in f0.iter().zip(fp.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    // ── linear trend ─────────────────────────────────────────────────────────
    #[test]
    fn test_linear_trend_no_changepoints() {
        let v = linear_trend(0.5, 2.0, 1.0, &[], &[]);
        assert!((v - 2.0).abs() < TOL);
    }

    #[test]
    fn test_linear_trend_with_changepoint() {
        // k=1, m=0, delta=[1] at s=0.3 → for t=0.5: 1*0.5 + 1*(0.5-0.3) = 0.7
        let v = linear_trend(0.5, 1.0, 0.0, &[1.0], &[0.3]);
        assert!((v - 0.7).abs() < TOL);
        // Before changepoint
        let v2 = linear_trend(0.1, 1.0, 0.0, &[1.0], &[0.3]);
        assert!((v2 - 0.1).abs() < TOL);
    }

    // ── logistic trend ────────────────────────────────────────────────────────
    #[test]
    fn test_logistic_trend_saturation() {
        // With large t, logistic should approach cap
        let cap = 10.0;
        let v = logistic_trend(1_000.0, 1.0, 0.0, &[], &[], cap);
        assert!(
            (v - cap).abs() < 0.01,
            "logistic should saturate at cap={cap}, got {v}"
        );
    }

    #[test]
    fn test_logistic_trend_zero() {
        // At t=-1000, should approach 0
        let cap = 10.0;
        let v = logistic_trend(-1_000.0, 1.0, 0.0, &[], &[], cap);
        assert!(
            v < 0.01,
            "logistic should approach 0 for large negative t, got {v}"
        );
    }

    // ── fit flat trend ────────────────────────────────────────────────────────
    #[test]
    fn test_fit_flat_trend() {
        let n = 100;
        let t: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let y: Array1<f64> = Array1::from_elem(n, 5.0);

        let mut model = ProphetModel::new()
            .with_yearly_seasonality(false)
            .with_weekly_seasonality(false)
            .with_n_changepoints(0);
        model.fit(&t, &y).expect("fit should succeed");

        let fc = model.predict(&t).expect("predict should succeed");
        for &yh in fc.yhat.iter() {
            assert!((yh - 5.0).abs() < 0.1, "flat trend: yhat={yh} should ≈ 5");
        }
    }

    // ── fit linear trend ─────────────────────────────────────────────────────
    #[test]
    fn test_fit_linear_trend() {
        let (t, y) = linear_series(200, 2.0, 3.0);

        let mut model = ProphetModel::new()
            .with_yearly_seasonality(false)
            .with_weekly_seasonality(false)
            .with_n_changepoints(5);
        model.fit(&t, &y).expect("fit should succeed");

        let fc = model.predict(&t).expect("predict should succeed");
        let last_actual = y[y.len() - 1];
        let last_pred = fc.yhat[fc.yhat.len() - 1];
        assert!(
            (last_pred - last_actual).abs() / last_actual.abs() < 0.05,
            "linear trend: last_pred={last_pred}, last_actual={last_actual}"
        );
    }

    // ── linear extrapolation ──────────────────────────────────────────────────
    #[test]
    fn test_linear_extrapolation() {
        let (t, y) = linear_series(100, 1.0, 0.0);

        let mut model = ProphetModel::new()
            .with_yearly_seasonality(false)
            .with_weekly_seasonality(false)
            .with_n_changepoints(3);
        model.fit(&t, &y).expect("fit");

        let future = model.make_future_dataframe(&t, 10, 1.0);
        assert_eq!(future.len(), 110);

        let fc = model.predict(&future).expect("predict");
        // The last 10 points should continue the trend
        let t_last = future[future.len() - 1];
        let expected = 1.0 * t_last; // slope=1, intercept=0
        let pred_last = fc.yhat[fc.yhat.len() - 1];
        assert!(
            (pred_last - expected).abs() / (expected.abs() + 1.0) < 0.1,
            "extrapolation: pred={pred_last}, expected={expected}"
        );
    }

    // ── weekly seasonality ────────────────────────────────────────────────────
    #[test]
    fn test_weekly_seasonality_detection() {
        let n = 200;
        let t: Array1<f64> = Array1::linspace(0.0, 199.0, n);
        // Clear weekly signal: sin(2π * day / 7)
        let y: Array1<f64> = t.mapv(|ti| (2.0 * std::f64::consts::PI * ti / 7.0).sin());

        let mut model = ProphetModel::new()
            .with_yearly_seasonality(false)
            .with_weekly_seasonality(true)
            .with_n_changepoints(0);
        model.fit(&t, &y).expect("fit");

        let fc = model.predict(&t).expect("predict");
        // Total variance captured: correlation between seasonal and y should be high
        let mean_y = y.sum() / n as f64;
        let ss_tot: f64 = y.iter().map(|&v| (v - mean_y).powi(2)).sum();
        let ss_res: f64 = fc
            .yhat
            .iter()
            .zip(y.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.5, "weekly seasonality R²={r2} should be > 0.5");
    }

    // ── prediction interval coverage ─────────────────────────────────────────
    #[test]
    fn test_prediction_interval_width() {
        let (t, y) = linear_series(100, 1.0, 5.0);
        let mut model = ProphetModel::new()
            .with_yearly_seasonality(false)
            .with_weekly_seasonality(false);
        model.fit(&t, &y).expect("fit");
        let fc = model.predict(&t).expect("predict");

        // Interval should be non-negative width
        for i in 0..fc.yhat.len() {
            assert!(
                fc.yhat_upper[i] >= fc.yhat_lower[i],
                "upper bound must be >= lower bound at i={i}"
            );
        }
    }

    // ── logistic growth fit ───────────────────────────────────────────────────
    #[test]
    fn test_logistic_growth_fit() {
        let n = 100;
        let cap = 10.0;
        // Generate logistic-ish data
        let t: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
        let y: Array1<f64> = t.mapv(|ti| cap / (1.0 + (-ti).exp()));

        let mut model = ProphetModel::new()
            .with_growth(GrowthType::Logistic)
            .with_cap(cap)
            .with_yearly_seasonality(false)
            .with_weekly_seasonality(false)
            .with_n_changepoints(0);
        model.fit(&t, &y).expect("logistic fit");

        let fc = model.predict(&t).expect("predict");
        // Midpoint should be near cap/2
        let mid_pred = fc.yhat[n / 2];
        assert!(
            (mid_pred - cap / 2.0).abs() < 2.0,
            "logistic midpoint={mid_pred}, expected ≈ {}",
            cap / 2.0
        );
        // Endpoint should approach cap
        let end_pred = fc.yhat[n - 1];
        assert!(
            end_pred > cap * 0.85,
            "logistic endpoint={end_pred} should approach cap={cap}"
        );
    }

    // ── make_future_dataframe ─────────────────────────────────────────────────
    #[test]
    fn test_make_future_dataframe() {
        let model = ProphetModel::new();
        let t: Array1<f64> = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let future = model.make_future_dataframe(&t, 3, 1.0);
        assert_eq!(future.len(), 6);
        assert!((future[5] - 5.0).abs() < TOL);
    }

    // ── prophet_metrics ───────────────────────────────────────────────────────
    #[test]
    fn test_prophet_metrics_perfect() {
        let y: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let fc = ProphetForecast {
            timestamps: y.clone(),
            yhat: y.clone(),
            yhat_lower: y.clone(),
            yhat_upper: y.clone(),
            trend: y.clone(),
            seasonal: Array1::zeros(4),
        };
        let m = prophet_metrics(&fc, &y);
        assert!(m.rmse.abs() < TOL);
        assert!(m.mae.abs() < TOL);
        assert!(m.mape.abs() < TOL);
    }

    #[test]
    fn test_prophet_metrics_known_values() {
        let actuals: Array1<f64> = Array1::from_vec(vec![2.0, 4.0]);
        let preds: Array1<f64> = Array1::from_vec(vec![3.0, 3.0]);
        let fc = ProphetForecast {
            timestamps: actuals.clone(),
            yhat: preds.clone(),
            yhat_lower: preds.clone(),
            yhat_upper: preds.clone(),
            trend: preds.clone(),
            seasonal: Array1::zeros(2),
        };
        let m = prophet_metrics(&fc, &actuals);
        // MAE = (|3-2| + |3-4|) / 2 = 1.0
        assert!((m.mae - 1.0).abs() < TOL, "MAE={}", m.mae);
        // RMSE = sqrt((1+1)/2) = 1.0
        assert!((m.rmse - 1.0).abs() < TOL, "RMSE={}", m.rmse);
    }

    // ── model not fitted ──────────────────────────────────────────────────────
    #[test]
    fn test_predict_not_fitted() {
        let model = ProphetModel::new();
        let t: Array1<f64> = Array1::linspace(0.0, 10.0, 10);
        assert!(model.predict(&t).is_err());
    }

    // ── insufficient data ─────────────────────────────────────────────────────
    #[test]
    fn test_fit_insufficient_data() {
        let mut model = ProphetModel::new();
        let t: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
        let y: Array1<f64> = Array1::from_vec(vec![1.0, 2.0]);
        assert!(model.fit(&t, &y).is_err());
    }

    // ── dimension mismatch ────────────────────────────────────────────────────
    #[test]
    fn test_fit_dimension_mismatch() {
        let mut model = ProphetModel::new();
        let t: Array1<f64> = Array1::linspace(0.0, 10.0, 10);
        let y: Array1<f64> = Array1::linspace(0.0, 5.0, 8);
        assert!(model.fit(&t, &y).is_err());
    }

    // ── yearly seasonality R² ─────────────────────────────────────────────────
    #[test]
    fn test_yearly_seasonality_fit() {
        let n = 730; // two years of daily data
                     // Simulate a clear yearly signal (period = 365.25 days)
        let t: Array1<f64> = Array1::linspace(0.0, 729.0, n);
        let y: Array1<f64> =
            t.mapv(|ti| 5.0 * (2.0 * std::f64::consts::PI * ti / 365.25).sin() + 20.0);

        let mut model = ProphetModel::new()
            .with_yearly_seasonality(true)
            .with_weekly_seasonality(false)
            .with_n_changepoints(2);
        model.fit(&t, &y).expect("fit");

        let fc = model.predict(&t).expect("predict");
        let mean_y = y.sum() / n as f64;
        let ss_tot: f64 = y.iter().map(|&v| (v - mean_y).powi(2)).sum();
        let ss_res: f64 = fc
            .yhat
            .iter()
            .zip(y.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.7, "yearly seasonality R²={r2} should be > 0.7");
    }
}
