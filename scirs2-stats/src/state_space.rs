//! State Space Models
//!
//! This module provides linear and nonlinear state space model implementations:
//!
//! - **Kalman Filter** – optimal linear-Gaussian filter with Rauch-Tung-Striebel
//!   smoother and full log-likelihood computation.
//! - **Unscented Kalman Filter (UKF)** – sigma-point method for nonlinear systems.
//! - **Structural Time Series** – local-level (random walk + noise) and local-linear
//!   trend models with Kalman-filter fitting.
//!
//! # Notation
//!
//! The standard state-space model is:
//! ```text
//! x_{t+1} = F * x_t + w_t,   w_t ~ N(0, Q)   (state transition)
//! y_t     = H * x_t + v_t,   v_t ~ N(0, R)   (observation)
//! ```
//!
//! where `x_t ∈ ℝ^n` is the latent state, `y_t ∈ ℝ^m` is the observation,
//! `F ∈ ℝ^{n×n}`, `H ∈ ℝ^{m×n}`, `Q ∈ ℝ^{n×n}`, `R ∈ ℝ^{m×m}`.
//!
//! # References
//! - Kalman, R.E. (1960). "A new approach to linear filtering and prediction problems."
//!   *J. Basic Engineering* 82(1).
//! - Rauch, H.E., Tung, F., & Striebel, C.T. (1965). "Maximum likelihood estimates
//!   of linear dynamic systems." *AIAA J.* 3(8).
//! - Julier, S.J. & Uhlmann, J.K. (1997). "New extension of the Kalman filter to
//!   nonlinear systems." *Proc. SPIE* 3068.
//! - Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─────────────────────────────────────────────────────────────────────────────
// KalmanState
// ─────────────────────────────────────────────────────────────────────────────

/// Kalman filter state: mean vector and covariance matrix.
#[derive(Clone, Debug)]
pub struct KalmanState {
    /// State mean `x` of dimension `n`.
    pub x: Array1<f64>,
    /// State covariance `P` of shape `n × n`.
    pub p: Array2<f64>,
}

impl KalmanState {
    /// Create a new `KalmanState`.
    pub fn new(x: Array1<f64>, p: Array2<f64>) -> StatsResult<Self> {
        let n = x.len();
        if p.nrows() != n || p.ncols() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "KalmanState: x has length {} but P is {}×{}",
                n,
                p.nrows(),
                p.ncols()
            )));
        }
        Ok(Self { x, p })
    }

    /// State dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.x.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kalman Filter
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Kalman filter.
///
/// Provides `predict` and `update` steps that can be composed into a full
/// filtering pass, together with Rauch-Tung-Striebel (RTS) fixed-interval
/// smoothing.
pub struct KalmanFilter;

impl KalmanFilter {
    /// **Predict** step: propagate state through the transition model.
    ///
    /// ```text
    /// x̄_{t+1} = F * x_t
    /// P̄_{t+1} = F * P_t * F^T + Q
    /// ```
    ///
    /// # Arguments
    /// * `state` – current filtered state.
    /// * `f`     – `n × n` state transition matrix.
    /// * `q`     – `n × n` process noise covariance.
    pub fn predict(
        state: &KalmanState,
        f: &Array2<f64>,
        q: &Array2<f64>,
    ) -> StatsResult<KalmanState> {
        let n = state.dim();
        check_square(f, n, "F")?;
        check_square(q, n, "Q")?;

        let x_pred = mat_vec_mul(f, &state.x)?;
        let fp = mat_mat_mul(f, &state.p)?;
        let p_pred = mat_mat_mul_at(&fp, f)? + q;

        KalmanState::new(x_pred, p_pred)
    }

    /// **Update** (correct) step: incorporate a new observation.
    ///
    /// ```text
    /// v   = y - H * x̄            (innovation)
    /// S   = H * P̄ * H^T + R      (innovation covariance)
    /// K   = P̄ * H^T * S^{-1}     (Kalman gain)
    /// x_t = x̄ + K * v
    /// P_t = (I - K*H) * P̄
    /// ```
    ///
    /// # Arguments
    /// * `state` – prior (predicted) state.
    /// * `y`     – observation vector of dimension `m`.
    /// * `h`     – `m × n` observation matrix.
    /// * `r`     – `m × m` observation noise covariance.
    pub fn update(
        state: &KalmanState,
        y: &Array1<f64>,
        h: &Array2<f64>,
        r: &Array2<f64>,
    ) -> StatsResult<KalmanState> {
        let n = state.dim();
        let m = y.len();

        if h.nrows() != m || h.ncols() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "H must be {}×{}, got {}×{}",
                m,
                n,
                h.nrows(),
                h.ncols()
            )));
        }
        if r.nrows() != m || r.ncols() != m {
            return Err(StatsError::DimensionMismatch(format!(
                "R must be {}×{}, got {}×{}",
                m,
                m,
                r.nrows(),
                r.ncols()
            )));
        }

        // Innovation: v = y - H x̄
        let hx = mat_vec_mul(h, &state.x)?;
        let innovation = y - &hx;

        // S = H P̄ H^T + R
        let hp = mat_mat_mul(h, &state.p)?; // m × n
        let s = mat_mat_mul_at(&hp, h)? + r; // m × m

        // K = P̄ H^T S^{-1}
        let ph_t = mat_mat_mul_bt(&state.p, h)?; // n × m
        let s_inv = inv_symmetric(s)?; // m × m
        let k = mat_mat_mul(&ph_t, &s_inv)?; // n × m

        // x updated
        let kv = mat_vec_mul(&k, &innovation)?;
        let x_upd = &state.x + &kv;

        // P updated: (I - K H) P̄  (Joseph form for numerical stability)
        let kh = mat_mat_mul(&k, h)?; // n × n
        let i_kh = eye_minus(kh)?; // n × n
        let p_upd = mat_mat_mul(&i_kh, &state.p)?; // n × n

        KalmanState::new(x_upd, p_upd)
    }

    /// Run the full Kalman filter over an observation time series.
    ///
    /// Observations may be univariate (`y_t ∈ ℝ`) or multivariate
    /// (`y_t ∈ ℝ^m`). Pass each row of your observation matrix as a
    /// separate element in `observations`.
    ///
    /// # Arguments
    /// * `observations` – length-T slice of observation vectors.
    /// * `f`            – `n × n` transition matrix.
    /// * `h`            – `m × n` observation matrix.
    /// * `q`            – `n × n` process noise covariance.
    /// * `r`            – `m × m` observation noise covariance.
    /// * `x0`           – initial state mean.
    /// * `p0`           – initial state covariance.
    ///
    /// # Returns
    /// `(filtered_states, log_likelihood)`.
    pub fn filter_series(
        observations: &[Array1<f64>],
        f: &Array2<f64>,
        h: &Array2<f64>,
        q: &Array2<f64>,
        r: &Array2<f64>,
        x0: Array1<f64>,
        p0: Array2<f64>,
    ) -> StatsResult<(Vec<KalmanState>, f64)> {
        if observations.is_empty() {
            return Err(StatsError::InsufficientData(
                "filter_series: observation list is empty".into(),
            ));
        }

        let n = x0.len();
        let m = observations[0].len();

        if p0.nrows() != n || p0.ncols() != n {
            return Err(StatsError::DimensionMismatch("p0 must be n×n".into()));
        }

        let log2pi = (2.0 * std::f64::consts::PI).ln();
        let mut log_lik = 0.0_f64;
        let mut states = Vec::with_capacity(observations.len());
        let mut state = KalmanState::new(x0, p0)?;

        for (t, y) in observations.iter().enumerate() {
            if y.len() != m {
                return Err(StatsError::DimensionMismatch(format!(
                    "Observation {} has length {}, expected {}",
                    t,
                    y.len(),
                    m
                )));
            }

            // Predict
            let pred = Self::predict(&state, f, q)?;

            // Compute innovation and its covariance for log-likelihood
            let hx = mat_vec_mul(h, &pred.x)?;
            let innovation = y - &hx;
            let hp = mat_mat_mul(h, &pred.p)?;
            let s = mat_mat_mul_at(&hp, h)? + r;

            // Log-likelihood contribution: -0.5 * (m*log2pi + log|S| + v^T S^{-1} v)
            let s_inv = inv_symmetric(s.clone())?;
            let log_det_s = log_det_posdef(&s)?;
            let sv = mat_vec_mul(&s_inv, &innovation)?;
            let quad: f64 = innovation.iter().zip(sv.iter()).map(|(&a, &b)| a * b).sum();
            log_lik += -0.5 * (m as f64 * log2pi + log_det_s + quad);

            // Update
            state = Self::update(&pred, y, h, r)?;
            states.push(state.clone());
        }

        Ok((states, log_lik))
    }

    /// Rauch-Tung-Striebel (RTS) fixed-interval smoother.
    ///
    /// Given a sequence of *filtered* states (output of [`KalmanFilter::filter_series`])
    /// and the transition matrix, computes the smoothed state estimates
    /// E[x_t | y_{1:T}] for all t.
    ///
    /// # Arguments
    /// * `filtered` – filtered states in forward time order.
    /// * `f`        – `n × n` transition matrix.
    /// * `q`        – `n × n` process noise covariance.
    ///
    /// # Returns
    /// Smoothed states in forward time order.
    pub fn smooth(
        filtered: &[KalmanState],
        f: &Array2<f64>,
        q: &Array2<f64>,
    ) -> StatsResult<Vec<KalmanState>> {
        let t_len = filtered.len();
        if t_len == 0 {
            return Ok(Vec::new());
        }

        let mut smoothed = filtered.to_vec();

        for t in (0..t_len - 1).rev() {
            let n = filtered[t].dim();
            // Predicted state at t+1 (given filtered state at t)
            let x_pred = mat_vec_mul(f, &filtered[t].x)?;
            let fp = mat_mat_mul(f, &filtered[t].p)?;
            let p_pred = mat_mat_mul_at(&fp, f)? + q;

            // Smoother gain: G_t = P_t * F^T * P̄_{t+1}^{-1}
            let pf_t = mat_mat_mul_bt(&filtered[t].p, f)?; // n × n
            let p_pred_inv = inv_symmetric(p_pred)?;
            let g = mat_mat_mul(&pf_t, &p_pred_inv)?; // n × n

            // Smoothed mean: x_s_t = x_t + G_t * (x_s_{t+1} - x̄_{t+1})
            let diff = &smoothed[t + 1].x - &x_pred;
            let g_diff = mat_vec_mul(&g, &diff)?;
            let x_smooth = &filtered[t].x + &g_diff;

            // Smoothed covariance: P_s_t = P_t + G_t * (P_s_{t+1} - P̄_{t+1}) * G_t^T
            let dp = &smoothed[t + 1].p
                - &{
                    // Reconstruct P̄_{t+1} from filtered[t]
                    let fp2 = mat_mat_mul(f, &filtered[t].p)?;
                    mat_mat_mul_at(&fp2, f)? + q
                };
            let g_dp = mat_mat_mul(&g, &dp)?;
            let correction = mat_mat_mul_at(&g_dp, &g)?;
            let p_smooth = &filtered[t].p + &correction;

            smoothed[t] = KalmanState::new(x_smooth, p_smooth)?;
        }

        Ok(smoothed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unscented Kalman Filter (UKF)
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the scaled unscented transform.
#[derive(Clone, Debug)]
pub struct UkfParams {
    /// Spread of sigma points around the mean (typically 1e-3).
    pub alpha: f64,
    /// Incorporates prior knowledge of the distribution (typically 0 for Gaussian).
    pub beta: f64,
    /// Secondary scaling parameter (typically 0).
    pub kappa: f64,
}

impl Default for UkfParams {
    fn default() -> Self {
        Self {
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Sigma-point weights for mean and covariance estimation.
#[derive(Clone, Debug)]
struct SigmaWeights {
    /// Weights for mean estimation.
    wm: Vec<f64>,
    /// Weights for covariance estimation.
    wc: Vec<f64>,
    /// Sigma-point scaling factor.
    lambda: f64,
}

impl SigmaWeights {
    fn compute(n: usize, params: &UkfParams) -> Self {
        let n_f = n as f64;
        let lambda = params.alpha * params.alpha * (n_f + params.kappa) - n_f;
        let n_sigma = 2 * n + 1;
        let mut wm = vec![0.0_f64; n_sigma];
        let mut wc = vec![0.0_f64; n_sigma];

        wm[0] = lambda / (n_f + lambda);
        wc[0] = wm[0] + 1.0 - params.alpha * params.alpha + params.beta;

        let w_rest = 1.0 / (2.0 * (n_f + lambda));
        for i in 1..n_sigma {
            wm[i] = w_rest;
            wc[i] = w_rest;
        }

        Self { wm, wc, lambda }
    }
}

/// Generate `2n+1` sigma points from a state (mean + Cholesky of covariance).
///
/// Returns a `Vec<Array1<f64>>` of length `2n+1`.
fn sigma_points(
    state: &KalmanState,
    params: &UkfParams,
    weights: &SigmaWeights,
) -> StatsResult<Vec<Array1<f64>>> {
    let n = state.dim();
    // Scaled covariance matrix: (n + λ) P
    let scale = n as f64 + weights.lambda;
    let scaled_p = state.p.mapv(|v| v * scale);

    // Cholesky decomposition of scaled_p
    let sqrt_p = cholesky_lower(scaled_p)?;

    let mut sigmas = Vec::with_capacity(2 * n + 1);
    sigmas.push(state.x.clone()); // σ_0 = x̄

    for i in 0..n {
        let col = sqrt_p.column(i).to_owned();
        sigmas.push(&state.x + &col); // σ_{i+1}   = x̄ + sqrt_col_i
        sigmas.push(&state.x - &col); // σ_{n+i+1} = x̄ - sqrt_col_i
    }

    Ok(sigmas)
}

/// Unscented Kalman Filter for nonlinear systems.
pub struct UnscentedKalmanFilter {
    /// Sigma-point parameters.
    pub params: UkfParams,
}

impl UnscentedKalmanFilter {
    /// Create a new UKF with the given sigma-point parameters.
    pub fn new(params: UkfParams) -> Self {
        Self { params }
    }

    /// Create a new UKF with default sigma-point parameters.
    pub fn default() -> Self {
        Self {
            params: UkfParams::default(),
        }
    }

    /// **Predict** step for a nonlinear transition function `f_fn(x) -> x'`.
    ///
    /// # Arguments
    /// * `state`  – current posterior state.
    /// * `f_fn`   – state transition function `x_t -> x_{t+1}`.
    /// * `q`      – process noise covariance (n × n).
    pub fn predict<F>(
        &self,
        state: &KalmanState,
        f_fn: F,
        q: &Array2<f64>,
    ) -> StatsResult<KalmanState>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let n = state.dim();
        let weights = SigmaWeights::compute(n, &self.params);
        let sigmas = sigma_points(state, &self.params, &weights)?;

        // Propagate sigma points through f
        let propagated: Vec<Array1<f64>> = sigmas.iter().map(|s| f_fn(s)).collect();

        // Compute predicted mean
        let x_pred = weighted_mean(&propagated, &weights.wm)?;

        // Compute predicted covariance
        let p_pred = weighted_covariance(&propagated, &x_pred, &weights.wc, Some(q))?;

        KalmanState::new(x_pred, p_pred)
    }

    /// **Update** step for a nonlinear observation function `h_fn(x) -> y`.
    ///
    /// # Arguments
    /// * `state`  – prior (predicted) state.
    /// * `y`      – actual observation vector.
    /// * `h_fn`   – observation function `x -> y`.
    /// * `r`      – observation noise covariance (m × m).
    pub fn update<H>(
        &self,
        state: &KalmanState,
        y: &Array1<f64>,
        h_fn: H,
        r: &Array2<f64>,
    ) -> StatsResult<KalmanState>
    where
        H: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let n = state.dim();
        let weights = SigmaWeights::compute(n, &self.params);
        let sigmas = sigma_points(state, &self.params, &weights)?;

        // Propagate sigma points through h
        let y_sigmas: Vec<Array1<f64>> = sigmas.iter().map(|s| h_fn(s)).collect();

        // Predicted measurement mean
        let y_pred = weighted_mean(&y_sigmas, &weights.wm)?;

        // Innovation covariance S_yy = Σ wc_i (z_i - ȳ)(z_i - ȳ)^T + R
        let s_yy = weighted_covariance(&y_sigmas, &y_pred, &weights.wc, Some(r))?;

        // Cross covariance P_xy = Σ wc_i (σ_i - x̄)(z_i - ȳ)^T
        let p_xy = weighted_cross_covariance(&sigmas, &state.x, &y_sigmas, &y_pred, &weights.wc)?;

        // Kalman gain K = P_xy * S_yy^{-1}
        let s_inv = inv_symmetric(s_yy)?;
        let k = mat_mat_mul(&p_xy, &s_inv)?; // n × m

        // Update mean
        let innovation = y - &y_pred;
        let kv = mat_vec_mul(&k, &innovation)?;
        let x_upd = &state.x + &kv;

        // Update covariance P = P̄ - K * S_yy * K^T
        let ks = mat_mat_mul(&k, &{
            let s_yy2 = weighted_covariance(&y_sigmas, &y_pred, &weights.wc, Some(r))?;
            s_yy2
        })?;
        let correction = mat_mat_mul_at(&ks, &k)?;
        let p_upd = &state.p - &correction;

        KalmanState::new(x_upd, p_upd)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural Time Series
// ─────────────────────────────────────────────────────────────────────────────

/// Structural Time Series model variant.
#[derive(Clone, Debug, PartialEq)]
pub enum StsModel {
    /// Local level model: random walk + observation noise.
    LocalLevel {
        /// Variance of the level innovation σ²_η.
        level_variance: f64,
        /// Variance of the observation noise σ²_ε.
        obs_variance: f64,
    },
    /// Local linear trend model: level + slope random walk.
    LocalLinearTrend {
        /// Variance of level innovation σ²_η.
        level_variance: f64,
        /// Variance of slope innovation σ²_ζ.
        slope_variance: f64,
        /// Variance of observation noise σ²_ε.
        obs_variance: f64,
    },
}

/// Result of fitting a Structural Time Series model.
#[derive(Clone, Debug)]
pub struct StsFitResult {
    /// The fitted model variant.
    pub model: StsModel,
    /// Smoothed state estimates (one per time point).
    pub smoothed_states: Vec<KalmanState>,
    /// Filtered state estimates (one per time point).
    pub filtered_states: Vec<KalmanState>,
    /// Log-likelihood of the observations under the model.
    pub log_likelihood: f64,
    /// One-step-ahead forecasts (level component).
    pub fitted_values: Vec<f64>,
    /// Standardised prediction residuals.
    pub residuals: Vec<f64>,
}

/// Structural Time Series builder and fitter.
pub struct StructuralTimeSeries;

impl StructuralTimeSeries {
    /// Fit a **local level** model to a univariate time series.
    ///
    /// The local level model is:
    /// ```text
    /// y_t = μ_t + ε_t,  ε_t ~ N(0, σ²_ε)
    /// μ_{t+1} = μ_t + η_t,  η_t ~ N(0, σ²_η)
    /// ```
    ///
    /// State `x_t = [μ_t]`.
    ///
    /// # Arguments
    /// * `y`             – univariate time series (length T).
    /// * `level_var`     – process noise variance σ²_η (must be > 0).
    /// * `obs_var`       – observation noise variance σ²_ε (must be > 0).
    /// * `init_level`    – initial level estimate (defaults to `y[0]`).
    /// * `init_var`      – initial state variance (defaults to 1e6 for diffuse init).
    pub fn fit_local_level(
        y: &[f64],
        level_var: f64,
        obs_var: f64,
        init_level: Option<f64>,
        init_var: Option<f64>,
    ) -> StatsResult<StsFitResult> {
        if y.is_empty() {
            return Err(StatsError::InsufficientData(
                "Time series must not be empty".into(),
            ));
        }
        if level_var <= 0.0 {
            return Err(StatsError::DomainError("level_var must be positive".into()));
        }
        if obs_var <= 0.0 {
            return Err(StatsError::DomainError("obs_var must be positive".into()));
        }

        use scirs2_core::ndarray::{array, Array1, Array2};

        // State: [level]
        let f = array![[1.0_f64]]; // transition
        let h = array![[1.0_f64]]; // observation
        let q = array![[level_var]]; // process noise
        let r = array![[obs_var]]; // observation noise

        let x0 = Array1::from_elem(1, init_level.unwrap_or(y[0]));
        let p0 = Array2::from_elem((1, 1), init_var.unwrap_or(1e6));

        let obs_vecs: Vec<Array1<f64>> = y.iter().map(|&yi| Array1::from_elem(1, yi)).collect();

        let (filtered, log_lik) = KalmanFilter::filter_series(&obs_vecs, &f, &h, &q, &r, x0, p0)?;

        let smoothed = KalmanFilter::smooth(&filtered, &f, &q)?;

        let fitted_values: Vec<f64> = filtered.iter().map(|s| s.x[0]).collect();
        let residuals: Vec<f64> = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(&yi, &fi)| (yi - fi) / obs_var.sqrt())
            .collect();

        Ok(StsFitResult {
            model: StsModel::LocalLevel {
                level_variance: level_var,
                obs_variance: obs_var,
            },
            smoothed_states: smoothed,
            filtered_states: filtered,
            log_likelihood: log_lik,
            fitted_values,
            residuals,
        })
    }

    /// Fit a **local linear trend** model to a univariate time series.
    ///
    /// The local linear trend model is:
    /// ```text
    /// y_t     = μ_t + ε_t,       ε_t  ~ N(0, σ²_ε)
    /// μ_{t+1} = μ_t + ν_t + η_t, η_t  ~ N(0, σ²_η)
    /// ν_{t+1} = ν_t + ζ_t,       ζ_t  ~ N(0, σ²_ζ)
    /// ```
    ///
    /// State `x_t = [μ_t, ν_t]`.
    ///
    /// # Arguments
    /// * `y`           – univariate time series.
    /// * `level_var`   – level innovation variance σ²_η.
    /// * `slope_var`   – slope innovation variance σ²_ζ.
    /// * `obs_var`     – observation noise variance σ²_ε.
    /// * `init_level`  – initial level (defaults to `y[0]`).
    /// * `init_slope`  – initial slope (defaults to 0).
    /// * `init_var`    – initial state variance (defaults to 1e6).
    pub fn fit_local_linear_trend(
        y: &[f64],
        level_var: f64,
        slope_var: f64,
        obs_var: f64,
        init_level: Option<f64>,
        init_slope: Option<f64>,
        init_var: Option<f64>,
    ) -> StatsResult<StsFitResult> {
        if y.is_empty() {
            return Err(StatsError::InsufficientData(
                "Time series must not be empty".into(),
            ));
        }
        if level_var < 0.0 || slope_var < 0.0 {
            return Err(StatsError::DomainError(
                "Variance parameters must be non-negative".into(),
            ));
        }
        if obs_var <= 0.0 {
            return Err(StatsError::DomainError("obs_var must be positive".into()));
        }

        use scirs2_core::ndarray::{array, Array1, Array2};

        // State: [level, slope]
        let f = array![[1.0_f64, 1.0], [0.0, 1.0]];
        let h = array![[1.0_f64, 0.0]];
        let q = array![[level_var, 0.0], [0.0, slope_var]];
        let r = array![[obs_var]];

        let iv = init_var.unwrap_or(1e6);
        let x0 = Array1::from_vec(vec![init_level.unwrap_or(y[0]), init_slope.unwrap_or(0.0)]);
        let p0 = Array2::from_diag(&Array1::from_vec(vec![iv, iv]));

        let obs_vecs: Vec<Array1<f64>> = y.iter().map(|&yi| Array1::from_elem(1, yi)).collect();

        let (filtered, log_lik) = KalmanFilter::filter_series(&obs_vecs, &f, &h, &q, &r, x0, p0)?;

        let smoothed = KalmanFilter::smooth(&filtered, &f, &q)?;

        let fitted_values: Vec<f64> = filtered.iter().map(|s| s.x[0]).collect();
        let residuals: Vec<f64> = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(&yi, &fi)| (yi - fi) / obs_var.sqrt())
            .collect();

        Ok(StsFitResult {
            model: StsModel::LocalLinearTrend {
                level_variance: level_var,
                slope_variance: slope_var,
                obs_variance: obs_var,
            },
            smoothed_states: smoothed,
            filtered_states: filtered,
            log_likelihood: log_lik,
            fitted_values,
            residuals,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical linear-algebra helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix-vector product y = A * x.
fn mat_vec_mul(a: &Array2<f64>, x: &Array1<f64>) -> StatsResult<Array1<f64>> {
    if a.ncols() != x.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_vec_mul: A is {}×{} but x has len {}",
            a.nrows(),
            a.ncols(),
            x.len()
        )));
    }
    let n = a.nrows();
    let m = a.ncols();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for k in 0..m {
            s += a[[i, k]] * x[k];
        }
        y[i] = s;
    }
    Ok(y)
}

/// Matrix-matrix product C = A * B.
fn mat_mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    if a.ncols() != b.nrows() {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_mat_mul: A is {}×{} but B is {}×{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }
    let n = a.nrows();
    let k = a.ncols();
    let m = b.ncols();
    let mut c = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

/// Compute C = A * B^T.
fn mat_mat_mul_bt(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_mat_mul_bt: A is {}×{} but B^T is {}×{}",
            a.nrows(),
            a.ncols(),
            b.ncols(),
            b.nrows()
        )));
    }
    let n = a.nrows();
    let k = a.ncols();
    let m = b.nrows();
    let mut c = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[[i, l]] * b[[j, l]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

/// Compute C = A * B^T (where A is a result matrix and B is the original).
/// This is a synonym for mat_mat_mul_bt used in RTS smoother notation.
#[allow(dead_code)]
fn mat_mat_mul_at(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    // C = A * B^T
    mat_mat_mul_bt(a, b)
}

/// Compute I - A (identity minus matrix), check square.
fn eye_minus(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "eye_minus: matrix not square".into(),
        ));
    }
    let mut result = -a;
    for i in 0..n {
        result[[i, i]] += 1.0;
    }
    Ok(result)
}

/// Assert that a matrix is square with the expected dimension.
fn check_square(m: &Array2<f64>, expected: usize, name: &str) -> StatsResult<()> {
    if m.nrows() != expected || m.ncols() != expected {
        Err(StatsError::DimensionMismatch(format!(
            "{} must be {}×{}, got {}×{}",
            name,
            expected,
            expected,
            m.nrows(),
            m.ncols()
        )))
    } else {
        Ok(())
    }
}

/// Invert a symmetric positive-definite matrix using Cholesky decomposition.
fn inv_symmetric(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "inv_symmetric: matrix must be square".into(),
        ));
    }

    if n == 1 {
        let val = a[[0, 0]];
        if val.abs() < 1e-15 {
            return Err(StatsError::ComputationError(
                "inv_symmetric: 1×1 matrix is singular".into(),
            ));
        }
        let mut inv = Array2::<f64>::zeros((1, 1));
        inv[[0, 0]] = 1.0 / val;
        return Ok(inv);
    }

    // Cholesky L such that A = L L^T
    let l = cholesky_lower(a)?;

    // Invert L by forward substitution, then L^{-T} = (L^{-1})^T
    let l_inv = lower_tri_inv(&l)?;

    // A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1} = (L^{-1})^T * L^{-1}
    let l_inv_t = l_inv.t().to_owned();
    mat_mat_mul(&l_inv_t, &l_inv)
}

/// Compute the lower Cholesky factor L of a positive-definite symmetric matrix A.
fn cholesky_lower(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "cholesky_lower: matrix must be square".into(),
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    // Apply a small regularisation for near-PSD matrices
                    let eps = 1e-10_f64.max(s.abs() * 1e-8);
                    let s_reg = s + eps;
                    if s_reg <= 0.0 {
                        return Err(StatsError::ComputationError(format!(
                            "Cholesky failed at ({},{}): diagonal entry {} is non-positive",
                            i, j, s
                        )));
                    }
                    l[[i, j]] = s_reg.sqrt();
                } else {
                    l[[i, j]] = s.sqrt();
                }
            } else {
                let ljj = l[[j, j]];
                if ljj.abs() < 1e-15 {
                    return Err(StatsError::ComputationError(
                        "Cholesky: near-zero diagonal encountered".into(),
                    ));
                }
                l[[i, j]] = s / ljj;
            }
        }
    }

    Ok(l)
}

/// Invert a lower-triangular matrix L by forward substitution.
fn lower_tri_inv(l: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = l.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));

    for j in 0..n {
        inv[[j, j]] = 1.0 / l[[j, j]];
        for i in j + 1..n {
            let mut s = 0.0_f64;
            for k in j..i {
                s += l[[i, k]] * inv[[k, j]];
            }
            inv[[i, j]] = -s / l[[i, i]];
        }
    }

    Ok(inv)
}

/// Compute the log-determinant of a symmetric positive-definite matrix.
fn log_det_posdef(a: &Array2<f64>) -> StatsResult<f64> {
    let l = cholesky_lower(a.clone())?;
    let n = l.nrows();
    let log_det: f64 = (0..n).map(|i| 2.0 * l[[i, i]].ln()).sum();
    Ok(log_det)
}

// ─────────────────────────────────────────────────────────────────────────────
// UKF utilities (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the weighted mean of a set of vectors.
fn weighted_mean(vecs: &[Array1<f64>], weights: &[f64]) -> StatsResult<Array1<f64>> {
    if vecs.is_empty() {
        return Err(StatsError::InsufficientData(
            "weighted_mean: empty vector set".into(),
        ));
    }
    if vecs.len() != weights.len() {
        return Err(StatsError::DimensionMismatch(
            "weighted_mean: vecs and weights must have same length".into(),
        ));
    }
    let d = vecs[0].len();
    let mut mean = Array1::<f64>::zeros(d);
    for (v, &w) in vecs.iter().zip(weights.iter()) {
        mean = mean + v.mapv(|x| x * w);
    }
    Ok(mean)
}

/// Compute the weighted covariance of a set of vectors around a given mean.
/// If `additive` is `Some(matrix)`, it is added to the covariance.
fn weighted_covariance(
    vecs: &[Array1<f64>],
    mean: &Array1<f64>,
    weights: &[f64],
    additive: Option<&Array2<f64>>,
) -> StatsResult<Array2<f64>> {
    let d = mean.len();
    let mut cov = Array2::<f64>::zeros((d, d));
    for (v, &w) in vecs.iter().zip(weights.iter()) {
        let diff = v - mean;
        for i in 0..d {
            for j in 0..d {
                cov[[i, j]] += w * diff[i] * diff[j];
            }
        }
    }
    if let Some(add) = additive {
        cov = cov + add;
    }
    Ok(cov)
}

/// Compute the weighted cross-covariance between two sets of vectors.
fn weighted_cross_covariance(
    xs: &[Array1<f64>],
    x_mean: &Array1<f64>,
    ys: &[Array1<f64>],
    y_mean: &Array1<f64>,
    weights: &[f64],
) -> StatsResult<Array2<f64>> {
    let dx = x_mean.len();
    let dy = y_mean.len();
    if xs.len() != ys.len() || xs.len() != weights.len() {
        return Err(StatsError::DimensionMismatch(
            "weighted_cross_covariance: dimension mismatch".into(),
        ));
    }
    let mut cov = Array2::<f64>::zeros((dx, dy));
    for ((x, y), &w) in xs.iter().zip(ys.iter()).zip(weights.iter()) {
        let xd = x - x_mean;
        let yd = y - y_mean;
        for i in 0..dx {
            for j in 0..dy {
                cov[[i, j]] += w * xd[i] * yd[j];
            }
        }
    }
    Ok(cov)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    // ── Kalman filter ──────────────────────────────────────────────────────

    #[test]
    fn test_kalman_predict() {
        // 1-D constant velocity example
        let x = Array1::from_vec(vec![1.0, 0.5]);
        let p = Array2::from_diag(&Array1::from_vec(vec![0.1, 0.1]));
        let state = KalmanState::new(x, p).expect("state");

        let f = array![[1.0, 1.0], [0.0, 1.0]];
        let q = array![[0.01, 0.0], [0.0, 0.01]];

        let pred = KalmanFilter::predict(&state, &f, &q).expect("predict");
        // x_pred = F * [1, 0.5] = [1.5, 0.5]
        assert!((pred.x[0] - 1.5).abs() < 1e-12);
        assert!((pred.x[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_kalman_update_reduces_variance() {
        let x0 = Array1::from_vec(vec![0.0]);
        let p0 = Array2::from_elem((1, 1), 1.0);
        let state = KalmanState::new(x0, p0).expect("state");

        let h = array![[1.0_f64]];
        let r = array![[0.1_f64]];
        let y = Array1::from_vec(vec![1.0_f64]);

        let upd = KalmanFilter::update(&state, &y, &h, &r).expect("update");
        // Posterior variance < prior variance
        assert!(upd.p[[0, 0]] < 1.0);
        // Posterior mean should shift toward observation
        assert!(upd.x[0] > 0.0);
    }

    #[test]
    fn test_filter_series_scalar() {
        // Constant signal + noise
        let obs: Vec<Array1<f64>> = (0..20)
            .map(|i| Array1::from_vec(vec![3.0 + 0.1 * (i as f64 % 3.0 - 1.0)]))
            .collect();

        let f = array![[1.0_f64]];
        let h = array![[1.0_f64]];
        let q = array![[0.01_f64]];
        let r = array![[0.5_f64]];
        let x0 = Array1::from_vec(vec![0.0_f64]);
        let p0 = Array2::from_elem((1, 1), 100.0_f64);

        let (states, log_lik) =
            KalmanFilter::filter_series(&obs, &f, &h, &q, &r, x0, p0).expect("filter_series");
        assert_eq!(states.len(), 20);
        assert!(log_lik.is_finite());
        // After 20 observations the estimate should be close to 3.0
        let final_est = states.last().map(|s| s.x[0]).unwrap_or(0.0);
        assert!((final_est - 3.0).abs() < 0.5, "final_est = {}", final_est);
    }

    #[test]
    fn test_rts_smoother() {
        let obs: Vec<Array1<f64>> = (0..10).map(|_| Array1::from_vec(vec![2.0_f64])).collect();

        let f = array![[1.0_f64]];
        let h = array![[1.0_f64]];
        let q = array![[0.1_f64]];
        let r = array![[1.0_f64]];
        let x0 = Array1::from_vec(vec![0.0_f64]);
        let p0 = Array2::from_elem((1, 1), 10.0_f64);

        let (filtered, _) =
            KalmanFilter::filter_series(&obs, &f, &h, &q, &r, x0, p0).expect("filter_series");
        let smoothed = KalmanFilter::smooth(&filtered, &f, &q).expect("smooth");

        assert_eq!(smoothed.len(), 10);
        // Smoother should give tighter (or equal) uncertainty
        let filtered_var = filtered[5].p[[0, 0]];
        let smoothed_var = smoothed[5].p[[0, 0]];
        assert!(smoothed_var <= filtered_var + 1e-10);
    }

    // ── Structural time series ─────────────────────────────────────────────

    #[test]
    fn test_local_level_fit() {
        let y: Vec<f64> = (0..30)
            .map(|i| 2.0 + 0.1 * (i as f64 % 5.0 - 2.0))
            .collect();
        let result = StructuralTimeSeries::fit_local_level(&y, 0.1, 0.5, None, None)
            .expect("fit_local_level");
        assert_eq!(result.fitted_values.len(), 30);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_local_linear_trend_fit() {
        let y: Vec<f64> = (0..30)
            .map(|i| i as f64 + 0.1 * (i as f64 % 3.0 - 1.0))
            .collect();
        let result =
            StructuralTimeSeries::fit_local_linear_trend(&y, 0.01, 0.001, 0.5, None, None, None)
                .expect("fit_local_linear_trend");
        assert_eq!(result.fitted_values.len(), 30);
        assert!(result.log_likelihood.is_finite());
        // Slope component should be approximately 1.0 by the end
        let final_slope = result.smoothed_states.last().map(|s| s.x[1]).unwrap_or(0.0);
        assert!(final_slope > 0.5, "slope = {}", final_slope);
    }

    // ── UKF ───────────────────────────────────────────────────────────────

    #[test]
    fn test_ukf_linear_matches_kf() {
        // For a linear system, UKF should closely match the Kalman filter.
        let x0 = Array1::from_vec(vec![0.0_f64]);
        let p0 = Array2::from_elem((1, 1), 1.0_f64);
        let f_mat = array![[1.0_f64]];
        let h_mat = array![[1.0_f64]];
        let q_mat = array![[0.1_f64]];
        let r_mat = array![[0.5_f64]];
        let y = Array1::from_vec(vec![1.0_f64]);

        let state = KalmanState::new(x0.clone(), p0.clone()).expect("state");

        // KF result
        let kf_pred = KalmanFilter::predict(&state, &f_mat, &q_mat).expect("kf_pred");
        let kf_upd = KalmanFilter::update(&kf_pred, &y, &h_mat, &r_mat).expect("kf_upd");

        // UKF result (linear functions)
        let ukf = UnscentedKalmanFilter::default();
        let f_fn = |x: &Array1<f64>| x.clone(); // identity transition
        let h_fn = |x: &Array1<f64>| x.clone(); // identity observation

        let ukf_pred = ukf.predict(&state, f_fn, &q_mat).expect("ukf_pred");
        let ukf_upd = ukf.update(&ukf_pred, &y, h_fn, &r_mat).expect("ukf_upd");

        // Means should match closely
        assert!(
            (kf_upd.x[0] - ukf_upd.x[0]).abs() < 1e-6,
            "KF x={}, UKF x={}",
            kf_upd.x[0],
            ukf_upd.x[0]
        );
    }

    // ── Numerical helpers ──────────────────────────────────────────────────

    #[test]
    fn test_cholesky_correctness() {
        // A = [[4, 2], [2, 3]] → L = [[2,0],[1, sqrt(2)]]
        let a = array![[4.0_f64, 2.0], [2.0, 3.0]];
        let l = cholesky_lower(a.clone()).expect("cholesky");
        // Verify L L^T ≈ A
        let lt = l.t().to_owned();
        let a_reconstructed = mat_mat_mul(&l, &lt).expect("mul");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a_reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-12,
                    "({},{}) mismatch",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_inv_symmetric_2x2() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let a_inv = inv_symmetric(a.clone()).expect("inv");
        // A * A^{-1} should be ~ I
        let prod = mat_mat_mul(&a, &a_inv).expect("mul");
        assert!((prod[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((prod[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(prod[[0, 1]].abs() < 1e-10);
        assert!(prod[[1, 0]].abs() < 1e-10);
    }
}
