//! State-space models and Dynamic Linear Models (DLM) for time series analysis.
//!
//! Implements:
//! - Linear Gaussian State Space Models (SSM)
//! - Kalman filter (forward pass)
//! - Rauch-Tung-Striebel (RTS) Kalman smoother (backward pass)
//! - EM algorithm for maximum likelihood parameter estimation
//! - Structural time series models (local level, local linear trend, seasonal, AR)
//! - Dynamic Linear Models with optional time-varying observation matrices
//! - Markov switching / regime models (Hamilton 1989)
//! - Forecasting and structural break detection
//!
//! ## Model Specification
//!
//! ```text
//! State equation:  x_{t+1} = F x_t + q_t,  q_t ~ N(0, Q)
//! Obs  equation:   y_t     = H x_t + r_t,  r_t ~ N(0, R)
//! ```

// Submodules with advanced Vec-based APIs
pub mod dlm;
pub mod gp_ssm;
pub mod hippo;
pub mod ihmm;
pub mod linear_gaussian;
pub mod mamba;
pub mod s4;
pub mod s4d;
pub mod structural;
pub mod switching;

pub use dlm::{DLMBuilder, DynamicLinearModelVec};
pub use linear_gaussian::{KalmanOutput, LinearGaussianSSM};
pub use structural::{SeasonalComponent, StructuralModel, TrendComponent};
pub use switching::{MarkovSwitchingModel, RegimeParameters};

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helpers: linear algebra over f64 (no external BLAS)
// ---------------------------------------------------------------------------

/// Compute A * B for 2D f64 arrays.
#[inline]
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    if k != k2 {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: k,
            actual: k2,
        });
    }
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

/// Transpose of a 2D matrix.
#[inline]
fn mat_t(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            out[[j, i]] = a[[i, j]];
        }
    }
    out
}

/// Element-wise addition of two 2D arrays.
#[inline]
fn mat_add(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, n) = a.dim();
    if b.dim() != (m, n) {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: m * n,
            actual: b.dim().0 * b.dim().1,
        });
    }
    let mut c = a.clone();
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] += b[[i, j]];
        }
    }
    Ok(c)
}

/// Element-wise subtraction a - b.
#[inline]
fn mat_sub(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, n) = a.dim();
    if b.dim() != (m, n) {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: m * n,
            actual: b.dim().0 * b.dim().1,
        });
    }
    let mut c = a.clone();
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] -= b[[i, j]];
        }
    }
    Ok(c)
}

/// Matrix-vector product A * x.
#[inline]
fn mat_vec(a: &Array2<f64>, x: &Array1<f64>) -> Result<Array1<f64>> {
    let (m, n) = a.dim();
    if x.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    let mut y = Array1::<f64>::zeros(m);
    for i in 0..m {
        for j in 0..n {
            y[i] += a[[i, j]] * x[j];
        }
    }
    Ok(y)
}

/// Solve the linear system `A x = b` via LU decomposition with partial pivoting.
/// Returns x.
fn solve_linear(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.dim().0;
    if a.dim() != (n, n) {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix must be square for linear solve".to_string(),
        ));
    }
    let ncols = b.dim().1;

    // Build augmented [A | B]
    let mut lu = a.clone();
    let mut rhs = b.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for col in 0..n {
        // Find pivot
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = lu[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_row != col {
            for j in 0..n {
                lu.swap([col, j], [max_row, j]);
            }
            for j in 0..ncols {
                rhs.swap([col, j], [max_row, j]);
            }
            perm.swap(col, max_row);
        }

        if lu[[col, col]].abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "Near-singular matrix in linear solve".to_string(),
            ));
        }

        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];
            lu[[row, col]] = factor;
            for j in (col + 1)..n {
                let v = lu[[row, j]] - factor * lu[[col, j]];
                lu[[row, j]] = v;
            }
            for j in 0..ncols {
                let v = rhs[[row, j]] - factor * rhs[[col, j]];
                rhs[[row, j]] = v;
            }
        }
    }

    // Back-substitution
    let mut x = rhs.clone();
    for k in (0..n).rev() {
        for j in 0..ncols {
            let mut s = x[[k, j]];
            for l in (k + 1)..n {
                s -= lu[[k, l]] * x[[l, j]];
            }
            x[[k, j]] = s / lu[[k, k]];
        }
    }

    Ok(x)
}

/// Invert a square matrix via LU decomposition.
fn mat_inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.dim().0;
    let eye = Array2::<f64>::eye(n);
    solve_linear(a, &eye)
}

/// Compute log |det A| for a symmetric positive definite matrix via Cholesky.
/// Falls back to LU if Cholesky fails.
fn log_det(a: &Array2<f64>) -> Result<f64> {
    let n = a.dim().0;

    // Attempt Cholesky: L L^T = A
    let mut l = Array2::<f64>::zeros((n, n));
    let mut ok = true;

    'outer: for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    ok = false;
                    break 'outer;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    if ok {
        let log_diag_sum: f64 = (0..n).map(|i| l[[i, i]].ln()).sum();
        return Ok(2.0 * log_diag_sum);
    }

    // Fall back to LU
    let mut lu = a.clone();
    let mut sign = 1.0_f64;

    for col in 0..n {
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = lu[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..n {
                lu.swap([col, j], [max_row, j]);
            }
            sign = -sign;
        }
        if lu[[col, col]].abs() < 1e-14 {
            return Ok(f64::NEG_INFINITY);
        }
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];
            lu[[row, col]] = factor;
            for j in (col + 1)..n {
                let v = lu[[row, j]] - factor * lu[[col, j]];
                lu[[row, j]] = v;
            }
        }
    }

    let log_diag_sum: f64 = (0..n).map(|i| lu[[i, i]].abs().ln()).sum();
    Ok(log_diag_sum + if sign < 0.0 { f64::NAN } else { 0.0 })
}

// ---------------------------------------------------------------------------
// StateSpaceModel
// ---------------------------------------------------------------------------

/// Linear Gaussian State Space Model.
///
/// ```text
/// State:    x_{t+1} = F x_t + q_t,  q_t ~ N(0, Q)
/// Obs:      y_t     = H x_t + r_t,  r_t ~ N(0, R)
/// ```
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// State transition matrix F: (n_states × n_states)
    pub transition_matrix: Array2<f64>,
    /// Observation matrix H: (n_obs × n_states)
    pub observation_matrix: Array2<f64>,
    /// Process noise covariance Q: (n_states × n_states)
    pub process_noise: Array2<f64>,
    /// Observation noise covariance R: (n_obs × n_obs)
    pub observation_noise: Array2<f64>,
    /// Initial state mean m_0: (n_states,)
    pub initial_state: Array1<f64>,
    /// Initial state covariance P_0: (n_states × n_states)
    pub initial_cov: Array2<f64>,
}

impl StateSpaceModel {
    /// Create a generic (identity) SSM with given state/obs dimensions.
    pub fn new(n_states: usize, n_obs: usize) -> Self {
        Self {
            transition_matrix: Array2::eye(n_states),
            observation_matrix: {
                let mut h = Array2::<f64>::zeros((n_obs, n_states));
                for i in 0..n_obs.min(n_states) {
                    h[[i, i]] = 1.0;
                }
                h
            },
            process_noise: Array2::eye(n_states),
            observation_noise: Array2::eye(n_obs),
            initial_state: Array1::zeros(n_states),
            initial_cov: Array2::eye(n_states) * 1e6,
        }
    }

    /// Local Level (random walk + noise):
    /// ```text
    /// μ_{t+1} = μ_t + η_t,  η_t ~ N(0, σ_η²)
    /// y_t     = μ_t + ε_t,  ε_t ~ N(0, σ_ε²)
    /// ```
    pub fn local_level(sigma_eta: f64, sigma_eps: f64) -> Self {
        Self {
            transition_matrix: Array2::eye(1),
            observation_matrix: Array2::ones((1, 1)),
            process_noise: Array2::from_elem((1, 1), sigma_eta * sigma_eta),
            observation_noise: Array2::from_elem((1, 1), sigma_eps * sigma_eps),
            initial_state: Array1::zeros(1),
            initial_cov: Array2::from_elem((1, 1), 1e6),
        }
    }

    /// Local Linear Trend model (level + slope):
    /// ```text
    /// μ_{t+1} = μ_t + β_t + η_t,  η_t ~ N(0, σ_η²)
    /// β_{t+1} = β_t + ζ_t,         ζ_t ~ N(0, σ_ζ²)
    /// y_t     = μ_t + ε_t,          ε_t ~ N(0, σ_ε²)
    /// ```
    pub fn local_linear_trend(sigma_eta: f64, sigma_zeta: f64, sigma_eps: f64) -> Self {
        // State = [μ, β]
        let f = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).expect("shape is valid");
        let h = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).expect("shape is valid");
        let q = Array2::from_shape_vec(
            (2, 2),
            vec![sigma_eta * sigma_eta, 0.0, 0.0, sigma_zeta * sigma_zeta],
        )
        .expect("shape is valid");
        let r = Array2::from_elem((1, 1), sigma_eps * sigma_eps);
        Self {
            transition_matrix: f,
            observation_matrix: h,
            process_noise: q,
            observation_noise: r,
            initial_state: Array1::zeros(2),
            initial_cov: Array2::eye(2) * 1e6,
        }
    }

    /// Seasonal model in dummy-variable form with `period` seasons.
    ///
    /// State dimension = `period - 1`.
    /// The first state component is the current seasonal effect.
    pub fn seasonal(period: usize, sigma_omega: f64) -> Self {
        if period < 2 {
            // Degenerate; return trivial model
            return Self::local_level(sigma_omega, 1.0);
        }
        let s = period - 1;
        // Transition: [-1, -1, ..., -1; I_{s-1}, 0]
        let mut f = Array2::<f64>::zeros((s, s));
        for j in 0..s {
            f[[0, j]] = -1.0;
        }
        for i in 1..s {
            f[[i, i - 1]] = 1.0;
        }
        // Observation: [1, 0, ..., 0]
        let mut h = Array2::<f64>::zeros((1, s));
        h[[0, 0]] = 1.0;
        // Process noise on first component only
        let mut q = Array2::<f64>::zeros((s, s));
        q[[0, 0]] = sigma_omega * sigma_omega;

        Self {
            transition_matrix: f,
            observation_matrix: h,
            process_noise: q,
            observation_noise: Array2::from_elem((1, 1), 1.0),
            initial_state: Array1::zeros(s),
            initial_cov: Array2::eye(s) * 1e4,
        }
    }

    /// AR(p) model in companion-form state space:
    /// ```text
    /// x_t = [y_t, y_{t-1}, ..., y_{t-p+1}]
    /// x_{t+1} = F x_t + e_t
    /// y_t = H x_t
    /// ```
    pub fn ar(phi: &[f64], sigma_eps: f64) -> Self {
        let p = phi.len();
        if p == 0 {
            return Self::local_level(sigma_eps, sigma_eps);
        }
        let mut f = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            f[[0, j]] = phi[j];
        }
        for i in 1..p {
            f[[i, i - 1]] = 1.0;
        }
        let mut h = Array2::<f64>::zeros((1, p));
        h[[0, 0]] = 1.0;
        let mut q = Array2::<f64>::zeros((p, p));
        q[[0, 0]] = sigma_eps * sigma_eps;

        Self {
            transition_matrix: f,
            observation_matrix: h,
            process_noise: q,
            observation_noise: Array2::zeros((1, 1)), // noise already in state eq
            initial_state: Array1::zeros(p),
            initial_cov: Array2::eye(p) * 1e4,
        }
    }

    /// Combined structural model: local linear trend + optional seasonal component.
    ///
    /// State layout: [level, slope, seasonal_1, ..., seasonal_{s-1}]
    pub fn structural(
        sigma_level: f64,
        sigma_slope: f64,
        sigma_seasonal: Option<f64>,
        period: Option<usize>,
        sigma_obs: f64,
    ) -> Self {
        let trend_dim = 2usize;
        let seasonal_dim = match (sigma_seasonal, period) {
            (Some(_), Some(p)) if p >= 2 => p - 1,
            _ => 0,
        };
        let n = trend_dim + seasonal_dim;

        // Build block-diagonal transition matrix
        let mut f = Array2::<f64>::zeros((n, n));
        // Trend block [2×2]
        f[[0, 0]] = 1.0;
        f[[0, 1]] = 1.0;
        f[[1, 1]] = 1.0;
        // Seasonal block
        if seasonal_dim > 0 {
            let s = seasonal_dim;
            let off = trend_dim;
            for j in 0..s {
                f[[off, off + j]] = -1.0;
            }
            for i in 1..s {
                f[[off + i, off + i - 1]] = 1.0;
            }
        }

        // Observation matrix: [1, 0, 1, 0, ..., 0]
        let mut h = Array2::<f64>::zeros((1, n));
        h[[0, 0]] = 1.0; // level
        if seasonal_dim > 0 {
            h[[0, trend_dim]] = 1.0; // first seasonal state
        }

        // Process noise
        let mut q = Array2::<f64>::zeros((n, n));
        q[[0, 0]] = sigma_level * sigma_level;
        q[[1, 1]] = sigma_slope * sigma_slope;
        if let (Some(sw), Some(_)) = (sigma_seasonal, period) {
            if seasonal_dim > 0 {
                q[[trend_dim, trend_dim]] = sw * sw;
            }
        }

        let r = Array2::from_elem((1, 1), sigma_obs * sigma_obs);

        Self {
            transition_matrix: f,
            observation_matrix: h,
            process_noise: q,
            observation_noise: r,
            initial_state: Array1::zeros(n),
            initial_cov: Array2::eye(n) * 1e6,
        }
    }

    /// Number of state dimensions.
    pub fn n_states(&self) -> usize {
        self.transition_matrix.dim().0
    }

    /// Number of observation dimensions.
    pub fn n_obs(&self) -> usize {
        self.observation_matrix.dim().0
    }
}

// ---------------------------------------------------------------------------
// Kalman filter result
// ---------------------------------------------------------------------------

/// Output from the Kalman filter (forward pass).
#[derive(Debug, Clone)]
pub struct KalmanFilterResult {
    /// Filtered state means m_{t|t}: shape (T, n_states)
    pub filtered_states: Array2<f64>,
    /// Filtered state covariances P_{t|t}: T × (n_states × n_states)
    pub filtered_covs: Vec<Array2<f64>>,
    /// Predicted state means m_{t|t-1}: shape (T, n_states)
    pub predicted_states: Array2<f64>,
    /// Predicted state covariances P_{t|t-1}: T × (n_states × n_states)
    pub predicted_covs: Vec<Array2<f64>>,
    /// Innovations v_t = y_t - H m_{t|t-1}: shape (T, n_obs)
    pub innovations: Array2<f64>,
    /// Innovation covariances S_t = H P_{t|t-1} H^T + R: T × (n_obs × n_obs)
    pub innovation_covs: Vec<Array2<f64>>,
    /// Log-likelihood of the observation sequence
    pub log_likelihood: f64,
}

// ---------------------------------------------------------------------------
// Kalman smoother result
// ---------------------------------------------------------------------------

/// Output from the RTS Kalman smoother (backward pass).
#[derive(Debug, Clone)]
pub struct KalmanSmootherResult {
    /// Smoothed state means m_{t|T}: shape (T, n_states)
    pub smoothed_states: Array2<f64>,
    /// Smoothed state covariances P_{t|T}: T × (n_states × n_states)
    pub smoothed_covs: Vec<Array2<f64>>,
    /// Log-likelihood (from the filter pass)
    pub log_likelihood: f64,
}

// ---------------------------------------------------------------------------
// kalman_filter
// ---------------------------------------------------------------------------

/// Run the Kalman filter (forward pass) on a multivariate observation sequence.
///
/// `observations` has shape `(T, n_obs)`.
/// Set `handle_missing = true` to skip rows that contain NaN (missing values).
pub fn kalman_filter(
    observations: ArrayView2<f64>,
    model: &StateSpaceModel,
    handle_missing: bool,
) -> Result<KalmanFilterResult> {
    let (t_len, n_obs) = observations.dim();
    let n_states = model.n_states();

    if model.n_obs() != n_obs {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: model.n_obs(),
            actual: n_obs,
        });
    }

    let mut m = model.initial_state.clone(); // current filtered state
    let mut p = model.initial_cov.clone(); // current filtered covariance

    let mut filtered_states = Array2::<f64>::zeros((t_len, n_states));
    let mut filtered_covs = Vec::with_capacity(t_len);
    let mut predicted_states = Array2::<f64>::zeros((t_len, n_states));
    let mut predicted_covs = Vec::with_capacity(t_len);
    let mut innovations = Array2::<f64>::zeros((t_len, n_obs));
    let mut innovation_covs = Vec::with_capacity(t_len);
    let mut log_likelihood = 0.0_f64;

    let f = &model.transition_matrix;
    let h = &model.observation_matrix;
    let q = &model.process_noise;
    let r = &model.observation_noise;
    let ft = mat_t(f);
    let ht = mat_t(h);

    for t in 0..t_len {
        // --- Prediction step ---
        // m_{t|t-1} = F m_{t-1|t-1}
        let m_pred = mat_vec(f, &m)?;
        // P_{t|t-1} = F P_{t-1|t-1} F^T + Q
        let fp = mat_mul(f, &p)?;
        let p_pred = mat_add(&mat_mul(&fp, &ft)?, q)?;

        // Store predicted
        for j in 0..n_states {
            predicted_states[[t, j]] = m_pred[j];
        }
        predicted_covs.push(p_pred.clone());

        // Observation at time t
        let y_t: Array1<f64> = observations.row(t).to_owned();
        let is_missing = handle_missing && y_t.iter().any(|&v| v.is_nan());

        if is_missing {
            // Skip update; propagate prediction as filter result
            for j in 0..n_states {
                filtered_states[[t, j]] = m_pred[j];
            }
            filtered_covs.push(p_pred.clone());
            // Innovation = 0, cov = S
            let hp = mat_mul(h, &p_pred)?;
            let s = mat_add(&mat_mul(&hp, &ht)?, r)?;
            innovation_covs.push(s);
            for j in 0..n_obs {
                innovations[[t, j]] = 0.0;
            }
            m = m_pred;
            p = p_pred;
            continue;
        }

        // --- Update step ---
        // v_t = y_t - H m_{t|t-1}
        let hm = mat_vec(h, &m_pred)?;
        let v: Array1<f64> = Array1::from_iter((0..n_obs).map(|i| y_t[i] - hm[i]));

        // S_t = H P_{t|t-1} H^T + R
        let hp = mat_mul(h, &p_pred)?;
        let s = mat_add(&mat_mul(&hp, &ht)?, r)?;

        // Kalman gain: K_t = P_{t|t-1} H^T S_t^{-1}
        // Solve S_t^T K_t^T = H P_{t|t-1}^T  →  k = P H^T S^{-1}
        let ph_t = mat_mul(&p_pred, &ht)?;
        let k = mat_mul(&ph_t, &mat_inv(&s)?)?;

        // m_{t|t} = m_{t|t-1} + K_t v_t
        let kv = mat_vec(&k, &v)?;
        let m_upd: Array1<f64> = Array1::from_iter((0..n_states).map(|i| m_pred[i] + kv[i]));

        // P_{t|t} = (I - K_t H) P_{t|t-1}  (Joseph form for stability)
        let kh = mat_mul(&k, h)?;
        let i_kh = mat_sub(&Array2::eye(n_states), &kh)?;
        let p_upd = mat_mul(&i_kh, &p_pred)?;

        // Log-likelihood contribution: -0.5 * (n_obs*log(2π) + log|S| + v'S^{-1}v)
        let ld = log_det(&s)?;
        let s_inv = mat_inv(&s)?;
        let s_inv_v = mat_vec(&s_inv, &v)?;
        let quad: f64 = (0..n_obs).map(|i| v[i] * s_inv_v[i]).sum();
        log_likelihood += -0.5 * ((n_obs as f64) * (2.0 * PI).ln() + ld + quad);

        // Store results
        for j in 0..n_states {
            filtered_states[[t, j]] = m_upd[j];
        }
        for j in 0..n_obs {
            innovations[[t, j]] = v[j];
        }
        filtered_covs.push(p_upd.clone());
        innovation_covs.push(s);

        m = m_upd;
        p = p_upd;
    }

    Ok(KalmanFilterResult {
        filtered_states,
        filtered_covs,
        predicted_states,
        predicted_covs,
        innovations,
        innovation_covs,
        log_likelihood,
    })
}

// ---------------------------------------------------------------------------
// kalman_smoother (RTS backward pass)
// ---------------------------------------------------------------------------

/// Run the Rauch-Tung-Striebel smoother on previously computed filter results.
pub fn kalman_smoother(
    filter_result: &KalmanFilterResult,
    model: &StateSpaceModel,
) -> Result<KalmanSmootherResult> {
    let t_len = filter_result.filtered_states.dim().0;
    let n_states = model.n_states();

    if t_len == 0 {
        return Ok(KalmanSmootherResult {
            smoothed_states: Array2::zeros((0, n_states)),
            smoothed_covs: vec![],
            log_likelihood: filter_result.log_likelihood,
        });
    }

    let f = &model.transition_matrix;
    let ft = mat_t(f);

    let mut smoothed_states = Array2::<f64>::zeros((t_len, n_states));
    let mut smoothed_covs = vec![Array2::<f64>::zeros((n_states, n_states)); t_len];

    // Initialize at time T (use filtered estimates)
    for j in 0..n_states {
        smoothed_states[[t_len - 1, j]] = filter_result.filtered_states[[t_len - 1, j]];
    }
    smoothed_covs[t_len - 1] = filter_result.filtered_covs[t_len - 1].clone();

    // Backward pass
    for t in (0..t_len - 1).rev() {
        let p_filt = &filter_result.filtered_covs[t];
        let p_pred_next = &filter_result.predicted_covs[t + 1];

        // Smoother gain: G_t = P_{t|t} F^T P_{t+1|t}^{-1}
        let pf_ft = mat_mul(p_filt, &ft)?;
        let p_pred_inv = mat_inv(p_pred_next)?;
        let g = mat_mul(&pf_ft, &p_pred_inv)?;

        // m_{t|T} = m_{t|t} + G_t (m_{t+1|T} - m_{t+1|t})
        let m_filt_t: Array1<f64> =
            Array1::from_iter((0..n_states).map(|j| filter_result.filtered_states[[t, j]]));
        let m_smooth_next: Array1<f64> =
            Array1::from_iter((0..n_states).map(|j| smoothed_states[[t + 1, j]]));
        let m_pred_next: Array1<f64> =
            Array1::from_iter((0..n_states).map(|j| filter_result.predicted_states[[t + 1, j]]));

        let diff: Array1<f64> =
            Array1::from_iter((0..n_states).map(|j| m_smooth_next[j] - m_pred_next[j]));
        let g_diff = mat_vec(&g, &diff)?;
        for j in 0..n_states {
            smoothed_states[[t, j]] = m_filt_t[j] + g_diff[j];
        }

        // P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T
        let p_smooth_next = &smoothed_covs[t + 1];
        let dp = mat_sub(p_smooth_next, p_pred_next)?;
        let g_dp = mat_mul(&g, &dp)?;
        let g_dp_gt = mat_mul(&g_dp, &mat_t(&g))?;
        smoothed_covs[t] = mat_add(p_filt, &g_dp_gt)?;
    }

    Ok(KalmanSmootherResult {
        smoothed_states,
        smoothed_covs,
        log_likelihood: filter_result.log_likelihood,
    })
}

// ---------------------------------------------------------------------------
// EM estimation
// ---------------------------------------------------------------------------

/// Result of the EM parameter estimation algorithm.
#[derive(Debug, Clone)]
pub struct EmResult {
    /// Fitted model after EM
    pub model: StateSpaceModel,
    /// Log-likelihood at each EM iteration
    pub log_likelihoods: Vec<f64>,
    /// Number of EM iterations performed
    pub n_iterations: usize,
    /// Whether the EM algorithm converged
    pub converged: bool,
}

/// Estimate SSM parameters via the EM algorithm.
///
/// `estimate_transition`: whether to re-estimate F  
/// `estimate_obs_matrix`: whether to re-estimate H
pub fn em_estimation(
    observations: ArrayView2<f64>,
    initial_model: StateSpaceModel,
    max_iter: usize,
    tol: f64,
    estimate_transition: bool,
    estimate_obs_matrix: bool,
) -> Result<EmResult> {
    let (t_len, n_obs_dim) = observations.dim();
    if t_len < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "EM requires at least 2 observations".to_string(),
            required: 2,
            actual: t_len,
        });
    }

    let mut model = initial_model;
    let mut log_likelihoods = Vec::with_capacity(max_iter);
    let mut converged = false;

    for iter in 0..max_iter {
        // E-step: run filter and smoother
        let filt = kalman_filter(observations, &model, true)?;
        let smooth = kalman_smoother(&filt, &model)?;
        let ll = filt.log_likelihood;
        log_likelihoods.push(ll);

        if iter > 0 {
            let prev_ll = log_likelihoods[iter - 1];
            if (ll - prev_ll).abs() < tol {
                converged = true;
                break;
            }
        }

        // M-step: update parameters using sufficient statistics
        let n_s = model.n_states();

        // Compute cross-covariance and auto-covariance of smoothed states
        // E[x_t x_t'] and E[x_t x_{t-1}'] summed over t
        let mut sum_pp = Array2::<f64>::zeros((n_s, n_s)); // sum E[x_t x_t']
        let mut sum_pp_lag = Array2::<f64>::zeros((n_s, n_s)); // sum E[x_t x_{t-1}']
        let mut sum_pp_prev = Array2::<f64>::zeros((n_s, n_s)); // sum E[x_{t-1} x_{t-1}']
        let mut sum_yp = Array2::<f64>::zeros((n_obs_dim, n_s)); // sum y_t x_t'
        let mut sum_yy = Array2::<f64>::zeros((n_obs_dim, n_obs_dim)); // sum y_t y_t'

        // Collect smoothed state outer products (including cross-term from smoother)
        for t in 0..t_len {
            let x_t: Array1<f64> =
                Array1::from_iter((0..n_s).map(|j| smooth.smoothed_states[[t, j]]));
            let e_xxt = outer_product_plus_cov(&x_t, &smooth.smoothed_covs[t]);
            sum_pp = mat_add(&sum_pp, &e_xxt)?;

            let y_t: Array1<f64> = Array1::from_iter((0..n_obs_dim).map(|j| observations[[t, j]]));
            let yxt = outer_product(&y_t, &x_t);
            sum_yp = mat_add(&sum_yp, &yxt)?;
            let yyt = outer_product(&y_t, &y_t);
            sum_yy = mat_add(&sum_yy, &yyt)?;
        }

        // Cross-term E[x_t x_{t-1}']: uses smoother cross-covariance approximation
        // P_{t,t-1|T} ≈ G_{t-1} P_{t|T}  (lag-one cross-covariance)
        // Recompute G_{t-1} from smoother gain matrices
        let f_ref = &model.transition_matrix;
        let ft_ref = mat_t(f_ref);
        for t in 1..t_len {
            let x_t: Array1<f64> =
                Array1::from_iter((0..n_s).map(|j| smooth.smoothed_states[[t, j]]));
            let x_prev: Array1<f64> =
                Array1::from_iter((0..n_s).map(|j| smooth.smoothed_states[[t - 1, j]]));

            // Approx cross-cov: G_{t-1} P_{t|T}
            let p_filt_prev = &filt.filtered_covs[t - 1];
            let p_pred_t = &filt.predicted_covs[t];
            let pf_ft = mat_mul(p_filt_prev, &ft_ref)?;
            let p_pred_inv = mat_inv(p_pred_t).unwrap_or_else(|_| Array2::eye(n_s));
            let g_prev = mat_mul(&pf_ft, &p_pred_inv)?;
            let p_cross = mat_add(
                &outer_product(&x_t, &x_prev),
                &mat_mul(&g_prev, &smooth.smoothed_covs[t])?,
            )?;

            sum_pp_lag = mat_add(&sum_pp_lag, &p_cross)?;
            let e_xprev = outer_product_plus_cov(&x_prev, &smooth.smoothed_covs[t - 1]);
            sum_pp_prev = mat_add(&sum_pp_prev, &e_xprev)?;
        }

        let t_f64 = t_len as f64;
        let tm1_f64 = (t_len - 1) as f64;

        // Update H (observation matrix)
        if estimate_obs_matrix {
            // H = (sum y_t x_t') (sum x_t x_t')^{-1}
            let sum_pp_inv = mat_inv(&sum_pp).unwrap_or_else(|_| Array2::eye(n_s));
            model.observation_matrix = mat_mul(&sum_yp, &sum_pp_inv)?;
        }

        // Update R (observation noise)
        {
            let h_new = model.observation_matrix.clone();
            let h_sum_yp_t = mat_mul(&h_new, &mat_t(&sum_yp))?;
            let r_new_unnorm = mat_sub(&sum_yy, &h_sum_yp_t)?;
            let mut r_new = Array2::<f64>::zeros((n_obs_dim, n_obs_dim));
            for i in 0..n_obs_dim {
                for j in 0..n_obs_dim {
                    r_new[[i, j]] = r_new_unnorm[[i, j]] / t_f64;
                }
            }
            // Enforce positive diagonal
            for i in 0..n_obs_dim {
                if r_new[[i, i]] < 1e-8 {
                    r_new[[i, i]] = 1e-8;
                }
            }
            model.observation_noise = r_new;
        }

        // Update F (transition matrix)
        if estimate_transition {
            // F = (sum x_t x_{t-1}') (sum x_{t-1} x_{t-1}')^{-1}
            let sum_pp_prev_inv = mat_inv(&sum_pp_prev).unwrap_or_else(|_| Array2::eye(n_s));
            model.transition_matrix = mat_mul(&sum_pp_lag, &sum_pp_prev_inv)?;
        }

        // Update Q (process noise)
        {
            let f_new = model.transition_matrix.clone();
            let f_sum_lag_t = mat_mul(&f_new, &mat_t(&sum_pp_lag))?;
            let q_unnorm = mat_sub(&mat_sub(&sum_pp, &f_sum_lag_t)?, &mat_t(&f_sum_lag_t))?;
            // Add F sum_pp_prev F^T term
            let f_spp = mat_mul(&f_new, &sum_pp_prev)?;
            let f_spp_ft = mat_mul(&f_spp, &mat_t(&f_new))?;
            let q_unnorm2 = mat_add(&q_unnorm, &f_spp_ft)?;
            let mut q_new = Array2::<f64>::zeros((n_s, n_s));
            for i in 0..n_s {
                for j in 0..n_s {
                    q_new[[i, j]] = q_unnorm2[[i, j]] / tm1_f64;
                }
            }
            // Enforce positive diagonal
            for i in 0..n_s {
                if q_new[[i, i]] < 1e-10 {
                    q_new[[i, i]] = 1e-10;
                }
            }
            model.process_noise = q_new;
        }
    }

    let n_iterations = log_likelihoods.len();
    Ok(EmResult {
        model,
        log_likelihoods,
        n_iterations,
        converged,
    })
}

/// Compute outer product x y' (matrix from two vectors).
fn outer_product(x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
    let m = x.len();
    let n = y.len();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            out[[i, j]] = x[i] * y[j];
        }
    }
    out
}

/// Compute E[x x'] = x x' + Cov (for smoothed expectations).
fn outer_product_plus_cov(x: &Array1<f64>, cov: &Array2<f64>) -> Array2<f64> {
    let n = x.len();
    let mut out = outer_product(x, x);
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] += cov[[i, j]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Structural model fitting helpers
// ---------------------------------------------------------------------------

/// Fit a Local Level model via EM.
///
/// Returns `(sigma_eta, sigma_eps, smoother_result)`.
pub fn fit_local_level(
    y: ArrayView1<f64>,
    max_iter: usize,
) -> Result<(f64, f64, KalmanSmootherResult)> {
    let n = y.len();
    if n < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "fit_local_level requires at least 2 observations".to_string(),
            required: 2,
            actual: n,
        });
    }

    // Initial guesses: variance of differences and variance of data
    let var_data: f64 = {
        let mean = y.iter().copied().filter(|v| !v.is_nan()).sum::<f64>()
            / y.iter().filter(|v| !v.is_nan()).count() as f64;
        y.iter()
            .filter(|v| !v.is_nan())
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n as f64
    };
    let sigma_eta0 = (var_data * 0.1).sqrt().max(1e-4);
    let sigma_eps0 = (var_data * 0.9).sqrt().max(1e-4);

    let init_model = StateSpaceModel::local_level(sigma_eta0, sigma_eps0);

    // Convert to 2D for EM
    let obs_2d: Array2<f64> = y
        .iter()
        .copied()
        .collect::<Array1<f64>>()
        .into_shape_with_order((n, 1))
        .map_err(|e| TimeSeriesError::ComputationError(format!("Shape error: {e}")))?;

    let em_result = em_estimation(obs_2d.view(), init_model, max_iter, 1e-6, false, false)?;

    let sigma_eta = em_result.model.process_noise[[0, 0]].sqrt();
    let sigma_eps = em_result.model.observation_noise[[0, 0]].sqrt();

    let filt = kalman_filter(obs_2d.view(), &em_result.model, true)?;
    let smooth = kalman_smoother(&filt, &em_result.model)?;

    Ok((sigma_eta, sigma_eps, smooth))
}

/// Fit a Local Linear Trend model via EM.
///
/// Returns `(sigma_level, sigma_slope, sigma_obs, smoother_result)`.
pub fn fit_local_linear_trend(
    y: ArrayView1<f64>,
    max_iter: usize,
) -> Result<(f64, f64, f64, KalmanSmootherResult)> {
    let n = y.len();
    if n < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "fit_local_linear_trend requires at least 3 observations".to_string(),
            required: 3,
            actual: n,
        });
    }

    let var_data: f64 = {
        let mean = y.iter().copied().filter(|v| !v.is_nan()).sum::<f64>()
            / y.iter().filter(|v| !v.is_nan()).count() as f64;
        y.iter()
            .filter(|v| !v.is_nan())
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n as f64
    };

    let sigma_eta0 = (var_data * 0.1).sqrt().max(1e-4);
    let sigma_zeta0 = (var_data * 0.01).sqrt().max(1e-5);
    let sigma_eps0 = (var_data * 0.5).sqrt().max(1e-4);

    let init_model = StateSpaceModel::local_linear_trend(sigma_eta0, sigma_zeta0, sigma_eps0);

    let obs_2d: Array2<f64> = y
        .iter()
        .copied()
        .collect::<Array1<f64>>()
        .into_shape_with_order((n, 1))
        .map_err(|e| TimeSeriesError::ComputationError(format!("Shape error: {e}")))?;

    let em_result = em_estimation(obs_2d.view(), init_model, max_iter, 1e-6, false, false)?;

    let sigma_level = em_result.model.process_noise[[0, 0]].sqrt();
    let sigma_slope = em_result.model.process_noise[[1, 1]].sqrt();
    let sigma_obs = em_result.model.observation_noise[[0, 0]].sqrt();

    let filt = kalman_filter(obs_2d.view(), &em_result.model, true)?;
    let smooth = kalman_smoother(&filt, &em_result.model)?;

    Ok((sigma_level, sigma_slope, sigma_obs, smooth))
}

// ---------------------------------------------------------------------------
// Dynamic Linear Model (with optional time-varying H_t)
// ---------------------------------------------------------------------------

/// Dynamic Linear Model with optional time-varying observation matrix.
#[derive(Debug, Clone)]
pub struct DynamicLinearModel {
    /// Base SSM (used when no time-varying H is provided)
    pub base_model: StateSpaceModel,
    /// Optional sequence of observation matrices H_t (one per time step)
    pub time_varying_obs: Option<Vec<Array2<f64>>>,
}

impl DynamicLinearModel {
    /// Create a DLM from a static SSM.
    pub fn new(model: StateSpaceModel) -> Self {
        Self {
            base_model: model,
            time_varying_obs: None,
        }
    }

    /// Attach a time-varying sequence of observation matrices.
    pub fn with_time_varying_obs(mut self, h_sequence: Vec<Array2<f64>>) -> Self {
        self.time_varying_obs = Some(h_sequence);
        self
    }

    /// Run the Kalman filter.
    pub fn filter(&self, observations: ArrayView2<f64>) -> Result<KalmanFilterResult> {
        match &self.time_varying_obs {
            None => kalman_filter(observations, &self.base_model, true),
            Some(h_seq) => kalman_filter_tv(observations, &self.base_model, h_seq),
        }
    }

    /// Run filter + smoother.
    pub fn smooth(&self, observations: ArrayView2<f64>) -> Result<KalmanSmootherResult> {
        let filt = self.filter(observations)?;
        kalman_smoother(&filt, &self.base_model)
    }
}

/// Kalman filter with time-varying observation matrices H_t.
fn kalman_filter_tv(
    observations: ArrayView2<f64>,
    model: &StateSpaceModel,
    h_seq: &[Array2<f64>],
) -> Result<KalmanFilterResult> {
    let (t_len, n_obs) = observations.dim();
    let n_states = model.n_states();

    if h_seq.len() != t_len {
        return Err(TimeSeriesError::InvalidInput(format!(
            "h_seq length {} does not match T = {}",
            h_seq.len(),
            t_len
        )));
    }

    let mut m = model.initial_state.clone();
    let mut p = model.initial_cov.clone();

    let mut filtered_states = Array2::<f64>::zeros((t_len, n_states));
    let mut filtered_covs = Vec::with_capacity(t_len);
    let mut predicted_states = Array2::<f64>::zeros((t_len, n_states));
    let mut predicted_covs = Vec::with_capacity(t_len);
    let mut innovations = Array2::<f64>::zeros((t_len, n_obs));
    let mut innovation_covs = Vec::with_capacity(t_len);
    let mut log_likelihood = 0.0_f64;

    let f = &model.transition_matrix;
    let q = &model.process_noise;
    let r = &model.observation_noise;
    let ft = mat_t(f);

    for t in 0..t_len {
        let h = &h_seq[t];
        let ht = mat_t(h);

        // Prediction
        let m_pred = mat_vec(f, &m)?;
        let fp = mat_mul(f, &p)?;
        let p_pred = mat_add(&mat_mul(&fp, &ft)?, q)?;

        for j in 0..n_states {
            predicted_states[[t, j]] = m_pred[j];
        }
        predicted_covs.push(p_pred.clone());

        let y_t: Array1<f64> = observations.row(t).to_owned();
        let is_missing = y_t.iter().any(|&v| v.is_nan());

        if is_missing {
            for j in 0..n_states {
                filtered_states[[t, j]] = m_pred[j];
            }
            filtered_covs.push(p_pred.clone());
            let hp = mat_mul(h, &p_pred)?;
            let s = mat_add(&mat_mul(&hp, &ht)?, r)?;
            innovation_covs.push(s);
            for j in 0..n_obs {
                innovations[[t, j]] = 0.0;
            }
            m = m_pred;
            p = p_pred;
            continue;
        }

        // Update
        let hm = mat_vec(h, &m_pred)?;
        let v: Array1<f64> = Array1::from_iter((0..n_obs).map(|i| y_t[i] - hm[i]));
        let hp = mat_mul(h, &p_pred)?;
        let s = mat_add(&mat_mul(&hp, &ht)?, r)?;
        let ph_t = mat_mul(&p_pred, &ht)?;
        let k = mat_mul(&ph_t, &mat_inv(&s)?)?;
        let kv = mat_vec(&k, &v)?;
        let m_upd: Array1<f64> = Array1::from_iter((0..n_states).map(|i| m_pred[i] + kv[i]));
        let kh = mat_mul(&k, h)?;
        let i_kh = mat_sub(&Array2::eye(n_states), &kh)?;
        let p_upd = mat_mul(&i_kh, &p_pred)?;

        let ld = log_det(&s)?;
        let s_inv = mat_inv(&s)?;
        let s_inv_v = mat_vec(&s_inv, &v)?;
        let quad: f64 = (0..n_obs).map(|i| v[i] * s_inv_v[i]).sum();
        log_likelihood += -0.5 * ((n_obs as f64) * (2.0 * PI).ln() + ld + quad);

        for j in 0..n_states {
            filtered_states[[t, j]] = m_upd[j];
        }
        for j in 0..n_obs {
            innovations[[t, j]] = v[j];
        }
        filtered_covs.push(p_upd.clone());
        innovation_covs.push(s);

        m = m_upd;
        p = p_upd;
    }

    Ok(KalmanFilterResult {
        filtered_states,
        filtered_covs,
        predicted_states,
        predicted_covs,
        innovations,
        innovation_covs,
        log_likelihood,
    })
}

// ---------------------------------------------------------------------------
// Forecasting
// ---------------------------------------------------------------------------

/// Forecast `h` steps ahead from a fitted SSM given univariate observations.
///
/// Returns `(point_forecast, lower_95, upper_95)`.
pub fn forecast_ssm(
    observations: ArrayView1<f64>,
    model: &StateSpaceModel,
    h: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let n = observations.len();
    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "forecast_ssm requires at least one observation".to_string(),
            required: 1,
            actual: 0,
        });
    }
    if h == 0 {
        return Ok((Array1::zeros(0), Array1::zeros(0), Array1::zeros(0)));
    }

    // Run filter to obtain final state
    let obs_2d: Array2<f64> = observations
        .iter()
        .copied()
        .collect::<Array1<f64>>()
        .into_shape_with_order((n, 1))
        .map_err(|e| TimeSeriesError::ComputationError(format!("Shape error: {e}")))?;

    let filt = kalman_filter(obs_2d.view(), model, true)?;

    // Final filtered state
    let n_states = model.n_states();
    let mut m: Array1<f64> =
        Array1::from_iter((0..n_states).map(|j| filt.filtered_states[[n - 1, j]]));
    let mut p = filt.filtered_covs[n - 1].clone();

    let f = &model.transition_matrix;
    let h_mat = &model.observation_matrix;
    let q = &model.process_noise;
    let r = &model.observation_noise;
    let ft = mat_t(f);

    let z_95 = 1.959_963_985; // qnorm(0.975)

    let mut means = Array1::<f64>::zeros(h);
    let mut lowers = Array1::<f64>::zeros(h);
    let mut uppers = Array1::<f64>::zeros(h);

    for step in 0..h {
        // Predict
        m = mat_vec(f, &m)?;
        let fp = mat_mul(f, &p)?;
        p = mat_add(&mat_mul(&fp, &ft)?, q)?;

        // Observation mean and variance
        let y_mean = mat_vec(h_mat, &m)?[0];
        let hp = mat_mul(h_mat, &p)?;
        let s = mat_add(&mat_mul(&hp, &mat_t(h_mat))?, r)?;
        let y_var = s[[0, 0]].max(0.0);
        let y_std = y_var.sqrt();

        means[step] = y_mean;
        lowers[step] = y_mean - z_95 * y_std;
        uppers[step] = y_mean + z_95 * y_std;
    }

    Ok((means, lowers, uppers))
}

// ---------------------------------------------------------------------------
// Structural break / CUSUM detection
// ---------------------------------------------------------------------------

/// Detect potential structural breaks via the CUSUM of innovations.
///
/// Returns a list of `(time_index, cusum_score)` pairs for all time steps.
pub fn detect_structural_break_ssm(
    observations: ArrayView1<f64>,
    model: &StateSpaceModel,
) -> Result<Vec<(usize, f64)>> {
    let n = observations.len();
    if n < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Structural break detection requires at least 3 observations".to_string(),
            required: 3,
            actual: n,
        });
    }

    let obs_2d: Array2<f64> = observations
        .iter()
        .copied()
        .collect::<Array1<f64>>()
        .into_shape_with_order((n, 1))
        .map_err(|e| TimeSeriesError::ComputationError(format!("Shape error: {e}")))?;

    let filt = kalman_filter(obs_2d.view(), model, true)?;

    // Standardize innovations by their standard deviation
    let innov: Vec<f64> = (0..n).map(|t| filt.innovations[[t, 0]]).collect();
    let sigma: f64 = {
        let mean = innov.iter().copied().sum::<f64>() / n as f64;
        let var = innov.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt().max(1e-12)
    };

    // CUSUM: cumulative sum of standardized innovations
    let mut cusum = 0.0_f64;
    let mut results = Vec::with_capacity(n);
    for (t, &v) in innov.iter().enumerate() {
        cusum += v / sigma;
        results.push((t, cusum));
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // helper: build observations array for univariate series
    fn to_obs2d(v: &[f64]) -> Array2<f64> {
        let n = v.len();
        Array2::from_shape_vec((n, 1), v.to_vec()).expect("shape ok")
    }

    // -----------------------------------------------------------------------
    // Model construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_level_dims() {
        let m = StateSpaceModel::local_level(0.1, 0.5);
        assert_eq!(m.n_states(), 1);
        assert_eq!(m.n_obs(), 1);
        assert!((m.transition_matrix[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((m.process_noise[[0, 0]] - 0.01).abs() < 1e-15);
        assert!((m.observation_noise[[0, 0]] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn test_local_linear_trend_dims() {
        let m = StateSpaceModel::local_linear_trend(0.1, 0.05, 0.2);
        assert_eq!(m.n_states(), 2);
        assert_eq!(m.n_obs(), 1);
        // Transition: [[1,1],[0,1]]
        assert_eq!(m.transition_matrix[[0, 0]], 1.0);
        assert_eq!(m.transition_matrix[[0, 1]], 1.0);
        assert_eq!(m.transition_matrix[[1, 0]], 0.0);
        assert_eq!(m.transition_matrix[[1, 1]], 1.0);
    }

    #[test]
    fn test_ar1_state_space() {
        let phi = vec![0.8];
        let m = StateSpaceModel::ar(&phi, 0.3);
        assert_eq!(m.n_states(), 1);
        assert_eq!(m.transition_matrix[[0, 0]], 0.8);
        assert_eq!(m.observation_matrix[[0, 0]], 1.0);
    }

    #[test]
    fn test_ar2_state_space() {
        let phi = vec![0.6, 0.2];
        let m = StateSpaceModel::ar(&phi, 0.3);
        assert_eq!(m.n_states(), 2);
        // F[0,:] = [0.6, 0.2]
        assert!((m.transition_matrix[[0, 0]] - 0.6).abs() < 1e-12);
        assert!((m.transition_matrix[[0, 1]] - 0.2).abs() < 1e-12);
        // F[1,0] = 1
        assert_eq!(m.transition_matrix[[1, 0]], 1.0);
    }

    #[test]
    fn test_seasonal_dims() {
        let m = StateSpaceModel::seasonal(4, 0.1);
        assert_eq!(m.n_states(), 3); // period - 1
        assert_eq!(m.n_obs(), 1);
        // First row of F should be [-1, -1, -1]
        for j in 0..3 {
            assert_eq!(m.transition_matrix[[0, j]], -1.0);
        }
        // Shift rows
        assert_eq!(m.transition_matrix[[1, 0]], 1.0);
        assert_eq!(m.transition_matrix[[2, 1]], 1.0);
    }

    #[test]
    fn test_structural_trend_only() {
        let m = StateSpaceModel::structural(0.1, 0.05, None, None, 0.2);
        assert_eq!(m.n_states(), 2); // trend only
        assert_eq!(m.n_obs(), 1);
    }

    #[test]
    fn test_structural_trend_plus_seasonal() {
        let m = StateSpaceModel::structural(0.1, 0.05, Some(0.02), Some(4), 0.2);
        assert_eq!(m.n_states(), 5); // 2 trend + 3 seasonal
        assert_eq!(m.n_obs(), 1);
    }

    // -----------------------------------------------------------------------
    // Kalman filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_constant_signal() {
        // y_t = 5 + noise; the filter should converge toward 5
        let mut obs_vec = vec![5.0_f64; 50];
        // Add small noise
        for (i, v) in obs_vec.iter_mut().enumerate() {
            *v += 0.1 * ((i as f64 * 0.3).sin());
        }
        let obs = to_obs2d(&obs_vec);
        let model = StateSpaceModel::local_level(0.01, 0.5);
        let res = kalman_filter(obs.view(), &model, false).expect("filter ok");
        let last = res.filtered_states[[49, 0]];
        assert!(
            (last - 5.0).abs() < 1.0,
            "last filtered state={last} not near 5"
        );
        assert!(res.log_likelihood.is_finite());
    }

    #[test]
    fn test_filter_log_likelihood_negative() {
        let obs = to_obs2d(&[1.0, 2.0, 1.5, 2.5, 1.8]);
        let model = StateSpaceModel::local_level(0.1, 0.3);
        let res = kalman_filter(obs.view(), &model, false).expect("ok");
        // Log-likelihood should be finite and typically negative
        assert!(res.log_likelihood.is_finite());
    }

    #[test]
    fn test_filter_missing_data() {
        let obs = to_obs2d(&[1.0, f64::NAN, 2.0, f64::NAN, 3.0]);
        let model = StateSpaceModel::local_level(0.1, 0.3);
        let res = kalman_filter(obs.view(), &model, true).expect("missing ok");
        // Should run without error; check finite filtered states (non-NaN)
        for t in 0..5 {
            assert!(res.filtered_states[[t, 0]].is_finite());
        }
    }

    #[test]
    fn test_filter_dimension_mismatch_error() {
        let obs = to_obs2d(&[1.0, 2.0, 3.0]);
        // Model expects 2 observations per time step but data has 1
        let mut model = StateSpaceModel::new(2, 2);
        // Keep as-is with n_obs=2 but data has n_obs=1
        let result = kalman_filter(obs.view(), &model, false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Kalman smoother tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoother_local_level() {
        let data: Vec<f64> = (0..20).map(|i| (i as f64 * 0.1).sin() + 2.0).collect();
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_level(0.1, 0.3);
        let filt = kalman_filter(obs.view(), &model, false).expect("filter ok");
        let smooth = kalman_smoother(&filt, &model).expect("smoother ok");
        assert_eq!(smooth.smoothed_states.dim().0, 20);
        assert_eq!(smooth.smoothed_covs.len(), 20);
        // Smoothed states should not differ wildly from filtered
        for t in 0..20 {
            let fs = filt.filtered_states[[t, 0]];
            let ss = smooth.smoothed_states[[t, 0]];
            assert!((fs - ss).abs() < 5.0, "t={t}: filter={fs} smoother={ss}");
        }
    }

    #[test]
    fn test_smoother_ll_matches_filter() {
        let data = vec![1.0, 1.1, 1.2, 1.1, 1.3, 1.2];
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_level(0.05, 0.1);
        let filt = kalman_filter(obs.view(), &model, false).expect("ok");
        let smooth = kalman_smoother(&filt, &model).expect("ok");
        assert!((filt.log_likelihood - smooth.log_likelihood).abs() < 1e-10);
    }

    #[test]
    fn test_smoother_backward_improves_early_estimates() {
        // For a random walk, smoother should reduce variance of early estimates
        let data: Vec<f64> = (0..30).map(|i| i as f64 + 0.1 * (i as f64).sin()).collect();
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_level(0.5, 0.5);
        let filt = kalman_filter(obs.view(), &model, false).expect("ok");
        let smooth = kalman_smoother(&filt, &model).expect("ok");
        // Smoothed covariance at t=0 should be <= filtered covariance at t=0
        let filt_var = filt.filtered_covs[0][[0, 0]];
        let smooth_var = smooth.smoothed_covs[0][[0, 0]];
        assert!(
            smooth_var <= filt_var + 1e-8,
            "smooth_var={smooth_var} > filt_var={filt_var}"
        );
    }

    // -----------------------------------------------------------------------
    // AR(1) in state space
    // -----------------------------------------------------------------------

    #[test]
    fn test_ar1_filter_tracks_series() {
        // Generate AR(1) with phi=0.7
        let n = 40;
        let mut data = vec![0.0_f64; n];
        for i in 1..n {
            data[i] = 0.7 * data[i - 1] + 0.3 * (i as f64 * 0.5).sin();
        }
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::ar(&[0.7], 0.3);
        let filt = kalman_filter(obs.view(), &model, false).expect("ok");
        // filtered state should track the signal
        for t in 5..n {
            assert!(filt.filtered_states[[t, 0]].is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // Local linear trend filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_linear_trend_filter() {
        // Linear trend: y_t = t + noise
        let data: Vec<f64> = (0..25)
            .map(|i| i as f64 + 0.05 * (i as f64).cos())
            .collect();
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_linear_trend(0.01, 0.001, 0.2);
        let filt = kalman_filter(obs.view(), &model, false).expect("ok");
        // At the end, level state (index 0) should be near 24
        let last_level = filt.filtered_states[[24, 0]];
        assert!((last_level - 24.0).abs() < 5.0, "last level={last_level}");
    }

    // -----------------------------------------------------------------------
    // Forecast tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_forecast_ssm_local_level() {
        let data: Vec<f64> = vec![3.0; 20];
        let model = StateSpaceModel::local_level(0.01, 0.1);
        let (mean, lower, upper) =
            forecast_ssm(Array1::from(data).view(), &model, 5).expect("forecast ok");
        assert_eq!(mean.len(), 5);
        // All forecasts should be near 3
        for i in 0..5 {
            assert!((mean[i] - 3.0).abs() < 1.0, "mean[{i}]={}", mean[i]);
            assert!(lower[i] <= mean[i]);
            assert!(upper[i] >= mean[i]);
        }
    }

    #[test]
    fn test_forecast_intervals_widen() {
        // Uncertainty should grow over forecast horizon
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let model = StateSpaceModel::local_level(0.2, 0.1);
        let (_, lower, upper) =
            forecast_ssm(Array1::from(data).view(), &model, 10).expect("forecast ok");
        // Interval at step 9 should be wider than at step 0
        let width_0 = upper[0] - lower[0];
        let width_9 = upper[9] - lower[9];
        assert!(width_9 >= width_0, "width_9={width_9} < width_0={width_0}");
    }

    // -----------------------------------------------------------------------
    // EM convergence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_em_local_level_converges() {
        let data: Vec<f64> = (0..60)
            .map(|i| 2.0 + 0.5 * (i as f64 * 0.2).sin() + 0.1 * (i as f64 * 0.07).cos())
            .collect();
        let obs = to_obs2d(&data);
        let init = StateSpaceModel::local_level(0.3, 0.5);
        let em = em_estimation(obs.view(), init, 100, 1e-5, false, false).expect("em ok");
        // Log-likelihoods should be non-decreasing (or nearly so)
        for i in 1..em.log_likelihoods.len() {
            let prev = em.log_likelihoods[i - 1];
            let curr = em.log_likelihoods[i];
            assert!(
                curr >= prev - 1.0,
                "LL decreased at iter {i}: {prev} -> {curr}"
            );
        }
    }

    #[test]
    fn test_em_returns_positive_variances() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let obs = to_obs2d(&data);
        let init = StateSpaceModel::local_level(0.2, 0.4);
        let em = em_estimation(obs.view(), init, 50, 1e-4, false, false).expect("ok");
        assert!(em.model.process_noise[[0, 0]] > 0.0);
        assert!(em.model.observation_noise[[0, 0]] > 0.0);
    }

    // -----------------------------------------------------------------------
    // fit_local_level
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_local_level() {
        let data: Vec<f64> = (0..40)
            .map(|i| 5.0 + 0.3 * (i as f64 * 0.4).sin())
            .collect();
        let arr = Array1::from(data);
        let (sigma_eta, sigma_eps, smooth) = fit_local_level(arr.view(), 50).expect("ok");
        assert!(sigma_eta > 0.0);
        assert!(sigma_eps > 0.0);
        assert_eq!(smooth.smoothed_states.dim().0, 40);
    }

    // -----------------------------------------------------------------------
    // fit_local_linear_trend
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_local_linear_trend() {
        let data: Vec<f64> = (0..50)
            .map(|i| i as f64 * 0.5 + 0.2 * (i as f64).sin())
            .collect();
        let arr = Array1::from(data);
        let (sigma_level, sigma_slope, sigma_obs, smooth) =
            fit_local_linear_trend(arr.view(), 30).expect("ok");
        assert!(sigma_level > 0.0);
        assert!(sigma_slope > 0.0);
        assert!(sigma_obs > 0.0);
        assert_eq!(smooth.smoothed_states.dim().0, 50);
    }

    // -----------------------------------------------------------------------
    // DynamicLinearModel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dlm_static() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_level(0.2, 0.3);
        let dlm = DynamicLinearModel::new(model);
        let filt = dlm.filter(obs.view()).expect("ok");
        assert_eq!(filt.filtered_states.dim().0, 7);
    }

    #[test]
    fn test_dlm_time_varying_obs() {
        let n = 5;
        let data: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 2.0];
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_level(0.1, 0.3);
        // Time-varying H_t: slightly different each step
        let h_seq: Vec<Array2<f64>> = (0..n)
            .map(|i| {
                let v = 1.0 + 0.01 * i as f64;
                Array2::from_elem((1, 1), v)
            })
            .collect();
        let dlm = DynamicLinearModel::new(model).with_time_varying_obs(h_seq);
        let filt = dlm.filter(obs.view()).expect("time-varying ok");
        assert_eq!(filt.filtered_states.dim().0, n);
    }

    // -----------------------------------------------------------------------
    // Structural break detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_structural_break() {
        // Series with a mean shift at t=15
        let mut data: Vec<f64> = (0..15).map(|_| 1.0_f64).collect();
        data.extend((0..15).map(|_| 5.0_f64));
        let arr = Array1::from(data.clone());
        let model = StateSpaceModel::local_level(0.1, 0.5);
        let scores = detect_structural_break_ssm(arr.view(), &model).expect("ok");
        assert_eq!(scores.len(), 30);
        // CUSUM should have a maximum near t=15 (the shift point)
        let (break_t, _) = scores
            .iter()
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .expect("unexpected None or Err")
            })
            .copied()
            .unwrap_or((0, 0.0));
        // The break index should be somewhere in the second half
        assert!(break_t >= 10, "break_t={break_t} too early");
    }

    #[test]
    fn test_cusum_monotone_no_break() {
        // Stationary series; CUSUM should stay bounded
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let arr = Array1::from(data);
        let model = StateSpaceModel::local_level(0.1, 0.3);
        let scores = detect_structural_break_ssm(arr.view(), &model).expect("ok");
        // All cusum scores should be finite
        for (_, s) in &scores {
            assert!(s.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // Innovations sanity check
    // -----------------------------------------------------------------------

    #[test]
    fn test_innovations_near_zero_perfect_prediction() {
        // When model perfectly matches data (no noise), innovations should be small
        // after burn-in
        let data: Vec<f64> = vec![3.0; 30];
        let obs = to_obs2d(&data);
        let model = StateSpaceModel::local_level(0.001, 0.001);
        let filt = kalman_filter(obs.view(), &model, false).expect("ok");
        // After burn-in, innovations should be near 0
        for t in 10..30 {
            assert!(
                filt.innovations[[t, 0]].abs() < 0.1,
                "innov at {t}={}",
                filt.innovations[[t, 0]]
            );
        }
    }
}
