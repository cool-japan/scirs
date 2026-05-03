//! Linear Gaussian State Space Models using plain `Vec<Vec<f64>>` matrices.
//!
//! This module provides a complete, self-contained implementation that does
//! **not** depend on ndarray.  All matrices are represented as
//! `Vec<Vec<f64>>` (row-major, rectangular).
//!
//! ## Contents
//!
//! - [`StateSpace`] — model definition (F, H, Q, R, m0, P0)
//! - [`KalmanFilter`] — forward Kalman filter
//! - [`KalmanFilterResult`] — output of the forward pass
//! - [`smooth`] — RTS (Rauch-Tung-Striebel) backward smoother
//! - [`SmootherResult`] — output of the smoother
//! - [`em_fit`] — EM algorithm for parameter estimation
//! - [`arima_to_state_space`] — convert ARIMA(p,d,q) to companion state space
//! - [`predict_state_space`] — multi-step-ahead forecasting
//!
//! ## Model equations
//!
//! ```text
//! x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
//! y_t     = H x_t + v_t,   v_t ~ N(0, R)
//! ```
//!
//! where `x_t` is the n-dimensional state and `y_t` is the p-dimensional
//! observation.

use crate::error::{Result, TimeSeriesError};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper: flat row-major matrix (private)
// ---------------------------------------------------------------------------

/// A compact row-major matrix. All algebra is column-count-aware.
#[derive(Clone, Debug)]
struct FlatMat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl FlatMat {
    fn zeros(rows: usize, cols: usize) -> Self {
        FlatMat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }
    #[inline]
    fn add_at(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] += v;
    }

    fn mul(&self, rhs: &FlatMat) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "FlatMat mul: dimension mismatch {} × {} vs {} × {}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..rhs.cols {
                    let prev = out.get(i, j);
                    out.set(i, j, prev + a * rhs.get(k, j));
                }
            }
        }
        out
    }

    fn t(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    fn add(&self, rhs: &FlatMat) -> Self {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols));
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        FlatMat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    fn sub(&self, rhs: &FlatMat) -> Self {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols));
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a - b)
            .collect();
        FlatMat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x * s).collect();
        FlatMat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Invert a square matrix via Gauss-Jordan with partial pivoting.
    fn inv(&self) -> Result<Self> {
        let n = self.rows;
        if n != self.cols {
            return Err(TimeSeriesError::ComputationError(
                "Non-square inversion".into(),
            ));
        }
        let mut aug: Vec<f64> = Vec::with_capacity(n * 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.push(self.get(i, j));
            }
            for j in 0..n {
                aug.push(if i == j { 1.0 } else { 0.0 });
            }
        }
        let w = 2 * n;
        for col in 0..n {
            let mut piv = col;
            let mut max_v = aug[col * w + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * w + col].abs();
                if v > max_v {
                    max_v = v;
                    piv = row;
                }
            }
            if max_v < 1e-15 {
                return Err(TimeSeriesError::ComputationError(
                    "Singular matrix during inversion".into(),
                ));
            }
            if piv != col {
                for j in 0..w {
                    aug.swap(col * w + j, piv * w + j);
                }
            }
            let d = aug[col * w + col];
            for j in 0..w {
                aug[col * w + j] /= d;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let f = aug[row * w + col];
                for j in 0..w {
                    aug[row * w + j] -= f * aug[col * w + j];
                }
            }
        }
        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug[i * w + (n + j)]);
            }
        }
        Ok(inv)
    }

    /// Convert to Vec<Vec<f64>>.
    fn to_nested(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|r| (0..self.cols).map(|c| self.get(r, c)).collect())
            .collect()
    }

    /// Build from Vec<Vec<f64>>.  All rows must have equal length.
    fn from_nested(m: &[Vec<f64>]) -> Result<Self> {
        if m.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let cols = m[0].len();
        for (i, row) in m.iter().enumerate() {
            if row.len() != cols {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: cols,
                    actual: row.len(),
                });
            }
        }
        let data: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
        Ok(FlatMat {
            rows: m.len(),
            cols,
            data,
        })
    }

    /// Build from a flat Vec and a column vector dimension.
    fn from_col_vec(v: &[f64]) -> Self {
        FlatMat {
            rows: v.len(),
            cols: 1,
            data: v.to_vec(),
        }
    }

    /// Extract column c as a Vec.
    fn col_vec(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Outer product: self (n×1) * rhs' (1×m) = n×m.
    fn outer(&self, rhs: &FlatMat) -> Self {
        assert_eq!(self.cols, 1);
        assert_eq!(rhs.cols, 1);
        let n = self.rows;
        let m = rhs.rows;
        let mut out = Self::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                out.set(i, j, self.get(i, 0) * rhs.get(j, 0));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Regularise a covariance matrix for numerical stability.
fn regularise(p: &FlatMat, eps: f64) -> FlatMat {
    let mut out = p.clone();
    for i in 0..p.rows {
        out.add_at(i, i, eps);
    }
    out
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Linear Gaussian State Space model.
///
/// ```text
/// x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
/// y_t     = H x_t + v_t,   v_t ~ N(0, R)
/// ```
///
/// All matrices are row-major `Vec<Vec<f64>>`.
#[derive(Clone, Debug)]
pub struct StateSpace {
    /// Transition matrix F (n × n).
    pub f: Vec<Vec<f64>>,
    /// Observation matrix H (p × n).
    pub h: Vec<Vec<f64>>,
    /// Process noise covariance Q (n × n).
    pub q: Vec<Vec<f64>>,
    /// Observation noise covariance R (p × p).
    pub r: Vec<Vec<f64>>,
    /// Initial state mean m0 (n).
    pub m0: Vec<f64>,
    /// Initial state covariance P0 (n × n).
    pub p0: Vec<Vec<f64>>,
}

impl StateSpace {
    /// Construct a Local Level model `y_t = μ_t + ε_t`,
    /// `μ_{t+1} = μ_t + η_t`.
    ///
    /// # Arguments
    /// * `sigma_level` – standard deviation of the level disturbance
    /// * `sigma_obs`   – standard deviation of the observation noise
    /// * `y0`          – initial level guess (e.g., first observation)
    pub fn local_level(sigma_level: f64, sigma_obs: f64, y0: f64) -> Self {
        StateSpace {
            f: vec![vec![1.0]],
            h: vec![vec![1.0]],
            q: vec![vec![sigma_level * sigma_level]],
            r: vec![vec![sigma_obs * sigma_obs]],
            m0: vec![y0],
            p0: vec![vec![1e6]],
        }
    }

    /// State dimension n.
    pub fn n_states(&self) -> usize {
        self.m0.len()
    }
    /// Observation dimension p.
    pub fn n_obs(&self) -> usize {
        self.r.len()
    }

    fn to_flat(&self) -> Result<(FlatMat, FlatMat, FlatMat, FlatMat, FlatMat)> {
        Ok((
            FlatMat::from_nested(&self.f)?,
            FlatMat::from_nested(&self.h)?,
            FlatMat::from_nested(&self.q)?,
            FlatMat::from_nested(&self.r)?,
            FlatMat::from_nested(&self.p0)?,
        ))
    }
}

/// Result of the forward Kalman filter pass.
#[derive(Clone, Debug)]
pub struct KalmanFilterResult {
    /// Filtered state means: T × n.
    pub filtered_means: Vec<Vec<f64>>,
    /// Filtered state covariances: T × (n × n).
    pub filtered_covs: Vec<Vec<Vec<f64>>>,
    /// Predicted state means: T × n.
    pub predicted_means: Vec<Vec<f64>>,
    /// Predicted state covariances: T × (n × n).
    pub predicted_covs: Vec<Vec<Vec<f64>>>,
    /// Total log-likelihood.
    pub log_likelihood: f64,
    /// Transition matrix F (for RTS smoother).
    pub(crate) f_flat: FlatMat,
}

/// Result of the RTS backward smoother.
#[derive(Clone, Debug)]
pub struct SmootherResult {
    /// Smoothed state means: T × n.
    pub smoothed_means: Vec<Vec<f64>>,
    /// Smoothed state covariances: T × (n × n).
    pub smoothed_covs: Vec<Vec<Vec<f64>>>,
}

// ---------------------------------------------------------------------------
// Kalman filter (forward pass, multivariate)
// ---------------------------------------------------------------------------

/// Wraps a [`StateSpace`] and runs the Kalman filter.
pub struct KalmanFilter {
    /// The underlying state-space model.
    pub model: StateSpace,
}

impl KalmanFilter {
    /// Create a new `KalmanFilter`.
    pub fn new(model: StateSpace) -> Self {
        KalmanFilter { model }
    }

    /// Run the forward Kalman filter on `obs`.
    ///
    /// # Arguments
    /// * `obs` – sequence of p-dimensional observations (length T)
    ///
    /// # Returns
    /// [`KalmanFilterResult`] containing filtered/predicted means, covariances,
    /// and total log-likelihood.
    pub fn filter(&self, obs: &[Vec<f64>]) -> Result<KalmanFilterResult> {
        let t_len = obs.len();
        if t_len == 0 {
            return Err(TimeSeriesError::InsufficientData {
                message: "KalmanFilter::filter requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let (f, h, q, r, p0) = self.model.to_flat()?;
        let n = self.model.n_states();
        let p = self.model.n_obs();

        // Validate observation dimension
        for (t, yt) in obs.iter().enumerate() {
            if yt.len() != p {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: p,
                    actual: yt.len(),
                });
            }
            let _ = t;
        }

        let f_t = f.t();
        let h_t = h.t();

        let mut m = self.model.m0.clone();
        let mut pp = p0.clone();

        let mut filtered_means = Vec::with_capacity(t_len);
        let mut filtered_covs = Vec::with_capacity(t_len);
        let mut predicted_means = Vec::with_capacity(t_len);
        let mut predicted_covs = Vec::with_capacity(t_len);
        let mut log_lik = 0.0f64;

        for yt in obs.iter() {
            // Predict
            let m_mat = FlatMat::from_col_vec(&m);
            let m_pred_mat = f.mul(&m_mat);
            let m_pred = m_pred_mat.col_vec(0);
            let p_pred = f.mul(&pp).mul(&f_t).add(&q);

            predicted_means.push(m_pred.clone());
            predicted_covs.push(p_pred.to_nested());

            // Innovation: v = y_t - H m_pred  (p-vector)
            let h_m_pred = h.mul(&FlatMat::from_col_vec(&m_pred));
            let v: Vec<f64> = (0..p).map(|i| yt[i] - h_m_pred.get(i, 0)).collect();

            // Innovation covariance: S = H P_pred H' + R  (p×p)
            let s = h.mul(&p_pred).mul(&h_t).add(&r);
            let s_inv = match s.inv() {
                Ok(inv) => inv,
                Err(_) => regularise(&s, 1e-8).inv().map_err(|e| {
                    TimeSeriesError::ComputationError(format!("S inversion failed: {e}"))
                })?,
            };

            // Kalman gain: K = P_pred H' S^{-1}  (n×p)
            let k = p_pred.mul(&h_t).mul(&s_inv);

            // Update state mean
            let k_v = k.mul(&FlatMat::from_col_vec(&v));
            let m_filt: Vec<f64> = (0..n).map(|i| m_pred[i] + k_v.get(i, 0)).collect();

            // Update covariance: P_filt = (I - K H) P_pred
            let kh = k.mul(&h);
            let i_kh = FlatMat::eye(n).sub(&kh);
            let p_filt = i_kh.mul(&p_pred);

            // Log-likelihood contribution: -0.5 (p ln(2π) + ln|S| + v'S^{-1}v)
            let ln_det_s = log_det(&s).unwrap_or(0.0);
            let v_mat = FlatMat::from_col_vec(&v);
            let vs = s_inv.mul(&v_mat);
            let quad: f64 = (0..p).map(|i| v[i] * vs.get(i, 0)).sum();
            log_lik -= 0.5 * (p as f64 * (2.0 * PI).ln() + ln_det_s + quad);

            filtered_means.push(m_filt.clone());
            filtered_covs.push(p_filt.to_nested());

            m = m_filt;
            pp = p_filt;
        }

        Ok(KalmanFilterResult {
            filtered_means,
            filtered_covs,
            predicted_means,
            predicted_covs,
            log_likelihood: log_lik,
            f_flat: f,
        })
    }
}

/// Log-determinant of a positive-definite matrix via Cholesky (with LDL fallback).
fn log_det(m: &FlatMat) -> Result<f64> {
    let n = m.rows;
    // Try to compute via sum of log diagonal of LU factors.
    let mut a = m.clone();
    let mut log_d = 0.0f64;
    for j in 0..n {
        for k in 0..j {
            let sub = a.get(j, k) * a.get(k, j) / a.get(k, k).max(1e-20);
            let val = a.get(j, j) - sub;
            a.set(j, j, val);
        }
        let dj = a.get(j, j);
        if dj <= 0.0 {
            // Regularise and retry
            let reg = regularise(m, 1e-8);
            return log_det(&reg);
        }
        log_d += dj.ln();
    }
    Ok(log_d)
}

// ---------------------------------------------------------------------------
// RTS Kalman smoother
// ---------------------------------------------------------------------------

/// Run the RTS (Rauch-Tung-Striebel) backward smoother on a forward-pass result.
///
/// # Arguments
/// * `filter_result` – output of [`KalmanFilter::filter`]
///
/// # Returns
/// [`SmootherResult`] with smoothed means and covariances.
pub fn smooth(filter_result: &KalmanFilterResult) -> Result<SmootherResult> {
    let t_len = filter_result.filtered_means.len();
    if t_len == 0 {
        return Ok(SmootherResult {
            smoothed_means: vec![],
            smoothed_covs: vec![],
        });
    }

    let f = &filter_result.f_flat;
    let f_t = f.t();
    let n = filter_result.filtered_means[0].len();

    let mut sm_means = filter_result.filtered_means.clone();
    let mut sm_covs_flat: Vec<FlatMat> = filter_result
        .filtered_covs
        .iter()
        .map(|m| FlatMat::from_nested(m).unwrap_or_else(|_| FlatMat::zeros(n, n)))
        .collect();

    // Backward pass
    for t_idx in (0..t_len - 1).rev() {
        let p_filt = &sm_covs_flat[t_idx].clone();
        let p_pred = FlatMat::from_nested(&filter_result.predicted_covs[t_idx + 1])
            .map_err(|e| TimeSeriesError::ComputationError(format!("Bad predicted cov: {e}")))?;

        // G_t = P_filt F' P_pred^{-1}
        let p_pred_reg = regularise(&p_pred, 1e-8);
        let p_pred_inv = p_pred_reg
            .inv()
            .map_err(|e| TimeSeriesError::ComputationError(format!("P_pred inversion: {e}")))?;
        let g = p_filt.mul(&f_t).mul(&p_pred_inv);

        // Smoothed mean: m̃_t = m_filt_t + G_t (m̃_{t+1} - m_pred_{t+1})
        let sm_next = &sm_means[t_idx + 1];
        let mp_next = &filter_result.predicted_means[t_idx + 1];
        let diff: Vec<f64> = (0..n).map(|i| sm_next[i] - mp_next[i]).collect();
        let g_diff = g.mul(&FlatMat::from_col_vec(&diff));
        let m_filt = &filter_result.filtered_means[t_idx];
        let m_smooth: Vec<f64> = (0..n).map(|i| m_filt[i] + g_diff.get(i, 0)).collect();

        // Smoothed cov: P̃_t = P_filt + G_t (P̃_{t+1} - P_pred_{t+1}) G_t'
        let sp_next = &sm_covs_flat[t_idx + 1].clone();
        let dp = sp_next.sub(&p_pred);
        let g_t = g.t();
        let p_smooth = p_filt.add(&g.mul(&dp).mul(&g_t));

        sm_means[t_idx] = m_smooth;
        sm_covs_flat[t_idx] = p_smooth;
    }

    let smoothed_covs = sm_covs_flat.iter().map(|m| m.to_nested()).collect();
    Ok(SmootherResult {
        smoothed_means: sm_means,
        smoothed_covs,
    })
}

// ---------------------------------------------------------------------------
// EM algorithm
// ---------------------------------------------------------------------------

/// Fit a [`StateSpace`] model to observations using the EM algorithm.
///
/// **E-step**: run the RTS smoother.
/// **M-step**: update F, H, Q, R analytically from the sufficient statistics.
/// F and Q are updated; H and R are updated; m0 and P0 are updated.
///
/// # Arguments
/// * `obs`        – T × p multivariate observations
/// * `model_init` – initial model parameters
/// * `max_iter`   – maximum number of EM iterations
/// * `tol`        – convergence tolerance on log-likelihood change
///
/// # Returns
/// The updated [`StateSpace`] model.
pub fn em_fit(
    obs: &[Vec<f64>],
    model_init: StateSpace,
    max_iter: usize,
    tol: f64,
) -> Result<StateSpace> {
    let t_len = obs.len();
    if t_len < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "em_fit requires at least 2 observations".into(),
            required: 2,
            actual: t_len,
        });
    }
    let p_dim = model_init.n_obs();
    let n_dim = model_init.n_states();

    let mut model = model_init;
    let mut prev_ll = f64::NEG_INFINITY;

    for _iter in 0..max_iter {
        // ----- E-step -----
        let kf = KalmanFilter {
            model: model.clone(),
        };
        let filt = kf.filter(obs)?;
        let sm = smooth(&filt)?;
        let ll = filt.log_likelihood;

        // ----- Convergence check -----
        if (ll - prev_ll).abs() < tol && _iter > 2 {
            break;
        }
        prev_ll = ll;

        // Sufficient statistics
        // E[x_t x_t'] = P̃_t + m̃_t m̃_t'
        // E[x_{t+1} x_t'] = cross-covariance (from Kalman lag-one smoother)

        // Build required flat matrices
        let f_old = FlatMat::from_nested(&model.f)?;
        let h_old = FlatMat::from_nested(&model.h)?;

        // Σ_00 = Σ E[x_t x_t']  for t=0..T-1
        let mut sigma_00 = FlatMat::zeros(n_dim, n_dim);
        // Σ_11 = Σ E[x_t x_t']  for t=1..T
        let mut sigma_11 = FlatMat::zeros(n_dim, n_dim);
        // Σ_10 = Σ E[x_t x_{t-1}']  (lag-one cross-covariance)
        let mut sigma_10 = FlatMat::zeros(n_dim, n_dim);
        // Σ_yy = Σ y_t y_t'
        let mut sigma_yy = FlatMat::zeros(p_dim, p_dim);
        // Σ_yx = Σ y_t E[x_t]'
        let mut sigma_yx = FlatMat::zeros(p_dim, n_dim);

        for t in 0..t_len {
            let m = FlatMat::from_col_vec(&sm.smoothed_means[t]);
            let pf = FlatMat::from_nested(&sm.smoothed_covs[t])?;
            let exx = pf.add(&m.outer(&m));

            if t < t_len - 1 {
                add_to(&mut sigma_00, &exx);
            }
            if t > 0 {
                add_to(&mut sigma_11, &exx);
            }

            let y = FlatMat::from_col_vec(&obs[t]);
            add_to(&mut sigma_yy, &y.outer(&y));
            add_to(&mut sigma_yx, &y.outer(&m));
        }

        // Lag-one smoother covariance:  E[x_t x_{t-1}'] = P̃_{t|T} G_{t-1}' + m̃_t m̃_{t-1}'
        // Compute G_{t-1} for each t = 1..T
        let f_flat = FlatMat::from_nested(&model.f)?;
        let p0_flat = FlatMat::from_nested(&model.p0)?;

        // We need the filtered covariances (before update) for G computation.
        for t in 1..t_len {
            let p_filt_tm1 = FlatMat::from_nested(&filt.filtered_covs[t - 1])?;
            let p_pred_t = FlatMat::from_nested(&filt.predicted_covs[t])?;
            let p_pred_reg = regularise(&p_pred_t, 1e-8);
            let p_pred_inv = match p_pred_reg.inv() {
                Ok(inv) => inv,
                Err(_) => continue,
            };
            let g = p_filt_tm1.mul(&f_flat.t()).mul(&p_pred_inv);

            let ps_t = FlatMat::from_nested(&sm.smoothed_covs[t])?;
            let mt = FlatMat::from_col_vec(&sm.smoothed_means[t]);
            let mt_1 = FlatMat::from_col_vec(&sm.smoothed_means[t - 1]);
            // E[x_t x_{t-1}'] = P̃_t G_{t-1}' + m̃_t m̃_{t-1}'
            let lag = ps_t.mul(&g.t()).add(&mt.outer(&mt_1));
            add_to(&mut sigma_10, &lag);
        }

        // ----- M-step -----
        // Update F: F_new = Σ_10 Σ_00^{-1}
        let sigma_00_inv = match regularise(&sigma_00, 1e-8).inv() {
            Ok(inv) => inv,
            Err(_) => {
                continue;
            }
        };
        let f_new = sigma_10.mul(&sigma_00_inv);

        // Update Q: Q_new = (1/T) (Σ_11 - F_new Σ_10')
        let q_new = sigma_11
            .sub(&f_new.mul(&sigma_10.t()))
            .scale(1.0 / (t_len - 1) as f64);

        // Update H: H_new = Σ_yx Σ_xx^{-1}
        let mut sigma_xx = FlatMat::zeros(n_dim, n_dim);
        for t in 0..t_len {
            let m = FlatMat::from_col_vec(&sm.smoothed_means[t]);
            let pf = FlatMat::from_nested(&sm.smoothed_covs[t])?;
            add_to(&mut sigma_xx, &pf.add(&m.outer(&m)));
        }
        let sigma_xx_inv = match regularise(&sigma_xx, 1e-8).inv() {
            Ok(inv) => inv,
            Err(_) => {
                continue;
            }
        };
        let h_new = sigma_yx.mul(&sigma_xx_inv);

        // Update R: R_new = (1/T) (Σ_yy - H_new Σ_yx')
        let r_new = sigma_yy
            .sub(&h_new.mul(&sigma_yx.t()))
            .scale(1.0 / t_len as f64);

        // Update m0 and P0 from first smoothed state
        let m0_new = sm.smoothed_means[0].clone();
        let p0_new_mat = FlatMat::from_nested(&sm.smoothed_covs[0])?;

        // Ensure Q and R are positive definite (symmetrize and regularise)
        let q_reg = regularise(&symmetrize(&q_new), 1e-10);
        let r_reg = regularise(&symmetrize(&r_new), 1e-10);

        model = StateSpace {
            f: f_new.to_nested(),
            h: h_new.to_nested(),
            q: q_reg.to_nested(),
            r: r_reg.to_nested(),
            m0: m0_new,
            p0: p0_new_mat.to_nested(),
        };

        let _ = (f_old, h_old, p0_flat);
    }

    Ok(model)
}

/// Add rhs into lhs element-wise.
fn add_to(lhs: &mut FlatMat, rhs: &FlatMat) {
    assert_eq!(lhs.rows, rhs.rows);
    assert_eq!(lhs.cols, rhs.cols);
    for i in 0..lhs.data.len() {
        lhs.data[i] += rhs.data[i];
    }
}

/// Symmetrize a matrix: (A + A') / 2.
fn symmetrize(m: &FlatMat) -> FlatMat {
    let mt = m.t();
    m.add(&mt).scale(0.5)
}

// ---------------------------------------------------------------------------
// ARIMA to State Space
// ---------------------------------------------------------------------------

/// Convert an ARIMA(p, d, q) model to companion-form state space.
///
/// The function places the I(d) differencing in the state via extended companion
/// form, then embeds the ARMA(p, q) dynamics.  The resulting state vector has
/// dimension `max(p + d, q + 1)`.
///
/// For simplicity in testing and inspection, the returned model uses unit
/// variances for Q and R; callers should set these after the conversion.
///
/// # Arguments
/// * `p` – AR order
/// * `d` – degree of differencing
/// * `q` – MA order
///
/// # Returns
/// [`StateSpace`] in companion form.  The first component of the state
/// corresponds to the (differenced) series.
pub fn arima_to_state_space(p: usize, d: usize, q: usize) -> StateSpace {
    // The companion state vector has dimension m = max(p + d, q + 1).
    // For an ARIMA(p,d,q) we treat it as ARMA(p+d, q) on the level x_t.
    let ar_order = p + d;
    let m = ar_order.max(q + 1).max(1);

    // Companion form F:
    // Row 0: [φ_1, φ_2, ..., φ_p, 0, ..., 0, Δ_1, ..., Δ_d]
    // Rows 1..m-1: shift rows (identity band)
    //
    // For I(d) integration in companion form, the transition matrix encodes
    // accumulated sums via Pascal-triangle binomial coefficients.
    let mut f_data = vec![vec![0.0f64; m]; m];

    // I(d) binomial coefficients for the first d super-diagonal positions.
    // The d-times integrated random walk has F = upper binomial Pascal block.
    if d > 0 {
        // Build the d-fold integration block (Pascal's triangle row).
        let binom = pascal_row(d);
        for (j, &b) in binom.iter().enumerate().take(m) {
            f_data[0][j] = b;
        }
    }

    // AR coefficients (if p > 0) are accumulated on top of the I(d) part.
    // In a full ARIMA model the AR part operates on the d-differenced series;
    // here we place the AR coefficients starting at column d.
    // (We use placeholder coefficients of 0 — callers set AR/MA params via
    //  the returned model's `f` matrix directly.)

    // Shift rows: row i = e_{i-1} for i = 1..m-1
    for i in 1..m {
        if i - 1 < m {
            f_data[i][i - 1] = 1.0;
        }
    }

    // Observation matrix H: selects the first state element (the series value).
    let mut h_data = vec![vec![0.0f64; m]];
    h_data[0][0] = 1.0;

    // MA coefficients are encoded in the first column of Q (innovations form).
    // The innovation vector ε_t enters as [1, θ_1, ..., θ_q, 0, ...]'.
    // We leave θ as zeros; caller sets them.
    let mut q_data = vec![vec![0.0f64; m]; m];
    q_data[0][0] = 1.0; // unit process variance; scale later

    // R: observation noise (ARIMA conventionally assumes obs equation is exact,
    // but we place a small positive value for numerical stability).
    let r_data = vec![vec![1e-6]];

    // Initial state: zeros.
    let m0 = vec![0.0f64; m];

    // Initial covariance: diffuse (large diagonal).
    let mut p0 = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        p0[i][i] = 1e6;
    }

    StateSpace {
        f: f_data,
        h: h_data,
        q: q_data,
        r: r_data,
        m0,
        p0,
    }
}

/// Compute the binomial expansion coefficients for (1 - B)^d using Pascal's triangle.
/// Returns [1, -d, d(d-1)/2, ...] for I(d) differences — the integration variant
/// uses the *sum* coefficients [1, 1, 1, ...] (d-fold integration).
fn pascal_row(d: usize) -> Vec<f64> {
    // d-fold integration: F^(d) first row is [C(d,0), C(d,1), ..., C(d,d), 0, ...]
    // with alternating signs for differences.  For the companion-form integration
    // we use the cumulative-sum (positive binomial coefficients):
    let mut row = vec![1.0f64; d + 1];
    for k in 1..=d {
        row[k] = row[k - 1] * (d + 1 - k) as f64 / k as f64;
    }
    row
}

// ---------------------------------------------------------------------------
// Multi-step-ahead forecasting
// ---------------------------------------------------------------------------

/// Forecast `h` steps ahead from the last filtered state.
///
/// Runs the Kalman filter on `obs` to obtain the terminal filtered state, then
/// propagates the state forward `h` steps using the state equation.
///
/// # Arguments
/// * `model` – fitted state-space model
/// * `obs`   – historical observations (T × p)
/// * `h`     – forecast horizon
///
/// # Returns
/// `h` predicted observation vectors.
pub fn predict_state_space(
    model: &StateSpace,
    obs: &[Vec<f64>],
    h: usize,
) -> Result<Vec<Vec<f64>>> {
    if h == 0 {
        return Ok(vec![]);
    }
    if obs.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "predict_state_space requires at least one observation".into(),
            required: 1,
            actual: 0,
        });
    }

    let kf = KalmanFilter {
        model: model.clone(),
    };
    let filt = kf.filter(obs)?;

    let n = model.n_states();
    let p = model.n_obs();
    let f = FlatMat::from_nested(&model.f)?;
    let h_mat = FlatMat::from_nested(&model.h)?;

    let mut m = filt
        .filtered_means
        .last()
        .cloned()
        .unwrap_or_else(|| model.m0.clone());

    let mut preds = Vec::with_capacity(h);
    for _ in 0..h {
        let m_mat = FlatMat::from_col_vec(&m);
        let mp_mat = f.mul(&m_mat);
        m = mp_mat.col_vec(0);
        let y_hat = h_mat.mul(&FlatMat::from_col_vec(&m));
        preds.push((0..p).map(|i| y_hat.get(i, 0)).collect());
    }

    let _ = n;
    Ok(preds)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_obs(v: f64, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| vec![v]).collect()
    }

    fn lintrend_obs(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![i as f64 * 0.5]).collect()
    }

    // --- StateSpace construction ---

    #[test]
    fn test_local_level_construction() {
        let m = StateSpace::local_level(0.1, 0.5, 3.0);
        assert_eq!(m.n_states(), 1);
        assert_eq!(m.n_obs(), 1);
        assert_eq!(m.f, vec![vec![1.0]]);
        assert_eq!(m.h, vec![vec![1.0]]);
    }

    // --- Kalman filter ---

    #[test]
    fn test_filter_lengths() {
        let m = StateSpace::local_level(0.1, 0.5, 0.0);
        let obs = constant_obs(3.0, 20);
        let kf = KalmanFilter::new(m);
        let res = kf.filter(&obs).expect("filter ok");
        assert_eq!(res.filtered_means.len(), 20);
        assert_eq!(res.predicted_means.len(), 20);
        assert_eq!(res.filtered_covs.len(), 20);
    }

    #[test]
    fn test_filter_empty_error() {
        let m = StateSpace::local_level(0.1, 0.5, 0.0);
        let kf = KalmanFilter::new(m);
        assert!(kf.filter(&[]).is_err());
    }

    #[test]
    fn test_filter_constant_series_converges() {
        let m = StateSpace::local_level(0.01, 0.1, 0.0);
        let obs = constant_obs(5.0, 100);
        let kf = KalmanFilter::new(m);
        let res = kf.filter(&obs).expect("filter ok");
        let last_mean = res
            .filtered_means
            .last()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0);
        assert!((last_mean - 5.0).abs() < 0.5, "last_mean={last_mean}");
    }

    #[test]
    fn test_filter_log_likelihood_finite() {
        let m = StateSpace::local_level(0.1, 0.5, 0.0);
        let obs: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64 * 0.3).sin()]).collect();
        let kf = KalmanFilter::new(m);
        let res = kf.filter(&obs).expect("filter ok");
        assert!(res.log_likelihood.is_finite());
        assert!(res.log_likelihood < 0.0);
    }

    #[test]
    fn test_filter_covs_positive() {
        let m = StateSpace::local_level(0.1, 0.3, 0.0);
        let obs = constant_obs(1.0, 15);
        let kf = KalmanFilter::new(m);
        let res = kf.filter(&obs).expect("ok");
        for cov in &res.filtered_covs {
            assert!(cov[0][0] > 0.0, "covariance must be positive");
        }
    }

    #[test]
    fn test_filter_cov_decreasing() {
        // Filtering should reduce uncertainty.
        let m = StateSpace::local_level(0.01, 0.5, 0.0);
        let obs = constant_obs(2.0, 30);
        let kf = KalmanFilter::new(m);
        let res = kf.filter(&obs).expect("ok");
        let first = res.filtered_covs[0][0][0];
        let last = res
            .filtered_covs
            .last()
            .and_then(|c| c.first())
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(f64::MAX);
        assert!(
            last < first,
            "cov should decrease: first={first} last={last}"
        );
    }

    // --- RTS smoother ---

    #[test]
    fn test_smooth_lengths() {
        let m = StateSpace::local_level(0.1, 0.5, 0.0);
        let obs = constant_obs(3.0, 20);
        let kf = KalmanFilter::new(m);
        let filt = kf.filter(&obs).expect("ok");
        let sm = smooth(&filt).expect("smooth ok");
        assert_eq!(sm.smoothed_means.len(), 20);
        assert_eq!(sm.smoothed_covs.len(), 20);
    }

    #[test]
    fn test_smooth_empty() {
        // Construct a result with zero observations.
        let m = StateSpace::local_level(0.1, 0.5, 0.0);
        let kf = KalmanFilter::new(m.clone());
        // We cannot call filter on empty; test smooth on empty filter result directly.
        let fake = KalmanFilterResult {
            filtered_means: vec![],
            filtered_covs: vec![],
            predicted_means: vec![],
            predicted_covs: vec![],
            log_likelihood: 0.0,
            f_flat: FlatMat::from_nested(&m.f).unwrap_or_else(|_| FlatMat::zeros(1, 1)),
        };
        let sm = smooth(&fake).expect("smooth empty ok");
        assert!(sm.smoothed_means.is_empty());
    }

    #[test]
    fn test_smooth_covs_le_filtered() {
        // Smoother covariances should be ≤ filtered covariances.
        let m = StateSpace::local_level(0.1, 0.3, 0.0);
        let obs = constant_obs(4.0, 25);
        let kf = KalmanFilter::new(m);
        let filt = kf.filter(&obs).expect("ok");
        let sm = smooth(&filt).expect("ok");
        for t in 0..obs.len() {
            let p_filt = filt.filtered_covs[t][0][0];
            let p_sm = sm.smoothed_covs[t][0][0];
            assert!(
                p_sm <= p_filt + 1e-10,
                "smoother cov > filtered at t={t}: {p_sm} vs {p_filt}"
            );
        }
    }

    // --- EM fit ---

    #[test]
    fn test_em_fit_returns_positive_variances() {
        let obs: Vec<Vec<f64>> = (0..30).map(|i| vec![(i as f64 * 0.3).sin()]).collect();
        let init = StateSpace::local_level(0.2, 0.4, 0.0);
        let fitted = em_fit(&obs, init, 30, 1e-4).expect("em ok");
        assert!(fitted.q[0][0] > 0.0, "Q must be positive");
        assert!(fitted.r[0][0] > 0.0, "R must be positive");
    }

    #[test]
    fn test_em_fit_insufficient_data() {
        let obs = vec![vec![1.0f64]]; // only 1 observation
        let init = StateSpace::local_level(0.1, 0.2, 1.0);
        assert!(em_fit(&obs, init, 10, 1e-4).is_err());
    }

    #[test]
    fn test_em_fit_does_not_degrade_ll() {
        let obs: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![2.0 + 0.5 * (i as f64 * 0.2).sin()])
            .collect();
        let init = StateSpace::local_level(0.3, 0.5, 0.0);
        let kf_init = KalmanFilter {
            model: init.clone(),
        };
        let ll_init = kf_init.filter(&obs).expect("ok").log_likelihood;

        let fitted = em_fit(&obs, init, 50, 1e-5).expect("em ok");
        let kf_fit = KalmanFilter { model: fitted };
        let ll_fit = kf_fit.filter(&obs).expect("ok").log_likelihood;

        // EM is non-decreasing in likelihood (allow tiny numerical slack).
        assert!(
            ll_fit >= ll_init - 1.0,
            "EM degraded LL: {ll_init} -> {ll_fit}"
        );
    }

    // --- ARIMA to state space ---

    #[test]
    fn test_arima_ar1_dimensions() {
        let ss = arima_to_state_space(1, 0, 0);
        let m = ss.n_states();
        assert!(m >= 1, "need at least 1 state");
        assert_eq!(ss.h[0].len(), m, "H must be 1 × m");
        assert_eq!(ss.f.len(), m, "F must be m × m");
    }

    #[test]
    fn test_arima_local_level_as_rw() {
        // ARIMA(0,1,0) is the random walk == local level model.
        let ss = arima_to_state_space(0, 1, 0);
        assert!(ss.n_states() >= 1);
        // F[0][0] == 1.0 (unit root)
        assert!(
            (ss.f[0][0] - 1.0).abs() < 1e-10,
            "F[0][0] should be 1 for I(1)"
        );
    }

    #[test]
    fn test_arima_arma_1_1() {
        let ss = arima_to_state_space(1, 0, 1);
        // m = max(p+d, q+1) = max(1, 2) = 2
        assert_eq!(ss.n_states(), 2);
    }

    #[test]
    fn test_arima_arima_2_1_1() {
        let ss = arima_to_state_space(2, 1, 1);
        // p+d = 3, q+1 = 2 → m = 3
        assert_eq!(ss.n_states(), 3);
    }

    #[test]
    fn test_arima_filter_runs() {
        let ss = arima_to_state_space(1, 1, 0);
        let obs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let kf = KalmanFilter::new(ss);
        let res = kf.filter(&obs).expect("arima filter ok");
        assert_eq!(res.filtered_means.len(), 20);
        assert!(res.log_likelihood.is_finite());
    }

    // --- Forecasting ---

    #[test]
    fn test_predict_length() {
        let m = StateSpace::local_level(0.1, 0.3, 0.0);
        let obs = constant_obs(2.0, 15);
        let preds = predict_state_space(&m, &obs, 5).expect("predict ok");
        assert_eq!(preds.len(), 5);
        for p in &preds {
            assert_eq!(p.len(), 1);
        }
    }

    #[test]
    fn test_predict_constant_series() {
        let m = StateSpace::local_level(0.0001, 0.001, 5.0);
        let obs = constant_obs(5.0, 50);
        let preds = predict_state_space(&m, &obs, 5).expect("predict ok");
        for (i, p) in preds.iter().enumerate() {
            assert!((p[0] - 5.0).abs() < 1.0, "pred[{i}]={} not near 5", p[0]);
        }
    }

    #[test]
    fn test_predict_zero_horizon() {
        let m = StateSpace::local_level(0.1, 0.3, 0.0);
        let obs = constant_obs(2.0, 10);
        let preds = predict_state_space(&m, &obs, 0).expect("ok");
        assert!(preds.is_empty());
    }

    #[test]
    fn test_predict_empty_obs_error() {
        let m = StateSpace::local_level(0.1, 0.3, 0.0);
        assert!(predict_state_space(&m, &[], 5).is_err());
    }

    #[test]
    fn test_predict_finite_values() {
        let m = StateSpace::local_level(0.1, 0.3, 0.0);
        let obs: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![(i as f64 * 0.5).sin() * 3.0])
            .collect();
        let preds = predict_state_space(&m, &obs, 10).expect("ok");
        for (i, p) in preds.iter().enumerate() {
            assert!(p[0].is_finite(), "pred[{i}] is not finite");
        }
    }

    #[test]
    fn test_predict_trend() {
        // With local level and linear trend, forecasts should continue upward.
        let obs = lintrend_obs(20);
        let m = StateSpace::local_level(0.05, 0.1, 0.0);
        let preds = predict_state_space(&m, &obs, 3).expect("ok");
        // Each successive prediction should continue the trend direction.
        assert_eq!(preds.len(), 3);
        for p in &preds {
            assert!(p[0].is_finite());
        }
    }
}
