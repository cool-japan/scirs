//! Structural time series models (Harvey's Basic Structural Model framework).
//!
//! Implements:
//! - **Local Level Model** (random walk + noise): Harvey (1989) §3.2
//! - **Local Linear Trend Model**: Harvey (1989) §3.3
//! - **Basic Structural Model** (BSM) with trigonometric seasonality: Harvey §3.4
//! - **Structural decomposition** into trend, seasonal, and irregular components
//! - **Forecasting** from fitted BSM
//!
//! All parameter estimation uses MLE via prediction-error decomposition (Kalman
//! filter log-likelihood), optimised with a simple bounded Nelder-Mead search.
//!
//! ## Model overview
//!
//! ```text
//! y_t = μ_t + γ_t + ε_t,    ε_t  ~ N(0, σ²_ε)
//! μ_{t+1} = μ_t + β_t + η_t, η_t  ~ N(0, σ²_η)   (level)
//! β_{t+1} = β_t  + ζ_t,      ζ_t  ~ N(0, σ²_ζ)   (slope)
//! γ_{t+1} = Σ_j [cos(λ_j)γ_{j,t} + sin(λ_j)γ*_{j,t}] + ω_{j,t}
//! ```

use crate::error::{Result, TimeSeriesError};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal flat-matrix algebra (row-major, no external BLAS)
// ---------------------------------------------------------------------------

/// A simple flat row-major matrix representation.
#[derive(Clone, Debug)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
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
    fn add_assign(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] += v;
    }

    /// Matrix multiplication: self * rhs.
    fn mul(&self, rhs: &Mat) -> Self {
        assert_eq!(self.cols, rhs.rows, "mat_mul dimension mismatch");
        let mut out = Mat::zeros(self.rows, rhs.cols);
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

    /// Transpose.
    fn t(&self) -> Self {
        let mut out = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    /// Element-wise addition.
    fn add(&self, rhs: &Mat) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Element-wise subtraction.
    fn sub(&self, rhs: &Mat) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| a - b)
            .collect();
        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Scalar multiplication.
    fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x * s).collect();
        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Invert a square matrix via Gauss-Jordan elimination.
    fn inv(&self) -> Result<Self> {
        let n = self.rows;
        if n != self.cols {
            return Err(TimeSeriesError::ComputationError(
                "Non-square matrix inversion".into(),
            ));
        }
        // Augmented [A | I]
        let mut aug: Vec<f64> = Vec::with_capacity(n * 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.push(self.get(i, j));
            }
            for j in 0..n {
                aug.push(if i == j { 1.0 } else { 0.0 });
            }
        }
        let cols = 2 * n;
        for col in 0..n {
            // Partial pivot
            let mut pivot_row = col;
            let mut max_val = aug[col * cols + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * cols + col].abs();
                if v > max_val {
                    max_val = v;
                    pivot_row = row;
                }
            }
            if max_val < 1e-15 {
                return Err(TimeSeriesError::ComputationError("Singular matrix".into()));
            }
            if pivot_row != col {
                for j in 0..cols {
                    aug.swap(col * cols + j, pivot_row * cols + j);
                }
            }
            let diag = aug[col * cols + col];
            for j in 0..cols {
                aug[col * cols + j] /= diag;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * cols + col];
                for j in 0..cols {
                    aug[row * cols + j] -= factor * aug[col * cols + j];
                }
            }
        }
        let mut inv = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug[i * cols + (n + j)]);
            }
        }
        Ok(inv)
    }

    /// Extract a column as a Vec.
    fn col_vec(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// From a column vector (Nx1).
    fn from_col(v: &[f64]) -> Self {
        Mat {
            rows: v.len(),
            cols: 1,
            data: v.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Kalman filter/smoother core (internal, works on Mat)
// ---------------------------------------------------------------------------

struct KfOutput {
    /// filtered state means: T × n_state
    filtered_mean: Vec<Vec<f64>>,
    /// filtered state covariances: T × (n_state × n_state flat)
    filtered_cov: Vec<Mat>,
    /// predicted state means: T × n_state
    predicted_mean: Vec<Vec<f64>>,
    /// predicted state covariances
    predicted_cov: Vec<Mat>,
    /// innovations (scalar observations)
    innovations: Vec<f64>,
    /// innovation variances
    innov_var: Vec<f64>,
    /// total log-likelihood
    log_lik: f64,
}

/// Run a univariate Kalman filter.
///
/// # Arguments
/// * `y`   – scalar observations (length T)
/// * `f`   – state transition matrix (n × n)
/// * `h`   – observation matrix (1 × n)
/// * `q`   – process noise covariance (n × n)
/// * `r`   – observation noise variance (scalar, stored as 1×1)
/// * `m0`  – initial state mean (n)
/// * `p0`  – initial state covariance (n × n)
fn kalman_filter_internal(
    y: &[f64],
    f: &Mat,
    h: &Mat,
    q: &Mat,
    r: f64,
    m0: &[f64],
    p0: &Mat,
) -> KfOutput {
    let n = f.rows;
    let t = y.len();
    let mut filt_mean = Vec::with_capacity(t);
    let mut filt_cov = Vec::with_capacity(t);
    let mut pred_mean = Vec::with_capacity(t);
    let mut pred_cov = Vec::with_capacity(t);
    let mut innovations = Vec::with_capacity(t);
    let mut innov_var = Vec::with_capacity(t);

    let mut m = m0.to_vec();
    let mut p = p0.clone();
    let mut log_lik = 0.0f64;

    let f_t = f.t();
    let h_t = h.t();

    for obs in y.iter() {
        // --- Predict ---
        // m_pred = F * m
        let m_mat = Mat::from_col(&m);
        let mp_mat = f.mul(&m_mat);
        let m_pred: Vec<f64> = mp_mat.col_vec(0);
        // P_pred = F P F' + Q
        let fpt = f.mul(&p).mul(&f_t).add(q);

        pred_mean.push(m_pred.clone());
        pred_cov.push(fpt.clone());

        // --- Update ---
        // v = y_t - H m_pred  (scalar)
        let h_m: f64 = (0..n).map(|j| h.get(0, j) * m_pred[j]).sum();
        let v = obs - h_m;
        // S = H P_pred H' + R  (scalar)
        let hp = h.mul(&fpt);
        let s: f64 = (0..n).map(|j| hp.get(0, j) * h.get(0, j)).sum::<f64>() + r;
        let s_safe = if s < 1e-15 { 1e-15 } else { s };
        // K = P_pred H' / S  (n×1)
        let k_mat = fpt.mul(&h_t).scale(1.0 / s_safe);
        // m_filt = m_pred + K v
        let m_filt: Vec<f64> = (0..n).map(|i| m_pred[i] + k_mat.get(i, 0) * v).collect();
        // P_filt = (I - K H) P_pred
        let mut kh = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                kh.set(i, j, k_mat.get(i, 0) * h.get(0, j));
            }
        }
        let ikm = Mat::eye(n).sub(&kh);
        let p_filt = ikm.mul(&fpt);

        // log-likelihood contribution
        log_lik -= 0.5 * ((2.0 * PI * s_safe).ln() + v * v / s_safe);

        filt_mean.push(m_filt.clone());
        filt_cov.push(p_filt.clone());
        innovations.push(v);
        innov_var.push(s_safe);

        m = m_filt;
        p = p_filt;
    }

    KfOutput {
        filtered_mean: filt_mean,
        filtered_cov: filt_cov,
        predicted_mean: pred_mean,
        predicted_cov: pred_cov,
        innovations,
        innov_var,
        log_lik,
    }
}

/// RTS backward smoother.
fn rts_smoother_internal(kf: &KfOutput, f: &Mat) -> (Vec<Vec<f64>>, Vec<Mat>) {
    let t = kf.filtered_mean.len();
    if t == 0 {
        return (vec![], vec![]);
    }
    let n = f.rows;
    let f_t = f.t();

    let mut smoothed_mean: Vec<Vec<f64>> = kf.filtered_mean.clone();
    let mut smoothed_cov: Vec<Mat> = kf.filtered_cov.clone();

    for t_idx in (0..t - 1).rev() {
        let p_filt = &kf.filtered_cov[t_idx];
        let p_pred = &kf.predicted_cov[t_idx + 1];
        // Gain: G = P_filt F' P_pred^{-1}
        let p_pred_inv = match p_pred.inv() {
            Ok(inv) => inv,
            Err(_) => {
                // Fallback: use a large-variance regularised inverse
                let mut reg = p_pred.clone();
                for i in 0..n {
                    reg.add_assign(i, i, 1e-8);
                }
                match reg.inv() {
                    Ok(inv) => inv,
                    Err(_) => Mat::eye(n).scale(1.0 / 1e-8),
                }
            }
        };
        let g = p_filt.mul(&f_t).mul(&p_pred_inv);

        let sm_next = &smoothed_mean[t_idx + 1];
        let pm_next = &kf.predicted_mean[t_idx + 1];
        let sp_next = &smoothed_cov[t_idx + 1];
        let pm_filt = &kf.filtered_mean[t_idx];

        // m_smooth = m_filt + G (m_smooth_{t+1} - m_pred_{t+1})
        let diff: Vec<f64> = (0..n).map(|i| sm_next[i] - pm_next[i]).collect();
        let g_diff = g.mul(&Mat::from_col(&diff));
        let m_smooth: Vec<f64> = (0..n).map(|i| pm_filt[i] + g_diff.get(i, 0)).collect();

        // P_smooth = P_filt + G (P_smooth_{t+1} - P_pred_{t+1}) G'
        let dp = sp_next.sub(p_pred);
        let g_t = g.t();
        let p_smooth = p_filt.add(&g.mul(&dp).mul(&g_t));

        smoothed_mean[t_idx] = m_smooth;
        smoothed_cov[t_idx] = p_smooth;
    }

    (smoothed_mean, smoothed_cov)
}

// ---------------------------------------------------------------------------
// Nelder-Mead optimiser (unbounded, minimises a scalar closure)
// ---------------------------------------------------------------------------

/// Nelder-Mead simplex optimiser. Minimises `f(x)`.
fn nelder_mead<F>(f: F, x0: &[f64], max_iter: usize, tol: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if v[i].abs() < 1e-10 {
            0.1
        } else {
            0.05 * v[i].abs() + 0.05
        };
        simplex.push(v);
    }
    let mut vals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        // Sort by function value
        let mut idx: Vec<usize> = (0..n + 1).collect();
        idx.sort_by(|&a, &b| {
            vals[a]
                .partial_cmp(&vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let simplex_sorted: Vec<Vec<f64>> = idx.iter().map(|&i| simplex[i].clone()).collect();
        let vals_sorted: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
        simplex = simplex_sorted;
        vals = vals_sorted;

        // Convergence check
        let span = vals.last().copied().unwrap_or(f64::INFINITY) - vals[0];
        if span < tol {
            break;
        }

        // Centroid of all but worst
        let mut centroid = vec![0.0f64; n];
        for v in &simplex[..n] {
            for (j, &x) in v.iter().enumerate() {
                centroid[j] += x / n as f64;
            }
        }

        // Reflect
        let reflect: Vec<f64> = (0..n)
            .map(|j| centroid[j] + 1.0 * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&reflect);
        if fr < vals[0] {
            // Expand
            let expand: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (reflect[j] - centroid[j]))
                .collect();
            let fe = f(&expand);
            if fe < fr {
                simplex[n] = expand;
                vals[n] = fe;
            } else {
                simplex[n] = reflect;
                vals[n] = fr;
            }
        } else if fr < vals[n - 1] {
            simplex[n] = reflect;
            vals[n] = fr;
        } else {
            // Contract
            let contract: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 0.5 * (simplex[n][j] - centroid[j]))
                .collect();
            let fc = f(&contract);
            if fc < vals[n] {
                simplex[n] = contract;
                vals[n] = fc;
            } else {
                // Shrink
                for i in 1..=n {
                    let shrunk: Vec<f64> = (0..n)
                        .map(|j| simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]))
                        .collect();
                    vals[i] = f(&shrunk);
                    simplex[i] = shrunk;
                }
            }
        }
    }
    simplex[0].clone()
}

// Convert log-parameterised variances to natural (ensures positivity).
#[inline]
fn from_log_params(p: &[f64]) -> Vec<f64> {
    p.iter().map(|x| x.exp()).collect()
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Output from Kalman filter / smoother.
#[derive(Clone, Debug)]
pub struct KalmanState {
    /// Filtered (or smoothed) state means, one per time point.
    pub state_means: Vec<Vec<f64>>,
    /// Filtered (or smoothed) state covariances, one per time point.
    pub state_covs: Vec<Vec<Vec<f64>>>,
    /// Prediction innovations (one-step-ahead residuals).
    pub innovations: Vec<f64>,
    /// Innovation variances.
    pub innov_vars: Vec<f64>,
    /// Total log-likelihood of the observations.
    pub log_likelihood: f64,
}

impl KalmanState {
    fn from_filter(kf: &KfOutput) -> Self {
        let state_covs = kf.filtered_cov.iter().map(|m| mat_to_nested(m)).collect();
        KalmanState {
            state_means: kf.filtered_mean.clone(),
            state_covs,
            innovations: kf.innovations.clone(),
            innov_vars: kf.innov_var.clone(),
            log_likelihood: kf.log_lik,
        }
    }
    fn from_smoother(
        sm: (Vec<Vec<f64>>, Vec<Mat>),
        log_lik: f64,
        innov: &[f64],
        iv: &[f64],
    ) -> Self {
        let state_covs = sm.1.iter().map(|m| mat_to_nested(m)).collect();
        KalmanState {
            state_means: sm.0,
            state_covs,
            innovations: innov.to_vec(),
            innov_vars: iv.to_vec(),
            log_likelihood: log_lik,
        }
    }
}

fn mat_to_nested(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m.get(r, c)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Local Level Model  y_t = μ_t + ε_t,  μ_{t+1} = μ_t + η_t
// ---------------------------------------------------------------------------

/// Harvey's Local Level Model (random walk plus noise).
///
/// State equation:  `μ_{t+1} = μ_t + η_t`,  η_t ~ N(0, σ²_η)
/// Observation:     `y_t = μ_t + ε_t`,       ε_t ~ N(0, σ²_ε)
#[derive(Clone, Debug)]
pub struct LocalLevel {
    /// Process (level) noise variance σ²_η.
    pub level_var: f64,
    /// Observation noise variance σ²_ε.
    pub obs_var: f64,
}

impl LocalLevel {
    /// Fit a Local Level model by MLE using prediction-error decomposition.
    pub fn fit(data: &[f64]) -> Result<Self> {
        if data.len() < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "LocalLevel::fit requires at least 3 observations".into(),
                required: 3,
                actual: data.len(),
            });
        }
        let neg_ll = |p: &[f64]| -> f64 {
            let vars = from_log_params(p);
            let (lv, ov) = (vars[0], vars[1]);
            let model = LocalLevel {
                level_var: lv,
                obs_var: ov,
            };
            let kf = model.build_kf(data);
            -kf.log_lik
        };
        // Initialise from empirical variance
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let x0 = vec![(var * 0.3 + 1e-6).ln(), (var * 0.7 + 1e-6).ln()];
        let best = nelder_mead(neg_ll, &x0, 2000, 1e-8);
        let vars = from_log_params(&best);
        Ok(LocalLevel {
            level_var: vars[0].max(1e-10),
            obs_var: vars[1].max(1e-10),
        })
    }

    /// Run the Kalman filter and return filtered states.
    pub fn filter(&self, data: &[f64]) -> Result<KalmanState> {
        if data.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "filter requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let kf = self.build_kf(data);
        Ok(KalmanState::from_filter(&kf))
    }

    /// Run the RTS smoother and return smoothed states.
    pub fn smoother(&self, data: &[f64]) -> Result<KalmanState> {
        if data.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "smoother requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let kf = self.build_kf(data);
        let f = Self::make_f();
        let sm = rts_smoother_internal(&kf, &f);
        Ok(KalmanState::from_smoother(
            sm,
            kf.log_lik,
            &kf.innovations,
            &kf.innov_var,
        ))
    }

    fn make_f() -> Mat {
        Mat::eye(1)
    }

    fn build_kf(&self, data: &[f64]) -> KfOutput {
        let f = Self::make_f();
        let h = Mat::eye(1);
        let mut q = Mat::zeros(1, 1);
        q.set(0, 0, self.level_var);
        let r = self.obs_var;
        let m0 = vec![data[0]];
        let mut p0 = Mat::zeros(1, 1);
        p0.set(0, 0, 1e6);
        kalman_filter_internal(data, &f, &h, &q, r, &m0, &p0)
    }
}

// ---------------------------------------------------------------------------
// 2. Local Linear Trend  (level + slope)
// ---------------------------------------------------------------------------

/// Harvey's Local Linear Trend model.
///
/// Level: `μ_{t+1} = μ_t + β_t + η_t`,  η ~ N(0, σ²_η)
/// Slope: `β_{t+1} = β_t + ζ_t`,        ζ ~ N(0, σ²_ζ)
/// Obs:   `y_t = μ_t + ε_t`,             ε ~ N(0, σ²_ε)
#[derive(Clone, Debug)]
pub struct LocalLinearTrend {
    /// Level noise variance σ²_η.
    pub level_var: f64,
    /// Slope noise variance σ²_ζ.
    pub slope_var: f64,
    /// Observation noise variance σ²_ε.
    pub obs_var: f64,
}

impl LocalLinearTrend {
    /// Fit by MLE.
    pub fn fit(data: &[f64]) -> Result<Self> {
        if data.len() < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "LocalLinearTrend::fit requires at least 4 observations".into(),
                required: 4,
                actual: data.len(),
            });
        }
        let neg_ll = |p: &[f64]| -> f64 {
            let vars = from_log_params(p);
            let model = LocalLinearTrend {
                level_var: vars[0],
                slope_var: vars[1],
                obs_var: vars[2],
            };
            let kf = model.build_kf(data);
            -kf.log_lik
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let x0 = vec![
            (var * 0.2 + 1e-6).ln(),
            (var * 0.1 + 1e-6).ln(),
            (var * 0.7 + 1e-6).ln(),
        ];
        let best = nelder_mead(neg_ll, &x0, 3000, 1e-8);
        let vars = from_log_params(&best);
        Ok(LocalLinearTrend {
            level_var: vars[0].max(1e-10),
            slope_var: vars[1].max(1e-10),
            obs_var: vars[2].max(1e-10),
        })
    }

    /// Run Kalman filter.
    pub fn filter(&self, data: &[f64]) -> Result<KalmanState> {
        if data.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "filter requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let kf = self.build_kf(data);
        Ok(KalmanState::from_filter(&kf))
    }

    /// Run RTS smoother.
    pub fn smoother(&self, data: &[f64]) -> Result<KalmanState> {
        if data.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "smoother requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let kf = self.build_kf(data);
        let (f, _, _, _) = Self::build_matrices(self.level_var, self.slope_var);
        let sm = rts_smoother_internal(&kf, &f);
        Ok(KalmanState::from_smoother(
            sm,
            kf.log_lik,
            &kf.innovations,
            &kf.innov_var,
        ))
    }

    fn build_matrices(level_var: f64, slope_var: f64) -> (Mat, Mat, Mat, Mat) {
        // State: [μ, β]'
        // F = [[1, 1], [0, 1]]
        let mut f = Mat::eye(2);
        f.set(0, 1, 1.0);
        // H = [1, 0]
        let mut h = Mat::zeros(1, 2);
        h.set(0, 0, 1.0);
        // Q = diag(level_var, slope_var)
        let mut q = Mat::zeros(2, 2);
        q.set(0, 0, level_var);
        q.set(1, 1, slope_var);
        // P0 = large diagonal
        let mut p0 = Mat::zeros(2, 2);
        p0.set(0, 0, 1e6);
        p0.set(1, 1, 1e6);
        (f, h, q, p0)
    }

    fn build_kf(&self, data: &[f64]) -> KfOutput {
        let (f, h, q, p0) = Self::build_matrices(self.level_var, self.slope_var);
        let r = self.obs_var;
        let m0 = vec![data[0], 0.0];
        kalman_filter_internal(data, &f, &h, &q, r, &m0, &p0)
    }
}

// ---------------------------------------------------------------------------
// 3. Basic Structural Model (BSM)
// ---------------------------------------------------------------------------

/// Structural decomposition output.
#[derive(Clone, Debug)]
pub struct StructuralComponents {
    /// Smoothed trend component (μ_t).
    pub trend: Vec<f64>,
    /// Smoothed seasonal component (γ_t).
    pub seasonal: Vec<f64>,
    /// Irregular (residual) component (ε_t = y_t − trend − seasonal).
    pub irregular: Vec<f64>,
}

/// Forecast result.
#[derive(Clone, Debug)]
pub struct ForecastResult {
    /// Point forecasts.
    pub mean: Vec<f64>,
    /// Lower bound (95 % prediction interval).
    pub lower: Vec<f64>,
    /// Upper bound (95 % prediction interval).
    pub upper: Vec<f64>,
}

/// Harvey's Basic Structural Model with trigonometric seasonality.
///
/// ```text
/// y_t = μ_t + γ_t + ε_t
/// μ_{t+1} = μ_t + β_t + η_t     (level)
/// β_{t+1} = β_t + ζ_t            (slope)
/// γ_t = Σ_{j=1}^{⌊s/2⌋} γ_{j,t}  (trigonometric seasonal)
/// ```
///
/// Each seasonal harmonic j evolves as:
/// ```text
/// [γ_{j,t+1}]   [cos λ_j   sin λ_j] [γ_{j,t}]   [ω_{j,t}  ]
/// [γ*_{j,t+1}] = [-sin λ_j  cos λ_j] [γ*_{j,t}] + [ω*_{j,t} ]
/// ```
/// where λ_j = 2πj/s.
#[derive(Clone, Debug)]
pub struct BasicStructural {
    /// Level noise variance σ²_η.
    pub level_var: f64,
    /// Slope noise variance σ²_ζ.
    pub slope_var: f64,
    /// Seasonal disturbance variance σ²_ω (shared across harmonics).
    pub seasonal_var: f64,
    /// Observation noise variance σ²_ε.
    pub obs_var: f64,
    /// Seasonal period s.
    pub period: usize,
}

impl BasicStructural {
    /// Fit a BSM to `data` with given `period` by MLE.
    ///
    /// # Arguments
    /// * `data`   – univariate time series
    /// * `period` – seasonal period (e.g. 12 for monthly, 4 for quarterly)
    pub fn fit(data: &[f64], period: usize) -> Result<Self> {
        if period < 2 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "period".into(),
                message: "period must be ≥ 2".into(),
            });
        }
        if data.len() < 2 * period {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("BSM fit requires at least {} observations", 2 * period),
                required: 2 * period,
                actual: data.len(),
            });
        }
        let neg_ll = |p: &[f64]| -> f64 {
            let vars = from_log_params(p);
            let model = BasicStructural {
                level_var: vars[0],
                slope_var: vars[1],
                seasonal_var: vars[2],
                obs_var: vars[3],
                period,
            };
            let (f, h, q, p0, m0) = model.build_matrices(data);
            let kf = kalman_filter_internal(data, &f, &h, &q, vars[3], &m0, &p0);
            -kf.log_lik
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let x0 = vec![
            (var * 0.15 + 1e-6).ln(),
            (var * 0.05 + 1e-6).ln(),
            (var * 0.10 + 1e-6).ln(),
            (var * 0.70 + 1e-6).ln(),
        ];
        let best = nelder_mead(neg_ll, &x0, 4000, 1e-8);
        let vars = from_log_params(&best);
        Ok(BasicStructural {
            level_var: vars[0].max(1e-10),
            slope_var: vars[1].max(1e-10),
            seasonal_var: vars[2].max(1e-10),
            obs_var: vars[3].max(1e-10),
            period,
        })
    }

    /// Number of seasonal harmonics: ⌊period/2⌋.
    pub fn n_harmonics(&self) -> usize {
        self.period / 2
    }

    /// State dimension: 2 (trend/slope) + 2 * n_harmonics.
    pub fn n_states(&self) -> usize {
        2 + 2 * self.n_harmonics()
    }

    /// Build the state-space matrices for this model.
    ///
    /// Returns (F, H, Q, P0, m0).
    fn build_matrices(&self, data: &[f64]) -> (Mat, Mat, Mat, Mat, Vec<f64>) {
        let ns = self.n_states();
        let nh = self.n_harmonics();

        // --- Transition matrix F ---
        // [1  1  |  0  0  ...  ]    trend-level row
        // [0  1  |  0  0  ...  ]    trend-slope row
        // [       | Rj blocks  ]    seasonal harmonic blocks
        let mut f = Mat::zeros(ns, ns);
        // Trend block
        f.set(0, 0, 1.0);
        f.set(0, 1, 1.0);
        f.set(1, 1, 1.0);
        // Seasonal harmonic blocks (each 2×2 rotation)
        for j in 1..=nh {
            let lam = 2.0 * PI * j as f64 / self.period as f64;
            let (c, s) = (lam.cos(), lam.sin());
            let base = 2 + (j - 1) * 2;
            f.set(base, base, c);
            f.set(base, base + 1, s);
            f.set(base + 1, base, -s);
            f.set(base + 1, base + 1, c);
        }

        // --- Observation matrix H: y_t = μ_t + Σ γ_{j,t} ---
        let mut h = Mat::zeros(1, ns);
        h.set(0, 0, 1.0);
        for j in 0..nh {
            h.set(0, 2 + j * 2, 1.0);
        }

        // --- Process noise Q ---
        let mut q = Mat::zeros(ns, ns);
        q.set(0, 0, self.level_var);
        q.set(1, 1, self.slope_var);
        for j in 0..nh {
            let base = 2 + j * 2;
            q.set(base, base, self.seasonal_var);
            q.set(base + 1, base + 1, self.seasonal_var);
        }

        // --- Initial covariance P0 (diffuse) ---
        let mut p0 = Mat::zeros(ns, ns);
        for i in 0..ns {
            p0.set(i, i, 1e6);
        }

        // --- Initial state mean m0 ---
        let first = if data.is_empty() { 0.0 } else { data[0] };
        let mut m0 = vec![0.0f64; ns];
        m0[0] = first;

        (f, h, q, p0, m0)
    }

    fn build_kf(&self, data: &[f64]) -> KfOutput {
        let (f, h, q, p0, m0) = self.build_matrices(data);
        kalman_filter_internal(data, &f, &h, &q, self.obs_var, &m0, &p0)
    }

    /// Kalman filter output.
    pub fn filter(&self, data: &[f64]) -> Result<KalmanState> {
        if data.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "filter requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let kf = self.build_kf(data);
        Ok(KalmanState::from_filter(&kf))
    }

    /// RTS smoother output.
    pub fn smoother(&self, data: &[f64]) -> Result<KalmanState> {
        if data.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "smoother requires at least one observation".into(),
                required: 1,
                actual: 0,
            });
        }
        let kf = self.build_kf(data);
        let (f, _, _, _, _) = self.build_matrices(data);
        let sm = rts_smoother_internal(&kf, &f);
        Ok(KalmanState::from_smoother(
            sm,
            kf.log_lik,
            &kf.innovations,
            &kf.innov_var,
        ))
    }
}

// ---------------------------------------------------------------------------
// 4. Structural decomposition
// ---------------------------------------------------------------------------

/// Decompose a time series into trend, seasonal, and irregular components
/// using the smoothed states from a fitted `BasicStructural` model.
///
/// # Arguments
/// * `model` – fitted BSM
/// * `data`  – original observations
///
/// Returns [`StructuralComponents`] containing smoothed trend, seasonal, and irregular.
pub fn decompose(model: &BasicStructural, data: &[f64]) -> Result<StructuralComponents> {
    if data.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "decompose requires at least one observation".into(),
            required: 1,
            actual: 0,
        });
    }
    let smoothed = model.smoother(data)?;
    let nh = model.n_harmonics();
    let n = data.len();

    let mut trend = Vec::with_capacity(n);
    let mut seasonal = Vec::with_capacity(n);

    for t in 0..n {
        let st = &smoothed.state_means[t];
        trend.push(st[0]); // μ_t is state index 0
                           // Seasonal = sum of first element of each harmonic pair
        let s: f64 = (0..nh).map(|j| st[2 + j * 2]).sum();
        seasonal.push(s);
    }

    let irregular: Vec<f64> = (0..n).map(|t| data[t] - trend[t] - seasonal[t]).collect();

    Ok(StructuralComponents {
        trend,
        seasonal,
        irregular,
    })
}

// ---------------------------------------------------------------------------
// 5. Forecasting
// ---------------------------------------------------------------------------

/// Forecast `h` steps ahead from a fitted `BasicStructural` model.
///
/// Uses the Kalman-filtered last state as the initial distribution and
/// propagates uncertainty forward through the state equations.
///
/// # Arguments
/// * `model` – fitted BSM
/// * `data`  – observed history (used to run Kalman filter)
/// * `h`     – forecast horizon (number of steps)
pub fn forecast(model: &BasicStructural, data: &[f64], h: usize) -> Result<ForecastResult> {
    if data.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "forecast requires at least one observation".into(),
            required: 1,
            actual: 0,
        });
    }
    if h == 0 {
        return Ok(ForecastResult {
            mean: vec![],
            lower: vec![],
            upper: vec![],
        });
    }

    let (f, h_mat, q, _, _) = model.build_matrices(data);
    let f_t = f.t();
    let kf = model.build_kf(data);

    let t_last = kf.filtered_mean.len() - 1;
    let mut m = kf.filtered_mean[t_last].clone();
    let mut p = kf.filtered_cov[t_last].clone();

    let mut means = Vec::with_capacity(h);
    let mut lowers = Vec::with_capacity(h);
    let mut uppers = Vec::with_capacity(h);
    let z95 = 1.959_964f64; // 95 % CI

    for _ in 0..h {
        // Predict one step
        let m_mat = Mat::from_col(&m);
        let mp_mat = f.mul(&m_mat);
        m = mp_mat.col_vec(0);
        p = f.mul(&p).mul(&f_t).add(&q);

        // Forecast mean: y_pred = H m
        let n = f.rows;
        let y_pred: f64 = (0..n).map(|j| h_mat.get(0, j) * m[j]).sum();

        // Forecast variance: S = H P H' + R
        let hp = h_mat.mul(&p);
        let s: f64 = (0..n).map(|j| hp.get(0, j) * h_mat.get(0, j)).sum::<f64>() + model.obs_var;
        let std_dev = s.sqrt();

        means.push(y_pred);
        lowers.push(y_pred - z95 * std_dev);
        uppers.push(y_pred + z95 * std_dev);
    }

    Ok(ForecastResult {
        mean: means,
        lower: lowers,
        upper: uppers,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(n: usize, slope: f64, noise_amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| slope * i as f64 + noise_amp * (i as f64 * 0.7).sin())
            .collect()
    }

    fn seasonal_series(n: usize, period: usize, amp: f64, noise: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let s = amp * (2.0 * PI * i as f64 / period as f64).sin();
                let trend = 0.05 * i as f64;
                s + trend + noise * (i as f64 * 1.3).cos()
            })
            .collect()
    }

    // --- LocalLevel tests ---

    #[test]
    fn test_local_level_fit_smoke() {
        let data: Vec<f64> = (0..30)
            .map(|i| 5.0 + (i as f64 * 0.2).sin() * 0.5)
            .collect();
        let model = LocalLevel::fit(&data).expect("fit ok");
        assert!(model.level_var > 0.0, "level_var must be positive");
        assert!(model.obs_var > 0.0, "obs_var must be positive");
    }

    #[test]
    fn test_local_level_filter_length() {
        let data: Vec<f64> = (0..20).map(|i| 3.0 + (i as f64 * 0.3).cos()).collect();
        let model = LocalLevel {
            level_var: 0.1,
            obs_var: 0.5,
        };
        let ks = model.filter(&data).expect("filter ok");
        assert_eq!(ks.state_means.len(), 20);
        assert_eq!(ks.innovations.len(), 20);
    }

    #[test]
    fn test_local_level_smoother_length() {
        let data: Vec<f64> = (0..15).map(|i| (i as f64 * 0.4).sin()).collect();
        let model = LocalLevel {
            level_var: 0.2,
            obs_var: 0.3,
        };
        let ks = model.smoother(&data).expect("smoother ok");
        assert_eq!(ks.state_means.len(), 15);
    }

    #[test]
    fn test_local_level_filter_empty_err() {
        let model = LocalLevel {
            level_var: 0.1,
            obs_var: 0.1,
        };
        assert!(model.filter(&[]).is_err());
    }

    #[test]
    fn test_local_level_constant_series() {
        // For a constant series, the filtered level should converge to the constant.
        let data = vec![5.0f64; 50];
        let model = LocalLevel {
            level_var: 0.01,
            obs_var: 0.1,
        };
        let ks = model.filter(&data).expect("filter ok");
        let last_level = ks
            .state_means
            .last()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0);
        assert!((last_level - 5.0).abs() < 0.5, "last level={last_level}");
    }

    #[test]
    fn test_local_level_log_likelihood_finite() {
        let data: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin() * 2.0).collect();
        let model = LocalLevel {
            level_var: 0.1,
            obs_var: 0.5,
        };
        let ks = model.filter(&data).expect("filter ok");
        assert!(ks.log_likelihood.is_finite());
        assert!(ks.log_likelihood < 0.0, "ll should be negative");
    }

    // --- LocalLinearTrend tests ---

    #[test]
    fn test_llt_fit_smoke() {
        let data = linspace(40, 0.5, 0.3);
        let model = LocalLinearTrend::fit(&data).expect("fit ok");
        assert!(model.level_var > 0.0);
        assert!(model.slope_var > 0.0);
        assert!(model.obs_var > 0.0);
    }

    #[test]
    fn test_llt_filter_tracks_trend() {
        let data: Vec<f64> = (0..25).map(|i| i as f64 + 0.1 * (i as f64).cos()).collect();
        let model = LocalLinearTrend {
            level_var: 0.01,
            slope_var: 0.001,
            obs_var: 0.5,
        };
        let ks = model.filter(&data).expect("filter ok");
        let last_level = ks
            .state_means
            .last()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0);
        assert!((last_level - 24.0).abs() < 5.0, "last_level={last_level}");
    }

    #[test]
    fn test_llt_smoother_length() {
        let data = linspace(30, 1.0, 0.2);
        let model = LocalLinearTrend {
            level_var: 0.05,
            slope_var: 0.005,
            obs_var: 0.3,
        };
        let ks = model.smoother(&data).expect("smoother ok");
        assert_eq!(ks.state_means.len(), 30);
        assert_eq!(ks.state_means[0].len(), 2, "state dim should be 2");
    }

    // --- BasicStructural tests ---

    #[test]
    fn test_bsm_fit_smoke() {
        let data = seasonal_series(48, 12, 2.0, 0.3);
        let model = BasicStructural::fit(&data, 12).expect("BSM fit ok");
        assert!(model.level_var > 0.0);
        assert!(model.slope_var > 0.0);
        assert!(model.seasonal_var > 0.0);
        assert!(model.obs_var > 0.0);
        assert_eq!(model.period, 12);
    }

    #[test]
    fn test_bsm_invalid_period() {
        let data = vec![1.0f64; 20];
        assert!(BasicStructural::fit(&data, 1).is_err());
    }

    #[test]
    fn test_bsm_insufficient_data() {
        let data = vec![1.0f64; 5];
        assert!(BasicStructural::fit(&data, 12).is_err());
    }

    #[test]
    fn test_bsm_n_states_quarterly() {
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.05,
            obs_var: 0.3,
            period: 4,
        };
        // period=4 → harmonics=2 → states = 2 + 2*2 = 6
        assert_eq!(model.n_harmonics(), 2);
        assert_eq!(model.n_states(), 6);
    }

    #[test]
    fn test_bsm_filter_length() {
        let data = seasonal_series(24, 4, 1.5, 0.2);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.4,
            period: 4,
        };
        let ks = model.filter(&data).expect("filter ok");
        assert_eq!(ks.state_means.len(), 24);
        assert_eq!(ks.state_means[0].len(), model.n_states());
    }

    #[test]
    fn test_bsm_smoother_length() {
        let data = seasonal_series(20, 4, 1.0, 0.1);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let ks = model.smoother(&data).expect("smoother ok");
        assert_eq!(ks.state_means.len(), 20);
    }

    // --- Decomposition tests ---

    #[test]
    fn test_decompose_lengths() {
        let data = seasonal_series(24, 4, 2.0, 0.2);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let comps = decompose(&model, &data).expect("decompose ok");
        assert_eq!(comps.trend.len(), 24);
        assert_eq!(comps.seasonal.len(), 24);
        assert_eq!(comps.irregular.len(), 24);
    }

    #[test]
    fn test_decompose_reconstruction() {
        // trend + seasonal + irregular should approximate original series.
        let data = seasonal_series(24, 4, 2.0, 0.2);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let comps = decompose(&model, &data).expect("decompose ok");
        for t in 0..24 {
            let recon = comps.trend[t] + comps.seasonal[t] + comps.irregular[t];
            assert!(
                (recon - data[t]).abs() < 1e-9,
                "mismatch at t={t}: recon={recon} data={}",
                data[t]
            );
        }
    }

    #[test]
    fn test_decompose_values_finite() {
        let data = seasonal_series(24, 12, 3.0, 0.5);
        let model = BasicStructural {
            level_var: 0.2,
            slope_var: 0.02,
            seasonal_var: 0.2,
            obs_var: 0.5,
            period: 12,
        };
        let comps = decompose(&model, &data).expect("decompose ok");
        for t in 0..data.len() {
            assert!(comps.trend[t].is_finite(), "trend not finite at {t}");
            assert!(comps.seasonal[t].is_finite(), "seasonal not finite at {t}");
            assert!(
                comps.irregular[t].is_finite(),
                "irregular not finite at {t}"
            );
        }
    }

    // --- Forecast tests ---

    #[test]
    fn test_forecast_length() {
        let data = seasonal_series(24, 4, 1.5, 0.2);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let fc = forecast(&model, &data, 8).expect("forecast ok");
        assert_eq!(fc.mean.len(), 8);
        assert_eq!(fc.lower.len(), 8);
        assert_eq!(fc.upper.len(), 8);
    }

    #[test]
    fn test_forecast_intervals_valid() {
        let data = seasonal_series(24, 4, 1.5, 0.2);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let fc = forecast(&model, &data, 6).expect("forecast ok");
        for i in 0..6 {
            assert!(fc.lower[i] <= fc.mean[i], "lower > mean at h={i}");
            assert!(fc.upper[i] >= fc.mean[i], "upper < mean at h={i}");
        }
    }

    #[test]
    fn test_forecast_intervals_widen() {
        // Prediction intervals should generally widen over the horizon.
        let data = seasonal_series(24, 4, 1.5, 0.2);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let fc = forecast(&model, &data, 8).expect("forecast ok");
        let w0 = fc.upper[0] - fc.lower[0];
        let w7 = fc.upper[7] - fc.lower[7];
        assert!(w7 >= w0, "interval did not widen: w0={w0} w7={w7}");
    }

    #[test]
    fn test_forecast_zero_horizon() {
        let data = seasonal_series(12, 4, 1.0, 0.1);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.05,
            obs_var: 0.2,
            period: 4,
        };
        let fc = forecast(&model, &data, 0).expect("zero-h forecast ok");
        assert!(fc.mean.is_empty());
    }

    #[test]
    fn test_forecast_finite_values() {
        let data = seasonal_series(24, 4, 2.0, 0.3);
        let model = BasicStructural {
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            obs_var: 0.3,
            period: 4,
        };
        let fc = forecast(&model, &data, 12).expect("forecast ok");
        for i in 0..12 {
            assert!(fc.mean[i].is_finite(), "mean[{i}] is not finite");
            assert!(fc.lower[i].is_finite(), "lower[{i}] is not finite");
            assert!(fc.upper[i].is_finite(), "upper[{i}] is not finite");
        }
    }
}
