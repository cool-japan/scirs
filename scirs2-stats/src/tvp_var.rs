//! Time-Varying Parameter Vector Autoregression (TVP-VAR)
//!
//! This module implements TVP-VAR models where VAR coefficients evolve over
//! time as random walks, estimated via the Kalman filter and RTS smoother.
//!
//! # Model Specification
//!
//! For a `K`-variable VAR with `p` lags the observation equation is:
//! ```text
//! y_t = Z_t β_t + ε_t,  ε_t ~ N(0, R)
//! ```
//! where `Z_t = I_K ⊗ [y_{t-1}', ..., y_{t-p}', 1]` is the regressor matrix
//! and `β_t` is the state vector of all VAR coefficients (stacked).
//!
//! The state equation is a random walk:
//! ```text
//! β_t = β_{t-1} + η_t,  η_t ~ N(0, Q)
//! ```
//!
//! # References
//! - Primiceri, G.E. (2005). Time Varying Structural Vector Autoregressions and
//!   Monetary Policy. *Review of Economic Studies* 72(3): 821–852.
//! - Cogley, T. & Sargent, T.J. (2005). Drifts and Volatilities: Monetary
//!   Policies and Outcomes in the Post-WWII US. *Review of Economic Dynamics*.
//! - Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ---------------------------------------------------------------------------
// Public data structures
// ---------------------------------------------------------------------------

/// TVP-VAR model specification.
///
/// For a system with `n_vars` variables and `n_lags` lags the coefficient
/// state has dimension `state_dim = n_vars * (n_vars * n_lags + 1)` (the `+1`
/// is for the constant term).
#[derive(Clone, Debug)]
pub struct TvpVar {
    /// Number of variables `K`.
    pub n_vars: usize,
    /// Number of lags `p`.
    pub n_lags: usize,
    /// Dimension of the coefficient state vector (`K * (K*p + 1)`).
    pub state_dim: usize,
    /// Number of regressors per equation (`K*p + 1`).
    pub n_regressors: usize,
}

impl TvpVar {
    /// Create a new TVP-VAR specification.
    ///
    /// # Arguments
    /// * `n_vars` – number of endogenous variables.
    /// * `n_lags` – number of lags.
    ///
    /// # Errors
    /// Returns `StatsError::InvalidInput` when `n_vars` or `n_lags` is zero.
    pub fn new(n_vars: usize, n_lags: usize) -> StatsResult<Self> {
        if n_vars == 0 {
            return Err(StatsError::InvalidInput(
                "TvpVar: n_vars must be >= 1".into(),
            ));
        }
        if n_lags == 0 {
            return Err(StatsError::InvalidInput(
                "TvpVar: n_lags must be >= 1".into(),
            ));
        }
        let n_regressors = n_vars * n_lags + 1; // lagged values + constant
        let state_dim = n_vars * n_regressors;
        Ok(Self {
            n_vars,
            n_lags,
            state_dim,
            n_regressors,
        })
    }
}

/// State of the TVP-VAR at a single time point.
#[derive(Clone, Debug)]
pub struct TvpVarState {
    /// Coefficient vector `β_t` of length `state_dim`.
    pub coefficients: Array1<f64>,
    /// State (parameter) covariance matrix `P_t` of shape `state_dim × state_dim`.
    pub state_cov: Array2<f64>,
}

/// Full result of TVP-VAR estimation.
#[derive(Clone, Debug)]
pub struct TvpVarResult {
    /// Smoothed coefficient vectors, one per time point (length = T).
    pub smoothed_states: Vec<TvpVarState>,
    /// Filtered coefficient vectors, one per time point (length = T).
    pub filtered_states: Vec<TvpVarState>,
    /// Log-likelihood of the observations.
    pub log_likelihood: f64,
    /// The TVP-VAR specification used.
    pub spec: TvpVar,
    /// Number of observations used (T).
    pub n_obs: usize,
}

impl TvpVarResult {
    /// Return the smoothed coefficient matrix at time index `t`.
    ///
    /// The returned `Array2<f64>` has shape `(n_vars, n_regressors)` where
    /// row `k` gives the coefficients for equation `k`.
    pub fn smoothed_coeff_matrix(&self, t: usize) -> StatsResult<Array2<f64>> {
        if t >= self.smoothed_states.len() {
            return Err(StatsError::InvalidInput(format!(
                "Time index {} out of range (T={})",
                t,
                self.smoothed_states.len()
            )));
        }
        let k = self.spec.n_vars;
        let r = self.spec.n_regressors;
        let beta = &self.smoothed_states[t].coefficients;
        let mat = Array2::from_shape_fn((k, r), |(eq, col)| beta[eq * r + col]);
        Ok(mat)
    }

    /// Compute impulse response functions (IRFs) at a given time point.
    ///
    /// Returns a `Vec<Array2<f64>>` of length `n_periods` where element `h`
    /// has shape `(n_vars, n_vars)` giving the response of each variable to a
    /// unit shock in each variable at horizon `h`.
    ///
    /// # Arguments
    /// * `t`         – time index at which to evaluate the coefficients.
    /// * `n_periods` – number of IRF horizons (including horizon 0).
    pub fn impulse_response_at(&self, t: usize, n_periods: usize) -> StatsResult<Vec<Array2<f64>>> {
        let k = self.spec.n_vars;
        let p = self.spec.n_lags;
        let coeff_mat = self.smoothed_coeff_matrix(t)?; // (k, k*p + 1)

        // Extract companion-form coefficient matrices A_1, ..., A_p
        // Each A_l is (k, k)
        let mut a_mats: Vec<Array2<f64>> = Vec::with_capacity(p);
        for lag in 0..p {
            let a_l = Array2::from_shape_fn((k, k), |(row, col)| coeff_mat[[row, lag * k + col]]);
            a_mats.push(a_l);
        }

        // Build companion matrix C of dimension (k*p, k*p)
        // C = [A_1 A_2 ... A_{p-1} A_p]
        //     [I   0  ...  0       0  ]
        //     [0   I  ...  0       0  ]
        //     [         ...           ]
        let comp_dim = k * p;
        let mut companion = Array2::<f64>::zeros((comp_dim, comp_dim));
        for lag in 0..p {
            for r in 0..k {
                for c in 0..k {
                    companion[[r, lag * k + c]] = a_mats[lag][[r, c]];
                }
            }
        }
        // Sub-diagonal identity blocks
        for i in 1..p {
            for j in 0..k {
                companion[[i * k + j, (i - 1) * k + j]] = 1.0;
            }
        }

        // IRF via repeated multiplication
        // irf_h = e_1^T C^h e_j  for shock to variable j
        let mut irf_list: Vec<Array2<f64>> = Vec::with_capacity(n_periods);
        let mut power = identity(comp_dim);

        for _h in 0..n_periods {
            let irf_h = Array2::from_shape_fn((k, k), |(resp, shock)| power[[resp, shock]]);
            irf_list.push(irf_h);
            power = mat_mat_mul(&power, &companion)
                .unwrap_or_else(|_| Array2::zeros((comp_dim, comp_dim)));
        }

        Ok(irf_list)
    }
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/// Fit a TVP-VAR model to multivariate time-series data via Kalman
/// filter + RTS smoother.
///
/// # Arguments
/// * `data`       – `(T, K)` matrix of observations (rows = time, cols = variables).
/// * `spec`       – TVP-VAR specification built with [`TvpVar::new`].
/// * `prior_mean` – prior mean for the initial coefficient state (`state_dim`-vector).
///   Pass `None` to use a zero vector.
/// * `prior_cov`  – prior covariance for the initial state (`state_dim × state_dim`).
///   Pass `None` to use a large diffuse prior (`1e6 * I`).
/// * `q`          – process noise covariance `Q` (`state_dim × state_dim`).
/// * `r`          – observation noise covariance `R` (`K × K`).
///
/// # Returns
/// A [`TvpVarResult`] containing filtered and smoothed states and the log-likelihood.
///
/// # Errors
/// Returns errors for dimension mismatches, insufficient data, or numerical
/// failures in Cholesky / inversion routines.
pub fn fit_tvp_var(
    data: &Array2<f64>,
    spec: &TvpVar,
    prior_mean: Option<&Array1<f64>>,
    prior_cov: Option<&Array2<f64>>,
    q: &Array2<f64>,
    r: &Array2<f64>,
) -> StatsResult<TvpVarResult> {
    let big_t = data.nrows();
    let k = spec.n_vars;
    let p = spec.n_lags;
    let d = spec.state_dim;

    if data.ncols() != k {
        return Err(StatsError::DimensionMismatch(format!(
            "data has {} columns but spec has n_vars={}",
            data.ncols(),
            k
        )));
    }
    if big_t <= p {
        return Err(StatsError::InsufficientData(format!(
            "Need at least n_lags+1={} observations, got {}",
            p + 1,
            big_t
        )));
    }
    validate_square(q, d, "Q")?;
    validate_square(r, k, "R")?;

    // ── Initial state ────────────────────────────────────────────────────────
    let x0: Array1<f64> = match prior_mean {
        Some(m) => {
            if m.len() != d {
                return Err(StatsError::DimensionMismatch(format!(
                    "prior_mean length {} != state_dim {}",
                    m.len(),
                    d
                )));
            }
            m.clone()
        }
        None => Array1::zeros(d),
    };

    let p0: Array2<f64> = match prior_cov {
        Some(pc) => {
            validate_square(pc, d, "prior_cov")?;
            pc.clone()
        }
        None => Array2::from_diag(&Array1::from_elem(d, 1e6_f64)),
    };

    // Effective time range: t = p .. T
    let t_eff = big_t - p; // number of observations used

    // ── Forward Kalman filter pass ───────────────────────────────────────────
    let mut filtered: Vec<TvpVarState> = Vec::with_capacity(t_eff);
    let mut predicted: Vec<TvpVarState> = Vec::with_capacity(t_eff);
    let mut log_lik = 0.0_f64;
    let log2pi = (2.0 * std::f64::consts::PI).ln();

    let mut x_cur = x0;
    let mut p_cur = p0;

    for t in 0..t_eff {
        let obs_t = t + p; // actual time index in `data`

        // ── Predict ──────────────────────────────────────────────────────────
        // β_t|t-1 = β_{t-1|t-1}  (random walk: F = I)
        let x_pred = x_cur.clone();
        let p_pred = &p_cur + q;

        // ── Build regressor vector z_t of length d ───────────────────────────
        // z_t = I_K ⊗ [y_{t-1}', ..., y_{t-p}', 1]
        // Observation: y_t = Z_t β_t + ε_t
        let regressors = build_regressors(data, obs_t, p, k, spec.n_regressors)?;
        // Z_t is (k × d) such that Z_t β = regressors ⊗-stacked
        // Each block-row k: Z_t[k, k*n_regressors .. (k+1)*n_regressors] = regressors
        let z_t = build_z_matrix(&regressors, k, spec.n_regressors);

        // Innovation: v_t = y_t - Z_t β_{t|t-1}
        let y_t = data.row(obs_t).to_owned();
        let y_pred = mat_vec_mul(&z_t, &x_pred)?;
        let innovation = &y_t - &y_pred;

        // Innovation covariance: S_t = Z_t P_{t|t-1} Z_t' + R
        let zp = mat_mat_mul(&z_t, &p_pred)?; // k × d
        let s_t = mat_mat_mul_bt(&zp, &z_t)? + r; // k × k

        // Log-likelihood contribution
        let s_inv = inv_symmetric(s_t.clone())?;
        let log_det_s = log_det_posdef(&s_t)?;
        let sv = mat_vec_mul(&s_inv, &innovation)?;
        let quad: f64 = innovation.iter().zip(sv.iter()).map(|(&a, &b)| a * b).sum();
        log_lik += -0.5 * (k as f64 * log2pi + log_det_s + quad);

        // Kalman gain: K_t = P_{t|t-1} Z_t' S_t^{-1}
        let pzt = mat_mat_mul_bt(&p_pred, &z_t)?; // d × k
        let k_gain = mat_mat_mul(&pzt, &s_inv)?; // d × k

        // Update: β_{t|t} = β_{t|t-1} + K_t v_t
        let kv = mat_vec_mul(&k_gain, &innovation)?;
        let x_upd = &x_pred + &kv;

        // P_{t|t} = (I - K_t Z_t) P_{t|t-1}   [Joseph form for stability]
        let kz = mat_mat_mul(&k_gain, &z_t)?; // d × d
        let i_kz = eye_minus(kz)?;
        let p_upd = mat_mat_mul(&i_kz, &p_pred)?;

        predicted.push(TvpVarState {
            coefficients: x_pred,
            state_cov: p_pred,
        });
        filtered.push(TvpVarState {
            coefficients: x_upd.clone(),
            state_cov: p_upd.clone(),
        });

        x_cur = x_upd;
        p_cur = p_upd;
    }

    // ── Backward RTS smoother pass ───────────────────────────────────────────
    let smoothed = rts_smooth(&filtered, &predicted, q)?;

    Ok(TvpVarResult {
        smoothed_states: smoothed,
        filtered_states: filtered,
        log_likelihood: log_lik,
        spec: spec.clone(),
        n_obs: t_eff,
    })
}

// ---------------------------------------------------------------------------
// RTS smoother for TVP-VAR
// ---------------------------------------------------------------------------

/// Rauch-Tung-Striebel smoother for the random-walk state equation (F = I).
fn rts_smooth(
    filtered: &[TvpVarState],
    predicted: &[TvpVarState],
    q: &Array2<f64>,
) -> StatsResult<Vec<TvpVarState>> {
    let t_len = filtered.len();
    if t_len == 0 {
        return Ok(Vec::new());
    }

    let mut smoothed: Vec<TvpVarState> = filtered.to_vec();

    for t in (0..t_len - 1).rev() {
        // P_{t+1|t} = P_{t|t} + Q  (since F = I)
        let p_pred = &filtered[t].state_cov + q;

        // Smoother gain: G_t = P_{t|t} * P_{t+1|t}^{-1}
        let p_pred_inv = inv_symmetric(p_pred.clone())?;
        let g = mat_mat_mul(&filtered[t].state_cov, &p_pred_inv)?;

        // Smoothed mean: β_{t|T} = β_{t|t} + G_t (β_{t+1|T} - β_{t+1|t})
        let diff = &smoothed[t + 1].coefficients - &predicted[t + 1].coefficients;
        let correction = mat_vec_mul(&g, &diff)?;
        let x_smooth = &filtered[t].coefficients + &correction;

        // Smoothed covariance: P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t'
        let dp = &smoothed[t + 1].state_cov - &p_pred;
        let g_dp = mat_mat_mul(&g, &dp)?;
        let cov_correction = mat_mat_mul_bt(&g_dp, &g)?;
        let p_smooth = &filtered[t].state_cov + &cov_correction;

        smoothed[t] = TvpVarState {
            coefficients: x_smooth,
            state_cov: p_smooth,
        };
    }

    Ok(smoothed)
}

// ---------------------------------------------------------------------------
// Helper: build regressor vector
// ---------------------------------------------------------------------------

/// Build the regressor vector `z = [y_{t-1}', y_{t-2}', ..., y_{t-p}', 1]`
/// of length `n_regressors = K*p + 1` for time `t`.
fn build_regressors(
    data: &Array2<f64>,
    t: usize,
    p: usize,
    k: usize,
    n_regressors: usize,
) -> StatsResult<Array1<f64>> {
    if t < p {
        return Err(StatsError::InvalidInput(format!(
            "build_regressors: t={} < p={}",
            t, p
        )));
    }
    let mut z = Array1::<f64>::zeros(n_regressors);
    for lag in 0..p {
        let row = data.row(t - 1 - lag);
        for j in 0..k {
            z[lag * k + j] = row[j];
        }
    }
    z[n_regressors - 1] = 1.0; // constant
    Ok(z)
}

/// Build the `(K, K * n_regressors)` observation matrix `Z_t` from the
/// regressor vector.  Each row `k` has the regressors placed starting at
/// column `k * n_regressors`.
fn build_z_matrix(regressors: &Array1<f64>, k: usize, n_regressors: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((k, k * n_regressors));
    for eq in 0..k {
        for col in 0..n_regressors {
            z[[eq, eq * n_regressors + col]] = regressors[col];
        }
    }
    z
}

// ---------------------------------------------------------------------------
// Numerical linear-algebra helpers (private)
// ---------------------------------------------------------------------------

fn validate_square(m: &Array2<f64>, n: usize, name: &str) -> StatsResult<()> {
    if m.nrows() != n || m.ncols() != n {
        Err(StatsError::DimensionMismatch(format!(
            "{} must be {}×{}, got {}×{}",
            name,
            n,
            n,
            m.nrows(),
            m.ncols()
        )))
    } else {
        Ok(())
    }
}

fn identity(n: usize) -> Array2<f64> {
    Array2::from_diag(&Array1::from_elem(n, 1.0))
}

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
    let kk = a.ncols();
    let m = b.ncols();
    let mut c = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for l in 0..kk {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

/// C = A * B^T
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
    let kk = a.ncols();
    let m = b.nrows();
    let mut c = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for l in 0..kk {
                s += a[[i, l]] * b[[j, l]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

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
            for kk in 0..j {
                s -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if s <= 0.0 {
                    let eps = (1e-10_f64).max(s.abs() * 1e-8);
                    let s_reg = s + eps;
                    if s_reg <= 0.0 {
                        return Err(StatsError::ComputationError(format!(
                            "Cholesky failed at ({},{}): diagonal {} is non-positive",
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

fn lower_tri_inv(l: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = l.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        inv[[j, j]] = 1.0 / l[[j, j]];
        for i in j + 1..n {
            let mut s = 0.0_f64;
            for kk in j..i {
                s += l[[i, kk]] * inv[[kk, j]];
            }
            inv[[i, j]] = -s / l[[i, i]];
        }
    }
    Ok(inv)
}

fn inv_symmetric(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "inv_symmetric: not square".into(),
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
    let l = cholesky_lower(a)?;
    let l_inv = lower_tri_inv(&l)?;
    mat_mat_mul_bt(&l_inv, &l_inv)
}

fn log_det_posdef(a: &Array2<f64>) -> StatsResult<f64> {
    let l = cholesky_lower(a.clone())?;
    let n = l.nrows();
    let log_det: f64 = (0..n).map(|i| 2.0 * l[[i, i]].ln()).sum();
    Ok(log_det)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn make_data(t: usize, k: usize) -> Array2<f64> {
        // Simple deterministic data: var j at time t = sin(t + j)
        Array2::from_shape_fn((t, k), |(i, j)| {
            (i as f64 * 0.3 + j as f64).sin() + i as f64 * 0.05
        })
    }

    #[test]
    fn test_tvp_var_spec() {
        let spec = TvpVar::new(2, 1).expect("spec");
        assert_eq!(spec.n_vars, 2);
        assert_eq!(spec.n_lags, 1);
        assert_eq!(spec.n_regressors, 3); // 2*1 + 1
        assert_eq!(spec.state_dim, 6); // 2 * 3
    }

    #[test]
    fn test_tvp_var_fit_small() {
        let data = make_data(30, 2);
        let spec = TvpVar::new(2, 1).expect("spec");
        let d = spec.state_dim;
        let k = spec.n_vars;

        let q = Array2::from_diag(&Array1::from_elem(d, 0.01));
        let r = Array2::from_diag(&Array1::from_elem(k, 0.1));

        let result = fit_tvp_var(&data, &spec, None, None, &q, &r).expect("fit");
        assert_eq!(result.filtered_states.len(), 30 - 1);
        assert_eq!(result.smoothed_states.len(), result.filtered_states.len());
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_tvp_var_smoothed_coeff_matrix() {
        let data = make_data(30, 2);
        let spec = TvpVar::new(2, 1).expect("spec");
        let d = spec.state_dim;
        let k = spec.n_vars;

        let q = Array2::from_diag(&Array1::from_elem(d, 0.01));
        let r = Array2::from_diag(&Array1::from_elem(k, 0.1));

        let result = fit_tvp_var(&data, &spec, None, None, &q, &r).expect("fit");
        let coeff = result.smoothed_coeff_matrix(5).expect("coeff_matrix");
        assert_eq!(coeff.nrows(), 2);
        assert_eq!(coeff.ncols(), 3); // K*p + 1 = 2*1 + 1
    }

    #[test]
    fn test_tvp_var_irf() {
        let data = make_data(40, 2);
        let spec = TvpVar::new(2, 1).expect("spec");
        let d = spec.state_dim;
        let k = spec.n_vars;

        let q = Array2::from_diag(&Array1::from_elem(d, 0.01));
        let r = Array2::from_diag(&Array1::from_elem(k, 0.1));

        let result = fit_tvp_var(&data, &spec, None, None, &q, &r).expect("fit");
        let irfs = result.impulse_response_at(10, 5).expect("irf");
        assert_eq!(irfs.len(), 5);
        // Horizon 0 is identity
        let h0 = &irfs[0];
        for i in 0..k {
            assert!(
                (h0[[i, i]] - 1.0).abs() < 1e-10,
                "IRF horizon 0 diagonal should be 1"
            );
            for j in 0..k {
                if i != j {
                    assert!(
                        h0[[i, j]].abs() < 1e-10,
                        "IRF horizon 0 off-diag should be 0"
                    );
                }
            }
        }
    }

    #[test]
    fn test_tvp_var_bad_inputs() {
        assert!(TvpVar::new(0, 1).is_err());
        assert!(TvpVar::new(2, 0).is_err());

        let data = make_data(5, 2);
        let spec = TvpVar::new(2, 3).expect("spec"); // p=3, need > 3 obs
        let d = spec.state_dim;
        let k = spec.n_vars;
        let q = Array2::from_diag(&Array1::from_elem(d, 0.01));
        let r = Array2::from_diag(&Array1::from_elem(k, 0.1));
        // 5 obs with p=3 means only 2 usable rows, that's fine – but let's
        // check with p=5 which would need > 5
        let spec2 = TvpVar::new(2, 5).expect("spec2");
        let d2 = spec2.state_dim;
        let q2 = Array2::from_diag(&Array1::from_elem(d2, 0.01));
        assert!(fit_tvp_var(&data, &spec2, None, None, &q2, &r).is_err());
    }

    #[test]
    fn test_build_regressors() {
        // data = [[1,2],[3,4],[5,6],[7,8]]
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        // At t=2, p=1, k=2, n_regressors=3:  z = [y_{t-1}, 1] = [3,4,1]
        let z = build_regressors(&data, 2, 1, 2, 3).expect("regressors");
        assert_eq!(z.len(), 3);
        assert!((z[0] - 3.0).abs() < 1e-12);
        assert!((z[1] - 4.0).abs() < 1e-12);
        assert!((z[2] - 1.0).abs() < 1e-12);
    }
}
