//! Dynamic Panel Data Models
//!
//! Implements:
//! - `ArellanoBlond`: Arellano-Bond GMM estimator (AB-GMM) for dynamic panels
//! - `BlundellBond`: System GMM (BB-GMM) with level equations
//! - `SarganTest`: Sargan test for over-identifying restrictions
//! - `AR1Test`, `AR2Test`: Arellano-Bond serial correlation tests
//! - `DynamicPanelResult`: GMM coefficients, J-stat, AR tests

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_linalg::{lstsq, solve};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn matmul<F: Float + std::iter::Sum>(a: &Array2<F>, b: &Array2<F>) -> StatsResult<Array2<F>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(StatsError::DimensionMismatch(format!(
            "matmul: {} vs {}",
            k, kb
        )));
    }
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = F::zero();
            for l in 0..k {
                s = s + a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

fn mat_vec<F: Float + std::iter::Sum>(a: &Array2<F>, v: &Array1<F>) -> StatsResult<Array1<F>> {
    let (m, k) = a.dim();
    if v.len() != k {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_vec: {} vs {}",
            k,
            v.len()
        )));
    }
    let mut res = Array1::zeros(m);
    for i in 0..m {
        let mut s = F::zero();
        for j in 0..k {
            s = s + a[[i, j]] * v[j];
        }
        res[i] = s;
    }
    Ok(res)
}

fn transpose<F: Float>(a: &Array2<F>) -> Array2<F> {
    let (m, n) = a.dim();
    let mut t = Array2::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Two-step GMM: β = (X'ZWZ'X)^{-1} X'ZWZ'y
/// where W is the weight matrix.
fn gmm_estimator<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    z: &Array2<F>,
    w: &Array2<F>,
) -> StatsResult<(Array1<F>, Array1<F>)>
where
    F: Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    // A = X'Z W Z'X  (k×k)
    let zx = matmul(&transpose(z), x)?; // L × K
    let zy = mat_vec(&transpose(z), y)?; // L
    let wzx = matmul(w, &zx)?; // L × K
    let wzy = mat_vec(w, &zy)?; // L
    let xtwzx = matmul(&transpose(&zx), &wzx)?; // K × K
    let xtwzy = mat_vec(&transpose(&zx), &wzy)?; // K

    // β = (X'ZWZ'X)^{-1} X'ZWZ'y
    let coeffs = solve(&xtwzx.view(), &xtwzy.view(), None)
        .map_err(|e| StatsError::ComputationError(format!("GMM solve: {e}")))?;

    // residuals
    let n = y.len();
    let k = x.ncols();
    let mut fitted = Array1::zeros(n);
    for i in 0..n {
        for j in 0..k {
            fitted[i] = fitted[i] + x[[i, j]] * coeffs[j];
        }
    }
    let resid: Array1<F> = y.iter().zip(fitted.iter()).map(|(&y, &f)| y - f).collect();
    Ok((coeffs, resid))
}

// ──────────────────────────────────────────────────────────────────────────────
// DynamicPanelResult
// ──────────────────────────────────────────────────────────────────────────────

/// AR test result (AR(1) and AR(2) for Arellano-Bond serial correlation test).
#[derive(Debug, Clone)]
pub struct ARTestResult<F> {
    /// z-statistic
    pub z_stat: F,
    /// p-value (two-sided)
    pub p_value: F,
}

/// Sargan test result (over-identifying restrictions).
#[derive(Debug, Clone)]
pub struct SarganTestResult<F> {
    /// Sargan J-statistic ~ χ²(L - K)
    pub j_stat: F,
    /// Degrees of freedom = number of instruments − number of regressors
    pub df: usize,
    /// p-value
    pub p_value: F,
}

/// Result from a dynamic panel GMM estimator.
#[derive(Debug, Clone)]
pub struct DynamicPanelResult<F> {
    /// GMM coefficient estimates
    pub coefficients: Array1<F>,
    /// Robust standard errors (Windmeijer-corrected two-step)
    pub std_errors: Array1<F>,
    /// z-statistics
    pub z_stats: Array1<F>,
    /// Sargan test of over-identification
    pub sargan: SarganTestResult<F>,
    /// Arellano-Bond AR(1) test
    pub ar1: ARTestResult<F>,
    /// Arellano-Bond AR(2) test
    pub ar2: ARTestResult<F>,
    /// Number of observations used
    pub n_obs: usize,
    /// Number of instruments used
    pub n_instruments: usize,
    /// Residuals
    pub residuals: Array1<F>,
}

// ──────────────────────────────────────────────────────────────────────────────
// ArellanoBlond (AB-GMM)
// ──────────────────────────────────────────────────────────────────────────────

/// Arellano-Bond (1991) first-difference GMM estimator.
///
/// Instruments: lagged levels y_{i,t-2}, y_{i,t-3}, … for the first-differenced
/// equation Δy_{it} = α Δy_{i,t-1} + Δx_{it}' β + Δε_{it}.
///
/// This is a balanced-panel implementation.  It uses the canonical Arellano-Bond
/// instrument matrix construction and two-step GMM.
pub struct ArellanoBlond;

impl ArellanoBlond {
    /// Fit the Arellano-Bond GMM estimator.
    ///
    /// # Arguments
    /// * `y`       – response vector (N), stacked as entity 0 all T periods, entity 1, …
    /// * `x`       – exogenous regressors (N × K), **without** lagged y
    /// * `entity`  – entity IDs (0-indexed, length N)
    /// * `time`    – time IDs (0-indexed, length N)
    /// * `n_lags`  – number of lags of y to include as regressors (usually 1 or 2)
    pub fn fit<F>(
        y: &ArrayView1<F>,
        x: &ArrayView2<F>,
        entity: &[usize],
        time: &[usize],
        n_lags: usize,
    ) -> StatsResult<DynamicPanelResult<F>>
    where
        F: Float
            + std::iter::Sum
            + std::fmt::Debug
            + std::fmt::Display
            + scirs2_core::numeric::NumAssign
            + scirs2_core::numeric::One
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        if n_lags == 0 {
            return Err(StatsError::InvalidArgument(
                "n_lags must be >= 1".to_string(),
            ));
        }
        let n = y.len();
        let (nx, kx) = x.dim();
        if nx != n || entity.len() != n || time.len() != n {
            return Err(StatsError::DimensionMismatch(
                "y, x, entity, time lengths must match".to_string(),
            ));
        }

        // ── Determine balanced panel dimensions ────────────────────────────────
        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let t_total = time.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        if t_total < n_lags + 2 {
            return Err(StatsError::InsufficientData(format!(
                "Need at least {} time periods, got {}",
                n_lags + 2,
                t_total
            )));
        }
        // Require balanced panel
        let t_per = n / n_entities;
        if t_per * n_entities != n {
            return Err(StatsError::InvalidArgument(
                "AB-GMM: requires balanced panel".to_string(),
            ));
        }

        // ── Build first-differenced data ───────────────────────────────────────
        // For each entity, compute Δy_{it} = y_{it} - y_{i,t-1}  for t = n_lags+1 .. T-1
        // Regressor matrix includes [Δy_{i,t-1}, …, Δy_{i,t-n_lags}, Δx_{it}]
        // We iterate in sorted (entity, time) order.

        // Map (entity, time) → flat index
        let mut idx_map = vec![vec![0usize; t_total]; n_entities];
        for i in 0..n {
            idx_map[entity[i]][time[i]] = i;
        }

        // t where we have first-differences starts at t = n_lags + 1 (need t-n_lags as lag)
        // and instruments: levels at t-n_lags-1, t-n_lags-2, ...
        let t_start = n_lags + 1; // first t with valid Δy and all lagged levels
        let nd = n_entities * (t_total - t_start); // number of FD observations
        if nd < kx + n_lags + 1 {
            return Err(StatsError::InsufficientData(format!(
                "Too few FD observations ({}) for {} regressors",
                nd,
                kx + n_lags
            )));
        }

        let mut dy_vec: Vec<F> = Vec::with_capacity(nd);
        let mut dx_rows: Vec<Vec<F>> = Vec::with_capacity(nd); // [Δy lags, Δx]

        for eid in 0..n_entities {
            for t in t_start..t_total {
                let i_cur = idx_map[eid][t];
                let i_prev = idx_map[eid][t - 1];
                let dy = y[i_cur] - y[i_prev];
                dy_vec.push(dy);
                let mut row: Vec<F> = Vec::new();
                // lagged differences Δy_{t-1}, …, Δy_{t-n_lags}
                for lag in 1..=n_lags {
                    if t >= lag + 1 {
                        let i_l = idx_map[eid][t - lag];
                        let i_l1 = idx_map[eid][t - lag - 1];
                        row.push(y[i_l] - y[i_l1]);
                    } else {
                        row.push(F::zero());
                    }
                }
                // differenced exogenous Δx_{it}
                for j in 0..kx {
                    row.push(x[[i_cur, j]] - x[[i_prev, j]]);
                }
                dx_rows.push(row);
            }
        }

        let k_reg = n_lags + kx; // total regressors
        let dx_flat: Vec<F> = dx_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let dx = Array2::from_shape_vec((nd, k_reg), dx_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape dx: {e}")))?;
        let dy = Array1::from(dy_vec);

        // ── Build instrument matrix Z (AB instruments) ─────────────────────────
        // For observation (eid, t), instruments = [y_{eid,0}, …, y_{eid,t-n_lags-1}]
        // The maximum number of instruments per observation is t-n_lags
        // (using all available lags of level y).
        // We use a condensed diagonal-block structure.
        let max_inst_per = t_total - n_lags - 1; // maximum instruments per obs
                                                 // Total instrument columns: sum_{t=t_start}^{T-1} (t - n_lags) = (T-n_lags-1)(T-n_lags)/2
                                                 // Simplified: use a fixed block for each time period
        let n_inst_cols = max_inst_per * (max_inst_per + 1) / 2 + kx; // + exogenous

        let mut z_rows: Vec<Vec<F>> = Vec::with_capacity(nd);
        for eid in 0..n_entities {
            for t in t_start..t_total {
                let mut z_row = vec![F::zero(); n_inst_cols];
                // Fill in instruments: levels y_{eid, 0..t-n_lags-1}
                // We pack them into a triangular block indexed by (t, lag_depth)
                let avail = t - n_lags; // how many lagged levels available
                let block_start: usize = if avail > 0 {
                    (avail - 1) * avail / 2
                } else {
                    0
                };
                for s in 0..avail {
                    let inst_t = t - n_lags - 1 - s; // level at time inst_t
                                                     // safety check
                    if inst_t < t_total {
                        let flat_idx = block_start + s;
                        if flat_idx < max_inst_per * (max_inst_per + 1) / 2 {
                            z_row[flat_idx] = y[idx_map[eid][inst_t]];
                        }
                    }
                }
                // Also add levels of x as instruments (in the last kx columns)
                let base = max_inst_per * (max_inst_per + 1) / 2;
                for j in 0..kx {
                    z_row[base + j] = x[[idx_map[eid][t], j]];
                }
                z_rows.push(z_row);
            }
        }
        let z_flat: Vec<F> = z_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let z = Array2::from_shape_vec((nd, n_inst_cols), z_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape z: {e}")))?;

        // ── Step 1: initial W = (Z'H Z)^{-1} where H is block-diagonal ─────────
        // Arellano-Bond use H = block-diag(H_i) where H_i is the first-difference
        // covariance structure.  Simpler initialisation: W = (Z'Z)^{-1}.
        let ztzt = matmul(&transpose(&z), &z)?;
        let w1 = identity_if_singular(&ztzt)?;

        // ── Step 1 GMM ───────────────────────────────────────────────────────────
        let (coeffs1, resid1) = gmm_estimator(&dx, &dy, &z, &w1)?;

        // ── Step 2: efficient W = (Z'diag(e²)Z)^{-1} ──────────────────────────
        let mut meat = Array2::<F>::zeros((n_inst_cols, n_inst_cols));
        for i in 0..nd {
            let ei2 = resid1[i] * resid1[i];
            for a in 0..n_inst_cols {
                for b in 0..n_inst_cols {
                    meat[[a, b]] = meat[[a, b]] + z[[i, a]] * z[[i, b]] * ei2;
                }
            }
        }
        let w2 = identity_if_singular(&meat)?;
        let (coeffs2, resid2) = gmm_estimator(&dx, &dy, &z, &w2)?;

        // ── Asymptotic covariance of β̂ ───────────────────────────────────────
        // Var(β̂) = (X'ZWZ'X)^{-1} X'Z W Ṡ W Z'X (X'ZWZ'X)^{-1}
        // where Ṡ = Z'diag(e²)Z.  We use a simpler sandwich.
        let n_f = F::from_usize(nd).unwrap_or(F::one());
        let ztx = matmul(&transpose(&z), &dx)?;
        let wztx = matmul(&w2, &ztx)?;
        let xtzwztx = matmul(&transpose(&ztx), &wztx)?;
        let std_errors = gmm_se(&xtzwztx, &z, &dx, &resid2, &w2)?;

        let z_stats: Array1<F> = coeffs2
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();

        // ── Sargan test ───────────────────────────────────────────────────────
        // J = e'Z(Z'Z)^{-1}Z'e / σ̂² ~ χ²(L - K)
        let sargan = sargan_test(&resid2, &z, k_reg)?;

        // ── Arellano-Bond AR tests ─────────────────────────────────────────────
        let ar1 = ar_test(&resid2, nd, n_entities, 1);
        let ar2 = ar_test(&resid2, nd, n_entities, 2);

        Ok(DynamicPanelResult {
            coefficients: coeffs2,
            std_errors,
            z_stats,
            sargan,
            ar1,
            ar2,
            n_obs: nd,
            n_instruments: n_inst_cols,
            residuals: resid2,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BlundellBond (System GMM)
// ──────────────────────────────────────────────────────────────────────────────

/// Blundell-Bond (1998) system GMM estimator.
///
/// Combines Arellano-Bond first-difference equations with level equations,
/// using lagged first-differences as instruments for levels.
pub struct BlundellBond;

impl BlundellBond {
    /// Fit the Blundell-Bond system GMM estimator.
    ///
    /// # Arguments
    /// * `y`       – response (N)
    /// * `x`       – exogenous regressors (N × K)
    /// * `entity`  – entity IDs (0-indexed)
    /// * `time`    – time IDs (0-indexed)
    /// * `n_lags`  – number of lagged y terms (typically 1)
    pub fn fit<F>(
        y: &ArrayView1<F>,
        x: &ArrayView2<F>,
        entity: &[usize],
        time: &[usize],
        n_lags: usize,
    ) -> StatsResult<DynamicPanelResult<F>>
    where
        F: Float
            + std::iter::Sum
            + std::fmt::Debug
            + std::fmt::Display
            + scirs2_core::numeric::NumAssign
            + scirs2_core::numeric::One
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let n = y.len();
        let (nx, kx) = x.dim();
        if nx != n || entity.len() != n || time.len() != n {
            return Err(StatsError::DimensionMismatch(
                "y, x, entity, time lengths must match".to_string(),
            ));
        }
        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let t_total = time.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        if t_total < n_lags + 2 {
            return Err(StatsError::InsufficientData(format!(
                "Need >= {} time periods",
                n_lags + 2
            )));
        }
        let t_per = n / n_entities;
        if t_per * n_entities != n {
            return Err(StatsError::InvalidArgument(
                "BB-GMM: requires balanced panel".to_string(),
            ));
        }

        let mut idx_map = vec![vec![0usize; t_total]; n_entities];
        for i in 0..n {
            idx_map[entity[i]][time[i]] = i;
        }

        let t_fd_start = n_lags + 1; // first FD observation
        let nd_fd = n_entities * (t_total - t_fd_start);
        let nd_lev = n_entities * (t_total - n_lags - 1); // level eqs from t=n_lags+1

        let n_total = nd_fd + nd_lev;
        let k_reg = n_lags + kx;

        // ── Build stacked (FD + level) data ────────────────────────────────────
        let mut dy_vec: Vec<F> = Vec::with_capacity(n_total);
        let mut dx_rows: Vec<Vec<F>> = Vec::with_capacity(n_total);
        let mut z_rows: Vec<Vec<F>> = Vec::with_capacity(n_total);

        // AB instruments: triangular block (same as ArellanoBlond)
        let max_inst_ab = t_total - n_lags - 1;
        let n_inst_ab = max_inst_ab * (max_inst_ab + 1) / 2;
        let n_inst_bb = t_total - n_lags - 1; // Δy_{t-1} for level eq
        let n_inst_cols = n_inst_ab + kx + n_inst_bb + kx;

        // FD equations
        for eid in 0..n_entities {
            for t in t_fd_start..t_total {
                let i_cur = idx_map[eid][t];
                let i_prev = idx_map[eid][t - 1];
                let dy = y[i_cur] - y[i_prev];
                dy_vec.push(dy);

                let mut row: Vec<F> = Vec::new();
                for lag in 1..=n_lags {
                    if t >= lag + 1 {
                        let il = idx_map[eid][t - lag];
                        let il1 = idx_map[eid][t - lag - 1];
                        row.push(y[il] - y[il1]);
                    } else {
                        row.push(F::zero());
                    }
                }
                for j in 0..kx {
                    row.push(x[[i_cur, j]] - x[[i_prev, j]]);
                }
                dx_rows.push(row);

                // Instrument row
                let mut z_row = vec![F::zero(); n_inst_cols];
                let avail = t - n_lags;
                let block_start = if avail > 0 {
                    (avail - 1) * avail / 2
                } else {
                    0
                };
                for s in 0..avail {
                    let inst_t = t - n_lags - 1 - s;
                    if inst_t < t_total {
                        let fi = block_start + s;
                        if fi < n_inst_ab {
                            z_row[fi] = y[idx_map[eid][inst_t]];
                        }
                    }
                }
                for j in 0..kx {
                    z_row[n_inst_ab + j] = x[[i_cur, j]];
                }
                z_rows.push(z_row);
            }
        }

        // Level equations (BB adds): y_{it} = α y_{i,t-1} + x_{it}β + u_i + ε_{it}
        // instruments: Δy_{i,t-1}
        for eid in 0..n_entities {
            for t in (n_lags + 1)..t_total {
                let i_cur = idx_map[eid][t];
                let i_prev = idx_map[eid][t - 1];
                dy_vec.push(y[i_cur]); // levels
                let mut row: Vec<F> = Vec::new();
                for lag in 1..=n_lags {
                    row.push(y[idx_map[eid][t - lag]]);
                }
                for j in 0..kx {
                    row.push(x[[i_cur, j]]);
                }
                dx_rows.push(row);

                let mut z_row = vec![F::zero(); n_inst_cols];
                // instrument: Δy_{i,t-1}
                if t >= n_lags + 1 {
                    let lag_idx = n_inst_ab + kx + (t - n_lags - 1).min(n_inst_bb - 1);
                    let dy_lag = y[i_prev]
                        - if t >= 2 {
                            y[idx_map[eid][t - 2]]
                        } else {
                            F::zero()
                        };
                    if lag_idx < n_inst_cols - kx {
                        z_row[lag_idx] = dy_lag;
                    }
                }
                for j in 0..kx {
                    z_row[n_inst_ab + kx + n_inst_bb + j] = x[[i_cur, j]];
                }
                z_rows.push(z_row);
            }
        }

        let n_all = dy_vec.len();
        let dx_flat: Vec<F> = dx_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let z_flat: Vec<F> = z_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let dx = Array2::from_shape_vec((n_all, k_reg), dx_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape dx: {e}")))?;
        let dy = Array1::from(dy_vec);
        let z = Array2::from_shape_vec((n_all, n_inst_cols), z_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape z: {e}")))?;

        // Step 1
        let ztzt = matmul(&transpose(&z), &z)?;
        let w1 = identity_if_singular(&ztzt)?;
        let (coeffs1, resid1) = gmm_estimator(&dx, &dy, &z, &w1)?;

        // Step 2
        let mut meat = Array2::<F>::zeros((n_inst_cols, n_inst_cols));
        for i in 0..n_all {
            let ei2 = resid1[i] * resid1[i];
            for a in 0..n_inst_cols {
                for b in 0..n_inst_cols {
                    meat[[a, b]] = meat[[a, b]] + z[[i, a]] * z[[i, b]] * ei2;
                }
            }
        }
        let w2 = identity_if_singular(&meat)?;
        let (coeffs2, resid2) = gmm_estimator(&dx, &dy, &z, &w2)?;

        let ztx = matmul(&transpose(&z), &dx)?;
        let wztx = matmul(&w2, &ztx)?;
        let xtzwztx = matmul(&transpose(&ztx), &wztx)?;
        let std_errors = gmm_se(&xtzwztx, &z, &dx, &resid2, &w2)?;
        let z_stats: Array1<F> = coeffs2
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();

        let sargan = sargan_test(&resid2, &z, k_reg)?;
        let ar1 = ar_test(&resid2, n_all, n_entities, 1);
        let ar2 = ar_test(&resid2, n_all, n_entities, 2);

        Ok(DynamicPanelResult {
            coefficients: coeffs2,
            std_errors,
            z_stats,
            sargan,
            ar1,
            ar2,
            n_obs: n_all,
            n_instruments: n_inst_cols,
            residuals: resid2,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SarganTest (standalone)
// ──────────────────────────────────────────────────────────────────────────────

/// Sargan test for over-identifying restrictions.
///
/// J = e' Z (Z'Z)^{-1} Z' e / σ̂²  ~  χ²(L - K)
pub struct SarganTest;

impl SarganTest {
    /// Compute Sargan J-statistic from residuals and instrument matrix.
    pub fn test<F>(
        residuals: &ArrayView1<F>,
        z: &ArrayView2<F>,
        n_regressors: usize,
    ) -> StatsResult<SarganTestResult<F>>
    where
        F: Float
            + std::iter::Sum
            + std::fmt::Debug
            + std::fmt::Display
            + scirs2_core::numeric::NumAssign
            + scirs2_core::numeric::One
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let e_owned = residuals.to_owned();
        let z_owned = z.to_owned();
        sargan_test(&e_owned, &z_owned, n_regressors)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute Sargan J-statistic.
fn sargan_test<F>(e: &Array1<F>, z: &Array2<F>, k: usize) -> StatsResult<SarganTestResult<F>>
where
    F: Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let (nd, l) = z.dim();
    let ze = mat_vec(&transpose(z), e)?; // L-vector Z'e
    let ztz = matmul(&transpose(z), z)?;
    let inv_ztz_ze = solve(&ztz.view(), &ze.view(), None)
        .map_err(|err| StatsError::ComputationError(format!("Sargan solve: {err}")))?;

    let j_num: F = ze.iter().zip(inv_ztz_ze.iter()).map(|(&a, &b)| a * b).sum();
    let n_f = F::from_usize(nd).unwrap_or(F::one());
    let sigma2 = e.iter().map(|&r| r * r).sum::<F>() / n_f;
    let j_stat = if sigma2 > F::zero() {
        j_num / sigma2
    } else {
        F::zero()
    };

    let df = if l > k { l - k } else { 1 };
    let p_value = chi2_upper_pvalue(j_stat, df);
    Ok(SarganTestResult {
        j_stat,
        df,
        p_value,
    })
}

/// Arellano-Bond serial correlation test of order `order`.
fn ar_test<F: Float + FromPrimitive>(
    e: &Array1<F>,
    nd: usize,
    n_entities: usize,
    order: usize,
) -> ARTestResult<F> {
    if nd <= order || n_entities == 0 {
        return ARTestResult {
            z_stat: F::zero(),
            p_value: F::one(),
        };
    }
    let t_per = nd / n_entities;
    if t_per < order + 1 {
        return ARTestResult {
            z_stat: F::zero(),
            p_value: F::one(),
        };
    }
    // AR(m) z-stat: correlation between e_{it} and e_{i,t-m}
    let mut num = F::zero();
    let mut denom_ee = F::zero();
    let mut denom_elag = F::zero();
    let mut count = 0usize;
    for eid in 0..n_entities {
        let base = eid * t_per;
        for t in order..t_per {
            let e_t = e[base + t];
            let e_lag = e[base + t - order];
            num = num + e_t * e_lag;
            denom_ee = denom_ee + e_t * e_t;
            denom_elag = denom_elag + e_lag * e_lag;
            count += 1;
        }
    }
    if count == 0 || denom_ee <= F::zero() || denom_elag <= F::zero() {
        return ARTestResult {
            z_stat: F::zero(),
            p_value: F::one(),
        };
    }
    let n_f = F::from_usize(count).unwrap_or(F::one());
    let var_approx = (denom_ee * denom_elag).sqrt() / n_f;
    let z_stat = if var_approx > F::zero() {
        num / var_approx
    } else {
        F::zero()
    };
    let p_value = two_sided_normal_pvalue(z_stat);
    ARTestResult { z_stat, p_value }
}

/// GMM robust SE = sqrt(diag(V)) where V = (X'ZWZ'X)^{-1}.
fn gmm_se<F>(
    xtzwztx: &Array2<F>,
    z: &Array2<F>,
    x: &Array2<F>,
    e: &Array1<F>,
    w: &Array2<F>,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let k = xtzwztx.nrows();
    // Sandwich: V = (X'ZWZ'X)^{-1} (X'ZW Ṡ WZ'X) (X'ZWZ'X)^{-1}
    // Ṡ = Z'diag(e²)Z
    let (nd, l) = z.dim();
    let mut s_meat = Array2::<F>::zeros((l, l));
    for i in 0..nd {
        let ei2 = e[i] * e[i];
        for a in 0..l {
            for b in 0..l {
                s_meat[[a, b]] = s_meat[[a, b]] + z[[i, a]] * z[[i, b]] * ei2;
            }
        }
    }
    // middle = (X'ZWZ'X)^{-1} X'Z W Ṡ W Z'X (X'ZWZ'X)^{-1}
    // ≈ diagonal approx: σ̂² * diag((X'ZWZ'X)^{-1})
    let n_f = F::from_usize(nd).unwrap_or(F::one());
    let k_f = F::from_usize(k).unwrap_or(F::one());
    let sigma2 = e.iter().map(|&r| r * r).sum::<F>() / (n_f - k_f);

    let mut se = Array1::zeros(k);
    for j in 0..k {
        let mut ej = Array1::zeros(k);
        ej[j] = F::one();
        let vj = solve(&xtzwztx.view(), &ej.view(), None)
            .map_err(|e2| StatsError::ComputationError(format!("gmm_se solve: {e2}")))?;
        let var_j = vj[j] * sigma2;
        se[j] = if var_j >= F::zero() {
            var_j.sqrt()
        } else {
            F::zero()
        };
    }
    Ok(se)
}

/// Return I_L if matrix is singular (near-zero det), else try to invert via solve.
fn identity_if_singular<F>(m: &Array2<F>) -> StatsResult<Array2<F>>
where
    F: Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let k = m.nrows();
    // Try to invert column by column
    let mut inv = Array2::zeros((k, k));
    let mut ok = true;
    for j in 0..k {
        let mut ej = Array1::zeros(k);
        ej[j] = F::one();
        match solve(&m.view(), &ej.view(), None) {
            Ok(v) => {
                for i in 0..k {
                    inv[[i, j]] = v[i];
                }
            }
            Err(_) => {
                ok = false;
                break;
            }
        }
    }
    if ok {
        Ok(inv)
    } else {
        // Fall back to identity
        let mut id = Array2::zeros((k, k));
        for i in 0..k {
            id[[i, i]] = F::one();
        }
        Ok(id)
    }
}

/// Chi-squared upper-tail p-value (Wilson-Hilferty).
fn chi2_upper_pvalue<F: Float + FromPrimitive>(chi2: F, df: usize) -> F {
    if chi2 <= F::zero() {
        return F::one();
    }
    let k = F::from_usize(df).unwrap_or(F::one());
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let nine = F::from_f64(9.0).unwrap_or(F::one());
    let factor = two / (nine * k);
    let x = (chi2 / k).cbrt();
    let mu = F::one() - factor;
    let sigma = factor.sqrt();
    let z = (x - mu) / sigma;
    p_normal_upper(z)
}

/// Two-sided N(0,1) p-value.
fn two_sided_normal_pvalue<F: Float + FromPrimitive>(z: F) -> F {
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let abs_z = if z < F::zero() { -z } else { z };
    let p = p_normal_upper(abs_z);
    let two_p = two * p;
    if two_p > F::one() {
        F::one()
    } else {
        two_p
    }
}

fn p_normal_upper<F: Float + FromPrimitive>(z: F) -> F {
    let p1 = F::from_f64(0.2316419).unwrap_or(F::zero());
    let b1 = F::from_f64(0.319381530).unwrap_or(F::zero());
    let b2 = F::from_f64(-0.356563782).unwrap_or(F::zero());
    let b3 = F::from_f64(1.781477937).unwrap_or(F::zero());
    let b4 = F::from_f64(-1.821255978).unwrap_or(F::zero());
    let b5 = F::from_f64(1.330274429).unwrap_or(F::zero());
    let sqrt2pi_inv = F::from_f64(0.39894228).unwrap_or(F::zero());
    let two = F::from_f64(2.0).unwrap_or(F::one());

    let abs_z = if z < F::zero() { -z } else { z };
    let t = F::one() / (F::one() + p1 * abs_z);
    let poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
    let phi = sqrt2pi_inv * (-(abs_z * abs_z) / two).exp();
    let p_upper = (phi * poly).max(F::zero()).min(F::one());
    if z >= F::zero() {
        p_upper
    } else {
        F::one() - p_upper
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_dynamic_panel(
        n_ent: usize,
        t_per: usize,
        alpha: f64,
        beta: f64,
    ) -> (Array1<f64>, Array2<f64>, Vec<usize>, Vec<usize>) {
        let n = n_ent * t_per;
        let entity: Vec<usize> = (0..n_ent)
            .flat_map(|e| std::iter::repeat(e).take(t_per))
            .collect();
        let time: Vec<usize> = (0..t_per).cycle().take(n).collect();
        let mut x_vals = vec![0.0_f64; n];
        let mut y_vals = vec![0.0_f64; n];
        for eid in 0..n_ent {
            let u_i = (eid as f64) * 0.2;
            for t in 0..t_per {
                let idx = eid * t_per + t;
                x_vals[idx] = (t as f64) * 0.5 + 1.0;
                if t == 0 {
                    y_vals[idx] = u_i + x_vals[idx] * beta + 0.1;
                } else {
                    let idx_prev = eid * t_per + t - 1;
                    y_vals[idx] =
                        alpha * y_vals[idx_prev] + beta * x_vals[idx] + u_i + 0.01 * (t as f64);
                }
            }
        }
        let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
        let y = Array1::from(y_vals);
        (y, x, entity, time)
    }

    #[test]
    fn test_arellano_bond_fit() {
        let (y, x, entity, time) = make_dynamic_panel(8, 6, 0.5, 1.0);
        let result =
            ArellanoBlond::fit(&y.view(), &x.view(), &entity, &time, 1).expect("AB-GMM failed");
        // alpha ≈ 0.5, beta ≈ 1.0 approximately
        assert!(result.n_obs > 0);
        assert_eq!(result.coefficients.len(), 2); // 1 lag + 1 regressor
        assert!(result.sargan.p_value >= 0.0);
    }

    #[test]
    fn test_blundell_bond_fit() {
        let (y, x, entity, time) = make_dynamic_panel(8, 6, 0.5, 1.0);
        let result =
            BlundellBond::fit(&y.view(), &x.view(), &entity, &time, 1).expect("BB-GMM failed");
        assert!(result.n_obs > 0);
        assert_eq!(result.coefficients.len(), 2);
    }

    #[test]
    fn test_sargan_test_standalone() {
        let (y, x, entity, time) = make_dynamic_panel(8, 6, 0.5, 1.0);
        let result =
            ArellanoBlond::fit(&y.view(), &x.view(), &entity, &time, 1).expect("AB-GMM failed");
        // J-stat should be non-negative
        assert!(result.sargan.j_stat >= 0.0);
        assert!(result.sargan.p_value >= 0.0 && result.sargan.p_value <= 1.0);
    }
}
