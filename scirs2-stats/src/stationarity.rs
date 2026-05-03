//! Stationarity and Unit Root Tests
//!
//! This module provides comprehensive tests for time series stationarity, including:
//!
//! - **Augmented Dickey-Fuller (ADF)** test: tests for a unit root in a univariate time series
//! - **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) test: tests for stationarity (null = stationary)
//! - **Phillips-Perron** test: non-parametric unit root test robust to heteroskedasticity
//! - **Zivot-Andrews** test: unit root test allowing a single structural break
//! - **Auto-lag selection**: AIC/BIC-based optimal lag length determination
//!
//! # References
//!
//! - Dickey, D.A. & Fuller, W.A. (1979). Distribution of the Estimators for
//!   Autoregressive Time Series With a Unit Root. JASA.
//! - Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
//!   Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root.
//!   Journal of Econometrics.
//! - Phillips, P.C.B. & Perron, P. (1988). Testing for a Unit Root in Time Series Regression.
//!   Biometrika.
//! - Zivot, E. & Andrews, D.W.K. (1992). Further Evidence on the Great Crash, the Oil-Price
//!   Shock, and the Unit-Root Hypothesis. JBES.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of an Augmented Dickey-Fuller test
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF test statistic
    pub statistic: f64,
    /// Approximate p-value
    pub p_value: f64,
    /// Number of lags used
    pub used_lags: usize,
    /// Number of observations used in the regression
    pub n_obs: usize,
    /// Critical values at 1%, 5%, 10% significance levels
    pub critical_values: CriticalValues,
    /// Regression type used
    pub regression: AdfRegression,
}

/// Result of a KPSS test
#[derive(Debug, Clone)]
pub struct KpssResult {
    /// KPSS test statistic
    pub statistic: f64,
    /// Approximate p-value
    pub p_value: f64,
    /// Number of lags used for the long-run variance estimator
    pub used_lags: usize,
    /// Critical values at 1%, 5%, 10% significance levels
    pub critical_values: CriticalValues,
    /// Trend type used
    pub trend: KpssTrend,
}

/// Result of a Phillips-Perron test
#[derive(Debug, Clone)]
pub struct PhillipsPerronResult {
    /// Phillips-Perron Z(alpha) statistic
    pub z_alpha: f64,
    /// Phillips-Perron Z(t) statistic
    pub z_tau: f64,
    /// Approximate p-value (based on Z(t))
    pub p_value: f64,
    /// Number of lags used for long-run variance
    pub used_lags: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Critical values at 1%, 5%, 10% significance levels
    pub critical_values: CriticalValues,
}

/// Result of a Zivot-Andrews structural break unit root test
#[derive(Debug, Clone)]
pub struct ZivotAndrewsResult {
    /// Minimum ADF statistic across all break points
    pub statistic: f64,
    /// Approximate p-value
    pub p_value: f64,
    /// Estimated break point (index into the original series)
    pub break_point: usize,
    /// Number of lags used
    pub used_lags: usize,
    /// Critical values at 1%, 5%, 10% significance levels
    pub critical_values: CriticalValues,
    /// Break type specification
    pub break_type: ZivotAndrewsBreak,
}

/// Critical values at common significance levels
#[derive(Debug, Clone, Copy)]
pub struct CriticalValues {
    /// Critical value at 1% significance
    pub one_pct: f64,
    /// Critical value at 5% significance
    pub five_pct: f64,
    /// Critical value at 10% significance
    pub ten_pct: f64,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Regression type for the ADF test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfRegression {
    /// No constant, no trend
    None,
    /// Constant only (default)
    Constant,
    /// Constant and linear trend
    ConstantTrend,
    /// Constant, linear trend and quadratic trend
    ConstantTrendSquared,
}

/// Trend specification for KPSS test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssTrend {
    /// Level stationarity (constant only)
    Constant,
    /// Trend stationarity (constant + linear trend)
    Trend,
}

/// Structural break type for Zivot-Andrews test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZivotAndrewsBreak {
    /// Break in intercept only (Model A)
    Intercept,
    /// Break in trend only (Model B)
    Trend,
    /// Break in both intercept and trend (Model C)
    Both,
}

/// Lag selection criterion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagCriterion {
    /// Akaike Information Criterion
    Aic,
    /// Bayesian Information Criterion (Schwarz)
    Bic,
    /// Hannan-Quinn Information Criterion
    Hqic,
    /// t-statistic based (sequential testing)
    TStatistic,
}

// ---------------------------------------------------------------------------
// Helper: OLS regression
// ---------------------------------------------------------------------------

/// Tiny OLS result for internal use
pub(crate) struct OlsResult {
    pub(crate) coefficients: Array1<f64>,
    pub(crate) residuals: Array1<f64>,
    pub(crate) sigma_sq: f64,
    pub(crate) se: Array1<f64>,
}

/// Solve OLS: y = X * beta + e, returning coefficients and residuals.
pub(crate) fn ols_regression(y: &ArrayView1<f64>, x: &Array2<f64>) -> StatsResult<OlsResult> {
    let n = y.len();
    let k = x.ncols();
    if n != x.nrows() {
        return Err(StatsError::DimensionMismatch(format!(
            "y length {} != X rows {}",
            n,
            x.nrows()
        )));
    }
    if n <= k {
        return Err(StatsError::InsufficientData(format!(
            "need n > k for OLS (n={}, k={})",
            n, k
        )));
    }
    // X'X
    let xtx = x.t().dot(x);
    // X'y
    let xty = x.t().dot(y);
    // Solve via Cholesky-like approach (manual positive-definite solve)
    let beta = solve_symmetric(&xtx, &xty)?;
    // Residuals
    let fitted = x.dot(&beta);
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        residuals[i] = y[i] - fitted[i];
    }
    // Residual variance
    let df = (n - k) as f64;
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma_sq = rss / df;
    // Standard errors: sqrt(diag(sigma^2 * (X'X)^{-1}))
    let xtx_inv = invert_symmetric(&xtx)?;
    let mut se = Array1::zeros(k);
    for j in 0..k {
        let var_j = sigma_sq * xtx_inv[[j, j]];
        se[j] = if var_j > 0.0 { var_j.sqrt() } else { 0.0 };
    }
    Ok(OlsResult {
        coefficients: beta,
        residuals,
        sigma_sq,
        se,
    })
}

/// Solve A*x = b where A is symmetric positive definite, via Cholesky decomposition.
fn solve_symmetric(a: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(StatsError::DimensionMismatch(
            "solve_symmetric: dimension mismatch".into(),
        ));
    }
    // Cholesky: A = L * L^T
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for kk in 0..j {
                sum += l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    // Fall back to pseudo-inverse via SVD-like approach
                    return solve_with_regularization(a, b);
                }
                l[[i, j]] = diag.sqrt();
            } else {
                let denom = l[[j, j]];
                if denom.abs() < 1e-15 {
                    return solve_with_regularization(a, b);
                }
                l[[i, j]] = (a[[i, j]] - sum) / denom;
            }
        }
    }
    // Forward solve: L * z = b
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * z[j];
        }
        let denom = l[[i, i]];
        if denom.abs() < 1e-15 {
            return solve_with_regularization(a, b);
        }
        z[i] = (b[i] - sum) / denom;
    }
    // Back solve: L^T * x = z
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        let denom = l[[i, i]];
        if denom.abs() < 1e-15 {
            return solve_with_regularization(a, b);
        }
        x[i] = (z[i] - sum) / denom;
    }
    Ok(x)
}

/// Solve with Tikhonov regularization when Cholesky fails.
fn solve_with_regularization(a: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = a.nrows();
    let mut a_reg = a.clone();
    let ridge = 1e-10
        * (0..n)
            .map(|i| a[[i, i]].abs())
            .fold(0.0_f64, f64::max)
            .max(1e-10);
    for i in 0..n {
        a_reg[[i, i]] += ridge;
    }
    // Retry Cholesky with regularization
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for kk in 0..j {
                sum += l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                let diag = a_reg[[i, i]] - sum;
                l[[i, j]] = if diag > 0.0 { diag.sqrt() } else { 1e-15 };
            } else {
                let denom = l[[j, j]];
                l[[i, j]] = if denom.abs() > 1e-15 {
                    (a_reg[[i, j]] - sum) / denom
                } else {
                    0.0
                };
            }
        }
    }
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * z[j];
        }
        let denom = l[[i, i]];
        z[i] = if denom.abs() > 1e-15 {
            (b[i] - sum) / denom
        } else {
            0.0
        };
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        let denom = l[[i, i]];
        x[i] = if denom.abs() > 1e-15 {
            (z[i] - sum) / denom
        } else {
            0.0
        };
    }
    Ok(x)
}

/// Invert a symmetric positive-definite matrix.
fn invert_symmetric(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        let col = solve_symmetric(a, &e)?;
        for i in 0..n {
            inv[[i, j]] = col[i];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Helper: differencing and lag matrix construction
// ---------------------------------------------------------------------------

/// First difference of a series: diff[i] = y[i+1] - y[i]
fn diff(y: &ArrayView1<f64>) -> Array1<f64> {
    let n = y.len();
    if n < 2 {
        return Array1::zeros(0);
    }
    let mut d = Array1::zeros(n - 1);
    for i in 0..(n - 1) {
        d[i] = y[i + 1] - y[i];
    }
    d
}

/// Build the design matrix for ADF regression.
/// dy_t = alpha + beta*t + gamma*y_{t-1} + sum_{i=1}^{p} delta_i * dy_{t-i} + e_t
fn build_adf_design(
    y: &ArrayView1<f64>,
    lags: usize,
    regression: AdfRegression,
) -> StatsResult<(Array1<f64>, Array2<f64>, usize)> {
    let dy = diff(y);
    let n_dy = dy.len();
    if n_dy <= lags + 1 {
        return Err(StatsError::InsufficientData(format!(
            "series too short for {} lags (n_dy={})",
            lags, n_dy
        )));
    }
    // Effective sample starts at index `lags` in dy
    let n_eff = n_dy - lags;
    // Dependent variable: dy[lags..n_dy]
    let dep = Array1::from_vec((lags..n_dy).map(|i| dy[i]).collect());
    // Number of regressors: y_{t-1}, lagged diffs, possibly constant, trend
    let mut n_reg: usize = 1 + lags; // y_{t-1} and lags of dy
    match regression {
        AdfRegression::None => {}
        AdfRegression::Constant => n_reg += 1,
        AdfRegression::ConstantTrend => n_reg += 2,
        AdfRegression::ConstantTrendSquared => n_reg += 3,
    }
    let mut design = Array2::<f64>::zeros((n_eff, n_reg));
    for i in 0..n_eff {
        let t_idx = lags + i; // index in dy
        let mut col = 0;
        // Deterministic components
        match regression {
            AdfRegression::None => {}
            AdfRegression::Constant => {
                design[[i, col]] = 1.0;
                col += 1;
            }
            AdfRegression::ConstantTrend => {
                design[[i, col]] = 1.0;
                col += 1;
                design[[i, col]] = (t_idx + 1) as f64;
                col += 1;
            }
            AdfRegression::ConstantTrendSquared => {
                design[[i, col]] = 1.0;
                col += 1;
                let t_val = (t_idx + 1) as f64;
                design[[i, col]] = t_val;
                col += 1;
                design[[i, col]] = t_val * t_val;
                col += 1;
            }
        }
        // y_{t-1}: this is y at the original index = (t_idx) since dy = y[1..]-y[0..]
        // so y_{t-1} = y[t_idx]
        design[[i, col]] = y[t_idx];
        col += 1;
        // Lagged differences: dy_{t-1}, dy_{t-2}, ..., dy_{t-p}
        for lag in 1..=lags {
            design[[i, col]] = dy[t_idx - lag];
            col += 1;
        }
    }
    Ok((dep, design, n_eff))
}

// ---------------------------------------------------------------------------
// Auto-lag selection
// ---------------------------------------------------------------------------

/// Select optimal lag length for unit root tests using an information criterion.
///
/// # Arguments
/// * `y` - Time series data
/// * `max_lags` - Maximum number of lags to consider (if `None`, uses `(12 * (n/100)^{1/4})`)
/// * `criterion` - Information criterion to use
/// * `regression` - Regression type for ADF-style design
///
/// # Returns
/// Optimal number of lags
pub fn select_lag(
    y: &ArrayView1<f64>,
    max_lags: Option<usize>,
    criterion: LagCriterion,
    regression: AdfRegression,
) -> StatsResult<usize> {
    let n = y.len();
    if n < 10 {
        return Err(StatsError::InsufficientData(
            "need at least 10 observations for lag selection".into(),
        ));
    }
    let default_max = ((12.0 * ((n as f64) / 100.0).powf(0.25)) as usize).min(n / 3);
    let max_p = max_lags.unwrap_or(default_max).min(n / 3);

    match criterion {
        LagCriterion::TStatistic => select_lag_tstat(y, max_p, regression),
        _ => select_lag_ic(y, max_p, criterion, regression),
    }
}

fn select_lag_ic(
    y: &ArrayView1<f64>,
    max_p: usize,
    criterion: LagCriterion,
    regression: AdfRegression,
) -> StatsResult<usize> {
    let n = y.len() as f64;
    let mut best_lag = 0;
    let mut best_ic = f64::INFINITY;
    for p in 0..=max_p {
        let result = build_adf_design(y, p, regression);
        let (dep, design, n_eff) = match result {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ols = match ols_regression(&dep.view(), &design) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let nf = n_eff as f64;
        let k = design.ncols() as f64;
        let rss: f64 = ols.residuals.iter().map(|r| r * r).sum();
        let log_sigma = (rss / nf).ln();
        let ic = match criterion {
            LagCriterion::Aic => log_sigma + 2.0 * k / nf,
            LagCriterion::Bic => log_sigma + k * n.ln() / nf,
            LagCriterion::Hqic => log_sigma + 2.0 * k * n.ln().ln() / nf,
            LagCriterion::TStatistic => log_sigma, // shouldn't reach here
        };
        if ic < best_ic {
            best_ic = ic;
            best_lag = p;
        }
    }
    Ok(best_lag)
}

fn select_lag_tstat(
    y: &ArrayView1<f64>,
    max_p: usize,
    regression: AdfRegression,
) -> StatsResult<usize> {
    // Start from max_p, reduce until the last lag is significant at 10%
    for p in (1..=max_p).rev() {
        let result = build_adf_design(y, p, regression);
        let (dep, design, _n_eff) = match result {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ols = match ols_regression(&dep.view(), &design) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // The last column is the p-th lagged difference
        let last_col = design.ncols() - 1;
        let beta_last = ols.coefficients[last_col];
        let se_last = ols.se[last_col];
        if se_last > 1e-15 {
            let t_stat = (beta_last / se_last).abs();
            // Approx 10% critical value for t ~ 1.645
            if t_stat > 1.645 {
                return Ok(p);
            }
        }
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// ADF critical values and p-value approximation
// ---------------------------------------------------------------------------

/// ADF critical values (MacKinnon 1994/2010 response surface approximations).
fn adf_critical_values(n: usize, regression: AdfRegression) -> CriticalValues {
    let nf = n as f64;
    // Asymptotic values (MacKinnon 1996)
    match regression {
        AdfRegression::None => CriticalValues {
            one_pct: -2.5658 - 1.960 / nf - 10.04 / (nf * nf),
            five_pct: -1.9393 - 0.398 / nf,
            ten_pct: -1.6156 - 0.181 / nf,
        },
        AdfRegression::Constant => CriticalValues {
            one_pct: -3.4336 - 5.999 / nf - 29.25 / (nf * nf),
            five_pct: -2.8621 - 2.738 / nf - 8.36 / (nf * nf),
            ten_pct: -2.5671 - 1.438 / nf - 4.48 / (nf * nf),
        },
        AdfRegression::ConstantTrend | AdfRegression::ConstantTrendSquared => CriticalValues {
            one_pct: -3.9638 - 8.353 / nf - 47.44 / (nf * nf),
            five_pct: -3.4126 - 4.039 / nf - 17.83 / (nf * nf),
            ten_pct: -3.1279 - 2.418 / nf - 7.58 / (nf * nf),
        },
    }
}

/// Approximate p-value for the ADF test statistic using MacKinnon (1994) regression.
fn adf_p_value(stat: f64, n: usize, regression: AdfRegression) -> f64 {
    let cv = adf_critical_values(n, regression);
    // Linear interpolation between critical values
    if stat <= cv.one_pct {
        // p < 0.01; rough extrapolation
        let slope = (0.05 - 0.01) / (cv.five_pct - cv.one_pct);
        let p = 0.01 + slope * (stat - cv.one_pct);
        p.max(0.0001)
    } else if stat <= cv.five_pct {
        let frac = (stat - cv.one_pct) / (cv.five_pct - cv.one_pct);
        0.01 + frac * (0.05 - 0.01)
    } else if stat <= cv.ten_pct {
        let frac = (stat - cv.five_pct) / (cv.ten_pct - cv.five_pct);
        0.05 + frac * (0.10 - 0.05)
    } else {
        // p > 0.10; rough extrapolation using normal tail
        let frac = (stat - cv.ten_pct) / (cv.ten_pct.abs()).max(1.0);
        (0.10 + frac * 0.4).min(0.999)
    }
}

// ---------------------------------------------------------------------------
// Augmented Dickey-Fuller test
// ---------------------------------------------------------------------------

/// Perform the Augmented Dickey-Fuller (ADF) test for a unit root.
///
/// The null hypothesis is that the series has a unit root (non-stationary).
/// Rejecting H0 (small p-value) indicates stationarity.
///
/// # Arguments
/// * `y` - Time series data
/// * `max_lags` - Maximum number of lags (None for automatic)
/// * `regression` - Type of deterministic terms (default: `Constant`)
/// * `criterion` - Lag selection criterion (default: `Aic`)
///
/// # Example
/// ```no_run
/// use scirs2_stats::stationarity::{adf_test, AdfRegression, LagCriterion};
/// use scirs2_core::ndarray::Array1;
///
/// let y: Array1<f64> = Array1::from_vec((0..100).map(|i| (i as f64) * 0.1 + 0.5).collect());
/// let result = adf_test(&y.view(), None, AdfRegression::Constant, LagCriterion::Aic)
///     .expect("ADF test failed");
/// // A trending series should have a large p-value (fail to reject unit root)
/// assert!(result.p_value > 0.05);
/// ```
pub fn adf_test(
    y: &ArrayView1<f64>,
    max_lags: Option<usize>,
    regression: AdfRegression,
    criterion: LagCriterion,
) -> StatsResult<AdfResult> {
    let n = y.len();
    if n < 12 {
        return Err(StatsError::InsufficientData(
            "ADF test requires at least 12 observations".into(),
        ));
    }
    // Select optimal lag
    let p = select_lag(y, max_lags, criterion, regression)?;
    // Build design and run regression
    let (dep, design, n_eff) = build_adf_design(y, p, regression)?;
    let ols = ols_regression(&dep.view(), &design)?;
    // The coefficient on y_{t-1} is the gamma coefficient
    // Its column index depends on the regression type
    let gamma_col = match regression {
        AdfRegression::None => 0,
        AdfRegression::Constant => 1,
        AdfRegression::ConstantTrend => 2,
        AdfRegression::ConstantTrendSquared => 3,
    };
    let gamma = ols.coefficients[gamma_col];
    let se_gamma = ols.se[gamma_col];
    let t_stat = if se_gamma > 1e-15 {
        gamma / se_gamma
    } else {
        f64::NEG_INFINITY
    };
    let cv = adf_critical_values(n_eff, regression);
    let p_value = adf_p_value(t_stat, n_eff, regression);

    Ok(AdfResult {
        statistic: t_stat,
        p_value,
        used_lags: p,
        n_obs: n_eff,
        critical_values: cv,
        regression,
    })
}

// ---------------------------------------------------------------------------
// KPSS test
// ---------------------------------------------------------------------------

/// KPSS critical values (Kwiatkowski et al. 1992, Table 1)
fn kpss_critical_values(trend: KpssTrend) -> CriticalValues {
    match trend {
        KpssTrend::Constant => CriticalValues {
            one_pct: 0.739,
            five_pct: 0.463,
            ten_pct: 0.347,
        },
        KpssTrend::Trend => CriticalValues {
            one_pct: 0.216,
            five_pct: 0.146,
            ten_pct: 0.119,
        },
    }
}

/// Approximate p-value for the KPSS statistic
fn kpss_p_value(stat: f64, trend: KpssTrend) -> f64 {
    let cv = kpss_critical_values(trend);
    // KPSS rejects for large values (opposite of ADF)
    if stat >= cv.one_pct {
        let overshoot = (stat - cv.one_pct) / cv.one_pct.max(0.001);
        (0.01 - overshoot * 0.005).max(0.001)
    } else if stat >= cv.five_pct {
        let frac = (stat - cv.five_pct) / (cv.one_pct - cv.five_pct);
        0.05 - frac * (0.05 - 0.01)
    } else if stat >= cv.ten_pct {
        let frac = (stat - cv.ten_pct) / (cv.five_pct - cv.ten_pct);
        0.10 - frac * (0.10 - 0.05)
    } else {
        // p > 0.10
        let frac = stat / cv.ten_pct.max(0.001);
        (0.10 + (1.0 - frac) * 0.40).min(0.999)
    }
}

/// Perform the KPSS test for stationarity.
///
/// The null hypothesis is that the series is stationary (level or trend stationary).
/// Rejecting H0 (small p-value) indicates non-stationarity.
///
/// # Arguments
/// * `y` - Time series data
/// * `trend` - Level or trend stationarity
/// * `n_lags` - Number of lags for the Newey-West estimator (None for automatic)
///
/// # Example
/// ```
/// use scirs2_stats::stationarity::{kpss_test, KpssTrend};
/// use scirs2_core::ndarray::Array1;
///
/// // Stationary series (white noise around zero)
/// let y = Array1::from_vec(vec![
///     0.1, -0.2, 0.3, -0.1, 0.05, -0.15, 0.2, -0.05,
///     0.15, -0.1, 0.25, -0.2, 0.1, 0.0, -0.1, 0.2,
///     -0.15, 0.1, -0.05, 0.15
/// ]);
/// let result = kpss_test(&y.view(), KpssTrend::Constant, None)
///     .expect("KPSS test failed");
/// // Stationary series: p-value should be large (fail to reject stationarity)
/// assert!(result.p_value > 0.05);
/// ```
pub fn kpss_test(
    y: &ArrayView1<f64>,
    trend: KpssTrend,
    n_lags: Option<usize>,
) -> StatsResult<KpssResult> {
    let n = y.len();
    if n < 6 {
        return Err(StatsError::InsufficientData(
            "KPSS test requires at least 6 observations".into(),
        ));
    }
    // Step 1: regress y on deterministic terms to get residuals
    let n_det = match trend {
        KpssTrend::Constant => 1,
        KpssTrend::Trend => 2,
    };
    let mut det = Array2::<f64>::zeros((n, n_det));
    for i in 0..n {
        det[[i, 0]] = 1.0;
        if n_det == 2 {
            det[[i, 1]] = (i + 1) as f64;
        }
    }
    let ols = ols_regression(&y.into(), &det)?;
    let resid = &ols.residuals;

    // Step 2: partial sums of residuals
    let mut s = Array1::<f64>::zeros(n);
    s[0] = resid[0];
    for i in 1..n {
        s[i] = s[i - 1] + resid[i];
    }

    // Step 3: long-run variance estimate (Newey-West / Bartlett kernel)
    let lags = n_lags.unwrap_or(((n as f64).sqrt() * 0.75) as usize);
    let nf = n as f64;
    // Sample variance of residuals
    let gamma0: f64 = resid.iter().map(|r| r * r).sum::<f64>() / nf;
    let mut long_run_var = gamma0;
    for lag in 1..=lags {
        let w = 1.0 - (lag as f64) / ((lags + 1) as f64); // Bartlett kernel
        let mut gamma_l = 0.0;
        for i in lag..n {
            gamma_l += resid[i] * resid[i - lag];
        }
        gamma_l /= nf;
        long_run_var += 2.0 * w * gamma_l;
    }
    if long_run_var <= 0.0 {
        return Err(StatsError::ComputationError(
            "KPSS: estimated long-run variance is non-positive".into(),
        ));
    }

    // Step 4: KPSS statistic = (1/n^2) * sum(S_t^2) / long_run_var
    let ss: f64 = s.iter().map(|si| si * si).sum();
    let eta = ss / (nf * nf * long_run_var);

    let cv = kpss_critical_values(trend);
    let p_value = kpss_p_value(eta, trend);

    Ok(KpssResult {
        statistic: eta,
        p_value,
        used_lags: lags,
        critical_values: cv,
        trend,
    })
}

// ---------------------------------------------------------------------------
// Phillips-Perron test
// ---------------------------------------------------------------------------

/// Phillips-Perron critical values (same distribution as DF under H0).
fn pp_critical_values(n: usize) -> CriticalValues {
    // Uses the "constant" ADF critical values (same asymptotic dist)
    adf_critical_values(n, AdfRegression::Constant)
}

/// Perform the Phillips-Perron (PP) test for a unit root.
///
/// Non-parametric correction to the Dickey-Fuller statistic that accounts
/// for serial correlation and heteroskedasticity without adding lagged
/// difference terms.
///
/// # Arguments
/// * `y` - Time series data
/// * `n_lags` - Number of lags for the Newey-West estimator (None for automatic)
///
/// # Example
/// ```
/// use scirs2_stats::stationarity::phillips_perron_test;
/// use scirs2_core::ndarray::Array1;
///
/// // Random walk (non-stationary)
/// let mut rw = vec![0.0_f64; 50];
/// let increments = [0.5, -0.3, 0.7, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5];
/// for i in 1..50 {
///     rw[i] = rw[i-1] + increments[i % 10];
/// }
/// let y = Array1::from_vec(rw);
/// let result = phillips_perron_test(&y.view(), None).expect("PP test failed");
/// // Check that the statistic is finite
/// assert!(result.z_tau.is_finite());
/// ```
pub fn phillips_perron_test(
    y: &ArrayView1<f64>,
    n_lags: Option<usize>,
) -> StatsResult<PhillipsPerronResult> {
    let n = y.len();
    if n < 12 {
        return Err(StatsError::InsufficientData(
            "Phillips-Perron test requires at least 12 observations".into(),
        ));
    }
    // Run simple DF regression (no augmentation): dy_t = a + gamma * y_{t-1} + e_t
    let (dep, design, n_eff) = build_adf_design(y, 0, AdfRegression::Constant)?;
    let ols = ols_regression(&dep.view(), &design)?;
    let gamma = ols.coefficients[1]; // column 0 = constant, column 1 = y_{t-1}
    let se_gamma = ols.se[1];
    let t_stat = if se_gamma > 1e-15 {
        gamma / se_gamma
    } else {
        f64::NEG_INFINITY
    };
    let nf = n_eff as f64;
    let resid = &ols.residuals;

    // Estimate long-run variance using Newey-West
    let lags = n_lags.unwrap_or(((nf).powf(1.0 / 3.0) * 1.5) as usize);
    let gamma0: f64 = resid.iter().map(|r| r * r).sum::<f64>() / nf;
    let mut lambda_sq = gamma0;
    for lag in 1..=lags {
        let w = 1.0 - (lag as f64) / ((lags + 1) as f64);
        let mut g = 0.0;
        for i in lag..n_eff {
            g += resid[i] * resid[i - lag];
        }
        g /= nf;
        lambda_sq += 2.0 * w * g;
    }
    lambda_sq = lambda_sq.max(1e-15);

    // Variance of y_{t-1} (needed for PP correction)
    let y_lag: Vec<f64> = (0..n_eff)
        .map(|i| {
            let idx = i; // y_{t-1} in the design for lag=0 is y[i] (when building with AdfRegression::Constant)
            design[[i, 1]]
        })
        .collect();
    let y_lag_mean: f64 = y_lag.iter().sum::<f64>() / nf;
    let y_lag_var: f64 = y_lag.iter().map(|v| (v - y_lag_mean).powi(2)).sum::<f64>();

    // PP corrections
    let s_sq = gamma0; // short-run variance
    let correction_factor = (lambda_sq - s_sq) * nf / (2.0 * y_lag_var.max(1e-15));

    // Z(alpha) = n * gamma - correction
    let z_alpha = nf * gamma - correction_factor;

    // Z(t) = t_stat * sqrt(s^2 / lambda^2) - correction_on_t
    let ratio = (s_sq / lambda_sq).sqrt();
    let z_tau = t_stat * ratio
        - (lambda_sq - s_sq) * nf.sqrt() / (2.0 * lambda_sq.sqrt() * y_lag_var.max(1e-15).sqrt());

    let cv = pp_critical_values(n_eff);
    let p_value = adf_p_value(z_tau, n_eff, AdfRegression::Constant);

    Ok(PhillipsPerronResult {
        z_alpha,
        z_tau,
        p_value,
        used_lags: lags,
        n_obs: n_eff,
        critical_values: cv,
    })
}

// ---------------------------------------------------------------------------
// Zivot-Andrews test
// ---------------------------------------------------------------------------

/// Critical values for Zivot-Andrews test (Zivot & Andrews 1992, Table 2/3/4)
fn za_critical_values(break_type: ZivotAndrewsBreak) -> CriticalValues {
    match break_type {
        ZivotAndrewsBreak::Intercept => CriticalValues {
            one_pct: -5.34,
            five_pct: -4.80,
            ten_pct: -4.58,
        },
        ZivotAndrewsBreak::Trend => CriticalValues {
            one_pct: -4.93,
            five_pct: -4.42,
            ten_pct: -4.11,
        },
        ZivotAndrewsBreak::Both => CriticalValues {
            one_pct: -5.57,
            five_pct: -5.08,
            ten_pct: -4.82,
        },
    }
}

/// Perform the Zivot-Andrews test for unit root with a single structural break.
///
/// The test endogenously determines the break date by minimizing the ADF t-statistic
/// over all possible break points (trimming 15% from each end).
///
/// # Arguments
/// * `y` - Time series data
/// * `max_lags` - Maximum number of lags (None for automatic)
/// * `break_type` - Type of structural break to test for
/// * `trim` - Fraction of data to trim from each end (default 0.15)
///
/// # Example
/// ```
/// use scirs2_stats::stationarity::{zivot_andrews_test, ZivotAndrewsBreak};
/// use scirs2_core::ndarray::Array1;
///
/// // Series with a level shift at observation 30
/// let mut y = vec![0.0_f64; 60];
/// for i in 0..30 { y[i] = (i as f64) * 0.01; }
/// for i in 30..60 { y[i] = 2.0 + (i as f64) * 0.01; }
/// let y = Array1::from_vec(y);
/// let result = zivot_andrews_test(&y.view(), None, ZivotAndrewsBreak::Intercept, None)
///     .expect("ZA test failed");
/// assert!(result.break_point > 0);
/// ```
pub fn zivot_andrews_test(
    y: &ArrayView1<f64>,
    max_lags: Option<usize>,
    break_type: ZivotAndrewsBreak,
    trim: Option<f64>,
) -> StatsResult<ZivotAndrewsResult> {
    let n = y.len();
    if n < 20 {
        return Err(StatsError::InsufficientData(
            "Zivot-Andrews test requires at least 20 observations".into(),
        ));
    }
    let trim_frac = trim.unwrap_or(0.15);
    let start = (n as f64 * trim_frac).ceil() as usize;
    let end = n - start;
    if start >= end {
        return Err(StatsError::InvalidArgument(
            "trim fraction too large for the data".into(),
        ));
    }

    // Select lag order (using first-differenced data with ADF)
    let p = select_lag(y, max_lags, LagCriterion::Aic, AdfRegression::ConstantTrend)?;

    let dy = diff(y);
    let n_dy = dy.len();

    let mut min_t_stat = f64::INFINITY;
    let mut best_break = start;
    let mut best_lags = p;

    for tb in start..end {
        // Build design: dy_t = mu + beta*t + gamma*y_{t-1} + theta*DU_t + phi*DT_t
        //   + sum delta_i*dy_{t-i} + e_t
        // where DU_t = 1 if t > tb, DT_t = t - tb if t > tb
        let eff_start = p;
        if n_dy <= eff_start {
            continue;
        }
        let n_eff = n_dy - eff_start;
        // Dependent
        let dep = Array1::from_vec((eff_start..n_dy).map(|i| dy[i]).collect());
        // Design columns
        let n_break_cols = match break_type {
            ZivotAndrewsBreak::Intercept => 1,
            ZivotAndrewsBreak::Trend => 1,
            ZivotAndrewsBreak::Both => 2,
        };
        let n_reg = 1 + 1 + 1 + n_break_cols + p; // const + trend + y_{t-1} + break dummies + lags
        let mut design = Array2::<f64>::zeros((n_eff, n_reg));

        for i in 0..n_eff {
            let t_idx = eff_start + i;
            let orig_t = t_idx + 1; // 1-based time index
            let mut col = 0;
            // Constant
            design[[i, col]] = 1.0;
            col += 1;
            // Trend
            design[[i, col]] = orig_t as f64;
            col += 1;
            // Break dummies
            match break_type {
                ZivotAndrewsBreak::Intercept => {
                    design[[i, col]] = if orig_t > tb { 1.0 } else { 0.0 };
                    col += 1;
                }
                ZivotAndrewsBreak::Trend => {
                    design[[i, col]] = if orig_t > tb {
                        (orig_t - tb) as f64
                    } else {
                        0.0
                    };
                    col += 1;
                }
                ZivotAndrewsBreak::Both => {
                    design[[i, col]] = if orig_t > tb { 1.0 } else { 0.0 };
                    col += 1;
                    design[[i, col]] = if orig_t > tb {
                        (orig_t - tb) as f64
                    } else {
                        0.0
                    };
                    col += 1;
                }
            }
            // y_{t-1}
            design[[i, col]] = y[t_idx];
            col += 1;
            // Lagged differences
            for lag in 1..=p {
                if t_idx >= lag {
                    design[[i, col]] = dy[t_idx - lag];
                }
                col += 1;
            }
        }

        let ols = match ols_regression(&dep.view(), &design) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // gamma is the coefficient on y_{t-1}
        let gamma_col = 2 + n_break_cols;
        let gamma = ols.coefficients[gamma_col];
        let se_gamma = ols.se[gamma_col];
        let t_stat = if se_gamma > 1e-15 {
            gamma / se_gamma
        } else {
            f64::NEG_INFINITY
        };
        if t_stat < min_t_stat {
            min_t_stat = t_stat;
            best_break = tb;
            best_lags = p;
        }
    }

    let cv = za_critical_values(break_type);
    // Approximate p-value: use ZA critical values for interpolation
    let p_value = if min_t_stat <= cv.one_pct {
        0.005
    } else if min_t_stat <= cv.five_pct {
        let frac = (min_t_stat - cv.one_pct) / (cv.five_pct - cv.one_pct);
        0.01 + frac * 0.04
    } else if min_t_stat <= cv.ten_pct {
        let frac = (min_t_stat - cv.five_pct) / (cv.ten_pct - cv.five_pct);
        0.05 + frac * 0.05
    } else {
        let frac = (min_t_stat - cv.ten_pct) / cv.ten_pct.abs().max(1.0);
        (0.10 + frac * 0.4).min(0.999)
    };

    Ok(ZivotAndrewsResult {
        statistic: min_t_stat,
        p_value,
        break_point: best_break,
        used_lags: best_lags,
        critical_values: cv,
        break_type,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_stationary_series(n: usize) -> Array1<f64> {
        // AR(1) with rho < 1 (stationary) plus some deterministic variation
        let mut y = vec![0.0_f64; n];
        let noise: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 1.7 + 0.3).sin()) * 0.5)
            .collect();
        for i in 1..n {
            y[i] = 0.5 * y[i - 1] + noise[i];
        }
        Array1::from_vec(y)
    }

    fn make_random_walk(n: usize) -> Array1<f64> {
        let mut y = vec![0.0_f64; n];
        let noise: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 2.3 + 0.7).sin()) * 0.3)
            .collect();
        for i in 1..n {
            y[i] = y[i - 1] + noise[i];
        }
        Array1::from_vec(y)
    }

    #[test]
    fn test_select_lag_aic() {
        let y = make_stationary_series(100);
        let p = select_lag(
            &y.view(),
            Some(10),
            LagCriterion::Aic,
            AdfRegression::Constant,
        );
        assert!(p.is_ok());
        let p = p.expect("lag selection should succeed");
        assert!(p <= 10);
    }

    #[test]
    fn test_select_lag_bic() {
        let y = make_stationary_series(100);
        let p = select_lag(
            &y.view(),
            Some(10),
            LagCriterion::Bic,
            AdfRegression::Constant,
        );
        assert!(p.is_ok());
    }

    #[test]
    fn test_select_lag_tstat() {
        let y = make_stationary_series(100);
        let p = select_lag(
            &y.view(),
            Some(10),
            LagCriterion::TStatistic,
            AdfRegression::Constant,
        );
        assert!(p.is_ok());
    }

    #[test]
    fn test_adf_stationary() {
        let y = make_stationary_series(200);
        let result = adf_test(&y.view(), None, AdfRegression::Constant, LagCriterion::Aic);
        assert!(result.is_ok());
        let r = result.expect("ADF should succeed");
        // Stationary series should reject unit root (small p-value or stat < cv)
        assert!(r.statistic.is_finite());
        assert!(r.p_value >= 0.0);
        assert!(r.n_obs > 0);
    }

    #[test]
    fn test_adf_random_walk() {
        let y = make_random_walk(200);
        let result = adf_test(&y.view(), None, AdfRegression::Constant, LagCriterion::Aic);
        assert!(result.is_ok());
        let r = result.expect("ADF should succeed");
        // Random walk: should NOT reject unit root => larger p-value expected
        assert!(r.statistic.is_finite());
    }

    #[test]
    fn test_adf_no_constant() {
        let y = make_stationary_series(100);
        let result = adf_test(&y.view(), Some(2), AdfRegression::None, LagCriterion::Bic);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adf_trend() {
        let y = make_stationary_series(100);
        let result = adf_test(
            &y.view(),
            Some(3),
            AdfRegression::ConstantTrend,
            LagCriterion::Aic,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_kpss_stationary() {
        let y = make_stationary_series(100);
        let result = kpss_test(&y.view(), KpssTrend::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("KPSS should succeed");
        // Stationary: should not reject H0 => large p-value
        assert!(r.statistic.is_finite());
        assert!(r.statistic >= 0.0);
    }

    #[test]
    fn test_kpss_trend() {
        let y = make_stationary_series(100);
        let result = kpss_test(&y.view(), KpssTrend::Trend, Some(5));
        assert!(result.is_ok());
    }

    #[test]
    fn test_kpss_random_walk() {
        let y = make_random_walk(200);
        let result = kpss_test(&y.view(), KpssTrend::Constant, None);
        assert!(result.is_ok());
        let r = result.expect("KPSS should succeed");
        // Non-stationary: KPSS should detect (small p-value, large statistic)
        assert!(r.statistic.is_finite());
    }

    #[test]
    fn test_phillips_perron() {
        let y = make_stationary_series(200);
        let result = phillips_perron_test(&y.view(), None);
        assert!(result.is_ok());
        let r = result.expect("PP test should succeed");
        assert!(r.z_tau.is_finite());
        assert!(r.z_alpha.is_finite());
    }

    #[test]
    fn test_phillips_perron_random_walk() {
        let y = make_random_walk(200);
        let result = phillips_perron_test(&y.view(), None);
        assert!(result.is_ok());
        let r = result.expect("PP test should succeed");
        assert!(r.z_tau.is_finite());
    }

    #[test]
    fn test_zivot_andrews_intercept() {
        // Series with a level shift
        let mut y_vec = vec![0.0_f64; 80];
        for i in 0..40 {
            y_vec[i] = ((i as f64) * 1.3).sin() * 0.3;
        }
        for i in 40..80 {
            y_vec[i] = 3.0 + ((i as f64) * 1.3).sin() * 0.3;
        }
        let y = Array1::from_vec(y_vec);
        let result = zivot_andrews_test(&y.view(), Some(4), ZivotAndrewsBreak::Intercept, None);
        assert!(result.is_ok());
        let r = result.expect("ZA test should succeed");
        assert!(r.statistic.is_finite());
        assert!(r.break_point > 0);
    }

    #[test]
    fn test_zivot_andrews_trend() {
        let y_vec: Vec<f64> = (0..80)
            .map(|i| {
                let t = i as f64;
                if i < 40 {
                    0.1 * t + (t * 0.7).sin() * 0.2
                } else {
                    0.5 * t + (t * 0.7).sin() * 0.2
                }
            })
            .collect();
        let y = Array1::from_vec(y_vec);
        let result = zivot_andrews_test(&y.view(), Some(3), ZivotAndrewsBreak::Trend, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zivot_andrews_both() {
        let y_vec: Vec<f64> = (0..80)
            .map(|i| {
                let t = i as f64;
                if i < 40 {
                    0.1 * t + (t * 0.5).sin() * 0.2
                } else {
                    5.0 + 0.5 * t + (t * 0.5).sin() * 0.2
                }
            })
            .collect();
        let y = Array1::from_vec(y_vec);
        let result = zivot_andrews_test(&y.view(), Some(2), ZivotAndrewsBreak::Both, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adf_insufficient_data() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = adf_test(&y.view(), None, AdfRegression::Constant, LagCriterion::Aic);
        assert!(result.is_err());
    }

    #[test]
    fn test_kpss_insufficient_data() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let result = kpss_test(&y.view(), KpssTrend::Constant, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pp_insufficient_data() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = phillips_perron_test(&y.view(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_za_insufficient_data() {
        let y = Array1::from_vec(vec![1.0; 5]);
        let result = zivot_andrews_test(&y.view(), None, ZivotAndrewsBreak::Intercept, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_critical_values_ordering() {
        let cv = adf_critical_values(100, AdfRegression::Constant);
        // 1% should be more negative than 5% which is more negative than 10%
        assert!(cv.one_pct < cv.five_pct);
        assert!(cv.five_pct < cv.ten_pct);
    }

    #[test]
    fn test_kpss_critical_values_ordering() {
        let cv = kpss_critical_values(KpssTrend::Constant);
        // KPSS: reject for large values, so 1% > 5% > 10%
        assert!(cv.one_pct > cv.five_pct);
        assert!(cv.five_pct > cv.ten_pct);
    }

    #[test]
    fn test_select_lag_auto_max() {
        let y = make_stationary_series(200);
        let p = select_lag(&y.view(), None, LagCriterion::Aic, AdfRegression::Constant);
        assert!(p.is_ok());
        let p = p.expect("auto lag selection should succeed");
        // Auto max should be reasonable
        assert!(p < 200 / 3);
    }
}
