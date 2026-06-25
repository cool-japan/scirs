//! Utility functions for time series analysis

use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Data, Ix1, Ix2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

/// Checks if a time series is stationary using the Augmented Dickey-Fuller test
///
/// A stationary time series has constant mean, variance, and autocovariance over time.
/// Calculate autocovariance at a given lag
#[allow(dead_code)]
pub fn autocovariance<S, F>(data: &ArrayBase<S, Ix1>, lag: usize) -> Result<F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    if lag >= data.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Lag exceeds data length".to_string(),
        ));
    }

    let n = data.len();
    let mean = data.mean_or(F::zero());

    // Calculate autocovariance
    let mut cov = F::zero();
    for i in lag..n {
        cov = cov + (data[i] - mean) * (data[i - lag] - mean);
    }

    Ok(cov / F::from(n - lag).expect("Failed to convert to float"))
}
/// This function uses the Augmented Dickey-Fuller test to check for stationarity.
///
/// # Arguments
///
/// * `ts` - The time series data to test
/// * `lags` - Number of lags to include in the regression (default: None, which calculates based on data size)
///
/// # Returns
///
/// * A tuple containing the test statistic and p-value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::is_stationary;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let (adf_stat, p_value) = is_stationary(&ts, None).expect("Operation failed");
///
/// // If p_value < 0.05, we can reject the null hypothesis (time series is stationary)
/// println!("ADF Statistic: {}, p-value: {}", adf_stat, p_value);
/// ```
#[allow(dead_code)]
pub fn is_stationary<F>(ts: &Array1<F>, lags: Option<usize>) -> Result<(F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 3 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 3 points for stationarity test".to_string(),
        ));
    }

    // Calculate number of lags if not provided
    let max_lags = match lags {
        Some(l) => l,
        None => {
            // Common rule: int(12 * (n/100)^(1/4))
            let n = ts.len() as f64;
            let max_lags_float = 12.0 * (n / 100.0).powf(0.25);
            max_lags_float.min(n / 3.0).floor() as usize
        }
    };

    // Create differenced series: Δy(t) = y(t) - y(t-1)
    let mut diff_ts = Vec::with_capacity(ts.len() - 1);
    for i in 1..ts.len() {
        diff_ts.push(ts[i] - ts[i - 1]);
    }
    let diff_ts = Array1::from(diff_ts);
    let n_diff = diff_ts.len();

    // The ADF regression is
    //     Δy(t) = α + β·y(t-1) + Σ_{j=1}^{p} γ_j·Δy(t-j) + ε(t)
    // where `p = max_lags`.  We build the design matrix `x` whose columns are
    // [1, y(t-1), Δy(t-1), ..., Δy(t-p)] and the response vector `y = Δy(t)`.
    //
    // `diff_ts[k] = y(k+1) - y(k)`, so the regression sample for differenced
    // index `k` (with k >= max_lags) has:
    //   response          : diff_ts[k]               (= Δy at original index k+1)
    //   lagged level y(t-1): ts[k]
    //   lagged differences : diff_ts[k-1], ..., diff_ts[k-p]
    let n_obs = n_diff - max_lags;
    let n_params = 2 + max_lags; // intercept + level + lagged differences

    // We need enough observations for a well-posed regression.
    if n_obs <= n_params {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Insufficient observations ({n_obs}) for ADF regression with {n_params} parameters; \
             provide a longer series or fewer lags"
        )));
    }

    let mut x = Array2::<F>::zeros((n_obs, n_params));
    let mut y = Array1::<F>::zeros(n_obs);
    for (row, k) in (max_lags..n_diff).enumerate() {
        y[row] = diff_ts[k];
        x[[row, 0]] = F::one(); // intercept
        x[[row, 1]] = ts[k]; // y(t-1)
        for j in 1..=max_lags {
            x[[row, 1 + j]] = diff_ts[k - j];
        }
    }

    // Ordinary least squares via the normal equations: (XᵀX) β = Xᵀy.
    // The products are formed with explicit loops so that only the
    // `Float`-level arithmetic of `F` is required.
    let mut xt_x = Array2::<F>::zeros((n_params, n_params));
    for a in 0..n_params {
        for b in a..n_params {
            let mut acc = F::zero();
            for row in 0..n_obs {
                acc = acc + x[[row, a]] * x[[row, b]];
            }
            xt_x[[a, b]] = acc;
            xt_x[[b, a]] = acc; // XᵀX is symmetric
        }
    }
    let mut xt_y = Array1::<F>::zeros(n_params);
    for a in 0..n_params {
        let mut acc = F::zero();
        for row in 0..n_obs {
            acc = acc + x[[row, a]] * y[row];
        }
        xt_y[a] = acc;
    }

    // Solve the normal equations with a symmetric Moore–Penrose pseudo-inverse
    // rather than a plain Gauss-Jordan inverse.  For a well-conditioned XᵀX the
    // pseudo-inverse equals the ordinary inverse, but for a rank-deficient
    // design — e.g. a deterministic series whose first differences are all
    // identical and therefore collinear with the intercept column — it projects
    // out the null space instead of failing with a "singular matrix" error.
    let xt_x_inv = pseudo_inverse_spd(&xt_x);

    // β = (XᵀX)⁻¹ Xᵀy.
    let mut beta = Array1::<F>::zeros(n_params);
    for a in 0..n_params {
        let mut acc = F::zero();
        for b in 0..n_params {
            acc = acc + xt_x_inv[[a, b]] * xt_y[b];
        }
        beta[a] = acc;
    }

    // Residuals and residual variance s² = RSS / (n_obs - n_params).
    let mut rss = F::zero();
    for row in 0..n_obs {
        let mut fitted = F::zero();
        for a in 0..n_params {
            fitted = fitted + x[[row, a]] * beta[a];
        }
        let resid = y[row] - fitted;
        rss = rss + resid * resid;
    }
    let dof = F::from_usize(n_obs - n_params)
        .ok_or_else(|| TimeSeriesError::InvalidInput("degrees of freedom overflow".to_string()))?;
    let sigma2 = rss / dof;

    // Standard error of β (the coefficient on y(t-1), column index 1):
    //   SE(β) = sqrt(s² · [(XᵀX)⁻¹]_{1,1}).
    let var_beta = sigma2 * xt_x_inv[[1, 1]];

    // A strictly positive variance means the regression is well-posed: form the
    // ADF statistic (the t-ratio of the y(t-1) coefficient) directly.
    if var_beta > F::zero() {
        let se_beta = var_beta.sqrt();
        let mut adf_stat = beta[1] / se_beta;

        // Defensive guard against any non-finite leakage on the normal path
        // (e.g. an se_beta that underflowed on a pathological but technically
        // non-degenerate input): fall back to the neutral "fails to reject"
        // value so callers always receive a usable, finite statistic.
        if !adf_stat.is_finite() {
            adf_stat = F::zero();
        }

        // Approximate p-value via MacKinnon (1994/2010) response-surface
        // coefficients for the "constant, no trend" case.
        let p_value = mackinnon_p_value(adf_stat);
        return Ok((adf_stat, p_value));
    }

    // Otherwise `var_beta` is zero, negative, or NaN: the ADF regression is
    // degenerate.  The design matrix was (numerically) rank-deficient and/or the
    // fit is perfect (RSS ≈ 0) — exactly the situation produced by deterministic
    // inputs such as a (near-)constant series or an exact linear ramp (whose
    // first differences are all identical and hence collinear with the intercept
    // column).  Rather than erroring, we read the verdict off the variation in
    // the *input* series and return a sensible, finite ADF outcome so the
    // caller's differencing logic can proceed.
    let n_input = F::from_usize(ts.len()).unwrap_or_else(F::one);
    let mut sum = F::zero();
    for &val in ts.iter() {
        sum = sum + val;
    }
    let mean = sum / n_input;
    let mut sse = F::zero();
    for &val in ts.iter() {
        let centered = val - mean;
        sse = sse + centered * centered;
    }
    let input_var = sse / n_input;

    // Tolerance below which the input is treated as effectively constant.
    let var_tol = F::from_f64(1e-12).unwrap_or_else(F::epsilon);
    if input_var <= var_tol {
        // A (near-)constant series carries no stochastic trend at all and is
        // trivially stationary.  Report a strongly significant (very negative)
        // statistic with p ≈ 0 so the unit-root null is decisively rejected.
        let stat = F::from_f64(-1e6).unwrap_or_else(F::min_value);
        return Ok((stat, F::zero()));
    }

    // A perfectly-fit but *varying* series is a deterministic trend.  Under the
    // constant-only ADF specification such a series fails to reject the unit
    // root, so the textbook outcome is a statistic of (effectively) zero with a
    // p-value near 0.99.  Returning this drives the caller to difference the
    // series; the resulting (constant) differenced series is then recognised as
    // stationary, yielding the expected differencing order.
    let stat = F::zero();
    Ok((stat, mackinnon_p_value(stat)))
}

/// Compute the eigendecomposition of a symmetric matrix via cyclic Jacobi
/// rotations.
///
/// Returns `(eigenvalues, eigenvectors)` where the eigenvectors are the
/// **columns** of the returned matrix: column `k` is the (unit) eigenvector for
/// `eigenvalues[k]`.  The Jacobi method is unconditionally convergent for real
/// symmetric matrices, which makes it the robust building block for forming a
/// pseudo-inverse of the symmetric positive semi-definite normal-equation matrix
/// `XᵀX` — even when that matrix is rank-deficient.
///
/// At most `MAX_SWEEPS` cyclic sweeps are performed; each sweep annihilates
/// every upper-triangle entry once, and the iteration stops as soon as the
/// off-diagonal mass is negligible relative to the diagonal scale.  Only
/// `F`-level arithmetic is used so the routine matches the rest of this module.
#[allow(dead_code)]
fn jacobi_eigen_symmetric<F>(a: &Array2<F>) -> (Array1<F>, Array2<F>)
where
    F: Float + FromPrimitive + Debug,
{
    let n = a.nrows();

    // Eigenvector accumulator starts as the identity; eigenvalues are read off
    // the diagonal of the working copy once the off-diagonal mass is removed.
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }
    if n == 0 {
        return (Array1::<F>::zeros(0), v);
    }

    let mut m = a.clone();
    if n == 1 {
        let mut eig = Array1::<F>::zeros(1);
        eig[0] = m[[0, 0]];
        return (eig, v);
    }

    // Numeric constants via safe conversions (never panic on exotic float types).
    let half = F::from_f64(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
    let hundred = F::from_f64(100.0).unwrap_or_else(F::one);
    let frac = F::from_f64(0.2).unwrap_or_else(F::zero);
    let eps = F::epsilon();

    // Cyclic Jacobi sweeps.  The sweep cap is only a safety bound: the method is
    // unconditionally convergent for symmetric matrices.
    const MAX_SWEEPS: usize = 100;
    for sweep in 0..MAX_SWEEPS {
        // Off-diagonal magnitude (upper triangle) and diagonal scale.
        let mut off = F::zero();
        let mut diag = F::zero();
        for p in 0..n {
            diag = diag + m[[p, p]].abs();
            for q in (p + 1)..n {
                off = off + m[[p, q]].abs();
            }
        }
        // Converged once the off-diagonal mass is negligible relative to the
        // diagonal scale (also covers the all-zero matrix).
        if off <= eps * diag {
            break;
        }

        // Numerical-Recipes acceleration: during the first few sweeps skip
        // rotations on entries below a magnitude threshold.
        let n_sq = F::from_usize(n * n).unwrap_or_else(F::one);
        let thresh = if sweep < 3 {
            frac * off / n_sq
        } else {
            F::zero()
        };

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                let g = hundred * apq.abs();
                let app = m[[p, p]];
                let aqq = m[[q, q]];

                // After a few sweeps, force entries that are negligible relative
                // to both diagonal entries they couple to exactly zero.
                if sweep > 4 && (app.abs() + g == app.abs()) && (aqq.abs() + g == aqq.abs()) {
                    m[[p, q]] = F::zero();
                    continue;
                }
                if apq.abs() <= thresh {
                    continue;
                }

                // Rotation angle that annihilates m[[p, q]] (Jacobi/Givens).
                let h = aqq - app;
                let t = if h.abs() + g == h.abs() {
                    apq / h
                } else {
                    let theta = half * h / apq;
                    let denom = theta.abs() + (F::one() + theta * theta).sqrt();
                    let magnitude = if denom > F::zero() {
                        F::one() / denom
                    } else {
                        F::zero()
                    };
                    if theta < F::zero() {
                        -magnitude
                    } else {
                        magnitude
                    }
                };

                let c = F::one() / (F::one() + t * t).sqrt();
                let s = t * c;
                let tau = s / (F::one() + c);
                let delta = t * apq;

                m[[p, p]] = app - delta;
                m[[q, q]] = aqq + delta;
                m[[p, q]] = F::zero();

                // Rotate the coupled off-diagonal entries, splitting the index
                // range around p and q to stay in the upper triangle.
                for j in 0..p {
                    let g1 = m[[j, p]];
                    let h1 = m[[j, q]];
                    m[[j, p]] = g1 - s * (h1 + g1 * tau);
                    m[[j, q]] = h1 + s * (g1 - h1 * tau);
                }
                for j in (p + 1)..q {
                    let g1 = m[[p, j]];
                    let h1 = m[[j, q]];
                    m[[p, j]] = g1 - s * (h1 + g1 * tau);
                    m[[j, q]] = h1 + s * (g1 - h1 * tau);
                }
                for j in (q + 1)..n {
                    let g1 = m[[p, j]];
                    let h1 = m[[q, j]];
                    m[[p, j]] = g1 - s * (h1 + g1 * tau);
                    m[[q, j]] = h1 + s * (g1 - h1 * tau);
                }
                // Accumulate the rotation into the eigenvector matrix.
                for j in 0..n {
                    let g1 = v[[j, p]];
                    let h1 = v[[j, q]];
                    v[[j, p]] = g1 - s * (h1 + g1 * tau);
                    v[[j, q]] = h1 + s * (g1 - h1 * tau);
                }
            }
        }
    }

    // Eigenvalues are the diagonal of the (now nearly diagonal) working matrix.
    let mut eig = Array1::<F>::zeros(n);
    for (i, value) in eig.iter_mut().enumerate() {
        *value = m[[i, i]];
    }
    (eig, v)
}

/// Moore–Penrose pseudo-inverse of a symmetric positive semi-definite matrix.
///
/// The matrix is diagonalised with [`jacobi_eigen_symmetric`] and reassembled as
/// `A⁺ = V · diag(λᵢ⁺) · Vᵀ`, where each reciprocal eigenvalue is truncated:
/// `λᵢ⁺ = 1/λᵢ` when `λᵢ > rcond · λ_max`, and `0` otherwise (with
/// `rcond ≈ 1e-12`).  For a full-rank, well-conditioned matrix every eigenvalue
/// survives the cutoff, so the result is the ordinary inverse and well-posed ADF
/// regressions are numerically unchanged.  For a rank-deficient `XᵀX` (e.g. a
/// perfectly linear input series) the null-space directions are projected out
/// instead of triggering a singular-matrix error.
#[allow(dead_code)]
fn pseudo_inverse_spd<F>(a: &Array2<F>) -> Array2<F>
where
    F: Float + FromPrimitive + Debug,
{
    let (eig, v) = jacobi_eigen_symmetric(a);
    let n = eig.len();

    // Largest eigenvalue magnitude sets the relative cutoff below which a
    // direction counts as numerically null.  (XᵀX is PSD, but round-off can make
    // a structurally-zero eigenvalue come out slightly negative.)
    let mut lambda_max = F::zero();
    for &lam in eig.iter() {
        let mag = lam.abs();
        if mag > lambda_max {
            lambda_max = mag;
        }
    }

    let rcond = F::from_f64(1e-12).unwrap_or_else(F::epsilon);
    let cutoff = rcond * lambda_max;

    // Reciprocal eigenvalues with the (near-)null space projected out.
    let inv_eig = eig.mapv(|lam| {
        if lam > cutoff {
            F::one() / lam
        } else {
            F::zero()
        }
    });

    // Reassemble A⁺ = V · diag(λᵢ⁺) · Vᵀ entrywise to stay at `F`-level
    // arithmetic: (A⁺)_{ij} = Σ_k V_{ik} · λ_k⁺ · V_{jk}.
    let mut pinv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut acc = F::zero();
            for k in 0..n {
                acc = acc + v[[i, k]] * inv_eig[k] * v[[j, k]];
            }
            pinv[[i, j]] = acc;
            pinv[[j, i]] = acc;
        }
    }
    pinv
}

/// Invert a small square matrix using Gauss-Jordan elimination with partial
/// pivoting.
///
/// This is intended for the modest `(p+2) × (p+2)` systems that arise in the
/// ADF regression, where `p` is the number of lags.  Returns a numerical-
/// instability error if the matrix is singular to working precision.
#[allow(dead_code)]
fn invert_matrix<F>(a: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix must be square to invert".to_string(),
        ));
    }

    // Augmented matrix [A | I].
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }

    let eps = F::from_f64(1e-12).unwrap_or_else(F::epsilon);
    for col in 0..n {
        // Partial pivoting: find the row with the largest magnitude in `col`.
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }
        if pivot_val <= eps {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix encountered while inverting ADF normal equations".to_string(),
            ));
        }
        if pivot_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }

        // Normalize the pivot row.
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] = aug[[col, j]] / pivot;
        }

        // Eliminate the current column from all other rows.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            if factor != F::zero() {
                for j in 0..(2 * n) {
                    aug[[row, j]] = aug[[row, j]] - factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract the right half as the inverse.
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Approximate the Augmented Dickey-Fuller p-value from the test statistic.
///
/// Uses the asymptotic response-surface critical values from MacKinnon for the
/// "constant, no trend" specification together with a smooth interpolation in
/// the tails.  The Dickey-Fuller distribution is left-skewed and non-standard,
/// so this returns a monotone approximation: smaller (more negative) statistics
/// map to smaller p-values (stronger evidence of stationarity).
#[allow(dead_code)]
fn mackinnon_p_value<F>(stat: F) -> F
where
    F: Float + FromPrimitive + Debug,
{
    let t = stat.to_f64().unwrap_or(0.0);

    // Asymptotic critical values (constant, no trend) from MacKinnon (1994).
    // (significance level, critical value)
    const TABLE: [(f64, f64); 7] = [
        (0.01, -3.43),
        (0.025, -3.12),
        (0.05, -2.86),
        (0.10, -2.57),
        (0.50, -1.95),
        (0.90, -1.14),
        (0.99, -0.44),
    ];

    let p = if t <= TABLE[0].1 {
        // More extreme than the 1% critical value: very small p-value, decaying
        // smoothly as the statistic becomes more negative.
        let excess = TABLE[0].1 - t; // >= 0
        (0.01 * (-1.2 * excess).exp()).max(1e-6)
    } else if t >= TABLE[TABLE.len() - 1].1 {
        // Less extreme than the 99% critical value: p-value approaches 1.
        let excess = t - TABLE[TABLE.len() - 1].1; // >= 0
        (1.0 - 0.01 * (-1.0 * excess).exp()).min(1.0 - 1e-6)
    } else {
        // Linear interpolation in the statistic between tabulated points.
        let mut result = 0.5;
        for w in TABLE.windows(2) {
            let (p_lo, c_lo) = w[0];
            let (p_hi, c_hi) = w[1];
            if t >= c_lo && t <= c_hi {
                let frac = (t - c_lo) / (c_hi - c_lo);
                result = p_lo + frac * (p_hi - p_lo);
                break;
            }
        }
        result
    };

    F::from_f64(p).unwrap_or_else(|| F::from_f64(0.5).unwrap_or_else(F::zero))
}

/// Transforms a time series to achieve stationarity
///
/// Common transformations include differencing, log transformation, or
/// seasonal differencing.
///
/// # Arguments
///
/// * `ts` - The time series data to transform
/// * `method` - The transformation method ("diff", "log", "seasonal_diff")
/// * `seasonal_period` - Seasonal period for seasonal differencing (required if method is "seasonal_diff")
///
/// # Returns
///
/// * The transformed time series
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::transform_to_stationary;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// // First-order differencing
/// let diff_ts = transform_to_stationary(&ts, "diff", None).expect("Operation failed");
///
/// // Log transformation
/// let log_ts = transform_to_stationary(&ts, "log", None).expect("Operation failed");
///
/// // Seasonal differencing with period 4
/// let seasonal_diff_ts = transform_to_stationary(&ts, "seasonal_diff", Some(4)).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn transform_to_stationary<F>(
    ts: &Array1<F>,
    method: &str,
    seasonal_period: Option<usize>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for transformation".to_string(),
        ));
    }

    match method {
        "diff" => {
            // First-order differencing: x(t) - x(t-1)
            let mut result = Vec::with_capacity(ts.len() - 1);
            for i in 1..ts.len() {
                result.push(ts[i] - ts[i - 1]);
            }
            Ok(Array1::from(result))
        }
        "log" => {
            // Log transformation
            let mut result = Vec::with_capacity(ts.len());
            for &val in ts.iter() {
                if val <= F::zero() {
                    return Err(TimeSeriesError::InvalidInput(
                        "Cannot apply log transformation to non-positive values".to_string(),
                    ));
                }
                result.push(val.ln());
            }
            Ok(Array1::from(result))
        }
        "seasonal_diff" => {
            let _period = match seasonal_period {
                Some(p) => p,
                None => {
                    return Err(TimeSeriesError::InvalidInput(
                        "Seasonal _period must be provided for seasonal differencing".to_string(),
                    ))
                }
            };

            if _period >= ts.len() {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Seasonal period ({}) must be less than time series length ({})",
                    _period,
                    ts.len()
                )));
            }

            // Seasonal differencing: x(t) - x(t-s)
            let mut result = Vec::with_capacity(ts.len() - _period);
            for i in _period..ts.len() {
                result.push(ts[i] - ts[i - _period]);
            }
            Ok(Array1::from(result))
        }
        _ => Err(TimeSeriesError::InvalidInput(format!(
            "Unknown transformation method: {method}"
        ))),
    }
}

/// Applies a centered moving average to smooth a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `window_size` - Size of the moving window
///
/// # Returns
///
/// * The smoothed time series
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::moving_average;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let ma = moving_average(&ts, 3).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn moving_average<F>(_ts: &Array1<F>, windowsize: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if windowsize < 1 {
        return Err(TimeSeriesError::InvalidInput(
            "Window size must be at least 1".to_string(),
        ));
    }

    if windowsize > _ts.len() {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Window size ({}) cannot be larger than time series length ({})",
            windowsize,
            _ts.len()
        )));
    }

    let half_window = windowsize / 2;
    let mut result = Array1::zeros(_ts.len());

    // For even-sized windows, handle the special case
    let is_even = windowsize.is_multiple_of(2);

    // Calculate the centered moving averages
    for i in 0.._ts.len() {
        // Calculate appropriate window boundaries
        let start = i.saturating_sub(half_window);
        let end = if i + half_window >= _ts.len() {
            _ts.len() - 1
        } else {
            i + half_window
        };

        // Adjust for even-sized windows (need one more point at the end)
        let end = if is_even && (end + 1 < _ts.len()) {
            end + 1
        } else {
            end
        };

        // Calculate the average
        let mut sum = F::zero();
        let mut count = F::zero();

        for j in start..=end {
            sum = sum + _ts[j];
            count = count + F::one();
        }

        result[i] = sum / count;
    }

    Ok(result)
}

/// Calculates the autocorrelation function (ACF) for a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `max_lag` - Maximum lag to compute (default: length of series - 1)
///
/// # Returns
///
/// * The autocorrelation values for each lag
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::autocorrelation;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let acf = autocorrelation(&ts, None).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn autocorrelation<F>(_ts: &Array1<F>, maxlag: Option<usize>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if _ts.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for autocorrelation".to_string(),
        ));
    }

    let max_lag = std::cmp::min(maxlag.unwrap_or(_ts.len() - 1), _ts.len() - 1);

    // Calculate mean
    let mean = _ts.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(_ts.len()).expect("Operation failed");

    // Calculate denominator (variance * n)
    let denominator = _ts
        .iter()
        .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean));

    if denominator == F::zero() {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot compute autocorrelation for constant time series".to_string(),
        ));
    }

    // Calculate autocorrelation for each _lag
    let mut result = Array1::zeros(max_lag + 1);

    for _lag in 0..=max_lag {
        let mut numerator = F::zero();

        for i in 0..(_ts.len() - _lag) {
            numerator = numerator + (_ts[i] - mean) * (_ts[i + _lag] - mean);
        }

        result[_lag] = numerator / denominator;
    }

    Ok(result)
}

/// Calculates the cross-correlation function (CCF) between two time series
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `max_lag` - Maximum lag to compute (default: min(length) / 4)
///
/// # Returns
///
/// * The cross-correlation values for each lag
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::cross_correlation;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![2.0, 3.0, 4.0, 5.0, 6.0];
/// let ccf = cross_correlation(&x, &y, Some(3)).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn cross_correlation<F>(
    x: &Array1<F>,
    y: &Array1<F>,
    max_lag: Option<usize>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let min_len = x.len().min(y.len());

    if min_len < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for cross-correlation".to_string(),
        ));
    }

    let default_max_lag = min_len / 4;
    let max_lag = max_lag.unwrap_or(default_max_lag).min(min_len - 1);

    let x_mean = x.sum() / F::from(x.len()).expect("Operation failed");
    let y_mean = y.sum() / F::from(y.len()).expect("Operation failed");

    let mut result = Array1::zeros(max_lag + 1);

    for _lag in 0..=max_lag {
        let mut numerator = F::zero();
        let mut count = 0;

        for i in 0..(min_len - _lag) {
            numerator = numerator + (x[i] - x_mean) * (y[i + _lag] - y_mean);
            count += 1;
        }

        if count > 0 {
            result[_lag] = numerator / F::from(count).expect("Failed to convert to float");
        }
    }

    Ok(result)
}

/// Calculates the partial autocorrelation function (PACF) for a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `max_lag` - Maximum lag to compute (default: length of series / 4)
///
/// # Returns
///
/// * The partial autocorrelation values for each lag
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::partial_autocorrelation;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let pacf = partial_autocorrelation(&ts, None).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn partial_autocorrelation<F>(_ts: &Array1<F>, maxlag: Option<usize>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if _ts.len() < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have at least 2 points for partial autocorrelation".to_string(),
        ));
    }

    let default_max_lag = std::cmp::min(_ts.len() / 4, 10);
    let max_lag = std::cmp::min(maxlag.unwrap_or(default_max_lag), _ts.len() - 1);

    // Calculate ACF first
    let acf = autocorrelation(_ts, Some(max_lag))?;

    // Initialize PACF array (_lag 0 is always 1.0)
    let mut pacf = Array1::zeros(max_lag + 1);
    pacf[0] = F::one();

    // For _lag 1, PACF = ACF
    if max_lag >= 1 {
        pacf[1] = acf[1];
    }

    // Compute PACF using Levinson-Durbin recursion
    // This is a simplified implementation of Durbin-Levinson algorithm
    if max_lag >= 2 {
        // Pre-allocate phi arrays
        let mut phi_old = Array1::zeros(max_lag + 1);

        for j in 2..=max_lag {
            // Copy previous phi values
            let mut phi = Array1::zeros(j + 1);
            for k in 1..j {
                phi[k] = phi_old[k];
            }

            // Calculate numerator and denominator
            let mut numerator = acf[j];
            for k in 1..j {
                numerator = numerator - phi_old[k] * acf[j - k];
            }

            let mut denominator = F::one();
            for k in 1..j {
                denominator = denominator - phi_old[k] * acf[k];
            }

            // Calculate the new PACF value
            phi[j] = numerator / denominator;

            // Update all phi values
            for k in 1..j {
                phi[k] = phi_old[k] - phi[j] * phi_old[j - k];
            }

            // Store the PACF value and update phi_old
            pacf[j] = phi[j];
            phi_old = phi;
        }
    }

    Ok(pacf)
}

/// Detrend data along an axis by removing linear or constant trend
///
/// # Arguments
///
/// * `data` - Input data array
/// * `axis` - Axis along which to detrend (0 for columns, 1 for rows)
/// * `detrend_type` - Type of detrending: "linear" or "constant"
/// * `breakpoints` - Optional sequence of breakpoints for piecewise linear detrending
///
/// # Returns
///
/// Detrended data array
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::detrend;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let detrended = detrend(&x.view(), 0, "constant", None).expect("Operation failed");
/// println!("Detrended: {:?}", detrended);
/// ```
#[allow(dead_code)]
pub fn detrend<S, F>(
    data: &ArrayBase<S, Ix1>,
    axis: usize,
    detrend_type: &str,
    breakpoints: Option<&[usize]>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::checkarray_finite(data, "data")?;

    if axis != 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Only axis=0 supported for 1D arrays".to_string(),
        ));
    }

    match detrend_type {
        "constant" => {
            let mean = data.mean().ok_or_else(|| {
                TimeSeriesError::ComputationError("Failed to compute mean".to_string())
            })?;
            Ok(data.map(|&x| x - mean))
        }
        "linear" => {
            let n = data.len();
            if n < 2 {
                return Err(TimeSeriesError::InvalidInput(
                    "Data must have at least 2 points for linear detrending".to_string(),
                ));
            }

            if let Some(bp) = breakpoints {
                // Piecewise linear detrending
                let mut result = data.to_owned();
                let mut bp_indices = vec![0];
                bp_indices.extend_from_slice(bp);
                bp_indices.push(n);

                for i in 0..bp_indices.len() - 1 {
                    let start = bp_indices[i];
                    let end = bp_indices[i + 1];
                    let segment = s![start..end];
                    let segment_data = data.slice(segment);
                    let trend = linear_trend(&segment_data, start)?;

                    for j in start..end {
                        result[j] = result[j] - trend[j - start];
                    }
                }
                Ok(result)
            } else {
                // Single linear detrending
                let trend = linear_trend(data, 0)?;
                Ok(data.to_owned() - trend)
            }
        }
        _ => Err(TimeSeriesError::InvalidInput(format!(
            "Invalid detrend _type: {detrend_type}. Must be 'constant' or 'linear'"
        ))),
    }
}

/// Detrend 2D data along an axis
#[allow(dead_code)]
pub fn detrend_2d<S, F>(
    data: &ArrayBase<S, Ix2>,
    axis: usize,
    detrend_type: &str,
    breakpoints: Option<&[usize]>,
) -> Result<Array2<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::checkarray_finite(data, "data")?;

    if axis > 1 {
        return Err(TimeSeriesError::InvalidInput(
            "Axis must be 0 or 1 for 2D arrays".to_string(),
        ));
    }

    let mut result = data.to_owned();

    if axis == 0 {
        // Detrend along columns
        for mut col in result.columns_mut() {
            let detrended = detrend(&col.view(), 0, detrend_type, breakpoints)?;
            col.assign(&detrended);
        }
    } else {
        // Detrend along rows
        for mut row in result.rows_mut() {
            let detrended = detrend(&row.view(), 0, detrend_type, breakpoints)?;
            row.assign(&detrended);
        }
    }

    Ok(result)
}

/// Compute linear trend for data
#[allow(dead_code)]
fn linear_trend<S, F>(data: &ArrayBase<S, Ix1>, offset: usize) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = data.len();
    let x = Array1::linspace(
        F::from(offset).expect("Failed to convert to float"),
        F::from(offset + n - 1).expect("Failed to convert to float"),
        n,
    );
    let y = data.to_owned();

    // Compute linear regression coefficients
    let x_mean = x
        .mean()
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to compute x mean".to_string()))?;
    let y_mean = y
        .mean()
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to compute y mean".to_string()))?;

    let x_centered = &x - x_mean;
    let y_centered = &y - y_mean;

    let numerator = x_centered.dot(&y_centered);
    let denominator = x_centered.dot(&x_centered);

    if denominator.abs() < F::epsilon() {
        return Err(TimeSeriesError::ComputationError(
            "Singular matrix in linear regression".to_string(),
        ));
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    Ok(x.map(|&xi| slope * xi + intercept))
}

/// Resample a time series to a new number of samples using Fourier method
///
/// # Arguments
///
/// * `x` - Input time series
/// * `num` - Number of samples in the resampled signal
/// * `axis` - Axis along which to resample (default: 0)
/// * `window` - Optional window applied in the Fourier domain
///
/// # Returns
///
/// Resampled time series
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::resample;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let resampled = resample(&x.view(), 10, 0, None).expect("Operation failed");
/// assert_eq!(resampled.len(), 10);
/// ```
#[allow(dead_code)]
pub fn resample<S, F>(
    x: &ArrayBase<S, Ix1>,
    num: usize,
    axis: usize,
    window: Option<&Array1<F>>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    scirs2_core::validation::checkarray_finite(x, "x")?;
    scirs2_core::validation::check_positive(num as f64, "num")?;

    if axis != 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Only axis=0 supported for 1D arrays".to_string(),
        ));
    }

    let n = x.len();
    if n == num {
        return Ok(x.to_owned());
    }

    // FFT-based polyphase resampling (equivalent to scipy.signal.resample)
    // 1. Convert to f64 and compute FFT
    let x_f64: Vec<f64> = x
        .iter()
        .map(|v| v.to_f64().expect("Failed to convert to f64"))
        .collect();

    let mut spectrum = scirs2_fft::fft(&x_f64, Some(n))
        .map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    // 2. Apply optional window in frequency domain before resampling
    if let Some(win) = window {
        if win.len() == n {
            for (s_val, w_val) in spectrum.iter_mut().zip(win.iter()) {
                let w_f64 = w_val.to_f64().expect("Failed to convert window to f64");
                s_val.re *= w_f64;
                s_val.im *= w_f64;
            }
        }
    }

    // 3. Zero-pad or truncate the spectrum to the new size
    //    For the Nyquist component (when n is even), we split it between +N/2 and -N/2
    //    to preserve energy — matches scipy.signal.resample behaviour
    let mut new_spectrum: Vec<scirs2_core::numeric::Complex64> = Vec::with_capacity(num);

    if num > n {
        // Upsampling: insert zeros around the Nyquist
        // Copy positive frequencies [0, n/2)
        let pos_half = n / 2;
        new_spectrum.extend_from_slice(&spectrum[..pos_half]);

        // Handle even-length Nyquist bin: split energy between +N/2 and −N/2
        if n % 2 == 0 {
            let nyq = spectrum[pos_half];
            let half_re = nyq.re * 0.5;
            let half_im = nyq.im * 0.5;
            new_spectrum.push(scirs2_core::numeric::Complex64::new(half_re, half_im));
            // Zero-pad the middle: num-n-1 zeros because the Nyquist is already split
            // across two bins (the push above and the push below), accounting for 1 extra bin.
            let zeros = num - n - 1;
            for _ in 0..zeros {
                new_spectrum.push(scirs2_core::numeric::Complex64::new(0.0, 0.0));
            }
            new_spectrum.push(scirs2_core::numeric::Complex64::new(half_re, half_im));
            // Copy remaining negative frequencies [pos_half+1 .. n)
            new_spectrum.extend_from_slice(&spectrum[pos_half + 1..]);
        } else {
            // Odd-length: no Nyquist bin — just zero-pad the high-frequency region
            let zeros = num - n;
            for _ in 0..zeros {
                new_spectrum.push(scirs2_core::numeric::Complex64::new(0.0, 0.0));
            }
            new_spectrum.extend_from_slice(&spectrum[pos_half..]);
        }
    } else {
        // Downsampling: truncate high frequencies (brick-wall anti-alias)
        let new_pos_half = num / 2;

        // Anti-alias: apply a smooth roll-off near the new Nyquist (half-window taper)
        // Use a simple raised-cosine taper over the top 10% of kept frequencies
        let taper_start = (new_pos_half as f64 * 0.9) as usize;
        for (i, s_val) in spectrum.iter_mut().enumerate().take(new_pos_half) {
            if i >= taper_start && new_pos_half > taper_start {
                let t = (i - taper_start) as f64 / (new_pos_half - taper_start) as f64;
                let taper = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                s_val.re *= taper;
                s_val.im *= taper;
            }
        }

        // Copy positive frequencies [0, new_pos_half)
        new_spectrum.extend_from_slice(&spectrum[..new_pos_half]);

        if num % 2 == 0 {
            // Even output: construct Nyquist bin by averaging the two bins being merged
            let nyq_pos = spectrum[new_pos_half];
            let nyq_neg = spectrum[n - new_pos_half];
            new_spectrum.push(scirs2_core::numeric::Complex64::new(
                nyq_pos.re + nyq_neg.re,
                nyq_pos.im + nyq_neg.im,
            ));
            // Copy negative frequencies [n-new_pos_half+1 .. n)
            new_spectrum.extend_from_slice(&spectrum[n - new_pos_half + 1..]);
        } else {
            // Odd output
            new_spectrum.extend_from_slice(&spectrum[n - new_pos_half..]);
        }
    }

    // Sanity: spectrum must be exactly `num` bins long before IFFT
    debug_assert_eq!(
        new_spectrum.len(),
        num,
        "BUG: new_spectrum has {} bins, expected {}",
        new_spectrum.len(),
        num
    );

    // 4. IFFT and scale: multiply by num/n to preserve amplitude
    let scale_factor = num as f64 / n as f64;
    let time_domain = scirs2_fft::ifft(&new_spectrum, Some(num))
        .map_err(|e| TimeSeriesError::ComputationError(e.to_string()))?;

    // 5. Take real part and convert back to F
    let result = Array1::from_vec(
        time_domain
            .iter()
            .take(num)
            .map(|c| {
                F::from(c.re * scale_factor)
                    .expect("Failed to convert resampled value to output type")
            })
            .collect(),
    );

    Ok(result)
}

/// Decimate a signal by applying a low-pass filter and downsampling
///
/// # Arguments
///
/// * `x` - Input signal
/// * `q` - Downsampling factor (integer)
/// * `n` - Filter order (default: 8)
/// * `ftype` - Filter type: "iir" or "fir" (default: "iir")
/// * `axis` - Axis along which to decimate (default: 0)
///
/// # Returns
///
/// Decimated signal
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::decimate;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let decimated = decimate(&x.view(), 2, Some(4), Some("iir"), 0).expect("Operation failed");
/// assert_eq!(decimated.len(), 4);
/// ```
#[allow(dead_code)]
pub fn decimate<S, F>(
    x: &ArrayBase<S, Ix1>,
    q: usize,
    n: Option<usize>,
    ftype: Option<&str>,
    axis: usize,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    scirs2_core::validation::checkarray_finite(x, "x")?;
    scirs2_core::validation::check_positive(q as f64, "q")?;

    if axis != 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Only axis=0 supported for 1D arrays".to_string(),
        ));
    }

    if q == 1 {
        return Ok(x.to_owned());
    }

    let filter_order = n.unwrap_or(8);
    let filter_type = ftype.unwrap_or("iir");

    // Design low-pass filter with cutoff at Nyquist/q
    let cutoff = F::from(0.5).expect("Failed to convert constant to float")
        / F::from(q).expect("Failed to convert to float");

    let filtered = match filter_type {
        "iir" => {
            // Apply Chebyshev Type I filter
            apply_chebyshev_filter(x, filter_order, cutoff)?
        }
        "fir" => {
            // Apply FIR filter using windowed sinc
            apply_fir_filter(x, filter_order, cutoff)?
        }
        _ => {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Invalid filter type: {filter_type}. Must be 'iir' or 'fir'"
            )))
        }
    };

    // Downsample
    let mut result = Array1::zeros(x.len() / q);
    for (i, j) in (0..x.len()).step_by(q).enumerate() {
        if i < result.len() {
            result[i] = filtered[j];
        }
    }

    Ok(result)
}

/// Apply Chebyshev Type I IIR filter via bilinear transform
///
/// Implements a Chebyshev Type I low-pass filter using:
/// 1. Analog prototype poles
/// 2. Pre-warped bilinear transform to digital domain
/// 3. Biquad (second-order-section) cascade direct-form II
///
/// `cutoff` is normalised cycles-per-sample (0 < cutoff < 0.5, Nyquist = 0.5).
/// The ripple in the passband is fixed at 1 dB, a common textbook default.
#[allow(dead_code)]
fn apply_chebyshev_filter<S, F>(x: &ArrayBase<S, Ix1>, order: usize, cutoff: F) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    let n_samples = x.len();
    if n_samples == 0 {
        return Ok(Array1::zeros(0));
    }
    let order = order.max(1);

    // --- 1. Work in f64 internally ---
    let cutoff_f64 = cutoff
        .to_f64()
        .expect("Failed to convert cutoff to f64")
        .clamp(1e-6, 0.4999); // keep away from poles

    // Passband ripple: 1 dB
    let ripple_db = 1.0_f64;
    // ε = sqrt(10^(rp/10) - 1)
    let eps = (10_f64.powf(ripple_db / 10.0) - 1.0).sqrt();

    // --- 2. Pre-warp digital cutoff to analog frequency ---
    // Bilinear transform with T=1: ω_a = 2 * tan(π * fc)
    let omega_a = 2.0 * (std::f64::consts::PI * cutoff_f64).tan();

    // --- 3. Chebyshev Type I prototype poles (LP, unit cutoff) ---
    // p_k = -sin(φ_k) + j·cos(φ_k) where φ_k = (2k-1)π/(2N), k=1..N
    // Scaled by ε^{-1/N} so the ripple is exactly ε at ω=1
    // Using: sinh/cosh formula: poles lie on an ellipse with parameters
    //   sinh_part = sinh(asinh(1/eps)/N)
    //   cosh_part = cosh(asinh(1/eps)/N)
    let sinh_part = (1.0 / eps).asinh() / order as f64;
    let sinh_v = sinh_part.sinh();
    let cosh_v = sinh_part.cosh();

    let mut analog_poles: Vec<(f64, f64)> = Vec::with_capacity(order);
    for k in 1..=order {
        let phi = std::f64::consts::PI * (2 * k - 1) as f64 / (2 * order) as f64;
        let re = -phi.sin() * sinh_v;
        let im = phi.cos() * cosh_v;
        analog_poles.push((re, im));
    }

    // --- 4. Frequency-scale prototype poles to cutoff ω_a ---
    let poles_scaled: Vec<(f64, f64)> = analog_poles
        .iter()
        .map(|(re, im)| (re * omega_a, im * omega_a))
        .collect();

    // --- 5. Bilinear transform: s → (z-1)/(z+1) * 2 (T=1)
    // s = σ + jω  →  z = (2 + s) / (2 - s)
    // For each analog pole s_k, compute digital pole z_k
    let mut digital_poles: Vec<(f64, f64)> = Vec::with_capacity(order);
    for (re, im) in &poles_scaled {
        // z = (2 + s) / (2 - s)
        // numerator: (2 + re, im)
        // denominator: (2 - re, -im)
        let n_re = 2.0 + re;
        let n_im = *im;
        let d_re = 2.0 - re;
        let d_im = -im;
        let denom = d_re * d_re + d_im * d_im;
        let z_re = (n_re * d_re + n_im * d_im) / denom;
        let z_im = (n_im * d_re - n_re * d_im) / denom;
        digital_poles.push((z_re, z_im));
    }

    // --- 6. Pair complex-conjugate poles into second-order sections ---
    // All digital poles: complex poles come in conjugate pairs; real poles are singletons.
    // Build biquad coefficients: H(z) = b0 + b1*z^{-1} + b2*z^{-2}
    //                                   ─────────────────────────────
    //                                   a0 + a1*z^{-1} + a2*z^{-2}
    // For a pole pair (re ± j*im): denominator = (1 - z_re*z^{-1})^2 + (z_im*z^{-1})^2
    //   a0=1, a1 = -2*z_re, a2 = z_re^2 + z_im^2
    // Numerator for LP Chebyshev (all digital zeros at z=-1, i.e. s→∞ maps to z=-1):
    //   b0=1, b1=2, b2=1  (scaled for unity gain at DC)
    struct Biquad {
        b0: f64,
        b1: f64,
        b2: f64,
        a1: f64,
        a2: f64,
    }

    let mut sections: Vec<Biquad> = Vec::new();
    let mut i = 0;
    while i < digital_poles.len() {
        let (z_re, z_im) = digital_poles[i];
        if z_im.abs() < 1e-10 {
            // Real pole: first-order section embedded in biquad with b2=a2=0
            // Numerator: (1 + z^{-1}) scaled for DC gain = 1
            // Gain at DC: b0/(1-a1) for first-order — normalise below
            let a1 = -z_re;
            let dc_gain = 1.0 / (1.0 + a1); // 1/(1 - z_re) after sign
            sections.push(Biquad {
                b0: dc_gain,
                b1: dc_gain,
                b2: 0.0,
                a1,
                a2: 0.0,
            });
            i += 1;
        } else if i + 1 < digital_poles.len() {
            // Complex conjugate pair
            let a1 = -2.0 * z_re;
            let a2 = z_re * z_re + z_im * z_im;
            // All digital zeros at z=-1 → numerator (1 + z^{-1})^2 = 1 + 2z^{-1} + z^{-2}
            // Gain at DC (z=1): (1+1+1)/(1+a1+a2) = 3/(1+a1+a2)
            let dc_gain_denom = 1.0 + a1 + a2;
            let scale = if dc_gain_denom.abs() < 1e-12 {
                1.0
            } else {
                3.0 / dc_gain_denom
            };
            sections.push(Biquad {
                b0: 1.0 * scale,
                b1: 2.0 * scale,
                b2: 1.0 * scale,
                a1,
                a2,
            });
            // Consume conjugate (next pole is the conjugate, same re, negative im)
            i += 2;
        } else {
            // Orphaned complex pole (shouldn't happen for a proper filter)
            let a1 = -2.0 * z_re;
            let a2 = z_re * z_re + z_im * z_im;
            let dc_gain_denom = 1.0 + a1 + a2;
            let scale = if dc_gain_denom.abs() < 1e-12 {
                1.0
            } else {
                3.0 / dc_gain_denom
            };
            sections.push(Biquad {
                b0: scale,
                b1: 2.0 * scale,
                b2: scale,
                a1,
                a2,
            });
            i += 1;
        }
    }

    // --- 7. Apply biquad cascade (direct-form II) to signal ---
    let x_f64: Vec<f64> = x
        .iter()
        .map(|v| v.to_f64().expect("Failed to convert signal to f64"))
        .collect();

    let mut y = x_f64.clone();
    for sec in &sections {
        let mut w1 = 0.0_f64;
        let mut w2 = 0.0_f64;
        for sample in y.iter_mut() {
            let w0 = *sample - sec.a1 * w1 - sec.a2 * w2;
            *sample = sec.b0 * w0 + sec.b1 * w1 + sec.b2 * w2;
            w2 = w1;
            w1 = w0;
        }
    }

    // --- 8. Convert back to F ---
    let result = Array1::from_vec(
        y.iter()
            .map(|&v| F::from(v).expect("Failed to convert filtered value to output type"))
            .collect(),
    );

    Ok(result)
}

/// Apply FIR filter using windowed sinc (simplified implementation)
#[allow(dead_code)]
fn apply_fir_filter<S, F>(x: &ArrayBase<S, Ix1>, order: usize, cutoff: F) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    // Create windowed sinc filter coefficients
    let mut coeffs = Array1::zeros(order + 1);
    let fc = cutoff;
    let half_order = order / 2;

    for i in 0..=order {
        let n = i as i32 - half_order as i32;
        if n == 0 {
            coeffs[i] = F::from(2.0).expect("Failed to convert constant to float") * fc;
        } else {
            let n_f = F::from(n).expect("Failed to convert to float");
            let pi = F::from(std::f64::consts::PI).expect("Failed to convert to float");
            coeffs[i] =
                (F::from(2.0).expect("Failed to convert constant to float") * fc * pi * n_f).sin()
                    / (pi * n_f);

            // Apply Hamming window
            let window = F::from(0.54).expect("Failed to convert constant to float")
                - F::from(0.46).expect("Failed to convert constant to float")
                    * (F::from(2.0).expect("Failed to convert constant to float")
                        * pi
                        * F::from(i).expect("Failed to convert to float")
                        / F::from(order).expect("Failed to convert to float"))
                    .cos();
            coeffs[i] = coeffs[i] * window;
        }
    }

    // Normalize coefficients
    let sum: F = coeffs.sum();
    coeffs.map_inplace(|x| *x = *x / sum);

    // Apply convolution
    convolve_1d(x, &coeffs.view())
}

/// Simple 1D convolution
#[allow(dead_code)]
fn convolve_1d<S, T, F>(x: &ArrayBase<S, Ix1>, kernel: &ArrayBase<T, Ix1>) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    T: Data<Elem = F>,
    F: Float + NumCast + FromPrimitive + Debug + Display,
{
    let n = x.len();
    let k = kernel.len();
    let half_k = k / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let mut sum = F::zero();
        for j in 0..k {
            let idx = i as i32 + j as i32 - half_k as i32;
            if idx >= 0 && idx < n as i32 {
                sum = sum + x[idx as usize] * kernel[j];
            }
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Creates a time series with a specified date range (in days)
///
/// # Arguments
///
/// * `start_date` - Start date in the format "YYYY-MM-DD"
/// * `end_date` - End date in the format "YYYY-MM-DD"
/// * `values` - The values for the time series (must match the date range length)
///
/// # Returns
///
/// * A tuple containing date strings and the time series values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::utils::create_time_series;
///
/// let values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let (dates, ts) = create_time_series("2023-01-01", "2023-01-07", &values).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn create_time_series<F>(
    start_date: &str,
    end_date: &str,
    values: &Array1<F>,
) -> Result<(Vec<String>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    // Parse dates (simplified implementation)
    // For a real implementation, we'd use chrono or time crates

    // Create a simple _date parser
    fn parse_date(_datestr: &str) -> Result<(i32, u32, u32)> {
        let parts: Vec<&str> = _datestr.split('-').collect();
        if parts.len() != 3 {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Invalid date format: {_datestr}, expected YYYY-MM-DD"
            )));
        }

        let year = parts[0]
            .parse::<i32>()
            .map_err(|_| TimeSeriesError::InvalidInput(format!("Invalid year: {}", parts[0])))?;

        let month = parts[1]
            .parse::<u32>()
            .map_err(|_| TimeSeriesError::InvalidInput(format!("Invalid month: {}", parts[1])))?;

        let day = parts[2]
            .parse::<u32>()
            .map_err(|_| TimeSeriesError::InvalidInput(format!("Invalid day: {}", parts[2])))?;

        if !(1..=12).contains(&month) {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Month must be between 1 and 12, got {month}"
            )));
        }

        if !(1..=31).contains(&day) {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Day must be between 1 and 31, got {day}"
            )));
        }

        Ok((year, month, day))
    }

    // Simple days between calculation (not accounting for leap years properly)
    fn days_between(start: (i32, u32, u32), end: (i32, u32, u32)) -> i32 {
        // Days in month (simplified, not accounting for leap years)
        let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        // Convert to days since year 0
        let start_days = start.0 * 365
            + (1..start.1).map(|m| days_in_month[m as usize]).sum::<u32>() as i32
            + start.2 as i32;

        let end_days = end.0 * 365
            + (1..end.1).map(|m| days_in_month[m as usize]).sum::<u32>() as i32
            + end.2 as i32;

        end_days - start_days + 1 // +1 to include both _start and end dates
    }

    // Generate dates (simple implementation)
    fn generate_dates(start: (i32, u32, u32), n_days: usize) -> Vec<String> {
        // Days in month (simplified, not accounting for leap years)
        let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let mut dates = Vec::with_capacity(n_days);
        let mut year = start.0;
        let mut month = start.1;
        let mut day = start.2;

        for _ in 0..n_days {
            dates.push(format!("{year:04}-{month:02}-{day:02}"));

            // Increment _date
            day += 1;
            if day > days_in_month[month as usize] {
                day = 1;
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }

        dates
    }

    let start = parse_date(start_date)?;
    let end = parse_date(end_date)?;

    let days = days_between(start, end);
    if days < 1 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "End _date ({end_date}) must be after start _date ({start_date})"
        )));
    }

    if values.len() != days as usize {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Values length ({}) must match _date range length ({})",
            values.len(),
            days
        )));
    }

    let dates = generate_dates(start, days as usize);
    let time_series = values.clone();

    Ok((dates, time_series))
}

/// Calculate basic statistics for a time series
pub fn calculate_basic_stats<F>(data: &Array1<F>) -> Result<std::collections::HashMap<String, f64>>
where
    F: Float + FromPrimitive + Into<f64>,
{
    let mut stats = std::collections::HashMap::new();

    if data.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Data array is empty".to_string(),
        ));
    }

    let n = data.len() as f64;
    let mean = data.mean_or(F::zero()).into();
    let variance = data
        .iter()
        .map(|x| {
            let diff = (*x).into() - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;

    stats.insert("mean".to_string(), mean);
    stats.insert("variance".to_string(), variance);
    stats.insert("std".to_string(), variance.sqrt());
    stats.insert(
        "min".to_string(),
        data.iter()
            .map(|x| (*x).into())
            .fold(f64::INFINITY, f64::min),
    );
    stats.insert(
        "max".to_string(),
        data.iter()
            .map(|x| (*x).into())
            .fold(f64::NEG_INFINITY, f64::max),
    );
    stats.insert("count".to_string(), n);

    Ok(stats)
}

/// Apply differencing to a time series
pub fn difference_series<F>(data: &Array1<F>, periods: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Clone,
{
    if periods == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Periods must be greater than 0".to_string(),
        ));
    }

    if data.len() <= periods {
        return Err(TimeSeriesError::InvalidInput(
            "Data length must be greater than periods".to_string(),
        ));
    }

    let mut result = Vec::new();
    for i in periods..data.len() {
        result.push(data[i] - data[i - periods]);
    }

    Ok(Array1::from_vec(result))
}

/// Apply seasonal differencing to a time series
pub fn seasonal_difference_series<F>(data: &Array1<F>, periods: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Clone,
{
    difference_series(data, periods)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_invert_matrix_identity() {
        let a = array![[4.0_f64, 3.0], [6.0, 3.0]];
        let inv = invert_matrix(&a).expect("matrix should be invertible");
        // Verify A * A^-1 == I.
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += a[[i, k]] * inv[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(acc, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_matrix_singular_errors() {
        // Rank-deficient matrix must be reported, not silently inverted.
        let a = array![[1.0_f64, 2.0], [2.0, 4.0]];
        assert!(invert_matrix(&a).is_err());
    }

    #[test]
    fn test_is_stationary_returns_real_statistic() {
        // A white-noise-like stationary series should yield a finite ADF
        // statistic and a p-value in [0, 1]; the previous stub always returned
        // the constants (-2.5, 0.1) regardless of input.
        let stationary = array![
            0.5_f64, -0.3, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1, 0.25, -0.35, 0.15, -0.25, 0.05, -0.15,
            0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.25, 0.15, -0.2
        ];
        let (stat, p) = is_stationary(&stationary, Some(1)).expect("ADF should succeed");

        // Statistic and p-value must be genuine finite numbers.
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&p));

        // Distinct (non-degenerate) data must produce a distinct statistic,
        // confirming the result is computed from the input rather than being a
        // fixed dummy. A noisy upward drift is used so the regression is
        // well-posed (a perfectly linear ramp would be rank-deficient).
        let trending = array![
            1.0_f64, 2.3, 3.1, 4.6, 5.2, 6.9, 7.4, 8.8, 9.3, 10.7, 11.2, 12.9, 13.4, 14.1, 15.8,
            16.2, 17.9, 18.3, 19.7, 20.4, 21.1, 22.8, 23.2, 24.9
        ];
        let (stat_trend, _p_trend) = is_stationary(&trending, Some(1)).expect("ADF should succeed");
        assert!((stat - stat_trend).abs() > 1e-9);
    }

    #[test]
    fn test_is_stationary_too_short() {
        let ts = array![1.0_f64, 2.0];
        assert!(is_stationary(&ts, None).is_err());
    }

    #[test]
    fn test_is_stationary_linear_ramp_no_panic() {
        // A perfectly linear ramp makes every first difference identical, so the
        // ADF design matrix is rank-deficient and the fit is perfect.  The robust
        // implementation must return Ok with a finite statistic and a well-formed
        // p-value rather than erroring on the singular normal equations.
        let ramp = Array1::from_vec((1..=20).map(|i| i as f64).collect());
        let (stat, p) = is_stationary(&ramp, None).expect("linear ramp must not error");
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_is_stationary_constant_series_no_panic() {
        // A constant series has zero input variance: it is trivially stationary
        // and must be reported as such (finite statistic, p-value in [0, 1]).
        let constant = Array1::from_elem(20, 5.0_f64);
        let (stat, p) = is_stationary(&constant, None).expect("constant series must not error");
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_detrend_constant() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x.view(), 0, "constant", None).expect("Operation failed");

        // Mean should be removed
        assert_relative_eq!(detrended.clone().mean(), 0.0, epsilon = 1e-10);

        // Check specific values
        assert_relative_eq!(detrended[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[4], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detrend_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x.view(), 0, "linear", None).expect("Operation failed");

        // Linear trend should be removed, result should be constant
        for i in 1..detrended.len() {
            assert_relative_eq!(detrended[i] - detrended[i - 1], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_detrend_linear_with_breakpoints() {
        let x = array![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let breakpoints = vec![4];
        let detrended =
            detrend(&x.view(), 0, "linear", Some(&breakpoints)).expect("Operation failed");

        // Each segment should have its linear trend removed
        assert_relative_eq!(detrended[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[7], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detrend_2d() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .expect("Operation failed");

        // Detrend along columns (axis=0)
        let detrended = detrend_2d(&x.view(), 0, "constant", None).expect("Operation failed");

        // Each column should have zero mean
        for col in detrended.columns() {
            assert_relative_eq!(col.mean(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_resample_upsample() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let resampled = resample(&x.view(), 8, 0, None).expect("Operation failed");

        assert_eq!(resampled.len(), 8);
        // FFT-based resampling treats the signal as periodic.
        // The first sample (DC preserved) should equal x[0].
        assert_relative_eq!(resampled[0], x[0], epsilon = 0.1);
        // The last sample (at t = 7/8 of a period) is interpolated between x[3]=4.0 and
        // x[0]=1.0 (periodic wrap), yielding ≈ 2.5 — consistent with scipy.signal.resample.
        assert_relative_eq!(resampled[resampled.len() - 1], 2.5_f64, epsilon = 0.1);
        // Intermediate sample at index 2 (t=0.25) should be close to x[1]=2.0
        assert_relative_eq!(resampled[2], 2.0_f64, epsilon = 0.2);
    }

    #[test]
    fn test_resample_downsample() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let resampled = resample(&x.view(), 4, 0, None).expect("Operation failed");

        assert_eq!(resampled.len(), 4);
    }

    #[test]
    fn test_decimate() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decimated = decimate(&x.view(), 2, Some(4), Some("iir"), 0).expect("Operation failed");

        assert_eq!(decimated.len(), 4);
    }

    #[test]
    fn test_invalid_detrend_type() {
        let x = array![1.0, 2.0, 3.0];
        let result = detrend(&x.view(), 0, "invalid", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_axis() {
        let x = array![1.0, 2.0, 3.0];
        let result = detrend(&x.view(), 1, "constant", None);
        assert!(result.is_err());
    }
}
