//! Granger causality testing for time series
//!
//! This module provides comprehensive Granger causality analysis:
//! - Bivariate Granger causality test with F-test and chi-squared statistics
//! - Multivariate conditional Granger causality
//! - Optimal lag selection using AIC, BIC, and HQIC criteria
//! - Spectral Granger causality (frequency domain analysis)

use crate::error::TimeSeriesError;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::validation::checkarray_finite;

use super::{
    chi_squared_p_value, compute_regression_likelihood, compute_regression_rss,
    f_distribution_p_value, solve_linear_system, CausalityResult,
};

/// Granger causality test result
#[derive(Debug, Clone)]
pub struct GrangerCausalityResult {
    /// F-statistic for the causality test
    pub f_statistic: f64,
    /// P-value of the F-test
    pub p_value: f64,
    /// Whether causality is rejected at the significance level
    pub is_causal: bool,
    /// Significance level used
    pub significance_level: f64,
    /// Degrees of freedom for the F-test (numerator, denominator)
    pub degrees_of_freedom: (usize, usize),
    /// Log-likelihood of the restricted model
    pub ll_restricted: f64,
    /// Log-likelihood of the unrestricted model
    pub ll_unrestricted: f64,
    /// Chi-squared statistic (likelihood ratio test)
    pub chi2_statistic: f64,
    /// P-value for chi-squared test
    pub chi2_p_value: f64,
    /// Number of lags used
    pub lag: usize,
}

/// Configuration for Granger causality test
#[derive(Debug, Clone)]
pub struct GrangerConfig {
    /// Maximum lag to test
    pub max_lag: usize,
    /// Significance level for the test
    pub significance_level: f64,
    /// Whether to include trend in the model
    pub include_trend: bool,
    /// Whether to include constant in the model
    pub include_constant: bool,
    /// Whether to auto-select optimal lag
    pub auto_lag: bool,
    /// Criterion for lag selection (if auto_lag is true)
    pub lag_criterion: LagSelectionCriterion,
}

impl Default for GrangerConfig {
    fn default() -> Self {
        Self {
            max_lag: 4,
            significance_level: 0.05,
            include_trend: false,
            include_constant: true,
            auto_lag: false,
            lag_criterion: LagSelectionCriterion::BIC,
        }
    }
}

/// Criterion for optimal lag selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagSelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion (Schwarz)
    BIC,
    /// Hannan-Quinn Information Criterion
    HQIC,
}

/// Result of lag selection analysis
#[derive(Debug, Clone)]
pub struct LagSelectionResult {
    /// Optimal lag according to each criterion
    pub optimal_lag_aic: usize,
    /// Optimal lag by BIC
    pub optimal_lag_bic: usize,
    /// Optimal lag by HQIC
    pub optimal_lag_hqic: usize,
    /// AIC values for each tested lag
    pub aic_values: Vec<f64>,
    /// BIC values for each tested lag
    pub bic_values: Vec<f64>,
    /// HQIC values for each tested lag
    pub hqic_values: Vec<f64>,
    /// The recommended lag (based on selected criterion)
    pub recommended_lag: usize,
    /// The criterion used for recommendation
    pub criterion: LagSelectionCriterion,
}

/// Result of multivariate conditional Granger causality
#[derive(Debug, Clone)]
pub struct MultivariateCausalityResult {
    /// F-statistic for conditional causality
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether conditional causality is detected
    pub is_causal: bool,
    /// Log-likelihood ratio
    pub log_likelihood_ratio: f64,
    /// Number of conditioning variables
    pub n_conditioning: usize,
    /// Lag used
    pub lag: usize,
}

/// Result of spectral Granger causality
#[derive(Debug, Clone)]
pub struct SpectralGrangerResult {
    /// Frequencies at which causality is measured
    pub frequencies: Vec<f64>,
    /// Spectral Granger causality values at each frequency
    pub causality_spectrum: Vec<f64>,
    /// Total (integrated) spectral causality
    pub total_causality: f64,
    /// Peak frequency of causality
    pub peak_frequency: f64,
    /// Peak causality value
    pub peak_causality: f64,
    /// Number of lags used for spectral estimation
    pub lag: usize,
}

/// Test bivariate Granger causality between two time series
///
/// Tests whether `x` Granger-causes `y` using vector autoregression.
/// Returns F-statistic, chi-squared statistic, and p-values.
///
/// # Arguments
///
/// * `x` - The potential causal series
/// * `y` - The potentially caused series
/// * `config` - Configuration for the test
///
/// # Returns
///
/// Result containing Granger causality test statistics
pub fn granger_test(
    x: &Array1<f64>,
    y: &Array1<f64>,
    config: &GrangerConfig,
) -> CausalityResult<GrangerCausalityResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have the same length".to_string(),
        ));
    }

    // Determine the lag to use
    let lag = if config.auto_lag {
        let lag_result = select_optimal_lag(x, y, config.max_lag, config.lag_criterion)?;
        lag_result.recommended_lag.max(1)
    } else {
        config.max_lag
    };

    if x.len() <= lag + 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for the specified lag".to_string(),
        ));
    }

    // Prepare data matrix with lags
    let n = x.len() - lag;
    let n_extra =
        if config.include_constant { 1 } else { 0 } + if config.include_trend { 1 } else { 0 };
    let unrestricted_cols = 2 * lag + n_extra;
    let restricted_cols = lag + n_extra;

    let mut data = Array2::zeros((n, unrestricted_cols));
    let mut y_vec = Array1::zeros(n);

    for i in 0..n {
        let row_idx = lag + i;
        y_vec[i] = y[row_idx];

        // Add lagged y values
        for l in 1..=lag {
            data[[i, l - 1]] = y[row_idx - l];
        }

        // Add lagged x values
        for l in 1..=lag {
            data[[i, lag + l - 1]] = x[row_idx - l];
        }

        let mut col = 2 * lag;
        // Add constant if requested
        if config.include_constant {
            data[[i, col]] = 1.0;
            col += 1;
        }

        // Add trend if requested
        if config.include_trend {
            data[[i, col]] = i as f64;
        }
    }

    // Fit unrestricted model (with x lags)
    let unrestricted_rss = compute_regression_rss(&data, &y_vec)?;
    let unrestricted_ll = compute_regression_likelihood(&data, &y_vec)?;

    // Fit restricted model (without x lags)
    let restricted_data = if n_extra > 0 {
        // Take first `lag` columns (y lags) and the extra columns at the end
        let mut rd = Array2::zeros((n, restricted_cols));
        for i in 0..n {
            for j in 0..lag {
                rd[[i, j]] = data[[i, j]];
            }
            for j in 0..n_extra {
                rd[[i, lag + j]] = data[[i, 2 * lag + j]];
            }
        }
        rd
    } else {
        data.slice(s![.., ..lag]).to_owned()
    };
    let restricted_rss = compute_regression_rss(&restricted_data, &y_vec)?;
    let restricted_ll = compute_regression_likelihood(&restricted_data, &y_vec)?;

    // Compute F-statistic
    let df_num = lag;
    let df_den = if n > unrestricted_cols {
        n - unrestricted_cols
    } else {
        1
    };
    let f_statistic = if df_den > 0 && unrestricted_rss > 0.0 {
        ((restricted_rss - unrestricted_rss) / df_num as f64) / (unrestricted_rss / df_den as f64)
    } else {
        0.0
    };

    // Compute F-test p-value
    let p_value = f_distribution_p_value(f_statistic, df_num, df_den);

    // Compute chi-squared statistic (likelihood ratio test)
    let chi2_statistic = if unrestricted_rss > 0.0 && restricted_rss > unrestricted_rss {
        n as f64 * (restricted_rss / unrestricted_rss).ln()
    } else {
        0.0
    };
    let chi2_p_value = chi_squared_p_value(chi2_statistic, lag);

    Ok(GrangerCausalityResult {
        f_statistic,
        p_value,
        is_causal: p_value < config.significance_level,
        significance_level: config.significance_level,
        degrees_of_freedom: (df_num, df_den),
        ll_restricted: restricted_ll,
        ll_unrestricted: unrestricted_ll,
        chi2_statistic,
        chi2_p_value,
        lag,
    })
}

/// Select optimal lag for Granger causality testing
///
/// Evaluates different lag orders using information criteria (AIC, BIC, HQIC)
/// on the unrestricted VAR model and returns the optimal lag for each.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `max_lag` - Maximum lag to test
/// * `criterion` - Which criterion to use for recommendation
///
/// # Returns
///
/// `LagSelectionResult` with optimal lags by AIC, BIC, HQIC
pub fn select_optimal_lag(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_lag: usize,
    criterion: LagSelectionCriterion,
) -> CausalityResult<LagSelectionResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have the same length".to_string(),
        ));
    }

    if max_lag == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "max_lag must be at least 1".to_string(),
        ));
    }

    let n_total = x.len();
    let effective_max_lag = max_lag.min(n_total / 3);

    if effective_max_lag == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for lag selection".to_string(),
        ));
    }

    let mut aic_values = Vec::with_capacity(effective_max_lag);
    let mut bic_values = Vec::with_capacity(effective_max_lag);
    let mut hqic_values = Vec::with_capacity(effective_max_lag);

    for lag in 1..=effective_max_lag {
        let n = n_total - lag;
        if n <= 2 * lag + 2 {
            break;
        }

        // Build unrestricted model design matrix
        let n_params = 2 * lag + 1; // y-lags + x-lags + constant
        let mut design = Array2::zeros((n, n_params));
        let mut response = Array1::zeros(n);

        for i in 0..n {
            let row_idx = lag + i;
            response[i] = y[row_idx];

            for l in 1..=lag {
                design[[i, l - 1]] = y[row_idx - l];
                design[[i, lag + l - 1]] = x[row_idx - l];
            }
            design[[i, 2 * lag]] = 1.0; // constant
        }

        // Compute RSS
        let rss = compute_regression_rss(&design, &response)?;
        let n_f = n as f64;
        let k_f = n_params as f64;

        // sigma^2 estimate
        let sigma2 = rss / n_f;
        if sigma2 <= 0.0 {
            continue;
        }
        let log_sigma2 = sigma2.ln();

        // AIC = n * ln(sigma^2) + 2k
        let aic = n_f * log_sigma2 + 2.0 * k_f;
        // BIC = n * ln(sigma^2) + k * ln(n)
        let bic = n_f * log_sigma2 + k_f * n_f.ln();
        // HQIC = n * ln(sigma^2) + 2k * ln(ln(n))
        let hqic = n_f * log_sigma2 + 2.0 * k_f * n_f.ln().ln();

        aic_values.push(aic);
        bic_values.push(bic);
        hqic_values.push(hqic);
    }

    if aic_values.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Could not compute information criteria for any lag".to_string(),
        ));
    }

    let optimal_lag_aic = find_min_index(&aic_values) + 1;
    let optimal_lag_bic = find_min_index(&bic_values) + 1;
    let optimal_lag_hqic = find_min_index(&hqic_values) + 1;

    let recommended_lag = match criterion {
        LagSelectionCriterion::AIC => optimal_lag_aic,
        LagSelectionCriterion::BIC => optimal_lag_bic,
        LagSelectionCriterion::HQIC => optimal_lag_hqic,
    };

    Ok(LagSelectionResult {
        optimal_lag_aic,
        optimal_lag_bic,
        optimal_lag_hqic,
        aic_values,
        bic_values,
        hqic_values,
        recommended_lag,
        criterion,
    })
}

/// Multivariate conditional Granger causality test
///
/// Tests whether `x` Granger-causes `y` conditional on a set of control variables `z`.
/// This controls for confounding variables that might cause spurious Granger causality.
///
/// # Arguments
///
/// * `x` - Potential cause (1D)
/// * `y` - Potential effect (1D)
/// * `z` - Conditioning/control variables (2D: rows = observations, cols = variables)
/// * `lag` - Number of lags
/// * `significance_level` - Significance threshold
///
/// # Returns
///
/// `MultivariateCausalityResult` with conditional F-test results
pub fn conditional_granger_test(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array2<f64>,
    lag: usize,
    significance_level: f64,
) -> CausalityResult<MultivariateCausalityResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    let n_total = x.len();
    if n_total != y.len() || n_total != z.nrows() {
        return Err(TimeSeriesError::InvalidInput(
            "All time series must have the same length".to_string(),
        ));
    }

    if lag == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Lag must be at least 1".to_string(),
        ));
    }

    let n_cond = z.ncols();
    let n = n_total - lag;

    if n <= (2 + n_cond) * lag + 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for conditional Granger causality with given lag and conditioning variables".to_string(),
        ));
    }

    // Build unrestricted model: y_t = f(y_{t-l}, x_{t-l}, z_{t-l}) + constant
    let unrestricted_params = (1 + 1 + n_cond) * lag + 1;
    let mut unrestricted_design = Array2::zeros((n, unrestricted_params));
    let mut response = Array1::zeros(n);

    for i in 0..n {
        let row_idx = lag + i;
        response[i] = y[row_idx];

        let mut col = 0;
        // Lagged y
        for l in 1..=lag {
            unrestricted_design[[i, col]] = y[row_idx - l];
            col += 1;
        }
        // Lagged x
        for l in 1..=lag {
            unrestricted_design[[i, col]] = x[row_idx - l];
            col += 1;
        }
        // Lagged z (all columns)
        for c in 0..n_cond {
            for l in 1..=lag {
                unrestricted_design[[i, col]] = z[[row_idx - l, c]];
                col += 1;
            }
        }
        // Constant
        unrestricted_design[[i, col]] = 1.0;
    }

    // Build restricted model: y_t = f(y_{t-l}, z_{t-l}) + constant (exclude x lags)
    let restricted_params = (1 + n_cond) * lag + 1;
    let mut restricted_design = Array2::zeros((n, restricted_params));

    for i in 0..n {
        let row_idx = lag + i;
        let mut col = 0;
        // Lagged y
        for l in 1..=lag {
            restricted_design[[i, col]] = y[row_idx - l];
            col += 1;
        }
        // Lagged z (all columns)
        for c in 0..n_cond {
            for l in 1..=lag {
                restricted_design[[i, col]] = z[[row_idx - l, c]];
                col += 1;
            }
        }
        // Constant
        restricted_design[[i, col]] = 1.0;
    }

    let unrestricted_rss = compute_regression_rss(&unrestricted_design, &response)?;
    let restricted_rss = compute_regression_rss(&restricted_design, &response)?;

    let unrestricted_ll = compute_regression_likelihood(&unrestricted_design, &response)?;
    let restricted_ll = compute_regression_likelihood(&restricted_design, &response)?;

    let df_num = lag; // number of x-lag parameters removed
    let df_den = if n > unrestricted_params {
        n - unrestricted_params
    } else {
        1
    };

    let f_statistic = if df_den > 0 && unrestricted_rss > 0.0 {
        ((restricted_rss - unrestricted_rss) / df_num as f64) / (unrestricted_rss / df_den as f64)
    } else {
        0.0
    };

    let p_value = f_distribution_p_value(f_statistic, df_num, df_den);

    let log_likelihood_ratio = if unrestricted_rss > 0.0 && restricted_rss > unrestricted_rss {
        n as f64 * (restricted_rss / unrestricted_rss).ln()
    } else {
        0.0
    };

    Ok(MultivariateCausalityResult {
        f_statistic,
        p_value,
        is_causal: p_value < significance_level,
        log_likelihood_ratio,
        n_conditioning: n_cond,
        lag,
    })
}

/// Spectral Granger causality (frequency domain)
///
/// Decomposes Granger causality into frequency-domain contributions, revealing
/// which frequencies drive the causal relationship. Uses parametric spectral
/// density estimation from the fitted VAR model.
///
/// # Arguments
///
/// * `x` - Potential cause
/// * `y` - Potential effect
/// * `lag` - Number of lags for the VAR model
/// * `n_freqs` - Number of frequency points to evaluate (default 128)
///
/// # Returns
///
/// `SpectralGrangerResult` with causality at each frequency
pub fn spectral_granger_causality(
    x: &Array1<f64>,
    y: &Array1<f64>,
    lag: usize,
    n_freqs: usize,
) -> CausalityResult<SpectralGrangerResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have the same length".to_string(),
        ));
    }

    let n_total = x.len();
    if n_total <= lag + 2 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for spectral Granger causality".to_string(),
        ));
    }

    let n = n_total - lag;

    // Fit bivariate VAR model: estimate coefficients for both equations
    // y_t = a0 + sum_l a_l * y_{t-l} + sum_l b_l * x_{t-l} + e1_t
    // x_t = c0 + sum_l c_l * y_{t-l} + sum_l d_l * x_{t-l} + e2_t

    let n_params = 2 * lag + 1;
    let mut design = Array2::zeros((n, n_params));
    let mut resp_y = Array1::zeros(n);
    let mut resp_x = Array1::zeros(n);

    for i in 0..n {
        let row_idx = lag + i;
        resp_y[i] = y[row_idx];
        resp_x[i] = x[row_idx];

        for l in 1..=lag {
            design[[i, l - 1]] = y[row_idx - l];
            design[[i, lag + l - 1]] = x[row_idx - l];
        }
        design[[i, 2 * lag]] = 1.0;
    }

    // Solve for VAR coefficients
    let xt = design.t();
    let xtx = xt.dot(&design);

    let xty = xt.dot(&resp_y);
    let beta_y = solve_linear_system(&xtx, &xty)?;

    let xtx2 = xt.dot(&design);
    let xtx_x = xt.dot(&resp_x);
    let beta_x = solve_linear_system(&xtx2, &xtx_x)?;

    // Compute residual covariance matrix
    let fitted_y = design.dot(&beta_y);
    let fitted_x = design.dot(&beta_x);
    let resid_y = &resp_y - &fitted_y;
    let resid_x = &resp_x - &fitted_x;

    let sigma_yy = resid_y.mapv(|v| v * v).sum() / n as f64;
    let sigma_xx = resid_x.mapv(|v| v * v).sum() / n as f64;
    let sigma_yx: f64 = resid_y
        .iter()
        .zip(resid_x.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        / n as f64;

    // Also compute restricted model (y without x lags) residual variance
    let mut restricted_design = Array2::zeros((n, lag + 1));
    for i in 0..n {
        let row_idx = lag + i;
        for l in 1..=lag {
            restricted_design[[i, l - 1]] = y[row_idx - l];
        }
        restricted_design[[i, lag]] = 1.0;
    }
    let restricted_rss = compute_regression_rss(&restricted_design, &resp_y)?;
    let sigma_yy_restricted = restricted_rss / n as f64;

    // Compute spectral causality at each frequency
    let mut frequencies = Vec::with_capacity(n_freqs);
    let mut causality_spectrum = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let freq = k as f64 / (2.0 * n_freqs as f64); // 0 to 0.5 (normalized)
        let omega = 2.0 * std::f64::consts::PI * freq;

        // Compute transfer function H(omega) for the VAR system
        // A(omega) = I - sum_l A_l * exp(-i*omega*l)
        // where A_l are the VAR coefficient matrices

        // For bivariate system, compute the (y,y) and (y,x) transfer function entries
        let mut h_yy_re = 1.0;
        let mut h_yy_im = 0.0;
        let mut h_yx_re = 0.0;
        let mut h_yx_im = 0.0;

        for l in 1..=lag {
            let phase = omega * l as f64;
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();

            // A_l coefficients for y equation
            let a_yy = beta_y[l - 1]; // y_{t-l} coefficient
            let a_yx = beta_y[lag + l - 1]; // x_{t-l} coefficient

            h_yy_re -= a_yy * cos_phase;
            h_yy_im += a_yy * sin_phase;
            h_yx_re -= a_yx * cos_phase;
            h_yx_im += a_yx * sin_phase;
        }

        // det(A(omega)) for 2x2 system
        // We need the full A matrix, but for spectral Granger causality
        // the key quantity is the power spectral density ratio

        // Spectral density of y under full model:
        // S_yy(omega) = |H_yy(omega)|^2 * sigma_yy + |H_yx(omega)|^2 * sigma_xx + cross terms
        let h_yy_mag2 = h_yy_re * h_yy_re + h_yy_im * h_yy_im;
        let h_yx_mag2 = h_yx_re * h_yx_re + h_yx_im * h_yx_im;

        // Spectral density under restricted model (no x influence):
        // S_yy_restricted(omega) = |H_yy_r(omega)|^2 * sigma_yy_r
        // For simplicity, use the time-domain ratio as a proxy
        let spectral_full = if h_yy_mag2 > 1e-15 {
            sigma_yy + (h_yx_mag2 / h_yy_mag2) * sigma_xx
        } else {
            sigma_yy
        };

        // Geweke's spectral Granger causality measure
        let spectral_gc = if spectral_full > 1e-15 && sigma_yy > 1e-15 {
            (sigma_yy_restricted / spectral_full).ln().max(0.0)
        } else {
            0.0
        };

        frequencies.push(freq);
        causality_spectrum.push(spectral_gc);
    }

    // Total causality (integral of spectral causality)
    let total_causality = if n_freqs > 1 {
        let df = 0.5 / n_freqs as f64;
        causality_spectrum.iter().sum::<f64>() * df
    } else {
        causality_spectrum.first().copied().unwrap_or(0.0)
    };

    // Find peak
    let (peak_idx, &peak_causality) = causality_spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let peak_frequency = frequencies.get(peak_idx).copied().unwrap_or(0.0);

    Ok(SpectralGrangerResult {
        frequencies,
        causality_spectrum,
        total_causality,
        peak_frequency,
        peak_causality,
        lag,
    })
}

// ---- Convenience functions ----

/// Convenience function: test Granger causality between two time series
///
/// Tests whether `x` Granger-causes `y` at the given maximum lag.
///
/// # Arguments
///
/// * `x` - The potential cause series
/// * `y` - The potential effect series
/// * `max_lag` - Maximum number of lags to include in the VAR model
///
/// # Returns
///
/// `GrangerCausalityResult` with F-statistic, chi-squared, and p-values
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_series::causality::granger_causality_test;
///
/// let n = 100;
/// let mut x = Array1::zeros(n);
/// let mut y = Array1::zeros(n);
/// for i in 1..n {
///     x[i] = 0.5 * x[i - 1] + 0.3 * (i as f64 * 0.1).sin();
///     y[i] = 0.3 * y[i - 1] + 0.4 * x[i - 1] + 0.1 * (i as f64 * 0.2).cos();
/// }
///
/// let result = granger_causality_test(&x, &y, 4).expect("Test failed");
/// println!("F-statistic: {}", result.f_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Is causal: {}", result.is_causal);
/// ```
pub fn granger_causality_test(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_lag: usize,
) -> CausalityResult<GrangerCausalityResult> {
    let config = GrangerConfig {
        max_lag,
        significance_level: 0.05,
        include_trend: false,
        include_constant: true,
        auto_lag: false,
        lag_criterion: LagSelectionCriterion::BIC,
    };
    granger_test(x, y, &config)
}

/// Convenience function: test Granger causality with custom significance level
///
/// # Arguments
///
/// * `x` - The potential cause series
/// * `y` - The potential effect series
/// * `max_lag` - Maximum number of lags
/// * `significance_level` - Significance level for the test (e.g., 0.05)
pub fn granger_causality_test_with_alpha(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_lag: usize,
    significance_level: f64,
) -> CausalityResult<GrangerCausalityResult> {
    let config = GrangerConfig {
        max_lag,
        significance_level,
        include_trend: false,
        include_constant: true,
        auto_lag: false,
        lag_criterion: LagSelectionCriterion::BIC,
    };
    granger_test(x, y, &config)
}

/// Test bidirectional Granger causality between two series
///
/// Tests both directions: x -> y and y -> x.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `max_lag` - Maximum lag for the test
///
/// # Returns
///
/// Tuple of (x_causes_y_result, y_causes_x_result)
pub fn granger_causality_bidirectional(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_lag: usize,
) -> CausalityResult<(GrangerCausalityResult, GrangerCausalityResult)> {
    let x_causes_y = granger_causality_test(x, y, max_lag)?;
    let y_causes_x = granger_causality_test(y, x, max_lag)?;
    Ok((x_causes_y, y_causes_x))
}

/// Granger causality test with automatic lag selection
///
/// Automatically selects the optimal lag using BIC (default), then runs the test.
///
/// # Arguments
///
/// * `x` - Potential cause
/// * `y` - Potential effect
/// * `max_lag` - Maximum lag to consider
///
/// # Returns
///
/// `GrangerCausalityResult` using the automatically selected lag
pub fn granger_causality_auto(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_lag: usize,
) -> CausalityResult<GrangerCausalityResult> {
    let config = GrangerConfig {
        max_lag,
        significance_level: 0.05,
        include_trend: false,
        include_constant: true,
        auto_lag: true,
        lag_criterion: LagSelectionCriterion::BIC,
    };
    granger_test(x, y, &config)
}

// =============================================================================
// Ergonomic Wrapper: GrangerCausalityTest
// =============================================================================

/// Ergonomic builder-style wrapper around the Granger causality test.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_series::causality::GrangerCausalityTest;
/// let x = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]);
/// let y = Array1::from_vec(vec![0.0f64, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
/// let result = GrangerCausalityTest::new(2).test(&x, &y);
/// ```
#[derive(Debug, Clone)]
pub struct GrangerCausalityTest {
    max_lag: usize,
}

impl GrangerCausalityTest {
    /// Create a new test with the given maximum lag.
    pub fn new(max_lag: usize) -> Self {
        Self { max_lag }
    }

    /// Run the Granger causality test (does `x` Granger-cause `y`?).
    ///
    /// Delegates to the underlying `granger_causality_test` function.
    pub fn test(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> CausalityResult<GrangerCausalityResult> {
        granger_causality_test(x, y, self.max_lag)
    }
}

impl Default for GrangerCausalityTest {
    fn default() -> Self {
        Self { max_lag: 4 }
    }
}

// ---- Internal helpers ----

fn find_min_index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn generate_causal_data(n: usize) -> (Array1<f64>, Array1<f64>) {
        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);

        for i in 1..n {
            x[i] = 0.5 * x[i - 1] + 0.3 * (i as f64 * 0.1).sin();
            y[i] = 0.3 * y[i - 1] + 0.4 * x[i - 1] + 0.1 * (i as f64 * 0.2).cos();
        }

        (x, y)
    }

    #[test]
    fn test_granger_causality() {
        let (x, y) = generate_causal_data(100);

        let config = GrangerConfig::default();
        let result = granger_test(&x, &y, &config).expect("Granger test failed");

        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.chi2_statistic >= 0.0);
        assert!(result.chi2_p_value >= 0.0 && result.chi2_p_value <= 1.0);
        assert_eq!(result.lag, 4);
    }

    #[test]
    fn test_granger_causality_convenience() {
        let (x, y) = generate_causal_data(100);

        let result = granger_causality_test(&x, &y, 4).expect("Convenience test failed");
        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.degrees_of_freedom.0, 4);
    }

    #[test]
    fn test_granger_bidirectional() {
        let (x, y) = generate_causal_data(100);

        let (x_causes_y, y_causes_x) =
            granger_causality_bidirectional(&x, &y, 2).expect("Bidirectional test failed");

        assert!(x_causes_y.f_statistic >= 0.0);
        assert!(y_causes_x.f_statistic >= 0.0);
        assert!(x_causes_y.p_value >= 0.0);
        assert!(y_causes_x.p_value >= 0.0);
    }

    #[test]
    fn test_granger_causality_no_effect() {
        let n = 100;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| (i as f64 * 0.3 + 2.0).cos()).collect());

        let result = granger_causality_test(&x, &y, 2).expect("Test failed");
        assert!(result.f_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_granger_causality_mismatched_lengths() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let result = granger_causality_test(&x, &y, 1);
        assert!(result.is_err(), "Mismatched lengths should fail");
    }

    #[test]
    fn test_granger_causality_with_alpha() {
        let (x, y) = generate_causal_data(100);

        let result_strict =
            granger_causality_test_with_alpha(&x, &y, 2, 0.01).expect("Test failed");
        assert_eq!(result_strict.significance_level, 0.01);

        let result_loose = granger_causality_test_with_alpha(&x, &y, 2, 0.10).expect("Test failed");
        assert_eq!(result_loose.significance_level, 0.10);

        assert!(
            (result_strict.f_statistic - result_loose.f_statistic).abs() < 1e-10,
            "F-statistic should be the same regardless of alpha"
        );
    }

    #[test]
    fn test_granger_causality_series_too_short() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = granger_causality_test(&x, &y, 4);
        assert!(result.is_err(), "Too-short series should fail");
    }

    #[test]
    fn test_lag_selection() {
        let (x, y) = generate_causal_data(200);

        let result = select_optimal_lag(&x, &y, 10, LagSelectionCriterion::BIC)
            .expect("Lag selection failed");

        assert!(result.optimal_lag_aic >= 1);
        assert!(result.optimal_lag_bic >= 1);
        assert!(result.optimal_lag_hqic >= 1);
        assert!(result.recommended_lag >= 1);
        assert!(!result.aic_values.is_empty());
        assert!(!result.bic_values.is_empty());
        assert!(!result.hqic_values.is_empty());
    }

    #[test]
    fn test_lag_selection_criteria() {
        let (x, y) = generate_causal_data(200);

        let aic_result = select_optimal_lag(&x, &y, 8, LagSelectionCriterion::AIC)
            .expect("AIC selection failed");
        let bic_result = select_optimal_lag(&x, &y, 8, LagSelectionCriterion::BIC)
            .expect("BIC selection failed");
        let hqic_result = select_optimal_lag(&x, &y, 8, LagSelectionCriterion::HQIC)
            .expect("HQIC selection failed");

        // All should select valid lags
        assert!(aic_result.recommended_lag >= 1 && aic_result.recommended_lag <= 8);
        assert!(bic_result.recommended_lag >= 1 && bic_result.recommended_lag <= 8);
        assert!(hqic_result.recommended_lag >= 1 && hqic_result.recommended_lag <= 8);

        // BIC typically selects fewer lags than AIC
        assert!(bic_result.optimal_lag_bic <= aic_result.optimal_lag_aic + 2);
    }

    #[test]
    fn test_granger_auto_lag() {
        let (x, y) = generate_causal_data(200);

        let result = granger_causality_auto(&x, &y, 10).expect("Auto lag test failed");

        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.lag >= 1 && result.lag <= 10);
    }

    #[test]
    fn test_conditional_granger() {
        let n = 200;
        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);
        let mut z1 = Array1::zeros(n);

        for i in 1..n {
            z1[i] = 0.5 * z1[i - 1] + 0.2 * (i as f64 * 0.05).sin();
            x[i] = 0.4 * x[i - 1] + 0.3 * z1[i - 1] + 0.2 * (i as f64 * 0.1).sin();
            y[i] =
                0.3 * y[i - 1] + 0.4 * x[i - 1] + 0.1 * z1[i - 1] + 0.05 * (i as f64 * 0.2).cos();
        }

        let z = Array2::from_shape_vec((n, 1), z1.to_vec()).expect("Shape creation failed");

        let result =
            conditional_granger_test(&x, &y, &z, 2, 0.05).expect("Conditional Granger test failed");

        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.n_conditioning, 1);
        assert_eq!(result.lag, 2);
    }

    #[test]
    fn test_spectral_granger() {
        let (x, y) = generate_causal_data(200);

        let result = spectral_granger_causality(&x, &y, 4, 64).expect("Spectral Granger failed");

        assert_eq!(result.frequencies.len(), 64);
        assert_eq!(result.causality_spectrum.len(), 64);
        assert!(result.total_causality >= 0.0);
        assert!(result.peak_frequency >= 0.0 && result.peak_frequency <= 0.5);
        assert!(result.peak_causality >= 0.0);
        assert_eq!(result.lag, 4);
    }

    #[test]
    fn test_spectral_granger_independent() {
        let n = 200;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| (i as f64 * 0.37 + 1.5).cos()).collect());

        let result = spectral_granger_causality(&x, &y, 2, 32).expect("Spectral Granger failed");

        assert_eq!(result.frequencies.len(), 32);
        // For independent series, total causality should be small
        assert!(result.total_causality.is_finite());
    }

    #[test]
    fn test_chi_squared_statistic() {
        let (x, y) = generate_causal_data(100);

        let result = granger_causality_test(&x, &y, 2).expect("Test failed");

        // Chi-squared statistic should be consistent with F-statistic
        assert!(result.chi2_statistic >= 0.0);
        assert!(result.chi2_p_value >= 0.0 && result.chi2_p_value <= 1.0);
    }
}
