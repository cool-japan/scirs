//! Time series transformations for preprocessing and analysis
//!
//! This module provides comprehensive transformation methods including:
//! - Stationarity transformations (Box-Cox, differencing, detrending)
//! - Normalization and scaling methods
//! - Stationarity tests (ADF, KPSS)
//! - Dimensionality reduction techniques

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

/// Box-Cox transformation parameters
#[derive(Debug, Clone)]
pub struct BoxCoxTransform<F> {
    /// Lambda parameter for Box-Cox transformation
    pub lambda: F,
    /// Whether lambda was estimated from data
    pub lambda_estimated: bool,
    /// Minimum value adjustment for zero/negative values
    pub min_adjustment: F,
}

/// Differencing transformation parameters
#[derive(Debug, Clone)]
pub struct DifferencingTransform {
    /// Order of differencing (1 = first differences, 2 = second differences, etc.)
    pub order: usize,
    /// Seasonal differencing lag (0 = no seasonal differencing)
    pub seasonal_lag: Option<usize>,
}

/// Normalization method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Z-score normalization (standardization)
    ZScore,
    /// Min-max scaling to [0, 1]
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// Min-max scaling to custom range
    MinMaxCustom(f64, f64),
}

/// Normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizationParams<F> {
    /// Mean (for Z-score) or minimum (for Min-Max)
    pub location: F,
    /// Standard deviation (for Z-score) or range (for Min-Max)
    pub scale: F,
    /// Method used for normalization
    pub method: NormalizationMethod,
}

/// Stationarity test results
#[derive(Debug, Clone)]
pub struct StationarityTest<F> {
    /// Test statistic
    pub statistic: F,
    /// p-value
    pub p_value: F,
    /// Critical values at different significance levels
    pub critical_values: Vec<(F, F)>, // (significance_level, critical_value)
    /// Whether the null hypothesis is rejected
    pub is_stationary: bool,
    /// Test type
    pub test_type: StationarityTestType,
}

/// Type of stationarity test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationarityTestType {
    /// Augmented Dickey-Fuller test
    ADF,
    /// Kwiatkowski-Phillips-Schmidt-Shin test
    KPSS,
}

/// Apply Box-Cox transformation to time series
///
/// The Box-Cox transformation is defined as:
/// - If λ ≠ 0: y(λ) = (y^λ - 1) / λ
/// - If λ = 0: y(λ) = ln(y)
///
/// # Arguments
///
/// * `ts` - Input time series (must be positive)
/// * `lambda` - Box-Cox parameter. If None, optimal lambda is estimated
///
/// # Returns
///
/// Tuple of (transformed_series, transformation_parameters)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::transformations::box_cox_transform;
///
/// let ts = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let (transformed, params) = box_cox_transform(&ts, Some(0.5)).unwrap();
/// ```
pub fn box_cox_transform<F, S>(
    ts: &ArrayBase<S, Ix1>,
    lambda: Option<F>,
) -> Result<(Array1<F>, BoxCoxTransform<F>)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = ts.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series cannot be empty".to_string(),
        ));
    }

    // Check for non-positive values and adjust if necessary
    let min_val = ts.iter().fold(F::infinity(), |acc, &x| acc.min(x));
    let min_adjustment = if min_val <= F::zero() {
        F::one() - min_val
    } else {
        F::zero()
    };

    let adjusted_ts = if min_adjustment > F::zero() {
        ts.mapv(|x| x + min_adjustment)
    } else {
        ts.to_owned()
    };

    // Estimate lambda if not provided
    let lambda_val = if let Some(l) = lambda {
        l
    } else {
        estimate_box_cox_lambda(&adjusted_ts)?
    };

    // Apply Box-Cox transformation
    let transformed = if lambda_val.abs() < F::from(1e-10).unwrap() {
        // Lambda ≈ 0: use natural logarithm
        adjusted_ts.mapv(|x| x.ln())
    } else {
        // Lambda ≠ 0: use power transformation
        adjusted_ts.mapv(|x| (x.powf(lambda_val) - F::one()) / lambda_val)
    };

    let transform_params = BoxCoxTransform {
        lambda: lambda_val,
        lambda_estimated: lambda.is_none(),
        min_adjustment,
    };

    Ok((transformed, transform_params))
}

/// Estimate optimal Box-Cox lambda parameter using maximum likelihood
fn estimate_box_cox_lambda<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let n = ts.len();
    let n_f = F::from(n).unwrap();

    // Search over a range of lambda values
    let lambda_range = Array1::linspace(-2.0, 2.0, 41);
    let mut best_lambda = F::zero();
    let mut best_log_likelihood = F::neg_infinity();

    for &lambda_f64 in lambda_range.iter() {
        let lambda = F::from(lambda_f64).unwrap();

        // Transform the data
        let transformed = if lambda.abs() < F::from(1e-10).unwrap() {
            ts.mapv(|x| x.ln())
        } else {
            ts.mapv(|x| (x.powf(lambda) - F::one()) / lambda)
        };

        // Calculate log-likelihood
        let mean = transformed.sum() / n_f;
        let variance = transformed.mapv(|x| (x - mean) * (x - mean)).sum() / n_f;

        if variance <= F::zero() {
            continue;
        }

        // Log-likelihood for normal distribution
        let log_likelihood = -n_f / F::from(2.0).unwrap()
            * (F::from(2.0 * std::f64::consts::PI).unwrap().ln() + variance.ln())
            - n_f / F::from(2.0).unwrap();

        // Add Jacobian term: (λ - 1) * Σ ln(x_i)
        let jacobian = (lambda - F::one()) * ts.mapv(|x| x.ln()).sum();
        let total_log_likelihood = log_likelihood + jacobian;

        if total_log_likelihood > best_log_likelihood {
            best_log_likelihood = total_log_likelihood;
            best_lambda = lambda;
        }
    }

    Ok(best_lambda)
}

/// Inverse Box-Cox transformation
///
/// # Arguments
///
/// * `transformed_ts` - Box-Cox transformed time series
/// * `params` - Box-Cox transformation parameters
///
/// # Returns
///
/// Original time series (approximately)
pub fn inverse_box_cox_transform<F, S>(
    transformed_ts: &ArrayBase<S, Ix1>,
    params: &BoxCoxTransform<F>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display,
{
    let lambda = params.lambda;

    let original = if lambda.abs() < F::from(1e-10).unwrap() {
        // Lambda ≈ 0: inverse of ln(x) is exp(x)
        transformed_ts.mapv(|x| x.exp())
    } else {
        // Lambda ≠ 0: inverse of (x^λ - 1)/λ is (λ*y + 1)^(1/λ)
        transformed_ts.mapv(|x| (lambda * x + F::one()).powf(F::one() / lambda))
    };

    // Remove the minimum adjustment
    let result = if params.min_adjustment > F::zero() {
        original.mapv(|x| x - params.min_adjustment)
    } else {
        original
    };

    Ok(result)
}

/// Apply differencing transformation to make time series stationary
///
/// # Arguments
///
/// * `ts` - Input time series
/// * `order` - Order of differencing (1 = first differences, 2 = second differences, etc.)
/// * `seasonal_lag` - Optional seasonal differencing lag
///
/// # Returns
///
/// Tuple of (differenced_series, transformation_parameters)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::transformations::difference_transform;
///
/// let ts = Array1::from_vec(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
/// let (differenced, params) = difference_transform(&ts, 1, None).unwrap();
/// // Result: [2.0, 3.0, 4.0, 5.0] (first differences)
/// ```
pub fn difference_transform<F, S>(
    ts: &ArrayBase<S, Ix1>,
    order: usize,
    seasonal_lag: Option<usize>,
) -> Result<(Array1<F>, DifferencingTransform)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Clone,
{
    if order == 0 && seasonal_lag.is_none() {
        return Ok((
            ts.to_owned(),
            DifferencingTransform {
                order: 0,
                seasonal_lag: None,
            },
        ));
    }

    let mut result = ts.to_owned();

    // Apply seasonal differencing first if specified
    if let Some(lag) = seasonal_lag {
        if lag == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Seasonal lag must be positive".to_string(),
            ));
        }

        if result.len() <= lag {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "Time series length {} is not sufficient for seasonal lag {}",
                    result.len(),
                    lag
                ),
                required: lag + 1,
                actual: result.len(),
            });
        }

        let seasonal_diff =
            Array1::from_shape_fn(result.len() - lag, |i| result[i + lag] - result[i]);
        result = seasonal_diff;
    }

    // Apply regular differencing
    for _ in 0..order {
        if result.len() <= 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Cannot apply more differences: series too short".to_string(),
                required: 2,
                actual: result.len(),
            });
        }

        let diff = Array1::from_shape_fn(result.len() - 1, |i| result[i + 1] - result[i]);
        result = diff;
    }

    let params = DifferencingTransform {
        order,
        seasonal_lag,
    };
    Ok((result, params))
}

/// Integrate (reverse difference) a time series
///
/// # Arguments
///
/// * `differenced_ts` - Differenced time series
/// * `params` - Differencing transformation parameters
/// * `initial_values` - Initial values needed for integration
///
/// # Returns
///
/// Integrated (original level) time series
pub fn integrate_transform<F, S>(
    differenced_ts: &ArrayBase<S, Ix1>,
    params: &DifferencingTransform,
    initial_values: &Array1<F>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Clone,
{
    let mut result = differenced_ts.to_owned();

    // Reverse regular differencing
    for _ in 0..params.order {
        let mut integrated = Array1::zeros(result.len() + 1);

        // Set initial value (should be provided in initial_values)
        let init_idx = params.order - 1;
        if init_idx >= initial_values.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Insufficient initial values for integration".to_string(),
            ));
        }
        integrated[0] = initial_values[init_idx];

        // Integrate by cumulative sum
        for i in 0..result.len() {
            integrated[i + 1] = integrated[i] + result[i];
        }
        result = integrated;
    }

    // Reverse seasonal differencing if applied
    if let Some(lag) = params.seasonal_lag {
        let mut seasonal_integrated = Array1::zeros(result.len() + lag);

        // Set initial seasonal values
        for i in 0..lag {
            if i >= initial_values.len() {
                return Err(TimeSeriesError::InvalidInput(
                    "Insufficient initial values for seasonal integration".to_string(),
                ));
            }
            seasonal_integrated[i] = initial_values[i];
        }

        // Integrate seasonally
        for i in 0..result.len() {
            seasonal_integrated[i + lag] = seasonal_integrated[i] + result[i];
        }
        result = seasonal_integrated;
    }

    Ok(result)
}

/// Normalize time series using specified method
///
/// # Arguments
///
/// * `ts` - Input time series
/// * `method` - Normalization method to use
///
/// # Returns
///
/// Tuple of (normalized_series, normalization_parameters)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::transformations::{normalize_transform, NormalizationMethod};
///
/// let ts = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let (normalized, params) = normalize_transform(&ts, NormalizationMethod::ZScore).unwrap();
/// ```
pub fn normalize_transform<F, S>(
    ts: &ArrayBase<S, Ix1>,
    method: NormalizationMethod,
) -> Result<(Array1<F>, NormalizationParams<F>)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + Clone,
{
    let n = ts.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Time series cannot be empty".to_string(),
        ));
    }

    let (location, scale, normalized) = match method {
        NormalizationMethod::ZScore => {
            // Z-score normalization: (x - μ) / σ
            let mean = ts.sum() / F::from(n).unwrap();
            let variance = ts.mapv(|x| (x - mean) * (x - mean)).sum() / F::from(n - 1).unwrap();
            let std_dev = variance.sqrt();

            if std_dev <= F::zero() {
                return Err(TimeSeriesError::InvalidInput(
                    "Cannot normalize: standard deviation is zero".to_string(),
                ));
            }

            let normalized = ts.mapv(|x| (x - mean) / std_dev);
            (mean, std_dev, normalized)
        }

        NormalizationMethod::MinMax => {
            // Min-max normalization: (x - min) / (max - min)
            let min_val = ts.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = ts.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
            let range = max_val - min_val;

            if range <= F::zero() {
                return Err(TimeSeriesError::InvalidInput(
                    "Cannot normalize: min equals max".to_string(),
                ));
            }

            let normalized = ts.mapv(|x| (x - min_val) / range);
            (min_val, range, normalized)
        }

        NormalizationMethod::MinMaxCustom(min_target, max_target) => {
            // Min-max normalization to custom range
            let min_val = ts.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = ts.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
            let range = max_val - min_val;

            if range <= F::zero() {
                return Err(TimeSeriesError::InvalidInput(
                    "Cannot normalize: min equals max".to_string(),
                ));
            }

            let min_target_f = F::from(min_target).unwrap();
            let max_target_f = F::from(max_target).unwrap();
            let target_range = max_target_f - min_target_f;

            let normalized = ts.mapv(|x| {
                let normalized_01 = (x - min_val) / range;
                min_target_f + normalized_01 * target_range
            });
            (min_val, range, normalized)
        }

        NormalizationMethod::Robust => {
            // Robust normalization using median and IQR
            let mut sorted_values: Vec<F> = ts.iter().cloned().collect();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if n % 2 == 0 {
                (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / F::from(2.0).unwrap()
            } else {
                sorted_values[n / 2]
            };

            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx.min(n - 1)];
            let iqr = q3 - q1;

            if iqr <= F::zero() {
                return Err(TimeSeriesError::InvalidInput(
                    "Cannot normalize: IQR is zero".to_string(),
                ));
            }

            let normalized = ts.mapv(|x| (x - median) / iqr);
            (median, iqr, normalized)
        }
    };

    let params = NormalizationParams {
        location,
        scale,
        method,
    };

    Ok((normalized, params))
}

/// Inverse normalization transformation
///
/// # Arguments
///
/// * `normalized_ts` - Normalized time series
/// * `params` - Normalization parameters from the forward transformation
///
/// # Returns
///
/// Original scale time series
pub fn inverse_normalize_transform<F, S>(
    normalized_ts: &ArrayBase<S, Ix1>,
    params: &NormalizationParams<F>,
) -> Array1<F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Clone,
{
    match params.method {
        NormalizationMethod::ZScore => {
            // Inverse Z-score: x = (z * σ) + μ
            normalized_ts.mapv(|x| x * params.scale + params.location)
        }

        NormalizationMethod::MinMax => {
            // Inverse min-max: x = (norm * range) + min
            normalized_ts.mapv(|x| x * params.scale + params.location)
        }

        NormalizationMethod::MinMaxCustom(min_target, max_target) => {
            // Inverse custom min-max
            let min_target_f = F::from(min_target).unwrap();
            let max_target_f = F::from(max_target).unwrap();
            let target_range = max_target_f - min_target_f;

            normalized_ts.mapv(|x| {
                let normalized_01 = (x - min_target_f) / target_range;
                normalized_01 * params.scale + params.location
            })
        }

        NormalizationMethod::Robust => {
            // Inverse robust: x = (norm * IQR) + median
            normalized_ts.mapv(|x| x * params.scale + params.location)
        }
    }
}

/// Augmented Dickey-Fuller test for stationarity
///
/// Tests the null hypothesis that a unit root is present in the time series.
/// H0: The series has a unit root (non-stationary)
/// H1: The series is stationary
///
/// # Arguments
///
/// * `ts` - Input time series
/// * `max_lags` - Maximum number of lags to include (auto-selected if None)
/// * `regression_type` - Type of regression ('c' = constant, 'ct' = constant and trend, 'nc' = no constant)
///
/// # Returns
///
/// Stationarity test results
pub fn adf_test<F, S>(
    ts: &ArrayBase<S, Ix1>,
    max_lags: Option<usize>,
    regression_type: &str,
) -> Result<StationarityTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + Clone + 'static,
{
    let n = ts.len();
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "ADF test requires at least 10 observations".to_string(),
            required: 10,
            actual: n,
        });
    }

    // Determine optimal number of lags using information criteria
    let lags = max_lags
        .unwrap_or_else(|| {
            // Rule of thumb: 12 * (n/100)^(1/4)
            let lag_estimate = 12.0 * (n as f64 / 100.0).powf(0.25);
            lag_estimate.floor() as usize
        })
        .min((n - 1) / 3);

    // Prepare regression data
    let y_diff = Array1::from_shape_fn(n - 1, |i| ts[i + 1] - ts[i]);
    let y_lag = Array1::from_shape_fn(n - 1, |i| ts[i]);

    let start_idx = lags;
    let regression_length = n - 1 - start_idx;

    if regression_length < 5 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Insufficient data for ADF regression after accounting for lags".to_string(),
            required: start_idx + 5,
            actual: n,
        });
    }

    // Build regression matrix
    let mut n_regressors = 1; // y_{t-1} coefficient
    if regression_type.contains('c') {
        n_regressors += 1;
    } // constant
    if regression_type.contains('t') {
        n_regressors += 1;
    } // trend
    n_regressors += lags; // lagged differences

    let mut x_matrix = Array2::zeros((regression_length, n_regressors));
    let mut y_vector = Array1::zeros(regression_length);

    let mut col_idx = 0;

    // Constant term
    if regression_type.contains('c') {
        for i in 0..regression_length {
            x_matrix[[i, col_idx]] = F::one();
        }
        col_idx += 1;
    }

    // Trend term
    if regression_type.contains('t') {
        for i in 0..regression_length {
            x_matrix[[i, col_idx]] = F::from(i + 1).unwrap();
        }
        col_idx += 1;
    }

    // y_{t-1} term (this is what we test)
    for i in 0..regression_length {
        x_matrix[[i, col_idx]] = y_lag[start_idx + i];
        y_vector[i] = y_diff[start_idx + i];
    }
    col_idx += 1;

    // Lagged difference terms
    for lag in 1..=lags {
        for i in 0..regression_length {
            let diff_idx = start_idx + i - lag;
            x_matrix[[i, col_idx]] = y_diff[diff_idx];
        }
        col_idx += 1;
    }

    // Perform OLS regression
    let xtx = x_matrix.t().dot(&x_matrix);
    let xty = x_matrix.t().dot(&y_vector);

    // Solve normal equations (simplified - would use proper linear algebra in practice)
    let beta = solve_ols_simple(&xtx, &xty)?;

    // Calculate residuals and standard errors
    let y_pred = x_matrix.dot(&beta);
    let residuals = &y_vector - &y_pred;
    let mse = residuals.mapv(|x| x * x).sum() / F::from(regression_length - n_regressors).unwrap();

    // Standard error of the coefficient of interest (y_{t-1})
    let coeff_idx = if regression_type.contains('c') { 1 } else { 0 };
    let coeff_idx = if regression_type.contains('t') {
        coeff_idx + 1
    } else {
        coeff_idx
    };

    let var_coeff = mse * pseudo_inverse_diag(&xtx, coeff_idx)?;
    let se_coeff = var_coeff.sqrt();

    // t-statistic for unit root test
    let t_stat = beta[coeff_idx] / se_coeff;

    // Critical values (approximated)
    let critical_values = get_adf_critical_values(regression_type);

    // Determine if stationary (reject H0)
    let is_stationary = t_stat < critical_values[1].1; // 5% level

    // P-value approximation (simplified)
    let p_value = approximate_adf_p_value(t_stat, regression_type);

    Ok(StationarityTest {
        statistic: t_stat,
        p_value,
        critical_values,
        is_stationary,
        test_type: StationarityTestType::ADF,
    })
}

/// Simple OLS solver for small matrices
fn solve_ols_simple<F>(xtx: &Array2<F>, xty: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display + Clone,
{
    let n = xtx.nrows();

    // Simple case: 1x1 matrix
    if n == 1 {
        if xtx[[0, 0]].abs() < F::from(1e-12).unwrap() {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in OLS".to_string(),
            ));
        }
        return Ok(Array1::from_elem(1, xty[0] / xtx[[0, 0]]));
    }

    // For larger matrices, use simplified Gaussian elimination
    let mut a = xtx.clone();
    let mut b = xty.clone();

    // Forward elimination
    for k in 0..(n - 1) {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[[i, k]].abs() > a[[max_row, k]].abs() {
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in k..n {
                let temp = a[[k, j]];
                a[[k, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            let temp = b[k];
            b[k] = b[max_row];
            b[max_row] = temp;
        }

        // Eliminate
        for i in (k + 1)..n {
            if a[[k, k]].abs() < F::from(1e-12).unwrap() {
                return Err(TimeSeriesError::NumericalInstability(
                    "Near-zero pivot in OLS".to_string(),
                ));
            }
            let factor = a[[i, k]] / a[[k, k]];
            for j in k..n {
                a[[i, j]] = a[[i, j]] - factor * a[[k, j]];
            }
            b[i] = b[i] - factor * b[k];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum = sum + a[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / a[[i, i]];
    }

    Ok(x)
}

/// Get diagonal element of pseudo-inverse (simplified)
fn pseudo_inverse_diag<F>(matrix: &Array2<F>, idx: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Simplified: just return 1/diagonal for well-conditioned case
    if matrix[[idx, idx]].abs() < F::from(1e-12).unwrap() {
        return Err(TimeSeriesError::NumericalInstability(
            "Matrix is singular".to_string(),
        ));
    }
    Ok(F::one() / matrix[[idx, idx]])
}

/// Get ADF critical values (approximated)
fn get_adf_critical_values<F>(regression_type: &str) -> Vec<(F, F)>
where
    F: Float + FromPrimitive,
{
    // Simplified critical values - in practice these would be more sophisticated
    match regression_type {
        "nc" => vec![
            (F::from(0.01).unwrap(), F::from(-2.58).unwrap()),
            (F::from(0.05).unwrap(), F::from(-1.95).unwrap()),
            (F::from(0.10).unwrap(), F::from(-1.62).unwrap()),
        ],
        "c" => vec![
            (F::from(0.01).unwrap(), F::from(-3.43).unwrap()),
            (F::from(0.05).unwrap(), F::from(-2.86).unwrap()),
            (F::from(0.10).unwrap(), F::from(-2.57).unwrap()),
        ],
        "ct" => vec![
            (F::from(0.01).unwrap(), F::from(-3.96).unwrap()),
            (F::from(0.05).unwrap(), F::from(-3.41).unwrap()),
            (F::from(0.10).unwrap(), F::from(-3.13).unwrap()),
        ],
        _ => vec![
            (F::from(0.01).unwrap(), F::from(-3.43).unwrap()),
            (F::from(0.05).unwrap(), F::from(-2.86).unwrap()),
            (F::from(0.10).unwrap(), F::from(-2.57).unwrap()),
        ],
    }
}

/// Approximate p-value for ADF test (simplified)
fn approximate_adf_p_value<F>(t_stat: F, _regression_type: &str) -> F
where
    F: Float + FromPrimitive,
{
    // Very simplified p-value approximation
    if t_stat < F::from(-3.0).unwrap() {
        F::from(0.01).unwrap()
    } else if t_stat < F::from(-2.5).unwrap() {
        F::from(0.05).unwrap()
    } else if t_stat < F::from(-2.0).unwrap() {
        F::from(0.10).unwrap()
    } else {
        F::from(0.20).unwrap()
    }
}

/// KPSS test for stationarity
///
/// Tests the null hypothesis that the time series is stationary around a deterministic trend.
/// H0: The series is stationary
/// H1: The series has a unit root (non-stationary)
///
/// # Arguments
///
/// * `ts` - Input time series
/// * `regression_type` - Type of regression ('c' = level stationary, 'ct' = trend stationary)
///
/// # Returns
///
/// Stationarity test results
pub fn kpss_test<F, S>(ts: &ArrayBase<S, Ix1>, regression_type: &str) -> Result<StationarityTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + Clone,
{
    let n = ts.len();
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "KPSS test requires at least 10 observations".to_string(),
            required: 10,
            actual: n,
        });
    }

    // Determine regression type
    let include_trend = regression_type.contains('t');

    // Detrend the series
    let detrended = if include_trend {
        // Remove linear trend
        detrend_linear(ts)?
    } else {
        // Remove mean (level)
        let mean = ts.sum() / F::from(n).unwrap();
        ts.mapv(|x| x - mean)
    };

    // Calculate partial sums
    let mut partial_sums = Array1::zeros(n);
    partial_sums[0] = detrended[0];
    for i in 1..n {
        partial_sums[i] = partial_sums[i - 1] + detrended[i];
    }

    // Calculate LM statistic
    let sum_squares = partial_sums.mapv(|x| x * x).sum();
    let _residual_variance = detrended.mapv(|x| x * x).sum() / F::from(n).unwrap();

    // Long-run variance estimation (Newey-West)
    let long_run_variance = estimate_long_run_variance(&detrended)?;

    let lm_stat = sum_squares / (F::from(n * n).unwrap() * long_run_variance);

    // Critical values
    let critical_values = get_kpss_critical_values(include_trend);

    // Determine if stationary (fail to reject H0)
    let is_stationary = lm_stat < critical_values[1].1; // 5% level

    // P-value approximation
    let p_value = approximate_kpss_p_value(lm_stat, include_trend);

    Ok(StationarityTest {
        statistic: lm_stat,
        p_value,
        critical_values,
        is_stationary,
        test_type: StationarityTestType::KPSS,
    })
}

/// Remove linear trend from time series
fn detrend_linear<F, S>(ts: &ArrayBase<S, Ix1>) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    let n_f = F::from(n).unwrap();

    // Create time index
    let time_index: Array1<F> = (0..n).map(|i| F::from(i).unwrap()).collect();

    // Calculate linear regression coefficients
    let sum_t = time_index.sum();
    let sum_y = ts.sum();
    let sum_tt = time_index.mapv(|t| t * t).sum();
    let sum_ty = time_index
        .iter()
        .zip(ts.iter())
        .map(|(&t, &y)| t * y)
        .fold(F::zero(), |acc, x| acc + x);

    let mean_t = sum_t / n_f;
    let mean_y = sum_y / n_f;

    let denominator = sum_tt - n_f * mean_t * mean_t;
    if denominator.abs() < F::from(1e-12).unwrap() {
        return Err(TimeSeriesError::NumericalInstability(
            "Cannot detrend: degenerate time series".to_string(),
        ));
    }

    let slope = (sum_ty - n_f * mean_t * mean_y) / denominator;
    let intercept = mean_y - slope * mean_t;

    // Remove trend
    let detrended = time_index
        .iter()
        .zip(ts.iter())
        .map(|(&t, &y)| y - (intercept + slope * t))
        .collect();

    Ok(detrended)
}

/// Estimate long-run variance using Newey-West estimator
fn estimate_long_run_variance<F>(residuals: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = residuals.len();
    let n_f = F::from(n).unwrap();

    // Base variance
    let mut variance = residuals.mapv(|x| x * x).sum() / n_f;

    // Add autocovariance terms
    let max_lag = (n as f64).powf(1.0 / 3.0).floor() as usize; // Rule of thumb

    for lag in 1..=max_lag.min(n - 1) {
        let mut autocovariance = F::zero();
        for i in lag..n {
            autocovariance = autocovariance + residuals[i] * residuals[i - lag];
        }
        autocovariance = autocovariance / n_f;

        // Bartlett kernel weights
        let weight = F::one() - F::from(lag).unwrap() / F::from(max_lag + 1).unwrap();
        variance = variance + F::from(2.0).unwrap() * weight * autocovariance;
    }

    Ok(variance.max(F::from(1e-10).unwrap())) // Ensure positive
}

/// Get KPSS critical values
fn get_kpss_critical_values<F>(include_trend: bool) -> Vec<(F, F)>
where
    F: Float + FromPrimitive,
{
    if include_trend {
        vec![
            (F::from(0.01).unwrap(), F::from(0.216).unwrap()),
            (F::from(0.05).unwrap(), F::from(0.146).unwrap()),
            (F::from(0.10).unwrap(), F::from(0.119).unwrap()),
        ]
    } else {
        vec![
            (F::from(0.01).unwrap(), F::from(0.739).unwrap()),
            (F::from(0.05).unwrap(), F::from(0.463).unwrap()),
            (F::from(0.10).unwrap(), F::from(0.347).unwrap()),
        ]
    }
}

/// Approximate p-value for KPSS test
fn approximate_kpss_p_value<F>(lm_stat: F, include_trend: bool) -> F
where
    F: Float + FromPrimitive,
{
    let critical_vals = get_kpss_critical_values::<F>(include_trend);

    if lm_stat > critical_vals[0].1 {
        F::from(0.01).unwrap()
    } else if lm_stat > critical_vals[1].1 {
        F::from(0.05).unwrap()
    } else if lm_stat > critical_vals[2].1 {
        F::from(0.10).unwrap()
    } else {
        F::from(0.20).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_box_cox_transform() {
        let ts = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test with lambda = 0 (log transformation)
        let (transformed, params) = box_cox_transform(&ts, Some(0.0)).unwrap();
        let expected: Array1<f64> = ts.mapv(|x| x.ln());

        for i in 0..ts.len() {
            assert_relative_eq!(transformed[i], expected[i], epsilon = 1e-10);
        }

        // Test inverse transformation
        let recovered = inverse_box_cox_transform(&transformed, &params).unwrap();
        for i in 0..ts.len() {
            assert_relative_eq!(recovered[i], ts[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_box_cox_lambda_estimation() {
        let ts = array![1.0, 4.0, 9.0, 16.0, 25.0]; // Perfect squares
        let (transformed, params) = box_cox_transform(&ts, None).unwrap();

        // Should estimate a lambda that makes the series more normal
        assert!(params.lambda_estimated);
        assert!(transformed.len() == ts.len());
    }

    #[test]
    fn test_difference_transform() {
        let ts = array![1.0, 3.0, 6.0, 10.0, 15.0, 21.0];

        // First differences
        let (diff1, params) = difference_transform(&ts, 1, None).unwrap();
        let expected_diff1 = array![2.0, 3.0, 4.0, 5.0, 6.0];

        assert_eq!(diff1, expected_diff1);
        assert_eq!(params.order, 1);
        assert_eq!(params.seasonal_lag, None);

        // Second differences
        let (diff2, _) = difference_transform(&ts, 2, None).unwrap();
        let expected_diff2 = array![1.0, 1.0, 1.0, 1.0];

        assert_eq!(diff2, expected_diff2);
    }

    #[test]
    fn test_seasonal_difference() {
        let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Seasonal differencing with lag 4
        let (seasonal_diff, params) = difference_transform(&ts, 0, Some(4)).unwrap();
        let expected = array![4.0, 4.0, 4.0, 4.0]; // [5-1, 6-2, 7-3, 8-4]

        assert_eq!(seasonal_diff, expected);
        assert_eq!(params.seasonal_lag, Some(4));
    }

    #[test]
    fn test_normalize_z_score() {
        let ts = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (normalized, params) = normalize_transform(&ts, NormalizationMethod::ZScore).unwrap();

        // Check that mean is approximately 0 and std is approximately 1
        let mean = normalized.sum() / normalized.len() as f64;
        let variance =
            normalized.mapv(|x| (x - mean) * (x - mean)).sum() / (normalized.len() - 1) as f64;
        let std = variance.sqrt();

        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        assert_relative_eq!(std, 1.0, epsilon = 1e-10);

        // Test inverse transformation
        let recovered = inverse_normalize_transform(&normalized, &params);
        for i in 0..ts.len() {
            assert_relative_eq!(recovered[i], ts[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize_min_max() {
        let ts = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (normalized, params) = normalize_transform(&ts, NormalizationMethod::MinMax).unwrap();

        // Check that min is 0 and max is 1
        let min_val = normalized.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = normalized
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        assert_relative_eq!(min_val, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-10);

        // Test inverse transformation
        let recovered = inverse_normalize_transform(&normalized, &params);
        for i in 0..ts.len() {
            assert_relative_eq!(recovered[i], ts[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adf_test() {
        // Create a non-stationary random walk with more variation
        let mut ts = Array1::zeros(100);
        ts[0] = 10.0;
        for i in 1..100 {
            ts[i] = ts[i - 1] + 0.5 * (i as f64 / 10.0).sin() + 0.1 * (i as f64);
            // Trending with variation
        }

        let result = adf_test(&ts, None, "c").unwrap();

        // Should have proper structure
        assert_eq!(result.test_type, StationarityTestType::ADF);
        assert!(result.critical_values.len() >= 3);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_kpss_test() {
        // Create a stationary series
        let ts: Array1<f64> = (0..50)
            .map(|i| (i as f64 / 10.0).sin() + 0.1 * (i as f64))
            .collect();

        let result = kpss_test(&ts, "c").unwrap();

        // Should have proper structure
        assert_eq!(result.test_type, StationarityTestType::KPSS);
        assert!(result.critical_values.len() >= 3);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_detrend_linear() {
        let ts = array![1.0, 3.0, 5.0, 7.0, 9.0]; // Perfect linear trend
        let detrended = detrend_linear(&ts).unwrap();

        // After detrending a perfect linear series, should be approximately zero
        for &val in detrended.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_integration_differentiation_inverse() {
        let original = array![1.0, 3.0, 6.0, 10.0, 15.0];

        // Difference then integrate
        let (differenced, params) = difference_transform(&original, 1, None).unwrap();
        let initial_vals = array![original[0]];
        let integrated = integrate_transform(&differenced, &params, &initial_vals).unwrap();

        // Should recover original (approximately)
        for i in 0..original.len() {
            assert_relative_eq!(integrated[i], original[i], epsilon = 1e-10);
        }
    }
}
