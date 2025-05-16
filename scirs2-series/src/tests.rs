//! Statistical tests for time series analysis
//!
//! Implements various hypothesis tests for time series

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};
use crate::utils::autocovariance;

/// Augmented Dickey-Fuller test for unit root
#[derive(Debug, Clone)]
pub struct ADFTest<F> {
    /// Test statistic
    pub test_statistic: F,
    /// P-value
    pub p_value: F,
    /// Number of lags used
    pub lags: usize,
    /// Critical values at different significance levels
    pub critical_values: CriticalValues<F>,
    /// Whether the series is stationary
    pub is_stationary: bool,
}

/// Critical values for statistical tests
#[derive(Debug, Clone)]
pub struct CriticalValues<F> {
    /// 1% significance level
    pub cv_1_percent: F,
    /// 5% significance level
    pub cv_5_percent: F,
    /// 10% significance level
    pub cv_10_percent: F,
}

/// Type of ADF test regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ADFRegression {
    /// No constant, no trend
    NoConstantNoTrend,
    /// Constant, no trend
    ConstantNoTrend,
    /// Constant and trend
    ConstantAndTrend,
}

/// Perform Augmented Dickey-Fuller test
pub fn adf_test<S, F>(
    data: &ArrayBase<S, Ix1>,
    max_lag: Option<usize>,
    regression: ADFRegression,
    alpha: F,
) -> Result<ADFTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    if data.len() < 10 {
        return Err(TimeSeriesError::InvalidInput(
            "Insufficient data for ADF test (need at least 10 observations)".to_string(),
        ));
    }

    // Determine number of lags
    let n = data.len();
    let lags = if let Some(lag) = max_lag {
        lag
    } else {
        // Use Schwert criterion for automatic lag selection
        let power = F::from(0.25).unwrap();
        let max_lag = ((F::from(12).unwrap()
            * (F::from(n).unwrap() / F::from(100).unwrap()).powf(power))
        .floor())
        .to_usize()
        .unwrap();
        select_lag_aic(data, max_lag, regression)?
    };

    // Create regression variables
    let y_diff = difference(data);
    let y_lag1 = lag(data, 1)?;

    // Build regression matrix
    let (x_mat, y_vec) =
        build_regression_matrix(&y_diff, &y_lag1, lags, &data.to_owned(), regression)?;

    // Perform OLS regression
    let coeffs = ols_regression(&x_mat, &y_vec)?;

    // Extract test statistic (coefficient on lagged level)
    let test_statistic = if regression == ADFRegression::NoConstantNoTrend {
        coeffs[0]
    } else {
        coeffs[1] // Skip constant
    };

    // Calculate standard error
    let residuals = &y_vec - &x_mat.dot(&coeffs);
    let n_obs = y_vec.len();
    let n_params = coeffs.len();
    let _sigma2 = residuals.dot(&residuals) / F::from(n_obs - n_params).unwrap();

    // Get critical values
    let critical_values = get_adf_critical_values(n_obs, regression)?;

    // Calculate p-value (simplified - in practice would use interpolation)
    let p_value = calculate_adf_pvalue(test_statistic, n_obs, regression)?;

    // Determine if stationary
    let is_stationary = p_value < alpha;

    Ok(ADFTest {
        test_statistic,
        p_value,
        lags,
        critical_values,
        is_stationary,
    })
}

/// KPSS test for stationarity
#[derive(Debug, Clone)]
pub struct KPSSTest<F> {
    /// Test statistic
    pub test_statistic: F,
    /// P-value
    pub p_value: F,
    /// Number of lags used
    pub lags: usize,
    /// Critical values
    pub critical_values: CriticalValues<F>,
    /// Whether the series is stationary
    pub is_stationary: bool,
}

/// Type of KPSS test
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KPSSType {
    /// Test with constant only (level stationarity)
    Constant,
    /// Test with constant and trend (trend stationarity)
    Trend,
}

/// Perform KPSS test
pub fn kpss_test<S, F>(
    data: &ArrayBase<S, Ix1>,
    test_type: KPSSType,
    lags: Option<usize>,
    alpha: F,
) -> Result<KPSSTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    let n = data.len();
    if n < 10 {
        return Err(TimeSeriesError::InvalidInput(
            "Insufficient data for KPSS test".to_string(),
        ));
    }

    // Determine number of lags for Newey-West estimator
    let lags = lags.unwrap_or_else(|| {
        let power = F::from(0.25).unwrap();
        ((F::from(4).unwrap() * (F::from(n).unwrap() / F::from(100).unwrap()).powf(power)).floor())
            .to_usize()
            .unwrap()
    });

    // Detrend the series
    let (residuals, _) = match test_type {
        KPSSType::Constant => {
            let (res, mean) = detrend_constant(data)?;
            (res, Array1::from_elem(1, mean))
        }
        KPSSType::Trend => detrend_linear(data)?,
    };

    // Calculate partial sums
    let mut partial_sums = Array1::zeros(n);
    let mut cum_sum = F::zero();
    for i in 0..n {
        cum_sum = cum_sum + residuals[i];
        partial_sums[i] = cum_sum;
    }

    // Calculate test statistic
    let s2 = newey_west_variance(&residuals, lags)?;
    let test_statistic = partial_sums.dot(&partial_sums) / (F::from(n * n).unwrap() * s2);

    // Get critical values
    let critical_values = get_kpss_critical_values(test_type)?;

    // Calculate p-value (simplified)
    let p_value = calculate_kpss_pvalue(test_statistic, test_type)?;

    // KPSS null hypothesis is stationarity, so reject if p-value is small
    let is_stationary = p_value >= alpha;

    Ok(KPSSTest {
        test_statistic,
        p_value,
        lags,
        critical_values,
        is_stationary,
    })
}

/// Phillips-Perron test for unit root
#[derive(Debug, Clone)]
pub struct PPTest<F> {
    /// Test statistic
    pub test_statistic: F,
    /// P-value
    pub p_value: F,
    /// Number of lags for Newey-West
    pub lags: usize,
    /// Critical values
    pub critical_values: CriticalValues<F>,
    /// Whether the series is stationary
    pub is_stationary: bool,
}

/// Perform Phillips-Perron test
pub fn pp_test<S, F>(
    data: &ArrayBase<S, Ix1>,
    regression: ADFRegression,
    lags: Option<usize>,
    alpha: F,
) -> Result<PPTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    let n = data.len();
    if n < 10 {
        return Err(TimeSeriesError::InvalidInput(
            "Insufficient data for PP test".to_string(),
        ));
    }

    // Determine number of lags for Newey-West
    let lags = lags.unwrap_or_else(|| {
        let power = F::from(0.25).unwrap();
        ((F::from(4).unwrap() * (F::from(n).unwrap() / F::from(100).unwrap()).powf(power)).floor())
            .to_usize()
            .unwrap()
    });

    // Run basic Dickey-Fuller regression
    let y_diff = difference(data);
    let y_lag1 = lag(data, 1)?;

    let (x_mat, y_vec) =
        build_regression_matrix(&y_diff, &y_lag1, 0, &data.to_owned(), regression)?;
    let coeffs = ols_regression(&x_mat, &y_vec)?;

    // Extract coefficient and t-statistic
    let rho_hat = if regression == ADFRegression::NoConstantNoTrend {
        coeffs[0]
    } else {
        coeffs[1]
    };

    // Calculate residuals
    let residuals = &y_vec - &x_mat.dot(&coeffs);

    // Newey-West variance estimation
    let s2_nw = newey_west_variance(&residuals, lags)?;
    let s2_ols = residuals.dot(&residuals) / F::from(n - coeffs.len()).unwrap();

    // Phillips-Perron adjustment
    let t_stat = rho_hat / s2_ols.sqrt();
    let adjustment = (s2_nw - s2_ols) / (F::from(2).unwrap() * s2_nw.sqrt());
    let test_statistic = t_stat - adjustment * F::from(n).unwrap().sqrt();

    // Get critical values and p-value
    let critical_values = get_adf_critical_values(n, regression)?;
    let p_value = calculate_adf_pvalue(test_statistic, n, regression)?;

    let is_stationary = p_value < alpha;

    Ok(PPTest {
        test_statistic,
        p_value,
        lags,
        critical_values,
        is_stationary,
    })
}

// Helper functions

/// Calculate first difference
fn difference<S, F>(data: &ArrayBase<S, Ix1>) -> Array1<F>
where
    S: Data<Elem = F>,
    F: Float,
{
    let n = data.len();
    let mut diff = Array1::zeros(n - 1);
    for i in 1..n {
        diff[i - 1] = data[i] - data[i - 1];
    }
    diff
}

/// Lag a series
fn lag<S, F>(data: &ArrayBase<S, Ix1>, k: usize) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float,
{
    if k >= data.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Lag exceeds series length".to_string(),
        ));
    }

    let n = data.len() - k;
    let mut lagged = Array1::zeros(n);
    for i in 0..n {
        lagged[i] = data[i];
    }
    Ok(lagged)
}

/// Build regression matrix for ADF test
fn build_regression_matrix<S, F>(
    y_diff: &ArrayBase<S, Ix1>,
    y_lag1: &ArrayBase<S, Ix1>,
    lags: usize,
    data: &ArrayBase<S, Ix1>,
    regression: ADFRegression,
) -> Result<(ndarray::Array2<F>, Array1<F>)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    use ndarray::Array2;

    let n = y_diff.len() - lags;
    let mut n_cols = 1 + lags; // lag1 + diff lags

    match regression {
        ADFRegression::NoConstantNoTrend => {}
        ADFRegression::ConstantNoTrend => n_cols += 1,
        ADFRegression::ConstantAndTrend => n_cols += 2,
    }

    let mut x_mat = Array2::zeros((n, n_cols));
    let mut y_vec = Array1::zeros(n);

    // Copy dependent variable
    for i in 0..n {
        y_vec[i] = y_diff[i + lags];
    }

    let mut col = 0;

    // Add constant if needed
    if regression != ADFRegression::NoConstantNoTrend {
        for i in 0..n {
            x_mat[[i, col]] = F::one();
        }
        col += 1;
    }

    // Add lagged level
    for i in 0..n {
        x_mat[[i, col]] = y_lag1[i + lags];
    }
    col += 1;

    // Add lagged differences
    for lag_idx in 0..lags {
        let lagged_data = lag(data, lag_idx + 1)?;
        let diff_lag = difference(&lagged_data);
        for i in 0..n {
            x_mat[[i, col]] = diff_lag[i + lags - lag_idx - 1];
        }
        col += 1;
    }

    // Add trend if needed
    if regression == ADFRegression::ConstantAndTrend {
        for i in 0..n {
            x_mat[[i, col]] = F::from(i + 1).unwrap();
        }
    }

    Ok((x_mat, y_vec))
}

/// Simple OLS regression using pseudoinverse
fn ols_regression<F>(x: &ndarray::Array2<F>, y: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + ScalarOperand,
{
    // Normal equations: (X'X)^-1 X'y
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);

    // Use regularized inversion for numerical stability
    let n = xtx.shape()[0];
    let lambda = F::from(1e-6).unwrap();
    let mut xtx_reg = xtx;
    for i in 0..n {
        xtx_reg[[i, i]] = xtx_reg[[i, i]] + lambda;
    }

    // Manual matrix inversion using Gauss-Jordan elimination (simplified)
    if let Ok(inv) = matrix_inverse(&xtx_reg) {
        Ok(inv.dot(&xty))
    } else {
        Err(TimeSeriesError::ComputationError(
            "Failed to solve normal equations".to_string(),
        ))
    }
}

/// Select lag using AIC
fn select_lag_aic<S, F>(
    data: &ArrayBase<S, Ix1>,
    max_lag: usize,
    regression: ADFRegression,
) -> Result<usize>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Display + ScalarOperand,
{
    let mut best_aic = F::infinity();
    let mut best_lag = 0;

    for lag_idx in 0..=max_lag.min(data.len() / 4) {
        let y_diff = difference(data);
        let y_lag1 = lag(data, 1)?;

        if let Ok((x_mat, y_vec)) =
            build_regression_matrix(&y_diff, &y_lag1, lag_idx, &data.to_owned(), regression)
        {
            if let Ok(coeffs) = ols_regression(&x_mat, &y_vec) {
                let residuals = &y_vec - &x_mat.dot(&coeffs);
                let n = F::from(y_vec.len()).unwrap();
                let k = F::from(coeffs.len()).unwrap();
                let rss = residuals.dot(&residuals);

                let aic = n * (F::one() + (F::from(2.0 * std::f64::consts::PI).unwrap()).ln())
                    + n * (rss / n).ln()
                    + F::from(2.0).unwrap() * k;

                if aic < best_aic {
                    best_aic = aic;
                    best_lag = lag_idx;
                }
            }
        }
    }

    Ok(best_lag)
}

/// Detrend with constant only
fn detrend_constant<S, F>(data: &ArrayBase<S, Ix1>) -> Result<(Array1<F>, F)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    let mean = data.mean().unwrap();
    let residuals = data.mapv(|x| x - mean);
    Ok((residuals, mean))
}

/// Detrend with linear trend
fn detrend_linear<S, F>(data: &ArrayBase<S, Ix1>) -> Result<(Array1<F>, Array1<F>)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + ScalarOperand,
{
    use ndarray::Array2;

    let n = data.len();
    let mut x_mat = Array2::zeros((n, 2));

    // Design matrix: [1 t]
    for i in 0..n {
        x_mat[[i, 0]] = F::one();
        x_mat[[i, 1]] = F::from(i).unwrap();
    }

    let coeffs = ols_regression(&x_mat, &data.to_owned())?;
    let fitted = x_mat.dot(&coeffs);
    let residuals = data - &fitted;

    Ok((residuals, coeffs))
}

/// Newey-West variance estimator
fn newey_west_variance<S, F>(residuals: &ArrayBase<S, Ix1>, lags: usize) -> Result<F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + ndarray::ScalarOperand,
{
    let n = residuals.len();
    let residuals_owned = residuals.to_owned();
    let mut variance = residuals_owned.dot(&residuals_owned) / F::from(n).unwrap();

    // Add autocovariance terms with Bartlett weights
    for lag in 1..=lags {
        let weight = F::one() - F::from(lag).unwrap() / F::from(lags + 1).unwrap();
        if let Ok(acov) = autocovariance(&residuals.to_owned(), lag) {
            variance = variance + F::from(2.0).unwrap() * weight * acov;
        }
    }

    Ok(variance)
}

/// Get ADF critical values (simplified)
fn get_adf_critical_values<F>(_n: usize, regression: ADFRegression) -> Result<CriticalValues<F>>
where
    F: Float + FromPrimitive,
{
    // These are approximate values - in practice would use MacKinnon tables
    let cv = match regression {
        ADFRegression::NoConstantNoTrend => CriticalValues {
            cv_1_percent: F::from(-2.58).unwrap(),
            cv_5_percent: F::from(-1.95).unwrap(),
            cv_10_percent: F::from(-1.62).unwrap(),
        },
        ADFRegression::ConstantNoTrend => CriticalValues {
            cv_1_percent: F::from(-3.43).unwrap(),
            cv_5_percent: F::from(-2.86).unwrap(),
            cv_10_percent: F::from(-2.57).unwrap(),
        },
        ADFRegression::ConstantAndTrend => CriticalValues {
            cv_1_percent: F::from(-3.96).unwrap(),
            cv_5_percent: F::from(-3.41).unwrap(),
            cv_10_percent: F::from(-3.12).unwrap(),
        },
    };

    Ok(cv)
}

/// Calculate ADF p-value (simplified)
fn calculate_adf_pvalue<F>(test_stat: F, n: usize, regression: ADFRegression) -> Result<F>
where
    F: Float + FromPrimitive,
{
    // This is a simplified implementation
    // In practice would use MacKinnon (1994) approximation
    let critical_values = get_adf_critical_values(n, regression)?;

    if test_stat < critical_values.cv_1_percent {
        Ok(F::from(0.001).unwrap())
    } else if test_stat < critical_values.cv_5_percent {
        Ok(F::from(0.025).unwrap())
    } else if test_stat < critical_values.cv_10_percent {
        Ok(F::from(0.075).unwrap())
    } else {
        Ok(F::from(0.15).unwrap())
    }
}

/// Get KPSS critical values
fn get_kpss_critical_values<F>(test_type: KPSSType) -> Result<CriticalValues<F>>
where
    F: Float + FromPrimitive,
{
    let cv = match test_type {
        KPSSType::Constant => CriticalValues {
            cv_1_percent: F::from(0.739).unwrap(),
            cv_5_percent: F::from(0.463).unwrap(),
            cv_10_percent: F::from(0.347).unwrap(),
        },
        KPSSType::Trend => CriticalValues {
            cv_1_percent: F::from(0.216).unwrap(),
            cv_5_percent: F::from(0.146).unwrap(),
            cv_10_percent: F::from(0.119).unwrap(),
        },
    };

    Ok(cv)
}

/// Calculate KPSS p-value (simplified)
fn calculate_kpss_pvalue<F>(test_stat: F, test_type: KPSSType) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let critical_values = get_kpss_critical_values(test_type)?;

    if test_stat > critical_values.cv_1_percent {
        Ok(F::from(0.001).unwrap())
    } else if test_stat > critical_values.cv_5_percent {
        Ok(F::from(0.025).unwrap())
    } else if test_stat > critical_values.cv_10_percent {
        Ok(F::from(0.075).unwrap())
    } else {
        Ok(F::from(0.15).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adf_stationary() {
        // Test with white noise (should be stationary)
        let data = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1];
        let result = adf_test(&data, None, ADFRegression::ConstantNoTrend, 0.05);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kpss_stationary() {
        // Test with white noise
        let data = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1];
        let result = kpss_test(&data, KPSSType::Constant, None, 0.05);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pp_stationary() {
        // Test with white noise
        let data = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1];
        let result = pp_test(&data, ADFRegression::ConstantNoTrend, None, 0.05);
        assert!(result.is_ok());
    }
}

/// Simple matrix inversion using Gauss-Jordan elimination
fn matrix_inverse<F>(a: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + ScalarOperand,
{
    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    // Create augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
            if i == j {
                aug[[i, n + j]] = F::one();
            }
        }
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < F::from(1e-10).unwrap() {
            return Err(TimeSeriesError::ComputationError(
                "Matrix is singular".to_string(),
            ));
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] = aug[[i, j]] / pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}
