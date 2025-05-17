//! Model diagnostics and validation tools for time series models
//!
//! Implements various diagnostic tests and residual analysis

use ndarray::{Array1, ArrayBase, Data, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};
use crate::utils::{autocorrelation, partial_autocorrelation};

/// Residual diagnostics for time series models
#[derive(Debug, Clone)]
pub struct ResidualDiagnostics<F> {
    /// Residuals
    pub residuals: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Mean of residuals
    pub mean: F,
    /// Standard deviation of residuals
    pub std_dev: F,
    /// Skewness
    pub skewness: F,
    /// Kurtosis
    pub kurtosis: F,
    /// Ljung-Box test results
    pub ljung_box: LjungBoxTest<F>,
    /// Jarque-Bera test for normality
    pub jarque_bera: JarqueBeraTest<F>,
    /// ACF of residuals
    pub acf: Array1<F>,
    /// PACF of residuals
    pub pacf: Array1<F>,
}

/// Ljung-Box test for autocorrelation
#[derive(Debug, Clone)]
pub struct LjungBoxTest<F> {
    /// Test statistic
    pub statistic: F,
    /// P-value
    pub p_value: F,
    /// Degrees of freedom
    pub df: usize,
    /// Number of lags tested
    pub lags: usize,
    /// Whether residuals are white noise
    pub is_white_noise: bool,
}

/// Jarque-Bera test for normality
#[derive(Debug, Clone)]
pub struct JarqueBeraTest<F> {
    /// Test statistic
    pub statistic: F,
    /// P-value
    pub p_value: F,
    /// Whether residuals are normal
    pub is_normal: bool,
}

/// ARCH test for heteroskedasticity
#[derive(Debug, Clone)]
pub struct ArchTest<F> {
    /// Test statistic
    pub statistic: F,
    /// P-value
    pub p_value: F,
    /// Number of lags
    pub lags: usize,
    /// Whether there is ARCH effect
    pub has_arch_effect: bool,
}

/// Model validation results
#[derive(Debug, Clone)]
pub struct ModelValidation<F> {
    /// In-sample fit statistics
    pub in_sample: FitStatistics<F>,
    /// Out-of-sample statistics (if available)
    pub out_of_sample: Option<FitStatistics<F>>,
    /// Cross-validation results
    pub cross_validation: Option<CrossValidationResults<F>>,
}

/// Fit statistics
#[derive(Debug, Clone)]
pub struct FitStatistics<F> {
    /// Mean Absolute Error
    pub mae: F,
    /// Mean Squared Error
    pub mse: F,
    /// Root Mean Squared Error
    pub rmse: F,
    /// Mean Absolute Percentage Error
    pub mape: Option<F>,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: Option<F>,
    /// R-squared
    pub r2: F,
    /// Adjusted R-squared
    pub adj_r2: F,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults<F> {
    /// Average MAE across folds
    pub avg_mae: F,
    /// Average RMSE across folds
    pub avg_rmse: F,
    /// Average MAPE across folds
    pub avg_mape: Option<F>,
    /// MAE for each fold
    pub fold_mae: Vec<F>,
    /// RMSE for each fold
    pub fold_rmse: Vec<F>,
}

/// Perform residual diagnostics
pub fn residual_diagnostics<S, F>(
    residuals: &ArrayBase<S, Ix1>,
    max_lag: Option<usize>,
    alpha: F,
) -> Result<ResidualDiagnostics<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand + ndarray_linalg::Lapack,
{
    scirs2_core::validation::check_array_finite(residuals, "residuals")?;

    if residuals.len() < 4 {
        return Err(TimeSeriesError::InvalidInput(
            "Need at least 4 residuals for diagnostics".to_string(),
        ));
    }

    // Basic statistics
    let mean = residuals.mean().unwrap_or(F::zero());
    let variance = residuals
        .mapv(|x| (x - mean) * (x - mean))
        .mean()
        .unwrap_or(F::zero());
    let std_dev = num_traits::Float::sqrt(variance);

    // Standardized residuals
    let standardized_residuals = if std_dev > F::zero() {
        residuals.mapv(|x| (x - mean) / std_dev)
    } else {
        residuals.to_owned()
    };

    // Skewness and kurtosis
    let (skewness, kurtosis) = calculate_moments(&standardized_residuals)?;

    // ACF and PACF
    let lag_max = max_lag.unwrap_or((residuals.len() as f64).sqrt() as usize);
    let acf = autocorrelation(&residuals.to_owned(), Some(lag_max))?;
    let pacf = partial_autocorrelation(&residuals.to_owned(), Some(lag_max))?;

    // Ljung-Box test
    let ljung_box = ljung_box_test(residuals, lag_max, alpha)?;

    // Jarque-Bera test
    let jarque_bera = jarque_bera_test(&standardized_residuals, alpha)?;

    Ok(ResidualDiagnostics {
        residuals: residuals.to_owned(),
        standardized_residuals,
        mean,
        std_dev,
        skewness,
        kurtosis,
        ljung_box,
        jarque_bera,
        acf,
        pacf,
    })
}

/// Calculate skewness and kurtosis
fn calculate_moments<S, F>(data: &ArrayBase<S, Ix1>) -> Result<(F, F)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Display,
{
    let n = F::from(data.len()).unwrap();
    let mean = data.mean().unwrap_or(F::zero());

    let mut m2 = F::zero();
    let mut m3 = F::zero();
    let mut m4 = F::zero();

    for &x in data.iter() {
        let diff = x - mean;
        let diff2 = diff * diff;
        m2 = m2 + diff2;
        m3 = m3 + diff2 * diff;
        m4 = m4 + diff2 * diff2;
    }

    m2 = m2 / n;
    m3 = m3 / n;
    m4 = m4 / n;

    let skewness = m3 / m2.powf(F::from(1.5).unwrap());
    let kurtosis = m4 / (m2 * m2) - F::from(3.0).unwrap(); // Excess kurtosis

    Ok((skewness, kurtosis))
}

/// Ljung-Box test for autocorrelation
pub fn ljung_box_test<S, F>(
    residuals: &ArrayBase<S, Ix1>,
    lags: usize,
    alpha: F,
) -> Result<LjungBoxTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand + ndarray_linalg::Lapack,
{
    scirs2_core::validation::check_array_finite(residuals, "residuals")?;

    let n = residuals.len();
    if lags >= n {
        return Err(TimeSeriesError::InvalidInput(
            "Number of lags exceeds residual length".to_string(),
        ));
    }

    // Calculate autocorrelations
    let acf = autocorrelation(&residuals.to_owned(), Some(lags))?;

    // Ljung-Box statistic
    let mut statistic = F::zero();
    for k in 1..=lags {
        let rk = acf[k];
        statistic += rk * rk / F::from(n - k).unwrap();
    }
    statistic = F::from(n * (n + 2)).unwrap() * statistic;

    // Calculate p-value using chi-squared distribution
    let df = lags;
    let p_value = chi_squared_pvalue(statistic, df)?;

    let is_white_noise = p_value > alpha;

    Ok(LjungBoxTest {
        statistic,
        p_value,
        df,
        lags,
        is_white_noise,
    })
}

/// Jarque-Bera test for normality
pub fn jarque_bera_test<S, F>(residuals: &ArrayBase<S, Ix1>, alpha: F) -> Result<JarqueBeraTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Display,
{
    let n = F::from(residuals.len()).unwrap();
    let (skewness, kurtosis) = calculate_moments(residuals)?;

    // Jarque-Bera statistic
    let statistic = n / F::from(6.0).unwrap()
        * (skewness * skewness + kurtosis * kurtosis / F::from(4.0).unwrap());

    // P-value using chi-squared distribution with 2 df
    let p_value = chi_squared_pvalue(statistic, 2)?;

    let is_normal = p_value > alpha;

    Ok(JarqueBeraTest {
        statistic,
        p_value,
        is_normal,
    })
}

/// ARCH test for heteroskedasticity
pub fn arch_test<S, F>(residuals: &ArrayBase<S, Ix1>, lags: usize, alpha: F) -> Result<ArchTest<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand + ndarray_linalg::Lapack,
{
    scirs2_core::validation::check_array_finite(residuals, "residuals")?;

    let n = residuals.len();
    if lags >= n {
        return Err(TimeSeriesError::InvalidInput(
            "Number of lags exceeds residual length".to_string(),
        ));
    }

    // Square the residuals
    let squared_residuals = residuals.mapv(|x| x * x);

    // Regress squared residuals on their lags
    use ndarray::Array2;

    let y = squared_residuals.slice(ndarray::s![lags..]).to_owned();
    let mut x = Array2::zeros((n - lags, lags + 1));

    // Add constant
    for i in 0..(n - lags) {
        x[[i, 0]] = F::one();
    }

    // Add lags
    for lag in 1..=lags {
        for i in 0..(n - lags) {
            x[[i, lag]] = squared_residuals[i + lags - lag];
        }
    }

    // Perform regression
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y);

    // Simple matrix inversion for OLS
    let n = xtx.shape()[0];
    if n == 0 {
        return Err(TimeSeriesError::ComputationError(
            "Empty matrix".to_string(),
        ));
    }

    // Regularized pseudo-inverse
    let lambda = F::from(1e-6).unwrap();
    let mut xtx_reg = xtx.clone();
    for i in 0..n {
        xtx_reg[[i, i]] += lambda;
    }

    // Simple matrix solve
    if let Ok(coeffs) = matrix_solve(&xtx_reg, &xty) {
        let fitted = x.dot(&coeffs);
        let residuals_arch = y.clone() - &fitted;

        // Calculate R-squared
        let y_mean = y.mean().unwrap_or(F::zero());
        let ss_tot = y.mapv(|yi| (yi - y_mean) * (yi - y_mean)).sum();
        let ss_res = residuals_arch.dot(&residuals_arch);
        let r2 = if ss_tot > F::zero() {
            F::one() - ss_res / ss_tot
        } else {
            F::zero()
        };

        // LM statistic
        let statistic = F::from(n - lags).unwrap() * r2;

        // P-value using chi-squared distribution
        let p_value = chi_squared_pvalue(statistic, lags)?;

        let has_arch_effect = p_value < alpha;

        Ok(ArchTest {
            statistic,
            p_value,
            lags,
            has_arch_effect,
        })
    } else {
        Err(TimeSeriesError::ComputationError(
            "Failed to perform ARCH test regression".to_string(),
        ))
    }
}

/// Calculate fit statistics
pub fn calculate_fit_statistics<S, F>(
    actual: &ArrayBase<S, Ix1>,
    predicted: &ArrayBase<S, Ix1>,
    n_params: Option<usize>,
) -> Result<FitStatistics<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Display,
{
    scirs2_core::validation::check_array_finite(actual, "actual")?;
    scirs2_core::validation::check_array_finite(predicted, "predicted")?;

    if actual.len() != predicted.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Actual and predicted arrays must have same length".to_string(),
        ));
    }

    let n = actual.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Calculate errors
    let errors = actual - predicted;
    let squared_errors = errors.mapv(|e| e * e);

    // Basic metrics
    let mae = errors.mapv(|e| e.abs()).mean().unwrap();
    let mse = squared_errors.mean().unwrap();
    let rmse = mse.sqrt();

    // MAPE and SMAPE (if no zeros in actual)
    let has_zeros = actual.iter().any(|&x| x == F::zero());
    let (mape, smape) = if !has_zeros {
        let mape = errors
            .iter()
            .zip(actual.iter())
            .map(|(e, a)| (*e / *a).abs())
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n).unwrap();

        let smape = errors
            .iter()
            .zip(actual.iter())
            .zip(predicted.iter())
            .map(|((e, a), p)| F::from(2.0).unwrap() * e.abs() / (a.abs() + p.abs()))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n).unwrap();

        (Some(mape), Some(smape))
    } else {
        (None, None)
    };

    // R-squared
    let y_mean = actual.mean().unwrap();
    let ss_tot = actual.mapv(|y| (y - y_mean) * (y - y_mean)).sum();
    let ss_res = squared_errors.sum();
    let r2 = if ss_tot > F::zero() {
        F::one() - ss_res / ss_tot
    } else {
        F::one()
    };

    // Adjusted R-squared
    let adj_r2 = if let Some(p) = n_params {
        let n_f = F::from(n).unwrap();
        let p_f = F::from(p).unwrap();
        F::one() - (F::one() - r2) * (n_f - F::one()) / (n_f - p_f - F::one())
    } else {
        r2
    };

    Ok(FitStatistics {
        mae,
        mse,
        rmse,
        mape,
        smape,
        r2,
        adj_r2,
    })
}

/// Perform time series cross-validation
pub fn time_series_cv<S, F, Model, Fit, Predict>(
    data: &ArrayBase<S, Ix1>,
    n_folds: usize,
    min_train_size: usize,
    forecast_horizon: usize,
    model: Model,
    fit: Fit,
    predict: Predict,
) -> Result<CrossValidationResults<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display,
    Model: Clone,
    Fit: Fn(&mut Model, &Array1<F>) -> Result<()>,
    Predict: Fn(&Model, usize, &Array1<F>) -> Result<Array1<F>>,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    let n = data.len();
    if n < min_train_size + forecast_horizon {
        return Err(TimeSeriesError::InvalidInput(
            "Insufficient data for cross-validation".to_string(),
        ));
    }

    let fold_size = (n - min_train_size - forecast_horizon) / n_folds;
    if fold_size == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Too many folds for available data".to_string(),
        ));
    }

    let mut fold_mae = Vec::new();
    let mut fold_rmse = Vec::new();
    let mut fold_mape = Vec::new();

    for fold in 0..n_folds {
        let train_end = min_train_size + fold * fold_size;
        let test_end = train_end + forecast_horizon;

        if test_end > n {
            break;
        }

        // Split data
        let train_data = data.slice(ndarray::s![..train_end]).to_owned();
        let test_data = data.slice(ndarray::s![train_end..test_end]).to_owned();

        // Fit model
        let mut fold_model = model.clone();
        fit(&mut fold_model, &train_data)?;

        // Make predictions
        let predictions = predict(&fold_model, forecast_horizon, &train_data)?;

        // Calculate metrics
        let stats = calculate_fit_statistics(&test_data, &predictions, None)?;
        fold_mae.push(stats.mae);
        fold_rmse.push(stats.rmse);
        if let Some(mape) = stats.mape {
            fold_mape.push(mape);
        }
    }

    // Calculate averages
    let avg_mae =
        fold_mae.iter().fold(F::zero(), |acc, x| acc + *x) / F::from(fold_mae.len()).unwrap();
    let avg_rmse =
        fold_rmse.iter().fold(F::zero(), |acc, x| acc + *x) / F::from(fold_rmse.len()).unwrap();
    let avg_mape = if !fold_mape.is_empty() {
        Some(
            fold_mape.iter().fold(F::zero(), |acc, x| acc + *x) / F::from(fold_mape.len()).unwrap(),
        )
    } else {
        None
    };

    Ok(CrossValidationResults {
        avg_mae,
        avg_rmse,
        avg_mape,
        fold_mae,
        fold_rmse,
    })
}

/// Simplified chi-squared p-value calculation
fn chi_squared_pvalue<F>(statistic: F, df: usize) -> Result<F>
where
    F: Float + FromPrimitive + Display,
{
    // This is a simplified implementation
    // In practice, would use a proper statistical library

    // Use normal approximation for large df
    if df > 30 {
        let mean = F::from(df).unwrap();
        let std_dev = (F::from(2 * df).unwrap()).sqrt();
        let z = (statistic - mean) / std_dev;

        // Approximate p-value using standard normal
        if z > F::from(3.0).unwrap() {
            Ok(F::from(0.001).unwrap())
        } else if z > F::from(2.0).unwrap() {
            Ok(F::from(0.05).unwrap())
        } else if z > F::from(1.0).unwrap() {
            Ok(F::from(0.16).unwrap())
        } else {
            Ok(F::from(0.5).unwrap())
        }
    } else {
        // For small df, use simple approximation
        let critical_values = match df {
            1 => (F::from(3.841).unwrap(), F::from(6.635).unwrap()),
            2 => (F::from(5.991).unwrap(), F::from(9.210).unwrap()),
            3 => (F::from(7.815).unwrap(), F::from(11.345).unwrap()),
            4 => (F::from(9.488).unwrap(), F::from(13.277).unwrap()),
            5 => (F::from(11.070).unwrap(), F::from(15.086).unwrap()),
            10 => (F::from(18.307).unwrap(), F::from(23.209).unwrap()),
            _ => (
                F::from(df).unwrap() * F::from(1.5).unwrap(),
                F::from(df).unwrap() * F::from(2.0).unwrap(),
            ),
        };

        if statistic > critical_values.1 {
            Ok(F::from(0.01).unwrap())
        } else if statistic > critical_values.0 {
            Ok(F::from(0.05).unwrap())
        } else {
            Ok(F::from(0.1).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_residual_diagnostics() {
        let residuals = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2];
        let result = residual_diagnostics(&residuals, None, 0.05);
        assert!(result.is_ok());

        let diag = result.unwrap();
        assert!(diag.mean.abs() < 0.1);
        assert!(diag.std_dev > 0.0);
    }

    #[test]
    fn test_ljung_box() {
        let residuals = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2];
        let result = ljung_box_test(&residuals, 3, 0.05);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fit_statistics() {
        let actual = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = array![1.1, 2.1, 2.9, 3.9, 5.1];

        let result = calculate_fit_statistics(&actual, &predicted, Some(2));
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert!(stats.mae > 0.0);
        assert!(stats.rmse > 0.0);
        assert!(stats.r2 > 0.9);
    }
}

/// Simple matrix solve using Gaussian elimination
fn matrix_solve<F>(a: &ndarray::Array2<F>, b: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + ScalarOperand,
{
    let n = a.shape()[0];
    if n != a.shape()[1] || n != b.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix dimensions mismatch".to_string(),
        ));
    }

    // Create augmented matrix [A | b]
    let mut aug = a.clone();
    let mut rhs = b.clone();

    // Forward elimination
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
            for j in 0..n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
            let temp = rhs[i];
            rhs[i] = rhs[max_row];
            rhs[max_row] = temp;
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < F::from(1e-10).unwrap() {
            return Err(TimeSeriesError::ComputationError(
                "Matrix is singular".to_string(),
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..n {
                aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
            }
            rhs[k] = rhs[k] - factor * rhs[i];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}
