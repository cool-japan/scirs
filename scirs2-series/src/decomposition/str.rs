//! Seasonal-Trend decomposition using Regression (STR)

use scirs2_core::ndarray::{s, Array1, Array2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use scirs2_linalg::{inv, solve};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Options for STR (Seasonal-Trend decomposition using Regression)
#[derive(Debug, Clone)]
pub struct STROptions {
    /// Type of regularization to use
    pub regularization_type: RegularizationType,
    /// Regularization parameter for trend
    pub trend_lambda: f64,
    /// Regularization parameter for seasonal components
    pub seasonal_lambda: f64,
    /// Seasonal periods (can include non-integer values)
    pub seasonal_periods: Vec<f64>,
    /// Whether to use robust estimation (less sensitive to outliers)
    pub robust: bool,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Degrees of freedom for the trend
    pub trend_degrees: usize,
    /// Whether to allow the seasonal pattern to change over time
    pub flexible_seasonal: bool,
    /// Number of harmonics for each seasonal component
    pub seasonal_harmonics: Option<Vec<usize>>,
}

/// Type of regularization to use in STR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularizationType {
    /// Ridge regularization (L2 penalty)
    Ridge,
    /// LASSO regularization (L1 penalty)
    Lasso,
    /// Elastic Net regularization (combination of L1 and L2)
    ElasticNet,
}

impl Default for STROptions {
    fn default() -> Self {
        Self {
            regularization_type: RegularizationType::Ridge,
            trend_lambda: 10.0,
            seasonal_lambda: 0.5,
            seasonal_periods: Vec::new(),
            robust: false,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            trend_degrees: 3,
            flexible_seasonal: false,
            seasonal_harmonics: None,
        }
    }
}

/// Result of STR decomposition
#[derive(Debug, Clone)]
pub struct STRResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal components (one for each seasonal period)
    pub seasonal_components: Vec<Array1<F>>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
    /// Confidence intervals for trend (if computed)
    pub trend_ci: Option<(Array1<F>, Array1<F>)>, // (lower, upper)
    /// Confidence intervals for seasonal components (if computed)
    pub seasonal_ci: Option<Vec<(Array1<F>, Array1<F>)>>, // (lower, upper) for each component
}

/// Performs STR (Seasonal-Trend decomposition using Regression) on a time series
///
/// STR uses regularized regression to extract trend and seasonal components from
/// a time series. It allows for multiple seasonal components, non-integer periods,
/// and can provide confidence intervals for the components.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for STR decomposition
///
/// # Returns
///
/// * STR decomposition result
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::decomposition::{str_decomposition, STROptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = STROptions::default();
/// options.seasonal_periods = vec![4.0, 12.0]; // Both quarterly and yearly patterns
///
/// let result = str_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {:?}", result.seasonal_components);
/// println!("Residual: {:?}", result.residual);
/// ```
#[allow(dead_code)]
pub fn str_decomposition<F>(ts: &Array1<F>, options: &STROptions) -> Result<STRResult<F>>
where
    F: Float + FromPrimitive + Debug + ScalarOperand + NumCast + std::iter::Sum,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for STR decomposition".to_string(),
        ));
    }

    if options.seasonal_periods.is_empty() {
        return Err(TimeSeriesError::DecompositionError(
            "At least one seasonal period must be specified for STR".to_string(),
        ));
    }

    for &period in &options.seasonal_periods {
        if period <= 1.0 {
            return Err(TimeSeriesError::DecompositionError(
                "Seasonal periods must be greater than 1".to_string(),
            ));
        }
    }

    if options.trend_lambda < 0.0 || options.seasonal_lambda < 0.0 {
        return Err(TimeSeriesError::DecompositionError(
            "Regularization parameters must be non-negative".to_string(),
        ));
    }

    if options.confidence_level <= 0.0 || options.confidence_level >= 1.0 {
        return Err(TimeSeriesError::DecompositionError(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    // Step 1: Prepare design matrices for trend and seasonal components
    let time_indices: Array1<F> = Array1::from_iter((0..n).map(|i| F::from_usize(i).unwrap()));

    // Trend design matrix using polynomial basis functions
    let trend_degree = options.trend_degrees;
    let mut trend_basis = Array2::zeros((n, trend_degree + 1));

    // Fill trend design matrix with polynomial terms (1, t, t^2, t^3, ...)
    for i in 0..n {
        for j in 0..=trend_degree {
            if j == 0 {
                trend_basis[[i, j]] = F::one(); // Constant term
            } else {
                let time_idx = time_indices[i];
                trend_basis[[i, j]] = Float::powf(time_idx, F::from_usize(j).unwrap());
            }
        }
    }

    // Seasonal design matrices using Fourier basis functions for each seasonal component
    let mut seasonal_bases = Vec::with_capacity(options.seasonal_periods.len());
    let mut total_seasonal_cols = 0;

    for (idx, &period) in options.seasonal_periods.iter().enumerate() {
        // Number of harmonics for this seasonal component
        let harmonics = if let Some(ref harms) = options.seasonal_harmonics {
            harms
                .get(idx)
                .copied()
                .unwrap_or(((period / 2.0).floor() as usize).max(1))
        } else {
            ((period / 2.0).floor() as usize).max(1)
        };

        let mut seasonal_basis = Array2::zeros((n, 2 * harmonics)); // 2 columns per harmonic (sin and cos)

        for i in 0..n {
            let t = time_indices[i];
            for j in 0..harmonics {
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                // Sin term
                seasonal_basis[[i, 2 * j]] = Float::sin(freq * t);
                // Cos term
                seasonal_basis[[i, 2 * j + 1]] = Float::cos(freq * t);
            }
        }

        total_seasonal_cols += 2 * harmonics;
        seasonal_bases.push(seasonal_basis);
    }

    // Step 2: Combine all design matrices
    let total_cols = trend_degree + 1 + total_seasonal_cols;
    let mut design_matrix = Array2::zeros((n, total_cols));

    // Fill trend columns
    design_matrix
        .slice_mut(s![.., 0..=trend_degree])
        .assign(&trend_basis);

    // Fill seasonal columns
    let mut col_offset = trend_degree + 1;
    for seasonal_basis in &seasonal_bases {
        let next_offset = col_offset + seasonal_basis.ncols();
        design_matrix
            .slice_mut(s![.., col_offset..next_offset])
            .assign(seasonal_basis);
        col_offset = next_offset;
    }

    // Step 3: Set up regularization matrix
    let mut regularization_matrix = Array2::zeros((total_cols, total_cols));

    // Trend regularization (penalize higher-order polynomial coefficients)
    for i in 0..=trend_degree {
        let weight = if i == 0 {
            0.0 // Don't penalize the constant term
        } else {
            options.trend_lambda * (i as f64).powi(2)
        };
        regularization_matrix[[i, i]] = F::from_f64(weight).unwrap();
    }

    // Seasonal regularization
    col_offset = trend_degree + 1;
    for seasonal_basis in &seasonal_bases {
        let seasonal_cols = seasonal_basis.ncols();
        for i in 0..seasonal_cols {
            regularization_matrix[[col_offset + i, col_offset + i]] =
                F::from(options.seasonal_lambda).unwrap();
        }
        col_offset += seasonal_cols;
    }

    // Step 4: Solve regularized least squares problem
    // (X^T X + λR) β = X^T y
    let xtx = design_matrix.t().dot(&design_matrix);
    let xty = design_matrix.t().dot(ts);

    // Add regularization
    let system_matrix = xtx + regularization_matrix;

    // Solve the system
    let coefficients = match options.regularization_type {
        RegularizationType::Ridge => {
            // Ridge regression: solve (X^T X + λI) β = X^T y
            solve_regularized_system(&system_matrix, &xty)?
        }
        RegularizationType::Lasso => {
            // LASSO regression using coordinate descent
            solve_lasso(
                &design_matrix,
                ts,
                options.seasonal_lambda,
                1000,
                F::from(1e-6).unwrap(),
            )?
        }
        RegularizationType::ElasticNet => {
            // Elastic Net regression using coordinate descent
            solve_elastic_net(
                &design_matrix,
                ts,
                options.seasonal_lambda,
                options.trend_lambda,
                1000,
                F::from(1e-6).unwrap(),
            )?
        }
    };

    // Step 5: Extract components from coefficients
    // Trend component
    let trend_coeffs = coefficients.slice(s![0..=trend_degree]);
    let trend = trend_basis.dot(&trend_coeffs);

    // Seasonal components
    let mut seasonal_components = Vec::with_capacity(options.seasonal_periods.len());
    col_offset = trend_degree + 1;

    for seasonal_basis in &seasonal_bases {
        let seasonal_cols = seasonal_basis.ncols();
        let seasonal_coeffs = coefficients.slice(s![col_offset..col_offset + seasonal_cols]);
        let seasonal_component = seasonal_basis.dot(&seasonal_coeffs);
        seasonal_components.push(seasonal_component);
        col_offset += seasonal_cols;
    }

    // Compute residuals
    let mut residual = ts.clone();
    for i in 0..n {
        residual[i] = residual[i] - trend[i];
        for seasonal_component in &seasonal_components {
            residual[i] = residual[i] - seasonal_component[i];
        }
    }

    // Compute confidence intervals if requested
    let (trend_ci, seasonal_ci) = if options.compute_confidence_intervals {
        compute_confidence_intervals(
            &design_matrix,
            &system_matrix,
            &residual,
            &trend_basis,
            &seasonal_bases,
            options.confidence_level,
        )?
    } else {
        (None, None)
    };

    // Create result
    let result = STRResult {
        trend,
        seasonal_components,
        residual,
        original: ts.clone(),
        trend_ci,
        seasonal_ci,
    };

    Ok(result)
}

/// Type alias for confidence interval bounds (lower, upper)
type ConfidenceInterval<F> = (Array1<F>, Array1<F>);

/// Type alias for confidence intervals result
type ConfidenceIntervalsResult<F> = Result<(
    Option<ConfidenceInterval<F>>,
    Option<Vec<ConfidenceInterval<F>>>,
)>;

/// Compute confidence intervals for STR components
#[allow(dead_code)]
fn compute_confidence_intervals<F>(
    design_matrix: &Array2<F>,
    system_matrix: &Array2<F>,
    residual: &Array1<F>,
    trend_basis: &Array2<F>,
    seasonal_bases: &[Array2<F>],
    confidence_level: f64,
) -> ConfidenceIntervalsResult<F>
where
    F: Float + FromPrimitive + Debug + ScalarOperand + NumCast + std::iter::Sum,
{
    let n = residual.len();
    let p = design_matrix.ncols();

    if n <= p {
        return Ok((None, None));
    }

    // Estimate residual variance
    let residual_variance = residual.mapv(|x| x * x).sum() / F::from_usize(n - p).unwrap();

    // Compute covariance _matrix: σ² (X^T X + λR)^(-1)
    let covariance_matrix = match matrix_inverse(system_matrix) {
        Ok(inv) => inv.mapv(|x| x * residual_variance),
        Err(_) => return Ok((None, None)), // Skip CI if _matrix is singular
    };

    // Get t-distribution critical value (approximation for large samples)
    let alpha = 1.0 - confidence_level;
    let df = n - p;
    let t_critical = if df > 30 {
        // Normal approximation for large df
        match alpha {
            a if a <= 0.01 => F::from(2.576).unwrap(), // 99% CI
            a if a <= 0.05 => F::from(1.96).unwrap(),  // 95% CI
            _ => F::from(1.645).unwrap(),              // 90% CI
        }
    } else {
        // Simple t-distribution approximation
        let base = F::from(2.0).unwrap();
        base + F::from(df as f64).unwrap().recip()
    };

    // Compute standard errors for trend component
    let trend_se = compute_component_standard_errors(trend_basis, &covariance_matrix)?;
    let trend_margin = trend_se.mapv(|se| se * t_critical);
    let trend_fitted = trend_basis.dot(&covariance_matrix.diag().slice(s![0..trend_basis.ncols()]));
    let trend_lower = &trend_fitted - &trend_margin;
    let trend_upper = &trend_fitted + &trend_margin;

    // Compute standard errors for seasonal components
    let mut seasonal_cis = Vec::new();
    let mut col_offset = trend_basis.ncols();

    for seasonal_basis in seasonal_bases {
        let seasonal_cols = seasonal_basis.ncols();
        let seasonal_cov = covariance_matrix.slice(s![
            col_offset..col_offset + seasonal_cols,
            col_offset..col_offset + seasonal_cols
        ]);

        let seasonal_se =
            compute_component_standard_errors(seasonal_basis, &seasonal_cov.to_owned())?;
        let seasonal_margin = seasonal_se.mapv(|se| se * t_critical);
        let seasonal_fitted = seasonal_basis.dot(&seasonal_cov.diag());
        let seasonal_lower = &seasonal_fitted - &seasonal_margin;
        let seasonal_upper = &seasonal_fitted + &seasonal_margin;

        seasonal_cis.push((seasonal_lower, seasonal_upper));
        col_offset += seasonal_cols;
    }

    Ok((Some((trend_lower, trend_upper)), Some(seasonal_cis)))
}

/// Compute standard errors for a component given its basis and covariance matrix
#[allow(dead_code)]
fn compute_component_standard_errors<F>(
    basis: &Array2<F>,
    covariance: &Array2<F>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + ScalarOperand + NumCast + std::iter::Sum,
{
    let n = basis.nrows();
    let mut standard_errors = Array1::zeros(n);

    for i in 0..n {
        let basis_row = basis.row(i);
        let variance = basis_row.dot(&covariance.dot(&basis_row));
        standard_errors[i] = variance.sqrt();
    }

    Ok(standard_errors)
}

/// Matrix solve using scirs2-linalg
#[allow(dead_code)]
fn solve_regularized_system<F>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + ScalarOperand + NumCast + 'static,
{
    let n = a.shape()[0];
    if n != a.shape()[1] || n != b.len() {
        return Err(TimeSeriesError::DecompositionError(
            "Matrix dimensions mismatch".to_string(),
        ));
    }

    // Convert to f64 for scirs2-linalg computation
    let a_f64 = a.mapv(|x| x.to_f64().unwrap_or(0.0));
    let b_f64 = b.mapv(|x| x.to_f64().unwrap_or(0.0));

    // Solve using scirs2-linalg
    let x_f64 = solve(&a_f64.view(), &b_f64.view(), None)
        .map_err(|e| TimeSeriesError::DecompositionError(format!("Linear solve failed: {e}")))?;

    // Convert back to original type
    let x = x_f64.mapv(|val| F::from_f64(val).unwrap_or_else(F::zero));

    Ok(x)
}

/// LASSO regression using coordinate descent algorithm
#[allow(dead_code)]
fn solve_lasso<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    lambda: f64,
    max_iter: usize,
    tol: F,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + ScalarOperand + NumCast + std::iter::Sum,
{
    let (n, p) = (x.nrows(), x.ncols());
    let mut beta = Array1::zeros(p);
    let lambda_f = F::from(lambda).unwrap();

    // Precompute X^T X diagonal (for efficiency)
    let mut xtx_diag = Array1::zeros(p);
    for j in 0..p {
        xtx_diag[j] = x.column(j).dot(&x.column(j));
    }

    for _iter in 0..max_iter {
        let beta_old = beta.clone();

        for j in 0..p {
            // Compute partial residual
            let mut r = y.clone();
            for k in 0..p {
                if k != j {
                    let x_k = x.column(k);
                    for i in 0..n {
                        r[i] = r[i] - beta[k] * x_k[i];
                    }
                }
            }

            // Compute coordinate update
            let x_j = x.column(j);
            let xty_j = x_j.dot(&r);

            // Soft thresholding
            let z = xty_j;
            beta[j] = if z > lambda_f {
                (z - lambda_f) / xtx_diag[j]
            } else if z < -lambda_f {
                (z + lambda_f) / xtx_diag[j]
            } else {
                F::zero()
            };
        }

        // Check convergence
        let mut diff = F::zero();
        for j in 0..p {
            diff = diff + (beta[j] - beta_old[j]).abs();
        }

        if diff < tol {
            break;
        }
    }

    Ok(beta)
}

/// Elastic Net regression using coordinate descent algorithm
#[allow(dead_code)]
fn solve_elastic_net<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    l1_lambda: f64,
    l2_lambda: f64,
    max_iter: usize,
    tol: F,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + ScalarOperand + NumCast + std::iter::Sum,
{
    let (n, p) = (x.nrows(), x.ncols());
    let mut beta = Array1::zeros(p);
    let l1_lambda_f = F::from(l1_lambda).unwrap();
    let l2_lambda_f = F::from(l2_lambda).unwrap();

    // Precompute X^T X diagonal + L2 penalty
    let mut xtx_diag = Array1::zeros(p);
    for j in 0..p {
        xtx_diag[j] = x.column(j).dot(&x.column(j)) + l2_lambda_f;
    }

    for _iter in 0..max_iter {
        let beta_old = beta.clone();

        for j in 0..p {
            // Compute partial residual
            let mut r = y.clone();
            for k in 0..p {
                if k != j {
                    let x_k = x.column(k);
                    for i in 0..n {
                        r[i] = r[i] - beta[k] * x_k[i];
                    }
                }
            }

            // Compute coordinate update
            let x_j = x.column(j);
            let xty_j = x_j.dot(&r);

            // Soft thresholding with L2 penalty
            let z = xty_j;
            beta[j] = if z > l1_lambda_f {
                (z - l1_lambda_f) / xtx_diag[j]
            } else if z < -l1_lambda_f {
                (z + l1_lambda_f) / xtx_diag[j]
            } else {
                F::zero()
            };
        }

        // Check convergence
        let mut diff = F::zero();
        for j in 0..p {
            diff = diff + (beta[j] - beta_old[j]).abs();
        }

        if diff < tol {
            break;
        }
    }

    Ok(beta)
}

/// Matrix inversion using scirs2-linalg
#[allow(dead_code)]
fn matrix_inverse<F>(a: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + ScalarOperand + NumCast + 'static,
{
    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(TimeSeriesError::DecompositionError(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    // Convert to f64 for scirs2-linalg computation
    let a_f64 = a.mapv(|x| x.to_f64().unwrap_or(0.0));

    // Compute inverse using scirs2-linalg
    let inv_f64 = inv(&a_f64.view(), None).map_err(|e| {
        TimeSeriesError::DecompositionError(format!("Matrix inversion failed: {e}"))
    })?;

    // Convert back to original type
    let inverse = inv_f64.mapv(|val| F::from_f64(val).unwrap_or_else(F::zero));

    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_str_basic() {
        // Create a simple time series with trend and seasonality
        let n = 50;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.1 * i as f64;
            let seasonal = 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = 0.1 * (i as f64 * 0.456).sin();
            ts[i] = trend + seasonal + noise;
        }

        let options = STROptions {
            seasonal_periods: vec![12.0],
            trend_degrees: 2,
            trend_lambda: 1.0,
            seasonal_lambda: 0.1,
            ..Default::default()
        };

        let result = str_decomposition(&ts, &options).unwrap();

        // Check that decomposition sums to original (approximately)
        for i in 0..n {
            let reconstructed =
                result.trend[i] + result.seasonal_components[0][i] + result.residual[i];
            assert_abs_diff_eq!(reconstructed, ts[i], epsilon = 1e-10);
        }

        // Check that we extracted a trend
        assert!(result.trend.len() == n);
        // Check that we extracted seasonal components
        assert!(result.seasonal_components.len() == 1);
        assert!(result.seasonal_components[0].len() == n);
    }

    #[test]
    fn test_str_multiple_seasons() {
        // Create a time series with multiple seasonal patterns
        let n = 100;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.05 * i as f64;
            let seasonal1 = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let seasonal2 = 1.5 * (2.0 * std::f64::consts::PI * i as f64 / 4.0).cos();
            ts[i] = trend + seasonal1 + seasonal2;
        }

        let options = STROptions {
            seasonal_periods: vec![12.0, 4.0],
            trend_degrees: 1,
            trend_lambda: 5.0,
            seasonal_lambda: 0.5,
            ..Default::default()
        };

        let result = str_decomposition(&ts, &options).unwrap();

        // Check that decomposition sums to original
        for i in 0..n {
            let mut reconstructed = result.trend[i] + result.residual[i];
            for seasonal_component in &result.seasonal_components {
                reconstructed += seasonal_component[i];
            }
            assert_abs_diff_eq!(reconstructed, ts[i], epsilon = 1e-10);
        }

        // Check that we have the right number of seasonal components
        assert_eq!(result.seasonal_components.len(), 2);
    }

    #[test]
    fn test_str_edge_cases() {
        // Test with minimum size time series
        let ts = array![1.0, 2.0, 3.0];
        let mut options = STROptions {
            seasonal_periods: vec![2.0],
            ..Default::default()
        };

        let result = str_decomposition(&ts, &options);
        assert!(result.is_ok());

        // Test with invalid seasonal period
        options.seasonal_periods = vec![0.5];
        let result = str_decomposition(&ts, &options);
        assert!(result.is_err());

        // Test with no seasonal periods
        options.seasonal_periods = vec![];
        let result = str_decomposition(&ts, &options);
        assert!(result.is_err());

        // Test with too small time series
        let ts = array![1.0, 2.0];
        options.seasonal_periods = vec![2.0];
        let result = str_decomposition(&ts, &options);
        assert!(result.is_err());
    }
}
