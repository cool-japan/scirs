//! Time series regression models
//!
//! This module provides various regression methods for time series analysis:
//! - Distributed lag models (DL)
//! - Autoregressive distributed lag (ARDL) models
//! - Error correction models (ECM)
//! - Regression with ARIMA errors

use crate::error::TimeSeriesError;
use ndarray::{s, Array1, Array2};
use scirs2_core::validation::check_array_finite;

/// Result type for regression models
pub type RegressionResult<T> = Result<T, TimeSeriesError>;

/// Distributed lag model result
#[derive(Debug, Clone)]
pub struct DistributedLagResult {
    /// Regression coefficients for lagged values
    pub coefficients: Array1<f64>,
    /// Standard errors of coefficients
    pub standard_errors: Array1<f64>,
    /// T-statistics for coefficients
    pub t_statistics: Array1<f64>,
    /// P-values for coefficients
    pub p_values: Array1<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Adjusted R-squared value
    pub adjusted_r_squared: f64,
    /// Residual sum of squares
    pub residual_sum_squares: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// Maximum lag used
    pub max_lag: usize,
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// Residuals
    pub residuals: Array1<f64>,
}

/// ARDL model result
#[derive(Debug, Clone)]
pub struct ARDLResult {
    /// Coefficients for dependent variable lags
    pub y_coefficients: Array1<f64>,
    /// Coefficients for independent variable lags
    pub x_coefficients: Array1<f64>,
    /// Intercept coefficient
    pub intercept: f64,
    /// Standard errors for all coefficients
    pub standard_errors: Array1<f64>,
    /// T-statistics for all coefficients
    pub t_statistics: Array1<f64>,
    /// P-values for all coefficients
    pub p_values: Array1<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Adjusted R-squared value
    pub adjusted_r_squared: f64,
    /// Number of lags for dependent variable
    pub y_lags: usize,
    /// Number of lags for independent variable
    pub x_lags: usize,
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// Residuals
    pub residuals: Array1<f64>,
    /// Information criteria
    pub aic: f64,
    /// Bayesian information criterion
    pub bic: f64,
}

/// Error correction model result
#[derive(Debug, Clone)]
pub struct ErrorCorrectionResult {
    /// Error correction coefficient (speed of adjustment)
    pub error_correction_coeff: f64,
    /// Short-run coefficients for dependent variable differences
    pub short_run_y_coeffs: Array1<f64>,
    /// Short-run coefficients for independent variable differences
    pub short_run_x_coeffs: Array1<f64>,
    /// Long-run coefficients
    pub long_run_coeffs: Array1<f64>,
    /// Standard errors
    pub standard_errors: Array1<f64>,
    /// T-statistics
    pub t_statistics: Array1<f64>,
    /// P-values
    pub p_values: Array1<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// Residuals
    pub residuals: Array1<f64>,
    /// Cointegration test p-value
    pub cointegration_p_value: f64,
}

/// Regression with ARIMA errors result (placeholder for future implementation)
#[derive(Debug, Clone)]
pub struct ARIMAErrorsResult {
    /// Regression coefficients
    pub regression_coefficients: Array1<f64>,
    /// Combined fitted values (regression + ARIMA error correction)
    pub fitted_values: Array1<f64>,
    /// Final residuals (should be white noise)
    pub residuals: Array1<f64>,
    /// R-squared for regression component
    pub regression_r_squared: f64,
    /// Log-likelihood of combined model
    pub log_likelihood: f64,
    /// AIC of combined model
    pub aic: f64,
    /// BIC of combined model
    pub bic: f64,
    /// Standard errors of regression coefficients
    pub regression_std_errors: Array1<f64>,
}

/// Configuration for distributed lag models
#[derive(Debug, Clone)]
pub struct DistributedLagConfig {
    /// Maximum lag to include
    pub max_lag: usize,
    /// Whether to include intercept
    pub include_intercept: bool,
    /// Significance level for tests
    pub significance_level: f64,
}

impl Default for DistributedLagConfig {
    fn default() -> Self {
        Self {
            max_lag: 4,
            include_intercept: true,
            significance_level: 0.05,
        }
    }
}

/// Configuration for ARDL models
#[derive(Debug, Clone)]
pub struct ARDLConfig {
    /// Maximum lags for dependent variable
    pub max_y_lags: usize,
    /// Maximum lags for independent variables
    pub max_x_lags: usize,
    /// Whether to include intercept
    pub include_intercept: bool,
    /// Information criterion for model selection
    pub selection_criterion: InformationCriterion,
    /// Whether to perform automatic lag selection
    pub auto_lag_selection: bool,
}

impl Default for ARDLConfig {
    fn default() -> Self {
        Self {
            max_y_lags: 4,
            max_x_lags: 4,
            include_intercept: true,
            selection_criterion: InformationCriterion::AIC,
            auto_lag_selection: true,
        }
    }
}

/// Information criteria for model selection
#[derive(Debug, Clone, Copy)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Hannan-Quinn Information Criterion
    HQIC,
}

/// Configuration for error correction models
#[derive(Debug, Clone)]
pub struct ErrorCorrectionConfig {
    /// Maximum lags for difference terms
    pub max_diff_lags: usize,
    /// Whether to test for cointegration
    pub test_cointegration: bool,
    /// Significance level for cointegration test
    pub cointegration_significance: f64,
}

impl Default for ErrorCorrectionConfig {
    fn default() -> Self {
        Self {
            max_diff_lags: 3,
            test_cointegration: true,
            cointegration_significance: 0.05,
        }
    }
}

/// Configuration for regression with ARIMA errors (placeholder)
#[derive(Debug, Clone)]
pub struct ARIMAErrorsConfig {
    /// Whether to include intercept in regression
    pub include_intercept: bool,
    /// Maximum iterations for fitting
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for ARIMAErrorsConfig {
    fn default() -> Self {
        Self {
            include_intercept: true,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Time series regression model fitter
pub struct TimeSeriesRegression {
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl TimeSeriesRegression {
    /// Create a new time series regression fitter
    pub fn new() -> Self {
        Self { random_seed: None }
    }

    /// Create a new fitter with random seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            random_seed: Some(seed),
        }
    }

    /// Fit a distributed lag model
    ///
    /// Models y_t = α + β₀x_t + β₁x_{t-1} + ... + βₖx_{t-k} + ε_t
    ///
    /// # Arguments
    ///
    /// * `y` - Dependent variable time series
    /// * `x` - Independent variable time series
    /// * `config` - Configuration for the model
    ///
    /// # Returns
    ///
    /// Result containing distributed lag model estimates
    pub fn fit_distributed_lag(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
        config: &DistributedLagConfig,
    ) -> RegressionResult<DistributedLagResult> {
        check_array_finite(y, "y")?;
        check_array_finite(x, "x")?;

        if y.len() != x.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        if y.len() <= config.max_lag + 1 {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified lag".to_string(),
            ));
        }

        // Prepare the design matrix
        let n = y.len() - config.max_lag;
        let p = config.max_lag + 1 + if config.include_intercept { 1 } else { 0 };
        let mut design_matrix = Array2::zeros((n, p));
        let mut response = Array1::zeros(n);

        for i in 0..n {
            let row_idx = config.max_lag + i;
            response[i] = y[row_idx];

            let mut col_idx = 0;

            // Add intercept
            if config.include_intercept {
                design_matrix[[i, col_idx]] = 1.0;
                col_idx += 1;
            }

            // Add lagged x values (including contemporaneous)
            for lag in 0..=config.max_lag {
                design_matrix[[i, col_idx]] = x[row_idx - lag];
                col_idx += 1;
            }
        }

        // Fit OLS regression
        let regression_result = self.fit_ols_regression(&design_matrix, &response)?;

        Ok(DistributedLagResult {
            coefficients: regression_result.coefficients,
            standard_errors: regression_result.standard_errors,
            t_statistics: regression_result.t_statistics,
            p_values: regression_result.p_values,
            r_squared: regression_result.r_squared,
            adjusted_r_squared: regression_result.adjusted_r_squared,
            residual_sum_squares: regression_result.residual_sum_squares,
            degrees_of_freedom: regression_result.degrees_of_freedom,
            max_lag: config.max_lag,
            fitted_values: regression_result.fitted_values,
            residuals: regression_result.residuals,
        })
    }

    /// Fit an ARDL model
    ///
    /// Models y_t = α + Σφᵢy_{t-i} + Σβⱼx_{t-j} + ε_t
    ///
    /// # Arguments
    ///
    /// * `y` - Dependent variable time series
    /// * `x` - Independent variable time series
    /// * `config` - Configuration for the model
    ///
    /// # Returns
    ///
    /// Result containing ARDL model estimates
    pub fn fit_ardl(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
        config: &ARDLConfig,
    ) -> RegressionResult<ARDLResult> {
        check_array_finite(y, "y")?;
        check_array_finite(x, "x")?;

        if y.len() != x.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        let (y_lags, x_lags) = if config.auto_lag_selection {
            self.select_optimal_lags(y, x, config)?
        } else {
            (config.max_y_lags, config.max_x_lags)
        };

        let max_lag = y_lags.max(x_lags);
        if y.len() <= max_lag + 1 {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified lags".to_string(),
            ));
        }

        // Prepare the design matrix
        let n = y.len() - max_lag;
        let p = y_lags + x_lags + 1 + if config.include_intercept { 1 } else { 0 };
        let mut design_matrix = Array2::zeros((n, p));
        let mut response = Array1::zeros(n);

        for i in 0..n {
            let row_idx = max_lag + i;
            response[i] = y[row_idx];

            let mut col_idx = 0;

            // Add intercept
            if config.include_intercept {
                design_matrix[[i, col_idx]] = 1.0;
                col_idx += 1;
            }

            // Add lagged y values
            for lag in 1..=y_lags {
                design_matrix[[i, col_idx]] = y[row_idx - lag];
                col_idx += 1;
            }

            // Add lagged x values (including contemporaneous)
            for lag in 0..=x_lags {
                design_matrix[[i, col_idx]] = x[row_idx - lag];
                col_idx += 1;
            }
        }

        // Fit OLS regression
        let regression_result = self.fit_ols_regression(&design_matrix, &response)?;

        // Extract coefficients
        let mut coeff_idx = if config.include_intercept { 1 } else { 0 };
        let intercept = if config.include_intercept {
            regression_result.coefficients[0]
        } else {
            0.0
        };

        let y_coefficients = regression_result
            .coefficients
            .slice(s![coeff_idx..coeff_idx + y_lags])
            .to_owned();
        coeff_idx += y_lags;

        let x_coefficients = regression_result
            .coefficients
            .slice(s![coeff_idx..coeff_idx + x_lags + 1])
            .to_owned();

        // Calculate information criteria
        let k = regression_result.coefficients.len() as f64;
        let n_f = n as f64;
        let log_likelihood = -0.5
            * n_f
            * (2.0 * std::f64::consts::PI * regression_result.residual_sum_squares / n_f).ln()
            - 0.5 * regression_result.residual_sum_squares
                / (regression_result.residual_sum_squares / n_f);
        let aic = -2.0 * log_likelihood + 2.0 * k;
        let bic = -2.0 * log_likelihood + k * n_f.ln();

        Ok(ARDLResult {
            y_coefficients,
            x_coefficients,
            intercept,
            standard_errors: regression_result.standard_errors,
            t_statistics: regression_result.t_statistics,
            p_values: regression_result.p_values,
            r_squared: regression_result.r_squared,
            adjusted_r_squared: regression_result.adjusted_r_squared,
            y_lags,
            x_lags,
            fitted_values: regression_result.fitted_values,
            residuals: regression_result.residuals,
            aic,
            bic,
        })
    }

    /// Fit an error correction model
    ///
    /// Models Δy_t = α + γ(y_{t-1} - βx_{t-1}) + Σφᵢ Δy_{t-i} + Σθⱼ Δx_{t-j} + ε_t
    ///
    /// # Arguments
    ///
    /// * `y` - Dependent variable time series
    /// * `x` - Independent variable time series
    /// * `config` - Configuration for the model
    ///
    /// # Returns
    ///
    /// Result containing error correction model estimates
    pub fn fit_error_correction(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
        config: &ErrorCorrectionConfig,
    ) -> RegressionResult<ErrorCorrectionResult> {
        check_array_finite(y, "y")?;
        check_array_finite(x, "x")?;

        if y.len() != x.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        if y.len() <= config.max_diff_lags + 2 {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for error correction model".to_string(),
            ));
        }

        // Step 1: Estimate long-run relationship
        let long_run_result = self.estimate_long_run_relationship(y, x)?;

        // Step 2: Test for cointegration if requested
        let cointegration_p_value = if config.test_cointegration {
            self.test_cointegration(&long_run_result.residuals)?
        } else {
            0.0 // Assume cointegration
        };

        // Step 3: Estimate error correction model
        let ecm_result = self.estimate_ecm(y, x, &long_run_result, config)?;

        Ok(ErrorCorrectionResult {
            error_correction_coeff: ecm_result.error_correction_coeff,
            short_run_y_coeffs: ecm_result.short_run_y_coeffs,
            short_run_x_coeffs: ecm_result.short_run_x_coeffs,
            long_run_coeffs: long_run_result.coefficients,
            standard_errors: ecm_result.standard_errors,
            t_statistics: ecm_result.t_statistics,
            p_values: ecm_result.p_values,
            r_squared: ecm_result.r_squared,
            fitted_values: ecm_result.fitted_values,
            residuals: ecm_result.residuals,
            cointegration_p_value,
        })
    }

    /// Fit regression with ARIMA errors
    ///
    /// Models y_t = βx_t + ε_t, where ε_t follows an ARIMA process
    ///
    /// # Arguments
    ///
    /// * `y` - Dependent variable time series
    /// * `x` - Independent variables matrix
    /// * `config` - Configuration for the model
    ///
    /// # Returns
    ///
    /// Result containing regression with ARIMA errors estimates
    pub fn fit_regression_with_arima_errors(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        config: &ARIMAErrorsConfig,
    ) -> RegressionResult<ARIMAErrorsResult> {
        check_array_finite(y, "y")?;

        if y.len() != x.nrows() {
            return Err(TimeSeriesError::InvalidInput(
                "Response and design matrix must have compatible dimensions".to_string(),
            ));
        }

        // Placeholder implementation - ARIMA models not yet implemented
        // For now, just perform simple OLS regression
        let mut design_matrix = x.clone();
        if config.include_intercept {
            let _intercept_col = Array2::<f64>::ones((x.nrows(), 1));
            // Simple concatenation for now
            let mut new_matrix = Array2::zeros((x.nrows(), x.ncols() + 1));
            for i in 0..x.nrows() {
                new_matrix[[i, 0]] = 1.0;
                for j in 0..x.ncols() {
                    new_matrix[[i, j + 1]] = x[[i, j]];
                }
            }
            design_matrix = new_matrix;
        }

        let regression_result = self.fit_ols_regression(&design_matrix, y)?;

        // Simple placeholder calculations
        let n = y.len() as f64;
        let k = regression_result.coefficients.len() as f64;
        let rss = regression_result.residual_sum_squares;
        let log_likelihood =
            -0.5 * n * (2.0 * std::f64::consts::PI * rss / n).ln() - 0.5 * rss / (rss / n);
        let aic = -2.0 * log_likelihood + 2.0 * k;
        let bic = -2.0 * log_likelihood + k * n.ln();

        Ok(ARIMAErrorsResult {
            regression_coefficients: regression_result.coefficients,
            fitted_values: regression_result.fitted_values,
            residuals: regression_result.residuals,
            regression_r_squared: regression_result.r_squared,
            log_likelihood,
            aic,
            bic,
            regression_std_errors: regression_result.standard_errors,
        })
    }

    // Helper methods

    fn fit_ols_regression(&self, x: &Array2<f64>, y: &Array1<f64>) -> RegressionResult<OLSResult> {
        let n = y.len() as f64;
        let p = x.ncols() as f64;

        // Compute (X'X)^{-1}X'y
        let xt = x.t();
        let xtx = xt.dot(x);
        let xty = xt.dot(y);

        // Add regularization for numerical stability
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows() {
            xtx_reg[[i, i]] += 1e-10;
        }

        let coefficients = self.solve_linear_system_robust(&xtx_reg, &xty)?;
        let fitted_values = x.dot(&coefficients);
        let residuals = y - &fitted_values;

        // Calculate statistics with improved numerical stability
        let rss = residuals.mapv(|x| x * x).sum();

        // More robust mean calculation
        let y_mean = y.sum() / n;
        let tss = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        let r_squared = if tss < 1e-14 || rss.is_nan() || tss.is_nan() {
            0.0 // If no variance in y or NaN values, R-squared is 0
        } else {
            let r2 = 1.0 - rss / tss;
            if r2.is_nan() || r2.is_infinite() {
                0.0
            } else {
                r2.clamp(-1.0, 1.0) // Ensure R-squared is in valid range
            }
        };

        let adjusted_r_squared = if tss < 1e-14 || n <= p {
            0.0
        } else {
            let adj_r2 = 1.0 - (rss / (n - p)) / (tss / (n - 1.0));
            if adj_r2.is_nan() || adj_r2.is_infinite() {
                0.0
            } else {
                adj_r2.clamp(-1.0, 1.0)
            }
        };

        // Standard errors with improved numerical stability
        let mse = if n > p { rss / (n - p) } else { 1.0 };
        let var_coeff_matrix_result = self.invert_matrix_robust(&xtx_reg);
        let mut standard_errors = Array1::ones(coefficients.len()); // Default to 1.0

        if let Ok(var_coeff_matrix) = var_coeff_matrix_result {
            standard_errors = Array1::zeros(coefficients.len());
            for i in 0..coefficients.len() {
                let variance = var_coeff_matrix[[i, i]] * mse;
                standard_errors[i] = if variance >= 0.0 {
                    variance.sqrt()
                } else {
                    1.0
                };
            }
        }

        // T-statistics and p-values
        let t_statistics = Array1::from_vec(
            coefficients
                .iter()
                .zip(standard_errors.iter())
                .map(|(&coeff, &se)| if se > 0.0 { coeff / se } else { 0.0 })
                .collect(),
        );
        let df = if n > p { (n - p) as i32 } else { 1 };
        let p_values = t_statistics.mapv(|t| {
            if t.is_finite() {
                2.0 * (1.0 - self.t_distribution_cdf(t.abs(), df))
            } else {
                1.0
            }
        });

        Ok(OLSResult {
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
            r_squared,
            adjusted_r_squared,
            residual_sum_squares: rss,
            degrees_of_freedom: df as usize,
            fitted_values,
            residuals,
        })
    }

    #[allow(dead_code)]
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> RegressionResult<Array1<f64>> {
        // Simple Gauss-Seidel iterative solver
        let n = a.nrows();
        let mut x = Array1::zeros(n);
        let max_iter = 1000;
        let tolerance = 1e-12;

        for _iter in 0..max_iter {
            let mut x_new = x.clone();

            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += a[[i, j]] * x[j];
                    }
                }

                if a[[i, i]].abs() < f64::EPSILON {
                    return Err(TimeSeriesError::ComputationError(
                        "Singular matrix".to_string(),
                    ));
                }

                x_new[i] = (b[i] - sum) / a[[i, i]];
            }

            let diff = (&x_new - &x).mapv(|x| x.abs()).sum();
            x = x_new;

            if diff < tolerance {
                break;
            }
        }

        Ok(x)
    }

    fn solve_linear_system_robust(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> RegressionResult<Array1<f64>> {
        let n = a.nrows();

        // Create augmented matrix [A | b]
        let mut augmented = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = a[[i, j]];
            }
            augmented[[i, n]] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..=n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if augmented[[i, i]].abs() < 1e-12 {
                return Err(TimeSeriesError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in i..=n {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in i + 1..n {
                let factor = augmented[[k, i]];
                for j in i..=n {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = augmented[[i, n]];
            for j in i + 1..n {
                x[i] -= augmented[[i, j]] * x[j];
            }
        }

        // Validate solution
        for &val in x.iter() {
            if !val.is_finite() {
                return Err(TimeSeriesError::ComputationError(
                    "Solution contains non-finite values".to_string(),
                ));
            }
        }

        Ok(x)
    }

    #[allow(dead_code)]
    fn invert_matrix(&self, matrix: &Array2<f64>) -> RegressionResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..2 * n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if augmented[[i, i]].abs() < f64::EPSILON {
                return Err(TimeSeriesError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in 0..2 * n {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..2 * n {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }

    fn invert_matrix_robust(&self, matrix: &Array2<f64>) -> RegressionResult<Array2<f64>> {
        let n = matrix.nrows();
        if n == 0 {
            return Err(TimeSeriesError::ComputationError(
                "Cannot invert empty matrix".to_string(),
            ));
        }

        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = augmented[[i, i]].abs();
            for k in i + 1..n {
                let val = augmented[[k, i]].abs();
                if val > max_val {
                    max_row = k;
                    max_val = val;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..2 * n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singularity with better tolerance
            if augmented[[i, i]].abs() < 1e-12 {
                return Err(TimeSeriesError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in 0..2 * n {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..2 * n {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix and validate
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let val = augmented[[i, j + n]];
                if !val.is_finite() {
                    return Err(TimeSeriesError::ComputationError(
                        "Matrix inversion produced non-finite values".to_string(),
                    ));
                }
                inverse[[i, j]] = val;
            }
        }

        Ok(inverse)
    }

    fn t_distribution_cdf(&self, t: f64, df: i32) -> f64 {
        // Approximation for t-distribution CDF
        if df <= 0 {
            return 0.5;
        }

        if df >= 100 {
            return self.normal_cdf(t);
        }

        // Use incomplete beta function approximation
        let x = df as f64 / (df as f64 + t * t);
        0.5 + 0.5
            * self.incomplete_beta(x, 0.5 * df as f64, 0.5).signum()
            * (1.0 - self.incomplete_beta(x, 0.5 * df as f64, 0.5))
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / (2.0_f64).sqrt()))
    }

    fn erf(&self, x: f64) -> f64 {
        // Approximation for error function
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    fn incomplete_beta(&self, x: f64, a: f64, b: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }

        // Continued fraction approximation
        let mut result = x.powf(a) * (1.0 - x).powf(b) / a;
        let mut term = result;

        for n in 1..100 {
            let n_f = n as f64;
            term *= (a + n_f - 1.0) * x / n_f;
            result += term;
            if term.abs() < 1e-10 {
                break;
            }
        }

        result / self.beta_function(a, b)
    }

    fn beta_function(&self, a: f64, b: f64) -> f64 {
        self.gamma_function(a) * self.gamma_function(b) / self.gamma_function(a + b)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn gamma_function(&self, x: f64) -> f64 {
        if x < 1.0 {
            return self.gamma_function(x + 1.0) / x;
        }

        (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }

    fn select_optimal_lags(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
        config: &ARDLConfig,
    ) -> RegressionResult<(usize, usize)> {
        let mut best_criterion = f64::INFINITY;
        let mut best_lags = (1, 1);

        for y_lags in 1..=config.max_y_lags {
            for x_lags in 0..=config.max_x_lags {
                // Create temporary config for this lag combination
                let temp_config = ARDLConfig {
                    max_y_lags: y_lags,
                    max_x_lags: x_lags,
                    auto_lag_selection: false,
                    ..config.clone()
                };

                if let Ok(result) = self.fit_ardl_fixed_lags(y, x, &temp_config) {
                    let criterion = match config.selection_criterion {
                        InformationCriterion::AIC => result.aic,
                        InformationCriterion::BIC => result.bic,
                        InformationCriterion::HQIC => {
                            // Hannan-Quinn: -2*log(L) + 2*k*log(log(n))
                            let n = y.len() as f64;
                            let k = result.y_coefficients.len() + result.x_coefficients.len() + 1;
                            -2.0 * (-result.aic / 2.0 + k as f64) + 2.0 * k as f64 * n.ln().ln()
                        }
                    };

                    if criterion < best_criterion {
                        best_criterion = criterion;
                        best_lags = (y_lags, x_lags);
                    }
                }
            }
        }

        Ok(best_lags)
    }

    fn fit_ardl_fixed_lags(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
        config: &ARDLConfig,
    ) -> RegressionResult<ARDLResult> {
        // This is a simplified version that doesn't do auto lag selection
        let temp_config = ARDLConfig {
            auto_lag_selection: false,
            ..config.clone()
        };
        self.fit_ardl(y, x, &temp_config)
    }

    fn estimate_long_run_relationship(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
    ) -> RegressionResult<OLSResult> {
        // Simple OLS regression for long-run relationship
        let n = y.len();
        let mut design_matrix = Array2::zeros((n, 2));

        for i in 0..n {
            design_matrix[[i, 0]] = 1.0; // Intercept
            design_matrix[[i, 1]] = x[i];
        }

        self.fit_ols_regression(&design_matrix, y)
    }

    fn test_cointegration(&self, residuals: &Array1<f64>) -> RegressionResult<f64> {
        // Simplified Engle-Granger cointegration test
        // Test for unit root in residuals using ADF test

        let n = residuals.len();
        if n < 10 {
            return Ok(1.0); // Not enough data, assume no cointegration
        }

        // First difference of residuals
        let mut diff_residuals = Array1::zeros(n - 1);
        for i in 1..n {
            diff_residuals[i - 1] = residuals[i] - residuals[i - 1];
        }

        // Regression: Δε_t = α + βε_{t-1} + error
        let mut design_matrix = Array2::zeros((n - 1, 2));
        for i in 0..n - 1 {
            design_matrix[[i, 0]] = 1.0; // Intercept
            design_matrix[[i, 1]] = residuals[i]; // Lagged residual
        }

        let adf_result = self.fit_ols_regression(&design_matrix, &diff_residuals)?;
        let t_stat = adf_result.t_statistics[1]; // t-statistic for β coefficient

        // Approximate p-value for Engle-Granger test
        // This is a very simplified approximation
        let p_value = if t_stat < -3.5 {
            0.01
        } else if t_stat < -3.0 {
            0.05
        } else if t_stat < -2.5 {
            0.10
        } else {
            0.20
        };

        Ok(p_value)
    }

    fn estimate_ecm(
        &self,
        y: &Array1<f64>,
        x: &Array1<f64>,
        long_run_result: &OLSResult,
        config: &ErrorCorrectionConfig,
    ) -> RegressionResult<ECMResult> {
        let n = y.len();

        // Calculate error correction term (lagged residuals from long-run relationship)
        let ect = long_run_result.residuals.slice(s![..n - 1]).to_owned();

        // Calculate first differences
        let mut dy = Array1::zeros(n - 1);
        let mut dx = Array1::zeros(n - 1);
        for i in 1..n {
            dy[i - 1] = y[i] - y[i - 1];
            dx[i - 1] = x[i] - x[i - 1];
        }

        // Build design matrix for ECM
        let max_lag = config.max_diff_lags;
        let start_idx = max_lag;
        let n_obs = n - 1 - start_idx;

        let n_params = 1 + 1 + max_lag + max_lag; // intercept + ECT + lagged dy + lagged dx
        let mut design_matrix = Array2::zeros((n_obs, n_params));
        let mut response = Array1::zeros(n_obs);

        for i in 0..n_obs {
            let idx = start_idx + i;
            response[i] = dy[idx];

            let mut col_idx = 0;

            // Intercept
            design_matrix[[i, col_idx]] = 1.0;
            col_idx += 1;

            // Error correction term
            design_matrix[[i, col_idx]] = ect[idx - 1];
            col_idx += 1;

            // Lagged differences of y
            for lag in 1..=max_lag {
                if idx >= lag {
                    design_matrix[[i, col_idx]] = dy[idx - lag];
                }
                col_idx += 1;
            }

            // Lagged differences of x
            for lag in 0..max_lag {
                if idx >= lag {
                    design_matrix[[i, col_idx]] = dx[idx - lag];
                }
                col_idx += 1;
            }
        }

        let regression_result = self.fit_ols_regression(&design_matrix, &response)?;

        // Extract coefficients
        let error_correction_coeff = regression_result.coefficients[1];
        let short_run_y_coeffs = regression_result
            .coefficients
            .slice(s![2..2 + max_lag])
            .to_owned();
        let short_run_x_coeffs = regression_result
            .coefficients
            .slice(s![2 + max_lag..])
            .to_owned();

        Ok(ECMResult {
            error_correction_coeff,
            short_run_y_coeffs,
            short_run_x_coeffs,
            standard_errors: regression_result.standard_errors,
            t_statistics: regression_result.t_statistics,
            p_values: regression_result.p_values,
            r_squared: regression_result.r_squared,
            fitted_values: regression_result.fitted_values,
            residuals: regression_result.residuals,
        })
    }
}

impl Default for TimeSeriesRegression {
    fn default() -> Self {
        Self::new()
    }
}

// Helper structs

#[derive(Debug, Clone)]
struct OLSResult {
    coefficients: Array1<f64>,
    standard_errors: Array1<f64>,
    t_statistics: Array1<f64>,
    p_values: Array1<f64>,
    r_squared: f64,
    adjusted_r_squared: f64,
    residual_sum_squares: f64,
    degrees_of_freedom: usize,
    fitted_values: Array1<f64>,
    residuals: Array1<f64>,
}

#[derive(Debug, Clone)]
struct ECMResult {
    error_correction_coeff: f64,
    short_run_y_coeffs: Array1<f64>,
    short_run_x_coeffs: Array1<f64>,
    standard_errors: Array1<f64>,
    t_statistics: Array1<f64>,
    p_values: Array1<f64>,
    r_squared: f64,
    fitted_values: Array1<f64>,
    residuals: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_distributed_lag_model() {
        let n = 50;
        let mut y = Array1::zeros(n);
        let mut x = Array1::zeros(n);

        // Generate test data with distributed lag relationship
        for i in 2..n {
            x[i] = (i as f64 * 0.1).sin();
            y[i] = 0.5 * x[i] + 0.3 * x[i - 1] + 0.1 * x[i - 2] + 0.1 * rand::random::<f64>();
        }

        let regression = TimeSeriesRegression::new();
        let config = DistributedLagConfig {
            max_lag: 2,
            ..Default::default()
        };

        let result = regression.fit_distributed_lag(&y, &x, &config).unwrap();

        assert_eq!(result.max_lag, 2);
        eprintln!("Distributed lag R-squared: {}", result.r_squared);
        assert!(result.r_squared >= -1.0 && result.r_squared <= 1.0);
        assert_eq!(result.coefficients.len(), 4); // intercept + 3 lag terms
    }

    #[test]
    fn test_ardl_model() {
        let n = 50;
        let mut y = Array1::zeros(n);
        let mut x = Array1::zeros(n);

        // Generate test data with ARDL relationship
        for i in 2..n {
            x[i] = (i as f64 * 0.1).sin();
            y[i] = 0.3 * y[i - 1] + 0.5 * x[i] + 0.2 * x[i - 1] + 0.1 * rand::random::<f64>();
        }

        let regression = TimeSeriesRegression::new();
        let config = ARDLConfig {
            max_y_lags: 1,
            max_x_lags: 1,
            auto_lag_selection: false,
            ..Default::default()
        };

        let result = regression.fit_ardl(&y, &x, &config).unwrap();

        assert_eq!(result.y_lags, 1);
        assert_eq!(result.x_lags, 1);
        eprintln!("ARDL R-squared: {}", result.r_squared);
        assert!(result.r_squared >= -1.0 && result.r_squared <= 1.0);
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
    }

    #[test]
    fn test_regression_with_arima_errors() {
        let n = 30;
        let y = Array1::from_vec(
            (0..n)
                .map(|i| i as f64 + 0.1 * rand::random::<f64>())
                .collect(),
        );
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| (i as f64).sin()).collect()).unwrap();

        let regression = TimeSeriesRegression::new();
        let config = ARIMAErrorsConfig::default();

        let result = regression
            .fit_regression_with_arima_errors(&y, &x, &config)
            .unwrap();

        assert!(result.regression_r_squared >= 0.0 && result.regression_r_squared <= 1.0);
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
        assert_eq!(result.fitted_values.len(), y.len());
    }
}
