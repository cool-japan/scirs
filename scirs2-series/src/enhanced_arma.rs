//! Enhanced autoregressive and moving average models
//!
//! This module provides advanced implementations of AR, MA, and ARMA models
//! with robust estimation, enhanced diagnostics, and improved forecasting capabilities.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};
use crate::optimization::{LBFGSOptimizer, OptimizationOptions};
use crate::utils::autocorrelation;

/// Enhanced AR model with robust estimation capabilities
#[derive(Debug, Clone)]
pub struct EnhancedARModel<F> {
    /// AR order
    pub p: usize,
    /// AR coefficients
    pub ar_coeffs: Array1<F>,
    /// Model intercept
    pub intercept: F,
    /// Residual variance
    pub sigma2: F,
    /// Standard errors of coefficients
    pub coefficient_se: Array1<F>,
    /// Model fit statistics
    pub fit_stats: ModelFitStatistics<F>,
    /// Confidence intervals for coefficients
    pub confidence_intervals: Array2<F>,
    /// Whether the model is fitted
    pub is_fitted: bool,
    /// Estimation method used
    pub estimation_method: EstimationMethod,
}

/// Enhanced MA model with improved estimation
#[derive(Debug, Clone)]
pub struct EnhancedMAModel<F> {
    /// MA order
    pub q: usize,
    /// MA coefficients
    pub ma_coeffs: Array1<F>,
    /// Model intercept
    pub intercept: F,
    /// Residual variance
    pub sigma2: F,
    /// Standard errors of coefficients
    pub coefficient_se: Array1<F>,
    /// Model fit statistics
    pub fit_stats: ModelFitStatistics<F>,
    /// Confidence intervals for coefficients
    pub confidence_intervals: Array2<F>,
    /// Whether the model is fitted
    pub is_fitted: bool,
    /// Estimation method used
    pub estimation_method: EstimationMethod,
}

/// Enhanced ARMA model combining AR and MA components
#[derive(Debug, Clone)]
pub struct EnhancedARMAModel<F> {
    /// AR order
    pub p: usize,
    /// MA order
    pub q: usize,
    /// AR coefficients
    pub ar_coeffs: Array1<F>,
    /// MA coefficients
    pub ma_coeffs: Array1<F>,
    /// Model intercept
    pub intercept: F,
    /// Residual variance
    pub sigma2: F,
    /// Standard errors of coefficients
    pub coefficient_se: Array1<F>,
    /// Model fit statistics
    pub fit_stats: ModelFitStatistics<F>,
    /// Confidence intervals for coefficients
    pub confidence_intervals: Array2<F>,
    /// Whether the model is fitted
    pub is_fitted: bool,
    /// Estimation method used
    pub estimation_method: EstimationMethod,
}

/// Model fit statistics
#[derive(Debug, Clone)]
pub struct ModelFitStatistics<F> {
    /// Log-likelihood
    pub log_likelihood: F,
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Hannan-Quinn Information Criterion
    pub hqc: F,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// R-squared (goodness of fit)
    pub r_squared: F,
    /// Adjusted R-squared
    pub adj_r_squared: F,
    /// Durbin-Watson statistic
    pub durbin_watson: F,
    /// Ljung-Box test statistic
    pub ljung_box: F,
    /// Ljung-Box p-value
    pub ljung_box_pvalue: F,
}

/// Estimation methods for ARMA models
#[derive(Debug, Clone, Copy)]
pub enum EstimationMethod {
    /// Maximum Likelihood Estimation
    MLE,
    /// Conditional Sum of Squares
    CSS,
    /// Yule-Walker equations (AR only)
    YuleWalker,
    /// Burg method (AR only)
    Burg,
    /// Robust M-estimation
    RobustM,
    /// Generalized Method of Moments
    GMM,
}

/// Forecasting options
#[derive(Debug, Clone)]
pub struct ForecastOptions {
    /// Confidence level for prediction intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Whether to include prediction intervals
    pub include_prediction_intervals: bool,
    /// Method for calculating prediction intervals
    pub interval_method: IntervalMethod,
    /// Number of bootstrap samples for bootstrap intervals
    pub bootstrap_samples: usize,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            include_prediction_intervals: true,
            interval_method: IntervalMethod::Analytical,
            bootstrap_samples: 1000,
        }
    }
}

/// Methods for calculating prediction intervals
#[derive(Debug, Clone, Copy)]
pub enum IntervalMethod {
    /// Analytical formula based on model variance
    Analytical,
    /// Bootstrap resampling
    Bootstrap,
    /// Monte Carlo simulation
    MonteCarlo,
}

/// Forecast result with prediction intervals
#[derive(Debug, Clone)]
pub struct ForecastResult<F> {
    /// Point forecasts
    pub forecasts: Array1<F>,
    /// Lower bounds of prediction intervals
    pub lower_bounds: Option<Array1<F>>,
    /// Upper bounds of prediction intervals
    pub upper_bounds: Option<Array1<F>>,
    /// Forecast standard errors
    pub forecast_se: Array1<F>,
    /// Confidence level used
    pub confidence_level: f64,
}

impl<F> EnhancedARModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new enhanced AR model
    pub fn new(p: usize) -> Result<Self> {
        if p == 0 || p > 20 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "p".to_string(),
                message: "AR order must be between 1 and 20".to_string(),
            });
        }

        Ok(Self {
            p,
            ar_coeffs: Array1::zeros(p),
            intercept: F::zero(),
            sigma2: F::one(),
            coefficient_se: Array1::zeros(p + 1), // +1 for intercept
            fit_stats: ModelFitStatistics {
                log_likelihood: F::neg_infinity(),
                aic: F::infinity(),
                bic: F::infinity(),
                hqc: F::infinity(),
                n_obs: 0,
                n_params: p + 1,
                r_squared: F::zero(),
                adj_r_squared: F::zero(),
                durbin_watson: F::zero(),
                ljung_box: F::zero(),
                ljung_box_pvalue: F::zero(),
            },
            confidence_intervals: Array2::zeros((p + 1, 2)),
            is_fitted: false,
            estimation_method: EstimationMethod::YuleWalker,
        })
    }

    /// Fit the AR model using specified estimation method
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix1>, method: EstimationMethod) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        let min_required = self.p * 3 + 10; // Need more data for robust estimation
        if data.len() < min_required {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Need at least {} observations for AR({}) model, got {}",
                min_required,
                self.p,
                data.len()
            )));
        }

        self.estimation_method = method;

        match method {
            EstimationMethod::YuleWalker => self.fit_yule_walker(data)?,
            EstimationMethod::Burg => self.fit_burg(data)?,
            EstimationMethod::MLE => self.fit_mle(data)?,
            EstimationMethod::CSS => self.fit_css(data)?,
            EstimationMethod::RobustM => self.fit_robust_m(data)?,
            EstimationMethod::GMM => self.fit_gmm(data)?,
        }

        // Calculate fit statistics
        self.calculate_fit_statistics(data)?;

        // Calculate confidence intervals
        self.calculate_confidence_intervals(data)?;

        self.is_fitted = true;
        Ok(())
    }

    /// Fit using Yule-Walker equations (improved implementation)
    fn fit_yule_walker<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        self.fit_stats.n_obs = n;

        // Center the data
        let mean = data.mean().unwrap_or(F::zero());
        let centered: Array1<F> = data.mapv(|x| x - mean);

        // Calculate sample autocorrelations
        let acf = autocorrelation(&centered, Some(self.p))?;

        // Set up Yule-Walker equations
        let mut r_matrix = Array2::zeros((self.p, self.p));
        let mut r_vector = Array1::zeros(self.p);

        for i in 0..self.p {
            r_vector[i] = acf[i + 1];
            for j in 0..self.p {
                r_matrix[[i, j]] = acf[(i as i32 - j as i32).unsigned_abs() as usize];
            }
        }

        // Add regularization to handle numerical instability
        let reg_param =
            F::from(1e-10).unwrap_or(F::epsilon() * F::from(1000.0).unwrap_or(F::one()));
        for i in 0..self.p {
            r_matrix[[i, i]] = r_matrix[[i, i]] + reg_param;
        }

        // Solve Yule-Walker equations: R * phi = r
        let r_inv = Self::matrix_inverse(&r_matrix)?;
        self.ar_coeffs = r_inv.dot(&r_vector);

        // Estimate intercept
        self.intercept = mean * (F::one() - self.ar_coeffs.sum());

        // Estimate noise variance
        let mut variance_multiplier = F::one();
        for k in 0..self.p {
            variance_multiplier = variance_multiplier - self.ar_coeffs[k] * acf[k + 1];
        }

        let sample_variance = centered.mapv(|x| x * x).mean().unwrap_or(F::one());
        self.sigma2 = sample_variance * variance_multiplier;

        Ok(())
    }

    /// Fit using Burg method (better for short time series)
    fn fit_burg<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        self.fit_stats.n_obs = n;

        // Center the data
        let mean = data.mean().unwrap_or(F::zero());
        let x: Array1<F> = data.mapv(|x| x - mean);

        // Initialize
        let mut ar_coeffs = Array1::zeros(self.p);
        let mut forward_error = x.clone();
        let mut backward_error = x.clone();

        for order in 1..=self.p {
            // Calculate reflection coefficient
            let mut numerator = F::zero();
            let mut denominator = F::zero();

            for i in order..n {
                numerator = numerator + forward_error[i] * backward_error[i - 1];
                denominator = denominator
                    + (forward_error[i] * forward_error[i]
                        + backward_error[i - 1] * backward_error[i - 1]);
            }

            let eps = F::from(1e-12).unwrap_or(F::epsilon());
            if denominator.abs() < eps {
                return Err(TimeSeriesError::ComputationError(
                    "Burg algorithm failed: division by zero or near-zero".to_string(),
                ));
            }

            let reflection_coeff = F::from(2.0).unwrap() * numerator / denominator;
            ar_coeffs[order - 1] = reflection_coeff;

            // Update AR coefficients using Levinson-Durbin recursion
            for i in 0..(order - 1) {
                ar_coeffs[i] = ar_coeffs[i] - reflection_coeff * ar_coeffs[order - 2 - i];
            }

            // Update prediction errors
            for i in order..n {
                let temp_forward = forward_error[i];
                forward_error[i] = forward_error[i] - reflection_coeff * backward_error[i - 1];
                backward_error[i - 1] = backward_error[i - 1] - reflection_coeff * temp_forward;
            }
        }

        self.ar_coeffs = ar_coeffs;
        self.intercept = mean * (F::one() - self.ar_coeffs.sum());

        // Calculate residual variance
        let residuals = self.calculate_residuals(&x)?;
        self.sigma2 = residuals.mapv(|x| x * x).mean().unwrap_or(F::one());

        Ok(())
    }

    /// Fit using Maximum Likelihood Estimation
    fn fit_mle<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        self.fit_stats.n_obs = n;

        // Initialize with Yule-Walker estimates
        self.fit_yule_walker(data)?;

        // Optimize using MLE
        let n_params = self.p + 1; // AR coeffs + intercept
        let mut params = Array1::zeros(n_params);

        // Pack parameters
        for i in 0..self.p {
            params[i] = self.ar_coeffs[i];
        }
        params[self.p] = self.intercept;

        let mut optimizer = LBFGSOptimizer::new(OptimizationOptions::default());
        let data_clone = data.to_owned();
        let p = self.p;

        // Objective function (negative log-likelihood)
        let objective = |params: &Array1<F>| -> F {
            let mut ar_coeffs = Array1::zeros(p);
            for i in 0..p {
                ar_coeffs[i] = params[i];
            }
            let intercept = params[p];

            // Calculate residuals
            let mut residuals = Array1::zeros(n);
            for t in p..n {
                let mut pred = intercept;
                for i in 0..p {
                    pred = pred + ar_coeffs[i] * data_clone[t - i - 1];
                }
                residuals[t] = data_clone[t] - pred;
            }

            // Calculate likelihood
            let sigma2 = residuals
                .slice(ndarray::s![p..])
                .mapv(|x| x * x)
                .mean()
                .unwrap_or(F::one());
            let n_eff = F::from(n - p).unwrap();

            n_eff / F::from(2.0).unwrap() * sigma2.ln()
                + residuals.slice(ndarray::s![p..]).mapv(|x| x * x).sum()
                    / (F::from(2.0).unwrap() * sigma2)
        };

        // Gradient function (numerical approximation)
        let gradient = |params: &Array1<F>| -> Array1<F> {
            let mut grad = Array1::zeros(n_params);
            let eps = F::from(1e-8).unwrap();

            for i in 0..n_params {
                let mut params_plus = params.clone();
                let mut params_minus = params.clone();
                params_plus[i] = params_plus[i] + eps;
                params_minus[i] = params_minus[i] - eps;

                grad[i] = (objective(&params_plus) - objective(&params_minus))
                    / (F::from(2.0).unwrap() * eps);
            }

            grad
        };

        let result = optimizer.optimize(objective, gradient, &params)?;

        // Update parameters
        for i in 0..self.p {
            self.ar_coeffs[i] = result.x[i];
        }
        self.intercept = result.x[self.p];

        // Update variance
        let residuals = self.calculate_residuals(data)?;
        self.sigma2 = residuals
            .slice(ndarray::s![self.p..])
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(F::one());

        Ok(())
    }

    /// Fit using Conditional Sum of Squares
    fn fit_css<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        self.fit_stats.n_obs = n;

        // Initialize with Yule-Walker estimates
        self.fit_yule_walker(data)?;

        // Set up least squares problem for conditional estimation
        let n_eff = n - self.p;
        let mut y = Array1::zeros(n_eff);
        let mut x = Array2::zeros((n_eff, self.p + 1));

        for t in self.p..n {
            y[t - self.p] = data[t];
            x[[t - self.p, 0]] = F::one(); // Intercept column
            for i in 0..self.p {
                x[[t - self.p, i + 1]] = data[t - i - 1];
            }
        }

        // Solve least squares: (X'X)^(-1) X'y
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let xtx_inv = Self::matrix_inverse(&xtx)?;
        let coeffs = xtx_inv.dot(&xty);

        self.intercept = coeffs[0];
        for i in 0..self.p {
            self.ar_coeffs[i] = coeffs[i + 1];
        }

        // Calculate residual variance
        let residuals = self.calculate_residuals(data)?;
        self.sigma2 =
            residuals.slice(ndarray::s![self.p..]).mapv(|x| x * x).sum() / F::from(n_eff).unwrap();

        Ok(())
    }

    /// Fit using robust M-estimation
    fn fit_robust_m<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        self.fit_stats.n_obs = n;

        // Initialize with Yule-Walker estimates
        self.fit_yule_walker(data)?;

        let max_iterations = 50;
        let tolerance = F::from(1e-6).unwrap();

        for _iter in 0..max_iterations {
            let old_coeffs = self.ar_coeffs.clone();
            let old_intercept = self.intercept;

            // Calculate current residuals
            let residuals = self.calculate_residuals(data)?;

            // Calculate robust weights using Huber function
            let weights = self.calculate_robust_weights(&residuals)?;

            // Weighted least squares
            let n_eff = n - self.p;
            let mut wy = Array1::zeros(n_eff);
            let mut wx = Array2::zeros((n_eff, self.p + 1));

            for t in self.p..n {
                let weight_sqrt = weights[t].sqrt();
                wy[t - self.p] = weight_sqrt * data[t];
                wx[[t - self.p, 0]] = weight_sqrt; // Intercept
                for i in 0..self.p {
                    wx[[t - self.p, i + 1]] = weight_sqrt * data[t - i - 1];
                }
            }

            // Solve weighted least squares
            let wxtx = wx.t().dot(&wx);
            let wxty = wx.t().dot(&wy);
            let wxtx_inv = Self::matrix_inverse(&wxtx)?;
            let coeffs = wxtx_inv.dot(&wxty);

            self.intercept = coeffs[0];
            for i in 0..self.p {
                self.ar_coeffs[i] = coeffs[i + 1];
            }

            // Check convergence
            let coeff_change = (&self.ar_coeffs - &old_coeffs).mapv(|x| x.abs()).sum();
            let intercept_change = (self.intercept - old_intercept).abs();

            if coeff_change + intercept_change < tolerance {
                break;
            }
        }

        // Update variance
        let residuals = self.calculate_residuals(data)?;
        self.sigma2 = residuals
            .slice(ndarray::s![self.p..])
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(F::one());

        Ok(())
    }

    /// Fit using Generalized Method of Moments
    fn fit_gmm<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        // For now, fall back to MLE (GMM implementation is more complex)
        self.fit_mle(data)
    }

    /// Calculate robust weights using Huber function
    fn calculate_robust_weights(&self, residuals: &Array1<F>) -> Result<Array1<F>> {
        let n = residuals.len();
        let mut weights = Array1::ones(n);

        // Calculate median absolute deviation (MAD)
        let residual_vec: Vec<F> = residuals.to_vec();
        let median_residual = self.median(&residual_vec);
        let abs_deviations: Vec<F> = residual_vec
            .iter()
            .map(|&r| (r - median_residual).abs())
            .collect();
        let mad = self.median(&abs_deviations) * F::from(1.4826).unwrap(); // Scale factor for normal distribution

        if mad == F::zero() {
            return Ok(weights);
        }

        // Huber weights
        let c = F::from(1.345).unwrap(); // Huber tuning constant
        for i in 0..n {
            let standardized = (residuals[i] - median_residual).abs() / mad;
            if standardized > c {
                weights[i] = c / standardized;
            }
        }

        Ok(weights)
    }

    /// Calculate median of a vector
    fn median(&self, values: &[F]) -> F {
        if values.is_empty() {
            return F::zero();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        if len % 2 == 0 {
            let mid1 = sorted[len / 2 - 1];
            let mid2 = sorted[len / 2];
            (mid1 + mid2) / F::from(2.0).unwrap()
        } else {
            sorted[len / 2]
        }
    }

    /// Calculate residuals
    fn calculate_residuals<S>(&self, data: &ArrayBase<S, Ix1>) -> Result<Array1<F>>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        let mut residuals = Array1::zeros(n);

        for t in self.p..n {
            let mut pred = self.intercept;
            for i in 0..self.p {
                pred = pred + self.ar_coeffs[i] * data[t - i - 1];
            }
            residuals[t] = data[t] - pred;
        }

        Ok(residuals)
    }

    /// Calculate fit statistics
    fn calculate_fit_statistics<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        let residuals = self.calculate_residuals(data)?;

        // Log-likelihood
        let n_eff = F::from(n - self.p).unwrap();
        self.fit_stats.log_likelihood = -n_eff / F::from(2.0).unwrap()
            * (F::from(2.0 * std::f64::consts::PI).unwrap() * self.sigma2).ln()
            - residuals.slice(ndarray::s![self.p..]).mapv(|x| x * x).sum()
                / (F::from(2.0).unwrap() * self.sigma2);

        // Information criteria
        let k = F::from(self.p + 1).unwrap(); // Number of parameters
        self.fit_stats.aic =
            F::from(2.0).unwrap() * k - F::from(2.0).unwrap() * self.fit_stats.log_likelihood;
        self.fit_stats.bic = k * n_eff.ln() - F::from(2.0).unwrap() * self.fit_stats.log_likelihood;
        self.fit_stats.hqc = F::from(2.0).unwrap() * k * n_eff.ln().ln()
            - F::from(2.0).unwrap() * self.fit_stats.log_likelihood;

        // R-squared
        let y_slice = data.slice(ndarray::s![self.p..]);
        let y_mean = y_slice.mean().unwrap_or(F::zero());
        let tss = y_slice.mapv(|x| (x - y_mean) * (x - y_mean)).sum();
        let rss = residuals.slice(ndarray::s![self.p..]).mapv(|x| x * x).sum();

        if tss != F::zero() {
            self.fit_stats.r_squared = F::one() - rss / tss;
            self.fit_stats.adj_r_squared =
                F::one() - (rss / (n_eff - k)) / (tss / (n_eff - F::one()));
        }

        // Durbin-Watson statistic
        let mut dw_numerator = F::zero();
        let mut dw_denominator = F::zero();
        for t in (self.p + 1)..n {
            dw_numerator = dw_numerator
                + (residuals[t] - residuals[t - 1]) * (residuals[t] - residuals[t - 1]);
        }
        for t in self.p..n {
            dw_denominator = dw_denominator + residuals[t] * residuals[t];
        }
        if dw_denominator != F::zero() {
            self.fit_stats.durbin_watson = dw_numerator / dw_denominator;
        }

        Ok(())
    }

    /// Calculate confidence intervals for coefficients
    fn calculate_confidence_intervals<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        let _n_eff = n - self.p;

        // Calculate Hessian matrix (Fisher Information Matrix)
        let hessian = self.calculate_hessian(data)?;
        let hessian_inv = Self::matrix_inverse(&hessian)?;

        // Standard errors are square roots of diagonal elements
        for i in 0..self.p {
            self.coefficient_se[i] = hessian_inv[[i, i]].sqrt();
        }
        self.coefficient_se[self.p] = hessian_inv[[self.p, self.p]].sqrt(); // Intercept SE

        // Calculate confidence intervals (95% by default)
        let t_value = F::from(1.96).unwrap(); // Approximate for large samples
        for i in 0..self.p {
            self.confidence_intervals[[i, 0]] =
                self.ar_coeffs[i] - t_value * self.coefficient_se[i];
            self.confidence_intervals[[i, 1]] =
                self.ar_coeffs[i] + t_value * self.coefficient_se[i];
        }
        // Intercept CI
        self.confidence_intervals[[self.p, 0]] =
            self.intercept - t_value * self.coefficient_se[self.p];
        self.confidence_intervals[[self.p, 1]] =
            self.intercept + t_value * self.coefficient_se[self.p];

        Ok(())
    }

    /// Calculate Hessian matrix (second derivatives of log-likelihood)
    fn calculate_hessian<S>(&self, data: &ArrayBase<S, Ix1>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        let n = data.len();
        let n_eff = n - self.p;
        let n_params = self.p + 1;
        let hessian;

        // For AR models, the Hessian can be approximated using the design matrix
        let mut x = Array2::zeros((n_eff, n_params));

        for t in self.p..n {
            x[[t - self.p, 0]] = F::one(); // Intercept
            for i in 0..self.p {
                x[[t - self.p, i + 1]] = data[t - i - 1];
            }
        }

        // Hessian = X'X / sigma^2
        hessian = x.t().dot(&x) / self.sigma2;

        Ok(hessian)
    }

    /// Simple matrix inverse for small matrices
    fn matrix_inverse(matrix: &Array2<F>) -> Result<Array2<F>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(TimeSeriesError::ComputationError(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // For small matrices, use simple methods
        if n == 1 {
            let eps = F::from(1e-12).unwrap_or(F::epsilon());
            if matrix[[0, 0]].abs() < eps {
                return Err(TimeSeriesError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }
            let mut inv = Array2::zeros((1, 1));
            inv[[0, 0]] = F::one() / matrix[[0, 0]];
            return Ok(inv);
        }

        if n == 2 {
            let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
            let eps = F::from(1e-12).unwrap_or(F::epsilon());
            if det.abs() < eps {
                return Err(TimeSeriesError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }
            let mut inv = Array2::zeros((2, 2));
            inv[[0, 0]] = matrix[[1, 1]] / det;
            inv[[0, 1]] = -matrix[[0, 1]] / det;
            inv[[1, 0]] = -matrix[[1, 0]] / det;
            inv[[1, 1]] = matrix[[0, 0]] / det;
            return Ok(inv);
        }

        // For larger matrices, use Gaussian elimination with partial pivoting
        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { F::one() } else { F::zero() };
            }
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix with better tolerance
            let eps = F::from(1e-12).unwrap_or(F::epsilon());
            if augmented[[i, i]].abs() < eps {
                return Err(TimeSeriesError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Make diagonal element 1
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] = augmented[[i, j]] / pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] = augmented[[k, j]] - factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse from right half of augmented matrix
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inv)
    }

    /// Enhanced forecast with prediction intervals
    pub fn forecast_with_intervals<S>(
        &self,
        data: &ArrayBase<S, Ix1>,
        steps: usize,
        options: &ForecastOptions,
    ) -> Result<ForecastResult<F>>
    where
        S: Data<Elem = F>,
    {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidModel(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        // Point forecasts
        let mut forecasts = Array1::zeros(steps);
        let mut extended_data = data.to_vec();

        for h in 0..steps {
            let mut pred = self.intercept;
            for i in 0..self.p {
                let idx = extended_data.len() - i - 1;
                if idx < extended_data.len() {
                    pred = pred + self.ar_coeffs[i] * extended_data[idx];
                }
            }
            forecasts[h] = pred;
            extended_data.push(pred);
        }

        // Calculate forecast standard errors
        let mut forecast_se = Array1::zeros(steps);
        let mut mse_h = self.sigma2;

        for h in 0..steps {
            forecast_se[h] = mse_h.sqrt();

            // Update MSE for next horizon (simplified version)
            if h < steps - 1 {
                let mut phi_sum_sq = F::zero();
                for j in 0..=h.min(self.p - 1) {
                    if j < self.ar_coeffs.len() {
                        phi_sum_sq = phi_sum_sq + self.ar_coeffs[j] * self.ar_coeffs[j];
                    }
                }
                mse_h = self.sigma2 * (F::one() + phi_sum_sq);
            }
        }

        // Calculate prediction intervals if requested
        let (lower_bounds, upper_bounds) = if options.include_prediction_intervals {
            let z_value = F::from(Self::normal_quantile(
                1.0 - (1.0 - options.confidence_level) / 2.0,
            ))
            .unwrap();

            let lower = forecast_se.mapv(|se| se * (-z_value));
            let upper = forecast_se.mapv(|se| se * z_value);

            let lower_bounds = &forecasts + &lower;
            let upper_bounds = &forecasts + &upper;

            (Some(lower_bounds), Some(upper_bounds))
        } else {
            (None, None)
        };

        Ok(ForecastResult {
            forecasts,
            lower_bounds,
            upper_bounds,
            forecast_se,
            confidence_level: options.confidence_level,
        })
    }

    /// Normal distribution quantile (approximation)
    fn normal_quantile(p: f64) -> f64 {
        // Beasley-Springer-Moro algorithm approximation
        if p <= 0.0 || p >= 1.0 {
            return if p <= 0.0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        if p == 0.5 {
            return 0.0;
        }

        let a0 = 2.515517;
        let a1 = 0.802853;
        let a2 = 0.010328;
        let b0 = 1.0;
        let b1 = 1.432788;
        let b2 = 0.189269;
        let b3 = 0.001308;

        let r = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * r.ln()).sqrt();

        let numerator = a0 + a1 * t + a2 * t * t;
        let denominator = b0 + b1 * t + b2 * t * t + b3 * t * t * t;

        let result = t - numerator / denominator;

        if p > 0.5 {
            result
        } else {
            -result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_enhanced_ar_creation() {
        let model = EnhancedARModel::<f64>::new(2).unwrap();
        assert_eq!(model.p, 2);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_enhanced_ar_yule_walker_fit() {
        // Generate more realistic AR(2) data with noise
        let data = array![
            1.5, 2.1, 3.4, 2.8, 4.2, 3.9, 5.1, 4.7, 6.3, 5.8, 7.2, 6.9, 8.1, 7.5, 9.3, 8.8, 10.2,
            9.7, 11.1, 10.5, 12.3, 11.8, 13.2, 12.7, 14.1, 13.6, 15.4, 14.9, 16.2, 15.8
        ];
        let mut model = EnhancedARModel::new(2).unwrap();

        let result = model.fit(&data, EstimationMethod::YuleWalker);
        assert!(result.is_ok());
        assert!(model.is_fitted);
    }

    #[test]
    fn test_enhanced_ar_burg_fit() {
        // Generate more realistic AR(2) data with noise
        let data = array![
            1.2, 2.3, 2.9, 3.1, 4.5, 3.8, 5.2, 4.9, 6.1, 5.7, 7.3, 6.8, 8.0, 7.6, 9.2, 8.9, 10.1,
            9.8, 11.0, 10.6, 12.2, 11.9, 13.1, 12.8, 14.0, 13.7, 15.3, 14.8, 16.1, 15.9
        ];
        let mut model = EnhancedARModel::new(2).unwrap();

        let result = model.fit(&data, EstimationMethod::Burg);
        assert!(result.is_ok());
        assert!(model.is_fitted);
    }

    #[test]
    fn test_enhanced_ar_forecast_with_intervals() {
        // Generate more realistic AR(2) data with noise
        let data = array![
            1.3, 2.2, 3.1, 2.9, 4.4, 3.7, 5.3, 4.8, 6.2, 5.9, 7.1, 6.7, 8.2, 7.4, 9.1, 8.8, 10.3,
            9.6, 11.2, 10.7, 12.1, 11.8, 13.3, 12.6, 14.2, 13.9, 15.1, 14.7, 16.3, 15.6
        ];
        let mut model = EnhancedARModel::new(2).unwrap();
        model.fit(&data, EstimationMethod::YuleWalker).unwrap();

        let options = ForecastOptions::default();
        let result = model.forecast_with_intervals(&data, 5, &options).unwrap();

        assert_eq!(result.forecasts.len(), 5);
        assert!(result.lower_bounds.is_some());
        assert!(result.upper_bounds.is_some());
        assert_eq!(result.forecast_se.len(), 5);
    }

    #[test]
    fn test_matrix_inverse_2x2() {
        let matrix = array![[4.0, 2.0], [3.0, 1.0]];
        let inv = EnhancedARModel::<f64>::matrix_inverse(&matrix).unwrap();

        // Check that matrix * inv ≈ identity
        let product = matrix.dot(&inv);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(product[[0, 1]].abs() < 1e-10);
        assert!(product[[1, 0]].abs() < 1e-10);
    }
}
