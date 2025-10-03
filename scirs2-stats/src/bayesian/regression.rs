//! Bayesian linear regression models
//!
//! This module implements Bayesian approaches to linear regression, providing
//! posterior distributions over model parameters and predictions.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::validation::*;
use scirs2_linalg;
use statrs::statistics::Statistics;

/// Bayesian linear regression with normal-inverse-gamma prior
///
/// This implements Bayesian linear regression where:
/// - Coefficients have a normal prior
/// - Noise variance has an inverse-gamma prior
/// - Posterior is analytically tractable
#[derive(Debug, Clone)]
pub struct BayesianLinearRegression {
    /// Prior mean for coefficients
    pub prior_mean: Array1<f64>,
    /// Prior precision matrix for coefficients
    pub prior_precision: Array2<f64>,
    /// Prior shape parameter for noise variance
    pub prior_alpha: f64,
    /// Prior scale parameter for noise variance
    pub prior_beta: f64,
    /// Whether to include intercept
    pub fit_intercept: bool,
}

/// Result of Bayesian linear regression fit
#[derive(Debug, Clone)]
pub struct BayesianRegressionResult {
    /// Posterior mean of coefficients
    pub posterior_mean: Array1<f64>,
    /// Posterior covariance of coefficients
    pub posterior_covariance: Array2<f64>,
    /// Posterior shape parameter
    pub posterior_alpha: f64,
    /// Posterior scale parameter
    pub posterior_beta: f64,
    /// Number of training samples
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
    /// Training data mean (for centering)
    pub x_mean: Option<Array1<f64>>,
    /// Training target mean (for centering)
    pub y_mean: Option<f64>,
    /// Log marginal likelihood
    pub log_marginal_likelihood: f64,
}

impl BayesianLinearRegression {
    /// Create a new Bayesian linear regression model
    pub fn new(n_features: usize, fit_intercept: bool) -> StatsResult<Self> {
        check_positive(n_features, "n_features")?;

        // Default to weakly informative priors
        let prior_mean = Array1::zeros(n_features);
        let prior_precision = Array2::eye(n_features) * 1e-6; // Very small precision (large variance)
        let prior_alpha = 1e-6; // Very small shape
        let prior_beta = 1e-6; // Very small scale

        Ok(Self {
            prior_mean,
            prior_precision,
            prior_alpha,
            prior_beta,
            fit_intercept,
        })
    }

    /// Create with custom priors
    pub fn with_priors(
        prior_mean: Array1<f64>,
        prior_precision: Array2<f64>,
        prior_alpha: f64,
        prior_beta: f64,
        fit_intercept: bool,
    ) -> StatsResult<Self> {
        checkarray_finite(&prior_mean, "prior_mean")?;
        checkarray_finite(&prior_precision, "prior_precision")?;
        check_positive(prior_alpha, "prior_alpha")?;
        check_positive(prior_beta, "prior_beta")?;

        if prior_precision.nrows() != prior_mean.len()
            || prior_precision.ncols() != prior_mean.len()
        {
            return Err(StatsError::DimensionMismatch(format!(
                "prior_precision shape ({}, {}) must match prior_mean length ({})",
                prior_precision.nrows(),
                prior_precision.ncols(),
                prior_mean.len()
            )));
        }

        Ok(Self {
            prior_mean,
            prior_precision,
            prior_alpha,
            prior_beta,
            fit_intercept,
        })
    }

    /// Fit the Bayesian regression model
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> StatsResult<BayesianRegressionResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;
        let (n_samples_, n_features) = x.dim();

        if y.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples_
            )));
        }

        if n_samples_ < 2 {
            return Err(StatsError::InvalidArgument(
                "n_samples_ must be at least 2".to_string(),
            ));
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean();

            let mut x_centered = x.to_owned();
            for mut row in x_centered.rows_mut() {
                row -= &x_mean;
            }

            let y_centered = &y.to_owned() - y_mean;

            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.to_owned(), y.to_owned(), None, None)
        };

        // Compute posterior parameters
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        // Posterior precision
        let posterior_precision = &self.prior_precision + &xtx;
        let posterior_covariance =
            scirs2_linalg::inv(&posterior_precision.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Failed to invert posterior precision: {}", e))
            })?;

        // Posterior mean
        let prior_contribution = self.prior_precision.dot(&self.prior_mean);
        let data_contribution = &xty;
        let posterior_mean = posterior_covariance.dot(&(&prior_contribution + data_contribution));

        // Posterior shape and scale for noise variance
        let posterior_alpha = self.prior_alpha + n_samples_ as f64 / 2.0;

        // Compute residual sum of squares
        let y_pred = x_centered.dot(&posterior_mean);
        let residuals = &y_centered - &y_pred;
        let rss = residuals.dot(&residuals);

        // Prior contribution to scale
        let prior_quad_form = (&self.prior_mean - &posterior_mean).t().dot(
            &self
                .prior_precision
                .dot(&(&self.prior_mean - &posterior_mean)),
        );

        let posterior_beta = self.prior_beta + 0.5 * (rss + prior_quad_form);

        // Compute log marginal likelihood
        let log_marginal = self.compute_log_marginal_likelihood(
            &x_centered,
            &y_centered,
            &posterior_precision,
            posterior_alpha,
            posterior_beta,
        )?;

        Ok(BayesianRegressionResult {
            posterior_mean,
            posterior_covariance,
            posterior_alpha,
            posterior_beta,
            n_samples_,
            n_features,
            x_mean,
            y_mean,
            log_marginal_likelihood: log_marginal,
        })
    }

    /// Compute log marginal likelihood
    fn compute_log_marginal_likelihood(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
        posterior_precision: &Array2<f64>,
        posterior_alpha: f64,
        posterior_beta: f64,
    ) -> StatsResult<f64> {
        let n = x.nrows() as f64;
        let _p = x.ncols() as f64;

        // Log determinant terms
        let prior_log_det =
            scirs2_linalg::det(&self.prior_precision.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Failed to compute prior determinant: {}", e))
            })?;

        let posterior_log_det =
            scirs2_linalg::det(&posterior_precision.view(), None).map_err(|e| {
                StatsError::ComputationError(format!(
                    "Failed to compute posterior determinant: {}",
                    e
                ))
            })?;

        if prior_log_det <= 0.0 || posterior_log_det <= 0.0 {
            return Err(StatsError::ComputationError(
                "Precision matrices must be positive definite".to_string(),
            ));
        }

        // Gamma function terms
        let gamma_ratio = gamma_log(posterior_alpha) - gamma_log(self.prior_alpha);

        // Assemble log marginal likelihood
        let log_ml = -0.5 * n * (2.0 * std::f64::consts::PI).ln() + 0.5 * prior_log_det.ln()
            - 0.5 * posterior_log_det.ln()
            + self.prior_alpha * self.prior_beta.ln()
            - posterior_alpha * posterior_beta.ln()
            + gamma_ratio;

        Ok(log_ml)
    }

    /// Make predictions on new data
    pub fn predict(
        &self,
        x: ArrayView2<f64>,
        result: &BayesianRegressionResult,
    ) -> StatsResult<BayesianPredictionResult> {
        checkarray_finite(&x, "x")?;
        let (n_test, n_features) = x.dim();

        if n_features != result.n_features {
            return Err(StatsError::DimensionMismatch(format!(
                "x has {} features, expected {}",
                n_features, result.n_features
            )));
        }

        // Center test data if model was fit with intercept
        let x_centered = if let Some(ref x_mean) = result.x_mean {
            let mut x_c = x.to_owned();
            for mut row in x_c.rows_mut() {
                row -= x_mean;
            }
            x_c
        } else {
            x.to_owned()
        };

        // Predictive mean
        let y_pred_centered = x_centered.dot(&result.posterior_mean);
        let y_pred = if let Some(y_mean) = result.y_mean {
            &y_pred_centered + y_mean
        } else {
            y_pred_centered.clone()
        };

        // Predictive variance
        let noise_variance = result.posterior_beta / (result.posterior_alpha - 1.0);
        let mut predictive_variance = Array1::zeros(n_test);

        for i in 0..n_test {
            let x_row = x_centered.row(i);
            let model_variance = x_row.dot(&result.posterior_covariance.dot(&x_row));
            predictive_variance[i] = noise_variance * (1.0 + model_variance);
        }

        // Degrees of freedom for t-distribution
        let df = 2.0 * result.posterior_alpha;

        Ok(BayesianPredictionResult {
            mean: y_pred,
            variance: predictive_variance,
            degrees_of_freedom: df,
            credible_interval: None,
        })
    }

    /// Compute credible intervals for predictions
    pub fn predict_with_credible_interval(
        &self,
        x: ArrayView2<f64>,
        result: &BayesianRegressionResult,
        confidence: f64,
    ) -> StatsResult<BayesianPredictionResult> {
        check_probability(confidence, "confidence")?;

        let mut pred_result = self.predict(x, result)?;

        // Compute credible intervals using t-distribution
        let alpha = (1.0 - confidence) / 2.0;
        let df = pred_result.degrees_of_freedom;

        // For simplicity, use normal approximation when df is large
        let t_critical = if df > 30.0 {
            // Use normal approximation
            normal_ppf(1.0 - alpha)?
        } else {
            // Use t-distribution (simplified)
            t_ppf(1.0 - alpha, df)?
        };

        let mut lower_bounds = Array1::zeros(pred_result.mean.len());
        let mut upper_bounds = Array1::zeros(pred_result.mean.len());

        for i in 0..pred_result.mean.len() {
            let std_err = pred_result.variance[i].sqrt();
            lower_bounds[i] = pred_result.mean[i] - t_critical * std_err;
            upper_bounds[i] = pred_result.mean[i] + t_critical * std_err;
        }

        pred_result.credible_interval = Some((lower_bounds, upper_bounds));
        Ok(pred_result)
    }
}

/// Result of Bayesian prediction
#[derive(Debug, Clone)]
pub struct BayesianPredictionResult {
    /// Predictive mean
    pub mean: Array1<f64>,
    /// Predictive variance
    pub variance: Array1<f64>,
    /// Degrees of freedom for t-distribution
    pub degrees_of_freedom: f64,
    /// Credible interval (lower, upper) if computed
    pub credible_interval: Option<(Array1<f64>, Array1<f64>)>,
}

/// Automatic Relevance Determination (ARD) Bayesian regression
///
/// ARD uses separate precision parameters for each feature to perform
/// automatic feature selection by driving irrelevant features to zero.
#[derive(Debug, Clone)]
pub struct ARDBayesianRegression {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Initial alpha (precision) parameters
    pub alpha_init: Option<Array1<f64>>,
    /// Initial beta (noise precision) parameter
    pub beta_init: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
}

impl Default for ARDBayesianRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl ARDBayesianRegression {
    /// Create a new ARD Bayesian regression model
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_init: None,
            beta_init: 1.0,
            fit_intercept: true,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit ARD Bayesian regression using iterative optimization
    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> StatsResult<ARDRegressionResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;
        let (n_samples_, n_features) = x.dim();

        if y.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples_
            )));
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean();

            let mut x_centered = x.to_owned();
            for mut row in x_centered.rows_mut() {
                row -= &x_mean;
            }

            let y_centered = &y.to_owned() - y_mean;

            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.to_owned(), y.to_owned(), None, None)
        };

        // Initialize hyperparameters
        let mut alpha = self
            .alpha_init
            .clone()
            .unwrap_or_else(|| Array1::from_elem(n_features, 1.0));
        let mut beta = self.beta_init;

        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        let mut prev_log_ml = f64::NEG_INFINITY;

        for iteration in 0..self.max_iter {
            // Update posterior mean and covariance
            let alpha_diag = Array2::from_diag(&alpha);
            let precision = &alpha_diag + beta * &xtx;

            let covariance = scirs2_linalg::inv(&precision.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Failed to invert precision: {}", e))
            })?;

            let mean = beta * covariance.dot(&xty);

            // Update alpha (feature precisions)
            let mut new_alpha = Array1::zeros(n_features);
            for i in 0..n_features {
                let gamma_i = 1.0 - alpha[i] * covariance[[i, i]];
                new_alpha[i] = gamma_i / (mean[i] * mean[i]);

                // Prevent numerical issues
                if !new_alpha[i].is_finite() || new_alpha[i] < 1e-12 {
                    new_alpha[i] = 1e-12;
                }
            }

            // Update beta (noise precision)
            let y_pred = x_centered.dot(&mean);
            let residuals = &y_centered - &y_pred;
            let rss = residuals.dot(&residuals);

            let _trace_cov = covariance.diag().sum();
            let new_beta =
                (n_samples_ as f64 - new_alpha.sum() + alpha.dot(&covariance.diag())) / rss;

            // Check convergence
            let log_ml = self.compute_ard_log_marginal_likelihood(
                &x_centered,
                &y_centered,
                &new_alpha,
                new_beta,
            )?;

            if (log_ml - prev_log_ml).abs() < self.tol {
                alpha = new_alpha;
                beta = new_beta;
                break;
            }

            alpha = new_alpha;
            beta = new_beta;
            prev_log_ml = log_ml;

            if iteration == self.max_iter - 1 {
                return Err(StatsError::ComputationError(format!(
                    "ARD failed to converge after {} iterations",
                    self.max_iter
                )));
            }
        }

        // Final posterior computation
        let alpha_diag = Array2::from_diag(&alpha);
        let precision = &alpha_diag + beta * &xtx;
        let covariance = scirs2_linalg::inv(&precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute final covariance: {}", e))
        })?;
        let mean = beta * covariance.dot(&xty);

        Ok(ARDRegressionResult {
            posterior_mean: mean,
            posterior_covariance: covariance,
            alpha,
            beta,
            n_samples_,
            n_features,
            x_mean,
            y_mean,
            log_marginal_likelihood: prev_log_ml,
        })
    }

    /// Compute log marginal likelihood for ARD
    fn compute_ard_log_marginal_likelihood(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
        beta: f64,
    ) -> StatsResult<f64> {
        let n = x.nrows() as f64;
        let p = x.ncols() as f64;

        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);

        let alpha_diag = Array2::from_diag(alpha);
        let precision = &alpha_diag + beta * &xtx;

        let covariance = scirs2_linalg::inv(&precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert precision for log ML: {}", e))
        })?;

        let mean = beta * covariance.dot(&xty);

        // Compute log determinant
        let log_det_precision = scirs2_linalg::det(&precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        if log_det_precision <= 0.0 {
            return Err(StatsError::ComputationError(
                "Precision matrix must be positive definite".to_string(),
            ));
        }

        // Compute quadratic forms
        let y_pred = x.dot(&mean);
        let residuals = y - &y_pred;
        let data_fit = beta * residuals.dot(&residuals);
        let penalty = alpha
            .iter()
            .zip(mean.iter())
            .map(|(&a, &m)| a * m * m)
            .sum::<f64>();

        let log_ml = 0.5
            * (p * alpha.mapv(f64::ln).sum() + n * beta.ln() + log_det_precision.ln()
                - n * (2.0 * std::f64::consts::PI).ln()
                - data_fit
                - penalty);

        Ok(log_ml)
    }
}

/// Result of ARD Bayesian regression fit
#[derive(Debug, Clone)]
pub struct ARDRegressionResult {
    /// Posterior mean of coefficients
    pub posterior_mean: Array1<f64>,
    /// Posterior covariance of coefficients
    pub posterior_covariance: Array2<f64>,
    /// Feature precision parameters (higher = less relevant)
    pub alpha: Array1<f64>,
    /// Noise precision parameter
    pub beta: f64,
    /// Number of training samples
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
    /// Training data mean (for centering)
    pub x_mean: Option<Array1<f64>>,
    /// Training target mean (for centering)
    pub y_mean: Option<f64>,
    /// Log marginal likelihood
    pub log_marginal_likelihood: f64,
}

// Helper functions for statistical distributions

/// Log of gamma function (simplified implementation)
#[allow(dead_code)]
fn gamma_log(x: f64) -> f64 {
    // Using Stirling's approximation for simplicity
    // In practice, you'd use a more accurate implementation
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 1.0 {
        return gamma_log(x + 1.0) - x.ln();
    }

    0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * x.ln() - x + 1.0 / (12.0 * x)
}

/// Normal distribution percent point function (inverse CDF)
#[allow(dead_code)]
fn normal_ppf(p: f64) -> StatsResult<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "p must be between 0 and 1".to_string(),
        ));
    }

    // Using Box-Muller inspired approximation
    // In practice, you'd use a more accurate inverse error function
    let q = p - 0.5;
    let result = if q.abs() < 0.5 {
        let r = q * q;
        let num =
            (((-25.44106049637) * r + 41.39119773534) * r + (-18.61500062529)) * r + 2.50662823884;
        let den = (((-7.784894002430) * r + 14.38718147627) * r + (-3.47396220392)) * r + 1.0;
        q * num / den
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let num = (2.01033439929 * r.ln() + 4.8232411251) * r.ln() + 6.6;
        let result = (num.exp() - 1.0).sqrt();
        if q < 0.0 {
            -result
        } else {
            result
        }
    };

    Ok(result)
}

/// Student's t distribution percent point function (simplified)
#[allow(dead_code)]
fn t_ppf(p: f64, df: f64) -> StatsResult<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "p must be between 0 and 1".to_string(),
        ));
    }

    // Simplified approximation - in practice use proper t-distribution
    let z = normal_ppf(p)?;

    if df > 4.0 {
        let correction = z * z * z / (4.0 * df) + z * z * z * z * z / (96.0 * df * df);
        Ok(z + correction)
    } else {
        // Very rough approximation for small df
        Ok(z * (1.0 + (z * z + 1.0) / (4.0 * df)))
    }
}
