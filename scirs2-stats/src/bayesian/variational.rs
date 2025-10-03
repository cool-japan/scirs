//! Variational inference for Bayesian models
//!
//! This module implements variational inference methods as alternatives to MCMC
//! for approximate Bayesian inference.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::validation::*;
use statrs::statistics::Statistics;
use std::f64::consts::PI;

/// Mean-field variational inference for Bayesian linear regression
///
/// Approximates the posterior with a factorized normal distribution:
/// q(β, τ) = q(β)q(τ) where q(β) ~ N(μ_β, Σ_β) and q(τ) ~ Gamma(a_τ, b_τ)
#[derive(Debug, Clone)]
pub struct VariationalBayesianRegression {
    /// Variational mean for coefficients
    pub mean_beta: Array1<f64>,
    /// Variational covariance for coefficients
    pub cov_beta: Array2<f64>,
    /// Variational shape parameter for precision
    pub shape_tau: f64,
    /// Variational rate parameter for precision
    pub rate_tau: f64,
    /// Prior parameters
    pub prior_mean_beta: Array1<f64>,
    pub prior_cov_beta: Array2<f64>,
    pub priorshape_tau: f64,
    pub prior_rate_tau: f64,
    /// Model dimensionality
    pub n_features: usize,
    /// Whether to fit intercept
    pub fit_intercept: bool,
}

impl VariationalBayesianRegression {
    /// Create a new variational Bayesian regression model
    pub fn new(n_features: usize, fit_intercept: bool) -> Result<Self> {
        check_positive(n_features, "n_features")?;

        // Initialize with weakly informative priors
        let prior_mean_beta = Array1::zeros(n_features);
        let prior_cov_beta = Array2::eye(n_features) * 100.0; // Large variance = weak prior
        let priorshape_tau = 1e-3;
        let prior_rate_tau = 1e-3;

        Ok(Self {
            mean_beta: prior_mean_beta.clone(),
            cov_beta: prior_cov_beta.clone(),
            shape_tau: priorshape_tau,
            rate_tau: prior_rate_tau,
            prior_mean_beta,
            prior_cov_beta,
            priorshape_tau,
            prior_rate_tau,
            n_features,
            fit_intercept,
        })
    }

    /// Set custom priors
    pub fn with_priors(
        mut self,
        prior_mean_beta: Array1<f64>,
        prior_cov_beta: Array2<f64>,
        priorshape_tau: f64,
        prior_rate_tau: f64,
    ) -> Result<Self> {
        checkarray_finite(&prior_mean_beta, "prior_mean_beta")?;
        checkarray_finite(&prior_cov_beta, "prior_cov_beta")?;
        check_positive(priorshape_tau, "priorshape_tau")?;
        check_positive(prior_rate_tau, "prior_rate_tau")?;

        self.prior_mean_beta = prior_mean_beta.clone();
        self.prior_cov_beta = prior_cov_beta.clone();
        self.priorshape_tau = priorshape_tau;
        self.prior_rate_tau = prior_rate_tau;
        self.mean_beta = prior_mean_beta;
        self.cov_beta = prior_cov_beta;
        self.shape_tau = priorshape_tau;
        self.rate_tau = prior_rate_tau;

        Ok(self)
    }

    /// Fit the model using coordinate ascent variational inference
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<VariationalRegressionResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;
        check_positive(max_iter, "max_iter")?;
        check_positive(tol, "tol")?;

        let (n_samples_, n_features) = x.dim();
        if y.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples_
            )));
        }

        if n_features != self.n_features {
            return Err(StatsError::DimensionMismatch(format!(
                "x features ({}) must match model features ({})",
                n_features, self.n_features
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

        // Precompute matrices
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);
        let yty = y_centered.dot(&y_centered);

        // Prior precision matrix
        let prior_precision =
            scirs2_linalg::inv(&self.prior_cov_beta.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Failed to invert prior covariance: {}", e))
            })?;

        let mut prev_elbo = f64::NEG_INFINITY;
        let mut elbo_history = Vec::new();

        for _iter in 0..max_iter {
            // Update q(β)
            self.update_beta_variational(&xtx, &xty, &prior_precision)?;

            // Update q(τ)
            self.update_tau_variational(n_samples_ as f64, &xtx, yty)?;

            // Compute ELBO
            let elbo = self.compute_elbo(n_samples_ as f64, &xtx, &xty, yty, &prior_precision)?;
            elbo_history.push(elbo);

            // Check convergence
            if _iter > 0 && (elbo - prev_elbo).abs() < tol {
                break;
            }

            prev_elbo = elbo;
        }

        Ok(VariationalRegressionResult {
            mean_beta: self.mean_beta.clone(),
            cov_beta: self.cov_beta.clone(),
            shape_tau: self.shape_tau,
            rate_tau: self.rate_tau,
            elbo: prev_elbo,
            elbo_history: elbo_history.clone(),
            n_samples_,
            n_features: self.n_features,
            x_mean,
            y_mean,
            converged: elbo_history.len() < max_iter,
        })
    }

    /// Update variational distribution for β
    fn update_beta_variational(
        &mut self,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        prior_precision: &Array2<f64>,
    ) -> Result<()> {
        // Expected _precision: E[τ] = shape / rate
        let expected_tau = self.shape_tau / self.rate_tau;

        // Posterior _precision
        let precision_beta = prior_precision + &(xtx * expected_tau);

        // Posterior covariance
        self.cov_beta = scirs2_linalg::inv(&precision_beta.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert precision: {}", e))
        })?;

        // Posterior mean
        let prior_contrib = prior_precision.dot(&self.prior_mean_beta);
        let data_contrib = xty * expected_tau;
        self.mean_beta = self.cov_beta.dot(&(prior_contrib + data_contrib));

        Ok(())
    }

    /// Update variational distribution for τ
    fn update_tau_variational(
        &mut self,
        n_samples_: f64,
        xtx: &Array2<f64>,
        yty: f64,
    ) -> Result<()> {
        // Shape parameter
        self.shape_tau = self.priorshape_tau + n_samples_ / 2.0;

        // Rate parameter
        let expected_beta_outer = &self.cov_beta + outer_product(&self.mean_beta);
        let trace_term = (xtx * &expected_beta_outer).sum();
        let quadratic_term = 2.0 * self.mean_beta.dot(&xtx.dot(&self.mean_beta));

        self.rate_tau = self.prior_rate_tau + 0.5 * (yty - quadratic_term + trace_term);

        Ok(())
    }

    /// Compute Evidence Lower BOund (ELBO)
    fn compute_elbo(
        &self,
        n_samples_: f64,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
        prior_precision: &Array2<f64>,
    ) -> Result<f64> {
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_log_tau = digamma(self.shape_tau) - self.rate_tau.ln();

        // E[log p(y|X,β,τ)]
        let diff =
            yty - 2.0 * self.mean_beta.dot(xty) + self.mean_beta.dot(&xtx.dot(&self.mean_beta));
        let trace_term = (xtx * &self.cov_beta).sum();
        let likelihood_term = 0.5 * n_samples_ * expected_log_tau
            - 0.5 * n_samples_ * (2.0_f64 * PI).ln()
            - 0.5 * expected_tau * (diff + trace_term);

        // E[log p(β)]
        let beta_diff = &self.mean_beta - &self.prior_mean_beta;
        let beta_quad = beta_diff.dot(&prior_precision.dot(&beta_diff));
        let beta_trace = (prior_precision * &self.cov_beta).sum();

        let prior_det = scirs2_linalg::det(&prior_precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        let beta_prior_term = 0.5 * prior_det.ln()
            - 0.5 * self.n_features as f64 * (2.0_f64 * PI).ln()
            - 0.5 * (beta_quad + beta_trace);

        // E[log p(τ)]
        let tau_prior_term = self.priorshape_tau * self.prior_rate_tau.ln()
            - lgamma(self.priorshape_tau)
            + (self.priorshape_tau - 1.0) * expected_log_tau
            - self.prior_rate_tau * expected_tau;

        // -E[log q(β)]
        let var_det = scirs2_linalg::det(&self.cov_beta.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;
        let beta_entropy =
            0.5 * self.n_features as f64 * (1.0 + (2.0_f64 * PI).ln()) + 0.5 * var_det.ln();

        // -E[log q(τ)]
        let tau_entropy = self.shape_tau - self.rate_tau.ln()
            + lgamma(self.shape_tau)
            + (1.0 - self.shape_tau) * digamma(self.shape_tau);

        Ok(likelihood_term + beta_prior_term + tau_prior_term + beta_entropy + tau_entropy)
    }

    /// Predict on new data
    pub fn predict(
        &self,
        x: ArrayView2<f64>,
        result: &VariationalRegressionResult,
    ) -> Result<VariationalPredictionResult> {
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
        let y_pred_centered = x_centered.dot(&result.mean_beta);
        let y_pred = if let Some(y_mean) = result.y_mean {
            &y_pred_centered + y_mean
        } else {
            y_pred_centered.clone()
        };

        // Predictive variance
        let expected_noise_variance = result.rate_tau / result.shape_tau;
        let mut predictive_variance = Array1::zeros(n_test);

        for i in 0..n_test {
            let x_row = x_centered.row(i);
            let model_variance = x_row.dot(&result.cov_beta.dot(&x_row));
            predictive_variance[i] = expected_noise_variance + model_variance;
        }

        Ok(VariationalPredictionResult {
            mean: y_pred,
            variance: predictive_variance.clone(),
            model_uncertainty: predictive_variance.mapv(|v| (v - expected_noise_variance).max(0.0)),
            noise_variance: expected_noise_variance,
        })
    }
}

/// Results from variational Bayesian regression
#[derive(Debug, Clone)]
pub struct VariationalRegressionResult {
    /// Posterior mean of coefficients
    pub mean_beta: Array1<f64>,
    /// Posterior covariance of coefficients  
    pub cov_beta: Array2<f64>,
    /// Posterior shape parameter for precision
    pub shape_tau: f64,
    /// Posterior rate parameter for precision
    pub rate_tau: f64,
    /// Final ELBO value
    pub elbo: f64,
    /// ELBO history during optimization
    pub elbo_history: Vec<f64>,
    /// Number of training samples
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
    /// Training data mean (for centering)
    pub x_mean: Option<Array1<f64>>,
    /// Training target mean (for centering)
    pub y_mean: Option<f64>,
    /// Whether optimization converged
    pub converged: bool,
}

impl VariationalRegressionResult {
    /// Get posterior standard deviations of coefficients
    pub fn std_beta(&self) -> Array1<f64> {
        self.cov_beta.diag().mapv(f64::sqrt)
    }

    /// Get posterior mean and standard deviation of noise precision
    pub fn precision_stats(&self) -> (f64, f64) {
        let mean = self.shape_tau / self.rate_tau;
        let variance = self.shape_tau / (self.rate_tau * self.rate_tau);
        (mean, variance.sqrt())
    }

    /// Compute credible intervals for coefficients
    pub fn credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;

        let n_features = self.mean_beta.len();
        let mut intervals = Array2::zeros((n_features, 2));
        let alpha = (1.0 - confidence) / 2.0;

        // Use normal approximation for coefficients
        for i in 0..n_features {
            let mean = self.mean_beta[i];
            let std = self.cov_beta[[i, i]].sqrt();

            // Using standard normal quantiles (approximate)
            let z_critical = normal_ppf(1.0 - alpha)?;
            intervals[[i, 0]] = mean - z_critical * std;
            intervals[[i, 1]] = mean + z_critical * std;
        }

        Ok(intervals)
    }
}

/// Results from variational prediction
#[derive(Debug, Clone)]
pub struct VariationalPredictionResult {
    /// Predictive mean
    pub mean: Array1<f64>,
    /// Total predictive variance (model + noise)
    pub variance: Array1<f64>,
    /// Model uncertainty component
    pub model_uncertainty: Array1<f64>,
    /// Noise variance
    pub noise_variance: f64,
}

impl VariationalPredictionResult {
    /// Get predictive standard deviations
    pub fn std(&self) -> Array1<f64> {
        self.variance.mapv(f64::sqrt)
    }

    /// Compute predictive credible intervals
    pub fn credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;

        let n_predictions = self.mean.len();
        let mut intervals = Array2::zeros((n_predictions, 2));
        let alpha = (1.0 - confidence) / 2.0;

        let z_critical = normal_ppf(1.0 - alpha)?;

        for i in 0..n_predictions {
            let mean = self.mean[i];
            let std = self.variance[i].sqrt();
            intervals[[i, 0]] = mean - z_critical * std;
            intervals[[i, 1]] = mean + z_critical * std;
        }

        Ok(intervals)
    }
}

/// Automatic Relevance Determination with Variational Inference
///
/// Uses sparse priors to perform automatic feature selection
#[derive(Debug, Clone)]
pub struct VariationalARD {
    /// Variational mean for coefficients
    pub mean_beta: Array1<f64>,
    /// Variational variance for coefficients (diagonal)
    pub var_beta: Array1<f64>,
    /// Variational parameters for precision (alpha)
    pub shape_alpha: Array1<f64>,
    pub rate_alpha: Array1<f64>,
    /// Variational parameters for noise precision
    pub shape_tau: f64,
    pub rate_tau: f64,
    /// Prior parameters
    pub priorshape_alpha: f64,
    pub prior_rate_alpha: f64,
    pub priorshape_tau: f64,
    pub prior_rate_tau: f64,
    /// Model parameters
    pub n_features: usize,
    pub fit_intercept: bool,
}

impl VariationalARD {
    /// Create new Variational ARD model
    pub fn new(n_features: usize, fit_intercept: bool) -> Result<Self> {
        check_positive(n_features, "n_features")?;

        // Weakly informative priors
        let priorshape_alpha = 1e-3;
        let prior_rate_alpha = 1e-3;
        let priorshape_tau = 1e-3;
        let prior_rate_tau = 1e-3;

        Ok(Self {
            mean_beta: Array1::zeros(n_features),
            var_beta: Array1::from_elem(n_features, 1.0),
            shape_alpha: Array1::from_elem(n_features, priorshape_alpha),
            rate_alpha: Array1::from_elem(n_features, prior_rate_alpha),
            shape_tau: priorshape_tau,
            rate_tau: prior_rate_tau,
            priorshape_alpha,
            prior_rate_alpha,
            priorshape_tau,
            prior_rate_tau,
            n_features,
            fit_intercept,
        })
    }

    /// Fit ARD model using variational inference
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<VariationalARDResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;
        check_positive(max_iter, "max_iter")?;
        check_positive(tol, "tol")?;

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

        // Precompute matrices
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);
        let yty = y_centered.dot(&y_centered);

        let mut prev_elbo = f64::NEG_INFINITY;
        let mut elbo_history = Vec::new();

        for _iter in 0..max_iter {
            // Update q(β)
            self.update_beta_ard(&xtx, &xty)?;

            // Update q(α)
            self.update_alpha_ard()?;

            // Update q(τ)
            self.update_tau_ard(n_samples_ as f64, &xtx, yty)?;

            // Compute ELBO
            let elbo = self.compute_elbo_ard(n_samples_ as f64, &xtx, &xty, yty)?;
            elbo_history.push(elbo);

            // Check convergence
            if _iter > 0 && (elbo - prev_elbo).abs() < tol {
                break;
            }

            // Prune irrelevant features
            if _iter % 10 == 0 {
                self.prune_features()?;
            }

            prev_elbo = elbo;
        }

        Ok(VariationalARDResult {
            mean_beta: self.mean_beta.clone(),
            var_beta: self.var_beta.clone(),
            shape_alpha: self.shape_alpha.clone(),
            rate_alpha: self.rate_alpha.clone(),
            shape_tau: self.shape_tau,
            rate_tau: self.rate_tau,
            elbo: prev_elbo,
            elbo_history: elbo_history.clone(),
            n_samples_,
            n_features: self.n_features,
            x_mean,
            y_mean,
            converged: elbo_history.len() < max_iter,
        })
    }

    /// Update variational distribution for β in ARD model
    fn update_beta_ard(&mut self, xtx: &Array2<f64>, xty: &Array1<f64>) -> Result<()> {
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_alpha = &self.shape_alpha / &self.rate_alpha;

        // Update variance (diagonal approximation)
        for i in 0..self.n_features {
            let precision_i = expected_alpha[i] + expected_tau * xtx[[i, i]];
            self.var_beta[i] = 1.0 / precision_i;
        }

        // Update mean
        for i in 0..self.n_features {
            let sum_j = (0..self.n_features)
                .filter(|&j| j != i)
                .map(|j| xtx[[i, j]] * self.mean_beta[j])
                .sum::<f64>();

            self.mean_beta[i] = expected_tau * self.var_beta[i] * (xty[i] - sum_j);
        }

        Ok(())
    }

    /// Update variational distribution for α (precision parameters)
    fn update_alpha_ard(&mut self) -> Result<()> {
        for i in 0..self.n_features {
            self.shape_alpha[i] = self.priorshape_alpha + 0.5;
            self.rate_alpha[i] =
                self.prior_rate_alpha + 0.5 * (self.mean_beta[i].powi(2) + self.var_beta[i]);
        }

        Ok(())
    }

    /// Update variational distribution for τ (noise precision)
    fn update_tau_ard(&mut self, n_samples_: f64, xtx: &Array2<f64>, yty: f64) -> Result<()> {
        self.shape_tau = self.priorshape_tau + n_samples_ / 2.0;

        let mut quadratic_term = 0.0;
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                if i == j {
                    quadratic_term += xtx[[i, j]] * (self.mean_beta[i].powi(2) + self.var_beta[i]);
                } else {
                    quadratic_term += xtx[[i, j]] * self.mean_beta[i] * self.mean_beta[j];
                }
            }
        }

        self.rate_tau = self.prior_rate_tau
            + 0.5 * (yty - 2.0 * self.mean_beta.dot(&xtx.dot(&self.mean_beta)) + quadratic_term);

        Ok(())
    }

    /// Compute ELBO for ARD model
    fn compute_elbo_ard(
        &self,
        n_samples_: f64,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
    ) -> Result<f64> {
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_log_tau = digamma(self.shape_tau) - self.rate_tau.ln();

        // Likelihood term
        let mut quadratic_form = yty - 2.0 * self.mean_beta.dot(xty);
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                if i == j {
                    quadratic_form += xtx[[i, j]] * (self.mean_beta[i].powi(2) + self.var_beta[i]);
                } else {
                    quadratic_form += xtx[[i, j]] * self.mean_beta[i] * self.mean_beta[j];
                }
            }
        }

        let likelihood_term = 0.5 * n_samples_ * expected_log_tau
            - 0.5 * n_samples_ * (2.0_f64 * PI).ln()
            - 0.5 * expected_tau * quadratic_form;

        // Prior terms
        let mut prior_term = 0.0;
        for i in 0..self.n_features {
            let expected_alpha_i = self.shape_alpha[i] / self.rate_alpha[i];
            let expected_log_alpha_i = digamma(self.shape_alpha[i]) - self.rate_alpha[i].ln();

            prior_term += 0.5 * expected_log_alpha_i
                - 0.5 * (2.0_f64 * PI).ln()
                - 0.5 * expected_alpha_i * (self.mean_beta[i].powi(2) + self.var_beta[i]);
        }

        // Entropy terms
        let mut entropy_term = 0.0;
        for i in 0..self.n_features {
            entropy_term += 0.5 * (1.0 + (2.0 * PI * self.var_beta[i]).ln());
        }

        Ok(likelihood_term + prior_term + entropy_term)
    }

    /// Prune features with small precision (large variance in prior)
    fn prune_features(&mut self) -> Result<()> {
        let threshold = 1e12; // Large precision = irrelevant feature

        for i in 0..self.n_features {
            let expected_alpha = self.shape_alpha[i] / self.rate_alpha[i];
            if expected_alpha > threshold {
                // Feature is irrelevant, set to zero
                self.mean_beta[i] = 0.0;
                self.var_beta[i] = 1e-12;
            }
        }

        Ok(())
    }

    /// Get relevance scores for features
    pub fn feature_relevance(&self) -> Array1<f64> {
        let expected_alpha = &self.shape_alpha / &self.rate_alpha;
        // Relevance is inverse of precision (features with low precision are more relevant)
        expected_alpha.mapv(|alpha| 1.0 / alpha)
    }
}

/// Results from Variational ARD
#[derive(Debug, Clone)]
pub struct VariationalARDResult {
    /// Posterior mean of coefficients
    pub mean_beta: Array1<f64>,
    /// Posterior variance of coefficients
    pub var_beta: Array1<f64>,
    /// Posterior shape parameters for feature precisions
    pub shape_alpha: Array1<f64>,
    /// Posterior rate parameters for feature precisions
    pub rate_alpha: Array1<f64>,
    /// Posterior shape parameter for noise precision
    pub shape_tau: f64,
    /// Posterior rate parameter for noise precision
    pub rate_tau: f64,
    /// Final ELBO value
    pub elbo: f64,
    /// ELBO history
    pub elbo_history: Vec<f64>,
    /// Number of training samples
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
    /// Training data mean
    pub x_mean: Option<Array1<f64>>,
    /// Training target mean
    pub y_mean: Option<f64>,
    /// Whether optimization converged
    pub converged: bool,
}

impl VariationalARDResult {
    /// Get selected features based on relevance threshold
    pub fn selected_features(&self, threshold: f64) -> Vec<usize> {
        let expected_alpha = &self.shape_alpha / &self.rate_alpha;
        expected_alpha
            .iter()
            .enumerate()
            .filter(|(_, &alpha)| alpha < threshold) // Low precision = high relevance
            .map(|(i, _)| i)
            .collect()
    }

    /// Get feature importance scores
    pub fn feature_importance(&self) -> Array1<f64> {
        self.mean_beta.mapv(f64::abs)
    }
}

// Helper functions

/// Compute outer product of a vector with itself
#[allow(dead_code)]
fn outer_product(v: &Array1<f64>) -> Array2<f64> {
    let n = v.len();
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = v[i] * v[j];
        }
    }
    result
}

/// Approximate normal PPF using rational approximation
#[allow(dead_code)]
fn normal_ppf(p: f64) -> Result<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "p must be between 0 and 1".to_string(),
        ));
    }

    // Beasley-Springer-Moro algorithm
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        Ok(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
        )
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        Ok(
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0),
        )
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        Ok(
            (-((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
        )
    }
}

/// Digamma function (approximate)
#[allow(dead_code)]
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 8.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }

    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0
}

/// Log gamma function (approximate)
#[allow(dead_code)]
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 1.0 {
        return lgamma(x + 1.0) - x.ln();
    }

    // Stirling's approximation
    0.5 * (2.0_f64 * PI).ln() + (x - 0.5) * x.ln() - x + 1.0 / (12.0 * x)
}
