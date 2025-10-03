//! Enhanced Bayesian regression methods
//!
//! This module provides advanced Bayesian regression techniques including
//! variational inference, hierarchical models, and robust Bayesian regression.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, One, ToPrimitive, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use std::marker::PhantomData;

/// Enhanced Bayesian linear regression with multiple inference methods
#[derive(Debug, Clone)]
pub struct EnhancedBayesianRegression<F> {
    /// Design matrix (X)
    pub design_matrix: Array2<F>,
    /// Response vector (y)
    pub response: Array1<F>,
    /// Prior parameters
    pub prior: BayesianRegressionPrior<F>,
    /// Inference method
    pub inference_method: InferenceMethod,
    /// Model configuration
    pub config: BayesianRegressionConfig,
    _phantom: PhantomData<F>,
}

/// Prior specification for Bayesian regression
#[derive(Debug, Clone)]
pub struct BayesianRegressionPrior<F> {
    /// Prior mean for coefficients
    pub beta_mean: Array1<F>,
    /// Prior precision matrix for coefficients
    pub beta_precision: Array2<F>,
    /// Prior shape parameter for noise precision
    pub noiseshape: F,
    /// Prior rate parameter for noise precision
    pub noise_rate: F,
}

/// Inference methods for Bayesian regression
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceMethod {
    /// Exact conjugate inference (when applicable)
    Exact,
    /// Variational Bayes inference
    VariationalBayes,
    /// MCMC sampling
    MCMC,
    /// Expectation Propagation
    ExpectationPropagation,
}

/// Configuration for Bayesian regression
#[derive(Debug, Clone)]
pub struct BayesianRegressionConfig {
    /// Maximum iterations for iterative methods
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BayesianRegressionConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance: 1e-6,
            parallel: true,
            seed: None,
        }
    }
}

/// Posterior results for Bayesian regression
#[derive(Debug, Clone)]
pub struct BayesianRegressionResult<F> {
    /// Posterior mean of coefficients
    pub beta_mean: Array1<F>,
    /// Posterior covariance of coefficients
    pub beta_covariance: Array2<F>,
    /// Posterior mean of noise precision
    pub noise_precision_mean: F,
    /// Posterior variance of noise precision
    pub noise_precision_var: F,
    /// Log marginal likelihood (model evidence)
    pub log_marginal_likelihood: F,
    /// Predictive mean
    pub predictive_mean: Array1<F>,
    /// Predictive variance
    pub predictive_var: Array1<F>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether convergence was achieved
    pub converged: bool,
    /// Number of iterations taken
    pub iterations: usize,
    /// Final tolerance achieved
    pub final_tolerance: f64,
}

impl<F> EnhancedBayesianRegression<F>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + std::fmt::Display
        + 'static
        + std::iter::Sum
        + NumAssign
        + ScalarOperand
        + ToPrimitive
        + FromPrimitive,
{
    /// Create new enhanced Bayesian regression model
    pub fn new(
        design_matrix: Array2<F>,
        response: Array1<F>,
        prior: BayesianRegressionPrior<F>,
        inference_method: InferenceMethod,
    ) -> StatsResult<Self> {
        checkarray_finite(&design_matrix, "design_matrix")?;
        checkarray_finite(&response, "response")?;
        checkarray_finite(&prior.beta_mean, "beta_mean")?;
        checkarray_finite(&prior.beta_precision, "beta_precision")?;

        let (n, p) = design_matrix.dim();

        if response.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "Response length ({}) must match design _matrix rows ({})",
                response.len(),
                n
            )));
        }

        if prior.beta_mean.len() != p {
            return Err(StatsError::DimensionMismatch(format!(
                "Prior mean length ({}) must match design _matrix columns ({})",
                prior.beta_mean.len(),
                p
            )));
        }

        if prior.beta_precision.nrows() != p || prior.beta_precision.ncols() != p {
            return Err(StatsError::DimensionMismatch(format!(
                "Prior precision shape ({}, {}) must be ({}, {})",
                prior.beta_precision.nrows(),
                prior.beta_precision.ncols(),
                p,
                p
            )));
        }

        Ok(Self {
            design_matrix,
            response,
            prior,
            inference_method,
            config: BayesianRegressionConfig::default(),
            _phantom: PhantomData,
        })
    }

    /// Set configuration
    pub fn with_config(mut self, config: BayesianRegressionConfig) -> Self {
        self.config = config;
        self
    }

    /// Fit the Bayesian regression model
    pub fn fit(&self) -> StatsResult<BayesianRegressionResult<F>> {
        match self.inference_method {
            InferenceMethod::Exact => self.fit_exact(),
            InferenceMethod::VariationalBayes => self.fit_variational_bayes(),
            InferenceMethod::MCMC => self.fit_mcmc(),
            InferenceMethod::ExpectationPropagation => self.fit_expectation_propagation(),
        }
    }

    /// Exact conjugate inference (Normal-Gamma conjugacy)
    fn fit_exact(&self) -> StatsResult<BayesianRegressionResult<F>> {
        let x = &self.design_matrix;
        let y = &self.response;
        let n = x.nrows() as f64;
        let p = x.ncols();

        // Compute posterior parameters using matrix operations
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);

        // Convert to f64 for numerical stability
        let xtx_f64 = xtx.mapv(|v| v.to_f64().unwrap_or(0.0));
        let xty_f64 = xty.mapv(|v| v.to_f64().unwrap_or(0.0));
        let prior_precision_f64 = self
            .prior
            .beta_precision
            .mapv(|v| v.to_f64().unwrap_or(0.0));
        let prior_mean_f64 = self.prior.beta_mean.mapv(|v| v.to_f64().unwrap_or(0.0));
        let noiseshape_f64 = self.prior.noiseshape.to_f64().unwrap_or(1.0);
        let noise_rate_f64 = self.prior.noise_rate.to_f64().unwrap_or(1.0);

        // Posterior precision matrix
        let posterior_precision_f64 = xtx_f64.clone() + prior_precision_f64.clone();

        // Invert posterior precision to get covariance
        let posterior_covariance_f64 = scirs2_linalg::inv(&posterior_precision_f64.view(), None)
            .map_err(|e| {
                StatsError::ComputationError(format!("Failed to invert posterior precision: {}", e))
            })?;

        // Posterior mean
        let posterior_mean_f64 = posterior_covariance_f64
            .dot(&(xtx_f64.dot(&xty_f64) + prior_precision_f64.dot(&prior_mean_f64)));

        // Posterior noise parameters
        let posterior_mean_f: Array1<F> = posterior_mean_f64.mapv(|v| F::from(v).unwrap());
        let residual = y - &x.dot(&posterior_mean_f);
        let residual_sum_squares = residual.dot(&residual).to_f64().unwrap_or(0.0);

        let posterior_noiseshape = noiseshape_f64 + n / 2.0;
        let posterior_noise_rate = noise_rate_f64 + residual_sum_squares / 2.0;

        // Convert back to F type
        let beta_mean = posterior_mean_f64.mapv(|v| F::from(v).unwrap());
        let beta_covariance = posterior_covariance_f64.mapv(|v| F::from(v).unwrap());

        let noise_precision_mean = F::from(posterior_noiseshape / posterior_noise_rate).unwrap();
        let noise_precision_var =
            F::from(posterior_noiseshape / (posterior_noise_rate * posterior_noise_rate)).unwrap();

        // Compute predictive distribution
        let predictive_mean = x.dot(&beta_mean);
        let predictive_var_diag =
            self.compute_predictive_variance(x.view(), &beta_covariance, noise_precision_mean)?;

        // Compute log marginal likelihood
        let log_marginal_likelihood = self.compute_log_marginal_likelihood(
            &xtx_f64,
            &xty_f64,
            &prior_precision_f64,
            &prior_mean_f64,
            noiseshape_f64,
            noise_rate_f64,
            n,
            p,
        )?;

        Ok(BayesianRegressionResult {
            beta_mean,
            beta_covariance,
            noise_precision_mean,
            noise_precision_var,
            log_marginal_likelihood,
            predictive_mean,
            predictive_var: predictive_var_diag,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 1,
                final_tolerance: 0.0,
            },
        })
    }

    /// Variational Bayes inference
    fn fit_variational_bayes(&self) -> StatsResult<BayesianRegressionResult<F>> {
        let x = &self.design_matrix;
        let y = &self.response;
        let (n, p) = x.dim();

        // Initialize variational parameters
        let mut q_beta_mean = self.prior.beta_mean.clone();
        let mut q_beta_precision = self.prior.beta_precision.clone();
        let mut q_noiseshape = self.prior.noiseshape;
        let mut q_noise_rate = self.prior.noise_rate;

        let mut converged = false;
        let mut iterations = 0;
        let mut prev_elbo = F::neg_infinity();

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // Update beta parameters
            let xtx = x.t().dot(x);
            let xty = x.t().dot(y);
            let expected_noise_precision = q_noiseshape / q_noise_rate;

            q_beta_precision =
                self.prior.beta_precision.clone() + xtx.mapv(|v| v * expected_noise_precision);

            let q_beta_covariance = scirs2_linalg::inv(&q_beta_precision.view(), None)
                .map_err(|e| StatsError::ComputationError(format!("VB update failed: {}", e)))?;

            q_beta_mean = q_beta_covariance.dot(
                &(self.prior.beta_precision.dot(&self.prior.beta_mean)
                    + xty.mapv(|v| v * expected_noise_precision)),
            );

            // Update noise parameters
            q_noiseshape = self.prior.noiseshape + F::from(n).unwrap() / F::from(2.0).unwrap();

            let _expected_beta_squared =
                q_beta_mean.dot(&q_beta_mean) + q_beta_covariance.diag().sum();
            let residual_term = y.dot(y) - F::from(2.0).unwrap() * y.dot(&x.dot(&q_beta_mean))
                + x.dot(&q_beta_mean).dot(&x.dot(&q_beta_mean))
                + (x.t().dot(x) * q_beta_covariance).diag().sum();

            q_noise_rate = self.prior.noise_rate + residual_term / F::from(2.0).unwrap();

            // Compute ELBO for convergence check
            let elbo =
                self.compute_elbo(&q_beta_mean, &q_beta_precision, q_noiseshape, q_noise_rate)?;

            if (elbo - prev_elbo).abs() < F::from(self.config.tolerance).unwrap() {
                converged = true;
                break;
            }

            prev_elbo = elbo;
        }

        // Compute final results
        let beta_covariance = scirs2_linalg::inv(&q_beta_precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Final covariance computation failed: {}", e))
        })?;

        let noise_precision_mean = q_noiseshape / q_noise_rate;
        let noise_precision_var = q_noiseshape / (q_noise_rate * q_noise_rate);

        let predictive_mean = x.dot(&q_beta_mean);
        let predictive_var =
            self.compute_predictive_variance(x.view(), &beta_covariance, noise_precision_mean)?;

        let log_marginal_likelihood = prev_elbo; // ELBO approximates log marginal likelihood

        Ok(BayesianRegressionResult {
            beta_mean: q_beta_mean,
            beta_covariance,
            noise_precision_mean,
            noise_precision_var,
            log_marginal_likelihood,
            predictive_mean,
            predictive_var,
            convergence_info: ConvergenceInfo {
                converged,
                iterations,
                final_tolerance: if converged {
                    self.config.tolerance
                } else {
                    f64::INFINITY
                },
            },
        })
    }

    /// MCMC inference using Gibbs sampling
    fn fit_mcmc(&self) -> StatsResult<BayesianRegressionResult<F>> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        use scirs2_core::random::{Distribution, Gamma};

        let x = &self.design_matrix;
        let y = &self.response;
        let (n, p) = x.dim();

        // Initialize MCMC chain
        let n_samples_ = self.config.max_iter;
        let n_burnin = n_samples_ / 4; // 25% burn-in
        let n_thin = 1; // No thinning for simplicity

        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut rng = scirs2_core::random::thread_rng();
                StdRng::from_rng(&mut rng)
            }
        };

        // Initialize parameters
        #[allow(unused_assignments)]
        let mut beta = self.prior.beta_mean.clone();
        let mut noise_precision = self.prior.noiseshape / self.prior.noise_rate;

        // Storage for samples
        let mut beta_samples = Vec::with_capacity(n_samples_ - n_burnin);
        let mut noise_precision_samples_ = Vec::with_capacity(n_samples_ - n_burnin);
        let mut log_likelihood_history = Vec::new();

        // Precompute matrices for efficiency
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);

        // Gibbs sampling
        for iter in 0..n_samples_ {
            // Sample beta | noise_precision, y
            let precision_matrix =
                self.prior.beta_precision.clone() + xtx.mapv(|v| v * noise_precision);

            // Convert to f64 for numerical stability
            let precision_f64 = precision_matrix.mapv(|v| v.to_f64().unwrap_or(0.0));
            let posterior_cov_f64 =
                scirs2_linalg::inv(&precision_f64.view(), None).map_err(|e| {
                    StatsError::ComputationError(format!("MCMC covariance inversion failed: {}", e))
                })?;

            let mean_term = self.prior.beta_precision.dot(&self.prior.beta_mean)
                + xty.mapv(|v| v * noise_precision);
            let posterior_mean_f64 =
                posterior_cov_f64.dot(&mean_term.mapv(|v| v.to_f64().unwrap_or(0.0)));

            // Sample from multivariate normal
            beta =
                self.sample_multivariate_normal(&posterior_mean_f64, &posterior_cov_f64, &mut rng)?;

            // Sample noise_precision | beta, y
            let residual = y - &x.dot(&beta);
            let sum_squared_residuals = residual.dot(&residual).to_f64().unwrap_or(0.0);

            let posteriorshape = self.prior.noiseshape.to_f64().unwrap_or(1.0) + (n as f64) / 2.0;
            let posterior_rate =
                self.prior.noise_rate.to_f64().unwrap_or(1.0) + sum_squared_residuals / 2.0;

            let gamma_dist = Gamma::new(posteriorshape, 1.0 / posterior_rate).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create gamma distribution: {}", e))
            })?;
            noise_precision = F::from(gamma_dist.sample(&mut rng)).unwrap();

            // Store samples after burn-in
            if iter >= n_burnin && (iter - n_burnin).is_multiple_of(n_thin) {
                beta_samples.push(beta.clone());
                noise_precision_samples_.push(noise_precision);
            }

            // Compute log-likelihood for convergence monitoring
            if iter % 100 == 0 {
                let ll = self.compute_mcmc_log_likelihood(&beta, noise_precision)?;
                log_likelihood_history.push(ll);
            }
        }

        // Compute posterior statistics from samples
        let n_kept_samples = beta_samples.len();
        if n_kept_samples == 0 {
            return Err(StatsError::ComputationError(
                "No MCMC samples collected".to_string(),
            ));
        }

        // Posterior mean of beta
        let mut posterior_beta_mean = Array1::zeros(p);
        for sample in &beta_samples {
            posterior_beta_mean += sample;
        }
        posterior_beta_mean /= F::from(n_kept_samples).unwrap();

        // Posterior covariance of beta
        let mut posterior_beta_cov = Array2::zeros((p, p));
        for sample in &beta_samples {
            let centered = sample - &posterior_beta_mean;
            for i in 0..p {
                for j in 0..p {
                    posterior_beta_cov[[i, j]] += centered[i] * centered[j];
                }
            }
        }
        posterior_beta_cov /= F::from(n_kept_samples.saturating_sub(1).max(1)).unwrap();

        // Posterior statistics for noise precision
        let noise_precision_mean = noise_precision_samples_
            .iter()
            .fold(F::zero(), |acc, &x| acc + x)
            / F::from(n_kept_samples).unwrap();

        let noise_precision_var = {
            let mean_sq = noise_precision_samples_
                .iter()
                .map(|&x| (x - noise_precision_mean) * (x - noise_precision_mean))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(n_kept_samples.saturating_sub(1).max(1)).unwrap();
            mean_sq
        };

        // Predictive distribution
        let predictive_mean = x.dot(&posterior_beta_mean);
        let predictive_var =
            self.compute_predictive_variance(x.view(), &posterior_beta_cov, noise_precision_mean)?;

        // Compute final log marginal likelihood estimate
        let final_log_likelihood = if log_likelihood_history.is_empty() {
            self.compute_mcmc_log_likelihood(&posterior_beta_mean, noise_precision_mean)?
        } else {
            *log_likelihood_history.last().unwrap()
        };

        // Check convergence based on effective sample size and stability
        let converged = self.check_mcmc_convergence(&beta_samples, &noise_precision_samples_)?;

        Ok(BayesianRegressionResult {
            beta_mean: posterior_beta_mean,
            beta_covariance: posterior_beta_cov,
            noise_precision_mean,
            noise_precision_var,
            log_marginal_likelihood: final_log_likelihood,
            predictive_mean,
            predictive_var,
            convergence_info: ConvergenceInfo {
                converged,
                iterations: n_samples_,
                final_tolerance: if converged {
                    self.config.tolerance
                } else {
                    f64::INFINITY
                },
            },
        })
    }

    /// Expectation Propagation inference
    fn fit_expectation_propagation(&self) -> StatsResult<BayesianRegressionResult<F>> {
        // For now, fall back to variational Bayes
        // Full EP implementation would be more complex
        self.fit_variational_bayes()
    }

    /// Compute predictive variance
    fn compute_predictive_variance(
        &self,
        x: ArrayView2<F>,
        beta_covariance: &Array2<F>,
        noise_precision_mean: F,
    ) -> StatsResult<Array1<F>> {
        let n = x.nrows();
        let mut predictive_var = Array1::zeros(n);

        for i in 0..n {
            let x_i = x.row(i);
            let var_beta = x_i.dot(&beta_covariance.dot(&x_i));
            let var_noise = F::one() / noise_precision_mean;
            predictive_var[i] = var_beta + var_noise;
        }

        Ok(predictive_var)
    }

    /// Compute log marginal likelihood for exact inference
    fn compute_log_marginal_likelihood(
        &self,
        xtx: &Array2<f64>,
        _xty: &Array1<f64>,
        prior_precision: &Array2<f64>,
        _prior_mean: &Array1<f64>,
        noiseshape: f64,
        noise_rate: f64,
        n: f64,
        p: usize,
    ) -> StatsResult<F> {
        // This is a simplified version - full implementation would include all normalization terms
        let posterior_precision = xtx + prior_precision;
        let det_prior = scirs2_linalg::det(&prior_precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Determinant computation failed: {}", e))
        })?;
        let det_posterior = scirs2_linalg::det(&posterior_precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Determinant computation failed: {}", e))
        })?;

        // Simplified log marginal likelihood computation
        let log_ml = 0.5 * (det_prior / det_posterior).ln() + noiseshape * noise_rate.ln()
            - (n / 2.0) * (2.0 * std::f64::consts::PI).ln();

        Ok(F::from(log_ml).unwrap())
    }

    /// Compute Evidence Lower BOund (ELBO) for variational inference
    fn compute_elbo(
        &self,
        q_beta_mean: &Array1<F>,
        _q_beta_precision: &Array2<F>,
        q_noiseshape: F,
        q_noise_rate: F,
    ) -> StatsResult<F> {
        // Simplified ELBO computation
        // Full implementation would include entropy terms and expected log-likelihood
        let expected_noise_precision = q_noiseshape / q_noise_rate;
        let residual = &self.response - &self.design_matrix.dot(q_beta_mean);
        let data_term = -F::from(0.5).unwrap() * expected_noise_precision * residual.dot(&residual);

        Ok(data_term)
    }

    /// Sample from multivariate normal distribution
    fn sample_multivariate_normal<R: scirs2_core::random::Rng>(
        &self,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
        rng: &mut R,
    ) -> StatsResult<Array1<F>> {
        use scirs2_core::random::{Distribution, StandardNormal};

        let d = mean.len();

        // Cholesky decomposition of covariance
        let chol = scirs2_linalg::cholesky(&covariance.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Cholesky decomposition failed: {}", e))
        })?;

        // Sample from standard normal
        let z: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
        let z_array = Array1::from_vec(z);

        // Transform: mean + L * z where L is lower triangular Cholesky factor
        let sample_f64 = mean + &chol.dot(&z_array);
        let sample = sample_f64.mapv(|x| F::from(x).unwrap());

        Ok(sample)
    }

    /// Compute log-likelihood for MCMC monitoring
    fn compute_mcmc_log_likelihood(&self, beta: &Array1<F>, noise_precision: F) -> StatsResult<F> {
        let x = &self.design_matrix;
        let y = &self.response;
        let n = x.nrows() as f64;

        let residual = y - &x.dot(beta);
        let sum_squared_residuals = residual.dot(&residual).to_f64().unwrap_or(0.0);

        let log_likelihood = (n / 2.0) * noise_precision.to_f64().unwrap_or(1.0).ln()
            - (n / 2.0) * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * noise_precision.to_f64().unwrap_or(1.0) * sum_squared_residuals;

        Ok(F::from(log_likelihood).unwrap())
    }

    /// Check MCMC convergence using various diagnostics
    fn check_mcmc_convergence(
        &self,
        beta_samples: &[Array1<F>],
        noise_precision_samples_: &[F],
    ) -> StatsResult<bool> {
        if beta_samples.len() < 100 {
            return Ok(false); // Need minimum _samples for convergence assessment
        }

        // Split _samples into two halves for Gelman-Rubin diagnostic
        let n = beta_samples.len();
        let mid = n / 2;

        // Simplified convergence check: compare variance of first and second half
        let first_half = &beta_samples[..mid];
        let second_half = &beta_samples[mid..];

        // Check if variance stabilized for first parameter
        if !beta_samples.is_empty() && !beta_samples[0].is_empty() {
            let first_half_var = self
                .compute_sample_variance_1d(&first_half.iter().map(|x| x[0]).collect::<Vec<_>>());
            let second_half_var = self
                .compute_sample_variance_1d(&second_half.iter().map(|x| x[0]).collect::<Vec<_>>());

            let var_ratio =
                first_half_var.max(second_half_var) / first_half_var.min(second_half_var);
            if var_ratio > F::from(2.0).unwrap() {
                return Ok(false); // Variance not stabilized
            }
        }

        // Check effective sample size (simplified)
        let eff_samplesize = self.compute_effective_samplesize(noise_precision_samples_)?;
        if eff_samplesize < 100.0 {
            return Ok(false); // Need larger effective sample size
        }

        Ok(true)
    }

    /// Compute sample variance for 1D samples
    fn compute_sample_variance_1d(&self, samples: &[F]) -> F {
        if samples.is_empty() {
            return F::one();
        }

        let n = samples.len();
        let mean = samples.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap();
        let variance = samples
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n.saturating_sub(1).max(1)).unwrap();

        variance.max(F::from(1e-10).unwrap()) // Avoid zero variance
    }

    /// Compute effective sample size (simplified autocorrelation-based estimate)
    fn compute_effective_samplesize(&self, samples: &[F]) -> StatsResult<f64> {
        if samples.len() < 10 {
            return Ok(samples.len() as f64);
        }

        let n = samples.len();
        let mean = samples.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap();

        // Compute lag-1 autocorrelation (simplified)
        let mut numerator = F::zero();
        let mut denominator = F::zero();

        for i in 0..n - 1 {
            let x_i = samples[i] - mean;
            let x_i1 = samples[i + 1] - mean;
            numerator += x_i * x_i1;
            denominator += x_i * x_i;
        }

        let autocorr = if denominator > F::from(1e-10).unwrap() {
            (numerator / denominator).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };

        // Simplified effective sample size estimate
        let eff_n = if autocorr > 0.1 {
            n as f64 * (1.0 - autocorr) / (1.0 + autocorr)
        } else {
            n as f64
        };

        Ok(eff_n.max(1.0))
    }

    /// Make predictions on new data
    pub fn predict(
        &self,
        x_new: &Array2<F>,
        result: &BayesianRegressionResult<F>,
    ) -> StatsResult<(Array1<F>, Array1<F>)> {
        checkarray_finite(x_new, "x_new")?;

        if x_new.ncols() != self.design_matrix.ncols() {
            return Err(StatsError::DimensionMismatch(format!(
                "New data columns ({}) must match training data columns ({})",
                x_new.ncols(),
                self.design_matrix.ncols()
            )));
        }

        let pred_mean = x_new.dot(&result.beta_mean);
        let pred_var = self.compute_predictive_variance(
            x_new.view(),
            &result.beta_covariance,
            result.noise_precision_mean,
        )?;

        Ok((pred_mean, pred_var))
    }
}

impl<F> BayesianRegressionPrior<F>
where
    F: Float + Zero + One + Copy + ScalarOperand + std::fmt::Display + FromPrimitive,
{
    /// Create uninformative prior
    pub fn uninformative(p: usize) -> Self {
        let beta_mean = Array1::zeros(p);
        let beta_precision = Array2::eye(p) * F::from(1e-6).unwrap(); // Very small precision = large variance
        let noiseshape = F::from(1e-3).unwrap();
        let noise_rate = F::from(1e-3).unwrap();

        Self {
            beta_mean,
            beta_precision,
            noiseshape,
            noise_rate,
        }
    }

    /// Create ridge-like prior
    pub fn ridge(p: usize, alpha: F) -> Self {
        let beta_mean = Array1::zeros(p);
        let beta_precision = Array2::eye(p) * alpha;
        let noiseshape = F::one();
        let noise_rate = F::one();

        Self {
            beta_mean,
            beta_precision,
            noiseshape,
            noise_rate,
        }
    }
}

/// Convenience functions
#[allow(dead_code)]
pub fn bayesian_linear_regression_exact<F>(
    x: Array2<F>,
    y: Array1<F>,
    prior: Option<BayesianRegressionPrior<F>>,
) -> StatsResult<BayesianRegressionResult<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static
        + std::iter::Sum
        + NumAssign
        + ScalarOperand
        + std::fmt::Display
        + ToPrimitive
        + FromPrimitive,
{
    let p = x.ncols();
    let prior = prior.unwrap_or_else(|| BayesianRegressionPrior::uninformative(p));

    let model = EnhancedBayesianRegression::new(x, y, prior, InferenceMethod::Exact)?;
    model.fit()
}

#[allow(dead_code)]
pub fn bayesian_linear_regression_vb<F>(
    x: Array2<F>,
    y: Array1<F>,
    prior: Option<BayesianRegressionPrior<F>>,
    config: Option<BayesianRegressionConfig>,
) -> StatsResult<BayesianRegressionResult<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static
        + std::iter::Sum
        + NumAssign
        + ScalarOperand
        + std::fmt::Display
        + ToPrimitive
        + FromPrimitive,
{
    let p = x.ncols();
    let prior = prior.unwrap_or_else(|| BayesianRegressionPrior::uninformative(p));
    let config = config.unwrap_or_default();

    let model = EnhancedBayesianRegression::new(x, y, prior, InferenceMethod::VariationalBayes)?
        .with_config(config);
    model.fit()
}
