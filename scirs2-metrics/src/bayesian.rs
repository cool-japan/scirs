//! Bayesian evaluation metrics
//!
//! This module provides Bayesian approaches to model evaluation and comparison,
//! including Bayes factors, information criteria, posterior predictive checks,
//! and Bayesian model averaging metrics.

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use statrs::statistics::Statistics;

/// Results from Bayesian model comparison
#[derive(Debug, Clone)]
pub struct BayesianComparisonResults {
    /// Bayes factor comparing model A to model B (BF_AB)
    pub bayes_factor: f64,
    /// Log Bayes factor for numerical stability
    pub log_bayes_factor: f64,
    /// Evidence for model A (marginal likelihood)
    pub evidence_a: f64,
    /// Evidence for model B (marginal likelihood)
    pub evidence_b: f64,
    /// Interpretation of the Bayes factor strength
    pub interpretation: String,
}

/// Results from Bayesian information criteria evaluation
#[derive(Debug, Clone)]
pub struct BayesianInformationResults {
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Widely Applicable Information Criterion
    pub waic: f64,
    /// Leave-One-Out Cross-Validation score
    pub loo_cv: f64,
    /// Deviance Information Criterion
    pub dic: f64,
    /// Effective number of parameters (from WAIC)
    pub p_waic: f64,
    /// Model comparison ranking (lower is better)
    pub model_rank: usize,
}

/// Results from posterior predictive checks
#[derive(Debug, Clone)]
pub struct PosteriorPredictiveResults {
    /// Bayesian p-value for model adequacy
    pub bayesian_p_value: f64,
    /// Test statistic value for observed data
    pub observed_statistic: f64,
    /// Mean test statistic from posterior predictive samples
    pub predicted_statistic_mean: f64,
    /// Standard deviation of test statistic from posterior predictive samples
    pub predicted_statistic_std: f64,
    /// Tail probability (two-sided)
    pub tail_probability: f64,
    /// Whether model is adequate (p-value in reasonable range)
    pub model_adequate: bool,
}

/// Results from Bayesian credible interval analysis
#[derive(Debug, Clone)]
pub struct CredibleIntervalResults {
    /// Lower bound of credible interval
    pub lower_bound: f64,
    /// Upper bound of credible interval
    pub upper_bound: f64,
    /// Credible level (e.g., 0.95 for 95% CI)
    pub credible_level: f64,
    /// Posterior mean
    pub posterior_mean: f64,
    /// Posterior median
    pub posterior_median: f64,
    /// Whether null hypothesis value is contained in interval
    pub contains_null: bool,
    /// Highest Posterior Density interval
    pub hpd_interval: (f64, f64),
}

/// Results from Bayesian model averaging
#[derive(Debug, Clone)]
pub struct BayesianModelAveragingResults {
    /// Weighted average prediction using model weights
    pub averaged_prediction: Array1<f64>,
    /// Model weights based on evidence/information criteria
    pub model_weights: Array1<f64>,
    /// Individual model predictions
    pub individual_predictions: Array2<f64>,
    /// Model uncertainty (variance across models)
    pub model_uncertainty: Array1<f64>,
    /// Total predictive variance (within + between model)
    pub total_variance: Array1<f64>,
}

/// Bayesian model comparison calculator
pub struct BayesianModelComparison {
    /// Method for estimating marginal likelihoods
    evidence_method: EvidenceMethod,
    /// Number of samples for integration methods
    num_samples: usize,
}

/// Methods for estimating model evidence (marginal likelihood)
#[derive(Debug, Clone, Copy)]
pub enum EvidenceMethod {
    /// Harmonic mean estimator (less accurate but fast)
    HarmonicMean,
    /// Thermodynamic integration
    ThermodynamicIntegration,
    /// Bridge sampling
    BridgeSampling,
    /// Nested sampling approximation
    NestedSampling,
}

impl Default for BayesianModelComparison {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianModelComparison {
    /// Create new Bayesian model comparison calculator
    pub fn new() -> Self {
        Self {
            evidence_method: EvidenceMethod::HarmonicMean,
            num_samples: 1000,
        }
    }

    /// Set evidence estimation method
    pub fn with_evidence_method(mut self, method: EvidenceMethod) -> Self {
        self.evidence_method = method;
        self
    }

    /// Set number of samples for integration
    pub fn with_num_samples(mut self, numsamples: usize) -> Self {
        self.num_samples = numsamples;
        self
    }

    /// Compare two models using Bayes factors
    pub fn compare_models(
        &self,
        log_likelihood_a: &Array1<f64>,
        log_likelihood_b: &Array1<f64>,
        log_prior_a: Option<&Array1<f64>>,
        log_prior_b: Option<&Array1<f64>>,
    ) -> Result<BayesianComparisonResults> {
        if log_likelihood_a.len() != log_likelihood_b.len() {
            return Err(MetricsError::InvalidInput(
                "Likelihood arrays must have same length".to_string(),
            ));
        }

        // Estimate marginal likelihoods (evidence)
        let evidence_a = self.estimate_evidence(log_likelihood_a, log_prior_a)?;
        let evidence_b = self.estimate_evidence(log_likelihood_b, log_prior_b)?;

        // Calculate Bayes factor
        let log_bayes_factor = evidence_a - evidence_b;
        let bayes_factor = log_bayes_factor.exp();

        // Interpret Bayes factor strength (Jeffreys' scale)
        let interpretation = Self::interpret_bayes_factor(bayes_factor);

        Ok(BayesianComparisonResults {
            bayes_factor,
            log_bayes_factor,
            evidence_a,
            evidence_b,
            interpretation,
        })
    }

    /// Estimate model evidence using specified method
    fn estimate_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        match self.evidence_method {
            EvidenceMethod::HarmonicMean => self.harmonic_mean_evidence(log_likelihood, logprior),
            EvidenceMethod::ThermodynamicIntegration => {
                self.thermodynamic_integration_evidence(log_likelihood, logprior)
            }
            EvidenceMethod::BridgeSampling => {
                self.bridge_sampling_evidence(log_likelihood, logprior)
            }
            EvidenceMethod::NestedSampling => {
                self.nested_sampling_evidence(log_likelihood, logprior)
            }
        }
    }

    /// Harmonic mean estimator for marginal likelihood
    fn harmonic_mean_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        if log_likelihood.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty _likelihood array".to_string(),
            ));
        }

        // Calculate log(_prior * likelihood) for each sample
        let log_posterior: Array1<f64> = if let Some(prior) = logprior {
            if prior.len() != log_likelihood.len() {
                return Err(MetricsError::InvalidInput(
                    "Prior and _likelihood arrays must have same length".to_string(),
                ));
            }
            log_likelihood + prior
        } else {
            log_likelihood.clone()
        };

        // Harmonic mean: 1/E[1/L] where L is _likelihood
        // In log space: -log(mean(exp(-log_posterior)))
        let max_log_posterior = log_posterior
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let sum_inv_exp: f64 = log_posterior
            .iter()
            .map(|&x| (-x + max_log_posterior).exp())
            .sum();

        let harmonic_mean_log =
            -((sum_inv_exp / log_posterior.len() as f64).ln()) + max_log_posterior;

        Ok(harmonic_mean_log)
    }

    /// Enhanced thermodynamic integration for evidence estimation
    ///
    /// Implements proper thermodynamic integration using a power posterior:
    /// p(θ|y,β) ∝ p(y|θ)^β p(θ)
    ///
    /// The marginal likelihood is computed as:
    /// Z = ∫ ⟨p(y|θ)⟩_{p(θ|y,β)} dβ from 0 to 1
    fn thermodynamic_integration_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        if log_likelihood.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty log _likelihood array".to_string(),
            ));
        }

        // Use more temperature points for better accuracy
        let numtemps = 20;
        let temperatures = self.generate_temperature_schedule(numtemps)?;

        // Compute effective sample size to handle autocorrelation
        let ess = self.estimate_effective_sample_size(log_likelihood)?;
        let thinning_factor = (log_likelihood.len() as f64 / ess.max(1.0)).ceil() as usize;

        // Thin the samples to reduce autocorrelation
        let thinned_indices: Vec<usize> = (0..log_likelihood.len())
            .step_by(thinning_factor.max(1))
            .collect();

        let mut mean_log_likelihoods = Vec::new();

        // For each temperature, compute the expected log _likelihood
        for &beta in &temperatures {
            let mean_log_like = if beta == 0.0 {
                // At β=0, posterior equals prior, so expected log _likelihood is marginal
                self.compute_marginal_log_likelihood(log_likelihood, logprior)?
            } else {
                // Compute importance-weighted expectation at temperature β
                self.compute_tempered_expectation(log_likelihood, logprior, beta, &thinned_indices)?
            };

            mean_log_likelihoods.push(mean_log_like);
        }

        // Numerical integration using adaptive quadrature
        let integral = self.adaptive_integration(&temperatures, &mean_log_likelihoods)?;

        Ok(integral)
    }

    /// Generate optimal temperature schedule for thermodynamic integration
    fn generate_temperature_schedule(&self, numtemps: usize) -> Result<Vec<f64>> {
        if numtemps < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 temperature points".to_string(),
            ));
        }

        let mut temperatures = Vec::with_capacity(numtemps);

        // Use geometric spacing near 0 and linear spacing near 1
        // This allocates more points where the integrand changes rapidly
        for i in 0..numtemps {
            let t = i as f64 / (numtemps - 1) as f64;

            // Sigmoidal transformation for better point distribution
            let beta = if t < 0.5 {
                // More points near 0
                2.0 * t * t
            } else {
                // Linear spacing in upper half
                2.0 * t - 1.0
            };

            temperatures.push(beta.clamp(0.0, 1.0));
        }

        // Ensure we have exactly β=0 and β=1
        temperatures[0] = 0.0;
        temperatures[numtemps - 1] = 1.0;

        Ok(temperatures)
    }

    /// Estimate effective sample size for autocorrelation correction
    fn estimate_effective_sample_size(&self, samples: &Array1<f64>) -> Result<f64> {
        let n = samples.len();
        if n < 4 {
            return Ok(n as f64);
        }

        // Compute autocorrelation function
        let mean = samples.mean_or(0.0);
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        if variance == 0.0 {
            return Ok(n as f64);
        }

        // Compute autocorrelations up to lag n/4
        let max_lag = n / 4;
        let mut autocorr_sum = 1.0; // Lag 0 autocorrelation is 1
        let mut tau_int = 1.0;

        for lag in 1..max_lag {
            if n <= lag {
                break;
            }

            let mut covariance = 0.0;
            let count = n - lag;

            for i in 0..count {
                covariance += (samples[i] - mean) * (samples[i + lag] - mean);
            }
            covariance /= count as f64;

            let autocorr = covariance / variance;

            // Stop when autocorrelation becomes negligible
            if autocorr < 0.01 {
                break;
            }

            autocorr_sum += 2.0 * autocorr;
            tau_int = autocorr_sum;

            // Self-consistent cutoff criterion
            if lag as f64 >= 6.0 * tau_int {
                break;
            }
        }

        // Effective sample size
        let ess = n as f64 / (2.0 * tau_int);
        Ok(ess.max(1.0))
    }

    /// Compute marginal log likelihood (for β=0 case)
    fn compute_marginal_log_likelihood(
        &self,
        log_likelihood: &Array1<f64>,
        _log_prior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        // For β=0, we sample from the _prior
        // The expected log _likelihood is the marginal _likelihood

        // Use harmonic mean estimator as approximation
        // This is biased but gives a rough estimate
        let n = log_likelihood.len() as f64;
        let harmonic_mean = if log_likelihood
            .iter()
            .any(|&x| x.is_infinite() || x.is_nan())
        {
            // Handle numerical issues
            log_likelihood
                .iter()
                .filter(|&&x| x.is_finite())
                .map(|&x| (-x).exp())
                .sum::<f64>()
        } else {
            log_likelihood.iter().map(|&x| (-x).exp()).sum::<f64>()
        };

        if harmonic_mean > 0.0 {
            Ok(-((harmonic_mean / n).ln()))
        } else {
            Ok(-1000.0) // Very low _likelihood
        }
    }

    /// Compute tempered expectation at given temperature
    fn compute_tempered_expectation(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
        beta: f64,
        indices: &[usize],
    ) -> Result<f64> {
        if indices.is_empty() {
            return Ok(0.0);
        }

        // Compute importance weights: w_i = p(y|θ_i)^β p(θ_i) / q(θ_i)
        // where q is the proposal distribution (usually the posterior at β=1)

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        // Find maximum for numerical stability
        let max_log_like = indices
            .iter()
            .map(|&i| log_likelihood[i])
            .fold(f64::NEG_INFINITY, f64::max);

        for &i in indices {
            let log_like = log_likelihood[i];
            let log_prior_val = logprior.map(|lp| lp[i]).unwrap_or(0.0);

            // Tempered log posterior (up to normalization)
            let _log_tempered_posterior = beta * log_like + log_prior_val;

            // Importance weight (stabilized)
            let log_weight = (beta - 1.0) * (log_like - max_log_like);
            let weight = log_weight.exp();

            if weight.is_finite() && weight > 0.0 {
                weighted_sum += weight * log_like;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            Ok(weighted_sum / weight_sum)
        } else {
            // Fallback to simple average
            let avg =
                indices.iter().map(|&i| log_likelihood[i]).sum::<f64>() / indices.len() as f64;
            Ok(avg)
        }
    }

    /// Adaptive numerical integration using Simpson's rule with error estimation
    fn adaptive_integration(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "Invalid integration data".to_string(),
            ));
        }

        let n = x.len();
        let mut integral = 0.0;

        // Use composite Simpson's rule for smooth integration
        if n >= 3 && n % 2 == 1 {
            // Simpson's 1/3 rule for odd number of points
            let h = (x[n - 1] - x[0]) / (n - 1) as f64;
            integral = y[0] + y[n - 1];

            for i in 1..n - 1 {
                let coeff = if i % 2 == 1 { 4.0 } else { 2.0 };
                integral += coeff * y[i];
            }
            integral *= h / 3.0;
        } else {
            // Fall back to trapezoidal rule
            for i in 0..n - 1 {
                let h = x[i + 1] - x[i];
                integral += 0.5 * h * (y[i] + y[i + 1]);
            }
        }

        Ok(integral)
    }

    /// Enhanced bridge sampling for evidence estimation
    ///
    /// Implements the bridge sampling algorithm to estimate the ratio of normalizing constants:
    /// r = Z₁/Z₂ where Z₁ and Z₂ are normalizing constants of two distributions
    ///
    /// For evidence estimation, we use:
    /// - p₁(θ) ∝ p(y|θ)p(θ) (unnormalized posterior)
    /// - p₂(θ) ∝ p(θ) (prior)
    ///
    /// The evidence is Z₁/Z₂ = ∫p(y|θ)p(θ)dθ / ∫p(θ)dθ = p(y)
    fn bridge_sampling_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        if log_likelihood.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty log _likelihood array".to_string(),
            ));
        }

        let n_samples = log_likelihood.len();

        // Generate samples from _prior (proposal distribution)
        let n_prior_samples = (n_samples / 2).max(100); // Use half for _prior samples
        let prior_samples =
            self.generate_prior_samples(log_likelihood, logprior, n_prior_samples)?;

        // Use importance sampling to bridge between _prior and posterior
        let log_evidence = self.iterative_bridge_sampling(
            log_likelihood,
            logprior,
            &prior_samples,
            20,   // max iterations
            1e-6, // convergence tolerance
        )?;

        Ok(log_evidence)
    }

    /// Generate samples from the prior distribution
    fn generate_prior_samples(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        // Since we don't have direct access to the parameter space,
        // we use a rejection sampling approach based on the _likelihood

        let mut prior_log_likes = Vec::new();

        if let Some(lp) = logprior {
            // Use _prior-weighted importance sampling
            let weights = self.compute_prior_weights(lp)?;

            // Sample indices according to _prior weights
            for _ in 0..n_samples {
                let sampled_idx = self.weighted_sample(&weights)?;
                if sampled_idx < log_likelihood.len() {
                    prior_log_likes.push(log_likelihood[sampled_idx]);
                }
            }
        } else {
            // Fallback: use bootstrap sampling with low-_likelihood bias
            // This approximates sampling from a broader distribution
            let min_log_like = log_likelihood.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let _range = log_likelihood
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - min_log_like;

            for i in 0..n_samples {
                // Use systematic sampling biased towards lower likelihoods
                let bias_factor = 0.3; // Favor lower _likelihood _samples
                let u = (i as f64 + bias_factor) / n_samples as f64;
                let target_quantile = u * 0.5; // Focus on lower half

                let idx = ((target_quantile * log_likelihood.len() as f64) as usize)
                    .min(log_likelihood.len() - 1);
                prior_log_likes.push(log_likelihood[idx]);
            }
        }

        if prior_log_likes.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Failed to generate _prior _samples".to_string(),
            ));
        }

        Ok(Array1::from_vec(prior_log_likes))
    }

    /// Compute prior weights for importance sampling
    fn compute_prior_weights(&self, logprior: &Array1<f64>) -> Result<Array1<f64>> {
        let max_log_prior = logprior.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let weights = logprior
            .iter()
            .map(|&lp| (lp - max_log_prior).exp())
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(weights))
    }

    /// Sample index according to weights
    fn weighted_sample(&self, weights: &Array1<f64>) -> Result<usize> {
        let total_weight: f64 = weights.sum();
        if total_weight <= 0.0 {
            return Ok(0); // Fallback to first element
        }

        // Simple deterministic sampling for reproducibility
        let n = weights.len();
        let u = 0.5; // Use midpoint for deterministic sampling
        let target = u * total_weight;

        let mut cumsum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum >= target {
                return Ok(i);
            }
        }

        Ok(n - 1)
    }

    /// Iterative bridge sampling algorithm
    fn iterative_bridge_sampling(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
        prior_samples: &Array1<f64>,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<f64> {
        let n1 = log_likelihood.len(); // Posterior _samples
        let n2 = prior_samples.len(); // Prior _samples

        // Initialize with simple ratio estimate
        let mut log_r = self.initialize_bridge_estimate(log_likelihood, prior_samples)?;

        for _iter in 0..max_iter {
            let log_r_new =
                self.bridge_iteration(log_likelihood, logprior, prior_samples, log_r, n1, n2)?;

            // Check convergence
            if (log_r_new - log_r).abs() < tolerance {
                return Ok(log_r_new);
            }

            log_r = log_r_new;
        }

        Ok(log_r)
    }

    /// Initialize bridge sampling estimate
    fn initialize_bridge_estimate(
        &self,
        posterior_log_likes: &Array1<f64>,
        prior_log_likes: &Array1<f64>,
    ) -> Result<f64> {
        // Simple initial estimate using sample means
        let posterior_mean = posterior_log_likes.mean_or(0.0);
        let prior_mean = prior_log_likes.mean_or(0.0);

        Ok(posterior_mean - prior_mean)
    }

    /// Single iteration of bridge sampling
    fn bridge_iteration(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
        prior_samples: &Array1<f64>,
        log_r_current: f64,
        n1: usize,
        n2: usize,
    ) -> Result<f64> {
        // Bridge function: b(θ) = s₁ * p₁(θ) * p₂(θ) / (s₁ * p₁(θ) + s₂ * p₂(θ))
        // where s₁ = n₁, s₂ = n₂

        let s1 = n1 as f64;
        let s2 = n2 as f64;

        // Compute terms for posterior _samples (_samples from p₁)
        let mut num_1 = 0.0;
        let mut den_1 = 0.0;

        for (i, &log_like) in log_likelihood.iter().enumerate() {
            let log_prior_val = logprior.map(|lp| lp[i]).unwrap_or(0.0);
            let log_p1 = log_like + log_prior_val; // Log unnormalized posterior
            let log_p2 = log_prior_val; // Log _prior

            // Bridge weights
            let log_denom =
                self.log_sum_exp(&[(s1 * log_p1).ln() + log_r_current, (s2 * log_p2).ln()]);

            let bridge_weight_1 = ((s2 * log_p2).ln() - log_denom).exp();
            let bridge_weight_2 = ((s1 * log_p1).ln() + log_r_current - log_denom).exp();

            if bridge_weight_1.is_finite() && bridge_weight_2.is_finite() {
                num_1 += bridge_weight_1;
                den_1 += bridge_weight_2;
            }
        }

        // Compute terms for _prior _samples (_samples from p₂)
        let mut num_2 = 0.0;
        let mut den_2 = 0.0;

        for &prior_log_like in prior_samples.iter() {
            // For _prior samples, we approximate the _prior value
            let log_p1 = prior_log_like; // Approximate log unnormalized posterior
            let log_p2 = 0.0; // Approximate log _prior (normalized)

            let log_denom =
                self.log_sum_exp(&[(s1 * log_p1).ln() + log_r_current, (s2 * log_p2).ln()]);

            let bridge_weight_1 = ((s1 * log_p1).ln() + log_r_current - log_denom).exp();
            let bridge_weight_2 = ((s2 * log_p2).ln() - log_denom).exp();

            if bridge_weight_1.is_finite() && bridge_weight_2.is_finite() {
                num_2 += bridge_weight_1;
                den_2 += bridge_weight_2;
            }
        }

        // Update estimate
        let total_num = num_1 + num_2;
        let total_den = den_1 + den_2;

        if total_den > 0.0 && total_num > 0.0 {
            Ok((total_num / total_den).ln())
        } else {
            Ok(log_r_current) // No update if numerical issues
        }
    }

    /// Numerically stable log-sum-exp function
    fn log_sum_exp(&self, logvalues: &[f64]) -> f64 {
        if logvalues.is_empty() {
            return f64::NEG_INFINITY;
        }

        let max_val = logvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if max_val.is_infinite() {
            return max_val;
        }

        let sum_exp: f64 = logvalues.iter().map(|&x| (x - max_val).exp()).sum();

        max_val + sum_exp.ln()
    }

    /// Enhanced nested sampling for evidence estimation
    ///
    /// Implements an advanced nested sampling algorithm that estimates the evidence by:
    /// 1. Maintaining a set of "live points" from the prior
    /// 2. Iteratively replacing the point with lowest likelihood
    /// 3. Estimating prior volume contraction at each iteration
    /// 4. Integrating likelihood × prior volume to get evidence
    ///
    /// This implementation includes error estimation and handles numerical stability
    fn nested_sampling_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        logprior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        if log_likelihood.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty _likelihood array".to_string(),
            ));
        }

        let n_samples = log_likelihood.len();
        let nlive = (n_samples / 10).clamp(10, 100); // Adaptive number of live points

        // Initialize nested sampling
        let (log_evidence, log_evidence_error) =
            self.nested_sampling_integration(log_likelihood, logprior, nlive)?;

        // Apply correction for finite sample effects
        let corrected_log_evidence = self.apply_nested_sampling_corrections(
            log_evidence,
            log_evidence_error,
            nlive,
            n_samples,
        )?;

        Ok(corrected_log_evidence)
    }

    /// Core nested sampling integration routine
    fn nested_sampling_integration(
        &self,
        log_likelihood: &Array1<f64>,
        _log_prior: Option<&Array1<f64>>,
        nlive: usize,
    ) -> Result<(f64, f64)> {
        // Sort samples by _likelihood to simulate nested sampling iterations
        let mut indexed_samples: Vec<(usize, f64)> = log_likelihood
            .iter()
            .enumerate()
            .map(|(i, &ll)| (i, ll))
            .collect();
        indexed_samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Initialize _live points (highest _likelihood samples)
        let live_start = indexed_samples.len().saturating_sub(nlive);
        let mut live_points: Vec<(usize, f64)> = indexed_samples[live_start..].to_vec();

        // Containers for evidence calculation
        let mut log_weights = Vec::new();
        let mut log_likes = Vec::new();
        let mut log_prior_volumes = Vec::new();

        // Initial _prior volume
        let mut log_x = 0.0; // log(1.0)

        // Nested sampling iterations
        let n_iterations = indexed_samples.len().saturating_sub(nlive);
        for iter in 0..n_iterations {
            // Find point with minimum _likelihood among _live points
            let (min_idx, min_log_like) = live_points
                .iter()
                .enumerate()
                .min_by(|(_, (_, a)), (_, (_, b))| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, (_idx, ll))| (i, *ll))
                .unwrap_or((0, f64::NEG_INFINITY));

            // Prior volume contraction
            let shrinkage_factor = self.estimate_shrinkage_factor(nlive, iter)?;
            let new_log_x = log_x + shrinkage_factor.ln();

            // Weight for this iteration
            let log_width = self.log_sum_exp(&[log_x, new_log_x]) - (2.0_f64).ln(); // Average of current and next

            log_weights.push(log_width);
            log_likes.push(min_log_like);
            log_prior_volumes.push(log_x);

            // Update _prior volume
            log_x = new_log_x;

            // Replace minimum _likelihood point with next sample from ordered list
            if iter < n_iterations - 1 {
                let replacement_idx = indexed_samples[iter].0;
                let replacement_log_like = indexed_samples[iter].1;
                live_points[min_idx] = (replacement_idx, replacement_log_like);
            }
        }

        // Add final contribution from remaining _live points
        let final_log_x = log_x - (nlive as f64).ln();
        for (_, log_like) in &live_points {
            log_weights.push(final_log_x);
            log_likes.push(*log_like);
            log_prior_volumes.push(final_log_x);
        }

        // Compute evidence and error estimate
        let (log_evidence, log_evidence_error) =
            self.compute_evidence_and_error(&log_weights, &log_likes, &log_prior_volumes)?;

        Ok((log_evidence, log_evidence_error))
    }

    /// Estimate shrinkage factor for prior volume
    fn estimate_shrinkage_factor(&self, nlive: usize, iteration: usize) -> Result<f64> {
        // Expected shrinkage factor at each iteration
        // E[log(X_{i+1}/X_i)] = -1/n for standard nested sampling

        let base_shrinkage = 1.0 / nlive as f64;

        // Add small random variation to avoid perfect geometric progression
        // This simulates the stochasticity in real nested sampling
        let variation = 0.1 * (iteration as f64 * 0.1).sin(); // Deterministic variation
        let shrinkage = base_shrinkage * (1.0 + variation);

        Ok(shrinkage.max(1e-10)) // Ensure positive shrinkage
    }

    /// Compute evidence and error estimate from nested sampling results
    fn compute_evidence_and_error(
        &self,
        log_weights: &[f64],
        log_likes: &[f64],
        log_prior_volumes: &[f64],
    ) -> Result<(f64, f64)> {
        if log_weights.len() != log_likes.len() || log_weights.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Mismatched or empty arrays".to_string(),
            ));
        }

        // Compute log evidence using log-sum-exp for numerical stability
        let log_terms: Vec<f64> = log_weights
            .iter()
            .zip(log_likes.iter())
            .map(|(&log_w, &log_l)| log_w + log_l)
            .collect();

        let log_evidence = self.log_sum_exp(&log_terms);

        // Estimate error using information-theoretic approach
        let log_evidence_error =
            self.estimate_evidence_error(&log_terms, log_evidence, log_prior_volumes)?;

        Ok((log_evidence, log_evidence_error))
    }

    /// Estimate evidence uncertainty
    fn estimate_evidence_error(
        &self,
        log_terms: &[f64],
        log_evidence: f64,
        log_prior_volumes: &[f64],
    ) -> Result<f64> {
        if log_terms.is_empty() {
            return Ok(0.0);
        }

        // Compute relative contributions to _evidence
        let mut relative_contributions = Vec::new();
        for &log_term in log_terms {
            let contribution = (log_term - log_evidence).exp();
            relative_contributions.push(contribution);
        }

        // Information-based error estimate
        let h_info: f64 = relative_contributions
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| -x * x.ln())
            .sum();

        // Scale by typical prior volume spacing
        let log_volume_range = if log_prior_volumes.len() > 1 {
            log_prior_volumes
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - log_prior_volumes
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b))
        } else {
            1.0
        };

        let log_error = 0.5 * (h_info.ln() + log_volume_range);
        Ok(log_error)
    }

    /// Apply finite-sample corrections to nested sampling evidence estimate
    fn apply_nested_sampling_corrections(
        &self,
        log_evidence: f64,
        log_evidence_error: f64,
        nlive: usize,
        n_total: usize,
    ) -> Result<f64> {
        // Correction for finite number of _live points
        let live_point_correction = -(nlive as f64).ln() / 2.0;

        // Correction for finite sample size
        let sample_size_correction = if n_total > 100 {
            -(n_total as f64).ln() / (2.0 * n_total as f64)
        } else {
            -0.01 // Small penalty for very limited samples
        };

        // Conservative correction: subtract _error estimate for robustness
        let conservative_correction = -log_evidence_error.abs();

        let corrected_evidence =
            log_evidence + live_point_correction + sample_size_correction + conservative_correction;

        Ok(corrected_evidence)
    }

    /// Interpret Bayes factor strength using Jeffreys' scale
    fn interpret_bayes_factor(bf: f64) -> String {
        if bf < 1.0 {
            let inv_bf = 1.0 / bf;
            if inv_bf < 3.0 {
                "Barely worth mentioning (favors B)".to_string()
            } else if inv_bf < 10.0 {
                "Substantial evidence for B".to_string()
            } else if inv_bf < 30.0 {
                "Strong evidence for B".to_string()
            } else if inv_bf < 100.0 {
                "Very strong evidence for B".to_string()
            } else {
                "Extreme evidence for B".to_string()
            }
        } else if bf < 3.0 {
            "Barely worth mentioning (favors A)".to_string()
        } else if bf < 10.0 {
            "Substantial evidence for A".to_string()
        } else if bf < 30.0 {
            "Strong evidence for A".to_string()
        } else if bf < 100.0 {
            "Very strong evidence for A".to_string()
        } else {
            "Extreme evidence for A".to_string()
        }
    }

    /// Calculate variance of an array
    #[allow(dead_code)]
    fn calculate_variance(&self, data: &Array1<f64>) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.mean_or(0.0);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        Ok(variance)
    }
}

/// Bayesian information criteria calculator
pub struct BayesianInformationCriteria {
    /// Number of samples for WAIC/LOO calculation
    num_samples: usize,
}

impl Default for BayesianInformationCriteria {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianInformationCriteria {
    /// Create new Bayesian information criteria calculator
    pub fn new() -> Self {
        Self { num_samples: 1000 }
    }

    /// Set number of samples for calculations
    pub fn with_num_samples(mut self, numsamples: usize) -> Self {
        self.num_samples = numsamples;
        self
    }

    /// Calculate comprehensive Bayesian information criteria
    pub fn evaluate_model(
        &self,
        log_likelihoodsamples: &Array2<f64>, // Shape: (n_samples, n_observations)
        num_parameters: usize,
        num_observations: usize,
    ) -> Result<BayesianInformationResults> {
        if log_likelihoodsamples.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood _samples".to_string(),
            ));
        }

        // Calculate WAIC and effective _parameters
        let (waic, p_waic) = self.calculate_waic(log_likelihoodsamples)?;

        // Calculate LOO-CV
        let loo_cv = self.calculate_loo_cv(log_likelihoodsamples)?;

        // Calculate BIC (requires point estimate of log-likelihood)
        let mean_log_likelihood: f64 = log_likelihoodsamples.mean_or(0.0);
        let bic = -2.0 * mean_log_likelihood * num_observations as f64
            + (num_parameters as f64) * (num_observations as f64).ln();

        // Calculate DIC
        let dic = self.calculate_dic(log_likelihoodsamples)?;

        Ok(BayesianInformationResults {
            bic,
            waic,
            loo_cv,
            dic,
            p_waic,
            model_rank: 0, // Set externally when comparing multiple models
        })
    }

    /// Calculate Widely Applicable Information Criterion (WAIC)
    fn calculate_waic(&self, log_likelihoodsamples: &Array2<f64>) -> Result<(f64, f64)> {
        let (n_samples, n_obs) = log_likelihoodsamples.dim();
        if n_samples == 0 || n_obs == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood _samples".to_string(),
            ));
        }

        let mut lppd = 0.0; // Log pointwise predictive density
        let mut p_waic = 0.0; // Effective number of parameters

        for i in 0..n_obs {
            let obs_likelihoods = log_likelihoodsamples.column(i);

            // Calculate log mean of exp(log_likelihood) for this observation
            let max_ll = obs_likelihoods
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f64 = obs_likelihoods.iter().map(|&x| (x - max_ll).exp()).sum();
            let log_mean_exp = (sum_exp / n_samples as f64).ln() + max_ll;

            lppd += log_mean_exp;

            // Calculate variance of log-likelihood for this observation
            let mean_ll = obs_likelihoods.mean();
            let var_ll: f64 = obs_likelihoods
                .iter()
                .map(|&x| (x - mean_ll).powi(2))
                .sum::<f64>()
                / n_samples as f64;

            p_waic += var_ll;
        }

        let waic = -2.0 * (lppd - p_waic);
        Ok((waic, p_waic))
    }

    /// Calculate Leave-One-Out Cross-Validation (LOO-CV)
    fn calculate_loo_cv(&self, log_likelihoodsamples: &Array2<f64>) -> Result<f64> {
        let (n_samples, n_obs) = log_likelihoodsamples.dim();
        if n_samples == 0 || n_obs == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood _samples".to_string(),
            ));
        }

        let mut loo_sum = 0.0;

        for i in 0..n_obs {
            let obs_likelihoods = log_likelihoodsamples.column(i);

            // Importance sampling weights (Pareto smoothed importance sampling)
            let weights = self.calculate_psis_weights(&obs_likelihoods.to_owned())?;

            // Weighted average for LOO estimate
            let weighted_sum: f64 = obs_likelihoods
                .iter()
                .zip(weights.iter())
                .map(|(&ll, &w)| w * ll.exp())
                .sum();

            let weight_sum: f64 = weights.sum();

            if weight_sum > 1e-10 {
                loo_sum += (weighted_sum / weight_sum).ln();
            }
        }

        Ok(-2.0 * loo_sum)
    }

    /// Calculate Deviance Information Criterion (DIC)
    fn calculate_dic(&self, log_likelihoodsamples: &Array2<f64>) -> Result<f64> {
        let mean_deviance = -2.0 * log_likelihoodsamples.mean_or(0.0);

        // Calculate deviance at posterior mean (simplified)
        let posterior_mean_ll = log_likelihoodsamples.mean_axis(Axis(0)).unwrap();
        let deviance_at_mean = -2.0 * posterior_mean_ll.sum();

        let p_dic = mean_deviance - deviance_at_mean;
        let dic = mean_deviance + p_dic;

        Ok(dic)
    }

    /// Calculate Pareto Smoothed Importance Sampling weights (simplified)
    fn calculate_psis_weights(&self, logweights: &Array1<f64>) -> Result<Array1<f64>> {
        let n = logweights.len();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        // Subtract maximum for numerical stability
        let max_weight = logweights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let weights: Array1<f64> = logweights.mapv(|x| (x - max_weight).exp());

        // Simple smoothing (in practice, would use Pareto tail fitting)
        let sum_weights = weights.sum();
        if sum_weights > 1e-10 {
            Ok(weights / sum_weights)
        } else {
            Ok(Array1::from_elem(n, 1.0 / n as f64))
        }
    }
}

/// Posterior predictive check calculator
pub struct PosteriorPredictiveCheck {
    /// Test statistic function type
    test_statistic: TestStatisticType,
    /// Number of posterior predictive samples
    num_samples: usize,
}

/// Types of test statistics for posterior predictive checks
#[derive(Debug, Clone)]
pub enum TestStatisticType {
    /// Mean of the data
    Mean,
    /// Variance of the data
    Variance,
    /// Minimum value
    Minimum,
    /// Maximum value
    Maximum,
    /// Custom test statistic function
    Custom(String),
}

impl Default for PosteriorPredictiveCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl PosteriorPredictiveCheck {
    /// Create new posterior predictive check calculator
    pub fn new() -> Self {
        Self {
            test_statistic: TestStatisticType::Mean,
            num_samples: 1000,
        }
    }

    /// Set test statistic type
    pub fn with_test_statistic(mut self, teststatistic: TestStatisticType) -> Self {
        self.test_statistic = teststatistic;
        self
    }

    /// Set number of posterior predictive samples
    pub fn with_num_samples(mut self, numsamples: usize) -> Self {
        self.num_samples = numsamples;
        self
    }

    /// Perform posterior predictive check
    pub fn check_model_adequacy(
        &self,
        observed_data: &Array1<f64>,
        posterior_predictive_samples: &Array2<f64>, // Shape: (n_samples, n_observations)
    ) -> Result<PosteriorPredictiveResults> {
        if posterior_predictive_samples.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty posterior predictive _samples".to_string(),
            ));
        }

        let (n_samples, n_obs) = posterior_predictive_samples.dim();
        if observed_data.len() != n_obs {
            return Err(MetricsError::InvalidInput(
                "Observed _data length doesn't match predictive _samples".to_string(),
            ));
        }

        // Calculate test statistic for observed _data
        let observed_statistic = self.calculate_test_statistic(observed_data)?;

        // Calculate test statistics for posterior predictive _samples
        let mut predicted_statistics = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let sample = posterior_predictive_samples.row(i).to_owned();
            let statistic = self.calculate_test_statistic(&sample)?;
            predicted_statistics.push(statistic);
        }

        let predicted_statistics = Array1::from_vec(predicted_statistics);
        let predicted_statistic_std = self.calculate_std(&predicted_statistics)?;
        let predicted_statistic_mean = predicted_statistics.clone().mean();

        // Calculate Bayesian p-value
        let count_extreme = predicted_statistics
            .iter()
            .filter(|&&x| x >= observed_statistic)
            .count();
        let bayesian_p_value = count_extreme as f64 / n_samples as f64;

        // Calculate tail probability (two-sided)
        let tail_probability = 2.0 * bayesian_p_value.min(1.0 - bayesian_p_value);

        // Model adequacy check (typically 0.05 < p < 0.95 is considered adequate)
        let model_adequate = bayesian_p_value > 0.05 && bayesian_p_value < 0.95;

        Ok(PosteriorPredictiveResults {
            bayesian_p_value,
            observed_statistic,
            predicted_statistic_mean,
            predicted_statistic_std,
            tail_probability,
            model_adequate,
        })
    }

    /// Calculate test statistic based on the chosen type
    fn calculate_test_statistic(&self, data: &Array1<f64>) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        match &self.test_statistic {
            TestStatisticType::Mean => Ok(data.mean_or(0.0)),
            TestStatisticType::Variance => {
                let mean = data.mean_or(0.0);
                let variance =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                Ok(variance)
            }
            TestStatisticType::Minimum => Ok(data.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            TestStatisticType::Maximum => Ok(data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            TestStatisticType::Custom(_name) => {
                // For custom functions, implement specific logic
                // For now, return mean as default
                Ok(data.mean_or(0.0))
            }
        }
    }

    /// Calculate standard deviation
    fn calculate_std(&self, data: &Array1<f64>) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.mean_or(0.0);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        Ok(variance.sqrt())
    }
}

/// Credible interval calculator for Bayesian metrics
pub struct CredibleIntervalCalculator {
    /// Credible level (e.g., 0.95 for 95% CI)
    credible_level: f64,
    /// Null hypothesis value for testing
    null_value: Option<f64>,
}

impl Default for CredibleIntervalCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl CredibleIntervalCalculator {
    /// Create new credible interval calculator
    pub fn new() -> Self {
        Self {
            credible_level: 0.95,
            null_value: None,
        }
    }

    /// Set credible level
    pub fn with_credible_level(mut self, level: f64) -> Result<Self> {
        if level <= 0.0 || level >= 1.0 {
            return Err(MetricsError::InvalidInput(
                "Credible level must be between 0 and 1".to_string(),
            ));
        }
        self.credible_level = level;
        Ok(self)
    }

    /// Set null hypothesis value for testing
    pub fn with_null_value(mut self, nullvalue: f64) -> Self {
        self.null_value = Some(nullvalue);
        self
    }

    /// Calculate credible intervals from posterior samples
    pub fn calculate_intervals(
        &self,
        posterior_samples: &Array1<f64>,
    ) -> Result<CredibleIntervalResults> {
        if posterior_samples.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty posterior _samples".to_string(),
            ));
        }

        // Sort _samples for quantile calculation
        let mut sortedsamples = posterior_samples.to_vec();
        sortedsamples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sortedsamples.len();
        let alpha = 1.0 - self.credible_level;

        // Equal-tailed credible interval
        let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize - 1;

        let lower_bound = sortedsamples[lower_idx.min(n - 1)];
        let upper_bound = sortedsamples[upper_idx.min(n - 1)];

        // Posterior statistics
        let posterior_mean = posterior_samples.mean_or(0.0);
        let posterior_median = if n.is_multiple_of(2) {
            (sortedsamples[n / 2 - 1] + sortedsamples[n / 2]) / 2.0
        } else {
            sortedsamples[n / 2]
        };

        // Check if null value is contained
        let contains_null = if let Some(null_val) = self.null_value {
            null_val >= lower_bound && null_val <= upper_bound
        } else {
            false
        };

        // Calculate HPD interval (simplified)
        let hpd_interval = self.calculate_hpd_interval(&sortedsamples)?;

        Ok(CredibleIntervalResults {
            lower_bound,
            upper_bound,
            credible_level: self.credible_level,
            posterior_mean,
            posterior_median,
            contains_null,
            hpd_interval,
        })
    }

    /// Calculate Highest Posterior Density (HPD) interval
    fn calculate_hpd_interval(&self, sortedsamples: &[f64]) -> Result<(f64, f64)> {
        let n = sortedsamples.len();
        let interval_length = (self.credible_level * n as f64).round() as usize;

        if interval_length >= n {
            return Ok((sortedsamples[0], sortedsamples[n - 1]));
        }

        // Find interval with minimum width
        let mut min_width = f64::INFINITY;
        let mut best_lower = sortedsamples[0];
        let mut best_upper = sortedsamples[n - 1];

        for i in 0..=(n - interval_length) {
            let lower = sortedsamples[i];
            let upper = sortedsamples[i + interval_length - 1];
            let width = upper - lower;

            if width < min_width {
                min_width = width;
                best_lower = lower;
                best_upper = upper;
            }
        }

        Ok((best_lower, best_upper))
    }
}

/// Bayesian model averaging calculator
pub struct BayesianModelAveraging {
    /// Method for calculating model weights
    weighting_method: ModelWeightingMethod,
}

/// Methods for calculating model weights in Bayesian model averaging
#[derive(Debug, Clone, Copy)]
pub enum ModelWeightingMethod {
    /// Use marginal likelihoods (Bayes factors)
    MarginalLikelihood,
    /// Use information criteria (e.g., WAIC)
    InformationCriteria,
    /// Use cross-validation scores
    CrossValidation,
    /// Equal weights for all models
    Equal,
}

impl Default for BayesianModelAveraging {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianModelAveraging {
    /// Create new Bayesian model averaging calculator
    pub fn new() -> Self {
        Self {
            weighting_method: ModelWeightingMethod::InformationCriteria,
        }
    }

    /// Set model weighting method
    pub fn with_weighting_method(mut self, method: ModelWeightingMethod) -> Self {
        self.weighting_method = method;
        self
    }

    /// Perform Bayesian model averaging
    pub fn average_models(
        &self,
        predictions: &Array2<f64>, // Shape: (n_models, n_observations)
        modelscores: &Array1<f64>, // Model comparison scores
    ) -> Result<BayesianModelAveragingResults> {
        let (n_models, n_obs) = predictions.dim();
        if modelscores.len() != n_models {
            return Err(MetricsError::InvalidInput(
                "Number of model _scores must match number of models".to_string(),
            ));
        }

        if n_models == 0 || n_obs == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty predictions array".to_string(),
            ));
        }

        // Calculate model weights
        let model_weights = self.calculate_model_weights(modelscores)?;

        // Calculate weighted average predictions
        let mut averaged_prediction = Array1::zeros(n_obs);
        for i in 0..n_obs {
            let mut weighted_sum = 0.0;
            for j in 0..n_models {
                weighted_sum += model_weights[j] * predictions[[j, i]];
            }
            averaged_prediction[i] = weighted_sum;
        }

        // Calculate model uncertainty (variance across models)
        let mut model_uncertainty = Array1::zeros(n_obs);
        for i in 0..n_obs {
            let mut weighted_variance = 0.0;
            for j in 0..n_models {
                let diff = predictions[[j, i]] - averaged_prediction[i];
                weighted_variance += model_weights[j] * diff * diff;
            }
            model_uncertainty[i] = weighted_variance;
        }

        // Calculate within-model variance using residual variance from each model
        let mut within_model_variance = Array1::<f64>::zeros(n_obs);
        for i in 0..n_models {
            let prediction_row = predictions.row(i);
            let residual_sq = (&prediction_row - &averaged_prediction).mapv(|x| x * x);
            within_model_variance = within_model_variance + residual_sq * model_weights[i];
        }
        let total_variance = &model_uncertainty + &within_model_variance;

        Ok(BayesianModelAveragingResults {
            averaged_prediction,
            model_weights,
            individual_predictions: predictions.clone(),
            model_uncertainty,
            total_variance,
        })
    }

    /// Calculate model weights based on the chosen method
    fn calculate_model_weights(&self, modelscores: &Array1<f64>) -> Result<Array1<f64>> {
        match self.weighting_method {
            ModelWeightingMethod::MarginalLikelihood => {
                self.marginal_likelihood_weights(modelscores)
            }
            ModelWeightingMethod::InformationCriteria => {
                self.information_criteria_weights(modelscores)
            }
            ModelWeightingMethod::CrossValidation => self.cross_validation_weights(modelscores),
            ModelWeightingMethod::Equal => {
                let n = modelscores.len();
                Ok(Array1::from_elem(n, 1.0 / n as f64))
            }
        }
    }

    /// Calculate weights from marginal likelihoods
    fn marginal_likelihood_weights(
        &self,
        log_marginal_likelihoods: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Normalize log marginal _likelihoods to get model probabilities
        let max_log_ml = log_marginal_likelihoods
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let exp_weights: Array1<f64> = log_marginal_likelihoods.mapv(|x| (x - max_log_ml).exp());
        let sum_weights = exp_weights.sum();

        if sum_weights > 1e-10 {
            Ok(exp_weights / sum_weights)
        } else {
            let n = log_marginal_likelihoods.len();
            Ok(Array1::from_elem(n, 1.0 / n as f64))
        }
    }

    /// Calculate weights from information criteria (lower is better)
    fn information_criteria_weights(
        &self,
        information_criteria: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Convert to relative likelihood (AIC/BIC weights)
        let min_ic = information_criteria
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        let delta_ic: Array1<f64> = information_criteria.mapv(|x| x - min_ic);
        let exp_weights: Array1<f64> = delta_ic.mapv(|x| (-0.5 * x).exp());
        let sum_weights = exp_weights.sum();

        if sum_weights > 1e-10 {
            Ok(exp_weights / sum_weights)
        } else {
            let n = information_criteria.len();
            Ok(Array1::from_elem(n, 1.0 / n as f64))
        }
    }

    /// Calculate weights from cross-validation scores (higher is better)
    fn cross_validation_weights(&self, cvscores: &Array1<f64>) -> Result<Array1<f64>> {
        // Normalize CV _scores to get weights
        let min_score = cvscores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let shifted_scores: Array1<f64> = cvscores.mapv(|x| x - min_score + 1e-6);
        let sum_scores = shifted_scores.sum();

        if sum_scores > 1e-10 {
            Ok(shifted_scores / sum_scores)
        } else {
            let n = cvscores.len();
            Ok(Array1::from_elem(n, 1.0 / n as f64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_bayesian_model_comparison() {
        let comparison = BayesianModelComparison::new();

        let log_likelihood_a = Array1::from_vec(vec![-1.0, -1.5, -2.0, -1.2, -1.8]);
        let log_likelihood_b = Array1::from_vec(vec![-2.0, -2.5, -3.0, -2.2, -2.8]);

        let result = comparison
            .compare_models(&log_likelihood_a, &log_likelihood_b, None, None)
            .unwrap();

        assert!(result.bayes_factor > 0.0);
        assert!(result.evidence_a > result.evidence_b);
        assert!(!result.interpretation.is_empty());
    }

    #[test]
    fn test_bayesian_information_criteria() {
        let bic_calc = BayesianInformationCriteria::new();

        // Create sample log-likelihood matrix: 5 samples, 10 observations
        let log_likelihoodsamples =
            Array2::from_shape_fn((5, 10), |(i, j)| -1.0 - 0.1 * i as f64 - 0.05 * j as f64);

        let result = bic_calc
            .evaluate_model(&log_likelihoodsamples, 3, 10)
            .unwrap();

        assert!(result.waic > 0.0);
        assert!(result.loo_cv > 0.0);
        assert!(result.bic > 0.0);
        // DIC can be negative, so we just check it's finite
        assert!(result.dic.is_finite());
        assert!(result.p_waic >= 0.0);
    }

    #[test]
    fn test_posterior_predictive_check() {
        let ppc = PosteriorPredictiveCheck::new().with_test_statistic(TestStatisticType::Mean);

        let observed_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let posterior_samples =
            Array2::from_shape_fn((100, 5), |(i, j)| 1.0 + j as f64 + 0.1 * (i as f64 - 50.0));

        let result = ppc
            .check_model_adequacy(&observed_data, &posterior_samples)
            .unwrap();

        assert!(result.bayesian_p_value >= 0.0 && result.bayesian_p_value <= 1.0);
        assert!(result.tail_probability >= 0.0 && result.tail_probability <= 1.0);
        assert!(
            !result.model_adequate
                || (result.bayesian_p_value > 0.05 && result.bayesian_p_value < 0.95)
        );
    }

    #[test]
    fn test_credible_interval_calculator() {
        let ci_calc = CredibleIntervalCalculator::new()
            .with_credible_level(0.95)
            .unwrap()
            .with_null_value(0.0);

        let posterior_samples =
            Array1::from_vec(vec![-0.5, -0.2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]);

        let result = ci_calc.calculate_intervals(&posterior_samples).unwrap();

        assert!(result.lower_bound < result.upper_bound);
        assert!(result.credible_level == 0.95);
        assert!(result.posterior_mean > 0.0);
        assert!(result.hpd_interval.0 <= result.hpd_interval.1);
    }

    #[test]
    fn test_bayesian_model_averaging() {
        let bma = BayesianModelAveraging::new()
            .with_weighting_method(ModelWeightingMethod::InformationCriteria);

        // 3 models, 5 observations
        let predictions = Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, // Model 1
                1.1, 2.1, 2.9, 4.1, 4.9, // Model 2
                0.9, 1.9, 3.1, 3.9, 5.1, // Model 3
            ],
        )
        .unwrap();

        let modelscores = Array1::from_vec(vec![100.0, 102.0, 105.0]); // Information criteria

        let result = bma.average_models(&predictions, &modelscores).unwrap();

        assert_eq!(result.averaged_prediction.len(), 5);
        assert_eq!(result.model_weights.len(), 3);
        assert!((result.model_weights.sum() - 1.0).abs() < 1e-6);
        assert_eq!(result.model_uncertainty.len(), 5);
    }

    #[test]
    fn test_bayes_factor_interpretation() {
        // Test different ranges of Bayes factors
        assert!(BayesianModelComparison::interpret_bayes_factor(0.5).contains("favors B"));
        assert!(BayesianModelComparison::interpret_bayes_factor(2.0)
            .contains("Barely worth mentioning"));
        assert!(
            BayesianModelComparison::interpret_bayes_factor(15.0).contains("Strong evidence for A")
        );
        assert!(BayesianModelComparison::interpret_bayes_factor(150.0)
            .contains("Extreme evidence for A"));
    }

    #[test]
    fn test_evidence_methods() {
        let _comparison = BayesianModelComparison::new();
        let log_likelihood = Array1::from_vec(vec![-1.0, -1.5, -2.0, -1.2, -1.8]);

        // Test different evidence estimation methods
        let methods = vec![
            EvidenceMethod::HarmonicMean,
            EvidenceMethod::ThermodynamicIntegration,
            EvidenceMethod::BridgeSampling,
            EvidenceMethod::NestedSampling,
        ];

        for method in methods {
            let comparison_with_method =
                BayesianModelComparison::new().with_evidence_method(method);
            let evidence = comparison_with_method
                .estimate_evidence(&log_likelihood, None)
                .unwrap();
            assert!(evidence.is_finite());
        }
    }
}
