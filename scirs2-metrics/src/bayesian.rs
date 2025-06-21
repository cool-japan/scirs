//! Bayesian evaluation metrics
//!
//! This module provides Bayesian approaches to model evaluation and comparison,
//! including Bayes factors, information criteria, posterior predictive checks,
//! and Bayesian model averaging metrics.

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

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
    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
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
        log_prior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        match self.evidence_method {
            EvidenceMethod::HarmonicMean => self.harmonic_mean_evidence(log_likelihood, log_prior),
            EvidenceMethod::ThermodynamicIntegration => {
                self.thermodynamic_integration_evidence(log_likelihood, log_prior)
            }
            EvidenceMethod::BridgeSampling => {
                self.bridge_sampling_evidence(log_likelihood, log_prior)
            }
            EvidenceMethod::NestedSampling => {
                self.nested_sampling_evidence(log_likelihood, log_prior)
            }
        }
    }

    /// Harmonic mean estimator for marginal likelihood
    fn harmonic_mean_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        log_prior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        if log_likelihood.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood array".to_string(),
            ));
        }

        // Calculate log(prior * likelihood) for each sample
        let log_posterior: Array1<f64> = if let Some(prior) = log_prior {
            if prior.len() != log_likelihood.len() {
                return Err(MetricsError::InvalidInput(
                    "Prior and likelihood arrays must have same length".to_string(),
                ));
            }
            log_likelihood + prior
        } else {
            log_likelihood.clone()
        };

        // Harmonic mean: 1/E[1/L] where L is likelihood
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

    /// Thermodynamic integration for evidence estimation
    fn thermodynamic_integration_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        _log_prior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        // Simplified thermodynamic integration
        // In practice, this would require running MCMC at different temperatures
        let mean_log_likelihood = log_likelihood.mean().unwrap_or(0.0);

        // Approximate integration using trapezoidal rule
        let num_temps = 10;
        let mut integral = 0.0;

        for i in 0..num_temps {
            let beta1 = i as f64 / (num_temps - 1) as f64;
            let beta2 = (i + 1) as f64 / (num_temps - 1) as f64;

            // Simplified: assume mean likelihood at each temperature
            let val1 = beta1 * mean_log_likelihood;
            let val2 = beta2 * mean_log_likelihood;

            integral += 0.5 * (val1 + val2) * (beta2 - beta1);
        }

        Ok(integral)
    }

    /// Bridge sampling for evidence estimation
    fn bridge_sampling_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        _log_prior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        // Simplified bridge sampling approximation
        // In practice, this requires samples from both prior and posterior
        let mean_log_likelihood = log_likelihood.mean().unwrap_or(0.0);
        let var_log_likelihood = self.calculate_variance(log_likelihood)?;

        // Rough approximation using normal bridge
        let evidence_approx = mean_log_likelihood - 0.5 * var_log_likelihood.ln();

        Ok(evidence_approx)
    }

    /// Nested sampling approximation for evidence
    fn nested_sampling_evidence(
        &self,
        log_likelihood: &Array1<f64>,
        _log_prior: Option<&Array1<f64>>,
    ) -> Result<f64> {
        // Simplified nested sampling approximation
        let n = log_likelihood.len();
        if n == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood array".to_string(),
            ));
        }

        // Sort likelihoods in ascending order
        let mut sorted_likelihoods: Vec<f64> = log_likelihood.to_vec();
        sorted_likelihoods.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Approximate evidence using trapezoidal integration
        let mut evidence = 0.0;
        for (i, &likelihood) in sorted_likelihoods.iter().enumerate() {
            let prior_mass = (n - i - 1) as f64 / n as f64;
            let next_prior_mass = if i + 1 < n {
                (n - i - 2) as f64 / n as f64
            } else {
                0.0
            };

            evidence += likelihood.exp() * (prior_mass - next_prior_mass);
        }

        Ok(evidence.max(1e-300).ln()) // Ensure positive for log
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
    fn calculate_variance(&self, data: &Array1<f64>) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.mean().unwrap_or(0.0);
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
    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
        self
    }

    /// Calculate comprehensive Bayesian information criteria
    pub fn evaluate_model(
        &self,
        log_likelihood_samples: &Array2<f64>, // Shape: (n_samples, n_observations)
        num_parameters: usize,
        num_observations: usize,
    ) -> Result<BayesianInformationResults> {
        if log_likelihood_samples.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood samples".to_string(),
            ));
        }

        // Calculate WAIC and effective parameters
        let (waic, p_waic) = self.calculate_waic(log_likelihood_samples)?;

        // Calculate LOO-CV
        let loo_cv = self.calculate_loo_cv(log_likelihood_samples)?;

        // Calculate BIC (requires point estimate of log-likelihood)
        let mean_log_likelihood: f64 = log_likelihood_samples.mean().unwrap_or(0.0);
        let bic = -2.0 * mean_log_likelihood * num_observations as f64
            + (num_parameters as f64) * (num_observations as f64).ln();

        // Calculate DIC
        let dic = self.calculate_dic(log_likelihood_samples)?;

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
    fn calculate_waic(&self, log_likelihood_samples: &Array2<f64>) -> Result<(f64, f64)> {
        let (n_samples, n_obs) = log_likelihood_samples.dim();
        if n_samples == 0 || n_obs == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood samples".to_string(),
            ));
        }

        let mut lppd = 0.0; // Log pointwise predictive density
        let mut p_waic = 0.0; // Effective number of parameters

        for i in 0..n_obs {
            let obs_likelihoods = log_likelihood_samples.column(i);

            // Calculate log mean of exp(log_likelihood) for this observation
            let max_ll = obs_likelihoods
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f64 = obs_likelihoods.iter().map(|&x| (x - max_ll).exp()).sum();
            let log_mean_exp = (sum_exp / n_samples as f64).ln() + max_ll;

            lppd += log_mean_exp;

            // Calculate variance of log-likelihood for this observation
            let mean_ll = obs_likelihoods.mean().unwrap_or(0.0);
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
    fn calculate_loo_cv(&self, log_likelihood_samples: &Array2<f64>) -> Result<f64> {
        let (n_samples, n_obs) = log_likelihood_samples.dim();
        if n_samples == 0 || n_obs == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty likelihood samples".to_string(),
            ));
        }

        let mut loo_sum = 0.0;

        for i in 0..n_obs {
            let obs_likelihoods = log_likelihood_samples.column(i);

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
    fn calculate_dic(&self, log_likelihood_samples: &Array2<f64>) -> Result<f64> {
        let mean_deviance = -2.0 * log_likelihood_samples.mean().unwrap_or(0.0);

        // Calculate deviance at posterior mean (simplified)
        let posterior_mean_ll = log_likelihood_samples.mean_axis(Axis(0)).unwrap();
        let deviance_at_mean = -2.0 * posterior_mean_ll.sum();

        let p_dic = mean_deviance - deviance_at_mean;
        let dic = mean_deviance + p_dic;

        Ok(dic)
    }

    /// Calculate Pareto Smoothed Importance Sampling weights (simplified)
    fn calculate_psis_weights(&self, log_weights: &Array1<f64>) -> Result<Array1<f64>> {
        let n = log_weights.len();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        // Subtract maximum for numerical stability
        let max_weight = log_weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let weights: Array1<f64> = log_weights.mapv(|x| (x - max_weight).exp());

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
    pub fn with_test_statistic(mut self, test_statistic: TestStatisticType) -> Self {
        self.test_statistic = test_statistic;
        self
    }

    /// Set number of posterior predictive samples
    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
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
                "Empty posterior predictive samples".to_string(),
            ));
        }

        let (n_samples, n_obs) = posterior_predictive_samples.dim();
        if observed_data.len() != n_obs {
            return Err(MetricsError::InvalidInput(
                "Observed data length doesn't match predictive samples".to_string(),
            ));
        }

        // Calculate test statistic for observed data
        let observed_statistic = self.calculate_test_statistic(observed_data)?;

        // Calculate test statistics for posterior predictive samples
        let mut predicted_statistics = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let sample = posterior_predictive_samples.row(i).to_owned();
            let statistic = self.calculate_test_statistic(&sample)?;
            predicted_statistics.push(statistic);
        }

        let predicted_statistics = Array1::from_vec(predicted_statistics);
        let predicted_statistic_mean = predicted_statistics.mean().unwrap_or(0.0);
        let predicted_statistic_std = self.calculate_std(&predicted_statistics)?;

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
            TestStatisticType::Mean => Ok(data.mean().unwrap_or(0.0)),
            TestStatisticType::Variance => {
                let mean = data.mean().unwrap_or(0.0);
                let variance =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                Ok(variance)
            }
            TestStatisticType::Minimum => Ok(data.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            TestStatisticType::Maximum => Ok(data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            TestStatisticType::Custom(_name) => {
                // For custom functions, implement specific logic
                // For now, return mean as default
                Ok(data.mean().unwrap_or(0.0))
            }
        }
    }

    /// Calculate standard deviation
    fn calculate_std(&self, data: &Array1<f64>) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.mean().unwrap_or(0.0);
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
    pub fn with_credible_level(mut self, level: f64) -> Self {
        if level <= 0.0 || level >= 1.0 {
            panic!("Credible level must be between 0 and 1");
        }
        self.credible_level = level;
        self
    }

    /// Set null hypothesis value for testing
    pub fn with_null_value(mut self, null_value: f64) -> Self {
        self.null_value = Some(null_value);
        self
    }

    /// Calculate credible intervals from posterior samples
    pub fn calculate_intervals(
        &self,
        posterior_samples: &Array1<f64>,
    ) -> Result<CredibleIntervalResults> {
        if posterior_samples.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty posterior samples".to_string(),
            ));
        }

        // Sort samples for quantile calculation
        let mut sorted_samples = posterior_samples.to_vec();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_samples.len();
        let alpha = 1.0 - self.credible_level;

        // Equal-tailed credible interval
        let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize - 1;

        let lower_bound = sorted_samples[lower_idx.min(n - 1)];
        let upper_bound = sorted_samples[upper_idx.min(n - 1)];

        // Posterior statistics
        let posterior_mean = posterior_samples.mean().unwrap_or(0.0);
        let posterior_median = if n % 2 == 0 {
            (sorted_samples[n / 2 - 1] + sorted_samples[n / 2]) / 2.0
        } else {
            sorted_samples[n / 2]
        };

        // Check if null value is contained
        let contains_null = if let Some(null_val) = self.null_value {
            null_val >= lower_bound && null_val <= upper_bound
        } else {
            false
        };

        // Calculate HPD interval (simplified)
        let hpd_interval = self.calculate_hpd_interval(&sorted_samples)?;

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
    fn calculate_hpd_interval(&self, sorted_samples: &[f64]) -> Result<(f64, f64)> {
        let n = sorted_samples.len();
        let interval_length = (self.credible_level * n as f64).round() as usize;

        if interval_length >= n {
            return Ok((sorted_samples[0], sorted_samples[n - 1]));
        }

        // Find interval with minimum width
        let mut min_width = f64::INFINITY;
        let mut best_lower = sorted_samples[0];
        let mut best_upper = sorted_samples[n - 1];

        for i in 0..=(n - interval_length) {
            let lower = sorted_samples[i];
            let upper = sorted_samples[i + interval_length - 1];
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
        predictions: &Array2<f64>,  // Shape: (n_models, n_observations)
        model_scores: &Array1<f64>, // Model comparison scores
    ) -> Result<BayesianModelAveragingResults> {
        let (n_models, n_obs) = predictions.dim();
        if model_scores.len() != n_models {
            return Err(MetricsError::InvalidInput(
                "Number of model scores must match number of models".to_string(),
            ));
        }

        if n_models == 0 || n_obs == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty predictions array".to_string(),
            ));
        }

        // Calculate model weights
        let model_weights = self.calculate_model_weights(model_scores)?;

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

        // Calculate within-model variance (simplified)
        let within_model_variance = Array1::from_elem(n_obs, 0.1); // Placeholder
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
    fn calculate_model_weights(&self, model_scores: &Array1<f64>) -> Result<Array1<f64>> {
        match self.weighting_method {
            ModelWeightingMethod::MarginalLikelihood => {
                self.marginal_likelihood_weights(model_scores)
            }
            ModelWeightingMethod::InformationCriteria => {
                self.information_criteria_weights(model_scores)
            }
            ModelWeightingMethod::CrossValidation => self.cross_validation_weights(model_scores),
            ModelWeightingMethod::Equal => {
                let n = model_scores.len();
                Ok(Array1::from_elem(n, 1.0 / n as f64))
            }
        }
    }

    /// Calculate weights from marginal likelihoods
    fn marginal_likelihood_weights(
        &self,
        log_marginal_likelihoods: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Normalize log marginal likelihoods to get model probabilities
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
    fn cross_validation_weights(&self, cv_scores: &Array1<f64>) -> Result<Array1<f64>> {
        // Normalize CV scores to get weights
        let min_score = cv_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let shifted_scores: Array1<f64> = cv_scores.mapv(|x| x - min_score + 1e-6);
        let sum_scores = shifted_scores.sum();

        if sum_scores > 1e-10 {
            Ok(shifted_scores / sum_scores)
        } else {
            let n = cv_scores.len();
            Ok(Array1::from_elem(n, 1.0 / n as f64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

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
        let log_likelihood_samples =
            Array2::from_shape_fn((5, 10), |(i, j)| -1.0 - 0.1 * i as f64 - 0.05 * j as f64);

        let result = bic_calc
            .evaluate_model(&log_likelihood_samples, 3, 10)
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

        let model_scores = Array1::from_vec(vec![100.0, 102.0, 105.0]); // Information criteria

        let result = bma.average_models(&predictions, &model_scores).unwrap();

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
