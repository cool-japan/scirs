//! Bayesian optimization implementation for hyperparameter tuning
//!
//! This module provides Gaussian Process-based Bayesian optimization
//! for efficient hyperparameter search.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rng, Rng, SeedableRng};
use std::collections::HashMap;

use crate::error::{ClusteringError, Result};

use super::config::*;

/// Bayesian optimizer using Gaussian Processes
pub struct BayesianOptimizer {
    parameter_names: Vec<String>,
    acquisition_function: AcquisitionFunction,
    bayesian_state: BayesianState,
    random_seed: Option<u64>,
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer
    pub fn new(
        parameter_names: Vec<String>,
        acquisition_function: AcquisitionFunction,
        random_seed: Option<u64>,
    ) -> Self {
        let bayesian_state = BayesianState {
            observations: Vec::new(),
            gp_mean: None,
            gp_covariance: None,
            acquisition_values: Vec::new(),
            parameter_names: parameter_names.clone(),
            gp_hyperparameters: GpHyperparameters {
                length_scales: vec![1.0; parameter_names.len()],
                signal_variance: 1.0,
                noise_variance: 0.1,
                kernel_type: KernelType::RBF { length_scale: 1.0 },
            },
            noise_level: 0.1,
            currentbest: f64::NEG_INFINITY,
        };

        Self {
            parameter_names,
            acquisition_function,
            bayesian_state,
            random_seed,
        }
    }

    /// Update observations with new parameter combinations
    pub fn update_observations(&mut self, combinations: &[HashMap<String, f64>]) {
        if combinations.is_empty() {
            return;
        }

        let n_samples = combinations.len();
        let _n_features = self.parameter_names.len();

        if n_samples < 2 {
            return;
        }

        self.optimize_gp_hyperparameters(combinations);
        self.build_covariance_matrix(combinations);
    }

    /// Optimize acquisition function to find next evaluation point
    pub fn optimize_acquisition_function(
        &self,
        search_space: &SearchSpace,
    ) -> Result<HashMap<String, f64>> {
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_point = HashMap::new();

        let n_candidates = 1000;
        let candidates = self.generate_random_candidates(search_space, n_candidates)?;

        for candidate in candidates {
            let acquisition_value = self.evaluate_acquisition_function(&candidate);

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_point = candidate;
            }
        }

        Ok(best_point)
    }

    /// Generate random candidate points for acquisition optimization
    fn generate_random_candidates(
        &self,
        search_space: &SearchSpace,
        n_candidates: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut candidates = Vec::new();
        let mut rng = match self.random_seed {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        for _ in 0..n_candidates {
            let mut candidate = HashMap::new();

            for (name, param) in &search_space.parameters {
                let value = match param {
                    HyperParameter::Integer { min, max } => rng.random_range(*min..=*max) as f64,
                    HyperParameter::Float { min, max } => rng.random_range(*min..=*max),
                    HyperParameter::Categorical { choices } => {
                        rng.random_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.random_range(0.0..1.0) < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_value = rng.random_range(log_min..=log_max);
                        log_value.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        choices[idx] as f64
                    }
                };

                candidate.insert(name.clone(), value);
            }

            candidates.push(candidate);
        }

        Ok(candidates)
    }

    /// Evaluate acquisition function at a point
    fn evaluate_acquisition_function(&self, point: &HashMap<String, f64>) -> f64 {
        let x = self.extract_feature_vector(point);
        let (mean, variance) = self.predict_gp(&x);
        let std_dev = variance.sqrt();

        match &self.acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                self.expected_improvement(mean, std_dev, self.bayesian_state.currentbest)
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mean + beta * std_dev,
            AcquisitionFunction::ProbabilityOfImprovement => {
                self.probability_of_improvement(mean, std_dev, self.bayesian_state.currentbest)
            }
            AcquisitionFunction::EntropySearch => -variance * (variance.ln()),
            AcquisitionFunction::KnowledgeGradient => std_dev * (1.0 / (1.0 + variance)),
            AcquisitionFunction::ThompsonSampling => {
                let mut rng = scirs2_core::random::rng();
                let sample: f64 = rng.random_range(0.0..1.0);
                mean + std_dev * self.inverse_normal_cdf(sample)
            }
        }
    }

    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std_dev: f64, currentbest: f64) -> f64 {
        if std_dev <= 1e-10 {
            return 0.0;
        }

        let improvement = mean - currentbest;
        let z = improvement / std_dev;

        improvement * self.normal_cdf(z) + std_dev * self.normal_pdf(z)
    }

    /// Probability of Improvement acquisition function
    fn probability_of_improvement(&self, mean: f64, std_dev: f64, currentbest: f64) -> f64 {
        if std_dev <= 1e-10 {
            return if mean > currentbest { 1.0 } else { 0.0 };
        }

        let z = (mean - currentbest) / std_dev;
        self.normal_cdf(z)
    }

    /// Gaussian Process prediction
    fn predict_gp(&self, x: &[f64]) -> (f64, f64) {
        if self.bayesian_state.observations.is_empty() {
            return (0.0, 1.0);
        }

        let mut mean = 0.0;
        let mut variance = 1.0;

        let mut total_weight = 0.0;
        for (params, score) in &self.bayesian_state.observations {
            let x_obs = self.extract_feature_vector(params);
            let similarity = self.compute_kernel(x, &x_obs);
            mean += similarity * score;
            total_weight += similarity;
        }

        if total_weight > 1e-10 {
            mean /= total_weight;
            variance = 1.0 - total_weight.min(1.0);
        }

        (mean, variance.max(1e-6))
    }

    /// Compute kernel function
    fn compute_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match &self.bayesian_state.gp_hyperparameters.kernel_type {
            KernelType::RBF { length_scale } => {
                let squared_distance: f64 =
                    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                self.bayesian_state.gp_hyperparameters.signal_variance
                    * (-squared_distance / (2.0 * length_scale.powi(2))).exp()
            }
            KernelType::Matern { length_scale, nu } => {
                let distance: f64 = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance == 0.0 {
                    self.bayesian_state.gp_hyperparameters.signal_variance
                } else {
                    let scaled_distance = (2.0 * nu).sqrt() * distance / length_scale;
                    let bessel_term = if nu == &0.5 {
                        (-scaled_distance).exp()
                    } else if nu == &1.5 {
                        (1.0 + scaled_distance) * (-scaled_distance).exp()
                    } else {
                        (-scaled_distance).exp()
                    };
                    self.bayesian_state.gp_hyperparameters.signal_variance * bessel_term
                }
            }
            KernelType::Linear => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                self.bayesian_state.gp_hyperparameters.signal_variance * dot_product
            }
            KernelType::Polynomial { degree } => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                self.bayesian_state.gp_hyperparameters.signal_variance
                    * (1.0 + dot_product).powf(*degree as f64)
            }
        }
    }

    /// Optimize GP hyperparameters using maximum likelihood
    fn optimize_gp_hyperparameters(&mut self, combinations: &[HashMap<String, f64>]) {
        if combinations.len() < 3 {
            return;
        }

        for (i, param_name) in self.parameter_names.iter().enumerate() {
            let values: Vec<f64> = combinations
                .iter()
                .filter_map(|c| c.get(param_name))
                .copied()
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

                if i < self.bayesian_state.gp_hyperparameters.length_scales.len() {
                    self.bayesian_state.gp_hyperparameters.length_scales[i] =
                        variance.sqrt().max(0.1);
                }
            }
        }

        if !self.bayesian_state.observations.is_empty() {
            let scores: Vec<f64> = self
                .bayesian_state
                .observations
                .iter()
                .map(|(_, s)| *s)
                .collect();
            let score_mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let score_variance =
                scores.iter().map(|s| (s - score_mean).powi(2)).sum::<f64>() / scores.len() as f64;

            self.bayesian_state.gp_hyperparameters.signal_variance = score_variance.max(0.1);
            self.bayesian_state.gp_hyperparameters.noise_variance =
                (score_variance * 0.1).max(0.01);
        }
    }

    /// Build covariance matrix for Gaussian Process
    fn build_covariance_matrix(&mut self, combinations: &[HashMap<String, f64>]) {
        let n_samples = combinations.len();
        let mut covariance = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let x_i = self.extract_feature_vector(&combinations[i]);
                let x_j = self.extract_feature_vector(&combinations[j]);
                covariance[[i, j]] = self.compute_kernel(&x_i, &x_j);
            }
        }

        for i in 0..n_samples {
            covariance[[i, i]] += self.bayesian_state.gp_hyperparameters.noise_variance;
        }

        self.bayesian_state.gp_covariance = Some(covariance);
    }

    /// Extract feature vector from parameter map
    fn extract_feature_vector(&self, params: &HashMap<String, f64>) -> Vec<f64> {
        self.parameter_names
            .iter()
            .map(|name| params.get(name).copied().unwrap_or(0.0))
            .collect()
    }

    /// Standard normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Standard normal PDF
    fn normal_pdf(&self, x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
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

    /// Inverse normal CDF approximation
    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if (p - 0.5).abs() < 1e-10 {
            return 0.0;
        }

        let a0 = -3.969683028665376e+01;
        let a1 = 2.209460984245205e+02;
        let a2 = -2.759285104469687e+02;
        let a3 = 1.383_577_518_672_69e2;
        let a4 = -3.066479806614716e+01;
        let a5 = 2.506628277459239e+00;

        let b1 = -5.447609879822406e+01;
        let b2 = 1.615858368580409e+02;
        let b3 = -1.556989798598866e+02;
        let b4 = 6.680131188771972e+01;
        let b5 = -1.328068155288572e+01;

        let c0 = -7.784894002430293e-03;
        let c1 = -3.223964580411365e-01;
        let c2 = -2.400758277161838e+00;
        let c3 = -2.549732539343734e+00;
        let c4 = 4.374664141464968e+00;
        let c5 = 2.938163982698783e+00;

        let d1 = 7.784695709041462e-03;
        let d2 = 3.224671290700398e-01;
        let d3 = 2.445134137142996e+00;
        let d4 = 3.754408661907416e+00;

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }

        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    }
}
