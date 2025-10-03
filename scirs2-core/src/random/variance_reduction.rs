//! Variance reduction techniques for Monte Carlo methods
//!
//! This module provides advanced variance reduction techniques that are essential
//! for efficient Monte Carlo simulations in scientific computing applications.

use crate::random::core::Random;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use std::collections::HashMap;

/// Antithetic variate sampling for variance reduction
///
/// Antithetic variates reduce variance by using negatively correlated samples.
/// For uniform random variables U, the antithetic variate is 1-U.
#[derive(Debug)]
pub struct AntitheticSampling<R: Rng> {
    rng: Random<R>,
    #[allow(dead_code)]
    stored_samples: HashMap<usize, Vec<f64>>,
}

impl<R: Rng> AntitheticSampling<R> {
    /// Create a new antithetic sampling generator
    pub fn new(rng: Random<R>) -> Self {
        Self {
            rng,
            stored_samples: HashMap::new(),
        }
    }

    /// Generate antithetic pairs of samples
    ///
    /// Returns (original_samples, antithetic_samples) where antithetic[i] = 1 - original[i]
    pub fn generate_antithetic_pairs(&mut self, count: usize) -> (Vec<f64>, Vec<f64>) {
        let original: Vec<f64> = (0..count)
            .map(|_| self.rng.sample(Uniform::new(0.0, 1.0).unwrap()))
            .collect();

        let antithetic: Vec<f64> = original.iter().map(|&x| 1.0 - x).collect();

        (original, antithetic)
    }

    /// Generate stratified samples for variance reduction
    ///
    /// Divides the unit interval into strata and samples uniformly within each stratum.
    /// This reduces variance compared to pure random sampling.
    pub fn stratified_samples(&mut self, strata: usize, samples_per_stratum: usize) -> Vec<f64> {
        let mut all_samples = Vec::new();

        for i in 0..strata {
            let stratum_start = i as f64 / strata as f64;
            let stratum_end = (i + 1) as f64 / strata as f64;

            for _ in 0..samples_per_stratum {
                let uniform_in_stratum = self.rng.sample(Uniform::new(0.0, 1.0).unwrap());
                let sample = stratum_start + uniform_in_stratum * (stratum_end - stratum_start);
                all_samples.push(sample);
            }
        }

        all_samples.shuffle(&mut self.rng.rng);
        all_samples
    }

    /// Generate Latin Hypercube samples
    ///
    /// A variance reduction technique that ensures samples are well-distributed
    /// across all dimensions of the parameter space.
    pub fn latin_hypercube_samples(
        &mut self,
        dimensions: usize,
        sample_count: usize,
    ) -> Vec<Vec<f64>> {
        let mut samples = vec![vec![0.0; dimensions]; sample_count];

        for dim in 0..dimensions {
            // Create stratified samples for this dimension
            let mut strata: Vec<f64> = (0..sample_count)
                .map(|i| {
                    (i as f64 + self.rng.sample(Uniform::new(0.0, 1.0).unwrap()))
                        / sample_count as f64
                })
                .collect();

            // Shuffle the strata for this dimension
            strata.shuffle(&mut self.rng.rng);

            // Assign to samples
            for (i, &value) in strata.iter().enumerate() {
                samples[i][dim] = value;
            }
        }

        samples
    }
}

impl AntitheticSampling<rand::rngs::ThreadRng> {
    /// Create antithetic sampling with default RNG
    pub fn with_default_rng() -> Self {
        Self::new(Random::default())
    }
}

/// Control variate method for variance reduction
///
/// Control variates use a correlated random variable with known expectation
/// to reduce the variance of the estimator.
#[derive(Debug)]
pub struct ControlVariate {
    control_mean: f64,
    optimal_coefficient: Option<f64>,
}

impl ControlVariate {
    /// Create a new control variate method
    ///
    /// # Parameters
    /// * `control_mean` - The known expectation of the control variate
    pub fn new(control_mean: f64) -> Self {
        Self {
            control_mean,
            optimal_coefficient: None,
        }
    }

    /// Convenience constructor for control variate with given mean
    pub fn mean(mean: f64) -> Self {
        Self::new(mean)
    }

    /// Estimate the optimal control coefficient
    ///
    /// Uses the sample covariance and variance to estimate the optimal coefficient
    /// that minimizes the variance of the control variate estimator.
    pub fn estimate_coefficient(&mut self, target_samples: &[f64], control_samples: &[f64]) {
        let n = target_samples.len() as f64;

        let target_mean = target_samples.iter().sum::<f64>() / n;
        let control_sample_mean = control_samples.iter().sum::<f64>() / n;

        let numerator: f64 = target_samples
            .iter()
            .zip(control_samples.iter())
            .map(|(&y, &x)| (y - target_mean) * (x - control_sample_mean))
            .sum();

        let denominator: f64 = control_samples
            .iter()
            .map(|&x| (x - control_sample_mean).powi(2))
            .sum();

        if denominator > 0.0 {
            self.optimal_coefficient = Some(numerator / denominator);
        }
    }

    /// Apply control variate correction
    ///
    /// Applies the control variate correction using the formula:
    /// Y_corrected = Y - c * (X - μ_X)
    /// where c is the optimal coefficient, X is the control variate, and μ_X is its mean.
    pub fn apply_correction(&self, target_samples: &[f64], control_samples: &[f64]) -> Vec<f64> {
        if let Some(c) = self.optimal_coefficient {
            target_samples
                .iter()
                .zip(control_samples.iter())
                .map(|(&y, &x)| y - c * (x - self.control_mean))
                .collect()
        } else {
            target_samples.to_vec()
        }
    }

    /// Get the current optimal coefficient
    pub fn coefficient(&self) -> Option<f64> {
        self.optimal_coefficient
    }

    /// Get the control mean
    pub fn control_mean(&self) -> f64 {
        self.control_mean
    }
}

/// Common ratio for variance reduction
///
/// Uses the known ratio between two correlated estimators to reduce variance.
#[derive(Debug)]
pub struct CommonRatio {
    known_ratio: f64,
}

impl CommonRatio {
    /// Create a new common ratio variance reduction method
    pub fn new(known_ratio: f64) -> Self {
        Self { known_ratio }
    }

    /// Apply common ratio correction
    ///
    /// Uses the formula: Y_corrected = Y * (known_ratio / sample_ratio)
    /// where sample_ratio is estimated from the sample data.
    pub fn apply_correction(
        &self,
        numerator_samples: &[f64],
        denominator_samples: &[f64],
    ) -> Vec<f64> {
        let n = numerator_samples.len();
        if n != denominator_samples.len() {
            return numerator_samples.to_vec();
        }

        let numerator_sum: f64 = numerator_samples.iter().sum();
        let denominator_sum: f64 = denominator_samples.iter().sum();

        if denominator_sum != 0.0 {
            let sample_ratio = numerator_sum / denominator_sum;
            let correction_factor = self.known_ratio / sample_ratio;
            numerator_samples
                .iter()
                .map(|&x| x * correction_factor)
                .collect()
        } else {
            numerator_samples.to_vec()
        }
    }
}

/// Importance splitting for rare event simulation
///
/// A variance reduction technique for estimating probabilities of rare events
/// by splitting the simulation into multiple levels.
#[derive(Debug)]
pub struct ImportanceSplitting<R: Rng> {
    rng: Random<R>,
    levels: Vec<f64>,
    splitting_factor: usize,
}

impl<R: Rng> ImportanceSplitting<R> {
    /// Create a new importance splitting method
    pub fn new(rng: Random<R>, levels: Vec<f64>, splitting_factor: usize) -> Self {
        Self {
            rng,
            levels,
            splitting_factor,
        }
    }

    /// Simulate rare event probability using importance splitting
    ///
    /// Returns an estimate of the probability of reaching the final level.
    pub fn estimate_probability<F>(&mut self, initial_samples: usize, simulation_fn: F) -> f64
    where
        F: Fn(&mut Random<R>) -> f64,
    {
        let mut current_samples = initial_samples;
        let mut probability = 1.0;

        for &level in &self.levels {
            let mut successes = 0;
            let mut new_samples = Vec::new();

            // Run simulations at current level
            for _ in 0..current_samples {
                let result = simulation_fn(&mut self.rng);
                if result >= level {
                    successes += 1;
                    new_samples.push(result);
                }
            }

            if successes == 0 {
                return 0.0; // No samples reached this level
            }

            let level_probability = successes as f64 / current_samples as f64;
            probability *= level_probability;

            // Split successful samples for next level
            current_samples = successes * self.splitting_factor;
        }

        probability
    }
}

impl ImportanceSplitting<rand::rngs::ThreadRng> {
    /// Create importance splitting with default RNG
    pub fn with_default_rng(levels: Vec<f64>, splitting_factor: usize) -> Self {
        Self::new(Random::default(), levels, splitting_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::core::seeded_rng;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_antithetic_sampling() {
        let mut antithetic = AntitheticSampling::new(seeded_rng(42));
        let (original, antithetic_vals) = antithetic.generate_antithetic_pairs(10);

        assert_eq!(original.len(), 10);
        assert_eq!(antithetic_vals.len(), 10);

        for (o, a) in original.iter().zip(antithetic_vals.iter()) {
            assert_abs_diff_eq!(o + a, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_stratified_sampling() {
        let mut antithetic = AntitheticSampling::new(seeded_rng(123));
        let samples = antithetic.stratified_samples(5, 10);

        assert_eq!(samples.len(), 50); // 5 strata * 10 samples per stratum
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_latin_hypercube_sampling() {
        let mut antithetic = AntitheticSampling::new(seeded_rng(456));
        let samples = antithetic.latin_hypercube_samples(3, 10);

        assert_eq!(samples.len(), 10); // 10 samples
        assert!(samples.iter().all(|sample| sample.len() == 3)); // 3 dimensions
        assert!(samples.iter().flatten().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_control_variate() {
        let mut control = ControlVariate::new(0.5);

        let target = vec![0.1, 0.3, 0.7, 0.9];
        let control_samples = vec![0.2, 0.4, 0.6, 0.8];

        control.estimate_coefficient(&target, &control_samples);
        let corrected = control.apply_correction(&target, &control_samples);

        assert_eq!(corrected.len(), target.len());
        assert!(control.coefficient().is_some());
    }

    #[test]
    fn test_common_ratio() {
        let ratio = CommonRatio::new(2.0);
        let numerator = vec![1.0, 2.0, 3.0, 4.0];
        let denominator = vec![0.6, 1.2, 1.8, 2.4]; // Should give ratio ≈ 1.67

        let corrected = ratio.apply_correction(&numerator, &denominator);
        assert_eq!(corrected.len(), numerator.len());
    }

    #[test]
    fn test_importance_splitting() {
        let levels = vec![0.5, 0.8, 0.95];
        let mut splitting = ImportanceSplitting::new(seeded_rng(789), levels, 2);

        let prob =
            splitting.estimate_probability(100, |rng| rng.sample(Uniform::new(0.0, 1.0).unwrap()));

        assert!((0.0..=1.0).contains(&prob));
    }
}
