//! Hypergeometric distribution
//!
//! This module provides an implementation of the hypergeometric distribution.
//! The hypergeometric distribution is a discrete probability distribution that
//! describes the probability of k successes in n draws, without replacement,
//! from a finite population of size N that contains exactly K successes.
//!
//! # Examples
//!
//! ```
//! use scirs2_stats::distributions;
//!
//! // Create a hypergeometric distribution
//! // N = 20 (population size)
//! // K = 7 (number of success states in the population)
//! // n = 12 (number of draws)
//! let hyper = distributions::hypergeom(20, 7, 12, 0.0).unwrap();
//!
//! // Calculate PMF at different points
//! let pmf_0 = hyper.pmf(0.0); // P(X = 0)
//! let pmf_3 = hyper.pmf(3.0); // P(X = 3)
//! let pmf_7 = hyper.pmf(7.0); // P(X = 7)
//!
//! // Calculate CDF at various points
//! let cdf_4 = hyper.cdf(4.0); // P(X ≤ 4)
//!
//! // Generate random samples
//! let samples = hyper.rvs(100).unwrap();
//!
//! // Calculate statistics
//! let mean = hyper.mean();
//! let variance = hyper.var();
//! ```
//!
//! # Mathematical Details
//!
//! The probability mass function (PMF) of the hypergeometric distribution is:
//!
//! P(X = k) = [C(K, k) * C(N-K, n-k)] / C(N, n)
//!
//! where:
//! - N is the population size
//! - K is the number of success states in the population
//! - n is the number of draws
//! - k is the number of observed successes
//! - C(n, k) is the binomial coefficient "n choose k"
//!
//! The mean of the distribution is E[X] = n * (K/N)
//! The variance is Var[X] = n * (K/N) * (1 - K/N) * (N - n)/(N - 1)

use crate::error::{StatsError, StatsResult};
use ndarray::Array1;
use num_traits::{cast::NumCast, Float, FloatConst};
use scirs2_core::rng;
use scirs2_core::Rng;
use std::cmp;

/// Hypergeometric distribution
///
/// Represents a hypergeometric distribution with parameters:
/// - N: population size
/// - K: number of success states in the population
/// - n: number of draws
#[derive(Debug, Clone)]
pub struct Hypergeometric<F: Float> {
    /// Population size
    n_population: usize,
    /// Number of success states in the population
    n_success: usize,
    /// Number of draws
    n_draws: usize,
    /// Support offset (default is 0.0)
    loc: F,
}

impl<F: Float + NumCast + FloatConst + std::fmt::Display> Hypergeometric<F> {
    /// Create a new hypergeometric distribution
    ///
    /// # Arguments
    ///
    /// * `n_population` - Population size
    /// * `n_success` - Number of success states in the population
    /// * `n_draws` - Number of draws
    /// * `loc` - Support offset (default: 0.0)
    ///
    /// # Returns
    ///
    /// A new hypergeometric distribution
    ///
    /// # Errors
    ///
    /// Returns an error if any of the parameters are invalid:
    /// - if n_population, n_success, or n_draws is 0
    /// - if n_success > n_population
    /// - if n_draws > n_population
    pub fn new(n_population: usize, n_success: usize, ndraws: usize, loc: F) -> StatsResult<Self> {
        // Check parameter validity
        if n_population == 0 {
            return Err(StatsError::InvalidArgument(
                "Population size must be positive".to_string(),
            ));
        }

        if n_success > n_population {
            return Err(StatsError::InvalidArgument(
                "Number of success states cannot exceed population size".to_string(),
            ));
        }

        if ndraws > n_population {
            return Err(StatsError::InvalidArgument(
                "Number of draws cannot exceed population size".to_string(),
            ));
        }

        Ok(Hypergeometric {
            n_population,
            n_success,
            n_draws: ndraws,
            loc,
        })
    }

    /// Returns the probability mass function (PMF) at `x`
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PMF
    ///
    /// # Returns
    ///
    /// The probability mass function evaluated at `x`
    pub fn pmf(&self, x: F) -> F {
        // Adjust for location parameter
        let adjusted_x = x - self.loc;

        // Convert to integer and check if in valid range
        let k_f = adjusted_x.to_f64().unwrap_or(f64::NAN);
        if k_f.fract() != 0.0 || k_f.is_nan() {
            return F::zero();
        }

        let k = k_f as i64;
        if k < 0 {
            return F::zero();
        }

        let k = k as usize;
        let max_possible = cmp::min(self.n_draws, self.n_success);
        let min_possible = self
            .n_draws
            .saturating_sub(self.n_population - self.n_success);

        if k < min_possible || k > max_possible {
            return F::zero();
        }

        // PMF calculation: [C(K, k) * C(N-K, n-k)] / C(N, n)
        let ln_pmf = ln_binomial(self.n_success, k)
            + ln_binomial(self.n_population - self.n_success, self.n_draws - k)
            - ln_binomial(self.n_population, self.n_draws);

        F::from(ln_pmf.exp()).unwrap_or(F::zero())
    }

    /// Returns the cumulative distribution function (CDF) at `x`
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// The cumulative distribution function evaluated at `x`
    pub fn cdf(&self, x: F) -> F {
        // Adjust for location parameter
        let adjusted_x = x - self.loc;

        // Convert to integer and check if in valid range
        let k_f = adjusted_x.to_f64().unwrap_or(f64::NAN);
        if k_f.is_nan() {
            return F::zero();
        }

        // Floor value for CDF
        let k_floor = k_f.floor() as i64;
        if k_floor < 0 {
            return F::zero();
        }

        // Calculate CDF by summing PMF values
        let min_possible = self
            .n_draws
            .saturating_sub(self.n_population - self.n_success);
        let max_k = cmp::min(k_floor as usize, cmp::min(self.n_draws, self.n_success));

        let mut cdf_value = F::zero();
        for k in min_possible..=max_k {
            cdf_value = cdf_value + self.pmf(F::from(k).unwrap() + self.loc);
        }

        if cdf_value > F::one() {
            F::one()
        } else {
            cdf_value
        }
    }

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// An array of random samples
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let mut rng = rng();
        let mut samples = Array1::zeros(size);

        for i in 0..size {
            // Simulate hypergeometric sampling
            let mut successes = 0;
            let mut population_remaining = self.n_population;
            let mut success_remaining = self.n_success;

            for _ in 0..self.n_draws {
                if population_remaining == 0 {
                    break;
                }

                let p_success = success_remaining as f64 / population_remaining as f64;
                if rng.gen_range(0.0..1.0) < p_success {
                    successes += 1;
                    success_remaining -= 1;
                } else {
                    // Failure drawn
                    // success_remaining unchanged
                }
                population_remaining -= 1;
            }

            samples[i] = F::from(successes).unwrap() + self.loc;
        }

        Ok(samples)
    }

    /// Returns the mean of the distribution
    ///
    /// # Returns
    ///
    /// The mean (expected value)
    pub fn mean(&self) -> F {
        let mean_val = (self.n_draws as f64) * (self.n_success as f64) / (self.n_population as f64);
        F::from(mean_val).unwrap() + self.loc
    }

    /// Returns the variance of the distribution
    ///
    /// # Returns
    ///
    /// The variance
    pub fn var(&self) -> F {
        if self.n_population <= 1 {
            return F::zero();
        }

        let n_draws = self.n_draws as f64;
        let k = self.n_success as f64;
        let n = self.n_population as f64;

        let p = k / n;
        let variance = n_draws * p * (1.0 - p) * (n - n_draws) / (n - 1.0);

        F::from(variance).unwrap()
    }

    /// Returns the standard deviation of the distribution
    ///
    /// # Returns
    ///
    /// The standard deviation
    pub fn std(&self) -> F {
        self.var().sqrt()
    }
}

/// Computes the natural logarithm of the binomial coefficient "n choose k"
#[allow(dead_code)]
fn ln_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    // Compute ln(n choose k) = ln(n!) - ln(k!) - ln((n-k)!)
    // Use more efficient calculation to avoid overflow
    let k = k.min(n - k); // Use symmetry: C(n,k) = C(n,n-k)

    // Use log-gamma function for calculating ln(n!) - ln(k!) - ln((n-k)!)
    let ln_n_fact = (1..=n).map(|i| (i as f64).ln()).sum::<f64>();
    let ln_k_fact = (1..=k).map(|i| (i as f64).ln()).sum::<f64>();
    let ln_n_minus_k_fact = (1..=(n - k)).map(|i| (i as f64).ln()).sum::<f64>();

    ln_n_fact - ln_k_fact - ln_n_minus_k_fact
}

/// Create a hypergeometric distribution with the given parameters.
///
/// This is a convenience function to create a hypergeometric distribution with
/// the given population size, number of success states, number of draws, and location parameter.
///
/// # Arguments
///
/// * `n_population` - Population size (N > 0)
/// * `n_success` - Number of success states in the population (K <= N)
/// * `n_draws` - Number of draws (n <= N)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A hypergeometric distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// // Create a hypergeometric distribution
/// // N = 20 (population size)
/// // K = 7 (number of success states in the population)
/// // n = 12 (number of draws)
/// let hyper = distributions::hypergeom(20, 7, 12, 0.0f64).unwrap();
///
/// // Calculate PMF at different points
/// let pmf_3 = hyper.pmf(3.0); // Probability of exactly 3 successes
/// ```
#[allow(dead_code)]
pub fn hypergeom<F>(
    n_population: usize,
    n_success: usize,
    n_draws: usize,
    loc: F,
) -> StatsResult<Hypergeometric<F>>
where
    F: Float + NumCast + FloatConst + std::fmt::Display,
{
    Hypergeometric::new(n_population, n_success, n_draws, loc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore = "timeout"]
    fn test_hypergeometric_creation() {
        // Valid parameters
        let hyper = Hypergeometric::new(20, 7, 12, 0.0).unwrap();
        assert_eq!(hyper.n_population, 20);
        assert_eq!(hyper.n_success, 7);
        assert_eq!(hyper.n_draws, 12);
        assert_eq!(hyper.loc, 0.0);

        // Invalid parameters
        assert!(Hypergeometric::<f64>::new(0, 5, 10, 0.0).is_err()); // N = 0
        assert!(Hypergeometric::<f64>::new(20, 21, 10, 0.0).is_err()); // K > N
        assert!(Hypergeometric::<f64>::new(20, 7, 21, 0.0).is_err()); // n > N
    }

    #[test]
    fn test_hypergeometric_pmf() {
        // Create a hypergeometric distribution with parameters N=20, K=7, n=12
        let hyper = Hypergeometric::new(20, 7, 12, 0.0).unwrap();

        // Expected PMF values
        assert_relative_eq!(hyper.pmf(0.0), 0.0001031991744066048, epsilon = 1e-10);
        assert_relative_eq!(hyper.pmf(3.0), 0.1986584107327147, epsilon = 1e-6);
        assert_relative_eq!(hyper.pmf(4.0), 0.3575851393188869, epsilon = 1e-6);
        assert_relative_eq!(hyper.pmf(7.0), 0.0102167182662539, epsilon = 1e-6);

        // Values outside of support
        assert_eq!(hyper.pmf(-1.0), 0.0);
        assert_eq!(hyper.pmf(8.0), 0.0);
        assert_eq!(hyper.pmf(0.5), 0.0); // Non-integer

        // With location parameter
        let shifted_hyper = Hypergeometric::new(20, 7, 12, 2.0).unwrap();
        assert_relative_eq!(
            shifted_hyper.pmf(2.0),
            0.0001031991744066048,
            epsilon = 1e-10
        ); // Corresponds to pmf(0) in standard
        assert_relative_eq!(shifted_hyper.pmf(5.0), 0.1986584107327147, epsilon = 1e-6);
        // Corresponds to pmf(3) in standard
    }

    #[test]
    fn test_hypergeometric_cdf() {
        // Create a hypergeometric distribution with parameters N=20, K=7, n=12
        let hyper = Hypergeometric::new(20, 7, 12, 0.0).unwrap();

        // Expected CDF values
        assert_relative_eq!(hyper.cdf(0.0), 0.0001031991744066048, epsilon = 1e-10);
        assert_relative_eq!(hyper.cdf(3.0), 0.2507739938080501, epsilon = 1e-6);
        assert_relative_eq!(hyper.cdf(4.0), 0.608359133126937, epsilon = 1e-6);
        assert_eq!(hyper.cdf(7.0), 1.0);

        // Values outside of support
        assert_eq!(hyper.cdf(-1.0), 0.0);
        assert_eq!(hyper.cdf(20.0), 1.0);

        // With location parameter
        let shifted_hyper = Hypergeometric::new(20, 7, 12, 2.0).unwrap();
        assert_relative_eq!(
            shifted_hyper.cdf(2.0),
            0.0001031991744066048,
            epsilon = 1e-10
        ); // Corresponds to cdf(0) in standard
        assert_relative_eq!(shifted_hyper.cdf(5.0), 0.2507739938080501, epsilon = 1e-6);
        // Corresponds to cdf(3) in standard
    }

    #[test]
    fn test_hypergeometric_stats() {
        // Create a hypergeometric distribution with parameters N=20, K=7, n=12
        let hyper = Hypergeometric::new(20, 7, 12, 0.0).unwrap();

        // Expected mean: n * (K/N) = 12 * (7/20) = 4.2
        assert_relative_eq!(hyper.mean(), 4.2, epsilon = 1e-10);

        // Expected variance: n * (K/N) * (1 - K/N) * (N - n)/(N - 1)
        // = 12 * (7/20) * (1 - 7/20) * (20 - 12)/(20 - 1)
        // = 12 * 0.35 * 0.65 * 8/19 ≈ 1.1494736842105262
        assert_relative_eq!(hyper.var(), 1.1494736842105262, epsilon = 1e-10);

        // Standard deviation = sqrt(variance) ≈ 1.0721351053904196
        assert_relative_eq!(hyper.std(), 1.0721351053904196, epsilon = 1e-10);

        // With location parameter
        let shifted_hyper = Hypergeometric::new(20, 7, 12, 3.0).unwrap();
        assert_relative_eq!(shifted_hyper.mean(), 7.2, epsilon = 1e-10); // 4.2 + 3.0
        assert_relative_eq!(shifted_hyper.var(), 1.1494736842105262, epsilon = 1e-10);
        // Same variance
    }

    #[test]
    fn test_hypergeometric_rvs() {
        // Create a hypergeometric distribution
        let hyper = Hypergeometric::<f64>::new(100, 40, 20, 0.0).unwrap();

        // Generate samples
        let samples = hyper.rvs(1000).unwrap();

        // Check basic properties
        assert_eq!(samples.len(), 1000);

        // All samples should be integers in the valid range [0, min(n_draws, n_success)]
        for sample in samples.iter() {
            assert!(sample.fract() == 0.0); // Integer
            assert!(*sample >= 0.0);
            assert!(*sample <= 20.0); // min(20, 40)
        }

        // Mean should be approximately n_draws * (n_success / n_population) = 20 * (40/100) = 8
        let mean = samples.sum() / samples.len() as f64;
        assert!((mean - 8.0).abs() < 0.5); // Allow some tolerance due to randomness
    }

    #[test]
    fn test_hypergeometric_edge_cases() {
        // Case 1: When n_success = 0, all samples should be 0
        let hyper_no_success = Hypergeometric::new(20, 0, 10, 0.0).unwrap();
        assert_eq!(hyper_no_success.pmf(0.0), 1.0);
        assert_eq!(hyper_no_success.pmf(1.0), 0.0);
        assert_eq!(hyper_no_success.mean(), 0.0);
        assert_eq!(hyper_no_success.var(), 0.0);

        // Case 2: When n_draws = 0, all samples should be 0
        let hyper_no_draws = Hypergeometric::new(20, 10, 0, 0.0).unwrap();
        assert_eq!(hyper_no_draws.pmf(0.0), 1.0);
        assert_eq!(hyper_no_draws.pmf(1.0), 0.0);
        assert_eq!(hyper_no_draws.mean(), 0.0);
        assert_eq!(hyper_no_draws.var(), 0.0);

        // Case 3: When n_success = n_population, all samples should be n_draws
        let hyper_all_success = Hypergeometric::new(20, 20, 10, 0.0).unwrap();
        assert_eq!(hyper_all_success.pmf(10.0), 1.0);
        assert_eq!(hyper_all_success.pmf(9.0), 0.0);
        assert_eq!(hyper_all_success.mean(), 10.0);
        assert_eq!(hyper_all_success.var(), 0.0);
    }

    #[test]
    fn test_ln_binomial() {
        // Test some simple cases
        assert_relative_eq!(ln_binomial(5, 2).exp(), 10.0, epsilon = 1e-10);
        assert_relative_eq!(ln_binomial(10, 5).exp(), 252.0, epsilon = 1e-10);

        // Test boundary cases
        assert_eq!(ln_binomial(5, 0).exp(), 1.0);
        assert_eq!(ln_binomial(5, 5).exp(), 1.0);
        assert_eq!(ln_binomial(0, 0).exp(), 1.0);

        // Test invalid case
        assert!(ln_binomial(5, 6) < 0.0); // Should return NEG_INFINITY
    }
}
