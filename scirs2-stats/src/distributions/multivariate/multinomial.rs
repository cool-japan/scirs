//! Multinomial distribution functions
//!
//! This module provides functionality for the Multinomial distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};
// NOTE: rand, distr: weighted may not be available in current version
// use scirs2_core::random::weighted::WeightedAliasIndex;
use scirs2_core::random::prelude::*;
use scirs2_core::validation::{check_probabilities, check_probabilities_sum_to_one};
use scirs2_core::Rng;
use std::fmt::Debug;

/// Implementation of the factorial function
#[allow(dead_code)]
fn factorial(n: u64) -> f64 {
    if n <= 1 {
        return 1.0;
    }

    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Compute the multinomial coefficient
///
/// (n choose n₁, n₂, ..., nₖ) = n! / (n₁! * n₂! * ... * nₖ!)
#[allow(dead_code)]
fn multinomial_coef(n: u64, xs: &[u64]) -> f64 {
    let mut denominator = 1.0;
    for &x in xs {
        denominator *= factorial(x);
    }
    factorial(n) / denominator
}

/// Multinomial distribution structure
///
/// The multinomial distribution is a generalization of the binomial distribution.
/// It models the probability of counts for each side of a k-sided die rolled n times.
#[derive(Debug, Clone)]
pub struct Multinomial {
    /// Number of trials
    pub n: u64,
    /// Probability of each outcome (must sum to 1)
    pub p: Array1<f64>,
    // Alias sampler for efficient random sampling (temporarily disabled)
    // alias_sampler: WeightedAliasIndex<f64>,
}

impl Multinomial {
    /// Create a new Multinomial distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `n` - Number of trials
    /// * `p` - Probability of each outcome (must sum to 1)
    ///
    /// # Returns
    ///
    /// * A new Multinomial distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// // Create a multinomial distribution for a 3-sided die rolled 10 times
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5]; // Probabilities for each outcome
    /// let multinomial = Multinomial::new(n, p).unwrap();
    /// ```
    pub fn new<D>(n: u64, p: ArrayBase<D, Ix1>) -> StatsResult<Self>
    where
        D: Data<Elem = f64>,
    {
        let p_owned = p.to_owned();

        // Validate that probabilities are non-negative and sum to 1 using core validation
        check_probabilities(&p_owned, "Probabilities").map_err(StatsError::from)?;
        check_probabilities_sum_to_one(&p_owned, "Probabilities", None)
            .map_err(StatsError::from)?;

        // Create alias sampler for efficient random sampling (temporarily disabled)
        // let alias_sampler = match WeightedAliasIndex::new(p_owned.iter().cloned().collect()) {
        //     Ok(sampler) => sampler,
        //     Err(_) => {
        //         return Err(StatsError::ComputationError(
        //             "Failed to create alias sampler for random sampling".to_string(),
        //         ))
        //     }
        // };

        Ok(Multinomial {
            n,
            p: p_owned,
            // alias_sampler,
        })
    }

    /// Calculate the probability mass function (PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PMF (must be a vector of non-negative integers that sum to n)
    ///
    /// # Returns
    ///
    /// * The value of the PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5];
    /// let multinomial = Multinomial::new(n, p).unwrap();
    ///
    /// // Calculate PMF at x = [2, 3, 5]
    /// let x = array![2.0, 3.0, 5.0];
    /// let pmf_value = multinomial.pmf(&x);
    /// ```
    pub fn pmf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        let x_vec = x.to_owned();

        // Check if x has the right dimension
        if x_vec.len() != self.p.len() {
            return 0.0;
        }

        // Convert x to u64 and check if all values are non-negative integers that sum to n
        let mut x_u64 = Vec::with_capacity(x_vec.len());
        let mut sum = 0;

        for &val in x_vec.iter() {
            // Check if value is a non-negative integer
            if val < 0.0 || (val - val.floor()).abs() > 1e-10 {
                return 0.0;
            }

            let val_u64 = val as u64;
            x_u64.push(val_u64);
            sum += val_u64;
        }

        // Check if values sum to n
        if sum != self.n {
            return 0.0;
        }

        // Calculate the multinomial PMF:
        // P(X = x) = n! / (x₁! * x₂! * ... * xₖ!) * p₁^x₁ * p₂^x₂ * ... * pₖ^xₖ

        // Multinomial coefficient
        let coef = multinomial_coef(self.n, &x_u64);

        // Product of p_i^x_i
        let mut product = 1.0;
        for (i, &count) in x_u64.iter().enumerate() {
            product *= self.p[i].powf(count as f64);
        }

        coef * product
    }

    /// Calculate the log probability mass function (log PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the log PMF (must be a vector of non-negative integers that sum to n)
    ///
    /// # Returns
    ///
    /// * The value of the log PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5];
    /// let multinomial = Multinomial::new(n, p).unwrap();
    ///
    /// // Calculate log PMF at x = [2, 3, 5]
    /// let x = array![2.0, 3.0, 5.0];
    /// let logpmf_value = multinomial.logpmf(&x);
    /// ```
    pub fn logpmf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        let x_vec = x.to_owned();

        // Check if x has the right dimension
        if x_vec.len() != self.p.len() {
            return f64::NEG_INFINITY;
        }

        // Convert x to u64 and check if all values are non-negative integers that sum to n
        let mut x_u64 = Vec::with_capacity(x_vec.len());
        let mut sum = 0;

        for &val in x_vec.iter() {
            // Check if value is a non-negative integer
            if val < 0.0 || (val - val.floor()).abs() > 1e-10 {
                return f64::NEG_INFINITY;
            }

            let val_u64 = val as u64;
            x_u64.push(val_u64);
            sum += val_u64;
        }

        // Check if values sum to n
        if sum != self.n {
            return f64::NEG_INFINITY;
        }

        // Calculate the log multinomial PMF:
        // log(P(X = x)) = log(n! / (x₁! * x₂! * ... * xₖ!)) + x₁*log(p₁) + x₂*log(p₂) + ... + xₖ*log(pₖ)

        // Log of multinomial coefficient
        let log_coef = factorial(self.n).ln();
        let mut log_denom = 0.0;
        for &count in &x_u64 {
            log_denom += factorial(count).ln();
        }

        // Sum of x_i*log(p_i)
        let mut log_prob_sum = 0.0;
        for (i, &count) in x_u64.iter().enumerate() {
            if count > 0 {
                log_prob_sum += (count as f64) * self.p[i].ln();
            }
        }

        log_coef - log_denom + log_prob_sum
    }

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// * Vector of random samples (each sample is a vector of counts)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5];
    /// let multinomial = Multinomial::new(n, p).unwrap();
    ///
    /// // Generate 5 random samples
    /// let samples = multinomial.rvs(5).unwrap();
    /// assert_eq!(samples.len(), 5);
    /// assert_eq!(samples[0].len(), 3);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<Array1<f64>>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);
        let k = self.p.len();

        for _ in 0..size {
            // Initialize counts to zero
            let mut counts = vec![0u64; k];

            // Simulate n trials
            for _ in 0..self.n {
                // Sample category using cumulative probability
                let u: f64 = rng.random();
                let mut cumulative = 0.0;
                let mut category = 0;
                for (i, &prob) in self.p.iter().enumerate() {
                    cumulative += prob;
                    if u <= cumulative {
                        category = i;
                        break;
                    }
                }
                counts[category] += 1;
            }

            // Convert to floating-point array for consistency with other distributions
            let sample = Array1::from_iter(counts.iter().map(|&x| x as f64));
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Generate a single random sample from the distribution
    ///
    /// # Returns
    ///
    /// * A random sample (a vector of counts)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5];
    /// let multinomial = Multinomial::new(n, p).unwrap();
    ///
    /// // Generate a single random sample
    /// let sample = multinomial.rvs_single().unwrap();
    /// assert_eq!(sample.len(), 3);
    /// ```
    pub fn rvs_single(&self) -> StatsResult<Array1<f64>> {
        let samples = self.rvs(1)?;
        Ok(samples[0].clone())
    }

    /// Calculate the mean of the distribution
    ///
    /// # Returns
    ///
    /// * Mean vector (n * p)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5];
    /// let multinomial = Multinomial::new(n, p).unwrap();
    ///
    /// let mean = multinomial.mean();
    /// // Mean should be [2.0, 3.0, 5.0]
    /// ```
    pub fn mean(&self) -> Array1<f64> {
        let n_f64 = self.n as f64;
        self.p.mapv(|p_i| n_f64 * p_i)
    }

    /// Calculate the covariance matrix of the distribution
    ///
    /// # Returns
    ///
    /// * Covariance matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
    ///
    /// let n = 10;
    /// let p = array![0.2, 0.3, 0.5];
    /// let multinomial = Multinomial::new(n, p).unwrap();
    ///
    /// let cov = multinomial.cov();
    /// ```
    pub fn cov(&self) -> scirs2_core::ndarray::Array2<f64> {
        let k = self.p.len();
        let n_f64 = self.n as f64;
        let mut cov = scirs2_core::ndarray::Array2::zeros((k, k));

        // Fill the covariance matrix
        // Diagonal: n*p_i*(1-p_i)
        // Off-diagonal: -n*p_i*p_j
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    cov[[i, j]] = n_f64 * self.p[i] * (1.0 - self.p[i]);
                } else {
                    cov[[i, j]] = -n_f64 * self.p[i] * self.p[j];
                }
            }
        }

        cov
    }
}

/// Create a Multinomial distribution with the given parameters.
///
/// This is a convenience function to create a Multinomial distribution with
/// the given number of trials and probability vector.
///
/// # Arguments
///
/// * `n` - Number of trials
/// * `p` - Probability of each outcome (must sum to 1)
///
/// # Returns
///
/// * A Multinomial distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let n = 10;
/// let p = array![0.2, 0.3, 0.5]; // Probabilities for each outcome
/// let multinomial = multivariate::multinomial(n, p).unwrap();
/// ```
#[allow(dead_code)]
pub fn multinomial<D>(n: u64, p: ArrayBase<D, Ix1>) -> StatsResult<Multinomial>
where
    D: Data<Elem = f64>,
{
    Multinomial::new(n, p)
}

/// Implementation of SampleableDistribution for Multinomial
impl SampleableDistribution<Array1<f64>> for Multinomial {
    fn rvs(&self, size: usize) -> StatsResult<Vec<Array1<f64>>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multinomial_creation() {
        // Valid multinomial
        let n = 10;
        let p = array![0.2, 0.3, 0.5];
        let multinomial = Multinomial::new(n, p.clone()).unwrap();
        assert_eq!(multinomial.n, n);
        assert_eq!(multinomial.p, p);

        // Invalid probabilities (don't sum to 1)
        let p_invalid_sum = array![0.2, 0.3, 0.6]; // sum = 1.1
        assert!(Multinomial::new(n, p_invalid_sum).is_err());

        // Invalid probabilities (negative values)
        let p_negative = array![0.2, -0.1, 0.9];
        assert!(Multinomial::new(n, p_negative).is_err());
    }

    #[test]
    fn test_multinomial_pmf() {
        let n = 5;
        let p = array![0.5, 0.5];
        let multinomial = Multinomial::new(n, p).unwrap();

        // PMF at x = [2, 3]
        let x1 = array![2.0, 3.0];
        let pmf1 = multinomial.pmf(&x1);

        // Calculate expected PMF: 5!/(2!*3!) * 0.5^2 * 0.5^3 = 10 * 0.25 * 0.125 = 0.3125
        let expected_pmf1 = 0.3125;
        assert_relative_eq!(pmf1, expected_pmf1, epsilon = 1e-10);

        // PMF at x = [5, 0]
        let x2 = array![5.0, 0.0];
        let pmf2 = multinomial.pmf(&x2);

        // Calculate expected PMF: 5!/(5!*0!) * 0.5^5 * 0.5^0 = 1 * 0.03125 * 1 = 0.03125
        let expected_pmf2 = 0.03125;
        assert_relative_eq!(pmf2, expected_pmf2, epsilon = 1e-10);

        // PMF at invalid x (doesn't sum to n)
        let x_invalid = array![2.0, 2.0]; // sum = 4 != 5
        let pmf_invalid = multinomial.pmf(&x_invalid);
        assert_eq!(pmf_invalid, 0.0);

        // PMF at invalid x (non-integer values)
        let x_non_int = array![2.5, 2.5];
        let pmf_non_int = multinomial.pmf(&x_non_int);
        assert_eq!(pmf_non_int, 0.0);

        // PMF at invalid x (wrong dimension)
        let x_wrong_dim = array![2.0, 3.0, 0.0];
        let pmf_wrong_dim = multinomial.pmf(&x_wrong_dim);
        assert_eq!(pmf_wrong_dim, 0.0);
    }

    #[test]
    fn test_multinomial_logpmf() {
        let n = 5;
        let p = array![0.5, 0.5];
        let multinomial = Multinomial::new(n, p).unwrap();

        // LogPMF at x = [2, 3]
        let x1 = array![2.0, 3.0];
        let logpmf1 = multinomial.logpmf(&x1);
        let pmf1 = multinomial.pmf(&x1);

        // Check that exp(logPMF) = PMF
        assert_relative_eq!(logpmf1.exp(), pmf1, epsilon = 1e-10);

        // LogPMF at invalid x (doesn't sum to n)
        let x_invalid = array![2.0, 2.0]; // sum = 4 != 5
        let logpmf_invalid = multinomial.logpmf(&x_invalid);
        assert_eq!(logpmf_invalid, f64::NEG_INFINITY);
    }

    #[test]
    fn test_multinomial_mean() {
        let n = 10;
        let p = array![0.2, 0.3, 0.5];
        let multinomial = Multinomial::new(n, p).unwrap();

        let mean = multinomial.mean();
        let expected_mean = array![2.0, 3.0, 5.0];

        for i in 0..3 {
            assert_relative_eq!(mean[i], expected_mean[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multinomial_cov() {
        let n = 10;
        let p = array![0.2, 0.3, 0.5];
        let multinomial = Multinomial::new(n, p).unwrap();

        let cov = multinomial.cov();

        // Expected covariance matrix:
        // [n*p1*(1-p1), -n*p1*p2, -n*p1*p3]
        // [-n*p2*p1, n*p2*(1-p2), -n*p2*p3]
        // [-n*p3*p1, -n*p3*p2, n*p3*(1-p3)]

        // Diagonal elements
        assert_relative_eq!(cov[[0, 0]], 10.0 * 0.2 * 0.8, epsilon = 1e-10); // 1.6
        assert_relative_eq!(cov[[1, 1]], 10.0 * 0.3 * 0.7, epsilon = 1e-10); // 2.1
        assert_relative_eq!(cov[[2, 2]], 10.0 * 0.5 * 0.5, epsilon = 1e-10); // 2.5

        // Off-diagonal elements
        assert_relative_eq!(cov[[0, 1]], -10.0 * 0.2 * 0.3, epsilon = 1e-10); // -0.6
        assert_relative_eq!(cov[[0, 2]], -10.0 * 0.2 * 0.5, epsilon = 1e-10); // -1.0
        assert_relative_eq!(cov[[1, 2]], -10.0 * 0.3 * 0.5, epsilon = 1e-10); // -1.5

        // Symmetry
        assert_relative_eq!(cov[[1, 0]], cov[[0, 1]], epsilon = 1e-10);
        assert_relative_eq!(cov[[2, 0]], cov[[0, 2]], epsilon = 1e-10);
        assert_relative_eq!(cov[[2, 1]], cov[[1, 2]], epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_rvs() {
        let n = 100;
        let p = array![0.2, 0.3, 0.5];
        let multinomial = Multinomial::new(n, p.clone()).unwrap();

        // Generate samples
        let num_samples = 100;
        let samples = multinomial.rvs(num_samples).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), num_samples);

        // Check the dimension of each sample
        for sample in &samples {
            assert_eq!(sample.len(), 3);

            // Check that each sample sums to n
            let sum: f64 = sample.sum();
            assert_eq!(sum, n as f64);
        }

        // Calculate sample means
        let mut sample_sum = array![0.0, 0.0, 0.0];
        for sample in &samples {
            sample_sum += sample;
        }
        let sample_mean = sample_sum / num_samples as f64;

        // Expected means
        let expected_mean = array![20.0, 30.0, 50.0];

        // Check that sample means are reasonably close to expected means
        // (using larger tolerance due to random sampling)
        for i in 0..3 {
            assert!((sample_mean[i] - expected_mean[i]).abs() < 5.0);
        }
    }

    #[test]
    fn test_multinomial_rvs_single() {
        let n = 10;
        let p = array![0.2, 0.3, 0.5];
        let multinomial = Multinomial::new(n, p).unwrap();

        let sample = multinomial.rvs_single().unwrap();

        // Check the dimension of the sample
        assert_eq!(sample.len(), 3);

        // Check that the sample sums to n
        let sum: f64 = sample.sum();
        assert_eq!(sum, n as f64);
    }

    #[test]
    fn test_multinomial_coef() {
        // (5 choose 2,3) = 5! / (2! * 3!)
        let coef1 = multinomial_coef(5, &[2, 3]);
        assert_eq!(coef1, 10.0);

        // (8 choose 3,2,3) = 8! / (3! * 2! * 3!)
        let coef2 = multinomial_coef(8, &[3, 2, 3]);
        assert_eq!(coef2, 560.0);
    }
}
