//! Bernoulli distribution functions
//!
//! This module provides functionality for the Bernoulli distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use num_traits::{Float, NumCast};
use rand_distr::{Bernoulli as RandBernoulli, Distribution};
use scirs2_core::validation::check_probability;

/// Bernoulli distribution structure
///
/// The Bernoulli distribution is a discrete probability distribution taking
/// value 1 with probability p and value 0 with probability q = 1 - p.
/// It is the discrete probability distribution of a random variable which takes
/// the value 1 with probability p and the value 0 with probability q.
pub struct Bernoulli<F: Float> {
    /// Success probability p (0 ≤ p ≤ 1)
    pub p: F,
    /// Random number generator
    rand_distr: RandBernoulli,
}

impl<F: Float + NumCast + std::fmt::Display> Bernoulli<F> {
    /// Create a new Bernoulli distribution with given success probability
    ///
    /// # Arguments
    ///
    /// * `p` - Success probability (0 ≤ p ≤ 1)
    ///
    /// # Returns
    ///
    /// * A new Bernoulli distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// ```
    pub fn new(p: F) -> StatsResult<Self> {
        // Validate parameters using core validation function
        let _ = check_probability(p, "Success probability").map_err(StatsError::from)?;

        // Create RNG for Bernoulli distribution
        let p_f64 = <f64 as num_traits::NumCast>::from(p).unwrap();
        let rand_distr = match RandBernoulli::new(p_f64) {
            Ok(distr) => distr,
            Err(_) => {
                return Err(StatsError::ComputationError(
                    "Failed to create Bernoulli distribution for sampling".to_string(),
                ))
            }
        };

        Ok(Bernoulli { p, rand_distr })
    }

    /// Calculate the probability mass function (PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the PMF (0 or 1)
    ///
    /// # Returns
    ///
    /// * The value of the PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let pmf_at_one = bern.pmf(1.0);
    /// assert!((pmf_at_one - 0.3).abs() < 1e-7);
    /// ```
    pub fn pmf(&self, k: F) -> F {
        let one = F::one();
        let zero = F::zero();

        // PMF is only defined for k = 0 and k = 1
        if k == zero {
            one - self.p // q = 1 - p
        } else if k == one {
            self.p
        } else {
            zero
        }
    }

    /// Calculate the log of the probability mass function (log-PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the log-PMF (0 or 1)
    ///
    /// # Returns
    ///
    /// * The value of the log-PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let log_pmf_at_one = bern.log_pmf(1.0);
    /// assert!((log_pmf_at_one - (-1.2039728)).abs() < 1e-6);
    /// ```
    pub fn log_pmf(&self, k: F) -> F {
        let one = F::one();
        let zero = F::zero();
        let neg_infinity = F::neg_infinity();

        // log-PMF is only defined for k = 0 and k = 1
        if k == zero {
            if self.p == one {
                neg_infinity
            } else {
                (one - self.p).ln() // ln(q) = ln(1 - p)
            }
        } else if k == one {
            if self.p == zero {
                neg_infinity
            } else {
                self.p.ln() // ln(p)
            }
        } else {
            neg_infinity
        }
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// * The value of the CDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let cdf_at_zero = bern.cdf(0.0);
    /// assert!((cdf_at_zero - 0.7).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        if k < zero {
            zero
        } else if k < one {
            one - self.p // F(0) = P(X ≤ 0) = P(X = 0) = 1 - p
        } else {
            one // F(k) = P(X ≤ k) = 1 for k ≥ 1
        }
    }

    /// Inverse of the cumulative distribution function (quantile function)
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value (between 0 and 1)
    ///
    /// # Returns
    ///
    /// * The value k such that CDF(k) = p
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let quant = bern.ppf(0.8).unwrap();
    /// assert_eq!(quant, 1.0);
    /// ```
    pub fn ppf(&self, p_val: F) -> StatsResult<F> {
        // Validate probability using core validation function
        let p_val = check_probability(p_val, "Probability value").map_err(StatsError::from)?;

        let zero = F::zero();
        let one = F::one();

        // Quantile function for Bernoulli
        let q = one - self.p; // q = 1 - p

        if p_val <= q {
            Ok(zero) // Q(p) = 0 for p ≤ q
        } else {
            Ok(one) // Q(p) = 1 for p > q
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
    /// * Vector of random samples
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let samples = bern.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(size);
        let zero = F::zero();
        let one = F::one();

        for _ in 0..size {
            // Generate random Bernoulli sample (0 or 1)
            let sample = if self.rand_distr.sample(&mut rng) {
                one
            } else {
                zero
            };

            samples.push(sample);
        }

        Ok(samples)
    }

    /// Calculate the mean of the distribution
    ///
    /// # Returns
    ///
    /// * The mean of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let mean = bern.mean();
    /// assert!((mean - 0.3).abs() < 1e-7);
    /// ```
    pub fn mean(&self) -> F {
        // Mean = p
        self.p
    }

    /// Calculate the variance of the distribution
    ///
    /// # Returns
    ///
    /// * The variance of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let variance = bern.var();
    /// assert!((variance - 0.21).abs() < 1e-7);
    /// ```
    pub fn var(&self) -> F {
        // Variance = p * (1 - p)
        let one = F::one();
        self.p * (one - self.p)
    }

    /// Calculate the standard deviation of the distribution
    ///
    /// # Returns
    ///
    /// * The standard deviation of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let std_dev = bern.std();
    /// assert!((std_dev - 0.458257).abs() < 1e-6);
    /// ```
    pub fn std(&self) -> F {
        // Std = sqrt(variance)
        self.var().sqrt()
    }

    /// Calculate the skewness of the distribution
    ///
    /// # Returns
    ///
    /// * The skewness of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let skewness = bern.skewness();
    /// assert!((skewness - 0.87287156).abs() < 1e-5);
    /// ```
    pub fn skewness(&self) -> F {
        // Skewness = (1 - 2p) / sqrt(p * (1 - p))
        let one = F::from(1.0).unwrap();
        let two = F::from(2.0).unwrap();

        let q = one - self.p; // q = 1 - p

        // Handle special cases to avoid division by zero
        if self.p == F::zero() || self.p == F::one() {
            return F::zero(); // Degenerate case, skewness is not well-defined
        }

        (one - two * self.p) / (self.p * q).sqrt()
    }

    /// Calculate the kurtosis of the distribution
    ///
    /// # Returns
    ///
    /// * The excess kurtosis of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let kurtosis = bern.kurtosis();
    /// assert!((kurtosis - (-1.2351)) < 1e-4);
    /// ```
    pub fn kurtosis(&self) -> F {
        // Excess Kurtosis = (1 - 6p(1-p)) / (p(1-p))
        let one = F::from(1.0).unwrap();
        let six = F::from(6.0).unwrap();

        let q = one - self.p; // q = 1 - p
        let pq = self.p * q;

        // Handle special cases to avoid division by zero
        if self.p == F::zero() || self.p == F::one() {
            return F::zero(); // Degenerate case, kurtosis is not well-defined
        }

        (one - six * pq) / pq
    }

    /// Calculate the entropy of the distribution
    ///
    /// # Returns
    ///
    /// * The entropy value
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let entropy = bern.entropy();
    /// assert!((entropy - 0.6108643).abs() < 1e-6);
    /// ```
    pub fn entropy(&self) -> F {
        // Entropy = -p * ln(p) - (1-p) * ln(1-p)
        let zero = F::zero();
        let one = F::one();

        // Handle special cases
        if self.p == zero || self.p == one {
            return zero; // Degenerate case, entropy is 0
        }

        let q = one - self.p; // q = 1 - p

        // H(X) = -p * ln(p) - q * ln(q)
        -(self.p * self.p.ln() + q * q.ln())
    }

    /// Calculate the median of the distribution
    ///
    /// # Returns
    ///
    /// * The median of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let median = bern.median();
    /// assert_eq!(median, 0.0);
    /// ```
    pub fn median(&self) -> F {
        let zero = F::zero();
        let one = F::one();
        let half = F::from(0.5).unwrap();

        // Median is 0 if p < 0.5, 0 or 1 if p = 0.5, and 1 if p > 0.5
        if self.p < half {
            zero
        } else if self.p > half {
            one
        } else {
            // When p = 0.5, both 0 and 1 are medians
            // We return 0 by convention
            zero
        }
    }

    /// Calculate the mode of the distribution
    ///
    /// # Returns
    ///
    /// * The mode of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::bernoulli::Bernoulli;
    ///
    /// let bern = Bernoulli::new(0.3f64).unwrap();
    /// let mode = bern.mode();
    /// assert_eq!(mode, 0.0);
    /// ```
    pub fn mode(&self) -> F {
        let zero = F::zero();
        let one = F::one();
        let half = F::from(0.5).unwrap();

        // Mode is 0 if p < 0.5, 0 or 1 if p = 0.5, and 1 if p > 0.5
        if self.p < half {
            zero
        } else if self.p > half {
            one
        } else {
            // When p = 0.5, both 0 and 1 are modes
            // We return 0 by convention
            zero
        }
    }
}

/// Create a Bernoulli distribution with the given parameter.
///
/// This is a convenience function to create a Bernoulli distribution with
/// the given success probability.
///
/// # Arguments
///
/// * `p` - Success probability (0 ≤ p ≤ 1)
///
/// # Returns
///
/// * A Bernoulli distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::bernoulli;
///
/// let b = bernoulli::bernoulli(0.3f64).unwrap();
/// let pmf_at_one = b.pmf(1.0);
/// assert!((pmf_at_one - 0.3).abs() < 1e-7);
/// ```
pub fn bernoulli<F>(p: F) -> StatsResult<Bernoulli<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Bernoulli::new(p)
}

/// Implementation of SampleableDistribution for Bernoulli
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Bernoulli<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bernoulli_creation() {
        // Valid p values
        let bern1 = Bernoulli::new(0.0).unwrap();
        assert_eq!(bern1.p, 0.0);

        let bern2 = Bernoulli::new(0.5).unwrap();
        assert_eq!(bern2.p, 0.5);

        let bern3 = Bernoulli::new(1.0).unwrap();
        assert_eq!(bern3.p, 1.0);

        // Invalid p values
        assert!(Bernoulli::<f64>::new(-0.1).is_err());
        assert!(Bernoulli::<f64>::new(1.1).is_err());
    }

    #[test]
    fn test_bernoulli_pmf() {
        let bern = Bernoulli::new(0.3).unwrap();

        // PMF at k = 0
        let pmf_at_zero = bern.pmf(0.0);
        assert_relative_eq!(pmf_at_zero, 0.7, epsilon = 1e-10);

        // PMF at k = 1
        let pmf_at_one = bern.pmf(1.0);
        assert_relative_eq!(pmf_at_one, 0.3, epsilon = 1e-10);

        // PMF at other values (should be 0)
        let pmf_at_other = bern.pmf(0.5);
        assert_eq!(pmf_at_other, 0.0);

        // Corner cases
        let bern_zero = Bernoulli::new(0.0).unwrap();
        assert_eq!(bern_zero.pmf(0.0), 1.0);
        assert_eq!(bern_zero.pmf(1.0), 0.0);

        let bern_one = Bernoulli::new(1.0).unwrap();
        assert_eq!(bern_one.pmf(0.0), 0.0);
        assert_eq!(bern_one.pmf(1.0), 1.0);
    }

    #[test]
    fn test_bernoulli_log_pmf() {
        let bern = Bernoulli::new(0.3).unwrap();

        // log-PMF at k = 0
        let log_pmf_at_zero = bern.log_pmf(0.0);
        assert_relative_eq!(log_pmf_at_zero, 0.7.ln(), epsilon = 1e-10);

        // log-PMF at k = 1
        let log_pmf_at_one = bern.log_pmf(1.0);
        assert_relative_eq!(log_pmf_at_one, 0.3.ln(), epsilon = 1e-10);

        // log-PMF at other values (should be -infinity)
        let log_pmf_at_other = bern.log_pmf(0.5);
        assert!(log_pmf_at_other.is_infinite() && log_pmf_at_other.is_sign_negative());

        // Corner cases
        let bern_zero = Bernoulli::new(0.0).unwrap();
        assert_eq!(bern_zero.log_pmf(0.0), 0.0);
        assert!(bern_zero.log_pmf(1.0).is_infinite() && bern_zero.log_pmf(1.0).is_sign_negative());

        let bern_one = Bernoulli::new(1.0).unwrap();
        assert!(bern_one.log_pmf(0.0).is_infinite() && bern_one.log_pmf(0.0).is_sign_negative());
        assert_eq!(bern_one.log_pmf(1.0), 0.0);
    }

    #[test]
    fn test_bernoulli_cdf() {
        let bern = Bernoulli::new(0.3).unwrap();

        // CDF for various values
        assert_eq!(bern.cdf(-0.1), 0.0); // F(-0.1) = 0
        assert_eq!(bern.cdf(0.0), 0.7); // F(0) = P(X ≤ 0) = P(X = 0) = 1 - p = 0.7
        assert_eq!(bern.cdf(0.5), 0.7); // F(0.5) = P(X ≤ 0.5) = P(X = 0) = 1 - p = 0.7
        assert_eq!(bern.cdf(1.0), 1.0); // F(1) = P(X ≤ 1) = 1
        assert_eq!(bern.cdf(2.0), 1.0); // F(2) = P(X ≤ 2) = 1
    }

    #[test]
    fn test_bernoulli_ppf() {
        let bern = Bernoulli::new(0.3).unwrap();

        // Quantile function
        assert_eq!(bern.ppf(0.0).unwrap(), 0.0); // Q(0) = 0
        assert_eq!(bern.ppf(0.3).unwrap(), 0.0); // Q(0.3) = 0 since 0.3 ≤ q = 0.7
        assert_eq!(bern.ppf(0.7).unwrap(), 0.0); // Q(0.7) = 0 since 0.7 = q = 0.7
        assert_eq!(bern.ppf(0.71).unwrap(), 1.0); // Q(0.71) = 1 since 0.71 > q = 0.7
        assert_eq!(bern.ppf(1.0).unwrap(), 1.0); // Q(1) = 1

        // Invalid p values
        assert!(bern.ppf(-0.1).is_err());
        assert!(bern.ppf(1.1).is_err());
    }

    #[test]
    fn test_bernoulli_rvs() {
        let bern = Bernoulli::new(0.5).unwrap();

        // Generate samples
        let samples = bern.rvs(100).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 100);

        // Check all values are either 0 or 1
        for &sample in &samples {
            assert!(sample == 0.0 || sample == 1.0);
        }

        // With p = 0.5, mean should be close to 0.5 for a large sample
        let sum: f64 = samples.iter().sum();
        let mean = sum / samples.len() as f64;

        // Allow for some randomness, but mean should be roughly around 0.5
        assert!(mean > 0.3 && mean < 0.7);
    }

    #[test]
    fn test_bernoulli_stats() {
        // Test with p = 0.3
        let bern = Bernoulli::new(0.3).unwrap();

        // Mean = p = 0.3
        assert_eq!(bern.mean(), 0.3);

        // Variance = p * (1 - p) = 0.3 * 0.7 = 0.21
        assert_relative_eq!(bern.var(), 0.21, epsilon = 1e-10);

        // Standard deviation = sqrt(variance) = sqrt(0.21) ≈ 0.458258
        assert_relative_eq!(bern.std(), 0.21_f64.sqrt(), epsilon = 1e-10);

        // Skewness = (1 - 2p) / sqrt(p * (1 - p)) = (1 - 2*0.3) / sqrt(0.3 * 0.7) = 0.4 / sqrt(0.21) ≈ 0.872872
        let expected_skewness = (1.0 - 2.0 * 0.3) / (0.3 * 0.7).sqrt();
        assert_relative_eq!(bern.skewness(), expected_skewness, epsilon = 1e-5);

        // Kurtosis = (1 - 6p(1-p)) / (p(1-p)) = (1 - 6*0.3*0.7) / (0.3*0.7) = (1 - 1.26) / 0.21 ≈ -1.238
        let expected_kurtosis = (1.0 - 6.0 * 0.3 * 0.7) / (0.3 * 0.7);
        assert_relative_eq!(bern.kurtosis(), expected_kurtosis, epsilon = 1e-6);

        // Entropy = -p * ln(p) - (1-p) * ln(1-p) = -0.3 * ln(0.3) - 0.7 * ln(0.7) ≈ 0.610864
        let expected_entropy = -0.3 * 0.3.ln() - 0.7 * 0.7.ln();
        assert_relative_eq!(bern.entropy(), expected_entropy, epsilon = 1e-6);

        // Median and mode for p < 0.5 are both 0
        assert_eq!(bern.median(), 0.0);
        assert_eq!(bern.mode(), 0.0);

        // Test with p = 0.8 (> 0.5)
        let bern2 = Bernoulli::new(0.8).unwrap();

        // Median and mode for p > 0.5 are both 1
        assert_eq!(bern2.median(), 1.0);
        assert_eq!(bern2.mode(), 1.0);

        // Test with p = 0.5
        let bern3 = Bernoulli::new(0.5).unwrap();

        // Median and mode for p = 0.5 are either 0 or 1 (we return 0 by convention)
        assert_eq!(bern3.median(), 0.0);
        assert_eq!(bern3.mode(), 0.0);
    }
}
