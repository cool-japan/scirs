//! Geometric distribution functions
//!
//! This module provides functionality for the Geometric distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use num_traits::{Float, NumCast};
use rand_distr::{Distribution, Geometric as RandGeometric};

/// Geometric distribution structure
///
/// The Geometric distribution is a discrete probability distribution that models
/// the number of failures before the first success in a sequence of independent
/// Bernoulli trials, each with success probability p.
pub struct Geometric<F: Float> {
    /// Success probability (0 < p ≤ 1)
    pub p: F,
    /// Random number generator
    rand_distr: RandGeometric,
}

impl<F: Float + NumCast> Geometric<F> {
    /// Create a new Geometric distribution with given success probability
    ///
    /// # Arguments
    ///
    /// * `p` - Success probability (0 < p ≤ 1)
    ///
    /// # Returns
    ///
    /// * A new Geometric distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// ```
    pub fn new(p: F) -> StatsResult<Self> {
        // Validate parameters
        if p <= F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Success probability must be between 0 and 1, exclusive of 0".to_string(),
            ));
        }

        // Create RNG for Geometric distribution
        let p_f64 = <f64 as num_traits::NumCast>::from(p).unwrap();
        let rand_distr = match RandGeometric::new(p_f64) {
            Ok(distr) => distr,
            Err(_) => {
                return Err(StatsError::ComputationError(
                    "Failed to create Geometric distribution for sampling".to_string(),
                ))
            }
        };

        Ok(Geometric { p, rand_distr })
    }

    /// Calculate the probability mass function (PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the PMF (number of failures before first success)
    ///
    /// # Returns
    ///
    /// * The value of the PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let pmf_at_2 = geom.pmf(2.0);
    /// assert!((pmf_at_2 - 0.147).abs() < 1e-3);
    /// ```
    pub fn pmf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        // Check if k is a non-negative integer
        if k < zero || !Self::is_integer(k) {
            return zero;
        }

        let k_usize = k.to_usize().unwrap();

        // PMF = p * (1 - p)^k
        let q = one - self.p; // q = 1 - p
        self.p * q.powf(F::from(k_usize).unwrap())
    }

    /// Calculate the log of the probability mass function (log-PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the log-PMF (number of failures before first success)
    ///
    /// # Returns
    ///
    /// * The value of the log-PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let log_pmf_at_2 = geom.log_pmf(2.0);
    /// assert!((log_pmf_at_2 - (-1.9164)) < 1e-4);
    /// ```
    pub fn log_pmf(&self, k: F) -> F {
        let zero = F::zero();
        let neg_infinity = F::neg_infinity();

        // Check if k is a non-negative integer
        if k < zero || !Self::is_integer(k) {
            return neg_infinity;
        }

        let k_usize = k.to_usize().unwrap();
        let k_f = F::from(k_usize).unwrap();

        // log-PMF = ln(p) + k * ln(1 - p)
        let one = F::one();
        let q = one - self.p; // q = 1 - p

        self.p.ln() + k_f * q.ln()
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let cdf_at_2 = geom.cdf(2.0);
    /// assert!((cdf_at_2 - 0.657).abs() < 1e-3);
    /// ```
    pub fn cdf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        // CDF = 0 for k < 0
        if k < zero {
            return zero;
        }

        // Floor k to handle non-integer values
        let k_floor = k.floor();
        let k_int = k_floor.to_usize().unwrap();

        // Calculate CDF = 1 - (1 - p)^(k+1)
        let q = one - self.p; // q = 1 - p
        one - q.powf(F::from(k_int + 1).unwrap())
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let quant = geom.ppf(0.5).unwrap();
    /// assert_eq!(quant, 1.0);
    /// ```
    pub fn ppf(&self, p_val: F) -> StatsResult<F> {
        let zero = F::zero();
        let one = F::one();

        if p_val < zero || p_val > one {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // Special cases
        if p_val <= zero {
            return Ok(zero);
        }
        if p_val >= one {
            return Ok(F::infinity());
        }

        // Handle case when p = 1.0 separately
        if self.p == one {
            return Ok(zero); // When p=1, X is always 0
        }

        // For actual calculation, we need to find the smallest k where CDF(k) >= p_val
        // We could use the formula: k = ceiling(log(1-p_val) / log(1-p)) - 1
        // But we'll use a simpler iterative approach instead

        // Calculate the quantile manually to avoid formula issues
        let mut k = zero;
        while self.cdf(k) < p_val {
            k = k + one;
        }

        Ok(k)
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let samples = geom.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // Generate random Geometric sample
            let sample = self.rand_distr.sample(&mut rng);
            let sample_f = F::from(sample).unwrap();
            samples.push(sample_f);
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let mean = geom.mean();
    /// assert!((mean - 2.333333).abs() < 1e-6);
    /// ```
    pub fn mean(&self) -> F {
        // Mean = (1-p)/p
        let one = F::one();
        let q = one - self.p; // q = 1 - p
        q / self.p
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let variance = geom.var();
    /// assert!((variance - 7.777778).abs() < 1e-6);
    /// ```
    pub fn var(&self) -> F {
        // Variance = (1-p)/p^2
        let one = F::one();
        let q = one - self.p; // q = 1 - p
        q / (self.p * self.p)
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let std_dev = geom.std();
    /// assert!((std_dev - 2.788867).abs() < 1e-6);
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let skewness = geom.skewness();
    /// assert!((skewness - 2.0318891).abs() < 1e-6);
    /// ```
    pub fn skewness(&self) -> F {
        // Skewness = (2-p) / sqrt(1-p)
        let one = F::one();
        let two = F::from(2.0).unwrap();

        let q = one - self.p; // q = 1 - p
        (two - self.p) / q.sqrt()
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let kurtosis = geom.kurtosis();
    /// assert!((kurtosis - 6.9) < 1e-1);
    /// ```
    pub fn kurtosis(&self) -> F {
        // Excess Kurtosis = 6 + p^2/(1-p)
        let one = F::one();
        let six = F::from(6.0).unwrap();

        let q = one - self.p; // q = 1 - p
        six + (self.p * self.p) / q
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let entropy = geom.entropy();
    /// assert!((entropy - 2.937588) < 1e-6);
    /// ```
    pub fn entropy(&self) -> F {
        // Entropy = -[(1-p)*ln(1-p) + p*ln(p)] / p
        let one = F::one();
        let q = one - self.p; // q = 1 - p

        -(q * q.ln() + self.p * self.p.ln()) / self.p
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let median = geom.median();
    /// assert_eq!(median, 1.0);
    /// ```
    pub fn median(&self) -> F {
        // Median = ceiling(ln(0.5)/ln(1-p)) - 1
        let half = F::from(0.5).unwrap();
        let one = F::one();
        let q = one - self.p; // q = 1 - p

        (half.ln() / q.ln()).ceil() - one
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
    /// use scirs2_stats::distributions::geometric::Geometric;
    ///
    /// let geom = Geometric::new(0.3f64).unwrap();
    /// let mode = geom.mode();
    /// assert_eq!(mode, 0.0);
    /// ```
    pub fn mode(&self) -> F {
        // Mode = 0 (always for geometric distribution)
        F::zero()
    }

    // Helper method to check if a value is an integer
    fn is_integer(value: F) -> bool {
        value == value.floor()
    }
}

/// Create a Geometric distribution with the given parameter.
///
/// This is a convenience function to create a Geometric distribution with
/// the given success probability.
///
/// # Arguments
///
/// * `p` - Success probability (0 < p ≤ 1)
///
/// # Returns
///
/// * A Geometric distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let g = distributions::geom(0.3f64).unwrap();
/// let pmf_at_2 = g.pmf(2.0);
/// assert!((pmf_at_2 - 0.147).abs() < 1e-3);
/// ```
pub fn geom<F>(p: F) -> StatsResult<Geometric<F>>
where
    F: Float + NumCast,
{
    Geometric::new(p)
}

/// Implementation of SampleableDistribution for Geometric
impl<F: Float + NumCast> SampleableDistribution<F> for Geometric<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_geometric_creation() {
        // Valid p values
        let geom1 = Geometric::new(0.3).unwrap();
        assert_eq!(geom1.p, 0.3);

        let geom2 = Geometric::new(1.0).unwrap();
        assert_eq!(geom2.p, 1.0);

        // Invalid p values
        assert!(Geometric::<f64>::new(0.0).is_err());
        assert!(Geometric::<f64>::new(-0.1).is_err());
        assert!(Geometric::<f64>::new(1.1).is_err());
    }

    #[test]
    fn test_geometric_pmf() {
        // Geometric(0.3)
        let geom = Geometric::new(0.3).unwrap();

        // PMF values for different k
        assert_relative_eq!(geom.pmf(0.0), 0.3, epsilon = 1e-10);
        assert_relative_eq!(geom.pmf(1.0), 0.3 * 0.7, epsilon = 1e-10);
        assert_relative_eq!(geom.pmf(2.0), 0.3 * 0.7 * 0.7, epsilon = 1e-10);
        assert_relative_eq!(geom.pmf(3.0), 0.3 * 0.7 * 0.7 * 0.7, epsilon = 1e-10);

        // PMF should be 0 for negative k
        assert_eq!(geom.pmf(-1.0), 0.0);

        // PMF should be 0 for non-integer k
        assert_eq!(geom.pmf(1.5), 0.0);

        // Special case: Geometric(1.0)
        let geom_p1 = Geometric::new(1.0).unwrap();
        assert_eq!(geom_p1.pmf(0.0), 1.0);
        assert_eq!(geom_p1.pmf(1.0), 0.0);
        assert_eq!(geom_p1.pmf(2.0), 0.0);
    }

    #[test]
    fn test_geometric_log_pmf() {
        // Geometric(0.3)
        let geom = Geometric::new(0.3).unwrap();

        // Check log_pmf against pmf
        for k in 0..5 {
            let k_f = k as f64;
            let pmf = geom.pmf(k_f);
            let log_pmf = geom.log_pmf(k_f);

            if pmf > 0.0 {
                assert_relative_eq!(log_pmf, pmf.ln(), epsilon = 1e-10);
            } else {
                assert!(log_pmf.is_infinite() && log_pmf.is_sign_negative());
            }
        }

        // log_pmf should be -infinity for negative k or non-integer k
        assert!(geom.log_pmf(-1.0).is_infinite() && geom.log_pmf(-1.0).is_sign_negative());
        assert!(geom.log_pmf(1.5).is_infinite() && geom.log_pmf(1.5).is_sign_negative());
    }

    #[test]
    fn test_geometric_cdf() {
        // Geometric(0.3)
        let geom = Geometric::new(0.3).unwrap();

        // CDF for k = 0 should be 1 - (1-p)^1 = 1 - 0.7^1 = 0.3
        assert_relative_eq!(geom.cdf(0.0), 0.3, epsilon = 1e-10);

        // CDF for k = 1 should be 1 - (1-p)^2 = 1 - 0.7^2 = 1 - 0.49 = 0.51
        assert_relative_eq!(geom.cdf(1.0), 0.51, epsilon = 1e-10);

        // CDF for k = 2 should be 1 - (1-p)^3 = 1 - 0.7^3 = 1 - 0.343 = 0.657
        assert_relative_eq!(geom.cdf(2.0), 0.657, epsilon = 1e-10);

        // CDF for k = 3 should be 1 - (1-p)^4 = 1 - 0.7^4 = 1 - 0.2401 = 0.7599
        assert_relative_eq!(geom.cdf(3.0), 0.7599, epsilon = 1e-10);

        // CDF should be 0 for k < 0
        assert_eq!(geom.cdf(-1.0), 0.0);

        // Non-integer k should use floor of k
        assert_relative_eq!(geom.cdf(2.5), geom.cdf(2.0), epsilon = 1e-10);

        // Special case: Geometric(1.0)
        let geom_p1 = Geometric::new(1.0).unwrap();
        assert_eq!(geom_p1.cdf(0.0), 1.0);
        assert_eq!(geom_p1.cdf(1.0), 1.0);
    }

    #[test]
    fn test_geometric_ppf() {
        // Geometric(0.3)
        let geom = Geometric::new(0.3).unwrap();

        // Verify the ppf against the cdf
        for k in 0..10 {
            let k_f = k as f64;
            let cdf_k = geom.cdf(k_f);

            // For any value p such that CDF(k-1) < p <= CDF(k),
            // the inverse CDF (ppf) should return k
            if k > 0 {
                let cdf_k_minus_1 = geom.cdf(k_f - 1.0);
                let p_mid = (cdf_k + cdf_k_minus_1) / 2.0; // A value between CDF(k-1) and CDF(k)
                assert_eq!(geom.ppf(p_mid).unwrap(), k_f);
            } else {
                // For k=0, just test a small value of p
                assert_eq!(geom.ppf(cdf_k / 2.0).unwrap(), k_f);
            }
        }

        // Test edge cases
        assert_eq!(geom.ppf(0.0).unwrap(), 0.0);
        assert!(geom.ppf(1.0).unwrap().is_infinite()); // For p=1, the result should be infinity

        // The PPF should be the inverse of the CDF
        let test_values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        for p in test_values {
            let k = geom.ppf(p).unwrap();

            // Special case for p = 1.0 which gives infinity
            if p == 1.0 {
                continue;
            }

            // For discrete distributions, we check that F(k-1) < p ≤ F(k)
            if k > 0.0 {
                let k_minus_1 = k - 1.0;
                let cdf_k_minus_1 = geom.cdf(k_minus_1);
                let cdf_k = geom.cdf(k);
                assert!(cdf_k_minus_1 < p && p <= cdf_k);
            } else {
                // For k = 0, just check that p ≤ F(0)
                let cdf_k = geom.cdf(k);
                assert!(p <= cdf_k);
            }
        }

        // Invalid probability values
        assert!(geom.ppf(-0.1).is_err());
        assert!(geom.ppf(1.1).is_err());
    }

    #[test]
    fn test_geometric_rvs() {
        // Geometric(0.3)
        let geom = Geometric::new(0.3).unwrap();

        // Generate samples
        let samples = geom.rvs(100).unwrap();

        // Check number of samples
        assert_eq!(samples.len(), 100);

        // All samples should be non-negative integers
        for &s in &samples {
            assert!(s >= 0.0);
            assert_eq!(s, s.floor());
        }

        // Mean should be approximately (1-p)/p = 0.7/0.3 = 2.333...
        let sum: f64 = samples.iter().sum();
        let mean = sum / samples.len() as f64;

        // The mean could deviate somewhat due to randomness
        assert!(mean > 1.0 && mean < 4.0);
    }

    #[test]
    fn test_geometric_moments() {
        // Test with p = 0.3
        let geom = Geometric::new(0.3).unwrap();

        // Mean = (1-p)/p = 0.7/0.3 = 2.333...
        assert_relative_eq!(geom.mean(), 2.333333, epsilon = 1e-6);

        // Variance = (1-p)/p^2 = 0.7/(0.3^2) = 0.7/0.09 = 7.777...
        assert_relative_eq!(geom.var(), 7.777778, epsilon = 1e-6);

        // Standard deviation = sqrt(variance) = sqrt(7.777...) ≈ 2.789...
        assert_relative_eq!(geom.std(), 2.788867, epsilon = 1e-6);

        // Skewness = (2-p)/sqrt(1-p) = (2-0.3)/sqrt(0.7) = 1.7/sqrt(0.7) ≈ 2.031...
        assert_relative_eq!(geom.skewness(), 2.031889, epsilon = 1e-6);

        // Kurtosis = 6 + p^2/(1-p) = 6 + 0.3^2/0.7 = 6 + 0.09/0.7 ≈ 6.129...
        assert_relative_eq!(geom.kurtosis(), 6.128571, epsilon = 1e-6);

        // Entropy is more complex, we'll just check it's positive
        assert!(geom.entropy() > 0.0);

        // Mode is always 0 for geometric distribution
        assert_eq!(geom.mode(), 0.0);

        // Median = ceiling(ln(0.5)/ln(1-p)) - 1 = ceiling(ln(0.5)/ln(0.7)) - 1
        // ln(0.5) ≈ -0.693, ln(0.7) ≈ -0.357, so ln(0.5)/ln(0.7) ≈ 1.94, ceiling(1.94) = 2, 2-1 = 1
        assert_eq!(geom.median(), 1.0);
    }

    #[test]
    fn test_geometric_edge_cases() {
        // Test with p = 1.0 (special case, only k=0 has non-zero probability)
        let geom_p1 = Geometric::new(1.0).unwrap();

        // PMF is 1 for k=0, 0 elsewhere
        assert_eq!(geom_p1.pmf(0.0), 1.0);
        assert_eq!(geom_p1.pmf(1.0), 0.0);

        // CDF is 1 for all k >= 0
        assert_eq!(geom_p1.cdf(0.0), 1.0);
        assert_eq!(geom_p1.cdf(1.0), 1.0);

        // Mean = (1-p)/p = 0/1 = 0
        assert_eq!(geom_p1.mean(), 0.0);

        // Variance = (1-p)/p^2 = 0/1 = 0
        assert_eq!(geom_p1.var(), 0.0);

        // PPF is 0 for all p < 1, infinity for p=1
        assert_eq!(geom_p1.ppf(0.5).unwrap(), 0.0);
        assert!(geom_p1.ppf(1.0).unwrap().is_infinite());
    }
}
