//! Binomial distribution functions
//!
//! This module provides functionality for the Binomial distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use num_traits::{Float, NumCast};
use rand_distr::{Binomial as RandBinomial, Distribution};
use statrs::function::gamma::ln_gamma;

/// Binomial distribution structure
///
/// The Binomial distribution is a discrete probability distribution that models
/// the number of successes in a sequence of n independent trials, each with a success
/// probability p. It is a generalization of the Bernoulli distribution for n > 1.
pub struct Binomial<F: Float> {
    /// Number of trials
    pub n: usize,
    /// Success probability (0 ≤ p ≤ 1)
    pub p: F,
    /// Random number generator
    rand_distr: RandBinomial,
}

impl<F: Float + NumCast> Binomial<F> {
    /// Create a new Binomial distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `n` - Number of trials (n ≥ 0)
    /// * `p` - Success probability (0 ≤ p ≤ 1)
    ///
    /// # Returns
    ///
    /// * A new Binomial distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// ```
    pub fn new(n: usize, p: F) -> StatsResult<Self> {
        // Validate parameters
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Success probability must be between 0 and 1".to_string(),
            ));
        }

        // Create RNG for Binomial distribution
        let p_f64 = <f64 as num_traits::NumCast>::from(p).unwrap();
        let rand_distr = match RandBinomial::new(n as u64, p_f64) {
            Ok(distr) => distr,
            Err(_) => {
                return Err(StatsError::ComputationError(
                    "Failed to create Binomial distribution for sampling".to_string(),
                ))
            }
        };

        Ok(Binomial { n, p, rand_distr })
    }

    /// Calculate the probability mass function (PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the PMF (0 ≤ k ≤ n)
    ///
    /// # Returns
    ///
    /// * The value of the PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let pmf_at_5 = binom.pmf(5.0);
    /// assert!((pmf_at_5 - 0.24609375).abs() < 1e-7);
    /// ```
    pub fn pmf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        // Check if k is a non-negative integer
        if k < zero || k > F::from(self.n).unwrap() || !Self::is_integer(k) {
            return zero;
        }

        let k_usize = k.to_usize().unwrap();

        // Special case for n = 0
        if self.n == 0 {
            return if k_usize == 0 { one } else { zero };
        }

        // Special cases for p = 0 or p = 1
        if self.p == zero {
            return if k_usize == 0 { one } else { zero };
        }
        if self.p == one {
            return if k_usize == self.n { one } else { zero };
        }

        // Normal case: Calculate PMF using the binomial coefficient
        let binom_coef = self.binom_coef(k_usize);
        let p_pow_k = self.p.powf(F::from(k_usize).unwrap());
        let q_pow_nk = (one - self.p).powf(F::from(self.n - k_usize).unwrap());

        binom_coef * p_pow_k * q_pow_nk
    }

    /// Calculate the log of the probability mass function (log-PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the log-PMF (0 ≤ k ≤ n)
    ///
    /// # Returns
    ///
    /// * The value of the log-PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let log_pmf_at_5 = binom.log_pmf(5.0);
    /// assert!((log_pmf_at_5 - (-1.402)) < 1e-3);
    /// ```
    pub fn log_pmf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();
        let neg_infinity = F::neg_infinity();

        // Check if k is a non-negative integer
        if k < zero || k > F::from(self.n).unwrap() || !Self::is_integer(k) {
            return neg_infinity;
        }

        let k_usize = k.to_usize().unwrap();

        // Special case for n = 0
        if self.n == 0 {
            return if k_usize == 0 { zero } else { neg_infinity };
        }

        // Special cases for p = 0 or p = 1
        if self.p == zero {
            return if k_usize == 0 { zero } else { neg_infinity };
        }
        if self.p == one {
            return if k_usize == self.n {
                zero
            } else {
                neg_infinity
            };
        }

        // Normal case: Calculate log-PMF
        let ln_binom_coef = self.ln_binom_coef(k_usize);
        let ln_p_pow_k = F::from(k_usize).unwrap() * self.p.ln();
        let ln_q_pow_nk = F::from(self.n - k_usize).unwrap() * (one - self.p).ln();

        ln_binom_coef + ln_p_pow_k + ln_q_pow_nk
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let cdf_at_5 = binom.cdf(5.0);
    /// assert!((cdf_at_5 - 0.623046875).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        // CDF = 0 for k < 0
        if k < zero {
            return zero;
        }

        // CDF = 1 for k ≥ n
        if k >= F::from(self.n).unwrap() {
            return one;
        }

        // Floor k to handle non-integer values
        let k_floor = k.floor();
        let k_int = k_floor.to_usize().unwrap();

        // Calculate CDF as sum of PMFs from 0 to k_int
        let mut sum = zero;
        for i in 0..=k_int {
            sum = sum + self.pmf(F::from(i).unwrap());
        }

        sum
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let quant = binom.ppf(0.5).unwrap();
    /// assert_eq!(quant, 5.0);
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
        if p_val == zero {
            return Ok(zero);
        }
        if p_val == one {
            return Ok(F::from(self.n).unwrap());
        }

        // Binary search to find the quantile
        let mut low = 0;
        let mut high = self.n;
        let mut mid;

        // Maximum number of iterations to avoid infinite loops
        let max_iter = 100;
        let mut iter = 0;

        while low < high && iter < max_iter {
            mid = (low + high) / 2;
            let mid_f = F::from(mid).unwrap();
            let cdf_mid = self.cdf(mid_f);

            if cdf_mid < p_val {
                low = mid + 1;
            } else {
                high = mid;
            }

            iter += 1;
        }

        // Return the quantile
        Ok(F::from(low).unwrap())
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let samples = binom.rvs(5).unwrap();
    /// assert_eq!(samples.len(), 5);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // Generate random Binomial sample
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.3f64).unwrap();
    /// let mean = binom.mean();
    /// assert!((mean - 3.0).abs() < 1e-7);
    /// ```
    pub fn mean(&self) -> F {
        // Mean = n * p
        F::from(self.n).unwrap() * self.p
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.3f64).unwrap();
    /// let variance = binom.var();
    /// assert!((variance - 2.1).abs() < 1e-7);
    /// ```
    pub fn var(&self) -> F {
        // Variance = n * p * (1 - p)
        let one = F::one();
        F::from(self.n).unwrap() * self.p * (one - self.p)
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.3f64).unwrap();
    /// let std_dev = binom.std();
    /// assert!((std_dev - 1.449138).abs() < 1e-6);
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.3f64).unwrap();
    /// let skewness = binom.skewness();
    /// assert!((skewness - 0.277350) < 1e-6);
    /// ```
    pub fn skewness(&self) -> F {
        // Skewness = (1 - 2p) / sqrt(n * p * (1 - p))
        let one = F::one();
        let two = F::from(2.0).unwrap();

        let n_f = F::from(self.n).unwrap();
        let q = one - self.p; // q = 1 - p

        // Handle special cases
        if self.p == F::zero() || self.p == F::one() || self.n == 0 {
            return F::zero(); // Undefined case
        }

        (one - two * self.p) / (n_f * self.p * q).sqrt()
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
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.3f64).unwrap();
    /// let kurtosis = binom.kurtosis();
    /// assert!((kurtosis - (-0.12380952)).abs() < 1e-6);
    /// ```
    pub fn kurtosis(&self) -> F {
        // Excess Kurtosis = (1 - 6p(1-p)) / (np(1-p))
        let one = F::one();
        let six = F::from(6.0).unwrap();

        let n_f = F::from(self.n).unwrap();
        let q = one - self.p; // q = 1 - p
        let pq = self.p * q;

        // Handle special cases
        if pq == F::zero() || self.n == 0 {
            return F::zero(); // Undefined case
        }

        (one - six * pq) / (n_f * pq)
    }

    /// Calculate the entropy of the distribution
    ///
    /// # Returns
    ///
    /// * Approximate entropy value using normal approximation
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let entropy = binom.entropy();
    /// assert!(entropy > 0.0);
    /// ```
    pub fn entropy(&self) -> F {
        // For large n, we can approximate the entropy using the normal distribution
        // This is not exact but provides a reasonable estimate
        let half = F::from(0.5).unwrap();
        let two_pi_e = F::from(2.0 * std::f64::consts::PI * std::f64::consts::E).unwrap();

        // Handle special cases
        if self.n == 0 || self.p == F::zero() || self.p == F::one() {
            return F::zero();
        }

        // Use normal approximation: H(X) ≈ 0.5 * ln(2πe * n * p * (1-p))
        let variance = self.var();
        half * (two_pi_e * variance).ln()
    }

    /// Calculate the mode of the distribution
    ///
    /// # Returns
    ///
    /// * The mode(s) of the distribution as a vector (may have one or two values)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let modes = binom.mode();
    /// assert_eq!(modes, vec![5.0]);
    /// ```
    pub fn mode(&self) -> Vec<F> {
        let n_plus_1 = F::from(self.n + 1).unwrap();

        // Mode calculation: floor((n+1)*p) and/or ceil((n+1)*p) - 1
        let lower_mode = ((n_plus_1) * self.p).floor();
        let upper_mode = ((n_plus_1) * self.p).ceil() - F::one();

        // Return both modes if they are different
        if lower_mode == upper_mode {
            vec![lower_mode]
        } else {
            vec![lower_mode, upper_mode]
        }
    }

    /// Calculate the median of the distribution
    ///
    /// # Returns
    ///
    /// * The median of the distribution (approximated as floor(mean) or ceil(mean))
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::binomial::Binomial;
    ///
    /// let binom = Binomial::new(10, 0.5f64).unwrap();
    /// let median = binom.median();
    /// assert_eq!(median, 5.0);
    /// ```
    pub fn median(&self) -> F {
        let mean = self.mean();

        // Approximate the median as the nearest integer to the mean
        // This is not exact for all cases but works well for most
        if mean - mean.floor() < F::from(0.5).unwrap() {
            mean.floor()
        } else {
            mean.ceil()
        }
    }

    // Helper method to check if a value is an integer
    fn is_integer(value: F) -> bool {
        value == value.floor()
    }

    // Helper method to calculate binomial coefficient n choose k
    fn binom_coef(&self, k: usize) -> F {
        // Calculate directly for small values
        if self.n <= 20 {
            let mut result: u64 = 1;
            for i in 0..k {
                result = result * (self.n - i) as u64 / (i + 1) as u64;
            }
            F::from(result).unwrap()
        } else {
            // For larger values, use ln_gamma for numerical stability
            self.ln_binom_coef(k).exp()
        }
    }

    // Helper method to calculate log of binomial coefficient ln(n choose k)
    fn ln_binom_coef(&self, k: usize) -> F {
        let n_f64 = self.n as f64;
        let k_f64 = k as f64;

        // ln(n choose k) = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1)
        let result = ln_gamma(n_f64 + 1.0) - ln_gamma(k_f64 + 1.0) - ln_gamma(n_f64 - k_f64 + 1.0);

        F::from(result).unwrap()
    }
}

/// Create a Binomial distribution with the given parameters.
///
/// This is a convenience function to create a Binomial distribution with
/// the given number of trials and success probability.
///
/// # Arguments
///
/// * `n` - Number of trials (n ≥ 0)
/// * `p` - Success probability (0 ≤ p ≤ 1)
///
/// # Returns
///
/// * A Binomial distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let b = distributions::binom(10, 0.5f64).unwrap();
/// let pmf_at_5 = b.pmf(5.0);
/// assert!((pmf_at_5 - 0.24609375).abs() < 1e-7);
/// ```
pub fn binom<F>(n: usize, p: F) -> StatsResult<Binomial<F>>
where
    F: Float + NumCast,
{
    Binomial::new(n, p)
}

/// Implementation of SampleableDistribution for Binomial
impl<F: Float + NumCast> SampleableDistribution<F> for Binomial<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binomial_creation() {
        // Valid parameters
        let binom1 = Binomial::new(10, 0.3).unwrap();
        assert_eq!(binom1.n, 10);
        assert_eq!(binom1.p, 0.3);

        let binom2 = Binomial::new(0, 0.5).unwrap();
        assert_eq!(binom2.n, 0);
        assert_eq!(binom2.p, 0.5);

        // Invalid p values
        assert!(Binomial::<f64>::new(10, -0.1).is_err());
        assert!(Binomial::<f64>::new(10, 1.1).is_err());
    }

    #[test]
    fn test_binomial_pmf() {
        // Binomial(10, 0.5)
        let binom = Binomial::new(10, 0.5).unwrap();

        // PMF values for different k
        assert_relative_eq!(binom.pmf(0.0), 0.0009765625, epsilon = 1e-10);
        assert_relative_eq!(binom.pmf(3.0), 0.1171875, epsilon = 1e-10);
        assert_relative_eq!(binom.pmf(5.0), 0.24609375, epsilon = 1e-10);
        assert_relative_eq!(binom.pmf(10.0), 0.0009765625, epsilon = 1e-10);

        // PMF should be 0 outside the domain
        assert_eq!(binom.pmf(-1.0), 0.0);
        assert_eq!(binom.pmf(11.0), 0.0);

        // PMF should be 0 for non-integer k
        assert_eq!(binom.pmf(3.5), 0.0);

        // Special case: Binomial(0, 0.5)
        let binom_zero = Binomial::new(0, 0.5).unwrap();
        assert_eq!(binom_zero.pmf(0.0), 1.0);
        assert_eq!(binom_zero.pmf(1.0), 0.0);

        // Special case: Binomial(n, 0)
        let binom_p0 = Binomial::new(10, 0.0).unwrap();
        assert_eq!(binom_p0.pmf(0.0), 1.0);
        assert_eq!(binom_p0.pmf(1.0), 0.0);

        // Special case: Binomial(n, 1)
        let binom_p1 = Binomial::new(10, 1.0).unwrap();
        assert_eq!(binom_p1.pmf(10.0), 1.0);
        assert_eq!(binom_p1.pmf(9.0), 0.0);
    }

    #[test]
    fn test_binomial_log_pmf() {
        // Binomial(10, 0.5)
        let binom = Binomial::new(10, 0.5).unwrap();

        // Check log_pmf against pmf
        for k in 0..=10 {
            let k_f = k as f64;
            let pmf = binom.pmf(k_f);
            let log_pmf = binom.log_pmf(k_f);

            if pmf > 0.0 {
                assert_relative_eq!(log_pmf, pmf.ln(), epsilon = 1e-10);
            } else {
                assert!(log_pmf.is_infinite() && log_pmf.is_sign_negative());
            }
        }

        // log_pmf should be -infinity outside the domain
        assert!(binom.log_pmf(-1.0).is_infinite() && binom.log_pmf(-1.0).is_sign_negative());
        assert!(binom.log_pmf(11.0).is_infinite() && binom.log_pmf(11.0).is_sign_negative());

        // log_pmf should be -infinity for non-integer k
        assert!(binom.log_pmf(3.5).is_infinite() && binom.log_pmf(3.5).is_sign_negative());
    }

    #[test]
    fn test_binomial_cdf() {
        // Binomial(10, 0.5)
        let binom = Binomial::new(10, 0.5).unwrap();

        // CDF values for different k
        assert_relative_eq!(binom.cdf(-1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(binom.cdf(0.0), 0.0009765625, epsilon = 1e-10);
        assert_relative_eq!(binom.cdf(3.0), 0.171875, epsilon = 1e-10);
        assert_relative_eq!(binom.cdf(5.0), 0.623046875, epsilon = 1e-10);
        assert_relative_eq!(binom.cdf(10.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(binom.cdf(11.0), 1.0, epsilon = 1e-10);

        // Non-integer k should be handled correctly (floor of k is used)
        assert_relative_eq!(binom.cdf(3.7), binom.cdf(3.0), epsilon = 1e-10);
    }

    #[test]
    fn test_binomial_ppf() {
        // Binomial(10, 0.5)
        let binom = Binomial::new(10, 0.5).unwrap();

        // PPF should be the inverse of CDF
        for k in 0..=10 {
            let k_f = k as f64;
            let cdf = binom.cdf(k_f);
            let ppf = binom.ppf(cdf).unwrap();

            // Tolerance - PPF may give the smallest k such that CDF(k) >= p
            assert!(ppf <= k_f + 1.0);
        }

        // Specific values
        assert_eq!(binom.ppf(0.0).unwrap(), 0.0);
        assert_eq!(binom.ppf(1.0).unwrap(), 10.0);

        // PPF should be 5 for p = 0.5
        assert_eq!(binom.ppf(0.5).unwrap(), 5.0);

        // Invalid probability values
        assert!(binom.ppf(-0.1).is_err());
        assert!(binom.ppf(1.1).is_err());
    }

    #[test]
    fn test_binomial_rvs() {
        // Binomial(10, 0.5)
        let binom = Binomial::new(10, 0.5).unwrap();

        // Generate samples
        let samples = binom.rvs(100).unwrap();

        // Check number of samples
        assert_eq!(samples.len(), 100);

        // All samples should be integers between 0 and 10
        for &s in &samples {
            assert!(s >= 0.0 && s <= 10.0);
            assert_eq!(s, s.floor());
        }

        // Mean should be approximately n*p = 10*0.5 = 5
        let sum: f64 = samples.iter().sum();
        let mean = sum / samples.len() as f64;

        // The mean could deviate somewhat due to randomness
        assert!(mean > 4.0 && mean < 6.0);
    }

    #[test]
    fn test_binomial_moments() {
        // Test with n = 10, p = 0.3
        let binom = Binomial::new(10, 0.3).unwrap();

        // Mean = n * p = 10 * 0.3 = 3
        assert_eq!(binom.mean(), 3.0);

        // Variance = n * p * (1 - p) = 10 * 0.3 * 0.7 = 2.1
        assert_relative_eq!(binom.var(), 2.1, epsilon = 1e-10);

        // Standard deviation = sqrt(variance) = sqrt(2.1) ≈ 1.449138
        assert_relative_eq!(binom.std(), 2.1_f64.sqrt(), epsilon = 1e-10);

        // Skewness = (1 - 2p) / sqrt(n * p * (1 - p)) = (1 - 2*0.3) / sqrt(10 * 0.3 * 0.7) ≈ 0.277350
        let expected_skewness = (1.0 - 2.0 * 0.3) / (10.0 * 0.3 * 0.7).sqrt();
        assert_relative_eq!(binom.skewness(), expected_skewness, epsilon = 1e-6);

        // Kurtosis = (1 - 6p(1-p)) / (np(1-p)) = (1 - 6*0.3*0.7) / (10*0.3*0.7) ≈ -0.133333
        let expected_kurtosis = (1.0 - 6.0 * 0.3 * 0.7) / (10.0 * 0.3 * 0.7);
        assert_relative_eq!(binom.kurtosis(), expected_kurtosis, epsilon = 1e-6);
    }

    #[test]
    fn test_binomial_mode() {
        // Binomial(10, 0.3) - mode = floor((n+1)*p) = floor(11*0.3) = 3
        let binom1 = Binomial::new(10, 0.3).unwrap();
        let modes1 = binom1.mode();
        assert_eq!(modes1, vec![3.0]);

        // Binomial(10, 0.5) - mode = floor((n+1)*p) = floor(11*0.5) = 5
        let binom2 = Binomial::new(10, 0.5).unwrap();
        let modes2 = binom2.mode();
        assert_eq!(modes2, vec![5.0]);

        // Edge case: Binomial(9, 0.4) - modes = {floor(10*0.4), ceil(10*0.4)-1} = {4, 3}
        // For bimodal case, both modes should be returned
        let binom3 = Binomial::new(9, 0.4).unwrap();
        let modes3 = binom3.mode();
        assert!(modes3.contains(&3.0) && modes3.contains(&4.0) && modes3.len() == 2);
    }

    #[test]
    fn test_binomial_median() {
        // Binomial(10, 0.5) - median should be equal to mean = 5
        let binom1 = Binomial::new(10, 0.5).unwrap();
        assert_eq!(binom1.median(), 5.0);

        // Binomial(10, 0.3) - median is approximately equal to mean = 3
        let binom2 = Binomial::new(10, 0.3).unwrap();
        assert_eq!(binom2.median(), 3.0);

        // Binomial(9, 0.4) - median is approximately equal to mean = 3.6 → 4
        let binom3 = Binomial::new(9, 0.4).unwrap();
        assert_eq!(binom3.median(), 4.0);
    }

    #[test]
    fn test_binomial_coef() {
        let binom = Binomial::new(10, 0.5).unwrap();

        // Check a few known values
        assert_eq!(binom.binom_coef(0), 1.0);
        assert_eq!(binom.binom_coef(1), 10.0);
        assert_eq!(binom.binom_coef(2), 45.0);
        assert_eq!(binom.binom_coef(5), 252.0);
        assert_eq!(binom.binom_coef(10), 1.0);

        // Test with larger n
        let binom_large = Binomial::new(100, 0.5).unwrap();

        // C(100, 0) = 1
        assert_relative_eq!(binom_large.binom_coef(0), 1.0, epsilon = 1e-8);

        // C(100, 1) = 100
        assert_relative_eq!(binom_large.binom_coef(1), 100.0, epsilon = 1e-8);

        // C(100, 2) = 4950
        assert_relative_eq!(binom_large.binom_coef(2), 4950.0, epsilon = 1e-8);
    }
}
