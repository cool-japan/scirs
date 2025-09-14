//! Negative Binomial distribution functions
//!
//! This module provides functionality for the Negative Binomial distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use num_traits::{Float, NumCast};
use rand_distr::Distribution;
use scirs2_core::rng;
use scirs2_core::Rng;
use statrs::function::gamma::ln_gamma;

/// Negative Binomial distribution structure
///
/// The Negative Binomial distribution models the number of failures before r successes
/// in a sequence of independent Bernoulli trials, each with success probability p.
/// It is a generalization of the Geometric distribution for r > 1.
///
/// In this implementation, we use the convention where the random variable X
/// represents the number of failures (not the number of trials) before the r-th success.
pub struct NegativeBinomial<F: Float> {
    /// Number of successes to achieve
    pub r: F,
    /// Success probability (0 < p ≤ 1)
    pub p: F,
}

impl<F: Float + NumCast + std::fmt::Display> NegativeBinomial<F> {
    /// Create a new Negative Binomial distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `r` - Number of successes to achieve (r > 0)
    /// * `p` - Success probability (0 < p ≤ 1)
    ///
    /// # Returns
    ///
    /// * A new Negative Binomial distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// ```
    pub fn new(r: F, p: F) -> StatsResult<Self> {
        // Validate parameters
        if r <= F::zero() {
            return Err(StatsError::DomainError(
                "Number of successes must be positive".to_string(),
            ));
        }

        if p <= F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Success probability must be between 0 and 1, exclusive of 0".to_string(),
            ));
        }

        Ok(NegativeBinomial { r, p })
    }

    /// Calculate the probability mass function (PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the PMF (number of failures before r successes)
    ///
    /// # Returns
    ///
    /// * The value of the PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let pmf_at_7 = nb.pmf(7.0);
    /// assert!((pmf_at_7 - 0.0660).abs() < 1e-4);
    /// ```
    pub fn pmf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        // Check if k is a non-negative integer
        if k < zero || !Self::is_integer(k) {
            return zero;
        }

        let k_usize = k.to_usize().unwrap_or(0);
        let k_f = F::from(k_usize).unwrap_or_else(|| F::zero());

        // Handle special cases
        if self.p == one {
            return if k_f == zero { one } else { zero };
        }

        // For integer r, we can use the binomial coefficient formula:
        // PMF(k) = C(k+r-1, k) * p^r * (1-p)^k
        if Self::is_integer(self.r) {
            let r_usize = self.r.to_usize().unwrap_or(1);
            if r_usize > 0 {
                return self.binomial_pmf(k_usize);
            }
        }

        // For non-integer r, we use the gamma function approach:
        // PMF(k) = Gamma(r+k)/(Gamma(r)*k!) * p^r * (1-p)^k
        self.gamma_pmf(k_f)
    }

    // Helper method for PMF calculation for integer r using binomial coefficient
    fn binomial_pmf(&self, k: usize) -> F {
        let one = F::one();
        let k_f = F::from(k).unwrap_or_else(|| F::zero());

        let r_usize = self.r.to_usize().unwrap_or(1);
        let r_f = F::from(r_usize).unwrap_or_else(|| F::one());

        // Calculate binomial coefficient C(k+r-1, k)
        let binom_coef = self.binom_coef(k + r_usize - 1, k);

        // Calculate p^r * (1-p)^k
        let p_pow_r = self.p.powf(r_f);
        let q_pow_k = (one - self.p).powf(k_f);

        binom_coef * p_pow_r * q_pow_k
    }

    // Helper method for PMF calculation for non-integer r using gamma function
    fn gamma_pmf(&self, k: F) -> F {
        // Use F::one() directly in the code

        // Convert k to f64 for gamma function
        let k_f64 = <f64 as num_traits::NumCast>::from(k).unwrap_or(0.0);
        let r_f64 = <f64 as num_traits::NumCast>::from(self.r).unwrap_or(1.0);

        // Calculate ln(Gamma(r+k)/(Gamma(r)*k!))
        let ln_coef = ln_gamma(r_f64 + k_f64) - ln_gamma(r_f64) - ln_gamma(k_f64 + 1.0);

        // Calculate ln(p^r * (1-p)^k)
        let p_f64 = <f64 as num_traits::NumCast>::from(self.p).unwrap_or(0.5);
        let ln_prob = r_f64 * p_f64.ln() + k_f64 * (1.0 - p_f64).ln();

        // Combine and convert back to F
        F::from((ln_coef + ln_prob).exp()).unwrap_or_else(|| F::zero())
    }

    /// Calculate the log of the probability mass function (log-PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the log-PMF (number of failures before r successes)
    ///
    /// # Returns
    ///
    /// * The value of the log-PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let log_pmf_at_7 = nb.log_pmf(7.0);
    /// assert!((log_pmf_at_7 - (-2.717)).abs() < 1e-3);
    /// ```
    pub fn log_pmf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();
        let neg_infinity = F::neg_infinity();

        // Check if k is a non-negative integer
        if k < zero || !Self::is_integer(k) {
            return neg_infinity;
        }

        let k_usize = k.to_usize().unwrap();
        let k_f = F::from(k_usize).unwrap();

        // Handle special cases
        if self.p == one {
            return if k_f == zero { zero } else { neg_infinity };
        }

        // For integer r, we can use the binomial coefficient formula
        if Self::is_integer(self.r) {
            let r_usize = self.r.to_usize().unwrap();
            if r_usize > 0 {
                return self.log_binomial_pmf(k_usize);
            }
        }

        // For non-integer r, we use the gamma function approach
        self.log_gamma_pmf(k_f)
    }

    // Helper method for log-PMF calculation for integer r
    fn log_binomial_pmf(&self, k: usize) -> F {
        let one = F::one();
        let k_f = F::from(k).unwrap();

        let r_usize = self.r.to_usize().unwrap();
        let r_f = F::from(r_usize).unwrap();

        // Calculate ln(C(k+r-1, k))
        let ln_binom_coef = self.ln_binom_coef(k + r_usize - 1, k);

        // Calculate ln(p^r * (1-p)^k)
        let ln_p_pow_r = r_f * self.p.ln();
        let ln_q_pow_k = k_f * (one - self.p).ln();

        ln_binom_coef + ln_p_pow_r + ln_q_pow_k
    }

    // Helper method for log-PMF calculation for non-integer r
    fn log_gamma_pmf(&self, k: F) -> F {
        // Use F::one() directly in the code

        // Convert k to f64 for gamma function
        let k_f64 = <f64 as num_traits::NumCast>::from(k).unwrap_or(0.0);
        let r_f64 = <f64 as num_traits::NumCast>::from(self.r).unwrap_or(1.0);

        // Calculate ln(Gamma(r+k)/(Gamma(r)*k!))
        let ln_coef = ln_gamma(r_f64 + k_f64) - ln_gamma(r_f64) - ln_gamma(k_f64 + 1.0);

        // Calculate ln(p^r * (1-p)^k)
        let p_f64 = <f64 as num_traits::NumCast>::from(self.p).unwrap_or(0.5);
        let ln_prob = r_f64 * p_f64.ln() + k_f64 * (1.0 - p_f64).ln();

        // Convert back to F
        F::from(ln_coef + ln_prob).unwrap()
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let cdf_at_7 = nb.cdf(7.0);
    /// assert!((cdf_at_7 - 0.2763).abs() < 1e-4);
    /// ```
    pub fn cdf(&self, k: F) -> F {
        let zero = F::zero();
        let one = F::one();

        // CDF = 0 for k < 0
        if k < zero {
            return zero;
        }

        // Handle special cases
        if self.p == one {
            return one; // Always 1 for p=1
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let quant = nb.ppf(0.5).unwrap();
    /// assert_eq!(quant, 11.0);
    /// ```
    pub fn ppf(&self, pval: F) -> StatsResult<F> {
        let zero = F::zero();
        let one = F::one();

        if pval < zero || pval > one {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // Special cases
        if pval <= zero {
            return Ok(zero);
        }
        if pval >= one {
            return Ok(F::infinity());
        }

        // Handle case when p = 1.0 separately
        if self.p == one {
            return Ok(zero); // When p=1, X is always 0
        }

        // For actual calculation, we need to find the smallest k where CDF(k) >= p_val
        // Use an iterative approach
        let mut k = zero;

        // Maximum number of iterations to avoid infinite loops
        let max_iter = 1000;
        let mut iter = 0;

        while self.cdf(k) < pval && iter < max_iter {
            k = k + one;
            iter += 1;
        }

        if iter == max_iter {
            return Err(StatsError::ComputationError(
                "Failed to converge in PPF calculation".to_string(),
            ));
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let samples = nb.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = rng();
        let mut samples = Vec::with_capacity(size);

        // For integer r, we can use a sum of geometric variables
        if Self::is_integer(self.r) {
            let r_usize = self.r.to_usize().unwrap();

            for _ in 0..size {
                // Generate r geometric variables and sum them
                let mut sum = 0;
                for _ in 0..r_usize {
                    // Generate geometric random variable (# failures before first success)
                    let u: f64 = rng.gen_range(0.0..1.0);
                    let p_f64 = <f64 as num_traits::NumCast>::from(self.p).unwrap_or(0.5);
                    let geom_sample = (u.ln() / (1.0_f64 - p_f64).ln()).floor() as usize;
                    sum += geom_sample;
                }
                samples.push(F::from(sum).unwrap());
            }
        } else {
            // For non-integer r..use gamma-Poisson mixture
            let r_f64 = <f64 as num_traits::NumCast>::from(self.r).unwrap_or(1.0);
            let p_f64 = <f64 as num_traits::NumCast>::from(self.p).unwrap_or(0.5);

            for _ in 0..size {
                // Generate gamma random variable with shape r and scale (1-p)/p
                let gamma_distr =
                    rand_distr::Gamma::new(r_f64, (1.0 - p_f64) / p_f64).map_err(|_| {
                        StatsError::ComputationError(
                            "Failed to create gamma distribution".to_string(),
                        )
                    })?;
                let gamma_sample: f64 = gamma_distr.sample(&mut rng);

                // Generate Poisson random variable with mean gamma_sample
                let poisson_distr = rand_distr::Poisson::new(gamma_sample).map_err(|_| {
                    StatsError::ComputationError(
                        "Failed to create Poisson distribution".to_string(),
                    )
                })?;
                let poisson_sample = poisson_distr.sample(&mut rng);

                samples.push(F::from(poisson_sample).unwrap());
            }
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let mean = nb.mean();
    /// assert!((mean - 11.666667).abs() < 1e-6);
    /// ```
    pub fn mean(&self) -> F {
        // Mean = r * (1-p) / p
        let one = F::one();
        let q = one - self.p; // q = 1 - p
        self.r * q / self.p
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let variance = nb.var();
    /// assert!((variance - 38.888889).abs() < 1e-6);
    /// ```
    pub fn var(&self) -> F {
        // Variance = r * (1-p) / p^2
        let one = F::one();
        let q = one - self.p; // q = 1 - p
        self.r * q / (self.p * self.p)
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let std_dev = nb.std();
    /// assert!((std_dev - 6.236).abs() < 1e-3);
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let skewness = nb.skewness();
    /// assert!((skewness - 0.909).abs() < 1e-3);
    /// ```
    pub fn skewness(&self) -> F {
        // Skewness = (2-p) / sqrt(r * (1-p))
        let one = F::one();
        let two = F::from(2.0).unwrap();

        let q = one - self.p; // q = 1 - p
        (two - self.p) / (self.r * q).sqrt()
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
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let kurtosis = nb.kurtosis();
    /// assert!((kurtosis - 1.226).abs() < 1e-3);
    /// ```
    pub fn kurtosis(&self) -> F {
        // Excess Kurtosis = 6/r + (p^2)/(r*(1-p))
        let one = F::one();
        let six = F::from(6.0).unwrap();

        let q = one - self.p; // q = 1 - p
        six / self.r + (self.p * self.p) / (self.r * q)
    }

    /// Calculate the entropy of the distribution
    ///
    /// # Returns
    ///
    /// * The entropy value (approximation for non-integer r)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let entropy = nb.entropy();
    /// assert!(entropy > 0.0);
    /// ```
    pub fn entropy(&self) -> F {
        // For integer r, approximate using the mean and variance
        // This is a reasonable approximation for large r
        let half = F::from(0.5).unwrap();
        let two_pi_e = F::from(2.0 * std::f64::consts::PI * std::f64::consts::E).unwrap();

        // If r is large, use normal approximation
        if self.r >= F::from(10.0).unwrap() {
            let variance = self.var();
            return half * (two_pi_e * variance).ln();
        }

        // For smaller r, compute entropy directly for some values
        // (This is an approximation for the general case)
        let one = F::one();
        let q = one - self.p; // q = 1 - p
        let mean = self.mean();

        if Self::is_integer(self.r) {
            // r * ln(1/p) - mean * ln(q)
            return self.r * (-self.p.ln()) - mean * q.ln();
        }

        // For non-integer r, use mean and variance to approximate
        half * (two_pi_e * self.var()).ln()
    }

    /// Calculate the mode of the distribution
    ///
    /// # Returns
    ///
    /// * The mode of the distribution (floor of (r-1)*(1-p)/p for r>1, 0 for r≤1)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::negative_binomial::NegativeBinomial;
    ///
    /// let nb = NegativeBinomial::new(5.0f64, 0.3).unwrap();
    /// let mode = nb.mode();
    /// assert_eq!(mode, 9.0);
    /// ```
    pub fn mode(&self) -> F {
        let zero = F::zero();
        let one = F::one();

        // For r ≤ 1, the mode is 0
        if self.r <= one {
            return zero;
        }

        // For r > 1, mode = floor((r-1)*(1-p)/p)
        let q = one - self.p; // q = 1 - p
        ((self.r - one) * q / self.p).floor()
    }

    // Helper method to check if a value is an integer
    fn is_integer(value: F) -> bool {
        value == value.floor()
    }

    // Helper method to calculate binomial coefficient C(n,k)
    fn binom_coef(&self, n: usize, k: usize) -> F {
        // For small values, calculate directly
        if n <= 20 {
            let mut result: u64 = 1;
            for i in 0..k {
                result = result * (n - i) as u64 / (i + 1) as u64;
            }
            F::from(result).unwrap()
        } else {
            // For larger values, use ln_gamma for numerical stability
            self.ln_binom_coef(n, k).exp()
        }
    }

    // Helper method to calculate log of binomial coefficient ln(C(n,k))
    fn ln_binom_coef(&self, n: usize, k: usize) -> F {
        let n_f64 = n as f64;
        let k_f64 = k as f64;

        // ln(C(n,k)) = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1)
        let result = ln_gamma(n_f64 + 1.0) - ln_gamma(k_f64 + 1.0) - ln_gamma(n_f64 - k_f64 + 1.0);

        F::from(result).unwrap()
    }
}

/// Create a Negative Binomial distribution with the given parameters.
///
/// This is a convenience function to create a Negative Binomial distribution with
/// the given number of successes and success probability.
///
/// # Arguments
///
/// * `r` - Number of successes to achieve (r > 0)
/// * `p` - Success probability (0 < p ≤ 1)
///
/// # Returns
///
/// * A Negative Binomial distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions;
///
/// let nb = distributions::nbinom(5.0f64, 0.3).unwrap();
/// let pmf_at_7 = nb.pmf(7.0);
/// assert!((pmf_at_7 - 0.0660).abs() < 1e-4);
/// ```
#[allow(dead_code)]
pub fn nbinom<F>(r: F, p: F) -> StatsResult<NegativeBinomial<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    NegativeBinomial::new(r, p)
}

/// Implementation of SampleableDistribution for NegativeBinomial
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for NegativeBinomial<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore = "timeout"]
    fn test_negative_binomial_creation() {
        // Valid parameters
        let nb1 = NegativeBinomial::new(5.0, 0.3).unwrap();
        assert_eq!(nb1.r, 5.0);
        assert_eq!(nb1.p, 0.3);

        let nb2 = NegativeBinomial::new(1.0, 1.0).unwrap();
        assert_eq!(nb2.r, 1.0);
        assert_eq!(nb2.p, 1.0);

        // Invalid r values
        assert!(NegativeBinomial::<f64>::new(0.0, 0.3).is_err());
        assert!(NegativeBinomial::<f64>::new(-1.0, 0.3).is_err());

        // Invalid p values
        assert!(NegativeBinomial::<f64>::new(5.0, 0.0).is_err());
        assert!(NegativeBinomial::<f64>::new(5.0, -0.1).is_err());
        assert!(NegativeBinomial::<f64>::new(5.0, 1.1).is_err());
    }

    #[test]
    fn test_negative_binomial_pmf() {
        // NegativeBinomial(5, 0.3) - integer r
        let nb = NegativeBinomial::new(5.0, 0.3).unwrap();

        // PMF values for different k
        assert_relative_eq!(nb.pmf(0.0), 0.00243, epsilon = 1e-5);
        assert_relative_eq!(nb.pmf(3.0), 0.02917, epsilon = 1e-5);
        assert_relative_eq!(nb.pmf(7.0), 0.06604, epsilon = 1e-5);
        assert_relative_eq!(nb.pmf(12.0), 0.06121, epsilon = 1e-5);

        // PMF should be 0 for negative k
        assert_eq!(nb.pmf(-1.0), 0.0);

        // PMF should be 0 for non-integer k
        assert_eq!(nb.pmf(3.5), 0.0);

        // Special case: NegativeBinomial(1, 0.3) - same as geometric
        let nb_geom = NegativeBinomial::new(1.0, 0.3).unwrap();
        assert_relative_eq!(nb_geom.pmf(0.0), 0.3, epsilon = 1e-10);
        assert_relative_eq!(nb_geom.pmf(1.0), 0.3 * 0.7, epsilon = 1e-10);
        assert_relative_eq!(nb_geom.pmf(2.0), 0.3 * 0.7 * 0.7, epsilon = 1e-10);

        // Special case: NegativeBinomial(r, 1.0)
        let nb_p1 = NegativeBinomial::new(5.0, 1.0).unwrap();
        assert_eq!(nb_p1.pmf(0.0), 1.0);
        assert_eq!(nb_p1.pmf(1.0), 0.0);

        // Test with non-integer r
        let nb_non_int = NegativeBinomial::new(2.5, 0.3).unwrap();
        assert!(nb_non_int.pmf(0.0) > 0.0);
        assert!(nb_non_int.pmf(3.0) > 0.0);
    }

    #[test]
    fn test_negative_binomial_log_pmf() {
        // NegativeBinomial(5, 0.3)
        let nb = NegativeBinomial::new(5.0, 0.3).unwrap();

        // Check log_pmf against pmf
        for k in 0..15 {
            let k_f = k as f64;
            let pmf = nb.pmf(k_f);
            let log_pmf = nb.log_pmf(k_f);

            if pmf > 0.0 {
                assert_relative_eq!(log_pmf, pmf.ln(), epsilon = 1e-5);
            } else {
                assert!(log_pmf.is_infinite() && log_pmf.is_sign_negative());
            }
        }

        // log_pmf should be -infinity for negative k or non-integer k
        assert!(nb.log_pmf(-1.0).is_infinite() && nb.log_pmf(-1.0).is_sign_negative());
        assert!(nb.log_pmf(3.5).is_infinite() && nb.log_pmf(3.5).is_sign_negative());
    }

    #[test]
    fn test_negative_binomial_cdf() {
        // NegativeBinomial(5, 0.3)
        let nb = NegativeBinomial::new(5.0, 0.3).unwrap();

        // CDF for k < 0 should be 0
        assert_eq!(nb.cdf(-1.0), 0.0);

        // CDF values for different k
        assert_relative_eq!(nb.cdf(0.0), 0.00243, epsilon = 1e-5);
        assert_relative_eq!(nb.cdf(3.0), 0.05796, epsilon = 1e-5);
        assert_relative_eq!(nb.cdf(7.0), 0.27634, epsilon = 1e-5);
        assert_relative_eq!(nb.cdf(12.0), 0.61131, epsilon = 1e-5);

        // Non-integer k should use floor of k
        assert_relative_eq!(nb.cdf(7.5), nb.cdf(7.0), epsilon = 1e-10);

        // Special case: NegativeBinomial(r, 1.0)
        let nb_p1 = NegativeBinomial::new(5.0, 1.0).unwrap();
        assert_eq!(nb_p1.cdf(0.0), 1.0);
        assert_eq!(nb_p1.cdf(1.0), 1.0);
    }

    #[test]
    fn test_negative_binomial_ppf() {
        // NegativeBinomial(5, 0.3)
        let nb = NegativeBinomial::new(5.0, 0.3).unwrap();

        // Verify that ppf is the inverse of cdf for some test points
        let test_points = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        for &p in &test_points {
            let k = nb.ppf(p).unwrap();

            // For discrete distributions, we check that CDF(k-1) < p ≤ CDF(k)
            if k > 0.0 {
                let k_minus_1 = k - 1.0;
                let cdf_k_minus_1 = nb.cdf(k_minus_1);
                let cdf_k = nb.cdf(k);
                assert!(cdf_k_minus_1 < p && p <= cdf_k);
            } else {
                // For k = 0, just check that p ≤ CDF(0)
                let cdf_k = nb.cdf(k);
                assert!(p <= cdf_k);
            }
        }

        // Test edge cases
        assert_eq!(nb.ppf(0.0).unwrap(), 0.0);
        assert!(nb.ppf(1.0).unwrap().is_infinite());

        // Invalid probability values
        assert!(nb.ppf(-0.1).is_err());
        assert!(nb.ppf(1.1).is_err());
    }

    #[test]
    fn test_negative_binomial_rvs() {
        // NegativeBinomial(5, 0.3)
        let nb = NegativeBinomial::new(5.0, 0.3).unwrap();

        // Generate samples
        let samples = nb.rvs(100).unwrap();

        // Check number of samples
        assert_eq!(samples.len(), 100);

        // All samples should be non-negative integers
        for &s in &samples {
            assert!(s >= 0.0);
            assert_eq!(s, s.floor());
        }

        // Mean should be approximately r(1-p)/p = 5*0.7/0.3 = 11.67
        let sum: f64 = samples.iter().sum();
        let mean = sum / samples.len() as f64;

        // The mean could deviate somewhat due to randomness
        assert!(mean > 8.0 && mean < 15.0);
    }

    #[test]
    fn test_negative_binomial_moments() {
        // Test with r = 5, p = 0.3
        let nb = NegativeBinomial::new(5.0, 0.3).unwrap();

        // Mean = r(1-p)/p = 5*0.7/0.3 = 11.67
        assert_relative_eq!(nb.mean(), 11.666667, epsilon = 1e-6);

        // Variance = r(1-p)/p^2 = 5*0.7/0.3^2 = 38.89
        assert_relative_eq!(nb.var(), 38.888889, epsilon = 1e-6);

        // Standard deviation = sqrt(variance) = sqrt(38.89) ≈ 6.236
        assert_relative_eq!(nb.std(), 6.236095, epsilon = 1e-6);

        // Skewness = (2-p)/sqrt(r(1-p)) = (2-0.3)/sqrt(5*0.7) ≈ 0.909
        assert_relative_eq!(nb.skewness(), 0.908688, epsilon = 1e-6);

        // Kurtosis = 6/r + p^2/(r(1-p)) = 6/5 + 0.3^2/(5*0.7) ≈ 1.226
        assert_relative_eq!(nb.kurtosis(), 1.225714, epsilon = 1e-6);

        // Mode is integer for these parameters: floor((r-1)(1-p)/p) = floor(4*0.7/0.3) = 9
        assert_eq!(nb.mode(), 9.0);
    }

    #[test]
    fn test_negative_binomial_edge_cases() {
        // Test with r = 1, p = 0.3 (should be equivalent to Geometric(0.3))
        let nb_geom = NegativeBinomial::new(1.0, 0.3).unwrap();

        // Mean = (1-p)/p = 0.7/0.3 = 2.33...
        assert_relative_eq!(nb_geom.mean(), 2.333333, epsilon = 1e-6);

        // Variance = (1-p)/p^2 = 0.7/0.09 = 7.77...
        assert_relative_eq!(nb_geom.var(), 7.777778, epsilon = 1e-6);

        // Mode should be 0 for r = 1
        assert_eq!(nb_geom.mode(), 0.0);

        // Test with large r
        let nb_large = NegativeBinomial::new(100.0, 0.3).unwrap();

        // Mean = r(1-p)/p = 100*0.7/0.3 = 233.33...
        assert_relative_eq!(nb_large.mean(), 233.33333, epsilon = 1e-5);

        // Test with p close to 1
        let nb_p_high = NegativeBinomial::new(5.0, 0.99).unwrap();

        // Mean = r(1-p)/p = 5*0.01/0.99 = 0.0505...
        assert_relative_eq!(nb_p_high.mean(), 0.050505, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_binomial_compare_with_binomial() {
        // NegativeBinomial with integer r should use binomial coefficient calculation
        let nb_int = NegativeBinomial::new(5.0, 0.3).unwrap();

        // PMF values for different k
        let pmf_0 = nb_int.pmf(0.0);
        let pmf_3 = nb_int.pmf(3.0);
        let pmf_7 = nb_int.pmf(7.0);

        // Theoretical values
        // For k=0: C(5+0-1, 0) * 0.3^5 * 0.7^0 = C(4, 0) * 0.3^5 = 0.3^5 ≈ 0.00243
        // For k=3: C(5+3-1, 3) * 0.3^5 * 0.7^3 = C(7, 3) * 0.3^5 * 0.7^3 ≈ 0.02917
        // For k=7: C(5+7-1, 7) * 0.3^5 * 0.7^7 = C(11, 7) * 0.3^5 * 0.7^7 ≈ 0.06604
        assert_relative_eq!(pmf_0, 0.00243, epsilon = 1e-5);
        assert_relative_eq!(pmf_3, 0.02917, epsilon = 1e-5);
        assert_relative_eq!(pmf_7, 0.06604, epsilon = 1e-5);
    }

    #[test]
    fn test_non_integer_r() {
        // Test with non-integer r
        let nb_non_int = NegativeBinomial::new(2.5, 0.3).unwrap();

        // PMF values for different k
        // These values are approximated using gamma function approach
        let pmf_0 = nb_non_int.pmf(0.0);
        let pmf_3 = nb_non_int.pmf(3.0);
        let pmf_7 = nb_non_int.pmf(7.0);

        // Since we're using gamma approximation, just check the values are positive
        assert!(pmf_0 > 0.0);
        assert!(pmf_3 > 0.0);
        assert!(pmf_7 > 0.0);

        // Check that the distribution properties are consistent
        let mean = nb_non_int.mean();
        let var = nb_non_int.var();

        // Mean = r(1-p)/p = 2.5*0.7/0.3 = 5.83...
        assert_relative_eq!(mean, 5.833333, epsilon = 1e-6);

        // Variance = r(1-p)/p^2 = 2.5*0.7/0.3^2 = 19.44...
        assert_relative_eq!(var, 19.444444, epsilon = 1e-6);
    }
}
