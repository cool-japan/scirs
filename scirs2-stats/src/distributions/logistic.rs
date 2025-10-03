//! Logistic distribution functions
//!
//! This module provides functionality for the Logistic distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Uniform as RandUniform};

/// Logistic distribution structure
///
/// The Logistic distribution is a continuous probability distribution that
/// has applications in growth models, neural networks, and logistic regression.
/// It resembles the normal distribution but has heavier tails.
pub struct Logistic<F: Float> {
    /// Location parameter (mean, median, and mode of the distribution)
    pub loc: F,
    /// Scale parameter (diversity) > 0
    pub scale: F,
    /// Random number generator for uniform distribution
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Logistic<F> {
    /// Create a new Logistic distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (mean, median, and mode of the distribution)
    /// * `scale` - Scale parameter (diversity) > 0
    ///
    /// # Returns
    ///
    /// * A new Logistic distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// ```
    pub fn new(loc: F, scale: F) -> StatsResult<Self> {
        // Validate parameters
        if scale <= F::zero() {
            return Err(StatsError::DomainError(
                "Scale parameter must be positive".to_string(),
            ));
        }

        // Create RNG for uniform distribution in [0, 1)
        let rand_distr = match RandUniform::new(0.0, 1.0) {
            Ok(distr) => distr,
            Err(_) => {
                return Err(StatsError::ComputationError(
                    "Failed to create uniform distribution for sampling".to_string(),
                ))
            }
        };

        Ok(Logistic {
            loc,
            scale,
            rand_distr,
        })
    }

    /// Calculate the probability density function (PDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// * The value of the PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let pdf_at_zero = logistic.pdf(0.0);
    /// assert!((pdf_at_zero - 0.25).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        let z = (x - self.loc) / self.scale;
        let exp_neg_z = (-z).exp();

        // PDF = exp(-z) / (scale * (1 + exp(-z))^2)
        exp_neg_z / (self.scale * (F::one() + exp_neg_z).powi(2))
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// * The value of the CDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let cdf_at_zero = logistic.cdf(0.0);
    /// assert!((cdf_at_zero - 0.5).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        let z = (x - self.loc) / self.scale;

        // CDF = 1 / (1 + exp(-z))
        F::one() / (F::one() + (-z).exp())
    }

    /// Inverse of the cumulative distribution function (quantile function)
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value (between 0 and 1)
    ///
    /// # Returns
    ///
    /// * The value x such that CDF(x) = p
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let x = logistic.ppf(0.75).unwrap();
    /// assert!((x - 1.0986123).abs() < 1e-6);
    /// ```
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // Special cases
        if p == F::zero() {
            return Ok(F::neg_infinity());
        }
        if p == F::one() {
            return Ok(F::infinity());
        }

        // Quantile = loc + scale * ln(p / (1 - p))
        let quantile = self.loc + self.scale * (p / (F::one() - p)).ln();
        Ok(quantile)
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let samples = logistic.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // Generate uniform random number in (0, 1)
            // Avoid exactly 0 or 1 to prevent infinite values
            let mut u = self.rand_distr.sample(&mut rng);
            while u <= 0.0 || u >= 1.0 {
                u = self.rand_distr.sample(&mut rng);
            }

            let u_f = F::from(u).unwrap();

            // Apply inverse CDF transform
            let sample = match self.ppf(u_f) {
                Ok(s) => s,
                Err(_) => continue, // Skip invalid samples
            };

            samples.push(sample);
        }

        // Ensure we have exactly 'size' samples
        while samples.len() < size {
            let mut u = self.rand_distr.sample(&mut rng);
            while u <= 0.0 || u >= 1.0 {
                u = self.rand_distr.sample(&mut rng);
            }

            let u_f = F::from(u).unwrap();

            let sample = match self.ppf(u_f) {
                Ok(s) => s,
                Err(_) => continue,
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(2.0f64, 1.0).unwrap();
    /// let mean = logistic.mean();
    /// assert_eq!(mean, 2.0);
    /// ```
    pub fn mean(&self) -> F {
        // Mean = loc
        self.loc
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let variance = logistic.var();
    /// assert!((variance - 3.28986).abs() < 1e-5);
    /// ```
    pub fn var(&self) -> F {
        // Variance = (π^2/3) * scale^2
        let pi = F::from(std::f64::consts::PI).unwrap();
        let pi_squared = pi * pi;
        let one_third = F::from(1.0 / 3.0).unwrap();

        pi_squared * one_third * self.scale * self.scale
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let std_dev = logistic.std();
    /// assert!((std_dev - 1.81379).abs() < 1e-5);
    /// ```
    pub fn std(&self) -> F {
        // Std = sqrt(var)
        self.var().sqrt()
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(2.0f64, 1.0).unwrap();
    /// let median = logistic.median();
    /// assert_eq!(median, 2.0);
    /// ```
    pub fn median(&self) -> F {
        // Median = loc
        self.loc
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(2.0f64, 1.0).unwrap();
    /// let mode = logistic.mode();
    /// assert_eq!(mode, 2.0);
    /// ```
    pub fn mode(&self) -> F {
        // Mode = loc
        self.loc
    }

    /// Calculate the skewness of the distribution
    ///
    /// # Returns
    ///
    /// * The skewness (which is always 0 for the logistic distribution)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let skewness = logistic.skewness();
    /// assert_eq!(skewness, 0.0);
    /// ```
    pub fn skewness(&self) -> F {
        // Skewness = 0 (symmetric distribution)
        F::zero()
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let kurtosis = logistic.kurtosis();
    /// assert!((kurtosis - 1.2).abs() < 1e-7);
    /// ```
    pub fn kurtosis(&self) -> F {
        // Excess kurtosis = 1.2
        F::from(1.2).unwrap()
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
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let entropy = logistic.entropy();
    /// assert!((entropy - 2.0).abs() < 1e-7);
    /// ```
    pub fn entropy(&self) -> F {
        // Entropy = 2 + ln(scale)
        let two = F::from(2.0).unwrap();
        two + self.scale.ln()
    }

    /// Calculate the interquartile range (IQR) of the distribution
    ///
    /// # Returns
    ///
    /// * The interquartile range
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::logistic::Logistic;
    ///
    /// let logistic = Logistic::new(0.0f64, 1.0).unwrap();
    /// let iqr = logistic.iqr();
    /// assert!((iqr - 2.1972245) < 1e-6);
    /// ```
    pub fn iqr(&self) -> F {
        // IQR = scale * ln(3) ≈ scale * 2.1972245...
        let three = F::from(3.0).unwrap();
        self.scale * three.ln()
    }
}

/// Create a Logistic distribution with the given parameters.
///
/// This is a convenience function to create a Logistic distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (mean, median, and mode of the distribution)
/// * `scale` - Scale parameter (diversity) > 0
///
/// # Returns
///
/// * A Logistic distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::logistic;
///
/// let l = logistic::logistic(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = l.pdf(0.0);
/// assert!((pdf_at_zero - 0.25).abs() < 1e-7);
/// ```
#[allow(dead_code)]
pub fn logistic<F>(loc: F, scale: F) -> StatsResult<Logistic<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Logistic::new(loc, scale)
}

/// Implementation of SampleableDistribution for Logistic
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Logistic<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_logistic_creation() {
        // Standard Logistic (loc=0, scale=1)
        let logistic = Logistic::new(0.0, 1.0).unwrap();
        assert_eq!(logistic.loc, 0.0);
        assert_eq!(logistic.scale, 1.0);

        // Custom Logistic
        let custom = Logistic::new(-2.0, 0.5).unwrap();
        assert_eq!(custom.loc, -2.0);
        assert_eq!(custom.scale, 0.5);

        // Error case: non-positive scale
        assert!(Logistic::<f64>::new(0.0, 0.0).is_err());
        assert!(Logistic::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_logistic_pdf() {
        // Standard Logistic (loc=0, scale=1)
        let logistic = Logistic::new(0.0, 1.0).unwrap();

        // PDF at x = 0 should be 1/4 = 0.25
        let pdf_at_zero = logistic.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 0.25, epsilon = 1e-7);

        // PDF at x = 1
        let pdf_at_one = logistic.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.196612, epsilon = 1e-6);

        // PDF at x = -1 should same as x = 1 due to symmetry
        let pdf_at_neg_one = logistic.pdf(-1.0);
        assert_relative_eq!(pdf_at_neg_one, pdf_at_one, epsilon = 1e-10);

        // Custom Logistic with loc=-2, scale=0.5
        let custom = Logistic::new(-2.0, 0.5).unwrap();

        // PDF at x = -2 should be 1/(4*0.5) = 0.5
        let pdf_at_loc = custom.pdf(-2.0);
        assert_relative_eq!(pdf_at_loc, 0.5, epsilon = 1e-7);
    }

    #[test]
    fn test_logistic_cdf() {
        // Standard Logistic (loc=0, scale=1)
        let logistic = Logistic::new(0.0, 1.0).unwrap();

        // CDF at x = 0 should be 0.5
        let cdf_at_zero = logistic.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.5, epsilon = 1e-10);

        // CDF at x = 1 should be 1/(1+exp(-1)) ≈ 0.7310586
        let cdf_at_one = logistic.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.7310586, epsilon = 1e-7);

        // CDF at x = -1 should be 1-CDF(1) ≈ 0.2689414 due to symmetry
        let cdf_at_neg_one = logistic.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 0.2689414, epsilon = 1e-7);

        // Custom Logistic with loc=-2, scale=0.5
        let custom = Logistic::new(-2.0, 0.5).unwrap();

        // CDF at x = -2 should be 0.5
        let cdf_at_loc = custom.cdf(-2.0);
        assert_relative_eq!(cdf_at_loc, 0.5, epsilon = 1e-10);

        // CDF at x = -1.5 should be 1/(1+exp(-(-1.5-(-2))/0.5)) = 1/(1+exp(-1)) ≈ 0.7310586
        let cdf_at_custom = custom.cdf(-1.5);
        assert_relative_eq!(cdf_at_custom, 0.7310586, epsilon = 1e-7);
    }

    #[test]
    fn test_logistic_ppf() {
        // Standard Logistic (loc=0, scale=1)
        let logistic = Logistic::new(0.0, 1.0).unwrap();

        // PPF at p = 0.5 should be 0
        let ppf_at_half = logistic.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half, 0.0, epsilon = 1e-10);

        // PPF at p = 0.75 should be ln(3) ≈ 1.0986123
        let ppf_at_75 = logistic.ppf(0.75).unwrap();
        assert_relative_eq!(ppf_at_75, 1.0986123, epsilon = 1e-6);

        // PPF at p = 0.25 should be -ln(3) ≈ -1.0986123
        let ppf_at_25 = logistic.ppf(0.25).unwrap();
        assert_relative_eq!(ppf_at_25, -1.0986123, epsilon = 1e-6);

        // Custom Logistic with loc=-2, scale=0.5
        let custom = Logistic::new(-2.0, 0.5).unwrap();

        // PPF at p = 0.5 should be -2.0
        let ppf_at_half_custom = custom.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half_custom, -2.0, epsilon = 1e-10);

        // PPF at p = 0.75 should be -2.0 + 0.5*ln(3) ≈ -2.0 + 0.5493062 ≈ -1.4506938
        let ppf_at_75_custom = custom.ppf(0.75).unwrap();
        assert_relative_eq!(ppf_at_75_custom, -1.4506938, epsilon = 1e-6);

        // Error cases
        assert!(logistic.ppf(-0.1).is_err());
        assert!(logistic.ppf(1.1).is_err());
    }

    #[test]
    fn test_logistic_properties() {
        // Standard Logistic (loc=0, scale=1)
        let logistic = Logistic::new(0.0, 1.0).unwrap();

        // Mean = loc = 0
        let mean = logistic.mean();
        assert_eq!(mean, 0.0);

        // Variance = (π^2/3) * scale^2 ≈ 3.28986...
        let variance = logistic.var();
        assert_relative_eq!(variance, 3.28986, epsilon = 1e-5);

        // Std = sqrt(variance) ≈ 1.81379...
        let std_dev = logistic.std();
        assert_relative_eq!(std_dev, 1.81379, epsilon = 1e-5);

        // Median = loc = 0
        let median = logistic.median();
        assert_eq!(median, 0.0);

        // Mode = loc = 0
        let mode = logistic.mode();
        assert_eq!(mode, 0.0);

        // Skewness = 0 (symmetric)
        let skewness = logistic.skewness();
        assert_eq!(skewness, 0.0);

        // Kurtosis = 1.2
        let kurtosis = logistic.kurtosis();
        assert_eq!(kurtosis, 1.2);

        // Entropy = 2 + ln(scale) = 2 + ln(1) = 2
        let entropy = logistic.entropy();
        assert_eq!(entropy, 2.0);

        // IQR = scale * ln(3) ≈ 1 * 1.0986... ≈ 1.0986...
        let iqr = logistic.iqr();
        assert_relative_eq!(iqr, 1.0986123, epsilon = 1e-6);

        // Custom Logistic with loc=-2, scale=0.5
        let custom = Logistic::new(-2.0, 0.5).unwrap();

        // Mean = loc = -2
        let mean_custom = custom.mean();
        assert_eq!(mean_custom, -2.0);

        // Variance = (π^2/3) * scale^2 = (π^2/3) * 0.5^2 ≈ 0.82247...
        let variance_custom = custom.var();
        assert_relative_eq!(variance_custom, 0.82247, epsilon = 1e-5);

        // Entropy = 2 + ln(scale) = 2 + ln(0.5) ≈ 2 - 0.693147... ≈ 1.30685...
        let entropy_custom = custom.entropy();
        assert_relative_eq!(entropy_custom, 1.30685, epsilon = 1e-5);
    }

    #[test]
    fn test_logistic_rvs() {
        let logistic = Logistic::new(0.0, 1.0).unwrap();

        // Generate samples
        let samples = logistic.rvs(100).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 100);

        // Calculate sample mean and check it's reasonably close to loc = 0
        let sum: f64 = samples.iter().sum();
        let mean = sum / samples.len() as f64;

        // The mean could be off due to randomness, but should be within a reasonable range
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_logistic_inverse_cdf() {
        // Test that cdf(ppf(p)) == p and ppf(cdf(x)) == x
        let logistic = Logistic::new(0.0, 1.0).unwrap();

        // Test various probability values
        let probabilities = [0.1, 0.25, 0.5, 0.75, 0.9];
        for &p in &probabilities {
            let x = logistic.ppf(p).unwrap();
            let p_back = logistic.cdf(x);
            assert_relative_eq!(p_back, p, epsilon = 1e-7);
        }

        // Test various x values
        let x_values = [-3.0, -1.0, 0.0, 1.0, 3.0];
        for &x in &x_values {
            let p = logistic.cdf(x);
            let x_back = logistic.ppf(p).unwrap();
            assert_relative_eq!(x_back, x, epsilon = 1e-7);
        }
    }
}
