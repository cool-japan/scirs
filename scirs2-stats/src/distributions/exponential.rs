//! Exponential distribution functions
//!
//! This module provides functionality for the Exponential distribution.

use crate::error::{StatsError, StatsResult};
use crate::error_messages::{helpers, validation};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Distribution, Exponential as RandExp};
use std::fmt::Debug;

/// Exponential distribution structure
pub struct Exponential<F: Float> {
    /// Rate parameter λ - inverse of scale
    pub rate: F,
    /// Scale parameter (θ = 1/λ)
    pub scale: F,
    /// Location parameter
    pub loc: F,
    /// Random number generator for this distribution
    rand_distr: RandExp<f64>,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> Exponential<F> {
    /// Create a new exponential distribution with given **rate** and location parameters
    ///
    /// # Important: Rate vs Scale
    ///
    /// This function takes the **rate parameter λ**, NOT the scale parameter.
    /// The relationship between rate and scale is:
    /// - rate = λ = 1/scale
    /// - scale = θ = 1/rate
    ///
    /// For an exponential distribution with rate λ:
    /// - Mean = 1/λ = scale
    /// - Variance = 1/λ² = scale²
    ///
    /// # Arguments
    ///
    /// * `rate` - Rate parameter λ > 0 (where λ = 1/scale)
    /// * `loc` - Location parameter (default: 0)
    ///
    /// # Returns
    ///
    /// * A new Exponential distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::exponential::Exponential;
    /// use scirs2_stats::traits::Distribution;
    ///
    /// // Create exponential with rate=0.5 (equivalent to scale=2.0)
    /// let exp = Exponential::new(0.5f64, 0.0).unwrap();
    /// assert!((exp.rate - 0.5).abs() < 1e-10);
    /// assert!((exp.scale - 2.0).abs() < 1e-10);
    /// assert!((exp.mean() - 2.0).abs() < 1e-10);  // mean = 1/rate = 2.0
    ///
    /// // If you want to specify scale directly, use from_scale() instead:
    /// // let exp = Exponential::from_scale(2.0f64, 0.0).unwrap();
    /// ```
    pub fn new(rate: F, loc: F) -> StatsResult<Self> {
        validation::ensure_positive(rate, "Rate parameter")?;

        // Set scale = 1/rate
        let scale = F::one() / rate;

        // Convert to f64 for rand_distr
        let rate_f64 = <f64 as NumCast>::from(rate).unwrap();

        match RandExp::new(rate_f64) {
            Ok(rand_distr) => Ok(Exponential {
                rate,
                scale,
                loc,
                rand_distr,
            }),
            Err(_) => Err(helpers::numerical_error(
                "exponential distribution creation",
            )),
        }
    }

    /// Create a new exponential distribution with given **scale** and location parameters
    ///
    /// This is an alternative constructor that takes the **scale parameter θ** instead of rate.
    /// Many users prefer this interface as scale directly represents the mean of the distribution.
    ///
    /// # Scale vs Rate
    ///
    /// - scale = θ = 1/λ = mean of the distribution
    /// - rate = λ = 1/θ
    ///
    /// For an exponential distribution with scale θ:
    /// - Mean = θ = scale
    /// - Variance = θ² = scale²
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale parameter θ > 0 (where θ = 1/rate = mean)
    /// * `loc` - Location parameter (default: 0)
    ///
    /// # Returns
    ///
    /// * A new Exponential distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::exponential::Exponential;
    /// use scirs2_stats::traits::Distribution;
    ///
    /// // Create exponential with scale=2.0 (equivalent to rate=0.5)
    /// let exp = Exponential::from_scale(2.0f64, 0.0).unwrap();
    /// assert!((exp.scale - 2.0).abs() < 1e-10);
    /// assert!((exp.rate - 0.5).abs() < 1e-10);
    /// assert!((exp.mean() - 2.0).abs() < 1e-10);  // mean = scale = 2.0
    ///
    /// // This is equivalent to:
    /// // let exp = Exponential::new(0.5f64, 0.0).unwrap();
    /// ```
    pub fn from_scale(scale: F, loc: F) -> StatsResult<Self> {
        validation::ensure_positive(scale, "scale")?;

        // Set rate = 1/scale
        let rate = F::one() / scale;

        // Convert to f64 for rand_distr
        let rate_f64 = <f64 as NumCast>::from(rate).unwrap();

        match RandExp::new(rate_f64) {
            Ok(rand_distr) => Ok(Exponential {
                rate,
                scale,
                loc,
                rand_distr,
            }),
            Err(_) => Err(helpers::numerical_error(
                "exponential distribution creation",
            )),
        }
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
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(1.0f64, 0.0).unwrap();
    /// let pdf_at_one = exp.pdf(1.0);
    /// assert!((pdf_at_one - 0.36787944).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn pdf(&self, x: F) -> F {
        // Adjust for location
        let x_adj = x - self.loc;

        // If x is less than loc, PDF is 0
        if x_adj < F::zero() {
            return F::zero();
        }

        // PDF = λ * exp(-λ * x)
        self.rate * (-self.rate * x_adj).exp()
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
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(1.0f64, 0.0).unwrap();
    /// let cdf_at_one = exp.cdf(1.0);
    /// assert!((cdf_at_one - 0.63212056).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn cdf(&self, x: F) -> F {
        // Adjust for location
        let x_adj = x - self.loc;

        // If x is less than loc, CDF is 0
        if x_adj <= F::zero() {
            return F::zero();
        }

        // CDF = 1 - exp(-λ * x)
        F::one() - (-self.rate * x_adj).exp()
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
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(1.0f64, 0.0).unwrap();
    /// let x = exp.ppf(0.5).unwrap();
    /// assert!((x - 0.69314718).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // Special cases
        if p == F::zero() {
            return Ok(self.loc);
        }
        if p == F::one() {
            return Ok(F::infinity());
        }

        // For exponential distribution, the quantile function has a simple analytic form:
        // x = -ln(1-p) / λ
        let result = -((F::one() - p).ln()) / self.rate;
        Ok(result + self.loc)
    }

    /// Calculate the mean of the distribution
    ///
    /// # Returns
    ///
    /// * The mean value
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(2.0f64, 1.0).unwrap();
    /// assert_eq!(exp.mean(), 1.5); // loc + 1/rate = 1 + 1/2 = 1.5
    /// ```
    pub fn mean(&self) -> F {
        self.loc + self.scale
    }

    /// Calculate the variance of the distribution
    ///
    /// # Returns
    ///
    /// * The variance value
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(2.0f64, 0.0).unwrap();
    /// assert_eq!(exp.variance(), 0.25); // (1/rate)^2 = (1/2)^2 = 0.25
    /// ```
    pub fn variance(&self) -> F {
        self.scale * self.scale
    }

    /// Generate random samples from the distribution as an Array1
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// * Array1 of random samples
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(1.0f64, 0.0).unwrap();
    /// let samples = exp.rvs(1000).unwrap();
    /// assert_eq!(samples.len(), 1000);
    /// ```
    #[inline]
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let samples = self.rvs_vec(size)?;
        Ok(Array1::from_vec(samples))
    }

    /// Generate random samples from the distribution as a Vec
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
    /// use scirs2_stats::distributions::exponential::Exponential;
    ///
    /// let exp = Exponential::new(1.0f64, 0.0).unwrap();
    /// let samples = exp.rvs_vec(1000).unwrap();
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            let sample = self.rand_distr.sample(&mut rng);
            samples.push(F::from(sample).unwrap() + self.loc);
        }

        Ok(samples)
    }
}

/// Implementation of the Distribution trait for Exponential
impl<F: Float + NumCast + Debug + std::fmt::Display> ScirsDist<F> for Exponential<F> {
    fn mean(&self) -> F {
        self.mean()
    }

    fn var(&self) -> F {
        self.variance()
    }

    fn std(&self) -> F {
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        // Entropy of exponential distribution is 1 - ln(rate)
        F::one() - self.rate.ln()
    }
}

/// Implementation of the ContinuousDistribution trait for Exponential
impl<F: Float + NumCast + Debug + std::fmt::Display> ContinuousDistribution<F> for Exponential<F> {
    fn pdf(&self, x: F) -> F {
        self.pdf(x)
    }

    fn cdf(&self, x: F) -> F {
        self.cdf(x)
    }

    fn ppf(&self, p: F) -> StatsResult<F> {
        self.ppf(p)
    }
}

impl<F: Float + NumCast + Debug + std::fmt::Display> ContinuousCDF<F> for Exponential<F> {
    // Default implementations from trait are sufficient
}

/// Implementation of SampleableDistribution for Exponential
impl<F: Float + NumCast + Debug + std::fmt::Display> SampleableDistribution<F> for Exponential<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs_vec(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ContinuousDistribution, Distribution as ScirsDist};
    use approx::assert_relative_eq;

    #[test]
    fn test_exponential_creation() {
        // Basic exponential distribution with rate=1
        let exp = Exponential::new(1.0, 0.0).unwrap();
        assert_eq!(exp.rate, 1.0);
        assert_eq!(exp.scale, 1.0);
        assert_eq!(exp.loc, 0.0);

        // From scale parameter
        let exp_scale = Exponential::from_scale(2.0, 0.0).unwrap();
        assert_eq!(exp_scale.rate, 0.5);
        assert_eq!(exp_scale.scale, 2.0);
        assert_eq!(exp_scale.loc, 0.0);

        // Custom exponential with location
        let custom = Exponential::new(2.0, 1.0).unwrap();
        assert_eq!(custom.rate, 2.0);
        assert_eq!(custom.scale, 0.5);
        assert_eq!(custom.loc, 1.0);

        // Error cases
        assert!(Exponential::<f64>::new(0.0, 0.0).is_err());
        assert!(Exponential::<f64>::new(-1.0, 0.0).is_err());
        assert!(Exponential::<f64>::from_scale(0.0, 0.0).is_err());
        assert!(Exponential::<f64>::from_scale(-1.0, 0.0).is_err());
    }

    #[test]
    fn test_exponential_pdf() {
        // Standard exponential PDF values (rate=1)
        let exp = Exponential::new(1.0, 0.0).unwrap();

        // PDF at x = 0
        let pdf_at_zero = exp.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 1.0, epsilon = 1e-10);

        // PDF at x = 1
        let pdf_at_one = exp.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.36787944, epsilon = 1e-7);

        // PDF at x = 2
        let pdf_at_two = exp.pdf(2.0);
        assert_relative_eq!(pdf_at_two, 0.13533528, epsilon = 1e-7);

        // PDF at x < loc
        assert_relative_eq!(exp.pdf(-1.0), 0.0, epsilon = 1e-10);

        // Custom rate
        let exp2 = Exponential::new(2.0, 0.0).unwrap();
        assert_relative_eq!(exp2.pdf(0.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(exp2.pdf(0.5), 0.73575888, epsilon = 1e-7);

        // With location parameter
        let shifted = Exponential::new(1.0, 1.0).unwrap();
        assert_relative_eq!(shifted.pdf(0.5), 0.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.pdf(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.pdf(2.0), 0.36787944, epsilon = 1e-7);
    }

    #[test]
    fn test_exponential_cdf() {
        // Standard exponential CDF values (rate=1)
        let exp = Exponential::new(1.0, 0.0).unwrap();

        // CDF at x = 0
        let cdf_at_zero = exp.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.0, epsilon = 1e-10);

        // CDF at x = 1
        let cdf_at_one = exp.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.63212056, epsilon = 1e-7);

        // CDF at x = 2
        let cdf_at_two = exp.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.86466472, epsilon = 1e-7);

        // CDF at x < loc
        assert_relative_eq!(exp.cdf(-1.0), 0.0, epsilon = 1e-10);

        // Custom rate
        let exp2 = Exponential::new(2.0, 0.0).unwrap();
        assert_relative_eq!(exp2.cdf(0.5), 0.63212056, epsilon = 1e-7);
        assert_relative_eq!(exp2.cdf(1.0), 0.86466472, epsilon = 1e-7);

        // With location parameter
        let shifted = Exponential::new(1.0, 1.0).unwrap();
        assert_relative_eq!(shifted.cdf(0.5), 0.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.cdf(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.cdf(2.0), 0.63212056, epsilon = 1e-7);
    }

    #[test]
    fn test_exponential_ppf() {
        // Standard exponential (rate=1)
        let exp = Exponential::new(1.0, 0.0).unwrap();

        // Median
        let median = exp.ppf(0.5).unwrap();
        assert_relative_eq!(median, 0.69314718, epsilon = 1e-7);

        // 95th percentile
        let p95 = exp.ppf(0.95).unwrap();
        assert_relative_eq!(p95, 2.9957323, epsilon = 1e-7);

        // With location parameter
        let shifted = Exponential::new(1.0, 1.0).unwrap();
        assert_relative_eq!(shifted.ppf(0.5).unwrap(), 1.69314718, epsilon = 1e-7);

        // Error cases
        assert!(exp.ppf(-0.1).is_err());
        assert!(exp.ppf(1.1).is_err());
    }

    #[test]
    fn test_exponential_mean_variance() {
        // Standard exponential (rate=1)
        let exp = Exponential::new(1.0, 0.0).unwrap();
        assert_relative_eq!(exp.mean(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp.variance(), 1.0, epsilon = 1e-10);

        // Custom rate (rate=2)
        let exp2 = Exponential::new(2.0, 0.0).unwrap();
        assert_relative_eq!(exp2.mean(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(exp2.variance(), 0.25, epsilon = 1e-10);

        // With location (rate=1, loc=1)
        let shifted = Exponential::new(1.0, 1.0).unwrap();
        assert_relative_eq!(shifted.mean(), 2.0, epsilon = 1e-10); // loc + 1/rate
        assert_relative_eq!(shifted.variance(), 1.0, epsilon = 1e-10); // location doesn't affect variance
    }

    #[test]
    fn test_exponential_rvs() {
        let exp = Exponential::new(1.0, 0.0).unwrap();

        // Generate samples using Vec method
        let samples_vec = exp.rvs_vec(1000).unwrap();
        assert_eq!(samples_vec.len(), 1000);

        // Generate samples using Array1 method
        let samples_array = exp.rvs(1000).unwrap();
        assert_eq!(samples_array.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples_vec.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to 1.0 for Exponential(1)
        assert!((mean - 1.0).abs() < 0.1);

        // Variance check
        let variance: f64 = samples_vec
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / 1000.0;

        // Variance should also be close to 1.0
        // Using a larger tolerance (0.3) for the statistical test since it can be affected by randomness
        assert!((variance - 1.0).abs() < 0.3);

        // Check all samples are non-negative
        for &sample in &samples_vec {
            assert!(sample >= 0.0);
        }
    }

    #[test]
    fn test_exponential_distribution_trait() {
        let exp = Exponential::new(1.0, 0.0).unwrap();

        // Test Distribution trait methods
        assert_relative_eq!(exp.mean(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp.var(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp.std(), 1.0, epsilon = 1e-10);

        // Check that rvs returns correct size and type
        let samples = exp.rvs(100).unwrap();
        assert_eq!(samples.len(), 100);

        // Entropy should be 1.0 for standard exponential
        assert_relative_eq!(exp.entropy(), 1.0, epsilon = 1e-10);

        // Entropy for different rate
        let exp2 = Exponential::new(2.0, 0.0).unwrap();
        // Entropy = 1 - ln(rate) = 1 - ln(2) ≈ 0.3069
        assert_relative_eq!(exp2.entropy(), 1.0 - 2.0f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_continuous_distribution_trait() {
        let exp = Exponential::new(1.0, 0.0).unwrap();

        // Test as a ContinuousDistribution
        let dist: &dyn ContinuousDistribution<f64> = &exp;

        // Check PDF
        assert_relative_eq!(dist.pdf(1.0), 0.36787944, epsilon = 1e-7);

        // Check CDF
        assert_relative_eq!(dist.cdf(1.0), 0.63212056, epsilon = 1e-7);

        // Check PPF
        assert_relative_eq!(dist.ppf(0.5).unwrap(), 0.69314718, epsilon = 1e-7);

        // Check derived methods using concrete type
        assert_relative_eq!(exp.sf(1.0), 1.0 - 0.63212056, epsilon = 1e-7);

        // Hazard function for exponential should be constant = rate
        assert_relative_eq!(exp.hazard(1.0), 1.0, epsilon = 1e-7);

        // Cumulative hazard function for exponential is just rate*x
        assert_relative_eq!(exp.cumhazard(1.0), 1.0, epsilon = 1e-7);

        // Inverse survival function should work
        assert_relative_eq!(
            exp.isf(0.5).unwrap(),
            dist.ppf(0.5).unwrap(),
            epsilon = 1e-7
        );
    }
}
