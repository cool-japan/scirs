//! Cauchy distribution functions
//!
//! This module provides functionality for the Cauchy (Lorentz) distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Uniform as RandUniform};

/// Cauchy distribution structure
///
/// The Cauchy distribution, also known as the Lorentz distribution, is a continuous
/// probability distribution that is the ratio of two independent normally distributed
/// random variables. It is notable for not having defined moments (like mean or variance).
pub struct Cauchy<F: Float> {
    /// Location parameter
    pub loc: F,
    /// Scale parameter (gamma > 0)
    pub scale: F,
    /// Random number generator for uniform distribution
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Cauchy<F> {
    /// Create a new Cauchy distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (median of the distribution)
    /// * `scale` - Scale parameter (half-width at half-maximum) > 0
    ///
    /// # Returns
    ///
    /// * A new Cauchy distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
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

        Ok(Cauchy {
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let pdf_at_zero = cauchy.pdf(0.0);
    /// assert!((pdf_at_zero - 0.3183099).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        let pi = F::from(std::f64::consts::PI).unwrap();
        let one = F::one();

        // PDF = 1 / (π * scale * (1 + ((x - loc) / scale)^2))
        let z = (x - self.loc) / self.scale;
        one / (pi * self.scale * (one + z * z))
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let cdf_at_zero = cauchy.cdf(0.0);
    /// assert!((cdf_at_zero - 0.5).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        let pi = F::from(std::f64::consts::PI).unwrap();
        let half = F::from(0.5).unwrap();

        // CDF = 0.5 + (1/π) * arctan((x - loc) / scale)
        let z = (x - self.loc) / self.scale;
        half + z.atan() / pi
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let x = cauchy.ppf(0.75).unwrap();
    /// assert!((x - 1.0).abs() < 1e-7);
    /// ```
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        let pi = F::from(std::f64::consts::PI).unwrap();
        let half = F::from(0.5).unwrap();

        // Quantile = loc + scale * tan(π * (p - 0.5))
        let tan_term = (pi * (p - half)).tan();
        let quantile = self.loc + self.scale * tan_term;

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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let samples = cauchy.rvs_vec(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);

        // Generate samples using the inverse transform sampling method
        for _ in 0..size {
            // Generate uniform random number in (0, 1)
            let u = self.rand_distr.sample(&mut rng);

            // Avoid exactly 0.5 to prevent undefined tan(0)
            if (u - 0.5).abs() < 1e-10 {
                continue;
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
            // Generate uniform random number in (0, 1)
            let u = self.rand_distr.sample(&mut rng);

            // Avoid exactly 0.5 to prevent undefined tan(0)
            if (u - 0.5).abs() < 1e-10 {
                continue;
            }

            let u_f = F::from(u).unwrap();

            // Apply inverse CDF transform
            let sample = match self.ppf(u_f) {
                Ok(s) => s,
                Err(_) => continue, // Skip invalid samples
            };

            samples.push(sample);
        }

        Ok(samples)
    }

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// * Array of random samples
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let samples = cauchy.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let samples_vec = self.rvs_vec(size)?;
        Ok(Array1::from(samples_vec))
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(2.0f64, 1.0).unwrap();
    /// let median = cauchy.median();
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(2.0f64, 1.0).unwrap();
    /// let mode = cauchy.mode();
    /// assert_eq!(mode, 2.0);
    /// ```
    pub fn mode(&self) -> F {
        // Mode = loc
        self.loc
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let iqr = cauchy.iqr();
    /// assert!((iqr - 2.0).abs() < 1e-7);
    /// ```
    pub fn iqr(&self) -> F {
        // IQR = 2 * scale
        self.scale + self.scale
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
    /// use scirs2_stats::distributions::cauchy::Cauchy;
    ///
    /// let cauchy = Cauchy::new(0.0f64, 1.0).unwrap();
    /// let entropy = cauchy.entropy();
    /// assert!((entropy - 2.5310242).abs() < 1e-7); // log(4π)
    /// ```
    pub fn entropy(&self) -> F {
        let pi = F::from(std::f64::consts::PI).unwrap();
        let four = F::from(4.0).unwrap();

        // Entropy = log(4 * π * scale)
        (four * pi * self.scale).ln()
    }
}

/// Create a Cauchy distribution with the given parameters.
///
/// This is a convenience function to create a Cauchy distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (median of the distribution)
/// * `scale` - Scale parameter (half-width at half-maximum) > 0
///
/// # Returns
///
/// * A Cauchy distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::cauchy;
///
/// let c = cauchy::cauchy(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = c.pdf(0.0);
/// assert!((pdf_at_zero - 0.3183099).abs() < 1e-7);
/// ```
#[allow(dead_code)]
pub fn cauchy<F>(loc: F, scale: F) -> StatsResult<Cauchy<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Cauchy::new(loc, scale)
}

/// Implementation of SampleableDistribution for Cauchy
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Cauchy<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs_vec(size)
    }
}

/// Implementation of Distribution trait for Cauchy
///
/// Note: Cauchy distribution doesn't have defined mean or variance,
/// but we implement the Distribution trait with appropriate behavior.
impl<F: Float + NumCast + std::fmt::Display> ScirsDist<F> for Cauchy<F> {
    /// Returns NaN as the mean is undefined for Cauchy distribution
    fn mean(&self) -> F {
        F::nan()
    }

    /// Returns NaN as the variance is undefined for Cauchy distribution
    fn var(&self) -> F {
        F::nan()
    }

    /// Returns NaN as the standard deviation is undefined for Cauchy distribution
    fn std(&self) -> F {
        F::nan()
    }

    /// Generate random samples from the distribution
    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    /// Calculate the entropy of the distribution
    fn entropy(&self) -> F {
        self.entropy()
    }
}

/// Implementation of ContinuousDistribution trait for Cauchy
impl<F: Float + NumCast + std::fmt::Display> ContinuousDistribution<F> for Cauchy<F> {
    /// Calculate the probability density function (PDF) at a given point
    fn pdf(&self, x: F) -> F {
        self.pdf(x)
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    fn cdf(&self, x: F) -> F {
        self.cdf(x)
    }

    /// Calculate the inverse cumulative distribution function (quantile function)
    fn ppf(&self, p: F) -> StatsResult<F> {
        self.ppf(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cauchy_creation() {
        // Standard Cauchy (loc=0, scale=1)
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();
        assert_eq!(cauchy.loc, 0.0);
        assert_eq!(cauchy.scale, 1.0);

        // Custom Cauchy
        let custom = Cauchy::new(-2.0, 0.5).unwrap();
        assert_eq!(custom.loc, -2.0);
        assert_eq!(custom.scale, 0.5);

        // Error case: negative scale
        assert!(Cauchy::<f64>::new(0.0, 0.0).is_err());
        assert!(Cauchy::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_cauchy_pdf() {
        // Standard Cauchy (loc=0, scale=1)
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();

        // PDF at x = 0 should be 1/(π*1) = 1/π ≈ 0.3183099
        let pdf_at_zero = cauchy.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 0.3183099, epsilon = 1e-7);

        // PDF at x = 1 should be 1/(π*1*(1+1)) = 1/(2π) ≈ 0.1591549
        let pdf_at_one = cauchy.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.1591549, epsilon = 1e-7);

        // PDF at x = -1 should same as x = 1 due to symmetry
        let pdf_at_neg_one = cauchy.pdf(-1.0);
        assert_relative_eq!(pdf_at_neg_one, pdf_at_one, epsilon = 1e-10);

        // Custom Cauchy with loc=-2, scale=0.5
        let custom = Cauchy::new(-2.0, 0.5).unwrap();

        // PDF at x = -2 should be 1/(π*0.5) = 2/π ≈ 0.6366198
        let pdf_at_loc = custom.pdf(-2.0);
        assert_relative_eq!(pdf_at_loc, 0.6366198, epsilon = 1e-7);

        // PDF at x = -1.5 should be 1/(π*0.5*(1+1)) = 1/π ≈ 0.3183099
        let pdf_at_custom = custom.pdf(-1.5);
        assert_relative_eq!(pdf_at_custom, 0.3183099, epsilon = 1e-7);
    }

    #[test]
    fn test_cauchy_cdf() {
        // Standard Cauchy (loc=0, scale=1)
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();

        // CDF at x = 0 should be 0.5
        let cdf_at_zero = cauchy.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.5, epsilon = 1e-10);

        // CDF at x = 1 should be 0.5 + (1/π)*arctan(1) = 0.5 + 1/4 = 0.75
        let cdf_at_one = cauchy.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.75, epsilon = 1e-7);

        // CDF at x = -1 should be 0.5 - (1/π)*arctan(1) = 0.5 - 1/4 = 0.25
        let cdf_at_neg_one = cauchy.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 0.25, epsilon = 1e-7);

        // Custom Cauchy with loc=-2, scale=0.5
        let custom = Cauchy::new(-2.0, 0.5).unwrap();

        // CDF at x = -2 should be 0.5
        let cdf_at_loc = custom.cdf(-2.0);
        assert_relative_eq!(cdf_at_loc, 0.5, epsilon = 1e-10);

        // CDF at x = -1.5 should be 0.5 + (1/π)*arctan(1) = 0.75
        let cdf_at_custom = custom.cdf(-1.5);
        assert_relative_eq!(cdf_at_custom, 0.75, epsilon = 1e-7);
    }

    #[test]
    fn test_cauchy_ppf() {
        // Standard Cauchy (loc=0, scale=1)
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();

        // PPF at p = 0.5 should be 0
        let ppf_at_half = cauchy.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half, 0.0, epsilon = 1e-10);

        // PPF at p = 0.75 should be 1.0
        let ppf_at_75 = cauchy.ppf(0.75).unwrap();
        assert_relative_eq!(ppf_at_75, 1.0, epsilon = 1e-7);

        // PPF at p = 0.25 should be -1.0
        let ppf_at_25 = cauchy.ppf(0.25).unwrap();
        assert_relative_eq!(ppf_at_25, -1.0, epsilon = 1e-7);

        // Custom Cauchy with loc=-2, scale=0.5
        let custom = Cauchy::new(-2.0, 0.5).unwrap();

        // PPF at p = 0.5 should be -2.0
        let ppf_at_half_custom = custom.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half_custom, -2.0, epsilon = 1e-10);

        // PPF at p = 0.75 should be -2.0 + 0.5*tan(π/4) = -2.0 + 0.5 = -1.5
        let ppf_at_75_custom = custom.ppf(0.75).unwrap();
        assert_relative_eq!(ppf_at_75_custom, -1.5, epsilon = 1e-7);

        // Error cases
        assert!(cauchy.ppf(-0.1).is_err());
        assert!(cauchy.ppf(1.1).is_err());
    }

    #[test]
    fn test_cauchy_properties() {
        // Standard Cauchy (loc=0, scale=1)
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();

        // Median = loc = 0
        let median = cauchy.median();
        assert_eq!(median, 0.0);

        // Mode = loc = 0
        let mode = cauchy.mode();
        assert_eq!(mode, 0.0);

        // IQR = 2 * scale = 2
        let iqr = cauchy.iqr();
        assert_eq!(iqr, 2.0);

        // Entropy = log(4π) ≈ 2.5310242
        let entropy = cauchy.entropy();
        assert_relative_eq!(entropy, 2.5310242, epsilon = 1e-7);

        // Custom Cauchy with loc=-2, scale=0.5
        let custom = Cauchy::new(-2.0, 0.5).unwrap();

        // Median = loc = -2
        let median_custom = custom.median();
        assert_eq!(median_custom, -2.0);

        // Mode = loc = -2
        let mode_custom = custom.mode();
        assert_eq!(mode_custom, -2.0);

        // IQR = 2 * scale = 1
        let iqr_custom = custom.iqr();
        assert_eq!(iqr_custom, 1.0);

        // Entropy = log(4π*0.5) ≈ log(4π) - log(2) ≈ 2.5310242 - 0.6931472 ≈ 1.837877
        let entropy_custom = custom.entropy();
        assert_relative_eq!(entropy_custom, 1.837877, epsilon = 1e-6);
    }

    #[test]
    fn test_cauchy_rvs() {
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();

        // Generate samples
        let samples_vec = cauchy.rvs_vec(100).unwrap();
        let samples = cauchy.rvs(100).unwrap();

        // Check the number of samples
        assert_eq!(samples_vec.len(), 100);
        assert_eq!(samples.len(), 100);

        // Basic statistical checks are not very meaningful for Cauchy distribution
        // since the mean and variance are undefined
        // But we can check that the median is close to loc = 0

        // Calculate sample median for vector samples
        let mut sorted_samples = samples_vec.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_samples.len() % 2 == 0 {
            (sorted_samples[49] + sorted_samples[50]) / 2.0
        } else {
            sorted_samples[50]
        };

        // Median could be far from 0 due to extreme values, but generally should be within ±5
        // This is a very loose test due to the heavy-tailed nature of the Cauchy
        assert!(median.abs() < 5.0);
    }

    #[test]
    fn test_cauchy_inverse_cdf() {
        // Test that cdf(ppf(p)) == p and ppf(cdf(x)) == x
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();

        // Test various probability values
        let probabilities = [0.1, 0.25, 0.5, 0.75, 0.9];
        for &p in &probabilities {
            let x = cauchy.ppf(p).unwrap();
            let p_back = cauchy.cdf(x);
            assert_relative_eq!(p_back, p, epsilon = 1e-7);
        }

        // Test various x values
        let x_values = [-3.0, -1.0, 0.0, 1.0, 3.0];
        for &x in &x_values {
            let p = cauchy.cdf(x);
            let x_back = cauchy.ppf(p).unwrap();
            assert_relative_eq!(x_back, x, epsilon = 1e-7);
        }
    }
}
