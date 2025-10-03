//! Pareto distribution functions
//!
//! This module provides functionality for the Pareto distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Uniform as RandUniform};

/// Pareto distribution structure
///
/// The Pareto distribution is a power-law probability distribution that is used to
/// model many types of observable phenomena, particularly those exhibiting the
/// "80-20 rule" (Pareto principle).
pub struct Pareto<F: Float> {
    /// Shape parameter (alpha > 0)
    pub shape: F,
    /// Scale parameter (x_m > 0)
    pub scale: F,
    /// Location parameter (default: 0)
    pub loc: F,
    /// Random number generator for uniform distribution
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Pareto<F> {
    /// Create a new Pareto distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter (alpha > 0)
    /// * `scale` - Scale parameter (x_m > 0)
    /// * `loc` - Location parameter (default: 0)
    ///
    /// # Returns
    ///
    /// * A new Pareto distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(shape: F, scale: F, loc: F) -> StatsResult<Self> {
        // Validate parameters
        if shape <= F::zero() {
            return Err(StatsError::DomainError(
                "Shape parameter must be positive".to_string(),
            ));
        }

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

        Ok(Pareto {
            shape,
            scale,
            loc,
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let pdf_at_two = pareto.pdf(2.0);
    /// assert!((pdf_at_two - 0.1875).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // Adjust x by location parameter
        let x_adjusted = x - self.loc;

        // For x <= scale + loc, PDF is 0
        if x_adjusted <= self.scale {
            return F::zero();
        }

        // PDF = (shape / scale) * (scale / (x - loc))^(shape + 1)
        let ratio = self.scale / x_adjusted;
        let shape_plus_one = self.shape + F::one();
        self.shape / self.scale * ratio.powf(shape_plus_one)
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let cdf_at_two = pareto.cdf(2.0);
    /// assert!((cdf_at_two - 0.875).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        // Adjust x by location parameter
        let x_adjusted = x - self.loc;

        // For x <= scale + loc, CDF is 0
        if x_adjusted <= self.scale {
            return F::zero();
        }

        // CDF = 1 - (scale / (x - loc))^shape
        let ratio = self.scale / x_adjusted;
        F::one() - ratio.powf(self.shape)
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let x = pareto.ppf(0.5).unwrap();
    /// assert!((x - 1.2599210).abs() < 1e-6);
    /// ```
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // Special cases
        if p == F::zero() {
            return Ok(self.scale + self.loc);
        }
        if p == F::one() {
            return Ok(F::infinity());
        }

        // Compute the quantile directly from the formula
        // x = scale / (1 - p)^(1/shape) + loc
        let one = F::one();
        let one_minus_p = one - p;
        let pow = one_minus_p.powf(one / self.shape);
        let quantile = self.scale / pow + self.loc;

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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let samples = pareto.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);

        // Generate samples using the inverse transform sampling method
        for _ in 0..size {
            // Generate uniform random number in [0, 1)
            let u = self.rand_distr.sample(&mut rng);
            let u_f = F::from(u).unwrap();

            // Apply inverse CDF transform
            let sample = self.ppf(u_f)?;
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let mean = pareto.mean();
    /// assert!((mean - 1.5).abs() < 1e-7);
    /// ```
    pub fn mean(&self) -> F {
        // Mean is only defined for shape > 1
        if self.shape <= F::one() {
            return F::infinity();
        }

        // Mean = (shape * scale) / (shape - 1) + loc
        let shape_minus_one = self.shape - F::one();
        (self.shape * self.scale) / shape_minus_one + self.loc
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let var = pareto.var();
    /// assert!((var - 0.75).abs() < 1e-7);
    /// ```
    pub fn var(&self) -> F {
        // Variance is only defined for shape > 2
        if self.shape <= F::from(2.0).unwrap() {
            return F::infinity();
        }

        // Variance = (scale^2 * shape) / ((shape - 1)^2 * (shape - 2))
        let one = F::one();
        let two = F::from(2.0).unwrap();
        let shape_minus_one = self.shape - one;
        let shape_minus_two = self.shape - two;

        (self.scale * self.scale * self.shape)
            / (shape_minus_one * shape_minus_one * shape_minus_two)
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let median = pareto.median();
    /// assert!((median - 1.2599210).abs() < 1e-6);
    /// ```
    pub fn median(&self) -> F {
        // Median = scale * 2^(1/shape) + loc
        let two = F::from(2.0).unwrap();
        let two_pow = two.powf(F::one() / self.shape);
        self.scale * two_pow + self.loc
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
    /// use scirs2_stats::distributions::pareto::Pareto;
    ///
    /// let pareto = Pareto::new(3.0f64, 1.0, 0.0).unwrap();
    /// let mode = pareto.mode();
    /// assert!((mode - 1.0).abs() < 1e-7);
    /// ```
    pub fn mode(&self) -> F {
        // Mode = scale + loc
        self.scale + self.loc
    }
}

/// Create a Pareto distribution with the given parameters.
///
/// This is a convenience function to create a Pareto distribution with
/// the given shape, scale, and location parameters.
///
/// # Arguments
///
/// * `shape` - Shape parameter (alpha > 0)
/// * `scale` - Scale parameter (x_m > 0)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Pareto distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::pareto;
///
/// let p = pareto::pareto(3.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_two = p.pdf(2.0);
/// assert!((pdf_at_two - 0.1875).abs() < 1e-7);
/// ```
#[allow(dead_code)]
pub fn pareto<F>(shape: F, scale: F, loc: F) -> StatsResult<Pareto<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Pareto::new(shape, scale, loc)
}

/// Implementation of SampleableDistribution for Pareto
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Pareto<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pareto_creation() {
        // Standard Pareto (shape=1, scale=1, loc=0)
        let pareto1 = Pareto::new(1.0, 1.0, 0.0).unwrap();
        assert_eq!(pareto1.shape, 1.0);
        assert_eq!(pareto1.scale, 1.0);
        assert_eq!(pareto1.loc, 0.0);

        // Pareto with shape=3
        let pareto3 = Pareto::new(3.0, 1.0, 0.0).unwrap();
        assert_eq!(pareto3.shape, 3.0);
        assert_eq!(pareto3.scale, 1.0);
        assert_eq!(pareto3.loc, 0.0);

        // Custom Pareto
        let custom = Pareto::new(2.5, 2.0, 1.0).unwrap();
        assert_eq!(custom.shape, 2.5);
        assert_eq!(custom.scale, 2.0);
        assert_eq!(custom.loc, 1.0);

        // Error cases
        assert!(Pareto::<f64>::new(0.0, 1.0, 0.0).is_err());
        assert!(Pareto::<f64>::new(-1.0, 1.0, 0.0).is_err());
        assert!(Pareto::<f64>::new(1.0, 0.0, 0.0).is_err());
        assert!(Pareto::<f64>::new(1.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_pareto_pdf() {
        // Pareto with shape=3, scale=1, loc=0
        let pareto = Pareto::new(3.0, 1.0, 0.0).unwrap();

        // PDF at x = 1 should be shape = 3.0
        let pdf_at_one = pareto.pdf(1.0);
        assert_eq!(pdf_at_one, 0.0); // At x=scale, PDF is 0

        // PDF at x = 2 should be 3 * (1/2)^4 = 3 * 1/16 = 0.1875
        let pdf_at_two = pareto.pdf(2.0);
        assert_relative_eq!(pdf_at_two, 0.1875, epsilon = 1e-7);

        // PDF at values less than scale should be 0
        let pdf_at_half = pareto.pdf(0.5);
        assert_eq!(pdf_at_half, 0.0);

        // Custom Pareto with location
        let custom = Pareto::new(3.0, 1.0, 1.0).unwrap();

        // PDF at x = 2 (with loc=1) should equal pareto.pdf(1.0)
        let pdf_at_two_loc = custom.pdf(2.0);
        assert_eq!(pdf_at_two_loc, 0.0);

        // PDF at x = 3 (with loc=1) should equal pareto.pdf(2.0)
        let pdf_at_three_loc = custom.pdf(3.0);
        assert_relative_eq!(pdf_at_three_loc, 0.1875, epsilon = 1e-7);
    }

    #[test]
    fn test_pareto_cdf() {
        // Pareto with shape=3, scale=1, loc=0
        let pareto = Pareto::new(3.0, 1.0, 0.0).unwrap();

        // CDF at x = 1 should be 0
        let cdf_at_one = pareto.cdf(1.0);
        assert_eq!(cdf_at_one, 0.0);

        // CDF at x = 2 should be 1 - (1/2)^3 = 1 - 1/8 = 0.875
        let cdf_at_two = pareto.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.875, epsilon = 1e-7);

        // CDF at values less than scale should be 0
        let cdf_at_half = pareto.cdf(0.5);
        assert_eq!(cdf_at_half, 0.0);

        // Custom Pareto with location
        let custom = Pareto::new(3.0, 1.0, 1.0).unwrap();

        // CDF at x = 2 (with loc=1) should equal pareto.cdf(1.0)
        let cdf_at_two_loc = custom.cdf(2.0);
        assert_eq!(cdf_at_two_loc, 0.0);

        // CDF at x = 3 (with loc=1) should equal pareto.cdf(2.0)
        let cdf_at_three_loc = custom.cdf(3.0);
        assert_relative_eq!(cdf_at_three_loc, 0.875, epsilon = 1e-7);
    }

    #[test]
    fn test_pareto_ppf() {
        // Pareto with shape=3, scale=1, loc=0
        let pareto = Pareto::new(3.0, 1.0, 0.0).unwrap();

        // PPF at p = 0 should be scale = 1.0
        let ppf_at_zero = pareto.ppf(0.0).unwrap();
        assert_eq!(ppf_at_zero, 1.0);

        // PPF at p = 0.5 should be scale / (1 - 0.5)^(1/shape) = 1 / 0.5^(1/3) ≈ 1.2599210
        let ppf_at_half = pareto.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half, 1.2599210, epsilon = 1e-6);

        // PPF at p = 0.875 should be close to 2.0 (inverse of CDF at x = 2.0)
        let ppf_at_875 = pareto.ppf(0.875).unwrap();
        assert_relative_eq!(ppf_at_875, 2.0, epsilon = 1e-6);

        // Custom Pareto with location
        let custom = Pareto::new(3.0, 1.0, 1.0).unwrap();

        // PPF at p = 0.5 should be pareto.ppf(0.5) + loc
        let ppf_at_half_loc = custom.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half_loc, ppf_at_half + 1.0, epsilon = 1e-6);

        // Error cases
        assert!(pareto.ppf(-0.1).is_err());
        assert!(pareto.ppf(1.1).is_err());
    }

    #[test]
    fn test_pareto_statistics() {
        // Pareto with shape=3, scale=1, loc=0
        let pareto = Pareto::new(3.0, 1.0, 0.0).unwrap();

        // Mean should be (shape * scale) / (shape - 1) = 3 / 2 = 1.5
        let mean = pareto.mean();
        assert_relative_eq!(mean, 1.5, epsilon = 1e-7);

        // Variance should be (scale^2 * shape) / ((shape - 1)^2 * (shape - 2))
        // = 1^2 * 3 / (2^2 * 1) = 3 / 4 = 0.75
        let var = pareto.var();
        assert_relative_eq!(var, 0.75, epsilon = 1e-7);

        // Median should be scale * 2^(1/shape) = 1 * 2^(1/3) ≈ 1.2599210
        let median = pareto.median();
        assert_relative_eq!(median, 1.2599210, epsilon = 1e-6);

        // Mode should be scale = 1.0
        let mode = pareto.mode();
        assert_eq!(mode, 1.0);

        // Pareto with shape=1 (mean is not defined)
        let pareto1 = Pareto::new(1.0, 1.0, 0.0).unwrap();
        assert!(pareto1.mean().is_infinite());
        assert!(pareto1.var().is_infinite());

        // Pareto with shape=1.5 (variance is not defined but mean is)
        let pareto15 = Pareto::new(1.5, 1.0, 0.0).unwrap();
        assert!(!pareto15.mean().is_infinite());
        assert!(pareto15.var().is_infinite());

        // Custom Pareto with location
        let custom = Pareto::new(3.0, 1.0, 1.0).unwrap();

        // Mean should be pareto.mean() + loc
        let mean_loc = custom.mean();
        assert_relative_eq!(mean_loc, mean + 1.0, epsilon = 1e-7);

        // Variance should be same as pareto.var()
        let var_loc = custom.var();
        assert_relative_eq!(var_loc, var, epsilon = 1e-7);
    }

    #[test]
    fn test_pareto_rvs() {
        let pareto = Pareto::new(3.0, 1.0, 0.0).unwrap();

        // Generate samples
        let samples = pareto.rvs(1000).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic check (all samples should be >= scale)
        for &sample in &samples {
            assert!(sample >= 1.0);
        }

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to true mean (within reason for random samples)
        // True mean for Pareto(3, 1, 0) is 1.5
        assert!((mean - 1.5).abs() < 0.2);

        // Calculate sample median as a sanity check
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if samples.len() % 2 == 0 {
            (sorted_samples[499] + sorted_samples[500]) / 2.0
        } else {
            sorted_samples[500]
        };

        // Median should be close to 1.2599210 (within reason for random samples)
        assert!((median - 1.2599210).abs() < 0.2);
    }
}
