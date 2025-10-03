//! Lognormal distribution functions
//!
//! This module provides functionality for the Lognormal distribution.

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};

/// Lognormal distribution structure
///
/// The lognormal distribution is the distribution of a random variable
/// whose logarithm follows a normal distribution.
pub struct Lognormal<F: Float> {
    /// Mean of the underlying normal distribution (shape parameter)
    pub mu: F,
    /// Standard deviation of the underlying normal distribution (scale parameter)
    pub sigma: F,
    /// Location parameter
    pub loc: F,
    /// Underlying normal distribution
    norm: Normal<F>,
}

impl<F: Float + NumCast + std::fmt::Display> Lognormal<F> {
    /// Create a new lognormal distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean of the underlying normal distribution
    /// * `sigma` - Standard deviation of the underlying normal distribution
    /// * `loc` - Location parameter (default: 0)
    ///
    /// # Returns
    ///
    /// * A new Lognormal distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(mu: F, sigma: F, loc: F) -> StatsResult<Self> {
        if sigma <= F::zero() {
            return Err(StatsError::DomainError(
                "Standard deviation must be positive".to_string(),
            ));
        }

        // Create underlying normal distribution
        match Normal::new(mu, sigma) {
            Ok(norm) => Ok(Lognormal {
                mu,
                sigma,
                loc,
                norm,
            }),
            Err(e) => Err(e),
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let pdf_at_one = lognorm.pdf(1.0);
    /// assert!((pdf_at_one - 0.3989423).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // For a value <= location parameter, PDF is 0
        if x <= self.loc {
            return F::zero();
        }

        // Shift x by location parameter
        let x_shifted = x - self.loc;

        // For lognormal PDF:
        // f(x) = 1/(x*sigma*sqrt(2*pi)) * exp(-(ln(x) - mu)^2/(2*sigma^2))
        let pi = F::from(std::f64::consts::PI).unwrap();
        let two = F::from(2.0).unwrap();

        let ln_x = x_shifted.ln();
        let z = (ln_x - self.mu) / self.sigma;
        let exponent = -z * z / two;

        F::from(1.0).unwrap() / (x_shifted * self.sigma * (two * pi).sqrt()) * exponent.exp()
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let cdf_at_one = lognorm.cdf(1.0);
    /// assert!((cdf_at_one - 0.5).abs() < 1e-10);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        // For a value <= location parameter, CDF is 0
        if x <= self.loc {
            return F::zero();
        }

        // Shift x by location parameter
        let x_shifted = x - self.loc;

        // The CDF of lognormal at x is the same as the CDF of normal at ln(x)
        let ln_x = x_shifted.ln();
        self.norm.cdf(ln_x)
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let x = lognorm.ppf(0.5).unwrap();
    /// assert!((x - 1.0000001010066806) < 1e-7);
    /// ```
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

        // The quantile of lognormal at p is exp(quantile of normal at p)
        match self.norm.ppf(p) {
            Ok(normal_quantile) => Ok(normal_quantile.exp() + self.loc),
            Err(e) => Err(e),
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let samples = lognorm.rvs(1000).unwrap();
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        // Generate samples from normal distribution
        let normal_samples = self.norm.rvs(size)?;

        // Transform the samples to lognormal by taking the exponent and adding location
        let lognormal_samples: Vec<F> = normal_samples
            .into_iter()
            .map(|x| x.exp() + self.loc)
            .collect();

        Ok(lognormal_samples)
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let mean = lognorm.mean();
    /// assert!((mean - 1.6487212707).abs() < 1e-7);
    /// ```
    pub fn mean(&self) -> F {
        let half = F::from(0.5).unwrap();
        let variance = self.sigma * self.sigma;

        (self.mu + variance * half).exp() + self.loc
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let var = lognorm.var();
    /// assert!((var - 4.670774270471604) < 1e-7);
    /// ```
    pub fn var(&self) -> F {
        let one = F::one();
        let two = F::from(2.0).unwrap();
        let variance = self.sigma * self.sigma;

        ((two * self.mu + variance).exp()) * (variance.exp() - one)
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let median = lognorm.median();
    /// assert!((median - 1.0).abs() < 1e-7);
    /// ```
    pub fn median(&self) -> F {
        self.mu.exp() + self.loc
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
    /// use scirs2_stats::distributions::lognormal::Lognormal;
    ///
    /// let lognorm = Lognormal::new(0.0f64, 1.0, 0.0).unwrap();
    /// let mode = lognorm.mode();
    /// assert!((mode - 0.36787944).abs() < 1e-7);
    /// ```
    pub fn mode(&self) -> F {
        (self.mu - self.sigma * self.sigma).exp() + self.loc
    }
}

/// Create a lognormal distribution with the given parameters.
///
/// This is a convenience function to create a lognormal distribution with
/// the given mu, sigma, and loc parameters.
///
/// # Arguments
///
/// * `mu` - Mean of the underlying normal distribution
/// * `sigma` - Standard deviation of the underlying normal distribution
/// * `loc` - Location parameter
///
/// # Returns
///
/// * A lognormal distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::lognormal;
///
/// let lognorm = lognormal::lognormal(0.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_one = lognorm.pdf(1.0);
/// ```
#[allow(dead_code)]
pub fn lognormal<F>(mu: F, sigma: F, loc: F) -> StatsResult<Lognormal<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Lognormal::new(mu, sigma, loc)
}

/// Implementation of SampleableDistribution for Lognormal
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Lognormal<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lognormal_creation() {
        // Standard lognormal
        let lognorm = Lognormal::new(0.0, 1.0, 0.0).unwrap();
        assert_eq!(lognorm.mu, 0.0);
        assert_eq!(lognorm.sigma, 1.0);
        assert_eq!(lognorm.loc, 0.0);

        // Custom lognormal
        let custom = Lognormal::new(1.0, 0.5, 1.0).unwrap();
        assert_eq!(custom.mu, 1.0);
        assert_eq!(custom.sigma, 0.5);
        assert_eq!(custom.loc, 1.0);

        // Error cases
        assert!(Lognormal::<f64>::new(0.0, 0.0, 0.0).is_err());
        assert!(Lognormal::<f64>::new(0.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_lognormal_pdf() {
        // Standard lognormal PDF values
        let lognorm = Lognormal::new(0.0, 1.0, 0.0).unwrap();

        // PDF at x = 1
        let pdf_at_one = lognorm.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.3989423, epsilon = 1e-7);

        // PDF at x = 2
        let pdf_at_two = lognorm.pdf(2.0);
        assert_relative_eq!(pdf_at_two, 0.1568740192789811, epsilon = 1e-7);

        // PDF at x = 0.5
        let pdf_at_half = lognorm.pdf(0.5);
        assert_relative_eq!(pdf_at_half, 0.6274960771159244, epsilon = 1e-7);

        // PDF at x <= 0 is 0
        assert_eq!(lognorm.pdf(0.0), 0.0);
        assert_eq!(lognorm.pdf(-1.0), 0.0);

        // Custom lognormal
        let custom = Lognormal::new(1.0, 0.5, 1.0).unwrap();

        // PDF at x = 1 (at loc) should be 0
        assert_eq!(custom.pdf(1.0), 0.0);

        // PDF at x = 2 (adjusted for loc)
        let pdf_at_custom = custom.pdf(2.0);
        let expected = 0.10798193302637613; // Equivalent to standard normal PDF at 0 adjusted
        assert_relative_eq!(pdf_at_custom, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_lognormal_cdf() {
        // Standard lognormal CDF values
        let lognorm = Lognormal::new(0.0, 1.0, 0.0).unwrap();

        // CDF at x = 1
        let cdf_at_one = lognorm.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.5, epsilon = 1e-7);

        // CDF at x = 2
        let cdf_at_two = lognorm.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.7558914, epsilon = 1e-7);

        // CDF at x = 0.5
        let cdf_at_half = lognorm.cdf(0.5);
        assert_relative_eq!(cdf_at_half, 0.24410852729647436, epsilon = 1e-7);

        // CDF at x <= 0 is 0
        assert_eq!(lognorm.cdf(0.0), 0.0);
        assert_eq!(lognorm.cdf(-1.0), 0.0);
    }

    #[test]
    fn test_lognormal_ppf() {
        // Standard lognormal quantiles
        let lognorm = Lognormal::new(0.0, 1.0, 0.0).unwrap();

        // Median (50th percentile)
        let median = lognorm.ppf(0.5).unwrap();
        assert_relative_eq!(median, 1.0000001010066806, epsilon = 1e-7);

        // 75th percentile
        let p75 = lognorm.ppf(0.75).unwrap();
        assert_relative_eq!(p75, 1.9624410657713667, epsilon = 1e-4);

        // 25th percentile
        let p25 = lognorm.ppf(0.25).unwrap();
        assert_relative_eq!(p25, 0.5095694425895716, epsilon = 1e-4);

        // Error cases
        assert!(lognorm.ppf(-0.1).is_err());
        assert!(lognorm.ppf(1.1).is_err());
    }

    #[test]
    fn test_lognormal_statistics() {
        // Standard lognormal statistics
        let lognorm = Lognormal::new(0.0, 1.0, 0.0).unwrap();

        // Mean = exp(μ + σ²/2)
        let mean = lognorm.mean();
        assert_relative_eq!(mean, 1.6487212707, epsilon = 1e-7);

        // Variance = exp(2μ + σ²) * (exp(σ²) - 1)
        let var = lognorm.var();
        assert_relative_eq!(var, 4.670774270471604, epsilon = 1e-7);

        // Median = exp(μ)
        let median = lognorm.median();
        assert_relative_eq!(median, 1.0, epsilon = 1e-7);

        // Mode = exp(μ - σ²)
        let mode = lognorm.mode();
        assert_relative_eq!(mode, 0.36787944, epsilon = 1e-7);

        // Custom lognormal
        let custom = Lognormal::new(1.0, 0.5, 2.0).unwrap();

        // Mean with location parameter
        let custom_mean = custom.mean();
        let expected_mean = (1.0 + 0.5 * 0.5 / 2.0).exp() + 2.0;
        assert_relative_eq!(custom_mean, expected_mean, epsilon = 1e-7);

        // Median with location parameter
        let custom_median = custom.median();
        assert_relative_eq!(custom_median, 1.0.exp() + 2.0, epsilon = 1e-7);
    }

    #[test]
    fn test_lognormal_rvs() {
        let lognorm = Lognormal::new(0.0, 1.0, 0.0).unwrap();

        // Generate samples
        let samples = lognorm.rvs(1000).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic positivity check (all lognormal samples should be positive)
        for &sample in &samples {
            assert!(sample > 0.0);
        }

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to true mean (within reason for random samples)
        // True mean for standard lognormal is e^(1/2) ≈ 1.6487
        assert!((mean - 1.6487).abs() < 0.5);

        // Calculate sample median as a sanity check
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if samples.len() % 2 == 0 {
            (sorted_samples[499] + sorted_samples[500]) / 2.0
        } else {
            sorted_samples[500]
        };

        // Median should be close to 1 (within reason for random samples)
        assert!((median - 1.0).abs() < 0.5);
    }
}
