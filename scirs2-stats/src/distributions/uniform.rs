//! Uniform distribution functions
//!
//! This module provides functionality for the Uniform distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousDistribution, Distribution};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Distribution as RandDistribution, Uniform as RandUniform};

/// Uniform distribution structure
pub struct Uniform<F: Float> {
    /// Lower bound (inclusive)
    pub low: F,
    /// Upper bound (exclusive)
    pub high: F,
    /// Random number generator for this distribution
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Uniform<F> {
    /// Create a new uniform distribution with given bounds
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive)
    ///
    /// # Returns
    ///
    /// * A new Uniform distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::uniform::Uniform;
    ///
    /// let unif = Uniform::new(0.0f64, 1.0).unwrap();
    /// ```
    pub fn new(low: F, high: F) -> StatsResult<Self> {
        if low >= high {
            return Err(StatsError::DomainError(
                "Lower bound must be less than upper bound".to_string(),
            ));
        }

        // Convert to f64 for rand_distr
        let low_f64 = <f64 as NumCast>::from(low).unwrap();
        let high_f64 = <f64 as NumCast>::from(high).unwrap();

        match RandUniform::new_inclusive(low_f64, high_f64) {
            Ok(rand_distr) => Ok(Uniform {
                low,
                high,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create uniform distribution".to_string(),
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
    /// use scirs2_stats::distributions::uniform::Uniform;
    ///
    /// let unif = Uniform::new(0.0f64, 2.0).unwrap();
    /// let pdf_at_one = unif.pdf(1.0);
    /// assert!((pdf_at_one - 0.5).abs() < 1e-10);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // PDF is 1/(high-low) for x in [low, high), 0 otherwise
        if x >= self.low && x < self.high {
            F::one() / (self.high - self.low)
        } else {
            F::zero()
        }
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
    /// use scirs2_stats::distributions::uniform::Uniform;
    ///
    /// let unif = Uniform::new(0.0f64, 1.0).unwrap();
    /// let cdf_at_half = unif.cdf(0.5);
    /// assert!((cdf_at_half - 0.5).abs() < 1e-10);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        if x <= self.low {
            F::zero()
        } else if x >= self.high {
            F::one()
        } else {
            (x - self.low) / (self.high - self.low)
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
    /// * The value x such that CDF(x) = p
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::uniform::Uniform;
    ///
    /// let unif = Uniform::new(0.0f64, 1.0).unwrap();
    /// let x = unif.ppf(0.75).unwrap();
    /// assert!((x - 0.75).abs() < 1e-10);
    /// ```
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // Quantile function is the inverse of CDF: low + p*(high-low)
        Ok(self.low + p * (self.high - self.low))
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
    /// use scirs2_stats::distributions::uniform::Uniform;
    ///
    /// let unif = Uniform::new(0.0f64, 1.0).unwrap();
    /// let samples = unif.rvs(1000).unwrap();
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            let sample = self.rand_distr.sample(&mut rng);
            samples.push(F::from(sample).unwrap());
        }

        Ok(Array1::from(samples))
    }
}

// Implement the Distribution trait for Uniform
impl<F: Float + NumCast + std::fmt::Display> Distribution<F> for Uniform<F> {
    fn mean(&self) -> F {
        (self.low + self.high) / F::from(2.0).unwrap()
    }

    fn var(&self) -> F {
        let range = self.high - self.low;
        range * range / F::from(12.0).unwrap()
    }

    fn std(&self) -> F {
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        (self.high - self.low).ln()
    }
}

// Implement the ContinuousDistribution trait for Uniform
impl<F: Float + NumCast + std::fmt::Display> ContinuousDistribution<F> for Uniform<F> {
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

/// Implementation of SampleableDistribution for Uniform
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Uniform<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let array = self.rvs(size)?;
        Ok(array.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_creation() {
        // Standard uniform
        let unif = Uniform::new(0.0, 1.0).unwrap();
        assert_eq!(unif.low, 0.0);
        assert_eq!(unif.high, 1.0);

        // Custom uniform
        let custom = Uniform::new(-1.0, 1.0).unwrap();
        assert_eq!(custom.low, -1.0);
        assert_eq!(custom.high, 1.0);

        // Error cases
        assert!(Uniform::<f64>::new(0.0, 0.0).is_err());
        assert!(Uniform::<f64>::new(1.0, 0.0).is_err());
    }

    #[test]
    fn test_uniform_pdf() {
        // Standard uniform PDF values
        let unif = Uniform::new(0.0, 1.0).unwrap();

        // PDF at x in range
        let pdf_in_range = unif.pdf(0.5);
        assert_relative_eq!(pdf_in_range, 1.0, epsilon = 1e-10);

        // PDF at x = low (inclusive)
        let pdf_at_low = unif.pdf(0.0);
        assert_relative_eq!(pdf_at_low, 1.0, epsilon = 1e-10);

        // PDF at x = high (exclusive)
        let pdf_at_high = unif.pdf(1.0);
        assert_relative_eq!(pdf_at_high, 0.0, epsilon = 1e-10);

        // PDF outside range
        let pdf_outside = unif.pdf(2.0);
        assert_relative_eq!(pdf_outside, 0.0, epsilon = 1e-10);

        // Non-unit range
        let unif2 = Uniform::new(0.0, 2.0).unwrap();
        let pdf2 = unif2.pdf(1.0);
        assert_relative_eq!(pdf2, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_cdf() {
        // Standard uniform CDF values
        let unif = Uniform::new(0.0, 1.0).unwrap();

        // CDF at midpoint
        let cdf_mid = unif.cdf(0.5);
        assert_relative_eq!(cdf_mid, 0.5, epsilon = 1e-10);

        // CDF at x = low
        let cdf_at_low = unif.cdf(0.0);
        assert_relative_eq!(cdf_at_low, 0.0, epsilon = 1e-10);

        // CDF at x = high
        let cdf_at_high = unif.cdf(1.0);
        assert_relative_eq!(cdf_at_high, 1.0, epsilon = 1e-10);

        // CDF outside range
        let cdf_below = unif.cdf(-1.0);
        assert_relative_eq!(cdf_below, 0.0, epsilon = 1e-10);

        let cdf_above = unif.cdf(2.0);
        assert_relative_eq!(cdf_above, 1.0, epsilon = 1e-10);

        // Non-unit range
        let unif2 = Uniform::new(-1.0, 1.0).unwrap();
        let cdf2 = unif2.cdf(0.0);
        assert_relative_eq!(cdf2, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_ppf() {
        // Standard uniform quantiles
        let unif = Uniform::new(0.0, 1.0).unwrap();

        // Median (50th percentile)
        let median = unif.ppf(0.5).unwrap();
        assert_relative_eq!(median, 0.5, epsilon = 1e-10);

        // 75th percentile
        let p75 = unif.ppf(0.75).unwrap();
        assert_relative_eq!(p75, 0.75, epsilon = 1e-10);

        // 25th percentile
        let p25 = unif.ppf(0.25).unwrap();
        assert_relative_eq!(p25, 0.25, epsilon = 1e-10);

        // Error cases
        assert!(unif.ppf(-0.1).is_err());
        assert!(unif.ppf(1.1).is_err());

        // Non-unit range
        let unif2 = Uniform::new(-1.0, 1.0).unwrap();
        let median2 = unif2.ppf(0.5).unwrap();
        assert_relative_eq!(median2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_rvs() {
        let unif = Uniform::new(0.0, 1.0).unwrap();

        // Generate samples
        let samples = unif.rvs(1000).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to 0.5 (within reason for random samples)
        assert!((mean - 0.5).abs() < 0.1);

        // Check that all values are within range
        for &sample in samples.iter() {
            assert!((0.0..=1.0).contains(&sample));
        }
    }
}
