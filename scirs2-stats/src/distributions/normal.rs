//! Normal distribution functions
//!
//! This module provides functionality for the Normal (Gaussian) distribution.

use crate::error::{StatsError, StatsResult};
use crate::error_messages::{helpers, validation};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousDistribution, Distribution};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Distribution as RandDistribution, Normal as RandNormal};

/// Normal distribution structure
pub struct Normal<F: Float> {
    /// Mean (location) parameter
    pub loc: F,
    /// Standard deviation (scale) parameter
    pub scale: F,
    /// Random number generator for this distribution
    rand_distr: RandNormal<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Normal<F> {
    /// Create a new normal distribution with given mean and standard deviation
    ///
    /// # Arguments
    ///
    /// * `loc` - Mean (location) parameter
    /// * `scale` - Standard deviation (scale) parameter
    ///
    /// # Returns
    ///
    /// * A new Normal distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::normal::Normal;
    ///
    /// let norm = Normal::new(0.0f64, 1.0).unwrap();
    /// ```
    pub fn new(loc: F, scale: F) -> StatsResult<Self> {
        // Validate scale parameter
        validation::ensure_positive(scale, "scale")?;

        // Convert to f64 for rand_distr
        let loc_f64 = <f64 as NumCast>::from(loc)
            .ok_or_else(|| helpers::numerical_error("failed to convert loc to f64"))?;
        let scale_f64 = <f64 as NumCast>::from(scale)
            .ok_or_else(|| helpers::numerical_error("failed to convert scale to f64"))?;

        match RandNormal::new(loc_f64, scale_f64) {
            Ok(rand_distr) => Ok(Normal {
                loc,
                scale,
                rand_distr,
            }),
            Err(_) => Err(helpers::numerical_error("normal distribution creation")),
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
    /// use scirs2_stats::distributions::normal::Normal;
    ///
    /// let norm = Normal::new(0.0f64, 1.0).unwrap();
    /// let pdf_at_zero = norm.pdf(0.0);
    /// assert!((pdf_at_zero - 0.3989423).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // PDF = (1 / (scale * sqrt(2*pi))) * exp(-0.5 * ((x-loc)/scale)^2)
        let pi = F::from(std::f64::consts::PI).unwrap_or_else(|| F::zero());
        let two = F::from(2.0).unwrap_or_else(|| F::zero());

        let z = (x - self.loc) / self.scale;
        let exponent = -z * z / two;

        F::from(1.0).unwrap_or_else(|| F::zero()) / (self.scale * (two * pi).sqrt())
            * exponent.exp()
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
    /// use scirs2_stats::distributions::normal::Normal;
    ///
    /// let norm = Normal::new(0.0f64, 1.0).unwrap();
    /// let cdf_at_zero = norm.cdf(0.0);
    /// assert!((cdf_at_zero - 0.5).abs() < 1e-10);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        // Standardize the variable
        let z = (x - self.loc) / self.scale;

        // For standard normal CDF at 0, the result should be exactly 0.5
        if z == F::zero() {
            return F::from(0.5).unwrap_or_else(|| F::zero());
        }

        // Use a standard implementation of the error function
        // CDF = 0.5 * (1 + erf(z / sqrt(2)))
        let two = F::from(2.0).unwrap_or_else(|| F::zero());
        let one = F::one();
        let half = F::from(0.5).unwrap_or_else(|| F::zero());

        half * (one + erf(z / two.sqrt()))
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
    /// use scirs2_stats::distributions::normal::Normal;
    ///
    /// let norm = Normal::new(0.0f64, 1.0).unwrap();
    /// let x = norm.ppf(0.975).unwrap();
    /// assert!((x - 1.96).abs() < 1e-2);
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

        // Use Abramowitz and Stegun approximation for the inverse standard normal CDF
        // We'll use a more accurate approximation than just inverse_erf

        let half = F::from(0.5).unwrap_or_else(|| F::zero());

        // Coefficients for approximation (shared between both branches)
        let c0 = F::from(2.515517).unwrap_or_else(|| F::zero());
        let c1 = F::from(0.802853).unwrap_or_else(|| F::zero());
        let c2 = F::from(0.010328).unwrap_or_else(|| F::zero());
        let d1 = F::from(1.432788).unwrap_or_else(|| F::zero());
        let d2 = F::from(0.189269).unwrap_or_else(|| F::zero());
        let d3 = F::from(0.001308).unwrap_or_else(|| F::zero());

        let z = if p <= half {
            // Lower region
            let q = p;
            let t = (-F::from(2.0).unwrap_or_else(|| F::zero()) * q.ln()).sqrt();
            -t + (c0 + c1 * t + c2 * t * t) / (F::one() + d1 * t + d2 * t * t + d3 * t * t * t)
        } else {
            // Upper region
            let q = F::one() - p;
            let t = (-F::from(2.0).unwrap_or_else(|| F::zero()) * q.ln()).sqrt();
            t - (c0 + c1 * t + c2 * t * t) / (F::one() + d1 * t + d2 * t * t + d3 * t * t * t)
        };

        // Scale and shift to get the quantile for the given parameters
        Ok(z * self.scale + self.loc)
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
    /// use scirs2_stats::distributions::normal::Normal;
    ///
    /// let norm = Normal::new(0.0f64, 1.0).unwrap();
    /// let samples = norm.rvs(1000).unwrap();
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

/// Calculate the error function (erf)
#[allow(dead_code)]
fn erf<F: Float>(x: F) -> F {
    // Approximation based on Abramowitz and Stegun
    let zero = F::zero();
    let one = F::one();

    // Handle negative values using erf(-x) = -erf(x)
    if x < zero {
        return -erf(-x);
    }

    // Constants for the approximation
    let a1 = F::from(0.254829592).unwrap();
    let a2 = F::from(-0.284496736).unwrap();
    let a3 = F::from(1.421413741).unwrap();
    let a4 = F::from(-1.453152027).unwrap();
    let a5 = F::from(1.061405429).unwrap();
    let p = F::from(0.3275911).unwrap();

    // Calculate the approximation
    let t = one / (one + p * x);
    one - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp()
}

// The inverse_erf function has been replaced with a more accurate
// approximation directly in the ppf method

// Implement the Distribution trait for Normal
impl<F: Float + NumCast + std::fmt::Display> Distribution<F> for Normal<F> {
    fn mean(&self) -> F {
        self.loc
    }

    fn var(&self) -> F {
        self.scale * self.scale
    }

    fn std(&self) -> F {
        self.scale
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        Normal::rvs(self, size)
    }

    fn entropy(&self) -> F {
        let half = F::from(0.5).unwrap();
        let two = F::from(2.0).unwrap();
        let pi = F::from(std::f64::consts::PI).unwrap();
        let e = F::from(std::f64::consts::E).unwrap();

        half + half * (two * pi * e * self.scale * self.scale).ln()
    }
}

// Implement the ContinuousDistribution trait for Normal
impl<F: Float + NumCast + std::fmt::Display> ContinuousDistribution<F> for Normal<F> {
    fn pdf(&self, x: F) -> F {
        Normal::pdf(self, x)
    }

    fn cdf(&self, x: F) -> F {
        Normal::cdf(self, x)
    }

    fn ppf(&self, p: F) -> StatsResult<F> {
        Normal::ppf(self, p)
    }
}

/// Implementation of SampleableDistribution for Normal
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Normal<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let array = Normal::rvs(self, size)?;
        Ok(array.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_creation() {
        // Standard normal
        let norm = Normal::new(0.0, 1.0).unwrap();
        assert_eq!(norm.loc, 0.0);
        assert_eq!(norm.scale, 1.0);

        // Custom normal
        let custom = Normal::new(5.0, 2.0).unwrap();
        assert_eq!(custom.loc, 5.0);
        assert_eq!(custom.scale, 2.0);

        // Error cases
        assert!(Normal::<f64>::new(0.0, 0.0).is_err());
        assert!(Normal::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_normal_pdf() {
        // Standard normal PDF values
        let norm = Normal::new(0.0, 1.0).unwrap();

        // PDF at x = 0
        let pdf_at_zero = norm.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 0.3989423, epsilon = 1e-7);

        // PDF at x = 1
        let pdf_at_one = norm.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.2419707, epsilon = 1e-7);

        // PDF at x = -1
        let pdf_at_neg_one = norm.pdf(-1.0);
        assert_relative_eq!(pdf_at_neg_one, 0.2419707, epsilon = 1e-7);

        // Custom normal
        let custom = Normal::new(5.0, 2.0).unwrap();
        assert_relative_eq!(custom.pdf(5.0), 0.19947114, epsilon = 1e-7);
    }

    #[test]
    fn test_normal_cdf() {
        // Standard normal CDF values
        let norm = Normal::new(0.0, 1.0).unwrap();

        // CDF at x = 0
        let cdf_at_zero = norm.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.5, epsilon = 1e-7);

        // CDF at x = 1
        let cdf_at_one = norm.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.8413447, epsilon = 1e-5);

        // CDF at x = -1
        let cdf_at_neg_one = norm.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 0.1586553, epsilon = 1e-5);
    }

    #[test]
    fn test_normal_ppf() {
        // Standard normal quantiles
        let norm = Normal::new(0.0, 1.0).unwrap();

        // Median (50th percentile)
        let median = norm.ppf(0.5).unwrap();
        assert_relative_eq!(median, 0.0, epsilon = 1e-5);

        // 97.5th percentile (often used for confidence intervals)
        let p975 = norm.ppf(0.975).unwrap();
        assert_relative_eq!(p975, 1.96, epsilon = 1e-2);

        // 2.5th percentile
        let p025 = norm.ppf(0.025).unwrap();
        assert_relative_eq!(p025, -1.96, epsilon = 1e-2);

        // Error cases
        assert!(norm.ppf(-0.1).is_err());
        assert!(norm.ppf(1.1).is_err());
    }

    #[test]
    fn test_normal_rvs() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        // Generate samples
        let samples = norm.rvs(1000).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to 0 (with more generous tolerance for random variation)
        // With 1000 samples, 99.7% confidence interval is roughly ±0.1
        assert!(
            mean.abs() < 0.15,
            "Sample mean {} is outside expected range",
            mean
        );

        // Standard deviation check
        let variance: f64 = samples
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / 1000.0;
        let std_dev = variance.sqrt();

        // Std dev should be close to 1 (with tolerance for random variation)
        assert!(
            (std_dev - 1.0).abs() < 0.15,
            "Sample std dev {} is outside expected range",
            std_dev
        );
    }
}
