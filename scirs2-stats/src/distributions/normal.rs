//! Normal distribution functions
//!
//! This module provides functionality for the Normal (Gaussian) distribution.

use crate::error::{StatsError, StatsResult};
use crate::error_messages::{helpers, validation};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
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
    /// let norm = Normal::new(0.0f64, 1.0).expect("Operation failed");
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
    /// let norm = Normal::new(0.0f64, 1.0).expect("Operation failed");
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
    /// let norm = Normal::new(0.0f64, 1.0).expect("Operation failed");
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

        // CDF = erfc(-z / sqrt(2)) / 2, from `libm` (~1 ulp). The previous local
        // Abramowitz-Stegun 7.1.26 `erf` was only accurate to ~1.5e-7
        // absolute and computed the lower tail as `0.5 * (1 + erf)`, i.e.
        // with cancellation, so e.g. cdf(-10) came back as 0 instead of 7.6e-24.
        let z_f64 = match <f64 as NumCast>::from(z) {
            Some(value) => value,
            None => return F::nan(),
        };
        F::from(0.5 * libm::erfc(-z_f64 / std::f64::consts::SQRT_2)).unwrap_or_else(F::nan)
    }

    /// Survival function `P(X > x) = 1 - CDF(x)`, evaluated directly.
    ///
    /// Computing `1 - cdf(x)` loses every digit once the CDF rounds to 1
    /// (e.g. beyond ~8 standard deviations for a normal), so the upper tail is
    /// computed from its own closed or regularized form instead.
    pub fn sf(&self, x: F) -> F {
        let z = match <f64 as NumCast>::from((x - self.loc) / self.scale) {
            Some(value) => value,
            None => return F::nan(),
        };
        // P(Z > z) = erfc(z / sqrt(2)) / 2
        F::from(0.5 * libm::erfc(z / std::f64::consts::SQRT_2)).unwrap_or_else(F::nan)
    }

    /// Inverse survival function: the `x` with `sf(x) = q`.
    ///
    /// By symmetry `isf(q) = 2*loc - ppf(q)`, which keeps full precision for
    /// small `q` where `ppf(1 - q)` would round `1 - q` to 1.
    pub fn isf(&self, q: F) -> StatsResult<F> {
        let lower = self.ppf(q)?;
        Ok(self.loc + self.loc - lower)
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
    /// let norm = Normal::new(0.0f64, 1.0).expect("Operation failed");
    /// let x = norm.ppf(0.975).expect("Operation failed");
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

        // Acklam's rational approximation for the inverse standard normal CDF.
        // Maximum relative error ~1.15e-9 across the full range.
        let half = F::from(0.5).unwrap_or_else(|| F::zero());

        // Coefficients for the rational approximation
        let a1 = F::from(-3.969683028665376e+01).unwrap_or_else(|| F::zero());
        let a2 = F::from(2.209460984245205e+02).unwrap_or_else(|| F::zero());
        let a3 = F::from(-2.759285104469687e+02).unwrap_or_else(|| F::zero());
        let a4 = F::from(1.383577518672690e+02).unwrap_or_else(|| F::zero());
        let a5 = F::from(-3.066479806614716e+01).unwrap_or_else(|| F::zero());
        let a6 = F::from(2.506628277459239e+00).unwrap_or_else(|| F::zero());

        let b1 = F::from(-5.447609879822406e+01).unwrap_or_else(|| F::zero());
        let b2 = F::from(1.615858368580409e+02).unwrap_or_else(|| F::zero());
        let b3 = F::from(-1.556989798598866e+02).unwrap_or_else(|| F::zero());
        let b4 = F::from(6.680131188771972e+01).unwrap_or_else(|| F::zero());
        let b5 = F::from(-1.328068155288572e+01).unwrap_or_else(|| F::zero());

        let c1 = F::from(-7.784894002430293e-03).unwrap_or_else(|| F::zero());
        let c2 = F::from(-3.223964580411365e-01).unwrap_or_else(|| F::zero());
        let c3 = F::from(-2.400758277161838e+00).unwrap_or_else(|| F::zero());
        let c4 = F::from(-2.549732539343734e+00).unwrap_or_else(|| F::zero());
        let c5 = F::from(4.374664141464968e+00).unwrap_or_else(|| F::zero());
        let c6 = F::from(2.938163982698783e+00).unwrap_or_else(|| F::zero());

        let d1c = F::from(7.784695709041462e-03).unwrap_or_else(|| F::zero());
        let d2c = F::from(3.224671290700398e-01).unwrap_or_else(|| F::zero());
        let d3c = F::from(2.445134137142996e+00).unwrap_or_else(|| F::zero());
        let d4c = F::from(3.754408661907416e+00).unwrap_or_else(|| F::zero());

        let p_low = F::from(0.02425).unwrap_or_else(|| F::zero());
        let p_high = F::one() - p_low;

        let z = if p < p_low {
            // Lower tail: rational approximation
            let q = (-F::from(2.0).unwrap_or_else(|| F::zero()) * p.ln()).sqrt();
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1c * q + d2c) * q + d3c) * q + d4c) * q + F::one())
        } else if p <= p_high {
            // Central region: rational approximation
            let q = p - half;
            let r = q * q;
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + F::one())
        } else {
            // Upper tail: rational approximation (by symmetry)
            let q = (-F::from(2.0).unwrap_or_else(|| F::zero()) * (F::one() - p).ln()).sqrt();
            -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1c * q + d2c) * q + d3c) * q + d4c) * q + F::one())
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
    /// let norm = Normal::new(0.0f64, 1.0).expect("Operation failed");
    /// let samples = norm.rvs(1000).expect("Operation failed");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            let sample = self.rand_distr.sample(&mut rng);
            samples.push(F::from(sample).expect("Failed to convert to float"));
        }

        Ok(Array1::from(samples))
    }
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
        let half = F::from(0.5).expect("Failed to convert constant to float");
        let two = F::from(2.0).expect("Failed to convert constant to float");
        let pi = F::from(std::f64::consts::PI).expect("Failed to convert to float");
        let e = F::from(std::f64::consts::E).expect("Failed to convert to float");

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

/// Tail-accurate survival functions for the `ContinuousCDF` helpers
/// (`sf`, `isf`, `hazard`, `cumhazard`).
impl<F: Float + NumCast + std::fmt::Display> ContinuousCDF<F> for Normal<F> {
    /// Direct upper tail (see the inherent `sf`), not `1 - cdf`.
    fn sf(&self, x: F) -> F {
        Normal::sf(self, x)
    }

    /// Tail-accurate inverse survival function (see the inherent `isf`).
    fn isf(&self, q: F) -> StatsResult<F> {
        Normal::isf(self, q)
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
        let norm = Normal::new(0.0, 1.0).expect("Operation failed");
        assert_eq!(norm.loc, 0.0);
        assert_eq!(norm.scale, 1.0);

        // Custom normal
        let custom = Normal::new(5.0, 2.0).expect("Operation failed");
        assert_eq!(custom.loc, 5.0);
        assert_eq!(custom.scale, 2.0);

        // Error cases
        assert!(Normal::<f64>::new(0.0, 0.0).is_err());
        assert!(Normal::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_normal_pdf() {
        // Standard normal PDF values
        let norm = Normal::new(0.0, 1.0).expect("Operation failed");

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
        let custom = Normal::new(5.0, 2.0).expect("Operation failed");
        assert_relative_eq!(custom.pdf(5.0), 0.19947114, epsilon = 1e-7);
    }

    #[test]
    fn test_normal_cdf() {
        // Standard normal CDF values
        let norm = Normal::new(0.0, 1.0).expect("Operation failed");

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
        let norm = Normal::new(0.0, 1.0).expect("Operation failed");

        // Median (50th percentile)
        let median = norm.ppf(0.5).expect("Operation failed");
        assert_relative_eq!(median, 0.0, epsilon = 1e-5);

        // 97.5th percentile (often used for confidence intervals)
        let p975 = norm.ppf(0.975).expect("Operation failed");
        assert_relative_eq!(p975, 1.96, epsilon = 1e-2);

        // 2.5th percentile
        let p025 = norm.ppf(0.025).expect("Operation failed");
        assert_relative_eq!(p025, -1.96, epsilon = 1e-2);

        // Error cases
        assert!(norm.ppf(-0.1).is_err());
        assert!(norm.ppf(1.1).is_err());
    }

    #[test]
    fn test_normal_rvs() {
        let norm = Normal::new(0.0, 1.0).expect("Operation failed");

        // Generate samples
        let samples = norm.rvs(1000).expect("Operation failed");

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
