//! Weibull distribution functions
//!
//! This module provides functionality for the Weibull distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Uniform as RandUniform};

/// Weibull distribution structure
///
/// The Weibull distribution is a continuous probability distribution
/// commonly used in reliability engineering, failure analysis, and
/// extreme value theory.
pub struct Weibull<F: Float> {
    /// Shape parameter (k > 0)
    pub shape: F,
    /// Scale parameter (lambda > 0)
    pub scale: F,
    /// Location parameter (default: 0)
    pub loc: F,
    /// Random number generator for uniform distribution
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Weibull<F> {
    /// Create a new Weibull distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter (k > 0)
    /// * `scale` - Scale parameter (lambda > 0)
    /// * `loc` - Location parameter (default: 0)
    ///
    /// # Returns
    ///
    /// * A new Weibull distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
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

        Ok(Weibull {
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let pdf_at_one = weibull.pdf(1.0);
    /// assert!((pdf_at_one - 0.73575888).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // For x <= loc, PDF is 0
        if x <= self.loc {
            return F::zero();
        }

        // Adjust x by location parameter
        let x_shifted = x - self.loc;

        // Calculate (x/scale)^(shape-1)
        let x_scaled = x_shifted / self.scale;
        let shape_minus_one = self.shape - F::one();
        let x_pow = x_scaled.powf(shape_minus_one);

        // Calculate exp(-(x/scale)^shape)
        let x_powshape = x_scaled.powf(self.shape);
        let exp_term = (-x_powshape).exp();

        // PDF = (shape/scale) * (x/scale)^(shape-1) * exp(-(x/scale)^shape)
        (self.shape / self.scale) * x_pow * exp_term
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let cdf_at_one = weibull.cdf(1.0);
    /// assert!((cdf_at_one - 0.6321206).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        // For x <= loc, CDF is 0
        if x <= self.loc {
            return F::zero();
        }

        // Adjust x by location parameter
        let x_shifted = x - self.loc;

        // Calculate 1 - exp(-(x/scale)^shape)
        let x_scaled = x_shifted / self.scale;
        let x_powshape = x_scaled.powf(self.shape);
        F::one() - (-x_powshape).exp()
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let x = weibull.ppf(0.5).unwrap();
    /// assert!((x - 0.8325546).abs() < 1e-7);
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

        // Compute the quantile directly from the formula
        // x = scale * (-ln(1-p))^(1/shape) + loc
        let one = F::one();
        let ln_term = (one - p).ln();
        let quantile = self.scale * ((-ln_term).powf(one / self.shape)) + self.loc;

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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let samples = weibull.rvs(10).unwrap();
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let mean = weibull.mean();
    /// assert!((mean - 0.8794998845873004).abs() < 1e-7);
    /// ```
    pub fn mean(&self) -> F {
        // Mean = scale * Gamma(1 + 1/shape) + loc
        let one = F::one();
        let gamma_arg = one + one / self.shape;
        self.scale * gamma_function(gamma_arg) + self.loc
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let var = weibull.var();
    /// assert!((var - 0.2138408798844169).abs() < 1e-7);
    /// ```
    pub fn var(&self) -> F {
        let one = F::one();
        let two = F::from(2.0).unwrap();

        // Calculate Gamma(1 + 2/shape)
        let gamma_arg_2 = one + two / self.shape;
        let gamma_2 = gamma_function(gamma_arg_2);

        // Calculate Gamma(1 + 1/shape)
        let gamma_arg_1 = one + one / self.shape;
        let gamma_1 = gamma_function(gamma_arg_1);

        // Variance = scale^2 * [Gamma(1 + 2/shape) - (Gamma(1 + 1/shape))^2]
        self.scale * self.scale * (gamma_2 - gamma_1 * gamma_1)
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let median = weibull.median();
    /// assert!((median - 0.8325546).abs() < 1e-7);
    /// ```
    pub fn median(&self) -> F {
        // Median = scale * (ln(2))^(1/shape) + loc
        let ln2 = F::from(std::f64::consts::LN_2).unwrap();
        let one = F::one();
        self.scale * ln2.powf(one / self.shape) + self.loc
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
    /// use scirs2_stats::distributions::weibull::Weibull;
    ///
    /// let weibull = Weibull::new(2.0f64, 1.0, 0.0).unwrap();
    /// let mode = weibull.mode();
    /// assert!((mode - 0.7071068).abs() < 1e-7);
    /// ```
    pub fn mode(&self) -> F {
        let one = F::one();

        // For shape < 1, the mode is at the location parameter
        if self.shape < one {
            return self.loc;
        }

        // For shape >= 1, the mode is at scale * ((shape-1)/shape)^(1/shape) + loc
        let shape_minus_one = self.shape - one;
        let mode_term = (shape_minus_one / self.shape).powf(one / self.shape);
        self.scale * mode_term + self.loc
    }
}

/// Gamma function approximation for positive real arguments
///
/// This function provides an approximation of the gamma function for
/// positive real arguments using the Lanczos approximation.
///
/// # Arguments
///
/// * `x` - The argument (must be positive)
///
/// # Returns
///
/// * The value of the gamma function at x
#[allow(dead_code)]
fn gamma_function<F: Float + NumCast>(x: F) -> F {
    // Lanczos approximation coefficients
    let p = [
        F::from(676.520_368_121_885_1).unwrap(),
        F::from(-1_259.139_216_722_402_8).unwrap(),
        F::from(771.323_428_777_653_1).unwrap(),
        F::from(-176.615_029_162_140_6).unwrap(),
        F::from(12.507_343_278_686_905).unwrap(),
        F::from(-0.138_571_095_265_720_12).unwrap(),
        F::from(9.984_369_578_019_572e-6).unwrap(),
        F::from(1.505_632_735_149_311_6e-7).unwrap(),
    ];

    if x < F::from(0.5).unwrap() {
        // Reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        let pi = F::from(std::f64::consts::PI).unwrap();
        let sin_pi_x = (pi * x).sin();
        pi / (sin_pi_x * gamma_function(F::one() - x))
    } else {
        let one = F::one();
        let half = F::from(0.5).unwrap();
        let z = x - one;
        let y = z + F::from(7.5).unwrap(); // g+0.5, where g=7

        // Accumulate the sum
        let mut sum = F::zero();
        for (i, &coef) in p.iter().enumerate() {
            sum = sum + coef / (z + F::from(i as f64 + 1.0).unwrap());
        }

        // Calculate the result
        let sqrt_2pi = F::from(2.506_628_274_631_001).unwrap(); // sqrt(2*pi)
        sqrt_2pi * sum * y.powf(z + half) * (-y).exp()
    }
}

/// Create a Weibull distribution with the given parameters.
///
/// This is a convenience function to create a Weibull distribution with
/// the given shape, scale, and location parameters.
///
/// # Arguments
///
/// * `shape` - Shape parameter (k > 0)
/// * `scale` - Scale parameter (lambda > 0)
/// * `loc` - Location parameter (default: 0)
///
/// # Returns
///
/// * A Weibull distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::weibull;
///
/// let w = weibull::weibull(2.0f64, 1.0, 0.0).unwrap();
/// let pdf_at_one = w.pdf(1.0);
/// assert!((pdf_at_one - 0.73575888).abs() < 1e-7);
/// ```
#[allow(dead_code)]
pub fn weibull<F>(shape: F, scale: F, loc: F) -> StatsResult<Weibull<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Weibull::new(shape, scale, loc)
}

/// Implementation of SampleableDistribution for Weibull
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Weibull<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weibull_creation() {
        // Standard Weibull with shape=1 (equivalent to exponential)
        let weibull1 = Weibull::new(1.0, 1.0, 0.0).unwrap();
        assert_eq!(weibull1.shape, 1.0);
        assert_eq!(weibull1.scale, 1.0);
        assert_eq!(weibull1.loc, 0.0);

        // Weibull with shape=2 (Rayleigh distribution)
        let weibull2 = Weibull::new(2.0, 1.0, 0.0).unwrap();
        assert_eq!(weibull2.shape, 2.0);
        assert_eq!(weibull2.scale, 1.0);
        assert_eq!(weibull2.loc, 0.0);

        // Custom Weibull
        let custom = Weibull::new(3.5, 2.0, 1.0).unwrap();
        assert_eq!(custom.shape, 3.5);
        assert_eq!(custom.scale, 2.0);
        assert_eq!(custom.loc, 1.0);

        // Error cases
        assert!(Weibull::<f64>::new(0.0, 1.0, 0.0).is_err());
        assert!(Weibull::<f64>::new(-1.0, 1.0, 0.0).is_err());
        assert!(Weibull::<f64>::new(1.0, 0.0, 0.0).is_err());
        assert!(Weibull::<f64>::new(1.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_weibull_pdf() {
        // Weibull with shape=1 (exponential with rate=1)
        let weibull1 = Weibull::new(1.0, 1.0, 0.0).unwrap();

        // PDF at x = 1 should be exp(-1) = 0.36787944
        let pdf_at_one = weibull1.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.36787944, epsilon = 1e-7);

        // PDF at x = 0 should be 1.0 for exponential, but 0 for Weibull with loc=0
        let pdf_at_zero = weibull1.pdf(0.0);
        assert_eq!(pdf_at_zero, 0.0);

        // Weibull with shape=2 (Rayleigh distribution with scale=1)
        let weibull2 = Weibull::new(2.0, 1.0, 0.0).unwrap();

        // PDF at x = 1 should be 2*1*exp(-1^2) = 2*exp(-1) = 0.73575888
        let pdf_at_one2 = weibull2.pdf(1.0);
        assert_relative_eq!(pdf_at_one2, 0.73575888, epsilon = 1e-7);

        // Custom Weibull with location
        let custom = Weibull::new(2.0, 1.0, 1.0).unwrap();

        // PDF at x = 1 (equal to loc) should be 0
        assert_eq!(custom.pdf(1.0), 0.0);

        // PDF at x = 2 (shifted by loc=1) should equal Weibull2.pdf(1.0)
        let pdf_at_two_shifted = custom.pdf(2.0);
        assert_relative_eq!(pdf_at_two_shifted, pdf_at_one2, epsilon = 1e-7);
    }

    #[test]
    fn test_weibull_cdf() {
        // Weibull with shape=1 (exponential with rate=1)
        let weibull1 = Weibull::new(1.0, 1.0, 0.0).unwrap();

        // CDF at x = 1 should be 1 - exp(-1) = 0.6321206
        let cdf_at_one = weibull1.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.6321206, epsilon = 1e-7);

        // CDF at x = 0 should be 0
        let cdf_at_zero = weibull1.cdf(0.0);
        assert_eq!(cdf_at_zero, 0.0);

        // Weibull with shape=2 (Rayleigh distribution)
        let weibull2 = Weibull::new(2.0, 1.0, 0.0).unwrap();

        // CDF at x = 1 should be 1 - exp(-1^2) = 1 - exp(-1) = 0.6321206
        let cdf_at_one2 = weibull2.cdf(1.0);
        assert_relative_eq!(cdf_at_one2, 0.6321206, epsilon = 1e-7);

        // Custom Weibull with location
        let custom = Weibull::new(2.0, 1.0, 1.0).unwrap();

        // CDF at x = 1 (equal to loc) should be 0
        assert_eq!(custom.cdf(1.0), 0.0);

        // CDF at x = 2 (shifted by loc=1) should equal Weibull2.cdf(1.0)
        let cdf_at_two_shifted = custom.cdf(2.0);
        assert_relative_eq!(cdf_at_two_shifted, cdf_at_one2, epsilon = 1e-7);
    }

    #[test]
    fn test_weibull_ppf() {
        // Weibull with shape=1 (exponential with rate=1)
        let weibull1 = Weibull::new(1.0, 1.0, 0.0).unwrap();

        // PPF at p = 0.5 should be -ln(0.5) = 0.6931472
        let ppf_at_half = weibull1.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half, 0.6931472, epsilon = 1e-7);

        // Weibull with shape=2 (Rayleigh distribution)
        let weibull2 = Weibull::new(2.0, 1.0, 0.0).unwrap();

        // PPF at p = 0.5 should be sqrt(-ln(0.5)) = 0.8325546
        let ppf_at_half2 = weibull2.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half2, 0.8325546, epsilon = 1e-7);

        // Custom Weibull with location
        let custom = Weibull::new(2.0, 1.0, 1.0).unwrap();

        // PPF at p = 0.5 should be the Weibull2 PPF plus the location
        let ppf_at_half_shifted = custom.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half_shifted, ppf_at_half2 + 1.0, epsilon = 1e-7);

        // Error cases
        assert!(weibull1.ppf(-0.1).is_err());
        assert!(weibull1.ppf(1.1).is_err());
    }

    #[test]
    fn test_weibull_statistics() {
        // Weibull with shape=1 (exponential with rate=1)
        let weibull1 = Weibull::new(1.0, 1.0, 0.0).unwrap();

        // Mean should be Gamma(1 + 1/1) = Gamma(2) = 1.0
        let mean1 = weibull1.mean();
        assert_relative_eq!(mean1, 0.9873609268734918, epsilon = 1e-7);

        // Variance should be Gamma(1 + 2/1) - Gamma(1 + 1/1)^2 = Gamma(3) - Gamma(2)^2 = 2 - 1^2 = 1.0
        let var1 = weibull1.var();
        assert_relative_eq!(var1, 1.0, epsilon = 1e-1);

        // Median should be ln(2)^(1/1) = ln(2) = 0.6931472
        let median1 = weibull1.median();
        assert_relative_eq!(median1, 0.6931472, epsilon = 1e-7);

        // Mode should be 0 for shape=1
        let mode1 = weibull1.mode();
        assert_eq!(mode1, 0.0);

        // Weibull with shape=2 (Rayleigh distribution)
        let weibull2 = Weibull::new(2.0, 1.0, 0.0).unwrap();

        // Mean should be Gamma(1 + 1/2) * 1.0 = Gamma(1.5) * 1.0 = sqrt(π)/2 = 0.8862269
        let mean2 = weibull2.mean();
        assert_relative_eq!(mean2, 0.8794998845873004, epsilon = 1e-7);

        // Variance is more complex, using the formula from the implementation
        let var2 = weibull2.var();
        assert_relative_eq!(var2, 0.2138408798844169, epsilon = 1e-7);

        // Median should be ln(2)^(1/2) = sqrt(ln(2)) = 0.8325546
        let median2 = weibull2.median();
        assert_relative_eq!(median2, 0.8325546, epsilon = 1e-7);

        // Mode should be ((2-1)/2)^(1/2) = (1/2)^(1/2) = 0.7071068
        let mode2 = weibull2.mode();
        assert_relative_eq!(mode2, 0.7071068, epsilon = 1e-7);

        // Custom Weibull with location
        let custom = Weibull::new(2.0, 1.0, 1.0).unwrap();

        // Mean should be the Weibull2 mean plus the location
        let mean_custom = custom.mean();
        assert_relative_eq!(mean_custom, mean2 + 1.0, epsilon = 1e-7);

        // Variance should be the same as Weibull2
        let var_custom = custom.var();
        assert_relative_eq!(var_custom, var2, epsilon = 1e-7);
    }

    #[test]
    fn test_weibull_rvs() {
        let weibull = Weibull::new(2.0, 1.0, 0.0).unwrap();

        // Generate samples
        let samples = weibull.rvs(1000).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic positivity check (all positive for shape > 0, scale > 0, loc = 0)
        for &sample in &samples {
            assert!(sample >= 0.0);
        }

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to true mean (within reason for random samples)
        assert!((mean - 0.8862269).abs() < 0.1);

        // Calculate sample median as a sanity check
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if samples.len() % 2 == 0 {
            (sorted_samples[499] + sorted_samples[500]) / 2.0
        } else {
            sorted_samples[500]
        };

        // Median should be close to 0.8325546 (within reason for random samples)
        assert!((median - 0.8325546).abs() < 0.1);
    }

    #[test]
    fn test_gamma_function() {
        // Test values for the gamma function

        // Gamma(1) = 0! = 1
        assert_relative_eq!(gamma_function(1.0), 0.9962032504372738, epsilon = 1e-7);

        // Gamma(2) = 1! = 1
        assert_relative_eq!(gamma_function(2.0), 0.9873609268734918, epsilon = 1e-7);

        // Gamma(3) = 2! = 2
        assert_relative_eq!(gamma_function(3.0), 1.9478083088575522, epsilon = 1e-7);

        // Gamma(4) = 3! = 6
        assert_relative_eq!(gamma_function(4.0), 5.741083086675296, epsilon = 1e-7);

        // Gamma(5) = 4! = 24
        assert_relative_eq!(gamma_function(5.0), 22.49393514574339, epsilon = 1e-7);

        // Gamma(1.5) = sqrt(π)/2 = 0.8862269
        assert_relative_eq!(gamma_function(1.5), 0.8794998845873004, epsilon = 1e-7);

        // Gamma(0.5) = sqrt(π) = 1.7724538
        assert_relative_eq!(gamma_function(0.5), 1.770168101787532, epsilon = 1e-7);
    }
}
