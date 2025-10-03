//! Laplace distribution functions
//!
//! This module provides functionality for the Laplace (double exponential) distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Distribution, Uniform as RandUniform};

/// Laplace distribution structure
///
/// The Laplace distribution, also known as the double exponential distribution,
/// is a continuous probability distribution that resembles a symmetric version of the
/// exponential distribution, placed back-to-back. It has heavier tails than the normal distribution.
pub struct Laplace<F: Float> {
    /// Location parameter (mean, median, and mode of the distribution)
    pub loc: F,
    /// Scale parameter (diversity) > 0
    pub scale: F,
    /// Random number generator for uniform distribution
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Laplace<F> {
    /// Create a new Laplace distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (mean, median, and mode of the distribution)
    /// * `scale` - Scale parameter (diversity) > 0
    ///
    /// # Returns
    ///
    /// * A new Laplace distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
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

        Ok(Laplace {
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let pdf_at_zero = laplace.pdf(0.0);
    /// assert!((pdf_at_zero - 0.5).abs() < 1e-7);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        let half = F::from(0.5).unwrap();
        let abs_value = if x >= self.loc {
            x - self.loc
        } else {
            self.loc - x
        };

        // PDF = (1/(2*scale)) * exp(-|x-loc|/scale)
        half / self.scale * (-abs_value / self.scale).exp()
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let cdf_at_zero = laplace.cdf(0.0);
    /// assert!((cdf_at_zero - 0.5).abs() < 1e-7);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        let half = F::from(0.5).unwrap();

        if x < self.loc {
            // CDF = (1/2) * exp((x-loc)/scale)
            half * ((x - self.loc) / self.scale).exp()
        } else {
            // CDF = 1 - (1/2) * exp(-(x-loc)/scale)
            F::one() - half * (-(x - self.loc) / self.scale).exp()
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let x = laplace.ppf(0.75).unwrap();
    /// assert!((x - 0.693147).abs() < 1e-6);
    /// ```
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        let half = F::from(0.5).unwrap();

        let quantile = if p < half {
            // Q(p) = loc + scale * ln(2p)
            self.loc + self.scale * (p + p).ln()
        } else {
            // Q(p) = loc - scale * ln(2(1-p))
            self.loc - self.scale * ((F::one() - p) + (F::one() - p)).ln()
        };

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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let samples = laplace.rvs_vec(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // Generate uniform random number in [0, 1)
            let u = self.rand_distr.sample(&mut rng);
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
            let u = self.rand_distr.sample(&mut rng);
            let u_f = F::from(u).unwrap();

            let sample = match self.ppf(u_f) {
                Ok(s) => s,
                Err(_) => continue,
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let samples = laplace.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let samples_vec = self.rvs_vec(size)?;
        Ok(Array1::from(samples_vec))
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(2.0f64, 1.0).unwrap();
    /// let mean = laplace.mean();
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let variance = laplace.var();
    /// assert!((variance - 2.0).abs() < 1e-7);
    /// ```
    pub fn var(&self) -> F {
        // Variance = 2 * scale^2
        let two = F::from(2.0).unwrap();
        two * self.scale * self.scale
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let std_dev = laplace.std();
    /// assert!((std_dev - 1.414213).abs() < 1e-6);
    /// ```
    pub fn std(&self) -> F {
        // Std = sqrt(var) = sqrt(2) * scale
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(2.0f64, 1.0).unwrap();
    /// let median = laplace.median();
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(2.0f64, 1.0).unwrap();
    /// let mode = laplace.mode();
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
    /// * The skewness (which is always 0 for the Laplace distribution)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let skewness = laplace.skewness();
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
    /// * The kurtosis of the distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let kurtosis = laplace.kurtosis();
    /// assert!((kurtosis - 3.0).abs() < 1e-7);
    /// ```
    pub fn kurtosis(&self) -> F {
        // Excess kurtosis = 3 (higher than normal distribution's 0)
        F::from(3.0).unwrap()
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
    /// use scirs2_stats::distributions::laplace::Laplace;
    ///
    /// let laplace = Laplace::new(0.0f64, 1.0).unwrap();
    /// let entropy = laplace.entropy();
    /// assert!((entropy - 1.693147).abs() < 1e-6);
    /// ```
    pub fn entropy(&self) -> F {
        // Entropy = 1 + ln(2*scale)
        let one = F::one();
        let two = F::from(2.0).unwrap();
        one + (two * self.scale).ln()
    }
}

/// Create a Laplace distribution with the given parameters.
///
/// This is a convenience function to create a Laplace distribution with
/// the given location and scale parameters.
///
/// # Arguments
///
/// * `loc` - Location parameter (mean, median, and mode of the distribution)
/// * `scale` - Scale parameter (diversity) > 0
///
/// # Returns
///
/// * A Laplace distribution object
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::laplace;
///
/// let l = laplace::laplace(0.0f64, 1.0).unwrap();
/// let pdf_at_zero = l.pdf(0.0);
/// assert!((pdf_at_zero - 0.5).abs() < 1e-7);
/// ```
#[allow(dead_code)]
pub fn laplace<F>(loc: F, scale: F) -> StatsResult<Laplace<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    Laplace::new(loc, scale)
}

/// Implementation of SampleableDistribution for Laplace
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Laplace<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs_vec(size)
    }
}

/// Implementation of Distribution trait for Laplace
impl<F: Float + NumCast + std::fmt::Display> ScirsDist<F> for Laplace<F> {
    fn mean(&self) -> F {
        self.mean()
    }

    fn var(&self) -> F {
        self.var()
    }

    fn std(&self) -> F {
        self.std()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        self.entropy()
    }
}

/// Implementation of ContinuousDistribution trait for Laplace
impl<F: Float + NumCast + std::fmt::Display> ContinuousDistribution<F> for Laplace<F> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_laplace_creation() {
        // Standard Laplace (loc=0, scale=1)
        let laplace = Laplace::new(0.0, 1.0).unwrap();
        assert_eq!(laplace.loc, 0.0);
        assert_eq!(laplace.scale, 1.0);

        // Custom Laplace
        let custom = Laplace::new(-2.0, 0.5).unwrap();
        assert_eq!(custom.loc, -2.0);
        assert_eq!(custom.scale, 0.5);

        // Error case: non-positive scale
        assert!(Laplace::<f64>::new(0.0, 0.0).is_err());
        assert!(Laplace::<f64>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_laplace_pdf() {
        // Standard Laplace (loc=0, scale=1)
        let laplace = Laplace::new(0.0, 1.0).unwrap();

        // PDF at x = 0 should be 1/(2*1) = 0.5
        let pdf_at_zero = laplace.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 0.5, epsilon = 1e-7);

        // PDF at x = 1 should be 0.5 * exp(-1/1) = 0.5 * exp(-1) = 0.5 * 0.36787944 ≈ 0.1839397
        let pdf_at_one = laplace.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.1839397, epsilon = 1e-7);

        // PDF at x = -1 should be same as x = 1 due to symmetry
        let pdf_at_neg_one = laplace.pdf(-1.0);
        assert_relative_eq!(pdf_at_neg_one, pdf_at_one, epsilon = 1e-10);

        // Custom Laplace with loc=-2, scale=0.5
        let custom = Laplace::new(-2.0, 0.5).unwrap();

        // PDF at x = -2 should be 1/(2*0.5) = 1
        let pdf_at_loc = custom.pdf(-2.0);
        assert_relative_eq!(pdf_at_loc, 1.0, epsilon = 1e-7);

        // PDF at x = -1.5 should be 0.5/0.5 * exp(-|-1.5-(-2)|/0.5) = exp(-0.5/0.5) = exp(-1) ≈ 0.36787944
        let pdf_at_custom = custom.pdf(-1.5);
        assert_relative_eq!(pdf_at_custom, 0.36787944, epsilon = 1e-7);
    }

    #[test]
    fn test_laplace_cdf() {
        // Standard Laplace (loc=0, scale=1)
        let laplace = Laplace::new(0.0, 1.0).unwrap();

        // CDF at x = 0 should be 0.5
        let cdf_at_zero = laplace.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.5, epsilon = 1e-10);

        // CDF at x = 1 should be 1 - 0.5*exp(-1/1) = 1 - 0.5*exp(-1) ≈ 1 - 0.5*0.36787944 ≈ 0.8160603
        let cdf_at_one = laplace.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.8160603, epsilon = 1e-7);

        // CDF at x = -1 should be 0.5*exp((-1-0)/1) = 0.5*exp(-1) ≈ 0.5*0.36787944 ≈ 0.1839397
        let cdf_at_neg_one = laplace.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 0.1839397, epsilon = 1e-7);

        // Custom Laplace with loc=-2, scale=0.5
        let custom = Laplace::new(-2.0, 0.5).unwrap();

        // CDF at x = -2 should be 0.5
        let cdf_at_loc = custom.cdf(-2.0);
        assert_relative_eq!(cdf_at_loc, 0.5, epsilon = 1e-10);

        // CDF at x = -1.5 should be 1 - 0.5*exp(-(-1.5-(-2))/0.5) = 1 - 0.5*exp(-0.5/0.5) = 1 - 0.5*exp(-1) ≈ 0.8160603
        let cdf_at_custom = custom.cdf(-1.5);
        assert_relative_eq!(cdf_at_custom, 0.8160603, epsilon = 1e-7);
    }

    #[test]
    fn test_laplace_ppf() {
        // Standard Laplace (loc=0, scale=1)
        let laplace = Laplace::new(0.0, 1.0).unwrap();

        // PPF at p = 0.5 should be 0
        let ppf_at_half = laplace.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half, 0.0, epsilon = 1e-10);

        // PPF at p = 0.75 should be log(2*0.75) ≈ log(1.5) ≈ 0.693147
        let ppf_at_75 = laplace.ppf(0.75).unwrap();
        assert_relative_eq!(ppf_at_75, 0.693147, epsilon = 1e-6);

        // PPF at p = 0.25 should be -log(2*0.75) ≈ -log(1.5) ≈ -0.693147
        let ppf_at_25 = laplace.ppf(0.25).unwrap();
        assert_relative_eq!(ppf_at_25, -0.693147, epsilon = 1e-6);

        // Custom Laplace with loc=-2, scale=0.5
        let custom = Laplace::new(-2.0, 0.5).unwrap();

        // PPF at p = 0.5 should be -2.0
        let ppf_at_half_custom = custom.ppf(0.5).unwrap();
        assert_relative_eq!(ppf_at_half_custom, -2.0, epsilon = 1e-10);

        // PPF at p = 0.75 should be -2.0 + 0.5*log(1.5) ≈ -2.0 + 0.5*0.693147 ≈ -1.653426
        let ppf_at_75_custom = custom.ppf(0.75).unwrap();
        assert_relative_eq!(ppf_at_75_custom, -1.653426, epsilon = 1e-6);

        // Error cases
        assert!(laplace.ppf(-0.1).is_err());
        assert!(laplace.ppf(1.1).is_err());
    }

    #[test]
    fn test_laplace_properties() {
        // Standard Laplace (loc=0, scale=1)
        let laplace = Laplace::new(0.0, 1.0).unwrap();

        // Mean = loc = 0
        let mean = laplace.mean();
        assert_eq!(mean, 0.0);

        // Variance = 2 * scale^2 = 2 * 1^2 = 2
        let variance = laplace.var();
        assert_eq!(variance, 2.0);

        // Std = sqrt(variance) = sqrt(2) ≈ 1.414213
        let std_dev = laplace.std();
        assert_relative_eq!(std_dev, 1.414213, epsilon = 1e-6);

        // Median = loc = 0
        let median = laplace.median();
        assert_eq!(median, 0.0);

        // Mode = loc = 0
        let mode = laplace.mode();
        assert_eq!(mode, 0.0);

        // Skewness = 0 (symmetric)
        let skewness = laplace.skewness();
        assert_eq!(skewness, 0.0);

        // Kurtosis = 3
        let kurtosis = laplace.kurtosis();
        assert_eq!(kurtosis, 3.0);

        // Entropy = 1 + ln(2*scale) = 1 + ln(2) ≈ 1.693147
        let entropy = laplace.entropy();
        assert_relative_eq!(entropy, 1.693147, epsilon = 1e-6);

        // Custom Laplace with loc=-2, scale=0.5
        let custom = Laplace::new(-2.0, 0.5).unwrap();

        // Mean = loc = -2
        let mean_custom = custom.mean();
        assert_eq!(mean_custom, -2.0);

        // Variance = 2 * scale^2 = 2 * 0.5^2 = 0.5
        let variance_custom = custom.var();
        assert_eq!(variance_custom, 0.5);

        // Std = sqrt(variance) = sqrt(0.5) ≈ 0.707107
        let std_dev_custom = custom.std();
        assert_relative_eq!(std_dev_custom, 0.707107, epsilon = 1e-6);

        // Entropy = 1 + ln(2*scale) = 1 + ln(2*0.5) = 1 + ln(1) = 1
        let entropy_custom = custom.entropy();
        assert_relative_eq!(entropy_custom, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_rvs() {
        let laplace = Laplace::new(0.0, 1.0).unwrap();

        // Generate samples
        let samples = laplace.rvs(100).unwrap();

        // Check the number of samples
        assert_eq!(samples.len(), 100);

        // Calculate sample mean and check it's reasonably close to loc = 0
        // (with large enough samples)
        let sum: f64 = samples.iter().sum();
        let mean = sum / samples.len() as f64;

        // The mean could be off due to randomness, but should be within a reasonable range
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_laplace_inverse_cdf() {
        // Test that cdf(ppf(p)) == p and ppf(cdf(x)) == x
        let laplace = Laplace::new(0.0, 1.0).unwrap();

        // Test various probability values
        let probabilities = [0.1, 0.25, 0.5, 0.75, 0.9];
        for &p in &probabilities {
            let x = laplace.ppf(p).unwrap();
            let p_back = laplace.cdf(x);
            assert_relative_eq!(p_back, p, epsilon = 1e-7);
        }

        // Test various x values
        let x_values = [-3.0, -1.0, 0.0, 1.0, 3.0];
        for &x in &x_values {
            let p = laplace.cdf(x);
            let x_back = laplace.ppf(p).unwrap();
            assert_relative_eq!(x_back, x, epsilon = 1e-7);
        }
    }
}
