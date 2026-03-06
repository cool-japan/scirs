//! Student's t distribution functions
//!
//! This module provides functionality for the Student's t distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, StudentT as RandStudentT};
use statrs::function::beta::{beta_reg, inv_beta_reg};
use std::f64::consts::PI;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + NumCast>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}

/// Student's t distribution structure
pub struct StudentT<F: Float + Send + Sync> {
    /// Degrees of freedom
    pub df: F,
    /// Location parameter
    pub loc: F,
    /// Scale parameter
    pub scale: F,
    /// Random number generator for this distribution
    rand_distr: RandStudentT<f64>,
}

impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> StudentT<F> {
    /// Create a new Student's t distribution with given degrees of freedom, location, and scale
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom (> 0)
    /// * `loc` - Location parameter (default: 0)
    /// * `scale` - Scale parameter (default: 1, must be > 0)
    ///
    /// # Returns
    ///
    /// * A new StudentT distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::student_t::StudentT;
    ///
    /// // Standard t-distribution with 5 degrees of freedom
    /// let t = StudentT::new(5.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// ```
    pub fn new(df: F, loc: F, scale: F) -> StatsResult<Self> {
        if df <= F::zero() {
            return Err(StatsError::DomainError(
                "Degrees of freedom must be positive".to_string(),
            ));
        }

        if scale <= F::zero() {
            return Err(StatsError::DomainError(
                "Scale parameter must be positive".to_string(),
            ));
        }

        // Convert to f64 for rand_distr
        let df_f64 = NumCast::from(df).expect("Failed to convert to f64");

        match RandStudentT::new(df_f64) {
            Ok(rand_distr) => Ok(StudentT {
                df,
                loc,
                scale,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create Student's t distribution".to_string(),
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
    /// use scirs2_stats::distributions::student_t::StudentT;
    ///
    /// let t = StudentT::new(5.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let pdf_at_zero = t.pdf(0.0);
    /// assert!((pdf_at_zero - 0.3796).abs() < 1e-4);
    /// ```
    #[inline]
    pub fn pdf(&self, x: F) -> F {
        // Standardize the variable
        let x_std = (x - self.loc) / self.scale;

        // Calculate gamma values for the PDF formula
        let df_half = self.df / const_f64::<F>(2.0);
        let df_plus_one_half = (self.df + F::one()) / const_f64::<F>(2.0);

        // Use the formula for the PDF
        let one = F::one();
        let pi = const_f64::<F>(PI);

        // Calculate the PDF value
        let numerator = gamma_function(df_plus_one_half);
        let denominator = gamma_function(df_half) * (self.df * pi).sqrt();

        let factor = numerator / denominator / self.scale;
        let exponent = -(df_plus_one_half) * (one + x_std * x_std / self.df).ln();

        factor * exponent.exp()
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
    /// use scirs2_stats::distributions::student_t::StudentT;
    ///
    /// let t = StudentT::new(5.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let cdf_at_zero = t.cdf(0.0);
    /// assert!((cdf_at_zero - 0.5).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn cdf(&self, x: F) -> F {
        // Standardize the variable
        let x_std = (x - self.loc) / self.scale;

        // Handle special values
        if x_std.is_nan() {
            return F::nan();
        }
        if x_std == F::infinity() {
            return F::one();
        }
        if x_std == F::neg_infinity() {
            return F::zero();
        }

        // For t-distribution, CDF at 0 is exactly 0.5 by symmetry
        if x_std == F::zero() {
            return const_f64::<F>(0.5);
        }

        // Use the regularized incomplete beta function:
        // CDF(t) = I_{v/(v+t²)}(v/2, 1/2) / 2        for t <= 0
        //        = 1 - I_{v/(v+t²)}(v/2, 1/2) / 2    for t > 0
        // where v = degrees of freedom
        let df_f64: f64 = NumCast::from(self.df).expect("Failed to convert df to f64");
        let x_std_f64: f64 = NumCast::from(x_std).expect("Failed to convert x_std to f64");

        let h = df_f64 / (df_f64 + x_std_f64 * x_std_f64);
        let ib = 0.5 * beta_reg(df_f64 / 2.0, 0.5, h);

        let result = if x_std_f64 <= 0.0 { ib } else { 1.0 - ib };
        const_f64::<F>(result)
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
    /// use scirs2_stats::distributions::student_t::StudentT;
    ///
    /// let t = StudentT::new(5.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let samples = t.rvs(1000).expect("test/example should not fail");
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
    /// use scirs2_stats::distributions::student_t::StudentT;
    ///
    /// let t = StudentT::new(5.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let samples = t.rvs_vec(1000).expect("test/example should not fail");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    #[inline]
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        // For small sample sizes, use the serial implementation
        if size < 1000 {
            let mut rng = thread_rng();
            let mut samples = Vec::with_capacity(size);

            for _ in 0..size {
                // Generate a standard Student's t random variable
                let std_sample = self.rand_distr.sample(&mut rng);

                // Scale and shift according to loc and scale parameters
                let sample = const_f64::<F>(std_sample) * self.scale + self.loc;
                samples.push(sample);
            }

            return Ok(samples);
        }

        // For larger sample sizes, use parallel implementation with scirs2-core's parallel module
        use scirs2_core::parallel_ops::parallel_map;

        // Clone distribution parameters for thread safety
        let df_f64 = NumCast::from(self.df).expect("Failed to convert to f64");
        let loc = self.loc;
        let scale = self.scale;

        // Create indices for parallelization
        let indices: Vec<usize> = (0..size).collect();

        // Generate samples in parallel
        let samples = parallel_map(&indices, move |_| {
            let mut rng = thread_rng();
            let rand_distr = RandStudentT::new(df_f64).expect("test/example should not fail");
            let sample = rand_distr.sample(&mut rng);
            const_f64::<F>(sample) * scale + loc
        });

        Ok(samples)
    }
}

/// Approximation of the gamma function for floating point types
#[inline]
#[allow(dead_code)]
fn gamma_function<F: Float>(x: F) -> F {
    if x == F::one() {
        return F::one();
    }

    if x == const_f64::<F>(0.5) {
        return const_f64::<F>(PI).sqrt();
    }

    // For integers and half-integers, use recurrence relation
    if x > F::one() {
        return (x - F::one()) * gamma_function(x - F::one());
    }

    // Use Lanczos approximation for other values
    let p = [
        const_f64::<F>(676.5203681218851),
        const_f64::<F>(-1259.1392167224028),
        const_f64::<F>(771.323_428_777_653_1),
        const_f64::<F>(-176.615_029_162_140_6),
        const_f64::<F>(12.507343278686905),
        const_f64::<F>(-0.13857109526572012),
        const_f64::<F>(9.984_369_578_019_572e-6),
        const_f64::<F>(1.5056327351493116e-7),
    ];

    let x_adj = x - F::one();
    let t = x_adj + const_f64::<F>(7.5);

    let mut sum = F::zero();
    for (i, &coef) in p.iter().enumerate() {
        sum = sum + coef / (x_adj + const_f64::<F>((i + 1) as f64));
    }

    let pi = const_f64::<F>(PI);
    let sqrt_2pi = (const_f64::<F>(2.0) * pi).sqrt();

    sqrt_2pi * sum * t.powf(x_adj + const_f64::<F>(0.5)) * (-t).exp()
}


/// Implementation of Distribution trait for StudentT
impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ScirsDist<F> for StudentT<F> {
    fn mean(&self) -> F {
        // Mean is 0 for df > 1, undefined for df <= 1
        if self.df <= F::one() {
            F::nan()
        } else {
            self.loc
        }
    }

    fn var(&self) -> F {
        // Variance is df/(df-2) * scale^2 for df > 2
        // Undefined for df <= 2
        if self.df <= const_f64::<F>(2.0) {
            F::nan()
        } else {
            self.df / (self.df - const_f64::<F>(2.0)) * self.scale * self.scale
        }
    }

    fn std(&self) -> F {
        // Standard deviation is sqrt(var)
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        // Entropy of the t-distribution is complex
        // For large df, it approaches the entropy of a normal distribution
        let df = self.df;
        let half = const_f64::<F>(0.5);
        let one = F::one();

        if df <= F::zero() {
            return F::nan();
        }

        // For very large df, use normal approximation
        if df > const_f64::<F>(1000.0) {
            let e = const_f64::<F>(std::f64::consts::E);
            return half * (const_f64::<F>(2.0) * const_f64::<F>(std::f64::consts::PI) * e).ln()
                + self.scale.ln();
        }

        // For small df, use the full formula
        let half_df_plus_half = (df + one) * half;
        let half_df = df * half;

        let term1 = half_df_plus_half * (gamma_function(half) / gamma_function(half_df)).ln();
        let term2 = half_df_plus_half;
        let term3 = half * (df * const_f64::<F>(std::f64::consts::PI)).ln();

        term1 + term2 + term3
    }
}

/// Implementation of ContinuousDistribution trait for StudentT
impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ContinuousDistribution<F>
    for StudentT<F>
{
    fn pdf(&self, x: F) -> F {
        // Call the implementation from the struct
        StudentT::pdf(self, x)
    }

    fn cdf(&self, x: F) -> F {
        // Call the implementation from the struct
        StudentT::cdf(self, x)
    }

    fn ppf(&self, p: F) -> StatsResult<F> {
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
        if p == const_f64::<F>(0.5) {
            return Ok(self.loc); // t-distribution is symmetric around loc
        }

        // Use the inverse of the regularized incomplete beta function
        // For t-distribution: PPF(p) = sign * sqrt(v * (1/y - 1))
        // where y = inv_beta_reg(v/2, 1/2, 2 * min(p, 1-p))
        let df_f64: f64 = NumCast::from(self.df).expect("Failed to convert df to f64");
        let p_f64: f64 = NumCast::from(p).expect("Failed to convert p to f64");

        let p1 = if p_f64 >= 0.5 { 1.0 - p_f64 } else { p_f64 };
        let a = df_f64 / 2.0;
        let b = 0.5;
        let y = inv_beta_reg(a, b, 2.0 * p1);
        // When y == 0 (extreme probabilities), the result is infinite
        let t = if y == 0.0 {
            f64::INFINITY
        } else {
            (df_f64 * (1.0 - y) / y).sqrt()
        };
        let t = if p_f64 >= 0.5 { t } else { -t };
        Ok(self.loc + const_f64::<F>(t) * self.scale)
    }
}

impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ContinuousCDF<F>
    for StudentT<F>
{
    // Default implementations from trait are sufficient
}

/// Implementation of SampleableDistribution for StudentT
impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> SampleableDistribution<F>
    for StudentT<F>
{
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
    fn test_student_t_creation() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");
        assert_eq!(t5.df, 5.0);
        assert_eq!(t5.loc, 0.0);
        assert_eq!(t5.scale, 1.0);

        // Custom t distribution
        let custom = StudentT::new(10.0, 1.0, 2.0).expect("test/example should not fail");
        assert_eq!(custom.df, 10.0);
        assert_eq!(custom.loc, 1.0);
        assert_eq!(custom.scale, 2.0);

        // Error cases
        assert!(StudentT::<f64>::new(0.0, 0.0, 1.0).is_err());
        assert!(StudentT::<f64>::new(-1.0, 0.0, 1.0).is_err());
        assert!(StudentT::<f64>::new(5.0, 0.0, 0.0).is_err());
        assert!(StudentT::<f64>::new(5.0, 0.0, -1.0).is_err());
    }

    #[test]
    fn test_student_t_pdf() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // PDF at x = 0
        let pdf_at_zero = t5.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 0.3796, epsilon = 1e-4);

        // PDF at x = 1
        let pdf_at_one = t5.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.220, epsilon = 1e-3);

        // PDF at x = -1 (symmetric)
        let pdf_at_neg_one = t5.pdf(-1.0);
        assert_relative_eq!(pdf_at_neg_one, 0.220, epsilon = 1e-3);
    }

    #[test]
    fn test_student_t_cdf() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // CDF at x = 0
        let cdf_at_zero = t5.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.5, epsilon = 1e-10);

        // CDF at x = 1 (t(5) distribution: scipy.stats.t.cdf(1, 5) ≈ 0.81839)
        let cdf_at_one = t5.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.8183912661754386, epsilon = 1e-6);

        // CDF at x = -1 (by symmetry)
        let cdf_at_neg_one = t5.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 0.18160873382456144, epsilon = 1e-6);

        // CDF at x = 2 (t(5) distribution: scipy.stats.t.cdf(2, 5) ≈ 0.94903)
        let cdf_at_two = t5.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.9490302605850708, epsilon = 1e-6);
    }

    #[test]
    fn test_student_t_ppf() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // Test PPF at median
        let median = t5.ppf(0.5).expect("test/example should not fail");
        assert_relative_eq!(median, 0.0, epsilon = 1e-10);

        // Test PPF at 95th percentile (scipy.stats.t.ppf(0.95, 5) ≈ 2.01505)
        let p95 = t5.ppf(0.95).expect("test/example should not fail");
        assert_relative_eq!(p95, 2.0150483733330233, epsilon = 1e-6);

        // Test PPF at 5th percentile (symmetric)
        let p05 = t5.ppf(0.05).expect("test/example should not fail");
        assert_relative_eq!(p05, -2.0150483733330233, epsilon = 1e-6);

        // Test PPF at 97.5th percentile (scipy.stats.t.ppf(0.975, 5) ≈ 2.57058)
        let p975 = t5.ppf(0.975).expect("test/example should not fail");
        assert_relative_eq!(p975, 2.5705818356363146, epsilon = 1e-6);

        // Verify PPF is inverse of CDF
        let cdf_p95 = t5.cdf(p95);
        assert_relative_eq!(cdf_p95, 0.95, epsilon = 1e-6);
    }

    #[test]
    fn test_student_t_rvs() {
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // Generate samples using Vec method
        let samples_vec = t5.rvs_vec(1000).expect("test/example should not fail");
        assert_eq!(samples_vec.len(), 1000);

        // Generate samples using Array1 method
        let samples_array = t5.rvs(1000).expect("test/example should not fail");
        assert_eq!(samples_array.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples_vec.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to 0 (within reason for random samples)
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_student_t_distribution_trait() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // Check mean and variance
        assert_relative_eq!(t5.mean(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(t5.var(), 5.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(t5.std(), (5.0 / 3.0f64).sqrt(), epsilon = 1e-10);

        // Test with t-distribution where mean/variance are undefined
        let t1 = StudentT::new(1.0, 0.0, 1.0).expect("test/example should not fail");
        assert!(t1.mean().is_nan());
        assert!(t1.var().is_nan());
        assert!(t1.std().is_nan());

        // Check that entropy returns a reasonable value
        let entropy = t5.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_student_t_continuous_distribution_trait() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // Test as a ContinuousDistribution
        let dist: &dyn ContinuousDistribution<f64> = &t5;

        // Check PDF
        assert_relative_eq!(dist.pdf(0.0), 0.3796, epsilon = 1e-4);

        // Check CDF
        assert_relative_eq!(dist.cdf(0.0), 0.5, epsilon = 1e-10);

        // Check PPF
        assert_relative_eq!(
            dist.ppf(0.5).expect("test/example should not fail"),
            0.0,
            epsilon = 1e-10
        );

        // Check derived methods using concrete type
        assert_relative_eq!(t5.sf(0.0), 0.5, epsilon = 1e-10);
        assert!(t5.hazard(0.0) > 0.0);
        assert!(t5.cumhazard(0.0) > 0.0);

        // Check that isf and ppf are consistent
        assert_relative_eq!(
            t5.isf(0.95).expect("test/example should not fail"),
            dist.ppf(0.05).expect("test/example should not fail"),
            epsilon = 1e-6
        );
    }
}
