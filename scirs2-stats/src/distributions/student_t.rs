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
use statrs::function::gamma::{digamma, ln_gamma};
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

        // Evaluated in log space:
        //   ln pdf = ln Γ((df+1)/2) - ln Γ(df/2) - ln(sqrt(df*pi))
        //            - ((df+1)/2) * ln(1 + x^2/df) - ln(scale)
        //
        // The previous version formed the ratio Γ((df+1)/2) / Γ(df/2)
        // explicitly from a recursive gamma approximation. Both gamma values
        // overflow to `inf` for df >= 343, so the ratio became `inf` and then
        // `inf / inf = NaN` for df >= 344, even though the density itself is
        // perfectly well behaved there (it converges to the standard normal).
        // Same defect class as cool-japan/scirs#131.
        let x_f64: f64 = match NumCast::from(x_std) {
            Some(value) => value,
            None => return F::nan(),
        };
        let df_f64: f64 = match NumCast::from(self.df) {
            Some(value) => value,
            None => return F::nan(),
        };
        let scale_f64: f64 = match NumCast::from(self.scale) {
            Some(value) => value,
            None => return F::nan(),
        };

        if x_f64.is_nan() || df_f64.is_nan() {
            return F::nan();
        }

        // The density vanishes in the tails.
        if x_f64.is_infinite() {
            return F::zero();
        }

        let df_half = df_f64 / 2.0;
        // `ln_1p` keeps the last digits for |x| << sqrt(df).
        let ln_pdf = ln_gamma(df_half + 0.5)
            - ln_gamma(df_half)
            - 0.5 * (df_f64 * PI).ln()
            - (df_half + 0.5) * (x_f64 * x_f64 / df_f64).ln_1p()
            - scale_f64.ln();

        F::from(ln_pdf.exp()).unwrap_or_else(F::nan)
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    ///
    /// Uses the regularized incomplete beta function for accuracy matching scipy.
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

        // Handle NaN
        if x_std.is_nan() {
            return F::nan();
        }

        // Handle infinities
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

        // Convert to f64 for statrs beta_reg computation
        let x_f64: f64 = NumCast::from(x_std).unwrap_or(0.0);
        let df_f64: f64 = NumCast::from(self.df).unwrap_or(1.0);

        // Regularized incomplete beta function approach:
        // h = df / (df + x^2)
        // ib = 0.5 * I_h(df/2, 0.5)
        // CDF = ib if x <= 0, 1 - ib if x > 0
        let h = df_f64 / (df_f64 + x_f64 * x_f64);
        let ib = 0.5 * beta_reg(df_f64 / 2.0, 0.5, h);

        let result = if x_f64 <= 0.0 { ib } else { 1.0 - ib };

        const_f64::<F>(result)
    }

    /// Survival function `P(X > x) = 1 - CDF(x)`, evaluated directly.
    ///
    /// Computing `1 - cdf(x)` loses every digit once the CDF rounds to 1
    /// (e.g. beyond ~8 standard deviations for a normal), so the upper tail is
    /// computed from its own closed or regularized form instead.
    pub fn sf(&self, x: F) -> F {
        let (Some(t), Some(df)) = (
            <f64 as NumCast>::from((x - self.loc) / self.scale),
            <f64 as NumCast>::from(self.df),
        ) else {
            return F::nan();
        };
        if t.is_nan() || df.is_nan() {
            return F::nan();
        }
        if t == f64::INFINITY {
            return F::zero();
        }
        if t == f64::NEG_INFINITY {
            return F::one();
        }
        // One tail is 0.5 * I_{df/(df+t^2)}(df/2, 1/2); take it directly for
        // t > 0 and its complement (no cancellation: it is >= 0.5) otherwise.
        let h = (df / (df + t * t)).clamp(0.0, 1.0);
        let tail = 0.5 * beta_reg(df / 2.0, 0.5, h);
        let sf = if t > 0.0 { tail } else { 1.0 - tail };
        F::from(sf.clamp(0.0, 1.0)).unwrap_or_else(F::nan)
    }

    /// Inverse survival function: the `x` with `sf(x) = q`.
    ///
    /// By symmetry `isf(q) = 2*loc - ppf(q)`, which keeps full precision for
    /// small `q` where `ppf(1 - q)` would round `1 - q` to 1.
    pub fn isf(&self, q: F) -> StatsResult<F> {
        let lower = <Self as ContinuousDistribution<F>>::ppf(self, q)?;
        Ok(self.loc + self.loc - lower)
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
        // Differential entropy of the (scaled) t-distribution:
        //
        //   h = ((df+1)/2) * [psi((df+1)/2) - psi(df/2)]
        //       + ln( sqrt(df) * B(df/2, 1/2) ) + ln(scale)
        //
        // with `ln B(df/2, 1/2) = lnΓ(df/2) + lnΓ(1/2) - lnΓ((df+1)/2)`.
        // Evaluating it through log-gamma and digamma keeps it finite for
        // every df: the previous code built `Γ(1/2) / Γ(df/2)` from a
        // recursive gamma approximation, which overflows to `inf` for
        // df >= 284 and therefore returned `-inf` for every df in
        // [284, 1000] (a separate `df > 1000` branch hid the problem above
        // that). It also dropped the digamma terms, which are what make the
        // value converge to the normal entropy 0.5*ln(2*pi*e) as df -> inf.
        // Same defect class as cool-japan/scirs#131.
        if self.df <= F::zero() {
            return F::nan();
        }

        let df_f64: f64 = match NumCast::from(self.df) {
            Some(value) => value,
            None => return F::nan(),
        };
        let scale_f64: f64 = match NumCast::from(self.scale) {
            Some(value) => value,
            None => return F::nan(),
        };

        if df_f64.is_nan() {
            return F::nan();
        }

        let df_half = df_f64 / 2.0;
        let df_half_plus_half = df_half + 0.5;

        let ln_beta = ln_gamma(df_half) + ln_gamma(0.5) - ln_gamma(df_half_plus_half);
        let entropy = df_half_plus_half * (digamma(df_half_plus_half) - digamma(df_half))
            + 0.5 * df_f64.ln()
            + ln_beta
            + scale_f64.ln();

        F::from(entropy).unwrap_or_else(F::nan)
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
        // Inverse CDF using the inverse regularized incomplete beta function
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

        let p_f64: f64 = NumCast::from(p).unwrap_or(0.5);
        let df_f64: f64 = NumCast::from(self.df).unwrap_or(1.0);

        // Use inverse beta regularized function:
        // p1 = min(p, 1-p)
        // y = inv_beta_reg(df/2, 0.5, 2*p1)
        // t = sqrt(df * (1-y) / y)
        // sign from p >= 0.5
        let p1 = p_f64.min(1.0 - p_f64);
        let y = inv_beta_reg(df_f64 / 2.0, 0.5, 2.0 * p1);

        let t_value = if y == 0.0 {
            // Edge case: y=0 means infinite quantile
            f64::INFINITY
        } else {
            (df_f64 * (1.0 - y) / y).sqrt()
        };

        let signed_t = if p_f64 >= 0.5 { t_value } else { -t_value };

        Ok(const_f64::<F>(signed_t) * self.scale + self.loc)
    }
}

impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ContinuousCDF<F>
    for StudentT<F>
{
    /// Direct upper tail (see the inherent `sf`), not `1 - cdf`.
    fn sf(&self, x: F) -> F {
        StudentT::sf(self, x)
    }

    /// Tail-accurate inverse survival function (see the inherent `isf`).
    fn isf(&self, q: F) -> StatsResult<F> {
        StudentT::isf(self, q)
    }
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

        // CDF at x = 1 (scipy: 0.8183916424979924)
        let cdf_at_one = t5.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.8183916424979924, epsilon = 1e-6);

        // CDF at x = -1 (by symmetry)
        let cdf_at_neg_one = t5.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 1.0 - 0.8183916424979924, epsilon = 1e-6);

        // CDF at x = 2 (scipy: 0.9490302071648776)
        let cdf_at_two = t5.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.9490302071648776, epsilon = 1e-6);
    }

    #[test]
    fn test_student_t_ppf() {
        // t distribution with 5 degrees of freedom
        let t5 = StudentT::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // Test PPF at median
        let median = t5.ppf(0.5).expect("test/example should not fail");
        assert_relative_eq!(median, 0.0, epsilon = 1e-10);

        // Test PPF at 95th percentile (scipy: 2.0150483726691575)
        let p95 = t5.ppf(0.95).expect("test/example should not fail");
        assert_relative_eq!(p95, 2.0150483726691575, epsilon = 1e-6);

        // Test PPF at 5th percentile (symmetric)
        let p05 = t5.ppf(0.05).expect("test/example should not fail");
        assert_relative_eq!(p05, -2.0150483726691575, epsilon = 1e-6);

        // CDF/PPF round-trip check
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = t5.ppf(p).expect("test/example should not fail");
            let p_roundtrip = t5.cdf(x);
            assert_relative_eq!(p_roundtrip, p, epsilon = 1e-6);
        }
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

    /// Regression test for the numerical-stability defect class of
    /// cool-japan/scirs#131. The density used to be assembled from an explicit
    /// `Gamma((df+1)/2) / Gamma(df/2)` ratio built with a recursive gamma
    /// approximation: that overflows to `inf` at df = 343 and to `inf / inf =
    /// NaN` from df = 344 on, so `pdf` was unusable for large samples. It is
    /// now evaluated in log space. Reference values computed independently
    /// with mpmath at 50 digits, NOT derived from this crate.
    #[test]
    fn test_student_t_pdf_matches_reference_values_including_large_df() {
        let cases: &[(f64, f64, f64)] = &[
            (0.0, 1.0, 0.31830988618379067),
            (2.0, 1.0, 0.063661977236758134),
            (0.0, 5.0, 0.37960668982249443),
            (1.5, 5.0, 0.12451734464635514),
            (0.0, 343.0, 0.39865161249775365),
            (0.5, 343.0, 0.35169668609708479),
            (0.0, 400.0, 0.39869301963792928),
            (2.0, 400.0, 0.054225452978854039),
            (0.0, 1000.0, 0.39884255731385816),
            (3.0, 10000.0, 0.0044387186123801527),
            (0.0, 1_000_000.0, 0.39894218066587504),
        ];

        for &(x, df, expected) in cases {
            let dist = StudentT::new(df, 0.0, 1.0).expect("test/example should not fail");
            let pdf = dist.pdf(x);
            assert!(
                pdf.is_finite(),
                "pdf({x}) for df = {df} is not finite: {pdf}"
            );
            // The tolerance leaves room for the cancellation in
            // `ln_gamma(df/2 + 1/2) - ln_gamma(df/2)`: those two log-gammas
            // grow like df*ln(df)/2 while their difference stays ~ln(df)/2, so
            // the density keeps ~12 significant digits up to df = 1e4 and
            // ~9 at the df = 1e6 stress point below (it used to be `NaN`
            // for anything past df = 343).
            let relative = ((pdf - expected) / expected).abs();
            assert!(
                relative < 1e-8,
                "pdf({x}) for df = {df}: got {pdf:e}, want {expected:e} (relative {relative:e})"
            );
            // The density is symmetric about the location parameter.
            assert_relative_eq!(dist.pdf(-x), pdf, epsilon = 1e-15);
        }
    }

    /// Location/scale handling must survive the move to log space.
    #[test]
    fn test_student_t_pdf_location_scale_large_df() {
        let dist = StudentT::new(500.0, 2.0, 3.0).expect("test/example should not fail");
        let standard = StudentT::new(500.0, 0.0, 1.0).expect("test/example should not fail");

        assert_relative_eq!(dist.pdf(2.0), standard.pdf(0.0) / 3.0, epsilon = 1e-14);
        assert_relative_eq!(dist.pdf(5.0), standard.pdf(1.0) / 3.0, epsilon = 1e-14);
        assert_eq!(dist.pdf(f64::INFINITY), 0.0);
        assert!(dist.pdf(f64::NAN).is_nan());
    }

    /// `entropy` used to return `-inf` for every df in [284, 1000] (the
    /// recursive gamma overflowing inside `ln(Gamma(1/2) / Gamma(df/2))`) and
    /// omitted the digamma terms, so even the values it did produce were off.
    /// Reference values: `((df+1)/2)*(psi((df+1)/2) - psi(df/2)) +
    /// ln(sqrt(df)*B(df/2, 1/2))` at 50 digits in mpmath.
    #[test]
    fn test_student_t_entropy_matches_reference_values() {
        let cases: &[(f64, f64)] = &[
            (1.0, 2.5310242469692908),
            (2.0, 1.960279229160082),
            (5.0, 1.627502672414396),
            (10.0, 1.5212624929756808),
            (30.0, 1.4525433297872075),
            (283.0, 1.4224752162638239),
            (284.0, 1.4224627522535827),
            (343.0, 1.4218561059255316),
            (500.0, 1.420939531869349),
            (1000.0, 1.4199387830378814),
            (10000.0, 1.4190385357045061),
            (1_000_000.0, 1.4189395332049227),
        ];

        for &(df, expected) in cases {
            let dist = StudentT::new(df, 0.0, 1.0).expect("test/example should not fail");
            let entropy = ScirsDist::entropy(&dist);
            assert!(
                entropy.is_finite(),
                "entropy for df = {df} is not finite: {entropy}"
            );
            let relative = ((entropy - expected) / expected).abs();
            assert!(
                relative < 1e-10,
                "entropy for df = {df}: got {entropy}, want {expected} (relative {relative:e})"
            );
        }

        // The values decrease monotonically towards the standard normal
        // entropy 0.5*ln(2*pi*e), which the t-distribution approaches as
        // df -> infinity.
        let normal_entropy = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln();
        let mut previous = f64::INFINITY;
        for &(df, _) in cases {
            let dist = StudentT::new(df, 0.0, 1.0).expect("test/example should not fail");
            let entropy = ScirsDist::entropy(&dist);
            assert!(
                entropy < previous && entropy > normal_entropy,
                "entropy for df = {df} ({entropy}) must lie between the previous value \
                 ({previous}) and the normal limit ({normal_entropy})"
            );
            previous = entropy;
        }
        // df = 1e6 is already within 1e-6 of the limit. (Beyond ~1e9 the
        // digamma difference psi(df/2 + 1/2) - psi(df/2) ~ 1/df is itself
        // cancellation-limited, so the value stays finite and correct to a few
        // digits rather than to full precision -- it used to be -inf.)
        assert!((previous - normal_entropy).abs() < 2e-6, "got {previous}");

        // Scaling shifts the entropy by ln(scale).
        let scaled = StudentT::new(500.0, 0.0, 3.0).expect("test/example should not fail");
        let standard = StudentT::new(500.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(
            ScirsDist::entropy(&scaled),
            ScirsDist::entropy(&standard) + 3.0_f64.ln(),
            epsilon = 1e-12
        );
    }
}
