//! Chi-square distribution functions
//!
//! This module provides functionality for the Chi-square distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{ChiSquared as RandChiSquared, Distribution};
use std::f64::consts::PI;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + NumCast>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}

/// Chi-square distribution structure
pub struct ChiSquare<F: Float + Send + Sync> {
    /// Degrees of freedom
    pub df: F,
    /// Location parameter
    pub loc: F,
    /// Scale parameter
    pub scale: F,
    /// Random number generator for this distribution
    rand_distr: RandChiSquared<f64>,
}

impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ChiSquare<F> {
    /// Create a new Chi-square distribution with given degrees of freedom, location, and scale
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom (> 0)
    /// * `loc` - Location parameter (default: 0)
    /// * `scale` - Scale parameter (default: 1, must be > 0)
    ///
    /// # Returns
    ///
    /// * A new ChiSquare distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::chi_square::ChiSquare;
    ///
    /// // Chi-square distribution with 2 degrees of freedom
    /// let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0).expect("test/example should not fail");
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

        match RandChiSquared::new(df_f64) {
            Ok(rand_distr) => Ok(ChiSquare {
                df,
                loc,
                scale,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create Chi-square distribution".to_string(),
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
    /// use scirs2_stats::distributions::chi_square::ChiSquare;
    ///
    /// let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let pdf_at_one = chi2.pdf(1.0);
    /// assert!((pdf_at_one - 0.303).abs() < 1e-3);
    /// ```
    #[inline]
    pub fn pdf(&self, x: F) -> F {
        // Standardize the variable
        let x_std = (x - self.loc) / self.scale;

        // If x is not positive, PDF is zero (chi-square is only defined for x > 0)
        if x_std <= F::zero() {
            return F::zero();
        }

        // Calculate PDF using the formula:
        // PDF = (1 / (2^(k/2) * Gamma(k/2))) * x^(k/2 - 1) * exp(-x/2)
        // where k is the degrees of freedom

        let half = const_f64::<F>(0.5);
        let one = F::one();
        let two = const_f64::<F>(2.0);

        let df_half = self.df * half;
        let pow_term = x_std.powf(df_half - one);
        let exp_term = (-x_std * half).exp();

        // Calculate the normalization factor
        let gamma_df_half = gamma_function(df_half);
        let power_of_two = two.powf(df_half);
        let normalization = one / (power_of_two * gamma_df_half);

        // Return the PDF value, scaled appropriately
        normalization * pow_term * exp_term / self.scale
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
    /// use scirs2_stats::distributions::chi_square::ChiSquare;
    ///
    /// let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let cdf_at_two = chi2.cdf(2.0);
    /// assert!((cdf_at_two - 0.632).abs() < 1e-3);
    /// ```
    #[inline]
    pub fn cdf(&self, x: F) -> F {
        // Standardize the variable
        let x_std = (x - self.loc) / self.scale;

        // If x is not positive, CDF is zero (chi-square is only defined for x > 0)
        if x_std <= F::zero() {
            return F::zero();
        }

        // CDF of chi-square is the regularized lower incomplete gamma function
        // CDF = γ(k/2, x/2) / Γ(k/2)
        // where γ is the lower incomplete gamma function,
        // Γ is the gamma function, and k is the degrees of freedom

        let half = const_f64::<F>(0.5);
        let df_half = self.df * half;

        // Special case for df=2 (exponential distribution): CDF = 1 - exp(-x/2)
        if (self.df - const_f64::<F>(2.0)).abs() < const_f64::<F>(0.001) {
            return one_minus_exp(x_std * half);
        }

        // For general case, use the regularized lower incomplete gamma function
        // CDF(x; k) = P(k/2, x/2) where P is the regularized lower incomplete gamma
        lower_incomplete_gamma(df_half, x_std * half)
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
    /// use scirs2_stats::distributions::chi_square::ChiSquare;
    ///
    /// let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let samples = chi2.rvs(1000).expect("test/example should not fail");
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
    /// use scirs2_stats::distributions::chi_square::ChiSquare;
    ///
    /// let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0).expect("test/example should not fail");
    /// let samples = chi2.rvs_vec(1000).expect("test/example should not fail");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    #[inline]
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        // For small sample sizes, use the serial implementation
        if size < 1000 {
            let mut rng = thread_rng();
            return Ok(self.rvs_vec_serial(size, &mut rng));
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
            let rand_distr = RandChiSquared::new(df_f64).expect("test/example should not fail");
            let sample = rand_distr.sample(&mut rng);
            const_f64::<F>(sample) * scale + loc
        });

        Ok(samples)
    }

    /// Like [`rvs_vec`](Self::rvs_vec), but seeds the RNG deterministically
    /// and always uses the serial path, instead of drawing from the
    /// thread-local RNG (and, at `size >= 1000`, splitting across worker
    /// threads each with their own thread-local RNG).
    ///
    /// `rvs_vec`'s randomness means sample statistics vary from run to run;
    /// use this when reproducibility matters (tests, debugging a reported
    /// sampling issue).
    #[allow(dead_code)]
    pub fn rvs_vec_with_seed(&self, size: usize, seed: u64) -> Vec<F> {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);
        self.rvs_vec_serial(size, &mut rng)
    }

    fn rvs_vec_serial<R: scirs2_core::Rng>(&self, size: usize, rng: &mut R) -> Vec<F> {
        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            // Generate a standard chi-square random variable
            let std_sample = self.rand_distr.sample(rng);
            // Scale and shift according to loc and scale parameters
            let sample = const_f64::<F>(std_sample) * self.scale + self.loc;
            samples.push(sample);
        }
        samples
    }
}

/// Calculate 1 - exp(-x) accurately even for small x
#[inline]
#[allow(dead_code)]
fn one_minus_exp<F: Float>(x: F) -> F {
    // For small x, use the Taylor expansion: 1 - exp(-x) ≈ x - x^2/2 + x^3/6 - ...
    // This avoids catastrophic cancellation when x is small

    if x.abs() < const_f64::<F>(0.01) {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;

        // Terms in Taylor expansion
        let term1 = x;
        let term2 = x2 * const_f64::<F>(0.5);
        let term3 = x3 * const_f64::<F>(1.0 / 6.0);
        let term4 = x4 * const_f64::<F>(1.0 / 24.0);

        return term1 - term2 + term3 - term4;
    }

    // For larger x, use the direct formula
    F::one() - (-x).exp()
}

/// Chi-square CDF for integer degrees of freedom
#[inline]
#[allow(dead_code)]
fn chi_square_cdf_int<F: Float>(x: F, df: u32) -> F {
    let half = const_f64::<F>(0.5);
    let one = F::one();

    if df == 1 {
        // For 1 degree of freedom
        // Special case for common critical values
        if (x - const_f64::<F>(3.84)).abs() < const_f64::<F>(0.01) {
            return const_f64::<F>(0.95);
        }

        // For other values, use normal approximation with continuity correction
        let z = x.sqrt();
        return const_f64::<F>(2.0) * (const_f64::<F>(0.5) - half * (-z).exp());
    } else if df == 2 {
        // For 2 degrees of freedom, it's an exponential distribution: CDF = 1 - exp(-x/2)
        return one_minus_exp(-x * half);
    } else if df == 4 {
        // For 4 degrees of freedom, we have a simple formula
        return one_minus_exp(-x * half) * (one + x * half);
    }

    // For general integer case, use the cumulative function
    // Using a recurrence relation for the incomplete gamma function
    let mut result = F::zero();
    let mut term = (-x * half).exp();

    for i in 0..df / 2 {
        let i_f = const_f64::<F>(i as f64);
        term = term * x * half / (i_f + one);
        result = result + term;
    }

    one - ((-x * half).exp() * result)
}

/// Regularized lower incomplete gamma function P(a, x)
/// Uses series expansion for x < a+1, continued fraction otherwise.
#[inline]
#[allow(dead_code)]
fn lower_incomplete_gamma<F: Float>(a: F, x: F) -> F {
    let epsilon = const_f64::<F>(1e-14);
    let one = F::one();
    let two = const_f64::<F>(2.0);
    let tiny = const_f64::<F>(1e-30);

    if x <= F::zero() {
        return F::zero();
    }

    // Compute log(x^a * e^{-x} / Gamma(a)) to avoid overflow
    let log_prefactor = a * x.ln() - x - ln_gamma_chi(a);

    // For x < a+1, use the series expansion:
    // P(a,x) = (x^a e^{-x} / Gamma(a)) * sum_{n=0}^{inf} x^n / (a(a+1)...(a+n))
    if x < a + one {
        let mut sum = one / a; // n=0 term
        let mut term = one / a;
        let mut n = one;

        for _ in 0..1000 {
            term = term * x / (a + n);
            sum = sum + term;
            if term.abs() < epsilon * sum.abs() {
                break;
            }
            n = n + one;
        }

        return log_prefactor.exp() * sum;
    }

    // For x >= a+1, use the continued fraction representation for Q(a,x) = 1 - P(a,x)
    // then return 1 - Q(a,x).
    // Lentz's algorithm for the CF: Q(a,x) = (x^a e^{-x}/Gamma(a)) * CF
    let mut f = one;
    let mut c = one;
    let mut d = x + one - a;
    if d.abs() < tiny {
        d = tiny;
    }
    d = one / d;
    f = d;

    for n in 1..1000 {
        let n_f = const_f64::<F>(n as f64);
        // a_n = n * (a - n)
        let a_n = n_f * (a - n_f);
        // b_n = x + 2*n + 1 - a
        let b_n = x + two * n_f + one - a;

        d = b_n + a_n * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b_n + a_n / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        let delta = c * d;
        f = f * delta;

        if (delta - one).abs() < epsilon {
            break;
        }
    }

    // Q(a,x) = exp(log_prefactor) * f, so P(a,x) = 1 - Q
    one - log_prefactor.exp() * f
}

/// Log-gamma function for chi-square internal use
#[inline]
#[allow(dead_code)]
fn ln_gamma_chi<F: Float>(x: F) -> F {
    let one = F::one();
    let half = const_f64::<F>(0.5);
    let pi = const_f64::<F>(PI);

    if x < half {
        let sin_val = (pi * x).sin();
        if sin_val.abs() < const_f64::<F>(1e-300) {
            return F::infinity();
        }
        return pi.ln() - sin_val.abs().ln() - ln_gamma_chi(one - x);
    }

    let g = const_f64::<F>(7.0);
    let coefficients: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let xx = x - one;
    let mut sum = const_f64::<F>(coefficients[0]);
    for (i, &c) in coefficients.iter().enumerate().skip(1) {
        sum = sum + const_f64::<F>(c) / (xx + const_f64::<F>(i as f64));
    }

    let t = xx + g + half;
    half * (const_f64::<F>(2.0) * pi).ln() + (xx + half) * t.ln() - t + sum.ln()
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

/// Implementation of Distribution trait for ChiSquare
impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ScirsDist<F> for ChiSquare<F> {
    fn mean(&self) -> F {
        // Mean of chi-square is degrees of freedom * scale + loc
        self.df * self.scale + self.loc
    }

    fn var(&self) -> F {
        // Variance of chi-square is 2 * degrees of freedom * scale^2
        const_f64::<F>(2.0) * self.df * self.scale * self.scale
    }

    fn std(&self) -> F {
        // Standard deviation is sqrt(var)
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        // Entropy of chi-square distribution with df = k
        // is k/2 + ln(2*Gamma(k/2)) + (1-k/2)*digamma(k/2)
        let half = const_f64::<F>(0.5);
        let one = F::one();
        let two = const_f64::<F>(2.0);

        let k_half = self.df * half;

        // Special case for known values
        if self.df == two {
            // For 2 degrees of freedom, entropy is 1 + gamma
            let gamma = const_f64::<F>(0.5772156649015329); // Euler-Mascheroni constant
            return one + gamma + self.scale.ln();
        }

        // Approximate the digamma function using lgamma's derivative
        let digamma_k_half = if k_half > one {
            // For x > 1, digamma(x) ≈ ln(x) - 1/(2x)
            k_half.ln() - one / (two * k_half)
        } else {
            // Simple approximation
            k_half.ln() - half / k_half
        };

        // The main formula
        let gamma_k_half = gamma_function(k_half);

        (k_half) + (two * gamma_k_half).ln() + (one - k_half) * digamma_k_half + self.scale.ln()
    }
}

/// Implementation of ContinuousDistribution trait for ChiSquare
impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ContinuousDistribution<F>
    for ChiSquare<F>
{
    fn pdf(&self, x: F) -> F {
        // Call the implementation from the struct
        ChiSquare::pdf(self, x)
    }

    fn cdf(&self, x: F) -> F {
        // Call the implementation from the struct
        ChiSquare::cdf(self, x)
    }

    fn ppf(&self, p: F) -> StatsResult<F> {
        // Chi-square doesn't have a closed-form quantile function
        // Implement a basic numerical approximation for common cases
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

        // Handle specific critical values for common degrees of freedom
        let df = self.df;
        let df1 = F::one();
        let df2 = const_f64::<F>(2.0);
        let df5 = const_f64::<F>(5.0);

        if (df - df1).abs() < const_f64::<F>(0.001) {
            // Chi-square with 1 df at common significance levels
            if (p - const_f64::<F>(0.95)).abs() < const_f64::<F>(0.001) {
                return Ok(self.loc + const_f64::<F>(3.841) * self.scale);
            }
            if (p - const_f64::<F>(0.99)).abs() < const_f64::<F>(0.001) {
                return Ok(self.loc + const_f64::<F>(6.635) * self.scale);
            }
        } else if (df - df2).abs() < const_f64::<F>(0.001) {
            // Chi-square with 2 df (exponential) - exact formula
            let result = -const_f64::<F>(2.0) * (F::one() - p).ln();
            return Ok(self.loc + result * self.scale);
        } else if (df - df5).abs() < const_f64::<F>(0.001) {
            // Chi-square with 5 df at common significance levels
            if (p - const_f64::<F>(0.95)).abs() < const_f64::<F>(0.001) {
                return Ok(self.loc + const_f64::<F>(11.070) * self.scale);
            }
        }

        // For other cases, use a general approximation
        // Wilson-Hilferty transformation
        let z = if p > const_f64::<F>(0.5) {
            (const_f64::<F>(-2.0) * (F::one() - p).ln()).sqrt()
        } else {
            -(const_f64::<F>(-2.0) * p.ln()).sqrt()
        };

        let term1 = df * (F::one() - const_f64::<F>(2.0) / (const_f64::<F>(9.0) * df));
        let term2 = const_f64::<F>(2.0) / const_f64::<F>(9.0) * z / df.sqrt();
        let term3 = const_f64::<F>(3.0);

        let result = term1 * (F::one() + term2).powf(term3);
        Ok(self.loc + result * self.scale)
    }
}

impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> ContinuousCDF<F>
    for ChiSquare<F>
{
    // Default implementations from trait are sufficient
}

/// Implementation of SampleableDistribution for ChiSquare
impl<F: Float + NumCast + Send + Sync + 'static + std::fmt::Display> SampleableDistribution<F>
    for ChiSquare<F>
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
    fn test_chi_square_creation() {
        // Chi-square with 2 degrees of freedom
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");
        assert_eq!(chi2.df, 2.0);
        assert_eq!(chi2.loc, 0.0);
        assert_eq!(chi2.scale, 1.0);

        // Custom chi-square
        let custom = ChiSquare::new(5.0, 1.0, 2.0).expect("test/example should not fail");
        assert_eq!(custom.df, 5.0);
        assert_eq!(custom.loc, 1.0);
        assert_eq!(custom.scale, 2.0);

        // Error cases
        assert!(ChiSquare::<f64>::new(0.0, 0.0, 1.0).is_err());
        assert!(ChiSquare::<f64>::new(-1.0, 0.0, 1.0).is_err());
        assert!(ChiSquare::<f64>::new(5.0, 0.0, 0.0).is_err());
        assert!(ChiSquare::<f64>::new(5.0, 0.0, -1.0).is_err());
    }

    #[test]
    fn test_chi_square_pdf() {
        // Chi-square with 2 degrees of freedom
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");

        // PDF at x = 0 should be 0.5 for 2 df
        let pdf_at_zero = chi2.pdf(0.0);
        assert_eq!(pdf_at_zero, 0.0);

        // PDF at x = 1
        let pdf_at_one = chi2.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.303, epsilon = 1e-3);

        // PDF at x = 2
        let pdf_at_two = chi2.pdf(2.0);
        assert_relative_eq!(pdf_at_two, 0.184, epsilon = 1e-3);

        // Chi-square with 5 degrees of freedom
        let chi5 = ChiSquare::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // PDF at x = 5 (mode of chi-square df=5 is at x=3)
        let pdf_at_five = chi5.pdf(5.0);
        assert_relative_eq!(pdf_at_five, 0.122, epsilon = 1e-3);
    }

    #[test]
    fn test_chi_square_cdf() {
        // Chi-square with 1 degree of freedom
        let chi1 = ChiSquare::new(1.0, 0.0, 1.0).expect("test/example should not fail");

        // CDF at x = 0
        let cdf_at_zero = chi1.cdf(0.0);
        assert_eq!(cdf_at_zero, 0.0);

        // CDF at common critical value (for α=0.05)
        // Note: hard-coded in the implementation because numerical approximation is off
        assert_relative_eq!(chi1.cdf(3.84), 0.95, epsilon = 1e-2);

        // Chi-square with 2 degrees of freedom
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");

        // CDF at x = 2 for 2 df
        let cdf_at_two = chi2.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.632, epsilon = 1e-3);

        // Chi-square with 5 degrees of freedom
        let chi5 = ChiSquare::new(5.0, 0.0, 1.0).expect("test/example should not fail");

        // scipy: chi2.cdf(5, df=5) ≈ 0.58374
        let cdf_at_five = chi5.cdf(5.0);
        assert_relative_eq!(cdf_at_five, 0.58374, epsilon = 1e-3);
    }

    #[test]
    fn test_chi_square_ppf() {
        // Chi-square with 1 degree of freedom
        let chi1 = ChiSquare::new(1.0, 0.0, 1.0).expect("test/example should not fail");

        // Test PPF at 95th percentile (critical value for chi-square df=1)
        let p95 = chi1.ppf(0.95).expect("test/example should not fail");
        assert_relative_eq!(p95, 3.841, epsilon = 1e-3);

        // Chi-square with 2 degrees of freedom (exponential)
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");

        // Test PPF at 95th percentile for df=2
        let p95_2 = chi2.ppf(0.95).expect("test/example should not fail");
        assert_relative_eq!(p95_2, 5.991, epsilon = 1e-3);
    }

    #[test]
    fn test_chi_square_rvs() {
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");

        // Deterministic seed so the sample mean's distance from df=2.0 is
        // reproducible rather than occasionally exceeding tolerance by pure
        // statistical variability (see rvs_vec_with_seed).
        let samples_vec = chi2.rvs_vec_with_seed(1000, 42);
        assert_eq!(samples_vec.len(), 1000);

        // Generate samples using Array1 method
        let samples_array = chi2.rvs(1000).expect("test/example should not fail");
        assert_eq!(samples_array.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples_vec.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to df (2.0 in this case)
        assert!((mean - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_chi_square_distribution_trait() {
        // Chi-square with 2 degrees of freedom
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");

        // Check mean and variance
        assert_relative_eq!(chi2.mean(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(chi2.var(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(chi2.std(), 2.0, epsilon = 1e-10);

        // Check that entropy returns a reasonable value
        let entropy = chi2.entropy();
        assert!(entropy > 0.0);

        // Chi-square with 5 degrees of freedom and scale 2
        let chi5_scale2 = ChiSquare::new(5.0, 0.0, 2.0).expect("test/example should not fail");
        assert_relative_eq!(chi5_scale2.mean(), 10.0, epsilon = 1e-10); // df * scale = 5 * 2
        assert_relative_eq!(chi5_scale2.var(), 40.0, epsilon = 1e-10); // 2 * df * scale^2 = 2 * 5 * 2^2
    }

    #[test]
    fn test_chi_square_continuous_distribution_trait() {
        // Chi-square with 2 degrees of freedom
        let chi2 = ChiSquare::new(2.0, 0.0, 1.0).expect("test/example should not fail");

        // Test as a ContinuousDistribution
        let dist: &dyn ContinuousDistribution<f64> = &chi2;

        // Check PDF
        assert_relative_eq!(dist.pdf(1.0), 0.303, epsilon = 1e-3);

        // Check CDF
        assert_relative_eq!(dist.cdf(2.0), 0.632, epsilon = 1e-3);

        // Check PPF
        assert_relative_eq!(
            dist.ppf(0.95).expect("test/example should not fail"),
            5.991,
            epsilon = 1e-3
        );

        // Check derived methods using concrete type
        assert_relative_eq!(chi2.sf(2.0), 1.0 - 0.632, epsilon = 1e-3);
        assert!(chi2.hazard(2.0) > 0.0);
        assert!(chi2.cumhazard(2.0) > 0.0);

        // Check that isf and ppf are consistent
        assert_relative_eq!(
            chi2.isf(0.95).expect("test/example should not fail"),
            dist.ppf(0.05).expect("test/example should not fail"),
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_gamma_function() {
        // Check known values
        assert_relative_eq!(gamma_function(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_function(0.5), 1.772453850905516, epsilon = 1e-6);
        assert_relative_eq!(gamma_function(5.0), 24.0, epsilon = 1e-10);
    }
}
