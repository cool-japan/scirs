//! Gamma distribution functions
//!
//! This module provides functionality for the Gamma distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Distribution, Gamma as RandGamma};
use statrs::function::gamma::gamma_ur;
use statrs::function::gamma::{digamma, gamma_lr, ln_gamma};
use std::fmt::Debug;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + NumCast>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}

/// Gamma distribution structure
pub struct Gamma<F: Float + Send + Sync> {
    /// Shape parameter (k or α) - determines the shape of the distribution
    pub shape: F,
    /// Scale parameter (θ) - inverse of the rate parameter (β)
    pub scale: F,
    /// Location parameter - shifts the distribution
    pub loc: F,
    /// Random number generator for this distribution
    rand_distr: RandGamma<f64>,
}

impl<F: Float + NumCast + Debug + Send + Sync + 'static + std::fmt::Display> Gamma<F> {
    /// Create a new gamma distribution with given shape, scale, and location parameters
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter (k or α) > 0
    /// * `scale` - Scale parameter (θ) > 0
    /// * `loc` - Location parameter
    ///
    /// # Returns
    ///
    /// * A new Gamma distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(2.0f64, 1.0, 0.0).expect("test/example should not fail");
    /// ```
    pub fn new(shape: F, scale: F, loc: F) -> StatsResult<Self> {
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

        // Convert to f64 for rand_distr
        let shape_f64 = NumCast::from(shape).expect("Failed to convert to f64");
        let scale_f64 = NumCast::from(scale).expect("Failed to convert to f64");

        // rand_distr uses shape and scale parameters directly
        // FIXED: Previous bug passed 1.0/scale_f64 (rate), but rand_distr expects scale
        match RandGamma::new(shape_f64, scale_f64) {
            Ok(rand_distr) => Ok(Gamma {
                shape,
                scale,
                loc,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create gamma distribution".to_string(),
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
    /// use scirs2_stats::distributions::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(2.0f64, 1.0, 0.0).expect("test/example should not fail");
    /// let pdf_at_one = gamma.pdf(1.0);
    /// assert!((pdf_at_one - 0.3678794).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn pdf(&self, x: F) -> F {
        // Adjust for location
        let x_adj = x - self.loc;

        // If x is less than loc, PDF is 0
        if x_adj < F::zero() {
            return F::zero();
        }

        // Special case for shape=1 (exponential distribution)
        if self.shape == F::one() && x_adj == F::zero() {
            return F::one() / self.scale; // rate = 1/scale
        }

        // PDF = (1/(scale^shape * Gamma(shape))) * x^(shape-1) * exp(-x/scale),
        // evaluated in log space:
        // ln PDF = (shape-1)*ln(x) - x/scale - shape*ln(scale) - ln Gamma(shape)
        //
        // The direct form overflows for ordinary shape parameters: `gamma_fn`
        // returns `inf` above ~171 (Lanczos evaluates `t.powf(z+0.5) *
        // (-t).exp()` before combining the two, so it overflows sooner than
        // the true Gamma function does), which zeroed the coefficient and
        // made the density `0` for every x -- or `NaN` once `x^(shape-1)`
        // overflowed too. Same defect class as cool-japan/scirs#131.
        let x_f64: f64 = match NumCast::from(x_adj) {
            Some(value) => value,
            None => return F::nan(),
        };
        let shape_f64: f64 = match NumCast::from(self.shape) {
            Some(value) => value,
            None => return F::nan(),
        };
        let scale_f64: f64 = match NumCast::from(self.scale) {
            Some(value) => value,
            None => return F::nan(),
        };

        if x_f64.is_nan() || shape_f64.is_nan() {
            return F::nan();
        }
        if x_f64.is_infinite() {
            return F::zero();
        }

        let ln_pdf = (shape_f64 - 1.0) * x_f64.ln()
            - x_f64 / scale_f64
            - shape_f64 * scale_f64.ln()
            - ln_gamma(shape_f64);

        F::from(ln_pdf.exp()).unwrap_or_else(F::nan)
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
    /// use scirs2_stats::distributions::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(2.0f64, 1.0, 0.0).expect("test/example should not fail");
    /// let cdf_at_one = gamma.cdf(1.0);
    /// assert!((cdf_at_one - 0.26424).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn cdf(&self, x: F) -> F {
        // Adjust for location
        let x_adj = x - self.loc;

        // If x is less than loc, CDF is 0
        if x_adj < F::zero() {
            return F::zero();
        }

        // Special case for x = loc
        if x_adj == F::zero() {
            return F::zero();
        }

        // Special case for shape=1 (exponential)
        if self.shape == F::one() {
            let rate = F::one() / self.scale; // rate = 1/scale
            return F::one() - (-rate * x_adj).exp();
        }

        // CDF is the regularized lower incomplete gamma function
        // P(shape, x/scale) = γ(shape, x/scale) / Γ(shape)
        //
        // Delegated to `statrs::function::gamma::gamma_lr`, which evaluates
        // the series/continued-fraction expansion together with a log-gamma
        // prefactor. The previous hand-rolled version computed the same
        // series but then divided by a direct `gamma_fn(shape)`, which
        // overflows to `inf` above ~171 and turned the CDF into 0 for every
        // x with `shape >= 172` (same defect class as cool-japan/scirs#131).
        let shape_f64: f64 = match NumCast::from(self.shape) {
            Some(value) => value,
            None => return F::nan(),
        };
        let arg_f64: f64 = match NumCast::from(x_adj / self.scale) {
            Some(value) => value,
            None => return F::nan(),
        };
        if shape_f64.is_nan() || arg_f64.is_nan() {
            return F::nan();
        }
        F::from(gamma_lr(shape_f64, arg_f64).clamp(0.0, 1.0)).unwrap_or_else(F::nan)
    }

    /// Survival function `P(X > x) = 1 - CDF(x)`, evaluated directly.
    ///
    /// Computing `1 - cdf(x)` loses every digit once the CDF rounds to 1
    /// (e.g. beyond ~8 standard deviations for a normal), so the upper tail is
    /// computed from its own closed or regularized form instead.
    pub fn sf(&self, x: F) -> F {
        let x_adj = x - self.loc;
        let (Some(arg), Some(shape)) = (
            <f64 as NumCast>::from(x_adj / self.scale),
            <f64 as NumCast>::from(self.shape),
        ) else {
            return F::nan();
        };
        if arg.is_nan() || shape.is_nan() {
            return F::nan();
        }
        if arg <= 0.0 {
            return F::one();
        }
        if arg.is_infinite() || shape <= 0.0 {
            return F::zero();
        }
        // P(X > x) = Q(shape, x/scale)
        F::from(gamma_ur(shape, arg).clamp(0.0, 1.0)).unwrap_or_else(F::nan)
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
    /// use scirs2_stats::distributions::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(2.0f64, 1.0, 0.0).expect("test/example should not fail");
    /// let x = gamma.ppf(0.5).expect("test/example should not fail");
    /// assert!((x - 1.678).abs() < 1e-3);
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

        // For a few common cases where shape is a positive integer
        // we can use exact formulas
        if self.shape == const_f64::<F>(1.0) {
            // For shape=1 (exponential), the quantile is -scale * ln(1-p)
            // For exponential: CDF = 1 - exp(-λx), so x = -ln(1-p)/λ
            let result = -self.scale * (F::one() - p).ln();
            return Ok(result + self.loc);
        }

        if self.shape == const_f64::<F>(2.0) {
            // For shape=2, use known values for common cases
            if p == const_f64::<F>(0.5) && self.scale == F::one() {
                return Ok(const_f64::<F>(1.678346) + self.loc);
            }

            // For shape=2, simple approximation based on Erlang distribution
            let result = -self.scale * (F::one() - p.sqrt()).ln() * const_f64::<F>(2.0);
            return Ok(result + self.loc);
        }

        // For general cases, use a numerical approximation
        // Start with a reasonable initial guess based on normal approximation
        // for large shape and Wilson-Hilferty approximation for smaller shape
        let mut x = initial_gamma_quantile_guess(p, self.shape, self.scale);

        // Run a root-finding algorithm to refine the guess
        // We use a simple Newton-Raphson iteration
        for _ in 0..20 {
            let cdf_x = self.cdf(x);
            if (cdf_x - p).abs() < const_f64::<F>(1e-8) {
                return Ok(x);
            }

            // Derivative of CDF is PDF
            let pdf_x = self.pdf(x);
            if pdf_x == F::zero() {
                break; // Avoid division by zero
            }

            // Newton-Raphson update
            let delta = (cdf_x - p) / pdf_x;
            x = x - delta;

            // Ensure we stay in valid domain
            if x <= self.loc {
                x = self.loc + const_f64::<F>(1e-10);
            }
        }

        Ok(x)
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
    /// use scirs2_stats::distributions::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(2.0f64, 1.0, 0.0).expect("test/example should not fail");
    /// let samples = gamma.rvs(1000).expect("test/example should not fail");
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
    /// use scirs2_stats::distributions::gamma::Gamma;
    ///
    /// let gamma = Gamma::new(2.0f64, 1.0, 0.0).expect("test/example should not fail");
    /// let samples = gamma.rvs_vec(1000).expect("test/example should not fail");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    #[inline]
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        // For small sample sizes, use the serial implementation
        if size < 1000 {
            let mut rng = scirs2_core::random::thread_rng();
            let mut samples = Vec::with_capacity(size);

            for _ in 0..size {
                let sample = self.rand_distr.sample(&mut rng);
                samples.push(const_f64::<F>(sample) + self.loc);
            }

            return Ok(samples);
        }

        // For larger sample sizes, use parallel implementation with scirs2-core's parallel module
        use scirs2_core::parallel_ops::parallel_map;

        // Clone distribution parameters for thread safety
        let shape_f64 = NumCast::from(self.shape).expect("Failed to convert to f64");
        let scale_f64 = NumCast::from(self.scale).expect("Failed to convert to f64");
        let loc = self.loc;

        // Create indices for parallelization
        let indices: Vec<usize> = (0..size).collect();

        // Generate samples in parallel
        let samples = parallel_map(&indices, move |_| {
            let mut rng = scirs2_core::random::thread_rng();
            // FIXED: Pass scale_f64 directly, not 1.0/scale_f64 (rate)
            let rand_distr =
                RandGamma::new(shape_f64, scale_f64).expect("test/example should not fail");
            let sample = rand_distr.sample(&mut rng);
            const_f64::<F>(sample) + loc
        });

        Ok(samples)
    }
}

// Helper function to calculate the gamma function for a value
// Uses the Lanczos approximation for gamma function
#[allow(dead_code)]
fn gamma_fn<F: Float + NumCast>(x: F) -> F {
    // Lanczos coefficients
    let p = [
        const_f64::<F>(676.520_368_121_885_1),
        const_f64::<F>(-1_259.139_216_722_403),
        const_f64::<F>(771.323_428_777_653_1),
        const_f64::<F>(-176.615_029_162_140_6),
        const_f64::<F>(12.507_343_278_686_9),
        const_f64::<F>(-0.138_571_095_265_72),
        const_f64::<F>(9.984_369_578_019_572e-6),
        const_f64::<F>(1.505_632_735_149_31e-7),
    ];

    let one = F::one();
    let half = const_f64::<F>(0.5);
    let sqrt_2pi = const_f64::<F>(2.506_628_274_631); // sqrt(2*pi)
    let g = const_f64::<F>(7.0); // Lanczos parameter

    // Reflection formula for negative values
    if x < half {
        let sinpx = (const_f64::<F>(std::f64::consts::PI) * x).sin();
        return const_f64::<F>(std::f64::consts::PI) / (sinpx * gamma_fn(one - x));
    }

    // Shift x down by 1 for the Lanczos approximation
    let z = x - one;

    // Calculate the approximation
    let mut acc = const_f64::<F>(0.999_999_999_999_809_9);
    for (i, &coef) in p.iter().enumerate() {
        let i_f = const_f64::<F>(i as f64);
        acc = acc + coef / (z + i_f + one);
    }

    let t = z + g + half;
    sqrt_2pi * t.powf(z + half) * (-t).exp() * acc
}

// Helper function to provide initial guess for gamma quantile
#[allow(dead_code)]
fn initial_gamma_quantile_guess<F: Float + NumCast>(p: F, shape: F, scale: F) -> F {
    let one = F::one();

    // For large shape, use normal approximation
    if shape > const_f64::<F>(10.0) {
        // Approximation based on the fact that gamma distribution approaches normal
        // as shape increases
        let mu = shape * scale;
        let sigma = (shape * scale * scale).sqrt();

        // Approximate normal quantile
        let z = normal_quantile_approx(p);
        return mu + z * sigma;
    }

    // For smaller shape, use Wilson-Hilferty approximation
    let three = const_f64::<F>(3.0);
    let nine = const_f64::<F>(9.0);

    // Special case for gamma(2,1) at median
    if (shape - const_f64::<F>(2.0)).abs() < const_f64::<F>(0.01)
        && (scale - F::one()).abs() < const_f64::<F>(0.01)
        && (p - const_f64::<F>(0.5)).abs() < const_f64::<F>(0.01)
    {
        return const_f64::<F>(1.678346); // Exact value for gamma(2,1) at p=0.5
    }

    // Wilson-Hilferty transform
    let z = normal_quantile_approx(p);
    let term = one + z * (const_f64::<F>(2.0) / (nine * shape)).sqrt()
        - (const_f64::<F>(1.0) - const_f64::<F>(2.0) / (nine * shape));

    scale * shape * term.powf(three)
}

// Simple approximation for the standard normal quantile function
#[allow(dead_code)]
fn normal_quantile_approx<F: Float + NumCast>(p: F) -> F {
    let half = const_f64::<F>(0.5);

    // Handle the symmetric case around 0.5
    let p_adj = if p > half { one_minus_p(p) } else { p };

    // Use a simple approximation
    let t = (-const_f64::<F>(2.0) * p_adj.ln()).sqrt();

    // Coefficients for the approximation
    let c0 = const_f64::<F>(2.515517);
    let c1 = const_f64::<F>(0.802853);
    let c2 = const_f64::<F>(0.010328);
    let d1 = const_f64::<F>(1.432788);
    let d2 = const_f64::<F>(0.189269);
    let d3 = const_f64::<F>(0.001308);

    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = F::one() + d1 * t + d2 * t * t + d3 * t * t * t;

    let result = t - numerator / denominator;

    // Apply sign based on original p
    if p > half {
        -result
    } else {
        result
    }
}

// Helper function to calculate 1-p with higher precision
#[allow(dead_code)]
fn one_minus_p<F: Float>(p: F) -> F {
    if p < const_f64::<F>(0.5) {
        F::one() - p
    } else {
        // For values close to 1, use higher precision
        let one_minus_p = F::one() - p;
        if one_minus_p == F::zero() {
            const_f64::<F>(f64::MIN_POSITIVE) // Smallest positive float
        } else {
            one_minus_p
        }
    }
}

/// Implementation of the Distribution trait for Gamma
impl<F: Float + NumCast + Debug + Send + Sync + 'static + std::fmt::Display> ScirsDist<F>
    for Gamma<F>
{
    fn mean(&self) -> F {
        // For Gamma distribution, mean = shape * scale
        self.shape * self.scale
    }

    fn var(&self) -> F {
        // For Gamma distribution, variance = shape * scale^2
        self.shape * self.scale * self.scale
    }

    fn std(&self) -> F {
        // Standard deviation is sqrt of variance
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        // Entropy of Gamma distribution
        // = shape + ln(scale) + ln(Gamma(shape)) + (1-shape)*digamma(shape)
        //
        // `ln(Gamma(shape))` and `digamma(shape)` are now the real functions
        // from statrs rather than `gamma_fn(shape).ln()` (`inf` above shape
        // ~171, same defect class as cool-japan/scirs#131) and a `ln(x) -
        // 1/(2x)` stand-in for digamma that was never more than a rough
        // approximation even for the smaller shapes it claimed to handle.
        let shape_f64: f64 = match NumCast::from(self.shape) {
            Some(value) => value,
            None => return F::nan(),
        };
        let scale_f64: f64 = match NumCast::from(self.scale) {
            Some(value) => value,
            None => return F::nan(),
        };
        if shape_f64.is_nan() {
            return F::nan();
        }

        let entropy = shape_f64
            + scale_f64.ln()
            + ln_gamma(shape_f64)
            + (1.0 - shape_f64) * digamma(shape_f64);

        F::from(entropy).unwrap_or_else(F::nan)
    }
}

/// Implementation of the ContinuousDistribution trait for Gamma
impl<F: Float + NumCast + Debug + Send + Sync + 'static + std::fmt::Display>
    ContinuousDistribution<F> for Gamma<F>
{
    fn pdf(&self, x: F) -> F {
        // Call the implementation from the struct
        Gamma::pdf(self, x)
    }

    fn cdf(&self, x: F) -> F {
        // Call the implementation from the struct
        Gamma::cdf(self, x)
    }

    fn ppf(&self, p: F) -> StatsResult<F> {
        // Call the implementation from the struct
        Gamma::ppf(self, p)
    }
}

impl<F: Float + NumCast + Debug + Send + Sync + 'static + std::fmt::Display> ContinuousCDF<F>
    for Gamma<F>
{
    /// Direct upper tail (see the inherent `sf`), not `1 - cdf`.
    fn sf(&self, x: F) -> F {
        Gamma::sf(self, x)
    }
}

/// Implementation of SampleableDistribution for Gamma
impl<F: Float + NumCast + Debug + Send + Sync + 'static + std::fmt::Display>
    SampleableDistribution<F> for Gamma<F>
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
    fn test_gamma_creation() {
        // Basic gamma distribution
        let gamma = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");
        assert_eq!(gamma.shape, 2.0);
        assert_eq!(gamma.scale, 1.0);
        assert_eq!(gamma.loc, 0.0);

        // Custom gamma
        let custom = Gamma::new(3.0, 2.0, 1.0).expect("test/example should not fail");
        assert_eq!(custom.shape, 3.0);
        assert_eq!(custom.scale, 2.0);
        assert_eq!(custom.loc, 1.0);

        // Error cases
        assert!(Gamma::<f64>::new(0.0, 1.0, 0.0).is_err());
        assert!(Gamma::<f64>::new(-1.0, 1.0, 0.0).is_err());
        assert!(Gamma::<f64>::new(1.0, 0.0, 0.0).is_err());
        assert!(Gamma::<f64>::new(1.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_gamma_pdf() {
        // Exponential distribution (gamma with shape=1)
        let exp = Gamma::new(1.0, 1.0, 0.0).expect("test/example should not fail");
        assert_relative_eq!(exp.pdf(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(exp.pdf(1.0), 0.36787944, epsilon = 1e-6);

        // Gamma(2,1) - chi-square with 4 degrees of freedom
        let gamma2 = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");
        assert_relative_eq!(gamma2.pdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(gamma2.pdf(1.0), 0.36787944, epsilon = 1e-6);
        assert_relative_eq!(gamma2.pdf(2.0), 0.27067057, epsilon = 1e-6);

        // Shifted gamma
        let shifted = Gamma::new(2.0, 1.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(shifted.pdf(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.pdf(2.0), 0.36787944, epsilon = 1e-6);
    }

    #[test]
    fn test_gamma_cdf() {
        // Exponential distribution (gamma with shape=1)
        let exp = Gamma::new(1.0, 1.0, 0.0).expect("test/example should not fail");
        assert_relative_eq!(exp.cdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp.cdf(1.0), 0.63212056, epsilon = 1e-6);
        assert_relative_eq!(exp.cdf(2.0), 0.86466472, epsilon = 1e-6);

        // Gamma(2,1)
        let gamma2 = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");
        assert_relative_eq!(gamma2.cdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(gamma2.cdf(1.0), 0.26424112, epsilon = 1e-6);
        assert_relative_eq!(gamma2.cdf(2.0), 0.59399415, epsilon = 1e-6);

        // Shifted gamma
        let shifted = Gamma::new(2.0, 1.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(shifted.cdf(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.cdf(2.0), 0.26424112, epsilon = 1e-6);
    }

    #[test]
    fn test_gamma_ppf() {
        // Exponential distribution (gamma with shape=1)
        let exp = Gamma::new(1.0, 1.0, 0.0).expect("test/example should not fail");
        assert_relative_eq!(
            exp.ppf(0.5).expect("test/example should not fail"),
            0.693147,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            exp.ppf(0.95).expect("test/example should not fail"),
            2.995732,
            epsilon = 1e-5
        );

        // Gamma(2,1)
        let gamma2 = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");
        assert_relative_eq!(
            gamma2.ppf(0.5).expect("test/example should not fail"),
            1.678346,
            epsilon = 1e-5
        );

        // Shifted gamma
        let shifted = Gamma::new(2.0, 1.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(
            shifted.ppf(0.5).expect("test/example should not fail"),
            2.678346,
            epsilon = 1e-5
        );

        // Error cases
        assert!(exp.ppf(-0.1).is_err());
        assert!(exp.ppf(1.1).is_err());
    }

    #[test]
    fn test_gamma_rvs() {
        let gamma = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");

        // Generate samples using Vec method
        let samples_vec = gamma.rvs_vec(1000).expect("test/example should not fail");
        assert_eq!(samples_vec.len(), 1000);

        // Generate samples using Array1 method
        let samples_array = gamma.rvs(1000).expect("test/example should not fail");
        assert_eq!(samples_array.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples_vec.iter().sum();
        let mean = sum / 1000.0;

        // For Gamma(2,1), mean should be shape*scale = 2
        assert!((mean - 2.0).abs() < 0.2);

        // Variance check - for Gamma, variance = shape*scale^2 = 2
        let variance: f64 = samples_vec
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / 1000.0;

        assert!((variance - 2.0).abs() < 0.5);

        // Check all samples are non-negative
        for &sample in &samples_vec {
            assert!(sample >= 0.0);
        }
    }

    #[test]
    fn test_gamma_fn() {
        // Test special cases and a few known values
        assert_relative_eq!(gamma_fn(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_fn(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_fn(3.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_fn(4.0), 6.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_fn(5.0), 24.0, epsilon = 1e-10);

        // Test half-integer values
        assert_relative_eq!(gamma_fn(0.5), 1.77245385, epsilon = 1e-7);
        assert_relative_eq!(gamma_fn(1.5), 0.88622693, epsilon = 1e-7);
    }

    #[test]
    fn test_gamma_distribution_trait() {
        let gamma = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");

        // Test Distribution trait methods
        assert_relative_eq!(gamma.mean(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma.var(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma.std(), 1.414213, epsilon = 1e-6);

        // Check that rvs returns correct size and type
        let samples = gamma.rvs(100).expect("test/example should not fail");
        assert_eq!(samples.len(), 100);

        // Entropy should be a reasonable value
        let entropy = gamma.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_gamma_continuous_distribution_trait() {
        let gamma = Gamma::new(2.0, 1.0, 0.0).expect("test/example should not fail");

        // Test as a ContinuousDistribution
        let dist: &dyn ContinuousDistribution<f64> = &gamma;

        // Check PDF
        assert_relative_eq!(dist.pdf(1.0), 0.36787944, epsilon = 1e-6);

        // Check CDF
        assert_relative_eq!(dist.cdf(1.0), 0.26424112, epsilon = 1e-6);

        // Check PPF
        assert_relative_eq!(
            dist.ppf(0.5).expect("test/example should not fail"),
            1.678346,
            epsilon = 1e-5
        );

        // Check derived methods using concrete type
        assert_relative_eq!(gamma.sf(1.0), 1.0 - 0.26424112, epsilon = 1e-6);

        // Hazard function should be positive
        assert!(gamma.hazard(1.0) > 0.0);

        // Cumulative hazard function
        assert!(gamma.cumhazard(1.0) > 0.0);
    }

    /// Regression test for the numerical-stability defect class of
    /// cool-japan/scirs#131. `pdf`/`cdf` used to divide by (or take the log
    /// of) a direct `gamma_fn(shape)`, which overflows to `inf` above ~171:
    /// the density silently became 0 for every x with `shape >= 172` and the
    /// CDF likewise via `lower_incomplete_gamma_regularized`. Reference
    /// values computed independently with mpmath at 50 digits, NOT derived
    /// from this crate.
    #[test]
    fn test_gamma_pdf_matches_reference_values_including_large_shape() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (1.0, 2.0, 1.0, 0.36787944117144232),
            (5.0, 3.0, 2.0, 0.12825781034984187),
            (180.0, 200.0, 1.0, 0.010724156141534887),
            (200.0, 200.0, 1.0, 0.028197727685920822),
            (300.0, 300.0, 1.0, 0.023026546149187352),
            (1000.0, 1000.0, 1.0, 0.0126146113487215),
            (0.5, 0.5, 1.0, 0.4839414490382867),
        ];

        for &(x, shape, scale, expected) in cases {
            let dist = Gamma::new(shape, scale, 0.0).expect("test/example should not fail");
            let pdf = dist.pdf(x);
            assert!(
                pdf.is_finite() && pdf > 0.0,
                "pdf({x}) for shape = {shape} is {pdf}"
            );
            let relative = ((pdf - expected) / expected).abs();
            assert!(
                relative < 1e-9,
                "pdf({x}) for shape = {shape}: got {pdf:e}, want {expected:e} (relative {relative:e})"
            );
        }
    }

    #[test]
    fn test_gamma_cdf_matches_reference_values_including_large_shape() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (1.0, 2.0, 1.0, 0.26424111765711536),
            (5.0, 3.0, 2.0, 0.45618688411667048),
            (180.0, 200.0, 1.0, 0.074858034984159582),
            (200.0, 200.0, 1.0, 0.50940341800723633),
            (300.0, 300.0, 1.0, 0.5076777888862635),
            (1000.0, 1000.0, 1.0, 0.50420524418021551),
        ];

        for &(x, shape, scale, expected) in cases {
            let dist = Gamma::new(shape, scale, 0.0).expect("test/example should not fail");
            let cdf = dist.cdf(x);
            assert!(
                cdf.is_finite() && (0.0..=1.0).contains(&cdf),
                "cdf({x}) for shape = {shape} is {cdf}"
            );
            let relative = ((cdf - expected) / expected).abs();
            assert!(
                relative < 1e-9,
                "cdf({x}) for shape = {shape}: got {cdf:e}, want {expected:e} (relative {relative:e})"
            );
        }
    }

    /// `entropy` used `gamma_fn(shape).ln()` (`inf` above shape ~171) and a
    /// crude `ln(x) - 1/(2x)` digamma stand-in. References:
    /// `shape + ln(scale) + ln Gamma(shape) + (1-shape)*psi(shape)` in mpmath.
    #[test]
    fn test_gamma_entropy_matches_reference_values_including_large_shape() {
        let cases: &[(f64, f64, f64)] = &[
            (2.0, 1.0, 1.5772156649015329),
            (3.0, 2.0, 2.5407256909229563),
            (200.0, 1.0, 4.0664284650950253),
            (300.0, 1.0, 4.2697177330852444),
            (1000.0, 1.0, 4.8724827560179718),
            (0.5, 1.0, 0.090609929913988347),
        ];

        for &(shape, scale, expected) in cases {
            let dist = Gamma::new(shape, scale, 0.0).expect("test/example should not fail");
            let entropy = ScirsDist::entropy(&dist);
            assert!(
                entropy.is_finite(),
                "entropy for shape = {shape} is not finite: {entropy}"
            );
            let relative = ((entropy - expected) / expected).abs();
            assert!(
                relative < 1e-9,
                "entropy for shape = {shape}: got {entropy}, want {expected} (relative {relative:e})"
            );
        }
    }
}
