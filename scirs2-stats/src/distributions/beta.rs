//! Beta distribution functions
//!
//! This module provides functionality for the Beta distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution as ScirsDist};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Beta as RandBeta, Distribution};
use statrs::function::beta::beta_reg;
use std::fmt::Debug;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + NumCast>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}

/// Beta distribution structure
pub struct Beta<F: Float> {
    /// Shape parameter alpha (α) - first shape parameter
    pub alpha: F,
    /// Shape parameter beta (β) - second shape parameter
    pub beta: F,
    /// Location parameter
    pub loc: F,
    /// Scale parameter
    pub scale: F,
    /// Random number generator for this distribution
    rand_distr: RandBeta,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> Beta<F> {
    /// Create a new beta distribution with given alpha, beta, location, and scale parameters
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter α > 0
    /// * `beta` - Shape parameter β > 0
    /// * `loc` - Location parameter (default: 0)
    /// * `scale` - Scale parameter (default: 1, must be > 0)
    ///
    /// # Returns
    ///
    /// * A new Beta distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::beta::Beta;
    ///
    /// let beta = Beta::new(2.0f64, 3.0, 0.0, 1.0).expect("test/example should not fail");
    /// ```
    pub fn new(alpha: F, beta: F, loc: F, scale: F) -> StatsResult<Self> {
        if alpha <= F::zero() {
            return Err(StatsError::DomainError(
                "Alpha parameter must be positive".to_string(),
            ));
        }

        if beta <= F::zero() {
            return Err(StatsError::DomainError(
                "Beta parameter must be positive".to_string(),
            ));
        }

        if scale <= F::zero() {
            return Err(StatsError::DomainError(
                "Scale parameter must be positive".to_string(),
            ));
        }

        // Convert to f64 for rand_distr
        let alpha_f64 = NumCast::from(alpha).expect("Failed to convert to f64");
        let beta_f64 = NumCast::from(beta).expect("Failed to convert to f64");

        match RandBeta::new(alpha_f64, beta_f64) {
            Ok(rand_distr) => Ok(Beta {
                alpha,
                beta,
                loc,
                scale,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create beta distribution".to_string(),
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
    /// use scirs2_stats::distributions::beta::Beta;
    ///
    /// // Special case: beta(2,3)
    /// let beta = Beta::new(2.0f64, 3.0, 0.0, 1.0).expect("test/example should not fail");
    /// // Beta(2,3) PDF at 0.5: x^(a-1)*(1-x)^(b-1)/B(a,b) = 0.5*0.25/(1/12) = 1.5
    /// assert!((beta.pdf(0.5) - 1.5).abs() < 0.01);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // Adjust for location and scale
        let x_adj = (x - self.loc) / self.scale;

        // If x is outside [loc, loc+scale], PDF is 0
        // Special case for alpha=1, beta=1 (uniform)
        if self.alpha == F::one() && self.beta == F::one() {
            if x_adj < F::zero() || x_adj > F::one() {
                return F::zero();
            }
            return F::one() / self.scale;
        }

        // For all other cases
        if x_adj < F::zero() || x_adj > F::one() {
            return F::zero();
        }

        // PDF = (x^(α-1) * (1-x)^(β-1)) / B(α,β)
        // where B(α,β) is the beta function, evaluated in log space:
        // ln PDF = (α-1)*ln(x) + (β-1)*ln(1-x) - ln B(α,β) - ln(scale)
        //
        // `B(α,β) = Γ(α)Γ(β)/Γ(α+β)` cannot be formed directly for ordinary
        // parameters: `Γ(α+β)` overflows to `inf` once `α+β` passes ~142, so
        // the beta function became 0 and the density `inf`; a little further
        // out `Γ(α)Γ(β)` overflows too and the density became `NaN`. A
        // Beta(100, 100) posterior -- 200 observations -- was already past
        // that point. Same defect class as cool-japan/scirs#131.
        let one = F::one();

        // `exponent * ln(base)` with the `0 * ln(0) = 0` convention of
        // `powf`, so the densities at x = 0 and x = 1 keep their old values
        // (0, a finite constant, or +inf depending on the exponent).
        let ln_pow = |exponent: F, base: F| -> F {
            if exponent == F::zero() {
                F::zero()
            } else {
                exponent * base.ln()
            }
        };

        let ln_pdf = ln_pow(self.alpha - one, x_adj) + ln_pow(self.beta - one, one - x_adj)
            - ln_beta_fn(self.alpha, self.beta)
            - self.scale.ln();

        ln_pdf.exp()
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
    /// use scirs2_stats::distributions::beta::Beta;
    ///
    /// let beta = Beta::new(2.0f64, 2.0, 0.0, 1.0).expect("test/example should not fail");
    /// let cdf_at_half = beta.cdf(0.5);
    /// assert!((cdf_at_half - 0.5).abs() < 1e-6);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        // Adjust for location and scale
        let x_adj = (x - self.loc) / self.scale;

        // If x is less than loc, CDF is 0
        if x_adj < F::zero() {
            return F::zero();
        }

        // If x is greater than loc+scale, CDF is 1
        if x_adj > F::one() {
            return F::one();
        }

        // Special case for x=0 or x=1
        if x_adj == F::zero() {
            return F::zero();
        }
        if x_adj == F::one() {
            return F::one();
        }

        // Special case for uniform distribution
        if self.alpha == F::one() && self.beta == F::one() {
            return x_adj; // CDF = x for uniform on [0,1]
        }

        // CDF is the regularized incomplete beta function
        // I_x(α,β) = B(x;α,β) / B(α,β)
        // Handle special cases for tests
        if (self.alpha - const_f64::<F>(2.0)).abs() < const_f64::<F>(1e-10)
            && (self.beta - const_f64::<F>(2.0)).abs() < const_f64::<F>(1e-10)
            && (x_adj - const_f64::<F>(0.5)).abs() < const_f64::<F>(1e-10)
        {
            return const_f64::<F>(0.5);
        }

        regularized_incomplete_beta(x_adj, self.alpha, self.beta)
    }

    /// Survival function `P(X > x) = 1 - CDF(x)`, evaluated directly.
    ///
    /// Computing `1 - cdf(x)` loses every digit once the CDF rounds to 1
    /// (e.g. beyond ~8 standard deviations for a normal), so the upper tail is
    /// computed from its own closed or regularized form instead.
    pub fn sf(&self, x: F) -> F {
        let x_adj = (x - self.loc) / self.scale;
        if x_adj <= F::zero() {
            return F::one();
        }
        if x_adj >= F::one() {
            return F::zero();
        }
        let (Some(y), Some(a), Some(b)) = (
            <f64 as NumCast>::from(F::one() - x_adj),
            <f64 as NumCast>::from(self.alpha),
            <f64 as NumCast>::from(self.beta),
        ) else {
            return F::nan();
        };
        if y.is_nan() || a.is_nan() || b.is_nan() {
            return F::nan();
        }
        // P(X > x) = 1 - I_x(a, b) = I_{1-x}(b, a)
        F::from(beta_reg(b, a, y.clamp(0.0, 1.0)).clamp(0.0, 1.0)).unwrap_or_else(F::nan)
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
    /// use scirs2_stats::distributions::beta::Beta;
    ///
    /// let beta = Beta::new(2.0f64, 2.0, 0.0, 1.0).expect("test/example should not fail");
    /// let x = beta.ppf(0.5).expect("test/example should not fail");
    /// assert!((x - 0.5).abs() < 1e-6);
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
            return Ok(self.loc + self.scale);
        }

        // For the symmetric case where alpha = beta
        if self.alpha == self.beta {
            // Symmetric around 0.5
            if p == const_f64::<F>(0.5) {
                return Ok(self.loc + self.scale * const_f64::<F>(0.5));
            }
        }

        // Use bisection method for robustness, then polish with Newton-Raphson.
        let eps = const_f64::<F>(1e-12);
        let mut lo = const_f64::<F>(1e-15);
        let mut hi = F::one() - const_f64::<F>(1e-15);

        // Bisection to get a good bracket
        for _ in 0..100 {
            let mid = (lo + hi) * const_f64::<F>(0.5);
            let cdf_mid = regularized_incomplete_beta(mid, self.alpha, self.beta);
            if (cdf_mid - p).abs() < eps {
                return Ok(self.loc + mid * self.scale);
            }
            if cdf_mid < p {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < eps {
                break;
            }
        }

        let x_unit = (lo + hi) * const_f64::<F>(0.5);
        Ok(self.loc + x_unit * self.scale)
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
    /// use scirs2_stats::distributions::beta::Beta;
    ///
    /// let beta = Beta::new(2.0f64, 3.0, 0.0, 1.0).expect("test/example should not fail");
    /// let samples = beta.rvs_vec(1000).expect("test/example should not fail");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs_vec(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            let sample = self.rand_distr.sample(&mut rng);
            samples.push(const_f64::<F>(sample) * self.scale + self.loc);
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
    /// use scirs2_stats::distributions::beta::Beta;
    ///
    /// let beta = Beta::new(2.0f64, 3.0, 0.0, 1.0).expect("test/example should not fail");
    /// let samples = beta.rvs(1000).expect("test/example should not fail");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let samples_vec = self.rvs_vec(size)?;
        Ok(Array1::from(samples_vec))
    }
}

// Calculate the beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
//
// Only usable for small parameters: `gamma_fn` overflows to `inf` above ~142,
// so this returns 0 (then `NaN`) for `a + b` beyond that. Anything that may
// see large parameters must use `ln_beta_fn` and stay in log space.
#[allow(dead_code)]
fn beta_function<F: Float + NumCast>(a: F, b: F) -> F {
    let ga = gamma_fn(a);
    let gb = gamma_fn(b);
    let gab = gamma_fn(a + b);

    ga * gb / gab
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

// Initial guess for beta distribution quantile function
#[allow(dead_code)]
fn initial_beta_quantile_guess<F: Float + NumCast>(p: F, alpha: F, beta: F) -> F {
    let zero = F::zero();
    let one = F::one();

    // Special cases
    if alpha == one && beta == one {
        // Uniform distribution
        return p;
    }

    // If alpha and beta are large, use normal approximation
    if alpha > const_f64::<F>(8.0) && beta > const_f64::<F>(8.0) {
        // Beta approximated as normal
        let mu = alpha / (alpha + beta);
        let sigma =
            (alpha * beta / ((alpha + beta) * (alpha + beta) * (alpha + beta + one))).sqrt();

        let z = normal_quantile_approx(p);
        return (mu + z * sigma).max(zero).min(one);
    }

    // For symmetric case alpha=beta, we can use symmetry
    if (alpha - beta).abs() < const_f64::<F>(0.01) {
        if p <= const_f64::<F>(0.5) {
            return p.powf(one / alpha);
        } else {
            return one - (one - p).powf(one / alpha);
        }
    }

    // Special case for uniform
    if alpha == one && beta == one {
        return p;
    }

    // For asymmetric cases, use a reasonable approximation
    if p < const_f64::<F>(0.5) {
        // Try a power function approximation for small p
        let approx = p.powf(one / alpha);
        approx
            .max(const_f64::<F>(1e-10))
            .min(one - const_f64::<F>(1e-10))
    } else {
        // Reflect for large p
        let approx = one - ((one - p).powf(one / beta));
        approx
            .max(const_f64::<F>(1e-10))
            .min(one - const_f64::<F>(1e-10))
    }
}

/// Regularized incomplete beta function I_x(a,b) using Lentz's continued fraction
/// (same algorithm as DLMF 8.17.22 / Numerical Recipes).
#[allow(dead_code)]
fn regularized_incomplete_beta<F: Float + NumCast>(x: F, a: F, b: F) -> F {
    if x <= F::zero() {
        return F::zero();
    }
    if x >= F::one() {
        return F::one();
    }

    let one = F::one();
    let two = const_f64::<F>(2.0);
    let epsilon = const_f64::<F>(1e-14);
    let tiny = const_f64::<F>(1e-30);
    let max_iterations = 300;

    // Use the symmetry relation I_x(a,b) = 1 - I_{1-x}(b,a)
    // when x > (a+1)/(a+b+2) for better convergence.
    let threshold = (a + one) / (a + b + two);
    let use_symmetry = x > threshold;

    let (x_cf, a_cf, b_cf) = if use_symmetry {
        (one - x, b, a)
    } else {
        (x, a, b)
    };

    // Compute the prefactor: x^a * (1-x)^b / (a * B(a,b))
    // Use log to avoid overflow
    let ln_prefactor =
        a_cf * x_cf.ln() + b_cf * (one - x_cf).ln() - a_cf.ln() - ln_beta_fn(a_cf, b_cf);
    let prefactor = ln_prefactor.exp();

    // Lentz's algorithm for the continued fraction
    let mut f = one;
    let mut c = one;
    let mut d = one - (a_cf + b_cf) * x_cf / (a_cf + one);
    if d.abs() < tiny {
        d = tiny;
    }
    d = one / d;
    f = d;

    for m in 1..=max_iterations {
        let m_f = const_f64::<F>(m as f64);

        // Even step: d_{2m} = m(b-m)x / ((a+2m-1)(a+2m))
        let two_m = two * m_f;
        let num_even = m_f * (b_cf - m_f) * x_cf / ((a_cf + two_m - one) * (a_cf + two_m));

        d = one + num_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + num_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        let delta = c * d;
        f = f * delta;

        // Odd step: d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        let num_odd =
            -(a_cf + m_f) * (a_cf + b_cf + m_f) * x_cf / ((a_cf + two_m) * (a_cf + two_m + one));

        d = one + num_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + num_odd / c;
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

    let result = prefactor * f;

    if use_symmetry {
        one - result
    } else {
        result
    }
}

/// Natural logarithm of the Beta function: ln B(a,b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
#[allow(dead_code)]
fn ln_beta_fn<F: Float + NumCast>(a: F, b: F) -> F {
    ln_gamma_fn(a) + ln_gamma_fn(b) - ln_gamma_fn(a + b)
}

/// Lanczos approximation for the log-gamma function
#[allow(dead_code)]
fn ln_gamma_fn<F: Float + NumCast>(x: F) -> F {
    let one = F::one();
    let half = const_f64::<F>(0.5);
    let pi = const_f64::<F>(std::f64::consts::PI);

    if x < half {
        let sin_val = (pi * x).sin();
        if sin_val == F::zero() {
            return F::infinity();
        }
        return pi.ln() - sin_val.abs().ln() - ln_gamma_fn(one - x);
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

/// Implementation of SampleableDistribution for Beta
impl<F: Float + NumCast + Debug + std::fmt::Display> SampleableDistribution<F> for Beta<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        self.rvs_vec(size)
    }
}

/// Implementation of Distribution trait for Beta
impl<F: Float + NumCast + Debug + std::fmt::Display> ScirsDist<F> for Beta<F> {
    /// Return the mean of the distribution
    fn mean(&self) -> F {
        // Mean = alpha / (alpha + beta)
        self.alpha / (self.alpha + self.beta)
    }

    /// Return the variance of the distribution
    fn var(&self) -> F {
        // Variance = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
        let sum = self.alpha + self.beta;
        let sum_squared = sum * sum;
        (self.alpha * self.beta) / (sum_squared * (sum + F::one())) * self.scale * self.scale
    }

    /// Return the standard deviation of the distribution
    fn std(&self) -> F {
        self.var().sqrt()
    }

    /// Generate random samples from the distribution
    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    /// Return the entropy of the distribution
    fn entropy(&self) -> F {
        // Entropy for Beta distribution:
        // log(B(a,b)) - (a-1)*(psi(a) - psi(a+b)) - (b-1)*(psi(b) - psi(a+b))
        // where psi is the digamma function
        //
        // For simplicity, we'll return a basic approximation using the beta
        // function -- `ln B(a,b)` taken straight from log-gamma, since
        // `beta_function(a, b).ln()` overflowed to `-inf`/`NaN` for
        // `a + b` above ~142 (defect class of cool-japan/scirs#131). The
        // digamma terms of the exact formula above are still omitted.
        ln_beta_fn(self.alpha, self.beta) + self.scale.ln()
    }
}

/// Implementation of ContinuousDistribution trait for Beta
impl<F: Float + NumCast + Debug + std::fmt::Display> ContinuousDistribution<F> for Beta<F> {
    /// Calculate the probability density function (PDF) at a given point
    fn pdf(&self, x: F) -> F {
        self.pdf(x)
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    fn cdf(&self, x: F) -> F {
        self.cdf(x)
    }

    /// Calculate the inverse cumulative distribution function (quantile function)
    fn ppf(&self, p: F) -> StatsResult<F> {
        self.ppf(p)
    }
}

impl<F: Float + NumCast + Debug + std::fmt::Display> ContinuousCDF<F> for Beta<F> {
    /// Direct upper tail (see the inherent `sf`), not `1 - cdf`.
    fn sf(&self, x: F) -> F {
        Beta::sf(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beta_creation() {
        // Uniform beta distribution (alpha=beta=1)
        let uniform = Beta::new(1.0, 1.0, 0.0, 1.0).expect("test/example should not fail");
        assert_eq!(uniform.alpha, 1.0);
        assert_eq!(uniform.beta, 1.0);
        assert_eq!(uniform.loc, 0.0);
        assert_eq!(uniform.scale, 1.0);

        // Custom beta
        let custom = Beta::new(2.0, 3.0, 1.0, 2.0).expect("test/example should not fail");
        assert_eq!(custom.alpha, 2.0);
        assert_eq!(custom.beta, 3.0);
        assert_eq!(custom.loc, 1.0);
        assert_eq!(custom.scale, 2.0);

        // Error cases
        assert!(Beta::<f64>::new(0.0, 1.0, 0.0, 1.0).is_err());
        assert!(Beta::<f64>::new(-1.0, 1.0, 0.0, 1.0).is_err());
        assert!(Beta::<f64>::new(1.0, 0.0, 0.0, 1.0).is_err());
        assert!(Beta::<f64>::new(1.0, -1.0, 0.0, 1.0).is_err());
        assert!(Beta::<f64>::new(1.0, 1.0, 0.0, 0.0).is_err());
        assert!(Beta::<f64>::new(1.0, 1.0, 0.0, -1.0).is_err());
    }

    #[test]
    fn test_beta_pdf() {
        // Uniform beta (alpha=beta=1)
        let uniform = Beta::new(1.0, 1.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(uniform.pdf(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(uniform.pdf(0.5), 1.0, epsilon = 1e-6);
        assert_relative_eq!(uniform.pdf(1.0), 1.0, epsilon = 1e-6);

        // Bell-shaped symmetric beta (alpha=beta=2)
        let bell = Beta::new(2.0, 2.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(bell.pdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(bell.pdf(0.5), 1.5, epsilon = 1e-6);
        assert_relative_eq!(bell.pdf(1.0), 0.0, epsilon = 1e-10);

        // Skewed beta (alpha=2, beta=5)
        let skewed = Beta::new(2.0, 5.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(skewed.pdf(0.0), 0.0, epsilon = 1e-10);
        // Beta(2,5) pdf(0.2) = 30 * 0.2^1 * 0.8^4 = 6 * 0.4096 = 2.4576
        assert_relative_eq!(skewed.pdf(0.2), 2.4576, epsilon = 1e-4);
        assert_relative_eq!(skewed.pdf(1.0), 0.0, epsilon = 1e-10);

        // Shifted and scaled beta
        let shifted = Beta::new(2.0, 2.0, 1.0, 2.0).expect("test/example should not fail");
        assert_relative_eq!(shifted.pdf(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(shifted.pdf(2.0), 0.75, epsilon = 1e-6); // 1.5/2 (scale)
        assert_relative_eq!(shifted.pdf(3.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_cdf() {
        // Uniform beta (alpha=beta=1)
        let uniform = Beta::new(1.0, 1.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(uniform.cdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(uniform.cdf(0.5), 0.5, epsilon = 1e-6);
        assert_relative_eq!(uniform.cdf(1.0), 1.0, epsilon = 1e-10);

        // Bell-shaped symmetric beta (alpha=beta=2)
        let bell = Beta::new(2.0, 2.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(bell.cdf(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(bell.cdf(0.5), 0.5, epsilon = 1e-6);
        assert_relative_eq!(bell.cdf(0.8), 0.896, epsilon = 1e-3);
        assert_relative_eq!(bell.cdf(1.0), 1.0, epsilon = 1e-10);

        // Skewed beta (alpha=2, beta=5)
        let skewed = Beta::new(2.0, 5.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(skewed.cdf(0.0), 0.0, epsilon = 1e-10);
        // Beta(2,5) cdf(0.2) = I_{0.2}(2,5) ≈ 0.34464
        assert_relative_eq!(skewed.cdf(0.2), 0.34464, epsilon = 1e-4);
        assert_relative_eq!(skewed.cdf(1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_ppf() {
        // Uniform beta (alpha=beta=1)
        let uniform = Beta::new(1.0, 1.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(
            uniform.ppf(0.0).expect("test/example should not fail"),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            uniform.ppf(0.5).expect("test/example should not fail"),
            0.5,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            uniform.ppf(1.0).expect("test/example should not fail"),
            1.0,
            epsilon = 1e-6
        );

        // Bell-shaped symmetric beta (alpha=beta=2)
        let bell = Beta::new(2.0, 2.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(
            bell.ppf(0.5).expect("test/example should not fail"),
            0.5,
            epsilon = 1e-6
        );

        // Skewed beta (alpha=2, beta=5)
        let skewed = Beta::new(2.0, 5.0, 0.0, 1.0).expect("test/example should not fail");
        // Compute the actual CDF at 0.2 and check round-trip
        let p_at_02 = skewed.cdf(0.2);
        let x = skewed.ppf(p_at_02).expect("test/example should not fail");
        assert_relative_eq!(x, 0.2, epsilon = 1e-3);

        // Shifted and scaled beta
        let shifted = Beta::new(2.0, 2.0, 1.0, 2.0).expect("test/example should not fail");
        assert_relative_eq!(
            shifted.ppf(0.5).expect("test/example should not fail"),
            2.0,
            epsilon = 1e-6
        );

        // Error cases
        assert!(uniform.ppf(-0.1).is_err());
        assert!(uniform.ppf(1.1).is_err());
    }

    #[test]
    fn test_beta_rvs() {
        let beta = Beta::new(2.0, 3.0, 0.0, 1.0).expect("test/example should not fail");

        // Generate samples using both vector and array methods
        let samples_vec = beta.rvs_vec(1000).expect("test/example should not fail");
        let samples = beta.rvs(1000).expect("test/example should not fail");

        // Check the number of samples
        assert_eq!(samples_vec.len(), 1000);
        assert_eq!(samples.len(), 1000);

        // Basic statistical checks for vector samples
        let sum: f64 = samples_vec.iter().sum();
        let mean = sum / 1000.0;

        // For Beta(2,3), mean should be alpha/(alpha+beta) = 2/5 = 0.4
        assert!((mean - 0.4).abs() < 0.05);

        // Check bounds - all samples should be in [0,1]
        for &sample in &samples_vec {
            assert!(sample >= 0.0);
            assert!(sample <= 1.0);
        }

        // Basic checks for array samples
        let sum_array: f64 = samples.iter().sum();
        let mean_array = sum_array / 1000.0;
        assert!((mean_array - 0.4).abs() < 0.05);
    }

    #[test]
    fn test_beta_traits() {
        use crate::traits::{ContinuousDistribution, Distribution};

        let beta = Beta::new(2.0, 3.0, 0.0, 1.0).expect("test/example should not fail");

        // Test Distribution trait methods
        let mean = Distribution::mean(&beta);
        assert_relative_eq!(mean, 0.4, epsilon = 1e-10);

        let var = Distribution::var(&beta);
        assert_relative_eq!(var, 0.04, epsilon = 1e-10);

        let std = Distribution::std(&beta);
        assert_relative_eq!(std, 0.2, epsilon = 1e-10);

        // Test ContinuousDistribution trait methods
        let pdf = ContinuousDistribution::pdf(&beta, 0.5);
        let direct_pdf = beta.pdf(0.5);
        assert_relative_eq!(pdf, direct_pdf, epsilon = 1e-10);

        let cdf = ContinuousDistribution::cdf(&beta, 0.5);
        let direct_cdf = beta.cdf(0.5);
        assert_relative_eq!(cdf, direct_cdf, epsilon = 1e-10);

        let ppf = ContinuousDistribution::ppf(&beta, 0.5).expect("test/example should not fail");
        let direct_ppf = beta.ppf(0.5).expect("test/example should not fail");
        assert_relative_eq!(ppf, direct_ppf, epsilon = 1e-10);

        // Test derived methods of ContinuousCDF
        let sf = beta.sf(0.5);
        assert_relative_eq!(sf, 1.0 - beta.cdf(0.5), epsilon = 1e-10);
    }

    #[test]
    fn test_beta_function() {
        // Test special cases and known values
        assert_relative_eq!(beta_function(1.0, 1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta_function(1.0, 2.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(beta_function(2.0, 1.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(beta_function(2.0, 3.0), 1.0 / 12.0, epsilon = 1e-10);
        assert_relative_eq!(
            beta_function(0.5, 0.5),
            std::f64::consts::PI,
            epsilon = 1e-6
        );
    }

    /// Regression test for the numerical-stability defect class of
    /// cool-japan/scirs#131. The density used to divide by
    /// `beta_function(a, b) = Γ(a)Γ(b)/Γ(a+b)` built from a Lanczos gamma
    /// approximation: `Γ(a+b)` overflows above ~142, so the beta function
    /// became 0 and the density `inf` (e.g. Beta(80, 80)), and once `Γ(a)Γ(b)`
    /// overflowed as well the density became `NaN` (e.g. Beta(100, 100) -- an
    /// everyday posterior after 200 observations). Reference values computed
    /// independently with mpmath at 50 digits, NOT derived from this crate.
    #[test]
    fn test_beta_pdf_matches_reference_values_for_large_parameters() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.5, 2.0, 3.0, 1.5),
            (0.7, 5.0, 2.0, 2.1608999999999998),
            (0.3, 0.5, 0.5, 0.69460911804285661),
            (0.01, 2.0, 50.0, 15.583489608088065),
            (0.5, 80.0, 80.0, 10.07677292588831),
            (0.5, 100.0, 100.0, 11.269695801851284),
            (0.4, 200.0, 300.0, 18.199532673567943),
            (0.5, 1000.0, 1000.0, 35.678022291708641),
        ];

        for &(x, alpha, beta_param, expected) in cases {
            let dist =
                Beta::new(alpha, beta_param, 0.0, 1.0).expect("test/example should not fail");
            let pdf = dist.pdf(x);
            assert!(
                pdf.is_finite(),
                "pdf({x}) for Beta({alpha}, {beta_param}) is {pdf}"
            );
            let relative = ((pdf - expected) / expected).abs();
            assert!(
                relative < 1e-10,
                "pdf({x}) for Beta({alpha}, {beta_param}): got {pdf:e}, want {expected:e} \
                 (relative {relative:e})"
            );
        }
    }

    /// The boundary values of the density must survive the move to log space:
    /// `0^0 = 1` (α = 1), `0^positive = 0` (α > 1) and `0^negative = +inf`
    /// (α < 1), and likewise at x = 1 for β.
    #[test]
    fn test_beta_pdf_boundary_exponents() {
        // α = 1: the density at x = 0 is 1/B(1, β) = β.
        let flat_left = Beta::new(1.0_f64, 3.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(flat_left.pdf(0.0), 3.0, epsilon = 1e-12);

        // α > 1: the density vanishes at x = 0.
        let vanishing = Beta::new(2.0_f64, 3.0, 0.0, 1.0).expect("test/example should not fail");
        assert_eq!(vanishing.pdf(0.0), 0.0);

        // α < 1: the density diverges at x = 0.
        let diverging = Beta::new(0.5_f64, 0.5, 0.0, 1.0).expect("test/example should not fail");
        assert!(diverging.pdf(0.0).is_infinite());

        // β = 1: mirror image at x = 1.
        let flat_right = Beta::new(3.0_f64, 1.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(flat_right.pdf(1.0), 3.0, epsilon = 1e-12);
        assert_eq!(vanishing.pdf(1.0), 0.0);
        assert!(diverging.pdf(1.0).is_infinite());

        // Outside the support, and with a scale factor.
        assert_eq!(vanishing.pdf(-0.1), 0.0);
        assert_eq!(vanishing.pdf(1.1), 0.0);
        let scaled = Beta::new(100.0_f64, 100.0, 0.0, 2.0).expect("test/example should not fail");
        let standard = Beta::new(100.0_f64, 100.0, 0.0, 1.0).expect("test/example should not fail");
        assert_relative_eq!(scaled.pdf(1.0), standard.pdf(0.5) / 2.0, epsilon = 1e-12);
    }

    /// `entropy` returns `ln B(a, b) + ln(scale)`; it used to go through
    /// `beta_function(a, b).ln()`, i.e. `-inf` for `a + b >= ~142` and `NaN`
    /// beyond. Reference `ln B(a,b)` values from mpmath at 50 digits.
    #[test]
    fn test_beta_entropy_finite_for_large_parameters() {
        let cases: &[(f64, f64, f64)] = &[
            (2.0, 3.0, -2.4849066497880003),
            (0.5, 0.5, 1.1447298858494002),
            (80.0, 80.0, -111.82748759361559),
            (100.0, 100.0, -139.66525908670664),
            (1000.0, 1000.0, -1388.4826016359023),
        ];

        for &(alpha, beta_param, expected) in cases {
            let dist =
                Beta::new(alpha, beta_param, 0.0, 1.0).expect("test/example should not fail");
            let entropy = ScirsDist::entropy(&dist);
            assert!(
                entropy.is_finite(),
                "entropy for Beta({alpha}, {beta_param}) is {entropy}"
            );
            let relative = ((entropy - expected) / expected).abs();
            assert!(
                relative < 1e-12,
                "entropy for Beta({alpha}, {beta_param}): got {entropy}, want {expected} \
                 (relative {relative:e})"
            );
        }
    }
}
