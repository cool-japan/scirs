//! Truncated distribution support for scirs2-stats
//!
//! This module provides truncated versions of common continuous distributions,
//! where samples are constrained to lie within a specified interval [a, b].
//!
//! # Overview
//!
//! A truncated distribution is derived from a base distribution by restricting
//! the support to a bounded interval [a, b]. The PDF is renormalized so it
//! integrates to 1 over [a, b].
//!
//! # Mathematical Background
//!
//! For a base distribution with CDF F(x) and PDF f(x), the truncated
//! distribution on [a, b] has:
//!
//! - PDF: `f_T(x) = f(x) / (F(b) - F(a))`  for `x ∈ [a, b]`
//! - CDF: `F_T(x) = (F(x) - F(a)) / (F(b) - F(a))`  for `x ∈ [a, b]`
//! - Mean: analytically computed for each distribution
//!
//! # Example
//!
//! ```rust
//! use scirs2_stats::distributions::truncated::{TruncatedNormal, TruncatedExponential};
//!
//! // Normal distribution truncated to [0, 3]
//! let tn = TruncatedNormal::new(1.0f64, 0.5, 0.0, 3.0).expect("valid params");
//! let pdf_val = tn.pdf(1.5);
//! assert!(pdf_val > 0.0);
//!
//! // Exponential distribution truncated to [0.5, 5.0]
//! let te = TruncatedExponential::new(1.0f64, 0.5, 5.0).expect("valid params");
//! let cdf_val = te.cdf(2.0);
//! assert!(cdf_val > 0.0 && cdf_val < 1.0);
//! ```

use crate::error::{StatsError, StatsResult};
use crate::traits::{ContinuousDistribution, Distribution};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{rngs::StdRng, Rng, RngExt, SeedableRng};
use std::fmt::Debug;

// ============================================================
// Helper functions for the standard normal distribution
// ============================================================

/// Standard normal PDF φ(x) = (1/√(2π)) exp(-x²/2)
fn standard_normal_pdf<F: Float>(x: F) -> F {
    let two = F::from(2.0).expect("F::from should not fail for 2.0");
    let sqrt_2pi = F::from(2.506_628_274_631_001).expect("F::from should not fail for sqrt(2π)");
    (-x * x / two).exp() / sqrt_2pi
}

/// Standard normal CDF Φ(x) using the error function
fn standard_normal_cdf<F: Float>(x: F) -> F {
    let half = F::from(0.5).expect("F::from should not fail for 0.5");
    let sqrt2 = F::from(std::f64::consts::SQRT_2).expect("F::from should not fail for SQRT_2");
    half * (F::one() + erf(x / sqrt2))
}

/// Error function erf(x) via Horner's method (Abramowitz & Stegun 7.1.26)
fn erf<F: Float>(x: F) -> F {
    let neg = x < F::zero();
    let x = x.abs();

    let t = F::one()
        / (F::one() + F::from(0.3275911).expect("F::from should not fail for 0.3275911") * x);
    let poly = t
        * (F::from(0.254829592).expect("F::from should not fail for 0.254829592")
            + t * (F::from(-0.284496736).expect("F::from should not fail for -0.284496736")
                + t * (F::from(1.421413741).expect("F::from should not fail for 1.421413741")
                    + t * (F::from(-1.453152027)
                        .expect("F::from should not fail for -1.453152027")
                        + t * F::from(1.061405429)
                            .expect("F::from should not fail for 1.061405429")))));
    let result = F::one() - poly * (-(x * x)).exp();

    if neg {
        -result
    } else {
        result
    }
}

/// Quantile (inverse CDF) of the standard normal using Acklam's rational
/// approximation (full-precision, max |error| < 1.15e-9).
fn standard_normal_ppf<F: Float>(p: F) -> F {
    let half = F::from(0.5).unwrap_or(F::zero());
    if p <= F::zero() {
        return F::from(-8.0).unwrap_or(F::neg_infinity());
    }
    if p >= F::one() {
        return F::from(8.0).unwrap_or(F::infinity());
    }
    if p == half {
        return F::zero();
    }

    let p_low = F::from(0.02425).unwrap_or(F::zero());
    let p_high = F::one() - p_low;

    if p < p_low {
        // Lower tail: rational approximation for small p
        let q = (-F::from(2.0).unwrap_or(F::zero()) * p.ln()).sqrt();
        let c = [
            F::from(-7.784894002430293e-03).unwrap_or(F::zero()),
            F::from(-3.223964580411365e-01).unwrap_or(F::zero()),
            F::from(-2.400758277161838e+00).unwrap_or(F::zero()),
            F::from(-2.549732539343734e+00).unwrap_or(F::zero()),
            F::from(4.374664141464968e+00).unwrap_or(F::zero()),
            F::from(2.938163982698783e+00).unwrap_or(F::zero()),
        ];
        let d = [
            F::from(7.784695709041462e-03).unwrap_or(F::zero()),
            F::from(3.224671290700398e-01).unwrap_or(F::zero()),
            F::from(2.445134137142996e+00).unwrap_or(F::zero()),
            F::from(3.754408661907416e+00).unwrap_or(F::zero()),
        ];
        let num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5];
        let den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + F::one();
        num / den
    } else if p <= p_high {
        // Central region
        let q = p - half;
        let r = q * q;
        let a = [
            F::from(-3.969683028665376e+01).unwrap_or(F::zero()),
            F::from(2.209460984245205e+02).unwrap_or(F::zero()),
            F::from(-2.759285104469687e+02).unwrap_or(F::zero()),
            F::from(1.383577518672690e+02).unwrap_or(F::zero()),
            F::from(-3.066479806614716e+01).unwrap_or(F::zero()),
            F::from(2.506628277459239e+00).unwrap_or(F::zero()),
        ];
        let b = [
            F::from(-5.447609879822406e+01).unwrap_or(F::zero()),
            F::from(1.615858368580409e+02).unwrap_or(F::zero()),
            F::from(-1.556989798598866e+02).unwrap_or(F::zero()),
            F::from(6.680131188771972e+01).unwrap_or(F::zero()),
            F::from(-1.328068155288572e+01).unwrap_or(F::zero()),
        ];
        let num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q;
        let den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + F::one();
        num / den
    } else {
        // Upper tail: use symmetry
        -standard_normal_ppf(F::one() - p)
    }
}

// ============================================================
// TruncatedNormal
// ============================================================

/// Truncated Normal distribution
///
/// A normal distribution with support restricted to [lower, upper].
///
/// # Fields
///
/// - `mu`: Mean of the underlying normal distribution
/// - `sigma`: Standard deviation of the underlying normal (> 0)
/// - `lower`: Lower truncation bound
/// - `upper`: Upper truncation bound (upper > lower)
///
/// # Example
///
/// ```rust
/// use scirs2_stats::distributions::truncated::TruncatedNormal;
///
/// let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).expect("valid params");
/// let pdf = tn.pdf(0.0);
/// // PDF at mean should be higher than standard normal due to renormalization
/// assert!(pdf > 0.3989); // std normal PDF at 0
/// ```
#[derive(Debug, Clone)]
pub struct TruncatedNormal<F: Float> {
    /// Mean of the base normal distribution
    pub mu: F,
    /// Standard deviation of the base normal distribution
    pub sigma: F,
    /// Lower truncation bound
    pub lower: F,
    /// Upper truncation bound
    pub upper: F,
    /// Standardized lower bound: (lower - mu) / sigma
    alpha: F,
    /// Standardized upper bound: (upper - mu) / sigma
    beta: F,
    /// Normalization constant: Φ(β) - Φ(α)
    z: F,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> TruncatedNormal<F> {
    /// Create a new truncated normal distribution.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean of the underlying normal
    /// * `sigma` - Standard deviation (> 0)
    /// * `lower` - Lower truncation bound
    /// * `upper` - Upper truncation bound (must be > lower)
    ///
    /// # Errors
    ///
    /// Returns error if sigma <= 0 or lower >= upper.
    pub fn new(mu: F, sigma: F, lower: F, upper: F) -> StatsResult<Self> {
        if sigma <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "sigma must be positive".to_string(),
            ));
        }
        if lower >= upper {
            return Err(StatsError::InvalidArgument(
                "lower bound must be strictly less than upper bound".to_string(),
            ));
        }

        let alpha = (lower - mu) / sigma;
        let beta = (upper - mu) / sigma;
        let phi_alpha = standard_normal_cdf(alpha);
        let phi_beta = standard_normal_cdf(beta);
        let z = phi_beta - phi_alpha;

        if z <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "Normalization constant (Φ(β) - Φ(α)) is zero; \
                 truncation interval likely too narrow relative to sigma"
                    .to_string(),
            ));
        }

        Ok(Self {
            mu,
            sigma,
            lower,
            upper,
            alpha,
            beta,
            z,
        })
    }

    /// PDF of the truncated normal at x.
    pub fn pdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::zero();
        }
        let xi = (x - self.mu) / self.sigma;
        standard_normal_pdf(xi) / (self.sigma * self.z)
    }

    /// Log-PDF of the truncated normal at x.
    pub fn logpdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::neg_infinity();
        }
        self.pdf(x).ln()
    }

    /// CDF of the truncated normal at x.
    pub fn cdf(&self, x: F) -> F {
        if x <= self.lower {
            return F::zero();
        }
        if x >= self.upper {
            return F::one();
        }
        let xi = (x - self.mu) / self.sigma;
        (standard_normal_cdf(xi) - standard_normal_cdf(self.alpha)) / self.z
    }

    /// Inverse CDF (quantile function) of the truncated normal.
    ///
    /// # Errors
    ///
    /// Returns error if p is outside [0, 1].
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::InvalidArgument(
                "probability p must be in [0, 1]".to_string(),
            ));
        }
        if p == F::zero() {
            return Ok(self.lower);
        }
        if p == F::one() {
            return Ok(self.upper);
        }
        // Inverse: x = μ + σ * Φ⁻¹(Φ(α) + p * Z)
        let phi_alpha = standard_normal_cdf(self.alpha);
        let target = phi_alpha + p * self.z;
        Ok(self.mu + self.sigma * standard_normal_ppf(target))
    }

    /// Generate random samples using the inverse-CDF method.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to draw
    /// * `seed` - Optional random seed
    pub fn rvs(&self, size: usize, seed: Option<u64>) -> StatsResult<Array1<F>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42);
                StdRng::seed_from_u64(s)
            }
        };

        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = rng.random();
            let p = F::from(u).expect("F::from should not fail for f64 uniform sample in [0,1)");
            let x = self.ppf(p)?;
            samples.push(x);
        }
        Ok(Array1::from_vec(samples))
    }

    /// Mean of the truncated normal distribution.
    ///
    /// Formula: μ + σ * (φ(α) - φ(β)) / Z
    /// where φ is the standard normal PDF, Z = Φ(β) - Φ(α).
    pub fn mean(&self) -> F {
        let phi_alpha = standard_normal_pdf(self.alpha);
        let phi_beta = standard_normal_pdf(self.beta);
        self.mu + self.sigma * (phi_alpha - phi_beta) / self.z
    }

    /// Variance of the truncated normal distribution.
    ///
    /// Formula: σ² * (1 + (α·φ(α) - β·φ(β))/Z - ((φ(α) - φ(β))/Z)²)
    pub fn var(&self) -> F {
        let phi_alpha = standard_normal_pdf(self.alpha);
        let phi_beta = standard_normal_pdf(self.beta);
        let ratio = (phi_alpha - phi_beta) / self.z;
        let correction = (self.alpha * phi_alpha - self.beta * phi_beta) / self.z;
        self.sigma * self.sigma * (F::one() + correction - ratio * ratio)
    }
}

// ============================================================
// TruncatedExponential
// ============================================================

/// Truncated Exponential distribution
///
/// An exponential distribution with support restricted to [lower, upper].
///
/// # Example
///
/// ```
/// use scirs2_stats::distributions::truncated::TruncatedExponential;
///
/// // Exponential with rate=1, restricted to [0, 3]
/// let te = TruncatedExponential::new(1.0f64, 0.0, 3.0).expect("valid params");
/// let mean = te.mean();
/// // Mean should be positive and less than the upper bound
/// assert!(mean > 0.0 && mean < 3.0);
/// ```
#[derive(Debug, Clone)]
pub struct TruncatedExponential<F: Float> {
    /// Rate parameter λ > 0
    pub rate: F,
    /// Lower truncation bound (>= 0 for exponential)
    pub lower: F,
    /// Upper truncation bound
    pub upper: F,
    /// Normalization: F(upper) - F(lower)
    z: F,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> TruncatedExponential<F> {
    /// Create a new truncated exponential distribution.
    ///
    /// # Arguments
    ///
    /// * `rate` - Rate parameter λ > 0
    /// * `lower` - Lower bound (>= 0)
    /// * `upper` - Upper bound (> lower)
    ///
    /// # Errors
    ///
    /// Returns error if rate <= 0, lower < 0, or lower >= upper.
    pub fn new(rate: F, lower: F, upper: F) -> StatsResult<Self> {
        if rate <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "rate must be positive".to_string(),
            ));
        }
        if lower < F::zero() {
            return Err(StatsError::InvalidArgument(
                "lower bound must be >= 0 for exponential distribution".to_string(),
            ));
        }
        if lower >= upper {
            return Err(StatsError::InvalidArgument(
                "lower bound must be strictly less than upper bound".to_string(),
            ));
        }

        // CDF of exponential: 1 - exp(-λx)
        let cdf_lower = F::one() - (-rate * lower).exp();
        let cdf_upper = F::one() - (-rate * upper).exp();
        let z = cdf_upper - cdf_lower;

        if z <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "Normalization constant is zero".to_string(),
            ));
        }

        Ok(Self {
            rate,
            lower,
            upper,
            z,
        })
    }

    /// PDF of the truncated exponential at x.
    pub fn pdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::zero();
        }
        self.rate * (-self.rate * x).exp() / self.z
    }

    /// Log-PDF of the truncated exponential at x.
    pub fn logpdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::neg_infinity();
        }
        self.rate.ln() - self.rate * x - self.z.ln()
    }

    /// CDF of the truncated exponential at x.
    pub fn cdf(&self, x: F) -> F {
        if x <= self.lower {
            return F::zero();
        }
        if x >= self.upper {
            return F::one();
        }
        let cdf_lower = F::one() - (-self.rate * self.lower).exp();
        ((F::one() - (-self.rate * x).exp()) - cdf_lower) / self.z
    }

    /// Inverse CDF (quantile function).
    ///
    /// # Errors
    ///
    /// Returns error if p is outside [0, 1].
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::InvalidArgument(
                "probability p must be in [0, 1]".to_string(),
            ));
        }
        if p == F::zero() {
            return Ok(self.lower);
        }
        if p == F::one() {
            return Ok(self.upper);
        }
        // x = -ln(1 - (CDF(lower) + p*Z)) / rate
        let cdf_lower = F::one() - (-self.rate * self.lower).exp();
        let target_cdf = cdf_lower + p * self.z;
        Ok(-(F::one() - target_cdf).ln() / self.rate)
    }

    /// Generate random samples using the inverse-CDF method.
    pub fn rvs(&self, size: usize, seed: Option<u64>) -> StatsResult<Array1<F>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42);
                StdRng::seed_from_u64(s)
            }
        };

        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = rng.random();
            let p = F::from(u).expect("F::from should not fail for f64 uniform sample in [0,1)");
            let x = self.ppf(p)?;
            samples.push(x);
        }
        Ok(Array1::from_vec(samples))
    }

    /// Mean of the truncated exponential.
    ///
    /// Formula: (1/λ) + (lower·exp(-λ·lower) - upper·exp(-λ·upper)) / Z  ─ simplified
    pub fn mean(&self) -> F {
        // E[X | a ≤ X ≤ b] = (1/λ) + (a·e^{-λa} - b·e^{-λb}) / (e^{-λa} - e^{-λb})
        //                        or equivalently using the normalization Z
        let ea = (-self.rate * self.lower).exp();
        let eb = (-self.rate * self.upper).exp();
        let denom = ea - eb; // = Z / rate  (since Z = (1-eb) - (1-ea) = ea-eb)
        if denom.abs() < F::from(1e-15).expect("F::from should not fail for 1e-15") {
            // When interval is very short, mean ≈ midpoint
            (self.lower + self.upper) / F::from(2.0).expect("F::from should not fail for 2.0")
        } else {
            F::one() / self.rate + (self.lower * ea - self.upper * eb) / denom
        }
    }

    /// Variance of the truncated exponential.
    pub fn var(&self) -> F {
        let mu = self.mean();
        // Use E[X²] - (E[X])²; E[X²] via integration by parts
        let ea = (-self.rate * self.lower).exp();
        let eb = (-self.rate * self.upper).exp();
        let denom = ea - eb;
        if denom.abs() < F::from(1e-15).expect("F::from should not fail for 1e-15") {
            let half_width =
                (self.upper - self.lower) / F::from(2.0).expect("F::from should not fail for 2.0");
            return half_width * half_width
                / F::from(3.0).expect("F::from should not fail for 3.0");
        }
        let two_over_lam2 =
            F::from(2.0).expect("F::from should not fail for 2.0") / (self.rate * self.rate);
        let ex2 = two_over_lam2
            + (self.lower * self.lower * ea - self.upper * self.upper * eb) / denom
            + (F::from(2.0).expect("F::from should not fail for 2.0") / self.rate)
                * (self.lower * ea - self.upper * eb)
                / denom;
        let var_raw = ex2 - mu * mu;
        // Clamp to avoid floating-point negatives
        if var_raw < F::zero() {
            F::zero()
        } else {
            var_raw
        }
    }
}

// ============================================================
// TruncatedGamma
// ============================================================

/// Truncated Gamma distribution
///
/// A gamma distribution with support restricted to [lower, upper].
///
/// The Gamma(α, β) distribution has shape parameter α > 0 and rate β > 0
/// (equivalently, scale θ = 1/β).
///
/// # Example
///
/// ```rust
/// use scirs2_stats::distributions::truncated::TruncatedGamma;
///
/// // Gamma(shape=2, rate=1) restricted to [0, 5]
/// let tg = TruncatedGamma::new(2.0f64, 1.0, 0.0, 5.0).expect("valid params");
/// let pdf_at_2 = tg.pdf(2.0);
/// assert!(pdf_at_2 > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct TruncatedGamma<F: Float> {
    /// Shape parameter α > 0
    pub shape: F,
    /// Rate parameter β > 0
    pub rate: F,
    /// Lower truncation bound (>= 0 for gamma)
    pub lower: F,
    /// Upper truncation bound
    pub upper: F,
    /// Normalization constant: Γ_reg(α, β·lower) - ... Actually CDF(upper) - CDF(lower)
    z: F,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> TruncatedGamma<F> {
    /// Create a new truncated gamma distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter α > 0
    /// * `rate` - Rate parameter β > 0 (scale = 1/rate)
    /// * `lower` - Lower truncation bound (>= 0)
    /// * `upper` - Upper truncation bound
    ///
    /// # Errors
    ///
    /// Returns error if shape <= 0, rate <= 0, lower < 0, or lower >= upper.
    pub fn new(shape: F, rate: F, lower: F, upper: F) -> StatsResult<Self> {
        if shape <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "shape must be positive".to_string(),
            ));
        }
        if rate <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "rate must be positive".to_string(),
            ));
        }
        if lower < F::zero() {
            return Err(StatsError::InvalidArgument(
                "lower bound must be >= 0 for gamma distribution".to_string(),
            ));
        }
        if lower >= upper {
            return Err(StatsError::InvalidArgument(
                "lower bound must be strictly less than upper bound".to_string(),
            ));
        }

        let cdf_lower = gamma_cdf(shape, rate, lower);
        let cdf_upper = gamma_cdf(shape, rate, upper);
        let z = cdf_upper - cdf_lower;

        if z <= F::from(1e-15).expect("F::from should not fail for 1e-15") {
            return Err(StatsError::InvalidArgument(
                "Normalization constant is effectively zero; \
                 truncation interval may be too narrow or outside main support"
                    .to_string(),
            ));
        }

        Ok(Self {
            shape,
            rate,
            lower,
            upper,
            z,
        })
    }

    /// PDF of the truncated gamma at x.
    pub fn pdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::zero();
        }
        gamma_pdf(self.shape, self.rate, x) / self.z
    }

    /// Log-PDF of the truncated gamma at x.
    pub fn logpdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::neg_infinity();
        }
        gamma_logpdf(self.shape, self.rate, x) - self.z.ln()
    }

    /// CDF of the truncated gamma at x.
    pub fn cdf(&self, x: F) -> F {
        if x <= self.lower {
            return F::zero();
        }
        if x >= self.upper {
            return F::one();
        }
        let cdf_lower = gamma_cdf(self.shape, self.rate, self.lower);
        (gamma_cdf(self.shape, self.rate, x) - cdf_lower) / self.z
    }

    /// Inverse CDF via bisection.
    ///
    /// # Errors
    ///
    /// Returns error if p is outside [0, 1].
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::InvalidArgument(
                "probability p must be in [0, 1]".to_string(),
            ));
        }
        if p == F::zero() {
            return Ok(self.lower);
        }
        if p == F::one() {
            return Ok(self.upper);
        }
        // Bisection on the CDF
        let mut lo = self.lower;
        let mut hi = self.upper;
        let eps = F::from(1e-10).expect("F::from should not fail for 1e-10");
        for _ in 0..100 {
            let mid = (lo + hi) / F::from(2.0).expect("F::from should not fail for 2.0");
            if self.cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < eps {
                break;
            }
        }
        Ok((lo + hi) / F::from(2.0).expect("F::from should not fail for 2.0"))
    }

    /// Generate random samples using the inverse-CDF method.
    pub fn rvs(&self, size: usize, seed: Option<u64>) -> StatsResult<Array1<F>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42);
                StdRng::seed_from_u64(s)
            }
        };

        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = rng.random();
            let p = F::from(u).expect("F::from should not fail for f64 uniform sample in [0,1)");
            let x = self.ppf(p)?;
            samples.push(x);
        }
        Ok(Array1::from_vec(samples))
    }

    /// Mean of the truncated gamma (numerical via quadrature).
    pub fn mean(&self) -> F {
        // Numerical integration via midpoint rule (100 points)
        let n = 200usize;
        let h =
            (self.upper - self.lower) / F::from(n).expect("F::from should not fail for usize n");
        let mut sum = F::zero();
        for i in 0..n {
            let x = self.lower
                + h * F::from(i).expect("F::from should not fail for usize i")
                + h / F::from(2.0).expect("F::from should not fail for 2.0");
            sum = sum + x * self.pdf(x);
        }
        sum * h
    }

    /// Variance of the truncated gamma (numerical via quadrature).
    pub fn var(&self) -> F {
        let mu = self.mean();
        let n = 200usize;
        let h =
            (self.upper - self.lower) / F::from(n).expect("F::from should not fail for usize n");
        let mut sum = F::zero();
        for i in 0..n {
            let x = self.lower
                + h * F::from(i).expect("F::from should not fail for usize i")
                + h / F::from(2.0).expect("F::from should not fail for 2.0");
            let diff = x - mu;
            sum = sum + diff * diff * self.pdf(x);
        }
        let v = sum * h;
        if v < F::zero() {
            F::zero()
        } else {
            v
        }
    }
}

// ============================================================
// Gamma distribution helpers
// ============================================================

/// Gamma PDF: f(x) = β^α / Γ(α) * x^(α-1) * exp(-β*x)
fn gamma_pdf<F: Float>(shape: F, rate: F, x: F) -> F {
    if x <= F::zero() {
        return F::zero();
    }
    gamma_logpdf(shape, rate, x).exp()
}

fn gamma_logpdf<F: Float>(shape: F, rate: F, x: F) -> F {
    if x <= F::zero() {
        return F::neg_infinity();
    }
    shape * rate.ln() - log_gamma(shape) + (shape - F::one()) * x.ln() - rate * x
}

/// Regularized incomplete gamma function P(a, x) via series expansion
/// (for the CDF: P(shape, rate*x))
fn gamma_cdf<F: Float>(shape: F, rate: F, x: F) -> F {
    if x <= F::zero() {
        return F::zero();
    }
    let x_scaled: F = rate * x;
    regularized_inc_gamma(shape, x_scaled)
}

/// Regularized incomplete gamma P(a, x) via series for x < a+1 and CF for x >= a+1
fn regularized_inc_gamma<F: Float>(a: F, x: F) -> F {
    if x < F::zero() {
        return F::zero();
    }
    if x == F::zero() {
        return F::zero();
    }
    if x < a + F::one() {
        // Series expansion
        inc_gamma_series(a, x)
    } else {
        // Continued fraction (complement)
        F::one() - inc_gamma_cf(a, x)
    }
}

/// Series expansion for the lower incomplete gamma function
fn inc_gamma_series<F: Float>(a: F, x: F) -> F {
    let max_iter = 200;
    let eps = F::from(3e-7).expect("F::from should not fail for 3e-7");

    let log_gam_a = log_gamma(a);
    let lnx = x.ln();

    let mut ap = a;
    let mut sum = F::one() / a;
    let mut del = sum;

    for _ in 0..max_iter {
        ap = ap + F::one();
        del = del * x / ap;
        sum = sum + del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }

    let result = sum * (-(x) + a * lnx - log_gam_a).exp();
    if result > F::one() {
        F::one()
    } else {
        result
    }
}

/// Continued fraction for the upper incomplete gamma function
fn inc_gamma_cf<F: Float>(a: F, x: F) -> F {
    let max_iter = 200;
    let eps = F::from(3e-7).expect("F::from should not fail for 3e-7");
    let fpmin = F::from(1e-300).expect("F::from should not fail for 1e-300");

    let log_gam_a = log_gamma(a);
    let lnx = x.ln();

    let mut b = x + F::one() - a;
    let mut c = F::one() / fpmin;
    let mut d = F::one() / b;
    let mut h = d;

    for i in 1..=max_iter {
        let fi = F::from(i).expect("F::from should not fail for loop index i");
        let an = -fi * (fi - a);
        b = b + F::from(2.0).expect("F::from should not fail for 2.0");
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = F::one() / d;
        let del = d * c;
        h = h * del;
        if (del - F::one()).abs() < eps {
            break;
        }
    }

    let result = h * (-(x) + a * lnx - log_gam_a).exp();
    if result > F::one() {
        F::one()
    } else {
        result
    }
}

/// Natural logarithm of the Gamma function via Stirling's approximation
/// (Lanczos approximation, g=7, n=9)
fn log_gamma<F: Float>(x: F) -> F {
    // Lanczos coefficients for g=7
    let c = [
        0.99999999999980993_f64,
        676.5203681218851_f64,
        -1259.1392167224028_f64,
        771.323_428_777_653_1_f64,
        -176.615_029_162_140_6_f64,
        12.507_343_278_686_905_f64,
        -0.138_571_095_265_720_12_f64,
        9.984_369_578_019_572e-6_f64,
        1.505_632_735_149_311_6e-7_f64,
    ];

    let x_f64: f64 = NumCast::from(x).unwrap_or(1.0);

    if x_f64 < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        let result = std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x_f64).sin().ln()
            - log_gamma_f64(1.0 - x_f64);
        return F::from(result).expect("F::from should not fail for log_gamma result");
    }

    F::from(log_gamma_f64(x_f64)).expect("F::from should not fail for log_gamma_f64 result")
}

fn log_gamma_f64(x: f64) -> f64 {
    let c = [
        0.99999999999980993_f64,
        676.5203681218851_f64,
        -1259.1392167224028_f64,
        771.3234287776531_f64,
        -176.6150291621406_f64,
        12.507343278686905_f64,
        -0.13857109526572012_f64,
        9.984369578019572e-6_f64,
        1.5056327351493116e-7_f64,
    ];
    if x < 0.5 {
        let v =
            std::f64::consts::PI - (std::f64::consts::PI * x).sin().ln() - log_gamma_f64(1.0 - x);
        return v;
    }
    let x = x - 1.0;
    let mut s = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        s += ci / (x + (i + 1) as f64);
    }
    let t = x + 7.5; // g + 0.5
    (2.0 * std::f64::consts::PI).sqrt().ln() + t.ln() * (x + 0.5) - t + s.ln()
}

// ============================================================
// TruncatedBeta
// ============================================================

/// Truncated Beta distribution
///
/// A Beta distribution with support restricted to [lower, upper] ⊆ [0, 1].
///
/// # Example
///
/// ```rust
/// use scirs2_stats::distributions::truncated::TruncatedBeta;
///
/// // Beta(2, 5) restricted to [0.1, 0.9]
/// let tb = TruncatedBeta::new(2.0f64, 5.0, 0.1, 0.9).expect("valid params");
/// let pdf_val = tb.pdf(0.3);
/// assert!(pdf_val > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct TruncatedBeta<F: Float> {
    /// First shape parameter α > 0
    pub alpha: F,
    /// Second shape parameter β > 0
    pub beta_param: F,
    /// Lower truncation bound (in [0, 1])
    pub lower: F,
    /// Upper truncation bound (in [0, 1])
    pub upper: F,
    /// Normalization constant
    z: F,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> TruncatedBeta<F> {
    /// Create a new truncated Beta distribution.
    ///
    /// # Arguments
    ///
    /// * `alpha` - First shape parameter α > 0
    /// * `beta_param` - Second shape parameter β > 0
    /// * `lower` - Lower bound (in [0, 1])
    /// * `upper` - Upper bound (in (lower, 1])
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid.
    pub fn new(alpha: F, beta_param: F, lower: F, upper: F) -> StatsResult<Self> {
        if alpha <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "alpha must be positive".to_string(),
            ));
        }
        if beta_param <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "beta must be positive".to_string(),
            ));
        }
        if lower < F::zero() || lower >= F::one() {
            return Err(StatsError::InvalidArgument(
                "lower must be in [0, 1)".to_string(),
            ));
        }
        if upper <= lower || upper > F::one() {
            return Err(StatsError::InvalidArgument(
                "upper must be in (lower, 1]".to_string(),
            ));
        }

        let cdf_lower = beta_cdf(alpha, beta_param, lower);
        let cdf_upper = beta_cdf(alpha, beta_param, upper);
        let z = cdf_upper - cdf_lower;

        if z <= F::from(1e-15).expect("F::from should not fail for 1e-15") {
            return Err(StatsError::InvalidArgument(
                "Normalization constant is effectively zero".to_string(),
            ));
        }

        Ok(Self {
            alpha,
            beta_param,
            lower,
            upper,
            z,
        })
    }

    /// PDF of the truncated beta at x.
    pub fn pdf(&self, x: F) -> F {
        if x < self.lower || x > self.upper {
            return F::zero();
        }
        beta_pdf(self.alpha, self.beta_param, x) / self.z
    }

    /// CDF of the truncated beta at x.
    pub fn cdf(&self, x: F) -> F {
        if x <= self.lower {
            return F::zero();
        }
        if x >= self.upper {
            return F::one();
        }
        let cdf_lower = beta_cdf(self.alpha, self.beta_param, self.lower);
        (beta_cdf(self.alpha, self.beta_param, x) - cdf_lower) / self.z
    }

    /// Inverse CDF via bisection.
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::InvalidArgument(
                "probability p must be in [0, 1]".to_string(),
            ));
        }
        if p == F::zero() {
            return Ok(self.lower);
        }
        if p == F::one() {
            return Ok(self.upper);
        }
        let mut lo = self.lower;
        let mut hi = self.upper;
        let eps = F::from(1e-10).expect("F::from should not fail for 1e-10");
        for _ in 0..100 {
            let mid = (lo + hi) / F::from(2.0).expect("F::from should not fail for 2.0");
            if self.cdf(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < eps {
                break;
            }
        }
        Ok((lo + hi) / F::from(2.0).expect("F::from should not fail for 2.0"))
    }

    /// Generate random samples.
    pub fn rvs(&self, size: usize, seed: Option<u64>) -> StatsResult<Array1<F>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42);
                StdRng::seed_from_u64(s)
            }
        };

        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = rng.random();
            let p = F::from(u).expect("F::from should not fail for f64 uniform sample in [0,1)");
            let x = self.ppf(p)?;
            samples.push(x);
        }
        Ok(Array1::from_vec(samples))
    }

    /// Mean of the truncated Beta (numerical).
    pub fn mean(&self) -> F {
        let n = 200usize;
        let h =
            (self.upper - self.lower) / F::from(n).expect("F::from should not fail for usize n");
        let mut sum = F::zero();
        for i in 0..n {
            let x = self.lower
                + h * F::from(i).expect("F::from should not fail for usize i")
                + h / F::from(2.0).expect("F::from should not fail for 2.0");
            sum = sum + x * self.pdf(x);
        }
        sum * h
    }

    /// Variance of the truncated Beta (numerical).
    pub fn var(&self) -> F {
        let mu = self.mean();
        let n = 200usize;
        let h =
            (self.upper - self.lower) / F::from(n).expect("F::from should not fail for usize n");
        let mut sum = F::zero();
        for i in 0..n {
            let x = self.lower
                + h * F::from(i).expect("F::from should not fail for usize i")
                + h / F::from(2.0).expect("F::from should not fail for 2.0");
            let diff = x - mu;
            sum = sum + diff * diff * self.pdf(x);
        }
        let v = sum * h;
        if v < F::zero() {
            F::zero()
        } else {
            v
        }
    }
}

// ============================================================
// Beta distribution helpers
// ============================================================

fn beta_pdf<F: Float>(a: F, b: F, x: F) -> F {
    if x <= F::zero() || x >= F::one() {
        return F::zero();
    }
    let log_beta_ab = log_beta(a, b);
    ((a - F::one()) * x.ln() + (b - F::one()) * (F::one() - x).ln() - log_beta_ab).exp()
}

fn beta_cdf<F: Float>(a: F, b: F, x: F) -> F {
    if x <= F::zero() {
        return F::zero();
    }
    if x >= F::one() {
        return F::one();
    }
    regularized_inc_beta(a, b, x)
}

fn log_beta<F: Float>(a: F, b: F) -> F {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction
fn regularized_inc_beta<F: Float>(a: F, b: F, x: F) -> F {
    if x < (a + F::one()) / (a + b + F::from(2.0).expect("F::from should not fail for 2.0")) {
        inc_beta_cf(a, b, x) * beta_prefactor(a, b, x)
    } else {
        F::one() - inc_beta_cf(b, a, F::one() - x) * beta_prefactor(b, a, F::one() - x)
    }
}

fn beta_prefactor<F: Float>(a: F, b: F, x: F) -> F {
    (a * x.ln() + b * (F::one() - x).ln() - log_beta(a, b)).exp() / a
}

fn inc_beta_cf<F: Float>(a: F, b: F, x: F) -> F {
    let max_iter = 200;
    let eps = F::from(3e-7).expect("F::from should not fail for 3e-7");
    let fpmin = F::from(1e-300).expect("F::from should not fail for 1e-300");

    let qab = a + b;
    let qap = a + F::one();
    let qam = a - F::one();

    let mut c = F::one();
    let mut d = F::one() - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = F::one() / d;
    let mut h = d;

    for m in 1..=max_iter {
        let mf = F::from(m).expect("F::from should not fail for loop index m");
        let two_mf = F::from(2 * m).expect("F::from should not fail for 2*m");

        // Even step
        let aa = mf * (b - mf) * x / ((qam + two_mf) * (a + two_mf));
        d = F::one() + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = F::one() + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = F::one() / d;
        h = h * d * c;

        // Odd step
        let aa = -(a + mf) * (qab + mf) * x / ((a + two_mf) * (qap + two_mf));
        d = F::one() + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = F::one() + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = F::one() / d;
        let del = d * c;
        h = h * del;
        if (del - F::one()).abs() < eps {
            break;
        }
    }
    h
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- TruncatedNormal tests ---

    #[test]
    fn test_truncated_normal_pdf_zero_outside() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        assert_eq!(tn.pdf(-3.0), 0.0);
        assert_eq!(tn.pdf(3.0), 0.0);
    }

    #[test]
    fn test_truncated_normal_pdf_positive_inside() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        assert!(tn.pdf(0.0) > 0.0);
        assert!(tn.pdf(-1.0) > 0.0);
        assert!(tn.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_truncated_normal_pdf_integrates_to_one() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        // Numerical integration with 1000 points
        let n = 1000;
        let h = 4.0 / n as f64;
        let mut sum = 0.0f64;
        for i in 0..n {
            let x = -2.0 + h * i as f64 + h / 2.0;
            sum += tn.pdf(x);
        }
        sum *= h;
        assert!((sum - 1.0).abs() < 1e-5, "Integral = {}", sum);
    }

    #[test]
    fn test_truncated_normal_cdf_bounds() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        assert_eq!(tn.cdf(-3.0), 0.0);
        assert_eq!(tn.cdf(-2.0), 0.0);
        assert_eq!(tn.cdf(2.0), 1.0);
        assert_eq!(tn.cdf(3.0), 1.0);
        let mid = tn.cdf(0.0);
        assert!(mid > 0.0 && mid < 1.0, "CDF at mean = {}", mid);
    }

    #[test]
    fn test_truncated_normal_cdf_monotone() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        let xs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let mut prev = 0.0f64;
        for &x in &xs {
            let c = tn.cdf(x);
            assert!(c > prev, "CDF not monotone at x={}: {} <= {}", x, c, prev);
            prev = c;
        }
    }

    #[test]
    fn test_truncated_normal_mean() {
        // Half-normal: N(0,1) truncated to [0, ∞) → mean = √(2/π) ≈ 0.7979
        let tn = TruncatedNormal::new(0.0f64, 1.0, 0.0, 10.0).unwrap();
        let mean = tn.mean();
        assert!(
            (mean - 0.7979).abs() < 0.01,
            "Half-normal mean = {}, expected ≈ 0.7979",
            mean
        );
    }

    #[test]
    fn test_truncated_normal_ppf_roundtrip() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = tn.ppf(p).unwrap();
            let cdf_x = tn.cdf(x);
            assert!(
                (cdf_x - p).abs() < 1e-5,
                "PPF roundtrip failed: ppf({}) = {}, cdf = {}",
                p,
                x,
                cdf_x
            );
        }
    }

    #[test]
    fn test_truncated_normal_rvs() {
        let tn = TruncatedNormal::new(0.0f64, 1.0, -2.0, 2.0).unwrap();
        let samples = tn.rvs(500, Some(42)).unwrap();
        assert_eq!(samples.len(), 500);
        for &s in samples.iter() {
            assert!(s >= -2.0 && s <= 2.0, "Sample {} outside bounds [-2, 2]", s);
        }
    }

    #[test]
    fn test_truncated_normal_invalid_params() {
        assert!(TruncatedNormal::new(0.0f64, -1.0, -2.0, 2.0).is_err());
        assert!(TruncatedNormal::new(0.0f64, 1.0, 2.0, -2.0).is_err());
        assert!(TruncatedNormal::new(0.0f64, 1.0, 2.0, 2.0).is_err());
    }

    // --- TruncatedExponential tests ---

    #[test]
    fn test_truncated_exponential_pdf_bounds() {
        let te = TruncatedExponential::new(1.0f64, 0.0, 3.0).unwrap();
        assert_eq!(te.pdf(-1.0), 0.0);
        assert_eq!(te.pdf(4.0), 0.0);
        assert!(te.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_truncated_exponential_cdf_bounds() {
        let te = TruncatedExponential::new(1.0f64, 0.0, 3.0).unwrap();
        assert_eq!(te.cdf(-1.0), 0.0);
        assert_eq!(te.cdf(0.0), 0.0);
        assert_eq!(te.cdf(3.0), 1.0);
    }

    #[test]
    fn test_truncated_exponential_pdf_integrates_to_one() {
        let te = TruncatedExponential::new(2.0f64, 0.5, 3.0).unwrap();
        let n = 1000;
        let h = 2.5 / n as f64;
        let mut sum = 0.0f64;
        for i in 0..n {
            let x = 0.5 + h * i as f64 + h / 2.0;
            sum += te.pdf(x);
        }
        sum *= h;
        assert!((sum - 1.0).abs() < 1e-5, "Integral = {}", sum);
    }

    #[test]
    fn test_truncated_exponential_mean() {
        // Exp(rate=1) truncated to [0, 3]
        // E[X | 0 <= X <= 3] = 1/lambda + (a*e^{-a} - b*e^{-b})/(e^{-a} - e^{-b})
        //   = 1 + (0 - 3*e^{-3})/(1 - e^{-3}) = 1 - 3*e^{-3}/(1 - e^{-3}) ≈ 0.8428
        let te = TruncatedExponential::new(1.0f64, 0.0, 3.0).expect("valid truncated exponential");
        let mean = te.mean();
        assert!(
            (mean - 0.8428).abs() < 0.005,
            "Mean = {}, expected ≈ 0.8428",
            mean
        );
    }

    #[test]
    fn test_truncated_exponential_ppf_roundtrip() {
        let te = TruncatedExponential::new(1.5f64, 0.2, 4.0).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let x = te.ppf(p).unwrap();
            let cdf_x = te.cdf(x);
            assert!(
                (cdf_x - p).abs() < 1e-8,
                "PPF roundtrip: ppf({}) = {}, cdf = {}",
                p,
                x,
                cdf_x
            );
        }
    }

    #[test]
    fn test_truncated_exponential_rvs() {
        let te = TruncatedExponential::new(1.0f64, 0.5, 5.0).unwrap();
        let samples = te.rvs(200, Some(123)).unwrap();
        assert_eq!(samples.len(), 200);
        for &s in samples.iter() {
            assert!(s >= 0.5 && s <= 5.0, "Sample {} outside [0.5, 5]", s);
        }
    }

    // --- TruncatedGamma tests ---

    #[test]
    fn test_truncated_gamma_pdf_bounds() {
        let tg = TruncatedGamma::new(2.0f64, 1.0, 0.0, 5.0).unwrap();
        assert_eq!(tg.pdf(-1.0), 0.0);
        assert_eq!(tg.pdf(6.0), 0.0);
        assert!(tg.pdf(2.0) > 0.0);
    }

    #[test]
    fn test_truncated_gamma_pdf_integrates_to_one() {
        let tg = TruncatedGamma::new(2.0f64, 1.0, 0.5, 5.0).unwrap();
        let n = 1000;
        let h = 4.5 / n as f64;
        let mut sum = 0.0f64;
        for i in 0..n {
            let x = 0.5 + h * i as f64 + h / 2.0;
            sum += tg.pdf(x);
        }
        sum *= h;
        assert!((sum - 1.0).abs() < 1e-4, "Integral = {}", sum);
    }

    #[test]
    fn test_truncated_gamma_cdf_monotone() {
        let tg = TruncatedGamma::new(3.0f64, 0.5, 1.0, 8.0).unwrap();
        let xs = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut prev = 0.0f64;
        for &x in &xs {
            let c = tg.cdf(x);
            assert!(c > prev, "CDF not monotone at {}", x);
            prev = c;
        }
    }

    #[test]
    fn test_truncated_gamma_ppf_roundtrip() {
        let tg = TruncatedGamma::new(2.0f64, 0.5, 0.0, 10.0).unwrap();
        for p in [0.2, 0.4, 0.6, 0.8] {
            let x = tg.ppf(p).unwrap();
            let c = tg.cdf(x);
            assert!(
                (c - p).abs() < 1e-5,
                "PPF roundtrip: ppf({}) = {}, cdf = {}",
                p,
                x,
                c
            );
        }
    }

    #[test]
    fn test_truncated_gamma_rvs() {
        let tg = TruncatedGamma::new(2.0f64, 1.0, 0.5, 5.0).unwrap();
        let samples = tg.rvs(200, Some(99)).unwrap();
        assert_eq!(samples.len(), 200);
        for &s in samples.iter() {
            assert!(s >= 0.5 && s <= 5.0, "Sample {} outside [0.5, 5]", s);
        }
    }

    // --- TruncatedBeta tests ---

    #[test]
    fn test_truncated_beta_pdf_bounds() {
        let tb = TruncatedBeta::new(2.0f64, 5.0, 0.1, 0.9).unwrap();
        assert_eq!(tb.pdf(-0.1), 0.0);
        assert_eq!(tb.pdf(0.95), 0.0);
        assert!(tb.pdf(0.3) > 0.0);
    }

    #[test]
    fn test_truncated_beta_pdf_integrates_to_one() {
        let tb = TruncatedBeta::new(2.0f64, 3.0, 0.1, 0.9).unwrap();
        let n = 1000;
        let h = 0.8 / n as f64;
        let mut sum = 0.0f64;
        for i in 0..n {
            let x = 0.1 + h * i as f64 + h / 2.0;
            sum += tb.pdf(x);
        }
        sum *= h;
        assert!((sum - 1.0).abs() < 1e-4, "Integral = {}", sum);
    }

    #[test]
    fn test_truncated_beta_ppf_roundtrip() {
        let tb = TruncatedBeta::new(3.0f64, 2.0, 0.0, 0.8).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let x = tb.ppf(p).unwrap();
            let c = tb.cdf(x);
            assert!(
                (c - p).abs() < 1e-5,
                "PPF roundtrip: ppf({}) = {}, cdf = {}",
                p,
                x,
                c
            );
        }
    }
}
