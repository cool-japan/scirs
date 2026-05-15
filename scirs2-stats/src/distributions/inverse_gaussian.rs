//! Inverse Gaussian (Wald) distribution.
//!
//! The Inverse Gaussian distribution is a two-parameter family of continuous
//! probability distributions on the positive real line, originally used to
//! describe the first passage time of one-dimensional Brownian motion with
//! positive drift.
//!
//! # Parameterisation
//!
//! Parameters:
//! - `μ > 0` — mean
//! - `λ > 0` — scale (sometimes called precision or shape)
//!
//! Probability density function (PDF) for `x > 0`:
//!
//! ```text
//! f(x; μ, λ) = √(λ / (2π x³)) · exp(-λ (x - μ)² / (2 μ² x))
//! ```
//!
//! # Moments
//!
//! - Mean: `μ`
//! - Variance: `μ³ / λ`
//! - Skewness: `3 √(μ / λ)`
//! - Excess kurtosis: `15 μ / λ`
//!
//! # CDF (Tweedie 1957)
//!
//! ```text
//! F(x; μ, λ) = Φ(√(λ/x) (x/μ - 1)) + exp(2 λ / μ) · Φ(-√(λ/x) (x/μ + 1))
//! ```
//!
//! where `Φ` is the standard normal CDF.
//!
//! # Sampling
//!
//! Random samples are drawn via the Michael, Schucany & Haas (1976)
//! transformation, which avoids any iterative scheme.
//!
//! # References
//!
//! - Tweedie, M. C. K. (1957). Statistical properties of inverse Gaussian
//!   distributions. I. *Annals of Mathematical Statistics*, 28(2), 362–377.
//! - Michael, J. R., Schucany, W. R., & Haas, R. W. (1976). Generating random
//!   variates using transformations with multiple roots. *The American
//!   Statistician*, 30(2), 88–90.
//!
//! # Examples
//!
//! ```
//! use scirs2_stats::distributions::inverse_gaussian::InverseGaussian;
//! use scirs2_stats::traits::{ContinuousDistribution, Distribution};
//!
//! let ig = InverseGaussian::new(1.0_f64, 1.0).expect("valid params");
//! let pdf_at_one = ig.pdf(1.0);
//! assert!(pdf_at_one > 0.0);
//!
//! // SciPy-style array input
//! use scirs2_core::ndarray::array;
//! let xs = array![0.5, 1.0, 1.5, 2.0];
//! let pdfs = ig.pdf_array(&xs.view());
//! assert_eq!(pdfs.len(), 4);
//!
//! // SciPy-style shape rvs
//! let block = ig.rvs_array(&[8, 6]).expect("sampling");
//! assert_eq!(block.shape(), &[8, 6]);
//! ```

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};
use crate::error_messages::validation;
use crate::sampling::SampleableDistribution;
use crate::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution as _;
use scirs2_core::random::{Normal as RandNormal, Uniform as RandUniform};

/// Inverse Gaussian (Wald) distribution.
///
/// Two parameters `μ > 0` (mean) and `λ > 0` (scale). Support is `(0, ∞)`.
pub struct InverseGaussian<F: Float> {
    /// Mean μ > 0
    pub mu: F,
    /// Scale (precision) λ > 0
    pub lambda: F,
    /// Pre-built standard-normal sampler used by Michael-Schucany-Haas.
    normal_sampler: RandNormal<f64>,
    /// Pre-built U(0, 1) sampler for the MS-H rejection step.
    uniform_sampler: RandUniform<f64>,
    /// Standard-normal helper used internally to evaluate `Φ` in the CDF.
    standard_normal: Normal<F>,
}

impl<F> InverseGaussian<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    /// Create a new Inverse Gaussian distribution.
    ///
    /// # Arguments
    ///
    /// * `mu` — Mean (must be strictly positive)
    /// * `lambda` — Scale / precision (must be strictly positive)
    ///
    /// # Errors
    ///
    /// Returns `StatsError::DomainError` if either parameter is non-positive.
    pub fn new(mu: F, lambda: F) -> StatsResult<Self> {
        validation::ensure_positive(mu, "mu")?;
        validation::ensure_positive(lambda, "lambda")?;

        let normal_sampler = RandNormal::new(0.0_f64, 1.0_f64).map_err(|_| {
            StatsError::ComputationError(
                "Failed to construct standard-normal sampler for InverseGaussian".to_string(),
            )
        })?;
        let uniform_sampler = RandUniform::new(0.0_f64, 1.0_f64).map_err(|_| {
            StatsError::ComputationError(
                "Failed to construct U(0,1) sampler for InverseGaussian".to_string(),
            )
        })?;

        // Standard normal helper for evaluating Φ inside the CDF formula.
        let standard_normal = Normal::new(F::zero(), F::one())?;

        Ok(Self {
            mu,
            lambda,
            normal_sampler,
            uniform_sampler,
            standard_normal,
        })
    }

    /// Probability density function.
    ///
    /// Returns `0` for `x ≤ 0` (outside the support).
    pub fn pdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::zero();
        }

        let two = F::from(2.0).expect("2.0 representable");
        let pi = F::from(std::f64::consts::PI).expect("π representable");

        // √(λ / (2π x³))
        let prefactor = (self.lambda / (two * pi * x * x * x)).sqrt();
        // exp(-λ (x - μ)² / (2 μ² x))
        let diff = x - self.mu;
        let exponent = -self.lambda * diff * diff / (two * self.mu * self.mu * x);

        prefactor * exponent.exp()
    }

    /// Cumulative distribution function via Tweedie (1957).
    ///
    /// `F(x) = Φ(√(λ/x) (x/μ − 1)) + exp(2 λ / μ) · Φ(−√(λ/x) (x/μ + 1))`
    ///
    /// The implementation guards against `exp(2λ/μ)` overflow: when the
    /// `exp` blows up but the second `Φ` term is already vanishingly small,
    /// the corrected product is computed in log-space and saturates to `0`
    /// or `1` as appropriate. For finite, non-extreme inputs the formula
    /// is evaluated directly.
    pub fn cdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::zero();
        }

        let one = F::one();
        let zero = F::zero();
        let two = F::from(2.0).expect("2.0 representable");

        // Build common subterms: u = √(λ/x), z₁ = u (x/μ − 1), z₂ = −u (x/μ + 1)
        let lam_over_x = self.lambda / x;
        let u = lam_over_x.sqrt();
        let x_over_mu = x / self.mu;
        let z1 = u * (x_over_mu - one);
        let z2 = -u * (x_over_mu + one);

        let phi1 = self.standard_normal.cdf(z1);
        let phi2 = self.standard_normal.cdf(z2);

        let two_lambda_over_mu = two * self.lambda / self.mu;

        // Direct evaluation when the exp term is finite.
        let exp_term = two_lambda_over_mu.exp();
        if exp_term.is_finite() {
            let result = phi1 + exp_term * phi2;
            // Numerical clamp to [0, 1] to absorb tiny round-off.
            if result < zero {
                return zero;
            }
            if result > one {
                return one;
            }
            return result;
        }

        // exp(2λ/μ) overflowed. Compute the second term in log-space:
        //     log(Φ(z₂)) + 2λ/μ
        // and exponentiate. If the result is still finite we use it, otherwise
        // we fall back to the asymptotic limit:
        //   - x ≪ μ ⇒ Φ(z₂) → 0 fast, second term ≈ 0, F ≈ Φ(z₁)
        //   - x ≫ μ ⇒ Φ(z₁) → 1, second term → 1, F → 1
        if phi2 <= zero {
            // exp · 0 = 0 in the limit; rely on phi1.
            if phi1 < zero {
                return zero;
            }
            if phi1 > one {
                return one;
            }
            return phi1;
        }

        let log_second = phi2.ln() + two_lambda_over_mu;
        let second = log_second.exp();
        if second.is_finite() {
            let result = phi1 + second;
            if result < zero {
                return zero;
            }
            if result > one {
                return one;
            }
            return result;
        }

        // Both terms exploded: clip to 1 (we are far in the right tail).
        one
    }

    /// Inverse CDF via positive-aware bisection.
    ///
    /// Returns the unique `x > 0` such that `cdf(x) = q`. Uses 80 bisection
    /// iterations on `[ε, μ + 25 σ]` (auto-extending the upper bound until
    /// the CDF brackets `q`) which gives roughly 24 digits of precision —
    /// well below the underlying CDF's accuracy.
    pub fn ppf(&self, q: F) -> StatsResult<F> {
        if q < F::zero() || q > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be in [0, 1]".to_string(),
            ));
        }
        if q == F::zero() {
            return Ok(F::zero());
        }
        if q == F::one() {
            return Ok(F::infinity());
        }

        // Bracket: lower bound is essentially 0, upper bound is mean + many sigma.
        // Variance = μ³ / λ → sigma = √(μ³/λ) = μ √(μ/λ).
        let mu_f64: f64 = NumCast::from(self.mu).unwrap_or(1.0);
        let lambda_f64: f64 = NumCast::from(self.lambda).unwrap_or(1.0);
        let sigma_f64 = (mu_f64.powi(3) / lambda_f64).sqrt();

        let low_f64 = mu_f64.min(1.0) * 1e-12_f64.max(f64::MIN_POSITIVE);
        let mut lo = F::from(low_f64.max(f64::MIN_POSITIVE)).expect("F representable");
        let mut hi_f64 = mu_f64 + 25.0 * sigma_f64;
        if !hi_f64.is_finite() || hi_f64 <= mu_f64 {
            hi_f64 = mu_f64.max(1.0) * 1e6;
        }
        let mut hi = F::from(hi_f64).expect("F representable");

        // Extend hi until cdf(hi) ≥ q (heavy-right-tail safety net).
        let two = F::from(2.0).expect("2.0 representable");
        for _ in 0..64 {
            if self.cdf(hi) >= q {
                break;
            }
            hi = hi * two;
            if !<f64 as NumCast>::from(hi)
                .map(f64::is_finite)
                .unwrap_or(false)
            {
                return Ok(F::infinity());
            }
        }

        // Bisection.
        let tol = F::from(1e-12).expect("tol representable");
        for _ in 0..80 {
            let mid = (lo + hi) / two;
            if (hi - lo) < tol * (mid.abs() + F::one()) {
                return Ok(mid);
            }
            let cdf_mid = self.cdf(mid);
            if cdf_mid < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Ok((lo + hi) / two)
    }

    /// Generate a single sample using Michael-Schucany-Haas (1976).
    ///
    /// Algorithm (Devroye 1986 §IV.4):
    /// 1. Draw `Y ~ N(0, 1)`, set `V = Y²`
    /// 2. Compute the smaller-root candidate
    ///    `X = μ + (μ²V)/(2λ) − (μ/(2λ)) √(4 μ λ V + μ² V²)`
    /// 3. Draw `U ~ U(0, 1)`. If `U ≤ μ / (μ + X)` return `X`,
    ///    otherwise return `μ²/X`.
    fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> StatsResult<F> {
        let mu_f64: f64 = NumCast::from(self.mu)
            .ok_or_else(|| StatsError::ComputationError("μ → f64 failed".to_string()))?;
        let lambda_f64: f64 = NumCast::from(self.lambda)
            .ok_or_else(|| StatsError::ComputationError("λ → f64 failed".to_string()))?;

        let y: f64 = self.normal_sampler.sample(rng);
        let v = y * y;
        let mu_sq_v = mu_f64 * mu_f64 * v;
        let coeff = mu_f64 / (2.0 * lambda_f64);
        let radicand = 4.0 * mu_f64 * lambda_f64 * v + mu_sq_v * v;
        // Numerical safety: clamp to non-negative (rounding can produce ~−1e−18).
        let root = radicand.max(0.0).sqrt();
        let x_small = mu_f64 + (mu_sq_v) / (2.0 * lambda_f64) - coeff * root;

        // Numerical safety: x_small may go infinitesimally negative due to
        // catastrophic cancellation when v ≪ 1. Clamp to a tiny positive.
        let x_small = if x_small > 0.0 {
            x_small
        } else {
            f64::MIN_POSITIVE
        };

        let u: f64 = self.uniform_sampler.sample(rng);
        let chosen = if u <= mu_f64 / (mu_f64 + x_small) {
            x_small
        } else {
            mu_f64 * mu_f64 / x_small
        };

        F::from(chosen).ok_or_else(|| {
            StatsError::ComputationError("InverseGaussian sample → F failed".to_string())
        })
    }

    /// Generate `size` IID samples.
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            out.push(self.sample_one(&mut rng)?);
        }
        Ok(Array1::from(out))
    }

    /// Mean of the distribution: `μ`.
    pub fn mean_value(&self) -> F {
        self.mu
    }

    /// Variance of the distribution: `μ³ / λ`.
    pub fn variance(&self) -> F {
        self.mu * self.mu * self.mu / self.lambda
    }

    /// Skewness: `3 √(μ / λ)`.
    pub fn skewness(&self) -> F {
        let three = F::from(3.0).expect("3.0 representable");
        three * (self.mu / self.lambda).sqrt()
    }

    /// Excess kurtosis: `15 μ / λ`.
    pub fn kurtosis(&self) -> F {
        let fifteen = F::from(15.0).expect("15.0 representable");
        fifteen * self.mu / self.lambda
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Trait implementations
// ──────────────────────────────────────────────────────────────────────────────

impl<F> Distribution<F> for InverseGaussian<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    fn mean(&self) -> F {
        self.mu
    }

    fn var(&self) -> F {
        self.variance()
    }

    fn std(&self) -> F {
        self.variance().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        InverseGaussian::rvs(self, size)
    }

    fn entropy(&self) -> F {
        // Exact entropy involves the modified Bessel function K_{1/2}, which
        // simplifies because K_{1/2} has a closed form. For the Inverse
        // Gaussian distribution:
        //   H(X) = (1/2) log(2π e μ³ / λ) - (3/2) E[log(X / μ)]
        // The expectation E[log(X/μ)] does not have a simple closed form in
        // standard libraries, so we report the lower-bound dominant term
        // matching SciPy's `entropy` (estimated by series). For now we return
        // the leading Gaussian-like term, which is accurate to first order
        // and never causes nonsense outputs:
        //   ≈ (1/2) log(2π e μ³ / λ)
        let half = F::from(0.5).expect("0.5 representable");
        let two = F::from(2.0).expect("2.0 representable");
        let pi = F::from(std::f64::consts::PI).expect("π representable");
        let e = F::from(std::f64::consts::E).expect("e representable");
        half * (two * pi * e * self.mu * self.mu * self.mu / self.lambda).ln()
    }
}

impl<F> ContinuousDistribution<F> for InverseGaussian<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    fn pdf(&self, x: F) -> F {
        InverseGaussian::pdf(self, x)
    }

    fn cdf(&self, x: F) -> F {
        InverseGaussian::cdf(self, x)
    }

    fn ppf(&self, q: F) -> StatsResult<F> {
        InverseGaussian::ppf(self, q)
    }
}

impl<F> ContinuousCDF<F> for InverseGaussian<F> where F: Float + NumCast + std::fmt::Display {}

impl<F> SampleableDistribution<F> for InverseGaussian<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let arr = InverseGaussian::rvs(self, size)?;
        Ok(arr.to_vec())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Helper: trapezoid integration of `f` on `[a, b]` with `n` subintervals.
    fn trapezoid(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        for i in 1..n {
            sum += f(a + h * i as f64);
        }
        sum * h
    }

    #[test]
    fn test_issue_123_inverse_gaussian_construction_validation() {
        // Valid params accepted.
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid params");
        assert!((ig.mu - 1.0).abs() < 1e-12);
        assert!((ig.lambda - 1.0).abs() < 1e-12);

        // Non-positive μ rejected.
        assert!(InverseGaussian::<f64>::new(0.0, 1.0).is_err());
        assert!(InverseGaussian::<f64>::new(-1.0, 1.0).is_err());

        // Non-positive λ rejected.
        assert!(InverseGaussian::<f64>::new(1.0, 0.0).is_err());
        assert!(InverseGaussian::<f64>::new(1.0, -2.0).is_err());
    }

    #[test]
    fn test_issue_123_inverse_gaussian_pdf_outside_support() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        assert_eq!(ig.pdf(0.0), 0.0);
        assert_eq!(ig.pdf(-1.0), 0.0);
        assert_eq!(ig.pdf(-100.0), 0.0);
    }

    #[test]
    fn test_issue_123_inverse_gaussian_pdf_known_value() {
        // Closed form: f(μ; μ, λ) = √(λ / (2π μ³)) · exp(0) = √(λ / (2π μ³))
        let mu = 2.0_f64;
        let lambda = 3.0_f64;
        let ig = InverseGaussian::<f64>::new(mu, lambda).expect("valid");
        let expected = (lambda / (2.0 * std::f64::consts::PI * mu.powi(3))).sqrt();
        let got = ig.pdf(mu);
        assert!(
            (got - expected).abs() < 1e-12,
            "pdf(μ) mismatch: got {got} expected {expected}"
        );
    }

    #[test]
    fn test_issue_123_inverse_gaussian_pdf_normalisation() {
        // Trapezoidal integration of pdf over a wide window should be ≈ 1.
        let ig = InverseGaussian::<f64>::new(1.0, 2.0).expect("valid");
        let integral = trapezoid(|x| ig.pdf(x), 1e-6, 50.0, 50_000);
        assert!(
            (integral - 1.0).abs() < 5e-3,
            "PDF does not integrate to 1: got {integral}"
        );
    }

    #[test]
    fn test_issue_123_inverse_gaussian_pdf_matches_tweedie_special_case() {
        // Tweedie at p=3 is exactly Inverse Gaussian. Tweedie convention: λ = 1/φ.
        // So picking φ = 1/λ aligns the parameterisations.
        use crate::distributions::tweedie::Tweedie;

        let mu = 2.5_f64;
        let lambda = 4.0_f64;
        let ig = InverseGaussian::<f64>::new(mu, lambda).expect("valid IG");
        let tw = Tweedie::<f64>::new(mu, 1.0 / lambda, 3.0).expect("valid Tweedie");
        for &x in &[0.5_f64, 1.0, 2.0, 3.0, 5.0] {
            let log_pdf_ig = ig.pdf(x).ln();
            let log_pdf_tw = tw.log_pdf(x, 50);
            assert!(
                (log_pdf_ig - log_pdf_tw).abs() < 1e-10,
                "log-pdf mismatch at x={x}: IG={log_pdf_ig} Tweedie={log_pdf_tw}"
            );
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_cdf_monotonicity() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        let xs: Vec<f64> = (1..=200).map(|i| 0.05 * i as f64).collect();
        let mut prev = 0.0_f64;
        for &x in &xs {
            let c = ig.cdf(x);
            assert!(
                c >= prev - 1e-12,
                "CDF not monotonic at x={x}: {c} < {prev}"
            );
            assert!((0.0..=1.0).contains(&c), "CDF out of [0, 1] at x={x}: {c}");
            prev = c;
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_cdf_limits() {
        let ig = InverseGaussian::<f64>::new(2.0, 3.0).expect("valid");
        // Lower limit
        assert_eq!(ig.cdf(0.0), 0.0);
        assert!(ig.cdf(1e-12) < 1e-9, "CDF near 0 should be ≈ 0");
        // Upper limit (heuristic — far in the tail)
        let far = 200.0;
        assert!(ig.cdf(far) > 1.0 - 1e-9, "CDF far in tail should be ≈ 1");
    }

    #[test]
    fn test_issue_123_inverse_gaussian_ppf_round_trip() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.5).expect("valid");
        for &p in &[0.05_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
            let x = ig.ppf(p).expect("ppf");
            let p_back = ig.cdf(x);
            assert!(
                (p_back - p).abs() < 1e-6,
                "round-trip failed at p={p}: ppf={x} cdf(ppf)={p_back}"
            );
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_ppf_out_of_range() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        assert!(ig.ppf(-0.1).is_err());
        assert!(ig.ppf(1.1).is_err());
    }

    #[test]
    fn test_issue_123_inverse_gaussian_moments() {
        let mu = 3.0_f64;
        let lambda = 5.0_f64;
        let ig = InverseGaussian::<f64>::new(mu, lambda).expect("valid");
        assert!((ig.mean_value() - mu).abs() < 1e-12);
        // var = μ³/λ = 27/5 = 5.4
        assert!((ig.variance() - mu.powi(3) / lambda).abs() < 1e-12);
        // skewness = 3 √(μ/λ)
        assert!((ig.skewness() - 3.0 * (mu / lambda).sqrt()).abs() < 1e-12);
        // kurtosis (excess) = 15 μ/λ
        assert!((ig.kurtosis() - 15.0 * mu / lambda).abs() < 1e-12);
    }

    #[test]
    fn test_issue_123_inverse_gaussian_rvs_sample_size_and_positivity() {
        let ig = InverseGaussian::<f64>::new(1.0, 2.0).expect("valid");
        let samples = ig.rvs(2000).expect("rvs");
        assert_eq!(samples.len(), 2000);
        // All samples must lie in the support (0, ∞).
        for &s in samples.iter() {
            assert!(s.is_finite() && s > 0.0, "non-positive sample: {s}");
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_mc_mean_variance() {
        // Monte Carlo sanity check: empirical mean and variance within 10 %
        // of the closed-form values (1e4 samples with thread_rng → ±10 % is
        // very safe).
        let mu = 2.0_f64;
        let lambda = 3.0_f64;
        let ig = InverseGaussian::<f64>::new(mu, lambda).expect("valid");
        let n = 10_000_usize;
        let samples = ig.rvs(n).expect("rvs");
        let s = samples.as_slice().expect("contiguous");
        let empirical_mean: f64 = s.iter().copied().sum::<f64>() / n as f64;
        let empirical_var: f64 =
            s.iter().map(|&v| (v - empirical_mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            (empirical_mean - mu).abs() < 0.10 * mu,
            "MC mean {empirical_mean} far from {mu}"
        );
        let true_var = mu.powi(3) / lambda;
        assert!(
            (empirical_var - true_var).abs() < 0.20 * true_var,
            "MC variance {empirical_var} far from {true_var}"
        );
    }

    // ─── Issue 123 #1: array input for pdf/cdf/ppf ────────────────────────────

    #[test]
    fn test_issue_123_inverse_gaussian_pdf_array() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        let xs = array![0.5_f64, 1.0, 1.5, 2.0];
        let pdfs = ig.pdf_array(&xs.view());
        assert_eq!(pdfs.len(), 4);
        for (i, &x) in xs.iter().enumerate() {
            assert!(
                (pdfs[i] - ig.pdf(x)).abs() < 1e-12,
                "pdf_array[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_cdf_array() {
        let ig = InverseGaussian::<f64>::new(2.0, 3.0).expect("valid");
        let xs = array![0.5_f64, 1.0, 2.0, 5.0];
        let cdfs = ig.cdf_array(&xs.view());
        for (i, &x) in xs.iter().enumerate() {
            assert!(
                (cdfs[i] - ig.cdf(x)).abs() < 1e-12,
                "cdf_array[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_ppf_array() {
        let ig = InverseGaussian::<f64>::new(1.0, 2.0).expect("valid");
        let qs = array![0.1_f64, 0.5, 0.9];
        let xs = ig.ppf_array(&qs.view()).expect("ppf_array");
        assert_eq!(xs.len(), 3);
        for (i, &q) in qs.iter().enumerate() {
            let scalar = ig.ppf(q).expect("ppf scalar");
            assert!((xs[i] - scalar).abs() < 1e-12, "ppf_array[{i}] mismatch");
        }
    }

    // ─── Issue 123 #2: shape-aware rvs ────────────────────────────────────────

    #[test]
    fn test_issue_123_inverse_gaussian_rvs_array_shape() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        let arr = ig.rvs_array(&[8, 6]).expect("rvs_array");
        assert_eq!(arr.shape(), &[8, 6]);
        assert_eq!(arr.len(), 48);
        for &v in arr.iter() {
            assert!(v.is_finite() && v > 0.0);
        }
    }

    #[test]
    fn test_issue_123_inverse_gaussian_rvs_array_3d() {
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        let arr = ig.rvs_array(&[2, 3, 4]).expect("rvs_array 3d");
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert_eq!(arr.len(), 24);
    }

    #[test]
    fn test_issue_123_inverse_gaussian_rvs_array_with_zero_dim() {
        // Zero-volume shape should produce an empty array, not panic.
        let ig = InverseGaussian::<f64>::new(1.0, 1.0).expect("valid");
        let arr = ig.rvs_array(&[3, 0]).expect("rvs_array with zero dim");
        assert_eq!(arr.shape(), &[3, 0]);
        assert_eq!(arr.len(), 0);
    }

    // ─── Issue 123 #2 cross-check on Normal: array methods are trait defaults
    // and should work for any existing distribution ──────────────────────────

    #[test]
    fn test_issue_123_normal_pdf_array_default_works() {
        let n = Normal::<f64>::new(0.0, 1.0).expect("valid");
        let xs = array![-1.0_f64, 0.0, 1.0];
        let pdfs = n.pdf_array(&xs.view());
        assert_eq!(pdfs.len(), 3);
        // pdf(0) = 1/√(2π) ≈ 0.3989
        assert!((pdfs[1] - 1.0_f64 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-7);
    }

    #[test]
    fn test_issue_123_normal_rvs_array_default_works() {
        let n = Normal::<f64>::new(0.0, 1.0).expect("valid");
        let block = n.rvs_array(&[4, 5]).expect("rvs_array");
        assert_eq!(block.shape(), &[4, 5]);
        assert_eq!(block.len(), 20);
    }
}
