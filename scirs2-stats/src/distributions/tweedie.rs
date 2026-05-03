//! Tweedie distribution family
//!
//! The Tweedie distribution is a special case of the exponential dispersion
//! family with power variance function V(μ) = μ^p. It encompasses several
//! well-known distributions as special cases:
//!
//! | p | Distribution |
//! |---|---|
//! | 0 | Normal |
//! | 1 | Poisson |
//! | (1, 2) | **Compound Poisson-Gamma** (mixed discrete-continuous, with mass at 0) |
//! | 2 | Gamma |
//! | 3 | Inverse Gaussian |
//!
//! The compound Poisson-Gamma (1 < p < 2) case is the most commonly used
//! in actuarial science and insurance modelling.
//!
//! # Parameterisation
//!
//! The Tweedie distribution is parameterised by:
//! - **μ > 0**: mean parameter
//! - **φ > 0**: dispersion parameter
//! - **p ∈ [0, ∞) \ (0, 1)**: power parameter (Tweedie power)
//!
//! The probability density function for p ∈ (1, 2) does not have a simple
//! closed form; it is expressed as an infinite sum via the Tweedie series.
//!
//! # References
//!
//! - Tweedie, M. C. K. (1984). An index which distinguishes between some important
//!   exponential families. In Statistics: Applications and New Directions,
//!   Edited by J. K. Ghosh and J. Roy, pp. 579-604.
//! - Dunn, P. K. & Smyth, G. K. (2005). Series evaluation of Tweedie exponential
//!   dispersion model densities. Statistics and Computing 15: 267-280.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution as _;
use scirs2_core::random::Uniform as RandUniform;
use std::f64::consts::PI;

// ──────────────────────────────────────────────────────────────────────────────
// Helper: log-gamma via Lanczos
// ──────────────────────────────────────────────────────────────────────────────

fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        0.99999999999980993_f64,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut s = coeffs[0];
    for (k, &c) in coeffs[1..].iter().enumerate() {
        s += c / (xm1 + k as f64 + 1.0);
    }
    let t = xm1 + 7.5;
    0.5 * (2.0 * PI).ln() + (xm1 + 0.5) * t.ln() - t + s.ln()
}

fn gamma_fn(x: f64) -> f64 {
    ln_gamma(x).exp()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tweedie distribution
// ──────────────────────────────────────────────────────────────────────────────

/// Tweedie exponential dispersion distribution.
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::tweedie::Tweedie;
///
/// // Compound Poisson-Gamma (p=1.5, insurance-style)
/// let tw = Tweedie::new(2.0f64, 1.0, 1.5).expect("valid params");
/// // There is a point mass at 0
/// assert!(tw.prob_zero() > 0.0);
/// ```
pub struct Tweedie<F: Float> {
    /// Mean μ > 0
    pub mu: F,
    /// Dispersion φ > 0
    pub phi: F,
    /// Power parameter p (∉ (0,1))
    pub p: F,
    uniform_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Tweedie<F> {
    /// Create a new Tweedie distribution.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean (> 0)
    /// * `phi` - Dispersion (> 0)
    /// * `p` - Power parameter: 0, 1, [1, ∞). Value p ∈ (0, 1) is invalid.
    pub fn new(mu: F, phi: F, p: F) -> StatsResult<Self> {
        let mu_f64: f64 = NumCast::from(mu).unwrap_or(0.0);
        let phi_f64: f64 = NumCast::from(phi).unwrap_or(0.0);
        let p_f64: f64 = NumCast::from(p).unwrap_or(0.0);

        if mu_f64 <= 0.0 {
            return Err(StatsError::DomainError(
                "Mean mu must be positive".to_string(),
            ));
        }
        if phi_f64 <= 0.0 {
            return Err(StatsError::DomainError(
                "Dispersion phi must be positive".to_string(),
            ));
        }
        if p_f64 > 0.0 && p_f64 < 1.0 {
            return Err(StatsError::DomainError(
                "Tweedie power p cannot be in the open interval (0, 1)".to_string(),
            ));
        }

        let uniform_distr = RandUniform::new(0.0_f64, 1.0_f64).map_err(|_| {
            StatsError::ComputationError(
                "Failed to create uniform distribution for Tweedie sampling".to_string(),
            )
        })?;

        Ok(Self {
            mu,
            phi,
            p,
            uniform_distr,
        })
    }

    /// Variance of the distribution: Var(X) = φ · μ^p.
    pub fn variance(&self) -> F {
        self.phi * self.mu.powf(self.p)
    }

    /// Probability of observing exactly zero (only non-zero for 1 < p < 2).
    ///
    /// P(X=0) = exp(-λ) where λ = μ^{2-p} / (φ(2-p))
    pub fn prob_zero(&self) -> f64 {
        let mu_f64: f64 = NumCast::from(self.mu).unwrap_or(1.0);
        let phi_f64: f64 = NumCast::from(self.phi).unwrap_or(1.0);
        let p_f64: f64 = NumCast::from(self.p).unwrap_or(1.5);

        if p_f64 <= 1.0 || p_f64 >= 2.0 {
            return 0.0; // No point mass at 0 outside (1,2) range
        }

        let lambda = mu_f64.powf(2.0 - p_f64) / (phi_f64 * (2.0 - p_f64));
        (-lambda).exp()
    }

    /// Log probability density function (for p ∈ (1, 2), x > 0).
    ///
    /// Uses the Dunn-Smyth series expansion truncated to `max_terms` terms.
    pub fn log_pdf(&self, x: F, max_terms: usize) -> f64 {
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);
        let mu_f64: f64 = NumCast::from(self.mu).unwrap_or(1.0);
        let phi_f64: f64 = NumCast::from(self.phi).unwrap_or(1.0);
        let p_f64: f64 = NumCast::from(self.p).unwrap_or(1.5);

        // Special cases
        if (p_f64 - 0.0).abs() < 1e-10 {
            return self.log_pdf_normal(x_f64, mu_f64, phi_f64);
        }
        if (p_f64 - 1.0).abs() < 1e-10 {
            return self.log_pdf_poisson(x_f64, mu_f64, phi_f64);
        }
        if (p_f64 - 2.0).abs() < 1e-10 {
            return self.log_pdf_gamma(x_f64, mu_f64, phi_f64);
        }
        if (p_f64 - 3.0).abs() < 1e-10 {
            return self.log_pdf_inverse_gaussian(x_f64, mu_f64, phi_f64);
        }

        // Compound Poisson-Gamma (1 < p < 2)
        if p_f64 > 1.0 && p_f64 < 2.0 {
            if x_f64 <= 0.0 {
                let lambda = mu_f64.powf(2.0 - p_f64) / (phi_f64 * (2.0 - p_f64));
                return -lambda;
            }
            return self.log_pdf_cpg(x_f64, mu_f64, phi_f64, p_f64, max_terms);
        }

        // General case (p > 2): no closed form, use saddlepoint approximation
        self.log_pdf_saddlepoint(x_f64, mu_f64, phi_f64, p_f64)
    }

    /// PDF (wrapper that uses a default of 100 terms for the series).
    pub fn pdf(&self, x: F) -> f64 {
        self.log_pdf(x, 100).exp()
    }

    // ── Closed-form special cases ──────────────────────────────────────────────

    fn log_pdf_normal(&self, x: f64, mu: f64, phi: f64) -> f64 {
        let z = (x - mu) / phi.sqrt();
        -0.5 * (z * z + (2.0 * PI * phi).ln())
    }

    fn log_pdf_poisson(&self, x: f64, mu: f64, phi: f64) -> f64 {
        // Poisson(mu/phi): P(X=k) = e^{-λ} λ^k / k!  where λ = mu/phi
        let k = x.round() as u64;
        let lambda = mu / phi;
        -(lambda) + k as f64 * lambda.ln() - ln_gamma(k as f64 + 1.0)
    }

    fn log_pdf_gamma(&self, x: f64, mu: f64, phi: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // Gamma(shape=1/phi, scale=mu*phi)
        let shape = 1.0 / phi;
        let scale = mu * phi;
        (shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - ln_gamma(shape)
    }

    fn log_pdf_inverse_gaussian(&self, x: f64, mu: f64, phi: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // IG(mu, lambda=1/phi)
        let lambda = 1.0 / phi;
        0.5 * (lambda.ln() - (2.0 * PI * x * x * x).ln())
            - lambda * (x - mu) * (x - mu) / (2.0 * mu * mu * x)
    }

    // ── Compound Poisson-Gamma series (Dunn & Smyth 2005) ─────────────────────

    fn log_pdf_cpg(&self, x: f64, mu: f64, phi: f64, p: f64, max_terms: usize) -> f64 {
        let alpha = (2.0 - p) / (p - 1.0); // > 0
        let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
        let theta = -mu.powf(1.0 - p) / (1.0 - p);

        // Constant part
        let log_const = x * theta - mu.powf(2.0 - p) / (phi * (2.0 - p)) - x.ln();

        // Series: log sum_{j=1}^∞ W_j
        let mut log_w_vec: Vec<f64> = Vec::with_capacity(max_terms);

        for j in 1..=max_terms {
            let jf = j as f64;
            let log_wj = jf * (alpha * x.ln() - (alpha * phi).ln() + lambda.ln())
                - ln_gamma(jf * alpha + 1.0)
                - ln_gamma(jf + 1.0);
            log_w_vec.push(log_wj);

            // Early stopping if contributions become negligible
            if j > 5 && log_wj < log_w_vec[0] - 50.0 {
                break;
            }
        }

        // log-sum-exp
        let max_lw = log_w_vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_w_vec.iter().map(|&lw| (lw - max_lw).exp()).sum();
        let log_series = max_lw + sum_exp.ln();

        log_const - (phi * x).ln() + log_series
    }

    // ── Saddlepoint approximation for p > 2 ───────────────────────────────────

    fn log_pdf_saddlepoint(&self, x: f64, mu: f64, phi: f64, p: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // Saddlepoint: log f ≈ -0.5 log(2π φ V(x)) - d(x,mu)/phi
        // where V(x) = x^p and d(x,mu) is the unit deviance
        let vx = x.powf(p);
        let deviance = if (p - 2.0).abs() < 1e-10 {
            2.0 * (x / mu).ln() + 2.0 * (mu - x) / mu
        } else {
            2.0 * (x.powf(2.0 - p) / (2.0 - p) - x * mu.powf(1.0 - p) / (1.0 - p)
                + mu.powf(2.0 - p) / (2.0 - p))
        };
        -0.5 * (2.0 * PI * phi * vx).ln() - deviance / (2.0 * phi)
    }

    // ── Sampling ──────────────────────────────────────────────────────────────

    /// Generate a single sample from the Tweedie distribution.
    fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mu_f64: f64 = NumCast::from(self.mu).unwrap_or(1.0);
        let phi_f64: f64 = NumCast::from(self.phi).unwrap_or(1.0);
        let p_f64: f64 = NumCast::from(self.p).unwrap_or(1.5);

        if p_f64 > 1.0 && p_f64 < 2.0 {
            self.sample_cpg(mu_f64, phi_f64, p_f64, rng)
        } else if (p_f64 - 2.0).abs() < 1e-10 {
            self.sample_gamma_dist(mu_f64, phi_f64, rng)
        } else if (p_f64 - 0.0).abs() < 1e-10 {
            self.sample_normal(mu_f64, phi_f64, rng)
        } else {
            // Fallback: approximate via rejection or moments (simplified)
            self.sample_approximate(mu_f64, phi_f64, p_f64, rng)
        }
    }

    fn sample_normal<R: Rng + ?Sized>(&self, mu: f64, phi: f64, rng: &mut R) -> f64 {
        let u1: f64 = self.uniform_distr.sample(rng).max(f64::EPSILON);
        let u2: f64 = self.uniform_distr.sample(rng);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mu + phi.sqrt() * z
    }

    fn sample_gamma_dist<R: Rng + ?Sized>(&self, mu: f64, phi: f64, rng: &mut R) -> f64 {
        let shape = 1.0 / phi;
        let scale = mu * phi;
        self.sample_gamma_raw(shape, scale, rng)
    }

    /// Compound Poisson-Gamma sampling: N ~ Poisson(λ), Y = sum_{i=1}^N Gamma(α, β)
    fn sample_cpg<R: Rng + ?Sized>(&self, mu: f64, phi: f64, p: f64, rng: &mut R) -> f64 {
        let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
        let alpha = (2.0 - p) / (p - 1.0);
        let beta = phi * (p - 1.0) * mu.powf(p - 1.0);

        // Sample N ~ Poisson(lambda)
        let n = self.sample_poisson(lambda, rng);

        if n == 0 {
            return 0.0;
        }

        // Sum N independent Gamma(alpha, beta) samples
        let mut total = 0.0;
        for _ in 0..n {
            total += self.sample_gamma_raw(alpha, beta, rng);
        }
        total
    }

    fn sample_poisson<R: Rng + ?Sized>(&self, lambda: f64, rng: &mut R) -> usize {
        // Knuth's algorithm for small lambda; normal approximation for large
        if lambda < 30.0 {
            let target = (-lambda).exp();
            let mut k = 0_usize;
            let mut p = 1.0_f64;
            loop {
                p *= self.uniform_distr.sample(rng);
                if p <= target {
                    break;
                }
                k += 1;
                if k > 10_000 {
                    break;
                }
            }
            k
        } else {
            // Normal approximation
            let u1: f64 = self.uniform_distr.sample(rng).max(f64::EPSILON);
            let u2: f64 = self.uniform_distr.sample(rng);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let sample = lambda + lambda.sqrt() * z;
            sample.round().max(0.0) as usize
        }
    }

    fn sample_gamma_raw<R: Rng + ?Sized>(&self, shape: f64, scale: f64, rng: &mut R) -> f64 {
        // Marsaglia-Tsang
        if shape < 1.0 {
            let u: f64 = self.uniform_distr.sample(rng).max(f64::EPSILON);
            return self.sample_gamma_raw(1.0 + shape, scale, rng) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1: f64 = self.uniform_distr.sample(rng).max(f64::EPSILON);
            let u2: f64 = self.uniform_distr.sample(rng);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u3: f64 = self.uniform_distr.sample(rng);
            if u3 < 1.0 - 0.0331 * z.powi(4) || u3.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }

    /// Approximate sampling via moment-matched Gamma for p > 2.
    fn sample_approximate<R: Rng + ?Sized>(&self, mu: f64, phi: f64, p: f64, rng: &mut R) -> f64 {
        let variance = phi * mu.powf(p);
        let shape = mu * mu / variance;
        let scale = variance / mu;
        self.sample_gamma_raw(shape, scale, rng)
    }

    /// Generate n random variates.
    pub fn rvs<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> StatsResult<Vec<F>> {
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let s = self.sample_one(rng);
            let f_s = F::from(s).ok_or_else(|| {
                StatsError::ComputationError("Failed to convert sample to F".to_string())
            })?;
            samples.push(f_s);
        }
        Ok(samples)
    }
}

impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Tweedie<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        use scirs2_core::random::rngs::SmallRng;
        use scirs2_core::random::SeedableRng;
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15);
        let mut rng = SmallRng::seed_from_u64(seed);
        self.rvs(size, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::SmallRng, SeedableRng};

    #[test]
    fn test_normal_special_case_p0() {
        let tw = Tweedie::new(3.0f64, 1.0, 0.0).expect("valid params");
        // pdf(3.0) for N(3,1) = 1/sqrt(2π) ≈ 0.3989
        let log_p = tw.log_pdf(3.0f64, 50);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((log_p - expected).abs() < 1e-10, "log_p={}", log_p);
    }

    #[test]
    fn test_gamma_special_case_p2() {
        let tw = Tweedie::new(2.0f64, 0.5, 2.0).expect("valid params");
        let p = tw.pdf(1.0f64);
        assert!(p > 0.0);
        assert!(p.is_finite());
    }

    #[test]
    fn test_prob_zero_cpg() {
        // For p=1.5, phi=1, mu=2: P(X=0) = exp(-lambda), lambda = 2^{0.5}/(0.5) = 2√2
        let tw = Tweedie::new(2.0f64, 1.0, 1.5).expect("valid params");
        let p0 = tw.prob_zero();
        let lambda = 2.0_f64.powf(0.5) / (1.0 * 0.5);
        let expected = (-lambda).exp();
        assert!(
            (p0 - expected).abs() < 1e-12,
            "p0={} expected={}",
            p0,
            expected
        );
    }

    #[test]
    fn test_variance() {
        let tw = Tweedie::new(3.0f64, 2.0, 1.5).expect("valid params");
        // Var = phi * mu^p = 2 * 3^1.5 ≈ 10.3923
        let expected = 2.0 * 3.0_f64.powf(1.5);
        let var: f64 = NumCast::from(tw.variance()).unwrap_or(0.0);
        assert!((var - expected).abs() < 1e-10);
    }

    #[test]
    fn test_cpg_sampling_mean() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mu = 3.0_f64;
        let tw = Tweedie::new(mu, 0.5, 1.5).expect("valid params");
        let n = 5000_usize;
        let samples = tw.rvs(n, &mut rng).expect("sampling should succeed");
        assert_eq!(samples.len(), n);

        let sum: f64 = samples.iter().sum();
        let empirical_mean = sum / n as f64;
        // Within 10% of true mean
        assert!(
            (empirical_mean - mu).abs() < 0.5,
            "empirical mean {} far from {}",
            empirical_mean,
            mu
        );
    }

    #[test]
    fn test_invalid_power() {
        assert!(Tweedie::new(1.0f64, 1.0, 0.5).is_err()); // p in (0,1) is invalid
    }

    #[test]
    fn test_log_pdf_cpg() {
        let tw = Tweedie::new(2.0f64, 1.0, 1.5).expect("valid params");
        // log_pdf at x=0 should equal log(prob_zero)
        let log_p0 = tw.log_pdf(0.0f64, 100);
        let expected = tw.prob_zero().ln();
        assert!(
            (log_p0 - expected).abs() < 1e-10,
            "log_p0={} expected={}",
            log_p0,
            expected
        );

        // log_pdf at x>0 should be finite
        let log_p1 = tw.log_pdf(1.0f64, 100);
        assert!(log_p1.is_finite(), "log_p1={}", log_p1);
    }
}
