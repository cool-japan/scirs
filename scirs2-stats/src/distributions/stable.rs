//! Lévy alpha-stable distribution
//!
//! The alpha-stable (or stable) distribution family is a rich class of heavy-tailed
//! distributions that generalise the Normal and Cauchy distributions. A random
//! variable X ~ S(α, β, γ, δ) is characterised by four parameters:
//!
//! - **α ∈ (0, 2]** (stability index): controls tail heaviness. α=2 → Normal; α=1 → Cauchy.
//! - **β ∈ [-1, 1]** (skewness): 0 gives a symmetric distribution.
//! - **γ > 0** (scale): analogous to standard deviation for the Normal.
//! - **δ ∈ ℝ** (location / shift).
//!
//! # Characteristic function (Zolotarev S₀ parameterisation)
//!
//! For α ≠ 1:
//! ```text
//! log φ(t) = -γ^α |t|^α (1 - iβ·sign(t)·tan(πα/2)) + iδt
//! ```
//! For α = 1:
//! ```text
//! log φ(t) = -γ|t|(1 + iβ·sign(t)·(2/π)·log|t|) + iδt
//! ```
//!
//! # Sampling (Chambers-Mallows-Stuck method)
//!
//! Exact simulation via the CMS algorithm avoids costly numerical inversion.
//!
//! # PDF / CDF
//!
//! No closed-form expressions exist for general (α, β). This implementation
//! uses numerical Fourier inversion of the characteristic function.
//!
//! # Special cases (closed-form)
//!
//! | α | β | Distribution |
//! |---|---|---|
//! | 2 | 0 | Normal(δ, √2 γ) |
//! | 1 | 0 | Cauchy(δ, γ) |
//! | 0.5 | 1 | Lévy(δ, γ) |

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution as _;
use scirs2_core::random::Uniform as RandUniform;

/// Lévy alpha-stable distribution S(α, β, γ, δ)
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::stable::StableDistribution;
///
/// // Standard Cauchy distribution: S(1, 0, 1, 0)
/// let cauchy = StableDistribution::new(1.0f64, 0.0, 1.0, 0.0).expect("valid params");
/// // PDF at 0 for Cauchy(0,1): 1/π ≈ 0.3183
/// let pdf_0 = cauchy.pdf(0.0);
/// assert!((pdf_0 - std::f64::consts::FRAC_1_PI).abs() < 1e-5, "pdf={}", pdf_0);
/// ```
pub struct StableDistribution<F: Float> {
    /// Stability index α ∈ (0, 2]
    pub alpha: F,
    /// Skewness β ∈ [-1, 1]
    pub beta: F,
    /// Scale γ > 0
    pub gamma: F,
    /// Location δ ∈ ℝ
    pub delta: F,
    uniform_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> StableDistribution<F> {
    /// Create a new alpha-stable distribution.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Stability index in (0, 2]
    /// * `beta` - Skewness in [-1, 1]
    /// * `gamma` - Scale (> 0)
    /// * `delta` - Location (shift)
    pub fn new(alpha: F, beta: F, gamma: F, delta: F) -> StatsResult<Self> {
        let alpha_f64: f64 = NumCast::from(alpha).unwrap_or(0.0);
        let beta_f64: f64 = NumCast::from(beta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(gamma).unwrap_or(0.0);

        if alpha_f64 <= 0.0 || alpha_f64 > 2.0 {
            return Err(StatsError::DomainError(
                "Stability index alpha must be in (0, 2]".to_string(),
            ));
        }
        if beta_f64 < -1.0 || beta_f64 > 1.0 {
            return Err(StatsError::DomainError(
                "Skewness beta must be in [-1, 1]".to_string(),
            ));
        }
        if gamma_f64 <= 0.0 {
            return Err(StatsError::DomainError(
                "Scale gamma must be positive".to_string(),
            ));
        }

        let uniform_distr = RandUniform::new(0.0_f64, 1.0_f64).map_err(|_| {
            StatsError::ComputationError("Failed to create uniform distribution".to_string())
        })?;

        Ok(Self {
            alpha,
            beta,
            gamma,
            delta,
            uniform_distr,
        })
    }

    /// Standard stable distribution S(α, β, 1, 0).
    pub fn standard(alpha: F, beta: F) -> StatsResult<Self> {
        Self::new(alpha, beta, F::one(), F::zero())
    }

    /// Generate one sample from the standard S(α, β, 1, 0) distribution
    /// using the Chambers-Mallows-Stuck (CMS) algorithm.
    fn cms_sample_standard<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(2.0);
        let beta_f64: f64 = NumCast::from(self.beta).unwrap_or(0.0);
        let pi = std::f64::consts::PI;

        // Draw U ~ Uniform(-π/2, π/2) and W ~ Exp(1)
        let u_raw: f64 = self.uniform_distr.sample(rng);
        let w_raw: f64 = self.uniform_distr.sample(rng);

        let u = pi * (u_raw - 0.5); // Uniform(-π/2, π/2)
        let w = -w_raw.ln().max(-1e300); // Exp(1), guarded against log(0)

        if (alpha_f64 - 1.0).abs() < 1e-10 {
            // α = 1 case (generalised Cauchy)
            let b_term = if beta_f64.abs() < 1e-15 {
                0.0
            } else {
                let zeta = beta_f64;
                zeta * (pi / 2.0 + beta_f64 * u).tan()
            };
            let term1 = (pi / 2.0 + beta_f64 * u) * u.tan();
            let term2 = beta_f64 * (w * u.cos() / (pi / 2.0 + beta_f64 * u)).ln();
            (term1 + term2) * (2.0 / pi) + b_term
        } else {
            // General α ≠ 1
            let zeta = -beta_f64 * (alpha_f64 * pi / 2.0).tan();
            let xi = (1.0_f64 / alpha_f64) * (-zeta).atan();

            let sin_a_u_xi = (alpha_f64 * (u + xi)).sin();
            let cos_u = u.cos();

            if cos_u.abs() < 1e-300 {
                return zeta; // degenerate
            }

            let a = sin_a_u_xi / cos_u.powf(1.0 / alpha_f64);
            let cos_diff = ((1.0 - alpha_f64) * u - alpha_f64 * xi).cos();

            let b_arg = cos_diff / w;
            if b_arg <= 0.0 {
                return zeta + a;
            }
            let b = b_arg.powf((1.0 - alpha_f64) / alpha_f64);

            a * b - zeta
        }
    }

    /// Generate one sample from S(α, β, γ, δ).
    fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(2.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let beta_f64: f64 = NumCast::from(self.beta).unwrap_or(0.0);

        let z = self.cms_sample_standard(rng);

        // Scale and shift: X = γ·Z + δ' where δ' accounts for the S₀ parameterisation
        if (alpha_f64 - 1.0).abs() < 1e-10 {
            gamma_f64 * z
                + delta_f64
                + (2.0 / std::f64::consts::PI) * beta_f64 * gamma_f64 * gamma_f64.ln()
        } else {
            gamma_f64 * z + delta_f64
        }
    }

    /// Compute the characteristic function log φ(t) at frequency `t`.
    fn log_char_fn(&self, t: f64) -> (f64, f64) {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(2.0);
        let beta_f64: f64 = NumCast::from(self.beta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let pi = std::f64::consts::PI;

        let sign_t = if t > 0.0 {
            1.0
        } else if t < 0.0 {
            -1.0
        } else {
            0.0
        };
        let abs_t = t.abs();
        let g_t = gamma_f64.powf(alpha_f64) * abs_t.powf(alpha_f64);

        let (re_log_phi, im_log_phi) = if (alpha_f64 - 1.0).abs() < 1e-10 {
            let re = -gamma_f64 * abs_t;
            let im =
                delta_f64 * t - (2.0 / pi) * beta_f64 * gamma_f64 * sign_t * abs_t.ln().max(-700.0);
            (re, im)
        } else {
            let tan_term = (alpha_f64 * pi / 2.0).tan();
            let re = -g_t;
            let im = delta_f64 * t + g_t * beta_f64 * sign_t * tan_term;
            (re, im)
        };

        (re_log_phi, im_log_phi)
    }

    /// Numerically invert the characteristic function to compute the PDF.
    ///
    /// Uses a truncated Fourier integral with adaptive step size.
    fn pdf_by_inversion(&self, x: F) -> F {
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);
        let pi = std::f64::consts::PI;

        // Integration parameters — more points for higher accuracy
        let n_points = 4096_usize;
        let t_max = 50.0_f64;
        let dt = t_max / n_points as f64;

        let mut integral = 0.0_f64;

        // Trapezoidal rule: ∫₀^T Re[φ(t) e^{-itx}] dt / π
        for k in 1..n_points {
            let t = k as f64 * dt;
            let (re_log, im_log) = self.log_char_fn(t);
            let amp = re_log.exp();
            let phase = im_log - t * x_f64;
            let re_integrand = amp * phase.cos();

            let weight = if k == 0 || k == n_points - 1 {
                0.5
            } else {
                1.0
            };
            integral += weight * re_integrand * dt;
        }

        let pdf_val = integral / pi;
        F::from(pdf_val.max(0.0)).unwrap_or(F::zero())
    }

    /// Numerically integrate the PDF to compute the CDF.
    fn cdf_by_integration(&self, x: F) -> F {
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);

        // We numerically integrate the PDF from a far-left point to x
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);

        let x_low = delta_f64 - 100.0 * gamma_f64;
        let n_steps = 2000_usize;
        let h = (x_f64 - x_low) / n_steps as f64;

        if h <= 0.0 {
            return F::zero();
        }

        let mut sum = 0.0_f64;
        for k in 0..=n_steps {
            let xi = x_low + k as f64 * h;
            let xi_f = F::from(xi).unwrap_or(F::zero());
            let pdf_val: f64 = NumCast::from(self.pdf(xi_f)).unwrap_or(0.0);
            let weight = if k == 0 || k == n_steps { 0.5 } else { 1.0 };
            sum += weight * pdf_val * h;
        }

        F::from(sum.clamp(0.0, 1.0)).unwrap_or(F::zero())
    }

    /// Probability density function.
    ///
    /// For α=2 (Normal) and α=1 (Cauchy) closed-form expressions are used.
    /// For all other α the PDF is computed via Fourier inversion.
    pub fn pdf(&self, x: F) -> F {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(2.0);

        if (alpha_f64 - 2.0).abs() < 1e-10 {
            // Normal(δ, √2·γ)
            return self.pdf_normal(x);
        }

        if (alpha_f64 - 1.0).abs() < 1e-10 {
            let beta_f64: f64 = NumCast::from(self.beta).unwrap_or(0.0);
            if beta_f64.abs() < 1e-10 {
                return self.pdf_cauchy(x);
            }
        }

        self.pdf_by_inversion(x)
    }

    /// CDF.
    pub fn cdf(&self, x: F) -> F {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(2.0);

        if (alpha_f64 - 2.0).abs() < 1e-10 {
            return self.cdf_normal(x);
        }
        if (alpha_f64 - 1.0).abs() < 1e-10 {
            let beta_f64: f64 = NumCast::from(self.beta).unwrap_or(0.0);
            if beta_f64.abs() < 1e-10 {
                return self.cdf_cauchy(x);
            }
        }

        self.cdf_by_integration(x)
    }

    // ── Closed-form helpers ────────────────────────────────────────────────────

    fn pdf_normal(&self, x: F) -> F {
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);
        let sigma = 2.0_f64.sqrt() * gamma_f64;
        let z = (x_f64 - delta_f64) / sigma;
        let pdf = (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        F::from(pdf).unwrap_or(F::zero())
    }

    fn cdf_normal(&self, x: F) -> F {
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);
        let sigma = 2.0_f64.sqrt() * gamma_f64;
        let z = (x_f64 - delta_f64) / sigma;
        let cdf = 0.5 * (1.0 + erf_approx(z / 2.0_f64.sqrt()));
        F::from(cdf).unwrap_or(F::zero())
    }

    fn pdf_cauchy(&self, x: F) -> F {
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);
        let z = (x_f64 - delta_f64) / gamma_f64;
        let pdf = 1.0 / (std::f64::consts::PI * gamma_f64 * (1.0 + z * z));
        F::from(pdf).unwrap_or(F::zero())
    }

    fn cdf_cauchy(&self, x: F) -> F {
        let delta_f64: f64 = NumCast::from(self.delta).unwrap_or(0.0);
        let gamma_f64: f64 = NumCast::from(self.gamma).unwrap_or(1.0);
        let x_f64: f64 = NumCast::from(x).unwrap_or(0.0);
        let z = (x_f64 - delta_f64) / gamma_f64;
        let cdf = 0.5 + z.atan() / std::f64::consts::PI;
        F::from(cdf).unwrap_or(F::zero())
    }

    /// Generate random samples.
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

    /// Mean of the distribution.
    ///
    /// Exists only when α > 1. For α ≤ 1 the mean is infinite.
    pub fn mean(&self) -> Option<F> {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(0.0);
        if alpha_f64 > 1.0 {
            Some(self.delta)
        } else {
            None
        }
    }

    /// Variance of the distribution.
    ///
    /// Exists and is finite only when α = 2. For α < 2 the variance is infinite.
    pub fn variance(&self) -> Option<F> {
        let alpha_f64: f64 = NumCast::from(self.alpha).unwrap_or(0.0);
        if (alpha_f64 - 2.0).abs() < 1e-10 {
            let two = F::from(2.0).unwrap_or(F::one() + F::one());
            Some(two * self.gamma * self.gamma)
        } else {
            None
        }
    }
}

/// Horner-scheme approximation of erf(x) (Abramowitz & Stegun 7.1.26)
fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for StableDistribution<F> {
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
    fn test_normal_special_case() {
        // S(2, 0, 1, 0) should be N(0, √2)
        let stable = StableDistribution::new(2.0f64, 0.0, 1.0, 0.0).expect("valid params");
        // PDF at 0: 1/(√2·√(2π)) = 1/(2√π) ≈ 0.2821
        let pdf_0 = stable.pdf(0.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
        assert!((pdf_0 - expected).abs() < 1e-6, "pdf_0={}", pdf_0);
    }

    #[test]
    fn test_cauchy_special_case() {
        // S(1, 0, 1, 0) = Cauchy(0, 1)
        let stable = StableDistribution::new(1.0f64, 0.0, 1.0, 0.0).expect("valid params");
        let pdf_0 = stable.pdf(0.0);
        let expected = std::f64::consts::FRAC_1_PI;
        assert!((pdf_0 - expected).abs() < 1e-6, "pdf_0={}", pdf_0);
    }

    #[test]
    fn test_sampling() {
        let mut rng = SmallRng::seed_from_u64(42);
        let stable = StableDistribution::new(1.5f64, 0.0, 1.0, 0.0).expect("valid params");
        let samples = stable.rvs(500, &mut rng).expect("sampling should succeed");
        assert_eq!(samples.len(), 500);
        // Median of S(1.5,0,1,0) should be close to 0
        let mut s: Vec<f64> = samples;
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = s[250];
        assert!(
            median.abs() < 2.0,
            "median {} far from 0 for symmetric stable",
            median
        );
    }

    #[test]
    fn test_mean_variance() {
        let stable_normal = StableDistribution::new(2.0f64, 0.0, 1.0, 0.0).expect("valid");
        assert!(stable_normal.mean().is_some());
        assert!(stable_normal.variance().is_some());

        let stable_cauchy = StableDistribution::new(1.0f64, 0.0, 1.0, 0.0).expect("valid");
        assert!(stable_cauchy.mean().is_none()); // undefined for α≤1
        assert!(stable_cauchy.variance().is_none());

        let stable_15 = StableDistribution::new(1.5f64, 0.0, 1.0, 5.0).expect("valid");
        assert_eq!(stable_15.mean().expect("mean should exist"), 5.0_f64);
        assert!(stable_15.variance().is_none());
    }

    #[test]
    fn test_invalid_params() {
        assert!(StableDistribution::new(0.0f64, 0.0, 1.0, 0.0).is_err()); // α=0
        assert!(StableDistribution::new(2.5f64, 0.0, 1.0, 0.0).is_err()); // α>2
        assert!(StableDistribution::new(1.5f64, 1.5, 1.0, 0.0).is_err()); // |β|>1
        assert!(StableDistribution::new(1.5f64, 0.0, -1.0, 0.0).is_err()); // γ≤0
    }
}
