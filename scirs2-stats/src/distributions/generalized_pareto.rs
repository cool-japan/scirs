//! Generalized Pareto Distribution (GPD) for extreme value analysis
//!
//! The GPD is used to model the distribution of exceedances over a threshold
//! in Extreme Value Theory (EVT). It arises as the limiting distribution for
//! threshold exceedances from a broad class of distributions via the
//! Pickands-Balkema-de Haan theorem.
//!
//! # Parameterisation
//!
//! For threshold `u`, shape `ξ` (xi), and scale `σ > 0`:
//!
//! ```text
//! F(x) = 1 - (1 + ξ·(x-u)/σ)^{-1/ξ}   if ξ ≠ 0
//! F(x) = 1 - exp(-(x-u)/σ)               if ξ = 0 (exponential)
//! ```
//!
//! Support:
//!   - `x ≥ u`                      for ξ ≥ 0
//!   - `u ≤ x ≤ u - σ/ξ`           for ξ < 0
//!
//! Special cases:
//! - ξ = 0 → Exponential distribution
//! - ξ = 1 → Uniform distribution (up to reparameterisation)
//! - ξ → ∞ → heavy-tailed (Pareto-like)

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution as _;
use scirs2_core::random::Uniform as RandUniform;

/// Generalized Pareto Distribution
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::generalized_pareto::GeneralizedPareto;
///
/// let gpd = GeneralizedPareto::new(0.5f64, 1.0, 0.0).expect("valid parameters");
/// assert!(gpd.pdf(1.0) > 0.0);
/// ```
pub struct GeneralizedPareto<F: Float> {
    /// Shape parameter ξ (xi); negative ξ gives bounded support
    pub xi: F,
    /// Scale parameter σ > 0
    pub sigma: F,
    /// Location (threshold) parameter u
    pub mu: F,
    rand_distr: RandUniform<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> GeneralizedPareto<F> {
    /// Create a new Generalized Pareto distribution.
    ///
    /// # Arguments
    ///
    /// * `xi` - Shape parameter ξ (any real number)
    /// * `sigma` - Scale parameter σ (must be > 0)
    /// * `mu` - Location (threshold) parameter u
    pub fn new(xi: F, sigma: F, mu: F) -> StatsResult<Self> {
        if sigma <= F::zero() {
            return Err(StatsError::DomainError(
                "Scale parameter sigma must be positive".to_string(),
            ));
        }
        let rand_distr = RandUniform::new(0.0_f64, 1.0_f64).map_err(|_| {
            StatsError::ComputationError(
                "Failed to create uniform distribution for GPD sampling".to_string(),
            )
        })?;
        Ok(Self {
            xi,
            sigma,
            mu,
            rand_distr,
        })
    }

    /// Upper bound of the support when ξ < 0 (`mu - sigma/xi`).
    ///
    /// Returns `None` when ξ ≥ 0 (unbounded support).
    pub fn upper_bound(&self) -> Option<F> {
        if self.xi < F::zero() {
            Some(self.mu - self.sigma / self.xi)
        } else {
            None
        }
    }

    /// Probability density function.
    pub fn pdf(&self, x: F) -> F {
        let z = (x - self.mu) / self.sigma;
        if z < F::zero() {
            return F::zero();
        }
        if let Some(ub) = self.upper_bound() {
            if x > ub {
                return F::zero();
            }
        }

        let xi_f64: f64 = NumCast::from(self.xi).unwrap_or(0.0);
        let abs_xi: f64 = xi_f64.abs();

        if abs_xi < 1e-10 {
            // ξ → 0: Exponential limit
            let exp_term = (-z).exp();
            exp_term / self.sigma
        } else {
            let one = F::one();
            let base = one + self.xi * z;
            if base <= F::zero() {
                return F::zero();
            }
            let exponent = -(one / self.xi + one);
            base.powf(exponent) / self.sigma
        }
    }

    /// Cumulative distribution function.
    pub fn cdf(&self, x: F) -> F {
        let z = (x - self.mu) / self.sigma;
        if z <= F::zero() {
            return F::zero();
        }
        if let Some(ub) = self.upper_bound() {
            if x >= ub {
                return F::one();
            }
        }

        let xi_f64: f64 = NumCast::from(self.xi).unwrap_or(0.0);
        let abs_xi: f64 = xi_f64.abs();

        if abs_xi < 1e-10 {
            // ξ → 0: Exponential
            F::one() - (-z).exp()
        } else {
            let one = F::one();
            let base = one + self.xi * z;
            if base <= F::zero() {
                return F::one();
            }
            let exponent = -one / self.xi;
            F::one() - base.powf(exponent)
        }
    }

    /// Survival function (1 - CDF).
    pub fn sf(&self, x: F) -> F {
        F::one() - self.cdf(x)
    }

    /// Percent point function (inverse CDF / quantile function).
    ///
    /// Returns `Err` if `q` is not in [0, 1).
    pub fn ppf(&self, q: F) -> StatsResult<F> {
        if q < F::zero() || q >= F::one() {
            return Err(StatsError::DomainError(
                "Quantile q must be in [0, 1)".to_string(),
            ));
        }

        let xi_f64: f64 = NumCast::from(self.xi).unwrap_or(0.0);
        let abs_xi: f64 = xi_f64.abs();

        let x = if abs_xi < 1e-10 {
            // Exponential quantile
            self.mu + self.sigma * (-(F::one() - q).ln())
        } else {
            // General GPD quantile
            let one = F::one();
            let base = (one - q).powf(-self.xi) - one;
            self.mu + self.sigma * base / self.xi
        };

        Ok(x)
    }

    /// Log probability density function.
    pub fn logpdf(&self, x: F) -> F {
        let pdf_val = self.pdf(x);
        if pdf_val <= F::zero() {
            F::from(f64::NEG_INFINITY).unwrap_or(F::zero())
        } else {
            pdf_val.ln()
        }
    }

    /// Mean of the distribution.
    ///
    /// Exists only when ξ < 1. Returns `None` otherwise.
    pub fn mean(&self) -> Option<F> {
        let xi_f64: f64 = NumCast::from(self.xi).unwrap_or(0.0);
        if xi_f64 >= 1.0 {
            None
        } else {
            let one = F::one();
            Some(self.mu + self.sigma / (one - self.xi))
        }
    }

    /// Variance of the distribution.
    ///
    /// Exists only when ξ < 0.5. Returns `None` otherwise.
    pub fn variance(&self) -> Option<F> {
        let xi_f64: f64 = NumCast::from(self.xi).unwrap_or(0.0);
        if xi_f64 >= 0.5 {
            None
        } else {
            let one = F::one();
            let two = F::from(2.0).unwrap_or(one + one);
            let numer = self.sigma * self.sigma;
            let denom = (one - self.xi) * (one - self.xi) * (one - two * self.xi);
            Some(numer / denom)
        }
    }

    /// Generate random variates from the GPD using the quantile transform.
    pub fn rvs<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> StatsResult<Vec<F>> {
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let u: f64 = self.rand_distr.sample(rng);
            let q = F::from(u).ok_or_else(|| {
                StatsError::ComputationError("Failed to convert uniform sample to F".to_string())
            })?;
            let x = self.ppf(q)?;
            samples.push(x);
        }
        Ok(samples)
    }

    /// Fit GPD parameters to data exceeding a threshold using MLE.
    ///
    /// Returns `(xi_hat, sigma_hat)` using the method of L-moments.
    pub fn fit_lmoments(data: &[F], threshold: F) -> StatsResult<(F, F)>
    where
        F: std::ops::Sub<Output = F>
            + std::iter::Sum<F>
            + std::ops::Mul<Output = F>
            + Copy
            + PartialOrd,
    {
        let exceedances: Vec<F> = data
            .iter()
            .filter_map(|&x| {
                if x > threshold {
                    Some(x - threshold)
                } else {
                    None
                }
            })
            .collect();

        if exceedances.len() < 3 {
            return Err(StatsError::InsufficientData(
                "Need at least 3 exceedances for L-moment fitting".to_string(),
            ));
        }

        let n = exceedances.len();
        let n_f = F::from(n)
            .ok_or_else(|| StatsError::ComputationError("Failed to convert n to F".to_string()))?;

        // L1 (mean of sorted exceedances)
        let mut sorted = exceedances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let l1: F = sorted.iter().cloned().sum::<F>() / n_f;

        // L2 using plotting positions
        let mut l2_sum = F::zero();
        for (i, &x) in sorted.iter().enumerate() {
            let two = F::from(2.0).unwrap_or(F::one() + F::one());
            let one = F::one();
            let pi = F::from(i + 1)
                .ok_or_else(|| StatsError::ComputationError("Conversion error".to_string()))?;
            let weight = (two * pi - one) / n_f - one;
            l2_sum = l2_sum + weight * x;
        }
        let l2 = l2_sum / n_f;

        // Hosking-Wallis L-moment estimators
        let two = F::from(2.0).unwrap_or(F::one() + F::one());
        let xi_hat = two - l1 / l2;
        let sigma_hat = two * l1 * l2 / (l1 - two * l2 + l2);

        if sigma_hat <= F::zero() {
            return Err(StatsError::ComputationError(
                "L-moment fit produced non-positive scale".to_string(),
            ));
        }

        Ok((xi_hat, sigma_hat))
    }
}

impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for GeneralizedPareto<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        use scirs2_core::random::rngs::SmallRng;
        use scirs2_core::random::SeedableRng;
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15);
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.rand_distr.sample(&mut rng);
            let q = F::from(u).ok_or_else(|| {
                StatsError::ComputationError("Failed to convert uniform sample".to_string())
            })?;
            samples.push(self.ppf(q)?);
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_limit() {
        // ξ = 0 → exponential with rate 1/σ
        let gpd = GeneralizedPareto::new(0.0f64, 1.0, 0.0).expect("valid params");
        // CDF at x=1 for Exp(1): 1 - e^{-1} ≈ 0.6321
        let cdf = gpd.cdf(1.0);
        assert!((cdf - 0.6321205588).abs() < 1e-7, "cdf={}", cdf);
    }

    #[test]
    fn test_pdf_non_negative() {
        let gpd = GeneralizedPareto::new(0.5f64, 1.0, 0.0).expect("valid params");
        for &x in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            assert!(gpd.pdf(x) >= 0.0);
        }
        // Support starts at mu=0
        assert_eq!(gpd.pdf(-0.1), 0.0);
    }

    #[test]
    fn test_bounded_support_negative_xi() {
        // ξ = -0.5, σ = 1, u = 0 → upper bound = 0 - 1/(-0.5) = 2
        let gpd = GeneralizedPareto::new(-0.5f64, 1.0, 0.0).expect("valid params");
        assert!((gpd.upper_bound().expect("should have upper bound") - 2.0).abs() < 1e-12);
        assert_eq!(gpd.pdf(2.5), 0.0); // Outside support
        assert!(gpd.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_cdf_ppf_roundtrip() {
        let gpd = GeneralizedPareto::new(0.3f64, 2.0, 1.0).expect("valid params");
        for &q in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let x = gpd.ppf(q).expect("ppf should succeed");
            let cdf_val = gpd.cdf(x);
            assert!(
                (cdf_val - q).abs() < 1e-9,
                "q={} cdf(ppf(q))={} diff={}",
                q,
                cdf_val,
                (cdf_val - q).abs()
            );
        }
    }

    #[test]
    fn test_mean_variance() {
        let gpd = GeneralizedPareto::new(0.0f64, 2.0, 0.0).expect("valid params");
        // ξ=0 (exponential): mean=σ=2, var=σ²=4
        assert!((gpd.mean().expect("mean exists") - 2.0).abs() < 1e-10);
        assert!((gpd.variance().expect("variance exists") - 4.0).abs() < 1e-10);
    }
}
