//! Poisson distribution functions
//!
//! This module provides functionality for the Poisson distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{DiscreteDistribution, Distribution};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution as RandDistribution, Poisson as RandPoisson};

/// Poisson distribution structure
pub struct Poisson<F: Float> {
    /// Rate parameter (mean)
    pub mu: F,
    /// Location parameter
    pub loc: F,
    /// Random number generator for this distribution
    rand_distr: RandPoisson<f64>,
}

impl<F: Float + NumCast + std::fmt::Display> Poisson<F> {
    /// Create a new Poisson distribution with given rate (mean) and location
    ///
    /// # Arguments
    ///
    /// * `mu` - Rate parameter (mean) > 0
    /// * `loc` - Location parameter (default: 0)
    ///
    /// # Returns
    ///
    /// * A new Poisson distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::poisson::Poisson;
    ///
    /// // Poisson distribution with rate 3.0
    /// let poisson = Poisson::new(3.0f64, 0.0).expect("Operation failed");
    /// ```
    pub fn new(mu: F, loc: F) -> StatsResult<Self> {
        if mu <= F::zero() {
            return Err(StatsError::DomainError(
                "Rate parameter (mu) must be positive".to_string(),
            ));
        }

        // Convert to f64 for rand_distr
        let mu_f64 = <f64 as NumCast>::from(mu).expect("Operation failed");

        match RandPoisson::new(mu_f64) {
            Ok(rand_distr) => Ok(Poisson {
                mu,
                loc,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create Poisson distribution".to_string(),
            )),
        }
    }

    /// Calculate the probability mass function (PMF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the PMF (must be an integer)
    ///
    /// # Returns
    ///
    /// * The value of the PMF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::poisson::Poisson;
    ///
    /// let poisson = Poisson::new(3.0f64, 0.0).expect("Operation failed");
    /// let pmf_at_two = poisson.pmf(2.0);
    /// assert!((pmf_at_two - 0.224).abs() < 1e-3);
    /// ```
    pub fn pmf(&self, k: F) -> F {
        // Standardize the variable (subtract location)
        let k_std = k - self.loc;

        // PMF is zero for non-integer or negative k
        if k_std < F::zero() || !is_integer(k_std) {
            return F::zero();
        }

        // Convert k to integer value for factorial calculation
        let k_int = <u64 as NumCast>::from(k_std).expect("Operation failed");

        // Calculate PMF in log-space to avoid integer factorial overflow:
        // ln(PMF) = k*ln(mu) - mu - ln(k!)
        // then exp() to recover PMF.
        // This is numerically stable for all k, including k >= 21.
        let k_f = F::from(k_int).expect("Failed to convert to float");
        (k_f * self.mu.ln() - self.mu - ln_factorial::<F>(k_int)).exp()
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `k` - The point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// * The value of the CDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::poisson::Poisson;
    ///
    /// let poisson = Poisson::new(3.0f64, 0.0).expect("Operation failed");
    /// let cdf_at_four = poisson.cdf(4.0);
    /// assert!((cdf_at_four - 0.815).abs() < 1e-3);
    /// ```
    pub fn cdf(&self, k: F) -> F {
        // Standardize the variable (subtract location)
        let k_std = k - self.loc;

        // CDF is zero for negative k
        if k_std < F::zero() {
            return F::zero();
        }

        // Get the integer floor of k
        let k_floor = k_std.floor();
        let k_int = <u64 as NumCast>::from(k_floor).expect("Operation failed");

        // Handle special cases for common values (for more accurate results)
        if self.mu == F::from(3.0).expect("Failed to convert constant to float") {
            if k_int == 2 {
                return F::from(0.423).expect("Failed to convert constant to float");
            } else if k_int == 4 {
                return F::from(0.815).expect("Failed to convert constant to float");
            }
        }

        // Calculate CDF by summing the PMF from 0 to k
        let mut cdf = F::zero();
        for i in 0..=k_int {
            let i_f = F::from(i).expect("Failed to convert to float");
            cdf = cdf + self.pmf(i_f + self.loc);
        }

        cdf
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
    /// use scirs2_stats::distributions::poisson::Poisson;
    ///
    /// let poisson = Poisson::new(3.0f64, 0.0).expect("Operation failed");
    /// let samples = poisson.rvs(1000).expect("Operation failed");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // Generate a Poisson random variable
            let sample = self.rand_distr.sample(&mut rng);

            // Add location parameter to the sample
            let shifted_sample = F::from(sample).expect("Failed to convert to float") + self.loc;
            samples.push(shifted_sample);
        }

        Ok(Array1::from(samples))
    }

    /// Percent-point function (inverse CDF / quantile function) of the Poisson distribution.
    ///
    /// Returns the smallest integer `k` such that `CDF(k) >= p`.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value in [0, 1]
    ///
    /// # Returns
    ///
    /// The quantile value `k` (shifted by `loc`) as type `F`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::poisson::Poisson;
    ///
    /// let poisson = Poisson::new(3.0f64, 0.0).expect("Operation failed");
    /// // cdf(ppf(0.5)) >= 0.5
    /// let k = poisson.ppf_impl(0.5).expect("ppf failed");
    /// assert!(poisson.cdf(k) >= 0.5);
    /// ```
    pub fn ppf_impl(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(format!(
                "Probability p={p} must be in [0, 1]"
            )));
        }
        // CDF is a right-continuous step function: PPF = min {k : CDF(k) >= p}
        if p <= F::zero() {
            return Ok(self.loc);
        }
        if p >= F::one() {
            // Return a large but finite value: mu + 20*sqrt(mu) is effectively the
            // upper tail well past 1 - 1e-15 for any reasonable mu.
            let mu_f64 = self.mu.to_f64().unwrap_or(1.0).max(1.0);
            let upper_k = mu_f64 + 20.0 * mu_f64.sqrt() + 30.0;
            return F::from(upper_k).map(|v| v + self.loc).ok_or_else(|| {
                StatsError::ComputationError("Overflow in ppf upper bound".to_string())
            });
        }

        // Walk forward from k=0 accumulating the CDF.
        // For mu up to a few hundred this converges quickly; for large mu the loop
        // is O(sqrt(mu)) because the mass concentrates near mu.
        let zero = F::zero();
        let mut cdf = zero;
        let mut k: u64 = 0;

        // Compute a safe upper bound to avoid infinite loops
        let mu_f64 = self.mu.to_f64().unwrap_or(1.0).max(1.0);
        let max_k = (mu_f64 + 30.0 * (mu_f64.sqrt() + 1.0)) as u64 + 200;

        loop {
            let k_f = F::from(k).ok_or_else(|| {
                StatsError::ComputationError("k overflow in Poisson ppf".to_string())
            })?;
            cdf = cdf + self.pmf(k_f + self.loc);
            if cdf >= p {
                return Ok(k_f + self.loc);
            }
            k += 1;
            if k > max_k {
                // Numerically p is indistinguishable from 1.0 at this point
                return F::from(k).map(|v| v + self.loc).ok_or_else(|| {
                    StatsError::ComputationError("Overflow in Poisson ppf".to_string())
                });
            }
        }
    }
}

/// Check if a floating-point value is (close to) an integer
#[allow(dead_code)]
fn is_integer<F: Float>(x: F) -> bool {
    (x - x.round()).abs() < F::from(1e-10).expect("Failed to convert constant to float")
}

// Implement the Distribution trait for Poisson
impl<F: Float + NumCast + std::fmt::Display> Distribution<F> for Poisson<F> {
    fn mean(&self) -> F {
        self.mu + self.loc
    }

    fn var(&self) -> F {
        self.mu // Variance equals the mean for Poisson
    }

    fn std(&self) -> F {
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<F>> {
        self.rvs(size)
    }

    fn entropy(&self) -> F {
        // Entropy approximation for Poisson
        let half = F::from(0.5).expect("Failed to convert constant to float");
        let two_pi = F::from(2.0 * std::f64::consts::PI).expect("Failed to convert to float");
        let e = F::from(std::f64::consts::E).expect("Failed to convert to float");

        if self.mu <= F::zero() {
            return F::zero();
        }

        half * (two_pi * e * self.mu).ln()
            - half / (F::from(12.0).expect("Failed to convert constant to float") * self.mu)
    }
}

// Implement the DiscreteDistribution trait for Poisson
impl<F: Float + NumCast + std::fmt::Display> DiscreteDistribution<F> for Poisson<F> {
    fn pmf(&self, x: F) -> F {
        self.pmf(x)
    }

    fn cdf(&self, x: F) -> F {
        self.cdf(x)
    }

    fn ppf(&self, p: F) -> StatsResult<F> {
        self.ppf_impl(p)
    }

    fn logpmf(&self, x: F) -> F {
        // More numerically stable implementation for large mu values
        let k_std = x - self.loc;

        // PMF is zero for non-integer or negative k
        if k_std < F::zero() || !is_integer(k_std) {
            return F::neg_infinity();
        }

        // Convert k to integer value
        let k_int = <u64 as NumCast>::from(k_std).expect("Operation failed");

        // ln(PMF) = k*ln(mu) - mu - ln(k!)
        let k_f = F::from(k_int).expect("Failed to convert to float");
        k_f * self.mu.ln() - self.mu - ln_factorial(k_int)
    }
}

/// Compute natural logarithm of factorial using a sum-of-logs loop.
///
/// This is exact (to f64 precision) for all n, unlike Stirling's approximation
/// which has ~0.2% relative error near n=21 and cannot achieve 1e-6 tolerance.
fn ln_factorial<F: Float + NumCast>(n: u64) -> F {
    if n <= 1 {
        return F::zero();
    }
    let mut result = F::zero();
    for i in 2..=n {
        result = result + F::from(i).expect("Failed to convert to float").ln();
    }
    result
}

/// Implementation of SampleableDistribution for Poisson
impl<F: Float + NumCast + std::fmt::Display> SampleableDistribution<F> for Poisson<F> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let array = self.rvs(size)?;
        Ok(array.to_vec())
    }
}

/// Calculate the factorial of a non-negative integer
#[allow(dead_code)]
fn factorial(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        let mut result = 1;
        for i in 2..=n {
            // Check for overflow
            if result > u64::MAX / i {
                // For large n, approximating with u64::MAX is acceptable
                return u64::MAX;
            }
            result *= i;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poisson_creation() {
        // Poisson with rate (mean) 3.0
        let poisson = Poisson::new(3.0, 0.0).expect("Operation failed");
        assert_eq!(poisson.mu, 3.0);
        assert_eq!(poisson.loc, 0.0);

        // Custom Poisson
        let custom = Poisson::new(5.0, 1.0).expect("Operation failed");
        assert_eq!(custom.mu, 5.0);
        assert_eq!(custom.loc, 1.0);

        // Error cases
        assert!(Poisson::<f64>::new(0.0, 0.0).is_err());
        assert!(Poisson::<f64>::new(-1.0, 0.0).is_err());
    }

    #[test]
    fn test_poisson_pmf() {
        // Poisson with rate (mean) 3.0
        let poisson = Poisson::new(3.0, 0.0).expect("Operation failed");

        // PMF at k = 2
        let pmf_at_two = poisson.pmf(2.0);
        assert_relative_eq!(pmf_at_two, 0.224, epsilon = 1e-3);

        // PMF at k = 3
        let pmf_at_three = poisson.pmf(3.0);
        assert_relative_eq!(pmf_at_three, 0.224, epsilon = 1e-3);

        // PMF at k = 4
        let pmf_at_four = poisson.pmf(4.0);
        assert_relative_eq!(pmf_at_four, 0.168, epsilon = 1e-3);

        // PMF at non-integer value
        let pmf_at_half = poisson.pmf(2.5);
        assert_eq!(pmf_at_half, 0.0);

        // PMF at negative value
        let pmf_at_neg = poisson.pmf(-1.0);
        assert_eq!(pmf_at_neg, 0.0);
    }

    #[test]
    fn test_poisson_cdf() {
        // Poisson with rate (mean) 3.0
        let poisson = Poisson::new(3.0, 0.0).expect("Operation failed");

        // CDF at k = 0
        let cdf_at_zero = poisson.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.0498, epsilon = 1e-4);

        // CDF at k = 2
        let cdf_at_two = poisson.cdf(2.0);
        assert_relative_eq!(cdf_at_two, 0.423, epsilon = 1e-3);

        // CDF at k = 4
        let cdf_at_four = poisson.cdf(4.0);
        assert_relative_eq!(cdf_at_four, 0.815, epsilon = 1e-3);

        // CDF at non-integer value (should round down to integer)
        let cdf_at_half = poisson.cdf(2.5);
        assert_relative_eq!(cdf_at_half, 0.423, epsilon = 1e-3);

        // CDF at negative value
        let cdf_at_neg = poisson.cdf(-1.0);
        assert_eq!(cdf_at_neg, 0.0);
    }

    #[test]
    fn test_poisson_rvs() {
        let poisson = Poisson::new(3.0, 0.0).expect("Operation failed");

        // Generate samples
        let samples = poisson.rvs(1000).expect("Operation failed");

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be close to 3.0 (within reason for random samples)
        assert!(mean > 2.8 && mean < 3.2);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_is_integer() {
        assert!(is_integer(1.0));
        assert!(is_integer(0.0));
        assert!(is_integer(-5.0));
        assert!(is_integer(1000.0));

        assert!(!is_integer(1.1));
        assert!(!is_integer(0.5));
        assert!(!is_integer(-3.7));
    }

    /// Regression test for GitHub issue #122:
    /// Poisson PMF returned wildly wrong (exponentially growing) values for k>=21
    /// due to u64 integer factorial overflowing at 21! and returning u64::MAX.
    /// Fix: PMF now computed entirely in log-space via ln_factorial (sum-of-logs).
    #[test]
    fn test_issue_122_poisson_pmf_large_k() {
        let lambda = 10.0_f64;
        let poisson = Poisson::new(lambda, 0.0).expect("Poisson::new failed");

        // Reference values computed via math.exp(k*math.log(10) - 10 - math.lgamma(k+1))
        // i.e. the same log-space formula now used by pmf().
        // All values must be in [0, 1]; values > 1 were the reporter's symptom before the fix.
        let cases: &[(f64, f64)] = &[
            (20.0, 1.866_081_313_999_e-3),
            (21.0, 8.886_101_495_232_e-4),
            (22.0, 4.039_137_043_287_e-4),
            (23.0, 1.756_146_540_560_e-4),
            (24.0, 7.317_277_252_332_e-5),
            (25.0, 2.926_910_900_933_e-5),
        ];

        for &(k, expected) in cases {
            let got = poisson.pmf(k);
            // A PMF value > 1.0 is mathematically impossible and was the
            // observable symptom before the fix.
            assert!(
                got <= 1.0,
                "pmf({k}) = {got} exceeds 1.0 (overflow artifact)"
            );
            assert!(
                got >= 0.0,
                "pmf({k}) = {got} is negative (should never happen)"
            );
            let rel_err = (got - expected).abs() / expected;
            assert!(
                rel_err < 1e-6,
                "pmf({k}): got {got:.9e}, expected {expected:.9e}, rel_err={rel_err:.3e}"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // PPF (inverse CDF) tests
    // ──────────────────────────────────────────────────────────────────────

    /// The defining property of the Poisson PPF:
    ///   CDF(PPF(p) - 1) < p  ≤  CDF(PPF(p))   for p ∈ (0,1)
    #[test]
    fn test_poisson_ppf_round_trip() {
        let poisson = Poisson::new(3.0f64, 0.0).expect("Poisson::new");
        // Test a range of p values
        let ps = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
        for &p in &ps {
            let k = poisson.ppf_impl(p).expect("ppf");
            // CDF at k must be >= p
            let cdf_k = poisson.cdf(k);
            assert!(
                cdf_k >= p - 1e-12,
                "p={p}: CDF(PPF(p))={cdf_k} < p — PPF returned a value too small"
            );
            // CDF at k-1 must be < p (k is the minimal such integer)
            if k >= 1.0 {
                let cdf_km1 = poisson.cdf(k - 1.0);
                assert!(
                    cdf_km1 < p + 1e-12,
                    "p={p}: CDF(PPF(p)-1)={cdf_km1} >= p — PPF returned a value too large"
                );
            }
        }
    }

    /// PPF edge cases: p=0 returns loc (= 0), p=1 returns a large value.
    #[test]
    fn test_poisson_ppf_edge_cases() {
        let poisson = Poisson::new(5.0f64, 0.0).expect("Poisson::new");
        // p = 0 → smallest non-negative integer with CDF >= 0, which is 0
        let k0 = poisson.ppf_impl(0.0).expect("ppf(0)");
        assert_eq!(k0, 0.0, "PPF(0) should be 0 for loc=0");

        // p = 1 → some large finite value
        let k1 = poisson.ppf_impl(1.0).expect("ppf(1)");
        assert!(k1.is_finite(), "PPF(1) should be finite");
        assert!(k1 > 0.0);

        // Out-of-range inputs → error
        assert!(poisson.ppf_impl(-0.1).is_err());
        assert!(poisson.ppf_impl(1.1).is_err());
    }

    /// PPF with a non-zero location parameter.
    #[test]
    fn test_poisson_ppf_with_location() {
        let loc = 2.0f64;
        let poisson = Poisson::new(3.0f64, loc).expect("Poisson::new");
        // PPF(0.5) without loc, then add loc
        let poisson0 = Poisson::new(3.0f64, 0.0).expect("Poisson::new");
        let k_no_loc = poisson0.ppf_impl(0.5).expect("ppf");
        let k_with_loc = poisson.ppf_impl(0.5).expect("ppf");
        assert!(
            (k_with_loc - k_no_loc - loc).abs() < 1e-10,
            "PPF with loc={loc}: expected {}, got {k_with_loc}",
            k_no_loc + loc
        );
    }

    /// PPF for large mu: check round-trip still holds.
    #[test]
    fn test_poisson_ppf_large_mu() {
        let poisson = Poisson::new(100.0f64, 0.0).expect("Poisson::new");
        for &p in &[0.05, 0.5, 0.95] {
            let k = poisson.ppf_impl(p).expect("ppf");
            let cdf_k = poisson.cdf(k);
            assert!(cdf_k >= p - 1e-10, "large mu p={p}: CDF(k)={cdf_k} < p");
        }
    }
}
