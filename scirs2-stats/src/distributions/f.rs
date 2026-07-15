//! F distribution functions
//!
//! This module provides functionality for the F distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, FisherF as RandFisherF};
use std::f64::consts::PI;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<T: Float + NumCast>(value: f64) -> T {
    T::from(value).expect("Failed to convert constant to target float type")
}

/// F distribution structure
pub struct F<T: Float> {
    /// Degrees of freedom for numerator
    pub dfn: T,
    /// Degrees of freedom for denominator
    pub dfd: T,
    /// Location parameter
    pub loc: T,
    /// Scale parameter
    pub scale: T,
    /// Random number generator for this distribution
    rand_distr: RandFisherF<f64>,
}

impl<T: Float + NumCast> F<T> {
    /// Create a new F distribution with given degrees of freedom, location, and scale
    ///
    /// # Arguments
    ///
    /// * `dfn` - Numerator degrees of freedom (> 0)
    /// * `dfd` - Denominator degrees of freedom (> 0)
    /// * `loc` - Location parameter (default: 0)
    /// * `scale` - Scale parameter (default: 1, must be > 0)
    ///
    /// # Returns
    ///
    /// * A new F distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::f::F;
    ///
    /// // F distribution with 2 and 10 degrees of freedom
    /// let f_dist = F::new(2.0f64, 10.0, 0.0, 1.0).expect("Test/example failed");
    /// ```
    pub fn new(dfn: T, dfd: T, loc: T, scale: T) -> StatsResult<Self> {
        if dfn <= T::zero() {
            return Err(StatsError::DomainError(
                "Numerator degrees of freedom must be positive".to_string(),
            ));
        }

        if dfd <= T::zero() {
            return Err(StatsError::DomainError(
                "Denominator degrees of freedom must be positive".to_string(),
            ));
        }

        if scale <= T::zero() {
            return Err(StatsError::DomainError(
                "Scale parameter must be positive".to_string(),
            ));
        }

        // Convert to f64 for rand_distr
        let dfn_f64 = <f64 as NumCast>::from(dfn).expect("Test/example failed");
        let dfd_f64 = <f64 as NumCast>::from(dfd).expect("Test/example failed");

        match RandFisherF::new(dfn_f64, dfd_f64) {
            Ok(rand_distr) => Ok(F {
                dfn,
                dfd,
                loc,
                scale,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create F distribution".to_string(),
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
    /// use scirs2_stats::distributions::f::F;
    ///
    /// let f_dist = F::new(2.0f64, 10.0, 0.0, 1.0).expect("Test/example failed");
    /// let pdf_at_one = f_dist.pdf(1.0);
    /// assert!((pdf_at_one - 0.335).abs() < 1e-3);
    /// ```
    pub fn pdf(&self, x: T) -> T {
        // Standardize the variable
        let x_std = (x - self.loc) / self.scale;

        // If x is not positive, PDF is zero (F is only defined for x > 0)
        if x_std <= T::zero() {
            return T::zero();
        }

        // Calculate PDF using log-space computation for numerical stability
        // PDF(x) = sqrt( (d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2) ) / (x * B(d1/2, d2/2))
        // Equivalently in log-space:
        // ln(pdf) = (d1/2)*ln(d1) + (d2/2)*ln(d2) + (d1/2-1)*ln(x)
        //         - ((d1+d2)/2)*ln(d1*x + d2) - ln(B(d1/2, d2/2))
        let two = const_f64::<T>(2.0);
        let d1 = self.dfn;
        let d2 = self.dfd;
        let d1h = d1 / two;
        let d2h = d2 / two;

        let ln_pdf = d1h * d1.ln() + d2h * d2.ln() + (d1h - T::one()) * x_std.ln()
            - (d1h + d2h) * (d1 * x_std + d2).ln()
            - ln_beta(d1h, d2h);

        ln_pdf.exp() / self.scale
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
    /// use scirs2_stats::distributions::f::F;
    ///
    /// let f_dist = F::new(2.0f64, 10.0, 0.0, 1.0).expect("Test/example failed");
    /// let cdf_at_one = f_dist.cdf(1.0);
    /// assert!((cdf_at_one - 0.5984).abs() < 0.01);
    /// ```
    pub fn cdf(&self, x: T) -> T {
        // Standardize the variable
        let x_std = (x - self.loc) / self.scale;

        // If x is not positive, CDF is zero (F is only defined for x > 0)
        if x_std <= T::zero() {
            return T::zero();
        }

        // The CDF of the F distribution is related to the incomplete beta function
        // CDF(x) = I_(dfn*x/(dfn*x + dfd))(dfn/2, dfd/2)
        // where I_x(a,b) is the regularized incomplete beta function

        let two = const_f64::<T>(2.0);

        let dfn_half = self.dfn / two;
        let dfd_half = self.dfd / two;

        // Calculate the argument for the incomplete beta function
        let z = self.dfn * x_std / (self.dfn * x_std + self.dfd);

        // Calculate the incomplete beta function
        regularized_beta(z, dfn_half, dfd_half)
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
    /// use scirs2_stats::distributions::f::F;
    ///
    /// let f_dist = F::new(2.0f64, 10.0, 0.0, 1.0).expect("Test/example failed");
    /// let samples = f_dist.rvs(1000).expect("Test/example failed");
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<T>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // Generate a standard F random variable
            let std_sample = self.rand_distr.sample(&mut rng);

            // Scale and shift according to loc and scale parameters
            let sample =
                T::from(std_sample).expect("Failed to convert to float") * self.scale + self.loc;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Percent-point function (inverse CDF / quantile function) at a given probability
    ///
    /// Returns the value `x` such that `CDF(x) = p`, by inverting the
    /// regularized-incomplete-beta relation used by [`F::cdf`] via bisection
    /// on the incomplete-beta argument `z`, following the same pattern as
    /// `Beta::ppf`.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value (between 0 and 1)
    ///
    /// # Returns
    ///
    /// * The value `x` such that `CDF(x) = p`
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::f::F;
    ///
    /// let f_dist = F::new(2.0f64, 10.0, 0.0, 1.0).expect("Test/example failed");
    /// let x = f_dist.ppf(0.5).expect("Test/example failed");
    /// let p_roundtrip = f_dist.cdf(x);
    /// assert!((p_roundtrip - 0.5).abs() < 1e-6);
    /// ```
    pub fn ppf(&self, p: T) -> StatsResult<T> {
        if p < T::zero() || p > T::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }

        // p = 0 maps to the (finite) lower boundary of the support: loc.
        if p == T::zero() {
            return Ok(self.loc);
        }

        // p = 1 is a genuine singularity: CDF(x) -> 1 only in the limit
        // x -> infinity (equivalently z -> 1 in the incomplete-beta argument
        // below), since the F distribution's support is unbounded above.
        // Unlike Beta (bounded support, where p=1 maps exactly to
        // `loc + scale`), there is no finite exact answer here. Mirror
        // `Poisson::ppf_impl`'s convention for unbounded-support
        // distributions and return a large-but-finite sentinel rather than
        // literal infinity.
        if p == T::one() {
            let dfn_f64 = self.dfn.to_f64().unwrap_or(1.0).max(1e-3);
            let dfd_f64 = self.dfd.to_f64().unwrap_or(1.0).max(1e-3);
            let huge_x = 1.0e15_f64 * (dfd_f64 / dfn_f64).max(1.0);
            return T::from(huge_x)
                .map(|v| self.loc + v * self.scale)
                .ok_or_else(|| {
                    StatsError::ComputationError("Overflow in F ppf upper bound".to_string())
                });
        }

        let two = const_f64::<T>(2.0);
        let dfn_half = self.dfn / two;
        let dfd_half = self.dfd / two;

        // Bisect on z in (0,1) against the regularized incomplete beta
        // function I_z(dfn/2, dfd/2) = p (same pattern as `Beta::ppf`), then
        // invert z = dfn*x_std/(dfn*x_std+dfd) to recover x_std.
        let eps = const_f64::<T>(1e-12);
        let mut lo = const_f64::<T>(1e-15);
        let mut hi = T::one() - const_f64::<T>(1e-15);

        for _ in 0..100 {
            let mid = (lo + hi) * const_f64::<T>(0.5);
            let cdf_mid = regularized_beta(mid, dfn_half, dfd_half);
            if (cdf_mid - p).abs() < eps {
                lo = mid;
                hi = mid;
                break;
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

        let z = (lo + hi) * const_f64::<T>(0.5);
        let x_std = self.dfd * z / (self.dfn * (T::one() - z));

        Ok(self.loc + x_std * self.scale)
    }
}

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
#[allow(dead_code)]
fn beta_function<T: Float>(a: T, b: T) -> T {
    gamma_function(a) * gamma_function(b) / gamma_function(a + b)
}

/// Regularized incomplete beta function I_x(a,b) using Lentz's continued fraction
/// from Numerical Recipes / DLMF 8.17.22.
#[allow(dead_code)]
fn regularized_beta<T: Float>(x: T, a: T, b: T) -> T {
    if x == T::zero() {
        return T::zero();
    }
    if x == T::one() {
        return T::one();
    }

    let one = T::one();
    let two = const_f64::<T>(2.0);
    let epsilon = const_f64::<T>(1e-14);
    let tiny = const_f64::<T>(1e-30);
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
        a_cf * x_cf.ln() + b_cf * (one - x_cf).ln() - a_cf.ln() - ln_beta(a_cf, b_cf);
    let prefactor = ln_prefactor.exp();

    // Lentz's algorithm for the continued fraction
    // CF = 1/(1 + d1/(1 + d2/(1 + ...)))
    // where d_{2m+1} = -(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
    //       d_{2m}   =  m(b-m) x       / ((a+2m-1)(a+2m))
    let mut f = one;
    let mut c = one;
    let mut d = one - (a_cf + b_cf) * x_cf / (a_cf + one);
    if d.abs() < tiny {
        d = tiny;
    }
    d = one / d;
    f = d;

    for m in 1..=max_iterations {
        let m_f = T::from(m as f64).expect("Failed to convert to float");

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
fn ln_beta<T: Float>(a: T, b: T) -> T {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation for the log-gamma function
#[allow(dead_code)]
fn ln_gamma<T: Float>(x: T) -> T {
    let one = T::one();
    let half = const_f64::<T>(0.5);
    let pi = const_f64::<T>(std::f64::consts::PI);

    // For x < 0.5, use the reflection formula
    if x < half {
        let sin_val = (pi * x).sin();
        if sin_val == T::zero() {
            return T::infinity();
        }
        return pi.ln() - sin_val.abs().ln() - ln_gamma(one - x);
    }

    let g = const_f64::<T>(7.0); // g parameter for Lanczos
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
    let mut sum = const_f64::<T>(coefficients[0]);
    for (i, &c) in coefficients.iter().enumerate().skip(1) {
        sum = sum + const_f64::<T>(c) / (xx + T::from(i as f64).expect("conv"));
    }

    let t = xx + g + half;
    half * (const_f64::<T>(2.0) * pi).ln() + (xx + half) * t.ln() - t + sum.ln()
}

/// Approximation of the gamma function for floating point types
#[allow(dead_code)]
fn gamma_function<T: Float>(x: T) -> T {
    if x == T::one() {
        return T::one();
    }

    if x == const_f64::<T>(0.5) {
        return T::from(PI).expect("Failed to convert to float").sqrt();
    }

    // For integers and half-integers, use recurrence relation
    if x > T::one() {
        return (x - T::one()) * gamma_function(x - T::one());
    }

    // Use Lanczos approximation for other values
    let p = [
        const_f64::<T>(676.5203681218851),
        const_f64::<T>(-1259.1392167224028),
        T::from(771.323_428_777_653_1).expect("Failed to convert to float"),
        T::from(-176.615_029_162_140_6).expect("Failed to convert to float"),
        const_f64::<T>(12.507343278686905),
        const_f64::<T>(-0.13857109526572012),
        T::from(9.984_369_578_019_572e-6).expect("Failed to convert to float"),
        const_f64::<T>(1.5056327351493116e-7),
    ];

    let x_adj = x - T::one();
    let t = x_adj + const_f64::<T>(7.5);

    let mut sum = T::zero();
    for (i, &coef) in p.iter().enumerate() {
        sum = sum + coef / (x_adj + T::from(i + 1).expect("Failed to convert to float"));
    }

    let pi = T::from(PI).expect("Failed to convert to float");
    let sqrt_2pi = (const_f64::<T>(2.0) * pi).sqrt();

    sqrt_2pi * sum * t.powf(x_adj + const_f64::<T>(0.5)) * (-t).exp()
}

/// Implementation of SampleableDistribution for F
impl<T: Float + NumCast> SampleableDistribution<T> for F<T> {
    fn rvs(&self, size: usize) -> StatsResult<Vec<T>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_f_creation() {
        // F with 2,10 degrees of freedom
        let f_dist = F::new(2.0, 10.0, 0.0, 1.0).expect("Test/example failed");
        assert_eq!(f_dist.dfn, 2.0);
        assert_eq!(f_dist.dfd, 10.0);
        assert_eq!(f_dist.loc, 0.0);
        assert_eq!(f_dist.scale, 1.0);

        // Custom F
        let custom = F::new(5.0, 20.0, 1.0, 2.0).expect("Test/example failed");
        assert_eq!(custom.dfn, 5.0);
        assert_eq!(custom.dfd, 20.0);
        assert_eq!(custom.loc, 1.0);
        assert_eq!(custom.scale, 2.0);

        // Error cases
        assert!(F::<f64>::new(0.0, 10.0, 0.0, 1.0).is_err());
        assert!(F::<f64>::new(-1.0, 10.0, 0.0, 1.0).is_err());
        assert!(F::<f64>::new(2.0, 0.0, 0.0, 1.0).is_err());
        assert!(F::<f64>::new(2.0, -1.0, 0.0, 1.0).is_err());
        assert!(F::<f64>::new(2.0, 10.0, 0.0, 0.0).is_err());
        assert!(F::<f64>::new(2.0, 10.0, 0.0, -1.0).is_err());
    }

    #[test]
    fn test_f_pdf() {
        // F with 2,10 degrees of freedom
        let f_dist = F::new(2.0, 10.0, 0.0, 1.0).expect("Test/example failed");

        // PDF at x = 0
        let pdf_at_zero = f_dist.pdf(0.0);
        assert_eq!(pdf_at_zero, 0.0);

        // scipy: f.pdf(1, dfn=2, dfd=10) ≈ 0.334898
        let pdf_at_one = f_dist.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.334898, epsilon = 1e-3);

        // scipy: f.pdf(2, dfn=2, dfd=10) ≈ 0.132686
        let pdf_at_two = f_dist.pdf(2.0);
        assert_relative_eq!(pdf_at_two, 0.132686, epsilon = 1e-3);

        // F with 5,20 degrees of freedom
        let f5_20 = F::new(5.0, 20.0, 0.0, 1.0).expect("Test/example failed");

        // scipy: f.pdf(1, dfn=5, dfd=20) ≈ 0.5449
        let pdf_at_one = f5_20.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.5449, epsilon = 1e-2);
    }

    #[test]
    fn test_f_cdf() {
        // F with 1,10 degrees of freedom
        let f1_10 = F::new(1.0, 10.0, 0.0, 1.0).expect("Test/example failed");

        // CDF at x = 0
        let cdf_at_zero = f1_10.cdf(0.0);
        assert_eq!(cdf_at_zero, 0.0);

        // scipy: f.cdf(1, dfn=1, dfd=10) ≈ 0.6591
        let cdf_at_one = f1_10.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.6591, epsilon = 1e-3);

        // F with 2,10 degrees of freedom
        let f2_10 = F::new(2.0, 10.0, 0.0, 1.0).expect("Test/example failed");

        // scipy: f.cdf(1, dfn=2, dfd=10) ≈ 0.59812
        let cdf_at_one = f2_10.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.59812, epsilon = 1e-3);

        // F with 5,20 degrees of freedom
        let f5_20 = F::new(5.0, 20.0, 0.0, 1.0).expect("Test/example failed");

        // scipy: f.cdf(1, dfn=5, dfd=20) ≈ 0.5560
        let cdf_at_one = f5_20.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.5560, epsilon = 1e-2);
    }

    #[test]
    fn test_f_rvs() {
        let f_dist = F::new(2.0, 10.0, 0.0, 1.0).expect("Test/example failed");

        // Generate samples
        let samples = f_dist.rvs(1000).expect("Test/example failed");

        // Check the number of samples
        assert_eq!(samples.len(), 1000);

        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Mean of F(2,10) should be close to 10/(10-2) = 1.25, within reasonable bounds for random samples
        assert!(mean > 0.9 && mean < 1.6);
    }

    #[test]
    fn test_beta_function() {
        // Check known values
        assert_relative_eq!(beta_function(1.0, 1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta_function(2.0, 3.0), 1.0 / 12.0, epsilon = 1e-10);
        assert_relative_eq!(beta_function(0.5, 0.5), PI, epsilon = 1e-10);
    }

    // ──────────────────────────────────────────────────────────────────────
    // PPF (inverse CDF) tests
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_f_ppf_round_trip() {
        // Round-trip ppf(cdf(x)) ≈ x across several (dfn, dfd) pairs and x values.
        let df_pairs: &[(f64, f64)] = &[(1.0, 1.0), (5.0, 10.0), (20.0, 30.0)];
        let xs: &[f64] = &[0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0];

        for &(dfn, dfd) in df_pairs {
            let f_dist = F::new(dfn, dfd, 0.0, 1.0).expect("F::new failed");
            for &x in xs {
                let p = f_dist.cdf(x);
                // Skip probabilities too close to the boundaries, where the
                // round trip is inherently lossy (extreme tails).
                if !(1e-6..=1.0 - 1e-6).contains(&p) {
                    continue;
                }
                let x_roundtrip = f_dist.ppf(p).expect("ppf failed");
                assert_relative_eq!(x_roundtrip, x, epsilon = 1e-6, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn test_f_ppf_with_loc_scale() {
        // Round trip still holds once loc/scale are non-trivial.
        let f_dist = F::new(3.0, 8.0, 2.0, 1.5).expect("F::new failed");
        for &x in &[2.5, 3.0, 5.0, 10.0] {
            let p = f_dist.cdf(x);
            if !(1e-6..=1.0 - 1e-6).contains(&p) {
                continue;
            }
            let x_roundtrip = f_dist.ppf(p).expect("ppf failed");
            assert_relative_eq!(x_roundtrip, x, epsilon = 1e-5, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_f_ppf_matches_known_cdf() {
        // scipy: f.cdf(1, dfn=2, dfd=10) ≈ 0.59812  =>  f.ppf(0.59812, dfn=2, dfd=10) ≈ 1
        let f2_10 = F::new(2.0, 10.0, 0.0, 1.0).expect("F::new failed");
        let x = f2_10.ppf(0.59812).expect("ppf failed");
        assert_relative_eq!(x, 1.0, epsilon = 1e-3);

        // scipy: f.cdf(1, dfn=5, dfd=20) ≈ 0.5560  =>  f.ppf(0.5560, dfn=5, dfd=20) ≈ 1
        let f5_20 = F::new(5.0, 20.0, 0.0, 1.0).expect("F::new failed");
        let x2 = f5_20.ppf(0.5560).expect("ppf failed");
        assert_relative_eq!(x2, 1.0, epsilon = 1e-2);

        // scipy: f.cdf(1, dfn=1, dfd=10) ≈ 0.6591  =>  f.ppf(0.6591, dfn=1, dfd=10) ≈ 1
        let f1_10 = F::new(1.0, 10.0, 0.0, 1.0).expect("F::new failed");
        let x3 = f1_10.ppf(0.6591).expect("ppf failed");
        assert_relative_eq!(x3, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_f_ppf_edge_cases() {
        let f_dist = F::new(2.0, 10.0, 0.0, 1.0).expect("F::new failed");

        // p = 0 -> loc boundary
        let x0 = f_dist.ppf(0.0).expect("ppf(0) failed");
        assert_eq!(x0, 0.0);

        // p = 1 -> large-but-finite sentinel value (unbounded support, no
        // finite exact answer; mirrors Poisson::ppf_impl's convention).
        let x1 = f_dist.ppf(1.0).expect("ppf(1) failed");
        assert!(x1.is_finite());
        assert!(x1 > 1.0e6);

        // Out-of-range probabilities are rejected.
        assert!(f_dist.ppf(-0.1).is_err());
        assert!(f_dist.ppf(1.1).is_err());
    }

    #[test]
    fn test_f_ppf_monotonic() {
        // PPF must be monotonically non-decreasing in p.
        let f_dist = F::new(4.0, 15.0, 0.0, 1.0).expect("F::new failed");
        let ps = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
        let mut prev = f_dist.ppf(ps[0]).expect("ppf failed");
        for &p in &ps[1..] {
            let x = f_dist.ppf(p).expect("ppf failed");
            assert!(
                x >= prev,
                "ppf({p}) = {x} should be >= ppf of smaller p = {prev}"
            );
            prev = x;
        }
    }
}
