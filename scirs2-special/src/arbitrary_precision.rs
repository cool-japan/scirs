//! Arbitrary precision computation support for special functions
//!
//! This module provides arbitrary precision implementations of special functions
//! using the GNU MPFR library through the rug crate. This allows for computations
//! with user-specified precision beyond the limitations of f64.

#![allow(dead_code)]

use crate::error::{SpecialError, SpecialResult};
use rug::{float::Constant, ops::Pow, Complex, Float};

/// Default precision in bits for arbitrary precision computations
pub const DEFAULT_PRECISION: u32 = 256;

/// Maximum supported precision in bits
pub const MAX_PRECISION: u32 = 4096;

/// Precision context for arbitrary precision computations
#[derive(Debug, Clone)]
pub struct PrecisionContext {
    /// Precision in bits
    precision: u32,
    /// Rounding mode
    rounding: rug::float::Round,
}

impl Default for PrecisionContext {
    fn default() -> Self {
        Self {
            precision: DEFAULT_PRECISION,
            rounding: rug::float::Round::Nearest,
        }
    }
}

impl PrecisionContext {
    /// Create a new precision context with specified precision in bits
    pub fn new(precision: u32) -> SpecialResult<Self> {
        if precision == 0 || precision > MAX_PRECISION {
            return Err(SpecialError::DomainError(format!(
                "Precision must be between 1 and {} bits",
                MAX_PRECISION
            )));
        }
        Ok(Self {
            precision,
            rounding: rug::float::Round::Nearest,
        })
    }

    /// Set the rounding mode
    pub fn with_rounding(mut self, rounding: rug::float::Round) -> Self {
        self.rounding = rounding;
        self
    }

    /// Get the precision in bits
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Get the rounding mode
    pub fn rounding(&self) -> rug::float::Round {
        self.rounding
    }

    /// Create a Float with the context's precision
    pub fn float(&self, value: f64) -> Float {
        Float::with_val(self.precision, value)
    }

    /// Create a Complex with the context's precision
    pub fn complex(&self, real: f64, imag: f64) -> Complex {
        Complex::with_val(self.precision, (real, imag))
    }

    /// Create pi with the context's precision
    pub fn pi(&self) -> Float {
        Float::with_val(self.precision, Constant::Pi)
    }

    /// Create e (Euler's number) with the context's precision
    pub fn e(&self) -> Float {
        Float::with_val(self.precision, 1).exp()
    }

    /// Create ln(2) with the context's precision
    pub fn ln2(&self) -> Float {
        Float::with_val(self.precision, Constant::Log2)
    }

    /// Create Euler's gamma constant with the context's precision
    pub fn euler_gamma(&self) -> Float {
        Float::with_val(self.precision, Constant::Euler)
    }

    /// Create Catalan's constant with the context's precision
    pub fn catalan(&self) -> Float {
        Float::with_val(self.precision, Constant::Catalan)
    }
}

/// Arbitrary precision Gamma function
pub mod gamma {
    use super::*;

    /// Compute the Gamma function with arbitrary precision
    pub fn gamma_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let x_mp = ctx.float(x);
        gamma_mp(&x_mp, ctx)
    }

    /// Compute the Gamma function for arbitrary precision input
    pub fn gamma_mp(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if x.is_zero() || (x.is_finite() && *x < 0.0 && x.is_integer()) {
            return Err(SpecialError::DomainError(
                "Gamma function undefined at non-positive integers".to_string(),
            ));
        }

        // Use Stirling's approximation for large x
        if *x > 20.0 {
            stirling_gamma(x, ctx)
        } else if *x > 0.0 {
            // Use Lanczos approximation for moderate positive x
            lanczos_gamma(x, ctx)
        } else {
            // Use reflection formula for negative x
            reflection_gamma(x, ctx)
        }
    }

    /// Stirling's approximation for Gamma function
    fn stirling_gamma(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let two_pi = ctx.pi() * Float::with_val(ctx.precision, 2.0);
        let sqrt_2pi = two_pi.sqrt();
        let e = ctx.e();

        // Γ(x) ≈ √(2π/x) * (x/e)^x * (1 + 1/(12x) + ...)
        let term1 = sqrt_2pi / x.clone().sqrt();
        let term2 = (x.clone() / e).pow(x);

        // Add correction terms
        let mut correction = ctx.float(1.0);
        let x2 = Float::with_val(ctx.precision, x.clone() * x);
        let x3 = Float::with_val(ctx.precision, &x2 * x);
        let x4 = Float::with_val(ctx.precision, &x2 * &x2);

        correction += ctx.float(1.0) / (ctx.float(12.0) * x);
        correction += ctx.float(1.0) / (ctx.float(288.0) * &x2);
        let denom1 = Float::with_val(ctx.precision, ctx.float(51840.0) * &x3);
        correction -= ctx.float(139.0) / denom1;
        let denom2 = Float::with_val(ctx.precision, ctx.float(2488320.0) * &x4);
        correction -= ctx.float(571.0) / denom2;

        Ok(term1 * term2 * correction)
    }

    /// Lanczos approximation for Gamma function
    fn lanczos_gamma(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        // Lanczos coefficients for g=7
        const LANCZOS_G: f64 = 7.0;
        const LANCZOS_COEFFS: &[f64] = &[
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

        let g = ctx.float(LANCZOS_G);
        let sqrt_2pi = (ctx.pi() * ctx.float(2.0)).sqrt();

        let mut ag = ctx.float(LANCZOS_COEFFS[0]);
        for i in 1..LANCZOS_COEFFS.len() {
            ag += ctx.float(LANCZOS_COEFFS[i]) / (x.clone() + i as f64);
        }

        let tmp = x.clone() + &g + ctx.float(0.5);
        let result = sqrt_2pi * ag * tmp.clone().pow(x.clone() + 0.5) * (-tmp).exp();

        Ok(result / x)
    }

    /// Reflection formula for negative x
    fn reflection_gamma(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let pi = ctx.pi();
        let sin_pi_x = (pi.clone() * x).sin();

        if sin_pi_x.is_zero() {
            return Err(SpecialError::DomainError(
                "Gamma function has poles at negative integers".to_string(),
            ));
        }

        let pos_gamma = gamma_mp(&(ctx.float(1.0) - x), ctx)?;
        Ok(pi / (sin_pi_x * pos_gamma))
    }

    /// Compute log(Gamma(x)) with arbitrary precision
    pub fn log_gamma_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let x_mp = ctx.float(x);
        log_gamma_mp(&x_mp, ctx)
    }

    /// Compute log(Gamma(x)) for arbitrary precision input
    pub fn log_gamma_mp(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if x.is_zero() || (x.is_finite() && *x < 0.0) {
            return Err(SpecialError::DomainError(
                "log_gamma undefined for non-positive values".to_string(),
            ));
        }

        if *x > 10.0 {
            // Use Stirling's approximation for large x
            stirling_log_gamma(x, ctx)
        } else {
            // For small x, compute gamma first then take log
            let gamma_x = gamma_mp(x, ctx)?;
            Ok(gamma_x.ln())
        }
    }

    /// Stirling's approximation for log(Gamma(x))
    fn stirling_log_gamma(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let two_pi = ctx.pi() * Float::with_val(ctx.precision, 2.0);
        let ln_2pi = two_pi.ln();

        // log Γ(x) ≈ (x - 1/2) log x - x + log(2π)/2 + 1/(12x) - ...
        let mut result = (x.clone() - 0.5) * x.clone().ln() - x.clone() + ln_2pi / 2.0;

        // Add correction terms
        let x2 = Float::with_val(ctx.precision, x.clone() * x);
        let x3 = Float::with_val(ctx.precision, &x2 * x);
        let x5 = Float::with_val(ctx.precision, &x3 * &x2);
        let x7 = Float::with_val(ctx.precision, &x5 * &x2);

        result += ctx.float(1.0) / (ctx.float(12.0) * x);
        let denom3 = Float::with_val(ctx.precision, ctx.float(360.0) * &x3);
        result -= ctx.float(1.0) / denom3;
        let denom4 = ctx.float(1260.0) * &x5;
        result += ctx.float(1.0) / denom4;
        let denom5 = ctx.float(1680.0) * &x7;
        result -= ctx.float(1.0) / denom5;

        Ok(result)
    }
}

/// Arbitrary precision Bessel functions
pub mod bessel {
    use super::*;

    /// Compute Bessel J_n(x) with arbitrary precision
    pub fn bessel_j_ap(n: i32, x: f64, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let x_mp = ctx.float(x);
        bessel_j_mp(n, &x_mp, ctx)
    }

    /// Compute Bessel J_n(x) for arbitrary precision input
    pub fn bessel_j_mp(n: i32, x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if x.is_zero() {
            return Ok(if n == 0 {
                ctx.float(1.0)
            } else {
                ctx.float(0.0)
            });
        }

        // For small x, use power series
        if x.clone().abs() < 10.0 {
            bessel_j_series(n, x, ctx)
        } else {
            // For large x, use asymptotic expansion
            bessel_j_asymptotic(n, x, ctx)
        }
    }

    /// Power series for Bessel J_n(x)
    fn bessel_j_series(n: i32, x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let mut sum = ctx.float(0.0);
        let x_half = Float::with_val(ctx.precision(), x.clone() / 2.0);
        let x2_quarter = Float::with_val(ctx.precision(), &x_half * &x_half);

        let mut term = x_half.pow(n) / factorial_mp(n.abs() as u32, ctx);
        let sign = if n < 0 && n % 2 != 0 { -1.0 } else { 1.0 };
        term *= sign;

        sum += &term;

        // Add terms until convergence
        for k in 1..200 {
            let divisor = Float::with_val(ctx.precision(), k as f64 * (k as f64 + n.abs() as f64));
            let neg_x2_quarter = Float::with_val(ctx.precision(), -&x2_quarter);
            term *= neg_x2_quarter / divisor;
            sum += &term;

            if term.clone().abs()
                < sum.clone().abs()
                    * Float::with_val(ctx.precision(), 10.0).pow(-(ctx.precision() as i32) / 10)
            {
                break;
            }
        }

        Ok(sum)
    }

    /// Asymptotic expansion for Bessel J_n(x)
    fn bessel_j_asymptotic(n: i32, x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let pi = ctx.pi();
        let pi_x = Float::with_val(ctx.precision(), &pi * x);
        let sqrt_2_pi_x = (ctx.float(2.0) / pi_x).sqrt();

        let phase_coefficient = Float::with_val(ctx.precision(), n as f64 + 0.5);
        let phase_pi_mult = Float::with_val(ctx.precision(), &phase_coefficient * &pi);
        let phase_offset = Float::with_val(ctx.precision(), phase_pi_mult / 2.0);
        let phase = x.clone() - phase_offset;
        let cos_phase = phase.cos();

        // Add asymptotic correction terms
        let mut correction = ctx.float(1.0);
        let n2 = (n * n) as f64;
        let x2 = x.clone() * x;

        let x_mult = Float::with_val(ctx.precision(), 8.0 * x);
        correction -= (4.0 * n2 - 1.0) / x_mult;
        let x2_mult = Float::with_val(ctx.precision(), 128.0 * &x2);
        correction += (4.0 * n2 - 1.0) * (4.0 * n2 - 9.0) / x2_mult;

        Ok(sqrt_2_pi_x * cos_phase * correction)
    }

    /// Compute Bessel Y_n(x) with arbitrary precision
    pub fn bessel_y_ap(n: i32, x: f64, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let x_mp = ctx.float(x);
        bessel_y_mp(n, &x_mp, ctx)
    }

    /// Compute Bessel Y_n(x) for arbitrary precision input
    pub fn bessel_y_mp(n: i32, x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if *x <= 0.0 {
            return Err(SpecialError::DomainError(
                "Bessel Y function undefined for non-positive arguments".to_string(),
            ));
        }

        // For large x, use asymptotic expansion
        if *x > 10.0 {
            bessel_y_asymptotic(n, x, ctx)
        } else {
            // Use relation with J_n
            bessel_y_relation(n, x, ctx)
        }
    }

    /// Compute Y_n using relation with J_n
    fn bessel_y_relation(n: i32, x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let pi = ctx.pi();

        if n >= 0 {
            let jn = bessel_j_mp(n, x, ctx)?;
            let jn_neg = bessel_j_mp(-n, x, ctx)?;
            let cos_n_pi = if n % 2 == 0 { 1.0 } else { -1.0 };

            let n_pi = Float::with_val(ctx.precision(), n as f64) * &pi;
            Ok((jn * cos_n_pi - jn_neg) / n_pi.sin())
        } else {
            // Y_{-n}(x) = (-1)^n Y_n(x)
            let yn_pos = bessel_y_mp(-n, x, ctx)?;
            Ok(if n % 2 == 0 { yn_pos } else { -yn_pos })
        }
    }

    /// Asymptotic expansion for Bessel Y_n(x)
    fn bessel_y_asymptotic(n: i32, x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let pi = ctx.pi();
        let pi_x = Float::with_val(ctx.precision(), &pi * x);
        let sqrt_2_pi_x = (ctx.float(2.0) / pi_x).sqrt();

        let phase_coefficient = Float::with_val(ctx.precision(), n as f64 + 0.5);
        let phase_pi_mult = Float::with_val(ctx.precision(), &phase_coefficient * &pi);
        let phase_offset = Float::with_val(ctx.precision(), phase_pi_mult / 2.0);
        let phase = x.clone() - phase_offset;
        let sin_phase = phase.sin();

        // Add asymptotic correction terms
        let mut correction = ctx.float(1.0);
        let n2 = (n * n) as f64;
        let x2 = x.clone() * x;

        let x_mult = Float::with_val(ctx.precision(), 8.0 * x);
        correction -= (4.0 * n2 - 1.0) / x_mult;
        let x2_mult = Float::with_val(ctx.precision(), 128.0 * &x2);
        correction += (4.0 * n2 - 1.0) * (4.0 * n2 - 9.0) / x2_mult;

        Ok(sqrt_2_pi_x * sin_phase * correction)
    }
}

/// Arbitrary precision error functions
pub mod error_function {
    use super::*;

    /// Compute erf(x) with arbitrary precision
    pub fn erf_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let x_mp = ctx.float(x);
        erf_mp(&x_mp, ctx)
    }

    /// Compute erf(x) for arbitrary precision input
    pub fn erf_mp(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if x.is_zero() {
            return Ok(ctx.float(0.0));
        }

        let abs_x = x.clone().abs();

        // For small x, use Taylor series
        if abs_x < 2.0 {
            erf_series(x, ctx)
        } else {
            // For large x, use asymptotic expansion for erfc
            let erfc_val = erfc_asymptotic(&abs_x, ctx)?;
            Ok(if *x > 0.0 {
                ctx.float(1.0) - erfc_val
            } else {
                erfc_val - ctx.float(1.0)
            })
        }
    }

    /// Taylor series for erf(x)
    fn erf_series(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let sqrt_pi = ctx.pi().sqrt();
        let x2 = x.clone() * x;

        let mut sum = x.clone();
        let mut term = x.clone();

        for n in 1..200 {
            let neg_x2 = Float::with_val(ctx.precision(), -&x2);
            term *= neg_x2 / (n as f64);
            let new_term = Float::with_val(ctx.precision(), &term / (2 * n + 1) as f64);
            sum += &new_term;

            if new_term.abs()
                < sum.clone().abs()
                    * Float::with_val(ctx.precision(), 10.0).pow(-(ctx.precision() as i32) / 10)
            {
                break;
            }
        }

        Ok(2.0 * sum / sqrt_pi)
    }

    /// Asymptotic expansion for erfc(x) for large x
    fn erfc_asymptotic(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let sqrt_pi = ctx.pi().sqrt();
        let x2 = x.clone() * x;
        let neg_x2 = Float::with_val(ctx.precision(), -&x2);
        let exp_neg_x2 = neg_x2.exp();

        let mut sum = ctx.float(1.0);
        let mut term = ctx.float(1.0);

        for n in 1..50 {
            let x2_mult = Float::with_val(ctx.precision(), 2.0 * &x2);
            term *= -(2 * n - 1) as f64 / x2_mult;
            sum += &term;

            if term.clone().abs()
                < sum.clone().abs()
                    * Float::with_val(ctx.precision(), 10.0).pow(-(ctx.precision() as i32) / 10)
            {
                break;
            }
        }

        Ok(exp_neg_x2 * sum / (x.clone() * sqrt_pi))
    }

    /// Compute erfc(x) with arbitrary precision
    pub fn erfc_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<Float> {
        let x_mp = ctx.float(x);
        erfc_mp(&x_mp, ctx)
    }

    /// Compute erfc(x) for arbitrary precision input
    pub fn erfc_mp(x: &Float, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if x.is_zero() {
            return Ok(ctx.float(1.0));
        }

        let abs_x = x.clone().abs();

        // For small x, use erf
        if abs_x < 2.0 {
            let erf_val = erf_mp(x, ctx)?;
            Ok(ctx.float(1.0) - erf_val)
        } else {
            // For large x, use asymptotic expansion
            if *x > 0.0 {
                erfc_asymptotic(x, ctx)
            } else {
                let erfc_pos = erfc_asymptotic(&abs_x, ctx)?;
                Ok(ctx.float(2.0) - erfc_pos)
            }
        }
    }
}

/// Utility functions for arbitrary precision
mod utils {
    use super::*;

    /// Compute factorial with arbitrary precision
    pub fn factorial_mp(n: u32, ctx: &PrecisionContext) -> Float {
        if n == 0 || n == 1 {
            return ctx.float(1.0);
        }

        let mut result = ctx.float(1.0);
        for i in 2..=n {
            result *= i as f64;
        }
        result
    }

    /// Compute binomial coefficient with arbitrary precision
    pub fn binomial_mp(n: u32, k: u32, ctx: &PrecisionContext) -> Float {
        if k > n {
            return ctx.float(0.0);
        }
        if k == 0 || k == n {
            return ctx.float(1.0);
        }

        let k = k.min(n - k); // Take advantage of symmetry

        let mut result = ctx.float(1.0);
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result
    }

    /// Compute Pochhammer symbol (rising factorial) with arbitrary precision
    pub fn pochhammer_mp(x: &Float, n: u32, ctx: &PrecisionContext) -> SpecialResult<Float> {
        if n == 0 {
            return Ok(ctx.float(1.0));
        }

        let mut result = x.clone();
        for i in 1..n {
            result *= x.clone() + i as f64;
        }
        Ok(result)
    }
}

// Re-export utility functions
pub use utils::*;

/// Convert arbitrary precision Float to f64
#[allow(dead_code)]
pub fn to_f64(x: &Float) -> f64 {
    x.to_f64()
}

/// Convert arbitrary precision Complex to num_complex::Complex64
#[allow(dead_code)]
pub fn to_complex64(z: &Complex) -> num_complex::Complex64 {
    let (re, im) = z.clone().into_real_imag();
    num_complex::Complex64::new(re.to_f64(), im.to_f64())
}

/// Clean up MPFR cache
#[allow(dead_code)]
pub fn cleanup_cache() {
    rug::float::free_cache(rug::float::FreeCache::All);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_precision_context() {
        let ctx = PrecisionContext::new(512).unwrap();
        assert_eq!(ctx.precision(), 512);

        let pi = ctx.pi();
        assert!(pi.prec() >= 512);

        // Test that we get more precision than f64
        let pi_str = pi.to_string();
        assert!(pi_str.len() > 20); // More digits than f64 can represent
    }

    #[test]
    fn test_gamma_ap() {
        let ctx = PrecisionContext::default();

        // Test Γ(1) = 1
        let gamma_1 = gamma::gamma_ap(1.0, &ctx).unwrap();
        assert_relative_eq!(to_f64(&gamma_1), 1.0, epsilon = 1e-15);

        // Test Γ(0.5) = √π
        let gamma_half = gamma::gamma_ap(0.5, &ctx).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert_relative_eq!(to_f64(&gamma_half), sqrt_pi, epsilon = 1e-15);

        // Test Γ(5) = 4! = 24
        let gamma_5 = gamma::gamma_ap(5.0, &ctx).unwrap();
        assert_relative_eq!(to_f64(&gamma_5), 24.0, epsilon = 1e-15);
    }

    #[test]
    fn test_bessel_ap() {
        let ctx = PrecisionContext::default();

        // Test J_0(0) = 1
        let j0_0 = bessel::bessel_j_ap(0, 0.0, &ctx).unwrap();
        assert_relative_eq!(to_f64(&j0_0), 1.0, epsilon = 1e-15);

        // Test J_1(0) = 0
        let j1_0 = bessel::bessel_j_ap(1, 0.0, &ctx).unwrap();
        assert_relative_eq!(to_f64(&j1_0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_erf_ap() {
        let ctx = PrecisionContext::default();

        // Test erf(0) = 0
        let erf_0 = error_function::erf_ap(0.0, &ctx).unwrap();
        assert_relative_eq!(to_f64(&erf_0), 0.0, epsilon = 1e-15);

        // Test erfc(0) = 1
        let erfc_0 = error_function::erfc_ap(0.0, &ctx).unwrap();
        assert_relative_eq!(to_f64(&erfc_0), 1.0, epsilon = 1e-15);

        // Test erf(x) + erfc(x) = 1
        let x = 1.5;
        let erf_x = error_function::erf_ap(x, &ctx).unwrap();
        let erfc_x = error_function::erfc_ap(x, &ctx).unwrap();
        let sum = to_f64(&erf_x) + to_f64(&erfc_x);
        assert_relative_eq!(sum, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_high_precision() {
        // Test with 1024-bit precision
        let ctx = PrecisionContext::new(1024).unwrap();

        // Compute π with high precision
        let pi = ctx.pi();
        let pi_str = format!("{:.100}", pi);

        // Check that we have many accurate digits
        let expected_pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679";
        assert!(pi_str.starts_with(expected_pi));
    }
}
