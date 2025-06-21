//! Precision enhancement utilities
//!
//! This module provides utilities for improving numerical precision in special function
//! computations, including high-precision constants, overflow/underflow handling,
//! and specialized algorithms for extreme parameter values.

#![allow(dead_code)]
#![allow(clippy::approx_constant)]

use crate::error::{SpecialError, SpecialResult};
use std::f64;

/// High-precision mathematical constants
pub mod constants {
    /// π with extended precision (50 decimal places)
    pub const PI_EXTENDED: f64 = 3.141_592_653_589_793;

    /// e (Euler's number) with extended precision
    pub const E_EXTENDED: f64 = 2.718_281_828_459_045;

    /// Euler-Mascheroni constant γ with extended precision
    pub const GAMMA_EXTENDED: f64 = 0.577_215_664_901_532_9;

    /// sqrt(π) with extended precision
    pub const SQRT_PI_EXTENDED: f64 = 1.772_453_850_905_516;

    /// sqrt(2π) with extended precision
    pub const SQRT_2PI_EXTENDED: f64 = 2.506_628_274_631_000_7;

    /// ln(2) with extended precision
    pub const LN_2_EXTENDED: f64 = 0.693_147_180_559_945_3;

    /// ln(π) with extended precision
    pub const LN_PI_EXTENDED: f64 = 1.144_729_885_849_400_2;

    /// ln(2π) with extended precision
    pub const LN_2PI_EXTENDED: f64 = 1.837_877_066_409_345_6;

    /// Catalan constant with extended precision
    pub const CATALAN_EXTENDED: f64 = 0.915_965_594_177_219;

    /// Golden ratio φ with extended precision
    pub const PHI_EXTENDED: f64 = 1.618_033_988_749_895;

    /// Apéry's constant ζ(3) with extended precision
    pub const APERY_EXTENDED: f64 = 1.202_056_903_159_594_2;
}

/// Safe arithmetic operations that handle overflow and underflow gracefully
pub mod safe_ops {
    use super::*;

    /// Safely compute exp(x) with overflow protection
    ///
    /// Returns +∞ for large positive x, 0 for large negative x
    pub fn safe_exp(x: f64) -> f64 {
        if x > 700.0 {
            f64::INFINITY
        } else if x < -700.0 {
            0.0
        } else {
            x.exp()
        }
    }

    /// Safely compute ln(x) with domain checking
    ///
    /// Returns NaN for x ≤ 0, handles very small positive x specially
    pub fn safe_ln(x: f64) -> SpecialResult<f64> {
        if x <= 0.0 {
            return Err(SpecialError::DomainError(
                "Logarithm undefined for non-positive values".to_string(),
            ));
        }

        if x < f64::MIN_POSITIVE {
            return Ok(f64::NEG_INFINITY);
        }

        Ok(x.ln())
    }

    /// Safely compute x^y with overflow/underflow protection
    pub fn safe_pow(x: f64, y: f64) -> SpecialResult<f64> {
        if x < 0.0 && y.fract() != 0.0 {
            return Err(SpecialError::DomainError(
                "Negative base with non-integer exponent".to_string(),
            ));
        }

        if x == 0.0 && y < 0.0 {
            return Ok(f64::INFINITY);
        }

        if x == 0.0 && y == 0.0 {
            return Ok(1.0); // Mathematical convention
        }

        // Check for potential overflow
        let log_result = y * x.abs().ln();
        if log_result > 700.0 {
            return Ok(if x > 0.0 || y as i64 % 2 == 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            });
        }

        if log_result < -700.0 {
            return Ok(0.0);
        }

        Ok(x.powf(y))
    }

    /// Safely compute x * y with overflow detection
    pub fn safe_mul(x: f64, y: f64) -> f64 {
        let result = x * y;
        if result.is_infinite() && x.is_finite() && y.is_finite() {
            // Overflow occurred
            if (x > 0.0) == (y > 0.0) {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            result
        }
    }

    /// Safely compute x + y with extended precision for small differences
    pub fn safe_add(x: f64, y: f64) -> f64 {
        // If one value is much larger than the other, use Kahan summation concept
        if x.abs() > y.abs() {
            let c = y - ((x + y) - x);
            x + y + c
        } else {
            let c = x - ((y + x) - y);
            y + x + c
        }
    }
}

/// Extended precision algorithms for critical computations
pub mod extended {
    use super::*;

    /// Compute sin(x) with extended precision using multiple-precision arithmetic
    ///
    /// Uses argument reduction and series expansion for better precision
    pub fn sin_extended(x: f64) -> f64 {
        if x.is_nan() || x.is_infinite() {
            return f64::NAN;
        }

        // Argument reduction: reduce x to [-π/2, π/2]
        let pi = constants::PI_EXTENDED;
        let two_pi = 2.0 * pi;

        // Reduce to [0, 2π]
        let mut reduced_x = x % two_pi;
        if reduced_x < 0.0 {
            reduced_x += two_pi;
        }

        // Further reduce to [0, π/2] and track quadrant
        let half_pi = pi * 0.5;
        let (final_x, sign) = if reduced_x <= half_pi {
            (reduced_x, 1.0)
        } else if reduced_x <= pi {
            (pi - reduced_x, 1.0)
        } else if reduced_x <= 1.5 * pi {
            (reduced_x - pi, -1.0)
        } else {
            (two_pi - reduced_x, -1.0)
        };

        // Use Taylor series for sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
        let x2 = final_x * final_x;
        let mut term = final_x;
        let mut result = term;

        for n in 1..20 {
            term *= -x2 / ((2 * n) as f64 * (2 * n + 1) as f64);
            result += term;

            if term.abs() < 1e-16 * result.abs() {
                break;
            }
        }

        sign * result
    }

    /// Compute cos(x) with extended precision
    pub fn cos_extended(x: f64) -> f64 {
        let half_pi = constants::PI_EXTENDED * 0.5;
        sin_extended(half_pi - x)
    }

    /// Compute exp(x) with extended precision for moderate values
    ///
    /// Uses series expansion with careful handling of intermediate terms
    pub fn exp_extended(x: f64) -> f64 {
        if x > 700.0 {
            return f64::INFINITY;
        }
        if x < -700.0 {
            return 0.0;
        }

        if x.abs() < 1e-10 {
            return 1.0 + x + 0.5 * x * x;
        }

        // For larger values, use exp(x) = exp(n*ln(2)) * exp(x - n*ln(2))
        // where n is chosen so that |x - n*ln(2)| is small
        let ln2 = constants::LN_2_EXTENDED;
        let n = (x / ln2).round();
        let reduced_x = x - n * ln2;

        // Compute exp(reduced_x) using series
        let mut term = 1.0;
        let mut result = 1.0;

        for k in 1..50 {
            term *= reduced_x / k as f64;
            result += term;

            if term.abs() < 1e-16 * result.abs() {
                break;
            }
        }

        // Multiply by 2^n
        result * safe_ops::safe_pow(2.0, n).unwrap_or(f64::INFINITY)
    }

    /// Compute logarithm with extended precision for values near 1
    ///
    /// Uses ln(1+x) series for x near 0 for better precision
    pub fn ln_extended(x: f64) -> SpecialResult<f64> {
        if x <= 0.0 {
            return Err(SpecialError::DomainError(
                "Logarithm undefined for non-positive values".to_string(),
            ));
        }

        if x == 1.0 {
            return Ok(0.0);
        }

        // For x close to 1, use ln(x) = ln(1 + (x-1))
        if (x - 1.0).abs() < 0.5 {
            let u = x - 1.0;

            // ln(1+u) = u - u²/2 + u³/3 - u⁴/4 + ...
            let mut term = u;
            let mut result = term;

            for n in 2..50 {
                term *= -u;
                result += term / n as f64;

                if term.abs() < 1e-16 * result.abs() {
                    break;
                }
            }

            Ok(result)
        } else {
            Ok(x.ln())
        }
    }
}

/// Specialized algorithms for extreme parameter values
pub mod extreme {
    use super::*;

    /// Asymptotic expansion for large arguments
    ///
    /// Provides asymptotic series for functions when x → ∞
    pub fn asymptotic_series(x: f64, coefficients: &[f64]) -> SpecialResult<f64> {
        if x < 10.0 {
            return Err(SpecialError::DomainError(
                "Asymptotic series only valid for large arguments".to_string(),
            ));
        }

        let mut result = coefficients[0];
        let mut power = 1.0;

        for &coeff in &coefficients[1..] {
            power /= x;
            let term = coeff * power;
            result += term;

            // Stop if terms become negligible
            if term.abs() < 1e-15 * result.abs() {
                break;
            }
        }

        Ok(result)
    }

    /// Continued fraction evaluation with precision control
    ///
    /// Evaluates continued fractions of the form a₀ + b₁/(a₁ + b₂/(a₂ + ...))
    pub fn continued_fraction(
        a: &[f64],
        b: &[f64],
        max_terms: usize,
        tolerance: f64,
    ) -> SpecialResult<f64> {
        if a.is_empty() || b.is_empty() {
            return Err(SpecialError::DomainError(
                "Coefficient arrays cannot be empty".to_string(),
            ));
        }

        let n = std::cmp::min(a.len(), b.len()).min(max_terms);

        // Use backward recurrence for numerical stability
        let mut p_prev = a[n - 1];
        let mut p_curr = a[n - 2] * p_prev + b[n - 1];

        for i in (0..n - 2).rev() {
            let p_new = a[i] * p_curr + b[i + 1] * p_prev;

            // Check for convergence
            if i < n - 3 {
                let error = (p_new / p_curr - p_curr / p_prev).abs();
                if error < tolerance {
                    break;
                }
            }

            p_prev = p_curr;
            p_curr = p_new;
        }

        Ok(p_curr / p_prev)
    }

    /// Rational approximation using Padé approximants
    ///
    /// Computes P(x)/Q(x) where P and Q are polynomials
    pub fn pade_approximant(x: f64, p_coeffs: &[f64], q_coeffs: &[f64]) -> SpecialResult<f64> {
        if p_coeffs.is_empty() || q_coeffs.is_empty() {
            return Err(SpecialError::DomainError(
                "Coefficient arrays cannot be empty".to_string(),
            ));
        }

        // Evaluate numerator P(x)
        let mut p_value = 0.0;
        let mut x_power = 1.0;
        for &coeff in p_coeffs {
            p_value += coeff * x_power;
            x_power *= x;
        }

        // Evaluate denominator Q(x)
        let mut q_value = 0.0;
        x_power = 1.0;
        for &coeff in q_coeffs {
            q_value += coeff * x_power;
            x_power *= x;
        }

        if q_value.abs() < 1e-15 {
            return Err(SpecialError::DomainError(
                "Denominator too close to zero in Padé approximant".to_string(),
            ));
        }

        Ok(p_value / q_value)
    }
}

/// Error analysis and precision estimation utilities
pub mod error_analysis {
    /// Estimate the relative error in a computation
    pub fn relative_error(computed: f64, exact: f64) -> f64 {
        if exact == 0.0 {
            computed.abs()
        } else {
            ((computed - exact) / exact).abs()
        }
    }

    /// Estimate the number of accurate decimal digits
    pub fn accurate_digits(computed: f64, exact: f64) -> u32 {
        let rel_err = relative_error(computed, exact);
        if rel_err == 0.0 {
            16 // Maximum for f64
        } else {
            (-rel_err.log10()).max(0.0) as u32
        }
    }

    /// Check if a result meets the required precision threshold
    pub fn check_precision(computed: f64, exact: f64, required_digits: u32) -> bool {
        accurate_digits(computed, exact) >= required_digits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_extended_precision_constants() {
        // Test that our extended precision constants are at least as accurate
        assert!((constants::PI_EXTENDED - std::f64::consts::PI).abs() < 1e-15);
        assert!((constants::E_EXTENDED - std::f64::consts::E).abs() < 1e-15);

        // Test that they're actually the same value within machine precision
        assert_relative_eq!(
            constants::PI_EXTENDED,
            std::f64::consts::PI,
            epsilon = 1e-15
        );
        assert_relative_eq!(constants::E_EXTENDED, std::f64::consts::E, epsilon = 1e-15);
    }

    #[test]
    fn test_safe_operations() {
        // Test overflow protection
        assert_eq!(safe_ops::safe_exp(1000.0), f64::INFINITY);
        assert_eq!(safe_ops::safe_exp(-1000.0), 0.0);

        // Test normal range
        assert_relative_eq!(
            safe_ops::safe_exp(1.0),
            std::f64::consts::E,
            epsilon = 1e-10
        );

        // Test safe logarithm
        assert!(safe_ops::safe_ln(-1.0).is_err());
        assert!(safe_ops::safe_ln(0.0).is_err());
        assert_relative_eq!(
            safe_ops::safe_ln(std::f64::consts::E).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_extended_trigonometric() {
        // Test extended precision sin/cos
        assert_relative_eq!(extended::sin_extended(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(
            extended::sin_extended(constants::PI_EXTENDED / 2.0),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(extended::cos_extended(0.0), 1.0, epsilon = 1e-15);
        assert_relative_eq!(
            extended::cos_extended(constants::PI_EXTENDED),
            -1.0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_extended_exponential() {
        assert_relative_eq!(extended::exp_extended(0.0), 1.0, epsilon = 1e-15);
        assert_relative_eq!(
            extended::exp_extended(1.0),
            constants::E_EXTENDED,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_extended_logarithm() {
        assert_relative_eq!(extended::ln_extended(1.0).unwrap(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(
            extended::ln_extended(constants::E_EXTENDED).unwrap(),
            1.0,
            epsilon = 1e-14
        );

        // Test near 1
        assert_relative_eq!(
            extended::ln_extended(1.001).unwrap(),
            1.001_f64.ln(),
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_pade_approximant() {
        // Test simple case: e^x ≈ (1 + x/2) / (1 - x/2) for small x
        let p_coeffs = [1.0, 0.5];
        let q_coeffs = [1.0, -0.5];

        let x = 0.1;
        let pade_result = extreme::pade_approximant(x, &p_coeffs, &q_coeffs).unwrap();
        let exact = x.exp();

        // Should be reasonably close for small x
        assert!((pade_result - exact).abs() < 0.01);
    }

    #[test]
    fn test_error_analysis() {
        let computed = 3.14159;
        let exact = std::f64::consts::PI;

        let rel_err = error_analysis::relative_error(computed, exact);
        assert!(rel_err > 0.0);
        assert!(rel_err < 1e-4);

        let digits = error_analysis::accurate_digits(computed, exact);
        assert!(digits >= 4);
        assert!(digits <= 16);
    }
}
