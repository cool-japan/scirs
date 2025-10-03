//! Incomplete gamma and related functions
//!
//! This module provides incomplete gamma functions and their regularized forms,
//! matching SciPy's special module functionality.

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{gamma, gammaln};
use crate::validation::{check_finite, check_positive};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Lower incomplete gamma function
///
/// Computes the lower incomplete gamma function:
/// γ(a, x) = ∫₀ˣ t^(a-1) e^(-t) dt
///
/// # Arguments
/// * `a` - Shape parameter (must be positive)
/// * `x` - Upper limit of integration
///
/// # Returns
/// The value of the lower incomplete gamma function
///
/// # Examples
/// ```
/// use scirs2_special::incomplete_gamma::gammainc_lower;
///
/// let result = gammainc_lower(2.0, 1.0).unwrap();
/// assert!((result - 0.2642411176571153f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn gammainc_lower<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // Use series expansion for x < a + 1
    if x < a + T::one() {
        // Series representation: γ(a,x) = x^a e^(-x) Σ(x^n / Γ(a+n+1))
        let mut sum = T::one() / a;
        let mut term = T::one() / a;
        let mut n = T::one();
        let tol = T::from_f64(1e-15).unwrap();

        while term.abs() > tol * sum.abs() {
            term *= x / (a + n);
            sum += term;
            n += T::one();

            if n > T::from_f64(1000.0).unwrap() {
                return Err(SpecialError::ConvergenceError(
                    "gammainc_lower: series did not converge".to_string(),
                ));
            }
        }

        Ok(x.powf(a) * (-x).exp() * sum)
    } else {
        // Use complement: γ(a,x) = Γ(a) - Γ(a,x)
        let gamma_a = gamma(a);
        let gamma_upper = gammainc_upper(a, x)?;
        Ok(gamma_a - gamma_upper)
    }
}

/// Upper incomplete gamma function
///
/// Computes the upper incomplete gamma function:
/// Γ(a, x) = ∫ₓ^∞ t^(a-1) e^(-t) dt
///
/// # Arguments
/// * `a` - Shape parameter (must be positive)
/// * `x` - Lower limit of integration
///
/// # Returns
/// The value of the upper incomplete gamma function
#[allow(dead_code)]
pub fn gammainc_upper<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(gamma(a));
    }

    // Use continued fraction for x >= a + 1
    if x >= a + T::one() {
        // Continued fraction representation
        let mut b = x + T::one() - a;
        let mut c = T::from_f64(1e30).unwrap();
        let mut d = T::one() / b;
        let mut h = d;
        let tol = T::from_f64(1e-15).unwrap();

        for i in 1..1000 {
            let an = -T::from_usize(i).unwrap() * (T::from_usize(i).unwrap() - a);
            b += T::from_f64(2.0).unwrap();
            d = an * d + b;

            if d.abs() < T::from_f64(1e-30).unwrap() {
                d = T::from_f64(1e-30).unwrap();
            }

            c = b + an / c;
            if c.abs() < T::from_f64(1e-30).unwrap() {
                c = T::from_f64(1e-30).unwrap();
            }

            d = T::one() / d;
            let delta = d * c;
            h *= delta;

            if (delta - T::one()).abs() < tol {
                return Ok(x.powf(a) * (-x).exp() * h);
            }
        }

        Err(SpecialError::ConvergenceError(
            "gammainc_upper: continued fraction did not converge".to_string(),
        ))
    } else {
        // Use complement
        let gamma_a = gamma(a);
        let gamma_lower = gammainc_lower(a, x)?;
        Ok(gamma_a - gamma_lower)
    }
}

/// Regularized lower incomplete gamma function
///
/// Computes P(a, x) = γ(a, x) / Γ(a)
///
/// # Arguments
/// * `a` - Shape parameter
/// * `x` - Upper limit
///
/// # Returns
/// The regularized lower incomplete gamma function
#[allow(dead_code)]
pub fn gammainc<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // For large a, use asymptotic expansion or specialized algorithms
    if a > T::from_f64(100.0).unwrap() {
        // Use log-space computation to avoid overflow
        let log_gamma_a = gammaln(a);
        let log_result = compute_log_gammainc(a, x, log_gamma_a)?;
        Ok(log_result.exp())
    } else {
        let gamma_lower = gammainc_lower(a, x)?;
        let gamma_a = gamma(a);
        Ok(gamma_lower / gamma_a)
    }
}

/// Regularized upper incomplete gamma function
///
/// Computes Q(a, x) = Γ(a, x) / Γ(a)
#[allow(dead_code)]
pub fn gammaincc<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(T::one());
    }

    // Use complement of regularized lower incomplete gamma
    let p = gammainc(a, x)?;
    Ok(T::one() - p)
}

/// Inverse of regularized lower incomplete gamma function
///
/// Find x such that P(a, x) = p
#[allow(dead_code)]
pub fn gammaincinv<T>(a: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign + SubAssign,
{
    check_positive(a, "a")?;
    crate::validation::check_probability(p, "p")?;

    if p == T::zero() {
        return Ok(T::zero());
    }

    if p == T::one() {
        return Ok(T::infinity());
    }

    // Initial guess using Wilson-Hilferty transformation
    let x0 = initial_guess_gammaincinv(a, p);

    // Newton-Raphson iteration
    let mut x = x0;
    let tol = T::from_f64(1e-12).unwrap();

    for _ in 0..100 {
        let f = gammainc(a, x)? - p;

        // Derivative: d/dx P(a,x) = x^(a-1) e^(-x) / Γ(a)
        let df = x.powf(a - T::one()) * (-x).exp() / gamma(a);

        let dx = f / df;
        x -= dx;

        // Ensure x stays positive
        if x <= T::zero() {
            x = T::from_f64(1e-10).unwrap();
        }

        if dx.abs() < tol * x.abs() {
            return Ok(x);
        }
    }

    Err(SpecialError::ConvergenceError(
        "gammaincinv: Newton iteration did not converge".to_string(),
    ))
}

/// Inverse of regularized upper incomplete gamma function
///
/// Find x such that Q(a, x) = q
#[allow(dead_code)]
pub fn gammainccinv<T>(a: T, q: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign + SubAssign,
{
    check_positive(a, "a")?;
    crate::validation::check_probability(q, "q")?;

    // Use Q(a, x) = 1 - P(a, x)
    let p = T::one() - q;
    gammaincinv(a, p)
}

/// Helper function for initial guess in gammaincinv
#[allow(dead_code)]
fn initial_guess_gammaincinv<T>(a: T, p: T) -> T
where
    T: Float + FromPrimitive + Display,
{
    // Wilson-Hilferty approximation
    let g = T::from_f64(2.0).unwrap() / (T::from_f64(9.0).unwrap() * a);
    let z = crate::distributions::ndtri(p).unwrap_or(T::zero());
    let w = T::one() + g * z;

    if w > T::zero() {
        a * w.powf(T::from_f64(3.0).unwrap())
    } else {
        // Fallback for extreme cases
        if p < T::from_f64(0.5).unwrap() {
            a * T::from_f64(0.1).unwrap()
        } else {
            a * T::from_f64(2.0).unwrap()
        }
    }
}

/// Compute log of regularized incomplete gamma in log space to avoid overflow
#[allow(dead_code)]
fn compute_log_gammainc<T>(a: T, x: T, log_gammaa: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + AddAssign + MulAssign,
{
    // Use asymptotic series for large a
    // This is a simplified implementation
    let log_x = x.ln();
    let log_result = a * log_x - x - log_gammaa;

    // Add correction terms for better accuracy
    let mut sum = T::one();
    let mut term = T::one();

    for n in 1..50 {
        term *= (a - T::from_usize(n).unwrap()) / x;
        sum += term;

        if term.abs() < T::from_f64(1e-15).unwrap() {
            break;
        }
    }

    Ok(log_result + sum.ln())
}

/// Gamma star function (used in some asymptotic expansions)
///
/// gammastar(x) = Γ(x) / (sqrt(2π) * x^(x-1/2) * e^(-x))
#[allow(dead_code)]
pub fn gammastar<T>(x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(x, "x")?;

    if x >= T::from_f64(10.0).unwrap() {
        // Use Stirling series
        let mut sum = T::one();
        let x2 = x * x;
        let mut xn = x;

        // Stirling coefficients
        let coeffs = [
            T::from_f64(1.0 / 12.0).unwrap(),
            T::from_f64(1.0 / 288.0).unwrap(),
            T::from_f64(-139.0 / 51840.0).unwrap(),
            T::from_f64(-571.0 / 2488320.0).unwrap(),
        ];

        for &c in &coeffs {
            sum += c / xn;
            xn *= x2;
        }

        Ok(sum)
    } else {
        // Direct computation
        let sqrt_2pi = T::from_f64((2.0 * std::f64::consts::PI).sqrt()).unwrap();
        let gamma_x = gamma(x);
        let x_power = x.powf(x - T::from_f64(0.5).unwrap());
        let exp_neg_x = (-x).exp();

        Ok(gamma_x / (sqrt_2pi * x_power * exp_neg_x))
    }
}

/// Sign of the gamma function
///
/// Returns 1.0 if gamma(x) > 0, -1.0 if gamma(x) < 0
#[allow(dead_code)]
pub fn gammasgn<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x > T::zero() {
        T::one()
    } else {
        // Gamma function alternates sign for negative non-integer values
        let floor_x = x.floor();
        if x == floor_x {
            // Gamma is undefined at negative integers
            T::nan()
        } else {
            // Sign alternates based on floor
            if floor_x.to_isize().unwrap_or(0) % 2 == 0 {
                T::one()
            } else {
                -T::one()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gammainc_lower() {
        // Test values verified against SciPy (non-regularized incomplete gamma)
        assert_relative_eq!(
            gammainc_lower(1.0, 1.0).unwrap(),
            0.6321205588285577, // γ(1,1) = P(1,1) * Γ(1) = 0.6321205588285577 * 1
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gammainc_lower(2.0, 1.0).unwrap(),
            0.264241117657115, // γ(2,1) = P(2,1) * Γ(2) = 0.2642411176571153 * 1
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gammainc_lower(3.0, 2.0).unwrap(),
            0.646647167633873, // γ(3,2) = P(3,2) * Γ(3) = 0.32332358381693654 * 2
            epsilon = 1e-10
        );

        // Edge cases
        assert_eq!(gammainc_lower(1.0, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_gammainc() {
        // Regularized lower incomplete gamma
        assert_relative_eq!(
            gammainc(1.0, 1.0).unwrap(),
            0.6321205588285577,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gammainc(2.0, 2.0).unwrap(),
            0.5939941502901619,
            epsilon = 1e-10
        );

        // P(a,0) = 0, P(a,∞) = 1
        assert_eq!(gammainc(1.0, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_gammaincc() {
        // Q(a,x) = 1 - P(a,x)
        let a = 2.0;
        let x = 1.5;
        let p = gammainc(a, x).unwrap();
        let q = gammaincc(a, x).unwrap();
        assert_relative_eq!(p + q, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gammaincinv() {
        // Test round trip: gammaincinv(a, gammainc(a, x)) ≈ x
        let a = 2.5;
        let x = 3.0;
        let p = gammainc(a, x).unwrap();
        let x_recovered = gammaincinv(a, p).unwrap();
        assert_relative_eq!(x_recovered, x, epsilon = 1e-8);
    }

    #[test]
    fn test_gammasgn() {
        assert_eq!(gammasgn(1.0), 1.0);
        assert_eq!(gammasgn(2.5), 1.0);
        assert_eq!(gammasgn(-0.5), -1.0);
        assert_eq!(gammasgn(-1.5), 1.0);
        assert_eq!(gammasgn(-2.5), -1.0);
    }
}
