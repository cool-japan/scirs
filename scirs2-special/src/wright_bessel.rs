//! Wright Bessel functions
//!
//! This module provides implementations of the Wright Bessel functions,
//! which are generalizations of Bessel functions.
//!
//! ## Functions
//!
//! * `wright_bessel(rho, beta, z)` - The Wright Bessel function J_{rho, beta}(z)
//! * `wright_bessel_zeros(rho, beta, n)` - The first n zeros of the Wright Bessel function
//!
//! ## References
//!
//! 1. Wright, E. M. (1935). "The asymptotic expansion of the generalized Bessel function."
//!    Proceedings of the London Mathematical Society, 38(1), 257-270.
//! 2. Wong, R., & Zhao, Y. Q. (1999). "Exponential asymptotics of the Wright Bessel functions."
//!    Journal of Mathematical Analysis and Applications, 235(1), 285-298.

use crate::error::{SpecialError, SpecialResult};
use num_complex::Complex64;
// Using f64 constants directly without imports
use crate::gamma;

/// Computes the Wright Bessel function J_{rho, beta}(z)
///
/// The Wright Bessel function is defined by the series:
///
/// J_{rho, beta}(z) = sum_{k=0}^{infty} ((-z)^k) / (k! * Gamma(rho*k + beta))
///
/// where rho > 0 and beta is a complex parameter.
///
/// For rho = 1 and beta = 1, this reduces to the ordinary Bessel function J_0(2*sqrt(z)).
///
/// # Arguments
///
/// * `rho` - Parameter rho (must be positive)
/// * `beta` - Parameter beta
/// * `z` - Argument z (real or complex)
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Wright Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::wright_bessel;
///
/// // Wright Bessel function with rho=1, beta=1 at z=1
/// // This equals J_0(2) where J_0 is the ordinary Bessel function
/// let result = wright_bessel(1.0, 1.0, 1.0).unwrap();
/// assert!((result - 0.2239).abs() < 1e-4);
/// ```
pub fn wright_bessel(rho: f64, beta: f64, z: f64) -> SpecialResult<f64> {
    // Parameter validation
    if rho <= 0.0 {
        return Err(SpecialError::DomainError(
            "Parameter rho must be positive".to_string(),
        ));
    }

    if z.is_nan() || beta.is_nan() || rho.is_nan() {
        return Ok(f64::NAN);
    }

    // Special cases
    if z == 0.0 {
        // For z=0, return 1/Gamma(beta) if beta > 0, or 0 if beta <= 0
        if beta > 0.0 {
            return Ok(1.0 / gamma(beta));
        } else {
            return Ok(0.0);
        }
    }

    // If |z| is large, we need asymptotic expansion (not implemented yet)
    if z.abs() > 100.0 {
        return Err(SpecialError::NotImplementedError(
            "Asymptotic expansion for large |z| not implemented yet".to_string(),
        ));
    }

    // Compute using series expansion
    let max_terms = 100;
    let tolerance = 1e-14;

    let mut sum: f64;
    // No need to initialize term variable
    let mut k_factorial = 1.0;

    // Add first term
    sum = 1.0 / gamma(beta);

    // Compute series
    for k in 1..max_terms {
        let k_f64 = k as f64;

        // Update factorial and compute gamma function term
        k_factorial *= k_f64;
        let gamma_term = gamma(rho * k_f64 + beta);

        // Compute next term and add to sum
        let term = ((-z).powi(k) / k_factorial) / gamma_term;
        sum += term;

        // Check for convergence
        if term.abs() < tolerance * sum.abs() {
            break;
        }
    }

    Ok(sum)
}

/// Computes the Wright Bessel function for complex arguments J_{rho, beta}(z)
///
/// # Arguments
///
/// * `rho` - Parameter rho (must be positive)
/// * `beta` - Parameter beta (complex)
/// * `z` - Argument z (complex)
///
/// # Returns
///
/// * `SpecialResult<Complex64>` - The Wright Bessel function value
pub fn wright_bessel_complex(rho: f64, beta: Complex64, z: Complex64) -> SpecialResult<Complex64> {
    // Parameter validation
    if rho <= 0.0 {
        return Err(SpecialError::DomainError(
            "Parameter rho must be positive".to_string(),
        ));
    }

    if z.re.is_nan() || z.im.is_nan() || beta.re.is_nan() || beta.im.is_nan() || rho.is_nan() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    // TODO: Implement full complex version

    // For now, return error for complex implementation
    Err(SpecialError::NotImplementedError(
        "Complex Wright Bessel function not implemented yet".to_string(),
    ))
}

/// Computes the first n zeros of the Wright Bessel function J_{rho, beta}(z)
///
/// # Arguments
///
/// * `rho` - Parameter rho (must be positive)
/// * `beta` - Parameter beta
/// * `n` - Number of zeros to compute (≥ 1)
///
/// # Returns
///
/// * `SpecialResult<Vec<f64>>` - The zeros of the Wright Bessel function
pub fn wright_bessel_zeros(rho: f64, _beta: f64, n: usize) -> SpecialResult<Vec<f64>> {
    // Parameter validation
    if rho <= 0.0 {
        return Err(SpecialError::DomainError(
            "Parameter rho must be positive".to_string(),
        ));
    }

    if n == 0 {
        return Err(SpecialError::DomainError(
            "Number of zeros must be at least 1".to_string(),
        ));
    }

    // TODO: Implement the computation of zeros

    // For now, return error for not implemented
    Err(SpecialError::NotImplementedError(
        "Computation of Wright Bessel function zeros not implemented yet".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wright_bessel_special_cases() {
        // For z=0, beta=1, the result should be 1/Gamma(1) = 1
        let result = wright_bessel(1.0, 1.0, 0.0).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);

        // For z=0, beta=2, the result should be 1/Gamma(2) = 1/1 = 1
        let result = wright_bessel(1.0, 2.0, 0.0).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);

        // For z=0, beta=3, the result should be 1/Gamma(3) = 1/2! = 0.5
        let result = wright_bessel(1.0, 3.0, 0.0).unwrap();
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_wright_bessel_invalid_parameters() {
        // Test with invalid rho
        assert!(wright_bessel(0.0, 1.0, 1.0).is_err());
        assert!(wright_bessel(-1.0, 1.0, 1.0).is_err());

        // Test with NaN parameters
        assert!(wright_bessel(1.0, 1.0, f64::NAN).unwrap().is_nan());
        assert!(wright_bessel(1.0, f64::NAN, 1.0).unwrap().is_nan());
        assert!(wright_bessel(f64::NAN, 1.0, 1.0).unwrap().is_nan());
    }

    // Additional tests would be added here when the function is fully implemented
}
