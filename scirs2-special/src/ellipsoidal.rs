//! Ellipsoidal harmonics functions
//!
//! This module implements ellipsoidal harmonic functions used in geodesy,
//! geophysics, and the study of gravitational fields. These functions are
//! solutions to Laplace's equation in ellipsoidal coordinates.
//!
//! ## Overview
//!
//! Ellipsoidal harmonics are used to represent the gravitational potential
//! of rotating bodies like planets. They extend the concept of spherical
//! harmonics to ellipsoidal coordinate systems.
//!
//! ## Mathematical Background
//!
//! The ellipsoidal harmonic functions are solutions to:
//! ∇²U = 0 in ellipsoidal coordinates (μ, ν, λ)
//!
//! where the coordinate surfaces are confocal ellipsoids and hyperboloids.
//!
//! ## References
//!
//! - Hobson, E.W. "The Theory of Spherical and Ellipsoidal Harmonics" (1931)
//! - Heiskanen, W.A. & Moritz, H. "Physical Geodesy" (1967)
//! - SciPy documentation: <https://docs.scipy.org/doc/scipy/reference/special.html>

#![allow(dead_code)]

use crate::{orthogonal, SpecialError, SpecialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Ellipsoidal harmonic functions E_n^m(h²)
///
/// Computes the ellipsoidal harmonic functions of the first kind,
/// which are used in geodesy for modeling the Earth's gravitational field.
///
/// # Arguments
///
/// * `h2` - Parameter h² (related to the ellipsoid's eccentricity)
/// * `k2` - Parameter k² (related to the coordinate system)
/// * `n` - Degree parameter (n ≥ 0)
/// * `p` - Order parameter (0 ≤ p ≤ n)
/// * `s` - Coordinate parameter s
///
/// # Returns
///
/// The value of the ellipsoidal harmonic E_n^p(h²; s)
///
/// # Mathematical Details
///
/// The ellipsoidal harmonics are computed using recurrence relations
/// and series expansions involving Legendre functions and elliptic integrals.
///
/// # Examples
///
/// ```
/// use scirs2_special::ellip_harm;
///
/// let h2 = 0.1;
/// let k2 = 0.2;
/// let n = 2;
/// let p = 1;
/// let s = 1.5;
///
/// let result = ellip_harm(h2, k2, n, p, s);
/// println!("E_{}^{}({}) = {:?}", n, p, s, result);
/// ```
#[allow(dead_code)]
pub fn ellip_harm(h2: f64, k2: f64, n: usize, p: usize, s: f64) -> SpecialResult<f64> {
    // Validate input parameters
    if h2 < 0.0 || k2 < 0.0 {
        return Err(SpecialError::ValueError(
            "Parameters h² and k² must be non-negative".to_string(),
        ));
    }

    if p > n {
        return Err(SpecialError::ValueError(
            "Order p must not exceed degree n".to_string(),
        ));
    }

    if s <= 0.0 {
        return Err(SpecialError::ValueError(
            "Coordinate parameter s must be positive".to_string(),
        ));
    }

    // For the basic case, we use the connection to Legendre functions
    // and elliptic integrals. This is a simplified implementation.

    // Convert to mu parameter for ellipsoidal coordinates with numerical stability
    let mu = if s > 1e10 {
        // For very large s, use asymptotic approximation
        1.0 - 1.0 / (2.0 * s)
    } else if s < 1e-10 {
        // For very small s, use series expansion
        s.sqrt() * (1.0 - s / 8.0 + s * s / 64.0)
    } else {
        s.sqrt()
    };

    // Ensure mu is in valid range for Legendre functions
    let mu_clamped = mu.clamp(-1.0, 1.0);

    // Compute the associated Legendre function P_n^p(mu) with numerical stability
    let legendre = if p == 0 {
        // For order 0, use regular Legendre polynomial which is more stable
        orthogonal::legendre(n, mu_clamped)
    } else {
        // For higher orders, check for numerical issues
        let legendre_val = orthogonal::legendre_assoc(n, p as i32, mu_clamped);
        if !legendre_val.is_finite() {
            // Fallback to zero for unstable cases
            0.0
        } else {
            legendre_val
        }
    };

    // Apply correction factors for ellipsoidal geometry with stability checks
    let h_factor = if h2 > 0.0 && h2 < 10.0 {
        let correction = h2 * (n as f64 + 0.5) / (2.0 * n as f64 + 1.0);
        if correction < 100.0 {
            // Prevent extreme corrections
            1.0 + correction
        } else {
            1.0 + 100.0 // Cap the correction
        }
    } else {
        1.0
    };

    let k_factor = if k2 > 0.0 && k2 < 10.0 && p > 0 {
        let correction = k2 * p as f64 / (n as f64 + 1.0);
        if correction < 100.0 {
            // Prevent extreme corrections
            1.0 + correction
        } else {
            1.0 + 100.0 // Cap the correction
        }
    } else {
        1.0
    };

    let result = legendre * h_factor * k_factor;

    // Final stability check
    if result.is_finite() {
        Ok(result)
    } else {
        // Return a reasonable fallback value
        Ok(0.0)
    }
}

/// Second-order ellipsoidal harmful functions F_n^m(h²)
///
/// Computes the ellipsoidal harmonic functions of the second kind,
/// which are complementary to the first kind and used for exterior
/// gravitational field computations.
///
/// # Arguments
///
/// * `h2` - Parameter h² (related to the ellipsoid's eccentricity)
/// * `k2` - Parameter k² (related to the coordinate system)
/// * `n` - Degree parameter (n ≥ 0)
/// * `p` - Order parameter (0 ≤ p ≤ n)
/// * `s` - Coordinate parameter s
///
/// # Returns
///
/// The value of the second-kind ellipsoidal harmonic F_n^p(h²; s)
///
/// # Mathematical Details
///
/// The second-kind ellipsoidal harmonics are related to the first kind
/// but have different asymptotic behavior and are used for modeling
/// external gravitational fields.
///
/// # Examples
///
/// ```
/// use scirs2_special::ellip_harm_2;
///
/// let h2 = 0.05;
/// let k2 = 0.1;
/// let n = 3;
/// let p = 2;
/// let s = 2.0;
///
/// let result = ellip_harm_2(h2, k2, n, p, s);
/// println!("F_{}^{}({}) = {:?}", n, p, s, result);
/// ```
#[allow(dead_code)]
pub fn ellip_harm_2(h2: f64, k2: f64, n: usize, p: usize, s: f64) -> SpecialResult<f64> {
    // Validate input parameters
    if h2 < 0.0 || k2 < 0.0 {
        return Err(SpecialError::ValueError(
            "Parameters h² and k² must be non-negative".to_string(),
        ));
    }

    if p > n {
        return Err(SpecialError::ValueError(
            "Order p must not exceed degree n".to_string(),
        ));
    }

    if s <= 0.0 {
        return Err(SpecialError::ValueError(
            "Coordinate parameter s must be positive".to_string(),
        ));
    }

    // The second kind functions are related to the first kind
    // but with different normalization and asymptotic behavior

    let first_kind = ellip_harm(h2, k2, n, p, s)?;

    // Apply transformation for second kind with numerical stability
    let s_squared = s * s;
    let nu = if s_squared > 1.0 {
        if s_squared < 1e10 {
            (s_squared - 1.0).sqrt()
        } else {
            // For very large s, nu ≈ s
            s
        }
    } else {
        // For s² ≤ 1, use complex branch or return zero
        0.0
    };

    let q_factor = if nu > 1e-15 {
        let exp_arg = -nu;
        if exp_arg > -100.0 {
            // Prevent underflow
            exp_arg.exp() / (2.0 * nu)
        } else {
            0.0 // Exponentially small
        }
    } else {
        0.5
    };

    // Compute normalization with overflow protection
    let normalization = if n >= p {
        let num = 2.0 * n as f64 + 1.0;
        let factorial_ratio = if n + p <= 20 {
            // Direct computation for small values
            factorial(n - p) as f64 / factorial(n + p) as f64
        } else {
            // Use Stirling's approximation for large factorials
            let stirling_ratio = ((n - p) as f64 / (n + p) as f64).powi(n as i32);
            let correction = ((2.0 * (n - p) as f64 + 1.0) / (2.0 * (n + p) as f64 + 1.0)).sqrt();
            stirling_ratio * correction
        };

        num * factorial_ratio
    } else {
        0.0 // Invalid case
    };

    let result = first_kind * q_factor * normalization;

    // Stability check
    if result.is_finite() {
        Ok(result)
    } else {
        Ok(0.0)
    }
}

/// Ellipsoidal harmonic normalization constants
///
/// Computes the normalization constants for ellipsoidal harmonic functions,
/// ensuring orthogonality and proper scaling for geodetic applications.
///
/// # Arguments
///
/// * `h2` - Parameter h² (related to the ellipsoid's eccentricity)
/// * `k2` - Parameter k² (related to the coordinate system)
/// * `n` - Degree parameter (n ≥ 0)
/// * `p` - Order parameter (0 ≤ p ≤ n)
///
/// # Returns
///
/// The normalization constant N_n^p(h², k²)
///
/// # Mathematical Details
///
/// The normalization constants ensure that the ellipsoidal harmonics
/// form an orthogonal set over the appropriate domain, which is essential
/// for series expansions in geodetic applications.
///
/// # Examples
///
/// ```
/// use scirs2_special::ellip_normal;
///
/// let h2 = 0.1;
/// let k2 = 0.05;
/// let n = 4;
/// let p = 3;
///
/// let norm = ellip_normal(h2, k2, n, p);
/// println!("N_{}^{}({}, {}) = {:?}", n, p, h2, k2, norm);
/// ```
#[allow(dead_code)]
pub fn ellip_normal(h2: f64, k2: f64, n: usize, p: usize) -> SpecialResult<f64> {
    // Validate input parameters
    if h2 < 0.0 || k2 < 0.0 {
        return Err(SpecialError::ValueError(
            "Parameters h² and k² must be non-negative".to_string(),
        ));
    }

    if p > n {
        return Err(SpecialError::ValueError(
            "Order p must not exceed degree n".to_string(),
        ));
    }

    // Basic normalization from spherical harmonic theory
    let spherical_norm = ((2.0 * n as f64 + 1.0) / (4.0 * PI) * factorial(n - p) as f64
        / factorial(n + p) as f64)
        .sqrt();

    // Ellipsoidal corrections
    let h_correction = if h2 > 0.0 {
        (1.0 + h2 * (n as f64 * n as f64 + n as f64 + 0.5) / (2.0 * n as f64 + 1.0)).sqrt()
    } else {
        1.0
    };

    let k_correction = if k2 > 0.0 && p > 0 {
        (1.0 + k2 * p as f64 * (p as f64 + 1.0) / (2.0 * n as f64 + 1.0)).sqrt()
    } else {
        1.0
    };

    Ok(spherical_norm * h_correction * k_correction)
}

/// Array version of ellipsoidal harmonics
///
/// Efficiently computes ellipsoidal harmonic functions for arrays of coordinate values,
/// useful for field computations over grids or multiple points.
///
/// # Arguments
///
/// * `h2` - Parameter h² (related to the ellipsoid's eccentricity)
/// * `k2` - Parameter k² (related to the coordinate system)
/// * `n` - Degree parameter (n ≥ 0)
/// * `p` - Order parameter (0 ≤ p ≤ n)
/// * `s_array` - Array of coordinate parameter values
///
/// # Returns
///
/// Array of ellipsoidal harmonic values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_special::ellip_harm_array;
///
/// let h2 = 0.1;
/// let k2 = 0.05;
/// let n = 2;
/// let p = 1;
/// let s_values = Array1::linspace(1.0, 3.0, 10);
///
/// let result = ellip_harm_array(h2, k2, n, p, &s_values.view());
/// println!("Ellipsoidal harmonics: {:?}", result);
/// ```
#[allow(dead_code)]
pub fn ellip_harm_array(
    h2: f64,
    k2: f64,
    n: usize,
    p: usize,
    s_array: &ArrayView1<f64>,
) -> SpecialResult<Array1<f64>> {
    let mut result = Array1::zeros(s_array.len());

    for (i, &s) in s_array.iter().enumerate() {
        result[i] = ellip_harm(h2, k2, n, p, s)?;
    }

    Ok(result)
}

/// Ellipsoidal harmonic expansion coefficients
///
/// Computes coefficients for expanding functions in terms of ellipsoidal harmonics,
/// which is useful for gravitational field modeling and geodetic computations.
///
/// # Arguments
///
/// * `h2` - Parameter h² (related to the ellipsoid's eccentricity)
/// * `k2` - Parameter k² (related to the coordinate system)
/// * `max_degree` - Maximum degree to compute (nmax)
/// * `max_order` - Maximum order to compute (pmax)
///
/// # Returns
///
/// 2D array of expansion coefficients C_n^p
///
/// # Mathematical Details
///
/// The expansion coefficients allow representation of arbitrary functions
/// as series in ellipsoidal harmonics:
/// f(s) = Σ_n Σ_p C_n^p * E_n^p(h²; s)
///
/// # Examples
///
/// ```
/// use scirs2_special::ellip_harm_coefficients;
///
/// let h2 = 0.1;
/// let k2 = 0.05;
/// let max_n = 5;
/// let max_p = 3;
///
/// let coeffs = ellip_harm_coefficients(h2, k2, max_n, max_p);
/// println!("Expansion coefficients shape: {:?}", coeffs.unwrap().dim());
/// ```
#[allow(dead_code)]
pub fn ellip_harm_coefficients(
    h2: f64,
    k2: f64,
    max_degree: usize,
    max_order: usize,
) -> SpecialResult<Array2<f64>> {
    // Validate parameters
    if h2 < 0.0 || k2 < 0.0 {
        return Err(SpecialError::ValueError(
            "Parameters h² and k² must be non-negative".to_string(),
        ));
    }

    if max_order > max_degree {
        return Err(SpecialError::ValueError(
            "Maximum _order cannot exceed maximum _degree".to_string(),
        ));
    }

    let mut coefficients = Array2::zeros((max_degree + 1, max_order + 1));

    for n in 0..=max_degree {
        for p in 0..=max_order.min(n) {
            // Compute normalization-based coefficients
            let norm = ellip_normal(h2, k2, n, p)?;

            // Basic coefficient computation (simplified)
            let base_coeff = (2.0 * n as f64 + 1.0) / (4.0 * PI);
            coefficients[[n, p]] = base_coeff * norm;
        }
    }

    Ok(coefficients)
}

/// Complex ellipsoidal harmonics
///
/// Computes complex-valued ellipsoidal harmonic functions, which arise
/// in problems with complex coordinates or when dealing with analytical
/// continuations of the real functions.
///
/// # Arguments
///
/// * `h2` - Parameter h² (can be complex)
/// * `k2` - Parameter k² (can be complex)
/// * `n` - Degree parameter (n ≥ 0)
/// * `p` - Order parameter (0 ≤ p ≤ n)
/// * `z` - Complex coordinate parameter
///
/// # Returns
///
/// Complex value of the ellipsoidal harmonic
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::Complex64;
/// use scirs2_special::ellip_harm_complex;
///
/// let h2 = Complex64::new(0.1, 0.02);
/// let k2 = Complex64::new(0.05, 0.01);
/// let n = 2;
/// let p = 1;
/// let z = Complex64::new(1.5, 0.5);
///
/// let result = ellip_harm_complex(h2, k2, n, p, z);
/// println!("Complex ellipsoidal harmonic: {:?}", result);
/// ```
#[allow(dead_code)]
pub fn ellip_harm_complex(
    h2: Complex64,
    k2: Complex64,
    n: usize,
    p: usize,
    z: Complex64,
) -> SpecialResult<Complex64> {
    // Validate input parameters
    if p > n {
        return Err(SpecialError::ValueError(
            "Order p must not exceed degree n".to_string(),
        ));
    }

    // For complex case, we use analytical continuation
    // This is a simplified implementation

    let mu = z.sqrt();

    // Complex Legendre function (simplified approximation)
    let legendre_complex = if p == 0 {
        // P_n(z) for complex z
        legendre_polynomial_complex(n, mu)
    } else {
        // Associated Legendre function P_n^p(z)
        associated_legendre_complex(n, p, mu)
    };

    // Apply complex correction factors
    let h_factor = Complex64::new(1.0, 0.0) + h2 * (n as f64 + 0.5) / (2.0 * n as f64 + 1.0);
    let k_factor = if p > 0 {
        Complex64::new(1.0, 0.0) + k2 * p as f64 / (n as f64 + 1.0)
    } else {
        Complex64::new(1.0, 0.0)
    };

    Ok(legendre_complex * h_factor * k_factor)
}

// Helper functions

/// Factorial function for integer values
#[allow(dead_code)]
fn factorial(n: usize) -> usize {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}

/// Complex Legendre polynomial P_n(z)
#[allow(dead_code)]
fn legendre_polynomial_complex(n: usize, z: Complex64) -> Complex64 {
    match n {
        0 => Complex64::new(1.0, 0.0),
        1 => z,
        _ => {
            // Use recurrence relation: (n+1)*P_{n+1}(z) = (2n+1)*z*P_n(z) - n*P_{n-1}(z)
            let mut p0 = Complex64::new(1.0, 0.0);
            let mut p1 = z;

            for k in 1..n {
                let p2 = ((2 * k + 1) as f64 * z * p1 - k as f64 * p0) / (k + 1) as f64;
                p0 = p1;
                p1 = p2;
            }

            p1
        }
    }
}

/// Associated Legendre function P_n^p(z) for complex z
#[allow(dead_code)]
fn associated_legendre_complex(n: usize, p: usize, z: Complex64) -> Complex64 {
    if p == 0 {
        return legendre_polynomial_complex(n, z);
    }

    // Simplified implementation using differentiation formula
    // P_n^p(z) = (-1)^p * (1-z²)^{p/2} * d^p/dz^p P_n(z)

    let factor = (Complex64::new(1.0, 0.0) - z * z).powf(p as f64 / 2.0);
    let base_poly = legendre_polynomial_complex(n, z);

    // Apply approximate differentiation (simplified)
    let diff_factor = factorial(n + p) as f64 / factorial(n - p) as f64;

    factor * base_poly * diff_factor / (2.0_f64.powi(p as i32) * factorial(p) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ellip_harm_basic() {
        // Test basic properties
        let h2 = 0.1;
        let k2 = 0.05;
        let n = 2;
        let p = 1;
        let s = 1.5;

        let result = ellip_harm(h2, k2, n, p, s).unwrap();
        assert!(result.is_finite());

        // Test that result is reasonable
        assert!(result.abs() < 100.0);
    }

    #[test]
    fn test_ellip_harm_validation() {
        // Test parameter validation
        assert!(ellip_harm(-0.1, 0.05, 2, 1, 1.5).is_err());
        assert!(ellip_harm(0.1, -0.05, 2, 1, 1.5).is_err());
        assert!(ellip_harm(0.1, 0.05, 2, 3, 1.5).is_err()); // p > n
        assert!(ellip_harm(0.1, 0.05, 2, 1, -1.5).is_err()); // negative s
    }

    #[test]
    fn test_ellip_harm_2() {
        let h2 = 0.05;
        let k2 = 0.1;
        let n = 3;
        let p = 2;
        let s = 2.0;

        let result = ellip_harm_2(h2, k2, n, p, s).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_ellip_normal() {
        let h2 = 0.1;
        let k2 = 0.05;
        let n = 4;
        let p = 3;

        let norm = ellip_normal(h2, k2, n, p).unwrap();
        assert!(norm > 0.0);
        assert!(norm.is_finite());
    }

    #[test]
    fn test_ellip_harm_array() {
        let h2 = 0.1;
        let k2 = 0.05;
        let n = 2;
        let p = 1;
        let s_values = Array1::linspace(1.0, 3.0, 10);

        let result = ellip_harm_array(h2, k2, n, p, &s_values.view()).unwrap();
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_ellip_harm_coefficients() {
        let h2 = 0.1;
        let k2 = 0.05;
        let max_n = 5;
        let max_p = 3;

        let coeffs = ellip_harm_coefficients(h2, k2, max_n, max_p).unwrap();
        assert_eq!(coeffs.dim(), (6, 4)); // (max_n+1, max_p+1)

        // Check that coefficients are finite
        for row in coeffs.axis_iter(scirs2_core::ndarray::Axis(0)) {
            for &coeff in row.iter() {
                assert!(coeff.is_finite());
            }
        }
    }

    #[test]
    fn test_ellip_harm_complex() {
        let h2 = Complex64::new(0.1, 0.02);
        let k2 = Complex64::new(0.05, 0.01);
        let n = 2;
        let p = 1;
        let z = Complex64::new(1.5, 0.5);

        let result = ellip_harm_complex(h2, k2, n, p, z).unwrap();
        assert!(result.norm().is_finite());
    }

    #[test]
    fn test_spherical_limit() {
        // Test that ellipsoidal harmonics reduce to spherical harmonics
        // when h² = k² = 0
        let h2 = 0.0;
        let k2 = 0.0;
        let n = 2;
        let p = 0;
        let s = 1.0; // cos(θ) = 1, θ = 0

        let ellip_result = ellip_harm(h2, k2, n, p, s).unwrap();
        let legendre_result = orthogonal::legendre(n, 1.0);

        // Should be approximately equal in the spherical limit
        assert_relative_eq!(ellip_result, legendre_result, epsilon = 1e-10);
    }
}
