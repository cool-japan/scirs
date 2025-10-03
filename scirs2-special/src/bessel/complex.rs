//! Complex Bessel functions
//!
//! This module provides complex number support for Bessel functions of the first kind,
//! second kind, and modified Bessel functions.
//!
//! The implementation uses various techniques for different regions of the complex plane:
//! - Series expansions for small |z|
//! - Asymptotic expansions for large |z|  
//! - Hankel transforms for complex arguments
//! - Connection formulas between different types

#![allow(dead_code)]

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Complex Bessel function J₀(z) of the first kind, order 0
///
/// Implements the complex Bessel function J₀(z) for z ∈ ℂ.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex Bessel function value J₀(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::j0_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = j0_complex(z);
/// // For real arguments, should match real J₀(1) ≈ 0.7651976866
/// assert!((result.re - 0.7651976866).abs() < 1e-8);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn j0_complex(z: Complex64) -> Complex64 {
    // For real values, use the real Bessel function for accuracy
    if z.im.abs() < 1e-15 && z.re >= 0.0 {
        let real_result = crate::bessel::j0(z.re);
        return Complex64::new(real_result, 0.0);
    }

    // Handle special cases
    if z.norm() == 0.0 {
        return Complex64::new(1.0, 0.0);
    }

    // For small |z|, use series expansion
    if z.norm() < 8.0 {
        return j0_series_complex(z);
    }

    // For large |z|, use asymptotic expansion
    j0_asymptotic_complex(z)
}

/// Complex Bessel function J₁(z) of the first kind, order 1
///
/// Implements the complex Bessel function J₁(z) for z ∈ ℂ.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex Bessel function value J₁(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::j1_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = j1_complex(z);
/// // For real arguments, should match real J₁(1) ≈ 0.4400505857
/// assert!((result.re - 0.4400505857).abs() < 1e-8);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn j1_complex(z: Complex64) -> Complex64 {
    // For real values, use the real Bessel function for accuracy
    if z.im.abs() < 1e-15 && z.re >= 0.0 {
        let real_result = crate::bessel::j1(z.re);
        return Complex64::new(real_result, 0.0);
    }

    // Handle special cases
    if z.norm() == 0.0 {
        return Complex64::new(0.0, 0.0);
    }

    // For small |z|, use series expansion
    if z.norm() < 8.0 {
        return j1_series_complex(z);
    }

    // For large |z|, use asymptotic expansion
    j1_asymptotic_complex(z)
}

/// Complex Bessel function Jₙ(z) of the first kind, integer order n
///
/// Implements the complex Bessel function Jₙ(z) for z ∈ ℂ and integer n.
///
/// # Arguments
///
/// * `n` - Integer order
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex Bessel function value Jₙ(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::jn_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(2.0, 0.0);
/// let result = jn_complex(2, z);
/// // For real arguments, should match real J₂(2) ≈ 0.3528340286
/// assert!((result.re - 0.3528340286).abs() < 1e-8);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn jn_complex(n: i32, z: Complex64) -> Complex64 {
    // For real values, use the real Bessel function for accuracy
    if z.im.abs() < 1e-15 && z.re >= 0.0 {
        let real_result = crate::bessel::jn(n, z.re);
        return Complex64::new(real_result, 0.0);
    }

    // Handle special cases
    if z.norm() == 0.0 {
        return if n == 0 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    // For n = 0, 1, use specialized implementations
    match n {
        0 => j0_complex(z),
        1 => j1_complex(z),
        -1 => -j1_complex(z),
        _ => {
            // For other orders, use recurrence relation or series
            if n.abs() <= 50 && z.norm() < 8.0 {
                jn_series_complex(n, z)
            } else {
                jn_asymptotic_complex(n, z)
            }
        }
    }
}

/// Complex Bessel function Jᵥ(z) of the first kind, arbitrary order v
///
/// Implements the complex Bessel function Jᵥ(z) for z ∈ ℂ and arbitrary real order v.
///
/// # Arguments
///
/// * `v` - Real order (can be non-integer)
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex Bessel function value Jᵥ(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::jv_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = jv_complex(0.5, z);
/// // J₀.₅(1) = √(2/π) * sin(1) ≈ 0.6713967
/// assert!((result.re - 0.6713967072).abs() < 1e-8);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn jv_complex(v: f64, z: Complex64) -> Complex64 {
    // For real values, use the real Bessel function for accuracy
    if z.im.abs() < 1e-15 && z.re >= 0.0 {
        let real_result = crate::bessel::jv(v, z.re);
        return Complex64::new(real_result, 0.0);
    }

    // Handle special cases
    if z.norm() == 0.0 {
        return if v == 0.0 {
            Complex64::new(1.0, 0.0)
        } else if v > 0.0 {
            Complex64::new(0.0, 0.0)
        } else {
            Complex64::new(f64::INFINITY, 0.0)
        };
    }

    // For integer orders, use the integer implementation
    if v.fract() == 0.0 && v.abs() < i32::MAX as f64 {
        return jn_complex(v as i32, z);
    }

    // For half-integer orders, use spherical Bessel functions
    if (v - 0.5).fract() == 0.0 {
        return jv_half_integer_complex(v, z);
    }

    // For small |z|, use series expansion
    if z.norm() < 8.0 {
        return jv_series_complex(v, z);
    }

    // For large |z|, use asymptotic expansion
    jv_asymptotic_complex(v, z)
}

/// Complex modified Bessel function I₀(z) of the first kind, order 0
///
/// Implements the complex modified Bessel function I₀(z) for z ∈ ℂ.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex modified Bessel function value I₀(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::i0_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = i0_complex(z);
/// // For real arguments, should match real I₀(1) ≈ 1.2661
/// assert!((result.re - 1.2660658480).abs() < 1e-8);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn i0_complex(z: Complex64) -> Complex64 {
    // For real values, use the real modified Bessel function for accuracy
    if z.im.abs() < 1e-15 && z.re >= 0.0 {
        let real_result = crate::bessel::i0(z.re);
        return Complex64::new(real_result, 0.0);
    }

    // I₀(z) = J₀(iz) for pure imaginary argument, but more generally:
    // I₀(z) = Σ(z/2)^(2k) / (k!)² from k=0 to ∞

    // Handle special cases
    if z.norm() == 0.0 {
        return Complex64::new(1.0, 0.0);
    }

    // For small |z|, use series expansion
    if z.norm() < 8.0 {
        return i0_series_complex(z);
    }

    // For large |z|, use asymptotic expansion
    i0_asymptotic_complex(z)
}

/// Complex modified Bessel function K₀(z) of the second kind, order 0
///
/// Implements the complex modified Bessel function K₀(z) for z ∈ ℂ.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex modified Bessel function value K₀(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::k0_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = k0_complex(z);
/// // For real arguments, should match real K₀(1) ≈ 0.4611
/// assert!((result.re - 0.4610685044).abs() < 1e-8);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn k0_complex(z: Complex64) -> Complex64 {
    // For real positive values, use the real modified Bessel function for accuracy
    if z.im.abs() < 1e-15 && z.re > 0.0 {
        let real_result = crate::bessel::k0(z.re);
        return Complex64::new(real_result, 0.0);
    }

    // Handle special cases
    if z.norm() == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    // For small |z|, use series expansion
    if z.norm() < 8.0 {
        return k0_series_complex(z);
    }

    // For large |z|, use asymptotic expansion
    k0_asymptotic_complex(z)
}

// Implementation functions for series expansions

/// Series expansion for J₀(z) for small |z|
#[allow(dead_code)]
fn j0_series_complex(z: Complex64) -> Complex64 {
    let mut result = Complex64::new(1.0, 0.0);
    let z2 = z * z;
    let mut term = Complex64::new(1.0, 0.0);

    for k in 1..=50 {
        term *= -z2 / Complex64::new((4 * k * k) as f64, 0.0);
        result += term;

        if term.norm() < 1e-15 * result.norm() {
            break;
        }
    }

    result
}

/// Series expansion for J₁(z) for small |z|
#[allow(dead_code)]
fn j1_series_complex(z: Complex64) -> Complex64 {
    let mut result = z / Complex64::new(2.0, 0.0);
    let z2 = z * z;
    let mut term = result;

    for k in 1..=50 {
        term *= -z2 / Complex64::new((4 * k * (k + 1)) as f64, 0.0);
        result += term;

        if term.norm() < 1e-15 * result.norm() {
            break;
        }
    }

    result
}

/// Series expansion for Jₙ(z) for small |z|
#[allow(dead_code)]
fn jn_series_complex(n: i32, z: Complex64) -> Complex64 {
    if n == 0 {
        return j0_series_complex(z);
    }
    if n == 1 {
        return j1_series_complex(z);
    }
    if n == -1 {
        return -j1_series_complex(z);
    }

    // For |n| > 1, use upward/downward recurrence
    if n > 1 {
        // Upward recurrence: J_{n+1}(z) = (2n/z)J_n(z) - J_{n-1}(z)
        let mut j_nm1 = j0_series_complex(z);
        let mut j_n = j1_series_complex(z);

        for k in 1..n {
            let j_np1 = Complex64::new(2.0 * k as f64, 0.0) / z * j_n - j_nm1;
            j_nm1 = j_n;
            j_n = j_np1;
        }

        j_n
    } else {
        // For negative n, use J_{-n}(z) = (-1)^n J_n(z)
        let result = jn_series_complex(-n, z);
        if n % 2 == 0 {
            result
        } else {
            -result
        }
    }
}

/// Series expansion for Jᵥ(z) for small |z|
#[allow(dead_code)]
fn jv_series_complex(v: f64, z: Complex64) -> Complex64 {
    use crate::gamma::gamma;

    let z_half = z / Complex64::new(2.0, 0.0);
    let z_half_pow_v = z_half.powf(v);
    let gamma_v_plus_1 = gamma(v + 1.0);

    let mut result = z_half_pow_v / Complex64::new(gamma_v_plus_1, 0.0);
    let z2 = z * z;
    let mut term = result;

    for k in 1..=50 {
        term *= -z2 / Complex64::new((4 * k) as f64 * (v + k as f64), 0.0);
        result += term;

        if term.norm() < 1e-15 * result.norm() {
            break;
        }
    }

    result
}

/// Half-integer order Bessel functions
#[allow(dead_code)]
fn jv_half_integer_complex(v: f64, z: Complex64) -> Complex64 {
    // For half-integer orders, J_{n+1/2}(z) = √(2/πz) * spherical bessel functions
    let sqrt_2_over_pi_z = (Complex64::new(2.0 / PI, 0.0) / z).sqrt();

    let n = (v - 0.5) as i32;

    if n == 0 {
        // J_{1/2}(z) = √(2/πz) * sin(z)
        sqrt_2_over_pi_z * z.sin()
    } else if n == -1 {
        // J_{-1/2}(z) = √(2/πz) * cos(z)
        sqrt_2_over_pi_z * z.cos()
    } else {
        // Use recurrence for other half-integers
        spherical_bessel_jn_complex(n, z) * sqrt_2_over_pi_z
    }
}

/// Spherical Bessel function jₙ(z) for complex arguments
#[allow(dead_code)]
fn spherical_bessel_jn_complex(n: i32, z: Complex64) -> Complex64 {
    if n == 0 {
        if z.norm() < 1e-8 {
            Complex64::new(1.0, 0.0)
        } else {
            z.sin() / z
        }
    } else if n == 1 {
        if z.norm() < 1e-8 {
            Complex64::new(0.0, 0.0)
        } else {
            z.sin() / (z * z) - z.cos() / z
        }
    } else {
        // Use recurrence relation
        let mut j_nm1 = spherical_bessel_jn_complex(0, z);
        let mut j_n = spherical_bessel_jn_complex(1, z);

        for k in 1..n {
            let j_np1 = Complex64::new((2 * k + 1) as f64, 0.0) / z * j_n - j_nm1;
            j_nm1 = j_n;
            j_n = j_np1;
        }

        j_n
    }
}

/// Series expansion for I₀(z) for small |z|
#[allow(dead_code)]
fn i0_series_complex(z: Complex64) -> Complex64 {
    let mut result = Complex64::new(1.0, 0.0);
    let z2 = z * z;
    let mut term = Complex64::new(1.0, 0.0);

    for k in 1..=50 {
        term *= z2 / Complex64::new((4 * k * k) as f64, 0.0);
        result += term;

        if term.norm() < 1e-15 * result.norm() {
            break;
        }
    }

    result
}

/// Series expansion for K₀(z) for small |z|
#[allow(dead_code)]
fn k0_series_complex(z: Complex64) -> Complex64 {
    // K₀(z) has a logarithmic singularity at z=0
    // K₀(z) = -ln(z/2)I₀(z) + Σ

    let i0_z = i0_series_complex(z);
    let ln_z_half = (z / Complex64::new(2.0, 0.0)).ln();

    let mut series_part = Complex64::new(0.0, 0.0);
    let z2 = z * z;
    let mut term = Complex64::new(1.0, 0.0);

    // Psi function values for small integers
    let psi_values = [
        0.0,
        -0.5772156649015329,
        0.4227843350984671,
        0.9227843350984671,
    ];

    for k in 1..=50 {
        term *= z2 / Complex64::new((4 * k * k) as f64, 0.0);
        let psi_k = if k < psi_values.len() {
            psi_values[k]
        } else {
            harmonic_number(k) - 0.5772156649015329 // Approximate psi function
        };

        series_part += term * Complex64::new(psi_k, 0.0);

        if term.norm() < 1e-15 {
            break;
        }
    }

    -ln_z_half * i0_z + series_part
}

/// Harmonic number H_n = 1 + 1/2 + ... + 1/n
#[allow(dead_code)]
fn harmonic_number(n: usize) -> f64 {
    (1..=n).map(|k| 1.0 / k as f64).sum()
}

// Implementation functions for asymptotic expansions

/// Asymptotic expansion for J₀(z) for large |z|
#[allow(dead_code)]
fn j0_asymptotic_complex(z: Complex64) -> Complex64 {
    let sqrt_2_over_pi_z = (Complex64::new(2.0 / PI, 0.0) / z).sqrt();
    let phase = z - Complex64::new(PI / 4.0, 0.0);
    sqrt_2_over_pi_z * phase.cos()
}

/// Asymptotic expansion for J₁(z) for large |z|
#[allow(dead_code)]
fn j1_asymptotic_complex(z: Complex64) -> Complex64 {
    let sqrt_2_over_pi_z = (Complex64::new(2.0 / PI, 0.0) / z).sqrt();
    let phase = z - Complex64::new(3.0 * PI / 4.0, 0.0);
    sqrt_2_over_pi_z * phase.cos()
}

/// Asymptotic expansion for Jₙ(z) for large |z|
#[allow(dead_code)]
fn jn_asymptotic_complex(n: i32, z: Complex64) -> Complex64 {
    let sqrt_2_over_pi_z = (Complex64::new(2.0 / PI, 0.0) / z).sqrt();
    let phase = z - Complex64::new((n as f64 + 0.5) * PI / 2.0, 0.0);
    sqrt_2_over_pi_z * phase.cos()
}

/// Asymptotic expansion for Jᵥ(z) for large |z|
#[allow(dead_code)]
fn jv_asymptotic_complex(v: f64, z: Complex64) -> Complex64 {
    let sqrt_2_over_pi_z = (Complex64::new(2.0 / PI, 0.0) / z).sqrt();
    let phase = z - Complex64::new((v + 0.5) * PI / 2.0, 0.0);
    sqrt_2_over_pi_z * phase.cos()
}

/// Asymptotic expansion for I₀(z) for large |z|
#[allow(dead_code)]
fn i0_asymptotic_complex(z: Complex64) -> Complex64 {
    let sqrt_2_pi_z = (Complex64::new(2.0 * PI, 0.0) * z).sqrt();
    z.exp() / sqrt_2_pi_z
}

/// Asymptotic expansion for K₀(z) for large |z|
#[allow(dead_code)]
fn k0_asymptotic_complex(z: Complex64) -> Complex64 {
    let sqrt_pi_over_2z = (Complex64::new(PI / 2.0, 0.0) / z).sqrt();
    sqrt_pi_over_2z * (-z).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_j0_complex_real_values() {
        // Test real values match real J₀ function
        let test_values = [0.0, 1.0, 2.0, 5.0, 10.0];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = j0_complex(z);
            let real_result = crate::bessel::j0(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-10);
            assert!(complex_result.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_j1_complex_real_values() {
        // Test real values match real J₁ function
        let test_values = [0.0, 1.0, 2.0, 5.0, 10.0];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = j1_complex(z);
            let real_result = crate::bessel::j1(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-10);
            assert!(complex_result.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_jn_complex_real_values() {
        // Test real values match real Jₙ function
        let test_values = [0.0, 1.0, 2.0, 5.0];
        let orders = [0, 1, 2, 3, 5];

        for &n in &orders {
            for &x in &test_values {
                let z = Complex64::new(x, 0.0);
                let complex_result = jn_complex(n, z);
                let real_result = crate::bessel::jn(n, x);

                assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-8);
                assert!(complex_result.im.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_i0_complex_real_values() {
        // Test real values match real I₀ function
        let test_values = [0.0, 1.0, 2.0, 5.0];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = i0_complex(z);
            let real_result = crate::bessel::i0(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-10);
            assert!(complex_result.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_k0_complex_real_values() {
        // Test real positive values match real K₀ function
        let test_values = [0.1, 1.0, 2.0, 5.0];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = k0_complex(z);
            let real_result = crate::bessel::k0(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-8);
            assert!(complex_result.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_bessel_recurrence_relation() {
        // Test recurrence relation: J_{n-1}(z) + J_{n+1}(z) = (2n/z)J_n(z)
        let z = Complex64::new(2.0, 1.0);
        let n = 2;

        let j_nm1 = jn_complex(n - 1, z);
        let j_n = jn_complex(n, z);
        let j_np1 = jn_complex(n + 1, z);

        let lhs = j_nm1 + j_np1;
        let rhs = Complex64::new(2.0 * n as f64, 0.0) / z * j_n;

        let error = (lhs - rhs).norm();
        assert!(error < 1e-10);
    }

    #[test]
    fn test_modified_bessel_properties() {
        // Test that I₀(0) = 1
        let z = Complex64::new(0.0, 0.0);
        let i0_result = i0_complex(z);
        assert_relative_eq!(i0_result.re, 1.0, epsilon = 1e-10);
        assert!(i0_result.im.abs() < 1e-12);
    }

    #[test]
    fn test_half_integer_bessel() {
        // Test J_{1/2}(x) = √(2/πx) sin(x)
        let x = 2.0;
        let z = Complex64::new(x, 0.0);
        let j_half = jv_complex(0.5, z);

        let expected = (2.0 / (PI * x)).sqrt() * x.sin();
        assert_relative_eq!(j_half.re, expected, epsilon = 1e-8);
        assert!(j_half.im.abs() < 1e-10);
    }
}
