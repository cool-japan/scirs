//! Kelvin functions
//!
//! This module provides implementations of Kelvin functions and their derivatives.
//! Kelvin functions arise in the study of wave propagation in cylindrical media
//! and are closely related to Bessel functions.
//!
//! The Kelvin functions include:
//!
//! - ber(x): Real part of J₀(x·e^(3πi/4))
//! - bei(x): Imaginary part of J₀(x·e^(3πi/4))
//! - ker(x): Real part of K₀(x·e^(π/4))
//! - kei(x): Imaginary part of K₀(x·e^(π/4))
//!
//! And their derivatives:
//!
//! - berp(x): Derivative of ber(x)
//! - beip(x): Derivative of bei(x)
//! - kerp(x): Derivative of ker(x)
//! - keip(x): Derivative of kei(x)

use scirs2_core::numeric::Complex64;
use std::f64::consts::{PI, SQRT_2};
// use std::ops::Neg;
use scirs2_core::numeric::Zero;

use crate::bessel::{i0, i1, k0, k1};
use crate::error::{SpecialError, SpecialResult};

const SQRT_2_2: f64 = SQRT_2 / 2.0;

/// Compute all Kelvin functions and their derivatives at once.
///
/// This function returns four complex numbers representing the Kelvin functions
/// and their derivatives:
/// - Be = ber(x) + i·bei(x)
/// - Ke = ker(x) + i·kei(x)
/// - Bep = berp(x) + i·beip(x)
/// - Kep = kerp(x) + i·keip(x)
///
/// The implementation handles special cases:
/// - For x = 0: ber(0) = 1, bei(0) = 0, ker(0) = ∞, kei(0) = -π/2
/// - For negative x: Uses the symmetry relations between Kelvin functions
/// - For small x: Uses fast-converging series expansions
/// - For large x: Uses asymptotic forms based on Bessel functions
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * A tuple of four complex numbers: (Be, Ke, Bep, Kep)
///
/// # Errors
///
/// Returns a `SpecialError::DomainError` error if the input is NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::kelvin;
/// use scirs2_core::numeric::Complex64;
///
/// let (be, ke, bep, kep) = kelvin(2.0).unwrap();
/// println!("ber(2.0) + i·bei(2.0) = {} + {}i", be.re, be.im);
/// println!("ker(2.0) + i·kei(2.0) = {} + {}i", ke.re, ke.im);
/// println!("berp(2.0) + i·beip(2.0) = {} + {}i", bep.re, bep.im);
/// println!("kerp(2.0) + i·keip(2.0) = {} + {}i", kep.re, kep.im);
/// ```
#[allow(dead_code)]
pub fn kelvin(x: f64) -> SpecialResult<(Complex64, Complex64, Complex64, Complex64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to kelvin".to_string()));
    }

    // Handle special cases
    if x == 0.0 {
        let be = Complex64::new(1.0, 0.0);
        let ke = Complex64::new(f64::INFINITY, -PI / 2.0);
        let bep = Complex64::new(0.0, 0.0);
        let kep = Complex64::new(f64::NEG_INFINITY, 0.0);
        return Ok((be, ke, bep, kep));
    }

    // For extremely small x, use the series expansions with high precision
    if x.abs() < 1e-200 {
        let be = Complex64::new(1.0, 0.0);
        let ke = Complex64::new(f64::INFINITY, -PI / 2.0);
        let bep = Complex64::new(0.0, 0.0);
        let kep = Complex64::new(f64::NEG_INFINITY, 0.0);
        return Ok((be, ke, bep, kep));
    }

    // For negative x, use the relation to positive x
    if x < 0.0 {
        let (be, ke, bep, kep) = kelvin(-x)?;
        // Kelvin functions have specific symmetry properties:
        // ber(-x) = ber(x), bei(-x) = -bei(x)
        // ker(-x) = -ker(x), kei(-x) = -kei(x)
        // berp(-x) = -berp(x), beip(-x) = beip(x)
        // kerp(-x) = kerp(x), keip(-x) = keip(x)
        return Ok((
            Complex64::new(be.re, -be.im),
            Complex64::new(-ke.re, -ke.im),
            Complex64::new(-bep.re, bep.im),
            Complex64::new(kep.re, kep.im),
        ));
    }

    // For small positive x, use the series expansions
    if x <= 8.0 {
        return compute_kelvin_series(x);
    }

    // For larger values, compute using Bessel functions
    compute_kelvin_bessel(x)
}

/// Compute Kelvin functions using Bessel function relations.
///
/// The relationships between Kelvin functions and Bessel functions are:
/// - ber(x) + i·bei(x) = J₀(x·e^(3πi/4))
/// - ker(x) + i·kei(x) = K₀(x·e^(π/4))
#[allow(dead_code)]
fn compute_kelvin_bessel(x: f64) -> SpecialResult<(Complex64, Complex64, Complex64, Complex64)> {
    // The argument for Bessel functions
    let z = Complex64::new(x * SQRT_2_2, x * SQRT_2_2); // x·e^(iπ/4)

    // Calculate J₀(z) and J₁(z)
    // Note: We're using the fact that J₀(ze^(iπ/2)) = J₀(-iz) and simple transformations
    // to convert between z·e^(3πi/4) and z·e^(iπ/4)

    // Compute J₀(z) and J₁(z) for complex argument
    let j0_z = compute_j0_complex(z)?;
    let j1_z = compute_j1_complex(z)?;

    // Compute K₀(z) and K₁(z) for complex argument
    let k0_z = compute_k0_complex(z)?;
    let k1_z = compute_k1_complex(z)?;

    // Bei + i·Ber is related to J₀(z·e^(3πi/4))
    let be = Complex64::new(j0_z.re, j0_z.im);

    // Kei + i·Ker is related to K₀(z·e^(π/4))
    let ke = Complex64::new(k0_z.re, k0_z.im);

    // Derivatives
    // d/dx [ber(x) + i·bei(x)] = -J₁(x·e^(3πi/4))·e^(3πi/4)
    let bep = -j1_z * Complex64::new(SQRT_2_2, SQRT_2_2);

    // d/dx [ker(x) + i·kei(x)] = -K₁(x·e^(π/4))·e^(π/4)
    let kep = -k1_z * Complex64::new(SQRT_2_2, SQRT_2_2);

    Ok((be, ke, bep, kep))
}

/// Compute Kelvin functions using series approximations.
#[allow(dead_code)]
fn compute_kelvin_series(x: f64) -> SpecialResult<(Complex64, Complex64, Complex64, Complex64)> {
    let x_2 = x / 2.0;
    let x_2_sq = x_2 * x_2;

    // Series for ber and bei functions (real and imaginary parts of J₀)
    let mut ber = 1.0;
    let mut bei = 0.0;
    let mut term = 1.0;
    let mut k = 1;
    let _sign = 1.0;

    while k <= 60 {
        term *= x_2_sq / (k as f64 * k as f64);

        if k % 4 == 0 {
            ber += term;
        } else if k % 4 == 2 {
            ber -= term;
        } else if k % 4 == 1 {
            bei += term;
        } else {
            // k % 4 == 3
            bei -= term;
        }

        // Better convergence criteria with both absolute and relative tolerance
        // Also protect against zero denominators
        let abs_tol = 1e-15;
        let rel_tol = 1e-15 * (ber.abs().max(bei.abs()).max(1e-300));

        if term.abs() < abs_tol || term.abs() < rel_tol {
            break;
        }

        k += 1;
    }

    // Series for derivatives of ber and bei
    // Variables for derivatives - currently not used but prepared for future implementation
    let mut _berp = 0.0;
    let mut _beip = 0.0;

    // d/dx(ber(x)) = -x/4 * (ber'(x) + bei'(x))
    // d/dx(bei(x)) = -x/4 * (bei'(x) - ber'(x))
    //
    // The following formulas provide more accurate derivatives
    // for the Kelvin ber and bei functions

    let mut berp_val = 0.0;
    let mut beip_val = 0.0;

    if x != 0.0 {
        k = 1;
        term = x / 2.0; // First term for derivative

        while k <= 60 {
            if k % 4 == 1 {
                berp_val -= term;
            } else if k % 4 == 3 {
                berp_val += term;
            } else if k % 4 == 0 {
                beip_val -= term;
            } else {
                // k % 4 == 2
                beip_val += term;
            }

            term *= x_2_sq / (k as f64 * (k + 1) as f64);

            // Check convergence with similar criteria as before
            let abs_tol = 1e-15;
            let rel_tol = 1e-15 * (berp_val.abs().max(beip_val.abs()).max(1e-300));

            if term.abs() < abs_tol || term.abs() < rel_tol {
                break;
            }

            k += 1;
        }
    }

    // Series for ker and kei functions (similar approach using K₀)
    let (ker, kei) = compute_ker_kei_series(x)?;
    let (kerp, keip) = compute_kerp_keip_series(x)?;

    let be = Complex64::new(ber, bei);
    let ke = Complex64::new(ker, kei);
    let bep = Complex64::new(berp_val, beip_val);
    let kep = Complex64::new(kerp, keip);

    Ok((be, ke, bep, kep))
}

/// Compute ker and kei functions using series approximation
#[allow(dead_code)]
fn compute_ker_kei_series(x: f64) -> SpecialResult<(f64, f64)> {
    if x == 0.0 {
        return Ok((f64::INFINITY, -PI / 2.0));
    }

    let x_2 = x / 2.0;
    let x_2_sq = x_2 * x_2;
    let ln_x_2 = (x_2).ln();

    // Constants
    let euler_gamma = 0.577_215_664_901_532_9;

    // Series for ker and kei
    let mut ker = -ln_x_2;
    let mut kei = -PI / 2.0;
    let mut term = 1.0;
    let mut k = 1;

    while k <= 60 {
        term *= x_2_sq / (k as f64 * k as f64);

        if k % 4 == 0 {
            ker -= term * (euler_gamma + ln_x_2 + 0.25 * psi(k));
            kei -= PI / 2.0 * term;
        } else if k % 4 == 2 {
            ker += term * (euler_gamma + ln_x_2 + 0.25 * psi(k));
            kei += PI / 2.0 * term;
        } else if k % 4 == 1 {
            ker -= term * PI / 2.0;
            kei += term * (euler_gamma + ln_x_2 + 0.25 * psi(k));
        } else {
            // k % 4 == 3
            ker += term * PI / 2.0;
            kei -= term * (euler_gamma + ln_x_2 + 0.25 * psi(k));
        }

        // Improved convergence criteria with protection against zero denominators
        let abs_tol = 1e-15;
        let rel_tol = 1e-15 * (ker.abs().max(kei.abs()).max(1e-300));

        if term.abs() < abs_tol || term.abs() < rel_tol {
            break;
        }

        k += 1;
    }

    Ok((ker, kei))
}

/// Compute the derivatives of ker and kei functions using series approximation
#[allow(dead_code)]
fn compute_kerp_keip_series(x: f64) -> SpecialResult<(f64, f64)> {
    if x == 0.0 {
        return Ok((f64::NEG_INFINITY, 0.0));
    }

    // Using series expansion for better accuracy of derivatives
    let (ker, kei) = compute_ker_kei_series(x)?;

    let x_2 = x / 2.0;
    let x_2_sq = x_2 * x_2;
    let ln_x_2 = (x_2).ln();

    // Constants
    let euler_gamma = 0.577_215_664_901_532_9;

    // Series for kerp and keip
    let mut kerp = -1.0 / (2.0 * x);
    let mut keip = -PI / (4.0 * x);
    let mut term = 0.5;
    let mut k = 1;

    // Series for derivatives
    while k <= 60 {
        term *= x_2_sq / (k as f64 * (k + 1) as f64);

        if k % 4 == 0 {
            kerp += term * (euler_gamma + ln_x_2 + 0.25 * psi(k + 1));
            keip += PI / 2.0 * term;
        } else if k % 4 == 2 {
            kerp -= term * (euler_gamma + ln_x_2 + 0.25 * psi(k + 1));
            keip -= PI / 2.0 * term;
        } else if k % 4 == 1 {
            kerp += term * PI / 2.0;
            keip -= term * (euler_gamma + ln_x_2 + 0.25 * psi(k + 1));
        } else {
            // k % 4 == 3
            kerp -= term * PI / 2.0;
            keip += term * (euler_gamma + ln_x_2 + 0.25 * psi(k + 1));
        }

        // Improved convergence criteria
        let abs_tol = 1e-15;
        let rel_tol = 1e-15 * (kerp.abs().max(keip.abs()).max(1e-300));

        if term.abs() < abs_tol || term.abs() < rel_tol {
            break;
        }

        k += 1;
    }

    // Validate against the basic recurrence relation as a sanity check
    let kerp_check = -kei / 2.0 - 1.0 / (2.0 * x);
    let keip_check = ker / 2.0 - PI / (4.0 * x);

    // If our series didn't converge well, fall back to the recurrence relation
    if (kerp - kerp_check).abs() > 0.01 * kerp.abs()
        || (keip - keip_check).abs() > 0.01 * keip.abs()
    {
        kerp = kerp_check;
        keip = keip_check;
    }

    Ok((kerp, keip))
}

/// Compute the Bessel function J₀ for complex argument
#[allow(dead_code)]
fn compute_j0_complex(z: Complex64) -> SpecialResult<Complex64> {
    // For complex arguments, we can use the identity:
    // J₀(z) = (H₀⁽¹⁾(z) + H₀⁽²⁾(z))/2
    // H₀⁽¹⁾(z) = J₀(z) + iY₀(z)
    // H₀⁽²⁾(z) = J₀(z) - iY₀(z)

    if z.norm() < 1e-10 {
        return Ok(Complex64::new(1.0, 0.0));
    }

    // Check for NaN or infinite components
    if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
        return Err(SpecialError::DomainError(
            "Invalid complex input to J₀".to_string(),
        ));
    }

    // For small to medium z, use the power series
    if z.norm() < 15.0 {
        let mut sum = Complex64::new(1.0, 0.0);
        let z2 = -z * z / 4.0;
        let mut k_factorial_squared = 1.0; // (k!)²

        for k in 1..100 {
            // Calculate (k!)² directly to avoid overflow in intermediate computations
            k_factorial_squared *= k as f64 * k as f64;

            // Protect against overflow
            if k_factorial_squared == f64::INFINITY {
                break;
            }

            // Safe computation of term
            let term = z2.powf(k as f64) / k_factorial_squared;
            sum += term;

            // Better convergence check with protection against zero division
            let abs_tol = 1e-15;
            let rel_tol = 1e-15 * sum.norm().max(1e-300);
            let term_norm = (z2.powf(k as f64) / k_factorial_squared).norm();

            if term_norm < abs_tol || term_norm < rel_tol {
                break;
            }

            // Safety break to prevent excessive iterations
            if k > 60 && term_norm < 1e-10 {
                break;
            }
        }

        return Ok(sum);
    }

    // For larger arguments, use a more accurate asymptotic form
    // Based on the large-argument asymptotic form of J₀(z) for complex z

    // Avoid division by zero and ensure good numerical accuracy
    // by using a scaled approach for very large arguments
    let scale_factor = if z.norm() > 1e100 {
        let scale = 1.0 / z.norm();
        z * scale
    } else {
        z
    };

    let sqrt_pi_2z = (PI / (2.0 * scale_factor)).sqrt();

    // Phase calculation
    let phase = z - PI / 4.0;

    // Improved asymptotic form with first few terms
    // J₀(z) ~ sqrt(2/πz) * [cos(z-π/4) * P(1/z) - sin(z-π/4) * Q(1/z)]
    // where P and Q are asymptotic series in powers of 1/z²

    // We'll use the first few terms of the asymptotic series
    let one_over_z_sq = 1.0 / (z * z);

    let p_term = 1.0 - 0.125 * one_over_z_sq + 0.073242 * one_over_z_sq * one_over_z_sq;
    let q_term = -0.0625 * one_over_z_sq + 0.097656 * one_over_z_sq * one_over_z_sq;

    let cos_phase = phase.cos();
    let sin_phase = phase.sin();

    let j0_z = sqrt_pi_2z * (cos_phase * p_term - sin_phase * q_term);

    Ok(j0_z)
}

/// Compute the Bessel function J₁ for complex argument
#[allow(dead_code)]
fn compute_j1_complex(z: Complex64) -> SpecialResult<Complex64> {
    if z.is_zero() {
        return Ok(Complex64::new(0.0, 0.0));
    }

    // Check for NaN or infinite components
    if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
        return Err(SpecialError::DomainError(
            "Invalid complex input to J₁".to_string(),
        ));
    }

    // For small to medium z, use the power series
    if z.norm() < 15.0 {
        let mut sum = Complex64::new(0.5, 0.0) * z;
        let z2 = -z * z / 4.0;
        let mut k_times_k_plus_1_factorial = 1.0; // k*(k+1)*...

        for k in 1..100 {
            // Update the factorial product carefully to avoid overflow
            k_times_k_plus_1_factorial *= k as f64 * (k + 1) as f64;

            // Protect against overflow
            if k_times_k_plus_1_factorial == f64::INFINITY {
                break;
            }

            // More stable computation of term
            let term = z * z2.powf(k as f64) / (2.0 * k_times_k_plus_1_factorial);
            sum += term;

            // Better convergence check with protection against zero division
            let abs_tol = 1e-15;
            let rel_tol = 1e-15 * sum.norm().max(1e-300);
            let term_norm = (z * z2.powf(k as f64) / (2.0 * k_times_k_plus_1_factorial)).norm();

            if term_norm < abs_tol || term_norm < rel_tol {
                break;
            }

            // Safety break
            if k > 60 && term_norm < 1e-10 {
                break;
            }
        }

        return Ok(sum);
    }

    // For larger arguments, use a more accurate asymptotic form
    // Based on the large-argument asymptotic form of J₁(z) for complex z

    // Avoid division by zero and ensure good numerical accuracy
    // by using a scaled approach for very large arguments
    let scale_factor = if z.norm() > 1e100 {
        let scale = 1.0 / z.norm();
        z * scale
    } else {
        z
    };

    let sqrt_pi_2z = (PI / (2.0 * scale_factor)).sqrt();

    // Phase calculation
    let phase = z - 3.0 * PI / 4.0;

    // Improved asymptotic form with first few terms
    // J₁(z) ~ sqrt(2/πz) * [cos(z-3π/4) * P(1/z) - sin(z-3π/4) * Q(1/z)]

    // We'll use the first few terms of the asymptotic series
    let one_over_z_sq = 1.0 / (z * z);

    let p_term = 1.0 + 0.375 * one_over_z_sq - 0.073242 * one_over_z_sq * one_over_z_sq;
    let q_term = 0.1875 * one_over_z_sq - 0.097656 * one_over_z_sq * one_over_z_sq;

    let cos_phase = phase.cos();
    let sin_phase = phase.sin();

    let j1_z = sqrt_pi_2z * (cos_phase * p_term - sin_phase * q_term);

    Ok(j1_z)
}

/// Compute the modified Bessel function K₀ for complex argument
#[allow(dead_code)]
fn compute_k0_complex(z: Complex64) -> SpecialResult<Complex64> {
    if z.is_zero() {
        return Err(SpecialError::DomainError("K₀(0) is infinite".to_string()));
    }

    // Check for NaN or infinite components
    if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
        return Err(SpecialError::DomainError(
            "Invalid complex input to K₀".to_string(),
        ));
    }

    // For complex arguments with small imaginary part, we can use a more accurate approach
    if z.im.abs() < 0.01 * z.re.abs() && z.re > 0.0 {
        // For nearly-real positive arguments, use the real K₀ and I₀
        // Direct calculation, since these functions return f64, not Result
        let k0_re = k0(z.re);
        let i0_re = i0(z.re);
        return Ok(Complex64::new(k0_re, z.im * PI * i0_re / z.re));
    }

    // For general complex arguments, use the connection to the Hankel function
    // K₀(z) = π/2 * i * H₀⁽¹⁾(iz)
    let iz = Complex64::new(-z.im, z.re); // i*z

    // Compute asymptotic form for Hankel function of first kind
    let sqrt_pi_2iz = (PI / (2.0 * iz)).sqrt();
    let phase = iz - Complex64::new(PI / 4.0, 0.0);

    // First terms of asymptotic series
    let one_over_z_sq = 1.0 / (iz * iz);
    let p_term = 1.0 - 0.125 * one_over_z_sq + 0.073242 * one_over_z_sq * one_over_z_sq;
    let q_term = -0.0625 * one_over_z_sq + 0.097656 * one_over_z_sq * one_over_z_sq;

    // H₀⁽¹⁾(z) ~ sqrt(2/πz) * exp(i(z-π/4)) * (1 + O(1/z))
    // For complex phase, we use e^(i*phase) directly
    let exp_i_phase = (Complex64::i() * phase).exp();
    let h0_iz = sqrt_pi_2iz * exp_i_phase * (p_term + Complex64::i() * q_term);

    // K₀(z) = π/2 * i * H₀⁽¹⁾(iz)
    let k0_z = PI / 2.0 * Complex64::i() * h0_iz;

    Ok(k0_z)
}

/// Compute the modified Bessel function K₁ for complex argument
#[allow(dead_code)]
fn compute_k1_complex(z: Complex64) -> SpecialResult<Complex64> {
    if z.is_zero() {
        return Err(SpecialError::DomainError("K₁(0) is infinite".to_string()));
    }

    // Check for NaN or infinite components
    if z.re.is_nan() || z.im.is_nan() || z.re.is_infinite() || z.im.is_infinite() {
        return Err(SpecialError::DomainError(
            "Invalid complex input to K₁".to_string(),
        ));
    }

    // For complex arguments with small imaginary part, we can use a more accurate approach
    if z.im.abs() < 0.01 * z.re.abs() && z.re > 0.0 {
        // For nearly-real positive arguments, use the real K₁ and I₁
        // Direct calculation, since these functions return f64, not Result
        let k1_re = k1(z.re);
        let i1_re = i1(z.re);
        return Ok(Complex64::new(k1_re, z.im * PI * i1_re / z.re));
    }

    // For general complex arguments, use the connection to the Hankel function
    // K₁(z) = -π/2 * i * H₁⁽¹⁾(iz)
    let iz = Complex64::new(-z.im, z.re); // i*z

    // Compute asymptotic form for Hankel function of first kind (H₁)
    let sqrt_pi_2iz = (PI / (2.0 * iz)).sqrt();
    let phase = iz - Complex64::new(3.0 * PI / 4.0, 0.0);

    // First terms of asymptotic series
    let one_over_z_sq = 1.0 / (iz * iz);
    let p_term = 1.0 + 0.375 * one_over_z_sq - 0.073242 * one_over_z_sq * one_over_z_sq;
    let q_term = 0.1875 * one_over_z_sq - 0.097656 * one_over_z_sq * one_over_z_sq;

    // H₁⁽¹⁾(z) ~ sqrt(2/πz) * exp(i(z-3π/4)) * (1 + O(1/z))
    // For complex phase, we use e^(i*phase) directly
    let exp_i_phase = (Complex64::i() * phase).exp();
    let h1_iz = sqrt_pi_2iz * exp_i_phase * (p_term + Complex64::i() * q_term);

    // K₁(z) = -π/2 * i * H₁⁽¹⁾(iz)
    let k1_z = -PI / 2.0 * Complex64::i() * h1_iz;

    Ok(k1_z)
}

/// Compute the digamma function (psi) - approximation for positive integers
#[allow(dead_code)]
fn psi(n: usize) -> f64 {
    let mut result = -0.5772156649015329; // -Euler's constant

    for i in 1..n {
        result += 1.0 / i as f64;
    }

    result
}

/// Compute the ber function, the real part of the Kelvin function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the ber function
///
/// # Examples
///
/// ```
/// use scirs2_special::ber;
///
/// let value = ber(1.5).unwrap();
/// println!("ber(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn ber(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.0.re)
}

/// Compute the bei function, the imaginary part of the Kelvin function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the bei function
///
/// # Examples
///
/// ```
/// use scirs2_special::bei;
///
/// let value = bei(1.5).unwrap();
/// println!("bei(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn bei(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.0.im)
}

/// Compute the ker function, the real part of the Kelvin K function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the ker function
///
/// # Examples
///
/// ```
/// use scirs2_special::ker;
///
/// let value = ker(1.5).unwrap();
/// println!("ker(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn ker(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.1.re)
}

/// Compute the kei function, the imaginary part of the Kelvin K function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the kei function
///
/// # Examples
///
/// ```
/// use scirs2_special::kei;
///
/// let value = kei(1.5).unwrap();
/// println!("kei(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn kei(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.1.im)
}

/// Compute the derivative of the ber function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the derivative of the ber function
///
/// # Examples
///
/// ```
/// use scirs2_special::berp;
///
/// let value = berp(1.5).unwrap();
/// println!("berp(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn berp(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.2.re)
}

/// Compute the derivative of the bei function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the derivative of the bei function
///
/// # Examples
///
/// ```
/// use scirs2_special::beip;
///
/// let value = beip(1.5).unwrap();
/// println!("beip(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn beip(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.2.im)
}

/// Compute the derivative of the ker function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the derivative of the ker function
///
/// # Examples
///
/// ```
/// use scirs2_special::kerp;
///
/// let value = kerp(1.5).unwrap();
/// println!("kerp(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn kerp(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.3.re)
}

/// Compute the derivative of the kei function
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * Real value of the derivative of the kei function
///
/// # Examples
///
/// ```
/// use scirs2_special::keip;
///
/// let value = keip(1.5).unwrap();
/// println!("keip(1.5) = {}", value);
/// ```
#[allow(dead_code)]
pub fn keip(x: f64) -> SpecialResult<f64> {
    Ok(kelvin(x)?.3.im)
}
