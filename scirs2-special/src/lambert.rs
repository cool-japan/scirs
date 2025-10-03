//! Lambert W function implementation
//!
//! The Lambert W function is the inverse function of `w * exp(w)`.
//! It's a multivalued function with infinitely many branches,
//! indexed by the integer k.
//!
//! This implementation follows the approach from SciPy's lambertw function,
//! using Halley's iteration with initial guesses from asymptotic approximations.

use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;
use std::f64::consts::{E, PI};

use crate::error::{SpecialError, SpecialResult};

const _EXPN1: f64 = E; // e (prefixed with _ since not currently used)
const EXPN1_INV: f64 = 1.0 / E; // 1/e
const TWO_PI: f64 = 2.0 * PI;
const MAX_ITERATIONS: usize = 100;

/// Lambert W function for real and complex arguments.
///
/// The Lambert W function W(z) is defined as the inverse function of w * exp(w).
/// In other words, the value of W(z) is such that z = W(z) * exp(W(z)) for any complex number z.
///
/// # Arguments
///
/// * `z` - Complex input argument
/// * `k` - Branch index (integer)
/// * `tol` - Tolerance for convergence (default = 1e-8)
///
/// # Returns
///
/// * `Complex64` - The value of W(z) on branch k
///
/// # Examples
///
/// ```
/// use scirs2_special::lambert_w;
/// use scirs2_core::numeric::Complex64;
///
/// let w = lambert_w(Complex64::new(1.0, 0.0), 0, 1e-8).unwrap();
/// assert!((w - Complex64::new(0.56714329040978384, 0.0)).norm() < 1e-10);
///
/// // Verify that w * exp(w) = z
/// let z = Complex64::new(1.0, 0.0);
/// let w_exp_w = w * w.exp();
/// assert!((w_exp_w - z).norm() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn lambert_w(z: Complex64, k: i32, tol: f64) -> SpecialResult<Complex64> {
    if z.is_nan() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    // Special cases for infinite inputs
    if z.is_infinite() {
        if k == 0 {
            return Ok(Complex64::new(f64::INFINITY, 0.0));
        } else if k == 1 {
            return Ok(Complex64::new(f64::INFINITY, TWO_PI));
        } else if k == -1 {
            return Ok(Complex64::new(f64::INFINITY, 3.0 * PI));
        } else {
            // For other branches with infinite input
            // This may not match SciPy for all k values, but it's a reasonable default
            let imag = (2.0 * k as f64 + 1.0) * PI;
            return Ok(Complex64::new(f64::INFINITY, imag));
        }
    }

    // For k=0 branch and very small inputs, the result is approximately the input
    if k == 0 && z.norm() < 1e-300 {
        return Ok(z);
    }

    // Special case for z = 0
    if z.is_zero() {
        if k == 0 {
            return Ok(Complex64::new(0.0, 0.0));
        } else {
            // All other branches have a logarithmic singularity at z = 0
            return Ok(Complex64::new(f64::NEG_INFINITY, 0.0));
        }
    }

    // Compute the initial guess based on the branch
    let mut w = initial_guess(z, k);

    // Halley's iteration to refine the result
    for _ in 0..MAX_ITERATIONS {
        // Handle extreme values of w carefully to avoid overflow
        if w.re > 700.0 {
            // For very large w.re, exp(w) would overflow, so handle specially
            return Ok(w); // At these extreme values, further refinement is unlikely to help
        }

        let ew = w.exp();
        let wew = w * ew;
        let wewz = wew - z;

        // Check if we've converged
        // Using both absolute and relative tolerance for better stability
        let abs_tol = tol.max(1e-15);
        let rel_tol = tol * w.norm().max(1.0);
        if wewz.norm() < abs_tol || wewz.norm() < rel_tol {
            break;
        }

        // Compute the next iteration using Halley's method
        // The formula is: w_next = w - f(w)/f'(w) * (1 + f(w)*f''(w)/(2*f'(w)^2))
        // where f(w) = w*e^w - z
        let w1 = w + Complex64::new(1.0, 0.0);
        let w1ew = w1 * ew;
        let denominator =
            w1ew - (w + Complex64::new(2.0, 0.0)) * wewz / (Complex64::new(2.0, 0.0) * w1);

        // More robust handling of potential numerical issues
        if denominator.norm() < 1e-15 {
            // In case of near-zero denominator, use a dampened step
            let safe_step = Complex64::new(0.1, 0.0)
                * if w.norm() > 1.0 {
                    w / w.norm()
                } else {
                    Complex64::new(1.0, 0.0)
                };
            w -= safe_step;
        } else {
            let delta = wewz / denominator;

            // Limit step size to prevent overshooting
            let delta_norm = delta.norm();
            if delta_norm > 10.0 {
                w -= delta * (10.0 / delta_norm);
            } else {
                w -= delta;
            }
        }
    }

    Ok(w)
}

/// Initial guess for the Lambert W function.
///
/// Uses different approximations depending on the branch and region:
/// 1. Near the branch point at -1/e
/// 2. Asymptotic series for large |z|
/// 3. Pade approximation around 0 for the principal branch
/// 4. General approximation for other cases
#[allow(dead_code)]
fn initial_guess(z: Complex64, k: i32) -> Complex64 {
    // Near the branch point at -1/e for k=0 or k=-1
    if (z + EXPN1_INV).norm() < 0.3 && (k == 0 || k == -1) {
        let p = (2.0 * (E * z + 1.0)).sqrt();
        if k == 0 {
            return Complex64::new(-1.0, 0.0) + p - p.powi(2) / 3.0;
        } else {
            return Complex64::new(-1.0, 0.0) - p - p.powi(2) / 3.0;
        }
    }

    // For large |z|, use the asymptotic series
    if z.norm() > 3.0 {
        let mut w = z.ln();
        if w.is_zero() {
            // Avoid division by zero
            w = Complex64::new(1e-300, 0.0);
        }
        w -= w.ln().ln();

        // Adjust for non-principal branches
        if k != 0 {
            w += Complex64::new(0.0, TWO_PI * k as f64);
        }
        return w;
    }

    // Use Pade approximation for principal branch near 0
    if k == 0 && z.norm() < 1.0 {
        // Coefficients from lambertw_pade in SciPy
        let p = [1.0, 2.331_643_981_597_124, 1.812_187_885_639_363_4, 0.1];
        let q = [1.0, 3.331_643_981_597_124, 1.812_187_885_639_363_4];

        let numerator = p[0] + z * (p[1] + z * (p[2] + z * p[3]));
        let denominator = q[0] + z * (q[1] + z * q[2]);

        return numerator / denominator;
    }

    // For other cases, use a general approximation
    let mut w = z.ln();
    if w.is_zero() {
        // Avoid division by zero
        w = Complex64::new(1e-300, 0.0);
    }

    // For non-principal branches, add the branch offset
    if k != 0 {
        w += Complex64::new(0.0, TWO_PI * k as f64);
    }

    w
}

/// Lambert W function for real arguments on the principal branch (k=0).
///
/// # Arguments
///
/// * `x` - Real input value
/// * `tol` - Tolerance for convergence (default = 1e-8)
///
/// # Returns
///
/// * Result containing the real value of W(x) or a complex value when the result is not real
///
/// # Examples
///
/// ```
/// use scirs2_special::lambert_w_real;
///
/// let w = lambert_w_real(1.0, 1e-8).unwrap();
/// assert!((w - 0.56714329040978384).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn lambert_w_real(x: f64, tol: f64) -> SpecialResult<f64> {
    let result = lambert_w(Complex64::new(x, 0.0), 0, tol)?;

    // For the principal branch (k=0), the result is real when x > -1/e
    if x > -EXPN1_INV && result.im.abs() < 1e-15 {
        Ok(result.re)
    } else {
        Err(SpecialError::DomainError(format!(
            "Lambert W function gives a complex result for x={x}"
        )))
    }
}
