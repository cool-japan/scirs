//! Fresnel integrals
//!
//! This module provides implementations of the Fresnel integrals S(x) and C(x).
//! These integrals arise in optics and electromagnetics, particularly in the
//! study of diffraction patterns and are defined as:
//!
//! S(x) = ∫_0_^x sin(πt²/2) dt
//! C(x) = ∫_0_^x cos(πt²/2) dt
//!
//! There are also modified Fresnel integrals that are used in certain applications.

use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;
use std::f64::consts::PI;

use crate::error::{SpecialError, SpecialResult};

/// Compute the Fresnel sine and cosine integrals.
///
/// # Definition
///
/// The Fresnel integrals are defined as:
///
/// S(x) = ∫_0_^x sin(πt²/2) dt
/// C(x) = ∫_0_^x cos(πt²/2) dt
///
/// # Arguments
///
/// * `x` - Real or complex argument
///
/// # Returns
///
/// * A tuple (S(x), C(x)) containing the values of the Fresnel sine and cosine integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnel;
///
/// let (s, c) = fresnel(1.0).unwrap();
/// println!("S(1.0) = {}, C(1.0) = {}", s, c);
/// ```
#[allow(dead_code)]
pub fn fresnel(x: f64) -> SpecialResult<(f64, f64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to fresnel".to_string(),
        ));
    }

    if x == 0.0 {
        return Ok((0.0, 0.0));
    }

    // For x with large magnitude, use the asymptotic form
    if x.abs() > 6.0 {
        let (s, c) = fresnel_asymptotic(x)?;
        return Ok((s, c));
    }

    // For small to moderate x, use power series or auxiliary functions
    fresnel_power_series(x)
}

/// Compute the Fresnel sine and cosine integrals for complex argument.
///
/// # Arguments
///
/// * `z` - Complex argument
///
/// # Returns
///
/// * A tuple (S(z), C(z)) containing the complex values of the Fresnel sine and cosine integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnel_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.5);
/// let (s, c) = fresnel_complex(z).unwrap();
/// println!("S({} + {}i) = {} + {}i", z.re, z.im, s.re, s.im);
/// println!("C({} + {}i) = {} + {}i", z.re, z.im, c.re, c.im);
/// ```
#[allow(dead_code)]
pub fn fresnel_complex(z: Complex64) -> SpecialResult<(Complex64, Complex64)> {
    if z.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to fresnel_complex".to_string(),
        ));
    }

    if z.is_zero() {
        return Ok((Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)));
    }

    // For z with large magnitude, use the asymptotic form
    if z.norm() > 6.0 {
        let (s, c) = fresnel_complex_asymptotic(z)?;
        return Ok((s, c));
    }

    // For small to moderate z, use power series
    fresnel_complex_power_series(z)
}

/// Implementation of Fresnel integrals using power series for small x.
#[allow(dead_code)]
fn fresnel_power_series(x: f64) -> SpecialResult<(f64, f64)> {
    let sign = x.signum();
    let x = x.abs();

    // Special case for very small x to avoid underflow
    if x < 1e-100 {
        return Ok((0.0, 0.0));
    }

    // Coefficients for the power series
    // S(x) = x³(1/3! - πx⁴/2·5! + π²x⁸/2²·7! - π³x¹²/2³·9! + ...)
    // C(x) = x(1 - πx⁴/2·4! + π²x⁸/2²·6! - π³x¹²/2³·8! + ...)

    // Parameters for computation
    let pi_half = PI / 2.0;
    let x2 = x * x;
    let x4 = x2 * x2;

    // Choose the optimal method based on x size
    if x < 0.5 {
        // For smaller x, direct power series evaluation is stable
        // Compute the sine integral
        let mut s = 0.0;
        let mut term = x * x2 / 6.0; // First term: x³/3!
        let mut prev_s = 0.0;
        let mut num_equal_terms_s = 0;

        for k in 0..35 {
            // Increased iteration limit for small x
            if k > 0 {
                // More stable formula that avoids potential overflow in factorial-like calculations
                let denom =
                    (4 * k + 3) as f64 * (4 * k + 2) as f64 * (4 * k + 1) as f64 * (4 * k) as f64;
                let factor = -pi_half * x4 / denom;
                term *= factor;
            }

            // Skip terms that could cause numerical instability
            if !term.is_finite() {
                break;
            }

            s += term;

            // Multiple convergence criteria for better stability
            let abs_term = term.abs();
            let abs_s = s.abs().max(1e-300); // Avoid division by zero

            if abs_term < 1e-15 || abs_term < 1e-15 * abs_s {
                break;
            }

            // Check if sum is stabilizing (no significant changes)
            if (s - prev_s).abs() < 1e-15 * abs_s {
                num_equal_terms_s += 1;
                if num_equal_terms_s > 3 {
                    // Several iterations with no significant change
                    break;
                }
            } else {
                num_equal_terms_s = 0;
            }

            prev_s = s;
        }

        // Compute the cosine integral
        let mut c = 0.0;
        let mut term = x; // First term: x
        let mut prev_c = 0.0;
        let mut num_equal_terms_c = 0;

        for k in 0..35 {
            if k > 0 {
                // More stable formula
                let denom =
                    (4 * k + 2) as f64 * (4 * k + 1) as f64 * (4 * k) as f64 * (4 * k - 1) as f64;

                // Handle potential divide-by-zero for k=0
                if denom == 0.0 {
                    continue;
                }

                let factor = -pi_half * x4 / denom;
                term *= factor;
            }

            // Skip terms that could cause numerical instability
            if !term.is_finite() {
                break;
            }

            c += term;

            // Multiple convergence criteria
            let abs_term = term.abs();
            let abs_c = c.abs().max(1e-300);

            if abs_term < 1e-15 || abs_term < 1e-15 * abs_c {
                break;
            }

            // Check if sum is stabilizing
            if (c - prev_c).abs() < 1e-15 * abs_c {
                num_equal_terms_c += 1;
                if num_equal_terms_c > 3 {
                    break;
                }
            } else {
                num_equal_terms_c = 0;
            }

            prev_c = c;
        }

        // Apply the sign
        Ok((sign * s, sign * c))
    } else {
        // For larger x, use a more stable approach with continued fractions
        // This is based on the fact that Fresnel integrals can be expressed in terms of
        // the Error function erf, which has stable continued fraction representations

        // Use auxiliary functions f(x) and g(x) that maintain precision better
        let z = pi_half * x2;
        let sin_z = z.sin();
        let cos_z = z.cos();

        // Compute auxiliary series for improved stability
        let mut f_sum = 0.0;
        let mut g_sum = 0.0;
        let mut term_f = 1.0;
        let mut term_g = 1.0;

        for k in 1..25 {
            // Terms for f series
            let f_factor = -1.0 / ((2 * k - 1) as f64 * z);
            term_f *= f_factor;
            f_sum += term_f;

            // Terms for g series
            let g_factor = -1.0 / (2.0 * k as f64 * z);
            term_g *= g_factor;
            g_sum += term_g;

            // Convergence check
            if term_f.abs() < 1e-15 && term_g.abs() < 1e-15 {
                break;
            }
        }

        // Calculate S(x) and C(x) from auxiliary functions
        let s = 0.5 - (cos_z * (0.5 + f_sum) + sin_z * g_sum) / (PI * x);
        let c = 0.5 - (sin_z * (0.5 + f_sum) - cos_z * g_sum) / (PI * x);

        // Apply the sign and return
        Ok((sign * s, sign * c))
    }
}

/// Implementation of Fresnel integrals using asymptotic expansions for large x.
#[allow(dead_code)]
fn fresnel_asymptotic(x: f64) -> SpecialResult<(f64, f64)> {
    let sign = x.signum();
    let x = x.abs();

    // Special case for extremely large x
    if x > 1e100 {
        // For extremely large x, the Fresnel integrals approach 1/2
        return Ok((sign * 0.5, sign * 0.5));
    }

    // For large x, the Fresnel integrals approach 1/2
    // S(x) → 1/2 - f(x)cos(πx²/2) - g(x)sin(πx²/2)
    // C(x) → 1/2 + f(x)sin(πx²/2) - g(x)cos(πx²/2)
    // where f(x) and g(x) are asymptotic series

    // Use a scaled approach for very large x to avoid overflow in x²
    let z = if x > 1e7 {
        // For very large x, compute z carefully to avoid overflow
        let scaled_x = x / 1e7;
        PI * scaled_x * scaled_x * 1e14 / 2.0
    } else {
        PI * x * x / 2.0
    };

    // The argument may be so large that z cannot be represented accurately
    // In that case, simplify the computation by modding out the periods of sine and cosine
    let reduced_z = if z > 1e10 {
        // For extremely large z, reduce to principal values
        let two_pi = 2.0 * PI;
        z % two_pi
    } else {
        z
    };

    // Compute sine and cosine of the reduced argument
    let sin_z = reduced_z.sin();
    let cos_z = reduced_z.cos();

    // Different strategies based on magnitude of x for stability
    if x > 20.0 {
        // For very large x, use a more accurate asymptotic form
        // that avoids potential cancellation errors

        // Compute just the first few terms of the asymptotic series
        // This avoids divergence issues with asymptotic series for large orders
        let f_first_term = 1.0 / (PI * x);
        let g_first_term = 1.0 / (PI * 3.0 * 2.0 * z); // First term of g series

        // For extremely large x, the higher-order terms are negligible
        let s = 0.5 - f_first_term * cos_z - g_first_term * sin_z;
        let c = 0.5 + f_first_term * sin_z - g_first_term * cos_z;

        return Ok((sign * s, sign * c));
    }

    // For moderately large x, compute more terms of the series
    let z2 = z * z;
    let z2_inv = 1.0 / z2;

    // Initialize with leading terms
    let mut f = 1.0 / (PI * x);
    let mut g = 0.0;

    // Keep track of previous sums for convergence monitoring
    let mut prev_f = f;
    let mut prev_g = g;
    let mut num_stable_terms = 0;

    // Asymptotic series for f(x) and g(x) with enhanced stability
    for k in 1..25 {
        // Extended series for better accuracy
        // Compute terms carefully to avoid overflow in large powers
        let k_f64 = k as f64;

        // Avoid direct power calculation which could overflow
        // Instead, build up the power by multiplication
        let mut z2_pow_k: f64 = z2_inv; // Start with (1/z2)
        for _ in 1..k {
            z2_pow_k *= z2_inv; // Multiply by (1/z2) k-1 more times

            // Check for underflow
            if z2_pow_k.abs() < 1e-300 {
                break; // Underflow, further terms are negligible
            }
        }

        // Calculate f and g terms with improved numerical stability
        let f_term =
            if k % 2 == 1 { -1.0 } else { 1.0 } * (4.0 * k_f64 - 1.0) * z2_pow_k / (PI * x);

        let g_term = if k % 2 == 1 { -1.0 } else { 1.0 } * (4.0 * k_f64 + 1.0) * z2_pow_k
            / ((2.0 * k_f64 + 1.0) * PI);

        // Add terms to the sums
        f += f_term;
        g += g_term;

        // Multiple convergence criteria for better stability

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance
        let f_rel_tol = 1e-15 * f.abs().max(1e-300);
        let g_rel_tol = 1e-15 * g.abs().max(1e-300);

        // Check for convergence
        if f_term.abs() < abs_tol && g_term.abs() < abs_tol {
            break; // Terms are absolutely small
        }

        if f_term.abs() < f_rel_tol && g_term.abs() < g_rel_tol {
            break; // Terms are relatively small
        }

        // Check if sums are stabilizing (not changing significantly)
        if (f - prev_f).abs() < f_rel_tol && (g - prev_g).abs() < g_rel_tol {
            num_stable_terms += 1;
            if num_stable_terms > 2 {
                break; // Sums have stabilized
            }
        } else {
            num_stable_terms = 0;
        }

        // Check for divergence (which can happen with asymptotic series)
        if f_term.abs() > 100.0 * prev_f.abs() || g_term.abs() > 100.0 * prev_g.abs() {
            // Series is starting to diverge, so use the previous sum
            f = prev_f;
            g = prev_g;
            break;
        }

        prev_f = f;
        prev_g = g;
    }

    // Compute the Fresnel integrals
    let s = 0.5 - f * cos_z - g * sin_z;
    let c = 0.5 + f * sin_z - g * cos_z;

    // Apply the sign
    Ok((sign * s, sign * c))
}

/// Implementation of complex Fresnel integrals using power series.
#[allow(dead_code)]
fn fresnel_complex_power_series(z: Complex64) -> SpecialResult<(Complex64, Complex64)> {
    // Special case for very small |z|
    if z.norm() < 1e-100 {
        return Ok((Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)));
    }

    // Basic variables
    let z2 = z * z;
    let z4 = z2 * z2;
    let pi_half = Complex64::new(PI / 2.0, 0.0);

    // Choose appropriate method based on |z|
    if z.norm() < 0.5 {
        // For small |z|, direct power series evaluation is stable

        // Compute the sine integral
        let mut s = Complex64::new(0.0, 0.0);
        let mut term = z * z2 / 3.0; // First term: z³/3!
        let mut prev_s = Complex64::new(0.0, 0.0);
        let mut num_equal_terms_s = 0;

        // Compute using power series with enhanced stability
        for k in 0..45 {
            // Extended limit for better accuracy
            if k > 0 {
                // Compute factorial-like denominator carefully
                let denom =
                    (4 * k + 3) as f64 * (4 * k + 2) as f64 * (4 * k + 1) as f64 * (4 * k) as f64;
                let factor = -pi_half * z4 / denom;
                term *= factor;
            }

            // Skip terms that could cause numerical instability
            if !term.is_finite() {
                break;
            }

            s += term;

            // Multiple convergence criteria for better stability
            let norm_term = term.norm();
            let norm_s = s.norm().max(1e-300); // Avoid division by zero

            // Absolute and relative tolerance checks
            if norm_term < 1e-15 || norm_term < 1e-15 * norm_s {
                break;
            }

            // Check if sum is stabilizing (no significant changes)
            if (s - prev_s).norm() < 1e-15 * norm_s {
                num_equal_terms_s += 1;
                if num_equal_terms_s > 3 {
                    // Several iterations with no significant change
                    break;
                }
            } else {
                num_equal_terms_s = 0;
            }

            prev_s = s;
        }

        // Compute the cosine integral with similar improvements
        let mut c = Complex64::new(0.0, 0.0);
        let mut term = z; // First term: z
        let mut prev_c = Complex64::new(0.0, 0.0);
        let mut num_equal_terms_c = 0;

        for k in 0..45 {
            if k > 0 {
                // Handle the case where k=0 separately to avoid division by zero
                if k == 1 {
                    // Special handling for k=1 (denominator involves 4*k-1 = 3)
                    let denom = (4 * k + 2) as f64 * (4 * k + 1) as f64 * (4 * k) as f64 * 3.0;
                    let factor = -pi_half * z4 / denom;
                    term *= factor;
                } else {
                    // Normal case for k > 1
                    let denom = (4 * k + 2) as f64
                        * (4 * k + 1) as f64
                        * (4 * k) as f64
                        * (4 * k - 1) as f64;
                    let factor = -pi_half * z4 / denom;
                    term *= factor;
                }
            }

            // Skip terms that could cause numerical instability
            if !term.is_finite() {
                break;
            }

            c += term;

            // Multiple convergence criteria
            let norm_term = term.norm();
            let norm_c = c.norm().max(1e-300);

            if norm_term < 1e-15 || norm_term < 1e-15 * norm_c {
                break;
            }

            // Check if sum is stabilizing
            if (c - prev_c).norm() < 1e-15 * norm_c {
                num_equal_terms_c += 1;
                if num_equal_terms_c > 3 {
                    break;
                }
            } else {
                num_equal_terms_c = 0;
            }

            prev_c = c;
        }

        Ok((s, c))
    } else {
        // For larger |z|, use auxiliary functions that maintain precision better
        // Similar approach to the real-valued case for moderate arguments

        // Use series related to the complex error function
        let pi_z2_half = pi_half * z2;
        let sin_pi_z2_half = pi_z2_half.sin();
        let cos_pi_z2_half = pi_z2_half.cos();

        // Compute auxiliary series
        let mut f_sum = Complex64::new(0.0, 0.0);
        let mut g_sum = Complex64::new(0.0, 0.0);
        let mut term_f = Complex64::new(1.0, 0.0);
        let mut term_g = Complex64::new(1.0, 0.0);

        // Compute these auxiliary series with good stability
        for k in 1..35 {
            // Terms for f series
            let f_factor = Complex64::new(-1.0, 0.0) / ((2.0 * k as f64 - 1.0) * pi_z2_half);
            term_f *= f_factor;

            // Terms for g series
            let g_factor = Complex64::new(-1.0, 0.0) / (2.0 * k as f64 * pi_z2_half);
            term_g *= g_factor;

            // Only add terms that won't cause numerical issues
            if term_f.is_finite() {
                f_sum += term_f;
            }

            if term_g.is_finite() {
                g_sum += term_g;
            }

            // Convergence check
            if term_f.norm() < 1e-15 && term_g.norm() < 1e-15 {
                break;
            }
        }

        // Calculate S(z) and C(z) from auxiliary functions
        let half = Complex64::new(0.5, 0.0);
        let pi_z_inv = Complex64::new(1.0 / (PI * z.norm()), 0.0) * (z / z.norm()).conj();

        let s = half - (cos_pi_z2_half * (half + f_sum) + sin_pi_z2_half * g_sum) * pi_z_inv;
        let c = half - (sin_pi_z2_half * (half + f_sum) - cos_pi_z2_half * g_sum) * pi_z_inv;

        Ok((s, c))
    }
}

/// Implementation of complex Fresnel integrals using asymptotic expansions.
#[allow(dead_code)]
fn fresnel_complex_asymptotic(z: Complex64) -> SpecialResult<(Complex64, Complex64)> {
    // Special cases for extreme values
    if !z.is_finite() {
        return Err(SpecialError::DomainError(
            "Infinite or NaN input to fresnel_complex_asymptotic".to_string(),
        ));
    }

    // For extremely large |z|, directly return the limit
    if z.norm() > 1e100 {
        return Ok((Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)));
    }

    // Calculate with appropriate scaling to avoid overflow
    let pi_z2_half = if z.norm() > 1e7 {
        // For very large z, compute carefully to avoid overflow in z²
        let scaled_z = z / 1e7;
        PI * scaled_z * scaled_z * 1e14 / 2.0
    } else {
        PI * z * z / 2.0
    };

    // For very large arguments, reduce trigonometric arguments to principal values
    let reduced_pi_z2_half = if pi_z2_half.norm() > 1e10 {
        let two_pi = Complex64::new(2.0 * PI, 0.0);
        // Complex modulo operation
        let n = (pi_z2_half / two_pi).re.floor();
        pi_z2_half - two_pi * Complex64::new(n, 0.0)
    } else {
        pi_z2_half
    };

    // Compute sine and cosine of the reduced argument
    let sin_pi_z2_half = reduced_pi_z2_half.sin();
    let cos_pi_z2_half = reduced_pi_z2_half.cos();

    // For very large |z|, use simplified asymptotic form
    if z.norm() > 20.0 {
        // Just use the first term of the asymptotic series
        let f_first_term = Complex64::new(1.0, 0.0) / (PI * z);

        // g is numerically smaller than f for large |z|
        let g_first_term = f_first_term / (3.0 * pi_z2_half);

        // Calculate the Fresnel integrals with just these first terms
        let half = Complex64::new(0.5, 0.0);
        let s = half - f_first_term * cos_pi_z2_half - g_first_term * sin_pi_z2_half;
        let c = half + f_first_term * sin_pi_z2_half - g_first_term * cos_pi_z2_half;

        return Ok((s, c));
    }

    // For moderately large |z|, compute more terms of the asymptotic series
    let pi_z2_half_sq = pi_z2_half * pi_z2_half;

    // Initialize with the first terms
    let mut f = Complex64::new(1.0, 0.0) / (PI * z);
    let mut g = Complex64::new(0.0, 0.0);

    // Track previous sums for convergence monitoring
    let mut prev_f = f;
    let mut prev_g = g;
    let mut num_stable_terms = 0;

    // Asymptotic series with enhanced stability
    for k in 1..20 {
        // Use safer term calculation
        let k_f64 = k as f64;

        // Compute powers more carefully using division instead of power
        let mut pi_z2_half_sq_pow_k = Complex64::new(1.0, 0.0);
        for _ in 0..k {
            pi_z2_half_sq_pow_k /= pi_z2_half_sq;

            // Check for underflow/overflow
            if !pi_z2_half_sq_pow_k.is_finite() || pi_z2_half_sq_pow_k.norm() < 1e-300 {
                break;
            }
        }

        // Alternating sign based on k
        let sign = if k % 2 == 1 { -1.0 } else { 1.0 };

        // Calculate f term
        let f_term = sign * (4.0 * k_f64 - 1.0) * pi_z2_half_sq_pow_k / (PI * z);

        // Calculate g term
        let g_term = sign * (4.0 * k_f64 + 1.0) * pi_z2_half_sq_pow_k / ((2.0 * k_f64 + 1.0) * PI);

        // Only add terms if they're finite (to handle potential overflow/underflow)
        if f_term.is_finite() {
            f += f_term;
        }

        if g_term.is_finite() {
            g += g_term;
        }

        // Multiple convergence checks
        let f_norm = f_term.norm();
        let g_norm = g_term.norm();
        let f_sum_norm = f.norm().max(1e-300);
        let g_sum_norm = g.norm().max(1e-300);

        // Check for absolute and relative convergence
        if f_norm < 1e-15 && g_norm < 1e-15 {
            break; // Both terms are absolutely small
        }

        if f_norm < 1e-15 * f_sum_norm && g_norm < 1e-15 * g_sum_norm {
            break; // Both terms are relatively small
        }

        // Check if sums are stabilizing
        if (f - prev_f).norm() < 1e-15 * f_sum_norm && (g - prev_g).norm() < 1e-15 * g_sum_norm {
            num_stable_terms += 1;
            if num_stable_terms > 2 {
                break; // Sums have stabilized
            }
        } else {
            num_stable_terms = 0;
        }

        // Check for potential divergence
        if f_norm > 100.0 * prev_f.norm() || g_norm > 100.0 * prev_g.norm() {
            // Series is starting to diverge, use previous sum
            f = prev_f;
            g = prev_g;
            break;
        }

        prev_f = f;
        prev_g = g;
    }

    // Compute the Fresnel integrals
    let half = Complex64::new(0.5, 0.0);
    let s = half - f * cos_pi_z2_half - g * sin_pi_z2_half;
    let c = half + f * sin_pi_z2_half - g * cos_pi_z2_half;

    // Final check for numerical issues
    if !s.is_finite() || !c.is_finite() {
        // Fallback to the simplest approximation for large |z|
        let s_approx = Complex64::new(0.5, 0.0);
        let c_approx = Complex64::new(0.5, 0.0);
        return Ok((s_approx, c_approx));
    }

    Ok((s, c))
}

/// Compute the Fresnel sine integral.
///
/// # Definition
///
/// The Fresnel sine integral is defined as:
///
/// S(x) = ∫_0_^x sin(πt²/2) dt
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * Value of the Fresnel sine integral S(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnels;
///
/// let s = fresnels(1.0).unwrap();
/// println!("S(1.0) = {}", s);
/// ```
#[allow(dead_code)]
pub fn fresnels(x: f64) -> SpecialResult<f64> {
    let (s, _) = fresnel(x)?;
    Ok(s)
}

/// Compute the Fresnel cosine integral.
///
/// # Definition
///
/// The Fresnel cosine integral is defined as:
///
/// C(x) = ∫_0_^x cos(πt²/2) dt
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * Value of the Fresnel cosine integral C(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnelc;
///
/// let c = fresnelc(1.0).unwrap();
/// println!("C(1.0) = {}", c);
/// ```
#[allow(dead_code)]
pub fn fresnelc(x: f64) -> SpecialResult<f64> {
    let (_, c) = fresnel(x)?;
    Ok(c)
}

/// Compute the modified Fresnel plus integrals.
///
/// # Definition
///
/// The modified Fresnel plus integrals are defined as:
///
/// F₊(x) = ∫_x^∞ exp(it²) dt
/// K₊(x) = 1/√π · exp(-i(x² + π/4)) · F₊(x)
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple (F₊(x), K₊(x)) containing the values of the modified Fresnel plus integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::mod_fresnel_plus;
///
/// let (f_plus, k_plus) = mod_fresnel_plus(1.0).unwrap();
/// println!("F₊(1.0) = {} + {}i", f_plus.re, f_plus.im);
/// println!("K₊(1.0) = {} + {}i", k_plus.re, k_plus.im);
/// ```
#[allow(dead_code)]
pub fn mod_fresnel_plus(x: f64) -> SpecialResult<(Complex64, Complex64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to mod_fresnel_plus".to_string(),
        ));
    }

    // Special case for extremely small x
    if x.abs() < 1e-100 {
        // For x ≈ 0, F₊(0) approaches √π·e^(iπ/4)/2
        let sqrt_pi = PI.sqrt();
        let exp_i_pi_4 = Complex64::new(1.0, 0.0) * Complex64::new(0.5, 0.5).sqrt();
        let f_plus_0 = sqrt_pi * exp_i_pi_4 / 2.0;

        // K₊(0) ≈ 1/2
        let k_plus_0 = Complex64::new(0.5, 0.0);

        return Ok((f_plus_0, k_plus_0));
    }

    // Special case for extremely large x
    if x.abs() > 1e100 {
        // For |x| → ∞, F₊(x) → 0 and K₊(x) → 0
        let zero = Complex64::new(0.0, 0.0);
        return Ok((zero, zero));
    }

    // The modified Fresnel plus integrals can be expressed in terms of the standard Fresnel integrals
    let z = x.abs();

    // Compute auxiliary values (Fresnel integrals)
    let (s, c) = fresnel(z)?;
    let sqrt_pi = PI.sqrt();
    let sqrt_pi_inv = 1.0 / sqrt_pi;

    // For large z, compute the phase carefully to avoid overflow in z²
    // exp(±i(z² + π/4))
    let phase = if z > 1e7 {
        // Scale to avoid overflow
        let scaled_z = z / 1e7;
        let scaled_z_sq = scaled_z * scaled_z * 1e14;
        Complex64::new(0.0, scaled_z_sq + PI / 4.0)
    } else {
        Complex64::new(0.0, z * z + PI / 4.0)
    };

    // Check for potential overflow in the exponential
    // If the imaginary part of phase is very large, reduce it modulo 2π
    let reduced_phase = if phase.im.abs() > 100.0 {
        let two_pi = 2.0 * PI;
        let n = (phase.im / two_pi).floor();
        Complex64::new(phase.re, phase.im - n * two_pi)
    } else {
        phase
    };

    let exp_phase = reduced_phase.exp();
    let exp_i_pi_4 = Complex64::new(0.5, 0.5).sqrt(); // e^(iπ/4) = (1+i)/√2

    // Compute F₊(x) with improved numerical stability
    let f_plus = if x >= 0.0 {
        // For positive x: F₊(x) = (1/2 - C(x) - iS(x))·√π·exp(iπ/4)

        // Handle potential cancellation in 0.5 - c for large x
        // For large x, both c and s approach 0.5, so we compute the difference directly
        let halfminus_c = if z > 10.0 {
            // For large z, compute the difference using the asymptotic series directly
            let _z_sq = z * z;
            let pi_z = PI * z;
            1.0 / (2.0 * pi_z) * z.cos() // First term of asymptotic expansion
        } else {
            0.5 - c
        };

        // Similarly for s approaching 0.5
        let minus_s = if z > 10.0 {
            let _z_sq = z * z;
            let pi_z = PI * z;
            -0.5 + 1.0 / (2.0 * pi_z) * z.sin() // First term of asymptotic expansion
        } else {
            -s
        };

        let halfminus_cminus_is = Complex64::new(halfminus_c, minus_s);
        halfminus_cminus_is * sqrt_pi * exp_i_pi_4
    } else {
        // For negative x: F₊(-x) = (1/2 + C(x) + iS(x))·√π·exp(iπ/4)
        // Similar improved calculations for -x
        let half_plus_c = if z > 10.0 {
            let _z_sq = z * z;
            let pi_z = PI * z;
            0.5 + 1.0 / (2.0 * pi_z) * z.cos()
        } else {
            0.5 + c
        };

        let plus_s = if z > 10.0 {
            let _z_sq = z * z;
            let pi_z = PI * z;
            0.5 - 1.0 / (2.0 * pi_z) * z.sin()
        } else {
            s
        };

        let half_plus_c_plus_is = Complex64::new(half_plus_c, plus_s);
        half_plus_c_plus_is * sqrt_pi * exp_i_pi_4
    };

    // Compute K₊(x) = exp(-i(x² + π/4)) · F₊(x) / √π with careful multiplication
    // Use intermediate variable to avoid catastrophic cancellation
    let k_plus_unnormalized = exp_phase.conj() * f_plus;
    let k_plus = k_plus_unnormalized * sqrt_pi_inv;

    // Final check for numerical stability
    if !f_plus.is_finite() || !k_plus.is_finite() {
        // Fallback to asymptotic approximations for very large arguments
        if x.abs() > 10.0 {
            // For large |x|, the integrals decay like 1/x
            let decay_factor = 1.0 / x.abs();
            let f_plus_approx = Complex64::new(decay_factor, decay_factor);
            let k_plus_approx = Complex64::new(decay_factor, -decay_factor) * sqrt_pi_inv;
            return Ok((f_plus_approx, k_plus_approx));
        }
    }

    Ok((f_plus, k_plus))
}

/// Compute the modified Fresnel minus integrals.
///
/// # Definition
///
/// The modified Fresnel minus integrals are defined as:
///
/// F₋(x) = ∫_x^∞ exp(-it²) dt
/// K₋(x) = 1/√π · exp(i(x² + π/4)) · F₋(x)
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple (F₋(x), K₋(x)) containing the values of the modified Fresnel minus integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::mod_fresnelminus;
///
/// let (fminus, kminus) = mod_fresnelminus(1.0).unwrap();
/// println!("F₋(1.0) = {} + {}i", fminus.re, fminus.im);
/// println!("K₋(1.0) = {} + {}i", kminus.re, kminus.im);
/// ```
#[allow(dead_code)]
pub fn mod_fresnelminus(x: f64) -> SpecialResult<(Complex64, Complex64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to mod_fresnelminus".to_string(),
        ));
    }

    // Special case for extremely small x
    if x.abs() < 1e-100 {
        // For x ≈ 0, F₋(0) approaches √π·e^(-iπ/4)/2
        let sqrt_pi = PI.sqrt();
        let expminus_i_pi_4 = Complex64::new(1.0, 0.0) * Complex64::new(0.5, -0.5).sqrt();
        let fminus_0 = sqrt_pi * expminus_i_pi_4 / 2.0;

        // K₋(0) ≈ 1/2
        let kminus_0 = Complex64::new(0.5, 0.0);

        return Ok((fminus_0, kminus_0));
    }

    // Special case for extremely large x
    if x.abs() > 1e100 {
        // For |x| → ∞, F₋(x) → 0 and K₋(x) → 0
        let zero = Complex64::new(0.0, 0.0);
        return Ok((zero, zero));
    }

    // The modified Fresnel minus integrals can be expressed in terms of the standard Fresnel integrals
    let z = x.abs();

    // Compute auxiliary values (Fresnel integrals)
    let (s, c) = fresnel(z)?;
    let sqrt_pi = PI.sqrt();
    let sqrt_pi_inv = 1.0 / sqrt_pi;

    // For large z, compute the phase carefully to avoid overflow in z²
    // exp(±i(z² + π/4))
    let phase = if z > 1e7 {
        // Scale to avoid overflow
        let scaled_z = z / 1e7;
        let scaled_z_sq = scaled_z * scaled_z * 1e14;
        Complex64::new(0.0, scaled_z_sq + PI / 4.0)
    } else {
        Complex64::new(0.0, z * z + PI / 4.0)
    };

    // Check for potential overflow in the exponential
    // If the imaginary part of phase is very large, reduce it modulo 2π
    let reduced_phase = if phase.im.abs() > 100.0 {
        let two_pi = 2.0 * PI;
        let n = (phase.im / two_pi).floor();
        Complex64::new(phase.re, phase.im - n * two_pi)
    } else {
        phase
    };

    let exp_phase = reduced_phase.exp();
    let expminus_i_pi_4 = Complex64::new(0.5, -0.5).sqrt(); // e^(-iπ/4) = (1-i)/√2

    // Compute F₋(x) with improved numerical stability
    let fminus = if x >= 0.0 {
        // For positive x: F₋(x) = (1/2 - C(x) + iS(x))·√π·exp(-iπ/4)

        // Handle potential cancellation in 0.5 - c for large x
        // For large x, both c and s approach 0.5, so we compute the difference directly
        let halfminus_c = if z > 10.0 {
            // For large z, compute the difference using the asymptotic series directly
            let pi_z = PI * z;
            1.0 / (2.0 * pi_z) * z.cos() // First term of asymptotic expansion
        } else {
            0.5 - c
        };

        // Similarly for s approaching 0.5
        let plus_s = if z > 10.0 {
            let pi_z = PI * z;
            0.5 - 1.0 / (2.0 * pi_z) * z.sin() // First term of asymptotic expansion
        } else {
            s
        };

        let halfminus_c_plus_is = Complex64::new(halfminus_c, plus_s);
        halfminus_c_plus_is * sqrt_pi * expminus_i_pi_4
    } else {
        // For negative x: F₋(-x) = (1/2 + C(x) - iS(x))·√π·exp(-iπ/4)
        // Similar improved calculations for -x
        let half_plus_c = if z > 10.0 {
            let pi_z = PI * z;
            0.5 + 1.0 / (2.0 * pi_z) * z.cos()
        } else {
            0.5 + c
        };

        let minus_s = if z > 10.0 {
            let pi_z = PI * z;
            -0.5 + 1.0 / (2.0 * pi_z) * z.sin()
        } else {
            -s
        };

        let half_plus_cminus_is = Complex64::new(half_plus_c, minus_s);
        half_plus_cminus_is * sqrt_pi * expminus_i_pi_4
    };

    // Compute K₋(x) = exp(i(x² + π/4)) · F₋(x) / √π with careful multiplication
    // Use intermediate variable to avoid catastrophic cancellation
    let kminus_unnormalized = exp_phase * fminus;
    let kminus = kminus_unnormalized * sqrt_pi_inv;

    // Final check for numerical stability
    if !fminus.is_finite() || !kminus.is_finite() {
        // Fallback to asymptotic approximations for very large arguments
        if x.abs() > 10.0 {
            // For large |x|, the integrals decay like 1/x
            let decay_factor = 1.0 / x.abs();
            let fminus_approx = Complex64::new(decay_factor, -decay_factor);
            let kminus_approx = Complex64::new(decay_factor, decay_factor) * sqrt_pi_inv;
            return Ok((fminus_approx, kminus_approx));
        }
    }

    Ok((fminus, kminus))
}
