//! Parabolic Cylinder Functions
//!
//! This module provides implementations of the Parabolic Cylinder functions,
//! which are solutions to the differential equation:
//!
//! d²y/dx² + (v + 1/2 - x²/4)y = 0
//!
//! These functions have applications in quantum mechanics, wave propagation,
//! and other areas of mathematical physics.

use std::f64::consts::{PI, SQRT_2};
// use scirs2_core::numeric::Float;
// use scirs2_core::numeric::Complex64;

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{gamma, gammaln};

/// Parabolic cylinder function D_v(x) and its derivative.
///
/// The parabolic cylinder function D_v(x) is a solution to the differential equation:
/// d²y/dx² + (v + 1/2 - x²/4)y = 0
///
/// # Arguments
///
/// * `v` - Order parameter
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple containing (D_v(x), D_v'(x))
///
/// # Examples
///
/// ```
/// use scirs2_special::pbdv;
///
/// let (d, dp) = pbdv(1.0, 0.5).unwrap();
/// println!("D_1(0.5) = {}, D_1'(0.5) = {}", d, dp);
/// ```
#[allow(dead_code)]
pub fn pbdv(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if v.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to pbdv".to_string()));
    }

    // Special cases for v = 0, ±1, ±2
    if v == 0.0 {
        return pbdv_0(x);
    } else if v == 1.0 {
        return pbdv_1(x);
    } else if v == 2.0 {
        return pbdv_2(x);
    } else if v == -1.0 {
        return pbdv_m1(x);
    } else if v == -2.0 {
        return pbdv_m2(x);
    }

    // General case
    if v.floor() == v && (-20.0..=20.0).contains(&v) {
        // Integer case - use recurrence relations
        pbdv_integer(v as i32, x)
    } else {
        // General case - use series expansion
        pbdv_general(v, x)
    }
}

/// Implementation of D_0(x) for v = 0
#[allow(dead_code)]
fn pbdv_0(x: f64) -> SpecialResult<(f64, f64)> {
    // D_0(x) = e^(-x²/4)
    let x2 = x * x;
    let d = (-x2 / 4.0).exp();
    // D_0'(x) = -x/2 * e^(-x²/4)
    let dp = -x / 2.0 * d;

    Ok((d, dp))
}

/// Implementation of D_1(x) for v = 1
#[allow(dead_code)]
fn pbdv_1(x: f64) -> SpecialResult<(f64, f64)> {
    // D_1(x) = x * e^(-x²/4)
    let x2 = x * x;
    let exp_term = (-x2 / 4.0).exp();
    let d = x * exp_term;
    // D_1'(x) = (1 - x²/2) * e^(-x²/4)
    let dp = (1.0 - x2 / 2.0) * exp_term;

    Ok((d, dp))
}

/// Implementation of D_2(x) for v = 2
#[allow(dead_code)]
fn pbdv_2(x: f64) -> SpecialResult<(f64, f64)> {
    // D_2(x) = (x² - 2) * e^(-x²/4)
    let x2 = x * x;
    let exp_term = (-x2 / 4.0).exp();
    let d = (x2 - 2.0) * exp_term;
    // D_2'(x) = x * (x² - 4) * e^(-x²/4) / 2
    let dp = x * (x2 - 4.0) * exp_term / 2.0;

    Ok((d, dp))
}

/// Implementation of D_{-1}(x) for v = -1
#[allow(dead_code)]
fn pbdv_m1(x: f64) -> SpecialResult<(f64, f64)> {
    // D_{-1}(x) = (√(π/2) * e^(x²/4) * [1 - erf(x/√2)]) - x*e^(-x²/4)/2
    let x2 = x * x;
    let sqrt_pi_over_2 = (PI / 2.0).sqrt();
    let exp_pos = (x2 / 4.0).exp();
    let exp_neg = (-x2 / 4.0).exp();

    // Error function approximation
    let erf_val = erf_approx(x / SQRT_2);

    let d = sqrt_pi_over_2 * exp_pos * (1.0 - erf_val) - x * exp_neg / 2.0;

    // Derivative calculation
    let dp = x * d / 2.0 + exp_neg;

    Ok((d, dp))
}

/// Implementation of D_{-2}(x) for v = -2
#[allow(dead_code)]
fn pbdv_m2(x: f64) -> SpecialResult<(f64, f64)> {
    // D_{-2}(x) can be calculated using recurrence relations with D_{-1} and D_0
    let (d_m1, _) = pbdv_m1(x)?;
    let (d_0, _) = pbdv_0(x)?;

    let d = x * d_m1 - d_0;
    let dp = d_m1 - x * d / 2.0;

    Ok((d, dp))
}

/// Implementation of D_v(x) for integer v using recurrence relations
#[allow(dead_code)]
fn pbdv_integer(v: i32, x: f64) -> SpecialResult<(f64, f64)> {
    if v >= 0 {
        // Forward recurrence for positive v
        let (d0, d0p) = pbdv_0(x)?;
        let (d1, d1p) = pbdv_1(x)?;

        let mut d_prev = d0;
        let mut d = d1;
        let mut dp_prev = d0p;
        let mut dp = d1p;

        for k in 1..v {
            let d_next = x * d - k as f64 * d_prev;
            let dp_next = x * dp - k as f64 * dp_prev - d;

            d_prev = d;
            d = d_next;
            dp_prev = dp;
            dp = dp_next;
        }

        Ok((d, dp))
    } else {
        // Backward recurrence for negative v
        let (d0, d0p) = pbdv_0(x)?;
        let (dm1, dm1p) = pbdv_m1(x)?;

        let mut d_prev = dm1;
        let mut d = d0;
        let mut dp_prev = dm1p;
        let mut dp = d0p;

        for k in 0..(-v - 1) {
            let d_next = (x * d + d_prev) / (k as f64 + 1.0);
            let dp_next = (x * dp + dp_prev - d) / (k as f64 + 1.0);

            d_prev = d;
            d = d_next;
            dp_prev = dp;
            dp = dp_next;
        }

        Ok((d, dp))
    }
}

/// General implementation of D_v(x) for any v using series expansions
#[allow(dead_code)]
fn pbdv_general(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // For small |x|, use power series
    if x.abs() < 5.0 {
        return pbdv_series(v, x);
    }

    // For large positive x, use asymptotic expansion
    if x > 0.0 && x.abs() > (2.0 * v.abs()).max(20.0) {
        return pbdv_asymptotic_pos(v, x);
    }

    // For large negative x, use relationship with V_v
    if x < 0.0 && x.abs() > (2.0 * v.abs()).max(20.0) {
        return pbdv_asymptotic_neg(v, x);
    }

    // Fallback to numerical integration using the series expansion
    pbdv_series(v, x)
}

/// Power series implementation for D_v(x) with enhanced numerical stability
#[allow(dead_code)]
fn pbdv_series(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // Special case for x = 0
    if x == 0.0 {
        // When x = 0, D_v(0) = 2^(-v/2) * Γ((1+v)/2) / Γ(1)
        // And D_v'(0) = 0 for v > 0, and infinity for v <= 0
        let gamma_1_plus_v_half = gamma((1.0 + v) / 2.0);
        let gamma_1 = 1.0; // gamma(1.0) is always 1.0
        let d_at_zero = 2_f64.powf(-v / 2.0) * gamma_1_plus_v_half / gamma_1;
        let dp_at_zero = if v > 0.0 { 0.0 } else { f64::INFINITY };
        return Ok((d_at_zero, dp_at_zero));
    }

    // Special case for very large |x|
    if x.abs() > 100.0 {
        // For very large |x|, use the asymptotic form instead
        return if x > 0.0 {
            pbdv_asymptotic_pos(v, x)
        } else {
            pbdv_asymptotic_neg(v, x)
        };
    }

    // For normal range of x, use enhanced series method
    let x2 = x * x;

    // Use log-based computation for the exponential term to avoid underflow
    let log_exp_term = -x2 / 4.0;

    // Only compute the actual exponent if it won't underflow
    let exp_term = if log_exp_term < -700.0 {
        0.0 // Effectively zero due to underflow
    } else {
        log_exp_term.exp()
    };

    // Series for D_v(x)
    let mut d_sum; // Will be set in the loop
    let mut dp_sum = 0.0;

    // For numerical stability, keep track of previous sums to detect stagnation
    let mut prev_d_sum = 0.0;
    let mut prev_dp_sum = 0.0;
    let mut stagnation_count = 0;

    // Initial coefficient computation, using log-space for better precision
    let v_half = v / 2.0;
    let log_first_term = -v_half * 2.0_f64.ln() + gammaln(1.0 + v_half) - gammaln(1.0);
    let first_term = log_first_term.exp();

    let mut term = first_term;
    let mut dp_term = 0.0;
    d_sum = term;

    // Extended iteration limit for more challenging cases
    for k in 1..100 {
        // Different handling for odd and even k
        if k % 2 == 0 {
            // Even k - computation in log space to avoid overflow/underflow
            let k_f64 = k as f64;
            let log_coef = (k / 2) as f64 * (-(k_f64) / 2.0).ln() - log_factorial(k);
            let coef = log_coef.exp();

            // For x^k, use log-space if k is large
            let sign_correction = if x < 0.0 && k % 2 == 1 { -1.0 } else { 1.0 };
            let log_x_pow_k = k_f64 * x.abs().ln();
            let x_pow_k = sign_correction * log_x_pow_k.exp();

            // Compute terms with protection against overflow
            let new_term = coef * x_pow_k;

            // For derivatives, we need x^(k-1)
            let x_pow_kminus_1 = if k > 0 { x_pow_k / x } else { 1.0 };
            let new_dp_term = coef * x_pow_kminus_1 * k_f64;

            if new_term.is_finite() {
                term = new_term;
                d_sum += term;
            }

            if new_dp_term.is_finite() {
                dp_term = new_dp_term;
                dp_sum += dp_term;
            }
        } else {
            // Odd k - similar approach
            let k_f64 = k as f64;
            let log_coef = ((k - 1) / 2) as f64 * (-(k_f64) / 2.0).ln() - log_factorial(k);
            let coef = log_coef.exp();

            // For x^k
            let sign_correction = if x < 0.0 && k % 2 == 1 { -1.0 } else { 1.0 };
            let log_x_pow_k = k_f64 * x.abs().ln();
            let x_pow_k = sign_correction * log_x_pow_k.exp();

            let new_term = coef * x_pow_k;

            if new_term.is_finite() {
                term = new_term;
                d_sum += term;

                // For odd k, the derivative term is calculated differently
                dp_term = term * k_f64 / x;
                dp_sum += dp_term;
            }
        }

        // Multiple convergence criteria

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance (protecting against division by zero)
        let d_rel_tol = 1e-15 * d_sum.abs().max(1e-300);
        let dp_rel_tol = 1e-15 * dp_sum.abs().max(1e-300);

        // First convergence check: terms becoming small relative to sum
        if (term.abs() < abs_tol || term.abs() < d_rel_tol)
            && (dp_term.abs() < abs_tol || dp_term.abs() < dp_rel_tol)
        {
            break;
        }

        // Second check: detect series stagnation
        if (d_sum - prev_d_sum).abs() < d_rel_tol && (dp_sum - prev_dp_sum).abs() < dp_rel_tol {
            stagnation_count += 1;
            if stagnation_count >= 3 {
                // Series has stopped making meaningful progress
                break;
            }
        } else {
            stagnation_count = 0;
        }

        // Safety check for very small terms that won't affect the result
        if k > 50 && term.abs() < 1e-30 && dp_term.abs() < 1e-30 {
            break;
        }

        prev_d_sum = d_sum;
        prev_dp_sum = dp_sum;
    }

    // Final calculation with careful handling of potential overflow/underflow
    let d = d_sum * exp_term;

    // Derivative calculation: D_v'(x) = derivative of series - x/2 * D_v(x)
    let dp = dp_sum * exp_term - x / 2.0 * d;

    // Check for any NaN or Inf results (can happen for extreme parameter values)
    if !d.is_finite() || !dp.is_finite() {
        // Fall back to direct evaluation for specific orders if series fails
        if v.floor() == v && (-5.0..=5.0).contains(&v) {
            return pbdv_integer(v as i32, x);
        }

        // For non-integer v that causes overflow, make a best approximation
        let d_approx = if v < 0.0 && x.abs() > 10.0 {
            // For large |x| and negative v, D_v(x) grows very large
            if x > 0.0 {
                0.0 // Approaches 0
            } else {
                f64::INFINITY.copysign(if v.floor() as i32 % 2 == 0 { 1.0 } else { -1.0 })
            }
        } else if v > 0.0 && x.abs() > 10.0 {
            // For large |x| and positive v, D_v(x) approaches 0
            0.0
        } else {
            // Other cases - use direct formula for moderate v
            let gamma_val = gamma((1.0 + v) / 2.0);
            2_f64.powf(-v / 2.0) * gamma_val * (-x2 / 4.0).exp()
        };

        let dp_approx = -x / 2.0 * d_approx; // Simple approximation for the derivative

        return Ok((d_approx, dp_approx));
    }

    Ok((d, dp))
}

/// Asymptotic expansion for D_v(x) for large positive x with enhanced numerical stability
#[allow(dead_code)]
fn pbdv_asymptotic_pos(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // For extremely large x
    if x > 100.0 && v > 0.0 {
        // For large positive x and v > 0, D_v(x) approaches 0 exponentially
        return Ok((0.0, 0.0));
    }

    // Scale to avoid overflow
    let z = x / SQRT_2;
    let v2 = v * v;

    // Calculate the asymptotic series in log space to avoid overflow/underflow
    let log_pre_factor = -z * z / 2.0 - (v + 0.5) * z.abs().ln() - (v / 2.0) * 2.0_f64.ln();

    // Compute the sum for the asymptotic expansion
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_sum = 0.0;
    let mut stagnation_count = 0;

    // Keep track of the derivative term
    let mut deriv_term = 0.0;

    // Extended iteration limit for better accuracy
    for k in 1..30 {
        // Calculate the numerator carefully to avoid overflow
        let numerator = v2 - (2 * k - 1) as f64 * (2 * k - 1) as f64;
        let denominator = 2.0 * k as f64 * z * z;

        // Check for potential numerical issues
        if denominator.abs() < 1e-300 {
            // Avoid division by zero
            break;
        }

        let term_factor = numerator / denominator;
        let new_term = term * term_factor;

        // Check for numerical stability
        if !new_term.is_finite() {
            // If the term would cause overflow, stop here
            break;
        }

        term = new_term;
        sum += term;

        // Save this term for derivative calculation
        deriv_term = term;

        // Multiple convergence criteria

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance with protection against zero division
        let rel_tol = 1e-15 * sum.abs().max(1e-300);

        // First convergence check
        if term.abs() < abs_tol || term.abs() < rel_tol {
            break;
        }

        // Check for series stagnation
        if (sum - prev_sum).abs() < rel_tol {
            stagnation_count += 1;
            if stagnation_count >= 3 {
                break;
            }
        } else {
            stagnation_count = 0;
        }

        // Check for potential divergence in the asymptotic series
        if k > 5 && term.abs() > prev_sum.abs() {
            // Series may be starting to diverge, use the value before divergence
            sum = prev_sum;
            break;
        }

        prev_sum = sum;
    }

    // Calculate function value in log space, then exponentiate
    // Only exponentiate if it won't overflow
    let d = if log_pre_factor < -700.0 {
        0.0 // Underflow to zero
    } else if log_pre_factor > 700.0 {
        f64::INFINITY.copysign(if v.floor() as i32 % 2 == 0 { 1.0 } else { -1.0 })
    // Overflow
    } else {
        log_pre_factor.exp() * sum
    };

    // Compute the derivative more carefully
    let dp = if d.abs() < 1e-300 {
        // For very small function values, the derivative is also near zero
        0.0
    } else {
        // Derivative calculation: D_v'(x) = D_v(x) * (-z - (v+0.5)/z + correction)
        // where correction accounts for the derivative of the sum
        let correction = deriv_term * z / sum;
        -d * (z + (v + 0.5) / z - correction)
    };

    // Final validation for potential numerical errors
    if !d.is_finite() || !dp.is_finite() {
        // For extreme cases, approximation for v >> 0 or x >> 0
        let d_approx = 0.0; // D_v(x) → 0 for large positive x
        let dp_approx = 0.0; // Similarly for the derivative
        return Ok((d_approx, dp_approx));
    }

    Ok((d, dp))
}

/// Asymptotic expansion for D_v(x) for large negative x
#[allow(dead_code)]
fn pbdv_asymptotic_neg(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // For large negative x, use relationship with V_v(x)
    let (v_val, vp_val) = pbvv(v, -x)?;

    // Relationship: D_v(-x) = (sin(πv) * V_v(x) + V_{v-1}(x) / cos(πv/2)) / sin(π(v+1)/2)
    let pi_v = PI * v;
    let sin_pi_v = pi_v.sin();
    let sin_pi_v_plus_1_half = (PI * (v + 1.0) / 2.0).sin();
    let cos_pi_v_half = (PI * v / 2.0).cos();

    let (v_val_prev, _) = pbvv(v - 1.0, -x)?;

    let d = (sin_pi_v * v_val + v_val_prev / cos_pi_v_half) / sin_pi_v_plus_1_half;
    let dp = -vp_val; // Derivative needs sign change due to x → -x

    Ok((d, dp))
}

/// Parabolic cylinder function V_v(x) and its derivative.
///
/// The parabolic cylinder function V_v(x) is another solution to the differential equation:
/// d²y/dx² + (v + 1/2 - x²/4)y = 0
///
/// # Arguments
///
/// * `v` - Order parameter
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple containing (V_v(x), V_v'(x))
///
/// # Examples
///
/// ```
/// use scirs2_special::pbvv;
///
/// let (v, vp) = pbvv(1.0, 0.5).unwrap();
/// println!("V_1(0.5) = {}, V_1'(0.5) = {}", v, vp);
/// ```
#[allow(dead_code)]
pub fn pbvv(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if v.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to pbvv".to_string()));
    }

    // V_v can be expressed in terms of D_v for certain cases
    if v.floor() == v && (-20.0..=20.0).contains(&v) {
        // Integer case - can be derived from D_v
        pbvv_integer(v as i32, x)
    } else {
        // General case - use series expansion or asymptotic forms
        pbvv_general(v, x)
    }
}

/// Implementation of V_v(x) for integer v
#[allow(dead_code)]
fn pbvv_integer(v: i32, x: f64) -> SpecialResult<(f64, f64)> {
    // For integer v, V_v can be expressed in terms of D_v and D_{-v-1}
    let (d_v, d_v_prime) = pbdv(v as f64, x)?;

    if v >= 0 {
        // For v ≥ 0
        let (d_neg_vminus_1, d_neg_vminus_1_prime) = pbdv(-(v as f64) - 1.0, x)?;

        let gamma_arg1 = v as f64 + 1.0;
        let gamma_v_plus_1 = gamma(gamma_arg1);

        // Formula: V_v(x) = D_v(x)·cos(πv) - D_{-v-1}(x)·√(2/π)·Γ(v+1)
        let v_val =
            d_v * (PI * v as f64).cos() - d_neg_vminus_1 * (2.0 / PI).sqrt() * gamma_v_plus_1;
        let vp_val = d_v_prime * (PI * v as f64).cos()
            - d_neg_vminus_1_prime * (2.0 / PI).sqrt() * gamma_v_plus_1;

        Ok((v_val, vp_val))
    } else {
        // For v < 0
        let (d_neg_vminus_1, d_neg_vminus_1_prime) = pbdv(-(v as f64) - 1.0, x)?;

        let gamma_arg1 = -(v as f64);
        let _gamma_neg_v = gamma(gamma_arg1);

        // Formula for v < 0
        let v_val = d_neg_vminus_1;
        let vp_val = d_neg_vminus_1_prime;

        Ok((v_val, vp_val))
    }
}

/// General implementation of V_v(x) using series or asymptotic forms
#[allow(dead_code)]
fn pbvv_general(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // For small |x|, use series expansion
    if x.abs() < 5.0 {
        return pbvv_series(v, x);
    }

    // For large |x|, use asymptotic forms
    if x.abs() > (2.0 * v.abs()).max(20.0) {
        return pbvv_asymptotic(v, x);
    }

    // Fallback to series implementation
    pbvv_series(v, x)
}

/// Series implementation for V_v(x) with enhanced numerical stability
#[allow(dead_code)]
fn pbvv_series(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // Special case for x = 0
    if x == 0.0 {
        // For x = 0, V_v(0) depends on gamma function
        let sqrt_2_pi = (2.0 / PI).sqrt();
        let gamma_term = gamma(v + 0.5);

        // V_v(0) = sqrt(2/π) / Γ(v+1/2)
        let v_at_zero = sqrt_2_pi / gamma_term;

        // V_v'(0) depends on whether v is odd or even
        let vp_at_zero = if v.floor() == v && v >= 0.0 {
            if (v as i32) % 2 == 0 {
                0.0 // Even v
            } else {
                // Odd v, calculate from recurrence relation
                let mut val = sqrt_2_pi;
                for k in 1..((v as i32 + 1) / 2) {
                    val *= (2 * k - 1) as f64 / 2.0;
                }
                val
            }
        } else {
            // For non-integer v, approximate
            0.0
        };

        return Ok((v_at_zero, vp_at_zero));
    }

    // Special case for very large |x|
    if x.abs() > 100.0 {
        // For very large |x|, use the asymptotic form instead
        return pbvv_asymptotic(v, x);
    }

    // For normal range of x, use enhanced series method
    let x2 = x * x;

    // Use log-based calculation for exponential to avoid overflow
    let log_exp_term = x2 / 4.0;

    // Check for potential overflow
    let exp_term = if log_exp_term > 700.0 {
        // For very large x², handle the exponential carefully
        // V_v(x) grows very large for large |x|
        f64::INFINITY
    } else {
        log_exp_term.exp()
    };

    // Coefficient calculation in log space for better precision
    let sqrt_2_pi = (2.0 / PI).sqrt();

    // Handle potential issues with gamma function
    let gamma_term = if v + 0.5 > 170.0 {
        // For large v, use logarithmic gamma
        gammaln(v + 0.5).exp()
    } else {
        gamma(v + 0.5)
    };

    // Check for potential division by zero or infinity
    let leading = if gamma_term.abs() < 1e-300 {
        f64::INFINITY
    } else if gamma_term.is_infinite() {
        0.0
    } else {
        sqrt_2_pi * exp_term / gamma_term
    };

    // Series for V_v(x)
    let mut v_sum; // Will be set below
    let mut vp_sum = 0.0;

    // For numerical stability, track previous sums
    let mut prev_v_sum = 0.0;
    let mut prev_vp_sum = 0.0;
    let mut stagnation_count = 0;

    // Initial term
    let mut term = 1.0;
    let mut vp_term = 0.0;
    v_sum = term;

    // Extended iteration limit for challenging cases
    for k in 1..80 {
        // Calculate coefficient in log space for better precision
        let k_f64 = k as f64;

        if k <= 20 {
            // For small k, direct calculation is stable
            let coef = (k_f64 / 2.0).powi(k / 2) / factorial(k as usize);

            // Calculate x^k safely
            let x_pow_k = x.powi(k);
            let new_term = coef * x_pow_k;

            // For derivatives, we need x^(k-1)
            let x_pow_kminus_1 = if k > 0 { x_pow_k / x } else { 1.0 };
            let new_vp_term = coef * x_pow_kminus_1 * k_f64;

            if new_term.is_finite() {
                term = new_term;
                v_sum += term;
            }

            if new_vp_term.is_finite() {
                vp_term = new_vp_term;
                vp_sum += vp_term;
            }
        } else {
            // For large k, use logarithmic calculation
            let log_coef = (k / 2) as f64 * (k_f64 / 2.0).ln() - log_factorial(k as usize);
            let coef = log_coef.exp();

            // For x^k
            let sign_correction = if x < 0.0 && k % 2 == 1 { -1.0 } else { 1.0 };
            let log_x_pow_k = k_f64 * x.abs().ln();
            let x_pow_k = sign_correction * log_x_pow_k.exp();

            let new_term = coef * x_pow_k;

            // For derivatives
            let sign_correctionminus_1 = if x < 0.0 && (k - 1) % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            let log_x_pow_kminus_1 = (k_f64 - 1.0) * x.abs().ln();
            let x_pow_kminus_1 = sign_correctionminus_1 * log_x_pow_kminus_1.exp();
            let new_vp_term = coef * x_pow_kminus_1 * k_f64;

            if new_term.is_finite() {
                term = new_term;
                v_sum += term;
            }

            if new_vp_term.is_finite() {
                vp_term = new_vp_term;
                vp_sum += vp_term;
            }
        }

        // Multiple convergence criteria

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance with protection against zero division
        let v_rel_tol = 1e-15 * v_sum.abs().max(1e-300);
        let vp_rel_tol = 1e-15 * vp_sum.abs().max(1e-300);

        // First check: terms becoming small
        if (term.abs() < abs_tol || term.abs() < v_rel_tol)
            && (vp_term.abs() < abs_tol || vp_term.abs() < vp_rel_tol)
        {
            break;
        }

        // Second check: sum stagnation
        if (v_sum - prev_v_sum).abs() < v_rel_tol && (vp_sum - prev_vp_sum).abs() < vp_rel_tol {
            stagnation_count += 1;
            if stagnation_count >= 3 {
                break;
            }
        } else {
            stagnation_count = 0;
        }

        // Safety check for very small terms
        if k > 50 && term.abs() < 1e-30 && vp_term.abs() < 1e-30 {
            break;
        }

        prev_v_sum = v_sum;
        prev_vp_sum = vp_sum;
    }

    // Final multiplication with the leading term
    let v_val = leading * v_sum;

    // Derivative calculation: V_v'(x) = V_v(x) * x/2 + derivative of series
    let vp_val = leading * (vp_sum + v_sum * x / 2.0);

    // Check for numerical issues
    if !v_val.is_finite() || !vp_val.is_finite() {
        // Handle extreme cases
        if x.abs() > 20.0 {
            // For large |x|, V_v(x) grows very rapidly
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let v_approx = sign * f64::INFINITY;
            let vp_approx = sign * f64::INFINITY;
            return Ok((v_approx, vp_approx));
        } else if v > 100.0 {
            // For large v, approximate behavior
            let v_approx = 0.0;
            let vp_approx = 0.0;
            return Ok((v_approx, vp_approx));
        }
    }

    Ok((v_val, vp_val))
}

/// Asymptotic expansion for V_v(x) with enhanced numerical stability
#[allow(dead_code)]
fn pbvv_asymptotic(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    // For extremely large |x|
    if x.abs() > 100.0 {
        // For very large |x|, V_v(x) grows exponentially
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };

        // For extremely large |x|, V_v(x) approaches sign * infinity
        if x.abs() > 700.0 {
            return Ok((sign * f64::INFINITY, sign * f64::INFINITY));
        }
    }

    // Scale to avoid overflow
    let z = x.abs() / SQRT_2;
    let v2 = v * v;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };

    // Calculate the asymptotic series in log space to avoid overflow/underflow
    // Compute the sum for the asymptotic expansion with enhanced stability
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_sum = 0.0;
    let mut stagnation_count = 0;

    // Keep track of the derivative term
    let mut deriv_term = 0.0;

    // Extended iteration limit for better accuracy
    for k in 1..30 {
        // Calculate the numerator carefully to avoid overflow
        let numerator = v2 - (2 * k - 1) as f64 * (2 * k - 1) as f64;
        let denominator = 2.0 * k as f64 * z * z;

        // Check for potential numerical issues
        if denominator.abs() < 1e-300 {
            // Avoid division by zero
            break;
        }

        let term_factor = numerator / denominator;
        let new_term = term * term_factor;

        // Check for numerical stability
        if !new_term.is_finite() {
            // If the term would cause overflow, stop here
            break;
        }

        term = new_term;
        sum += term;

        // Save this term for derivative calculation
        deriv_term = term;

        // Multiple convergence criteria

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance with protection against zero division
        let rel_tol = 1e-15 * sum.abs().max(1e-300);

        // First convergence check
        if term.abs() < abs_tol || term.abs() < rel_tol {
            break;
        }

        // Check for series stagnation
        if (sum - prev_sum).abs() < rel_tol {
            stagnation_count += 1;
            if stagnation_count >= 3 {
                break;
            }
        } else {
            stagnation_count = 0;
        }

        // Check for potential divergence in the asymptotic series
        if k > 5 && term.abs() > prev_sum.abs() {
            // Series may be starting to diverge, use the value before divergence
            sum = prev_sum;
            break;
        }

        prev_sum = sum;
    }

    // For V_v, the leading term differs from D_v
    let _sqrt_2_pi = (2.0 / PI).sqrt();

    // Carefully handle gamma function for large v
    let gamma_term = if v + 0.5 > 170.0 {
        // For large v, use logarithmic gamma
        gammaln(v + 0.5).exp()
    } else {
        gamma(v + 0.5)
    };

    // Calculate in log space to avoid overflow/underflow
    // log(V_v(x)) = log(sqrt(2/π)) + log(exp(z²/2)) - log(z^(v+0.5)) + log(sum) - log(gamma_term)
    let log_sqrt_2_pi = (2.0 / PI).sqrt().ln();
    let log_exp_term = z * z / 2.0;
    let log_z_term = (v + 0.5) * z.ln();
    let log_sum = sum.ln();
    let log_gamma = gamma_term.ln();

    // Combined log calculation
    let log_v_val = log_sqrt_2_pi + log_exp_term - log_z_term + log_sum - log_gamma;

    // Only exponentiate if it won't overflow/underflow
    let v_val = if log_v_val > 700.0 {
        sign * f64::INFINITY
    } else if log_v_val < -700.0 {
        0.0
    } else {
        sign * log_v_val.exp()
    };

    // Enhanced derivative calculation that handles extreme cases
    let vp_val = if !v_val.is_finite() {
        // For infinite function values, the derivative is also infinite
        v_val
    } else if v_val.abs() < 1e-300 {
        // For very small function values, the derivative is also near zero
        0.0
    } else {
        // Standard derivative calculation with correction term
        let correction = if sum.abs() < 1e-300 {
            0.0 // Avoid division by zero
        } else {
            deriv_term * z / sum
        };

        v_val * (sign * z + (v + 0.5) / z - correction)
    };

    // Final validation for potential numerical errors
    if !v_val.is_finite() && x.abs() < 100.0 {
        // For cases where we get overflow but x isn't extremely large
        // This can happen for some combinations of v and x
        // Provide a reasoned approximation
        let asymptotic_sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let v_approx = asymptotic_sign * f64::MAX * 0.1; // Large but not infinity
        let vp_approx = v_approx; // Derivative grows at similar rate

        return Ok((v_approx, vp_approx));
    }

    Ok((v_val, vp_val))
}

/// Compute a sequence of parabolic cylinder functions D_v(x) for v = 0, 1, ..., vmax.
///
/// # Arguments
///
/// * `vmax` - Maximum order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple of two vectors containing (D_0(x)...D_vmax(x), D_0'(x)...D_vmax'(x))
///
/// # Examples
///
/// ```
/// use scirs2_special::pbdv_seq;
///
/// let (d_values, dp_values) = pbdv_seq(3, 0.5).unwrap();
/// println!("D_0(0.5) = {}, D_1(0.5) = {}, D_2(0.5) = {}, D_3(0.5) = {}",
///         d_values[0], d_values[1], d_values[2], d_values[3]);
/// ```
#[allow(dead_code)]
pub fn pbdv_seq(vmax: usize, x: f64) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to pbdv_seq".to_string(),
        ));
    }

    if vmax == 0 {
        let (d0, d0p) = pbdv(0.0, x)?;
        return Ok((vec![d0], vec![d0p]));
    }

    // Initialize arrays to hold function values and derivatives
    let mut d_values = vec![0.0; vmax + 1];
    let mut dp_values = vec![0.0; vmax + 1];

    // Compute D_0 and D_1 directly
    let (d0, d0p) = pbdv(0.0, x)?;
    let (d1, d1p) = pbdv(1.0, x)?;

    d_values[0] = d0;
    dp_values[0] = d0p;

    if vmax >= 1 {
        d_values[1] = d1;
        dp_values[1] = d1p;
    }

    // Use recurrence relation to compute higher orders
    for v in 2..=vmax {
        // Recurrence relation: D_{v+1}(x) = x*D_v(x) - v*D_{v-1}(x)
        d_values[v] = x * d_values[v - 1] - (v as f64 - 1.0) * d_values[v - 2];
        // Derivative recurrence: D'_{v+1}(x) = x*D'_v(x) - v*D'_{v-1}(x) + D_v(x)
        dp_values[v] = x * dp_values[v - 1] - (v as f64 - 1.0) * dp_values[v - 2] + d_values[v - 1];
    }

    Ok((d_values, dp_values))
}

/// Compute a sequence of parabolic cylinder functions V_v(x) for v = 0, 1, ..., vmax.
///
/// # Arguments
///
/// * `vmax` - Maximum order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple of two vectors containing (V_0(x)...V_vmax(x), V_0'(x)...V_vmax'(x))
///
/// # Examples
///
/// ```
/// use scirs2_special::pbvv_seq;
///
/// let (v_values, vp_values) = pbvv_seq(3, 0.5).unwrap();
/// println!("V_0(0.5) = {}, V_1(0.5) = {}, V_2(0.5) = {}, V_3(0.5) = {}",
///         v_values[0], v_values[1], v_values[2], v_values[3]);
/// ```
#[allow(dead_code)]
pub fn pbvv_seq(vmax: usize, x: f64) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to pbvv_seq".to_string(),
        ));
    }

    // Initialize arrays to hold function values and derivatives
    let mut v_values = vec![0.0; vmax + 1];
    let mut vp_values = vec![0.0; vmax + 1];

    // Compute each V_v individually - this can be optimized further
    for v in 0..=vmax {
        let (v_val, vp_val) = pbvv(v as f64, x)?;
        v_values[v] = v_val;
        vp_values[v] = vp_val;
    }

    Ok((v_values, vp_values))
}

/// Compute parabolic cylinder function W(a,x) and its derivative.
///
/// The function W(a,x) is related to the parabolic cylinder functions
/// D_v(x) and satisfies the differential equation:
/// d²W/dx² + (a - x²/4)W = 0
///
/// # Arguments
///
/// * `a` - Parameter
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple containing (W(a,x), W'(a,x))
///
/// # Examples
///
/// ```
/// use scirs2_special::pbwa;
///
/// let (w, wp) = pbwa(1.0, 0.5).unwrap();
/// println!("W(1.0, 0.5) = {}, W'(1.0, 0.5) = {}", w, wp);
/// ```
#[allow(dead_code)]
pub fn pbwa(a: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if a.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to pbwa".to_string()));
    }

    // W(a,x) is related to D_v(x) by: W(a,x) = D_{-a-1/2}(x)
    let v = -a - 0.5;
    pbdv(v, x)
}

// Helper function to calculate factorial with overflow protection
#[allow(dead_code)]
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }

    // For small values, direct computation is fine
    if n <= 20 {
        let mut result = 1.0;
        for i in 2..=n {
            result *= i as f64;
        }
        return result;
    }

    // For larger values, use the logarithmic approach to avoid overflow
    // Switch to gammaln for large factorials, since n! = Γ(n+1)
    let result = gammaln(n as f64 + 1.0).exp();

    // Check if the result is finite
    if result.is_finite() {
        result
    } else {
        // For extremely large n that would overflow even with gammaln,
        // use Stirling's approximation: n! ≈ sqrt(2πn) * (n/e)^n
        let n_f64 = n as f64;
        (2.0 * PI * n_f64).sqrt() * (n_f64 / std::f64::consts::E).powf(n_f64)
    }
}

// Helper function to approximate error function with improved accuracy and stability
#[allow(dead_code)]
fn erf_approx(x: f64) -> f64 {
    // Special cases
    if x == 0.0 {
        return 0.0;
    }

    if !x.is_finite() {
        return if x.is_sign_positive() { 1.0 } else { -1.0 };
    }

    // Extremely large values of |x|
    if x.abs() > 6.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // A more accurate approximation based on Abramowitz and Stegun (1964)
    // with optimized parameters for floating-point arithmetic stability
    if x.abs() <= 0.5 {
        // Use Taylor series expansion for small |x|
        // erf(x) ≈ (2/sqrt(π)) * x * (1 - x²/3 + x⁴/10 - x⁶/42 + x⁸/216 - ...)
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        let x8 = x4 * x4;

        let two_over_sqrt_pi = 2.0 / PI.sqrt();

        x * two_over_sqrt_pi * (1.0 - x2 / 3.0 + x4 / 10.0 - x6 / 42.0 + x8 / 216.0)
    } else {
        // For larger |x|, use the approximation from Numerical Recipes
        let z = x.abs();
        let t = 1.0 / (1.0 + 0.5 * z);

        // More accurate coefficients for better stability
        let tau = t
            * (-z * z - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398
                                        + t * (1.48851587
                                            + t * (-0.82215223 + t * 0.17087277)))))))));

        let result = if x >= 0.0 { 1.0 - tau } else { tau - 1.0 };

        // Check for underflow or near-zero values
        if result.abs() < 1e-16 {
            if x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            result
        }
    }
}

// Log factorial function for use with very large factorials
#[allow(dead_code)]
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0; // log(1) = 0
    }

    // Use gammaln for log(n!)
    gammaln(n as f64 + 1.0)
}
