//! Optimizations for special mathematical functions.
//!
//! This module contains optimized implementations of various special functions
//! using techniques like Padé approximations, range-specific algorithms,
//! and caching for frequently used values.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Mutex;

lazy_static::lazy_static! {
    // Cache for polylogarithm values
    static ref POLYLOG_CACHE: Mutex<HashMap<(i32, i32), f64>> = Mutex::new(HashMap::new());

    // Cache for special values and constants
    static ref SPECIAL_VALUES: Mutex<HashMap<&'static str, f64>> = {
        let mut m = HashMap::new();
        // Common constants
        m.insert("euler_mascheroni", 0.577_215_664_901_532_9);
        m.insert("pi_squared_div_6", PI * PI / 6.0);
        m.insert("pi_squared_div_12", PI * PI / 12.0);
        m.insert("pi_fourth_div_90", PI.powi(4) / 90.0);
        Mutex::new(m)
    };
}

/// Get a cached constant value.
///
/// # Arguments
///
/// * `name` - Name of the constant to retrieve
///
/// # Returns
///
/// * `f64` - Value of the constant
pub fn get_constant(name: &'static str) -> f64 {
    if let Some(value) = SPECIAL_VALUES.lock().unwrap().get(name) {
        return *value;
    }

    // Compute and cache common mathematical constants
    let value = match name {
        "euler_mascheroni" => 0.577_215_664_901_532_9,
        "pi_squared_div_6" => PI * PI / 6.0,
        "pi_squared_div_12" => PI * PI / 12.0,
        "pi_fourth_div_90" => PI.powi(4) / 90.0,
        _ => 0.0, // Default for unknown constants
    };

    // Cache the computed value
    SPECIAL_VALUES.lock().unwrap().insert(name, value);
    value
}

/// Get cached polylogarithm value if available.
///
/// # Arguments
///
/// * `s` - Order of the polylogarithm
/// * `x` - Argument value
///
/// # Returns
///
/// * `Option<f64>` - Cached value if available, None otherwise
pub fn get_cached_polylog(s: f64, x: f64) -> Option<f64> {
    // We only cache for integer s and discretized x (to avoid filling memory)
    let s_int = s.round() as i32;
    let x_int = (x * 1000.0).round() as i32; // Round to 3 decimal places

    // Only use cache for exact integer s values
    if (s - s_int as f64).abs() < f64::EPSILON && s_int > 0 && s_int <= 10 {
        if let Some(value) = POLYLOG_CACHE.lock().unwrap().get(&(s_int, x_int)) {
            return Some(*value);
        }
    }

    None
}

/// Store polylogarithm value in cache.
///
/// # Arguments
///
/// * `s` - Order of the polylogarithm
/// * `x` - Argument value
/// * `value` - Result to cache
pub fn cache_polylog(s: f64, x: f64, value: f64) {
    // We only cache for integer s and discretized x (to avoid filling memory)
    let s_int = s.round() as i32;
    let x_int = (x * 1000.0).round() as i32; // Round to 3 decimal places

    // Only cache for exact integer s values
    if (s - s_int as f64).abs() < f64::EPSILON && s_int > 0 && s_int <= 10 {
        // Limit cache size
        let mut cache = POLYLOG_CACHE.lock().unwrap();
        if cache.len() < 10000 {
            // Prevent unbounded growth
            cache.insert((s_int, x_int), value);
        }
    }
}

/// Optimized ln(1+x) to handle small x more accurately.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `f64` - ln(1+x) computed accurately
pub fn ln_1p_optimized(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        // Use series expansion for very small x
        let x2 = x * x;
        x - x2 / 2.0 + x2 * x / 3.0 - x2 * x2 / 4.0
    } else {
        (1.0 + x).ln()
    }
}

/// Pade approximation for the exponential integral Ei(x).
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `f64` - Exponential integral
pub fn exponential_integral_pade(x: f64) -> f64 {
    // Only use Pade approximation for x > 0
    if x <= 0.0 {
        return -exponential_integral_e1_pade(-x);
    }

    if x < 6.0 {
        // Coefficients for Pade approximation (numerator)
        let num_coeffs = [1.0, 7.5, 18.75, 18.75, 7.5, 1.0];

        // Coefficients for Pade approximation (denominator)
        let den_coeffs = [1.0, -7.5, 18.75, -18.75, 7.5, -1.0];

        // Evaluate the approximation
        let mut num = 0.0;
        let mut den = 0.0;
        let z = x;

        for i in 0..6 {
            num = num * z + num_coeffs[5 - i];
            den = den * z + den_coeffs[5 - i];
        }

        let euler_mascheroni = get_constant("euler_mascheroni");
        euler_mascheroni + x.ln() + x * (num / den)
    } else {
        // For large x, use asymptotic expansion
        let mut sum = 1.0;
        let mut term = 1.0;

        for k in 1..10 {
            term *= k as f64 / x;
            sum += term;

            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        sum * x.exp() / x
    }
}

/// Pade approximation for the exponential integral E₁(x).
///
/// # Arguments
///
/// * `x` - Input value, must be positive
///
/// # Returns
///
/// * `f64` - Exponential integral E₁(x)
pub fn exponential_integral_e1_pade(x: f64) -> f64 {
    assert!(x > 0.0, "E₁(x) is only defined for x > 0");

    if x < 1.0 {
        // Use series expansion for small x
        let euler_mascheroni = get_constant("euler_mascheroni");
        let mut sum = -euler_mascheroni - x.ln();
        let mut term = -1.0;
        let mut factorial = 1.0;

        for k in 1..15 {
            let k_f64 = k as f64;
            term *= -x / k_f64;
            factorial *= k_f64;
            let contribution = term / (k_f64 * factorial);
            sum -= contribution;

            if contribution.abs() < 1e-15 * sum.abs() {
                break;
            }
        }

        -sum
    } else {
        // Use a more efficient continued fraction for larger x
        let mut b = x + 1.0;
        let mut c = 1.0 / 1e-30; // Arbitrary large value
        let mut d = 1.0 / b;
        let mut h = d;

        for i in 1..20 {
            // Fewer iterations for efficiency
            let a = -(i * i) as f64;
            b += 2.0;
            d = 1.0 / (b + a * d);
            c = b + a / c;
            let del = c * d;
            h *= del;

            if (del - 1.0).abs() < 1e-10 {
                break;
            }
        }

        h * (-x).exp()
    }
}
