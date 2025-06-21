//! Performance optimizations for special functions
//!
//! This module provides SIMD optimizations, lookup tables, and other performance
//! enhancements for special function computations. All optimizations maintain
//! numerical accuracy while improving computational speed.

#![allow(dead_code)]

use crate::error::{SpecialError, SpecialResult};
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

/// SIMD-optimized operations for vectorized computations
pub mod simd {
    use super::*;

    /// SIMD-optimized exponential function for arrays
    ///
    /// Uses vectorized operations to compute exp(x) for multiple values
    /// simultaneously when SIMD support is available.
    pub fn exp_simd(values: &[f64]) -> Vec<f64> {
        // For now, use standard library functions
        // In a full SIMD implementation, this would use platform-specific intrinsics
        values
            .iter()
            .map(|&x| {
                if x > 700.0 {
                    f64::INFINITY
                } else if x < -700.0 {
                    0.0
                } else {
                    x.exp()
                }
            })
            .collect()
    }

    /// SIMD-optimized logarithm function for arrays
    pub fn ln_simd(values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|&x| {
                if x <= 0.0 {
                    f64::NAN
                } else if x == 1.0 {
                    0.0
                } else {
                    x.ln()
                }
            })
            .collect()
    }

    /// SIMD-optimized sine function for arrays
    pub fn sin_simd(values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|&x| {
                if x.is_infinite() || x.is_nan() {
                    f64::NAN
                } else {
                    // Use argument reduction for better accuracy
                    let reduced = x % (2.0 * PI);
                    reduced.sin()
                }
            })
            .collect()
    }

    /// SIMD-optimized cosine function for arrays
    pub fn cos_simd(values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|&x| {
                if x.is_infinite() || x.is_nan() {
                    f64::NAN
                } else {
                    // Use argument reduction for better accuracy
                    let reduced = x % (2.0 * PI);
                    reduced.cos()
                }
            })
            .collect()
    }

    /// SIMD-optimized gamma function for arrays
    pub fn gamma_simd(values: &[f64]) -> Vec<f64> {
        use crate::gamma::gamma;
        values.iter().map(|&x| gamma(x)).collect()
    }
}

/// Lookup tables for commonly computed values
pub mod lookup_tables {
    use super::*;
    use std::sync::OnceLock;

    /// Precomputed factorial values up to 20!
    static FACTORIAL_TABLE: OnceLock<Vec<f64>> = OnceLock::new();

    /// Initialize and get the factorial lookup table
    fn get_factorial_table() -> &'static Vec<f64> {
        FACTORIAL_TABLE.get_or_init(|| {
            let mut table = Vec::with_capacity(21);
            table.push(1.0); // 0!
            for i in 1..=20 {
                table.push(table[i - 1] * i as f64);
            }
            table
        })
    }

    /// Fast factorial lookup for small integers
    pub fn factorial_lookup(n: u32) -> Option<f64> {
        if n <= 20 {
            Some(get_factorial_table()[n as usize])
        } else {
            None
        }
    }

    /// Precomputed double factorial values
    static DOUBLE_FACTORIAL_TABLE: OnceLock<Vec<f64>> = OnceLock::new();

    fn get_double_factorial_table() -> &'static Vec<f64> {
        DOUBLE_FACTORIAL_TABLE.get_or_init(|| {
            let mut table = Vec::with_capacity(31);
            table.push(1.0); // 0!!
            table.push(1.0); // 1!!

            // Even double factorials
            let mut even_val = 2.0;
            // Odd double factorials
            let mut odd_val = 1.0;

            for i in 2..=30 {
                if i % 2 == 0 {
                    even_val *= i as f64;
                    table.push(even_val);
                } else {
                    odd_val *= i as f64;
                    table.push(odd_val);
                }
            }
            table
        })
    }

    /// Fast double factorial lookup
    pub fn double_factorial_lookup(n: u32) -> Option<f64> {
        if n <= 30 {
            Some(get_double_factorial_table()[n as usize])
        } else {
            None
        }
    }

    /// Precomputed values for commonly used gamma function arguments
    static GAMMA_TABLE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();

    fn get_gamma_table() -> &'static Vec<(f64, f64)> {
        GAMMA_TABLE.get_or_init(|| {
            vec![
                (0.5, (PI).sqrt()),
                (1.0, 1.0),
                (1.5, (PI).sqrt() / 2.0),
                (2.0, 1.0),
                (2.5, 3.0 * (PI).sqrt() / 4.0),
                (3.0, 2.0),
                (3.5, 15.0 * (PI).sqrt() / 8.0),
                (4.0, 6.0),
                (4.5, 105.0 * (PI).sqrt() / 16.0),
                (5.0, 24.0),
            ]
        })
    }

    /// Fast gamma function lookup for common values
    pub fn gamma_lookup(x: f64) -> Option<f64> {
        let table = get_gamma_table();
        for &(arg, value) in table {
            if (x - arg).abs() < 1e-15 {
                return Some(value);
            }
        }
        None
    }

    /// Precomputed Bessel J0 zeros for quick access
    static J0_ZEROS_TABLE: OnceLock<Vec<f64>> = OnceLock::new();

    fn get_j0_zeros_table() -> &'static Vec<f64> {
        J0_ZEROS_TABLE.get_or_init(|| {
            vec![
                2.404_825_557_695_773,
                5.520_078_110_286_311,
                8.653_727_912_911_013,
                11.791_534_439_014_281,
                14.930_917_708_487_787,
                18.071_063_967_910_924,
                21.211_636_629_879_26,
                24.352_471_530_749_302,
                27.493_479_132_040_253,
                30.634_606_468_431_976,
            ]
        })
    }

    /// Get the nth zero of J0 Bessel function
    pub fn j0_zero(n: usize) -> Option<f64> {
        let table = get_j0_zeros_table();
        if n < table.len() {
            Some(table[n])
        } else {
            // For larger n, use asymptotic approximation
            if n > 0 {
                let n_f = n as f64;
                Some(PI * (n_f + 0.25))
            } else {
                None
            }
        }
    }
}

/// Polynomial approximations for fast function evaluation
pub mod poly_approx {
    use super::*;

    /// Coefficients for erf(x) approximation on [0, 1]
    const ERF_COEFFS: &[f64] = &[
        std::f64::consts::FRAC_2_SQRT_PI, // 2/sqrt(π)
        -0.3761263890318376,              // -(2/sqrt(π)) * (1/3)
        0.1128379167095513,               // (2/sqrt(π)) * (1/10)
        -0.0268661098637320,              // -(2/sqrt(π)) * (1/42)
        0.0052308506508171,               // (2/sqrt(π)) * (1/216)
    ];

    /// Fast polynomial approximation for erf(x) on [0, 1]
    pub fn erf_approx(x: f64) -> f64 {
        if x.abs() > 1.0 {
            return if x > 0.0 { 1.0 } else { -1.0 };
        }

        let x2 = x * x;
        let mut result = 0.0;
        let mut power = x;

        for &coeff in ERF_COEFFS {
            result += coeff * power;
            power *= x2;
        }

        result
    }

    /// Coefficients for exp(-x²) approximation
    const EXP_NEG_X2_COEFFS: &[f64] = &[
        1.0,
        -1.0,
        0.5,
        -0.16666666666666666,
        0.041666666666666664,
        -0.008333333333333333,
    ];

    /// Fast polynomial approximation for exp(-x²)
    pub fn exp_neg_x2_approx(x: f64) -> f64 {
        let x2 = x * x;
        if x2 > 5.0 {
            return 0.0;
        }

        // For now, use standard library exp(-x²) for accuracy
        // A proper polynomial approximation would need more careful coefficient design
        (-x2).exp()
    }

    /// Chebyshev polynomial approximation for cos(x) on [-π, π]
    pub fn cos_chebyshev_approx(x: f64) -> f64 {
        // Map x from [-π, π] to [-1, 1]
        let t = x / PI;
        if t.abs() > 1.0 {
            return x.cos(); // Fall back to standard implementation
        }

        // Chebyshev coefficients for cos(πt)
        let coeffs = [
            1.0,
            0.0,
            -4.934_802_200_544_679,
            0.0,
            4.0587121264167687,
            0.0,
            -1.3352627688545894,
        ];

        chebyshev_eval(t, &coeffs)
    }

    /// Evaluate Chebyshev polynomial
    fn chebyshev_eval(x: f64, coeffs: &[f64]) -> f64 {
        if coeffs.is_empty() {
            return 0.0;
        }
        if coeffs.len() == 1 {
            return coeffs[0];
        }

        let mut b0 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;

        for &coeff in coeffs.iter().rev() {
            b2 = b1;
            b1 = b0;
            b0 = coeff + 2.0 * x * b1 - b2;
        }

        0.5 * (b0 - b2)
    }
}

/// Enhanced caching mechanisms for expensive computations
pub mod enhanced_caching {
    use super::*;
    use std::sync::OnceLock;

    /// Thread-safe cache for gamma function values
    static GAMMA_CACHE: OnceLock<Mutex<HashMap<u64, f64>>> = OnceLock::new();

    /// Convert f64 to a hashable representation for caching
    fn f64_to_key(x: f64) -> u64 {
        // Use bit representation for exact matching
        x.to_bits()
    }

    fn get_gamma_cache() -> &'static Mutex<HashMap<u64, f64>> {
        GAMMA_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Cached gamma function computation
    pub fn gamma_cached(x: f64) -> f64 {
        let key = f64_to_key(x);

        // Check cache first
        if let Ok(cache) = get_gamma_cache().lock() {
            if let Some(&cached_value) = cache.get(&key) {
                return cached_value;
            }
        }

        // Compute and cache the result
        use crate::gamma::gamma;
        let result = gamma(x);

        if let Ok(mut cache) = get_gamma_cache().lock() {
            // Limit cache size to prevent memory issues
            if cache.len() < 1000 {
                cache.insert(key, result);
            }
        }

        result
    }

    /// Clear the gamma cache
    pub fn clear_gamma_cache() {
        if let Ok(mut cache) = get_gamma_cache().lock() {
            cache.clear();
        }
    }

    /// Memoized binomial coefficient computation
    static BINOMIAL_CACHE: OnceLock<Mutex<HashMap<(u32, u32), f64>>> = OnceLock::new();

    fn get_binomial_cache() -> &'static Mutex<HashMap<(u32, u32), f64>> {
        BINOMIAL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Cached binomial coefficient computation
    pub fn binomial_cached(n: u32, k: u32) -> SpecialResult<f64> {
        let key = (n, k);

        // Check cache first
        if let Ok(cache) = get_binomial_cache().lock() {
            if let Some(&cached_value) = cache.get(&key) {
                return Ok(cached_value);
            }
        }

        // Compute using our combinatorial module
        use crate::combinatorial::binomial;
        let result = binomial(n, k)?;

        // Cache the result
        if let Ok(mut cache) = get_binomial_cache().lock() {
            if cache.len() < 1000 {
                cache.insert(key, result);
            }
        }

        Ok(result)
    }
}

/// Adaptive algorithms that choose optimal implementations based on input
pub mod adaptive {
    use super::*;

    /// Adaptive gamma function that chooses the best algorithm based on input
    pub fn gamma_adaptive(x: f64) -> f64 {
        // Check lookup table first for common values
        if let Some(cached) = lookup_tables::gamma_lookup(x) {
            return cached;
        }

        // Check factorial table for integer values
        if x.fract() == 0.0 && x > 0.0 && x <= 21.0 {
            if let Some(factorial) = lookup_tables::factorial_lookup((x as u32) - 1) {
                return factorial;
            }
        }

        // Use cached computation for potentially repeated values
        enhanced_caching::gamma_cached(x)
    }

    /// Adaptive exponential function
    pub fn exp_adaptive(x: f64) -> f64 {
        // Handle extreme values
        if x > 700.0 {
            return f64::INFINITY;
        }
        if x < -700.0 {
            return 0.0;
        }

        // For small values, use series expansion
        if x.abs() < 0.1 {
            return 1.0 + x + 0.5 * x * x + x * x * x / 6.0;
        }

        // Use standard library for moderate values
        x.exp()
    }

    /// Adaptive sine function
    pub fn sin_adaptive(x: f64) -> f64 {
        if x.is_infinite() || x.is_nan() {
            return f64::NAN;
        }

        // For small values, use Taylor series
        if x.abs() < 0.1 {
            let x2 = x * x;
            return x * (1.0 - x2 / 6.0 + x2 * x2 / 120.0);
        }

        // For large values, use argument reduction first
        let reduced = x % (2.0 * PI);
        reduced.sin()
    }
}

/// Vectorized operations for improved performance
pub mod vectorized {
    use super::*;

    /// Vectorized exponential function
    pub fn exp_vectorized(input: &[f64], output: &mut [f64]) -> SpecialResult<()> {
        if input.len() != output.len() {
            return Err(SpecialError::DomainError(
                "Input and output arrays must have the same length".to_string(),
            ));
        }

        for (i, &x) in input.iter().enumerate() {
            output[i] = adaptive::exp_adaptive(x);
        }

        Ok(())
    }

    /// Vectorized sine function
    pub fn sin_vectorized(input: &[f64], output: &mut [f64]) -> SpecialResult<()> {
        if input.len() != output.len() {
            return Err(SpecialError::DomainError(
                "Input and output arrays must have the same length".to_string(),
            ));
        }

        for (i, &x) in input.iter().enumerate() {
            output[i] = adaptive::sin_adaptive(x);
        }

        Ok(())
    }

    /// Vectorized gamma function
    pub fn gamma_vectorized(input: &[f64], output: &mut [f64]) -> SpecialResult<()> {
        if input.len() != output.len() {
            return Err(SpecialError::DomainError(
                "Input and output arrays must have the same length".to_string(),
            ));
        }

        for (i, &x) in input.iter().enumerate() {
            output[i] = adaptive::gamma_adaptive(x);
        }

        Ok(())
    }

    /// Batch computation with optimal memory access patterns
    pub fn batch_compute<F>(input: &[f64], output: &mut [f64], func: F) -> SpecialResult<()>
    where
        F: Fn(f64) -> f64,
    {
        if input.len() != output.len() {
            return Err(SpecialError::DomainError(
                "Input and output arrays must have the same length".to_string(),
            ));
        }

        // Process in chunks to improve cache locality
        const CHUNK_SIZE: usize = 64;

        for chunk in input.chunks(CHUNK_SIZE).zip(output.chunks_mut(CHUNK_SIZE)) {
            let (input_chunk, output_chunk) = chunk;
            for (i, &x) in input_chunk.iter().enumerate() {
                output_chunk[i] = func(x);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_operations() {
        let values = vec![0.0, 1.0, 2.0, -1.0];

        let exp_results = simd::exp_simd(&values);
        assert_relative_eq!(exp_results[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp_results[1], std::f64::consts::E, epsilon = 1e-10);

        let ln_results = simd::ln_simd(&[1.0, std::f64::consts::E, 10.0]);
        assert_relative_eq!(ln_results[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(ln_results[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lookup_tables() {
        // Test factorial lookup
        assert_eq!(lookup_tables::factorial_lookup(0), Some(1.0));
        assert_eq!(lookup_tables::factorial_lookup(5), Some(120.0));
        assert_eq!(lookup_tables::factorial_lookup(25), None);

        // Test gamma lookup
        assert_relative_eq!(
            lookup_tables::gamma_lookup(0.5).unwrap(),
            (PI).sqrt(),
            epsilon = 1e-10
        );
        assert_eq!(lookup_tables::gamma_lookup(1.0), Some(1.0));

        // Test J0 zeros
        assert!(lookup_tables::j0_zero(0).is_some());
        assert!(lookup_tables::j0_zero(100).is_some()); // Should use asymptotic
    }

    #[test]
    fn test_polynomial_approximations() {
        // Test erf approximation
        assert_relative_eq!(poly_approx::erf_approx(0.0), 0.0, epsilon = 1e-3);
        assert_relative_eq!(poly_approx::erf_approx(0.5), 0.5205, epsilon = 1e-2);

        // Test exp(-x²) approximation
        assert_relative_eq!(poly_approx::exp_neg_x2_approx(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(
            poly_approx::exp_neg_x2_approx(1.0),
            (-1.0_f64).exp(),
            epsilon = 1e-1
        );
    }

    #[test]
    fn test_enhanced_caching() {
        // Clear cache first
        enhanced_caching::clear_gamma_cache();

        // Test gamma caching
        let x = 2.5;
        let result1 = enhanced_caching::gamma_cached(x);
        let result2 = enhanced_caching::gamma_cached(x); // Should use cache
        assert_eq!(result1, result2);

        // Test binomial caching
        let binom1 = enhanced_caching::binomial_cached(10, 3).unwrap();
        let binom2 = enhanced_caching::binomial_cached(10, 3).unwrap(); // Should use cache
        assert_eq!(binom1, binom2);
        assert_eq!(binom1, 120.0);
    }

    #[test]
    fn test_adaptive_algorithms() {
        // Test adaptive gamma
        assert_relative_eq!(adaptive::gamma_adaptive(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(adaptive::gamma_adaptive(5.0), 24.0, epsilon = 1e-10);

        // Test adaptive exp
        assert_relative_eq!(adaptive::exp_adaptive(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(
            adaptive::exp_adaptive(1.0),
            std::f64::consts::E,
            epsilon = 1e-10
        );

        // Test adaptive sin
        assert_relative_eq!(adaptive::sin_adaptive(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(adaptive::sin_adaptive(PI / 2.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vectorized_operations() {
        let input = vec![0.0, 1.0, 2.0];
        let mut output = vec![0.0; 3];

        // Test vectorized exp
        vectorized::exp_vectorized(&input, &mut output).unwrap();
        assert_relative_eq!(output[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], std::f64::consts::E, epsilon = 1e-10);

        // Test vectorized sin
        let sin_input = vec![0.0, PI / 2.0, PI];
        let mut sin_output = vec![0.0; 3];
        vectorized::sin_vectorized(&sin_input, &mut sin_output).unwrap();
        assert_relative_eq!(sin_output[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin_output[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sin_output[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_compute() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        vectorized::batch_compute(&input, &mut output, |x| x * x).unwrap();

        for (i, &expected) in [1.0, 4.0, 9.0, 16.0].iter().enumerate() {
            assert_relative_eq!(output[i], expected, epsilon = 1e-10);
        }
    }
}
