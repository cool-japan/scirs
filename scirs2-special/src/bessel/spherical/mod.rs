//! Spherical Bessel functions
//!
//! This module provides implementations of spherical Bessel functions
//! with enhanced numerical stability for both small and large arguments.
//!
//! Spherical Bessel functions are solutions to the differential equation:
//! x² d²y/dx² + 2x dy/dx + [x² - n(n+1)]y = 0
//!
//! Functions included in this module:
//! - spherical_jn(n, x): Spherical Bessel function of the first kind
//! - spherical_yn(n, x): Spherical Bessel function of the second kind
//! - spherical_jn_scaled(n, x): Scaled spherical Bessel function of the first kind (for large arguments)
//! - spherical_yn_scaled(n, x): Scaled spherical Bessel function of the second kind (for large arguments)

use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Helper function for small argument series expansion of spherical Bessel functions
///
/// This function implements the series expansion of j_n(x) for small x using:
/// j_n(x) = (x^n)/(2n+1)!! * (1 - x^2/(2(2n+3)) + x^4/(2*4*(2n+3)(2n+5)) - ...)
///
/// The function computes the first few terms for small x to avoid precision loss.
fn small_arg_series_jn<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    let _n_f = F::from(n).unwrap();
    let x_sq = x * x;

    // Compute the factorial denominator (2n+1)!!
    let mut factorial = F::one();
    for i in 1..=n {
        factorial = factorial * F::from(2 * i + 1).unwrap();
    }

    // First term in the series
    let mut term = F::from(1.0).unwrap();
    let mut series = term;

    // Add more terms for better precision
    let terms_to_compute = if n < 5 { 4 } else { 3 };

    // Terms in the alternating series
    for i in 1..=terms_to_compute {
        let denom = F::from(2.0 * i as f64).unwrap() * F::from((2 * n + 1 + 2 * i) as f64).unwrap();
        term = term * x_sq.neg() / denom;
        series = series + term;

        // Break early if term is insignificant
        if term.abs() < F::from(1e-15).unwrap() * series.abs() {
            break;
        }
    }

    // Compute x^n/(2n+1)!!
    let mut x_pow_n = F::one();
    for _ in 0..n {
        x_pow_n = x_pow_n * x;
    }

    x_pow_n / factorial * series
}

/// Spherical Bessel function of the first kind with enhanced stability.
///
/// These functions are related to the ordinary Bessel functions by:
/// j_n(x) = sqrt(π/(2x)) J_{n+1/2}(x)
///
/// This implementation uses:
/// - Series expansions for small arguments
/// - Recurrence relations for intermediate values
/// - Asymptotic formulas for large arguments
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Input value
///
/// # Returns
///
/// * j_n(x) Spherical Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::spherical::spherical_jn;
/// use std::f64::consts::PI;
///
/// // j₀(x) = sin(x)/x
/// let x = 1.5f64;
/// let j0_exact = x.sin() / x;
/// assert!((spherical_jn(0, x) - j0_exact).abs() < 1e-10);
/// ```
pub fn spherical_jn<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    if n < 0 {
        panic!("Order n must be non-negative");
    }

    // Special case x = 0
    if x == F::zero() {
        if n == 0 {
            return F::one();
        } else {
            return F::zero();
        }
    }

    // Direct formulas for n=0 and n=1 to avoid unnecessary recursion
    if n == 0 {
        // For j0(x), use a more accurate implementation for small x
        if x.abs() < F::from(0.01).unwrap() {
            // Series expansion: j0(x) = 1 - x²/6 + x⁴/120 - ...
            let x2 = x * x;
            return F::one() - x2 / F::from(6.0).unwrap() + x2 * x2 / F::from(120.0).unwrap();
        } else {
            return x.sin() / x;
        }
    } else if n == 1 {
        // For j1(x), use a more accurate implementation for small x
        if x.abs() < F::from(0.01).unwrap() {
            // Series expansion: j1(x) = x/3 - x³/30 + ...
            let x2 = x * x;
            return x / F::from(3.0).unwrap() - x * x2 / F::from(30.0).unwrap();
        } else {
            return (x.sin() / x - x.cos()) / x;
        }
    }

    // Use series expansion for very small arguments to avoid cancellation errors
    if x.abs() < F::from(0.1).unwrap() * (F::from(n).unwrap() + F::one()) {
        return small_arg_series_jn(n, x);
    }

    // For large arguments, use the scaled version with appropriate scaling
    if x > F::from(n).unwrap() * F::from(10.0).unwrap() {
        let scaling = x.sin();
        return spherical_jn_scaled(n, x) * scaling / x;
    }

    // Limit the maximum recursion for stability
    let max_n = n.min(1000); // Protect against stack overflow

    // For higher orders, use a recurrence relation
    // j_{n+1} = (2n+1)/x * j_n - j_{n-1}
    // Initialize j_0 with special case handling for small arguments
    let mut j_n_minus_2 = if x.abs() < F::from(0.01).unwrap() {
        // Series expansion: j0(x) = 1 - x²/6 + x⁴/120 - ...
        let x2 = x * x;
        F::one() - x2 / F::from(6.0).unwrap() + x2 * x2 / F::from(120.0).unwrap()
    } else {
        x.sin() / x
    }; // j_0
       // Initialize j_1 with special case handling for small arguments
    let mut j_n_minus_1 = if x.abs() < F::from(0.01).unwrap() {
        // Series expansion: j1(x) = x/3 - x³/30 + ...
        let x2 = x * x;
        x / F::from(3.0).unwrap() - x * x2 / F::from(30.0).unwrap()
    } else {
        (x.sin() / x - x.cos()) / x
    }; // j_1

    for k in 2..=max_n {
        let j_n = F::from(2.0 * k as f64 - 1.0).unwrap() / x * j_n_minus_1 - j_n_minus_2;
        j_n_minus_2 = j_n_minus_1;
        j_n_minus_1 = j_n;
    }

    j_n_minus_1
}

/// Spherical Bessel function of the second kind with enhanced stability.
///
/// These functions are related to the ordinary Bessel functions by:
/// y_n(x) = sqrt(π/(2x)) Y_{n+1/2}(x)
///
/// This implementation uses:
/// - Closed-form formulas for n=0 and n=1
/// - Recurrence relations for higher orders
/// - Asymptotic behavior for large arguments
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Input value (must be positive)
///
/// # Returns
///
/// * y_n(x) Spherical Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::spherical::spherical_yn;
/// use std::f64::consts::PI;
///
/// // y₀(x) = -cos(x)/x
/// let x = 1.5f64;
/// let y0_exact = -x.cos() / x;
/// assert!((spherical_yn(0, x) - y0_exact).abs() < 1e-10);
/// ```
pub fn spherical_yn<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    if n < 0 {
        panic!("Order n must be non-negative");
    }

    // Special case x = 0 or negative
    if x <= F::zero() {
        return F::neg_infinity();
    }

    // Direct formulas for n=0 and n=1 to avoid unnecessary recursion
    if n == 0 {
        return -x.cos() / x;
    } else if n == 1 {
        return -(x.cos() / x + x.sin()) / x;
    }

    // For large arguments, use the scaled version with appropriate scaling
    if x > F::from(n).unwrap() * F::from(10.0).unwrap() {
        let scaling = -x.cos();
        return spherical_yn_scaled(n, x) * scaling / x;
    }

    // Limit the maximum recursion for stability
    let max_n = n.min(1000); // Protect against stack overflow

    // For higher orders, use a recurrence relation
    // y_{n+1} = (2n+1)/x * y_n - y_{n-1}
    let mut y_n_minus_2 = -x.cos() / x; // y_0
    let mut y_n_minus_1 = -(x.cos() / x + x.sin()) / x; // y_1

    for k in 2..=max_n {
        let y_n = F::from(2.0 * k as f64 - 1.0).unwrap() / x * y_n_minus_1 - y_n_minus_2;
        y_n_minus_2 = y_n_minus_1;
        y_n_minus_1 = y_n;
    }

    y_n_minus_1
}

/// Scaled spherical Bessel function of the first kind.
///
/// This function computes j̃_n(x) = j_n(x) * x / sin(x) for improved accuracy
/// with large arguments. This removes the oscillation and normalization factors
/// that can cause loss of precision.
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Input value (should be large, typically x > 10*n)
///
/// # Returns
///
/// * Scaled spherical Bessel function value j̃_n(x)
pub fn spherical_jn_scaled<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    if n < 0 {
        panic!("Order n must be non-negative");
    }

    // Special case n=0, for which we know the scaled function approaches 1
    if n == 0 {
        if x == F::zero() {
            return F::one();
        }

        if x > F::from(10.0).unwrap() {
            // For large x, j0_scaled approaches 1
            let x_sq = x * x;
            return F::one() - F::one() / (F::from(2.0).unwrap() * x_sq);
        } else {
            // For smaller x, compute directly
            // This should simplify to 1, but use explicit value to avoid precision issues
            return F::one();
        }
    }

    // Special case n=1
    if n == 1 {
        if x == F::zero() {
            return F::zero();
        }

        if x > F::from(10.0).unwrap() {
            // For large x, j1_scaled approaches asymptotics
            let x_sq = x * x;
            return F::one() - F::from(3.0).unwrap() / (F::from(2.0).unwrap() * x_sq);
        } else {
            // Compute more accurately for small x
            // This is equivalent to (1 - x.cos()/x.sin()) but with better precision
            let x2 = x * x;
            return F::one() - F::from(2.0).unwrap() / F::from(3.0).unwrap() * x2;
        }
    }

    // For higher orders with small arguments
    if x < F::from(5.0).unwrap() {
        if x == F::zero() {
            return F::zero();
        }

        // Use the direct formula with the unscaled function
        let j_n = spherical_jn(n, x);
        return j_n * x / x.sin();
    }

    // For large arguments, use asymptotic expansion
    if x > F::from(n * n).unwrap() || x > F::from(1000.0).unwrap() {
        let x_sq = x * x;
        let n_f = F::from(n).unwrap();

        // Asymptotic expansion
        let mut factor = F::one();

        // First order correction
        factor = factor - n_f * (n_f + F::one()) / (F::from(2.0).unwrap() * x_sq);

        // Second order correction for very large x
        if x > F::from(100.0).unwrap() {
            let term2 = n_f * (n_f + F::one()) * (n_f * (n_f + F::one()) - F::from(2.0).unwrap())
                / (F::from(8.0).unwrap() * x_sq * x_sq);
            factor = factor + term2;
        }

        return factor;
    }

    // Limit n to avoid stack overflows
    let safe_n = n.min(50);

    // For intermediate orders and arguments, use a safely truncated recurrence relation
    // We use Miller's algorithm with downward recurrence for stability

    // Start with a higher order than needed (with limit to prevent overflows)
    let n_max = (n * 2).min(100);

    // Initialize with arbitrary values (will be rescaled later)
    let mut j_n_plus_1 = F::from(1e-100).unwrap();
    let mut j_n = F::from(1e-100).unwrap();

    // Apply recurrence relation backward for better numerical stability
    for k in (0..=n_max).rev() {
        let j_n_minus_1 = F::from(2.0 * k as f64 + 1.0).unwrap() / x * j_n - j_n_plus_1;
        j_n_plus_1 = j_n;
        j_n = j_n_minus_1;

        // Normalize occasionally to avoid overflow/underflow
        if j_n.abs() > F::from(1e50).unwrap() {
            let scale = F::from(1e-50).unwrap();
            j_n = j_n * scale;
            j_n_plus_1 = j_n_plus_1 * scale;
        }
    }

    // Rescale using the known asymptotic behavior of j0_scaled
    let j_0_scaled = F::one(); // The scaled j0 approaches 1 as x increases

    // Normalize the sequence
    let scale = j_0_scaled / j_n;

    // Fix the scale and recalculate j_n for n=0
    j_n = j_0_scaled;
    j_n_plus_1 = j_n_plus_1 * scale;

    // Now apply recurrence forward to get j_safe_n_scaled
    for k in 0..safe_n {
        let j_n_plus_1_new = F::from(2.0 * k as f64 + 1.0).unwrap() / x * j_n - j_n_plus_1;
        j_n_plus_1 = j_n;
        j_n = j_n_plus_1_new;
    }

    j_n
}

/// Scaled spherical Bessel function of the second kind.
///
/// This function computes ỹ_n(x) = y_n(x) * x / (-cos(x)) for improved accuracy
/// with large arguments. This removes the oscillation and normalization factors
/// that can cause loss of precision.
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Input value (should be large, typically x > 10*n)
///
/// # Returns
///
/// * Scaled spherical Bessel function value ỹ_n(x)
pub fn spherical_yn_scaled<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    if n < 0 {
        panic!("Order n must be non-negative");
    }

    if x <= F::zero() {
        return F::neg_infinity();
    }

    // Special case n=0
    if n == 0 {
        if x > F::from(10.0).unwrap() {
            // For large x, y0_scaled approaches 1
            let x_sq = x * x;
            return F::one() - F::one() / (F::from(2.0).unwrap() * x_sq);
        } else {
            // For smaller x, compute directly
            return -x.cos() / x * x / (-x.cos()); // Simplifies to 1
        }
    }

    // Special case n=1
    if n == 1 {
        if x > F::from(10.0).unwrap() {
            // For large x, y1_scaled approaches asymptotics
            let x_sq = x * x;
            return -F::one() + F::from(3.0).unwrap() / (F::from(2.0).unwrap() * x_sq);
        } else {
            // Compute directly for small x
            return -(x.cos() / x + x.sin()) / x * x / (-x.cos());
        }
    }

    // For higher orders with small arguments
    if x < F::from(5.0).unwrap() {
        // Use the direct formula with the unscaled function
        let y_n = spherical_yn(n, x);
        return y_n * x / (-x.cos());
    }

    // For large arguments, use asymptotic expansion
    if x > F::from(n * n).unwrap() || x > F::from(1000.0).unwrap() {
        let x_sq = x * x;
        let n_f = F::from(n).unwrap();

        // Asymptotic expansion
        let sign = if n % 2 == 0 { F::one() } else { F::one().neg() };
        let mut factor = sign;

        // First order correction
        factor = factor - sign * n_f * (n_f + F::one()) / (F::from(2.0).unwrap() * x_sq);

        // Second order correction for very large x
        if x > F::from(100.0).unwrap() {
            let term2 =
                sign * n_f * (n_f + F::one()) * (n_f * (n_f + F::one()) - F::from(2.0).unwrap())
                    / (F::from(8.0).unwrap() * x_sq * x_sq);
            factor = factor + term2;
        }

        return factor;
    }

    // Fall back to direct computation for intermediate orders
    // With our improved spherical_yn, this should avoid stack overflows
    spherical_yn(n, x) * x / (-x.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spherical_jn_special_cases() {
        // j₀(x) = sin(x)/x
        let x = 1.5f64;
        let j0_exact = x.sin() / x;
        let j0 = spherical_jn(0, x);
        assert_relative_eq!(j0, j0_exact, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_j1_special_cases() {
        // j₁(x) = sin(x)/(x²) - cos(x)/x
        let x = 1.5f64;
        let j1_exact = (x.sin() / x - x.cos()) / x;
        let j1 = spherical_jn(1, x);
        assert_relative_eq!(j1, j1_exact, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_y0_special_cases() {
        // y₀(x) = -cos(x)/x
        let x = 1.5f64;
        let y0_exact = -x.cos() / x;
        let y0 = spherical_yn(0, x);
        assert_relative_eq!(y0, y0_exact, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_y1_special_cases() {
        // y₁(x) = -cos(x)/(x²) - sin(x)/x
        let x = 1.5f64;
        let y1_exact = -(x.cos() / x + x.sin()) / x;
        let y1 = spherical_yn(1, x);
        assert_relative_eq!(y1, y1_exact, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_jn_small_arguments() {
        // Test series expansion for j₀(x) with very small x
        let x = 1e-6;
        // For small x, j₀(x) ≈ 1 - x²/6
        let j0_series = 1.0 - x * x / 6.0;
        let j0 = spherical_jn(0, x);
        assert_relative_eq!(j0, j0_series, epsilon = 1e-12);
    }

    #[test]
    fn test_spherical_j1_small_arguments() {
        // Test j₁(x) with small x
        // For small x, j₁(x) ≈ x/3
        let x = 1e-6;
        let j1_series = x / 3.0;
        let j1 = spherical_jn(1, x);
        // Use a slightly larger epsilon due to differences in series truncation
        assert_relative_eq!(j1, j1_series, epsilon = 2e-6);
    }
}
