//! Core gamma function implementations

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::validation::check_finite;
use std::fmt::{Debug, Display};

use super::approximations::{
    improved_lanczos_gamma, improved_lanczos_gammaln, stirling_approximation,
    stirling_approximation_ln,
};
use super::constants;
use super::utils::{
    asymptotic_gamma_large_negative, enhanced_log_sin_pi_x, enhanced_reflection_sign,
    stable_gamma_near_large_negative_integer,
};

/// Gamma function with enhanced numerical stability and comprehensive domain handling.
///
/// This implementation uses mathematically rigorous algorithms to compute Γ(z) across
/// the entire complex plane, with special attention to numerical stability near
/// singularities and in extreme parameter ranges.
///
/// ## Mathematical Definition
///
/// **Primary Definition** (Euler's integral of the second kind):
/// ```text
/// Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt,    Re(z) > 0
/// ```
///
/// **Analytic Continuation**: For Re(z) ≤ 0, Γ(z) is defined using the functional equation:
/// ```text
/// Γ(z) = Γ(z+n)/[z(z+1)...(z+n-1)]
/// ```
/// where n is chosen such that Re(z+n) > 0.
///
/// ## Key Mathematical Properties
///
/// 1. **Functional Equation**: Γ(z+1) = z·Γ(z)
///    - **Derivation**: From integration by parts on the defining integral
///    - **Application**: Relates factorial to gamma: n! = Γ(n+1)
///
/// 2. **Reflection Formula**: Γ(z)Γ(1-z) = π/sin(πz)
///    - **Application**: Computes Γ(z) for Re(z) < 0 using values with Re(z) > 0
///    - **Poles**: Function has simple poles at z = 0, -1, -2, -3, ...
///
/// 3. **Special Values**:
///    - Γ(1) = 1
///    - Γ(1/2) = √π
///    - Γ(n) = (n-1)! for positive integers n
///    - Γ(n+1/2) = (2n-1)!!·√π/2ⁿ for non-negative integers n
///
/// # Arguments
///
/// * `x` - Input value of type F (generic floating-point type)
///
/// # Returns
///
/// * Value of Γ(x) as type F
/// * Returns NaN for invalid inputs (e.g., negative integers)
/// * Returns appropriate infinities for pole behavior
///
/// # Examples
///
/// ```
/// use scirs2_special::gamma;
///
/// // Standard cases
/// assert!((gamma(1.0f64) - 1.0).abs() < 1e-10);
/// assert!((gamma(5.0f64) - 24.0).abs() < 1e-10);
///
/// // Half-integer case
/// assert!((gamma(0.5f64) - std::f64::consts::PI.sqrt()).abs() < 1e-10);
/// ```
pub fn gamma<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(x: F) -> F {
    // Special cases
    if x.is_nan() {
        return F::nan();
    }

    if x == F::zero() {
        return F::infinity();
    }

    // Enhanced handling for very small positive values with higher-order series
    // around x=0: Γ(x) = 1/x - γ + (γ²/2 + π²/12)x - (γ³/6 + π²γ/12 + ψ₂(1)/6)x² + O(x³)
    if x > F::zero() && x < F::from(1e-8).expect("Failed to convert constant to float") {
        let gamma_euler = F::from(constants::EULER_MASCHERONI).expect("Failed to convert to float");
        let pi_squared = F::from(std::f64::consts::PI * std::f64::consts::PI)
            .expect("Failed to convert to float");

        // Enhanced series expansion with more terms for better accuracy
        let c0 = F::one() / x; // Leading singular term
        let c1 = -gamma_euler; // Linear term
        let c2 = F::from(0.5).expect("Failed to convert constant to float")
            * (gamma_euler * gamma_euler
                + pi_squared / F::from(6.0).expect("Failed to convert constant to float")); // Quadratic term

        // Third-order term for extreme precision near zero
        let psi2_1 = F::from(2.4041138063191885).expect("Failed to convert constant to float"); // ψ₂(1) = π²/6 + 2ζ(3) where ζ(3) ≈ 1.202
        let c3 = -(gamma_euler * gamma_euler * gamma_euler
            / F::from(6.0).expect("Failed to convert constant to float")
            + pi_squared * gamma_euler
                / F::from(12.0).expect("Failed to convert constant to float")
            + psi2_1 / F::from(6.0).expect("Failed to convert constant to float"));

        return c0 + c1 + c2 * x + c3 * x * x;
    }

    let x_f64 = x.to_f64().expect("Operation failed");

    // For negative x - Enhanced numerical stability for extreme values
    if x < F::zero() {
        // Check if x is very close to a negative integer
        let nearest_int = x_f64.round() as i32;
        if nearest_int <= 0 && (x_f64 - nearest_int as f64).abs() < 1e-14 {
            return F::nan(); // At negative integers, gamma is undefined
        }

        // Enhanced handling for extreme negative values with better overflow protection
        if x < F::from(-1000.0).expect("Failed to convert constant to float") {
            // For very large negative values, use asymptotic expansion
            // with enhanced precision to avoid catastrophic cancellation
            return asymptotic_gamma_large_negative(x);
        }

        // For values very close to negative integers, use enhanced series approximation
        if nearest_int <= 0 && (x_f64 - nearest_int as f64).abs() < 1e-8 {
            // Enhanced expansion with higher-order terms for better accuracy
            let n = -nearest_int;
            let epsilon = x - F::from(nearest_int).expect("Failed to convert to float");

            // Compute n! and H_n with overflow protection
            if n > 100 {
                // Use Stirling's approximation for large factorials
                return stable_gamma_near_large_negative_integer(x, n);
            }

            let mut factorial = F::one();
            let mut harmonic = F::zero();

            for i in 1..=n {
                let i_f = F::from(i).expect("Failed to convert to float");
                factorial = factorial * i_f;
                harmonic += F::one() / i_f;
            }

            let sign = if n % 2 == 0 { F::one() } else { -F::one() };

            // Enhanced series with second-order correction for better accuracy
            let leading_term = sign / (factorial * epsilon);
            let first_correction = F::one() - epsilon * harmonic;

            // Add second-order term: + ε²(H_n² - H_n^(2))/2
            let harmonic_squared_sum = (1..=n)
                .map(|i| 1.0 / ((i * i) as f64))
                .fold(F::zero(), |acc, val| {
                    acc + F::from(val).expect("Failed to convert to float")
                });
            let second_correction =
                epsilon * epsilon * (harmonic * harmonic - harmonic_squared_sum)
                    / F::from(2.0).expect("Failed to convert constant to float");

            return leading_term * (first_correction + second_correction);
        }

        // Enhanced reflection formula with better numerical stability
        let pi = F::from(std::f64::consts::PI).expect("Failed to convert to float");
        let sinpix = (pi * x).sin();

        if sinpix.abs() < F::from(1e-14).expect("Failed to convert constant to float") {
            // x is extremely close to a negative integer
            return F::nan();
        }

        // Apply reflection formula with enhanced overflow protection
        if x < F::from(-100.0).expect("Failed to convert constant to float") {
            // For very negative x, use enhanced logarithmic computation
            // with better condition number handling
            let oneminus_x = F::one() - x;

            // Check if 1-x would cause issues in gammaln
            if oneminus_x > F::from(171.0).expect("Failed to convert constant to float") {
                // Use Stirling approximation directly for better stability
                let log_gamma_1minus_x = stirling_approximation_ln(oneminus_x);
                let log_sinpix = enhanced_log_sin_pi_x(x);
                let log_pi = pi.ln();

                let log_result = log_pi - log_sinpix - log_gamma_1minus_x;

                // Enhanced sign computation for extreme values
                let sign: F = enhanced_reflection_sign(x_f64);

                if log_result < F::from(std::f64::MAX.ln() * 0.9).expect("Operation failed") {
                    return sign * log_result.exp();
                } else {
                    return if sign > F::zero() {
                        F::infinity()
                    } else {
                        F::neg_infinity()
                    };
                }
            } else {
                let log_gamma_1minus_x = gammaln(oneminus_x);
                let log_sinpix = enhanced_log_sin_pi_x(x);
                let log_pi = pi.ln();

                let sign: F = enhanced_reflection_sign(x_f64);
                let log_result = log_pi - log_sinpix - log_gamma_1minus_x;

                if log_result < F::from(std::f64::MAX.ln() * 0.9).expect("Operation failed") {
                    return sign * log_result.exp();
                } else {
                    return if sign > F::zero() {
                        F::infinity()
                    } else {
                        F::neg_infinity()
                    };
                }
            }
        }

        // Standard reflection formula with overflow check
        let gamma_complement = gamma(F::one() - x);
        if gamma_complement.is_infinite() {
            // Handle overflow in reflection formula
            return F::zero();
        }

        return pi / (sinpix * gamma_complement);
    }

    // Handle integer values exactly
    if x_f64.fract() == 0.0 && x_f64 > 0.0 && x_f64 <= 21.0 {
        let n = x_f64 as i32;
        let mut result = F::one();
        for i in 1..(n) {
            result = result * F::from(i).expect("Failed to convert to float");
        }
        return result;
    }

    // Handle half-integer values efficiently.
    //
    // NOTE: `(x_f64 * 2.0).fract() == 0.0` alone is satisfied by BOTH
    // half-integers (n + 0.5) *and* plain integers (2*n is an integer for
    // any integer n), so it must be paired with `x_f64.fract() != 0.0` to
    // exclude whole integers -- otherwise every whole integer x > 21 (which
    // the exact-factorial branch above only handles up to 21.0) would fall
    // into this branch and silently be evaluated as if it were `x - 0.5`
    // instead (e.g. gamma(51.0) would silently return gamma(50.5) ~=
    // 4.29e63 instead of the true gamma(51.0) = 50! ~= 3.04e64, a ~7x
    // error), rather than falling through to the general Lanczos/Stirling
    // path below that correctly handles arbitrary large integers.
    if (x_f64 * 2.0).fract() == 0.0 && x_f64 > 0.0 && x_f64.fract() != 0.0 {
        let n = (x_f64 - 0.5) as i32;
        if n >= 0 {
            // Γ(n + 0.5) = (2n-1)!!/(2^n) * sqrt(π)
            //
            // The direct product (2n-1)!! overflows f64 at n = 151 (it grows
            // much faster than the final result, which is later rescaled
            // down by 2^n) -- e.g. gamma(170.5) has n = 170 and a true value
            // of ~5.56e305 (well within f64 range), but computing the raw
            // double factorial first produced `inf` before the division by
            // 2^n could bring it back down. For large n, delegate to the
            // overflow-safe log-space Stirling approximation instead, which
            // is accurate to <1e-15 relative error for x > 20.
            if n > 140 {
                return stirling_approximation(x);
            }

            let mut double_factorial = F::one();
            for i in 1..=n {
                let double_iminus_1 = match 2_i32.checked_mul(i).and_then(|x| x.checked_sub(1)) {
                    Some(val) => val,
                    None => return F::infinity(), // Handle overflow gracefully
                };
                double_factorial = double_factorial
                    * F::from(double_iminus_1).expect("Failed to convert to float");
            }

            let sqrt_pi = F::from(std::f64::consts::PI.sqrt()).expect("Operation failed");
            let two_pow_n = F::from(2.0_f64.powi(n)).expect("Operation failed");

            return double_factorial / two_pow_n * sqrt_pi;
        }
    }

    // Enhanced threshold for Stirling's approximation with better overflow detection
    if x_f64 > 171.0 {
        return stirling_approximation(x);
    }

    // Additional safety check for potential overflow in Lanczos approximation
    if x_f64 > 150.0 {
        // Check if Lanczos would overflow, if so use Stirling
        let test_lanczos =
            improved_lanczos_gamma(F::from(150.0).expect("Failed to convert constant to float"));
        if test_lanczos.is_infinite()
            || test_lanczos > F::from(1e100).expect("Failed to convert constant to float")
        {
            return stirling_approximation(x);
        }
    }

    // For other values, use the Lanczos approximation with enhanced accuracy
    improved_lanczos_gamma(x)
}

/// Gamma function with full error handling and validation.
///
/// This is the safe version of the gamma function that returns a Result type
/// with comprehensive error handling and validation.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `Ok(gamma(x))` if computation is successful
/// * `Err(SpecialError)` if there's a domain error or computation failure
///
/// # Examples
///
/// ```
/// use scirs2_special::gamma_safe;
///
/// // Valid input
/// let result = gamma_safe(5.0);
/// assert!(result.is_ok());
/// assert!((result.unwrap() - 24.0f64).abs() < 1e-10);
///
/// // Domain error at negative integer
/// let result = gamma_safe(-1.0);
/// assert!(result.is_err());
/// ```
#[allow(dead_code)]
pub fn gamma_safe<F>(x: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + Display + std::ops::AddAssign,
{
    // Validate input
    check_finite(x, "x value")?;

    // Special cases
    if x.is_nan() {
        return Ok(F::nan());
    }

    if x == F::zero() {
        return Ok(F::infinity()); // Gamma(0) = +infinity
    }

    // For negative x, check if it's a negative integer (where gamma is undefined)
    if x < F::zero() {
        let x_f64 = x.to_f64().expect("Operation failed");
        let nearest_int = x_f64.round() as i32;
        if nearest_int <= 0 && (x_f64 - nearest_int as f64).abs() < 1e-14 {
            return Err(SpecialError::DomainError(format!(
                "Gamma function is undefined at negative integer x = {x}"
            )));
        }
    }

    // Use the existing gamma implementation
    let result = gamma(x);

    // Validate output
    if result.is_nan() && !x.is_nan() {
        return Err(SpecialError::ComputationError(format!(
            "Gamma function computation failed for x = {x}"
        )));
    }

    Ok(result)
}

/// Compute the natural logarithm of the gamma function with enhanced numerical stability.
///
/// For x > 0, computes log(Γ(x)) with improved handling of edge cases and numerical accuracy.
///
/// # Arguments
///
/// * `x` - Input value (must be positive)
///
/// # Returns
///
/// * Natural logarithm of the gamma function at x
///
/// # Examples
///
/// ```
/// use scirs2_special::{gammaln, gamma};
///
/// let x = 5.0f64;
/// let gamma_x = gamma(x);
/// let log_gamma_x = gammaln(x);
///
/// assert!((log_gamma_x - gamma_x.ln()).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn gammaln<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(x: F) -> F {
    if x <= F::zero() {
        // For negative x or zero, logarithm of gamma is not defined
        return F::nan();
    }

    // Handle values close to zero specially
    if x < F::from(1e-8).expect("Failed to convert constant to float") {
        // Near zero: log(Γ(x)) ≈ -log(x) - γx + O(x²)
        let gamma_euler = F::from(constants::EULER_MASCHERONI).expect("Failed to convert to float");
        return -x.ln() - gamma_euler * x;
    }

    let x_f64 = x.to_f64().expect("Operation failed");

    // For integer values, we know gamma(n) = (n-1)! so ln(gamma(n)) = ln((n-1)!)
    if x_f64.fract() == 0.0 && x_f64 > 0.0 && x_f64 <= 21.0 {
        let n = x_f64 as i32;
        let mut result = F::zero();
        for i in 1..(n) {
            result += F::from(i).expect("Failed to convert to float").ln();
        }
        return result;
    }

    // For large positive x, use Stirling's approximation directly
    if x_f64 > 50.0 {
        return stirling_approximation_ln(x);
    }

    // For half-integer values, use the specialized implementation.
    //
    // NOTE: `(x_f64 * 2.0).fract() == 0.0` alone is satisfied by BOTH
    // half-integers (n + 0.5) *and* plain integers (2*n is an integer for
    // any integer n), so it must be paired with `x_f64.fract() != 0.0` to
    // exclude whole integers -- otherwise every whole integer x in (21, 50]
    // (the exact-factorial-sum branch above only handles up to 21.0; the
    // `x_f64 > 50.0` Stirling branch above catches everything past 50)
    // would silently be evaluated as if it were `x - 0.5` instead (e.g.
    // `gammaln(26.0)` would silently return `gammaln(25.5) ~= 56.389`
    // instead of the true `gammaln(26.0) ~= 58.004`, i.e. `gamma(26.0)` off
    // by a factor of ~5). Mirrors the identical fix already applied to
    // `gamma()`'s analogous branch above.
    if (x_f64 * 2.0).fract() == 0.0 && x_f64 > 0.0 && x_f64.fract() != 0.0 {
        let n = (x_f64 - 0.5) as i32;
        if n >= 0 {
            // ln(Γ(n + 0.5)) = ln((2n-1)!!) - n*ln(2) + ln(sqrt(π))
            let mut log_double_factorial = F::zero();
            for i in (1..=n).map(|i| 2 * i - 1) {
                log_double_factorial += F::from(i).expect("Failed to convert to float").ln();
            }

            // Use ln(sqrt(π)) NOT ln(sqrt(2π))
            // ln(sqrt(π)) = ln(π)/2 = 0.5723649429247001
            let log_sqrt_pi = F::from(std::f64::consts::PI)
                .expect("Failed to convert to float")
                .ln()
                / F::from(2.0).expect("Failed to convert constant to float");
            let n_log_2 = F::from(n).expect("Failed to convert to float")
                * F::from(std::f64::consts::LN_2).expect("Failed to convert to float");

            return log_double_factorial - n_log_2 + log_sqrt_pi;
        }
    }

    // For other values, use the Lanczos approximation for ln(gamma)
    improved_lanczos_gammaln(x)
}

/// Alias for gammaln function.
#[allow(dead_code)]
pub fn loggamma<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(x: F) -> F {
    gammaln(x)
}

/// Natural logarithm of the beta function with enhanced numerical stability.
///
/// Computes log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
///
/// This implementation provides better handling of:
/// - Very large arguments
/// - Arguments close to zero
/// - Arguments with large disparities in magnitude
///
/// # Arguments
///
/// * `a` - First parameter (must be positive)
/// * `b` - Second parameter (must be positive)
///
/// # Returns
///
/// * Natural logarithm of the beta function value for (a,b)
///
/// # Examples
///
/// ```
/// use scirs2_special::{betaln, beta};
///
/// let a = 5.0;
/// let b = 3.0;
/// // Define type explicitly to avoid ambiguity
/// let beta_ab: f64 = beta(a, b);
/// let log_beta_ab = betaln(a, b);
///
/// assert!((log_beta_ab - beta_ab.ln()).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn betaln<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(a: F, b: F) -> F {
    if a <= F::zero() || b <= F::zero() {
        return F::nan();
    }

    // For small to moderate values, use gammaln directly
    if a <= F::from(100.0).expect("Failed to convert constant to float")
        && b <= F::from(100.0).expect("Failed to convert constant to float")
    {
        let ln_gamma_a = gammaln(a);
        let ln_gamma_b = gammaln(b);
        let ln_gamma_ab = gammaln(a + b);

        // Use careful summation to minimize errors
        // Add the two gamma values and subtract the combined gamma
        return ln_gamma_a + ln_gamma_b - ln_gamma_ab;
    }

    // For very large values, use asymptotic formulas
    // Use Stirling's approximation for each gamma term
    // log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
    let ln_gamma_a = stirling_approximation_ln(a);
    let ln_gamma_b = stirling_approximation_ln(b);
    let ln_gamma_ab = stirling_approximation_ln(a + b);

    ln_gamma_a + ln_gamma_b - ln_gamma_ab
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Regression test for a silent-wrong-result bug: the "handle
    /// half-integer values efficiently" fast path's guard,
    /// `(x*2).fract() == 0.0`, is also satisfied by any WHOLE integer
    /// (doubling an integer is still an integer), so every plain integer
    /// x in (21, 141] used to be silently misidentified as the
    /// half-integer `x - 0.5` and evaluated with the wrong formula (e.g.
    /// `gamma(51.0)` returned `gamma(50.5)` ~= 4.29e63 instead of the true
    /// `50! ~= 3.04e64`, off by a factor of ~7). Uses non-constant,
    /// widely-spaced integers so a stub/constant fallback could not
    /// coincidentally pass.
    #[test]
    fn test_gamma_large_integers_not_misrouted_to_half_integer_path() {
        // Reference factorials: gamma(n) = (n-1)!
        assert_relative_eq!(gamma(22.0_f64), 51090942171709440000.0, max_relative = 1e-9);
        assert_relative_eq!(gamma(25.0_f64), 6.204484017332394e23, max_relative = 1e-9);
        assert_relative_eq!(gamma(51.0_f64), 3.0414093201713376e64, max_relative = 1e-9);
        assert_relative_eq!(gamma(100.0_f64), 9.33262154439441e155, max_relative = 1e-6);
        assert_relative_eq!(gamma(101.0_f64), 9.332621544394415e157, max_relative = 1e-6);
        assert_relative_eq!(gamma(140.0_f64), 9.61572319694109e238, max_relative = 1e-6);

        // Direct check on the exact reported failure mode: gamma(51.0)
        // must NOT equal gamma(50.5).
        let g51 = gamma(51.0_f64);
        let g50_5 = gamma(50.5_f64);
        assert!(
            (g51 - g50_5).abs() / g51 > 0.1,
            "gamma(51.0)={g51} must differ substantially from gamma(50.5)={g50_5}"
        );
    }

    /// True half-integers must still take the fast double-factorial path
    /// and remain correct (this is a non-regression check for the guard
    /// added above).
    #[test]
    fn test_gamma_half_integers_still_correct() {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert_relative_eq!(gamma(0.5_f64), sqrt_pi, epsilon = 1e-10);
        assert_relative_eq!(gamma(1.5_f64), 0.5 * sqrt_pi, epsilon = 1e-10);
        assert_relative_eq!(gamma(2.5_f64), 0.75 * sqrt_pi, epsilon = 1e-10);
        // Gamma(50.5) reference value (scipy.special.gamma(50.5)).
        assert_relative_eq!(gamma(50.5_f64), 4.29046291235196e63, max_relative = 1e-9);
    }

    #[test]
    fn test_gamma_small_integers_exact() {
        assert_relative_eq!(gamma(1.0_f64), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(5.0_f64), 24.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(10.0_f64), 362880.0, epsilon = 1e-6);
        assert_relative_eq!(gamma(21.0_f64), 2432902008176640000.0, max_relative = 1e-9);
    }

    /// Regression test for the exact same silent-wrong-result bug as
    /// `test_gamma_large_integers_not_misrouted_to_half_integer_path`
    /// above, but in `gammaln`: its "half-integer" branch guard was missing
    /// the `x_f64.fract() != 0.0` exclusion that `gamma()`'s analogous
    /// branch already carries, so whole integers x in (21, 50] (the
    /// exact-factorial-sum branch only covers up to 21.0; the `x > 50.0`
    /// Stirling branch covers everything past 50) were silently evaluated
    /// as `gammaln(x - 0.5)` instead -- e.g. the old code returned
    /// `gammaln(26.0) ~= 56.389` (i.e. `gammaln(25.5)`) against the true
    /// `~= 58.004`, off by a factor of ~5 in `gamma(26.0)` terms. This bug
    /// was newly surfaced by `gammaincinv`'s large-a fix (its Newton
    /// derivative now calls `gammaln` directly for every `a`, including
    /// plain integers in this range), which is what caused
    /// `pdtri`/`gammainccinv` round-trip tests to start failing to
    /// converge until this was fixed too.
    #[test]
    fn test_gammaln_large_integers_not_misrouted_to_half_integer_path() {
        // Reference values from `scipy.special.gammaln`.
        assert_relative_eq!(gammaln(22.0_f64), 45.380138898476915, epsilon = 1e-9);
        assert_relative_eq!(gammaln(25.0_f64), 54.78472939811232, epsilon = 1e-9);
        assert_relative_eq!(gammaln(26.0_f64), 58.003605222980525, epsilon = 1e-9);
        assert_relative_eq!(gammaln(30.0_f64), 71.257038967168, epsilon = 1e-9);
        assert_relative_eq!(gammaln(45.0_f64), 125.31727114935688, epsilon = 1e-8);
        assert_relative_eq!(gammaln(50.0_f64), 144.56574394634487, epsilon = 1e-8);

        // Direct check on the exact reported failure mode: gammaln(26.0)
        // must NOT equal gammaln(25.5).
        let ln_g26 = gammaln(26.0_f64);
        let ln_g25_5 = gammaln(25.5_f64);
        assert!(
            (ln_g26 - ln_g25_5).abs() > 1.0,
            "gammaln(26.0)={ln_g26} must differ substantially from gammaln(25.5)={ln_g25_5}"
        );

        // Must also stay self-consistent with `gamma()` across the whole
        // previously-affected range (both were derived from the same
        // Lanczos machinery once routed correctly).
        for n in 22..=50 {
            let x = n as f64;
            assert_relative_eq!(gammaln(x), gamma(x).ln(), max_relative = 1e-9);
        }
    }

    /// True half-integers must still take the fast double-factorial path
    /// and remain correct (non-regression check for the guard above).
    #[test]
    fn test_gammaln_half_integers_still_correct() {
        let ln_sqrt_pi = 0.5 * std::f64::consts::PI.ln();
        assert_relative_eq!(gammaln(0.5_f64), ln_sqrt_pi, epsilon = 1e-10);
        assert_relative_eq!(gammaln(25.5_f64), 56.389167643719944, epsilon = 1e-9);
    }
}
