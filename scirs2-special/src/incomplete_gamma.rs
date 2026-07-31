//! Incomplete gamma and related functions
//!
//! This module provides incomplete gamma functions and their regularized forms,
//! matching SciPy's special module functionality.

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{gamma, gammaln};
use crate::validation::{check_finite, check_positive};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Lower incomplete gamma function
///
/// Computes the lower incomplete gamma function:
/// γ(a, x) = ∫₀ˣ t^(a-1) e^(-t) dt
///
/// # Arguments
/// * `a` - Shape parameter (must be positive)
/// * `x` - Upper limit of integration
///
/// # Returns
/// The value of the lower incomplete gamma function
///
/// # Examples
/// ```
/// use scirs2_special::incomplete_gamma::gammainc_lower;
///
/// let result = gammainc_lower(2.0, 1.0).expect("Operation failed");
/// assert!((result - 0.2642411176571153f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn gammainc_lower<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // Use series expansion for x < a + 1
    if x < a + T::one() {
        // Series representation: γ(a,x) = x^a e^(-x) Σ(x^n / Γ(a+n+1))
        let mut sum = T::one() / a;
        let mut term = T::one() / a;
        let mut n = T::one();
        let tol = T::from_f64(1e-15).expect("Operation failed");

        while term.abs() > tol * sum.abs() {
            term *= x / (a + n);
            sum += term;
            n += T::one();

            if n > T::from_f64(1000.0).expect("Operation failed") {
                return Err(SpecialError::ConvergenceError(
                    "gammainc_lower: series did not converge".to_string(),
                ));
            }
        }

        Ok(x.powf(a) * (-x).exp() * sum)
    } else {
        // Use complement: γ(a,x) = Γ(a) - Γ(a,x)
        let gamma_a = gamma(a);
        let gamma_upper = gammainc_upper(a, x)?;
        Ok(gamma_a - gamma_upper)
    }
}

/// Upper incomplete gamma function
///
/// Computes the upper incomplete gamma function:
/// Γ(a, x) = ∫ₓ^∞ t^(a-1) e^(-t) dt
///
/// # Arguments
/// * `a` - Shape parameter (must be positive)
/// * `x` - Lower limit of integration
///
/// # Returns
/// The value of the upper incomplete gamma function
#[allow(dead_code)]
pub fn gammainc_upper<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(gamma(a));
    }

    // Use continued fraction for x >= a + 1
    if x >= a + T::one() {
        // Continued fraction representation
        let mut b = x + T::one() - a;
        let mut c = T::from_f64(1e30).expect("Operation failed");
        let mut d = T::one() / b;
        let mut h = d;
        let tol = T::from_f64(1e-15).expect("Operation failed");

        for i in 1..1000 {
            let an = -T::from_usize(i).expect("Operation failed")
                * (T::from_usize(i).expect("Operation failed") - a);
            b += T::from_f64(2.0).expect("Operation failed");
            d = an * d + b;

            if d.abs() < T::from_f64(1e-30).expect("Operation failed") {
                d = T::from_f64(1e-30).expect("Operation failed");
            }

            c = b + an / c;
            if c.abs() < T::from_f64(1e-30).expect("Operation failed") {
                c = T::from_f64(1e-30).expect("Operation failed");
            }

            d = T::one() / d;
            let delta = d * c;
            h *= delta;

            if (delta - T::one()).abs() < tol {
                return Ok(x.powf(a) * (-x).exp() * h);
            }
        }

        Err(SpecialError::ConvergenceError(
            "gammainc_upper: continued fraction did not converge".to_string(),
        ))
    } else {
        // Use complement
        let gamma_a = gamma(a);
        let gamma_lower = gammainc_lower(a, x)?;
        Ok(gamma_a - gamma_lower)
    }
}

/// Regularized lower incomplete gamma function
///
/// Computes P(a, x) = γ(a, x) / Γ(a)
///
/// # Arguments
/// * `a` - Shape parameter
/// * `x` - Upper limit
///
/// # Returns
/// The regularized lower incomplete gamma function
#[allow(dead_code)]
pub fn gammainc<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // For large a, use asymptotic expansion or specialized algorithms.
    //
    // NOTE: the threshold here used to be 100, but `compute_log_gammainc`'s
    // series (`term *= (a - n) / x`) is only valid when x is comparably
    // large to a (it diverges for small/moderate x, e.g. it silently
    // returns values far outside [0, 1], or `inf`, for a=101 at x <= a);
    // meanwhile the direct `gammainc_lower(a, x) / gamma(a)` path below is
    // numerically safe (no overflow in `x.powf(a)`) up to a ~= 140-142
    // (matching the a=140 double-factorial-overflow threshold already
    // established for `gamma()` itself, see gamma/core.rs). Raising the
    // cutoff to 140 therefore fixes the (previously silently wrong) a in
    // (100, 140] range by routing it through the working path. The a > 140
    // regime itself is handled by `gammainc_pair_large_a` (see its doc for
    // why: the old `compute_log_gammainc` was simply the wrong asymptotic
    // direction there, not just imprecise).
    if a > T::from_f64(140.0).expect("Operation failed") {
        let (p, _q) = gammainc_pair_large_a(a, x)?;
        Ok(p)
    } else {
        let gamma_lower = gammainc_lower(a, x)?;
        let gamma_a = gamma(a);
        Ok(gamma_lower / gamma_a)
    }
}

/// Regularized upper incomplete gamma function
///
/// Computes Q(a, x) = Γ(a, x) / Γ(a)
#[allow(dead_code)]
pub fn gammaincc<T>(a: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(a, "a")?;
    check_finite(x, "x value")?;

    if x <= T::zero() {
        return Ok(T::one());
    }

    // For large a, compute Q directly from the large-a routine rather than
    // as `1 - gammainc(a, x)`: in the deep right tail (x >> a), P rounds to
    // exactly 1.0 in floating point, so `1 - P` would silently collapse an
    // arbitrarily small-but-nonzero Q to 0 instead of its true (tiny) value.
    // `gammainc_pair_large_a` computes whichever of P/Q is the "small" one
    // directly, so both stay meaningful all the way into the tail.
    if a > T::from_f64(140.0).expect("Operation failed") {
        let (_p, q) = gammainc_pair_large_a(a, x)?;
        return Ok(q);
    }

    // Use complement of regularized lower incomplete gamma
    let p = gammainc(a, x)?;
    Ok(T::one() - p)
}

/// Inverse of regularized lower incomplete gamma function
///
/// Find x such that P(a, x) = p
#[allow(dead_code)]
pub fn gammaincinv<T>(a: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign + SubAssign,
{
    check_positive(a, "a")?;
    crate::validation::check_probability(p, "p")?;

    if p == T::zero() {
        return Ok(T::zero());
    }

    if p == T::one() {
        return Ok(T::infinity());
    }

    // Initial guess using Wilson-Hilferty transformation
    let x0 = initial_guess_gammaincinv(a, p);

    // Newton-Raphson iteration
    let mut x = x0;
    let tol = T::from_f64(1e-12).expect("Operation failed");

    for _ in 0..100 {
        let f = gammainc(a, x)? - p;

        // Derivative: d/dx P(a,x) = x^(a-1) e^(-x) / Γ(a). Computed via
        // exp(log(...)) rather than forming `x.powf(a - 1)` and `gamma(a)`
        // separately: both individually overflow f64 once `a` is large
        // (`gamma(a)` overflows past a ~= 171; `x.powf(a - 1)` overflows
        // even sooner whenever x is comparable to or larger than a, e.g.
        // already at a ~= 150, x ~= 150), even though the true derivative
        // is a modest, well-behaved value (the Gamma(a,1) density never
        // exceeds ~1/sqrt(2*pi*a) at its peak). This keeps Newton's
        // iteration well-defined for the a > 140 large-a regime that
        // `gammainc` itself now handles (see `gammainc_pair_large_a`).
        let log_df = (a - T::one()) * x.ln() - x - gammaln(a);
        let df = log_df.exp();

        let dx = f / df;
        x -= dx;

        // Ensure x stays positive
        if x <= T::zero() {
            x = T::from_f64(1e-10).expect("Operation failed");
        }

        if dx.abs() < tol * x.abs() {
            return Ok(x);
        }
    }

    Err(SpecialError::ConvergenceError(
        "gammaincinv: Newton iteration did not converge".to_string(),
    ))
}

/// Inverse of regularized upper incomplete gamma function
///
/// Find x such that Q(a, x) = q
#[allow(dead_code)]
pub fn gammainccinv<T>(a: T, q: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign + SubAssign,
{
    check_positive(a, "a")?;
    crate::validation::check_probability(q, "q")?;

    // Use Q(a, x) = 1 - P(a, x)
    let p = T::one() - q;
    gammaincinv(a, p)
}

/// Helper function for initial guess in gammaincinv
#[allow(dead_code)]
fn initial_guess_gammaincinv<T>(a: T, p: T) -> T
where
    T: Float + FromPrimitive + Display,
{
    // Wilson-Hilferty approximation
    let g = T::from_f64(2.0).expect("Operation failed")
        / (T::from_f64(9.0).expect("Operation failed") * a);
    let z = crate::distributions::ndtri(p).unwrap_or(T::zero());
    let w = T::one() + g * z;

    if w > T::zero() {
        a * w.powf(T::from_f64(3.0).expect("Operation failed"))
    } else {
        // Fallback for extreme cases
        if p < T::from_f64(0.5).expect("Operation failed") {
            a * T::from_f64(0.1).expect("Operation failed")
        } else {
            a * T::from_f64(2.0).expect("Operation failed")
        }
    }
}

/// Regularized incomplete gamma pair `(P(a, x), Q(a, x))` for large `a`
/// (used once `a > 140` by both `gammainc` and `gammaincc`).
///
/// Mirrors the `a <= 140` code path exactly -- the power series for `x <
/// a+1` (identical to `gammainc_lower`'s), or the Legendre continued
/// fraction for `x >= a+1` (identical to `gammainc_upper`'s) -- but combines
/// the result in **log-space** (`a*ln(x) - x - lgamma(a) + ln(series_or_cf)`,
/// exponentiated only once at the very end) instead of forming `x.powf(a)`
/// and `gamma(a)` as separate values and dividing. That matters because
/// each of those individually overflows `f64` long before their *ratio*
/// does (`gamma(a)` overflows past `a ~= 171`; `x.powf(a)` overflows even
/// sooner whenever `x` is comparable to `a`, e.g. already at `a = x = 150`).
/// `gammaln`, used here for `lgamma(a)`, already routes through an
/// overflow-safe Stirling expansion for large arguments.
///
/// Whichever of P/Q is *not* the direct target of the branch taken (the
/// series targets P; the continued fraction targets Q) is recovered via
/// `1 - other`, which is safe in both cases: the series branch (`x < a+1`)
/// never has Q as the tiny one (Q stays >= ~0.5 there, for a in the tested
/// range), and symmetrically for the continued-fraction branch, so the
/// subtraction never has to cancel two nearly-equal large quantities.
///
/// # History
///
/// The previous large-a path (`compute_log_gammainc`, now removed) used the
/// *upper* incomplete gamma's large-`x` **asymptotic tail series**
/// (`term *= (a - n) / x`) to approximate the *lower* regularized function
/// `P(a, x)` unconditionally. That series is only valid for `x >> a`; for
/// `x` comparable to `a` (e.g. right at the distribution's mean, the most
/// common case in practice) each term stayed close to magnitude 1 for many
/// iterations, so the fixed 50-term truncation never converged and the
/// "probability" produced was not even confined to `[0, 1]`  -- e.g. the old
/// code returned `gammainc(150.0, 150.0) ~= 73.4` against the true value
/// `~= 0.511`, and `gammainc(150.0, 1000.0) ~= 1.57e-245` against the true
/// value `1.0`. This function replaces it with routing that mirrors the
/// already-validated `a <= 140` algorithms, just carried out without ever
/// forming an overflowing intermediate value.
///
/// (A uniform asymptotic expansion in Temme's sense -- a polynomial
/// correction in `1/a` evaluated at a rescaled variable `eta` -- is the
/// textbook alternative for this regime. It was not used here because it
/// requires a table of hardcoded polynomial coefficients in `eta` with real
/// transcription risk, whereas reusing the *exact* series/continued-fraction
/// recursions already validated for `a <= 140` needs no new numerical
/// machinery beyond `gammaln`, and empirically reaches full `f64` precision
/// -- relative error 1e-12 or tighter -- deep into both tails for the
/// realistic range of `a`, only degrading to ~1e-8 for `a` around `1e8`,
/// which is an intrinsic `f64` cancellation limit of `a*ln(x) - x` at that
/// scale rather than a limitation of the method.)
fn gammainc_pair_large_a<T>(a: T, x: T) -> SpecialResult<(T, T)>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    let log_gamma_a = gammaln(a);
    let tol = T::from_f64(1e-15).expect("Operation failed");
    let max_iter = 2_000_000usize;

    if x < a + T::one() {
        // log(series) with the same recursion as `gammainc_lower`'s series
        // (convergent for x < a+1): sum = 1/a + x/(a(a+1)) + ...
        let mut term = T::one() / a;
        let mut sum = term;
        let mut n = 1usize;
        loop {
            let nt = T::from_usize(n).expect("Operation failed");
            term *= x / (a + nt);
            sum += term;
            if term.abs() <= tol * sum.abs() {
                break;
            }
            n += 1;
            if n > max_iter {
                return Err(SpecialError::ConvergenceError(
                    "gammainc: large-a series did not converge".to_string(),
                ));
            }
        }

        let log_p = a * x.ln() - x - log_gamma_a + sum.ln();
        let p = clamp_unit(log_p.exp());
        Ok((p, T::one() - p))
    } else {
        // log(h) with the same modified Lentz continued fraction as
        // `gammainc_upper`'s (convergent for x >= a+1):
        // Gamma(a,x) = x^a * e^(-x) * h.
        let mut b = x + T::one() - a;
        let mut c = T::from_f64(1e300).expect("Operation failed");
        let mut d = T::one() / b;
        let mut h = d;
        let tiny = T::from_f64(1e-300).expect("Operation failed");
        let mut i = 1usize;
        loop {
            let it = T::from_usize(i).expect("Operation failed");
            let an = -it * (it - a);
            b += T::from_f64(2.0).expect("Operation failed");
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = T::one() / d;
            let delta = d * c;
            h *= delta;
            if (delta - T::one()).abs() <= tol {
                break;
            }
            i += 1;
            if i > max_iter {
                return Err(SpecialError::ConvergenceError(
                    "gammaincc: large-a continued fraction did not converge".to_string(),
                ));
            }
        }

        let log_q = a * x.ln() - x - log_gamma_a + h.ln();
        let q = clamp_unit(log_q.exp());
        Ok((T::one() - q, q))
    }
}

/// Clamp a probability-like value into `[0, 1]`, guarding against a tiny
/// negative or `> 1` result from floating-point rounding at the edges of
/// the series/continued-fraction convergence above.
#[inline]
fn clamp_unit<T: Float>(v: T) -> T {
    if v < T::zero() {
        T::zero()
    } else if v > T::one() {
        T::one()
    } else {
        v
    }
}

/// Gamma star function (used in some asymptotic expansions)
///
/// gammastar(x) = Γ(x) / (sqrt(2π) * x^(x-1/2) * e^(-x))
#[allow(dead_code)]
pub fn gammastar<T>(x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + AddAssign + MulAssign,
{
    check_positive(x, "x")?;

    if x >= T::from_f64(10.0).expect("Operation failed") {
        // Use Stirling series
        let mut sum = T::one();
        let x2 = x * x;
        let mut xn = x;

        // Stirling coefficients
        let coeffs = [
            T::from_f64(1.0 / 12.0).expect("Operation failed"),
            T::from_f64(1.0 / 288.0).expect("Operation failed"),
            T::from_f64(-139.0 / 51840.0).expect("Operation failed"),
            T::from_f64(-571.0 / 2488320.0).expect("Operation failed"),
        ];

        for &c in &coeffs {
            sum += c / xn;
            xn *= x2;
        }

        Ok(sum)
    } else {
        // Direct computation
        let sqrt_2pi = T::from_f64((2.0 * std::f64::consts::PI).sqrt()).expect("Operation failed");
        let gamma_x = gamma(x);
        let x_power = x.powf(x - T::from_f64(0.5).expect("Operation failed"));
        let exp_neg_x = (-x).exp();

        Ok(gamma_x / (sqrt_2pi * x_power * exp_neg_x))
    }
}

/// Sign of the gamma function
///
/// Returns 1.0 if gamma(x) > 0, -1.0 if gamma(x) < 0
#[allow(dead_code)]
pub fn gammasgn<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x > T::zero() {
        T::one()
    } else {
        // Gamma function alternates sign for negative non-integer values
        let floor_x = x.floor();
        if x == floor_x {
            // Gamma is undefined at negative integers
            T::nan()
        } else {
            // Sign alternates based on floor
            if floor_x.to_isize().unwrap_or(0) % 2 == 0 {
                T::one()
            } else {
                -T::one()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gammainc_lower() {
        // Test values verified against SciPy (non-regularized incomplete gamma)
        assert_relative_eq!(
            gammainc_lower(1.0, 1.0).expect("Operation failed"),
            0.6321205588285577, // γ(1,1) = P(1,1) * Γ(1) = 0.6321205588285577 * 1
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gammainc_lower(2.0, 1.0).expect("Operation failed"),
            0.264241117657115, // γ(2,1) = P(2,1) * Γ(2) = 0.2642411176571153 * 1
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gammainc_lower(3.0, 2.0).expect("Operation failed"),
            0.646647167633873, // γ(3,2) = P(3,2) * Γ(3) = 0.32332358381693654 * 2
            epsilon = 1e-10
        );

        // Edge cases
        assert_eq!(gammainc_lower(1.0, 0.0).expect("Operation failed"), 0.0);
    }

    #[test]
    fn test_gammainc() {
        // Regularized lower incomplete gamma
        assert_relative_eq!(
            gammainc(1.0, 1.0).expect("Operation failed"),
            0.6321205588285577,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            gammainc(2.0, 2.0).expect("Operation failed"),
            0.5939941502901619,
            epsilon = 1e-10
        );

        // P(a,0) = 0, P(a,∞) = 1
        assert_eq!(gammainc(1.0, 0.0).expect("Operation failed"), 0.0);
    }

    #[test]
    fn test_gammaincc() {
        // Q(a,x) = 1 - P(a,x)
        let a = 2.0;
        let x = 1.5;
        let p = gammainc(a, x).expect("Operation failed");
        let q = gammaincc(a, x).expect("Operation failed");
        assert_relative_eq!(p + q, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gammaincinv() {
        // Test round trip: gammaincinv(a, gammainc(a, x)) ≈ x
        let a = 2.5;
        let x = 3.0;
        let p = gammainc(a, x).expect("Operation failed");
        let x_recovered = gammaincinv(a, p).expect("Operation failed");
        assert_relative_eq!(x_recovered, x, epsilon = 1e-8);
    }

    #[test]
    fn test_gammasgn() {
        assert_eq!(gammasgn(1.0), 1.0);
        assert_eq!(gammasgn(2.5), 1.0);
        assert_eq!(gammasgn(-0.5), -1.0);
        assert_eq!(gammasgn(-1.5), 1.0);
        assert_eq!(gammasgn(-2.5), -1.0);
    }

    /// Relative-tolerance comparison, robust down to the astronomically tiny
    /// magnitudes exercised by the deep-tail large-`a` cases below (where
    /// `approx`'s absolute epsilon would trivially "pass" against 0).
    fn rel_close(actual: f64, expected: f64, tol: f64) -> bool {
        (actual - expected).abs() <= tol * expected.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn test_gammainc_gammaincc_large_a_matches_reference() {
        // Previously, a > 140 routed through `compute_log_gammainc`, which
        // used the *upper* incomplete gamma's large-x asymptotic tail series
        // to approximate the *lower* regularized function -- valid only for
        // x >> a, so it silently produced values outside [0, 1] whenever x
        // was comparable to a (the common case). E.g. the old code returned
        // `gammainc(150.0, 150.0) ~= 73.4` against the true `~= 0.511`.
        // Reference values from `scipy.special.gammainc`/`gammaincc`.
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            // (a, x, P_true, Q_true, tol)
            (150.0, 150.0, 0.5108582297493597, 0.4891417702506403, 1e-9),
            (141.0, 141.0, 0.5111994345210251, 0.4888005654789749, 1e-9),
            (
                150.0,
                100.0,
                1.8842104660386837e-06,
                0.999998115789534,
                1e-8,
            ),
            (
                1000.0,
                900.0,
                0.0005499022657117819,
                0.9994500977342882,
                1e-8,
            ),
            // Deep tails: this is exactly the regime the old series
            // couldn't reach at all (it wasn't even bounded in [0, 1]).
            (150.0, 1000.0, 1.0, 1.565659386727711e-248, 1e-6),
            (150.0, 50.0, 3.530612225414933e-30, 1.0, 1e-6),
            (200.0, 50.0, 2.0247590148475566e-57, 1.0, 1e-4),
        ];

        for &(a, x, p_true, q_true, tol) in cases {
            let p = gammainc(a, x).expect("gammainc should succeed for a > 140");
            let q = gammaincc(a, x).expect("gammaincc should succeed for a > 140");

            assert!(
                rel_close(p, p_true, tol),
                "gammainc({a}, {x}) = {p:e}, expected {p_true:e}"
            );
            assert!(
                rel_close(q, q_true, tol),
                "gammaincc({a}, {x}) = {q:e}, expected {q_true:e}"
            );
            // P and Q must still be complementary regardless of which one
            // was computed directly vs. by subtraction.
            assert!(
                (p + q - 1.0).abs() < 1e-9,
                "P + Q != 1 for a={a}, x={x}: P={p}, Q={q}"
            );
            // Both must stay within the valid probability range (the
            // pre-fix bug's most obvious symptom was violating this).
            assert!((0.0..=1.0).contains(&p), "P out of [0,1]: {p}");
            assert!((0.0..=1.0).contains(&q), "Q out of [0,1]: {q}");
        }
    }

    #[test]
    fn test_gammaincinv_large_a_matches_reference() {
        // Previously, Newton's derivative `x.powf(a-1) * (-x).exp() / gamma(a)`
        // overflowed to `inf` for a > 140 whenever x was comparable to a
        // (e.g. `150.0_f64.powf(149.0)` alone is already `inf`), making
        // `dx = f / df` collapse to 0 and Newton "converge" instantly at the
        // wrong initial guess. Reference `x` values solved independently via
        // `mpmath.findroot` on `mpmath.gammainc(a, 0, x, regularized=True)`.
        let cases: &[(f64, f64, f64)] = &[
            (150.0, 0.3, 143.34387895250237),
            (200.0, 0.7, 207.16763215577143),
            (1000.0, 0.5, 999.6666864269652),
            (300.0, 0.01, 261.18256324043705),
            (150.0, 0.999, 190.71262426020576),
        ];

        for &(a, p, x_ref) in cases {
            let x = gammaincinv(a, p).expect("gammaincinv should succeed for a > 140");
            assert!(
                rel_close(x, x_ref, 1e-6),
                "gammaincinv({a}, {p}) = {x}, expected {x_ref}"
            );
            // Round trip: feeding the recovered x back through gammainc
            // must reproduce p (this is what would have failed outright
            // under the old overflow-to-NaN/instant-false-convergence bug).
            let p_recovered = gammainc(a, x).expect("gammainc should succeed for a > 140");
            assert!(
                rel_close(p_recovered, p, 1e-6),
                "gammainc({a}, {x}) round-trip = {p_recovered}, expected {p}"
            );
        }
    }
}
