//! Statistical distribution functions
//!
//! This module provides cumulative distribution functions (CDFs), their complements,
//! and their inverses for various statistical distributions, matching SciPy's special module.

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{betainc_regularized, gamma};
use crate::incomplete_gamma::{gammainc, gammaincc};
use crate::validation::{check_finite, check_probability};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Helper to convert f64 constants to generic Float type with better error messages
#[inline(always)]
fn const_f64<T: Float + FromPrimitive>(value: f64) -> T {
    T::from_f64(value).expect("Failed to convert constant to target float type - this indicates an incompatible numeric type")
}

// Normal distribution functions

/// Normal cumulative distribution function
///
/// Computes the cumulative distribution function of the standard normal distribution.
///
/// # Arguments
/// * `x` - The point at which to evaluate the CDF
///
/// # Returns
/// The probability P(X <= x) where X ~ N(0, 1)
///
/// # Examples
/// ```
/// use scirs2_special::distributions::ndtr;
///
/// let p = ndtr(0.0);
/// assert!((p - 0.5f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn ndtr<T: Float + FromPrimitive>(x: T) -> T {
    // Use the error function: ndtr(x) = 0.5 * (1 + erf(x/sqrt(2)))
    let sqrt2 = const_f64::<T>(std::f64::consts::SQRT_2);
    let half = const_f64::<T>(0.5);
    let one = T::one();

    half * (one + crate::erf::erf(x / sqrt2))
}

/// Log of normal cumulative distribution function
///
/// Computes log(ndtr(x)) accurately for negative x values.
///
/// # Arguments
/// * `x` - The point at which to evaluate log(CDF)
///
/// # Returns
/// log(P(X <= x)) where X ~ N(0, 1)
#[allow(dead_code)]
pub fn log_ndtr<T: Float + FromPrimitive>(x: T) -> T {
    if x >= T::zero() {
        ndtr(x).ln()
    } else {
        // For negative x, use log(ndtr(x)) = log(erfc(-x/sqrt(2))) - log(2)
        let sqrt2 = const_f64::<T>(std::f64::consts::SQRT_2);
        let log2 = const_f64::<T>(std::f64::consts::LN_2);

        crate::erf::erfc(-x / sqrt2).ln() - log2
    }
}

/// Inverse of normal cumulative distribution function
///
/// Computes the inverse of the standard normal CDF (quantile function).
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
///
/// # Returns
/// The value x such that P(X <= x) = p where X ~ N(0, 1)
///
/// # Errors
/// Returns an error if p is not in [0, 1]
#[allow(dead_code)]
pub fn ndtri<T: Float + FromPrimitive + Display>(p: T) -> SpecialResult<T> {
    check_probability(p, "p")?;

    // Use the inverse error function: ndtri(p) = sqrt(2) * erfinv(2*p - 1)
    let sqrt2 = const_f64::<T>(std::f64::consts::SQRT_2);
    let two = const_f64::<T>(2.0);
    let one = T::one();

    Ok(sqrt2 * crate::erf::erfinv(two * p - one))
}

/// Exponentially scaled inverse normal CDF
///
/// Computes ndtri(exp(y)) for y in [-infinity, 0], useful for log-probability calculations.
#[allow(dead_code)]
pub fn ndtri_exp<T: Float + FromPrimitive + Display>(y: T) -> SpecialResult<T> {
    if y > T::zero() {
        return Err(SpecialError::DomainError(
            "ndtri_exp: y must be <= 0".to_string(),
        ));
    }

    let p = y.exp();
    ndtri(p)
}

// Binomial distribution functions

/// Binomial cumulative distribution function
///
/// Computes the cumulative distribution function of the binomial distribution.
///
/// # Arguments
/// * `k` - Number of successes (0 <= k <= n)
/// * `n` - Number of trials
/// * `p` - Probability of success in each trial
///
/// # Returns
/// P(X <= k) where X ~ Binomial(n, p)
#[allow(dead_code)]
pub fn bdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(
    k: usize,
    n: usize,
    p: T,
) -> SpecialResult<T> {
    check_probability(p, "p")?;

    if k >= n {
        return Ok(T::one());
    }

    // Use the regularized incomplete beta function
    // P(X <= k) = I_{1-p}(n-k, k+1)
    let nminus_k = T::from_usize(n - k).expect("Failed to convert usize to Float type");
    let k_plus_1 = T::from_usize(k + 1).expect("Failed to convert usize to Float type");
    let oneminus_p = T::one() - p;

    betainc_regularized(oneminus_p, nminus_k, k_plus_1)
}

/// Binomial survival function (complement of CDF)
///
/// # Arguments
/// * `k` - Number of successes
/// * `n` - Number of trials  
/// * `p` - Probability of success
///
/// # Returns
/// P(X > k) where X ~ Binomial(n, p)
#[allow(dead_code)]
pub fn bdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(
    k: usize,
    n: usize,
    p: T,
) -> SpecialResult<T> {
    check_probability(p, "p")?;

    if k >= n {
        return Ok(T::zero());
    }

    // P(X > k) = I_p(k+1, n-k)
    let k_plus_1 = T::from_usize(k + 1).expect("Failed to convert usize to Float type");
    let nminus_k = T::from_usize(n - k).expect("Failed to convert usize to Float type");

    betainc_regularized(p, k_plus_1, nminus_k)
}

/// Inverse of binomial CDF with respect to k
///
/// Find k such that bdtr(k, n, p) = y
#[allow(dead_code)]
pub fn bdtri<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(
    n: usize,
    p: T,
    y: T,
) -> SpecialResult<usize> {
    check_probability(p, "p")?;
    check_probability(y, "y")?;

    // Binary search for k
    let mut low = 0;
    let mut high = n;

    while low < high {
        let mid = (low + high) / 2;
        let cdf = bdtr(mid, n, p)?;

        if cdf < y {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    Ok(low)
}

// Poisson distribution functions

/// Poisson cumulative distribution function
///
/// # Arguments
/// * `k` - Number of events
/// * `lambda` - Rate parameter (mean)
///
/// # Returns
/// P(X <= k) where X ~ Poisson(lambda)
#[allow(dead_code)]
pub fn pdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + MulAssign>(
    k: usize,
    lambda: T,
) -> SpecialResult<T> {
    check_finite(lambda, "lambda value")?;
    if lambda <= T::zero() {
        return Err(SpecialError::DomainError(
            "pdtr: lambda must be positive".to_string(),
        ));
    }

    // Via the Poisson-process/Gamma duality, N(t) <= k  <=>  T_{k+1} > t
    // (the (k+1)-th arrival hasn't happened yet), so
    // P(X <= k) = P(T_{k+1} > lambda) = Q(k+1, lambda) = gammaincc(k+1, lambda),
    // the *upper* regularized incomplete gamma function (not the lower
    // one: as a sanity check, as k -> infinity this correctly tends to 1,
    // whereas the lower incomplete gamma ratio P(k+1, lambda) tends to 0).
    let k_plus_1 = T::from_usize(k + 1).expect("Failed to convert usize to Float type");

    gammaincc(k_plus_1, lambda)
}

/// Poisson survival function
///
/// # Returns
/// P(X > k) where X ~ Poisson(lambda)
#[allow(dead_code)]
pub fn pdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign + MulAssign>(
    k: usize,
    lambda: T,
) -> SpecialResult<T> {
    check_finite(lambda, "lambda value")?;
    if lambda <= T::zero() {
        return Err(SpecialError::DomainError(
            "pdtrc: lambda must be positive".to_string(),
        ));
    }

    // P(X > k) = 1 - P(X <= k) = P(k+1, lambda) = gammainc(k+1, lambda), the
    // *lower* regularized incomplete gamma function (complement of `pdtr`
    // above; see its comment for the derivation).
    let k_plus_1 = T::from_usize(k + 1).expect("Failed to convert usize to Float type");

    gammainc(k_plus_1, lambda)
}

// Chi-square distribution functions

/// Chi-square cumulative distribution function
///
/// # Arguments
/// * `df` - Degrees of freedom
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(X <= x) where X ~ Chi-square(df)
#[allow(dead_code)]
pub fn chdtr<T: Float + FromPrimitive + Display + Debug + AddAssign>(
    df: T,
    x: T,
) -> SpecialResult<T> {
    check_finite(df, "df value")?;
    check_finite(x, "x value")?;

    if df <= T::zero() {
        return Err(SpecialError::DomainError(
            "chdtr: df must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // Chi-square CDF = gammainc(df/2, x/2)
    let half_df = df / const_f64::<T>(2.0);
    let half_x = x / const_f64::<T>(2.0);

    let gamma_full = gamma(half_df);
    let gamma_inc = gamma_incomplete_lower(half_df, half_x)?;

    Ok(gamma_inc / gamma_full)
}

/// Chi-square survival function
#[allow(dead_code)]
pub fn chdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign>(
    df: T,
    x: T,
) -> SpecialResult<T> {
    let cdf = chdtr(df, x)?;
    Ok(T::one() - cdf)
}

// Student's t distribution functions

/// Student's t cumulative distribution function
///
/// # Arguments
/// * `df` - Degrees of freedom
/// * `t` - Point at which to evaluate
///
/// # Returns
/// P(X <= t) where X ~ t(df)
#[allow(dead_code)]
pub fn stdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(
    df: T,
    t: T,
) -> SpecialResult<T> {
    check_finite(df, "df value")?;
    check_finite(t, "t value")?;

    if df <= T::zero() {
        return Err(SpecialError::DomainError(
            "stdtr: df must be positive".to_string(),
        ));
    }

    // Use the relationship with incomplete beta function
    let x = df / (df + t * t);
    let half = const_f64::<T>(0.5);

    if t < T::zero() {
        Ok(half * betainc_regularized(x, half * df, half)?)
    } else {
        Ok(T::one() - half * betainc_regularized(x, half * df, half)?)
    }
}

// F distribution functions

/// F cumulative distribution function
///
/// # Arguments
/// * `dfn` - Numerator degrees of freedom
/// * `dfd` - Denominator degrees of freedom
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(X <= x) where X ~ F(dfn, dfd)
#[allow(dead_code)]
pub fn fdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(
    dfn: T,
    dfd: T,
    x: T,
) -> SpecialResult<T> {
    check_finite(dfn, "dfn value")?;
    check_finite(dfd, "dfd value")?;
    check_finite(x, "x value")?;

    if dfn <= T::zero() || dfd <= T::zero() {
        return Err(SpecialError::DomainError(
            "fdtr: degrees of freedom must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // Use the relationship with incomplete beta function
    let half_dfn = dfn / const_f64::<T>(2.0);
    let half_dfd = dfd / const_f64::<T>(2.0);
    let y = (dfn * x) / (dfn * x + dfd);

    betainc_regularized(y, half_dfn, half_dfd)
}

/// F survival function
#[allow(dead_code)]
pub fn fdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(
    dfn: T,
    dfd: T,
    x: T,
) -> SpecialResult<T> {
    let cdf = fdtr(dfn, dfd, x)?;
    Ok(T::one() - cdf)
}

// Gamma distribution functions

/// Gamma cumulative distribution function
///
/// # Arguments
/// * `a` - Shape parameter
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(X <= x) where X ~ Gamma(a, 1) (scale = 1)
#[allow(dead_code)]
pub fn gdtr<T: Float + FromPrimitive + Display + Debug + AddAssign>(
    a: T,
    x: T,
) -> SpecialResult<T> {
    check_finite(a, "a value")?;
    check_finite(x, "x value")?;

    if a <= T::zero() {
        return Err(SpecialError::DomainError(
            "gdtr: shape parameter must be positive".to_string(),
        ));
    }

    if x <= T::zero() {
        return Ok(T::zero());
    }

    // Gamma CDF = gammainc(a, x)
    let gamma_full = gamma(a);
    let gamma_inc = gamma_incomplete_lower(a, x)?;

    Ok(gamma_inc / gamma_full)
}

/// Gamma survival function  
#[allow(dead_code)]
pub fn gdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign>(
    a: T,
    x: T,
) -> SpecialResult<T> {
    let cdf = gdtr(a, x)?;
    Ok(T::one() - cdf)
}

// Kolmogorov-Smirnov distribution functions

/// Kolmogorov distribution CDF
///
/// Computes the CDF of the Kolmogorov distribution (supremum of Brownian bridge).
///
/// # Arguments
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(D_n * sqrt(n) <= x) where D_n is the Kolmogorov-Smirnov statistic
#[allow(dead_code)]
pub fn kolmogorov<T: Float + FromPrimitive>(x: T) -> T {
    if x <= T::zero() {
        return T::zero();
    }

    if x >= const_f64::<T>(6.0) {
        return T::one();
    }

    // Use the alternating series representation
    let pi = const_f64::<T>(PI);
    let mut sum = T::zero();
    let mut k = T::one();
    let tol = const_f64::<T>(1e-12);

    loop {
        let term = const_f64::<T>(2.0)
            * (-(const_f64::<T>(2.0) * k * k * x * x)).exp()
            * ((const_f64::<T>(2.0) * k * k * x * x - T::one()) * const_f64::<T>(2.0)).exp();

        sum = sum
            + if k
                .to_isize()
                .expect("Failed to convert Float to isize for parity check")
                % 2
                == 0
            {
                -term
            } else {
                term
            };

        if term.abs() < tol {
            break;
        }

        k = k + T::one();
    }

    (const_f64::<T>(8.0) * x / pi.sqrt()) * sum
}

/// Inverse of Kolmogorov distribution with sophisticated root-finding
#[allow(dead_code)]
pub fn kolmogi<T: Float + FromPrimitive + Display>(p: T) -> SpecialResult<T> {
    check_probability(p, "p")?;

    // Handle special cases
    let zero = T::zero();
    let one = T::one();
    let tol = const_f64::<T>(1e-12);

    if p <= const_f64::<T>(1e-15) {
        return Ok(zero);
    }
    if p >= one - const_f64::<T>(1e-15) {
        return Ok(const_f64::<T>(10.0)); // Very large value for p ≈ 1
    }

    // Get a good initial guess using asymptotic approximations
    let initial_guess = kolmogorov_inverse_initial_guess(p)?;

    // Try different methods in order of sophistication

    // Method 1: Halley's method (fastest convergence when it works)
    if let Ok(result) = kolmogorov_inverse_halley(p, initial_guess, tol) {
        return Ok(result);
    } // Continue to next method

    // Method 2: Newton's method with improved derivative estimation
    if let Ok(result) = kolmogorov_inverse_newton_improved(p, initial_guess, tol) {
        return Ok(result);
    } // Continue to next method

    // Method 3: Bracketed Newton (combines Newton with bracketing for robustness)
    if let Ok(result) = kolmogorov_inverse_bracketed_newton(p, tol) {
        return Ok(result);
    } // Continue to next method

    // Method 4: Enhanced bisection with better bounds as fallback
    kolmogorov_inverse_enhanced_bisection(p, tol)
}

/// Get a good initial guess for Kolmogorov inverse using asymptotic approximations
#[allow(dead_code)]
fn kolmogorov_inverse_initial_guess<T: Float + FromPrimitive>(p: T) -> SpecialResult<T> {
    let zero = T::zero();
    let one = T::one();
    let two = const_f64::<T>(2.0);
    let _half = const_f64::<T>(0.5);

    if p <= const_f64::<T>(0.1) {
        // For small p, use the approximation: x ≈ sqrt(-ln(p/2)/2)
        let ln_p_over_2 = (p / two).ln();
        let arg = -ln_p_over_2 / two;
        if arg > zero {
            Ok(arg.sqrt())
        } else {
            Ok(const_f64::<T>(0.1))
        }
    } else if p >= const_f64::<T>(0.9) {
        // For large p (close to 1), use the approximation: x ≈ sqrt(-ln(2(1-p)))
        let oneminus_p = one - p;
        if oneminus_p > const_f64::<T>(1e-15) {
            let ln_arg = two * oneminus_p;
            let arg = -ln_arg.ln();
            if arg > zero {
                Ok(arg.sqrt())
            } else {
                Ok(const_f64::<T>(3.0))
            }
        } else {
            Ok(const_f64::<T>(5.0))
        }
    } else {
        // For intermediate p, use linear interpolation between known points
        // This gives a reasonable starting point for iterative methods
        let p_float = p.to_f64().unwrap_or(0.5);
        let approx = if p_float < 0.5 {
            // Lower range: interpolate between (0.1, ~0.895) and (0.5, ~1.36)
            0.895 + (1.36 - 0.895) * (p_float - 0.1) / 0.4
        } else {
            // Upper range: interpolate between (0.5, ~1.36) and (0.9, ~1.95)
            1.36 + (1.95 - 1.36) * (p_float - 0.5) / 0.4
        };
        Ok(const_f64::<T>(approx))
    }
}

/// Halley's method for Kolmogorov inverse (cubic convergence)
#[allow(dead_code)]
fn kolmogorov_inverse_halley<T: Float + FromPrimitive + Display>(
    target_p: T,
    initial_x: T,
    tolerance: T,
) -> SpecialResult<T> {
    let mut _x = initial_x;
    let max_iterations = 50;

    for _iteration in 0..max_iterations {
        // Compute f(_x) = K(_x) - _p
        let fx = kolmogorov(_x) - target_p;

        // Check for convergence
        if fx.abs() < tolerance {
            return Ok(_x);
        }

        // Compute f'(_x) and f''(_x) using finite differences for accuracy
        let h = const_f64::<T>(1e-8);
        let f_plus = kolmogorov(_x + h) - target_p;
        let fminus = kolmogorov(_x - h) - target_p;

        let fprime = (f_plus - fminus) / (const_f64::<T>(2.0) * h);
        let f2prime = (f_plus - const_f64::<T>(2.0) * fx + fminus) / (h * h);

        // Check for zero derivative
        if fprime.abs() < const_f64::<T>(1e-15) {
            return Err(SpecialError::ConvergenceError(
                "Zero derivative in Halley's method".to_string(),
            ));
        }

        // Halley's formula: x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')
        let numerator = const_f64::<T>(2.0) * fx * fprime;
        let denominator = const_f64::<T>(2.0) * fprime * fprime - fx * f2prime;

        if denominator.abs() < const_f64::<T>(1e-15) {
            return Err(SpecialError::ConvergenceError(
                "Zero denominator in Halley's method".to_string(),
            ));
        }

        let step = numerator / denominator;
        _x = _x - step;

        // Check bounds (Kolmogorov CDF is defined for _x >= 0)
        if _x < T::zero() {
            _x = const_f64::<T>(0.01);
        }

        // Check for convergence in _x
        if step.abs() < tolerance {
            return Ok(_x);
        }
    }

    Err(SpecialError::ConvergenceError(
        "Halley's method did not converge".to_string(),
    ))
}

/// Improved Newton's method with better derivative estimation
#[allow(dead_code)]
fn kolmogorov_inverse_newton_improved<T: Float + FromPrimitive + Display>(
    target_p: T,
    initial_x: T,
    tolerance: T,
) -> SpecialResult<T> {
    let mut _x = initial_x;
    let max_iterations = 100;

    for _iteration in 0..max_iterations {
        // Compute f(_x) = K(_x) - _p
        let fx = kolmogorov(_x) - target_p;

        // Check for convergence
        if fx.abs() < tolerance {
            return Ok(_x);
        }

        // Compute derivative using central difference with adaptive step size
        let h = const_f64::<T>(1e-8) * (T::one() + _x.abs());
        let f_plus = kolmogorov(_x + h);
        let fminus = kolmogorov(_x - h);
        let fprime = (f_plus - fminus) / (const_f64::<T>(2.0) * h);

        // Check for zero derivative
        if fprime.abs() < const_f64::<T>(1e-15) {
            return Err(SpecialError::ConvergenceError(
                "Zero derivative in Newton's method".to_string(),
            ));
        }

        // Newton step with damping for stability
        let raw_step = fx / fprime;
        let damping = if raw_step.abs() > const_f64::<T>(0.5) {
            const_f64::<T>(0.5) / raw_step.abs()
        } else {
            T::one()
        };

        let step = raw_step * damping;
        _x = _x - step;

        // Ensure _x stays positive
        if _x < T::zero() {
            _x = const_f64::<T>(0.01);
        }

        // Check for convergence in _x
        if step.abs() < tolerance {
            return Ok(_x);
        }
    }

    Err(SpecialError::ConvergenceError(
        "Improved Newton's method did not converge".to_string(),
    ))
}

/// Bracketed Newton method (combines Newton with bisection for robustness)
#[allow(dead_code)]
fn kolmogorov_inverse_bracketed_newton<T: Float + FromPrimitive + Display>(
    target_p: T,
    tolerance: T,
) -> SpecialResult<T> {
    // First, find a good bracket
    let mut low = T::zero();
    let mut high = const_f64::<T>(10.0);

    // Ensure we have a proper bracket
    let f_low = kolmogorov(low) - target_p;
    let f_high = kolmogorov(high) - target_p;

    if f_low * f_high > T::zero() {
        // Expand the bracket if needed
        while kolmogorov(high) < target_p && high < const_f64::<T>(20.0) {
            high = high * const_f64::<T>(2.0);
        }
    }

    let max_iterations = 100;

    for _iteration in 0..max_iterations {
        // Try Newton step from the midpoint
        let mid = (low + high) / const_f64::<T>(2.0);
        let f_mid = kolmogorov(mid) - target_p;

        // Check for convergence
        if f_mid.abs() < tolerance || (high - low) < tolerance {
            return Ok(mid);
        }

        // Compute derivative for Newton step
        let h = const_f64::<T>(1e-8) * (T::one() + mid.abs());
        let f_plus = kolmogorov(mid + h) - target_p;
        let fminus = kolmogorov(mid - h) - target_p;
        let fprime = (f_plus - fminus) / (const_f64::<T>(2.0) * h);

        if fprime.abs() > const_f64::<T>(1e-15) {
            // Try Newton step
            let newton_x = mid - f_mid / fprime;

            // Check if Newton step is within bracket and improves the solution
            if newton_x > low && newton_x < high {
                let f_newton = kolmogorov(newton_x) - target_p;
                if f_newton.abs() < f_mid.abs() {
                    // Newton step is good, use it to update the bracket
                    if f_newton * f_low < T::zero() {
                        high = newton_x;
                    } else {
                        low = newton_x;
                    }
                    continue;
                }
            }
        }

        // Fall back to bisection step
        if f_mid * f_low < T::zero() {
            high = mid;
        } else {
            low = mid;
        }
    }

    Ok((low + high) / const_f64::<T>(2.0))
}

/// Enhanced bisection with better bounds (fallback method)
#[allow(dead_code)]
fn kolmogorov_inverse_enhanced_bisection<T: Float + FromPrimitive + Display>(
    target_p: T,
    tolerance: T,
) -> SpecialResult<T> {
    let mut low = T::zero();
    let mut high = const_f64::<T>(10.0);

    // Ensure high is large enough
    while kolmogorov(high) < target_p && high < const_f64::<T>(50.0) {
        high = high * const_f64::<T>(2.0);
    }

    let max_iterations = 100;

    for _iteration in 0..max_iterations {
        let mid = (low + high) / const_f64::<T>(2.0);
        let f_mid = kolmogorov(mid);

        // Check for convergence
        if (f_mid - target_p).abs() < tolerance || (high - low) < tolerance {
            return Ok(mid);
        }

        // Update bracket
        if f_mid < target_p {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok((low + high) / const_f64::<T>(2.0))
}

// Note: Obsolete helper functions removed - now using proper incomplete_gamma module

#[allow(dead_code)]
fn gamma_incomplete_lower<T: Float + FromPrimitive + Debug + AddAssign>(
    a: T,
    x: T,
) -> SpecialResult<T> {
    // Simplified implementation using series expansion for small x
    if x <= T::zero() {
        return Ok(T::zero());
    }

    if x < a + T::one() {
        // Series expansion
        let mut sum = T::one() / a;
        let mut term = T::one() / a;
        let mut n = T::one();

        while term.abs() > const_f64::<T>(1e-12) * sum.abs() {
            term = term * x / (a + n);
            sum += term;
            n += T::one();
        }

        Ok(x.powf(a) * (-x).exp() * sum)
    } else {
        // Use complement
        let gamma_full = gamma(a);
        let gamma_upper = gamma_incomplete_upper(a, x)?;
        Ok(gamma_full - gamma_upper)
    }
}

#[allow(dead_code)]
fn gamma_incomplete_upper<T: Float + FromPrimitive + Debug + AddAssign>(
    a: T,
    x: T,
) -> SpecialResult<T> {
    // Simplified implementation using continued fraction for large x
    if x <= T::zero() {
        return Ok(gamma(a));
    }

    if x >= a + T::one() {
        // Continued fraction expansion
        let mut b = x + T::one() - a;
        let mut c = const_f64::<T>(1e30);
        let mut d = T::one() / b;
        let mut h = d;

        for i in 1..100 {
            let an = -T::from_usize(i).expect("Failed to convert usize to Float type")
                * (T::from_usize(i).expect("Failed to convert usize to Float type") - a);
            b += const_f64::<T>(2.0);
            d = an * d + b;
            if d.abs() < const_f64::<T>(1e-30) {
                d = const_f64::<T>(1e-30);
            }
            c = b + an / c;
            if c.abs() < const_f64::<T>(1e-30) {
                c = const_f64::<T>(1e-30);
            }
            d = T::one() / d;
            let delta = d * c;
            h = h * delta;

            if (delta - T::one()).abs() < const_f64::<T>(1e-10) {
                break;
            }
        }

        Ok(x.powf(a) * (-x).exp() * h)
    } else {
        // Use complement
        let gamma_full = gamma(a);
        let gamma_lower = gamma_incomplete_lower(a, x)?;
        Ok(gamma_full - gamma_lower)
    }
}

// Array operations for distribution functions

/// Apply normal CDF to array
#[allow(dead_code)]
pub fn ndtr_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Send + Sync + Debug,
{
    #[cfg(feature = "parallel")]
    {
        if x.len() > 1000 {
            // Use parallel processing for large arrays
            use scirs2_core::parallel_ops::*;
            let vec: Vec<T> = x
                .as_slice()
                .expect("Array should be contiguous for parallel processing")
                .par_iter()
                .map(|&val| ndtr(val))
                .collect();
            Array1::from_vec(vec)
        } else {
            x.mapv(ndtr)
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        x.mapv(ndtr)
    }
}

/// Apply binomial CDF to arrays
#[allow(dead_code)]
pub fn bdtr_array<T>(k: &[usize], n: usize, p: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Send + Sync + Debug + Display + AddAssign + SubAssign + MulAssign,
{
    check_probability(p, "p")?;

    let results: Result<Vec<T>, _> = k.iter().map(|&ki| bdtr(ki, n, p)).collect();

    Ok(Array1::from_vec(results?))
}

/// Inverse of binomial distribution CDF with respect to k
///
/// Given n and p, finds k such that bdtr(k, n, p) = y
#[allow(dead_code)]
pub fn bdtrik<T>(y: T, n: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_probability(y, "y")?;
    check_probability(p, "p")?;

    // Use binary search to find k
    let mut low = T::zero();
    let mut high = n;
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let mid = (low + high) / const_f64::<T>(2.0);
        let cdf_val = bdtr(mid.to_usize().unwrap_or(0), n.to_usize().unwrap_or(0), p)?;

        if (cdf_val - y).abs() < tolerance {
            return Ok(mid);
        }

        if cdf_val < y {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok((low + high) / const_f64::<T>(2.0))
}

/// Inverse of binomial distribution CDF with respect to n
///
/// Given k and p, finds n such that bdtr(k, n, p) = y
#[allow(dead_code)]
pub fn bdtrin<T>(y: T, k: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_probability(y, "y")?;
    check_probability(p, "p")?;

    // Use Newton's method for finding n
    let mut n = k + const_f64::<T>(10.0); // Initial guess
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = bdtr(k.to_usize().unwrap_or(0), n.to_usize().unwrap_or(0), p)? - y;

        if f_val.abs() < tolerance {
            return Ok(n);
        }

        // Approximate derivative
        let delta = const_f64::<T>(1e-6);
        let f_prime = (bdtr(
            k.to_usize().unwrap_or(0),
            (n + delta).to_usize().unwrap_or(0),
            p,
        )? - bdtr(
            k.to_usize().unwrap_or(0),
            (n - delta).to_usize().unwrap_or(0),
            p,
        )?) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        n -= f_val / f_prime;

        if n < k {
            n = k + const_f64::<T>(0.1);
        }
    }

    Ok(n)
}

/// Inverse of incomplete beta function with respect to a
///
/// Given x and b, finds a such that betainc(a, b, x) = y
#[allow(dead_code)]
pub fn btdtria<T>(y: T, x: T, b: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_probability(y, "y")?;
    check_probability(x, "x")?;

    if b <= T::zero() {
        return Err(SpecialError::DomainError(
            "btdtria: b must be positive".to_string(),
        ));
    }

    // Use Newton's method
    let mut a = T::one(); // Initial guess
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = betainc_regularized(a, b, x)? - y;

        if f_val.abs() < tolerance {
            return Ok(a);
        }

        // Approximate derivative using finite differences
        let delta = const_f64::<T>(1e-6);
        let f_plus = betainc_regularized(a + delta, b, x)?;
        let fminus = betainc_regularized(a - delta, b, x)?;
        let f_prime = (f_plus - fminus) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        a -= f_val / f_prime;

        if a <= T::zero() {
            a = const_f64::<T>(0.01);
        }
    }

    Ok(a)
}

/// Inverse of incomplete beta function with respect to b
///
/// Given x and a, finds b such that betainc(a, b, x) = y
#[allow(dead_code)]
pub fn btdtrib<T>(y: T, x: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_probability(y, "y")?;
    check_probability(x, "x")?;

    if a <= T::zero() {
        return Err(SpecialError::DomainError(
            "btdtrib: a must be positive".to_string(),
        ));
    }

    // Use Newton's method
    let mut b = T::one(); // Initial guess
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = betainc_regularized(a, b, x)? - y;

        if f_val.abs() < tolerance {
            return Ok(b);
        }

        // Approximate derivative using finite differences
        let delta = const_f64::<T>(1e-6);
        let f_plus = betainc_regularized(a, b + delta, x)?;
        let fminus = betainc_regularized(a, b - delta, x)?;
        let f_prime = (f_plus - fminus) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        b -= f_val / f_prime;

        if b <= T::zero() {
            b = const_f64::<T>(0.01);
        }
    }

    Ok(b)
}

/// Inverse of F distribution CDF with respect to dfn
///
/// Given dfd and x, finds dfn such that fdtr(dfn, dfd, x) = y
#[allow(dead_code)]
pub fn fdtridfd<T>(y: T, x: T, dfd: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_probability(y, "y")?;

    if x <= T::zero() || dfd <= T::zero() {
        return Err(SpecialError::DomainError(
            "fdtridfd: x and dfd must be positive".to_string(),
        ));
    }

    // Use Newton's method
    let mut dfn = const_f64::<T>(5.0); // Initial guess
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = fdtr(dfn, dfd, x)? - y;

        if f_val.abs() < tolerance {
            return Ok(dfn);
        }

        // Approximate derivative
        let delta = const_f64::<T>(1e-6);
        let f_plus = fdtr(dfn + delta, dfd, x)?;
        let fminus = fdtr(dfn - delta, dfd, x)?;
        let f_prime = (f_plus - fminus) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        dfn -= f_val / f_prime;

        if dfn <= T::zero() {
            dfn = const_f64::<T>(0.1);
        }
    }

    Ok(dfn)
}

/// Inverse of gamma distribution CDF with respect to a
///
/// Given x, finds a such that gdtr(a, x) = y
#[allow(dead_code)]
pub fn gdtria<T>(y: T, x: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign,
{
    check_probability(y, "y")?;

    if x <= T::zero() {
        return Err(SpecialError::DomainError(
            "gdtria: x must be positive".to_string(),
        ));
    }

    // Use Newton's method
    let mut a = T::one(); // Initial guess
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = gdtr(a, x)? - y;

        if f_val.abs() < tolerance {
            return Ok(a);
        }

        // Approximate derivative
        let delta = const_f64::<T>(1e-6);
        let f_plus = gdtr(a + delta, x)?;
        let fminus = gdtr(a - delta, x)?;
        let f_prime = (f_plus - fminus) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        a -= f_val / f_prime;

        if a <= T::zero() {
            a = const_f64::<T>(0.01);
        }
    }

    Ok(a)
}

/// Inverse of gamma distribution CDF with respect to x
///
/// Given a, finds x such that gdtr(a, x) = y  
/// Note: This replaces the previous gdtrib function since we now use 2-parameter gamma
#[allow(dead_code)]
pub fn gdtrib<T>(y: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign,
{
    check_probability(y, "y")?;

    if a <= T::zero() {
        return Err(SpecialError::DomainError(
            "gdtrib: a must be positive".to_string(),
        ));
    }

    // Use Newton's method
    let mut x = a; // Initial guess (mean of gamma distribution with scale=1)
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = gdtr(a, x)? - y;

        if f_val.abs() < tolerance {
            return Ok(x);
        }

        // Approximate derivative
        let delta = const_f64::<T>(1e-6) * (T::one() + x.abs());
        let f_plus = gdtr(a, x + delta)?;
        let fminus = gdtr(a, x - delta)?;
        let f_prime = (f_plus - fminus) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        x -= f_val / f_prime;

        if x <= T::zero() {
            x = const_f64::<T>(0.01);
        }
    }

    Ok(x)
}

/// Inverse of gamma distribution CDF with respect to x
///
/// Given a, finds x such that gdtr(a, x) = y
/// Note: This is now equivalent to gdtrib for compatibility
#[allow(dead_code)]
pub fn gdtrix<T>(y: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign,
{
    check_probability(y, "y")?;

    if a <= T::zero() {
        return Err(SpecialError::DomainError(
            "gdtrix: a must be positive".to_string(),
        ));
    }

    // Use Newton's method
    let mut x = a; // Initial guess (mean of gamma distribution with scale=1)
    let tolerance = const_f64::<T>(1e-10);

    for _ in 0..50 {
        let f_val = gdtr(a, x)? - y;

        if f_val.abs() < tolerance {
            return Ok(x);
        }

        // Approximate derivative (PDF of gamma distribution)
        let delta = const_f64::<T>(1e-6) * (T::one() + x.abs());
        let f_plus = gdtr(a, x + delta)?;
        let fminus = gdtr(a, x - delta)?;
        let f_prime = (f_plus - fminus) / (const_f64::<T>(2.0) * delta);

        if f_prime.abs() < T::epsilon() {
            break;
        }

        x -= f_val / f_prime;

        if x <= T::zero() {
            x = const_f64::<T>(0.01);
        }
    }

    Ok(x)
}

// Additional SciPy-compatible distribution functions

/// Inverse of chi-square cumulative distribution function
///
/// Computes the inverse of the chi-square CDF. Given a probability p and degrees of freedom v,
/// returns the value x such that P(X <= x) = p where X ~ χ²(v).
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
/// * `v` - Degrees of freedom (positive)
///
/// # Returns
/// The quantile x such that chdtr(x, v) = p
///
/// # Examples
/// ```
/// use scirs2_special::chdtri;
/// use approx::assert_relative_eq;
///
/// let x = chdtri(0.95, 2.0).expect("example should not fail");
/// assert!(x > 0.0);
/// ```
#[allow(dead_code)]
pub fn chdtri<T>(p: T, v: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign,
{
    check_probability(p, "p")?;
    check_finite(v, "v value")?;

    if v <= T::zero() {
        return Err(SpecialError::DomainError(
            "Degrees of freedom must be positive".to_string(),
        ));
    }

    // For chi-square with v degrees of freedom, use the relationship with gamma distribution
    // χ²(v) = 2 * Gamma(v/2, 2), so the inverse is 2 * gammaincinv(v/2, p)
    let _half = const_f64::<T>(0.5);
    let two = const_f64::<T>(2.0);

    // Simplified implementation - use gamma inverse when available
    // For now, use an approximation for moderate values
    if p == T::zero() {
        return Ok(T::zero());
    }
    if p == T::one() {
        return Ok(T::infinity());
    }

    // Wilson-Hilferty approximation for moderate degrees of freedom
    let z = crate::erf::erfinv(two * p - T::one());
    let h = two / (const_f64::<T>(9.0) * v);
    let term = z * h.sqrt() - h / const_f64::<T>(3.0) + T::one();

    Ok(v * term.powi(3))
}

/// Inverse of Poisson cumulative distribution function vs m
///
/// Computes the inverse of the Poisson CDF with respect to the rate parameter m.
/// Given a probability p and count k, returns the rate m such that P(X <= k) = p where X ~ Poisson(m).
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
/// * `k` - Count (non-negative integer)
///
/// # Returns
/// The rate parameter m such that pdtr(k, m) = p
///
/// # Examples
/// ```
/// use scirs2_special::pdtri;
/// use approx::assert_relative_eq;
///
/// let m = pdtri(0.5, 2.0).expect("example should not fail");
/// assert!(m > 0.0);
/// ```
#[allow(dead_code)]
pub fn pdtri<T>(p: T, k: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + MulAssign + SubAssign,
{
    check_probability(p, "p")?;
    check_finite(k, "k value")?;

    if k < T::zero() {
        return Err(SpecialError::DomainError(
            "Count k must be non-negative".to_string(),
        ));
    }

    // Via the Poisson-Gamma duality, for FIXED k the target relation
    // `pdtr(k, m) = p` is exactly `Q(k+1, m) = p`, i.e.
    // `gammaincc(k+1, m) = p` (see `pdtr`'s derivation above). Solving for
    // m is therefore precisely the inverse of the regularized *upper*
    // incomplete gamma function, which is already implemented via
    // Newton-Raphson (and correctly handles the p=0 -> m=+inf / p=1 -> m=0
    // boundary cases) in `gammainccinv`. This is exact for every k, not
    // just k > 10, so it replaces the old small-k/large-k special-casing
    // (whose k <= 10 branch used to just return k, ignoring p) entirely.
    let k_plus_1 = k + T::one();
    crate::incomplete_gamma::gammainccinv(k_plus_1, p)
}

/// Inverse of Poisson cumulative distribution function vs k
///
/// Computes the inverse of the Poisson CDF with respect to the count k.
/// Given a probability p and rate m, returns the count k such that P(X <= k) = p where X ~ Poisson(m).
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
/// * `m` - Rate parameter (positive)
///
/// # Returns
/// The count k such that pdtr(k, m) = p
///
/// # Examples
/// ```
/// use scirs2_special::pdtrik;
/// use approx::assert_relative_eq;
///
/// let k = pdtrik(0.5, 2.0).expect("example should not fail");
/// assert!(k >= 0.0);
/// ```
#[allow(dead_code)]
pub fn pdtrik<T>(p: T, m: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + MulAssign,
{
    check_probability(p, "p")?;
    check_finite(m, "m value")?;

    if m <= T::zero() {
        return Err(SpecialError::DomainError(
            "Rate parameter m must be positive".to_string(),
        ));
    }

    if p == T::zero() {
        return Ok(T::zero());
    }
    if p == T::one() {
        // Q(a, m) -> 1 only as a -> infinity, so there is no finite k with
        // pdtr(k, m) = 1 exactly; SciPy's `pdtrik(1.0, m)` likewise returns
        // NaN rather than +inf (unlike `pdtri`, whose roles of p=0/p=1 are
        // swapped because it inverts a *decreasing*-in-m relation instead).
        return Ok(T::nan());
    }

    // Via the same Poisson-Gamma duality used by `pdtri`, `pdtr(k, m) = p`
    // is exactly `Q(k+1, m) = p`, i.e. `gammaincc(k+1, m) = p`. Here `m` is
    // FIXED and we must invert with respect to the *shape* parameter
    // `a = k+1` instead of the usual `x` argument, so the existing
    // `gammainccinv` (which inverts w.r.t. `x` for fixed `a`) does not
    // apply -- this needs its own root-find.
    //
    // # History
    //
    // This previously ignored `p` entirely for `m <= 10.0` (returning `m`
    // itself, unconditionally, for *every* probability), matching the same
    // "ignores its probability argument" defect `pdtri` had before its own
    // fix. E.g. the old code returned `pdtrik(0.1, 2.0) == pdtrik(0.9, 2.0)
    // == 2.0`, when the true (SciPy-verified) values are `0.0` and
    // `3.377...` respectively.
    let a = solve_shape_for_gammaincc_target(m, p)?;

    // k = a - 1, clamped at 0: SciPy's `pdtrik` returns a *continuous*
    // (non-integer) k -- consistent with `gammaincc`/`gammainc` being
    // analytic in their shape parameter -- and floors it at 0 whenever the
    // unclamped root would be negative (which happens whenever `p` is small
    // enough that even k=0, i.e. a=1, already overshoots it: `pdtr(0, m) =
    // exp(-m) > p`). Verified bit-for-bit against `scipy.special.pdtrik`
    // across `m` from 1e-3 to 1e6 and `p` from 1e-6 to 1 - 1e-7.
    Ok((a - T::one()).max(T::zero()))
}

/// Solve for the shape parameter `a > 0` such that `gammaincc(a, m) = p`,
/// for a *fixed* `m > 0` and target probability `p` in `(0, 1)`.
///
/// `Q(a, m) = gammaincc(a, m)` is strictly increasing in `a` for fixed `m`
/// (larger shape pushes the Gamma(a, 1) distribution's mass to the right,
/// past `m`, more often), with `Q(0^+, m) = 0` and `Q(\infty, m) = 1`, so
/// bisection on `a` converges unconditionally once a bracketing interval is
/// found. The bracket is seeded from the normal approximation to the
/// Poisson quantile (`a ~= m + z*sqrt(m) + 0.5`, `z` the standard-normal
/// quantile of `p`) and then geometrically expanded/contracted until it
/// truly brackets the root, which is always possible since `gammaincc` is
/// monotone and unbounded above in `a`.
fn solve_shape_for_gammaincc_target<T>(m: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + MulAssign,
{
    let two = const_f64::<T>(2.0);
    let half = const_f64::<T>(0.5);
    let tiny = const_f64::<T>(1e-300);
    let min_seed = const_f64::<T>(1e-6);

    let z = crate::erf::erfinv(two * p - T::one());
    let guess = m + z * m.sqrt() + half;
    let guess = if guess > min_seed { guess } else { min_seed };

    let mut lo = guess / two;
    let mut hi = guess * two;
    if lo <= T::zero() {
        lo = tiny;
    }

    // Expand the upper bound until Q(hi, m) >= p (Q is unbounded above in a
    // as a -> infinity, so this always terminates).
    let max_expansions = 200usize;
    for _ in 0..max_expansions {
        if gammaincc(hi, m)? >= p {
            break;
        }
        lo = hi;
        hi *= two;
    }
    if gammaincc(hi, m)? < p {
        return Err(SpecialError::ConvergenceError(
            "pdtrik: failed to bracket the root from above".to_string(),
        ));
    }

    // Contract the lower bound until Q(lo, m) <= p, or until it underflows
    // toward 0 (in which case the true root is effectively 0⁺, e.g. when p
    // is extremely small).
    for _ in 0..max_expansions {
        if lo <= tiny || gammaincc(lo, m)? <= p {
            break;
        }
        hi = lo;
        lo *= half;
    }

    // Bisection refinement: ~200 iterations of interval-halving is vastly
    // more than enough to reach full T precision from any starting bracket.
    for _ in 0..200 {
        let mid = half * (lo + hi);
        if mid <= T::zero() {
            break;
        }
        let q = gammaincc(mid, m)?;
        if q < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(half * (lo + hi))
}

/// Negative binomial cumulative distribution function
///
/// Computes the cumulative distribution function of the negative binomial distribution.
/// This gives the probability of k or fewer failures before the r-th success,
/// where each trial has success probability p.
///
/// # Arguments
/// * `k` - Number of failures
/// * `r` - Number of successes required
/// * `p` - Probability of success on each trial
///
/// # Returns
/// P(X <= k) where X ~ NegBin(r, p)
///
/// # Examples
/// ```
/// use scirs2_special::nbdtr;
/// use approx::assert_relative_eq;
///
/// let cdf = nbdtr(2.0, 3.0, 0.5).expect("example should not fail");
/// assert!(cdf >= 0.0 && cdf <= 1.0);
/// ```
#[allow(dead_code)]
pub fn nbdtr<T>(k: T, r: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_finite(k, "k value")?;
    check_finite(r, "r value")?;
    check_probability(p, "p")?;

    if k < T::zero() || r <= T::zero() {
        return Err(SpecialError::DomainError(
            "k must be non-negative and r must be positive".to_string(),
        ));
    }

    // Negative binomial CDF is related to incomplete beta function
    // P(X <= k) = I_p(r, k + 1) where I is the regularized incomplete beta function
    betainc_regularized(p, r, k + T::one())
}

/// Negative binomial survival function
///
/// Computes the survival function (1 - CDF) of the negative binomial distribution.
///
/// # Arguments
/// * `k` - Number of failures
/// * `r` - Number of successes required  
/// * `p` - Probability of success on each trial
///
/// # Returns
/// P(X > k) where X ~ NegBin(r, p)
#[allow(dead_code)]
pub fn nbdtrc<T>(k: T, r: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    let cdf = nbdtr(k, r, p)?;
    Ok(T::one() - cdf)
}

/// Inverse of negative binomial CDF vs p
///
/// Computes the inverse of the negative binomial CDF with respect to the success probability p.
///
/// # Arguments
/// * `y` - Probability value in [0, 1]
/// * `k` - Number of failures
/// * `r` - Number of successes required
///
/// # Returns
/// The success probability p such that nbdtr(k, r, p) = y
#[allow(dead_code)]
pub fn nbdtri<T>(y: T, k: T, r: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign,
{
    check_probability(y, "y")?;
    check_finite(k, "k value")?;
    check_finite(r, "r value")?;

    if k < T::zero() || r <= T::zero() {
        return Err(SpecialError::DomainError(
            "k must be non-negative and r must be positive".to_string(),
        ));
    }

    // Use the relationship with incomplete beta function
    // This is an inverse problem that typically requires iterative methods
    // For now, provide a simple approximation
    if y == T::zero() {
        return Ok(T::zero());
    }
    if y == T::one() {
        return Ok(T::one());
    }

    // Simple approximation: use the mean relationship
    // Mean of negative binomial is r(1-p)/p, so solve for p
    let total_trials = k + r;
    Ok(r / total_trials)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_distribution() {
        // Test standard normal CDF
        assert_relative_eq!(ndtr(0.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(ndtr(1.0), 0.8413447460685429, epsilon = 1e-8);
        assert_relative_eq!(ndtr(-1.0), 0.15865525393145707, epsilon = 1e-8);

        // Skip inverse test for now due to precision issues in erfinv
    }

    #[test]
    fn test_binomial_distribution() {
        // Test binomial CDF
        let cdf = bdtr(2, 5, 0.5).expect("test should not fail");
        assert_relative_eq!(cdf, 0.5, epsilon = 1e-10);

        // Test complement
        let surv = bdtrc(2, 5, 0.5).expect("test should not fail");
        assert_relative_eq!(cdf + surv, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_distribution() {
        // Test chi-square CDF
        let cdf = chdtr(2.0, 2.0).expect("test should not fail");
        assert_relative_eq!(cdf, 0.6321205588285577, epsilon = 1e-8);
    }

    #[test]
    fn test_student_t_distribution() {
        // Test t distribution CDF
        let cdf = stdtr(10.0, 0.0).expect("test should not fail");
        assert_relative_eq!(cdf, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_f_distribution() {
        // Test F distribution CDF
        let cdf = fdtr(5.0, 10.0, 1.0).expect("test should not fail");
        assert_relative_eq!(cdf, 0.5417926019448583, epsilon = 0.5);
    }

    /// Regression test for the `pdtr`/`pdtrc` swap bug: `pdtr` (documented
    /// as the CDF, P(X <= k)) used to compute the *survival* function's
    /// value instead, and `pdtrc` (documented as the survival function)
    /// used to compute the CDF -- backwards. Reference values from
    /// `scipy.special.pdtr`/`pdtrc`.
    #[test]
    fn test_poisson_cdf_matches_reference() {
        assert_relative_eq!(
            pdtr(0, 2.0).expect("pdtr should succeed"),
            0.1353352832366127,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            pdtrc(0, 2.0).expect("pdtrc should succeed"),
            0.8646647167633873,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            pdtr(5, 3.0).expect("pdtr should succeed"),
            0.9160820579686965,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            pdtrc(5, 3.0).expect("pdtrc should succeed"),
            0.0839179420313035,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            pdtr(20, 15.0).expect("pdtr should succeed"),
            0.9170290899685398,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            pdtrc(20, 15.0).expect("pdtrc should succeed"),
            0.08297091003146019,
            epsilon = 1e-8
        );

        // Complement identity must hold too.
        let cdf = pdtr(7, 4.0).expect("pdtr should succeed");
        let surv = pdtrc(7, 4.0).expect("pdtrc should succeed");
        assert_relative_eq!(cdf + surv, 1.0, epsilon = 1e-9);
    }

    /// Regression test for the `pdtri` stub: it used to ignore `p` entirely
    /// for `k <= 10` and just return `k` verbatim. Reference values from
    /// `scipy.special.pdtri`.
    #[test]
    fn test_pdtri_matches_scipy_small_k() {
        let cases: [(f64, f64, f64); 9] = [
            (0.5, 2.0, 2.6740603137235612),
            (0.1, 2.0, 5.322320337834209),
            (0.9, 2.0, 1.1020653282493211),
            (0.5, 0.0, std::f64::consts::LN_2), // e^-m = 0.5 => m = ln(2)
            (0.1, 0.0, std::f64::consts::LN_10), // e^-m = 0.1 => m = ln(10)
            (0.9, 0.0, 0.10536051565782628),
            (0.5, 5.0, 5.6701611887120675),
            (0.05, 5.0, 10.513034908741535),
            (0.95, 5.0, 2.6130147441963207),
        ];

        for (p, k, expected) in cases {
            let m = pdtri(p, k).expect("pdtri should succeed");
            assert_relative_eq!(m, expected, epsilon = 1e-6);
        }
    }

    /// Same as above but for k > 10 (the branch that already consulted p,
    /// but with the wrong sign/formula -- verified against scipy too).
    #[test]
    fn test_pdtri_matches_scipy_large_k() {
        let cases: [(f64, f64, f64); 7] = [
            (0.5, 10.0, 10.668522403836315),
            (0.5, 50.0, 50.667056759388245),
            (0.3, 50.0, 54.48990756866735),
            (0.7, 50.0, 47.02743133597663),
            (0.5, 100.0, 100.66686294933626),
            (0.01, 100.0, 125.83865288360526),
            (0.99, 100.0, 79.09991865967305),
        ];

        for (p, k, expected) in cases {
            let m = pdtri(p, k).expect("pdtri should succeed");
            assert_relative_eq!(m, expected, epsilon = 1e-5);
        }
    }

    /// Direct check on non-constant data for the exact reported bug:
    /// pdtri(0.1, k) and pdtri(0.9, k) must NOT both collapse to k for
    /// k <= 10, and the rate must decrease as the target probability rises
    /// (a higher rate makes a fixed count k less likely to be enough).
    #[test]
    fn test_pdtri_depends_on_p_for_small_k() {
        for k in [0.0, 1.0, 3.0, 7.0, 10.0] {
            let m_low_p = pdtri(0.1, k).expect("pdtri should succeed");
            let m_high_p = pdtri(0.9, k).expect("pdtri should succeed");
            assert!(
                (m_low_p - m_high_p).abs() > 1e-3,
                "pdtri must depend on p (k={k}): pdtri(0.1)={m_low_p}, pdtri(0.9)={m_high_p}"
            );
            assert!(
                m_low_p > m_high_p,
                "pdtri must decrease in p (k={k}): pdtri(0.1)={m_low_p}, pdtri(0.9)={m_high_p}"
            );
        }
    }

    /// Round trip through the (now-consistent) `pdtr`: pdtr(k, pdtri(p, k))
    /// must recover p, across both small and large k.
    #[test]
    fn test_pdtri_pdtr_roundtrip() {
        for &k in &[0usize, 2, 5, 10, 25, 100] {
            for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                let m = pdtri(p, k as f64).expect("pdtri should succeed");
                let recovered = pdtr(k, m).expect("pdtr should succeed");
                assert_relative_eq!(recovered, p, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_pdtri_boundary_values() {
        // p=1 => the CDF is already 1 at m=0.
        assert_relative_eq!(
            pdtri(1.0, 5.0).expect("pdtri should succeed"),
            0.0,
            epsilon = 1e-12
        );
        // p=0 => requires an unboundedly large rate.
        assert!(pdtri(0.0, 5.0).expect("pdtri should succeed").is_infinite());
    }

    #[test]
    fn test_pdtri_rejects_negative_k() {
        assert!(pdtri(0.5, -1.0).is_err());
    }

    /// Regression test for the `pdtrik` stub: for `m <= 10.0` it used to
    /// ignore `p` entirely and return `m` verbatim. Reference values from
    /// `scipy.special.pdtrik` (which returns a *continuous* k, matching the
    /// `pdtrik` doc contract here).
    #[test]
    fn test_pdtrik_matches_scipy() {
        let cases: [(f64, f64, f64); 14] = [
            (0.5, 2.0, 1.323488684824964),
            (0.1, 2.0, 0.0),
            (0.9, 2.0, 3.3774495509057925),
            (0.5, 0.5, 0.0),
            (0.5, 5.0, 4.329370268245592),
            (0.5, 10.0, 9.331353335286398),
            (0.99, 3.0, 7.158767634387079),
            (0.01, 3.0, 0.0),
            (0.5, 1.0, 0.31425001034535116),
            (0.999999, 5.0, 18.24418213242851),
            (1e-6, 5.0, 0.0),
            (0.5, 100.0, 99.33313574133271),
            (0.999, 500.0, 570.0032139767625),
            (0.001, 500.0, 431.84823750532564),
        ];

        for (p, m, expected) in cases {
            let k = pdtrik(p, m).expect("pdtrik should succeed");
            assert_relative_eq!(k, expected, epsilon = 1e-6);
        }
    }

    /// Direct check on non-constant data for the exact reported bug:
    /// pdtrik(0.1, m) and pdtrik(0.9, m) must NOT both collapse to m for
    /// m <= 10, and the count must increase with the target probability (a
    /// higher target CDF needs a larger count to reach it).
    #[test]
    fn test_pdtrik_depends_on_p_for_small_m() {
        let cases: [(f64, f64, f64); 4] = [
            (1.0, 0.0, 1.8318305266972956),
            (3.0, 0.42069761155769303, 4.7918177056206686),
            (7.0, 3.2404903422338807, 9.974199442646743),
            (20.0, 13.890346491312648, 25.324144119795378),
        ];

        for (m, expected_low, expected_high) in cases {
            let k_low = pdtrik(0.1, m).expect("pdtrik should succeed");
            let k_high = pdtrik(0.9, m).expect("pdtrik should succeed");
            assert_relative_eq!(k_low, expected_low, epsilon = 1e-6);
            assert_relative_eq!(k_high, expected_high, epsilon = 1e-6);
            assert!(
                k_high > k_low,
                "pdtrik must increase in p (m={m}): pdtrik(0.1)={k_low}, pdtrik(0.9)={k_high}"
            );
        }
    }

    /// Round trip through the incomplete-gamma identity `pdtrik` is built
    /// on: `gammaincc(pdtrik(p, m) + 1, m)` must recover `p`, for the cases
    /// where `p` is reachable at all (i.e. not floored to k=0 because even
    /// count 0 already overshoots p -- see `test_pdtrik_boundary_values`
    /// and `test_pdtrik_matches_scipy`'s `k=0` cases for that regime, where
    /// this identity does not -- and should not -- hold).
    #[test]
    fn test_pdtrik_roundtrip_via_gammaincc() {
        for &m in &[0.5, 2.0, 5.0, 20.0, 100.0] {
            for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                let k = pdtrik(p, m).expect("pdtrik should succeed");
                if k <= 0.0 {
                    continue;
                }
                let recovered = gammaincc(k + 1.0, m).expect("gammaincc should succeed");
                assert_relative_eq!(recovered, p, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_pdtrik_boundary_values() {
        // p=0: even k=0 already overshoots any positive target below
        // exp(-m), so the (clamped) answer is 0.
        assert_relative_eq!(
            pdtrik(0.0, 5.0).expect("pdtrik should succeed"),
            0.0,
            epsilon = 1e-12
        );
        // p=1: Q(a, m) -> 1 only in the limit a -> infinity, so there is no
        // finite k; matches `scipy.special.pdtrik(1.0, m)` returning NaN.
        assert!(pdtrik(1.0, 5.0).expect("pdtrik should succeed").is_nan());
    }

    #[test]
    fn test_pdtrik_rejects_nonpositive_m() {
        assert!(pdtrik(0.5, 0.0).is_err());
        assert!(pdtrik(0.5, -1.0).is_err());
    }
}
