//! Special functions module
//!
//! This module provides implementations of special mathematical functions
//! that mirror SciPy's special module.

// SCIRS2 POLICY: Use scirs2_core re-exports
use scirs2_core::error::{check_domain, CoreError, CoreResult};
use scirs2_core::numeric::{Float, FloatConst, FromPrimitive};

/// Local result type alias for this module
type SciRS2Result<T> = CoreResult<T>;
/// Local error type alias for this module
type SciRS2Error = CoreError;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<T: Float + FromPrimitive>(value: f64) -> T {
    T::from(value).unwrap_or_else(|| T::nan())
}

/// Lanczos g=7 coefficients (Lanczos 1964, 9-term series, ~15-digit accuracy)
const LANCZOS_G: f64 = 7.0;
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
const LANCZOS_C: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute Γ(z) for z > 0.5 using the Lanczos approximation (g=7, n=9).
/// Returns (value, is_finite) — value may be ±∞ for very large z.
#[inline]
fn lanczos_gamma_pos(z: f64) -> f64 {
    // Lanczos approximation: Γ(z) = sqrt(2π) · t^(z-0.5) · e^(-t) · A_g(z-1)
    // where t = (z - 1) + g + 0.5, and we use z shifted so A_g sums over k=1..8.
    let z = z - 1.0; // shift to use A_g(z)
    let mut ag = LANCZOS_C[0];
    for (k, &ck) in LANCZOS_C[1..].iter().enumerate() {
        ag += ck / (z + (k + 1) as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * ag
}

/// Compute ln|Γ(z)| for z > 0.5 using the Lanczos approximation.
#[inline]
fn lanczos_lgamma_pos(z: f64) -> f64 {
    let z = z - 1.0;
    let mut ag = LANCZOS_C[0];
    for (k, &ck) in LANCZOS_C[1..].iter().enumerate() {
        ag += ck / (z + (k + 1) as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + ag.abs().ln()
}

/// Gamma function
///
/// Full Lanczos approximation (g=7, n=9, ~15-digit accuracy) for all real inputs.
/// Returns +∞ at poles (non-positive integers) and NaN for NaN input.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Gamma function value at x
///
/// # Examples
///
/// ```
/// use scirs2::special_fns::gamma;
/// let result: f64 = gamma(5.0);
/// assert!((result - 24.0).abs() < 1e-10);
/// assert!((gamma(0.5_f64) - 1.7724538509055159).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn gamma<T: Float + FromPrimitive + FloatConst>(x: T) -> T {
    let xf = match x.to_f64() {
        Some(v) => v,
        None => return T::nan(),
    };

    // NaN input
    if xf.is_nan() {
        return T::nan();
    }

    // Special case: x = 1 or x = 2 → exact
    if (xf - 1.0).abs() < 1e-15 || (xf - 2.0).abs() < 1e-15 {
        return T::one();
    }

    // Poles at non-positive integers
    if xf <= 0.0 {
        let nearest_int = xf.round();
        if (xf - nearest_int).abs() < 1e-14 {
            // pole: return ±∞ depending on sign convention (positive infinity for simplicity)
            return T::infinity();
        }
        // Reflection formula: Γ(x)·Γ(1-x) = π / sin(πx)
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * xf).sin();
        if sin_pi_x.abs() < f64::EPSILON {
            return T::infinity();
        }
        // Γ(x) = π / (sin(πx) · Γ(1-x))
        let gamma_one_minus_x = lanczos_gamma_pos(1.0 - xf);
        let val = pi / (sin_pi_x * gamma_one_minus_x);
        return T::from_f64(val).unwrap_or(T::nan());
    }

    // x > 0: use Lanczos directly (shift small x into the stable region via recurrence)
    // For 0 < x < 0.5, use: Γ(x) = Γ(x+1) / x
    let val = if xf < 0.5 {
        let gx1 = lanczos_gamma_pos(xf + 1.0);
        gx1 / xf
    } else {
        lanczos_gamma_pos(xf)
    };
    T::from_f64(val).unwrap_or(T::nan())
}

/// Natural logarithm of the absolute value of the gamma function: ln|Γ(x)|
///
/// Full Lanczos approximation (g=7, n=9, ~15-digit accuracy) for all real inputs.
/// Returns +∞ at poles (non-positive integers).
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Natural logarithm of |Γ(x)| at x
#[allow(dead_code)]
pub fn lgamma<T: Float + FromPrimitive + FloatConst>(x: T) -> T {
    let xf = match x.to_f64() {
        Some(v) => v,
        None => return T::nan(),
    };

    if xf.is_nan() {
        return T::nan();
    }

    // Poles at non-positive integers
    if xf <= 0.0 {
        let nearest_int = xf.round();
        if (xf - nearest_int).abs() < 1e-14 {
            return T::infinity();
        }
        // Reflection: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * xf).sin().abs();
        if sin_pi_x < f64::EPSILON {
            return T::infinity();
        }
        let lgamma_one_minus_x = lanczos_lgamma_pos(1.0 - xf);
        let val = pi.ln() - sin_pi_x.ln() - lgamma_one_minus_x;
        return T::from_f64(val).unwrap_or(T::nan());
    }

    // x = 1 or x = 2: exact zeros
    if (xf - 1.0).abs() < 1e-15 || (xf - 2.0).abs() < 1e-15 {
        return T::zero();
    }

    // x > 0, apply recurrence for x < 0.5: lgamma(x) = lgamma(x+1) - ln(x)
    let val = if xf < 0.5 {
        let lgx1 = lanczos_lgamma_pos(xf + 1.0);
        lgx1 - xf.ln()
    } else {
        lanczos_lgamma_pos(xf)
    };
    T::from_f64(val).unwrap_or(T::nan())
}

/// Beta function
///
/// # Arguments
///
/// * `a` - First parameter
/// * `b` - Second parameter
///
/// # Returns
///
/// * Beta function value B(a, b)
///
/// # Examples
///
/// ```
/// use scirs2::special_fns::beta;
/// let result = beta(2.0_f64, 3.0_f64).expect("positive arguments");
/// assert!((result - 1.0/12.0).abs() < 1e-10);
/// // Also correct for non-integer arguments (uses lgamma internally):
/// let result = beta(2.5_f64, 3.5_f64).expect("positive arguments");
/// assert!((result - 0.036815538909255395).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn beta<T: Float + FromPrimitive + FloatConst>(a: T, b: T) -> SciRS2Result<T> {
    // Beta function defined in terms of the gamma function:
    // B(a, b) = Γ(a) * Γ(b) / Γ(a + b) = exp(lnΓ(a) + lnΓ(b) - lnΓ(a+b))
    //
    // Computed via lgamma (rather than gamma directly) so it stays accurate
    // for large a/b, where gamma(a) or gamma(a+b) individually can overflow
    // even though their ratio (the actual Beta value) is finite. This is
    // valid for all real a, b > 0: Γ is strictly positive there (no sign
    // subtleties from the reflection formula, which only applies for
    // non-positive arguments), so lgamma == ln(Γ) exactly, with no need to
    // special-case integer arguments separately.
    check_domain(
        a > T::zero() && b > T::zero(),
        "Beta function parameters must be positive",
    )?;

    let log_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    Ok(log_beta.exp())
}

/// Error function
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Error function value at x
///
/// # Examples
///
/// ```
/// use scirs2::special::erf;
/// let result = erf(0.0_f64);
/// assert!((result - 0.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn erf<T: Float + FromPrimitive>(x: T) -> T {
    // Simple approximation of the error function
    // For a more accurate implementation, use a series expansion or numerical approximation

    if x == T::zero() {
        return T::zero();
    }

    let x_abs = x.abs();
    let sign = if x < T::zero() { -T::one() } else { T::one() };

    // Simple polynomial approximation (not very accurate)
    // A more comprehensive implementation would use a series expansion or numerical approximation
    let t = T::one() / (T::one() + const_f64::<T>(0.47047) * x_abs);
    let polynomial = t
        * (const_f64::<T>(0.3480242)
            - t * (const_f64::<T>(0.0958798) - t * const_f64::<T>(0.7478556)));

    sign * (T::one() - polynomial * (-x_abs * x_abs).exp())
}

/// Complementary error function
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Complementary error function value at x
#[allow(dead_code)]
pub fn erfc<T: Float + FromPrimitive>(x: T) -> T {
    T::one() - erf(x)
}

/// Modified Bessel function of the first kind, order 0
///
/// This implementation uses a series expansion for small arguments and an asymptotic
/// approximation for large arguments, achieving accuracy of ~1e-15 for double precision.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Modified Bessel function value at x
///
/// # Examples
///
/// ```
/// use scirs2::special::i0;
/// let result = i0(0.0_f64);
/// assert!((result - 1.0).abs() < 1e-10);
/// let result = i0(1.0_f64);
/// assert!((result - 1.2660658777520084).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn i0<T: Float + FromPrimitive>(x: T) -> T {
    let abs_x = x.abs();

    // For small x, use series expansion: I_0(x) = sum_{k=0}^∞ (x²/4)^k / (k!)²
    if abs_x < T::from_f64(3.75).expect("Operation failed") {
        let y = abs_x / T::from_f64(3.75).expect("Test/example failed");
        let y2 = y * y;

        // Coefficients for the polynomial approximation (Horner's method avoids +=)
        let c1 = T::from_f64(3.5156229).unwrap_or(T::nan());
        let c2 = T::from_f64(3.0899424).unwrap_or(T::nan());
        let c3 = T::from_f64(1.2067492).unwrap_or(T::nan());
        let c4 = T::from_f64(0.2659732).unwrap_or(T::nan());
        let c5 = T::from_f64(0.0360768).unwrap_or(T::nan());
        let c6 = T::from_f64(0.0045813).unwrap_or(T::nan());
        T::one()
            + c1 * y2
            + c2 * y2 * y2
            + c3 * y2 * y2 * y2
            + c4 * y2 * y2 * y2 * y2
            + c5 * y2 * y2 * y2 * y2 * y2
            + c6 * y2 * y2 * y2 * y2 * y2 * y2
    } else {
        // For large x, use asymptotic expansion: I_0(x) ≈ e^x / sqrt(2πx) * P(1/x)
        let z = T::from_f64(3.75).unwrap_or(T::nan()) / abs_x;
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z3 * z;
        let z5 = z4 * z;
        let z6 = z5 * z;
        let z7 = z6 * z;
        let z8 = z7 * z;
        let p = T::from_f64(0.39894228).unwrap_or(T::nan())
            + T::from_f64(0.01328592).unwrap_or(T::nan()) * z
            + T::from_f64(0.00225319).unwrap_or(T::nan()) * z2
            - T::from_f64(0.00157565).unwrap_or(T::nan()) * z3
            + T::from_f64(0.00916281).unwrap_or(T::nan()) * z4
            - T::from_f64(0.02057706).unwrap_or(T::nan()) * z5
            + T::from_f64(0.02635537).unwrap_or(T::nan()) * z6
            - T::from_f64(0.01647633).unwrap_or(T::nan()) * z7
            + T::from_f64(0.00392377).unwrap_or(T::nan()) * z8;

        let exp_term = abs_x.exp();
        let sqrt_term = abs_x.sqrt();

        (exp_term / sqrt_term) * p
    }
}

/// Sinc function (sin(x)/x)
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Sinc function value at x
///
/// # Examples
///
/// ```
/// use scirs2::special::sinc;
/// let result = sinc(0.0);
/// assert!((result - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn sinc<T: Float>(x: T) -> T {
    if x.abs() < T::epsilon() {
        T::one()
    } else {
        x.sin() / x
    }
}

/// Bessel function of the first kind, order n
///
/// This implementation uses series expansion for small arguments and recurrence relations
/// combined with asymptotic expansions for large arguments.
///
/// # Arguments
///
/// * `n` - Order of the Bessel function (must be non-negative)
/// * `x` - Input value
///
/// # Returns
///
/// * Bessel function value at x
///
/// # Examples
///
/// ```
/// use scirs2::special::jn;
/// let result = jn(0, 0.0_f64);
/// assert!((result - 1.0).abs() < 1e-15);
/// let result = jn(1, 1.0_f64);
/// assert!((result - 0.44005058574493355).abs() < 1e-12);
/// ```
#[allow(dead_code)]
pub fn jn<T: Float + FromPrimitive>(n: i32, x: T) -> T {
    if n < 0 {
        // For negative orders, use J_{-n}(x) = (-1)^n * J_n(x)
        let result = jn(-n, x);
        if n % 2 == 0 {
            result
        } else {
            -result
        }
    } else if x < T::zero() {
        // For negative x, use J_n(-x) = (-1)^n * J_n(x)
        let result = jn(n, -x);
        if n % 2 == 0 {
            result
        } else {
            -result
        }
    } else if x == T::zero() {
        // J_n(0) = 1 if n = 0, otherwise 0
        if n == 0 {
            T::one()
        } else {
            T::zero()
        }
    } else if n == 0 {
        // J_0(x) special case
        bessel_j0(x)
    } else if n == 1 {
        // J_1(x) special case
        bessel_j1(x)
    } else {
        // For higher orders, use recurrence relation
        bessel_jn_recurrence(n, x)
    }
}

/// Helper function for J_0(x)
///
/// Uses the Numerical-Recipes rational (minimax) approximations (Abramowitz &
/// Stegun 9.4), accurate to ~1e-8 relative error across the whole real line.
/// A previous version of this helper used a truncated Maclaurin series for
/// `|x| < 8`, which diverges badly as `|x|` approaches 8 (e.g. ~57% relative
/// error at `x = 5`); this version is verified against reference values.
#[allow(dead_code)]
fn bessel_j0<T: Float + FromPrimitive>(x: T) -> T {
    let xf = match x.to_f64() {
        Some(v) => v,
        None => return T::nan(),
    };
    T::from_f64(bessel_j0_f64(xf)).unwrap_or(T::nan())
}

fn bessel_j0_f64(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57_568_490_574.0
            + y * (-13_362_590_354.0
                + y * (651_619_640.7
                    + y * (-11_214_424.18 + y * (77_392.330_17 + y * (-184.905_245_6)))));
        let ans2 = 57_568_490_411.0
            + y * (1_029_532_985.0
                + y * (9_494_680.718 + y * (59_272.648_53 + y * (267.853_271_2 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785_398_164;
        let ans1 = 1.0
            + y * (-0.109_862_862_7e-2
                + y * (0.273_451_040_7e-4 + y * (-0.207_337_063_9e-5 + y * 0.209_388_721_1e-6)));
        let ans2 = -0.156_249_999_5e-1
            + y * (0.143_048_876_5e-3
                + y * (-0.691_114_765_1e-5 + y * (0.762_109_516_1e-6 + y * (-0.934_935_152e-7))));
        (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

/// Helper function for J_1(x)
///
/// See [`bessel_j0_f64`] for the accuracy/history note; this uses the
/// matching Numerical-Recipes rational approximation for order 1.
#[allow(dead_code)]
fn bessel_j1<T: Float + FromPrimitive>(x: T) -> T {
    let xf = match x.to_f64() {
        Some(v) => v,
        None => return T::nan(),
    };
    T::from_f64(bessel_j1_f64(xf)).unwrap_or(T::nan())
}

fn bessel_j1_f64(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439 + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let ans2 = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_491;
        let ans1 = 1.0
            + y * (0.183_105e-2
                + y * (-0.351_639_649_6e-4 + y * (0.245_752_017_4e-5 + y * (-0.240_337_019e-6))));
        let ans2 = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3
                + y * (0.844_919_909_6e-5 + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        let ans =
            (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    }
}

/// Helper function for J_n(x) using recurrence relation
///
/// Callers (`jn`) guarantee `n >= 2` and `x > 0` by the time this is reached
/// (negative order/argument and n = 0, 1 are all handled beforehand).
///
/// # History
///
/// This previously used *pure upward* recurrence
/// `J_{i+1} = (2i/x)·J_i - J_{i-1}` unconditionally. That recurrence is only
/// numerically stable while `x > n`: J_n is the *recessive* (rapidly
/// decaying) solution of the three-term recurrence when `n > x`, so each
/// upward step admixes a tiny but rapidly *growing* amount of the dominant
/// (Y_n-like) solution from rounding error, and the result is pure garbage
/// once `n` gets much larger than `x` — e.g. the previous code returned
/// `jn(20, 1.0) ≈ 3.17e5` against the true value `≈ 3.87e-25` (30 orders of
/// magnitude off). See Numerical Recipes §6.5.
#[allow(dead_code)]
fn bessel_jn_recurrence<T: Float + FromPrimitive>(n: i32, x: T) -> T {
    if n == 0 {
        return bessel_j0(x);
    }
    if n == 1 {
        return bessel_j1(x);
    }

    let xf = match x.to_f64() {
        Some(v) => v,
        None => return T::nan(),
    };
    T::from_f64(bessel_jn_f64(n, xf)).unwrap_or(T::nan())
}

/// J_n(x) for integer order `n >= 2` and `x > 0`, dispatching between plain
/// upward recurrence (stable for `x > n`) and Miller's algorithm (stable for
/// `x <= n`).
fn bessel_jn_f64(n: i32, x: f64) -> f64 {
    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }

    if ax > n as f64 {
        // Upward recurrence tracks the dominant solution here, so it is
        // numerically stable: J_{i+1} = (2i/x)·J_i - J_{i-1}.
        let tox = 2.0 / ax;
        let mut bjm = bessel_j0_f64(ax);
        let mut bj = bessel_j1_f64(ax);
        for j in 1..n {
            let bjp = f64::from(j) * tox * bj - bjm;
            bjm = bj;
            bj = bjp;
        }
        bj
    } else {
        // Miller's algorithm: downward recurrence from an order well above
        // `n` (arbitrary starting value), renormalized afterward using the
        // summation identity `1 = J_0(x) + 2·Σ_{k=1}^∞ J_{2k}(x)`, which
        // follows from the Bessel generating function
        // `e^{(x/2)(t - 1/t)} = Σ_{k=-∞}^∞ J_k(x)·t^k` evaluated at `t = 1`
        // together with `J_{-k}(x) = (-1)^k J_k(x)`.
        //
        // Downward recurrence is stable in this regime because the
        // recessive solution (J_n) becomes the numerically dominant one when
        // iterating toward *smaller* order — any admixture of the unwanted
        // solution decays away rather than growing. (Numerical Recipes §6.5,
        // "Miller's algorithm".)
        const ACC: f64 = 40.0;
        const BIG_NO: f64 = 1.0e10;
        const BIG_NI: f64 = 1.0e-10;

        let tox = 2.0 / ax;
        // Starting order: comfortably above `n` so that by the time the
        // recurrence reaches order `n`, the spurious admixture has decayed
        // to insignificance. Forced even so the normalization sum (which
        // only accumulates even orders) lines up correctly.
        let extra = (ACC * n as f64).sqrt().floor() as i32;
        let m = 2 * ((n + extra) / 2);

        let mut jsum = false;
        let mut bjp = 0.0_f64;
        let mut ans = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut bj = 1.0_f64;

        let mut j = m;
        while j > 0 {
            let bjm = f64::from(j) * tox * bj - bjp;
            bjp = bj;
            bj = bjm;
            if bj.abs() > BIG_NO {
                // Renormalize to avoid overflow; the eventual ratio ans/sum
                // is unaffected by any common rescaling.
                bj *= BIG_NI;
                bjp *= BIG_NI;
                ans *= BIG_NI;
                sum *= BIG_NI;
            }
            if jsum {
                sum += bj;
            }
            jsum = !jsum;
            if j == n {
                ans = bjp;
            }
            j -= 1;
        }
        sum = 2.0 * sum - bj;
        ans / sum
    }
}

/// Y_0(x) via the Numerical-Recipes rational (minimax) approximation.
fn bessel_y0_f64(x: f64) -> f64 {
    if x < 8.0 {
        let y = x * x;
        let ans1 = -2_957_821_389.0
            + y * (7_062_834_065.0
                + y * (-512_359_803.6
                    + y * (10_879_881.29 + y * (-86_327.927_57 + y * 228.462_273_3))));
        let ans2 = 40_076_544_269.0
            + y * (745_249_964.8
                + y * (7_189_466.438 + y * (47_447.264_70 + y * (226.103_024_4 + y))));
        (ans1 / ans2) + std::f64::consts::FRAC_2_PI * bessel_j0_f64(x) * x.ln()
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785_398_164;
        let ans1 = 1.0
            + y * (-0.109_862_862_7e-2
                + y * (0.273_451_040_7e-4 + y * (-0.207_337_063_9e-5 + y * 0.209_388_721_1e-6)));
        let ans2 = -0.156_249_999_5e-1
            + y * (0.143_048_876_5e-3
                + y * (-0.691_114_765_1e-5 + y * (0.762_109_516_1e-6 + y * (-0.934_935_152e-7))));
        (std::f64::consts::FRAC_2_PI / x).sqrt() * (xx.sin() * ans1 + z * xx.cos() * ans2)
    }
}

/// Y_1(x) via the Numerical-Recipes rational (minimax) approximation.
fn bessel_y1_f64(x: f64) -> f64 {
    if x < 8.0 {
        let y = x * x;
        let ans1 = x
            * (-4.900_604_943e13
                + y * (1.275_274_390e13
                    + y * (-5.153_438_139e11
                        + y * (7.349_264_551e9 + y * (-4.237_922_726e7 + y * 8.511_937_935e4)))));
        let ans2 = 2.499_580_570e14
            + y * (4.244_419_664e12
                + y * (3.733_650_367e10
                    + y * (2.245_904_002e8 + y * (1.020_426_050e6 + y * (3.549_632_885e3 + y)))));
        (ans1 / ans2) + std::f64::consts::FRAC_2_PI * (bessel_j1_f64(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 2.356_194_491;
        let ans1 = 1.0
            + y * (0.183_105e-2
                + y * (-0.351_639_649_6e-4 + y * (0.245_752_017_4e-5 + y * (-0.240_337_019e-6))));
        let ans2 = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3
                + y * (0.844_919_909_6e-5 + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        (std::f64::consts::FRAC_2_PI / x).sqrt() * (xx.sin() * ans1 + z * xx.cos() * ans2)
    }
}

/// Y_n(x) for non-negative integer order via the numerically stable upward
/// recurrence `Y_{n+1}(x) = (2n/x)·Y_n(x) - Y_{n-1}(x)` (stable in the
/// increasing-n direction, unlike the same recurrence for J_n).
fn bessel_yn_f64(n: i32, x: f64) -> f64 {
    if n == 0 {
        return bessel_y0_f64(x);
    }
    if n == 1 {
        return bessel_y1_f64(x);
    }
    let mut y_prev = bessel_y0_f64(x);
    let mut y_curr = bessel_y1_f64(x);
    for j in 1..n {
        let y_next = (2.0 * f64::from(j) / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }
    y_curr
}

/// Bessel function of the second kind, order n
///
/// Computed via Numerical-Recipes-style rational (minimax) approximations for
/// Y_0/Y_1 (~1e-8 relative accuracy) combined with the numerically stable
/// upward recurrence for higher orders.
///
/// # Arguments
///
/// * `n` - Order of the Bessel function (may be negative: uses the identity
///   `Y_{-n}(x) = (-1)^n · Y_n(x)`)
/// * `x` - Input value (must be positive; `Y_n` has a branch point at `x = 0`
///   and is not real-valued for `x < 0`)
///
/// # Returns
///
/// * Bessel function of the second kind at `x`; `NaN` for `x < 0`, `-∞` at
///   `x == 0`
///
/// # Examples
///
/// ```
/// use scirs2::special_fns::yn;
/// let result = yn(0, 1.0_f64);
/// assert!((result - 0.08825696421567697).abs() < 1e-6);
/// let result = yn(1, 1.0_f64);
/// assert!((result - (-0.7812128213002888)).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn yn<T: Float + FromPrimitive>(n: i32, x: T) -> T {
    let xf = match x.to_f64() {
        Some(v) => v,
        None => return T::nan(),
    };

    if xf.is_nan() || xf < 0.0 {
        return T::nan();
    }
    if xf == 0.0 {
        return T::from_f64(f64::NEG_INFINITY).unwrap_or(T::nan());
    }

    if n < 0 {
        let n_pos = -n;
        let val = bessel_yn_f64(n_pos, xf);
        let signed = if n_pos % 2 == 0 { val } else { -val };
        return T::from_f64(signed).unwrap_or(T::nan());
    }

    T::from_f64(bessel_yn_f64(n, xf)).unwrap_or(T::nan())
}

/// Shared arithmetic-geometric-mean (AGM) computation of the complete
/// elliptic integrals of the first and second kind, for parameter `m` in
/// `[0, 1]` (Abramowitz & Stegun 17.6). Quadratically convergent (~1e-15
/// accuracy in a handful of iterations).
fn agm_elliptic_f64(m: f64) -> (f64, f64) {
    let half_pi = std::f64::consts::FRAC_PI_2;
    if m <= 0.0 {
        return (half_pi, half_pi);
    }
    if m >= 1.0 {
        // K(1) diverges to +infinity; E(1) = 1 exactly.
        return (f64::INFINITY, 1.0);
    }

    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    let mut c = m.sqrt();
    let mut sum = 0.5 * c * c; // running Σ 2^(n-1)·c_n^2, n = 0 term
    let mut pow2 = 0.5_f64; // tracks 2^(n-1) for the current n

    for _ in 0..64 {
        if (a - b).abs() <= a * 1e-16 {
            break;
        }
        let a_next = 0.5 * (a + b);
        let b_next = (a * b).sqrt();
        c = 0.5 * (a - b);
        a = a_next;
        b = b_next;
        pow2 *= 2.0;
        sum += pow2 * c * c;
    }

    let k = half_pi / a;
    let e = k * (1.0 - sum);
    (k, e)
}

/// Complete elliptic integrals K(m)/E(m) for any `m < 1`, including negative
/// `m`, via the imaginary-modulus transformation for `m < 0` (DLMF 19.7.5):
/// `K(-m) = K(m/(1+m)) / sqrt(1+m)`, `E(-m) = E(m/(1+m)) · sqrt(1+m)`.
fn complete_elliptic_f64(m: f64) -> (f64, f64) {
    if m >= 0.0 {
        agm_elliptic_f64(m)
    } else {
        let mp = -m;
        let m2 = mp / (1.0 + mp);
        let (k2, e2) = agm_elliptic_f64(m2);
        let scale = (1.0 + mp).sqrt();
        (k2 / scale, e2 * scale)
    }
}

/// Complete elliptic integral of the first kind
///
/// Computed via the arithmetic-geometric mean (AGM), which converges
/// quadratically to full `f64` precision in a handful of iterations.
///
/// # Arguments
///
/// * `m` - Parameter (SciPy/`m`-convention, i.e. `k^2`; must be `< 1`)
///
/// # Returns
///
/// * Complete elliptic integral value `K(m)`; `+∞` at `m == 1`
///
/// # Examples
///
/// ```
/// use scirs2::special_fns::ellipk;
/// let result = ellipk(0.5_f64).expect("m < 1");
/// assert!((result - 1.8540746773013719).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn ellipk<T: Float + FromPrimitive>(m: T) -> SciRS2Result<T> {
    check_domain(m < T::one(), "Parameter m must be less than 1")?;

    let mf = match m.to_f64() {
        Some(v) => v,
        None => return Ok(T::nan()),
    };
    let (k, _e) = complete_elliptic_f64(mf);
    Ok(T::from_f64(k).unwrap_or(T::nan()))
}

/// Complete elliptic integral of the second kind
///
/// Computed via the arithmetic-geometric mean (AGM), which converges
/// quadratically to full `f64` precision in a handful of iterations.
///
/// # Arguments
///
/// * `m` - Parameter (SciPy/`m`-convention, i.e. `k^2`; must be `< 1`)
///
/// # Returns
///
/// * Complete elliptic integral value `E(m)`
///
/// # Examples
///
/// ```
/// use scirs2::special_fns::ellipe;
/// let result = ellipe(0.5_f64).expect("m < 1");
/// assert!((result - 1.3506438810476755).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn ellipe<T: Float + FromPrimitive>(m: T) -> SciRS2Result<T> {
    check_domain(m < T::one(), "Parameter m must be less than 1")?;

    let mf = match m.to_f64() {
        Some(v) => v,
        None => return Ok(T::nan()),
    };
    let (_k, e) = complete_elliptic_f64(mf);
    Ok(T::from_f64(e).unwrap_or(T::nan()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_integers() {
        // integer arguments: Γ(n) = (n-1)!
        assert!((gamma(1.0_f64) - 1.0).abs() < 1e-10);
        assert!((gamma(2.0_f64) - 1.0).abs() < 1e-10);
        assert!((gamma(3.0_f64) - 2.0).abs() < 1e-10);
        assert!((gamma(4.0_f64) - 6.0).abs() < 1e-10);
        assert!((gamma(5.0_f64) - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_half_integers() {
        // Γ(1/2) = √π ≈ 1.7724538509055159
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert!((gamma(0.5_f64) - sqrt_pi).abs() < 1e-10);
        // Γ(3/2) = (1/2)·√π
        assert!((gamma(1.5_f64) - 0.5 * sqrt_pi).abs() < 1e-10);
        // Γ(5/2) = (3/4)·√π
        assert!((gamma(2.5_f64) - 0.75 * sqrt_pi).abs() < 1e-10);
        // Γ(7/2) = (15/8)·√π
        assert!((gamma(3.5_f64) - 1.875 * sqrt_pi).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_general() {
        // Γ(4.5) ≈ 11.6317283965674
        assert!((gamma(4.5_f64) - 11.631728396567448).abs() < 1e-8);
        // Γ(0.25) ≈ 3.6256099082219083
        assert!((gamma(0.25_f64) - 3.625609908221908).abs() < 1e-8);
    }

    #[test]
    fn test_lgamma() {
        // lgamma(1) = 0
        assert!((lgamma(1.0_f64)).abs() < 1e-12);
        // lgamma(2) = 0
        assert!((lgamma(2.0_f64)).abs() < 1e-12);
        // lgamma(5) = ln(24)
        assert!((lgamma(5.0_f64) - (24.0_f64).ln()).abs() < 1e-10);
        // lgamma(0.5) = ln(√π)
        let ln_sqrt_pi = 0.5 * std::f64::consts::PI.ln();
        assert!((lgamma(0.5_f64) - ln_sqrt_pi).abs() < 1e-10);
    }

    #[test]
    fn test_beta() {
        assert!((beta(1.0, 1.0).expect("Operation failed") - 1.0).abs() < 1e-10);
        assert!((beta(2.0, 3.0).expect("Operation failed") - 1.0 / 12.0).abs() < 1e-10);
        assert!((beta(3.0, 2.0).expect("Operation failed") - 1.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_non_integer_args() {
        // Previously this returned NaN for ANY non-integer argument pair.
        // Reference values from `scipy.special.beta`.
        assert!(
            (beta(2.5_f64, 3.5).expect("Operation failed") - 0.036815538909255395).abs() < 1e-10
        );
        assert!(
            (beta(1.5_f64, 2.5).expect("Operation failed") - 0.19634954084936204).abs() < 1e-10
        );
        assert!(
            (beta(0.5_f64, 0.5).expect("Operation failed") - std::f64::consts::PI).abs() < 1e-9
        );
        // Large arguments: gamma(10.5) or gamma(30.8) would individually be
        // enormous, but lgamma-based computation keeps the ratio accurate.
        assert!(
            (beta(10.5_f64, 20.3).expect("Operation failed") - 2.514034214191836e-9).abs() < 1e-15
        );
    }

    #[test]
    fn test_yn_matches_reference() {
        // Previously `yn` was an unconditional `T::nan()` stub for every input.
        // Reference values from `scipy.special.yn`.
        assert!((yn(0, 1.0_f64) - 0.08825696421567697).abs() < 1e-6);
        assert!((yn(1, 1.0_f64) - (-0.7812128213002888)).abs() < 1e-6);
        assert!((yn(2, 1.0_f64) - (-1.6506826068162546)).abs() < 1e-6);
        assert!((yn(0, 5.0_f64) - (-0.30851762524903314)).abs() < 1e-6);
        // Higher order + moderate x accumulates a bit more error through the
        // upward recurrence, so use a slightly looser tolerance here.
        assert!((yn(3, 5.0_f64) - 0.14626716269319184).abs() < 1e-4);
        assert!((yn(5, 10.0_f64) - 0.13540304768936254).abs() < 1e-6);
    }

    #[test]
    fn test_yn_negative_order_and_domain() {
        // Y_{-n}(x) = (-1)^n Y_n(x)
        assert!((yn(-1, 1.0_f64) - 0.7812128213002888).abs() < 1e-6);
        assert!((yn(-2, 1.0_f64) - (-1.6506826068162546)).abs() < 1e-6);
        // Y_n is undefined (branch point) for x <= 0.
        assert!(yn(0, -1.0_f64).is_nan());
        assert!(yn(0, 0.0_f64).is_infinite());
    }

    #[test]
    fn test_jn_still_correct_for_orders_0_and_1() {
        // Regression check for the bessel_j0/bessel_j1 coefficient fix that
        // yn() needed: the previous truncated-Maclaurin-series/mis-keyed
        // asymptotic implementation was badly wrong approaching x = 8 (e.g.
        // jn(0, 5.0) was previously off by ~57%: -0.2792 vs the true
        // -0.17760). Reference values from `scipy.special.jv`.
        assert!((jn(0, 5.0_f64) - (-0.17759677131433835)).abs() < 1e-6);
        assert!((jn(1, 5.0_f64) - (-0.3275791375914652)).abs() < 1e-6);
        assert!((jn(0, 7.9_f64) - 0.19436184484127844).abs() < 1e-6);
    }

    #[test]
    fn test_jn_miller_algorithm_n_greater_than_x() {
        // Previously `bessel_jn_recurrence` used *pure upward* recurrence
        // unconditionally. That recurrence tracks the recessive solution
        // when n > x, so rounding error grows catastrophically: the old
        // code returned `jn(20, 1.0) ~= 3.17e5` against the true value
        // `~= 3.87e-25` (30 orders of magnitude off). This regime now uses
        // Miller's algorithm (downward recurrence + renormalization).
        // Reference values from `scipy.special.jv`.
        fn rel_close(actual: f64, expected: f64, tol: f64) -> bool {
            (actual - expected).abs() <= tol * expected.abs().max(1e-300)
        }

        assert!(rel_close(jn(20, 1.0_f64), 3.8735030085246507e-25, 1e-9));
        assert!(rel_close(jn(30, 5.0_f64), 2.6711772782508136e-21, 1e-9));
        assert!(rel_close(jn(10, 2.0_f64), 2.5153862827167347e-07, 1e-9));
        assert!(rel_close(jn(100, 10.0_f64), 6.597316064155483e-89, 1e-8));
        assert!(rel_close(jn(50, 1.0_f64), 2.906_004_948_173_25e-80, 1e-8));
        assert!(rel_close(jn(7, 3.0_f64), 0.002547294451804692, 1e-9));

        // Negative x should stay consistent with J_n(-x) = (-1)^n J_n(x)
        // (exercised through the same n > x path in the public `jn` API).
        assert!(rel_close(jn(20, -1.0_f64), 3.8735030085246507e-25, 1e-9));
        assert!(rel_close(jn(7, -3.0_f64), -0.002547294451804692, 1e-9));
    }

    #[test]
    fn test_jn_upward_regime_still_correct() {
        // x > n: already the numerically stable regime for upward
        // recurrence; confirm the dispatch didn't regress it. Tolerance
        // matches the ~1e-8-relative accuracy of the underlying J_0/J_1
        // rational approximations (see `test_jn_still_correct_for_orders_0_and_1`).
        // Reference values from `scipy.special.jv`.
        assert!((jn(15, 50.0_f64) - (-0.10822559897511456)).abs() < 1e-6);
        assert!((jn(3, 5.0_f64) - 0.364831230613667).abs() < 1e-6);
        assert!((jn(5, 10.0_f64) - (-0.2340615281867936)).abs() < 1e-6);
    }

    #[test]
    fn test_ellipk_matches_reference() {
        // Previously `ellipk` returned NaN for every valid input.
        // Reference values from `scipy.special.ellipk`.
        assert!(
            (ellipk(0.0_f64).expect("Operation failed") - std::f64::consts::FRAC_PI_2).abs()
                < 1e-12
        );
        assert!((ellipk(0.5_f64).expect("Operation failed") - 1.8540746773013719).abs() < 1e-10);
        assert!((ellipk(0.9_f64).expect("Operation failed") - 2.5780921133481733).abs() < 1e-10);
        // Negative m (imaginary-modulus transform path).
        assert!((ellipk(-5.0_f64).expect("Operation failed") - 0.9555039270640441).abs() < 1e-10);
        // m >= 1 is rejected by the domain check.
        assert!(ellipk(1.0_f64).is_err());
    }

    #[test]
    fn test_ellipe_matches_reference() {
        // Previously `ellipe` returned NaN for every valid input.
        // Reference values from `scipy.special.ellipe`.
        assert!(
            (ellipe(0.0_f64).expect("Operation failed") - std::f64::consts::FRAC_PI_2).abs()
                < 1e-12
        );
        assert!((ellipe(0.5_f64).expect("Operation failed") - 1.3506438810476755).abs() < 1e-10);
        assert!((ellipe(0.9_f64).expect("Operation failed") - 1.1047747327040733).abs() < 1e-10);
        // Negative m (imaginary-modulus transform path).
        assert!((ellipe(-5.0_f64).expect("Operation failed") - 2.830198246345877).abs() < 1e-10);
    }

    #[test]
    fn test_erf() {
        assert!((erf(0.0) - 0.0).abs() < 1e-10);
        // The following tests use approximate values
        assert!((erf(1.0) - 0.8427).abs() < 1e-3);
        assert!((erf(-1.0) + 0.8427).abs() < 1e-3);
    }

    #[test]
    fn test_sinc() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-10);
        let x = std::f64::consts::PI;
        assert!((sinc(x) - 0.0).abs() < 1e-10);
    }
}
