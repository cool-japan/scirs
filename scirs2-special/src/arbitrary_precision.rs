//! Arbitrary precision computation support for special functions.
//!
//! This module provides arbitrary precision implementations of special functions
//! using Pure Rust arbitrary-precision arithmetic via `oxinum-float` (dashu-based,
//! no C/Fortran/MPFR dependencies).  This allows for computations with
//! user-specified precision beyond the limitations of f64.
//!
//! The public API is source-compatible with the former `rug`-based implementation:
//! - `PrecisionContext` stores precision in **bits** (same convention as `rug`).
//! - `MpFloat` (re-exported as `Float`) and `MpComplex` replace `rug::Float` /
//!   `rug::Complex`.
//! - `gamma_mpfr`, `erf_mpfr`, `bessel_j0_mpfr`, `bessel_k0_mpfr` etc. work
//!   identically from the caller's perspective.

#![allow(dead_code)]

use crate::error::{SpecialError, SpecialResult};

// Re-export the oxinum-float types so call sites that used `rug::Float` /
// `rug::Complex` can import `Float` and `Complex` from this module directly.
#[cfg(feature = "high-precision")]
pub use oxinum_float::mp_float::{MpComplex as Complex, MpFloat as Float};
#[cfg(feature = "high-precision")]
use oxinum_float::mp_float::{MpComplex, MpFloat, Round};

/// Default precision in bits for arbitrary precision computations.
pub const DEFAULT_PRECISION: u32 = 256;

/// Maximum supported precision in bits.
pub const MAX_PRECISION: u32 = 4096;

// ---------------------------------------------------------------------------
// PrecisionContext
// ---------------------------------------------------------------------------

/// Precision context for arbitrary precision computations.
#[derive(Debug, Clone)]
pub struct PrecisionContext {
    /// Precision in bits (same convention as `rug::Float`).
    precision: u32,
    /// Rounding mode.
    #[cfg(feature = "high-precision")]
    rounding: Round,
    #[cfg(not(feature = "high-precision"))]
    _rounding: (),
}

impl Default for PrecisionContext {
    fn default() -> Self {
        Self {
            precision: DEFAULT_PRECISION,
            #[cfg(feature = "high-precision")]
            rounding: Round::Nearest,
            #[cfg(not(feature = "high-precision"))]
            _rounding: (),
        }
    }
}

impl PrecisionContext {
    /// Create a new precision context with specified precision in bits.
    pub fn new(precision: u32) -> SpecialResult<Self> {
        if precision == 0 || precision > MAX_PRECISION {
            return Err(SpecialError::DomainError(format!(
                "Precision must be between 1 and {} bits",
                MAX_PRECISION
            )));
        }
        Ok(Self {
            precision,
            #[cfg(feature = "high-precision")]
            rounding: Round::Nearest,
            #[cfg(not(feature = "high-precision"))]
            _rounding: (),
        })
    }

    /// Set the rounding mode.
    #[cfg(feature = "high-precision")]
    pub fn with_rounding(mut self, rounding: Round) -> Self {
        self.rounding = rounding;
        self
    }

    /// Get the precision in bits.
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Get the rounding mode.
    #[cfg(feature = "high-precision")]
    pub fn rounding(&self) -> Round {
        self.rounding
    }

    /// Create a `MpFloat` with the context's precision from an f64.
    #[cfg(feature = "high-precision")]
    pub fn float(&self, value: f64) -> MpFloat {
        MpFloat::with_val(self.precision, value)
    }

    /// Create a `MpComplex` with the context's precision from `(real, imag)`.
    #[cfg(feature = "high-precision")]
    pub fn complex(&self, real: f64, imag: f64) -> MpComplex {
        MpComplex::with_val(self.precision, (real, imag))
    }

    /// Create pi (π) with the context's precision.
    #[cfg(feature = "high-precision")]
    pub fn pi(&self) -> MpFloat {
        use oxinum_float::compute_pi;
        use oxinum_float::mp_float::bits_to_decimal_prec;
        let prec = bits_to_decimal_prec(self.precision);
        let d = compute_pi(prec);
        MpFloat::from_dbig(&d, self.precision)
    }

    /// Create e (Euler's number) with the context's precision.
    #[cfg(feature = "high-precision")]
    pub fn e(&self) -> MpFloat {
        MpFloat::with_val(self.precision, 1.0).exp()
    }

    /// Create ln(2) with the context's precision.
    #[cfg(feature = "high-precision")]
    pub fn ln2(&self) -> MpFloat {
        use oxinum_float::compute_ln2;
        use oxinum_float::mp_float::bits_to_decimal_prec;
        let prec = bits_to_decimal_prec(self.precision);
        let d = compute_ln2(prec);
        MpFloat::from_dbig(&d, self.precision)
    }

    /// Create the Euler–Mascheroni constant γ with the context's precision.
    #[cfg(feature = "high-precision")]
    pub fn euler_gamma(&self) -> MpFloat {
        oxinum_float::mp_float::euler_gamma_at_bits(self.precision)
    }

    /// Create Catalan's constant G with the context's precision.
    #[cfg(feature = "high-precision")]
    pub fn catalan(&self) -> MpFloat {
        oxinum_float::mp_float::catalan_at_bits(self.precision)
    }
}

// ---------------------------------------------------------------------------
// Utility used throughout sub-modules: build MpFloat at prec from f64
// ---------------------------------------------------------------------------

#[cfg(feature = "high-precision")]
fn mpf(bits: u32, v: f64) -> MpFloat {
    MpFloat::with_val(bits, v)
}

// ---------------------------------------------------------------------------
// Arbitrary precision Gamma function
// ---------------------------------------------------------------------------

/// Arbitrary precision Gamma function.
pub mod gamma {
    use super::*;

    /// Compute the Gamma function Γ(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn gamma_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        gamma_mp(&x_mp, ctx)
    }

    /// Compute the Gamma function for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    pub fn gamma_mp(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if x.is_zero() || (x.is_finite() && x < &0.0 && x.is_integer()) {
            return Err(SpecialError::DomainError(
                "Gamma function undefined at non-positive integers".to_string(),
            ));
        }

        if x > &20.0 {
            stirling_gamma(x, ctx)
        } else if x > &0.0 {
            lanczos_gamma(x, ctx)
        } else {
            reflection_gamma(x, ctx)
        }
    }

    /// Stirling's approximation for Gamma function.
    #[cfg(feature = "high-precision")]
    fn stirling_gamma(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let two_pi = ctx.pi() * mpf(prec, 2.0);
        let sqrt_2pi = two_pi.sqrt();
        let e = ctx.e();

        let term1 = sqrt_2pi / x.clone().sqrt();
        // (x/e)^x via x * (ln(x) - 1) = x*ln(x) - x
        let term2 = (x.clone() / e).pow_float(x);

        let mut correction = ctx.float(1.0);
        let x2 = x.clone() * x;
        let x3 = x2.clone() * x;
        let x4 = x2.clone() * &x2;

        correction += ctx.float(1.0) / (ctx.float(12.0) * x);
        correction += ctx.float(1.0) / (ctx.float(288.0) * &x2);
        let denom1 = ctx.float(51840.0) * &x3;
        correction -= ctx.float(139.0) / denom1;
        let denom2 = ctx.float(2488320.0) * &x4;
        correction -= ctx.float(571.0) / denom2;

        Ok(term1 * term2 * correction)
    }

    /// Lanczos approximation for Gamma function.
    #[cfg(feature = "high-precision")]
    fn lanczos_gamma(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        const LANCZOS_G: f64 = 7.0;
        const LANCZOS_COEFFS: &[f64] = &[
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let prec = ctx.precision;
        let g = ctx.float(LANCZOS_G);
        let sqrt_2pi = (ctx.pi() * ctx.float(2.0)).sqrt();

        let mut ag = ctx.float(LANCZOS_COEFFS[0]);
        for (i, &c) in LANCZOS_COEFFS[1..].iter().enumerate() {
            ag += ctx.float(c) / (x.clone() + (i as f64 + 1.0));
        }

        let tmp = x.clone() + &g + ctx.float(0.5);
        let result =
            sqrt_2pi * ag * tmp.clone().pow_float(&(x.clone() + 0.5)) * (-tmp.clone()).exp();

        Ok(result / x)
    }

    /// Reflection formula for negative x.
    #[cfg(feature = "high-precision")]
    fn reflection_gamma(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let pi = ctx.pi();
        let sin_pi_x = (pi.clone() * x).sin();

        if sin_pi_x.is_zero() {
            return Err(SpecialError::DomainError(
                "Gamma function has poles at negative integers".to_string(),
            ));
        }

        let one_minus_x = ctx.float(1.0) - x;
        let pos_gamma = gamma_mp(&one_minus_x, ctx)?;
        Ok(pi / (sin_pi_x * pos_gamma))
    }

    /// Compute log(Gamma(x)) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn log_gamma_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        log_gamma_mp(&x_mp, ctx)
    }

    /// Compute log(Gamma(x)) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    pub fn log_gamma_mp(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if x.is_zero() || (x.is_finite() && x < &0.0) {
            return Err(SpecialError::DomainError(
                "log_gamma undefined for non-positive values".to_string(),
            ));
        }

        if x > &10.0 {
            stirling_log_gamma(x, ctx)
        } else {
            let gamma_x = gamma_mp(x, ctx)?;
            Ok(gamma_x.ln())
        }
    }

    /// Stirling's approximation for log(Gamma(x)).
    #[cfg(feature = "high-precision")]
    fn stirling_log_gamma(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let two_pi = ctx.pi() * mpf(prec, 2.0);
        let ln_2pi = two_pi.ln();

        let mut result = (x.clone() - 0.5) * x.clone().ln() - x.clone() + ln_2pi / 2.0;

        let x2 = x.clone() * x;
        let x3 = x2.clone() * x;
        let x5 = x3.clone() * &x2;
        let x7 = x5.clone() * &x2;

        result += ctx.float(1.0) / (ctx.float(12.0) * x);
        let denom3 = ctx.float(360.0) * &x3;
        result -= ctx.float(1.0) / denom3;
        let denom4 = ctx.float(1260.0) * &x5;
        result += ctx.float(1.0) / denom4;
        let denom5 = ctx.float(1680.0) * &x7;
        result -= ctx.float(1.0) / denom5;

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Arbitrary precision Bessel functions
// ---------------------------------------------------------------------------

/// Arbitrary precision Bessel functions.
pub mod bessel {
    use super::*;

    /// Compute Bessel J_n(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn bessel_j_ap(n: i32, x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        bessel_j_mp(n, &x_mp, ctx)
    }

    /// Compute Bessel J_n(x) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    pub fn bessel_j_mp(n: i32, x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if x.is_zero() {
            return Ok(if n == 0 {
                ctx.float(1.0)
            } else {
                ctx.float(0.0)
            });
        }

        if x.clone().abs() < 10.0 {
            bessel_j_series(n, x, ctx)
        } else {
            bessel_j_asymptotic(n, x, ctx)
        }
    }

    /// Power series for Bessel J_n(x).
    #[cfg(feature = "high-precision")]
    fn bessel_j_series(n: i32, x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let mut sum = ctx.float(0.0);
        let x_half = x.clone() / 2.0;
        let x2_quarter = x_half.clone() * &x_half;

        let mut term = x_half.pow_i32(n) / factorial_mp(n.unsigned_abs(), ctx);
        let sign = if n < 0 && n % 2 != 0 { -1.0 } else { 1.0 };
        term *= sign;

        sum += &term;

        for k in 1..200 {
            let divisor = mpf(prec, k as f64 * (k as f64 + n.unsigned_abs() as f64));
            let neg_x2_quarter = -x2_quarter.clone();
            term *= neg_x2_quarter / divisor;
            sum += &term;

            if term.clone().abs() < sum.clone().abs() * mpf(prec, 10.0).pow_i32(-(prec as i32) / 10)
            {
                break;
            }
        }

        Ok(sum)
    }

    /// Asymptotic expansion for Bessel J_n(x).
    #[cfg(feature = "high-precision")]
    fn bessel_j_asymptotic(n: i32, x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let pi = ctx.pi();
        let pi_x = pi.clone() * x;
        let sqrt_2_pi_x = (ctx.float(2.0) / pi_x).sqrt();

        let phase_coefficient = mpf(prec, n as f64 + 0.5);
        let phase_pi_mult = phase_coefficient * &pi;
        let phase_offset = phase_pi_mult / 2.0;
        let phase = x.clone() - phase_offset;
        let cos_phase = phase.cos();

        let mut correction = ctx.float(1.0);
        let n2 = (n * n) as f64;
        let x2 = x.clone() * x;

        let x_mult = mpf(prec, 8.0) * x;
        correction -= mpf(prec, 4.0 * n2 - 1.0) / x_mult;
        let x2_mult = mpf(prec, 128.0) * &x2;
        correction += mpf(prec, (4.0 * n2 - 1.0) * (4.0 * n2 - 9.0)) / x2_mult;

        Ok(sqrt_2_pi_x * cos_phase * correction)
    }

    /// Compute Bessel Y_n(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn bessel_y_ap(n: i32, x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        bessel_y_mp(n, &x_mp, ctx)
    }

    /// Compute Bessel Y_n(x) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    pub fn bessel_y_mp(n: i32, x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if x <= &0.0 {
            return Err(SpecialError::DomainError(
                "Bessel Y function undefined for non-positive arguments".to_string(),
            ));
        }

        if x > &10.0 {
            bessel_y_asymptotic(n, x, ctx)
        } else {
            bessel_y_relation(n, x, ctx)
        }
    }

    /// Compute Y_n using relation with J_n.
    #[cfg(feature = "high-precision")]
    fn bessel_y_relation(n: i32, x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let pi = ctx.pi();

        if n >= 0 {
            let jn = bessel_j_mp(n, x, ctx)?;
            let jn_neg = bessel_j_mp(-n, x, ctx)?;
            let cos_n_pi = if n % 2 == 0 { 1.0 } else { -1.0 };

            let n_pi = mpf(prec, n as f64) * &pi;
            Ok((jn * cos_n_pi - jn_neg) / n_pi.sin())
        } else {
            let yn_pos = bessel_y_mp(-n, x, ctx)?;
            Ok(if n % 2 == 0 { yn_pos } else { -yn_pos })
        }
    }

    /// Asymptotic expansion for Bessel Y_n(x).
    #[cfg(feature = "high-precision")]
    fn bessel_y_asymptotic(n: i32, x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let pi = ctx.pi();
        let pi_x = pi.clone() * x;
        let sqrt_2_pi_x = (ctx.float(2.0) / pi_x).sqrt();

        let phase_coefficient = mpf(prec, n as f64 + 0.5);
        let phase_pi_mult = phase_coefficient * &pi;
        let phase_offset = phase_pi_mult / 2.0;
        let phase = x.clone() - phase_offset;
        let sin_phase = phase.sin();

        let mut correction = ctx.float(1.0);
        let n2 = (n * n) as f64;
        let x2 = x.clone() * x;

        let x_mult = mpf(prec, 8.0) * x;
        correction -= mpf(prec, 4.0 * n2 - 1.0) / x_mult;
        let x2_mult = mpf(prec, 128.0) * &x2;
        correction += mpf(prec, (4.0 * n2 - 1.0) * (4.0 * n2 - 9.0)) / x2_mult;

        Ok(sqrt_2_pi_x * sin_phase * correction)
    }
}

// ---------------------------------------------------------------------------
// Arbitrary precision error functions
// ---------------------------------------------------------------------------

/// Arbitrary precision error functions.
pub mod error_function {
    use super::*;

    /// Compute erf(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn erf_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        erf_mp(&x_mp, ctx)
    }

    /// Compute erf(x) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    pub fn erf_mp(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if x.is_zero() {
            return Ok(ctx.float(0.0));
        }

        let abs_x = x.clone().abs();

        if abs_x < 2.0 {
            erf_series(x, ctx)
        } else {
            let erfc_val = erfc_asymptotic(&abs_x, ctx)?;
            Ok(if x > &0.0 {
                ctx.float(1.0) - erfc_val
            } else {
                erfc_val - ctx.float(1.0)
            })
        }
    }

    /// Taylor series for erf(x).
    #[cfg(feature = "high-precision")]
    fn erf_series(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let sqrt_pi = ctx.pi().sqrt();
        let x2 = x.clone() * x;

        let mut sum = x.clone();
        let mut term = x.clone();

        for n in 1..200 {
            let neg_x2 = -x2.clone();
            term *= neg_x2 / (n as f64);
            let new_term = term.clone() / (2 * n + 1) as f64;
            sum += &new_term;

            if new_term.clone().abs()
                < sum.clone().abs() * mpf(prec, 10.0).pow_i32(-(prec as i32) / 10)
            {
                break;
            }
        }

        Ok(mpf(prec, 2.0) * sum / sqrt_pi)
    }

    /// Asymptotic expansion for erfc(x) for large x.
    #[cfg(feature = "high-precision")]
    fn erfc_asymptotic(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        let sqrt_pi = ctx.pi().sqrt();
        let x2 = x.clone() * x;
        let neg_x2 = -x2.clone();
        let exp_neg_x2 = neg_x2.exp();

        let mut sum = ctx.float(1.0);
        let mut term = ctx.float(1.0);

        for n in 1..50 {
            let x2_mult = mpf(prec, 2.0) * &x2;
            term *= mpf(prec, -((2 * n - 1) as f64)) / x2_mult;
            sum += &term;

            if term.clone().abs() < sum.clone().abs() * mpf(prec, 10.0).pow_i32(-(prec as i32) / 10)
            {
                break;
            }
        }

        Ok(exp_neg_x2 * sum / (x.clone() * sqrt_pi))
    }

    /// Compute erfc(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn erfc_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        erfc_mp(&x_mp, ctx)
    }

    /// Compute erfc(x) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    pub fn erfc_mp(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if x.is_zero() {
            return Ok(ctx.float(1.0));
        }

        let abs_x = x.clone().abs();

        if abs_x < 2.0 {
            let erf_val = erf_mp(x, ctx)?;
            Ok(ctx.float(1.0) - erf_val)
        } else {
            if x > &0.0 {
                erfc_asymptotic(x, ctx)
            } else {
                let erfc_pos = erfc_asymptotic(&abs_x, ctx)?;
                Ok(ctx.float(2.0) - erfc_pos)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions for arbitrary precision
// ---------------------------------------------------------------------------

mod utils {
    use super::*;

    /// Compute factorial with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn factorial_mp(n: u32, ctx: &PrecisionContext) -> MpFloat {
        if n == 0 || n == 1 {
            return ctx.float(1.0);
        }
        let mut result = ctx.float(1.0);
        for i in 2..=n {
            result *= i as f64;
        }
        result
    }

    /// Compute binomial coefficient with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn binomial_mp(n: u32, k: u32, ctx: &PrecisionContext) -> MpFloat {
        if k > n {
            return ctx.float(0.0);
        }
        if k == 0 || k == n {
            return ctx.float(1.0);
        }
        let k = k.min(n - k);
        let mut result = ctx.float(1.0);
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result
    }

    /// Compute Pochhammer symbol (rising factorial) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    pub fn pochhammer_mp(x: &MpFloat, n: u32, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        if n == 0 {
            return Ok(ctx.float(1.0));
        }
        let mut result = x.clone();
        for i in 1..n {
            result *= x.clone() + i as f64;
        }
        Ok(result)
    }
}

// Re-export utility functions
pub use utils::*;

// ---------------------------------------------------------------------------
// Arbitrary precision hypergeometric functions
// ---------------------------------------------------------------------------

/// Arbitrary precision hypergeometric functions.
pub mod hypergeometric {
    use super::*;

    /// Compute the Gauss hypergeometric function ₂F₁(a,b;c;z) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn hyp2f1_ap(
        a: f64,
        b: f64,
        c: f64,
        z: f64,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let a_mp = ctx.float(a);
        let b_mp = ctx.float(b);
        let c_mp = ctx.float(c);
        let z_mp = ctx.float(z);
        hyp2f1_mp(&a_mp, &b_mp, &c_mp, &z_mp, ctx)
    }

    /// Compute ₂F₁(a,b;c;z) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn hyp2f1_mp(
        a: &MpFloat,
        b: &MpFloat,
        c: &MpFloat,
        z: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        if c.is_zero() || (c.is_finite() && c < &0.0 && c.is_integer()) {
            return Err(SpecialError::DomainError(
                "c must not be 0 or a negative integer".to_string(),
            ));
        }
        if z.is_zero() {
            return Ok(ctx.float(1.0));
        }

        let mut sum = ctx.float(1.0);
        let mut term = ctx.float(1.0);
        let tol = mpf(prec, 10.0).pow_i32(-(prec as i32) / 3);

        for n in 1..500 {
            let n_f = ctx.float(n as f64);
            let n_minus_1 = ctx.float((n - 1) as f64);

            let numerator = (a.clone() + &n_minus_1) * (b.clone() + &n_minus_1);
            let denominator = (c.clone() + &n_minus_1) * &n_f;
            term *= numerator / denominator;
            term *= z;
            sum += &term;

            if term.clone().abs() < sum.clone().abs() * &tol {
                break;
            }
        }
        Ok(sum)
    }

    /// Compute the confluent hypergeometric function ₁F₁(a;b;z) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn hyp1f1_ap(a: f64, b: f64, z: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let a_mp = ctx.float(a);
        let b_mp = ctx.float(b);
        let z_mp = ctx.float(z);
        hyp1f1_mp(&a_mp, &b_mp, &z_mp, ctx)
    }

    /// Compute ₁F₁(a;b;z) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn hyp1f1_mp(
        a: &MpFloat,
        b: &MpFloat,
        z: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        if b.is_zero() || (b.is_finite() && b < &0.0 && b.is_integer()) {
            return Err(SpecialError::DomainError(
                "b must not be 0 or a negative integer".to_string(),
            ));
        }
        if z.is_zero() {
            return Ok(ctx.float(1.0));
        }

        if z < &-20.0 {
            let exp_z = z.clone().exp();
            let b_minus_a = b.clone() - a;
            let neg_z = -z.clone();
            let transformed = hyp1f1_mp(&b_minus_a, b, &neg_z, ctx)?;
            return Ok(exp_z * transformed);
        }

        let mut sum = ctx.float(1.0);
        let mut term = ctx.float(1.0);
        let tol = mpf(prec, 10.0).pow_i32(-(prec as i32) / 3);

        for n in 1..500 {
            let n_f = ctx.float(n as f64);
            let n_minus_1 = ctx.float((n - 1) as f64);
            let a_plus_n = a.clone() + &n_minus_1;
            let b_plus_n = b.clone() + &n_minus_1;
            let factor = a_plus_n / (b_plus_n * &n_f);
            term *= factor * z;
            sum += &term;

            if term.clone().abs() < sum.clone().abs() * &tol {
                break;
            }
        }
        Ok(sum)
    }
}

// ---------------------------------------------------------------------------
// Arbitrary precision incomplete gamma functions
// ---------------------------------------------------------------------------

/// Arbitrary precision incomplete gamma functions.
pub mod incomplete_gamma {
    use super::*;

    /// Compute the lower incomplete gamma function γ(a,x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn gammainc_lower_ap(a: f64, x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let a_mp = ctx.float(a);
        let x_mp = ctx.float(x);
        gammainc_lower_mp(&a_mp, &x_mp, ctx)
    }

    /// Compute γ(a,x) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn gammainc_lower_mp(
        a: &MpFloat,
        x: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        if x < &0.0 {
            return Err(SpecialError::DomainError(
                "x must be non-negative for lower incomplete gamma".to_string(),
            ));
        }
        if x.is_zero() {
            return Ok(ctx.float(0.0));
        }

        let x_pow_a = x.clone().pow_float(a);
        let neg_x = -x.clone();
        let exp_neg_x = neg_x.exp();

        let mut sum = ctx.float(0.0);
        let mut term = ctx.float(1.0) / a;
        let tol = mpf(prec, 10.0).pow_i32(-(prec as i32) / 3);
        sum += &term;

        for n in 1..500 {
            let n_f = ctx.float(n as f64);
            let a_plus_n = a.clone() + &n_f;
            term *= x.clone() / a_plus_n;
            sum += &term;
            if term.clone().abs() < sum.clone().abs() * &tol {
                break;
            }
        }
        Ok(x_pow_a * exp_neg_x * sum)
    }

    /// Compute the upper incomplete gamma function Γ(a,x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn gammainc_upper_ap(a: f64, x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let a_mp = ctx.float(a);
        let x_mp = ctx.float(x);
        gammainc_upper_mp(&a_mp, &x_mp, ctx)
    }

    /// Compute Γ(a,x) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn gammainc_upper_mp(
        a: &MpFloat,
        x: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        if x < &0.0 {
            return Err(SpecialError::DomainError(
                "x must be non-negative for upper incomplete gamma".to_string(),
            ));
        }
        let gamma_a = super::gamma::gamma_mp(a, ctx)?;
        let lower = gammainc_lower_mp(a, x, ctx)?;
        Ok(gamma_a - lower)
    }

    /// Compute the regularized incomplete gamma function P(a,x) = γ(a,x)/Γ(a).
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn gammainc_regularized_ap(
        a: f64,
        x: f64,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let a_mp = ctx.float(a);
        let x_mp = ctx.float(x);
        gammainc_regularized_mp(&a_mp, &x_mp, ctx)
    }

    /// Compute P(a,x) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn gammainc_regularized_mp(
        a: &MpFloat,
        x: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let lower = gammainc_lower_mp(a, x, ctx)?;
        let gamma_a = super::gamma::gamma_mp(a, ctx)?;
        Ok(lower / gamma_a)
    }
}

// ---------------------------------------------------------------------------
// Arbitrary precision digamma (psi) function
// ---------------------------------------------------------------------------

/// Arbitrary precision digamma (psi) function.
pub mod digamma {
    use super::*;

    /// Compute the digamma function ψ(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn digamma_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        digamma_mp(&x_mp, ctx)
    }

    /// Compute ψ(x) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn digamma_mp(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        if x.is_zero() || (x.is_finite() && x < &0.0 && x.is_integer()) {
            return Err(SpecialError::DomainError(
                "Digamma undefined at non-positive integers".to_string(),
            ));
        }

        // For x < 1, use reflection formula: ψ(1-x) - ψ(x) = π cot(πx)
        if x < &1.0 {
            let pi = ctx.pi();
            let pi_x = pi.clone() * x;
            let cot_pi_x = pi_x.clone().cos() / pi_x.sin();
            let one_minus_x = ctx.float(1.0) - x;
            let psi_oneminus_x = digamma_mp(&one_minus_x, ctx)?;
            return Ok(psi_oneminus_x - pi * cot_pi_x);
        }

        // Use recurrence to bring x > 8, then asymptotic expansion.
        let mut result = ctx.float(0.0);
        let mut curr_x = x.clone();

        while curr_x < 8.0 {
            result -= ctx.float(1.0) / &curr_x;
            curr_x += 1.0;
        }

        // Asymptotic expansion: ψ(x) ~ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - ...
        let ln_x = curr_x.clone().ln();
        let x2 = curr_x.clone() * &curr_x;
        let x4 = x2.clone() * &x2;
        let x6 = x4.clone() * &x2;

        let asymp = ln_x
            - ctx.float(1.0) / (mpf(prec, 2.0) * &curr_x)
            - ctx.float(1.0) / (mpf(prec, 12.0) * &x2)
            + ctx.float(1.0) / (mpf(prec, 120.0) * &x4)
            - ctx.float(1.0) / (mpf(prec, 252.0) * &x6);

        Ok(result + asymp)
    }

    /// Compute the trigamma function ψ¹(x) = d/dx ψ(x) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn trigamma_ap(x: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        trigamma_mp(&x_mp, ctx)
    }

    /// Compute ψ¹(x) for a `MpFloat` input.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn trigamma_mp(x: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let prec = ctx.precision;
        if x.is_zero() || (x.is_finite() && x < &0.0 && x.is_integer()) {
            return Err(SpecialError::DomainError(
                "Trigamma undefined at non-positive integers".to_string(),
            ));
        }

        if x < &1.0 {
            let pi = ctx.pi();
            let pi_x = pi.clone() * x;
            let sin_pi_x = pi_x.sin();
            let csc_sq = ctx.float(1.0) / (sin_pi_x.clone() * &sin_pi_x);

            let one_minus_x = ctx.float(1.0) - x;
            let psi1_oneminus_x = trigamma_mp(&one_minus_x, ctx)?;
            return Ok(pi.clone() * &pi * csc_sq - psi1_oneminus_x);
        }

        let mut result = ctx.float(0.0);
        let mut curr_x = x.clone();

        while curr_x < 8.0 {
            let one_over_x2 = ctx.float(1.0) / (curr_x.clone() * &curr_x);
            result += one_over_x2;
            curr_x += 1.0;
        }

        let x2 = curr_x.clone() * &curr_x;
        let x3 = x2.clone() * &curr_x;
        let x4 = x2.clone() * &x2;
        let x5 = x4.clone() * &curr_x;

        let asymp = ctx.float(1.0) / &curr_x
            + ctx.float(1.0) / (mpf(prec, 2.0) * &x2)
            + ctx.float(1.0) / (mpf(prec, 6.0) * &x3)
            - ctx.float(1.0) / (mpf(prec, 30.0) * &x5);

        Ok(result + asymp)
    }
}

// ---------------------------------------------------------------------------
// Arbitrary precision beta function
// ---------------------------------------------------------------------------

/// Arbitrary precision beta function.
pub mod beta {
    use super::*;

    /// Compute the beta function B(a,b) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn beta_ap(a: f64, b: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let a_mp = ctx.float(a);
        let b_mp = ctx.float(b);
        beta_mp(&a_mp, &b_mp, ctx)
    }

    /// Compute B(a,b) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn beta_mp(a: &MpFloat, b: &MpFloat, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let gamma_a = super::gamma::gamma_mp(a, ctx)?;
        let gamma_b = super::gamma::gamma_mp(b, ctx)?;
        let a_plus_b = a.clone() + b;
        let gamma_aplusb = super::gamma::gamma_mp(&a_plus_b, ctx)?;
        Ok(gamma_a * gamma_b / gamma_aplusb)
    }

    /// Compute the incomplete beta function B(x; a,b) with arbitrary precision.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn betainc_ap(x: f64, a: f64, b: f64, ctx: &PrecisionContext) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        let a_mp = ctx.float(a);
        let b_mp = ctx.float(b);
        betainc_mp(&x_mp, &a_mp, &b_mp, ctx)
    }

    /// Compute B(x; a,b) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn betainc_mp(
        x: &MpFloat,
        a: &MpFloat,
        b: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        if !(ctx.float(0.0)..=ctx.float(1.0)).contains(x) {
            return Err(SpecialError::DomainError(
                "x must be in [0, 1] for incomplete beta".to_string(),
            ));
        }
        if x.is_zero() {
            return Ok(ctx.float(0.0));
        }
        if (x.clone() - 1.0).is_zero() {
            return beta_mp(a, b, ctx);
        }

        let x_pow_a = x.clone().pow_float(a);
        let hyp =
            super::hypergeometric::hyp2f1_mp(a, &(ctx.float(1.0) - b), &(a.clone() + 1.0), x, ctx)?;
        Ok(x_pow_a * hyp / a)
    }

    /// Compute the regularized incomplete beta function I_x(a,b) = B(x;a,b)/B(a,b).
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn betainc_regularized_ap(
        x: f64,
        a: f64,
        b: f64,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let x_mp = ctx.float(x);
        let a_mp = ctx.float(a);
        let b_mp = ctx.float(b);
        betainc_regularized_mp(&x_mp, &a_mp, &b_mp, ctx)
    }

    /// Compute I_x(a,b) for `MpFloat` inputs.
    #[cfg(feature = "high-precision")]
    #[allow(dead_code)]
    pub fn betainc_regularized_mp(
        x: &MpFloat,
        a: &MpFloat,
        b: &MpFloat,
        ctx: &PrecisionContext,
    ) -> SpecialResult<MpFloat> {
        let inc = betainc_mp(x, a, b, ctx)?;
        let full = beta_mp(a, b, ctx)?;
        Ok(inc / full)
    }
}

// ---------------------------------------------------------------------------
// Simple Float → f64 converter (mirrors rug::Float::to_f64)
// ---------------------------------------------------------------------------

/// Convert a `MpFloat` to `f64` (may lose precision).
#[cfg(feature = "high-precision")]
#[allow(dead_code)]
pub fn to_f64(x: &MpFloat) -> f64 {
    x.to_f64()
}

/// Convert a `MpComplex` to `scirs2_core::numeric::Complex64`.
#[cfg(feature = "high-precision")]
#[allow(dead_code)]
pub fn to_complex64(z: &MpComplex) -> scirs2_core::numeric::Complex64 {
    let re = z.real().to_f64();
    let im = z.imag().to_f64();
    scirs2_core::numeric::Complex64::new(re, im)
}

/// No-op cache cleanup (MPFR compatibility shim).
#[allow(dead_code)]
pub fn cleanup_cache() {
    #[cfg(feature = "high-precision")]
    oxinum_float::free_cache();
}

// ---------------------------------------------------------------------------
// MPFR-native API: gamma_mpfr, erf_mpfr, bessel_j0_mpfr, bessel_k0_mpfr
// These are thin wrappers that call the oxinum-float special functions and
// return `MpFloat` (previously returned `rug::Float`).
// ---------------------------------------------------------------------------

/// Compute the gamma function Γ(x) at arbitrary precision.
///
/// # Arguments
/// * `x` - The argument as a `MpFloat`.
/// * `precision_bits` - Precision in bits (53 = f64 precision, 500+ for extended).
///
/// # Returns
/// Γ(x) as a `MpFloat` at the requested precision.
#[cfg(feature = "high-precision")]
pub fn gamma_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let mut result = MpFloat::with_val_from(precision_bits, x);
    result.gamma_mut();
    result
}

/// Compute the log-gamma function ln(Γ(x)) at arbitrary precision.
#[cfg(feature = "high-precision")]
pub fn lgamma_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let mut result = MpFloat::with_val_from(precision_bits, x);
    result.ln_gamma_mut();
    result
}

/// Compute the digamma function ψ(x) = d/dx ln(Γ(x)) at arbitrary precision.
#[cfg(feature = "high-precision")]
pub fn digamma_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let mut result = MpFloat::with_val_from(precision_bits, x);
    result.digamma_mut();
    result
}

/// Compute the error function erf(x) at arbitrary precision.
#[cfg(feature = "high-precision")]
pub fn erf_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let mut result = MpFloat::with_val_from(precision_bits, x);
    result.erf_mut();
    result
}

/// Compute the complementary error function erfc(x) at arbitrary precision.
#[cfg(feature = "high-precision")]
pub fn erfc_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let mut result = MpFloat::with_val_from(precision_bits, x);
    result.erfc_mut();
    result
}

/// Compute the Bessel function J₀(x) at arbitrary precision.
#[cfg(feature = "high-precision")]
pub fn bessel_j0_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let mut result = MpFloat::with_val_from(precision_bits, x);
    result.j0_mut();
    result
}

/// Compute the modified Bessel function K₀(x) at arbitrary precision.
///
/// K₀(x) = −(ln(x/2) + γ) I₀(x) + Σ_{k=1}^∞ H_k (x/2)^{2k} / (k!)²
///
/// where γ is the Euler-Mascheroni constant, H_k = 1 + 1/2 + ... + 1/k,
/// and I₀ is the modified Bessel function of the first kind.
#[cfg(feature = "high-precision")]
pub fn bessel_k0_mpfr(x: &MpFloat, precision_bits: u32) -> MpFloat {
    let prec = precision_bits;

    let half_x = MpFloat::with_val_from(prec, x) * 0.5_f64;
    let half_x_sq = half_x.clone() * &half_x;

    // Euler-Mascheroni constant γ
    let gamma_em = oxinum_float::mp_float::euler_gamma_at_bits(prec);
    let ln_half_x = half_x.clone().ln();
    let log_term = ln_half_x + &gamma_em;

    // I₀(x) = Σ_{k=0}^∞ (x/2)^{2k} / (k!)²
    let i0 = bessel_i0_mpfr_internal(x, prec);

    // Second sum: Σ_{k=1}^∞ H_k * (x/2)^{2k} / (k!)²
    let mut second_sum = MpFloat::with_val(prec, 0.0_f64);
    let mut factorial_sq = MpFloat::with_val(prec, 1.0_f64);
    let mut power = MpFloat::with_val(prec, 1.0_f64);
    let mut harmonic = MpFloat::with_val(prec, 0.0_f64);

    let n_terms = ((prec as usize) / 3).max(80);
    for k in 1..=n_terms {
        let k_f = k as f64;
        power *= &half_x_sq;
        factorial_sq *= k_f * k_f;
        harmonic += 1.0_f64 / k_f;

        let term = harmonic.clone() * &power / &factorial_sq;
        let term_abs = term.clone().abs();
        let second_abs = second_sum.clone().abs();

        second_sum += &term;

        if !second_abs.is_zero() {
            let ratio = term_abs / &second_abs;
            let threshold = MpFloat::with_val(prec, 2.0_f64).pow_i32(-((prec - 4) as i32));
            if ratio < threshold {
                break;
            }
        }
    }

    // K₀(x) = −(ln(x/2) + γ) I₀(x) + second_sum
    -(i0 * log_term) + second_sum
}

/// Internal helper: compute I₀(x) = Σ_{k=0}^∞ (x/2)^{2k} / (k!)²
#[cfg(feature = "high-precision")]
fn bessel_i0_mpfr_internal(x: &MpFloat, prec: u32) -> MpFloat {
    let half_x = MpFloat::with_val_from(prec, x) * 0.5_f64;
    let half_x_sq = half_x.clone() * &half_x;

    let mut sum = MpFloat::with_val(prec, 1.0_f64);
    let mut term = MpFloat::with_val(prec, 1.0_f64);

    let n_terms = ((prec as usize) / 3).max(80);
    for k in 1..=n_terms {
        let k_f = k as f64;
        term *= &half_x_sq;
        term /= k_f * k_f;

        let term_abs = term.clone().abs();
        let sum_abs = sum.clone().abs();
        sum += &term;

        if !sum_abs.is_zero() {
            let ratio = term_abs / &sum_abs;
            let threshold = MpFloat::with_val(prec, 2.0_f64).pow_i32(-((prec - 4) as i32));
            if ratio < threshold {
                break;
            }
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_precision_context() {
        let ctx = PrecisionContext::new(512).expect("Operation failed");
        assert_eq!(ctx.precision(), 512);
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_precision_context_pi() {
        let ctx = PrecisionContext::new(512).expect("Operation failed");
        let pi = ctx.pi();
        // Check it starts with the right digits
        let pi_str = pi.to_string();
        assert!(pi_str.starts_with("3.14159"), "pi = {pi_str}");
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_gamma_ap() {
        let ctx = PrecisionContext::default();

        // Test Γ(1) = 1
        let gamma_1 = gamma::gamma_ap(1.0, &ctx).expect("Operation failed");
        assert_relative_eq!(to_f64(&gamma_1), 1.0, epsilon = 1e-10);

        // Test Γ(0.5) = √π
        let gamma_half = gamma::gamma_ap(0.5, &ctx).expect("Operation failed");
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert_relative_eq!(to_f64(&gamma_half), sqrt_pi, epsilon = 1e-10);

        // Test Γ(5) = 4! = 24
        let gamma_5 = gamma::gamma_ap(5.0, &ctx).expect("Operation failed");
        assert_relative_eq!(to_f64(&gamma_5), 24.0, epsilon = 1e-8);
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_bessel_ap() {
        let ctx = PrecisionContext::default();

        // Test J_0(0) = 1
        let j0_0 = bessel::bessel_j_ap(0, 0.0, &ctx).expect("Operation failed");
        assert_relative_eq!(to_f64(&j0_0), 1.0, epsilon = 1e-15);

        // Test J_1(0) = 0
        let j1_0 = bessel::bessel_j_ap(1, 0.0, &ctx).expect("Operation failed");
        assert_relative_eq!(to_f64(&j1_0), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_erf_ap() {
        let ctx = PrecisionContext::default();

        // Test erf(0) = 0
        let erf_0 = error_function::erf_ap(0.0, &ctx).expect("Operation failed");
        assert_relative_eq!(to_f64(&erf_0), 0.0, epsilon = 1e-15);

        // Test erfc(0) = 1
        let erfc_0 = error_function::erfc_ap(0.0, &ctx).expect("Operation failed");
        assert_relative_eq!(to_f64(&erfc_0), 1.0, epsilon = 1e-15);

        // Test erf(x) + erfc(x) = 1
        let x = 1.5;
        let erf_x = error_function::erf_ap(x, &ctx).expect("Operation failed");
        let erfc_x = error_function::erfc_ap(x, &ctx).expect("Operation failed");
        let sum = to_f64(&erf_x) + to_f64(&erfc_x);
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_high_precision_pi() {
        // Test with 512-bit precision
        let ctx = PrecisionContext::new(512).expect("Operation failed");
        let pi = ctx.pi();
        let pi_str = format!("{}", pi);
        assert!(pi_str.starts_with("3.14159265358979"), "pi = {pi_str}");
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_euler_gamma_constant() {
        let ctx = PrecisionContext::default();
        let eg = ctx.euler_gamma();
        let v = eg.to_f64();
        assert!((v - 0.5772156649_f64).abs() < 1e-8, "euler_gamma = {v}");
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_catalan_constant() {
        let ctx = PrecisionContext::default();
        let cat = ctx.catalan();
        let v = cat.to_f64();
        assert!((v - 0.9159655942_f64).abs() < 1e-8, "catalan = {v}");
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_gamma_mpfr() {
        let x = MpFloat::with_val(256, 5.0);
        let g = gamma_mpfr(&x, 256);
        assert_relative_eq!(g.to_f64(), 24.0, epsilon = 1e-8);
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_erf_mpfr() {
        let x = MpFloat::with_val(256, 0.0);
        let e = erf_mpfr(&x, 256);
        assert!(e.is_zero() || e.to_f64().abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "high-precision")]
    fn test_bessel_j0_mpfr() {
        let x = MpFloat::with_val(256, 0.0);
        let j = bessel_j0_mpfr(&x, 256);
        assert_relative_eq!(j.to_f64(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cleanup_cache_noop() {
        // Should not panic
        cleanup_cache();
    }
}
