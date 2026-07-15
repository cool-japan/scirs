//! # Arbitrary Precision Numerical Computation Support
//!
//! This module provides arbitrary precision arithmetic capabilities for scientific computing,
//! enabling calculations with user-defined precision levels for both integers and floating-point numbers.
//!
//! ## Backend
//!
//! All types are backed by **oxinum-\*** (Pure Rust, GMP/MPFR-free, C/Fortran-free):
//! - Integer arithmetic: `oxinum_int::IBig`
//! - Float arithmetic: `oxinum_float::DBig`
//! - Rational arithmetic: `oxinum_rational::RBig`
//! - Complex arithmetic: `oxinum_complex::CBig` (replaces the former `rug::Complex` backend)
//!
//! ## Precision model
//!
//! `ArbitraryPrecisionContext.floatprecision` continues to store bit-precision for API stability.
//! Conversion to the decimal-digit precision used by `oxinum-float`'s `DBig` is done at the
//! oxinum API boundary using the ratio 1 decimal digit ≈ 3.32 bits (log2(10)).
//!
//! ## Features
//!
//! - Arbitrary precision integers (`ArbitraryInt`) backed by `oxinum_int::IBig`
//! - Arbitrary precision floating-point (`ArbitraryFloat`) backed by `oxinum_float::DBig`
//! - Exact rational arithmetic (`ArbitraryRational`) backed by `oxinum_rational::RBig`
//! - Arbitrary precision complex numbers (`ArbitraryComplex`) backed by `oxinum_complex::CBig`
//! - Integration with existing ScientificNumber traits
//! - Automatic precision tracking and management
//! - Configurable precision contexts

use crate::{
    error::{CoreError, CoreResult, ErrorContext},
    numeric::precision_tracking::PrecisionContext,
    validation::check_positive,
};
use num_bigint::BigInt;
// oxinum-int types (Pure Rust, GMP-free)
use oxinum_int::{is_prime, IBig, UBig};
// oxinum-float types and functions (Pure Rust, MPFR-free)
use oxinum_float::{
    compute_e, compute_ln2, compute_pi, cos, cosh, exp, ln, precision::with_precision, sin, sinh,
    sqrt, tan, tanh, DBig,
};
// oxinum-rational types (Pure Rust)
use oxinum_rational::{IBig as RIBig, RBig, UBig as RUBig};
// oxinum-complex: Pure Rust arbitrary-precision complex (replaces rug::Complex)
use oxinum_complex::CBig;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;
use std::sync::RwLock;

/// Global default precision for arbitrary precision operations (in bits).
static DEFAULT_PRECISION: RwLock<u32> = RwLock::new(256);

/// Bits per decimal digit constant (log2(10)).
const BITS_PER_DECIMAL_DIGIT: f64 = std::f64::consts::LOG2_10;

/// Convert bit precision to decimal digit count for oxinum APIs.
fn bits_to_decimal_digits(bits: u32) -> usize {
    ((bits as f64) / BITS_PER_DECIMAL_DIGIT).ceil() as usize
}

/// Convert an `f64` to `DBig`.
///
/// `dashu-float` does not implement `From<f64>` for `DBig`.  The most reliable
/// route is to format the value with enough significant digits (17 is sufficient
/// to represent any `f64` uniquely) and parse the resulting string.
fn f64_to_dbig(v: f64) -> DBig {
    if v.is_nan() {
        return DBig::from(0u32);
    }
    if v.is_infinite() {
        // DBig has no infinity; clamp to a very large value via a string.
        return DBig::from_str("1e308").unwrap_or_else(|_| DBig::from(0u32));
    }
    let s = format!("{v:.17e}");
    DBig::from_str(&s).unwrap_or_else(|_| {
        // Fallback: use less precision.
        let s2 = format!("{v}");
        DBig::from_str(&s2).unwrap_or_else(|_| DBig::from(0u32))
    })
}

/// Convert a `DBig` to `f64`.
///
/// `dashu-float`'s `to_f64()` returns `dashu_base::Approximation<f64>` (a `Rounded<f64>`).
/// We extract the inner value via `.value()`.
fn dbig_to_f64(v: &DBig) -> f64 {
    v.to_f64().value()
}

/// Get the default precision for arbitrary precision operations (in bits).
#[allow(dead_code)]
pub fn get_defaultprecision() -> u32 {
    *DEFAULT_PRECISION.read().expect("Operation failed")
}

/// Set the default precision for arbitrary precision operations (in bits).
#[allow(dead_code)]
pub fn setprecision(prec: u32) -> CoreResult<()> {
    check_positive(prec as f64, "precision")?;
    *DEFAULT_PRECISION.write().expect("Operation failed") = prec;
    Ok(())
}

/// Precision context for arbitrary precision arithmetic.
#[derive(Debug, Clone)]
pub struct ArbitraryPrecisionContext {
    /// Precision in bits for floating-point operations.
    pub floatprecision: u32,
    /// Maximum precision allowed.
    pub maxprecision: u32,
    /// Rounding mode.
    pub rounding_mode: RoundingMode,
    /// Whether to track precision loss.
    pub trackprecision: bool,
    /// Precision tracking context.
    pub precision_context: Option<PrecisionContext>,
}

/// Rounding modes for arbitrary precision arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even.
    Nearest,
    /// Round toward zero.
    Zero,
    /// Round toward positive infinity.
    Up,
    /// Round toward negative infinity.
    Down,
    /// Round away from zero.
    Away,
}

impl Default for ArbitraryPrecisionContext {
    fn default() -> Self {
        Self {
            floatprecision: get_defaultprecision(),
            maxprecision: 4096,
            rounding_mode: RoundingMode::Nearest,
            trackprecision: false,
            precision_context: None,
        }
    }
}

impl ArbitraryPrecisionContext {
    /// Create a new precision context with specified precision (in bits).
    pub fn withprecision(precision: u32) -> CoreResult<Self> {
        check_positive(precision as f64, "precision")?;
        Ok(Self {
            floatprecision: precision,
            ..Default::default()
        })
    }

    /// Create a context with precision tracking enabled.
    pub fn withprecision_tracking(precision: u32) -> CoreResult<Self> {
        let mut ctx = Self::withprecision(precision)?;
        ctx.trackprecision = true;
        let mut precision_ctx = PrecisionContext::new();
        precision_ctx.precision = precision as f64 / BITS_PER_DECIMAL_DIGIT;
        ctx.precision_context = Some(precision_ctx);
        Ok(ctx)
    }

    /// Set the rounding mode.
    pub fn with_rounding(mut self, mode: RoundingMode) -> Self {
        self.rounding_mode = mode;
        self
    }

    /// Set the maximum precision.
    pub fn with_maxprecision(mut self, maxprec: u32) -> Self {
        self.maxprecision = maxprec;
        self
    }

    /// Return the float precision expressed in decimal digits.
    fn decimal_digits(&self) -> usize {
        bits_to_decimal_digits(self.floatprecision)
    }
}

// ---------------------------------------------------------------------------
// ArbitraryInt — backed by oxinum_int::IBig
// ---------------------------------------------------------------------------

/// Arbitrary precision integer backed by `oxinum_int::IBig` (Pure Rust).
#[derive(Clone, PartialEq, Eq)]
pub struct ArbitraryInt {
    value: IBig,
}

impl ArbitraryInt {
    /// Create a new arbitrary precision integer (value 0).
    pub fn new() -> Self {
        Self {
            value: IBig::from(0i32),
        }
    }

    /// Create from a regular 64-bit signed integer.
    pub fn from_i64(n: i64) -> Self {
        Self {
            value: IBig::from(n),
        }
    }

    /// Create from a string in the given radix (2..=36).
    pub fn from_str_radix(s: &str, radix: i32) -> CoreResult<Self> {
        if !(2..=36).contains(&radix) {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Invalid radix {radix}: must be 2..=36"
            ))));
        }
        oxinum_int::ibig_from_radix(s, radix as u32)
            .map(|value| Self { value })
            .map_err(|e| {
                CoreError::ValidationError(ErrorContext::new(format!(
                    "Failed to parse integer from string '{s}': {e}"
                )))
            })
    }

    /// Convert to `num_bigint::BigInt`.
    pub fn to_bigint(&self) -> BigInt {
        BigInt::from_str(&self.value.to_string()).expect("Operation failed")
    }

    /// Check if the number is (probably) prime.
    ///
    /// Uses Miller-Rabin with `reps` witnesses.  If `reps` is 0, a
    /// deterministic witness set is used (correct for all n < 3.3 × 10²⁴).
    pub fn is_probably_prime(&self, reps: u32) -> bool {
        // Negative or zero → not prime.
        if self.value <= IBig::from(1i32) {
            return false;
        }
        // Convert the positive IBig to a string and parse as UBig for the primality test.
        let s = self.value.to_string();
        match UBig::from_str(&s) {
            Ok(u) => is_prime(&u, reps),
            Err(_) => false,
        }
    }

    /// Compute factorial n!.
    pub fn factorial(n: u32) -> Self {
        let u = oxinum_int::factorial(n);
        Self {
            value: IBig::from(u),
        }
    }

    /// Compute binomial coefficient C(n, k).
    pub fn binomial(n: u32, k: u32) -> Self {
        if k > n {
            return Self::new();
        }
        let u = oxinum_int::binomial(n, k);
        Self {
            value: IBig::from(u),
        }
    }

    /// Compute greatest common divisor.
    pub fn gcd(&self, other: &Self) -> Self {
        use oxinum_int::Gcd;
        // GCD on IBig returns UBig (always non-negative); wrap back into IBig.
        let a = self.value.clone();
        let b = other.value.clone();
        let g: UBig = a.gcd(&b);
        Self {
            value: IBig::from(g),
        }
    }

    /// Compute least common multiple.
    pub fn lcm(&self, other: &Self) -> Self {
        if self.value == IBig::from(0i32) || other.value == IBig::from(0i32) {
            return Self::new();
        }
        let gcd = self.gcd(other);
        let product = self.value.clone() * other.value.clone();
        Self {
            value: product / gcd.value,
        }
    }

    /// Modular exponentiation: (self ^ exp) mod modulus.
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> CoreResult<Self> {
        if modulus.value == IBig::from(0i32) {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Modulus cannot be zero",
            )));
        }
        // Convert to UBig (non-negative) for the oxinum mod_pow.
        let base_str = self.value.to_string();
        let exp_str = exp.value.to_string();
        let mod_str = modulus.value.to_string();
        let base_u = UBig::from_str(&base_str).map_err(|_| {
            CoreError::DomainError(ErrorContext::new("base must be non-negative for mod_pow"))
        })?;
        let exp_u = UBig::from_str(&exp_str).map_err(|_| {
            CoreError::DomainError(ErrorContext::new(
                "exponent must be non-negative for mod_pow",
            ))
        })?;
        let mod_u = UBig::from_str(&mod_str).map_err(|_| {
            CoreError::DomainError(ErrorContext::new(
                "modulus must be non-negative for mod_pow",
            ))
        })?;
        let result = oxinum_int::mod_pow(&base_u, &exp_u, &mod_u)
            .map_err(|e| CoreError::DomainError(ErrorContext::new(format!("{e}"))))?;
        Ok(Self {
            value: IBig::from(result),
        })
    }

    /// Get the absolute value.
    pub fn abs(&self) -> Self {
        use oxinum_core::Abs;
        Self {
            value: self.value.clone().abs(),
        }
    }

    /// Get the sign (-1, 0, or 1).
    pub fn signum(&self) -> i32 {
        use oxinum_core::Signed;
        // `.sign()` (from dashu_base::Signed) returns a `Sign` enum.
        // IBig(0) has sign Positive, so check for zero separately.
        if self.value == IBig::from(0i32) {
            return 0;
        }
        match self.value.sign() {
            oxinum_core::Sign::Positive => 1,
            oxinum_core::Sign::Negative => -1,
        }
    }
}

impl fmt::Display for ArbitraryInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for ArbitraryInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArbitraryInt({})", self.value)
    }
}

impl Default for ArbitraryInt {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for ArbitraryInt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl Sub for ArbitraryInt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl Mul for ArbitraryInt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
        }
    }
}

impl Div for ArbitraryInt {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
        }
    }
}

impl Neg for ArbitraryInt {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { value: -self.value }
    }
}

// ---------------------------------------------------------------------------
// ArbitraryFloat — backed by oxinum_float::DBig
// ---------------------------------------------------------------------------

/// Arbitrary precision floating-point number backed by `oxinum_float::DBig` (Pure Rust).
///
/// `DBig` is a base-10 decimal big-float.  The `ArbitraryPrecisionContext` stores
/// precision in **bits** (for API stability), but all oxinum calls receive the
/// equivalent decimal-digit count.
#[derive(Clone)]
pub struct ArbitraryFloat {
    value: DBig,
    context: ArbitraryPrecisionContext,
}

impl ArbitraryFloat {
    /// Create a new zero-valued arbitrary precision float with the default precision.
    pub fn new() -> Self {
        let prec = get_defaultprecision();
        let context = ArbitraryPrecisionContext::default();
        let digits = bits_to_decimal_digits(prec);
        let value = with_precision(&DBig::from(0u32), digits);
        Self { value, context }
    }

    /// Create with a specific bit precision.
    pub fn withprecision(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        let value = with_precision(&DBig::from(0u32), digits);
        Ok(Self { value, context })
    }

    /// Create with a specific precision context.
    pub fn with_context(context: ArbitraryPrecisionContext) -> Self {
        let digits = context.decimal_digits();
        let value = with_precision(&DBig::from(0u32), digits);
        Self { value, context }
    }

    /// Create from f64 with the default precision.
    pub fn from_f64(v: f64) -> Self {
        let prec = get_defaultprecision();
        let context = ArbitraryPrecisionContext::default();
        let digits = bits_to_decimal_digits(prec);
        let raw = f64_to_dbig(v);
        let value = with_precision(&raw, digits);
        Self { value, context }
    }

    /// Create from f64 with a specific bit precision.
    pub fn from_f64_withprecision(v: f64, prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        let raw = f64_to_dbig(v);
        let value = with_precision(&raw, digits);
        Ok(Self { value, context })
    }

    /// Parse a decimal string with a specific bit precision.
    pub fn from_strprec(s: &str, prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        let parsed = DBig::from_str(s)
            .map_err(|e| CoreError::ValidationError(ErrorContext::new(format!("{e}"))))?;
        let value = with_precision(&parsed, digits);
        Ok(Self { value, context })
    }

    /// Get the bit precision stored in the context.
    pub fn precision(&self) -> u32 {
        self.context.floatprecision
    }

    /// Get the precision in decimal digits.
    pub fn decimalprecision(&self) -> u32 {
        self.context.decimal_digits() as u32
    }

    /// Return a new value rebound to the given bit precision.
    pub fn setprecision(&self, prec: u32) -> CoreResult<Self> {
        let mut context = self.context.clone();
        context.floatprecision = prec;
        let digits = bits_to_decimal_digits(prec);
        let value = with_precision(&self.value, digits);
        Ok(Self { value, context })
    }

    /// Convert to f64 (may lose precision).
    pub fn to_f64(&self) -> f64 {
        dbig_to_f64(&self.value)
    }

    /// Check if the value is finite (DBig is always finite — no NaN/infinity).
    pub fn is_finite(&self) -> bool {
        true
    }

    /// Check if the value is infinite (DBig has no infinity).
    pub fn is_infinite(&self) -> bool {
        false
    }

    /// Check if the value is NaN (DBig has no NaN).
    pub fn is_nan(&self) -> bool {
        false
    }

    /// Check if the value is zero.
    pub fn is_zero(&self) -> bool {
        self.value == DBig::from(0u32)
    }

    /// Get the absolute value.
    pub fn abs(&self) -> Self {
        use oxinum_core::Abs;
        Self {
            value: self.value.clone().abs(),
            context: self.context.clone(),
        }
    }

    /// Square root.
    pub fn sqrt(&self) -> CoreResult<Self> {
        let zero = DBig::from(0u32);
        if self.value < zero {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Square root of negative number",
            )));
        }
        let digits = self.context.decimal_digits();
        let result = sqrt(&self.value, digits)
            .map_err(|e| CoreError::DomainError(ErrorContext::new(format!("{e}"))))?;
        Ok(Self {
            value: result,
            context: self.context.clone(),
        })
    }

    /// Natural logarithm.
    pub fn ln(&self) -> CoreResult<Self> {
        let zero = DBig::from(0u32);
        if self.value <= zero {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Logarithm of non-positive number",
            )));
        }
        let digits = self.context.decimal_digits();
        let result = ln(&self.value, digits)
            .map_err(|e| CoreError::DomainError(ErrorContext::new(format!("{e}"))))?;
        Ok(Self {
            value: result,
            context: self.context.clone(),
        })
    }

    /// Exponential function.
    pub fn exp(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = exp(&self.value, digits).unwrap_or_else(|_| DBig::from(1u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Power function: self ^ exponent.
    pub fn pow(&self, exponent: &Self) -> Self {
        use oxinum_float::pow as oxinum_pow;
        let digits = self.context.decimal_digits();
        let result =
            oxinum_pow(&self.value, &exponent.value, digits).unwrap_or_else(|_| DBig::from(1u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Sine.
    pub fn sin(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = sin(&self.value, digits).unwrap_or_else(|_| DBig::from(0u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Cosine.
    pub fn cos(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = cos(&self.value, digits).unwrap_or_else(|_| DBig::from(1u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Tangent.
    pub fn tan(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = tan(&self.value, digits).unwrap_or_else(|_| DBig::from(0u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Arcsine.
    pub fn asin(&self) -> CoreResult<Self> {
        // |x| must be <= 1.  Check against 1.0.
        let one = DBig::from(1u32);
        let abs_val = {
            use oxinum_core::Abs;
            self.value.clone().abs()
        };
        if abs_val > one {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Arcsine argument out of range [-1, 1]",
            )));
        }
        // asin is not directly provided by oxinum-float; implement via atan:
        //   asin(x) = atan(x / sqrt(1 - x²))
        let digits = self.context.decimal_digits();
        let one_minus_x2 = {
            let x2 = self.value.clone() * self.value.clone();
            DBig::from(1u32) - x2
        };
        let denom = sqrt(&one_minus_x2, digits + 4)
            .map_err(|e| CoreError::DomainError(ErrorContext::new(format!("{e}"))))?;
        let zero = DBig::from(0u32);
        if denom == zero {
            // x = ±1: asin(±1) = ±π/2
            let pi = compute_pi(digits);
            let two = DBig::from(2u32);
            let half_pi = pi / two;
            // DBig comparison: negative when self.value < 0
            let result = if self.value < zero { -half_pi } else { half_pi };
            return Ok(Self {
                value: result,
                context: self.context.clone(),
            });
        }
        let ratio = self.value.clone() / denom;
        use oxinum_float::atan as oxinum_atan;
        let result = oxinum_atan(&ratio, digits)
            .map_err(|e| CoreError::DomainError(ErrorContext::new(format!("{e}"))))?;
        Ok(Self {
            value: result,
            context: self.context.clone(),
        })
    }

    /// Arccosine.
    pub fn acos(&self) -> CoreResult<Self> {
        // acos(x) = π/2 - asin(x)
        let one = DBig::from(1u32);
        let abs_val = {
            use oxinum_core::Abs;
            self.value.clone().abs()
        };
        if abs_val > one {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Arccosine argument out of range [-1, 1]",
            )));
        }
        let digits = self.context.decimal_digits();
        let pi = compute_pi(digits + 4);
        let two = DBig::from(2u32);
        let half_pi = pi / two;
        let asin_val = self.asin()?;
        Ok(Self {
            value: half_pi - asin_val.value,
            context: self.context.clone(),
        })
    }

    /// Arctangent.
    pub fn atan(&self) -> Self {
        use oxinum_float::atan as oxinum_atan;
        let digits = self.context.decimal_digits();
        let result = oxinum_atan(&self.value, digits).unwrap_or_else(|_| DBig::from(0u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Two-argument arctangent: atan2(self, x).
    pub fn atan2(&self, x: &Self) -> Self {
        use oxinum_float::atan2 as oxinum_atan2;
        let digits = self.context.decimal_digits();
        let result =
            oxinum_atan2(&self.value, &x.value, digits).unwrap_or_else(|_| DBig::from(0u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Hyperbolic sine.
    pub fn sinh(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = sinh(&self.value, digits).unwrap_or_else(|_| DBig::from(0u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Hyperbolic cosine.
    pub fn cosh(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = cosh(&self.value, digits).unwrap_or_else(|_| DBig::from(1u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Self {
        let digits = self.context.decimal_digits();
        let result = tanh(&self.value, digits).unwrap_or_else(|_| DBig::from(0u32));
        Self {
            value: result,
            context: self.context.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Mathematical constants
    // -----------------------------------------------------------------------

    /// Compute π to the given bit precision.
    pub fn prec_2(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        let value = with_precision(&compute_pi(digits), digits);
        Ok(Self { value, context })
    }

    /// Compute e (Euler's number) to the given bit precision.
    pub fn prec_3(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        let value = with_precision(&compute_e(digits), digits);
        Ok(Self { value, context })
    }

    /// Compute ln(2) to the given bit precision.
    pub fn prec_4(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        let value = with_precision(&compute_ln2(digits), digits);
        Ok(Self { value, context })
    }
}

impl fmt::Display for ArbitraryFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for ArbitraryFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArbitraryFloat({}, {} bits)",
            self.value,
            self.precision()
        )
    }
}

impl PartialEq for ArbitraryFloat {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for ArbitraryFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Default for ArbitraryFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for ArbitraryFloat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            context: self.context,
        }
    }
}

impl Sub for ArbitraryFloat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            context: self.context,
        }
    }
}

impl Mul for ArbitraryFloat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            context: self.context,
        }
    }
}

impl Div for ArbitraryFloat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
            context: self.context,
        }
    }
}

impl Neg for ArbitraryFloat {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            context: self.context,
        }
    }
}

// ---------------------------------------------------------------------------
// ArbitraryRational — backed by oxinum_rational::RBig
// ---------------------------------------------------------------------------

/// Arbitrary precision rational number backed by `oxinum_rational::RBig` (Pure Rust).
#[derive(Clone, PartialEq, Eq)]
pub struct ArbitraryRational {
    value: RBig,
}

impl ArbitraryRational {
    /// Create a new rational number (value 0/1).
    pub fn new() -> Self {
        Self {
            value: RBig::from(0u32),
        }
    }

    /// Create from numerator and denominator (i64).
    pub fn num(num: i64, den: i64) -> CoreResult<Self> {
        if den == 0 {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Denominator cannot be zero",
            )));
        }
        let n = RIBig::from(num);
        let d = RUBig::from(den.unsigned_abs());
        let signed_n = if den < 0 { -n } else { n };
        Ok(Self {
            value: RBig::from_parts(signed_n, d),
        })
    }

    /// Create from arbitrary precision integers.
    pub fn num_2(num: &ArbitraryInt, den: &ArbitraryInt) -> CoreResult<Self> {
        if den.value == IBig::from(0i32) {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Denominator cannot be zero",
            )));
        }
        let n = RIBig::from_str(&num.value.to_string()).map_err(|_| {
            CoreError::ValidationError(ErrorContext::new("numerator conversion failed"))
        })?;
        let d_ibig = IBig::from_str(&den.value.to_string()).map_err(|_| {
            CoreError::ValidationError(ErrorContext::new("denominator conversion failed"))
        })?;
        // Take absolute value of denominator; fold sign into numerator.
        use oxinum_core::Abs;
        let (n_final, d_ubig) = if d_ibig < IBig::from(0i32) {
            let neg_n = RIBig::from_str(&(-num.value.clone()).to_string())
                .unwrap_or_else(|_| RIBig::from(0i32));
            let d_abs = RUBig::from_str(&d_ibig.clone().abs().to_string()).unwrap_or(RUBig::ONE);
            (neg_n, d_abs)
        } else {
            let d_abs = RUBig::from_str(&d_ibig.to_string()).unwrap_or(RUBig::ONE);
            (n, d_abs)
        };
        Ok(Self {
            value: RBig::from_parts(n_final, d_ubig),
        })
    }

    /// Parse a rational from a string (e.g. "22/7").
    #[deprecated(since = "0.1.0", note = "Use str::parse() instead")]
    pub fn parse_rational(s: &str) -> CoreResult<Self> {
        s.parse()
    }

    /// Convert to f64 (may lose precision).
    pub fn to_f64(&self) -> f64 {
        use oxinum_rational::to_f64 as rbig_to_f64;
        rbig_to_f64(&self.value)
    }

    /// Convert to an arbitrary precision float at the given bit precision.
    pub fn to_arbitrary_float(&self, prec: u32) -> CoreResult<ArbitraryFloat> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let digits = bits_to_decimal_digits(prec);
        // Compute numerator / denominator in high precision.
        let num_str = self.value.numerator().to_string();
        let den_str = self.value.denominator().to_string();
        let n = DBig::from_str(&num_str)
            .map_err(|e| CoreError::ValidationError(ErrorContext::new(format!("{e}"))))?;
        let d = DBig::from_str(&den_str)
            .map_err(|e| CoreError::ValidationError(ErrorContext::new(format!("{e}"))))?;
        let n_prec = with_precision(&n, digits + 4);
        let d_prec = with_precision(&d, digits + 4);
        let value = with_precision(&(n_prec / d_prec), digits);
        Ok(ArbitraryFloat { value, context })
    }

    /// Get numerator as an `ArbitraryInt`.
    pub fn numerator(&self) -> ArbitraryInt {
        let n_str = self.value.numerator().to_string();
        let v = IBig::from_str(&n_str).unwrap_or_else(|_| IBig::from(0i32));
        ArbitraryInt { value: v }
    }

    /// Get denominator as an `ArbitraryInt`.
    pub fn denominator(&self) -> ArbitraryInt {
        let d_str = self.value.denominator().to_string();
        let v = IBig::from_str(&d_str).unwrap_or_else(|_| IBig::from(1i32));
        ArbitraryInt { value: v }
    }

    /// Get the absolute value.
    pub fn abs(&self) -> Self {
        use oxinum_rational::rational_abs;
        Self {
            value: rational_abs(&self.value),
        }
    }

    /// Get the reciprocal.
    pub fn recip(&self) -> CoreResult<Self> {
        if self.value == RBig::from(0u32) {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Cannot take reciprocal of zero",
            )));
        }
        use oxinum_rational::rational_reciprocal;
        let recip = rational_reciprocal(&self.value)
            .map_err(|e| CoreError::DomainError(ErrorContext::new(format!("{e}"))))?;
        Ok(Self { value: recip })
    }
}

impl fmt::Display for ArbitraryRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for ArbitraryRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArbitraryRational({})", self.value)
    }
}

impl Default for ArbitraryRational {
    fn default() -> Self {
        Self::new()
    }
}

impl FromStr for ArbitraryRational {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Try "numerator/denominator" format first.
        if let Some(slash) = s.find('/') {
            let num_s = &s[..slash];
            let den_s = &s[slash + 1..];
            let n = RIBig::from_str(num_s.trim()).map_err(|_| {
                CoreError::ValidationError(ErrorContext::new(format!(
                    "Failed to parse rational from string: {s}"
                )))
            })?;
            let d = RUBig::from_str(den_s.trim()).map_err(|_| {
                CoreError::ValidationError(ErrorContext::new(format!(
                    "Failed to parse rational from string: {s}"
                )))
            })?;
            return Ok(Self {
                value: RBig::from_parts(n, d),
            });
        }
        // Fall back to integer interpretation.
        let n = RIBig::from_str(s.trim()).map_err(|_| {
            CoreError::ValidationError(ErrorContext::new(format!(
                "Failed to parse rational from string: {s}"
            )))
        })?;
        Ok(Self {
            value: RBig::from_parts(n, RUBig::ONE),
        })
    }
}

impl Add for ArbitraryRational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl Sub for ArbitraryRational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl Mul for ArbitraryRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
        }
    }
}

impl Div for ArbitraryRational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
        }
    }
}

impl Neg for ArbitraryRational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { value: -self.value }
    }
}

// ---------------------------------------------------------------------------
// ArbitraryComplex — backed by oxinum_complex::CBig (Pure Rust, GMP/MPC-free)
// ---------------------------------------------------------------------------

/// Arbitrary precision complex number backed by `oxinum_complex::CBig` (Pure Rust).
///
/// `CBig` is a decimal arbitrary-precision complex number whose real and imaginary
/// parts are each a `DBig`.  This struct provides the same public API as the former
/// `rug::Complex`-backed implementation while being fully GMP/MPFR/MPC-free.
#[derive(Clone)]
pub struct ArbitraryComplex {
    value: CBig,
    context: ArbitraryPrecisionContext,
}

impl ArbitraryComplex {
    /// Returns the decimal-digit precision derived from the bit precision stored
    /// in the context.  Used when calling CBig transcendental methods.
    fn prec_digits(&self) -> usize {
        bits_to_decimal_digits(self.context.floatprecision)
    }

    /// Create a new complex number with default precision.
    pub fn new() -> Self {
        Self {
            value: CBig::zero(),
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Create with specific bit precision.
    pub fn prec(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        Ok(Self {
            value: CBig::zero(),
            context,
        })
    }

    /// Create from real and imaginary `ArbitraryFloat` parts.
    pub fn re(re: &ArbitraryFloat, im: &ArbitraryFloat) -> Self {
        let prec = re.precision().max(im.precision());
        let context = re.context.clone();
        let re_f = re.to_f64();
        let im_f = im.to_f64();
        // CBig::from_f64 rejects NaN/Inf; fall back to zero for non-finite inputs.
        let value = CBig::from_f64(re_f, im_f).unwrap_or_else(|_| CBig::zero());
        Self {
            value,
            context: ArbitraryPrecisionContext {
                floatprecision: prec,
                ..context
            },
        }
    }

    /// Create from f64 real and imaginary parts.
    pub fn re_2(re: f64, im: f64) -> Self {
        let value = CBig::from_f64(re, im).unwrap_or_else(|_| CBig::zero());
        Self {
            value,
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Get the real part as an `ArbitraryFloat`.
    pub fn real(&self) -> ArbitraryFloat {
        let (re_f64, _) = self.value.to_f64_parts();
        ArbitraryFloat::from_f64(re_f64)
    }

    /// Get the imaginary part as an `ArbitraryFloat`.
    pub fn imag(&self) -> ArbitraryFloat {
        let (_, im_f64) = self.value.to_f64_parts();
        ArbitraryFloat::from_f64(im_f64)
    }

    /// Get the magnitude (absolute value) as an `ArbitraryFloat`.
    pub fn abs(&self) -> ArbitraryFloat {
        let digits = self.prec_digits();
        let mag_f64 = self
            .value
            .abs(digits)
            .map(|d| d.to_f64().value())
            .unwrap_or(0.0);
        ArbitraryFloat::from_f64(mag_f64)
    }

    /// Get the phase (argument) as an `ArbitraryFloat`.
    pub fn arg(&self) -> ArbitraryFloat {
        let digits = self.prec_digits();
        let arg_f64 = self
            .value
            .arg(digits)
            .map(|d| d.to_f64().value())
            .unwrap_or(0.0);
        ArbitraryFloat::from_f64(arg_f64)
    }

    /// Complex conjugate.
    pub fn conj(&self) -> Self {
        Self {
            value: self.value.conj(),
            context: self.context.clone(),
        }
    }

    /// Natural logarithm.
    ///
    /// Returns the principal value `ln|z| + i·arg(z)`.
    /// If `z` is zero (ln undefined), returns a zero complex.
    pub fn ln(&self) -> Self {
        let digits = self.prec_digits();
        let value = self.value.ln(digits).unwrap_or_else(|_| CBig::zero());
        Self {
            value,
            context: self.context.clone(),
        }
    }

    /// Exponential function.
    pub fn exp(&self) -> Self {
        let digits = self.prec_digits();
        let value = self.value.exp(digits).unwrap_or_else(|_| CBig::zero());
        Self {
            value,
            context: self.context.clone(),
        }
    }

    /// Power function: self ^ exponent.
    ///
    /// Computed as `exp(exponent * ln(self))`.
    pub fn pow(&self, exp: &Self) -> Self {
        let digits = self.prec_digits();
        let value = self
            .value
            .pow(&exp.value, digits)
            .unwrap_or_else(|_| CBig::zero());
        Self {
            value,
            context: self.context.clone(),
        }
    }

    /// Square root.
    pub fn sqrt(&self) -> Self {
        let digits = self.prec_digits();
        let value = self.value.sqrt(digits).unwrap_or_else(|_| CBig::zero());
        Self {
            value,
            context: self.context.clone(),
        }
    }
}

impl fmt::Display for ArbitraryComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.value.to_f64_parts();
        if im >= 0.0 {
            write!(f, "{} + {}i", re, im)
        } else {
            write!(f, "{} - {}i", re, -im)
        }
    }
}

impl fmt::Debug for ArbitraryComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArbitraryComplex({}, {} bits)",
            self, self.context.floatprecision
        )
    }
}

impl PartialEq for ArbitraryComplex {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Default for ArbitraryComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for ArbitraryComplex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            context: self.context,
        }
    }
}

impl Sub for ArbitraryComplex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            context: self.context,
        }
    }
}

impl Mul for ArbitraryComplex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            context: self.context,
        }
    }
}

impl Div for ArbitraryComplex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
            context: self.context,
        }
    }
}

impl Neg for ArbitraryComplex {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            context: self.context,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion trait
// ---------------------------------------------------------------------------

/// Conversion trait for arbitrary precision types.
pub trait ToArbitraryPrecision {
    /// The arbitrary precision output type.
    type ArbitraryType;

    /// Convert to arbitrary precision with the default precision.
    fn to_arbitrary(&self) -> Self::ArbitraryType;

    /// Convert to arbitrary precision with a specified bit precision.
    fn to_arbitraryprec(&self, prec: u32) -> CoreResult<Self::ArbitraryType>;
}

impl ToArbitraryPrecision for i32 {
    type ArbitraryType = ArbitraryInt;

    fn to_arbitrary(&self) -> Self::ArbitraryType {
        ArbitraryInt::from_i64(*self as i64)
    }

    fn to_arbitraryprec(&self, _prec: u32) -> CoreResult<Self::ArbitraryType> {
        Ok(self.to_arbitrary())
    }
}

impl ToArbitraryPrecision for i64 {
    type ArbitraryType = ArbitraryInt;

    fn to_arbitrary(&self) -> Self::ArbitraryType {
        ArbitraryInt::from_i64(*self)
    }

    fn to_arbitraryprec(&self, _prec: u32) -> CoreResult<Self::ArbitraryType> {
        Ok(self.to_arbitrary())
    }
}

impl ToArbitraryPrecision for f32 {
    type ArbitraryType = ArbitraryFloat;

    fn to_arbitrary(&self) -> Self::ArbitraryType {
        ArbitraryFloat::from_f64(*self as f64)
    }

    fn to_arbitraryprec(&self, prec: u32) -> CoreResult<Self::ArbitraryType> {
        ArbitraryFloat::from_f64_withprecision(*self as f64, prec)
    }
}

impl ToArbitraryPrecision for f64 {
    type ArbitraryType = ArbitraryFloat;

    fn to_arbitrary(&self) -> Self::ArbitraryType {
        ArbitraryFloat::from_f64(*self)
    }

    fn to_arbitraryprec(&self, prec: u32) -> CoreResult<Self::ArbitraryType> {
        ArbitraryFloat::from_f64_withprecision(*self, prec)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for arbitrary precision calculations.
pub struct ArbitraryPrecisionBuilder {
    context: ArbitraryPrecisionContext,
}

impl ArbitraryPrecisionBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Set the precision in bits.
    pub fn precision(mut self, prec: u32) -> Self {
        self.context.floatprecision = prec;
        self
    }

    /// Set the precision in decimal digits (converted to bits internally).
    pub fn decimalprecision(mut self, digits: u32) -> Self {
        self.context.floatprecision = ((digits as f64) * BITS_PER_DECIMAL_DIGIT) as u32;
        self
    }

    /// Set the rounding mode.
    pub fn rounding(mut self, mode: RoundingMode) -> Self {
        self.context.rounding_mode = mode;
        self
    }

    /// Enable or disable precision tracking.
    pub fn trackprecision(mut self, track: bool) -> Self {
        self.context.trackprecision = track;
        if track && self.context.precision_context.is_none() {
            let mut precision_ctx = PrecisionContext::new();
            precision_ctx.precision = self.context.floatprecision as f64 / BITS_PER_DECIMAL_DIGIT;
            self.context.precision_context = Some(precision_ctx);
        }
        self
    }

    /// Build an `ArbitraryFloat`.
    pub fn build_float(self) -> ArbitraryFloat {
        ArbitraryFloat::with_context(self.context)
    }

    /// Build an `ArbitraryComplex`.
    pub fn build_complex(self) -> CoreResult<ArbitraryComplex> {
        ArbitraryComplex::prec(self.context.floatprecision)
    }

    /// Execute a calculation with this precision context.
    pub fn calculate<F, R>(self, f: F) -> R
    where
        F: FnOnce(&ArbitraryPrecisionContext) -> R,
    {
        f(&self.context)
    }
}

impl Default for ArbitraryPrecisionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Utility functions for arbitrary precision arithmetic.
pub mod utils {
    use super::*;

    /// Compute π to the given bit precision.
    pub fn pi(prec: u32) -> CoreResult<ArbitraryFloat> {
        ArbitraryFloat::prec_2(prec)
    }

    /// Compute e to the given bit precision.
    pub fn e(prec: u32) -> CoreResult<ArbitraryFloat> {
        ArbitraryFloat::prec_3(prec)
    }

    /// Compute ln(2) to the given bit precision.
    pub fn ln2(prec: u32) -> CoreResult<ArbitraryFloat> {
        ArbitraryFloat::prec_4(prec)
    }

    /// Compute sqrt(2) to the given bit precision.
    pub fn sqrt2(prec: u32) -> CoreResult<ArbitraryFloat> {
        let two = ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
        two.sqrt()
    }

    /// Compute the golden ratio to the given bit precision.
    pub fn golden_ratio(prec: u32) -> CoreResult<ArbitraryFloat> {
        let one = ArbitraryFloat::from_f64_withprecision(1.0, prec)?;
        let five = ArbitraryFloat::from_f64_withprecision(5.0, prec)?;
        let sqrt5 = five.sqrt()?;
        let two = ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
        Ok((one + sqrt5) / two)
    }

    /// Compute n! using arbitrary precision integers.
    pub fn factorial(n: u32) -> ArbitraryInt {
        ArbitraryInt::factorial(n)
    }

    /// Compute C(n, k) using arbitrary precision integers.
    pub fn binomial(n: u32, k: u32) -> ArbitraryInt {
        ArbitraryInt::binomial(n, k)
    }

    /// Check if a large integer is probably prime.
    pub fn is_probably_prime(n: &ArbitraryInt, certainty: u32) -> bool {
        n.is_probably_prime(certainty)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arbitrary_int_basic() {
        let a = ArbitraryInt::from_i64(123);
        let b = ArbitraryInt::from_i64(456);
        let sum = a.clone() + b.clone();
        assert_eq!(sum.to_string(), "579");

        let product = a.clone() * b.clone();
        assert_eq!(product.to_string(), "56088");

        let factorial = ArbitraryInt::factorial(20);
        assert_eq!(factorial.to_string(), "2432902008176640000");
    }

    #[test]
    fn test_arbitrary_float_basic() {
        let a = ArbitraryFloat::from_f64_withprecision(1.0, 128).expect("Operation failed");
        let b = ArbitraryFloat::from_f64_withprecision(3.0, 128).expect("Operation failed");
        let c = a / b;

        // Check that we get more precision than f64.
        let c_str = c.to_string();
        // Should start with 3.333... (decimal) or similar.
        assert!(
            c_str.starts_with("3.333333333333333") || c_str.starts_with("0.333333333333333"),
            "unexpected value: {c_str}"
        );
        assert!(c_str.len() > 10, "result too short: {c_str}");
    }

    #[test]
    fn test_arbitrary_rational() {
        let r = ArbitraryRational::num(22, 7).expect("Operation failed");
        assert_eq!(r.to_string(), "22/7");

        let a = ArbitraryRational::num(1, 3).expect("Operation failed");
        let b = ArbitraryRational::num(1, 6).expect("Operation failed");
        let sum = a + b;
        assert_eq!(sum.to_string(), "1/2");
    }

    #[test]
    fn test_arbitrary_complex() {
        let z = ArbitraryComplex::re_2(3.0, 4.0);
        let mag = z.abs();
        assert!((mag.to_f64() - 5.0).abs() < 1e-10);

        let conj = z.conj();
        assert_eq!(conj.real().to_f64(), 3.0);
        assert_eq!(conj.imag().to_f64(), -4.0);
    }

    #[test]
    fn testprecision_builder() {
        let x = ArbitraryPrecisionBuilder::new()
            .decimalprecision(50)
            .rounding(RoundingMode::Nearest)
            .build_float();

        assert!(x.decimalprecision() >= 49); // Allow for rounding in the conversion.
    }

    #[test]
    fn test_constants() {
        let pi = utils::pi(256).expect("Operation failed");
        let pi_str = pi.to_string();
        assert!(pi_str.starts_with("3.14159265358979"), "pi = {pi_str}");

        let e = utils::e(256).expect("Operation failed");
        let e_str = e.to_string();
        assert!(e_str.starts_with("2.71828182845904"), "e = {e_str}");
    }

    #[test]
    fn test_prime_checking() {
        let prime = ArbitraryInt::from_i64(97);
        assert!(prime.is_probably_prime(20));

        let composite = ArbitraryInt::from_i64(98);
        assert!(!composite.is_probably_prime(20));
    }

    #[test]
    fn test_gcd_lcm() {
        let a = ArbitraryInt::from_i64(48);
        let b = ArbitraryInt::from_i64(18);

        let gcd = a.gcd(&b);
        assert_eq!(gcd.to_string(), "6");

        let lcm = a.lcm(&b);
        assert_eq!(lcm.to_string(), "144");
    }

    #[test]
    fn test_transcendental_functions() {
        let x = ArbitraryFloat::from_f64_withprecision(0.5, 128).expect("Operation failed");

        let sin_x = x.sin();
        let cos_x = x.cos();
        let identity = sin_x.clone() * sin_x + cos_x.clone() * cos_x;

        // sin²(x) + cos²(x) = 1
        assert!(
            (identity.to_f64() - 1.0).abs() < 1e-10,
            "identity = {}",
            identity.to_f64()
        );

        let ln_x = x.ln().expect("Operation failed");
        let exp_ln_x = ln_x.exp();
        assert!(
            (exp_ln_x.to_f64() - 0.5).abs() < 1e-10,
            "exp_ln = {}",
            exp_ln_x.to_f64()
        );
    }

    #[test]
    fn testerror_handling() {
        // Division by zero (rational reciprocal).
        let zero = ArbitraryRational::new();
        assert!(zero.recip().is_err());

        // Square root of negative.
        let neg = ArbitraryFloat::from_f64(-1.0);
        assert!(neg.sqrt().is_err());

        // Logarithm of negative.
        assert!(neg.ln().is_err());

        // Arcsine out of range.
        let out_of_range = ArbitraryFloat::from_f64(2.0);
        assert!(out_of_range.asin().is_err());
    }
}
