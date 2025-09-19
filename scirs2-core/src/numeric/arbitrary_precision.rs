//! # Arbitrary Precision Numerical Computation Support
//!
//! This module provides arbitrary precision arithmetic capabilities for scientific computing,
//! enabling calculations with user-defined precision levels for both integers and floating-point numbers.
//!
//! ## Features
//!
//! - Arbitrary precision integers (BigInt)
//! - Arbitrary precision floating-point numbers (BigFloat)
//! - Exact rational arithmetic (BigRational)
//! - Arbitrary precision complex numbers (BigComplex)
//! - Integration with existing ScientificNumber traits
//! - Automatic precision tracking and management
//! - Configurable precision contexts
//! - Efficient operations with GMP/MPFR backend

use crate::{
    error::{CoreError, CoreResult, ErrorContext},
    numeric::precision_tracking::PrecisionContext,
    validation::check_positive,
};
use num_bigint::BigInt;
use rug::{
    float::Round, ops::Pow, Complex as RugComplex, Float as RugFloat, Integer as RugInteger,
    Rational as RugRational,
};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;
use std::sync::RwLock;

/// Global default precision for arbitrary precision operations
static DEFAULT_PRECISION: RwLock<u32> = RwLock::new(256);

/// Get the default precision for arbitrary precision operations
#[allow(dead_code)]
pub fn get_defaultprecision() -> u32 {
    *DEFAULT_PRECISION.read().unwrap()
}

/// Set the default precision for arbitrary precision operations
#[allow(dead_code)]
pub fn setprecision(prec: u32) -> CoreResult<()> {
    check_positive(prec as f64, "precision")?;
    *DEFAULT_PRECISION.write().unwrap() = prec;
    Ok(())
}

/// Precision context for arbitrary precision arithmetic
#[derive(Debug, Clone)]
pub struct ArbitraryPrecisionContext {
    /// Precision in bits for floating-point operations
    pub floatprecision: u32,
    /// Maximum precision allowed
    pub maxprecision: u32,
    /// Rounding mode
    pub rounding_mode: RoundingMode,
    /// Whether to track precision loss
    pub trackprecision: bool,
    /// Precision tracking context
    pub precision_context: Option<PrecisionContext>,
}

/// Rounding modes for arbitrary precision arithmetic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even
    Nearest,
    /// Round toward zero
    Zero,
    /// Round toward positive infinity
    Up,
    /// Round toward negative infinity
    Down,
    /// Round away from zero
    Away,
}

impl RoundingMode {
    /// Convert to rug::float::Round
    fn to_rug_round(self) -> Round {
        match self {
            RoundingMode::Nearest => Round::Nearest,
            RoundingMode::Zero => Round::Zero,
            RoundingMode::Up => Round::Up,
            RoundingMode::Down => Round::Down,
            RoundingMode::Away => Round::Up, // Use Up as Away is not available
        }
    }
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
    /// Create a new precision context with specified precision
    pub fn withprecision(precision: u32) -> CoreResult<Self> {
        check_positive(precision as f64, "precision")?;
        Ok(Self {
            floatprecision: precision,
            ..Default::default()
        })
    }

    /// Create a context with precision tracking enabled
    pub fn withprecision_tracking(precision: u32) -> CoreResult<Self> {
        let mut ctx = Self::withprecision(precision)?;
        ctx.trackprecision = true;
        let mut precision_ctx = PrecisionContext::new();
        precision_ctx.precision = precision as f64 / 3.32; // bits to decimal digits
        ctx.precision_context = Some(precision_ctx);
        Ok(ctx)
    }

    /// Set the rounding mode
    pub fn with_rounding(mut self, mode: RoundingMode) -> Self {
        self.rounding_mode = mode;
        self
    }

    /// Set the maximum precision
    pub fn with_maxprecision(mut self, maxprec: u32) -> Self {
        self.maxprecision = maxprec;
        self
    }
}

/// Arbitrary precision integer
#[derive(Clone, PartialEq, Eq)]
pub struct ArbitraryInt {
    value: RugInteger,
}

impl ArbitraryInt {
    /// Create a new arbitrary precision integer
    pub fn new() -> Self {
        Self {
            value: RugInteger::new(),
        }
    }

    /// Create from a regular integer
    pub fn from_i64(n: i64) -> Self {
        Self {
            value: RugInteger::from(n),
        }
    }

    /// Create from a string
    pub fn from_str_radix(s: &str, radix: i32) -> CoreResult<Self> {
        RugInteger::parse_radix(s, radix)
            .map(|incomplete| {
                let value = RugInteger::from(incomplete);
                Self { value }
            })
            .map_err(|_| {
                CoreError::ValidationError(ErrorContext::new(format!(
                    "Failed to parse integer from string: {}",
                    s
                )))
            })
    }

    /// Convert to BigInt
    pub fn to_bigint(&self) -> BigInt {
        BigInt::from_str(&self.value.to_string()).unwrap()
    }

    /// Check if the number is prime
    pub fn is_probably_prime(&self, reps: u32) -> bool {
        self.value.is_probably_prime(reps) != rug::integer::IsPrime::No
    }

    /// Compute factorial
    pub fn factorial(n: u32) -> Self {
        let mut result = RugInteger::from(1);
        for i in 2..=n {
            result *= i;
        }
        Self { value: result }
    }

    /// Compute binomial coefficient
    pub fn binomial(n: u32, k: u32) -> Self {
        if k > n {
            return Self::new();
        }
        let mut result = RugInteger::from(1);
        for i in 0..k {
            result *= n - i;
            result /= i + 1;
        }
        Self { value: result }
    }

    /// Compute greatest common divisor
    pub fn gcd(&self, other: &Self) -> Self {
        Self {
            value: self.value.clone().gcd(&other.value),
        }
    }

    /// Compute least common multiple
    pub fn lcm(&self, other: &Self) -> Self {
        if self.value.is_zero() || other.value.is_zero() {
            return Self::new();
        }
        let gcd = self.gcd(other);
        // Create a complete integer from the multiplication
        let product = RugInteger::from(&self.value * &other.value);
        Self {
            value: product / gcd.value,
        }
    }

    /// Modular exponentiation
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> CoreResult<Self> {
        if modulus.value.is_zero() {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Modulus cannot be zero",
            )));
        }
        Ok(Self {
            value: self
                .value
                .clone()
                .pow_mod(&exp.value, &modulus.value)
                .unwrap(),
        })
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Self {
            value: self.value.clone().abs(),
        }
    }

    /// Get the sign (-1, 0, or 1)
    pub fn signum(&self) -> i32 {
        self.value.cmp0() as i32
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

// Implement arithmetic operations for ArbitraryInt
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

/// Arbitrary precision floating-point number
#[derive(Clone)]
pub struct ArbitraryFloat {
    value: RugFloat,
    context: ArbitraryPrecisionContext,
}

impl ArbitraryFloat {
    /// Create a new arbitrary precision float with default precision
    pub fn new() -> Self {
        let prec = get_defaultprecision();
        Self {
            value: RugFloat::new(prec),
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Create with specific precision
    pub fn withprecision(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        Ok(Self {
            value: RugFloat::new(prec),
            context,
        })
    }

    /// Create with specific context
    pub fn with_context(context: ArbitraryPrecisionContext) -> Self {
        Self {
            value: RugFloat::new(context.floatprecision),
            context,
        }
    }

    /// Create from f64 with default precision
    pub fn from_f64(value: f64) -> Self {
        let prec = get_defaultprecision();
        Self {
            value: RugFloat::with_val(prec, value),
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Create from f64 with specific precision
    pub fn from_f64_withprecision(value: f64, prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        Ok(Self {
            value: RugFloat::with_val(prec, value),
            context,
        })
    }

    /// Create from string with specific precision
    pub fn from_strprec(s: &str, prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        RugFloat::parse(s)
            .map(|incomplete| Self {
                value: RugFloat::with_val(prec, incomplete),
                context,
            })
            .map_err(|e| CoreError::ValidationError(ErrorContext::new(format!("{e}"))))
    }

    /// Get the precision in bits
    pub fn precision(&self) -> u32 {
        self.value.prec()
    }

    /// Get the precision in decimal digits
    pub fn decimalprecision(&self) -> u32 {
        (self.value.prec() as f64 / 3.32) as u32
    }

    /// Set precision (returns a new value with the specified precision)
    pub fn setprecision(&self, prec: u32) -> CoreResult<Self> {
        let mut context = self.context.clone();
        context.floatprecision = prec;
        Ok(Self {
            value: RugFloat::with_val(prec, &self.value),
            context,
        })
    }

    /// Convert to f64 (may lose precision)
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    /// Check if the value is finite
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Check if the value is infinite
    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }

    /// Check if the value is NaN
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    /// Check if the value is zero
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        let mut result = self.clone();
        result.value.abs_mut();
        result
    }

    /// Square root
    pub fn sqrt(&self) -> CoreResult<Self> {
        if self.value.is_sign_negative() && !self.value.is_zero() {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Square root of negative number",
            )));
        }
        let mut result = self.clone();
        result.value.sqrt_mut();
        Ok(result)
    }

    /// Natural logarithm
    pub fn ln(&self) -> CoreResult<Self> {
        if self.value.is_sign_negative() || self.value.is_zero() {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Logarithm of non-positive number",
            )));
        }
        let mut result = self.clone();
        result.value.ln_mut();
        Ok(result)
    }

    /// Exponential function
    pub fn exp(&self) -> Self {
        let mut result = self.clone();
        result.value.exp_mut();
        result
    }

    /// Power function
    pub fn pow(&self, exp: &Self) -> Self {
        let mut result = self.clone();
        let pow_result = self.value.clone().pow(&exp.value);
        result.value = RugFloat::with_val(self.context.floatprecision, pow_result);
        result
    }

    /// Sine
    pub fn sin(&self) -> Self {
        let mut result = self.clone();
        result.value.sin_mut();
        result
    }

    /// Cosine
    pub fn cos(&self) -> Self {
        let mut result = self.clone();
        result.value.cos_mut();
        result
    }

    /// Tangent
    pub fn tan(&self) -> Self {
        let mut result = self.clone();
        result.value.tan_mut();
        result
    }

    /// Arcsine
    pub fn asin(&self) -> CoreResult<Self> {
        if self.value.clone().abs() > 1 {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Arcsine argument out of range [-1, 1]",
            )));
        }
        let mut result = self.clone();
        result.value.asin_mut();
        Ok(result)
    }

    /// Arccosine
    pub fn acos(&self) -> CoreResult<Self> {
        if self.value.clone().abs() > 1 {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Arccosine argument out of range [-1, 1]",
            )));
        }
        let mut result = self.clone();
        result.value.acos_mut();
        Ok(result)
    }

    /// Arctangent
    pub fn atan(&self) -> Self {
        let mut result = self.clone();
        result.value.atan_mut();
        result
    }

    /// Two-argument arctangent
    pub fn atan2(&self, x: &Self) -> Self {
        let mut result = self.clone();
        result.value = RugFloat::with_val(
            self.context.floatprecision,
            self.value.clone().atan2(&x.value),
        );
        result
    }

    /// Hyperbolic sine
    pub fn sinh(&self) -> Self {
        let mut result = self.clone();
        result.value.sinh_mut();
        result
    }

    /// Hyperbolic cosine
    pub fn cosh(&self) -> Self {
        let mut result = self.clone();
        result.value.cosh_mut();
        result
    }

    /// Hyperbolic tangent
    pub fn tanh(&self) -> Self {
        let mut result = self.clone();
        result.value.tanh_mut();
        result
    }

    /// Constants
    pub fn prec_2(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let value = RugFloat::with_val(prec, rug::float::Constant::Pi);
        Ok(Self { value, context })
    }

    pub fn prec_3(prec: u32) -> CoreResult<Self> {
        let one = Self::from_f64_withprecision(1.0, prec)?;
        Ok(one.exp())
    }

    pub fn prec_4(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        let value = RugFloat::with_val(prec, rug::float::Constant::Log2);
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
        self.value.eq(&other.value)
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

// Arithmetic operations for ArbitraryFloat
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

/// Arbitrary precision rational number
#[derive(Clone, PartialEq, Eq)]
pub struct ArbitraryRational {
    value: RugRational,
}

impl ArbitraryRational {
    /// Create a new rational number (0/1)
    pub fn new() -> Self {
        Self {
            value: RugRational::new(),
        }
    }

    /// Create from numerator and denominator
    pub fn num(num: i64, den: i64) -> CoreResult<Self> {
        if den == 0 {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Denominator cannot be zero",
            )));
        }
        Ok(Self {
            value: RugRational::from((num, den)),
        })
    }

    /// Create from arbitrary precision integers
    pub fn num_2(num: &ArbitraryInt, den: &ArbitraryInt) -> CoreResult<Self> {
        if den.value.is_zero() {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Denominator cannot be zero",
            )));
        }
        Ok(Self {
            value: RugRational::from((&num.value, &den.value)),
        })
    }

    /// Create from string (e.g., "22/7" or "3.14159")  
    /// Note: This is deprecated, use `str::parse()` instead
    #[deprecated(note = "Use str::parse() instead")]
    pub fn parse_rational(s: &str) -> CoreResult<Self> {
        s.parse()
    }

    /// Convert to f64 (may lose precision)
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    /// Convert to arbitrary precision float
    pub fn to_arbitrary_float(&self, prec: u32) -> CoreResult<ArbitraryFloat> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        Ok(ArbitraryFloat {
            value: RugFloat::with_val(prec, &self.value),
            context,
        })
    }

    /// Get numerator
    pub fn numerator(&self) -> ArbitraryInt {
        ArbitraryInt {
            value: self.value.numer().clone(),
        }
    }

    /// Get denominator
    pub fn denominator(&self) -> ArbitraryInt {
        ArbitraryInt {
            value: self.value.denom().clone(),
        }
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Self {
            value: self.value.clone().abs(),
        }
    }

    /// Get the reciprocal
    pub fn recip(&self) -> CoreResult<Self> {
        if self.value.is_zero() {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Cannot take reciprocal of zero",
            )));
        }
        Ok(Self {
            value: self.value.clone().recip(),
        })
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
        RugRational::from_str(s)
            .map(|value| Self { value })
            .map_err(|_| {
                CoreError::ValidationError(ErrorContext::new(format!(
                    "Failed to parse rational from string: {}",
                    s
                )))
            })
    }
}

// Arithmetic operations for ArbitraryRational
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

/// Arbitrary precision complex number
#[derive(Clone)]
pub struct ArbitraryComplex {
    value: RugComplex,
    context: ArbitraryPrecisionContext,
}

impl ArbitraryComplex {
    /// Create a new complex number with default precision
    pub fn new() -> Self {
        let prec = get_defaultprecision();
        Self {
            value: RugComplex::new(prec),
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Create with specific precision
    pub fn prec(prec: u32) -> CoreResult<Self> {
        let context = ArbitraryPrecisionContext::withprecision(prec)?;
        Ok(Self {
            value: RugComplex::new(prec),
            context,
        })
    }

    /// Create from real and imaginary parts
    pub fn re(re: &ArbitraryFloat, im: &ArbitraryFloat) -> Self {
        let prec = re.precision().max(im.precision());
        let context = re.context.clone();
        Self {
            value: RugComplex::with_val(prec, (&re.value, &im.value)),
            context,
        }
    }

    /// Create from f64 parts
    pub fn re_2(re: f64, im: f64) -> Self {
        let prec = get_defaultprecision();
        Self {
            value: RugComplex::with_val(prec, (re, im)),
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Get real part
    pub fn real(&self) -> ArbitraryFloat {
        ArbitraryFloat {
            value: self.value.real().clone(),
            context: self.context.clone(),
        }
    }

    /// Get imaginary part
    pub fn imag(&self) -> ArbitraryFloat {
        ArbitraryFloat {
            value: self.value.imag().clone(),
            context: self.context.clone(),
        }
    }

    /// Get magnitude (absolute value)
    pub fn abs(&self) -> ArbitraryFloat {
        let mut result = ArbitraryFloat::with_context(self.context.clone());
        result.value = RugFloat::with_val(self.context.floatprecision, self.value.abs_ref());
        result
    }

    /// Get phase (argument)
    pub fn arg(&self) -> ArbitraryFloat {
        let mut result = ArbitraryFloat::with_context(self.context.clone());
        result.value = RugFloat::with_val(self.context.floatprecision, self.value.arg_ref());
        result
    }

    /// Complex conjugate
    pub fn conj(&self) -> Self {
        let mut result = self.clone();
        result.value.conj_mut();
        result
    }

    /// Natural logarithm
    pub fn ln(&self) -> Self {
        let mut result = self.clone();
        result.value.ln_mut();
        result
    }

    /// Exponential function
    pub fn exp(&self) -> Self {
        let mut result = self.clone();
        result.value.exp_mut();
        result
    }

    /// Power function
    pub fn pow(&self, exp: &Self) -> Self {
        // z^w = exp(w * ln(z))
        let ln_self = self.ln();
        let product = ln_self * exp.clone();
        product.exp()
    }

    /// Square root
    pub fn sqrt(&self) -> Self {
        let mut result = self.clone();
        result.value.sqrt_mut();
        result
    }
}

impl fmt::Display for ArbitraryComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let re = self.value.real();
        let im = self.value.imag();
        if im.is_sign_positive() {
            write!(f, "{} + {}i", re, im)
        } else {
            write!(f, "{} - {}i", re, -im.clone())
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
        self.value.eq(&other.value)
    }
}

impl Default for ArbitraryComplex {
    fn default() -> Self {
        Self::new()
    }
}

// Arithmetic operations for ArbitraryComplex
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

/// Conversion trait for arbitrary precision types
pub trait ToArbitraryPrecision {
    /// The arbitrary precision type
    type ArbitraryType;

    /// Convert to arbitrary precision with default precision
    fn to_arbitrary(&self) -> Self::ArbitraryType;

    /// Convert to arbitrary precision with specified precision
    fn to_arbitraryprec(&self, prec: u32) -> CoreResult<Self::ArbitraryType>;
}

impl ToArbitraryPrecision for i32 {
    type ArbitraryType = ArbitraryInt;

    fn to_arbitrary(&self) -> Self::ArbitraryType {
        ArbitraryInt::from_i64(*self as i64)
    }

    fn to_arbitraryprec(&self, prec: u32) -> CoreResult<Self::ArbitraryType> {
        Ok(self.to_arbitrary())
    }
}

impl ToArbitraryPrecision for i64 {
    type ArbitraryType = ArbitraryInt;

    fn to_arbitrary(&self) -> Self::ArbitraryType {
        ArbitraryInt::from_i64(*self)
    }

    fn to_arbitraryprec(&self, prec: u32) -> CoreResult<Self::ArbitraryType> {
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

/// Builder for arbitrary precision calculations
pub struct ArbitraryPrecisionBuilder {
    context: ArbitraryPrecisionContext,
}

impl ArbitraryPrecisionBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            context: ArbitraryPrecisionContext::default(),
        }
    }

    /// Set the precision in bits
    pub fn precision(mut self, prec: u32) -> Self {
        self.context.floatprecision = prec;
        self
    }

    /// Set the precision in decimal digits
    pub fn decimalprecision(mut self, digits: u32) -> Self {
        self.context.floatprecision = ((digits as f64) * 3.32) as u32;
        self
    }

    /// Set the rounding mode
    pub fn rounding(mut self, mode: RoundingMode) -> Self {
        self.context.rounding_mode = mode;
        self
    }

    /// Enable precision tracking
    pub fn trackprecision(mut self, track: bool) -> Self {
        self.context.trackprecision = track;
        if track && self.context.precision_context.is_none() {
            let mut precision_ctx = PrecisionContext::new();
            precision_ctx.precision = self.context.floatprecision as f64 / 3.32;
            self.context.precision_context = Some(precision_ctx);
        }
        self
    }

    /// Build an ArbitraryFloat
    pub fn build_float(self) -> ArbitraryFloat {
        ArbitraryFloat::with_context(self.context)
    }

    /// Build an ArbitraryComplex
    pub fn build_complex(self) -> CoreResult<ArbitraryComplex> {
        ArbitraryComplex::prec(self.context.floatprecision)
    }

    /// Execute a calculation with this precision context
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

/// Utility functions for arbitrary precision arithmetic
pub mod utils {
    use super::*;

    /// Compute π to specified precision
    pub fn pi(prec: u32) -> CoreResult<ArbitraryFloat> {
        ArbitraryFloat::prec_2(prec)
    }

    /// Compute e to specified precision
    pub fn e(prec: u32) -> CoreResult<ArbitraryFloat> {
        ArbitraryFloat::prec_3(prec)
    }

    /// Compute ln(2) to specified precision
    pub fn ln2(prec: u32) -> CoreResult<ArbitraryFloat> {
        ArbitraryFloat::prec_4(prec)
    }

    /// Compute sqrt(2) to specified precision
    pub fn sqrt2(prec: u32) -> CoreResult<ArbitraryFloat> {
        let two = ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
        two.sqrt()
    }

    /// Compute golden ratio to specified precision
    pub fn golden_ratio(prec: u32) -> CoreResult<ArbitraryFloat> {
        let one = ArbitraryFloat::from_f64_withprecision(1.0, prec)?;
        let five = ArbitraryFloat::from_f64_withprecision(5.0, prec)?;
        let sqrt5 = five.sqrt()?;
        let two = ArbitraryFloat::from_f64_withprecision(2.0, prec)?;
        Ok((one + sqrt5) / two)
    }

    /// Compute factorial using arbitrary precision
    pub fn factorial(n: u32) -> ArbitraryInt {
        ArbitraryInt::factorial(n)
    }

    /// Compute binomial coefficient using arbitrary precision
    pub fn binomial(n: u32, k: u32) -> ArbitraryInt {
        ArbitraryInt::binomial(n, k)
    }

    /// Check if a large integer is probably prime
    pub fn is_probably_prime(n: &ArbitraryInt, certainty: u32) -> bool {
        n.is_probably_prime(certainty)
    }
}

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
        let a = ArbitraryFloat::from_f64_withprecision(1.0, 128).unwrap();
        let b = ArbitraryFloat::from_f64_withprecision(3.0, 128).unwrap();
        let c = a / b;

        // Check that we get more precision than f64
        let c_str = c.to_string();
        // Scientific notation: 3.3333...e-1 or decimal: 0.3333...
        assert!(c_str.starts_with("3.333333333333333") || c_str.starts_with("0.333333333333333"));
        assert!(c_str.len() > 20); // More digits than f64 can represent
    }

    #[test]
    fn test_arbitrary_rational() {
        let r = ArbitraryRational::num(22, 7).unwrap();
        assert_eq!(r.to_string(), "22/7");

        let a = ArbitraryRational::num(1, 3).unwrap();
        let b = ArbitraryRational::num(1, 6).unwrap();
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

        assert!(x.decimalprecision() >= 49); // Allow for rounding
    }

    #[test]
    fn test_constants() {
        let pi = utils::pi(256).unwrap();
        let pi_str = pi.to_string();
        assert!(pi_str.starts_with("3.14159265358979"));

        let e = utils::e(256).unwrap();
        let e_str = e.to_string();
        assert!(e_str.starts_with("2.71828182845904"));
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
        let x = ArbitraryFloat::from_f64_withprecision(0.5, 128).unwrap();

        let sin_x = x.sin();
        let cos_x = x.cos();
        let identity = sin_x.clone() * sin_x + cos_x.clone() * cos_x;

        // sin²(x) + cos²(x) = 1
        assert!((identity.to_f64() - 1.0).abs() < 1e-15);

        let ln_x = x.ln().unwrap();
        let exp_ln_x = ln_x.exp();
        assert!((exp_ln_x.to_f64() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn testerror_handling() {
        // Division by zero
        let zero = ArbitraryRational::new();
        assert!(zero.recip().is_err());

        // Square root of negative
        let neg = ArbitraryFloat::from_f64(-1.0);
        assert!(neg.sqrt().is_err());

        // Logarithm of negative
        assert!(neg.ln().is_err());

        // Arcsine out of range
        let out_of_range = ArbitraryFloat::from_f64(2.0);
        assert!(out_of_range.asin().is_err());
    }
}
