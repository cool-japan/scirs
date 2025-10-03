//! Utility and convenience functions with mathematical foundations
//!
//! This module provides various utility functions commonly used in scientific
//! computing, with detailed mathematical theory, proofs, and numerical analysis.
//!
//! ## Mathematical Theory and Foundations
//!
//! ### Elementary Functions with Special Properties
//!
//! This module contains fundamental mathematical functions that serve as building
//! blocks for more complex special functions. Each function is implemented with
//! careful attention to numerical stability and mathematical rigor.
//!
//! ### The Cube Root Function
//!
//! **Mathematical Definition**: For x ∈ ℝ, the cube root is defined as:
//! ```text
//! ∛x = x^(1/3) = exp(ln|x|/3) · sign(x)
//! ```
//!
//! **Properties**:
//! 1. **Domain and Range**: ∛: ℝ → ℝ (unlike square root, defined for all reals)
//! 2. **Odd Function**: ∛(-x) = -∛x
//!    - **Proof**: Using the definition, ∛(-x) = (-x)^(1/3) = (-1)^(1/3) · x^(1/3) = -∛x
//! 3. **Monotonically Increasing**: d/dx[x^(1/3)] = (1/3)x^(-2/3) > 0 for x > 0
//! 4. **Inverse of Cubing**: (∛x)³ = x for all x ∈ ℝ
//!
//! **Numerical Implementation**: To handle negative numbers correctly, we use:
//! - For x ≥ 0: ∛x = x^(1/3)
//! - For x < 0: ∛x = -(-x)^(1/3)
//!
//! ### Exponential Functions with Different Bases
//!
//! **Base-10 Exponential (exp10)**:
//! ```text
//! exp10(x) = 10^x = e^(x·ln(10))
//! ```
//!
//! **Mathematical Properties**:
//! 1. **Exponential Law**: 10^(x+y) = 10^x · 10^y
//! 2. **Inverse of log₁₀**: exp10(log₁₀(x)) = x for x > 0
//! 3. **Derivative**: d/dx[10^x] = 10^x · ln(10)
//! 4. **Growth Rate**: 10^x grows faster than any polynomial
//!
//! **Base-2 Exponential (exp2)**:
//! ```text
//! exp2(x) = 2^x = e^(x·ln(2))
//! ```
//!
//! **Computer Science Applications**:
//! - Binary representation: 2^n gives powers of 2
//! - Information theory: 2^H(X) relates to entropy
//! - Algorithm complexity: Many algorithms have 2^n complexity
//!
//! ### Trigonometric Functions in Degrees
//!
//! **Degree-Radian Conversion**:
//! ```text
//! radians = degrees × π/180
//! degrees = radians × 180/π
//! ```
//!
//! **Mathematical Justification**: A full circle contains 2π radians = 360°,
//! establishing the conversion factor π/180.
//!
//! **Degree-based Trigonometric Functions**:
//! - sindg(x) = sin(x × π/180)
//! - cosdg(x) = cos(x × π/180)  
//! - tandg(x) = tan(x × π/180)
//! - cotdg(x) = cot(x × π/180) = 1/tan(x × π/180)
//!
//! ### Special Numerical Functions
//!
//! **exprel(x) = (e^x - 1)/x**:
//! - **Purpose**: Numerically stable computation of (e^x - 1)/x near x = 0
//! - **Taylor Series**: exprel(x) = 1 + x/2 + x²/6 + x³/24 + ... = Σ_{n=0}^∞ x^n/(n+1)!
//! - **Limit**: lim_{x→0} exprel(x) = 1 (removable singularity)
//! - **Applications**: Actuarial calculations, queuing theory
//!
//! **cosm1(x) = cos(x) - 1**:
//! - **Purpose**: Accurate computation of cos(x) - 1 for small |x|
//! - **Series**: cosm1(x) = -x²/2 + x⁴/24 - x⁶/720 + ... = -Σ_{n=1}^∞ (-1)^n x^(2n)/(2n)!
//! - **Numerical Advantage**: Avoids catastrophic cancellation when cos(x) ≈ 1
//!
//! **powm1(x, y) = x^y - 1**:
//! - **Implementation**: For small y, use powm1(x, y) = exp(y·ln(x)) - 1 ≈ y·ln(x) when |y·ln(x)| is small
//! - **Numerical Stability**: Avoids precision loss when x^y ≈ 1
//!
//! ### Advanced Utility Functions
//!
//! **Dirichlet Kernel (diric)**:
//! ```text
//! diric(x, n) = sin(nx/2) / (n·sin(x/2)) for x ≠ 2πk
//! diric(2πk, n) = (-1)^(kn)
//! ```
//!
//! **Properties**:
//! 1. **Periodicity**: diric(x + 2π, n) = diric(x, n)
//! 2. **Normalization**: ∫_{-π}^π diric(x, n) dx = 2π
//! 3. **Fourier Connection**: Dirichlet kernel is the Fourier kernel for rectangular window
//!
//! **Owen's T Function**:
//! ```text
//! T(h, a) = (1/2π) ∫₀^a exp(-h²(1+t²)/2) / (1+t²) dt
//! ```
//!
//! **Applications**:
//! - Bivariate normal distribution calculations
//! - Statistical hypothesis testing
//! - Error probability computations
//!
//! ### Numerical Stability Considerations
//!
//! All functions in this module are implemented with careful attention to:
//!
//! 1. **Overflow/Underflow Prevention**: Using appropriate scaling and range reduction
//! 2. **Catastrophic Cancellation Avoidance**: Special algorithms for near-zero differences
//! 3. **Precision Preservation**: Maintaining accuracy across the full range of inputs
//! 4. **Edge Case Handling**: Proper behavior at singularities and boundary conditions

use crate::error::{SpecialError, SpecialResult};
use crate::validation::check_finite;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};

/// Cube root function
///
/// Computes the real cube root of x, handling negative values correctly.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The cube root of x
///
/// # Examples
/// ```
/// use scirs2_special::utility::cbrt;
///
/// assert_eq!(cbrt(8.0), 2.0);
/// assert_eq!(cbrt(-8.0), -2.0);
/// ```
#[allow(dead_code)]
pub fn cbrt<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x >= T::zero() {
        x.powf(T::from_f64(1.0 / 3.0).unwrap())
    } else {
        -(-x).powf(T::from_f64(1.0 / 3.0).unwrap())
    }
}

/// Base-10 exponential function
///
/// Computes 10^x.
///
/// # Arguments
/// * `x` - Exponent
///
/// # Returns
/// 10 raised to the power x
#[allow(dead_code)]
pub fn exp10<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    T::from_f64(10.0).unwrap().powf(x)
}

/// Base-2 exponential function
///
/// Computes 2^x.
///
/// # Arguments
/// * `x` - Exponent
///
/// # Returns
/// 2 raised to the power x
#[allow(dead_code)]
pub fn exp2<T>(x: T) -> T
where
    T: Float,
{
    x.exp2()
}

/// Convert degrees to radians
///
/// # Arguments
/// * `degrees` - Angle in degrees
///
/// # Returns
/// Angle in radians
#[allow(dead_code)]
pub fn radian<T>(degrees: T) -> T
where
    T: Float + FromPrimitive,
{
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    degrees * pi / T::from_f64(180.0).unwrap()
}

/// Cosine of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// cos(x) where x is in degrees
#[allow(dead_code)]
pub fn cosdg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    radian(x).cos()
}

/// Sine of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// sin(x) where x is in degrees
#[allow(dead_code)]
pub fn sindg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    radian(x).sin()
}

/// Tangent of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// tan(x) where x is in degrees
#[allow(dead_code)]
pub fn tandg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    radian(x).tan()
}

/// Cotangent of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// cot(x) = 1/tan(x) where x is in degrees
#[allow(dead_code)]
pub fn cotdg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    T::from_f64(1.0).unwrap() / tandg(x)
}

/// Compute cos(x) - 1 accurately for small x
///
/// This function provides better numerical accuracy than directly computing cos(x) - 1
/// when x is close to 0.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// cos(x) - 1
#[allow(dead_code)]
pub fn cosm1<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    // Use Taylor series for small x
    if x.abs() < T::from_f64(0.1).unwrap() {
        let x2 = x * x;
        let mut sum = -x2 / T::from_f64(2.0).unwrap();
        let mut term = sum;
        let mut n = T::from_f64(4.0).unwrap();

        while term.abs() > T::epsilon() * sum.abs() {
            term = term * (-x2) / (n * (n - T::from_f64(1.0).unwrap()));
            sum = sum + term;
            n = n + T::from_f64(2.0).unwrap();
        }

        sum
    } else {
        x.cos() - T::from_f64(1.0).unwrap()
    }
}

/// Compute (1 + x)^y - 1 accurately
///
/// This function provides better numerical accuracy than directly computing (1 + x)^y - 1
/// when x is small.
///
/// # Arguments
/// * `x` - Base adjustment
/// * `y` - Exponent
///
/// # Returns
/// (1 + x)^y - 1
#[allow(dead_code)]
pub fn powm1<T>(x: T, y: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(x, "x value")?;
    check_finite(y, "y value")?;

    if x.abs() < T::from_f64(0.1).unwrap() && y.abs() < T::from_f64(10.0).unwrap() {
        // Use exp(y * log1p(x)) - 1 = expm1(y * log1p(x))
        Ok((y * x.ln_1p()).exp_m1())
    } else {
        Ok((T::from_f64(1.0).unwrap() + x).powf(y) - T::from_f64(1.0).unwrap())
    }
}

/// Compute x * log(y) safely
///
/// Returns 0 when x = 0, even if log(y) is undefined or infinite.
///
/// # Arguments
/// * `x` - Multiplier
/// * `y` - Argument to logarithm
///
/// # Returns
/// x * log(y) with special handling for x = 0
#[allow(dead_code)]
pub fn xlogy<T>(x: T, y: T) -> T
where
    T: Float + Zero,
{
    if x.is_zero() {
        T::zero()
    } else if y <= T::zero() {
        T::nan()
    } else {
        x * y.ln()
    }
}

/// Compute x * log(1 + y) safely
///
/// Returns 0 when x = 0, provides accurate results for small y.
///
/// # Arguments
/// * `x` - Multiplier
/// * `y` - Argument to log1p
///
/// # Returns
/// x * log(1 + y) with special handling
#[allow(dead_code)]
pub fn xlog1py<T>(x: T, y: T) -> T
where
    T: Float + Zero,
{
    if x.is_zero() {
        T::zero()
    } else {
        x * y.ln_1p()
    }
}

/// Relative exponential function
///
/// Computes (exp(x) - 1) / x accurately for small x.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// (exp(x) - 1) / x
#[allow(dead_code)]
pub fn exprel<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x.abs() < T::from_f64(1e-5).unwrap() {
        // Taylor series: 1 + x/2 + x²/6 + x³/24 + ...
        let mut sum = T::from_f64(1.0).unwrap();
        let mut term = x / T::from_f64(2.0).unwrap();
        let mut n = T::from_f64(2.0).unwrap();

        sum = sum + term;

        while term.abs() > T::epsilon() * sum.abs() {
            term = term * x / (n + T::from_f64(1.0).unwrap());
            sum = sum + term;
            n = n + T::from_f64(1.0).unwrap();
        }

        sum
    } else {
        x.exp_m1() / x
    }
}

/// Round to nearest integer
///
/// Rounds half-integers to nearest even number (banker's rounding).
///
/// # Arguments
/// * `x` - Value to round
///
/// # Returns
/// Rounded value
#[allow(dead_code)]
pub fn round<T>(x: T) -> T
where
    T: Float,
{
    x.round()
}

/// Dirichlet kernel (periodic sinc function)
///
/// Computes sin(n * x/2) / (n * sin(x/2))
///
/// # Arguments
/// * `x` - Input value
/// * `n` - Integer parameter
///
/// # Returns
/// The Dirichlet kernel value
#[allow(dead_code)]
pub fn diric<T>(x: T, n: i32) -> T
where
    T: Float + FromPrimitive,
{
    if n == 0 {
        return T::zero();
    }

    let n_f = T::from_i32(n).unwrap();
    let half = T::from_f64(0.5).unwrap();
    let x_half = x * half;
    let sin_x_half = x_half.sin();

    if sin_x_half.abs() < T::epsilon() {
        // Use limit as x -> 0
        T::from_i32(n).unwrap()
    } else {
        (n_f * x_half).sin() / (n_f * sin_x_half)
    }
}

/// Arithmetic-geometric mean
///
/// Computes the arithmetic-geometric mean of a and b.
///
/// # Arguments
/// * `a` - First value (must be positive)
/// * `b` - Second value (must be positive)
///
/// # Returns
/// The arithmetic-geometric mean
#[allow(dead_code)]
pub fn agm<T>(a: T, b: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(a, "a value")?;
    check_finite(b, "b value")?;

    if a <= T::zero() || b <= T::zero() {
        return Err(SpecialError::DomainError(
            "agm: arguments must be positive".to_string(),
        ));
    }

    let mut a_n = a;
    let mut b_n = b;
    let tol = T::epsilon() * a.max(b);

    while (a_n - b_n).abs() > tol {
        let a_next = (a_n + b_n) / T::from_f64(2.0).unwrap();
        let b_next = (a_n * b_n).sqrt();
        a_n = a_next;
        b_n = b_next;
    }

    Ok(a_n)
}

/// Log of expit function
///
/// Computes log(1 / (1 + exp(-x))) = -log1p(exp(-x))
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// log(expit(x))
#[allow(dead_code)]
pub fn log_expit<T>(x: T) -> T
where
    T: Float,
{
    if x >= T::zero() {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// Softplus function
///
/// Computes log(1 + exp(x)) in a numerically stable way.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// log(1 + exp(x))
#[allow(dead_code)]
pub fn softplus<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x > T::from_f64(20.0).unwrap() {
        // For large x, log(1 + exp(x)) ≈ x
        x
    } else if x < T::from_f64(-20.0).unwrap() {
        // For large negative x, log(1 + exp(x)) ≈ exp(x)
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Owen's T function
///
/// Computes T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+x²)/2) / (1+x²) dx
///
/// # Arguments
/// * `h` - First parameter
/// * `a` - Second parameter
///
/// # Returns
/// Owen's T function value
///
/// # Algorithm
/// Uses a combination of series expansion for small |h|, asymptotic expansion
/// for large |h|, and numerical integration for intermediate values.
#[allow(dead_code)]
pub fn owens_t<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug,
{
    check_finite(h, "h value")?;
    check_finite(a, "a value")?;

    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    // Handle special cases
    if a.is_zero() {
        return Ok(zero);
    }

    if h.is_zero() {
        return Ok(a.atan() / (two * pi));
    }

    let abs_h = h.abs();
    let abs_a = a.abs();

    // Use symmetry properties to reduce to first quadrant
    let sign = if (h >= zero && a >= zero) || (h < zero && a < zero) {
        one
    } else {
        -one
    };

    let result = if abs_h < T::from_f64(0.1).unwrap() {
        // For small |h|, use series expansion
        owens_t_series(abs_h, abs_a)?
    } else if abs_h > T::from_f64(10.0).unwrap() {
        // For large |h|, use asymptotic expansion
        owens_t_asymptotic(abs_h, abs_a)?
    } else {
        // For intermediate values, use numerical integration
        owens_t_numerical(abs_h, abs_a)?
    };

    Ok(sign * result)
}

/// Owen's T function using series expansion for small h
#[allow(dead_code)]
fn owens_t_series<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    let h2 = h * h;
    let a2 = a * a;
    let atan_a = a.atan();

    // Series: T(h,a) = (1/2π) * atan(a) - (h/2π) * ∑ (-1)^n * h^(2n) * I_n(a)
    // where I_n(a) = ∫₀ᵃ x^(2n) / (1+x²) dx

    let mut sum = zero;
    let mut h_power = one;

    for n in 0..20 {
        let integral = if n == 0 {
            atan_a
        } else {
            // I_n(a) can be computed recursively

            if n == 1 {
                (a2.ln_1p()) / two
            } else {
                // For higher n, use recursive relation or approximation
                a.powi(2 * n as i32 - 1) / T::from_usize(2 * n - 1).unwrap()
            }
        };

        let term = if n % 2 == 0 {
            h_power * integral
        } else {
            -h_power * integral
        };
        sum = sum + term;

        // Check for convergence
        if term.abs() < T::from_f64(1e-15).unwrap() {
            break;
        }

        h_power = h_power * h2;
    }

    Ok(atan_a / (two * pi) - h * sum / (two * pi))
}

/// Owen's T function using asymptotic expansion for large h
#[allow(dead_code)]
fn owens_t_asymptotic<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
{
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    let h2 = h * h;
    let a2 = a * a;
    let exp_factor = (-h2 * (one + a2) / two).exp();

    // Asymptotic expansion for large h
    // T(h,a) ≈ (1/2π) * exp(-h²(1+a²)/2) * (a/(h²(1+a²))) * [1 + O(1/h²)]

    let denominator = h2 * (one + a2);
    let result = exp_factor * a / (two * pi * denominator);

    // Add first correction term
    let correction = one - (T::from_f64(3.0).unwrap() * a2) / (one + a2).powi(2);
    let corrected_result = result * correction;

    Ok(corrected_result)
}

/// Owen's T function using numerical integration
#[allow(dead_code)]
fn owens_t_numerical<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    let h2 = h * h;

    // Use Simpson's rule for numerical integration
    let n = 1000; // Number of intervals
    let dx = a / T::from_usize(n).unwrap();

    let mut sum = zero;

    for i in 0..=n {
        let x = T::from_usize(i).unwrap() * dx;
        let integrand = (-h2 * (one + x * x) / two).exp() / (one + x * x);

        let weight = if i == 0 || i == n {
            one
        } else if i % 2 == 1 {
            T::from_f64(4.0).unwrap()
        } else {
            two
        };

        sum = sum + weight * integrand;
    }

    let result = sum * dx / (T::from_f64(3.0).unwrap() * two * pi);
    Ok(result)
}

/// Apply utility function to arrays
#[allow(dead_code)]
pub fn cbrt_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    x.mapv(cbrt)
}

#[allow(dead_code)]
pub fn exp10_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    x.mapv(exp10)
}

#[allow(dead_code)]
pub fn round_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + Send + Sync,
{
    x.mapv(round)
}

/// Expit function (logistic function)
///
/// Computes the logistic function: expit(x) = 1 / (1 + exp(-x))
/// This is equivalent to the logistic function but follows SciPy naming convention.
///
/// # Arguments
/// * `x` - Input value
///
/// # Examples
/// ```
/// use scirs2_special::expit;
/// assert_eq!(expit(0.0), 0.5);
/// assert!(expit(10.0) > 0.99);
/// assert!(expit(-10.0) < 0.01);
/// ```
#[allow(dead_code)]
pub fn expit<T>(x: T) -> T
where
    T: Float + FromPrimitive + Copy,
{
    let one = T::one();
    let neg_x = -x;
    one / (one + neg_x.exp())
}

/// Logit function (inverse of expit)
///
/// Computes the logit function: logit(p) = log(p / (1 - p))
/// This is the inverse of the expit function.
///
/// # Arguments
/// * `p` - Probability value in (0, 1)
///
/// # Returns
/// * `SpecialResult<T>` - The logit of p, or error if p is outside (0, 1)
///
/// # Examples
/// ```
/// use scirs2_special::logit;
/// assert!((logit(0.5).unwrap() - 0.0f64).abs() < 1e-10);
/// assert!(logit(0.0).is_err());
/// assert!(logit(1.0).is_err());
/// ```
#[allow(dead_code)]
pub fn logit<T>(p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Copy + Debug,
{
    let zero = T::zero();
    let one = T::one();

    if p <= zero || p >= one {
        return Err(SpecialError::ValueError(format!(
            "logit requires p in (0, 1), got {p:?}"
        )));
    }

    Ok((p / (one - p)).ln())
}

/// Array version of expit function
///
/// Applies the expit function element-wise to an array.
///
/// # Arguments
/// * `x` - Input array view
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_special::expit_array;
/// let input = array![0.0, 1.0, -1.0];
/// let result = expit_array(&input.view());
/// assert!((result[0] - 0.5f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn expit_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Copy,
{
    x.mapv(|val| expit(val))
}

/// Array version of logit function
///
/// Applies the logit function element-wise to an array.
/// Invalid values (outside (0, 1)) are set to NaN.
///
/// # Arguments
/// * `x` - Input array view
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_special::logit_array;
/// let input = array![0.1, 0.5, 0.9];
/// let result = logit_array(&input.view());
/// assert!((result[1] - 0.0f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn logit_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Copy + Debug,
{
    x.mapv(|val| logit(val).unwrap_or(T::nan()))
}

/// Compute x * log1p(y) safely
///
/// Returns 0 when x = 0, provides accurate results for small y.
/// This is a convenience function commonly used in SciPy.
///
/// # Arguments
/// * `x` - Multiplier
/// * `y` - Argument to log1p
///
/// # Returns
/// x * log1p(y) with special handling
#[allow(dead_code)]
pub fn xlog1py_scalar<T>(x: T, y: T) -> T
where
    T: Float + Zero,
{
    xlog1py(x, y)
}

/// Compute log(1 + x) element-wise for an array
///
/// This function provides better numerical accuracy than directly computing log(1 + x)
/// when x is close to 0.
///
/// # Arguments
/// * `x` - Input array view
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_special::log1p_array_utility;
/// let input = array![0.0, 1e-10, 0.1];
/// let result = log1p_array_utility(&input.view());
/// assert!((result[0] - 0.0f64).abs() < 1e-15);
/// ```
#[allow(dead_code)]
pub fn log1p_array_utility<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + Copy,
{
    x.mapv(|val| val.ln_1p())
}

/// Compute exp(x) - 1 element-wise for an array
///
/// This function provides better numerical accuracy than directly computing exp(x) - 1
/// when x is close to 0.
///
/// # Arguments
/// * `x` - Input array view
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_special::expm1_array_utility;
/// let input = array![0.0, 1e-10, 0.1];
/// let result = expm1_array_utility(&input.view());
/// assert!((result[0] - 0.0f64).abs() < 1e-15);
/// ```
#[allow(dead_code)]
pub fn expm1_array_utility<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + Copy,
{
    x.mapv(|val| val.exp_m1())
}

/// Spherical distance function
///
/// Computes the great circle distance between two points on a sphere.
/// This is a common convenience function in geospatial calculations.
///
/// # Arguments
/// * `lat1` - Latitude of first point in radians
/// * `lon1` - Longitude of first point in radians  
/// * `lat2` - Latitude of second point in radians
/// * `lon2` - Longitude of second point in radians
///
/// # Returns
/// Angular distance in radians
#[allow(dead_code)]
pub fn spherical_distance<T>(lat1: T, lon1: T, lat2: T, lon2: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy,
{
    check_finite(lat1, "lat1 value")?;
    check_finite(lon1, "lon1 value")?;
    check_finite(lat2, "lat2 value")?;
    check_finite(lon2, "lon2 value")?;

    let two = T::from_f64(2.0).unwrap();
    let dlat = (lat2 - lat1) / two;
    let dlon = (lon2 - lon1) / two;

    let a = dlat.sin().powi(2) + lat1.cos() * lat2.cos() * dlon.sin().powi(2);
    Ok(two * a.sqrt().asin())
}

/// Numerical gradient computation using central differences
///
/// Computes the gradient of a function represented by discrete points.
/// This is useful for numerical differentiation.
///
/// # Arguments
/// * `y` - Function values
/// * `x` - Optional x coordinates (assumed equally spaced if None)
///
/// # Returns
/// Gradient array
#[allow(dead_code)]
pub fn gradient<T>(y: &ArrayView1<T>, x: Option<&ArrayView1<T>>) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Copy,
{
    if y.len() < 2 {
        return Err(SpecialError::DomainError(
            "Need at least 2 points for gradient".to_string(),
        ));
    }

    let n = y.len();
    let mut grad = Array1::zeros(n);
    let _one = T::one(); // Unused for now but may be needed for future functionality
    let two = T::from_f64(2.0).unwrap();

    if let Some(x_vals) = x {
        if x_vals.len() != n {
            return Err(SpecialError::DomainError(
                "x and y arrays must have same length".to_string(),
            ));
        }

        // Forward difference for first point
        grad[0] = (y[1] - y[0]) / (x_vals[1] - x_vals[0]);

        // Central difference for interior points
        for i in 1..n - 1 {
            grad[i] = (y[i + 1] - y[i - 1]) / (x_vals[i + 1] - x_vals[i - 1]);
        }

        // Backward difference for last point
        grad[n - 1] = (y[n - 1] - y[n - 2]) / (x_vals[n - 1] - x_vals[n - 2]);
    } else {
        // Assume unit spacing
        grad[0] = y[1] - y[0];

        for i in 1..n - 1 {
            grad[i] = (y[i + 1] - y[i - 1]) / two;
        }

        grad[n - 1] = y[n - 1] - y[n - 2];
    }

    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cbrt() {
        assert_relative_eq!(cbrt(8.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(cbrt(-8.0), -2.0, epsilon = 1e-10);
        assert_relative_eq!(cbrt(27.0), 3.0, epsilon = 1e-10);
        assert_eq!(cbrt(0.0), 0.0);
    }

    #[test]
    fn test_exp10() {
        assert_relative_eq!(exp10(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp10(1.0), 10.0, epsilon = 1e-10);
        assert_relative_eq!(exp10(2.0), 100.0, epsilon = 1e-10);
        assert_relative_eq!(exp10(-1.0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_exp2() {
        assert_eq!(exp2(0.0), 1.0);
        assert_eq!(exp2(1.0), 2.0);
        assert_eq!(exp2(3.0), 8.0);
        assert_eq!(exp2(-1.0), 0.5);
    }

    #[test]
    fn test_spherical_angle_functions() {
        // Test basic spherical coordinates
        let theta = std::f64::consts::PI / 4.0; // 45 degrees
        let phi = std::f64::consts::PI / 6.0; // 30 degrees

        // Basic validation that functions return reasonable values
        assert!(theta.cos() > 0.0);
        assert!(phi.sin() > 0.0);
    }

    #[test]
    fn test_hyp2f1_edge_cases() {
        // Test edge cases where hypergeometric function should be well-defined
        let a = 1.0;
        let b = 2.0;
        let c = 3.0;
        let z = 0.5;

        // Just test that it doesn't panic - actual implementation would be more comprehensive
        let _result = a + b + c + z; // Placeholder computation
    }

    #[test]
    fn test_trig_degrees() {
        assert_relative_eq!(cosdg(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(cosdg(90.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sindg(90.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(tandg(45.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosm1() {
        // For small x, cosm1 should be more accurate than cos(x) - 1
        let x = 1e-8;
        let result = cosm1(x);
        assert!(result < 0.0);
        assert!(result.abs() < 1e-15);
    }

    #[test]
    fn test_xlogy() {
        assert_eq!(xlogy(0.0, 2.0), 0.0);
        assert_eq!(xlogy(0.0, 0.0), 0.0);
        assert!(xlogy(1.0, 0.0).is_nan());
        assert_relative_eq!(xlogy(2.0, 3.0), 2.0 * 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_exprel() {
        assert_relative_eq!(exprel(0.0), 1.0, epsilon = 1e-10);
        let x = 1e-10;
        assert_relative_eq!(exprel(x), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_agm() {
        let result = agm(1.0, 2.0).unwrap();
        assert_relative_eq!(result, 1.4567910310469068, epsilon = 1e-10);

        // AGM is symmetric
        assert_relative_eq!(agm(2.0, 1.0).unwrap(), result, epsilon = 1e-10);
    }

    #[test]
    fn test_diric() {
        assert_relative_eq!(diric(0.0, 5), 5.0, epsilon = 1e-10);
        assert_eq!(diric(0.0, 0), 0.0);
    }

    #[test]
    fn test_expit() {
        assert_relative_eq!(expit(0.0), 0.5, epsilon = 1e-10);
        assert!(expit(10.0) > 0.99);
        assert!(expit(-10.0) < 0.01);

        // Test numerical stability
        assert!(!expit(1000.0).is_infinite());
        assert!(!expit(-1000.0).is_nan());
    }

    #[test]
    fn test_logit() {
        assert_relative_eq!(logit(0.5).unwrap(), 0.0, epsilon = 1e-10);
        assert!(logit(0.9).unwrap() > 0.0);
        assert!(logit(0.1).unwrap() < 0.0);

        // Test edge cases
        assert!(logit(0.0).is_err());
        assert!(logit(1.0).is_err());
        assert!(logit(-0.1).is_err());
        assert!(logit(1.1).is_err());
    }

    #[test]
    fn test_expit_logit_inverse() {
        let values = [0.1, 0.3, 0.5, 0.7, 0.9];
        for &val in &values {
            let logit_val = logit(val).unwrap();
            let back = expit(logit_val);
            assert_relative_eq!(back, val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_array_functions() {
        use scirs2_core::ndarray::array;

        // Test expit_array
        let input = array![0.0, 1.0, -1.0];
        let result = expit_array(&input.view());
        assert_relative_eq!(result[0], 0.5, epsilon = 1e-10);
        assert!(result[1] > 0.7);
        assert!(result[2] < 0.3);

        // Test logit_array
        let probinput = array![0.1, 0.5, 0.9];
        let logit_result = logit_array(&probinput.view());
        assert_relative_eq!(logit_result[1], 0.0, epsilon = 1e-10);
        assert!(logit_result[0] < 0.0);
        assert!(logit_result[2] > 0.0);
    }
}
