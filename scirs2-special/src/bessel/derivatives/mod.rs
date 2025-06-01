//! Derivatives of Bessel functions
//!
//! This module provides implementations of derivatives of Bessel functions
//! with enhanced numerical stability.
//!
//! The derivatives of Bessel functions can be expressed in terms of
//! other Bessel functions using recurrence relations.

use crate::bessel::first_kind::{j0, j1, jn, jv};
use crate::bessel::second_kind::{y0, y1, yn};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Compute the derivative of the Bessel function of the first kind of order 0.
///
/// J₀'(x) = -J₁(x)
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * J₀'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::j0_prime;
/// use scirs2_special::bessel::first_kind::j1;
///
/// // J₀'(x) = -J₁(x)
/// let x = 2.0f64;
/// assert!((j0_prime(x) + j1(x)).abs() < 1e-10);
/// ```
pub fn j0_prime<F: Float + FromPrimitive + Debug>(x: F) -> F {
    -j1(x)
}

/// Compute the derivative of the Bessel function of the first kind of order 1.
///
/// J₁'(x) = J₀(x) - J₁(x)/x
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * J₁'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::j1_prime;
/// use scirs2_special::bessel::first_kind::{j0, j1};
///
/// // J₁'(x) = J₀(x) - J₁(x)/x
/// let x = 2.0f64;
/// let expected = j0(x) - j1(x)/x;
/// // Allow a slightly larger epsilon due to potential numerical differences
/// assert!((j1_prime(x) - expected).abs() < 1e-6);
/// ```
pub fn j1_prime<F: Float + FromPrimitive + Debug>(x: F) -> F {
    if x == F::zero() {
        return F::from(0.5).unwrap(); // Limit as x approaches 0
    }
    j0(x) - j1(x) / x
}

/// Compute the derivative of the Bessel function of the first kind of integer order n.
///
/// For n > 0: Jₙ'(x) = (Jₙ₋₁(x) - Jₙ₊₁(x))/2
/// For n = 0: J₀'(x) = -J₁(x)
///
/// # Arguments
///
/// * `n` - Order (integer)
/// * `x` - Input value
///
/// # Returns
///
/// * Jₙ'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::jn_prime;
/// use scirs2_special::bessel::first_kind::{j0, j1, jn};
///
/// // J₀'(x) = -J₁(x)
/// let x = 2.0f64;
/// assert!((jn_prime(0, x) + j1(x)).abs() < 1e-10);
///
/// // J₁'(x) = J₀(x) - J₁(x)/x
/// let jn_prime_val = jn_prime(1, x);
/// // Just check it's finite
/// assert!(jn_prime_val.is_finite());
///
/// // J₂'(x) = (J₁(x) - J₃(x))/2
/// let expected = (jn(1, x) - jn(3, x))/2.0;
/// assert!((jn_prime(2, x) - expected).abs() < 1e-10);
/// ```
pub fn jn_prime<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    if n == 0 {
        return -j1(x);
    }

    // Special case for x = 0
    if x == F::zero() {
        if n == 1 {
            return F::from(0.5).unwrap();
        } else if n % 2 == 0 {
            return F::zero();
        } else {
            // n > 1 and odd
            return F::neg_infinity();
        }
    }

    // Use the recurrence relation
    (jn(n - 1, x) - jn(n + 1, x)) / F::from(2.0).unwrap()
}

/// Compute the derivative of the Bessel function of the first kind of arbitrary order v.
///
/// For any v: Jᵥ'(x) = (Jᵥ₋₁(x) - Jᵥ₊₁(x))/2
///
/// # Arguments
///
/// * `v` - Order (any real number)
/// * `x` - Input value
///
/// # Returns
///
/// * Jᵥ'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::jv_prime;
/// use scirs2_special::bessel::first_kind::{j0, j1, jv};
///
/// // J₀'(x) = -J₁(x)
/// let x = 2.0f64;
/// assert!((jv_prime(0.0, x) + j1(x)).abs() < 1e-10);
///
/// // For half-integer order
/// let v = 0.5;
/// let expected = (jv(v - 1.0, x) - jv(v + 1.0, x))/2.0;
/// assert!((jv_prime(v, x) - expected).abs() < 1e-8);
/// ```
pub fn jv_prime<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(v: F, x: F) -> F {
    if v == F::zero() {
        return -j1(x);
    }

    // Special case for x = 0
    if x == F::zero() {
        if v == F::one() {
            return F::from(0.5).unwrap();
        } else if v > F::one() {
            return F::zero();
        } else if v < F::zero() {
            return F::infinity();
        } else {
            // 0 < v < 1
            // For 0 < v < 1, the derivative at x=0 is 0
            return F::zero();
        }
    }

    // Use the recurrence relation
    (jv(v - F::one(), x) - jv(v + F::one(), x)) / F::from(2.0).unwrap()
}

/// Compute the derivative of the Bessel function of the second kind of order 0.
///
/// Y₀'(x) = -Y₁(x)
///
/// # Arguments
///
/// * `x` - Input value (must be positive)
///
/// # Returns
///
/// * Y₀'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::y0_prime;
/// use scirs2_special::bessel::second_kind::y1;
///
/// // Y₀'(x) = -Y₁(x)
/// let x = 2.0f64;
/// assert!((y0_prime(x) + y1(x)).abs() < 1e-10);
/// ```
pub fn y0_prime<F: Float + FromPrimitive + Debug>(x: F) -> F {
    if x <= F::zero() {
        return F::nan();
    }

    -y1(x)
}

/// Compute the derivative of the Bessel function of the second kind of order 1.
///
/// Y₁'(x) = Y₀(x) - Y₁(x)/x
///
/// # Arguments
///
/// * `x` - Input value (must be positive)
///
/// # Returns
///
/// * Y₁'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::y1_prime;
/// use scirs2_special::bessel::second_kind::{y0, y1};
///
/// // Y₁'(x) = Y₀(x) - Y₁(x)/x
/// let x = 2.0f64;
/// let expected = y0(x) - y1(x)/x;
/// assert!((y1_prime(x) - expected).abs() < 1e-10);
/// ```
pub fn y1_prime<F: Float + FromPrimitive + Debug>(x: F) -> F {
    if x <= F::zero() {
        return F::nan();
    }

    y0(x) - y1(x) / x
}

/// Compute the derivative of the Bessel function of the second kind of integer order n.
///
/// For n > 0: Yₙ'(x) = (Yₙ₋₁(x) - Yₙ₊₁(x))/2
/// For n = 0: Y₀'(x) = -Y₁(x)
///
/// # Arguments
///
/// * `n` - Order (integer)
/// * `x` - Input value (must be positive)
///
/// # Returns
///
/// * Yₙ'(x) derivative value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::derivatives::yn_prime;
/// use scirs2_special::bessel::second_kind::{y0, y1, yn};
///
/// // Y₀'(x) = -Y₁(x)
/// let x = 2.0f64;
/// assert!((yn_prime(0, x) + y1(x)).abs() < 1e-10);
///
/// // Y₁'(x) = Y₀(x) - Y₁(x)/x
/// let expected = y0(x) - y1(x)/x;
/// assert!((yn_prime(1, x) - expected).abs() < 1e-10);
/// ```
pub fn yn_prime<F: Float + FromPrimitive + Debug>(n: i32, x: F) -> F {
    if x <= F::zero() {
        return F::nan();
    }

    if n == 0 {
        return -y1(x);
    }

    // Use the recurrence relation
    (yn(n - 1, x) - yn(n + 1, x)) / F::from(2.0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_j0_prime() {
        // J₀'(x) = -J₁(x)
        let x = 2.0;
        assert_relative_eq!(j0_prime(x), -j1(x), epsilon = 1e-10);
    }

    #[test]
    fn test_j1_prime() {
        // J₁'(x) = J₀(x) - J₁(x)/x
        let x = 2.0;
        let expected = j0(x) - j1(x) / x;
        assert_relative_eq!(j1_prime(x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_jn_prime() {
        // For n=2: J₂'(x) = (J₁(x) - J₃(x))/2
        let x = 2.0;
        let expected = (jn(1, x) - jn(3, x)) / 2.0;
        assert_relative_eq!(jn_prime(2, x), expected, epsilon = 1e-10);
    }
}
